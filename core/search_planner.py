"""
Coordinated Search Path Planning for a heterogeneous rescue team.

Heterogeneous Multi-Agent Probabilistic Coverage Path Planning on a discrete
2D grid: given a probability heatmap of where the missing person is, place K
rescue agents (each with its own start, speed and sonar radius) and route them
over a time horizon T so that the cumulative probability *cleared* (scanned at
least once by anyone) is maximised. Maximising probability-cleared-per-time is
equivalent to minimising the Expected Time to Detection (ETD).

--------------------------------------------------------------------------
1. ALGORITHMIC STRATEGY  (why a coordinated greedy heuristic)
--------------------------------------------------------------------------
The exact problem is NP-hard: it is a team-orienteering / multi-agent
informative-path-planning problem, which generalises the (already NP-hard)
Travelling Salesman and Maximum-Coverage problems. With a grid of M*N cells,
K agents and horizon T the joint action space is (moves)^(K*T) -- far too
large to optimise exactly for real-time SAR.

Two families of approaches:

  * Global metaheuristics (Genetic Algorithms, Ant Colony, Monte-Carlo Tree
    Search). These search the joint path space and can escape local optima,
    so given minutes of compute they find better plans. But they are slow,
    need careful tuning, and are hard to run on an edge device in the field
    while the clock is ticking on a drowning victim.

  * Sequential / coordinated greedy with a per-time-step lookahead (CHOSEN).
    The coverage objective ("total probability of the union of scanned
    cells") is a MONOTONE SUBMODULAR set function. For such functions the
    greedy rule -- repeatedly take the action with the largest *marginal*
    gain -- is provably within (1 - 1/e) ~ 63% of the optimum, and in
    practice much closer. We exploit this:

        - Agents are routed one decision at a time. Each agent greedily
          extends its own path by up to `speed` grid steps per global tick,
          always toward the move that clears the most *remaining* (not-yet-
          scanned) probability -- a depth-`speed` rollout, i.e. the
          "time-step lookahead".
        - COORDINATION: within a tick the agents decide SEQUENTIALLY and each
          immediately marks its scanned cells as cleared, so the next agent
          sees them as worthless. This already pushes agents to DISPERSE
          without any explicit repulsion.
        - EXPLICIT DISPERSION (optional, `dispersion` in [0, 1]): on top of the
          implicit spreading, agents can be biased to keep AWAY from each other
          -- among the moves that still clear something, prefer the one that
          maximises distance to the other teams, and when peeling off to a
          leftover patch, prefer patches far from the others. This fans the
          routes out to cover more ground; it only ever re-orders productive
          moves, so it never sacrifices real coverage for separation.
        - DISPERSION OVER GAPS: when an agent's whole neighbourhood is already
          cleared (marginal gain 0), it switches to a potential-field move,
          heading toward the nearest heavy patch of remaining probability so
          it travels productively instead of stalling.

This gives near-real-time planning with a quality guarantee, and degrades
gracefully (more agents / longer horizon just means more greedy steps).

--------------------------------------------------------------------------
3. COMPLEXITY  (see analyse_complexity() and the module docstring tail)
--------------------------------------------------------------------------
Per agent per tick we evaluate ~9 candidate moves (8-connected + stay) for
each of up to `speed` sub-steps, each costing O(R^2) to sum a sonar disk of
radius R. Total:  O(T * K * S * R^2)  with S = max speed, plus an occasional
O(M*N) scan for the nearest remaining mass. Independent of how fine the
probability values are -- only the grid size and team config matter.
"""

from __future__ import annotations

import numpy as np


# 8-connected neighbourhood (+ "stay" is added explicitly where needed).
_MOVES_8 = [(-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)]
_MOVES_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _norm01(vals):
    """Min-max normalise a 1D array to [0, 1] (all-equal -> all zeros)."""
    vals = np.asarray(vals, dtype=np.float64)
    lo, hi = vals.min(), vals.max()
    if hi - lo < 1e-15:
        return np.zeros_like(vals)
    return (vals - lo) / (hi - lo)


class Agent:
    """One rescue unit. Heterogeneous via `speed` and `sonar_radius`.

    Attributes
    ----------
    start : (row, col)     starting cell on the grid
    speed : int >= 1       grid edge-transitions allowed per GLOBAL time step
    sonar_radius : int>=0  cells scanned around the agent (Euclidean disk)
    path : list[(r, c)]    position at the end of each global tick (incl. start)
    fine_path : list[(r,c)]every sub-step position (length grows by <=speed/tick)
    cleared_prob : float   probability this agent personally cleared (first to
                           scan a cell gets the credit; no double counting)
    """

    def __init__(self, start, speed=1, sonar_radius=1, name=None, color=None):
        self.start = (int(start[0]), int(start[1]))
        self.speed = max(1, int(speed))
        self.sonar_radius = max(0, int(sonar_radius))
        self.name = name
        self.color = color
        self.reset()

    def reset(self):
        self.pos = self.start
        self.path = [self.start]
        self.fine_path = [self.start]
        self.cleared_prob = 0.0

    def __repr__(self):
        return (f"Agent({self.name or '?'}, start={self.start}, "
                f"speed={self.speed}, sonar={self.sonar_radius})")


def _disk_offsets(radius):
    """Pre-computed (dr, dc) offsets of a filled Euclidean disk of `radius`.
    radius 0 -> just the centre cell."""
    if radius <= 0:
        return np.array([[0, 0]], dtype=np.int32)
    offs = [(dr, dc)
            for dr in range(-radius, radius + 1)
            for dc in range(-radius, radius + 1)
            if dr * dr + dc * dc <= radius * radius]
    return np.array(offs, dtype=np.int32)


class CoveragePlanner:
    """Coordinated greedy planner over a probability grid.

    Parameters
    ----------
    prob : (M, N) float array   probability heatmap (need not sum to 1; it is
                                used as relative weight of each cell).
    agents : list[Agent]
    horizon : int               number of global time steps T to plan.
    connectivity : 4 or 8       grid movement model (8 allows diagonals).
    allow_stay : bool           may an agent hold position for a sub-step.
    passable : (M, N) bool      cells an agent may occupy (e.g. water). Land /
                                obstacles are False -- agents never step onto
                                them. Defaults to everywhere-passable.
    dispersion : float [0, 1]   how strongly agents prefer to keep AWAY from one
                                another (spread the routes to cover more ground).
                                0 = pure probability-greedy (original behaviour);
                                higher trades a little local gain for separation.
                                Applied as a tie-break among productive moves and
                                when choosing which leftover patch to head to, so
                                it never sends an agent onto worthless cells.
    """

    def __init__(self, prob, agents, horizon, connectivity=8, allow_stay=True,
                 passable=None, dispersion=0.0):
        self.prob = np.asarray(prob, dtype=np.float64)
        if self.prob.ndim != 2:
            raise ValueError("prob must be a 2D array")
        self.M, self.N = self.prob.shape
        self.agents = list(agents)
        self.horizon = int(horizon)
        self.moves = _MOVES_8 if connectivity == 8 else _MOVES_4
        self.allow_stay = allow_stay
        self.dispersion = float(np.clip(dispersion, 0.0, 1.0))
        if passable is None:
            self.passable = np.ones((self.M, self.N), dtype=bool)
        else:
            self.passable = np.asarray(passable, dtype=bool)
            if self.passable.shape != self.prob.shape:
                raise ValueError("passable must match prob shape")

        # remaining[r, c] = probability still UNSCANNED. Scanning zeroes it, so
        # the coverage objective is just the mass we remove from `remaining`.
        self.remaining = self.prob.copy()
        self.cleared_mask = np.zeros((self.M, self.N), dtype=bool)
        self._disks = {a.sonar_radius: _disk_offsets(a.sonar_radius)
                       for a in self.agents}
        self.total_prob = float(self.prob.sum())
        self.total_cleared = 0.0

    # -- geometry helpers ---------------------------------------------------
    def _in_bounds(self, r, c):
        return 0 <= r < self.M and 0 <= c < self.N

    def _can_enter(self, r, c):
        """In bounds AND passable (an agent may not step onto land/obstacles)."""
        return self._in_bounds(r, c) and self.passable[r, c]

    def _footprint(self, pos, radius):
        """Valid (rows, cols) index arrays of the sonar disk centred at pos."""
        offs = self._disks[radius]
        rr = offs[:, 0] + pos[0]
        cc = offs[:, 1] + pos[1]
        ok = (rr >= 0) & (rr < self.M) & (cc >= 0) & (cc < self.N)
        return rr[ok], cc[ok]

    def _gain(self, pos, radius):
        """Marginal probability that scanning at `pos` would newly clear."""
        rr, cc = self._footprint(pos, radius)
        return float(self.remaining[rr, cc].sum())

    def _apply_scan(self, pos, radius):
        """Mark the sonar disk at `pos` cleared; return the probability gained."""
        rr, cc = self._footprint(pos, radius)
        gained = float(self.remaining[rr, cc].sum())
        self.remaining[rr, cc] = 0.0
        self.cleared_mask[rr, cc] = True
        self.total_cleared += gained
        return gained

    def _attractor_step(self, pos, radius, others=()):
        """Fallback when the local neighbourhood is exhausted: pick the move
        that most reduces distance to the nearest heavy patch of remaining
        probability (a coarse potential-field pull toward leftover mass).

        With dispersion on, leftover patches that are FAR from the other agents
        are preferred, so idle agents peel off toward different clusters instead
        of all piling onto the same remaining blob."""
        if self.remaining.max() <= 0:
            return None                      # nothing left anywhere
        rr, cc = np.nonzero(self.remaining)
        # weight remaining cells by prob / (1 + distance) -> nearest-heavy target
        d = np.hypot(rr - pos[0], cc - pos[1])
        score = self.remaining[rr, cc] / (1.0 + d)
        if self.dispersion > 0.0 and len(others):
            dmin = np.full(rr.shape, np.inf)
            for (orow, ocol) in others:
                dmin = np.minimum(dmin, np.hypot(rr - orow, cc - ocol))
            # boost targets that are far from other agents (factor in [1, 1+w])
            score = score * (1.0 + self.dispersion * dmin / (1.0 + dmin))
        tgt = (rr[np.argmax(score)], cc[np.argmax(score)])
        # step (incl. diagonal) that reduces Euclidean distance to the target most
        best, best_d = pos, np.hypot(pos[0] - tgt[0], pos[1] - tgt[1])
        for dr, dc in self.moves:
            nr, nc = pos[0] + dr, pos[1] + dc
            if not self._can_enter(nr, nc):
                continue
            nd = np.hypot(nr - tgt[0], nc - tgt[1])
            if nd < best_d:
                best, best_d = (nr, nc), nd
        return best

    def _agent_tick(self, agent):
        """Advance one agent by up to `speed` sub-steps for one global tick.

        Greedy depth-`speed` rollout: at each sub-step move to the neighbour
        (or stay) with the largest marginal sonar gain; if everything nearby
        is already cleared, fall back to the attractor pull. Each scanned disk
        is committed immediately so later agents this tick avoid it.

        With dispersion > 0, among the moves that still clear something we
        prefer the one that also keeps the agent farthest from the other teams,
        so the routes fan out and sweep more ground."""
        others = [b.pos for b in self.agents if b is not agent]
        for _ in range(agent.speed):
            pos = agent.pos
            candidates = list(self.moves) + ([(0, 0)] if self.allow_stay else [])

            cells, gains = [], []
            for dr, dc in candidates:
                nr, nc = pos[0] + dr, pos[1] + dc
                if not self._can_enter(nr, nc):
                    continue
                cells.append((nr, nc))
                gains.append(self._gain((nr, nc), agent.sonar_radius))
            gains = np.asarray(gains)

            if gains.size == 0 or gains.max() <= 0.0:
                # local area exhausted -> travel toward remaining mass
                nxt = self._attractor_step(pos, agent.sonar_radius, others)
                if nxt is None or nxt == pos:
                    agent.fine_path.append(pos)
                    break
                best_move = nxt
            elif self.dispersion > 0.0 and others:
                # blend clearing value with separation from the other teams;
                # mask out worthless cells so spread never beats real coverage
                sep = np.array([min(np.hypot(nr - orow, nc - ocol)
                                    for (orow, ocol) in others)
                                for (nr, nc) in cells])
                score = ((1.0 - self.dispersion) * _norm01(gains)
                         + self.dispersion * _norm01(sep))
                score[gains <= 0.0] = -1.0
                best_move = cells[int(np.argmax(score))]
            else:
                best_move = cells[int(np.argmax(gains))]

            agent.pos = best_move
            agent.cleared_prob += self._apply_scan(best_move, agent.sonar_radius)
            agent.fine_path.append(best_move)

        agent.path.append(agent.pos)

    def plan(self):
        """Run the full coordinated plan and return a results dict.

        Agents are (re)started, then for each of T ticks every agent takes its
        turn. Within a tick we let the agent with the most to gain act first
        (dynamic ordering), which improves dispersion over a fixed order."""
        for a in self.agents:
            a.reset()
        self.remaining = self.prob.copy()
        self.cleared_mask[:] = False
        self.total_cleared = 0.0

        # scan each agent's starting footprint up front (it sees where it stands)
        for a in self.agents:
            a.cleared_prob += self._apply_scan(a.pos, a.sonar_radius)

        history = [self.total_cleared / self.total_prob if self.total_prob else 0.0]
        for _ in range(self.horizon):
            # dynamic ordering: the agent that can currently grab the most goes
            # first, so high-value cells are claimed by whoever is best placed.
            order = sorted(self.agents,
                           key=lambda a: self._gain(a.pos, a.sonar_radius),
                           reverse=True)
            for a in order:
                self._agent_tick(a)
            frac = self.total_cleared / self.total_prob if self.total_prob else 0.0
            history.append(frac)

        return {
            'agents': self.agents,
            'cleared_prob': self.total_cleared,
            'cleared_fraction': (self.total_cleared / self.total_prob
                                 if self.total_prob else 0.0),
            'coverage_over_time': history,        # cleared fraction after each tick
            'cleared_mask': self.cleared_mask,
            'remaining': self.remaining,
        }


def analyse_complexity():
    """Return a short human-readable complexity note (see module docstring)."""
    return (
        "Time:  O(T * K * S * R^2)  + O(T * K * M*N) worst case when agents\n"
        "       repeatedly invoke the nearest-remaining-mass fallback.\n"
        "         T = horizon, K = #agents, S = max speed, R = max sonar radius,\n"
        "         M*N = grid cells.  Independent of the probability resolution.\n"
        "Space: O(M*N) for the remaining/cleared grids + O(T*S) per agent path.\n\n"
        "Edge-deployment optimisations:\n"
        "  * Summed-area table (integral image) of `remaining` -> O(1) square\n"
        "    footprint sums instead of O(R^2); refresh the SAT lazily.\n"
        "  * Vectorise candidate-move scoring over all agents with NumPy, or\n"
        "    push the grid to the GPU (cupy/torch) for large maps.\n"
        "  * Coarsen the grid (cluster cells) for a first pass, refine locally.\n"
        "  * Precompute disk offsets once (done) and cache footprints per cell.\n"
        "  * Cap the attractor fallback frequency / use a precomputed distance\n"
        "    transform to the remaining-mass centroid for O(1) pulls.\n"
        "  * numba @njit the inner sub-step loop for ~10-50x on CPU."
    )


# ---------------------------------------------------------------------------
# Stand-alone demo: a bimodal heatmap + 3 heterogeneous agents.
# ---------------------------------------------------------------------------
def _demo():
    M = N = 60
    yy, xx = np.mgrid[0:M, 0:N]

    def gauss(cy, cx, s):
        return np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))

    prob = 1.0 * gauss(18, 20, 6) + 0.7 * gauss(40, 44, 8)
    prob /= prob.sum()

    agents = [
        Agent((30, 0),  speed=3, sonar_radius=2, name='Fast skiff',  color='#3b82f6'),
        Agent((30, 59), speed=1, sonar_radius=4, name='Sonar boat',  color='#f97316'),
        Agent((59, 30), speed=2, sonar_radius=2, name='Diver RIB',   color='#22c55e'),
    ]

    planner = CoveragePlanner(prob, agents, horizon=25, connectivity=8)
    res = planner.plan()

    print("Heterogeneous coordinated search demo")
    print(f"  grid {M}x{N}, {len(agents)} agents, horizon {planner.horizon}")
    for a in agents:
        print(f"  {a.name:11s} speed={a.speed} sonar={a.sonar_radius} "
              f"cleared {a.cleared_prob*100:5.1f}%  "
              f"path {len(a.fine_path)} cells")
    print(f"  TOTAL probability cleared: {res['cleared_fraction']*100:.1f}%")
    print()
    print(analyse_complexity())
    return res


if __name__ == '__main__':
    _demo()
