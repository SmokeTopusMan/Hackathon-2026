"""
Coordinated Search Path Planning for a heterogeneous rescue team.

Heterogeneous Multi-Agent Probabilistic Coverage Path Planning on a fine 2D
grid: given a probability heatmap of where the missing person is, place K
rescue craft (each with its own start, real-world SPEED and sonar/lookout
radius) and route them so the cumulative probability *cleared* (scanned at
least once by anyone) is maximised. Maximising probability-cleared-per-time is
equivalent to minimising the Expected Time to Detection (ETD).

WHAT CHANGED IN THIS VERSION
----------------------------
  * REAL UNITS.  The grid has a physical cell size (`cell_m`, e.g. 3 m). Each
    agent carries a real speed in m/s and a sonar radius in metres; the planner
    converts those to cells-per-tick and a sonar disk. So a fast jet-ski covers
    more ground per minute than a slow boat, and the route length / ETA come out
    in metres and minutes.
  * CONVERGE, DON'T COUNT STEPS.  Planning runs until it CONVERGES -- the target
    coverage is reached, or coverage stops improving (saturated), or a maximum
    mission time is hit -- instead of a fixed number of ticks. This stops the
    routes from wandering on forever once the area is effectively cleared.
  * REFERENCE GRID.  `make_reference_grid()` builds a coarse labelled grid
    (rows A, B, C ... from the north, columns 1, 2, 3 ... from the west) that
    auto-fits the search area, giving the teams a shared "B-4" map language.

--------------------------------------------------------------------------
ALGORITHMIC STRATEGY  (why a coordinated greedy heuristic)
--------------------------------------------------------------------------
The exact problem is NP-hard (multi-agent informative-path-planning, which
generalises TSP and Maximum-Coverage). The coverage objective ("total
probability of the union of scanned cells") is MONOTONE SUBMODULAR, so the
greedy rule -- repeatedly take the action with the largest marginal gain -- is
within (1 - 1/e) ~ 63% of optimal and usually much closer. We route agents one
decision at a time:

  - Each tick an agent greedily extends its path by `cells_per_tick`
    (= speed / cell size) sub-steps, each toward the move that clears the most
    REMAINING probability.
  - COORDINATION: within a tick agents decide SEQUENTIALLY and commit their
    scanned cells immediately, so others see them as worthless -> they spread
    out with no explicit repulsion.
  - EXPLICIT DISPERSION (optional, `dispersion` in [0, 1]): among moves that
    still clear something, prefer the one farthest from the other teams; when
    peeling off to a leftover patch, prefer patches far from the others. It only
    re-orders PRODUCTIVE moves, so it never trades real coverage for spread.
  - GAP TRAVEL: when the whole neighbourhood is already cleared, switch to a
    potential-field move toward the nearest heavy patch of remaining mass.

--------------------------------------------------------------------------
COMPLEXITY
--------------------------------------------------------------------------
Per tick per agent we evaluate ~9 candidate moves for each of `cells_per_tick`
sub-steps, each O(R^2) to sum a sonar disk of radius R cells. Total to converge
is O(ticks * K * cells_per_tick * R^2). On a fine grid keep R (sonar_m / cell_m)
modest, or add a summed-area table for O(1) footprint sums (see analyse_complexity).
"""

from __future__ import annotations

import math

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


class Agent:
    """One rescue craft, in REAL-WORLD units.

    Parameters
    ----------
    start : (row, col)        starting cell on the planning grid
    speed_mps : float         cruising/search speed in metres per second
                              (boat ~6-8 m/s, jet-ski ~12-15 m/s)
    sonar_radius_m : float    detection radius in metres (sonar / lookout swath)
    name, color : optional    labels for plotting

    The planner fills in, from its `cell_m`:
        _sonar_cells     sonar radius in grid cells
        _cells_per_tick  how many cells this craft advances each planning tick
    and tracks, after planning:
        path / fine_path tick-resolution and sub-step routes (row, col)
        cleared_prob     probability this craft personally cleared
        distance_m       length of its route in metres
        time_s           distance_m / speed_mps (its own travel time)
    """

    def __init__(self, start, speed_mps=7.0, sonar_radius_m=100.0,
                 name=None, color=None):
        self.start = (int(start[0]), int(start[1]))
        self.speed_mps = float(max(0.1, speed_mps))
        self.sonar_radius_m = float(max(0.0, sonar_radius_m))
        self.name = name
        self.color = color
        # filled in by the planner once the cell size is known
        self._sonar_cells = 0
        self._cells_per_tick = 1
        self.reset()

    def reset(self):
        self.pos = self.start
        self.path = [self.start]
        self.fine_path = [self.start]
        self.cleared_prob = 0.0
        self.distance_m = 0.0
        self.time_s = 0.0

    def __repr__(self):
        return (f"Agent({self.name or '?'}, start={self.start}, "
                f"speed={self.speed_mps:.1f} m/s, sonar={self.sonar_radius_m:.0f} m)")


class CoveragePlanner:
    """Coordinated greedy coverage planner that runs UNTIL IT CONVERGES.

    Parameters
    ----------
    prob : (M, N) float array   probability heatmap (relative weights; need not
                                sum to 1).
    agents : list[Agent]
    cell_m : float              physical size of one grid cell, metres.
    tick_seconds : float        wall-clock duration of one planning tick. Each
                                agent advances speed*tick/cell cells per tick, so
                                faster craft cover more ground per tick.
    coverage_target : float     stop once this FRACTION of the probability is
                                cleared (e.g. 0.95).
    max_time_s : float          hard cap on mission time; stop even if the target
                                isn't met.
    stall_patience : int        stop after this many consecutive ticks with
                                negligible new coverage (search saturated).
    stall_eps : float           "negligible" coverage increment per tick.
    connectivity : 4 or 8       grid movement model (8 allows diagonals).
    allow_stay : bool           may an agent hold position for a sub-step.
    passable : (M, N) bool      cells an agent may occupy (water). Land/obstacles
                                are False; defaults to everywhere-passable.
    dispersion : float [0, 1]   how strongly agents keep apart (spread routes to
                                cover more ground). 0 = pure probability-greedy.
    """

    def __init__(self, prob, agents, cell_m, tick_seconds=20.0,
                 coverage_target=0.95, max_time_s=3600.0,
                 stall_patience=8, stall_eps=1e-4,
                 connectivity=8, allow_stay=True, passable=None, dispersion=0.0,
                 hard_cap_ticks=5000):
        self.prob = np.asarray(prob, dtype=np.float64)
        if self.prob.ndim != 2:
            raise ValueError("prob must be a 2D array")
        self.M, self.N = self.prob.shape
        self.agents = list(agents)
        self.cell_m = float(cell_m)
        self.tick_seconds = float(tick_seconds)
        self.coverage_target = float(np.clip(coverage_target, 0.0, 1.0))
        self.max_time_s = float(max_time_s)
        self.stall_patience = int(max(1, stall_patience))
        self.stall_eps = float(stall_eps)
        self.hard_cap_ticks = int(hard_cap_ticks)
        self.moves = _MOVES_8 if connectivity == 8 else _MOVES_4
        self.allow_stay = allow_stay
        self.dispersion = float(np.clip(dispersion, 0.0, 1.0))
        if passable is None:
            self.passable = np.ones((self.M, self.N), dtype=bool)
        else:
            self.passable = np.asarray(passable, dtype=bool)
            if self.passable.shape != self.prob.shape:
                raise ValueError("passable must match prob shape")

        # real units -> per-agent sonar cells and a per-tick DISTANCE budget
        # (metres). Using a distance budget -- not a cell count -- keeps the
        # physics honest: a diagonal step costs sqrt(2) cells, so an agent can't
        # secretly travel faster than its speed by zig-zagging.
        for a in self.agents:
            a._sonar_cells = max(0, int(round(a.sonar_radius_m / self.cell_m)))
            a._dist_per_tick = a.speed_mps * self.tick_seconds
            a._cells_per_tick = max(1, int(round(a._dist_per_tick / self.cell_m)))

        # remaining[r, c] = probability still UNSCANNED. Scanning zeroes it, so
        # the coverage objective is just the mass we remove from `remaining`.
        self.remaining = self.prob.copy()
        self.cleared_mask = np.zeros((self.M, self.N), dtype=bool)
        self._disks = {a._sonar_cells: _disk_offsets(a._sonar_cells)
                       for a in self.agents}
        self.total_prob = float(self.prob.sum())
        self.total_cleared = 0.0

    # -- geometry helpers ---------------------------------------------------
    def _in_bounds(self, r, c):
        return 0 <= r < self.M and 0 <= c < self.N

    def _can_enter(self, r, c):
        """In bounds AND passable (an agent may not step onto land/obstacles)."""
        return self._in_bounds(r, c) and self.passable[r, c]

    def _footprint(self, pos, sonar_cells):
        """Valid (rows, cols) index arrays of the sonar disk centred at pos."""
        offs = self._disks[sonar_cells]
        rr = offs[:, 0] + pos[0]
        cc = offs[:, 1] + pos[1]
        ok = (rr >= 0) & (rr < self.M) & (cc >= 0) & (cc < self.N)
        return rr[ok], cc[ok]

    def _gain(self, pos, sonar_cells):
        """Marginal probability that scanning at `pos` would newly clear."""
        rr, cc = self._footprint(pos, sonar_cells)
        return float(self.remaining[rr, cc].sum())

    def _apply_scan(self, pos, sonar_cells):
        """Mark the sonar disk at `pos` cleared; return the probability gained."""
        rr, cc = self._footprint(pos, sonar_cells)
        gained = float(self.remaining[rr, cc].sum())
        self.remaining[rr, cc] = 0.0
        self.cleared_mask[rr, cc] = True
        self.total_cleared += gained
        return gained

    def _attractor_step(self, pos, others=()):
        """Fallback when the local neighbourhood is exhausted: step toward the
        nearest heavy patch of remaining probability (a coarse potential field).

        With dispersion on, leftover patches FAR from the other agents are
        preferred, so idle craft peel off to different clusters."""
        if self.remaining.max() <= 0:
            return None                      # nothing left anywhere
        rr, cc = np.nonzero(self.remaining)
        d = np.hypot(rr - pos[0], cc - pos[1])
        score = self.remaining[rr, cc] / (1.0 + d)
        if self.dispersion > 0.0 and len(others):
            dmin = np.full(rr.shape, np.inf)
            for (orow, ocol) in others:
                dmin = np.minimum(dmin, np.hypot(rr - orow, cc - ocol))
            score = score * (1.0 + self.dispersion * dmin / (1.0 + dmin))
        tgt = (rr[np.argmax(score)], cc[np.argmax(score)])
        best, best_d = pos, np.hypot(pos[0] - tgt[0], pos[1] - tgt[1])
        for dr, dc in self.moves:
            nr, nc = pos[0] + dr, pos[1] + dc
            if not self._can_enter(nr, nc):
                continue
            nd = np.hypot(nr - tgt[0], nc - tgt[1])
            if nd < best_d:
                best, best_d = (nr, nc), nd
        return best

    def _commit_move(self, agent, best_move):
        """Move the agent, accrue route distance (m), and scan."""
        dr = best_move[0] - agent.pos[0]
        dc = best_move[1] - agent.pos[1]
        if dr or dc:
            agent.distance_m += math.hypot(dr, dc) * self.cell_m
        agent.pos = best_move
        agent.cleared_prob += self._apply_scan(best_move, agent._sonar_cells)
        agent.fine_path.append(best_move)

    def _agent_tick(self, agent):
        """Advance one agent by up to `_cells_per_tick` sub-steps for one tick.

        Greedy: each sub-step move to the neighbour (or stay) with the largest
        marginal sonar gain; if everything nearby is cleared, take the attractor
        pull toward remaining mass. With dispersion > 0, among the moves that
        still clear something prefer the one farthest from the other teams."""
        others = [b.pos for b in self.agents if b is not agent]
        budget = agent._dist_per_tick                     # metres left this tick
        while budget >= self.cell_m - 1e-9:
            pos = agent.pos
            candidates = list(self.moves) + ([(0, 0)] if self.allow_stay else [])

            cells, gains = [], []
            for dr, dc in candidates:
                nr, nc = pos[0] + dr, pos[1] + dc
                if not self._can_enter(nr, nc):
                    continue
                cells.append((nr, nc))
                gains.append(self._gain((nr, nc), agent._sonar_cells))
            gains = np.asarray(gains)

            if gains.size == 0 or gains.max() <= 0.0:
                # local area exhausted -> travel toward remaining mass
                nxt = self._attractor_step(pos, others)
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

            if best_move == pos:                          # staying gains nothing
                agent.fine_path.append(pos)
                break
            step_m = math.hypot(best_move[0] - pos[0],
                                best_move[1] - pos[1]) * self.cell_m
            self._commit_move(agent, best_move)
            budget -= step_m

        agent.path.append(agent.pos)

    def plan(self):
        """Run the coordinated plan UNTIL CONVERGENCE and return a results dict.

        Stops when coverage_target is reached, coverage saturates (no new ground
        for `stall_patience` ticks), or max_time_s is exceeded."""
        for a in self.agents:
            a.reset()
            a._sonar_cells = max(0, int(round(a.sonar_radius_m / self.cell_m)))
            a._dist_per_tick = a.speed_mps * self.tick_seconds
            a._cells_per_tick = max(1, int(round(a._dist_per_tick / self.cell_m)))
        self.remaining = self.prob.copy()
        self.cleared_mask[:] = False
        self.total_cleared = 0.0

        def frac():
            return self.total_cleared / self.total_prob if self.total_prob else 0.0

        # scan each agent's starting footprint up front (it sees where it stands)
        for a in self.agents:
            a.cleared_prob += self._apply_scan(a.pos, a._sonar_cells)

        history = [frac()]
        max_ticks = min(self.hard_cap_ticks,
                        int(math.ceil(self.max_time_s / self.tick_seconds)))
        tick = 0
        stalled = 0
        reason = 'max_ticks'
        if self.total_prob <= 0:
            reason = 'empty'
            max_ticks = 0

        while tick < max_ticks:
            # dynamic ordering: whoever can grab the most acts first
            order = sorted(self.agents,
                           key=lambda a: self._gain(a.pos, a._sonar_cells),
                           reverse=True)
            for a in order:
                self._agent_tick(a)
            tick += 1
            f = frac()
            gained = f - history[-1]
            history.append(f)

            if f >= self.coverage_target:
                reason = 'coverage_target'
                break
            if self.remaining.max() <= 0:
                reason = 'fully_cleared'
                break
            if gained < self.stall_eps:
                stalled += 1
                if stalled >= self.stall_patience:
                    reason = 'saturated'
                    break
            else:
                stalled = 0
        else:
            if self.total_prob > 0 and tick >= max_ticks:
                reason = 'time_budget'

        for a in self.agents:
            a.time_s = a.distance_m / a.speed_mps

        mission_time_s = tick * self.tick_seconds
        return {
            'agents': self.agents,
            'cleared_prob': self.total_cleared,
            'cleared_fraction': frac(),
            'coverage_over_time': history,        # cleared fraction after each tick
            'cleared_mask': self.cleared_mask,
            'remaining': self.remaining,
            'ticks': tick,
            'tick_seconds': self.tick_seconds,
            'mission_time_s': mission_time_s,
            'stop_reason': reason,
            'total_prob': self.total_prob,
        }


# ---------------------------------------------------------------------------
# Reference grid -- a coarse labelled overlay ("comms language") for the teams.
# ---------------------------------------------------------------------------
def _row_label(i):
    """0 -> 'A', 25 -> 'Z', 26 -> 'AA' (spreadsheet-style, for >26 rows)."""
    s = ''
    i += 1
    while i > 0:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def make_reference_grid(extent, target_cell_m=500.0, max_rows=26, max_cols=40):
    """Build a coarse LABELLED grid over the search area so teams can talk in a
    shared "B-4" language.

    Rows are lettered A, B, C ... from the NORTH (top of the map) downward;
    columns are numbered 1, 2, 3 ... from the WEST (left) eastward. The cell size
    starts at `target_cell_m` and is enlarged automatically if that would need
    more than `max_rows` rows or `max_cols` columns, so the grid always fits the
    area with sensible labels (and is typically much coarser than the algorithm
    grid).

    Parameters
    ----------
    extent : [lon0, lon1, lat0, lat1]   geographic bounds of the search area.

    Returns a JSON-serialisable dict:
        cell_m, rows, cols,
        lon_edges  (cols+1 west->east), lat_edges (rows+1 NORTH->south),
        row_labels (['A', 'B', ...]),   col_labels (['1', '2', ...]),
        extent.
    Use `reference_cell_label(grid, lon, lat)` to label any coordinate.
    """
    lon0, lon1, lat0, lat1 = [float(v) for v in extent]
    if lon1 < lon0:
        lon0, lon1 = lon1, lon0
    if lat1 < lat0:
        lat0, lat1 = lat1, lat0
    midlat = 0.5 * (lat0 + lat1)
    m_per_deg_lat = 110_540.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(midlat))
    width_m = max(1.0, (lon1 - lon0) * m_per_deg_lon)
    height_m = max(1.0, (lat1 - lat0) * m_per_deg_lat)

    cell = float(target_cell_m)
    ncols = max(1, math.ceil(width_m / cell))
    nrows = max(1, math.ceil(height_m / cell))
    if nrows > max_rows or ncols > max_cols:
        cell *= max(nrows / max_rows, ncols / max_cols)
        ncols = max(1, math.ceil(width_m / cell))
        nrows = max(1, math.ceil(height_m / cell))

    dlon = cell / m_per_deg_lon
    dlat = cell / m_per_deg_lat
    lon_edges = [round(lon0 + i * dlon, 6) for i in range(ncols + 1)]
    # north (lat1) first so row A is the top band on the map
    lat_edges = [round(lat1 - i * dlat, 6) for i in range(nrows + 1)]

    return {
        'cell_m': round(cell, 1),
        'rows': nrows,
        'cols': ncols,
        'lon_edges': lon_edges,
        'lat_edges': lat_edges,
        'row_labels': [_row_label(i) for i in range(nrows)],
        'col_labels': [str(i + 1) for i in range(ncols)],
        'extent': [lon0, lon1, lat0, lat1],
    }


def reference_cell_label(grid, lon, lat):
    """Return the 'B4'-style label of the reference-grid cell containing (lon,
    lat), or None if it falls outside the grid."""
    lon_edges = grid['lon_edges']
    lat_edges = grid['lat_edges']          # north -> south (descending)
    col = None
    for j in range(len(lon_edges) - 1):
        if lon_edges[j] <= lon <= lon_edges[j + 1]:
            col = j
            break
    row = None
    for i in range(len(lat_edges) - 1):
        if lat_edges[i] >= lat >= lat_edges[i + 1]:
            row = i
            break
    if row is None or col is None:
        return None
    return f"{grid['row_labels'][row]}{grid['col_labels'][col]}"


def analyse_complexity():
    """Return a short human-readable complexity note."""
    return (
        "Time:  O(ticks * K * cells_per_tick * R^2), ticks set by CONVERGENCE\n"
        "       (coverage target / saturation / time budget), not a fixed horizon.\n"
        "         K = #agents, R = sonar_m / cell_m, cells_per_tick = speed*tick/cell.\n"
        "Space: O(M*N) for the remaining/cleared grids + O(path) per agent.\n\n"
        "Fine-grid (small cell_m) tips:\n"
        "  * Keep R = sonar_m / cell_m modest, or add a summed-area table of\n"
        "    `remaining` for O(1) footprint sums instead of O(R^2).\n"
        "  * Plan on the high-probability CORE box, not the whole drift cloud.\n"
        "  * Raise tick_seconds (fewer, longer ticks) or coarsen for a first pass.\n"
        "  * numba @njit the inner sub-step loop for ~10-50x on CPU."
    )


# ---------------------------------------------------------------------------
# Stand-alone demo: a bimodal heatmap on a 3 m grid + a boat and a jet-ski.
# ---------------------------------------------------------------------------
def _demo():
    M = N = 120
    cell_m = 3.0                       # 3 m planning cells
    yy, xx = np.mgrid[0:M, 0:N]

    def gauss(cy, cx, s):
        return np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))

    prob = 1.0 * gauss(40, 45, 12) + 0.7 * gauss(80, 85, 14)
    prob /= prob.sum()

    agents = [
        Agent((60, 0),  speed_mps=7.0,  sonar_radius_m=60, name='Boat',   color='#3b82f6'),
        Agent((60, 119), speed_mps=14.0, sonar_radius_m=45, name='Jetski', color='#f97316'),
    ]

    planner = CoveragePlanner(prob, agents, cell_m=cell_m, tick_seconds=20.0,
                              coverage_target=0.95, max_time_s=3600,
                              dispersion=0.3)
    res = planner.plan()

    print("Heterogeneous coordinated search demo (converge mode)")
    print(f"  grid {M}x{N} @ {cell_m:.0f} m, {len(agents)} craft")
    print(f"  stopped: {res['stop_reason']} after {res['ticks']} ticks "
          f"= {res['mission_time_s']/60:.1f} min")
    for a in agents:
        print(f"  {a.name:7s} speed={a.speed_mps:4.1f} m/s  "
              f"{a._cells_per_tick} cells/tick  sonar={a._sonar_cells} cells  "
              f"cleared {a.cleared_prob*100:5.1f}%  "
              f"route {a.distance_m/1000:5.2f} km / {a.time_s/60:4.1f} min")
    print(f"  TOTAL probability cleared: {res['cleared_fraction']*100:.1f}%")

    grid = make_reference_grid([34.90, 34.97, 32.80, 32.86], target_cell_m=500)
    print(f"\n  reference grid: {grid['rows']}x{grid['cols']} cells @ "
          f"{grid['cell_m']:.0f} m  (rows {grid['row_labels'][0]}.."
          f"{grid['row_labels'][-1]}, cols 1..{grid['cols']})")
    print(f"  LKP cell label: {reference_cell_label(grid, 34.92, 32.83)}")
    print()
    print(analyse_complexity())
    return res


if __name__ == '__main__':
    _demo()
