"""
Brute-force joint multi-vehicle search-path optimization (Approach 1).

Given a precomputed sequence of 2D probability heatmaps and a fleet of search
vehicles, this plans cooperative paths one short timestep at a time. At each
step every vehicle has 16 candidate headings (every 22.5 degrees); the planner
brute-forces all combinations of the vehicles' candidate moves and commits the
combination that clears the most probability mass from the look-ahead heatmap.

TIMESTEP vs HEATMAP SPACING
---------------------------
The planner timestep (config.timestep_s) is independent of the spacing between
heatmaps (config.heatmap_interval_s). Each heatmap is held across
round(heatmap_interval_s / timestep_s) sub-steps, so vehicles take many short
turns inside one heatmap frame and the routes curve, instead of committing to a
single long straight leg per frame. The look-ahead heatmap for planner step t is
heatmaps[min(t // substeps + 1, last)].

PER-CELL CLEARING
-----------------
Clearing combines detection probability with the fraction of a cell the sonar
swath actually covers:  clearing = P_d * min(1, swath_width / cell_size). For a
swath smaller than a cell this avoids crediting a whole coarse cell as searched.
A swept cell keeps a cumulative factor of (1 - clearing); the factor carries
forward to all future heatmaps so already-searched ground is discounted. Cells
swept by several vehicles take the maximum clearing among them (no double count).

KNOWN APPROXIMATIONS (acceptable for the POC)
---------------------------------------------
  * The sweep factor does NOT advect with the current field: a cell discounted at
    one frame stays discounted at that grid location even though the physical
    probability mass there has drifted by later frames.
  * The swath is the rasterized centerline (dilated by a disk only for swaths
    wider than a cell), not a true oriented rectangle.

The auction-based coordinator (Approach 3) is not implemented here; it would be a
drop-in replacement for `select_joint_move` with the same signature.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Vehicle:
    id: str
    type: str
    velocity_mps: float
    swath_width_m: float
    start_position: tuple[float, float]
    detection_probability: float = 0.8


@dataclass(frozen=True)
class GridMeta:
    rows: int
    cols: int
    cell_size_m: float
    origin_lat: float = 0.0
    origin_lon: float = 0.0
    navigable_mask: np.ndarray | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class PlanConfig:
    timestep_s: float = 120.0
    coverage_threshold: float = 0.85
    n_headings: int = 16
    heatmap_interval_s: float = 3600.0


@dataclass(frozen=True)
class Move:
    vehicle_id: str
    next_position: tuple[float, float]
    swath_cells: tuple[tuple[int, int], ...] = field(default_factory=tuple)


def _disk_offsets(radius: int) -> list[tuple[int, int]]:
    if radius <= 0:
        return [(0, 0)]
    return [(dr, dc)
            for dr in range(-radius, radius + 1)
            for dc in range(-radius, radius + 1)
            if dr * dr + dc * dc <= radius * radius]


def _bresenham(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dc - dr
    r, c = r0, c0
    while True:
        cells.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c += sc
        if e2 < dc:
            err += dc
            r += sr
    return cells


def _line_cells(
    a: tuple[float, float],
    b: tuple[float, float],
    rows: int,
    cols: int,
) -> list[tuple[int, int]]:
    """Rasterized centerline from continuous (x, y) point `a` to `b`, as in-bounds
    (row, col) cells. Endpoints are rounded and clamped to the grid."""
    c0 = min(max(int(round(a[0])), 0), cols - 1)
    r0 = min(max(int(round(a[1])), 0), rows - 1)
    c1 = min(max(int(round(b[0])), 0), cols - 1)
    r1 = min(max(int(round(b[1])), 0), rows - 1)
    return _bresenham(r0, c0, r1, c1)


def _dilate(
    line: Sequence[tuple[int, int]],
    disk: Sequence[tuple[int, int]],
    rows: int,
    cols: int,
) -> tuple[tuple[int, int], ...]:
    """Sonar footprint: the centerline cells dilated by the swath-radius disk,
    deduplicated and clipped to the grid."""
    touched: set[tuple[int, int]] = set()
    for r, c in line:
        for dr, dc in disk:
            rr, cc = r + dr, c + dc
            if 0 <= rr < rows and 0 <= cc < cols:
                touched.add((rr, cc))
    return tuple(touched)


def _clearing_by_id(
    vehicles: Sequence[Vehicle],
    grid_meta: GridMeta,
) -> dict[str, float]:
    """Per-vehicle effective per-sweep clearing probability,
    P_d * min(1, swath_width / cell_size)."""
    return {v.id: v.detection_probability
            * min(1.0, v.swath_width_m / grid_meta.cell_size_m)
            for v in vehicles}


def _lookahead_frame(t: int, n_frames: int, config: PlanConfig) -> int:
    """Index of the heatmap a planner step `t` is scored against, holding each
    frame across round(heatmap_interval_s / timestep_s) sub-steps."""
    substeps = max(1, round(config.heatmap_interval_s / config.timestep_s))
    return min(t // substeps + 1, n_frames - 1)


def _candidate_moves(
    vehicle: Vehicle,
    position: tuple[float, float],
    grid_meta: GridMeta,
    config: PlanConfig,
) -> list[Move]:
    distance_cells = vehicle.velocity_mps * config.timestep_s / grid_meta.cell_size_m
    half_width_cells = round(vehicle.swath_width_m / (2.0 * grid_meta.cell_size_m))
    disk = _disk_offsets(half_width_cells)
    mask = grid_meta.navigable_mask
    rows, cols = grid_meta.rows, grid_meta.cols
    moves: list[Move] = []
    for k in range(config.n_headings):
        angle = 2.0 * math.pi * k / config.n_headings
        nx = position[0] + distance_cells * math.cos(angle)
        ny = position[1] + distance_cells * math.sin(angle)
        if not (0.0 <= nx < cols and 0.0 <= ny < rows):
            continue
        line = _line_cells(position, (nx, ny), rows, cols)
        if mask is not None and any(not mask[r, c] for r, c in line):
            continue
        cells = _dilate(line, disk, rows, cols)
        moves.append(Move(vehicle.id, (nx, ny), cells))
    if not moves:
        moves.append(Move(vehicle.id, position, ()))
    return moves


def _combo_cell_clearing(
    moves: Sequence[Move],
    clear_by_id: dict[str, float],
) -> dict[tuple[int, int], float]:
    cell_clear: dict[tuple[int, int], float] = {}
    for move in moves:
        d = clear_by_id[move.vehicle_id]
        for cell in move.swath_cells:
            if d > cell_clear.get(cell, -1.0):
                cell_clear[cell] = d
    return cell_clear


def _combo_mass(
    moves: Sequence[Move],
    effective: np.ndarray,
    clear_by_id: dict[str, float],
) -> float:
    cell_clear = _combo_cell_clearing(moves, clear_by_id)
    total = 0.0
    for (r, c), d in cell_clear.items():
        total += float(effective[r, c]) * d
    return total


def select_joint_move(
    heatmaps: Sequence[np.ndarray],
    vehicles: Sequence[Vehicle],
    current_positions: list[tuple[float, float]],
    effective_factor: np.ndarray,
    t: int,
    grid_meta: GridMeta,
    config: PlanConfig,
) -> tuple[Move, ...]:
    frame_idx = _lookahead_frame(t, len(heatmaps), config)
    effective = heatmaps[frame_idx] * effective_factor
    clear_by_id = _clearing_by_id(vehicles, grid_meta)
    candidate_lists = [
        _candidate_moves(v, current_positions[i], grid_meta, config)
        for i, v in enumerate(vehicles)
    ]
    best_combo: tuple[Move, ...] = tuple(c[0] for c in candidate_lists)
    best_total = -1.0
    for combo in itertools.product(*candidate_lists):
        total = _combo_mass(combo, effective, clear_by_id)
        if total > best_total:
            best_total = total
            best_combo = combo
    return best_combo


def plan_search(
    heatmaps: Sequence[np.ndarray],
    vehicles: Sequence[Vehicle],
    grid_meta: GridMeta,
    config: PlanConfig | None = None,
) -> dict:
    config = config or PlanConfig()
    n_frames = len(heatmaps)
    substeps = max(1, round(config.heatmap_interval_s / config.timestep_s))
    max_steps = max(0, (n_frames - 1) * substeps)

    original_total_mass = float(np.asarray(heatmaps[0]).sum())
    clear_by_id = _clearing_by_id(vehicles, grid_meta)
    cumulative_sweep_factor = np.ones((grid_meta.rows, grid_meta.cols), dtype=np.float64)

    current_positions: list[tuple[float, float]] = [
        (float(v.start_position[0]), float(v.start_position[1])) for v in vehicles
    ]
    paths: dict[str, list[tuple[float, float]]] = {
        v.id: [current_positions[i]] for i, v in enumerate(vehicles)
    }

    swept_mass = 0.0
    timesteps_used = 0
    termination_reason = "t_max"

    for step in range(max_steps):
        moves = select_joint_move(
            heatmaps, vehicles, current_positions,
            cumulative_sweep_factor, step, grid_meta, config,
        )

        if all(len(m.swath_cells) == 0 for m in moves):
            termination_reason = "no_valid_moves"
            break

        frame_idx = _lookahead_frame(step, n_frames, config)
        nxt = heatmaps[frame_idx]
        cell_clear = _combo_cell_clearing(moves, clear_by_id)
        reduction = 0.0
        for (r, c), d in cell_clear.items():
            reduction += float(nxt[r, c]) * float(cumulative_sweep_factor[r, c]) * d
        for (r, c), d in cell_clear.items():
            cumulative_sweep_factor[r, c] *= (1.0 - d)
        swept_mass += reduction

        for i, move in enumerate(moves):
            current_positions[i] = move.next_position
            paths[move.vehicle_id].append(move.next_position)

        timesteps_used = step + 1

        coverage = swept_mass / original_total_mass if original_total_mass > 0.0 else 0.0
        if coverage >= config.coverage_threshold:
            termination_reason = "coverage_reached"
            break
        if step + 1 == max_steps:
            termination_reason = "t_max"
            break

    swept_mass_fraction = (
        swept_mass / original_total_mass if original_total_mass > 0.0 else 0.0
    )
    return {
        "paths": paths,
        "swept_mass_fraction": swept_mass_fraction,
        "timesteps_used": timesteps_used,
        "termination_reason": termination_reason,
    }
