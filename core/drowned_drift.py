"""
DrownedBodyDrift -- an OpenDrift OceanDrift subclass that models the
sink -> rest-on-bottom -> refloat -> re-sink cycle of a drowned body.

This is the PHYSICS PRIOR for Nahshon. Unlike Leeway (a surface-only,
wind-driven model), a submerged body is moved by the 3D current AT ITS
OWN DEPTH and has no windage while underwater. OceanDrift gives us that
3D advection for free; this subclass adds the vertical *behaviour*.

State machine (per particle, vectorised over the whole ensemble):

    phase 0  SUBMERGED   terminal_velocity < 0  -> sinks, then sticks to
                         the seafloor and drifts with the bottom current.
                         Stays for `refloat_time` seconds, then -> phase 2.
    phase 2  RISING      terminal_velocity > 0  -> floats up. When it
                         reaches the surface (z >= -SURFACE_EPS) -> phase 3.
    phase 3  AT SURFACE  terminal_velocity = 0  -> drifts at the surface
                         for `surface_time` seconds, then gas escapes and
                         it sinks again -> phase 0.

The whole cycle repeats, so a body can surface, drift, and re-sink several
times over a long search window -- exactly the behaviour SAR teams report.

WHAT IS DELIBERATELY SIMPLE FOR NOW
-----------------------------------
`refloat_time` is set by `refloat_time_seconds()` below, currently a
CONSTANT (independent of depth/temperature). That is the agreed placeholder:
later it becomes a real function of water temperature (accumulated
degree-days), depth, body mass, etc. Everything downstream already treats
it as a per-particle value, so swapping in a smarter function is a one-line
change with no other code touched.
"""

import numpy as np
from opendrift.models.oceandrift import OceanDrift, Lagrangian3DArray


# --- sign convention reminder (verified against OpenDrift 1.14) -------------
# z is metres, NEGATIVE downwards (0 = surface).
# terminal_velocity is metres/s: POSITIVE = upward (rising),
#                                NEGATIVE = downward (sinking).
# z is automatically clamped to 0 at the top and to the seafloor at the
# bottom by OceanDrift.vertical_buoyancy(), so we never have to clamp it.

SURFACE_EPS = 0.1  # m; treat z >= -0.1 as "at the surface"


class DrownedBody(Lagrangian3DArray):
    """OceanDrift element + the extra per-particle state our model needs."""

    variables = Lagrangian3DArray.add_variables([
        # which phase of the cycle this particle is in (0 / 2 / 3 above)
        ('phase', {'dtype': np.float32, 'units': '1', 'default': 0.0}),
        # age_seconds at which the current phase began (for phase timers)
        ('phase_start_age', {'dtype': np.float32, 'units': 's', 'default': 0.0}),
        # how long THIS particle stays submerged before refloating
        ('refloat_time', {'dtype': np.float32, 'units': 's', 'default': 86400.0}),
        # how long THIS particle stays at the surface before re-sinking
        ('surface_time', {'dtype': np.float32, 'units': 's', 'default': 6 * 3600.0}),
    ])


class DrownedBodyDrift(OceanDrift):
    """3D drift of a drowned body with sink / rest / refloat / re-sink."""

    ElementType = DrownedBody

    # vertical speeds (m/s). Tunable; small, like a real body.
    sink_speed = 0.03   # ~3 cm/s descent  (negative terminal_velocity)
    rise_speed = 0.02   # ~2 cm/s ascent while bloated

    # OceanDrift needs to know which environment variables it may demand and
    # what to fall back to when a reader doesn't provide them. We keep the
    # OceanDrift defaults and only ensure a seafloor depth is available.
    required_variables = dict(OceanDrift.required_variables)
    required_variables['sea_floor_depth_below_sea_level'] = {'fallback': 100.0}

    def update(self):
        # ------------------------------------------------------------------
        # 1. Vertical BEHAVIOUR: set each particle's terminal_velocity and
        #    advance the phase state machine. Do this BEFORE OceanDrift moves
        #    things, so the velocity we set is the one applied this step.
        # ------------------------------------------------------------------
        e = self.elements
        age = e.age_seconds
        tip = age - e.phase_start_age          # time-in-phase (s)

        submerged = e.phase == 0
        rising    = e.phase == 2
        surfaced  = e.phase == 3

        # set vertical velocity per phase
        e.terminal_velocity[submerged] = -self.sink_speed
        e.terminal_velocity[rising]    = +self.rise_speed
        e.terminal_velocity[surfaced]  = 0.0

        # --- transitions --------------------------------------------------
        # SUBMERGED -> RISING once submerged long enough
        to_rise = submerged & (tip >= e.refloat_time)
        e.phase[to_rise] = 2
        e.phase_start_age[to_rise] = age[to_rise]

        # RISING -> AT SURFACE once it breaks the surface
        at_surface = rising & (e.z >= -SURFACE_EPS)
        e.phase[at_surface] = 3
        e.phase_start_age[at_surface] = age[at_surface]

        # AT SURFACE -> SUBMERGED once gas escapes
        to_sink = surfaced & (tip >= e.surface_time)
        e.phase[to_sink] = 0
        e.phase_start_age[to_sink] = age[to_sink]

        # ------------------------------------------------------------------
        # 2. Let OceanDrift do the actual physics: horizontal advection by
        #    the 3D current at each particle's depth, plus vertical motion
        #    from the terminal_velocity we just set, plus seafloor clamping.
        # ------------------------------------------------------------------
        super().update()


# ---------------------------------------------------------------------------
# The placeholder timing law. Swap the body of this function later.
# ---------------------------------------------------------------------------
def refloat_time_seconds(n, depth_m=None, rng=None):
    """Per-particle submerged duration before refloating, in seconds.

    CURRENT (placeholder): a constant ~1 day for every particle, with a
    small random spread so the ensemble doesn't refloat all at once (which
    would give an unphysically sharp ring in the heatmap).

    `depth_m` is accepted but unused for now -- it's here so the later
    temperature/depth-driven version is a drop-in replacement.
    """
    base = 24 * 3600.0          # 1 day
    spread = 6 * 3600.0         # +/- a few hours so the refloat isn't a knife-edge
    if rng is None:
        return np.full(n, base, dtype=np.float32)
    return (base + rng.uniform(-spread, spread, size=n)).astype(np.float32)
