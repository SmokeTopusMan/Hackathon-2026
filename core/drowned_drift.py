"""
DrownedBodyDrift -- an OpenDrift OceanDrift subclass that models the
sink -> rest-on-bottom -> refloat -> re-sink cycle of a drowned body.

This is the PHYSICS PRIOR for Nahshon. Unlike Leeway (a surface-only,
wind-driven model), a submerged body is moved by the 3D current AT ITS
OWN DEPTH and has no windage while underwater. OceanDrift gives us that
3D advection for free; this subclass adds the vertical *behaviour*.

State machine (per particle, vectorised over the whole ensemble):

    phase 1  DROWNING    terminal_velocity = 0  -> still at/near the surface,
                         struggling. Stays for `drown_delay` seconds (sampled
                         0..30 min per particle: we don't know exactly how long
                         after the LKP the victim actually went under), then
                         -> phase 0. THIS is where the simulation "begins".
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

BODY SIZE -> TIMING
-------------------
`sink_speed`, `rise_speed` and the `refloat_time` base are no longer fixed:
`body_dynamics()` below derives them from the victim's HEIGHT and WEIGHT
(BMI -> body-fat fraction -> density -> buoyancy/drag). Leaner bodies are
denser, so they sink faster and take longer to refloat; heavier-set bodies
are near-neutral, sink slowly and bob back up sooner. These are documented
first-order heuristics -- water temperature (the dominant real driver of
refloat time) is still TODO and slots into `body_dynamics`/`refloat_time_seconds`.
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
        # which phase of the cycle this particle is in (1 / 0 / 2 / 3 above)
        ('phase', {'dtype': np.float32, 'units': '1', 'default': 1.0}),
        # age_seconds at which the current phase began (for phase timers)
        ('phase_start_age', {'dtype': np.float32, 'units': 's', 'default': 0.0}),
        # how long THIS particle drifts at the surface BEFORE going under
        # (the uncertain LKP->drown delay, ~0..30 min)
        ('drown_delay', {'dtype': np.float32, 'units': 's', 'default': 0.0}),
        # how long THIS particle stays submerged before refloating
        ('refloat_time', {'dtype': np.float32, 'units': 's', 'default': 86400.0}),
        # how long THIS particle stays at the surface before re-sinking
        ('surface_time', {'dtype': np.float32, 'units': 's', 'default': 6 * 3600.0}),
    ])


class DrownedBodyDrift(OceanDrift):
    """3D drift of a drowned body with sink / rest / refloat / re-sink."""

    ElementType = DrownedBody

    # vertical speeds (m/s). Defaults are a sane fallback for an average adult;
    # the runner overrides o.sink_speed / o.rise_speed from body_dynamics(height,
    # weight) so they reflect the actual victim.
    sink_speed = 0.30   # descent speed (negative terminal_velocity)
    rise_speed = 0.20   # ascent speed while bloated

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

        # --- transitions (use this step's phase + time-in-phase) ----------
        # DROWNING  -> SUBMERGED   victim finally goes under
        # SUBMERGED -> RISING      submerged long enough (refloat_time)
        # RISING    -> AT SURFACE  breaks the surface
        # AT SURFACE-> SUBMERGED   gas escapes, re-sinks (surface_time)
        for mask, new_phase in (
            ((e.phase == 1) & (tip >= e.drown_delay),  0),
            ((e.phase == 0) & (tip >= e.refloat_time), 2),
            ((e.phase == 2) & (e.z >= -SURFACE_EPS),   3),
            ((e.phase == 3) & (tip >= e.surface_time), 0),
        ):
            e.phase[mask] = new_phase
            e.phase_start_age[mask] = age[mask]

        # --- set vertical velocity from the UPDATED phase -----------------
        e.terminal_velocity[e.phase == 1] = 0.0            # drowning, at surface
        e.terminal_velocity[e.phase == 0] = -self.sink_speed
        e.terminal_velocity[e.phase == 2] = +self.rise_speed
        e.terminal_velocity[e.phase == 3] = 0.0            # afloat after refloat

        # OceanDrift's buoyancy only moves particles already BELOW the surface
        # (z < 0); a body that has just gone under is still at z = 0 and would
        # otherwise stay pinned at the surface. Nudge it just beneath so the
        # sink actually starts (and it then drifts with the current AT DEPTH).
        just_under = (e.phase == 0) & (e.z >= -SURFACE_EPS)
        e.z[just_under] = -SURFACE_EPS - 0.5

        # ------------------------------------------------------------------
        # 2. Let OceanDrift do the actual physics: horizontal advection by
        #    the 3D current at each particle's depth, plus vertical motion
        #    from the terminal_velocity we just set, plus seafloor clamping.
        # ------------------------------------------------------------------
        super().update()


# ===========================================================================
# Body size -> vertical dynamics & timing
# ===========================================================================
# First-order, documented heuristics. They give physically-sensible TRENDS
# (leaner = denser = sinks faster & refloats slower; heavier-set = near
# neutral = sinks slowly & bobs back sooner) without claiming forensic
# accuracy. Water temperature -- the dominant real driver of refloat time --
# is still TODO; it would multiply `refloat` in body_dynamics().

SEAWATER_DENSITY = 1027.0   # kg/m^3, Mediterranean surface (tune for the basin)
GRAVITY          = 9.81     # m/s^2
GAS_BUOYANCY     = 12.0     # kg/m^3 effective lift once decomposition-bloated
_DRAG_CD         = 1.1      # drag coeff of an irregular body
_AREA_PER_M      = 0.22     # frontal area (m^2) per metre of height, rough


def estimate_body_density(height_m, weight_kg):
    """Density (kg/m^3) of a DROWNED body (lungs flooded), from height & weight.

    BMI -> body-fat fraction (Deurenberg, simplified & sex/age-neutral)
        -> two-compartment (Siri) fat(900)/lean(1100) mix
        -> + a small increment for water-filled lungs.
    Returns (density, bmi, fat_fraction).
    """
    bmi = weight_kg / height_m**2
    fat = np.clip(0.012 * bmi - 0.02, 0.03, 0.55)          # fraction 3%..55%
    rho_tissue = 1.0 / (fat / 900.0 + (1.0 - fat) / 1100.0)
    return rho_tissue + 18.0, bmi, fat                     # lungs flooded


def _terminal_velocity(delta_rho, volume_m3, height_m, water_density):
    """Vertical terminal speed (m/s, magnitude) from a buoyancy/drag balance."""
    area = _AREA_PER_M * height_m
    force = abs(delta_rho) * volume_m3 * GRAVITY
    return float(np.sqrt(2.0 * force / (water_density * _DRAG_CD * area)))


# reference adult (75 kg, 1.75 m) ~ 1 day to refloat in temperate water
_REF_RHO, _, _ = estimate_body_density(1.75, 75.0)
_REF_FORCE = max(_REF_RHO - SEAWATER_DENSITY, 1.0) * (75.0 / _REF_RHO)


def body_dynamics(height_m, weight_kg, water_density=SEAWATER_DENSITY):
    """Map a victim's height & weight to sink/rise speeds and a refloat-time base.

    Returns a dict: sink (m/s), rise (m/s), refloat (s), density (kg/m^3),
    bmi, fat. All clamped to physically plausible ranges.
    """
    rho, bmi, fat = estimate_body_density(height_m, weight_kg)
    volume = weight_kg / rho
    d_sink = max(rho - water_density, 1.0)                 # ensure it sinks
    sink = float(np.clip(
        _terminal_velocity(d_sink, volume, height_m, water_density), 0.01, 0.6))
    rise = float(np.clip(
        _terminal_velocity(GAS_BUOYANCY, volume, height_m, water_density),
        0.005, 0.4))
    # gas must overcome the body's negative buoyancy -> scale vs the reference
    refloat = 86400.0 * (d_sink * volume) / _REF_FORCE
    refloat = float(np.clip(refloat, 0.25 * 86400, 5 * 86400))
    return dict(sink=sink, rise=rise, refloat=refloat,
                density=rho, bmi=bmi, fat=float(fat))


def drown_delay_seconds(n, max_minutes=30.0, mode_minutes=4.0, rng=None):
    """Per-particle time from the LKP until the victim actually goes under (s).

    Triangular over [0, max] peaking at `mode`: most victims drown within a
    few minutes, but some linger up to `max`. This smears the effective start
    of the underwater drift, widening the LKP-time uncertainty.
    """
    if rng is None:
        return np.full(n, mode_minutes * 60.0, dtype=np.float32)
    mins = rng.triangular(0.0, mode_minutes, max_minutes, size=n)
    return (mins * 60.0).astype(np.float32)


def refloat_time_seconds(n, height_m, weight_kg, rng=None,
                         water_density=SEAWATER_DENSITY, spread_frac=0.25):
    """Per-particle submerged duration before refloating, in seconds.

    Base value comes from `body_dynamics` (height/weight driven); a +/-
    `spread_frac` random jitter keeps the ensemble from refloating all at
    once (which would give an unphysically sharp ring in the heatmap).
    """
    base = body_dynamics(height_m, weight_kg, water_density)['refloat']
    if rng is None:
        return np.full(n, base, dtype=np.float32)
    jitter = rng.uniform(-spread_frac, spread_frac, size=n)
    return np.clip(base * (1.0 + jitter), 600.0, None).astype(np.float32)
