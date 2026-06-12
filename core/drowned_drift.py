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

    # -- effective drift weighting (current vs wave Stokes) ------------------
    # WHY THIS EXISTS: a body's horizontal motion is current + wave Stokes drift
    # (+ wind). For a FLOATING object the wave-driven Stokes transport often
    # DOMINATES the slow background current near shore -- which is why a
    # current-only model is inaccurate for surface drifters. We therefore form an
    # *effective* drift  =  weight_current * current  +  weight_stokes * stokes,
    # with the weights set by the runner (see DRIFT_WEIGHT_* in
    # test/sim_drowned_body.py). OpenDrift gives us the hooks for free:
    #   * the ocean current is scaled per-particle by `current_drift_factor`
    #     (used inside OceanDrift.advect_ocean_current),
    #   * the Stokes drift is scaled by overriding stokes_drift() below.
    #
    # PHYSICS NOTE: down-weighting the current is only appropriate for the AFLOAT
    # phases (a body AT the surface, where waves act). A SUBMERGED body has no
    # wave drift and must follow the FULL current -- so when `weight_afloat_only`
    # is True (default) we apply the weights only to the afloat phases and keep
    # the submerged phases on the full current. Set it False to weight globally.
    wave_weighting   = False   # off by default -> identical to plain OceanDrift
    weight_current   = 1.0     # multiplier on the ocean current (afloat phases)
    weight_stokes    = 1.0     # multiplier on the wave Stokes drift
    weight_afloat_only = True  # weight only the surface phases (physically right)

    # -- nearshore (surf-zone) longshore current ----------------------------
    # A 4.2 km basin model cannot resolve the SURF-ZONE longshore current: the
    # wave-driven current within a few km of the beach that carries a floating
    # object ALONG the shore. Along the Israeli coast the NET-ANNUAL longshore
    # drift is northward, but it reverses SOUTHWARD in spring/summer (the wave-
    # induced longshore sediment transport studies for the Israeli shelf). We add
    # it as a signed alongshore (meridional) current that is strong at the coast
    # and decays offshore, applied to the AFLOAT phases. The sign/magnitude is a
    # seasonal knob (negative = southward, for the spring case here); offshore
    # cases are untouched because the term decays to zero away from the beach.
    nearshore_longshore_v = 0.0   # m/s alongshore near the coast; + north / - south
    nearshore_scale_km    = 5.0   # offshore e-folding distance of the surf zone
    nearshore_afloat_boost = 2.0  # a FLOATING body feels the surf-zone current more
    coast_tangent_e = 0.28        # north-alongshore unit vector (east comp) for the
    coast_tangent_n = 0.96        #   Israeli central coast (~16 deg tilt from N)
    # The southward longshore reversal is NOT a basin-wide feature: it is observed
    # on the open central-shelf (Netanya / Herzliya), while Haifa Bay to the north
    # -- a curved, sheltered embayment -- keeps the general northward flow. So the
    # current only acts inside a latitude band, full strength within [lat_min,
    # lat_max] and tapering linearly to zero over `lat_taper` deg on each side.
    # Leave lat_min/lat_max = None to apply it everywhere (old behaviour).
    nearshore_lat_min   = None    # deg N; south edge of the active band
    nearshore_lat_max   = None    # deg N; north edge of the active band
    nearshore_lat_taper = 0.15    # deg of smooth roll-off outside the band
    coast_distance_fn     = None  # callable (lat, lon) -> km to the nearest coast
    # Only a FLOATING body beaches; a SUBMERGED one drifts along the bottom past
    # the shore. With coastline_action='previous' OpenDrift bounces everything off
    # the coast, and we strand only the afloat phases within this distance.
    strand_afloat_km = 0.3

    def stokes_drift(self, factor=1):
        """Scale the wave Stokes drift by `weight_stokes` so it can be made the
        dominant term of the effective drift. OpenDrift already decays Stokes
        with depth, so the submerged phases barely feel it regardless."""
        super().stokes_drift(factor=factor * self.weight_stokes)

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

        # --- effective-drift weighting: scale the ocean current --------------
        # current_drift_factor multiplies the current inside advect_ocean_current
        # (the Stokes side is scaled in stokes_drift() above). Phases 1 (drowning,
        # at surface) and 3 (afloat after refloat) are FLOATING -> down-weight the
        # current so waves dominate. Phases 0/2 are SUBMERGED/rising in the water
        # column -> keep the FULL current. Set weight_afloat_only=False to apply
        # the current weight everywhere.
        if self.wave_weighting:
            afloat = (e.phase == 1) | (e.phase == 3)
            if self.weight_afloat_only:
                e.current_drift_factor[:] = 1.0
                e.current_drift_factor[afloat] = self.weight_current
            else:
                e.current_drift_factor[:] = self.weight_current

        # ------------------------------------------------------------------
        # 2. Let OceanDrift do the actual physics: horizontal advection by
        #    the 3D current at each particle's depth, plus vertical motion
        #    from the terminal_velocity we just set, plus seafloor clamping.
        # ------------------------------------------------------------------
        super().update()

        # ------------------------------------------------------------------
        # 3 + 4. Nearshore longshore current and phase-aware beaching. Both need
        #    the distance-to-coast field and the afloat mask, so compute each
        #    ONCE per step (the coast-distance interpolation is the costliest part
        #    of update()). The alongshore displacement in step 3 is tangent to the
        #    coast, so it changes distance-to-coast by far less than a metre --
        #    negligible against the ~300 m beaching threshold -- hence reusing the
        #    same d_km for the beaching test in step 4 is numerically equivalent.
        # ------------------------------------------------------------------
        if self.coast_distance_fn is not None:
            d_km = np.asarray(self.coast_distance_fn(e.lat, e.lon), dtype=float)
            afloat = (e.phase == 1) | (e.phase == 3)

            # 3. Nearshore (surf-zone) longshore current: a signed alongshore
            #    current for AFLOAT objects close to the beach, decaying offshore.
            if self.nearshore_longshore_v:
                v = self.nearshore_longshore_v * np.exp(-d_km / self.nearshore_scale_km)
                # GEOGRAPHIC limit: only let it act inside the central-shelf
                # latitude band (Netanya area); zero it out toward Haifa Bay,
                # which does not show the reversal. Trapezoidal window in lat.
                if self.nearshore_lat_min is not None and self.nearshore_lat_max is not None:
                    t = max(self.nearshore_lat_taper, 1e-6)
                    below = np.clip((e.lat - (self.nearshore_lat_min - t)) / t, 0.0, 1.0)
                    above = np.clip(((self.nearshore_lat_max + t) - e.lat) / t, 0.0, 1.0)
                    v = v * below * above
                # the longshore current moves bedload (a body rolling on the
                # shallow bottom) AND floating bodies -- the floating ones more.
                v = v * np.where(afloat, self.nearshore_afloat_boost, 1.0)
                # follow the COASTLINE TANGENT (Israeli central coast tilts ~16
                # deg), so the current runs ALONG the shore instead of into it --
                # +v = north-alongshore (NNE), -v = south-alongshore (SSW).
                te, tn = self.coast_tangent_e, self.coast_tangent_n
                dt = self.time_step.total_seconds()
                coslat = np.maximum(np.cos(np.radians(e.lat)), 1e-6)
                e.lat += (v * tn * dt) / 110570.0
                e.lon += (v * te * dt) / (111320.0 * coslat)

            # 4. Phase-aware beaching. A SUBMERGED body rolls along the bottom and
            #    drifts PAST the shore (it cannot wash up while underwater), so
            #    with coastline_action='previous' OpenDrift just nudges it back to
            #    deeper water and it keeps going. Only a FLOATING body strands --
            #    we deactivate the afloat phases once they reach the surf line.
            #    This matches the physics and stops the whole cloud from beaching
            #    at the entry point before it can travel alongshore.
            if np.any(afloat):
                beach = afloat & (d_km < self.strand_afloat_km)
                if np.any(beach):
                    self.deactivate_elements(beach, reason='stranded')


# ===========================================================================
# Body size -> vertical dynamics & timing
# ===========================================================================
# First-order, documented heuristics. They give physically-sensible TRENDS
# (leaner = denser = sinks faster & refloats slower; heavier-set = near
# neutral = sinks slowly & bobs back sooner). Water temperature -- the dominant
# real driver of WHEN a body refloats -- now drives the refloat time via
# Accumulated Degree Days (see refloat_days below).

SEAWATER_DENSITY = 1027.0   # kg/m^3, Mediterranean surface (tune for the basin)
GRAVITY          = 9.81     # m/s^2
GAS_BUOYANCY     = 12.0     # kg/m^3 effective lift once decomposition-bloated
_DRAG_CD         = 1.1      # drag coeff of an irregular body
_AREA_PER_M      = 0.22     # frontal area (m^2) per metre of height, rough

# --- REFLOAT / FLOAT timing, temperature-driven (forensic) -----------------
# A drowned body sinks, then decomposition gas refloats it after a delay set
# mostly by WATER TEMPERATURE. Forensic studies put resurfacing at an Accumulated
# Degree Days (ADD) of ~100-140 deg C-days: ~7-10 d at ~16 C, ~3-7 d at ~21 C,
# ~1-2 d above ~27 C (justanswer / ScienceDirect ADD studies). We model:
#       refloat_days ~= ADD_REFLOAT_DEGDAYS / T_water        (x a mild body factor)
# and then the body FLOATS for a few days (FLOAT_MIN/MAX_DAYS) before gas escapes
# and it re-sinks. The per-particle SPREAD is wide on purpose so the ensemble
# surfaces across a range of days -- some bodies are afloat (and wave-driven)
# during whatever wave window matters, instead of all surfacing at the same hour.
# TUNE THESE THREE for the basin / season:
ADD_REFLOAT_DEGDAYS  = 100.0   # deg C-days to reach the bloat/refloat stage
DEFAULT_WATER_TEMP_C = 19.0    # Mediterranean spring SST used if none supplied
MIN_WATER_TEMP_C     = 4.0     # floor so cold water doesn't give infinite refloat
FLOAT_MIN_DAYS       = 2.0     # a refloated body stays up ~2-5 days (temperate)
FLOAT_MAX_DAYS       = 5.0


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


def refloat_days(water_temp_c=None, height_m=1.75, weight_kg=75.0):
    """Days a drowned body stays SUBMERGED before decomposition gas refloats it.

    Temperature-driven via Accumulated Degree Days (resurface at ADD ~100-140
    deg C-days), with a mild body-buoyancy modulation (leaner = denser = a bit
    slower; heavier-set = near-neutral = a bit faster). Examples at the default
    ADD: ~19 C (Israeli spring) -> ~5 d; ~26 C (summer) -> ~4 d; ~14 C -> ~7 d.
    """
    T = max(DEFAULT_WATER_TEMP_C if water_temp_c in (None, '') else float(water_temp_c),
            MIN_WATER_TEMP_C)
    base = ADD_REFLOAT_DEGDAYS / T
    body_factor = float(np.clip(
        body_dynamics(height_m, weight_kg)['refloat'] / 86400.0, 0.6, 1.6))
    return float(np.clip(base * body_factor, 0.5, 30.0))


def refloat_time_seconds(n, height_m, weight_kg, rng=None, water_temp_c=None,
                         spread_frac=0.4):
    """Per-particle submerged duration before refloating, in seconds, TEMPERATURE
    driven (see refloat_days). The wide +/- `spread_frac` jitter spreads the
    resurfacing across a range of days so the ensemble isn't a single sharp ring
    and some bodies are afloat during whatever wave window matters."""
    base = refloat_days(water_temp_c, height_m, weight_kg) * 86400.0
    if rng is None:
        return np.full(n, base, dtype=np.float32)
    jitter = rng.uniform(-spread_frac, spread_frac, size=n)
    return np.clip(base * (1.0 + jitter), 3600.0, None).astype(np.float32)


def surface_time_seconds(n, water_temp_c=None, rng=None,
                         min_days=FLOAT_MIN_DAYS, max_days=FLOAT_MAX_DAYS):
    """Per-particle FLOATING duration, in seconds. Once it refloats, a body stays
    at the surface for DAYS (forensic: ~2-5 d in temperate water) before gas
    escapes and it re-sinks -- this is the window in which it is wave-driven, so
    it is deliberately days, not the few hours used previously."""
    if rng is None:
        return np.full(n, 0.5 * (min_days + max_days) * 86400.0, dtype=np.float32)
    return (rng.uniform(min_days, max_days, size=n) * 86400.0).astype(np.float32)
