"""
Monte Carlo run of the DrownedBodyDrift prior off Haifa, with:

  * a STATIC probability heatmap drawn on a real coastline basemap (cartopy)
  * an INTERACTIVE, time-animated heatmap over the 0-7 day span (folium)

    python sim_drowned_body.py

By default it runs OFFLINE on constant forcing (no account needed). Set
SOURCE = 'copernicus' (and run `copernicusmarine login` once) to drive it
with real 3D Mediterranean currents.

Outputs go to ./output/ next to this script.
"""

import os
import re
import sys
import logging
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# copernicusmarine downloads via many parallel S3 connections; urllib3's default
# pool holds only 10, so it logs harmless "Connection pool is full" warnings as
# it closes the extras. The download still completes fully -- just quiet them.
logging.getLogger('urllib3.connectionpool').setLevel(logging.ERROR)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from opendrift.readers import reader_constant
from opendrift.readers import reader_global_landmask
from opendrift.readers import reader_netCDF_CF_generic

from core.drowned_drift import (
    DrownedBodyDrift, refloat_time_seconds, surface_time_seconds, refloat_days,
    drown_delay_seconds, body_dynamics, SURFACE_EPS)
from core.search_planner import (
    Agent, CoveragePlanner, make_reference_grid, reference_cell_label)

OUTDIR = os.path.join(HERE, 'output')
os.makedirs(OUTDIR, exist_ok=True)

# Persistent cache for the (large, ~100 MB) Copernicus current/wave subsets.
# Kept SEPARATE from OUTDIR so clearing outputs never wipes the cache, and
# gitignored. Downloads land here and are reused on every later run with the
# same inputs -- so a demo of an already-run case (e.g. Netanya) never has to
# pull data live. See CACHE_ONLY below for a guaranteed-offline demo mode.
CACHE_DIR = os.path.join(HERE, 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# --- run parameters --------------------------------------------------------
SOURCE        = 'copernicus'             # 'offline' (constant) or 'copernicus' (real 3D)
N_PARTICLES   = 10000                 # Monte Carlo ensemble size
LKP_LON       = 34.92                 # last-known position (lon)
LKP_LAT       = 32.83                 # last-known position (lat)
LKP_RADIUS_M  = 300                   # position uncertainty of the LKP (m)

# --- victim (drives sink/refloat timing via body_dynamics) -----------------
BODY_HEIGHT_M = 1.75                  # victim height (m)
BODY_WEIGHT_KG = 75.0                 # victim weight (kg)
WATER_TEMP_C = None                   # sea temp (C) from the form; refloat-timing
                                      # driver (TODO in body_dynamics), stored now

# --- LKP-time (drowning) uncertainty ---------------------------------------
# We rarely know the exact moment of drowning: the victim drifts at the
# surface for some minutes after the last-known position, THEN goes under and
# the sink/drift cycle begins. Each particle draws its own delay (triangular,
# peaking at MODE, capped at MAX), so the start of the underwater drift -- and
# thus the whole forecast -- is itself uncertain.
DROWN_TIME_MAX_MIN  = 30.0            # latest the victim could have gone under
DROWN_TIME_MODE_MIN = 4.0            # most-likely time to drown (peak of the pdf)

# --- when to search (you control this) -------------------------------------
# LKP date/time in UTC: when the victim was last seen / went into the water.
# The drift forecast runs from here for RUN_DURATION. Must fall inside the
# Copernicus anfc window (~8 months back to a few days ahead of today).
SEARCH_TIME   = datetime(2026, 6, 8, 6, 0, 0)
RUN_DURATION  = timedelta(days=7)     # search window
TIME_STEP     = 900                   # 15 min integration step
OUTPUT_STEP   = 3600                  # save a frame every hour
NCFILE        = os.path.join(OUTDIR, 'drowned.nc')

# --- Copernicus data subset (download-once, read-local) --------------------
# Streaming the whole Mediterranean per step is what made the run slow. Instead
# we download a SMALL local NetCDF covering just the search box / time window /
# shallow depth, then read it locally. Re-downloaded every run so you always
# get the latest forecast for the SEARCH_TIME you picked.
CMEMS_USER        = 'yairtheop1@gmail.com'
CMEMS_PASS        = '6dewk7ymVUj*j4h'
MAX_DEPTH_M       = 200.0   # only fetch currents down to here (Haifa shelf is shallow)
SUBSET_MARGIN_DEG = 1.0     # half-width of the box around the LKP (~95 km of drift
                            # room -- bigger for the 7-day window so the cloud
                            # stays inside the downloaded box)
REFRESH_DATA      = False   # True = always re-download; False = reuse a cached subset
                            # with the SAME params (faster, and never overwrites a
                            # file another run might still hold open on Windows)
CACHE_ONLY        = False   # True = OFFLINE DEMO mode: never download -- use only
                            # cached data and raise a clear error if it is missing.
                            # Also settable via env: NAHSHON_CACHE_ONLY=1. Pre-cache
                            # a case once online, then demo it with this on.

# --- stochastic forcing (Gaussian noise) -----------------------------------
# Real drift is noisier than any gridded current field: sub-grid turbulence,
# eddies and unresolved wind gusts scatter a search object. We model that as
# Gaussian noise so the ensemble spreads into a realistic uncertainty cone --
# and, crucially, so a body can wander a little, hit a DIFFERENT current and
# branch off ("went 200 m north, got caught in another stream"). That branching
# only happens with SOURCE='copernicus' (the real, spatially-varying field); a
# constant offline current just blurs, it has no other stream to catch.
# Turn these UP for a wilder, more uncertain fan-out; set to 0 for deterministic.
#   * CURRENT_UNCERTAINTY    -- std-dev of Gaussian noise added to the current
#                               velocity EVERY step (m/s). 0.05 = mild,
#                               0.1-0.2 = clearly chaotic. 0 = off.
#   * HORIZONTAL_DIFFUSIVITY -- eddy-diffusion random walk (m^2/s); cloud grows
#                               ~sqrt(2*K*t). 1 = tight, 10-50 = bodies roam far
#                               enough to catch other streams. 0 = off.
CURRENT_UNCERTAINTY    = 0.10   # m/s
HORIZONTAL_DIFFUSIVITY = 10.0   # m^2/s

# --- surface wind & wave (Stokes) drift ------------------------------------
# WHY WAVES MATTER: a drowned body's horizontal motion is the OCEAN CURRENT plus
# the WAVE STOKES DRIFT (the net mass transport waves carry in their direction of
# travel) plus wind. A SUBMERGED body has no wave drift, but a FLOATING one does
# -- and near shore the wave Stokes transport is frequently STRONGER than the slow
# background current. That is exactly why a current-only model is accurate
# offshore yet wrong for surface drifters (e.g. the Netanya body that went south
# while the current ran north). We pull the Copernicus Med WAVE product
# (MEDSEA_ANALYSISFORECAST_WAV_006_017) and form an EFFECTIVE drift that lets the
# waves dominate.
USE_STOKES        = True    # add Copernicus wave Stokes drift (afloat phases)

# EFFECTIVE-DRIFT WEIGHTING (THE ONE PLACE TO TUNE):
#     effective = DRIFT_WEIGHT_CURRENT * current  +  DRIFT_WEIGHT_STOKES * stokes
# Defaults make waves the dominant term, per the SAR requirement. Set both to 1.0
# for the physically-neutral sum (current + full Stokes). These weights flow into
# DrownedBodyDrift (current scaled via current_drift_factor, Stokes via its
# stokes_drift() override) and therefore into EVERY downstream product -- the
# Monte-Carlo cloud, the heatmap, the search plan and the diffusion -- because all
# of them derive from the single weighted o.run().
DRIFT_WEIGHT_CURRENT = 0.3   # multiplier on ocean current
DRIFT_WEIGHT_STOKES  = 0.7   # multiplier on wave Stokes drift
# Apply the weighting only to the AFLOAT phases (a submerged body must still
# follow the FULL current). True = physically correct; False = weight globally.
WAVE_WEIGHT_AFLOAT_ONLY = True

# NEARSHORE (surf-zone) LONGSHORE CURRENT. The 4.2 km Copernicus grid cannot
# resolve the wave-driven longshore current within a few km of the beach. Along
# the Israeli coast that nearshore drift is NORTHWARD in the annual mean but
# reverses SOUTHWARD in spring/summer (Israeli longshore sediment-transport
# studies). We add it as a signed alongshore current that decays offshore and
# acts on the AFLOAT phases (see DrownedBodyDrift). Offshore cases are untouched
# (the term -> 0 away from the beach), so northward open-sea drift stays northward.
#   NEARSHORE_LONGSHORE_V -- m/s alongshore near the coast; + north / - south.
#                            0 = off; spring along this coast -> negative (south).
#   NEARSHORE_SCALE_KM    -- offshore decay distance (surf-zone width).
# SEASONAL: along the Israeli coast this nearshore drift is SOUTHWARD in spring/
# summer (negative) and northward in winter. The default is the spring value used
# for the Netanya case; it decays offshore so open-sea drift is unaffected. For a
# winter case set it >= 0. Calibrated against the Netanya->Herzliya recovery.
NEARSHORE_LONGSHORE_V = -0.07
NEARSHORE_SCALE_KM    = 6.0
# GEOGRAPHIC EXTENT of the southward nearshore current. It is a central-shelf
# feature (open Netanya / Herzliya coast). Haifa Bay to the north is a curved,
# sheltered embayment that keeps the general NORTHWARD flow, so we only apply the
# southward reversal inside this latitude band and taper it to zero before Haifa.
# A run anywhere outside the band (e.g. Haifa ~32.8 N) gets no southward push.
# Set NEARSHORE_LAT_MIN/MAX = None to apply it everywhere (old global behaviour).
NEARSHORE_LAT_MIN   = 32.05   # deg N (south of Herzliya)
NEARSHORE_LAT_MAX   = 32.42   # deg N (just north of Netanya)
NEARSHORE_LAT_TAPER = 0.15    # deg roll-off; ~0 by ~32.6 N, fully off at Haifa

# Direct windage needs a separate wind field (ERA5 for past dates / GFS for recent
# ones); WIND_DRIFT_FACTOR only takes effect once such a wind reader is added.
WIND_DRIFT_FACTOR = 0.0     # windage coefficient; needs a wind reader to act

# --- afloat / drowned probability ------------------------------------------
# At each of these times (hours since the LKP) we report the probability that
# the body is AFLOAT (at the surface -> visible, wind-driftable, spottable) vs
# SUBMERGED (underwater), measured as the fraction of the ensemble in each
# state. Useful for deciding when a surface/air search is worthwhile.
PROB_QUERY_HOURS = [1, 6, 12, 24, 48, 72, 120, 168]   # spans the 7-day window

# --- heatmap grid ----------------------------------------------------------
# The static map bins the final particle cloud into square cells and colours
# each cell by HOW MANY bodies land in it. Smaller cell -> finer grid.
HEATMAP_GRID_M = 300    # grid cell size in metres
HEATMAP_SMOOTH = 0      # gaussian smoothing in cells (0 = crisp grid boxes)

# --- React drift-app bridge ------------------------------------------------
# The run writes a JSON the drift-app map loads (one heatmap frame per output
# hour + afloat/submerged probabilities + the LKP + the search plan).
#
# IMPORTANT: this MUST NOT live inside drift-app/ (e.g. public/). Vite's dev
# server watches that tree and FULL-RELOADS the page whenever a file there
# changes -- so writing the result mid-run reloaded the app and wiped React
# state right as the run finished. We write it to OUTDIR (outside Vite's watch)
# and the frontend reads it only through the /api/drift_data endpoint.
APP_JSON          = os.path.join(OUTDIR, 'drift_data.json')
APP_MAX_PARTICLES = 1500   # particles per frame in the JSON (keeps the file light)

# --- coordinated search plan (core/search_planner.py) ----------------------
# After the drift produces a probability heatmap we route a heterogeneous
# rescue team over it (maximise probability cleared = minimise time-to-detect).
# The plan is drawn on the heatmap PNG and exported into drift_data.json so the
# React Search-Plan screen renders the REAL paths instead of mock ones.
PLAN_HOUR        = 24      # which forecast hour's heatmap to plan the search on
# --- algorithm grid: FINE cells over the high-probability CORE -------------
# We plan on a small, fine grid focused on the dense core of the cloud (not the
# whole multi-km drift), so routes are realistic boat tracks instead of coarse
# jumps. A literal 3 m grid over a big box is millions of cells, so the cell
# auto-grows if the core would exceed PLAN_MAX_CELLS per side.
PLAN_GRID_M      = 3       # target algorithm cell size (m)
PLAN_MAX_CELLS   = 600     # cap on cells/side (cell grows past 3 m if needed)
PLAN_CORE_FRAC   = 0.95    # plan over the box holding this much probability mass
# --- rescue fleet: real vehicle speeds drive time AND route ----------------
PLAN_TEAMS       = 3       # number of rescue craft
PLAN_SONAR_M     = 80      # detection radius per craft (sonar/lookout swath, m)
PLAN_BOAT_MPS    = 7.0     # search boat   (~13 kn)
PLAN_JETSKI_MPS  = 14.0    # jet-ski       (~27 kn)
# fleet cycled across the teams: (label, speed m/s). Faster craft cover more.
PLAN_FLEET       = [('Boat', PLAN_BOAT_MPS), ('Jetski', PLAN_JETSKI_MPS)]
# user-placed vehicle types (from the Search Plan map) -> real speed (m/s).
PLAN_TYPE_SPEED  = {'jetski': PLAN_JETSKI_MPS, 'boat': PLAN_BOAT_MPS}
# --- convergence (replaces a fixed step horizon) ---------------------------
PLAN_TICK_SEC     = 20.0   # wall-clock seconds per planning tick
PLAN_COVERAGE     = 0.95   # stop once this fraction of probability is cleared ...
PLAN_MAX_TIME_MIN = 90.0   # ... but never search longer than this
PLAN_DISPERSION   = 0.25   # 0..1: how strongly teams keep apart to cover more
                           # ground (0 = pure probability-greedy)
# --- reference ("comms") grid drawn on the map -----------------------------
# Coarse labelled grid (rows A,B,C.. from north, cols 1,2,3.. from west) that
# auto-fits the search area, giving teams a shared "B-4" language. Bigger cells
# than the algorithm grid.
REFGRID_CELL_M    = 500    # target reference-cell size (m); auto-fits the area
# team colours match the React SearchPlan TEAM_COLORS palette
PLAN_COLORS      = ['#3b82f6', '#f97316', '#22c55e', '#a855f7', '#ec4899']

# --- incident input (frontend -> simulation) -------------------------------
# If this JSON exists it OVERRIDES the LKP / victim / time / team settings
# above, so the lon-lat (and victim profile) entered in the React Incident
# Report drive the actual simulation. The Incident Report screen can export it
# with its "Download incident.json" button; drop it here (or in public/).
INCIDENT_JSON    = os.path.join(HERE, '..', 'drift-app', 'public', 'incident.json')

rng = np.random.default_rng(42)       # reproducible ensemble

# environment overrides (used by the API / a quick demo without Copernicus):
#   NAHSHON_SOURCE=offline    -> constant forcing, no download
#   NAHSHON_PARTICLES=1500    -> smaller, faster ensemble
#   NAHSHON_CACHE_ONLY=1      -> offline demo: use cached data only, never download
if os.environ.get('NAHSHON_SOURCE'):
    SOURCE = os.environ['NAHSHON_SOURCE']
if os.environ.get('NAHSHON_PARTICLES'):
    N_PARTICLES = int(os.environ['NAHSHON_PARTICLES'])
if os.environ.get('NAHSHON_CACHE_ONLY') not in (None, '', '0', 'false', 'False'):
    CACHE_ONLY = True


def _cache_lookup(fname, kind):
    """Find a cached subset `fname` (in CACHE_DIR, or the legacy OUTDIR) before
    downloading. Returns the path if a usable cache exists, else None.

      * REFRESH_DATA=True forces a re-download (ignored when CACHE_ONLY).
      * CACHE_ONLY=True is offline mode: if nothing is cached, raise a clear
        error instead of hitting the network.
    """
    for d in (CACHE_DIR, OUTDIR):
        p = os.path.join(d, fname)
        if os.path.exists(p):
            if REFRESH_DATA and not CACHE_ONLY:
                return None                      # caller will re-download
            print(f"  using cached {kind}: {fname}")
            return p
    if CACHE_ONLY:
        raise RuntimeError(
            f"CACHE_ONLY/offline is on but no cached {kind} is available.\n"
            f"  missing file: {fname}\n"
            f"  Fix: run this case once ONLINE (CACHE_ONLY off) to populate "
            f"{CACHE_DIR}, then demo it offline.")
    return None


def apply_incident(inc):
    """Override the LKP / victim / search-time / team globals from an incident
    dict (the React Incident Report form), so the coordinates and victim
    profile entered in the frontend ARE the inputs of this simulation.

    Recognised keys (all optional): lat, lng/lon, victimHeight (cm),
    victimWeight (kg), date (YYYY-MM-DD), time/timeFrom (HH:MM), waterTemp,
    durationHours, teams, sonarRadius (m), speed (cells/tick), horizon,
    planHour, nParticles."""
    if not inc:
        return
    global LKP_LAT, LKP_LON, BODY_HEIGHT_M, BODY_WEIGHT_KG, SEARCH_TIME
    global RUN_DURATION, N_PARTICLES, WATER_TEMP_C
    global PLAN_TEAMS, PLAN_SONAR_M, PLAN_HOUR, PLAN_COVERAGE

    if inc.get('lat') not in (None, ''):
        LKP_LAT = float(inc['lat'])
    lon = inc.get('lng', inc.get('lon'))
    if lon not in (None, ''):
        LKP_LON = float(lon)
    if inc.get('victimHeight') not in (None, ''):
        BODY_HEIGHT_M = float(inc['victimHeight']) / 100.0      # cm -> m
    if inc.get('victimWeight') not in (None, ''):
        BODY_WEIGHT_KG = float(inc['victimWeight'])
    if inc.get('waterTemp') not in (None, ''):
        WATER_TEMP_C = float(inc['waterTemp'])
    if inc.get('date'):
        hhmm = (inc.get('time') or inc.get('timeFrom') or '00:00')
        SEARCH_TIME = datetime.strptime(f"{inc['date']} {hhmm}", '%Y-%m-%d %H:%M')
    if inc.get('durationHours') not in (None, ''):
        RUN_DURATION = timedelta(hours=float(inc['durationHours']))
    for key, g in (('teams', 'PLAN_TEAMS'), ('sonarRadius', 'PLAN_SONAR_M'),
                   ('planHour', 'PLAN_HOUR'), ('nParticles', 'N_PARTICLES')):
        if inc.get(key) not in (None, ''):
            globals()[g] = type(globals()[g])(inc[key])
    print(f"  incident -> LKP ({LKP_LAT:.5f}, {LKP_LON:.5f}), "
          f"body {BODY_HEIGHT_M:.2f} m / {BODY_WEIGHT_KG:.0f} kg, "
          f"{PLAN_TEAMS} teams, T+{int(RUN_DURATION.total_seconds()//3600)}h window")


def load_incident(path=INCIDENT_JSON):
    """Apply incident.json from disk if it exists (CLI / manual workflow)."""
    import json
    if not os.path.exists(path):
        return
    with open(path, encoding='utf-8') as fh:
        apply_incident(json.load(fh))


def download_currents_subset():
    """Download a SMALL local NetCDF of 3D currents for the search box / time
    window / depth cap, so the run reads locally instead of streaming the whole
    Mediterranean per step. Returns the local file path.

    The file is named after the subset PARAMETERS (LKP / time / depth / box), so
    two runs with the same inputs reuse one cache and -- crucially on Windows --
    a new run never tries to overwrite a file an earlier run may still hold open
    (that lock was what made the live run hang). Set REFRESH_DATA=True to force a
    fresh download for the current parameters.
    """
    import copernicusmarine
    t0 = SEARCH_TIME - timedelta(hours=2)                  # small read-buffer
    t1 = SEARCH_TIME + RUN_DURATION + timedelta(hours=2)

    key = (f"{LKP_LAT:.3f}_{LKP_LON:.3f}_{SEARCH_TIME:%Y%m%d%H}_"
           f"{int(RUN_DURATION.total_seconds()//3600)}h_"
           f"{int(MAX_DEPTH_M)}m_{SUBSET_MARGIN_DEG}")
    fname = f"currents_{key}.nc"
    hit = _cache_lookup(fname, 'currents')
    if hit:
        return hit

    print(f"  currents subset: {t0:%Y-%m-%d %H:%M}..{t1:%Y-%m-%d %H:%M} UTC, "
          f"0-{MAX_DEPTH_M:.0f} m, +/-{SUBSET_MARGIN_DEG} deg around LKP -> {fname}")
    copernicusmarine.subset(
        dataset_id='cmems_mod_med_phy-cur_anfc_4.2km-3D_PT1H-m',
        variables=['uo', 'vo'],
        minimum_longitude=LKP_LON - SUBSET_MARGIN_DEG,
        maximum_longitude=LKP_LON + SUBSET_MARGIN_DEG,
        minimum_latitude=LKP_LAT - SUBSET_MARGIN_DEG,
        maximum_latitude=LKP_LAT + SUBSET_MARGIN_DEG,
        minimum_depth=0.0, maximum_depth=MAX_DEPTH_M,
        start_datetime=t0.strftime('%Y-%m-%dT%H:%M:%S'),
        end_datetime=t1.strftime('%Y-%m-%dT%H:%M:%S'),
        output_filename=fname, output_directory=CACHE_DIR,
        username=CMEMS_USER, password=CMEMS_PASS,
        overwrite=True, disable_progress_bar=True)
    return os.path.join(CACHE_DIR, fname)


# Med WAVE product variables relevant to object transport:
#   VSDX/VSDY -> Stokes drift east/north (the actual wave mass transport),
#   VHM0      -> significant wave height (energy/sea state),
#   VMDR      -> mean wave direction (deg FROM, oceanographic convention),
#   VPED      -> wave principal/peak direction (deg FROM),
#   VTPK      -> wave period at the spectral peak (s, for the Stokes estimate).
WAVE_VARIABLES = ['VSDX', 'VSDY', 'VHM0', 'VMDR', 'VPED', 'VTPK']


def estimate_stokes_from_waves(hs, tp, dir_from_deg):
    """FALLBACK: estimate surface Stokes drift (east, north) from significant
    wave height `hs` (m), peak period `tp` (s) and mean wave direction
    `dir_from_deg` (deg the waves come FROM) when the model's VSDX/VSDY are
    missing. Deep-water surface Stokes for the significant wave:
        us ~= omega^3 * (Hs/2)^2 / g ,   omega = 2*pi/Tp
    scaled by ~0.5 because a real spectrum drifts less than a monochromatic one.
    Direction is the way the waves TRAVEL (= from-direction + 180)."""
    g = 9.81
    tp = np.where(np.isfinite(tp) & (tp > 1.0), tp, 6.0)   # guard bad periods
    omega = 2.0 * np.pi / tp
    us = 0.5 * omega**3 * (hs / 2.0)**2 / g                # magnitude (m/s)
    to_rad = np.radians((dir_from_deg + 180.0) % 360.0)    # travel direction
    return us * np.sin(to_rad), us * np.cos(to_rad)        # east, north


def download_waves_subset():
    """Download the Copernicus Med WAVE product (MEDSEA_ANALYSISFORECAST_WAV_006_
    017) for the same box/time as the currents: Stokes drift VSDX/VSDY plus wave
    height/direction/period (for diagnostics and the Stokes fallback). Stokes
    drift is the wave-driven mass transport that pushes a FLOATING body downwind;
    OpenDrift applies it with a depth decay, so only the afloat phases feel it.
    Same param-named cache as the currents (never overwrites a locked file).

    If the file lacks VSDX/VSDY (older product / partial coverage) we DERIVE them
    from wave height + period + direction via estimate_stokes_from_waves and write
    them in, so OpenDrift always has a Stokes field to read."""
    import copernicusmarine
    t0 = SEARCH_TIME - timedelta(hours=2)
    t1 = SEARCH_TIME + RUN_DURATION + timedelta(hours=2)
    key = (f"{LKP_LAT:.3f}_{LKP_LON:.3f}_{SEARCH_TIME:%Y%m%d%H}_"
           f"{int(RUN_DURATION.total_seconds()//3600)}h_{SUBSET_MARGIN_DEG}")
    fname = f"waves_{key}.nc"
    hit = _cache_lookup(fname, 'waves')
    if hit:
        return hit
    path = os.path.join(CACHE_DIR, fname)
    print(f"  waves subset (Stokes + height/dir/period) -> {fname}")
    sub = dict(
        dataset_id='cmems_mod_med_wav_anfc_4.2km_PT1H-i',
        minimum_longitude=LKP_LON - SUBSET_MARGIN_DEG,
        maximum_longitude=LKP_LON + SUBSET_MARGIN_DEG,
        minimum_latitude=LKP_LAT - SUBSET_MARGIN_DEG,
        maximum_latitude=LKP_LAT + SUBSET_MARGIN_DEG,
        start_datetime=t0.strftime('%Y-%m-%dT%H:%M:%S'),
        end_datetime=t1.strftime('%Y-%m-%dT%H:%M:%S'),
        output_filename=fname, output_directory=CACHE_DIR,
        username=CMEMS_USER, password=CMEMS_PASS,
        overwrite=True, disable_progress_bar=True)
    try:
        copernicusmarine.subset(variables=WAVE_VARIABLES, **sub)
    except Exception as exc:
        # an auxiliary variable name may be off for this product version -> keep
        # the essential Stokes drift so the model still gets the wave push.
        print(f'  [full wave var set failed ({exc}); fetching VSDX/VSDY only]')
        copernicusmarine.subset(variables=['VSDX', 'VSDY'], **sub)

    # Fallback: synthesise Stokes from Hs/Tp/direction if the product lacks it.
    ds = xr.open_dataset(path)
    if ('VSDX' not in ds or 'VSDY' not in ds) and 'VHM0' in ds:
        print('  [VSDX/VSDY missing -> estimating Stokes from Hs/Tp/direction]')
        hs = ds['VHM0']
        tp = ds['VTPK'] if 'VTPK' in ds else xr.full_like(hs, 6.0)
        dr = ds['VMDR'] if 'VMDR' in ds else (ds['VPED'] if 'VPED' in ds
                                              else xr.zeros_like(hs))
        sx, sy = estimate_stokes_from_waves(hs, tp, dr)
        sx.attrs['standard_name'] = 'sea_surface_wave_stokes_drift_x_velocity'
        sy.attrs['standard_name'] = 'sea_surface_wave_stokes_drift_y_velocity'
        ds = ds.assign(VSDX=sx, VSDY=sy)
        ds.to_netcdf(path)
    ds.close()
    return path


def download_bathymetry_subset():
    """Download (and CACHE) the static sea-floor depth for the search box, so
    bodies rest on the real seabed without a LIVE network read on every run --
    which is what made each app run stall ('loading currents'). The static
    product has no time axis, so it is one tiny 2D file per box, reused forever.

    Returns the local path, or None -> the caller then uses the flat-seabed
    fallback (no network read).

    NOTE: in CACHE_ONLY (offline-demo) mode we deliberately return None and keep
    the flat-seabed fallback, so the offline demo's result is IDENTICAL to before
    this caching was added (this stays a pure speed change, not a physics change).
    The real-seabed path is the LIVE mode, which already read this static field --
    only now from a tiny local cache instead of a live network stream each run."""
    if CACHE_ONLY:
        return None                 # offline demo -> flat seabed, unchanged result
    fname = f"bathy_{LKP_LAT:.3f}_{LKP_LON:.3f}_{SUBSET_MARGIN_DEG}.nc"
    for d in (CACHE_DIR, OUTDIR):
        p = os.path.join(d, fname)
        if os.path.exists(p) and not REFRESH_DATA:
            print(f"  using cached bathymetry: {fname}")
            return p
    import copernicusmarine
    path = os.path.join(CACHE_DIR, fname)
    print(f"  bathymetry subset (static sea-floor depth) -> {fname}")
    copernicusmarine.subset(
        dataset_id='cmems_mod_med_phy_anfc_4.2km_static',
        variables=['deptho'],
        minimum_longitude=LKP_LON - SUBSET_MARGIN_DEG,
        maximum_longitude=LKP_LON + SUBSET_MARGIN_DEG,
        minimum_latitude=LKP_LAT - SUBSET_MARGIN_DEG,
        maximum_latitude=LKP_LAT + SUBSET_MARGIN_DEG,
        output_filename=fname, output_directory=CACHE_DIR,
        username=CMEMS_USER, password=CMEMS_PASS,
        overwrite=True, disable_progress_bar=True)
    return path


def _build_coast_distance():
    """Interpolator (lat, lon) -> distance to the nearest coast in KM over the
    search box, from the global landmask (offline, no download) + a Euclidean
    distance transform. Feeds the nearshore longshore-current correction."""
    from scipy.ndimage import distance_transform_edt
    from scipy.interpolate import RegularGridInterpolator
    from global_land_mask import globe
    n = 260
    lons = np.linspace(LKP_LON - SUBSET_MARGIN_DEG, LKP_LON + SUBSET_MARGIN_DEG, n)
    lats = np.linspace(LKP_LAT - SUBSET_MARGIN_DEG, LKP_LAT + SUBSET_MARGIN_DEG, n)
    LON, LAT = np.meshgrid(lons, lats)                 # (n_lat, n_lon)
    water = ~globe.is_land(LAT, LON)
    dy = (lats[1] - lats[0]) * 110.57                  # km per lat cell
    dx = (lons[1] - lons[0]) * 111.32 * np.cos(np.radians(LKP_LAT))
    dist_km = distance_transform_edt(water, sampling=[dy, dx])
    interp = RegularGridInterpolator((lats, lons), dist_km,
                                     bounds_error=False, fill_value=999.0)

    def fn(lat, lon):
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        return interp(np.column_stack([lat.ravel(), lon.ravel()])).reshape(lat.shape)
    return fn


def build_model():
    o = DrownedBodyDrift(loglevel=20)

    # Height & weight -> sink/rise speeds (refloat base is applied per-particle
    # in seed()). Leaner bodies sink faster and refloat slower; heavier-set
    # bodies are near-neutral and bob back sooner.
    dyn = body_dynamics(BODY_HEIGHT_M, BODY_WEIGHT_KG)
    o.sink_speed = dyn['sink']
    o.rise_speed = dyn['rise']
    rf_days = refloat_days(WATER_TEMP_C, BODY_HEIGHT_M, BODY_WEIGHT_KG)
    print(f"  body {BODY_HEIGHT_M:.2f} m / {BODY_WEIGHT_KG:.0f} kg "
          f"(BMI {dyn['bmi']:.1f}, density {dyn['density']:.0f} kg/m^3) -> "
          f"sink {dyn['sink']:.3f} m/s, rise {dyn['rise']:.3f} m/s")
    print(f"  water {WATER_TEMP_C if WATER_TEMP_C is not None else 'default ~19'} C "
          f"-> refloat ~{rf_days:.1f} days, then floats ~2-5 days "
          f"(wave-driven at the surface)")

    o.add_reader(reader_global_landmask.Reader())

    if SOURCE == 'copernicus':
        # === REAL 3D DATA (downloaded once, read locally) ===================
        # Requires: pip install copernicusmarine. We subset a small local file
        # for the search box/time/depth (see download_currents_subset) and read
        # it with the generic NetCDF reader -- uo/vo carry standard_names, so
        # they are auto-rotated to x/y_sea_water_velocity and interpolated to
        # each particle's depth.
        cur_path = download_currents_subset()
        currents = reader_netCDF_CF_generic.Reader(cur_path, name='CMEMS-3D-local')
        o.add_reader(currents)
        o._cur_path = cur_path          # kept for the drift diagnostics
        o._wav_path = None

        # Bathymetry for sea_floor_depth_below_sea_level so bodies rest on the
        # real seabed instead of the flat-100 m fallback. The static field is now
        # CACHED to a tiny local file (download_bathymetry_subset) and read with
        # the generic NetCDF reader -- no per-run live network read. Offline with
        # no cached bathy -> flat-seabed fallback (keeps the run network-free).
        try:
            bathy_path = download_bathymetry_subset()
            if bathy_path:
                bathy = reader_netCDF_CF_generic.Reader(bathy_path, name='CMEMS-bathy')
                o.add_reader(bathy)
            else:
                print('  [no cached bathymetry offline -> flat-seabed fallback]')
        except Exception as exc:
            print('  [bathymetry unavailable, using flat-seabed fallback]:', exc)

        # Surface WAVE Stokes drift -> the wave/wind-sea push on the AFLOAT
        # phases (OpenDrift decays it with depth, so the submerged phases barely
        # feel it). VSDX/VSDY carry Stokes-drift standard_names, auto-mapped.
        if USE_STOKES:
            try:
                wav_path = download_waves_subset()
                waves = reader_netCDF_CF_generic.Reader(wav_path, name='CMEMS-waves')
                o.add_reader(waves)
                o._wav_path = wav_path
            except Exception as exc:
                print('  [wave Stokes unavailable, continuing without it]:', exc)
    else:
        # --- OFFLINE constant forcing --------------------------------------
        # Gentle OFFSHORE drift so the cluster stays in open water long enough
        # for the sink/refloat cycle to be visible in the demo.
        const = reader_constant.Reader({
            'x_sea_water_velocity': -0.05,   # m/s westward (away from coast)
            'y_sea_water_velocity':  0.02,   # m/s slightly northward
            'sea_floor_depth_below_sea_level': 60.0,
        })
        o.add_reader(const)

    # 'previous' (not 'stranding'): OpenDrift bounces EVERYTHING off the coast.
    # The model itself then strands only the AFLOAT phases at the surf line
    # (DrownedBodyDrift.update step 4) -- a submerged body must drift past the
    # shore rather than beach at the entry point. Afloat strandings still get a
    # 'stranded' status, so the heatmap's shore layer is populated as before.
    o.set_config('general:coastline_action', 'previous')
    o.set_config('general:seafloor_action', 'lift_to_seafloor')
    o.set_config('drift:vertical_advection', True)
    o.set_config('drift:vertical_mixing', False)
    o.set_config('drift:stokes_drift', USE_STOKES)   # wave push on afloat phases

    # Gaussian noise -> realistic ensemble spread (see top of file to tune).
    o.set_config('drift:current_uncertainty', CURRENT_UNCERTAINTY)
    o.set_config('drift:horizontal_diffusivity', HORIZONTAL_DIFFUSIVITY)

    # EFFECTIVE-DRIFT WEIGHTING: make the waves the dominant term on the afloat
    # phases. Only enabled when the wave Stokes field actually loaded, so a
    # current-only run (offline, or no wave product) keeps the FULL current and
    # behaves exactly as before -> backward compatible (requirement 10).
    if getattr(o, '_wav_path', None):
        o.wave_weighting = True
        o.weight_current = DRIFT_WEIGHT_CURRENT
        o.weight_stokes = DRIFT_WEIGHT_STOKES
        o.weight_afloat_only = WAVE_WEIGHT_AFLOAT_ONLY
        print(f"  effective drift = {DRIFT_WEIGHT_CURRENT:.2f}*current + "
              f"{DRIFT_WEIGHT_STOKES:.2f}*stokes "
              f"({'afloat phases only' if WAVE_WEIGHT_AFLOAT_ONLY else 'all phases'})")
    else:
        o.wave_weighting = False     # current-only -> unchanged behaviour

    # Coast-distance field: needed both for the nearshore longshore current AND
    # for the phase-aware afloat beaching, so build it once and always attach it.
    o.coast_distance_fn = _build_coast_distance()

    # Nearshore surf-zone longshore current (afloat objects near the beach).
    if NEARSHORE_LONGSHORE_V:
        o.nearshore_longshore_v = NEARSHORE_LONGSHORE_V
        o.nearshore_scale_km = NEARSHORE_SCALE_KM
        o.nearshore_lat_min = NEARSHORE_LAT_MIN
        o.nearshore_lat_max = NEARSHORE_LAT_MAX
        o.nearshore_lat_taper = NEARSHORE_LAT_TAPER
        band = ('everywhere' if NEARSHORE_LAT_MIN is None
                else f"{NEARSHORE_LAT_MIN}-{NEARSHORE_LAT_MAX} N (off toward Haifa)")
        print(f"  nearshore longshore current: {NEARSHORE_LONGSHORE_V:+.2f} m/s "
              f"({'south' if NEARSHORE_LONGSHORE_V < 0 else 'north'}), "
              f"decay {NEARSHORE_SCALE_KM} km offshore, band {band}")
    return o


def seed(o):
    start = SEARCH_TIME                      # you set this at the top of the file
    # per-particle uncertain LKP->drown delay (0..30 min) and body-driven timers
    drown   = drown_delay_seconds(N_PARTICLES, DROWN_TIME_MAX_MIN,
                                  DROWN_TIME_MODE_MIN, rng=rng)
    # refloat is TEMPERATURE-driven (WATER_TEMP_C); the body then FLOATS for days.
    refloat = refloat_time_seconds(N_PARTICLES, BODY_HEIGHT_M, BODY_WEIGHT_KG,
                                   rng=rng, water_temp_c=WATER_TEMP_C)
    surface = surface_time_seconds(N_PARTICLES, water_temp_c=WATER_TEMP_C, rng=rng)
    o.seed_elements(
        lon=LKP_LON, lat=LKP_LAT, radius=LKP_RADIUS_M,
        number=N_PARTICLES, time=start, z=0,
        phase=1, phase_start_age=0,          # start in the DROWNING phase
        drown_delay=drown, refloat_time=refloat, surface_time=surface,
        wind_drift_factor=WIND_DRIFT_FACTOR,  # windage (acts once a wind reader is added)
    )
    return o


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def _count_grid(lon, lat, extent, cell_m, sigma=0):
    """Particle COUNT per grid cell on a regular lon/lat grid whose cells are
    ~`cell_m` metres square. Returns the count grid (axis0=lat, axis1=lon) and
    the cell EDGES (xe, ye) so pcolormesh can draw real boxes."""
    from scipy.ndimage import gaussian_filter
    midlat = 0.5 * (extent[2] + extent[3])
    dlon = cell_m / (111_320.0 * np.cos(np.radians(midlat)))   # m -> deg lon
    dlat = cell_m / 110_540.0                                  # m -> deg lat
    nx = max(1, int(round((extent[1] - extent[0]) / dlon)))
    ny = max(1, int(round((extent[3] - extent[2]) / dlat)))
    H, xe, ye = np.histogram2d(
        lon, lat, bins=[nx, ny],
        range=[[extent[0], extent[1]], [extent[2], extent[3]]])
    H = H.T                                         # .T -> axis0 = lat
    if sigma:
        H = gaussian_filter(H, sigma=sigma)
    return H, xe, ye


def make_heatmap_map(o):
    """Static probability heatmap on a real coastline basemap (cartopy)."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    groups = [o.elements, o.elements_deactivated]
    lon = np.concatenate([g.lon for g in groups if g.lon.size])
    lat = np.concatenate([g.lat for g in groups if g.lat.size])
    m = np.isfinite(lon) & np.isfinite(lat)
    lon, lat = lon[m], lat[m]

    pad = 0.02
    extent = [min(lon.min(), LKP_LON) - pad, max(lon.max(), LKP_LON) + pad,
              min(lat.min(), LKP_LAT) - pad, max(lat.max(), LKP_LAT) + pad]
    # always reach the Haifa coastline (~35.0 E) so the land shows on the map
    extent[1] = max(extent[1], 35.05)
    counts, xe, ye = _count_grid(lon, lat, extent, HEATMAP_GRID_M,
                                 sigma=HEATMAP_SMOOTH)
    # colour each cell by particle count; leave empty cells transparent
    Hm = np.ma.masked_where(counts < 1, counts)

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(9, 8))
    ax = plt.axes(projection=proj)
    ax.set_extent(extent, crs=proj)
    ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#e8e2d0', zorder=1)
    ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor='#dfeefc', zorder=0)
    ax.coastlines(resolution='10m', linewidth=0.8, zorder=2)

    # shading='flat' with edge arrays draws one solid box per grid cell
    mesh = ax.pcolormesh(xe, ye, Hm, transform=proj, cmap='inferno',
                         alpha=0.8, zorder=3, shading='flat')
    ax.plot(LKP_LON, LKP_LAT, marker='*', color='cyan', ms=18,
            mec='black', transform=proj, zorder=4, label='LKP')
    gl = ax.gridlines(draw_labels=True, alpha=0.3)
    gl.top_labels = gl.right_labels = False
    ax.legend(loc='upper right')
    ax.set_title(f'Drowned-body density after {RUN_DURATION} '
                 f'({len(lon)} particles, {HEATMAP_GRID_M} m grid, source={SOURCE})')
    fig.colorbar(mesh, ax=ax,
                 label=f'bodies per {HEATMAP_GRID_M} m cell', shrink=0.8)
    out = os.path.join(OUTDIR, 'drowned_heatmap_map.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print('  static map ->', out)


def state_probabilities(ncfile):
    """Per-output-frame probability that the body is AFLOAT vs SUBMERGED vs
    STRANDED, from the fraction of the ensemble in each state. Returns
    (hours, p_afloat, p_submerged, p_stranded), each a 1D array over time."""
    ds = xr.open_dataset(ncfile)
    z = ds['z'].values                          # (trajectory, time)
    times = ds['time'].values
    ds.close()
    ntraj = z.shape[0]
    hours = (times - times[0]) / np.timedelta64(1, 'h')

    finite = np.isfinite(z)                     # still-active particles
    afloat = finite & (z >= -SURFACE_EPS)       # at/near the surface
    submerged = finite & (z < -SURFACE_EPS)     # underwater
    p_afloat = afloat.sum(axis=0) / ntraj
    p_submerged = submerged.sum(axis=0) / ntraj
    p_stranded = 1.0 - finite.sum(axis=0) / ntraj   # deactivated (beached)
    return hours, p_afloat, p_submerged, p_stranded


def make_state_probability(ncfile, query_hours=PROB_QUERY_HOURS):
    """Report P(afloat) vs P(submerged) at the requested times and save a
    time-series plot of how the body's state probability evolves."""
    hours, p_afloat, p_sub, p_strand = state_probabilities(ncfile)

    print('  afloat / submerged probability:')
    for qh in query_hours:
        if qh > hours[-1] + 1e-6:
            continue                            # beyond the simulated window
        k = int(np.argmin(np.abs(hours - qh)))
        print(f'    t={hours[k]:5.1f} h  afloat={p_afloat[k]*100:5.1f}%  '
              f'submerged={p_sub[k]*100:5.1f}%  stranded={p_strand[k]*100:5.1f}%')

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(hours, p_afloat * 100, color='#2f80ed', lw=2,
            label='afloat (at surface)')
    ax.plot(hours, p_sub * 100, color='#c0392b', lw=2,
            label='submerged (underwater)')
    if p_strand.max() > 0.005:
        ax.plot(hours, p_strand * 100, color='#7f8c8d', lw=1.5, ls='--',
                label='stranded (beached)')
    for qh in query_hours:
        if qh <= hours[-1] + 1e-6:
            ax.axvline(qh, color='k', alpha=0.12, lw=1)
    ax.set_xlabel('hours since LKP')
    ax.set_ylabel('probability (%)')
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_title(f'Body-state probability over time '
                 f'({N_PARTICLES} particles, source={SOURCE})')
    out = os.path.join(OUTDIR, 'drowned_vs_afloat.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print('  state-probability plot ->', out)


def export_app_json(ncfile, out_path=APP_JSON, max_particles=APP_MAX_PARTICLES,
                    intensity=0.45, search_plan=None):
    """Write the simulation result as JSON the React drift-app map loads:
    one heatmap frame per output hour ([lat, lon, intensity] points) plus the
    afloat/submerged/stranded probability of each frame, the LKP, and (if
    provided) the coordinated search plan from core.search_planner."""
    import json

    ds = xr.open_dataset(ncfile)
    lon = ds['lon'].values            # (trajectory, time)
    lat = ds['lat'].values
    status = ds['status'].values      # 0 = active, 1 = stranded (beached on shore)
    times = ds['time'].values
    ds.close()
    ntraj, ntime = lon.shape

    step = max(1, ntraj // max_particles)
    sel = np.arange(0, ntraj, step)
    lon_s, lat_s, status_s = lon[sel], lat[sel], status[sel]

    _, p_afloat, p_sub, p_strand = state_probabilities(ncfile)

    # SHORE landings: many real cases end with the body washed ashore. A stranded
    # particle is deactivated (NaN) afterwards, so it would vanish from the heat-
    # map -- we capture WHERE and WHEN each one beached and carry it forward, so
    # the shore hot-spots persist and accumulate over time.
    stranded_ever = (status_s == 1)
    has_strand = stranded_ever.any(axis=1)
    strand_step = np.where(has_strand, np.argmax(stranded_ever, axis=1), -1)
    si = np.where(has_strand)[0]
    shore_la = [round(float(lat_s[i, strand_step[i]]), 5) for i in si]
    shore_lo = [round(float(lon_s[i, strand_step[i]]), 5) for i in si]
    shore_step = np.array([int(strand_step[i]) for i in si])

    # Vectorise the per-frame point list: round ONCE for the whole array, then
    # slice each frame and emit with one .tolist() instead of a Python-level
    # round(float(...)) per coordinate (~280k calls). Same values, far faster.
    # cast to float64 BEFORE rounding (the NetCDF arrays are float32; rounding
    # those leaves representation noise that bloats the JSON) so .tolist() emits
    # the same clean 5-decimal numbers the old round(float(...)) produced.
    lat_r = np.round(lat_s.astype(np.float64), 5)
    lon_r = np.round(lon_s.astype(np.float64), 5)
    finite_all = np.isfinite(lat_s) & np.isfinite(lon_s)
    # pre-sort shore landings by the hour they beached, so each frame just takes
    # the prefix [shore_step <= t] via one searchsorted instead of a full scan.
    order = np.argsort(shore_step, kind='stable')
    shore_sorted = [[shore_la[k], shore_lo[k]] for k in order]
    shore_step_sorted = shore_step[order]

    frames = []
    for t in range(ntime):
        good = finite_all[:, t]
        la = lat_r[good, t]
        lo = lon_r[good, t]
        pts = np.column_stack(
            [la, lo, np.full(la.shape[0], intensity)]).tolist()
        # every particle beached at or before this hour, at its landing point
        n_shore = int(np.searchsorted(shore_step_sorted, t, side='right'))
        shore = shore_sorted[:n_shore]
        frames.append({
            'hour': int(round((times[t] - times[0]) / np.timedelta64(1, 'h'))),
            'label': np.datetime_as_string(times[t], unit='m').replace('T', ' '),
            'points': pts,
            'shore': shore,                                  # beached-body locations
            'afloat': round(float(p_afloat[t]) * 100, 1),
            'submerged': round(float(p_sub[t]) * 100, 1),
            'stranded': round(float(p_strand[t]) * 100, 1),
        })

    data = {
        'lkp': {'lat': LKP_LAT, 'lon': LKP_LON},
        'search_time': SEARCH_TIME.strftime('%Y-%m-%dT%H:%M:%S'),
        'duration_hours': int(RUN_DURATION.total_seconds() // 3600),
        'body': {'height_m': BODY_HEIGHT_M, 'weight_kg': BODY_WEIGHT_KG},
        'source': SOURCE,
        'n_particles': int(N_PARTICLES),
        'frames': frames,
        'search_plan': search_plan,        # may be None if planning was skipped
    }
    out_path = os.path.normpath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(data, fh)
    print(f'  app JSON -> {out_path} ({ntime} frames x {len(sel)} pts)')


def _plan_prob_grid(ncfile, hour=PLAN_HOUR, cell_m=PLAN_GRID_M):
    """Fine probability grid of the body's location at forecast `hour`, focused
    on the high-probability CORE of the cloud. Returns
    (prob, xe, ye, hour, cell_m_used): prob[row=lat, col=lon] sums to 1, xe/ye
    are the cell EDGES in lon/lat, and cell_m_used is the actual cell size (it
    grows past PLAN_GRID_M only if the core box would exceed PLAN_MAX_CELLS/side).

    We crop to the box holding PLAN_CORE_FRAC of the probability mass (dropping
    the long thin drift tails) so a ~3 m grid stays small and the routes don't
    sprawl over the whole region."""
    ds = xr.open_dataset(ncfile)
    lon = ds['lon'].values
    lat = ds['lat'].values
    times = ds['time'].values
    ds.close()
    hours = (times - times[0]) / np.timedelta64(1, 'h')
    t = int(np.argmin(np.abs(hours - hour)))     # nearest available frame

    la = lat[:, t]
    lo = lon[:, t]
    good = np.isfinite(la) & np.isfinite(lo)
    la, lo = la[good], lo[good]

    # focus on the dense CORE: drop the (1 - PLAN_CORE_FRAC) tail on each side
    q = (1.0 - PLAN_CORE_FRAC) / 2.0
    lo0, lo1 = np.quantile(lo, [q, 1 - q])
    la0, la1 = np.quantile(la, [q, 1 - q])
    midlat = 0.5 * (la0 + la1)
    # pad so teams can sweep a sonar width past the edge of the mass
    pad_lon = 2 * PLAN_SONAR_M / (111_320.0 * np.cos(np.radians(midlat)))
    pad_lat = 2 * PLAN_SONAR_M / 110_540.0
    extent = [lo0 - pad_lon, lo1 + pad_lon, la0 - pad_lat, la1 + pad_lat]

    # pick the cell size: target PLAN_GRID_M but cap cells/side
    width_m = (extent[1] - extent[0]) * 111_320.0 * np.cos(np.radians(midlat))
    height_m = (extent[3] - extent[2]) * 110_540.0
    cell = float(cell_m)
    side = max(width_m, height_m) / cell
    if side > PLAN_MAX_CELLS:
        cell *= side / PLAN_MAX_CELLS            # auto-coarsen to stay tractable

    counts, xe, ye = _count_grid(lo, la, extent, cell, sigma=1)
    total = counts.sum()
    prob = counts / total if total else counts
    return prob, xe, ye, int(round(hours[t])), cell


def _water_mask(xe, ye):
    """Boolean grid (row=lat, col=lon): True where the cell centre is SEA.
    Uses global_land_mask; falls back to all-water if it isn't installed."""
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    LON, LAT = np.meshgrid(xc, yc)               # shape (ny, nx)
    try:
        from global_land_mask import globe
        return ~globe.is_land(LAT, LON)
    except Exception as exc:
        print('  [land mask unavailable, treating all cells as water]:', exc)
        return np.ones((len(yc), len(xc)), dtype=bool)


def _shore_launch_cells(water, prob, k, min_spacing=3):
    """Pick `k` SHORE launch cells: sea cells adjacent to land, chosen nearest
    to the probability mass and spaced at least `min_spacing` cells apart so
    teams don't all put in at the same beach. Returns a list of (row, col).

    Falls back to the probability peak if there is no coastline in view."""
    from scipy.ndimage import binary_dilation
    land = ~water
    # sea cells touching land = coastline launch points
    coast = water & binary_dilation(land)
    rr, cc = np.nonzero(coast)
    if rr.size == 0:                              # no shore in grid -> peak
        pr, pc = np.unravel_index(int(np.argmax(prob)), prob.shape)
        return [(int(pr), int(pc))] * k

    # mass centroid of the probability cloud (what teams want to reach)
    pr_idx, pc_idx = np.nonzero(prob > 0)
    if pr_idx.size:
        w = prob[pr_idx, pc_idx]
        cy = float((pr_idx * w).sum() / w.sum())
        cx = float((pc_idx * w).sum() / w.sum())
    else:
        cy, cx = prob.shape[0] / 2, prob.shape[1] / 2

    order = np.argsort(np.hypot(rr - cy, cc - cx))   # nearest shore to mass first
    chosen = []
    for idx in order:
        cell = (int(rr[idx]), int(cc[idx]))
        if all(np.hypot(cell[0] - s[0], cell[1] - s[1]) >= min_spacing
               for s in chosen):
            chosen.append(cell)
        if len(chosen) == k:
            break
    while len(chosen) < k:                         # fewer shore cells than teams
        chosen.append(chosen[len(chosen) % max(1, len(chosen))]
                      if chosen else (int(rr[order[0]]), int(cc[order[0]])))
    return chosen


def _cell_to_lonlat(r, c, xe, ye):
    """Centre lon/lat of grid cell (row=lat index r, col=lon index c)."""
    lon = 0.5 * (float(xe[c]) + float(xe[c + 1]))
    lat = 0.5 * (float(ye[r]) + float(ye[r + 1]))
    return lon, lat


def _lonlat_to_cell(lon, lat, xe, ye):
    """Inverse of _cell_to_lonlat: the grid cell (row=lat, col=lon) holding
    (lon, lat), clamped to the grid bounds. xe/ye are ascending edge arrays."""
    c = int(np.clip(np.searchsorted(xe, lon) - 1, 0, len(xe) - 2))
    r = int(np.clip(np.searchsorted(ye, lat) - 1, 0, len(ye) - 2))
    return r, c


def _nearest_water_cell(r, c, water):
    """Nearest water cell to (r, c). Returns (r, c) unchanged if it is already
    water or if the grid holds no water. Lets a user drop a launch point on or
    near land and still get a valid sea start."""
    if water[r, c]:
        return r, c
    rr, cc = np.nonzero(water)
    if rr.size == 0:
        return r, c
    k = int(np.argmin((rr - r) ** 2 + (cc - c) ** 2))
    return int(rr[k]), int(cc[k])


def plan_search(ncfile, hour=None, vehicles=None):
    """Route the rescue team over the body-probability heatmap at forecast `hour`
    (defaults to PLAN_HOUR) using core.search_planner. Teams may only travel over
    water. When `vehicles` is given (a list of {lat, lng, type} dicts from the
    Search Plan map) the fleet launches from THOSE points with type-driven speeds;
    otherwise it auto-launches from the shore points nearest the mass. Returns a
    JSON-serialisable plan dict plus the raw planner result and grid for plotting."""
    if hour is None:
        hour = PLAN_HOUR
    prob, xe, ye, hour, cell_m = _plan_prob_grid(ncfile, hour=hour)
    ny, nx = prob.shape

    # water-only navigation: agents can't cross land, and land cells hold no
    # probability so scanning them is worthless.
    water = _water_mask(xe, ye)
    prob = prob * water
    if prob.sum():
        prob = prob / prob.sum()

    # fleet: user-placed vehicles (start point + type) when provided, else the
    # automatic shore-launch fleet nearest the probability mass.
    agents = []
    user_starts = [] if vehicles else None
    if vehicles:
        for i, v in enumerate(vehicles):
            lat = float(v.get('lat'))
            lon = float(v.get('lng', v.get('lon')))
            vtype = str(v.get('type', 'boat')).lower()
            speed = PLAN_TYPE_SPEED.get(vtype, PLAN_BOAT_MPS)
            r, c = _nearest_water_cell(*_lonlat_to_cell(lon, lat, xe, ye), water)
            user_starts.append((lat, lon))
            agents.append(Agent((r, c), speed_mps=speed, sonar_radius_m=PLAN_SONAR_M,
                                name=f"{vtype.title()} {i + 1}",
                                color=PLAN_COLORS[i % len(PLAN_COLORS)]))
    else:
        starts = _shore_launch_cells(water, prob, PLAN_TEAMS)
        for i in range(PLAN_TEAMS):
            label, speed = PLAN_FLEET[i % len(PLAN_FLEET)]
            agents.append(Agent(starts[i], speed_mps=speed, sonar_radius_m=PLAN_SONAR_M,
                                name=f"{label} {i + 1}",
                                color=PLAN_COLORS[i % len(PLAN_COLORS)]))
    planner = CoveragePlanner(prob, agents, cell_m=cell_m,
                              tick_seconds=PLAN_TICK_SEC,
                              coverage_target=PLAN_COVERAGE,
                              max_time_s=PLAN_MAX_TIME_MIN * 60.0,
                              connectivity=8, passable=water,
                              dispersion=PLAN_DISPERSION)
    res = planner.plan()

    # coarse labelled comms grid over the search area
    refgrid = make_reference_grid(
        [float(xe[0]), float(xe[-1]), float(ye[0]), float(ye[-1])],
        target_cell_m=REFGRID_CELL_M)

    teams = []
    for i, a in enumerate(agents):
        waypoints = [[round(lat, 5), round(lon, 5)]              # -> [lat, lon]
                     for (r, c) in a.path
                     for (lon, lat) in [_cell_to_lonlat(r, c, xe, ye)]]
        s_lon, s_lat = _cell_to_lonlat(a.path[0][0], a.path[0][1], xe, ye)
        launch = ([round(user_starts[i][0], 5), round(user_starts[i][1], 5)]
                  if user_starts else None)
        teams.append({
            'team': a.name, 'color': a.color,
            'launch': launch,
            'speed_mps': round(a.speed_mps, 1),
            'sonar_radius_m': PLAN_SONAR_M,
            'cleared_pct': round(a.cleared_prob * 100, 1),
            'distance_km': round(a.distance_m / 1000.0, 2),
            'time_min': round(a.time_s / 60.0, 1),
            'start_cell': reference_cell_label(refgrid, s_lon, s_lat),
            'waypoints': waypoints,
        })

    plan = {
        'plan_hour': hour,
        'grid_m': round(cell_m, 1),
        'sonar_radius_m': PLAN_SONAR_M,
        'tick_seconds': PLAN_TICK_SEC,
        'ticks': res['ticks'],
        'mission_time_min': round(res['mission_time_s'] / 60.0, 1),
        'stop_reason': res['stop_reason'],
        'coverage_target_pct': round(PLAN_COVERAGE * 100),
        'total_cleared_pct': round(res['cleared_fraction'] * 100, 1),
        'coverage_over_time': [round(float(x) * 100, 1)
                               for x in res['coverage_over_time']],
        'reference_grid': refgrid,
        'teams': teams,
    }
    print(f"  search plan: {len(agents)} craft @ {cell_m:.0f} m grid -> "
          f"{plan['total_cleared_pct']:.1f}% cleared in {plan['mission_time_min']:.0f} min "
          f"({res['stop_reason']}, plan hour {hour})")
    return plan, res, prob, xe, ye


def make_search_plan_map(plan, prob, xe, ye):
    """Draw the coordinated search plan (team paths) over the probability grid
    on a coastline basemap."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    extent = [xe[0], xe[-1], ye[0], ye[-1]]
    extent[1] = max(extent[1], 35.05)
    Hm = np.ma.masked_where(prob <= 0, prob)

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(9, 8))
    ax = plt.axes(projection=proj)
    ax.set_extent(extent, crs=proj)
    ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#e8e2d0', zorder=1)
    ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor='#dfeefc', zorder=0)
    ax.coastlines(resolution='10m', linewidth=0.8, zorder=2)
    ax.pcolormesh(xe, ye, Hm, transform=proj, cmap='inferno', alpha=0.7,
                  zorder=3, shading='flat')

    # reference ("comms") grid: rows A,B,C.. from the north, cols 1,2,3.. west
    rg = plan.get('reference_grid')
    if rg:
        lon_e, lat_e = rg['lon_edges'], rg['lat_edges']     # lat_e: north->south
        for x in lon_e:
            ax.plot([x, x], [lat_e[-1], lat_e[0]], color='#334155',
                    lw=0.6, alpha=0.45, zorder=4, transform=proj)
        for y in lat_e:
            ax.plot([lon_e[0], lon_e[-1]], [y, y], color='#334155',
                    lw=0.6, alpha=0.45, zorder=4, transform=proj)
        for i, lab in enumerate(rg['row_labels']):          # row letters (left)
            yc = 0.5 * (lat_e[i] + lat_e[i + 1])
            ax.text(lon_e[0], yc, lab, color='#334155', fontsize=8, weight='bold',
                    ha='right', va='center', zorder=8, transform=proj)
        for j, lab in enumerate(rg['col_labels']):          # col numbers (top)
            xc = 0.5 * (lon_e[j] + lon_e[j + 1])
            ax.text(xc, lat_e[0], lab, color='#334155', fontsize=8, weight='bold',
                    ha='center', va='bottom', zorder=8, transform=proj)

    for tm in plan['teams']:
        lats = [p[0] for p in tm['waypoints']]
        lons = [p[1] for p in tm['waypoints']]
        ax.plot(lons, lats, '-o', color=tm['color'], ms=2, lw=1.8, zorder=5,
                transform=proj,
                label=f"{tm['team']} ({tm['cleared_pct']}%, "
                      f"{tm['distance_km']} km / {tm['time_min']} min)")
        ax.plot(lons[0], lats[0], marker='s', color=tm['color'], ms=9,
                mec='black', zorder=6, transform=proj)
    ax.plot(LKP_LON, LKP_LAT, marker='*', color='cyan', ms=18, mec='black',
            transform=proj, zorder=7, label='LKP')

    gl = ax.gridlines(draw_labels=True, alpha=0.3)
    gl.top_labels = gl.right_labels = False
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(f"Coordinated search plan @ T+{plan['plan_hour']}h  "
                 f"({PLAN_TEAMS} craft, {plan['total_cleared_pct']}% cleared in "
                 f"{plan['mission_time_min']:.0f} min)")
    out = os.path.join(OUTDIR, 'search_plan_map.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print('  search-plan map ->', out)


def make_interactive_map(ncfile, max_particles=2500):
    """Interactive, time-animated heatmap saved as a SELF-CONTAINED HTML page
    with its own Play / Pause / step controls and a time slider.

    Built directly on Leaflet + leaflet.heat (no time-dimension plugin), so
    the controls are always visible and don't depend on a fragile CDN."""
    import json

    ds = xr.open_dataset(ncfile)
    lon = ds['lon'].values            # (trajectory, time)
    lat = ds['lat'].values
    times = ds['time'].values
    ds.close()
    ntraj, ntime = lon.shape

    step = max(1, ntraj // max_particles)   # subsample particles -> lighter file
    sel = np.arange(0, ntraj, step)

    frames, labels = [], []
    for t in range(ntime):
        la = lat[sel, t]
        lo = lon[sel, t]
        good = np.isfinite(la) & np.isfinite(lo)
        # round to ~1 m to shrink the embedded JSON
        frames.append([[round(float(a), 5), round(float(o), 5)]
                       for a, o in zip(la[good], lo[good])])
        labels.append(np.datetime_as_string(times[t], unit='h').replace('T', '  '))

    # per-frame afloat / submerged / stranded percentages for the overlay
    _, p_afloat, p_sub, p_strand = state_probabilities(ncfile)
    afloat_pct = [round(float(x) * 100, 1) for x in p_afloat]
    sub_pct    = [round(float(x) * 100, 1) for x in p_sub]
    strand_pct = [round(float(x) * 100, 1) for x in p_strand]

    html = _TIME_MAP_TEMPLATE.format(
        lat=LKP_LAT, lon=LKP_LON,
        frames=json.dumps(frames), labels=json.dumps(labels),
        afloat=json.dumps(afloat_pct), submerged=json.dumps(sub_pct),
        stranded=json.dumps(strand_pct),
    )
    out = os.path.join(OUTDIR, 'drowned_heatmap_time.html')
    with open(out, 'w', encoding='utf-8') as fh:
        fh.write(html)
    print('  interactive map ->', out, f'({len(sel)} particles x {ntime} frames)')


_TIME_MAP_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Drowned-body probability over time</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.css"/>
<script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.js"></script>
<script src="https://cdn.jsdelivr.net/npm/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
<style>
  html,body{{margin:0;height:100%;font-family:system-ui,Arial,sans-serif}}
  #map{{height:calc(100% - 64px)}}
  #bar{{height:64px;display:flex;align-items:center;gap:12px;padding:0 14px;
        background:#1f2933;color:#fff;box-sizing:border-box}}
  #bar button{{font-size:18px;padding:6px 16px;border:0;border-radius:6px;
        background:#2f80ed;color:#fff;cursor:pointer}}
  #bar button:hover{{background:#1c63c4}}
  #slider{{flex:1}}
  #label{{font-variant-numeric:tabular-nums;min-width:170px;font-size:15px}}
  #stats{{position:absolute;top:12px;right:12px;z-index:1000;pointer-events:none;
        background:rgba(31,41,51,0.85);color:#fff;padding:10px 14px;border-radius:8px;
        font-size:14px;line-height:1.5;box-shadow:0 2px 8px rgba(0,0,0,.3);
        font-variant-numeric:tabular-nums}}
  #stats b{{font-size:15px}}
  #stats .afloat{{color:#5aa9ff}} #stats .sub{{color:#ff7a6b}} #stats .strand{{color:#b7c0c7}}
</style></head><body>
<div id="map"></div>
<div id="stats"></div>
<div id="bar">
  <button id="play">&#9658; Play</button>
  <button id="step">Step &#9654;&#9654;</button>
  <input id="slider" type="range" min="0" value="0"/>
  <span id="label"></span>
</div>
<script>
const FRAMES = {frames};
const LABELS = {labels};
const AFLOAT = {afloat};
const SUB    = {submerged};
const STRAND = {stranded};
const N = FRAMES.length;
const map = L.map('map').setView([{lat}, {lon}], 11);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png',
  {{attribution:'&copy; OpenStreetMap, &copy; CARTO', maxZoom:19}}).addTo(map);
L.marker([{lat}, {lon}]).addTo(map).bindTooltip('LKP', {{permanent:true}});
const heat = L.heatLayer([], {{radius:18, blur:18, maxZoom:13,
  gradient:{{0.2:'blue',0.4:'lime',0.6:'orange',1.0:'red'}}}}).addTo(map);

const slider = document.getElementById('slider');
const label  = document.getElementById('label');
const playBtn= document.getElementById('play');
const stats  = document.getElementById('stats');
slider.max = N - 1;
let i = 0, timer = null;

function show(k){{
  i = (k + N) % N;
  heat.setLatLngs(FRAMES[i]);
  slider.value = i;
  label.textContent = 'frame ' + (i+1) + ' / ' + N + '   |   ' + LABELS[i];
  let s = '<b>Body state</b><br>'
        + '<span class="afloat">afloat&nbsp;&nbsp;&nbsp;' + AFLOAT[i].toFixed(1) + '%</span><br>'
        + '<span class="sub">submerged ' + SUB[i].toFixed(1) + '%</span>';
  if (STRAND[i] > 0.05) s += '<br><span class="strand">stranded&nbsp;' + STRAND[i].toFixed(1) + '%</span>';
  stats.innerHTML = s;
}}
function play(){{
  if(timer){{ clearInterval(timer); timer=null; playBtn.innerHTML='&#9658; Play'; return; }}
  playBtn.innerHTML='&#10074;&#10074; Pause';
  timer = setInterval(()=>show(i+1), 250);
}}
playBtn.onclick = play;
document.getElementById('step').onclick = ()=>show(i+1);
slider.oninput = ()=>show(parseInt(slider.value));
show(0);
</script></body></html>
"""


def make_drift_diagnostics(cur_path, wav_path):
    """Diagnostic quivers (req 9): the OCEAN CURRENT, the WAVE STOKES drift, and
    the combined EFFECTIVE drift the model actually advects with, at the surface
    (time-averaged over the window). Side by side they show WHY waves can be the
    dominant transport for a floating object."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    dsc = xr.open_dataset(cur_path)
    depth_dim = 'depth' if 'depth' in dsc.dims else None
    uo = dsc['uo'].isel({depth_dim: 0}) if depth_dim else dsc['uo']
    vo = dsc['vo'].isel({depth_dim: 0}) if depth_dim else dsc['vo']
    uo = uo.mean('time', skipna=True)
    vo = vo.mean('time', skipna=True)
    lon = dsc['longitude'].values
    lat = dsc['latitude'].values
    dsc.close()

    dsw = xr.open_dataset(wav_path)
    sx = dsw['VSDX'].mean('time', skipna=True).interp(longitude=lon, latitude=lat)
    sy = dsw['VSDY'].mean('time', skipna=True).interp(longitude=lon, latitude=lat)
    dsw.close()

    U, V, SX, SY = uo.values, vo.values, sx.values, sy.values
    EFX = DRIFT_WEIGHT_CURRENT * U + DRIFT_WEIGHT_STOKES * SX
    EFY = DRIFT_WEIGHT_CURRENT * V + DRIFT_WEIGHT_STOKES * SY

    LON, LAT = np.meshgrid(lon, lat)
    stride = max(1, len(lon) // 22)                  # thin arrows for readability
    panels = [('Ocean current', U, V, '#1f6feb'),
              ('Wave Stokes drift', SX, SY, '#e36209'),
              (f'Effective  {DRIFT_WEIGHT_CURRENT:.1f}*cur + '
               f'{DRIFT_WEIGHT_STOKES:.1f}*stokes', EFX, EFY, '#cf222e')]
    extent = [lon.min(), lon.max(), lat.min(), max(lat.max(), 35.05)]

    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.2),
                             subplot_kw={'projection': proj})
    for ax, (title, u, v, color) in zip(axes, panels):
        # scale each panel to its OWN peak so directions are readable; the peak
        # magnitude is in the title so the three are still quantitatively comparable.
        peak = np.nanmax(np.hypot(u, v)) or 0.1
        ax.set_extent(extent, crs=proj)
        ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#e8e2d0', zorder=1)
        ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor='#dfeefc', zorder=0)
        ax.coastlines(resolution='10m', linewidth=0.6, zorder=2)
        ax.quiver(LON[::stride, ::stride], LAT[::stride, ::stride],
                  u[::stride, ::stride], v[::stride, ::stride], color=color,
                  transform=proj, scale=peak * 22, width=0.005, zorder=3)
        ax.plot(LKP_LON, LKP_LAT, marker='*', color='cyan', ms=14, mec='black',
                transform=proj, zorder=4)
        ax.set_title(f'{title}\n(peak {peak:.2f} m/s)', fontsize=10)
    fig.suptitle('Drift field diagnostics (surface, time-averaged)', fontsize=13)
    out = os.path.join(OUTDIR, 'drift_diagnostics.png')
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print('  drift diagnostics ->', out)


class _ProgressLogHandler(logging.Handler):
    """Parses OpenDrift's 'step i of N' INFO lines into a 0..1 fraction so a
    caller (e.g. the API loading bar) can track the drift run in real time."""
    _re = re.compile(r'step (\d+) of (\d+)')

    def __init__(self, cb):
        super().__init__()
        self.cb = cb

    def emit(self, record):
        try:
            m = self._re.search(record.getMessage())
            if m and int(m.group(2)):
                self.cb(min(1.0, int(m.group(1)) / int(m.group(2))))
        except Exception:
            pass


def run_pipeline(progress_cb=None, incident=None):
    """Run the WHOLE pipeline end-to-end and report progress.

    `incident` (a dict from the React form) overrides LKP / victim / time /
    teams; if None we fall back to incident.json on disk. `progress_cb(percent,
    stage)` is called throughout (0..100) so a frontend can show a loading bar.
    Returns the computed search-plan dict."""
    def p(pct, stage):
        if progress_cb:
            progress_cb(float(pct), stage)
        print(f'  [{pct:5.1f}%] {stage}')

    if incident is not None:
        apply_incident(incident)
    else:
        load_incident()

    p(2, 'Fetching ocean currents' if SOURCE == 'copernicus'
         else 'Initialising model')
    o = build_model()

    # diagnostic quivers of current / Stokes / effective drift (if waves loaded)
    if getattr(o, '_cur_path', None) and getattr(o, '_wav_path', None):
        try:
            make_drift_diagnostics(o._cur_path, o._wav_path)
        except Exception as exc:
            print('  [drift diagnostics skipped]:', exc)

    p(8, 'Seeding particles')
    seed(o)

    handler = _ProgressLogHandler(
        lambda frac: p(10 + 68 * frac, 'Running drift simulation'))
    oplog = logging.getLogger('opendrift')
    oplog.addHandler(handler)
    try:
        p(10, 'Running drift simulation')
        o.run(duration=RUN_DURATION, time_step=TIME_STEP,
              time_step_output=OUTPUT_STEP, outfile=NCFILE)
    finally:
        oplog.removeHandler(handler)

    groups = [o.elements, o.elements_deactivated]
    phase = np.concatenate([g.phase for g in groups if g.phase.size])
    print(f'  final phases: drowning={int(np.sum(phase==1))} '
          f'submerged={int(np.sum(phase==0))} '
          f'rising={int(np.sum(phase==2))} surface={int(np.sum(phase==3))}')

    p(80, 'Rendering heatmap')
    make_heatmap_map(o)
    make_state_probability(NCFILE)

    p(88, 'Planning coordinated search')
    plan, _res, prob, xe, ye = plan_search(NCFILE)
    make_search_plan_map(plan, prob, xe, ye)

    p(94, 'Exporting results')
    make_interactive_map(NCFILE)
    export_app_json(NCFILE, search_plan=plan)
    p(100, 'Done')
    return plan


def _write_progress(path, **state):
    """Atomically write the progress JSON (temp + replace) so the API never
    reads a half-written file."""
    import json
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(state, fh)
    os.replace(tmp, path)


def run_as_job(incident_path, progress_path):
    """Run the pipeline as a stand-alone process, streaming progress to a JSON
    file the API polls. Used by api/server.py so each run is isolated (releases
    file handles on exit; Copernicus runs in a clean main thread)."""
    import json
    with open(incident_path, encoding='utf-8') as fh:
        incident = json.load(fh)
    state = {'percent': 0.0, 'stage': 'Starting', 'done': False, 'error': None}
    _write_progress(progress_path, **state)

    def cb(percent, stage):
        state['percent'] = round(float(percent), 1)
        state['stage'] = stage
        _write_progress(progress_path, **state)

    try:
        run_pipeline(progress_cb=cb, incident=incident)
        state.update(percent=100.0, stage='Done', done=True)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        state.update(done=True, error=str(exc))
    _write_progress(progress_path, **state)


def main():
    run_pipeline()
    print('Done. Outputs in', OUTDIR)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Drowned-body drift + search plan')
    ap.add_argument('--incident', help='incident JSON to run as an API job')
    ap.add_argument('--progress', help='progress JSON file to stream into')
    a = ap.parse_args()
    if a.incident and a.progress:
        run_as_job(a.incident, a.progress)
    else:
        main()
