"""
Monte Carlo run of the DrownedBodyDrift prior off Haifa, with:

  * a STATIC probability heatmap drawn on a real coastline basemap (cartopy)
  * an INTERACTIVE, time-animated heatmap over the 0-3 day span (folium)

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
    DrownedBodyDrift, refloat_time_seconds, drown_delay_seconds, body_dynamics,
    SURFACE_EPS)
from core.search_planner import Agent, CoveragePlanner

OUTDIR = os.path.join(HERE, 'output')
os.makedirs(OUTDIR, exist_ok=True)

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
RUN_DURATION  = timedelta(days=3)     # search window
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
SUBSET_MARGIN_DEG = 0.8     # half-width of the box around the LKP (~75 km of drift room)
REFRESH_DATA      = False   # True = always re-download; False = reuse a cached subset
                            # with the SAME params (faster, and never overwrites a
                            # file another run might still hold open on Windows)

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
# A drowned body has NO windage while submerged, but each time it REFLOATS the
# wind and breaking waves push it -- and that wave/wind-sea transport is what a
# current-only model misses near shore. We add the Copernicus WAVE product's
# Stokes drift (VSDX/VSDY); OpenDrift applies it with a depth decay, so it acts
# on the AFLOAT phases and barely touches the submerged ones. Direct windage
# needs a separate wind field (ERA5 for past dates / GFS for recent ones), so
# WIND_DRIFT_FACTOR only takes effect once such a wind reader is added.
USE_STOKES        = True    # add Copernicus wave Stokes drift (afloat phases)
WIND_DRIFT_FACTOR = 0.0     # windage coefficient; needs a wind reader to act

# --- afloat / drowned probability ------------------------------------------
# At each of these times (hours since the LKP) we report the probability that
# the body is AFLOAT (at the surface -> visible, wind-driftable, spottable) vs
# SUBMERGED (underwater), measured as the fraction of the ensemble in each
# state. Useful for deciding when a surface/air search is worthwhile.
PROB_QUERY_HOURS = [1, 6, 12, 24, 48, 72]

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
PLAN_GRID_M      = 200     # planning grid cell size (finer than the display grid)
PLAN_TEAMS       = 3       # number of rescue teams (agents)
PLAN_SONAR_M     = 400     # sonar sweep radius per team (m)
PLAN_SPEED_CELLS = 2       # grid cells a team advances per planning tick
PLAN_HORIZON     = 30      # number of planning time-steps T
PLAN_DISPERSION  = 0.4     # 0..1: how strongly teams keep apart to cover more
                           # ground (0 = pure probability-greedy). ~0.3-0.5 spreads
                           # the routes without giving up real coverage.
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
#   NAHSHON_SOURCE=offline   -> constant forcing, no download
#   NAHSHON_PARTICLES=1500   -> smaller, faster ensemble
if os.environ.get('NAHSHON_SOURCE'):
    SOURCE = os.environ['NAHSHON_SOURCE']
if os.environ.get('NAHSHON_PARTICLES'):
    N_PARTICLES = int(os.environ['NAHSHON_PARTICLES'])


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
    global PLAN_TEAMS, PLAN_SONAR_M, PLAN_SPEED_CELLS, PLAN_HORIZON, PLAN_HOUR

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
                   ('speed', 'PLAN_SPEED_CELLS'), ('horizon', 'PLAN_HORIZON'),
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
    path = os.path.join(OUTDIR, fname)
    if os.path.exists(path) and not REFRESH_DATA:
        print(f"  reusing cached currents: {fname}")
        return path

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
        output_filename=fname, output_directory=OUTDIR,
        username=CMEMS_USER, password=CMEMS_PASS,
        overwrite=True, disable_progress_bar=True)
    return path


def download_waves_subset():
    """Download surface Stokes drift (VSDX/VSDY) from the Copernicus Med WAVE
    product for the same box/time as the currents. Stokes drift is the wave-
    driven mass transport that pushes a FLOATING body downwind; OpenDrift
    applies it with a depth decay, so only the afloat phases feel it. Same
    param-named cache as the currents (never overwrites a locked file)."""
    import copernicusmarine
    t0 = SEARCH_TIME - timedelta(hours=2)
    t1 = SEARCH_TIME + RUN_DURATION + timedelta(hours=2)
    key = (f"{LKP_LAT:.3f}_{LKP_LON:.3f}_{SEARCH_TIME:%Y%m%d%H}_"
           f"{int(RUN_DURATION.total_seconds()//3600)}h_{SUBSET_MARGIN_DEG}")
    fname = f"waves_{key}.nc"
    path = os.path.join(OUTDIR, fname)
    if os.path.exists(path) and not REFRESH_DATA:
        print(f"  reusing cached waves: {fname}")
        return path
    print(f"  waves subset (Stokes drift) -> {fname}")
    copernicusmarine.subset(
        dataset_id='cmems_mod_med_wav_anfc_4.2km_PT1H-i',
        variables=['VSDX', 'VSDY'],
        minimum_longitude=LKP_LON - SUBSET_MARGIN_DEG,
        maximum_longitude=LKP_LON + SUBSET_MARGIN_DEG,
        minimum_latitude=LKP_LAT - SUBSET_MARGIN_DEG,
        maximum_latitude=LKP_LAT + SUBSET_MARGIN_DEG,
        start_datetime=t0.strftime('%Y-%m-%dT%H:%M:%S'),
        end_datetime=t1.strftime('%Y-%m-%dT%H:%M:%S'),
        output_filename=fname, output_directory=OUTDIR,
        username=CMEMS_USER, password=CMEMS_PASS,
        overwrite=True, disable_progress_bar=True)
    return path


def build_model():
    o = DrownedBodyDrift(loglevel=20)

    # Height & weight -> sink/rise speeds (refloat base is applied per-particle
    # in seed()). Leaner bodies sink faster and refloat slower; heavier-set
    # bodies are near-neutral and bob back sooner.
    dyn = body_dynamics(BODY_HEIGHT_M, BODY_WEIGHT_KG)
    o.sink_speed = dyn['sink']
    o.rise_speed = dyn['rise']
    print(f"  body {BODY_HEIGHT_M:.2f} m / {BODY_WEIGHT_KG:.0f} kg "
          f"(BMI {dyn['bmi']:.1f}, density {dyn['density']:.0f} kg/m^3) -> "
          f"sink {dyn['sink']:.3f} m/s, rise {dyn['rise']:.3f} m/s, "
          f"refloat ~{dyn['refloat']/3600:.1f} h")

    o.add_reader(reader_global_landmask.Reader())

    if SOURCE == 'copernicus':
        # === REAL 3D DATA (downloaded once, read locally) ===================
        # Requires: pip install copernicusmarine. We subset a small local file
        # for the search box/time/depth (see download_currents_subset) and read
        # it with the generic NetCDF reader -- uo/vo carry standard_names, so
        # they are auto-rotated to x/y_sea_water_velocity and interpolated to
        # each particle's depth.
        from opendrift.readers import reader_copernicusmarine
        cur_path = download_currents_subset()
        currents = reader_netCDF_CF_generic.Reader(cur_path, name='CMEMS-3D-local')
        o.add_reader(currents)

        # Bathymetry for sea_floor_depth_below_sea_level so bodies rest on the
        # real seabed instead of the flat 100 m fallback. Static (no time axis),
        # so a single one-off stream is cheap -- no need to cache it locally.
        try:
            bathy = reader_copernicusmarine.Reader(
                'cmems_mod_med_phy_anfc_4.2km_static',
                username=CMEMS_USER, password=CMEMS_PASS)
            o.add_reader(bathy)
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

    o.set_config('general:coastline_action', 'stranding')
    o.set_config('general:seafloor_action', 'lift_to_seafloor')
    o.set_config('drift:vertical_advection', True)
    o.set_config('drift:vertical_mixing', False)
    o.set_config('drift:stokes_drift', USE_STOKES)   # wave push on afloat phases

    # Gaussian noise -> realistic ensemble spread (see top of file to tune).
    o.set_config('drift:current_uncertainty', CURRENT_UNCERTAINTY)
    o.set_config('drift:horizontal_diffusivity', HORIZONTAL_DIFFUSIVITY)
    return o


def seed(o):
    start = SEARCH_TIME                      # you set this at the top of the file
    # per-particle uncertain LKP->drown delay (0..30 min) and body-driven timers
    drown   = drown_delay_seconds(N_PARTICLES, DROWN_TIME_MAX_MIN,
                                  DROWN_TIME_MODE_MIN, rng=rng)
    refloat = refloat_time_seconds(N_PARTICLES, BODY_HEIGHT_M, BODY_WEIGHT_KG,
                                   rng=rng)
    surface = (rng.uniform(3, 9, N_PARTICLES) * 3600.0).astype(np.float32)
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
    times = ds['time'].values
    ds.close()
    ntraj, ntime = lon.shape

    step = max(1, ntraj // max_particles)
    sel = np.arange(0, ntraj, step)

    _, p_afloat, p_sub, p_strand = state_probabilities(ncfile)

    frames = []
    for t in range(ntime):
        la = lat[sel, t]
        lo = lon[sel, t]
        good = np.isfinite(la) & np.isfinite(lo)
        pts = [[round(float(a), 5), round(float(o), 5), intensity]
               for a, o in zip(la[good], lo[good])]
        frames.append({
            'hour': int(round((times[t] - times[0]) / np.timedelta64(1, 'h'))),
            'label': np.datetime_as_string(times[t], unit='m').replace('T', ' '),
            'points': pts,
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
    """Probability heatmap (as a dense grid) of the body's location at forecast
    `hour`, built from the particle cloud at that frame. Returns
    (prob, xe, ye, hour) where prob[row=lat, col=lon] sums to 1 and xe/ye are
    the cell EDGES in lon/lat so grid cells map back to real coordinates.

    The grid extent is stretched east to the coastline (~35.05 E) so the SHORE
    is inside the grid -- the planner launches teams from there."""
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

    pad = 0.01
    extent = [min(lo.min(), LKP_LON) - pad, max(lo.max(), LKP_LON) + pad,
              min(la.min(), LKP_LAT) - pad, max(la.max(), LKP_LAT) + pad]
    extent[1] = max(extent[1], 35.05)            # reach the shore
    counts, xe, ye = _count_grid(lo, la, extent, cell_m, sigma=1)
    total = counts.sum()
    prob = counts / total if total else counts
    return prob, xe, ye, int(round(hours[t]))


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


def plan_search(ncfile, hour=None):
    """Route the heterogeneous rescue team over the body-probability heatmap at
    forecast `hour` (defaults to PLAN_HOUR) using core.search_planner. Teams
    launch FROM SHORE and may only travel over water. Returns a JSON-
    serialisable plan dict plus the raw planner result and grid for plotting."""
    if hour is None:
        hour = PLAN_HOUR
    prob, xe, ye, hour = _plan_prob_grid(ncfile, hour=hour)
    ny, nx = prob.shape

    # water-only navigation: agents can't cross land, and land cells hold no
    # probability so scanning them is worthless.
    water = _water_mask(xe, ye)
    prob = prob * water
    if prob.sum():
        prob = prob / prob.sum()

    # sonar radius (m) -> grid cells (>=1 so a sweep always covers neighbours)
    sonar_cells = max(1, int(round(PLAN_SONAR_M / PLAN_GRID_M)))
    # teams put to sea from the nearest shore points to the probability mass
    starts = _shore_launch_cells(water, prob, PLAN_TEAMS)

    agents = [Agent(starts[i], speed=PLAN_SPEED_CELLS, sonar_radius=sonar_cells,
                    name=chr(65 + i), color=PLAN_COLORS[i % len(PLAN_COLORS)])
              for i in range(PLAN_TEAMS)]
    planner = CoveragePlanner(prob, agents, horizon=PLAN_HORIZON,
                              connectivity=8, passable=water,
                              dispersion=PLAN_DISPERSION)
    res = planner.plan()

    teams = []
    for a in agents:
        waypoints = [[round(lat, 5), round(lon, 5)]              # -> [lat, lon]
                     for (r, c) in a.path
                     for (lon, lat) in [_cell_to_lonlat(r, c, xe, ye)]]
        teams.append({
            'team': a.name, 'color': a.color,
            'speed': a.speed, 'sonar_radius_m': PLAN_SONAR_M,
            'cleared_pct': round(a.cleared_prob * 100, 1),
            'waypoints': waypoints,
        })

    plan = {
        'plan_hour': hour,
        'grid_m': PLAN_GRID_M,
        'horizon': PLAN_HORIZON,
        'sonar_radius_m': PLAN_SONAR_M,
        'total_cleared_pct': round(res['cleared_fraction'] * 100, 1),
        'coverage_over_time': [round(float(x) * 100, 1)
                               for x in res['coverage_over_time']],
        'teams': teams,
    }
    print(f"  search plan: {PLAN_TEAMS} teams, T={PLAN_HORIZON}, "
          f"sonar {PLAN_SONAR_M} m -> {plan['total_cleared_pct']:.1f}% "
          f"probability cleared (plan hour {hour})")
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

    for tm in plan['teams']:
        lats = [p[0] for p in tm['waypoints']]
        lons = [p[1] for p in tm['waypoints']]
        ax.plot(lons, lats, '-o', color=tm['color'], ms=3, lw=1.8, zorder=5,
                transform=proj, label=f"Team {tm['team']} ({tm['cleared_pct']}%)")
        ax.plot(lons[0], lats[0], marker='s', color=tm['color'], ms=9,
                mec='black', zorder=6, transform=proj)
    ax.plot(LKP_LON, LKP_LAT, marker='*', color='cyan', ms=18, mec='black',
            transform=proj, zorder=7, label='LKP')

    gl = ax.gridlines(draw_labels=True, alpha=0.3)
    gl.top_labels = gl.right_labels = False
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title(f"Coordinated search plan @ T+{plan['plan_hour']}h  "
                 f"({PLAN_TEAMS} teams, {plan['total_cleared_pct']}% cleared)")
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
