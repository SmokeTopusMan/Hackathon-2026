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
REFRESH_DATA      = True    # True = re-download the subset each run (latest forecast)
CURFILE           = os.path.join(OUTDIR, 'haifa_currents.nc')

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
# The run also writes a JSON the drift-app map loads (drift-app/public/), with
# one heatmap frame per output hour + afloat/submerged probabilities + the LKP.
APP_JSON          = os.path.join(HERE, '..', 'drift-app', 'public', 'drift_data.json')
APP_MAX_PARTICLES = 1500   # particles per frame in the JSON (keeps the file light)

rng = np.random.default_rng(42)       # reproducible ensemble


def download_currents_subset():
    """Download a SMALL local NetCDF of 3D currents for the search box / time
    window / depth cap, so the run reads locally instead of streaming the whole
    Mediterranean per step. Returns the local file path.

    Re-downloaded when REFRESH_DATA is True (always-fresh forecast); otherwise an
    existing file for the same name is reused.
    """
    import copernicusmarine
    t0 = SEARCH_TIME - timedelta(hours=2)                  # small read-buffer
    t1 = SEARCH_TIME + RUN_DURATION + timedelta(hours=2)
    print(f"  currents subset: {t0:%Y-%m-%d %H:%M}..{t1:%Y-%m-%d %H:%M} UTC, "
          f"0-{MAX_DEPTH_M:.0f} m, +/-{SUBSET_MARGIN_DEG} deg around LKP")
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
        output_filename=os.path.basename(CURFILE), output_directory=OUTDIR,
        username=CMEMS_USER, password=CMEMS_PASS,
        overwrite=REFRESH_DATA, skip_existing=not REFRESH_DATA,
        disable_progress_bar=True)
    return CURFILE


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
                    intensity=0.45):
    """Write the simulation result as JSON the React drift-app map loads:
    one heatmap frame per output hour ([lat, lon, intensity] points) plus the
    afloat/submerged/stranded probability of each frame and the LKP."""
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
    }
    out_path = os.path.normpath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(data, fh)
    print(f'  app JSON -> {out_path} ({ntime} frames x {len(sel)} pts)')


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


def main():
    o = build_model()
    seed(o)
    o.run(duration=RUN_DURATION, time_step=TIME_STEP,
          time_step_output=OUTPUT_STEP, outfile=NCFILE)
    print(o)

    groups = [o.elements, o.elements_deactivated]
    phase = np.concatenate([g.phase for g in groups if g.phase.size])
    print(f'  final phases: drowning={int(np.sum(phase==1))} '
          f'submerged={int(np.sum(phase==0))} '
          f'rising={int(np.sum(phase==2))} surface={int(np.sum(phase==3))}')

    make_heatmap_map(o)
    make_state_probability(NCFILE)
    make_interactive_map(NCFILE)
    export_app_json(NCFILE)
    print('Done. Outputs in', OUTDIR)


if __name__ == '__main__':
    main()
