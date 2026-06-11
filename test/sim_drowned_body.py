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
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from opendrift.readers import reader_constant
from opendrift.readers import reader_global_landmask

from core.drowned_drift import DrownedBodyDrift, refloat_time_seconds

OUTDIR = os.path.join(HERE, 'output')
os.makedirs(OUTDIR, exist_ok=True)

# --- run parameters --------------------------------------------------------
SOURCE        = 'offline'             # 'offline' (constant) or 'copernicus' (real 3D)
N_PARTICLES   = 10000                 # Monte Carlo ensemble size
LKP_LON       = 34.92                 # last-known position (lon)
LKP_LAT       = 32.83                 # last-known position (lat)
LKP_RADIUS_M  = 300                   # position uncertainty of the LKP (m)
RUN_DURATION  = timedelta(days=3)     # search window
TIME_STEP     = 900                   # 15 min integration step
OUTPUT_STEP   = 3600                  # save a frame every hour
NCFILE        = os.path.join(OUTDIR, 'drowned.nc')

rng = np.random.default_rng(42)       # reproducible ensemble


def build_model():
    o = DrownedBodyDrift(loglevel=20)
    o.add_reader(reader_global_landmask.Reader())

    if SOURCE == 'copernicus':
        # === REAL 3D DATA ===================================================
        # Requires: pip install copernicusmarine  +  `copernicusmarine login`.
        from opendrift.readers import reader_copernicusmarine

        # 3D Mediterranean hourly currents -- NOTE: no "-2D", so it carries a
        # `depth` axis. OpenDrift interpolates the current to each particle's z.
        currents = reader_copernicusmarine.Reader(
            'cmems_mod_med_phy-cur_anfc_4.2km_PT1H-m')
        o.add_reader(currents)

        # Bathymetry for sea_floor_depth_below_sea_level so bodies rest on the
        # real seabed instead of the flat 100 m fallback. The Med static
        # dataset carries `deptho`; after `copernicusmarine login` confirm the
        # id and variable with `copernicusmarine describe -c <id>`. If its
        # standard_name isn't picked up automatically, pass
        # standard_name_mapping={'deptho': 'sea_floor_depth_below_sea_level'}.
        try:
            bathy = reader_copernicusmarine.Reader(
                'cmems_mod_med_phy_anfc_4.2km_static')
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
    return o


def seed(o):
    start = datetime(2025, 1, 1, 0, 0, 0)
    refloat = refloat_time_seconds(N_PARTICLES, rng=rng)          # ~1 day placeholder
    surface = (rng.uniform(3, 9, N_PARTICLES) * 3600.0).astype(np.float32)
    o.seed_elements(
        lon=LKP_LON, lat=LKP_LAT, radius=LKP_RADIUS_M,
        number=N_PARTICLES, time=start, z=0,
        phase=0, phase_start_age=0,
        refloat_time=refloat, surface_time=surface,
    )
    return o


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def _heat_grid(lon, lat, extent, bins=200, sigma=2):
    """Smoothed, normalised 2D density on a regular grid."""
    from scipy.ndimage import gaussian_filter
    H, xe, ye = np.histogram2d(
        lon, lat, bins=bins,
        range=[[extent[0], extent[1]], [extent[2], extent[3]]])
    H = gaussian_filter(H.T, sigma=sigma)          # .T -> axis0 = lat
    if H.sum():
        H = H / H.sum()
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    return H, xc, yc


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
    H, xc, yc = _heat_grid(lon, lat, extent)
    # hide the faint Gaussian halo: only show cells above 2% of the peak
    Hm = np.ma.masked_where(H < H.max() * 0.02, H)

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(9, 8))
    ax = plt.axes(projection=proj)
    ax.set_extent(extent, crs=proj)
    ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#e8e2d0', zorder=1)
    ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor='#dfeefc', zorder=0)
    ax.coastlines(resolution='10m', linewidth=0.8, zorder=2)

    mesh = ax.pcolormesh(xc, yc, Hm, transform=proj, cmap='inferno',
                         alpha=0.75, zorder=3, shading='auto')
    ax.plot(LKP_LON, LKP_LAT, marker='*', color='cyan', ms=18,
            mec='black', transform=proj, zorder=4, label='LKP')
    gl = ax.gridlines(draw_labels=True, alpha=0.3)
    gl.top_labels = gl.right_labels = False
    ax.legend(loc='upper right')
    ax.set_title(f'Drowned-body probability after {RUN_DURATION} '
                 f'({len(lon)} particles, source={SOURCE})')
    fig.colorbar(mesh, ax=ax, label='probability density', shrink=0.8)
    out = os.path.join(OUTDIR, 'drowned_heatmap_map.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print('  static map ->', out)


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

    html = _TIME_MAP_TEMPLATE.format(
        lat=LKP_LAT, lon=LKP_LON,
        frames=json.dumps(frames), labels=json.dumps(labels),
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
</style></head><body>
<div id="map"></div>
<div id="bar">
  <button id="play">&#9658; Play</button>
  <button id="step">Step &#9654;&#9654;</button>
  <input id="slider" type="range" min="0" value="0"/>
  <span id="label"></span>
</div>
<script>
const FRAMES = {frames};
const LABELS = {labels};
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
slider.max = N - 1;
let i = 0, timer = null;

function show(k){{
  i = (k + N) % N;
  heat.setLatLngs(FRAMES[i]);
  slider.value = i;
  label.textContent = 'frame ' + (i+1) + ' / ' + N + '   |   ' + LABELS[i];
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
    print(f'  final phases: submerged={int(np.sum(phase==0))} '
          f'rising={int(np.sum(phase==2))} surface={int(np.sum(phase==3))}')

    make_heatmap_map(o)
    make_interactive_map(NCFILE)
    print('Done. Outputs in', OUTDIR)


if __name__ == '__main__':
    main()
