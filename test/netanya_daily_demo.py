"""
DEMO (standalone -- does not touch the main pipeline).

Runs the real DrownedBodyDrift model (sink / refloat + effective current+wave
drift) from the Netanya beaches, 10 April 2026, for 10 days, and draws ONE map
per day so you can watch the cloud drift NORTH early (Apr 10-13, flow northward)
then turn SOUTH after the ~Apr 14 reversal -- toward Herzliya, where the real
body was found.

    python netanya_daily_demo.py
Output: ./output/netanya_daily.png  (one panel per day)
"""

import os
import sys
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import sim_drowned_body as sim
from core.drowned_drift import SURFACE_EPS

NET_LAT, NET_LON = 32.33, 34.84        # Netanya (last-known position)
HERZ_LAT, HERZ_LON = 32.162, 34.797    # Herzliya / Sidna Ali (recovery)
START = datetime(2026, 4, 10, 6, 0, 0)
DAYS = 10
NC = os.path.join(HERE, 'output', 'netanya_daily.nc')
PNG = os.path.join(HERE, 'output', 'netanya_daily.png')


def run_sim():
    # point the main model at Netanya / April 10 / 10 days (demo-sized ensemble)
    sim.LKP_LAT, sim.LKP_LON = NET_LAT, NET_LON
    sim.LKP_RADIUS_M = 400
    sim.SEARCH_TIME = START
    sim.RUN_DURATION = timedelta(days=DAYS)
    sim.N_PARTICLES = 4000
    sim.SUBSET_MARGIN_DEG = 0.5            # box big enough for a ~25 km drift
    sim.REFRESH_DATA = False
    # OFFLINE DEMO: use only the cached Copernicus subset, never download live
    # (the ~100 MB Netanya data is pre-cached in test/cache/). Set this False to
    # allow a one-off download the first time you run a new case.
    sim.CACHE_ONLY = True
    # April Mediterranean SST -> refloat ~4-5 days, so the body surfaces into the
    # mid-April southward window.
    sim.WATER_TEMP_C = 20.0
    # Uses the SAME settings as the live app (effective current+wave drift + the
    # nearshore longshore current set in the main config) -- nothing demo-only.
    o = sim.build_model()
    # DEMO choice: the 4.2 km field has a strong onshore component off Netanya
    # on Apr 10-13 that beaches the whole cloud in ~2 days (a known nearshore
    # over-stranding artifact of the coarse grid). Use 'previous' so particles
    # hug the coast instead of deactivating -> they survive the full 10 days and
    # the alongshore north->south reversal becomes visible.
    o.set_config('general:coastline_action', 'previous')
    sim.seed(o)
    o.run(duration=sim.RUN_DURATION, time_step=900, time_step_output=3600,
          outfile=NC)
    return o


def daily_panels():
    ds = xr.open_dataset(NC)
    lon = ds['lon'].values
    lat = ds['lat'].values
    z = ds['z'].values
    times = ds['time'].values
    ds.close()
    hours = (times - times[0]) / np.timedelta64(1, 'h')
    day_idx = [int(np.argmin(np.abs(hours - d * 24))) for d in range(DAYS + 1)]

    fin = np.isfinite(lon) & np.isfinite(lat)
    extent = [min(lon[fin].min(), HERZ_LON, NET_LON) - 0.03,
              max(lon[fin].max(), NET_LON) + 0.03,
              min(lat[fin].min(), HERZ_LAT) - 0.03,
              max(lat[fin].max(), NET_LAT) + 0.03]
    extent[1] = max(extent[1], 35.0)

    proj = ccrs.PlateCarree()
    ncol, nrow = 4, 3
    fig, axes = plt.subplots(nrow, ncol, figsize=(18, 12),
                             subplot_kw={'projection': proj})
    axes = axes.ravel()
    for k, di in enumerate(day_idx):
        ax = axes[k]
        la, lo, zz = lat[:, di], lon[:, di], z[:, di]
        m = np.isfinite(la) & np.isfinite(lo)
        afloat = m & (zz >= -SURFACE_EPS)
        sub = m & (zz < -SURFACE_EPS)
        ax.set_extent(extent, crs=proj)
        ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#e8e2d0', zorder=1)
        ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor='#dfeefc', zorder=0)
        ax.coastlines(resolution='10m', linewidth=0.6, zorder=2)
        ax.scatter(lo[sub], la[sub], s=3, c='#c0392b', alpha=0.25, zorder=3,
                   transform=proj, label='submerged')
        ax.scatter(lo[afloat], la[afloat], s=3, c='#2f80ed', alpha=0.5, zorder=4,
                   transform=proj, label='afloat')
        ax.plot(NET_LON, NET_LAT, marker='*', color='cyan', ms=13, mec='black',
                transform=proj, zorder=5)
        ax.plot(HERZ_LON, HERZ_LAT, marker='v', color='#16a34a', ms=10, mec='black',
                transform=proj, zorder=5)
        ax.set_title(f'Day {k}  ({(START + timedelta(days=k)):%b %d})', fontsize=11)
    for j in range(len(day_idx), len(axes)):
        axes[j].axis('off')

    # one shared legend in the spare cell
    handles = [plt.Line2D([], [], marker='o', ls='', color='#2f80ed', label='afloat body'),
               plt.Line2D([], [], marker='o', ls='', color='#c0392b', label='submerged body'),
               plt.Line2D([], [], marker='*', ls='', color='cyan', mec='k', label='Netanya (LKP)'),
               plt.Line2D([], [], marker='v', ls='', color='#16a34a', mec='k', label='Herzliya (found)')]
    axes[-1].legend(handles=handles, loc='center', fontsize=12, frameon=True)
    fig.suptitle('Drowned-body drift: Netanya (10 Apr) -> found at Herzliya '
                 '(19 Apr 2026)', fontsize=14)
    fig.savefig(PNG, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print('daily panels ->', PNG)


if __name__ == '__main__':
    run_sim()
    daily_panels()
