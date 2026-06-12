"""
STANDALONE EXPERIMENT (does not touch the main model).

Question: if a surface object near the Netanya beaches were driven by the WAVE
FLOW ALONE (no ocean current), which way would it drift on 10 April 2026?

We use OpenDrift's operational SAR model -- Leeway -- and feed it ONLY the
Copernicus wave Stokes drift (VSDX/VSDY) as the water velocity. Leeway requires a
wind field, so we give it a NEGLIGIBLE (zero) constant wind: that switches off the
wind/leeway term, leaving the object to drift with the supplied water flow = the
wave flow. So this isolates "wave-only" transport.

    python leeway_wave_netanya.py

Outputs go to ./output/leeway_wave_netanya.{nc,png}.
"""

import os
import sys
from datetime import datetime, timedelta

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from opendrift.models.leeway import Leeway
from opendrift.readers import reader_netCDF_CF_generic
from opendrift.readers import reader_constant
from opendrift.readers import reader_global_landmask

# reuse the main project's wave downloader + Copernicus credentials (read-only;
# importing does not run anything)
import sim_drowned_body as sim

# --- experiment parameters -------------------------------------------------
NETANYA_LAT = 32.33          # Netanya beaches
NETANYA_LON = 34.83          # a touch offshore so objects start in the water
LKP_RADIUS_M = 500
START = datetime(2026, 4, 10, 6, 0, 0)
DURATION = timedelta(days=3)
N = 2000
OBJECT_TYPE = 27             # Leeway "Person-in-water (PIW)" class
OUTDIR = os.path.join(HERE, 'output')
NC = os.path.join(OUTDIR, 'leeway_wave_netanya.nc')
PNG = os.path.join(OUTDIR, 'leeway_wave_netanya.png')


def get_wave_file():
    """Download the Copernicus Med wave subset (Stokes drift) for this box/time
    by reusing the main project's downloader."""
    sim.LKP_LAT, sim.LKP_LON = NETANYA_LAT, NETANYA_LON
    sim.SEARCH_TIME = START
    sim.RUN_DURATION = DURATION
    sim.SUBSET_MARGIN_DEG = 0.4
    sim.REFRESH_DATA = False
    return sim.download_waves_subset()


def main():
    wav = get_wave_file()

    o = Leeway(loglevel=20)
    o.add_reader(reader_global_landmask.Reader())

    # WAVE FLOW as the water velocity: remap the Stokes-drift variables to the
    # sea-water-velocity standard names the model advects with.
    waves = reader_netCDF_CF_generic.Reader(
        wav, name='wave-flow',
        standard_name_mapping={'VSDX': 'x_sea_water_velocity',
                               'VSDY': 'y_sea_water_velocity'})
    o.add_reader(waves)

    # Leeway needs wind; a zero constant wind makes the wind/leeway term ~0 so the
    # drift reflects the WAVE FLOW ONLY.
    o.add_reader(reader_constant.Reader({'x_wind': 0.0, 'y_wind': 0.0}))

    o.set_config('general:coastline_action', 'stranding')

    o.seed_elements(lon=NETANYA_LON, lat=NETANYA_LAT, radius=LKP_RADIUS_M,
                    number=N, time=START, object_type=OBJECT_TYPE)

    o.run(duration=DURATION, time_step=900, time_step_output=3600, outfile=NC)
    print(o)

    # net drift direction (active + stranded)
    groups = [o.elements, o.elements_deactivated]
    la = np.concatenate([g.lat for g in groups if g.lat.size])
    lo = np.concatenate([g.lon for g in groups if g.lon.size])
    m = np.isfinite(la) & np.isfinite(lo)
    dlat = (la[m].mean() - NETANYA_LAT) * 111.0
    dlon = (lo[m].mean() - NETANYA_LON) * 111.0 * np.cos(np.radians(NETANYA_LAT))
    ns = 'NORTH' if dlat > 0 else 'SOUTH'
    ew = 'EAST(onshore)' if dlon > 0 else 'WEST(offshore)'
    print(f"\nWAVE-ONLY mean drift after {DURATION}: "
          f"{dlat:+.2f} km {ns}, {dlon:+.2f} km {ew}")
    print(f"Herzliya (recovery) is ~19 km SOUTH of Netanya -> a southward result "
          f"would support the wave-flow hypothesis.")

    o.plot(fast=True, filename=PNG)
    print('map ->', PNG)


if __name__ == '__main__':
    main()
