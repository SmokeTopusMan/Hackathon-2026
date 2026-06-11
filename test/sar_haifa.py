"""
SAR (Search and Rescue) drift simulation off the Haifa shore.

Uses OpenDrift's Leeway model -- the standard model used operationally
(e.g. by the US Coast Guard's SAROPS and many national SAR agencies) to
predict where a drifting object / person will go under wind + current.

Run:
    python sar_haifa.py

This version uses CONSTANT wind + current so it runs offline with no
credentials. See the bottom of the file for how to swap in real forecast
data (Copernicus Marine ocean currents + a wind model) for an operational run.

Outputs are written to the ./output/ folder next to this script.
"""

import os
from datetime import datetime, timedelta
from opendrift.models.leeway import Leeway
from opendrift.readers import reader_constant
from opendrift.readers import reader_global_landmask
from opendrift.readers import reader_copernicusmarine
from opendrift.readers import reader_netCDF_CF_generic

# All outputs go in ./output relative to this script.
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, 'output')
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Create the Leeway SAR model
# ---------------------------------------------------------------------------
o = Leeway(loglevel=20)  # 20 = INFO; use 0 for full debug, 50 for quiet

# ---------------------------------------------------------------------------
# 2. Forcing (environment): wind + surface current + a coastline
# ---------------------------------------------------------------------------
# Realistic landmask so particles strand on the Israeli coast instead of
# drifting through it.
o.add_reader(reader_global_landmask.Reader())

# Constant forcing for a self-contained demo:
#   wind blowing TOWARD the east-south-east at ~5 m/s (typical sea breeze
#   pushing things toward the Haifa/Carmel coast), weak northward current.
# x_wind/y_wind are eastward/northward components in m/s.
# x_sea_water_velocity/y_sea_water_velocity are eastward/northward in m/s.
currents = reader_copernicusmarine.Reader(
    'cmems_mod_med_phy-cur_anfc_4.2km-2D_PT1H-m',
    username='yairtheop1@gmail.com',
    password='6dewk7ymVUj*j4h'
)
o.add_reader(currents)

gfs_url = 'https://thredds.ucar.edu/thredds/dodsC/grib/NCEP/GFS/Global_0p25deg/Best'

wind_reader = reader_netCDF_CF_generic.Reader(
    gfs_url,
    standard_name_mapping={
        'u-component_of_wind_height_above_ground': 'x_wind',
        'v-component_of_wind_height_above_ground': 'y_wind'
    }
)

o.add_reader(wind_reader)
# Stop particles when they hit the coast (typical for SAR planning).
o.set_config('general:coastline_action', 'stranding')

# ---------------------------------------------------------------------------
# 3. Seed the search object off Haifa
# ---------------------------------------------------------------------------
# Haifa port is ~32.82 N, 34.99 E. We seed a cluster a few km OFFSHORE
# (west of the city, in the bay) to represent the last-known-position (LKP).
haifa_lon = 34.92      # ~7 km west of the port, in open water
haifa_lat = 32.83
radius_m  = 1000       # uncertainty radius of the last known position (m)

# Leeway object categories -- pick the one matching the casualty.
# 27 = "Person-in-water (PIW), vertical / scuba suit"  is a common PIW class.
# Run  o.list_categories()  to see all categories and their ID numbers.
object_type = 27

start_time = datetime.now()

o.seed_elements(
    lon=haifa_lon,
    lat=haifa_lat,
    radius=radius_m,
    number=2000,               # ensemble of 2000 particles for a probability map
    time=start_time,
    object_type=object_type,
)

# ---------------------------------------------------------------------------
# 4. Run the drift simulation
# ---------------------------------------------------------------------------
o.run(
    duration=timedelta(hours=12),   # how far ahead to predict
    time_step=600,                  # integration step: 600 s = 10 min
    time_step_output=1800,          # save output every 30 min
    outfile=os.path.join(OUTDIR, 'haifa_sar.nc'),  # NetCDF trajectory output
)

print(o)

# ---------------------------------------------------------------------------
# 5. Visualise
# ---------------------------------------------------------------------------
# Static map of all trajectories + final positions:
o.plot(fast=True, filename=os.path.join(OUTDIR, 'haifa_sar.png'))

# Animation (drift over time) -- comment out if you only want the still image:
o.animation(fast=True, filename=os.path.join(OUTDIR, 'haifa_sar.gif'))

print("\nDone. Outputs written to:", OUTDIR)
print("  haifa_sar.nc   - trajectories (NetCDF)")
print("  haifa_sar.png  - static map")
print("  haifa_sar.gif  - animation")

# ===========================================================================
# OPERATIONAL VERSION -- real forecast data instead of constant forcing
# ===========================================================================
# Replace the reader_constant block above with real Copernicus Marine
# currents + a wind product. OpenDrift is already installed; you then need a
# free Copernicus Marine account (https://marine.copernicus.eu) and run once:
#     copernicusmarine login
#
# Then:
#
#   from opendrift.readers import reader_netCDF_CF_generic
#   from opendrift.readers import reader_copernicusmarine
#
#   # Mediterranean physics analysis/forecast (hourly surface currents):
#   currents = reader_copernicusmarine.Reader(
#       'cmems_mod_med_phy-cur_anfc_4.2km_PT1H-m')
#   o.add_reader(currents)
#
#   # Wind: e.g. a downloaded GFS/ERA5 NetCDF, or a THREDDS URL:
#   wind = reader_netCDF_CF_generic.Reader('https://...your_wind_file_or_url')
#   o.add_reader(wind)
#
# Everything else (seeding, run, plotting) stays the same. With real data the
# trajectory and the stranding pattern on the Haifa coast become meaningful
# for actual search-area planning.
