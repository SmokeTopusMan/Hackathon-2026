# DeepDrift

A 3D marine drift simulation engine designed to track and predict the trajectory of submerged bodies. DeepDrift ingests oceanographic data to calculate environmental forces across depth layers, utilizing Monte Carlo simulations to generate probabilistic heatmaps of an object's location over time.

## Core Features

* **Subsurface Kinematics:** Calculates vector sums of deep-water currents without surface interference (wind leeway/Stokes drift).
* **Dynamic Buoyancy:** Evaluates water density profiles (temperature and salinity) to determine vertical movement and neutral buoyancy resting layers (pycnoclines).
* **Bathymetric Grounding:** Includes sea floor collision detection and bottom friction calculations.
* **Probability Heatmapping:** Uses Monte Carlo random walk algorithms to account for turbulence, temporal diffusion, and initial position uncertainty.

## Data Dependencies

To run accurate simulations, DeepDrift requires the following continuous data streams:
* **3D Hydrodynamic Models:** Current speed and direction mapped across depth intervals (e.g., NetCDF/GRIB formats).
* **Oceanographic Density Profiles:** Thermocline and halocline data.
* **High-Resolution Bathymetry:** Topographical seafloor grids.
* **Object Physics:** Mass, volume, drag coefficients, and estimated initial state vectors (XYZ + error margin).

## Installation

```bash
git clone [https://github.com/yourusername/DeepDrift.git](https://github.com/yourusername/DeepDrift.git)
cd DeepDrift
pip install -r requirements.txt
