<div align="center">

![Project Logo](hacklogo.png)

# נחשון · Nahshon

### Probabilistic drift modeling & coordinated search planning for maritime SAR

*"The first to step into the sea."*

When someone is lost beneath the waves, every hour and every square meter matters.  
**Nahshon** turns real ocean physics into a living heatmap — and tells each rescue team exactly where to search next.

<br>

![Status](https://img.shields.io/badge/status-prototype-orange?style=flat-square)
![Domain](https://img.shields.io/badge/domain-search%20%26%20rescue-0a7ea4?style=flat-square)
![Engine](https://img.shields.io/badge/drift%20engine-OpenDrift-1f6feb?style=flat-square)
![Data](https://img.shields.io/badge/ocean%20data-Copernicus%20Marine-00b4d8?style=flat-square)
![Frontend](https://img.shields.io/badge/frontend-React%2019%20%2B%20Leaflet-61dafb?style=flat-square)
![Backend](https://img.shields.io/badge/backend-Flask-black?style=flat-square)

</div>

---

## What it does

Most drift tools model floating debris or life rafts. **Nahshon is built for the hardest case** — a submerged body near the Israeli coast — where wind no longer applies and the only forces are slow, hidden subsurface currents.

Given a last-known position and a victim profile, Nahshon:

1. **Simulates** 10,000 virtual particles through real 3D ocean currents using a physics-based 4-phase body model (drowning → submerged → refloat → surface drift)
2. **Renders** a time-evolving probability heatmap over the next 7 days, frame by frame
3. **Plans** optimally coordinated search paths for a heterogeneous fleet (boats, jet-skis) that maximizes probability cleared per hour

All of this runs end-to-end from a single incident form in a web browser.

---

## System architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        React Frontend (Vite)                    │
│                                                                 │
│  IncidentReport ──▶ DriftHeatmap (time-slider) ──▶ SearchPlan  │
│     form input         heatmap + shore overlay    fleet paths   │
└────────────────────────────┬────────────────────────────────────┘
                             │  POST /api/simulate
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                        Flask API  (server.py)                    │
│  /api/simulate  ·  /api/progress  ·  /api/drift_data  ·  /api/plan│
└──────────────────────────────┬───────────────────────────────────┘
                               │  subprocess
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                  Simulation Orchestrator                          │
│                  (test/sim_drowned_body.py)                       │
│                                                                  │
│  1. Download Copernicus Marine 3D currents + Stokes drift        │
│  2. Run DrownedBodyDrift  (core/drowned_drift.py)                │
│     └─ OpenDrift subclass · 10k particles · 15-min timesteps     │
│  3. Bin particle cloud → hourly heatmap frames (300 m cells)     │
│  4. Run CoveragePlanner  (core/search_planner.py)                │
│     └─ Greedy submodular · heterogeneous fleet · real SI units   │
│  5. Export  drift_data.json  →  drift-app/public/               │
└──────────────────────────────────────────────────────────────────┘
```

---

## Physics: the 4-phase body model

`core/drowned_drift.py` implements a custom OpenDrift subclass where each of the 10,000 particles independently cycles through four phases:

```
  Phase 1 · DROWNING     (0 – 30 min, stochastic)
  ─────────────────────────────────────────────────
  Particle is at the surface, drifting with
  70% Stokes wave drift + 30% subsurface current.
  Sinking begins after a random delay.

       │
       ▼ sink at 0.3 m/s

  Phase 0 · SUBMERGED
  ─────────────────────────────────────────────────
  Particle rests on the seabed or drifts slowly
  with 3D bottom currents. Beaching disabled —
  submerged bodies roll past the shore.

       │
       ▼ refloat after ~100–140 Accumulated Degree Days
         (temperature-driven forensic model)

  Phase 2 · RISING       (rise at ~0.2 m/s)
  ─────────────────────────────────────────────────
  Gas produced by decomposition overcomes body
  density, lifting it toward the surface.

       │
       ▼

  Phase 3 · AT SURFACE   (2 – 5 days, then re-sinks)
  ─────────────────────────────────────────────────
  Floating body drifts with wave dominance.
  Can beach on the shoreline.
```

**Body dynamics** are victim-specific: height/weight → BMI → fat fraction (Deurenberg formula) → body density (Siri model) → sink/rise speeds clamped to physically plausible ranges.

---

## Search planning

`core/search_planner.py` runs a **coordinated greedy coverage planner** across the heatmap:

- **Algorithm**: Sequential submodular greedy — each agent commits its path in turn, conditioning the next agent's choices on what was already cleared. Achieves ~63%+ of the optimal joint solution.
- **Movement**: 8-connected grid (N/NE/E/.../stay), hard sonar clearing (scanned cells → 0), real distance budget per vehicle.
- **Fleet**: `Boat` (7 m/s, 80 m sonar swath) and `Jet-ski` (14 m/s, 40 m swath) with per-vehicle paths, time, and distance outputs in SI units.
- **Reference grid**: Auto-generated comms grid (rows A–Z, columns 1–N) at 500 m cells for radio coordination.
- **Convergence**: Planning stops when coverage target is hit, marginal gain saturates, or max time is exceeded — not at a fixed step count.

Users can click the map in the frontend to place additional vehicles; the API replans in real time via `POST /api/plan`.

---

## Tech stack

| Layer | Technology |
|---|---|
| **Drift physics** | [OpenDrift](https://github.com/OpenDrift/opendrift) — custom `DrownedBodyDrift` subclass |
| **Ocean data** | [Copernicus Marine (CMEMS)](https://marine.copernicus.eu/) — 3D currents + wave Stokes drift via `xarray` / NetCDF |
| **Scientific computing** | NumPy · SciPy · scikit-learn |
| **Backend API** | Flask + Flask-CORS |
| **Frontend** | React 19 · Vite 8 · React Router 7 · Tailwind CSS 4 |
| **Map & heatmap** | Leaflet · leaflet.heat · react-leaflet |
| **Visualization (dev)** | Folium · Cartopy · Matplotlib |

---

## Running the project

### Prerequisites

```bash
# Python (simulation + API)
pip install opendrift flask flask-cors numpy scipy scikit-learn xarray netcdf4

# Node (frontend)
cd drift-app && npm install
```

### Start the API

```bash
cd api
python server.py
# Listening on http://localhost:5000
```

### Start the frontend

```bash
cd drift-app
npm run dev
# Opens http://localhost:5173
```

Fill in the incident form, submit, and watch the simulation run. The heatmap and search plan update automatically when processing completes.

---

## Repository layout

```
Hackathon-2026/
├── core/
│   ├── drowned_drift.py        # Physics: 4-phase body model (OpenDrift subclass)
│   ├── search_planner.py       # Greedy coordinated coverage planner
│   └── joint_search_planner.py # Brute-force heading planner (experimental)
├── test/
│   └── sim_drowned_body.py     # Simulation orchestrator + global config
├── api/
│   └── server.py               # Flask API (simulate / progress / drift_data / plan)
├── drift-app/
│   └── src/
│       ├── screens/
│       │   ├── IncidentReport.jsx  # Incident form + sim progress bar
│       │   ├── DriftHeatmap.jsx    # Time-slider heatmap + shore overlay
│       │   └── SearchPlan.jsx      # Animated fleet deployment + user vehicle placement
│       └── context/
│           └── IncidentContext.jsx # Global state: polling, drift data, plan fetching
└── evidence/                   # Bayesian fusion layer (planned, not yet implemented)
```

---

## What's working · what's next

| Feature | Status |
|---|---|
| 4-phase body drift physics | ✅ |
| 10k-particle Monte Carlo ensemble | ✅ |
| Real Copernicus Marine ocean data | ✅ |
| Time-evolving probability heatmap | ✅ |
| Heterogeneous fleet search planning | ✅ |
| User vehicle placement + live replanning | ✅ |
| Shore-stranding detection & overlay | ✅ |
| Bayesian sonar-miss fusion | 🔲 planned |
| Eyewitness / local-knowledge soft priors | 🔲 planned |
| Wind drift (ERA5/GFS integration) | 🔲 planned |
| Field validation with rescue teams | 🔲 planned |

---

## Why physics, not neural networks?

The physics of how subsurface currents move objects is well understood — and real drowning cases with verified recovery locations are rare. There isn't enough labeled data to relearn hydrodynamics from scratch, and a physics model generalizes to conditions it has never seen. **Nahshon keeps the physics** and reserves machine learning for where data actually accumulates: sonar detection-probability calibration and local current bias correction.

---

## Built for the Eastern Mediterranean

- **Microtidal sea** — the Mediterranean has negligible tides, sidestepping the tidal-reversal complexity of Atlantic SAR models
- **Wave dominance** — afloat phases use 70% Stokes / 30% current weighting, tuned for Israeli coastal dynamics
- **Regional data pipeline** — ingests Copernicus Marine CMEMS subsets for the Eastern Med at runtime

---

## ⚠️ Disclaimer

Nahshon is a **decision-support tool**, not a replacement for trained SAR judgment. Its predictions are probabilistic and depend on input data quality. All search decisions remain with qualified rescue professionals. This is an early-stage prototype, not yet validated for operational use.

---

<div align="center">

*Built to bring people home from the sea.*

</div>
