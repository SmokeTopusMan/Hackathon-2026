<div align="center">

![Nahshol](hacklogo.png)

# 🌊 Nahshol · נחשול

### Drift simulation & coordinated search-route planning for maritime rescue

*Nahshol (נחשול) — "a surging swell of the sea."*

When someone goes under near the shore, every hour and every square metre matters.
**Nahshol** simulates where a drowned body drifts beneath the surface — hour by hour —
and then plans the routes a rescue fleet should sweep to clear the most probable water
in the least time.

<br>

![Status](https://img.shields.io/badge/status-POC%20·%20demo--ready-2ea043)
![Domain](https://img.shields.io/badge/domain-maritime%20rescue-1e5c9e)
![Drift engine](https://img.shields.io/badge/drift%20engine-OpenDrift%20OceanDrift-1e5c9e)
![Planner](https://img.shields.io/badge/planner-coordinated%20coverage-1e5c9e)
![Stack](https://img.shields.io/badge/stack-React%20%2B%20Flask-13366a)

</div>

---

## What Nahshol does

Nahshol is a **decision-support tool** for drowning incidents on the Israeli coast. It does
two things, end to end, live from the operator's inputs:

1. **Simulates the drift** of a submerged body from the last-known point — a physics
   ensemble that produces an hour-by-hour probability heatmap of where the body is likely
   to be.
2. **Plans the search** — routes a fleet of rescue craft (jet-skis, boats) so they cover
   the most probable water as fast as possible, and reports how much probability they clear.

It is **not** a generic surface-drift model. It is built for the hardest case: a body
**below the surface**, where wind and waves no longer apply and only the slow currents at
depth move it.

---

## The operator flow

Three screens, left to right — the way an operator actually uses it:

| Step | Screen | What happens |
|---|---|---|
| **1** | **Incident Report** | Enter the last-seen point (type coordinates or click the map), time, and victim profile (height / weight / age). Press **Run Simulation** — the drift model runs live with a progress bar. |
| **2** | **Drift Heatmap** | Scrub an hour-by-hour slider over the probability heatmap. See the split between **afloat / submerged / stranded**, and bodies accumulating on the shore. |
| **3** | **Search Plan** | A coordinated plan appears over the chosen hour: teams launch from shore and sweep over water only. Drop your own vehicles to re-plan from real launch points. Read the **probability cleared** and mission time, and replay the drift + search as an animation. |

---

## How it works

### 1 · The physics — `DrownedBodyDrift`

An [OpenDrift](https://github.com/OpenDrift/opendrift) `OceanDrift` subclass that models the
full **sink → rest-on-bottom → refloat → re-sink** cycle of a drowned body. OceanDrift gives
the 3D current advection *at the body's own depth*; Nahshol adds the vertical behaviour:

- A **state machine** per particle — drowning → submerged → rising → at-surface — so a body
  can surface, drift, and re-sink several times across a long search window (exactly what SAR
  teams report).
- **Body-size physics** — the victim's height and weight set buoyancy and timing
  (BMI → body-fat → density). Leaner bodies sink faster and refloat later; heavier-set bodies
  bob back up sooner.
- A **Monte Carlo ensemble** of thousands of particles released from the last-known point with
  realistic uncertainty in position and time → the probability heatmap.

### 2 · The planner — coordinated coverage

Heterogeneous multi-agent **probabilistic coverage path planning** over the heatmap:

- **Real units** — each craft carries a real speed (m/s) and sonar radius (m); a fast jet-ski
  covers more water per minute than a slow boat, and route length / ETA come out in metres and
  minutes.
- **Coordinated greedy** — the coverage objective is monotone-submodular, so a greedy rule is
  provably within (1 − 1/e) ≈ 63% of optimal; agents decide sequentially and commit scanned
  cells so they spread out without colliding.
- **Shore-aware** — teams launch from land and sweep over water only; a labelled reference grid
  (rows A, B, C… / columns 1, 2, 3…) gives crews a shared "B-4" map language.
- **Converges** — planning stops when target coverage is reached or improvement saturates, so
  routes don't wander once the area is effectively cleared.

---

## Architecture

```
  React UI (Vite + Leaflet)                  Python backend (Flask)
  ┌───────────────────────┐   POST /api/simulate   ┌──────────────────────────────┐
  │ Incident Report  ─────────────────────────────▶│ background run:               │
  │ Drift Heatmap    ◀──── GET /api/drift_data ─────│  test/sim_drowned_body.py     │
  │ Search Plan      ◀──── GET /api/plan?hour=H ────│   • core/drowned_drift.py     │  physics
  └───────────────────────┘   GET /api/progress/<id>│   • core/search_planner.py    │  planner
                                                    └──────────────────────────────┘
```

| Layer | Tooling |
|---|---|
| Frontend | React + Vite + React-Leaflet, Tailwind |
| Backend | Flask (`api/server.py`) — runs each simulation as an isolated subprocess |
| Drift physics | OpenDrift `OceanDrift` subclass (`core/drowned_drift.py`) |
| Search planner | Coordinated greedy coverage (`core/search_planner.py`) |
| Ocean / wave data | Copernicus Marine (CMEMS); offline mode for fast demos |

---

## Run it

Two servers:

```bash
# 1) Backend — Flask on :5000
python api/server.py

# 2) Frontend — Vite on :5173 (proxies /api → :5000)
cd drift-app
npm install
npm run dev
```

**Fast offline demo** (no Copernicus download — great for a presentation):

```bash
# constant forcing, smaller ensemble → runs in seconds
NAHSHON_SOURCE=offline NAHSHON_PARTICLES=1200 python api/server.py
```

A live Copernicus run (the default) downloads a current/wave subset and takes a few minutes.

---

## Roadmap

What's built today is the **simulate → plan** loop above. Next:

- [ ] **Bayesian fusion layer** — fold eyewitness sightings and local fishermen's knowledge into the prior as soft likelihoods.
- [ ] **Sonar-sweep ingestion** — treat a sweep that finds *nothing* as information too (down-weight searched water by probability of detection).
- [ ] **Water-temperature** in the refloat-timing model (the dominant real driver).
- [ ] Move CMEMS credentials out of source into env / login.
- [ ] Field validation with rescue teams.

---

## ⚠️ Disclaimer

Nahshol is a **decision-support tool**, not a replacement for trained search-and-rescue
judgement. Its predictions are probabilistic and depend on the quality of input data.
**All search decisions remain with qualified rescue professionals.** Under active
development; not yet validated for operational use.

---

<div align="center">

*Built to bring people home from the sea.* 🕊️

</div>
