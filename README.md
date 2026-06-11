<div align="center">

# 🌊 Nahshon · נחשון

### Probabilistic drift modeling for maritime search & rescue

*"The first to step into the sea."*

When someone is lost beneath the waves, every hour and every square meter matters.
**Nahshon** turns ocean physics, sonar sweeps, and local knowledge into a single, living
probability map — so rescue teams know where to look **next**.

<br>

![Status](https://img.shields.io/badge/status-early%20development-orange)
![Domain](https://img.shields.io/badge/domain-search%20%26%20rescue-0a7ea4)
![Engine](https://img.shields.io/badge/drift%20engine-OpenDrift-1f6feb)
![Method](https://img.shields.io/badge/method-Bayesian%20search%20theory-6f42c1)
![License](https://img.shields.io/badge/license-TBD-lightgrey)

</div>

---

## The mission

Most drift models are built for objects floating on the **surface** — life rafts, debris, a person in a life jacket. Nahshon is built for the hardest and most painful case: a **submerged body near the shore**, where wind and waves no longer apply and the only forces are the slow, hidden currents beneath the surface.

Our goal is to give the volunteers and professionals who search the Israeli coastline a tool that doesn't just guess — it **learns from every sweep** and points them to the most likely place to search next.

---

## How it works

Nahshon is not one model. It's a **physics engine** that produces a prior, wrapped in a **Bayesian fusion layer** that folds in everything else you know.

```
                    ┌─────────────────────────────────────┐
   Last-known pos ─▶│  PHYSICS  ·  OpenDrift (OceanDrift)  │
   + time           │  • 3D subsurface currents            │──▶  Prior heatmap
   + uncertainty    │  • sinking velocity / buoyancy       │     (where could it be?)
                    │  • seabed interaction & refloat       │
                    └─────────────────────────────────────┘
                                      │
                                      ▼
   👁  Eyewitness    ┌─────────────────────────────────────┐
   🎣 Fisherman ────▶│  FUSION  ·  Bayesian reweighting     │──▶  Posterior heatmap
   📡 Sonar sweeps   │  • soft priors from local knowledge  │     (where to look next)
      (incl. misses) │  • negative information from sonar    │
                    └─────────────────────────────────────┘
                                      │
                                      ▼
                        🎯  Next-search recommendation
                        (loops back as new data arrives)
```

### 1. The physics gives us a prior
A Monte Carlo ensemble of thousands of virtual particles is released from the last-known position, each carrying realistic uncertainty in **position, time, and currents**. Driven by 3D ocean data, they spread into a cloud — the prior probability of where the body could be.

### 2. Every observation updates the map
- **Eyewitness & local fishermen** — a sighting, or a local's knowledge of how the current pulls toward a particular cove, becomes a soft likelihood that gently reshapes the map.
- **Sonar sweeps** — and crucially, **a sweep that finds nothing is information too.** Searched areas are down-weighted by the sonar's probability of detection (never to zero — sonar misses things), and that probability flows to where the target is more likely to be.

### 3. The map tells you where to search next
The result is a posterior heatmap that gets **sharper with every sweep**, closing the loop between the searchers in the water and the model on shore.

---

## Why not just train a neural network?

Because the physics of how water moves is already well understood — and real cases are rare. There isn't enough labeled data to relearn hydrodynamics from scratch, and a physics model generalizes to conditions it has never seen. So Nahshon **keeps the physics** and applies machine learning only where data actually accumulates: sonar detection-probability models and local current bias correction.

---

## Built for the Eastern Mediterranean

- **Microtidal sea** — the Mediterranean has tiny tides, so we sidestep the tidal-reversal problems that complicate Atlantic SAR.
- **Regional ocean data** — designed to ingest high-resolution Eastern-Med current fields (ISRAMAR / Copernicus Marine).
- **Coastline-aware** — particles strand realistically on the shore via land masking.

---

## Tech stack

| Layer | Tooling |
|---|---|
| Drift physics | [OpenDrift](https://github.com/OpenDrift/opendrift) (`OceanDrift`) |
| Ocean / current data | Copernicus Marine (CMEMS), ISRAMAR, NetCDF via `xarray` |
| Fusion layer | Bayesian particle reweighting (Python) |
| Heatmap density | `scipy` / `scikit-learn` KDE |
| Visualization | `folium` / `leaflet`, `cartopy`, `matplotlib` |

---

## Roadmap

- [ ] OpenDrift `OceanDrift` submerged-body prior (sinking + seabed + refloat)
- [ ] Bayesian particle-reweighting fusion layer
- [ ] Sonar sweep ingestion with probability-of-detection model
- [ ] Eyewitness & local-knowledge soft priors
- [ ] Interactive web heatmap + next-search recommendations
- [ ] Real Eastern-Med current data integration (ISRAMAR / CMEMS)
- [ ] Field validation with rescue teams

---

## ⚠️ Disclaimer

Nahshon is a **decision-support tool**, not a replacement for trained search-and-rescue judgment. Its predictions are probabilistic and depend on the quality of input data. **All search decisions remain with qualified rescue professionals.** This project is under active development and is not yet validated for operational use.

---

<div align="center">

*Built to bring people home from the sea.* 🕊️

</div>
