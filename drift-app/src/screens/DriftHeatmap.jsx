import React, { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { useIncident } from '../context/IncidentContext';
import { MapContainer, TileLayer, Marker, useMap, Tooltip } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet.heat';

const customIcon = new L.Icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41]
});

function HeatmapLayer({ points, gradient, radius = 18, blur = 18 }) {
  const map = useMap();
  React.useEffect(() => {
    if (!points || points.length === 0) return;
    const heat = L.heatLayer(points, {
      radius,
      blur,
      maxZoom: 13,
      gradient: gradient || { 0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1.0: 'red' }
    }).addTo(map);
    return () => {
      map.removeLayer(heat);
    };
  }, [map, points, gradient, radius, blur]);
  return null;
}

// Heatmap frames now come from the real DrownedBodyDrift simulation, exported
// by test/sim_drowned_body.py to public/drift_data.json (one frame per hour).

function MapTooltip() {
  const map = useMap();
  React.useEffect(() => {
    if (!map) return;
    const info = L.control({ position: 'bottomright' });
    info.onAdd = function () {
      this._div = L.DomUtil.create('div', 'bg-white p-3 border border-[#E2E8F0] shadow-md text-sm font-medium text-[#0F172A]');
      this.update();
      return this._div;
    };
    info.update = function () {
      this._div.innerHTML = 'Zoom in and hover over heatmap for probability estimation';
    };
    info.addTo(map);
    return () => info.remove();
  }, [map]);
  return null;
}

export default function DriftHeatmap() {
  const navigate = useNavigate();
  const { incidentData, driftData, setDriftData, setCurrentHour, setRunState } = useIncident();

  const [loadError, setLoadError] = useState(null);

  // We've arrived at the results -> clear the run's `done` flag so navigating
  // back to the Incident Report later isn't instantly redirected here again.
  React.useEffect(() => {
    setRunState((s) => (s.done ? { ...s, done: false } : s));
  }, [setRunState]);

  // Prefer the live result from the run we just triggered (context). If we
  // landed here without a run, pull the latest from the API, then fall back to
  // the static file so the screen still works with no backend.
  React.useEffect(() => {
    if (driftData) return;
    fetch(`/api/drift_data?t=${Date.now()}`)
      .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
      .then(setDriftData)
      .catch(() =>
        fetch(`${import.meta.env.BASE_URL}drift_data.json`)
          .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
          .then(setDriftData)
          .catch((e) => setLoadError(e.message)));
  }, [driftData, setDriftData]);

  const frames = driftData?.frames ?? [];
  const maxHour = frames.length ? frames.length - 1 : 168;

  const [timeOffset, setTimeOffset] = useState(0);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [showShore, setShowShore] = useState(true);

  // keep the slider within the available frames once data loads
  React.useEffect(() => {
    if (timeOffset > maxHour) setTimeOffset(maxHour);
  }, [maxHour]); // eslint-disable-line react-hooks/exhaustive-deps

  // publish the hour shown here so the Search Plan screen plans for THIS time
  React.useEffect(() => {
    setCurrentHour(timeOffset);
  }, [timeOffset, setCurrentHour]);

  const frame = frames[timeOffset] ?? null;

  const mapCenter = driftData?.lkp
    ? [driftData.lkp.lat, driftData.lkp.lon]
    : (incidentData.lat && incidentData.lng)
      ? [parseFloat(incidentData.lat), parseFloat(incidentData.lng)]
      : [32.82, 34.99];

  const heatmapPoints = useMemo(() => frame?.points ?? [], [frame]);
  // beached-body locations, accumulated up to this hour (carried forward by the sim)
  const shorePoints = useMemo(
    () => (frame?.shore ?? []).map(([la, lo]) => [la, lo, 0.8]), [frame]);

  return (
    <div className="flex-1 flex overflow-hidden">
      
      {/* Left Sidebar */}
      <div className="w-[320px] bg-white border-r border-[#E2E8F0] flex flex-col shrink-0 overflow-y-auto">
        
        <div className="p-4 border-b border-[#E2E8F0]">
          <div className="flex justify-between items-start mb-2">
            <h2 className="font-bold text-[#0F172A]">{incidentData.id}</h2>
            <span className="text-[10px] uppercase font-bold tracking-wider text-[#DC2626] bg-red-50 px-1 border border-red-200">
              ACTIVE
            </span>
          </div>
          <p className="text-xs text-[#64748B]">LSP: {incidentData.lat || '--'}, {incidentData.lng || '--'}</p>
        </div>

        <div className="p-4 border-b border-[#E2E8F0]">
          <h3 className="text-sm font-semibold text-[#0F172A] mb-3">Probability Distribution</h3>
          <div className="flex h-3 w-full rounded-sm mb-2" style={{ background: 'linear-gradient(to right, blue, lime, orange, red)' }}></div>
          <div className="flex justify-between text-xs text-[#64748B] mb-2 font-medium">
            <span>Low</span>
            <span>Medium</span>
            <span>High</span>
            <span className="text-[#DC2626]">Critical</span>
          </div>
        </div>

        <div className="p-4 border-b border-[#E2E8F0]">
          <div className="flex justify-between text-sm font-semibold text-[#0F172A] mb-1">
            <span>Model time:</span>
            <span className="text-[#0F766E]">T+{timeOffset}h</span>
          </div>
          {frame?.label && (
            <p className="text-xs text-[#64748B] mb-3">{frame.label} UTC</p>
          )}
          <input
            type="range"
            min="0" max={maxHour} step="1"
            value={timeOffset}
            onChange={(e) => setTimeOffset(parseInt(e.target.value))}
            className="w-full slider-grab"
          />
          {frame && (
            <div className="mt-3 text-xs space-y-1">
              <div className="flex justify-between"><span className="text-[#2563EB] font-medium">Afloat (surface)</span><span className="font-semibold">{frame.afloat}%</span></div>
              <div className="flex justify-between"><span className="text-[#DC2626] font-medium">Submerged</span><span className="font-semibold">{frame.submerged}%</span></div>
              {frame.stranded > 0.05 && (
                <div className="flex justify-between"><span className="text-[#64748B] font-medium">Stranded</span><span className="font-semibold">{frame.stranded}%</span></div>
              )}
            </div>
          )}
          {!driftData && !loadError && (
            <p className="text-xs text-[#64748B] mt-3">Loading simulation…</p>
          )}
          {loadError && (
            <p className="text-xs text-[#DC2626] mt-3">No sim data ({loadError}). Run sim_drowned_body.py.</p>
          )}
        </div>

        <div className="flex-1"></div>

        {/* Generate Plan Button */}
        <div className="p-4 mt-auto">
          <button 
            onClick={() => navigate('/search-plan')}
            className="w-full py-3 bg-white border-2 border-[#0F766E] text-[#0F766E] font-medium hover:bg-[#F0FDFA] transition-colors"
          >
            Switch to Search Plan →
          </button>
        </div>
      </div>

      {/* Right Map Panel */}
      <div className="flex-1 relative bg-blue-50">

        <MapContainer center={mapCenter} zoom={12} className="w-full h-full" zoomControl={false}>
          <TileLayer
            attribution='&copy; <a href="https://carto.com/attributions">CARTO</a>'
            url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
          />

          {/* LKP marker — prefer the position the simulation actually used */}
          {driftData?.lkp ? (
            <Marker position={[driftData.lkp.lat, driftData.lkp.lon]} icon={customIcon}>
              <Tooltip permanent direction="top" offset={[0, -36]}>LKP</Tooltip>
            </Marker>
          ) : (incidentData.lat && incidentData.lng && (
            <Marker position={[parseFloat(incidentData.lat), parseFloat(incidentData.lng)]} icon={customIcon} />
          ))}

          {/* Realistic Heatmap */}
          {showHeatmap && <HeatmapLayer points={heatmapPoints} />}

          {/* Beached bodies (stranded on shore) — distinct dark/brown layer */}
          {showShore && (
            <HeatmapLayer
              points={shorePoints}
              gradient={{ 0.2: '#fbbf24', 0.5: '#b45309', 1.0: '#1c1917' }}
              radius={12}
              blur={9}
            />
          )}

          <MapTooltip />
        </MapContainer>

        {/* Map Overlays & Controls */}
        <div className="absolute top-4 right-4 z-[400] flex flex-col gap-2 mt-10">
          <button 
            onClick={() => setShowHeatmap(!showHeatmap)}
            className={`px-3 py-1.5 text-xs font-semibold shadow-sm border ${showHeatmap ? 'bg-white border-[#0F766E] text-[#0F766E]' : 'bg-gray-100 border-[#E2E8F0] text-[#64748B]'}`}
          >
            Heatmap: {showHeatmap ? 'ON' : 'OFF'}
          </button>
          <button
            onClick={() => setShowShore(!showShore)}
            title="Bodies washed ashore (stranded particles), accumulated over time"
            className={`px-3 py-1.5 text-xs font-semibold shadow-sm border ${showShore ? 'bg-white border-[#92400e] text-[#92400e]' : 'bg-gray-100 border-[#E2E8F0] text-[#64748B]'}`}
          >
            Shore: {showShore ? 'ON' : 'OFF'}
          </button>
        </div>
      </div>
    </div>
  );
}
