import React, { useState, useEffect, useMemo, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useIncident } from '../context/IncidentContext';
import { MapContainer, TileLayer, Marker, Polyline, Tooltip, CircleMarker, Circle, useMap, useMapEvents } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet.heat';

const lspIcon = new L.Icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
  iconSize: [25, 41], iconAnchor: [12, 41]
});

function HeatmapLayer({ points }) {
  const map = useMap();
  React.useEffect(() => {
    if (!points || points.length === 0) return;
    const heat = L.heatLayer(points, {
      radius: 18, blur: 18, maxZoom: 13,
      gradient: { 0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1.0: 'red' }
    }).addTo(map);
    return () => map.removeLayer(heat);
  }, [map, points]);
  return null;
}

// drop consecutive duplicate waypoints (an agent that "stays" repeats a cell)
function dedupe(waypoints) {
  return waypoints.filter((p, i) => i === 0 ||
    p[0] !== waypoints[i - 1][0] || p[1] !== waypoints[i - 1][1]);
}

// small text label for the reference grid (row letters / column numbers)
function gridLabelIcon(text) {
  return L.divIcon({
    className: 'ref-grid-label',
    html: `<div style="font:700 11px system-ui;color:#334155;text-shadow:0 0 2px #fff,0 0 2px #fff">${text}</div>`,
    iconSize: [16, 16], iconAnchor: [8, 8],
  });
}

// the coarse labelled "comms" grid (rows A,B,C.. from north, cols 1,2,3.. west)
function ReferenceGrid({ grid }) {
  if (!grid) return null;
  const xs = grid.lon_edges, ys = grid.lat_edges;        // ys: north -> south
  const top = ys[0], bot = ys[ys.length - 1], left = xs[0], right = xs[xs.length - 1];
  const opts = { color: '#334155', weight: 0.8, opacity: 0.4 };
  return (
    <>
      {xs.map((x, i) => <Polyline key={`v${i}`} positions={[[top, x], [bot, x]]} pathOptions={opts} interactive={false} />)}
      {ys.map((y, i) => <Polyline key={`h${i}`} positions={[[y, left], [y, right]]} pathOptions={opts} interactive={false} />)}
      {grid.row_labels.map((lab, i) => (
        <Marker key={`r${i}`} position={[(ys[i] + ys[i + 1]) / 2, left]} icon={gridLabelIcon(lab)} interactive={false} />
      ))}
      {grid.col_labels.map((lab, j) => (
        <Marker key={`c${j}`} position={[top, (xs[j] + xs[j + 1]) / 2]} icon={gridLabelIcon(lab)} interactive={false} />
      ))}
    </>
  );
}

function PlanClickHandler({ onMapClick }) {
  useMapEvents({ click(e) { onMapClick(e.latlng); } });
  return null;
}

const VEHICLE_EMOJI = { boat: '🚤' };   // jet-ski uses a generated SVG (no emoji fits)
const VEHICLE_LABEL = { jetski: 'Jet-ski', boat: 'Boat' };

// side-view jet-ski: angled handlebar, sleek hull, water spray
const JETSKI_SVG = '<svg viewBox="0 0 32 24" xmlns="http://www.w3.org/2000/svg" width="100%" height="100%">'
  + '<path d="M19 8 l5 -4" stroke="#0f766e" stroke-width="2.2" stroke-linecap="round" fill="none"/>'
  + '<path d="M3 13 C5 9 9 8 14 8 L21 8 C24 8 25 10 26 13 C24 16 20 17 14 17 C8 17 5 16 3 13 Z" fill="#0f766e"/>'
  + '<path d="M12 8 C13 6 17 6 18 8 Z" fill="#0f766e"/>'
  + '<path d="M2 20 q3 2 6 0 t6 0 t6 0 t6 0" stroke="#38bdf8" stroke-width="1.6" fill="none" stroke-linecap="round"/>'
  + '</svg>';

function VehicleGlyph({ type, size = 18 }) {
  if (type === 'jetski') {
    return <span style={{ display: 'inline-block', width: Math.round(size * 1.35), height: size, lineHeight: 0 }}
      dangerouslySetInnerHTML={{ __html: JETSKI_SVG }} />;
  }
  return <span style={{ fontSize: size, lineHeight: 1 }}>{VEHICLE_EMOJI[type] || '🚤'}</span>;
}

function vehicleIcon(type) {
  const html = type === 'jetski'
    ? `<div style="width:28px;height:21px;line-height:0">${JETSKI_SVG}</div>`
    : `<div style="font-size:22px;line-height:22px">${VEHICLE_EMOJI[type] || '🚤'}</div>`;
  return L.divIcon({ className: 'veh-marker', html, iconSize: [28, 22], iconAnchor: [14, 11] });
}

const pendingIcon = L.divIcon({
  className: 'veh-pending',
  html: '<div style="font-size:22px;line-height:22px">📍</div>',
  iconSize: [24, 24], iconAnchor: [12, 24],
});

export default function SearchPlan() {
  const navigate = useNavigate();
  const { incidentData, driftData, setDriftData, currentHour, setCurrentHour, fetchPlanForHour, fetchPlanForVehicles } = useIncident();

  // make sure we have sim output (in case the user deep-linked here)
  useEffect(() => {
    if (driftData) return;
    fetch(`/api/drift_data?t=${Date.now()}`)
      .then((r) => { if (!r.ok) throw new Error(); return r.json(); })
      .then(setDriftData)
      .catch(() => fetch(`${import.meta.env.BASE_URL}drift_data.json`)
        .then((r) => r.ok ? r.json() : null).then((d) => d && setDriftData(d)));
  }, [driftData, setDriftData]);

  const frames = driftData?.frames ?? [];
  const maxHour = frames.length ? frames.length - 1 : 168;

  const [plan, setPlan] = useState(null);
  const [planError, setPlanError] = useState(null);
  // delay (hours) before the rescue forces actually deploy: the search is planned
  // on the heatmap at SEARCH START = selected hour + this delay.
  const [searchDelayHours, setSearchDelayHours] = useState(0);

  // user-placed fleet: each vehicle is { id, lat, lng, type }
  const [userVehicles, setUserVehicles] = useState([]);
  const [pendingPos, setPendingPos] = useState(null);
  const [planMode, setPlanMode] = useState('auto');   // 'auto' | 'user'
  const [generating, setGenerating] = useState(false);
  const [genProgress, setGenProgress] = useState(0);

  const abortRef = useRef(null);
  const reqIdRef = useRef(0);

  const searchHour = Math.min(currentHour + searchDelayHours, maxHour);

  // Generate a plan. A request token (reqIdRef) makes Cancel authoritative: a
  // cancelled or superseded run is ignored when it finally settles, so the UI
  // never flips back to "generating" and never shows a stale plan.
  const doPlan = async (mode, vehicles, hour) => {
    const myId = ++reqIdRef.current;
    if (abortRef.current) abortRef.current.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setGenerating(true);
    setGenProgress(12);
    setPlanError(null);
    try {
      const p = (mode === 'user' && vehicles.length)
        ? await fetchPlanForVehicles(hour, vehicles, ctrl.signal)
        : await fetchPlanForHour(hour, ctrl.signal);
      if (reqIdRef.current !== myId) return;
      setPlan(p);
      setGenProgress(100);
      setGenerating(false);
    } catch (e) {
      if (reqIdRef.current !== myId) return;
      if (mode !== 'user' && driftData?.search_plan) setPlan(driftData.search_plan);
      else if (e.name !== 'AbortError') setPlanError(e.message);
      setGenerating(false);
    }
  };

  // animate the progress bar while a plan is being formed
  useEffect(() => {
    if (!generating) return;
    const id = setInterval(
      () => setGenProgress((p) => (p < 92 ? p + Math.max(1, (92 - p) * 0.12) : p)), 180);
    return () => clearInterval(id);
  }, [generating]);

  const addVehicle = (type) => {
    if (!pendingPos) return;
    setUserVehicles((v) => [...v, { id: Date.now(), lat: pendingPos.lat, lng: pendingPos.lng, type }]);
    setPendingPos(null);
  };
  const removeVehicle = (id) => setUserVehicles((v) => v.filter((x) => x.id !== id));
  const handleGeneratePlan = () => {
    if (!userVehicles.length || generating) return;
    setPlanMode('user');
    doPlan('user', userVehicles, searchHour);
  };
  const handleCancelPlan = () => {
    reqIdRef.current += 1;
    if (abortRef.current) { abortRef.current.abort(); abortRef.current = null; }
    setGenerating(false);
    setGenProgress(0);
  };
  const handleClearVehicles = () => {
    if (generating) return;
    setUserVehicles([]);
    setPendingPos(null);
    setPlanMode('auto');          // the auto-plan effect below re-plans for us
  };

  // AUTO mode: (re)compute the shore-launch plan on first load and whenever the
  // search-start hour (forecast slider + deploy delay) changes, so the routes
  // always match the heatmap underneath. USER mode is driven by the Generate
  // button instead. doPlan supersedes any in-flight request (reqIdRef/abort),
  // so dragging the slider just cancels the stale fetch.
  useEffect(() => {
    if (planMode !== 'auto' || !driftData) return;
    // doPlan kicks off an async fetch (its setState is the standard
    // fetch-on-change pattern; it can't re-trigger this effect since it touches
    // none of the deps).
    // eslint-disable-next-line react-hooks/set-state-in-effect
    doPlan('auto', [], searchHour);
    // doPlan omitted from deps: stable setters/refs + context fetchers only.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [planMode, searchHour, driftData]);

  const mapCenter = (incidentData.lat && incidentData.lng)
    ? [parseFloat(incidentData.lat), parseFloat(incidentData.lng)]
    : driftData?.lkp ? [driftData.lkp.lat, driftData.lkp.lon] : [32.82, 34.99];

  const teams = plan?.teams ?? [];

  // ---- animation: watch the teams sweep out from shore (algorithm running) --
  // anchor each track at the user's placed launch point (echoed by the backend
  // as `launch`), so the vehicle visibly leaves where it was dropped rather than
  // starting at the snapped grid cell out at sea.
  const teamTracks = useMemo(() => teams.map((t) => {
    const wp = dedupe(t.waypoints);
    return t.launch ? dedupe([t.launch, ...wp]) : wp;
  }), [teams]);
  const maxLen = useMemo(
    () => Math.max(1, ...teamTracks.map((p) => p.length)), [teamTracks]);
  const [animStep, setAnimStep] = useState(1);
  const [playing, setPlaying] = useState(true);

  // ---- two-phase replay timeline ----------------------------------------
  // phase 1 (steps 1..driftSteps): the body drifts hour by hour with the
  // vehicles parked at their launch points; phase 2: the fleet deploys and
  // sweeps along the planned paths over the search-start heatmap.
  const driftSteps = plan ? searchHour : 0;
  const animTotal = driftSteps + maxLen;
  const inDrift = !!plan && animStep <= driftSteps;
  const replayFrameIdx = inDrift ? Math.min(animStep - 1, searchHour) : searchHour;
  const revealLen = inDrift ? 1 : (animStep - driftSteps);

  const heatmapPoints = useMemo(() => {
    if (!frames.length) return [];
    return frames[Math.min(replayFrameIdx, frames.length - 1)]?.points ?? [];
  }, [frames, replayFrameIdx]);

  // (re)start the animation whenever a new plan arrives
  useEffect(() => {
    if (plan) { setAnimStep(1); setPlaying(true); }
  }, [plan]);

  // advance one waypoint at a time while playing
  useEffect(() => {
    if (!playing) return;
    if (animStep >= animTotal) { setPlaying(false); return; }
    const id = setTimeout(() => setAnimStep((s) => Math.min(animTotal, s + 1)), 240);
    return () => clearTimeout(id);
  }, [playing, animStep, animTotal]);

  return (
    <div className="flex-1 flex overflow-hidden">

      {/* Left Sidebar */}
      <div className="w-[400px] bg-white border-r border-[#E2E8F0] flex flex-col shrink-0 overflow-y-auto">

        <div className="p-6 border-b border-[#E2E8F0]">
          <h2 className="text-xl font-bold text-[#0F172A] mb-1">Search Plan</h2>
          <p className="text-xs text-[#64748B] mb-4">
            Coordinated greedy coverage of the drift heatmap. Teams launch from
            shore and sweep over water only.
          </p>

          {/* the plan tracks the heatmap time; adjustable here too */}
          <div className="mb-2 flex justify-between text-sm font-semibold text-[#0F172A]">
            <span>Plan for forecast time</span>
            <span className="text-[#0F766E]">T+{currentHour}h</span>
          </div>
          <input type="range" min="0" max={maxHour} step="1" value={currentHour}
            onChange={(e) => setCurrentHour(parseInt(e.target.value))}
            className="w-full slider-grab mb-4" />

          <div className="space-y-2 text-sm">
            <div className="flex justify-between"><span className="text-[#64748B]">Mission time</span><span className="font-semibold">{plan ? `${plan.mission_time_min} min` : '–'}</span></div>
            <div className="flex justify-between"><span className="text-[#64748B]">Converged</span><span className="font-semibold">{plan ? (plan.stop_reason || '–').replace('_', ' ') : '–'}</span></div>
          </div>

          {generating && (
            <div className="mt-4 flex items-center gap-2 text-sm text-[#0F766E]">
              <span className="inline-block w-4 h-4 border-2 border-[#0F766E] border-t-transparent rounded-full animate-spin"></span>
              Planning for T+{searchHour}h…
            </div>
          )}
          {planError && (
            <p className="text-xs text-[#DC2626] mt-3">No plan ({planError}). Run a simulation first.</p>
          )}
        </div>

        <div className="p-6 border-b border-[#E2E8F0]">
          <h3 className="font-semibold text-[#0F172A] mb-1">Deploy Vehicles</h3>
          <p className="text-xs text-[#64748B] mb-3">
            Click anywhere on the map to drop a launch point, then pick the craft.
            The plan routes the fleet from your points.
          </p>

          <div className="mb-3">
            <label className="block text-xs font-medium text-[#0F172A] mb-1">
              Force deployment delay (hours)
            </label>
            <input
              type="number" min="0" max={maxHour} step="1" value={searchDelayHours}
              onChange={(e) => setSearchDelayHours(Math.max(0, Math.min(maxHour, parseInt(e.target.value) || 0)))}
              disabled={generating}
              className="w-full p-2 border border-[#E2E8F0] focus:border-[#0F766E] outline-none text-sm disabled:bg-gray-100"
            />
            <p className="text-[11px] text-[#64748B] mt-1">
              Search starts at <span className="font-semibold text-[#0F766E]">T+{searchHour}h</span> — the
              heatmap when the forces actually reach the water.
            </p>
          </div>

          {userVehicles.length > 0 ? (
            <div className="space-y-1 mb-3">
              {userVehicles.map((v, i) => (
                <div key={v.id} className="flex items-center justify-between text-sm border border-[#E2E8F0] bg-gray-50 px-2 py-1">
                  <span className="flex items-center gap-2">
                    <VehicleGlyph type={v.type} size={18} />
                    <span className="font-medium">{VEHICLE_LABEL[v.type]} {i + 1}</span>
                  </span>
                  <button type="button" onClick={() => removeVehicle(v.id)} disabled={generating} className="text-[#94A3B8] hover:text-[#DC2626] font-bold px-1 disabled:opacity-40">×</button>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-xs text-[#94A3B8] italic mb-3">No vehicles placed — the plan uses automatic shore launch points.</p>
          )}

          {generating && (
            <div className="mb-3">
              <div className="flex justify-between text-[11px] text-[#64748B] mb-1">
                <span>Forming plan…</span>
                <span>{Math.round(genProgress)}%</span>
              </div>
              <div className="w-full bg-[#E2E8F0] h-2 rounded-full overflow-hidden">
                <div className="bg-[#0F766E] h-full transition-all duration-200 ease-out"
                     style={{ width: `${Math.max(5, genProgress)}%` }}></div>
              </div>
            </div>
          )}

          <div className="flex gap-2">
            {generating ? (
              <button
                type="button"
                onClick={handleCancelPlan}
                className="flex-1 py-2 text-sm font-medium bg-[#DC2626] text-white hover:bg-[#B91C1C]"
              >
                ✕ Cancel
              </button>
            ) : (
              <button
                type="button"
                onClick={handleGeneratePlan}
                disabled={!userVehicles.length}
                className={`flex-1 py-2 text-sm font-medium ${userVehicles.length ? 'bg-[#0F766E] text-white hover:bg-[#115E59]' : 'bg-[#E2E8F0] text-[#94A3B8] cursor-not-allowed'}`}
              >
                Generate Plan
              </button>
            )}
            {planMode === 'user' && !generating && (
              <button type="button" onClick={handleClearVehicles} className="px-3 py-2 text-sm font-medium border border-[#E2E8F0] text-[#64748B] hover:bg-gray-50">
                Auto
              </button>
            )}
          </div>
        </div>

        {plan && (
          <div className="p-6 bg-[#F8F9FA] flex-1">
            <div className="bg-teal-50 border border-teal-200 p-4 mb-6">
              <p className="text-center font-bold text-teal-900 text-lg">
                Probability cleared in {plan.mission_time_min} min: {plan.total_cleared_pct}%
              </p>
            </div>
          </div>
        )}

        <div className="p-4 mt-auto border-t border-[#E2E8F0] flex-shrink-0">
          <button onClick={() => navigate('/heatmap')}
            className="w-full py-3 bg-white border-2 border-[#0F766E] text-[#0F766E] font-medium hover:bg-[#F0FDFA] transition-colors">
            ← Switch to Heatmap
          </button>
        </div>
      </div>

      {/* Right Map Panel */}
      <div className="flex-1 relative bg-blue-50">

        {!generating && !pendingPos && (
          <div className="absolute top-0 left-0 right-0 z-[500] bg-blue-50/90 border-b border-blue-200 text-blue-800 text-xs p-2 text-center font-medium shadow-sm pointer-events-none">
            Click the map to add a vehicle launch point
          </div>
        )}

        <MapContainer center={mapCenter} zoom={12} className="w-full h-full" zoomControl={false}>
          <TileLayer
            attribution='&copy; <a href="https://carto.com/attributions">CARTO</a>'
            url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
          />

          <HeatmapLayer points={heatmapPoints} />

          {!generating && !pendingPos && (
            <PlanClickHandler onMapClick={(pos) => setPendingPos(pos)} />
          )}

          {userVehicles.map((v, i) => (
            <Marker key={v.id} position={[v.lat, v.lng]} icon={vehicleIcon(v.type)}>
              <Tooltip direction="top" offset={[0, -10]} className="text-xs font-bold">{VEHICLE_LABEL[v.type]} {i + 1}</Tooltip>
            </Marker>
          ))}
          {pendingPos && (
            <Marker position={[pendingPos.lat, pendingPos.lng]} icon={pendingIcon} />
          )}

          {/* coarse labelled comms grid (A,B,C.. / 1,2,3..) for the teams */}
          <ReferenceGrid grid={plan?.reference_grid} />

          {/* LKP marker = the coordinates entered in the Incident Report */}
          <Marker position={mapCenter} icon={lspIcon}>
            <Tooltip permanent direction="top" offset={[0, -30]} className="font-bold">LKP</Tooltip>
          </Marker>

          {/* Animated coordinated team paths (shore -> sea): the search
              algorithm sweeping out, revealed one waypoint per tick. */}
          {teams.map((t, i) => {
            const full = teamTracks[i];
            const shown = full.slice(0, Math.max(1, revealLen));   // parked during drift, then revealed
            const head = shown[shown.length - 1];                 // current position
            return (
              <React.Fragment key={`team-${i}`}>
                <Polyline positions={shown} pathOptions={{ color: t.color, weight: 3, opacity: 0.9 }} />
                {/* shore launch point */}
                <CircleMarker center={full[0]} radius={7}
                  pathOptions={{ color: t.color, fillColor: t.color, fillOpacity: 1, weight: 2 }}>
                  <Tooltip direction="right" offset={[8, 0]} className="text-xs">
                    <div className="font-bold">{t.team} — launch{t.start_cell ? ` (${t.start_cell})` : ''}</div>
                  </Tooltip>
                </CircleMarker>
                {/* current sonar sweep at the head of the path */}
                {plan?.sonar_radius_m && (
                  <Circle center={head} radius={plan.sonar_radius_m}
                    pathOptions={{ color: t.color, fillColor: t.color, fillOpacity: 0.18, weight: 1 }} />
                )}
                <CircleMarker center={head} radius={5}
                  pathOptions={{ color: t.color, fillColor: 'white', fillOpacity: 1, weight: 2 }}>
                  <Tooltip direction="right" offset={[8, 0]} className="text-xs">
                    <div className="font-bold">{t.team}</div>
                    <div>Elapsed: +{Math.round((shown.length - 1) * (plan?.tick_seconds ?? 20) / 60)} min</div>
                  </Tooltip>
                </CircleMarker>
              </React.Fragment>
            );
          })}
        </MapContainer>

        {pendingPos && (
          <div className="absolute inset-0 z-[1000] flex items-center justify-center bg-black/30">
            <div className="bg-white rounded-lg shadow-xl p-5 w-80">
              <h4 className="font-bold text-[#0F172A] mb-1 text-center">Choose vehicle type</h4>
              <p className="text-xs text-[#64748B] text-center mb-4">Launch point selected. Pick the craft, or cancel.</p>
              <div className="grid grid-cols-3 gap-3">
                <button type="button" onClick={() => addVehicle('jetski')} className="flex flex-col items-center gap-1 py-4 border-2 border-[#E2E8F0] rounded-md hover:border-[#0F766E] hover:bg-[#F0FDFA]">
                  <VehicleGlyph type="jetski" size={30} />
                  <span className="text-sm font-semibold text-[#0F172A]">Jet-ski</span>
                  <span className="text-[10px] text-[#64748B]">fast</span>
                </button>
                <button type="button" onClick={() => addVehicle('boat')} className="flex flex-col items-center gap-1 py-4 border-2 border-[#E2E8F0] rounded-md hover:border-[#0F766E] hover:bg-[#F0FDFA]">
                  <VehicleGlyph type="boat" size={30} />
                  <span className="text-sm font-semibold text-[#0F172A]">Boat</span>
                  <span className="text-[10px] text-[#64748B]">slower</span>
                </button>
                <button type="button" onClick={() => setPendingPos(null)} className="flex flex-col items-center gap-1 py-4 border-2 border-red-200 rounded-md text-[#DC2626] hover:border-[#DC2626] hover:bg-red-50">
                  <span className="text-3xl leading-none">✕</span>
                  <span className="text-sm font-semibold">Cancel</span>
                  <span className="text-[10px] text-[#DC2626]/70">don't add</span>
                </button>
              </div>
            </div>
          </div>
        )}

        {/* animation controls */}
        {teams.length > 0 && (
          <div className="absolute top-4 left-4 z-[400] bg-white border border-[#E2E8F0] shadow-md p-2 flex items-center gap-2">
            <button onClick={() => { if (animStep >= animTotal) setAnimStep(1); setPlaying((p) => !p); }}
              className="px-3 py-1.5 text-sm font-semibold bg-[#0F766E] text-white rounded hover:bg-[#115E59]">
              {playing ? '❚❚ Pause' : (animStep >= animTotal ? '↻ Replay' : '▶ Play')}
            </button>
            <input type="range" min="1" max={animTotal} value={animStep}
              onChange={(e) => { setPlaying(false); setAnimStep(parseInt(e.target.value)); }}
              className="accent-[#0F766E] w-44" />
            <span className="text-xs font-medium tabular-nums whitespace-nowrap">
              {inDrift
                ? <span className="text-[#2563EB]">Drift T+{replayFrameIdx}h</span>
                : <span className="text-[#0F766E]">Search {Math.max(1, animStep - driftSteps)}/{maxLen}</span>}
            </span>
          </div>
        )}

        {teams.length > 0 && (
          <div className="absolute bottom-6 right-6 z-[400] bg-white border border-[#E2E8F0] p-3 shadow-md">
            <h4 className="text-xs font-bold text-[#64748B] uppercase mb-2 tracking-wider">Team Deployment</h4>
            <div className="space-y-1">
              {teams.map((t, i) => (
                <div key={i} className="flex items-center gap-2 text-sm font-medium text-[#0F172A]">
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: t.color }}></div>
                  {t.team} — {t.cleared_pct}%
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
