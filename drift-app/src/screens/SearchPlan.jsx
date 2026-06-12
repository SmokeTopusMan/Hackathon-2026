import React, { useState, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { useIncident } from '../context/IncidentContext';
import { MapContainer, TileLayer, Marker, Polyline, Tooltip, CircleMarker, Circle, useMap } from 'react-leaflet';
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

// great-circle length of a poly-path, km
function pathKm(waypoints) {
  const R = 6371;
  let km = 0;
  for (let i = 1; i < waypoints.length; i++) {
    const [la1, lo1] = waypoints[i - 1];
    const [la2, lo2] = waypoints[i];
    const dLa = (la2 - la1) * Math.PI / 180, dLo = (lo2 - lo1) * Math.PI / 180;
    const a = Math.sin(dLa / 2) ** 2 +
      Math.cos(la1 * Math.PI / 180) * Math.cos(la2 * Math.PI / 180) * Math.sin(dLo / 2) ** 2;
    km += 2 * R * Math.asin(Math.sqrt(a));
  }
  return km;
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

export default function SearchPlan() {
  const navigate = useNavigate();
  const { incidentData, driftData, setDriftData, currentHour, setCurrentHour, fetchPlanForHour } = useIncident();

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

  // the plan is (re)computed for the hour currently shown on the heatmap
  const [plan, setPlan] = useState(null);
  const [planLoading, setPlanLoading] = useState(false);
  const [planError, setPlanError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setPlanLoading(true);
    setPlanError(null);
    fetchPlanForHour(currentHour)
      .then((p) => { if (!cancelled) setPlan(p); })
      .catch((e) => {
        if (cancelled) return;
        // no backend -> fall back to the plan embedded in the sim output
        if (driftData?.search_plan) setPlan(driftData.search_plan);
        else setPlanError(e.message);
      })
      .finally(() => { if (!cancelled) setPlanLoading(false); });
    return () => { cancelled = true; };
  }, [currentHour, fetchPlanForHour, driftData]);

  const mapCenter = (incidentData.lat && incidentData.lng)
    ? [parseFloat(incidentData.lat), parseFloat(incidentData.lng)]
    : driftData?.lkp ? [driftData.lkp.lat, driftData.lkp.lon] : [32.82, 34.99];

  // heatmap underlay = the frame the plan is computed on (current hour)
  const heatmapPoints = useMemo(() => {
    if (!frames.length) return [];
    return frames[Math.min(currentHour, frames.length - 1)]?.points ?? [];
  }, [frames, currentHour]);

  const teams = plan?.teams ?? [];
  const totalKm = teams.reduce((s, t) => s + pathKm(t.waypoints), 0);

  // ---- animation: watch the teams sweep out from shore (algorithm running) --
  const teamTracks = useMemo(() => teams.map((t) => dedupe(t.waypoints)), [teams]);
  const maxLen = useMemo(
    () => Math.max(1, ...teamTracks.map((p) => p.length)), [teamTracks]);
  const [animStep, setAnimStep] = useState(1);
  const [playing, setPlaying] = useState(true);

  // (re)start the animation whenever a new plan arrives
  useEffect(() => {
    if (plan) { setAnimStep(1); setPlaying(true); }
  }, [plan]);

  // advance one waypoint at a time while playing
  useEffect(() => {
    if (!playing) return;
    if (animStep >= maxLen) { setPlaying(false); return; }
    const id = setTimeout(() => setAnimStep((s) => Math.min(maxLen, s + 1)), 240);
    return () => clearTimeout(id);
  }, [playing, animStep, maxLen]);

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
            className="w-full accent-[#0F766E] mb-4" />

          <div className="space-y-2 text-sm">
            <div className="flex justify-between"><span className="text-[#64748B]">Teams</span><span className="font-semibold">{plan ? plan.teams.length : '–'}</span></div>
            <div className="flex justify-between"><span className="text-[#64748B]">Sonar radius</span><span className="font-semibold">{plan ? `${plan.sonar_radius_m} m` : '–'}</span></div>
            <div className="flex justify-between"><span className="text-[#64748B]">Grid cell</span><span className="font-semibold">{plan ? `${plan.grid_m} m` : '–'}</span></div>
            <div className="flex justify-between"><span className="text-[#64748B]">Mission time</span><span className="font-semibold">{plan ? `${plan.mission_time_min} min` : '–'}</span></div>
            <div className="flex justify-between"><span className="text-[#64748B]">Converged</span><span className="font-semibold">{plan ? (plan.stop_reason || '–').replace('_', ' ') : '–'}</span></div>
          </div>
          <p className="text-[11px] text-[#64748B] italic mt-3">
            Runs until coverage converges (target {plan ? plan.coverage_target_pct : 95}%),
            not a fixed step count. Team count / sonar from the simulation.
          </p>

          {planLoading && (
            <div className="mt-4 flex items-center gap-2 text-sm text-[#0F766E]">
              <span className="inline-block w-4 h-4 border-2 border-[#0F766E] border-t-transparent rounded-full animate-spin"></span>
              Planning for T+{currentHour}h…
            </div>
          )}
          {planError && (
            <p className="text-xs text-[#DC2626] mt-3">No plan ({planError}). Run a simulation first.</p>
          )}
        </div>

        {plan && (
          <div className="p-6 bg-[#F8F9FA] flex-1">
            <div className="bg-teal-50 border border-teal-200 p-4 mb-6">
              <p className="text-center font-bold text-teal-900 text-lg">
                Probability cleared in {plan.mission_time_min} min: {plan.total_cleared_pct}%
              </p>
            </div>

            <h3 className="font-semibold text-[#0F172A] mb-3">Team Assignments</h3>
            <div className="border border-[#E2E8F0] bg-white text-sm mb-6">
              <table className="w-full text-left">
                <thead className="bg-gray-50 border-b border-[#E2E8F0]">
                  <tr>
                    <th className="p-2 font-medium text-[#64748B]">Craft</th>
                    <th className="p-2 font-medium text-[#64748B]">Dist</th>
                    <th className="p-2 font-medium text-[#64748B]">Time</th>
                    <th className="p-2 font-medium text-[#64748B]">Cleared</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#E2E8F0]">
                  {teams.map((t, i) => (
                    <tr key={i}>
                      <td className="p-2 flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full" style={{ backgroundColor: t.color }}></div>
                        {t.team}{t.start_cell ? ` · ${t.start_cell}` : ''}
                      </td>
                      <td className="p-2">{t.distance_km ?? pathKm(t.waypoints).toFixed(1)} km</td>
                      <td className="p-2">{t.time_min != null ? `${t.time_min} min` : '–'}</td>
                      <td className="p-2">{t.cleared_pct}%</td>
                    </tr>
                  ))}
                  <tr className="bg-gray-50 font-bold text-[#0F172A]">
                    <td className="p-2">TOTAL</td>
                    <td className="p-2">~{teams.reduce((s, t) => s + (t.distance_km ?? 0), 0).toFixed(1)} km</td>
                    <td className="p-2">{plan.mission_time_min} min</td>
                    <td className="p-2">{plan.total_cleared_pct}%</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div className="border border-[#E2E8F0] bg-white p-4">
              <span className="text-xs font-bold uppercase tracking-wider text-[#64748B]">AI Tactical Summary</span>
              <p className="text-sm text-[#0F172A] leading-relaxed my-2">
                {teams.length} teams launch from the nearest shore points and are dispersed by a
                coordinated greedy planner (sonar {plan.sonar_radius_m} m, {plan.grid_m} m grid) over the
                T+{plan.plan_hour}h probability field — clearing <b>{plan.total_cleared_pct}%</b> with no
                overlapping sweeps. Re-plan at a later hour as the body drifts.
              </p>
              <p className="text-[10px] text-[#64748B] italic">
                Generated by core/search_planner.py on the live drift heatmap
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
        <MapContainer center={mapCenter} zoom={12} className="w-full h-full" zoomControl={false}>
          <TileLayer
            attribution='&copy; <a href="https://carto.com/attributions">CARTO</a>'
            url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
          />

          <HeatmapLayer points={heatmapPoints} />

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
            const shown = full.slice(0, Math.max(1, animStep));   // path so far
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

        {/* animation controls */}
        {teams.length > 0 && (
          <div className="absolute top-4 left-4 z-[400] bg-white border border-[#E2E8F0] shadow-md p-2 flex items-center gap-2">
            <button onClick={() => { if (animStep >= maxLen) setAnimStep(1); setPlaying((p) => !p); }}
              className="px-3 py-1.5 text-sm font-semibold bg-[#0F766E] text-white rounded hover:bg-[#115E59]">
              {playing ? '❚❚ Pause' : (animStep >= maxLen ? '↻ Replay' : '▶ Play')}
            </button>
            <input type="range" min="1" max={maxLen} value={animStep}
              onChange={(e) => { setPlaying(false); setAnimStep(parseInt(e.target.value)); }}
              className="accent-[#0F766E] w-40" />
            <span className="text-xs text-[#64748B] font-medium tabular-nums">step {animStep}/{maxLen}</span>
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
