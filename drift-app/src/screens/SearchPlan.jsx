import React, { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useIncident } from '../context/IncidentContext';
import { MapContainer, TileLayer, Marker, Polyline, Tooltip, Circle, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet.heat';

// Reusing same icon for LSP
const lspIcon = new L.Icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41]
});

// Generate mock team path
const generateMockPath = (startPos, teamIndex) => {
  const path = [];
  const [lat, lng] = startPos;
  let curLat = lat;
  let curLng = lng;
  
  // Base offset depending on team
  const latOffsetDir = teamIndex === 0 ? 1 : teamIndex === 1 ? -1 : 0;
  const lngOffsetDir = teamIndex === 2 ? 1 : teamIndex === 0 ? 0.5 : -0.5;

  for(let i=0; i<6; i++) {
    curLat += (Math.random() * 0.01 + 0.005) * latOffsetDir;
    curLng += (Math.random() * 0.01 + 0.005) * lngOffsetDir;
    path.push([curLat, curLng]);
  }
  return path;
};

function HeatmapLayer({ points }) {
  const map = useMap();
  React.useEffect(() => {
    if (!points || points.length === 0) return;
    const heat = L.heatLayer(points, {
      radius: 18,
      blur: 18,
      maxZoom: 13,
      gradient: { 0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1.0: 'red' }
    }).addTo(map);
    return () => {
      map.removeLayer(heat);
    };
  }, [map, points]);
  return null;
}

function generateHeatmapPoints(timeOffset) {
  const points = [];
  const count = 1500;
  const cLat = 32.85 + (timeOffset * 0.0003);
  const cLng = 34.95 + (timeOffset * 0.0001);
  const spread = 0.01 + (timeOffset * 0.0005);
  
  for(let i=0; i<count; i++) {
    const u1 = Math.random() || 0.001;
    const u2 = Math.random() || 0.001;
    const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    const z1 = Math.sqrt(-2.0 * Math.log(u1)) * Math.sin(2.0 * Math.PI * u2);
    
    const intensity = Math.random() * 0.8 + 0.2;
    
    points.push([
      cLat + z0 * spread,
      cLng + z1 * spread,
      intensity
    ]);
  }
  return points;
}

const TEAM_COLORS = ['#3b82f6', '#f97316', '#22c55e', '#a855f7', '#ec4899']; // blue, orange, green, purple, pink

export default function SearchPlan() {
  const navigate = useNavigate();
  const { incidentData } = useIncident();
  
  const [config, setConfig] = useState({
    teams: 3,
    radius: 50,
    duration: 2,
    strategy: 'coverage'
  });

  const [isComputing, setIsComputing] = useState(false);
  const [planGenerated, setPlanGenerated] = useState(false);
  const [teamPaths, setTeamPaths] = useState([]);
  
  const mapCenter = (incidentData.lat && incidentData.lng) 
    ? [parseFloat(incidentData.lat), parseFloat(incidentData.lng)] 
    : [32.82, 34.99];

  // We use a fixed timeOffset of 12h for the search plan heatmap preview
  const heatmapPoints = React.useMemo(() => generateHeatmapPoints(12), []);

  const handleGenerate = () => {
    setIsComputing(true);
    setPlanGenerated(false);
    
    setTimeout(() => {
      // generate mock paths
      const paths = [];
      for(let i=0; i<config.teams; i++) {
        paths.push(generateMockPath(mapCenter, i));
      }
      setTeamPaths(paths);
      setIsComputing(false);
      setPlanGenerated(true);
    }, 2000);
  };

  return (
    <div className="flex-1 flex overflow-hidden">
      
      {/* Left Sidebar */}
      <div className="w-[400px] bg-white border-r border-[#E2E8F0] flex flex-col shrink-0 overflow-y-auto">
        
        <div className="p-6 border-b border-[#E2E8F0]">
          <h2 className="text-xl font-bold text-[#0F172A] mb-4">Search Plan Configuration</h2>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-[#0F172A] mb-1">Number of Teams</label>
              <input type="number" min="1" max="10" value={config.teams} onChange={e => setConfig({...config, teams: parseInt(e.target.value)})} className="w-full p-2 border border-[#E2E8F0] focus:border-[#0F766E] outline-none" />
            </div>

            <div>
              <label className="block text-sm font-medium text-[#0F172A] mb-1" title="The effective detection radius of the sonar equipment">Sonar Radius per team (m) ℹ️</label>
              <input type="number" value={config.radius} onChange={e => setConfig({...config, radius: parseInt(e.target.value)})} className="w-full p-2 border border-[#E2E8F0] focus:border-[#0F766E] outline-none" />
            </div>

            <div>
              <label className="block text-sm font-medium text-[#0F172A] mb-1">Max Duration per team (hours)</label>
              <input type="number" value={config.duration} onChange={e => setConfig({...config, duration: parseInt(e.target.value)})} className="w-full p-2 border border-[#E2E8F0] focus:border-[#0F766E] outline-none" />
            </div>

            <div>
              <label className="block text-sm font-medium text-[#0F172A] mb-2">Search Strategy</label>
              <div className="space-y-2 text-sm text-[#0F172A]">
                <label className="flex items-center gap-2">
                  <input type="radio" name="strategy" value="coverage" checked={config.strategy === 'coverage'} onChange={() => setConfig({...config, strategy: 'coverage'})} className="accent-[#0F766E]" />
                  Maximize probability coverage
                </label>
                <label className="flex items-center gap-2">
                  <input type="radio" name="strategy" value="systematic" checked={config.strategy === 'systematic'} onChange={() => setConfig({...config, strategy: 'systematic'})} className="accent-[#0F766E]" />
                  Systematic sweep (Lawnmower)
                </label>
                <label className="flex items-center gap-2">
                  <input type="radio" name="strategy" value="converge" checked={config.strategy === 'converge'} onChange={() => setConfig({...config, strategy: 'converge'})} className="accent-[#0F766E]" />
                  Converge on peak zone
                </label>
              </div>
            </div>

            <button 
              onClick={handleGenerate}
              disabled={isComputing}
              className={`w-full mt-4 py-3 font-medium transition-colors flex justify-center items-center ${isComputing ? 'bg-[#0F766E]/80 text-white cursor-wait' : 'bg-[#0F766E] text-white hover:bg-[#115E59]'}`}
            >
              {isComputing ? 'Computing optimal paths...' : 'Generate Plan'}
            </button>
          </div>
        </div>

        {planGenerated && (
          <div className="p-6 bg-[#F8F9FA] flex-1">
            <div className="bg-teal-50 border border-teal-200 p-4 mb-6">
              <p className="text-center font-bold text-teal-900 text-lg">
                Estimated probability of locating victim: 73%
              </p>
            </div>

            <h3 className="font-semibold text-[#0F172A] mb-3">Team Assignments</h3>
            <div className="border border-[#E2E8F0] bg-white text-sm mb-6">
              <table className="w-full text-left">
                <thead className="bg-gray-50 border-b border-[#E2E8F0]">
                  <tr>
                    <th className="p-2 font-medium text-[#64748B]">Team</th>
                    <th className="p-2 font-medium text-[#64748B]">Pts</th>
                    <th className="p-2 font-medium text-[#64748B]">Dist</th>
                    <th className="p-2 font-medium text-[#64748B]">Cov</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#E2E8F0]">
                  {teamPaths.map((_, i) => (
                    <tr key={i}>
                      <td className="p-2 flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full" style={{ backgroundColor: TEAM_COLORS[i % TEAM_COLORS.length] }}></div>
                        Team {String.fromCharCode(65 + i)}
                      </td>
                      <td className="p-2">6</td>
                      <td className="p-2">{(Math.random()*4 + 2).toFixed(1)} km</td>
                      <td className="p-2">{Math.floor(Math.random()*15 + 10)}%</td>
                    </tr>
                  ))}
                  <tr className="bg-gray-50 font-bold text-[#0F172A]">
                    <td className="p-2">TOTAL</td>
                    <td className="p-2">{teamPaths.length * 6}</td>
                    <td className="p-2">~{teamPaths.length * 3} km</td>
                    <td className="p-2">73%</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div className="border border-[#E2E8F0] bg-white p-4">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-xs font-bold uppercase tracking-wider text-[#64748B]">AI Tactical Summary</span>
              </div>
              <p className="text-sm text-[#0F172A] leading-relaxed mb-3">
                Team A is assigned the high-confidence northern corridor near the LSP. Teams B and C are deployed to cover the eastern dispersion zone. If no contact is made within 45 minutes, recommend Teams B and C converge on Team A's sector. Current model confidence: HIGH.
              </p>
              <p className="text-[10px] text-[#64748B] italic">
                Generated by Nahshol AI — based on drift model output and search parameters
              </p>
            </div>
          </div>
        )}

        <div className="p-4 mt-auto border-t border-[#E2E8F0] mt-auto mt-4 flex-shrink-0">
          <button 
            onClick={() => navigate('/heatmap')}
            className="w-full py-3 bg-white border-2 border-[#0F766E] text-[#0F766E] font-medium hover:bg-[#F0FDFA] transition-colors"
          >
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

          {/* LSP Marker */}
          <Marker position={mapCenter} icon={lspIcon}>
            <Tooltip permanent direction="top" offset={[0, -30]} className="font-bold">LSP</Tooltip>
          </Marker>

          {/* Team Paths */}
          {planGenerated && teamPaths.map((path, i) => {
            const color = TEAM_COLORS[i % TEAM_COLORS.length];
            // Combine start point with generated path
            const fullPath = [mapCenter, ...path];
            
            return (
              <React.Fragment key={`team-${i}`}>
                <Polyline 
                  positions={fullPath} 
                  pathOptions={{ color, weight: 3, dashArray: '8, 8' }} 
                />
                {fullPath.map((pt, j) => {
                  if (j === 0) return null; // Skip LSP
                  return (
                    <Circle 
                      key={`pt-${i}-${j}`}
                      center={pt} 
                      radius={40} 
                      pathOptions={{ color, fillColor: 'white', fillOpacity: 1, weight: 2 }}
                    >
                      <Tooltip direction="right" offset={[10, 0]} className="text-xs">
                        <div className="font-bold">Waypoint {j} — Team {String.fromCharCode(65 + i)}</div>
                        <div>Est. arrival: T+{j * 15}min</div>
                        <div>Zone probability: {Math.floor(Math.random() * 30 + 10)}%</div>
                      </Tooltip>
                    </Circle>
                  )
                })}
              </React.Fragment>
            );
          })}
        </MapContainer>

        {planGenerated && (
          <div className="absolute bottom-6 right-6 z-[400] bg-white border border-[#E2E8F0] p-3 shadow-md">
            <h4 className="text-xs font-bold text-[#64748B] uppercase mb-2 tracking-wider">Team Deployment</h4>
            <div className="space-y-1">
              {teamPaths.map((_, i) => (
                <div key={i} className="flex items-center gap-2 text-sm font-medium text-[#0F172A]">
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: TEAM_COLORS[i % TEAM_COLORS.length] }}></div>
                  Team {String.fromCharCode(65 + i)}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
