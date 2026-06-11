import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { MapContainer, TileLayer, Marker, useMapEvents } from 'react-leaflet';
import { useIncident } from '../context/IncidentContext';
import L from 'leaflet';

// Fix for default Leaflet marker icons not showing in React Leaflet
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

const customIcon = new L.Icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41]
});

function MapClickHandler({ setPos }) {
  useMapEvents({
    click(e) {
      setPos(e.latlng);
    },
  });
  return null;
}

export default function IncidentReport() {
  const navigate = useNavigate();
  const { incidentData, updateIncident, runSimulation, runState, setRunState } = useIncident();
  const [mapPos, setMapPos] = useState(incidentData.lat && incidentData.lng ? { lat: incidentData.lat, lng: incidentData.lng } : null);

  // Submit -> run the drift simulation LIVE on these inputs, show a loading bar,
  // then go to the Search Plan screen (heatmap + the algorithm running on it).
  const handleSubmit = (e) => {
    e.preventDefault();
    runSimulation(incidentData)
      .then(() => navigate('/heatmap'))
      .catch(() => { /* failure surfaced via runState.error in the overlay */ });
  };

  // Backup navigation: the moment a run signals done, move to the results
  // (the heatmap screen — confirmed-working — shows the new simulation).
  // Driven by the `done` flag so it never depends on promise timing.
  React.useEffect(() => {
    if (runState.done) {
      setRunState((s) => ({ ...s, done: false }));   // consume the flag
      navigate('/heatmap');
    }
  }, [runState.done, navigate, setRunState]);

  const handleMapClick = (pos) => {
    setMapPos(pos);
    if (incidentData.lspMode === 'map') {
      updateIncident({ lat: pos.lat.toFixed(5), lng: pos.lng.toFixed(5) });
    }
  };

  // Export the entered LKP + victim + time as incident.json. Dropping this file
  // in drift-app/public/ makes the simulation (test/sim_drowned_body.py) run at
  // THESE coordinates — i.e. the lon/lat entered here become the LKP.
  const handleDownloadIncident = () => {
    const incident = {
      id: incidentData.id,
      lat: incidentData.lat,
      lng: incidentData.lng,
      date: incidentData.date,
      timeFrom: incidentData.timeFrom,
      victimHeight: incidentData.victimHeight,   // cm
      victimWeight: incidentData.victimWeight,   // kg
      waterTemp: incidentData.waterTemp,
    };
    const blob = new Blob([JSON.stringify(incident, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'incident.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  const isFormValid = () => {
    return incidentData.date && 
           incidentData.timeFrom && 
           incidentData.timeTo && 
           incidentData.victimHeight && 
           incidentData.victimWeight && 
           incidentData.lat && 
           incidentData.lng;
  };

  const requiredFields = 7;
  const filledFields = [incidentData.date, incidentData.timeFrom, incidentData.timeTo, incidentData.victimHeight, incidentData.victimWeight, incidentData.lat, incidentData.lng].filter(Boolean).length;
  const progressPercent = Math.round((filledFields / requiredFields) * 100);

  return (
    <div className="flex-1 flex flex-col md:flex-row overflow-hidden bg-[#F8F9FA]">

      {/* Live simulation loading overlay */}
      {(runState.running || runState.error || runState.done) && (
        <div className="fixed inset-0 z-[2000] bg-[#0F172A]/70 flex items-center justify-center">
          <div className="bg-white rounded-lg shadow-2xl p-8 w-[440px] max-w-[90vw]">
            {runState.error ? (
              <>
                <h3 className="text-lg font-bold text-[#DC2626] mb-2">Simulation failed</h3>
                <p className="text-sm text-[#64748B] mb-4 break-words">{runState.error}</p>
                <p className="text-xs text-[#64748B]">
                  Is the backend running? Start it with
                  <span className="font-mono"> python api/server.py</span>.
                </p>
              </>
            ) : runState.done ? (
              <>
                <h3 className="text-lg font-bold text-[#0F766E] mb-1">✓ Simulation complete</h3>
                <p className="text-sm text-[#64748B] mb-5">
                  The new drift forecast and coordinated search plan are ready.
                </p>
                <button
                  type="button"
                  onClick={() => { setRunState((s) => ({ ...s, done: false })); navigate('/heatmap'); }}
                  className="w-full py-3 font-medium bg-[#0F766E] text-white hover:bg-[#115E59] rounded transition-colors"
                >
                  View drift heatmap →
                </button>
                <p className="text-xs text-[#64748B] mt-3 text-center">
                  Then open <b>Search Plan</b> to watch the search algorithm run.
                </p>
              </>
            ) : (
              <>
                <h3 className="text-lg font-bold text-[#0F172A] mb-1">Running drift simulation</h3>
                <p className="text-sm text-[#0F766E] mb-4">{runState.stage || 'Working…'}</p>
                <div className="w-full bg-[#E2E8F0] h-3 rounded-full overflow-hidden">
                  <div className="bg-[#0F766E] h-full transition-all duration-500 ease-out"
                       style={{ width: `${Math.max(3, runState.percent)}%` }}></div>
                </div>
                <p className="text-right text-xs text-[#64748B] mt-2 font-medium">{Math.round(runState.percent)}%</p>
                <p className="text-[11px] text-[#94A3B8] mt-3">
                  Forecasting body drift, sink/refloat &amp; the coordinated search plan
                  from your inputs. This can take a few minutes with live ocean data.
                </p>
              </>
            )}
          </div>
        </div>
      )}

      {/* Left Column: Form */}
      <div className="w-full md:w-1/2 overflow-y-auto p-6 md:p-8 border-r border-[#E2E8F0]">
        <h1 className="text-2xl font-bold mb-6 text-[#0F172A]">New Incident Report</h1>
        
        <form className="space-y-8" onSubmit={handleSubmit}>
          
          {/* Section 1: Incident Details */}
          <section className="bg-white p-6 border border-[#E2E8F0] shadow-sm rounded-sm">
            <h2 className="text-lg font-semibold border-b border-[#E2E8F0] pb-2 mb-4 text-[#0F766E]">1. Incident Details</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="col-span-1 md:col-span-2">
                <label className="block text-sm font-medium text-[#64748B] mb-1">Incident ID</label>
                <input type="text" readOnly value={incidentData.id} className="w-full p-2 bg-gray-100 border border-[#E2E8F0] text-[#64748B] cursor-not-allowed outline-none" />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-[#0F172A] mb-1">Date <span className="text-[#DC2626]">*</span></label>
                <input type="date" required value={incidentData.date} onChange={e => updateIncident({ date: e.target.value })} className="w-full p-2 border border-[#E2E8F0] focus:border-[#0F766E] focus:ring-1 focus:ring-[#0F766E] outline-none" />
              </div>
              
              <div className="flex gap-2">
                <div className="w-1/2">
                  <label className="block text-sm font-medium text-[#0F172A] mb-1">Time From <span className="text-[#DC2626]">*</span></label>
                  <input type="time" required value={incidentData.timeFrom} onChange={e => updateIncident({ timeFrom: e.target.value })} className="w-full p-2 border border-[#E2E8F0] focus:border-[#0F766E] focus:ring-1 focus:ring-[#0F766E] outline-none" />
                </div>
                <div className="w-1/2">
                  <label className="block text-sm font-medium text-[#0F172A] mb-1">Time To <span className="text-[#DC2626]">*</span></label>
                  <input type="time" required value={incidentData.timeTo} onChange={e => updateIncident({ timeTo: e.target.value })} className="w-full p-2 border border-[#E2E8F0] focus:border-[#0F766E] focus:ring-1 focus:ring-[#0F766E] outline-none" />
                </div>
              </div>

              <div className="col-span-1 md:col-span-2">
                <label className="block text-sm font-medium text-[#0F172A] mb-1">Water Body</label>
                <input type="text" list="water-bodies" placeholder="e.g. Mediterranean Sea" value={incidentData.waterBody} onChange={e => updateIncident({ waterBody: e.target.value })} className="w-full p-2 border border-[#E2E8F0] focus:border-[#0F766E] outline-none" />
                <datalist id="water-bodies">
                  <option value="Mediterranean Sea" />
                  <option value="Sea of Galilee (Kinneret)" />
                  <option value="Dead Sea" />
                  <option value="Red Sea (Eilat)" />
                  <option value="Jordan River" />
                  <option value="Other" />
                </datalist>
              </div>
            </div>
          </section>

          {/* Section 2: Last Seen Point */}
          <section className="bg-white p-6 border border-[#E2E8F0] shadow-sm rounded-sm">
            <h2 className="text-lg font-semibold border-b border-[#E2E8F0] pb-2 mb-4 text-[#0F766E]">2. Last Seen Point (LSP)</h2>
            
            <div className="flex border border-[#E2E8F0] mb-4">
              <button type="button" onClick={() => updateIncident({ lspMode: 'coordinates' })} className={`flex-1 py-2 text-sm font-medium ${incidentData.lspMode === 'coordinates' ? 'bg-[#0F766E] text-white' : 'bg-white text-[#64748B] hover:bg-gray-50'}`}>Enter Coordinates</button>
              <button type="button" onClick={() => updateIncident({ lspMode: 'map' })} className={`flex-1 py-2 text-sm font-medium ${incidentData.lspMode === 'map' ? 'bg-[#0F766E] text-white' : 'bg-white text-[#64748B] hover:bg-gray-50'}`}>Select on Map</button>
            </div>

            {incidentData.lspMode === 'map' && (
              <div className="bg-blue-50 border border-blue-200 text-blue-800 text-sm p-3 mb-4 flex items-center">
                <span className="mr-2">ℹ️</span> Click on the map to place the LSP marker.
              </div>
            )}

            <div className="grid grid-cols-2 gap-4 mb-4">
              <div>
                <label className="block text-sm font-medium text-[#0F172A] mb-1">Latitude <span className="text-[#DC2626]">*</span></label>
                <input type="number" step="any" required placeholder="e.g. 32.82" value={incidentData.lat} onChange={e => updateIncident({ lat: e.target.value })} disabled={incidentData.lspMode === 'map'} className="w-full p-2 border border-[#E2E8F0] focus:border-[#0F766E] outline-none disabled:bg-gray-100 disabled:text-gray-500" />
              </div>
              <div>
                <label className="block text-sm font-medium text-[#0F172A] mb-1">Longitude <span className="text-[#DC2626]">*</span></label>
                <input type="number" step="any" required placeholder="e.g. 34.99" value={incidentData.lng} onChange={e => updateIncident({ lng: e.target.value })} disabled={incidentData.lspMode === 'map'} className="w-full p-2 border border-[#E2E8F0] focus:border-[#0F766E] outline-none disabled:bg-gray-100 disabled:text-gray-500" />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-[#0F172A] mb-1">Coordinate Accuracy</label>
              <select value={incidentData.accuracy} onChange={e => updateIncident({ accuracy: e.target.value })} className="w-full p-2 border border-[#E2E8F0] focus:border-[#0F766E] outline-none bg-white">
                <option>Exact</option>
                <option>Approximate (~10m)</option>
                <option>Approximate (~50m)</option>
                <option>Unknown</option>
              </select>
            </div>
          </section>

          {/* Section 3: Victim Profile */}
          <section className="bg-white p-6 border border-[#E2E8F0] shadow-sm rounded-sm">
            <h2 className="text-lg font-semibold border-b border-[#E2E8F0] pb-2 mb-4 text-[#0F766E]">3. Victim Profile</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="col-span-1 md:col-span-2">
                <label className="block text-sm font-medium text-[#0F172A] mb-1">Full Name</label>
                <input type="text" value={incidentData.victimName} onChange={e => updateIncident({ victimName: e.target.value })} className="w-full p-2 border border-[#E2E8F0] focus:border-[#0F766E] outline-none" />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-[#0F172A] mb-1">Age</label>
                <input type="number" min="0" value={incidentData.victimAge} onChange={e => updateIncident({ victimAge: e.target.value })} className="w-full p-2 border border-[#E2E8F0] focus:border-[#0F766E] outline-none" />
              </div>

              <div>
                <label className="block text-sm font-medium text-[#0F172A] mb-1">Gender</label>
                <select value={incidentData.victimGender} onChange={e => updateIncident({ victimGender: e.target.value })} className="w-full p-2 border border-[#E2E8F0] focus:border-[#0F766E] outline-none bg-white">
                  <option>Unknown</option>
                  <option>Male</option>
                  <option>Female</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-[#0F172A] mb-1" title="Used to estimate body density and drift depth profile">Height (cm) <span className="text-[#DC2626]">*</span> ℹ️</label>
                <input type="number" required min="0" value={incidentData.victimHeight} onChange={e => updateIncident({ victimHeight: e.target.value })} className="w-full p-2 border border-[#E2E8F0] focus:border-[#0F766E] outline-none" />
              </div>

              <div>
                <label className="block text-sm font-medium text-[#0F172A] mb-1" title="Used to estimate body density and drift depth profile">Weight (kg) <span className="text-[#DC2626]">*</span> ℹ️</label>
                <input type="number" required min="0" value={incidentData.victimWeight} onChange={e => updateIncident({ victimWeight: e.target.value })} className="w-full p-2 border border-[#E2E8F0] focus:border-[#0F766E] outline-none" />
              </div>

              <div className="col-span-1 md:col-span-2">
                <label className="block text-sm font-medium text-[#0F172A] mb-1">Clothing Description</label>
                <textarea rows="2" placeholder="e.g. red shirt, black shorts, no shoes..." value={incidentData.victimClothing} onChange={e => updateIncident({ victimClothing: e.target.value })} className="w-full p-2 border border-[#E2E8F0] focus:border-[#0F766E] outline-none"></textarea>
              </div>

              <div>
                <label className="block text-sm font-medium text-[#0F172A] mb-1">Swimming Ability</label>
                <select value={incidentData.swimmingAbility} onChange={e => updateIncident({ swimmingAbility: e.target.value })} className="w-full p-2 border border-[#E2E8F0] focus:border-[#0F766E] outline-none bg-white">
                  <option>Unknown</option>
                  <option>Non-swimmer</option>
                  <option>Beginner</option>
                  <option>Intermediate</option>
                  <option>Strong swimmer</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-[#0F172A] mb-1">Physical Condition</label>
                <select value={incidentData.physicalCondition} onChange={e => updateIncident({ physicalCondition: e.target.value })} className="w-full p-2 border border-[#E2E8F0] focus:border-[#0F766E] outline-none bg-white">
                  <option>Unknown</option>
                  <option>Healthy</option>
                  <option>Injured</option>
                  <option>Impaired</option>
                </select>
              </div>
            </div>
          </section>

          {/* Section 4: Environmental Snapshot */}
          <section className="bg-white p-6 border border-[#E2E8F0] shadow-sm rounded-sm">
            <div className="flex justify-between items-center border-b border-[#E2E8F0] pb-2 mb-4">
              <h2 className="text-lg font-semibold text-[#0F766E]">4. Environmental Snapshot</h2>
              <span className="text-xs text-[#64748B]">Optional but improves accuracy</span>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-[#0F172A] mb-1">Wind Direction</label>
                <select value={incidentData.windDirection} onChange={e => updateIncident({ windDirection: e.target.value })} className="w-full p-2 border border-[#E2E8F0] focus:border-[#0F766E] outline-none bg-white">
                  <option value="">Select...</option>
                  <option>N</option><option>NE</option><option>E</option><option>SE</option>
                  <option>S</option><option>SW</option><option>W</option><option>NW</option>
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-[#0F172A] mb-1">Wind Speed (km/h)</label>
                <input type="number" min="0" value={incidentData.windSpeed} onChange={e => updateIncident({ windSpeed: e.target.value })} className="w-full p-2 border border-[#E2E8F0] focus:border-[#0F766E] outline-none" />
              </div>

              <div>
                <label className="block text-sm font-medium text-[#0F172A] mb-1">Water Conditions</label>
                <select value={incidentData.waterConditions} onChange={e => updateIncident({ waterConditions: e.target.value })} className="w-full p-2 border border-[#E2E8F0] focus:border-[#0F766E] outline-none bg-white">
                  <option value="">Select...</option>
                  <option>Calm</option>
                  <option>Light Chop</option>
                  <option>Moderate</option>
                  <option>Rough</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-[#0F172A] mb-1" title="Critical for estimating time to resurface">Water Temp (°C) ℹ️</label>
                <input type="number" step="0.1" value={incidentData.waterTemp} onChange={e => updateIncident({ waterTemp: e.target.value })} className="w-full p-2 border border-[#E2E8F0] focus:border-[#0F766E] outline-none" />
              </div>
            </div>
          </section>

          {/* Form Actions */}
          <div className="sticky bottom-0 bg-[#F8F9FA] pt-4 pb-8 border-t border-[#E2E8F0] mt-8 z-10">
            <div className="mb-4">
              <div className="flex justify-between text-xs text-[#64748B] mb-1">
                <span>Form Completion</span>
                <span>{progressPercent}%</span>
              </div>
              <div className="w-full bg-[#E2E8F0] h-1.5 rounded-none overflow-hidden">
                <div className="bg-[#0F766E] h-full transition-all duration-300" style={{ width: `${progressPercent}%` }}></div>
              </div>
            </div>
            <div className="flex gap-4">
              <button type="button" className="px-4 py-3 bg-white border border-[#0F766E] text-[#0F766E] font-medium hover:bg-[#F0FDFA] transition-colors">
                Save Draft
              </button>
              <button
                type="button"
                onClick={handleDownloadIncident}
                disabled={!incidentData.lat || !incidentData.lng}
                title="Download incident.json to drive the simulation at these coordinates"
                className={`px-4 py-3 border font-medium transition-colors ${incidentData.lat && incidentData.lng ? 'bg-white border-[#0F766E] text-[#0F766E] hover:bg-[#F0FDFA]' : 'bg-gray-100 border-[#E2E8F0] text-[#94A3B8] cursor-not-allowed'}`}
              >
                ⬇ incident.json
              </button>
              <button
                type="submit"
                disabled={!isFormValid() || runState.running}
                className={`flex-1 py-3 font-medium transition-colors ${(!isFormValid() || runState.running) ? 'bg-[#94A3B8] text-white cursor-not-allowed' : 'bg-[#0F766E] text-white hover:bg-[#115E59]'}`}
              >
                {runState.running ? 'Running simulation…' : 'Run Simulation & Continue →'}
              </button>
            </div>
          </div>
        </form>
      </div>

      {/* Right Column: Map */}
      <div className={`w-full md:w-1/2 h-[50vh] md:h-auto relative ${incidentData.lspMode === 'map' ? 'ring-4 ring-blue-400 ring-inset z-10' : ''}`}>
        <MapContainer center={[32.82, 34.99]} zoom={12} className="w-full h-full">
          <TileLayer
            attribution='&copy; <a href="https://carto.com/attributions">CARTO</a>'
            url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
          />
          {incidentData.lspMode === 'map' && <MapClickHandler setPos={handleMapClick} />}
          {mapPos && <Marker position={mapPos} icon={customIcon} />}
          {incidentData.lat && incidentData.lng && incidentData.lspMode === 'coordinates' && (
             <Marker position={{ lat: parseFloat(incidentData.lat), lng: parseFloat(incidentData.lng) }} icon={customIcon} />
          )}
        </MapContainer>
      </div>
    </div>
  );
}
