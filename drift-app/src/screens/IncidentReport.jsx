import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { MapContainer, TileLayer, Marker, useMapEvents, ScaleControl } from 'react-leaflet';
import MapGrid from '../components/MapGrid';
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
  const { incidentData, updateIncident } = useIncident();
  const [mapPos, setMapPos] = useState(incidentData.lat && incidentData.lng ? { lat: incidentData.lat, lng: incidentData.lng } : null);

  const handleMapClick = (pos) => {
    setMapPos(pos);
    if (incidentData.lspMode === 'map') {
      updateIncident({ lat: pos.lat.toFixed(5), lng: pos.lng.toFixed(5) });
    }
  };

  const isFormValid = () => {
    return incidentData.date && 
           incidentData.time && 
           incidentData.victimHeight && 
           incidentData.victimWeight && 
           incidentData.lat && 
           incidentData.lng;
  };

  const requiredFields = 6;
  const filledFields = [incidentData.date, incidentData.time, incidentData.victimHeight, incidentData.victimWeight, incidentData.lat, incidentData.lng].filter(Boolean).length;
  const progressPercent = Math.round((filledFields / requiredFields) * 100);

  return (
    <div className="flex-1 flex flex-col md:flex-row overflow-hidden bg-[#F8F9FA]">
      {/* Left Column: Form */}
      <div className="w-full md:w-1/2 overflow-y-auto p-6 md:p-8 border-r border-[#E2E8F0]">
        <h1 className="text-2xl font-bold mb-6 text-[#0F172A]">New Incident Report</h1>
        
        <form className="space-y-8" onSubmit={(e) => { e.preventDefault(); navigate('/heatmap'); }}>
          
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
                <div className="w-full">
                  <label className="block text-sm font-medium text-[#0F172A] mb-1">Time <span className="text-[#DC2626]">*</span></label>
                  <input type="time" required value={incidentData.time} onChange={e => updateIncident({ time: e.target.value })} className="w-full p-2 border border-[#E2E8F0] focus:border-[#0F766E] focus:ring-1 focus:ring-[#0F766E] outline-none" />
                </div>
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
          </section>

          {/* Section 3: Victim Profile */}
          <section className="bg-white p-6 border border-[#E2E8F0] shadow-sm rounded-sm">
            <h2 className="text-lg font-semibold border-b border-[#E2E8F0] pb-2 mb-4 text-[#0F766E]">3. Victim Profile</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
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
              <button 
                type="submit" 
                disabled={!isFormValid()}
                className={`flex-1 py-3 font-medium transition-colors ${isFormValid() ? 'bg-[#0F766E] text-white hover:bg-[#115E59]' : 'bg-[#94A3B8] text-white cursor-not-allowed'}`}
              >
                Submit Incident & Continue →
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
          <MapGrid />
          <ScaleControl position="bottomright" />
          {incidentData.lspMode === 'map' && <MapClickHandler setPos={handleMapClick} />}
          {mapPos && <Marker position={mapPos} icon={customIcon} />}
          {incidentData.lat && incidentData.lng && incidentData.lspMode === 'coordinates' && (
             <Marker position={{ lat: parseFloat(incidentData.lat), lng: parseFloat(incidentData.lng) }} icon={customIcon} />
          )}
        </MapContainer>
        <div className="absolute top-4 right-4 z-[1000] bg-white/90 px-2 py-1 border border-[#0F766E] text-xs font-bold text-[#0F766E] shadow-sm rounded-sm pointer-events-none">
          Grid: 1km x 1km
        </div>
      </div>
    </div>
  );
}
