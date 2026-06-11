import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useIncident } from '../context/IncidentContext';
import { Trash2 } from 'lucide-react';

export default function CivilianIntelligence() {
  const navigate = useNavigate();
  const { incidentData, reports, addReport, deleteReport } = useIncident();
  
  const [showForm, setShowForm] = useState(false);
  const [isSimulating, setIsSimulating] = useState(false);
  const [newReport, setNewReport] = useState({
    type: 'Eyewitness',
    name: '',
    time: '',
    location: '',
    text: ''
  });

  const getBadgeColor = (type) => {
    switch (type) {
      case 'Eyewitness': return 'bg-blue-100 text-blue-800';
      case 'Local Knowledge': return 'bg-amber-100 text-amber-800';
      case 'Rescue Personnel': return 'bg-[#0F766E]/10 text-[#0F766E]';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const handleAddReport = (e) => {
    e.preventDefault();
    if (newReport.text) {
      addReport(newReport);
      setNewReport({ type: 'Eyewitness', name: '', time: '', location: '', text: '' });
      setShowForm(false);
    }
  };

  const handleRunModel = () => {
    setIsSimulating(true);
    setTimeout(() => {
      setIsSimulating(false);
      navigate('/heatmap');
    }, 2500);
  };

  return (
    <div className="flex-1 overflow-y-auto bg-[#F8F9FA] p-6">
      <div className="max-w-[800px] mx-auto">
        <div className="mb-8">
          <h1 className="text-2xl font-bold mb-2 text-[#0F172A]">Civilian Intelligence Reports</h1>
          <p className="text-[#64748B] text-sm">
            Add eyewitness accounts, local knowledge, or any informal observations that may help locate the victim. Each report will be analyzed and incorporated into the search model.
          </p>
        </div>

        {/* Incident Summary Card */}
        <div className="bg-white border border-[#E2E8F0] border-l-4 border-l-[#0F766E] p-4 mb-8 shadow-sm">
          <div className="flex justify-between items-start">
            <div>
              <h3 className="font-bold text-[#0F172A]">{incidentData.id}</h3>
              <p className="text-sm text-[#64748B]">Victim: {incidentData.victimName || 'Unknown'}</p>
            </div>
            <div className="text-right text-sm">
              <p className="text-[#0F172A]">LSP: {incidentData.lat || '--'}, {incidentData.lng || '--'}</p>
              <p className="text-[#64748B]">{incidentData.date} • {incidentData.timeFrom} - {incidentData.timeTo}</p>
            </div>
          </div>
        </div>

        {/* Report List */}
        <div className="mb-6 space-y-4">
          {reports.length === 0 ? (
            <div className="text-center py-10 bg-white border border-dashed border-[#E2E8F0] text-[#64748B]">
              No intelligence reports logged yet.
            </div>
          ) : (
            reports.map(report => (
              <div key={report.id} className="bg-white border border-[#E2E8F0] p-4 shadow-sm relative group">
                <div className="flex justify-between items-start mb-2">
                  <div className="flex gap-2 items-center">
                    <span className={`text-xs font-semibold px-2 py-0.5 rounded-sm ${getBadgeColor(report.type)}`}>
                      {report.type}
                    </span>
                    <span className="text-xs text-[#64748B]">
                      {new Date(report.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </span>
                    <span className="text-[10px] uppercase font-bold tracking-wider text-green-600 bg-green-50 px-1 border border-green-200">
                      {report.status}
                    </span>
                  </div>
                  <button 
                    onClick={() => deleteReport(report.id)}
                    className="text-[#DC2626] opacity-0 group-hover:opacity-100 transition-opacity hover:bg-red-50 p-1"
                    title="Delete report"
                  >
                    <Trash2 size={16} />
                  </button>
                </div>
                {report.name && <p className="text-xs font-medium text-[#0F172A] mb-1">Reporter: {report.name} {report.time && `• At: ${report.time}`}</p>}
                <p className="text-sm text-[#0F172A] whitespace-pre-wrap">{report.text}</p>
              </div>
            ))
          )}
        </div>

        {/* Add New Report Button / Form */}
        {!showForm ? (
          <button 
            onClick={() => setShowForm(true)}
            className="w-full py-4 border-2 border-dashed border-[#E2E8F0] text-[#0F766E] font-medium hover:bg-white transition-colors mb-12"
          >
            + Add New Report
          </button>
        ) : (
          <div className="bg-white border border-[#E2E8F0] p-6 shadow-sm mb-12">
            <h3 className="font-semibold text-[#0F172A] mb-4">New Intelligence Report</h3>
            <form onSubmit={handleAddReport} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-[#0F172A] mb-1">Reporter Type</label>
                <div className="flex border border-[#E2E8F0] text-sm">
                  {['Eyewitness', 'Local Knowledge', 'Rescue Personnel', 'Other'].map(type => (
                    <button 
                      key={type} type="button"
                      onClick={() => setNewReport({...newReport, type})}
                      className={`flex-1 py-2 font-medium ${newReport.type === type ? 'bg-[#0F766E] text-white' : 'bg-white text-[#64748B] hover:bg-gray-50'}`}
                    >
                      {type}
                    </button>
                  ))}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-[#0F172A] mb-1">Reporter Name/ID (Optional)</label>
                  <input type="text" placeholder="e.g. Fisherman at dock" value={newReport.name} onChange={e => setNewReport({...newReport, name: e.target.value})} className="w-full p-2 border border-[#E2E8F0] focus:border-[#0F766E] outline-none" />
                </div>
                <div>
                  <label className="block text-sm font-medium text-[#0F172A] mb-1">Time of Observation (Optional)</label>
                  <input type="time" value={newReport.time} onChange={e => setNewReport({...newReport, time: e.target.value})} className="w-full p-2 border border-[#E2E8F0] focus:border-[#0F766E] outline-none" />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-[#0F172A] mb-1">Report Details <span className="text-[#DC2626]">*</span></label>
                <textarea 
                  required rows="4" 
                  placeholder="Describe what was seen or known in plain language. Include any reference to currents, sightings, weather observations, or other relevant information..."
                  value={newReport.text} onChange={e => setNewReport({...newReport, text: e.target.value})}
                  className="w-full p-2 border border-[#E2E8F0] focus:border-[#0F766E] outline-none"
                ></textarea>
              </div>

              <div className="flex justify-end gap-4 items-center pt-2">
                <button type="button" onClick={() => setShowForm(false)} className="text-[#64748B] text-sm font-medium hover:text-[#0F172A]">
                  Cancel
                </button>
                <button type="submit" className="px-6 py-2 bg-[#0F766E] text-white font-medium hover:bg-[#115E59]">
                  Add Report
                </button>
              </div>
            </form>
          </div>
        )}

        {/* Bottom Navigation */}
        <div className="flex items-center justify-between border-t border-[#E2E8F0] pt-6">
          <Link to="/" className="text-[#0F766E] font-medium hover:underline">
            ← Back to Incident
          </Link>
          
          <div className="flex gap-4">
            <Link to="/heatmap" className="px-6 py-3 bg-white border border-[#0F766E] text-[#0F766E] font-medium hover:bg-[#F0FDFA] transition-colors flex items-center justify-center">
              Skip for now
            </Link>
            
            <button 
              onClick={handleRunModel}
              disabled={isSimulating}
              className={`min-w-[200px] px-6 py-3 font-medium transition-colors flex justify-center items-center ${isSimulating ? 'bg-[#0F766E]/80 text-white cursor-wait' : 'bg-[#0F766E] text-white hover:bg-[#115E59]'}`}
            >
              {isSimulating ? (
                <>
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Analyzing...
                </>
              ) : 'Run Drift Model →'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
