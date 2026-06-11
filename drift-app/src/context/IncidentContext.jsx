import React, { createContext, useState, useContext } from 'react';

const IncidentContext = createContext();

export const useIncident = () => useContext(IncidentContext);

export const IncidentProvider = ({ children }) => {
  const [incidentData, setIncidentData] = useState({
    id: `INC-${new Date().getFullYear()}-${Math.floor(Math.random() * 1000).toString().padStart(4, '0')}`,
    date: '',
    time: '',
    lspMode: 'map', // 'coordinates' or 'map'
    lat: '',
    lng: '',
    victimAge: '',
    victimGender: 'Unknown',
    victimHeight: '',
    victimWeight: ''
  });

  const [reports, setReports] = useState([]);

  const updateIncident = (updates) => {
    setIncidentData(prev => ({ ...prev, ...updates }));
  };

  const addReport = (report) => {
    setReports(prev => [...prev, { ...report, id: Date.now(), timestamp: new Date().toISOString(), status: 'Pending Analysis' }]);
  };

  const deleteReport = (id) => {
    setReports(prev => prev.filter(r => r.id !== id));
  };

  return (
    <IncidentContext.Provider value={{ incidentData, updateIncident, reports, addReport, deleteReport }}>
      {children}
    </IncidentContext.Provider>
  );
};
