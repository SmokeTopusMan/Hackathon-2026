import React, { createContext, useState, useContext, useCallback } from 'react';

const IncidentContext = createContext();

export const useIncident = () => useContext(IncidentContext);

// API base — proxied to the Flask backend by vite during dev (see vite.config.js)
const API = import.meta.env.VITE_API ?? '/api';

export const IncidentProvider = ({ children }) => {
  const [incidentData, setIncidentData] = useState({
    id: `INC-${new Date().getFullYear()}-${Math.floor(Math.random() * 1000).toString().padStart(4, '0')}`,
    date: '',
    timeFrom: '',
    timeTo: '',
    waterBody: '',
    lspMode: 'map', // 'coordinates' or 'map'
    lat: '',
    lng: '',
    accuracy: 'Unknown',
    victimName: '',
    victimAge: '',
    victimGender: 'Unknown',
    victimHeight: '',
    victimWeight: '',
    victimClothing: '',
    swimmingAbility: 'Unknown',
    physicalCondition: 'Unknown',
    windDirection: '',
    windSpeed: '',
    waterConditions: '',
    waterTemp: ''
  });

  const [reports, setReports] = useState([]);

  // ---- live simulation state (shared across screens) ----------------------
  const [driftData, setDriftData] = useState(null);     // latest sim result
  const [currentHour, setCurrentHour] = useState(0);    // hour shown on the heatmap slider
  // `done` flips true once a run has fully finished AND its data is loaded; the
  // Incident Report watches it to navigate to the results (robust to promise timing).
  const [runState, setRunState] = useState({ running: false, percent: 0, stage: '', error: null, done: false });

  const updateIncident = (updates) => {
    setIncidentData(prev => ({ ...prev, ...updates }));
  };

  const addReport = (report) => {
    setReports(prev => [...prev, { ...report, id: Date.now(), timestamp: new Date().toISOString(), status: 'Pending Analysis' }]);
  };

  const deleteReport = (id) => {
    setReports(prev => prev.filter(r => r.id !== id));
  };

  // POST the incident, run the drift simulation live, poll progress for the
  // loading bar, then load the fresh result. Returns the drift data on success.
  const runSimulation = useCallback(async (incident) => {
    setRunState({ running: true, percent: 0, stage: 'Starting simulation…', error: null, done: false });
    try {
      const start = await fetch(`${API}/simulate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(incident),
      });
      if (!start.ok) throw new Error(`simulate failed (HTTP ${start.status})`);
      const { job_id } = await start.json();

      // poll progress until done
      for (;;) {
        await new Promise(r => setTimeout(r, 700));
        const pr = await fetch(`${API}/progress/${job_id}`);
        if (!pr.ok) throw new Error(`progress failed (HTTP ${pr.status})`);
        const st = await pr.json();
        setRunState({ running: !st.done, percent: st.percent ?? 0, stage: st.stage ?? '', error: st.error ?? null, done: false });
        if (st.done) {
          if (st.error) throw new Error(st.error);
          break;
        }
      }

      setRunState({ running: true, percent: 99, stage: 'Loading results…', error: null, done: false });
      // Load the result best-effort. Navigation must NOT depend on this — the
      // results screens fetch the data themselves — so a hiccup here can't
      // block moving to the next screen.
      let data = null;
      try {
        const dd = await fetch(`${API}/drift_data?t=${Date.now()}`);
        if (dd.ok) {
          data = await dd.json();
          setDriftData(data);
          setCurrentHour(data?.search_plan?.plan_hour ?? 0);   // land on a meaningful frame
        }
      } catch (loadErr) {
        console.warn('result will be (re)loaded by the results screen:', loadErr);
      }
      // `done:true` always fires here -> triggers navigation in the Incident Report
      setRunState({ running: false, percent: 100, stage: 'Done', error: null, done: true });
      return data;
    } catch (e) {
      setRunState({ running: false, percent: 0, stage: '', error: e.message, done: false });
      throw e;
    }
  }, []);

  // Recompute the coordinated search plan for a specific forecast hour
  // (the hour currently shown on the heatmap). Teams launch from shore.
  const fetchPlanForHour = useCallback(async (hour) => {
    const r = await fetch(`${API}/plan?hour=${hour}`);
    if (!r.ok) throw new Error(`plan failed (HTTP ${r.status})`);
    return r.json();
  }, []);

  return (
    <IncidentContext.Provider value={{
      incidentData, updateIncident, reports, addReport, deleteReport,
      driftData, setDriftData, currentHour, setCurrentHour,
      runState, setRunState, runSimulation, fetchPlanForHour,
    }}>
      {children}
    </IncidentContext.Provider>
  );
};
