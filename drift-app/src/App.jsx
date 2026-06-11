import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { IncidentProvider } from './context/IncidentContext';
import Layout from './components/Layout';
import IncidentReport from './screens/IncidentReport';
import DriftHeatmap from './screens/DriftHeatmap';
import SearchPlan from './screens/SearchPlan';

function App() {
  return (
    <IncidentProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<IncidentReport />} />
            <Route path="heatmap" element={<DriftHeatmap />} />
            <Route path="search-plan" element={<SearchPlan />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </IncidentProvider>
  );
}

export default App;
