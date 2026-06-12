import React from 'react';
import { Outlet, useLocation, useNavigate } from 'react-router-dom';
import logo from '../assets/logo.png';

const STEPS = [
  { path: '/', label: 'Incident Report' },
  { path: '/heatmap', label: 'Heatmap' },
  { path: '/search-plan', label: 'Search Plan' }
];

export default function Layout() {
  const location = useLocation();
  const navigate = useNavigate();
  const currentStepIndex = STEPS.findIndex(step => step.path === location.pathname);

  const handleStepClick = (stepPath) => {
    if (location.pathname === '/' && stepPath !== '/') return;
    navigate(stepPath);
  };

  return (
    <div className="min-h-screen bg-[#F8F9FA] flex flex-col text-[#0F172A] font-sans">
      <header className="bg-white border-b border-[#E2E8F0] h-16 flex items-center px-6 shrink-0 z-[1000] sticky top-0 relative">
        <button type="button" onClick={() => navigate('/')} title="Back to start"
          className="flex items-center gap-3 mr-8 cursor-pointer hover:opacity-80 outline-none">
          <img src={logo} alt="Nahshol" className="w-10 h-10 rounded-full" />
          <div className="flex flex-col items-start leading-none">
            <span className="font-bold text-xl tracking-tight text-[#13366A]">Nahshol</span>
            <span className="text-[10px] uppercase tracking-[0.18em] text-[#1E5C9E]">Simulate · Predict · Plan</span>
          </div>
        </button>

        <nav className="flex-1 flex items-center justify-center">
          <ol className="flex items-center w-full max-w-3xl">
            {STEPS.map((step, index) => {
              const isCompleted = index < currentStepIndex;
              const isCurrent = index === currentStepIndex;
              const isClickable = location.pathname !== '/' || step.path === '/';

              return (
                <li key={step.path} className={`flex items-center ${index < STEPS.length - 1 ? 'flex-1' : ''}`}>
                  <button
                    onClick={() => handleStepClick(step.path)}
                    disabled={!isClickable}
                    className={`flex items-center outline-none ${isClickable ? 'cursor-pointer hover:opacity-80' : 'cursor-default'}`}
                  >
                    <span className={`
                      flex items-center justify-center w-6 h-6 rounded-full text-xs font-semibold
                      ${isCurrent ? 'bg-[#1E5C9E] text-white' :
                        isCompleted ? 'bg-[#1E5C9E]/20 text-[#1E5C9E]' : 'bg-[#E2E8F0] text-[#64748B]'}
                    `}>
                      {isCompleted ? '✓' : index + 1}
                    </span>
                    <span className={`ml-2 text-sm font-medium ${isCurrent ? 'text-[#0F172A]' : 'text-[#64748B]'}`}>
                      {step.label}
                    </span>
                  </button>
                  {index < STEPS.length - 1 && (
                    <div className={`flex-1 h-px mx-4 ${isCompleted ? 'bg-[#1E5C9E]/50' : 'bg-[#E2E8F0]'}`}></div>
                  )}
                </li>
              );
            })}
          </ol>
        </nav>
      </header>

      <main className="flex-1 flex flex-col overflow-hidden">
        <Outlet />
      </main>
    </div>
  );
}
