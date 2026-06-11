import React from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import { Waves, Eye } from 'lucide-react';

const STEPS = [
  { path: '/', label: 'Incident Report' },
  { path: '/intelligence', label: 'Intelligence' },
  { path: '/heatmap', label: 'Heatmap' },
  { path: '/search-plan', label: 'Search Plan' }
];

export default function Layout() {
  const location = useLocation();
  const currentStepIndex = STEPS.findIndex(step => step.path === location.pathname);

  return (
    <div className="min-h-screen bg-[#F8F9FA] flex flex-col text-[#0F172A] font-sans">
      <header className="bg-white border-b border-[#E2E8F0] h-16 flex items-center px-6 shrink-0 z-[1000] sticky top-0 relative">
        <div className="flex items-center gap-2 mr-8">
          <div className="relative flex items-center justify-center w-8 h-8 bg-[#0F766E] rounded-sm text-white">
            <Waves size={20} />
            <Eye size={12} className="absolute bottom-1 right-1" />
          </div>
          <span className="font-bold text-xl tracking-tight text-[#0F172A]">Nahshol</span>
        </div>
        
        <nav className="flex-1 flex items-center justify-center">
          <ol className="flex items-center w-full max-w-3xl">
            {STEPS.map((step, index) => {
              const isCompleted = index < currentStepIndex;
              const isCurrent = index === currentStepIndex;
              
              return (
                <li key={step.path} className={`flex items-center ${index < STEPS.length - 1 ? 'flex-1' : ''}`}>
                  <div className="flex items-center">
                    <span className={`
                      flex items-center justify-center w-6 h-6 rounded-full text-xs font-semibold
                      ${isCurrent ? 'bg-[#0F766E] text-white' : 
                        isCompleted ? 'bg-[#0F766E]/20 text-[#0F766E]' : 'bg-[#E2E8F0] text-[#64748B]'}
                    `}>
                      {isCompleted ? '✓' : index + 1}
                    </span>
                    <span className={`ml-2 text-sm font-medium ${isCurrent ? 'text-[#0F172A]' : 'text-[#64748B]'}`}>
                      {step.label}
                    </span>
                  </div>
                  {index < STEPS.length - 1 && (
                    <div className={`flex-1 h-px mx-4 ${isCompleted ? 'bg-[#0F766E]/50' : 'bg-[#E2E8F0]'}`}></div>
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
