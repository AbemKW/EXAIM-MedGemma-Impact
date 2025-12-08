'use client';

import { useEffect } from 'react';
import Header from '@/components/Header';
import CaseInput from '@/components/CaseInput';
import AgentTracesPanel from '@/components/AgentTracesPanel';
import SummariesPanel from '@/components/SummariesPanel';
import AgentModal from '@/components/AgentModal';
import { getWebSocketService } from '@/lib/websocket';

export default function Home() {
  // Initialize WebSocket connection on mount
  useEffect(() => {
    const ws = getWebSocketService();
    ws.connect();

    // Cleanup on unmount
    return () => {
      ws.disconnect();
    };
  }, []);

  return (
    <div className="relative liquid-glass-bg">
      {/* Fixed Header */}
      <Header />

      {/* Scroll-Snap Container */}
      <main className="snap-container h-screen overflow-y-auto">
        {/* Section 1: Case Input - Full viewport height */}
        <section className="snap-section h-screen pt-[var(--header-height)] flex items-center justify-center px-6">
          <div className="w-full max-w-3xl">
            <CaseInput />
          </div>
        </section>

        {/* Section 2: Reasoning Panels - Full viewport height */}
        <section className="snap-section h-screen pt-[calc(var(--header-height)+1rem)] pb-4 px-6">
          <div className="max-w-[1800px] mx-auto h-full">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 h-full">
              {/* Raw Agent Traces Panel */}
              <AgentTracesPanel />

              {/* EXAID Summaries Panel */}
              <SummariesPanel />
            </div>
          </div>
        </section>
      </main>

      {/* Modal */}
      <AgentModal />
    </div>
  );
}
