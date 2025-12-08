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
    <div className="h-screen flex flex-col overflow-hidden relative liquid-glass-bg">
      <div className="max-w-[1800px] mx-auto px-6 py-4 flex flex-col gap-3 flex-1 min-h-0 w-full relative z-10">
        {/* Header */}
        <Header />

        {/* Chat Input Section */}
        <CaseInput />

        {/* Main Panels Section */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 flex-1 min-h-0 overflow-hidden">
          {/* Raw Agent Traces Panel */}
          <AgentTracesPanel />

          {/* EXAID Summaries Panel */}
          <SummariesPanel />
        </div>
      </div>

      {/* Modal */}
      <AgentModal />
    </div>
  );
}
