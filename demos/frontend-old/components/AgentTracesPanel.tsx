'use client';

import { useAllAgents } from '@/store/cdssStore';
import AgentWindow from './AgentWindow';

export default function AgentTracesPanel() {
  // Subscribe to agents array directly
  const agents = useAllAgents();

  return (
    <div className="flex flex-col bg-white rounded-lg shadow-md border border-gray-200 overflow-hidden h-full">
      {/* Panel Header */}
      <div className="px-6 py-4 bg-gradient-to-r from-blue-50 to-blue-100 border-b border-gray-200">
        <div className="flex justify-between items-center">
          <h2 className="text-xl font-bold text-gray-800">Raw Agent Traces</h2>
          <span className="px-3 py-1 bg-blue-600 text-white text-sm font-semibold rounded-full">
            {agents.length} trace{agents.length !== 1 ? 's' : ''}
          </span>
        </div>
      </div>

      {/* Panel Content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-2">
        {agents.length === 0 ? (
          <p className="text-gray-500 text-center py-8">
            No traces yet. Process a case to see verbose reasoning traces.
          </p>
        ) : (
          agents.map((agent) => (
            <AgentWindow key={agent.id} cardId={agent.id} />
          ))
        )}
      </div>
    </div>
  );
}
