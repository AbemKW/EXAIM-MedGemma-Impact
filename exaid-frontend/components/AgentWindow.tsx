'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { useCDSSStore, useAgentTrace } from '@/store/cdssStore';
import AgentWindowContent from './AgentWindowContent';

interface AgentWindowProps {
  agentId: string;
}

function AgentWindow({ agentId }: AgentWindowProps) {
  const agent = useAgentTrace(agentId);
  const toggleAgent = useCDSSStore((state) => state.toggleAgent);
  const openModal = useCDSSStore((state) => state.openModal);

  if (!agent) return null;

  const handleToggle = () => {
    toggleAgent(agentId);
  };

  const handleViewFull = (e: React.MouseEvent) => {
    e.stopPropagation();
    openModal(agentId, agent.fullText);
  };

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden"
    >
      {/* Header */}
      <div
        onClick={handleToggle}
        className="px-4 py-3 bg-gray-50 border-b border-gray-200 cursor-pointer hover:bg-gray-100 transition-colors flex justify-between items-center"
      >
        <div className="flex items-center gap-3">
          <span className="font-semibold text-gray-800">{agentId}</span>
          <span className="px-2 py-1 bg-green-100 text-green-700 text-xs font-semibold rounded uppercase">
            Active
          </span>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={handleViewFull}
            className="px-3 py-1 bg-blue-600 text-white text-sm rounded hover:bg-blue-700 transition-colors"
          >
            View Full
          </button>
          <motion.span
            animate={{ rotate: agent.isExpanded ? 180 : 0 }}
            transition={{ duration: 0.3 }}
            className="text-gray-600"
          >
            ▼
          </motion.span>
        </div>
      </div>

      {/* Content */}
      <motion.div
        initial={false}
        animate={{
          height: agent.isExpanded ? '400px' : '200px',
        }}
        transition={{ duration: 0.3, ease: 'easeInOut' }}
        className="overflow-hidden"
      >
        <AgentWindowContent
          fullText={agent.fullText}
          isExpanded={agent.isExpanded}
        />
      </motion.div>
    </motion.div>
  );
}

// Memoize to prevent unnecessary re-renders
export default React.memo(AgentWindow);
