'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { useCDSSStore } from '@/store/cdssStore';
import type { Summary } from '@/lib/types';

interface SummaryCardProps {
  summary: Summary;
}

function SummaryCard({ summary }: SummaryCardProps) {
  const toggleSummary = useCDSSStore((state) => state.toggleSummary);

  const handleToggle = () => {
    toggleSummary(summary.id);
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  };

  const fields = [
    { label: 'Status / Action', value: summary.data.status_action },
    { label: 'Key Findings', value: summary.data.key_findings },
    { label: 'Differential & Rationale', value: summary.data.differential_rationale },
    { label: 'Uncertainty / Confidence', value: summary.data.uncertainty_confidence },
    { label: 'Recommendation / Next Step', value: summary.data.recommendation_next_step },
    { label: 'Agent Contributions', value: summary.data.agent_contributions },
  ];

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`bg-white rounded-lg shadow-sm border overflow-hidden transition-colors ${
        summary.isExpanded
          ? 'border-blue-300 bg-blue-50'
          : 'border-gray-200'
      }`}
    >
      {/* Header - Always Visible */}
      <div
        onClick={handleToggle}
        className="px-4 py-3 cursor-pointer hover:bg-gray-50 transition-colors flex justify-between items-center"
      >
        <div className="flex-1 pr-4">
          <div className="font-semibold text-gray-800 text-sm line-clamp-2">
            {summary.data.status_action}
          </div>
        </div>
        <div className="flex items-center gap-3 flex-shrink-0">
          <span className="text-xs text-gray-500 font-medium">
            {formatTime(summary.timestamp)}
          </span>
          <motion.span
            animate={{ rotate: summary.isExpanded ? 180 : 0 }}
            transition={{ duration: 0.3 }}
            className="text-gray-600"
          >
            ▼
          </motion.span>
        </div>
      </div>

      {/* Content - Expandable */}
      <motion.div
        initial={false}
        animate={{
          height: summary.isExpanded ? 'auto' : 0,
          opacity: summary.isExpanded ? 1 : 0,
        }}
        transition={{ duration: 0.3, ease: 'easeInOut' }}
        className="overflow-hidden"
      >
        <div className="px-4 py-4 border-t border-gray-200 space-y-3 bg-white">
          {fields.map((field, index) => (
            <div key={index} className="space-y-1">
              <div className="text-xs font-semibold text-gray-600 uppercase tracking-wide">
                {field.label}
              </div>
              <div className="text-sm text-gray-800">{field.value}</div>
            </div>
          ))}
        </div>
      </motion.div>
    </motion.div>
  );
}

// Memoize to prevent unnecessary re-renders
export default React.memo(SummaryCard);
