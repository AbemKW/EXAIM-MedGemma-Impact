'use client';

import React, { forwardRef } from 'react';
import { motion } from 'framer-motion';
import { useCDSSStore } from '@/store/cdssStore';
import type { Summary } from '@/lib/types';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';

interface SummaryCardProps {
  summary: Summary;
  showComparison?: boolean;
  mode?: 'spotlight' | 'list';
  onClick?: () => void;
}

const SummaryCard = forwardRef<HTMLDivElement, SummaryCardProps>(({ 
  summary, 
  showComparison = false, 
  mode = 'list',
  onClick 
}, ref) => {

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  };

  const fields = [
    { 
      label: 'Status / Action', 
      value: summary.data.status_action,
      color: 'var(--summary-status-action)',
      bgColor: 'oklch(0.50 0.08 260 / 0.04)',
    },
    { 
      label: 'Key Findings', 
      value: summary.data.key_findings,
      color: 'var(--summary-key-findings)',
      bgColor: 'oklch(0.55 0.08 150 / 0.04)',
    },
    { 
      label: 'Differential & Rationale', 
      value: summary.data.differential_rationale,
      color: 'var(--summary-differential)',
      bgColor: 'oklch(0.50 0.08 300 / 0.04)',
    },
    { 
      label: 'Uncertainty / Confidence', 
      value: summary.data.uncertainty_confidence,
      color: 'var(--summary-uncertainty)',
      bgColor: 'oklch(0.58 0.08 60 / 0.04)',
    },
    { 
      label: 'Recommendation / Next Step', 
      value: summary.data.recommendation_next_step,
      color: 'var(--summary-recommendation)',
      bgColor: 'oklch(0.50 0.08 180 / 0.04)',
    },
    { 
      label: 'Agent Contributions', 
      value: summary.data.agent_contributions,
      color: 'var(--summary-contributions)',
      bgColor: 'oklch(0.50 0.03 0 / 0.04)',
    },
  ];

  // Spotlight mode - always show full expanded content
  if (mode === 'spotlight') {
    return (
      <motion.div
        ref={ref}
        layout
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.95 }}
        transition={{ duration: 0.3 }}
        data-summary-id={summary.id}
      >
        <Card className="overflow-hidden transition-all duration-300 border-white/20 bg-teal-950/30 backdrop-blur-xl shadow-2xl glass-card spotlight-glow h-full flex flex-col">
          {/* Header */}
          <CardHeader className="py-1.5 px-2 border-b border-white/10 flex-shrink-0">
            <div className="flex justify-between items-center">
              <CardTitle className="text-sm font-bold text-teal-100 leading-tight">
                {summary.data.status_action}
              </CardTitle>
              <span className="text-xs text-muted-foreground font-medium flex-shrink-0 ml-2">
                {formatTime(summary.timestamp)}
              </span>
            </div>
          </CardHeader>

          {/* Full Content - Always Visible */}
          <CardContent className="pt-0.5 pb-1 px-2 flex-1 overflow-hidden">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-1 items-start h-full">
              {fields.map((field, index) => (
                <div 
                  key={index} 
                  className="summary-field-group rounded-lg p-1 transition-all backdrop-blur-md border border-white/10 hover:border-white/20 min-h-0"
                  style={{
                    borderLeft: `2px solid ${field.color}`,
                    backgroundColor: field.bgColor,
                    boxShadow: 'inset 0 1px 1px 0 rgba(255, 255, 255, 0.05)',
                  }}
                >
                  <div 
                    className="text-xs font-extrabold uppercase tracking-wider mb-0.5 leading-tight"
                    style={{ 
                      color: field.color,
                      fontWeight: 800,
                      letterSpacing: '0.05em'
                    }}
                  >
                    {field.label}
                  </div>
                  <div className="text-xs text-foreground leading-relaxed font-medium break-words" style={{ fontWeight: 500 }}>{field.value}</div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </motion.div>
    );
  }

  // List mode - show collapsed header only
  return (
    <motion.div
      ref={ref}
      layout
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      transition={{ duration: 0.2 }}
      data-summary-id={summary.id}
    >
      <Card 
        className="overflow-hidden transition-all duration-200 border-white/10 bg-card/40 backdrop-blur-md hover:bg-card/60 hover:border-white/20 cursor-pointer glass-card"
        onClick={onClick}
      >
        {/* Header - Clickable */}
        <CardHeader className="py-2 px-4">
          <div className="flex justify-between items-center">
            <CardTitle className="text-sm line-clamp-1 font-semibold flex-1 pr-4">
              {summary.data.status_action}
            </CardTitle>
            <div className="flex items-center gap-2 flex-shrink-0">
              <span className="text-xs text-muted-foreground font-medium">
                {formatTime(summary.timestamp)}
              </span>
              <svg 
                xmlns="http://www.w3.org/2000/svg" 
                width="16" 
                height="16" 
                viewBox="0 0 24 24" 
                fill="none" 
                stroke="currentColor" 
                strokeWidth="2" 
                strokeLinecap="round" 
                strokeLinejoin="round"
                className="text-muted-foreground"
              >
                <polyline points="9 18 15 12 9 6"></polyline>
              </svg>
            </div>
          </div>
        </CardHeader>
      </Card>
    </motion.div>
  );
});

SummaryCard.displayName = 'SummaryCard';

// Memoize to prevent unnecessary re-renders
export default React.memo(SummaryCard);

