'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { useCDSSStore } from '@/store/cdssStore';
import type { Summary } from '@/lib/types';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import {
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';

interface SummaryCardProps {
  summary: Summary;
}

function SummaryCard({ summary }: SummaryCardProps) {

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

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <Card className={`overflow-hidden transition-all duration-500 border-white/10 backdrop-blur-md ${
        summary.isExpanded
          ? 'border-white/20 bg-teal-950/25 backdrop-blur-xl shadow-xl'
          : 'bg-card/40 backdrop-blur-md'
      } glass-card`}>
        <AccordionItem value={summary.id} className="border-0">
          {/* Header - Always Visible */}
          <CardHeader className="cursor-pointer hover:bg-teal-950/15 transition-colors py-3 px-4">
            <AccordionTrigger className="hover:no-underline py-0">
              <div className="flex-1 pr-4 text-left">
                <CardTitle className="text-sm line-clamp-1 font-semibold">
                  {summary.data.status_action}
                </CardTitle>
              </div>
              <div className="flex items-center gap-3 flex-shrink-0">
                <span className="text-xs text-muted-foreground font-medium">
                  {formatTime(summary.timestamp)}
                </span>
              </div>
            </AccordionTrigger>
          </CardHeader>

          {/* Content - Expandable */}
          <AccordionContent>
            <CardContent className="pt-0 pb-4 px-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
                {fields.map((field, index) => (
                  <div 
                    key={index} 
                    className="summary-field-group rounded-lg p-2.5 transition-all backdrop-blur-md border border-white/10 hover:border-white/20"
                    style={{
                      borderLeft: `2px solid ${field.color}`,
                      backgroundColor: field.bgColor,
                      boxShadow: 'inset 0 1px 1px 0 rgba(255, 255, 255, 0.05)',
                    }}
                  >
                    <div 
                      className="text-xs font-semibold uppercase tracking-wide mb-2 leading-tight"
                      style={{ color: field.color }}
                    >
                      {field.label}
                    </div>
                    <div className="text-sm text-foreground leading-relaxed">{field.value}</div>
                  </div>
                ))}
              </div>
            </CardContent>
          </AccordionContent>
        </AccordionItem>
      </Card>
    </motion.div>
  );
}

// Memoize to prevent unnecessary re-renders
export default React.memo(SummaryCard);

