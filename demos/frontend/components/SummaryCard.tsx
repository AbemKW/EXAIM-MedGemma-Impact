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
    >
      <Card className={`overflow-hidden transition-colors ${
        summary.isExpanded
          ? 'border-primary/50 bg-primary/5'
          : ''
      }`}>
        <AccordionItem value={summary.id} className="border-0">
          {/* Header - Always Visible */}
          <CardHeader className="cursor-pointer hover:bg-muted/50 transition-colors py-3 px-4">
            <AccordionTrigger className="hover:no-underline py-0">
              <div className="flex-1 pr-4 text-left">
                <CardTitle className="text-sm line-clamp-2 font-semibold">
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
            <CardContent className="pt-0 pb-4 px-4 space-y-4">
              {fields.map((field, index) => (
                <div key={index} className="space-y-2">
                  <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
                    {field.label}
                  </div>
                  <div className="text-sm text-foreground leading-relaxed">{field.value}</div>
                </div>
              ))}
            </CardContent>
          </AccordionContent>
        </AccordionItem>
      </Card>
    </motion.div>
  );
}

// Memoize to prevent unnecessary re-renders
export default React.memo(SummaryCard);

