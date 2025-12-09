'use client';

import React, { useState } from 'react';
import { useSummaries, useTotalWords, useTotalSummaryWords } from '@/store/cdssStore';
import SummaryCard from './SummaryCard';
import CompressionStats from './CompressionStats';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';

export default function SummariesPanel() {
  const summaries = useSummaries();
  const totalWords = useTotalWords();
  const totalSummaryWords = useTotalSummaryWords();
  const [comparisonMode, setComparisonMode] = useState(false);
  
  // Get only the spotlight summary (latest/expanded)
  const spotlightSummary = summaries.find(s => s.isExpanded) || summaries[0];
  
  // Debug: Log summaries state
  React.useEffect(() => {
    console.log('SummariesPanel - summaries:', summaries.length);
    console.log('SummariesPanel - spotlightSummary:', spotlightSummary?.id);
  }, [summaries, spotlightSummary]);

  return (
    <Card className="flex flex-col overflow-hidden h-full bg-card/30 backdrop-blur-xl border-white/10 glass-card">
      {/* Panel Header */}
      <CardHeader className="bg-gradient-to-r from-zinc-950/40 to-zinc-900/30 backdrop-blur-md border-b border-white/10 py-3 px-5 glass-header">
        <div className="flex justify-between items-center">
          <CardTitle className="text-xl font-bold">EXAID Summaries</CardTitle>
          <div className="flex items-center gap-2">
            <Button
              variant={comparisonMode ? 'default' : 'outline'}
              size="sm"
              onClick={() => setComparisonMode(!comparisonMode)}
              className="text-xs"
            >
              {comparisonMode ? '✓ Comparison' : 'Comparison'}
            </Button>
            <Badge variant="secondary" className="text-sm">
              {summaries.length} summar{summaries.length !== 1 ? 'ies' : 'y'}
            </Badge>
          </div>
        </div>
      </CardHeader>

      {/* Panel Content - Only Spotlight Summary */}
      <div className="flex-1 flex flex-col overflow-hidden min-h-0">
        {!spotlightSummary ? (
          <CardContent className="flex-1 flex items-center justify-center px-4 py-8">
            <p className="text-muted-foreground text-center text-base">
              No summaries yet. Summaries will appear here as EXAID processes agent traces.
            </p>
          </CardContent>
        ) : (
          <div className="flex-1 flex flex-col overflow-hidden px-4 pt-2 pb-2">
            {comparisonMode && (
              <div className="mb-2 flex-shrink-0">
                <CompressionStats
                  totalWords={totalWords}
                  summaryWords={totalSummaryWords}
                />
              </div>
            )}
            <div className="mb-1 flex-shrink-0">
              <div className="text-xs font-semibold text-teal-400 uppercase tracking-wider flex items-center gap-2">
                <span className="inline-block w-2 h-2 bg-teal-400 rounded-full animate-pulse"></span>
                Latest Summary
              </div>
            </div>
            <div className="flex-1 overflow-y-auto custom-scrollbar min-h-0">
              <SummaryCard
                key={spotlightSummary.id}
                summary={spotlightSummary}
                showComparison={comparisonMode}
                mode="spotlight"
              />
            </div>
          </div>
        )}
      </div>
    </Card>
  );
}

