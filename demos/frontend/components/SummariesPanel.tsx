'use client';

import React, { useState } from 'react';
import { useSummaries, useCDSSStore, useTotalWords, useTotalSummaryWords } from '@/store/cdssStore';
import SummaryCard from './SummaryCard';
import CompressionStats from './CompressionStats';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';

export default function SummariesPanel() {
  const summaries = useSummaries();
  const totalWords = useTotalWords();
  const totalSummaryWords = useTotalSummaryWords();
  const toggleSummary = useCDSSStore((state) => state.toggleSummary);
  const [comparisonMode, setComparisonMode] = useState(false);
  
  // Split summaries into spotlight and list
  const spotlightSummary = summaries.find(s => s.isExpanded);
  const listSummaries = summaries.filter(s => !s.isExpanded);
  
  const handleSummaryClick = (id: string) => {
    toggleSummary(id);
  };

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

      {/* Panel Content - Split Layout */}
      <div className="flex-1 flex flex-col overflow-hidden min-h-0">
        {summaries.length === 0 ? (
          <CardContent className="flex-1 overflow-y-auto px-4 pt-2 pb-4">
            <p className="text-muted-foreground text-center py-8 text-base">
              No summaries yet. Summaries will appear here as EXAID processes agent traces.
            </p>
          </CardContent>
        ) : (
          <>
            {/* Spotlight Section - Takes available space, scrollable if needed */}
            {spotlightSummary && (
              <div className="spotlight-section flex-1 overflow-y-auto px-4 pt-4 pb-2 custom-scrollbar min-h-0">
                {comparisonMode && (
                  <div className="mb-3">
                    <CompressionStats
                      totalWords={totalWords}
                      summaryWords={totalSummaryWords}
                    />
                  </div>
                )}
                <div className="mb-2">
                  <div className="text-xs font-semibold text-teal-400 mb-2 uppercase tracking-wider flex items-center gap-2">
                    <span className="inline-block w-2 h-2 bg-teal-400 rounded-full animate-pulse"></span>
                    Latest Summary
                  </div>
                  <SummaryCard
                    key={spotlightSummary.id}
                    summary={spotlightSummary}
                    showComparison={comparisonMode}
                    mode="spotlight"
                  />
                </div>
              </div>
            )}

            {/* Divider */}
            {spotlightSummary && listSummaries.length > 0 && (
              <div className="border-t border-white/10 mx-4 my-2 flex-shrink-0"></div>
            )}

            {/* List Section - Scrollable, only if there are previous summaries */}
            {listSummaries.length > 0 && (
              <div className="flex-1 overflow-y-auto px-4 pt-2 pb-4 custom-scrollbar min-h-0" style={{ maxHeight: '40%' }}>
                <div className="text-xs font-semibold text-muted-foreground mb-2 uppercase tracking-wider">
                  Previous Summaries ({listSummaries.length})
                </div>
                <div className="space-y-1.5">
                  {listSummaries.map((summary) => (
                    <SummaryCard
                      key={summary.id}
                      summary={summary}
                      showComparison={comparisonMode}
                      mode="list"
                      onClick={() => handleSummaryClick(summary.id)}
                    />
                  ))}
                </div>
              </div>
            )}

            {/* If only spotlight exists, no list section */}
            {spotlightSummary && listSummaries.length === 0 && (
              <div className="flex-shrink-0 px-4 pb-4">
                <p className="text-muted-foreground text-center text-sm py-4">
                  No previous summaries yet.
                </p>
              </div>
            )}
          </>
        )}
      </div>
    </Card>
  );
}

