'use client';

import React, { useRef, useCallback, useState } from 'react';
import { useSummaries, useCDSSStore, useTotalWords, useTotalSummaryWords } from '@/store/cdssStore';
import SummaryCard from './SummaryCard';
import SummaryTimeline from './SummaryTimeline';
import CompressionStats from './CompressionStats';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Accordion } from '@/components/ui/accordion';

export default function SummariesPanel() {
  const summaries = useSummaries();
  const totalWords = useTotalWords();
  const totalSummaryWords = useTotalSummaryWords();
  const toggleSummary = useCDSSStore((state) => state.toggleSummary);
  const expandedSummary = summaries.find(s => s.isExpanded);
  const contentRef = useRef<HTMLDivElement>(null);
  const [comparisonMode, setComparisonMode] = useState(false);
  // Use a ref callback to maintain stable refs
  const summaryRefsMap = useRef<Map<string, HTMLDivElement>>(new Map());
  
  const setSummaryRef = useCallback((summaryId: string, element: HTMLDivElement | null) => {
    if (element) {
      summaryRefsMap.current.set(summaryId, element);
    } else {
      summaryRefsMap.current.delete(summaryId);
    }
  }, []);

  const handleValueChange = (value: string) => {
    if (value) {
      // Expanding a summary - toggle it (which will collapse others)
      toggleSummary(value);
    } else {
      // Collapsing - find the currently expanded one and toggle it
      if (expandedSummary) {
        toggleSummary(expandedSummary.id);
      }
    }
  };

  const handleTimelineDotClick = useCallback((summaryId: string) => {
    // Always expand the summary (toggleSummary will expand if collapsed)
    const isAlreadyExpanded = expandedSummary?.id === summaryId;
    
    if (!isAlreadyExpanded) {
      toggleSummary(summaryId);
    }
    
    // Function to perform the scroll - try multiple methods
    const performScroll = () => {
      if (!contentRef.current) return;
      
      // Try ref first
      let summaryElement = summaryRefsMap.current.get(summaryId);
      
      // Fallback: find by data attribute
      if (!summaryElement && contentRef.current) {
        const foundElement = contentRef.current.querySelector(`[data-summary-id="${summaryId}"]`) as HTMLDivElement;
        if (foundElement) {
          summaryElement = foundElement;
        }
      }
      
      if (summaryElement && contentRef.current) {
        // Calculate position relative to scroll container
        const containerRect = contentRef.current.getBoundingClientRect();
        const elementRect = summaryElement.getBoundingClientRect();
        const scrollTop = contentRef.current.scrollTop;
        const elementTop = elementRect.top - containerRect.top + scrollTop;
        
        // Account for timeline and padding
        const offset = 80; // Timeline height + padding
        
        contentRef.current.scrollTo({
          top: Math.max(0, elementTop - offset),
          behavior: 'smooth',
        });
      }
    };
    
    // Wait for accordion animation to complete
    // Use longer timeout if we need to expand, shorter if already expanded
    const timeout = isAlreadyExpanded ? 100 : 500;
    
    setTimeout(performScroll, timeout);
    
    // Also try after a longer delay as fallback
    setTimeout(performScroll, timeout + 300);
  }, [toggleSummary, expandedSummary]);

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

      {/* Panel Content */}
      <CardContent className="flex-1 overflow-y-auto p-4 custom-scrollbar" ref={contentRef}>
        {summaries.length === 0 ? (
          <p className="text-muted-foreground text-center py-8 text-base">
            No summaries yet. Summaries will appear here as EXAID processes agent traces.
          </p>
        ) : (
          <>
            {comparisonMode && (
              <div className="mb-4">
                <CompressionStats
                  totalWords={totalWords}
                  summaryWords={totalSummaryWords}
                />
              </div>
            )}
            <SummaryTimeline
              summaries={summaries}
              expandedSummaryId={expandedSummary?.id}
              onDotClick={handleTimelineDotClick}
              summaryRefs={new Map()} // Not used anymore but kept for compatibility
            />
            <Accordion
              type="single"
              collapsible
              value={expandedSummary?.id || undefined}
              onValueChange={handleValueChange}
              className="space-y-2 mt-4"
            >
              {summaries.map((summary) => {
                return (
                  <SummaryCard
                    key={summary.id}
                    ref={(el) => setSummaryRef(summary.id, el)}
                    summary={summary}
                    showComparison={comparisonMode}
                  />
                );
              })}
            </Accordion>
          </>
        )}
      </CardContent>
    </Card>
  );
}

