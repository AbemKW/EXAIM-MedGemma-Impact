'use client';

import React, { useRef, useCallback, useState, useEffect } from 'react';
import { useSummaries, useCDSSStore, useTotalWords, useTotalSummaryWords } from '@/store/cdssStore';
import SummaryCard from './SummaryCard';
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
      
      // Scroll to ensure the expanded summary is fully visible
      // Use multiple timeouts to account for accordion animation and DOM updates
      setTimeout(() => {
        scrollToSummary(value, true);
      }, 400); // Wait for accordion animation to complete
      
      // Additional attempt after longer delay to ensure DOM is fully updated
      setTimeout(() => {
        scrollToSummary(value, true);
      }, 600);
    } else {
      // Collapsing - find the currently expanded one and toggle it
      if (expandedSummary) {
        toggleSummary(expandedSummary.id);
      }
    }
  };

  const scrollToSummary = useCallback((summaryId: string, ensureFullVisibility: boolean = false) => {
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
      const container = contentRef.current;
      
      // Use requestAnimationFrame to ensure DOM has updated after accordion animation
      requestAnimationFrame(() => {
        if (!container || !summaryElement) return;
        
        // Calculate positions using offsetTop for more reliable measurements
        let elementTop = summaryElement.offsetTop;
        let currentElement: HTMLElement | null = summaryElement;
        
        // Walk up the DOM tree to account for all parent offsets
        while (currentElement && currentElement !== container) {
          const parent = currentElement.offsetParent as HTMLElement | null;
          if (parent && parent !== container) {
            elementTop += currentElement.offsetTop;
          }
          currentElement = parent;
        }
        
        // Account for padding
        const topOffset = 16; // Padding offset
        const containerHeight = container.clientHeight;
        const elementHeight = summaryElement.offsetHeight;
        const currentScrollTop = container.scrollTop;
        
        if (ensureFullVisibility) {
          // Calculate the visible area
          const visibleTop = currentScrollTop;
          const visibleBottom = currentScrollTop + containerHeight;
          const elementBottom = elementTop + elementHeight;
          
          // Check if summary is fully visible
          const isFullyVisible = 
            elementTop >= visibleTop + topOffset && 
            elementBottom <= visibleBottom;
          
          if (!isFullyVisible) {
            // If summary is taller than viewport, scroll to show top
            if (elementHeight > containerHeight - topOffset) {
              container.scrollTo({
                top: Math.max(0, elementTop - topOffset),
                behavior: 'smooth',
              });
            } else {
              // Summary fits in viewport - ensure it's fully visible
              // Check if we need to scroll up or down
              if (elementTop < visibleTop + topOffset) {
                // Summary starts above visible area - scroll to show top
                container.scrollTo({
                  top: Math.max(0, elementTop - topOffset),
                  behavior: 'smooth',
                });
              } else if (elementBottom > visibleBottom) {
                // Summary extends below visible area - scroll to show bottom
                const targetScroll = elementBottom - containerHeight;
                container.scrollTo({
                  top: Math.max(0, targetScroll),
                  behavior: 'smooth',
                });
              }
            }
          }
        } else {
          // Simple scroll to top of summary
          container.scrollTo({
            top: Math.max(0, elementTop - topOffset),
            behavior: 'smooth',
          });
        }
      });
    }
  }, []);


  // Auto-scroll when a summary expands
  useEffect(() => {
    if (expandedSummary) {
      // Wait for accordion animation and DOM updates
      const timeoutId = setTimeout(() => {
        scrollToSummary(expandedSummary.id, true);
      }, 450);
      
      // Additional attempt after longer delay
      const timeoutId2 = setTimeout(() => {
        scrollToSummary(expandedSummary.id, true);
      }, 700);
      
      return () => {
        clearTimeout(timeoutId);
        clearTimeout(timeoutId2);
      };
    }
  }, [expandedSummary?.id, scrollToSummary]);

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

