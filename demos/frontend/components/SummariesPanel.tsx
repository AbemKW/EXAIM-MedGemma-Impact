'use client';

import { useSummaries, useCDSSStore } from '@/store/cdssStore';
import SummaryCard from './SummaryCard';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Accordion } from '@/components/ui/accordion';

export default function SummariesPanel() {
  const summaries = useSummaries();
  const toggleSummary = useCDSSStore((state) => state.toggleSummary);
  const expandedSummary = summaries.find(s => s.isExpanded);

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

  return (
    <Card className="flex flex-col overflow-hidden h-full">
      {/* Panel Header */}
      <CardHeader className="bg-gradient-to-r from-zinc-800 to-zinc-900 border-b border-border py-3 px-5">
        <div className="flex justify-between items-center">
          <CardTitle className="text-lg font-semibold">EXAID Summaries</CardTitle>
          <Badge variant="secondary" className="text-sm">
            {summaries.length} summar{summaries.length !== 1 ? 'ies' : 'y'}
          </Badge>
        </div>
      </CardHeader>

      {/* Panel Content */}
      <CardContent className="flex-1 overflow-y-auto p-4">
        {summaries.length === 0 ? (
          <p className="text-muted-foreground text-center py-8 text-sm">
            No summaries yet. Summaries will appear here as EXAID processes agent traces.
          </p>
        ) : (
          <Accordion
            type="single"
            collapsible
            value={expandedSummary?.id}
            onValueChange={handleValueChange}
            className="space-y-3"
          >
            {summaries.map((summary) => (
              <SummaryCard key={summary.id} summary={summary} />
            ))}
          </Accordion>
        )}
      </CardContent>
    </Card>
  );
}

