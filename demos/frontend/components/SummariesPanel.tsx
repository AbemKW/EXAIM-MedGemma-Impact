'use client';

import { useSummaries } from '@/store/cdssStore';
import SummaryCard from './SummaryCard';

export default function SummariesPanel() {
  const summaries = useSummaries();

  return (
    <div className="flex flex-col bg-white rounded-lg shadow-md border border-gray-200 overflow-hidden h-full">
      {/* Panel Header */}
      <div className="px-6 py-4 bg-gradient-to-r from-teal-50 to-teal-100 border-b border-gray-200">
        <div className="flex justify-between items-center">
          <h2 className="text-xl font-bold text-gray-800">EXAID Summaries</h2>
          <span className="px-3 py-1 bg-teal-600 text-white text-sm font-semibold rounded-full">
            {summaries.length} summar{summaries.length !== 1 ? 'ies' : 'y'}
          </span>
        </div>
      </div>

      {/* Panel Content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-2">
        {summaries.length === 0 ? (
          <p className="text-gray-500 text-center py-8">
            No summaries yet. Summaries will appear here as EXAID processes agent traces.
          </p>
        ) : (
          summaries.map((summary) => (
            <SummaryCard key={summary.id} summary={summary} />
          ))
        )}
      </div>
    </div>
  );
}
