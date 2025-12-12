'use client';

import React from 'react';

interface CompressionStatsProps {
  totalWords: number;
  summaryWords: number;
}

export default function CompressionStats({ totalWords, summaryWords }: CompressionStatsProps) {
  if (totalWords === 0 || summaryWords === 0) {
    return null;
  }

  // Calculate compression percentage
  // Compression = (original - compressed) / original * 100
  // Formula: (totalWords - summaryWords) / totalWords * 100
  // 
  // IMPORTANT: Summaries have a minimum size (~135-150 words) due to 6 structured fields,
  // even when created from small buffers. When summaries are created frequently from
  // small buffers (20-30 words), cumulative summary words can exceed cumulative trace words.
  // This is expected behavior - summaries add structure and context beyond raw compression.
  const compressionValue = totalWords > 0 
    ? ((totalWords - summaryWords) / totalWords) * 100
    : 0;
  
  const compressionPercentage = Math.abs(compressionValue).toFixed(1);
  const isCompression = compressionValue > 0; // Only show compression if summaries are actually smaller

  // Only display compression stats when actual compression has occurred
  // When summaries exceed traces (due to structured format overhead), hide the component
  // to avoid confusing users with explanatory notes
  if (!isCompression) {
    return null;
  }

  return (
    <div className="compression-stats">
      <div className="compression-stats-content">
        <span className="compression-label">EXAID reduced</span>
        <span className="compression-number">{totalWords.toLocaleString()}</span>
        <span className="compression-arrow">→</span>
        <span className="compression-number">{summaryWords.toLocaleString()}</span>
        <span className="compression-label">words</span>
        <span className="compression-percentage">
          ({compressionPercentage}% compression)
        </span>
      </div>
    </div>
  );
}

