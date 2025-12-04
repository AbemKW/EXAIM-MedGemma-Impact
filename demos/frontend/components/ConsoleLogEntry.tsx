'use client';

import React from 'react';

interface ConsoleLogEntryProps {
  agentName: string;
  content: string;
  agentColor: string;
}

function ConsoleLogEntry({ agentName, content, agentColor }: ConsoleLogEntryProps) {
  return (
    <div className="console-log-entry">
      <div 
        className="console-agent-label-sticky"
        style={{ color: agentColor }}
      >
        [{agentName}]
      </div>
      <div className="console-content-wrapper">
        <span className="console-content">{content}</span>
      </div>
    </div>
  );
}

export default React.memo(ConsoleLogEntry);

