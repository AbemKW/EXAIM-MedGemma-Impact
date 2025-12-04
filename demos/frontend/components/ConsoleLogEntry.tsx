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
      <span 
        className="console-agent-label"
        style={{ color: agentColor }}
      >
        [{agentName}]
      </span>
      <span className="console-content">{content}</span>
    </div>
  );
}

export default React.memo(ConsoleLogEntry);

