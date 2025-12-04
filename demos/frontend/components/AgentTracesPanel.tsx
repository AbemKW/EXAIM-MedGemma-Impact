'use client';

import { useRef, useEffect, useMemo } from 'react';
import { useAllAgents } from '@/store/cdssStore';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import ConsoleLogEntry from './ConsoleLogEntry';

// Generate consistent color for an agent name using hash
function getAgentColor(agentName: string): string {
  let hash = 0;
  for (let i = 0; i < agentName.length; i++) {
    hash = agentName.charCodeAt(i) + ((hash << 5) - hash);
  }
  
  // Generate a color in the blue-green-purple range (console-friendly)
  const hue = (hash % 180) + 180; // 180-360 range (blue to purple)
  const saturation = 60 + (hash % 20); // 60-80%
  const lightness = 65 + (hash % 15); // 65-80%
  
  return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
}

export default function AgentTracesPanel() {
  const agents = useAllAgents();
  const consoleRef = useRef<HTMLDivElement>(null);
  const scrollPositionRef = useRef({ wasAtBottom: true, hadScrollableContent: false });

  // Create a map of agent colors for consistency
  const agentColors = useMemo(() => {
    const colorMap = new Map<string, string>();
    agents.forEach(agent => {
      if (!colorMap.has(agent.agentName)) {
        colorMap.set(agent.agentName, getAgentColor(agent.agentName));
      }
    });
    return colorMap;
  }, [agents]);

  // Track scroll position before updates
  useEffect(() => {
    if (!consoleRef.current) return;

    const container = consoleRef.current;
    const scrollHeight = container.scrollHeight;
    const scrollTop = container.scrollTop;
    const clientHeight = container.clientHeight;

    const wasAtBottom = scrollHeight - scrollTop - clientHeight < 50;
    const hadScrollableContent = scrollHeight > clientHeight;

    scrollPositionRef.current = { wasAtBottom, hadScrollableContent };
  }, [agents]);

  // Auto-scroll logic
  useEffect(() => {
    if (!consoleRef.current) return;

    const { wasAtBottom, hadScrollableContent } = scrollPositionRef.current;
    const shouldAutoScroll = !hadScrollableContent || wasAtBottom;

    if (shouldAutoScroll) {
      requestAnimationFrame(() => {
        if (consoleRef.current) {
          consoleRef.current.scrollTop = consoleRef.current.scrollHeight;
        }
      });
    }
  }, [agents]);

  // Count unique agents
  const uniqueAgentCount = useMemo(() => {
    return new Set(agents.map(a => a.agentName)).size;
  }, [agents]);

  return (
    <Card className="flex flex-col overflow-hidden h-full console-panel">
      {/* Panel Header */}
      <CardHeader className="bg-gradient-to-r from-blue-950/40 to-blue-950/30 border-b border-blue-900/20 py-3 px-5">
        <div className="flex justify-between items-center">
          <CardTitle className="text-xl font-bold">Raw Agent Traces</CardTitle>
          <Badge variant="secondary" className="text-sm">
            {uniqueAgentCount} agent{uniqueAgentCount !== 1 ? 's' : ''}
          </Badge>
        </div>
      </CardHeader>

      {/* Console Content */}
      <CardContent className="flex-1 overflow-hidden p-0">
        <div
          ref={consoleRef}
          className="console-container h-full overflow-y-auto"
        >
          {agents.length === 0 ? (
            <div className="console-empty p-8 text-center">
              <p className="text-base" style={{ color: 'var(--console-text)', opacity: 0.6 }}>
                No traces yet. Process a case to see verbose reasoning traces.
              </p>
            </div>
          ) : (
            <div className="console-logs">
              {agents.map((agent) => {
                const agentColor = agentColors.get(agent.agentName) || '#60a5fa';
                return (
                  <ConsoleLogEntry
                    key={agent.id}
                    agentName={agent.agentName}
                    content={agent.fullText}
                    agentColor={agentColor}
                  />
                );
              })}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

