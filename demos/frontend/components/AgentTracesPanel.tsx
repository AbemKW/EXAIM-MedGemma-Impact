'use client';

import { useRef, useEffect, useMemo, useCallback, useState } from 'react';
import { useAllAgents, useTotalWords } from '@/store/cdssStore';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
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
  const totalWords = useTotalWords();
  const consoleRef = useRef<HTMLDivElement>(null);
  const shouldAutoScrollRef = useRef(true);
  const [showGoToLatest, setShowGoToLatest] = useState(false);
  const userScrollTimeoutRef = useRef<NodeJS.Timeout | null>(null);

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

  // Check if user is at the bottom of the scroll container
  const isAtBottom = useCallback((element: HTMLDivElement, threshold = 150): boolean => {
    const { scrollTop, scrollHeight, clientHeight } = element;
    const distanceFromBottom = scrollHeight - scrollTop - clientHeight;
    return distanceFromBottom <= threshold;
  }, []);

  // Scroll to bottom function - more aggressive approach
  const scrollToBottom = useCallback(() => {
    if (!consoleRef.current) return;
    const container = consoleRef.current;
    
    // Direct scroll assignment
    container.scrollTop = container.scrollHeight;
    
    // Also try smooth scroll as fallback
    container.scrollTo({
      top: container.scrollHeight,
      behavior: 'auto'
    });
  }, []);

  // Handle "Go to Latest" button click
  const handleGoToLatest = useCallback(() => {
    shouldAutoScrollRef.current = true;
    setShowGoToLatest(false);
    scrollToBottom();
  }, [scrollToBottom]);

  // Detect user scroll and disable auto-scroll when user scrolls away
  useEffect(() => {
    const container = consoleRef.current;
    if (!container) return;

    let isUserScrolling = false;
    let scrollCheckTimeout: NodeJS.Timeout | null = null;

    const handleScroll = () => {
      if (!container) return;
      
      // Clear any pending checks
      if (scrollCheckTimeout) {
        clearTimeout(scrollCheckTimeout);
        scrollCheckTimeout = null;
      }
      
      // Check if user is at bottom
      const atBottom = isAtBottom(container);
      
      // If user scrolls away from bottom, disable auto-scroll
      if (!atBottom) {
        isUserScrolling = true;
        shouldAutoScrollRef.current = false;
        setShowGoToLatest(true);
      } else {
        // User scrolled back to bottom - re-enable auto-scroll after a delay
        scrollCheckTimeout = setTimeout(() => {
          if (container && isAtBottom(container)) {
            shouldAutoScrollRef.current = true;
            setShowGoToLatest(false);
            isUserScrolling = false;
          }
        }, 300);
      }
    };

    container.addEventListener('scroll', handleScroll, { passive: true });
    
    return () => {
      container.removeEventListener('scroll', handleScroll);
      if (scrollCheckTimeout) {
        clearTimeout(scrollCheckTimeout);
      }
    };
  }, [isAtBottom]);

  // Track content changes to trigger auto-scroll
  const agentsContentKey = useMemo(() => {
    // Create a key that changes when any agent's content changes
    return agents.map(a => `${a.id}:${a.fullText.length}`).join('|');
  }, [agents]);

  // Use MutationObserver to detect DOM changes and auto-scroll
  useEffect(() => {
    const container = consoleRef.current;
    if (!container) return;

    let rafId: number | null = null;
    let timeoutId: NodeJS.Timeout | null = null;
    
    const observer = new MutationObserver(() => {
      // Clear any pending scrolls
      if (rafId) cancelAnimationFrame(rafId);
      if (timeoutId) clearTimeout(timeoutId);
      
      // Only scroll to bottom if auto-scroll is enabled
      if (shouldAutoScrollRef.current) {
        rafId = requestAnimationFrame(() => {
          requestAnimationFrame(() => {
            scrollToBottom();
            rafId = null;
          });
        });
        
        // Also use timeout as backup
        timeoutId = setTimeout(() => {
          scrollToBottom();
          timeoutId = null;
        }, 50);
      }
    });

    observer.observe(container, {
      childList: true,
      subtree: true,
      characterData: true,
    });

    return () => {
      observer.disconnect();
      if (rafId) cancelAnimationFrame(rafId);
      if (timeoutId) clearTimeout(timeoutId);
    };
  }, [scrollToBottom]);

  // Trigger scroll when content actually changes - only if auto-scroll is enabled
  useEffect(() => {
    if (agents.length > 0 && shouldAutoScrollRef.current) {
      // Multiple timing strategies to ensure DOM has updated
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          scrollToBottom();
        });
      });
      
      const timeoutId = setTimeout(() => {
        scrollToBottom();
      }, 50);
      
      return () => clearTimeout(timeoutId);
    }
  }, [agentsContentKey, scrollToBottom]);

  // Count unique agents
  const uniqueAgentCount = useMemo(() => {
    return new Set(agents.map(a => a.agentName)).size;
  }, [agents]);

  return (
    <Card className="flex flex-col overflow-hidden h-full console-panel bg-card/30 backdrop-blur-xl border-white/10 glass-card">
      {/* Panel Header */}
      <CardHeader className="bg-gradient-to-r from-zinc-950/40 to-zinc-900/30 backdrop-blur-md border-b border-white/10 py-3 px-5 glass-header">
        <div className="flex justify-between items-center">
          <CardTitle className="text-xl font-bold">Raw Agent Traces</CardTitle>
          <div className="flex gap-2 items-center">
            <Badge variant="secondary" className="text-sm">
              {totalWords.toLocaleString()} word{totalWords !== 1 ? 's' : ''}
            </Badge>
            <Badge variant="secondary" className="text-sm">
              {uniqueAgentCount} agent{uniqueAgentCount !== 1 ? 's' : ''}
            </Badge>
          </div>
        </div>
      </CardHeader>

      {/* Console Content */}
      <CardContent className="flex-1 overflow-hidden p-0 relative">
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
        
        {/* Go to Latest Button */}
        {showGoToLatest && (
          <div className="absolute bottom-4 right-4 z-20 animate-in fade-in slide-in-from-bottom-2 duration-200">
            <Button
              onClick={handleGoToLatest}
              variant="default"
              size="sm"
              className="shadow-lg backdrop-blur-md bg-blue-600/90 hover:bg-blue-600 text-white border border-white/20 transition-all hover:scale-105"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M12 5v14" />
                <path d="m19 12-7 7-7-7" />
              </svg>
              Go to Latest
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

