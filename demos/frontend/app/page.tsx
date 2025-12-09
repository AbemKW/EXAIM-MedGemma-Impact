'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import Header from '@/components/Header';
import CaseInput from '@/components/CaseInput';
import AgentTracesPanel from '@/components/AgentTracesPanel';
import SummariesPanel from '@/components/SummariesPanel';
import AgentModal from '@/components/AgentModal';
import { getWebSocketService } from '@/lib/websocket';

export default function Home() {
  const [currentPage, setCurrentPage] = useState(0);
  const [isTransitioning, setIsTransitioning] = useState(false);
  const containerRef = useRef<HTMLMainElement>(null);
  const wheelTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const lastWheelTimeRef = useRef<number>(0);

  // Initialize WebSocket connection on mount
  useEffect(() => {
    const ws = getWebSocketService();
    ws.connect();

    // Cleanup on unmount
    return () => {
      ws.disconnect();
      if (wheelTimeoutRef.current) {
        clearTimeout(wheelTimeoutRef.current);
      }
    };
  }, []);

  // Handle wheel events for carousel navigation
  const handleWheel = useCallback((e: WheelEvent) => {
    if (isTransitioning) {
      e.preventDefault();
      return;
    }

    const now = Date.now();
    const timeSinceLastWheel = now - lastWheelTimeRef.current;
    
    // Throttle wheel events
    if (timeSinceLastWheel < 100) {
      e.preventDefault();
      return;
    }

    lastWheelTimeRef.current = now;

    const delta = e.deltaY;
    const threshold = 30; // Minimum scroll delta to trigger page change

    if (Math.abs(delta) > threshold) {
      e.preventDefault();
      setIsTransitioning(true);

      let nextPage = currentPage;
      if (delta > 0 && currentPage < 1) {
        nextPage = currentPage + 1;
      } else if (delta < 0 && currentPage > 0) {
        nextPage = currentPage - 1;
      }

      if (nextPage !== currentPage) {
        setCurrentPage(nextPage);
        
        // Reset transitioning state after animation completes
        if (wheelTimeoutRef.current) {
          clearTimeout(wheelTimeoutRef.current);
        }
        wheelTimeoutRef.current = setTimeout(() => {
          setIsTransitioning(false);
        }, 300);
      } else {
        setIsTransitioning(false);
      }
    }
  }, [currentPage, isTransitioning]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    container.addEventListener('wheel', handleWheel, { passive: false });

    return () => {
      container.removeEventListener('wheel', handleWheel);
    };
  }, [handleWheel]);

  return (
    <div className="relative liquid-glass-bg">
      {/* Fixed Header */}
      <Header />

      {/* Carousel Container */}
      <main 
        ref={containerRef}
        className={`carousel-container h-screen overflow-hidden ${isTransitioning ? 'transitioning' : ''}`}
      >
        {/* Section 1: Case Input */}
        <section 
          className="carousel-page h-screen flex items-center justify-center px-6"
          style={{ 
            paddingTop: 'var(--header-height)',
            transform: `translateY(${currentPage === 0 ? '0' : '-100%'})`
          }}
        >
          <div className="w-full max-w-3xl">
            <CaseInput />
          </div>
        </section>

        {/* Section 2: Reasoning Panels */}
        <section 
          className="carousel-page h-screen flex items-center px-6"
          style={{ 
            paddingTop: 'var(--header-height)',
            transform: `translateY(${currentPage === 1 ? '0' : '100%'})`
          }}
        >
          <div className="max-w-[1800px] mx-auto w-full" style={{ height: 'calc(100vh - var(--header-height))' }}>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 h-full">
              {/* Raw Agent Traces Panel */}
              <AgentTracesPanel />

              {/* EXAID Summaries Panel */}
              <SummariesPanel />
            </div>
          </div>
        </section>
      </main>

      {/* Modal */}
      <AgentModal />
    </div>
  );
}
