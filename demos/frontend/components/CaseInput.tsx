'use client';

import { useState, FormEvent, useRef, useEffect } from 'react';
import { useIsProcessing } from '@/store/cdssStore';
import type { CaseRequest } from '@/lib/types';
import { Textarea } from '@/components/ui/textarea';
import { Button } from '@/components/ui/button';

export default function CaseInput() {
  const [caseText, setCaseText] = useState('');
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [lastSentMessage, setLastSentMessage] = useState('');
  const isProcessing = useIsProcessing();
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const formRef = useRef<HTMLFormElement>(null);

  // Auto-resize textarea (only when expanded)
  useEffect(() => {
    if (textareaRef.current) {
      if (isCollapsed) {
        // Reset height when collapsed
        textareaRef.current.style.height = 'auto';
      } else {
        // Auto-resize when expanded (reduced max height)
        textareaRef.current.style.height = 'auto';
        const newHeight = Math.min(textareaRef.current.scrollHeight, 100);
        textareaRef.current.style.height = `${newHeight}px`;
      }
    }
  }, [caseText, isCollapsed]);

  // Handle expanding from collapsed state
  const handleExpand = () => {
    if (isCollapsed) {
      setIsCollapsed(false);
      setCaseText(lastSentMessage);
      // Focus textarea after expansion
      setTimeout(() => {
        textareaRef.current?.focus();
      }, 100);
    }
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();

    const trimmedCase = caseText.trim();
    if (!trimmedCase) {
      alert('Please enter a patient case');
      return;
    }

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/process-case`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ case: trimmedCase } as CaseRequest),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to process case');
      }

      const result = await response.json();
      console.log('Case processed:', result);

      // Save message and collapse input after successful submission
      setLastSentMessage(trimmedCase);
      setCaseText('');
      setIsCollapsed(true);
    } catch (error) {
      console.error('Error processing case:', error);
      alert(`Error: ${(error as Error).message}`);
    }
  };

  // Truncate message for collapsed display
  const truncateMessage = (text: string, maxLength: number = 80) => {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
  };

  return (
    <div 
      className="w-full transition-all duration-300 overflow-hidden flex-shrink-0"
      style={isCollapsed ? { height: '48px', minHeight: '48px', maxHeight: '48px' } : { maxHeight: '100px' }}
    >
      <form 
        ref={formRef} 
        onSubmit={handleSubmit} 
        className={`relative flex items-end gap-2 ${isCollapsed ? 'h-full' : ''}`}
        style={isCollapsed ? { height: '100%' } : {}}
      >
        <div className={`flex-1 relative ${isCollapsed ? 'h-full' : ''}`} style={isCollapsed ? { height: '100%' } : {}}>
          {isCollapsed ? (
            // Collapsed state: single-line minimized input
            <div
              onClick={handleExpand}
              className="h-full px-4 py-3 rounded-2xl bg-muted/20 backdrop-blur-xl border border-white/10 cursor-text transition-all shadow-lg hover:bg-muted/30 hover:border-white/20 flex items-center"
            >
              <span className="text-base text-muted-foreground flex-1 truncate">
                {lastSentMessage ? truncateMessage(lastSentMessage) : 'Click to add another case...'}
              </span>
              <svg 
                className="h-4 w-4 text-muted-foreground ml-2 flex-shrink-0" 
                fill="none" 
                viewBox="0 0 24 24" 
                stroke="currentColor"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </div>
          ) : (
            // Expanded state: full input form
            <>
              <Textarea
                ref={textareaRef}
                value={caseText}
                onChange={(e) => setCaseText(e.target.value)}
                placeholder="Enter patient case description..."
                rows={1}
                disabled={isProcessing}
                className="resize-none text-base pr-14 py-3 px-4 rounded-2xl bg-muted/20 backdrop-blur-xl border border-white/10 focus:border-primary/50 focus:bg-muted/30 transition-all shadow-lg case-input-textarea glass-morphism liquid-ripple"
                style={{ maxHeight: '100px' }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey && !isProcessing && caseText.trim()) {
                    e.preventDefault();
                    formRef.current?.requestSubmit();
                  }
                }}
              />
              <button
                type="submit"
                disabled={isProcessing || !caseText.trim()}
                className="absolute right-3 bottom-3 h-10 w-10 rounded-lg bg-primary/80 backdrop-blur-md hover:bg-primary/90 text-primary-foreground disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg hover:shadow-xl flex items-center justify-center focus:outline-none focus:ring-2 focus:ring-primary/50 z-10 border border-white/20 liquid-button"
              >
                {isProcessing ? (
                  <svg className="h-5 w-5 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                ) : (
                  <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5" />
                  </svg>
                )}
              </button>
            </>
          )}
        </div>
      </form>
    </div>
  );
}

