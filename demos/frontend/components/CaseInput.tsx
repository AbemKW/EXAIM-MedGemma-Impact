'use client';

import { useState, FormEvent, useRef, useEffect } from 'react';
import { useIsProcessing } from '@/store/cdssStore';
import type { CaseRequest } from '@/lib/types';
import { Textarea } from '@/components/ui/textarea';
import { Button } from '@/components/ui/button';

export default function CaseInput() {
  const [caseText, setCaseText] = useState('');
  const isProcessing = useIsProcessing();
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const formRef = useRef<HTMLFormElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      const newHeight = Math.min(textareaRef.current.scrollHeight, 200);
      textareaRef.current.style.height = `${newHeight}px`;
    }
  }, [caseText]);

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

      // Clear input after successful submission
      setCaseText('');
    } catch (error) {
      console.error('Error processing case:', error);
      alert(`Error: ${(error as Error).message}`);
    }
  };

  return (
    <div className="flex-shrink-0 w-full">
      <form ref={formRef} onSubmit={handleSubmit} className="relative flex items-end gap-2">
        <div className="flex-1 relative">
          <Textarea
            ref={textareaRef}
            value={caseText}
            onChange={(e) => setCaseText(e.target.value)}
            placeholder="Enter patient case description..."
            rows={1}
            disabled={isProcessing}
            className="resize-none text-base pr-14 py-3 px-4 rounded-2xl bg-muted/50 border border-border/50 focus:border-primary/50 focus:bg-background transition-all shadow-sm case-input-textarea"
            style={{ maxHeight: '200px' }}
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
            className="absolute right-3 bottom-3 h-9 w-9 rounded-full bg-muted/70 hover:bg-muted/90 text-muted-foreground hover:text-foreground disabled:opacity-40 disabled:cursor-not-allowed transition-all shadow-sm border border-border/40 flex items-center justify-center focus:outline-none focus:ring-2 focus:ring-ring/50 z-10"
          >
            {isProcessing ? (
              <svg className="h-5 w-5 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
            ) : (
              <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
              </svg>
            )}
          </button>
        </div>
      </form>
    </div>
  );
}

