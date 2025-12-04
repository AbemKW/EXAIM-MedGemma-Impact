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

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      const newHeight = Math.min(textareaRef.current.scrollHeight, 120);
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
    <div className="bg-card rounded-lg shadow-sm border border-border p-4 flex-shrink-0">
      <form onSubmit={handleSubmit} className="flex gap-3">
        <div className="flex-1">
          <Textarea
            ref={textareaRef}
            value={caseText}
            onChange={(e) => setCaseText(e.target.value)}
            placeholder="Enter patient case description..."
            rows={1}
            disabled={isProcessing}
            className="resize-none text-base"
            style={{ maxHeight: '120px' }}
          />
        </div>
        <Button
          type="submit"
          disabled={isProcessing}
          className="min-w-[60px]"
        >
          <span className="text-xl">{isProcessing ? '⏳' : '→'}</span>
        </Button>
      </form>
    </div>
  );
}

