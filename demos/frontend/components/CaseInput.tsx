'use client';

import { useState, FormEvent, useRef, useEffect } from 'react';
import { useIsProcessing } from '@/store/cdssStore';
import type { CaseRequest } from '@/lib/types';

export default function CaseInput() {
  const [caseText, setCaseText] = useState('');
  const isProcessing = useIsProcessing();
  const textareaRef = useRef<HTMLTextAreaElement>(null);

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
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
      <form onSubmit={handleSubmit} className="flex gap-3">
        <div className="flex-1">
          <textarea
            ref={textareaRef}
            value={caseText}
            onChange={(e) => setCaseText(e.target.value)}
            placeholder="Enter patient case description..."
            rows={1}
            disabled={isProcessing}
            className="w-full px-4 py-3 border border-gray-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all disabled:bg-gray-50 disabled:cursor-not-allowed"
            style={{ maxHeight: '200px' }}
          />
        </div>
        <button
          type="submit"
          disabled={isProcessing}
          className="px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center justify-center min-w-[60px]"
        >
          <span className="text-xl">{isProcessing ? '⏳' : '→'}</span>
        </button>
      </form>
    </div>
  );
}
