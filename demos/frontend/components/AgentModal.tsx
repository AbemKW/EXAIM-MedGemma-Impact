'use client';

import { useEffect } from 'react';
import { createPortal } from 'react-dom';
import FocusLock from 'react-focus-lock';
import { motion, AnimatePresence } from 'framer-motion';
import { useCDSSStore, useModal } from '@/store/cdssStore';

export default function AgentModal() {
  const modal = useModal();
  const closeModal = useCDSSStore((state) => state.closeModal);

  // ESC key handler
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && modal.isOpen) {
        closeModal();
      }
    };

    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [modal.isOpen, closeModal]);

  // Prevent body scroll when modal is open
  useEffect(() => {
    if (modal.isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }

    return () => {
      document.body.style.overflow = '';
    };
  }, [modal.isOpen]);

  if (typeof window === 'undefined') return null;

  const portalRoot = document.getElementById('modal-portal');
  if (!portalRoot) return null;

  return createPortal(
    <AnimatePresence>
      {modal.isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
          className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 backdrop-blur-sm p-4"
          onClick={(e) => {
            if (e.target === e.currentTarget) {
              closeModal();
            }
          }}
          role="dialog"
          aria-modal="true"
          aria-labelledby="modal-title"
        >
          <FocusLock>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 20 }}
              transition={{ duration: 0.2 }}
              className="bg-white rounded-lg shadow-2xl max-w-4xl w-full max-h-[80vh] flex flex-col"
            >
              {/* Modal Header */}
              <div className="flex justify-between items-center px-6 py-4 border-b border-gray-200">
                <h3
                  id="modal-title"
                  className="text-xl font-bold text-gray-800"
                >
                  {modal.agentId} - Full Output
                </h3>
                <button
                  onClick={closeModal}
                  className="text-gray-500 hover:text-gray-700 text-3xl leading-none font-light transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 rounded"
                  aria-label="Close modal"
                >
                  &times;
                </button>
              </div>

              {/* Modal Body */}
              <div className="flex-1 overflow-y-auto p-6">
                <pre className="font-mono text-sm text-gray-800 whitespace-pre-wrap break-words">
                  {modal.content}
                </pre>
              </div>
            </motion.div>
          </FocusLock>
        </motion.div>
      )}
    </AnimatePresence>,
    portalRoot
  );
}
