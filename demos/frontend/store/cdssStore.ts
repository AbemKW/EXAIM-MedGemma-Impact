import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import type { AgentTrace, Summary, ConnectionStatus, ModalState, SummaryData } from '@/lib/types';

interface CDSSState {
  // Agent traces (array for multiple invocations of same agent)
  agents: AgentTrace[];
  
  // Summaries
  summaries: Summary[];
  summaryIdCounter: number;
  
  // WebSocket state
  wsStatus: ConnectionStatus;
  reconnectAttempts: number;
  
  // Processing state
  isProcessing: boolean;
  
  // Word counting
  totalWords: number;
  
  // Modal state
  modal: ModalState;
  
  // Actions
  startNewAgent: (agentId: string) => void;
  addToken: (agentId: string, token: string) => void;
  addSummary: (data: SummaryData, timestamp: Date) => void;
  toggleAgent: (cardId: string) => void;
  toggleSummary: (id: string) => void;
  setWsStatus: (status: ConnectionStatus) => void;
  setReconnectAttempts: (attempts: number) => void;
  setProcessing: (isProcessing: boolean) => void;
  openModal: (cardId: string, content: string) => void;
  closeModal: () => void;
  resetState: () => void;
}

// Helper function to count words in a string
function countWords(text: string): number {
  if (!text || text.trim().length === 0) return 0;
  // Split by whitespace and filter out empty strings
  return text.trim().split(/\s+/).filter(word => word.length > 0).length;
}

export const useCDSSStore = create<CDSSState>()(
  subscribeWithSelector((set, get) => ({
    // Initial state
    agents: [],
    summaries: [],
    summaryIdCounter: 0,
    wsStatus: 'disconnected',
    reconnectAttempts: 0,
    isProcessing: false,
    totalWords: 0,
    modal: {
      isOpen: false,
      agentId: null,
      content: '',
    },
    
    // Start a new agent invocation (creates new card)
    startNewAgent: (agentId: string) => {
      set((state) => {
        const newCard: AgentTrace = {
          id: `${agentId}_${state.agents.length}_${Date.now()}`,  // Unique ID for React key
          agentName: agentId,              // Base name for display
          fullText: '',
          isExpanded: false,
          lastUpdate: new Date(),
        };
        
        // Add new card at the end (append to bottom)
        return {
          agents: [...state.agents, newCard],
        };
      });
    },
    
    // Add token immediately - no batching, streams token-by-token
    addToken: (agentId: string, token: string) => {
      const state = get();
      
      // Find the most recent card for this agent (first match in array)
      const cardIndex = state.agents.findIndex(a => a.agentName === agentId);
      
      if (cardIndex === -1) {
        // No card exists for this agent. This may indicate a missing agent_started message or a race condition.
        console.warn(`No active card found for agent '${agentId}'. This may indicate a missing agent_started message or a race condition. Auto-creating card to prevent token loss.`);
        get().startNewAgent(agentId);
        // After creating, retrieve the most recent card for this agent
        const updatedState = get();
        const updatedCardIndex = updatedState.agents.findIndex(a => a.agentName === agentId);
        if (updatedCardIndex === -1) {
          // Still not found, abort
          console.error(`Failed to auto-create card for agent '${agentId}'. Token will be lost.`);
          return;
        }
        // Update the newly created card immediately with the token
        set((currentState) => {
          const newAgents = [...currentState.agents];
          newAgents[updatedCardIndex] = {
            ...newAgents[updatedCardIndex],
            fullText: newAgents[updatedCardIndex].fullText + token,
            lastUpdate: new Date(),
          };
          const wordsInToken = countWords(token);
          return { 
            agents: newAgents,
            totalWords: currentState.totalWords + wordsInToken,
          };
        });
        return;
      }
      
      // Update the card immediately with the token
      set((currentState) => {
        const newAgents = [...currentState.agents];
        newAgents[cardIndex] = {
          ...newAgents[cardIndex],
          fullText: newAgents[cardIndex].fullText + token,
          lastUpdate: new Date(),
        };
        const wordsInToken = countWords(token);
        return { 
          agents: newAgents,
          totalWords: currentState.totalWords + wordsInToken,
        };
      });
    },
    
    // Add new summary
    addSummary: (data: SummaryData, timestamp: Date) => {
      set((state) => {
        const newSummary: Summary = {
          id: `summary-${state.summaryIdCounter}`,
          data,
          timestamp,
          isExpanded: true,
        };
        
        // Collapse all existing summaries
        const updatedSummaries = state.summaries.map(s => ({
          ...s,
          isExpanded: false,
        }));
        
        // Add new summary at the beginning (newest first)
        return {
          summaries: [newSummary, ...updatedSummaries],
          summaryIdCounter: state.summaryIdCounter + 1,
        };
      });
    },
    
    // Toggle agent expand/collapse
    toggleAgent: (cardId: string) => {
      set((state) => {
        const newAgents = [...state.agents];
        const cardIndex = newAgents.findIndex(a => a.id === cardId);
        if (cardIndex !== -1) {
          newAgents[cardIndex] = {
            ...newAgents[cardIndex],
            isExpanded: !newAgents[cardIndex].isExpanded,
          };
        }
        return { agents: newAgents };
      });
    },
    
    // Toggle summary expand/collapse (accordion - only one expanded)
    toggleSummary: (id: string) => {
      set((state) => {
        const updatedSummaries = state.summaries.map(summary => {
          if (summary.id === id) {
            return { ...summary, isExpanded: !summary.isExpanded };
          }
          // Collapse all others
          return { ...summary, isExpanded: false };
        });
        
        return { summaries: updatedSummaries };
      });
    },
    
    // Set WebSocket connection status
    setWsStatus: (status: ConnectionStatus) => {
      set({ wsStatus: status });
    },
    
    // Set reconnect attempts
    setReconnectAttempts: (attempts: number) => {
      set({ reconnectAttempts: attempts });
    },
    
    // Set processing state
    setProcessing: (isProcessing: boolean) => {
      set({ isProcessing });
    },
    
    // Open modal with agent trace
    openModal: (cardId: string, content: string) => {
      set({
        modal: {
          isOpen: true,
          agentId: cardId,  // Store cardId for modal
          content,
        },
      });
    },
    
    // Close modal
    closeModal: () => {
      set({
        modal: {
          isOpen: false,
          agentId: null,
          content: '',
        },
      });
    },
    
    // Reset state (called on processing_started)
    resetState: () => {
      set({
        agents: [],
        summaries: [],
        summaryIdCounter: 0,
        totalWords: 0,
        modal: {
          isOpen: false,
          agentId: null,
          content: '',
        },
      });
    },
  }))
);

// Selector hooks for optimized subscriptions
export const useAgentTrace = (cardId: string) => {
  return useCDSSStore((state) => state.agents.find(a => a.id === cardId));
};

export const useAllAgents = () => {
  return useCDSSStore((state) => state.agents);
};

export const useSummaries = () => {
  return useCDSSStore((state) => state.summaries);
};

export const useWsStatus = () => {
  return useCDSSStore((state) => state.wsStatus);
};

export const useIsProcessing = () => {
  return useCDSSStore((state) => state.isProcessing);
};

export const useModal = () => {
  return useCDSSStore((state) => state.modal);
};

export const useTotalWords = () => {
  return useCDSSStore((state) => state.totalWords);
};

