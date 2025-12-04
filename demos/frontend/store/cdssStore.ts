import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import type { AgentTrace, Summary, ConnectionStatus, ModalState, SummaryData } from '@/lib/types';

// Token batching interval in milliseconds - controls how frequently buffered tokens
// are flushed to the UI. Lower values mean more responsive updates but higher CPU usage.
const TOKEN_BATCH_INTERVAL_MS = 50;

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
  
  // Modal state
  modal: ModalState;
  
  // Actions
  startNewAgent: (agentId: string) => void;
  addToken: (agentId: string, token: string) => void;
  flushTokens: () => void;
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

// Token buffer outside Zustand state to prevent triggering updates
// Still a Map structure, now keyed by cardId instead of agentId
const tokenBuffer = new Map<string, string[]>();
let flushInterval: NodeJS.Timeout | null = null;

export const useCDSSStore = create<CDSSState>()(
  subscribeWithSelector((set, get) => ({
    // Initial state
    agents: [],  // Changed from Map to Array
    summaries: [],
    summaryIdCounter: 0,
    wsStatus: 'disconnected',
    reconnectAttempts: 0,
    isProcessing: false,
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
        
        // Add new card at the beginning (most recent first)
        return {
          agents: [newCard, ...state.agents],
        };
      });
    },
    
    // Add token to buffer for the most recent card of this agent
    addToken: (agentId: string, token: string) => {
      const state = get();
      
      // Find the most recent card for this agent (first match in array)
      const cardIndex = state.agents.findIndex(a => a.agentName === agentId);
      
      if (cardIndex === -1) {
        // No card exists for this agent. This may indicate a missing agent_started message or a race condition.
        console.warn(`No active card found for agent '${agentId}'. This may indicate a missing agent_started message or a race condition. Auto-creating card to prevent token loss.`);
        get().startNewAgent(agentId);
        // Try to find the card again after creating it
        const updatedState = get();
        const updatedCardIndex = updatedState.agents.findIndex(a => a.agentName === agentId);
        if (updatedCardIndex === -1) {
          // Still not found, abort
          console.error(`Failed to auto-create card for agent '${agentId}'. Token will be lost.`);
          return;
        }
        // Use the newly created card
        const card = updatedState.agents[updatedCardIndex];
        if (!tokenBuffer.has(card.id)) {
          tokenBuffer.set(card.id, []);
        }
        tokenBuffer.get(card.id)!.push(token);
        // Start flush interval if not already running
        if (!flushInterval) {
          flushInterval = setInterval(() => {
            const store = get();
            // Only flush if there are tokens in the buffer
            if (tokenBuffer.size > 0) {
              store.flushTokens();
            }
          }, TOKEN_BATCH_INTERVAL_MS);
        }
        return;
      }
      
      const card = state.agents[cardIndex];
      
      // Add token to external buffer using card ID
      if (!tokenBuffer.has(card.id)) {
        tokenBuffer.set(card.id, []);
      }
      tokenBuffer.get(card.id)!.push(token);
      
      // Start flush interval if not already running
      if (!flushInterval) {
        flushInterval = setInterval(() => {
          const store = get();
          // Only flush if there are tokens in the buffer
          if (tokenBuffer.size > 0) {
            store.flushTokens();
          }
        }, TOKEN_BATCH_INTERVAL_MS);
      }
    },
    
    // Flush all buffered tokens to agents
    flushTokens: () => {
      // Check if there are any tokens to flush
      const hasTokens = tokenBuffer.size > 0 && 
        Array.from(tokenBuffer.values()).some(arr => arr.length > 0);
      
      if (!hasTokens) {
        // No tokens to flush - clear interval
        if (flushInterval) {
          clearInterval(flushInterval);
          flushInterval = null;
        }
        return;
      }
      
      set((state) => {
        const newAgents = [...state.agents];
        let hasUpdates = false;
        
        // Update all cards with buffered tokens
        tokenBuffer.forEach((tokens, cardId) => {
          if (tokens.length > 0) {
            const cardIndex = newAgents.findIndex(a => a.id === cardId);
            if (cardIndex !== -1) {
              newAgents[cardIndex] = {
                ...newAgents[cardIndex],
                fullText: newAgents[cardIndex].fullText + tokens.join(''),
                lastUpdate: new Date(),
              };
              hasUpdates = true;
            }
          }
        });
        
        // Clear the external buffer
        tokenBuffer.clear();
        
        // If no updates were made, just return current state
        if (!hasUpdates) {
          return state;
        }
        
        return { agents: newAgents };
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
      // Clear external flush timer and buffer
      if (flushInterval) {
        clearInterval(flushInterval);
        flushInterval = null;
      }
      tokenBuffer.clear();
      
      set({
        agents: [],  // Clear array
        summaries: [],
        summaryIdCounter: 0,
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
