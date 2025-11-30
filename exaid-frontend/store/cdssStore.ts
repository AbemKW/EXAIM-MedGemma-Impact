import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import type { AgentTrace, Summary, ConnectionStatus, ModalState, SummaryData } from '@/lib/types';

interface CDSSState {
  // Agent traces
  agents: Map<string, AgentTrace>;
  
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
  addToken: (agentId: string, token: string) => void;
  flushTokens: () => void;
  addSummary: (data: SummaryData, timestamp: Date) => void;
  toggleAgent: (agentId: string) => void;
  toggleSummary: (id: string) => void;
  setWsStatus: (status: ConnectionStatus) => void;
  setReconnectAttempts: (attempts: number) => void;
  setProcessing: (isProcessing: boolean) => void;
  openModal: (agentId: string, content: string) => void;
  closeModal: () => void;
  resetState: () => void;
}

// Token buffer outside Zustand state to prevent triggering updates
const tokenBuffer = new Map<string, string[]>();
let flushInterval: NodeJS.Timeout | null = null;

export const useCDSSStore = create<CDSSState>()(
  subscribeWithSelector((set, get) => ({
    // Initial state
    agents: new Map(),
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
    
    // Add token to buffer (doesn't trigger state update)
    addToken: (agentId: string, token: string) => {
      const state = get();
      
      // Initialize agent if doesn't exist
      if (!state.agents.has(agentId)) {
        set((state) => ({
          agents: new Map(state.agents).set(agentId, {
            fullText: '',
            isExpanded: false,
            lastUpdate: new Date(),
          }),
        }));
      }
      
      // Add token to external buffer
      if (!tokenBuffer.has(agentId)) {
        tokenBuffer.set(agentId, []);
      }
      tokenBuffer.get(agentId)!.push(token);
      
      // Start flush interval if not already running (50ms batching)
      if (!flushInterval) {
        flushInterval = setInterval(() => {
          const store = get();
          // Only flush if there are tokens in the buffer
          if (tokenBuffer.size > 0) {
            store.flushTokens();
          }
        }, 50);
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
        const newAgents = new Map(state.agents);
        const agentIdsToUpdate: string[] = [];
        
        // Update all agents with buffered tokens
        tokenBuffer.forEach((tokens, agentId) => {
          if (tokens.length > 0) {
            const agent = newAgents.get(agentId);
            if (agent) {
              newAgents.set(agentId, {
                ...agent,
                fullText: agent.fullText + tokens.join(''),
                lastUpdate: new Date(),
              });
              agentIdsToUpdate.push(agentId);
            }
          }
        });
        
        // Clear the external buffer
        tokenBuffer.clear();
        
        // If no updates were made, just return current state
        if (agentIdsToUpdate.length === 0) {
          return state;
        }
        
        // Reorder agents - move most recently updated to top
        const sortedAgents = new Map();
        // Add recently updated agents first
        agentIdsToUpdate.forEach(id => {
          const agent = newAgents.get(id);
          if (agent) {
            sortedAgents.set(id, agent);
          }
        });
        // Add remaining agents
        newAgents.forEach((agent, id) => {
          if (!agentIdsToUpdate.includes(id)) {
            sortedAgents.set(id, agent);
          }
        });
        
        return { agents: sortedAgents };
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
    toggleAgent: (agentId: string) => {
      set((state) => {
        const newAgents = new Map(state.agents);
        const agent = newAgents.get(agentId);
        if (agent) {
          newAgents.set(agentId, {
            ...agent,
            isExpanded: !agent.isExpanded,
          });
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
    openModal: (agentId: string, content: string) => {
      set({
        modal: {
          isOpen: true,
          agentId,
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
        agents: new Map(),
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
export const useAgentTrace = (agentId: string) => {
  return useCDSSStore((state) => state.agents.get(agentId));
};

export const useAgentIds = () => {
  return useCDSSStore((state) => Array.from(state.agents.keys()));
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
