// Type definitions for EXAID CDSS Frontend

export interface AgentTrace {
  id: string;  // Unique ID for React key
  agentName: string;  // Base agent name for display
  fullText: string;
  isExpanded: boolean;
  lastUpdate: Date;
}

export interface SummaryData {
  status_action: string;
  key_findings: string;
  differential_rationale: string;
  uncertainty_confidence: string;
  recommendation_next_step: string;
  agent_contributions: string;
}

export interface Summary {
  id: string;
  data: SummaryData;
  timestamp: Date;
  isExpanded: boolean;
}

export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

export interface ModalState {
  isOpen: boolean;
  agentId: string | null;
  content: string;
}

export interface CaseRequest {
  case: string;
}

// WebSocket message types
export interface AgentStartedMessage {
  type: 'agent_started';
  agent_id: string;
  timestamp: string;
}

export interface TokenMessage {
  type: 'token';
  agent_id: string;
  token: string;
  timestamp: string;
}

export interface SummaryMessage {
  type: 'summary';
  summary_data: SummaryData;
  timestamp: string;
}

export interface ProcessingStartedMessage {
  type: 'processing_started';
  timestamp: string;
}

export interface ProcessingCompleteMessage {
  type: 'processing_complete';
  timestamp: string;
}

export interface ErrorMessage {
  type: 'error';
  message: string;
  timestamp: string;
}

export type WebSocketMessage = 
  | AgentStartedMessage
  | TokenMessage 
  | SummaryMessage 
  | ProcessingStartedMessage 
  | ProcessingCompleteMessage 
  | ErrorMessage;

