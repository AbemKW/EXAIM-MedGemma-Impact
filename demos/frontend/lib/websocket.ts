import { useCDSSStore } from '@/store/cdssStore';
import type { WebSocketMessage, SummaryData } from '@/lib/types';

// WebSocket URL validation - must be ws:// or wss://
function validateWebSocketUrl(url: string | undefined): string {
  // Use default if not provided
  const wsUrl = url || 'ws://localhost:8000/ws';
  
  if (!wsUrl.startsWith('ws://') && !wsUrl.startsWith('wss://')) {
    throw new Error(
      `Invalid WebSocket URL: "${wsUrl}". URL must start with ws:// or wss://`
    );
  }
  
  // Validate URL format
  try {
    new URL(wsUrl);
  } catch {
    throw new Error(
      `Invalid WebSocket URL format: "${wsUrl}". Please provide a valid URL.`
    );
  }
  
  return wsUrl;
}

// Constants
const MAX_RECONNECT_ATTEMPTS = 5;
const RECONNECT_DELAY_MS = 3000;

/**
 * WebSocket service for real-time communication with the EXAID backend.
 * Implements singleton pattern to prevent duplicate connections.
 */
class WebSocketService {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private reconnectTimeout: NodeJS.Timeout | null = null;
  private intentionalClose = false;
  private wsUrl: string | null = null;
  private validationError: string | null = null;

  constructor() {
    // Attempt to validate URL on construction for early feedback
    // But don't throw - store the error for connect() to handle
    // Uses default ws://localhost:8000/ws if NEXT_PUBLIC_WS_URL is not set
    try {
      this.wsUrl = validateWebSocketUrl(process.env.NEXT_PUBLIC_WS_URL);
    } catch (error) {
      this.validationError = error instanceof Error ? error.message : String(error);
      console.error('WebSocket URL validation failed:', this.validationError);
    }
  }

  /**
   * Connect to the WebSocket server.
   * Validates the URL and handles automatic reconnection on disconnect.
   */
  connect(): void {
    const store = useCDSSStore.getState();
    
    // Check for validation errors from constructor
    if (this.validationError || !this.wsUrl) {
      console.error('Cannot connect: WebSocket URL is invalid or not configured');
      console.error(this.validationError || 'NEXT_PUBLIC_WS_URL environment variable is not set');
      store.setWsStatus('error');
      return;
    }
    
    // Prevent duplicate connections
    if (this.ws?.readyState === WebSocket.CONNECTING || 
        this.ws?.readyState === WebSocket.OPEN) {
      return;
    }

    this.intentionalClose = false;
    store.setWsStatus('connecting');

    try {
      this.ws = new WebSocket(this.wsUrl);
      this.setupEventHandlers();
    } catch (error) {
      console.error('Failed to create WebSocket:', error);
      store.setWsStatus('error');
      this.scheduleReconnect();
    }
  }

  /**
   * Disconnect from the WebSocket server.
   * Clears reconnection attempts.
   */
  disconnect(): void {
    this.intentionalClose = true;
    
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
    
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    
    const store = useCDSSStore.getState();
    store.setWsStatus('disconnected');
    store.setReconnectAttempts(0);
    this.reconnectAttempts = 0;
  }

  /**
   * Set up WebSocket event handlers.
   */
  private setupEventHandlers(): void {
    if (!this.ws) return;

    this.ws.onopen = () => {
      console.log('WebSocket connected');
      const store = useCDSSStore.getState();
      store.setWsStatus('connected');
      store.setReconnectAttempts(0);
      this.reconnectAttempts = 0;
    };

    this.ws.onclose = (event) => {
      console.log('WebSocket closed:', event.code, event.reason);
      const store = useCDSSStore.getState();
      store.setWsStatus('disconnected');
      
      if (!this.intentionalClose) {
        this.scheduleReconnect();
      }
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      const store = useCDSSStore.getState();
      store.setWsStatus('error');
    };

    this.ws.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);
        this.handleMessage(message);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };
  }

  /**
   * Handle incoming WebSocket messages.
   */
  private handleMessage(message: WebSocketMessage): void {
    const store = useCDSSStore.getState();

    switch (message.type) {
      case 'agent_started':
        // Create a new card for this agent invocation
        store.startNewAgent(message.agent_id);
        break;

      case 'token':
        store.addToken(message.agent_id, message.token);
        break;

      case 'summary':
        store.addSummary(
          message.summary_data as SummaryData,
          new Date(message.timestamp)
        );
        break;

      case 'processing_started':
        store.resetState();
        store.setProcessing(true);
        break;

      case 'processing_complete':
        store.setProcessing(false);
        break;

      case 'error':
        console.error('Server error:', message.message);
        store.setProcessing(false);
        break;

      default:
        console.warn('Unknown message type:', message);
    }
  }

  /**
   * Schedule a reconnection attempt with fixed delay.
   */
  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
      console.error('Max reconnection attempts reached');
      return;
    }

    this.reconnectAttempts++;
    const store = useCDSSStore.getState();
    store.setReconnectAttempts(this.reconnectAttempts);

    console.log(
      `Scheduling reconnect attempt ${this.reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS} in ${RECONNECT_DELAY_MS}ms`
    );

    this.reconnectTimeout = setTimeout(() => {
      this.connect();
    }, RECONNECT_DELAY_MS);
  }

  /**
   * Get current connection state.
   */
  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}

// Singleton instance
let wsService: WebSocketService | null = null;

/**
 * Get the WebSocket service singleton instance.
 * Creates the instance on first call.
 */
export function getWebSocketService(): WebSocketService {
  if (!wsService) {
    wsService = new WebSocketService();
  }
  return wsService;
}

