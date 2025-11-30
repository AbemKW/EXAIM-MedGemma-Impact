import { useCDSSStore } from '@/store/cdssStore';
import type { WebSocketMessage, SummaryData } from '@/lib/types';

// WebSocket URL validation - must be ws:// or wss://
function validateWebSocketUrl(url: string | undefined): string {
  if (!url) {
    throw new Error(
      'WebSocket URL not configured. Set NEXT_PUBLIC_WS_URL environment variable.'
    );
  }
  
  if (!url.startsWith('ws://') && !url.startsWith('wss://')) {
    throw new Error(
      `Invalid WebSocket URL: "${url}". URL must start with ws:// or wss://`
    );
  }
  
  // Validate URL format
  try {
    new URL(url);
  } catch {
    throw new Error(
      `Invalid WebSocket URL format: "${url}". Please provide a valid URL.`
    );
  }
  
  return url;
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
  private wsUrl: string;

  constructor() {
    // Validate URL on construction - fail fast with clear error message
    this.wsUrl = validateWebSocketUrl(process.env.NEXT_PUBLIC_WS_URL);
  }

  /**
   * Connect to the WebSocket server.
   * Handles automatic reconnection on disconnect.
   */
  connect(): void {
    // Prevent duplicate connections
    if (this.ws?.readyState === WebSocket.CONNECTING || 
        this.ws?.readyState === WebSocket.OPEN) {
      return;
    }

    this.intentionalClose = false;
    const store = useCDSSStore.getState();
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
        // Flush any remaining tokens
        store.flushTokens();
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
