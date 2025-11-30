import sys
import asyncio
import logging
from pathlib import Path
from typing import List
from datetime import datetime
from contextlib import asynccontextmanager

# Add project root to path
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from cdss_demo.cdss import CDSS
from cdss_demo.message_bus import message_queue


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    # Startup: Start background message broadcaster
    broadcaster_task = asyncio.create_task(message_broadcaster())
    yield
    # Shutdown: Cancel background task
    broadcaster_task.cancel()
    try:
        await broadcaster_task
    except asyncio.CancelledError:
        pass  # Expected during shutdown - task was cancelled gracefully


# Configure logging
logger = logging.getLogger(__name__)


app = FastAPI(lifespan=lifespan)

# Add CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js dev server default
        "http://localhost:3001",  # Next.js dev server alternate
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active WebSocket connections
active_connections: List[WebSocket] = []
# Lock for synchronizing access to active_connections
connections_lock = asyncio.Lock()


async def get_active_connections() -> List[WebSocket]:
    """Get a copy of the active connections list safely."""
    async with connections_lock:
        return active_connections.copy()


async def add_connection(websocket: WebSocket):
    """Add a connection to the active connections list safely."""
    async with connections_lock:
        active_connections.append(websocket)


async def remove_connection(websocket: WebSocket):
    """Remove a connection from the active connections list safely."""
    async with connections_lock:
        if websocket in active_connections:
            active_connections.remove(websocket)

# Global CDSS instance for the current request.
# Note: This is intentionally request-scoped - each POST to /api/process-case creates
# a new instance. Access is synchronized using cdss_lock to prevent race conditions
# when multiple requests arrive simultaneously.
cdss_instance: CDSS = None
cdss_lock = asyncio.Lock()

# Maximum allowed length for case text to prevent resource abuse
MAX_CASE_LENGTH = 100000


class CaseRequest(BaseModel):
    case: str
    
    @field_validator('case')
    @classmethod
    def validate_case(cls, v: str) -> str:
        """Validate case text is not empty and within size limits."""
        if not v or not v.strip():
            raise ValueError("Case text cannot be empty")
        if len(v) > MAX_CASE_LENGTH:
            raise ValueError(f"Case text exceeds maximum length of {MAX_CASE_LENGTH} characters")
        return v.strip()


async def send_token_direct(agent_id: str, token: str):
    """Send token directly to WebSocket clients without queuing.
    
    This bypasses the message queue for immediate token-by-token delivery.
    """
    connections = await get_active_connections()
    if not connections:
        return
    
    message = {
        "type": "token",
        "agent_id": agent_id,
        "token": token,
        "timestamp": datetime.now().isoformat()
    }
    
    disconnected = []
    for connection in connections:
        try:
            await connection.send_json(message)
        except Exception as e:
            logger.warning(f"Error sending token to client: {e}")
            disconnected.append(connection)
    
    # Remove disconnected clients
    for conn in disconnected:
        await remove_connection(conn)


async def send_summary_direct(summary):
    """Send summary directly to WebSocket clients.
    
    Args:
        summary: AgentSummary Pydantic object to broadcast
    """
    connections = await get_active_connections()
    if not connections:
        return
    
    # Convert Pydantic model to dict
    summary_dict = {
        "status_action": summary.status_action,
        "key_findings": summary.key_findings,
        "differential_rationale": summary.differential_rationale,
        "uncertainty_confidence": summary.uncertainty_confidence,
        "recommendation_next_step": summary.recommendation_next_step,
        "agent_contributions": summary.agent_contributions
    }
    
    message = {
        "type": "summary",
        "summary_data": summary_dict,
        "timestamp": datetime.now().isoformat()
    }
    
    disconnected = []
    for connection in connections:
        try:
            await connection.send_json(message)
        except Exception as e:
            logger.warning(f"Error sending summary to client: {e}")
            disconnected.append(connection)
    
    # Remove disconnected clients
    for conn in disconnected:
        await remove_connection(conn)


def _queue_message_fallback(message: dict, context: str = "message"):
    """Helper function to queue a message as fallback when direct WebSocket send fails.
    
    Args:
        message: The message dict to queue
        context: Description of the message context for logging
    """
    try:
        message_queue.put_nowait(message)
    except asyncio.QueueFull:
        logger.warning(f"Message queue full for {context}, message dropped.")
    except Exception as queue_error:
        logger.error(f"Error queueing {context}: {queue_error}")


def trace_callback(agent_id: str, token: str):
    """Callback function to send tokens directly to WebSocket clients.
    
    This is called from async code (token stream), so we can safely schedule
    async operations. We send tokens directly without queuing to ensure
    token-by-token delivery.
    
    Every token from every agent that uses received_streamed_tokens() will
    trigger this callback, ensuring all tokens are captured and displayed.
    """
    # Schedule immediate async send without blocking
    # Since we're called from async context, we can safely create a task
    try:
        # Try to get the running event loop - will succeed if called from async context
        asyncio.get_running_loop()
        # Schedule the send immediately as a task
        asyncio.create_task(send_token_direct(agent_id, token))
    except RuntimeError:
        # No running event loop - unexpected but log warning and fallback to queue
        logger.warning(f"No running event loop in trace_callback for agent {agent_id} - using queue fallback")
        message = {
            "type": "token",
            "agent_id": agent_id,
            "token": token,
            "timestamp": datetime.now().isoformat()
        }
        _queue_message_fallback(message, f"agent {agent_id} token")
    except Exception as e:
        # Fallback to queue if direct send fails
        logger.warning(f"Error sending token directly, falling back to queue: {e}")
        message = {
            "type": "token",
            "agent_id": agent_id,
            "token": token,
            "timestamp": datetime.now().isoformat()
        }
        _queue_message_fallback(message, f"agent {agent_id} token")


def _summary_to_dict(summary) -> dict:
    """Convert a summary Pydantic model to a dict for serialization."""
    return {
        "status_action": summary.status_action,
        "key_findings": summary.key_findings,
        "differential_rationale": summary.differential_rationale,
        "uncertainty_confidence": summary.uncertainty_confidence,
        "recommendation_next_step": summary.recommendation_next_step,
        "agent_contributions": summary.agent_contributions
    }


def summary_callback(summary):
    """Callback function to send summaries directly to WebSocket clients.
    
    This is called when EXAID generates a new summary from buffered traces.
    
    Args:
        summary: AgentSummary Pydantic object
    """
    try:
        # Try to get the running event loop - will succeed if called from async context
        asyncio.get_running_loop()
        # Schedule the send immediately as a task
        asyncio.create_task(send_summary_direct(summary))
    except RuntimeError:
        # No running event loop - unexpected but log warning and fallback to queue
        logger.warning("No running event loop in summary_callback - using queue fallback")
        message = {
            "type": "summary",
            "summary_data": _summary_to_dict(summary),
            "timestamp": datetime.now().isoformat()
        }
        _queue_message_fallback(message, "summary")
    except Exception as e:
        # Fallback to queue if direct send fails
        logger.warning(f"Error sending summary directly, falling back to queue: {e}")
        message = {
            "type": "summary",
            "summary_data": _summary_to_dict(summary),
            "timestamp": datetime.now().isoformat()
        }
        _queue_message_fallback(message, "summary")


async def broadcast_message(message: dict):
    """Broadcast a message to all connected WebSocket clients."""
    connections = await get_active_connections()
    if not connections:
        return
    
    disconnected = []
    for connection in connections:
        try:
            await connection.send_json(message)
        except Exception as e:
            logger.warning(f"Error sending message to client: {e}")
            disconnected.append(connection)
    
    # Remove disconnected clients
    for conn in disconnected:
        await remove_connection(conn)


async def message_broadcaster():
    """Background task to process message queue and broadcast to clients.
    
    Processes messages aggressively to ensure smooth streaming without batching.
    """
    while True:
        try:
            # Wait for at least one message
            message = await message_queue.get()
            await broadcast_message(message)
            message_queue.task_done()
            
            # Process any additional messages immediately without waiting
            # This prevents batching and ensures smooth streaming
            while not message_queue.empty():
                try:
                    message = message_queue.get_nowait()
                    await broadcast_message(message)
                    message_queue.task_done()
                except asyncio.QueueEmpty:
                    break
                except Exception as e:
                    logger.error(f"Error processing queued message: {e}")
                    message_queue.task_done()
                    
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in message broadcaster: {e}")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time trace updates.
    
    This endpoint maintains a persistent connection for streaming agent traces
    and summaries to the client. Client messages are received to keep the
    connection alive but are not processed - all communication is server-to-client
    via broadcast_message().
    """
    await websocket.accept()
    await add_connection(websocket)
    connections = await get_active_connections()
    logger.info(f"Client connected. Total connections: {len(connections)}")
    
    try:
        # Keep connection alive - just wait for disconnect
        # Messages are sent via broadcast_message() from other parts of the app
        while True:
            # Wait for any message (or disconnect) - messages are received but not
            # processed as this is a server-push only endpoint for trace streaming
            await websocket.receive_text()
    except WebSocketDisconnect:
        await remove_connection(websocket)
        connections = await get_active_connections()
        logger.info(f"Client disconnected. Total connections: {len(connections)}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await remove_connection(websocket)


@app.post("/api/process-case")
async def process_case(request: CaseRequest):
    """Process a clinical case through the CDSS system.
    
    A new CDSS instance is created for each request. The cdss_lock ensures that only
    one request can process at a time, preventing race conditions with the global
    cdss_instance and its callbacks. Requests will be queued and processed sequentially.
    
    Input validation (empty check, length limit) is handled by Pydantic validator
    and will return 422 Unprocessable Entity for invalid input.
    """
    global cdss_instance
    
    # Acquire lock to ensure only one request processes at a time
    async with cdss_lock:
        try:
            # Create a new CDSS instance for this request
            cdss_instance = CDSS()
            
            # Register trace callback with EXAID
            cdss_instance.exaid.register_trace_callback(trace_callback)
            logger.debug(f"Trace callback registered. Callbacks: {len(cdss_instance.exaid.trace_callbacks)}")
            
            # Register summary callback with EXAID
            cdss_instance.exaid.register_summary_callback(summary_callback)
            logger.debug(f"Summary callback registered. Callbacks: {len(cdss_instance.exaid.summary_callbacks)}")
            
            # Send start message
            await broadcast_message({
                "type": "processing_started",
                "timestamp": datetime.now().isoformat()
            })
            
            # Process the case (this will stream traces via callbacks)
            # request.case is already validated and trimmed by Pydantic validator
            await cdss_instance.process_case(request.case)
            
            # Send completion message
            await broadcast_message({
                "type": "processing_complete",
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                "status": "success",
                "message": "Case processed successfully"
            }
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error processing case: {error_msg}")
            
            # Send error message to clients
            await broadcast_message({
                "type": "error",
                "message": error_msg,
                "timestamp": datetime.now().isoformat()
            })
            
            raise HTTPException(status_code=500, detail=error_msg)


if __name__ == "__main__":
    import uvicorn
    print("Starting EXAID API server...")
    print("WebSocket endpoint: ws://localhost:8000/ws")
    print("API endpoint: http://localhost:8000/api/process-case")
    uvicorn.run(app, host="0.0.0.0", port=8000)

