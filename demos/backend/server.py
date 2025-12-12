import asyncio
import logging
import subprocess
import sys
import os
import uuid
from pathlib import Path
from typing import List, Dict
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from demos.cdss_example.cdss import CDSS
from demos.cdss_example.message_bus import message_queue


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

# Track active run IDs per agent (agent_id -> run_id)
# This allows us to associate tokens with the correct agent invocation
active_run_ids: Dict[str, str] = {}
run_ids_lock = asyncio.Lock()


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
cdss_process_task: asyncio.Task = None
should_stop = False
cancellation_event = asyncio.Event()

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
    Includes the active run_id for this agent to ensure tokens are routed
    to the correct agent invocation card.
    """
    connections = await get_active_connections()
    if not connections:
        return
    
    # Look up the active run_id for this agent
    async with run_ids_lock:
        run_id = active_run_ids.get(agent_id)
    
    # If no run_id found, log warning but still send token
    # (frontend will handle gracefully)
    if not run_id:
        logger.warning(f"No active run_id found for agent '{agent_id}'. Token may be routed incorrectly.")
    
    message = {
        "type": "token",
        "agent_id": agent_id,
        "run_id": run_id,  # Include run_id to match tokens to correct card
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


async def send_agent_started(agent_id: str):
    """Send agent_started message to create a new card in the UI.
    
    This should be called before each agent invocation to signal that
    a new reasoning session is beginning. Generates a unique run_id
    for this invocation to track tokens correctly.
    
    Args:
        agent_id: The base agent identifier (e.g., 'OrchestratorAgent')
    """
    connections = await get_active_connections()
    if not connections:
        return
    
    # Generate a unique run_id for this agent invocation
    run_id = str(uuid.uuid4())
    
    # Store the active run_id for this agent
    async with run_ids_lock:
        active_run_ids[agent_id] = run_id
    
    message = {
        "type": "agent_started",
        "agent_id": agent_id,
        "run_id": run_id,
        "timestamp": datetime.now().isoformat()
    }
    
    disconnected = []
    for connection in connections:
        try:
            await connection.send_json(message)
        except Exception as e:
            logger.warning(f"Error sending agent_started to client: {e}")
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
        # Look up the active run_id for this agent (synchronous access for fallback)
        run_id = None
        try:
            # Try to get run_id synchronously (may not be perfect but better than nothing)
            run_id = active_run_ids.get(agent_id)
        except:
            pass
        message = {
            "type": "token",
            "agent_id": agent_id,
            "run_id": run_id,  # Include run_id to match tokens to correct card
            "token": token,
            "timestamp": datetime.now().isoformat()
        }
        _queue_message_fallback(message, f"agent {agent_id} token")
    except Exception as e:
        # Fallback to queue if direct send fails
        logger.warning(f"Error sending token directly, falling back to queue: {e}")
        # Look up the active run_id for this agent (synchronous access for fallback)
        run_id = None
        try:
            # Try to get run_id synchronously (may not be perfect but better than nothing)
            run_id = active_run_ids.get(agent_id)
        except:
            pass
        message = {
            "type": "token",
            "agent_id": agent_id,
            "run_id": run_id,  # Include run_id to match tokens to correct card
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
    global cdss_instance, cdss_process_task, should_stop, cancellation_event
    
    # Acquire lock to ensure only one request processes at a time
    async with cdss_lock:
        try:
            # Reset stop flag and cancellation event
            should_stop = False
            cancellation_event.clear()
            
            # Create a new CDSS instance for this request
            cdss_instance = CDSS()
            
            # Register trace callback with EXAID
            cdss_instance.exaid.register_trace_callback(trace_callback)
            logger.debug(f"Trace callback registered. Callbacks: {len(cdss_instance.exaid.trace_callbacks)}")
            
            # Register summary callback with EXAID
            cdss_instance.exaid.register_summary_callback(summary_callback)
            logger.debug(f"Summary callback registered. Callbacks: {len(cdss_instance.exaid.summary_callbacks)}")
            
            # Clear active run IDs when starting a new case
            async with run_ids_lock:
                active_run_ids.clear()
            
            # Send start message
            await broadcast_message({
                "type": "processing_started",
                "timestamp": datetime.now().isoformat()
            })
            
            # Process the case (this will stream traces via callbacks)
            # request.case is already validated and trimmed by Pydantic validator
            # Wrap in a cancellable task
            async def process_with_cancellation():
                try:
                    return await cdss_instance.process_case(request.case)
                except asyncio.CancelledError:
                    logger.info("Case processing was cancelled during execution")
                    raise
                except Exception as e:
                    # Check if cancellation was requested
                    if cancellation_event.is_set() or should_stop:
                        logger.info("Case processing was stopped due to cancellation request")
                        raise asyncio.CancelledError("Processing stopped by user")
                    raise
            
            cdss_process_task = asyncio.create_task(process_with_cancellation())
            
            try:
                await cdss_process_task
            except asyncio.CancelledError:
                logger.info("Case processing was cancelled")
                # Send stop message
                await broadcast_message({
                    "type": "processing_stopped",
                    "timestamp": datetime.now().isoformat()
                })
                return {
                    "status": "stopped",
                    "message": "Case processing was stopped"
                }
            
            # Check if stop was requested
            if should_stop or cancellation_event.is_set():
                logger.info("Case processing was stopped")
                await broadcast_message({
                    "type": "processing_stopped",
                    "timestamp": datetime.now().isoformat()
                })
                return {
                    "status": "stopped",
                    "message": "Case processing was stopped"
                }
            
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
        finally:
            cdss_process_task = None
            cancellation_event.clear()


@app.post("/api/stop-case")
async def stop_case():
    """Stop the currently running case processing and restart servers.
    
    This will kill both backend and frontend processes and restart them.
    """
    global cdss_instance, cdss_process_task, should_stop, cancellation_event
    
    # Clear all state before restarting
    should_stop = True
    cancellation_event.set()
    
    # Cancel and clear the current task
    async with cdss_lock:
        if cdss_process_task and not cdss_process_task.done():
            cdss_process_task.cancel()
            try:
                await asyncio.wait_for(cdss_process_task, timeout=0.5)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass  # Expected if task was cancelled or timed out during shutdown
            cdss_process_task = None
        
        # Reset CDSS instance
        if cdss_instance:
            try:
                cdss_instance.reset()
            except Exception as e:
                logger.warning(f"Error resetting CDSS instance: {e}")
            cdss_instance = None
    
    # Send stop message to frontend before restarting
    try:
        await broadcast_message({
            "type": "processing_stopped",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.warning(f"Error sending stop message: {e}")
    
    # Get project root directory (parent of demos/)
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent  # Go up from server.py -> backend -> demos -> project root
    restart_script = project_root / "restart_servers.ps1"
    
    if not restart_script.exists():
        logger.error(f"Restart script not found at {restart_script}")
        return {
            "status": "error",
            "message": f"Restart script not found at {restart_script}"
        }
    
    # Execute restart script in background
    # Use subprocess.Popen to run in background without blocking
    try:
        if sys.platform == "win32":
            # Windows: Use PowerShell to run the script
            subprocess.Popen(
                ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(restart_script)],
                cwd=str(project_root),
                creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            logger.info("Restart script executed - servers will restart shortly")
        else:
            # Unix/Mac: Use bash
            bash_script = project_root / "restart_servers.sh"
            if bash_script.exists():
                # Make script executable
                os.chmod(str(bash_script), 0o755)
                subprocess.Popen(
                    ["bash", str(bash_script)],
                    cwd=str(project_root),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                logger.info("Restart script executed - servers will restart shortly")
            else:
                logger.error(f"Restart script not found at {bash_script}")
                return {
                    "status": "error",
                    "message": f"Restart script not found at {bash_script}"
                }
        
        # Give a moment for the script to start, then this process will be killed
        await asyncio.sleep(1)
        
        return {
            "status": "success",
            "message": "Servers are being restarted"
        }
    except Exception as e:
        logger.error(f"Error executing restart script: {e}")
        return {
            "status": "error",
            "message": f"Failed to restart servers: {str(e)}"
        }


if __name__ == "__main__":
    import uvicorn
    print("Starting EXAID API server...")
    print("WebSocket endpoint: ws://localhost:8000/ws")
    print("API endpoint: http://localhost:8000/api/process-case")
    uvicorn.run(app, host="0.0.0.0", port=8000)

