import sys
import json
import asyncio
from pathlib import Path
from typing import List
from datetime import datetime
from contextlib import asynccontextmanager

# Add project root to path
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from cdss_demo.cdss import CDSS

# Message queue for trace events (thread-safe)
# Use a large maxsize to prevent dropping tokens
message_queue = asyncio.Queue(maxsize=10000)


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
        pass


app = FastAPI(lifespan=lifespan)

# Store active WebSocket connections
active_connections: List[WebSocket] = []

# Global CDSS instance (will be created per request)
cdss_instance: CDSS = None


class CaseRequest(BaseModel):
    case: str


async def send_token_direct(agent_id: str, token: str):
    """Send token directly to WebSocket clients without queuing.
    
    This bypasses the message queue for immediate token-by-token delivery.
    """
    if not active_connections:
        return
    
    message = {
        "type": "token",
        "agent_id": agent_id,
        "token": token,
        "timestamp": datetime.now().isoformat()
    }
    
    disconnected = []
    for connection in active_connections:
        try:
            await connection.send_json(message)
        except Exception as e:
            print(f"Error sending token to client: {e}")
            disconnected.append(connection)
    
    # Remove disconnected clients
    for conn in disconnected:
        if conn in active_connections:
            active_connections.remove(conn)


async def send_summary_direct(summary):
    """Send summary directly to WebSocket clients.
    
    Args:
        summary: AgentSummary Pydantic object to broadcast
    """
    if not active_connections:
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
    for connection in active_connections:
        try:
            await connection.send_json(message)
        except Exception as e:
            print(f"Error sending summary to client: {e}")
            disconnected.append(connection)
    
    # Remove disconnected clients
    for conn in disconnected:
        if conn in active_connections:
            active_connections.remove(conn)


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
        # Try to get the running event loop
        loop = asyncio.get_running_loop()
        # Schedule the send immediately as a task
        asyncio.create_task(send_token_direct(agent_id, token))
    except RuntimeError:
        # No running event loop - this shouldn't happen but fallback to queue
        message = {
            "type": "token",
            "agent_id": agent_id,
            "token": token,
            "timestamp": datetime.now().isoformat()
        }
        try:
            message_queue.put_nowait(message)
        except asyncio.QueueFull:
            print(f"Warning: Message queue full for agent {agent_id}, token dropped.")
        except Exception as queue_error:
            print(f"Error queueing message for agent {agent_id}: {queue_error}")
    except Exception as e:
        # Fallback to queue if direct send fails
        print(f"Error sending token directly, falling back to queue: {e}")
        message = {
            "type": "token",
            "agent_id": agent_id,
            "token": token,
            "timestamp": datetime.now().isoformat()
        }
        try:
            message_queue.put_nowait(message)
        except asyncio.QueueFull:
            print(f"Warning: Message queue full for agent {agent_id}, token dropped.")
        except Exception as queue_error:
            print(f"Error queueing message for agent {agent_id}: {queue_error}")


def summary_callback(summary):
    """Callback function to send summaries directly to WebSocket clients.
    
    This is called when EXAID generates a new summary from buffered traces.
    
    Args:
        summary: AgentSummary Pydantic object
    """
    try:
        # Try to get the running event loop
        loop = asyncio.get_running_loop()
        # Schedule the send immediately as a task
        asyncio.create_task(send_summary_direct(summary))
    except RuntimeError:
        # No running event loop - fallback to queue
        print(f"Warning: No running event loop for summary callback")
    except Exception as e:
        print(f"Error sending summary directly: {e}")


async def broadcast_message(message: dict):
    """Broadcast a message to all connected WebSocket clients."""
    if not active_connections:
        return
    
    disconnected = []
    for connection in active_connections:
        try:
            await connection.send_json(message)
        except Exception as e:
            print(f"Error sending message to client: {e}")
            disconnected.append(connection)
    
    # Remove disconnected clients
    for conn in disconnected:
        if conn in active_connections:
            active_connections.remove(conn)


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
                    print(f"Error processing queued message: {e}")
                    message_queue.task_done()
                    
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Error in message broadcaster: {e}")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time trace updates."""
    await websocket.accept()
    active_connections.append(websocket)
    print(f"Client connected. Total connections: {len(active_connections)}")
    
    try:
        # Keep connection alive - just wait for disconnect
        # Messages are sent via broadcast_message() from other parts of the app
        while True:
            # Wait for any message (or disconnect)
            data = await websocket.receive_text()
            # Optional: handle client messages here if needed
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        print(f"Client disconnected. Total connections: {len(active_connections)}")
    except Exception as e:
        print(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)


@app.post("/api/process-case")
async def process_case(request: CaseRequest):
    """Process a clinical case through the CDSS system."""
    global cdss_instance
    
    try:
        # Create a new CDSS instance for this request
        cdss_instance = CDSS()
        
        # Register trace callback with EXAID
        cdss_instance.exaid.register_trace_callback(trace_callback)
        print(f"Trace callback registered. Callbacks: {len(cdss_instance.exaid.trace_callbacks)}")
        
        # Register summary callback with EXAID
        cdss_instance.exaid.register_summary_callback(summary_callback)
        print(f"Summary callback registered. Callbacks: {len(cdss_instance.exaid.summary_callbacks)}")
        
        # Send start message
        await broadcast_message({
            "type": "processing_started",
            "timestamp": datetime.now().isoformat()
        })
        
        # Process the case (this will stream traces via callbacks)
        result = await cdss_instance.process_case(request.case)
        
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
        print(f"Error processing case: {error_msg}")
        
        # Send error message to clients
        await broadcast_message({
            "type": "error",
            "message": error_msg,
            "timestamp": datetime.now().isoformat()
        })
        
        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/")
async def read_root():
    """Serve the main HTML page."""
    return FileResponse(Path(__file__).parent / "index.html")


# Mount static files (CSS and JS)
app.mount("/css", StaticFiles(directory=Path(__file__).parent / "css"), name="css")
app.mount("/js", StaticFiles(directory=Path(__file__).parent / "js"), name="js")


if __name__ == "__main__":
    import uvicorn
    print("Starting Reasoning Traces UI server...")
    print("Open http://localhost:8000 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=8000)

