// WebSocket connection
let ws = null;
let reconnectAttempts = 0;
const maxReconnectAttempts = 5;
const reconnectDelay = 3000;

// Trace storage: agent_id -> trace text
const traces = new Map();

// Summary storage: array of summaries
const summaries = [];

// Token update queue for smooth rendering
let tokenUpdateQueue = [];
let isUpdatingTokens = false;

// DOM elements
const connectionStatus = document.getElementById('connection-status');
const caseForm = document.getElementById('case-form');
const caseInput = document.getElementById('case-input');
const submitBtn = document.getElementById('submit-btn');
const clearBtn = document.getElementById('clear-btn');
const tracesContainer = document.getElementById('traces-container');
const summariesContainer = document.getElementById('summaries-container');
const tracesCountBadge = document.getElementById('traces-count');
const summariesCountBadge = document.getElementById('summaries-count');

// Initialize WebSocket connection
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        console.log('WebSocket connected');
        updateConnectionStatus(true);
        reconnectAttempts = 0;
    };
    
    ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        handleMessage(message);
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        updateConnectionStatus(false);
    };
    
    ws.onclose = () => {
        console.log('WebSocket disconnected');
        updateConnectionStatus(false);
        attemptReconnect();
    };
}

function attemptReconnect() {
    if (reconnectAttempts < maxReconnectAttempts) {
        reconnectAttempts++;
        console.log(`Attempting to reconnect (${reconnectAttempts}/${maxReconnectAttempts})...`);
        setTimeout(() => {
            connectWebSocket();
        }, reconnectDelay);
    } else {
        console.error('Max reconnection attempts reached');
        connectionStatus.textContent = 'Connection Failed';
        connectionStatus.className = 'status-indicator error';
    }
}

function updateConnectionStatus(connected) {
    if (connected) {
        connectionStatus.textContent = 'Connected';
        connectionStatus.className = 'status-indicator connected';
    } else {
        connectionStatus.textContent = 'Disconnected';
        connectionStatus.className = 'status-indicator disconnected';
    }
}

function handleMessage(message) {
    switch (message.type) {
        case 'token':
            // Queue token for smooth rendering
            scheduleTokenUpdate(message.agent_id, message.token);
            break;
        case 'summary':
            appendSummary(message.summary);
            break;
        case 'processing_started':
            clearAll();
            submitBtn.disabled = true;
            submitBtn.textContent = 'Processing...';
            break;
        case 'processing_complete':
            submitBtn.disabled = false;
            submitBtn.textContent = 'Process Case';
            break;
        case 'error':
            console.error('Server error:', message.message);
            alert(`Error: ${message.message}`);
            submitBtn.disabled = false;
            submitBtn.textContent = 'Process Case';
            break;
        default:
            console.log('Unknown message type:', message.type);
    }
}

function scheduleTokenUpdate(agentId, token) {
    // Update immediately without batching for smooth streaming
    appendTokenImmediate(agentId, token);
}

function appendTokenImmediate(agentId, token) {
    // Initialize trace for agent if it doesn't exist
    if (!traces.has(agentId)) {
        traces.set(agentId, '');
        createAgentTraceElement(agentId);
        updateTracesCount();
    }
    
    // Append token to trace immediately
    const currentTrace = traces.get(agentId);
    traces.set(agentId, currentTrace + token);
    
    // Update UI immediately - no batching, no requestAnimationFrame delay
    const traceTextElement = document.getElementById(`trace-text-${agentId}`);
    if (traceTextElement) {
        traceTextElement.textContent = traces.get(agentId);
        
        // Throttle scroll updates to avoid performance issues
        // But update text immediately for smooth streaming
        if (!traceTextElement._scrollTimeout) {
            traceTextElement._scrollTimeout = setTimeout(() => {
                tracesContainer.scrollTop = tracesContainer.scrollHeight;
                traceTextElement._scrollTimeout = null;
            }, 50); // Scroll every 50ms max
        }
    }
}

function updateTracesCount() {
    const count = traces.size;
    tracesCountBadge.textContent = `${count} trace${count !== 1 ? 's' : ''}`;
}

function updateSummariesCount() {
    const count = summaries.length;
    summariesCountBadge.textContent = `${count} summar${count !== 1 ? 'ies' : 'y'}`;
}

// Legacy function - now handled by appendTokenImmediate via scheduleTokenUpdate
function appendToken(agentId, token) {
    scheduleTokenUpdate(agentId, token);
}

function createAgentTraceElement(agentId) {
    // Remove empty message if present
    const emptyMessage = tracesContainer.querySelector('.empty-message');
    if (emptyMessage) {
        emptyMessage.remove();
    }
    
    const traceDiv = document.createElement('div');
    traceDiv.className = 'trace-item';
    traceDiv.id = `trace-${agentId}`;
    
    const agentLabel = document.createElement('span');
    agentLabel.className = 'agent-label';
    agentLabel.textContent = `${agentId}: `;
    
    const traceText = document.createElement('span');
    traceText.className = 'trace-text';
    traceText.id = `trace-text-${agentId}`;
    traceText.textContent = '';
    
    traceDiv.appendChild(agentLabel);
    traceDiv.appendChild(traceText);
    tracesContainer.appendChild(traceDiv);
    
    // Auto-scroll to bottom
    tracesContainer.scrollTop = tracesContainer.scrollHeight;
}

function updateAgentTraceElement(agentId) {
    const traceTextElement = document.getElementById(`trace-text-${agentId}`);
    if (traceTextElement) {
        traceTextElement.textContent = traces.get(agentId);
        
        // Auto-scroll to bottom
        tracesContainer.scrollTop = tracesContainer.scrollHeight;
    }
}

function clearTraces() {
    traces.clear();
    tracesContainer.innerHTML = '<p class="empty-message">No traces yet. Process a case to see verbose reasoning traces.</p>';
    updateTracesCount();
}

function clearSummaries() {
    summaries.length = 0;
    summariesContainer.innerHTML = '<p class="empty-message">No summaries yet. Process a case to see simplified summaries.</p>';
    updateSummariesCount();
}

function clearAll() {
    // Clear update queue
    tokenUpdateQueue = [];
    isUpdatingTokens = false;
    
    clearTraces();
    clearSummaries();
}

function appendSummary(summaryData) {
    // Remove empty message if present
    const emptyMessage = summariesContainer.querySelector('.empty-message');
    if (emptyMessage) {
        emptyMessage.remove();
    }
    
    // Collapse all existing summaries
    const existingCards = summariesContainer.querySelectorAll('.summary-card');
    existingCards.forEach(card => {
        card.classList.remove('expanded');
        card.classList.add('collapsed');
    });
    
    // Create summary card
    const summaryCard = document.createElement('div');
    summaryCard.className = 'summary-card expanded';
    summaryCard.setAttribute('data-summary-id', summaries.length);
    
    const now = new Date();
    const timestamp = now.toLocaleTimeString('en-US', { 
        hour: '2-digit', 
        minute: '2-digit', 
        second: '2-digit',
        hour12: true 
    });
    const dateStr = now.toLocaleDateString('en-US', { 
        month: 'short', 
        day: 'numeric' 
    });
    
    summaryCard.innerHTML = `
        <div class="summary-header clickable">
            <span class="summary-timestamp">${dateStr} • ${timestamp}</span>
            <span class="expand-icon">▼</span>
        </div>
        <div class="summary-content">
            <div class="summary-field">
                <div class="summary-field-label">Status / Action</div>
                <div class="summary-field-value">${escapeHtml(summaryData.status_action)}</div>
            </div>
            <div class="summary-field">
                <div class="summary-field-label">Key Findings</div>
                <div class="summary-field-value">${escapeHtml(summaryData.key_findings)}</div>
            </div>
            <div class="summary-field">
                <div class="summary-field-label">Differential & Rationale</div>
                <div class="summary-field-value">${escapeHtml(summaryData.differential_rationale)}</div>
            </div>
            <div class="summary-field">
                <div class="summary-field-label">Uncertainty / Confidence</div>
                <div class="summary-field-value">${escapeHtml(summaryData.uncertainty_confidence)}</div>
            </div>
            <div class="summary-field">
                <div class="summary-field-label">Recommendation / Next Step</div>
                <div class="summary-field-value">${escapeHtml(summaryData.recommendation_next_step)}</div>
            </div>
            <div class="summary-field">
                <div class="summary-field-label">Agent Contributions</div>
                <div class="summary-field-value">${escapeHtml(summaryData.agent_contributions)}</div>
            </div>
        </div>
    `;
    
    // Add click handler for expand/collapse
    const header = summaryCard.querySelector('.summary-header');
    header.addEventListener('click', () => {
        toggleSummary(summaryCard);
    });
    
    // Insert at the top (newest first)
    summariesContainer.insertBefore(summaryCard, summariesContainer.firstChild);
    
    // Scroll to top to show new summary (but don't force scroll if user is reading)
    if (summariesContainer.scrollTop < 50) {
        summariesContainer.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    }
    
    // Store summary
    summaries.push(summaryData);
    updateSummariesCount();
}

function toggleSummary(summaryCard) {
    const isExpanded = summaryCard.classList.contains('expanded');
    
    if (isExpanded) {
        summaryCard.classList.remove('expanded');
        summaryCard.classList.add('collapsed');
    } else {
        // Collapse all others first
        const allCards = summariesContainer.querySelectorAll('.summary-card');
        allCards.forEach(card => {
            if (card !== summaryCard) {
                card.classList.remove('expanded');
                card.classList.add('collapsed');
            }
        });
        
        // Expand this one
        summaryCard.classList.remove('collapsed');
        summaryCard.classList.add('expanded');
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Initialize count badges
updateTracesCount();
updateSummariesCount();

// Form submission handler
caseForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const caseText = caseInput.value.trim();
    if (!caseText) {
        alert('Please enter a clinical case');
        return;
    }
    
    try {
        submitBtn.disabled = true;
        submitBtn.textContent = 'Processing...';
        
        const response = await fetch('/api/process-case', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ case: caseText }),
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to process case');
        }
        
        const result = await response.json();
        console.log('Case processed:', result);
        
    } catch (error) {
        console.error('Error processing case:', error);
        alert(`Error: ${error.message}`);
        submitBtn.disabled = false;
        submitBtn.textContent = 'Process Case';
    }
});

// Clear button handler
clearBtn.addEventListener('click', () => {
    clearAll();
});

// Initialize WebSocket on page load
connectWebSocket();

