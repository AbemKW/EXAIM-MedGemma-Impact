// WebSocket connection
let ws = null;
let reconnectAttempts = 0;
const maxReconnectAttempts = 5;
const reconnectDelay = 3000;

// Trace storage: agent_id -> { fullText: string, element: HTMLElement }
const agentTraces = new Map();

// Summary storage: summary_id -> { element: HTMLElement, data: object }
const summaryCards = new Map();
let summaryIdCounter = 0;

// DOM elements
const connectionStatus = document.getElementById('connection-status');
const caseForm = document.getElementById('case-form');
const caseInput = document.getElementById('case-input');
const submitBtn = document.getElementById('submit-btn');
const tracesContainer = document.getElementById('traces-container');
const tracesCountBadge = document.getElementById('traces-count');
const summariesContainer = document.getElementById('summaries-container');
const summariesCountBadge = document.getElementById('summaries-count');
const agentModal = document.getElementById('agent-modal');
const modalCloseBtn = document.getElementById('modal-close-btn');
const modalAgentName = document.getElementById('modal-agent-name');
const modalAgentContent = document.getElementById('modal-agent-content');

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
            appendToken(message.agent_id, message.token);
            break;
        case 'summary':
            appendSummary(message.summary_data, message.timestamp);
            break;
        case 'processing_started':
            clearAll();
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<span class="submit-icon">⏳</span>';
            break;
        case 'processing_complete':
            submitBtn.disabled = false;
            submitBtn.innerHTML = '<span class="submit-icon">→</span>';
            break;
        case 'error':
            console.error('Server error:', message.message);
            alert(`Error: ${message.message}`);
            submitBtn.disabled = false;
            submitBtn.innerHTML = '<span class="submit-icon">→</span>';
            break;
        default:
            console.log('Unknown message type:', message.type);
    }
}

// MAS Agents Panel Functions
function appendToken(agentId, token) {
    // Initialize agent trace if it doesn't exist
    if (!agentTraces.has(agentId)) {
        agentTraces.set(agentId, { fullText: '', element: null });
        createAgentWindow(agentId);
        updateTracesCount();
    }
    
    // Append token to full trace
    const agentData = agentTraces.get(agentId);
    agentData.fullText += token;
    
    // Update the display with sliding window effect
    updateAgentWindow(agentId);
}

function createAgentWindow(agentId) {
    // Remove empty message if present
    const emptyMessage = tracesContainer.querySelector('.empty-message');
    if (emptyMessage) {
        emptyMessage.remove();
    }
    
    const agentWindow = document.createElement('div');
    agentWindow.className = 'agent-window';
    agentWindow.id = `agent-window-${agentId}`;
    
    const header = document.createElement('div');
    header.className = 'agent-window-header';
    
    const headerLeft = document.createElement('div');
    headerLeft.className = 'agent-window-header-left';
    
    const agentName = document.createElement('span');
    agentName.className = 'agent-name';
    agentName.textContent = agentId;
    
    const agentBadge = document.createElement('span');
    agentBadge.className = 'agent-badge';
    agentBadge.textContent = 'ACTIVE';
    
    headerLeft.appendChild(agentName);
    headerLeft.appendChild(agentBadge);
    
    const actions = document.createElement('div');
    actions.className = 'agent-window-actions';
    
    const viewFullBtn = document.createElement('button');
    viewFullBtn.className = 'view-full-btn';
    viewFullBtn.textContent = 'View Full';
    viewFullBtn.onclick = (e) => {
        e.stopPropagation();
        showFullAgentOutput(agentId);
    };
    
    const expandIcon = document.createElement('span');
    expandIcon.className = 'expand-icon';
    expandIcon.textContent = '▼';
    
    actions.appendChild(viewFullBtn);
    actions.appendChild(expandIcon);
    
    header.appendChild(headerLeft);
    header.appendChild(actions);
    
    // Make header clickable for expand/collapse
    header.onclick = () => toggleAgentWindow(agentId);
    
    const content = document.createElement('div');
    content.className = 'agent-window-content';
    
    const traceText = document.createElement('div');
    traceText.className = 'agent-trace-text';
    traceText.id = `agent-trace-${agentId}`;
    traceText.textContent = '';
    
    content.appendChild(traceText);
    
    agentWindow.appendChild(header);
    agentWindow.appendChild(content);
    
    // Insert at the top (newest agents first)
    tracesContainer.insertBefore(agentWindow, tracesContainer.firstChild);
    
    // Store reference to element
    const agentData = agentTraces.get(agentId);
    agentData.element = agentWindow;
}

function updateAgentWindow(agentId) {
    const agentData = agentTraces.get(agentId);
    if (!agentData || !agentData.element) return;
    
    const traceTextElement = document.getElementById(`agent-trace-${agentId}`);
    if (!traceTextElement) return;
    
    const contentContainer = traceTextElement.parentElement; // agent-window-content
    
    // Store scroll position before updating content
    const wasAtBottom = contentContainer.scrollHeight - contentContainer.scrollTop - contentContainer.clientHeight < 50;
    const hadScrollableContent = contentContainer.scrollHeight > contentContainer.clientHeight;
    
    // Update text content
    traceTextElement.textContent = agentData.fullText;
    
    // Auto-scroll to bottom if:
    // 1. User was already at/near bottom, OR
    // 2. Content wasn't scrollable before (first time showing content)
    // This ensures new content is always visible unless user manually scrolled up
    const shouldAutoScroll = !hadScrollableContent || wasAtBottom;
    
    if (shouldAutoScroll) {
        // Use requestAnimationFrame to ensure DOM update is complete before scrolling
        requestAnimationFrame(() => {
            contentContainer.scrollTop = contentContainer.scrollHeight;
        });
    }
    
    // Move agent window to top if not already there
    const agentWindow = agentData.element;
    if (agentWindow && agentWindow.parentNode && agentWindow.parentNode.firstChild !== agentWindow) {
        agentWindow.parentNode.removeChild(agentWindow);
        tracesContainer.insertBefore(agentWindow, tracesContainer.firstChild);
    }
}

function toggleAgentWindow(agentId) {
    const agentData = agentTraces.get(agentId);
    if (!agentData || !agentData.element) return;
    
    const agentWindow = agentData.element;
    const isExpanded = agentWindow.classList.contains('expanded');
    
    if (isExpanded) {
        agentWindow.classList.remove('expanded');
    } else {
        agentWindow.classList.add('expanded');
        // Update window after expansion to scroll to bottom and show latest content
        setTimeout(() => {
            updateAgentWindow(agentId);
            // Force scroll to bottom when expanding
            const contentContainer = document.getElementById(`agent-trace-${agentId}`)?.parentElement;
            if (contentContainer) {
                contentContainer.scrollTop = contentContainer.scrollHeight;
            }
        }, 300);
    }
}

function showFullAgentOutput(agentId) {
    const agentData = agentTraces.get(agentId);
    if (!agentData) return;
    
    modalAgentName.textContent = `${agentId} - Full Output`;
    modalAgentContent.textContent = agentData.fullText;
    agentModal.classList.add('show');
}

function closeModal() {
    agentModal.classList.remove('show');
}

function updateTracesCount() {
    const count = agentTraces.size;
    tracesCountBadge.textContent = `${count} trace${count !== 1 ? 's' : ''}`;
}

function clearTraces() {
    agentTraces.clear();
    tracesContainer.innerHTML = '<p class="empty-message">No traces yet. Process a case to see verbose reasoning traces.</p>';
    updateTracesCount();
}

// Summary Panel Functions
function appendSummary(summaryData, timestamp) {
    // Remove empty message if present
    const emptyMessage = summariesContainer.querySelector('.empty-message');
    if (emptyMessage) {
        emptyMessage.remove();
    }
    
    // Generate unique ID for this summary
    const summaryId = `summary-${summaryIdCounter++}`;
    
    // Create summary card element
    const summaryCard = createSummaryCard(summaryId, summaryData, timestamp);
    
    // Collapse all existing cards
    summaryCards.forEach((cardInfo) => {
        cardInfo.element.classList.remove('expanded');
        cardInfo.element.classList.add('collapsed');
    });
    
    // Insert at the top (newest summaries first)
    summariesContainer.insertBefore(summaryCard, summariesContainer.firstChild);
    
    // Store reference
    summaryCards.set(summaryId, {
        element: summaryCard,
        data: summaryData
    });
    
    // Update count
    updateSummariesCount();
}

function createSummaryCard(summaryId, summaryData, timestamp) {
    const card = document.createElement('div');
    card.className = 'summary-card expanded';
    card.id = summaryId;
    
    // Format timestamp
    const timeStr = new Date(timestamp).toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });
    
    // Header (always visible)
    const header = document.createElement('div');
    header.className = 'summary-card-header';
    header.onclick = () => toggleSummaryCard(summaryId);
    
    const statusAction = document.createElement('div');
    statusAction.className = 'summary-status-action';
    statusAction.textContent = summaryData.status_action;
    
    const timestampEl = document.createElement('span');
    timestampEl.className = 'summary-timestamp';
    timestampEl.textContent = timeStr;
    
    const expandIcon = document.createElement('span');
    expandIcon.className = 'summary-expand-icon';
    expandIcon.textContent = '▼';
    
    header.appendChild(statusAction);
    header.appendChild(timestampEl);
    header.appendChild(expandIcon);
    
    // Content (visible when expanded)
    const content = document.createElement('div');
    content.className = 'summary-card-content';
    
    // Create fields
    const fields = [
        { label: 'Status / Action', value: summaryData.status_action, className: 'status-action' },
        { label: 'Key Findings', value: summaryData.key_findings, className: 'key-findings' },
        { label: 'Differential & Rationale', value: summaryData.differential_rationale, className: 'differential' },
        { label: 'Uncertainty / Confidence', value: summaryData.uncertainty_confidence, className: 'uncertainty' },
        { label: 'Recommendation / Next Step', value: summaryData.recommendation_next_step, className: 'recommendation' },
        { label: 'Agent Contributions', value: summaryData.agent_contributions, className: 'agent-contributions' }
    ];
    
    fields.forEach(field => {
        const fieldDiv = document.createElement('div');
        fieldDiv.className = `summary-field ${field.className}`;
        
        const label = document.createElement('div');
        label.className = 'summary-field-label';
        label.textContent = field.label;
        
        const value = document.createElement('div');
        value.className = 'summary-field-value';
        value.textContent = field.value;
        
        fieldDiv.appendChild(label);
        fieldDiv.appendChild(value);
        content.appendChild(fieldDiv);
    });
    
    card.appendChild(header);
    card.appendChild(content);
    
    return card;
}

function toggleSummaryCard(summaryId) {
    const cardInfo = summaryCards.get(summaryId);
    if (!cardInfo) return;
    
    const card = cardInfo.element;
    const isExpanded = card.classList.contains('expanded');
    
    if (isExpanded) {
        // Collapse this card
        card.classList.remove('expanded');
        card.classList.add('collapsed');
    } else {
        // Collapse all other cards first
        summaryCards.forEach((info, id) => {
            if (id !== summaryId) {
                info.element.classList.remove('expanded');
                info.element.classList.add('collapsed');
            }
        });
        
        // Expand this card
        card.classList.remove('collapsed');
        card.classList.add('expanded');
    }
}

function updateSummariesCount() {
    const count = summaryCards.size;
    summariesCountBadge.textContent = `${count} summar${count !== 1 ? 'ies' : 'y'}`;
}

function clearSummaries() {
    summaryCards.clear();
    summaryIdCounter = 0;
    summariesContainer.innerHTML = '<p class="empty-message">No summaries yet. Summaries will appear here as EXAID processes agent traces.</p>';
    updateSummariesCount();
}

function clearAll() {
    clearTraces();
    clearSummaries();
}

// Auto-resize textarea
caseInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 200) + 'px';
});

// Form submission handler
caseForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const caseText = caseInput.value.trim();
    if (!caseText) {
        alert('Please enter a patient case');
        return;
    }
    
    try {
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<span class="submit-icon">⏳</span>';
        
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
        
        // Clear input after successful submission
        caseInput.value = '';
        caseInput.style.height = 'auto';
        
    } catch (error) {
        console.error('Error processing case:', error);
        alert(`Error: ${error.message}`);
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<span class="submit-icon">→</span>';
    }
});

// Modal close handlers
modalCloseBtn.addEventListener('click', closeModal);

agentModal.addEventListener('click', (e) => {
    if (e.target === agentModal) {
        closeModal();
    }
});

// Close modal on Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && agentModal.classList.contains('show')) {
        closeModal();
    }
});

// Initialize count badges
updateTracesCount();
updateSummariesCount();

// Initialize WebSocket on page load
connectWebSocket();

