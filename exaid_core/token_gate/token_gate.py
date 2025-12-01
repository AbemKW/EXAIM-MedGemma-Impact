from typing import Optional, Dict
from datetime import datetime, timezone


class TokenGate:
    """A lightweight, syntax-aware pre-buffer that regulates token flow into BufferAgent.
    
    The Token Gate does not interpret meaning - it only decides when enough structure
    has accumulated to pass tokens upstream to the BufferAgent for semantic evaluation.
    """
    
    def __init__(
        self,
        min_tokens: int = 35,
        max_tokens: int = 90,
        boundary_cues: str = ".?!\n",
        silence_timer: float = 15,
        max_wait_timeout: float = 40
    ):
        """Initialize TokenGate with configurable flush triggers.
        
        Args:
            min_tokens: Minimum token threshold before flushing (default: 35)
            max_tokens: Maximum token cap to force flush (default: 90)
            boundary_cues: Punctuation/newline characters that trigger early flush (default: ".?!\n")
            silence_timer: Seconds of inactivity before flush (default: 15)
            max_wait_timeout: Maximum seconds before forced flush (default: 40)
        """
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.boundary_cues = boundary_cues
        self.silence_timer = silence_timer
        self.max_wait_timeout = max_wait_timeout
        
        # Per-agent token buffers
        self.buffers: Dict[str, str] = {}
        
        # Track when each agent's buffer started (for max wait timeout)
        self.buffer_start_times: Dict[str, datetime] = {}
        
        # Track when last token was received (for silence timer)
        self.last_token_times: Dict[str, datetime] = {}
    
    def _count_tokens(self, text: str) -> int:
        """Approximate token count using whitespace-based splitting.
        
        This is a simple approximation - actual tokenizers would be more accurate,
        but this avoids external dependencies and is fast enough for flow control.
        """
        if not text:
            return 0
        # Split by whitespace and count non-empty tokens
        tokens = text.split()
        return len(tokens)
    
    def _has_boundary_cue(self, text: str, min_threshold: int) -> bool:
        """Check if text contains boundary cues after reaching ~70% of min_threshold.
        
        Args:
            text: Text to check
            min_threshold: Minimum token threshold
            
        Returns:
            True if boundary cue found after threshold, False otherwise
        """
        token_count = self._count_tokens(text)
        early_flush_threshold = int(min_threshold * 0.7)
        
        if token_count < early_flush_threshold:
            return False
        
        # Check if any boundary cue exists in the text
        for cue in self.boundary_cues:
            if cue in text:
                return True
        
        return False
    
    def _should_flush(self, agent_id: str) -> bool:
        """Check if any flush condition is met for the given agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            True if buffer should be flushed, False otherwise
        """
        if agent_id not in self.buffers:
            return False
        
        buffer_text = self.buffers[agent_id]
        token_count = self._count_tokens(buffer_text)
        
        # Maximum token cap - force flush
        if token_count >= self.max_tokens:
            return True
        
        # Minimum token threshold reached
        if token_count >= self.min_tokens:
            # Check for boundary cue early flush
            if self._has_boundary_cue(buffer_text, self.min_tokens):
                return True
            # Even without boundary cue, flush at min threshold
            return True
        
        return False
    
    def _check_timer_conditions(self, agent_id: str) -> bool:
        """Check if timer-based flush conditions are met.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            True if silence timer or max wait timeout expired, False otherwise
        """
        if agent_id not in self.buffers or not self.buffers[agent_id]:
            return False
        
        now = datetime.now(timezone.utc)
        
        # Check silence timer - if no token received for silence_timer seconds
        if agent_id in self.last_token_times:
            silence_elapsed = (now - self.last_token_times[agent_id]).total_seconds()
            if silence_elapsed >= self.silence_timer:
                return True
        
        # Check max wait timeout - if buffer has existed for max_wait_timeout seconds
        if agent_id in self.buffer_start_times:
            max_wait_elapsed = (now - self.buffer_start_times[agent_id]).total_seconds()
            if max_wait_elapsed >= self.max_wait_timeout:
                return True
        
        return False
    
    async def add_token(self, agent_id: str, token: str) -> Optional[str]:
        """Add a token to the buffer for the given agent.
        
        If flush conditions are met, returns the buffered text and clears the buffer.
        Otherwise, returns None.
        
        Args:
            agent_id: Agent identifier
            token: Token string to add
            
        Returns:
            Flushed chunk text if flush triggered, None otherwise
        """
        now = datetime.now(timezone.utc)
        
        # Check silence timer BEFORE updating (check gap since last token)
        # This handles the case where there's a pause in the stream
        silence_flush = None
        if agent_id in self.buffers and agent_id in self.last_token_times:
            silence_elapsed = (now - self.last_token_times[agent_id]).total_seconds()
            if silence_elapsed >= self.silence_timer and self.buffers[agent_id]:
                # Silence timer expired - flush old buffer
                silence_flush = await self.flush(agent_id)
        
        # Initialize buffer if needed (after potential flush)
        if agent_id not in self.buffers:
            self.buffers[agent_id] = ""
            self.buffer_start_times[agent_id] = now
        
        # Add token to buffer
        self.buffers[agent_id] += token
        
        # Update last token time (silence timer resets on each token)
        self.last_token_times[agent_id] = now
        
        # If silence timer triggered, return the flushed chunk
        # The new token is now in a fresh buffer
        if silence_flush:
            return silence_flush
        
        # Check max wait timeout
        if agent_id in self.buffer_start_times:
            max_wait_elapsed = (now - self.buffer_start_times[agent_id]).total_seconds()
            if max_wait_elapsed >= self.max_wait_timeout:
                return await self.flush(agent_id)
        
        # Check structural flush conditions
        if self._should_flush(agent_id):
            return await self.flush(agent_id)
        
        return None
    
    async def flush(self, agent_id: str) -> Optional[str]:
        """Force flush the buffer for the given agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Flushed buffer text, or None if buffer is empty
        """
        if agent_id not in self.buffers or not self.buffers[agent_id]:
            return None
        
        # Get buffered text
        flushed_text = self.buffers[agent_id]
        
        # Clear buffer
        self.buffers[agent_id] = ""
        
        # Reset timers
        if agent_id in self.buffer_start_times:
            del self.buffer_start_times[agent_id]
        if agent_id in self.last_token_times:
            del self.last_token_times[agent_id]
        
        return flushed_text
    
    async def check_timers(self, agent_id: str) -> Optional[str]:
        """Check if timers have expired and flush if needed.
        
        This should be called periodically or after async operations to check
        if silence timer or max wait timeout has expired.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Flushed chunk if timer expired, None otherwise
        """
        if self._check_timer_conditions(agent_id):
            return await self.flush(agent_id)
        return None

