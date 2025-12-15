"""LLM Registry for EXAID Infrastructure

Centralized LLM management with role-based configuration.
Supports YAML configuration with environment variable overrides.
"""
import os
import yaml
from enum import Enum
from pathlib import Path
from typing import Optional, Union
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI


class LLMRole(str, Enum):
    """Enumeration of LLM roles for type-safe role specification."""
    MAS = "mas"
    SUMMARIZER = "summarizer"
    BUFFER_AGENT = "buffer_agent"


# Lazy loading: Don't load configs or instantiate registry at import time
_CONFIG_PATH = Path(__file__).parent / "model_configs.yaml"
_DEFAULT_CONFIGS = None


def _load_default_configs():
    """Lazy load default configs from YAML."""
    global _DEFAULT_CONFIGS
    if _DEFAULT_CONFIGS is None:
        _DEFAULT_CONFIGS = {}
        if _CONFIG_PATH.exists():
            with open(_CONFIG_PATH, 'r') as f:
                _DEFAULT_CONFIGS = yaml.safe_load(f) or {}
    return _DEFAULT_CONFIGS


def _create_llm_instance(provider: str, model: Optional[str] = None, streaming: bool = True):
    """Factory function to create LLM instances based on provider type.
    
    Args:
        provider: LLM provider name (google, groq, openai)
        model: Model name to use (overrides environment defaults)
        streaming: Whether to enable streaming
        
    Returns:
        Configured LLM instance
        
    Environment variables:
        For Google (Gemini):
            - LLM_API_KEY or GOOGLE_API_KEY: Google API key
            - LLM_MODEL_NAME: Default model name (default: gemini-2.5-flash-lite)
            
        For Groq:
            - GROQ_API_KEY: Groq API key
            - GROQ_MODEL: Default model name (default: llama-3.3-70b-versatile)
            
        For OpenAI (or OpenAI-compatible):
            - OPENAI_API_KEY: API key
            - OPENAI_BASE_URL: Base URL for API (optional, for compatible endpoints)
            - OPENAI_MODEL: Default model name (optional)
    """
    provider = provider.lower()
    
    if provider == "google":
        return ChatGoogleGenerativeAI(
            model=model or os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash-lite"),
            google_api_key=os.getenv("LLM_API_KEY") or os.getenv("GOOGLE_API_KEY"),
            streaming=streaming
        )
    elif provider == "groq":
        return ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model=model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            streaming=streaming
        )
    elif provider == "openai":
        kwargs = {
            "api_key": os.getenv("OPENAI_API_KEY", "NONE"),
            "streaming": streaming
        }
        if base_url := os.getenv("OPENAI_BASE_URL"):
            kwargs["base_url"] = base_url
        if model:
            kwargs["model"] = model
        elif openai_model := os.getenv("OPENAI_MODEL"):
            kwargs["model"] = openai_model
        return ChatOpenAI(**kwargs)
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider}. "
            f"Supported providers: google, groq, openai"
        )


class LLMRegistry:
    """Registry for managing LLM instances by role."""
    
    def __init__(self):
        self._instances = {}
        self._configs = {}
        self._load_configs()
    
    def _load_configs(self):
        """Load configurations from YAML and environment variables."""
        load_dotenv()  # Load env vars when registry is first created
        default_configs = _load_default_configs()
        for role, default_config in default_configs.items():
            # Start with YAML defaults
            config = default_config.copy()
            
            # Override with environment variables if present
            provider_env = f"{role.upper()}_LLM_PROVIDER"
            model_env = f"{role.upper()}_LLM_MODEL"
            
            if provider := os.getenv(provider_env):
                config["provider"] = provider
            if model := os.getenv(model_env):
                config["model"] = model
            
            # Backward compatibility: support old env var names
            if role == "mas":
                if provider := os.getenv("MAS_LLM_PROVIDER"):
                    config["provider"] = provider
                if model := os.getenv("MAS_LLM_MODEL"):
                    config["model"] = model
            elif role == "summarizer":
                # Summarizer uses EXAID_LLM_PROVIDER for backward compatibility
                if provider := os.getenv("EXAID_LLM_PROVIDER"):
                    config["provider"] = provider
                if model := os.getenv("EXAID_LLM_MODEL"):
                    config["model"] = model
            elif role == "buffer_agent":
                # BufferAgent also uses EXAID_LLM_PROVIDER for backward compatibility
                if provider := os.getenv("EXAID_LLM_PROVIDER"):
                    config["provider"] = provider
                if model := os.getenv("EXAID_LLM_MODEL"):
                    config["model"] = model
            
            self._configs[role] = config
    
    def get_llm(self, role: Union[str, LLMRole]):
        """Get or create an LLM instance for the specified role.
        
        Args:
            role: Role name (mas, summarizer, buffer_agent) or LLMRole enum
            
        Returns:
            Configured LLM instance
            
        Raises:
            ValueError: If role is not configured
        """
        # Convert enum to string if needed
        role_str = role.value if isinstance(role, LLMRole) else role
        
        if role_str not in self._configs:
            raise ValueError(
                f"Unknown LLM role: {role}. "
                f"Available roles: {list(self._configs.keys())}"
            )
        
        # Return cached instance if available
        if role_str in self._instances:
            return self._instances[role_str]
        
        # Create new instance
        config = self._configs[role_str]
        instance = _create_llm_instance(
            provider=config["provider"],
            model=config.get("model"),
            streaming=config.get("streaming", True)
        )
        
        # Cache and return
        self._instances[role_str] = instance
        return instance


# Lazy registry: Don't instantiate at import time
_registry: Optional[LLMRegistry] = None


def get_llm(role: Union[str, LLMRole]):
    """Convenience function to get an LLM instance for a role.
    
    Args:
        role: Role name (mas, summarizer, buffer_agent) or LLMRole enum
        
    Returns:
        Configured LLM instance
    """
    global _registry
    if _registry is None:
        _registry = LLMRegistry()
    return _registry.get_llm(role)

