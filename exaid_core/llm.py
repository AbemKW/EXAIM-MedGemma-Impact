import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

load_dotenv()


def _create_llm_instance(provider: str, model: str = None, streaming: bool = True):
    """Factory function to create LLM instances based on provider type.
    
    Args:
        provider: LLM provider name (google, groq, openai)
        model: Model name to use (overrides environment defaults)
        streaming: Whether to enable streaming
        
    Returns:
        Configured LLM instance
        
    Environment variables:
        LLM_PROVIDER: Default provider (default: google)
        MAS_LLM_PROVIDER: Provider for MAS LLM (default: groq)  
        EXAID_LLM_PROVIDER: Provider for EXAID LLM (default: google)
        
        For Google (Gemini):
            - LLM_MODEL_NAME: Model name (default: gemini-2.5-flash-lite)
            - LLM_API_KEY or GOOGLE_API_KEY: Google API key
            
        For Groq:
            - GROQ_API_KEY: Groq API key
            - GROQ_MODEL: Model name (default: llama-3.3-70b-versatile)
            
        For OpenAI (or OpenAI-compatible):
            - OPENAI_API_KEY: API key
            - OPENAI_BASE_URL: Base URL for API (optional, for compatible endpoints)
            - OPENAI_MODEL: Model name (optional)
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


# Default LLM for general use (Gemini Flash - fast and cost-effective)
llm = _create_llm_instance(
    provider=os.getenv("LLM_PROVIDER", "google"),
    streaming=True
)

# Groq LLM for specific use cases requiring Groq
groq_llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
)

# MAS LLM (Multi-Agent System) - configurable provider
# Default: Groq for fast, cost-effective multi-agent reasoning
mas_llm = _create_llm_instance(
    provider=os.getenv("MAS_LLM_PROVIDER", "groq"),
    streaming=True
)

# EXAID LLM (strong reasoning) - configurable provider
# Default: Google Gemini Pro for enhanced reasoning capabilities  
exaid_llm = _create_llm_instance(
    provider=os.getenv("EXAID_LLM_PROVIDER", "google"),
    model=os.getenv("EXAID_LLM_MODEL", "gemini-2.5-pro"),
    streaming=True
)