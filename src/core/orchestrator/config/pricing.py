"""Centralized model pricing configuration."""

# Pricing per 1M tokens (convert to per-token in calculations)
MODEL_PRICING = {
    "google/gemini-3-pro-preview": {
        "input": 2.0,      # $2 per 1M input tokens
        "output": 12.0,    # $12 per 1M output tokens
    },
    "gemini-3-pro-preview": {
        "input": 2.0,      # $2 per 1M input tokens
        "output": 12.0,    # $12 per 1M output tokens
    },
    "gemini-3-flash-preview": {
        "input": 0.5,      # $0.50 per 1M input tokens
        "output": 3.0,     # $3.00 per 1M output tokens
    },
    "gemini-2.0-flash": {
        "input": 0.1,      # $0.10 per 1M input tokens (cheaper)
        "output": 0.4,     # $0.40 per 1M output tokens
    },
    "openai/gpt-4.1-mini": {
        "input": 0.4,      # $0.4 per 1M input tokens
        "output": 1.6,     # $1.6 per 1M output tokens
    },
    "gpt-4.1-mini-2025-04-14": {
        "input": 0.4,      # $0.40 per 1M input tokens
        "output": 1.6,     # $1.60 per 1M output tokens
    },
    "gpt-4o": {
        "input": 2.50,     # $2.50 per 1M input tokens
        "output": 10.0,    # $10 per 1M output tokens
    },
    "openai/gpt-4o": {
        "input": 2.50,     # $2.50 per 1M input tokens
        "output": 10.0,    # $10 per 1M output tokens
    }
}
