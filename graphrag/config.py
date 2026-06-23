import os
from dotenv import load_dotenv

load_dotenv()


def get_config() -> dict:
    return {
        "provider": os.getenv("LLM_PROVIDER", "openai").strip().lower(),
        "model": os.getenv("LLM_MODEL", "gpt-4o-mini").strip(),
        "api_key": os.getenv("LLM_API_KEY", "").strip(),
        "base_url": os.getenv("LLM_BASE_URL", "").strip() or None,
        "temperature": float(os.getenv("LLM_TEMPERATURE", "0")),
        "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "4096")),
    }
