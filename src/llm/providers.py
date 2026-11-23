"""Provider configuration for LLM backends."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from .client import LLMClient

load_dotenv()


@dataclass
class ProviderConfig:
    name: str
    base_url: str
    model: str
    api_key_env: str


PROVIDERS: dict[str, ProviderConfig] = {
    "gemini": ProviderConfig(
        name="gemini",
        base_url="https://generativelanguage.googleapis.com",
        model="gemini-2.5-flash",
        api_key_env="GEMINI_API_KEY",
    ),
}


def get_client(provider: str | None = None) -> LLMClient:
    provider = provider or os.getenv("LLM_PROVIDER", "gemini")
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(PROVIDERS)}")
    cfg = PROVIDERS[provider]
    api_key = os.getenv(cfg.api_key_env)
    if not api_key:
        raise ValueError(f"Missing API key env var: {cfg.api_key_env}")
    return LLMClient(
        api_key=api_key,
        base_url=os.getenv("GEMINI_BASE_URL", cfg.base_url),
        model=os.getenv("GEMINI_MODEL", cfg.model),
        provider=cfg.name,
    )
