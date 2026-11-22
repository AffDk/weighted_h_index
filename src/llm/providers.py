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
    "qwen": ProviderConfig(
        name="qwen",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen3.7-flash",
        api_key_env="DASHSCOPE_API_KEY",
    ),
    "deepseek": ProviderConfig(
        name="deepseek",
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        api_key_env="DEEPSEEK_API_KEY",
    ),
    "kimi": ProviderConfig(
        name="kimi",
        base_url="https://api.moonshot.cn/v1",
        model="kimi-k2",
        api_key_env="MOONSHOT_API_KEY",
    ),
    "glm": ProviderConfig(
        name="glm",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        model="glm-4-flash",
        api_key_env="ZHIPU_API_KEY",
    ),
}


def get_client(provider: str | None = None) -> LLMClient:
    """Build an LLMClient for the given provider (default from env or qwen)."""
    provider = provider or os.getenv("LLM_PROVIDER", "qwen")
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(PROVIDERS)}")
    cfg = PROVIDERS[provider]
    api_key = os.getenv(cfg.api_key_env)
    if not api_key:
        raise ValueError(f"Missing API key env var: {cfg.api_key_env}")
    return LLMClient(
        api_key=api_key,
        base_url=cfg.base_url,
        model=cfg.model,
        provider=cfg.name,
    )
