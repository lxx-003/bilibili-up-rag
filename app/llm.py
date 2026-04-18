from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_EMBEDDING_MODEL = "text-embedding-v4"
DEFAULT_CHAT_MODEL = "qwen3.6-flash"
DEFAULT_EMBEDDING_DIMENSIONS = 1024


def get_api_key() -> str | None:
    return os.getenv("DASHSCOPE_API_KEY")


def is_configured() -> bool:
    return bool(get_api_key())


def get_base_url() -> str:
    return os.getenv("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL)


def get_embedding_model() -> str:
    return os.getenv("DASHSCOPE_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def get_chat_model() -> str:
    return os.getenv("DASHSCOPE_CHAT_MODEL", DEFAULT_CHAT_MODEL)


def get_embedding_dimensions() -> int:
    raw_value = os.getenv("DASHSCOPE_EMBEDDING_DIMENSIONS", str(DEFAULT_EMBEDDING_DIMENSIONS))
    return int(raw_value)


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("Missing DASHSCOPE_API_KEY")
    return OpenAI(
        api_key=api_key,
        base_url=get_base_url(),
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    response = get_client().embeddings.create(
        model=get_embedding_model(),
        input=texts,
        dimensions=get_embedding_dimensions(),
        encoding_format="float",
    )
    return [item.embedding for item in response.data]


def generate_answer(messages: list[dict], temperature: float = 0.2) -> str:
    response = get_client().chat.completions.create(
        model=get_chat_model(),
        messages=messages,
        temperature=temperature,
    )
    message = response.choices[0].message.content
    if isinstance(message, str):
        return message.strip()
    if isinstance(message, list):
        return "".join(
            part.get("text", "")
            for part in message
            if isinstance(part, dict)
        ).strip()
    return ""
