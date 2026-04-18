from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class SubtitleCue:
    index: int
    start: float
    end: float
    text: str


@dataclass(slots=True)
class VideoDocument:
    video_id: str
    title: str
    file_path: str
    summary: str
    full_text: str
    keyword_list: list[str]
    entity_list: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ChunkDocument:
    chunk_id: str
    video_id: str
    video_title: str
    file_path: str
    start_time: float
    end_time: float
    section_title: str
    summary: str
    text: str
    context_text: str
    keyword_list: list[str]
    entity_list: list[str]
    pinyin_text: str
    jump_url: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SearchHit:
    chunk: ChunkDocument
    score: float
    rerank_score: float
    bm25_score: float
    vector_score: float
    graph_score: float
    explanation: str


def chunk_from_dict(payload: dict[str, Any]) -> ChunkDocument:
    return ChunkDocument(**payload)
