from __future__ import annotations

import json
import os
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import chromadb
import jieba
from pypinyin import Style, lazy_pinyin
from rank_bm25 import BM25Okapi

from app.llm import get_embedding_dimensions, get_embedding_model, embed_texts, get_base_url
from app.models import ChunkDocument, VideoDocument
from app.srt_parser import parse_srt

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "subs"
INDEX_DIR = ROOT / "data" / "index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
VIDEOS_PATH = INDEX_DIR / "videos.json"
MATRIX_PATH = INDEX_DIR / "matrix.pkl"
KG_PATH = INDEX_DIR / "graph.json"
CHROMA_DIR = ROOT / "data" / "chroma"
CHROMA_COLLECTION = "bilibili_up_rag"

FILENAME_RE = re.compile(r"^(?P<title>.+?)\s+\[(?P<bv>BV[0-9A-Za-z]+)\]\.ai-zh\.srt$")
ZH_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]+")

STOPWORDS = {
    "我们",
    "你们",
    "他们",
    "这个",
    "那个",
    "一个",
    "一种",
    "就是",
    "还是",
    "以及",
    "因为",
    "所以",
    "然后",
    "如果",
    "没有",
    "可以",
    "起来",
    "东西",
    "问题",
    "视频",
    "内容",
    "什么",
}
DEFAULT_VIDEO_LIMIT = int(os.getenv("VIDEO_LIMIT", "20"))


def ensure_index() -> None:
    if _index_is_compatible():
        return
    build_index()


def build_index() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    video_docs: list[VideoDocument] = []
    chunk_docs: list[ChunkDocument] = []
    graph: dict[str, Counter[str]] = defaultdict(Counter)
    source_paths = sorted(DATA_DIR.glob("*.srt"))
    if DEFAULT_VIDEO_LIMIT > 0:
        source_paths = source_paths[:DEFAULT_VIDEO_LIMIT]

    for path in source_paths:
        parsed = _parse_filename(path)
        if parsed is None:
            continue
        title, video_id = parsed
        cues = parse_srt(path)
        if not cues:
            continue
        full_text = " ".join(cue.text for cue in cues)
        video_keywords = _top_keywords(f"{title} {full_text}", limit=18)
        video_entities = _extract_entities(f"{title} {full_text}", limit=14)
        video_summary = _build_summary(full_text)
        video_doc = VideoDocument(
            video_id=video_id,
            title=title,
            file_path=str(path.relative_to(ROOT)),
            summary=video_summary,
            full_text=full_text,
            keyword_list=video_keywords,
            entity_list=video_entities,
        )
        video_docs.append(video_doc)

        for chunk_index, chunk in enumerate(_build_chunks(title, video_id, path, cues, video_summary)):
            chunk_docs.append(chunk)
            graph_terms = set(chunk.keyword_list[:10] + chunk.entity_list[:10])
            for term in graph_terms:
                others = graph_terms - {term}
                graph[term].update(others)

    _fit_retrieval_artifacts(chunk_docs)
    _build_chroma_collection(chunk_docs)
    CHUNKS_PATH.write_text(
        json.dumps([chunk.to_dict() for chunk in chunk_docs], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    VIDEOS_PATH.write_text(
        json.dumps([video.to_dict() for video in video_docs], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    KG_PATH.write_text(
        json.dumps({k: v.most_common(12) for k, v in graph.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metadata = {
        "video_limit": DEFAULT_VIDEO_LIMIT,
        "indexed_video_count": len(video_docs),
        "indexed_chunk_count": len(chunk_docs),
        "embedding_model": get_embedding_model(),
        "embedding_dimensions": get_embedding_dimensions(),
        "embedding_base_url": get_base_url(),
    }
    (INDEX_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_index_payload() -> tuple[list[dict], dict, dict, dict]:
    ensure_index()
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    matrix = pickle.loads(MATRIX_PATH.read_bytes())
    graph = json.loads(KG_PATH.read_text(encoding="utf-8"))
    metadata = json.loads((INDEX_DIR / "metadata.json").read_text(encoding="utf-8"))
    return chunks, matrix, graph, metadata


def _index_is_compatible() -> bool:
    metadata_path = INDEX_DIR / "metadata.json"
    if not (
        CHUNKS_PATH.exists()
        and MATRIX_PATH.exists()
        and KG_PATH.exists()
        and CHROMA_DIR.exists()
        and metadata_path.exists()
    ):
        return False

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return (
        metadata.get("embedding_model") == get_embedding_model()
        and int(metadata.get("embedding_dimensions", 0)) == get_embedding_dimensions()
        and metadata.get("embedding_base_url") == get_base_url()
    )


def _parse_filename(path: Path) -> tuple[str, str] | None:
    match = FILENAME_RE.match(path.name)
    if not match:
        return None
    return match.group("title"), match.group("bv")


def _build_chunks(
    title: str,
    video_id: str,
    path: Path,
    cues,
    video_summary: str,
) -> Iterable[ChunkDocument]:
    buffer = []
    for cue in cues:
        if not buffer:
            buffer.append(cue)
            continue
        duration = cue.end - buffer[0].start
        text_size = sum(len(item.text) for item in buffer)
        gap = cue.start - buffer[-1].end
        if duration > 55 or text_size > 230 or gap > 2.8:
            yield _flush_chunk(title, video_id, path, buffer, video_summary)
            buffer = [cue]
        else:
            buffer.append(cue)
    if buffer:
        yield _flush_chunk(title, video_id, path, buffer, video_summary)


def _flush_chunk(
    title: str,
    video_id: str,
    path: Path,
    cues,
    video_summary: str,
) -> ChunkDocument:
    start = cues[0].start
    end = cues[-1].end
    text = " ".join(cue.text for cue in cues).strip()
    section_title = f"{_format_mmss(start)} - {_headline(text)}"
    summary = _build_summary(text, limit=80)
    context_text = (
        f"标题：{title}\n"
        f"视频摘要：{video_summary}\n"
        f"章节：{section_title}\n"
        f"字幕：{text}"
    )
    keywords = _top_keywords(f"{title} {text}", limit=12)
    entities = _extract_entities(f"{title} {text}", limit=10)
    pinyin_text = _to_pinyin(f"{title} {text}")
    chunk_id = f"{video_id}_{int(start * 1000):010d}"
    return ChunkDocument(
        chunk_id=chunk_id,
        video_id=video_id,
        video_title=title,
        file_path=str(path.relative_to(ROOT)),
        start_time=round(start, 3),
        end_time=round(end, 3),
        section_title=section_title,
        summary=summary,
        text=text,
        context_text=context_text,
        keyword_list=keywords,
        entity_list=entities,
        pinyin_text=pinyin_text,
        jump_url=f"https://www.bilibili.com/video/{video_id}?t={round(start, 1)}",
    )


def _build_summary(text: str, limit: int = 120) -> str:
    compact = re.sub(r"\s+", "", text)
    return compact[:limit] + ("..." if len(compact) > limit else "")


def _headline(text: str, limit: int = 16) -> str:
    compact = re.sub(r"\s+", "", text)
    return compact[:limit] + ("..." if len(compact) > limit else "")


def _format_mmss(seconds: float) -> str:
    total = int(seconds)
    minutes = total // 60
    remain = total % 60
    return f"{minutes:02d}:{remain:02d}"


def _tokenize(text: str) -> list[str]:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text.lower())
    tokens = [token.strip() for token in jieba.lcut(cleaned) if token.strip()]
    output: list[str] = []
    for token in tokens:
        if len(token) <= 1 and not token.isdigit():
            continue
        if token in STOPWORDS:
            continue
        output.append(token)
    return output


def _top_keywords(text: str, limit: int) -> list[str]:
    counter = Counter(_tokenize(text))
    return [word for word, _ in counter.most_common(limit)]


def _extract_entities(text: str, limit: int) -> list[str]:
    candidates = [item.group(0) for item in ZH_RE.finditer(text)]
    filtered = [
        word
        for word in candidates
        if len(word) >= 2 and word not in STOPWORDS and not word.isdigit()
    ]
    return [word for word, _ in Counter(filtered).most_common(limit)]


def _to_pinyin(text: str) -> str:
    letters = lazy_pinyin(text, style=Style.NORMAL, errors="ignore")
    return " ".join(part for part in letters if part)


def _fit_retrieval_artifacts(chunk_docs: list[ChunkDocument]) -> None:
    bm25_tokens = [
        _tokenize(
            " ".join(
                [
                    chunk.video_title,
                    chunk.section_title,
                    chunk.summary,
                    chunk.text,
                    " ".join(chunk.keyword_list),
                    chunk.pinyin_text,
                ]
            )
        )
        for chunk in chunk_docs
    ]
    payload = {
        "bm25": BM25Okapi(bm25_tokens),
    }
    MATRIX_PATH.write_bytes(pickle.dumps(payload))


def _build_chroma_collection(chunk_docs: list[ChunkDocument]) -> None:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
    )

    batch_size = 10
    for start in range(0, len(chunk_docs), batch_size):
        batch = chunk_docs[start : start + batch_size]
        documents = [
            "\n".join(
                [
                    chunk.video_title,
                    chunk.section_title,
                    chunk.summary,
                    chunk.context_text,
                    " ".join(chunk.keyword_list),
                    " ".join(chunk.entity_list),
                    chunk.pinyin_text,
                ]
            )
            for chunk in batch
        ]
        embeddings = embed_texts(documents)
        collection.add(
            ids=[chunk.chunk_id for chunk in batch],
            documents=documents,
            embeddings=embeddings,
            metadatas=[
                {
                    "video_id": chunk.video_id,
                    "video_title": chunk.video_title,
                    "start_time": chunk.start_time,
                    "end_time": chunk.end_time,
                    "jump_url": chunk.jump_url,
                    "section_title": chunk.section_title,
                }
                for chunk in batch
            ],
        )
