from __future__ import annotations

import math
import re
from collections import defaultdict

import chromadb
import numpy as np
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from app.indexer import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    _extract_entities,
    _to_pinyin,
    _tokenize,
    load_index_payload,
)
from app.models import SearchHit, chunk_from_dict


class SearchEngine:
    def __init__(self) -> None:
        chunks, matrix, graph, metadata = load_index_payload()
        self.chunks = [chunk_from_dict(item) for item in chunks]
        self.chunk_by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        self.chunk_index_by_id = {chunk.chunk_id: idx for idx, chunk in enumerate(self.chunks)}
        self.bm25 = matrix["bm25"]
        self.metadata = metadata
        self.graph = {
            key: [name for name, _ in values]
            for key, values in graph.items()
        }
        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            embedding_function=DefaultEmbeddingFunction(),
        )

    def search(self, query: str, top_k: int = 8) -> dict:
        cleaned_query = query.strip()
        if not cleaned_query:
            return {
                "query": "",
                "rewrites": [],
                "answer": "",
                "hits": [],
            }

        rewrites = self._rewrite_queries(cleaned_query)
        scores = defaultdict(lambda: {"bm25": 0.0, "vector": 0.0, "graph": 0.0})

        for current in rewrites:
            bm25_scores = self._bm25_search(current)
            vector_scores = self._vector_search(current)
            graph_scores = self._graph_boost(current)
            for idx, value in bm25_scores.items():
                scores[idx]["bm25"] = max(scores[idx]["bm25"], value)
            for idx, value in vector_scores.items():
                scores[idx]["vector"] = max(scores[idx]["vector"], value)
            for idx, value in graph_scores.items():
                scores[idx]["graph"] = max(scores[idx]["graph"], value)

        ranked = self._rerank(cleaned_query, scores, top_k=top_k)
        answer = self._synthesize_answer(cleaned_query, ranked)
        return {
            "query": cleaned_query,
            "rewrites": rewrites,
            "answer": answer,
            "hits": ranked,
        }

    def _rewrite_queries(self, query: str) -> list[str]:
        rewrites = [query]
        normalized = re.sub(r"[，。！？、\s]+", " ", query).strip()
        if normalized and normalized != query:
            rewrites.append(normalized)

        keywords = _tokenize(query)
        if keywords:
            rewrites.append(" ".join(keywords[:6]))
        if len(keywords) >= 2:
            rewrites.extend(keywords[:3])

        pseudo_answer = f"这个问题可能涉及 {' '.join(keywords[:5]) or query} 的背景、原因、影响和案例"
        rewrites.append(pseudo_answer)

        pinyin = _to_pinyin(query)
        if pinyin and pinyin != query:
            rewrites.append(pinyin)

        entities = _extract_entities(query, limit=8)
        for entity in entities[:4]:
            neighbors = self.graph.get(entity, [])[:3]
            if neighbors:
                rewrites.append(" ".join([entity, *neighbors]))

        # 保留顺序去重
        result = []
        seen = set()
        for item in rewrites:
            current = item.strip()
            if current and current not in seen:
                seen.add(current)
                result.append(current)
        return result[:10]

    def _bm25_search(self, query: str) -> dict[int, float]:
        tokens = _tokenize(f"{query} {_to_pinyin(query)}")
        if not tokens:
            return {}
        raw_scores = self.bm25.get_scores(tokens)
        max_score = float(np.max(raw_scores)) if len(raw_scores) else 0.0
        if max_score <= 0:
            return {}
        best_indices = np.argsort(raw_scores)[-30:]
        return {int(idx): float(raw_scores[idx] / max_score) for idx in best_indices if raw_scores[idx] > 0}

    def _vector_search(self, query: str) -> dict[int, float]:
        result = self.collection.query(
            query_texts=[f"{query}\n{_to_pinyin(query)}"],
            n_results=30,
            include=["distances"],
        )
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        scores: dict[int, float] = {}
        for chunk_id, distance in zip(ids, distances):
            idx = self.chunk_index_by_id.get(chunk_id)
            if idx is None:
                continue
            similarity = max(0.0, 1.0 - float(distance))
            if similarity > 0:
                scores[idx] = similarity
        return scores

    def _graph_boost(self, query: str) -> dict[int, float]:
        entities = _extract_entities(query, limit=6) + _tokenize(query)[:6]
        expanded = set(entities)
        for entity in entities:
            expanded.update(self.graph.get(entity, [])[:3])
        if not expanded:
            return {}
        boosts = {}
        for idx, chunk in enumerate(self.chunks):
            overlap = expanded.intersection(set(chunk.keyword_list + chunk.entity_list))
            if overlap:
                boosts[idx] = min(1.0, len(overlap) / 4.0)
        return boosts

    def _rerank(self, query: str, scores: dict[int, dict[str, float]], top_k: int) -> list[SearchHit]:
        query_terms = set(_tokenize(query))
        query_entities = set(_extract_entities(query, limit=10))
        ranked: list[SearchHit] = []
        for idx, feature in scores.items():
            chunk = self.chunks[idx]
            chunk_terms = set(chunk.keyword_list)
            chunk_entities = set(chunk.entity_list)
            overlap = len(query_terms.intersection(chunk_terms))
            entity_overlap = len(query_entities.intersection(chunk_entities))
            title_bonus = 0.15 if any(term in chunk.video_title for term in query_terms) else 0.0
            section_bonus = 0.1 if any(term in chunk.section_title for term in query_terms) else 0.0
            rerank = (
                0.45 * feature["bm25"]
                + 0.30 * feature["vector"]
                + 0.10 * feature["graph"]
                + 0.08 * min(1.0, overlap / 3.0)
                + 0.07 * min(1.0, entity_overlap / 2.0)
                + title_bonus
                + section_bonus
            )
            explanation_parts = []
            if feature["bm25"] > 0.35:
                explanation_parts.append("关键词命中强")
            if feature["vector"] > 0.35:
                explanation_parts.append("语义相似")
            if feature["graph"] > 0:
                explanation_parts.append("相关实体扩展")
            if title_bonus:
                explanation_parts.append("标题相关")
            if not explanation_parts:
                explanation_parts.append("上下文匹配")

            ranked.append(
                SearchHit(
                    chunk=chunk,
                    score=rerank,
                    rerank_score=rerank,
                    bm25_score=feature["bm25"],
                    vector_score=feature["vector"],
                    graph_score=feature["graph"],
                    explanation=" + ".join(explanation_parts),
                )
            )
        ranked.sort(key=lambda item: item.score, reverse=True)
        deduped: list[SearchHit] = []
        seen = set()
        for hit in ranked:
            signature = (hit.chunk.video_id, int(hit.chunk.start_time // 20))
            if signature in seen:
                continue
            seen.add(signature)
            deduped.append(hit)
            if len(deduped) >= top_k:
                break
        return deduped

    def _synthesize_answer(self, query: str, hits: list[SearchHit]) -> str:
        if not hits:
            return "暂时没有找到明确命中的字幕片段，可以换一个更短的关键词，或者直接输入国家、人物、事件名。"
        lead = hits[0].chunk
        points = []
        for hit in hits[:3]:
            points.append(
                f"{hit.chunk.video_title} 在 {self._format_time(hit.chunk.start_time)} 附近提到：{hit.chunk.summary}"
            )
        return f"围绕“{query}”，当前最相关的内容集中在《{lead.video_title}》等视频里。" + " ".join(points)

    def _format_time(self, value: float) -> str:
        minutes = int(value // 60)
        seconds = int(math.floor(value % 60))
        return f"{minutes:02d}:{seconds:02d}"
