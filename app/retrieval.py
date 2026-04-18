from __future__ import annotations

import math
import re
from collections import defaultdict

import chromadb
import numpy as np

from app.indexer import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    _extract_entities,
    _to_pinyin,
    _tokenize,
    load_index_payload,
)
from app.llm import embed_texts, is_configured, stream_answer
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
        ranked = self._search_hits(cleaned_query, rewrites, top_k=top_k)
        answer = self._synthesize_answer(cleaned_query, ranked)
        return {
            "query": cleaned_query,
            "rewrites": rewrites,
            "answer": answer,
            "hits": ranked,
        }

    def stream_search(self, query: str, top_k: int = 8):
        cleaned_query = query.strip()
        if not cleaned_query:
            yield {"type": "done"}
            return

        yield {"type": "status", "value": "LLM 改写中"}
        rewrites = []
        for rewrite in self._stream_rewrite_queries(cleaned_query):
            rewrites.append(rewrite)
            yield {"type": "rewrite", "value": rewrite}

        ranked = self._search_hits(cleaned_query, rewrites, top_k=top_k)
        yield {"type": "hits", "value": ranked}

        for chunk in self._stream_answer(cleaned_query, ranked):
            yield {"type": "answer", "value": chunk}
        yield {"type": "done"}

    def _search_hits(self, query: str, rewrites: list[str], top_k: int = 8) -> list[SearchHit]:
        scores = defaultdict(lambda: {"bm25": 0.0, "vector": 0.0, "graph": 0.0})
        vector_score_sets = self._vector_search_many(rewrites)

        for current, vector_scores in zip(rewrites, vector_score_sets):
            bm25_scores = self._bm25_search(current)
            graph_scores = self._graph_boost(current)
            for idx, value in bm25_scores.items():
                scores[idx]["bm25"] = max(scores[idx]["bm25"], value)
            for idx, value in vector_scores.items():
                scores[idx]["vector"] = max(scores[idx]["vector"], value)
            for idx, value in graph_scores.items():
                scores[idx]["graph"] = max(scores[idx]["graph"], value)

        return self._rerank(query, scores, top_k=top_k)

    def _rewrite_queries(self, query: str) -> list[str]:
        return list(self._stream_rewrite_queries(query))

    def _stream_rewrite_queries(self, query: str):
        if not is_configured():
            yield from self._rule_rewrite_queries(query)
            return

        rewrites = []
        seen = set()
        buffer = ""

        try:
            for part in stream_answer(self._rewrite_messages(query), temperature=0.1):
                buffer += part.replace("\r\n", "\n")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    current = self._clean_rewrite_line(line)
                    if not current or current in seen:
                        continue
                    seen.add(current)
                    rewrites.append(current)
                    yield current
                    if len(rewrites) >= 10:
                        return

            current = self._clean_rewrite_line(buffer)
            if current and current not in seen and len(rewrites) < 10:
                seen.add(current)
                rewrites.append(current)
                yield current
        except Exception:
            rewrites = []

        if not rewrites:
            yield from self._rule_rewrite_queries(query)

    def _rewrite_messages(self, query: str) -> list[dict]:
        return [
            {
                "role": "system",
                "content": (
                    "你是 B 站字幕 RAG 的查询改写器。"
                    "你的任务是生成用于混合检索的多个搜索 query，不是回答问题。"
                    "输出 6 到 10 行，每行一个 query。"
                    "不要编号、不要项目符号、不要 Markdown、不要解释。"
                    "第一行必须保留用户原始问题。"
                    "其余行应覆盖：核心关键词、同义说法、可能的视频标题表达、背景/原因/影响类伪答案查询。"
                    "如果用户问题包含中文专名，可以加入一行拼音或常见别名。"
                    "query 要短，适合字幕检索。"
                ),
            },
            {
                "role": "user",
                "content": f"用户问题：{query}",
            },
        ]

    def _clean_rewrite_line(self, line: str) -> str:
        current = line.strip()
        current = re.sub(r"^[-*+\d\s.、)）]+", "", current).strip()
        current = current.strip("`'\"“”‘’，,[]")
        current = re.sub(r"\s+", " ", current)
        if not current:
            return ""
        lowered = current.lower()
        if lowered in {"query", "queries", "rewrites"}:
            return ""
        if current.startswith("{") or current.endswith("}"):
            return ""
        return current

    def _rule_rewrite_queries(self, query: str) -> list[str]:
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

    def _vector_search_many(self, queries: list[str]) -> list[dict[int, float]]:
        if not queries:
            return []
        embeddings = embed_texts([f"{query}\n{_to_pinyin(query)}" for query in queries])
        result = self.collection.query(
            query_embeddings=embeddings,
            n_results=30,
            include=["distances"],
        )
        score_sets: list[dict[int, float]] = []
        ids_groups = result.get("ids", [])
        distance_groups = result.get("distances", [])
        for ids, distances in zip(ids_groups, distance_groups):
            current_scores: dict[int, float] = {}
            for chunk_id, distance in zip(ids, distances):
                idx = self.chunk_index_by_id.get(chunk_id)
                if idx is None:
                    continue
                similarity = max(0.0, 1.0 - float(distance))
                if similarity > 0:
                    current_scores[idx] = similarity
            score_sets.append(current_scores)
        return score_sets

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
        return "".join(self._stream_answer(query, hits)).strip()

    def _stream_answer(self, query: str, hits: list[SearchHit]):
        if not hits:
            yield "暂时没有找到明确命中的字幕片段，可以换一个更短的关键词，或者直接输入国家、人物、事件名。"
            return
        if not is_configured():
            lead = hits[0].chunk
            points = []
            for hit in hits[:3]:
                points.append(
                    f"{hit.chunk.video_title} 在 {self._format_time(hit.chunk.start_time)} 附近提到：{hit.chunk.summary}"
                )
            fallback = f"围绕“{query}”，当前最相关的内容集中在《{lead.video_title}》等视频里。" + " ".join(points)
            for part in self._chunk_text(fallback):
                yield part
            return

        evidence_blocks = []
        for index, hit in enumerate(hits[:4], start=1):
            evidence_blocks.append(
                "\n".join(
                    [
                        f"[证据{index}] 视频：{hit.chunk.video_title}",
                        f"时间：{self._format_time(hit.chunk.start_time)} - {self._format_time(hit.chunk.end_time)}",
                        f"章节：{hit.chunk.section_title}",
                        f"摘要：{hit.chunk.summary}",
                        f"字幕：{hit.chunk.text}",
                    ]
                )
            )
        evidence_text = "\n\n".join(evidence_blocks)

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个专业的 B 站视频字幕检索助手，帮助用户从 UP 主的视频字幕中找到准确信息。\n\n"
                    "## 核心原则\n"
                    "- **严格基于证据**：只使用提供的字幕证据作答，绝不编造或推测证据中没有的内容。\n"
                    "- **标注来源**：回答中引用观点时，用「《视频标题》MM:SS」的格式标注出处。\n"
                    "- **诚实透明**：证据不足时明确告知用户，并建议更精确的检索关键词。\n\n"
                    "## 回答结构\n"
                    "1. **一句话结论**：直接回答用户问题的核心。\n"
                    "2. **关键要点**（2-4 条）：从不同证据中提炼要点，每条附带来源标注。\n"
                    "3. **补充说明**（可选）：如果多条证据存在互补或不同视角，简要点明。\n\n"
                    "## 注意事项\n"
                    "- 字幕是语音转录，可能存在错别字或断句不准，理解时注意上下文。\n"
                    "- 如果多条证据来自同一视频的不同时间段，合并概括而非重复引用。\n"
                    "- **重要**：如果证据与用户问题关联度低，直接说明「当前检索结果与问题关联有限」，并省略「关键要点」部分——不要展示无关证据的内容摘要。只需给出结论说明未找到相关内容，并在补充说明中建议用户使用更精确的检索词。\n"
                    "- 保持口语化、易读的风格，避免学术腔。\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"## 用户问题\n{query}\n\n"
                    f"## 检索到的字幕证据\n\n{evidence_text}"
                ),
            },
        ]
        try:
            has_content = False
            for part in stream_answer(messages):
                has_content = True
                yield part
            if not has_content:
                yield "检索到了相关片段，但模型没有返回可用答案。"
        except Exception:
            lead = hits[0].chunk
            points = []
            for hit in hits[:3]:
                points.append(
                    f"{hit.chunk.video_title} 在 {self._format_time(hit.chunk.start_time)} 附近提到：{hit.chunk.summary}"
                )
            fallback = f"围绕“{query}”，当前最相关的内容集中在《{lead.video_title}》等视频里。" + " ".join(points)
            for part in self._chunk_text(fallback):
                yield part

    def _chunk_text(self, text: str, size: int = 24):
        for index in range(0, len(text), size):
            yield text[index:index + size]

    def _format_time(self, value: float) -> str:
        minutes = int(value // 60)
        seconds = int(math.floor(value % 60))
        return f"{minutes:02d}:{seconds:02d}"
