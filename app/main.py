from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.indexer import build_index, ensure_index
from app.retrieval import SearchEngine

BASE_DIR = Path(__file__).resolve().parent
engine: SearchEngine | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global engine
    ensure_index()
    engine = SearchEngine()
    yield


app = FastAPI(title="Bilibili UP RAG", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.get("/", response_class=HTMLResponse)
def home(request: Request, q: str = Query(default="")):
    payload = engine.search(q) if engine and q.strip() else {
        "query": q,
        "rewrites": [],
        "answer": "",
        "hits": [],
    }
    payload["metadata"] = engine.metadata if engine else {}
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context=payload,
    )


@app.get("/api/search")
def search_api(q: str = Query(..., min_length=1), top_k: int = Query(default=8, ge=1, le=20)):
    payload = engine.search(q, top_k=top_k) if engine else {"query": q, "rewrites": [], "answer": "", "hits": []}
    return {
        "query": payload["query"],
        "rewrites": payload["rewrites"],
        "answer": payload["answer"],
        "metadata": engine.metadata if engine else {},
        "hits": [
            {
                "video_title": hit.chunk.video_title,
                "video_id": hit.chunk.video_id,
                "start_time": hit.chunk.start_time,
                "end_time": hit.chunk.end_time,
                "section_title": hit.chunk.section_title,
                "summary": hit.chunk.summary,
                "text": hit.chunk.text,
                "jump_url": hit.chunk.jump_url,
                "explanation": hit.explanation,
                "scores": {
                    "rerank": round(hit.rerank_score, 4),
                    "bm25": round(hit.bm25_score, 4),
                    "vector": round(hit.vector_score, 4),
                    "graph": round(hit.graph_score, 4),
                },
            }
            for hit in payload["hits"]
        ],
    }


@app.post("/admin/rebuild")
def rebuild_index():
    global engine
    build_index()
    engine = SearchEngine()
    return {"ok": True, "message": "索引已重建"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
