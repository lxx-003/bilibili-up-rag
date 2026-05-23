# AGENTS.md

## Cursor Cloud specific instructions

### 项目概述

Bilibili UP 字幕 RAG — 基于 Python/FastAPI 的混合检索增强生成应用，从 B 站视频字幕（.srt 文件）中检索并生成回答。详见 `README.md`。

### 前置条件

- Python 3.12+（VM 系统自带）
- 虚拟环境位于 `.venv/`
- `DASHSCOPE_API_KEY` 环境变量（阿里云百炼 API Key）— **必需**，用于索引构建和搜索功能

### 启动应用

```bash
source .venv/bin/activate
python -m app.main
# 监听 http://127.0.0.1:8000
```

首次启动（或删除 `data/index` 和 `data/chroma` 后）会自动构建索引，需调用 Embedding API，因此必须配置 `DASHSCOPE_API_KEY`。索引构建完成后，后续启动无需重建，速度很快。

### 重要注意事项

- 项目根目录必须存在 `.env` 文件（从 `.env.example` 复制并填入 `DASHSCOPE_API_KEY`）。
- 索引构建在 FastAPI lifespan 启动事件中同步执行 — 索引完成前服务器不会接受 HTTP 请求。
- 默认 `VIDEO_LIMIT=20`，仅索引前 20 个视频，适合快速开发验证。设置 `VIDEO_LIMIT=0` 可全量索引。
- 项目当前没有自动化测试。
- 项目没有 lint 配置；如需要可使用 `ruff` 或 `pyright`。
- 无 Docker、无 Makefile、无 Node.js — 纯 Python 项目。

### 常用命令

| 操作 | 命令 |
|------|------|
| 安装依赖 | `source .venv/bin/activate && pip install -r requirements.txt` |
| 启动应用 | `source .venv/bin/activate && python -m app.main` |
| Lint（需安装 ruff） | `source .venv/bin/activate && ruff check app/` |
| 测试 | （项目暂无测试套件） |
