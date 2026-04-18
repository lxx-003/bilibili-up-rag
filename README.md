# Bilibili UP 字幕 RAG

一个可本地运行的面试项目 MVP：输入问题，系统会从 `data/subs` 下全部 `.srt` 字幕中做检索，并返回带 B 站空降链接的证据片段。

## 功能

- SRT 解析与时间戳保留
- 父子文档建模：视频级摘要 + 字幕 chunk
- 查询改写：归一化、关键词拆分、拼音、伪答案查询
- 混合检索：BM25 + 向量检索
- 轻量知识图谱增强：实体/关键词共现扩展
- 检索后重排序
- FastAPI 页面展示与 `/api/search` 接口

## 目录

- `app/indexer.py`：索引构建
- `app/retrieval.py`：混合检索与重排序
- `app/main.py`：Web 服务
- `data/subs`：原始字幕
- `data/index`：BM25、元数据、图谱等索引文件
- `data/chroma`：Chroma 持久化向量库

## 启动

1. 安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. 启动服务

```bash
python -m app.main
```

3. 打开浏览器

```text
http://127.0.0.1:8000
```

第一次启动会自动构建索引。
当前向量检索使用 `Chroma PersistentClient`，向量会持久化在 `data/chroma`。

默认只索引前 `20` 个视频，适合先快速验证。想切到全量时：

```bash
VIDEO_LIMIT=0 python -m app.main
```

或者指定任意数量：

```bash
VIDEO_LIMIT=50 python -m app.main
```

## API

```bash
curl "http://127.0.0.1:8000/api/search?q=圣多美为什么发展不起来"
```

返回结果里会包含：

- `video_id`
- `start_time`
- `jump_url`
- `summary`
- `text`
- `scores`

## 后续可继续增强

- 接入更强的 embedding 模型
- 引入真正的 cross-encoder reranker
- 用 LLM 生成更自然的综合答案
- 增加对话记忆和多轮改写
- 加入人工标注评测集做召回率对比
