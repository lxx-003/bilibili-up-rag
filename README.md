# Bilibili UP 字幕 RAG

一个可本地运行的 RAG 项目：输入问题，系统从 B 站 UP 主的 `.srt` 字幕中做混合检索 + LLM 生成回答，并返回带 **B 站空降链接** 的证据片段，点击即可跳转到视频对应时刻。

## 技术栈

| 类别 | 选型 |
|------|------|
| Web 框架 | FastAPI + Uvicorn |
| 向量模型 | 阿里云百炼 `text-embedding-v4`（1024 维） |
| 生成模型 | 阿里云百炼 `qwen3.6-flash` |
| 向量数据库 | ChromaDB（本地持久化） |
| 关键词检索 | rank-bm25 + jieba 中文分词 |
| 拼音匹配 | pypinyin |
| 前端 | 原生 HTML/CSS/JS + SSE 流式渲染 |

## 功能亮点

- **SRT 解析与时间戳保留**：自动解析字幕文件，保留精确时间码，生成 B 站空降链接
- **父子文档建模**：视频级摘要 + 字幕 chunk（按 55s / 230 字 / 2.8s 间隔自动切分）
- **LLM 查询改写**：调用 LLM 生成 6-10 个语义改写查询；LLM 不可用时回退到规则改写（归一化、分词、拼音、伪答案）
- **四维混合检索**：BM25 关键词 + 向量语义 + 知识图谱扩展 + 自定义重排序
- **轻量知识图谱**：基于 chunk 内关键词 / 命名实体共现构建邻接图，支持 3 跳扩展
- **加权重排序**：BM25 (45%) + 向量 (30%) + 图谱 (10%) + 关键词重叠 (8%) + 实体重叠 (7%)，标题 / 章节命中额外加分
- **同视频去重**：20s 时间窗口内仅保留最高分结果
- **SSE 流式响应**：搜索过程实时推送改写查询 → 检索结果 → LLM 生成回答
- **LLM 异常回退**：生成失败时自动回退为基于摘要的证据拼接

## 项目结构

```
app/
├── main.py          # FastAPI 入口，/api/search (SSE) 与 /api/search/sync 端点
├── indexer.py        # 索引构建：SRT 解析 → 分块 → Embedding → BM25 / ChromaDB / 图谱
├── retrieval.py      # SearchEngine：查询改写 → 混合检索 → 重排序 → LLM 生成
├── llm.py            # 阿里云百炼 OpenAI 兼容封装（Embedding / Chat / 重试）
├── models.py         # 数据模型：VideoDocument, ChunkDocument, SearchHit
├── srt_parser.py     # SRT 格式解析器
├── templates/
│   └── index.html    # 前端页面（SSE 流式 + 按视频分组 + 得分可视化）
└── static/
    └── styles.css    # 响应式样式

data/
├── subs/             # 原始 .srt 字幕文件（命名：{标题} [{BV号}].ai-zh.srt）
├── index/            # BM25 矩阵 (matrix.pkl)、元数据、图谱 (graph.json)、chunks/videos JSON
└── chroma/           # ChromaDB 持久化向量库
```

## 快速开始

### 1. 安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的百炼 API Key
```

`.env` 支持的完整变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DASHSCOPE_API_KEY` | — | **必填**，阿里云百炼 API Key |
| `DASHSCOPE_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | API 基础地址 |
| `DASHSCOPE_EMBEDDING_MODEL` | `text-embedding-v4` | 向量模型 |
| `DASHSCOPE_EMBEDDING_DIMENSIONS` | `1024` | 向量维度 |
| `DASHSCOPE_CHAT_MODEL` | `qwen3.6-flash` | 生成模型 |
| `VIDEO_LIMIT` | `20` | 索引视频数上限，`0` 表示全量 |
| `EMBED_BATCH_SIZE` | `10` | Embedding API 批大小 |
| `EMBED_WORKERS` | `2` | 并发 Embedding 线程数 |
| `EMBED_INTERVAL` | `0.3` | 请求间隔（秒），防限流 |
| `EMBED_MAX_RETRIES` | `6` | Embedding 失败重试次数 |

### 3. 启动服务

```bash
python -m app.main
```

首次启动会自动构建索引（解析字幕 → Embedding → 写入 ChromaDB + BM25 + 图谱）。
如果更换了 Embedding 模型、维度或 `base_url`，项目会自动检测并重建索引。

默认只索引前 **20** 个视频，适合快速验证。全量索引：

```bash
VIDEO_LIMIT=0 python -m app.main
```

### 4. 打开浏览器

```
http://127.0.0.1:8000
```

## API

### SSE 流式搜索（推荐）

```bash
curl -N "http://127.0.0.1:8000/api/search?q=圣多美为什么发展不起来"
```

SSE 事件流依次推送：`rewritten_queries` → `hits` → `answer`（逐 token）→ `done`

### 同步搜索

```bash
curl "http://127.0.0.1:8000/api/search/sync?q=圣多美为什么发展不起来"
```

返回 JSON，字段包含：

| 字段 | 说明 |
|------|------|
| `video_id` | 视频 BV 号 |
| `video_title` | 视频标题 |
| `start_time` | 字幕起始秒数 |
| `jump_url` | B 站空降链接 |
| `summary` | 视频 / 章节摘要 |
| `text` | 命中的字幕原文 |
| `score` | 综合重排序得分 |
| `scores` | 各维度得分明细（BM25 / 向量 / 图谱） |

## 检索流程

```
用户提问
  │
  ▼
LLM 查询改写（6-10 个候选查询）
  │
  ├─► BM25 关键词检索（Top 30）
  ├─► 向量语义检索（Top 30 × N 个改写查询）
  └─► 知识图谱 3 跳扩展
  │
  ▼
候选合并 + 加权重排序 + 同视频去重
  │
  ▼
取 Top K 证据 → 构建 Prompt → LLM 流式生成回答
```

## 清除索引

如需强制重建索引，删除以下两个目录后重启即可：

```bash
rm -rf data/index data/chroma
```

## 后续可继续增强

- 引入 cross-encoder reranker 替代当前规则加权
- 增加对话记忆和多轮改写
- 加入人工标注评测集做召回率对比
- 支持增量索引（新增字幕文件无需全量重建）
