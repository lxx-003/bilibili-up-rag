# Bilibili UP 字幕 RAG

一个可本地运行的面试项目 MVP：输入问题，系统会从 `data/subs` 下全部 `.srt` 字幕中做检索，并返回带 B 站空降链接的证据片段。

当前版本已切换为阿里云百炼 OpenAI 兼容模式：

- 向量模型：`text-embedding-v4`
- 文本生成模型：`qwen3.6-flash`
- `base_url`：`https://dashscope.aliyuncs.com/compatible-mode/v1`

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

2. 配置百炼环境变量

```bash
cp .env.example .env
export DASHSCOPE_API_KEY=你的百炼Key
export DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export DASHSCOPE_EMBEDDING_MODEL=text-embedding-v4
export DASHSCOPE_CHAT_MODEL=qwen3.6-flash
```

3. 启动服务

```bash
python -m app.main
```

4. 打开浏览器

```text
http://127.0.0.1:8000
```

第一次启动会自动构建索引。
当前向量检索使用阿里云百炼 `text-embedding-v4` 生成向量，向量会持久化在 `data/chroma`。
如果你更换了 embedding 模型、维度或 `base_url`，项目会自动重建索引。

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

## 清除索引

如需强制重建索引，删除以下两个目录后重启即可：

```bash
rm -rf data/index data/chroma
```

## 后续可继续增强

- 引入真正的 cross-encoder reranker
- 增加对话记忆和多轮改写
- 加入人工标注评测集做召回率对比
