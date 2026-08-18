# Modular RAG MCP Server

> 一个可插拔、可观测的模块化 RAG（检索增强生成）服务框架，通过 MCP（Model Context Protocol）协议对外暴露工具接口，支持 Copilot / Claude 等 AI 助手直接调用。


本项目将 RAG ——**检索（Hybrid Search + Rerank）**、**多模态视觉处理（Image Captioning）**、**RAG 评估（Ragas + Custom）**、**生成（LLM Response）**——以及当下热门的应用协议 **MCP（Model Context Protocol）** 串联为一个完整的、可运行的工程项目。


### 环境配置（uv 快速开始）

本项目使用 [uv](https://docs.astral.sh/uv/) 统一管理 Python 版本、虚拟环境与依赖，支持 Python `>=3.11,<3.14`（推荐 3.13，见 `.python-version`）。

#### 1. 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# 或 pip install uv
```

#### 2. 一键同步环境（从 `uv sync` 开始）

在项目根目录执行：

```bash
uv sync
```

该命令会：

- 按 `.python-version` 自动下载 / 复用 Python 解释器；
- 根据 `pyproject.toml` 解析依赖，并生成 / 更新 `uv.lock` 锁文件；
- 创建 `.venv` 虚拟环境，安装全部运行时依赖与 `dev` 开发组依赖（pytest / pytest-cov / ruff / mypy 等），可直接开始开发与测试。

可选扩展（Cross-Encoder 重排，需要下载本地模型）：

```bash
uv sync --extra rerank
```

#### 3. 激活环境与常用命令

```bash
source .venv/bin/activate        # 激活虚拟环境
python -V                        # 应输出 3.13.x
uv run python scripts/ingest.py  # 运行摄取脚本
uv run pytest                    # 运行测试
uv run ruff check src            # 代码检查
uv run mypy src                  # 类型检查
```

> 不激活虚拟环境时，所有命令都可通过 `uv run <command>` 前缀直接执行。

#### 4. 配置文件与验证

复制凭据模板，并按需编辑 `config/settings.yaml`（LLM / Embedding 的 Provider、API Key 等）：

```bash
cp config/test_credentials.yaml.example config/test_credentials.yaml
```

验证环境与配置是否就绪：

```bash
uv run python -c "import chromadb, mcp, yaml; print('core deps OK')"
uv run python -c "from src.core.settings import load_settings; load_settings(); print('config OK')"
```

#### 平台与版本说明

- 项目已在 **macOS x86_64 + Python 3.13** 下验证通过。`chromadb` 的传递依赖 `onnxruntime>=1.24` 不再提供 macOS x86_64 wheel，`pyproject.toml` 已通过平台标记将 `onnxruntime` 限制为 `<1.24`，保证 `uv sync` 可直接成功；
- `ragas` 需要 langchain 0.3.x 线（`langchain-community 0.4+` 移除了其依赖的 `chat_models.vertexai`），`pyproject.toml` 通过 `[tool.uv] constraint-dependencies` 固定 langchain 系列版本；
- 代码库使用 MCP SDK 1.x API（`types.Tool.inputSchema` 等），因此 `mcp` 锁定为 `>=1.0.0,<2.0.0`。


### 核心能力一览

| 模块 | 能力 | 说明 |
|------|------|------|
| **Ingestion Pipeline** | PDF → Markdown → Chunk → Transform → Embedding → Upsert | 全链路数据摄取，支持多模态图片描述（Image Captioning） |
| **Hybrid Search** | Dense (向量) + Sparse (BM25) + RRF Fusion + Rerank | 粗排召回 + 精排重排的两段式检索架构 |
| **MCP Server** | 标准 MCP 协议暴露 Tools + Prompts | `query_knowledge_hub`、`list_collections`、`get_document_summary`（`tools/list` / `tools/call` / `prompts/list` / `prompts/get`） |
| **Dashboard** | Streamlit 六页面管理平台 | 系统总览 / 数据浏览 / Ingestion 管理 / 摄取追踪 / 查询追踪 / 评估面板 |
| **Evaluation** | Ragas + Custom 评估体系 | 支持 golden test set 回归测试，拒绝"凭感觉"调优 |
| **Observability** | 全链路白盒化追踪 | Ingestion 与 Query 两条链路的每一个中间状态透明可见 |
| **Skill 驱动全流程** | 从编写到测试、打包、配置一键完成 | auto-coder / qa-tester / package / setup 等 Skill 覆盖完整开发生命周期（笔记中每个 Skill 的使用和设计思路均有讲解，请参考配套视频） |

### 技术亮点

** 全链路可插拔架构**：LLM / Embedding / Reranker / Splitter / VectorStore / Evaluator 每一个核心环节均定义了抽象接口，支持"乐高积木式"替换，通过配置文件一键切换后端，零代码修改。

** 混合检索 + 重排**：BM25 稀疏检索解决专有名词精确匹配 + Dense Embedding 解决同义词语义匹配，RRF 融合后可选 Cross-Encoder / LLM Rerank 精排，平衡查全率与查准率。

** 多模态图像处理**：采用 Image-to-Text 策略，利用 Vision LLM 自动生成图片描述并缝合进 Chunk，复用纯文本 RAG 链路即可实现"搜文字出图"。

** MCP 生态集成**：遵循 Model Context Protocol 标准，可直接对接 GitHub Copilot、Claude Desktop 等 MCP Client，零前端开发，一次开发处处可用。

** 可视化管理 + 自动化评估**：Streamlit Dashboard 提供完整的数据管理与链路追踪能力，集成 Ragas 等评估框架，建立基于数据的迭代反馈回路。

** 三层测试体系**：Unit / Integration / E2E 分层测试，覆盖独立模块逻辑、模块间交互、完整链路（MCP Client / Dashboard）。

** Skill 驱动全流程**：内置 auto-coder（自动编码）、qa-tester（自动测试）、package（清理打包）、setup（一键配置）等 Agent Skill，覆盖从代码编写到测试、打包、部署的完整开发生命周期。



一键配置（Setup Skill）

本项目提供了 **Setup Skill** 一键完成所有环境配置，包括：Provider 选择 → API Key 配置 → 依赖安装 → 配置文件生成 → Dashboard 启动。

在 VS Code 中打开项目，通过 Copilot / Claude 对话框输入：

```
setup
```

Agent 会自动引导你完成全部配置流程。



