# DataIntelligent 业务服务层设计（KnowledgeService）

> **状态**：设计定稿 v1（2026-08-22 实施）
> **目的**：落实"MCP 侧瘦身"——业务编排与存储访问从 `src/mcp_server/tool/` 下沉到 `src/core/service/`，MCP Tool 只做协议适配。
> **关联文档**：`docs/design/tool_manager_design.md`、`DEV_SPEC.md` §3.2.7 / 阶段 J-J12

---

## 1. 背景与目标

此前 MCP Tool 类内部裹了完整业务实现：

- `query_knowledge_hub`：混合检索（Dense+Sparse+RRF）+ 重排 + 响应构建 + 查询追踪；
- `list_collections` / `get_document_summary`：直接管理 ChromaDB client、集合枚举、文档摘要提取；
- `server.py`：`_preload_heavy_imports` 直接持有业务模块清单（chromadb、query_engine、embedding、vector_store…）。

问题：

1. **MCP 侧臃肿**：协议适配与业务编排混在一个类里，职责不清；
2. **复用困难**：Dashboard / 脚本想复用检索或集合能力只能 import MCP Tool；
3. **演进受限**：新增业务能力（J5–J8）被迫以"再写一个 Tool"的方式实现。

目标分层：

| 层 | 位置 | 职责 |
|----|------|------|
| 协议层 | `src/mcp_server/`（server / http_app / protocol_handler） | 传输 + JSON-RPC + 能力协商 |
| 适配层 | `src/mcp_server/tool/*.py` | `BaseTool` 描述符 + 薄 `execute()` |
| 业务服务层 | `src/core/service/*.py` | 检索编排、存储访问、追踪、格式化、重依赖预载 |

## 2. 服务层模块

| 模块 | 类 / 函数 | 职责 |
|------|-----------|------|
| `query.py` | `KnowledgeQueryService` / `QueryKnowledgeHubConfig` | 混合检索 + 重排 + `MCPToolResponse` 构建 + 查询追踪（`async query(...)`） |
| `collections.py` | `CollectionService` / `CollectionInfo` / `ListCollectionsConfig` / `format_collections_response()` | ChromaDB 集合枚举、统计、文本格式化 |
| `document_summary.py` | `DocumentSummaryService` / `DocumentSummary` / `DocumentNotFoundError` / `GetDocumentSummaryConfig` / `format_document_summary()` / `format_document_error()` | 文档块发现、标题/摘要/标签提取、格式化 |
| `preload.py` | `preload_heavy_dependencies()` | 主线程预载重依赖，避免 `to_thread` 导入锁死锁 |

### 2.1 设计约束

- 服务层**不 import** `mcp` 或 `src.mcp_server.*`（`MCPToolResponse` 例外：它是 `src/core/response` 的业务响应模型，非协议类型）；
- 服务方法保持**同步/阻塞**（业务视角），异步边界（`asyncio.to_thread`）由适配层负责；`KnowledgeQueryService.query` 因内部编排异步追踪而保留 `async`；
- 服务层返回**业务对象**（`MCPToolResponse` / `CollectionInfo` / `DocumentSummary`），不做 `ToolResult` 映射——那是适配层职责。

## 3. 适配层（MCP Tool）瘦身后的形态

```python
class QueryKnowledgeHubTool:
    def __init__(self, settings=None, config=None, service: KnowledgeQueryService | None = None):
        ...
    @property
    def service(self) -> KnowledgeQueryService:
        # 惰性构建，支持测试注入
    async def execute(self, query, top_k=None, collection=None) -> ToolResult:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")   # 输入契约校验
        response = await self.service.query(query=query, top_k=top_k, collection=collection)
        return self._to_tool_result(response)          # 协议映射（D9 structuredContent）
```

三个 Tool 均保留：模块级 `TOOL_NAME / TOOL_DESCRIPTION / TOOL_INPUT_SCHEMA / TOOL_PROMPT`、`execute()`、`get_tool_instance()`（query 工具）——MCP 对外契约完全不变。

## 4. server / http 启动侧的变化

- `server.py` 的 `_preload_heavy_imports()` 改为薄委托：`preload_heavy_dependencies()`（业务层持有模块清单）；
- stdio / Streamable HTTP 两传输共用同一 `ToolRegistry`，行为一致。

## 5. 测试策略

- **业务测试**随代码迁移：`test_service_collections.py`、`test_service_document_summary.py`（原 tool 业务单测改名迁入）；
- **适配测试**：`test_tool_adapters.py`（输入校验、委托服务、`ToolResult` 映射、契约常量）；
- 协议/注册/传输测试（`test_tool_registry`、`test_protocol_handler`、MCP 集成/E2E）保持不变。

## 6. 兼容性红线

- 三个已发布 Tool 的 MCP 名称、输入 Schema、返回内容（含图片与 `structuredContent`）**不变**；
- stdio 默认启动行为不变（`python main.py` / `mcp-server`）；
- 全量单测 / 集成 / E2E 回归通过。

## 7. 范围外（后续任务）

- J4 Knowledge Engine 重构与 `search_knowledge` 兼容策略；
- J5 MySQL DataSource、J6 数据资产探索、J7 自然语言查询（新业务一律先落 `src/core/service/` 再注册薄 Tool）；
- 远程工具注册表（上传案例 ToolsManage 模式）仅作参考设计，未实施。
