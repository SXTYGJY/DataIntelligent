# DataIntelligent MCP Tool 架构设计（tool_manager）

> **状态**：设计定稿 v1（T1–T4 已于 2026-08-19 实现并回归通过，见 §9.4）
> **日期**：2026-08-18
> **分支**：dev
> **关联文档**：
> - 《DataIntelligent MCP Server 架构定位与设计基线》（外部基线）
> - `DEV_SPEC.md` §3.2 MCP Server 设计 / §6 阶段 J

---

## 1. 背景与目标

DataIntelligent 定位收敛为：**面向 Claude / Codex / Cursor 等 MCP Host 的通用数据知识 MCP Server**。
Server 提供可发现、可调用、稳定的数据知识能力；LLM 推理、Tool Selection、Orchestration、最终答案生成均由 MCP Host 承担。

本设计聚焦 MCP Server 的 **Tool 层**（阶段 J 的 J2/J3），目标：

1. 以 `BaseTool`（声明式描述符）+ `ToolMetadata`（治理元数据）统一工具定义；
2. 以 `tool_manager.py` 中的 `ToolRegistry` 作为工具注册、发现、prompt 与执行的**唯一边界**；
3. `ToolRegistry.exec()` 保持唯一执行入口，内部通过 executor 完成执行；
4. 保持 `query_knowledge_hub`、`list_collections`、`get_document_summary` 的 MCP 名称与行为完全兼容；
5. 对外暴露工具 prompt（MCP `prompts/list` / `prompts/get`）。

## 2. 设计决策记录（ADR）

| # | 决策点 | 结论 |
|---|--------|------|
| D1 | 执行入口 | `ToolRegistry.exec()` 是唯一执行入口；**不引入独立公开 ToolExecutor** |
| D2 | ToolMetadata | 独立组件；当前精简为 4 个字段，其余字段在文档中登记为 reserved |
| D3 | 渐进式加载 | **不做**；`tools/list` 直接返回全部完整定义 |
| D4 | executor 定位 | executor 是 `exec` 内部的实现模块，不对外导出、不独立公开 |
| D5 | Prompt 暴露 | `list_prompt` / `get_prompt` 映射为 MCP `prompts/list` / `prompts/get`，对外暴露；`prompts/list` 返回**全部工具**，`prompts/get` 以 **user 消息**返回 prompt 文本（MCP SDK 1.x `PromptMessage.role` 仅允许 user/assistant；system 意图以 user 消息实现） |
| D6 | 文件布局 | `base.py` 存放契约（`BaseTool` / `ToolMetadata` / `ToolResult` / `ToolContext`）；`tool_manager.py` 单文件存放 `ToolRegistry` 类 + 模块内 executor；两者同在 `src/mcp_server/tool/` 目录（取代 `tool_registry.py`） |
| D7 | Transport | 协议与 Transport 已解耦（2026-08-21）：stdio（默认）+ Streamable HTTP（可选）双传输落地，见 `src/mcp_server/http_app.py` / `http_server.py`；传输层不感知 Tool 细节，协议层保持单一 `ToolRegistry` 边界 |
| D8 | from_dict | `BaseTool` **不保留** `from_dict`（工具为代码定义，非配置/DB 加载） |
| D9 | structuredContent | `query_knowledge_hub` 的 citations 由 JSON 文本块改为 `ToolResult.structured_content` → `CallToolResult.structuredContent` |

## 3. Tool 领域模型

### 3.1 `ToolMetadata`（治理元数据）

```python
@dataclass
class ToolMetadata:
    """工具治理元数据（精简版）。

    仅为后续扩展预留：tenant_id / class_minor / class_minor_desc /
    software_version / character 已登记为 reserved，未来渐进式加载时再补齐，
    不改变当前字段与序列化结构。
    """
    user_id: str | None = None
    access_level: str | None = None
    class_major: str | None = None        # 一级分类
    class_major_desc: str | None = None   # 一级分类描述

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolMetadata":
        fields = {
            f.name: data[f.name]
            for f in cls.__dataclass_fields__.values()
            if f.name in data
        }
        return cls(**fields)
```

### 3.2 `BaseTool`（声明式工具描述符）

```python
@dataclass
class BaseTool:
    """所有工具的抽象基类：声明式工具定义。

    不包含执行逻辑（执行统一走 ToolRegistry.exec → executor）。
    不含 type / url / method（均已从设计中移除）。
    """
    name: str                                # MCP 工具名，必填（构造时校验）
    input_schema: dict[str, Any]             # JSON Schema，必填
    id: str | None = None
    show_name: str | None = None
    description: str | None = None           # 一句话简介（tools/list 展示）
    prompt: str | None = None                # 给 LLM 的详细调用指引（prompts/get 返回）
    metadata: ToolMetadata | None = None     # 治理元数据（强类型字段）

    def to_mcp_definition(self) -> types.Tool:
        """生成 MCP tools/list 定义（name / description / inputSchema）。"""
        return types.Tool(
            name=self.name,
            description=self.description or "",
            inputSchema=self.input_schema,
        )
```

### 3.3 `ToolRegistration`（注册记录）

```python
@dataclass
class ToolRegistration:
    """一条注册记录：声明式描述符 + 可调用实现。"""
    tool: BaseTool
    implementation: Callable[..., Awaitable[ToolResult]]   # 现有工具实现类的 execute 挂载点
```

> `ToolResult` / `ToolContext` 沿用现有契约（`base.py`）：
> `ToolResult(content, is_error, structured_content, metadata)`，`content` 首项为 `TextContent`。

### 3.4 `PromptInfo`（工具 prompt 数据）

```python
@dataclass
class PromptInfo:
    """工具 prompt 摘要（MCP prompts/list 与 prompts/get 的数据来源）。"""
    name: str                     # 与工具 name 一致
    description: str | None = None
    prompt: str | None = None     # 给 LLM 的详细调用指引（prompts/get 返回）
```

- `list_prompt()` 返回全部工具的 `PromptInfo` 列表；
- `get_prompt(name)` 返回单个工具的 `PromptInfo`；工具不存在或未配置 prompt 时返回 `None`；
- MCP 适配时：`prompts/list` 返回**全部已注册工具**（含未配置 prompt 的工具）；`prompts/get` 将 `prompt` 文本组装为 **user 消息**的 MCP `PromptMessage`（见 §5；MCP SDK 1.x 仅支持 user/assistant 角色）。

## 4. `tool_manager.py` 模块设计

```
tool_manager.py（单文件）
├── class ToolRegistry
│   ├── register(tool, implementation)      # 工具 + 实现注册
│   ├── unregister(name)
│   ├── get(name)
│   ├── list_tools()                        # → MCP tools/list
│   ├── list_prompt()                       # → MCP prompts/list
│   ├── get_prompt(name)                    # → MCP prompts/get
│   └── exec(name, arguments, context)      # 唯一执行入口
└── class _Executor（模块内实现，不导出）
    └── run(name, arguments, context)       # exec 的内部执行实现
```

### 4.1 `ToolRegistry` 接口

```python
class ToolRegistry:
    def register(self, tool: BaseTool, implementation: Callable) -> None:
        """注册工具描述符 + 实现。重名校验：重复 name 抛 ValueError，启动即失败。"""

    def unregister(self, name: str) -> BaseTool:
        """仅受控关闭/测试使用；不存在抛 KeyError。"""

    def get(self, name: str) -> BaseTool | None:
        """按名返回描述符；不存在返回 None。"""

    def list_tools(self) -> list[BaseTool]:
        """按注册顺序返回全部工具描述符（MCP tools/list 唯一来源）。"""

    def list_prompt(self) -> list[PromptInfo]:
        """返回全部工具 prompt 摘要（MCP prompts/list 唯一来源）。"""

    def get_prompt(self, name: str) -> PromptInfo | None:
        """返回单个工具 prompt（MCP prompts/get 唯一来源）。"""

    async def exec(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        context: ToolContext | None = None,
    ) -> types.CallToolResult:
        """唯一执行入口：委托模块内 _Executor 完成执行与归一化。"""
```

### 4.2 `_Executor`（exec 的内部实现）

```python
class _Executor:
    """执行实现：查找 → 参数校验 → 调用实现 → 归一化。不对外公开。"""

    async def run(
        self, name: str, arguments: dict[str, Any] | None, context: ToolContext | None
    ) -> types.CallToolResult:
        # 1. 查注册记录：name → ToolRegistration
        # 2. 参数校验（input_schema：required / 类型 / additionalProperties）
        # 3. 调用 implementation.execute(**arguments)（阻塞 I/O 走 asyncio.to_thread）
        # 4. 得到 ToolResult
        # 5. 归一化：成功 / 业务错误 / 未处理异常脱敏 → CallToolResult
```

### 4.3 执行路径与错误语义

```text
MCP tools/call
  → ProtocolHandler.handle_call_tool(name, arguments)
  → ToolRegistry.exec(name, arguments, context)      # 唯一入口
  → _Executor.run(...)
  → implementation.execute(...) → ToolResult
  → types.CallToolResult
```

| 场景 | 返回 |
|------|------|
| Tool 不存在 | `CallToolResult(isError=True)`，文本 `Tool '<name>' not found` |
| 参数校验失败 | `CallToolResult(isError=True)`，文本为不含堆栈的参数错误 |
| 业务错误（ToolResult.is_error=True） | 原样保留可读错误内容与结构化结果 |
| 未处理异常 | 记录日志，返回 `CallToolResult(isError=True)`，文本 `Internal error while executing '<name>'`，不泄露堆栈 |
| 成功 | `CallToolResult(content, isError=False, structuredContent=...)` |

## 5. ProtocolHandler 映射

`ProtocolHandler` 保持为纯 MCP SDK 适配层，不持有第二份工具字典、不 import 具体工具类。

| MCP 协议 | ProtocolHandler 委托 |
|----------|----------------------|
| `tools/list` | `registry.list_tools()` → `BaseTool.to_mcp_definition()` |
| `tools/call` | `await registry.exec(name, arguments)` |
| `prompts/list` | `registry.list_prompt()` → 全部工具的 MCP `Prompt` 列表 |
| `prompts/get` | `registry.get_prompt(name)` → 以 user 消息组装 MCP `GetPromptResult`（SDK 1.x 角色限制） |

## 6. 现有工具迁移方案

| 现有工具 | 描述符来源 | 实现来源（注册为 implementation） |
|----------|------------|-----------------------------------|
| `query_knowledge_hub` | `TOOL_NAME` / `TOOL_DESCRIPTION` / `TOOL_INPUT_SCHEMA` → `BaseTool` | `QueryKnowledgeHubTool.execute`（保留多模态 + citations 组装） |
| `list_collections` | 同上 | `ListCollectionsTool.execute` |
| `get_document_summary` | 同上 | `GetDocumentSummaryTool.execute` |

迁移步骤：
1. 为 3 个工具各建 `BaseTool` 描述符（`prompt` 先放**占位文案**，`metadata` 先取**默认值**）；
2. 保留现有实现类，注册为 `ToolRegistration.implementation`；
3. 移除 `tool_registry.py` 中的 `CallbackTool` 适配与模块级 `register_tool()` 死代码；
4. 删除 `CallbackTool` 迁移适配器（目标态不再需要）。
5. `query_knowledge_hub` 的 citations 由 JSON 文本块改为走 `ToolResult.structured_content` → `CallToolResult.structuredContent`（按 D9 决策）。

兼容性红线：MCP 工具名称、输入参数与返回内容（含引用与图像）保持不变；`structuredContent` 承载方式按 D9 决策调整。

## 7. 预留 Tool（ToolPlan）

以下工具**只登记、不注册、不暴露**（不作为 `ToolRegistration` 加入 Registry，不出现在 `tools/list` / `prompts/list`）：

| 预留名称 | 目标能力 | 说明 |
|----------|----------|------|
| `search_data_assets` | 数据资产搜索 | Schema/权限 TBD，待 Metadata 域数据源 |
| `get_data_asset` | 资产详情 | 同上 |
| `query_data` | 自然语言数据查询 | Schema/安全边界 TBD，待 Data 域 |
| `analyze_data` | 数据分析 | 同上 |
| `explain_query` | 查询解释 | 同上 |

预留记录可先以 `ToolMetadata` + `BaseTool` 描述符形态存在于配置/文档清单，后续实现时直接走 `register()` 上线。

## 8. 兼容性与约束

- MCP 已发布 Tool 名称与行为不因本次重构改变；
- `exec` 是唯一执行入口，协议层不直接调用 executor；
- `stdout` 仅输出 MCP 消息，日志走 stderr（沿用现状）；
- 阻塞 I/O（Chroma / Embedding / BM25）继续通过 `asyncio.to_thread` 执行，避免阻塞 stdio 事件循环。

## 9. 实施排期

### 9.1 任务分解

| 任务 | 内容 | 预估 | 前置 |
|------|------|------|------|
| T0 | 设计定稿 + 文档收口（本文档 + DEV_SPEC 阶段 J 更新） | 0.5d | - |
| T1 | `base.py` 改造：`ToolMetadata` + `BaseTool` 描述符（from_dict / to_mcp_definition）；`tool_manager.py` 新建 `ToolRegistry` + `_Executor`（注册 / 列表 / prompt / exec / 参数校验 / 异常归一化） | 1.5d | T0 |
| T2 | 现有 3 工具迁移：生成描述符、实现注册为 implementation、删除 CallbackTool 与死代码 | 1.5d | T1 |
| T3 | ProtocolHandler 扩展：`prompts/list`、`prompts/get` 映射；`tools/list`、`tools/call` 改走 tool_manager | 1d | T2 |
| T4 | 测试更新与回归：单测（registry / base / 3 工具 / protocol）、集成（test_mcp_server）、E2E（test_mcp_client 增加 prompts 用例）、ruff / mypy | 1.5d | T3 |
| T5 | 文档收口：DEV_SPEC 阶段 J 进度更新、README 补充（如需） | 0.5d | T4 |

**总计：约 6.5 人日（不含用户评审等待）**

### 9.2 里程碑

| 里程碑 | 内容 | 验收标准 |
|--------|------|----------|
| M-J2a | tool_manager 基建可用（T1） | `ToolRegistry` 注册/列表/prompt/exec 契约单测全绿 |
| M-J2b | 3 工具迁移完成（T2） | 3 工具均以 BaseTool 描述符 + implementation 注册，无 CallbackTool / 死代码 |
| M-J3 | 回归 + prompts 暴露（T3+T4） | 全量单测 + 集成 + E2E 全绿；`prompts/list`、`prompts/get` 线级验证通过 |

### 9.4 实施状态（2026-08-19）

- **T1** ✅ `base.py`（`ToolMetadata` + `BaseTool` 描述符）+ `tool_manager.py`（`ToolRegistry` + 内部 `_Executor` + `PromptInfo` / `ToolRegistration`）；`tool_registry.py` 已删除。
- **T2** ✅ 三个既有 Tool 迁移为描述符 + implementation，`execute` 统一返回 `ToolResult`；`CallbackTool` 与各 Tool 的 `register_tool` 死代码已删除；`query_knowledge_hub` citations 已按 D9 移入 `structuredContent`。
- **T3** ✅ `ProtocolHandler` 增加 `prompts/list` / `prompts/get` 委托；MCP Server 注册对应 handler（capabilities 自动声明 prompts）。
- **T4** ✅ 单测（registry / base / 3 工具 / protocol）+ MCP 集成 + E2E（含 prompts 线级用例）回归通过；ruff 全绿。
- **T5** ⏳ 本文档与 `DEV_SPEC.md` 阶段 J 已更新；README 补充见下。

**偏差记录**：`prompts/get` 的返回角色由「system 消息」调整为「user 消息」——MCP SDK 1.x（mcp==1.29.0）的 `PromptMessage.role` 仅接受 `user`/`assistant`，system 意图以 user 消息承载。

**遗留（非本轮）**：`get_document_summary.py` / `query_knowledge_hub.py` 的 3 处既有 mypy 类型问题；numpy stub 与 mypy `python_version=3.11` 的环境兼容问题（默认 mypy 配置暂不可跑）。

### 9.3 风险与依赖

- 依赖：`PromptInfo` 结构已定稿（§3.4）；3 个工具的 prompt 文案先占位、`metadata` 先默认值；Transport 解耦另立大任务，不影响本排期；
- 风险：3 工具内部依赖 Chroma/Embedding 等重组件，迁移时保持 `asyncio.to_thread` 与懒加载语义不变；
- 已推进：Transport 解耦（stdio + Streamable HTTP，见 `src/mcp_server/http_app.py` / `http_server.py`）与 KnowledgeService 层（见 `docs/design/service_layer_design.md`）已落地；
- 范围外：MySQL 数据源、依赖清理（LangChain 去耦）仍另立任务，不阻塞本排期。

