# MCP Tool Registry 实施与验收报告

## 实施 TODO

- [x] 新建 `src/mcp_server/tool/`，定义不依赖具体业务 Tool 的基础契约。
- [x] 实现统一 `ToolRegistry`，由其提供注册、查询、列表和 `exec`。
- [x] 将 MCP 协议层改为只委托 Registry，不再维护 handler 字典。
- [x] 以 `BaseTool` 适配器纳入现有三个已发布 Tool，保持 MCP 名称与输出兼容。
- [x] 新增 Tool Registry 单元测试，并更新协议层测试。
- [x] 执行静态验证；已尝试执行目标测试，结果见下文。

## 设计约束

- 仅 `ToolRegistry` 持有可调用 Tool；未实现的规划 Tool 不注册、不暴露。
- `BaseTool.execute(arguments, context)` 是所有已注册 Tool 的统一调用形态。
- 为平滑迁移，当前三个业务实现保持原有 service/handler 代码；Registry 以 `CallbackTool` 适配它们。后续新 Tool 直接继承 `BaseTool`，不再新增模块级 `register_tool`。
- `ToolRegistry.exec()` 统一负责 Tool 不存在、基础 JSON Schema 参数错误、业务错误和未处理异常的 MCP 响应。

## 测试结果

### 已通过

| 检查 | 命令 | 结果 |
|------|------|------|
| Python 语法编译 | `PYTHONPYCACHEPREFIX=/private/tmp/dataintelligent-pycache python3 -m compileall -q src/mcp_server/tool src/mcp_server/protocol_handler.py tests/unit/test_tool_registry.py tests/unit/test_protocol_handler.py` | 通过 |
| 变更格式 | `git diff --check` | 通过 |

### 待具备依赖后执行

目标命令：

```bash
uv run pytest -q \
  tests/unit/test_tool_registry.py \
  tests/unit/test_protocol_handler.py \
  tests/unit/test_list_collections.py \
  tests/unit/test_get_document_summary.py
```

本机系统 Python 未安装 `pytest` 和 `mcp`。已尝试使用受控联网安装最小依赖，但安装过程未成功写入虚拟环境（`mcp` 仍不可导入），因此不能把 pytest 标记为通过。该阻断不影响已完成的静态编译；验收环境安装项目依赖后可直接运行上述命令。

## 新增测试覆盖

- `tests/unit/test_tool_registry.py`
  - 注册、重名拒绝、注销、MCP Tool 定义生成；
  - `exec` 成功路径、缺失参数、类型错误、额外参数、未知 Tool；
  - 业务可预期错误透传、未处理异常脱敏；
  - 三个既有公开 Tool 均被包装为 `BaseTool` 并注册。
- `tests/unit/test_protocol_handler.py`
  - `tools/list` 与 `tools/call` 均委托 Registry；
  - Server 工厂保留并暴露注入的 Registry；
  - JSON-RPC 错误码常量回归。
