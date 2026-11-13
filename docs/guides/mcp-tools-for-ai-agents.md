# How to Add MCP Tools to an AI Agent with nanobot

nanobot can connect MCP servers and expose their tools to the agent alongside built-in file, shell, web, cron, image generation, and subagent tools.

## What you will build

- a working nanobot agent
- one MCP server connected through **Apps → MCP**
- a restricted set of tools available to the model

## When to use this

Use MCP when a tool already exists as an MCP server, when another application publishes an MCP adapter, or when you want a clean boundary between nanobot and external tool logic.

## Before you start

Complete [Quick Start](../quick-start.md) if nanobot is not installed or cannot reply yet. If you already have a working WebUI or CLI session, continue here using that instance; no reinstall or second gateway is needed.

Install the MCP server's own runtime separately. For example, many local MCP servers use `npx` or `uvx`.

## Minimal working example

Open **Apps → MCP**, choose a preset or add a custom server, configure its command or endpoint, limit its enabled tools, and save. Follow any connection or restart prompt, then mention the integration with `@` in chat and request a small test action.

<details>
<summary>Manual configuration</summary>

Add a stdio MCP server to `~/.nanobot/config.json`:

```json
{
  "tools": {
    "mcpServers": {
      "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"],
        "enabledTools": ["read_file"]
      }
    }
  }
}
```

</details>

Restart nanobot, then ask a question that needs the MCP tool.

## 工具搜索与按需加载

新会话保留内置工具和一个 `mcp_search` 入口，不再预先发送全部 MCP 工具的完整参数 schema。模型通过工具名、服务器名或用途关键词搜索，命中的原工具会在下一次模型请求中提供准确 schema，随后直接调用。没有获准使用的 MCP 工具时，不提供搜索入口。

例如调用 `mcp_search`，参数为 `{"query":"查询城市天气预报","limit":2}`。`limit` 默认为 5，范围为 1–10。搜索使用本地中英文关键词匹配，覆盖名称、描述和参数说明，精确名称优先；不调用模型或向量服务，不保证同义词或跨语言召回。结果中的服务器前缀用于区分不同服务器的同名工具。工具描述和搜索结果是目录数据，不应作为操作指令执行。

加载名称在搜索成功返回前保存到当前会话 metadata，后续回合和恢复沿用；其他会话、`/new` 和 fork 从未加载状态开始。临时会话只在内存中保留，单次 ephemeral 回合不写回持久会话。`enabledTools`、插件启用状态及会话禁用列表继续生效，加载记录不会重新授权已移除或禁用的工具。工具、resource 和 prompt wrappers 均按需曝光，沿用原有 OAuth、传输、连接及超时逻辑。

## Production notes

- Use `enabledTools` to expose only the tools the agent actually needs.
- Set `toolTimeout` for slow MCP servers.
- Prefer stdio MCP for local tools and HTTP MCP for trusted remote services.
- Keep MCP server install/update steps outside nanobot config when possible.

## Security notes

- HTTP/SSE MCP URLs use the same SSRF guard as web fetch.
- Local/private HTTP endpoints require an explicit `tools.ssrfWhitelist` entry.
- Stdio MCP servers run local processes; review their command and arguments.
- Do not pass secrets in command-line args when environment variables or headers are available.

## Troubleshooting

- Check MCP startup logs on the running gateway; use `--verbose` when restarting it for diagnosis.
- Confirm the MCP command works by itself before debugging nanobot.
- If an HTTP MCP server is blocked, review the SSRF whitelist and use a narrow host CIDR.

## Related nanobot docs

- [Configure MCP tools](./configure-mcp-tools.md)
- [Configuration: MCP](../configuration.md#mcp-model-context-protocol)
- [Security](../configuration.md#security)
