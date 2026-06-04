
# Feral Agent

A proactive, portable AI agent with a **native security sandbox**, built in
TypeScript on [Bun](https://bun.sh). It is the agent layer for Feral, a
local-first AI desktop app (Rust + Tauri v2 + Leptos), and runs the same core on
any transport via an adapter pattern.

## V1 status

Working V1 implements the full request/response cycle with the security sandbox
enforced on every action. Inner thoughts, mood, semantic memory, vector search,
and non-Tauri transports are intentionally deferred to V2.

## Architecture (4 layers)

```
src/
├── core/agent-loop.ts        Layer 1 — reasoning cycle: prompt → infer → tools → loop
├── memory/
│   ├── working.ts            Layer 2 — in-session transcript, auto-compresses
│   └── episodic.ts           Layer 2 — FTS5 searchable history
├── sandbox/                  Layer 3 — SECURITY (built first)
│   ├── audit-log.ts          every action → SQLite audit_log
│   ├── tool-permissions.ts   manifest validation + path/permission enforcement
│   ├── egress-proxy.ts       feralFetch(): whitelist, SSRF guard, rate limit
│   └── inference-router.ts   single LLM choke point: budgets, fallback, usage
├── transports/               Layer 4 — adapters (core is transport-agnostic)
│   ├── interface.ts          Transport contract
│   ├── tauri.ts              stdin/stdout newline-delimited JSON (active)
│   ├── telegram.ts           V2 stub
│   └── whatsapp.ts           V2 stub
├── tools/
│   ├── registry.ts           the gate every tool call passes through
│   └── builtin/{read-file,web-search}.ts
├── db.ts                     centralized bun:sqlite schema/migrations
├── types.ts                  shared contracts
└── index.ts                  wires everything; starts the transport
```

### Security guarantees (non-negotiable)

- No tool runs outside the registry; a tool can only register with a valid,
  consistent manifest and can only exercise permissions it declared.
- No network request bypasses `feralFetch` — loopback/private/link-local hosts
  are blocked, domains are whitelisted per-tool, and requests are rate-limited.
- No LLM call bypasses the inference router — per-conversation and per-day token
  budgets are enforced; a fallback model is used when the primary fails.
- Every action (tool call, inference, network, memory write, block) produces an
  `audit_log` row. Errors are caught and returned as structured messages, never
  crashes. TypeScript strict mode, no `any`.

## Run

```sh
bun install
bun run src/index.ts      # speaks the Tauri stdin/stdout protocol
bun test                  # sandbox guarantee suite (no LLM required)
bun run build             # compile single binary for the Tauri sidecar
```

Requires a local [Ollama](https://ollama.com) by default
(`qwen2.5:7b` at `http://localhost:11434`). The router is the sanctioned
exception to the egress proxy so local inference can reach localhost.

### Configuration (environment variables)

| Variable | Default | Purpose |
| --- | --- | --- |
| `FERAL_DB` | `data/feral.db` | SQLite path (`:memory:` supported) |
| `FERAL_WORKSPACE` | cwd | root `read_file` is allowed to access |
| `FERAL_MODEL` | `qwen2.5:7b` | primary model |
| `FERAL_BASE_URL` | `http://localhost:11434` | primary endpoint |
| `FERAL_PROVIDER` | `ollama` | `ollama` or OpenAI-compatible |
| `FERAL_FALLBACK_MODEL` | — | enables a fallback target when set |
| `FERAL_TRUSTED_BASE_URLS` | configured targets | comma-separated inference-endpoint allowlist |
| `FERAL_INNER_THOUGHTS_ENABLED` | `false` | set `true` to enable the proactive background loop (V2, off by default) |
| `FERAL_THOUGHTS_INTERVAL_MS` | `300000` | inner-thoughts tick interval (only relevant when enabled) |
| `FERAL_BUDGET_CONVERSATION` | `50000` | per-conversation token cap |
| `FERAL_BUDGET_DAY` | `500000` | per-day token cap |
| `FERAL_BUDGET_POLICY` | `compress_and_continue` | or `stop` |

## IPC protocol (Tauri sidecar)

Newline-delimited JSON. Inbound on stdin: `{"type":"message","id","content","sessionId"}`,
`{"type":"ping"}`, `{"type":"shutdown"}`. Outbound on stdout: `chunk`, `done`,
`tool_start`, `tool_done`, `proactive`, `pong`, `error`. stdout carries protocol
traffic only; diagnostics go to stderr.

