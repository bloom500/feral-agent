/**
 * Feral Agent — entry point.
 *
 * Wires the four layers together and starts the selected transport:
 *   Sandbox (audit → egress → inference) → Memory → Tools → Agent core → Transport
 *
 * Security is constructed first: the audit log, egress proxy, and inference
 * router exist before any tool is registered or any message is handled.
 */

import { resolve } from "node:path";
import { openDatabase } from "./db.ts";
import { AuditLog } from "./sandbox/audit-log.ts";
import { EgressProxy } from "./sandbox/egress-proxy.ts";
import { InferenceRouter } from "./sandbox/inference-router.ts";
import { EpisodicMemory } from "./memory/episodic.ts";
import { SemanticMemory } from "./memory/semantic.ts";
import { RecallEngine } from "./memory/recall.ts";
import { ToolRegistry } from "./tools/registry.ts";
import { createReadFileTool } from "./tools/builtin/read-file.ts";
import { createWebSearchTool } from "./tools/builtin/web-search.ts";
import { AgentLoop } from "./core/agent-loop.ts";
import { MoodEngine } from "./core/mood.ts";
import { InnerThoughtsLoop } from "./core/inner-thoughts.ts";
import { TauriTransport } from "./transports/tauri.ts";
import type { InferenceConfig, Transport } from "./types.ts";

interface AppConfig {
  transport: "tauri";
  dbPath: string;
  workspace: string;
  inference: InferenceConfig;
}

function loadConfig(): AppConfig {
  const env = process.env;
  const workspace = resolve(env.FERAL_WORKSPACE ?? process.cwd());

  // ":memory:" is a SQLite sentinel and must not be path-resolved.
  const dbEnv = env.FERAL_DB ?? "data/feral.db";
  const dbPath = dbEnv === ":memory:" ? ":memory:" : resolve(dbEnv);

  return {
    transport: "tauri", // only transport wired in V1
    dbPath,
    workspace,
    inference: {
      primary: {
        provider: env.FERAL_PROVIDER ?? "ollama",
        model: env.FERAL_MODEL ?? "qwen2.5:7b",
        baseUrl: env.FERAL_BASE_URL ?? "http://localhost:11434",
      },
      ...(env.FERAL_FALLBACK_MODEL
        ? {
            fallback: {
              provider: env.FERAL_FALLBACK_PROVIDER ?? "ollama",
              model: env.FERAL_FALLBACK_MODEL,
              baseUrl: env.FERAL_FALLBACK_BASE_URL ?? "http://localhost:11434",
            },
          }
        : {}),
      tokenBudget: {
        perConversation: Number(env.FERAL_BUDGET_CONVERSATION ?? 50_000),
        perDay: Number(env.FERAL_BUDGET_DAY ?? 500_000),
        onExhausted:
          env.FERAL_BUDGET_POLICY === "stop" ? "stop" : "compress_and_continue",
      },
      // Comma-separated allowlist of inference endpoints. Omitted → defaults to
      // exactly the configured primary/fallback targets.
      ...(env.FERAL_TRUSTED_BASE_URLS
        ? {
            trustedBaseUrls: env.FERAL_TRUSTED_BASE_URLS.split(",")
              .map((u) => u.trim())
              .filter(Boolean),
          }
        : {}),
    },
  };
}

function buildTransport(kind: AppConfig["transport"]): Transport {
  switch (kind) {
    case "tauri":
      return new TauriTransport();
    default:
      // Exhaustive: V2 transports are stubbed and not selectable yet.
      throw new Error(`unsupported transport: ${kind}`);
  }
}

function main(): void {
  const config = loadConfig();
  const db = openDatabase(config.dbPath);

  // --- Layer 3: Sandbox (built first) ---
  const audit = new AuditLog(db.raw);
  const egress = new EgressProxy(audit.logger);
  const router = new InferenceRouter(config.inference, audit.logger, db.raw);

  // --- Layer 2: Memory ---
  const episodic = new EpisodicMemory(db.raw, audit.logger);
  const semantic = new SemanticMemory(db.raw, audit.logger);
  const recall = new RecallEngine(episodic, semantic);

  // --- Tools (each gated by the sandbox) ---
  const registry = new ToolRegistry(egress, audit);
  registry.register(createReadFileTool([config.workspace]));
  registry.register(createWebSearchTool());

  // --- Mood engine ---
  const mood = new MoodEngine();

  // --- Layer 1: Agent core ---
  const agent = new AgentLoop(
    router, registry, episodic,
    { onBudgetExhausted: config.inference.tokenBudget.onExhausted },
    recall,
  );

  // --- Inner thoughts loop (proactive background) ---
  // Disabled by default in V1. The dental-pilot deliverable does not require it
  // and the loop contends for inference budget with real user requests.
  // Enable with: FERAL_INNER_THOUGHTS_ENABLED=true
  const innerThoughtsEnabled = process.env.FERAL_INNER_THOUGHTS_ENABLED === "true";
  const innerThoughts = new InnerThoughtsLoop(router, episodic, mood, db.raw, {
    intervalMs: Number(process.env.FERAL_THOUGHTS_INTERVAL_MS ?? 5 * 60 * 1000),
  });

  // --- Layer 4: Transport ---
  const transport = buildTransport(config.transport);

  transport.onMessage(async (msg) => {
    switch (msg.type) {
      case "ping":
        transport.send({ type: "pong" });
        break;

      case "shutdown":
        log(`shutdown requested`);
        db.close();
        process.exit(0);
        break;

      case "message": {
        const id = msg.id ?? crypto.randomUUID();
        const sessionId = msg.sessionId ?? "default";
        const content = msg.content ?? "";
        if (!content.trim()) {
          transport.send({
            type: "error",
            id,
            message: "empty message content",
          });
          return;
        }
        mood.applyEvent("message_received");
        await agent.handle(sessionId, content, id, (event) => {
          transport.send(event);
          // Update mood based on what the agent loop emits.
          if (event.type === "done")       mood.applyEvent("message_answered");
          if (event.type === "tool_done") {
            const r = event.result as { ok?: boolean } | null;
            mood.applyEvent(r?.ok === false ? "tool_error" : "tool_success");
          }
          if (event.type === "error")      mood.applyEvent("inference_error");
        });
        break;
      }
    }
  });

  transport.onReady(() => {
    log(
      `ready — transport=${config.transport} model=${config.inference.primary.model} ` +
        `workspace=${config.workspace}`,
    );
    if (innerThoughtsEnabled) {
      innerThoughts.setEmit((event) => transport.send(event));
      innerThoughts.start();
      log("inner-thoughts loop enabled");
    }
  });

  // Persist final audit state on unexpected termination.
  const shutdown = () => {
    if (innerThoughtsEnabled) innerThoughts.stop();
    try {
      db.close();
    } finally {
      process.exit(0);
    }
  };
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);

  transport.start();
}

/** Diagnostics go to stderr; stdout is reserved for the transport protocol. */
function log(message: string): void {
  process.stderr.write(`[feral] ${message}\n`);
}

try {
  main();
} catch (err) {
  // Startup misconfiguration (e.g. a target outside trustedBaseUrls) should
  // fail fast with a clear, single-line reason rather than a raw stack trace.
  log(`fatal: failed to start — ${err instanceof Error ? err.message : String(err)}`);
  process.exit(1);
}
