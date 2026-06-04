/**
 * Agent loop — the core reasoning cycle.
 *
 *   build prompt → inference (via router) → parse tool calls
 *     → if tool calls: execute each through the sandboxed registry, feed
 *       results back, loop
 *     → else: final answer, persist, done
 *
 * Constraints honored here:
 *   - every LLM call goes through the InferenceRouter (never a provider direct)
 *   - every tool call goes through the ToolRegistry (the sandbox choke point)
 *   - all errors are caught and surfaced as structured events, never crashes
 *   - budget exhaustion triggers compression or a clean stop, per config
 */

import type { InferenceRouter } from "../sandbox/inference-router.ts";
import {
  BudgetExhaustedError,
  InferenceError,
} from "../sandbox/inference-router.ts";
import type { ToolRegistry } from "../tools/registry.ts";
import type { EpisodicMemory } from "../memory/episodic.ts";
import type { RecallEngine } from "../memory/recall.ts";
import { WorkingMemory } from "../memory/working.ts";
import type {
  ChatMessage,
  InferenceConfig,
  OutboundEvent,
  ParsedResponse,
  ParsedToolCall,
} from "../types.ts";

export interface AgentLoopConfig {
  /** Hard cap on tool-call/inference cycles per user message. */
  maxIterations: number;
  /** Soft token cap passed to each completion. */
  maxTokensPerCall: number;
  /** Behavior when a budget is exhausted (mirrors InferenceConfig). */
  onBudgetExhausted: InferenceConfig["tokenBudget"]["onExhausted"];
}

const DEFAULT_CONFIG: AgentLoopConfig = {
  maxIterations: 6,
  maxTokensPerCall: 1024,
  onBudgetExhausted: "compress_and_continue",
};

export type EventSink = (event: OutboundEvent) => void;

export class AgentLoop {
  readonly #router: InferenceRouter;
  readonly #registry: ToolRegistry;
  readonly #episodic: EpisodicMemory;
  readonly #recall: RecallEngine | null;
  readonly #config: AgentLoopConfig;
  readonly #systemPrompt: string;
  /** One working-memory transcript per session, retained across messages. */
  readonly #sessions = new Map<string, WorkingMemory>();

  constructor(
    router: InferenceRouter,
    registry: ToolRegistry,
    episodic: EpisodicMemory,
    config: Partial<AgentLoopConfig> = {},
    recall: RecallEngine | null = null,
  ) {
    this.#router = router;
    this.#registry = registry;
    this.#episodic = episodic;
    this.#recall = recall;
    this.#config = { ...DEFAULT_CONFIG, ...config };
    this.#systemPrompt = buildSystemPrompt(registry);
  }

  /**
   * Process one user message end-to-end. Emits chunk/tool/done/error events to
   * the sink and returns the final assistant text. Never throws.
   */
  async handle(
    sessionId: string,
    userText: string,
    messageId: string,
    emit: EventSink,
  ): Promise<string> {
    const memory = this.#memoryFor(sessionId);

    // Inject relevant past context before the user message lands in the prompt.
    // This runs synchronously (no I/O — pure DB reads) and never throws.
    if (this.#recall) {
      const result = this.#recall.recall(userText, sessionId);
      memory.setMemoryContext(result.context);
    }

    memory.addUser(userText);
    this.#episodic.record(sessionId, "user", userText);

    try {
      const final = await this.#run(sessionId, memory, messageId, emit);
      memory.addAssistant(final);
      this.#episodic.record(sessionId, "assistant", final);
      emit({ type: "done", id: messageId, content: final });
      return final;
    } catch (err) {
      const message = errorMessage(err);
      emit({ type: "error", id: messageId, message });
      return message;
    }
  }

  async #run(
    sessionId: string,
    memory: WorkingMemory,
    messageId: string,
    emit: EventSink,
  ): Promise<string> {
    for (let i = 0; i < this.#config.maxIterations; i++) {
      const completion = await this.#complete(sessionId, memory);
      const parsed = parseResponse(completion);

      if (parsed.toolCalls.length === 0) {
        // No tool calls → this is the final answer.
        if (parsed.text.trim()) {
          emit({ type: "chunk", id: messageId, content: parsed.text });
        }
        return parsed.text.trim() || "(no response)";
      }

      // Record the assistant's tool-calling turn so the model sees its own
      // decisions on the next pass.
      memory.addAssistant(completion);

      for (const call of parsed.toolCalls) {
        emit({ type: "tool_start", tool: call.name, args: call.args });
        const result = await this.#registry.call(call.name, call.args, sessionId);
        emit({ type: "tool_done", tool: call.name, result });

        const rendered = result.ok
          ? result.content
          : `ERROR: ${result.content}`;
        memory.addToolResult(call.name, rendered);
        this.#episodic.record(sessionId, "tool", `${call.name}: ${rendered}`);
      }
    }

    // Exhausted the iteration budget without a final answer.
    return (
      "I reached the maximum number of reasoning steps before finishing. " +
      "Please narrow the request or try again."
    );
  }

  /** One completion with budget handling (compress-and-retry or stop). */
  async #complete(
    sessionId: string,
    memory: WorkingMemory,
  ): Promise<string> {
    try {
      const res = await this.#router.complete({
        sessionId,
        messages: memory.render(),
        maxTokens: this.#config.maxTokensPerCall,
      });
      return res.content;
    } catch (err) {
      if (
        err instanceof BudgetExhaustedError &&
        this.#config.onBudgetExhausted === "compress_and_continue"
      ) {
        const compressed = await memory.maybeCompress((msgs) =>
          this.#summarize(sessionId, msgs),
        );
        if (compressed) {
          const res = await this.#router.complete({
            sessionId,
            messages: memory.render(),
            maxTokens: this.#config.maxTokensPerCall,
          });
          return res.content;
        }
      }
      throw err;
    }
  }

  /** Summarize older turns into a compact note (used by working-memory). */
  async #summarize(sessionId: string, msgs: ChatMessage[]): Promise<string> {
    const transcript = msgs
      .map((m) => `${m.role}: ${m.content}`)
      .join("\n")
      .slice(0, 6_000);
    const res = await this.#router.complete({
      sessionId,
      messages: [
        {
          role: "system",
          content:
            "Summarize the following conversation excerpt in 3-4 sentences, " +
            "preserving facts, decisions, and open questions.",
        },
        { role: "user", content: transcript },
      ],
      maxTokens: 256,
    });
    return res.content.trim();
  }

  #memoryFor(sessionId: string): WorkingMemory {
    let memory = this.#sessions.get(sessionId);
    if (!memory) {
      memory = new WorkingMemory(this.#systemPrompt);
      this.#sessions.set(sessionId, memory);
    }
    return memory;
  }
}

// ---------------------------------------------------------------------------
// Prompt construction & response parsing
// ---------------------------------------------------------------------------

function buildSystemPrompt(registry: ToolRegistry): string {
  return [
    "You are Feral, a proactive, helpful local AI agent.",
    "You run inside a security sandbox; you can only act through declared tools.",
    "",
    "Available tools:",
    registry.describe() || "(none)",
    "",
    "To call a tool, output a fenced code block tagged `tool` containing a JSON",
    'object: {"name": "<tool>", "args": { ... }}. You may emit several such',
    "blocks to call multiple tools. Example:",
    "```tool",
    '{"name": "web_search", "args": {"query": "weather in Tokyo"}}',
    "```",
    "",
    "After tool results are returned to you, continue reasoning. When you have",
    "the final answer, reply in plain text with no tool block.",
  ].join("\n");
}

/**
 * Parse a model response into free text plus any tool calls. Tool calls are
 * fenced ```tool / ```json blocks holding a {name, args} object; a bare JSON
 * object that is itself a tool call is also accepted. Malformed blocks are
 * ignored rather than treated as calls.
 */
export function parseResponse(raw: string): ParsedResponse {
  const toolCalls: ParsedToolCall[] = [];
  const fence = /```(?:tool|json)?\s*([\s\S]*?)```/g;
  let text = raw;
  let match: RegExpExecArray | null;

  while ((match = fence.exec(raw)) !== null) {
    const call = tryParseCall(match[1] ?? "");
    if (call) {
      toolCalls.push(call);
      text = text.replace(match[0], "");
    }
  }

  if (toolCalls.length === 0) {
    // Allow a bare JSON tool call with no fences.
    const bare = tryParseCall(raw);
    if (bare) {
      return { text: "", toolCalls: [bare] };
    }
  }

  return { text: text.trim(), toolCalls };
}

function tryParseCall(candidate: string): ParsedToolCall | null {
  const trimmed = candidate.trim();
  if (!trimmed.startsWith("{")) return null;
  let obj: unknown;
  try {
    obj = JSON.parse(trimmed);
  } catch {
    return null;
  }
  if (typeof obj !== "object" || obj === null) return null;

  const record = obj as Record<string, unknown>;
  const name = record.name ?? record.tool;
  if (typeof name !== "string" || !name) return null;

  const args =
    typeof record.args === "object" && record.args !== null
      ? (record.args as Record<string, unknown>)
      : {};

  return { name, args };
}

function errorMessage(err: unknown): string {
  if (err instanceof BudgetExhaustedError) {
    return `Token budget exhausted (${err.reason}). ${err.message}`;
  }
  if (err instanceof InferenceError) {
    return `Inference unavailable: ${err.message}`;
  }
  return `Unexpected error: ${String(err)}`;
}
