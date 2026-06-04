/**
 * Inference router — the single controlled point for every LLM call.
 *
 * Non-negotiable constraint: no LLM call may bypass this router. It is the
 * sanctioned exception to the egress proxy (local inference talks to Ollama on
 * localhost, which the proxy would otherwise block). Responsibilities:
 *   - enforce per-conversation and per-day token budgets
 *   - track and persist token usage
 *   - fall back to a secondary model when the primary target fails
 *   - audit every completion (and every budget block)
 *
 * Supported providers: "ollama" (default, local) and any OpenAI-compatible
 * chat-completions endpoint ("openai").
 */

import type { Database } from "bun:sqlite";
import type {
  AuditLogger,
  BudgetExhaustedReason,
  ChatMessage,
  InferenceConfig,
  InferenceRequest,
  InferenceResponse,
  ModelTarget,
} from "../types.ts";

export class BudgetExhaustedError extends Error {
  readonly reason: BudgetExhaustedReason;
  constructor(reason: BudgetExhaustedReason, message: string) {
    super(message);
    this.name = "BudgetExhaustedError";
    this.reason = reason;
  }
}

export class InferenceError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "InferenceError";
  }
}

export class InferenceRouter {
  readonly #config: InferenceConfig;
  readonly #audit: AuditLogger;
  readonly #db: Database;
  /** Normalized allowlist of base URLs the router is permitted to contact. */
  readonly #trusted: Set<string>;
  /** In-memory per-conversation token totals (reset when process restarts). */
  readonly #conversationTokens = new Map<string, number>();

  constructor(config: InferenceConfig, audit: AuditLogger, db: Database) {
    this.#config = config;
    this.#audit = audit;
    this.#db = db;

    // The allowlist defaults to exactly the configured targets, so an env-driven
    // misconfiguration can never silently redirect inference elsewhere. Validate
    // at construction (fail fast) that every configured target is trusted.
    const sources =
      config.trustedBaseUrls && config.trustedBaseUrls.length > 0
        ? config.trustedBaseUrls
        : [config.primary.baseUrl, ...(config.fallback ? [config.fallback.baseUrl] : [])];
    this.#trusted = new Set(sources.map(normalizeBaseUrl));

    for (const target of [config.primary, config.fallback]) {
      if (target && !this.#trusted.has(normalizeBaseUrl(target.baseUrl))) {
        throw new InferenceError(
          `inference target "${target.baseUrl}" is not in trustedBaseUrls`,
        );
      }
    }
  }

  /** Current per-conversation token total for a session. */
  conversationTokens(sessionId: string): number {
    return this.#conversationTokens.get(sessionId) ?? 0;
  }

  /** Tokens consumed today (UTC) across all sessions. */
  dayTokens(): number {
    const row = this.#db
      .query<{ tokens: number }, [string]>(
        "SELECT tokens FROM token_usage WHERE day = ?",
      )
      .get(today());
    return row?.tokens ?? 0;
  }

  /**
   * Run one completion through the router. Throws BudgetExhaustedError when a
   * budget is hit (the agent loop decides whether to stop or compress) and
   * InferenceError when both primary and fallback fail.
   */
  async complete(req: InferenceRequest): Promise<InferenceResponse> {
    this.#enforceBudget(req.sessionId);

    const start = Date.now();
    const { primary, fallback } = this.#config;

    let response: InferenceResponse;
    try {
      response = await this.#callTarget(primary, req, false);
    } catch (primaryErr) {
      if (!fallback) {
        this.#auditFailure(req.sessionId, start, String(primaryErr));
        throw new InferenceError(
          `primary inference failed and no fallback configured: ${String(
            primaryErr,
          )}`,
        );
      }
      try {
        response = await this.#callTarget(fallback, req, true);
      } catch (fallbackErr) {
        this.#auditFailure(
          req.sessionId,
          start,
          `primary: ${String(primaryErr)}; fallback: ${String(fallbackErr)}`,
        );
        throw new InferenceError(
          `both primary and fallback inference failed: ${String(fallbackErr)}`,
        );
      }
    }

    this.#recordUsage(req.sessionId, response.totalTokens);

    this.#audit({
      timestamp: Date.now(),
      sessionId: req.sessionId,
      actionType: "inference",
      toolName: response.model,
      result: "success",
      tokenCost: response.totalTokens,
      durationMs: Date.now() - start,
    });

    return response;
  }

  #enforceBudget(sessionId: string): void {
    const { perConversation, perDay } = this.#config.tokenBudget;

    if (this.conversationTokens(sessionId) >= perConversation) {
      this.#auditBlocked(sessionId, "conversation token budget exhausted");
      throw new BudgetExhaustedError(
        "conversation",
        `conversation budget of ${perConversation} tokens exhausted`,
      );
    }
    if (this.dayTokens() >= perDay) {
      this.#auditBlocked(sessionId, "daily token budget exhausted");
      throw new BudgetExhaustedError(
        "day",
        `daily budget of ${perDay} tokens exhausted`,
      );
    }
  }

  #recordUsage(sessionId: string, tokens: number): void {
    this.#conversationTokens.set(
      sessionId,
      this.conversationTokens(sessionId) + tokens,
    );
    this.#db
      .query(
        `INSERT INTO token_usage (day, tokens) VALUES ($day, $tokens)
         ON CONFLICT(day) DO UPDATE SET tokens = tokens + $tokens`,
      )
      .run({ $day: today(), $tokens: tokens });
  }

  #auditBlocked(sessionId: string, reason: string): void {
    this.#audit({
      timestamp: Date.now(),
      sessionId,
      actionType: "blocked",
      result: "blocked",
      blockedReason: reason,
    });
  }

  #auditFailure(sessionId: string, start: number, reason: string): void {
    this.#audit({
      timestamp: Date.now(),
      sessionId,
      actionType: "inference",
      result: "error",
      blockedReason: reason,
      durationMs: Date.now() - start,
    });
  }

  async #callTarget(
    target: ModelTarget,
    req: InferenceRequest,
    isFallback: boolean,
  ): Promise<InferenceResponse> {
    // Defense in depth: re-verify the destination at call time, not just at
    // construction, so no code path can reach an untrusted inference endpoint.
    if (!this.#trusted.has(normalizeBaseUrl(target.baseUrl))) {
      this.#auditBlocked(
        req.sessionId,
        `inference target not in trustedBaseUrls: ${target.baseUrl}`,
      );
      throw new InferenceError(
        `refusing to contact untrusted inference endpoint: ${target.baseUrl}`,
      );
    }
    if (target.provider === "ollama") {
      return this.#callOllama(target, req, isFallback);
    }
    // Treat everything else as an OpenAI-compatible chat endpoint.
    return this.#callOpenAICompatible(target, req, isFallback);
  }

  async #callOllama(
    target: ModelTarget,
    req: InferenceRequest,
    isFallback: boolean,
  ): Promise<InferenceResponse> {
    const url = `${trimSlash(target.baseUrl)}/api/chat`;

    if (req.onToken) {
      return this.#streamOllama(url, target, req, isFallback);
    }

    const body = {
      model: target.model,
      messages: req.messages.map(toProviderMessage),
      stream: false,
      options: {
        temperature: req.temperature ?? 0.7,
        ...(req.maxTokens ? { num_predict: req.maxTokens } : {}),
      },
    };

    const raw = await this.#postJson(url, body);
    const content: string =
      (raw as { message?: { content?: string } }).message?.content ?? "";
    const promptTokens =
      (raw as { prompt_eval_count?: number }).prompt_eval_count ??
      estimateTokens(req.messages);
    const completionTokens =
      (raw as { eval_count?: number }).eval_count ?? estimateText(content);

    return {
      content,
      promptTokens,
      completionTokens,
      totalTokens: promptTokens + completionTokens,
      model: target.model,
      usedFallback: isFallback,
    };
  }

  async #streamOllama(
    url: string,
    target: ModelTarget,
    req: InferenceRequest,
    isFallback: boolean,
  ): Promise<InferenceResponse> {
    const body = {
      model: target.model,
      messages: req.messages.map(toProviderMessage),
      stream: true,
      options: {
        temperature: req.temperature ?? 0.7,
        ...(req.maxTokens ? { num_predict: req.maxTokens } : {}),
      },
    };

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 120_000);

    let content = "";
    let promptTokens = 0;
    let completionTokens = 0;

    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      if (!res.ok) {
        const detail = await res.text().catch(() => "");
        throw new InferenceError(
          `inference endpoint ${url} returned ${res.status}: ${detail.slice(0, 200)}`,
        );
      }
      if (!res.body) throw new InferenceError("no response body for streaming");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";

      const processLine = (line: string): void => {
        const trimmed = line.trim();
        if (!trimmed) return;
        let chunk: unknown;
        try { chunk = JSON.parse(trimmed); } catch { return; }

        const token =
          (chunk as { message?: { content?: string } }).message?.content ?? "";
        if (token) {
          content += token;
          req.onToken!(token);
        }

        const isDone = (chunk as { done?: boolean }).done === true;
        if (isDone) {
          promptTokens =
            (chunk as { prompt_eval_count?: number }).prompt_eval_count ??
            estimateTokens(req.messages);
          completionTokens =
            (chunk as { eval_count?: number }).eval_count ??
            estimateText(content);
        }
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });

        // Ollama streams one JSON object per line. Keep the last incomplete
        // line in the buffer in case it spans a chunk boundary.
        const lines = buf.split("\n");
        buf = lines.pop() ?? "";
        for (const line of lines) processLine(line);
      }

      // Flush any remaining content (e.g. a complete response without a
      // trailing newline, which is the case for non-streaming mock responses).
      if (buf.trim()) processLine(buf);
      buf = "";
    } finally {
      clearTimeout(timer);
    }

    if (!promptTokens) promptTokens = estimateTokens(req.messages);
    if (!completionTokens) completionTokens = estimateText(content);

    return {
      content,
      promptTokens,
      completionTokens,
      totalTokens: promptTokens + completionTokens,
      model: target.model,
      usedFallback: isFallback,
    };
  }

  async #callOpenAICompatible(
    target: ModelTarget,
    req: InferenceRequest,
    isFallback: boolean,
  ): Promise<InferenceResponse> {
    const url = `${trimSlash(target.baseUrl)}/v1/chat/completions`;

    if (req.onToken) {
      return this.#streamOpenAI(url, target, req, isFallback);
    }

    const body = {
      model: target.model,
      messages: req.messages.map(toProviderMessage),
      temperature: req.temperature ?? 0.7,
      ...(req.maxTokens ? { max_tokens: req.maxTokens } : {}),
      stream: false,
    };

    const raw = (await this.#postJson(url, body)) as {
      choices?: { message?: { content?: string } }[];
      usage?: { prompt_tokens?: number; completion_tokens?: number };
    };
    const content = raw.choices?.[0]?.message?.content ?? "";
    const promptTokens =
      raw.usage?.prompt_tokens ?? estimateTokens(req.messages);
    const completionTokens =
      raw.usage?.completion_tokens ?? estimateText(content);

    return {
      content,
      promptTokens,
      completionTokens,
      totalTokens: promptTokens + completionTokens,
      model: target.model,
      usedFallback: isFallback,
    };
  }

  async #streamOpenAI(
    url: string,
    target: ModelTarget,
    req: InferenceRequest,
    isFallback: boolean,
  ): Promise<InferenceResponse> {
    const body = {
      model: target.model,
      messages: req.messages.map(toProviderMessage),
      temperature: req.temperature ?? 0.7,
      ...(req.maxTokens ? { max_tokens: req.maxTokens } : {}),
      stream: true,
    };

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 120_000);

    let content = "";

    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      if (!res.ok) {
        const detail = await res.text().catch(() => "");
        throw new InferenceError(
          `inference endpoint ${url} returned ${res.status}: ${detail.slice(0, 200)}`,
        );
      }
      if (!res.body) throw new InferenceError("no response body for streaming");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";

      const processSSELine = (line: string): void => {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data:")) return;
        const data = trimmed.slice(5).trim();
        if (data === "[DONE]") return;
        let chunk: unknown;
        try { chunk = JSON.parse(data); } catch { return; }

        const token =
          (chunk as { choices?: { delta?: { content?: string } }[] })
            .choices?.[0]?.delta?.content ?? "";
        if (token) {
          content += token;
          req.onToken!(token);
        }
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });

        // SSE format: "data: {...}\n\n"
        const lines = buf.split("\n");
        buf = lines.pop() ?? "";
        for (const line of lines) processSSELine(line);
      }

      if (buf.trim()) processSSELine(buf);
    } finally {
      clearTimeout(timer);
    }

    return {
      content,
      promptTokens: estimateTokens(req.messages),
      completionTokens: estimateText(content),
      totalTokens: estimateTokens(req.messages) + estimateText(content),
      model: target.model,
      usedFallback: isFallback,
    };
  }

  /**
   * Direct JSON POST. The router is the sanctioned bypass of the egress proxy:
   * local inference must reach localhost, which the proxy blocks by design.
   *
   * KNOWN V1 LIMITATION — DNS rebinding. The trusted-endpoint check validates
   * the URL *string*, not the IP the host ultimately resolves to. A poisoned
   * resolver or /etc/hosts entry could point a trusted hostname at, e.g.,
   * 169.254.169.254 (cloud metadata) and slip past the allowlist.
   *
   * We deliberately do NOT resolve-and-reject private/loopback/link-local IPs
   * here, because that is incompatible with this router's whole reason to exist:
   * the default target is Ollama on localhost (→ 127.0.0.1), which such a check
   * would itself block. Resolve-then-pin rebinding defense (resolve once, pin
   * the IP, fetch against the pinned address, reject on later drift) only
   * becomes meaningful for V2 cloud inference over *public* endpoints, and is
   * deferred to that milestone. For V1 (local inference) this is not exploitable
   * in the intended deployment.
   */
  async #postJson(url: string, body: unknown): Promise<unknown> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 120_000);
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      if (!res.ok) {
        const detail = await res.text().catch(() => "");
        throw new InferenceError(
          `inference endpoint ${url} returned ${res.status}: ${detail.slice(
            0,
            200,
          )}`,
        );
      }
      return await res.json();
    } finally {
      clearTimeout(timer);
    }
  }
}

function toProviderMessage(m: ChatMessage): { role: string; content: string } {
  // Providers expect tool output folded into a user/system turn; map "tool"
  // role to "user" with a clear marker so local models understand context.
  const role = m.role === "tool" ? "user" : m.role;
  const content =
    m.role === "tool" ? `[tool:${m.name ?? "unknown"}] ${m.content}` : m.content;
  return { role, content };
}

function estimateTokens(messages: ChatMessage[]): number {
  return messages.reduce((sum, m) => sum + estimateText(m.content), 0);
}

/**
 * Rough token estimate used ONLY as a fallback when a provider omits usage
 * counts. The `length / 4` heuristic is tuned for English ASCII; it
 * underestimates by ~33-50% for text with multi-byte or symbol-dense content —
 * Romanian and other diacritics, source code, and especially CJK scripts —
 * because those pack more tokens per character than the 4-chars/token average.
 *
 * KNOWN V1 LIMITATION. This is cosmetic for V1: local Ollama returns real
 * `prompt_eval_count` / `eval_count`, so this path is rarely taken. V2 cloud
 * inference (where accurate budgeting affects cost) should swap this for a real
 * tokenizer (e.g. tiktoken / the provider's tokenizer).
 */
function estimateText(text: string): number {
  return Math.ceil(text.length / 4);
}

function trimSlash(url: string): string {
  return url.replace(/\/+$/, "");
}

/** Default port for a URL scheme, or "" when the scheme has none we track. */
function defaultPortFor(scheme: string): string {
  if (scheme === "https:") return "443";
  if (scheme === "http:") return "80";
  return "";
}

/**
 * Canonical form for base-URL allowlist comparison.
 *
 * Built from `hostname` and the *explicit* port only — never `URL.host`, which
 * folds the port into the string and makes an explicit default port
 * (`https://ollama.com:443`) compare unequal to its implicit form
 * (`https://ollama.com`). A port is included only when it is present AND differs
 * from the scheme's default, so the two forms canonicalize identically.
 *
 * The exact same function is used for allowlist construction and request-time
 * validation, so there is no asymmetry between the two checks.
 */
export function normalizeBaseUrl(url: string): string {
  try {
    const u = new URL(url);
    const scheme = u.protocol.toLowerCase();
    const port =
      u.port && u.port !== defaultPortFor(scheme) ? `:${u.port}` : "";
    const path = trimSlash(u.pathname);
    return `${scheme}//${u.hostname}${port}${path}`.toLowerCase();
  } catch {
    return trimSlash(url.trim()).toLowerCase();
  }
}

function today(): string {
  return new Date().toISOString().slice(0, 10);
}
