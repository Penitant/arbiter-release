// HTTP client for the Server's Notary contract; fails soft so a down Server never blocks the session.

/** The resolved proposal the Server returns from POST /classify. */
export interface Decision {
  difficulty: "easy" | "medium" | "hard";
  confidence: number;
  tier: "local-simple" | "cloud-mid" | "cloud-heavy";
  model: string; // "provider/id", e.g. "google/gemini-2.5-pro"
  reasoning_effort: string | null;
  reason: string;
}

export interface ClassifyRequest {
  task_text: string;
  session_id?: string;
  context_tokens_estimate?: number;
}

/** One logged resolution — accepted or overridden. */
export interface Correction {
  task_text: string;
  predicted_tier: string;
  predicted_model: string;
  confidence: number;
  session_id: string;
  corrected_tier?: string | null;
  corrected_model?: string | null;
}

const DEFAULT_BASE_URL = "http://127.0.0.1:4317";

export class ServerClient {
  private readonly baseUrl: string;
  private readonly timeoutMs: number;

  constructor(baseUrl?: string, timeoutMs = 4000) {
    // ARBITER_SERVER_URL overrides the default localhost host/port.
    this.baseUrl = (
      baseUrl ?? process.env.ARBITER_SERVER_URL ?? DEFAULT_BASE_URL
    ).replace(/\/+$/, "");
    this.timeoutMs = timeoutMs;
  }

  /** Resolve a proposal for a task; null on any failure (caller falls back silently). */
  async classify(req: ClassifyRequest): Promise<Decision | null> {
    try {
      const resp = await this.post("/classify", req);
      if (!resp.ok) return null; // includes 503 -> documented soft fallback
      return (await resp.json()) as Decision;
    } catch {
      return null;
    }
  }

  /** Log a resolution. Best-effort: errors are swallowed, never disrupting the session. */
  async logCorrection(c: Correction): Promise<void> {
    try {
      await this.post("/corrections", c);
    } catch {
      /* swallow — logging is never allowed to disrupt the session */
    }
  }

  private async post(path: string, body: unknown): Promise<Response> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      return await fetch(`${this.baseUrl}${path}`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
    } finally {
      clearTimeout(timer);
    }
  }
}
