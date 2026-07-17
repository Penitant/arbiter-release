# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and versions follow [SemVer](https://semver.org/).

## [1.0.0] — 2026-07-17

First public release.

### Added
- **Server** (Python/FastAPI): a local HTTP proxy OpenCode points its `baseUrl` at.
  - **Triage** — a trained PyTorch classifier (difficulty: easy/medium/hard) plus a
    deterministic resolver that maps difficulty → tier → a concrete model via the Registry.
  - **Ferry** — reactive, tier-scaled context payloads on a mid-session model switch or a new
    child session (minimal / structured handoff summary / full raw history).
  - **Registry** — live model catalog from models.dev + OpenRouter benchmarks + local Ollama
    tags, with cross-catalog identity matching.
  - Wire forwarding for Ollama, OpenAI-compatible providers, Google, and Anthropic.
  - Provider discovery and credential validation at startup; correction log in SQLite with
    incremental on-device fine-tuning.
- **Notary** (OpenCode plugin): plan-mode proposal menu, per-subtask overrides, correction
  logging, and the `/train-mode` toggle.
- **`arbiter` CLI**: `setup` / `start` / `stop` / `status` for installing and running the
  server cross-platform.
