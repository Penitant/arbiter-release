# opencode-arbiter

Local-first model router for [OpenCode](https://opencode.ai). This package ships two things:

- the **`arbiter` CLI**, which installs and runs the router server, and
- the **Notary plugin**, the OpenCode-side UI (plan-mode menu, overrides, correction logging).

Routing lives entirely in the server; the plugin is UI and logging only — it never calls a
model directly.

## Install

```bash
npm install -g opencode-arbiter
arbiter setup     # download + install the server, create a venv, patch OpenCode config
arbiter start     # run the router on 127.0.0.1:4317
```

`setup` needs Python 3.11+ on your PATH (used for a CPU-only PyTorch venv). Point OpenCode's
provider `baseUrl` at the router and it decides local (Ollama) vs cloud per task.

## CLI

| Command | Does |
|---|---|
| `arbiter setup`  | Downloads the server bundle + classifier checkpoint for this package version, verifies checksums, creates `~/.arbiter/.venv`, installs deps, patches OpenCode config. |
| `arbiter start`  | Starts the server detached, writes `~/.arbiter/server.pid`. |
| `arbiter stop`   | Stops the running server. |
| `arbiter status` | Reports running/stopped and the server's `/health`. |

Env: `ARBITER_HOST`, `ARBITER_PORT`, `ARBITER_RELEASE_REPO`, `OPENCODE_CONFIG`.

## Plugin

OpenCode loads the `.ts` plugin directly (no build step). Add the entrypoint to your OpenCode
config's `plugin` array:

```jsonc
// ~/.config/opencode/opencode.json
{ "plugin": ["opencode-arbiter/src/index.ts"] }
```

`/train-mode` toggles manual approval. **Manual** (default) shows each resolution as an
accept/override menu and feeds the correction loop; **auto** passes decisions straight through
and only logs them.

## Develop

```bash
npm install
npm run typecheck
npm test
```
