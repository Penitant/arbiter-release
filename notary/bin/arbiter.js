#!/usr/bin/env node
// opencode-arbiter CLI: setup / start / stop / status.
// Pure Node, cross-platform (Linux + macOS + Windows). No shell-outs for file ops or downloads.

import { createHash } from "node:crypto";
import { spawn, execFile } from "node:child_process";
import { createWriteStream, createReadStream, existsSync } from "node:fs";
import { mkdir, readFile, writeFile, rm } from "node:fs/promises";
import { homedir, platform } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { pipeline } from "node:stream/promises";
import { Readable } from "node:stream";
import { x as tarExtract } from "tar";

const execFileAsync = promisify(execFile);

// --- constants ---------------------------------------------------------------

const HOME = join(homedir(), ".arbiter");
const SERVER_DIR = join(HOME, "server");
const VENV_DIR = join(HOME, ".venv");
const PID_FILE = join(HOME, "server.pid");
const CHECKPOINT_REL = join("triage", "weights", "checkpoint.pt");

const HOST = process.env.ARBITER_HOST || "127.0.0.1";
const PORT = process.env.ARBITER_PORT || "4317";

// Where release assets live. Override with ARBITER_RELEASE_REPO=owner/repo.
const RELEASE_REPO = process.env.ARBITER_RELEASE_REPO || "Penitant/arbiter-release";

const IS_WIN = platform() === "win32";

// --- tiny logger -------------------------------------------------------------

const c = {
  dim: (s) => `\x1b[2m${s}\x1b[0m`,
  green: (s) => `\x1b[32m${s}\x1b[0m`,
  red: (s) => `\x1b[31m${s}\x1b[0m`,
  cyan: (s) => `\x1b[36m${s}\x1b[0m`,
};
const log = (...a) => console.log(...a);
const step = (s) => log(c.cyan("›"), s);
const ok = (s) => log(c.green("✓"), s);
const fail = (s) => { console.error(c.red("✗"), s); process.exitCode = 1; };

// --- version -----------------------------------------------------------------

async function pkgVersion() {
  const here = dirname(fileURLToPath(import.meta.url));
  const raw = await readFile(join(here, "..", "package.json"), "utf8");
  return JSON.parse(raw).version;
}

// --- github release asset download -------------------------------------------

function assetUrl(tag, name) {
  return `https://github.com/${RELEASE_REPO}/releases/download/${tag}/${name}`;
}

async function download(url, dest) {
  const res = await fetch(url, { redirect: "follow" });
  if (!res.ok || !res.body) {
    throw new Error(`download failed (${res.status}) for ${url}`);
  }
  await mkdir(dirname(dest), { recursive: true });
  await pipeline(Readable.fromWeb(res.body), createWriteStream(dest));
}

async function fetchText(url) {
  const res = await fetch(url, { redirect: "follow" });
  if (!res.ok) throw new Error(`fetch failed (${res.status}) for ${url}`);
  return res.text();
}

async function sha256File(path) {
  const hash = createHash("sha256");
  await pipeline(createReadStream(path), hash);
  return hash.digest("hex");
}

// A .sha256 file may be "<hex>" or "<hex>  filename"; take the first field.
function parseChecksum(text) {
  return text.trim().split(/\s+/)[0].toLowerCase();
}

// --- python discovery --------------------------------------------------------

async function findPython() {
  const candidates = IS_WIN ? ["python", "python3"] : ["python3", "python"];
  for (const bin of candidates) {
    try {
      const { stdout } = await execFileAsync(bin, ["--version"]);
      if (/Python 3\.(1[1-9]|[2-9]\d)/.test(stdout)) return bin;
    } catch {
      // try next candidate
    }
  }
  throw new Error(
    "no suitable Python found. Install Python 3.11+ and put `python3` (or `python`) on PATH.",
  );
}

function venvBin(name) {
  return IS_WIN
    ? join(VENV_DIR, "Scripts", `${name}.exe`)
    : join(VENV_DIR, "bin", name);
}

// --- opencode config patch ---------------------------------------------------

function opencodeConfigPath() {
  if (process.env.OPENCODE_CONFIG) return process.env.OPENCODE_CONFIG;
  const xdg = process.env.XDG_CONFIG_HOME || join(homedir(), ".config");
  return join(xdg, "opencode", "opencode.json");
}

async function patchOpencodeConfig() {
  const path = opencodeConfigPath();
  const baseURL = `http://${HOST}:${PORT}/v1`;
  let cfg = {};
  if (existsSync(path)) {
    try {
      cfg = JSON.parse(await readFile(path, "utf8"));
    } catch {
      log(c.red("!"), `could not parse ${path}. Add this provider manually:`);
      log(c.dim(`    provider.arbiter.options.baseURL = "${baseURL}"`));
      return;
    }
  }
  cfg.provider ??= {};
  cfg.provider.arbiter = {
    npm: "@ai-sdk/openai-compatible",
    name: "Arbiter (local router)",
    ...cfg.provider.arbiter,
    options: { ...(cfg.provider.arbiter?.options || {}), baseURL },
  };
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, JSON.stringify(cfg, null, 2) + "\n");
  ok(`patched OpenCode config ${c.dim(path)} → baseURL ${baseURL}`);
}

// --- commands ----------------------------------------------------------------

async function cmdSetup() {
  const version = await pkgVersion();
  const tag = `v${version}`;
  log(`Setting up Arbiter ${c.cyan(tag)} into ${c.dim(HOME)}\n`);
  await mkdir(HOME, { recursive: true });

  const tarName = `arbiter-server-${tag}.tar.gz`;
  const tarPath = join(HOME, tarName);

  step("downloading server bundle…");
  await download(assetUrl(tag, tarName), tarPath);
  const publishedTarSum = parseChecksum(await fetchText(assetUrl(tag, `${tarName}.sha256`)));
  const actualTarSum = await sha256File(tarPath);
  if (actualTarSum !== publishedTarSum) {
    throw new Error(`server bundle checksum mismatch (got ${actualTarSum}, expected ${publishedTarSum})`);
  }
  ok("server bundle verified");

  step("extracting server…");
  await rm(SERVER_DIR, { recursive: true, force: true });
  await mkdir(SERVER_DIR, { recursive: true });
  // The tarball's top-level entry is `server/`; strip it so files land directly in SERVER_DIR.
  await tarExtract({ file: tarPath, cwd: SERVER_DIR, strip: 1 });
  ok("server extracted");

  step("downloading classifier checkpoint…");
  const ckptPath = join(SERVER_DIR, CHECKPOINT_REL);
  await download(assetUrl(tag, "checkpoint.pt"), ckptPath);
  const publishedCkptSum = parseChecksum(await fetchText(assetUrl(tag, "checkpoint.pt.sha256")));
  const actualCkptSum = await sha256File(ckptPath);
  if (actualCkptSum !== publishedCkptSum) {
    await rm(ckptPath, { force: true });
    throw new Error(`checkpoint checksum mismatch (got ${actualCkptSum}, expected ${publishedCkptSum})`);
  }
  ok("checkpoint verified");

  step("creating Python virtualenv…");
  const python = await findPython();
  await execFileAsync(python, ["-m", "venv", VENV_DIR]);
  ok(`virtualenv at ${c.dim(VENV_DIR)} (${python})`);

  step("installing server dependencies (CPU torch — can take a few minutes)…");
  const pip = venvBin("pip");
  await execFileAsync(pip, ["install", "--upgrade", "pip"], { maxBuffer: 1 << 26 });
  await execFileAsync(
    pip,
    [
      "install",
      "--extra-index-url", "https://download.pytorch.org/whl/cpu",
      "fastapi>=0.115",
      "uvicorn[standard]>=0.30",
      "httpx>=0.27",
      "torch>=2.4",
      "transformers>=4.44,<5",
    ],
    { maxBuffer: 1 << 26 },
  );
  ok("dependencies installed");

  step("patching OpenCode config…");
  await patchOpencodeConfig();

  log(`\n${c.green("Done.")} Start the server with ${c.cyan("arbiter start")}.`);
}

async function readPid() {
  try {
    const pid = parseInt(await readFile(PID_FILE, "utf8"), 10);
    return Number.isInteger(pid) ? pid : null;
  } catch {
    return null;
  }
}

function isAlive(pid) {
  try {
    process.kill(pid, 0);
    return true;
  } catch (e) {
    return e.code === "EPERM"; // exists but owned by someone else
  }
}

async function cmdStart() {
  const existing = await readPid();
  if (existing && isAlive(existing)) {
    ok(`already running (pid ${existing}) on ${HOST}:${PORT}`);
    return;
  }
  if (!existsSync(SERVER_DIR)) {
    throw new Error(`server not installed. Run ${c.cyan("arbiter setup")} first.`);
  }
  const python = venvBin("python");
  const child = spawn(
    python,
    ["-m", "uvicorn", "main:app", "--host", HOST, "--port", PORT],
    { cwd: SERVER_DIR, detached: true, stdio: "ignore", windowsHide: true },
  );
  child.unref();
  await writeFile(PID_FILE, String(child.pid) + "\n");
  ok(`started (pid ${child.pid}) on http://${HOST}:${PORT}`);
}

async function cmdStop() {
  const pid = await readPid();
  if (!pid) {
    log("not running (no pid file).");
    return;
  }
  try {
    process.kill(pid, "SIGTERM");
    ok(`stopped (pid ${pid}).`);
  } catch (e) {
    if (e.code === "ESRCH") log(`process ${pid} was not running.`);
    else throw e;
  } finally {
    await rm(PID_FILE, { force: true });
  }
}

async function cmdStatus() {
  const pid = await readPid();
  if (!pid || !isAlive(pid)) {
    log(c.red("● stopped"));
    return;
  }
  log(c.green("● running"), c.dim(`pid ${pid} — http://${HOST}:${PORT}`));
  try {
    const res = await fetch(`http://${HOST}:${PORT}/health`, {
      signal: AbortSignal.timeout(3000),
    });
    log("  /health:", JSON.stringify(await res.json()));
  } catch {
    log("  /health: unreachable (server starting or bound elsewhere)");
  }
}

// --- dispatch ----------------------------------------------------------------

const HELP = `opencode-arbiter — local-first model router for OpenCode

Usage: arbiter <command>

  setup    download + install the server, create a venv, patch OpenCode config
  start    start the server (detached) on ${HOST}:${PORT}
  stop     stop the running server
  status   show whether the server is running + /health

Env: ARBITER_HOST, ARBITER_PORT, ARBITER_RELEASE_REPO, OPENCODE_CONFIG`;

const commands = { setup: cmdSetup, start: cmdStart, stop: cmdStop, status: cmdStatus };

async function main() {
  const cmd = process.argv[2];
  if (!cmd || cmd === "-h" || cmd === "--help") return void log(HELP);
  const handler = commands[cmd];
  if (!handler) return fail(`unknown command "${cmd}"\n\n${HELP}`);
  try {
    await handler();
  } catch (e) {
    fail(e.message || String(e));
  }
}

main();
