# Colab MCP server — ready-to-apply Claude Code config (2026-08-16)

⛔ **Not applied by the agent that wrote this.** Registering an MCP server
changes live Claude Code config; that is the orchestrator's/PI's call and takes
effect next session. Everything below is copy-paste-ready.

The server is **already installed and handshake-verified on this box**
(MEASURED 2026-08-16): `C:\Users\Admin\venvs\colab\Scripts\colab-mcp.exe`
(official `googlecolab/colab-mcp` 1.0.1, stdio transport, Python 3.14 venv).
Background: `colab/COLAB_CLI_MCP.md` §5.

## Option A (recommended on this box): pinned venv executable

No `uv` dependency, no network at server start, survives offline. Either add to
the **project** `.mcp.json` (repo root — shared, so commit-reviewable):

```json
{
  "mcpServers": {
    "colab": {
      "command": "C:\\Users\\Admin\\venvs\\colab\\Scripts\\colab-mcp.exe",
      "args": []
    }
  }
}
```

or equivalently via CLI (writes the same thing; `--scope user` for user-global
instead):

```
claude mcp add colab --scope project -- C:\Users\Admin\venvs\colab\Scripts\colab-mcp.exe
```

⚠️ The venv path is box-local — if this entry goes into the shared project
`.mcp.json`, other machines need the same venv path or their own Option-B
entry. `--scope user` (or `.claude/settings.local.json` permissions) keeps it
box-local.

## Option B (official README style): uvx from GitHub

What Google's README ships (requires `uv` on PATH; first run clones + builds,
so it needs network and can be slow/proxy-sensitive — behind this proxy prefer
Option A):

```json
{
  "mcpServers": {
    "colab": {
      "command": "uvx",
      "args": ["git+https://github.com/googlecolab/colab-mcp"],
      "timeout": 30000
    }
  }
}
```

## After registration — how a session actually connects

1. New Claude Code session: the server shows **one tool**,
   `open_colab_browser_connection` (MEASURED pre-pairing tool list).
2. Precondition: the **PI is signed in to Colab in this box's default
   browser** — the browser session is the entire auth; the server holds no
   Google credentials, only an ephemeral pairing token.
3. Calling the tool opens
   `https://colab.research.google.com/notebooks/empty.ipynb#mcpProxyToken=<t>&mcpProxyPort=<p>`
   (a scratch notebook) and waits **60 s** for the tab to pair (returns
   `true`/`false`).
4. On pairing, the notebook tools (cells: get/add/update/delete/run) appear via
   `tools/list_changed` — Claude Code supports this.
5. Runtime type (e.g. T4) is set **by hand in the tab**; the official server
   has no runtime-control tool.

## Hygiene / known sharp edges

- **One connection per server, ephemeral port**: kill stray `colab-mcp.exe`
  processes before re-pairing (orphans from dead sessions cause the
  "Disconnected" symptom — INHERITED from the community fork's analysis,
  consistent with the source).
- Server is local-only by design (official README: it cannot serve remote
  clients).
- Server logs land in `%TEMP%\colab-mcp-logs-*\colab-mcp.<ts>.log` (MEASURED).
- ⛔ No agent may perform the sign-in of step 2 — PI only.
