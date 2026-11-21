# Official Colab CLI + Colab MCP server — evaluation on the dev box (2026-08-16)

Task: PI asked us to try Google's newly shipped official Colab CLI and official
open-source Colab MCP server, toward autonomous operation of
`colab/SAM3_BACKFILL_115.ipynb` and `colab/STRATEGIC_LABEL_LAB.ipynb` from this
terminal / Claude Code. This doc carries the findings, MEASURED transcripts, the
PI's auth sequence, and the use-case map. Ready-to-apply Claude Code config lives
in `colab/MCP_SETUP.md`. Every claim carries its evidence class.

**Status: both tools are real, official, installed on this box, and driven to the
auth boundary. Nothing was authenticated and nothing executed on Colab — the PI
holds the auth step (§4).**

---

## 1. Identity — which packages are the real ones

| tool | official artifact | install | NOT to be confused with |
|---|---|---|---|
| **Colab CLI** | GitHub `googlecolab/google-colab-cli`, PyPI **`google-colab-cli`** (0.6.0, Python ≥3.12, Apache-2.0) | `pip install google-colab-cli` or `uv tool install google-colab-cli` | ⛔ PyPI **`colab-cli`** = third-party by Akshay Ashok (@Akshay090), last release 2020-04-27 — a gdrive-sync helper, no execution. Do not install it. |
| **Colab MCP server** | GitHub **`googlecolab/colab-mcp`** (package `colab-mcp` 1.0.1, Python ≥3.13, FastMCP-based) | `uvx git+https://github.com/googlecolab/colab-mcp` (official README style) or `pip install git+…` into a venv | third-party MCP servers named "colab" on lobehub/glama, and the fork `SebastianGilPinzon/colab-mcp` (see §6.4) |

PUBLISHED: announcements — "Introducing the Google Colab CLI" (Google Developers
Blog, ~2026-06-05; InfoQ 2026-06) and "Announcing the Colab MCP Server" (Google
Developers Blog; MarkTechPost 2026-03-19, InfoQ 2026-04). PyPI metadata for both
names re-verified 2026-08-16 (owners `sethtroisi`/`teeler`, homepage =
googlecolab org, vs. Akshay Ashok for `colab-cli`).

## 2. What was installed on this box (MEASURED, 2026-08-16)

Fresh venv, nothing touched in the tanitad venv:

```
C:\Users\Admin\venvs\colab            # Python 3.14.5, pip 26.1.1
  google-colab-cli 0.6.0              # pip install google-colab-cli
  colab-mcp 1.0.1                     # pip install git+https://github.com/googlecolab/colab-mcp
                                      # (fastmcp 2.14.5, mcp 1.29.0)
  Scripts\colab.exe, Scripts\colab-mcp.exe
```

Both installs completed clean behind the proxy (no truststore needed for pip
here). `colab-mcp` needs Python ≥3.13 — the 3.14 venv satisfies both tools.

## 3. Colab CLI findings

### 3.1 ⚠️ Windows status — unsupported upstream, import-crash measured, shim probe works

PUBLISHED (bundled README, printed by `colab readme`): *"Platform support: the
Colab CLI currently supports **Linux and macOS** only. Windows is not supported
at this time."*

MEASURED on this box: bare `colab --help` dies at import —
`colab_cli\console.py:20: import termios` → `ModuleNotFoundError: No module
named 'termios'` (Unix-only module, imported unconditionally through the command
tree). WSL is **not installed** on this box (`wsl --status`: not installed), so
there is no supported local path without a PI action.

MEASURED probe: with a 17-line stub `termios.py` on `PYTHONPATH` (import-only
shim; every real terminal call raises), the CLI is fully alive on native
Windows: `colab --help` renders the complete command tree (exit 0), subcommand
helps render, and the OAuth flow runs to the code prompt (§4). Runtime uses of
termios are confined to `console.py:137–169` (raw-TTY handling for
`repl`/`console`) and `automation.py:106` (`tty.readline` for interactive
`auth`/`drivemount`) — i.e. exactly the four commands the bundled skill says
agents must not run interactively anyway. The agent-relevant surface
(`new/exec/run/upload/download/ls/log/install/sessions/status/stop`) speaks
HTTP/WebSocket and plausibly works under the shim.

**Claim strength, stated precisely:** help + auth-boundary on native Windows are
MEASURED. Actual VM provisioning/exec on native Windows is HYPOTHESIS until the
first post-auth smoke — Google explicitly does not support it. The two clean
paths, PI's choice:
1. **WSL** (`wsl --install`, admin + reboot — PI action), then the same
   pip-install inside; this is the supported configuration; or
2. **native + shim** (zero extra install; first post-auth smoke decides whether
   it holds; the shim is vendored at `colab/win_shims/termios.py` — put that
   directory on `PYTHONPATH`).
Fallback for either: run the CLI from any Linux box (a pod) — it is just pip.

### 3.2 Command surface (MEASURED from the installed 0.6.0 `--help` tree)

Top-level: `console download drivemount edit exec help install log ls new pay
readme repl restart-kernel rm run sessions skill status stop update` + hidden
`url` (prints a browser URL that attaches the Colab web UI to a CLI session)
and hidden `whoami` (prints active identity/scopes/expiry — the 403 debugger).

The load-bearing ones for us:

| command | measured flags | notes |
|---|---|---|
| `colab new` | `-s NAME`, `--gpu {T4,L4,G4,H100,A100}`, `--tpu {v5e1,v6e1}` | "Availability varies by Colab subscription tier" (measured help text) |
| `colab exec` | `-s`, `-f FILE` (`.py` **or `.ipynb`**), `--output-image PATH`, `--timeout FLOAT [default 30.0]` | ⚠️ default 30 s/execution — our SAM3/VLM cells need an explicit large `--timeout` |
| `colab run` | `[--gpu…] [--keep] [-s NAME] SCRIPT [ARGS…]` | provision→exec→teardown in one shot; exit codes propagate; `[colab]` chatter on stderr, script stdout clean |
| `colab download/upload` | `REMOTE LOCAL` / `LOCAL REMOTE` | plus `ls`, `rm`, `edit` |
| `colab log` | `-n N`, `-t TYPE`, `-o FILE` (`.ipynb/.md/.txt/.jsonl`) | session history export = replayable audit artifact |
| `colab drivemount` | `[PATH]` default `/content/drive` | ⚠️ interactive — needs the PI at the terminal (PUBLISHED, bundled skill) |

PUBLISHED (bundled `colab skill` output, dumped 2026-08-16 — the authoritative
operating notes, verbatim quotes):
- *"Kernel state PERSISTS across `colab exec` / `colab repl` calls in the same
  session"* — build state incrementally, don't re-import per call.
- *"`colab exec -s <name> -f nb.ipynb` runs each code cell and writes results to
  `<basename>_output.ipynb` next to the input"* — headless notebook execution
  with a local output notebook.
- *"an unstopped session burns compute units indefinitely"* (24 h keep-alive
  cap); **always `colab stop`**.
- *"Never run `colab repl`, `colab console`, `colab auth`, or `colab drivemount`
  interactively from an agent"* — the first two accept piped stdin;
  `auth`/`drivemount` *"genuinely require a human at the terminal"*.
- *"an unrecognized `--gpu` value silently falls back to **A100**"* — spell `T4`
  exactly.
- Parallel agents: isolate state with `colab --config <path>` per stream.
- Recovery: 404/401 on exec = backend pruned the VM → `colab sessions` +
  `colab new`; wedged kernel → `colab restart-kernel`; keep-alive
  `consecutive_4xx_errors` = missing `colaboratory` scope.

Doc-vs-binary discrepancy, recorded: bundled README/skill say the default auth
strategy is `adc`; the installed 0.6.0's measured `--help` says `--auth
[default: oauth2]`. The measured zero-config behavior (§4) is what counts for
this box; `colab whoami` settles identity/scope questions post-auth.

### 3.3 Tier constraints (PUBLISHED — say-so lines, decisions are the PI's)

- Headless execution itself is **not** tier-gated anywhere in the docs; what is
  gated is **accelerator entitlement**: *"Accelerator availability is
  tier-gated; most accounts can only get CPU. Don't assume a GPU/TPU will
  allocate"* (bundled skill). A `400` on `colab new --gpu X` = no
  quota/entitlement for X on this account → fall back `--gpu T4` or CPU.
- Repo README (fetched 2026-08-16): high-RAM shape *"requires Colab Pro or
  Pro+ entitlement"*. `colab pay` opens the compute-units page.
- `colab new` pre-flights the keep-alive RPC and **unassigns the fresh VM** if
  the token lacks the `colaboratory` scope (no leaked billable assignment).
- Free-tier T4 availability/session-length variability is the same
  best-effort story as the Colab UI; the CLI docs promise nothing stronger.

## 4. The auth boundary (MEASURED) — and the PI's 2-minute sequence

Deliberately stopped here. No credentials exist for the agent and none were
entered. MEASURED on native Windows (shim), zero config files present — the
first authenticated-endpoint command prints:

```
$ colab sessions
To authorize colab-cli, visit this URL in any browser:

  https://accounts.google.com/o/oauth2/auth?response_type=code
    &client_id=764086051850-6qr4p6gpi6hn506pt8ejuq83di341hur.apps.googleusercontent.com
    &redirect_uri=https%3A%2F%2Fsdk.cloud.google.com%2Fapplicationdefaultauthcode.html
    &scope=openid+userinfo.profile+userinfo.email+cloud-platform+colaboratory+drive.file
    &state=<ephemeral>&code_challenge=<ephemeral-PKCE>&code_challenge_method=S256
    &prompt=consent&token_usage=remote&access_type=offline

After approving, Google will display an authorization code.
Enter the authorization code:
```

(URL reflowed; `state`/`code_challenge` are per-invocation PKCE values — the
printed URL is single-use. The client_id is Google's standard public
gcloud/installed-app client. Scopes requested: `openid`, `userinfo.profile`,
`userinfo.email`, `cloud-platform`, **`colaboratory`**, **`drive.file`**.)

This is a **copy-URL / paste-code flow — no local browser, no localhost
redirect** — it works from any terminal including headless ones. It aborts
cleanly on EOF (that is where we stopped).

**PI sequence (~2 minutes, one-time per box/account):**
1. On the dev box (native, PowerShell):
   ```powershell
   $env:PYTHONPATH = "G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\colab\win_shims"
   C:\Users\Admin\venvs\colab\Scripts\colab.exe sessions
   ```
   (under WSL, once installed: plain `colab sessions`, no shim).
2. Open the printed URL in any browser; sign in with the **Drive account that
   owns this repo** (the `drive.file` scope + Drive-mount both key off it);
   approve the consent screen.
3. Copy the authorization code Google displays; paste it at
   `Enter the authorization code:`.
4. The command then lists sessions (empty list expected). Verify with
   `colab whoami` — it must show the expected email and the `colaboratory`
   scope. Token caches under `~/.config/colab-cli/` (`token.json` per the
   bundled skill; treat that directory as a credential store — never commit).
5. Optional, recommended for our notebooks: on the first real session,
   `colab drivemount -s <name>` (interactive prompt — PI at the terminal), which
   exposes this repo at `/content/drive/MyDrive/SayBouBase/raw/Projects/TanitAD`.

No step of this may be performed by an agent; agents resume only after §4 is
done. (⚠️ also: `colab auth` ≠ CLI login — it injects GCP creds *into the VM*;
quoting the bundled skill: never "fix" a CLI 401/403 with it.)

## 5. Colab MCP server findings

### 5.1 Architecture (MEASURED from installed source, `colab_mcp` 1.0.1)

`colab-mcp` is **not** a headless Colab API client. It is a **stdio MCP server
that proxies to the Colab web UI in the user's browser**:

1. On start it opens a WebSocket listener on `ws://localhost:<ephemeral port>`
   (`websocket_server.py:42–61`: `host="localhost"`, `port=0` → OS-assigned;
   pairing secret `secrets.token_urlsafe(16)`; single connection only; allowed
   origins `https://colab.research.google.com` / `https://colab.google.com`).
2. Pre-pairing it exposes exactly **one** injected tool. Calling
   `open_colab_browser_connection` runs `webbrowser.open_new(
   "https://colab.research.google.com/notebooks/empty.ipynb#mcpProxyToken=
   <token>&mcpProxyPort=<port>")` (`session.py:162–171`) — i.e. it opens a
   **scratch notebook** in the default browser; the Colab page JS reads the URL
   fragment and connects back to the local WebSocket with the token.
3. The server waits **60 s** for the pairing (`UI_CONNECTION_TIMEOUT = 60.0`,
   progress messages "Waiting for user to connect in Colab - will wait for
   60s"), then the page-side tools are **proxied** through (FastMCPProxy) and
   a `notifications/tools/list_changed` unlocks them in the client.
4. **Auth model: there is none in the server.** The signed-in browser session
   IS the auth; the local process never holds Google credentials, only the
   ephemeral pairing token. Nothing to leak, nothing for an agent to enter.

### 5.2 MCP handshake on this box (MEASURED, stdio probe 2026-08-16)

`colab-mcp.exe` runs on **native Windows** (no termios issue — pure
fastmcp/websockets). Probe transcript (initialize → initialized → tools/list):

- stderr: `Starting MCP server 'ColabMCP' with transport 'stdio'`; logs to
  `%TEMP%\colab-mcp-logs-<rand>\colab-mcp.<ts>.log`.
- initialize result: `protocolVersion 2025-06-18`, `capabilities.tools.
  listChanged: true`, serverInfo `ColabMCP`.
- `tools/list` pre-pairing returns **exactly one tool**:
  `open_colab_browser_connection` — "Opens a connection to a Google Colab
  browser session and unlocks notebook editing tools. Returns a boolean…".

The post-pairing notebook tools cannot be enumerated without an authenticated
browser (INHERITED, two independent third-party reports of the official server:
`get_cells`, `add_code_cell`, `add_text_cell`, `update_cell`, `delete_cell`,
`run_code_cell` (+ `move_cell` per the fork) — to be re-verified on our first
paired session and this line updated).

### 5.3 What MCP pairing means for the PI (also ~2 minutes, per session)

1. Be **signed in to Google Colab in the default browser** of this box (that is
   the whole auth).
2. Agent calls `open_colab_browser_connection` → a Colab scratch-notebook tab
   opens → within **60 s** the tab pairs (tool returns `true`).
3. Notebook tools appear in the client; the agent edits/runs cells **in that
   tab**, with everything the UI session has — including **Colab Secrets
   (`HF_TOKEN`) and the account's GPU entitlement**.
4. Runtime type (T4) must be set **by hand in the tab** (Runtime → Change
   runtime type): the official server ships no runtime-control tool
   (MEASURED absence pre-pairing; INHERITED post-pairing — the fork added
   `change_runtime` precisely because the official lacks it).

⚠️ Open question for the first paired session: whether pairing can target an
**existing** notebook tab (e.g. our `SAM3_BACKFILL_115.ipynb`) by appending
`#mcpProxyToken=…&mcpProxyPort=…` to its URL instead of using the scratch
notebook — the frontend hash-parsing suggests yes, the server always opens the
scratch path. HYPOTHESIS until tried; if it fails, the scratch notebook can
still `%run` our Drive-mounted notebooks' lib code.

### 5.4 Known official-server sharp edges (INHERITED — third-party fork's claims, NOT re-verified)

From `SebastianGilPinzon/colab-mcp` (a fixed fork; we do NOT install it by
default — decisions on running non-Google forks are the PI's):
- clients that ignore `tools/list_changed` only ever see the one connect tool
  (Claude Code handles list_changed; our probe confirms the server advertises it);
- **orphaned `colab-mcp` processes from prior sessions** hold a port the browser
  tab still points at → "Disconnected" symptoms; our source read confirms
  single-connection + ephemeral ports make stale pairs possible. Hygiene: kill
  stray `colab-mcp.exe` before re-pairing;
- `localhost` dual-stack (IPv4/IPv6) resolution can connect the tab to the
  wrong family on Windows;
- no GPU/runtime selection via tools (matches §5.3.4).

## 6. Use-case map (the four asks), with the honest gaps

Legend: CLI = google-colab-cli (post-§4-auth), MCP = colab-mcp (paired tab).

### (a) Run SAM3_BACKFILL_115 headlessly on a T4
**CLI: YES — this is exactly the documented workflow** (PUBLISHED: bundled
README's "Workspace Notebook Execution with Drive Integration" example, plus
skill notes):
```bash
colab new -s backfill --gpu T4
colab drivemount -s backfill                      # interactive once (PI); repo appears under /content/drive
echo "import os; os.chdir('/content/drive/MyDrive/SayBouBase/raw/Projects/TanitAD/colab')" | colab exec -s backfill
colab exec -s backfill --timeout 14400 -f "G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\colab\SAM3_BACKFILL_115.ipynb"
colab stop -s backfill
```
Caveats: (i) `--timeout` default is 30 s — set it ≥ the notebook's worst cell;
(ii) ⚠️ **`google.colab.userdata.get("HF_TOKEN")` (Colab Secrets) is a
UI-frontend feature — whether it resolves in a CLI-provisioned headless session
is UNVERIFIED and doubtful.** Two clean options, both PI-decidable: attach the
UI to the same session via `colab url -s backfill --open` (secrets then served
by the attached frontend — HYPOTHESIS, cheap to test first session), or a
notebook-side fallback that reads the token from the Drive-mounted, git-ignored
`Keys.txt` in place (token never leaves Drive/VM; notebook edits belong to the
notebooks' owner, not this doc); (iii) T4 allocation is entitlement/availability
-gated (§3.3); (iv) native-Windows CLI is unsupported upstream (§3.1).
**MCP: partially** — it runs the notebook in the PI's tab (Secrets work there),
but the tab must be open and the runtime hand-picked; that is supervised, not
headless.

### (b) Monitor / retrieve outputs
**CLI: YES** (PUBLISHED): `colab status -s X` (hardware, IDLE/BUSY, last
execution), `colab log -s X -n 20` (structured events incl. keep-alive errors
with raw `response_body`), `colab exec` streams the kernel's stdout/stderr
live; `.ipynb` runs write `<basename>_output.ipynb` next to the input;
`colab download` fetches any VM file; `colab log -s X -o run.ipynb` exports a
replayable session notebook. Our notebooks additionally bank every clip to HF
with far-side verification (RUNNER.md §3), so the HF listing remains the
ground-truth progress monitor regardless of transport.
**MCP: YES within the tab** — `run_code_cell` returns outputs to the agent
(INHERITED tool list, §5.2).

### (c) Resume after session death
**CLI: YES, semantics documented** (PUBLISHED, bundled skill "Recovery"):
death is detected as 404/401 on exec → local state auto-pruned → `colab new`
and re-run. Kernel state is lost with the VM (state persists across `exec`
calls only while the session lives); wedged-kernel-but-live-VM has
`colab restart-kernel`. Our notebooks were built for exactly this
(RUNNER.md §3: every clip banked immediately, restart = "Run all") — under the
CLI the resume is `colab new --gpu T4` + re-`exec` the same notebook, and the
done-set skip makes it idempotent. **No re-auth needed** (token cached).
**MCP:** re-pair (§5.3) after any death; the notebook tab's own state follows
Colab UI semantics.

### (d) The label-lab iterate loop (edit prompts → re-run legs → judge sheet)
**CLI: YES, and kernel persistence makes it cheap** (PUBLISHED §3.2): one
session, then per iteration — edit `stack/scripts/ph0_v2.py` locally (Drive
syncs it) → `echo "import importlib, ph0_v2; importlib.reload(ph0_v2)" |
colab exec -s lab` → re-run the VLM/fusion/sheet legs by piping those cells or
re-`exec`-ing the notebook (done-set keeps it incremental) → `colab download`
or HF for the sheet. The models stay loaded between calls — no per-iteration
reload of the 4-bit VLM.
**MCP: YES, arguably the better fit under supervision** — the agent edits and
re-runs exactly the cells in question in the PI's tab while the PI watches the
review sheet; Secrets available; no notebook-side changes needed. Cost: tab +
hand-set runtime + 60 s pairing per session.

### Bottom line
- **Nothing here is Pro-gated for our T4 case**; the gates are (1) the PI's
  one-time CLI auth, (2) per-session MCP pairing, (3) T4
  availability/entitlement on the account, (4) Windows support for the CLI
  (WSL or shim-probe, §3.1).
- The CLI is the headless/automation surface; the MCP server is the
  supervised in-tab surface. They are complementary, not substitutes — and
  `colab url` even bridges them (attach the UI to a CLI session).

## 7. Artifacts of this evaluation

| artifact | where |
|---|---|
| this doc | `colab/COLAB_CLI_MCP.md` |
| ready Claude Code MCP config | `colab/MCP_SETUP.md` |
| RUNNER.md §1 supersession | `colab/RUNNER.md` |
| venv with both tools | `C:\Users\Admin\venvs\colab` (box-local, not in repo) |
| termios import-shim | `colab/win_shims/termios.py` (import-only; real terminal calls raise) |
| bundled skill/README dumps, MCP probe script + transcript | session scratchpad (`COLAB_SKILL_bundled.md`, `COLAB_README_bundled.md`, `mcp_probe2.py`, `probe_out.txt`) |

Evidence classes used: MEASURED = ran on this box 2026-08-16 with transcripts
above; PUBLISHED = Google's own repos/blog/bundled docs (quoted); INHERITED =
third-party reports incl. the fork (named inline, not re-verified); HYPOTHESIS
= explicitly flagged (native-Windows exec, Secrets-in-CLI-session, pairing an
existing tab).
