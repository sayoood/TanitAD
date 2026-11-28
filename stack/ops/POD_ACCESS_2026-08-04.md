# Pod access from a cloud session — what is actually blocked, and what to do

> ## ⛔ THE PODS ARE GONE — but read the ⚠️ key note before filing this away. (2026-08-16)
> Both endpoint rows are dead machines: pod4 `69.30.85.48:22192` and tanitad-new
> `69.30.85.106:22039`, with their `POD4_PORT` / `PODNEW_PORT` overrides. The copy-paste Option-A
> block still names `RUN_DIR=/workspace/experiments/flagship-v1arch-v2bal-30k`, two model
> generations back. The whole RunPod fleet was released 2026-08-15; the live fleet is
> **Thor + the dev box**.
>
> ⭐ **Keep the MEASURED core.** *"All non-443 TCP silently dropped, with `github.com:22` as the
> control"* is doctrine about the cloud-session gateway, not about any particular pod, and it is
> still the reason a cloud session cannot reach a training box directly.
>
> ⚠️ **OPEN, and a PI call: `TANITAD_POD_SSH_KEY`.** This document instructs installing a live SSH
> **private key** as a repo secret behind a `pods` environment, and `.github/workflows/pod-exec.yml`
> is still checked in. The hosts it unlocks no longer exist, so the secret now protects nothing and
> only carries risk. ⇒ **Rotate/remove the secret and retire `pod-exec.yml`** — flagged here rather
> than done, because credentials and workflow retirement are the PI's.

**MEASURED 2026-08-04 from the Claude Code cloud session** (probes run in-session;
the gateway's own `__agentproxy/status` failure log is quoted where it recorded one).
Evidence class: **MEASURED**. Supersedes the INHERITED claim in the resumption handoff.

## The measurement

| destination | result |
|---|---|
| `api.runpod.io:443` | **403** — gateway-recorded: `connect_rejected … policy denial` |
| `rest.runpod.io:443`, `api.runpod.ai:443` | **403** |
| `huggingface.co:443` | **403** |
| `api.wandb.ai:443` | **403** |
| `github.com:443`, `raw.githubusercontent.com:443` | **reachable** |
| PyPI / npm / crates (in `no_proxy`) | reachable |
| `69.30.85.48:22192` (pod4 SSH) | CONNECT answered `200` locally, **then nothing** |
| `69.30.85.106:22039` (tanitad-new SSH) | same |
| `github.com:22` — a host that certainly serves an SSH banner instantly | **same: `200`, then nothing** |

## ⚠️ The correction that matters

The obvious reading — *"the pods' SSH ports are denied, allow-list them"* — is **wrong**,
and the `github.com:22` row is what proves it. GitHub's SSH is unquestionably up and
answers with a banner immediately; through this proxy it returns nothing, exactly like
the pods. **All non-443 TCP is silently dropped.** The pod ports are not individually
denied; there is no raw-TCP egress at all.

Two consequences:

1. **Allow-listing the pod IPs would NOT buy a shell.** The egress path is an
   HTTPS-CONNECT proxy, not a router.
2. ⛔ **No SSH private key belongs in an environment variable** to chase one anyway —
   env vars are readable by anyone using the environment. This is not a
   "we could if we had the key" situation; the transport is absent.

⚠️ Note also that the local `200 Connection Established` is answered **before** the
upstream decision, so it is **not** evidence a port is permitted. Only the gateway's
recorded rejection, or actually receiving bytes, settles it. (Same family as the `df`,
`tegrastats` and `memory.usage_in_bytes` traps in CLAUDE.md: a probe reporting the
wrong scope is worse than no probe, because it looks like an answer.)

## Option 0 — a GitHub-hosted runner IS the shell (recommended)

`github.com` / `api.github.com` are reachable, and a GitHub runner has ordinary
internet egress, so it reaches the pods fine and its output comes back through the
API. Two workflows, split by blast radius:

- `.github/workflows/pod-telemetry.yml` — **read-only by construction**, every 15 min.
- `.github/workflows/pod-exec.yml` — **real `ssh`**, behind `environment: pods` with a
  required reviewer, so every run needs approval.

⭐ **This is also the right place for the key.** An SSH private key must never go in an
environment variable (readable by anyone using the environment); **GitHub Actions
secrets are encrypted at rest, injected only into the job, and masked in logs.**

**Setup is ONE secret**, because host/port are an IP and a port, not secrets —
they default to the console's *"SSH over exposed TCP"* values, CONFIRMED
2026-08-04 from the PI's console screenshots with both pods green:

| pod | console name / id | endpoint |
|---|---|---|
| pod4 | `interesting_gray_ant` / `v9ni8rpan3qyn3` | `root@69.30.85.48 -p 22192` |
| tanitad-new | `added_red_guine…` / `szv0r2e1qgjq09` | `root@69.30.85.106 -p 22039` |

1. **Create the `pods` environment FIRST** (Settings → Environments), self as required
   reviewer. ⛔ Until it exists `pod-exec.yml` is a loaded surface, not a control.
2. Add secret **`TANITAD_POD_SSH_KEY`** = contents of `~/.ssh/tanitad_pod`.
   ⚠️ That key, **not** the console's `id_ed25519` — the console key does not work.

⚠️ Optional overrides `POD4_PORT` / `PODNEW_PORT` exist because **a RunPod volume
resize stops the pod and reassigns its SSH port** (symptom: `Connection refused`, not
`timed out`). Set them if the defaults stop working.

## Option A — pod-side push, also with NO policy change

GitHub is already reachable, so **push from the pod** instead of pulling from outside.
`stack/ops/pod_telemetry_push.sh` drains GPU, cgroup `rss`/`failcnt`, a real `dd` disk
test, trainer `ps`, `metrics.json`, checkpoint mtimes and the trainer-log tail to an
**orphan** branch `telemetry/<pod>` every 5 minutes.

Run once per pod. **RunPod's browser terminal is enough — no SSH needed**, which is why
this works from a phone:

```bash
cd /workspace/TanitAD && git pull
export GITHUB_TOKEN=<fine-grained PAT: contents:write on sayoood/TanitAD ONLY>
export POD_NAME=pod4                     # tanitad-new on the other box
export RUN_DIR=/workspace/experiments/flagship-v1arch-v2bal-30k
export TRAIN_LOG=$RUN_DIR/train.log
nohup nice -n 19 bash stack/ops/pod_telemetry_push.sh >/tmp/telemetry.log 2>&1 &
```

Then I read it with `git fetch origin telemetry/pod4`.

- ⛔ **Adds no GPU/RAM load to a training pod** (the CLAUDE.md invariant): `nice -n 19`,
  file reads, and `nvidia-smi --query` — a driver query, not a kernel. No torch import,
  no GPU allocation.
- Orphan branch ⇒ a telemetry commit can never touch program code.
- ⚠️ Use a **fine-grained** token scoped to this repo. A classic `repo`-scope PAT on a
  shared pod is far more access than a log drain needs.

## Option B — control-plane visibility (needs a policy change)

Environment → **Custom** network access → allow `api.runpod.io`, and set a **read-only**
`RUNPOD_API_KEY`.

Buys: pod up/down, GPU utilisation, uptime — i.e. *"is it alive"*.
Does **not** buy: step counts, losses, `metrics.json`, checkpoints. **Option A is what
answers "how is training going"**; B only answers "is the box on".

⚠️ A read-only API key in an env var is still a secret readable by anyone using the
environment — much smaller blast radius than an SSH key, but not zero.

## Option C — a real shell

Not available from this environment at any policy setting short of raw-TCP egress.
For one-off pod commands use **RunPod's browser terminal**, which works from a phone.

⚠️ If a shell is ever restored, the CLAUDE.md pod traps still bind — in particular
`pgrep -f`/`pkill -f` self-matching (kill by explicit PID), the `PYTHONPATH=` requirement,
and `ssh -n` inside any heredoc or pipe.
