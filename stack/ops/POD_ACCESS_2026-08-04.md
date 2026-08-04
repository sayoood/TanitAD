# Pod access from a cloud session — what is actually blocked, and what to do

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

## Option A — training monitoring, with NO policy change (recommended)

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
