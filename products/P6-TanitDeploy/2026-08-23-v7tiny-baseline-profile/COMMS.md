# COMMS — E-DEPLOY-1

`TanitAD_DeployFlyWheel → Master Mind. 2026-08-23.`

## ⛔ ESCALATIONS — need a decision or another agent's action

| # | to | what | why it cannot wait |
|---|---|---|---|
| **C-1** | **PI / Master Mind** | **Ratify a materiality threshold for the four-family gate** (backlog D-01) | The best measured config (60.30 ms p50) has a paired 859-window gate whose `F_LONGITUDINAL` fires at effect sizes of **0.12–0.58 % of level**. With no pre-registered "how small is small enough", P6 **has no shippable config** — and cannot get one by doing more work. This is the single thing blocking the product |
| **C-2** | **whoever owns champ30k** (Architecture / Lab) | 🔴 **`champ30k` exists on ONE disk** — `thor:~/v7tiny/champ30k/`, 133 MB, 4.1 h of compute, the first collapse-free trunk | The programme's dominant measured failure mode is good work stranded outside git. The phase-0 arms were pushed to HF `Sayood/`; this one was not |
| **C-3** | **model owner (v6.py)** | **Pin the kinematic integrator to fp32** (backlog D-05) | The fix is MEASURED and **proven bit-identical**, cost ~zero. It should land with its regression test in the same commit |
| **C-4** | **EvalFlyWheel** | `distance_keeping` is **structurally unmeasurable** — `status: UNAVAILABLE`, n=0, because the ingest does not read `obstacle.offline` (present in 97.44 % of the corpus) | One of the four binding families cannot be computed at all. P6 is a consumer here and must not patch the instrument |
| **C-5** | **Master Mind** | 🔴 Deployment evidence is split across two hubs — the runbook's own evidence base, the ONNX/TRT export path and the **only real INT8 benchmark** live under *Architecture & Inference*, not *Deployment & Optimization* | A P6 reader opening the obvious folder misses half the product's evidence |

## ⭐ TRANSFER HANDOFFS (charters §7 — accept or reject with a reason, never silent)

| to | finding | what it would change |
|---|---|---|
| **Training / Architecture** | ⭐ **Triton 3.7.1 IS present on Thor.** The "no Triton ⇒ no `torch.compile`" rule is a **dev-box** fact that has been generalised to the fleet | `torch.compile` has never been tried on the primary target, and on the A40 it was the best rollout lever measured (52.89 vs graph 57.18 ms). Backlog D-04 |
| **Eval / Architecture** | **bf16 deviates 30× more than fp16 for ZERO latency gain** on v7-tiny (2.618 m vs 0.087 m at b1) | A **second independent** signal against bf16, converging with the 4060 decision-agreement result (67.2 %, 47.7 cm). The 6.76× bf16 encoder lever — the biggest in the playbook — rests on a numerics rel-err and should be confirmed in decision space or withdrawn. Backlog D-10 |
| **Architecture / Lab** | **CUDA-graph capture succeeds on the full V6Stack forward and replays bit-identically**, at every batch and both precisions | Graphs are a *training-side* lever too, and the capture succeeding proves the forward has no capture-hostile control flow at these settings |

## Status notes (facts, not verdicts)

- **`champ30k` finished** 2026-08-23 12:55 Thor-local: 30 000 steps, `elapsed_s`
  14 774.7, `summary.json {"done": true}`, `gate_verdict` **INCONCLUSIVE**,
  19 337 289 params. ⚠️ **The register moved under me mid-session**: when I
  started, `GOALS_AND_CLAIMS.md` read *"H-RANK-10 OPEN — champ30k running"*; by
  the time I wrote my rows another agent had adjudicated it **REFUTED as stated**
  (it plateaus rather than keeps gaining) with a 150-row trajectory artifact. My
  rows were added to the current file, not the one I first read. The Deploy
  FlyWheel does not adjudicate H-RANK-10 and makes no claim about it.
- **Thor was verified idle** (full `ps -eo args` grep for `python`, not a top-N
  sample) before and between every GPU job. No training was disturbed.
- The census's precondition *"Thor is currently training"* was **stale by the time
  it was written** — that is why the profiling window existed.
- **Nothing was installed** on Thor. No `uv pip install` ran.

## Deliverable manifest

| artifact | location | one place only? |
|---|---|---|
| Pre-registration (both outcomes) | `repo:products/P6-TanitDeploy/2026-08-23-v7tiny-baseline-profile/SPEC.md` | no |
| Result + verdicts | `repo:…/2026-08-23-v7tiny-baseline-profile/RESULT.md` | no |
| 5 raw JSONs (incl. the DEFECTIVE probe, kept deliberately) | `repo:…/raw/` | no |
| 6 probe scripts | `repo:…/code/` | no (also `thor:~/`) |
| **P6 product spec** | `repo:products/P6-TanitDeploy/SPEC.md` | no |
| **P6 backlog** (15 items, ordered) | `repo:products/P6-TanitDeploy/BACKLOG.md` | no |
| **Engine registry** — recipes + sha256, closes census E1 | `repo:products/P6-TanitDeploy/engines/THOR_ENGINE_REGISTRY.md` | no |
| Engine descriptors + rescued Thor manifest | `repo:products/P6-TanitDeploy/engines/` | no |
| `champ30k` checkpoint | `thor:~/v7tiny/champ30k/` | 🔴 **YES** |
| TRT engines + ONNX (10.7 GB) | `thor:~/trt_deploy`, `~/trt_d1`, `~/trt_c3`, `~/trt`, `~/trt_c2` | 🔴 **YES** — mitigated: recipes + sha256 now in git |

## Integration status

Everything is **STAGED, NOT COMMITTED, NOT PUSHED**, per the operating standard.
Nothing in this package modifies existing repo files except the
`GOALS_AND_CLAIMS.md` register rows required by the same-turn rule.
