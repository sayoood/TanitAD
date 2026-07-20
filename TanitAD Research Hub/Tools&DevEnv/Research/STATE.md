# STATE — Tools&DevEnv

LAST_RUN: 2026-07-20 (W4, Monday) — branch `agent/tools-devenv-20260720`
  (worktree `C:/Users/Admin/wt-tools-0720`, off `c4d8451` = `agent/tools-devenv-20260718`)
QUALITY: full (G-A…G-I + G-T1 met; 2 measured experiments — `ci_gate` v2 on two real trees
  + the CUDA device-parity tripwire on the RTX 4060)
RESOURCE (G-I): **local RTX 4060 8 GB** (CUDA parity probes + I8 latency proxy, torch
  2.11+cu128) + dev-box CPU (2 full pytest trees) + web sweep. ~2.3 h wall, **$0**.
  Why not the eval pod: this run's experiments are (a) a device-*parity* check, which needs
  a GPU but not a big one — the 4060 answers it exactly as well as an A40 and costs nothing —
  and (b) a CI gate, a pure git/pytest workload. The A40 would have idled. The pod-worthy
  item this run surfaced (AlpaSim closed loop) was already answered NO-GO on infra grounds
  by the 2026-07-19 investigation — it needs a docker-capable host, not more GPU.

## ESCALATION — D-026 debt is GROWING, and the worst class is NEW
`python tools/session_guard.py --repo <Drive tree>` (2026-07-20). **For the orchestrator:**

| Class | 2026-07-18 | **2026-07-20** |
|---|---|---|
| uncommitted hub deliverables | 5 | **30** |
| **uncommitted `stack/` paths** | (not checked) | **40 — 22 UNTRACKED** |
| unmerged `agent/*` branches | 9 | **11** |
| stale INTAKE verdicts (>3 d) | 5 | **8** |

- **Priority: the 40 uncommitted `stack/` paths.** 12 whole test modules (~135 tests),
  9 `tanitad/lake/*` + `eval/ckpt_compat.py` + `train/decorr.py`, 18 modified core files
  (`config.py`, `fourbrain.py`, `predictor.py`, `refa.py`, `flagship_losses.py`, 10 scripts).
  **In no commit, on no branch, anywhere** — one `git clean` from gone. Strictly worse than
  an unmerged branch, which is at least pushed. Found by a 396-vs-531 collected discrepancy
  between my worktree and the Drive tree; `session_guard` v1 called that tree clean because
  it only looked at hub prefixes. Fixed this run.
- **Uncommitted hub deliverables incl. whole packages:** Benchmarks & Eval's
  `2026-07-19-alpasim-closedloop-v1/` **with its results JSONs**; Architecture's
  `V3_HIERARCHICAL_PLANNING_DESIGN.md`, `V3_GOAL_VOCABULARY_V1.md` + four 07-19 notes;
  Data-Eng's `TANITDATASET_V1_STRATEGY.md` + three surveys; `HYPOTHESIS_LEDGER.md`;
  `DECISIONS.md`; `PROJECT_STATE.md`; the `2026-W33` progress report.
- **Newly stale INTAKEs:** `2026-07-15-h15-logging-fidelity`, `2026-07-15-baseline-floor`,
  `2026-07-15-pandaset-loader` (5 d). Still unfilled at 11 d: `lal-v2-anticipation`,
  `physicalai-r1-selection`, `models-predictor-failfast`, `testsuite-io-profiling`.

## HANDOFF
No blocking handoff. Two things every agent should adopt **now**:
1. `python tools/ci_gate.py --rootdir stack --gpu-smoke require` before push (on a CPU-only
   box use `--gpu-smoke warn` — it prints and never blocks).
2. `python tools/session_guard.py` at session end (unchanged), which now also names
   uncommitted `stack/`/`tools/` work.

**Intake ledger cleaned:** `2026-07-17-ci-gate/` verdict written by me and the package
**withdrawn as SUPERSEDED** — `ci_gate` is repo-level tooling, so it belongs in `tools/`
(no intake), the same class as `session_guard`. The duplicated v1 code was deleted from the
intake dir so there is one source of truth; `INTAKE.md` stays as the record. Lesson: decide
the target directory *before* opening an intake — `tools/` = no intake, `stack/` = intake.
Still open for the orchestrator: `2026-07-09-testsuite-io-profiling/` (11 d, this
discipline's own — KB says "shipped via intake" but no verdict was ever written).

## Done this run
- **`tools/ci_gate.py` v2** (backlog P0#2): **SUITE_MANIFEST** — 16 load-bearing modules
  pinned to a collected-count floor, because a named-node tripwire only guards nodes somebody
  thought to name, while whole modules vanish silently in a six-agent tree. Plus `--min-total`
  (390), `--gpu-smoke off|warn|require`, `--json`. Skips stay green **unless a whole module is
  skipped**. Measured: **396 / 39.0 s** (off-Drive worktree) and **531 / 60.2 s** (Drive tree),
  both **GATE PASS**; tall pole 8.02 s. **Sharding NOT needed — 5× under the 5-min ceiling.**
- **`tools/gpu_tripwire.py`** — closes a hole I measured before writing any code: **the whole
  396/531-test suite is CPU-only** (`grep -rl cuda stack/tests` is empty) while every trainer,
  eval and deploy tick runs on a GPU. Four probes on the real model, RTX 4060, 1.7 s: encode
  parity **9.54e-07**, imagine **7.15e-07**, I2-on-device **1.66e-07**, 0 non-finite grads;
  batch-1 encode **0.85–1.43 ms** (I8 proxy). Falsifier at `tol=0` proves the probes can fail.
- **`session_guard` source check** — see ESCALATION. Untracked listed separately from modified.
  A second bug caught by its own falsifier: `git status --porcelain` collapses a wholly
  untracked directory to one `?? stack/` row → `--untracked-files=all`.
- **Two bugs caught pre-ship by falsifiers**, not in review: the porcelain `-uall` collapse,
  and suite-greenness treating a legitimate `skipped` as rot (the live run failed on
  `test_scena::test_minilm_search_ranks_sc01`).
- **Backlog text corrected against measurement:** `test_eval_behavior` is **13**, not 22;
  **`test_calib_r1.py` does not exist** (folded into `test_calib`), so the "calib trio" is a pair.
- `tools/tests/` **57 falsifiers, 15.5 s**. `tools/README.md` rewritten for all three tools.
- **Literature:** Rerun **0.34.1 Viewer MCP** (agents can verify their own renders — GO, →P0.1);
  **JetPack 7.2 correction** (Orin is in the JetPack 7 line now, plan the export against 7.2 not
  7.1); the **L0–L4 world-model-as-oracle admissibility ladder** (2607.07196 — the citable form
  of our open-loop ⊥ closed-loop result); **DynaDreamer** (2607.13410 — the published relative of
  our v0-channel fix); **Orbis 2** (2607.15898 — diffusion-forcing→teacher-forcing schedule);
  **TerraZero** (2607.13028 — 1.3 M steps/s, no code → WATCH). Nothing new on AlpaSim/CARLA/
  Bench2Drive or dev-tooling releases.
- Research note `2026-07-20-ci-gate-v2-suite-manifest-gpu-tripwire-and-the-uncommitted-stack.md`;
  KB +12 deltas; BACKLOG re-prioritized (P0#2 done, P1.0 retired, 4 new findings-driven items).

## Open threads / proposals to raise
- **P1.0 AlpaSim retired — it became an INFRA ask.** The 2026-07-19 investigation measured a
  hard NO-GO: the eval pod is an unprivileged container with **no nested container runtime**
  and AlpaSim's NuRec renderer is image-only (`nvcr.io/nvidia/nre/nre-ga:26.04`). Everything
  else is GO and the policy adapter is written. **For Sayed: a docker-capable GPU host is now
  the single blocker on an AlpaSim closed loop** — same infra class as the pending graphics-pod
  ask, worth deciding once. Fallback if the answer is no: TerraZero-class rendering-free sim.
- **`ci_gate` is still skippable** — nothing runs it automatically; it is a discipline, not a
  gate. Wiring it into a real pre-push/session-end hook is new backlog P0.2.
- **`gpu_tripwire` is fp32+eager only** — Prod-Opt's CUDA-graph deploy tick and every bf16
  training path remain unguarded. New backlog P0.3 (measure the bf16 deviation before setting
  its tolerance; do not guess it).
- **Gate timings are contention-sensitive**: the same suite measured 39.0 s / 8.02 s clean vs
  65.0 s / 14.90 s beside a second pytest process — within 0.1 s of a false slow-test failure.
  The 15 s per-test budget therefore stays; tightening it toward the original 6 s intent needs
  either fixture work or load detection (backlog P0.4).
- **RESIM_ROADMAP.md is still missing** (third run carrying this) — mission P1 says the
  TanitResim roadmap lives there. The Rerun 0.34.1 Viewer-MCP upgrade is the natural anchor to
  write it around next run, together with the 3-arm view (REF-B is live).
- Note to Architecture/Prod-Opt: DynaDreamer's rollout-time ego-dynamics propagation and
  Orbis 2's forcing schedule are both cheap, measure-first levers on the longitudinal 83 %.

## Prior handoff (2026-07-09, still open)
- **Sayed ~1 click:** pin `stack/` to Drive "Available offline" → removes the cold-I/O tax.
  Fresh datapoint: an **off-Drive worktree runs 396 tests in 39.0 s** while the Drive tree runs
  531 in 60.2 s (0.099 vs 0.113 s/test) — the gap has narrowed since the 40.6 s-cold/10.7 s-warm
  measurement, but off-Drive is still the faster place to work. Verification tool ready
  (`profile_testsuite.py`).
- CARLA camera pixels: graphics-capable pod recreation (`NVIDIA_DRIVER_CAPABILITIES=all`, gate
  on `vulkaninfo`) — NOT urgent; milestone 1 (LAL/OKRI/LOPS) needs no pixels.
