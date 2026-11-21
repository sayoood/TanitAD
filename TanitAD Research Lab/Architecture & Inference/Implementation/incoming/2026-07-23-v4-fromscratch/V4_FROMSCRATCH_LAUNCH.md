# v4 FROM-SCRATCH fallback — launch-ready spec

**Date:** 2026-07-23 (Europe/Berlin, UTC+2) · **Type:** trainer implementation + validation, pre-staged.
**Status:** CODE STAGED + VALIDATED (dev-box + pod-import). **NOT LAUNCHED.** This fires *only* if
v4.2b's Phase-B canary runs away, and only on Sayed's go (§17). **No GPU-day committed here.**

**Number discipline (CLAUDE.md).** Evidence class on every load-bearing claim: **MEASURED** (ours +
artifact) · **PUBLISHED** · **INHERITED** (not re-verified) · **ESTIMATED** · **HYPOTHESIS**.

---

## 0. Why this exists (one paragraph)

v4 / v4.1 / v4.2 / v4.2b all **warm-start the trunk from v1** (`flagship4b-speedjerk-30k`) and every one
of them stresses the world model: coupling a new anchored-diffusion planner's gradient into v1's
**already-prediction-converged** WM yanks it off-manifold (v4 hot-trunk: canary **0.452 → 1.3 by ~step
3500**, MEASURED, MODEL_REGISTRY / [PM]; v4.1 "fixed" it only by *starving* the planner, `lam_mult`
1.5e-5). The original **v1 avoided this by training its WM + its (simple) planner JOINTLY FROM SCRATCH** —
they co-evolved to a joint optimum and the intent-free canary stayed **0.42** (MEASURED, MODEL_REGISTRY
§1.2). **The fallback is: train v4 the v1 way — WM + the anchored-diffusion planner from random init, no
warm-start.** v1 is the existence proof that from-scratch joint training does not degrade the WM (there is
no pre-converged manifold to fall off of).

---

## 1. The one-line launch command (ready to fire — run on the pod, NOT from an agent)

```bash
PYTHONPATH=/workspace/TanitAD/stack python3 scripts/train_flagship_v4.py \
  --from-scratch \
  --train-cache /workspace/data/physicalai_phase0/_epcache/physicalai-train-e438721ae894 \
  --val-cache   /workspace/data/physicalai_phase0/_epcache/physicalai-val-0c5f7dac3b11 \
  --anchors-dense /workspace/experiments/flagship_v4_anchors_dense.pt \
  --out /workspace/experiments/flagship-v4-fromscratch \
  --labels v3 --lambda-plan sched --phase-a-steps 2000 --phase-b-steps 8000 \
  --strategic full --long-horizon-k 50 \
  --steps 30000 --gate-step 10000 --batch 16 --accum 4 \
  --lr-head 1e-4 --lr-trunk 1e-4 --lam-mult-floor 0.25 \
  --warmup 2000 --workers 4 --eval-every 500 --save-every 1000 \
  --eval-episodes 40 --rollout-k 4 --seed 0 --device cuda
```

Preview it first with `--print-launch --from-scratch …` (prints the reconstructed command + the §17
preflight; it must read `PREFLIGHT: OK`).

**This differs from the v4.2b launch in EXACTLY ONE thing:** `--from-scratch` replaces
`--trunk /workspace/experiments/flagship4b-speedjerk-30k/ckpt.pt`. Everything else — labels, λ_plan
schedule, phases, strategic, steps, gate, **effective batch 64 (16×4)**, `lr_trunk 1e-4`,
`lam_mult_floor 0.25`, rollout-k, eval cadence — is byte-identical to the v4.2b arm. That single-lever
diff is deliberate: **maximal attributability.** If v4.2b's canary runs away and this one does not, the
warm-start coupling is confirmed as the cause. ⚠️ Launch-time: `diff` this against the *actual* v4.2b
`config.json` to confirm no other flag drifted since this was written.

**Parity:** the corpus stays `physicalai-train-e438721ae894` (2376 eps, skip-hash `f09e44db`); the
trainer refuses any `--train-cache` that does not reference the key (`_assert_parity`). From-scratch
changes the *weights' init*, never the episode set.

---

## 2. What was implemented (`stack/scripts/train_flagship_v4.py`)

The insight that made this small: **`WorldModel(cfg)` and `build_grounding()` already random-initialize
every trunk/grounding tensor.** "From scratch" is therefore simply **not calling `_warmstart_trunk`** —
no new init code, no architecture change. Changes:

| # | change | where |
|---|---|---|
| 1 | `--from-scratch` flag (and the `--trunk none` sentinel) → `_is_from_scratch(a)` | `build_parser`, new helper |
| 2 | `train()`: `--trunk` no longer required when from-scratch; skips `_warmstart_trunk`, random-init trunk+grounding, `trunk_step=-1`, loud `[v4][from-scratch]` banner | `train()` |
| 3 | `config.json` provenance records `trunk.init="from-scratch (random)"` + a `from_scratch_schedule_note` (the λ_plan/floor reasoning, §4 below) | `train()` |
| 4 | `preflight_asserts`: `--from-scratch` + a **real** `--trunk` is a hard conflict (fail before a GPU-day) | `preflight_asserts` |
| 5 | `_staged_command` / `--print-launch`: emit `--from-scratch`, omit `--trunk`; the launch banner states the from-scratch regime | `_staged_command`, `main` |
| 6 | `--real-smoke` honors from-scratch (no warm-start) | `real_smoke`, `main` |
| 7 | 5 new unit tests (flag+sentinel detection, preflight-clean-without-trunk, trunk-conflict, staged-command, **not-frozen gate passes on a random-init real WorldModel**) | `stack/tests/test_train_flagship_v4.py` |

**The not-frozen launch gate passes trivially from scratch** (mission item 1): `_assert_trunk_trainable`
requires every trunk param to `require_grad` and sit in the AdamW `trunk` group at `lr_trunk>0` — all
true by construction from random init (test `test_from_scratch_trunk_is_random_and_passes_not_frozen_gate`
asserts `not_frozen=True`, `trunk_tensors_frozen=0`).

---

## 3. Validation (mission item 3)

### 3.1 Dev-box (CPU) — DECISIVE from-scratch training proof

| proof | result | evidence class |
|---|---|---|
| `pytest -q` (full stack) | **786 passed, 2 skipped** — nothing broke | MEASURED |
| `test_train_flagship_v4.py` (9 existing + **5 new from-scratch**) | **14 passed** | MEASURED |
| `--smoke` (joint loss assembly, random init, phases A/B/C) | finite + **dropping**: total 14.37→13.07, wm 3.53→3.07; λ_plan 0→0.5→1.0 | MEASURED |
| `--smoke-loop` (**the real `_training_loop`, random init**) | see below | MEASURED |

`--smoke-loop` runs the identical multi-day loop function on toy episodes from **random init** — it *is* a
from-scratch loop proof:
- **canary baseline `1.5189`** — high/untrained, exactly the from-scratch signature (warm-start starts at
  ~0.45); and the canary then **drops monotonically 1.385 → 1.312 → 1.232 → 1.179 → 1.165** as the WM
  co-evolves from random init. **The WM improves; it does not collapse.**
- λ_plan controller **down-only, held at floor 0.25, never 0**; milestone `ckpt_step3.pt` archived;
  checkpoint save→resume **bit-exact** (`mult` restored, step advances).
- `gnorm_encoder`/`gnorm_predictor` logged >0, `eff_batch = batch×accum` exercised.

### 3.2 Pod1 (`tanitad-pod`, A6000, CUDA torch 2.4.1+cu124) — import proof

All 7 v4 modules were synced to pod1 and `train_flagship_v4` **imports cleanly on the pod's CUDA torch**;
`_is_from_scratch(--from-scratch)` → `True`; `--print-launch --from-scratch` → `PREFLIGHT: OK`. The
real-256px `--real-smoke` then stopped at `ModuleNotFoundError: tanitad.lake` — the **on-the-fly v4
label-minting path** (`FlagshipV4Dataset → v4_labels → tanitad.lake.vocab`) needs the `tanitad.lake`
package, which pod1's stale `main` checkout lacks (and which the cross-pod-migration guidance says to
*skip* syncing). **This gap is orthogonal to the from-scratch change** — warm-start v4.2b imports the exact
same `v4_labels`/`tanitad.lake` chain, so the properly-provisioned pod that runs v4.2b already satisfies
it. The from-scratch delta (skip `_warmstart_trunk`) is fully proven by 3.1; the pod merely confirms the
code imports and the arg path executes under CUDA. ⚠️ Launch-time: the target pod must have `tanitad.lake`
(true for any v4.2b-class pod) — or precompute labels once (`v4_labels.build`) and index them.

---

## 4. The λ_plan / floor tension — a launch-time decision (mission item 2)

**Flagged, not silently assumed.** From-scratch changes the λ_plan/floor tension because **there is no
pre-converged WM to protect** — the whole job the canary controller does in warm-start (guard a converged
WM from planner-gradient degradation) does not exist here.

**What actually happens from scratch (MEASURED in `--smoke-loop`):** the canary controller's baseline is
the **step-0 UNTRAINED canary (high, 1.52 in the smoke)**. A healthily co-evolving WM's canary only
*improves* below that (`delta = canary − baseline < 0`), so the controller reads `ok` and holds λ_plan on
its pure schedule up to 1.0. **The controller is inert by construction**, and the floor
(`--lam-mult-floor 0.25`) is a no-op safety net in this regime. WM protection, to the extent it is needed,
comes from `lr_trunk` (1e-4) and the step-10000 gate — never the controller.

**Decision taken — default to the current schedule + floor, UNCHANGED, and here is why:**
1. **Attributability.** Keeping phases (A 0-2000 / B 2000-8000 / C 8000+) and floor identical to v4.2b
   makes from-scratch a **one-lever** change (trunk init only). Any other simultaneous knob change would
   confound the "was it the warm-start?" read this fallback exists to give.
2. **Phase A is retained on purpose.** λ_plan=0 for steps 0-2000 lets the *random* WM establish a
   predictive latent on its own WM losses **before** the planner gradient couples in during the Phase B
   ramp. From scratch this is arguably more useful than in warm-start (it gives structure to couple into),
   and it costs nothing.
3. The floor being inert is **fine** — it cannot starve a planner it never cuts (the v4.1 failure needed a
   *converged* WM whose canary a live planner gradient could breach; from scratch the baseline is the
   untrained canary, which the co-evolving WM sits below).

**Alternative a launcher may weigh (documented, not chosen):** `--lambda-plan 1` (full joint from step 0)
is the *closest* match to v1's actual regime (v1 had no λ_plan curriculum — WM and planner trained together
from step 0). If the goal is to reproduce v1 as literally as possible rather than to stay one-lever from
v4.2b, `--lambda-plan 1` is defensible. **Default stays `sched`** for the attributability reason (1). This
is a genuine launch-time call — make it explicitly, don't let it default silently.

---

## 5. The honest tradeoff

**From-scratch loses the v1 warm-start shortcut but sidesteps the coupling degradation.** Concretely:

- **Cost.** Warm-start hands v4 a WM that already spent 30k steps converging to canary 0.452. From scratch,
  the 30k budget must **re-converge the WM *and* fit the planner jointly**. So at a matched 30k the
  from-scratch WM may be *less* converged than a (non-degraded) warm-started WM would have been — this buys
  robustness by spending capacity/steps on WM convergence the warm-start got for free. HYPOTHESIS: the 30k
  gate may read a higher canary than v1's 0.452 simply from fewer effective WM-only steps; judge it against
  the gate card, not against 0.452 directly.
- **Benefit.** It **cannot** exhibit the warm-start coupling degradation (v4's 0.452→1.3, v4.1's planner
  starvation) because there is no pre-converged optimum for the planner gradient to knock the WM off.
- **The existence proof.** v1 trained WM + (simple) planner jointly from scratch and held canary 0.42
  (MEASURED). v4's operative planner is REF-C's anchored diffusion — a *better-behaved* planner than v1's
  tactical head (tight-deviation, anchor-vocabulary-clamped; 2026-07-23 planner-bottleneck synthesis §2),
  so from-scratch joint training has, if anything, a friendlier planner to co-evolve with than v1 did.

**Net:** this is a *fallback*, not the default arm. It trades the warm-start head-start for immunity to the
one failure mode that has dogged every warm-start v4. Fire it only if v4.2b's Phase-B canary confirms that
failure mode is unavoidable on this trunk.

---

## 6. Wall-clock estimate — ~53 h / 30k (ESTIMATED, MEASURED basis)

From-scratch has **identical per-step compute to v4.2b** (same arch, same batch 16 × accum 4 = eff 64;
random vs warm init does not change FLOPs), so it inherits v4.2b's wall-clock.

Basis: v4.1's in-loop log (`…/2026-07-23-v41-10k-gate/`, `elapsed_s` deltas) measures **~1.57 s/step at
accum 1** (~78 s per 50-step log interval, MEASURED). At **accum 4** (the eff-batch-64 config) that is
~4× the forward/backward per optimizer step → **~6.3 s/step**, and `6.3 × 30000 ≈ 189000 s ≈ 52.5 h`,
plus eval/canary overhead every 500 steps → **~53 h / 30k**. Consistent with the design's per-step ceiling
(G0 `s/step` budget, V4_FLAGSHIP_DESIGN §9). Gate read at step 10000 (~18 h) is the first decision point.

---

## 7. Fire conditions (recap) — this is a conditional fallback

- **Trigger:** v4.2b's **Phase-B canary runs away** (the warm-start coupling degradation reproduces).
- **Gate:** Sayed's explicit go (§17). An agent never launches; the orchestrator runs the §1 command on a
  v4.2b-class pod (has `tanitad.lake`; has the parity **val** cache + the dense anchors).
- **Read:** the step-10000 gate on `Project Steering/Gates/flagship-v4.card.json` (needs the v4 held-out
  eval driver — the standing blocker on every v4 path, per the 2026-07-23 synthesis §7).
- **Non-committal:** nothing here spends a GPU-day; only the code path is implemented + validated.

---

## Deliverable manifest

| artifact | where | status |
|---|---|---|
| from-scratch trainer path (`--from-scratch` / `--trunk none`) | `stack/scripts/train_flagship_v4.py` | **STAGED** (git add; not committed, not pushed) |
| 5 new from-scratch unit tests | `stack/tests/test_train_flagship_v4.py` | **STAGED** |
| this launch spec | `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-23-v4-fromscratch/V4_FROMSCRATCH_LAUNCH.md` | **STAGED** |
| pod-side validation (transient) | `tanitad-pod:/workspace/TanitAD/stack/` (7 v4 `.py` files synced for the import proof, uncommitted) + `tanitad-pod:/tmp/v4files.tgz` | left in place (pod on a stale `main`; harmless, disclosed — NOT deleted per the shared-pod rule) |

**Validation artifacts (this session, MEASURED):** full `pytest -q` = 786 passed / 2 skipped;
`test_train_flagship_v4.py` = 14 passed; `--smoke`, `--smoke-loop`, `--print-launch --from-scratch`
(PREFLIGHT OK) all green on the dev box; `train_flagship_v4` imports on pod1 CUDA torch 2.4.1+cu124.

**Escalation:** the target launch pod must carry `tanitad.lake` (every v4.2b-class pod does) and the
parity **val** cache + dense anchors — pod1 lacked all three (stale `main` checkout); it is **not** a v4
launch pod as-is. No commit, no push, no GPU launched.
