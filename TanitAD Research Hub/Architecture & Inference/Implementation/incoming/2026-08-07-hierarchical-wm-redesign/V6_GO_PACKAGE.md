# V6 GO PACKAGE — the v6 ladder's OPERATOR RUNBOOK
### §2 is the part you paste at 3 a.m. It is generated, not written. Decisions: `PI_DECISIONS_2026-08-12.md`.

> ## ⛔ STATUS — 2026-08-16. §2 WAS REWRITTEN; §1's "BLOCKED" VERDICT IS SUPERSEDED.
>
> **S-W IS LIVE.** It is training on **Thor**, not on a provisioned A40 pod — step ~6,400 of
> 30,000 at a **MEASURED 27.18 s/step**, trainer **pid 25477**, ops loop **pid 29587**, run dir
> **`~/experiments/v6F-SW-30k`**. *(INHERITED from the coordinator's fleet read, 2026-08-16; not
> re-probed here — Thor holds the only GPU and this turn was CPU-only by instruction.)*
> ⇒ **Do not follow §2.1 to "start S-W". It started.** The next launch is **S-T**, and it launches
> from `v6_chain.py`.
>
> **⛔ THE LAUNCH LINES IN THIS DOCUMENT ARE NO LONGER HAND-WRITTEN.** They are the output of
> **`stack/scripts/v6_chain.py commands`**, and `stack/tests/test_runbook_commands.py` FAILS THE
> SUITE if this file and that generator disagree. If you are reading a line here, it parsed
> against the trainer's real `build_parser()` the last time `pytest` ran.
>
> **SEL-1 is REFUSED** (E-WC2, 2026-08-16). S-T's default is `--selector none`, `--w-select 0`,
> and a selector arm is **refused at launch**, not merely left out of the plan. §2.6.

**Tier stamp.** Nothing produced by this trainer is quotable as driving performance. Capability
claims are **T1** (`taniteval/tools/t1_eval.py`) with the **four metric families** and the
**paired episode-cluster bootstrap** (`taniteval/ci.py`). `EVAL_DOCTRINE.md`.

**Sources.** Model facts cite `Project Steering/MODEL_REGISTRY.md §x` or a raw gate JSON.
Rows marked *(mine 2026-08-12)* were measured by this doc's original author; rows marked
*(mine 2026-08-16)* were re-measured on the dev box during the §2 rewrite.

---

## 1. GO / NO-GO — every precursor for S-W

| # | precursor | state | evidence |
|---|---|---|---|
| 1 | v6 model, staged trainer, tests exist in the repo | ✅ **DONE** | `stack/tanitad/models/v6.py` · `stack/scripts/train_v6_staged.py` · `stack/tests/test_v6_staged.py`, all in HEAD `2b8d09e` |
| 2 | the test suite is green | ✅ **DONE** | **MEASURED (mine 2026-08-12): `pytest tests/test_v6_staged.py -q -W error::UserWarning` → 80 passed, 0 warnings** |
| 3 | it imports, builds and takes real optimiser steps | ✅ **DONE** | **MEASURED (mine 2026-08-12): `--stage S-W --dry-run --device cpu` → two `[v6 dry N]` rows with live O1/O2/O3/O5/O6 terms, then `dry-run OK`** |
| 4 | sub-300M invariant enforced **before** GPU time | ✅ **DONE** | **MEASURED (mine): `[v6] params 87.89 M / budget 300 M · arm shared-encoder+adapters`** — `build_stack_from_args` refuses over budget pre-launch |
| 5 | X3 gradient isolation is a real autograd probe and passes | ✅ **DONE** | **MEASURED (mine): `[v6] X3 isolation pass=True violations={'planner_to_encoder': 0, 'tactical_to_below': 0, 'strategic_to_below': 0}`** |
| 6 | the 6 s horizon is trainable — v4's `max_horizon` 20 is NOT inherited | ✅ **DONE** | **MEASURED (mine): derived `max_horizon` = 20 (S-W as specced, S-S) / 60 (S-T, S-J)** from `train_v6_staged.py:1071–1084`; the trainer prints its choice beside v4's |
| 7 | §4b seam-free by construction — ONE 60-step (a,κ) rollout, bands are slices | ✅ **DONE** | `V6Config` refuses a band gap/overlap at construction; **MEASURED (mine): `plan_steps` 60, `dt` 0.1, bands (0,2) and (2,6)** |
| 8 | resume + done-marker guards (the supervisor-resurrection defence) | ✅ **DONE** | `V6_TRAINER_DESIGN §3.4b`: refuses a DONE dir without `--force-rerun`; refuses `--resume off` over an existing `ckpt.pt`; writes `summary.json {"done": true}` in the same turn it finishes |
| 9 | **the corpus caches exist on the target pod** | 🟡 **UNVERIFIED** | this agent has **no pod access**. `/workspace/data/physicalai-{train,val}-…-w120-256x640cyl` is INHERITED from `V6_TRAINER_DESIGN §3.1`. ⇒ **§2.0 STEP ZERO below verifies it** |
| 10 | **a free A40 pod for 7–12 days** | ⛔ **BLOCKED — PI (D2)** | `OVERNIGHT_PLAN §0` assigns pod4=VLM, pod5=release row; `CLAUDE.md`: never add load to a training pod |
| 11 | **cost authorisation** | ⛔ **BLOCKED — PI (D1)** | ESTIMATED 175–290 A40-h; `V6_TRAINER_DESIGN §5` |
| 12 | code shipped to the pod, md5-verified — ⚠️ **THREE files**: `v6.py`, `train_v6_staged.py`, **and `train_v58f_unicycle_head.py`** (an import-time dependency; MEASURED mine 2026-08-12) | ⛔ **NOT-STARTED** — needs #10 | ⛔ **file-ship, never git**: pods have no git credentials, `git fetch` HANGS and a failed-fetch `checkout -B` RESETS the tree (MEASURED 2026-08-11, `CLAUDE.md`) |
| 13 | the direct SSH mapping to that pod actually answers | ⛔ **NOT-STARTED** — needs #10 | **MEASURED 2026-08-11: BOTH pod4 and pod5 refuse on their own `$RUNPOD_PUBLIC_IP:$RUNPOD_TCP_PORT_22`.** ⇒ probe with `ssh -n … 'echo OK'` FIRST, fall back to the HF relay |
| 14 | disk headroom on the pod | ⛔ **NOT-STARTED** — needs #10 | ⛔ **never judge pod disk with `df`** (reports the 965 TB cluster, hides the per-pod MooseFS quota — a full quota killed the flagship mid-checkpoint). Use a real `dd` write test |
| 15 | S-W's gate probes (P1/P3/P6) are runnable on that pod | 🟡 **PARTIAL** | the battery is a **separate frozen instrument by design** — the trainer does not run it in-loop. Owners are named in `STAGE_GATE_SPEC`: P1 `scripts/probe_latent_state.py`, P3/P6 `scripts/stage_a_probes.py`. Folded in via `--gate-probes` (§4.3) |
| 16 | a `runs.d/<run>.env` manifest + supervisor for S-W | ⛔ **NOT-STARTED** — 0-GPU, filed as **F2** in `BACKLOG.md` | needed before a 7–12-day run is left unattended |
| 17 | **W5/E-H1 — v5.8f's 6 s baseline** | ⛔ **NOT-DONE** | ⚠️ **This is a YARDSTICK requirement, not a launch blocker.** See §1.1 |
| 18 | **S2 — `g_str` supervision from PH2** | ⛔ **NOT-STARTED** | ⚠️ blocks **S-S's gate**, not S-W and not S-T. See §1.1 |
| 19 | whole-suite `pytest -q` green (the commit invariant) | ⚠️ **STANDING HAZARD — not v6** | 23 failures occur **with and without** the v6 files (control run, `V6_TRAINER_DESIGN §7.0`): `onnx` absent (1), a Windows-basename assert (2), a suite-order polluter (20). Filed as **F5** |

### 1.1 The two escalated precursors, answered precisely

**W5/E-H1 (#17) — does it block the launch? NO. It blocks the comparison.**

`HIERARCHY_VOCABULARY §4b` promotes it to a REQUIRED precursor: *"it baselines v5.8f at 6 s
before v6 trains against it."* But **MEASURED (mine, source read 2026-08-12): S-W consumes no
v5.8f MEASUREMENT.** S-W's required gate is **P1 / P3 / P6**, none of which reference v5.8f, and
no path in `train_v6_staged.py` reads a v5.8f **checkpoint, gate JSON or eval result**.

> ⚠️ **Precision, because the grep is misleading.** v6 *does* import v5.8f **code** —
> `train_v58f_unicycle_head`'s `A_MAX`/`KAPPA_MAX` (v6.py:1096, train_v6_staged.py:116),
> `build_train_episodes` (:1001), `make_sampler` (:1159), `UnicycleEmission`. That is
> **deliberate reuse of gated parts**, not a dependency on a v5.8f *number*. It has one
> operational consequence, in precursor #12: **`train_v58f_unicycle_head.py` is a hard
> import-time dependency and must be shipped and md5-verified alongside the two v6 files** —
> §2.0 covers it.

⇒ **S-W can start; the 6 s v6-vs-v5.8f comparison is not quotable until W5 lands.** Run them in
parallel on different pods (D3). If the PI prefers serial, that is a schedule choice, not a
technical one.

**S2 / PH2 (#18) — does it block the launch? NO. It blocks S-S's GATE.**

| stage | can it start? | why |
|---|---|---|
| **S-W** | ✅ **YES** | S2 appears in no S-W loss term and in no S-W gate probe |
| **S-T** | ✅ yes, once S-W's gate exists | its gate is TACTICAL family + `sel_gap`; S2 is not involved |
| **S-S** | ⛔ **NO — structurally** | `STAGE_GATE_SPEC["S-S"]["required"] = ("STRATEGIC_family",)` and S2 is its only supervision source. **The stage will terminate at `pass: null`, and `pass: null` is not a pass** — S-J then refuses to launch |

⇒ **exactly as the brief suspected: S-W can start, S-T can follow its gate, and S-S cannot
complete a valid gate before PH2.** The fix is the pre-registered amendment in **D6** (promote
`S1_ade_8_30s` to required for the PH2-pending case, STRATEGIC reported `n/a` **with its reason
and its n**) — a ~1 h code change in `stack/`, filed as **F1**. It must land BEFORE S-S runs, not
after seeing S1's number.

---

## 2. The exact command lines

⛔ **`PYTHONPATH=/workspace/TanitAD/stack` is REQUIRED** or the trainer dies with
`ModuleNotFound: tanitad`. `cd` alone is not enough.
⛔ **`OMP_NUM_THREADS=6`** — torch spawns ~113 threads per process; 7 concurrent arms sat at GPU
`sm` **0–6 % for 50 minutes** without it (MEASURED 2026-07-27). The trainer sets it defensively;
set it in the launch line too so it is visible in `ps`.
⚠️ Every block below creates its own `--out` directory first — the `nohup … > …/train.out`
redirect fails if the directory does not exist yet.

### 2.0 STEP ZERO — on the pod, before any launch (~1 min, 0 GPU)

```bash
# a. the files arrived by md5-verified FILE-SHIP (pods have NO git credentials)
#    ⚠️ THREE files, not two: train_v58f_unicycle_head.py is an IMPORT-TIME dependency
#    (A_MAX/KAPPA_MAX, build_train_episodes, make_sampler, UnicycleEmission). A stale
#    copy does not fail loudly — it changes the emission envelope.
md5sum /workspace/TanitAD/stack/tanitad/models/v6.py \
       /workspace/TanitAD/stack/scripts/train_v6_staged.py \
       /workspace/TanitAD/stack/scripts/train_v58f_unicycle_head.py

# b. grep-verify the fix you think is there IS there
grep -c "assert_isolation" /workspace/TanitAD/stack/tanitad/models/v6.py

# c. the caches exist (precursor #9)
ls -d /workspace/data/physicalai-train-e438721ae894-w120-256x640cyl \
      /workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl

# d. disk headroom — ⛔ NOT `df` (it reports the 965 TB cluster, not the pod quota)
dd if=/dev/zero of=/workspace/_ddtest bs=1M count=4096 oflag=direct && \
  rm -f /workspace/_ddtest

# e. it imports and steps — 0 GPU, 0 corpus, ~20 s
cd /workspace/TanitAD/stack && \
PYTHONPATH=/workspace/TanitAD/stack OMP_NUM_THREADS=6 \
python3 scripts/train_v6_staged.py --stage S-W --dry-run \
    --out /workspace/experiments/v6-dryrun --dry-batch 1 --dry-steps 2 \
    --dry-k 12 --o5-k 12
```

**Expect (MEASURED on the dev box 2026-08-12, byte-for-byte the same three lines):**

```
[v6] params 87.89 M / budget 300 M · arm shared-encoder+adapters · per-group {...}
[v6] X3 isolation pass=True violations={'planner_to_encoder': 0, 'tactical_to_below': 0, 'strategic_to_below': 0}
[v6 dry 1] {...}  [v6 dry 2] {...}  [v6] dry-run OK -> .../dry_run.json + config.json
```

⛔ **If any of a–e disagrees, do not launch.** A launch from a stale pod checkout resurrects
fixed bugs (MEASURED 2026-07-27: pod2 sat at `0f93b98` while the v5 gate fix was at HEAD).

### 2.1 S-W — the WORLD stage · **the only stage that can start tomorrow**

**Pod:** the dedicated S-W pod (D2). ⛔ **not** a pod that is training or evaluating.
**Expected `step_s`:** ESTIMATED **21–35** (A40). ⛔ **Read it at step 500 and re-cost — D1.**
**Derived `max_horizon`: 20** (MEASURED, mine) ⇒ **94 windows per 120-frame episode, unchanged
from v5f.** Confirm from the trainer's own `[v6] windowing:` line.

```bash
mkdir -p /workspace/experiments/v6-SW-30k && \
cd /workspace/TanitAD/stack && \
PYTHONPATH=/workspace/TanitAD/stack OMP_NUM_THREADS=6 \
nohup python3 scripts/train_v6_staged.py \
  --stage S-W \
  --out /workspace/experiments/v6-SW-30k \
  --v2-cache     /workspace/data/physicalai-train-e438721ae894-w120-256x640cyl \
  --v2-val-cache /workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl \
  --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical \
  --v2-subframe 176x624 --require-parity \
  --steps 30000 --batch 16 --lr 1e-4 \
  --o1-k 10 --o5-k 20 --o5-mode uniform \
  --o3-mode action --o3-blocks 2 --o3-block-h 2 --o3-block-w 2 \
  --o2-tau-s 2.0 --o4-alpha 1.0 --w-o6 0.1 --spectrum-every 200 \
  > /workspace/experiments/v6-SW-30k/train.out 2>&1 &
```

* **trains:** `encoder`, `readout`, `predictor_op` (+ `step_readout_op`), `aux` (the O3 masked-cell head)
* **frozen:** `layer_tac`, `layer_str`, `planner`
* **losses in force:** O1 (ctrl 1.0 / fact 1.0 / scene 0.3) · O2 1.0 · O3 1.0 · O5 1.0 · O6 0.1
* **λ_plan ≡ 0** — the trainer **refuses to start** S-W with a non-zero `--lambda-plan`
* **one-off startup:** the O4 pre-pass scores every window once, reading **action arrays only**
  (no frame decode). Expect **seconds**. If it takes hours, that is a symptom, not the design

### 2.2 S-T — tactical layer + operative planner, on the FROZEN S-W trunk

**Pod:** any free A40 (S-W's pod is free once S-W finishes). **Expected `step_s`:** ESTIMATED
**6–10**. **Derived `max_horizon`: 60** ⇒ **54 windows/episode, −42.6 % vs v5f** — unavoidable
(a 6 s planner has no target in a shorter window), and it belongs in the run row (D4).

```bash
mkdir -p /workspace/experiments/v6-ST-10k && \
cd /workspace/TanitAD/stack && \
PYTHONPATH=/workspace/TanitAD/stack OMP_NUM_THREADS=6 \
nohup python3 scripts/train_v6_staged.py \
  --stage S-T \
  --prev-gate /workspace/experiments/v6-SW-30k/stage_gate.json \
  --init-from /workspace/experiments/v6-SW-30k/ckpt.pt \
  --max-horizon 60 \
  --out /workspace/experiments/v6-ST-10k \
  --v2-cache     /workspace/data/physicalai-train-e438721ae894-w120-256x640cyl \
  --v2-val-cache /workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl \
  --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical \
  --v2-subframe 176x624 --require-parity \
  --steps 10000 --batch 16 --lr 1e-4 --w-t1 1.0 \
  > /workspace/experiments/v6-ST-10k/train.out 2>&1 &
```

* ⛔ **`--init-from` is REQUIRED** and preflight refuses without it. A gate saying "S-W passed" is
  worthless if this stage then trains on a randomly-initialised trunk — that is not a staged
  protocol, it is four unrelated models with a gate between them. The load is `strict=True` and
  the run config records the **md5 of the loaded trunk**
* **trains:** `layer_tac` + `planner` · **frozen:** everything below (Drive-JEPA's shape — the
  planner is a *post-trained consumer*)
* **λ_plan 1.0** here; the plan loss reports `plan_ade_0_2s` and `plan_ade_2_6s` **separately**,
  because a pooled 0–6 s number cannot see the seam

### 2.3 S-S — strategic layer on the FROZEN S-T stack · ⛔ hold for D6

**Pod:** any free A40. **Expected `step_s`:** ESTIMATED **5–9**. **Derived `max_horizon`: 20**
(`stride_str`) ⇒ **94 windows, unchanged.**
⛔ **Do not launch until the D6 gate amendment lands** — otherwise this stage terminates at
`pass: null` and S-J refuses.

```bash
mkdir -p /workspace/experiments/v6-SS-8k && \
cd /workspace/TanitAD/stack && \
PYTHONPATH=/workspace/TanitAD/stack OMP_NUM_THREADS=6 \
nohup python3 scripts/train_v6_staged.py \
  --stage S-S \
  --prev-gate /workspace/experiments/v6-ST-10k/stage_gate.json \
  --init-from /workspace/experiments/v6-ST-10k/ckpt.pt \
  --out /workspace/experiments/v6-SS-8k \
  --v2-cache     /workspace/data/physicalai-train-e438721ae894-w120-256x640cyl \
  --v2-val-cache /workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl \
  --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical \
  --v2-subframe 176x624 --require-parity \
  --steps 8000 --batch 16 --lr 1e-4 --w-s1 1.0 \
  > /workspace/experiments/v6-SS-8k/train.out 2>&1 &
```

* **trains:** `layer_str` only · **frozen:** everything else · **λ_plan 0**
* ⚠️ **S2 (`g_str` supervision) is not wired and must not be faked.** Until PH2 lands, S-S trains
  the strategic **latent prediction (S1)** only, and the STRATEGIC family is reported **`n/a`
  with its reason and its n** — never silently dropped

### 2.4 S-J — optional brief joint polish · run ONLY if S-T/S-S plateau

**Expected `step_s`:** ESTIMATED **21–35** (all terms live again). **Derived `max_horizon`: 60.**

```bash
mkdir -p /workspace/experiments/v6-SJ-3k && \
cd /workspace/TanitAD/stack && \
PYTHONPATH=/workspace/TanitAD/stack OMP_NUM_THREADS=6 \
nohup python3 scripts/train_v6_staged.py \
  --stage S-J \
  --prev-gate /workspace/experiments/v6-SS-8k/stage_gate.json \
  --init-from /workspace/experiments/v6-SS-8k/ckpt.pt \
  --max-horizon 60 \
  --out /workspace/experiments/v6-SJ-3k \
  --v2-cache     /workspace/data/physicalai-train-e438721ae894-w120-256x640cyl \
  --v2-val-cache /workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl \
  --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical \
  --v2-subframe 176x624 --require-parity \
  --steps 3000 --batch 16 --lr 3e-5 \
  > /workspace/experiments/v6-SJ-3k/train.out 2>&1 &
```

* trains everything, **isolation still ON**. Gate: the frozen battery **FLAT** across the joint phase

### 2.5 The `--init-from` chain, and the control arms

```
S-W ──stage_gate.json──► S-T ──stage_gate.json──► S-S ──stage_gate.json──► S-J
     └──── ckpt.pt ─────►      └──── ckpt.pt ────►      └──── ckpt.pt ────►
```

Every downstream stage passes **both** `--prev-gate` (the X5 precondition) and `--init-from`
(the weights). Missing either is a refusal, not a warning.

| pre-registered control arm | flag | ⚠️ never a default |
|---|---|---|
| E-ENC (b) per-layer encoders | `--per-layer-encoders` | D5 |
| uplink = V-JEPA teacher | `--uplink ema --ema-decay 0.996` | |
| O4 off (attributability control) | `--o4-alpha 0` | reproduces uniform sampling exactly |
| O5 endpoint (reproduces the defect O5 fixes) | `--o5-mode endpoint` | |
| ⛔ isolation off (the co-trained path) | `--no-isolate-planner` / `--no-isolate-uplink` **+ `--i-know-this-is-the-control-arm`** | preflight refuses without the acknowledgement |

---

## 3. Gates — what is measured, what passes, and what happens on FAIL

`--gate` runs automatically at the end of every stage and writes `<out>/stage_gate.json`.
Launching stage N+1 runs `assert_stage_precondition`, which reads stage N's gate.

| stage | REQUIRED probes | passes if | reported-only |
|---|---|---|---|
| **S-W** | P1, P3, P6 | P1 retention **≥ 0.85×** R²(z) at k=10 **per target** · P3 sign **≥ 0.95** on **both** channels · P3 gain median **∈ [0.5, 2.0]** *without post-training* · P6 action-subspace dims **≤ 32** | P2, P5, P8, O6 spectrum |
| **S-T** | TACTICAL family, `sel_gap` | `sel_gap` **≤ 0.5×** the fan oracle **at T1 tier** · TACTICAL confusion improves on the E4.1-derived strata | P7 (ρ ≥ 0.3, CI excluding 0, **per stratum**), LATERAL family, X2 seam |
| **S-S** | STRATEGIC family | **computable at all** (measured vs `n/a` today) · S1 ADE(8–30 s) beats CV/corridor baselines at T1 | X2 seam |
| **S-J** | X3 isolation, no-harm | zero live forbidden edges · battery **FLAT** across the joint phase | P1/P3/P6, four families |

### 3.1 Three verdicts — and why the third exists

| verdict | meaning | effect |
|---|---|---|
| `pass: true` | every required probe ran and passed | may launch the next stage |
| `pass: false` | a required probe **FAILED** | ⛔ **REFUSED, no override.** X5: *a failed stage never propagates upward.* A FAIL is a finding **about the layer below**; propagating it is how a defect gets attributed to the wrong layer three stages later |
| `pass: null` | a required probe **did not run** or reported null | ⛔ **INCONCLUSIVE IS NOT A PASS.** Refused unless `--allow-inconclusive-gate` **AND** a non-empty `--gate-off-reason`, stamped into `config.json` and printed as a banner |

An unavailable probe is recorded with **what** could not be reached and **where it lives**
(`STAGE_GATE_SPEC[...]["owners"]`) — never silently dropped, never counted as a pass.
**X3 is the one gate this module measures on its own, always.**

### 3.2 Pre-registered FAIL branches — both outcomes committed in advance

| stage | on FAIL | ⛔ what is NOT allowed |
|---|---|---|
| **S-W / P1 retention** | the WM did not retain the driving targets. Branch to the **objective**, not the schedule: the LF-levers (`JEPA_PHYSICS_SURVEY §3`) — O3 masking rate, O2 τ, then LF4 (`--o5-k 60`) — one lever at a time | re-running with more steps and re-reading the gate. `CLAUDE.md`: an exponent below R² 0.80 has no quotable value and **may never decide a restart** |
| **S-W / P3 sign or gain** | the action interface is muffled — this is the **exact** defect stage-A repaired post-hoc (0.27 → 0.97) and O1 exists to prevent from step 0. If O1 does not prevent it, **O1's response form is the finding**, and the fix is the `L_ctrl` weighting, not the trunk | declaring it fixed from a **trainer log**. Trainer val watches a curve; only eval output is quotable |
| **S-W / P6 dims > 32** | the action subspace did not factorise. Report it and continue to S-T **only** if P1+P3 pass — P6 is a structure diagnostic, and a FAIL here is a paper result | quietly widening the threshold after seeing the number |
| **S-T / `sel_gap`** | ⚠️ **the most likely FAIL on the board.** See §5 risk R2 — selection has failed three times already. Branch: the noise-robust selection rule (top-m, not argmin) + `--w-prog`, pre-registered off the banked windows **before** S-T is re-run | reading a T0 number as the verdict. The gate says **at T1 tier** |
| **S-S / STRATEGIC** | if PH2 has not landed this is `pass: null`, not `pass: false` — the D6 amendment is the answer, not an override | `--allow-inconclusive-gate` without the amendment, as a habit |
| **S-J / no-harm** | the joint phase moved the battery ⇒ **stop S-J and keep S-S's checkpoint.** That is the H-COTRAIN rule applied preventively | keeping S-J because its planner number improved while the battery moved |

### 3.3 Folding in the externally-run battery

The battery is a **separate, frozen instrument by design** — an instrument that moves with the
trainer stops being a yardstick. Run it, then:

```bash
cat > /workspace/experiments/v6-SW-30k/probes.json <<'JSON'
{"P1": {"pass": true, "status": "run", "artifact": "p12_gate.json"},
 "P3": {"pass": true, "status": "run", "artifact": "w3_gate.json"},
 "P6": {"pass": true, "status": "run", "artifact": "w3_gate.json"}}
JSON
# then re-run the gate hook with --gate-probes, or pass it on the next launch
```

---

## 4. The cost ladder — ⚠️ ESTIMATED, and the re-cost is the first action on the pod

### 4.1 The basis (MEASURED — the only two rows that anchor this)

| run | wallclock | steps | s/step | source |
|---|---|---|---|---|
| `flagship4b-speedjerk-30k` (v1) | `wallclock_s` **191 206.2** | 30 000 | **6.374** | `MODEL_REGISTRY.md §1.2` |
| `flagship-v4-fromscratch` | **59.04 h** = `wallclock_s` **212 544.6** | 30 000 | **7.085** | `MODEL_REGISTRY.md §1.5.5` |

> ⚠️ **Conflict reported.** `V6_TRAINER_DESIGN §5` attributes the 59.04 h basis to **"v4.2"**.
> The registry says it belongs to **`flagship-v4-fromscratch` (§1.5.5)**; `flagship-v4.2-30k`
> (§1.5.3) is a different arm, **killed at ~step 5 k**. **Registry wins** — the estimate is
> unaffected (same two MEASURED runs), the name is not. Filed as **F4**.

### 4.2 The ladder — ⚠️ every number in this table is **ESTIMATED**

| stage | steps | ESTIMATED s/step (A40) | ESTIMATED A40-hours | why it costs what it costs |
|---|---|---|---|---|
| **S-W** | 30 000 | **21–35** | **175–290** (7–12 A40-days) | **the whole cost centre.** 26 encoder passes/sample (window 6 + 20 futures) vs v1's ~8; ~80 predictor rolls/sample (O1 6 arms × k=10, O5 × 20) vs v1's handful, on a 60.3 M-param predictor |
| S-T | 10 000 | 6–10 | 17–28 | trunk frozen, no O1/O5 rolls, **no future-frame encodes** — forward-only through the encoder |
| S-S | 8 000 | 5–9 | 11–20 | strategic layer only; one future-frame encode at `stride_str` |
| S-J | 3 000 | 21–35 | 18–29 | all terms live again |
| **total** | | | **≈220–370 A40-hours** | |

**Why the estimate must not be trusted as-is:** this programme's own history says estimates here
run **~11 % low** — `MODEL_REGISTRY §1.5.5` records *"~53 h ESTIMATED"* against **MEASURED
59.04 h**, understating the spend by **≈2.5 GPU-days**.

### 4.3 ⛔ THE FIRST ACTION ON THE POD IS TO RE-COST FROM THE RUN'S OWN LOG

The trainer logs `step_s` **already divided**, with `step_s_note` naming the divisor — precisely
so nobody re-derives the false *"training is 430 s/step"* alarm from an accumulated counter
(the older trainers accumulate `step_s` over `--log-every`, ÷50).

**Read `step_s` at step 500. Then apply the D1 rule** (< 21 → proceed and re-cost down · 21–35 →
proceed · 35–50 → apply levers, `--o5-k 10` first, relaunch, re-measure · > 50 → ⛔ STOP and
diagnose: at >2× a band built from two MEASURED A40 runs it is a defect, not a cost).

**Cost levers, in the order they cost the least science**

| lever | effect | what it costs |
|---|---|---|
| `--o5-k 10` (from 20) | halves the O5 roll **and** the future-frame encode | O5's compounding shaping measured over 1 s instead of 2 s |
| `--batch 8` | fits a smaller GPU | fewer windows/step — **not** a free swap |
| drop the O1 `random` arm | 5 rolls instead of 6 | −17 % of O1; the named channels are the gated ones |
| `--steps 20000` | linear | shorter ladder — state it in the run row |
| ⛔ **not a lever** | reducing `--plan-steps` below 60 | that is §4b, and it is **binding** |

---

## 5. Risk register — what could make v6 fail, and the early warning

Each row names the **MEASURED** failure that earned it. The early-warning column is what to watch
in the first hours, not after 30 k steps.

| # | risk | why it could bite v6 | ⭐ EARLY WARNING | defence |
|---|---|---|---|---|
| **R1** | **representation collapse** — the latent narrows and the WM stops carrying the world | a from-scratch trunk with five simultaneous losses is exactly where collapse lives | the **O6 spectrum series**, logged every `--spectrum-every 200`: participation ratio and top-k share. **Falling PR in the first 2 k steps is the alarm** | SIGReg (LeJEPA, λ 0.1, 512 slices) is the defence and it is **MEASURED to work**: retention **1.532** vs the ≥0.8× gate — effective rank *expanded* 53 % (PR 4.53→6.94 of 2048) across the full λ_plan ramp (`h_cotrain_curve.json`). ⚠️ that was measured on the **v5f** run, so S-W re-measures it as a **series**, never assumes it |
| **R2** | **selection fails again — the winner's curse** | argmin over a large noisy-cost fan selects for cost **under-estimation**. MEASURED: W7-FULL `sel_gap` **3.2075** vs the ≤0.4505 m gate (**FAIL**), and the roll-cost's argmin sits at error-rank **132.32 of 256** — the median, i.e. **no better than chance at the extreme** (`w7_full_gate.json`, `w7_selection_rules.json`, EXPLORATORY) | S-T's `plan_ade_0_2s` vs `plan_ade_2_6s` **reported separately** from step 1; and the shortlist **ceiling** vs the **pick** — a falling ceiling with a flat pick is the winner's curse, live | the top-m ceilings **do** fall (0.356 at m=32), so a better rule inside a top-m set exists — **just not argmin**. Pre-register the noise-robust rule and the `--w-prog` anti-degeneracy term (**weight 0.0 in every W7 run to date**) off the banked windows, 0-GPU, **before** S-T. Filed as **F3** |
| **R3** | **consumer invalidation** — S-T's planner is trained on S-W's trunk, and any later trunk change invalidates it | MEASURED: repairing the trunk moved the frozen selector from **0.7933 → 4.4159** — a **5.6×** degradation of a consumer that was never retrained (`w7_full_gate.json`). *You cannot repair a trunk and keep its planner* | any change to S-W's checkpoint after S-T starts — including an "innocent" resume from a different `ckpt.pt` | `--init-from` is `strict=True` and the run config records the **md5 of the loaded trunk**, so a row always names exactly which S-W it stands on. ⇒ **freeze and md5 the S-W checkpoint the moment its gate passes**, and never re-run S-W into the same directory |
| **R4** | **the action-echo trap** — open-loop "skill" that is action recitation | MEASURED (`EVAL_DOCTRINE §1.12`): S-curve reproduction **97.9 % open-loop, 0.0 % hold-action, ~5 % closed-loop**. It is the reason T1 is the primary tier | any capability claim quoted from a **T0** number, or from a trainer log | **T1 is the primary tier and the gates say so** (`sel_gap ≤ 0.5× oracle at T1 tier`). T0 is a WM diagnostic, never driving performance. Instrument: `taniteval/tools/t1_eval.py` |
| **R5** | **stage N+1 silently trains on a random trunk** | `--init-from` omitted; the gate passed, the log looks healthy, and the ladder is four unrelated models | the run config's recorded trunk md5 being absent | preflight **refuses** S-T/S-S/S-J without `--init-from`; `strict=True` load |
| **R6** | **the 6 s horizon is quietly abandoned as "a corpus limitation"** | inheriting v4's `plan.max_horizon` (**MEASURED 20**) makes §4b untrainable — S-T could never start, because a 6 s planner has no target in a 2 s window | the `[v6] windowing:` line printing 20 on an S-T launch | v6 **derives its own** and prints it beside v4's; below-need / below-`maneuver_h` / 0-window are explicit refusals. *A silently shortened horizon is not the same experiment* |
| **R7** | **a supervised run is RESURRECTED days after it finished**, or a relaunch restarts at step 0 over a live checkpoint | MEASURED 2026-08-09/11: the v5f run finished but never wrote its done-marker; its supervisor relaunched for **two days**, then a fixed crash-cause let a relaunch SUCCEED, resume from a stale `ckpt.pt`, and overwrite `config.json`/`metrics.json`/`ckpt.pt` **next to a live eval** | any `ps` showing a supervisor for a run whose `summary.json` says done | the trainer writes `summary.json {"done": true}` **in the same turn it finishes**; `resume_guard` refuses a DONE dir (only `--force-rerun`) and refuses `--resume off` over an existing `ckpt.pt`. ⭐ the done-marker is also the correct **remote off-switch** |
| **R8** | **the manifest edit that changes nothing** | `supervise_run.sh` **sources its manifest ONCE, at supervisor startup**, and replays the captured `TRAIN_CMD`. The relaunch looks successful while running the OLD config | flags in `ps` disagreeing with `runs.d/<run>.env` | to change a supervised run: edit the manifest → kill the **SUPERVISOR** first → kill the trainer → start a fresh supervisor. **Verify by grepping the flags out of the RUNNING process**, never by reading the manifest. ⚠️ and do not restart immediately — the new supervisor **races the old one's `flock`**, prints *"another supervisor holds …lock"* and dies, leaving **nothing running** |
| **R9** | **a probe that reports the wrong scope, read as an answer** | `df` reports the 965 TB cluster and hides the pod quota (a full quota killed the flagship mid-checkpoint); cgroup `memory.usage_in_bytes` counted **37.2 GB of reclaimable page cache with nothing running** and cost ~40 min of training to an invented OOM diagnosis | any alarm raised from a single system counter | disk ⇒ a real `dd` write test. Memory ⇒ `memory.stat`'s **`rss`** and **`memory.failcnt`** — `failcnt 0` settles it. **Verify before alarming**, always two samples |
| **R10** | **the estimate is the plan** | ESTIMATED 175–290 A40-h is an extrapolation, and this programme's estimates ran ~11 % low once already | `step_s` at step 500 outside 21–35 | the D1 re-cost rule, with all four branches pre-registered **before** the number is seen |
| **R11** | **`pkill -f train_v6_staged` kills your own ssh session** | the pattern self-matches the ssh command line — it returns empty output and looks like nothing happened | — | kill by **explicit PID** |
| **R12** | **concurrent arms at GPU `sm` 0–6 %, looking exactly like a hang** | torch spawns ~113 threads **per process**; MEASURED 2026-07-27: 7 arms, **50 minutes**, zero progress → `OMP_NUM_THREADS=6` and the same arm finished in **232 s** | `nvidia-smi` `sm` in single digits with jobs "running" | `OMP_NUM_THREADS=6` in the launch line so it is visible in `ps`. **Do not diagnose it as a deadlock** |

### 5.1 The one risk with no built-in defence

**R2 (selection) is the open one.** S-W's gate does not touch it, so it cannot stop the launch —
but it is where v5.8f's arc stalled, and S-T's gate is `sel_gap ≤ 0.5× oracle at T1`. The
mitigation is **0-GPU and can be done while S-W trains** (F3): sweep the selection rule and
`--w-prog` off the already-banked `w7_eval_windows.pt`, and pre-register the rule before S-T
launches. That converts a 7–12-day wait into parallel work.

---

## 6. Deliverable manifest

| artifact | where it lives | state |
|---|---|---|
| `…/incoming/2026-08-07-hierarchical-wm-redesign/V6_GO_PACKAGE.md` | this file — repo working tree + index | **staged**, verified with `git ls-files --cached` |
| `…/incoming/2026-08-07-hierarchical-wm-redesign/PI_DECISIONS_2026-08-12.md` | repo working tree + index | **staged**, verified with `git ls-files --cached` |
| `Project Steering/BACKLOG.md` §F | repo working tree + index | **staged**, verified with `git ls-files --cached` |
| the v6 trainer itself | `stack/tanitad/models/v6.py` · `stack/scripts/train_v6_staged.py` · `stack/tests/test_v6_staged.py` | already in HEAD `2b8d09e` — **not** stranded |

**This agent committed nothing and pushed nothing** — the `AGENT_OPERATING_STANDARD` contract.
**Nothing produced here lives only on a pod or only in a worktree.**

### 6.1 Escalations — stated here, not buried

1. ⛔ **S-W is GO on code and blocked ONLY on D1 (cost) + D2 (pod).** Both are provision/spend.
2. ⚠️ **S-S cannot pass a valid gate before PH2** — `STAGE_GATE_SPEC["S-S"]["required"]` is
   `("STRATEGIC_family",)` and S2 is unwired. Needs the **D6 amendment in `stack/`**, which this
   agent does **not** own. **Filed as F1 with an owner-file, not written into a doc as
   "please merge".**
3. ⚠️ **Registry-vs-doc conflict:** `V6_TRAINER_DESIGN §5` names the 59.04 h basis "v4.2"; the
   registry (§1.5.5) says `flagship-v4-fromscratch`. **Registry wins.** F4 fixes the doc — this
   agent does not own `V6_TRAINER_DESIGN.md`.
4. ⚠️ **23 standing `pytest` failures**, none of them v6, against the *"`pytest -q` must stay
   green before any commit"* invariant. F5.
