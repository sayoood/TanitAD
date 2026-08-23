# P4 — TRAINING PIPELINES · SPEC

`Owner: TanitAD_TrainingFlyWheel (AGENT_CHARTERS.md §3). Created 2026-08-23.
Status: v1 — SPEC'd AGAINST THE MEASURED SURFACE, not against memory. Every
"today" claim below carries a file:line or an artifact path. Where the charter's
premise turned out to be already-fixed, this document says so rather than
restating it.`

**Scope.** P4 makes training *a reproducible instrument, not an event*: any model
type × architecture × dataset, launchable, resumable, and gated, on any of our
machines, **without losing efficiency**.

**Non-scope.** Dataset construction and parity selection (P2/DataFlyWheel), what
counts as a good number (P7/EvalFlyWheel), target deployment (P6/DeployFlyWheel).
P4 owns the *act of training* and the *record it leaves*.

---

## 0. THREE PREMISE CORRECTIONS (evidence beat memory)

The mandate that opened this work named specific failures. Two are already fixed,
and saying so is part of the job — a spec that re-legislates solved problems
wastes the next agent's turn.

| premise as briefed | measured reality | consequence |
|---|---|---|
| *"`config.json` must record EVERY knob"* — implying it does not | **All 174 CLI dests ARE recorded.** `_run_config` dumps `vars(a)` wholesale (`stack/scripts/train_v6_staged.py:3561-3562`). Zero CLI gaps. | The CLI half of the problem is CLOSED. Do not re-solve it. |
| *"two arms differed only in `TANITAD_RESIDUAL_INIT_SCALE` with identical configs"* | **Fixed.** Recorded as `residual_head_init_scale` in `config.json` (`:3560`) *and* in `summary.json` (`:4529`). | That exact failure cannot recur. |
| *"env vars are an uncontrolled surface"* | The trainer reads **exactly one** env var in its whole import graph (`TANITAD_RESIDUAL_INIT_SCALE`, `stack/tanitad/models/predictor.py:21-22`) and *writes* one (`OMP_NUM_THREADS`, `:6031`). | The env surface is small and nearly closed — §3 names the three remaining holes. |

⭐ **The real completeness gap is not knobs — it is PROVENANCE.** §3.

---

## 1. THE MEASURED SURFACE (what P4 is spec'd against)

| surface | reality | anchor |
|---|---|---|
| live trainer | `stack/scripts/train_v6_staged.py`, **6,050 lines**, 175 `add_argument` / 174 dests | `:4854-6025` |
| stage ladder | `STAGES = ("S-W","S-T","S-S","S-J")` | `stack/tanitad/models/v6.py:4079` |
| chain driver | `stack/scripts/v6_chain.py` (2,323 lines) — Python because it BRANCHES ON A GATE VERDICT | `v6_chain.py:1-27` |
| supervisor | `stack/scripts/supervise_run.sh` (145 lines) | — |
| run manifests | `stack/ops/runs.d/` — **one** manifest, and it is for `train_flagship_v4.py`, not v6 | `runs.d/flagship-v5f-w120-30k.env` |
| Colab | `colab/` — S2 label lab only, **no training entry point** (`grep -rn train_v6_staged colab/` → nothing) | `colab/win_shims/termios.py` |
| done-marker | `<out>/summary.json` with `done: true` | `:4522-4536` |

### 1.1 Machines, as verified this session

| machine | reach | venv | code path | verified |
|---|---|---|---|---|
| **Thor** (Jetson) | `ssh tanitad-thor-wifi` (192.168.178.93; `.194` is the stable ethernet path) | `~/venvs/tanitad-train` | `~/TanitAD/stack`, `PYTHONPATH` REQUIRED | ✔ 2026-08-23 — ran the val probe end-to-end |
| **dev box** (RTX 4060) | local | `C:/Users/Admin/venvs/tanitad` | repo `stack/`, mirror `C:\Users\Admin\tanitad-mirror\stack` | ✔ instrument present |
| **Colab** | `colab/` CLI | — | needs `PYTHONPATH=<repo>/colab/win_shims` | ⚠️ no trainer entry point exists |
| **HF Pro** | `hf-tanitad` | — | — | ⛔ METERED — asks first, always |

⛔ **Thor has NO git credentials.** Code reaches it by **file-ship + md5 verify**,
never by pull. Verified this session: shipped `stack/scripts/val_rank_probe.py`,
local `c486fffc530f747334f5957bffd58c67` == remote — identical. **That md5 step is
the code-freshness gate; a run launched without it is not a reproducible run.**

---

## 2. THE REPRODUCIBILITY CONTRACT (R-rules)

A run is *reproducible* when a fresh agent, given only `<out>/`, can rebuild what
produced it. Today `<out>/` answers "with which flags" completely and "on what"
not at all.

| id | rule | today | action |
|---|---|---|---|
| **R1** | every CLI knob recorded | ✅ all 174 (`:3561`) | keep; add a regression test that fails if the wholesale dump is ever narrowed |
| **R2** | every env knob that changes numerics recorded | ✅ for `TANITAD_RESIDUAL_INIT_SCALE` (`:3560`) | ⛔ **`OMP_NUM_THREADS`, `PYTORCH_CUDA_ALLOC_CONF`, `PYTHONPATH` are NOT recorded** — and `PYTHONPATH` decides whether `--dump-seam-plan` banks anything at all (`:5990-5994`) |
| **R3** | **code identity** recorded | ❌ no git SHA anywhere | record `git rev-parse HEAD` + dirty-flag + **the md5 of the trainer file actually executing** (the only thing true on a file-shipped machine, where the SHA is meaningless) |
| **R4** | **machine identity** recorded | ❌ no hostname, GPU model, torch/CUDA version, resolved device | record all four. `--device cuda` may silently become `cpu` (`:3603-3606`) and `config.json` would still say `cuda` |
| **R5** | **wall-clock start** recorded | ❌ (only `elapsed_s` at the end) | record start timestamp **and its timezone** — pods/Thor are UTC, the PI reads Europe/Berlin |
| **R6** | data identity recorded | ✅ parity key + geometry bound (`:5897`, `train_v58f_unicycle_head.py:340-349`) | keep |
| **R7** | precision regime recorded | ⚠️ `--no-amp` is in `args`, but **`torch.bfloat16` is hardcoded** with no `is_bf16_supported()` probe (`:4278-4279`) | record the *resolved* autocast dtype, and probe support (§4.2) |

⛔ **R3–R5 are the live completeness gap.** Two runs on two machines can produce
byte-identical `config.json` today and differ in torch version, GPU, thread count
and allocator. That is the *same class* as the residual-init failure — it is just
one layer further out.

---

## 3. PREFLIGHT CONTRACT

The trainer's preflight is genuinely strong: **54 `problems.append` sites in
`preflight()` (`:5469-5912`), all of which abort with exit 2** (`:6038-6041`)
before any GPU time. Build-time checks add four more.

| check | behaviour today | verdict |
|---|---|---|
| param budget | **RAISES** `ValueError` if over (`v6.py:5668-5674`) | ✅ correct |
| X3 isolation | **RAISES** `IsolationViolation` when strict (`:3196-3205`) | ✅ correct |
| stage precondition (X5) | **RAISES**; a `--dry-run` gate cannot license a real launch (`:2787-2793`) | ✅ correct |
| resume guard / lineage | **RAISES** (`:4555-4599`, `:4652`) | ✅ correct |
| `--gate-probes` file existence | **ABORTS** (`:5879`) — the fix for "3.1 GPU-days then die before the done-marker" | ✅ correct |
| residual-init banner | **PRINTS ONLY** (`:3188-3189`) | ✅ acceptable — the value is recorded (R2) |
| **rank-gate CAN-RULE arithmetic** | ⛔ **PRINTS ONLY, CANNOT ABORT** (`:3190`, banner `:3526-3537`) | ⛔ **DEFECT — see §3.1** |

### 3.1 ⛔ THE OPEN PREFLIGHT DEFECT — `--spectrum-accum` defaults to 1

`--spectrum-accum` default **1** (`:5351`) means one spectrum call sees
`batch × window` rows. At the champ30k settings that is 24 rows against an
admissible ceiling of 1024 — **the O6 rank criterion is structurally unable to
rule**, and the code says so itself (`:3501-3505`).

This is **H-GATE-1 re-entering through the default**. H-GATE-1 was closed by
adding the arithmetic; but the arithmetic only *prints*. A run launched without
`--spectrum-accum 43` produces a gate that reads INCONCLUSIVE for a reason that
has nothing to do with the model.

**Required:** the CAN-RULE check joins the aborting class whenever O6 is a
*required* probe for the stage. Warn-only is admissible **only** when O6 is
`reported_only`. A gate that cannot rule must fail loudly at second zero, not at
hour four. *(Charter success criterion: "A gate returns PASS/FAIL, not perpetual
INCONCLUSIVE.")*

### 3.2 Remote code-freshness (the preflight that lives outside the trainer)

⛔ **No run starts against stale code on a remote machine — verified by content.**
Because Thor has no git, this cannot be a SHA check. The binding procedure:

```bash
scp stack/scripts/<file>.py tanitad-thor-wifi:~/TanitAD/stack/scripts/<file>.py
md5sum stack/scripts/<file>.py
ssh tanitad-thor-wifi 'md5sum ~/TanitAD/stack/scripts/<file>.py'
```

Identical md5 ⇒ proceed. **Ship every file the run imports, not just the entry
point** — the measured cross-pod failure was a partial ship that imported fine
and trained different code.

---

## 4. HARDWARE ABSTRACTION LAYER (HAL) — abstract WITHOUT losing efficiency

The mandate's hard constraint is that abstraction must not cost throughput. The
inventory makes the design tractable: **the trainer is already almost
device-agnostic** (`V6Stack` takes `device=` from its inputs everywhere —
`v6.py:522,543,2143,…`). The coupling is concentrated in a handful of places.

### 4.1 The interception points

| # | coupling | today | HAL requirement |
|---|---|---|---|
| H1 | device string | `--device` default `cuda`, silent downgrade to `cpu` (`:3603-3606`) | a `Machine` object resolves the device and **records what it resolved** (R4). A silent downgrade must be a *loud* downgrade |
| H2 | autocast dtype | **`torch.bfloat16` hardcoded**, no support probe, no `GradScaler` (`:4278-4279`) | probe `is_bf16_supported()`; fall back fp16+GradScaler or fp32; record the resolved dtype |
| H3 | **data loading** | ⛔ **ZERO DataLoader workers.** Every JPEG decode + collate is synchronous on the main process: `default_collate([ds_train[i] for i in idx])` (`:4178`). `_to_device(non_blocking=True)` is a **no-op without pinned memory** | ⭐ **the single largest efficiency lever — see §4.2** |
| H4 | threads | `OMP_NUM_THREADS` `setdefault` at `:6031`, i.e. **after `import torch`** (`:100`) | set before torch import, or (better) let the launcher own it and record it |
| H5 | allocator | `PYTORCH_CUDA_ALLOC_CONF` set only by launchers, never the trainer | HAL sets per-machine and records |
| H6 | memory probe | `torch.cuda.max_memory_allocated()` (`:4422-4424`) | ✅ already the only admissible probe on Thor — keep, do not "improve" |
| H7 | path layout | `sys.path` self-mutation (`:103-104`); `taniteval` must be a sibling (`:5990-5994`); supervisor `OPS_DIR=/workspace` is **wrong on Thor** (`supervise_run.sh:32`) | HAL owns per-machine paths; `OPS_DIR` per machine |
| H8 | grad checkpointing | `resolve_gc` (`:3030-3042`), `auto` per surface | ✅ already the right shape — it is a memory/throughput trade the HAL should *set*, not remove |
| H9 | I/O strategy | in-process LRU sized by `--v2-lru`; episode-grouped sampling exists to cut MooseFS cold loads ~8× (`train_v58f_unicycle_head.py:353-358`) | per-machine LRU + sampling policy; the MooseFS tuning is pod-specific and must not be paid on Thor/dev-box |

### 4.2 ⭐ H3 is the efficiency mandate, and it is measurable

Zero workers means GPU time is serialised behind JPEG decode. The charter says
abstract *without losing efficiency*; here abstraction and efficiency point the
same way. **Before changing anything, measure** — the programme's own rule, and
the `step_s`-is-accumulated trap is exactly this class of error.

**Pre-registered, both outcomes committed:**
- **E-P4-HAL-1.** Instrument the existing loop to report `data_wait_s` vs
  `compute_s` per step on Thor and on the dev box, at the champ30k settings.
  - *If `data_wait_s` ≥ 25 % of step time* → a worker/prefetch path is justified;
    implement it behind the HAL with pinned memory, and re-measure paired.
  - *If `data_wait_s` < 25 %* → **the zero-worker design is correct** and stays;
    record the number so this is never re-litigated. Thor's unified memory makes
    worker processes genuinely costly, so this is a real possible outcome.
- ⛔ **Do not add workers on the strength of "zero workers is obviously bad".**
  Thor inverts both A40 batching instincts already (`CLAUDE.md:189`).

### 4.3 The HAL contract (shape)

```
Machine(name)  ->  resolves: device, autocast dtype, thread count, allocator
                   config, OPS_DIR, cache roots, LRU size, sampling policy,
                   grad-checkpoint policy
               ->  records ALL of it into config.json  (R2/R4/R7)
               ->  NEVER changes numerics silently; any fallback is a banner
```

Machines: `thor` · `devbox` · `colab` · `hf` (metered ⇒ asks) · `pod` (legacy).
⛔ The HAL **sets** knobs; it never invents defaults that differ per machine
without recording them, because that is how two "identical" arms diverge.

---

## 5. DONE-MARKER & SUPERVISOR DISCIPLINE

### 5.1 Done-marker
`<out>/summary.json` with `done: true` (`:4522-4536`), written **in the same turn
the run finishes**, *after* `run_stage_gate` (`:4515`). Consumed by
`supervise_run.sh:84` and `resume_guard` (`:4577-4590`).

⛔ **A supervised run that never writes its done-marker is RESURRECTED FOREVER.**
Measured: a finished run relaunched for two days. Hence `--gate-probes` is
preflighted (`:5879`) — a gate crash between "training done" and "marker written"
is the failure mode that costs GPU-days.

**Verified working, 2026-08-23:** champ30k wrote `summary.json` with
`"done": true`, `steps: 30000`, `elapsed_s: 14774.7`. Discipline held.

### 5.2 Supervisor rules (each earned)
| rule | why | anchor |
|---|---|---|
| **the manifest is read ONCE**, at supervisor startup | editing `runs.d/<run>.env` under a live supervisor changes NOTHING; it replays the captured `TRAIN_CMD` | `supervise_run.sh:23` |
| **kill the SUPERVISOR first**, then the trainer | killing the trainer first makes the supervisor restore the stale command | `CLAUDE.md:209-213` |
| **wait before restarting** a supervisor | a new one races the old `flock`, prints "another supervisor holds …lock", dies, and leaves NOTHING running while the log looks normal | `CLAUDE.md:226-228`; `supervise_run.sh:54-56` |
| **kill by explicit PID** | `pkill -f <trainer>` self-matches your own ssh command | `CLAUDE.md` traps |
| fails closed if `pgrep` missing | refuses rather than double-launching | `supervise_run.sh:109-112` |

⛔ **There is no v6 manifest banked in the repo** — `v6_chain.py:manifest_text`
generates them on the fly (`:1877-1915`). ⇒ a supervised v6 run is reproducible
only if the generated manifest is banked with the run. **BACKLOG item.**

---

## 6. MULTI-STAGE TRAINING (S-W → S-T → S-S → S-J)

The ladder is already properly specified in code; P4's job is to keep it
*rulable*, not to redesign it.

| axis | table | anchor |
|---|---|---|
| trainable groups per stage | `STAGE_GROUPS` / `stage_trainable_groups()` | `v6.py:4160,4169` |
| freeze | `apply_stage_freeze` — partitions EVERY parameter, raises if one escapes | `v6.py:4196` |
| `λ_plan` | `{"S-W":0.0,"S-T":1.0,"S-S":0.0,"S-J":1.0}` | `:506-508` |
| loss weights in force | `V6LossWeights.for_stage` zeroes what a stage cannot train | `:286` |
| required predecessor | `{"S-W":None,"S-T":"S-W","S-S":"S-T","S-J":"S-S"}` | `:335-337` |
| gate spec | `STAGE_GATE_SPEC` | `:515+` |

**Gate semantics (correct as built):** `INCONCLUSIVE` is NOT-PASS (`:2711-2713`);
`NOT_APPLICABLE` is excluded entirely and never counts as satisfied
(`:2716-2720`); `pass:false` has **no override**; `pass:null` needs
`--allow-inconclusive-gate` **plus** a non-empty `--gate-off-reason` (`:5887`).

⛔ **The live risk is §3.1**: with `--spectrum-accum 1`, O6 returns INCONCLUSIVE
by construction, and INCONCLUSIVE-as-NOT-PASS then blocks the ladder for an
instrument reason. Fixing §3.1 is what keeps the ladder moving.

---

## 7. FRONTIER-METHOD SEAM

The proven method library (RL online/offline, GRPO, DPO, GSPO + variants) is
spec'd in **`METHOD_LIBRARY.md`** (same directory). P4's structural requirement:

- A new objective enters as a **loss term with a weight flag** (the existing
  `--w-*` pattern) and a `for_stage` entry, so it is stage-gated like everything
  else and lands in `config.json` for free via R1.
- ⛔ Any method needing **online rollouts** needs a closed-loop environment. Ours
  is not routinely available (AlpaSim ran bare on an A40; the eval pod is gone).
  A method whose data requirement we cannot meet is marked
  `ADOPT-LATER (blocked on X)` — never quietly mapped onto a proxy signal.

---

## 8. ACCEPTANCE TESTS (what makes this SPEC "done")

| id | test | passes when |
|---|---|---|
| T-P4-1 | config completeness regression | a run's `config.json` contains all 174 dests **and** R3/R4/R5/R7 provenance |
| T-P4-2 | gate can-rule | launching with O6 required and `--spectrum-accum 1` **aborts** with exit 2 |
| T-P4-3 | code freshness | the ship helper refuses to launch when local≠remote md5 |
| T-P4-4 | done-marker | a run that finishes writes `summary.json{done:true}` in the same turn; a killed run does not |
| T-P4-5 | HAL parity | the same config on two machines produces configs differing ONLY in the recorded machine block |
| T-P4-6 | deliberate regression | `TANITAD_RESIDUAL_INIT_SCALE=1.0` is visible in `config.json` and banner — the guard is shown ABLE TO FAIL |

---

## 9. OPEN DEFECTS FOUND WHILE SPEC'ING

Ranked, with the full list and costs in `BACKLOG.md`.

| # | defect | anchor | severity |
|---|---|---|---|
| D1 | rank-gate CAN-RULE check prints but cannot abort; `--spectrum-accum` defaults to 1 | `:3190`, `:5351` | **HIGH** — silently unrulable gates |
| D2 | no run provenance (git SHA, torch/CUDA, GPU, hostname, resolved device, start time) | §C3 of inventory | **HIGH** |
| D3 | `ds_val` is built, printed, and **never read** — `--v2-val-cache` mounts a corpus nothing consumes | `:3795-3799` | **HIGH** — a val cache that looks wired and is not |
| D4 | `--fps` does not exist; `getattr(a,"fps",10)` hardcodes the seam dump to 10 Hz while `--dt` is live | `:4486` | MEDIUM |
| D5 | `torch.bfloat16` hardcoded, no support probe / no fp16 fallback | `:4278-4279` | MEDIUM |
| D6 | `OMP_NUM_THREADS` set after `import torch` | `:6031` vs `:100` | LOW |
| D7 | no v6 manifest banked in the repo | `runs.d/` | MEDIUM |
| D8 | `supervise_run.sh` `OPS_DIR=/workspace` wrong on Thor | `supervise_run.sh:32` | MEDIUM |
| D9 | stale comment: "the trainer + its DataLoader workers" (there are none) | `supervise_run.sh:127-129` | LOW |

---

## 10. INSTRUMENT-BANKING RULE (new, earned this session)

⛔ **An instrument that lives only in a scratchpad is not an instrument.**

Measured 2026-08-23: the val-rank probe behind five registered claims existed
ONLY at `thor:/tmp/vp2.py`; the entire E-TRUNK-2/3 decodability battery — the
instrument behind the programme's central T0 conclusion — lives only in a
**temp scratchpad of a previous session**. A `/tmp` sweep destroys both.

**Rule:** any script that produces a number quoted in `GOALS_AND_CLAIMS.md`,
`MODEL_REGISTRY.md` or the paper is banked into the repo under `stack/scripts/`
(or the work package's `code/`) **in the same turn it first produces that
number**, with its comparability contract written into the docstring.

First application: `stack/scripts/val_rank_probe.py` (recovered, staged).
