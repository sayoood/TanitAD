# The v6 ladder edges, EXECUTED

**Date:** 2026-08-16 · **Branch:** `agent/arch-inf-20260803` · **Base:** `8e215b3`
**Compute:** CPU only, tiny `V6Stack` (~1000× fewer params, identical wiring). No GPU touched;
Thor's 336M run was not disturbed.
**Suite:** `2995 passed / 0 failed / 17 skipped / 2 xfailed` (baseline 2919 + this file's 26 +
50 from two other agents' staged files). MEASURED.

---

## Why this was worth doing

Three ladder-edge defects were found in the two days before this, and **every one was found by
running the transition, none by reading the code**. The remaining edges had never been executed
end-to-end. The prior was not good, and it held: **three more defects, all on the resume path.**

The pattern repeats exactly. Each defect is a guard that *looks* present and *is* absent — or a
guard that fires for a reason unrelated to what it is supposed to check.

---

## Results, edge by edge

| edge | verdict | evidence |
|---|---|---|
| S-T →`--init-from`→ S-S | **CLEAN** | `missing=[] unexpected=[] introduced=[]`, `prev_stage="S-T"` |
| S-S →`--init-from`→ S-J | **CLEAN** | same |
| ⛔ **`--resume auto` across a stage boundary** | **DEFECTIVE** | no stage check existed; caught only incidentally |
| ⛔ **`--init-from` + `--resume auto`** | **DEFECTIVE, SILENT** | `config.json` recorded the wrong ancestor's md5 |
| ⛔ **`--init-from <fp16 snapshot>`** | **DEFECTIVE** | refused, blaming geometry for an unopened container |
| freeze × init (all 4 stages) | **CLEAN** | real backward, per-group `.grad` census, + vacuity control |
| X3 isolation after `--init-from` | **CLEAN** | 0 live forbidden edges, 19/54/121 params probed |

`stack/tanitad/models/v6.py` needed **no change** — edges 4 and 5 were clean. Every fix landed in
`stack/scripts/train_v6_staged.py`.

---

## 1 ⛔ The stage-boundary `--resume auto` — the priority edge, and it bit

### What was measured

`load_resume` did a strict state-dict load and adopted `ck["step"]`, **with no stage check at all**.
The load *succeeds* across the ladder: every stage saves the WHOLE `V6Stack`, so an S-T checkpoint
is key-for-key loadable into an S-S run. The stage label was already in the file the whole time —
`_run_config` writes `"stage": a.stage` and `_save_ckpt` stores it — and nothing read it.

MEASURED, S-T ckpt (step 30000) resumed into an S-S run:

```
resume_guard -> {'mode': 'resume', 'from': '.../ckpt.pt'}     # never looks at the stage
load_resume  -> ValueError: loaded state dict contains a parameter group
                that doesn't match the size of optimizer's group
```

**That refusal is not a guard.** It is `torch.optim` complaining about list lengths, and it is
worthless in three separate ways:

1. **It names nothing.** An operator reading it goes to look at the optimiser. The problem is the
   ladder.
2. **It holds only by numeric coincidence.** MEASURED trainable-*tensor* counts at the production
   geometry: **S-W 240 · S-T 80 · S-S 54 · S-J 374**. One `STAGE_GROUPS` edit that makes two stages
   share a count and the resume passes **silently** — S-T's `exp_avg` for `layer_tac` landing on
   `layer_str` by list position.
3. ⛔ **It is skipped entirely when the checkpoint has no `opt` key** — `if opt is not None and
   "opt" in ck`. That is precisely the shape of `ops/ckpt_fp16_snapshot.py`, the documented
   pod-handover artifact ("what makes a pod handover survivable"). On that path there was no
   barrier of any kind; it died on a bare `KeyError: 'stack'`.

It also failed **late**. `load_resume` sits ~130 lines after `resume_guard`, *after* episode
selection, dataset windowing and the O4 saliency pass over every window in the corpus. A wrong-stage
resume had already paid for all of it.

This is the C13 family: **a guard that passes for a reason unrelated to what it is meant to check.**

### The fix — explicit data plus a refusal, not a relaxed check

* **`RESUME_CONTRACT`** — the three things a resume requires of a checkpoint (`same_stage`,
  `labelled`, `has_optimiser`), each carrying its mechanism, in the style of
  `STAGE_INVALIDATION_MECHANISM`. A refusal with no mechanism teaches nobody.
* **`read_ckpt_provenance()`** — reads stage/step/`has_opt` via `torch.load(..., mmap=True)`, so the
  tensor bytes are never touched. MEASURED 0.005 s vs 0.024 s on a 52 MB file; on a 3.5 GB `ckpt.pt`
  this is the difference between a transient full-checkpoint allocation and none — and it runs on
  pod2, which is RAM-bound (~54/55 GB cgroup). Never raises: an unreadable file is *reported* as
  `readable: False` so the caller refuses with a diagnosis rather than a pickle traceback.
* **`assert_resume_lineage()` / `ResumeLineageError`** — called from `train()` **immediately after
  `resume_guard`, before the corpus build.**
* **`load_resume(..., stage=...)`** — defence in depth. A caller that does not name a stage gets the
  old unchecked path, the same contract `load_stage_init` uses for its allowance: *a check must be
  asked for, never inherited by default*. `train()` always names it.

What the operator sees now (MEASURED):

```
[v6] ⛔ --resume auto found .../ckpt.pt written by stage 'S-T' at step 30000,
     but this run is stage 'S-S'.
  the checkpoint's `config.stage` must EQUAL the stage being launched. Every stage saves the
  whole V6Stack, so the state_dict load cannot tell them apart — the stage label is the only
  thing that can, and `_save_ckpt` has always written it.
  ⚠️ Nothing downstream would have caught this reliably: the state_dict load SUCCEEDS across the
  ladder, and the only accidental barrier — the optimiser's param-group size — holds solely
  because the stages happen to train different numbers of tensors (S-W 240 · S-T 80 · S-S 54 ·
  S-J 374, MEASURED). The run would have adopted step 30000 and replayed the LR schedule to the
  wrong point.
  ⇒ this is an --init-from, not a resume. Point --out at a fresh directory and pass --init-from …
```

**The strongest pin** is `test_the_refusal_does_NOT_depend_on_the_optimiser_shape`: it builds a
checkpoint labelled S-T whose optimiser was constructed over **S-S's** trainable set, so the
param-group sizes *collide* and the old accidental barrier is defeated. That is the silent
wrong-stage resume the old code would have performed. The stage check refuses anyway — it never
looks at an optimiser. A companion test pins that the four stage counts are currently distinct, so
the day an edit makes two collide, the suite says so rather than the ladder quietly losing its
fallback.

---

## 2 ⛔ `--init-from` + `--resume auto` — a silent provenance lie

`train()` runs `load_stage_init` first (l. 1242) and `load_resume` afterwards (l. 1372). When both
flags are present the resume **overwrites every weight the init loaded** — while `config.json` still
carried the init's `trunk_md5_after_load`.

MEASURED, two different ancestors:

```
config.json recorded  init.trunk_md5_after_load = fbce009ab9fe1b064ab9c44dccf1dc6b
the trunk actually in the model after the resume = 326034884273a8f459855ce515bdfb95
                                          SAME?   False
```

…and nothing printed a warning. **The run row names an ancestor the run is not standing on** — the
exact failure `MODEL_REGISTRY.md` exists to prevent, and the reason "three errors propagated for
days because they were copied from prose".

⚠️ **Refusing the flag combination would be the wrong fix.** `supervise_run.sh` replays the
`TRAIN_CMD` it captured at supervisor startup, so the relaunch that resumes *necessarily* still
carries the `--init-from` that seeded the run. The flag pair is normal operation; the lying record
was the defect.

**Fix:** `supersede_init_on_resume()` demotes the init report in place — `init_from: null`,
the evidence preserved under `superseded_by_resume` with an explicit `_status: "OVERWRITTEN …"`, and
`resumed_from` naming the real lineage — plus a printed warning. The evidence is demoted, never
deleted: the launch really did do it.

---

## 3 ⛔ `--init-from <fp16 snapshot>` — refused, blaming the wrong thing

`ops/ckpt_fp16_snapshot.py` states its snapshot *"is enough for the P-battery, any eval, and
`--init-from`"*. **MEASURED: it was not.** The writer stores `{"model", "_meta",
"_fp16_weights_only"}`; both readers looked only for `"stack"`. `load_stage_init` fell through to
the wrapper dict and refused with:

```
[v6] ⛔ --init-from …/weights_fp16.pt is not a valid predecessor for stage 'S-S'.
  missing (NOT introducible at this stage): ['act_head_lat.arg_head.bias', … ]   (400+ keys)
```

A **geometry mismatch** blamed for an unopened container — a message worse than a crash, because it
sends the operator to look at the architecture. This is the path a rebuilt pod actually takes: the
3.53 GB `ckpt.pt` never once pushed to HF.

**Fix, on the reader side** (`ops/` untouched, and the fix makes its docstring true):

* `load_stage_init` unwraps the snapshot and reads `step`/`config` from `_meta`;
* the report gains `init_source` and `init_precision: "fp16->fp32 (lossy)"` — **stated, never
  hidden**, because a trunk md5 that has been through half precision cannot equal the fp32 source's
  and would otherwise look like a lineage break the next time md5s are compared;
* `load_resume` **refuses** it, naming the container (`no 'stack' key`) and pointing at
  `--init-from`. A weights-only file is an init artifact, not a resume point — and closing that door
  is what stops the fp16 path from becoming a *silent* wrong-stage resume the moment the key
  mismatch was fixed.

---

## 4 ✅ S-T → S-S and S-S → S-J execute clean — and the empty allowance is *correct*

Both ran key-for-key: `missing=[] unexpected=[] introduced=[]`. `STAGE_MAY_INTRODUCE["S-S"] == ()`
and `["S-J"] == ()` are **right**, and now demonstrated rather than asserted: S-T's checkpoint
already carries `layer_str` because every stage saves the whole stack, so S-S introduces nothing.

That distinction matters — declaring an empty allowance over a transition that actually *needs* one
is precisely the S-W → S-T defect (`8e215b3`), one rung up. The test asserts the allowance is empty
**and** that a real load over that edge introduces nothing, so the two cannot drift apart.

The drifted-flag case is still refused: launching S-S with `--selector none` against an S-T
checkpoint that has one is a mis-specified arm, not an introduction.

---

## 5 ✅ Freeze × init — clean, and pinned NON-VACUOUSLY

For all four stages: `--init-from` a real predecessor → `apply_stage_freeze` → the **real**
`v6_loss_step` → backward → per-group `.grad` census.

* `requires_grad` matches `stage_trainable_groups` exactly — **0 mismatches**.
* Every out-of-stage group has `.grad is None` — **0 leaks at every stage**.

⚠️ **The vacuity control is what makes that a result.** "Group X received no gradient" proves the
freeze only if the same loss *reaches* X when X is unfrozen; otherwise a loss that simply never
touches a module certifies it as frozen. MEASURED with every parameter trainable under the S-J loss:

| group | reachable |
|---|---|
| encoder | 17/17 |
| readout | 2/2 |
| aux | 30/30 |
| predictor_op | 31/35 |
| layer_tac | 54/67 |
| layer_str | 41/54 |
| planner | 2/13 |

Every group has live parameters, so every "no grad" above is a real statement. The fixture asserts
this and fails if any group becomes unreachable — the same discipline `V6Stack.assert_isolation`
applies to `requires_grad`, because **vacuous passes are how a guarantee rots**.

---

## 6 ✅ X3 isolation survives the transition

`assert_isolation` passes after `--init-from` into S-S and S-J: `planner_to_encoder 0`,
`tactical_to_below 0`, `strategic_to_below 0`, with **19 / 54 / 121** parameters probed (non-vacuity
checked). X3 had only ever been measured at fresh construction — and a stage never starts from
fresh construction. An isolation guarantee measured only on random init is a guarantee about a
model nobody trains.

---

## Escalation

1. **`ops/ckpt_fp16_snapshot.py` needs no code change, but its docstring's `--init-from` claim was
   false until this landed.** It is true now via the reader. If anyone re-reads that file, the claim
   is load-bearing and now backed by `test_init_from_CAN_now_read_the_snapshot_its_docstring_promises`.
2. **No chain script wires these stages yet** (`stack/scripts/` has no `v6_chain.sh`). When one is
   written it must give each stage its **own `--out`**; the refusal now makes a shared directory
   fail loudly at launch instead of quietly at step 0.
3. **`RETRACTION_LOG.md`** — three entries are warranted, all one root-cause class:
   *a guard that fires for a reason unrelated to what it checks* (the optimiser param-group barrier;
   the same family as the `df` / Thor `free` / cgroup `usage_in_bytes` traps, and as C13).

## Deliverable manifest

| artifact | repo path | state |
|---|---|---|
| the three fixes + `RESUME_CONTRACT` | `stack/scripts/train_v6_staged.py` | staged |
| 26 tests pinning all five edges | `stack/tests/test_v6_ladder_edges.py` | staged (new file) |
| this writeup | `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-16-ladder-edges/LADDER_EDGES.md` | staged |

`stack/tanitad/models/v6.py` was read and **not modified** — edges 4 and 5 were clean.

**Evidence class:** every number above is **MEASURED (ours)**, produced on this dev box by the
probes in the scratchpad and by `stack/tests/test_v6_ladder_edges.py`, which reproduces each one.
