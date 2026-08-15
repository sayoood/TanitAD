# v6F S-W resume on Thor — the tooling, and what each piece exists to prevent

**Status 2026-08-16 ~00:25 CEST:** train cache **VERIFIED CLEAN BY BYTES** (2,403/2,403,
size-mismatch **0**, 12 random episodes `torch.load` OK); val cache pulling (~88/603).
Trainer not yet launched.

## Pre-launch verification already passed (MEASURED on Thor)

| check | result |
|---|---|
| rebuild from the banked `config.json["args"]` | **336,542,025 params** — config E, budget 350 M |
| per-group | encoder 87.28 · readout 0.1 · predictor_op 189.96 · layer_tac 29.32 · layer_str 27.44 · planner 0.79 · aux 1.65 (M) |
| X3 isolation | `pass=True` · `{planner_to_encoder: 0, tactical_to_below: 0, strategic_to_below: 0}` |
| strict load of the step-6250 checkpoint | **OK, 573 tensors** (matches the stop runbook) |
| emission bound | `emission.squash='squash'` — the `_squash` fix, live |
| stack currency | proven by a **real import** of `train_v6_staged` + `V6Stack`, and a **real conv2d on CUDA** (torch 2.13.0+cu130) — never `git log` |

## The four scripts

**`thor_v6_pull.py`** — pulls the checkpoint + both w120 caches, then verifies **by loading**.
`max_workers=4`: the first attempt ran two transfers concurrently and took the box off the
network entirely.

**`thor_v6_launch.py`** — ⭐ **replays the banked `config.json["args"]` verbatim**, swapping only
`out` / `v2_cache` / `v2_val_cache`. It does not retype flags. Retyping is how an argument
containing spaces gets lost — the `--heldout-off-reason 'PI directive: …'` class — and here it
would silently change the architecture and turn a strict resume into a refused one. Refuses to
launch if `ckpt.pt` is absent from `--out`.

**`thor_v6_chain.sh`** — sequences pull → verify → launch, and **refuses to launch unless the pull
log reports `ALL DONE`**. Sequential by design.

**`thor_verify_caches.py` + `thor_postlaunch_gate.sh`** — ⛔ **the check the chain lacks.** The
chain's own verification loads *one* mid-episode per split. That cannot see a **short** shard, and
this programme has been bitten by exactly that gap twice in two days: `batch_00184` looked present
with 4 of N SAM3 records (**14×** understated), and the A2 card claimed "one row missing" against a
measured **356** (**356×**). ⇒ *A listing probe sees a MISSING file but never a SHORT one.*

The verifier compares **every** shard's size against the far-side listing and then loads a sample.
The gate runs it after the launch and stops the trainer **by explicit PID** only on an attributable
corruption; an unreachable far side is **UNKNOWN, never a failure**. A false kill costs a restart;
a missed truncation costs 4.8 days of training on corrupt data — and would surface as *"the model
got worse"*, not as an error.

This mattered concretely: the pull was interrupted when Thor dropped off the network, and
`huggingface_hub` logged `Invalid metadata file … Removing it from disk and continue` for several
shards. It did re-fetch them — **and that is now proven by bytes rather than assumed.**

## What is measured next

Marginal **s/step over ≥3 logged points** (`--log-every 50`, so ≥150 steps) against the A40
baseline **17.37 s/step** (`STOP_2026-08-15_RESUME_RUNBOOK.md` §1). ⚠️ Not a first-call number and
not a single line — both traps are in the preflight.
