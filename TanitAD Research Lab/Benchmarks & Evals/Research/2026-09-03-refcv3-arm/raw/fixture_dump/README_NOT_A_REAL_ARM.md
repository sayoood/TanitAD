# ⛔ THIS IS A SYNTHETIC FIXTURE. IT IS NOT A REAL ARM.

**Placed here 2026-09-06 by the dump-banking pass** (`…/Architecture & Inference/Research/2026-09-06-dump-banking/`).
Nothing in this directory was edited; this file was added beside the fixture so the warning is where
someone would trip over it.

`fixture_dump/` is a **synthetic 3-episode slice produced on a random-init model**. It must never be
quoted as evidence about any trained checkpoint, and never compared against a real arm.

**MEASURED from its own `manifest.json`:**

| field | this fixture | a REAL arm (the refcv4b landing dump) |
|---|---|---|
| `model.ckpt` | a `…\Temp\claude\…\<other-session-id>\scratchpad\out\fixture\run\ckpt.pt` | `/workspace/experiments/refcv4b-b1-v72-40k/ckpt_40284_FINAL.pt` |
| `model.step` | **11** | **40284** |
| `grid.n_episodes` | **3** | **141** |
| `grid.n_windows` | **42** | **4,823** |
| `model.n_anchors` | **20** | **117** |
| `_unverified` present | **True** | ⛔ **True — the identical string** |

## ⛔⛔ Do NOT identify this fixture by its `_unverified` key

The self-label *"UNVERIFIED on a real checkpoint … random-init RefCV3Model at `refc_v3_smoke_config`
… synthetic 3-episode slice only"* is **BOILERPLATE emitted by `taniteval/tools/refcv3_arm.py`**.
MEASURED 2026-09-06: it is present on **every** refcv3-family manifest we hold — **including the real
refcv4b landing arm** at step 40284 over 141 episodes / 4,823 windows, whose `state_dict_load` reports
`missing_keys: []` and `unexpected_keys: []` (a random-init model loads no state dict). It is absent
only from the refav1 dump, which a **different tool** wrote.

⇒ **Using `_unverified` to tell a fixture from a real dump would misclassify every real refcv3-family
dump in the programme.** The discriminators that actually work are `model.step`,
`grid.n_episodes` / `grid.n_windows`, and `model.n_anchors` — the table above.

*Same family as the `df` / Thor `free` / cgroup `usage_in_bytes` scope traps: a TRUE statement (the
tool **was** validated only on a fixture) quoted outside its scope, where it reads as a fact about the
**artifact** rather than about the **tool**.*

**Register rows:** `D-BANK-TEMP-1b` (and `D-BANK-TEMP-1`) in `Project Steering/GOALS_AND_CLAIMS.md`.
**Real, verified dumps live at:** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-06-dump-banking/raw/*.tgz`,
each with md5, content digest and identity in `raw/BANK_MANIFEST.json`.
