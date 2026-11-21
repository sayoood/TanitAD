# RL pilot config contract — closing a dormant hazard while it is dormant

**2026-09-07 · Architecture & Inference · `agent/arch-inf-20260803`**
Tier **T0**, training-side. ⛔ **No model produced a trajectory in this work — there is no
eval tier and no four-family table here, and nothing below is a driving claim.**
⛔ **Zero GPU**: every model built and every checkpoint read on CPU
(`CUDA_VISIBLE_DEVICES=""`); the A40 refcv5-v2 run was not touched.

## The gap

`stack/scripts/rl_pilot_refc21.py::load_model` rebuilt `refc.RefCModel(refc.refc_config())`
from **defaults, unconditionally**, and read `ck["cfg"]` / `ck["config"]` **zero times**. Its
two guards — `EXPECT_PARAMS` and `load_state_dict(strict=True)` — are both **SHAPE** guards,
and the divergence that matters is **BEHAVIOURAL**.

### MEASURED: the two existing guards are blind by construction (proof case T0)

Building `refc_config()` with `anchors.v0_conditioned` flipped, on CPU:

| | `v0_conditioned=False` | `v0_conditioned=True` |
|---|---|---|
| `sum(p.numel())` | **104,191,577** | **104,191,577** |
| `state_dict` entries | **488** | **488** |
| key names **and order** | — | **identical** |
| every tensor shape | — | **identical** |
| `load_state_dict(strict=True)` False-weights → True-model | — | **OK** |

⇒ a shape guard cannot see a behaviour flag. Not "unlikely to" — **cannot**.

## Which of the three cases the real cold start falls into — established by loading it

`C:/Users/Admin/tanitad-data/models/refc-base-30k/ckpt.pt`, 1,250,838,325 B,
md5 **`8f10d6f934f4199e11ddc7352e074939`** (matches `MODEL_REGISTRY.md` §4.3), `step` **29999**.

* **Inside the checkpoint** the top-level keys are exactly **`['model', 'opt', 'step']`** —
  **no config of any kind**. Consistent with `refc_train.py::_save_ckpt` and
  `refc_v3_train.py`, both of which save `{model, opt, step}`.
* **Beside it** there is a sidecar `config.json` carrying a `cfg` — a **complete-for-its-era**
  `dataclasses.asdict` dump from 2026-07-20 (`refc_train.py:1008`).

⇒ **HYBRID, and it is the LIVE path, not a hypothetical.** Of the rebuilt config's **96**
leaves the checkpoint's record carries **38** — **all 38 AGREE, 0 disagreements** — and
**58 are absent**, including `anchors.v0_conditioned` itself. The fields that appeared after
July are absent because **the fields did not exist**, not because the dump was sloppy.

⇒ the three cases in the brief are not mutually exclusive on a real checkpoint. The contract
therefore splits **per field**: COMPARED, or ABSENT-and-STATED.

## What was built

A **comparison that refuses** — not a new source of truth. ⛔ The pilot still builds from
`refc_config()`; it never rebuilds from a stored config, because a stored config that is
wrong or partial would just swap one unverified source for another.

1. **AGREE** → proceed, and **stamp `<out>/config_contract.json`** with every field compared.
2. **DISAGREE** → `SystemExit` naming the field, **both** values, and **what it would silently
   change** (`raw/REFUSAL_v0_conditioned.txt`).
3. **ABSENT** → neither proceed silently nor refuse: print `ASSUMED <field> = <value>
   (UNVERIFIED)` for each behaviour-affecting field and stamp all of them.
4. **Type-confusion guard.** A `cfg` key is not proof of a *model* config: this pilot's own
   `ckpt_after.pt` stores `PostTrainConfig.to_dict()`, whose keys share **nothing** with
   `RefCConfig`. Chained back in as `--ckpt` it is rejected **with the reason recorded**,
   not silently compared.
5. **Weight evidence for the one field that matters.** `decoder.anchor_controls` is a
   persistent buffer registered unconditionally (`refc.py:1400`), so the state dict itself is
   evidence: **absent** ⇒ predates the buffer ⇒ predates `v0_conditioned`; **all-zero** ⇒ the
   fixed bank's signature (`refc.py:2093-2096`). ⚠️ The inference is **one-directional on
   purpose** — `load_anchors` (`refc.py:1643-1649`) copies controls *without* consulting the
   flag, so "non-zero while running fixed" is **NOTED, never refused**; refusing it would be a
   false positive on a legal build. Only `True` + absent/all-zero is a hard contradiction, and
   that refuses. ⛔ It never *sets* the flag.

## Mutation proof — `code/mutation_proof.py`, **79/79 PASS**, exit 0

Run against the **landed** repo file. ⛔ Every expectation is a hard-coded literal; not one
case has the shape `if mismatch: expect refusal; else: expect pass`.

| | case | expectation (hard-coded) | result |
|---|---|---|---|
| T0 | control: shape guards blind | counts/keys/shapes identical, strict cross-load `OK`; 34 consequence entries, **0 orphans** | 9/9 |
| T1 | **disagree → REFUSE** | `SystemExit`; names `anchors.v0_conditioned`, `checkpoint says True`, `this run would use False`, `0.3773`, `0.2610`, the 10 m/s sentence, `ACTION SPACE THE CHECKPOINT WAS NEVER TRAINED IN`; stamp `DISAGREE`/`REFUSED` | 16/16 |
| T2 | **agree → PASS** | no exception, model returned, stamp `AGREE`/`PROCEEDING`, **96/96** compared, `{}` disagreements, `{}` assumed, weights load | 12/12 |
| T3 | **absent → STAMPED** | model returned, **stamp file exists**, case `ABSENT`, 96 assumed, 34 behaviour-critical, stdout carries `ASSUMED anchors.v0_conditioned = False  (UNVERIFIED)` | 15/15 |
| T4 | PostTrainConfig as `cfg` | case `ABSENT`, provenance `None`, rejection names the source **and** the reason | 6/6 |
| T5 | weights contradict an assumption | `SystemExit`; stamp `REFUSED (v0 weight evidence)` | 7/7 |
| T6 | **the real cold start** | `PARTIAL`, step 29999, **38** compared / **0** disagreements / **58** assumed, `v0_conditioned` assumed `False` and `CORROBORATED` via `BUFFER_ABSENT` | 14/14 |

⭐ T1 and T2 differ in **exactly one of the same 96 leaves**, so the passing half is not
vacuous and the refusing half is not indiscriminate.

## ⛔ Separate, PRE-EXISTING defect found by loading the cold start — NOT fixed here

The pilot **cannot load its own cold start today**, for a reason that has nothing to do with
this change:

```
RuntimeError: Error(s) in loading state_dict for RefCModel:
	Missing key(s) in state_dict: "decoder.anchor_controls".
```

The July checkpoint has **487** state-dict entries; today's `refc_config()` model has **488**.
`decoder.anchor_controls` was added as an unconditional persistent buffer by the same
2026-09-04 change that introduced `v0_conditioned`. So `p_rc21_chain.sh` as written would die
at load.

⚠️ **The tempting fix is the dangerous one.** Relaxing `strict=True` to `strict=False` makes
the crash go away and simultaneously deletes the only guard that currently notices anything —
and it would let a `v0_conditioned=True` build run on an **all-zero** control vocabulary, which
`refc.py:2091` already calls *a plausible-looking WRONG experiment*. The contract landed here
refuses that specific combination (T5), but the registration decision belongs to whoever owns
`refc.py`. ⛔ **Escalated, not patched** — `refc.py` was held read-only.

⭐ Ordering note: the contract runs **before** `load_state_dict`, so on the live path the
operator gets the stamped assumption banner **and the banked `config_contract.json`** first,
then the strict-load failure. Evidence survives the crash.

## Can the pilot still silently post-train on an action space its checkpoint was never trained in?

**No — not silently.** Every path now ends in a written record: a disagreement refuses by name
with its consequence, an agreement is stamped field-by-field, and an absent field is printed
and stamped as `ASSUMED … (UNVERIFIED)`. ⚠️ **It can still post-train on an *unverified* action
space** — on the real cold start 58 of 96 fields, `v0_conditioned` among them, have no
recorded value to check against, and for that one field the checkpoint's own weights
corroborate the assumption while the other 57 rest on defaults alone. **The silence is closed;
the uncertainty is now written down instead of hidden.**

## Files

* `stack/scripts/rl_pilot_refc21.py` — the contract (`CFG_CONSEQUENCE`, `_cfg_leaves`,
  `_stored_cfg`, `_v0_corroboration`, `assert_config_contract`; `load_model` now takes the
  run's `out_dir` so the stamp lands with the run).
* `code/mutation_proof.py` · `raw/mutation_proof.log` · `raw/mutation_proof.json`
* `raw/REFUSAL_v0_conditioned.txt` — the verbatim refusal text
* `raw/LIVE_PATH_real_coldstart.txt` — the verbatim live-path output
* `raw/config_contract_real_coldstart.json` — the stamp the real cold start produces
