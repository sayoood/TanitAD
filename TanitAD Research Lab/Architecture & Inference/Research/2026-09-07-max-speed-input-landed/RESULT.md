# `--max-speed-input` IS LANDED — and it cleared the rollability battery

**Date** 2026-09-07 (Europe/Berlin) · **Branch** `agent/arch-inf-20260803` · **Agent**
Architecture & Inference · **Supersedes the status of**
`.../Research/2026-09-06-max-speed-input/WIRING_DIFF.md` (that patch is now APPLIED,
re-derived against the file as it stands, not fuzzy-matched).

---

## 0. THE DEFECT THIS CLOSES, IN ONE LINE

`stack/tanitad/refs/max_speed_input.py` (the channel), its 29 tests and its real
consumer `stack/tanitad/eval/speed_limit_scoring.py` were all in HEAD — and
`--max-speed-input` had **0 hits** in `stack/scripts/refc_v3_train.py` (control:
`add_argument` = 89). **A channel with no flag is a channel no run can use.** That is
exactly what made `--tac-goal-tok-head` silently gate the entire tactical vocabulary:
the head existed, `refc_v3.py:948` gated on a config field, and with no CLI flag that
field could never be set.

`add_argument` is now **91**. The flag parses, pins onto the *config* (not only onto
`args`), reaches the model, feeds the loader, and is stamped in `config.json`.

---

## 1. ⭐⭐ THE ACCEPTANCE BAR — ROLLABILITY, NOT "THE TESTS PASS"

Building a head on the vocabulary alone made refcv4b, three refcv3 checkpoints and a
LIVE refcv5 run **UNROLLABLE**. So this edge was held to the exact battery
`--tac-goal-tok-head` cleared before it was allowed into the 40k run. **Two arms were
trained for real** on the dev-box RTX 4060 (the A40 was never touched), 4 steps each,
`--arm hier --size small`, v2ep corpus `refav1-eval141`, labels
`s2_labels_v8_eval.jsonl.gz` (md5 `baf5dadca9feaef658c2261452338582`).

| requirement | ON (`--max-speed-input`) | OFF (control) |
|---|---|---|
| **1. strict load, 0 missing / 0 unexpected** | **0 / 0**, `tolerated_inert=[]` | **0 / 0** |
| literal `load_state_dict(..., strict=True)` on a fresh rebuild | **OK** | **OK** |
| **2. recorded `param_breakdown` == rebuilt** | **key-for-key, 10 lines** | key-for-key, 9 lines |
| lines sum to total | 63,183,997 == 63,183,997 | 63,158,525 == 63,158,525 |
| `max_speed_inject` line == measured module params | **25,472 == 25,472** | line absent, module absent |
| **3. REAL `refcv3_arm.cross_check_config`** | **PASS on 6 facts** | **PASS on 6 facts** |
| the ROLLED config carries the flag | `max_speed_input=True, mode=quantized` | `False` |

**CROSS-ARM:** ON total − OFF total = **25,472 = exactly the `max_speed_inject` line**
— the opt-in costs the conditioner and nothing else.

⚠️ **The instrument is the real one.** `refcv3_arm.load_model` was called, not
reimplemented: it runs `rebuild_config` → `cross_check_config` → strict load and
`SystemExit`s on any contradiction. The mirror's copy of `refcv3_arm.py` was **stale**
(`5e1349a` vs the repo's `6e87fd8`) and was refreshed from the repo before the run, so
the battery ran against the current instrument. Raw:
[`raw/roll_battery.txt`](raw/roll_battery.txt).

---

## 2. ⛔ THE OFF PATH IS BIT-IDENTICAL — AND WHAT THE MUTATION DOES *NOT* PROVE

`code/off_parity_proof.py` extracts the **pre-patch `refc_v3.py` from git HEAD itself**
(never a copy made earlier), builds both modules on the same fixed seed, runs the same
forward on the same frames, and compares every tensor.

```
BASELINE  state_dict     : 193 tensors bit-identical
BASELINE  forward outputs:  37 tensors bit-identical      -> BIT-IDENTICAL
MUTATION  state_dict     : KEY SET DIFFERS (+6 max_speed_cond.* keys)
MUTATION  forward outputs: bank_speed_pred NOT bit-identical -> DIFFERS (GOOD)
```

The mutation is the **D-ROLL-1 construction itself** — `if cfg.max_speed_input:`
replaced by `if True:` — i.e. the unconditional build that made every banked v7.0
checkpoint unrollable.

* ⭐ **WHAT THE MUTATION PROVES:** the comparator has **teeth**. This equality *can*
  go red, so the baseline PASS is a measurement rather than an assertion that cannot
  fail.
* ⚠️ **WHAT IT DOES NOT PROVE, STATED PLAINLY:** it does **not** show that a zero-init
  survived a mutation. **With the flag OFF nothing is constructed and nothing is fed**,
  so the identity is **STRUCTURAL, not a cancellation** — there is no zero-init doing
  work on the OFF path to be mutated. Claiming the OFF test "survived the mutation"
  would be a false strength claim. The zero-init claim belongs to the **ON** path
  (emission bit-inert at step 0) and is tested separately
  (`test_the_edge_is_BIT_INERT_at_init`, plus its `..._is_REACHABLE_once_trained`
  partner, because an edge that is inert at init *and* unreachable after training is a
  dead wire that reads as a clean ablation).

---

## 3. ⛔ THE REFUSALS — 20 PROBES, EACH WITH ITS POSITIVE CONTROL

A guard that always fires is indistinguishable from a broken build, so every refusal
is exercised beside the same call with the offending condition removed. Rows **1-14**
are the standalone probe's **20 probes** (14 refusals + 6 positive controls), all PASS —
raw: [`raw/refusal_probe.txt`](raw/refusal_probe.txt). Row **15** is exercised by `test_preflight_refuses_EARLY_and_says_it_is_not_the_flag_failing`.

| # | refusal | where | control |
|---|---|---|---|
| 1 | `--max-speed-input` without `--v7-labels` (dead switch that looks switched) | `_check_max_speed_args`, both entry points | with labels → allowed |
| 2 | `--max-speed-input` + in-training eval without `--eval-labels` | same | — |
| 3 | `--max-speed-mode raw` **without** `--max-speed-input` → **INERT** | `_pin_trainer_cfg` | with the flag → allowed |
| 4 | `--max-speed-input` on `--arm flat` (no `z_tac`/`ctx` to inject into) | `_pin_trainer_cfg` | hier → allowed |
| 5 | `max_speed_input=True` on a flat `RefCV3Config` | `RefCV3Model.__init__`, **before** the `not cfg.hier` return | — |
| 6 | a record that declares **no units** — error **names the field, the clip and the 1.61× spread** | `enable_max_speed` | declared → allowed |
| 7 | a record shipping a **different ladder** ("two ladders is two experiments") | `enable_max_speed` | agreeing → allowed |
| 8 | a record whose **shipped bucket** disagrees with this build's snap | `enable_max_speed` | agreeing → allowed |
| 9 | **no clip carries the block** — names it as a *v8 addition* (v7.2: 0/4,572) | `enable_max_speed` | — |
| 10 | `v_max_ms` supplied to a build with no seam → *SILENTLY DROPPED* | `RefCV3Model.forward` | — |
| 11 | seam built but nothing fed (a silent 0.0 says *"the limit is 0 m/s — stop"*) | `RefCV3Model.forward` | both present → allowed |
| 12 | the batch lacks `v_max_ms` on a `--max-speed-input` build | `compute_losses_v3` | — |
| 13 | reading the field **without `allow_oracle_nav`** | `v7_labels.oracle_max_speed` | with the stamp → allowed |
| 14 | record and weights disagree about the conditioner (**bidirectional**) | `assert_seams_are_built` | — |
| 15 | `--preflight --max-speed-input` — refused **early**, saying *"this is NOT the flag failing"* | `_check_max_speed_args` | preflight without the flag → PASS |

This matches the four refusals the trainer already implements (`--w-agent` under
`--agents off`, `--nav-args` without `--nav-from-v7`, both halves of the P14 selection
split, `--tac-goal-tok-head` under kin3).

---

## 3b. ⚠️ THE LATE FAILURE THAT WAS FIXED BEFORE IT COULD MISLEAD

MEASURED while exercising the SECOND entry point: `--preflight --max-speed-input`
built the model correctly — the ledger printed `max_speed_inject: 1448`, so **the gate
IS reached on this path too** — and then died **inside the loss step**, after the
freeze-history gate had already printed `pass: True`. Cause: `preflight` always builds
its corpus with `_synth_episodes` (CI-only), a synthetic clip carries no
`speed_max_input`, so `enable_max_speed` is never called and the forward correctly
refuses a batch with no `v_max_ms`.

The error was TRUE and its message named the exact cause — and it was still **wrong for
the reader**. An operator preflighting their real launch line before committing
GPU-days would have seen a red twenty lines in and concluded the flag was broken, which
is the *"true but wrong for the reader"* class: a correct claim that implies a wrong
next action, and one that verification cannot catch.

It now refuses at the TOP and says so in as many words (*"this is NOT the flag
failing"*), points at the arm-without-the-flag preflight for the build/delta/gates, and
names the channel's own preflight — the `enable_max_speed` census at `train()` startup.
Pinned by `test_preflight_refuses_EARLY_and_says_it_is_not_the_flag_failing`, with both
controls in the same breath.

---

## 4. ⛔⛔ THE ONE NEW MEASURED FACT: **DO NOT FEED THE SHIPPED BUCKET BACK**

The brief's rule — *score against the SHIPPED value, never a recomputed one* — has a
trap in the other direction, and it is quantified here for the first time.

The record ships **both** `v_max_ms` (2 dp) and `v_max_bucket_ms` (**4 dp**:
`13.8889`). The pinned ladder's 50 km/h step is `50/3.6 = 13.888888…`. **The shipped
bucket is strictly GREATER than the step it names**, so passing it through
`encode_block` snaps it UP to the next one.

> **MEASURED on `s2_labels_v8_train.jsonl.gz` (n = 4,572): feeding the shipped bucket
> back through the ladder moves 2,631 clips (57.5 %) one step up** — 50→70 on 1,593,
> 20→30 on 809, 100→120 on 229.

This is the *same mechanism* that manufactured 2,203 phantom violations from a
re-derived `v_hi`: a rounding difference in the third decimal read as a different
quantity.

**⇒ THE DESIGN.** The loader ships the **raw shipped `v_max_ms`**, in the record's
**declared** units, and the ladder is applied **exactly once** — inside
`encode_block`, on the model side, under the arm's `mode`. The shipped bucket is used
as a **cross-check, not as the value**: for every clip, this build's snap must
reproduce the record's `v_max_bucket_kmh`, and a disagreement is a refusal.

**Cross-check, run on the full corpus before the guard was written** (so the guard is
not a rule that fires on real data):

| blob | n | ladder mismatch | bucket mismatch | over-ceiling | undeclared units |
|---|---|---|---|---|---|
| `s2_labels_v8_train.jsonl.gz` | 4,572 | **0** | **0** | 28 (0.61 %) | **0** |
| `s2_labels_v8_eval.jsonl.gz` | 147 | **0** | **0** | 0 | **0** |
| `s2_labels_v7.2_train.jsonl.gz` | 4,572 | — | — | — | **field absent on 4,572** |

---

## 5. WHAT THE CHANNEL ACTUALLY IS — so nothing is oversold

Every one of these travels **in the run record**, not only in a docstring
(`config.json → seams.max_speed_input.meta` and `max_speed_stats`).

* **Source.** The DataFlyWheel's shipped `speed_max_input` block —
  `v_max_ms` + `v_max_bucket_kmh`/`_ms`, ladder `(20,30,50,70,80,100,120,130)` km/h,
  `units: "m/s"` on the wire. **Not** `g_tac.goals.SPEED_BAND.v_hi_ms` read directly:
  that would bypass the units declaration and the provenance stamp.
* **Provenance `ego-future`** — a **TRAIN/DEPLOY MISMATCH, not a leak**. Training
  carries `max(ego realised speed)` over [anchor+2 s, +6 s]; deployment supplies a
  limit the driver may not reach.
* **Quantization is COSMETIC as a leak fix** — bin + `v0` recovers R² 0.9702 vs
  `v0`-alone 0.8789, so **75.4 % of the future survives** — and **REAL as a semantics
  fix**: the raw `v_hi` sits *below* the ego's own current speed on **34.8 %** of clips
  (incoherent for a ceiling); quantized, **8.4 %**.
* ⛔ **The `frac_over` argument is NOT repeated here.** The 28 clips that fire are
  exactly those above the 130 km/h top step: the **cap** produces that number, not the
  ladder.
* ⚠️ **The residual defect, unfixable in this design.** Snapping UP from a *stopped*
  ego reports the **lowest** limit — 75 % of intersection clips get ≤ 30 km/h where a
  map would say 50 — so the channel can teach *"slow ego ⇒ low limit"*. It needs a
  real map to fix, and the caveat is written into `config.json` and the `--help`.
* ⛔ **Eval obligation, stamped as `required_controls: ["shuffled", "withheld"]`.**
  Any max-speed-conditioned result carries a SHUFFLE control (serve another clip's
  ceiling) and a WITHHOLD control (`valid = 0`), or *"the arm improved"* cannot be
  separated from *"the arm gained a parameter"*. MEASURED on the linear readout: a
  shuffled channel collapses **exactly** onto ego-only on every longitudinal target
  (delta +0.0002, not separated) — the control has teeth. Both controls are pinned as
  behavioural tests (`test_the_WITHHOLD_control_is_exactly_the_no_ceiling_state`,
  `test_the_SHUFFLE_control_moves_the_state`).

---

## 6. WHAT CHANGED (staged, never committed to `main`)

| file | change |
|---|---|
| `stack/tanitad/refs/refc_v3.py` | +122/−2. Import; `max_speed_input` / `max_speed_cfg` config fields (**default OFF**); a flat-arm guard placed **before** the `not cfg.hier` return; **one gated construction site**; `_hook` signature + the two injection sites (`z_tac`, `ctx`) + `max_speed_injected` cache stamp; `forward` signature + the two-way silent-drop refusal + hook call site; the `max_speed_inject` **ledger line, read off the built object**. |
| `stack/scripts/refc_v3_train.py` | +360/−2. `--max-speed-input` / `--max-speed-mode`; the config pin; `_check_max_speed_args` on **both** entry points; `V3Dataset.enable_max_speed` (+ census, units refusal, ladder & bucket cross-checks); always-emitted `v_max_ms`/`v_max_valid` item keys; `compute_losses_v3` read + reverse refusal + forward call; the `_seam_stamp` block (requested/cfg/mode/meta/built); the **bidirectional** `assert_seams_are_built` check; the `config.json` census. |
| `stack/tanitad/data/v7_labels.py` | +40/−1. `speed_max_input` carried on the private `_oracle` channel and read only through the new `oracle_max_speed(label, manifest)` — **the same manifest permission `nav_command` needs**, because the record itself declares `oracle: true` / `provenance: "ego-future"`. |
| `stack/tests/test_max_speed_wiring.py` | **new, 25 tests.** The file `WIRING_DIFF.md` said could only be written against the patched file — including `test_v3_parity_when_off…`, the reachability pair, both controls, the double-quantization pin, and the *"exactly one construction site"* AST check. |

**Test state:** `test_max_speed_wiring.py` 25 · `test_max_speed_input.py` 29 ·
`test_refc_v3_rollability.py` 11 → **65 passed**. `test_refc_v4.py` +
`test_goal_point_wiring.py` + `test_tac_goal_trainer_flag.py` +
`test_goal_point_trainer_flags.py` → all pass.

⚠️ **ONE ORDER-DEPENDENT SUITE, AND IT IS NOT MINE.** `stack/tests/test_refcv3_arm.py`
fails **12 / 20** when run ALONE under the random-order plugin (`KeyError:
'_provenance'` in a manifest fixture) and passes **25 / 25** under `-p no:randomly`
alongside its siblings. ⛔ The important measurement is the CONTROL: in the failing
order it fails **identically (12 F / 20 P) with the patch applied and with the three
files restored to HEAD** — measured both ways in this session rather than assumed. So
it is a pre-existing order dependence in that suite's fixtures, not a regression from
this edge.

Full suite with the patch: **7,184 passed / 23 failed / 7 errors** (11 min 40 s).
⚠️ TWO of those failures are ones this edge WIDENS without causing:
`test_rl_refc_adapter_robust.py::test_forward_kwargs_plumbs_every_channel_the_forward_accepts`
and `test_rl_forward_keys_cover_signature.py`. `tanitad/rl/refc_adapter.py:70`'s
`FORWARD_KEYS` has drifted behind `RefCV3Model.forward`: **at HEAD it was already
missing three channels** (`nav_args`, `gp_point`, `gp_valid`); with this edge it is
missing **five** (`+ v_max_ms`, `v_max_valid`). ⛔ NOT fixed here on purpose —
`gp_point` is documented in the forward's own docstring as *"a DIAGNOSTIC PORT, never
a training input"*, so plumbing it into an RL rollout is a decision that needs its own
reading, not a drive-by. Escalated as its own task.

---

## 7. CAN THE NEXT ARM USE MAX SPEED — AND ON WHAT?

**Yes.** Add `--max-speed-input` (and optionally `--max-speed-mode raw` for the
comparison arm) to a `--arm hier` launch that already passes `--v7-labels` pointed at a
**v8** blob:

```
--arm hier --v7-labels .../s2_labels_v8_train.jsonl.gz --max-speed-input
```

It would be training on the **v8 corpus's shipped `speed_max_input.v_max_ms`** —
present on **4,572/4,572** train and **147/147** eval records, units declared `m/s`,
snapped UP to the pinned posted-limit ladder inside the model, fed additively into the
tactical (`z_tac`) and strategic (`ctx`) nodes beside `v0` and the nav token. ⚠️ Its
provenance is `ego-future`, so it is a **stand-in for a map service with a train/deploy
mismatch**, and any result it produces is evidence only alongside its shuffle and
withhold controls.
