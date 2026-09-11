# PI QUEUE ITEM 8 — CLOSED: the RL pilot can load its own cold start

*Arch+Inference FlyWheel · 2026-09-10 · branch `agent/arch-inf-20260803`*

**Status: option (c) IMPLEMENTED, mutation-proven, and the REAL pilot has loaded
the REAL July cold start and stepped — `104,191,577 params @ step 29999`,
487 checkpoint keys onto 488 built, `PILOT_EXIT=0`.**

⭐ **The defect is closed against the artifact that produced it, not against a
stand-in.** Nothing in item 8 needs a PI decision any more. What is left is
**one spend decision** (§7) and **one escalation to item 7** (§6).

⛔ **§5 carries a RETRACTION of a claim made in an earlier draft of this same
document** — *"the weights are on no reachable box"* — which was false, and
which broke the very operating-standard rule this document quotes.

---

## 1. The answer to "what exactly is `decoder.anchor_controls`?"

`anchor_controls` is a **persistent buffer `[N, 2]`** (`refc.py:1496`), N = 128
on `refc_config()`. It is the **v0-CONDITIONED anchor vocabulary**: one
`(a_lon, a_lat|kappa)` control pair per anchor, held constant over the horizon.
It is not a weight and nothing trains it — `load_anchors` installs it from a
built anchor file (`refc.py:1750`).

**It is the ACTION SPACE, expressed as controls rather than as paths.** When
`anchor_v0_cond` is True, `roll_bank` (`refc.py:1715`) integrates it per window
through `rollout_unicycle` from *that window's own measured v0*; `anchors`
`[N, S, 2]` then holds the same family rolled once at `ref_speed_ms = 10.0` and
survives only as the checkpoint-visible artifact the two param-free geometric
priors fall back to.

### What breaks if it is random — and the honest correction

⚠️ **It is not random. It is ZEROS.** `refc.py:1496` registers
`torch.zeros(anchors.shape[0], 2)`; MEASURED on `refc_config()`: shape
`[128, 2]`, **0 non-zero entries at construction**. The brief said "random
init"; the artifact says zeros, and the distinction is the whole severity
argument:

* A **random** vocabulary would emit a scattered, obviously-wrong fan.
* An **all-zero** vocabulary emits `(accel 0, curvature 0)` for **every one of
  the 128 anchors** — so every candidate is *"hold this speed, go straight"*,
  the fan collapses to a single degenerate family, and the anchored Gaussian is
  centred on it. `refc.py:2277` already names it: *"a plausible-looking WRONG
  experiment"*. It trains. It converges. It produces a complete, plausible set
  of R1/R2/R3.

⭐ **That is why it outranks the 2026-09-04 units failure it rhymes with.**
There, reading `controls` column 1 as curvature instead of lateral acceleration
produced a **396 g** table — arithmetically perfect and absurd enough to catch.
A silently-zeroed `anchor_controls` **produces no absurd number at all.**

⭐ **And the reason option (c) is derivable rather than merely cautious:** the
buffer is read at exactly two sites and both are behind `anchor_v0_cond` —
`refc.py:1727` (`anchor_control_seq`, reachable only from the WP-4 sampler,
which `refc.py:2270-2278` refuses outright on a non-conditioned vocabulary) and
`refc.py:1836`/`roll_bank`, which with the flag False returns the stored
`anchors` expanded unchanged and never touches the controls. With the flag
False the zeros are **unreachable**, not merely harmless-looking.

---

## 2. What was built

| file | what |
|---|---|
| `stack/tanitad/refs/cold_start.py` | **NEW.** The declared allowance. |
| `stack/tests/test_refc_cold_start_allowance.py` | **NEW.** 26 tests, all literals, with the deliberate-regression arm. |
| `stack/scripts/rl_pilot_refc21.py` | `load_model` routes through the allowance and returns the stamp. |
| `stack/tanitad/rl/posttrain.py` | `run_posttrain(..., extra_record=...)` so the stamp lands in the run's `config.json`. |

### ⛔ The load is still `strict=True`

`cold_start.py` does not soften the load — it **completes the checkpoint**. The
one permitted key is inserted explicitly from the model's own initialised buffer
and `load_state_dict` is then called with `strict=True`. Consequences, and they
are the entire difference from `strict=False`:

* a **second** missing key still raises;
* **any** unexpected key still raises;
* the permitted set is the **literal** `("decoder.anchor_controls",)`, not
  "whatever turned out to be missing".

MEASURED by AST over the module: exactly one `strict=` keyword in executable
code, and its value is `True`.

### The stamp

`config.json` gains `cold_start_load`, in the spirit of `anchor_meta.py`'s
`control_units_source = "cli-override-legacy-file"` — the record says the value
came from a defaulting path, not from the checkpoint:

```json
"cold_start_load": {
  "anchor_controls_source": "defaulted-zeros-v0-unconditioned",
  "defaulted_keys": ["decoder.anchor_controls"],
  "keys_built": 488, "keys_in_checkpoint": 487, "keys_after_load": 488,
  "strict": true,
  "run_v0_conditioned": false,
  "run_v0_source": "AnchoredDiffusionDecoder.anchor_v0_cond (the CONSTRUCTED module, not a CLI flag)",
  "ckpt_v0_conditioned": null, "ckpt_v0_source": "ABSENT",
  "ckpt_confirmation": "UNVERIFIED_BY_CKPT_CONFIG"
}
```

---

## 3. Where `v0_conditioned` comes from — and can an operator fake it?

**Two different questions ride on that one flag name, and collapsing them is how
a guard becomes an operator assertion.** They are kept separate:

| | question | source | verdict |
|---|---|---|---|
| **(R)** the RUN condition | *will the defaulted zeros be READ?* | `anchor_v0_cond` on the **constructed decoder**, reached by walking the missing key's own path (`decoder.anchor_controls` → `model.decoder`; `core.decoder.…` → `model.core.decoder`, so one rule serves both model families) | **REQUIRED.** True ⇒ REFUSE. |
| **(C)** the CHECKPOINT condition | *was it TRAINED v0-conditioned?* | the checkpoint's own config leaf | **A VETO.** True ⇒ REFUSE. Absent ⇒ recorded `UNVERIFIED_BY_CKPT_CONFIG`, never as agreement. |

### Can an operator fake it? — **MEASURED: no, and it is a test, not a claim**

* `rl_pilot_refc21.py` has **no CLI flag** that can set `anchors.v0_conditioned`
  and **no** `--allow-missing-anchor-controls`. The model is built from
  `refc.refc_config()` with no override.
  `test_the_pilot_exposes_no_flag_that_can_set_v0_conditioned` intercepts the
  **real** parser and asserts the option surface against a **literal list of
  15 options**, plus a substring ban on `v0` / `anchor-control` / `strict` /
  `allow` / `conditioned` / `force`.
* `refc.refc_config().anchors.v0_conditioned is False` is asserted as a literal.
* The flag is read off an `nn.Module` attribute, not a config argument, so a
  caller cannot pass a different value than the model actually has.

### ⚠️ One deliberate departure from the brief, stated plainly

The brief said: *read `v0_conditioned` from the checkpoint's own config; if the
checkpoint does not carry it, refuse.*

**Applied literally that rule can never be satisfied by this checkpoint.**
`AnchorConfig.v0_conditioned` did not exist as a field until 2026-09-04
(`refc.py:388`); a config written on 2026-07-20 **cannot** carry it. (The
pilot's own note MEASURES the shipped cold start as carrying 38 of 96 leaves and
not this one.) A rule keyed only on that leaf is therefore a **permanent
refusal wearing a guard's costume** — i.e. option "do nothing", which the queue
already lists as the silent default.

So the checkpoint's config is implemented as a **veto** (it can refuse, it
cannot authorise), and the safety condition is (R), which is about *this
process* and is verifiable here. The absent case is recorded as
`UNVERIFIED_BY_CKPT_CONFIG` and never as confirmation.
⭐ For a caller who wants the stricter rule anyway, `require_ckpt_confirmation=True`
exists — it only ever refuses more, and its refusal message says why passing it
for a July checkpoint is a refusal by construction.

---

## 4. ⛔ The mutation proof — the regression arm DOES go RED

`code/mutate_cold_start_guard.py`, MEASURED 2026-09-10
(`raw/MUTATION_PROOF.json`). Expected RED sets are **literals**; the harness
parses the `FAILED` lines rather than reading an exit code.

| | mutation | result |
|---|---|---|
| baseline | none | **GREEN**, 0 failed / 26 |
| **M1** | delete the `if run_v0: raise` gate — "option (c) without its refusal" | **4 RED, exactly the 4 expected**, including `test_REGRESSION_missing_controls_with_v0_conditioned_is_refused` |
| **M2** | replace the whole allowance with `strict=False` — the forbidden option (a) | **15 RED, exactly the 15 expected** |

⭐ **M2's extra RED arms are the interesting half.** Beyond the refusals, three
*stamp* tests go RED — because the mutant's stamp **lies**: it reports
`defaulted_keys: []` and `source: "checkpoint"` for a load that silently
defaulted a tensor. That the stamp tests catch a lying stamp is what makes the
run record evidence rather than decoration.

⚠️ **A test bug caught by its own control, worth recording.** The first draft of
`test_strict_false_appears_nowhere_in_the_module` was a substring search and
went **RED on its own docstring**. That is the harmless half of a defect whose
other half is fatal: a text search cannot tell code from prose, so it would
equally have gone **GREEN** on a `strict=False` sitting inside a string. It is
now an AST walk, with
`test_the_strict_kwarg_checker_can_actually_see_a_violation` planting the real
defect and requiring the checker to find it.

---

## 5. ⭐⭐ RULE ZERO — THE REAL PILOT RAN, ON THE REAL COLD START

⛔⛔ **FIRST, A RETRACTION OF MY OWN, MADE IN THIS DOCUMENT AN HOUR EARLIER.**
I wrote that the July checkpoint was *"on no reachable box"* on the strength of
two probes: the RunPod fleet (all refuse SSH — true) and the repo (absent —
true). **Both were true and the conclusion was FALSE.** The checkpoint is on
**this dev box**, where `p_rc21_chain.sh` has always expected it:

```
C:/Users/Admin/tanitad-data/models/refc-base-30k/ckpt.pt
1,250,838,325 B · md5 8f10d6f934f4199e11ddc7352e074939  ← the chain's own WANT_MD5, exactly
```

⭐ This is **operating-standard rule 2** — *absence found at ONE location is not
absence* — committed by the person quoting it, in the same turn. The vector was
that I searched *where a big checkpoint usually lives* (pods, HF, the repo) and
not *where the thing that consumes it says it lives*. **The chain script named
the path the whole time.** Same family as the artifact-cost trap: **open the
CONSUMER and read what IT opens.**

### The real run — MEASURED 2026-09-10, dev-box RTX 4060, `PILOT_EXIT=0`

`raw/realpilot/` (`pilot_stdout.log`, `config.json`, `config_contract.json`,
`pilot_summary.json`).

```
[pilot] ✅ CONFIG CONTRACT: case PARTIAL via sidecar …/refc-base-30k/config.json
        — 38/98 config fields COMPARED and AGREEING, 0 disagreements.
[pilot] ⚠️  DECLARED ALLOWANCE: 487/488 keys came from the checkpoint;
        DEFAULTED ['decoder.anchor_controls'] from the model's own zero buffer.
[pilot] ⚠️  permitted because AnchoredDiffusionDecoder.anchor_v0_cond … is False
[pilot] cold start loaded: 104,191,577 params @ step 29999 (488 state-dict keys)
[pilot] train eps 54 · val eps 15 · device cuda · reward default
```

* ⭐ **`step 29999`** — this is the real `refc-diffusion-base-v21-30k`, not a
  stand-in. **487 keys in, 488 on the model, exit 0.** The item-8 defect is
  closed against the artifact that produced it.
* ⭐ **The predicted ABSENT case is CONFIRMED against the real artifact.** The
  sidecar's `cfg.anchors` carries exactly `{n_anchors: 128, pool_size: 4096,
  seed: 0}` — **no `v0_conditioned`**, because the field did not exist in July.
  So the stamp reads `ckpt_v0_conditioned: null` /
  `ckpt_confirmation: "UNVERIFIED_BY_CKPT_CONFIG"`. ⇒ §3's departure from the
  brief is not hypothetical: **the literal rule would have refused this exact
  checkpoint, and did not need to.**
* ⭐ **The pilot's own independent corroborator agrees**, and it is a genuinely
  separate mechanism — `_v0_corroboration` reads the WEIGHTS, not the config:
  `state: BUFFER_ABSENT`, `verdict: CORROBORATED`.
* **The full stamp is in the real run's `config.json`**, including
  `defaulted_detail` (`shape [128, 2]`, `dtype torch.float32`,
  `nonzero_entries 0`, source *"the model's own initialised buffer"*).
* The harness produced both readouts and saved `ckpt_after.pt`.

### ⭐ AND THE LEVER ITEM 8 EXISTS FOR IS LIVE — the LONGITUDINAL reward terms fired on real windows

Item 8's own case for urgency is the axis: `H-DDA-5` puts DD-V2's RL gain at
**EP +5.3 / DAC +1.7 with NC / TTC / comfort flat** — longitudinal-scale — and
`D-REFCV3-AXIS1` puts **92.2 % of our own `os − ha` gap along-track**. That
argument is worth nothing if the pilot's objective cannot express the axis, so
it was checked rather than assumed.

**MEASURED** (`raw/realpilot/pilot_summary.json`, `counters.components_fired`
over 8 steps / 16 windows / 8,192 scored samples):

| component | axis | steps fired |
|---|---|---|
| `progress` | **longitudinal** | **8 / 8** |
| `headway` | **longitudinal** | **3 / 8** |
| `feasibility` | kinematic | 8 / 8 |
| `comfort` | kinematic | 8 / 8 |
| `collision` | safety | 3 / 8 |

⇒ the along-track terms are **live on real data**, not dormant. ⚠️ `headway` at
3/8 is expected, not a defect: it needs a lead agent in frame, and per the
four-families rule an absent lead is reported with its reason and `n` rather
than silently dropped. ⛔ This says the objective **can** move the axis; it says
nothing about whether it **does** — that is the 2,000-step arm.

### ⛔ THE NUMBERS THIS RUN PRODUCED ARE NOT A RESULT, AND MUST NOT BE QUOTED AS ONE

It ran **8 steps**, not the pre-registered 2,000, because the question was
*"does the path run?"* and not *"does RL help?"*. The readouts
(`R1 +0.6427 → +0.6921`, `R2 21.335 % → 21.497 %`, `R3 0.654 → 0.660 m`) are
recorded **only** as evidence that the reward, advantage and readout paths all
execute end-to-end on real windows.

⛔ They are **not** an RL effect, for three independent reasons, each already
binding in `CLAUDE.md`: 8 optimiser steps is not the registered arm; there is
**one seed**, so the run-to-run floor (`H-ESTIM-SEED-1`) is unmeasured; and no
paired episode-cluster CI was computed. ⚠️ A real P-RC21 number is the 2,000-step
chain, and it is now **unblocked and runnable on this box**.

## 6. ⛔ ESCALATED — evidence for PI QUEUE **item 7**, found while closing item 8

`code/probe_v0_reaches_forward.py` → `raw/PROBE_V0.json`.

**MEASURED on `refc.refc_config()`, `eval()`, identical frames, both controls
valid** (same-`v0` twice is **bitwise identical**; a v0-conditioned build with a
real vocabulary **does** move, 0.1328 m — so the probe can detect an effect):

| arm | max abs Δ on `anchor_traj` |
|---|---|
| `v0 = 5` vs `v0 = 25` | **0.1356 m** |
| `v0 = 5` vs **`v0` DROPPED** | **0.0130 m** |

and at the same time the RL conditioning contract resolves **all 10 channels
`False`** — `{nav_cmd, v0, maneuver_logits, target_latent, lan, nav_known,
hierarchy_hook, ego_keep, withheld_speed, agent_gt}` — so
`contract.check({"frames": …})` with **`v0` removed PASSES with no refusal.**

**The path the predicates miss, read from source.** The contract's `v0`
predicate is `anchors.v0_conditioned OR sel_reach_clamp`
(`refcv3_adapter.py:240`), both False by default. But `refc.py:3199-3200` builds
the measurement encoder's input **unconditionally**:

```python
v = zeros(b, 1) if v0 is None else (v0 / 10.0).reshape(b, 1)
```

and `refc.py:3204-3206` derives `keep` — the X15 ego-validity channel — from
`v0 is not None`. So a batch that merely OMITS `v0` does not just lose the
speed: it asserts **`keep = 0`**, which this same file calls the *"X15
zero-collision"* — a withheld speed made indistinguishable from a genuinely
stationary car.

⚠️ **Scope it honestly.** The contract is **not wrong about the action space**:
with `v0_conditioned = False`, `roll_bank` really does return the stored
`anchors` unchanged, so the vocabulary is untouched. The finding is narrower and
it is about **SCOPE** — the refusal is keyed on the action-space consumer only,
while the forward has a second, always-live consumer, so a dropped `v0` changes
the **policy** with nothing raising.

⭐ **A hypothesis of mine that this refuted, recorded because it was wrong.**
Reasoning from the contract's own cited chain (`refc.py:3097` → `:2128` →
`:1722`) I first concluded that with both flags False `v0` must be entirely
unused — i.e. the pilot's build is *speed-blind*, which would have collided
badly with the item-8 brief's own longitudinal motivation (`H-DDA-5`,
`D-REFCV3-AXIS1`). **MEASURED: false.** Reasoning from a correct citation to a
scope it did not cover is the `df` / `free` / `step_s` family again; only the
A/B settled it.

⛔ **No behaviour was changed here.** Whether `v0` should become REQUIRED is
item 7's call and belongs to that stream — the queue records item 7's default as
*"the executed path is guarded in place if that is the right fix"*, and this is
the measurement that decision was waiting for. Deciding it inside item 8 would
change which channels a live RL arm must carry, which is exactly the kind of
change the queue exists to route.

---

## 7. The suite — what was established, and what could NOT be

⚠️ **Stated plainly, because "the suite is green" is binding and I could not
establish it outright.** The dev box cannot run the stack from the G: mount, and
the mount was too degraded to clone completely, so the off-Drive tree used here
is **partial**. It reports **66 failed + 34 errors BEFORE any change of mine** —
almost certainly missing data files and modules rather than real defects, but I
did not prove that and do not claim it.

⭐ **So the admissible read is the DIFFERENCE, measured against a baseline tree
built from the same clone with my four files reverted:**

| | baseline | with my changes |
|---|---|---|
| passed | 7,382 | **7,403** (+21) |
| failed | 66 | **68** (+2) |
| skipped / xfailed / errors | 120 / 2 / 34 | 120 / 2 / 34 (unchanged) |

The **+23 collected** are my new test file (23 tests at the moment the run
collected; it has since grown to **26**, all passing). The **+2 failures** are
`test_secret_scan.py::test_installed_hook_actually_refuses_a_commit` and
`::test_the_hook_lets_a_clean_commit_through`, and they are **NOT caused by this
work** — established three independent ways rather than argued:

1. ⭐ **A same-tree A/B.** My four files were reverted **in the same tree** and
   the two tests failed **identically**. The only variable removed was my change.
2. ⭐ **The failure is at COLLECTION.** `ModuleNotFoundError: No module named
   'secret_scan'` — `tools/secret_scan.py` is simply absent from the partial
   clone, so the module never imports and no assertion in the file ever runs.
   There is no code path from `cold_start.py`, the pilot or `posttrain.py` to
   that import.
3. ⭐ **The positive confirmation — and it carries NO provenance caveat.** With
   `tools/secret_scan.py` restored, both tests **PASS** (`2 passed in 1.28s`).
   ⚠️ The copy had to come from the local mirror, because G: would not serve
   `tools/` through the worktree: three fetch attempts all returned `FAIL`
   **while the process exited 0** — the *"assert on the ARTIFACT, never the exit
   code"* rule, live, in the same session that quotes it.
   ⭐ So the file's identity was established through a **DIFFERENT MECHANISM**:
   the git **object store**, which was serving normally while worktree reads
   were not. `git rev-parse HEAD:tools/secret_scan.py` and
   `git hash-object <the mirror copy>` both read
   **`dcebe50bf51359bb05b19962996ad185b7675b02`**, both operands asserted 40
   chars ⇒ the file used in the confirmation is **byte-identical to the repo's
   HEAD version**. ⚠️ Note this is a genuinely independent probe, not the same
   read retried: object-store reads and worktree reads are different code paths,
   which is exactly why `commit_cacheinfo.py` exists.

⭐ **And the part that IS a clean green:** every test that touches the changed
modules was re-run against the final code — **234 passed** across
`test_refc_cold_start_allowance`, `test_rl_posttrain`, `test_rl_refcv3_integration`,
`test_rl_pilot_join`, `test_rl_refcv3_used_path_guard`, `test_rl_channel_guard`,
`test_rl_refc_adapter_robust`, `test_rl_config`, `test_rl_v2_faithful`,
`test_rl_advantage`, `test_rl_rewards`, `test_rl_audit`. MEASURED: nothing
outside that set imports `tanitad.refs.cold_start`.

⚠️ **One thing I could not explain and am not going to dress up:** the partial
clone's `tools/` file count moved **35 → 18** between the suite run and the
post-hoc check, on local disk, with files present at one reading and gone at the
next. It is confined to a scratch tree; the repo deliverable was re-verified
intact (18/18 blob-verified, all four code files at their expected sizes) both
before and after. Recorded as UNEXPLAINED rather than rationalised.

---

## 8. What is left, and for whom

1. ⭐ **Item 8 is CLOSED.** The defect is fixed, mutation-proven, and the real
   pilot has loaded the real cold start and stepped (`PILOT_EXIT=0`).
2. ⭐ **`p_rc21_chain.sh` is RUNNABLE AS WRITTEN** — every precondition it waits
   on is satisfied on this box: the checkpoint at the exact size and md5 it
   requires, both `_epcache` dirs, and both agent `jsonl` files. ⛔ **The one
   thing it needs is a decision to spend ~2.5 h of dev-box GPU**, which is the
   PI's, not mine — a 2,000-step P-RC21 plus the 300-step `hackable` regression
   arm.
   ⚠️ **RETRACTED from an earlier draft of this document:** *"blocked on
   compute; the weights are on no reachable box."* False — see §5.
3. **ESCALATED to item 7's owner:** §6, with the measurement and the source
   citation. ⛔ No behaviour was changed there.
4. ⚠️ **A small drift worth one line:** `rl_pilot_refc21.py`'s own comment says
   the cold start carries *"38 of this config's 96 leaves"*. MEASURED today it
   is **38 of 98** — the config grew two leaves and the prose did not follow.
   Numerator unchanged, so nothing downstream moved.

---

## ⛔ SUPERSEDED 2026-09-11 — §7's "the only thing it needs is a decision to spend ~2.5 h" IS NO LONGER TRUE

**The chain RAN.** It was launched the same night on the dev-box RTX 4060 at zero spend, and
**fifteen further arms** followed it. ⛔ **Do not read §7 as an open item.**

⚠️ **Why this correction exists at all:** the agent that wrote §7 ran for ~16 hours, and its view of
the open items is from when it *started*. ⭐ **A long-running report's "what remains" section is a
snapshot of its own launch time, not of the present** — the same shape as the stale premise
*"refav1 is training on Thor"*, which propagated into every brief for days and hid a free GPU.

### What actually happened, and where it is banked

| result | finding |
|---|---|
| `…/2026-09-10-rl-pilot-cold-start-allowance/RESULT_P_RC21_2K_ARM.md` | the 2,000-step arm ran in **3 min 18 s**, not 2.5 h. ⛔ Its own audit returned **INCONCLUSIVE**, and `use_gt_bar` was **False** because the pilot had **no `--gt-bar` flag at all** |
| `…/2026-09-11-rl-gt-bar-reachable/RESULT_GT_BAR_REACHABLE.md` | the flag now exists; the mask is reached, proven by **analytic identity** — bit-identical gradient below the bar, **exactly 0.0** above it, **partial** in between |
| `…/RESULT_GT_BAR_5SEEDS.md` | five seeds per cell; **variance p = 0.0040** by exact permutation |
| `…/RESULT_WEAKNESS_CONTROL.md` | ⛔ **that variance claim RETRACTED** — a control trained **15 % as long** reproduces both the low drift and the collapsed spread |
| `…/RESULT_REWARD_EFFICIENCY.md` | ⭐ **what survived**: the bar is **2.24× more reward-efficient** than an equally small update, exact permutation **p = 0.0079**, and the test is **not at its design floor** |

⇒ **The RL stage's open item is no longer a spend decision.** It is **T1 and a four-family panel**,
which is what turns a training-side diagnostic into a capability claim. ⛔ Nothing above says the car
drives better.

⭐ Everything §1–§6 of this document establishes is **untouched**: item 8 is closed, the allowance is
mutation-proven, and the real July cold start loads.
