# PI QUEUE ITEM 8 — CLOSED: the RL pilot can load its own cold start

*Arch+Inference FlyWheel · 2026-09-10 · branch `agent/arch-inf-20260803`*

**Status: option (c) IMPLEMENTED, tested, mutation-proven, and the pilot has been
MEASURED loading a 487-key cold start and taking real GRPO steps.**
Nothing here needs a PI decision to proceed; two things below are ESCALATED.

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
| `stack/tests/test_refc_cold_start_allowance.py` | **NEW.** 25 tests, all literals, with the deliberate-regression arm. |
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
| baseline | none | **GREEN**, 0 failed / 25 |
| **M1** | delete the `if run_v0: raise` gate — "option (c) without its refusal" | **3 RED, exactly the 3 expected**, including `test_REGRESSION_missing_controls_with_v0_conditioned_is_refused` |
| **M2** | replace the whole allowance with `strict=False` — the forbidden option (a) | **14 RED, exactly the 14 expected** |

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

## 5. ⭐ RULE ZERO — the pilot was actually run

`code/smoke_cold_start_rl.py` → `raw/SMOKE.json`. Dev-box RTX 4060 (8 GB),
torch 2.11.0+cu128.

* **built a 487-key cold start against 488 built keys**, missing exactly
  `decoder.anchor_controls`, with **no `cfg` key and no sidecar** — the ABSENT
  case the real July checkpoint presents;
* drove **the pilot's own `load_model`** (not a re-implementation), through
  `assert_config_contract` and the new allowance;
* **LOADED: 488 state-dict keys on the model, 104,191,577 params @ step 30000**,
  `anchor_controls` present and all-zero, stamp
  `defaulted-zeros-v0-unconditioned` / `UNVERIFIED_BY_CKPT_CONFIG`;
* **3 real GRPO steps** through `make_refcv3_sample_fn` + `rl_objective` +
  AdamW in **1.4 s**; losses finite (`-0.0126`, `0.0130`, `0.0100`), trainable
  weights moved (max |Δ| 3.06e-05);
* the stamp **read back off disk** from `config.json`.

⛔ **What this does NOT claim.** The weights are the model's own initial weights
with one key deleted — they reproduce the DEFECT exactly and carry nothing about
the July run's quality. The frames and reward context are synthetic. **This
proves the PATH; it makes no R1/R2/R3 claim.** Those need the real checkpoint
and the real corpus, and the real checkpoint is on **no reachable box**: MEASURED
2026-09-10, `tanitad-pod3` / `pod4` / `pod5` / `a40` all refuse SSH
(*Connection refused*); Thor is alive.

---

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

## 7. What is left, and for whom

1. **Nothing blocks the RL stage on this defect any more.** `p_rc21_chain.sh`
   can run as written once a box and the real cold start exist.
2. ⛔ **BLOCKED ON COMPUTE, not on code:** no RunPod pod is alive and the
   `refc-diffusion-base-v21-30k` weights are not on this box or in the repo. A
   real R1/R2/R3 pilot needs a box + that checkpoint. **That is a PI/provisioning
   item, and it is the only thing standing between here and a real RL number.**
3. **ESCALATED to item 7's owner:** §6, with the measurement and the source
   citation.
