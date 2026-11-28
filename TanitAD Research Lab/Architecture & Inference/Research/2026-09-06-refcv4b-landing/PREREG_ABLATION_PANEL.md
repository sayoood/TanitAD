# PRE-REGISTRATION — the refcv4b landing ablation panel

**Written 2026-09-05T23:5xZ, BEFORE `ckpt_40284_FINAL.pt` exists** (the run is at step ~32,650 of
40,284). Both outcomes are committed here for every arm. ⛔ Nothing below may be edited after the
numbers land; a changed criterion is a moved goalpost and the correction goes in
`RETRACTION_LOG.md` instead.

**Package:** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-06-refcv4b-landing/`
**Surface:** 141 held-out B1 clips, `--window-stride 5` ⇒ the **same 4,823-window grid** refcv3
@40,284 was scored on. Estimator: paired episode-cluster bootstrap, `n_boot 2000`, `seed 0`.
Every arm is scored against the **same** `os` baseline on the **same** windows.

---

## 0. ⛔ THE PANEL'S OWN VALIDITY GATE — run this arm or nothing else is admissible

| arm | flag | what it does |
|---|---|---|
| **`frames_blind`** | `--ablate frames_blind` | every observed frame replaced by the window's own scalar mean ⇒ the encoder sees a constant image and the arm is **an echo BY CONSTRUCTION** |

**Committed in advance:**
* **PASS (panel admissible):** `frames_blind` is separated WORSE than `os`, **and** the model-free
  controls `ha` / `ha0` / `ha0_ext` come back **bit-identical** to the full run (they read no
  frames, so anything else means the harness moved something it should not have).
* ⛔ **VOID (not "negative"):** if `frames_blind` passes the echo gate, or reads a high nav
  compliance, **the whole panel is void** (PREREG_REFC_V4 §7 OUTCOME IV). A panel whose
  deliberate regression does not regress has not shown that its instruments can fail.

⚠️ **This is not ceremony.** A sibling burned a GPU hour tonight on a jerk-seam arm whose `W_JERK`
was 0.0 — a lever multiplied by a zero coefficient is not a null about the lever. The
constant-image arm is the same check pointed at the instrument instead of the knob.

---

## 1. `sel_refined` — the free lever, and the one I expect most from

| | |
|---|---|
| flag | `--ablate sel_refined` (`decoder.sel.refined = True`, `sel.score_emitted = True`) |
| **parameters changed** | **ZERO** |
| motivating measurement | refcv3's ranking reads the **t = 0** confidence and is **unchanged on 201/201 windows** for every `steps` value (`.../2026-09-05-refc-vs-diffusiondrive-audit/raw/mechanism_check.json`, `GOALS_AND_CLAIMS.md` `D-REFC-DDAUDIT-3`). Selection is decided **before** refinement. |
| mechanism | `refc.py:1596` keeps `refined is conf` and `:1630` gates `score_emitted` on `steps > 0`. refcv4b's decoder is `diffusion/2`, so the switch is **live** here (it REFUSES at `steps == 0` rather than parsing and doing nothing). |
| scale of what is being discarded | the decode displaces the **selected** anchor by a mean **0.9171 m** over the 8 slots, monotonically 0.092 m @0.5 s → **2.384 m @6 s**, max 7.556 m — MEASURED on the preflight dump's `sel_bank_nav_true` vs `plan_full_nav_true`. ⚠️ **n = 5 windows, 1 clip — NOT decision-grade**, quoted only to show the quantity is not negligible. |

**Committed in advance:**
* **SUPPORTED** if `sel_refined − os` on `ade_m` is **negative and separated**. ⇒ refcv5 ships
  refined-ranking, and it is a **zero-parameter** improvement — the highest-value kind.
* **REFUTED** if the interval covers zero or is separated the wrong way. ⇒ the audit's ranking
  defect is real but does not cost ADE on this checkpoint, and WP-4's case must rest on
  something else.
* ⚠️ Either way, report **turn recall per class beside ADE** — a ranking change that improves ADE
  by picking straighter paths is a regression wearing a win. refav1's best-ADE arm executes
  **zero** turns.

---

## 2. `h19_off` — the discriminating experiment for the curve finding

| | |
|---|---|
| flag | `--ablate h19_off` (sets `lat_to_anchor` / `lon_to_anchor` to `None`) |
| motivating measurement | the core's **longitudinal kin3 head is WORSE than its own class prior**: eval CE **1.0090** nats against a prior-predictor floor of **0.8964** (prior read from `ckpt_30000.pt`'s `core.lon_log_prior`), only 0.0896 below uniform (ln 3). The lateral head beats its floor (0.4558 vs 0.5591). |
| mechanism | `tac_vocab_version = "v7.0"` ⇒ `refc_v3.py:929` sets `man5 = None` ⇒ H19's anchor prior falls back to the **core's own kin3 heads** (`refc.py:2294`, `:2313` prior-centred, `:1578-1580` added to the anchor confidences). **A signal measured worse than a constant is being added to the selection surface.** |

**Committed in advance:**
* **SUPPORTED** if `h19_off − os` is **negative and separated** (removing the prior HELPS).
  ⇒ refcv5 must not drive its anchor prior from a head that cannot beat its own marginal; the
  sharp follow-up is a one-line variant removing **only** `lon_to_anchor`, since the lateral half
  does beat its floor.
* **REFUTED** if it is positive and separated (the prior HELPS despite the CE). ⇒ the CE floor is
  the wrong readout for a *prior-centred* logit, and the finding is retracted to "the head is a
  poor classifier but a useful ranker."
* **INCONCLUSIVE** if the interval covers zero — which on a one-seed arm it may, and that is
  reported as inconclusive, not as a null.

### 2b. ⭐ THE PRIMARY READOUT FOR `h19_off` IS THE SELECTION-FLIP FRACTION, NOT ADE

A prior enters as a **term added to the anchor confidences** (`refc.py:1578-1580`) and the pick is
an **argmax**. It can therefore move every confidence and change **no decision**. ADE alone cannot
tell those apart, so the panel reports **`n_windows where sel_idx differs from the baseline`**
first, and ADE second.

**MEASURED on the CPU preflight — ⚠️ n = 3 windows, 1 clip, stride 60, `ckpt_30000.pt`. Directional
only; it fixes the READOUT, it does not decide the question:**

| arm | `sel_idx` | selection changed | `os` ADE (m) |
|---|---|---|---|
| baseline | `[49, 67, 75]` | — | 0.2361 |
| `sel_refined` | `[58, 67, 76]` | **2/3** | 0.2932 |
| `ego_zero` | `[67, 40, 46]` | **3/3** | 0.7693 |
| **`h19_off`** | `[49, 67, 75]` | **0/3** | **0.2361 — bit-identical** |
| `frames_blind` | `[69, 76, 84]` | **3/3** | 0.7220 |

CONTROL: baseline against itself reads **0** changed. CONTROL: the model-free arms `ha` 0.1767 /
`ha0` 0.8262 / `ha0_ext` 0.1767 read **bit-identically across all four ablations** — they consume no
frames and no ego, and anything else would mean the harness moved something it should not have.

⛔ **AND THE ZERO-COEFFICIENT CHECK WAS RUN BEFORE THIS WAS READ AS A NULL**, because *"a lever
multiplied by a zero coefficient is not a null about the lever."* From `ckpt_30000.pt`:
`core.decoder.lat_to_anchor.weight` **absmean 2.802e-01**, `core.decoder.lon_to_anchor.weight`
**absmean 7.047e-02**, neither all-zero (controls: `conf_head.weight` 1.730e-02,
`goal_gate` 6.424e-02, both non-zero). ⇒ **`h19_off` is a LIVE lever**, and a 0/3 flip rate is a
real observation about its effect rather than an artifact of an inert knob.

⭐ **An independent corroboration of §2 falls out of those same weights:** the longitudinal
prior's matrix is **4.0× smaller in absolute mean than the lateral one** (0.0705 vs 0.2802).
The network itself down-weighted the head whose cross-entropy cannot beat its own class marginal.

**Committed in advance, for the flip fraction at n ≈ 4,823:**
* flip fraction **≈ 0** ⇒ H19 is **advertised but effectively inert on selection** — live weights,
  no decisions. That is a *structural* finding, and the refcv5 consequence is the opposite of the
  one §2 anticipated: do not fix the prior, **stop spending the edge on it**.
* flip fraction **materially > 0** with `h19_off − os` negative and separated ⇒ §2's SUPPORTED
  branch stands and the sharp follow-up is the `lon_to_anchor`-only variant.
* flip fraction > 0 with ADE unchanged ⇒ the prior reshuffles picks without improving them; report
  turn recall per class before saying anything else.

⛔ **The scope caveat travels with it:** the prior in the checkpoint is a **TRAIN EMA**
(`update_tactical_prior` is gated on `model.training`), so the floor is a train statistic quoted
against an eval CE. `raw/kin3_marginal.py` closes this from the landing dump at n ≈ 4,823 and
**runs before the verdict is written**, not after.

---

## 3. `e9_off` — is the strategic goal gate doing anything at all?

| | |
|---|---|
| flag | `--ablate e9_off` (`model.goal_gate = 0.0`) |
| motivating measurement | the gate rises **monotonically** 0.0008 @500 → **0.0658** @31,500, gradient still non-zero; the parameter reads **0.06424** in `ckpt_30000.pt`. ⇒ the strategic goal path contributes at ≈6.6 % amplitude after 31.5 k steps. |
| the question it separates | *"the gate found its optimum"* vs *"the gate is LR-bottlenecked and the goal path is under-used"* — indistinguishable from the gate value alone, which is why the trainer emits `goal_score_absmean` and `goal_gate_grad` beside it (CAVEAT-B, PI 2026-09-02). |

**Committed in advance:**
* If `e9_off − os` is **not separated**, the goal-selection graft is **inert at this amplitude** ⇒
  refcv5's lever is the gate's **parameterisation** (init/LR/normalisation), not more goal content.
* If it is separated **worse**, the graft is load-bearing at 6.6 % ⇒ refcv5 should let it open
  further and the experiment is a gate-LR sweep.
* ⛔ Neither outcome licenses "the hierarchy works/does not work" — this is one edge.

---

## 4. `ego_zero` — the PI-binding deployment reading

| | |
|---|---|
| flag | `--ablate ego_zero` (`keep = 0` **and** `v0 = None` at the core) |
| why it binds | *"for ground truth data of scenario classification you can use both ego and other label, **for inference only vision**"* (PI 2026-08-03), tempered by the 2026-09-02 ruling that **measured v0 at t0 is a legal initial state**. `ego_zero` is the conservative end of that range and refcv4b **trains its own withheld regime** (`--ego-dropout 0.5`), so this is a regime the model has seen — not an invented null. |
| prior measurement | at step **9,500** the withheld ceiling read **7.5941 m** against **1.1112 m** kept, and the selected-anchor offset **6.18 m [4.80, 7.81]** withheld vs **0.64 m [0.61, 0.67]** kept (`D-REFCV4B-EGODROP2`). |

**Committed in advance:** the number to report is the **step-40,284 withheld-vs-kept gap**, and
whether it CLOSED relative to 9,500. A gap that has not closed by 40,284 means `--ego-dropout 0.5`
did not buy a vision-only arm, and refcv5 must obtain v0 differently (the `--withheld-bank pred`
route a sibling is building) rather than by dropping it harder.
⚠️ The model-free controls **must read bit-identically** across this ablation — they integrate the
MEASURED v0 by design and are controls, not arms.

---

## 5. Ranked execution order — a kill at any point still leaves value

1. the headline four-family read + the reel (**Goal 1** — the PI asked for these first);
2. `frames_blind` (without it nothing else is admissible);
3. `sel_refined` (zero-parameter, largest expected effect);
4. `h19_off` (the curve finding's discriminating experiment);
5. `ego_zero` (PI-binding deployment number);
6. `e9_off`, then `e7_off` / `gstr_zero` / `gstr_shuffle` if GPU remains.

⚠️ **Compute etiquette:** the PI has queued **refcv5** training on this A40. The headline read and
the reel are ~1 h together. Each ablation arm is roughly one more eval pass. The Master Mind is
told before the panel is extended past the ranked items above.

---

## 6. ⛔ The interval caveat that applies to EVERY row here

A separated CI from a **one-seed** arm is **necessary, not sufficient** (`H-ESTIM-SEED-1`): the
episode-cluster bootstrap resamples EPISODES with the models held fixed, so it answers *"would
another draw of episodes say this?"* and never *"would another training run say this?"*. On the
v7-tiny rig a **zero-lever replicate** produced "separated" differences on 3 of 18 family metrics.
⇒ Every ablation here is an **eval-time** intervention on **one** checkpoint, so the training-run
variance does not enter — but the correct claim form is *"this switch changes this metric on this
checkpoint"*, never *"this lever is worth X in refcv5"*. The refcv5 claim needs the refcv5 arm.
