# SPEC — the refcv3 eval instrument (BACKLOG R20), pre-registered

**Owner:** Benchmarks & Evals FlyWheel · **date:** 2026-09-03 · **compute:** 0 GPU (dev-box CPU) ·
**written BEFORE any code was executed and before any number below existed.**

---

## 0. The question this package answers

refcv3's epoch ends tonight (≈ 22:15–22:43Z) and **there is no instrument that can read it at T1**
(`D-V7-READINESS-2026-09-02` §D). `T1_CHECKLIST.md` is a runbook, not an instrument, and it says so
at its first gate. This package builds the instrument.

⛔ **The blocking deliverable is a DEFINITION, not compute** (`RESULT.md` §5.4 work item W6): what
refcv3's arm *is*, derived with file:line, reviewed by the Master Mind **before any number is
quoted**. The killed agent never wrote it.

---

## 1. The decision taken FIRST, and the evidence for it

**Finish the 1,746-line rescued draft, or restart from the current `refav1_arm.py`?**

**DECISION: RESTART, harvesting the draft's model-rebuild, corpus, grid and lead-join work.**

**Justification (two sentences, as the brief requires).** The draft is missing every one of the
instruments that now define this read — MEASURED by grep over
`…/incoming/2026-09-03-refcv3-arm-UNVERIFIED/refcv3_arm.py`: **0** occurrences each of `ha0`,
`trivial_profile`, `action_units`, `sel_score_v3`, `a_star`, `oracle_sel` and `STEER_WHEELBASE` —
and its spine is the exact claim `D-HF-COMPARABILITY` was written to forbid: it names the deployed
arm `cl` throughout and its module docstring calls it a *"Mirror of `t1_eval.roll_closed` … the loop
collapses to the single tick"*. Finishing it therefore means deleting its spine and adding five
instruments, whereas restarting from `refav1_arm.py` inherits `ha0`, the trivial profile, the
`--action-units` bridge, the dump contract, the lead-block join and the paired bootstrap **by
import**, which is also the only way the programme keeps one convention instead of two.

**What is harvested from the draft (credited at each definition in the new file):** the
config-rebuild path through the trainer's own `build_parser` + `_pin_trainer_cfg`
(`refc_v3_train.py` writes no model config), the `cross_check_config` refusal, the **seam-clamp
fail-loud neutralisation** (`apply_seam_clamp` raises after N consecutive saturated calls and the
counter lives on the model, so a long eval would die mid-run), `grid_slots`' index-select rule, the
`V3Dataset` subclass that skips the unused future-frame decode, the corpus/label/nav join, and the
**common-grid** lead-block view.

---

## 2. Pre-registered contract for the instrument

### 2.1 Arms (and the two that must NOT appear)

| arm | tier | definition |
|---|---|---|
| `os` | **T1, RULING OPEN** | one forward at t0: observed frames + v7.2 nav token + measured `v0`; the path is the model's OWN `out["traj"]` |
| `os_navshuf` | T1, ruling open | the same with `nav_cmd` permuted across the eval windows — breaks the PAIRING, preserves the marginal |
| `os_navzero` | T1, ruling open | ⭐ the same with nav WITHHELD (`nav_cmd=None`) — removes the SIGNAL. **Added by amendment, §4.** |
| `ha` | T1 | hold the last **observed** `(a, steer)` for the horizon |
| `ha0` | T1 | ⭐ constant velocity (`a = 0, κ = 0`) at the measured `v0` |
| `oracle_sel` | T0 (opt-in) | the `a_star`/GT-nearest anchor's refinement — the ceiling |
| ⛔ `cl` | — | **must not exist.** A shared name puts two procedures in one column. |
| ⛔ `ol` | — | **ABSENT with its reason.** refcv3 consumes no actions. |

### 2.2 Committed in advance — what each control MUST read on a random-init model

*(These are the pre-registered outcomes. A control that does not read its known value means the
instrument is broken, not that the model is interesting.)*

1. `ha0` is **straight and constant-speed on 100 %** of windows, and its per-step chord is `v0·dt`.
2. The trivial-profile instrument reads `trivial_frac = 1.0` for `ha0`.
3. `os` and `os_navshuf` are **bit-identical exactly on the windows whose nav token did not change**.
4. `os − ha0` is **positive and separated** — a random-init model must LOSE to the trivial floor.
5. `os − os_navshuf` is **not separated** — untrained weights carry no nav dependence.
6. `anchor_acc` sits at or below chance (`1/n_anchors`).
7. A perturbation of every pose/action **after** the last window's origin leaves `os`, `ha`, `ha0`
   **bit-identical** while `g` (the GT waypoints) **changes**.
8. Two arms that are deliberately made equal are reported as **equal and degenerate**, i.e. VOID.

---

## 4. ⭐ AMENDMENT (2026-09-03, after the first run) — the nav-ZERO arm

**Provenance, stated plainly:** §§1–3 above were written before any code ran. This section was added
**after** the first fixture run, on a coordinator requirement, and it is marked as an amendment
rather than folded into the pre-registration — a spec silently grown to fit a result is not a spec.
Its own predictions (4.3) were committed **before** the nav-zero code was executed.

**The requirement.** `BACKLOG R39` binds every nav-conditioned claim to carry a nav-**ZERO** arm
beside the nav-**SHUFFLE** one. They are different interventions: a shuffle preserves the nav
marginal and breaks only the pairing; a zero removes the signal. The measured basis is
`D-REFAV1-TAC-DECODER-PANEL` — refav1's tactical head ranked turns at **AUC 0.873** under true nav,
**collapsed to 0.520 (chance)** under nav_zero, and a **nav-only predictor beat the model outright
(0.684 > 0.650)**. refcv3 is nav-conditioned at all three layers, and its nav token is an **oracle**
(provenance `ego-future`) that will not exist at deployment — so the zero arm is the
deployment-relevant one.

### 4.1 The null, chosen from source rather than invented

Checked first, as required: **`nav_known` is NOT usable here.** `RefCConfig.nav_known_channel`
defaults to `False` (`refc.py:594`), nothing in the v3 path turns it on (0 references in
`refc_v3.py`, 0 in `_pin_trainer_cfg`), and `refc.py:2042–2045` **raises** if `nav_known` is supplied
while the gate is off. ⇒ the null is **`nav_cmd=None`**, the model's own documented no-nav path and
the published REF-C eval convention (`refc.py:343–345`).

### 4.2 What it removes, per layer — and the honest caveat

* **tactical + strategic (E13):** REMOVED ENTIRELY — `refc_v3.py:437–441` guards the injection on
  `nav_cmd is not None`; `out["nav_injected"]` reads `False`.
* **core:** ⚠️ NOT removed — **collapsed onto the majority token** (`one_hot(0)` = `follow`,
  `refc.py:2021–2024`). `NAV_COMMANDS` has no `unknown` and `nav_known_channel` is off, so "no nav"
  and "a genuine follow" are byte-identical at the core's input.
* ⇒ `os_navzero` is a **LOWER BOUND** on nav dependence, and the record must say so.

### 4.3 Committed in advance (before the nav-zero code ran)

1. On the **hier** build the E13 edge is **live under the fed nav and dead under the null** —
   `nav_injected_true == 1` and `nav_injected_zero == 0` on every window.
2. `os_navzero` differs from `os` on turn windows **and on `follow` windows**, because E13 is removed
   for the whole call — something the shuffle provably cannot do.
3. On a **flat** (`hier=False`) build, where no E13 path exists, `nav_cmd=None` is **bit-identical**
   to a fed `follow` token and differs on every other window.
4. Both paired blocks exist: `os_navzero − ha0` (the deployment margin) and `os − os_navzero` (what
   the oracle nav is worth).
5. On a random-init model both nav controls read **not separated**, and `os_navzero − ha0` is
   **positive and separated** exactly as `os − ha0` is.

⚠️ **Prediction 3 was stated as "exactly 0.0" and did not hold in that form on the first run** — it
read `4.77e-07`. The RESULT records what the investigation found rather than quietly relaxing the
threshold; the prediction was correct and the *comparison* was confounded.

### 2.3 Instruments that must print BEFORE any family row

* the **trivial profile** (`refav1_arm.trivial_profile`, imported): per arm `straight_frac`,
  `const_speed_frac`, `CONSTANT-VELOCITY`, `identical_to`;
* ⭐ **NEW, pre-registered here:** a **selection profile** — `n_distinct_selected`, the modal anchor
  and its share, the selection entropy, and agreement with the oracle. **Registered in advance
  because an anchored one-shot model can be fully degenerate while its PATHS look non-trivial**, so
  the trivial profile alone is not sufficient for this model class.

### 2.4 Units, estimator, tiers

* `--action-units` default **`steer`** (`κ = tan(steer)/L_enc` before integration). ⚠️ Deliberately
  different from `refav1_arm.py`'s `kappa` default, which exists only for byte-identical re-analysis
  of pre-2026-09-03 banked dumps; refcv3 has none, so shipping the known defect as the default would
  buy nothing. It applies to **`ha` only** — `os` has no control channel and `ha0` is zero.
* Point estimates **full-set pooled**; intervals **episode-cluster bootstrap** (`ci.py:225`), paired
  across arms (`ci.py:275`). `overlapping_holdout_se` never used.
* ⛔ ADE recomputed as an **L2 norm** from the dumped path; never converted from refcv3's
  training-time mean-L1-per-coordinate `traj`.
* The `os` tier is stamped `T1` **with `status: UNRULED`** on every block, and the headline statistic
  is the **margin over `ha0`**, which survives the ruling in either direction.

### 2.5 What would make this package's own claims inadmissible

1. Any number quoted about **refcv3 itself** — no real checkpoint may be loaded from this box.
2. A control that does not read its known value (§2.2) being reported as a model finding.
3. An arm named `cl`, an `ol` column, or a family dropped without a stated reason and `n`.

---

## 3. Deliverables (pre-registered)

| # | artifact | path |
|---|---|---|
| 1 | the T1 definition + the contract, §1–§5 | `taniteval/tools/REFCV3_ARM.md` |
| 2 | the instrument | `taniteval/tools/refcv3_arm.py` |
| 3 | the tests | `stack/tests/test_refcv3_arm.py` |
| 4 | this SPEC + the RESULT + `raw/` | this directory |

⛔ **Not in scope, deliberately:** the pod `tanitad-refcv3` (TRAINING) and Thor are not contacted;
`t1_eval.py`, `refav1_arm.py`, `refc_v3.py`, `refc_v3_train.py` and the rescued draft are
**read-only**; nothing is committed or pushed.
