# PREREG — D-GSTR-1: why the steering signal cannot turn left, and what fixes it

`Work package: TanitAD Research Lab/Architecture & Inference/Research/2026-09-06-steering-signal/`
`Owner: steering-signal agent. Date: 2026-09-06. Written BEFORE the oracle probe and the head refit were run.`
`Antecedent: .../Research/2026-09-06-nav-vs-routehead/RESULT.md (commit a07d733d7).`

**Evidence class:** MEASURED (ours) unless a row says otherwise. **Tier:** T1
(self-action OPEN loop, 2026-09-02 ruling) for every model number; the LAN
corridor target is a LABEL and carries no tier. **Estimator:** paired
episode-cluster bootstrap over the dump's episodes. ⛔ No `overlapping_holdout_se`.

---

## 0. The defect, restated with its scope

MEASURED (`raw/GSTR_SIGN_CENSUS.json`, `raw/GSTR_LOSS_DECOMP.json`), n = 4,823
windows / 141 episodes, refcv4b `ckpt_40284_FINAL.pt`:

`g_str`'s lateral component is **negative on 4,823 / 4,823** windows, max
**−0.1022**, median **−0.5308** — a median bearing of **−29.995 deg**, i.e. 30
degrees to the RIGHT, on every window, forever.

## 1. P1 — the candidate mechanisms, and what would DISCRIMINATE each

⛔ **A structure, not a bias — so a reason must be found, not inferred.** The
brief names four candidates. Each gets a probe whose outcome is committed here.

| # | candidate | probe | what REFUTES it |
|---|---|---|---|
| M1 | sign / absolute-value in `g_str`'s construction or its target | read the construction; census the TARGET's signs | the TARGET takes both signs, and a MIRROR control flips it |
| M2 | coordinate-frame / handedness error | C2 mirror control + C3 straight-path control + GT cross-check | all three read their known values |
| M3 | a parameterisation that cannot represent negative-to-positive lateral | the same normalisation `g[:, :2] / ‖g[:, :2]‖` is applied to the TARGET path too | the target, through that path, takes both signs |
| M4 | a LABEL defect (the LAN corridor teacher is one-sided) | census the target's sign histogram | the target is two-sided |

⭐ **If M1–M4 are all refuted, the remaining hypothesis is M5: the head is a
FAILING LEARNER on a sound teacher.** M5 is only admissible with a floor:

**M5 test — the head must be scored against a NO-INFORMATION FLOOR.**
* `l_bear_MODEL` vs `l_bear` of a **constant straight-ahead (1, 0)** predictor and
  of the **best constant unit bearing** (closed-form: the direction of the mean
  target). A head that loses to a constant is not "unconverged"; it is anti-fit.
* Same for `l_dist` against the **median** target (the L1-optimal constant).
* ⛔ Both floors must be reported with their CI on the same windows.

**M5 splits in two, and the split decides the FIX:**

| | M5a OPTIMISATION failure | M5b REPRESENTATION failure |
|---|---|---|
| claim | a linear map on `ctx` CAN predict the bearing; training did not find it | `ctx` does not carry lateral route topology linearly |
| probe | **ORACLE RIDGE PROBE** `ctx -> bearing_tgt`, fit split / scored split disjoint | the same probe |
| ⛔ controls | (a) **CONSTANT-ONLY** control must read the constant floor EXACTLY; (b) a **RAW-INPUT floor**; (c) **n and d printed** | same |
| ⛔ discipline | λ and any basis fit on the FIT split ONLY, selected on a fit-internal validation split, never on the scored split | same |
| fix if true | loss weight / head capacity / gradient path — a TRAINING change | a different context or an architectural change |

⛔ **Committed in advance:** if the oracle ridge does **not** beat the constant
control on the scored split, M5b is the verdict and the fix is NOT a loss knob.
⚠️ A negative from a linear probe is a negative about **linear** decodability
only; the verdict will say so.

## 2. P2 — THE PRE-REGISTERED ACCEPTANCE THRESHOLD

⛔ **Registered before any refit was run. The acceptance test is NOT a loss curve.**

Let `S+` = the set of windows on which the ground-truth future turns LEFT.
On the banked dump that is **1,597 of 4,823 windows** (GT terminal ego-frame
`y > +1.0 m`), reproduced independently by this work package's own census.

**GATE G1 — TWO-SIDEDNESS.** The repaired `g_str`'s lateral component must be
**positive (LEFT) on at least 15.0 %** of all scored windows.
*Rationale, fixed in advance:* the LAN TARGET is LEFT on **268 / 3,970 = 6.75 %**
of label-valid windows at the τ = 0.10 deadband and **41.4 %** by raw sign; the
GT future turns left on **33.1 %**. 15.0 % by raw sign sits strictly between the
τ-deadband label rate and the GT rate, so it cannot be cleared by noise around
zero and does not demand more left-turning than the teacher contains.

**GATE G2 — LEFT-TURN COVERAGE.** On `S+` (GT-left windows) the repaired
`g_str` must be **non-negative on at least 30.0 %**. *Rationale:* the current
arm is **0.0 %** on 1,597 / 1,597; 30 % is the point at which "cannot express
left" becomes false as a statement about the signal, not merely about its mean.

**GATE G3 — NOT BOUGHT WITH ACCURACY.** The repaired head's `l_bear` must be
**≤ the constant-straight floor's upper CI bound** on the scored split. A head
that turns left by becoming random is not a fix.

⛔ **ALL THREE must pass. Any one failing = the arm FAILED, reported as FAILED.**

**DOWNSTREAM CONSEQUENCE (committed both ways).** If the mechanism is right,
`turn_left` recall (currently **0.0000 of 11**) must become **non-zero**.
⛔ **If the sign is fixed and `turn_left` recall stays at zero, the mechanism was
NOT the cause** — that will be stated in those words, and the next candidate named.

**GATE G0 — THE DELIBERATE REGRESSION.** A `ONESIDED` arm that re-introduces the
defect by construction (the head's lateral output passed through `-softplus`,
so it is negative by construction and left is unreachable) must **FAIL G1 and
G2**. ⛔ A gate that has never been shown to fail measures nothing.

## 3. P3 — the nav-args wiring, and what would make it inadmissible

**Change:** the refcv3 loader reads `nav_command["args"]["distance_m"]` and
`["time_s"]` and presents them to the model, following `NavConditioner`'s
existing contract (`arg_proj = nn.Linear(2, d_embed)`), not a new one.

⛔ **Three failure modes, each with a committed guard:**
1. **`NAV_FOLLOW_ROAD` carries `args: {}`** on 2,897/2,897 train and 96/96 eval.
   ⇒ an explicit **VALIDITY CHANNEL** ships with the values; a silent `0.0` is a
   lie that reads *"the turn is here, now"*. The tensor is **[B, 3]**:
   `(distance_norm, time_norm, args_valid)`.
2. **69.97 % of train records present `distance_m = 0.0`** under the consumer's
   `t0_constant` default. ⇒ the **fraction of windows receiving a REAL distance
   is measured and reported** before the channel is claimed to carry information.
3. ⛔ **UNITS.** `distance_m` is **METRES**, `time_s` is **SECONDS**. The
   normalisation constants and the unit names are written into `config.json`
   and into the tensor's own metadata; a builder that declares nothing is
   refused. *(An anchor file without declared units once produced 396 g instead
   of 0.31 g and both tables looked plausible.)*

⛔ **Zero-init.** The new projection is zero-initialised so an arm with the flag
ON is **bit-identical at step 0** to one without it — the E13/H19 discipline
already used for nav and for the goal point. Attribution, not decoration.

⛔ **Default OFF.** No banked arm's recipe changes. The five hard-guarded
zero-weight terms (`--w-agent`, `--w-u0`, `--goal-point-w`,
`--agent-w-project`, `--agent-w-ground`) are not touched.

## 4. What this pre-registration does NOT do

* ⛔ It does not touch `route_logits`. The arm's config records
  `"graft_route": false` ⇒ `route_prior` is `None`; it causes nothing, and its
  fate is a deliberate decision ranked third, not a side effect of this work.
* ⛔ It does not claim a full retrain. A 40 k-step arm is not in this turn's
  budget; the head refit is a **frozen-trunk** experiment and is labelled as one
  everywhere it is quoted.
* ⚠️ A separated CI from one seed is necessary, not sufficient (measured
  false-positive rate 6/42 = 14.3 %). The refit therefore carries a
  **REPLICATE** arm (same data, same loss, different seed) as its own floor.
