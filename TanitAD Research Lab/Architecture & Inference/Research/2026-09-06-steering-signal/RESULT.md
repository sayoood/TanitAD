# RESULT — D-GSTR-1: the steering signal's teacher is fine, its student is not

`Work package: TanitAD Research Lab/Architecture & Inference/Research/2026-09-06-steering-signal/`
`Owner: steering-signal agent. Date: 2026-09-06. Pre-registration: PREREG.md (gates written before any refit ran).`
`Antecedent: .../Research/2026-09-06-nav-vs-routehead/RESULT.md (commit a07d733d7).`

**Evidence class: MEASURED (ours)** unless a row says otherwise. **Tier: T1**
(self-action OPEN loop, 2026-09-02 ruling) for every model number; the LAN
corridor is a **LABEL** and carries no tier. **Estimator:** paired
episode-cluster bootstrap over the dump's episodes, 2,000 resamples, seed 0.
⛔ No `overlapping_holdout_se` anywhere in this file.
**Surface:** one — Thor, refcv4b `ckpt_40284_FINAL.pt`, the banked 141-episode
`navflip_dump`. **Roll:** the banked A40-landing roll, re-read; no arm re-rolled.

---

## 0. The answer, in one table

⭐ **THE SIGN HISTOGRAMS, SIDE BY SIDE** — same 4,823 windows / 141 episodes,
same deadband `tau_g = 0.10`, same lateral slot (`[:, 1]`), `+` is LEFT.
*(`raw/GSTR_SIGN_CENSUS.json`)*

| | LEFT | STRAIGHT | RIGHT |
|---|---|---|---|
| **`g_str`'s TARGET** — the LAN corridor label, all windows | **268** | 4,083 | **472** |
| **`g_str`'s TARGET** — label-valid only (n = 3,970) | **268** | 3,230 | **472** |
| ⛔ **`g_str`'s OUTPUT** — the deployed head, all windows | **0** | **0** | **4,823** |
| ⛔ **`g_str`'s OUTPUT** — label-valid only | **0** | **0** | **3,970** |
| *(reference)* GT future, 6 s terminal `y`, `tau = 1.0 m` | 1,597 | 981 | 2,245 |

⇒ ⭐⭐ **THE TEACHER IS TWO-SIDED AND THE STUDENT IS NOT.** Every candidate the
brief named is REFUTED by this table plus its controls:

| candidate | verdict | what refutes it |
|---|---|---|
| M1 sign / absolute-value in the construction or the target | ⛔ **REFUTED** | the target takes both signs through the SAME normalisation path |
| M2 coordinate-frame / handedness error | ⛔ **REFUTED** | three controls read their known values (below) |
| M3 a parameterisation that cannot represent negative-to-positive | ⛔ **REFUTED** | `bearing = g[:, :2] / ‖g[:, :2]‖` is sign-symmetric, and the target proves it |
| M4 a LABEL defect (one-sided teacher) | ⛔ **REFUTED** | 41.39 % of valid targets are positive by raw sign |
| ⭐ **M5 the head is a FAILING LEARNER on a sound teacher** | **SUPPORTED, and localised further in §2** | it loses to a constant |

⛔ **CONTROLS THAT READ THEIR KNOWN VALUES** — without these the table above is
a claim about my probe, not about the model:

| control | rule | reading | |
|---|---|---|---|
| **C1 INDEX** | `ep_poses[ws + off]` must equal the dumped `pose_last` on every window | **4,823 / 4,823 resolved, 0 unresolved, offset 0** | **PASS** |
| **C2 MIRROR** | negating world `y` (and yaw) must FLIP the target's lateral sign | flipped on all 3,970 non-zero rows; histogram mirrors to LEFT 472 / RIGHT 268 | **PASS** |
| **C3 STRAIGHT** | a synthetic straight path must give lateral EXACTLY 0.0 | `lateral = 0.0`, `cos = 1.0`, valid | **PASS** |
| **C4 GT cross-check** | the census must reproduce the antecedent's GT histogram | **1,597 / 981 / 2,245 — exact** | **PASS** |

---

## 1. P1 — the shape of the defect: a RIGID −29 DEGREE ROTATION

*(`raw/GSTR_SIGN_CENSUS.json`, `raw/GSTR_LOSS_DECOMP.json`)*

| quantity | TARGET (label-valid, n = 3,970) | OUTPUT `g_str` (n = 4,823) |
|---|---|---|
| lateral min / p25 / median / p75 / max | −0.8061 / −0.0233 / **−0.0025** / +0.0094 / +0.8559 | −0.9711 / −0.6583 / **−0.5308** / −0.3917 / **−0.1022** |
| frac negative / positive | 0.5861 / **0.4139** | **1.0000** / **0.0000** |
| bearing angle p5 / median / p95 | −15.949° / **−0.144°** / +8.676° | −48.233° / **−29.995°** / −11.311° |
| frac LEFT by angle | **0.4139** | **0.0000** |

⭐ **mean signed offset, model − target: −29.008°.** The output's angular SPREAD
(p5→p95 ≈ 37°) is comparable to the target's (≈ 25°) — the head is **not**
collapsed to a point. It is the target distribution **rigidly rotated ~30° to
the right**, far enough that its whole range sits on one side of zero. And the
`|lateral| ≥ 0.1022` floor is why the deadband never even records a STRAIGHT.

**SHAPE ASSERTION (so the numbers are about what I think they are):**
`gstr_nav_true[:, :2]` has norm **1.000000** on every window — it IS the model's
unit bearing (`refc_v3.py:979-982`) — and its `cos` is **positive on 100 %**
(min 0.2386), so the head never points backwards. **PASS.**

### 1.1 ⛔ IT LOSES TO A CONSTANT — and that is the sharpest fact in this file

`strategic_goal_loss = l_bear + l_dist`, masked by label validity; scored on the
same windows the head was measured on (n = 3,970 / 137 episodes):

| term | MODEL | no-information FLOOR | model − floor |
|---|---|---|---|
| `l_bear` = 1 − cos(model, target) | **0.1500 [0.1313, 0.1689]** | constant straight-ahead **0.0119 [0.0080, 0.0164]**; best constant **0.0117** | ⛔ **+0.1381 [+0.1197, +0.1558] SEPARATED** |
| `l_dist` = \|tanh(g₂) − dist_target\| | **0.5046 [0.4715, 0.5361]** | best constant (the median, −1/3) **0.3266 [0.2757, 0.3732]** | worse |
| **total** | **0.6546** | **0.3385** | **1.93×** |

The loss-optimal constant bearing is **[0.9998, −0.0182] = −1.04°**. The head
sits at **−30°**. ⇒ **A one-layer linear head, stepped ~40,000 times by Adam on
its own dedicated loss, does not reach the score of a fixed vector.**

⚠️ **AND IT IS NOT OVERFITTING OR DISTRIBUTION SHIFT.** The trainer's own
`goal_str` on the **TRAIN** batches (`refcv4b_metrics_31700.jsonl`, 634 logged
rows) reads **0.98 mean over steps 0–2 k**, RISES to **1.18 at 2–4 k**, and
decays only to **0.84 at 30–32 k** — i.e. the head is **worse on its training
data (0.84) than on the held-out eval (0.65)**, and never approaches the ~0.34
constant floor. It is equally bad everywhere.

### 1.2 Where the −29° lives: NOT the bias, and NOT a frozen head

*(`raw/GSTR_HEAD_AUDIT.json`, from the checkpoint alone)*

* **The head was trained.** The checkpoint carries optimizer state (`opt`,
  347 parameters with Adam state, **step 40,127–40,284**). It is in the
  optimizer and it was stepped. ⛔ "the head is frozen" is REFUTED.
* **The offset is not in the bias.** `bias = [−0.00336, +0.000341, +0.033871]`;
  `‖bias[:2]‖ = 0.003377` against weight-row norms **0.6879 / 0.4798 / 0.4501**.
  The bias is ~200× too small to set a 30° offset, and its own angle is +174°.
  ⇒ **the constant direction is produced by `W · ctx`**, not by an offset the
  optimiser failed to move.
* **The weights barely moved.** `‖W‖₂ = 0.9518` against a fresh
  `Linear(256, 3)`'s **1.0167**; `absmean 0.0281` vs **0.0320**; `max|w| 0.0887`
  against the init bound `1/sqrt(256) = 0.0625`.

⚠️ **A DEFECT IN MY OWN PROBE, DISCLOSED:** the audit's control list matched
`goal_head` as a SUBSTRING of `str_goal_head`, so its `goal_head` row is a
duplicate of the subject and is **not** an independent control. It is not used
in any conclusion above; the fresh-init row is.

---

## 2. P1 — M5a or M5b? THE ORACLE PROBE, and its own retraction

*(`raw/GSTR_REFIT.json`; episode-disjoint fit 99 ep / 3,464 win, scored 42 ep /
1,463 win, label-valid scored 1,229; **d = 256**; λ selected on a fit-INTERNAL,
episode-disjoint validation split — the scored split is scored, never tuned on.)*

⚠️ **RETRACTION, SAME DAY, SAME FILE.** v1 of this probe had **no intercept**:
its ridge shrank toward the **zero vector**, not toward the constant, so it
scored **+0.6738 WORSE than its own control** and I nearly published "M5b —
`ctx` does not carry the bearing". That is the 2026-08-22 ridge family exactly:
*a probe whose degenerate limit is not the control it is compared against.*
⇒ Fixed **with a control, not a patch**: v2 asserts that at λ = 1e12 the ridge's
score EQUALS the constant's, and refuses a verdict otherwise.
**Reading: 0.014024 vs 0.014024. PASS.**

| row | `l_bear` on the scored split | |
|---|---|---|
| ⭐ **ORACLE RIDGE on `ctx`** | **0.0060 [0.0028, 0.0101]** | |
| CONTROL — constant only | 0.0140 [0.0063, 0.0238] | must read the closed-form floor — it does |
| CONTROL — constant straight-ahead | 0.0139 [0.0060, 0.0237] | |
| FLOOR — nav-token one-hot (trivial input) | 0.0140 [0.0067, 0.0235] | the nav token alone carries **nothing** about the bearing |
| ⛔ **LIVE MODEL `g_str`** | **0.1321 [0.1030, 0.1631]** | |
| **oracle − constant** | ⭐ **−0.0080 [−0.0142, −0.0033] SEPARATED** | |
| **model − constant** | ⛔ **+0.1180 [+0.0913, +0.1461] SEPARATED** | |

### ⭐⭐ 2.1 The question that actually matters — and its answer

The full-set regression is **81 % near-straight windows**, so it can hide the
only thing a route bearing is for. Restricted to the windows where the LAN route
**actually turns** (|target lateral| > 0.10; fit 535, **scored 216 over 17
episodes**):

| arm | turn-direction accuracy |
|---|---|
| ⭐ **ridge on the SAME `ctx`** | **0.8981 [0.8333, 0.9555]** |
| CONTROL — majority class | **0.5139** |
| ⛔ **live `g_str`** | **0.4861** — *below the majority control* |

⇒ ⭐⭐ **THE MECHANISM: `ctx` CARRIES THE LEFT/RIGHT ROUTE TOPOLOGY ALMOST
PERFECTLY, LINEARLY — AND THE DEPLOYED HEAD THROWS IT AWAY.** The information
is one `Linear(256, 3)` away and 89.8 % recoverable. **This is an OPTIMISATION
failure (M5a), not a representation failure (M5b).**

⚠️ Scope: a linear probe's POSITIVE result is strong (it proves decodability);
this one is positive. The function class is stated because the rule requires it.

---

## 3. P2 — the fix, against its PRE-REGISTERED gates

Arms: the **same** `nn.Linear(256, 3)`, initialised from the checkpoint's own
weights, retrained on the trainer's own `strategic_goal_loss` with **the trunk
FROZEN**, scored on the held-out 42 episodes / 1,463 windows (GT-left at 6 s:
**n = 294**). ⚠️ **A frozen-trunk refit is not a retrained arm and is never
quoted as one.**

**Gates, fixed in `PREREG.md` §2 before any arm ran:** **G1** lateral positive on
≥ **15.0 %** of scored windows · **G2** lateral non-negative on ≥ **30.0 %** of
GT-left windows · **G3** `l_bear` ≤ the constant floor's upper CI (**0.0237**).
⛔ All three must pass.

| arm | G1 | G2 | G3 | turn-sign acc ⁽ᵖᵒˢᵗ⁾ | VERDICT |
|---|---|---|---|---|---|
| ⛔ **LIVE refcv4b** | 0.0000 ✗ | 0.0000 ✗ | 0.1321 ✗ | 0.4861 | **FAILED** |
| CONTROL — constant only | 0.0000 ✗ | 0.0000 ✗ | 0.0140 ✓ | 0.4861 | FAILED *(as required)* |
| ⛔ **D — ONESIDED, the deliberate regression** | **0.0000 ✗** | **0.0000 ✗** | 0.0143 ✓ | 0.4861 | **FAILED — as required** |
| B — refit, seed 0 | 0.3684 ✓ | 0.3197 ✓ | 0.0144 ✓ | **0.4491** | PASS |
| ⛔ **C — REPLICATE, seed 1** | 0.2379 ✓ | **0.2347 ✗** | 0.0144 ✓ | 0.4491 | **FAILED** |
| ⭐ **E — turn-weighted, seed 0** | 0.2454 ✓ | **0.4966 ✓** | **0.0116 ✓** | **0.8935** | **PASS** |
| ⭐ **E2 — turn-weighted REPLICATE, seed 1** | 0.2297 ✓ | **0.4762 ✓** | **0.0113 ✓** | **0.8843** | **PASS** |

### 3.1 ⛔ THE GATE CAN FAIL — three ways, all shown

`GATE_VALIDITY`: `regression_failed_as_required` **true** ·
`constant_control_failed_G1_as_required` **true** ·
`live_arm_failed_as_expected` **true** ·
`ridge_no_information_limit_control` **true**.
The `ONESIDED` arm re-introduces the defect by construction (lateral through
`−softplus`, so left is unreachable however good the fit) and reads **0.0000 on
both G1 and G2**. ⭐ A gate that has never been shown to FAIL measures nothing;
this one fails on the regression, on the constant control, and on the live arm.

### 3.2 ⛔ B FAILED, AND SAYING SO IS THE POINT

The unweighted refit **B passes on seed 0 and its REPLICATE fails G2 on seed 1**
(0.2347 vs the 0.300 threshold). Under `H-ESTIM-SEED-1` a one-seed pass is
necessary, not sufficient. **B is NOT certified.**

⚠️ **AND THE PRE-REGISTERED GATES TURNED OUT TO BE NECESSARY, NOT SUFFICIENT —
stated because it is a finding about my own instrument.** B cleared G1+G2+G3
while its **turn-direction accuracy was 0.4491 — worse than the live model's
0.4861 and worse than the 0.5139 majority control.** A head can satisfy "takes
both signs" and "matches the constant's accuracy" while knowing **nothing** about
which way the route goes. ⛔ The `turn_sign_accuracy` column is **POST-HOC** (it
was added after seeing §2.1) and is reported as a readout, not as a gate; using
it to decide anything needs its own pre-registration.

### 3.3 ⭐ THE CERTIFIED FIX: reweight the loss by the turn, not the head

§2.1 named the lever and the arithmetic ranks it first: the cosine loss is 81 %
near-straight rows, so the turns — the only rows carrying left/right — contribute
almost none of the gradient. **E** upweights each row by its own label lateral,
`w = 1 + 20·|sin_target|` — a property of the LABEL alone: **no new input, no
threshold tuned on the scored split, no architecture change.**

⭐ **E and its REPLICATE both clear all three gates**, and E's turn-direction
accuracy is **0.8935 / 0.8843 against the 0.8981 linear oracle** — it recovers
**essentially all** of what `ctx` linearly contains. G3 does not merely hold, it
**improves**: `l_bear` **0.0116 / 0.0113**, *better* than the 0.0140 constant.
⇒ the left turns are **not** bought with accuracy.

`SEED_ROBUSTNESS`: `B_vs_C_unweighted` **CERTIFIED: false** ·
`E_vs_E2_turnweighted` **CERTIFIED: true**.

---

## 4. P3 — the nav args are now WIRED, with their validity channel and their units

**Landed (staged, not committed to `main`):**

| file | what |
|---|---|
| `stack/tanitad/refs/refc_v3.py` | `NAV_ARG_DIMS = 3`, `NAV_ARG_UNITS`, `cfg.nav_args_inject` (default **False**), `nav_arg_proj`, the hook's additive term, **both** forward refusals, the capacity ledger line |
| `stack/scripts/refc_v3_train.py` | the args captured in `enable_nav_from_v7` (they were being discarded one line from where they arrive), `enable_nav_args()` + census, the `__getitem__` block, `--nav-args`, the forward wiring, the `config.json` record |
| `stack/tests/test_refc_v3_nav_args.py` | **21 tests, all passing** — every guard proven by MUTATION |

**The contract**: `NavConditioner.encode` is `embed(token_id) + arg_proj(args)`;
this is that line with a third slot. ⛔ Following the class rather than importing
it is deliberate — its `arg_proj` is `Linear(2, d_embed)` and widening it to 3
would change the shape every v6/v7f checkpoint loads.

**1. THE VALIDITY CHANNEL.** The block is **[B, 3] = (distance_norm, time_norm,
args_valid)**. The bit is read from **the KEY'S PRESENCE**, never the value —
`_args_for_window` defaults a missing slot to `0.0`, so `distance_m == 0.0` is
ambiguous between *"the turn is 0 m away"* and *"there is no turn"*. The MODEL
re-applies the bit (X15's rule) so a caller that forgot to zero an invalid row
cannot leak a phantom range; the bit itself always passes, so *"no range known"*
stays a distinct, learnable input. Two mutation tests pin exactly this.

**2. THE FRACTION OF WINDOWS CARRYING A REAL DISTANCE** *(`raw/NAVARGS_CENSUS.json`)*:

| split | records | with real args | per-record | per-WINDOW |
|---|---|---|---|---|
| train (`md5 0ff90213…`) | 4,572 | 1,675 | **0.3664** | *(computed and stamped by the loader at run time)* |
| eval (`md5 aa12c948…`) | 147 | 51 | **0.3469** | ⭐ **0.3623 (1,785 / 4,927 windows, 51 / 141 clips)** |

⭐ **The split is EXACT and it is the token**: `NAV_TURN_L` **811/811** and
`NAV_TURN_R` **864/864** carry args; `NAV_FOLLOW_ROAD` **0 / 2,897**. Same on
eval (13/13, 38/38, 0/96). Mean real distance **71.7 m / 10.1 s** (train),
**102.3 m / 13.1 s** (eval).
⚠️ **Stated honestly, because it scopes the claim:** on THIS corpus `args_valid`
is an exact function of the token, so the BIT adds no information the token does
not already carry. It is a **safety** channel (it makes the silent-zero lie
impossible), not an information channel. **The information is the VALUE on the
36 % of windows that are turns** — and per §2.1 those are exactly the windows
where the steering signal has been wrong.
⛔ The per-record figure is **not** the per-window figure and neither is quoted
for the other; the loader computes the per-window number for whatever corpus it
is given and writes it into `config.json` as `window_real_distance_frac`.

**3. UNITS.** `distance_m` is **METRES**, `time_s` is **SECONDS**;
`NAV_ARG_UNITS = ("distance_m:metres", "time_s:seconds")` is a module constant,
asserted by a test against the census, and written into `config.json` together
with the slot names, the `t0_constant` semantics and the **fit-split-only**
normaliser (`NavArgStats`, fitted on the TRAIN split's VALID rows and handed
down to the eval dataset — a test pins that the eval dataset never fits its own).

**Refusals, all mutation-tested:** args without the seam · the seam without args
· a 2-wide block (the bit dropped) · `--nav-args` without `--nav-from-v7` ·
`--nav-args` with the nav path switched off by `--goal-point-inject`.
**Zero-init:** unchanged — `nav_to_tac` / `nav_to_str` are already zeroed, so an
arm with the flag ON is **bit-identical at step 0** to one without it (pinned).

⛔ **Not touched, deliberately:** `route_logits` (the arm's config records
`"graft_route": false`); the five hard-guarded zero-weight terms; every default.

---

## 5. VERDICT

⭐ **The steering signal's defect is NOT a sign error, NOT a frame error, NOT a
parameterisation limit and NOT a broken teacher. It is a head that loses to a
constant on a two-sided label while the context it reads supports 89.8 %
turn-direction accuracy — and the lever is the LOSS BALANCE, not the head.**

**What is certified:** the turn-weighted refit **E** clears all three
pre-registered gates on **both** seeds, recovers **0.8935 / 0.8843** turn-direction
accuracy against the **0.8981** linear oracle, and *improves* bearing accuracy
while doing it. **What FAILED:** the live arm (all three gates), the unweighted
refit's replicate (G2), and the deliberate regression (as required).

### ⛔ What is NOT claimed, and what blocks it

* ⛔ **`turn_left` recall is NOT moved by this work, and I do not claim the
  mechanism is proven to cause it.** `turn_left` recall 0.0000 of 11 is a
  **TACTICAL** head metric; `g_str` reaches the tactical state only through the
  zero-init E4 FiLM. A broken strategic goal is a **necessary-not-sufficient**
  explanation. ⛔ **The discriminating experiment is a RETRAIN with the
  turn-weighted `strategic_goal_loss`** — a ~40 k-step arm. **BLOCKER: compute +
  a PI decision on arm scheduling.** It is the single cheapest experiment that
  would settle it and it is not in this turn's budget.
* ⚠️ The refit is **frozen-trunk**. The FiLM and every downstream weight were
  trained against the broken goal, so §6 is a **lower bound** on a retrain.
* ⚠️ `turn_sign_accuracy` is **POST-HOC**; it decides nothing here.
* ⚠️ Both `l_dist` and the `dist_pref` half carry the DECLARED speed confound
  (`REFC_V3_DESIGN §4.4`); nothing above rests on that half.

### The next levers, ranked by MEASURED effect

1. ⭐ **Retrain refcv4b with the turn-weighted `strategic_goal_loss`.** Measured
   frozen-trunk effect: turn-direction 0.4861 → 0.8935/0.8843, `l_bear`
   0.1321 → 0.0116/0.0113. One line in `refc_v3.py`; needs an arm slot.
2. ⭐ **Run that arm WITH `--nav-args`** (landed here, default OFF). The
   information is on the 36 % of windows that are turns — the same windows the
   steering signal was wrong on — and a range-carrying goal is the published
   lever (bearing-only is separated WORSE by +2.3632 m).
3. **Then, and only then, re-read `turn_left` recall.** ⛔ If the sign is fixed
   and left-turn recall stays at zero, the mechanism was NOT the cause and the
   next candidate is the **E4 FiLM gate** — whether `g_str` reaches `z_tac` with
   any magnitude at all.
