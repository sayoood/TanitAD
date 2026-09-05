# RESULT — CLOSING refav1's GAPS

**Package** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-close-the-gaps`
**Pre-registration** `SPEC_CLOSE_THE_GAPS.md`, written 00:40 with §6 amended 01:35 —
**both before any arm below existed.**
**Rig** refav1 `ckpt_ep3` step 21109 · p4 panel 40 windows / 8 episodes / stride 16 ·
`--no-lead-block --no-navshuf` · `ccos` · planner arm `cl` at **T1 (self-action open loop)**.
⛔ **T1, never T0.** `ol` is the recorded-future control and is never a driving number.

---

## §A — THE HEADLINE, IN ONE PARAGRAPH

⭐⭐ **refav1's turn collapse is NOT a turn suppression — it is a 3.9x UNDER-TURN caused by an
operative cost that charges curvature without asking what the tactical brain decoded.** MEASURED,
not hypothesised: at `W_KAPPA = 15.11245` the plan still emits curvature on **100 %** of
TURN_L-goal windows, at **0.02066** against the goal's commanded **0.08000** — under the
trajectory labeller's turn gate, so `turn_left` recall reads 0.0000 while ADE improves.
**The hierarchy was fighting itself: the tactical brain says TURN_L and the operative cost fines
the planner for obeying.** The fix — a **goal-conditioned lateral cost** — is implemented, tested
(15 new tests, 394 refa_v1 tests green, flag-off bit-identical), and **two arms are running**.

---

## §B — WHAT IS MEASURED AND FINISHED (0 GPU)

### B1 ⭐⭐ THE MECHANISM — where the penalty's damage actually falls

`raw/kappa_by_goal_all.txt` (predecessor stream, n = 40, vocabulary imported from
`tanitad.models.vocab_v7`), realised planner curvature **split by the DECODED lateral goal token**:

| arm | `W_KAPPA` | LANE_KEEP (n=18) mean k² | LANE_KEEP frac k≠0 | **TURN_L (n=9) med\|k\|max** | TURN_R (n=13) med\|k\|max |
|---|---|---|---|---|---|
| `ccos_argmax` | 0 | 0.002908 | 0.4444 | **0.08000** | **0.08000** |
| `wk15` | 15.11245 | **0.000000** | **0.0000** | ⛔ **0.02066** | 0.08000 |
| `wk151` | 151.1245 | 0.000000 | 0.0000 | ⛔ 0.00840 | ⛔ 0.01317 |

Three findings, each independent:

1. ⛔ **The damage is a MAGNITUDE collapse, not an on/off.** `frac k != 0` stays **1.0000** on
   TURN_L under `wk15`. The planner turns; it turns at **26 %** of what it was told to. Recall only
   thresholds that. ⇒ Every prior reading of "`W_KAPPA` kills turning" was reading a **gate**, not
   the quantity — and the quantity is a knob.
2. ⚠️ **The damage is ASYMMETRIC and a symmetric penalty cannot be its source.** TURN_R is
   **unchanged at 0.08000** under the same k² term. ⇒ the **goal term's magnitude differs by
   direction**, so an identical k² overwhelms the left-turn goal gradient and not the right-turn
   one. **refav1's turn asymmetry is a COST-BALANCE fact, not a head fact** — which is a different
   place to look than the turn-asymmetry panel is currently looking.
3. ⭐ **This is `M48` validated on the frontier.** A penalty RANKS, so a quadratic curvature penalty
   prefers the cheapest non-zero curvature and under-turns everywhere, **including where turning is
   correct**. A constraint FORBIDS without ranking. Hence `kamm07` (constraint) captures
   **77 %** of `wk15`'s (penalty) ADE gain at **zero turning cost**: `ccos_argmax -> kamm07` is
   **−0.3345 m** with `turn_left` unchanged at 0.3636; `ccos_argmax -> wk15` is **−0.4338 m** and
   kills it. `0.3345 / 0.4338 = 0.7712`. **Re-derived from the records, reproduces.**

### B2 ⭐⭐ THE SEED FLOOR IS A PROPERTY OF THE ARM, NOT OF THE RIG — a 22x spread

MEASURED in `raw/frontier.txt`, three inference-seed replicate pairs on the identical grid, every
grid control PASSED (`ha` 0.8888 / `ha0` 0.9251 / `ha0_ext` 0.8772 / `ol` 0.8052 in all six runs):

| pair | configuration | **ADE seed floor** | turn_L floor | turn_R floor |
|---|---|---|---|---|
| `ccos_argmax` / `ccos_seed1` | W_KAPPA **0** — search unconstrained | **0.06070** | 0.00000 | 0.00000 |
| `combined` / `combined_seed1` | W_KAPPA 0 + Kamm cap + ladder | **0.10350** | 0.00000 | 0.00000 |
| `T_wk15` / `T_wk15_s1` (Thor) | W_KAPPA **15.11245** — curvature crushed | **0.00470** | 0.00000 | 0.00000 |

⛔ **A 22x spread in the ADE floor across configurations of the same model on the same windows.**
The heavily penalised arm barely moves under a seed change because the penalty has removed the
search's freedom. ⇒ **Quoting one arm's floor for a different arm's configuration is the
`df` / Thor-`free` / `step_s` scope error in a new costume** — a true measurement applied outside
the thing it measured. A floor must be quoted for the **statistic AND the configuration**.
⭐ **`turn_left` / `turn_right` recall floors are EXACTLY 0.00000 on all three pairs** — every
replicate reproduced its recall to four decimals. ⇒ **any turn-recall change on this rig is above
its own inference-seed noise floor**, which is what makes the 0.3636 -> 0.0000 collapse admissible
as a lever effect rather than sampling noise.

### B3 — THREE PREDECESSOR CAUTIONS DISCHARGED

| caution | verdict |
|---|---|
| `cos_wk` suggests a tiny penalty collapses turning | ⛔ **EXCLUDED.** `kappa_by_goal_all.txt`: med\|k\|max **0.00000** and frac k≠0 **0.0000 on all three goal tokens** — an all-zero path. Its 0.9251 is a degenerate-baseline property, the poisoned-floor-arm class. |
| `oracle_s0` / `ccosh_w000` / `ccos_argmax` are bit-identical (`M40` shared denominator?) | ⭐ **NOT a shared denominator — a MEASURED behavioural null.** `raw/ccosh_null.txt`: the `ccosh` hold branch **fires** (`basecost_cv` frac exactly 1.0 goes 0.2500 -> 0.0000; min 0.884896 -> 3.9105e-08; cem frac 0.750 -> 0.925) and **no plan changes** (max abs diff **0.000000e+00**), against a same-breath control that must differ (`wk15` vs `ccos`, **1.508402e+01**). `oracle_s0` names a nav provenance, not a lever (`M51`: the GT arm is `ol`). |
| `lonseam` is a longitudinal lever result | ⛔ **INERT BY CONSTRUCTION** and correctly refused by the tool's own guard (`refav1_arm.py:809`): `--jerk-seam a0` multiplied by `W_JERK = 0.0` cannot move the objective by a bit. Reads bit-identical to `wk15` on **both rigs independently** (dev box 0.8934/0.03098; Thor `T_lonseam` 0.8910/0.03042 = `T_wk15`). **A null control that reads a known value — quoted only as one.** |

### B4 — TWO NEW FACTS THE FRONTIER RE-DERIVATION TURNED UP

* ⛔ **`bestlad` (W_KAPPA + Kamm cap + ladder) reads ADE 0.9418 with `turn_left` 0.0000 AND
  `turn_right` 0.0000.** That answers the sibling SPEC's open question — *"when you add W_KAPPA on
  top of cap+ladder, which behaviour wins?"* — **the penalty wins and turning dies entirely.** Its
  gate **T (STILL TURNS) FAILS.** So does `best`'s (0.8838, turn_left 0.0000).
* ⛔ **`combined_seed1` FAILS the sibling's §1 committed criterion**: `kamm_over_rate`
  0.0000 -> **0.07407** under an inference-seed change alone. The programme's "first zero-violation
  arm" is **seed-dependent**. That is the sibling stream's retraction to make (their
  `seed_floor_ext.py` was running when this was written); flagged here, not duplicated.

---

## §C — A3, THE FIX: IMPLEMENTED, TESTED, RUNNING

### C1 — What was built

`w_kappa_by_goal` makes the lateral cost a function of the **decoded tactical goal token**:

* `stack/tanitad/refs/refa_v1.py` — `GOAL_KAPPA_COST_CLASSES`, `goal_lat_cost_class()`,
  `_parse_w_kappa_by_goal()`, the `plan(w_kappa_by_goal=...)` parameter, the application site, and
  three provenance fields on the result (`w_kappa_by_goal`, `w_kappa_goal_class`,
  `w_kappa_effective`).
* `taniteval/tools/refav1_arm.py` — `--w-kappa-by-goal`, a **stale-stack signature gate**, an
  **inert-map refusal** (an all-equal map is bit-identical to the scalar and would bank a
  meaningless +0.0000 under a lever's name — the `--jerk-seam` refusal generalised), the record's
  `cost.w_kappa_by_goal` block, and per-window `w_kappa_class_<arm>` / `w_kappa_eff_<arm>` columns
  in the decisions sidecar.
* `stack/tests/test_refa_v1_goal_kappa_cost.py` — **15 tests, all passing.**

**Regression: 394 passed, 1 skipped** over the whole `refa_v1`/`refav1` suite on the edited tree.
The flag-off path is pinned **bit-identical**, with a same-breath control that must differ.

### C2 ⛔ ADMISSIBILITY — answered from SOURCE before a line of code

The conditioning signal is `lat_i = argmax(self.lat_head(intent))` (`refa_v1.py:2203-2212`), the
model's own decoded tactical lateral **action** token.

1. ⭐ **"Could this have been computed from the SITUATION CLASSIFIER's output?" — NO, and it is a
   DIFFERENT LABEL FAMILY.** `stack/tanitad/data/situations.py` emits `lane_change` /
   `intersection` / `roundabout`. `lat_head` is supervised by `lat_label` from the v7.2 label
   release over `TACTICAL_LAT_ACTIONS_V7` = `LANE_KEEP, LANE_CHANGE_L, LANE_CHANGE_R, ABORT_LC,
   NUDGE_L, NUDGE_R, TURN_L, TURN_R` (`vocab_v7.py:290`). **Literal probe with a same-breath
   control:** `grep -cF situations stack/scripts/s2_labels.py` = **0** against
   `grep -cF "def " …` = **23** on the same file, so the channel read. No situation-classifier
   posterior, argmax, embedding or derivative enters this path.
2. ⭐⭐ **The strongest form: A3 ADDS NO INFORMATION AT ALL.** That token **already** builds
   `goal_t` (the goal term itself, `refa_v1.py:2546`) and **already** seeds iCEM
   (`goal_action["controls"]`, `:2572`). A3 re-uses a signal the cost already consumes; it opens
   **no new channel**, so there is nothing it could smuggle. Pinned mechanically by
   `test_e_conditioning_signal_is_the_same_token_that_builds_the_goal`.
3. ⭐ **Inference stays VISION-ONLY.** `intent` comes from `_run_brains(pooled_win, nav_cmd)`;
   `pooled_win` is the vision feature window, and in these arms `nav_cmd = 0` with
   `nav_valid = False`. The only ego quantity in the path is **`v0` measured at t0**, legal under
   the PI ruling of 2026-09-02. No future, no privileged channel.
4. ⭐ **Attribution survives.** The goal path and the k weight are the same path by construction, so
   there is no second path to confound, and the realised class + weight are banked per window.

### C3 — The arms, and their committed readings (⛔ written before they existed)

Running on **Thor**, in the two slots `queueTHOR.sh` freed when it exhausted its 7-arm plan.
⛔ Paired **within-rig** against the Thor-local `T_wk15` (CEM is not bit-reproducible across GPUs).

| arm | `--w-kappa-by-goal` | role | committed reading |
|---|---|---|---|
| `G_gkappa` | `15.11245,0.0` | ⭐ **THE ARM** | `turn_left` recall **> 0.0000** AND `TURN_L med\|k\|max` **≥ 0.06** (back toward the commanded 0.08000), with ADE better than the unconstrained base by more than **0.06070**. |
| `G_gkappa_inv` | `0.0,15.11245` | ⛔ **DELIBERATE REGRESSION** | the INVERTED cost. **Must be worse than `G_gkappa` on `turn_left` recall AND on ADE.** If it is not, the conditioning is inert and `G_gkappa` is noise. |

⭐ **The known-value control is WITHIN `G_gkappa` and costs no extra GPU** (§6.3): it charges the
**same 15.11245** on LANE_KEEP-goal windows that `T_wk15` charged there. ⇒ its **LANE_KEEP column
must reproduce `T_wk15`'s and its TURN column must differ**. That is stronger than a separate
flag-off arm: it proves the lever is inert exactly where it should be *and* active exactly where it
should be, on the trained checkpoint, in one arm. **If both columns differ, or neither does, §C is
void.**

⚠️ **Floor scoping, committed:** `G_gkappa` sets `W_KAPPA = 0` on TURN goals, so it is *less*
constrained than `T_wk15` and its ADE floor is **not** 0.00470 (§B2). Until it has its own seed
replicate its ADE bar is the **unconstrained** neighbour's **0.06070**, and any smaller ADE claim is
reported as **NOT SEPARATED FROM SEED NOISE** with the replicate named as the required next arm.
The **turning** verdict does not depend on this: that floor is 0.00000 on all three pairs.

### C4 ⚠️ A CAUTION THE IMPLEMENTATION ITSELF MEASURED

The first draft of the parity control asserted that lowering `w_kappa` on a TURN-goal window must
change the **plan**. It **FAILED against a correct implementation**: on a window whose canonical
goal seed is already the argmin, dropping the weight changes the **objective** by exactly
`dw * mean(kappa^2)` and changes the **argmin not at all**.
⇒ ⭐ **A null ARGMIN is not a null LEVER** — the mirror image of `M23`(4) (*a NULL arm with a live
optimiser is not a null*). The control now asserts the closed-form cost difference. The real-rig
effect must come from windows where the penalty was making a **different** candidate win, and
`kappa_by_goal_all.txt` says those exist: `wk15` drove TURN_L to 0.02066 against the seed's own
0.08000, so on those windows the seed was **not** winning.

---

## §D — A1 AND A2: QUEUED, PREDICTIONS COMMITTED, NOT YET RUN

Both are pre-registered in `SPEC_CLOSE_THE_GAPS.md` §1–§2 with **both outcomes written in advance**
and are armed in `raw/gap_queue3.py` on the dev box, in priority order.

* **A1 `kammshift` = `kamm07` + `a0_shift`.** Committed point prediction **ADE ≈ 0.886 with
  `turn_left` unchanged at 0.3636**, from the measured `wk15 -> lonshift` transfer of −0.1066.
  ⚠️ The risk is named in advance and is visible in the predecessor: `a0_shift` moves the goal's
  **target speed**, and the goal profile is rolled through the tactical predictor, so it changes
  the lateral ranking too — `wk15 -> lonshift` took **turn_right 0.5000 -> 0.0000**.
* **A2 the `W_KAPPA` micro-sweep {1, 3, 7}**, launched `1` first as the highest-information single
  point. Endpoints already banked at zero cost (`ccos_argmax` W=0, `wk15` W=15.11245).
  ⭐ **Committed:** if `turn_left` is 0.0000 at every `W` that buys ADE ≤ 1.2237, **the penalty is
  retired as a lateral lever** and every `W_KAPPA > 0` row in the register is re-labelled
  *"accurate because it does not turn"*. §B1 makes that the prior: the TURN_L magnitude is already
  at 26 % of command at W = 15.11 and the attenuation is monotone over the three banked points.
  ⚠️ **A2's verdict cannot be written from B1 alone** — B1 shows the endpoint, not the shape, and
  the whole question is whether a usable knee exists between 0 and 15.11.

**GPU state at 01:40** — Thor 3/3 arms (its measured ceiling): the sibling's `T_loncomb` plus my two
A3 arms, ETA ~02:45 Berlin. Dev box 2/2 (6737/8188 MiB): the sibling's `ta_wk15_s0` / `ta_wk15_s1`.
`raw/gap_queue3.py` **defers**: it launches only when a slot is free **and** `ta_queue3.py`'s
launcher is gone, so the two queues can never take the same freed slot and place a third arm on a
card with ~1.4 GB free. A failed probe is never read as "no arms".
