# M15's L=3 lateral vocabulary — implemented, parity-pinned, and one gap named

**Row:** `D-VOCAB-L3-IMPL` · **Class:** MEASURED (parity + mechanism), **zero GPU**
**Authority:** Master Mind decision **M15** (`Decisions/2026-09-05-mm-decisions.md`)
**Code:** `stack/tanitad/refs/refa_v1.py`, `taniteval/tools/refav1_arm.py`
**Tests:** `stack/tests/test_refa_v1_kappa_levels.py` — **9/9**

---

## §1 What landed

| surface | what it is |
|---|---|
| `GOAL_KAPPA_TURN_LEVELS = (0.01155, 0.01942, 0.05805)` | the three magnitudes, sited at the corpus's own \|κ\| quantiles among real turns — **DERIVED from the road**, copied verbatim from `vocab_design.json`'s `L=3 at corpus quantiles` row, and pinned by a test so a later edit is a deliberate act |
| `canonical_controls_levels(lat, lon, v0, K, dt, levels) -> [L, K, 2]` | the token's profile at every magnitude — the surface M15 named |
| `choose_kappa_level(levels, kappa_hint) -> float` | ⭐ **the chooser, made explicit**: nearest level by magnitude, sign-blind (the token carries direction) |
| `plan(..., goal_kappa_levels=, goal_kappa_hint=)` | threaded through `_imagine_tactical_goal`; `None` is the shipped path |
| `res.goal_kappa_vocab` / `res.goal_kappa_levels` / `goal_action["kappa_turn_used"]` | ⚠️ **THE PARITY STAMP** — per-result and per-window |
| `refav1_arm.py --goal-kappa-levels {m15\|<list>} --goal-kappa-hint gt` | the arm, with a **stale-stack gate** and a **per-window verify-gate** that refuses if `plan()` reports a different vocabulary than the one asked for |
| `manifest["goal_vocab"]` | the dump names its own action space |

## §2 ⚠️ THE PARITY STAMP, and why it is not cosmetic

M16 (1) makes a cross-vocabulary comparison **inadmissible** unless it says so: every refav1
number banked before 2026-09-05 is under the single shipped magnitude. So:

* `goal_kappa_levels=None` is **bit-identical** to the pre-change expression — `test_a_*`, with a
  same-breath control (a non-shipped level set **must** change the goal's control profile) so the
  parity assertion cannot pass vacuously;
* a one-level set at exactly `GOAL_KAPPA_TURN` reproduces the shipped profile **bit-for-bit**, and
  correctly keeps the **same** vocabulary id `L1-0.08` — the id names the ACTION SPACE, not the
  call. A different magnitude, or any wider set, gets a different id;
* the id travels on the result, in `goal_action`, and into the dump manifest, so a reader can tell
  two arms apart from the artifact alone rather than from a run record — the `anchors.pt`
  units lesson (2026-09-04) applied before it costs anything;
* ⛔ a non-`TURN_` token is **invariant** to the level set (asserted for `LANE_KEEP`, `NUDGE_L`,
  `LANE_CHANGE_R`), so the level set cannot act where it was never approved to act.

**Suite:** `pytest -k "refa_v1 or cost_ccos or cost_chord or ego_plan or seed_goal or steer_curv"`
→ **341 passed, 1 skipped**. One pre-existing assertion was updated deliberately:
`test_refa_v1_plan_goal.py::test_e_*` pinned `goal_action`'s key set exactly; it now pins the two
new provenance keys **and their shipped values**, still as an exact set.

## §3 ⛔⛔ THE GAP M15 DOES NOT CLOSE: A VOCABULARY WITH THREE MAGNITUDES NEEDS A CHOOSER, AND THE TRAINED HEAD CANNOT BE ONE

M15's table gives every design an **ORACLE** token chooser. That is the right way to bound a
vocabulary and it is labelled as such. But an *arm* needs a real chooser, and:

* the v7.0 lateral vocabulary has **one** `TURN_L` and **one** `TURN_R` slot;
* `kappa_turn` **does not touch the head's logits** — it is applied after the argmax;
* ⇒ MEASURED by the goal-margin stream on the dense bank: **`turns goaled correctly` is 0.2811
  for EVERY `kappa_turn`** (`ESCALATION.md` §4), and the realised gain from tuning the magnitude
  alone is **2.3 %**, not 3.7×.

⇒ **`goal_kappa_hint` is REQUIRED rather than defaulted, and the refusal names the reason.** A
default would silently choose a tier:

| hint source | tier | what the arm means |
|---|---|---|
| the window's TRUE \|κ\| (`--goal-kappa-hint gt`) | ⛔ **T0** | the M15 oracle bound **realised at plan level** — the measurement nobody has taken, and the cheapest test of whether the vocabulary bound converts into driving at all |
| a κ predictor from vision | T1 | the deployable arm; its own error composes with the table and must be reported beside it |
| the head | ⛔ **does not exist** | one TURN slot per direction |

⭐ **The T0 arm is worth more than the pre-registered A/B, and it is cheaper.** If an
*oracle-chosen* L=3 does not improve the plan, the vocabulary is refuted as the mechanism without
anyone building a chooser first; if it does, the size of the prize is bounded before a chooser is
funded. This is the same logic that made the de-confounded oracle the gate on the goal head.

⚠️ **The alternative realisation — three magnitudes as SEED CANDIDATES, chosen by the planner's
own cost — was considered and NOT built.** It is `RESULT.md` §4's lateral-baseline asymmetry in
vocabulary clothing, and it is not equivalent: under `goal_time_grid="full"` the goal field is
rolled from the canonical controls, so three seeds against **one** goal would be scored against a
goal built at a single magnitude, which is not the design M15 approved. It needs its own
pre-registration, and it should be written beside the `ccos` hold-branch item.

## §4 What this does NOT claim

* ⛔ **No ADE, no four-family number, no driving claim.** Nothing here has been run on a
  checkpoint. The panel that would produce those numbers is **BLOCKED** by the power check
  (`POWER_CHECK.md`) and must be redesigned first.
* ⛔ It does **not** inherit M15's 3.7×. That is an oracle bound on the GOAL's curvature error,
  not on ADE, and the same package already priced the realised magnitude effect at 2.3 %.
* ⚠️ The levels are fitted to **this** corpus's curvature distribution and must be re-derived,
  not copied, for another.
