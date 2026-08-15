# The `"mlp"` selector — the CAPACITY CONTROL, implemented and dry-run

**2026-08-16 · branch `agent/arch-inf-20260803` · closes escalation #4 of
`Project Steering/V6F_PLANNER_DESIGN.md`.**
⛔ Nothing was trained. Thor's v6F S-W run was not touched and is unaffected (proof in §3).

---

## 1. Why this had to exist before a `"goal"` arm is judged

`GoalDistanceScorer` selects over the fan with **+267** parameters and a hard-wired
`−‖endpoint − ĝ‖` rule. If that arm beats the incumbent, exactly two stories fit:

1. **MECHANISM** — a candidate-INDEPENDENT reference has no degenerate minimiser, which is why its
   normalised error-rank *falls* with N while a roll-consistency score's *rises* (0.241 → 0.286 at
   N=256, MEASURED on the banked REF-C-XL fan);
2. **CAPACITY** — the selector head was simply underpowered, and any extra parameters on the same
   inputs would have done as well.

A `"goal"`-only experiment cannot separate them. Reading (2) as (1) is the **C6 confound** verbatim —
a decoder compared on its marginal — which this programme has already been burned by once.
`V6F_PLANNER_DESIGN.md` §5.3 pre-registers the refutation: *if `"mlp"` matches or beats `"goal"`,
SEL-1's story is wrong.*

## 2. What landed

| file | change |
|---|---|
| `stack/tanitad/models/v6.py` | **NEW** `MLPCandidateScorer`; `V6Config.selector` accepts `"mlp"`; **NEW** `V6Config.selector_mlp_hidden` (default 256); built last in `__init__` like `"goal"`, so `"none"` still draws no RNG |
| `stack/scripts/train_v6_staged.py` | `--selector` gains `mlp`; **NEW** `--selector-mlp-hidden`; threaded into the config build |
| `stack/tests/test_v6_selector_capacity_control.py` | **NEW** — 8 tests |
| `stack/tests/test_v6_selector.py` | the test asserting `"mlp"` is *refused as unimplemented* is SUPERSEDED — rewritten to assert `"mlp"` is accepted and unknown names are still refused |

**Shape:** `Linear(2 + d_goal_embed → hidden) · GELU · Linear(hidden → 1)` + a per-candidate bias.
The output layer is **zero-init**, mirroring `GoalDistanceScorer`'s discipline, so the control starts
FLAT over the fan and any ranking it acquires is visibly *learned*.

### 2.1 ⚠️ The design doc's `+41,089` was an ESTIMATE and is CORRECTED

| quantity | value | class |
|---|---|---|
| `V6Stack` params, `selector="none"` | **87,893,449** (405 state_dict keys) | MEASURED (ours) |
| `selector="goal"` | **87,893,716** — delta **+267**, 409 keys | MEASURED |
| `selector="mlp"`, hidden 256 | **87,927,250** — delta **+33,801**, 410 keys | MEASURED |
| capacity ratio mlp/goal | **126.6×** | MEASURED |
| `selector="mlp"`, hidden 512 | delta **+67,593** | MEASURED |

`V6F_PLANNER_DESIGN.md` carried **+41,089** in three tables and derived **"154×"** from it. Neither
number corresponds to any realisable shape at `d_goal_embed=128, n_candidates=8` — they were
design-time arithmetic that was never built. **All three tables and escalation #4 are corrected to
the measured values.** *(Evidence class moves INHERITED → MEASURED.)*

### 2.2 The property that makes it a CONTROL and not just another arm

It is **information-MATCHED, not information-enriched**: its inputs are exactly what the goal rule
reads — the candidate ENDPOINT `waypoints[:, :, -1]` and `e_g_tac` — and nothing else. Handing it the
full 60×2 path would silently make it an *information* control, and its result would no longer speak
to capacity at all.

`test_it_is_information_MATCHED_not_information_enriched` proves this by construction: perturbing
**every waypoint except the endpoint by +7 m leaves the score bit-identical**, while perturbing the
endpoint alone moves it.

⚠️ **The control is deliberately generous** — 126.6× the goal rule's parameters. That is the
conservative direction for the conclusion we most want to avoid over-claiming: if a control this
large still loses, "capacity" is a weak explanation; if it wins, SEL-1 is refuted and we want to know.

⛔ **Admissibility (PI 2026-08-03) is inherited unchanged**: the only inputs are the emitted
trajectory and `e_g_tac`, which comes from `goal_head_tac(z_tac_p, cond=e_g_str)`. No
situation-classifier output in any form, no ego state at inference.

⚠️ It emits **no** `goal_point` / `goal_dist`. It has no goal point, and a zero-filled field would be
a fabricated number that later reads as a measurement. It emits `mechanism="mlp"` instead, so a dump
is self-identifying.

## 3. Evidence

| check | result | class |
|---|---|---|
| new tests | **8 passed** (`test_v6_selector_capacity_control.py`) | MEASURED |
| whole v6 set — staged + selector + capacity + ckpt-layout + probe-trunk + revalidation | **130 passed** | MEASURED |
| `"none"` still byte-identical to HEAD | `test_all_off_is_byte_identical_to_head` **passes** ⇒ **the live S-W resume on Thor is unaffected** | MEASURED |
| S-T dry-run, `--selector goal --w-select 1.0` | builds, **X3 `pass=True`** `{planner_to_encoder: 0, tactical_to_below: 0, strategic_to_below: 0}`, 2 synthetic steps OK, `terms: [plan, seam, select, t1]` | MEASURED — `raw/st_dryrun_selector_goal.json` |
| S-T dry-run, `--selector mlp --w-select 1.0` | builds, **X3 `pass=True`** (same three zeros), 2 synthetic steps OK, `planner` group 0.67 M → **0.71 M** | MEASURED — `raw/st_dryrun_selector_mlp.json` |

⚠️ **What the dry-run does NOT show.** The synthetic fan is degenerate — `fan_mean_ade` and
`fan_oracle_ade` agree to 6 decimals — so `sel_norm_err_rank` there (0.0 for goal, 1.0 for mlp at
step 2) is **noise on an undifferentiated fan and carries no information about either selector**. The
dry-run is a CONSTRUCTION and ISOLATION smoke test only. Both arms must be judged on a real fan, at
T1 tier, per family, with the paired episode-cluster bootstrap.

## 4. What is still owed

1. ⛔ **Implemented is not run.** A `"goal"` arm must still not be judged until `"mlp"` has actually
   been *trained beside it* on the same windows. The control's value is the comparison, not its
   existence.
2. ⚠️ **E-WC2 (§5.2) should still refuse-or-fund SEL-1 before S-T launches**, and its stated "0 GPU,
   banked latents" premise is now **STALE** — two independent probes (dev box + Thor) find only
   REF-C latent dumps (`latents_refc-xl-30k.pt`, `latents_refc-base-30k.pt`); **no frozen S-W latents
   exist anywhere**, because the eval pod that held them is gone. E-WC2 now needs a GPU pass at a
   deliberate training pause.
3. ⚠️ `hidden` is a free knob. 256 was chosen as a natural width, not tuned. If the control LOSES, a
   width sweep is the honest follow-up before concluding "mechanism" — a control that loses because
   it was mis-sized is not a control.

## Deliverable manifest

| artifact | where it lives | state |
|---|---|---|
| `stack/tanitad/models/v6.py` (`MLPCandidateScorer` + config) | repo | staged |
| `stack/scripts/train_v6_staged.py` (`--selector mlp`, `--selector-mlp-hidden`) | repo | staged |
| `stack/tests/test_v6_selector_capacity_control.py` (8 tests) | repo | staged |
| `stack/tests/test_v6_selector.py` (superseded test rewritten) | repo | staged |
| `Project Steering/V6F_PLANNER_DESIGN.md` (3 tables + escalation #4 + unresolved-items 3 & 5) | repo | staged |
| this document + 4 raw JSONs | `…/incoming/2026-08-16-selector-capacity-control/` | staged |

## Escalation

⭐ **The `"goal"` vs `"mlp"` comparison must be an ARM PAIR in the S-T plan, not an afterthought.**
Both are now launchable and dry-run-verified; if S-T runs `"goal"` alone the result will be
unattributable and the run cannot be re-used to answer the question afterwards.
