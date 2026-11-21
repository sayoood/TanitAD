"""Insert the RL-generator-collisions rows into GOALS_AND_CLAIMS.md.

Content-asserted and idempotent: refuses if the anchor row is absent, refuses to
double-apply, and verifies every new id is present after the write.
"""
import io
import os
import sys
import time

G = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
F = os.path.join(G, "Project Steering", "GOALS_AND_CLAIMS.md")

PKG = "TanitAD Research Lab/Deployment & Optimization/Research/2026-09-05-rl-generator-collisions"

ROWS = [
    ("D-RL-GEN-COLLIDES-1",
     "\u2b50\u2b50 **THE PI IS RIGHT AND THE NUMBER IS BIGGER THAN THE BRIEF SAID: refcv3's GENERATOR "
     "EMITS COLLIDING TRAJECTORIES, AND `sel_contact = 0` MEASURES THE SELECTOR, NOT THE MODEL.** "
     "MEASURED 2026-09-05, 0 GPU, on the banked fan (240 windows x 128 candidates) re-scored with the "
     "CURRENT swept `rewards._collision`: **`fan_contact` = 0.034277 (1,053 / 30,720)** over all "
     "windows and **0.126562** over the 65 lead-bearing ones; **19 / 240 windows (7.92 %) carry at "
     "least one collider** and the **worst carries 104 of 128 (81.2 %)**. The selected path is clean "
     "(`sel_contact` **0.0000**, a STRUCTURAL zero) purely because the selector filters the garbage "
     "\u2014 a model that depends on a downstream filter is not the model to ship. \u21d2 the RL target "
     "is the GENERATOR (`fan_contact`), never `sel_contact`",
     "**SUPPORTED (MEASURED)**",
     "`%s/raw/p2_feasible_vs_contact.json`; `%s/SPEC.md` \u00a70" % (PKG, PKG)),

    ("D-RL-CONTACT-DEFN-1",
     "\u26d4 **THE BANKED COLLISION FLAG USES A SUPERSEDED DEFINITION, AND EVERY PRE-2026-09-05 "
     "`contact` NUMBER IS 37.8 % LOW.** `rewards._collision` is SWEPT in the relative frame (a segment "
     "crossing the 2 m disc counts even when neither endpoint is inside); `fan_bank_base_240w.npz`'s "
     "`f_contact` predates it. **POSITIVELY IDENTIFIED, not inferred**: the per-step POINT test "
     "reproduces the banked flag with **0 disagreements over 30,720 candidates**, while every "
     "structural flag (`kamm_over`, `envelope`, `off_reach`, `infeasible`) reproduces exactly under "
     "the current scorer and only `contact` differs, one-sidedly (289 extra, 0 fewer). Rate "
     "**0.024870 (POINT) \u2192 0.034277 (SWEPT)**. \u2b50 Corroborated independently: `_swept_hit` "
     "landed in commit `9765634`, which `\u2026/veto-only-fan-safety/raw/chain_suite.sh` records as "
     "arriving AFTER the s0/s1 arms were frozen. \u21d2 **state the collision definition beside any "
     "`contact` number, or it is not quotable** \u2014 same family as the `control_units` trap",
     "**SUPPORTED (MEASURED, two independent probes)**",
     "`%s/raw/p2_feasible_vs_contact.json` control C1b; `%s/SPEC.md` \u00a70" % (PKG, PKG)),

    ("D-RL-TOP32-NOMOVE-1",
     "\u26d4 **RETRACTION OF A MOTIVATING NUMBER: NO RL ARM HAS EVER MOVED THE GENERATOR'S COLLISION "
     "RATE.** The Master Mind's brief stated *\"the veto arm shifted `top32_contact` 0.03299 \u2192 "
     "0.02691 = \u22120.00729 \u2026 the one gain\"*. MEASURED in the banked verdicts: `top32_contact` "
     "reads base **0.03298611 \u2192 after_s0 0.03298611 \u2192 after_s1 0.03298611**, i.e. **delta "
     "EXACTLY 0.0 with CI [0, 0], at BOTH seeds and BOTH doses (200 and 2,000 steps)**; `fan_contact` "
     "moved the WRONG way (+0.000391 / +0.001693, neither separated). The strings `0.02691`, `0.0269` "
     "and `0.00729` appear NOWHERE in that package's `RESULT.md` (searched in bare AND comma-grouped "
     "form). `0.0269` occurs once in `raw/fan_rerank_veto200s0.json`, whose `*__contact` entries are "
     "**all 0.0** for every selection rule \u2014 selected-path metrics, so that file cannot be the "
     "source of a *fan* number. \u21d2 ROOT-CAUSE CLASS: **a number quoted without its arm and its "
     "artifact path**. The PI's underlying point is untouched and strengthened by "
     "`D-RL-GEN-COLLIDES-1`",
     "**REFUTED (the gain does not exist in any banked artifact)**",
     "`\u2026/2026-09-05-veto-only-fan-safety/raw/veto_verdict_veto200.json`, `\u2026/veto_verdict_veto2k.json`; `%s/SPEC.md` \u00a70" % PKG),

    ("D-RL-FEASDECODE-CONTACT-1",
     "\u2b50 **THE FEASIBLE DECODE FIXES FEASIBILITY COMPLETELY AND DOES NOT FIX COLLISIONS \u2014 IT "
     "MILDLY WORSENS THEM.** MEASURED 2026-09-05, 0 GPU, `project_feasible` defaults, BOTH sides "
     "re-scored with the current swept scorer, paired episode-cluster bootstrap: `fan_kamm_over` "
     "**0.840625 \u2192 0.000000**, `fan_envelope` **0.887728 \u2192 0.000000**, `fan_infeasible` "
     "0.891960 \u2192 0.333301 (residual is `off_reach`, untargeted at `clamp_entry=False`), "
     "`fan_unsafe` 0.110319 \u2192 0.098210 (better, separated) \u2014 but **`fan_contact` 0.034277 "
     "\u2192 0.036230 (ns) and `top32_contact` 0.008984 \u2192 0.013672, +0.004687 [+0.001260, "
     "+0.009333], SEPARATED WORSE**. Mechanism: infeasible candidates were flying into unreachable "
     "geometry, often AWAY from the lead; making them realizable pulls them back into the reachable "
     "set, which is where the lead is. Controls: C1a structural flags reproduce the bank exactly, C1b "
     "bank definition identified, C2 `enabled=False` bit-identical (0.000e+00), C3 round-trip "
     "2.13e-14, C4 89.29 % of candidates moved. \u21d2 **\"run the RL collision stage on top of the "
     "projection\" cannot by itself be the fix**; the projection is a clean SUBSTRATE (it removes the "
     "feasibility confound) but collisions are ORTHOGONAL and must be attacked directly",
     "**SUPPORTED (MEASURED, 4 controls pass)**",
     "`%s/raw/p2_feasible_vs_contact.json`; `%s/SPEC.md` \u00a74" % (PKG, PKG)),

    ("D-RL-PROGRESS-COMPOSED-1",
     "\u2b50 **`progress` BINDS IN ISOLATION AND NOT IN THE COMPOSED REWARD \u2014 SO REPAIRING IT IS "
     "NOT THE COLLISION LEVER.** MEASURED 2026-09-05, 0 GPU, tie-safe AUC = P(score of a collider > "
     "score of a non-collider), ties 0.5: **`progress` 0.8947** [0.8514, 0.9354] unprojected / "
     "**0.7206** projected \u2014 it ranks a COLLIDER above a non-collider ~89 % of the time. **But "
     "the composed `DEFAULT_WEIGHTS` reward reads AUC = 0.0000 EXACTLY** (a structural zero: with "
     "`collision` at 1.00 no colliding candidate ever outscores a non-colliding one), and **deleting "
     "`progress` changes that by exactly nothing** (0.0000 either way). `headway` 0.1406 and `comfort` "
     "0.0261 already disprefer colliders. \u26a0 A first implementation using `argsort(argsort(\u00b7))` "
     "ordinal ranks **FAILED both controls** (constant read +0.6464 instead of 0; oracle read +0.0810 "
     "instead of \u22121) because the flag is ~97 % ties and `np.argsort` is stable \u2014 it would "
     "have shipped `progress` at +0.49 from a broken probe. Both controls PASS exactly after the fix. "
     "\u21d2 the reward's RANKING is already correct; what is missing is SIGNAL DENSITY "
     "(`D-RL-COLL-SPARSE-1`)",
     "**SUPPORTED (MEASURED, K1/K2 controls pass exactly)**",
     "`%s/raw/p3_progress_binds.json`; `%s/SPEC.md` \u00a73" % (PKG, PKG)),

    ("D-RL-COLL-SPARSE-1",
     "\u2b50\u2b50 **WHY NO RL ARM HAS MOVED THE COLLISION RATE: THE GRPO ADVANTAGE IS IDENTICALLY "
     "ZERO IN 92 % OF WINDOWS.** The advantage is group-relative ACROSS one window's candidates, so a "
     "window in which every candidate collides, or none does, contributes **exactly zero** collision "
     "gradient. MEASURED (eval side, 240 windows): only **19 / 240 = 7.92 %** contain BOTH a colliding "
     "and a non-colliding candidate; **221 / 240 = 92.08 %** contribute nothing; only **14 / 121 "
     "episodes** carry any such window; the worst 5 windows hold **54.7 %** of all colliders. "
     "\u2b50 CORROBORATED BY A SECOND, INDEPENDENT PROBE (train side, a different mechanism \u2014 the "
     "`ls-tree` lesson is that repeated samples through ONE channel are one sample): "
     "`arm_summary.json::counters.components_fired` reads **`collision = 41`** against **`comfort` "
     "200, `feasibility` 200, `progress` 200** over 200 optimizer steps \u2014 **20.5 % vs 100 %**. "
     "\u21d2 the collision channel is SPARSE and every other term is DENSE; the pre-registered "
     "successor is a **collider curriculum**, not a bigger weight",
     "**SUPPORTED (MEASURED, two independent probes)**",
     "`%s/SPEC.md` \u00a72; `%s/raw/p2_feasible_vs_contact.json`; `\u2026/veto_run/run/s0/ctrl_null/arm_summary.json`" % (PKG, PKG)),

    ("H-RL-COLL-1",
     "\u2b50 **PRE-REGISTERED, both outcomes committed BEFORE any arm ran: PUNISHING COLLISIONS "
     "DIRECTLY (`collision` weight 1.0, EVERY other weight 0.0) REDUCES THE GENERATOR'S COLLISION "
     "RATE `fan_contact`, and the gain CLEARS BOTH FLOORS.** \u26d4 No arm in the panel had ever run a "
     "collision-punishing reward in isolation \u2014 `rl` carries collision at 1.00 but CONFOUNDED "
     "with progress 0.30 / headway 0.30 / feasibility 0.50 / comfort 0.20, and every `ctrl_*`/`veto*` "
     "arm has EVERY weight 0.0. \u2b50 **`one_variable` is exact**: the new `coll200` differs from the "
     "already-banked `ctrl_null` in **the collision weight alone, 0.0 \u2192 1.0** \u2014 same "
     "`w_anchor` 1.0, lr 1e-5, 200 steps, `two_scalar`, `use_gt_bar=False`, `veto_enabled=False` "
     "(asserted mechanically by `raw/patch_add_coll_arms.py`, ONE_VARIABLE=PASS). Arms: `coll200` "
     "seeds 0+1 (the replicate floor), `ctrl0` (lr = 0, the ONLY admissible zero-lever floor), "
     "`ctrl_null` s1 (zero-information). **BLOCKING P1 CHECK PASSED**: `fan_contact` base **0.0974392 "
     "(449/4,608, n = 36 lead windows)** against a **0.0024740** replicate floor = **39.4x**, so a win "
     "is reportable; `top8_contact`, `sel_contact` (both 0.0) and `mass_rank_contact` (3.47e-05) are "
     "**UNDETECTABLE-DOWNWARD** and inadmissible as primaries. **SUCCESS** = `fan_contact` decreases, "
     "separated at BOTH seeds, same sign, smaller \\|d\\| above BOTH floors, `ctrl_null` not drifting "
     "the same way, AND T1 `ade_m` not regressed beyond the replicate floor. **FAILURE** = anything "
     "else, with `D-RL-COLL-SPARSE-1` as the PRE-REGISTERED explanation and a **collider curriculum** "
     "as the named successor",
     "**OPEN \u2014 pre-registered 2026-09-05 before any arm produced a number; arms running on the dev-box 4060**",
     "`%s/SPEC.md` \u00a75-\u00a76; `%s/raw/run_coll_arms.sh`, `%s/raw/patch_add_coll_arms.py`" % (PKG, PKG, PKG)),
]

ANCHOR_ID = "| H-VETO-FAN-1 |"

src = None
for _ in range(12):
    try:
        with io.open(F, encoding="utf-8") as fh:
            src = fh.read()
        break
    except OSError:
        time.sleep(3)
if src is None:
    raise SystemExit("INCONCLUSIVE: could not read GOALS_AND_CLAIMS.md (mount)")

if "D-RL-GEN-COLLIDES-1" in src:
    print("ALREADY APPLIED -- rows present, not double-applying")
    sys.exit(0)

i = src.find(ANCHOR_ID)
if i < 0:
    raise SystemExit("REFUSED: anchor row %r not found" % ANCHOR_ID)
j = src.find("\n", i)
if j < 0:
    raise SystemExit("REFUSED: anchor row has no newline terminator")

block = "".join("\n| %s | %s | %s | %s |" % r for r in ROWS)
out = src[:j] + block + src[j:]

for _ in range(12):
    try:
        with io.open(F, "w", encoding="utf-8", newline="") as fh:
            fh.write(out)
        break
    except OSError:
        time.sleep(3)

# ---- verify by CONTENT, per id ----
with io.open(F, encoding="utf-8") as fh:
    back = fh.read()
ok = True
for rid, _, _, _ in ROWS:
    n = back.count("| %s |" % rid)
    print("  %-28s occurrences=%d %s" % (rid, n, "OK" if n == 1 else "FAIL"))
    ok = ok and n == 1
# same-breath control: a pre-existing id must still read exactly once
ctrl = back.count("| H-VETO-FAN-1 |")
print("  %-28s occurrences=%d  <- CONTROL, must be 1 (0 means the file could not be READ)"
      % ("H-VETO-FAN-1", ctrl))
print("bytes %d -> %d (+%d)" % (len(src), len(back), len(back) - len(src)))
print("INSERT=%s" % ("PASS" if (ok and ctrl == 1) else "FAIL"))
if not (ok and ctrl == 1):
    raise SystemExit(1)
