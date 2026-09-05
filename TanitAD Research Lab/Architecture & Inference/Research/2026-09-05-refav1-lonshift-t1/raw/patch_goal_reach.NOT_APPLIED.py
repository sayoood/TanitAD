"""P4 NEXT LEVER, PREPARED BUT NOT APPLIED — `goal_reach_s` as a CLI lever.

⛔⛔ THIS PATCH IS DELIBERATELY NOT APPLIED IN THIS TURN, AND THAT IS THE POINT.
Both live code trees are in use by RUNNING panels:
  * `/home/nvidia/refav1_lon/code/` — my own Thor queue's batches 2 and 3 have NOT
    launched yet, and they must run on the SAME bytes as batch 1 or the panel stops
    being window-for-window comparable with its own baseline;
  * `C:/Users/Admin/tanitad-wt/` — the sibling stream's `queueLON2.sh` launches its
    remaining arms (`lonshift_s1`, `loncomb2`, `lonvocab`) from that tree.
Editing either mid-flight is the `supervise_run.sh` trap in a source-code costume:
already-imported processes keep the old bytes while the NEXT launch silently picks up
new ones, so the panel would contain two different experiments under one name.
⇒ APPLY ONLY once both queues have drained, then re-run a DEFAULT-PATH control arm and
require it to reproduce `T_wk15` before quoting any `goal_reach_s` arm.

WHY THIS LEVER. `HANDOFF.md` §4.2: every LON token realises only **0.6513x** its named
`dv` inside the 2 s plan window, because `canonical_controls` drives `v -> v_t` with a
time constant of `GOAL_REACH_S = 2.0 s` over a 2.0 s horizon (1 - e^-1 = 0.632). The
token's own semantics are therefore never delivered inside the window the planner
optimises. Shortening the reach time makes a token mean what it says WITHIN the horizon.
It is one variable and it is cheap.

⚠️ IT IS NOT A FREE KNOB. `GOAL_REACH_S = TACTICAL_S[0]` (`refa_v1.py:116`) — it is the
vocabulary's own 2 s tactical band, not an arbitrary constant. Changing it DECOUPLES the
goal profile from the band the tokens are defined on, so an arm using it must say so and
must not be presented as the same vocabulary. That is a design change, and per the
package's own rule it MEASURES a candidate, it does not authorise a default.

WHAT IT DOES. Adds `goal_reach_s: float | None = None` to `canonical_controls` and
threads it through `canonical_controls_batch` / `RefAV1.plan`, plus `--goal-reach-s` on
`refav1_arm.py`. `None` keeps the module constant, so every existing arm is
BIT-IDENTICAL -- the same discipline `a_sustain` / `a_shift` / `jerk_seam_a0` already
follow (all OFF by default, every default bit-identical to the pre-2026-09-05 planner).

THE TWO CALL SITES (verified by reading, `refa_v1.py`):
  :499   v_t = v_t + float(a_shift) * GOAL_REACH_S          # a_shift's own reach
  :539   a_i = max(-GOAL_A_MAX, min(GOAL_A_MAX, (v_t - v) / GOAL_REACH_S))
⚠️ BOTH must take the same value or `a_shift` and the decay disagree about the horizon.

CONTROLS THIS ARM MUST CARRY (pre-registered here, before any number exists):
  1. `--goal-reach-s 2.0` MUST reproduce the unpatched arm bit-for-bit (default path).
  2. A deliberate REGRESSION arm (`--goal-reach-s 8.0`, i.e. a LONGER reach) must be
     WORSE on LON speed, or the lever is not doing what this docstring claims.
  3. The emitted diagnostic (`lon_emitted.py`) must show mean |a| moving; a null with an
     unmoved mean |a| means the goal changed and the SEARCH did not follow -- the `ccosh`
     outcome, not a vocabulary finding.
  4. Its own inference-seed replicate, because a separated CI is necessary and not
     sufficient (`H-ESTIM-SEED-1`).

Usage (AFTER the queues drain):  python patch_goal_reach.py <path-to-stack> <path-to-taniteval>
"""
import re
import sys
from pathlib import Path


def patch_refa(p: Path) -> bool:
    s = p.read_text(encoding="utf-8")
    if "goal_reach_s" in s:
        print("ALREADY PATCHED", p)
        return False
    n = 0

    # 1. signature of canonical_controls
    old_sig = ("                       a_sustain: float | None = None,\n"
               "                       a_shift: float | None = None) -> Tensor:")
    new_sig = ("                       a_sustain: float | None = None,\n"
               "                       a_shift: float | None = None,\n"
               "                       goal_reach_s: float | None = None) -> Tensor:")
    if old_sig in s:
        s = s.replace(old_sig, new_sig, 1); n += 1
    else:
        print("MISS: canonical_controls signature"); return False

    # 2. bind the local, immediately after the signature's body starts
    anchor = "    if a_sustain is not None and a_shift is not None:"
    bind = ("    # ⭐ `goal_reach_s` — the vocabulary's reach time as a LEVER (P4).\n"
            "    # `None` keeps the module constant, so every pre-existing arm is\n"
            "    # BIT-IDENTICAL. ⚠️ GOAL_REACH_S is TACTICAL_S[0]; overriding it\n"
            "    # decouples the goal profile from the band the tokens are defined on.\n"
            "    _reach = GOAL_REACH_S if goal_reach_s is None else float(goal_reach_s)\n"
            "    if _reach <= 0.0:\n"
            "        raise ValueError(f\"goal_reach_s must be > 0, got {goal_reach_s!r}\")\n")
    if anchor in s:
        s = s.replace(anchor, bind + anchor, 1); n += 1
    else:
        print("MISS: canonical_controls body anchor"); return False

    # 3. both GOAL_REACH_S call sites inside canonical_controls
    a = "        v_t = v_t + float(a_shift) * GOAL_REACH_S"
    b = "            a_i = max(-GOAL_A_MAX, min(GOAL_A_MAX, (v_t - v) / GOAL_REACH_S))"
    for old, new in ((a, a.replace("GOAL_REACH_S", "_reach")),
                     (b, b.replace("GOAL_REACH_S", "_reach"))):
        if old in s:
            s = s.replace(old, new, 1); n += 1
        else:
            print("MISS call site:", old.strip()); return False

    p.write_text(s, encoding="utf-8", newline="\n")
    print(f"PATCHED {p}  ({n} edits)")
    return True


if __name__ == "__main__":
    stack = Path(sys.argv[1])
    ok = patch_refa(stack / "tanitad" / "refs" / "refa_v1.py")
    print("NOTE: canonical_controls_batch / RefAV1.plan / refav1_arm.py threading is "
          "NOT done here on purpose -- do it in the same edit session, with the "
          "default-path control run FIRST.")
    sys.exit(0 if ok else 1)
