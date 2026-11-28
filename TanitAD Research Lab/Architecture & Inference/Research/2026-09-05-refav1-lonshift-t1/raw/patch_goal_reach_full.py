"""P4's live next lever: `goal_reach_s` as a CLI knob, threaded end to end.

⛔ APPLY ONLY TO A TREE WHOSE PANEL HAS FINISHED. Editing a tree while arms are
queued puts two different experiments under one name (running processes keep the old
bytes; the next launch picks up the new ones).

WHY. `HANDOFF.md` §4.2: every LON token realises only **0.6513x** its named `dv` inside
the 2 s plan window, because `canonical_controls` drives `v -> v_t` with a time constant
`GOAL_REACH_S = 2.0 s` over a 2.0 s horizon (1 - e^-1 = 0.632). The token's semantics are
never delivered inside the window the planner optimises.

⚠️ NOT A FREE KNOB: `GOAL_REACH_S = TACTICAL_S[0]` (`refa_v1.py:116`) is the vocabulary's
own 2 s tactical band. Overriding it DECOUPLES the goal profile from the band the tokens
are defined on, so an arm using it must say so and must not be presented as "the same
vocabulary". This MEASURES a candidate; it does not authorise a default.

`None` everywhere keeps the module constant, so every pre-existing arm stays
BIT-IDENTICAL -- the discipline `a_sustain` / `a_shift` / `jerk_seam_a0` already follow.

Usage:  python patch_goal_reach_full.py <stack_dir> <taniteval_dir>
"""
import io
import sys
from pathlib import Path

EDITS_REFA = [
    # 1. canonical_controls signature
    ("                       a_sustain: float | None = None,\n"
     "                       a_shift: float | None = None) -> Tensor:",
     "                       a_sustain: float | None = None,\n"
     "                       a_shift: float | None = None,\n"
     "                       goal_reach_s: float | None = None) -> Tensor:"),
    # 2. bind the local at the top of the body
    ("    if a_sustain is not None and a_shift is not None:",
     "    # goal_reach_s -- the vocabulary's reach time as a LEVER (P4). `None` keeps\n"
     "    # the module constant, so every pre-existing arm is BIT-IDENTICAL.\n"
     "    _reach = GOAL_REACH_S if goal_reach_s is None else float(goal_reach_s)\n"
     "    if _reach <= 0.0:\n"
     "        raise ValueError(f\"goal_reach_s must be > 0, got {goal_reach_s!r}\")\n"
     "    if a_sustain is not None and a_shift is not None:"),
    # 3. both GOAL_REACH_S uses inside canonical_controls
    ("        v_t = v_t + float(a_shift) * GOAL_REACH_S",
     "        v_t = v_t + float(a_shift) * _reach"),
    ("            a_i = max(-GOAL_A_MAX, min(GOAL_A_MAX, (v_t - v) / GOAL_REACH_S))",
     "            a_i = max(-GOAL_A_MAX, min(GOAL_A_MAX, (v_t - v) / _reach))"),
    # 4. _imagine_tactical_goal signature
    ("                               a_sustain=None,\n"
     "                               a_shift=None\n"
     "                               ) -> tuple[Tensor, dict]:",
     "                               a_sustain=None,\n"
     "                               a_shift=None,\n"
     "                               goal_reach_s=None\n"
     "                               ) -> tuple[Tensor, dict]:"),
    # 5. its call into canonical_controls
    ("            canonical_controls(lat_v[i], lon_v[j], v, cfg.op_steps, cfg.op_dt,\n"
     "                               kappa_turn=k, a_sustain=s, a_shift=q)",
     "            canonical_controls(lat_v[i], lon_v[j], v, cfg.op_steps, cfg.op_dt,\n"
     "                               kappa_turn=k, a_sustain=s, a_shift=q,\n"
     "                               goal_reach_s=goal_reach_s)"),
    # 6. RefAV1.plan signature
    ("             a_shift=None,\n"
     "             jerk_seam_a0: float | None = None,",
     "             a_shift=None,\n"
     "             goal_reach_s: float | None = None,\n"
     "             jerk_seam_a0: float | None = None,"),
]

# 7. both plan() -> _imagine_tactical_goal call sites (identical text, replace ALL)
CALL_OLD = "                a_sustain=a_sustain, a_shift=a_shift)"
CALL_NEW = ("                a_sustain=a_sustain, a_shift=a_shift,\n"
            "                goal_reach_s=goal_reach_s)")

EDITS_ARM = [
    ('    ap.add_argument("--jerk-seam",',
     '    ap.add_argument("--goal-reach-s", type=float, default=None,\n'
     '                    help="OVERRIDE GOAL_REACH_S (default 2.0 s = TACTICAL_S[0]). "\n'
     '                         "Every LON token realises only 0.6513x its named dv inside "\n'
     '                         "the 2 s plan window because the profile decays with a 2 s "\n'
     '                         "time constant; a shorter reach makes a token deliver its own "\n'
     '                         "semantics inside the optimised window. WARNING: this "\n'
     '                         "DECOUPLES the goal profile from the tactical band the tokens "\n'
     '                         "are defined on -- say so when reporting. Omitting it is "\n'
     '                         "BIT-IDENTICAL to every pre-2026-09-06 arm.")\n'
     '    ap.add_argument("--jerk-seam",'),
    ("                                     a_sustain=a_sustain,\n"
     "                                     a_shift=a_shift,",
     "                                     a_sustain=a_sustain,\n"
     "                                     a_shift=a_shift,\n"
     "                                     goal_reach_s=getattr(a, \"goal_reach_s\", None),"),
]


def apply(path: Path, edits, expect_all=None):
    s = io.open(path, encoding="utf-8", newline="").read()
    orig = s
    # ⛔ THESE FILES ARE CRLF. Anchors are written with "\n" for readability, so every
    # MULTI-LINE anchor must be translated to the file's own newline or it silently
    # matches zero times while the single-line ones still match -- which reads exactly
    # like "the code moved" rather than "my anchor is wrong". MEASURED 2026-09-06.
    nl = "\r\n" if s.count("\r\n") > s.count("\n") // 2 else "\n"

    def fix(t):
        return t.replace("\n", nl) if nl != "\n" else t

    edits = [(fix(o), fix(n_)) for o, n_ in edits]
    if expect_all:
        expect_all = (fix(expect_all[0]), fix(expect_all[1]), expect_all[2])
    for old, new in edits:
        n = s.count(old)
        if n != 1:
            print("REFUSE %s: anchor count %d (want 1) for %r" % (path.name, n, old[:60]))
            return False
        s = s.replace(old, new, 1)
    if expect_all:
        old, new, want = expect_all
        n = s.count(old)
        if n != want:
            print("REFUSE %s: replace-all anchor count %d (want %d)" % (path.name, n, want))
            return False
        s = s.replace(old, new)
    if s == orig:
        print("REFUSE %s: no change" % path.name)
        return False
    io.open(path, "w", encoding="utf-8", newline="").write(s)
    print("PATCHED %s (%d -> %d chars)" % (path.name, len(orig), len(s)))
    return True


def main():
    stack, tev = Path(sys.argv[1]), Path(sys.argv[2])
    refa = stack / "tanitad" / "refs" / "refa_v1.py"
    arm = tev / "tools" / "refav1_arm.py"
    if "goal_reach_s" in io.open(refa, encoding="utf-8").read():
        print("ALREADY PATCHED"); return 0
    ok1 = apply(refa, EDITS_REFA, expect_all=(CALL_OLD, CALL_NEW, 2))
    ok2 = apply(arm, EDITS_ARM)
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    sys.exit(main())
