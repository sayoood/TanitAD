"""Produce every result artifact the SPEC's reporting contract requires, from
whatever arms have landed. Idempotent; an absent arm is NAMED, never silently
dropped. Zero GPU.  ASCII only in print().
"""
import glob
import os
import subprocess
import sys

SP = os.path.dirname(os.path.abspath(__file__))
P4 = "C:/Users/Admin/refav1_margin/p4out"
PY = r"C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
ENV = dict(os.environ)
ENV["PYTHONPATH"] = "C:/Users/Admin/tanitad-wt/stack;C:/Users/Admin/tanitad-wt/taniteval"
ENV["PYTHONIOENCODING"] = "utf-8"
ENV["OMP_NUM_THREADS"] = "4"

ALL = ["ccos_argmax", "ccos_seed1", "wk15", "kamm07", "l3ladder", "wk15_ladder",
       "combined", "combined_seed1", "best", "bestlad"]


def complete(tag):
    """An arm is usable only if its record exists AND its dump has all 8 episodes.
    A short dump is a DIFFERENT PANEL and pairing it compares two window grids."""
    return (os.path.exists("%s/rec_%s.json" % (P4, tag)) and
            len(glob.glob("%s/dump_%s/ep*.npz" % (P4, tag))) == 8)


def run(args, out, env_extra=None):
    e = dict(ENV)
    if env_extra:
        e.update(env_extra)
    with open(out, "wb") as fh:
        r = subprocess.run([PY] + args, stdout=fh, stderr=subprocess.STDOUT, env=e)
    print("  rc=%d  -> %s" % (r.returncode, os.path.basename(out)))
    return r.returncode


def main():
    have = [t for t in ALL if complete(t)]
    absent = [t for t in ALL if t not in have]
    print("ARMS COMPLETE (8/8 episodes): %s" % " ".join(have))
    print("ARMS ABSENT OR PARTIAL      : %s" % (" ".join(absent) or "(none)"))
    print()

    # ---- 1. the second seed-floor table, on the `combined` configuration ----- #
    if complete("combined") and complete("combined_seed1"):
        print("[1] seed floor on the COMBINED configuration (cap + ladder)")
        run([os.path.join(SP, "seed_floor_ext.py"),
             "%s/rec_combined.json" % P4, "%s/rec_combined_seed1.json" % P4,
             "%s/dump_combined" % P4, "%s/dump_combined_seed1" % P4,
             "combined (seed 0) vs combined_seed1 (seed 1) -- cap + ladder"],
            os.path.join(SP, "seed_floor_ext_combined.txt"))
    else:
        print("[1] SKIPPED - combined_seed1 not complete yet")

    # ---- 2. the factorial ---------------------------------------------------- #
    print("[2] the 2x2x2 factorial")
    run([os.path.join(SP, "factorial.py")], os.path.join(SP, "factorial.txt"))

    # ---- 3. four families + feasibility, per new arm ------------------------- #
    for tag in ("combined_seed1", "bestlad", "best"):
        if not complete(tag):
            print("[3] %-15s SKIPPED - not complete" % tag)
            continue
        print("[3] %-15s four families + feasibility" % tag)
        out = os.path.join(SP, "panel_%s.txt" % tag)
        with open(out, "wb") as fh:
            subprocess.run([PY, os.path.join(SP, "four_family_table.py"),
                            "%s/rec_%s.json" % (P4, tag)],
                           stdout=fh, stderr=subprocess.STDOUT, env=ENV)
            fh.write(b"\n# FEASIBILITY (assert_feasible, v0>=2). CONTROL g must be 0.0000.\n")
            e = dict(ENV)
            e["FEAS_VMIN"] = "2"
            subprocess.run([PY, os.path.join(SP, "feas_audit.py"),
                            "%s/dump_%s" % (P4, tag), tag],
                           stdout=fh, stderr=subprocess.STDOUT, env=e)
        print("  -> %s" % os.path.basename(out))

    # ---- 4. the paired deltas, four families, decision grade ----------------- #
    dumps, pairs = [], []
    for t in have:
        dumps += ["--dump", "%s=%s/dump_%s" % (t, P4, t)]
    for a, b in (("combined_seed1", "combined"),   # the seed replicate
                 ("bestlad", "combined"),          # gate A: the curvature weight
                 ("bestlad", "best"),              # the ladder, on top of W_KAPPA+cap
                 ("bestlad", "wk15_ladder"),       # the cap, on top of W_KAPPA+ladder
                 ("best", "kamm07"),               # W_KAPPA, on top of the cap
                 ("bestlad", "ccos_argmax")):      # all three vs the base
        if complete(a) and complete(b):
            pairs += ["--pair", "%s-%s" % (a, b)]
        else:
            print("[4] PAIR SKIPPED %s-%s (an arm is not complete)" % (a, b))
    if pairs:
        print("[4] paired episode-cluster bootstrap on %d pair(s)" % (len(pairs) // 2))
        run([os.path.join(SP, "gm_paired_delta.py")] + dumps + pairs +
            ["--stack", "C:/Users/Admin/tanitad-wt/stack",
             "--taniteval", "C:/Users/Admin/tanitad-wt/taniteval",
             "--out", os.path.join(SP, "pd_final.json"),
             "--md", os.path.join(SP, "pd_final.md")],
            os.path.join(SP, "pd_final.log"))
    else:
        print("[4] SKIPPED - no complete pair yet")
    print("\nANALYZE-DONE")


if __name__ == "__main__":
    main()
