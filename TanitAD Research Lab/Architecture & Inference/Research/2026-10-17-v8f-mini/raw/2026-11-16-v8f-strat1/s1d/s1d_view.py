"""s1d_view.py -- builds the T0 view folder the frozen STRATEGIC readers read for S1d.
strat1_strategic.py and strat1_ctx_read.py take the arm named S1 at each seed (eval_STRAT1-S1_s{s}_{real,donor,zero,k4}.npz) and the C0 arms at the same seeds.  They are frozen and unedited, so the S1d
passes are presented to them under the S1 name in a separate folder: eval_STRAT1-S1d_s{s}_<v>.npz is COPIED to <view>/eval_STRAT1-S1_s{s}_<v>.npz, and the C0 passes (C0_s0, C0_s1, C0_s10, C0_s11, C0_rep) are copied
beside them.  Nothing in the real T0 folder is renamed or modified.  The view is temporary: the launcher deletes it after the readers have run (the copies are re-creatable from the T0 folder).
Usage:  python s1d_view.py T0_DIR VIEW_DIR
"""
import os
import shutil
import sys

VARS = ("real", "donor", "zero", "k4")
SEEDS = (0, 1, 10, 11)


def plan(t0):
    """-> [(src, dst_name)] for the S1d passes (renamed to S1) and the C0 passes; every source must exist"""
    out = []
    for s in SEEDS:
        for v in VARS:
            out.append((f"{t0}/eval_STRAT1-S1d_s{s}_{v}.npz", f"eval_STRAT1-S1_s{s}_{v}.npz"))
    for arm in ("C0_s0", "C0_s1", "C0_s10", "C0_s11", "C0_rep"):
        for v in ("real", "donor", "zero"):
            out.append((f"{t0}/eval_STRAT1-{arm}_{v}.npz", f"eval_STRAT1-{arm}_{v}.npz"))
    return out


def build(t0, view):
    missing = [s for s, _ in plan(t0) if not os.path.exists(s)]
    if missing:
        raise FileNotFoundError(f"T0 passes missing: {missing[:6]} ({len(missing)} in all)")
    os.makedirs(view, exist_ok=True)
    for src, name in plan(t0):
        shutil.copyfile(src, os.path.join(view, name))
    return len(plan(t0))


if __name__ == "__main__":
    print("view files:", build(sys.argv[1], sys.argv[2]))
