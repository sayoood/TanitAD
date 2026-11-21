"""n1dh_view.py -- STAGED. Builds the T0 / ticks VIEW (N1 slot, as n1dw_view.py) the frozen STRAT1 readers strat1_strategic.py / strat1_ctx_read.py read for the N1DH arms (card 3). Usage: python n1dh_view.py VIEW_DIR [--no-ticks].
Ticks come from the executor of record (dumps141_hb02).  The S1-slot view is n1dw_view_s1slot.py (landed 94636bf7, unedited).
strat1_strategic.py and strat1_ctx_read.py take arms named S1 / N1 / C0 (eval_STRAT1-<arm>_<variant>.npz; ticks_STRAT1-<arm>_m10_causal_heads.npz) and are frozen, so the N1D passes are presented to them as N1 in a
separate folder: eval_NAVRBD-N1D_s{s}_<v>.npz is COPIED to <view>/t0/eval_STRAT1-N1_s{s}_<v>.npz, the banked C0 passes (C0_s0, C0_s1, C0_rep) beside them.  There is NO S1 arm in NAVRBD: the frozen readers
print S1 as NOT-YET (a missing pass), which is the truth.  Nothing in a real T0 / T1 folder is renamed or modified; the chain deletes the view after the readers ran.
Usage: python navrbd_view.py VIEW_DIR [--no-ticks]"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "F:/TanitAD-scratch/strat1/navrb2/read")
import navrb2_lib as NL  # noqa: E402

VARS = ("real", "donor", "zero")
T0_DEFAULT = os.environ.get("N1DH_T0", HERE + "/out/t0")
T1_DEFAULT = os.environ.get("N1DH_D_141", HERE + "/dumps141_hb02")


def plan(t0=None, t1=None, banked_t0=None, banked_t1=None, ticks=True):
    """-> [(src, dst relative to the view)]; every source must exist (checked by build)"""
    t0, t1, bt0, bt1 = t0 or T0_DEFAULT, t1 or T1_DEFAULT, banked_t0 or NL.BANKED_T0, banked_t1 or NL.BANKED_DUMPS
    out = []
    for s in NL.SEEDS:
        for v in VARS:
            out.append((f"{t0}/eval_N1DH-N1DH_s{s}_{v}.npz", f"t0/eval_STRAT1-N1_s{s}_{v}.npz"))
        if ticks:
            out.append((f"{t1}/ticks/ticks_N1DH-N1DH_s{s}_m10_causal_heads.npz", f"t1/ticks/ticks_STRAT1-N1_s{s}_m10_causal_heads.npz"))
    for arm in ("C0_s0", "C0_s1", "C0_rep"):
        for v in VARS:
            out.append((f"{bt0}/eval_STRAT1-{arm}_{v}.npz", f"t0/eval_STRAT1-{arm}_{v}.npz"))
        if ticks:
            out.append((f"{bt1}/ticks/ticks_STRAT1-{arm}_m10_causal_heads.npz", f"t1/ticks/ticks_STRAT1-{arm}_m10_causal_heads.npz"))
    return out


def build(view, **kw):
    p = plan(**kw)
    missing = [s for s, _ in p if not os.path.exists(s)]
    if missing:
        raise FileNotFoundError(f"passes missing: {missing[:6]} ({len(missing)} in all)")
    for src, rel in p:
        dst = os.path.join(view, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
    return len(p)


if __name__ == "__main__":
    print("view files:", build(sys.argv[1], ticks="--no-ticks" not in sys.argv))
