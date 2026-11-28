"""navrb2_view.py -- builds the T0 / ticks VIEW the frozen STRAT1 STRATEGIC readers read for the R arms.
strat1_strategic.py and strat1_ctx_read.py take the arms named S1 / N1 / C0 at seeds 0 and 1 (eval_STRAT1-<arm>_<variant>.npz; ticks_STRAT1-<arm>_m10_causal_heads.npz) and are frozen and unedited, so the R passes are
presented to them under those names in a separate folder: eval_NAVRB2-S1R_s{s}_<v>.npz is COPIED to <view>/t0/eval_STRAT1-S1_s{s}_<v>.npz (N1R likewise), the banked C0 passes (C0_s0, C0_s1, C0_rep) are copied beside them.
Nothing in a real T0 / T1 folder is renamed or modified; the launcher deletes the view after the readers ran (re-creatable from the folders).
Usage: python navrb2_view.py VIEW_DIR [--no-ticks]"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import navrb2_lib as NL  # noqa: E402

VARS = ("real", "donor", "zero")


def plan(t0=None, t1=None, banked_t0=None, banked_t1=None, ticks=True):
    """-> [(src, dst relative to the view)]; every source must exist (checked by build)"""
    t0, t1, bt0, bt1 = t0 or NL.T0, t1 or NL.T1, banked_t0 or NL.BANKED_T0, banked_t1 or NL.BANKED_DUMPS
    out = []
    for s in NL.SEEDS:
        for fam, name in (("S1R", "S1"), ("N1R", "N1")):
            for v in VARS + (("k4",) if fam == "S1R" else ()):
                out.append((f"{t0}/eval_NAVRB2-{fam}_s{s}_{v}.npz", f"t0/eval_STRAT1-{name}_s{s}_{v}.npz"))
            if ticks:
                out.append((f"{t1}/ticks/ticks_NAVRB2-{fam}_s{s}_m10_causal_heads.npz", f"t1/ticks/ticks_STRAT1-{name}_s{s}_m10_causal_heads.npz"))
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
