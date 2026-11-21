"""n1dw_view_s1slot.py -- NEW wrapper (verdict read, 2026-11-20). Builds a T0 / ticks VIEW for the FROZEN strat1_strategic.py in which ONE arm sits in the S1 slot.

Why (diagnosis, see MM_READING_N1DW_VERDICT.md section 2): the staged n1dw_view.py (eval141/n1dw_view.py:24-28) copies the N1DW passes into the N1 slot
(eval_STRAT1-N1_s{s}_<v>.npz). The frozen reader computes the route-following LABEL only from the S1 slot (strat1_strategic.py:284), the RC-A rows against C0 only
for S1 (:278-280), and the T1 per-tick S_CKPT row only for tags starting with "S1" (:300). With the S1 slot empty (:253-255 "a T0 pass is missing") the label is
NOT-YET by construction, whatever the N1DW passes say. strat1_strategic.py reads no ctx output and no nav_source field (grep), so the S1 slot is admissible for
any text-nav arm. strat1_ctx_read.py is NOT fed: it needs ctx_logits, which N1DW does not emit (no ctx head; strat1_ctx_read.py:6 "C0 and N1 have no head").

Nothing in a real T0 / T1 folder is renamed or modified. The caller deletes the view after the reader ran (re-creatable from the folders).
Usage: python n1dw_view_s1slot.py VIEW_DIR ARM_PREFIX T0_DIR [TICKS_DIR|-]
  ARM_PREFIX e.g. N1DW-N1DW4 (passes eval_<ARM_PREFIX>_s{s}_<v>.npz) or NAVRB2-N1R (reference)."""
import os
import shutil
import sys

VARS = ("real", "donor", "zero")
SEEDS = (0, 1)
BANKED_T0 = "F:/TanitAD-scratch/strat1/read/out/t0"      # navrb2_lib.BANKED_T0 (C0_s0 / C0_s1 / C0_rep passes)
BANKED_T1 = "F:/TanitAD-scratch/strat1/read/t1"          # navrb2_lib.BANKED_DUMPS


def plan(prefix, t0, ticks_dir):
    out = []
    for s in SEEDS:
        for v in VARS:
            out.append((f"{t0}/eval_{prefix}_s{s}_{v}.npz", f"t0/eval_STRAT1-S1_s{s}_{v}.npz"))
        if ticks_dir:
            out.append((f"{ticks_dir}/ticks_{prefix}_s{s}_m10_causal_heads.npz", f"t1/ticks/ticks_STRAT1-S1_s{s}_m10_causal_heads.npz"))
    for arm in ("C0_s0", "C0_s1", "C0_rep"):
        for v in VARS:
            out.append((f"{BANKED_T0}/eval_STRAT1-{arm}_{v}.npz", f"t0/eval_STRAT1-{arm}_{v}.npz"))
        if ticks_dir:
            out.append((f"{BANKED_T1}/ticks/ticks_STRAT1-{arm}_m10_causal_heads.npz", f"t1/ticks/ticks_STRAT1-{arm}_m10_causal_heads.npz"))
    return out


def build(view, prefix, t0, ticks_dir):
    p = plan(prefix, t0, ticks_dir)
    missing = [s for s, _ in p if not os.path.exists(s)]
    if missing:
        raise FileNotFoundError(f"passes missing: {missing[:6]} ({len(missing)} in all)")
    if os.path.exists(view):
        raise FileExistsError(f"{view} exists; the caller removes it first")
    for src, rel in p:
        dst = os.path.join(view, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
    return len(p)


if __name__ == "__main__":
    td = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != "-" else ""
    print("view files:", build(sys.argv[1], sys.argv[2], sys.argv[3], td))
