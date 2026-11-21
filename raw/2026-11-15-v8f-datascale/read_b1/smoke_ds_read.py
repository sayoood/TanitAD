"""smoke_ds_read.py -- a CODE CHECK of ds_read.main end to end on STAND-IN inputs (never a result): the STRAT1 dumps / T0 passes and the banked DSCALE evals are copied under the block-1 names, then ds_read runs with K-RE on.
R_s0 is a copy of U_s0's evals, so the reach precondition must read NO-REACH (a built-in check of that path).  Usage: python smoke_ds_read.py SMOKE_DIR   (deletes SMOKE_DIR's contents it created at the end)"""
import os, shutil, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ds_lib as L
S1 = L.S1R
d = sys.argv[1]
shutil.rmtree(d, ignore_errors=True)
t1, m, r = f"{d}/t1", f"{d}/t0mech", f"{d}/t0route"
for p in (t1, m, r):
    os.makedirs(p)
MAP = {("B_s0", "A"): "C0_s0", ("U_s0", "A"): "S1_s0", ("B_rep", "A"): "C0_rep", ("R_s0", "A"): "N1_s0", ("B_s0", "B"): "C0_s0", ("U_s0", "B"): "S1_s0", ("B_rep", "B"): "C0_rep"}
for (arm, st), src in MAP.items():
    shutil.copytree(f"{S1}/t1/dump_STRAT1-{src}_m10_causal_heads", f"{t1}/dump_{L.t1_tag(arm, st)}")
for arm, src in (("B_s0", "C0_s0"), ("U_s0", "S1_s0"), ("B_rep", "C0_rep")):
    for v in ("real", "donor", "zero"):
        shutil.copy(f"{S1}/out/t0/eval_STRAT1-{src}_{v}.npz", f"{r}/eval_DS-{arm}-B_{v}.npz")
SUF = ("", "-tacabs", "-tacorc", "-tacorc-all-latcons", "-tacorc-all-goal")
for name, q in (("DSBs0A", "M2b"), ("DSUs0A", "M2b-df50"), ("DSBrepA", "M2b"), ("DSRs0A", "M2b-df50")):
    for s in SUF:
        for ext in ("npz", "json"):
            src = f"{HERE}/kre/eval_{q}_s0{s}.{ext}"
            if os.path.exists(src):
                shutil.copy(src, f"{m}/eval_{name}_s0{s}.{ext}")
print("fixture built", d)
