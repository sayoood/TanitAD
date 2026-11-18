"""MM 2026-11-18: the G-EVAL identity pair on a REAL NAVRB2 checkpoint (BUILD_NAVRB2.md step 3b, never run before the read).
A = the study wrapper v8f_mini_eval_rb.py on zon.pt (N1R_s0 with e_nav_rb.weight exactly 0.0, nav_rb on, the real channel table);
B = the PINNED v8f_mini_eval.py on zoff.pt (the same weights, e_nav_rb.* dropped, cfg nav_rb off), with the read's T0 argv minus the five nav-rb flags.
Every float / int array of eval_A.npz and eval_B.npz must be bit-equal (navrb_geval_identity.py compare). That is what licenses R - banked contrasts
(R arms read through the wrapper, banked arms through the pinned evaluator). T0 batch = the tool's default (64), never changed."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import navrb2_t0 as T  # noqa: E402
import navrb2_lib as NL  # noqa: E402

OUT = HERE + "/out/geval_pair"
os.makedirs(OUT, exist_ok=True)
GI = NL.TREE + "/stack/scripts/navrb_geval_identity.py"
env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1",
           PYTHONPATH=";".join(os.path.normpath(p) for p in (NL.TREE + "/stack", NL.TREE + "/taniteval")), NAVRB2_TREE=NL.TREE)


def run(argv, log):
    with open(log, "w", encoding="utf-8") as lf:
        r = subprocess.run(argv, env=env, stdout=lf, stderr=subprocess.STDOUT, encoding="utf-8")
    print(os.path.basename(log), "rc", r.returncode, flush=True)
    return r.returncode


if run([T.PY, GI, "pair", f"{NL.CK}/N1R_s0/ckpt_last.pt", OUT], OUT + "/pair.log"):
    sys.exit("ZZ GEVAL PAIR FAIL ZZ pair")
a = T.argv_for("N1R_s0", "real", OUT)
ia = a.index("--ckpt"); a[ia + 1] = OUT + "/zon.pt"
it = a.index("--tag"); a[it + 1] = "GEV-zon"
b = list(a)
b[b.index(NL.TREE + "/stack/scripts/v8f_mini_eval_rb.py")] = NL.TREE + "/stack/scripts/v8f_mini_eval.py"
b[b.index("--ckpt") + 1] = OUT + "/zoff.pt"
b[b.index("--tag") + 1] = "GEV-zoff"
for flag in ("--nav-rb", "--nav-rb-npz", "--nav-rb-sha256", "--nav-rb-variant"):
    i = b.index(flag)
    del b[i:i + 2]
assert not any(x.startswith("--nav-rb") for x in b), b
open(OUT + "/argv_A.txt", "w", encoding="utf-8").write("\n".join(a) + "\n")
open(OUT + "/argv_B.txt", "w", encoding="utf-8").write("\n".join(b) + "\n")
if run(a, OUT + "/run_A.log") or run(b, OUT + "/run_B.log"):
    sys.exit("ZZ GEVAL PAIR FAIL ZZ an evaluator run failed (see run_A.log / run_B.log)")
rc = run([T.PY, GI, "compare", OUT + "/eval_GEV-zon.npz", OUT + "/eval_GEV-zoff.npz"], OUT + "/compare.log")
print(open(OUT + "/compare.log", encoding="utf-8").read()[-600:])
print("ZZ GEVAL PAIR PASS ZZ" if rc == 0 else "ZZ GEVAL PAIR FAIL ZZ compare (see compare.log)")
