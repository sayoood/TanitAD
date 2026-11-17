"""gen_argv_b3.py -- DSCALE10K block 3 (pod): writes argv/UE_s0.A.json and argv/BL_s0.A.json (card 2.3 rows 7 and 8; MM_RULING_BLOCK3_STEPS.md).
UE_s0 = U_s0.A with --steps = 4 x S_U (S_U read from dry/U_s0.A.dry.json, MEASURED 15,392 -> 61,568), --steps-mult 4 and --fit-join kept (the checker requires --steps on UE), --out runs/UE_s0/A.
BL_s0 = B_s0.A + --steps (the SAME total, read from the same union dry record), --out runs/BL_s0/A. Seed 0. DS_ROOT / DS_ARGV_OUT override the root and the output folder (default $DS_ROOT/argv). Idempotent; prints the argvs."""
import json
import os

D = os.environ.get("DS_ROOT", "/root/dscale10k")
OUT = os.environ.get("DS_ARGV_OUT", D + "/argv")
L = lambda n: json.load(open(f"{D}/argv/{n}.json"))["argv"]


def setf(t, f, vals):
    t = list(t)
    i = t.index(f)
    j = i + 1
    while j < len(t) and not (t[j].startswith("--") and len(t[j]) > 2):
        j += 1
    return t[:i + 1] + list(vals) + t[j:]


s_u = int(json.load(open(f"{D}/dry/U_s0.A.dry.json"))["S_U"])
ue = 4 * s_u
a_ue = setf(setf(L("U_s0.A"), "--steps", [str(ue)]), "--out", [f"{D}/runs/UE_s0/A"])
a_bl = setf(L("B_s0.A") + ["--steps", str(ue)], "--out", [f"{D}/runs/BL_s0/A"])
os.makedirs(OUT, exist_ok=True)
for n, t in (("UE_s0.A", a_ue), ("BL_s0.A", a_bl)):
    json.dump({"argv": t}, open(f"{OUT}/{n}.json", "w"), indent=1)
    print(n, " ".join(t))
    print()
print("ue_steps", ue)
