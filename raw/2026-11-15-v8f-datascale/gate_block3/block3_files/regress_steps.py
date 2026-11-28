"""regress_steps.py -- real-record evidence for the block-3 C-STEPS ruling (pod, CPU, read only; writes block3/argv_render, block3/REGRESSION_EVIDENCE.json). Imports the MM-ruled checker from block3/study."""
import hashlib, json, os, sys
D = "/root/dscale10k"; B3 = D + "/block3"
sys.path.insert(0, B3 + "/study")
import dscale10k_check as C
assert os.path.realpath(C.__file__) == os.path.realpath(B3 + "/study/dscale10k_check.py")
os.environ["DS_ROOT"] = D; os.environ["DS_ARGV_OUT"] = B3 + "/argv_render"
exec(open(B3 + "/gen_argv_b3.py").read())
L2 = lambda p: json.load(open(p))
ue = L2(B3 + "/argv_render/UE_s0.A.json")["argv"]; bl = L2(B3 + "/argv_render/BL_s0.A.json")["argv"]
base = L2(D + "/argv/B_s0.A.json")["argv"]
rb = L2(D + "/dry/B_s0.A.dry.json"); ru = L2(D + "/dry/U_s0.A.dry.json")
T = 4 * ru["S_U"]
def setsteps(t, n):
    t = list(t); t[t.index("--steps") + 1] = str(n); return t
res = {"S_U_union": ru["S_U"], "S_B": rb["S"], "ue_steps": T, "checker": B3 + "/study/dscale10k_check.py", "checker_sha256": hashlib.sha256(open(C.__file__, "rb").read()).hexdigest()}
rows = []
def row(name, got, want_red):
    rows.append({"case": name, "problems": got[:2], "REFUSED": bool(got), "expected": "REFUSED" if want_red else "PASS", "ok": bool(got) == want_red})
row("UE_s0 argv diff vs B_s0", C.argv_diff_problems(base, ue, "UE_s0", "A", T), False)
row("BL_s0 argv diff vs B_s0", C.argv_diff_problems(base, bl, "BL_s0", "A", T), False)
row("UE_s0 C-STEPS (union dry record)", C.steps_problems("UE_s0", ue, ru, T, "A", ru), False)
row("BL_s0 C-STEPS (B dry record + union record u_rec)", C.steps_problems("BL_s0", bl, rb, T, "A", ru), False)
row("REGRESSION BL_s0 at 19,164", C.steps_problems("BL_s0", setsteps(bl, 19164), rb, T, "A", ru), True)
row("REGRESSION BL_s0 at 61,569", C.steps_problems("BL_s0", setsteps(bl, T + 1), rb, T, "A", ru), True)
row("REGRESSION UE_s0 at 61,569", C.steps_problems("UE_s0", setsteps(ue, T + 1), ru, T, "A", ru), True)
row("REGRESSION BL_s0, target from the B record (no u_rec)", C.steps_problems("BL_s0", bl, rb, T, "A"), True)
row("REGRESSION BL_s0, the B record passed as u_rec (19,164 target)", C.steps_problems("BL_s0", setsteps(bl, 19164), rb, 19164, "A", rb), True)
row("REGRESSION BL_s0, evidence ue_steps 19,164 vs 61,568", C.steps_problems("BL_s0", bl, rb, 19164, "A", ru), True)
row("REGRESSION BL_s0 argv diff pinned (--steps 61,569 vs ue_steps)", C.argv_diff_problems(base, setsteps(bl, T + 1), "BL_s0", "A", T), True)
row("REGRESSION UE_s0 without --steps", C.argv_diff_problems(base, [t for i, t in enumerate(ue) if not (t == "--steps" or (i and ue[i - 1] == "--steps"))], "UE_s0", "A", T), True)
res["rows"] = rows; res["pass"] = all(r["ok"] for r in rows)
res["argv_file_sha256"] = {n: hashlib.sha256(open(B3 + f"/argv_render/{n}.json", "rb").read()).hexdigest() for n in ("UE_s0.A", "BL_s0.A")}
res["argv_sha256_checker"] = {"UE_s0.A": C.argv_sha(ue), "BL_s0.A": C.argv_sha(bl)}
json.dump(res, open(B3 + "/REGRESSION_EVIDENCE.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "rows"})); [print(("OK  " if r["ok"] else "BAD ") + r["expected"] + " " + r["case"]) for r in rows]
