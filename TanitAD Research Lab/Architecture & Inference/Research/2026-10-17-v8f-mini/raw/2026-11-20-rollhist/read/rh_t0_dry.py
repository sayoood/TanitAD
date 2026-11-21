"""rh_t0_dry.py -- dry run of G-T0 on real data: the T0 E0 eval of UE_s0.A once (CPU), copied as RH_s0, B0_s0, RH_s1, B0_s1 under out/t0dry; the guard must read delta 0.0, HOLDS at both seeds (K-SELF of the guard).
Writes G_T0_DRY.json; deletes out/t0dry (own temp, logged)."""
import json, os, shutil, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rh_common as C
import rh_t0 as T
D = C.OUT + "/t0dry"
os.makedirs(D, exist_ok=True)
t = time.time()
a = T.argv_t0("RH_s0", out=D, ckpt=C.ckpt_of("UEA", 0))
with open(D + "/log.txt", "w", encoding="utf-8") as lf:
    rc = subprocess.run(a, env=C.env(), stdout=lf, stderr=subprocess.STDOUT, encoding="utf-8").returncode
wall = round(time.time() - t, 1)
src = D + "/eval_RH_s0.npz"
assert rc == 0 and os.path.exists(src), f"T0 eval failed rc {rc}"
for n in ("B0_s0", "RH_s1", "B0_s1"):
    shutil.copyfile(src, f"{D}/eval_{n}.npz")
r = T.guard([0, 1], D)
r["dry"] = {"stand_in": "UE_s0.A for all four arms", "eval_wall_s": wall, "pass": bool(r["holds"] is True and all(v["point_m"] == 0.0 for v in r["per_seed"].values()))}
json.dump(r, open(C.HERE + "/G_T0_DRY.json", "w"), indent=1)
print(json.dumps(r["dry"]), r["per_seed"]["0"])
shutil.rmtree(D)
open("F:/TanitAD-scratch/rollhist/DELETIONS.log", "a").write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + f" deleted {D} (G-T0 dry-run copies; result in G_T0_DRY.json)\n")
