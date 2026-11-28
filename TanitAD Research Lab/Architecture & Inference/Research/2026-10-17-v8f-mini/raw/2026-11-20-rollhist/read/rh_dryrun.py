"""rh_dryrun.py -- the whole reader pipeline on REAL dump files with UE_s0.A standing in for every arm (no rollout beyond c-EXEC): builds out/dry/t1 from c-EXEC's tier-of-record dump (B), runs the frozen headway and
family readers and rh_strat.py exactly as rh_read_chain.sh does, then rh_read.py.  Controls on real data:
  K-SELF  (all 24 runs are copies of ONE rollout): every pair of every cell reads D = 0.0 exactly, every family NOT-SEPARATED, the outcome ROLLHIST-NO-EFFECT.
  K-PLANT (RH_R0C10XHb20 of both seeds gets its `cl` tracks ROTATED by 3 deg about the track's own start point, the origin that T1 readers prepend): step lengths are unchanged, so
          speed_mae_mps, yaw_rate_mae_degps and curvature_mae_1pm must read exactly 0 (tol 1e-9); heading_mae_deg, cross_mae_m and ade_tac_m must move (WORSE, separated) in P1 and in P4.
Usage: python rh_dryrun.py [build|read|plant|all]   Writes out/read_dry/DRY_RESULTS.json.  Deletes out/dry at the end of `all` (own temp, logged)."""
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
import rh_common as C  # noqa: E402

SRC = C.OUT + "/cexec/B_record_hb_rzg"
SRC_TAG = "cexec_UE_s0A_R0C10XHb20_m10_causal_heads"
DRYBASE = "C:/Users/Admin/rhdry"          # NTFS: hard links (F: is exFAT, no links, 24 copies of the dump took > 2 min each)
DRY = DRYBASE + "/dry"
RD = DRYBASE + "/read_dry"
V = "F:/TanitAD-scratch/v8proof"
ROT_DEG = 3.0


def build(plant=False):
    base = DRYBASE + "/base"
    if not os.path.exists(base + "/COPIED"):
        os.makedirs(base + "/dump", exist_ok=True)
        os.makedirs(base + "/ticks", exist_ok=True)
        for x in glob.glob(f"{SRC}/dump_{SRC_TAG}/ep*.npz"):
            shutil.copyfile(x, f"{base}/dump/{os.path.basename(x)}")
        shutil.copyfile(glob.glob(f"{SRC}/ticks/ticks_*.npz")[0], base + "/ticks/ticks.npz")
        open(base + "/COPIED", "w").write("one copy of the c-EXEC tier-of-record rollout (UE_s0.A, R0C10XHb20)")
    for kind in C.KINDS:
        for f in C.FORM_ORDER:
            for s in C.SEEDS:
                out = C.folder(kind, f, s, w=DRY)
                if os.path.exists(out + "/COPIED"):
                    continue
                tg = C.tag(kind, f, s)
                name = f"I10{C.run_name(kind, f)}_s{s}"
                os.makedirs(f"{out}/dump_{tg}", exist_ok=True)
                os.makedirs(out + "/ticks", exist_ok=True)
                for x in glob.glob(base + "/dump/ep*.npz"):
                    os.link(x, f"{out}/dump_{tg}/{os.path.basename(x)}")
                os.link(base + "/ticks/ticks.npz", f"{out}/ticks/ticks_{name}_m10_causal_heads.npz")
                open(out + "/COPIED", "w").write("hard links to the one copy of the c-EXEC tier-of-record rollout")


def plant():
    th = np.deg2rad(ROT_DEG)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    for s in C.SEEDS:
        out = C.folder("RH", "R0C10XHb20", s, w=DRY)
        tg = C.tag("RH", "R0C10XHb20", s)
        for x in glob.glob(f"{out}/dump_{tg}/ep*.npz"):
            with np.load(x, allow_pickle=True) as z:
                d = {k: z[k] for k in z.files}
            d["cl"] = np.einsum("ij,...j->...i", R, d["cl"].astype(np.float64)).astype(d["cl"].dtype)
            np.savez(x + ".new.npz", **d)
            os.replace(x + ".new.npz", x)        # a new file replaces the hard link (the other 23 runs keep the original)
        open(out + "/PLANTED", "w").write(f"cl rotated {ROT_DEG} deg about the origin\n")


def readers(seeds="0,1"):
    if os.path.isdir(RD):
        shutil.rmtree(RD)                   # own temp from the previous stage of this script (literal path RD under out/)
    os.makedirs(RD + "/hw", exist_ok=True)
    os.makedirs(RD + "/q3", exist_ok=True)
    pairs = ",".join(f"I10{a}:I10{b}" for a, b in C.PAIRS.values())
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", HEADWAY_Q3DIR=C.CODE + "/q3", HIERREP_STAGEB=V + "/pod/stageB", TANITEVAL_ROOT=C.TREE + "/taniteval",
               HEADWAY_JOIN=C.INPUTS["--eval-join"], HEADWAY_BANK=V + "/res/cache/frames.npz", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    sp = "I10" + C.run_name("RH", "R0Hb20")
    for nm, cwd, cmd in (("headway", C.CODE + "/headway", ["headway_read_r2.py", "read", "--t1-root", DRY + "/t1", "--pairs", pairs, "--seeds", seeds, "--spread-arm", sp, "--spread-seeds", seeds, "--out-dir", RD + "/hw"]),
                         ("q3", C.CODE + "/q3", ["q3_read_r3.py", "read", "--t1-root", DRY + "/t1", "--pairs", pairs, "--seeds", seeds, "--spread-arm", sp, "--spread-seeds", seeds, "--same-sign", "--out-dir", RD + "/q3"])):
        t = time.time()
        r = subprocess.run([C.PY, "-u"] + cmd, cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8")
        open(f"{RD}/{nm}.log", "w", encoding="utf-8").write(r.stdout + r.stderr)
        print(nm, "rc", r.returncode, round(time.time() - t), "s", flush=True)
        if r.returncode != 0:
            raise SystemExit(f"{nm} reader failed: {(r.stdout + r.stderr)[-400:]}")
    t = time.time()
    r = subprocess.run([C.PY, "-u", HERE + "/rh_strat.py", "--out", RD, "--seeds", seeds, "--w", DRY], capture_output=True, text=True, encoding="utf-8", env=C.env())
    open(f"{RD}/strat.log", "w", encoding="utf-8").write(r.stdout + r.stderr)
    print("strat rc", r.returncode, round(time.time() - t), "s", flush=True)
    return r.returncode


def judge(planted):
    import rh_read as RR
    q3 = json.load(open(RD + "/q3/RESULT_Q3.json"))
    hw = json.load(open(RD + "/hw/RESULT_HEADWAY.json"))
    sj = json.load(open(RD + "/READ_STRAT_ROLLHIST.json"))
    r = RR.decide(q3, hw, sj, {"holds": True, "fails_at_all_seeds": False}, None, (0, 1), {"blocked": False})
    out = {"outcome": r["outcome"], "P1_families": r["pairs"]["P1"]["families"], "q3_controls": q3.get("controls"), "strat_controls_all_pass": sj.get("controls_all_pass")}
    P1 = q3["pairs"]["I10" + C.PAIRS["P1"][0] + "_minus_I10" + C.PAIRS["P1"][1]]["families"]
    P4 = q3["pairs"]["I10" + C.PAIRS["P4"][0] + "_minus_I10" + C.PAIRS["P4"][1]]["families"]
    if not planted:
        allD = [v["D"] for p in q3["pairs"].values() for v in p["families"].values()]
        allD += [p["cells"][c]["D"] for p in hw["pairs"].values() for c in p["cells"] if "D" in p["cells"][c]]
        out["K-SELF"] = {"cells_checked": len(allD), "max_abs_D": float(np.max(np.abs(allD))), "pass": bool(np.max(np.abs(allD)) == 0.0 and r["outcome"] == "ROLLHIST-NO-EFFECT"
                                                                                          and all(r["pairs"]["P1"]["families"][f] == "NOT-SEPARATED" for f in C.FAMILIES))}
    else:
        stay = ["LONGITUDINAL:speed_mae_mps", "LATERAL:yaw_rate_mae_degps", "LATERAL:curvature_mae_1pm"]
        move = ["LATERAL:heading_mae_deg", "LATERAL:cross_mae_m", "TACTICAL:ade_tac_m"]
        out["K-PLANT"] = {"rotation_deg": ROT_DEG, "must_stay_zero": {c: abs(P1[c]["D"]) for c in stay}, "must_move": {c: P1[c]["D"] for c in move},
                          "P1_family_verdicts": r["pairs"]["P1"]["families"], "P4_must_move": {c: P4[c]["D"] for c in move},
                          "pass": bool(all(abs(P1[c]["D"]) < 1e-9 for c in stay) and all(P1[c]["D"] > 0 and P1[c]["A_lo"] > 0 for c in move) and all(P4[c]["D"] > 0 for c in move))}
    return out


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    res = {}
    if cmd in ("build", "all"):
        build()
    if cmd in ("read", "all"):
        if readers() != 0:
            print("strat reader did not complete: see", RD + "/strat.log")
        res["self"] = judge(False)
        os.makedirs(RD, exist_ok=True)
        json.dump(res["self"], open(HERE + "/DRY_K_SELF.json", "w", encoding="utf-8"), indent=1)
        print(json.dumps(res["self"])[:600], flush=True)
    if cmd in ("plant", "all"):
        plant()
        if readers() != 0:
            print("strat reader did not complete (plant)")
        res["plant"] = judge(True)
        print(json.dumps(res["plant"])[:900], flush=True)
    if res:
        os.makedirs(RD, exist_ok=True)
        json.dump(res, open(RD + "/DRY_RESULTS.json", "w", encoding="utf-8"), indent=1)
        shutil.copyfile(RD + "/DRY_RESULTS.json", HERE + "/DRY_RESULTS.json")
    print("ZZ RH-DRYRUN " + ("PASS" if res and all(v.get("K-SELF", v.get("K-PLANT", {})).get("pass") for v in res.values()) else "FAIL") + " ZZ")


if __name__ == "__main__":
    main()
