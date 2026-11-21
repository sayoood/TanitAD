"""PREREG_V8F_DAG_HACC section 2: Read A's comps-based rows for the DAG-HACC arm, with readA.py's / rr_lib's OWN functions (imported, not re-implemented), arm swapped.
  per config (causal = DECISIVE, logged = limit-logged BESIDE, frozen_causal = STRICT BESIDE) and seed:
    rows_arm_minus_P10   (ADE6, speed / along / accel, heading / yaw-rate / curvature / cross-track, L1, L2, M1 per-window share, M2)   -> Read A
    rows_arm_minus_DAG   the same rows on DAG's own rollout of the same config, same windows                                           -> KEEP (ADE6) + the 'DAG-HACC - DAG' beside rows
    C2 identity          (arm, arm) reads exactly +0 [0, 0] on every row
    DAG - P10 reproduces the banked reread rows (raw/readA_DAG_causal.json) exactly
usage: readA_daghacc.py <arm prefix> <strict prefix> <out json> [--dag-prefix DAG --dag-strict DAGSTRICT]   (the prefixes are the packed-file stems; the real arm is DAGHACC / DAGHACCSTRICT)"""
import json
import os
import sys

os.environ.setdefault("RR_READERS", "E:/Projects/TanitAD-artifacts/dagreread/reader_src")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True
sys.path.insert(0, "E:/Projects/TanitAD-artifacts/dagreread/code")
import numpy as np  # noqa: E402
import readA as RA  # noqa: E402
import rr_lib as R  # noqa: E402

PK = "E:/Projects/TanitAD-artifacts/daghacc"
DAGP = "E:/Projects/TanitAD-artifacts/dagreread/packed"
arm, arm_strict, out_p = sys.argv[1], sys.argv[2], sys.argv[3]
BANKED = json.load(open("E:/Projects/TanitAD-artifacts/dagreread/raw/readA_DAG_causal.json"))
CFGS = (("causal", arm, "DAG", "causal"), ("logged", arm, "DAG", "logged"), ("frozen_causal", arm_strict, "DAGSTRICT", "frozen_causal"))
ONLY = os.environ.get("DAGHACC_CFGS")
if ONLY:
    CFGS = tuple(c for c in CFGS if c[0] in ONLY.split(","))
res = {"arm": arm, "tier": "T1-kin (self-action OPEN loop; m=10, nav deadreckon, periodic; causal = DECISIVE; logged = limit-logged beside; frozen_causal = STRICT beside)", "cfgs": {}}
for cfg, a_name, d_name, suffix in CFGS:
    res["cfgs"][cfg] = {}
    for s in R.SEEDS:
        d = R.load_npz(f"{PK}/packed/dumps_{a_name}_s{s}_m10_{suffix}.npz")
        dd = R.load_npz(f"{DAGP}/dumps_{d_name}_s{s}_m10_{suffix}.npz")
        eid = d["eid"]
        same = {k: bool(np.array_equal(d[k], dd[k])) for k in ("clip_index", "ws", "eid", "g", "v0", "fl")}
        same["n_windows"] = int(len(eid))
        same["n_episodes"] = int(len(np.unique(eid)))
        assert all(v for k, v in same.items() if k not in ("n_windows", "n_episodes")), ("C4 windows differ", cfg, s, same)
        blk = RA.rows_block(d["cl"], d["fl"], d["g"], eid)
        blk_dag = RA.rows_block(d["cl"], dd["cl"], d["g"], eid)
        ident = RA.ident_block(d["cl"], d["g"], eid)
        ident_ok = all(v[0] == 0.0 and v[1] == 0.0 and v[2] == 0.0 for v in ident.values())
        rec = {"rows_arm_minus_P10": blk, "rows_arm_minus_DAG": blk_dag, "C2_identity_comps_rows": {"all_exactly_zero": ident_ok, "rows": ident}, "C4_same_windows_as_DAG": same,
               "plan_moved_vs_DAG": {"max_abs_cl_m": float(np.abs(d["cl"] - dd["cl"]).max()), "share_windows_cl_differs_gt_1e-3m": float((np.abs(d["cl"] - dd["cl"]).max(axis=(1, 2)) > 1e-3).mean())}}
        if cfg == "causal" and a_name == "DAGHACC":
            ref = BANKED["seeds"][f"s{s}"]["rows_arm_minus_P10"]
            dag_blk = RA.rows_block(dd["cl"], dd["fl"], dd["g"], eid)
            rec["DAG_minus_P10_reproduces_banked_reread_rows"] = bool(all(dag_blk[k]["delta"] == ref[k]["delta"] and dag_blk[k]["lo"] == ref[k]["lo"] and dag_blk[k]["hi"] == ref[k]["hi"]
                                                                          for k in ref if not k.startswith("_")))
        res["cfgs"][cfg][f"s{s}"] = rec
        for lab, v in blk.items():
            if lab.startswith("_"):
                continue
            print(f"{cfg} s{s} {lab:30s} vs P10 {v['delta']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}] sep={v['separated']} | vs DAG {blk_dag[lab]['delta']:+.4f} [{blk_dag[lab]['lo']:+.4f}, {blk_dag[lab]['hi']:+.4f}] sep={blk_dag[lab]['separated']}", flush=True)
        print(f"{cfg} s{s} C2 identity exactly zero: {ident_ok}  plan moved vs DAG: {rec['plan_moved_vs_DAG']}", flush=True)
json.dump(res, open(out_p, "w"), indent=1)
print("ZZREADA-DONE", out_p)
