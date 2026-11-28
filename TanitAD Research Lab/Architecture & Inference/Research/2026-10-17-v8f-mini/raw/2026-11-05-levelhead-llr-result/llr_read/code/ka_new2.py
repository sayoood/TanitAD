#!/usr/bin/env python3
"""ka_new2.py -- KA-NEW2, the known answer of the LLR read tree (must pass before any LLR read).

The LLR read tree (F:/TanitAD-scratch/motion_c/llr_read/tree: the CL read tree with the 83c9ee0 lead-head model / data / trainer, built by build_llr_read_tree.py) re-rolls the FIRST 64 windows of the eval
grid (run_tool_subset.py, the banked batches) with the RI tool 0417c838 and the reader's flags, and every plan array is compared with the BANKED rollout on the same windows (cl_read/code/ka_new.py's method and
comparison, loaded BY FILE PATH with its sha256 checked; its globals are re-pointed, its comparison code is not touched):
  (A) C0 s0, --scene logged,    16 threads  vs the banked C0_s0     (C:/Users/Admin/rr_scratch/out/C0_s0)                          the lead-head tree reproduces the base rollout (the CL package's KA-NEW (1))
  (B) C0 s0, --scene reindexed, 12 threads  vs the banked RIC0_s0   (F:/TanitAD-scratch/motion_c/ri/out/RIC0_s0)                  the same on the reindexed clock (the CL package's KA-NEW (2))
  (C) CLR s0 (a LEVEL-ON checkpoint, no lead head), --scene reindexed, 12 threads  vs the banked RICLR_s0 (clr_read/out/RICLR_s0)  the lead-head code leaves a level-only model unchanged
  (D) CLR s0, --scene logged, 16 threads vs the banked CLR_s0 (beside)
PASS = (A), (B) and (C) each: every tick row matched, max plan / plan_raw / tau / P diff <= 1e-4 m, dumps cl / ol / fl the same.  (D) is reported beside and does not decide.  Writes raw/KA_NEW2.json and the four
raw/KA_NEW2_<name>.json / logs/KA_NEW2_<name>_run.out.   usage: python -I ka_new2.py [--only A,B,C,D] [--skip-run]"""
import hashlib
import importlib.util
import json
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
KA = "E:/Projects/TanitAD-artifacts/dagland/motion_c/cl_read/code/ka_new.py"
KA_SHA = "c886c0c3f3fb0f2f48f51dc84ff43fdfd142214febeb66c3d6ddd2ff05b250e4"
RDB = "E:/Projects/TanitAD-artifacts/v8f-recover"
TREE = "F:/TanitAD-scratch/motion_c/llr_read/tree"
SM = "F:/TanitAD-scratch/motion_c/llr_read/smoke"
RR = "C:/Users/Admin/rr_scratch/out/C0_s0"
RI = "F:/TanitAD-scratch/motion_c/ri/out/RIC0_s0"
CLRB = "F:/TanitAD-scratch/motion_c/clr_read/out"
CASES = {
    "A": dict(name="C0_logged_first64_16t_vs_banked_C0_s0", ck="C0", scene="logged", th=16, ref_t=f"{RR}/ticks/ticks_C0_s0.npz", ref_d=f"{RR}/dump_C0_s0", decides=True),
    "B": dict(name="C0_reindexed_first64_12t_vs_banked_RIC0_s0", ck="C0", scene="reindexed", th=12, ref_t=f"{RI}/ticks/ticks_RIC0_s0.npz", ref_d=f"{RI}/dump_RIC0_s0", decides=True),
    "C": dict(name="CLR_reindexed_first64_12t_vs_banked_RICLR_s0", ck="CLR", scene="reindexed", th=12, ref_t=f"{CLRB}/RICLR_s0/ticks/ticks_RICLR_s0.npz", ref_d=f"{CLRB}/RICLR_s0/dump_RICLR_s0", decides=True),
    "D": dict(name="CLR_logged_first64_16t_vs_banked_CLR_s0_beside", ck="CLR", scene="logged", th=16, ref_t=f"{CLRB}/CLR_s0/ticks/ticks_CLR_s0.npz", ref_d=f"{CLRB}/CLR_s0/dump_CLR_s0", decides=False),
}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_ka():
    got = sha(KA)
    if got != KA_SHA:
        raise SystemExit(f"REFUSED: {KA} is {got}, not the pinned {KA_SHA}")
    spec = importlib.util.spec_from_file_location("ka_new_cl", KA)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run_case(m, key, skip_run):
    c = CASES[key]
    m.PK, m.TREE, m.CKD = PK, TREE, f"{RDB}/M2b_s0_recft_{c['ck']}"
    argv = ["ka_new2", "--first", "--threads", str(c["th"]), "--max-windows", "64", "--scene", c["scene"], "--tree", TREE, "--tag", f"KANEW2_{key}", "--out-dir", f"{SM}/KA_NEW2_{key}",
            "--ref-ticks", c["ref_t"], "--ref-dump", c["ref_d"], "--result-name", f"KA_NEW2_{c['name']}"] + (["--skip-run"] if skip_run else [])
    old = sys.argv
    sys.argv = argv
    rc = None
    try:
        m.main()
    except SystemExit as e:
        rc = e.code
    finally:
        sys.argv = old
    d = json.load(open(f"{PK}/raw/KA_NEW2_{c['name']}.json", encoding="utf-8"))
    return {"key": key, "name": c["name"], "decides": c["decides"], "rc": rc, "PASS": bool(d["PASS"]), "max_plan_diff_m": d["max_plan_diff_m"], "max_dump_diff_m": d["max_dump_diff_m"],
            "rows": d["n_tick_rows_matched"], "windows": d["n_windows_new"], "threads": c["th"], "scene": c["scene"], "ckpt": c["ck"], "ckpt_sha256": d["ckpt_sha256"], "tool_sha256": d["tool_sha256"],
            "reference_ticks": c["ref_t"]}


def main():
    only = "ABCD"
    for i, t in enumerate(sys.argv):
        if t == "--only":
            only = sys.argv[i + 1].replace(",", "")
    skip = "--skip-run" in sys.argv
    m = load_ka()
    res, out = {"card": "KA-NEW2 of the LLR read package: the lead-head read tree (83c9ee0 model/data/trainer + RI tool) re-rolls C0 s0 and CLR s0", "tol_m": m.TOL, "tree": TREE, "cases": {}}, {}
    for k in only:
        out[k] = run_case(m, k, skip)
        res["cases"][k] = out[k]
    res["model_sha256"], res["data_sha256"], res["trainer_sha256"] = (sha(f"{TREE}/{p}") for p in ("stack/tanitad/models/v8f_mini.py", "stack/tanitad/data/v8f_mini_data.py", "stack/scripts/v8f_mini_train.py"))
    res["tool_sha256"] = sha(f"{TREE}/stack/scripts/v8f_t1_eval_ri.py")
    res["decides"] = "A, B and C must each PASS (D is beside)"
    res["PASS"] = bool(all(out.get(k, {}).get("PASS") for k in "ABC") and set("ABC") <= set(out))
    res["max_plan_diff_m"] = max(v["max_plan_diff_m"] for v in out.values())
    res["clock"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    json.dump(res, open(f"{PK}/raw/KA_NEW2.json", "w", encoding="utf-8", newline="\n"), indent=1)
    print("KA_NEW2", {k: (v["PASS"], v["max_plan_diff_m"], v["rows"]) for k, v in out.items()}, "->", "PASS" if res["PASS"] else "FAIL")
    sys.exit(0 if res["PASS"] else 2)


if __name__ == "__main__":
    main()
