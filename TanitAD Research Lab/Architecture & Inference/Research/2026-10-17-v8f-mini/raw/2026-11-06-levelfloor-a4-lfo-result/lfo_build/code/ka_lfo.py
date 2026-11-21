#!/usr/bin/env python3
"""ka_lfo.py -- the known answers KA-O1 and KA-O2 of the LFO tool (PREREG_LEVELFLOOR_A4 section 2), on real data, FULL rollouts, CPU only, 12 threads (the banked reindexed rollouts' setting).

  KA-O1  the LFO tool with ``--level-floor on --floor-kappa 0.75 --floor-ckpt LLR_s0 --floor-vhat-source heads`` re-rolls RILF075_s0 (bare arm, reindexed, sg72) and every array equals the BANKED LF rollout
         (``F:/TanitAD-scratch/motion_c/lf_read/out/RILF075_s0``) byte for byte: all common fields of the ticks file, every ``ep*.npz`` of the dump, the analysis JSON (everything but the run header), and the
         per-window metric arrays extracted by the read package's own extractor against ``lf_read/raw/arrays/RILF075_s0.npz``. Allowed differences, and only these: the ticks file gains the key
         ``floor_vhat_source``; the header differs in ``ticks_file`` (the path), ``floor_vhat_source`` and ``level_floor_detail.floor_vhat_source``.
  KA-O2  the LFO tool with the floor OFF (no floor flag at all) re-rolls RIC0_s0 and equals the banked RI rollout (``F:/TanitAD-scratch/motion_c/ri/out/RIC0_s0``) with NO allowed difference but ``ticks_file``;
         arrays against ``ri_read/raw/arrays/RIC0_s0.npz``.
usage: python -I ka_lfo.py [--cases O1,O2] [--skip-run]      writes raw/KA_LFO.json, raw/KA_LFO_<case>.json, logs/KA_LFO_<case>_run.out"""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time

sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
ART = "E:/Projects/TanitAD-artifacts"
MC = f"{ART}/dagland/motion_c"
RDB = f"{ART}/v8f-recover"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
SW = "F:/TanitAD-scratch/motion_c/lfo_build"
TREE = "F:/TanitAD-scratch/motion_c/lfo_read/tree"
TOOL = f"{TREE}/stack/scripts/v8f_t1_eval_ri.py"
DATA = f"{ART}/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
LF_READ_PY = f"{MC}/lf_read/code/lf_read.py"
CASES = {
    "O1": dict(tag="RILF075_s0", seed=0, extra=["--smooth", "sg72", "--scene", "reindexed", "--level-floor", "on", "--floor-kappa", "0.75", "--floor-ckpt", f"{RDB}/M2b_s0_recft_LLR/ckpt_last.pt", "--floor-vhat-source", "heads"],
               banked="F:/TanitAD-scratch/motion_c/lf_read/out/RILF075_s0", banked_arrays=f"{MC}/lf_read/raw/arrays/RILF075_s0.npz",
               tick_extra_allowed={"floor_vhat_source"}, header_allowed={"t1v8f.ticks_file", "t1v8f.floor_vhat_source", "t1v8f.level_floor_detail.floor_vhat_source"},
               what="the LFO tool, floor on at kappa 0.75, --floor-vhat-source heads, vs the banked RILF075_s0"),
    "O2": dict(tag="RIC0_s0", seed=0, extra=["--smooth", "sg72", "--scene", "reindexed"],
               banked="F:/TanitAD-scratch/motion_c/ri/out/RIC0_s0", banked_arrays=f"{MC}/ri_read/raw/arrays/RIC0_s0.npz",
               tick_extra_allowed=set(), header_allowed={"t1v8f.ticks_file"}, what="the LFO tool with the floor off (no floor flag) vs the banked RIC0_s0"),
}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def same_arr(x, y):
    """Byte-level equality of two arrays (dtype, shape and bytes; object arrays by value)."""
    if x.dtype != y.dtype or x.shape != y.shape:
        return False
    if x.dtype.kind == "O":
        return x.tolist() == y.tolist()
    return x.tobytes() == y.tobytes()


def deep_diff(a, b, path, out):
    """Paths where two parsed JSON values differ (NaN equals NaN); keys present on one side only are listed as ``path.key``."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}.{k}")
            else:
                deep_diff(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(path)
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                deep_diff(x, y, f"{path}[{i}]", out)
    elif isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return
    elif a != b:
        out.append(path)


def run_rollout(c, key, skip):
    od = f"{SW}/ka/KA{key}"
    out_dir = f"{od}/{c['tag']}"
    marker = f"{od}/done.txt"
    if skip and os.path.exists(marker):
        return {"wall_s": json.load(open(marker))["wall_s"], "skipped": True, "out_dir": out_dir}
    if os.path.isdir(od):
        os.replace(od, f"{od}_incomplete_{time.strftime('%H%M%S')}")
    os.makedirs(out_dir)
    s = c["seed"]
    cmd = [PY, "-I", TOOL, "run", "--ckpt", f"{RDB}/M2b_s{s}_recft_C0/ckpt_last.pt", "--tag", c["tag"], "--tracks", f"{ART}/sfresh_inputs/inputs/native_tracks.pt", "--eval-cache", f"{ART}/sfresh_inputs/cache",
           "--eval-grid", f"{DATA}/eval_grid_windows.json", "--eval-join", f"{DATA}/join_eval141.npz", "--out-dir", out_dir, "--replan-steps", "10", "--nav", "deadreckon", "--schedule", "periodic", "--limit", "causal",
           "--dump-ticks", f"{out_dir}/ticks", "--dump-heads", "--device", "cpu"] + c["extra"]
    env = dict(os.environ, OMP_NUM_THREADS="12", MKL_NUM_THREADS="12", OPENBLAS_NUM_THREADS="12", CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")
    t0 = time.time()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    wall = time.time() - t0
    os.makedirs(f"{PK}/logs", exist_ok=True)
    open(f"{PK}/logs/KA_LFO_{key}_run.out", "w", encoding="utf-8", newline="\n").write(" ".join(cmd) + "\n" + r.stdout[-6000:] + r.stderr[-3000:])
    if r.returncode:
        raise SystemExit(f"KA-{key} rollout failed rc={r.returncode}: {r.stderr[-800:]}")
    json.dump({"wall_s": wall, "cmd": cmd}, open(marker, "w"))
    return {"wall_s": wall, "skipped": False, "out_dir": out_dir}


def compare(c, key, ran):
    new_dir, old_dir, tag = ran["out_dir"], c["banked"], c["tag"]
    res = {"key": key, "what": c["what"], "tag": tag, "new": new_dir, "banked": old_dir, "wall_s": ran["wall_s"]}
    # ticks file
    zn = dict(np.load(f"{new_dir}/ticks/ticks_{tag}.npz", allow_pickle=True))
    zo = dict(np.load(f"{old_dir}/ticks/ticks_{tag}.npz", allow_pickle=True))
    only_new, only_old = sorted(set(zn) - set(zo)), sorted(set(zo) - set(zn))
    bad = [k for k in sorted(set(zn) & set(zo)) if not same_arr(zn[k], zo[k])]
    res["ticks"] = {"n_fields_common": len(set(zn) & set(zo)), "fields_not_byte_identical": bad, "only_in_new": only_new, "only_in_banked": only_old, "n_rows": int(len(zn["v0"]))}
    res["ticks"]["PASS"] = bool(not bad and not only_old and set(only_new) == set(c["tick_extra_allowed"]) and res["ticks"]["n_rows"] == 7197)
    # dumps
    dn, do = f"{new_dir}/dump_{tag}", f"{old_dir}/dump_{tag}"
    fo = sorted(f for f in os.listdir(do) if f.startswith("ep") and f.endswith(".npz"))
    fn = sorted(f for f in os.listdir(dn) if f.startswith("ep") and f.endswith(".npz"))
    bad_d = []
    for f in fo:
        if f not in fn:
            bad_d.append(f)
            continue
        a, b = dict(np.load(f"{dn}/{f}", allow_pickle=True)), dict(np.load(f"{do}/{f}", allow_pickle=True))
        if set(a) != set(b) or any(not same_arr(a[k], b[k]) for k in b):
            bad_d.append(f)
    res["dumps"] = {"n_banked": len(fo), "n_new": len(fn), "episodes_not_byte_identical": bad_d, "PASS": bool(len(fo) == len(fn) == 141 and not bad_d)}
    # analysis JSON: everything but the header, then the header with the allowed differences named
    jn, jo = json.load(open(f"{new_dir}/t1v8f_{tag}.json", encoding="utf-8")), json.load(open(f"{old_dir}/t1v8f_{tag}.json", encoding="utf-8"))
    hn, ho = jn.pop("t1v8f"), jo.pop("t1v8f")
    d_body, d_head = [], []
    deep_diff(jn, jo, "res", d_body)
    deep_diff({"t1v8f": hn}, {"t1v8f": ho}, "", d_head)
    d_head = [p.lstrip(".") for p in d_head]
    res["analysis_json"] = {"body_differences": d_body, "header_differences": d_head, "header_allowed": sorted(c["header_allowed"]), "PASS": bool(not d_body and set(d_head) <= set(c["header_allowed"]))}
    res["header_floor_vhat_source"] = hn.get("floor_vhat_source")
    res["header_label"] = hn.get("label")
    # per-window metric arrays through the read package's own extractor
    spec = importlib.util.spec_from_file_location("lf_read_for_ka_lfo", LF_READ_PY)
    LFR = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(LFR)
    LFR.init()
    rr = LFR.rr
    rr.PK = PK
    rr.WORK = f"{SW}/ka/work"
    arr_dir = f"{SW}/ka/arrays_{key}"
    os.makedirs(arr_dir, exist_ok=True)
    dst = f"{arr_dir}/{tag}.npz"
    if os.path.exists(dst):
        os.replace(dst, dst + f".old_{time.strftime('%H%M%S')}")
    rr.extract(tag, c["seed"], os.path.dirname(new_dir), arr_dir)
    an, ao = dict(np.load(dst, allow_pickle=True)), dict(np.load(c["banked_arrays"], allow_pickle=True))
    bad_a = [k for k in sorted(set(ao) | set(an)) if k not in an or k not in ao or not same_arr(an[k], ao[k])]
    res["metric_arrays"] = {"new": dst, "banked": c["banked_arrays"], "banked_sha256": sha(c["banked_arrays"]), "n_keys": len(ao), "keys_not_byte_identical": bad_a, "PASS": bool(not bad_a and len(ao) > 0)}
    res["PASS"] = bool(all(res[k]["PASS"] for k in ("ticks", "dumps", "analysis_json", "metric_arrays")))
    return res


def main():
    cases = [x for x in (sys.argv[sys.argv.index("--cases") + 1].split(",") if "--cases" in sys.argv else list(CASES))]
    skip = "--skip-run" in sys.argv
    out = {"card": "KA-O1 / KA-O2 of the LFO build (PREREG_LEVELFLOOR_A4 section 2)", "tool": TOOL, "tool_sha256": sha(TOOL), "operator_sha256": sha(f"{TREE}/stack/scripts/level_floor.py"), "threads": 12, "cases": {}}
    os.makedirs(f"{PK}/raw", exist_ok=True)
    for k in cases:
        ran = run_rollout(CASES[k], k, skip)
        out["cases"][k] = compare(CASES[k], k, ran)
        json.dump(out["cases"][k], open(f"{PK}/raw/KA_LFO_{k}.json", "w", encoding="utf-8", newline="\n"), indent=1, default=str)
        print(f"KA-{k}: ticks {out['cases'][k]['ticks']['PASS']} dumps {out['cases'][k]['dumps']['PASS']} analysis {out['cases'][k]['analysis_json']['PASS']} arrays {out['cases'][k]['metric_arrays']['PASS']} -> "
              f"{'PASS' if out['cases'][k]['PASS'] else 'FAIL'} ({out['cases'][k]['wall_s']:.0f} s)", flush=True)
    out["PASS"] = bool(all(v["PASS"] for v in out["cases"].values()) and set(out["cases"]) == set(CASES))
    out["clock"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    json.dump(out, open(f"{PK}/raw/KA_LFO.json", "w", encoding="utf-8", newline="\n"), indent=1, default=str)
    print("KA_LFO", "PASS" if out["PASS"] else "FAIL")
    sys.exit(0 if out["PASS"] else 2)


if __name__ == "__main__":
    main()
