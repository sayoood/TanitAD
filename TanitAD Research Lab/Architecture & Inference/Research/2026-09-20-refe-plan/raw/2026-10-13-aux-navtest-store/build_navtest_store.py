"""Build the NAVTEST label store with the SAME builder as the train store (refe/aux_label_store.py, unmodified). CPU only, dev box.

WHY A DRIVER AND NOT A NEW BUILDER
  The train store is built by `aux_label_store.py build` from a bank (origin_world) + the log's map. Navtest has no bank, but every one of its
  136 logs is held locally as a nuPlan DB (D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test, MEASURED 136/136 logs, 12,146/12,146 tokens in
  `lidar_pc`), so the builder's own DB branch applies: origin = the DB `ego_pose` (rear axle, devkit heading), map = the DB `log.map_version`.
  That branch is MEASURED identical to path A (NuPlanScenario) on the train store (`s0_gate_post.json`: A-vs-B 0 cells); `s0_gate_navtest.py`
  re-measures it on navtest. Nothing in the label maths is touched: this file only (1) turns navtest.json into the yaml the builder wants,
  (2) runs `logmap` + `build` + (the builder's own) `finalize` as subprocesses, (3) asserts the store is key-disjoint from the train store.

OUTPUT   D:/Projects/TanitAD-artifacts/aux_drivable/label_store_navtest/
           shards/<log>.aux1  manifest.json  run_meta_all.json  failures_all.jsonl  missing_tokens.txt
           inputs/navtest_tokens.yaml  inputs/empty_bank.jsonl  inputs/log_map_navtest.json  disjoint_vs_train.json  build_navtest.log
LAST LINE  ZZNAVTESTSTORE_DONE <stored_keys> <failed> overlap_tokens=<n> overlap_logs=<n>   (a trailing ABORTED/FAIL = read the log)

Run (driverl-eval venv, export PYTHONIOENCODING=utf-8; resumable by key):
  python build_navtest_store.py [--workers 4]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[1]
sys.path.insert(0, str(PKG / "refe"))
import aux_label_store as ALS  # noqa: E402

ROOT = "D:/Projects/TanitAD"
ART = f"{ROOT}-artifacts/aux_drivable"
NAVTEST_JSON = f"{ROOT}/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-estimator-and-route-leak/raw/cluster_maps/navtest.json"
DB_ROOT = f"{ROOT}/data/nuplan/nuplan-v1.1/splits/test"
MAPS_ROOT = f"{ROOT}/data/nuplan-maps/nuplan-maps-v1.0"
TRAIN_STORE = f"{ART}/label_store"
OUT = f"{ART}/label_store_navtest"


def write_inputs(out: Path, navtest_json: str) -> dict:
    """navtest.json -> inputs/navtest_tokens.yaml (the builder's `log_names` + `tokens`; every token a double-quoted string, so a digit-only
    token can never be read back as an int and lose its leading zeros) and an empty bank (the builder requires one)."""
    d = json.load(open(navtest_json, encoding="utf-8"))
    t2l = d["token_to_log_name"]
    toks = [t.lower() for t in t2l]
    logs = sorted(set(t2l.values()))
    inp = out / "inputs"
    inp.mkdir(parents=True, exist_ok=True)
    lines = ["log_names:"] + [f'  - "{l}"' for l in logs] + ["tokens:"] + [f'  - "{t}"' for t in toks]
    (inp / "navtest_tokens.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (inp / "empty_bank.jsonl").write_text("", encoding="utf-8")
    ylogs, ytoks = ALS.load_yaml_tokens(inp / "navtest_tokens.yaml")
    assert len(ytoks) == len(toks) == len(t2l) and set(ylogs) == set(logs), "yaml round-trip changed the navtest universe"
    return dict(tokens=len(toks), logs=len(logs), navtest_json_sha256=ALS.sha256_file(navtest_json), yaml_sha256=ALS.sha256_file(inp / "navtest_tokens.yaml"))


def keys_of(store: Path) -> dict:
    """{log: set(tokens)} from the shard HEADERS only (cheap)."""
    out = {}
    for p in sorted(glob.glob(str(store / "shards" / "*.aux1"))):
        head, _ = ALS.read_shard(Path(p), with_blobs=False)
        out[head["log"]] = {e["t"] for e in head["entries"]}
    return out


def disjointness(navtest_store: Path, train_store: Path, navtest_json: str) -> dict:
    nt = keys_of(navtest_store)
    tr = keys_of(train_store)
    nt_tok = {t for s in nt.values() for t in s}
    tr_tok = {t for s in tr.values() for t in s}
    nt_pairs = {(l, t) for l, s in nt.items() for t in s}
    tr_pairs = {(l, t) for l, s in tr.items() for t in s}
    ref_t, ref_l = ALS.load_navtest(navtest_json)
    return dict(navtest_store_keys=len(nt_pairs), navtest_store_logs=len(nt), train_store_keys=len(tr_pairs), train_store_logs=len(tr),
                key_overlap=len(nt_pairs & tr_pairs), token_overlap=len(nt_tok & tr_tok), log_overlap=len(set(nt) & set(tr)),
                navtest_store_tokens_not_in_navtest_json=len(nt_tok - ref_t), navtest_store_logs_not_in_navtest_json=len(set(nt) - ref_l),
                navtest_json_tokens_missing_from_store=len(ref_t - nt_tok))


def run(cmd: list, log, env) -> int:
    log.write("$ " + " ".join(cmd) + "\n")
    log.flush()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True, encoding="utf-8", errors="replace")
    for line in p.stdout:
        log.write(line)
        log.flush()
        print(line.rstrip(), flush=True)
    return p.wait()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--train-store", default=TRAIN_STORE)
    ap.add_argument("--navtest", default=NAVTEST_JSON)
    ap.add_argument("--db-root", default=DB_ROOT)
    ap.add_argument("--maps-root", default=MAPS_ROOT)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    info = write_inputs(out, a.navtest)
    inp = out / "inputs"
    py, builder = sys.executable, str(PKG / "refe" / "aux_label_store.py")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    t0 = time.time()
    with open(out / "build_navtest.log", "a", encoding="utf-8") as log:
        log.write(f"== {time.strftime('%Y-%m-%d %H:%M:%S')} navtest store build; inputs {json.dumps(info)}\n")
        rc = run([py, "-u", builder, "logmap", "--bank", str(inp / "empty_bank.jsonl"), "--db-root", a.db_root, "--maps-root", a.maps_root,
                  "--yaml", str(inp / "navtest_tokens.yaml"), "--out", str(inp / "log_map_navtest.json")], log, env)
        if rc != 0:
            print(f"ZZNAVTESTSTORE_DONE 0 0 overlap_tokens=-1 overlap_logs=-1 FAIL_LOGMAP rc={rc}")
            return rc
        # NOTE: no --navtest here -- the builder's navtest guard EXCLUDES those tokens (that is how the train store stays clean).
        # Disjointness from the train store is asserted below, key by key.
        rc = run([py, "-u", builder, "build", "--bank", str(inp / "empty_bank.jsonl"), "--db-root", a.db_root, "--maps-root", a.maps_root,
                  "--yaml", str(inp / "navtest_tokens.yaml"), "--logmap", str(inp / "log_map_navtest.json"), "--out", str(out),
                  "--workers", str(a.workers)], log, env)
        man = json.load(open(out / "manifest.json", encoding="utf-8")) if (out / "manifest.json").exists() else None
        dj = disjointness(out, Path(a.train_store), a.navtest) if man else None
        if dj:
            (out / "disjoint_vs_train.json").write_text(json.dumps(dj, indent=1), encoding="utf-8")
            log.write(json.dumps(dj) + "\n")
        stored = man["counts"]["stored_keys"] if man else 0
        rm = out / "run_meta_all.json"
        failed = json.load(open(rm, encoding="utf-8")).get("totals", {}).get("failed", -1) if rm.exists() else -1
        ok = rc == 0 and failed == 0 and dj is not None and dj["key_overlap"] == 0 and dj["log_overlap"] == 0 and dj["navtest_json_tokens_missing_from_store"] == 0 \
            and dj["navtest_store_tokens_not_in_navtest_json"] == 0
        last = (f"ZZNAVTESTSTORE_DONE {stored} {failed} overlap_tokens={dj['token_overlap'] if dj else -1} "
                f"overlap_logs={dj['log_overlap'] if dj else -1}" + ("" if ok else f" FAIL rc={rc}") + f" seconds={time.time() - t0:.0f}")
        log.write(last + "\n")
    print(last)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
