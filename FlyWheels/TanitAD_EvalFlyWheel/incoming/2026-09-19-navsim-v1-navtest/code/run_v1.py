#!/usr/bin/env python3
"""W3 driver — NAVSIM v1.1 metric cache + scoring on `navtest`, with count guards (any python).

Every devkit step runs as its OWN process through ``navsim_v1_win.py`` (the unmodified v1.1
entrypoint, imported through PYTHONPATH). This driver only launches, then REFUSES a run that
did no work — *success over an empty set is a failure*:

    python run_v1.py select-smoke  --out raw/smoke_tokens.json
    python run_v1.py cache  --label cache_smoke20 --tokens raw/smoke_tokens.json --cache-name smoke20
    python run_v1.py check-loader --cache-name smoke20              # control C7 (NAVSIM venv)
    python run_v1.py score  --label CV_smoke20 --arm CV --tokens raw/smoke_tokens.json --cache-name smoke20
    python run_v1.py score  --label CV_navtest --arm CV --cache-name navtest     # full split

Arms: ``CV`` (devkit ConstantVelocityAgent), ``HUMAN`` (devkit HumanAgent), ``STOP``
(``w3_agents_v1.StopAgent``), ``EGO_MLP:<ckpt>`` (devkit EgoStatusMLPAgent + a published
checkpoint), ``SEAM:<npz>`` (``w3_agents_v1.SeamAgentV1``).

Checks per score run (SPEC §5): C2 log counts + CSV rows, C3 token-set equality, C1 the PDMS
formula on every row, C8 the wrapper's import provenance. Per cache run: C9 metadata rows ==
expected tokens and every row's parent directory == its token.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
RAW = os.path.join(PKG, "raw")
NV_PY = "C:/Users/Admin/navsim-crun/venv/Scripts/python.exe"
V11_TREE = "D:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
EXP = "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1"
DATA = "D:/Archive/devbox-C/navsim/data/openscene"
NAVTEST_YAML = (f"{V11_TREE}/navsim/planning/script/config/common/train_test_split/"
                f"scene_filter/navtest.yaml")
WRAP = os.path.join(HERE, "navsim_v1_win.py")
ENV = {
    "NUPLAN_MAP_VERSION": "nuplan-maps-v1.0",
    "NUPLAN_MAPS_ROOT": "D:/Archive/devbox-C/navsim/data/maps",
    "NAVSIM_EXP_ROOT": EXP,
    "NAVSIM_DEVKIT_ROOT": V11_TREE,
    "OPENSCENE_DATA_ROOT": DATA,
    "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
    "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
    "CUDA_VISIBLE_DEVICES": "-1",
}
TERMS = ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
         "time_to_collision_within_bound", "comfort", "driving_direction_compliance", "score")
SHORT = {"no_at_fault_collisions": "NC", "drivable_area_compliance": "DAC", "ego_progress": "EP",
         "time_to_collision_within_bound": "TTC", "comfort": "C",
         "driving_direction_compliance": "DDC", "score": "PDMS"}


# --------------------------------------------------------------------------- #
def navtest_tokens() -> tuple[list, list]:
    """(log_names, tokens) of the v1.1 navtest scene filter, parsed WITHOUT yaml (plain
    `- 'x'` list items), so this driver runs in any interpreter."""
    logs, toks, cur = [], [], None
    for ln in open(NAVTEST_YAML, encoding="utf-8"):
        s = ln.rstrip("\n")
        if s.startswith("log_names:"):
            cur = logs
            continue
        if s.startswith("tokens:"):
            cur = toks
            continue
        m = re.match(r"^\s*-\s*'?([^']+?)'?\s*$", s)
        if m and cur is not None:
            cur.append(m.group(1))
        elif s and not s.startswith(" ") and not s.startswith("-"):
            cur = None
    return logs, toks


def split_overrides(tok_file: str | None) -> tuple[list, list | None]:
    ov = ["train_test_split=navtest"]
    if not tok_file:
        return ov, None
    doc = json.load(open(tok_file, encoding="utf-8"))
    toks = list(doc["tokens"])
    logs = sorted({doc["token_log"][t] for t in toks})
    ov.append("train_test_split.scene_filter.log_names=[" + ",".join(f"'{x}'" for x in logs) + "]")
    ov.append("train_test_split.scene_filter.tokens=[" + ",".join(f"'{x}'" for x in toks) + "]")
    return ov, toks


def env_for(hashseed: str = "1") -> dict:
    env = dict(os.environ)
    env.update(ENV)
    env["PYTHONHASHSEED"] = hashseed
    env["PYTHONPATH"] = V11_TREE + os.pathsep + HERE
    return env


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
def cmd_select_smoke(a) -> int:
    """≤ 20 tokens: 5 from each of the first 4 navtest logs (yaml order) with DISTINCT
    map_location. Deterministic; needs the NAVSIM venv (the log pickles hold numpy)."""
    import pickle
    logs, toks = navtest_tokens()
    tokset = set(toks)
    chosen, token_log, maps = [], {}, {}
    for ln in logs:
        fr = pickle.load(open(f"{DATA}/navsim_logs/test/{ln}.pkl", "rb"))
        loc = fr[0]["map_location"]
        if loc in maps.values():
            continue
        mine = [f["token"] for f in fr if f["token"] in tokset][: a.per_log]
        if len(mine) < a.per_log:
            continue
        maps[ln] = loc
        for t in mine:
            chosen.append(t)
            token_log[t] = ln
        if len(maps) == a.n_logs:
            break
    doc = {"rule": f"first {a.per_log} navtest tokens (log frame order) of each of the first "
                   f"{a.n_logs} navtest logs (yaml order) with a DISTINCT map_location",
           "n": len(chosen), "logs": maps, "tokens": chosen, "token_log": token_log,
           "navtest_yaml": NAVTEST_YAML, "navtest_n_logs": len(logs), "navtest_n_tokens": len(toks)}
    json.dump(doc, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: doc[k] for k in ("n", "logs")}, indent=1))
    return 0 if len(chosen) == a.per_log * a.n_logs else 1


def run_wrapped(script: str, label: str, overrides: list, out_dir: str, *, patch: bool,
                hashseed: str = "1", record_poses: bool = False) -> tuple[int, str, float]:
    os.makedirs(out_dir, exist_ok=True)
    cmd = [NV_PY, WRAP, "--script", script, "--label", label, "--out-dir", out_dir]
    if patch:
        cmd.append("--patch-loader")
    if record_poses:
        cmd.append("--record-poses")
    cmd += ["--"] + overrides
    log_path = os.path.join(out_dir, f"{label}.log")
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("CMD: " + json.dumps(cmd) + "\n")
        fh.flush()
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env_for(hashseed),
                            cwd=EXP).returncode
    return rc, log_path, time.time() - t0


def cmd_cache(a) -> int:
    ov, toks = split_overrides(a.tokens)
    cache = f"{EXP}/metric_cache_{a.cache_name}"
    ov += [f"cache.cache_path={cache}", f"worker={a.worker}"]
    out_dir = os.path.join(RAW, a.label)
    per_log_reps = []
    if getattr(a, "per_log", False):
        # ⛔ WHY PER LOG. `cache_data` builds ONE SceneLoader over ALL 136 navtest logs before it
        # caches anything: 2.1 GB RSS and ~2 min before the first scene (MEASURED 2026-09-20), and
        # a sibling agent's CPU training spike (33 GB committed) then tripped the RAM guard and
        # cost the whole pass. One log per process needs ~0.4 GB, is resumable per log, and a
        # memory spike costs ONE log. The devkit code is unchanged: only the scene filter's
        # log_names is overridden.
        logs = navtest_tokens()[0]
        want_per_log = _tokens_per_log()
        n_per = max(1, int(getattr(a, "logs_per_process", 1)))
        groups = [logs[i:i + n_per] for i in range(0, len(logs), n_per)]
        for i, grp in enumerate(groups):
            done = [ln for ln in grp
                    if want_per_log and _n_pkl(os.path.join(cache, ln)) >= want_per_log.get(ln, -1) > 0]
            if len(done) == len(grp):
                per_log_reps.append({"logs": grp, "rc": 0, "skipped": "already complete",
                                     "n_expected": sum(want_per_log[x] for x in grp)})
                continue
            ln = ",".join(grp)                      # for the record below
            sub_ov = [o for o in ov if not o.startswith("train_test_split.scene_filter.log_names=")]
            sub_ov.append("train_test_split.scene_filter.log_names=["
                          + ",".join(f"'{x}'" for x in grp) + "]")
            rc_i, wl, tries, waited = 3, 0.0, 0, 0.0
            # ⛔ WAIT FOR MEMORY BEFORE EVERY LOG, AND RETRY A RAM-GUARD ABORT (rc 3). MEASURED
            # 2026-09-20: a sibling CPU training smoke committed 33.9 GB and drove system-available
            # memory to 553 MB; without this wait the loop burned through logs at 1.9 s each,
            # aborting every one — "success over an empty set" in slow motion.
            while tries < a.log_tries:
                w = _wait_avail(a.min_avail_mb, a.max_wait_min)
                waited += w
                if w < 0:
                    print(json.dumps({"log": ln, "rc": "NO_RAM_WINDOW", "waited_s": -w}), flush=True)
                    rc_i = 4
                    break
                n_before = _n_pkl(cache)
                tries += 1
                rc_i, lp, wl = run_wrapped("metric_caching", f"{a.label}_log{i:03d}", sub_ov,
                                           os.path.join(out_dir, "per_log"), patch=False)
                added = _n_pkl(cache) - n_before
                if rc_i == 0 or added > 0 and rc_i != 3:
                    break
                if rc_i != 3:
                    break
                time.sleep(120)
            per_log_reps.append({"log": ln, "logs": grp, "n_logs": len(grp),
                                 "n_already_complete": len(done), "rc": rc_i, "tries": tries,
                                 "waited_s": round(waited, 1), "wall_s": round(wl, 1),
                                 "n_pkl_total": _n_pkl(cache)})
            print(json.dumps(per_log_reps[-1]), flush=True)
            if rc_i not in (0,):
                print(f"⛔ log {ln} rc={rc_i} — continuing; the final pass re-checks every token",
                      flush=True)
        json.dump(per_log_reps, open(os.path.join(out_dir, "per_log_runs.json"), "w",
                                     encoding="utf-8"), indent=1)
    # the FINAL pass is the devkit's own: every scene is already cached, so it only re-emits the
    # metadata CSV over the whole split (the file MetricCacheLoader reads). It needs the 2.1 GB
    # all-logs SceneLoader for ~53 min, so it waits for memory like every per-log step — and is
    # SKIPPED entirely when the CSV already names exactly the expected number of tokens (written
    # by the devkit's own save_cache_metadata via code/write_cache_metadata.py, verified on
    # content). Re-deriving a list of paths that are on disk is the expensive way to be as right.
    n_exp = len(toks) if toks is not None else len(navtest_tokens()[1])
    pre = sorted(glob.glob(f"{cache}/metadata/*.csv"))
    pre_rows = 0
    if len(pre) == 1:
        with open(pre[0], encoding="utf-8") as fh:
            pre_rows = max(0, sum(1 for _ in fh) - 1)
    skip_final = pre_rows == n_exp and n_exp > 0
    if skip_final:
        print(json.dumps({"final_pass": "SKIPPED", "why": "metadata csv already complete",
                          "rows": pre_rows, "expected": n_exp}), flush=True)
        rc, log_path, wall = 0, os.path.join(out_dir, f"{a.label}.log"), 0.0
    else:
        if getattr(a, "per_log", False):
            _wait_avail(a.min_avail_mb, a.max_wait_min)
        rc, log_path, wall = run_wrapped("metric_caching", a.label, ov, out_dir, patch=False)
    exp_tokens = set(toks) if toks is not None else set(navtest_tokens()[1])
    rep = {"label": a.label, "cache": cache, "rc": rc, "wall_s": round(wall, 1),
           "n_expected": len(exp_tokens), "per_log": bool(getattr(a, "per_log", False)),
           "n_per_log_runs": len(per_log_reps),
           "n_per_log_runs_failed": sum(1 for r in per_log_reps if r["rc"] != 0), "failures": []}
    meta = sorted(glob.glob(f"{cache}/metadata/*.csv"))
    rep["metadata_csvs"] = meta
    if len(meta) != 1:
        rep["failures"].append(f"expected exactly 1 metadata csv, found {len(meta)}")
    rows = []
    if meta:
        with open(meta[0], encoding="utf-8") as fh:
            rows = fh.read().splitlines()[1:]
    toks_c = [re.split(r"[\\/]", r.strip())[-2] for r in rows if r.strip()]
    rep["n_metadata_rows"] = len(rows)
    rep["n_distinct_tokens"] = len(set(toks_c))
    rep["n_missing"] = len(exp_tokens - set(toks_c))
    rep["n_unexpected"] = len(set(toks_c) - exp_tokens)
    rep["missing_tokens"] = sorted(exp_tokens - set(toks_c))[:50]
    rep["n_pkl_files_exist"] = sum(1 for r in rows if os.path.exists(r.strip()))
    rep["backslash_rows"] = sum(1 for r in rows if "\\" in r)
    rep["forward_slash_rows"] = sum(1 for r in rows if "/" in r)
    if rep["n_metadata_rows"] != len(exp_tokens) or rep["n_missing"] or rep["n_unexpected"] \
            or rep["n_distinct_tokens"] != rep["n_metadata_rows"] \
            or rep["n_pkl_files_exist"] != rep["n_metadata_rows"]:
        rep["failures"].append("C9 count guard: cached rows / tokens / files disagree with expected")
    rep["final_pass_skipped"] = bool(skip_final)
    text = open(log_path, encoding="utf-8", errors="replace").read() if os.path.exists(log_path) else ""
    m = re.findall(r"Failed features and targets: (\d+) out of (\d+)", text)
    m_ok = re.findall(r"All (\d+) features and targets were cached successfully", text)
    rep["log_all_ok_count"] = int(m_ok[-1]) if m_ok else None
    rep["log_failed"] = [int(x) for x in m[-1]] if m else None
    if rc != 0:
        rep["failures"].append(f"wrapper rc={rc}")
    rep["status"] = "FAIL" if rep["failures"] else "PASS"
    json.dump(rep, open(os.path.join(out_dir, f"{a.label}.counts.json"), "w", encoding="utf-8"),
              indent=1)
    print(json.dumps({k: rep[k] for k in ("label", "status", "n_expected", "n_metadata_rows",
                                          "n_missing", "backslash_rows", "wall_s", "failures")}))
    return 0 if rep["status"] == "PASS" else 1


CLUSTER_MAP = ("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
               "2026-09-19-navsim-estimator-and-route-leak/raw/cluster_maps/navtest.json")


def _tokens_per_log() -> dict:
    """{log_name: n navtest tokens} from W2's cluster map, so a COMPLETE log is skipped instead of
    re-imported (136 x ~25 s of import is an hour of nothing). Empty dict if the map is absent."""
    try:
        m = json.load(open(CLUSTER_MAP, encoding="utf-8"))["token_to_log_name"]
    except Exception:                                                  # noqa: BLE001
        return {}
    out: dict = {}
    for _, ln in m.items():
        out[ln] = out.get(ln, 0) + 1
    return out


def _wait_avail(min_mb: float, max_wait_min: float) -> float:
    """Block until system-available memory >= min_mb. Returns the seconds waited, or -waited when
    the window never opened (the caller then records NO_RAM_WINDOW rather than running blind)."""
    import psutil
    t0 = time.time()
    while True:
        if psutil.virtual_memory().available / 2**20 >= min_mb:
            return time.time() - t0
        if time.time() - t0 > max_wait_min * 60:
            return -(time.time() - t0)
        time.sleep(60)


def _n_pkl(cache: str) -> int:
    n = 0
    for _, _, files in os.walk(cache):
        n += sum(1 for f in files if f == "metric_cache.pkl")
    return n


def cmd_check_loader(a) -> int:
    """C7 (run in the NAVSIM venv with PYTHONPATH = v1.1 tree): unpatched loader vs E1's."""
    sys.path.insert(0, V11_TREE)
    import importlib.util
    import pathlib
    from navsim.common import dataloader as dl
    spec = importlib.util.spec_from_file_location(
        "e1w", "D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
               "2026-09-19-navsim-warmup-reference-epdms/code/navsim_win.py")
    e1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e1)
    cache = pathlib.Path(f"{EXP}/metric_cache_{a.cache_name}")
    rep = {"navsim_file": dl.__file__, "cache": str(cache)}
    try:
        got = dl.MetricCacheLoader(cache).metric_cache_paths
        rep["unpatched"] = {"raised": None, "n_keys": len(got),
                            "keys_equal_parent_dir": sum(
                                1 for k, v in got.items() if re.split(r"[\\/]", v)[-2] == k)}
    except Exception as e:                                           # noqa: BLE001
        rep["unpatched"] = {"raised": f"{type(e).__name__}: {e}"}
    m = e1._load_metric_cache_paths_patched(None, cache)
    rep["patched"] = {"n_keys": len(m), "keys_equal_parent_dir": sum(
        1 for k, v in m.items() if pathlib.PureWindowsPath(v).parent.name == k),
        "n_files_exist": sum(1 for v in m.values() if os.path.exists(v))}
    rep["C7_pass"] = bool(rep["unpatched"].get("raised", "").startswith("IndexError")
                          and rep["patched"]["n_keys"] == rep["patched"]["keys_equal_parent_dir"]
                          == rep["patched"]["n_files_exist"] > 0)
    out = os.path.join(RAW, f"C7_loader_{a.cache_name}.json")
    json.dump(rep, open(out, "w", encoding="utf-8"), indent=1)
    print(json.dumps(rep, indent=1))
    return 0 if rep["C7_pass"] else 1


def agent_overrides(arm: str, call_log: str) -> tuple[list, str]:
    if arm == "CV":
        return ["agent=constant_velocity_agent"], "navsim.agents.constant_velocity_agent.ConstantVelocityAgent"
    if arm == "HUMAN":
        return ["agent=human_agent"], "navsim.agents.human_agent.HumanAgent"
    if arm == "STOP":
        return ["agent=constant_velocity_agent", "agent._target_=w3_agents_v1.StopAgent"], \
            "w3_agents_v1.StopAgent"
    if arm.startswith("EGO_MLP:"):
        ck = arm.split(":", 1)[1]
        if not os.path.exists(ck):
            raise SystemExit(f"⛔ checkpoint missing: {ck}")
        return ["agent=ego_status_mlp_agent", f"agent.checkpoint_path={ck}"], \
            "navsim.agents.ego_status_mlp_agent.EgoStatusMLPAgent"
    if arm.startswith("SEAM:"):
        seam = arm.split(":", 1)[1]
        if not os.path.exists(seam):
            raise SystemExit(f"⛔ seam missing: {seam}")
        return ["agent=constant_velocity_agent", "agent._target_=w3_agents_v1.SeamAgentV1",
                f"+agent.seam_file={os.path.abspath(seam).replace(os.sep, '/')}",
                f"+agent.call_log={os.path.abspath(call_log).replace(os.sep, '/')}"], \
            "w3_agents_v1.SeamAgentV1"
    raise SystemExit(f"unknown arm {arm!r}")


def reconstruct_csv(rows_jsonl: str, dst: str) -> int:
    """Rebuild the devkit's per-token CSV from the wrapper's streamed rows (the aggregation
    hazard). The values are the ones ``pdm_score`` RETURNED — the hook records its result object —
    so the reconstruction is the same numbers, not a re-derivation. Returns the row count.
    Pinned bit-for-bit against a real devkit CSV by tests/test_reconstruct_rows.py."""
    recs = [json.loads(ln) for ln in open(rows_jsonl, encoding="utf-8") if ln.strip()]
    recs = [r for r in recs if "token" in r and "row" in r]
    if not recs:
        return 0
    with open(dst, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([""] + ["token", "valid"] + list(TERMS))
        for i, r in enumerate(recs):
            w.writerow([i, r["token"], True] + [r["row"][c] for c in TERMS])
        avg = [sum(r["row"][c] for r in recs) / len(recs) for c in TERMS]
        w.writerow([len(recs), "average", True] + avg)
    return len(recs)


def formula(r: dict) -> float:
    return (float(r["no_at_fault_collisions"]) * float(r["drivable_area_compliance"])
            * (5 * float(r["ego_progress"]) + 5 * float(r["time_to_collision_within_bound"])
               + 2 * float(r["comfort"])) / 12.0)


def cmd_score(a) -> int:
    ov, toks = split_overrides(a.tokens)
    cache = f"{EXP}/metric_cache_{a.cache_name}"
    run_out = f"{EXP}/runs/{a.label}"
    out_dir = os.path.join(RAW, a.label)
    os.makedirs(out_dir, exist_ok=True)
    call_log = os.path.join(out_dir, f"{a.label}.calls.jsonl")
    if os.path.exists(call_log):
        os.remove(call_log)
    aov, agent_cls = agent_overrides(a.arm, call_log)
    ov += [f"metric_cache_path={cache}", f"worker={a.worker}", f"experiment_name=w3_{a.label}",
           f"output_dir={run_out}"] + aov
    rc, log_path, wall = run_wrapped("pdm_score", a.label, ov, out_dir, patch=a.patch_loader,
                                     hashseed=a.hashseed, record_poses=a.record_poses)
    exp_tokens = set(toks) if toks is not None else set(navtest_tokens()[1])
    text = open(log_path, encoding="utf-8", errors="replace").read()
    m_ok = re.findall(r"Number of successful scenarios:\s*(\d+)", text)
    m_bad = re.findall(r"Number of failed scenarios:\s*(\d+)", text)
    m_res = re.findall(r"Results are stored in:\s*(.+?\.csv)", text)
    fails = []
    n_ok = int(m_ok[-1]) if m_ok else None
    n_bad = int(m_bad[-1]) if m_bad else None
    if n_ok is None:
        fails.append("log summary 'Number of successful scenarios' ABSENT — cannot confirm work")
    elif n_ok != len(exp_tokens):
        fails.append(f"C2 successful {n_ok} != expected {len(exp_tokens)}")
    if n_bad != 0:
        fails.append(f"C2 failed scenarios = {n_bad}")
    csv_src = m_res[-1].strip() if m_res else None
    if not csv_src or not os.path.exists(csv_src):
        cands = sorted(glob.glob(f"{run_out}/*.csv"), key=os.path.getmtime)
        csv_src = cands[-1] if cands else None
    rows_jsonl = os.path.join(out_dir, f"{a.label}_rows.jsonl")
    reconstructed = False
    if (not csv_src or not os.path.exists(csv_src)) and os.path.exists(rows_jsonl):
        # the aggregation hazard: the devkit scored every token and then died before writing the
        # CSV. The streamed per-token rows are the SAME numbers (the hook records what pdm_score
        # returned), so the run is recovered instead of repeated — and labelled as reconstructed.
        dst = os.path.join(run_out, f"{a.label}_RECONSTRUCTED_from_rows_jsonl.csv")
        os.makedirs(run_out, exist_ok=True)
        n = reconstruct_csv(rows_jsonl, dst)
        if n:
            csv_src = dst
            reconstructed = True
    rep = {"label": a.label, "arm": a.arm, "agent_class": agent_cls, "rc": rc, "wall_s": round(wall, 1),
           "patch_loader": bool(a.patch_loader),
           "hashseed": a.hashseed, "cache": cache, "n_expected": len(exp_tokens),
           "log_successful": n_ok, "log_failed": n_bad, "csv_src": csv_src,
           "csv_reconstructed_from_rows_jsonl": reconstructed}
    summ = None
    if csv_src and os.path.exists(csv_src):
        dst = os.path.join(out_dir, f"{a.label}.csv")
        shutil.copyfile(csv_src, dst)
        rep["csv"] = os.path.relpath(dst, PKG).replace(os.sep, "/")
        rep["csv_sha256"] = sha256_file(dst)
        rows = list(csv.DictReader(open(dst, encoding="utf-8")))
        tok_rows = [r for r in rows if r["token"] != "average"]
        avg_rows = [r for r in rows if r["token"] == "average"]
        valid = [r for r in tok_rows if r["valid"] in ("True", "true", "1")]
        got = {r["token"] for r in tok_rows}
        rep.update(csv_token_rows=len(tok_rows), csv_valid_rows=len(valid),
                   csv_average_rows=len(avg_rows),
                   C3_missing=len(exp_tokens - got), C3_unexpected=len(got - exp_tokens))
        if len(tok_rows) != len(exp_tokens) or len(valid) != len(exp_tokens) or len(avg_rows) != 1:
            fails.append(f"C2 CSV rows {len(tok_rows)} / valid {len(valid)} / average "
                         f"{len(avg_rows)} != expected {len(exp_tokens)} / 1")
        if rep["C3_missing"] or rep["C3_unexpected"]:
            fails.append(f"C3 token set differs: missing {rep['C3_missing']}, "
                         f"unexpected {rep['C3_unexpected']}")
        dmax = max((abs(formula(r) - float(r["score"])) for r in valid), default=float("nan"))
        rep["C1_max_abs_delta"] = dmax
        if not (dmax <= 1e-12):
            fails.append(f"C1 PDMS formula max |Δ| = {dmax}")
        n = len(valid)
        summ = {SHORT[t]: (sum(float(r[t]) for r in valid) / n if n else None) for t in TERMS}
        if avg_rows:
            summ["devkit_average_row_score"] = float(avg_rows[0]["score"])
            summ["abs_delta_mean_vs_average_row"] = abs(summ["PDMS"] - float(avg_rows[0]["score"]))
        summ["n"] = n
        rep["summary"] = summ
        rep["summary_x100_4dp"] = {k: round(100 * v, 4) for k, v in summ.items()
                                   if k in SHORT.values() and v is not None}
    else:
        fails.append("no result CSV found")
    man = os.path.join(out_dir, f"{a.label}_manifest.json")
    if os.path.exists(man):
        mj = json.load(open(man, encoding="utf-8"))
        rep["imported_from"] = mj.get("imported_from")
        rep["patches"] = [p["name"] for p in mj.get("patches", [])]
        rep["resources"] = {k: mj.get("resources", {}).get(k) for k in
                            ("wall_s", "peak_rss_mb", "min_system_available_mb")}
    if a.arm.startswith("SEAM:"):
        calls = {}
        if os.path.exists(call_log):
            for ln in open(call_log, encoding="utf-8"):
                r = json.loads(ln)
                if r.get("event") == "call":
                    calls[r["source"]] = calls.get(r["source"], 0) + 1
        rep["agent_calls"] = calls
        if calls.get("seam", 0) != len(exp_tokens) or any(v for k, v in calls.items() if k != "seam"):
            fails.append(f"seam calls {calls} != {len(exp_tokens)} seam rows")
    if rc != 0:
        fails.append(f"wrapper rc={rc}")
    rep["failures"] = fails
    rep["status"] = "FAIL" if fails else "PASS"
    json.dump(rep, open(os.path.join(out_dir, f"{a.label}.counts.json"), "w", encoding="utf-8"),
              indent=1, default=str)
    print(json.dumps({k: rep.get(k) for k in ("label", "status", "log_successful", "log_failed",
                                              "csv_valid_rows", "C1_max_abs_delta",
                                              "summary_x100_4dp", "wall_s", "failures")},
                     default=str))
    return 0 if not fails else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select-smoke")
    s.add_argument("--out", default=os.path.join(RAW, "smoke_tokens.json"))
    s.add_argument("--n-logs", type=int, default=4)
    s.add_argument("--per-log", type=int, default=5)
    c = sub.add_parser("cache")
    c.add_argument("--label", required=True)
    c.add_argument("--tokens", default=None)
    c.add_argument("--cache-name", required=True)
    c.add_argument("--worker", default="sequential")
    c.add_argument("--min-avail-mb", type=float, default=3500.0)
    c.add_argument("--max-wait-min", type=float, default=180.0)
    c.add_argument("--log-tries", type=int, default=3)
    c.add_argument("--logs-per-process", type=int, default=1,
                   help="logs per devkit process in --per-log mode: amortises the python import "
                        "and the per-process MAP LOAD (40-125 s each under contention) without "
                        "adding load")
    c.add_argument("--per-log", action="store_true",
                   help="one devkit process per navtest log (memory-robust, resumable), then one "
                        "final full pass so the DEVKIT writes the metadata CSV")
    k = sub.add_parser("check-loader")
    k.add_argument("--cache-name", required=True)
    r = sub.add_parser("score")
    r.add_argument("--label", required=True)
    r.add_argument("--arm", required=True)
    r.add_argument("--tokens", default=None)
    r.add_argument("--cache-name", required=True)
    r.add_argument("--worker", default="sequential")
    r.add_argument("--hashseed", default="1")
    r.add_argument("--record-poses", action="store_true")
    r.add_argument("--patch-loader", action="store_true",
                   help="E1's separator patch — pass ONLY after control C7 showed the defect")
    a = ap.parse_args(argv)
    return {"select-smoke": cmd_select_smoke, "cache": cmd_cache, "check-loader": cmd_check_loader,
            "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
