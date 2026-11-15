#!/usr/bin/env python3
"""W-A7 launch-blocker read for P5-C (PREREG_P5C_A2.md section 6, frozen).  CPU only.  No git writes.  Resumable.

Question: does applying A7 (refe/planner.py `repair_last_heading`: last pose heading := previous pose heading) to a label change its
tracked PDMS?  Population: the card's 1,018 navtrain held-out rows (the rows of p5c_labels.py `control` = the P5-S npz rows).
Labels: the PINNED dual labeller (blob 50b9f0db) at A1 settings -> Tc20s and P5-R (lambda* 3.0).  P5-R decides, Tc20s is reported only.
Scorer: NAVSIM PDM scorer + LQR tracker exactly as TF-1 (metric cache parts rebuilt from the nuPlan DB by
navsim_pdm_targets.metric_cache_parts; navtrain has no stored harness cache) through transp_score.score_arms (blob 3cbdd118).
A7 is applied with the programme's own `repair_last_heading` imported from refe/planner.py, never re-implemented.
(transp_score.py's `--a7` re-implements the same line inline and does not call repair_last_heading, so it is NOT used.)

Units: PDMS x 100 (percentage points), as TF-1 and the card's bar B = 0.7031.  The -0.10 bar is therefore -0.10 pp = -0.001 on the 0..1 scale.

stages (default all, each resumable):  prep -> labels -> score -> analyze
  python w_a7_read.py --run <dir> [--limit N --smoke] [--stage all|prep|labels|score|analyze]
Final log line: ZZW_A7 DONE <row>    (row in W-A7-PASS | W-A7-REBUILD | INVALID)

W-A7-RJ5 (PREREG_P5C_A3 section 5; A3.1): `--traj-source <RJ5 join> --expect-join-sha256 H` reads the SAME 1,018 rows with each row's label input traj `u(r)` := its RJ5 traj
(the pinned labeller runs on a FEED bank, traj := u).  A2 section 6's definition, estimator and bar are unchanged; only control (i) is replaced by (i'):
  (i')  the W-A7-RJ5 no-A7 P5-R labels reproduce the C1 table's labels on the same rows within R-TOL-1/2/3 (the pinned gates' own functions; `--c1-table`, `--c1-records`, `--gates-module`);
  (i'') the instrument is validated by the bank-traj W-A7 run's control (i), which must be VALID (`--bank-run-result`).  Controls (ii)-(iv) are unchanged.
REFUSED (ZZW_A7 STOP, exit 2, never labelled a W-A7-RJ5 read): a traj source that is not an RJ5 join (a bank file), a join whose sha256 is not the pinned one, a row without an RJ5 traj, a join row keyed to another bank traj,
a feed in which no traj moved, and labels computed from any traj other than the fed one (e.g. a bank-traj labels directory re-used).  Without --traj-source nothing changes (the bank-traj read, byte-identical outputs).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("REFE_SIM_HZ", "10")
os.environ.setdefault("NUPLAN_MAPS_ROOT", "E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SG = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107"
PKG = "E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
REFE = PKG + "/refe"
NAVSIM = "F:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
DB_ROOT = "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits"
PY = "C:/Users/Admin/navsim-crun/venv/Scripts/python.exe"
DUAL = SG + "/p5c/a1/p5c_dual_labels.py"
P5R_DIR = SG + "/p5r"
LAMBDA_JSON = P5R_DIR + "/raw/lambda_star.json"
NPZ = PKG + "/raw/2026-10-11-p5s-smoothed/p5s_cubic.npz"
TF1_ROWS = [SG + f"/teacher_fidelity/tf1/rows/tf1_{s}.jsonl" for s in ("s0", "s1", "s2", "s3", "s4r")]
INSTR = HERE + "/instr"

SEED, B_DRAWS = 20260927, 10000
BAR_POINT_PP, BAR_NOTE = -0.10, "-0.10 PDMS x100 (= -0.001 on the 0..1 scale)"
MIN_FREE_VM_GB = 5.0  # MM 2026-11-14 06:45Z: lowered from 8.0 (the box sits at 6-7 GB; the 8 GB was the brief's guess, not a measurement)
ARMS = ["R_noA7", "R_A7", "T_noA7", "T_A7", "RED_R_h19_plus05", "IDN_R_pre", "IDN_R_A7"]
GATES_PY = SG + "/p5c/a1/p5c_table_gates.py"          # the pinned gates module (A1 pin 5c2f49cf), imported by path for control (i')
GATES_BLOB = "5c2f49cf485241df4d7e7949ed629a04fd324d5c"
JOIN_KEYS = ("log_name", "token", "rank", "bank_key", "bank_traj_sha256", "status")
PIN_BLOBS = {"dual_labeller": ("p5c_dual_labels.py", DUAL, "50b9f0dbaa22563239d6c711e9f7dab846e1cc69"),
             "p5r_labels": ("p5r_labels.py", P5R_DIR + "/p5r_labels.py", "16d2aaf05eea5c00bd081628e07b02b7dc7f480f"),
             "p5r_run": ("p5r_run.py", P5R_DIR + "/p5r_run.py", "d0a4084b7187005df3e966a1ba0132a42a61e7d2"),
             "p5_target_labels": ("p5_target_labels.py", PKG + "/eval/p5_target_labels.py", "294644df61ca479097971ebc92b033e24f3b7019"),
             "planner": ("planner.py", REFE + "/planner.py", "6fb6598a2ecf230153a4d7adfb790afc30728070"),
             "transp_score": ("transp_score.py", INSTR + "/transp_score.py", "3cbdd118fa82790caf7c38bdd8ce6f92320c575b"),
             "navsim_dac": ("navsim_dac.py", REFE + "/navsim_dac.py", "1884fee7"),
             "navsim_pdm_targets": ("navsim_pdm_targets.py", REFE + "/navsim_pdm_targets.py", "f778e56e"),
             "navtrain_scenarios": ("navtrain_scenarios.py", REFE + "/navtrain_scenarios.py", "848fe684")}


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def blob(path):
    """git blob id, no filters (CRLF-safe: git hash-object --no-filters)."""
    out = subprocess.run(["git", "hash-object", "--no-filters", path], capture_output=True, text=True, encoding="utf-8")
    return out.stdout.strip()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def check_pins():
    bad = {}
    for k, (nm, p, want) in PIN_BLOBS.items():
        got = blob(p)
        if not got.startswith(want):
            bad[k] = {"path": p, "want": want, "got": got}
    return bad


def free_vm_gb():
    """FreeVirtualMemory (KB) of Win32_OperatingSystem, as the brief prescribes."""
    out = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory"],
                         capture_output=True, text=True, encoding="utf-8", timeout=120)
    return float(out.stdout.strip()) / 1048576.0


def wait_free_vm(what, max_wait_s=6 * 3600):
    t0 = time.time()
    while True:
        try:
            g = free_vm_gb()
        except Exception as e:  # noqa: BLE001
            log(f"VM-PROBE-FAIL {e!r}")
            return
        if g >= MIN_FREE_VM_GB:
            return
        log(f"VM-WAIT before {what}: free virtual memory {g:.1f} GB < {MIN_FREE_VM_GB} GB")
        if time.time() - t0 > max_wait_s:
            raise SystemExit(f"ZZW_A7 STOP vm-wait-timeout before {what}")
        time.sleep(30)


# ------------------------------------------------------------------ population
def load_population(limit=0):
    Z = np.load(NPZ, allow_pickle=True)
    logs, toks, lab = Z["logs"].tolist(), Z["tokens"].tolist(), Z["labels20_t20_tc20_tw"]
    rows = [{"log": str(l), "token": str(t), "traj": np.asarray(lab[i, 0], np.float64), "tc20s_banked": np.asarray(lab[i, 1], np.float64)}
            for i, (l, t) in enumerate(zip(logs, toks))]
    rows.sort(key=lambda r: (r["log"], r["token"]))
    if limit:
        rows = rows[:: max(1, len(rows) // limit)][:limit]
    return rows


def prep(run, rows):
    d = f"{run}/bank"
    os.makedirs(d, exist_ok=True)
    with open(f"{d}/targets_rank0.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps({"log_name": r["log"], "token": r["token"], "rank": 0, "traj": r["traj"].tolist()}) + "\n")
    open(f"{d}/targets_rank1.jsonl", "w", encoding="utf-8", newline="\n").close()
    json.dump({"n": len(rows), "logs": len({r["log"] for r in rows}), "source": NPZ,
               "note": "traj = labels20_t20_tc20_tw[:,0] of the P5-S npz (the dev-box bank copy was deleted 2026-11-12); rank 0"},
              open(f"{run}/prep.json", "w"), indent=1)


# ------------------------------------------------------------------ labels (pinned labeller, unmodified, run as a subprocess)
def run_labeller(run):
    wait_free_vm("labeller")
    env = dict(os.environ)
    env.update({"P5C_PKG": PKG, "P5C_BANKDIR": f"{run}/bank", "P5C_DBROOT": DB_ROOT, "P5R_DIR": P5R_DIR, "P5R_LAMBDA_JSON": LAMBDA_JSON,
                "PYTHONPATH": f"{NAVSIM};{REFE}", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "CUDA_VISIBLE_DEVICES": "-1", "PYTHONIOENCODING": "utf-8"})
    out = f"{run}/labels"
    if os.path.exists(f"{out}/DONE_s0of1.json"):
        log("labels already DONE")
        return
    with open(f"{run}/labeller.log", "a", encoding="utf-8") as lf:
        rc = subprocess.run([PY, DUAL, "label", "--shard", "0", "--nshard", "1", "--out", out], env=env, stdout=lf, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL).returncode
    if rc != 0 or not os.path.exists(f"{out}/DONE_s0of1.json"):
        raise SystemExit(f"ZZW_A7 STOP labeller rc={rc} (see {run}/labeller.log)")


def load_labels(run):
    out = {}
    fp = f"{run}/labels/labels_s0of1.jsonl"
    for ln in open(fp, encoding="utf-8"):
        r = json.loads(ln)
        out[(r["log"], r["token"])] = r
    return out


# ------------------------------------------------------------------ W-A7-RJ5 (A3 section 5)
class Rj5Refused(Exception):
    """a refusal of the W-A7-RJ5 read: printed as `ZZW_A7 STOP`, exit 2, never a W-A7-RJ5 outcome"""


def traj_sha256(traj):
    """== p5c_dual_labels.traj_sha == traj_label_table.traj_sha256: float64 little-endian bytes of the [20,3] array"""
    return hashlib.sha256(np.ascontiguousarray(np.asarray(traj, np.float64)).tobytes()).hexdigest()


def read_rj5_join(path, expect_sha256):
    """-> {(log, token): join row}.  The traj source of W-A7-RJ5 is the RJ5 join, pinned by sha256; a bank file (rows with a `traj` and no `status`) is REFUSED."""
    import gzip
    if not expect_sha256:
        raise Rj5Refused("--expect-join-sha256 is required with --traj-source: the RJ5 source is pinned")
    got = sha256(path)
    if got != expect_sha256:
        raise Rj5Refused(f"traj source {path}: sha256 {got[:12]}... is not the pinned {expect_sha256[:12]}...")
    out = {}
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        for n, ln in enumerate(f, 1):
            if not ln.strip():
                continue
            d = json.loads(ln)
            lack = [k for k in JOIN_KEYS if k not in d]
            if lack:
                kind = "a BANK file (rows carry `traj`)" if "traj" in d else "not an RJ5 join"
                raise Rj5Refused(f"traj source {path}, line {n}: lacks {lack}: this is {kind}; W-A7-RJ5 reads the RJ5 join, a run whose traj source is the bank is the bank-traj read and is refused as W-A7-RJ5")
            k = (d["log_name"], d["token"])
            if k in out:
                raise Rj5Refused(f"traj source: duplicate key {k}")
            out[k] = d
    if not out:
        raise Rj5Refused(f"traj source {path} holds no rows")
    return out


def rj5_feed_trajs(rows, join):
    """-> ([fed traj [20,3] per population row], stats).  Every population row must be a rank-0 key of the join WITH an RJ5 traj, keyed to the row's own bank traj (A3 section 5: 1,018/1,018)."""
    fed, moved, src = [], 0, {}
    for r in rows:
        k = (r["log"], r["token"])
        j = join.get(k)
        where = f"W-A7 row {k[0]}|{k[1]}"
        if j is None:
            raise Rj5Refused(f"{where}: not in the traj source")
        if int(j["rank"]) != 0:
            raise Rj5Refused(f"{where}: join rank {j['rank']} (RJ5 rolls exist for rank-0 keys only)")
        bsha = traj_sha256(r["traj"])
        if j["bank_traj_sha256"] != bsha or j["bank_key"] != f"{k[0]}|{k[1]}|0|{bsha}":
            raise Rj5Refused(f"{where}: the join row is keyed to another bank traj ({str(j['bank_traj_sha256'])[:12]}... vs {bsha[:12]}...): wrong-key override REFUSED")
        if j["status"] != "rj5":
            raise Rj5Refused(f"{where}: status {j['status']!r}, not rj5 (A3 section 5: every W-A7 row carries an RJ5 traj)")
        t = np.asarray(j.get("rj5_traj"), np.float64)
        if t.shape != np.asarray(r["traj"]).shape or not np.isfinite(t).all():
            raise Rj5Refused(f"{where}: the RJ5 traj is not a finite {np.asarray(r['traj']).shape} array")
        if traj_sha256(t) != j.get("rj5_traj_sha256"):
            raise Rj5Refused(f"{where}: sha256 of rj5_traj is not its recorded rj5_traj_sha256")
        moved += int(not np.array_equal(t, np.asarray(r["traj"], np.float64)))
        src[j.get("source")] = src.get(j.get("source"), 0) + 1
        fed.append(t)
    if moved == 0:
        raise Rj5Refused("every fed traj equals its bank traj: the traj source is the bank, not RJ5: refused as W-A7-RJ5")
    return fed, {"n_rows": len(rows), "n_traj_moved": moved, "n_traj_identical_to_bank": len(rows) - moved, "rj5_sources": dict(sorted(src.items(), key=lambda kv: str(kv[0])))}


def prep_rj5(run, rows, join_path, expect_sha256):
    """the feed bank (`run/bank`, traj := RJ5: what the pinned labeller reads), the real bank rows (`run/bank_real`, bank trajs: control (i') and the raw fallback), and the fed/bank sha map"""
    join = read_rj5_join(join_path, expect_sha256)
    fed, st = rj5_feed_trajs(rows, join)
    for d, trajs in ((f"{run}/bank", fed), (f"{run}/bank_real", [r["traj"] for r in rows])):
        os.makedirs(d, exist_ok=True)
        with open(f"{d}/targets_rank0.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for r, t in zip(rows, trajs):
                f.write(json.dumps({"log_name": r["log"], "token": r["token"], "rank": 0, "traj": np.asarray(t, np.float64).tolist()}) + "\n")
        open(f"{d}/targets_rank1.jsonl", "w", encoding="utf-8", newline="\n").close()
    fmap = {f"{r['log']}|{r['token']}": [traj_sha256(t), traj_sha256(r["traj"])] for r, t in zip(rows, fed)}
    json.dump(fmap, open(f"{run}/feed_sha_map.json", "w", encoding="utf-8", newline="\n"), indent=0, sort_keys=True)
    json.dump({"n": len(rows), "logs": len({r["log"] for r in rows}), "source": NPZ, "read_kind": "W-A7-RJ5", "traj_source": os.path.basename(join_path), "join_sha256": expect_sha256, **st,
               "note": "bank/ = the FEED bank (traj := RJ5), what the pinned labeller reads; bank_real/ = the bank trajs (npz labels20_t20_tc20_tw[:,0]); rank 0"},
              open(f"{run}/prep.json", "w", encoding="utf-8", newline="\n"), indent=1)
    return st


def check_labels_fed(run, rows):
    """every label record must have been computed from the FED traj (its traj_sha256 == the fed sha).  A labels directory from the bank-traj run carries the BANK sha: REFUSED"""
    fmap = json.load(open(f"{run}/feed_sha_map.json", encoding="utf-8"))
    labels = load_labels(run)
    bad = []
    for r in rows:
        k = (r["log"], r["token"])
        L = labels.get(k)
        if L is None:
            bad.append((k, "no label record"))
        elif L.get("traj_sha256") != fmap[f"{k[0]}|{k[1]}"][0]:
            bad.append((k, "labelled from " + str(L.get("traj_sha256"))[:12] + (" (the BANK traj)" if L.get("traj_sha256") == fmap[f"{k[0]}|{k[1]}"][1] else "")))
    if bad:
        raise Rj5Refused(f"{len(bad)} of {len(rows)} label records were not computed from the fed RJ5 traj (e.g. {bad[:2]}): the traj source of this run is not RJ5, refused as W-A7-RJ5")


def check_bank_run_result(path):
    """the bank-traj W-A7 run validates the INSTRUMENT: its control (i) must be VALID (A3 section 5).  Its delta is context only."""
    r = json.load(open(path, encoding="utf-8"))
    c = (r.get("controls") or {}).get("i_reproduction_tf1_tc20s_tracked") or {}
    ok = bool(c.get("pass") is True and r.get("controls_all_pass") is True and not r.get("smoke") and r.get("outcome_row") in ("W-A7-PASS", "W-A7-REBUILD"))
    return {"pass": ok, "path": path, "sha256": sha256(path), "outcome_row": r.get("outcome_row"), "control_i": c, "context_delta_P5R_pp_point_lo_hi": (r.get("delta_pp_point_lo_hi") or {}).get("P5-R"),
            "role": "validates the instrument (control i on bank trajs); its delta is context, it informs no rule"}


def _import_by_path(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def control_i_prime(run, rows, labels, c1_table, c1_records, gates_path):
    """(i'): the no-A7 P5-R labels of THIS run reproduce the C1 table's labels on the same rows within R-TOL-1/2/3 (sim8 identical >= 99.5 %; per-row max abs diff median <= 1e-3, p99 <= 5e-3;
    TB-6 ratios within +-0.01 and sim8-fail rate within +-0.25 pp), by the pinned gates' own `_p5r_vs_control`.  Also: the table's label equals its record's p5r on every row (the records ARE the table's)."""
    import gzip
    G = _import_by_path("p5c_table_gates_pinned_wa7", gates_path)
    gblob = blob(gates_path)
    if gblob != GATES_BLOB:
        return {"pass": False, "reason": f"the gates module blob {gblob} is not the pinned {GATES_BLOB}"}
    bank = G.load_bank([f"{run}/bank_real"])
    want = {(r["log"], r["token"]) for r in rows}
    recs = {}
    with open(c1_records, encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            d = json.loads(ln)
            if int(d["rank"]) == 0 and (d["log"], d["token"]) in want:
                recs[G.triple(d["log"], d["token"], 0)] = d
    with gzip.open(c1_table, "rt", encoding="utf-8") as f:
        table = json.load(f)["table"]
    bsha = {(r["log"], r["token"]): traj_sha256(r["traj"]) for r in rows}
    n_tab_bad = n_no_rec = 0
    btraj = {(r["log"], r["token"]): np.asarray(r["traj"], np.float64) for r in rows}
    for k in want:
        d = recs.get(G.triple(k[0], k[1], 0))
        if d is None:
            n_no_rec += 1
            continue
        lab = table.get(f"{k[0]}|{k[1]}|0|{bsha[k]}")
        expect = btraj[k] if ("p5r_error" in d or "p5r" not in d) else np.asarray(d["p5r"], np.float64)      # a raw-fallback record's table label is the bank traj
        n_tab_bad += int(lab is None or not np.array_equal(np.asarray(lab, np.float64), expect))
    M = G.merge_p5r(bank, recs)
    ctl = {}
    for (lg, tok), L in labels.items():
        if (lg, tok) in want and "p5r" in L and "p5r_error" not in L and "p5r_sim8" in L:
            ctl[G.triple(lg, tok, 0)] = {"label": L["p5r"], "sim8": bool(L["p5r_sim8"]), "learn": None, "raw_learn": None}
    ok_p, det = G._p5r_vs_control(bank, M, ctl, None)
    n_err_here = sum(1 for k in want if "p5r_error" in labels.get(k, {}))
    passed = bool(ok_p and det["n_common_rows"] == len(ctl) and len(ctl) > 0 and n_tab_bad == 0 and n_no_rec == 0 and n_err_here == 0)
    return {"pass": passed, "n_population": len(want), "n_control_rows": len(ctl), "n_common_rows": det["n_common_rows"], "n_rows_without_c1_record": n_no_rec, "n_table_label_not_its_record": n_tab_bad,
            "n_p5r_error_rows_in_this_run": n_err_here, "failed_criteria": det["failed_criteria"], "per_rank": det["per_rank"], "criteria": det["criteria"], "gates_module_blob": gblob,
            "c1_table_sha256": sha256(c1_table), "c1_records_sha256": sha256(c1_records),
            "rule": "A3 section 5 (i'): R-TOL-1/2/3 of A1/A2, thresholds unchanged, by the pinned gates' _p5r_vs_control"}


# ------------------------------------------------------------------ plans
def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def make_plans(traj, tc20s, p5r, a7):
    """-> {arm: plan20 [20,3]}.  `a7` is the A7 function (repair_last_heading).  p5r / tc20s None = raw fallback (label := traj)."""
    lp = np.asarray(p5r if p5r is not None else traj, np.float64)
    lt = np.asarray(tc20s if tc20s is not None else traj, np.float64)
    red = lp.copy()
    red[19, 2] = wrap(red[19, 2] + 0.5)
    pre = lp.copy()
    pre[19, 2] = pre[18, 2]
    return {"R_noA7": lp, "R_A7": a7(lp), "T_noA7": lt, "T_A7": a7(lt), "RED_R_h19_plus05": red, "IDN_R_pre": pre, "IDN_R_A7": a7(pre)}


def score_row(TS, ND, parts, plans):
    res, _ = TS.score_arms(parts, [ND.to_navsim(plans[a]) for a in ARMS], [])
    return {a: r for a, r in zip(ARMS, res)}


def done_tokens(fp):
    d = set()
    if os.path.exists(fp):
        for ln in open(fp, encoding="utf-8"):
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if "error" not in r:
                d.add((r["log"], r["token"]))
    return d


def score_stage(run, rows, labels, rerun_every=10):
    sys.path[:0] = [NAVSIM, REFE, INSTR]
    import navtrain_scenarios as NS
    import navsim_dac as ND
    import navsim_pdm_targets as PT
    from planner import repair_last_heading
    import transp_score as TS
    fp = f"{run}/scores.jsonl"
    done = done_tokens(fp)
    by = {}
    for i, r in enumerate(rows):
        by.setdefault(r["log"], []).append((i, r))
    dbs = NS.index_dbs(db_root=DB_ROOT)
    if os.path.exists(fp) and os.path.getsize(fp) > 0:
        with open(fp, "rb") as fb:
            fb.seek(-1, 2)
            torn = fb.read(1) != b"\n"
        if torn:
            open(fp, "a", encoding="utf-8").write("\n")
    t0, n = time.time(), 0
    with open(fp, "a", encoding="utf-8", newline="\n") as fo:
        for lg in sorted(by):
            todo = [(i, r) for i, r in by[lg] if (lg, r["token"]) not in done]
            if not todo:
                continue
            wait_free_vm(f"score {lg}")
            if not os.path.exists(dbs.get(lg, "")):
                for i, r in todo:
                    fo.write(json.dumps({"log": lg, "token": r["token"], "error": "no local DB"}) + "\n")
                continue
            scs = {sc._initial_lidar_token: sc for sc in NS.build_scenarios_for_log(dbs[lg], [r["token"] for _, r in todo])}
            for i, r in todo:
                tok = r["token"]
                rec = {"log": lg, "token": tok}
                ts = time.time()
                try:
                    L = labels[(lg, tok)]
                    p5r = None if "p5r_error" in L else np.asarray(L["p5r"], np.float64)
                    tc = None if "error" in L else np.asarray(L["tc20s"], np.float64)
                    plans = make_plans(r["traj"], tc, p5r, repair_last_heading)
                    rec["label_src"] = {"p5r": "p5r" if p5r is not None else "raw_fallback", "tc20s": "tc20s" if tc is not None else "raw_fallback"}
                    rec["idn_plan_equal"] = bool(np.array_equal(plans["IDN_R_pre"], plans["IDN_R_A7"]))
                    rec["tc20s_vs_banked_maxabs"] = None if tc is None else float(np.abs(tc - r["tc20s_banked"]).max())
                    rec["h19_h18_gap_p5r"] = float(abs(wrap(plans["R_noA7"][19, 2] - plans["R_noA7"][18, 2])))
                    parts = PT.metric_cache_parts(scs[tok])
                    rec["arms"] = score_row(TS, ND, parts, plans)
                    if rerun_every and i % rerun_every == 0:                      # control (iv): rebuild the parts and score again
                        parts2 = PT.metric_cache_parts(scs[tok])
                        rec["rerun"] = score_row(TS, ND, parts2, plans)
                    rec["t_s"] = round(time.time() - ts, 3)
                except Exception as e:  # noqa: BLE001
                    rec = {"log": lg, "token": tok, "error": f"{type(e).__name__}: {str(e)[:200]}"}
                fo.write(json.dumps(rec) + "\n")
                fo.flush()
                n += 1
                if n % 25 == 0:
                    log(f"scored {n}/{len(todo) + len(done)} ({time.time() - t0:.0f}s)")
    log(f"score stage done: {n} rows this call, {time.time() - t0:.0f}s")


# ------------------------------------------------------------------ analysis
def bootstrap_delta(deltas, logs, seed=SEED, b=B_DRAWS):
    """TF-1 / E2 estimator verbatim: log-cluster bootstrap, ratio mean, draws = rng.integers(0, L, (B, L)), 95 % percentile.
    `deltas` may be a dict {name: vector}: every vector shares the SAME draws (paired).  Returns {name: [mean, lo, hi]} in PDMS x 100."""
    ul, inv = np.unique(np.asarray(logs), return_inverse=True)
    L = len(ul)
    rng = np.random.default_rng(seed)
    cnt = np.bincount(inv, minlength=L).astype(float)
    draws = rng.integers(0, L, size=(b, L))
    den = cnt[draws].sum(1)
    out = {}
    for nm, v in deltas.items():
        v = np.asarray(v, float)
        s = np.bincount(inv, weights=v, minlength=L)
        e = s[draws].sum(1) / den
        out[nm] = [100 * float(v.mean()), 100 * float(np.percentile(e, 2.5)), 100 * float(np.percentile(e, 97.5))]
    return out


def outcome_row(r_delta, controls_ok):
    """A2 section 6: INVALID if any control fails; REBUILD iff CI upper bound < 0 AND point <= -0.10 (PDMS x 100); else PASS."""
    if not controls_ok:
        return "INVALID"
    pt, lo, hi = r_delta
    return "W-A7-REBUILD" if (hi < 0 and pt <= BAR_POINT_PP) else "W-A7-PASS"


def banked_tc20s():
    d = {}
    for fp in TF1_ROWS:
        for ln in open(fp, encoding="utf-8"):
            r = json.loads(ln)
            if "error" not in r:
                d[(r["log"], r["token"])] = r["Tc20s_tracked"]["pdms"]
    return d


def analyze(run, rows, smoke=False, banked=None, rj5=None):
    sc = [json.loads(ln) for ln in open(f"{run}/scores.jsonl", encoding="utf-8")]
    err = [r for r in sc if "error" in r]
    ok = {(r["log"], r["token"]): r for r in sc if "error" not in r}
    keys = [(r["log"], r["token"]) for r in rows]
    have = [k for k in keys if k in ok]
    rr = [ok[k] for k in have]
    logs = [k[0] for k in have]
    P = lambda arm: np.array([r["arms"][arm]["pdms"] for r in rr])                      # noqa: E731
    banked = (banked_tc20s() if banked is None else banked) if rj5 is None else {}
    ctl = {}
    # (0) coverage: every population row scored, no error rows
    ctl["coverage"] = {"n_population": len(keys), "n_scored": len(have), "n_error_rows": len(err), "pass": len(have) == len(keys) and not err}
    if rj5 is None:
        # (i) reproduction of banked TF1 tracked Tc20s per row
        cmp_ = [(abs(ok[k]["arms"]["T_noA7"]["pdms"] - banked[k]), k) for k in have if k in banked]
        mx = max((d for d, _ in cmp_), default=float("nan"))
        ctl["i_reproduction_tf1_tc20s_tracked"] = {"n_compared": len(cmp_), "max_abs_diff": mx, "n_over_1e-6": sum(d > 1e-6 for d, _ in cmp_),
                                                  "tol": 1e-6, "pass": len(cmp_) > 0 and mx <= 1e-6 and len(cmp_) == len(have)}
    else:
        # (i') replaces (i) on RJ5 inputs (A3 section 5): no banked TF-1 per-row PDMS exists for RJ5 labels.  Never silently skipped: a missing input is a FAILED control (INVALID)
        need = [k for k in ("c1_table", "c1_records", "gates", "bank_run_result") if not rj5.get(k)]
        if need:
            ctl["i_prime_c1_table_reproduction"] = {"pass": False, "reason": f"inputs not given: {need}"}
            ctl["i_bank_run_instrument_validity"] = {"pass": False, "reason": "bank-run result not given"}
        else:
            ctl["i_prime_c1_table_reproduction"] = control_i_prime(run, rows, load_labels(run), rj5["c1_table"], rj5["c1_records"], rj5["gates"])
            ctl["i_bank_run_instrument_validity"] = check_bank_run_result(rj5["bank_run_result"])
    # (ii) RED arm: +0.5 rad on index 19's heading must change PDMS on >= 1 row
    nchg = int(np.sum(P("RED_R_h19_plus05") != P("R_noA7")))
    ctl["ii_red_arm"] = {"n_rows_changed": nchg, "need": 1, "pass": nchg >= 1}
    # (iii) identity: A7 on a label whose idx-19 heading == idx-18's is bit-identical (plan arrays and every sub-score)
    same_plan = all(r["idn_plan_equal"] for r in rr)
    same_sc = all(r["arms"]["IDN_R_pre"] == r["arms"]["IDN_R_A7"] for r in rr)
    ctl["iii_identity"] = {"plans_bit_equal_rows": int(sum(r["idn_plan_equal"] for r in rr)), "scores_bit_equal_rows": int(sum(r["arms"]["IDN_R_pre"] == r["arms"]["IDN_R_A7"] for r in rr)),
                           "n": len(rr), "pass": bool(same_plan and same_sc)}
    # (iv) determinism: rerun (parts rebuilt, same plans) bit-identical on the rerun subset, all arms and all sub-scores
    rer = [r for r in rr if "rerun" in r]
    ndiff = sum(r["arms"] != r["rerun"] for r in rer)
    ctl["iv_determinism"] = {"n_rerun_rows": len(rer), "n_not_bit_identical": int(ndiff), "pass": len(rer) > 0 and ndiff == 0}
    ok_all = all(c["pass"] for c in ctl.values())
    res = {"outcome_row": None, "n_rows": len(have), "n_logs": len(set(logs)), "units": "PDMS x 100 (pp)", "bar": f"REBUILD iff CI hi < 0 AND point <= {BAR_NOTE}",
           "estimator": f"paired log-cluster bootstrap over {len(set(logs))} logs, {B_DRAWS} draws, seed {SEED}, ratio mean, 95% percentile (TF-1/E2 estimator verbatim); answers 'another draw of LOGS' only",
           "evidence_class": "MEASURED", "tier": "navtrain held-out tracked-label check (NAVSIM v1.1 PDMS, LQR tracker, non-reactive); not a driving number, not navtest",
           "controls": ctl, "controls_all_pass": bool(ok_all), "smoke": bool(smoke)}
    if rj5 is not None:
        res["read_kind"] = "W-A7-RJ5"
        res["traj_source"] = {"file": rj5.get("traj_source"), "join_sha256": rj5.get("join_sha256"), "note": "label input traj u(r) = the RJ5 traj of each row (A3 section 5); the bank traj is the raw-fallback target only"}
        res["tier"] += "; labels built from the RJ5 teacher trajectories (A3)"
    if len(have):
        dl = {"P5-R": P("R_A7") - P("R_noA7"), "Tc20s": P("T_A7") - P("T_noA7")}
        pdm = {"P5-R_noA7": P("R_noA7").mean() * 100, "P5-R_A7": P("R_A7").mean() * 100, "Tc20s_noA7": P("T_noA7").mean() * 100, "Tc20s_A7": P("T_A7").mean() * 100}
        cnts = {k: {"n_delta_nonzero": int(np.sum(v != 0)), "n_neg": int(np.sum(v < 0)), "n_pos": int(np.sum(v > 0))} for k, v in dl.items()}
        res["label_sources"] = {"p5r_raw_fallback_rows": sum(r["label_src"]["p5r"] != "p5r" for r in rr), "tc20s_raw_fallback_rows": sum(r["label_src"]["tc20s"] != "tc20s" for r in rr)}
        tcb = [r["tc20s_vs_banked_maxabs"] for r in rr if r["tc20s_vs_banked_maxabs"] is not None]
        res["info_tc20s_dual_vs_banked_npz_maxabs_label"] = max(tcb) if tcb else None
        if smoke:
            res["deltas"] = "WITHHELD in smoke"
        else:
            bs = bootstrap_delta(dl, logs)
            res["pdms_means_pp"] = pdm
            res["delta_pp_point_lo_hi"] = {k: [round(x, 5) for x in v] for k, v in bs.items()}
            res["delta_row_counts"] = cnts
            res["outcome_row"] = outcome_row(bs["P5-R"], ok_all)
            res["decides_on"] = "P5-R; Tc20s reported only"
    if smoke:
        res["outcome_row"] = "SMOKE (not an outcome)"
    res["instrument_blobs"] = {k: {"path": p, "blob": blob(p)} for k, (_, p, _) in PIN_BLOBS.items()}
    res["inputs"] = {"population_npz": NPZ, "population_npz_sha256": sha256(NPZ), "labels_sha256": sha256(f"{run}/labels/labels_s0of1.jsonl"),
                     "scores_sha256": sha256(f"{run}/scores.jsonl"), "tf1_rows": TF1_ROWS}
    return res


def write_result(run, res):
    json.dump(res, open(f"{run}/w_a7_result.json", "w", encoding="utf-8", newline="\n"), indent=1)
    c = res["controls"]
    L = [f"# {res.get('read_kind', 'W-A7')} result (PREREG_P5C_A2 section 6" + (", A3 section 5)" if res.get("read_kind") else ")"), "", f"**Outcome row: {res['outcome_row']}**", "",
         f"- Evidence class: {res['evidence_class']}. Tier: {res['tier']}.", f"- Rows {res['n_rows']}, logs {res['n_logs']}. Units: {res['units']}.",
         f"- Rule: {res['bar']}.", f"- Estimator: {res['estimator']}.", ""]
    if "delta_pp_point_lo_hi" in res:
        L += ["| label | PDMS no A7 | PDMS with A7 | delta [95% CI] (pp) | rows delta!=0 |", "|---|---|---|---|---|"]
        for k, m0, m1 in (("P5-R", "P5-R_noA7", "P5-R_A7"), ("Tc20s", "Tc20s_noA7", "Tc20s_A7")):
            d = res["delta_pp_point_lo_hi"][k]
            L.append(f"| {k}{' (decides)' if k == 'P5-R' else ' (reported only)'} | {res['pdms_means_pp'][m0]:.4f} | {res['pdms_means_pp'][m1]:.4f} | {d[0]:+.5f} [{d[1]:+.5f}, {d[2]:+.5f}] | {res['delta_row_counts'][k]['n_delta_nonzero']} |")
        L.append("")
    L += ["## Controls (any failure = INVALID)", "", "```", json.dumps(c, indent=1), "```", "", f"All controls pass: {res['controls_all_pass']}", ""]
    if "label_sources" in res:
        L += [f"- Raw-fallback label rows: {res['label_sources']}", f"- Info: max |dual Tc20s - banked P5-S Tc20s| = {res['info_tc20s_dual_vs_banked_npz_maxabs_label']}"]
    L += ["", "## Instrument blobs", ""] + [f"- {k}: `{v['blob']}` ({v['path']})" for k, v in res["instrument_blobs"].items()]
    L += ["", "## Inputs", ""] + [f"- {k}: {v}" for k, v in res["inputs"].items()]
    open(f"{run}/W_A7_RESULT.md", "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--rerun-every", type=int, default=10)
    ap.add_argument("--stage", default="all", choices=("all", "prep", "labels", "score", "analyze"))
    ap.add_argument("--traj-source", default=None, help="W-A7-RJ5: the RJ5 join (rj5_bank_join.jsonl.gz); the label input traj of each row := its RJ5 traj")
    ap.add_argument("--expect-join-sha256", default=None)
    ap.add_argument("--bank-run-result", default=None, help="W-A7-RJ5: the bank-traj run's w_a7_result.json (control (i) must be VALID)")
    ap.add_argument("--c1-table", default=None, help="W-A7-RJ5 control (i'): the C1 label table (json.gz)")
    ap.add_argument("--c1-records", default=None, help="W-A7-RJ5 control (i'): the rekeyed labeller records (labels_rekeyed.jsonl), the table's sim8")
    ap.add_argument("--gates-module", default=GATES_PY)
    a = ap.parse_args()
    run = a.run
    rj5 = None
    if a.traj_source:
        rj5 = {"traj_source": os.path.basename(a.traj_source), "join_sha256": a.expect_join_sha256, "c1_table": a.c1_table, "c1_records": a.c1_records, "gates": a.gates_module, "bank_run_result": a.bank_run_result}
    os.makedirs(run, exist_ok=True)
    tmp = os.environ.get("TMP", "")
    log(f"ZZW_A7 START run={run} limit={a.limit} smoke={a.smoke} stage={a.stage} TMP={tmp} OMP={os.environ['OMP_NUM_THREADS']}")
    bad = check_pins()
    if bad:
        log(f"ZZW_A7 STOP pinned blob mismatch {json.dumps(bad)}")
        return 2
    rows = load_population(a.limit)
    log(f"population {len(rows)} rows, {len({r['log'] for r in rows})} logs")
    t0 = time.time()
    try:
        if a.stage in ("all", "prep"):
            if rj5 is None:
                prep(run, rows)
            else:
                log(f"ZZW_A7 W-A7-RJ5 feed: {json.dumps(prep_rj5(run, rows, a.traj_source, a.expect_join_sha256))}")
        elif rj5 is not None and a.stage != "prep":
            read_rj5_join(a.traj_source, a.expect_join_sha256)                     # the source is re-verified by every stage
        if a.stage in ("all", "labels"):
            run_labeller(run)
            log(f"labels done {time.time() - t0:.0f}s: {open(f'{run}/labels/DONE_s0of1.json').read().replace(chr(10), ' ')}")
            if rj5 is not None:
                check_labels_fed(run, rows)
        if a.stage in ("all", "score"):
            if rj5 is not None:
                check_labels_fed(run, rows)
            score_stage(run, rows, load_labels(run), a.rerun_every)
        if a.stage in ("all", "analyze"):
            return _analyze_stage(run, rows, a, rj5)
    except Rj5Refused as e:
        log(f"ZZW_A7 STOP W-A7-RJ5 refused: {e}")
        return 2
    return 0


def _analyze_stage(run, rows, a, rj5):
    if True:
        res = analyze(run, rows, a.smoke, None, rj5)
        write_result(run, res)
        log(f"controls: {json.dumps({k: v['pass'] for k, v in res['controls'].items()})}")
        log(f"ZZW_A7 DONE {res['outcome_row']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
