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


def analyze(run, rows, smoke=False, banked=None):
    sc = [json.loads(ln) for ln in open(f"{run}/scores.jsonl", encoding="utf-8")]
    err = [r for r in sc if "error" in r]
    ok = {(r["log"], r["token"]): r for r in sc if "error" not in r}
    keys = [(r["log"], r["token"]) for r in rows]
    have = [k for k in keys if k in ok]
    rr = [ok[k] for k in have]
    logs = [k[0] for k in have]
    P = lambda arm: np.array([r["arms"][arm]["pdms"] for r in rr])                      # noqa: E731
    banked = banked_tc20s() if banked is None else banked
    ctl = {}
    # (0) coverage: every population row scored, no error rows
    ctl["coverage"] = {"n_population": len(keys), "n_scored": len(have), "n_error_rows": len(err), "pass": len(have) == len(keys) and not err}
    # (i) reproduction of banked TF1 tracked Tc20s per row
    cmp_ = [(abs(ok[k]["arms"]["T_noA7"]["pdms"] - banked[k]), k) for k in have if k in banked]
    mx = max((d for d, _ in cmp_), default=float("nan"))
    ctl["i_reproduction_tf1_tc20s_tracked"] = {"n_compared": len(cmp_), "max_abs_diff": mx, "n_over_1e-6": sum(d > 1e-6 for d, _ in cmp_),
                                              "tol": 1e-6, "pass": len(cmp_) > 0 and mx <= 1e-6 and len(cmp_) == len(have)}
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
    L = ["# W-A7 result (PREREG_P5C_A2 section 6)", "", f"**Outcome row: {res['outcome_row']}**", "",
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
    a = ap.parse_args()
    run = a.run
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
    if a.stage in ("all", "prep"):
        prep(run, rows)
    if a.stage in ("all", "labels"):
        run_labeller(run)
        log(f"labels done {time.time() - t0:.0f}s: {open(f'{run}/labels/DONE_s0of1.json').read().replace(chr(10), ' ')}")
    if a.stage in ("all", "score"):
        score_stage(run, rows, load_labels(run), a.rerun_every)
    if a.stage in ("all", "analyze"):
        res = analyze(run, rows, a.smoke)
        write_result(run, res)
        log(f"controls: {json.dumps({k: v['pass'] for k, v in res['controls'].items()})}")
        log(f"ZZW_A7 DONE {res['outcome_row']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
