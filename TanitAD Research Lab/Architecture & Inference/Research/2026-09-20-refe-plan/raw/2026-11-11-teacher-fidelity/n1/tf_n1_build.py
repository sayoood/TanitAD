#!/usr/bin/env python3
"""TF-N1 build (PREREG_TF_N1.md section 10): every N1 arm's plan, from the BANKED 20-pose teacher rollouts, in the
plan-file format `PK/eval/instruments/transp_score.py --plans` reads. CPU only. It never calls the PDM scorer, so it
reads no score of any arm on any token.

Arms (one row per token, all four from ONE derivation, the TF-2 code UNCHANGED):
  T      the banked rollout's `traj20` (20 rear-axle poses, ego frame, t = 0.2 .. 4.0 s), as banked
  Tc20s  `p5_target_labels.iterate20(ND, T, to_navsim(T), ego, ctl, None, "cubic")[0]`   (P5-S, unchanged; = TF-2 `tc_re`)
  Tw     the same call's [1]: the wrong-sign iterate, P5's negative control                   (= the banked P5-S `tw`)
  M      `tf2_score.shape(ND, T, ego, U, +1)` then `iterate20(..., "cubic")[0]`               (= TF-2 `lM`; CERT's 3rd fallback)
CERT is NOT built here: it is selected AFTER scoring from the scorer's own tracked `dac` of [Tc20s, T, M]
(tf_n1_analyze.py cert), exactly as `tf2_analyze.cert` does.

`ego` is `scenario.initial_ego_state` (= transp_score's `parts[1]`), with the scenario built exactly as the scorer builds
it (navtest: history_rows=1, future_rows=80, as p4_tracker_comp.py:99 / transp_score SPLIT_ROWS). `U` is the union of the
drivable polygons of PDM-Closed's map, `PDMDrivableMap.from_simulation(map_api, ego, pdm_closed._map_radius)`, which is
the line `pdm_closed_planner.py:76` runs inside `metric_cache_parts`; `--check-dam` also runs `metric_cache_parts` and
requires the M label from its map to be IDENTICAL (validation only; it costs ~1-2 s/token).

Inputs:
  --rollouts <jsonl[.xz] ...>  navtest: the banked P4 rollout rows {token, log, traj20, plan8}       (--split navtest)
  --p5s <npz>                  navtrain held-out: logs, tokens, labels20_t20_tc20_tw [N, >=3, 20, 3]  (--split navtrain)
                               column 0 = T, 1 = banked Tc20s, 2 = banked Tw  -> control C-DERIV
  --seam <npz>                 navtest only: the registered T seam (token, fingerprint, poses, sampling) -> C-SEAM

Modes:
  derive:  python tf_n1_build.py derive --w <dir with refe/ eval/> --tf2-score <tf2_score.py> --split navtest
               --rollouts <rows ...> --out <shard.jsonl> --shard i/N [--limit n] [--only tok,tok] [--check-dam]
           Resumable: tokens already in --out without "error" are skipped; errored ones are retried.
           Sharded by the SAME stable SHA-1 token hash as transp_score.py.
  merge:   python tf_n1_build.py merge --split navtest --rollouts <rows ...> --seam <T seam npz> --rows <shard.jsonl ...>
               --out-dir <dir>
           -> n1_plans_{T,Tc20s,Tw,M}.npz (tokens, logs, plans20 float64) in the T seam's token order,
              W3 seams refe_tfn1_{tc20s,tw,m}.npz (the T seam's tokens + fingerprints, poses float32 = to_navsim(label))
              for the pinned-harness fallback route, and n1_build_report.json with C-BUILD-COV, C-SEAM (navtest) or
              C-DERIV (navtrain), timing, and the sha256 of every file written. Prints ZZN1_BUILD PASS|FAIL.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import lzma
import math
import os
import sys
import time

os.environ.setdefault("REFE_SIM_HZ", "10")
os.environ.setdefault("NUPLAN_MAPS_ROOT", "E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

DB_ROOT = "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits"
SPLIT_ROWS = {"navtrain": (None, None), "navtest": (1, 80)}       # = transp_score.SPLIT_ROWS
ARMS = ("T", "Tc20s", "Tw", "M")
SEAM_ARMS = ("Tc20s", "Tw", "M")                                   # T's seam is the registered one, never rewritten
C_DERIV_BAR_M = 1e-9                                              # PREREG_TF_N1 C-DERIV (= TF-2 C2)


def sha256(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def shard_of(tok: str, n: int) -> int:
    return int(hashlib.sha1(tok.encode()).hexdigest(), 16) % n      # = transp_score.py's partition


def open_text(p: str):
    return lzma.open(p, "rt", encoding="utf-8") if p.endswith(".xz") else open(p, encoding="utf-8")


def load_inputs(a) -> dict:
    """-> {token: {"log", "t20" [20,3] float64, optional "plan8" [8,3], "tc_bank", "tw_bank"}}"""
    out: dict = {}
    if a.split == "navtest":
        if not a.rollouts:
            raise SystemExit("--split navtest needs --rollouts")
        for pat in a.rollouts:
            for fp in sorted(glob.glob(pat)):
                for ln in open_text(fp):
                    if not ln.strip():
                        continue
                    r = json.loads(ln)
                    t = str(r["token"])
                    if t in out:
                        raise SystemExit(f"duplicate rollout row for token {t} in {fp}")
                    out[t] = {"log": r["log"], "t20": np.asarray(r["traj20"], np.float64),
                              "plan8": np.asarray(r["plan8"], np.float64), "dt": r.get("dt")}
    else:
        if not a.p5s:
            raise SystemExit("--split navtrain needs --p5s")
        z = np.load(a.p5s, allow_pickle=False)
        L = z["labels20_t20_tc20_tw"]
        for i, (lg, tok) in enumerate(zip(z["logs"].tolist(), z["tokens"].tolist())):
            out[str(tok)] = {"log": str(lg), "t20": np.asarray(L[i, 0], np.float64),
                             "tc_bank": np.asarray(L[i, 1], np.float64), "tw_bank": np.asarray(L[i, 2], np.float64)}
    if getattr(a, "only", ""):                                       # a validation subset (navtrain smoke); never on navtest
        want = set(a.only.split(","))
        out = {t: v for t, v in out.items() if t in want}
    for t, v in out.items():
        if v["t20"].shape != (20, 3) or not np.all(np.isfinite(v["t20"])):
            raise SystemExit(f"token {t}: traj20 is not a finite [20, 3] array")
    return out


def load_mod(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def done_tokens(path: str) -> set:
    done = set()
    if os.path.exists(path):
        for ln in open(path, encoding="utf-8"):
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if "error" not in r and "token" in r:
                done.add(r["token"])
    return done


def derive(a) -> int:
    sys.path.insert(0, os.path.join(a.w, "refe"))
    sys.path.insert(0, os.path.join(a.w, "eval"))
    import navsim_dac as ND
    import navtrain_scenarios as NS
    import navsim_pdm_targets as PT
    from p5_target_labels import iterate20                                   # P5-S's derivation, unchanged
    from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap
    TF2 = load_mod("tf2_score_for_n1", a.tf2_score)                         # TF-2's shape() / dac_union(), unchanged
    inp = load_inputs(a)
    si, sn = (int(x) for x in a.shard.split("/"))
    done = done_tokens(a.out)
    todo = [t for t in sorted(inp) if shard_of(t, sn) == si and t not in done]
    by: dict = {}
    for t in todo:
        by.setdefault(inp[t]["log"], []).append(t)
    dbs = NS.index_dbs(db_root=a.db_root)
    hr, fr = SPLIT_ROWS[a.split]
    kw = {} if hr is None else {"history_rows": hr, "future_rows": fr}
    S = PT._setup()
    radius = S["proc"]._pdm_closed._map_radius
    if os.path.exists(a.out) and os.path.getsize(a.out) > 0:                # terminate a torn last line (killed run)
        with open(a.out, "rb") as fb:
            fb.seek(-1, 2)
            torn = fb.read(1) != b"\n"
        if torn:
            with open(a.out, "a", encoding="utf-8") as fb:
                fb.write("\n")
    n_done, t0 = 0, time.time()
    with open(a.out, "a", encoding="utf-8") as fo:
        for lg in sorted(by):
            if not os.path.exists(dbs.get(lg, "")):
                for t in by[lg]:
                    fo.write(json.dumps({"log": lg, "token": t, "error": "no local DB"}) + "\n")
                continue
            for sc in NS.build_scenarios_for_log(dbs[lg], by[lg], **kw):
                tok = sc._initial_lidar_token
                ts = time.time()
                rec = {"log": lg, "token": tok}
                try:
                    t20 = inp[tok]["t20"]
                    ego = sc.initial_ego_state
                    ctl = {"knot_exact": True, "knot_checks": 0}
                    tc, tw, _ident, devs = iterate20(ND, t20, ND.to_navsim(t20), ego, ctl, None, "cubic")
                    dam = PDMDrivableMap.from_simulation(sc.map_api, ego, radius)
                    U, _I = TF2.dac_union(dam)
                    qM, aM = TF2.shape(ND, t20, ego, U, +1.0)
                    ctlM = {"knot_exact": True, "knot_checks": 0}
                    lM = iterate20(ND, qM, ND.to_navsim(qM), ego, ctlM, None, "cubic")[0]
                    rec["labels"] = {"T": t20.tolist(), "Tc20s": tc.tolist(), "Tw": tw.tolist(), "M": lM.tolist()}
                    info = {"v0": float(ego.dynamic_car_state.speed), "dev_T_tracked_m": devs[0][0],
                            "dev_Tc20s_tracked_m": float(min(devs[0])), "knot_exact": bool(ctl["knot_exact"] and ctlM["knot_exact"]),
                            "knot_maxdev": float(max(ctl.get("knot_maxdev", 0.0), ctlM.get("knot_maxdev", 0.0))),
                            "m_shift_max_m": float(np.abs(aM).max()), "m_shift_n_poses": int(np.count_nonzero(aM[1:]))}
                    if a.check_dam:                                          # validation: the scorer's own map route
                        parts = PT.metric_cache_parts(sc)
                        U2, _ = TF2.dac_union(parts[5])
                        qM2, aM2 = TF2.shape(ND, t20, parts[1], U2, +1.0)
                        lM2 = iterate20(ND, qM2, ND.to_navsim(qM2), parts[1], {"knot_exact": True, "knot_checks": 0},
                                        None, "cubic")[0]
                        info["checkdam_union_equal"] = bool(U.equals(U2))
                        info["checkdam_M_identical"] = bool(np.array_equal(lM, lM2))
                        info["checkdam_ego_identical"] = bool(parts[1] is ego or (
                            parts[1].rear_axle.serialize() == ego.rear_axle.serialize()
                            and parts[1].time_point.time_us == ego.time_point.time_us))
                    if "tc_bank" in inp[tok]:                                 # C-DERIV (navtrain): the banked P5-S labels
                        info["cderiv_tc20s_maxdev_m"] = float(np.abs(tc - inp[tok]["tc_bank"]).max())
                        info["cderiv_tw_maxdev_m"] = float(np.abs(tw - inp[tok]["tw_bank"]).max())
                    rec["info"] = info
                    rec["t_s"] = round(time.time() - ts, 3)
                except Exception as e:                                       # noqa: BLE001
                    rec = {"log": lg, "token": tok, "error": f"{type(e).__name__}: {str(e)[:200]}"}
                fo.write(json.dumps(rec) + "\n")
                fo.flush()
                n_done += 1
                if a.limit and n_done >= a.limit:
                    print("ZZN1_BUILD_SHARD_DONE", a.shard, n_done, round(time.time() - t0, 1), flush=True)
                    return 0
    print("ZZN1_BUILD_SHARD_DONE", a.shard, n_done, round(time.time() - t0, 1), flush=True)
    return 0


def merge_impl(a) -> int:
    sys.path.insert(0, os.path.join(a.w, "refe"))
    import navsim_dac as ND                                                   # to_navsim, imported, never re-implemented
    inp = load_inputs(a)
    rows: dict = {}
    n_err, n_dup = 0, 0
    for pat in a.rows:
        for fp in sorted(glob.glob(pat)):
            for ln in open(fp, encoding="utf-8"):
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if "error" in r:
                    n_err += 1
                    continue
                if r["token"] in rows:
                    n_dup += 1
                    continue
                rows[r["token"]] = r
    rep: dict = {"split": a.split, "inputs": sorted(set(sum([sorted(glob.glob(p)) for p in (a.rollouts or [])], []))) or a.p5s,
                 "n_inputs": len(inp), "n_rows": len(rows), "n_error_lines": n_err, "n_duplicate_lines": n_dup,
                 "controls": {}, "files": {}}
    missing = sorted(set(inp) - set(rows))
    extra = sorted(set(rows) - set(inp))
    rep["controls"]["C-BUILD-COV"] = {"pass": not missing and not extra, "missing": len(missing), "extra": len(extra),
                                      "missing_examples": missing[:10]}
    if a.split == "navtest":
        z = np.load(a.seam, allow_pickle=False)
        order = [str(t) for t in z["token"]]
        fps = z["fingerprint"]
    else:
        order = sorted(inp)
        z, fps = None, None
    ok_rows = [t for t in order if t in rows]
    # T must be the banked traj20, bit for bit
    t_eq = all(np.array_equal(np.asarray(rows[t]["labels"]["T"], np.float64), inp[t]["t20"]) for t in ok_rows)
    rep["controls"]["C-BUILD-T-IS-BANKED"] = {"pass": bool(t_eq)}
    if a.split == "navtest":
        seam_tok_eq = set(order) == set(inp)
        poses = z["poses"]
        idx = {t: i for i, t in enumerate(order)}
        n_pose_eq, n_plan8_eq, worst_plan8 = 0, 0, 0.0
        for t in order:
            if t not in inp:
                continue
            p8 = ND.to_navsim(inp[t]["t20"])
            if np.array_equal(p8.astype(np.float32), poses[idx[t]]):
                n_pose_eq += 1
            d = float(np.abs(p8 - inp[t]["plan8"]).max())
            worst_plan8 = max(worst_plan8, d)
            if d <= 1e-12:
                n_plan8_eq += 1
        rep["controls"]["C-SEAM"] = {"pass": bool(seam_tok_eq and n_pose_eq == len(order) and n_plan8_eq == len(order)),
                                     "seam_tokens": len(order), "rollout_tokens": len(inp), "token_sets_equal": bool(seam_tok_eq),
                                     "to_navsim_traj20_float32_equals_seam_poses": n_pose_eq,
                                     "to_navsim_traj20_equals_row_plan8_1e-12": n_plan8_eq, "worst_plan8_abs_m": worst_plan8,
                                     "sampling": z["sampling"].tolist()}
    else:
        dv = [rows[t]["info"].get("cderiv_tc20s_maxdev_m", float("nan")) for t in ok_rows]
        dw = [rows[t]["info"].get("cderiv_tw_maxdev_m", float("nan")) for t in ok_rows]
        mx = (max(dv) if dv else float("nan"), max(dw) if dw else float("nan"))
        rep["controls"]["C-DERIV"] = {"pass": bool(dv) and mx[0] <= C_DERIV_BAR_M and mx[1] <= C_DERIV_BAR_M,
                                      "n": len(dv), "tc20s_maxdev_m": mx[0], "tw_maxdev_m": mx[1], "bar_m": C_DERIV_BAR_M}
    cd = [rows[t]["info"] for t in ok_rows if "checkdam_M_identical" in rows[t]["info"]]
    if cd:
        rep["controls"]["C-DAM (validation only)"] = {"n": len(cd), "pass": all(i["checkdam_M_identical"] and i["checkdam_union_equal"]
                                                                               and i["checkdam_ego_identical"] for i in cd),
                                                      "M_identical": sum(i["checkdam_M_identical"] for i in cd),
                                                      "union_equal": sum(i["checkdam_union_equal"] for i in cd),
                                                      "ego_identical": sum(i["checkdam_ego_identical"] for i in cd)}
    ts = [rows[t]["t_s"] for t in ok_rows if "t_s" in rows[t]]
    rep["timing"] = {"n": len(ts), "t_s_mean": float(np.mean(ts)) if ts else None,
                     "t_s_median": float(np.median(ts)) if ts else None,
                     "cpu_h_sum": float(np.sum(ts) / 3600.0) if ts else None,
                     "note": "per-token derivation time, excluding scenario build (about 1 s/token amortised, 6 s for a log's first)"}
    rep["knot_exact_all"] = bool(all(rows[t]["info"]["knot_exact"] for t in ok_rows))
    rep["m_shifted_tokens"] = int(sum(rows[t]["info"]["m_shift_n_poses"] > 0 for t in ok_rows))
    allpass = all(c["pass"] for c in rep["controls"].values())
    os.makedirs(a.out_dir, exist_ok=True)
    if allpass:
        logs = np.array([inp[t]["log"] for t in order])
        tok_arr = np.array(order)
        for arm in ARMS:
            P = np.stack([np.asarray(rows[t]["labels"][arm], np.float64) for t in order])
            fp = os.path.join(a.out_dir, f"n1_plans_{arm}.npz")
            np.savez(fp, tokens=tok_arr, logs=logs, plans20=P, arm=np.array(arm))
            rep["files"][os.path.basename(fp)] = sha256(fp)
            if a.split == "navtest" and arm in SEAM_ARMS:
                p8 = np.stack([ND.to_navsim(x).astype(np.float32) for x in P])
                sp = os.path.join(a.out_dir, f"refe_tfn1_{arm.lower()}.npz")
                np.savez(sp, token=z["token"], fingerprint=fps, poses=p8, sampling=z["sampling"],
                         arm=np.array(str(z["arm"]) + f"_tfn1_{arm.lower()}"))
                rep["files"][os.path.basename(sp)] = sha256(sp)
    rep["verdict"] = "PASS" if allpass else "FAIL"
    rp = os.path.join(a.out_dir, "n1_build_report.json")
    json.dump(rep, open(rp, "w", encoding="utf-8"), indent=1)
    print("ZZN1_BUILD", rep["verdict"], json.dumps({k: v["pass"] for k, v in rep["controls"].items()}), flush=True)
    return 0 if allpass else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("derive", "merge"))
    ap.add_argument("--w", required=True, help="dir holding refe/ and eval/ (a git-archive export of the tip)")
    ap.add_argument("--tf2-score", help="TF-2's tf2_score.py (PK/raw/2026-11-11-teacher-fidelity/tf2/tf2_score.py)")
    ap.add_argument("--split", choices=sorted(SPLIT_ROWS), required=True)
    ap.add_argument("--rollouts", nargs="*", default=[])
    ap.add_argument("--p5s", default=None)
    ap.add_argument("--seam", default=None)
    ap.add_argument("--db-root", default=DB_ROOT)
    ap.add_argument("--out", help="derive: the shard's jsonl")
    ap.add_argument("--rows", nargs="*", default=[], help="merge: the derive shards")
    ap.add_argument("--out-dir", help="merge: where the plan files, seams and report go")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="")
    ap.add_argument("--check-dam", action="store_true")
    a = ap.parse_args()
    if a.mode == "derive":
        if not (a.tf2_score and a.out):
            raise SystemExit("derive needs --tf2-score and --out")
        return derive(a)
    if not a.out_dir or (a.split == "navtest" and not a.seam):
        raise SystemExit("merge needs --out-dir (and --seam for navtest)")
    return merge_impl(a)


if __name__ == "__main__":
    sys.exit(main())
