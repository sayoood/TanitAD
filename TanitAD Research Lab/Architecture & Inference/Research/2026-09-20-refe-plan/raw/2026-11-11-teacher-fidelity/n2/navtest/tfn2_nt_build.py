#!/usr/bin/env python3
"""TF-N2 navtest build (PREREG_TF_N2.md): the plans of every N2 arm, from the RJ5 re-rolled 20-pose teacher rollouts (and the BANKED rollouts
for the comparator N1-C), as W3 seams (8 poses at 0.5 s, the T seam's tokens and fingerprints) and as 20-pose plan files for the families.
CPU only. It never calls the PDM scorer, so it reads no score of any arm on any token.

Per token, from ONE derivation each (every function imported unchanged, never re-implemented):
  N2C  Tc20s of the RJ5 rollout     iterate20(ND, t20, to_navsim(t20), ego, ctl, None, "cubic")[0]       (P5-S form = TF-N1's N1-C recipe)
  N2W  the same call's [1]          the wrong-sign iterate, P5's negative control                        (deliberate regression, C-REG)
  N2R  P5-R, lambda* = 3.0          p5r_labels.solve(t20, Tc20s - t20, [3.0], tracked8, to_navsim)        (RESULT_P5R; warm start Tc20s)
  N1C  Tc20s of the BANKED rollout  the same recipe as N2C on the banked traj20                            (comparator plan; its harness CSV is banked)
`ego` = scenario.initial_ego_state, the scenario built as the scorer builds it on navtest (history_rows=1, future_rows=80) = TF-N1's build.

  derive: python tfn2_nt_build.py derive --w <run>/w --rolls "<RJ5 rows glob>" --bank "<banked rows glob>" --out <shard.jsonl> --shard i/N
              [--rows-kw navtest|navtrain] [--db-root <dir>] [--cderiv-ref "<reroll score rows glob>"] [--limit n]
          Resumable; sharded by sha1(token) % N (= transp_score.py's partition).
  merge:  python tfn2_nt_build.py merge --rolls .. --bank .. --rows <shards> --out-dir <dir> [--seam <T seam npz>]
          -> refe_tfn2_{n2r,n2c,n2w,n1c}.npz W3 seams, n2_plans_{N2R,N2C,N1C}.npz, the 200-token C-ID-C subset (seam + token json),
             n2_build_report.json (C-BUILD-COV, C-BUILD-BANK-IS-SEAM, C-BUILD-RJ5-COVERS, C-DERIV when --cderiv-ref). Prints ZZN2_BUILD PASS|FAIL.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import lzma
import os
import sys
import time

os.environ.setdefault("REFE_SIM_HZ", "10")
os.environ.setdefault("NUPLAN_MAPS_ROOT", "E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

DB_ROOT = "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits"
LAM = 3.0                                   # P5-R lambda* (RESULT_P5R)
ROWS_KW = {"navtest": (1, 80), "navtrain": (None, None)}
C_DERIV_BAR_M = 1e-9
IDC_N = 200                                 # C-ID-C subset: the 200 tokens with the smallest sha1(token) (fixed in advance)
ARMS = ("N2R", "N2C", "N2W", "N1C")
SEAM_ARMS = ("N2R", "N2C", "N2W", "N1C")
FAM_PLAN_ARMS = ("N2R", "N2C", "N1C")


def sha256(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def shard_of(tok: str, n: int) -> int:
    return int(hashlib.sha1(tok.encode()).hexdigest(), 16) % n


def open_text(p: str):
    return lzma.open(p, "rt", encoding="utf-8") if p.endswith(".xz") else open(p, encoding="utf-8")


def load_rows(patterns) -> dict:
    """{token: {"log", "t20" [20,3]}} from rollout rows ({token, log, traj20}); a token twice, or a non-finite traj20, is a refusal"""
    out: dict = {}
    for pat in ([patterns] if isinstance(patterns, str) else patterns):
        for fp in sorted(glob.glob(pat)):
            for ln in open_text(fp):
                if not ln.strip():
                    continue
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if "error" in r or "traj20" not in r:
                    continue
                t = str(r["token"])
                if t in out:
                    raise SystemExit(f"duplicate rollout row for token {t} in {fp}")
                x = np.asarray(r["traj20"], np.float64)
                if x.shape != (20, 3) or not np.all(np.isfinite(x)):
                    raise SystemExit(f"token {t}: traj20 is not a finite [20, 3] array")
                out[t] = {"log": r["log"], "t20": x}
    return out


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


def load_cderiv_ref(pattern) -> dict:
    """{token: {"R": reference8, "C": reference8}} from the TF-N2 re-roll's score rows (arms.<R|C>.fam.reference8)"""
    ref = {}
    for fp in sorted(glob.glob(pattern)):
        for ln in open(fp, encoding="utf-8"):
            if not ln.strip() or '"error"' in ln:
                continue
            r = json.loads(ln)
            try:
                ref[r["token"]] = {k: np.asarray(r["arms"][k]["fam"]["reference8"], np.float64) for k in ("R", "C")}
            except KeyError:
                continue
    return ref


def derive(a) -> int:
    pk = os.path.join(a.w, "pk")
    os.environ["P5C_PKG"] = pk
    for p in (os.path.join(pk, "refe"), os.path.join(pk, "eval"), os.path.join(a.w, "p5r")):
        sys.path.insert(0, p)
    import navsim_dac as ND
    import navtrain_scenarios as NS
    from p5_target_labels import iterate20
    from p4_tracker_comp import tracked8
    import p5r_labels as PR
    rolls, bank = load_rows(a.rolls), load_rows(a.bank)
    ref = load_cderiv_ref(a.cderiv_ref) if a.cderiv_ref else {}
    si, sn = (int(x) for x in a.shard.split("/"))
    done = done_tokens(a.out)
    todo = [t for t in sorted(rolls) if shard_of(t, sn) == si and t not in done]
    by: dict = {}
    for t in todo:
        by.setdefault(rolls[t]["log"], []).append(t)
    dbs = NS.index_dbs(db_root=a.db_root)
    hr, fr = ROWS_KW[a.rows_kw]
    kw = {} if hr is None else {"history_rows": hr, "future_rows": fr}
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
                    t20, b20 = rolls[tok]["t20"], bank[tok]["t20"]
                    ego = sc.initial_ego_state
                    ctl = {"knot_exact": True, "knot_checks": 0}
                    tc, tw, _i, devs = iterate20(ND, t20, ND.to_navsim(t20), ego, ctl, None, "cubic")
                    track = lambda Pb: tracked8(ND, Pb, ego)                # noqa: E731
                    lr = PR.solve(t20, tc - t20, [LAM], track, ND.to_navsim)[LAM]["label"]
                    ctlb = {"knot_exact": True, "knot_checks": 0}
                    tcb = iterate20(ND, b20, ND.to_navsim(b20), ego, ctlb, None, "cubic")[0]
                    rec["labels"] = {"N2R": lr.tolist(), "N2C": tc.tolist(), "N2W": tw.tolist(), "N1C": tcb.tolist()}
                    info = {"v0": float(ego.dynamic_car_state.speed), "knot_exact": bool(ctl["knot_exact"] and ctlb["knot_exact"]),
                            "rj5_minus_bank_dxy_max_m": float(np.abs(t20[:, :2] - b20[:, :2]).max())}
                    if tok in ref:                                           # C-DERIV: against the re-roll's own scored plans
                        info["cderiv_R_maxdev_m"] = float(np.abs(ND.to_navsim(lr) - ref[tok]["R"]).max())
                        info["cderiv_C_maxdev_m"] = float(np.abs(ND.to_navsim(tc) - ref[tok]["C"]).max())
                    rec["info"] = info
                    rec["t_s"] = round(time.time() - ts, 3)
                except Exception as e:                                       # noqa: BLE001
                    rec = {"log": lg, "token": tok, "error": f"{type(e).__name__}: {str(e)[:200]}"}
                fo.write(json.dumps(rec) + "\n")
                fo.flush()
                n_done += 1
                if a.limit and n_done >= a.limit:
                    print("ZZN2_BUILD_SHARD_DONE", a.shard, n_done, round(time.time() - t0, 1), flush=True)
                    return 0
    print("ZZN2_BUILD_SHARD_DONE", a.shard, n_done, round(time.time() - t0, 1), flush=True)
    return 0


def merge_impl(a) -> int:
    sys.path.insert(0, os.path.join(a.w, "pk", "refe"))
    import navsim_dac as ND
    rolls, bank = load_rows(a.rolls), load_rows(a.bank)
    rows: dict = {}
    n_err = n_dup = 0
    for pat in a.rows:
        for fp in sorted(glob.glob(pat)):
            for ln in open(fp, encoding="utf-8"):
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if "error" in r:
                    n_err += 1
                elif r["token"] in rows:
                    n_dup += 1
                else:
                    rows[r["token"]] = r
    rep: dict = {"n_rj5_rollouts": len(rolls), "n_bank_rollouts": len(bank), "n_rows": len(rows), "n_error_lines": n_err,
                 "n_duplicate_lines": n_dup, "controls": {}, "files": {}}
    missing, extra = sorted(set(rolls) - set(rows)), sorted(set(rows) - set(rolls))
    rep["controls"]["C-BUILD-COV"] = {"pass": not missing and not extra and not set(rolls) - set(bank), "missing": len(missing),
                                      "extra": len(extra), "rj5_without_bank": len(set(rolls) - set(bank)), "missing_examples": missing[:10]}
    z = np.load(a.seam, allow_pickle=False) if a.seam else None
    order = [str(t) for t in z["token"]] if z is not None else sorted(rows)
    if z is not None:
        idx = {t: i for i, t in enumerate(order)}
        n_eq = sum(1 for t in order if t in bank and np.array_equal(ND.to_navsim(bank[t]["t20"]).astype(np.float32), z["poses"][idx[t]]))
        rep["controls"]["C-BUILD-BANK-IS-SEAM"] = {"pass": set(order) == set(bank) and n_eq == len(order), "seam_tokens": len(order),
                                                   "bank_tokens": len(bank), "to_navsim_bank_float32_equals_seam_poses": n_eq}
        rep["controls"]["C-BUILD-RJ5-COVERS"] = {"pass": set(order) == set(rolls), "seam_tokens": len(order), "rj5_tokens": len(rolls)}
    ok_rows = [t for t in order if t in rows]
    if a.cderiv_ref:
        dr = [rows[t]["info"].get("cderiv_R_maxdev_m", float("nan")) for t in ok_rows]
        dc = [rows[t]["info"].get("cderiv_C_maxdev_m", float("nan")) for t in ok_rows]
        mx = (max(dr) if dr else float("nan"), max(dc) if dc else float("nan"))
        rep["controls"]["C-DERIV"] = {"pass": bool(dr) and bool(mx[0] <= C_DERIV_BAR_M) and bool(mx[1] <= C_DERIV_BAR_M), "n": len(dr),
                                      "R_maxdev_m": mx[0], "C_maxdev_m": mx[1], "bar_m": C_DERIV_BAR_M}
    ts = [rows[t]["t_s"] for t in ok_rows if "t_s" in rows[t]]
    rep["timing"] = {"n": len(ts), "t_s_mean": float(np.mean(ts)) if ts else None, "t_s_median": float(np.median(ts)) if ts else None,
                     "cpu_h_sum": float(np.sum(ts) / 3600.0) if ts else None,
                     "note": "per-token derivation (2 x iterate20 + P5-R), excluding the scenario build"}
    rep["knot_exact_all"] = bool(all(rows[t]["info"]["knot_exact"] for t in ok_rows))
    allpass = all(c["pass"] for c in rep["controls"].values())
    os.makedirs(a.out_dir, exist_ok=True)
    if allpass:
        tok_arr = np.array(order)
        logs = np.array([rolls[t]["log"] for t in order])
        P = {arm: np.stack([np.asarray(rows[t]["labels"][arm], np.float64) for t in order]) for arm in ARMS}
        for arm in FAM_PLAN_ARMS:
            fp = os.path.join(a.out_dir, f"n2_plans_{arm}.npz")
            np.savez(fp, tokens=tok_arr, logs=logs, plans20=P[arm], arm=np.array(arm))
            rep["files"][os.path.basename(fp)] = sha256(fp)
        if z is not None:
            p8 = {arm: np.stack([ND.to_navsim(x).astype(np.float32) for x in P[arm]]) for arm in SEAM_ARMS}
            for arm in SEAM_ARMS:
                sp = os.path.join(a.out_dir, f"refe_tfn2_{arm.lower()}.npz")
                np.savez(sp, token=z["token"], fingerprint=z["fingerprint"], poses=p8[arm], sampling=z["sampling"],
                         arm=np.array(str(z["arm"]) + f"_tfn2_{arm.lower()}"))
                rep["files"][os.path.basename(sp)] = sha256(sp)
            sub = sorted(order, key=lambda t: hashlib.sha1(t.encode()).hexdigest())[:IDC_N]
            keep = [idx[t] for t in order if t in set(sub)]
            sp = os.path.join(a.out_dir, "refe_tfn2_n1c_sub200.npz")
            np.savez(sp, token=z["token"][keep], fingerprint=z["fingerprint"][keep], poses=p8["N1C"][keep], sampling=z["sampling"],
                     arm=np.array(str(z["arm"]) + "_tfn2_n1c_sub200"))
            tl = {t: rolls[t]["log"] for t in sub}
            tp = os.path.join(a.out_dir, "cidc_tokens.json")
            json.dump({"tokens": sorted(sub), "token_log": tl}, open(tp, "w", encoding="utf-8"), indent=1)
            rep["files"]["refe_tfn2_n1c_sub200.npz"], rep["files"]["cidc_tokens.json"] = sha256(sp), sha256(tp)
    rep["verdict"] = "PASS" if allpass else "FAIL"
    rp = os.path.join(a.out_dir, "n2_build_report.json")
    json.dump(rep, open(rp, "w", encoding="utf-8"), indent=1)
    print("ZZN2_BUILD", rep["verdict"], json.dumps({k: v["pass"] for k, v in rep["controls"].items()}), flush=True)
    return 0 if allpass else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("derive", "merge"))
    ap.add_argument("--w", required=True, help="<run>/w: holds pk/refe, pk/eval, p5r/")
    ap.add_argument("--rolls", nargs="+", required=True, help="RJ5 rollout rows (jsonl globs)")
    ap.add_argument("--bank", nargs="+", required=True, help="the banked teacher rollout rows (jsonl.xz globs)")
    ap.add_argument("--rows-kw", choices=sorted(ROWS_KW), default="navtest")
    ap.add_argument("--db-root", default=DB_ROOT)
    ap.add_argument("--cderiv-ref", default=None)
    ap.add_argument("--out", help="derive: the shard's jsonl")
    ap.add_argument("--rows", nargs="*", default=[], help="merge: the derive shards")
    ap.add_argument("--out-dir", help="merge: where the plan files, seams and report go")
    ap.add_argument("--seam", default=None, help="merge: the registered T seam (navtest); absent = smoke, no W3 seams are written")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.mode == "derive":
        if not a.out:
            raise SystemExit("derive needs --out")
        return derive(a)
    if not a.out_dir:
        raise SystemExit("merge needs --out-dir")
    return merge_impl(a)


if __name__ == "__main__":
    sys.exit(main())
