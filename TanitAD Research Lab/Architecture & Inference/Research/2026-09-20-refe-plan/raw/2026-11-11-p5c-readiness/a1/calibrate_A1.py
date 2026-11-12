#!/usr/bin/env python3
"""A1 calibration of R-TOL-1/2/3 on the 33 real P5-R EVAL fixture rows (CPU only, dev box).  The evidence the MM freezes the R-B thresholds on.

  python calibrate_A1.py run   --arm NAME --lam 3.0 --warm fixture|onethread --out FILE.jsonl     (ONE arm; set the BLAS env BEFORE python starts)
  python calibrate_A1.py score --arms-dir DIR --out calibration_A1.json                            (reads DIR/<arm>.jsonl for every arm, applies the gate's own functions)

Arms (driver `all` runs them, each in its own process so the BLAS env is in force before numpy imports):
  control_1thread   dev-box Tc20s warm start (fixture), P5-R at lam* 3.0, BLAS 1 thread     -> sanity: must read max abs diff 0.0
  noise_a_blas4     dev-box Tc20s warm start,           P5-R at lam* 3.0, BLAS 4 threads    -> MUST PASS R-TOL-1 and R-TOL-2
  noise_b_warm1t    warm start from a 1-thread Tc20s,   P5-R at lam* 3.0, BLAS 1 thread     -> MUST PASS R-TOL-1 and R-TOL-2
  reg_lam1          fixture warm start, lam = 1.0 (grid neighbour below lam*)               -> MUST FAIL R-TOL-2 or R-TOL-3
  reg_lam10         fixture warm start, lam = 10.0 (grid neighbour above lam*)              -> MUST FAIL R-TOL-2 or R-TOL-3
Nothing here changes a threshold: the thresholds are imported from p5c_table_gates.py."""
import os, sys, json, time, subprocess, argparse, math

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = HERE + "/fixtures"
S_ROOT = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107"
ARMS = [("control_1thread", 3.0, "fixture", 1), ("noise_a_blas4", 3.0, "fixture", 4), ("noise_b_warm1t", 3.0, "onethread", 1), ("reg_lam1", 1.0, "fixture", 1), ("reg_lam10", 10.0, "fixture", 1)]


def eval_rows():
    ev = {}
    for ln in open(FIX + "/eval_control_33.jsonl", encoding="utf-8"):
        d = json.loads(ln)
        ev[(d["log"], d["token"])] = d
    return ev


def run_arm(a):
    os.environ["P5R_DIR"] = S_ROOT + "/p5r"
    os.environ["P5C_BANKDIR"] = FIX + "/bank"
    sys.path.insert(0, HERE)
    import numpy as np
    import p5c_dual_labels as D
    import p5_target_labels as P
    import navsim_dac as ND
    import p5r_run as RUN
    import p5r_labels as R
    RUN.DBROOT = D.DBROOT
    S = RUN.Sim()
    NS, dbs = S.NS, S.dbs
    ev = eval_rows()
    fx = {(d["log"], d["token"]): d for d in map(json.loads, open(FIX + "/dual_records_40.jsonl", encoding="utf-8"))}
    rows = [r for r in D.load_rows() if r[2] == 0 and (r[0], r[1]) in ev]
    assert len(rows) == 33, len(rows)
    by = {}
    for lg, tok, rk, traj in rows:
        by.setdefault(lg, []).append((tok, traj))
    out = open(a.out, "w", encoding="utf-8", newline="\n")
    t0 = time.time()
    for lg, items in sorted(by.items()):
        scs = {sc._initial_lidar_token: sc for sc in NS.build_scenarios_for_log(dbs[lg], sorted(t for t, _ in items))}
        for tok, traj in items:
            ego = scs[tok].get_ego_state_at_iteration(0)
            polys = ND.drivable_polygons(scs[tok].map_api, ego)
            t20 = np.asarray(traj, np.float64)
            if a.warm == "fixture":
                tc20 = np.asarray(fx[(lg, tok)]["tc20s"], np.float64)
            else:
                ctl = {"knot_exact": True, "knot_checks": 0}
                tc20, _, _, _ = P.iterate20(ND, t20, ND.to_navsim(t20), ego, ctl, None, "cubic")
            rec = {"log": lg, "token": tok, "rank": 0, "traj_sha256": D.traj_sha(traj), "tc20s": [[float(v) for v in p] for p in tc20]}
            rec.update(D.p5r_row(S, R, RUN.sanity, ego, polys, t20, tc20, a.lam))
            out.write(json.dumps(rec) + "\n")
    out.close()
    print(f"arm {a.arm} lam={a.lam} warm={a.warm} rows=33 {time.time() - t0:.0f}s env OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS')} MKL={os.environ.get('MKL_NUM_THREADS')} OMP={os.environ.get('OMP_NUM_THREADS')}")


def score(a):
    import numpy as np
    sys.path.insert(0, HERE)
    import p5c_table_gates as G
    bank = G.load_bank([FIX + "/bank"])
    keep = np.where(bank["rank"] == 0)[0]
    b0 = {"log": [bank["log"][i] for i in keep], "token": [bank["token"][i] for i in keep], "rank": bank["rank"][keep], "traj": bank["traj"][keep], "sim_hz": [bank["sim_hz"][i] for i in keep],
          "sha": [bank["sha"][i] for i in keep]}
    ev = G.load_eval_control([FIX + "/eval_control_33.jsonl"], "3.0")
    ref_v = G.load_verdict_ref(S_ROOT + "/p5r/raw/verdict.json")
    # the dev-box EVAL ratios on the SAME 33 rows (the noise-free comparator for a 33-row sample)
    idx = {G.triple(b0["log"][i], b0["token"][i], 0): i for i in range(len(b0["log"]))}
    ii = [idx[t] for t in ev]
    la, lh = G.learn(np.stack([np.asarray(ev[t]["label"], np.float64) for t in ev]))
    ta, th = G.learn(b0["traj"][ii])
    ref_s = {0: {"R_acc": float(np.median(la) / np.median(ta)), "R_hr": float(np.median(lh) / np.median(th))}}
    res = {"n_rows": 33, "rank": 0, "thresholds": {"R-TOL-1_min_frac": G.TOL1_FRAC, "R-TOL-2_median_max": G.TOL2_MEDIAN, "R-TOL-2_p99_max": G.TOL2_P99, "R-TOL-3_ratio_abs": G.TOL3_RATIO,
                                                      "R-TOL-3_sim8_pp": G.TOL3_SIM8_PP},
           "verdict_json_rank0_context_only": ref_v[0], "row_matched_reference_dev_box": ref_s[0], "arms": {}}
    for name, lam, warm, thr in ARMS:
        p = f"{a.arms_dir}/{name}.jsonl"
        recs, _, _ = G.load_label_recs([p])
        M = G.merge_p5r(b0, recs)
        ok_v, det_v = G._p5r_vs_control(b0, M, ev, ref_v)
        d = det_v["per_rank"][0]
        dd = np.array([float(np.abs(M["label"][idx[t]] - np.asarray(ev[t]["label"], np.float64)).max()) for t in ev])
        r = {"lam": lam, "warm_start": warm, "blas_threads": thr, "n_common": d["n_common_rows"],
             "rowmax_quantiles": {"median": float(np.median(dd)), "p90": float(np.percentile(dd, 90)), "p99": float(np.percentile(dd, 99)), "max": float(dd.max())},
             "n_rows_bit_identical": int((dd == 0.0).sum()), "n_sim8_identical": d["n_sim8_identical"], "frac_sim8_identical": d["frac_sim8_identical"],
             "R-TOL-1": d["R-TOL-1"], "R-TOL-2": d["R-TOL-2"],
             "R-TOL-3_row_matched": {"pass": d["R-TOL-3"], "ratio_acc": d["tb6_ratio_acc"], "ratio_hr": d["tb6_ratio_hr"], "ref_acc": d["tb6_ref_acc"], "ref_hr": d["tb6_ref_hr"], "ref_source": d["tb6_ref_source"],
                                     "diff_acc": d["tb6_diff_acc"], "diff_hr": d["tb6_diff_hr"], "sim8_fail_rate_pod": d["sim8_fail_rate_pod"], "sim8_fail_rate_dev_eval": d["sim8_fail_rate_dev_eval"],
                                     "sim8_diff_pp": d["sim8_fail_rate_diff_pp"]},
             "verdict_json_context_only": d["verdict_json_context"]}
        res["arms"][name] = r
    res["verdicts"] = {}
    for name, lam, warm, thr in ARMS:
        r = res["arms"][name]
        if name.startswith("noise"):
            res["verdicts"][name] = {"must": "PASS R-TOL-1, R-TOL-2 and (row-matched) R-TOL-3", "result": "AS EXPECTED" if (r["R-TOL-1"] and r["R-TOL-2"] and r["R-TOL-3_row_matched"]["pass"]) else "UNEXPECTED: a noise arm FAILS"}
        elif name.startswith("reg"):
            fails = [c for c, v in (("R-TOL-2", r["R-TOL-2"]), ("R-TOL-3", r["R-TOL-3_row_matched"]["pass"])) if not v]
            res["verdicts"][name] = {"must": "FAIL R-TOL-2 or R-TOL-3", "fails": fails, "result": "AS EXPECTED" if ("R-TOL-2" in fails or "R-TOL-3" in fails) else "UNEXPECTED: a regression arm PASSES"}
        else:
            res["verdicts"][name] = {"must": "max abs diff 0.0 (sanity)", "result": "AS EXPECTED" if r["rowmax_quantiles"]["max"] == 0.0 else "UNEXPECTED"}
    json.dump(res, open(a.out, "w", newline="\n"), indent=1)
    print(json.dumps(res["verdicts"], indent=1))


def driver(a):
    os.makedirs(a.arms_dir, exist_ok=True)
    for name, lam, warm, thr in ARMS:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONUTF8="1", OPENBLAS_NUM_THREADS=str(thr), MKL_NUM_THREADS=str(thr), OMP_NUM_THREADS=str(thr), REFE_SIM_HZ="10")
        p = subprocess.run([sys.executable, __file__, "run", "--arm", name, "--lam", str(lam), "--warm", warm, "--out", f"{a.arms_dir}/{name}.jsonl"], env=env, capture_output=True, encoding="utf-8")
        print(name, "rc", p.returncode, (p.stdout.strip().splitlines() or [""])[-1], p.stderr[-300:] if p.returncode else "", flush=True)
        if p.returncode:
            return p.returncode
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "score", "all"))
    ap.add_argument("--arm"); ap.add_argument("--lam", type=float, default=3.0); ap.add_argument("--warm", default="fixture"); ap.add_argument("--out")
    ap.add_argument("--arms-dir", default="F:/tanitad_scratch/p5c_a1b/arms")
    a = ap.parse_args()
    if a.mode == "run":
        run_arm(a)
    elif a.mode == "score":
        score(a)
    else:
        rc = driver(a)
        if rc == 0:
            a.out = a.out or HERE + "/calibration_A1.json"
            score(a)
        sys.exit(rc)
