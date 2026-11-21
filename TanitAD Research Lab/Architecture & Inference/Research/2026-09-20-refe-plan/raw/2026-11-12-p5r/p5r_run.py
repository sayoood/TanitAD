"""P5-R driver: the pre-registered dev-box run (SG/p5r/PREREG_P5R.md, FROZEN sha256 cfd23e06..., 2026-11-12T08:03:43Z).

CPU only, nuPlan venv (C:/Users/Admin/navsim-crun/venv). No git writes. Stages (each writes <work>/<stage>.json):
  prep      parse the bank + Tc20s labels, full-bank raw learnability, the log split and the row samples (card §3)
  controls  C-R1..C-R5 on the CONTROL rows + timings + the budget rule (card §8, §3); GATE_FAIL stops the chain
  tune      every lam in the grid on the TUNING rows of one rank (card §4)        [--rank k]
  select    lam* by the committed rule, written BEFORE any EVAL row is solved   (card §4)
  eval      lam* on the EVAL rows of one rank (card §6)                         [--rank k]
  verdict   B1, B1p, B2-B5 per rank, comparators, families, cost formula (card §5-§7, §9)
  all       the chain; tune/eval run rank 1 in ONE child process and rank 0 inline (<= 2 solver processes)
  smoke     plumbing check on TUNING rows only (controls code on 2 rows/rank, tune+eval code on 1 row/rank, fixed lam)
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

for _k, _v in (("CUDA_VISIBLE_DEVICES", "-1"), ("OMP_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1"),
               ("OPENBLAS_NUM_THREADS", "1"), ("REFE_SIM_HZ", "10"), ("PYTHONIOENCODING", "utf-8"),
               ("NUPLAN_MAPS_ROOT", "E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0")):
    os.environ.setdefault(_k, _v)

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p5r_labels as R  # noqa: E402  (sets the PK paths)

P5 = R.P5
BANKDIR = "F:/tanitad_scratch/p5c_tb6/bank"
LABELS = ["E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c/labels/out/labels_s0of2.jsonl",
          "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c/labels/out/labels_s1of2.jsonl"]
DBROOT = "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits"
N_T, N_E, N_CTRL = 250, 1200, 50
BUDGET_CPU_S = 3.0 * 3600
B2_MARGIN, B4_MARGIN, TB6_BAR, ADMIT = 0.010, 0.10, 2.0, 1.80


def jdump(obj, fp):
    tmp = fp + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, fp)


def jload(fp):
    with open(fp, encoding="utf-8") as f:
        return json.load(f)


def sha_row(traj):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(traj, np.float64)).tobytes()).hexdigest()


def split_of(log):
    return "TUNING" if int(hashlib.sha256(("p5r-split|" + log).encode("utf-8")).hexdigest()[:8], 16) % 4 == 0 \
        else "EVAL"


def row_hash(log, tok, rk):
    return hashlib.sha256(f"p5r-row|{log}|{tok}|{rk}".encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------------------------------------- prep
def stage_prep(work):
    import navtrain_scenarios as NS
    t0 = time.time()
    lab, n_lab_ok = {}, 0
    for fp in LABELS:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if "tc20s" in r:
                    lab[(r["log"], r["token"], int(r["rank"]))] = (r["traj_sha256"], r["tc20s"])
                    n_lab_ok += 1
    dbs = NS.index_dbs(db_root=DBROOT)
    lab_logs = {k[0] for k in lab}
    elig = sorted(lg for lg in lab_logs if lg in dbs and os.path.exists(dbs[lg]))
    eligs = set(elig)
    bank = {0: [], 1: []}
    pool = {("TUNING", 0): [], ("TUNING", 1): [], ("EVAL", 0): [], ("EVAL", 1): []}
    n_hz_bad = n_sha_bad = 0
    for rk in (0, 1):
        with open(f"{BANKDIR}/targets_rank{rk}.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if int(r.get("sim_hz", 0)) != 10:
                    n_hz_bad += 1
                traj = np.asarray(r["traj"], np.float64)
                a, h = R.learnability(traj)
                bank[rk].append((a, h))
                lg, tok = r["log_name"], r["token"]
                if lg in eligs and (lg, tok, rk) in lab:
                    sh, tc = lab[(lg, tok, rk)]
                    if sh != sha_row(traj):
                        n_sha_bad += 1
                        continue
                    pool[(split_of(lg), rk)].append((row_hash(lg, tok, rk), lg, tok, rk, r["traj"], tc))
    samples = {}
    for (sp, rk), rows in pool.items():
        rows.sort(key=lambda x: x[0])
        n = N_T if sp == "TUNING" else N_E
        samples[f"{sp}_{rk}"] = [{"log": lg, "token": tok, "rank": rk, "traj": tr, "tc20s": tc}
                                 for _, lg, tok, rk, tr, tc in rows[:n]]
    ba = {rk: np.array(v) for rk, v in bank.items()}
    np.savez(os.path.join(work, "bank_learn.npz"), acc0=ba[0][:, 0], hr0=ba[0][:, 1], acc1=ba[1][:, 0],
             hr1=ba[1][:, 1])
    jdump(samples, os.path.join(work, "samples.json"))
    meta = {"n_label_ok": n_lab_ok, "n_eligible_logs": len(elig),
            "n_tuning_logs": sum(split_of(lg) == "TUNING" for lg in elig),
            "n_eval_logs": sum(split_of(lg) == "EVAL" for lg in elig),
            "pool_sizes": {f"{k[0]}_{k[1]}": len(v) for k, v in pool.items()},
            "sample_sizes": {k: len(v) for k, v in samples.items()},
            "bank_rows": {str(rk): len(v) for rk, v in bank.items()}, "sim_hz_not_10": n_hz_bad,
            "traj_sha_mismatch": n_sha_bad,
            "bank_raw_median": {str(rk): {"acc": float(np.median(ba[rk][:, 0])), "hr": float(np.median(ba[rk][:, 1]))}
                                for rk in (0, 1)},
            "seconds": round(time.time() - t0, 1)}
    jdump(meta, os.path.join(work, "prep.json"))
    print("ZZPREP_DONE", json.dumps(meta), flush=True)


# ------------------------------------------------------------------------------------------------------ helpers
class Sim:
    def __init__(self):
        import navsim_dac as ND
        import navtrain_scenarios as NS
        from p4_tracker_comp import tracked8, wrap
        from navsim_pdm.pdm_array_representation import state_array_to_coords_array
        from navsim_pdm.pdm_enums import BBCoordsIndex
        from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
        assert abs(ND.HALF_LENGTH - R.L_M_PER_RAD) == 0.0
        self.ND, self.NS, self.tracked8, self.wrap = ND, NS, tracked8, wrap
        self.coords, self.CENTER, self.pp = state_array_to_coords_array, int(BBCoordsIndex.CENTER), get_pacifica_parameters()
        self.dbs = NS.index_dbs(db_root=DBROOT)

    def real(self, ego):
        return lambda Pb: self.tracked8(self.ND, Pb, ego)

    def shifted(self, ego):
        """C-R4: the 8 knots sampled 0.1 s early (sim[:, 4::5]), an off-by-one bug."""
        def f(Pb):
            sim = self.ND.simulated_states(Pb, ego)
            s = sim[:, 4::5][:, :8, :3]
            ra = ego.rear_axle
            c, sn = math.cos(-ra.heading), math.sin(-ra.heading)
            dx, dy = s[..., 0] - ra.x, s[..., 1] - ra.y
            return np.stack([dx * c - dy * sn, dx * sn + dy * c, self.wrap(s[..., 2] - ra.heading)], -1)
        return f

    def score(self, ego, polys, T, labels):
        ND = self.ND
        t = time.perf_counter()
        P8 = np.stack([ND.to_navsim(np.asarray(q, np.float64)) for q in labels])
        sim = ND.simulated_states(P8, ego)
        t_sim = time.perf_counter() - t
        co = self.coords(sim, self.pp)
        ci = [i for i in range(co.shape[2]) if i != self.CENTER]
        dac = ND.violations(polys, co[:, :, ci])
        ra = ego.rear_axle
        itp = ND.violations(polys, np.stack([ND.corners_global(ND.at_10hz(p), ra.x, ra.y, ra.heading) for p in P8]))
        comf = ND.comfortable_from_states(sim)
        ex = P5.ego_frame(sim, ego)
        out = []
        for i, q in enumerate(labels):
            e = R.resid(T, ex[i])
            out.append({"sim8": bool(dac[i]), "interp8": bool(itp[i]), "comfort": float(comf[i]),
                        "E_maxxy": float(np.linalg.norm(ex[i, :, :2] - T[:, :2], axis=1).max()),
                        "J_track": float(e @ e), "ev": P5.exec_vs_intended(ex[i], T),
                        "learn": list(R.learnability(np.asarray(q, np.float64)))})
        return out, {"t_score": time.perf_counter() - t, "t_score_sim": t_sim, "n": len(labels)}

    def s3(self, ego, label, s_solver):
        s_alone = self.tracked8(self.ND, self.ND.to_navsim(label)[None], ego)[0]
        return float(np.max(np.abs(R.sdiff(s_alone, s_solver))))


def sanity(res, traj):
    lab = res["label"]
    d = res["d"]
    lm = res["lam"]
    s5 = True
    if lm > 0 and not math.isinf(lm) and float(np.max(np.abs(d))) > 0:
        s5 = res["J_reg"] > 0.0
    if res["J"] is not None:
        s5 = s5 and abs(res["J"] - (res["J_track"] + res["J_reg"])) <= 1e-9 * max(1.0, abs(res["J"]))
    return {"S1_finite": bool(np.isfinite(lab).all()),
            "S2_yaw_branch": bool(float(np.max(np.abs(lab[:, 2] - traj[:, 2]))) < math.pi), "S5_lambda_reached": bool(s5)}


def rows_by_log(rows):
    by = {}
    for i, r in enumerate(rows):
        by.setdefault(r["log"], []).append(i)
    return by


def iter_rows(S, rows):
    """Yields (i, row, ego, polys, t_bld) grouped by log; build failures yield (i, row, None, error, t)."""
    for lg, idx in sorted(rows_by_log(rows).items()):
        t = time.perf_counter()
        try:
            toks = sorted({rows[i]["token"] for i in idx})
            scs = {sc._initial_lidar_token: sc for sc in S.NS.build_scenarios_for_log(S.dbs[lg], toks)}
            err = None
        except Exception as e:  # noqa: BLE001
            scs, err = {}, f"{type(e).__name__}: {str(e)[:150]}"
        t_log = (time.perf_counter() - t) / max(1, len(idx))
        for i in idx:
            r = rows[i]
            t1 = time.perf_counter()
            try:
                if err:
                    raise RuntimeError(err)
                sc = scs[r["token"]]
                ego = sc.get_ego_state_at_iteration(0)
                polys = S.ND.drivable_polygons(sc.map_api, ego)
                yield i, r, ego, polys, t_log + time.perf_counter() - t1
            except Exception as e:  # noqa: BLE001
                yield i, r, None, f"{type(e).__name__}: {str(e)[:150]}", t_log + time.perf_counter() - t1


def done_keys(fp):
    out = set()
    if os.path.exists(fp):
        with open(fp, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    out.add((r["log"], r["token"], r["rank"]))
                except (json.JSONDecodeError, KeyError):
                    pass
    return out


def fitab(calls):
    n = np.array([c[0] for c in calls], float)
    t = np.array([c[1] for c in calls], float)
    X = np.stack([np.ones_like(n), n], 1)
    a, b = np.linalg.lstsq(X, t, rcond=None)[0]
    return float(a), float(b)


def b4_pooled(rows_a, rows_b, logs):
    W, li = P5.boot_weights(logs)
    return P5.paired_ci(rows_a, rows_b, W, li, False)


# ---------------------------------------------------------------------------------------------------- controls
def stage_controls(work, n_ctrl=N_CTRL, out_name="controls.json"):
    S = Sim()
    smp = jload(os.path.join(work, "samples.json"))
    rows = smp["TUNING_0"][:n_ctrl] + smp["TUNING_1"][:n_ctrl]
    t0 = time.time()
    recs, calls, t_bld, t_scr, t_poly_free = [], [], [], [], []
    cfg = [("lam1e12", 1e12, "real"), ("lam0", 0.0, "real"), ("lam1_real", 1.0, "real"),
           ("lam1_ident", 1.0, "ident"), ("lam1_shift", 1.0, "shift")]
    for i, r, ego, polys, tb in iter_rows(S, rows):
        rec = {"log": r["log"], "token": r["token"], "rank": r["rank"]}
        if ego is None:
            rec["error"] = polys
            recs.append(rec)
            continue
        t_bld.append(tb)
        traj, tc = np.asarray(r["traj"], np.float64), np.asarray(r["tc20s"], np.float64)
        T = S.ND.to_navsim(traj)
        sols = {}
        try:
            for name, lm, kind in cfg:
                trk = {"real": S.real(ego), "ident": (lambda Pb: np.asarray(Pb, np.float64).copy()),
                       "shift": S.shifted(ego)}[kind]
                cnt = R.Counter(trk)
                sols[name] = R.solve(traj, tc - traj, [lm], cnt, S.ND.to_navsim)[lm]
                sols[name]["calls"] = cnt.calls
                if kind == "real":
                    calls += cnt.log
            inf = R.solve(traj, tc - traj, [math.inf], S.real(ego), S.ND.to_navsim)[math.inf]
            names = ["lam1e12", "lam0", "lam1_real", "lam1_ident", "lam1_shift"]
            sc, tm = S.score(ego, polys, T, [traj, tc] + [sols[k]["label"] for k in names])
            t_scr.append(tm)
            rec["raw"], rec["tc20s"] = sc[0], sc[1]
            for j, k in enumerate(names):
                z = sols[k]
                rec[k] = dict(sc[2 + j], max_abs_label_minus_traj=float(np.max(np.abs(z["label"] - traj))),
                              J=z["J"], J_track_solver=z["J_track"], J_reg=z["J_reg"], J_track0=z["J_track0"],
                              iters=z["iters"], accepted=z["accepted"], stop=z["stop"], calls=z["calls"],
                              **sanity(z, traj))
                if k in ("lam1e12", "lam0", "lam1_real"):
                    rec[k]["S3_dev"] = S.s3(ego, z["label"], z["s"])
            rec["laminf_bit_exact"] = bool(np.array_equal(inf["label"], traj))
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {str(e)[:150]}"
        recs.append(rec)
    ok = [r for r in recs if "error" not in r]
    n = len(ok)
    lm_real = ("lam1e12", "lam0", "lam1_real")
    g = {
        "C_R1_lam1e12_le_1e-6": all(r["lam1e12"]["max_abs_label_minus_traj"] <= 1e-6 for r in ok),
        "C_R1_inf_bit_exact": all(r["laminf_bit_exact"] for r in ok),
        "C_R2_descent_all_rows": all(r["lam0"]["J"] <= r["lam0"]["J_track0"] for r in ok),
        "C_R2_median_strictly_less": bool(np.median([r["lam0"]["J"] for r in ok])
                                          < np.median([r["lam0"]["J_track0"] for r in ok])),
        "C_R3_ident_known_value": all(r["lam1_ident"]["max_abs_label_minus_traj"] <= 1e-6 for r in ok),
    }
    logs = [r["log"] for r in ok]
    b4_id = b4_pooled([r["tc20s"]["E_maxxy"] for r in ok], [r["lam1_ident"]["E_maxxy"] for r in ok], logs)
    b4_sh = b4_pooled([r["tc20s"]["E_maxxy"] for r in ok], [r["lam1_shift"]["E_maxxy"] for r in ok], logs)
    b4_real = b4_pooled([r["tc20s"]["E_maxxy"] for r in ok], [r["lam1_real"]["E_maxxy"] for r in ok], logs)
    g["C_R3_B4_FAILS"] = b4_id["hi"] > B4_MARGIN
    g["C_R4_B4_FAILS"] = b4_sh["hi"] > B4_MARGIN
    s3max = max(r[k]["S3_dev"] for r in ok for k in lm_real) if ok else None
    g["S1_finite"] = all(r[k]["S1_finite"] for r in ok for k in lm_real)
    g["S2_yaw_branch"] = all(r[k]["S2_yaw_branch"] for r in ok for k in lm_real)
    g["S3_resim_le_1e-6"] = s3max is not None and s3max <= 1e-6
    g["S4_fallback_le_1pct"] = (len(recs) - n) <= 0.01 * len(recs)
    g["S5_lambda_reached"] = all(r[k]["S5_lambda_reached"] for r in ok for k in lm_real)
    g["zero_control_lam0_Jreg_exactly_0"] = all(r["lam0"]["J_reg"] == 0.0 for r in ok)
    a, b = fitab(calls) if calls else (None, None)
    tb = float(np.mean(t_bld)) if t_bld else None
    ts = [x["t_score"] - x["t_score_sim"] for x in t_scr]
    t_scr_fixed = float(np.mean(ts)) if ts else None
    kbar1 = float(np.mean([r["lam1_real"]["iters"] for r in ok])) if ok else None

    def tsim(m):
        return a + b * m

    t_row1 = tb + tsim(25) + kbar1 * tsim(3) + tsim(1) + t_scr_fixed + tsim(3)
    t_row9 = tb + tsim(25) + R.K_MAX * tsim(27) + 9 * tsim(1) + t_scr_fixed + tsim(11)
    ctrl_s = time.time() - t0
    nt, ne = N_T, N_E

    def proj(nt_, ne_):
        return ctrl_s + 2 * nt_ * t_row9 + 2 * ne_ * t_row1 + 360.0
    budget_status = "OK"
    if proj(nt, ne) > BUDGET_CPU_S:
        ne = 600
        for cand in range(N_E, 599, -100):
            if proj(nt, cand) <= BUDGET_CPU_S:
                ne = cand
                break
        if proj(nt, ne) > BUDGET_CPU_S:
            nt = 150
            for cand in range(N_T, 149, -50):
                if proj(cand, ne) <= BUDGET_CPU_S:
                    nt = cand
                    break
            if proj(nt, ne) > BUDGET_CPU_S:
                budget_status = "STOP_OVER_BUDGET"
    budget = {"a_s": a, "b_s": b, "t_bld_s": tb, "t_scr_fixed_s": t_scr_fixed, "Kbar_lam1": kbar1,
              "t_row1_s": t_row1, "t_row9_s_bound_Kmax": t_row9, "controls_s": round(ctrl_s, 1),
              "N_t": nt, "N_e": ne, "projected_cpu_h": proj(nt, ne) / 3600, "status": budget_status,
              "bank_scale": {"N": 150603, "cpu_h": 150603 * t_row1 / 3600,
                             "pod_wall_h_128c_eta0.8_rho1": 150603 * t_row1 / (3600 * 128 * 0.8),
                             "note": "P5-R only; P5-C's pod build adds iterate20 (the warm start) on top"},
              "formula": "t_row(G) = t_bld + t_sim(25) + sum_k t_sim(3 G_k) + t_scr; t_sim(n) = a + b n (card §2.3); "
                         "t_row1 adds one S3 solo call; t_row9 bounds K by K_MAX"}
    res = {"n_rows": len(recs), "n_ok": n, "gates": g, "all_pass": all(g.values()),
           "B4_identity_tracker": b4_id, "B4_shifted_tracker": b4_sh, "B4_real_lam1_descriptive": b4_real,
           "S3_max": s3max,
           "max_abs_label_minus_traj": {k: float(max(r[k]["max_abs_label_minus_traj"] for r in ok))
                                        for k in ("lam1e12", "lam1_ident")} if ok else None,
           "median_J_lam0": float(np.median([r["lam0"]["J"] for r in ok])) if ok else None,
           "median_J_track_tc20s_warm": float(np.median([r["lam0"]["J_track0"] for r in ok])) if ok else None,
           "budget": budget, "errors": [r for r in recs if "error" in r][:5], "rows": recs}
    jdump(res, os.path.join(work, out_name))
    jdump(budget, os.path.join(work, "budget.json" if out_name == "controls.json" else "budget_smoke.json"))
    print("ZZCONTROLS", "PASS" if res["all_pass"] else "GATE_FAIL", json.dumps(g), json.dumps(budget), flush=True)
    return res


# ------------------------------------------------------------------------------------------------- tune / eval
def stage_solve(work, rank, which, lams, n_rows, out_fp):
    S = Sim()
    smp = jload(os.path.join(work, "samples.json"))
    rows = smp[f"{which}_{rank}"][:n_rows]
    done = done_keys(out_fp)
    rows = [r for r in rows if (r["log"], r["token"], r["rank"]) not in done]
    t0 = time.time()
    n_ok = n_err = 0
    with open(out_fp, "a", encoding="utf-8", newline="\n") as f:
        for i, r, ego, polys, tb in iter_rows(S, rows):
            rec = {"log": r["log"], "token": r["token"], "rank": r["rank"], "t_bld": tb}
            try:
                if ego is None:
                    raise RuntimeError(polys)
                traj, tc = np.asarray(r["traj"], np.float64), np.asarray(r["tc20s"], np.float64)
                T = S.ND.to_navsim(traj)
                cnt = R.Counter(S.real(ego))
                t1 = time.perf_counter()
                sol = R.solve(traj, tc - traj, list(lams), cnt, S.ND.to_navsim)
                rec["t_solve"] = time.perf_counter() - t1
                rec["calls"], rec["props"] = cnt.calls, cnt.props
                sc, tm = S.score(ego, polys, T, [traj, tc] + [sol[lm]["label"] for lm in lams])
                rec["t_score"] = tm["t_score"]
                rec["raw"], rec["tc20s"] = sc[0], sc[1]
                rec["lam"] = {}
                for j, lm in enumerate(lams):
                    z = sol[lm]
                    rec["lam"][repr(lm)] = dict(sc[2 + j], J=z["J"], J_track_solver=z["J_track"], J_reg=z["J_reg"],
                                                J_track0=z["J_track0"], iters=z["iters"], accepted=z["accepted"],
                                                stop=z["stop"], S3_dev=S.s3(ego, z["label"], z["s"]),
                                                **sanity(z, traj))
                    if which != "TUNING":
                        rec["lam"][repr(lm)]["label"] = z["label"].tolist()
                n_ok += 1
            except Exception as e:  # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {str(e)[:150]}"
                n_err += 1
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if (n_ok + n_err) % 25 == 0:
                print(f"[{which} r{rank}] {n_ok + n_err}/{len(rows)} ok={n_ok} err={n_err} {time.time() - t0:.0f}s",
                      flush=True)
    print(f"ZZSOLVE_DONE {which} rank={rank} ok={n_ok} err={n_err} {time.time() - t0:.0f}s", flush=True)


def read_jsonl(fp):
    with open(fp, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def bank_learn(work):
    z = np.load(os.path.join(work, "bank_learn.npz"))
    return {0: (z["acc0"], z["hr0"]), 1: (z["acc1"], z["hr1"])}


def ratios(lab_learn, raw_learn, bank):
    """TB-6 (card §5.1) + projection (§5.3) from per-row learnability pairs."""
    ll, lt = np.asarray(lab_learn, float), np.asarray(raw_learn, float)
    pa = R.projection(ll[:, 0], lt[:, 0], bank[0])
    ph = R.projection(ll[:, 1], lt[:, 1], bank[1])
    return {"R_acc": float(np.median(ll[:, 0]) / np.median(lt[:, 0])),
            "R_hr": float(np.median(ll[:, 1]) / np.median(lt[:, 1])),
            "R_acc_proj": pa["R_proj"], "R_hr_proj": ph["R_proj"],
            "med_acc_label": float(np.median(ll[:, 0])), "med_hr_label": float(np.median(ll[:, 1])),
            "med_acc_raw_sample": float(np.median(lt[:, 0])), "med_hr_raw_sample": float(np.median(lt[:, 1])),
            "den_bank_acc": pa["den_bank_median_raw"], "den_bank_hr": ph["den_bank_median_raw"],
            "dropped_mass": [pa["dropped_mass"], ph["dropped_mass"]],
            "p95_acc_label": float(np.percentile(ll[:, 0], 95)), "p95_hr_label": float(np.percentile(ll[:, 1], 95)),
            "n": int(len(ll))}


def stage_select(work, out_name="lambda_star.json", tune_glob="tune_r{}.jsonl"):
    bank = bank_learn(work)
    table, adm = {}, []
    data = {rk: [r for r in read_jsonl(os.path.join(work, tune_glob.format(rk))) if "error" not in r] for rk in (0, 1)}
    lams = list(data[0][0]["lam"].keys()) if data[0] else []
    tc_check = {}
    for rk in (0, 1):
        rows = data[rk]
        tc_check[str(rk)] = ratios([r["tc20s"]["learn"] for r in rows], [r["raw"]["learn"] for r in rows], bank[rk])
        tc_check[str(rk)]["sim8_rate_raw"] = float(np.mean([r["raw"]["sim8"] for r in rows]))
        tc_check[str(rk)]["sim8_rate_tc20s"] = float(np.mean([r["tc20s"]["sim8"] for r in rows]))
        tc_check[str(rk)]["mean_E_maxxy_tc20s"] = float(np.mean([r["tc20s"]["E_maxxy"] for r in rows]))
    for key in lams:
        ent = {}
        ok_all = True
        for rk in (0, 1):
            rows = data[rk]
            x = ratios([r["lam"][key]["learn"] for r in rows], [r["raw"]["learn"] for r in rows], bank[rk])
            x["sim8_rate"] = float(np.mean([r["lam"][key]["sim8"] for r in rows]))
            x["mean_E_maxxy"] = float(np.mean([r["lam"][key]["E_maxxy"] for r in rows]))
            x["median_J_track"] = float(np.median([r["lam"][key]["J_track"] for r in rows]))
            x["mean_iters"] = float(np.mean([r["lam"][key]["iters"] for r in rows]))
            x["admissible_rank"] = max(x["R_acc"], x["R_hr"], x["R_acc_proj"], x["R_hr_proj"]) <= ADMIT
            ok_all = ok_all and x["admissible_rank"]
            ent[str(rk)] = x
        ent["admissible"] = ok_all
        table[key] = ent
        if ok_all:
            adm.append(float(key))
    lam_star = min(adm) if adm else 100.0
    res = {"lambda_star": lam_star, "status": "OK" if adm else "NO_ADMISSIBLE_LAMBDA",
           "rule": "smallest lam in the grid with R_acc, R_hr, R_acc_proj, R_hr_proj <= 1.80 on BOTH ranks (TUNING)",
           "n_tuning": {str(rk): len(data[rk]) for rk in (0, 1)}, "admissible": adm, "table": table,
           "tc20s_on_tuning_descriptive": tc_check, "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    jdump(res, os.path.join(work, out_name))
    print("ZZSELECT", json.dumps({k: res[k] for k in ("lambda_star", "status", "admissible", "n_tuning")}), flush=True)
    return res


# ------------------------------------------------------------------------------------------------------ verdict
def ci_ratio_of_medians(lab, raw, W, li):
    """Descriptive log-cluster bootstrap CI of a ratio of medians (card §5.1)."""
    w = W[:, li]
    out = []
    for v in (np.asarray(lab, float), np.asarray(raw, float)):
        o = np.argsort(v, kind="stable")
        c = np.cumsum(w[:, o], 1)
        k = (c < 0.5 * c[:, -1:]).sum(1)
        out.append(v[o][np.minimum(k, len(v) - 1)])
    r = out[0] / out[1]
    lo, hi = np.percentile(r, [2.5, 97.5])
    return [float(lo), float(hi)]


def stage_verdict(work, out_name="verdict.json", eval_glob="eval_r{}.jsonl"):
    bank = bank_learn(work)
    ls = jload(os.path.join(work, "lambda_star.json"))
    key = repr(float(ls["lambda_star"]))
    res = {"lambda_star": ls["lambda_star"], "lambda_status": ls["status"], "per_rank": {}}
    for rk in (0, 1):
        allr = read_jsonl(os.path.join(work, eval_glob.format(rk)))
        rows = [r for r in allr if "error" not in r]
        n_err = len(allr) - len(rows)
        logs = [r["log"] for r in rows]
        W, li = P5.boot_weights(logs)
        g = lambda arm, m: [((r[arm] if arm != "p5r" else r["lam"][key])[m]) for r in rows]  # noqa: E731
        p5r = [r["lam"][key] for r in rows]
        tb = ratios([z["learn"] for z in p5r], [r["raw"]["learn"] for r in rows], bank[rk])
        tb["ci_R_acc"] = ci_ratio_of_medians([z["learn"][0] for z in p5r], [r["raw"]["learn"][0] for r in rows], W, li)
        tb["ci_R_hr"] = ci_ratio_of_medians([z["learn"][1] for z in p5r], [r["raw"]["learn"][1] for r in rows], W, li)
        tb_tc = ratios([r["tc20s"]["learn"] for r in rows], [r["raw"]["learn"] for r in rows], bank[rk])
        s8 = {a: np.array(g(a, "sim8"), float) for a in ("raw", "tc20s", "p5r")}
        em = {a: np.array(g(a, "E_maxxy"), float) for a in ("raw", "tc20s", "p5r")}
        b2 = P5.paired_ci(s8["tc20s"], s8["p5r"], W, li, True)
        b3 = P5.paired_ci(s8["raw"], s8["p5r"], W, li, True)
        b4 = P5.paired_ci(em["tc20s"], em["p5r"], W, li, False)
        s3max = max(z["S3_dev"] for z in p5r) if p5r else None
        sch = {"S1_finite": all(z["S1_finite"] for z in p5r), "S2_yaw_branch": all(z["S2_yaw_branch"] for z in p5r),
               "S3_resim_le_1e-6": s3max is not None and s3max <= 1e-6,
               "S4_fallback_le_1pct": n_err <= 0.01 * max(1, len(allr)),
               "S5_lambda_reached": all(z["S5_lambda_reached"] for z in p5r)}
        bars = {"B1": tb["R_acc"] <= TB6_BAR and tb["R_hr"] <= TB6_BAR,
                "B1p": tb["R_acc_proj"] <= TB6_BAR and tb["R_hr_proj"] <= TB6_BAR,
                "B2": b2["hi"] <= B2_MARGIN, "B3": b3["hi"] < 0.0, "B4": b4["hi"] <= B4_MARGIN}
        gate = all(sch.values())
        verdict = ("GATE_FAIL" if not gate else "PASS" if all(bars.values()) else "FAIL")
        # comparators (card §9): Tsel-s derived, decides nothing
        touched = s8["raw"] == 1
        tsel_learn = [r["tc20s"]["learn"] if r["raw"]["sim8"] else r["raw"]["learn"] for r in rows]
        tsel = {"sim8_rate": float(np.mean(np.where(touched, s8["tc20s"], s8["raw"]))),
                "tb6": ratios(tsel_learn, [r["raw"]["learn"] for r in rows], bank[rk]),
                "n_touched": int(touched.sum())}
        if touched.sum() > 0:
            tl = np.array([r["tc20s"]["learn"] for r, t in zip(rows, touched) if t])
            rl = np.array([r["raw"]["learn"] for r, t in zip(rows, touched) if t])
            tsel["touched_rows_ratio_acc"] = float(np.median(tl[:, 0]) / np.median(rl[:, 0]))
            tsel["touched_rows_ratio_hr"] = float(np.median(tl[:, 1]) / np.median(rl[:, 1]))
        fam = {}
        for m in ("mean_abs_along", "along_4s", "mean_abs_cross", "heading_mae_deg", "max_xy"):
            va = {a: np.array([((r[a] if a != "p5r" else r["lam"][key])["ev"][m]) for r in rows], float)
                  for a in ("raw", "tc20s", "p5r")}
            fam[m] = {"mean": {a: float(v.mean()) for a, v in va.items()},
                      "p5r_minus_tc20s": P5.paired_ci(va["tc20s"], va["p5r"], W, li, False),
                      "p5r_minus_raw": P5.paired_ci(va["raw"], va["p5r"], W, li, False)}
        desc = {"interp8_rate": {a: float(np.mean(g(a, "interp8"))) for a in ("raw", "tc20s", "p5r")},
                "comfort_rate": {a: float(np.mean(g(a, "comfort"))) for a in ("raw", "tc20s", "p5r")},
                "interp8_p5r_minus_tc20s": P5.paired_ci(np.array(g("tc20s", "interp8"), float),
                                                        np.array(g("p5r", "interp8"), float), W, li, True),
                "sim8_rate": {a: float(s8[a].mean()) for a in s8}, "mean_E_maxxy": {a: float(em[a].mean()) for a in em},
                "median_E_maxxy": {a: float(np.median(em[a])) for a in em},
                "solver": {"mean_iters": float(np.mean([z["iters"] for z in p5r])),
                           "stops": {s: sum(z["stop"] == s for z in p5r) for s in ("k_max", "stall", "step_tol")},
                           "mean_t_solve_s": float(np.mean([r["t_solve"] for r in rows])),
                           "mean_props": float(np.mean([r["props"] for r in rows]))}}
        res["per_rank"][str(rk)] = {
            "n": len(rows), "n_logs": len(set(logs)), "n_errors": n_err, "verdict": verdict,
            "failing_bars": [b for b, v in bars.items() if not v], "bars": bars, "sanity": sch, "S3_max": s3max,
            "TB6_p5r": tb, "TB6_tc20s_same_rows": tb_tc, "B2_sim8_p5r_minus_tc20s": b2, "B3_sim8_p5r_minus_raw": b3,
            "B4_Emaxxy_p5r_minus_tc20s": b4, "tsel_s_secondary": tsel,
            "families": {"LONGITUDINAL": {k: fam[k] for k in ("mean_abs_along", "along_4s")},
                         "LATERAL": {k: fam[k] for k in ("mean_abs_cross", "heading_mae_deg", "max_xy")},
                         "TACTICAL": "NOT COMPUTABLE: a label construction has no manoeuvre decision (n = 0)",
                         "STRATEGIC": "PAUSED (PI ruling 2026-09-19)"},
            "descriptive": desc}
    v0 = res["per_rank"]["0"]["verdict"]
    res["P5R"] = v0
    res["rank1_verdict"] = res["per_rank"]["1"]["verdict"]
    res["estimator"] = "P5 §3 paired log-cluster bootstrap (p5_target_labels.boot_weights/paired_ci), 10,000 draws, " \
                       "seed 20260927, 95 % percentile, per rank over that rank's EVAL logs; answers 'another draw of logs'"
    res["tier"] = "label-construction diagnostic, tracker-in-the-loop (NAVSIM v1.1 PDM components, vendored), navtrain " \
                  "P7-bank rows with a local nuPlan DB, CPU dev box; NOT navtest, NOT PDMS"
    jdump(res, os.path.join(work, out_name))
    print("ZZP5R_VERDICT", v0, "rank1", res["rank1_verdict"], json.dumps(
        {rk: {"bars": res["per_rank"][rk]["bars"], "sanity": res["per_rank"][rk]["sanity"]} for rk in ("0", "1")}),
        flush=True)
    return res


# ------------------------------------------------------------------------------------------------- orchestration
def child(args, logfp):
    cmd = [sys.executable, os.path.abspath(__file__)] + args
    lf = open(logfp, "a", encoding="utf-8")
    return subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)


def run_pair(stage, work, lams, n_rows, glob_):
    """rank 1 in one child process, rank 0 inline: at most 2 solver processes."""
    lam_arg = ",".join(repr(x) for x in lams)
    p = child([stage, "--work", work, "--rank", "1", "--lams", lam_arg, "--n", str(n_rows)],
              os.path.join(work, f"{stage}_r1.log"))
    which = "TUNING" if stage == "tune" else "EVAL"
    stage_solve(work, 0, which, lams, n_rows, os.path.join(work, glob_.format(0)))
    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"{stage} rank-1 child rc={rc}")


def stage_all(work, hold_raw):
    os.makedirs(hold_raw, exist_ok=True)
    if not os.path.exists(os.path.join(work, "prep.json")):
        stage_prep(work)
    if not os.path.exists(os.path.join(work, "controls.json")):
        stage_controls(work)
    ctl = jload(os.path.join(work, "controls.json"))
    copy_out(work, hold_raw, ["prep.json", "controls.json", "budget.json"])
    if not ctl["all_pass"]:
        print("ZZP5R_GATE_FAIL controls", [k for k, v in ctl["gates"].items() if not v], flush=True)
        return 2
    bud = ctl["budget"]
    if bud["status"] != "OK":
        print("ZZP5R_STOP budget", json.dumps(bud), flush=True)
        return 3
    if not os.path.exists(os.path.join(work, "lambda_star.json")):
        run_pair("tune", work, list(R.LAMBDA_GRID), bud["N_t"], "tune_r{}.jsonl")
        stage_select(work)
    copy_out(work, hold_raw, ["lambda_star.json"])
    lam_star = float(jload(os.path.join(work, "lambda_star.json"))["lambda_star"])
    run_pair("eval", work, [lam_star], bud["N_e"], "eval_r{}.jsonl")
    stage_verdict(work)
    copy_out(work, hold_raw, ["verdict.json", "eval_r0.jsonl", "eval_r1.jsonl", "tune_r0.jsonl", "tune_r1.jsonl"])
    print("ZZP5R_CHAIN_DONE", flush=True)
    return 0


def copy_out(work, hold_raw, names):
    import shutil
    for n in names:
        src = os.path.join(work, n)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(hold_raw, n))


def stage_smoke(work):
    """Plumbing only, TUNING rows only (never EVAL): controls code on 2 rows/rank; tune and eval code paths on 1
    row/rank with a fixed lam; select + verdict code on those files. Writes *_smoke files, read for errors only."""
    stage_controls(work, n_ctrl=2, out_name="controls_smoke.json")
    smp_fp = os.path.join(work, "samples.json")
    for rk in (0, 1):
        stage_solve(work, rk, "TUNING", [0.1, 1.0], 1, os.path.join(work, f"smoke_tune_r{rk}.jsonl"))
        stage_solve(work, rk, "TUNING", [1.0], 1, os.path.join(work, f"smoke_evalpath_r{rk}.jsonl"))
    stage_select(work, out_name="lambda_star_smoke.json", tune_glob="smoke_tune_r{}.jsonl")
    print("ZZSMOKE_DONE", smp_fp, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("prep", "controls", "tune", "select", "eval", "verdict", "all", "smoke"))
    ap.add_argument("--work", default="F:/tanitad_scratch/p5r_agent/run")
    ap.add_argument("--hold-raw", default=os.path.join(HERE, "raw"))
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--lams", default=None)
    ap.add_argument("--n", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    if a.stage == "prep":
        stage_prep(a.work)
    elif a.stage == "controls":
        stage_controls(a.work)
    elif a.stage in ("tune", "eval"):
        lams = [float(x) for x in a.lams.split(",")]
        which = "TUNING" if a.stage == "tune" else "EVAL"
        stage_solve(a.work, a.rank, which, lams, a.n, os.path.join(a.work, f"{a.stage}_r{a.rank}.jsonl"))
    elif a.stage == "select":
        stage_select(a.work)
    elif a.stage == "verdict":
        stage_verdict(a.work)
    elif a.stage == "smoke":
        stage_smoke(a.work)
    else:
        return stage_all(a.work, a.hold_raw)
    return 0


if __name__ == "__main__":
    sys.exit(main())
