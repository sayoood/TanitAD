"""P5-C label driver: a thin wrapper around the UNCHANGED P5-S code (eval/p5_target_labels.py blob 294644df:
iterate20 / interp20, interp=cubic, ITERS=3).  CPU only.  No git writes.

modes
  label    Tc20s for every bank row whose log has a local nuPlan DB.  Output: jsonl shards, one record per row:
           {"log","token","rank","traj_sha256","tc20s":[20][3]}  or an error record {"log","token","rank","error"}.
           Rows whose log has no local DB get an explicit error record ("no local DB"), never dropped.
           Resumable: a log is skipped when its line is in done_<shard>.txt.
  repair   sim8 DAC failure of raw traj vs Tc20s on a fixed-rule sample (lowest sha256("log|token")).
  control  bit-identity of the label against the banked P5-S npz on its held-out rows (read from the dev-box bank).

  python p5c_labels.py label --shard 0 --nshard 2 --out DIR
  python p5c_labels.py repair --rank 1 --n 300 --out FILE.json
  python p5c_labels.py control --n 100
Key = (log, token, rank, sha256 of the row's own `traj` bytes = json.dumps(traj) compact, see traj_sha()).
"""
import os, sys, json, glob, time, hashlib, argparse, math
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("REFE_SIM_HZ", "10")
os.environ.setdefault("NUPLAN_MAPS_ROOT", "E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0")
import numpy as np

PKG = os.environ.get("P5C_PKG", "E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan")   # pod: the refe-plan tree
for p in (PKG + "/eval", PKG + "/refe", PKG + "/code"):
    sys.path.insert(0, p)
BANKDIR = os.environ.get("P5C_BANKDIR", "C:/Users/Admin/p5c_scratch/bank")   # dev-box copy deleted 2026-11-12 (re-pull from the pod source to re-run here); pod: /workspace/data/refe_navtrain/train_grow_cmd_3c_p7
DBROOT = os.environ.get("P5C_DBROOT", "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits")   # pod: /workspace/data/navtrain_dbs


def traj_sha(traj):
    """sha256 of the row's own traj as float64 little-endian bytes of the [20,3] array (exact, format independent)."""
    return hashlib.sha256(np.ascontiguousarray(np.asarray(traj, np.float64)).tobytes()).hexdigest()


def load_rows():
    rows = []
    for rk in (0, 1):
        with open(f"{BANKDIR}/targets_rank{rk}.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                rows.append((r["log_name"], r["token"], int(r["rank"]), r["traj"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("label", "repair", "control"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()

    import p5_target_labels as P
    import navsim_dac as ND
    import navtrain_scenarios as NS

    dbs = NS.index_dbs(db_root=DBROOT)
    t_start = time.time()

    if a.mode == "control":
        Z = np.load(P.PKG + "/raw/2026-10-11-p5s-smoothed/p5s_cubic.npz", allow_pickle=True)
        banked = {(str(l), str(t)): Z["labels20_t20_tc20_tw"][i] for i, (l, t) in enumerate(zip(Z["logs"], Z["tokens"]))}
        todo = {}
        for fp in glob.glob(f"{P.BANK}/**/targets_rank0.jsonl", recursive=True):
            for line in open(fp, encoding="utf-8"):
                r = json.loads(line)
                k = (r.get("log_name"), r.get("token"))
                if k in banked and int(r.get("sim_hz", 0)) == 10 and int(r.get("step", 0)) == 0:
                    todo[k] = r["traj"]
        keys = sorted(todo)[:: max(1, len(todo) // a.n)][: a.n]
        rows_by_log = {}
        for lg, tok in keys:
            rows_by_log.setdefault(lg, []).append((tok, 0, todo[(lg, tok)]))
        n = ident = 0; maxd = 0.0; errs = []
        for lg, items in sorted(rows_by_log.items()):
            scs = {sc._initial_lidar_token: sc for sc in NS.build_scenarios_for_log(dbs[lg], [t for t, _, _ in items])}
            for tok, rk, traj in items:
                try:
                    ego = scs[tok].get_ego_state_at_iteration(0)
                    t20 = np.asarray(traj, np.float64)
                    tc20, _, _, _ = P.iterate20(ND, t20, ND.to_navsim(t20), ego, {"knot_exact": True, "knot_checks": 0}, None, "cubic")
                    d = float(np.abs(tc20 - banked[(lg, tok)][1]).max()); maxd = max(maxd, d); ident += int(d == 0.0); n += 1
                except Exception as e:  # noqa
                    errs.append((lg, tok, repr(e)[:120]))
        res = {"n": n, "bit_identical": ident, "max_abs_diff": maxd, "errors": errs[:5], "n_errors": len(errs), "seconds": round(time.time() - t_start, 1)}
        print("ZZCONTROL", json.dumps(res))
        if a.out: json.dump(res, open(a.out, "w"), indent=1)
        return 0

    rows = load_rows()

    if a.mode == "label":
        os.makedirs(a.out, exist_ok=True)
        tag = f"s{a.shard}of{a.nshard}"
        by = {}
        for lg, tok, rk, traj in rows:
            by.setdefault(lg, []).append((tok, rk, traj))
        mylogs = [lg for i, lg in enumerate(sorted(by)) if i % a.nshard == a.shard]
        done_fp = f"{a.out}/done_{tag}.txt"
        done = set(open(done_fp, encoding="utf-8").read().split("\n")) if os.path.exists(done_fp) else set()
        outf = open(f"{a.out}/labels_{tag}.jsonl", "a", encoding="utf-8")
        cnt = {"ok": 0, "err": 0, "nodb": 0}
        print(f"START {tag} logs={len(mylogs)} already_done={len(done)}", flush=True)
        for li, lg in enumerate(mylogs):
            if lg in done:
                continue
            items = by[lg]
            db = dbs.get(lg, "")
            recs = []
            if not db or not os.path.exists(db):
                for tok, rk, traj in items:
                    recs.append({"log": lg, "token": tok, "rank": rk, "traj_sha256": traj_sha(traj), "error": "no local DB"})
                    cnt["nodb"] += 1
            else:
                try:
                    scs = {sc._initial_lidar_token: sc for sc in NS.build_scenarios_for_log(db, sorted({t for t, _, _ in items}))}
                except Exception as e:  # noqa
                    scs = {}
                    print("SCEN-FAIL", lg, repr(e)[:150], flush=True)
                egos = {}
                for tok, rk, traj in items:
                    base = {"log": lg, "token": tok, "rank": rk, "traj_sha256": traj_sha(traj)}
                    try:
                        if tok not in scs:
                            raise KeyError("token not built by build_scenarios_for_log")
                        if tok not in egos:
                            egos[tok] = scs[tok].get_ego_state_at_iteration(0)
                        t20 = np.asarray(traj, np.float64)
                        ctl = {"knot_exact": True, "knot_checks": 0}
                        tc20, _, _, _ = P.iterate20(ND, t20, ND.to_navsim(t20), egos[tok], ctl, None, "cubic")
                        if not ctl["knot_exact"] or not np.isfinite(tc20).all() or tc20.shape != (20, 3):
                            raise ValueError("control/finite/shape check failed")
                        base["tc20s"] = [[float(v) for v in p] for p in tc20]
                        base["knot_maxdev"] = float(ctl.get("knot_maxdev", 0.0)); base["knot_checks"] = int(ctl["knot_checks"])   # TB-2 audit (C2' <= 1e-9 is enforced above)
                        cnt["ok"] += 1
                    except Exception as e:  # noqa
                        base["error"] = f"{type(e).__name__}: {str(e)[:160]}"
                        cnt["err"] += 1
                    recs.append(base)
            for r in recs:
                outf.write(json.dumps(r) + "\n")
            outf.flush()
            with open(done_fp, "a", encoding="utf-8") as f:
                f.write(lg + "\n")
            if li % 10 == 0:
                print(f"[{li + 1}/{len(mylogs)}] {cnt} {time.time() - t_start:.0f}s", flush=True)
        outf.close()
        done_rec = {"shard": a.shard, "nshard": a.nshard, "logs": len(mylogs), "rows_expected": sum(len(by[l]) for l in mylogs), "counts": cnt,
                    "seconds": round(time.time() - t_start, 1)}
        done_rec["rows_written_total_this_run_plus_prior"] = None
        json.dump(done_rec, open(f"{a.out}/DONE_{tag}.json", "w"), indent=1)
        print("ZZLABEL_DONE", json.dumps(done_rec), flush=True)
        return 0

    # repair
    from navsim_pdm.pdm_array_representation import state_array_to_coords_array
    from navsim_pdm.pdm_enums import BBCoordsIndex
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    pp = get_pacifica_parameters()
    cand = [(hashlib.sha256(f"{lg}|{tok}".encode()).hexdigest(), lg, tok, rk, traj) for lg, tok, rk, traj in rows
            if rk == a.rank and lg in dbs]
    cand.sort(key=lambda x: x[0])
    sel = cand[: a.n]
    by = {}
    for h, lg, tok, rk, traj in sel:
        by.setdefault(lg, []).append((tok, traj))
    res_rows = []; errs = []
    for li, (lg, items) in enumerate(sorted(by.items())):
        scs = {sc._initial_lidar_token: sc for sc in NS.build_scenarios_for_log(dbs[lg], [t for t, _ in items])}
        for tok, traj in items:
            try:
                sc = scs[tok]; ego = sc.get_ego_state_at_iteration(0)
                polys = ND.drivable_polygons(sc.map_api, ego)
                t20 = np.asarray(traj, np.float64); T = ND.to_navsim(t20)
                tc20, _, _, _ = P.iterate20(ND, t20, T, ego, {"knot_exact": True, "knot_checks": 0}, None, "cubic")
                Pn = np.stack([T, ND.to_navsim(tc20)])
                sim = ND.simulated_states(Pn, ego)
                coords = state_array_to_coords_array(sim, pp)
                ci = [i for i in range(coords.shape[2]) if i != int(BBCoordsIndex.CENTER)]
                dac = ND.violations(polys, coords[:, :, ci])
                res_rows.append({"log": lg, "token": tok, "raw_sim8": bool(dac[0]), "tc20s_sim8": bool(dac[1])})
            except Exception as e:  # noqa
                errs.append({"log": lg, "token": tok, "error": f"{type(e).__name__}: {str(e)[:120]}"})
        if li % 20 == 0:
            print(f"[{li + 1}/{len(by)}] rows={len(res_rows)} err={len(errs)} {time.time() - t_start:.0f}s", flush=True)
    n = len(res_rows)
    raw = np.array([r["raw_sim8"] for r in res_rows], int); tc = np.array([r["tc20s_sim8"] for r in res_rows], int)
    # log-cluster bootstrap of the paired difference (raw - tc20s), 10,000 draws, seed 20261111
    logs = sorted({r["log"] for r in res_rows}); li = {l: i for i, l in enumerate(logs)}
    idx = np.array([li[r["log"]] for r in res_rows])
    rng = np.random.default_rng(20261111)
    sr = np.bincount(idx, weights=raw, minlength=len(logs)); st = np.bincount(idx, weights=tc, minlength=len(logs)); cn = np.bincount(idx, minlength=len(logs)).astype(float)
    diffs = []
    for _ in range(10000):
        b = rng.integers(0, len(logs), len(logs))
        diffs.append((sr[b].sum() - st[b].sum()) / cn[b].sum())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    res = {"rank": a.rank, "rule": "lowest sha256('log|token') among rows of that rank whose log has a local DB", "n_requested": a.n, "n": n,
           "n_logs": len(logs), "n_errors": len(errs), "errors": errs[:5],
           "raw_sim8_fail_rate": float(raw.mean()), "tc20s_sim8_fail_rate": float(tc.mean()),
           "raw_minus_tc20s": float(raw.mean() - tc.mean()), "ci95_log_cluster_bootstrap": [float(lo), float(hi)],
           "both_fail": int(((raw == 1) & (tc == 1)).sum()), "raw_only": int(((raw == 1) & (tc == 0)).sum()),
           "tc_only": int(((raw == 0) & (tc == 1)).sum()), "seconds": round(time.time() - t_start, 1), "rows": res_rows}
    json.dump(res, open(a.out, "w"), indent=1)
    print("ZZREPAIR", json.dumps({k: v for k, v in res.items() if k != "rows"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
