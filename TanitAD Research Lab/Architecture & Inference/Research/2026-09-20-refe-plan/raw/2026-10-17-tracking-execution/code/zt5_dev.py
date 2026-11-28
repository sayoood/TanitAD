#!/usr/bin/env python3
"""zt5_dev.py -- PREREG_ZT5 section 5: the DEV pass on navtrain held-out sets (navtest-disjoint), one shard.

Per set: rebuilt nuPlan scenario -> NAVSIM's own metric-cache steps (refe/navsim_pdm_targets.metric_cache_parts) ->
all 64 executed proposals (A7) and their F_c versions: raw + simulated states, DAC, self-deviation delta -> picks of
the three scorers (N-s0 primary, N-s1, N-s2; rule navsim_v1_lane) -> F_b / F_bc picks over the rho grid -> the
section-3 counterfactual units -> FULL PDM scoring (navsim_pdm_targets.score_plans) of the baseline picks (gate DG1),
the picks under F_c and F_a, and every distinct F_bc pick.

    PYTHONPATH=<navsim v1.1>;<refe>;<this dir> navsim-crun python zt5_dev.py --shard 0/2 --out ../data/dev [--limit N]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import zt5_lib as L  # noqa: E402
import zt5_sim as Z  # noqa: E402

SETS = "D:/Projects/TanitAD-artifacts/_tlane_work/ho/refe_heldout/sets"
LABELS = "D:/Projects/TanitAD-artifacts/s1_read_cache/labels/x/heldout_pdm"
SCORES = {f"N-s{k}": f"D:/Projects/TanitAD-artifacts/s1_read_cache/arms/N-s{k}/dump/scores_update400_B.npz" for k in range(3)}
SUBS = ("nc", "dac", "ep", "ttc", "c", "pdms")
MAX_FAN = 2


def load_scores():
    sc = {}
    for name, p in SCORES.items():
        z = np.load(p)
        S = np.asarray(z["scores"], np.float32)
        sc[name] = {(str(lg), str(t), int(s), int(r)): S[i] for i, (lg, t, s, r, c) in
                    enumerate(zip(z["log"], z["token"], z["step"], z["rank"], z["ckpt_step"])) if int(c) == 10075}
    return sc


def set_logs():
    """{log: n_sets} over every banked held-out set (header fields only)."""
    cnt = {}
    for fp in sorted(glob.glob(f"{SETS}/onpolicy_*.jsonl")):
        for ln in open(fp, encoding="utf-8"):
            i = ln.index('"log_name": "') + 13
            lg = ln[i:ln.index('"', i)]
            cnt[lg] = cnt.get(lg, 0) + 1
    return cnt


def load_rows(logs):
    """the sets and banked labels of `logs` only; trajectories as compact numpy arrays."""
    rows, labs = [], {}
    for fp in sorted(glob.glob(f"{SETS}/onpolicy_*.jsonl")):
        for ln in open(fp, encoding="utf-8"):
            i = ln.index('"log_name": "') + 13
            if ln[i:ln.index('"', i)] not in logs:
                continue
            d = json.loads(ln)
            d["traj"] = np.asarray(d["traj"], np.float64)
            d["yaw"] = np.asarray(d["yaw"], np.float64)
            rows.append({k: d[k] for k in ("log_name", "token", "step", "rank", "ckpt_step", "traj", "yaw")})
        for ln in open(f"{LABELS}/pdm_{os.path.basename(fp)}", encoding="utf-8"):
            if not any(lg in ln[:200] for lg in logs):
                continue
            d = json.loads(ln)
            labs[(tuple(d["key"]), int(d["ckpt_step"]))] = {k: d[k] for k in SUBS}
    return rows, labs


def shard_logs(cnt_all, have, sn):
    cnt = {lg: n for lg, n in cnt_all.items() if lg in have}
    bins = [[] for _ in range(sn)]
    load = [0] * sn
    for lg in sorted(cnt, key=lambda x: (-cnt[x], x)):                  # greedy balance, deterministic
        i = int(np.argmin(load))
        bins[i].append(lg)
        load[i] += cnt[lg]
    return bins, cnt


def fam(Sx, Rx, ego):
    sxy, spsi = Z.ego_frame(Sx, ego)
    rxy, rpsi = Z.ego_frame(Rx, ego)
    f = L.families(sxy, spsi, rxy, rpsi)
    return [f["speed_mae"], f["cross_mae"], f["heading_mae_deg"], f["delta"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    si, sn = map(int, a.shard.split("/"))
    import navtrain_scenarios as NS
    import navsim_pdm_targets as PT
    scores = load_scores()
    dbs = NS.index_dbs()
    bins, cnt = shard_logs(set_logs(), set(dbs), sn)
    mylogs = bins[si]
    rows, labs = load_rows(set(mylogs))
    print(json.dumps({"shard": a.shard, "logs": mylogs, "sets": sum(cnt[x] for x in mylogs),
                      "dev_logs": len(cnt), "dev_sets": sum(cnt.values())}), flush=True)
    by = {}
    for r in rows:
        if r["log_name"] in mylogs:
            by.setdefault(r["log_name"], []).append(r)
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    tag = f"dev_{si}of{sn}" + (f"_lim{a.limit}" if a.limit else "")
    fu = open(od / f"{tag}_units.jsonl", "w", encoding="utf-8")
    recs = {k: [] for k in ("log", "token", "step", "rank", "ag1", "dg2", "dacS", "dacR", "dacSc", "dacRc", "delta",
                            "delta_c", "banked", "final_deg", "pick", "fb", "fbc", "sc_base", "sc_Fc", "sc_Fa",
                            "sc_Fbc", "dg1", "drawn_Fa", "fam_base", "fam_Fc", "fam_Fa", "fam_Fb", "fam_Fbc", "sec")}
    n_skip = 0
    t_start = time.time()
    done = 0
    arms = sorted(SCORES)
    for lg in mylogs:
        rs = by[lg]
        scs = {s.scenario_name: s for s in NS.build_scenarios_for_log(dbs[lg], sorted({r["token"] for r in rs}),
                                                                       history_rows=40, future_rows=100)}
        for r in rs:
            if a.limit and done >= a.limit:
                break
            sc = scs.get(r["token"])
            key = (r["log_name"], r["token"], int(r["step"]), int(r["rank"]))
            lab = labs.get((key, int(r["ckpt_step"])))
            if sc is None or lab is None or any(key not in scores[k] for k in arms):
                n_skip += 1
                continue
            t0 = time.time()
            parts = PT.metric_cache_parts(sc)
            ego, dm = parts[1], parts[5]
            idcs = Z.drivable_idx(dm)
            props20 = np.concatenate([np.asarray(r["traj"], np.float64), np.asarray(r["yaw"], np.float64)[..., None]], -1)
            P20a = props20.copy()
            P20a[..., -1, 2] = P20a[..., -2, 2]                                  # A7, identically
            P8 = PT.executed_plans8(props20, repair=True)
            P8c = np.stack([L.fix_heading_tangent(p) for p in P8])
            R, Rc = Z.raw_states(P8, ego), Z.raw_states(P8c, ego)
            S, Sc = Z.simulate(R, ego), Z.simulate(Rc, ego)
            dacS, dacR, dacSc, dacRc = (Z.dac(x, ego, dm, idcs) for x in (S, R, Sc, Rc))
            rxy, _ = Z.ego_frame(R, ego)
            sxy, _ = Z.ego_frame(S, ego)
            scxy, _ = Z.ego_frame(Sc, ego)
            delta = np.array([L.self_deviation(sxy[j], rxy[j]) for j in range(64)])
            delta_c = np.array([L.self_deviation(scxy[j], rxy[j]) for j in range(64)])
            banked = np.stack([np.asarray(lab[k], np.float64) for k in SUBS], -1)            # [64, 6]
            picks, fb, fbc = [], [], []
            for k in arms:
                s = L.selection_score(scores[k][key])
                pk = int(np.argmax(s))
                picks.append(pk)
                fb.append([L.fb_pick(s, delta, pk, rho) for rho in L.RHO_GRID])
                fbc.append([L.fb_pick(s, delta_c, pk, rho) for rho in L.RHO_GRID])
            # ---- scoring list (deduplicated)
            todo, idx = [], {}

            def want(tagv, plan):
                if tagv not in idx:
                    idx[tagv] = len(todo)
                    todo.append(plan)
                return idx[tagv]
            comp = {}
            for pk in picks:
                want(("base", pk), P8[pk])
                want(("Fc", pk), P8c[pk])
                if pk not in comp:
                    comp[pk] = Z.compensate(P8[pk], ego)
                want(("Fa", pk), comp[pk])
            for row in fbc:
                for j in row:
                    want(("Fc", j), P8c[j])
            res = PT.score_plans(parts, np.stack(todo))
            M = np.stack([np.asarray(res[k], np.float64) for k in SUBS], -1)                 # [n_todo, 6]
            sc_base = np.stack([M[idx[("base", pk)]] for pk in picks])
            sc_Fc = np.stack([M[idx[("Fc", pk)]] for pk in picks])
            sc_Fa = np.stack([M[idx[("Fa", pk)]] for pk in picks])
            sc_Fbc = np.stack([[M[idx[("Fc", j)]] for j in row] for row in fbc])
            dg1 = np.abs(np.round(sc_base, 5) - banked[picks]).max(axis=1)
            Ra = {pk: Z.raw_states(comp[pk][None], ego) for pk in comp}
            Sa = {pk: Z.simulate(Ra[pk], ego) for pk in comp}
            drawn_Fa = [float(Z.dac(Ra[pk], ego, dm, idcs)[0]) for pk in picks]
            fam_base = [fam(S[pk], R[pk], ego) for pk in picks]
            fam_Fc = [fam(Sc[pk], R[pk], ego) for pk in picks]
            fam_Fa = [fam(Sa[pk][0], R[pk], ego) for pk in picks]
            fam_Fb = [[fam(S[j], R[j], ego) for j in row] for row in fb]
            fam_Fbc = [[fam(Sc[j], R[j], ego) for j in row] for row in fbc]
            # ---- section-3 counterfactual units
            units = []
            for ai, pk in enumerate(picks):
                if banked[pk, 1] == 0.0 and dacR[pk] == 1.0:
                    units.append(("pick", pk, arms[ai]))
            tmf = np.flatnonzero((banked[:, 1] == 0.0) & (dacR == 1.0))
            if tmf.size:
                rng = np.random.default_rng(20261017)
                chosen = np.sort(rng.choice(tmf, size=min(MAX_FAN, tmf.size), replace=False))
                ctrl_pool = np.flatnonzero((dacS == 1.0) & (dacR == 1.0))
                plen = np.array([np.linalg.norm(np.diff(np.concatenate([np.zeros((1, 2)), P8[j, :, :2]]), axis=0),
                                                axis=-1).sum() for j in range(64)])
                fdeg = np.degrees(np.abs(L.wrap(P8[:, -1, 2])))
                for j in chosen:
                    units.append(("fan", int(j), None))
                    if ctrl_pool.size:
                        dd = np.hypot((fdeg[ctrl_pool] - fdeg[j]) / 10.0, (plen[ctrl_pool] - plen[j]) / 5.0)
                        units.append(("ctrl", int(ctrl_pool[int(np.argmin(dd))]), int(j)))
            for kind, j, extra in units:
                rec = {"log": lg, "token": r["token"], "step": int(r["step"]), "rank": int(r["rank"]), "kind": kind, "j": j,
                       "ref": extra, "banked_dac": float(banked[j, 1]), "dacS": float(dacS[j]), "dacR": float(dacR[j])}
                if kind == "ctrl":
                    rec["cf"] = {"base": float(dacS[j]), "raw": float(dacR[j]), "H": float(dacSc[j])}
                else:
                    rec["cf"], _ = Z.counterfactuals(P8[j], ego, dm, idcs, p20=P20a[j])
                rec.update(Z.descriptors(P8[j], ego, R[j], S[j]))
                fu.write(json.dumps(rec) + "\n")
            for k, v in (("log", lg), ("token", r["token"]), ("step", int(r["step"])), ("rank", int(r["rank"])),
                         ("ag1", Z.admissible_ego_ok(ego)), ("dg2", int((dacS == banked[:, 1]).sum())), ("dacS", dacS),
                         ("dacR", dacR), ("dacSc", dacSc), ("dacRc", dacRc), ("delta", delta), ("delta_c", delta_c),
                         ("banked", banked), ("final_deg", np.degrees(L.wrap(P8[:, -1, 2]))), ("pick", picks),
                         ("fb", fb), ("fbc", fbc), ("sc_base", sc_base), ("sc_Fc", sc_Fc), ("sc_Fa", sc_Fa),
                         ("sc_Fbc", sc_Fbc), ("dg1", dg1), ("drawn_Fa", drawn_Fa), ("fam_base", fam_base),
                         ("fam_Fc", fam_Fc), ("fam_Fa", fam_Fa), ("fam_Fb", fam_Fb), ("fam_Fbc", fam_Fbc),
                         ("sec", time.time() - t0)):
                recs[k].append(v)
            done += 1
            if done % 25 == 0:
                el = time.time() - t_start
                print(f"[{done}] {lg} {r['token']} {recs['sec'][-1]:.1f}s  elapsed {el:.0f}s  rate {el / done:.2f}s/set",
                      flush=True)
                fu.flush()
        if a.limit and done >= a.limit:
            break
    fu.close()
    out = {k: np.asarray(v) for k, v in recs.items()}
    out["arms"] = np.array(arms)
    out["rho_grid"] = np.array(L.RHO_GRID)
    out["subs"] = np.array(SUBS)
    out["n_skip"] = np.array(n_skip)
    np.savez_compressed(od / f"{tag}.npz", **out)
    print(f"ZZDEV_DONE {tag} n={done} skipped={n_skip} wall={time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
