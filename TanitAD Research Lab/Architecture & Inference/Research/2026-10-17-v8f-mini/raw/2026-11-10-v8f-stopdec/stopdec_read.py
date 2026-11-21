"""stopdec_read.py -- PREREAD_V8F_STOPDEC (theory/stopdec/PREREAD_V8F_STOPDEC.md).  CPU, zero training.

Does T's OWN tactical decision see the stop while its speed cells miss it?  The loop, window set, controls and strata of theory/speedsrc/speedsrc_read.py
(imported, not copied, for strata / thresholds / analysis helpers), plus T's own decision parts of forward_topk `code`:
  goal part  code[:, 0:22]  = sigmoid(goal_logits)  (independent per-goal probabilities, NOT a softmax / argmax; v8f_mini.py:968 and :890-892 in the read tree)
  lon part   code[:, 27:34] = softmax(lon_logits)   (7-way; D.LON9 = HOLD CREEP STOP FOLLOW DECELERATE ACCELERATE KEEP)
Usage: stopdec_read.py --tags PARITY_DAGOACC_s0 PARITY_DAGOACC_s1 --outdir <dir> [--smoke N]
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "speedsrc"))
import speedsrc_read as SS  # noqa: E402  (sets CUDA_VISIBLE_DEVICES=-1, OMP_NUM_THREADS=2; imports topk_read -> TREE tanitad)
import numpy as np  # noqa: E402
import torch  # noqa: E402
TK, M, D, BL, E_BANDS = SS.TK, SS.M, SS.D, SS.BL, SS.BANDS
K, free_gb, sha256 = SS.K, SS.free_gb, SS.sha256

# ----------------------------------------------------------------------------------------- DECISION DEFINITIONS: FIXED HERE AND IN THE CARD BEFORE ANY FULL-RUN NUMBER
P_STOP_MIN = 0.5                                   # "T predicts STOP_POINT": sigmoid goal prob of STOP_POINT >= 0.5 (the 0.5 of the GT mask `y > 0.5`, v8f_mini.py:313)
LON9 = tuple(D.LON9)
L_HOLD, L_STOP, L_DECEL = LON9.index("HOLD"), LON9.index("STOP"), LON9.index("DECELERATE")
N_G, N_LAT, N_LON = M.N_GOAL, M.N_LAT, M.N_LON
LON_SL = slice(N_G + N_LAT, N_G + N_LAT + N_LON)   # code[:, 27:34]
STOP_GI = SS.GI[SS.STOP_GOAL]
DECISIONS = ("goal_stop", "lon_stop", "lon_slow")  # goal_stop: p(STOP_POINT)>=0.5; lon_stop: argmax lon == STOP; lon_slow: argmax lon in {HOLD, STOP, DECELERATE}
GROUPS = ("STOP_POINT", "REG_STOP", "REG_DECEL")   # STOP_POINT = SPEEDSRC stratum tlstop==STOP_POINT (GT goal STOP_POINT and no TL goal); REG_* = SPEEDSRC regime strata
PRIMARY = ("STOP_POINT", "goal_stop")
MIN_GROUP_WINDOWS = 50
HIT_SEES, HIT_MISSES, RATIO_SEES = 0.6, 0.4, 2.0
NB, SEED = SS.NB, 0


def run_arm(E, tag, smoke=0):
    """speedsrc_read.run_arm, plus T's own goal probabilities and lon probabilities per window (own forward_topk code)."""
    A, ck_path, ddir = SS.tag_paths(tag)
    got = sha256(ck_path)
    man = os.path.dirname(ck_path) + "/BANK_MANIFEST.json"
    man_ok = None
    if os.path.exists(man):
        man_ok = json.load(open(man, encoding="utf-8"))["files"]["ckpt_last.pt"]["sha256"] == got
        assert man_ok, (tag, "checkpoint sha256 differs from its bank manifest")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**ck["cfg"])
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    model.load_state_dict(ck["model"])
    model.eval()
    sd = model.cons_sd[SS.SP_COLS].numpy().astype(np.float64)
    tracks = D.load_native_tracks(A["tracks"])
    J = D.load_join(A["eval-join"])
    store = D.ClipStore(A["eval-cache"], "eval", lru=32, grid=ck.get("grid", "pool4x10"), kin_mode="yaw", tracks=tracks)
    wins = D.eval_grid_windows(A["eval-grid"], J)
    if smoke:
        wins = [wins[i] for i in np.linspace(0, len(wins) - 1, smoke).round().astype(int)]
    proxy = E._OneClipStore()
    ds = D.V8FMiniWindows(proxy, J, [], hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s, window_start=cfg.window_start)
    a_t, a_s = E.ages_for(0, A["schedule"])
    N, nsp = len(wins), len(SS.SP)
    o = {"top1": np.zeros((N, 30, 2)), "ident": np.zeros((N, 30, 2)), "own_all": np.zeros((N, 30, 2)), "g": np.zeros((N, 30, 2)),
         "has": np.zeros(N, bool), "ci": np.zeros(N, np.int64), "ws": np.zeros(N, np.int64), "v0": np.zeros(N),
         "err_mps": np.full((N, nsp), np.nan), "cons_gt": np.full((N, M.N_CONS), np.nan), "tl_hit": np.zeros(N, bool), "stop_hit": np.zeros(N, bool),
         "goal_prob": np.zeros((N, N_G)), "lon_prob": np.zeros((N, N_LON)), "gt_lon_cls": np.zeros(N, np.int64)}
    t0 = time.time()
    for s in range(0, N, 32):
        chunk = wins[s:s + 32]
        SS.wait_free(3.0, 3600)
        st = E.make_states(store, tracks, J, chunk, A["nav"])
        items = [E.build_item(ds, proxy, x, 0, a_t, a_s, A["scene"], A["limit"], None) for x in st]
        full = D.collate(items)
        batch = {k: full[k] for k in E.MODEL_KEYS}
        lt = full["lab_t"]
        has = lt["has"].bool()
        with torch.no_grad():
            r = model.forward_topk(batch, K)
            own = r["code"]
            gt = model._code_gt(lt).to(own.dtype)
            gt = torch.where(has[:, None], gt, own)
            tr_id = model.forward_diag(batch, own)["traj30"]
            c_all = gt.clone()
            c_all[:, :] = own
            tr_all = model.forward_diag(batch, c_all)["traj30"]
            present = ~torch.isnan(lt["cons"])
            pm = (present[:, SS.SP_COLS] & has[:, None]).numpy()
            en = (own[:, SS.SP] - gt[:, SS.SP]).double().numpy()
            gy, gw = lt["goal_y"], lt["goal_w"]
            hit = ((gy > 0.5) & (gw > 0.5)).numpy() & has[:, None].numpy()
        sl = slice(s, s + len(chunk))
        o["top1"][sl] = r["traj30_top1"].float().numpy().astype(np.float64)
        o["ident"][sl] = tr_id.float().numpy().astype(np.float64)
        o["own_all"][sl] = tr_all.float().numpy().astype(np.float64)
        o["has"][sl] = has.numpy()
        o["v0"][sl] = batch["p10f"][:, -1, 0].double().numpy()
        o["err_mps"][sl] = np.where(pm, en * sd, np.nan)
        o["cons_gt"][sl] = np.where(has.numpy()[:, None], lt["cons"].double().numpy(), np.nan)
        o["tl_hit"][sl] = hit[:, [SS.GI[g] for g in SS.TL_GOALS]].any(1)
        o["stop_hit"][sl] = hit[:, STOP_GI]
        o["goal_prob"][sl] = own[:, :N_G].double().numpy()
        o["lon_prob"][sl] = own[:, LON_SL].double().numpy()
        o["gt_lon_cls"][sl] = lt["lon_cls"].numpy() if "lon_cls" in lt else -1
        for i, (ci, tt, _, _) in enumerate(chunk):
            o["g"][s + i] = store.get(str(J["clip"][ci]))["gt"][tt]
            o["ci"][s + i], o["ws"][s + i] = ci, tt
        print(f"  {tag} [{s + len(chunk)}/{N}] {time.time() - t0:.0f}s free {free_gb():.1f} GB", flush=True)
    o["ckpt_sha256"], o["manifest_ok"] = got, man_ok
    return o


def decisions(o):
    """-> {decision: bool [N]} from T's own code. Fixed definitions (see the constants above)."""
    lon_a = o["lon_prob"].argmax(1)
    return {"goal_stop": o["goal_prob"][:, STOP_GI] >= P_STOP_MIN,
            "lon_stop": lon_a == L_STOP,
            "lon_slow": np.isin(lon_a, [L_HOLD, L_STOP, L_DECEL])}, lon_a


def group_masks(S):
    return {"STOP_POINT": S["tlstop"] == SS.STRATA_VARS["tlstop"].index("STOP_POINT"),
            "REG_STOP": S["regime"] == SS.STRATA_VARS["regime"].index("STOP"),
            "REG_DECEL": S["regime"] == SS.STRATA_VARS["regime"].index("DECEL")}


def analyse(o, eid, n_boot=NB):
    err = o["err_mps"]
    npres = np.isfinite(err).sum(1)
    ok = o["has"] & (npres > 0)
    mass_w = np.where(np.isfinite(err), err ** 2, 0.0).sum(1)
    mse_w = np.where(npres > 0, mass_w / np.maximum(npres, 1), np.nan)
    S, _ = SS.build_strata(o["g"], o["v0"], o["cons_gt"], o["tl_hit"], o["stop_hit"])
    Dd, lon_a = decisions(o)
    Gm = group_masks(S)
    idx = np.where(ok)[0]
    mse, mass, eid_ok = mse_w[idx], mass_w[idx], np.asarray(eid)[idx]
    G = {g: Gm[g][idx] for g in GROUPS}
    Dm = {d: Dd[d][idx] for d in DECISIONS}
    keys = [(g, d) for g in GROUPS for d in DECISIONS]

    def stat(sel):
        sel = np.asarray(sel)
        out = {}
        m, w = mse[sel], mass[sel]
        for g, d in keys:
            gi, di = G[g][sel], Dm[d][sel]
            ing, outg = gi, ~gi
            h, ms = gi & di, gi & ~di
            with np.errstate(all="ignore"):
                n_g = ing.sum()
                comp = m[outg].mean() if outg.any() else np.nan
                out[f"hit:{g}:{d}"] = h.sum() / n_g if n_g else np.nan
                out[f"fpr:{g}:{d}"] = (di & outg).sum() / outg.sum() if outg.any() else np.nan
                out[f"mse_hit:{g}:{d}"] = m[h].mean() if h.any() else np.nan
                out[f"mse_miss:{g}:{d}"] = m[ms].mean() if ms.any() else np.nan
                out[f"ratio_hit_vs_comp:{g}:{d}"] = (m[h].mean() / comp) if h.any() and comp == comp and comp > 0 else np.nan
                out[f"miss_mass_share:{g}:{d}"] = w[ms].sum() / w[ing].sum() if n_g and w[ing].sum() > 0 else np.nan
        return out
    r = BL.CI.stat_cluster_bootstrap(list(range(len(idx))), eid_ok, stat, n_boot=n_boot, seed=SEED)
    res = {"n_windows_read": int(len(idx)), "n_episodes": int(len(np.unique(eid_ok))), "mean_mse_all_mps2": float(mse.mean()) if len(idx) else None, "groups": {}}
    pick = lambda k: {q: r[k].get(q) for q in ("point", "lo", "hi")}  # noqa: E731
    for g in GROUPS:
        n_g = int(G[g].sum())
        comp_mse = float(mse[~G[g]].mean()) if (~G[g]).any() else None
        res["groups"][g] = {"n_windows": n_g, "mean_mse_in": float(mse[G[g]].mean()) if n_g else None, "mean_mse_complement": comp_mse, "decisions": {}}
        for d in DECISIONS:
            res["groups"][g]["decisions"][d] = {
                "n_hit": int((G[g] & Dm[d]).sum()), "n_miss": int((G[g] & ~Dm[d]).sum()), "n_fp_in_complement": int((~G[g] & Dm[d]).sum()),
                **{q: pick(f"{q}:{g}:{d}") for q in ("hit", "fpr", "mse_hit", "mse_miss", "ratio_hit_vs_comp", "miss_mass_share")}}
    # lon decision census inside the groups (descriptive; no bar)
    res["lon_argmax_census"] = {g: {LON9[k]: int(((lon_a[idx] == k) & G[g]).sum()) for k in range(N_LON)} for g in GROUPS}
    res["goal_stop_prob_mean"] = {g: (float(o["goal_prob"][idx][G[g], STOP_GI].mean()) if G[g].any() else None) for g in GROUPS}
    return res


def verdict(arms):
    s0, s1 = arms.get("PARITY_DAGOACC_s0"), arms.get("PARITY_DAGOACC_s1")
    if not (s0 and s1):
        return {"status": "NOT-EVALUATED", "reason": "both PARITY_DAGOACC seeds are needed"}
    g, d = PRIMARY
    per = {}
    for tag, a in (("s0", s0), ("s1", s1)):
        x = a["groups"][g]
        dd = x["decisions"][d]
        if x["n_windows"] < MIN_GROUP_WINDOWS:
            per[tag] = {"class": "NOT-EVALUATED", "reason": f"{x['n_windows']} < {MIN_GROUP_WINDOWS} windows"}
            continue
        hit, ratio, mms = dd["hit"]["point"], dd["ratio_hit_vs_comp"], dd["miss_mass_share"]["point"]
        sees = hit is not None and hit >= HIT_SEES and ratio["point"] is not None and ratio["point"] >= RATIO_SEES and ratio["lo"] is not None and ratio["lo"] > 1.0
        miss = hit is not None and hit < HIT_MISSES and mms is not None and mms > 0.5
        per[tag] = {"class": "DECISION-SEES" if sees else ("DECISION-MISSES" if miss else "MIXED"), "hit": hit, "ratio_hit_vs_comp": ratio["point"], "ratio_lo": ratio["lo"], "miss_mass_share": mms}
    cls = {v["class"] for v in per.values()}
    out = cls.pop() if len(cls) == 1 and "NOT-EVALUATED" not in cls else ("NOT-EVALUATED" if "NOT-EVALUATED" in cls else "SEEDS-DISAGREE")
    return {"outcome": out, "primary": {"group": g, "decision": d}, "per_seed": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--smoke", type=int, default=0, help="N evenly spaced windows; dump controls skipped (a code check, never a result)")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    E = TK.load_eval_module()
    res = {"tool": "stopdec_read.py", "card": "theory/stopdec/PREREAD_V8F_STOPDEC.md", "tier": "T1-kin first tick, oracle diagnostic, never driving performance; STRATEGIC: PAUSED (PI item 25)",
           "estimator": "episode-cluster bootstrap (taniteval.ci.stat_cluster_bootstrap), B=2000, seed 0; answers 'another draw of episodes' only",
           "definitions": {"goal_stop": f"sigmoid goal prob of STOP_POINT (code[:, {STOP_GI}]) >= {P_STOP_MIN}", "lon_stop": "argmax of the 7-way lon softmax (code[:, 27:34]) == STOP",
                           "lon_slow": "argmax lon in {HOLD, STOP, DECELERATE}", "LON9": LON9, "primary": PRIMARY,
                           "bars": {"hit_sees": HIT_SEES, "hit_misses": HIT_MISSES, "ratio_sees": RATIO_SEES, "min_group_windows": MIN_GROUP_WINDOWS}},
           "tree": TK.TREE, "smoke": a.smoke, "controls": {}, "arms": {}}
    out_json = os.path.join(a.outdir, "STOPDEC_SMOKE.json" if a.smoke else "STOPDEC.json")
    for tag in a.tags:
        SS.wait_free(4.0, 6 * 3600)
        print(f"arm {tag}", flush=True)
        o = run_arm(E, tag, a.smoke)
        _, _, ddir = SS.tag_paths(tag)
        if a.smoke:
            Z, idx, eid = None, None, o["ci"]
        else:
            Z = BL.load_dir(ddir)
            key_d = {(int(c), int(w)): i for i, (c, w) in enumerate(zip(Z["clip_index"], Z["ws"]))}
            key_m = [(int(c), int(w)) for c, w in zip(o["ci"], o["ws"])]
            assert set(key_d) == set(key_m) and len(key_d) == len(key_m), "window set differs from the dump"
            idx = np.array([key_d[x] for x in key_m])
            eid = Z["eid"][idx]
        ctl = {"ckpt_sha256": o["ckpt_sha256"], "manifest_ok": o["manifest_ok"], "smoke": a.smoke,
               "top1_vs_dump_ol_max_abs_m": None if Z is None else float(np.abs(o["top1"] - Z["ol"][idx]).max()),
               "g_vs_dump_max_abs_m": None if Z is None else float(np.abs(o["g"] - Z["g"][idx]).max()),
               "v0_vs_dump_max_abs_mps": None if Z is None else float(np.abs(o["v0"] - Z["v0"][idx]).max()),
               "ident_vs_top1_max_abs_m": float(np.abs(o["ident"] - o["top1"]).max()),
               "own_all_vs_ident_max_abs_m": float(np.abs(o["own_all"] - o["ident"]).max()),
               "goal_prob_in_0_1": bool(((o["goal_prob"] >= 0) & (o["goal_prob"] <= 1)).all()),
               "lon_prob_rows_sum_to_1_max_dev": float(np.abs(o["lon_prob"].sum(1) - 1).max()),
               "n_windows": int(len(o["ci"])), "n_has": int(o["has"].sum()), "n_present_speed_cells": int(np.isfinite(o["err_mps"]).sum())}
        S, aux = SS.build_strata(o["g"], o["v0"], o["cons_gt"], o["tl_hit"], o["stop_hit"])
        Gm = group_masks(S)
        ctl["group_counts_has"] = {g: int((Gm[g] & o["has"]).sum()) for g in GROUPS}
        Dd, lon_a = decisions(o)
        npz = os.path.join(a.outdir, f"{'smoke_' if a.smoke else ''}stopdec_{tag}.npz")
        np.savez(npz, clip_index=o["ci"], ws=o["ws"], eid=eid, has=o["has"], v0=o["v0"], err_mps=o["err_mps"], goal_prob=o["goal_prob"], lon_prob=o["lon_prob"], lon_argmax=lon_a,
                 gt_lon_cls=o["gt_lon_cls"], tl_hit=o["tl_hit"], stop_hit=o["stop_hit"], cons_gt=o["cons_gt"], v_end_gt=aux["v_end"], v2_gt=aux["v2"],
                 **{f"dec_{k}": v for k, v in Dd.items()}, **{f"strat_{k}": v for k, v in S.items()})
        res["controls"][tag] = ctl
        res["arms"][tag] = analyse(o, eid, NB)
        res["verdict"] = verdict(res["arms"]) if not a.smoke else {"status": "SMOKE: code check only, never a result"}
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1, default=float)
        print(json.dumps({"tag": tag, "controls": ctl}), flush=True)


if __name__ == "__main__":
    main()
