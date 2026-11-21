"""speedsrc_read.py -- PREREAD_V8F_SPEEDSRC (theory/speedsrc/PREREAD_V8F_SPEEDSRC.md).  CPU, zero training.

Where is T's speed-cell error concentrated?  The T1-kin first-tick loop of theory/consorc/cons_orc_read.py (same batch build as theory/otband/read/topk_read.py), reduced to:
  TOP1 (forward_topk, plan on T's own code), IDENT / OWN-ALL (controls), GT-SPEED-ONLY (own code, the five speed cells set to the GT code),
  T's per-cell speed error own - gt (normalised units and m/s) on the PRESENT speed cells, and label-side strata from the GT future / labels.
One npz per arm (per-window rows, written before any analysis), then the stratum analysis (paired episode-cluster machinery of taniteval.ci, B = 2000, seed 0).
Usage: speedsrc_read.py --tags PARITY_DAGOACC_s0 PARITY_DAGOACC_s1 --outdir <dir> [--smoke N]
"""
import argparse
import json
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "otband", "read"))
import topk_read as TK  # noqa: E402  (sets TREE first on sys.path; asserts tanitad comes from it)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from topk_read import BL, BANDS, D, M, K, free_gb, sha256  # noqa: E402

# ----------------------------------------------------------------------------------------- strata: ALL THRESHOLDS FIXED HERE AND IN THE CARD BEFORE ANY FULL-RUN NUMBER
DT = 0.2
V_STOP_END = 0.5            # STOP: end speed (mean of knots 26-29, 5.2-6.0 s) < 0.5 m/s
DV_DECEL, DV_ACCEL = -1.0, 1.0   # dv = end speed - speed at 2 s (mean of knots 8-11, 1.6-2.4 s); DECEL dv < -1, ACCEL dv > +1, else CRUISE
LEAD_BINS = (15.0, 40.0)    # lead_gap_m cell present: < 15 m, 15-40 m, >= 40 m
CURV_BINS = (0.005, 0.02)   # max |curvature| 1/m over knots 10-27 (2.2-5.6 s): < 0.005, 0.005-0.02, >= 0.02
CURV_MIN_CHORD_M = 0.8      # a curvature sample needs all three centred chords > 0.8 m (else it is skipped; no valid sample = 0)
V0_BINS = (4.0, 12.0)       # speed at t0 (p10f[:, -1, 0], the dump's v0): < 4, 4-12, >= 12 m/s
TL_GOALS = ("TRAFFIC_LIGHT_REACT", "TRAFFIC_LIGHT_REACT_RED", "TRAFFIC_LIGHT_REACT_YELLOW", "TRAFFIC_LIGHT_REACT_GREEN")
STOP_GOAL = "STOP_POINT"
MIN_STRATUM_WINDOWS = 50
MASS_SHARE_MIN, RATIO_MIN = 0.30, 2.0
NB, SEED = 2000, 0

C0 = M.N_GOAL + M.N_LAT + M.N_LON
SP = list(M.CODE_KINDS["speed"])            # absolute code indices of the five speed cells
SP_COLS = [i - C0 for i in SP]
SP_NAMES = [M.CONS_NAMES[c] for c in SP_COLS]
GI = {g: i for i, g in enumerate(D.GOAL22)}
assert all(g in GI for g in TL_GOALS + (STOP_GOAL,)), "goal vocabulary changed"
LEAD_COL = M.CI["lead_gap_m"]
STRATA_VARS = {
    "regime": ("STOP", "DECEL", "ACCEL", "CRUISE"),
    "lead": ("LEAD_NONE", "LEAD_LT15", "LEAD_15_40", "LEAD_GE40"),
    "tlstop": ("TL", "STOP_POINT", "NO_TL_STOP"),
    "curv": ("CURV_LO", "CURV_MID", "CURV_HI"),
    "v0": ("V0_LO", "V0_MID", "V0_HI"),
}


def wait_free(gb, max_s):
    t0 = time.time()
    while free_gb() < gb:
        if time.time() - t0 > max_s:
            raise MemoryError(f"free physical memory stayed below {gb} GB for {max_s} s")
        time.sleep(30)


def tag_paths(tag):
    A = TK.parse_argv(f"{TK.T1}/argv_{tag}_m10_causal_heads.txt")
    return A, A["ckpt"], f"{TK.T1}/dump_{tag}_m10_causal_heads"


def gt_speed_curv(g):
    """g [N, 30, 2] GT ego-frame waypoints at 0.2 s -> (v2, v_end, kappa_max) from the GT future only."""
    g = np.asarray(g, np.float64)
    P = np.concatenate([np.zeros((len(g), 1, 2)), g], 1)
    v = np.linalg.norm(np.diff(P, axis=1), axis=-1) / DT                     # [N, 30]; v[:, i] between knots i-1 and i (knot -1 = origin)
    v2, v_end = v[:, 8:12].mean(1), v[:, 26:30].mean(1)
    kmax = np.zeros(len(g))
    for n in range(len(g)):
        best = 0.0
        for j in range(10, 28):
            c_a, c_b, c_c = g[n, j + 2] - g[n, j], g[n, j + 1] - g[n, j - 1], g[n, j] - g[n, j - 2]
            la, lb, lc = np.linalg.norm(c_a), np.linalg.norm(c_b), np.linalg.norm(c_c)
            if min(la, lb, lc) <= CURV_MIN_CHORD_M:
                continue
            dpsi = np.arctan2(c_a[1], c_a[0]) - np.arctan2(c_c[1], c_c[0])
            dpsi = (dpsi + np.pi) % (2 * np.pi) - np.pi
            best = max(best, abs(dpsi) / lb)
        kmax[n] = best
    return v2, v_end, kmax


def build_strata(g, v0, cons_gt, tl_hit, stop_hit):
    """-> {var: int array of level index into STRATA_VARS[var]} (labels / GT future only)."""
    v2, v_end, kmax = gt_speed_curv(g)
    dv = v_end - v2
    reg = np.where(v_end < V_STOP_END, 0, np.where(dv < DV_DECEL, 1, np.where(dv > DV_ACCEL, 2, 3)))
    lg = cons_gt[:, LEAD_COL]
    lead = np.where(np.isnan(lg), 0, np.where(lg < LEAD_BINS[0], 1, np.where(lg < LEAD_BINS[1], 2, 3)))
    tls = np.where(tl_hit, 0, np.where(stop_hit, 1, 2))
    curv = np.where(kmax < CURV_BINS[0], 0, np.where(kmax < CURV_BINS[1], 1, 2))
    v0b = np.where(v0 < V0_BINS[0], 0, np.where(v0 < V0_BINS[1], 1, 2))
    return {"regime": reg, "lead": lead, "tlstop": tls, "curv": curv, "v0": v0b}, {"v2": v2, "v_end": v_end, "kappa_max": kmax}


def run_arm(E, tag, smoke=0):
    A, ck_path, ddir = tag_paths(tag)
    got = sha256(ck_path)
    man = os.path.dirname(ck_path) + "/BANK_MANIFEST.json"
    man_ok = None
    if os.path.exists(man):
        man_ok = json.load(open(man))["files"]["ckpt_last.pt"]["sha256"] == got
        assert man_ok, (tag, "checkpoint sha256 differs from its bank manifest")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**ck["cfg"])
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    model.load_state_dict(ck["model"])
    model.eval()
    sd = model.cons_sd[SP_COLS].numpy().astype(np.float64)
    tracks = D.load_native_tracks(A["tracks"])
    J = D.load_join(A["eval-join"])
    store = D.ClipStore(A["eval-cache"], "eval", lru=32, grid=ck.get("grid", "pool4x10"), kin_mode="yaw", tracks=tracks)
    wins = D.eval_grid_windows(A["eval-grid"], J)
    if smoke:
        wins = [wins[i] for i in np.linspace(0, len(wins) - 1, smoke).round().astype(int)]
    proxy = E._OneClipStore()
    ds = D.V8FMiniWindows(proxy, J, [], hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s, window_start=cfg.window_start)
    a_t, a_s = E.ages_for(0, A["schedule"])
    N = len(wins)
    nsp = len(SP)
    o = {"top1": np.zeros((N, 30, 2)), "ident": np.zeros((N, 30, 2)), "own_all": np.zeros((N, 30, 2)), "gso": np.zeros((N, 30, 2)), "g": np.zeros((N, 30, 2)),
         "has": np.zeros(N, bool), "ci": np.zeros(N, np.int64), "ws": np.zeros(N, np.int64), "v0": np.zeros(N),
         "err_norm": np.full((N, nsp), np.nan), "err_mps": np.full((N, nsp), np.nan), "gt_mps": np.full((N, nsp), np.nan), "own_mps": np.full((N, nsp), np.nan),
         "cons_gt": np.full((N, M.N_CONS), np.nan), "tl_hit": np.zeros(N, bool), "stop_hit": np.zeros(N, bool)}
    t0 = time.time()
    for s in range(0, N, 32):
        chunk = wins[s:s + 32]
        wait_free(3.0, 3600)
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
            c_gso = own.clone()
            c_gso[:, SP] = gt[:, SP]
            tr_gso = model.forward_diag(batch, c_gso)["traj30"]
            tr_id = model.forward_diag(batch, own)["traj30"]
            c_all = gt.clone()
            c_all[:, :] = own
            tr_all = model.forward_diag(batch, c_all)["traj30"]
            present = ~torch.isnan(lt["cons"])
            pm = (present[:, SP_COLS] & has[:, None]).numpy()
            en = (own[:, SP] - gt[:, SP]).double().numpy()
            gtn = gt[:, SP].double().numpy()
            ownn = own[:, SP].double().numpy()
            mu = model.cons_mu[SP_COLS].numpy().astype(np.float64)
            gy, gw = lt["goal_y"], lt["goal_w"]
            hit = ((gy > 0.5) & (gw > 0.5)).numpy() & has[:, None].numpy()
        sl = slice(s, s + len(chunk))
        o["top1"][sl] = r["traj30_top1"].float().numpy().astype(np.float64)
        o["ident"][sl] = tr_id.float().numpy().astype(np.float64)
        o["own_all"][sl] = tr_all.float().numpy().astype(np.float64)
        o["gso"][sl] = tr_gso.float().numpy().astype(np.float64)
        o["has"][sl] = has.numpy()
        o["v0"][sl] = batch["p10f"][:, -1, 0].double().numpy()
        o["err_norm"][sl] = np.where(pm, en, np.nan)
        o["err_mps"][sl] = np.where(pm, en * sd, np.nan)
        o["gt_mps"][sl] = np.where(pm, gtn * sd + mu, np.nan)
        o["own_mps"][sl] = np.where(pm, ownn * sd + mu, np.nan)
        o["cons_gt"][sl] = np.where(has.numpy()[:, None], lt["cons"].double().numpy(), np.nan)
        o["tl_hit"][sl] = hit[:, [GI[g] for g in TL_GOALS]].any(1)
        o["stop_hit"][sl] = hit[:, GI[STOP_GOAL]]
        for i, (ci, tt, _, _) in enumerate(chunk):
            o["g"][s + i] = store.get(str(J["clip"][ci]))["gt"][tt]
            o["ci"][s + i], o["ws"][s + i] = ci, tt
        print(f"  {tag} [{s + len(chunk)}/{N}] {time.time() - t0:.0f}s free {free_gb():.1f} GB", flush=True)
    o["ckpt_sha256"], o["manifest_ok"] = got, man_ok
    return o


def analyse(o, eid, n_boot=NB):
    """The stratum read of one arm. Rows with has & >= 1 present speed cell. Every number is MEASURED on that arm's windows."""
    err = o["err_mps"]
    npres = np.isfinite(err).sum(1)
    ok = o["has"] & (npres > 0)
    sq = np.where(np.isfinite(err), err ** 2, 0.0)
    mass_w = sq.sum(1)                                                       # per-window sum of squared m/s error over present cells
    mse_w = np.where(npres > 0, mass_w / np.maximum(npres, 1), np.nan)       # per-window mean
    ade1 = BL.band_rows(o["top1"], o["g"], *BANDS["tactical_2_6s"])["ade_m"]
    adeg = BL.band_rows(o["gso"], o["g"], *BANDS["tactical_2_6s"])["ade_m"]
    gain = adeg - ade1                                                       # negative = GT speed cells help
    S, names = build_strata(o["g"], o["v0"], o["cons_gt"], o["tl_hit"], o["stop_hit"])[0], []
    cols, names, var_of = [], [], []
    for v, lv in STRATA_VARS.items():
        for k, nm in enumerate(lv):
            cols.append((S[v] == k) & ok)
            names.append(nm)
            var_of.append(v)
    I = np.stack(cols, 1).astype(np.float64)                                 # [N, strata]
    idx = np.where(ok)[0]
    I, mse_w, mass_w, gain, eid_ok = I[idx], mse_w[idx], mass_w[idx], gain[idx], np.asarray(eid)[idx]
    tot_mass = mass_w.sum()

    def stat(sel):
        sel = np.asarray(sel)
        Is, m, w, gn = I[sel], mse_w[sel], mass_w[sel], gain[sel]
        n_in = Is.sum(0)
        n_out = len(sel) - n_in
        mean_in = (Is * m[:, None]).sum(0) / np.maximum(n_in, 1)
        mean_out = ((1 - Is) * m[:, None]).sum(0) / np.maximum(n_out, 1)
        gain_in = (Is * gn[:, None]).sum(0) / np.maximum(n_in, 1)
        out = {}
        for j, nm in enumerate(names):
            out[f"diff:{nm}"] = mean_in[j] - mean_out[j] if (n_in[j] > 0 and n_out[j] > 0) else np.nan
            out[f"gain:{nm}"] = gain_in[j] if n_in[j] > 0 else np.nan
        return out
    r = BL.CI.stat_cluster_bootstrap(list(range(len(idx))), eid_ok, stat, n_boot=n_boot, seed=SEED)
    res = {"n_windows_read": int(len(idx)), "n_episodes": int(len(np.unique(eid_ok))), "total_mass_mps2": float(tot_mass),
           "mean_mse_all_mps2": float(mse_w.mean()) if len(idx) else None, "n_has": int(o["has"].sum()), "strata": {}}
    for j, nm in enumerate(names):
        n_in = int(I[:, j].sum())
        ms = float((I[:, j] * mass_w).sum() / tot_mass) if tot_mass > 0 else None
        ws = n_in / max(len(idx), 1)
        d, gq = r[f"diff:{nm}"], r[f"gain:{nm}"]
        qual = bool(n_in >= MIN_STRATUM_WINDOWS and ms is not None and ms >= MASS_SHARE_MIN and ws > 0 and ms / ws >= RATIO_MIN
                    and "lo" in d and d["lo"] > 0)
        res["strata"][nm] = {"var": var_of[j], "n_windows": n_in, "window_share": ws, "mass_share": ms, "mass_over_window_share": (ms / ws if ws > 0 and ms is not None else None),
                             "mean_mse_in": float((I[:, j] * mse_w).sum() / n_in) if n_in else None,
                             "diff_in_minus_complement_mps2": {k: d.get(k) for k in ("point", "lo", "hi")},
                             "gt_speed_only_tactical_ade_gain_m": {k: gq.get(k) for k in ("point", "lo", "hi")},
                             "qualifies": qual}
    res["qualifying"] = [nm for nm in names if res["strata"][nm]["qualifies"]]
    res["per_cell_mse_mps2"] = {nm: (float(np.nanmean(err[ok][:, c] ** 2)) if np.isfinite(err[ok][:, c]).any() else None) for c, nm in enumerate(SP_NAMES)}
    res["per_cell_n_present"] = {nm: int(np.isfinite(err[ok][:, c]).sum()) for c, nm in enumerate(SP_NAMES)}
    return res


def verdict(arms):
    s0, s1 = arms.get("PARITY_DAGOACC_s0"), arms.get("PARITY_DAGOACC_s1")
    if not (s0 and s1):
        return {"status": "NOT-EVALUATED", "reason": "both PARITY_DAGOACC seeds are needed"}
    both = [nm for nm in s0["qualifying"] if nm in s1["qualifying"]]
    return {"outcome": "CONCENTRATED" if both else "DIFFUSE", "strata_qualifying_at_both_seeds": both,
            "qualifying_s0": s0["qualifying"], "qualifying_s1": s1["qualifying"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--smoke", type=int, default=0, help="N evenly spaced windows; dump controls skipped (a code check, never a result)")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    E = TK.load_eval_module()
    res = {"tool": "speedsrc_read.py", "card": "theory/speedsrc/PREREAD_V8F_SPEEDSRC.md", "tier": "T1-kin first tick, oracle diagnostic, never driving performance; STRATEGIC: PAUSED (PI item 25)",
           "estimator": "episode-cluster bootstrap (taniteval.ci.stat_cluster_bootstrap), B=2000, seed 0, on windows with lab_t.has and >= 1 present speed cell; answers 'another draw of episodes' only",
           "speed_cells": SP_NAMES, "thresholds": {"V_STOP_END": V_STOP_END, "DV": [DV_DECEL, DV_ACCEL], "LEAD_BINS": LEAD_BINS, "CURV_BINS": CURV_BINS, "V0_BINS": V0_BINS,
                                                   "MIN_STRATUM_WINDOWS": MIN_STRATUM_WINDOWS, "MASS_SHARE_MIN": MASS_SHARE_MIN, "RATIO_MIN": RATIO_MIN},
           "tree": TK.TREE, "smoke": a.smoke, "controls": {}, "arms": {}}
    out_json = os.path.join(a.outdir, "SPEEDSRC_SMOKE.json" if a.smoke else "SPEEDSRC.json")
    for tag in a.tags:
        wait_free(4.0, 6 * 3600)
        print(f"arm {tag}", flush=True)
        o = run_arm(E, tag, a.smoke)
        _, _, ddir = tag_paths(tag)
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
               "n_windows": int(len(o["ci"])), "n_has": int(o["has"].sum()), "n_present_speed_cells": int(np.isfinite(o["err_mps"]).sum())}
        S, aux = build_strata(o["g"], o["v0"], o["cons_gt"], o["tl_hit"], o["stop_hit"])
        npz = os.path.join(a.outdir, f"{'smoke_' if a.smoke else ''}speedsrc_{tag}.npz")
        ade1 = BL.band_rows(o["top1"], o["g"], *BANDS["tactical_2_6s"])["ade_m"]
        adeg = BL.band_rows(o["gso"], o["g"], *BANDS["tactical_2_6s"])["ade_m"]
        np.savez(npz, clip_index=o["ci"], ws=o["ws"], eid=eid, has=o["has"], v0=o["v0"], err_norm=o["err_norm"], err_mps=o["err_mps"], gt_mps=o["gt_mps"], own_mps=o["own_mps"],
                 speed_cells=np.array(SP_NAMES), ade_top1_tac=ade1, ade_gso_tac=adeg, gso_gain_tac=adeg - ade1, cons_gt=o["cons_gt"], tl_hit=o["tl_hit"], stop_hit=o["stop_hit"],
                 v2_gt=aux["v2"], v_end_gt=aux["v_end"], kappa_max_gt=aux["kappa_max"], **{f"strat_{k}": v for k, v in S.items()})
        res["controls"][tag] = ctl
        ctl["strata_level_counts_has"] = {v: {STRATA_VARS[v][k]: int(((S[v] == k) & o["has"]).sum()) for k in range(len(STRATA_VARS[v]))} for v in S}
        res["arms"][tag] = analyse(o, eid, NB)
        res["verdict"] = verdict(res["arms"]) if not a.smoke else {"status": "SMOKE: code check only, never a result"}
        with open(out_json, "w", encoding="utf-8") as f:                      # written after every arm (bank incrementally)
            json.dump(res, f, indent=1, default=float)
        print(json.dumps({"tag": tag, "controls": ctl, "qualifying": res["arms"][tag]["qualifying"]}), flush=True)


if __name__ == "__main__":
    main()
