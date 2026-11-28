"""topk_pair_read.py -- PREREAD_V8F_TOPK_PAIR (theory/topk/PREREAD_V8F_TOPK_PAIR.md).  CPU, zero training.

The OTBAND top-K reader's loop (theory/otband/read/topk_read.py, imported for its helpers; its run_arm is repeated here with ONE added decode) plus:
  PAIR-SEL  = the K-plan whose (lat, lon) pair equals the logged pair when it is among T's top 3, TOP1 otherwise;
  GT-PAIR   = forward_diag on T's own code with the lat / lon parts set to the one-hot of the logged pair.
Rows per band; paired episode-cluster bootstrap (B = 2000, seed 0) on the labelled windows.  Memory guard: 10 GB free before an arm, 7.5 GB before a chunk.
Usage: topk_pair_read.py --tags PARITY_DAGOACC_s0 PARITY_DAGOACC_s1 OB-CTL_s0 --out <json>
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "otband", "read"))
import topk_read as TK  # noqa: E402  (sets TREE first on sys.path; asserts tanitad comes from it)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from topk_read import BL, BANDS, BAND_NAMES, D, M, K, free_gb, paired, single, sha256, tag_paths  # noqa: E402

ROWS = ("ade_m", "fde_m", "speed_mae_mps", "accel_mae_mps2", "along_mae_m", "cross_mae_m", "heading_mae_deg", "yaw_rate_mae_degps", "curvature_mae_1pm")
TOPK_FULL = "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/otband/read/out/TOPK_FULL.json"


def wait_free(gb, max_s):
    t0 = time.time()
    while free_gb() < gb:
        if time.time() - t0 > max_s:
            raise MemoryError(f"free physical memory stayed below {gb} GB for {max_s} s")
        time.sleep(30)


def run_arm(E, A, tag, smoke=0):
    ck_path, ddir, done = tag_paths(tag)
    got = sha256(ck_path)
    man = os.path.dirname(ck_path) + "/BANK_MANIFEST.json"
    if os.path.exists(man):
        assert json.load(open(man))["files"]["ckpt_last.pt"]["sha256"] == got, (tag, "checkpoint sha256 differs from its bank manifest")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**ck["cfg"])
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    model.load_state_dict(ck["model"])
    model.eval()
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
    out = {"traj": np.zeros((N, K, 30, 2)), "top1": np.zeros((N, 30, 2)), "gtpair": np.full((N, 30, 2), np.nan), "code_lat": np.zeros((N, K), np.int64),
           "code_lon": np.zeros((N, K), np.int64), "g": np.zeros((N, 30, 2)), "lat_cls": np.zeros(N, np.int64), "lon_cls": np.zeros(N, np.int64),
           "has": np.zeros(N, bool), "ci": np.zeros(N, np.int64), "ws": np.zeros(N, np.int64)}
    a0, b0 = M.N_GOAL, M.N_GOAL + M.N_LAT
    t0 = time.time()
    for s in range(0, N, 32):
        chunk = wins[s:s + 32]
        wait_free(7.5, 3600)
        st = E.make_states(store, tracks, J, chunk, A["nav"])
        items = [E.build_item(ds, proxy, x, 0, a_t, a_s, A["scene"], A["limit"], None) for x in st]
        full = D.collate(items)
        batch = {k: full[k] for k in E.MODEL_KEYS}
        lt = full["lab_t"]
        lat, lon, has = lt["lat_cls"], lt["lon_cls"], lt["has"]
        lab = has & (lat >= 0) & (lon >= 0)
        with torch.no_grad():
            r = model.forward_topk(batch, K)
            codes, _ = M.topk_codes(r["code"], K)
            c = r["code"].clone()                                         # GT-PAIR: T's own goals and constraints, the logged (lat, lon) pair
            c[:, a0:b0] = torch.nn.functional.one_hot(lat.clamp_min(0), M.N_LAT).to(c.dtype)
            c[:, b0:b0 + M.N_LON] = torch.nn.functional.one_hot(lon.clamp_min(0), M.N_LON).to(c.dtype)
            gp = model.forward_diag(batch, c)["traj30"]
        sl = slice(s, s + len(chunk))
        out["traj"][sl] = r["traj30"].float().numpy().astype(np.float64)
        out["top1"][sl] = r["traj30_top1"].float().numpy().astype(np.float64)
        g = gp.float().numpy().astype(np.float64)
        g[~lab.numpy()] = np.nan
        out["gtpair"][sl] = g
        out["code_lat"][sl] = torch.stack([x[:, a0:b0].argmax(1) for x in codes], 1).numpy()
        out["code_lon"][sl] = torch.stack([x[:, b0:b0 + M.N_LON].argmax(1) for x in codes], 1).numpy()
        out["lat_cls"][sl], out["lon_cls"][sl], out["has"][sl] = lat.numpy(), lon.numpy(), has.numpy()
        for i, (ci, tt, _, _) in enumerate(chunk):
            out["g"][s + i] = store.get(str(J["clip"][ci]))["gt"][tt]
            out["ci"][s + i], out["ws"][s + i] = ci, tt
        print(f"  {tag} [{s + len(chunk)}/{N}] {time.time() - t0:.0f}s free {free_gb():.1f} GB", flush=True)
    out["ckpt_sha256"] = got
    return out


def plans(o):
    lab = o["has"] & (o["lat_cls"] >= 0) & (o["lon_cls"] >= 0)
    match = (o["code_lat"] == o["lat_cls"][:, None]) & (o["code_lon"] == o["lon_cls"][:, None]) & lab[:, None]
    inK = match.any(1)
    k = match.argmax(1)
    sel = np.where(inK[:, None, None], o["traj"][np.arange(len(k)), k], o["top1"])
    return lab, match, inK, sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", type=int, default=0, help="N evenly spaced windows; dump controls skipped (a code check, never a result)")
    a = ap.parse_args()
    E = TK.load_eval_module()
    res = {"tool": "topk_pair_read.py", "card": "theory/topk/PREREAD_V8F_TOPK_PAIR.md", "K": K, "bands": {b: list(BANDS[b]) for b in BAND_NAMES},
           "estimator": "paired episode-cluster bootstrap, B=2000, seed 0, on labelled windows; answers 'another draw of episodes' only", "tree": TK.TREE, "arms": {}, "controls": {}}
    for tag in a.tags:
        wait_free(4.0 if a.smoke else 10.0, 6 * 3600)
        A = TK.parse_argv(f"{TK.T1}/argv_{tag}_m10_causal_heads.txt")
        print(f"arm {tag}", flush=True)
        o = run_arm(E, A, tag, a.smoke)
        _, ddir, _ = tag_paths(tag)
        if a.smoke:
            Z, idx, eid = None, None, o["ci"]
        else:
            Z = BL.load_dir(ddir)
            key_d = {(int(c), int(w)): i for i, (c, w) in enumerate(zip(Z["clip_index"], Z["ws"]))}
            key_m = [(int(c), int(w)) for c, w in zip(o["ci"], o["ws"])]
            assert set(key_d) == set(key_m) and len(key_d) == len(key_m), "window set differs from the dump"
            idx = np.array([key_d[x] for x in key_m])
            eid = Z["eid"][idx]
        lab, match, inK, sel = plans(o)
        n_lab = int(lab.sum())
        ctl = {"ckpt_sha256": o["ckpt_sha256"], "smoke": a.smoke,
               "top1_vs_dump_ol_max_abs_m": None if Z is None else float(np.abs(o["top1"] - Z["ol"][idx]).max()),
               "g_vs_dump_max_abs_m": None if Z is None else float(np.abs(o["g"] - Z["g"][idx]).max()),
               "pairsel_equals_top1_outside_K": bool(np.array_equal(sel[~inK], o["top1"][~inK])),
               "gtpair_vs_matching_kplan_max_abs_m": float(np.abs(o["gtpair"][inK] - sel[inK]).max()) if inK.any() else None}
        best = {}
        arm = {"n_windows": int(len(o["ci"])), "n_labelled": n_lab, "share_logged_pair_in_top3_of_labelled": float(inK[lab].mean()), "bands": {}}
        for bn in BAND_NAMES:
            a0, b0 = BANDS[bn]
            R = {nm: BL.band_rows(X, o["g"], a0, b0) for nm, X in (("top1", o["top1"]), ("pairsel", sel), ("gtpair", np.nan_to_num(o["gtpair"])))}
            per_k = np.stack([BL.band_rows(o["traj"][:, k], o["g"], a0, b0)["ade_m"] for k in range(K)], 1)
            best3 = np.nanmin(per_k, 1)
            m = lab
            row = {"ade_top1": single(R["top1"]["ade_m"][m], eid[m]), "ade_pairsel": single(R["pairsel"]["ade_m"][m], eid[m]),
                   "ade_gtpair": single(R["gtpair"]["ade_m"][m], eid[m]), "ade_best3": single(best3[m], eid[m]),
                   "pairsel_minus_top1": {x: paired(R["pairsel"][x][m], R["top1"][x][m], eid[m]) for x in ROWS},
                   "gtpair_minus_top1": {x: paired(R["gtpair"][x][m], R["top1"][x][m], eid[m]) for x in ROWS},
                   "best3_minus_top1": paired(best3[m], R["top1"]["ade_m"][m], eid[m]),
                   "all_windows_top1_mean": float(R["top1"]["ade_m"].mean()), "all_windows_best3_mean": float(best3.mean())}
            g_sel, g_best = row["pairsel_minus_top1"]["ade_m"]["delta"], row["best3_minus_top1"]["delta"]
            row["captured_share_S_estimated"] = float(g_sel / g_best) if g_best != 0 else None
            arm["bands"][bn] = row
            best[bn] = (row["all_windows_top1_mean"], row["all_windows_best3_mean"])
        if tag == "OB-CTL_s0" and not a.smoke and os.path.exists(TOPK_FULL):
            ref = json.load(open(TOPK_FULL))["arms"]["OB-CTL_s0"]["bands"]["tactical_2_6s"]
            ctl["vs_TOPK_FULL_tactical"] = {"top1_abs_diff": abs(best["tactical_2_6s"][0] - ref["ade_top1"]["mean"]),
                                            "best3_abs_diff": abs(best["tactical_2_6s"][1] - ref["ade_best_of_3"]["mean"])}
        res["arms"][tag] = arm
        res["controls"][tag] = ctl
        with open(a.out, "w", encoding="utf-8") as f:                        # written after every arm (bank incrementally)
            json.dump(res, f, indent=1)
        print(json.dumps({"tag": tag, "controls": ctl, "tactical_S": arm["bands"]["tactical_2_6s"]["captured_share_S_estimated"]}), flush=True)


if __name__ == "__main__":
    main()
