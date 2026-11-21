"""cons_orc_read.py -- PREREAD_V8F_CONSORC (theory/consorc/PREREAD_V8F_CONSORC.md).  CPU, zero training.

The pair read's T1-kin first-tick loop (theory/topk/topk_pair_read.py, same batch build as theory/otband/read/topk_read.py) with oracle decodes:
  GT-ALL (V8FMini._code_gt), OWN-DEC / OWN-CONS / OWN-<kind> (leave-one-out on the GT background), GT-SPEED-ONLY, IDENT and OWN-ALL (controls),
  plus T's present-slot squared error per constraint kind.  Paired episode-cluster bootstrap (B = 2000, seed 0) on windows with lab_t.has.
Usage: cons_orc_read.py --tags PARITY_DAGOACC_s0 ... --out <json> [--smoke N]
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
from topk_read import BL, BANDS, BAND_NAMES, D, M, K, free_gb, paired, single, sha256  # noqa: E402

ROWS = ("ade_m", "fde_m", "speed_mae_mps", "accel_mae_mps2", "along_mae_m", "cross_mae_m", "heading_mae_deg", "yaw_rate_mae_degps", "curvature_mae_1pm")
KINDS = ("speed", "angle", "tte", "timegap", "geom", "leadgap")
DEC = list(M.code_group_index("decisions"))
CONS = list(M.code_group_index("constraints"))
assert sorted(DEC + CONS) == list(range(M.CODE_DIM)) and not set(DEC) & set(CONS), "decisions and constraints must partition the code"
C0 = M.N_GOAL + M.N_LAT + M.N_LON
PAIR_JSON = os.path.join(os.path.dirname(HERE), "topk", "out", "TOPK_PAIR.json")
DECODES = ("ident", "own_all", "gtall", "own_dec", "own_cons", *[f"own_{k}" for k in KINDS], "gt_speed_only")


def wait_free(gb, max_s):
    t0 = time.time()
    while free_gb() < gb:
        if time.time() - t0 > max_s:
            raise MemoryError(f"free physical memory stayed below {gb} GB for {max_s} s")
        time.sleep(30)


def tag_paths(tag):
    A = TK.parse_argv(f"{TK.T1}/argv_{tag}_m10_causal_heads.txt")
    return A, A["ckpt"], f"{TK.T1}/dump_{tag}_m10_causal_heads"


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
    out = {"top1": np.zeros((N, 30, 2)), "g": np.zeros((N, 30, 2)), "has": np.zeros(N, bool), "ci": np.zeros(N, np.int64), "ws": np.zeros(N, np.int64),
           "terr": np.full((N, len(KINDS)), np.nan)}
    for d in DECODES:
        out[d] = np.zeros((N, 30, 2))
    t0 = time.time()
    for s in range(0, N, 32):
        chunk = wins[s:s + 32]
        wait_free(7.5, 3600)
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
            gt = torch.where(has[:, None], gt, own)                       # no label: the code stays T's own (those windows are not read)
            codes = {"ident": own}
            c = gt.clone(); c[:, :] = own; codes["own_all"] = c
            codes["gtall"] = gt
            c = gt.clone(); c[:, DEC] = own[:, DEC]; codes["own_dec"] = c
            c = gt.clone(); c[:, CONS] = own[:, CONS]; codes["own_cons"] = c
            for k in KINDS:
                idx = list(M.CODE_KINDS[k])
                c = gt.clone(); c[:, idx] = own[:, idx]; codes[f"own_{k}"] = c
            sp = list(M.CODE_KINDS["speed"])
            c = own.clone(); c[:, sp] = gt[:, sp]; codes["gt_speed_only"] = c
            trajs = {d: model.forward_diag(batch, codes[d])["traj30"] for d in DECODES}
            present = ~torch.isnan(lt["cons"])
            e2 = (own[:, C0:] - gt[:, C0:]) ** 2
        sl = slice(s, s + len(chunk))
        out["top1"][sl] = r["traj30_top1"].float().numpy().astype(np.float64)
        for d in DECODES:
            out[d][sl] = trajs[d].float().numpy().astype(np.float64)
        for j, k in enumerate(KINDS):
            cols = [i - C0 for i in M.CODE_KINDS[k]]
            pm = present[:, cols].float() * has[:, None].float()
            n = pm.sum(1)
            v = (e2[:, cols] * pm).sum(1) / n.clamp_min(1.0)
            out["terr"][sl, j] = torch.where(n > 0, v, torch.full_like(v, float("nan"))).numpy()
        out["has"][sl] = has.numpy()
        for i, (ci, tt, _, _) in enumerate(chunk):
            out["g"][s + i] = store.get(str(J["clip"][ci]))["gt"][tt]
            out["ci"][s + i], out["ws"][s + i] = ci, tt
        print(f"  {tag} [{s + len(chunk)}/{N}] {time.time() - t0:.0f}s free {free_gb():.1f} GB", flush=True)
    out["ckpt_sha256"], out["manifest_ok"] = got, man_ok
    return out


def read_arm(o, eid):
    m = o["has"]
    arm = {"n_windows": int(len(o["ci"])), "n_has": int(m.sum()), "bands": {}}
    for bn in BAND_NAMES:
        a0, b0 = BANDS[bn]
        R = {d: BL.band_rows(o[d], o["g"], a0, b0) for d in ("top1", *DECODES)}
        row = {f"ade_{d}": single(R[d]["ade_m"][m], eid[m]) for d in ("top1", "gtall", "own_dec", "own_cons", *[f"own_{k}" for k in KINDS], "gt_speed_only")}
        row["H_gtall_minus_top1"] = {x: paired(R["gtall"][x][m], R["top1"][x][m], eid[m]) for x in ROWS}
        for d in ("own_dec", "own_cons", *[f"own_{k}" for k in KINDS]):
            row[f"cost_{d}"] = {x: paired(R[d][x][m], R["gtall"][x][m], eid[m]) for x in ROWS}
        row["gt_speed_only_minus_top1"] = {x: paired(R["gt_speed_only"][x][m], R["top1"][x][m], eid[m]) for x in ROWS}
        H = row["H_gtall_minus_top1"]["ade_m"]["delta"]
        row["share_estimated"] = {d: (float(row[f"cost_{d}"]["ade_m"]["delta"] / abs(H)) if H != 0 else None) for d in ("own_dec", "own_cons", *[f"own_{k}" for k in KINDS])}
        arm["bands"][bn] = row
    te = {}
    for j, k in enumerate(KINDS):
        v = o["terr"][:, j]
        ok = m & ~np.isnan(v)
        te[k] = {"n": int(ok.sum()), "mse_mean": float(v[ok].mean()) if ok.any() else None}
    arm["t_present_slot_mse"] = te
    return arm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", type=int, default=0, help="N evenly spaced windows; dump controls skipped (a code check, never a result)")
    a = ap.parse_args()
    E = TK.load_eval_module()
    res = {"tool": "cons_orc_read.py", "card": "theory/consorc/PREREAD_V8F_CONSORC.md", "bands": {b: list(BANDS[b]) for b in BAND_NAMES}, "kinds": list(KINDS),
           "estimator": "paired episode-cluster bootstrap, B=2000, seed 0, on windows with lab_t.has; answers 'another draw of episodes' only", "tree": TK.TREE,
           "arms": {}, "controls": {}, "cross": {}}
    keep = {}
    for tag in a.tags:
        wait_free(10.0, 6 * 3600)
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
               "ident_vs_top1_max_abs_m": float(np.abs(o["ident"] - o["top1"]).max()),
               "own_all_vs_ident_max_abs_m": float(np.abs(o["own_all"] - o["ident"]).max())}
        arm = read_arm(o, eid)
        if tag == "PARITY_DAGOACC_s0" and not a.smoke and os.path.exists(PAIR_JSON):
            pj = json.load(open(PAIR_JSON))["arms"].get(tag)
            if pj:
                ctl["vs_pair_read_top1_all_windows_tactical_abs_diff"] = abs(float(BL.band_rows(o["top1"], o["g"], *BANDS["tactical_2_6s"])["ade_m"].mean())
                                                                           - pj["bands"]["tactical_2_6s"]["all_windows_top1_mean"])
        res["arms"][tag] = arm
        res["controls"][tag] = ctl
        keep[tag] = {"terr": o["terr"], "has": o["has"], "ci": o["ci"], "ws": o["ws"], "eid": eid}
        for seed in ("s0", "s1"):                                           # T accuracy, DAG-OACC - M2b, paired on the same windows
            ta, tb = f"PARITY_DAGOACC_{seed}", f"INIT_M2b_{seed}"
            if ta in keep and tb in keep and f"terr_{seed}" not in res["cross"]:
                A_, B_ = keep[ta], keep[tb]
                assert np.array_equal(A_["ci"], B_["ci"]) and np.array_equal(A_["ws"], B_["ws"]), "arms read different windows"
                cr = {}
                for j, k in enumerate(KINDS):
                    va, vb = A_["terr"][:, j], B_["terr"][:, j]
                    ok = A_["has"] & B_["has"] & ~np.isnan(va) & ~np.isnan(vb)
                    cr[k] = paired(va[ok], vb[ok], A_["eid"][ok]) if ok.sum() > 1 else None
                res["cross"][f"terr_{seed}"] = cr
        with open(a.out, "w", encoding="utf-8") as f:                        # written after every arm (bank incrementally)
            json.dump(res, f, indent=1)
        tb_ = arm["bands"]["tactical_2_6s"]
        print(json.dumps({"tag": tag, "controls": ctl, "tactical_share": tb_["share_estimated"], "H": tb_["H_gtall_minus_top1"]["ade_m"]["delta"]}), flush=True)


if __name__ == "__main__":
    main()
