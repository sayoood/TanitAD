"""s0p_extract_eval.py -- S0-P step 1: eval-side extraction (card section 8 S0-P; AMENDMENT_1_S0.md).  CPU, no model pass, no training.

For every STOPCUE eval window (the `has` windows of the eval-141 first-tick grid; STOPCUE read 2,365 windows / 131 positives / 139 episodes, RE-COUNTED here) write E (exactly the batch
fed to T at the first tick: p10f last row, nav last row, limit, limit_ok -- limit/limit_ok are the CAUSAL bucket, AMENDMENT 1), the NOW frame's pool4x10 / pool8x20 / mean, the label, clip id, window start, episode id.
The batch build is the STOPDEC / STOPORC one (stopdec_arm.install, speedsrc_read.tag_paths on BAL-NULL_s0's argv, E.make_states, E.build_item, D.collate); nothing is copied from run_arm.

Usage: python s0p_extract_eval.py --out F:/TanitAD-scratch/stopperc_s0p/s0p_eval.npz [--smoke N]
Controls (written into the npz meta and printed):
  C1  E[:, 0] (= p10f last row, speed) equals the dump's first-tick v0 (ticks_BAL-NULL_s0 ... tick 0): max |diff|.  The dumps hold NO other E field (v0 is the only one): C2 covers the rest.
  C2  E recomputed along a second route (ClipStore.get -> p10f[t], J["nav"][row], E.causal_bucket(v, t, 20)): max |diff| per block.
  C3  the batch's NOW-frame pool4x10 (fp16) equals the stored fp8 bytes of the clip file at row t: max |diff|.
  C4  has / stop_hit / (ci, ws) equal STOPDEC's npz of BAL-NULL_s0 AND BAL-NULL_s1 on the same windows; eid taken from it.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
THEORY = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(THEORY, "stopbal", "read"))
sys.path.insert(0, os.path.join(THEORY, "stopbal", "oracle"))
import numpy as np  # noqa: E402
import s0p_lib as L  # noqa: E402

TREE = os.environ.get("STOPBAL_TREE", "F:/TanitAD-scratch/stopbal_read/tree")
TAG = "BAL-NULL_s0"
STOPDEC_OUT = os.path.join(THEORY, "stopbal", "read", "out", "stopdec")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    import stopdec_arm as SA
    SA.install()
    sys.path.insert(0, os.path.join(THEORY, "stopdec"))
    sys.path.insert(0, os.path.join(THEORY, "speedsrc"))
    import speedsrc_read as SS
    import torch
    TK, M, D = SS.TK, SS.M, SS.D
    import tanitad
    assert os.path.abspath(tanitad.__file__).replace("\\", "/").startswith(TREE), tanitad.__file__
    E = TK.load_eval_module()
    assert E.CAUSAL_STEPS == L.CAUSAL_STEPS
    A, ck_path, ddir = SS.tag_paths(TAG)
    assert A["limit"] == "causal" and A["scene"] == "logged" and A["nav"] == "deadreckon" and A["schedule"] == "periodic", {k: A[k] for k in ("limit", "scene", "nav", "schedule")}
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**ck["cfg"])
    tracks = D.load_native_tracks(A["tracks"])
    J = D.load_join(A["eval-join"])
    store = D.ClipStore(A["eval-cache"], "eval", lru=32, grid=ck.get("grid", "pool4x10"), kin_mode="yaw", tracks=tracks)
    assert store.grid == "pool4x10"
    wins = D.eval_grid_windows(A["eval-grid"], J)
    if a.smoke:
        wins = [wins[i] for i in np.linspace(0, len(wins) - 1, a.smoke).round().astype(int)]
    proxy = E._OneClipStore()
    ds = D.V8FMiniWindows(proxy, J, [], hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s, window_start=cfg.window_start)
    a_t, a_s = E.ages_for(0, A["schedule"])
    assert (a_t, a_s) == (0, 0), "first tick must be age 0"
    N = len(wins)
    cdir = os.path.join(A["eval-cache"], "eval", "clips")
    out = {k: [] for k in ("E", "v40", "v160", "mean", "y", "has", "clip", "ws", "ci", "stop_hit")}
    c3, c2 = [], {"p10f": 0.0, "nav": 0.0, "limit": 0.0, "limit_ok": 0.0}
    t0 = time.time()
    for s in range(0, N, 32):
        chunk = wins[s:s + 32]
        st = E.make_states(store, tracks, J, chunk, A["nav"])
        items = [E.build_item(ds, proxy, x, 0, a_t, a_s, A["scene"], A["limit"], None) for x in st]
        full = D.collate(items)
        batch = {k: full[k] for k in E.MODEL_KEYS}
        lt = full["lab_t"]
        has = lt["has"].bool().numpy()
        hit = ((lt["goal_y"] > 0.5) & (lt["goal_w"] > 0.5)).numpy() & has[:, None]
        Eb = L.e_from_item_arrays(batch["p10f"][:, -1].numpy(), batch["nav"][:, -1].numpy(), batch["limit"].numpy(), batch["limit_ok"].numpy())
        now40 = batch["feats"][:, -1].numpy()                                   # fp16 [B, 40, 1024]
        for i, (ci, t, _, _) in enumerate(chunk):
            clip = str(J["clip"][ci])
            o = torch.load(os.path.join(cdir, clip + ".pt"), map_location="cpu", weights_only=True, mmap=True)
            r40 = o["pool4x10"][t].view(torch.uint8).numpy().copy()
            r160 = o["pool8x20"][t].view(torch.uint8).numpy().copy()
            rm = o["mean"][t].numpy().astype(np.float16).copy()
            c3.append(float(np.abs(L.decode_fp8(r40).astype(np.float64) - now40[i].astype(np.float64)).max()))
            d = store.get(clip)                                                  # second route for E (C2)
            row = int(J["off"][ci]) + t
            p10 = np.nan_to_num(d["p10f"][t], nan=0.0)
            lim, lok = E.causal_bucket(d["v"], t, E.CAUSAL_STEPS)
            c2["p10f"] = max(c2["p10f"], float(np.abs(Eb[i, :8] - p10.astype(np.float32)).max()))
            c2["nav"] = max(c2["nav"], float(np.abs(Eb[i, 8:13] - np.asarray(J["nav"][row], np.float32)).max()))
            c2["limit"] = max(c2["limit"], float(np.abs(Eb[i, 13:17] - lim).max()))
            c2["limit_ok"] = max(c2["limit_ok"], float(abs(Eb[i, 17] - float(lok))))
            del o
            for k, v in (("v40", r40), ("v160", r160), ("mean", rm), ("clip", clip), ("ws", t), ("ci", ci)):
                out[k].append(v)
        out["E"].append(Eb)
        out["has"].append(has)
        out["y"].append(hit[:, L.GOAL_STOP_BIT])
        out["stop_hit"].append(hit[:, L.GOAL_STOP_BIT])
        print(f"  [{s + len(chunk)}/{N}] {time.time() - t0:.0f}s", flush=True)
    for k in ("E", "has", "y", "stop_hit"):
        out[k] = np.concatenate(out[k])
    for k in ("v40", "v160", "mean"):
        out[k] = np.stack(out[k])
    ws, ci = np.asarray(out["ws"], np.int64), np.asarray(out["ci"], np.int64)
    clips = np.asarray(out["clip"])
    # C4 against STOPDEC's npz (both BAL-NULL seeds), and the episode id
    ctl = {"smoke": bool(a.smoke), "n_windows_grid": int(N), "n_has": int(out["has"].sum()), "n_pos": int((out["y"] & out["has"]).sum())}
    if not a.smoke:
        zs = [np.load(os.path.join(STOPDEC_OUT, f"stopdec_BAL-NULL_s{k}.npz"), allow_pickle=False) for k in (0, 1)]
        for z in zs:
            assert np.array_equal(z["clip_index"], ci) and np.array_equal(z["ws"], ws), "window order differs from STOPDEC"
            assert np.array_equal(z["has"], out["has"]) and np.array_equal(z["stop_hit"], out["stop_hit"]), "has / stop_hit differ from STOPDEC"
        # eid: the STOPDEC npz carries eid (per window)
        eid = zs[0]["eid"].astype(np.int64)
        ctl["c4_windows_has_stophit_equal_stopdec_s0_s1"] = True
        # C1 against the dump's first-tick v0
        tk = np.load(os.path.join(os.path.dirname(os.path.abspath(ddir)), "ticks", f"ticks_{TAG}_m10_causal_heads.npz"), allow_pickle=True)
        m0 = tk["tick"] == 0
        key = {(int(c), int(t)): float(v) for c, t, v in zip(tk["ci"][m0], tk["t0"][m0], tk["v0"][m0])}
        diffs = [abs(float(out["E"][i, 0]) - key[(int(ci[i]), int(ws[i]))]) for i in range(N) if (int(ci[i]), int(ws[i])) in key]
        ctl["c1_E_speed_vs_dump_v0"] = {"n_matched": len(diffs), "n_windows": int(N), "max_abs_diff": float(max(diffs)) if diffs else None}
    else:
        eid = ci.copy()
    ctl["c2_second_route_max_abs"] = c2
    ctl["c3_pool4x10_now_vs_stored_fp8_max_abs"] = float(max(c3))
    h = out["has"]
    ctl["causal_limit_ok_false_fraction_over_has_windows"] = float((out["E"][h, 17] == 0).mean())
    ctl["n_episodes_has"] = int(len(np.unique(eid[h])))
    ctl["n_clips_has"] = int(len(np.unique(clips[h])))
    ctl["E_layout"] = "p10f_last(8) nav_last(5) limit_causal(4) limit_ok_causal(1)"
    ctl["amendment"] = "AMENDMENT_1_S0.md: limit = causal bucket both sides"
    ctl["ckpt_cfg_for_batch_build"] = {"hist": cfg.hist, "hmax": cfg.hmax, "max_age_t": cfg.max_age_t}
    print(json.dumps(ctl, indent=1), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    sha = L.save_npz(a.out, out["E"][h], out["v40"][h], out["v160"][h], out["mean"][h], out["y"][h], clips[h], ws[h], eid[h], ctl, ci=ci[h])
    with open(a.out + ".sha256", "w", encoding="utf-8") as f:
        f.write(f"{sha}  {os.path.basename(a.out)}\n")
    print("banked", a.out, sha, flush=True)


if __name__ == "__main__":
    main()
