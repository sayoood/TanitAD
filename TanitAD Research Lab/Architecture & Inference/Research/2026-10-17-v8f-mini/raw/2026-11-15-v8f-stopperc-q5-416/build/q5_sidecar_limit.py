"""q5_sidecar_limit.py -- the input of Amendment 7 control C3c (MM, 2026-11-15).  CPU, no model pass, no fit.

C3c is a mutation: E's limit block replaced by the LOGGED sidecar `limit` (the ego's own realised speed ceiling over [t0+2 s, +6 s],
build_v8_speed_max_labels.py:16-19).  The mutated E is never fitted; the check must see that E changes on >= 1 window.
No extractor wrote the logged limit (they assert the causal bucket, Amendment 1), so this runs s0p_extract_eval.py's batch build UNCHANGED
except for one argument: build_item(..., limit="logged") instead of "causal" (v8f_t1_eval.build_item leaves ds.item's logged limit when
limit is neither "off" nor "causal*").  The causal block is built in the same pass beside it, so the output also proves its own alignment.

Output (--out X.npy): [N_has, 4] float32 logged limit, rows in s0p_eval.npz order (the `has` windows of the eval grid, grid order).
Beside it X.meta.json and X.align.npz (clip, ws, ci, causal_limit, logged_limit_ok) for the read's alignment check against C0's eval npz.
Usage: python q5_sidecar_limit.py --out F:/TanitAD-scratch/q5_416/run/c0/sidecar_limit_eval.npy [--smoke N]
"""
import argparse
import hashlib
import json
import os
import sys
import time

S0P = "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/stopperc/s0p"
THEORY = os.path.abspath(os.path.join(S0P, "..", ".."))
sys.path.insert(0, S0P)
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
    if os.path.exists(a.out):
        sys.exit(f"REFUSED: {a.out} exists (outputs are never overwritten)")
    import stopdec_arm as SA
    SA.install()
    sys.path.insert(0, os.path.join(THEORY, "stopdec"))
    sys.path.insert(0, os.path.join(THEORY, "speedsrc"))
    import speedsrc_read as SS
    TK, M, D = SS.TK, SS.M, SS.D
    import tanitad
    import torch
    assert os.path.abspath(tanitad.__file__).replace("\\", "/").startswith(TREE), tanitad.__file__
    E = TK.load_eval_module()
    assert E.CAUSAL_STEPS == L.CAUSAL_STEPS
    A, ck_path, _ = SS.tag_paths(TAG)
    assert A["limit"] == "causal" and A["scene"] == "logged" and A["nav"] == "deadreckon" and A["schedule"] == "periodic"
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**ck["cfg"])
    tracks = D.load_native_tracks(A["tracks"])
    J = D.load_join(A["eval-join"])
    store = D.ClipStore(A["eval-cache"], "eval", lru=32, grid=ck.get("grid", "pool4x10"), kin_mode="yaw", tracks=tracks)
    wins = D.eval_grid_windows(A["eval-grid"], J)
    if a.smoke:
        wins = [wins[i] for i in np.linspace(0, len(wins) - 1, a.smoke).round().astype(int)]
    proxy = E._OneClipStore()
    ds = D.V8FMiniWindows(proxy, J, [], hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s, window_start=cfg.window_start)
    a_t, a_s = E.ages_for(0, A["schedule"])
    assert (a_t, a_s) == (0, 0)
    N = len(wins)
    out = {k: [] for k in ("logged", "logged_ok", "causal", "has", "ci", "ws", "clip")}
    t0 = time.time()
    for s in range(0, N, 32):
        chunk = wins[s:s + 32]
        st = E.make_states(store, tracks, J, chunk, A["nav"])
        items_c = [E.build_item(ds, proxy, x, 0, a_t, a_s, A["scene"], "causal", None) for x in st]
        st = E.make_states(store, tracks, J, chunk, A["nav"])
        items_l = [E.build_item(ds, proxy, x, 0, a_t, a_s, A["scene"], "logged", None) for x in st]
        fc, fl = D.collate(items_c), D.collate(items_l)
        out["has"].append(fc["lab_t"]["has"].bool().numpy())
        out["causal"].append(fc["limit"].numpy().astype(np.float32))
        out["logged"].append(fl["limit"].numpy().astype(np.float32))
        out["logged_ok"].append(np.asarray(fl["limit_ok"]).astype(bool).reshape(-1))
        for (ci, t, _, _) in chunk:
            out["ci"].append(ci); out["ws"].append(t); out["clip"].append(str(J["clip"][ci]))
        print(f"  [{s + len(chunk)}/{N}] {time.time() - t0:.0f}s", flush=True)
    for k in ("has", "causal", "logged", "logged_ok"):
        out[k] = np.concatenate(out[k])
    ci, ws, clips = np.asarray(out["ci"], np.int64), np.asarray(out["ws"], np.int64), np.asarray(out["clip"])
    h = out["has"]
    meta = {"smoke": bool(a.smoke), "n_windows_grid": int(N), "n_has": int(h.sum()), "tag_for_batch_build": TAG,
            "route": "s0p_extract_eval.py batch build, build_item(limit='logged'); causal built beside in the same pass"}
    if not a.smoke:
        z = np.load(os.path.join(STOPDEC_OUT, "stopdec_BAL-NULL_s0.npz"), allow_pickle=False)
        assert np.array_equal(z["clip_index"], ci) and np.array_equal(z["ws"], ws), "window order differs from STOPDEC"
        assert np.array_equal(z["has"], h), "has differs from STOPDEC"
        meta["window_order_equals_stopdec_BAL-NULL_s0"] = True
    lg, ca, lok = out["logged"][h], out["causal"][h], out["logged_ok"][h]
    meta["n_rows"] = int(lg.shape[0])
    meta["n_rows_logged_ne_causal"] = int((np.abs(lg - ca).max(axis=1) > 0).sum())
    meta["n_rows_logged_ok_false"] = int((~lok).sum())
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.save(a.out, lg)
    np.savez(a.out[:-4] + ".align.npz", clip=clips[h], ws=ws[h], ci=ci[h], causal_limit=ca, logged_limit_ok=lok)
    with open(a.out, "rb") as fh:
        meta["sha256"] = hashlib.sha256(fh.read()).hexdigest()
    with open(a.out[:-4] + ".meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    print(json.dumps(meta, indent=1), flush=True)


if __name__ == "__main__":
    main()
