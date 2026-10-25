#!/usr/bin/env python3
"""Cross-TREE proof that an ABSENT ``hacc_w`` changes nothing: the same real checkpoint on the same real eval windows, forwarded by the code of c8ea1623 (BEFORE the post-map) and by the code
with it, must give BIT-identical outputs, and the checkpoint's state dict must load strictly into both with the same key set.

  PYTHONPATH=<tree>/stack;<tree>/taniteval  python verify_absent_identical.py run  --out X.npz [--fixtures <viz/work>]     (run once per tree)
  python verify_absent_identical.py compare A.npz B.npz                                                                    (bit-for-bit)

``run`` forwards (CPU, fp32) every window of the cached eval clip through ``forward_infer`` for both checkpoints (M2b_s0, M2b_s1) and for a synthetic tiny model, and stores every tensor
the forward returns. Edits nothing."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np  # noqa: E402
import torch  # noqa: E402

DEFAULT_FIX = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/viz/work"


def real_batch(D, FIX: Path, cfg):
    inp = FIX / "inputs"
    J = D.load_join(inp / "join_eval141.npz")
    tracks = D.load_native_tracks(inp / "native_tracks.pt")
    store = D.ClipStore(str(FIX / "cache"), "eval", lru=8, grid="pool4x10", kin_mode=cfg.kin_mode, tracks=tracks)
    wins = D.eval_grid_windows(str(inp / "eval_grid_windows.json"), J)
    cached = {p.stem for p in (FIX / "cache" / "eval" / "clips").glob("*.pt")}
    clips = J["clip"].tolist()
    idx = [i for i, w in enumerate(wins) if clips[w[0]] in cached]
    ds = D.V8FMiniWindows(store, J, wins, hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s, window_start=cfg.window_start)
    items = []
    for i in idx:
        ci, t, _, _ = wins[i]
        a_t = i % 3
        items.append(ds.item(ci, t, a_t, a_t + (i // 3) % 5))
    return D.collate(items), idx


def cmd_run(a) -> int:
    from tanitad.data import v8f_mini_data as D
    from tanitad.models import v8f_mini as M
    torch.set_num_threads(2)
    FIX = Path(a.fixtures)
    rec: dict = {}
    meta = {"has_hacc_api": hasattr(M.V8FMini, "set_hacc_weights"), "module_file": M.__file__}
    for seed in ("M2b_s0", "M2b_s1"):
        ck = torch.load(FIX / "ckpt" / f"{seed}.pt", map_location="cpu", weights_only=False)
        cfg = M.V8FMiniConfig(**ck["cfg"])
        model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
        res = model.load_state_dict(ck["model"], strict=True)                   # STRICT: a checkpoint written before the post-map must load unchanged
        meta[f"{seed}_strict_load"] = {"missing": list(res.missing_keys), "unexpected": list(res.unexpected_keys)}
        meta[f"{seed}_state_dict_keys_equal_checkpoint"] = set(model.state_dict()) == set(ck["model"])
        meta[f"{seed}_n_state_keys"] = len(model.state_dict())
        model.eval()
        b, idx = real_batch(D, FIX, cfg)
        with torch.no_grad():
            out = model.forward_infer(b)
        meta[f"{seed}_n_windows"] = len(idx)
        for k, v in out.items():
            if torch.is_tensor(v):
                rec[f"{seed}/{k}"] = v.detach().cpu().numpy()
    # a synthetic tiny model with a live O head (random weights, fixed seeds)
    import tempfile
    root = Path(tempfile.mkdtemp(prefix="vai"))
    w = D.make_synthetic_world(root, n_train=6, n_eval=2, T=60, seed=0)
    J = D.load_join(root / "join_train.npz")
    store = D.ClipStore(root, "train", lru=8)
    wins = D.train_windows(store, J, w["names"]["train"])
    ds = D.V8FMiniWindows(store, J, wins, hmax=10)
    idx = np.random.default_rng(0).choice(len(wins), 8, replace=False)
    b = D.collate([ds.item(wins[i][0], wins[i][1], 1, 3) for i in idx])
    cfg = M.V8FMiniConfig(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6)
    mu, sd = M.cons_stats(J)
    torch.manual_seed(0)                                       # the tiny model's random init must be the SAME in both trees
    model = M.V8FMini(cfg, M.P10Prior.synthetic(0), mu, sd).eval()
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        model.head_o[2].weight.copy_(torch.randn(model.head_o[2].weight.shape, generator=g) * 0.3)
        out = model.forward_infer(b)
    for k, v in out.items():
        if torch.is_tensor(v):
            rec[f"syn/{k}"] = v.detach().cpu().numpy()
    meta["syn_state_dict_keys"] = sorted(model.state_dict())
    np.savez(a.out, **rec)
    Path(a.out).with_suffix(".json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    print(f"[run] {a.out}: {len(rec)} tensors; has_hacc_api={meta['has_hacc_api']}; strict loads: " + str({k: v for k, v in meta.items() if k.endswith('strict_load')}))
    return 0


def cmd_compare(a) -> int:
    x, y = np.load(a.a), np.load(a.b)
    ma, mb = json.loads(Path(a.a).with_suffix(".json").read_text()), json.loads(Path(a.b).with_suffix(".json").read_text())
    ok = True
    if set(x.files) != set(y.files):
        print("KEY SETS DIFFER:", sorted(set(x.files) ^ set(y.files)))
        ok = False
    n_bit = 0
    for k in sorted(set(x.files) & set(y.files)):
        same = x[k].dtype == y[k].dtype and x[k].shape == y[k].shape and x[k].tobytes() == y[k].tobytes()
        n_bit += int(same)
        if not same:
            ok = False
            print(f"  DIFFERS: {k}  max|d| = {np.abs(x[k].astype(np.float64) - y[k].astype(np.float64)).max() if x[k].shape == y[k].shape else 'shape'}")
    sd_same = ma["syn_state_dict_keys"] == mb["syn_state_dict_keys"]
    print(f"[compare] {n_bit}/{len(set(x.files) & set(y.files))} tensors bit-identical; synthetic state-dict key list identical: {sd_same}")
    print(f"[compare] A has_hacc_api={ma['has_hacc_api']}  B has_hacc_api={mb['has_hacc_api']}")
    for s in ("M2b_s0", "M2b_s1"):
        print(f"[compare] {s}: strict load A {ma[s + '_strict_load']}  B {mb[s + '_strict_load']}  n_windows A/B {ma[s + '_n_windows']}/{mb[s + '_n_windows']}  "
              f"state-dict keys == checkpoint's: A {ma[s + '_state_dict_keys_equal_checkpoint']} B {mb[s + '_state_dict_keys_equal_checkpoint']}  n_keys A/B {ma[s + '_n_state_keys']}/{mb[s + '_n_state_keys']}")
        ok &= (not ma[s + "_strict_load"]["missing"] and not mb[s + "_strict_load"]["missing"] and not mb[s + "_strict_load"]["unexpected"]
               and mb[s + "_state_dict_keys_equal_checkpoint"] and ma[s + "_n_state_keys"] == mb[s + "_n_state_keys"])
    ok &= sd_same
    print("[compare] " + ("ABSENT hacc_w IS BIT-IDENTICAL TO c8ea1623 (real checkpoints, real eval windows, synthetic model)" if ok else "NOT IDENTICAL"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--fixtures", default=DEFAULT_FIX)
    c = sp.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args()
    return cmd_run(a) if a.cmd == "run" else cmd_compare(a)


if __name__ == "__main__":
    sys.exit(main())
