"""MEASURE the v9 join through the TRAINER's own path (refcv8 WP-B): build the eval139 dataset exactly as train() does
(``build_v2_providers`` -> ``V3Dataset(window, max_horizon=20, channels)`` -> ``enable_clip_clock(sidecar)``), then
``enable_r8_v9`` -- which calls the reader's refusing ``row_for_now`` on EVERY window -- and draw items through
``__getitem__``. An independent re-measurement of WP-A's INHERITED V11 ("23,772 / 23,772 eval139, now_s to 7e-15 s").

CPU only. Writes ``raw/v9_join_eval139.json`` (sid / sha12-free: counts and the clock residual only).

Run (dev box, from the package):  PYTHONPATH=<overlay>/stack python code/v9_join_measure.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = "D:/refcv6_eval_kit/data/refcv6-b1-416x1024-eval139"
SIDECAR = "D:/refcv6_eval_kit/data/refcv6_clip_clock_sidecar.jsonl"
REL = "D:/Projects/TanitAD-artifacts/v9labels/v9_labels_eval139.npz"


def main() -> None:
    import importlib
    import tanitad
    if "G:" in str(Path(tanitad.__file__).resolve()).upper()[:3]:
        raise SystemExit(f"tanitad imported from G: ({tanitad.__file__})")
    sys.path.insert(0, str(Path(tanitad.__file__).resolve().parents[1] / "scripts"))
    T = importlib.import_module("refc_v3_train")
    from tanitad.data.v2_dataset import build_v2_providers
    from tanitad.train import refcv8_train as RT
    t0 = time.time()
    cfg = T.v3.refc_v3_sized_config("small", hier=True)
    eps = build_v2_providers([CACHE], lru_size=2)
    ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20, channels=cfg.core.encoder.in_channels)
    clk = ds.enable_clip_clock(SIDECAR)
    j = RT.load_v9_join(REL, expect_md5=RT.V9_RELEASE_MD5["eval139"])
    t1 = time.time()
    c = ds.enable_r8_v9(j)
    t2 = time.time()
    # the clock residual over every joined window (the reader refused > 1e-6 s; here: the actual max)
    w = int(ds.window)
    res = np.empty(len(ds.index))
    for i, (e_i, t) in enumerate(ds.index):
        res[i] = abs(float(j.rel.rows["now_s"][int(c["rows"][i])]) - ds._now_s(ds.episodes[e_i], t))
    # items through __getitem__ on a fixed sample (seeded), the keys the losses read
    rng = np.random.default_rng(20261004)
    pick = rng.choice(len(ds.index), size=min(64, len(ds.index)), replace=False)
    ds.r8_nav_from_v9 = True
    n_turn_tok = n_rc_valid = n_known = n_lat_exact = n_lat_partial = 0
    for i in pick:
        it = ds[int(i)]
        n_turn_tok += int(int(it["nav_cmd"]) in (1, 2))
        n_rc_valid += int(bool(it["r8_rc_valid"]))
        n_known += int(bool(it["r8_nav_known"]))
        n_lat_exact += int(int(it["lat_v7"]) >= 0)
        n_lat_partial += int(int(it["lat_v7"]) < 0 and bool(it["lat_allowed_v7"].any()))
    out = {"what": "v9 eval139 joined through V3Dataset.enable_r8_v9 (the trainer's path), MEASURED",
           "cache": "refcv6-b1-416x1024-eval139 (D:/refcv6_eval_kit)", "release_md5": j.manifest["md5"],
           "window": w, "max_horizon": 20, "clip_clock": {k: v for k, v in clk.items() if not isinstance(v, (list, dict))},
           "n_episodes": len(eps), "n_windows": c["n_windows"], "n_joined": c["n_joined"], "n_clips": c["n_clips"],
           "clock_residual_max_s": float(res.max()), "clock_residual_median_s": float(np.median(res)),
           "census_violations": j.manifest["census"]["violations"],
           "item_sample": {"n": int(len(pick)), "turn_token": n_turn_tok, "rc_valid": n_rc_valid,
                           "nav_args_known": n_known, "lat_exact": n_lat_exact, "lat_partial": n_lat_partial},
           "wall_s": {"build": round(t1 - t0, 1), "join_every_window": round(t2 - t1, 1)}}
    p = HERE.parent / "raw" / "v9_join_eval139.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
