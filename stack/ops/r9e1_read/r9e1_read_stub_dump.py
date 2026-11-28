#!/usr/bin/env python3
"""r9e1_read_stub_dump.py -- DRY-RUN STAND-IN for ``stack/scripts/r9_e1_dump.py`` (the only GPU step of the E1 read).

``r9e1_read.sh`` with ``R9E1_DRY=1`` and ``R9E1_DUMP_SCRIPT=<this file>`` runs it with EXACTLY the argv it would give the real
dump; the parser below accepts only the flags the shell passes and nothing else (an unknown flag exits 2, as argparse does in the
real script). It writes a SYNTHETIC dump in the real layout -- ``perc_det.pkl`` (a list of ``{"sha12", "ws", "box3d": [pack]}``),
``rast/e<ep>_w<ws>.npz`` (planes ``gt`` and ``psw``) and, last, ``dump.json`` -- so the REAL ``r9_e1_eval.py`` and the REAL
``r9_e1_floor.py`` can run on it on the dev box. numpy only; no torch, no checkpoint is loaded.

The "checkpoint" is a small JSON file the dry-run test writes as ``model_ema.pt``: ``{"step": 4000, "miss": [<GT indices the arm
fails to detect>], "fp_rows": <rows of false sidewalk flags in the 5-10 m band>}``. 20 GT per window at x = 3 + 3 i (bands 0-20 /
20-40 / 40-60 m), 2 windows per DEV clip. Detected GT carry logit +8 on the GT position, missed ones are moved 500 m away.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for cand in (HERE.parents[1] / "scripts", HERE):                 # the repo layout (stack/ops/r9e1_read -> stack/scripts)
    if (cand / "r9_e1_eval.py").is_file():
        sys.path.insert(0, str(cand))
import r9_e1_eval as ev                                          # noqa: E402

N_GT = 20


def gt_xy() -> np.ndarray:
    return np.array([[3.0 + 3.0 * i, 2.0 * (1 if i % 2 else -1)] for i in range(N_GT)], np.float32)


def pack(miss) -> dict:
    g = gt_xy()
    xy = g.copy()
    logit = np.full(N_GT, 8.0, np.float32)
    for i in miss:
        xy[i] = (500.0 + i, 500.0)
        logit[i] = -8.0
    cls = np.zeros(N_GT, np.int16)
    return {"ep": None, "logit": logit, "xy": xy, "cls": cls, "cls_corr": cls.copy(), "matched": np.zeros(N_GT, bool),
            "exempt": np.zeros(N_GT, bool), "pair_err": np.zeros(0), "pair_size_err": np.zeros(0), "pair_z_err": np.zeros(0),
            "gt_xy": g, "gt_cls": cls.copy(), "pos": np.ones(N_GT, bool), "ign": np.zeros(N_GT, bool), "hidden": np.zeros(N_GT, bool)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--s2-windows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--require-ema", action="store_true")
    a = ap.parse_args(argv)
    if not a.require_ema:
        raise SystemExit("[stub-dump] the E1 read must pass --require-ema")
    if not Path(a.config).is_file():
        raise SystemExit("[stub-dump] no config at %s" % a.config)
    spec = json.loads(Path(a.ckpt).read_text(encoding="utf-8"))
    clips = ev.load_window_list(a.windows, a.split)
    out = Path(a.out)
    (out / "rast").mkdir(parents=True, exist_ok=True)
    recs = []
    for ci_, c in enumerate(clips):
        for ws in (0, 8):
            recs.append({"sha12": c, "ws": ws, "box3d": [pack(spec.get("miss", []))]})
            rng = np.random.default_rng(1000 * ci_ + ws)                        # the SAME label plane for both arms
            gt = rng.choice(np.array([0, 1, 2, 3, 5, 7, 7, 255], np.uint8), size=(500, 600))
            psw = np.where(gt == 7, 255, 0).astype(np.uint8)
            fp = int(spec.get("fp_rows", 0))
            if fp:
                psw[50:50 + fp, 270:330] = 255
            np.savez(out / "rast" / ("e%03d_w%04d.npz" % (ci_, ws)), gt=gt, psw=psw)
    (out / "perc_det.pkl").write_bytes(pickle.dumps(recs))
    rec = {"tool": "r9e1_read_stub_dump.py", "schema": "r9e1-dump-v1", "stub": True, "ckpt": str(a.ckpt), "weights_are_ema": True,
           "ckpt_kind": "ema_model_only", "step": int(spec.get("step", -1)), "split": {"set": a.split.upper(), "n_clips": len(clips)},
           "n_windows": len(recs), "n_clips": len(clips), "n_box3d_windows": len(recs), "n_rasters": len(recs),
           "order_digest": hashlib.sha256(" ".join("%s:%d" % (r["sha12"], r["ws"]) for r in recs).encode()).hexdigest(),
           "SMOKE_RESTRICTED": False, "seed": 0}
    (out / "dump.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")        # LAST: its presence is the done-marker
    print("stub dump %s: %d windows / %d clips" % (out, len(recs), len(clips)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
