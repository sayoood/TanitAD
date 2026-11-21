#!/usr/bin/env python3
"""VDAC: export P6's proposals and NAVSIM sub-scores of the H48 sets from the pod's label bank (PREREG_VDAC_P6 section 2 S1: S1-Q = the pooled AUC of -d_k for NAVSIM DAC = 0 over P6's proposals on H48 sets that carry P6 labels).
The S1 read runs on CPU locally and needs both this file and the head's predictions (head_pred_h48x.npz + the held-out head_pred_*.npz of the trainer).  Reads the bank through train.OnPolicyBank, the SAME loader the trainer uses (newest ckpt_step per key, the PDM labels
replacing all six components of the set they were computed on), so the sub-scores are the ones the scorer was trained on.  Sets of H48 logs that carry NO PDM label are not exported (they carry no sub-score) and are counted.
Output (npz): bank ('train' | 'heldout'), log, token, step, rank, ckpt_step per set; traj float32 [N, M, T, 3] (x, y, yaw in the t0 rear-axle frame, as the bank stores it); comp float32 [N, M, 6] in PDM_ORDER = nc, dac, ep, ttc, c, ddc (1.0 = no violation);
comp_order; h48_sha256.  A JSON report (counts per bank, per log) is written beside it.  CPU only.
  usage: vdac_export_h48.py --h48 <list> --h48-sha256 <hex> --train-sets D --train-pdm D --heldout-sets D --heldout-pdm D (--ckpt <model_final.pt> | --horizon T) [--n-prop 64] --out <file.npz>
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def export(bank_specs, names, n_prop, horizon):
    """bank_specs: [(tag, sets_dir, pdm_dir)] -> (arrays dict, report dict)"""
    import train as T
    rows, rep = [], {"banks": {}}
    nm = frozenset(names)
    for tag, sets, pdm in bank_specs:
        bank = T.OnPolicyBank(sets, n_prop, horizon, pdm_labels=pdm or None)
        ks = [k for k in sorted(bank.by) if k[0] in nm]
        kept = [k for k in ks if k in bank.pdm_keys]
        rep["banks"][tag] = {"sets_in_bank": len(bank.by), "sets_of_h48_logs": len(ks), "with_pdm_label": len(kept), "without_pdm_label": len(ks) - len(kept),
                             "logs": sorted({k[0] for k in kept}), "pdm_labels_applied": bank.n_pdm}
        for k in kept:
            e = bank.by[k]
            rows.append((tag, k, e))
    if not rows:
        raise SystemExit("the export holds 0 sets: the H48 list names no log of these banks")
    arr = {"bank": np.asarray([r[0] for r in rows]), "log": np.asarray([r[1][0] for r in rows]), "token": np.asarray([r[1][1] for r in rows]),
           "step": np.asarray([r[1][2] for r in rows], np.int64), "rank": np.asarray([r[1][3] for r in rows], np.int64), "ckpt_step": np.asarray([r[2][0] for r in rows], np.int64),
           "traj": np.stack([r[2][1] for r in rows]).astype(np.float32), "comp": np.stack([r[2][2] for r in rows]).astype(np.float32),
           "comp_order": np.asarray(list(T.OnPolicyBank.PDM_ORDER))}
    rep["sets"] = len(rows)
    rep["frames"] = len({(r[1][0], r[1][1]) for r in rows})
    return arr, rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h48", required=True)
    ap.add_argument("--h48-sha256", required=True)
    ap.add_argument("--train-sets", required=True)
    ap.add_argument("--train-pdm", required=True)
    ap.add_argument("--heldout-sets", required=True)
    ap.add_argument("--heldout-pdm", required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--backbone", default="vitl16")
    ap.add_argument("--horizon", type=int, default=0)
    ap.add_argument("--n-prop", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    raw = open(a.h48, "rb").read()
    if hashlib.sha256(raw).hexdigest() != a.h48_sha256:
        print(f"ZZVDAC_EXPORT_FAIL {a.h48} does not hash to the declared {a.h48_sha256}")
        return 1
    names = [x.strip() for x in raw.decode("utf-8").splitlines() if x.strip()]
    horizon = a.horizon
    if a.ckpt:
        import ckpt_io
        from model import REFeConfig
        cfg = ckpt_io.config_for_checkpoint(REFeConfig.for_backbone(a.backbone), a.ckpt)
        horizon, n_prop = int(cfg.horizon_steps), int(cfg.n_proposals)
    else:
        n_prop = a.n_prop
    if horizon <= 0:
        print("ZZVDAC_EXPORT_FAIL --ckpt or --horizon is required")
        return 1
    arr, rep = export([("train", a.train_sets, a.train_pdm), ("heldout", a.heldout_sets, a.heldout_pdm)], names, n_prop, horizon)
    arr["h48_sha256"] = np.asarray(a.h48_sha256)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, **arr)
    rep.update({"h48_sha256": a.h48_sha256, "h48_logs": len(names), "n_prop": n_prop, "horizon": horizon, "file": os.path.basename(a.out),
                "file_sha256": hashlib.sha256(open(a.out, "rb").read()).hexdigest()})
    with open(a.out + ".json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(rep, f, indent=1, sort_keys=True)
    print(f"ZZVDAC_EXPORT_OK {rep['sets']} sets {rep['frames']} frames of {len(names)} H48 logs (train {rep['banks']['train']['with_pdm_label']}, heldout {rep['banks']['heldout']['with_pdm_label']}) -> {os.path.basename(a.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
