"""HEADROW-PROBE additive export (PREREG_HEADROW_PROBE_AMENDMENT1.md): per-proposal OOF predictions, labels6 and P2's frozen scores, and (from the SOURCE TABLE, the per-log fit-set stores) each
proposal's mean speed.  NEW FILE: the gate-bound headrow_probe.py is imported, not edited.  No statistic by speed is computed here.

  python export_oof.py export  [--run-dir D] [--stage-dir D]   -> <stage-dir>/oof_proposals_f10.npz + oof_proposals_f10.meta.json   (written once)
  python export_oof.py verify  [--run-dir D] [--stage-dir D]   -> <stage-dir>/export_verify.json  (re-derives every concordance from the EXPORTED arrays and compares with probe_result.json)

Every file read goes through the landed PathGuard; the bank units are re-hashed against the manifest BEFORE np.load, and only the small arrays are loaded (the 1 GB feature tensor is never read).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import headrow_probe as H   # noqa: E402

E = H.E
RUN_DEFAULT = "C:/Users/Admin/par18/probe/run/f10"
STAGE_DEFAULT = ("C:/Users/Admin/par18/probe/out/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-19-headrow-probe/raw/f10")
SETS_DIR = "C:/Users/Admin/srttc_s0/work/sets"
OUT_NAME = "oof_proposals_f10.npz"
HORIZON_S = 4.0            # 20 steps at 5 Hz (ref/model.py:94 `horizon_steps: int = 20   # 20 @ 5 Hz = 4 s`): INHERITED, not re-measured
N_STEPS = 20
SPEED_DEF = ("mean_speed_mps = (sum over the 20 steps of the Euclidean xy step length, the first segment starting at the ego origin (0,0) at t0) / 4.0 s; traj3 = ego-frame x, y, yaw of the "
             "student's own 64 proposals of the fit set; dt = 0.2 s (5 Hz) taken from ref/model.py:94, INHERITED")


def guard_for(run_dir: str, stage_dir: str):
    opts = H.POptions(fraction=10)
    return E.PathGuard(read_dirs=[opts.bank_dir, SETS_DIR, run_dir, stage_dir], read_files=[opts.manifest, opts.ranked_logs, os.path.abspath(__file__), E.__file__, os.path.join(HERE, "headrow_probe.py")],
                       write_dirs=[stage_dir]), opts


def load_small_bank(guard, opts):
    """The bank's small arrays for the selected logs, in the SAME order as the landed `load_fit_data` (units sorted, rows in unit order)."""
    manifest, man_sha = E.load_manifest(guard, opts.manifest)
    rows, ranked_sha = E.load_ranked_logs(guard, opts.ranked_logs, E.RANKED_LOGS_SHA256, E.N_FIT_LOGS)
    selected, _ = E.select_logs(rows, 10, None)
    sel_pos = {r["log"]: i for i, r in enumerate(selected)}
    ranks = np.array([r["rank"] for r in selected], np.int64)
    units = [n for n in E._io_listdir(guard, opts.bank_dir) if n.endswith(".sets.npz")]
    acc = {k: [] for k in ("log", "token", "step", "rank", "ckpt_step", "labels6", "logits", "top8")}
    used = {}
    for n in units:
        p = os.path.join(opts.bank_dir, n)
        sha = E._io_sha256(guard, p)
        if n not in manifest or sha != manifest[n]:
            raise H.ProbeError(f"unit {n}: sha256 != manifest")
        z = E._io_npz(guard, p)
        lg = z["log"]
        mask = np.array([str(x) in sel_pos for x in lg], dtype=bool)
        if mask.any():
            for k in acc:
                acc[k].append(z[k][mask])
            used[n] = sha
        z.close()
    cat = {k: np.concatenate(v) for k, v in acc.items()}
    log_idx = np.array([sel_pos[str(x)] for x in cat["log"]], np.int64)
    fold = np.array([E.fold_of_rank(ranks[i]) for i in log_idx], np.int8)
    return cat, log_idx, fold, selected, {"manifest_sha256": man_sha, "ranked_sha256": ranked_sha, "units_used_sha256": used}


def mean_speed_from_traj(traj3: np.ndarray) -> np.ndarray:
    """[S,64,20,3] -> [S,64] mean speed (m/s): path length through (0,0), p1..p20, divided by 4.0 s.  float64 internally."""
    xy = traj3[..., :2].astype(np.float64)
    p = np.concatenate([np.zeros(xy.shape[:-2] + (1, 2)), xy], axis=-2)
    return (np.linalg.norm(np.diff(p, axis=-2), axis=-1).sum(-1) / HORIZON_S).astype(np.float32)


def speeds_for_sets(guard, cat) -> tuple:
    """Join the bank sets to the source-table stores by (token, step, rank, ckpt_step); accept a log only if labels6 of every matched row equals the bank's EXACTLY."""
    N = len(cat["log"])
    speed = np.full((N, 64), np.nan, np.float32)
    shas = {}
    for lg in sorted(set(str(x) for x in cat["log"])):
        p = f"{SETS_DIR}/{lg}.npz"
        raw = E._io_read_bytes(guard, p)
        shas[lg] = E.sha256_bytes(raw)
        z = np.load(io.BytesIO(raw), allow_pickle=False)
        key = {(str(t), int(s), int(r), int(c)): i for i, (t, s, r, c) in enumerate(zip(z["token"], z["step"], z["rank"], z["ckpt_step"]))}
        if len(key) != len(z["token"]):
            raise H.ProbeError(f"store {lg}: duplicate (token, step, rank, ckpt_step) keys")
        sp_store = mean_speed_from_traj(z["traj3"])
        lb_store = z["labels6"]
        rows = np.nonzero(cat["log"] == lg)[0]
        for i in rows:
            j = key.get((str(cat["token"][i]), int(cat["step"][i]), int(cat["rank"][i]), int(cat["ckpt_step"][i])))
            if j is None:
                raise H.ProbeError(f"store {lg}: bank set {i} ({cat['token'][i]}) not found")
            if not np.array_equal(lb_store[j], cat["labels6"][i]):
                raise H.ProbeError(f"store {lg} row {j} vs bank set {i}: labels6 differ (the join is not proven)")
            speed[i] = sp_store[j]
    if np.isnan(speed).any():
        raise H.ProbeError("some bank sets have no speed")
    return speed, shas


def do_export(a) -> dict:
    guard, opts = guard_for(a.run_dir, a.stage_dir)
    E._io_makedirs(guard, a.stage_dir)
    cat, log_idx, fold, selected, prov = load_small_bank(guard, opts)
    N = len(cat["log"])
    out = {"token": cat["token"].astype("U16"), "log": cat["log"].astype("U64"), "step": cat["step"].astype(np.int32), "rank": cat["rank"].astype(np.int32),
           "ckpt_step": cat["ckpt_step"].astype(np.int64), "set_index": np.arange(N, dtype=np.int64), "fold": fold, "proposal_index": np.arange(64, dtype=np.int16),
           "labels6": cat["labels6"].astype(np.float32), "label_order": np.array(["nc", "dac", "ep", "ttc", "c", "ddc"]), "p2_logits6": cat["logits"].astype(np.float32),
           "p2_top8": cat["top8"].astype(np.int16)}
    pk = E.picks_np(out["p2_logits6"])
    out["p2_pick_index"] = pk.astype(np.int16)
    speed, store_shas = speeds_for_sets(guard, cat)
    out["mean_speed_mps"] = speed
    out["p2_pick_mean_speed_mps"] = speed[np.arange(N), pk]
    oof_shas = {}
    for r in H.ROWS:
        for arm in H.ARMS:
            for sd in H.SEEDS:
                nm = f"oof_{r}_{arm}_s{sd}"
                p = os.path.join(a.run_dir, "oof", nm + ".npz")
                raw = E._io_read_bytes(guard, p)
                oof_shas[nm] = E.sha256_bytes(raw)
                Z = np.load(io.BytesIO(raw), allow_pickle=False)["Z"].astype(np.float32)
                if Z.shape != (N, 64) or not np.isfinite(Z).all():
                    raise H.ProbeError(f"{nm}: shape {Z.shape} / non-finite")
                out[nm] = Z
    out["what"] = np.array("HEADROW-PROBE per-proposal export for FASTNEG-CENSUS: oof_<row>_<arm>_s<seed> = OOF logit of the replaced column [sets,64] (row in nc,dac,ep,ttc; arm L = linear refit, "
                           "M = skip + hidden-256 GELU branch); labels6 / p2_logits6 [sets,64,6] (order label_order); mean_speed_mps [sets,64] and p2_pick_mean_speed_mps [sets]; a SET is "
                           "(token, step, rank, ckpt_step); a token can have several sets; set_index = the row used by every table.")
    out["speed_definition"] = np.array(SPEED_DEF)
    out["traj_dt_s"] = np.array(0.2)
    bio = io.BytesIO()
    np.savez_compressed(bio, **out)
    blob = bio.getvalue()
    sha = E._io_write_bytes(guard, os.path.join(a.stage_dir, OUT_NAME), blob, exclusive=True)
    tok_counts = np.unique(out["token"], return_counts=True)[1]
    meta = {"schema": "headrow-export-v1", "file": OUT_NAME, "sha256": sha, "bytes": len(blob), "n_sets": N, "n_tokens": int(len(tok_counts)), "sets_per_token_min_max": [int(tok_counts.min()), int(tok_counts.max())],
            "keys": sorted(out), "bank": prov, "oof_inputs_sha256": oof_shas, "store_sha256_combined": E.sha256_bytes("".join(f"{k}:{v}\n" for k, v in sorted(store_shas.items())).encode()),
            "n_stores": len(store_shas), "speed_definition": SPEED_DEF, "trajectories_in_bank_units": False,
            "looked_in": ["C:/Users/Admin/srttc_s0/work/bank/*.sets.npz (members: log, token, step, rank, ckpt_step, feat, logits, top8, labels6, label_order, ctrl_*; no trajectory)",
                          "C:/Users/Admin/srttc_s0/work/bank/*.fan.npz (lean dump: proposals / proposals20 present only as empty arrays)",
                          "C:/Users/Admin/srttc_s0/work/sets/<log>.npz (SOURCE TABLE: token, step, rank, ckpt_step, traj3 [S,64,20,3], labels6): USED"],
            "no_statistic_by_speed_computed": True, "written": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    E._io_write_bytes(guard, os.path.join(a.stage_dir, "oof_proposals_f10.meta.json"), E.canon_json(meta), exclusive=True)
    print(json.dumps({k: meta[k] for k in ("sha256", "bytes", "n_sets", "n_tokens", "sets_per_token_min_max", "n_stores")}), flush=True)
    return meta


def do_verify(a) -> dict:
    guard, opts = guard_for(a.run_dir, a.stage_dir)
    z = np.load(io.BytesIO(E._io_read_bytes(guard, os.path.join(a.stage_dir, OUT_NAME))), allow_pickle=False)
    R = json.loads(E._io_read_bytes(guard, os.path.join(a.run_dir, "probe_result.json")).decode("utf-8"))
    lb, p2 = z["labels6"], z["p2_logits6"]
    N = len(lb)
    res = {"schema": "headrow-export-verify-v1", "cells": {}, "ok": True}
    pk = E.picks_np(p2)
    res["p2_pick_equal"] = bool(np.array_equal(pk, z["p2_pick_index"].astype(np.int64)))
    res["p2_pick_speed_equal"] = bool(np.array_equal(z["mean_speed_mps"][np.arange(N), pk], z["p2_pick_mean_speed_mps"]))
    res["fold_sets"] = [int((z["fold"] == k).sum()) for k in range(5)]
    res["ok"] = bool(res["p2_pick_equal"] and res["p2_pick_speed_equal"] and res["fold_sets"] == list(H.EPR1_FOLD_SETS))
    for r in H.ROWS:
        c = H.ROW_COL[r]
        y = torch.from_numpy(np.ascontiguousarray(lb[..., c]))
        v = torch.from_numpy(np.ascontiguousarray(E.safe_np(lb) if r == "ep" else np.ones(lb.shape[:-1], bool)))
        for arm in H.ARMS:
            for sd in H.SEEDS:
                Zc = z[f"oof_{r}_{arm}_s{sd}"]
                n, d = H.conc_per_set(Zc, y, v)
                got = float(n.sum() / d.sum())
                want = R["rows"][r]["seeds"][str(sd)][arm]["conc64"]["point"]
                ok = abs(got - want) <= 1e-9
                res["cells"][f"{r}/{arm}/s{sd}"] = {"conc64_from_export": got, "conc64_in_probe_result": want, "ok": bool(ok)}
                res["ok"] = res["ok"] and ok
        n, d = H.conc_per_set(p2[..., c], y, v)
        want = R["rows"][r]["p2_reference"]["conc64"]["point"]
        res["cells"][f"{r}/p2_reference"] = {"conc64_from_export": float(n.sum() / d.sum()), "conc64_in_probe_result": want, "ok": bool(abs(float(n.sum() / d.sum()) - want) <= 1e-9)}
        res["ok"] = res["ok"] and res["cells"][f"{r}/p2_reference"]["ok"]
    E._io_write_bytes(guard, os.path.join(a.stage_dir, "export_verify.json"), E.canon_json(res), exclusive=True)
    print("verify ok:", res["ok"], flush=True)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("export", "verify"))
    ap.add_argument("--run-dir", default=RUN_DEFAULT)
    ap.add_argument("--stage-dir", default=STAGE_DEFAULT)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    {"export": do_export, "verify": do_verify}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
