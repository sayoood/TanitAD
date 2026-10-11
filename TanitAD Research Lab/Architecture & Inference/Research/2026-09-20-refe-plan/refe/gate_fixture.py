#!/usr/bin/env python3
"""Fixtures for the launch gate: (1) a TINY P2-shaped rehearsal bank on the dev box, (2) `carve_gate_bank`, the pod-side step that makes
the SMOKE bank (the real rows of the gate's labelled / unlabelled on-policy sets, so every micro-batch can reach the scorer loss).

Both reuse the existing, tested carve of `diag_onpolicy_train.py`; nothing here defines a label format of its own.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
CONSISTENT = Path("D:/Projects/TanitAD/data/refe_consistent")        # the dev box's real bank the existing diags carve from
PDM_ORDER = ("nc", "dac", "ep", "ttc", "c", "ddc")


def row_key(r: dict) -> tuple:
    return (r["log_name"], r.get("token", ""), int(r["step"]), int(r.get("rank", 0)))


def write_pdm_labels(keys: list, out: Path, n_labelled: int, ckpt_step: int = 1, n_prop: int = 64) -> list:
    """pdm_synth.jsonl for the first `n_labelled` keys (the others stay unlabelled, so --pdm-only has something to drop and the G3 mutation
    is not inert)."""
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    lab = keys[:n_labelled]
    lines = []
    for k in lab:
        d = {"key": list(row_key(k)), "ckpt_step": ckpt_step}
        for c in PDM_ORDER:
            d[c] = [float(x) for x in rng.random(n_prop)]
        lines.append(json.dumps(d) + "\n")
    (out / "pdm_synth.jsonl").write_text("".join(lines), encoding="utf-8")
    return [row_key(k) for k in lab]


def tiny_p2_fixture(root, src_bank=CONSISTENT, n_labelled: int = 3) -> dict:
    """root/bank (6 tuples, 4 scenes), root/sets (6 on-policy sets), root/pdm (labels for `n_labelled`), root/init.pt (a FULL ViT-S checkpoint
    with a 1-camera rig, as P2's --init-from is a FULL checkpoint).  Returns the paths."""
    import torch
    import ckpt_io
    import diag_onpolicy_train as D
    from model import REFe, REFeConfig
    root = Path(root)
    bank = root / "bank"
    keys = D.carve(Path(src_bank), bank)
    D.write_sets(keys, root / "sets", seed=7, shuffle=False)
    labelled = write_pdm_labels(keys, root / "pdm", n_labelled)
    cfg = REFeConfig.for_backbone("vits16")
    cfg.n_cameras, cfg.cameras = 1, cfg.cameras[:1]
    torch.manual_seed(0)
    m = REFe(cfg)
    init = root / "init.pt"
    ckpt_io.atomic_save({"format": ckpt_io.FORMAT_FULL, "model": ckpt_io.full_state(m), "meta": {"backbone": "vits16"}}, init)
    return {"bank": str(bank), "sets": str(root / "sets"), "pdm": str(root / "pdm"), "init": str(init), "keys": [row_key(k) for k in keys],
            "labelled": labelled}


def rehearsal_argv(fx: dict, out: str, *, steps: int = 2) -> list:
    """The rehearsal argv: P2's levers (on-policy + PDM-only + scorer-sees-trunk + warm start + cosine + AdamW) on the tiny bank, ViT-S, one
    camera, synthetic images, CPU.  `--synthetic`, `--cpu` and `--n-cameras 1` are the rehearsal's scale-downs, never a launch."""
    return ["--backbone", "vits16", "--n-cameras", "1", "--synthetic", "--cpu", "--targets", fx["bank"], "--scorer-targets", fx["bank"],
            "--init-from", fx["init"], "--scorer-mode", "onpolicy", "--onpolicy-targets", fx["sets"], "--pdm-labels", fx["pdm"], "--pdm-only",
            "--scorer-sees-trunk", "--batch", "1", "--accum", "2", "--steps", str(steps), "--lr", "5e-5", "--log-every", "1",
            "--ckpt-every-min", "999", "--out", out]


def carve_gate_bank(src_targets: str, sets_dir: str, dst: str) -> dict:
    """POD: copy into `dst` only the rows of `src_targets` (targets_rank*.jsonl and scorer_targets*.jsonl) whose key is a key of a set in
    `sets_dir` (onpolicy_*.jsonl) -- the SMOKE bank, in which the scorer loss is reachable on every micro-batch.  Streams; never loads a
    file whole.  Returns counts; raises when no row matched (a smoke on a bank that cannot reach the scorer proves nothing)."""
    import glob
    import os
    keys = set()
    for fp in sorted(glob.glob(os.path.join(sets_dir, "onpolicy_*.jsonl"))):
        with open(fp, encoding="utf-8") as fh:
            for ln in fh:
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if r.get("kind") == "onpolicy_set":
                    keys.add((r["log_name"], r.get("token", ""), int(r["step"]), int(r.get("rank", 0))))
    toks = {k[1] for k in keys}
    dst_p = Path(dst)
    dst_p.mkdir(parents=True, exist_ok=True)
    counts = {}
    for fp in sorted(glob.glob(os.path.join(src_targets, "targets_rank*.jsonl")) + glob.glob(os.path.join(src_targets, "scorer_targets*.jsonl"))):
        n = 0
        with open(fp, encoding="utf-8") as fh, open(dst_p / os.path.basename(fp), "w", encoding="utf-8", newline="\n") as out:
            for ln in fh:
                if not any(t in ln for t in toks):
                    continue
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if row_key(r) in keys:
                    out.write(ln if ln.endswith("\n") else ln + "\n")
                    n += 1
        counts[os.path.basename(fp)] = n
    # the files the trainer reads besides the jsonl (calib table etc.) are named on the argv and are NOT part of the bank dir
    if not any(v for k, v in counts.items() if k.startswith("targets_rank")):
        raise RuntimeError(f"no target row matched any of the {len(keys)} gate sets (counts {counts}): the smoke bank would be empty")
    return {"keys": len(keys), "rows": counts}


def carve_p6_gate_bank(src_targets: str, metric_cache: str, dst: str, n: int = 64) -> dict:
    """POD, `--labeller online` (P6): the SMOKE bank. `carve_gate_bank` keys on the on-policy sets and does not fit a run whose labels come from the D9 metric cache;
    here a row is kept only if ITS parts file exists -- `<cache>/<log>/<token>.pkl.xz`, or `<token>__lane_rank_<k>.pkl.xz` for a lane-rank twin (online_labeller.parts_path,
    the very path the labeller opens) -- so every micro-batch of the smoke reaches the online score loss and G-LIVE-LAB's coverage is real.  At most `n` rows per
    `targets_rank*.jsonl`, in file order; streams, never loads a file whole.  Raises when rank 0 yields nothing (a smoke on a bank the labeller cannot reach proves
    nothing).  Returns {"rows": {file: kept}, "scanned": {file: n}, "no_parts": {file: n}}."""
    import glob
    import os
    import online_labeller as OL
    dst_p = Path(dst)
    dst_p.mkdir(parents=True, exist_ok=True)
    kept, scanned, no_parts = {}, {}, {}
    files = sorted(glob.glob(os.path.join(src_targets, "targets_rank*.jsonl")))
    for fp in files:
        name = os.path.basename(fp)
        k = s = np_ = 0
        with open(fp, encoding="utf-8") as fh, open(dst_p / name, "w", encoding="utf-8", newline="\n") as out:
            for ln in fh:
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                s += 1
                if OL.parts_path(metric_cache, r["log_name"], r["token"], r.get("aug")).is_file():
                    out.write(ln if ln.endswith("\n") else ln + "\n")
                    k += 1
                    if k >= n:
                        break
                else:
                    np_ += 1
        kept[name], scanned[name], no_parts[name] = k, s, np_
    if not files or not kept.get("targets_rank0.jsonl"):
        raise RuntimeError(f"no rank-0 row of {src_targets} has a parts file under {metric_cache} (kept {kept}, rows without parts {no_parts}): "
                           "the P6 smoke bank would be empty")
    return {"rows": kept, "scanned": scanned, "no_parts": no_parts}
