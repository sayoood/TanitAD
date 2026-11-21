"""Synthetic scorer worlds for select_ckpt_r8 -- KNOWN answers, built deterministically (no navtrain data exists, and none is touched).

``make_world`` returns a ``World``: per-arm token tables, the sample dict, ``log_of`` and the exact deltas it was built with.  Construction
keeps the PDMS identity exactly satisfied row by row (``NC * DAC * (5 EP + 5 TTC + 2 C) / 12``): in the simple worlds DAC = EP = TTC = C = 1
and NC carries the PDMS; ``subscore_world`` builds arms whose sub-scores DISAGREE with the PDMS direction.

``write_world`` writes the world as the scorer + bridge artifacts the real run produces (CSV with the devkit's trailing ``average`` row,
``counts.json``, ``binding.json``, seam ``npz`` + manifests) so the file-level refusals can be tested at the real boundary.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass, field

import numpy as np

CKPT_MD5 = {"ema40000": "57807456bd7169800be159ff95871d22", "ema48852": "b4364029b68f29a33eeb4aebbbbd002b"}
CKPT_STEP = {"ema40000": 40000, "ema48852": 48852}
RT_DEPLOY = "3319b003fecc83577430f07dad762bce0a9a0b8594e9748e9a61bc4d264136b7"
SEAM_ARM = {0: "R8_RT", 1: "R8_RT_s1"}
SRC_ARM = {0: "R8_LEGAL", 1: "R8_LEGAL_s1"}


@dataclass
class World:
    tokens: list
    log_of: dict
    tables: dict            # (ckpt, seed) -> {token: row}
    sample: dict
    deltas: dict = field(default_factory=dict)


def _sha_tokens(tokens):
    return hashlib.sha256("\n".join(sorted(tokens)).encode()).hexdigest()


def _row(nc, dac=1.0, ep=1.0, ttc=1.0, c=1.0, ddc=1.0):
    pdms = nc * dac * (5 * ep + 5 * ttc + 2 * c) / 12.0
    return {"NC": float(nc), "DAC": float(dac), "EP": float(ep), "TTC": float(ttc), "C": float(c), "DDC": float(ddc),
            "PDMS": float(pdms), "valid": True, "standin": False}


def make_sample(tokens, log_of, status="FIXED"):
    return {"status": status, "tokens": sorted(tokens), "count": len(tokens), "sha256_sorted_tokens": _sha_tokens(tokens),
            "token_log": dict(log_of), "log_names": sorted(set(log_of.values()))}


def make_world(*, n_logs=13, per_log=80, shift=(0.02, 0.02), seed=7, base_lo=0.10, base_hi=0.80, log_sd=(0.0, 0.0),
               s1_noise=0.004, order_seed=11, constant_base=None, share_log_effect=False) -> World:
    """40k_s = 48,852_s + shift[s] + per-log effect (zero-mean over logs, so the delta is EXACTLY shift[s])."""
    rng = np.random.default_rng(seed)
    n = n_logs * per_log
    tokens = [f"tok{i:05d}" for i in range(n)]
    log_of = {t: f"log{i // per_log:02d}" for i, t in enumerate(tokens)}
    base0 = np.full(n, constant_base) if constant_base is not None else rng.uniform(base_lo, base_hi, n)
    noise = np.clip(rng.normal(0.0, s1_noise, n), -2.5 * s1_noise, 2.5 * s1_noise)
    noise -= noise.mean()
    base1 = base0 + (0 * noise if constant_base is not None else noise)
    eff = []
    for s in (0, 1):
        if s == 1 and share_log_effect:
            eff.append(eff[0])
            continue
        e = rng.normal(0.0, log_sd[s], n_logs) if log_sd[s] > 0 else np.zeros(n_logs)
        e = e - e.mean()
        eff.append(np.repeat(e, per_log))
    arms = {("ema48852", 0): base0, ("ema48852", 1): base1,
            ("ema40000", 0): base0 + shift[0] + eff[0], ("ema40000", 1): base1 + shift[1] + eff[1]}
    tables = {}
    orng = np.random.default_rng(order_seed)
    for key, arr in arms.items():
        order = orng.permutation(n)                                  # a DIFFERENT dict insertion order per arm
        tables[key] = {tokens[i]: _row(float(arr[i])) for i in order}
    deltas = {s: float(np.mean(arms[("ema40000", s)]) - np.mean(arms[("ema48852", s)])) for s in (0, 1)}
    return World(tokens, log_of, tables, make_sample(tokens, log_of), deltas)


def subscore_world(*, n_logs=13, per_log=80, seed=3) -> World:
    """PDMS favours 40k by a constant (+1/12) while TTC favours 48,852 (-0.1) and EP favours 40k (+0.3)."""
    rng = np.random.default_rng(seed)
    n = n_logs * per_log
    tokens = [f"tok{i:05d}" for i in range(n)]
    log_of = {t: f"log{i // per_log:02d}" for i, t in enumerate(tokens)}
    ep = rng.uniform(0.2, 0.69, n)
    tables = {}
    for s in (0, 1):
        tables[("ema48852", s)] = {t: _row(1.0, 1.0, float(ep[i]), 1.0, 1.0) for i, t in enumerate(tokens)}
        tables[("ema40000", s)] = {t: _row(1.0, 1.0, float(ep[i]) + 0.30, 0.9, 1.0) for i, t in enumerate(tokens)}
    return World(tokens, log_of, tables, make_sample(tokens, log_of))


# ------------------------------------------------------------------------------------------------------------------------------
# file-level world: the artifacts the real run leaves behind
# ------------------------------------------------------------------------------------------------------------------------------
TERMS = ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort",
         "driving_direction_compliance", "score")
KEYS = ("NC", "DAC", "EP", "TTC", "C", "DDC", "PDMS")


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def write_csv(path, table, order=None):
    toks = order if order is not None else sorted(table)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([""] + ["token", "valid"] + list(TERMS))
        for i, t in enumerate(toks):
            r = table[t]
            w.writerow([i, t, r["valid"]] + [repr(r[k]) for k in KEYS])
        avg = [sum(table[t][k] for t in toks) / len(toks) for k in KEYS]
        w.writerow([len(toks), "average", True] + avg)


def write_seam(path, table, toks, arm, standin=()):
    srcs = np.asarray(["cv_standin" if t in standin else "precomputed" for t in toks])
    np.savez(path, token=np.asarray(toks), fingerprint=np.asarray(["fp"] * len(toks)), source=srcs,
             poses=np.zeros((len(toks), 8, 3), np.float32), knots=np.zeros((len(toks), 8, 2), np.float32),
             sampling=np.asarray([8, 0.5]), arm=np.asarray(arm))


def write_world(root, w: World, *, drop=None) -> dict:
    """Write the layout ``select_ckpt_r8.spec_from_layout`` expects under ``root``; returns the spec dict (also written to root/spec.json)."""
    root = str(root).replace("\\", "/")
    os.makedirs(root, exist_ok=True)
    sample_path = f"{root}/NAVTRAIN_SAMPLE_TOKENS.json"
    json.dump(w.sample, open(sample_path, "w", encoding="utf-8"))
    sample_sha = w.sample["sha256_sorted_tokens"]
    toks = sorted(w.tokens)
    arms = {}
    for ck in ("ema40000", "ema48852"):
        bdir = f"{root}/{ck}/bridge"
        os.makedirs(bdir, exist_ok=True)
        for s in (0, 1):
            lab = f"r8sel_{ck}_s{s}"
            sd = f"{root}/scores/{lab}"
            os.makedirs(sd, exist_ok=True)
            tab = w.tables[(ck, s)]
            write_csv(f"{sd}/{lab}.csv", tab)
            write_seam(f"{bdir}/seam_{SEAM_ARM[s]}.npz", tab, toks, SEAM_ARM[s])
            fwd = {"arm": SRC_ARM[s], "split": "navtrain", "device": ["cuda"], "precision": ["as_trained"], "partial": False,
                   "n_cv_standin_rows": 0, "n_rows": len(toks), "spec": {"seed": s, "frames": "ST", "speed": "legal"},
                   "model": {"ckpt_md5": CKPT_MD5[ck], "step": CKPT_STEP[ck]}, "KPR": {"verdict": "PASS", "max_abs_m": 1e-6}}
            rt = {"arm": SEAM_ARM[s], "split": "navtrain", "device": ["derived"], "partial": False, "n_cv_standin_rows": 0,
                  "n_rows": len(toks), "derived_from": f"rows_{SRC_ARM[s]}.jsonl", "retime": {"table_sha256": RT_DEPLOY}}
            json.dump(fwd, open(f"{bdir}/seam_{SRC_ARM[s]}.manifest.json", "w"))
            json.dump(rt, open(f"{bdir}/seam_{SEAM_ARM[s]}.manifest.json", "w"))
            counts = {"status": "PASS", "failures": [], "n_expected": len(toks), "csv_valid_rows": len(toks), "C3_missing": 0,
                      "C3_unexpected": 0, "arm": f"SEAM:{bdir}/seam_{SEAM_ARM[s]}.npz"}
            json.dump(counts, open(f"{sd}/{lab}.counts.json", "w"))
            binding = {"label": lab, "split": "navtrain", "seam": f"{bdir}/seam_{SEAM_ARM[s]}.npz",
                       "seam_sha256": _sha(f"{bdir}/seam_{SEAM_ARM[s]}.npz"), "csv": f"{sd}/{lab}.csv",
                       "csv_sha256": _sha(f"{sd}/{lab}.csv"), "sample_sha256_sorted_tokens": sample_sha}
            json.dump(binding, open(f"{sd}/{lab}.binding.json", "w"))
            arms[f"{ck}_s{s}"] = {"csv": f"{sd}/{lab}.csv", "counts": f"{sd}/{lab}.counts.json", "binding": f"{sd}/{lab}.binding.json",
                                  "seam": f"{bdir}/seam_{SEAM_ARM[s]}.npz", "bridge_dir": bdir}
    spec = {"sample": sample_path, "arms": arms, "references": {}}
    json.dump(spec, open(f"{root}/spec.json", "w"), indent=1)
    return spec
