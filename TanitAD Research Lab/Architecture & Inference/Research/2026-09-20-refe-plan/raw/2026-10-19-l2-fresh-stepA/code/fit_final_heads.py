#!/usr/bin/env python3
"""L2-FRESH-STEPA step 2: the FINAL L2 EP head = the sealed recipe fit on ALL 400 Step A tokens (no folds), seeds 0 and 1.

Uses the SEALED L2 code unchanged (`l2lib.make_trainset`, `l2lib.fit_head`, `l2lib.g_live`); this file only drives it.  Refuses to run unless
  * the sealed prereg's sha256 equals the pinned value, the recipe file's sha256 equals the pinned value, and l2lib.py / outcome.py / gates.py / synthetic.py (LF sha256) equal the sealed MANIFEST;
  * G-LIVE passes ON THE ALL-400 TRAIN SET (every loss term finite, EP-row gradient non-zero, every frozen row's gradient EXACTLY 0.0, the mutated-loss control reads non-zero, the H0 / prox levers reach);
  * C-ID: a zero-step fit returns P2's six head rows bit for bit, and its EP scores on the 400 tokens' exported features equal the exported P2 logits to 0.0 (float64 recompute).
Writes <out>/final_head_s{0,1}.npz  (weight [6,256] f32, bias [6] f32, steps, recipe, provenance) + <out>/final_heads.json (sha256 of every file, deltas vs P2, gate records).

It reads NO outcome: nothing here computes A1-A5 on any carve.  CPU only.

    CUDA_VISIBLE_DEVICES=-1 python fit_final_heads.py --out D:/Projects/TanitAD-artifacts/l2_fresh_stepA/heads
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

L2_CODE = os.environ.get("L2_CODE", "C:/Users/Admin/par18/l2/code")
sys.path.insert(0, L2_CODE)
import l2lib  # noqa: E402

PREREG = "C:/Users/Admin/par18/l2/PREREG_L2_OFFPROP.md"
PREREG_SHA256_PIN = "3592a1039693dcb5cac1df465fa4dd597ee190d80a934d89fc765d223607a983"        # PREREG_L2_OFFPROP.sha256 (sealed 2026-10-19T15:43)
RECIPE = "D:/Projects/TanitAD-artifacts/l2_offprop/recipe_chosen.json"
RECIPE_SHA256_PIN = "c7f5b75b0a8f25f96c00dcc379ac583fa9c80622ce5600ad4af8387159119263"        # prereg PINS
FEATS = "D:/Projects/TanitAD-artifacts/l2_offprop/feats.npz"
SEALED_CODE = {"l2lib.py": "942033d373868a3028686a564ee99cc476dc8d82b22fc19544afc0052e898b4b",
               "outcome.py": "418494c22c08e71e571366624d17dc5d50d481a35ea6a128d6930c54e03bef37",
               "synthetic.py": "a02e032ac747fa9c1b6dd139c3a52010ad09dc7030e7051d67ed910e019e0f85",
               "gates.py": "9f74763b309700bd2daebbb44a7d2d20e471151392b2252fb7492e2a1a285bc1"}   # code/MANIFEST.sha256 of the seal


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def sha_lf(p):
    return hashlib.sha256(open(p, "rb").read().replace(b"\r\n", b"\n")).hexdigest()


def seal_checks(prereg=PREREG, recipe=RECIPE, code_dir=L2_CODE, pins=None) -> dict:
    """-> record; raises SystemExit(5) on any mismatch (a final head fitted on code or a recipe that is not the sealed one certifies nothing)."""
    pins = pins or {"prereg": PREREG_SHA256_PIN, "recipe": RECIPE_SHA256_PIN, "code": SEALED_CODE}
    rec = {"prereg_sha256": sha_lf(prereg), "recipe_sha256": sha_lf(recipe), "code": {fn: sha_lf(os.path.join(code_dir, fn)) for fn in pins["code"]}}
    bad = []
    if rec["prereg_sha256"] != pins["prereg"]:
        bad.append(f"prereg {rec['prereg_sha256'][:12]} != {pins['prereg'][:12]}")
    if rec["recipe_sha256"] != pins["recipe"]:
        bad.append(f"recipe {rec['recipe_sha256'][:12]} != {pins['recipe'][:12]}")
    for fn, want in pins["code"].items():
        if rec["code"][fn] != want:
            bad.append(f"{fn} {rec['code'][fn][:12]} != {want[:12]}")
    if bad:
        raise SystemExit("SEAL MISMATCH (5): " + "; ".join(bad))
    rec["ok"] = True
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0,1")
    a = ap.parse_args(argv)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "CPU only"
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    seal = seal_checks()
    rec = l2lib.Recipe.from_json(RECIPE)
    D = l2lib.load_real(FEATS)
    ntok = D.N
    if ntok != 400:
        raise SystemExit(f"expected 400 Step A tokens, got {ntok}")
    tokens = np.arange(D.N)
    ts = l2lib.make_trainset(D, tokens)
    Xi, Xp = D.tok_in.reshape(-1, D.D), D.tok_p.reshape(-1, D.D)
    out = {"what": "final L2 EP head: sealed recipe, all 400 Step A tokens, no folds", "seal": seal, "recipe": rec.as_dict(), "n_tokens": int(D.N), "n_logs": int(D.n_logs), "n_pairs": int(len(ts.ia)),
           "n_infan_rows": int(len(ts.fan_idx)), "evidence": "MEASURED (ours); SEEN-SCENE, SELECTION tier; ONE fit per seed (H-ESTIM-SEED-1 stays open)"}
    # ---- G-LIVE on THIS fit's train set
    gl = l2lib.g_live(Xi, Xp, ts, D.W6, D.b6, rec)
    out["g_live"] = {k: gl[k] for k in ("pass", "control_mutation_reads_nonzero_on_frozen", "h0_lever_reaches", "prox_lever_reaches")}
    out["g_live_batches"] = gl["batches"]
    if not gl["pass"] or not gl["control_mutation_reads_nonzero_on_frozen"]:
        raise SystemExit("G-LIVE FAILED on the all-400 train set; no head is written")
    # ---- C-ID: zero optimiser steps == P2's head bit for bit; its EP scores == the exported P2 logits
    z = l2lib.fit_head(Xi, Xp, ts, D.W6, D.b6, 0, rec, steps=0)
    cid_w = bool(np.array_equal(z["W"], np.asarray(D.W6, np.float32)) and np.array_equal(z["b"], np.asarray(D.b6, np.float32)))
    ep0 = l2lib.ep_apply(D.tok_in, z["W"], z["b"])
    f = np.load(FEATS)
    exp_logit = np.asarray(f["infan_logit"], np.float64)[..., l2lib.EP]
    out["c_id"] = {"zero_step_head_equals_P2_bitwise": cid_w, "steps_taken": int(z["steps"]),
                   "max_abs_ep_score_diff_vs_exported_infan_logit": float(np.abs(ep0 - exp_logit).max()),
                   "pass": bool(cid_w and z["steps"] == 0 and float(np.abs(ep0 - exp_logit).max()) < 1e-4)}
    if not out["c_id"]["pass"]:
        raise SystemExit("C-ID FAILED; no head is written")
    # ---- the fits
    heads = {}
    for s in [int(x) for x in a.seeds.split(",")]:
        fit = l2lib.fit_head(Xi, Xp, ts, D.W6, D.b6, s, rec, w_pair=1.0)
        W, b = fit["W"], fit["b"]
        frozen = [r for r in range(6) if r != l2lib.EP]
        p = os.path.join(a.out, f"final_head_s{s}.npz")
        np.savez(p, weight=W.astype(np.float32), bias=b.astype(np.float32), steps=np.array(fit["steps"]), seed=np.array(s), recipe_sha256=np.array(seal["recipe_sha256"]),
                 feats_sha256=np.array(sha_file(FEATS)), n_tokens=np.array(D.N))
        dW = W[l2lib.EP] - np.asarray(D.W6, np.float32)[l2lib.EP]
        heads[f"s{s}"] = {"path": p, "sha256": sha_file(p), "steps": int(fit["steps"]), "min_ep_grad_abs_sum": fit["min_ep_grad_abs_sum"], "frozen_rows_equal_P2_bitwise": bool(
            np.array_equal(W[frozen], np.asarray(D.W6, np.float32)[frozen]) and np.array_equal(b[frozen], np.asarray(D.b6, np.float32)[frozen])),
            "ep_row_l2_delta_vs_P2": float(np.linalg.norm(dW)), "ep_row_l2_norm_P2": float(np.linalg.norm(np.asarray(D.W6, np.float32)[l2lib.EP])),
            "ep_bias_delta_vs_P2": float(b[l2lib.EP] - np.asarray(D.b6, np.float32)[l2lib.EP]), "loss_trace_last": fit["loss_trace"][-1] if fit["loss_trace"] else None,
            "loss_trace_first": fit["loss_trace"][0] if fit["loss_trace"] else None}
    out["heads"] = heads
    if len(heads) == 2:
        w0, w1 = (np.load(heads[k]["path"])["weight"][l2lib.EP] for k in ("s0", "s1"))
        out["seed_pair_ep_row_cosine"] = float(np.dot(w0, w1) / (np.linalg.norm(w0) * np.linalg.norm(w1)))
        out["seed_pair_ep_row_l2_diff"] = float(np.linalg.norm(w0 - w1))
    out["seconds"] = round(time.time() - t0, 1)
    json.dump(out, open(os.path.join(a.out, "final_heads.json"), "w"), indent=1)
    print("ZZFINAL_HEADS_DONE", json.dumps({k: out[k] for k in ("g_live", "c_id", "n_pairs", "seconds")}), flush=True)
    for k, v in heads.items():
        print("ZZHEAD", k, v["sha256"], "steps", v["steps"], "dEP", round(v["ep_row_l2_delta_vs_P2"], 4), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
