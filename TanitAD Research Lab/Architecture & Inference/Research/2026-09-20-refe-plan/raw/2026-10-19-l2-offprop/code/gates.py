#!/usr/bin/env python3
"""L2-OFFPROP gates on the REAL arrays.  They are gates, not results: none trains arm H, H0 or H-null on real labels or reads an A1-OOF of any of them.

    python gates.py cfeat   [--feats F]                 C-FEAT: P2's EP head on the exported tokens reproduces Step A's banked EP scores (max abs diff <= 1e-4 on >= 99 % of rows)
    python gates.py cid     [--feats F]                 C-ID:   zero optimiser steps (H and H0) => P2's EP scores BIT FOR BIT, and A1-OOF == Step A's A1 (RESULT.json, pooled lon + lat)
    python gates.py glive   [--feats F]                 G-LIVE: finite losses, non-zero EP-row gradient, frozen rows EXACTLY 0.0; the mutated loss reads non-zero on a frozen row; the H0 and prox levers REACH the head.
                                                        Gradient probes on the real labels read no score; the 40-step training segment runs on REAL tokens with SYNTHETIC (semi-synthetic) labels.
    python gates.py reach   [--feats F] [--amp 10]      C-REACH (planted): a feature = sign of the true EP difference, appended, gives A1-OOF >= 0.98 on BOTH seeds   (reads the PLANTED arm only)
    python gates.py leak    [--feats F]                 C-LEAK (STRUCTURAL, MM addendum 2026-10-19): every fold's training set is disjoint from its held-out tokens and logs, on the real tokens with
                                                        SHUFFLED labels (no real label is read); each RED mutation (one held-out token injected; a whole fold; a held-out row routed into the pair / in-fan
                                                        index arrays) must make the production `fold_split` raise.  There is no statistical half: a linear 257-parameter head cannot memorise noise.
Each writes --out-dir/<gate>.json (with the recipe / code stamp `run_l2.check_seal` verifies) and prints a ZZ<GATE>_<OK|FAIL> marker.  CPU only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import l2lib  # noqa: E402
import synthetic  # noqa: E402
from l2lib import stepA  # noqa: E402

FEATS = "D:/Projects/TanitAD-artifacts/l2_offprop/feats.npz"
RECIPE = "D:/Projects/TanitAD-artifacts/l2_offprop/recipe_chosen.json"
OUT = "C:/Users/Admin/par18/l2/out/gates"
RUN = "D:/Projects/TanitAD-artifacts/l1_stepA_run"
REACH_BAR = 0.98


def _recipe(a):
    """the sealed recipe file if present; otherwise the PROVISIONAL source default, and the gate JSON says so (and its stamp carries no recipe sha, so the seal refuses it)"""
    if os.path.exists(a.recipe):
        return l2lib.Recipe.from_json(a.recipe), a.recipe
    return l2lib.Recipe(), "PROVISIONAL source default (no recipe file)"


def _write(a, name, obj):
    obj = {**obj, **l2lib.gate_stamp(a.recipe)}
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(l2lib.jsonable(obj), open(os.path.join(a.out_dir, name + ".json"), "w", encoding="utf-8"), indent=1)


# -------------------------------------------------------------------------------------------------------------------------------------------------------------- C-FEAT
def gate_cfeat(a):
    z = np.load(a.feats, allow_pickle=False)
    h = np.load(os.path.join(os.path.dirname(a.feats), "p2_score_head.npz"))
    W, b = h["weight"], h["bias"]
    sc = stepA.load_npz(f"{RUN}/scores.npz")
    res = {"gate": "C-FEAT", "tolerance": 1e-4, "required_share": 0.99, "rows": {}}
    ok = True
    for name, tok, banked, exported in (("in_fan", z["infan_tok"], sc["a_logit"], z["infan_logit"]), ("perturbed", z["pert_tok"], sc["p_logit"], z["pert_logit"])):
        ep = l2lib.ep_apply(tok, W, b)                                                         # P2's EP head applied to the EXPORTED tokens
        d = np.abs(ep - banked[..., l2lib.EP])
        d_all = np.abs(np.asarray(tok, np.float64) @ np.asarray(W, np.float64).T + np.asarray(b, np.float64) - banked)       # all six columns
        d_exp = np.abs(np.asarray(exported, np.float64) - banked)                              # the export run's own logits vs the banked ones
        share = float((d <= 1e-4).mean())
        res["rows"][name] = {"n_rows": int(d.size), "ep_head_on_tokens_vs_banked_max_abs": float(d.max()), "ep_head_on_tokens_vs_banked_share_le_1e-4": share,
                             "ep_head_on_tokens_vs_banked_p99_abs": float(np.quantile(d, 0.99)), "all_six_columns_max_abs": float(d_all.max()),
                             "export_run_logits_vs_banked_max_abs": float(d_exp.max()), "export_run_logits_vs_banked_exact_rows": float((d_exp == 0).all(-1).mean()) if d_exp.ndim == 3 else None}
        ok = ok and share >= 0.99
    res["bundle_sha256_match"] = str(z["bundle_sha256"]) == l2lib.sha256_file(f"{RUN}/bundle/bundle.npz")
    res["perts_sha256_match"] = str(z["perts_sha256"]) == l2lib.sha256_file(f"{RUN}/perts.npz")
    res["pass"] = bool(ok and res["bundle_sha256_match"] and res["perts_sha256_match"])
    _write(a, "cfeat", res)
    print("ZZCFEAT_" + ("OK" if res["pass"] else "FAIL"), json.dumps({k: (v["ep_head_on_tokens_vs_banked_max_abs"], v["ep_head_on_tokens_vs_banked_share_le_1e-4"]) for k, v in res["rows"].items()}), flush=True)
    return 0 if res["pass"] else 1


# -------------------------------------------------------------------------------------------------------------------------------------------------------------- C-ID
def gate_cid(a):
    D = l2lib.load_real(a.feats)
    rec, rec_src = _recipe(a)
    p2 = l2lib.oof_scores(D, l2lib.ArmSpec("P2", "P2"), rec)
    t0 = time.time()
    bit = {}
    for tag, spec in (("H_seed0", l2lib.ArmSpec("H_zero_steps", "H", seed=0, steps=0)), ("H_seed1", l2lib.ArmSpec("H_zero_steps_seed1", "H", seed=1, steps=0)),
                      ("H0_seed0", l2lib.ArmSpec("H0_zero_steps", "H0", seed=0, steps=0)), ("H0_seed1", l2lib.ArmSpec("H0_zero_steps_seed1", "H0", seed=1, steps=0))):
        o = l2lib.oof_scores(D, spec, rec)
        bit[tag] = bool(np.array_equal(o["ep_in"], p2["ep_in"]) and np.array_equal(o["ep_p"], p2["ep_p"]))
        if tag == "H_seed0":
            h0 = o
    ev0 = l2lib.evaluate_arm(D, h0["ep_in"], h0["ep_p"])
    ref = json.load(open(f"{RUN}/RESULT.json"))
    cmp = {}
    worst = 0.0
    for tag in ("all", "lon", "lat"):
        for k in ("A1", "A2", "A3", "A4", "A5", "A5b"):
            for f in ("point", "lo", "hi", "den"):
                d = abs(float(ev0[tag][k][f]) - float(ref[tag][k][f]))
                cmp[f"{tag}.{k}.{f}"] = d
                worst = max(worst, d)
    # the same read through L1's own analyse on the BANKED logits (the reference path), against RESULT.json
    b = stepA.load_npz(f"{RUN}/bundle/bundle.npz")
    P, li, _ = stepA.assemble_pairs(b, stepA.load_npz(f"{RUN}/perts.npz"), stepA.load_npz(f"{RUN}/scores.npz"), stepA.load_npz(f"{RUN}/labels.npz"), D.log_names)
    refpath = stepA.analyse(P, li, D.n_logs, l2lib.BOOT_B, l2lib.BOOT_SEED)
    worst_ref = max(abs(refpath[t][k][f] - ref[t][k][f]) for t in ("all", "lon", "lat") for k in ("A1", "A2", "A3", "A4", "A5") for f in ("point", "lo", "hi"))
    # pairs whose sign could differ between the banked fp32 logits and the float64 head-on-token scores (|diff| tiny): count near-ties
    pa = np.abs(D.P0["p_logit"][:, l2lib.EP] - D.P0["a_logit"][:, l2lib.EP])
    res = {"gate": "C-ID", "recipe_source": rec_src, "zero_step_scores_bit_identical_to_P2": bit,
           "A1_all_zero_step": ev0["all"]["A1"], "A1_all_StepA_RESULT_json": ref["all"]["A1"], "A1_abs_diff_point": abs(ev0["all"]["A1"]["point"] - ref["all"]["A1"]["point"]),
           "A1_is": "POOLED lon+lat (MM ruling E2), den " + str(ev0["all"]["A1"]["den"]),
           "worst_abs_diff_over_all_reported_A_quantities_vs_RESULT_json": worst, "reference_path_worst_abs_diff_vs_RESULT_json": worst_ref,
           "n_pairs_with_banked_EP_logit_gap_below_1e-4": int((pa < 1e-4).sum()), "infan_top8_ep_concordance_P2": ev0["infan_top8_ep_concordance"],
           "seconds": round(time.time() - t0, 1)}
    res["pass"] = bool(all(bit.values()) and res["A1_abs_diff_point"] <= 1e-9 and worst <= 1e-9)
    _write(a, "cid", res)
    print("ZZCID_" + ("OK" if res["pass"] else "FAIL"), json.dumps({"bit": bit, "A1_point_diff": res["A1_abs_diff_point"], "worst": worst, "worst_ref_path": worst_ref}), flush=True)
    return 0 if res["pass"] else 1


# -------------------------------------------------------------------------------------------------------------------------------------------------------------- G-LIVE
def gate_glive(a):
    D = l2lib.load_real(a.feats)
    rec, rec_src = _recipe(a)
    ts = l2lib.make_trainset(D, np.nonzero(D.fold != 0)[0])
    # gradient probes on the REAL labels (finite losses, non-zero EP gradient, frozen rows exactly 0.0, the H0 and prox levers reach the head): no optimiser step is taken on a real label
    g = l2lib.g_live(D.tok_in.reshape(-1, D.D), D.tok_p.reshape(-1, D.D), ts, D.W6, D.b6, rec)
    # a short TRAINING segment with the per-step assertions on, on REAL tokens with SYNTHETIC labels (the semi-synthetic world): H, and H0 (w_pair = 0)
    S = synthetic.semi_synthetic(D, beta=0.5, alpha=0.25, seed=21)
    ts_s = l2lib.make_trainset(S, np.nonzero(S.fold != 0)[0])
    Xi, Xp = S.tok_in.reshape(-1, S.D), S.tok_p.reshape(-1, S.D)
    fit = l2lib.fit_head(Xi, Xp, ts_s, S.W6, S.b6, 0, rec, steps=40)
    fit0 = l2lib.fit_head(Xi, Xp, ts_s, S.W6, S.b6, 0, rec, steps=40, w_pair=0.0)
    h_h0_differ = bool(not np.array_equal(fit["W"][l2lib.EP], fit0["W"][l2lib.EP]))
    red = None
    try:
        l2lib.fit_head(Xi, Xp, ts_s, S.W6, S.b6, 0, rec, steps=3, loss_mutation="leak_other_row")
        red = False
    except RuntimeError as e:
        red = "FROZEN row" in str(e)
    res = {"gate": "G-LIVE", "recipe_source": rec_src, "recipe": rec.as_dict(), "glive": g, "training_segment_world": "REAL tokens + SYNTHETIC labels (semi-synthetic, beta 0.5, alpha 0.25, seed 21): no real label trains a head here",
           "training_segment_steps": fit["steps"], "training_segment_min_ep_grad_abs_sum": fit["min_ep_grad_abs_sum"], "h0_training_segment_steps": fit0["steps"], "h_and_h0_trained_heads_differ": h_h0_differ,
           "mutation_leak_other_row_goes_RED": bool(red), "n_pairs_fold0_train_real": int(len(ts.ia)), "n_fan_rows": int(len(ts.fan_idx))}
    res["pass"] = bool(g["pass"] and g["control_mutation_reads_nonzero_on_frozen"] and red and fit["steps"] == 40 and fit0["steps"] == 40 and h_h0_differ)
    _write(a, "glive", res)
    print("ZZGLIVE_" + ("OK" if res["pass"] else "FAIL"), json.dumps({"pass": g["pass"], "control": g["control_mutation_reads_nonzero_on_frozen"], "mutation_RED": red, "h0_reaches": g["h0_lever_reaches"],
                                                                      "prox_reaches": g["prox_lever_reaches"], "h_h0_differ": h_h0_differ}), flush=True)
    return 0 if res["pass"] else 1


# -------------------------------------------------------------------------------------------------------------------------------------------------------------- C-REACH
def gate_reach(a):
    D = l2lib.load_real(a.feats)
    rec, rec_src = _recipe(a)
    t0 = time.time()
    per = {}
    for seed in l2lib.TRAIN_SEEDS:
        out = l2lib.oof_scores(D, l2lib.ArmSpec(f"planted_s{seed}", "planted", seed=seed), rec)
        ev = l2lib.evaluate_arm(D, out["ep_in"], out["ep_p"])
        per[f"seed{seed}"] = {"A1_OOF_all": ev["all"]["A1"], "A1_OOF_lon": ev["lon"]["A1"], "A1_OOF_lat": ev["lat"]["A1"], "steps_per_fold": [f["steps"] for f in out["diag"]["folds"]]}
    pts = {k: v["A1_OOF_all"]["point"] for k, v in per.items()}
    res = {"gate": "C-REACH (planted)", "recipe_source": rec_src, "planted_amp": rec.planted_amp, "recipe": rec.as_dict(), "per_seed": per, "A1_OOF_all_points": pts,
           "bar": REACH_BAR, "pass": bool(all(p >= REACH_BAR for p in pts.values())), "seconds": round(time.time() - t0, 1),
           "note": "reads the PLANTED arm only (a label-derived feature is appended on purpose); no unplanted head is trained or read.  Bar must hold on BOTH training seeds (MM addendum)."}
    _write(a, "reach", res)
    print("ZZREACH_" + ("OK" if res["pass"] else "FAIL"), json.dumps(pts), flush=True)
    return 0 if res["pass"] else 1


# -------------------------------------------------------------------------------------------------------------------------------------------------------------- C-LEAK (structural; real tokens, shuffled labels)
def gate_leak(a):
    D0 = l2lib.load_real(a.feats)
    D = l2lib.shuffled_labels_data(D0)                       # real tokens, EP labels globally shuffled: no real label is read
    rec, rec_src = _recipe(a)
    t0 = time.time()
    res = {"gate": "C-LEAK (STRUCTURAL; real tokens, SHUFFLED labels: no real label is read)", "recipe_source": rec_src, "folds": [], "red": {}}
    ok = True
    # (1) the clean split of every fold passes the production fold_split (tokens, logs and the TrainSet's own index arrays)
    for f in range(l2lib.N_FOLDS):
        tr, te, ts, chk = l2lib.fold_split(D, f, l2lib.ArmSpec("clean", "H"))
        res["folds"].append({"fold": f, "n_train_tokens": int(len(tr)), "n_test_tokens": int(len(te)), "n_pairs": int(len(ts.ia)), "check": chk})
    # (2) RED mutations: each must make the production path raise 'C-LEAK'
    def raises(fn):
        try:
            fn()
        except RuntimeError as e:
            return "C-LEAK" in str(e)
        return False

    for f in range(l2lib.N_FOLDS):
        res["red"][f"one_held_out_token_injected_fold{f}"] = raises(lambda f=f: l2lib.fold_split(D, f, l2lib.ArmSpec("m", "H", leak_fold=f, leak_n=1)))
    res["red"]["whole_fold_injected_fold0"] = raises(lambda: l2lib.fold_split(D, 0, l2lib.ArmSpec("m", "H", leak_fold=0)))
    res["red"]["oof_scores_trips_before_any_step"] = raises(lambda: l2lib.oof_scores(D, l2lib.ArmSpec("m", "H", leak_fold=2, leak_n=1), rec))
    orig = l2lib.make_trainset

    def routed(D_, tokens, t=None):                         # a held-out perturbation token routed into the pair index array, with the token LISTS left clean
        ts = orig(D_, tokens, t)
        held = int(np.nonzero(D_.fold == 1)[0][0])
        ts.ip = ts.ip.copy()
        ts.ip[0] = held * D_.Q
        return ts

    def routed_fan(D_, tokens, t=None):                     # ... and into the in-fan BCE index array
        ts = orig(D_, tokens, t)
        held = int(np.nonzero(D_.fold == 1)[0][0])
        ts.fan_idx = ts.fan_idx.copy()
        ts.fan_idx[0] = held * l2lib.N_ANCHOR_SLOTS
        return ts
    try:
        l2lib.make_trainset = routed
        res["red"]["held_out_row_routed_into_pair_indices"] = raises(lambda: l2lib.fold_split(D, 1, l2lib.ArmSpec("m", "H")))
        l2lib.make_trainset = routed_fan
        res["red"]["held_out_row_routed_into_infan_indices"] = raises(lambda: l2lib.fold_split(D, 1, l2lib.ArmSpec("m", "H")))
    finally:
        l2lib.make_trainset = orig
    # (3) the production oof_scores of a clean arm runs end to end on this data and scores every token exactly once (its own post-loop assertion)
    o = l2lib.oof_scores(D, l2lib.ArmSpec("clean", "H", seed=0), rec)
    res["clean_oof_scores_ran"] = {"folds": len(o["diag"]["folds"]), "every_fold_recorded_a_leak_check": all("leak_check" in f for f in o["diag"]["folds"])}
    res["all_red_mutations_fire"] = bool(all(res["red"].values()))
    res["pass"] = bool(res["all_red_mutations_fire"] and res["clean_oof_scores_ran"]["every_fold_recorded_a_leak_check"] and len(res["folds"]) == l2lib.N_FOLDS)
    res["seconds"] = round(time.time() - t0, 1)
    _write(a, "leak", res)
    print("ZZLEAK_" + ("OK" if res["pass"] else "FAIL"), json.dumps({"red": res["red"]}), flush=True)
    return 0 if res["pass"] else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("gate", choices=("cfeat", "cid", "glive", "reach", "leak"))
    ap.add_argument("--feats", default=FEATS)
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--recipe", default=RECIPE)
    a = ap.parse_args(argv)
    return {"cfeat": gate_cfeat, "cid": gate_cid, "glive": gate_glive, "reach": gate_reach, "leak": gate_leak}[a.gate](a)


if __name__ == "__main__":
    sys.exit(main())
