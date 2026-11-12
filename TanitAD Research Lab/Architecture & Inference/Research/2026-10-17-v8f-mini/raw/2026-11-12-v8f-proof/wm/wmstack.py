"""PREREG_V8F_PROOF_ADD2 (frozen, sha256 4a290133...): WMSTACK and JUMPCURVE. T0 world-model diagnostic, dev-box CPU. Imports wmceil.py (same windows, folds, SKILL).
Run: CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=<tree>/stack python wmstack.py"""
from __future__ import annotations
import hashlib
import json
import os
import sys
import time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wmceil as W

HERE = W.HERE
CURVE_KS = (1, 2, 4, 5, 10)
CARD_SHA = "4a290133a305a3c0eee038fb4b4e4e7adaa3e66f232292ee09c0525848bae9d5"


def fixed_coef_stack_pred(z0: np.ndarray, learned_pred: np.ndarray, coef: float = 1.0) -> np.ndarray:
    """Control k2: a stack whose ONLY input is (learned prediction - z_t), coefficient fixed: z_t + coef * (pred - z_t). coef 1 returns the learned prediction."""
    return z0 + coef * (learned_pred - z0)


def stack_features(z0, act_block, learned_pred):
    return np.concatenate([z0, act_block, learned_pred - z0], 1)


def run():
    import torch
    import tanitad
    t0 = time.time()
    tf = os.path.abspath(tanitad.__file__).replace("\\", "/").lower()
    assert tf.startswith(os.path.abspath(os.path.join(HERE, "tree")).replace("\\", "/").lower()), tf
    assert hashlib.sha256(open(os.path.join(HERE, "..", "PREREG_V8F_PROOF_ADD2.md"), "rb").read()).hexdigest() == CARD_SHA
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    L = torch.load(W.LATENTS, weights_only=False)
    sel = np.flatnonzero(L["ok"].all(1).numpy())
    eid = np.array(L["eid"])[sel]
    uniq, Wt = W.boot_weights(eid)
    fold = W.fold_ids(eid)
    banked = json.load(open(W.BANKED))
    res = {"card": "PREREG_V8F_PROOF_ADD2.md (sha256 " + CARD_SHA + ")", "tier": "T0 (world-model diagnostic, never driving performance)",
           "evidence_class": "MEASURED (wmstack.py on eval-141, M2b s0/s1)", "windows_used": int(len(sel)), "episodes_used": int(len(uniq)), "seeds": {}}
    rng = np.random.default_rng(123)
    for tag, path in W.CKPT.items():
        model, cfg = W.load_model(path)
        wm = model.wm
        Z1 = L["Z1"][tag][sel].float()
        A_sc = wm.scale_actions(L["A_raw"][sel].float())[:, :-1]
        pk = W.level_pack(wm, Z1, A_sc)
        out = {"stack": {}, "controls": {}, "curve": {}}
        for (l, k) in W.HEADLINE:
            key = f"L{l}/k{k}"
            zt, at = pk["true"][l], pk["act"][l]
            z0, zk, lp = zt[:, 0], zt[:, k], pk["pred"][(l, k)]
            copy = W.sqerr(z0, zk)
            Y = zk - z0
            ab = W.action_block(at, k)
            err_wm = W.sqerr(lp, zk)
            p_a, _ = W.cross_fit(np.concatenate([z0, ab], 1), Y, eid, fold)
            err_lin = W.sqerr(z0 + p_a, zk)
            Xs = stack_features(z0, ab, lp)
            p_s, lam = W.cross_fit(Xs, Y, eid, fold)
            err_st = W.sqerr(z0 + p_s, zk)
            sk = {n: W.skill_from_err(e, copy) for n, e in (("learned", err_wm), ("linear", err_lin), ("stack", err_st))}
            best = "learned" if sk["learned"] >= sk["linear"] else "linear"
            err_best = err_wm if best == "learned" else err_lin
            es = {n: W.ep_sum(v, eid, uniq) for n, v in {"C": copy, "best": err_best, "stack": err_st}.items()}
            d = W.diff_row(es, "C", "best", "stack", Wt)           # skill_stack - skill_best
            ex = sk["stack"] - max(sk["learned"], sk["linear"])
            out["stack"][key] = {"horizon_s": W.STEP_S[l] * k, "skills": sk, "best_single": best, "stack_minus_best": d,
                                 "exceeds_by_more_than_0.01": bool(ex > W.MARGIN), "adds": bool(ex > W.MARGIN and d["lo"] > 0), "lambda_per_fold": lam}
            # control k1: shuffled change (same stack inputs, the change permuted across windows)
            Ysh = Y[rng.permutation(len(Y))]
            p_sh, _ = W.cross_fit(Xs, Ysh, eid, fold)
            zsh = z0 + Ysh
            sh = W.skill_from_err(W.sqerr(z0 + p_sh, zsh), W.sqerr(z0, zsh))
            # control k2: learned prediction only, coefficient fixed at 1 -> exact reproduction
            e_k2 = W.sqerr(fixed_coef_stack_pred(z0, lp, 1.0), zk)
            ban = banked["rows"][tag][key]["SKILL"]["point"]
            out["controls"][key] = {"k1_shuffled_change_skill": sh, "k1_ok": bool(sh <= 0.005), "k2_fixed_coef_skill": W.skill_from_err(e_k2, copy),
                                    "k2_max_abs_err_diff_vs_learned": float(np.abs(e_k2 - err_wm).max()),
                                    "k2_ok": bool(np.abs(e_k2 - err_wm).max() < 1e-9 and abs(W.skill_from_err(e_k2, copy) - ban) <= 1e-6)}
        # JUMPCURVE, level 1, (z, actions) inputs
        zt1, a1 = pk["true"][1], pk["act"][1]
        for k in CURVE_KS:
            comp, eps1 = W.compose_check(zt1, a1, eid, fold, k, True)
            Xd = np.concatenate([zt1[:, 0], W.action_block(a1, k)], 1)
            pd_, _ = W.cross_fit(Xd, zt1[:, k] - zt1[:, 0], eid, fold)
            err_d = W.sqerr(zt1[:, 0] + pd_, zt1[:, k])
            pf = [{"ratio": float(err_d[fold == f].mean() / (k * eps1[fold == f].mean())),
                   "composed_over_direct": float(comp[fold == f].mean() / err_d[fold == f].mean())} for f in range(W.N_FOLDS)]
            out["curve"][str(k)] = {"horizon_s": 0.2 * k, "ratio_pooled": float(err_d.mean() / (k * eps1.mean())),
                                    "composed_over_direct_pooled": float(comp.mean() / err_d.mean()), "per_fold": pf}
        r = [out["curve"][str(k)]["ratio_pooled"] for k in CURVE_KS if k >= 2]
        out["curve_monotone"] = bool(all(x < 1 for x in r) and all(r[i] >= r[i + 1] for i in range(len(r) - 1)))
        res["seeds"][tag] = out
        print(tag, "done", round(time.time() - t0, 1), flush=True)
    seeds = res["seeds"]
    n_add = {t: sum(v["adds"] for v in s["stack"].values()) for t, s in seeds.items()}
    res["outcomes"] = {"WMSTACK": "STACK-ADDS" if all(n >= 2 for n in n_add.values()) else "STACK-EQUALS-BEST", "stack_horizons_adding": n_add,
                       "JUMPCURVE": "JUMP-MONOTONE" if all(s["curve_monotone"] for s in seeds.values()) else "JUMP-MIXED",
                       "controls_ok": bool(all(c["k1_ok"] and c["k2_ok"] for s in seeds.values() for c in s["controls"].values()))}
    res["runtime_s"] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(HERE, "RESULT_WMSTACK.json"), "w"), indent=1, default=float)
    o = res["outcomes"]
    Ls = ["# VERDICT_WMSTACK: PREREG_V8F_PROOF_ADD2 (WMSTACK and JUMPCURVE)", "",
          "MEASURED (wmstack.py; RESULT_WMSTACK.json), tier T0: a world-model diagnostic, never driving performance. eval-141, 1,976 windows, 141 episodes, 5 episode folds, episode-cluster bootstrap B 2,000.",
          "STRATEGIC family: not applicable to a world-model diagnostic.", "",
          f"**WMSTACK: {o['WMSTACK']}** (horizons where the stack adds, per seed: {o['stack_horizons_adding']}).  **JUMPCURVE: {o['JUMPCURVE']}.**  Controls (k1 shuffled change <= 0.005; k2 fixed-coefficient stack = learned exactly): {'ALL OK' if o['controls_ok'] else 'FAILED, read invalid'}.", ""]
    for tag, s in seeds.items():
        Ls += [f"## {tag}", "| horizon | learned | linear (z, a) | stack | stack - best [95% CI] | adds | k1 shuffled | k2 diff |", "|---|---|---|---|---|---|---|---|"]
        for key, v in s["stack"].items():
            sk, d, c = v["skills"], v["stack_minus_best"], s["controls"][key]
            Ls.append(f"| {v['horizon_s']:.1f} s ({key}) | {sk['learned']:+.4f} | {sk['linear']:+.4f} | {sk['stack']:+.4f} | {d['point']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}] | {v['adds']} | {c['k1_shuffled_change_skill']:+.4f} | {c['k2_max_abs_err_diff_vs_learned']:.1e} |")
        Ls += ["", "| k (s) | ratio(k) = eps_k/(k eps_1) pooled | composed/direct pooled | ratio per fold | composed/direct per fold |", "|---|---|---|---|---|"]
        for k, v in s["curve"].items():
            Ls.append(f"| {k} ({v['horizon_s']:.1f}) | {v['ratio_pooled']:.3f} | {v['composed_over_direct_pooled']:.3f} | " + ", ".join(f"{p['ratio']:.3f}" for p in v["per_fold"]) + " | " + ", ".join(f"{p['composed_over_direct']:.3f}" for p in v["per_fold"]) + " |")
        Ls += [f"monotone (ratio < 1 for k >= 2 and non-increasing over k = 2, 4, 5, 10): {s['curve_monotone']}", ""]
    Ls += ["Choices (card silent): linear = ridge on (z_t, the WM action inputs) as in WMCEIL; better single = the higher point skill of learned / linear; monotonicity read over k >= 2 (ratio(1) is shown, near 1 by construction); lambda from an inner 4-fold episode split; k1 shuffles the change across windows (same stack inputs); the stack at 2.0 s takes the composed ten-step rollout as its learned input."]
    open(os.path.join(HERE, "VERDICT_WMSTACK.md"), "w", encoding="utf-8").write("\n".join(Ls) + "\n")
    print(o, res["runtime_s"])


if __name__ == "__main__":
    run()
