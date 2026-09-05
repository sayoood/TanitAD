"""D-REFAV1-CCOS-EVAL step 1(b)-(d): the SHAPE of what the planner emitted, read from a refav1
dump's decisions sidecar BY CONTENT (never from `plan_source`, which mislabels — RESULT
2026-09-04 §5), for one or more dumps on the same windows.

Per dump / arm `cl`:
  (b) L / R / straight share of emitted plans, by the sign of the plan's MEAN curvature
      (|mean κ| > 1e-6 1/m = a turn); and the same restricted to windows whose DECODED goal is
      TURN_L / TURN_R (the TACTICAL goal-setting half: does the plan's turn sign agree with the
      goal it was planned against?)
  (c) distinct plans (bit-exact over the [K, 2] block) and the fraction bit-exact to an INJECTED
      baseline: cv / hold_v0 (all zeros), decel_1.5 (a = -1.5, κ = 0); a constant-acceleration
      straight line (any a, κ = 0) is reported beside them
  (d) κ ≡ 0 fraction (max |κ| < 1e-9), acceleration-constant-in-time fraction
  plus: plan_source label histogram (for the record, with the label-vs-content gap), the
  per-window cost of every injected baseline under the run's metric (basecost_<name>_cl,
  when banked): ccos(cv) IN THE PLANNER'S OWN BATCH — the degeneracy measured, not assumed.
Cross-dump: fraction of windows whose `cl` is bit-identical between two dumps (same windows,
asserted by `ws` and `v0`).
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np

LAT_VOCAB_HINT = "manifest['goal'] indexes the v7.0 tac vocabulary; names resolved from the stack when --stack is given"


def load_dump(d):
    files = sorted(glob.glob(os.path.join(d, "ep*.npz")))
    man = json.load(open(os.path.join(d, "manifest.json"), encoding="utf-8")) if os.path.exists(os.path.join(d, "manifest.json")) else {}
    E, D = {}, {}
    for f in files:
        k = os.path.basename(f)[:-4]
        with np.load(f) as z:
            E[k] = {kk: z[kk] for kk in z.files}
        df = os.path.join(d, "decisions", os.path.basename(f))
        if os.path.exists(df):
            with np.load(df) as z:
                D[k] = {kk: z[kk] for kk in z.files}
    return man, E, D


def lat_names(stack):
    try:
        import sys
        sys.path.insert(0, stack)
        from tanitad.models import vocab_v7 as V
        for attr in ("TACTICAL_LAT_ACTIONS_V7", "LAT_TOKENS", "LAT_NAMES", "LAT_VOCAB"):
            if hasattr(V, attr):
                return list(getattr(V, attr))
    except Exception:
        pass
    return None


def shape(man, E, D, arm="cl", stack=None, lat_vocab=None):
    keys = sorted(E)
    ctrl = np.concatenate([D[k][f"{arm}_controls"] for k in keys])          # [N, K, 2]
    N, K, _ = ctrl.shape
    ws = np.concatenate([E[k]["ws"] for k in keys]); v0 = np.concatenate([E[k]["v0"] for k in keys])
    eid = np.concatenate([[k] * len(E[k]["ws"]) for k in keys])
    kmean = ctrl[..., 1].mean(1); kmax = np.abs(ctrl[..., 1]).max(1)
    a_const = np.array([np.all(c[:, 0] == c[0, 0]) for c in ctrl])
    zeros = np.array([np.all(c == 0) for c in ctrl])
    decel = np.array([np.all(c[:, 1] == 0) and np.all(c[:, 0] == np.float32(-1.5)) for c in ctrl])
    straight_const_a = np.array([np.all(c[:, 1] == 0) and np.all(c[:, 0] == c[0, 0]) for c in ctrl])
    distinct = len({c.tobytes() for c in ctrl})
    turnL = kmean > 1e-6; turnR = kmean < -1e-6; straight = ~(turnL | turnR)
    out = {"arm": arm, "n_windows": int(N), "n_episodes": int(len(keys)), "K": int(K),
           "cost": man.get("cost"),
           "b_share": {"L": float(turnL.mean()), "R": float(turnR.mean()), "straight": float(straight.mean()),
                       "n_L": int(turnL.sum()), "n_R": int(turnR.sum()), "n_straight": int(straight.sum()),
                       "rule": "sign of the plan's mean curvature; |mean kappa| <= 1e-6 = straight"},
           "c_distinct_plans": int(distinct),
           "c_bit_exact_injected_baseline": {"zeros_cv_or_hold_v0": int(zeros.sum()), "decel_1.5": int(decel.sum()),
                                             "frac_zeros_or_decel": float((zeros | decel).mean()),
                                             "straight_constant_accel_any_a": int(straight_const_a.sum()),
                                             "frac_straight_constant_accel": float(straight_const_a.mean())},
           "d_kappa_identically_zero_frac": float((kmax < 1e-9).mean()),
           "d_accel_constant_in_time_frac": float(a_const.mean()),
           "kappa_abs_max_over_plans": float(kmax.max()),
           "kappa_mean_abs_median_over_turning": float(np.median(np.abs(kmean[turnL | turnR]))) if (turnL | turnR).any() else None,
           }
    src_key = f"plan_source_{arm}"
    if all(src_key in D[k] for k in keys):
        src = np.concatenate([D[k][src_key] for k in keys])
        names = man.get("plan_source_names") or ["cem", "baseline:cv", "baseline:hold_v0", "baseline:proposal", "baseline:decel_1.5"]
        hist = {names[i]: int((src == i).sum()) for i in range(len(names)) if (src == i).any()}
        cem = src == 0
        out["plan_source_label_hist"] = hist
        out["label_vs_content"] = {"labelled_cem": int(cem.sum()),
                                   "labelled_cem_but_bit_exact_zeros_or_decel": int((cem & (zeros | decel)).sum()),
                                   "baseline_won_frac_by_label": float(1 - cem.mean()),
                                   "trivial_by_content_zeros_or_decel": float((zeros | decel).mean())}
    # goal-conditioned turn agreement (the TACTICAL goal-setting half)
    gl_key = f"goal_lat_{arm}"
    if all(gl_key in D[k] for k in keys):
        gl = np.concatenate([D[k][gl_key] for k in keys])
        vocab = lat_vocab or (lat_names(stack) if stack else None)
        by = {}
        for tok in np.unique(gl):
            m = gl == tok
            nm = vocab[int(tok)] if vocab and 0 <= int(tok) < len(vocab) else f"tok{int(tok)}"
            by[nm] = {"n": int(m.sum()), "plan_L": int((turnL & m).sum()), "plan_R": int((turnR & m).sum()),
                      "plan_straight": int((straight & m).sum())}
        out["by_decoded_goal_lat"] = by
        if vocab:
            iL = [i for i, n in enumerate(vocab) if "TURN_L" in n]; iR = [i for i, n in enumerate(vocab) if "TURN_R" in n]
            gL = np.isin(gl, iL); gR = np.isin(gl, iR)
            out["goal_turn_agreement"] = {
                "n_goal_TURN_L": int(gL.sum()), "plan_L_on_goal_L": int((gL & turnL).sum()), "plan_R_on_goal_L": int((gL & turnR).sum()),
                "n_goal_TURN_R": int(gR.sum()), "plan_R_on_goal_R": int((gR & turnR).sum()), "plan_L_on_goal_R": int((gR & turnL).sum()),
                "n_goal_turn": int((gL | gR).sum()),
                "sign_agree_frac_over_goal_turn_windows": float(((gL & turnL) | (gR & turnR)).sum() / max(1, (gL | gR).sum())),
                "plan_turns_on_goal_turn_frac": float(((gL | gR) & (turnL | turnR)).sum() / max(1, (gL | gR).sum())),
                "plan_turns_on_goal_nonturn_frac": float((~(gL | gR) & (turnL | turnR)).sum() / max(1, (~(gL | gR)).sum()))}
    # the injected baselines' own costs under the run's metric, per window (D-REFAV1-CCOS-EVAL banking)
    bc = {}
    for k in keys:
        for kk in D[k]:
            if kk.startswith("basecost_") and kk.endswith(f"_{arm}"):
                bc.setdefault(kk, []).append(D[k][kk])
    if bc:
        out["basecosts"] = {}
        for kk, v in bc.items():
            v = np.concatenate(v).astype(float)
            out["basecosts"][kk] = {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max()),
                                    "n_exactly_1": int((v == 1.0).sum()), "n_exactly_0": int((v == 0.0).sum()),
                                    "abs_dev_from_1_median": float(np.median(np.abs(v - 1.0)))}
    pc_key = f"plan_cost_{arm}"
    if all(pc_key in D[k] for k in keys):
        pcst = np.concatenate([D[k][pc_key] for k in keys]).astype(float)
        out["plan_cost"] = {"median": float(np.median(pcst)), "min": float(pcst.min()), "max": float(pcst.max())}
    return out, (ws, v0, eid, ctrl)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="append", required=True, help="name=path (repeatable)")
    ap.add_argument("--arm", default="cl")
    ap.add_argument("--stack", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res, keyed = {}, {}
    for spec in a.dump:
        name, path = spec.split("=", 1)
        man, E, D = load_dump(path)
        s, key = shape(man, E, D, a.arm, a.stack)
        s["path"] = path
        res[name] = s; keyed[name] = key
    names = list(res)
    if len(names) >= 2:
        base = names[0]; ws0, v00, e0, c0 = keyed[base]
        res["_cross"] = {}
        for nm in names[1:]:
            ws1, v01, e1, c1 = keyed[nm]
            same_grid = ws0.shape == ws1.shape and np.array_equal(ws0, ws1) and np.array_equal(v00, v01)
            n = min(len(ws0), len(ws1))
            ident = [np.array_equal(c0[i], c1[i]) for i in range(n)] if same_grid else None
            res["_cross"][f"{base}_vs_{nm}"] = {"same_windows_asserted": bool(same_grid), "n_compared": int(n),
                                                "cl_bit_identical_frac": float(np.mean(ident)) if ident is not None else None}
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1, default=float)
    for nm in names:
        s = res[nm]
        print(f"[{nm}] n={s['n_windows']}/{s['n_episodes']} metric={(s.get('cost') or {}).get('metric','cos (pre-flag)')} "
              f"weights={(s.get('cost') or {}).get('weights','shipped')}\n"
              f"   share L/R/straight = {s['b_share']['n_L']}/{s['b_share']['n_R']}/{s['b_share']['n_straight']}  "
              f"distinct={s['c_distinct_plans']}  bit-exact zeros={s['c_bit_exact_injected_baseline']['zeros_cv_or_hold_v0']} "
              f"decel={s['c_bit_exact_injected_baseline']['decel_1.5']} (frac {s['c_bit_exact_injected_baseline']['frac_zeros_or_decel']:.4f})  "
              f"kappa==0 frac={s['d_kappa_identically_zero_frac']:.4f}  a-const frac={s['d_accel_constant_in_time_frac']:.4f}")
        if "goal_turn_agreement" in s:
            g = s["goal_turn_agreement"]
            print(f"   goal-turn windows {g['n_goal_turn']}: plan turns on {g['plan_turns_on_goal_turn_frac']:.3f}, sign agrees {g['sign_agree_frac_over_goal_turn_windows']:.3f}; plan turns on non-turn goals {g['plan_turns_on_goal_nonturn_frac']:.3f}")
        if "basecosts" in s:
            for kk, v in s["basecosts"].items():
                print(f"   {kk}: median {v['median']:.6g} min {v['min']:.6g} max {v['max']:.6g} exactly1={v['n_exactly_1']} exactly0={v['n_exactly_0']}")
    if "_cross" in res:
        print(res["_cross"])


if __name__ == "__main__":
    main()
