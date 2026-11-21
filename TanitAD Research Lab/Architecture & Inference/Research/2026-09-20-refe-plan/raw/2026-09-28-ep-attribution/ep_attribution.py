"""EXPLORATORY (the 200 SELECTION tokens -- never a claim): which scorer head costs the pick the most PDMS?

For each snapshot's E-6 table (all 64 proposals scored by the harness), re-pick with the shipped rule (navsim_v1:
NC*DAC*(5EP+5TTC+2C)/12 over sigmoid head outputs) after replacing ONE head's prediction by the TRUE harness sub-score.
The PDMS of that pick minus the shipped pick is the head's share of the selection gap (an oracle-substitution
attribution; it ranks levers, it does not estimate what a trainable fix would recover).

Also tests route-free progress proxies in place of the EP head: the endpoint's forward displacement divided by a
per-scene reference (max, or a quantile q of the 64), clipped to [0, 1]. q is FISHED here, on selection tokens.

    python ep_attribution.py [--out ep_attribution.json]
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import spearmanr

PT = "D:/Projects/TanitAD/data/refe_navtest/proptable"
SUBS = ["NC", "DAC", "EP", "TTC", "C", "DDC"]      # table `sub` order == sub_names (no_at_fault ... ddc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snaps", default="015,016,017,018")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ep_attribution.json"))
    a = ap.parse_args()
    out = {"_label": "EXPLORATORY: selection tokens (sub200); oracle-substitution attribution + progress proxies; "
                     "never a claim without a registered fresh-token confirmation", "snapshots": {}}
    for ep in a.snaps.split(","):
        T = np.load(os.path.join(PT, f"sub200_ep{ep}", "table.npz"), allow_pickle=True)
        assert [str(x) for x in T["head_order"]] == SUBS and str(T["rule"]) == "navsim_v1", (T["head_order"], T["rule"])
        P, S, V = T["pdms"], T["sub"], T["valid"].astype(bool)
        L, pk, X = T["logits"].astype(np.float64), T["pick"], T["proposals"]
        n = P.shape[0]
        ar = np.arange(n)
        sig = 1 / (1 + np.exp(-L))
        g = {k: sig[..., SUBS.index(k)] for k in SUBS}
        true = {k: S[..., j] for j, k in enumerate(SUBS)}

        def pick(h):
            agg = h["NC"] * h["DAC"] * (5 * h["EP"] + 5 * h["TTC"] + 2 * h["C"]) / 12
            return float(P[ar, np.where(V, agg, -np.inf).argmax(1)].mean() * 100)

        def with_true(keys):
            return pick({k: (true[k] if k in keys else g[k]) for k in SUBS})

        r = {"n_tokens": int(n), "shipped_pick": float(P[ar, pk].mean() * 100), "rule_recomputed": pick(g),
             "best": float(np.where(V, P, -1).max(1).mean() * 100),
             "random": float(np.nanmean(np.where(V, P, np.nan)) * 100)}
        assert abs(r["rule_recomputed"] - r["shipped_pick"]) < 1e-6, "the recomputed rule must reproduce the shipped pick"
        r["true_head"] = {k: with_true((k,)) for k in ("NC", "DAC", "EP", "TTC", "C")}
        r["true_NC+DAC"] = with_true(("NC", "DAC"))
        r["true_all5"] = with_true(("NC", "DAC", "EP", "TTC", "C"))
        rho = [spearmanr(g["EP"][i][V[i]], true["EP"][i][V[i]]).correlation
               for i in range(n) if V[i].sum() > 2 and np.ptp(true["EP"][i][V[i]]) > 0]
        r["EP_head_within_scene_spearman_median"] = float(np.nanmedian(rho))
        r["frac_candidates_trueEP_eq_1"] = float(np.mean(true["EP"][V] >= 0.999))
        xe = np.where(V, X[:, :, -1, 0], np.nan)
        r["proxy_forward_displacement"] = {}
        for q in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            ref = np.nanquantile(xe, q, axis=1, keepdims=True)
            prox = np.clip(np.nan_to_num(xe, nan=0.0) / np.maximum(ref, 1e-3), 0, 1)
            r["proxy_forward_displacement"][f"q{q}"] = pick({**g, "EP": prox})
        out["snapshots"][ep] = r
        print(ep, json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()
                              if not isinstance(v, dict)}), "\n   true_head", {k: round(v, 2) for k, v in r["true_head"].items()},
              "\n   proxy", {k: round(v, 2) for k, v in r["proxy_forward_displacement"].items()})
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
