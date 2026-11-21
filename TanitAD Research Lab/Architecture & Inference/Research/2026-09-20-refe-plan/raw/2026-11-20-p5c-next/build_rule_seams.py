#!/usr/bin/env python3
"""S-RULE families input (PREREG_P5C_NEXT.md §3 "Four families, on the rule's picks ... same EVAL logs"; AMENDMENT_1 float32).

Writes two EVAL-only seams in the 8-pose form (no pick20, so both arms are like-for-like):
  C1pick  = the dump's pick (navsim_v1)                 poses = proposals[pick]   (== the harness seam, asserted)
  C1rule  = the frozen S-RULE choice agg(a=2, w=5)      poses = proposals[argmax agg_f32]
and selection_quality_eval.json: per arm on the EVAL tokens, from the 64-proposal tracked table (T1, LQR-tracked), the TACTICAL
selection-quality read that P5-C reported NOT COMPUTED: regret = oracle - pick PDMS x100, top-1 rate (pick attains the oracle),
share of tokens whose pick is zeroed while a proposal > 0 exists.  Paired log-cluster bootstrap, 10,000 draws, seed 20260927.
Usage: build_rule_seams.py --rows "propscore/full_C1_s*of2.jsonl" --proposals <c1 .proposals.npz> --seam <c1 seam.npz> --outdir srule_fam
"""
import argparse, glob, hashlib, json, os
import numpy as np


def agg32(L, a, w):
    L = np.asarray(L, np.float32)
    p = 1.0 / (1.0 + np.exp(-L))
    return (p[..., 0] * p[..., 1]) ** a * (5 * p[..., 2] + w * p[..., 3] + 2 * p[..., 4]) / (7 + w)


def boot(x, logs, B=10000, seed=20260927):
    ul = sorted(set(logs)); ix = [np.where(logs == l)[0] for l in ul]
    rng = np.random.default_rng(seed); bs = []
    for _ in range(B):
        s = np.concatenate([ix[i] for i in rng.integers(0, len(ul), len(ul))]); bs.append(x[s].mean())
    return {"point": float(x.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)), "n_tokens": int(len(x)),
            "n_logs": len(ul)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True); ap.add_argument("--proposals", required=True); ap.add_argument("--seam", required=True)
    ap.add_argument("--outdir", required=True); ap.add_argument("--a", type=float, default=2.0); ap.add_argument("--w", type=float, default=5.0)
    a = ap.parse_args()
    R = {}
    for f in sorted(glob.glob(a.rows)):
        for ln in open(f, encoding="utf-8"):
            r = json.loads(ln)
            if "error" not in r:
                R[r["token"]] = r
    z = np.load(a.proposals, allow_pickle=False); s = np.load(a.seam, allow_pickle=False)
    tok = z["token"].astype(str)
    assert (s["token"].astype(str) == tok).all(), "seam / dump token order differs"
    n = len(tok); ar = np.arange(n)
    assert np.array_equal(z["proposals"][ar, z["pick"]], s["poses"]), "seam poses != proposals[pick]"
    pick = z["pick"]; L = z["logits"]
    assert (agg32(L, 1.0, 5.0).argmax(1) == pick).all(), "CTRL-S (float32) failed"
    rule = agg32(L, a.a, a.w).argmax(1)
    logs = np.array([R[t]["log"] for t in tok])
    ev = np.array([int(hashlib.sha1(l.encode()).hexdigest()[-1], 16) % 2 == 1 for l in logs])      # EVAL = odd last hex digit
    os.makedirs(a.outdir, exist_ok=True)
    E = np.where(ev)[0]
    for nm, sel in (("C1pick", pick), ("C1rule", rule)):
        np.savez(os.path.join(a.outdir, f"seam_{nm}_eval.npz"), token=tok[E], fingerprint=s["fingerprint"][E],
                 poses=z["proposals"][E, sel[E]], sampling=s["sampling"], arm=np.array(nm))
    tr = np.array([R[t]["tr"]["pdms"] for t in tok], np.float64)                                      # [n, 64] tracked PDMS
    orc = tr.max(1); q = {}
    for nm, sel in (("C1pick", pick), ("C1rule", rule)):
        v = tr[ar, sel]
        q[nm] = {"pdms_x100": boot(v[E] * 100, logs[E]), "regret_x100": boot((orc - v)[E] * 100, logs[E]),
                 "top1_rate": boot((v >= orc - 1e-12)[E].astype(float), logs[E]),
                 "zeroed_with_rescue_share": boot(((v == 0) & (orc > 0))[E].astype(float), logs[E])}
    d = (tr[ar, rule] - tr[ar, pick])[E] * 100
    out = {"tier": "T1 executed (NAVSIM v1.1 LQR-tracked), per-proposal table", "estimator": "paired log-cluster bootstrap, EVAL logs, 10000, seed 20260927",
           "rule": {"a": a.a, "w": a.w}, "n_eval_tokens": int(ev.sum()), "n_eval_logs": len(set(logs[E])),
           "n_rule_differs_from_pick_eval": int((rule != pick)[E].sum()), "selection_quality": q,
           "rule_minus_pick_pdms_x100": boot(d, logs[E]), "oracle_pdms_x100_eval": float(orc[E].mean() * 100)}
    json.dump(out, open(os.path.join(a.outdir, "selection_quality_eval.json"), "w", encoding="utf-8"), indent=1)
    print("ZZRULESEAM_OK eval", int(ev.sum()), "differs", out["n_rule_differs_from_pick_eval"], "delta", out["rule_minus_pick_pdms_x100"])


if __name__ == "__main__":
    main()
