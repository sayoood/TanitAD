#!/usr/bin/env python3
"""Within-token discrimination of each scorer head (descriptive attribution for the SCORER-FT design; no bar, no verdict).

For each head h in (NC, DAC, EP, TTC, C) and each token whose 64 proposals carry BOTH outcomes of the binary sub-score (NC/DAC/TTC/C),
AUC_h = P(p_h(passing proposal) > p_h(failing proposal)) within the token (ties 0.5); averaged over tokens, log-cluster bootstrap CI.
EP: within-token Spearman between p_EP and the tracked EP over the proposals with NC = DAC = 1.
Also: on tokens where the pick is ZEROED (NC or DAC = 0) and a passing proposal exists, the share where the pick's p_h is above the
median p_h of the passing proposals (the head was confidently wrong). T1 tracked table; head order per agg32 = logits[..., 0..4] = NC, DAC, EP, TTC, C.
Usage: head_auc.py --arm C1=<rows glob>=<npz> [--arm ...] --out OUT.json
"""
import argparse, json
import numpy as np
import s_rerank as A

HEADS = {"nc": 0, "dac": 1, "ep": 2, "ttc": 3, "c": 4}


def auc_within(s, y):
    pos, neg = s[y > 0.5], s[y < 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    d = pos[:, None] - neg[None, :]
    return float((d > 0).mean() + 0.5 * (d == 0).mean())


def spearman(a, b):
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def bootci(v, logs, B=2000, seed=20260927):
    m = ~np.isnan(v); v, logs = v[m], logs[m]
    ul = sorted(set(logs)); ix = [np.where(logs == l)[0] for l in ul]
    rng = np.random.default_rng(seed); bs = [v[np.concatenate([ix[i] for i in rng.integers(0, len(ul), len(ul))])].mean() for _ in range(B)]
    return {"mean": float(v.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)), "n_tokens": int(len(v)), "n_logs": len(ul)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", action="append", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); out = {}
    for spec in a.arm:
        d = A.load(spec)
        import glob
        name, g, npz = spec.split("=", 2)
        P = 1.0 / (1.0 + np.exp(-np.load(npz)["logits"].astype(np.float32)))
        tokidx = {t: i for i, t in enumerate(np.load(npz)["token"].astype(str))}
        I = np.array([tokidx[t] for t in d["tok"]]); P = P[I]
        n = len(d["tok"]); r = {}
        for h, k in HEADS.items():
            y = d["tr"][h]
            if h == "ep":
                ok = (d["tr"]["nc"] > 0.5) & (d["tr"]["dac"] > 0.5)
                v = np.array([spearman(P[i, ok[i], k], y[i, ok[i]]) if ok[i].sum() >= 3 else np.nan for i in range(n)])
                r[h] = {"within_token_spearman": bootci(v, d["logs"])}
                continue
            yb = (y > 0.5).astype(float)
            v = np.array([auc_within(P[i, :, k], yb[i]) for i in range(n)])
            mixed = ~np.isnan(v)
            r[h] = {"within_token_auc": bootci(v, d["logs"]), "share_tokens_mixed": float(mixed.mean())}
        ar = np.arange(n); pk = d["pick"]
        z = (d["tr"]["nc"][ar, pk] < 0.5) | (d["tr"]["dac"][ar, pk] < 0.5)
        resc = z & (d["tr"]["pdms"].max(1) > 0)
        conf = {}
        for h in ("nc", "dac"):
            k = HEADS[h]; vals = []
            for i in np.where(resc & (d["tr"][h][ar, pk] < 0.5))[0]:
                good = d["tr"][h][i] > 0.5
                vals.append(float(P[i, pk[i], k] > np.median(P[i, good, k])))
            conf[h] = {"n_pick_failed_this_head_rescuable": len(vals), "share_pick_head_above_passing_median": float(np.mean(vals)) if vals else None}
        r["zeroed_pick_confidently_wrong"] = conf
        out[d["name"]] = r
        print("ZZHEADAUC", d["name"], {h: round(r[h].get("within_token_auc", r[h].get("within_token_spearman"))["mean"], 3) for h in HEADS}, conf)
    out["note"] = "descriptive attribution, T1 tracked per-proposal tables, log-cluster bootstrap 2,000 (another draw of LOGS); no bar"
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
