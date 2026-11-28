#!/usr/bin/env python3
"""S-RULE read (PREREG_P5C_NEXT.md section 3): choose a selection rule on TUNE logs, read it on EVAL logs, replicate across checkpoints.

Split (fixed in the prereg): TUNE = logs whose sha1(log_name) hex digest ends in an even hex digit; EVAL = the rest.
Rules: agg(a, w) = (p_NC p_DAC)^a (5 p_EP + w p_TTC + 2 p_C) / (7 + w), a in {1,2,4,8}, w in {5,10}; (1,5) = navsim_v1 (the pick).
Choice: max TUNE mean tracked PDMS; ties -> smaller a, then smaller w. Read: delta = rule - pick on EVAL, paired log-cluster bootstrap,
10,000 draws, seed 20260927, 95 % percentile (another draw of LOGS only). Sub-scores NC, DAC, TTC, EP, C: separated adverse = CI-hi < 0.
ADOPT iff CI-lo(delta) > 0 on the primary arm AND no sub-score separated adverse AND delta point > 0 on EVAL of every replicate arm.
CONTROL: rule (1,5) must reproduce the dump's pick on every token of every arm (else exit 2).
Usage: s_rule_read.py --primary C1=<rows glob>=<proposals.npz> [--replicate C0=<glob>=<npz> ...] --out OUT.json [--boot 10000] [--smoke]"""
import argparse, glob, hashlib, json, sys
import numpy as np

RULES = [(a, w) for a in (1.0, 2.0, 4.0, 8.0) for w in (5.0, 10.0)]
SUBS = ("nc", "dac", "ttc", "ep", "c")


def agg(L, a, w):
    p = 1.0 / (1.0 + np.exp(-L))
    return (p[..., 0] * p[..., 1]) ** a * (5 * p[..., 2] + w * p[..., 3] + 2 * p[..., 4]) / (7 + w)


def load(spec):
    name, g, npz = spec.split("=", 2)
    R = {}
    for f in sorted(glob.glob(g)):
        for ln in open(f, encoding="utf-8"):
            r = json.loads(ln)
            if "error" not in r:
                R[r["token"]] = r
    toks = sorted(R)
    z = np.load(npz, allow_pickle=False)
    zi = {str(t): i for i, t in enumerate(z["token"])}
    I = np.array([zi[t] for t in toks])
    d = {"name": name, "toks": toks, "logs": np.array([R[t]["log"] for t in toks]), "L": z["logits"][I].astype(np.float64),
         "pick": z["pick"][I], "tr": {k: np.array([R[t]["tr"][k] for t in toks], np.float64) for k in SUBS + ("pdms",)}}
    d["tune"] = np.array([int(hashlib.sha1(l.encode()).hexdigest()[-1], 16) % 2 == 0 for l in d["logs"]])
    d["sel"] = {r: agg(d["L"], *r).argmax(1) for r in RULES}
    return d


def boot_delta(d, m, sel, B, rng_seed=20260927):
    n = m.sum()
    ar = np.arange(len(d["toks"]))
    logs = d["logs"][m]
    ul = sorted(set(logs))
    ix = [np.where(logs == l)[0] for l in ul]
    out = {}
    for k in ("pdms",) + SUBS:
        x = (d["tr"][k][ar, sel] - d["tr"][k][ar, d["pick"]])[m] * 100
        rng = np.random.default_rng(rng_seed)
        bs = []
        for _ in range(B):
            s = np.concatenate([ix[i] for i in rng.integers(0, len(ul), len(ul))])
            bs.append(x[s].mean())
        out[k] = {"point": float(x.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}
    out["n_tokens"], out["n_logs"] = int(n), len(ul)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", required=True)
    ap.add_argument("--replicate", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    P = load(a.primary)
    reps = [load(s) for s in a.replicate]
    ctrl = {d["name"]: bool((d["sel"][(1.0, 5.0)] == d["pick"]).all()) for d in [P] + reps}
    ar = np.arange(len(P["toks"]))
    tune_means = {f"a{r[0]:g}_w{r[1]:g}": float((P["tr"]["pdms"][ar, P["sel"][r]][P["tune"]] * 100).mean()) for r in RULES}
    best = max(RULES, key=lambda r: (round(tune_means[f"a{r[0]:g}_w{r[1]:g}"], 12), -r[0], -r[1]))
    ev = ~P["tune"]
    prim = boot_delta(P, ev, P["sel"][best], a.boot)
    rep = {}
    for d in reps:
        dar = np.arange(len(d["toks"]))
        x = (d["tr"]["pdms"][dar, d["sel"][best]] - d["tr"]["pdms"][dar, d["pick"]])[~d["tune"]] * 100
        rep[d["name"]] = {"delta_point_eval": float(x.mean()), "n_eval_tokens": int((~d["tune"]).sum())}
    adverse = [k for k in SUBS if prim[k]["hi"] < 0]
    adopt = prim["pdms"]["lo"] > 0 and not adverse and all(v["delta_point_eval"] > 0 for v in rep.values()) and len(rep) >= 2
    out = {"card": "PREREG_P5C_NEXT.md section 3", "smoke": bool(a.smoke), "controls_rule_1_5_reproduces_pick": ctrl,
           "split": {"tune_tokens": int(P["tune"].sum()), "eval_tokens": int(ev.sum()), "tune_logs": len(set(P["logs"][P["tune"]])),
                     "eval_logs": len(set(P["logs"][ev]))},
           "tune_mean_pdms_by_rule": tune_means, "chosen_rule": {"a": best[0], "w": best[1]},
           "eval_primary": prim, "eval_separated_adverse": adverse, "replicates": rep,
           "verdict": ("SMOKE (no verdict)" if a.smoke else ("ADOPT" if adopt else "NOT ADOPTED")),
           "estimator": f"paired log-cluster bootstrap over EVAL logs, {a.boot} draws, seed 20260927, 95 % percentile; LOGS only",
           "tier": "T1 executed (NAVSIM v1.1 LQR-tracked)", "evidence": "MEASURED (this read)"}
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps(out, indent=1))
    return 0 if all(ctrl.values()) else 2


if __name__ == "__main__":
    sys.exit(main())
