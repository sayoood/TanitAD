"""MM 2026-11-27 DESCRIPTIVE diagnostic (not a decision, not a card): does SH's failure come from re-scaled probabilities
reshuffling C1's near-tied picks?  Zero training: the gated SH heads from run1 (63a4af88 PASS tokens) are applied to the
navtrain cache's `s`.  Rules: C1 (argmax agg32 of C1 logits), SH (rows 0,1,3,5 replaced, the trainer's rule; control:
must reproduce heldout_select.json), and VETO(tau) = keep C1's pick unless SH's q = sig(NC)sig(DAC)sig(TTC) of that pick
is < tau; then argmax C1-agg over {k : q_k >= tau}; empty -> keep.  Reported per split (train = where SH was fitted: optimistic)."""
import csv, json, sys
import numpy as np

RUN = "E:/Projects/TanitAD-artifacts/safehead/run1"
IDX = "F:/TanitAD_safehead_tmp/adapted/index.csv"
sig = lambda x: (1.0 / (1.0 + np.exp(-x.astype(np.float32)))).astype(np.float32)


def agg(L):  # L [n,64,6] logits, order NC,DAC,EP,TTC,C,DDC; float32 navsim_v1 aggregate
    p = sig(L)
    return p[..., 0] * p[..., 1] * (np.float32(5) * p[..., 2] + np.float32(5) * p[..., 3] + np.float32(2) * p[..., 4]) / np.float32(12)


rows = list(csv.DictReader(open(IDX)))
files = {}
for r in rows:
    files.setdefault(r["file"], []).append(r)
data = {"train": [], "heldout": []}
for f, rs in files.items():
    z = np.load(f)
    split = rs[0]["split"]
    tok = [str(t) for t in z["token"]]
    want = {r["token"] for r in rs}
    ix = [i for i, t in enumerate(tok) if t in want]
    data[split].append({k: np.array(z[k][ix]) for k in ("s", "logits", "pick", "tr_nc", "tr_dac", "tr_ttc", "tr_ep", "tr_pdms")} | {"log": rs[0]["log"]})


def cat(split, k):
    return np.concatenate([d[k] for d in data[split]])


heads = {a: np.load(f"{RUN}/sh_{a}/head_step02000.npz") for a in ("s0", "s1", "red")}
out = {}
for split in ("train", "heldout"):
    s, L, pick = cat(split, "s"), cat(split, "logits"), cat(split, "pick")
    lab = {k: cat(split, "tr_" + k) for k in ("nc", "dac", "ttc", "ep", "pdms")}
    n = len(pick)
    safe = (lab["nc"] >= 1) & (lab["dac"] >= 1) & (lab["ttc"] >= 1)
    anysafe = safe.any(1)
    ar = np.arange(n)

    def score(k):
        miss = (~safe[ar, k]) & anysafe
        return {"miss_rate": float(miss.mean()), "misses": int(miss.sum()), "pdms": float(100 * lab["pdms"][ar, k].mean()),
                "ep": float(100 * lab["ep"][ar, k].mean()), "switch": float((k != pick).mean())}

    a1 = agg(L)
    k1 = a1.argmax(1)
    res = {"n": n, "C1_pick_agrees_stored": float((k1 == pick).mean()), "C1": score(k1)}
    for a, h in heads.items():
        W, b = h["weight"], h["bias"]
        Ls = (s @ W.T + b).astype(np.float32)
        Lmix = L.copy()
        for r in (0, 1, 3, 5):
            Lmix[..., r] = Ls[..., r]
        res[f"SH_{a}"] = score(agg(Lmix).argmax(1))
        ps = sig(Ls)
        q = ps[..., 0] * ps[..., 1] * ps[..., 3]
        for tau in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95):
            ok = q >= np.float32(tau)
            keep = ok[ar, k1] | ~ok.any(1)
            alt = np.where(ok, a1, -np.inf).argmax(1)
            res[f"VETO_{a}_tau{tau}"] = score(np.where(keep, k1, alt))
    out[split] = res
json.dump(out, open(sys.argv[1], "w"), indent=1)
for split, res in out.items():
    print(split, "n", res["n"], "agree", res["C1_pick_agrees_stored"])
    for k, v in res.items():
        if isinstance(v, dict):
            print(f"  {k:22s} miss {v['misses']:4d} {v['miss_rate']:.4f}  pdms {v['pdms']:.3f}  ep {v['ep']:.2f}  switch {v['switch']:.3f}")
