"""WP-Q2 - an interval on CON_rank itself, which WP-Q left as a point estimate.

WP-Q gave paired intervals to ADE and to the real-vs-cross gap, but "consistency
becomes measurable at beta = X" was still a POINT ESTIMATE compared against an
ARBITRARY threshold (0.9 x chance). That is the same shape as the control gates
this campaign has already had to fix twice: a decision rule pulled from a
remembered constant rather than from the statistic's own distribution.

Here CON_rank gets a paired episode-cluster bootstrap against its OWN chance
value, so "measurable" means an interval that excludes chance.
"""
import io as _io, json, sys
import numpy as np, torch
sys.path.insert(0, r"C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/ff19d6ac-620c-4636-a8e4-d5402c796859/scratchpad/localstack")   # local copy: the G: mount cannot be relied on mid-run
from tanitad.instruments.cot_influence import consistency, rerank_logits

O = r"C:/Users/Admin/tanitad-data/rl-pilot"
d = torch.load(O + "/wpn_port_tensors.pt", map_location="cpu", weights_only=False)
S0, S1, ADE, EPS = d["S0"], d["S1"], d["ADE"], d["EPS"]
E = np.array(EPS); UE = sorted(set(EPS)); N, K = S0.shape
energy = -(S1 - S0); IX = {e: np.where(E == e)[0] for e in UE}
chance = (K + 1) / 2.0
sd_s0 = float(S0.std(dim=1).median()); sd_e = float(energy.std(dim=1).median())

def con_per_row(beta):
    sel = (rerank_logits(S0, energy, beta) if beta else S0).argmax(-1)
    b = torch.arange(N)
    e_sel = energy[b, sel]
    return (energy <= e_sel.unsqueeze(1)).sum(1).float().numpy()   # pessimistic ties

print("%6s%9s%10s%26s" % ("beta", "auth%", "CONrank", "CI95 vs chance %.1f" % chance))
rows = []
for beta in [0.0, 0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]:
    v = con_per_row(beta)
    rr = np.random.default_rng(0); out = []
    for _ in range(4000):
        pick = np.concatenate([IX[UE[i]] for i in rr.integers(0, len(UE), len(UE))])
        out.append(float(v[pick].mean()))
    o = np.array(out); lo, hi = np.percentile(o, 2.5), np.percentile(o, 97.5)
    sep = hi < chance or lo > chance
    rows.append({"beta": beta, "auth": 100 * beta * sd_e / sd_s0,
                 "con": float(v.mean()), "ci": [float(lo), float(hi)],
                 "separated_from_chance": bool(sep)})
    print("%6.2f%9.1f%10.2f   [%7.2f,%7.2f] %s"
          % (beta, 100 * beta * sd_e / sd_s0, v.mean(), lo, hi,
             "SEPARATED" if sep else "overlaps chance"))
first = next((x for x in rows if x["separated_from_chance"] and x["con"] < chance), None)
print("\n  CONSISTENCY IS SEPARATED FROM CHANCE from beta = %s"
      % ("%.2f (auth %.1f %%)" % (first["beta"], first["auth"]) if first else "never"))
json.dump({"chance": chance, "rows": rows, "first_separated": first,
           "_tier": "T0; NON-PARITY pilot; paired episode-cluster bootstrap 4000 it"},
          _io.open(O + "/wpq2_con_ci.json", "w", encoding="utf-8"), indent=1)
print("-> wpq2_con_ci.json")
