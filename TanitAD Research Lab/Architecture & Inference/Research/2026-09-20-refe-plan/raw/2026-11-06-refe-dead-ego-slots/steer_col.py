"""Is REFe's ego slot 5 (tire_steering_angle) ever trained?  nuPlan's DB EgoState hardcodes tire_steering_angle=0.0
(nuplan_scenario_queries.py:551,594), so a Linear column fed only zeros gets EXACTLY zero gradient and keeps its init bytes.
Test: the slot-5 column is bit-identical across checkpoints taken at different training stages, while the other six columns (controls) differ."""
import hashlib, sys, torch
paths = sys.argv[1:]
def sd_of(p):
    o = torch.load(p, map_location="cpu", weights_only=False)
    for k in ("model", "state_dict", "model_state", "ema"):
        if isinstance(o, dict) and k in o and isinstance(o[k], dict) and any(torch.is_tensor(v) for v in o[k].values()):
            return o[k]
    return o
rows = {}
for p in paths:
    sd = sd_of(p)
    ks = [k for k in sd if k.endswith("ego_enc.0.weight")]
    if not ks:
        print("NO ego_enc.0.weight in", p, [k for k in sd if "ego" in k][:8]); continue
    W = sd[ks[0]].float()
    rows[p] = W
    print(p.split("/")[-1], ks[0], tuple(W.shape), "col-norms:", " ".join(f"{W[:, j].norm():.4f}" for j in range(min(9, W.shape[1]))))
ps = list(rows)
for a, b in zip(ps, ps[1:]):
    A, B = rows[a], rows[b]
    if A.shape != B.shape: print("shape differs", a, b); continue
    eq = [bool(torch.equal(A[:, j], B[:, j])) for j in range(A.shape[1])]
    mx = [float((A[:, j] - B[:, j]).abs().max()) for j in range(min(9, A.shape[1]))]
    print(f"{a.split('/')[-1]} vs {b.split('/')[-1]}: identical columns {[j for j, e in enumerate(eq) if e][:20]} of {A.shape[1]}; max|diff| cols 0-8: " + " ".join(f"{m:.2e}" for m in mx))
