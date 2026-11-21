"""Side observation (NOT part of the fix): the trainer seeds the GLOBAL RNG with the SAME formula every step (`torch.manual_seed(a.seed * 1_000_003 + step)`, v8f_mini_train.py ~1832), and the model's teacher-forcing
draws (`torch.rand(B) < p_gt`, v8f_mini.py:741/754) read that global RNG. If --seed equals --dagger-seed, are those draws related to the swap's selection uniforms? CPU, torch only."""
import numpy as np
import torch

B = 342
rows = []
for s, t in [(0, 0), (0, 7), (5, 1), (5, 100), (3, 1000)]:
    seed = (s * 1_000_003 + t) % (1 << 62)
    u = torch.rand(2, B, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)[0]        # the swap's / the perturbation's selection uniform (unseparated)
    torch.manual_seed(seed)
    v = torch.rand(B)                                                                                  # the first global float32 draw after the trainer's per-step seeding
    v64 = torch.rand(B, dtype=torch.float64) if False else None
    corr = float(np.corrcoef(u.numpy(), v.numpy())[0, 1])
    eq = bool(torch.equal(u.float(), v))
    sel = u < 0.1
    rows.append((s, t, round(corr, 4), eq, round(float((v[sel] < 0.5).float().mean()), 3), int(sel.sum())))
print("seed step corr(u,v) equal  P(v<0.5 | u<0.1)  n(u<0.1)")
for r in rows:
    print(*r)
# the float64 global draw WOULD be identical:
torch.manual_seed(7)
a = torch.rand(2, B, dtype=torch.float64)[0]
b = torch.rand(2, B, generator=torch.Generator().manual_seed(7), dtype=torch.float64)[0]
print("global float64 rand(2,B)[0] == private-generator float64 rand(2,B)[0]:", bool(torch.equal(a, b)))
