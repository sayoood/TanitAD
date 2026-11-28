import numpy as np, sys
z = dict(np.load("banked/eval_M2b_s0.npz"))
z["ade6"] = z["ade6"].copy(); z["ade6"][17] += 2e-3          # one window off by 2 mm: the identity gate must catch it
np.savez("dry/fail_case/eval_M2b_s0-idcheck.npz", **{k: z[k] for k in ("ade6", "fde6", "ade2", "eid")})
