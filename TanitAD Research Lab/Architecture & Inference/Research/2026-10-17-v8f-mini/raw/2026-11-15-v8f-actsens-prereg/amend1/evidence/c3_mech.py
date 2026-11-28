"""MM check: does the G-LIVE loop's last-step record explain the C3 mismatch? Three AdamW steps (lr 1e-3, as wm_only_live_check) on WMAux, gamma (100,100,100) vs (1000,100,100), same init, same batch."""
import json, sys
sys.path.insert(0, "F:/TanitAD-scratch/actsens/tree/stack/tests")
import torch
import test_v8f_actsens as T

DOSE = (1000.0, 100.0, 100.0)
UNW = ("pred/L1", "sigreg/L1", "idm/L1", "pred/L2", "sigreg/L2", "idm/L2", "pred/L3", "sigreg/L3", "idm/L3")


def steps(cfg):
    wm = T.build(cfg)
    z, a = T.batch(cfg)
    opt = torch.optim.AdamW(wm.parameters(), lr=1e-3)
    rec = []
    for s in range(3):
        opt.zero_grad(set_to_none=False)
        torch.manual_seed(s)
        out = wm.stack.loss(z, a, generator=torch.Generator().manual_seed(11))
        rec.append({k: float(v.detach()) for k, v in out.items()})
        out["total"].backward()
        opt.step()
    return rec


r0, r1 = steps(T.cfg_of()), steps(T.cfg_of(wm_gamma_levels=DOSE))
res = {}
for s in range(3):
    eq = [k for k in UNW if r0[s][k] == r1[s][k]]
    d, want = r1[s]["level/L1"] - r0[s]["level/L1"], 900.0 * r0[s]["idm/L1"]
    res["step%d" % s] = {"unweighted_equal": len(eq), "of": len(UNW), "pred_L2": [r0[s]["pred/L2"], r1[s]["pred/L2"]], "level_L1_diff": d, "900_idm_L1": want,
                         "identity_holds_1e-5": abs(d - want) <= 1e-5 * abs(want)}
print(json.dumps(res, indent=1))
json.dump(res, open("F:/TanitAD-scratch/actsens/mmcheck/c3_mech.json", "w", encoding="utf-8"), indent=1)
