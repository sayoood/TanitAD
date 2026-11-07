"""mutant_real_u_sign: on REAL models and REAL TRAIN windows (192, every 104th of the fit set; no eval-141 window), the correct edit and the wrong-u-sign MUTANT go through the same read-1 code path (wsp_edit.read1_verdict):
the mutant must FAIL read 1. The correct edit's train-side result is reported as information (not a pin criterion). usage: mutant_real_u_sign.py -> raw/MUTANT_REAL_U_SIGN.json"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("\\", "/"))
import numpy as np, torch
import wsp_common as W, wsp_measure as WM, wsp_edit as E
import fit_and_write as FW
t0 = time.time()
R, _ = FW.load_train_measure()
T = W.load_tool(); M = T.M
out = {"n_windows": 192, "seeds": {}}
CKS = {s_: torch.load(W.CKPT_C0[s_], map_location="cpu", weights_only=False) for s_ in (0, 1)}      # before Rig patches torch.load (mmap, weights_only)
rigs = None
for s in (0, 1):
    ck = CKS[s]
    sd, cfg = ck["model"], ck["cfg"]; nb = cfg["nb"]
    r_res = E.readout(sd["basis"].double().numpy(), cfg["res_scale_m"])
    nm = f"s{s}"
    models = {}
    for tag, sign in (("C0", None), ("good", 1.0), ("wrong_sign", -1.0)):
        if sign is None:
            m, c, _ = W.load_model(M, ck=ck)
        else:
            f = E.fit_edit(R[f"{nm}.h2"], R["kin3"], R[f"{nm}.gv"], R[f"{nm}.ga"], R["gb"], r_res, 1.0, u_sign=sign)
            ck2 = dict(ck); ck2["model"] = E.apply_edit(sd, f["dW"], f["db"], nb)
            m, c, _ = W.load_model(M, ck=ck2)
        models[tag] = (m, c)
    if rigs is None:
        rigs = WM.Rig("train", models["C0"][1], nt=8)
    wins = rigs.train_windows(W.HERE + "/raw/train_window_ids.txt")
    sub = wins[::104][:192]
    r = WM.run_measure(rigs, models, sub, bs=64)
    gC = (np.median(r["C0.gv"]), np.median(r["C0.ga"])); gb = (np.median(r["gb"][:, 0]), np.median(r["gb"][:, 1]))
    res = {"C0_gain": [float(x) for x in gC], "base_gain": [float(x) for x in gb]}
    for tag in ("good", "wrong_sign"):
        ge = (np.median(r[f"{tag}.gv"]), np.median(r[f"{tag}.ga"]))
        ok, rows = E.read1_verdict(ge, gC, gb, 1.0)
        res[tag] = {"read1_pass": bool(ok), "rows": rows}
    out["seeds"][nm] = res
    print(nm, "good pass", res["good"]["read1_pass"], [(x["gain"], round(x["edited"], 3), round(x["target"], 3)) for x in res["good"]["rows"]], "| wrong-sign pass", res["wrong_sign"]["read1_pass"], [(x["gain"], round(x["edited"], 3)) for x in res["wrong_sign"]["rows"]], flush=True)
out["MUTANT_FAILS_READ1_BOTH_SEEDS"] = bool(all(not v["wrong_sign"]["read1_pass"] for v in out["seeds"].values()))
out["wall_s"] = time.time() - t0
json.dump(out, open(W.HERE + "/raw/MUTANT_REAL_U_SIGN.json", "w"), indent=1)
print("MUTANT_FAILS_READ1_BOTH_SEEDS", out["MUTANT_FAILS_READ1_BOTH_SEEDS"], f"{out['wall_s']:.0f}s")
