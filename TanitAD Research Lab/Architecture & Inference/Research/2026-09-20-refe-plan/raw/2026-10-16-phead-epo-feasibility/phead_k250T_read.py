#!/usr/bin/env python3
"""K = 250 closed-loop follow-up (RESULT.md item 3): does the sim's FULL arm reproduce R0's TTC loss once it runs to K = 250 in regime T, and
does LAD_2 pay it too?  Reads closed_out/closed_s0_T_{FULL,LAD_2}_R_lr1_lam{0,1}k250T.pkl; B = lam 1 vs its own lam-0 control, at snaps
50 / 100 / 250; TTC / NC / DAC under P1 (pick) and P2 (policy expectation), pooled over train+val with phead_summarize.prox's estimator.
Writes k250T_read.json.  Reference (validated proxy, banked R0 states): P2 TTC B - A x100 -1.53 / -1.25 / -1.07 at u250 / u500 / u1000."""
import json, os, pickle, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import phead_summarize as S

out = {"what": __doc__.splitlines()[0], "rows": {}}
for g in ("FULL", "LAD_2"):
    r1 = pickle.load(open(os.path.join(HERE, "closed_out", f"closed_s0_T_{g}_R_lr1_lam1k250T.pkl"), "rb"))
    r0 = pickle.load(open(os.path.join(HERE, "closed_out", f"closed_s0_T_{g}_R_lr1_lam0k250T.pkl"), "rb"))
    for k in (50, 100, 250):
        pr = S.prox(r1["snaps"][k]["eval"], r0["snaps"][k]["eval"])
        out["rows"][f"{g}@{k}"] = {m: {"pooled": round(v["pooled"], 4), "ci": [round(c, 4) for c in v["pooled_ci"]],
                                        "train": round(v["train"], 4), "val": round(v["val"], 4)} for m, v in pr.items()}
        out["rows"][f"{g}@{k}"]["outside"] = [r1["snaps"][k]["outside"], r0["snaps"][k]["outside"]]
        t = out["rows"][f"{g}@{k}"]
        print(f"{g:6s} K={k:3d}  P2 TTC {t['P2_TTC']['pooled']:+.2f} [{t['P2_TTC']['ci'][0]:+.2f}, {t['P2_TTC']['ci'][1]:+.2f}]  "
              f"P1 TTC {t['P1_TTC']['pooled']:+.2f}  P2 NC {t['P2_NC']['pooled']:+.2f}  P2 DAC {t['P2_DAC']['pooled']:+.2f}", flush=True)
json.dump(out, open(os.path.join(HERE, "k250T_read.json"), "w"), indent=1)
print("ZZK250T_READ_OK")
