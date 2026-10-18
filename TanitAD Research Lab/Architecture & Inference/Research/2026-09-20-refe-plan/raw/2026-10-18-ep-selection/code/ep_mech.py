#!/usr/bin/env python3
"""EP-1 step 3b (fit table only): WHY did C1 and C2 not pass?  Exact sequential decomposition of the paired dPDMS of the already-read picks
(gate first, then the weighted terms), and the predicted-vs-realised movement of the changed picks.  Descriptive; evaluates no new rule.
"""
import json

import numpy as np

from ep_common import FIT, FIT_SHA, NC, DAC, EP, TTC, C, load, at, paired, Boot
from ep_rules import heads, pick_base, pick_c1, pick_c2

t = load(FIT, FIT_SHA[:16])
N = len(t["token"])
sub, pdms = t["sub"], t["pdms"]
p, g, w, L = heads(t)
boot = Boot(t["log_idx"], t["n_logs"])
b = pick_base(p, g, w)
fit = json.load(open("../raw/fit_read.json"))
rules = {"C1_XPDMS": pick_c1(p, g, w), "C2_BAND_delta0.02": pick_c2(p, g, w, L, 0.02)}
assert fit["candidates"]["C2_BAND"]["theta_star"] == 0.02
out = {}
for nm, c in rules.items():
    gate_b = at(sub[..., NC], b) * at(sub[..., DAC], b)
    gate_c = at(sub[..., NC], c) * at(sub[..., DAC], c)
    wb = [at(sub[..., k], b) for k in (EP, TTC, C)]
    wc = [at(sub[..., k], c) for k in (EP, TTC, C)]
    W = np.array([5.0, 5.0, 2.0]) / 12.0
    Wb = sum(W[i] * wb[i] for i in range(3))
    Wc = sum(W[i] * wc[i] for i in range(3))
    viol = (gate_c - gate_b) * Wb                         # gate channel (violation / zero events), at the baseline's W
    prog = {n: gate_c * W[i] * (wc[i] - wb[i]) for i, n in enumerate(("EP", "TTC", "C"))}
    tot = at(pdms, c) - at(pdms, b)
    chk = np.abs(viol + sum(prog.values()) - tot).max()
    d = {"max_abs_decomposition_error": float(chk), "total_dPDMS_x100": paired(boot, at(pdms, c), at(pdms, b)),
         "gate_channel_x100": paired(boot, viol + at(pdms, b), at(pdms, b)),
         "EP_channel_x100": paired(boot, prog["EP"] + at(pdms, b), at(pdms, b)),
         "TTC_channel_x100": paired(boot, prog["TTC"] + at(pdms, b), at(pdms, b)),
         "C_channel_x100": paired(boot, prog["C"] + at(pdms, b), at(pdms, b))}
    ch = c != b
    dp = tot[ch]
    d["changed_picks"] = int(ch.sum())
    d["changed_better"] = int((dp > 1e-9).sum()); d["changed_worse"] = int((dp < -1e-9).sum()); d["changed_equal"] = int((np.abs(dp) <= 1e-9).sum())
    d["mean_gain_when_better_x100"] = float(dp[dp > 1e-9].mean() * 100) if (dp > 1e-9).any() else None
    d["mean_loss_when_worse_x100"] = float(dp[dp < -1e-9].mean() * 100) if (dp < -1e-9).any() else None
    d["new_zero_picks"] = int(((at(pdms, c) == 0) & (at(pdms, b) > 0)).sum()); d["zero_picks_rescued"] = int(((at(pdms, c) > 0) & (at(pdms, b) == 0)).sum())
    # what the heads said about the move (predicted) against what happened (realised) on the changed picks
    d["predicted_vs_realised_on_changed_picks"] = {
        "pEP": [float(at(p[..., EP], b)[ch].mean()), float(at(p[..., EP], c)[ch].mean())], "EP_true": [float(wb[0][ch].mean()), float(wc[0][ch].mean())],
        "pTTC": [float(at(p[..., TTC], b)[ch].mean()), float(at(p[..., TTC], c)[ch].mean())], "TTC_true": [float(wb[1][ch].mean()), float(wc[1][ch].mean())],
        "pNCxDAC": [float(at(g, b)[ch].mean()), float(at(g, c)[ch].mean())], "NCxDAC_true": [float(gate_b[ch].mean()), float(gate_c[ch].mean())],
        "L_m": [float(at(L, b)[ch].mean()), float(at(L, c)[ch].mean())]}
    out[nm] = d
json.dump(out, open("../raw/mech.json", "w"), indent=1)
for nm, d in out.items():
    print("==", nm, "decomp err", d["max_abs_decomposition_error"])
    for k in ("total_dPDMS_x100", "gate_channel_x100", "EP_channel_x100", "TTC_channel_x100", "C_channel_x100"):
        print("  ", k, {x: round(v, 3) for x, v in d[k].items() if x in ("point", "lo", "hi")})
    print("  changed", d["changed_picks"], "better", d["changed_better"], "worse", d["changed_worse"], "equal", d["changed_equal"], "gain", d["mean_gain_when_better_x100"], "loss", d["mean_loss_when_worse_x100"],
          "new zero", d["new_zero_picks"], "rescued", d["zero_picks_rescued"])
    print("  ", {k: [round(x, 3) for x in v] for k, v in d["predicted_vs_realised_on_changed_picks"].items()})
