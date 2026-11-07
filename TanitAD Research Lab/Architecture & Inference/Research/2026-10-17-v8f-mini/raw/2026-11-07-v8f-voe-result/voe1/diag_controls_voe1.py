"""POST HOC DIAGNOSTIC (not part of the card): why did VOE1's controls 1 and 2 fail? Reads RESULT_VOE1.json (controls only: real-vs-real rows and the random-init FREEZE /
ACTION-SWAP rows; no M2b manipulation row is read) and, for the real-vs-real control, re-scores the window shifted by 2, 3, 4 and 5 frames instead of 1.
Moves no bar. Writes DIAG_CONTROLS_VOE1.json / .md. Needs WMACT_CODE_ROOT (like run_voe1.py)."""
import json, os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import voe1_core as C
import voe1_real as R

res = json.load(open(os.path.join(HERE, "RESULT_VOE1.json"), encoding="utf-8"))
tags = res["tags"]; LV = ("1", "2", "3")
c1_tags, c2_tags = R.DECIDING + R.CONTROL, R.CONTROL
out = {"note": "POST HOC diagnostic; the card's verdict on controls 1 and 2 stands as written"}

# (a) the committed control rules' own false-fail rate, from the run's 20 calibration draws (each cell has its own independent coin flips)
def draws_any(cells):
    n = len(cells[0])
    return [bool(any(c[d] for c in cells)) for d in range(n)]
rvr = [[a or b for a, b in zip(tags[t]["rvr"]["levels"][l]["cal"]["sep_above"], tags[t]["rvr"]["levels"][l]["cal"]["sep_below"])] for t in c1_tags for l in LV]
c2 = [tags[t]["manips"][m]["levels"][l]["acc"]["cal"]["sep_above"] for t in c2_tags for m in ("FREEZE", "ACTION-SWAP") for l in LV]
p1, p2 = np.mean(rvr), np.mean(c2)
out["fwer"] = {"control1": {"cells": len(rvr), "per_cell_share": float(p1), "draws_with_any_cell_separated": int(sum(draws_any(rvr))), "of_draws": len(rvr[0]),
                            "if_independent_1-(1-p)^12": float(1 - (1 - p1) ** len(rvr))},
               "control2": {"cells": len(c2), "per_cell_share": float(p2), "draws_with_any_cell_NOTICES_like": int(sum(draws_any(c2))), "of_draws": len(c2[0]),
                            "if_independent_1-(1-p)^12": float(1 - (1 - p2) ** len(c2))}}

# (b) the control cells themselves
def row(r):
    return {k: r[k] for k in ("point", "lo", "hi", "ref_lo", "ref_hi", "rank_above", "rank_below", "boot_above", "boot_below", "sep_above", "sep_below")}
out["control1_cells"] = {f"{t}/L{l}": row(tags[t]["rvr"]["levels"][l]) for t in c1_tags for l in LV}
out["control2_cells"] = {f"{t}/{m}/L{l}": {**row(tags[t]["manips"][m]["levels"][l]["acc"]), "label": tags[t]["manips"][m]["levels"][l]["label"], "n": tags[t]["manips"][m]["n"]} for t in c2_tags for m in ("FREEZE", "ACTION-SWAP") for l in LV}

# (c) real-vs-real with the later copy 2, 3, 4, 5 frames later (0.4 to 1.0 s): a systematic bias persists, noise does not
Bk = torch.load(os.path.join(HERE, "bank_voe1.pt"), weights_only=False)
w = Bk["win"]; keep = np.asarray(w["keep"])
eid = np.array(w["eid"])[keep]; base = np.asarray(w["base"])[keep]; g0 = np.asarray(w["g0"])[keep]
T_clip = np.array([Bk["T"][e] for e in eid]); ep, n_ep = C.episode_codes(eid)
trained = R.load_trained(); wms = R.build_wms(trained); act_t = Bk["act"].float()
out["rvr_shifts"] = {}
for tag in c1_tags:
    m = C.HJEPATraces(wms[tag].stack); z = Bk["z1"][tag].float()
    with torch.no_grad():
        a = wms[tag].scale_actions(act_t)
    take = lambda fr, ac: m.traces(z[torch.from_numpy(fr)], a[torch.from_numpy(ac)])
    tr_u = take(*C.idx_unmod(base)); med = C.medians(tr_u); Su = C.window_scores(tr_u, med)
    for k in (2, 3, 4, 5):
        q = g0 + k + (C.NF - 1) <= T_clip - 1
        Sb = C.window_scores(take(*C.idx_unmod(base[q], shift=k)), med)
        e2, n2 = C.episode_codes(eid[q])
        for l in LV:
            r = C.two_part(C.pair_x(Sb[int(l)]["sur"], Su[int(l)]["sur"][q]), e2, n2, 0.5, f"{tag}/RVRshift{k}/L{l}", cal_seeds=())
            out["rvr_shifts"].setdefault(f"{tag}/L{l}", {})[f"shift{k}"] = row(r)
json.dump(out, open(os.path.join(HERE, "DIAG_CONTROLS_VOE1.json"), "w", encoding="utf-8", newline="\n"), indent=1)

fl = lambda r: "S-above" if r["sep_above"] else "S-below" if r["sep_below"] else "-"
L = ["# DIAG_CONTROLS_VOE1 (POST HOC; not part of the card; the INVALID verdict stands as written)",
     "Source: `RESULT_VOE1.json` (control rows only) and a re-score of the real-vs-real control at larger shifts. No M2b manipulation row is read here.", "",
     "## (a) Family-wise false-fail rate of the controls as committed (from the run's own 20 calibration draws)",
     f"* Control 1 (12 cells, either side): per-cell separated share {p1:.3%}; {out['fwer']['control1']['draws_with_any_cell_separated']} of {out['fwer']['control1']['of_draws']} calibration draws had at least one of the 12 cells separated; independent-cells expectation {out['fwer']['control1']['if_independent_1-(1-p)^12']:.0%}.",
     f"* Control 2 (12 cells, NOTICES-like = separated above 0.5): per-cell share {p2:.3%}; {out['fwer']['control2']['draws_with_any_cell_NOTICES_like']} of {out['fwer']['control2']['of_draws']} draws had at least one; independent-cells expectation {out['fwer']['control2']['if_independent_1-(1-p)^12']:.0%}.",
     "* A permuted draw has no signal by construction, so these are the rates at which a clean pipeline fails the control as worded ('at any level', 12 cells each). Cells within a calibration draw are independent (own coin flips), real cells are not, so the real rate is lower than the independent figure but not zero.", "",
     "## (b) Control 1 cells (shift 1 = the card's 0.2 s)", "| cell | accuracy [interval] | reference 2.5 / 97.5 % | flag |", "|---|---|---|---|"]
for k, r in out["control1_cells"].items():
    L.append(f"| {k} | {r['point']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}] | {r['ref_lo']:.3f} / {r['ref_hi']:.3f} | {fl(r)} |")
L += ["", "## (c) Control 2 cells (random-init, accuracy above 0.5 needed for NOTICES)", "| cell | n | accuracy [interval] | reference 97.5 % | rank (above) | label |", "|---|---|---|---|---|---|"]
for k, r in out["control2_cells"].items():
    L.append(f"| {k} | {r['n']} | {r['point']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}] | {r['ref_hi']:.3f} | {r['rank_above']:.3f} | {r['label']} |")
L += ["", "## (d) Real-vs-real with the later copy 0.4 to 1.0 s later (a systematic bias would persist across shifts)", "| cell | shift 1 (card) | shift 2 | shift 3 | shift 4 | shift 5 |", "|---|---|---|---|---|---|"]
for k, d in out["rvr_shifts"].items():
    s1 = out["control1_cells"][k]
    L.append(f"| {k} | {s1['point']:.3f} {fl(s1)} | " + " | ".join(f"{d[f'shift{j}']['point']:.3f} {fl(d[f'shift{j}'])}" for j in (2, 3, 4, 5)) + " |")
open(os.path.join(HERE, "DIAG_CONTROLS_VOE1.md"), "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
print("\n".join(L))
