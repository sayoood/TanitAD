#!/usr/bin/env python3
"""WHERE does an IN-DOMAIN head beat the navtrain-trained head, on the SAME 1,058 navtest frames?  CPU, read-only.
The scene-tap probe (raw/2026-10-15-scene-tap-probe, A2 raw/2026-10-16-scene-tap-a2) banked, per (arm, seed, fold), the head's prediction on its held-out fold (log-disjoint 5-fold over the 1,058 C3 tokens).
V = visual tap = the S1 head's own config trained from scratch on ~861 navtest tokens (2,000 updates; A2: 10,000 updates).  Here: its P1 IoU [0,40) by map / stratum / place vs H1's, same frames, same labels.
Re-derivation first: V's pooled IoU must reproduce PROBE_RESULT.json / A2_RESULT.json."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shf_common as C
import shf_arrays as SA

ARMS = tuple(os.environ.get("SHF_ARMS", "H1-s0,H1-s1,H1-s2").split(","))
Dn = SA.domain("navtest", ARMS)
tab = np.load(os.path.join(C.PK, "raw", "2026-10-11-c3-selection-headroom", "table.npz"), allow_pickle=False)
tab_tokens = [str(t) for t in tab["token"]]
pos = {t: i for i, t in enumerate(Dn["tokens"])}
b = SA.band()[None]


def load_V(dirname, seed, upd_tag):
    """-> predicted mask [N,40,32] aligned to Dn row order (every token is in exactly one held-out fold)."""
    PM = np.zeros_like(Dn["TM"])
    seen = np.zeros(Dn["n"], bool)
    for f in range(5):
        z = np.load(f"{C.ART}/{dirname}/units/V_s{seed}_f{f}.npz", allow_pickle=True)
        for r, hi in enumerate(z["held_idx"]):
            j = pos[tab_tokens[int(hi)]]
            assert not seen[j]
            PM[j] = z["logit"][r] > 0
            seen[j] = True
    assert seen.all()
    return PM


def parts(PM):
    tm = Dn["TM"]
    return (tm & PM & b).sum((1, 2)).astype(float), ((tm | PM) & b).sum((1, 2)).astype(float)


def iou(i, u, m=None):
    m = np.ones(len(i), bool) if m is None else m
    return float(i[m].sum() / u[m].sum()) if m.any() else float("nan")


units = {"V2k_s0": load_V("scene_tap_probe", 0, "2k"), "V2k_s1": load_V("scene_tap_probe", 1, "2k"), "V2k_s2": load_V("scene_tap_probe", 2, "2k"),
         "V10k_s0": load_V("scene_tap_probe_a2", 0, "10k")}
for a in ARMS:
    units[a] = Dn["PM"][a]
P = {k: parts(v) for k, v in units.items()}
res = {}
probe = json.load(open(os.path.join(C.PK, "raw", "2026-10-15-scene-tap-probe", "PROBE_RESULT.json"), encoding="utf-8"))
a2 = json.load(open(os.path.join(C.PK, "raw", "2026-10-16-scene-tap-a2", "A2_RESULT.json"), encoding="utf-8"))
res["R0_pooled_all"] = {k: iou(*P[k]) for k in P}
res["R0_banked_V2k_s0"] = 0.634891  # A2_RESULT.md: 'H(V-s0, 2,000 updates), banked 0.634891'
res["R0_banked_V10k_s0"] = 0.675218
res["R0_abs_diff_V2k_s0"] = abs(res["R0_pooled_all"]["V2k_s0"] - 0.634891)
res["R0_abs_diff_V10k_s0"] = abs(res["R0_pooled_all"]["V10k_s0"] - 0.675218)
print("R0", res["R0_pooled_all"], res["R0_abs_diff_V2k_s0"], res["R0_abs_diff_V10k_s0"], flush=True)

def block(mask_fn):
    out = {}
    for k in P:
        out[k] = iou(*P[k], m=mask_fn)
    return out

res["by_map"] = {m: dict(n=int((Dn["maps"] == m).sum()), **block(Dn["maps"] == m)) for m in ("LV", "BOS", "PIT", "SG")}
res["by_stratum"] = {s: dict(n=int((Dn["stratum"] == s).sum()), **block(Dn["stratum"] == s)) for s in ("zero", "random")}
res["by_map_x_stratum"] = {f"{m}|{s}": dict(n=int(((Dn["maps"] == m) & (Dn["stratum"] == s)).sum()), **block((Dn["maps"] == m) & (Dn["stratum"] == s))) for m in ("LV", "BOS", "PIT", "SG") for s in ("zero", "random")}
pl = np.load(os.path.join(C.SCRATCH, "place_dist.npz"))
dn = pl["dn"]
edges = [0, 2, 10, 1e9]
bins = np.digitize(dn, edges[1:-1])
res["by_place_distance_to_navtrain_training_frames_m"] = {nm: dict(n=int((bins == k).sum()), **block(bins == k)) for k, nm in enumerate(["<2", "2-10", ">=10"])}
# gain of V over H1 (mean over available H1 arms) by map: where does in-domain training help?
h1 = [a for a in ARMS]
gain = {}
for m in ("LV", "BOS", "PIT", "SG"):
    k = Dn["maps"] == m
    gain[m] = {"V2k_s0_minus_H1mean": iou(*P["V2k_s0"], m=k) - float(np.mean([iou(*P[a], m=k) for a in h1])),
               "V10k_s0_minus_H1mean": iou(*P["V10k_s0"], m=k) - float(np.mean([iou(*P[a], m=k) for a in h1]))}
res["gain_by_map"] = gain
# cell-level: where do V10k and H1 differ? FN / FP shares in 0-40
for nm, PMx in (("V10k_s0", units["V10k_s0"]), (ARMS[0], units[ARMS[0]])):
    tm, pm = Dn["TM"] & b, PMx & b
    tot = b.sum() * Dn["n"]
    res.setdefault("confusion_0_40", {})[nm] = dict(pred_share=float(pm.sum() / tot), true_share=float(tm.sum() / tot), recall=float((tm & pm).sum() / tm.sum()),
                                                    precision=float((tm & pm).sum() / pm.sum()), fn_share=float((tm & ~pm).sum() / tot), fp_share=float((~(Dn["TM"]) & pm & b).sum() / tot))
C.jdump(res, os.path.join(C.HERE, os.environ.get("SHF_OUT", "d2_indomain_vs_transfer.json")))
print(json.dumps({k: res[k] for k in ("by_map", "by_stratum", "by_place_distance_to_navtrain_training_frames_m", "gain_by_map", "confusion_0_40")}, indent=1, sort_keys=True))
