"""SC-13 run-#5 analysis — speed-controlled AUROCs with EPISODE-CLUSTER intervals.

Reads the substrate written by `sc13_probe_v5.py` and answers, in one pass:

  1. Does run #4's in-domain positive survive 2x the anchors?  (falsifier F-A)
  2. Is any of it scene-specific, i.e. does `held` beat a REAL-window control
     drawn from another episode?                               (falsifier F-B)
  3. Does run #4's stride-2 result reproduce EXACTLY from this substrate?
     (`--subset stride2` keeps anchors whose START index is even — that is
     precisely run #4's anchor set, so its numbers are re-derived, not recalled.)

INTERVALS.  The decision-grade estimator here is the EPISODE-CLUSTER bootstrap:
resample the 40 val episodes with replacement, pool their anchors, rescore.
Anchors 0.1 s apart inside one episode are near-duplicates, so run #4's
anchor-level bootstrap treats correlated near-copies as independent facts and is
anticonservative — it is kept only for comparability. Arm differences are
bootstrapped PAIRED on the same resample; never combined from two marginal
intervals (CLAUDE.md).

SPEED CONTROL.  Braking anchors sit at a lower v0 than cruise anchors, and any
signal that merely grows as speed falls would score high while anticipating
nothing. Two independent controls, as in run #4: per-event +/-TOL m/s MATCHING
(the headline) and v0-STRATIFICATION (the cross-check).
"""
from __future__ import annotations

import argparse
import json

import torch

TOL = 1.0
DROP, DROP_FAR, VMIN = 2.0, 1.5, 5.0
ARMS = ("informed", "held", "blind", "shuffled", "frozen")
# every difference the falsifiers are stated over, bootstrapped paired
DIFFS = (("held", "reactive"), ("held", "shuffled"), ("held", "blind"),
         ("held", "frozen"), ("shuffled", "reactive"))


def matched_auroc(s, ev_i, cr_i, v0, tol=TOL):
    """AUROC of s(event) vs s(cruise), each event scored only against cruise
    anchors within +/-tol m/s of its own v0. Returns (auroc, n_pairs)."""
    if len(ev_i) == 0 or len(cr_i) == 0:
        return float("nan"), 0
    se, sc = s[ev_i].reshape(-1, 1), s[cr_i].reshape(1, -1)
    m = (v0[ev_i].reshape(-1, 1) - v0[cr_i].reshape(1, -1)).abs() <= tol
    n = int(m.sum())
    if n == 0:
        return float("nan"), 0
    w = ((se > sc).float() + 0.5 * (se == sc).float())[m].sum()
    return float(w) / n, n


def raw_auroc(s, ev_i, cr_i):
    if len(ev_i) == 0 or len(cr_i) == 0:
        return float("nan")
    se, sc = s[ev_i].reshape(-1, 1), s[cr_i].reshape(1, -1)
    return float(((se > sc).float() + 0.5 * (se == sc).float()).mean())


def strat_auroc(s, ev, cr, v0, edges=(5, 8, 11, 14, 17, 21, 40)):
    num, den = 0.0, 0
    for lo, hi in zip(edges, edges[1:]):
        b = (v0 >= lo) & (v0 < hi)
        e, c = torch.nonzero(ev & b).flatten(), torch.nonzero(cr & b).flatten()
        if len(e) >= 3 and len(c) >= 5:
            num += raw_auroc(s, e, c) * len(e)
            den += len(e)
    return (num / den if den else float("nan")), den


def pct(v, lo=0.025, hi=0.975):
    v = torch.tensor([x for x in v if x == x])          # drop NaNs
    if len(v) < 20:
        return [float("nan"), float("nan")]
    v = v.sort().values
    return [round(float(v[int(lo * len(v))]), 3),
            round(float(v[int(hi * len(v))]), 3)]


def run(p, subset, n_boot, out_prefix):
    v0, vf, eidx = p["v0"], p["vfut"], p["eidx"]
    keep = torch.ones(len(v0), dtype=torch.bool)
    if subset == "stride2":
        keep = (p["sidx"] % 2) == 0
    idx = torch.nonzero(keep).flatten()

    drop_near = v0 - vf[:, :20].min(dim=1).values
    drop_far = v0 - vf[:, 20:30].min(dim=1).values
    swing = (vf - v0[:, None]).abs().max(dim=1).values
    fast = v0 >= VMIN
    LAB = {"brake_near": fast & (drop_near >= DROP),
           "brake_far": fast & (drop_far >= DROP_FAR) & (drop_near < 0.75)}
    cruise = fast & (swing <= 0.5)

    sig = {a: p["cv"][:, 0] - p[a][:, 0] for a in ARMS}
    sig["gt_oracle"] = p["cv"][:, 0] - p["gt"][:, 0]
    sig["reactive"] = p["reactive"]

    n_ep = int(eidx.max()) + 1
    res = {"subset": subset, "n_anchors": int(keep.sum()), "n_episodes": n_ep,
           "tol_mps": TOL, "n_boot": n_boot,
           "ade2s_m": {a: round(float((p[a][idx] - p["gt"][idx]).norm(dim=-1)
                                      .mean()), 4) for a in ARMS}}
    res["ade2s_m"]["cv"] = round(
        float((p["cv"][idx] - p["gt"][idx]).norm(dim=-1).mean()), 4)

    g = torch.Generator().manual_seed(7)
    for lbl, ev_all in LAB.items():
        ev, cr = ev_all & keep, cruise & keep
        ev_i, cr_i = torch.nonzero(ev).flatten(), torch.nonzero(cr).flatten()
        row = {"n_events": int(ev.sum()), "n_cruise": int(cr.sum()),
               "n_event_episodes": int(len(set(eidx[ev_i].tolist()))),
               "median_v0_event": round(float(v0[ev_i].median()), 2)
               if len(ev_i) else float("nan"),
               "median_v0_cruise": round(float(v0[cr_i].median()), 2)
               if len(cr_i) else float("nan")}
        point = {}
        for k, s in sig.items():
            m, npair = matched_auroc(s, ev_i, cr_i, v0)
            st, den = strat_auroc(s, ev, cr, v0)
            point[k] = m
            row[f"raw_{k}"] = round(raw_auroc(s, ev_i, cr_i), 3)
            row[f"matched_{k}"] = round(m, 3) if m == m else float("nan")
            row[f"strat_{k}"] = round(st, 3) if st == st else float("nan")
            row["n_matched_pairs"] = npair
            row["n_stratified_events"] = den

        # ---- bootstraps -------------------------------------------------
        per_ep_ev = [torch.nonzero(ev & (eidx == e)).flatten()
                     for e in range(n_ep)]
        per_ep_cr = [torch.nonzero(cr & (eidx == e)).flatten()
                     for e in range(n_ep)]
        boot_ep = {k: [] for k in sig}
        boot_an = {k: [] for k in sig}
        dboot_ep = {f"{a}-{b}": [] for a, b in DIFFS}
        dboot_an = {f"{a}-{b}": [] for a, b in DIFFS}
        for _ in range(n_boot):
            pick = torch.randint(n_ep, (n_ep,), generator=g)
            e_i = torch.cat([per_ep_ev[int(j)] for j in pick]) \
                if any(len(per_ep_ev[int(j)]) for j in pick) \
                else torch.tensor([], dtype=torch.long)
            c_i = torch.cat([per_ep_cr[int(j)] for j in pick]) \
                if any(len(per_ep_cr[int(j)]) for j in pick) \
                else torch.tensor([], dtype=torch.long)
            cur = {}
            for k, s in sig.items():
                a, _ = matched_auroc(s, e_i, c_i, v0)
                cur[k] = a
                boot_ep[k].append(a)
            for a, b in DIFFS:
                dboot_ep[f"{a}-{b}"].append(cur[a] - cur[b])
            # anchor-level (run #4's estimator, kept for comparability)
            if len(ev_i) > 1 and len(cr_i) > 1:
                ea = ev_i[torch.randint(len(ev_i), (len(ev_i),), generator=g)]
                ca = cr_i[torch.randint(len(cr_i), (len(cr_i),), generator=g)]
                cur2 = {}
                for k, s in sig.items():
                    a, _ = matched_auroc(s, ea, ca, v0)
                    cur2[k] = a
                    boot_an[k].append(a)
                for a, b in DIFFS:
                    dboot_an[f"{a}-{b}"].append(cur2[a] - cur2[b])
        for k in sig:
            row[f"ci95_epcluster_{k}"] = pct(boot_ep[k])
            row[f"ci95_anchor_{k}"] = pct(boot_an[k])
        for a, b in DIFFS:
            row[f"diff_{a}_minus_{b}"] = (
                round(point[a] - point[b], 3)
                if point[a] == point[a] and point[b] == point[b]
                else float("nan"))
            row[f"diffci95_epcluster_{a}_minus_{b}"] = pct(dboot_ep[f"{a}-{b}"])
            row[f"diffci95_anchor_{a}_minus_{b}"] = pct(dboot_an[f"{a}-{b}"])
        res[lbl] = row

    # ---- pre-registered verdicts (BRAKE_FAR, matched) --------------------
    bf = res["brake_far"]
    h, rct = bf.get("matched_held"), bf.get("matched_reactive")
    ctrl = [bf.get(f"matched_{c}") for c in ("shuffled", "blind")]
    ctrl = [c for c in ctrl if c == c]
    best = max(ctrl) if ctrl else float("nan")
    fa = (h - rct) if (h == h and rct == rct) else float("nan")
    fb = (h - best) if (h == h and best == best) else float("nan")
    lo_a = bf.get("diffci95_epcluster_held_minus_reactive", [float("nan")])[0]
    res["verdict"] = {
        "F_A_volume_margin_held_minus_reactive": round(fa, 3) if fa == fa else None,
        "F_A_threshold": 0.10,
        "F_A_FIRED_effect_did_not_survive": bool(fa == fa and fa <= 0.10),
        "F_B_vision_margin_held_minus_best_control": round(fb, 3) if fb == fb else None,
        "F_B_threshold": 0.02,
        "F_B_best_control": ("shuffled" if ctrl and best == bf.get("matched_shuffled")
                             else "blind"),
        "F_B_FIRED_not_scene_specific": bool(fb == fb and fb <= 0.02),
        "epcluster_CI_lo_held_minus_reactive": lo_a,
        "survives_only_if": "F_A margin >0.10 AND F_B margin >0.02 AND "
                            "episode-cluster CIs of both differences exclude 0",
    }
    out = f"{out_prefix}_{subset}.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res["verdict"], indent=2))
    print(f"[sc13v5] brake_far n_ev={bf['n_events']} "
          f"over {bf['n_event_episodes']} eps -> {out}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("windows")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out-prefix", default=None)
    a = ap.parse_args()
    p = torch.load(a.windows, map_location="cpu", weights_only=False)
    pre = a.out_prefix or a.windows.replace("_windows.pt", "_analysis")
    for subset in ("all", "stride2"):
        print(f"\n===== subset={subset} =====")
        run(p, subset, a.boot, pre)


if __name__ == "__main__":
    main()
