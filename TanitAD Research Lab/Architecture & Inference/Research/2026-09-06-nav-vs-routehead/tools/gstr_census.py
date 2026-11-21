"""D-NAVROUTE-1 §2.3/§2.4 -- the g_str DEGENERACY census and the
presence-vs-content statistic, banked as an artifact.

⛔ WHY THIS FILE EXISTS. The first version of RESULT.md quoted `-0.1022`,
`0.0098`, `0.4018`, `40.9x`, `1.1226` and `1.8644` from throwaway one-liners with
NO artifact under `raw/`. That is a two-key-rule violation (every number needs an
artifact path AND an evidence class), and it was caught by the programme's own
review. This script recomputes exactly those numbers from the banked dump and
emits them as JSON so each is quotable.

⭐ THE STATISTIC IS NAMED, not merely the quantity: `g_str` movement is the MEAN
over windows of the RELATIVE L2 displacement of the 3-vector,
mean(||g_a - g_b|| / max(||g_b||, 1e-6)). A different normalisation gives a
different number for the same effect -- which is why the earlier bare "18.6x"
could not be reproduced here as a magnitude, only as a direction.

ASCII-only output (cp1252-safe).
"""
import glob
import json
import os
import sys

import numpy as np

BANK = os.environ.get("BANK", "/home/nvidia/navpred/navflip_dump")
CONDS = ("nav_true", "nav_shuffled", "nav_zero", "nav_flipped", "nav_predicted")


def main():
    files = sorted(glob.glob(os.path.join(BANK, "decisions", "*.npz")))
    G = {c: [] for c in CONDS}
    P = {c: [] for c in CONDS}
    gt = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        for c in CONDS:
            G[c].append(np.asarray(z[f"gstr_{c}"]))
            P[c].append(np.asarray(z[f"plan_full_{c}"]))
        gt.append(np.asarray(z["gt_future_ext"])[:, -1, 1])
    G = {c: np.concatenate(v) for c, v in G.items()}
    P = {c: np.concatenate(v) for c, v in P.items()}
    gt = np.concatenate(gt)
    n = G["nav_true"].shape[0]

    sin = G["nav_true"][:, 1]
    res = {
        "tool": "D-NAVROUTE-1 g_str census + presence-vs-content",
        "bank": BANK, "n_episodes": len(files), "n_windows": int(n),
        "tier": "T1 (self-action OPEN loop, 2026-09-02 ruling)",
        "evidence_class": "MEASURED (ours)",
        "DEGENERACY": {
            "_claim": "g_str's lateral bearing component never points LEFT",
            "statistic": "gstr_nav_true[:,1] (the sin/lateral component of the "
                         "unit bearing); +y is LEFT (orientation control in "
                         "CONTINGENCY.json PASSES)",
            "n_windows": int(n),
            "n_sin_gt_0_LEFT_pointing": int((sin > 0).sum()),
            "sin_min": round(float(sin.min()), 4),
            "sin_max": round(float(sin.max()), 4),
            "sin_median": round(float(np.median(sin)), 4),
            "sin_range_width": round(float(sin.max() - sin.min()), 4),
            "_reads": "the head VARIES but its SIGN never does: a direction "
                      "class constant at RIGHT. A structural zero, not an "
                      "estimate -- the same family as H-ECHO-4.",
        },
        "GROUND_TRUTH_FOR_CONTRAST": {
            "rule": "GT terminal lateral y, deadband tau_y = 1.0 m",
            "n_LEFT": int((gt > 1.0).sum()),
            "n_STRAIGHT": int((np.abs(gt) <= 1.0).sum()),
            "n_RIGHT": int((gt < -1.0).sum()),
            "_reads": "ground truth turns LEFT on these windows while g_str "
                      "never once points that way",
        },
    }

    def relmove(a, b):
        return float(np.mean(np.linalg.norm(a - b, axis=-1)
                             / np.maximum(np.linalg.norm(b, axis=-1), 1e-6)))

    base_g, base_p = G["nav_true"], P["nav_true"]
    pc = {"_statistic": "MEAN over windows of relative L2 displacement of the "
                        "g_str 3-vector vs nav_true",
          "_warning": "normalisation-dependent: quote WITH this definition or "
                      "not at all"}
    for c in ("nav_flipped", "nav_shuffled", "nav_zero"):
        pc[c] = round(relmove(G[c], base_g), 4)
    pc["ratio_removal_over_value_change"] = round(
        pc["nav_zero"] / max(pc["nav_flipped"], 1e-9), 2)
    pc["_reads"] = ("the strategic head responds to nav's PRESENCE, not its "
                    "CONTENT -- the presence-gating already measured at the "
                    "planner, reproduced one level up")
    res["PRESENCE_VS_CONTENT_gstr"] = pc

    pl = {"_statistic": "plan TERMINAL displacement (m) vs nav_true, over the "
                        "FULL banked dump (not the 29-episode panel subset)"}
    for c in ("nav_flipped", "nav_shuffled", "nav_zero"):
        d = np.linalg.norm(P[c][:, -1] - base_p[:, -1], axis=-1)
        pl[c] = {"mean_m": round(float(d.mean()), 4),
                 "median_m": round(float(np.median(d)), 4),
                 "max_m": round(float(d.max()), 4),
                 "frac_plan_UNCHANGED": round(float((d == 0).mean()), 4)}
    pl["_reads"] = ("inverting the commanded route changes the emitted plan by "
                    "EXACTLY ZERO on most windows")
    res["PLAN_DISPLACEMENT_full_bank"] = pl
    print(json.dumps(res, indent=1))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
