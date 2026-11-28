#!/usr/bin/env python3
"""clr_wiring.py -- the O-sensitivity of CLR and the wiring it fixes for LLR (PREREG_LEVELHEAD_A2.md, Arms and Reads).  Reads readout/CLR_READOUT.json (A1's instrument run on the two CLR checkpoints) and writes
raw/CLR_WIRING.json + WIRING.md.  A1's thresholds (cl_readout.py SENS_HI 0.5, SENS_LO 0.1 m/s on the |mean signed plan-speed change| for a +3 m/s shift of v_hat; the mean-of-absolute reading is reported beside).
Wiring rule (card): BOTH seeds O-SENSITIVE -> LLR adds e_lead to the ego token like e_level (zero-init); BOTH O-INSENSITIVE -> LLR feeds BOTH heads' outputs as one separate conditioning token of O (--aux-cond token,
non-zero init).  Anything else (a WEAK reading, or the seeds disagree) is a band the card does not name: reported, the Master Mind decides.   usage: python -I clr_wiring.py [--readout <json>] [--raw <dir>] [--md <file>]"""
import argparse
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
SENS_HI, SENS_LO = 0.5, 0.1
SENSITIVE, INSENSITIVE, WEAK = "O-SENSITIVE", "O-INSENSITIVE", "WEAK"
WIRING = {SENSITIVE: "e_lead added to the ego token like e_level (zero-init)",
          INSENSITIVE: "BOTH heads' outputs ([sigmoid(logit), gap_hat] and the level) as one separate conditioning token of O (--aux-cond token, non-zero init)"}


def sens_label(x):
    """A1's label function on a speed change in m/s (point estimate)."""
    return SENSITIVE if x >= SENS_HI else (INSENSITIVE if x < SENS_LO else WEAK)


def wiring(labels):
    """labels = {seed key: A1 sensitivity label}.  Returns {decision, wiring, note}."""
    vals = set(labels.values())
    if len(labels) == 2 and vals == {SENSITIVE}:
        return {"decision": SENSITIVE, "wiring": WIRING[SENSITIVE], "note": "both seeds"}
    if len(labels) == 2 and vals == {INSENSITIVE}:
        return {"decision": INSENSITIVE, "wiring": WIRING[INSENSITIVE], "note": "both seeds"}
    return {"decision": "NOT-NAMED-BY-THE-CARD", "wiring": None,
            "note": f"labels {dict(sorted(labels.items()))}: the card names the wiring only for O-SENSITIVE or O-INSENSITIVE at BOTH seeds; the Master Mind decides (WEAK is A1's band between {SENS_LO} and {SENS_HI} m/s)"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--readout", default=f"{PK}/readout/CLR_READOUT.json")
    ap.add_argument("--raw", default=f"{PK}/raw")
    ap.add_argument("--md", default=f"{PK}/WIRING.md")
    a = ap.parse_args(argv)
    d = json.load(open(a.readout, encoding="utf-8"))
    per, lab, ctl = {}, {}, {}
    for k, v in d["per_seed"].items():
        sn, tr = v["sensitivity"], v["tracking"]
        lab[k] = v["labels"]["sensitivity"]
        per[k] = {"sensitivity_label_A1": lab[k], "sensitivity_label_recomputed": sens_label(abs(sn["plus3"]["mean_signed_change_ms"]["point"])),
                  "plus3_mean_signed_change_ms": sn["plus3"]["mean_signed_change_ms"]["point"], "plus3_mean_abs_change_ms": sn["plus3"]["mean_abs_change_ms"]["point"],
                  "label_alt_mean_abs_reading": sn["label_alt_mean_abs_reading"], "tracking_label": v["labels"]["tracking"], "r2_all": tr["r2_all"]["point"], "band_mae": tr["band_mae"]["point"],
                  "e_level_over_ego_norm": sn["e_level_over_ego_norm"]["mean_of_per_window_ratio"]["point"] if isinstance(sn["e_level_over_ego_norm"]["mean_of_per_window_ratio"], dict) else sn["e_level_over_ego_norm"]["mean_of_per_window_ratio"],
                  "control_failures": v["control_failures"]}
        ctl[k] = bool(per[k]["sensitivity_label_A1"] == per[k]["sensitivity_label_recomputed"] and not v["control_failures"])
    w = wiring(lab)
    res = {"source": a.readout, "thresholds_ms": {"O-SENSITIVE_at_least": SENS_HI, "O-INSENSITIVE_below": SENS_LO}, "per_seed": per, "controls_ok": ctl, "WIRING": w}
    if not all(ctl.values()):
        res["WIRING"] = {"decision": "REFUSED", "wiring": None, "note": f"the readout's own controls failed or its label disagrees with the recomputed one: {ctl}"}
    os.makedirs(a.raw, exist_ok=True)
    json.dump(res, open(f"{a.raw}/CLR_WIRING.json", "w", encoding="utf-8", newline="\n"), indent=1)
    L = ["# WIRING: the O-sensitivity of CLR and the wiring it fixes for LLR (A1's instrument on the CLR checkpoints; T0 readout, not driving performance)", "",
         f"**Decision: {res['WIRING']['decision']}**" + (f" -> {res['WIRING']['wiring']}" if res["WIRING"]["wiring"] else ""), "", f"Note: {res['WIRING']['note']}.", "",
         "| seed | sensitivity (A1, signed) | +3 m/s mean signed change | mean-abs reading | label (mean-abs) | tracking | R2 | band MAE |", "|---|---|---|---|---|---|---|---|"]
    for k, p in sorted(per.items()):
        L.append(f"| {k} | {p['sensitivity_label_A1']} | {p['plus3_mean_signed_change_ms']:+.4f} | {p['plus3_mean_abs_change_ms']:.4f} | {p['label_alt_mean_abs_reading']} | {p['tracking_label']} | {p['r2_all']:.3f} | {p['band_mae']:.3f} |")
    open(a.md, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    print("WIRING", res["WIRING"]["decision"], "|", res["WIRING"]["wiring"])
    return 0 if res["WIRING"]["decision"] != "REFUSED" else 3


if __name__ == "__main__":
    sys.exit(main())
