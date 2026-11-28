"""combine_acchist.py -- card PREREG_V8F_PROOF_ADD153 (with amendment 1): one summary of the six runs and the ensemble read, from the pulled ACC.json / ENS.json files.
Names exactly as the card commits them; computes nothing new beyond seed means and spreads. Writes ACCHIST10K_RESULT.json.

  python -I combine_acchist.py
"""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = {"K25D": "acc_K25D", "K25DM": "acc_K25DM", "K10D": "acc_K10D", "K40D": "acc_K40D", "K25D_s149": "acc_s2rep", "K25DS": "acc_s2smooth"}
KEYS = ("S", "A", "A_derived", "A_floor", "A_rel", "A_lab2", "n_S", "n_A", "gap_mae_m", "presence_auc")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def r4(x):
    return round(x, 4) if isinstance(x, float) else x


def main():
    out = {"card": "PREREG_V8F_PROOF_ADD153 + AM1", "tier": "T0 (a head on cached tokens of logged frames); never driving performance", "runs": {}, "sources": {}}
    for k, j in RUNS.items():
        p = os.path.join(HERE, "box2_" + j, "ACC.json")
        r = json.load(open(p, encoding="utf-8"))
        out["sources"][k] = {"path": "rulefc/leadacc/box2_%s/ACC.json" % j, "sha256": sha(p)}
        out["runs"][k] = {"arm": r["arm"], "train_seed": r["seeds"]["train"], "label_run": r["label_run"], "controls_all_pass": r["controls_all_pass"], "name": r.get("name"),
                          "eval141_name": r.get("eval141_name"), "held_out": {x: r4(r["held_out"][x]) for x in KEYS}, "eval141": {x: r4(r["eval141"][x]) for x in KEYS},
                          "dA_floor_held_out": [r4(r["dA_floor_held_out"]["diff"])] + [r4(v) for v in r["dA_floor_held_out"]["ci95"]],
                          "dA_floor_eval141": [r4(r["dA_floor_eval141"]["diff"])] + [r4(v) for v in r["dA_floor_eval141"]["ci95"]],
                          "vs_row217_head_held_out": {"dS": [r4(r["paired_vs_ref_held_out"]["dS"]["diff"])] + [r4(v) for v in r["paired_vs_ref_held_out"]["dS"]["ci95"]],
                                                      "dA": [r4(r["paired_vs_ref_held_out"]["dA"]["diff"])] + [r4(v) for v in r["paired_vs_ref_held_out"]["dA"]["ci95"]]},
                          "halves_held_out_S_A": [[r4(h["S"]), r4(h["A"])] for h in r["held_out_halves"]],
                          "train": {x: r["train"][x] for x in ("steps", "windows_fit", "seconds", "s_per_step", "loss_first1000", "loss_last1000", "d_loss_first1000", "d_loss_last1000")},
                          "ckpt_30000_sha256": r["train"]["ckpt_sha256"]["30000"], "n_params": r["n_params"], "receptive_field_frames": r["receptive_field_frames"]}
    a, b = out["runs"]["K25D"], out["runs"]["K25D_s149"]
    pair = {}
    for s in ("held_out", "eval141"):
        for x in ("S", "A"):
            pair["%s_%s" % (s, x)] = {"seed148": a[s][x], "seed149": b[s][x], "mean": r4((a[s][x] + b[s][x]) / 2), "spread": r4(abs(a[s][x] - b[s][x]))}
    both = a["name"] == b["name"]
    out["card_name"] = a["name"] if both else "ACCHIST10K-SEED-SPLIT"
    out["selected_pair"] = pair
    out["consequence_condition"] = {"eval141_A_seed148": a["eval141"]["A"], "eval141_A_seed149": b["eval141"]["A"], "bar": 0.30,
                                    "met": bool(both and a["name"].endswith("CARRIED") and a["eval141"]["A"] >= 0.30 and b["eval141"]["A"] >= 0.30)}
    pe = os.path.join(HERE, "box2_acc_ens", "ENS.json")
    if os.path.exists(pe):
        e = json.load(open(pe, encoding="utf-8"))
        out["sources"]["ENS"] = {"path": "rulefc/leadacc/box2_acc_ens/ENS.json", "sha256": sha(pe)}
        out["ensemble"] = {"controls_all_pass": e.get("controls_all_pass"), "error": e.get("error"), "consequence": e.get("consequence")}
        for s in ("single", "ENS2", "ENS3"):
            if s in e:
                out["ensemble"][s] = {"name": e[s].get("name"), "eval141_name": e[s].get("eval141_name"),
                                      "held_out": {x: r4(e[s]["held_out"][x]) for x in KEYS}, "eval141": {x: r4(e[s]["eval141"][x]) for x in KEYS},
                                      "dA_floor_held_out": [r4(e[s]["dA_floor_held_out"]["diff"])] + [r4(v) for v in e[s]["dA_floor_held_out"]["ci95"]],
                                      "dA_floor_eval141": [r4(e[s]["dA_floor_eval141"]["diff"])] + [r4(v) for v in e[s]["dA_floor_eval141"]["ci95"]]}
        if "ENS2_vs_single" in e:
            out["ensemble"]["ENS2_vs_single"] = {s: {x: [r4(e["ENS2_vs_single"][s][x]["diff"])] + [r4(v) for v in e["ENS2_vs_single"][s][x]["ci95"]] for x in ("dS", "dA")}
                                                 for s in ("held_out", "eval141")}
    json.dump(out, open(os.path.join(HERE, "ACCHIST10K_RESULT.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: out[k] for k in ("card_name", "selected_pair", "consequence_condition")}, indent=1))
    if "ensemble" in out:
        print(json.dumps(out["ensemble"], indent=1))


if __name__ == "__main__":
    main()
