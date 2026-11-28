"""Builds RESULT_C10P.{json,md} from the two R35 step-0 reads (read/step0/ia3pk_R35_s{0,1}.json, produced by ia3pk_read.py). Card: AMEND_OPTB section 4, C10'."""
import json, os
H = os.path.dirname(os.path.abspath(__file__))
S0 = os.path.join(H, "..", "step0")
rows, ok = {}, True
for k in (0, 1):
    p = os.path.join(S0, f"ia3pk_R35_s{k}.json")
    r = json.load(open(p, encoding="utf-8"))
    ok &= bool(r["s_swap"]["ci95"][0] > 0)
    rows[f"R35_s{k}"] = {"artifact": os.path.normpath(p).replace("\\", "/"), "n_anchors": r["n_anchors"], "n_clips": r["n_clips"], "s_swap": r["s_swap"],
                         "ci_wholly_above_0": bool(r["s_swap"]["ci95"][0] > 0), "ra_crossfit_r2_heldout": r["ra_crossfit_r2_heldout"],
                         "step0_ia3pk_gain": r["ia3pk_gain_L1_2s_lon"], "step0_target_read_cycle_gain_frozen_probe": r["target_read_cycle_gain_frozen_probe"],
                         "train_clip_reference_amend1": {"s_swap": [0.201, 0.203][k], "G_aa": [1.244, 1.202][k]}}
out = {"control": "C10' (AMEND_OPTB section 4)", "label": "C10' PASS" if ok else "C10' FAIL", "tier": "T0", "evidence": "MEASURED (artifacts above)",
       "rule": "PASS iff the s_swap CI on the acceleration channel is wholly above 0 at both chassis, read on the 141 EVAL clips with IA-3's folds (lr_lib.fold_of, 2 folds)",
       "estimator": "episode-cluster bootstrap over the 141 eval clips, B 2000, seed 0; answers another draw of episodes with R35's weights and the cross-fitted probe FORM held fixed (not another training run)",
       "readout": "RA's form (ridge on z_{t+10} - z_t, lambda on inner clip-grouped folds) cross-fitted by IA-3's folds on the eval clips; s_swap on real pairs (same-clip partner >= 5 frames away, RandomState(0))",
       "chassis": rows}
json.dump(out, open(os.path.join(H, "RESULT_C10P.json"), "w", encoding="utf-8"), indent=1)
f = lambda d: f"{d['point']:.4f} [{d['ci95'][0]:.4f}, {d['ci95'][1]:.4f}]"
md = [f"# C10' on R35: {out['label']}", "", "Tier T0, MEASURED. 141 EVAL clips, IA-3's folds, acceleration channel, K = 10 (2 s). Zero training, CPU.", "",
      f"Estimator: {out['estimator']}.", "", "| chassis | n anchors | s_swap (point [95% CI]) | CI wholly above 0 | cross-fit R2 | train-clip s_swap (Amend 1) |", "|---|---|---|---|---|---|"]
for t, r in rows.items():
    md.append(f"| {t} | {r['n_anchors']} | {f(r['s_swap'])} | {r['ci_wholly_above_0']} | {r['ra_crossfit_r2_heldout']:.3f} | {r['train_clip_reference_amend1']['s_swap']} |")
md += ["", "Beside (step 0 of the fine-tune, no bar): the term's response on the eval clips.", "", "| chassis | IA-3P-K gain (cross-fit, s_swap) | cycle gain through the frozen train-fit RA | Amend 1 train-clip G_aa |", "|---|---|---|---|"]
for t, r in rows.items():
    md.append(f"| {t} | {f(r['step0_ia3pk_gain'])} | {f(r['step0_target_read_cycle_gain_frozen_probe'])} | {r['train_clip_reference_amend1']['G_aa']} |")
md += ["", "Source artifacts: " + "; ".join(r["artifact"] for r in rows.values()), "Code: read/ia3pk_read.py (read_core / crossfit_ra / s_swap_from_moments)."]
open(os.path.join(H, "RESULT_C10P.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
print(out["label"])
