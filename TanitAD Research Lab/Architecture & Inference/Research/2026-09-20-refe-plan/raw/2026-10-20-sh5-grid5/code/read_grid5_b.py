"""Grid-5 replicate controls (PREREG_GRID5 s4), wd attribution (s6), sensitivity readings, probe-5b table. CPU, stdlib only.
Reads banked JSON, writes raw/grid5_controls.json + raw/grid5_controls.txt."""
import json

ROOT = "C:/Users/Admin/par18/sh5diag"
W = ROOT + "/work"
FLOOR = 0.030
g5 = {s: json.load(open(f"{W}/diag5/diag_G5_s{s}.json")) for s in (0, 1, 2)}


def at(run_json, tag, cyc):
    for e in run_json["curve"][tag]:
        if e["cycle"] == cyc:
            return e
    raise KeyError((tag, cyc))


L = []
O = {}
# --- reference values re-read from the banked raw JSON of the earlier grids, and checked against the numbers PREREG_GRID5 s4 quotes
ref = {}
for s in (0, 1):
    w60 = json.load(open(f"{W}/diag4/diag_W60_s{s}.json"))
    s12 = json.load(open(f"{W}/diag3/diag_S12_s{s}.json"))
    ref[("BEST", s)] = at(w60, "BEST_60ep", 15)
    ref[("base", s)] = at(w60, "base_k8o8_60ep", 15)
    ref[("cos", s)] = at(s12, "cos16_k8o8", 3)
PREREG = {("BEST", 0): (0.846, 0.842), ("BEST", 1): (0.562, 0.597), ("base", 0): (0.653, 0.715),
          ("base", 1): (0.609, 0.679), ("cos", 0): (0.611, 0.698), ("cos", 1): (0.523, 0.614)}
L.append("REFERENCE values re-read from raw JSON vs the numbers PREREG_GRID5 s4 quotes (F / ALLQ)")
for k, (f, a) in PREREG.items():
    r = ref[k]
    ok = abs(r["auc_F"] - f) < 0.0006 and abs(r["auc_all_strata"] - a) < 0.0006
    L.append(f"  {k[0]} s{k[1]}: raw F {r['auc_F']:.4f} ALLQ {r['auc_all_strata']:.4f} | prereg {f:.3f}/{a:.3f} | match {ok}")
    O.setdefault("reference_check", {})[f"{k[0]}_s{k[1]}"] = {"raw_F": r["auc_F"], "raw_ALLQ": r["auc_all_strata"], "prereg_F": f, "prereg_ALLQ": a, "match": ok}


def ctl(name, tag, cyc, key):
    rows = []
    for s in (0, 1):
        e = at(g5[s], tag, cyc)
        r = ref[(key, s)]
        dF, dA = e["auc_F"] - r["auc_F"], e["auc_all_strata"] - r["auc_all_strata"]
        rows.append({"seed": s, "grid5_F": e["auc_F"], "grid5_ALLQ": e["auc_all_strata"], "grid5_R2": e["r2_reg"],
                     "ref_F": r["auc_F"], "ref_ALLQ": r["auc_all_strata"], "ref_R2": r["r2_reg"], "dF": dF, "dALLQ": dA,
                     "F_beyond_floor": abs(dF) > FLOOR, "ALLQ_beyond_floor": abs(dA) > FLOOR})
    O[name] = rows
    L.append(f"\n{name}  (grid5 {tag} c{cyc} vs the earlier banked cell; floor {FLOOR} was measured on ALLQ, applied to F as an extension)")
    for x in rows:
        L.append(f"  s{x['seed']}: grid5 F {x['grid5_F']:.3f} ALLQ {x['grid5_ALLQ']:.3f} R2 {x['grid5_R2']:.3f} | ref F {x['ref_F']:.3f} ALLQ {x['ref_ALLQ']:.3f} R2 {x['ref_R2']:.3f}"
                 f" | dF {x['dF']:+.3f} ({'BEYOND' if x['F_beyond_floor'] else 'within'}) dALLQ {x['dALLQ']:+.3f} ({'BEYOND' if x['ALLQ_beyond_floor'] else 'within'})")


ctl("C1_BEST_120ep_c15_vs_W60_BEST", "BEST_120ep", 15, "BEST")
ctl("C2_base_k8o8_120ep_c15_vs_W60_base", "base_k8o8_120ep", 15, "base")
ctl("C3_cos16_k8o8_120ep_c3_vs_grid3", "cos16_k8o8_120ep", 3, "cos")
f15 = at(g5[0], "BEST_120ep", 15)["auc_F"]
O["C1_prereg_rule"] = {"s0_F_c15": f15, "NOT_reproducible": f15 < 0.75}
L.append(f"\nC1 prereg rule: s0 c15 F < 0.75 => 'W60 s0 transition NOT reproducible'. grid5 s0 c15 F = {f15:.3f} => NOT_reproducible = {f15 < 0.75}")
O["C4_F12"] = "UNAVAILABLE: work/diag5f crashed before any eval (CUDA OOM in layer_norm, 20 MiB request, 3.58 GiB reported free, after 'cycle 1/3 shard 1/2' loaded); no diag_F12_*.json exists"
L.append("\nC4 BEST_lr3e-3_rep vs W12 / BEST_fp32 vs rep: UNAVAILABLE (F12 never evaluated; work/diag5f has no json)")

# --- wd attribution: wd1 vs plain cos16_k8o8, same seed
L.append("\nWD ATTRIBUTION: cos16_k8o8_wd1_120ep minus cos16_k8o8_120ep, same seed, floor 0.030")
O["wd_attribution"] = []
for s, cyc in ((0, 30), (1, 30), (2, 12)):
    a, b = at(g5[s], "cos16_k8o8_wd1_120ep", cyc), at(g5[s], "cos16_k8o8_120ep", cyc)
    d = {"seed": s, "cycle": cyc, "dF": a["auc_F"] - b["auc_F"], "dALLQ": a["auc_all_strata"] - b["auc_all_strata"],
         "dR2": a["r2_reg"] - b["r2_reg"], "complete_run": s != 2}
    O["wd_attribution"].append(d)
    L.append(f"  s{s} c{cyc}{'' if s != 2 else ' (PARTIAL run, crashed after c12)'}: dF {d['dF']:+.3f} dALLQ {d['dALLQ']:+.3f} dR2 {d['dR2']:+.3f}")
L.append("  context, wd1 minus plain at c12 on every seed:")
for s in (0, 1, 2):
    a, b = at(g5[s], "cos16_k8o8_wd1_120ep", 12), at(g5[s], "cos16_k8o8_120ep", 12)
    L.append(f"    s{s} c12: wd1 F {a['auc_F']:.3f} ALLQ {a['auc_all_strata']:.3f} | plain F {b['auc_F']:.3f} ALLQ {b['auc_all_strata']:.3f} | dF {a['auc_F'] - b['auc_F']:+.3f}")


# --- sensitivity for G3: ANY head (no plant-mass condition) with beta>=16 and top1>=0.5
def anyhead(e):
    return [(l["layer"], h) for l in e["attn"] for h in range(len(l["beta_eff"]))
            if l["beta_eff"][h] >= 16 and l["top1_within_0p5m"][h] >= 0.5]


L.append("\nG3 SENSITIVITY (drops the plant-mass>=0.8 condition): heads with beta_eff>=16 AND top1>=0.5")
O["g3_sensitivity"] = {}
for tag in ("cos16_k8o8_120ep", "cos16_k8o8_wd1_120ep", "BEST_120ep", "base_k8o8_120ep"):
    d = {}
    for s in (0, 1):
        d[f"s{s}_c30"] = anyhead(at(g5[s], tag, 30))
        d[f"s{s}_evals_with_such_head"] = [e["cycle"] for e in g5[s]["curve"][tag] if anyhead(e)]
    d["both_seeds_c30"] = bool(d["s0_c30"] and d["s1_c30"])
    d["both_seeds_any_eval"] = bool(d["s0_evals_with_such_head"] and d["s1_evals_with_such_head"])
    O["g3_sensitivity"][tag] = d
    L.append(f"  {tag}: s0 c30 {d['s0_c30']} (evals {d['s0_evals_with_such_head']}) | s1 c30 {d['s1_c30']} (evals {d['s1_evals_with_such_head']})"
             f" | both seeds: c30 {d['both_seeds_c30']}, any eval {d['both_seeds_any_eval']}")

# --- stability of the two single-seed passes across eval cycles (context, not a criterion)
L.append("\nSTABILITY of the two single-seed passes across the last six eval cycles (prereg rule = final cycle only; this is context)")
PF, PA, PR = 0.97, 0.95, 0.50
for tag, s in (("BEST_120ep", 0), ("cos16_k8o8_wd1_120ep", 1)):
    row = []
    for e in g5[s]["curve"][tag]:
        row.append((e["cycle"], round(e["auc_F"], 3), round(e["auc_all_strata"], 3), round(e["r2_reg"], 3),
                    bool(e["auc_F"] >= PF and e["auc_all_strata"] >= PA and e["r2_reg"] >= PR)))
    O[f"stability_{tag}_s{s}"] = row
    L.append(f"  {tag} s{s}: (cycle, F, ALLQ, R2, clears the three level bars) " + " ".join(str(x) for x in row[-6:]))

# --- probe 5b
P = json.load(open(f"{ROOT}/raw/probe5b_grid5_ckpt.json"))["ckpts"] + json.load(open(f"{ROOT}/raw/probe5b_grid5_ckpt_idx7.json"))["ckpts"]
L.append("\nPROBE 5b (CPU fp32; 48 TRAIN tokens, first 24 fold-0 tokens): train F, train F under each ablation, fold-0 (24 tok) F, attn_reproduces")
O["probe5b"] = []
for c in P:
    d = {"run": c["run"], "tag": c["tag"], "attn_reproduces": c["attn_reproduces"], "train_bce": c["full"]["train"]["bce"],
         "train_F": c["full"]["train"]["auc_F"], "train_ALLQ": c["full"]["train"]["auc_all_strata"],
         "train_F_zero_clear": c["ablate_zero_clear"]["train"]["auc_F"], "train_F_shuffle_clear": c["ablate_shuffle_clear"]["train"]["auc_F"],
         "train_F_no_plant": c["ablate_no_plant"]["train"]["auc_F"],
         "fold0_24tok_F": c["full"]["fold0"]["auc_F"], "fold0_24tok_F_no_plant": c["ablate_no_plant"]["fold0"]["auc_F"]}
    O["probe5b"].append(d)
    L.append(f"  {d['run']} {d['tag']}: train F {d['train_F']:.3f} (bce {d['train_bce']:.3f}) | zero_clear {d['train_F_zero_clear']:.3f}"
             f" shuffle_clear {d['train_F_shuffle_clear']:.3f} no_plant {d['train_F_no_plant']:.3f}"
             f" | f0(24tok) F {d['fold0_24tok_F']:.3f} no_plant {d['fold0_24tok_F_no_plant']:.3f} | reproduces {d['attn_reproduces']}")
json.dump(O, open(f"{ROOT}/raw/grid5_controls.json", "w"), indent=1)
open(f"{ROOT}/raw/grid5_controls.txt", "w").write("\n".join(L) + "\n")
print("\n".join(L))
