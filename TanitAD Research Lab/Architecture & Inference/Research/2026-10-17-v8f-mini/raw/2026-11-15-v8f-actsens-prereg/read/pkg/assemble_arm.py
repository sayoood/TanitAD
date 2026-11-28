"""Assemble ONE arm's compact read JSON (the only input verdict_actsens.py reads per arm). Tier T0 (oracle goal; a world-model diagnostic, never driving performance). Evidence MEASURED.
usage: assemble_arm.py <arm_out_dir> <name> <chassis 0|1> <out.json>
Reads <dir>/ia_read.json, contrast_hwm.json, contrast_flat.json, c6.json, decode_check/result_<name>.json (optional: absent -> recorded as missing, not silently dropped).
Sign convention (hwm_eval.contrast): delta = mean(A) - mean(B), LOWER IS BETTER. vs_r35 = arm - R35_k; vs_hold = arm - HOLD; rep_arm / rep_r35 = inference-seed-1 minus seed-0 of the arm / of R35_k."""
import json, os, sys, hashlib
d, name, chassis, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
if os.path.exists(out):
    sys.exit(f"REFUSED: {out} exists")
J = lambda p: json.load(open(p, encoding="utf-8"))
need = ["ia_read.json", "contrast_hwm.json", "contrast_flat.json", "c6.json"]
miss = [n for n in need if not os.path.exists(f"{d}/{n}")]
if miss:
    sys.exit(f"FAIL-CLOSED: missing inputs {miss}")
KEEP = ("delta", "lo", "hi", "label", "n_windows", "n_episodes")
pick = lambda r: {k: r[k] for k in KEEP if k in r}


def cells_of(c):
    cells = {}
    for r in c["rows"]:
        if r["kind"] != "contrast":
            continue
        key = f"{r['family']}|{r['band']}|{r['metric']}"
        slot = {("A", "B"): "vs_r35", ("A", "HOLD"): "vs_hold", ("B", "HOLD"): "r35_vs_hold"}[(r["a"], r["b"])]
        cells.setdefault(key, {})[slot] = pick(r)
    for r in c["replicate"]:
        key = f"{r['family']}|{r['band']}|{r['metric']}"
        cells.setdefault(key, {})["rep_arm" if r["arm"] == "A" else "rep_r35"] = pick(r)
    return cells


ia = J(f"{d}/ia_read.json")
ch, cf, c6 = J(f"{d}/contrast_hwm.json"), J(f"{d}/contrast_flat.json"), J(f"{d}/c6.json")
dc = J(f"{d}/decode_check/result_{name}.json") if os.path.exists(f"{d}/decode_check/result_{name}.json") else None
i3 = ia["beside"]["ia3"]
res = {"tier": "T0", "evidence": "MEASURED", "name": name, "chassis": chassis, "lever": name.split("_")[0],
       "strategic": "PAUSED (PI item 25, 2026-09-19): not a pass, not omitted",
       "n_windows": ch["n_windows"], "n_episodes": ch["n_episodes"], "estimator": "paired episode-cluster bootstrap, B 2000, seed 0 (answers: another draw of episodes)",
       "ia": {"zero_act": ia["zero_act"], "wm_gamma_levels": ia.get("wm_gamma_levels"), "ckpt_step": ia["ckpt_step"], "argv_sha256": ia["argv_sha256"], "adapter_sha256": ia["adapter_sha256"],
              "ia3": {k: {"gain_lon": v["gain_lon"], "gain_lat": v["gain_lat"], "cos": v["cos"], "n": v["n"]} for k, v in i3.items()},
              "band_met": ia["passes"], "ia0_skill": {lv: ia["passes"][lv].get("skill") for lv in ia["passes"]}},
       "hwm": {"cells": cells_of(ch), "tactical_kappa": ch["tactical_kappa"]},
       "flat": {"cells": cells_of(cf), "tactical_kappa": cf["tactical_kappa"]},
       "c6": c6, "decode_check": None if dc is None else {k: dc.get(k) for k in ("step3_nn", "step2_repro", "step2_planned_redecode_300it", "step1_inimage_n1_holdA0_300it")},
       "decode_check_present": dc is not None,
       "source_files": {n: hashlib.sha256(open(f"{d}/{n}", "rb").read()).hexdigest() for n in need}}
json.dump(res, open(out + ".tmp", "w", encoding="utf-8"), indent=1)
os.replace(out + ".tmp", out)
print("ASSEMBLED", out, "cells", len(res["hwm"]["cells"]))
