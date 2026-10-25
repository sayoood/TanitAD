"""plans_pack.py -- the banked per-window plans of the 4 windows (segment 4), the eval-wide tables (quoted programmatically from the source JSONs),
and CONTROLS_V2_part2.json.   CPU only, read-only on every source."""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2_lib import *  # noqa

HB, HS_ = f"{SC}/hblend", f"{SC}/hspec"
pack, ctl, tables = {}, {"per_window": {}}, {"sources": {}}
cf = np.load(f"{OUT}/data/cf_windows.npz")
for seed in SEEDS:
    sn = seed.replace("M2b_", "M2b_")
    hb = np.load(f"{HB}/score_{seed}_perwindow.npz")
    hs = np.load(f"{HS_}/score_hs2_{seed}_perwindow.npz")
    kind = np.load(f"{WORK}/banked/eval_{seed}-kind-own.npz")
    model, cfg = load_model(seed)
    ew = EvalWorld(cfg, lru=4)
    assert np.array_equal(hb["eid"], ew.X["eid"]) and np.array_equal(hs["eid"], ew.X["eid"]), "eid mismatch"
    for name in KEYS:
        w = WINDOWS[name]["window"]
        gt = ew.X["gt"][w].astype(np.float64)
        P = f"{seed}/{name}"
        arr = {"GT": gt, "B": hb["tau"][w], "P10": hb["P"][w], "H": hb["H"][w], "HS": hs["HS"][w]}
        for k, v in arr.items():
            pack[f"{P}/{k}"] = v
        d = {}
        # drawn arrays == banked per-window arrays (the pack is what the renderer reads; the source is re-read here)
        d["drawn_B_vs_source_tau"] = float(np.abs(pack[f"{P}/B"] - np.load(f"{HB}/score_{seed}_perwindow.npz")["tau"][w]).max())
        d["drawn_H_vs_source"] = float(np.abs(pack[f"{P}/H"] - np.load(f"{HB}/score_{seed}_perwindow.npz")["H"][w]).max())
        d["drawn_P10_vs_source"] = float(np.abs(pack[f"{P}/P10"] - np.load(f"{HB}/score_{seed}_perwindow.npz")["P"][w]).max())
        d["drawn_HS_vs_source"] = float(np.abs(pack[f"{P}/HS"] - np.load(f"{HS_}/score_hs2_{seed}_perwindow.npz")["HS"][w]).max())
        d["tau_vs_kind_banked_pred"] = float(np.abs(hb["tau"][w] - kind["pred"][w]).max())
        # ADE6 recomputed from the drawn plan + the eval's GT == the banked per-window ADE6 of that plan
        for nm, key, src in (("B", "B__ade6", hb), ("H", "H__ade6", hb), ("P10", "P10__ade6", hb), ("HS", "HS__ade6", hs)):
            a6 = np.linalg.norm(arr[nm].astype(np.float64) - gt, axis=-1).mean()
            d[f"ade6_recomputed_vs_banked_{nm}"] = float(abs(a6 - src[key][w]))
            pack[f"{P}/ade6_{nm}"] = a6
            pack[f"{P}/ade_h_{nm}"] = np.linalg.norm(arr[nm].astype(np.float64) - gt, axis=-1)      # per-step error, 30 steps (0.2 .. 6.0 s)
        d["ade6_B_vs_selection_json"] = float(abs(pack[f"{P}/ade6_B"] - WINDOWS[name][f"ade6_{'seed0' if seed.endswith('s0') else 'seed1'}_banked"]))
        # CPU forward: B (own-code plan) and P10
        b = ew.batch([w])
        o = own_forward(model, b)
        d["cpu_own_traj30_vs_B_tau"] = float(np.abs(o["traj30"][0].numpy() - hb["tau"][w]).max())
        d["cpu_p10_vs_P"] = float(np.abs(o["p10"][0].numpy() - hb["P"][w]).max())
        ctl["per_window"][P] = d
        print(P, {k: f"{v:.1e}" for k, v in d.items()})
np.savez_compressed(f"{OUT}/data/plans_pack.npz", **pack)

# ---------------- eval-wide tables, quoted programmatically
T = {}
for seed in SEEDS:
    hbj = json.load(open(f"{HB}/score_{seed}.json"))
    hsj = json.load(open(f"{HS_}/score_hs2_{seed}.json"))
    t = {"estimator_hblend": hbj["estimator"], "estimator_hspec": hsj["estimator"], "tier": hbj["tier"]}
    t["ade"] = {arm: {f"ade{h}": hbj["single"][arm][f"ade{h}"] for h in range(1, 7)} for arm in ("B", "H", "P10")}
    t["ade"]["HS"] = {f"ade{h}": hsj["single"]["HS"][f"ade{h}"] for h in range(1, 7)}
    t["paired_ade6"] = {"H-B": hbj["paired_H_minus_B"]["ade6"], "H-P10": hbj["paired_H_minus_P10"]["ade6"], "HS-B": hsj["paired"]["HS_minus_B"]["ade6"],
                        "HS-P10": hsj["paired"]["HS_minus_P10"]["ade6"], "HS-H": hsj["paired"]["HS_minus_H"]["ade6"]}
    t["accel"] = {"single": {"B": hbj["single"]["B"]["fw_lon_accel_mae"], "H": hbj["single"]["H"]["fw_lon_accel_mae"], "P10": hbj["single"]["P10"]["fw_lon_accel_mae"],
                             "HS": hsj["single"]["HS"]["fw_lon_accel_mae"]},
                  "H-P10": hbj["paired_H_minus_P10"]["fw_lon_accel_mae"], "HS-P10": hsj["paired"]["HS_minus_P10"]["fw_lon_accel_mae"], "H-B": hbj["paired_H_minus_B"]["fw_lon_accel_mae"],
                  "HS-B": hsj["paired"]["HS_minus_B"]["fw_lon_accel_mae"]}
    t["consistency_any"] = {"rate": {k: hsj["consistency"]["rate"][k]["viol_any"] for k in ("B", "H", "HS", "P10")},
                            "H-B": hbj["consistency_beside"]["paired_H_minus_B"]["viol_any"], "HS-B": hsj["consistency"]["paired"]["HS_minus_B"]["viol_any"],
                            "HS-H": hsj["consistency"]["paired"]["HS_minus_H"]["viol_any"]}
    T[seed] = t
tables["hblend_hspec"] = T
sfn = json.load(open(f"{SC}/sfresh/sfresh_numbers.json"))
tables["sfresh"] = {sd: {"Q_LAYER_50m": sfn["outcome_table"]["Q_LAYER_50m"][sd], "Q_STALE_50m": sfn["outcome_table"]["Q_STALE_50m"][sd],
                         "banked_cycle_minus_navonly_50": sfn["seeds"][sd]["banked_cycle_S_minus_navonly_pos_err"]["@50"],
                         "levels_pos_err_m": sfn["seeds"][sd]["levels_pos_err_m"], "heading": sfn["seeds"][sd]["heading_deg"]["levels"]} for sd in ("s0", "s1")}
tables["sfresh"]["estimator"] = sfn["estimator"]; tables["sfresh"]["tier"] = sfn["tier"]
json.dump(tables, open(f"{OUT}/data/tables.json", "w"), indent=1, default=float)
json.dump(ctl, open(f"{OUT}/CONTROLS_V2_part2.json", "w"), indent=1)
print("ok")
