"""render_result.py -- RESULT.json / RESULT.md from anticheat_score.py's RESULT_RAW.json (pure json -> text; no recomputation).
NEW file (EvalFlyWheel 2026-10-27).  No clip id exists in the input; none is written.

    python render_result.py --raw <RESULT_RAW.json> --meta <RESULT_META.json> --out <dir> [--handoff-note "..."]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PRIM_COLS = [("S2", "ade_m"), ("S6", "ade_m"), ("S2", "LON_speed_mae_mps"), ("S2", "LAT_heading_mae_deg"),
             ("S2", "TAC_traj_lat_correct"), ("S2", "TAC_traj_lon_correct"), ("S6", "TAC_traj_lat_correct"), ("S6", "TAC_traj_lon_correct")]
FAM_OF_COL = {("S2", "ade_m"): "ADE", ("S6", "ade_m"): "ADE", ("S2", "LON_speed_mae_mps"): "LONGITUDINAL", ("S2", "LAT_heading_mae_deg"): "LATERAL",
              ("S2", "TAC_traj_lat_correct"): "TACTICAL", ("S2", "TAC_traj_lon_correct"): "TACTICAL", ("S6", "TAC_traj_lat_correct"): "TACTICAL", ("S6", "TAC_traj_lon_correct"): "TACTICAL"}


def fmt(c, nd=4):
    if not c or c.get("delta") is None:
        return "n/a"
    return f"{c['delta']:+.{nd}f} [{c['lo']:+.{nd}f}, {c['hi']:+.{nd}f}]"


def status(c, lib):
    if not c or c.get("lo") is None:
        return "?"
    if lib:
        return "BETTER" if c["hi"] < 0 else ("WORSE" if c["lo"] > 0 else "n.s.")
    return "BETTER" if c["lo"] > 0 else ("WORSE" if c["hi"] < 0 else "n.s.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--handoff-note", default="")
    a = ap.parse_args()
    R = json.loads(Path(a.raw).read_text(encoding="utf-8"))
    M = json.loads(Path(a.meta).read_text(encoding="utf-8"))
    lib, label, unit = M["lower_is_better"], M["label"], M["unit"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    L = []
    w = L.append
    seeds = sorted(R["seeds"])
    w("# RESULT -- refcv8 (EMA-48,852 + shipped H2cn stage) vs CAMERA-BLIND (W4) and EGO-ONLY FLOOR (W5)\n")
    w(f"**Tier** {R['tier']}.  **Estimator** {R['estimator']}.  ")
    w(f"**Prereg** `PREREG_ANTICHEAT_REFCV8.md` sha256 `{R['prereg_sha256']}` (sealed before any score; `PREREG_ANTICHEAT_REFCV8.sha256`).  ")
    w("**Disclosures** " + "; ".join(R["disclosures"]) + ".\n")
    w(f"**W4 status:** {R['w4_status']}.  **W5 status:** {R['w5_status']}.\n")
    if a.handoff_note:
        w(a.handoff_note + "\n")
    for s in seeds:
        v = R["verdicts"][s]
        w(f"## Inference seed {s}\n")
        w("Cells are `new - ref` on the ceiling-compliant windows, paired episode-cluster bootstrap (B 2000, seed 0). REAL better = delta < 0 on error rows, delta > 0 on TAC rows; "
          "`BETTER`/`WORSE` = the 95 % CI excludes 0 on that side, `n.s.` = not separated.\n")
        contrasts = [("W4  REAL_OS - CAMERA-BLIND  (binding)", "REAL_OS-BLIND"), ("W5  REAL_REG - EGO-ONLY FLOOR  (binding)", "REAL_REG-FLOOR"),
                     ("beside  REAL_REG - CAMERA-BLIND", "REAL_REG-BLIND"), ("beside  REAL_OS - EGO-ONLY FLOOR", "REAL_OS-FLOOR"),
                     ("beside  REAL_REG - CONST (train-mean trajectory)", "REAL_REG-CONST"), ("beside  REAL_REG - FLOOR_NL (boosted, same features)", "REAL_REG-FLOOR_NL"),
                     ("beside  REAL_REG - ha0_ext_pose (kinematic prior)", "REAL_REG-HP"), ("beside  FLOOR - CONST", "FLOOR-CONST"), ("beside  FLOOR - ha0_ext_pose", "FLOOR-HP"),
                     ("draw floor  REAL_REG - REAL_OS (battery-tree plan vs os-only plan)", "REAL_REG-REAL_OS"), ("beside  RAW_REAL - RAW_BLIND (no stage)", "RAW_REAL-RAW_BLIND"),
                     ("beside  BLIND - FLOOR", "BLIND-FLOOR")]
        hdr = "| contrast | " + " | ".join(f"{sf} `{k}`" for sf, k in PRIM_COLS) + " |"
        w(hdr)
        w("|" + "---|" * (len(PRIM_COLS) + 1))
        for title, key in contrasts:
            ck = f"{key}|compliant"
            if ck not in R["seeds"][s]["surfaces"]["S2"]["contrasts"]:
                continue
            cells = []
            for sf, k in PRIM_COLS:
                c = R["seeds"][s]["surfaces"][sf]["contrasts"][ck][k]
                cells.append(f"{fmt(c, 4 if k != 'ade_m' else 4)} {status(c, lib[k])}")
            w(f"| {title} | " + " | ".join(cells) + " |")
        w("")
        for name in ("W4_REAL_OS_vs_BLIND", "W5_REAL_REG_vs_FLOOR", "beside_REAL_REG_vs_BLIND"):
            if name not in v:
                continue
            fv = v[name]
            w(f"**{name}**  OVERALL: **{fv['OVERALL']}**  ")
            for fam in ("ADE", "LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC"):
                rows = ", ".join(f"{k}={x}" for k, x in fv[fam]["rows"].items())
                w(f"* {fam}: **{fv[fam]['status']}**" + (f"  ({rows})" if rows else ""))
            w("")
        w("### Arm means on the compliant set (ade_m, m)\n")
        w("| arm | S2 ade_m | S6 ade_m |")
        w("|---|---|---|")
        for arm in R["seeds"][s]["surfaces"]["S2"]["means"]:
            w(f"| {arm} | {R['seeds'][s]['surfaces']['S2']['means'][arm]['ade_m']:.4f} | {R['seeds'][s]['surfaces']['S6']['means'][arm]['ade_m']:.4f} |")
        w("")
        for sf in ("S2", "S6"):
            sr = R["seeds"][s]["surfaces"][sf]
            w(f"n {sf}: {sr['n_compliant']} compliant windows / {sr['n_episodes_compliant']} episodes (all: {sr['n_all']} / {sr['n_episodes_all']}).")
        w("")
        for title, key in (("W4  REAL_OS - CAMERA-BLIND, EVERY registered row (compliant)", "REAL_OS-BLIND"), ("W5  REAL_REG - EGO-ONLY FLOOR, EVERY registered row (compliant)", "REAL_REG-FLOOR")):
            ck = f"{key}|compliant"
            if ck not in R["seeds"][s]["surfaces"]["S2"]["contrasts"]:
                continue
            w(f"### {title}\n")
            w("| family | surface | metric [unit] | REAL | other | delta [95 % CI] | |")
            w("|---|---|---|---|---|---|---|")
            for k in M["keys"]:
                for sf in ("S2", "S6"):
                    c = R["seeds"][s]["surfaces"][sf]["contrasts"][ck][k]
                    w(f"| {M['family'][k]} | {sf} | {label[k]} [{unit[k]}] | {c['mean_new']:.5g} | {c['mean_ref']:.5g} | {fmt(c, 4)} | {status(c, lib[k])} |")
            w("")
    w("## The EGO-ONLY floor (fit on TRAIN windows only)\n")
    w("| surface | n train windows | episodes | d | parametrisation | lambda | CV SSE / window (m^2) | train R^2 (positions) |")
    w("|---|---|---|---|---|---|---|---|")
    for sf in ("S2", "S6"):
        f = R["floor"][sf]
        w(f"| {sf} | {f['n_train_windows']} | {f['n_train_episodes']} | {f['d_features']} | {f['param']} | {f['lambda']:g} | {f['cv_best_sse_m2_per_window']:.4f} | {f['train_R2_positions']:.4f} |")
    w("\nFeatures: " + R["floor"]["S2"]["features"] + ".  Train set: " + R["floor"]["S2"]["train_set"] + ".  Function class: ridge (linear in these features).  "
      "FLOOR_NL (HistGradientBoostingRegressor, same features) is beside.  Tactical-HEAD (logit) rows: n = 0 for a ridge floor (no class logits); the TAC rows above are trajectory-derived. STRATEGIC: n = 0.\n")
    w("## Controls (verbatim)\n")
    w("```")
    c = R["controls"]
    w(f"floor_known_values (K-FL1 recovers v0, K-FL2 shuffled target): {c.get('floor_known_values')}  | "
      + "; ".join(f"{sf}: K_FL1 R2={R['floor'][sf]['known_value_controls']['K_FL1_recovers_v0_R2']:.12f} K_FL2 R2={R['floor'][sf]['known_value_controls']['K_FL2_shuffled_heldout_R2']:.5f}" for sf in ("S2", "S6")))
    w(f"K_BASE_all_positions_bit_equal (REAL_REG == landed H2CN_DEPLOY8 package): {c.get('K_BASE_all_positions_bit_equal')}  | " + json.dumps(c.get("K_BASE_positions")))
    w(f"K_BASE_cells_equal (E1/E3 reproduce landed): {c.get('K_BASE_cells_equal')}  | " + json.dumps(c.get("K_BASE_cells")))
    w(f"K_ID (REAL-vs-REAL = 0 with zero-width CI; GT TAC = 1.0, ADE = 0): {c.get('K_ID')}")
    if "w4_roll_controls" in c and c["w4_roll_controls"]:
        w("W4 roll controls: " + json.dumps(c["w4_roll_controls"]))
        w("derangement: " + json.dumps(c.get("derangement")))
        w("(a) identity vs BASE bank: " + json.dumps(c.get("a_identity_vs_BASE_bank")))
        w("(b) changed fraction vs identity roll: " + json.dumps(c.get("b_changed_fraction_vs_identity")))
        w("(b) own-frame sample inside the blind roll: " + json.dumps(c.get("b_own_sample")))
        w("(c) ego inputs equal vs identity roll: " + json.dumps(c.get("c_ego_inputs_equal_vs_identity")))
    w("```")
    (out / "RESULT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / "RESULT.json").write_text(json.dumps(R, indent=1), encoding="utf-8")
    print(f"WROTE {out / 'RESULT.md'} ZZRENDERDONEZZ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
