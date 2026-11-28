"""l2_oracle_s2_rows.py -- the S2 (0.5-2 s, 4-instant) four-family rows for the L2 ORACLE-class arm vs BASE, from banked arrays only (0 GPU, CPU).

Why: the L2 oracle arm (force the GT `lon_v7` label at the lon-head seam) cuts missed-decel 0.470 -> 0.341 on the S6 surface; this script records what that
gain COSTS or BUYS on the S2 panel (ADE/FDE at 0.5-2 s, longitudinal, lateral, tactical rows), so it is on record beside the MD number.

Surface: the S2 panel of the banked EMA-48,852 per-frame battery (4,754 windows, 139 episodes; S2-compliant 4,159 = the E1 surface; also ALL).  The oracle trace
(`bank/trace_oracle_s0.npz`) covers all 4,754 windows (the roll used windows_s0.json), so no window is lost.  Plans: oracle / BASE-os (L1's un-hooked os-only roll,
the SAME code path) at seed 0; BASE-bat (the battery's own `os`) as the second reference; BASE-os seed 1 vs seed 0 = the inference-draw FLOOR for every cell.
The shipped stage (DEPLOY8 per-frame table) is applied to the full 8-slot plan and sliced to the S2 slots [0,1,2,3] (S2_prior8 from the banked deploy npz).

TIER: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance).  ESTIMATOR: paired episode-cluster bootstrap (B 2000, seed 0, over EPISODES)
= 'another draw of episodes' only; NOT another training run, NOT another inference run (the floor block shows the latter).  EVIDENCE CLASS: MEASURED.
Not computed: distance-keeping and STRATEGIC (no lead track / --no-strategic, n = 0); false/missed-decel (defined on the 6 s horizon only, see the S6 rows).

    python l2_oracle_s2_rows.py --out <dir>
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import l2_common as C  # noqa: E402
import l2_score as S  # noqa: E402

L1_BANK = "C:/Users/Admin/refcv8_pf/refcv9_l1/bank"
S2_SLOTS = [0, 1, 2, 3]
KEYS = ("ade_m", "fde_m", "LON_speed_mae_mps", "LON_along_mae_m", "LON_along_abs_final_m", "LON_speed_within_0p5", "LON_accel_mae_mps2", "LAT_heading_mae_deg",
        "LAT_curvature_mae_1pm_valid", "LAT_yaw_rate_mae_radps_valid", "LAT_cross_mae_m", "LAT_cross_abs_final_m", "TAC_traj_lat_correct", "TAC_traj_lon_correct")


def load_panel(seed):
    root = C.ARCH / C.EVAL_TREE
    z = np.load(C.DEPLOY_DIR / f"DEPLOY8_POS_{C.EVAL_TREE}_s{seed}.npz", allow_pickle=True)
    files = sorted(glob.glob(str(root / f"panel_s{seed}" / "ep*.npz")))
    g, v0, os_, echo, ws, clip, eid = [], [], [], [], [], [], []
    for f in files:
        a = np.load(f)
        g.append(a["g"]); v0.append(a["v0"]); os_.append(a["os"]); echo.append(a["ha0_ext"]); ws.append(a["ws"])
        clip += [int(a["clip_index"][0])] * len(a["ws"])
    P = {"g": np.concatenate(g).astype(np.float64), "v0": np.concatenate(v0).astype(np.float64), "bat_os": np.concatenate(os_).astype(np.float32),
         "echo": np.concatenate(echo).astype(np.float32), "ws": np.concatenate(ws).astype(np.int64), "clip": np.asarray(clip, np.int64)}
    chk = {"n": int(len(P["ws"])), "ws_equal_deploy": bool(np.array_equal(P["ws"], z["S2_ws"])), "clip_equal_deploy": bool(np.array_equal(P["clip"], z["S2_clip"])),
           "bat_os_equal_deploy_S2_os": bool(np.array_equal(P["bat_os"], z["S2_os"])),
           "bat_os_equals_plan8_slots": float(np.abs(z["S2_plan8"][:, S2_SLOTS] - P["bat_os"]).max())}
    P.update(eid=np.asarray(z["S2_eid"]).astype(np.int64), compliant=np.asarray(z["S2_compliant"], bool), plan8=z["S2_plan8"], prior8=z["S2_prior8"],
             stage_banked=z["S2_pos"], dt=0.5, checks=chk)
    return P


def join(tr, P):
    j = np.array([tr["idx"].get(k, -1) for k in zip(P["clip"].tolist(), P["ws"].tolist())])
    if (j < 0).any():
        raise SystemExit(f"[s2rows] {int((j < 0).sum())} S2 windows have no entry in the trace")
    return tr["plan8"][j]


def cells(P, plan4, mask):
    c = C.arm_metrics(P, plan4)                      # uses P['g'], P['dt']
    return {k: np.where(mask, np.asarray(c[k], np.float64), np.nan) for k in KEYS if k in c}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bank", default=str(Path(__file__).resolve().parent.parent / "bank"))
    a = ap.parse_args()
    t0 = time.time()
    st = C.stack()
    PR = st["PR"]
    rows = [r for r in S.metric_rows(PR) if r[1] in KEYS]
    P0, P1 = load_panel(0), load_panel(1)
    tr_or = S.load_trace(Path(a.bank) / "trace_oracle_s0.npz")
    b0, b1 = S.load_l1_base(L1_BANK, 0), S.load_l1_base(L1_BANK, 1)
    for nm, t in (("oracle trace", tr_or), ("BASE-os s0", b0), ("BASE-os s1", b1)):
        if t is None or t.get("incomplete"):
            raise SystemExit(f"[s2rows] {nm} missing or incomplete")
    plan = {"oracle": join(tr_or, P0), "BASE-os": join(b0, P0), "BASE-os_s1": join(b1, P1)}
    table = C.deploy_table()

    def stage4(plan8, prior8):
        return C.stack()["RS"].retime_plan(np.asarray(plan8), np.asarray(prior8), C.ALL_T, table, cast=True)[:, S2_SLOTS]
    # ---- controls with known values
    ctl = {"alignment": P0["checks"], "alignment_s1": P1["checks"]}
    ctl["K1_S2_stage_bit_equal_banked"] = bool(np.array_equal(stage4(P0["plan8"], P0["prior8"]), P0["stage_banked"]))
    ctl["K2_S2_compliant_n"] = {"n": int(P0["compliant"].sum()), "literal": 4159, "pass": int(P0["compliant"].sum()) == 4159}
    ctl["K3_oracle_trace_final_n"] = {"n": int(tr_or["n"]), "literal": 4754, "pass": int(tr_or["n"]) == 4754}
    res = {"tool": "l2_oracle_s2_rows.py", "tier": "T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)",
           "estimator": "paired episode-cluster bootstrap (pair_read_r8.paired_cell -> taniteval.ci), B 2000, seed 0, over EPISODES: 'another draw of episodes' only; NOT another training run, NOT another inference run (floor block)",
           "evidence_class": "MEASURED (bank/trace_oracle_s0.npz, refcv9_l1/bank/ztac_eval_s{0,1}.npz, battery panel_s{0,1})", "controls": ctl, "surfaces": {},
           "families_not_computed": {"LONGITUDINAL distance-keeping": "no lead track in the banked dumps; n = 0", "STRATEGIC": "--no-strategic; n = 0",
                                    "false/missed-decel": "defined on the 6 s horizon (last chord 5->6 s): see L2_RESULT.md S6 rows"}}
    for surf, mask in (("S2_compliant", P0["compliant"]), ("S2_all", np.ones(len(P0["ws"]), bool))):
        Em = dict(P0, compliant=mask)
        block = {"n_windows": int(mask.sum()), "n_episodes": int(len(set(P0["eid"][mask].tolist()))), "stage": {}}
        for sk in ("stage", "nostage"):
            def p4(name, P=P0):
                p8 = plan[name]
                return (stage4(p8, P["prior8"]) if sk == "stage" else p8[:, S2_SLOTS])
            A_or = cells(Em, p4("oracle"), mask)
            A_bs = cells(Em, p4("BASE-os"), mask)
            A_bb = cells(Em, (stage4(P0["plan8"], P0["prior8"]) if sk == "stage" else P0["bat_os"]), mask)
            # the inference-draw floor: BASE-os seed 1 vs seed 0 (same windows; seed-1 plans staged with seed-0's S2 prior8 = same window prior)
            A_b1 = cells(Em, (stage4(plan["BASE-os_s1"], P1["prior8"]) if sk == "stage" else plan["BASE-os_s1"][:, S2_SLOTS]), mask)
            echo = cells(Em, P0["echo"], mask)
            idc = S.compare(PR, Em, A_bs, A_bs, rows)
            ctl.setdefault("K4_identity_BASEos_vs_itself_max_abs_delta", {})[f"{surf}|{sk}"] = float(max(abs(v["delta"]) for v in idc.values() if "delta" in v))
            d = {"oracle_vs_BASE-os": S.compare(PR, Em, A_or, A_bs, rows), "oracle_vs_BASE-bat": S.compare(PR, Em, A_or, A_bb, rows),
                 "FLOOR_BASE-os_s1_vs_s0": S.compare(PR, Em, A_b1, A_bs, rows),
                 "E1_style_ADE_minus_echo": {nm: PR.paired_cell(A["ade_m"], echo["ade_m"], list(Em["eid"])) for nm, A in (("oracle", A_or), ("BASE-os", A_bs), ("BASE-bat", A_bb))}}
            for nm in d["E1_style_ADE_minus_echo"]:
                c = d["E1_style_ADE_minus_echo"][nm]
                d["E1_style_ADE_minus_echo"][nm] = {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in c.items() if k in ("delta", "lo", "hi", "n_windows", "n_episodes", "separated")}
            block["stage"][sk] = d
        res["surfaces"][surf] = block
    res["wall_s"] = round(time.time() - t0, 1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    C.jdump(res, out / "ORACLE_S2_ROWS.json")
    # ---- markdown (primary: S2_compliant, stage, oracle vs BASE-os, with the floor beside it)
    L = ["# L2 ORACLE-class arm vs BASE -- S2 (0.5-2 s) four-family rows", "",
         f"tier: {res['tier']}", f"estimator: {res['estimator']}", "evidence class: MEASURED. Controls: " + json.dumps({k: (v if not isinstance(v, dict) else v.get('pass', v)) for k, v in ctl.items() if k != 'alignment' and k != 'alignment_s1'}),
         "Not computed: " + "; ".join(f"{k} ({v})" for k, v in res["families_not_computed"].items()), ""]
    for surf in ("S2_compliant", "S2_all"):
        b = res["surfaces"][surf]
        for sk in ("stage", "nostage"):
            d = b["stage"][sk]
            L += [f"## {surf} ({b['n_windows']} windows / {b['n_episodes']} episodes), {sk}: oracle vs BASE-os (delta = oracle - BASE-os; FLOOR = BASE-os s1 - s0)", "",
                  "| family | metric | oracle | BASE-os | delta [lo, hi] | sep | worse | FLOOR delta [lo, hi] |", "|---|---|---|---|---|---|---|---|"]
            for fam, k, lab, unit, lower in rows:
                c, f = d["oracle_vs_BASE-os"].get(k), d["FLOOR_BASE-os_s1_vs_s0"].get(k)
                if not c or "delta" not in c:
                    continue
                fl = f"{f['delta']:+.4f} [{f['lo']:+.4f}, {f['hi']:+.4f}]" if f and "delta" in f else "-"
                L.append(f"| {fam} | {lab} ({unit}) | {c['mean_new']:.4f} | {c['mean_ref']:.4f} | {c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}] | {c['separated']} | {c['worse']} | {fl} |")
            e1 = d["E1_style_ADE_minus_echo"]
            L += ["", "E1-style ADE(0-2 s) minus echo `ha0_ext`: " + "; ".join(f"{n} {v['delta']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]" for n, v in e1.items()), ""]
    (out / "ORACLE_S2_ROWS.md").write_text("\n".join(L), encoding="utf-8")
    print(json.dumps({"controls": {k: v for k, v in ctl.items() if k not in ("alignment", "alignment_s1")}, "wall_s": res["wall_s"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
