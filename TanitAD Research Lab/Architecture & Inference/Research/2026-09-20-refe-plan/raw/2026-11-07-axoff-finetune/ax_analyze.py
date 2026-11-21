#!/usr/bin/env python3
"""YAX ANALYSIS: mechanically evaluates the COMMITTED rule of PREREG_AXOFF.md section 5 for one recipe rung (X1 / X2).   CPU only.
    python ax_analyze.py --root <run root> --rung X1|X2 --tokens <w3 token json> [--seeds 0,1] [--smoke]
Arms read: <rung>_AXOFF_s<seed>, <rung>_AXON_s<seed> (eval dirs root/eval/<name>: perwindow.npz, arm.json, seam.npz, families.json). Contrast = AXOFF_s - AXON_s at the SAME seed.
Tier stamp in every output: NAVSIM v1.1 pseudo-simulation of an OPEN-LOOP plan against logged agents -- NOT the action-closed-loop tier. Last line: ZZAXVERDICT <rung> <CANDIDATE|NOT PROVEN|REFUTED|VOID> (...)
"""
import argparse
import csv
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ax_common as C  # noqa: E402
import ax_stats as S  # noqa: E402

sys.path.insert(0, str(C.EVAL))
TIER = ("NAVSIM v1.1 pseudo-simulation (PDMS) and families6 of an OPEN-LOOP plan against logged agents on the 923 A5 confirmation tokens -- NOT the EVAL_DOCTRINE action-closed-loop tier. "
        "Open loop has no ax feedback (the logged ax is the human's), so it shows what removing ax COSTS, never that the stall is fixed. Nothing here may be quoted as 'drives better'.")
TACTICAL = "n/a as a test: REFe has no tactical layer; families6 derives decisions from the executed geometry (descriptive fam_adverse only)"
STRATEGIC = "PAUSED (PI ruling 2026-09-19, PI_DECISION_QUEUE.md item 25): not a pass, not omitted."
VARIANCE = ("Q1 another draw of LOGS: paired log-cluster bootstrap, 10,000 resamples, percentile 95 %, seed 20260927, one resample matrix for all arms. Q2 another TRAINING run: two seeds, floor F = |Delta(s0)-Delta(s1)| "
            "(H-ESTIM-SEED-1). Q3 another INFERENCE run: deterministic planner (ego probe C1, INHERITED). NOT measured: another BASE training run.")
SUB = ["no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort", "driving_direction_compliance", "score"]
MARGIN, LON_BAR, MIN_CRUISE, MIN_CRUISE_LOGS = -0.5, 0.3, 60, 15


def read_csv(p):
    return {r["token"]: r for r in csv.DictReader(open(p, encoding="utf-8")) if r.get("token") not in (None, "", "average")}


def cruise_inputs(toks):
    doc = json.load(gzip.open(C.export_path(), "rt", encoding="utf-8"))["tokens"]
    v0, a0, vg, g2 = [], [], [], []
    for t in toks:
        e = doc[t]
        es = e["ego_statuses"][-1]
        p = np.vstack([[0.0, 0.0], np.asarray(e["human_future_poses"], float)[:, :2]])
        v0.append(math.hypot(*es["ego_velocity"]))
        a0.append(es["ego_acceleration"][0])
        vg.append(np.linalg.norm(p[8] - p[7]) / 0.5)
        g2.append(np.linalg.norm(p[4] - p[3]) / 0.5)
    v0, a0, vg, g2 = map(np.asarray, (v0, a0, vg, g2))
    return v0, a0, vg, g2, (v0 > 5.0) & (np.abs(a0) < 0.3) & (np.abs(vg - v0) < 1.0)


def plan_speed(poses, step):
    p = np.concatenate([np.zeros_like(poses[:, :1, :2]), poses[:, :, :2].astype(np.float64)], 1)
    return np.linalg.norm(p[:, step] - p[:, step - 1], axis=-1) / 0.5


def load_arm(root, name, toks):
    d = Path(root) / "eval" / name
    pw = np.load(d / "perwindow.npz", allow_pickle=False)
    order = {str(t): i for i, t in enumerate(pw["token"])}
    ix = np.asarray([order[t] for t in toks])
    rows = read_csv(C.jload(d / "arm.json")["score"]["csv"])
    z = np.load(d / "seam.npz")
    zi = {str(t): i for i, t in enumerate(z["token"])}
    poses = np.asarray([z["poses"][zi[t]] for t in toks])
    return {"name": name, "pw": {k: pw[k][ix] for k in pw.files}, "csv": rows, "sub": {k: np.asarray([100.0 * float(rows[t][k]) for t in toks]) for k in SUB}, "poses": poses,
            "fam": C.jload(d / "families.json"), "seam_misses": C.jload(d / "seam.report.json").get("misses")}


def decide(gates_ok, gates_red, pdms, d_lon, n_cruise, n_cruise_logs):
    """The committed rule, PREREG section 5. pdms = S.contrast result (AXOFF - AXON); d_lon = per-seed point estimates of |b_OFF| - |b_ON|."""
    if not gates_ok:
        return {"verdict": "VOID", "text": f"red gate(s): {gates_red}"}
    ps, ci = pdms["per_seed_point"], pdms["ci95"]
    refuted = bool(ci[1] < MARGIN and all(p < MARGIN for p in ps))
    ni = [bool(c[0] > MARGIN) for c in pdms["per_seed_ci95"]]
    lon_ok = [bool(x <= LON_BAR) for x in d_lon]
    floor_ok = bool(n_cruise >= MIN_CRUISE and n_cruise_logs >= MIN_CRUISE_LOGS)
    cl = {"REFUTED_clause(avg CI upper < -0.5 and both seeds < -0.5)": refuted, "PDMS_NI_per_seed(lower CI > -0.5)": ni, "D_lon_le_0.3_per_seed": lon_ok, "CRUISE_floor_met": floor_ok}
    if refuted:
        return {"verdict": "REFUTED", "text": f"AXOFF-AXON PDMS CI {ci} entirely below {MARGIN}", "clauses": cl}
    if all(ni) and all(lon_ok) and floor_ok:
        return {"verdict": "CANDIDATE", "text": "PDMS non-inferiority at both seeds and CRUISE |bias| not worse by > 0.3 m/s at both seeds", "clauses": cl}
    why = [k for k, v in cl.items() if (not all(v) if isinstance(v, list) else (not v and not k.startswith("REFUTED")))]
    return {"verdict": "NOT PROVEN", "text": "failing clause(s): " + "; ".join(why), "clauses": cl}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--rung", required=True, choices=("X1", "X2"))
    ap.add_argument("--tokens", default=str(C.A5_TOKENS))
    ap.add_argument("--seeds", default="0,1")
    a = ap.parse_args(argv)
    root, seeds = Path(a.root), [int(s) for s in a.seeds.split(",")]
    toks = sorted(C.jload(a.tokens)["tokens"])
    off_n, on_n = [f"{a.rung}_AXOFF_s{s}" for s in seeds], [f"{a.rung}_AXON_s{s}" for s in seeds]
    arms = {n: load_arm(root, n, toks) for n in off_n + on_n}
    logs = [arms[off_n[0]]["pw"]["log"][i] for i in range(len(toks))]
    bt = S.Boot(logs)
    # ---- gates
    red, gates = [], {}
    gr = root / "analysis" / "gates_real.json"
    gates["gates_real"] = C.jload(gr) if gr.exists() else {"pass": False}
    if not gates["gates_real"].get("pass"):
        red.append("gates_real")
    for n in off_n + on_n:
        ip = root / "identity" / f"{n}.json"
        gates[f"identity_{n}"] = bool(ip.exists() and C.jload(ip).get("pass"))
        if not gates[f"identity_{n}"]:
            red.append(f"identity_{n}")
        gates[f"seam_misses_{n}"] = arms[n]["seam_misses"]
        if arms[n]["seam_misses"] != 0:
            red.append(f"seam_misses_{n}")
    g8 = root / "analysis" / "G8_mini.json"
    gates["G8_mini"] = C.jload(g8) if g8.exists() else {"ok": False}
    if not gates["G8_mini"].get("ok"):
        red.append("G8_mini")
    # ---- PDMS (primary)
    pdms = S.contrast(bt, [(arms[n]["sub"]["score"], None) for n in off_n], [(arms[n]["sub"]["score"], None) for n in on_n])
    pdms["off_value"], pdms["on_value"] = [float(arms[n]["sub"]["score"].mean()) for n in off_n], [float(arms[n]["sub"]["score"].mean()) for n in on_n]
    subs = {k: S.contrast(bt, [(arms[n]["sub"][k], None) for n in off_n], [(arms[n]["sub"][k], None) for n in on_n]) for k in SUB[:-1]}
    # ---- LONGITUDINAL (CRUISE)
    v0, a0, vg4, vg2, cr = cruise_inputs(toks)
    ncr, nlog = int(cr.sum()), len(set(np.asarray(logs)[cr]))
    lon = {"n_cruise": ncr, "n_cruise_logs": nlog, "per_arm": {}}
    for n in off_n + on_n:
        b4, b2 = plan_speed(arms[n]["poses"], 8) - vg4, plan_speed(arms[n]["poses"], 4) - vg2
        arms[n]["b4"] = b4
        lon["per_arm"][n] = {"b4_cruise": float(b4[cr].mean()), "b2_cruise": float(b2[cr].mean()), "b4_all": float(b4.mean()), "speed_mae_4s_all": float(np.abs(b4).mean())}
    d_lon, d_ci = [], []
    for o, n_ in zip(off_n, on_n):
        d_lon.append(abs(lon["per_arm"][o]["b4_cruise"]) - abs(lon["per_arm"][n_]["b4_cruise"]))
        sm = bt.stat(arms[o]["b4"] * cr, cr.astype(float)), bt.stat(arms[n_]["b4"] * cr, cr.astype(float))
        dd = np.abs(sm[0]) - np.abs(sm[1])
        d_ci.append([float(np.nanpercentile(dd, 2.5)), float(np.nanpercentile(dd, 97.5))])
    lon["D_lon_point_per_seed"], lon["D_lon_ci95_per_seed"] = d_lon, d_ci
    # ---- LATERAL / along-track (paired, descriptive)
    def lat(num, den, scale=1.0):
        c = S.contrast(bt, [(arms[n]["pw"][num], arms[n]["pw"][den]) for n in off_n], [(arms[n]["pw"][num], arms[n]["pw"][den]) for n in on_n])
        return {"point": c["point"] * scale, "ci95": [x * scale for x in c["ci95"]], "per_seed_point": [x * scale for x in c["per_seed_point"]]}
    lateral = {"curvature_mae_1pm": lat("curv_abs_sum", "curv_n"), "heading_mae_deg": lat("head_abs_sum_rad", "head_n", 180 / math.pi), "yaw_rate_mae_degps": lat("yaw_abs_sum_radps", "yaw_n", 180 / math.pi),
               "cross_mae_m": lat("cross_abs_sum", "cross_n"), "along_mae_m(longitudinal)": lat("along_abs_sum", "along_n"), "along_bias_m(longitudinal)": lat("along_signed_sum", "along_n")}
    dec = decide(not red, red, pdms, d_lon, ncr, nlog)
    desc = {"pick_changed_share_AXOFF_vs_AXON": [float(np.mean([not np.array_equal(arms[o]["poses"][i], arms[n_]["poses"][i]) for i in range(len(toks))])) for o, n_ in zip(off_n, on_n)]}
    try:
        import proxy_eval as PE
        desc["fam_adverse_AXOFF_vs_AXON"] = [PE.fam_adverse(arms[n_]["fam"], arms[o]["fam"])["adverse"] for o, n_ in zip(off_n, on_n)]
    except Exception as e:  # descriptive only; never gates the verdict
        desc["fam_adverse_AXOFF_vs_AXON"] = f"unavailable: {e!r}"[:200]
    a0row = {}
    try:
        pwb = np.load(root / "eval" / "A0_banked" / "perwindow.npz", allow_pickle=False)
        order = {str(t): i for i, t in enumerate(pwb["token"])}
        ix = np.asarray([order[t] for t in toks])
        brow = read_csv(C.BANKED_CSV)
        zb = np.load(C.BANKED_SEAM)
        zi = {str(t): i for i, t in enumerate(zb["token"])}
        pb = np.asarray([zb["poses"][zi[t]] for t in toks])
        b4 = plan_speed(pb, 8) - vg4
        a0row = {"pdms_x100": float(np.mean([100 * float(brow[t]["score"]) for t in toks])), "curvature_mae_1pm": float(pwb["curv_abs_sum"][ix].sum() / pwb["curv_n"][ix].sum()), "b4_cruise": float(b4[cr].mean()),
                 "note": "BANKED A0 passed through the same per-window code; its bit-for-bit reproduction (G6) is YF3's gate -> INHERITED here. Descriptive row, never in a decision."}
    except Exception as e:
        a0row = {"unavailable": repr(e)[:200]}
    out = {"rung": a.rung, "verdict": dec["verdict"], "decision": dec, "tier": TIER, "estimator": VARIANCE, "n_tokens": len(toks), "n_logs": len(set(logs)), "AXOFF": off_n, "AXON": on_n,
           "primary_pdms_x100(AXOFF-AXON)": pdms, "pdms_subscores_x100": subs, "longitudinal_CRUISE": lon, "lateral_paired(AXOFF-AXON)": lateral, "descriptive": desc, "A0_banked_row": a0row,
           "gates": gates, "gates_red": red, "tactical": TACTICAL, "strategic": STRATEGIC, "written": C.now()}
    (root / "analysis").mkdir(parents=True, exist_ok=True)
    C.jdump(out, root / "analysis" / f"{a.rung}.json")
    md = [f"# YAX analysis {a.rung}: **{dec['verdict']}**", "", dec["text"], "", f"Tier: {TIER}", "", f"Estimator: {VARIANCE}", "",
          f"PDMS x100 AXOFF-AXON: point {pdms['point']:+.3f}, avg CI {pdms['ci95']}, per-seed {pdms['per_seed_point']} CI {pdms['per_seed_ci95']}, F={pdms['F_seed_floor']}; AXOFF {pdms['off_value']} AXON {pdms['on_value']}",
          f"CRUISE n={ncr} logs={nlog}: D_lon per seed {d_lon} CI {d_ci}; per arm {json.dumps(lon['per_arm'])}", f"LATERAL (paired): {json.dumps(lateral)}", f"A0 banked: {json.dumps(a0row)}",
          f"TACTICAL: {TACTICAL}", f"STRATEGIC: {STRATEGIC}", f"Gates red: {red}"]
    (root / "analysis" / f"{a.rung}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"ZZAXVERDICT {a.rung} {dec['verdict']}  ({dec['text'][:200]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
