"""run_band_eval.py -- per-band baseline read (operative 0-2 s, tactical 2-6 s, full 0-6 s) of DAG-OACC (s0, s1) and the P10 floor at T1-kin on eval-141.
Zero training, CPU, descriptive.  Arms are loaded ONE AT A TIME (per-window arrays only).  Usage: run_band_eval.py <out.json> [n_boot]"""
import json
import os
import sys
import time

sys.path.insert(0, "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/horizon_review/band_eval")
import numpy as np
import torch
import band_lib as BL
from band_lib import CI, LM, FF, DT

OUT = sys.argv[1]
NB = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
SEED = 0
ART = "E:/Projects/TanitAD-artifacts"
DUMP = ART + "/v8f-dagoacc-t1/dump_DAGOACC_s{s}_m10_causal_heads"
PKG = "E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
LEAD = ART + "/dagoacc_t1/inputs/b1_eval_lead_block_k30.npz"
LEAD_SHA = "7a879baa2b6f4c140c2b64c9c098a38f84ecb04926da61fa8e96cfffc33073d9"
t0 = time.time()


def finite_pair_paired(a, b, eid):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return {"status": "NOT-APPLICABLE", "n_windows": 0}
    r = CI.paired_episode_cluster_bootstrap(a[m], b[m], eid[m], n_boot=NB, seed=SEED)
    return {"status": "OK", "delta": float(r["delta"]), "lo": float(r["lo"]), "hi": float(r["hi"]), "separated": bool(r["separated"]),
            "n_windows": int(m.sum()), "n_episodes": int(len(np.unique(eid[m])))}


def single(a, eid):
    a = np.asarray(a, np.float64)
    m = np.isfinite(a)
    if not m.any():
        return {"status": "NOT-APPLICABLE", "n_windows": 0}
    r = CI.episode_cluster_bootstrap(a[m], eid[m], reduce="mean", n_boot=NB, seed=SEED)
    return {"status": "OK", "mean": float(a[m].mean()), "lo": float(r["lo"]), "hi": float(r["hi"]), "n_windows": int(m.sum()),
            "n_episodes": int(len(np.unique(eid[m])))}


def sha(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---- lead join (same keying and guards as v8f_dk_read.join_block: (clip, frame=2*ws), speed proof, ts grid) ----
assert sha(LEAD) == LEAD_SHA, "lead block sha mismatch"
blk = dict(np.load(LEAD, allow_pickle=True))
idx = {(c, int(f)): i for i, (c, f) in enumerate(zip(np.asarray(blk["clip_id"]).astype(str).tolist(), np.asarray(blk["frame"]).astype(np.int64).tolist()))}
assert len(idx) == len(blk["leads"]), "duplicate lead rows"
ts = np.asarray(blk["ts_rel_s"], np.float64).reshape(-1)
assert np.max(np.abs(ts[:30] - DT * np.arange(1, 31))) <= 1e-6
clip_of = np.load(PKG + "/join_eval141.npz", allow_pickle=False)["clip"].astype(str)


def join(d):
    clip = clip_of[d["clip_index"]]
    frame = 2 * d["ws"].astype(np.int64)
    rows = np.array([idx.get((c, int(f)), -1) for c, f in zip(clip.tolist(), frame.tolist())])
    assert (rows >= 0).all(), "grid mismatch"
    bsp = np.asarray(blk["speeds"], np.float64).reshape(-1)[rows]
    dmax = float(np.max(np.abs(d["v0"] - bsp)))
    assert dmax <= 1e-3, f"speed proof failed {dmax}"
    st = np.asarray(blk["state"]).astype(str).reshape(-1)[rows]
    is_lead = st == "LEAD"
    leads = np.full((rows.size, 30, 2), np.nan)
    lens = np.full(rows.size, np.nan)
    leads[is_lead] = np.asarray(blk["leads"], np.float64)[rows[is_lead]][:, :30]
    lens[is_lead] = np.asarray(blk["lead_lens"], np.float64).reshape(-1)[rows[is_lead]]
    return {"leads": leads, "lead_lens": lens, "speeds": d["v0"].astype(np.float64), "is_lead": is_lead, "state": st, "speed_proof_max_mps": dmax,
            "counts": {s: int((st == s).sum()) for s in ("LEAD", "NO_LEAD", "NOT_STRAIGHT", "NO_LABEL")}}


def labels(X):
    from tanitad.refs.refc_tactical import factor_from_kinematics
    t = torch.as_tensor(np.asarray(X, np.float64))
    dyaw, dv, v0, v1, _ = FF.maneuver_kinematics(t, DT)
    lat, lon = factor_from_kinematics(dyaw, dv, v0, v1)
    return lat.numpy().astype(np.int64), lon.numpy().astype(np.int64)


def per_window(X, G, J, a, b):
    """All per-window arrays of one track in knot band [a, b)."""
    r = BL.band_rows(X, G, a, b)
    dk = BL.band_dk(X, J, a, b)
    r["dk_headway_min_m"] = np.asarray(dk["headway_min_m"], np.float64)
    r["dk_time_gap_min_s"] = np.asarray(dk["time_gap_min_s"], np.float64)
    return r


report = {"schema": "band-baseline-v1", "tier": "T1-kin (self-action OPEN loop; m=10, causal bucket, dead-reckoned nav, logged scene; the lead follows its logged future; stated as open loop)",
          "evidence_class": "MEASURED (ours; banked per-window dumps, zero training)",
          "estimator": f"taniteval.ci.paired_episode_cluster_bootstrap (arm - P10, same windows) and episode_cluster_bootstrap (single arm), B={NB}, seed {SEED}, 141 episodes. "
                       "It answers 'would another draw of EPISODES say this?'. It is blind to training-run variance (H-ESTIM-SEED-1): the two seeds are shown side by side and no seed floor is applied.",
          "bands": {k: {"knots_0idx": [a, b], "t_s": [round((a + 1) * DT, 1), round(b * DT, 1)]} for k, (a, b) in BL.BANDS.items()},
          "inputs": {}, "seeds": {}}
Pw = {}          # P10 per-band arrays (one dump; P10 reads no model, equal across arms)
for s in (0, 1):
    d = BL.load_dir(DUMP.format(s=s))
    eid = d["eid"]
    J = join(d)
    report["inputs"][f"dag_oacc_s{s}"] = {"dump_dir": DUMP.format(s=s), "n_windows": int(len(eid)), "n_episodes": int(len(np.unique(eid))), "keys_used": ["g", "cl", "fl", "v0", "eid", "clip_index", "ws"],
                                          "lead_window_states": J["counts"], "lead_speed_proof_max_abs_mps": J["speed_proof_max_mps"]}
    sd = {}
    for band, (a, b) in BL.BANDS.items():
        A_ = per_window(d["cl"], d["g"], J, a, b)
        P_ = per_window(d["fl"], d["g"], J, a, b)
        H_ = BL.band_dk(d["g"], J, a, b)              # human (GT) distance keeping over the same knots
        hw, tg = np.asarray(H_["headway_min_m"], np.float64), np.asarray(H_["time_gap_min_s"], np.float64)
        A_["dk_hw_abs_err_m"] = np.abs(A_["dk_headway_min_m"] - hw)
        P_["dk_hw_abs_err_m"] = np.abs(P_["dk_headway_min_m"] - hw)
        A_["dk_tg_abs_err_s"] = np.abs(A_["dk_time_gap_min_s"] - tg)
        P_["dk_tg_abs_err_s"] = np.abs(P_["dk_time_gap_min_s"] - tg)
        # DK-VIOL (share of at-risk lead windows with a violation), sliced over the band
        try:
            def lv(X):
                dkx = BL.band_dk(X, J, a, b)
                r = LM.lead_viol(dkx, H_, eid, lead_valid=J["is_lead"], n_boot=NB, seed=SEED, plan_arc=LM.path_arc_length(np.asarray(X)[:, a:b]),
                                 human_arc=LM.path_arc_length(d["g"][:, a:b]))
                return {k: r.get(k) for k in ("status", "reason", "n_lead_windows", "n_viol", "share", "n_at_risk", "share_of_at_risk", "share_of_at_risk_ci")}
            lva, lvp = lv(d["cl"]), lv(d["fl"])
        except Exception as e:                            # noqa: BLE001
            lva = lvp = {"status": "ERROR", "reason": f"{type(e).__name__}: {e}"}
        rows = {}
        for m in list(BL.METRICS) + ["dk_hw_abs_err_m", "dk_tg_abs_err_s"]:
            rows[m] = {"arm": single(A_[m], eid), "P10": single(P_[m], eid), "arm_minus_P10": finite_pair_paired(A_[m], P_[m], eid)}
            rows[m]["family"] = BL.FAMILY.get(m, "LONGITUDINAL")
        n_dk = {"n_lead_windows": int(J["is_lead"].sum()), "n_arm_hw_finite": int(np.isfinite(A_["dk_hw_abs_err_m"]).sum()),
                "n_p10_hw_finite": int(np.isfinite(P_["dk_hw_abs_err_m"]).sum()), "n_human_hw_finite": int(np.isfinite(hw).sum()),
                "n_arm_tg_finite": int(np.isfinite(A_["dk_tg_abs_err_s"]).sum()), "n_human_tg_finite": int(np.isfinite(tg).sum())}
        sd[band] = {"rows": rows, "dk_counts": n_dk, "DK-VIOL": {"arm": lva, "P10": lvp}}
        if s == 0:
            Pw[band] = P_
        print(f"s{s} {band} done {time.time() - t0:.0f}s", flush=True)
    # TACTICAL decision reads (existing: trajectory-derived lateral/longitudinal manoeuvre of the whole 0-6 s plan vs the GT 0-6 s plan)
    glat, glon = labels(d["g"])
    lat, lon = labels(d["cl"])
    plat, plon = labels(d["fl"])
    tac = {}
    for nm, ga, ar, pr in (("TAC-LAT", glat, lat, plat), ("TAC-LON", glon, lon, plon)):
        ia, ip = (ar == ga).astype(np.float64), (pr == ga).astype(np.float64)
        tac[nm] = {"arm": single(ia, eid), "P10": single(ip, eid), "arm_minus_P10": finite_pair_paired(ia, ip, eid), "statistic": "per-window indicator executed label == GT label (accuracy)"}
    sd["TACTICAL_decision_reads"] = {"rows": tac, "label_band_of_these_reads": "labels derived from the whole 0-6 s plan (four_families.maneuver_kinematics: path tangent at the last moving knot, chord speed knot 1 vs knot 30)",
                                     "T_head_label_band": "the T head's v9 labels span [NOW+2 s, NOW+8 s] (build_v9_labels.py, AUDIT_MODEL F8); the tactical band is 2-6 s: a 2 s mismatch at the far edge and a 0-2 s overlap at the near edge for the plan-derived reads",
                                     "n_windows": int(len(eid))}
    report["seeds"][f"s{s}"] = sd
    del d, J
report["STRATEGIC"] = {"status": "PAUSED", "ruling": "PI item 25 (PI_DECISION_QUEUE.md ITEM 25, 2026-09-19): the strategic layer is switched off temporarily; reported as PAUSED, not as a pass and not omitted."}
report["wall_s"] = round(time.time() - t0, 1)
json.dump(report, open(OUT, "w"), indent=1)
print("WROTE", OUT, report["wall_s"], "s")
