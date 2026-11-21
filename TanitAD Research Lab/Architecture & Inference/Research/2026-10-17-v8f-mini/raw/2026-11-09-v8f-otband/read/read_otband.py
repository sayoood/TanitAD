"""read_otband.py -- the PREREG_V8F_OTBAND read (frozen card, prefix sha256 fd81c66e) on T1-kin dumps.  CPU, zero training.
Usage: python read_otband.py <out.json> [n_boot]

Arms per seed s: OB-CTL_s{s} (the reference of every bar), OB-BAND_s{s}; beside: DAG-OACC s{s} (banked baseline dump); floor: OB-BAND_rep
(OB-BAND s0's flags and init, training seed 2) minus OB-BAND_s0.  A seed is read when both of its dumps exist; the replicate when it exists.
Every dump must share the DAG-OACC dump's windows (eid, clip_index, ws) and its model-free arrays (g, fl, v0), or the read refuses.
Geometry and distance keeping are band_eval's (read_lib.py = band_lib.py pointed at the commit tree), the same as BAND_BASELINE.md.

Reading choices, written before any OTBAND dump was read (README_READ.md repeats them):
  (r1) "up" is the upper bound of the paired 95 % interval of arm minus OB-CTL; for an accuracy row (B5, higher is better) the bound on
       the worse side is the LOWER bound, so B5 reads lo(BAND - CTL) >= -0.02.
  (r2) B3's distance-keeping rows are the absolute errors against the human's own minimum headway and time gap over the same knots
       (dk_hw_abs_err_m, dk_tg_abs_err_s), in both bands, lower is better.
  (r3) "any tactical row worse than its bar" (BAND-COST) covers B3 and B5, and B2 when its interval is separated on the worse side
       (lo > 0 on ADE_tac or FDE_tac); a B2 that only fails to improve is BAND-NEUTRAL.  B4 is read on its own (card) and never sets
       the outcome.
  (r4) the replicate floor of a row is |delta(OB-BAND_rep - OB-BAND_s0)| on that row; the replicate exists at seed 0 only, so its
       floor is applied to the rows of both seeds, and that is stated in the output.
  (r5) without the replicate, an outcome that needs the floor is printed with the prefix PROVISIONAL and decides nothing.
  (r6) BAND-GAIN's "B1 and B2 hold and exceed the replicate floor" is read as: B1 holds, B2 holds, and B2's two rows (ADE_tac,
       FDE_tac) exceed their floors; B1 is a non-regression bar and has no effect to exceed.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import read_lib as BL  # noqa: E402
from read_lib import CI, LM, FF, DT  # noqa: E402

OUT = sys.argv[1]
NB = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
SEED = 0
ART = "E:/Projects/TanitAD-artifacts"
NEW = "F:/TanitAD-scratch/otband_read/t1"
DAG = ART + "/v8f-dagoacc-t1/dump_DAGOACC_s{s}_m10_causal_heads"
RAW = ART + "/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
LEAD = ART + "/dagoacc_t1/inputs/b1_eval_lead_block_k30.npz"
LEAD_SHA = "7a879baa2b6f4c140c2b64c9c098a38f84ecb04926da61fa8e96cfffc33073d9"
PULL = "C:/Users/Admin/t2v8f_pulls"
SAME = ("eid", "clip_index", "ws", "g", "fl", "v0")
OP, TAC = "operative_0_2s", "tactical_2_6s"
t0 = time.time()


def ddir(tag):
    return f"{NEW}/dump_{tag}_m10_causal_heads"


def have(tag):
    return os.path.exists(f"{NEW}/done_{tag}_m10_causal_heads.txt") and os.path.isdir(ddir(tag))


def paired(a, b, eid):
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


# ---- lead join: run_band_eval.py's, unchanged ----
assert sha(LEAD) == LEAD_SHA, "lead block sha mismatch"
blk = dict(np.load(LEAD, allow_pickle=True))
idx = {(c, int(f)): i for i, (c, f) in enumerate(zip(np.asarray(blk["clip_id"]).astype(str).tolist(), np.asarray(blk["frame"]).astype(np.int64).tolist()))}
assert len(idx) == len(blk["leads"]), "duplicate lead rows"
clip_of = np.load(RAW + "/join_eval141.npz", allow_pickle=False)["clip"].astype(str)


def join(d):
    clip = clip_of[d["clip_index"]]
    frame = 2 * d["ws"].astype(np.int64)
    rows = np.array([idx.get((c, int(f)), -1) for c, f in zip(clip.tolist(), frame.tolist())])
    assert (rows >= 0).all(), "grid mismatch"
    bsp = np.asarray(blk["speeds"], np.float64).reshape(-1)[rows]
    assert float(np.max(np.abs(d["v0"] - bsp))) <= 1e-3, "speed proof failed"
    st = np.asarray(blk["state"]).astype(str).reshape(-1)[rows]
    is_lead = st == "LEAD"
    leads = np.full((rows.size, 30, 2), np.nan)
    lens = np.full(rows.size, np.nan)
    leads[is_lead] = np.asarray(blk["leads"], np.float64)[rows[is_lead]][:, :30]
    lens[is_lead] = np.asarray(blk["lead_lens"], np.float64).reshape(-1)[rows[is_lead]]
    return {"leads": leads, "lead_lens": lens, "speeds": d["v0"].astype(np.float64), "is_lead": is_lead,
            "counts": {s: int((st == s).sum()) for s in ("LEAD", "NO_LEAD", "NOT_STRAIGHT", "NO_LABEL")}}


def labels(X):
    from tanitad.refs.refc_tactical import factor_from_kinematics
    t = torch.as_tensor(np.asarray(X, np.float64))
    dyaw, dv, v0, v1, _ = FF.maneuver_kinematics(t, DT)
    lat, lon = factor_from_kinematics(dyaw, dv, v0, v1)
    return lat.numpy().astype(np.int64), lon.numpy().astype(np.int64)


def arm_arrays(X, G, J, eid):
    """per-window rows of track X in every band, plus DK abs errors against the human, DK-VIOL and the plan-derived manoeuvre labels"""
    out = {}
    for band, (a, b) in BL.BANDS.items():
        r = BL.band_rows(X, G, a, b)
        dk, H = BL.band_dk(X, J, a, b), BL.band_dk(G, J, a, b)
        r["dk_hw_abs_err_m"] = np.abs(np.asarray(dk["headway_min_m"], np.float64) - np.asarray(H["headway_min_m"], np.float64))
        r["dk_tg_abs_err_s"] = np.abs(np.asarray(dk["time_gap_min_s"], np.float64) - np.asarray(H["time_gap_min_s"], np.float64))
        try:
            lv = LM.lead_viol(dk, H, eid, lead_valid=J["is_lead"], n_boot=NB, seed=SEED, plan_arc=LM.path_arc_length(np.asarray(X)[:, a:b]),
                              human_arc=LM.path_arc_length(np.asarray(G)[:, a:b]))
            r["_dk_viol"] = {k: lv.get(k) for k in ("status", "reason", "n_lead_windows", "n_viol", "share", "n_at_risk", "share_of_at_risk", "share_of_at_risk_ci")}
        except Exception as e:  # noqa: BLE001
            r["_dk_viol"] = {"status": "ERROR", "reason": f"{type(e).__name__}: {e}"}
        out[band] = r
    out["_lat"], out["_lon"] = labels(X)
    return out


def load(tag_or_dir, ref=None):
    d = BL.load_dir(tag_or_dir)
    if ref is not None:
        for k in SAME:
            assert np.array_equal(d[k], ref[k]), f"window or model-free array differs: {k} in {tag_or_dir}"
    return d


METR = list(BL.METRICS) + ["dk_hw_abs_err_m", "dk_tg_abs_err_s"]
B1 = {("ade_m", 0.03), ("fde_m", 0.05), ("speed_mae_mps", 0.03), ("cross_mae_m", 0.02), ("yaw_rate_mae_degps", 0.5)}


def bars(D, tac, floor):
    """D[band][metric] = paired(BAND - CTL); tac[name] = paired accuracy BAND - CTL; floor[band][metric] = |rep delta| or None"""
    def up(band, m):
        return D[band][m].get("hi", float("nan"))
    b1 = {f"{m}<= +{v}": up(OP, m) <= v for m, v in sorted(B1)}
    b2 = {"ade_m<0": up(TAC, "ade_m") < 0, "fde_m<0": up(TAC, "fde_m") < 0}
    b2_worse = D[TAC]["ade_m"].get("lo", -1) > 0 or D[TAC]["fde_m"].get("lo", -1) > 0
    b3 = {"speed_mae_mps<=0": up(TAC, "speed_mae_mps") <= 0, "along_mae_m<=0": up(TAC, "along_mae_m") <= 0}
    for band in (OP, TAC):
        b3[f"{band}:dk_hw_abs_err_m<=+0.2"] = (D[band]["dk_hw_abs_err_m"].get("status") != "OK") or up(band, "dk_hw_abs_err_m") <= 0.2
        b3[f"{band}:dk_tg_abs_err_s<=+0.03"] = (D[band]["dk_tg_abs_err_s"].get("status") != "OK") or up(band, "dk_tg_abs_err_s") <= 0.03
    b4 = {"yaw_rate_mae_degps<0": up(TAC, "yaw_rate_mae_degps") < 0, "curvature_mae_1pm<0": up(TAC, "curvature_mae_1pm") < 0}
    b5 = {f"{k} lo>=-0.02": tac[k].get("lo", float("nan")) >= -0.02 for k in ("TAC-LAT", "TAC-LON")}
    exceed = None
    if floor is not None:
        exceed = {m: abs(D[TAC][m]["delta"]) > floor[TAC][m] for m in ("ade_m", "fde_m")}
    hold = {k: all(v.values()) for k, v in (("B1", b1), ("B2", b2), ("B3", b3), ("B4", b4), ("B5", b5))}
    if not hold["B1"] or not hold["B3"] or not hold["B5"] or b2_worse:
        lab = "BAND-COST"
    elif hold["B2"] and exceed is None:
        lab = "PROVISIONAL BAND-GAIN (replicate floor not yet read)"
    elif hold["B2"] and all(exceed.values()):
        lab = "BAND-GAIN"
    else:
        lab = "BAND-NEUTRAL" if exceed is not None or not hold["B2"] else "PROVISIONAL BAND-NEUTRAL"
    return {"label": lab, "hold": hold, "B1": b1, "B2": b2, "B2_separated_worse": b2_worse, "B3": b3, "B4": b4, "B5": b5,
            "B6_exceeds_floor": exceed, "B4_named": "B4 holds" if hold["B4"] else "B4 fails"}


def train_tail(tag):
    p = f"{PULL}/otband_{tag}/train_log.jsonl"
    if not os.path.exists(p):
        return {"status": "absent", "path": p}
    last = None
    for line in open(p, encoding="utf-8"):
        if line.strip():
            last = json.loads(line)
    keep = {k: v for k, v in (last or {}).items() if any(s in k for s in ("ade", "fde", "op", "tac", "step", "loss"))}
    return {"tier": "T0 (teacher-forced train terms; a curve, not a result)", "last_row": keep, "path": p}


report = {"schema": "otband-read-v1", "card": "PREREG_V8F_OTBAND.md (frozen, prefix sha256 fd81c66e1459da16bddd44c1501394d30cc81d036f54ef9c4ad28e4594eb6c37)",
          "tier": "T1-kin (self-action open loop; m = 10, causal bucket, dead-reckoned nav, logged scene; the lead follows its logged future)",
          "evidence_class": "MEASURED (ours; dumps of the banked checkpoints, zero training)",
          "estimator": f"taniteval.ci.paired_episode_cluster_bootstrap (arm minus arm, same windows) and episode_cluster_bootstrap (one arm), B={NB}, seed {SEED}, "
                       "141 episodes / 2,399 windows. It answers 'would another draw of EPISODES say this?'. 'Would another TRAINING run say this?' is the "
                       "replicate row and the two seeds side by side; the planner is deterministic, so there is no inference-variance question.",
          "reading_choices": __doc__.split("Reading choices, written before any OTBAND dump was read (README_READ.md repeats them):")[1].strip(),
          "bands": {k: {"knots_0idx": [a, b], "t_s": [round((a + 1) * DT, 1), round(b * DT, 1)]} for k, (a, b) in BL.BANDS.items()},
          "seeds": {}, "replicate": None, "train_terms": {}}


def plabel(w):
    p = os.path.join(HERE, f"PARITY_{w}.json")
    return json.load(open(p, encoding="utf-8"))["label"] if os.path.exists(p) else "ABSENT"


# amendment 1 (README_READ.md): P0 read FAIL under its rule (float-level drift against the banked dump, made at another thread count);
# P0b, the baseline's own tool tree at this read's threads, equals the commit tree bit for bit.  The read runs if P0 passes, or if P0b is
# PASS-EXACT; the DAG-OACC rows then use the dumps made here (PARITY_DAGOACC_s{s}), so every compared pair shares one environment.
report["parity"] = {"P0": plabel("P0"), "P0b": plabel("P0b")}
assert report["parity"]["P0"] in ("PASS-EXACT", "PASS") or report["parity"]["P0b"] == "PASS-EXACT", f"parity {report['parity']}: the read is INVALID"

floor = None
ARR = {}
for s in (0, 1):
    ctl, band = f"OB-CTL_s{s}", f"OB-BAND_s{s}"
    if not (have(ctl) and have(band)):
        report["seeds"][f"s{s}"] = {"status": "NOT-YET", "reason": f"dumps present: {ctl} {have(ctl)}, {band} {have(band)}"}
        continue
    if not have(f"PARITY_DAGOACC_s{s}"):
        report["seeds"][f"s{s}"] = {"status": "NOT-YET", "reason": f"the same-environment DAG-OACC dump PARITY_DAGOACC_s{s} is missing"}
        continue
    banked = load(DAG.format(s=s))                       # window identity reference only
    dg = load(ddir(f"PARITY_DAGOACC_s{s}"), banked)
    eid = dg["eid"]
    J = join(dg)
    dc, db = load(ddir(ctl), banked), load(ddir(band), banked)
    A = {"OB-CTL": arm_arrays(dc["cl"], dg["g"], J, eid), "OB-BAND": arm_arrays(db["cl"], dg["g"], J, eid),
         "DAG-OACC": arm_arrays(dg["cl"], dg["g"], J, eid)}
    init = f"INIT_M2b_s{s}"
    if have(init):              # amendment 2: beside, post-hoc, decides nothing (not in the card)
        A["INIT-M2b"] = arm_arrays(load(ddir(init), banked)["cl"], dg["g"], J, eid)
    for lad, key in ((f"PARITY_DAG0_s{s}", "DAG0"), (f"PARITY_DAG_s{s}", "DAG")):
        if have(lad):           # amendment 3: the banked ladder M2b -> DAG0 (schedule) -> DAG (+ DAgger rows) -> DAG-OACC (+ w_acc); beside
            A[key] = arm_arrays(load(ddir(lad), banked)["cl"], dg["g"], J, eid)
    if s == 0:
        ARR["s0"] = (A, dg, J)
    glat, glon = labels(dg["g"])
    sd = {"inputs": {"dag_oacc_dump": ddir(f"PARITY_DAGOACC_s{s}"), "window_reference": DAG.format(s=s), "ctl_dump": ddir(ctl), "band_dump": ddir(band), "n_windows": int(len(eid)),
                     "n_episodes": int(len(np.unique(eid))), "lead_window_states": J["counts"]}, "bands": {}}
    D = {}
    for bnd in BL.BANDS:
        rows, D[bnd] = {}, {}
        for m in METR:
            x = {a: A[a][bnd][m] for a in A}
            rows[m] = {"family": BL.FAMILY.get(m, "LONGITUDINAL"), "OB-CTL": single(x["OB-CTL"], eid), "OB-BAND": single(x["OB-BAND"], eid),
                       "DAG-OACC": single(x["DAG-OACC"], eid), "BAND_minus_CTL": paired(x["OB-BAND"], x["OB-CTL"], eid),
                       "CTL_minus_DAG": paired(x["OB-CTL"], x["DAG-OACC"], eid), "BAND_minus_DAG": paired(x["OB-BAND"], x["DAG-OACC"], eid)}
            if "INIT-M2b" in A:
                rows[m]["beside_post_hoc"] = {"INIT-M2b": single(x["INIT-M2b"], eid), "CTL_minus_INIT": paired(x["OB-CTL"], x["INIT-M2b"], eid),
                                              "DAG_minus_INIT": paired(x["DAG-OACC"], x["INIT-M2b"], eid)}
            if "INIT-M2b" in A and "DAG0" in A and "DAG" in A:
                rows[m]["ladder_post_hoc"] = {
                    "DAG0": single(x["DAG0"], eid), "DAG": single(x["DAG"], eid),
                    "step1_schedule_DAG0_minus_INIT": paired(x["DAG0"], x["INIT-M2b"], eid),
                    "step2_dagger_rows_DAG_minus_DAG0": paired(x["DAG"], x["DAG0"], eid),
                    "step3_w_acc_DAGOACC_minus_DAG": paired(x["DAG-OACC"], x["DAG"], eid),
                    "CTL_minus_DAG0": paired(x["OB-CTL"], x["DAG0"], eid),
                    "BAND_minus_DAG0": paired(x["OB-BAND"], x["DAG0"], eid),
                    "note": "beside, post-hoc, decides nothing; the three steps sum to DAG_minus_INIT; DAG0, DAG and DAG-OACC share M2b's init, 3,000 steps, lr 1e-4, p_gt 0.2 and a fully trainable model"}
            D[bnd][m] = rows[m]["BAND_minus_CTL"]
        sd["bands"][bnd] = {"rows": rows, "DK-VIOL": {a: A[a][bnd]["_dk_viol"] for a in A}}
        print(f"s{s} {bnd} {time.time() - t0:.0f}s", flush=True)
    tac = {}
    for nm, gl, key in (("TAC-LAT", glat, "_lat"), ("TAC-LON", glon, "_lon")):
        ind = {a: (A[a][key] == gl).astype(np.float64) for a in A}
        tac[nm] = {"OB-CTL": single(ind["OB-CTL"], eid), "OB-BAND": single(ind["OB-BAND"], eid), "DAG-OACC": single(ind["DAG-OACC"], eid),
                   "BAND_minus_CTL": paired(ind["OB-BAND"], ind["OB-CTL"], eid), "BAND_minus_DAG": paired(ind["OB-BAND"], ind["DAG-OACC"], eid),
                   "statistic": "per-window indicator: the plan's own 0-6 s manoeuvre label equals the GT's (accuracy)"}
        if all(k in A for k in ("INIT-M2b", "DAG0", "DAG")):
            tac[nm]["ladder_post_hoc"] = {k: single(ind[k], eid) for k in ("INIT-M2b", "DAG0", "DAG")}
            tac[nm]["ladder_post_hoc"].update(step1=paired(ind["DAG0"], ind["INIT-M2b"], eid), step2=paired(ind["DAG"], ind["DAG0"], eid),
                                              step3=paired(ind["DAG-OACC"], ind["DAG"], eid))
    sd["TACTICAL_plan_reads"] = tac
    sd["_D"] = D
    report["seeds"][f"s{s}"] = sd
    for t in (ctl, band):
        report["train_terms"][t] = train_tail(t)

if have("OB-BAND_rep") and "s0" in ARR:
    A, dg, J = ARR["s0"]
    eid = dg["eid"]
    dr = load(ddir("OB-BAND_rep"), dg)
    R = arm_arrays(dr["cl"], dg["g"], J, eid)
    floor, rep = {}, {}
    for bnd in BL.BANDS:
        floor[bnd], rep[bnd] = {}, {}
        for m in METR:
            r = paired(R[bnd][m], A["OB-BAND"][bnd][m], eid)
            rep[bnd][m] = r
            floor[bnd][m] = abs(r["delta"]) if r.get("status") == "OK" else float("inf")
    report["replicate"] = {"rows_REP_minus_BAND_s0": rep, "floor_rule": "(r4): |delta| per row, read at seed 0, applied to both seeds"}
    report["train_terms"]["OB-BAND_rep"] = train_tail("OB-BAND_rep")

labs = {}
for s in ("s0", "s1"):
    sd = report["seeds"][s]
    if "_D" not in sd:
        continue
    D = sd.pop("_D")
    sd["bars"] = bars(D, sd["TACTICAL_plan_reads"] and {k: v["BAND_minus_CTL"] for k, v in sd["TACTICAL_plan_reads"].items()}, floor)
    labs[s] = sd["bars"]["label"]
if len(labs) == 2:
    report["outcome"] = labs["s0"] if labs["s0"] == labs["s1"] else f"SEEDS-DISAGREE (s0 {labs['s0']}; s1 {labs['s1']})"
else:
    report["outcome"] = f"NOT-YET: both seeds are required by the card; per-seed labels so far {labs} decide nothing"
report["TOKEN"] = {"status": "BLOCKED", "reason": "OB-BAND-TOK needs stream C's stok dump (card: named blocker at freeze); not trained"}
report["TOP-K"] = {"status": "PENDING", "reason": "forward_topk is not called by v8f_t1_eval.py; the beside read needs its own script (topk_read.py)"}
report["STRATEGIC"] = {"status": "PAUSED", "ruling": "PI item 25 (PI_DECISION_QUEUE.md ITEM 25, 2026-09-19): reported as PAUSED, not as a pass and not omitted."}
report["wall_s"] = round(time.time() - t0, 1)
json.dump(report, open(OUT, "w"), indent=1)
print("WROTE", OUT, "outcome", report["outcome"], report["wall_s"], "s")
