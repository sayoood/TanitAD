"""n1dw_ho_read.py -- the D1-HO part of the PREREG_V8F_N1DW read (card sha256 59f2b7f9 + Amendment 1).  POD CPU, zero training, one process.
Reuses navrbd_read.py's pure logic unchanged in intent (d1_ho, z_use_ho, k_const, k1_self, kin_acc, d2_ho are copied verbatim) and read_otband.labels / read_lib.band_rows
(same tree blobs: four_families.py 96839329, ci.py 528276c5, refc_tactical.py b7e44c3e, lead_metrics.py 25190ad6 on dev box and pod).
Extended only by the arm list (N1DW{2,4}_s{0,1}, their zero twins) and by the R1 LON/LAT cells on D1-HO.
Evidence class MEASURED (ours; T1-kin dumps).  Estimator: taniteval.ci.paired_episode_cluster_bootstrap, B 2000, seed 0 -- answers "another draw of EPISODES" only.
Tier: T1-kin (self-action open loop, m=10, causal bucket).  Optimistic by construction: the route text is supplied along the driven path.
The eval-141 guard (card 5.2) is NOT computed here: every label that needs it prints PENDING-GUARD.
"""
import glob
import json
import math
import os
import sys
import time
import importlib.util

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "2")
TREE = "/root/strat1/n1dw/tree"
os.environ["TANITEVAL_STACK_OVERRIDE"] = TREE + "/stack"
sys.path.insert(0, TREE + "/taniteval")
sys.path.insert(0, TREE + "/stack")
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(2)
import read_lib_pod as BL  # noqa: E402
from taniteval import four_families as FF  # noqa: E402

torch.set_num_threads(2)
spec = importlib.util.spec_from_file_location("ci_pod", TREE + "/taniteval/taniteval/ci.py")
CI = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CI)

NB, SEED = 2000, 0
DT = 0.2
BAR_FLOOR = 0.0383
HO_WINDOWS, HO_EPISODES = 5736, 497
CL = ("lane_keep", "turn_left", "turn_right")
BANDS = {"operative_0_2s": (0, 10), "tactical_2_6s": (10, 30)}
METRICS = ("ade_m", "fde_m", "speed_mae_mps", "along_mae_m", "accel_mae_mps2", "cross_mae_m", "heading_mae_deg", "yaw_rate_mae_degps", "curvature_mae_1pm", "dk_hw_abs_err_m", "dk_tg_abs_err_s")
CELLS = tuple(f"{b}:{m}" for b in BANDS for m in METRICS) + ("TAC-LAT", "TAC-LON")
FAMILY = {"ade_m": "PRIMARY", "fde_m": "PRIMARY", "speed_mae_mps": "LONGITUDINAL", "along_mae_m": "LONGITUDINAL", "accel_mae_mps2": "LONGITUDINAL", "dk_hw_abs_err_m": "LONGITUDINAL",
          "dk_tg_abs_err_s": "LONGITUDINAL", "cross_mae_m": "LATERAL", "heading_mae_deg": "LATERAL", "yaw_rate_mae_degps": "LATERAL", "curvature_mae_1pm": "LATERAL", "TAC-LAT": "TACTICAL", "TAC-LON": "TACTICAL"}
SAME = ("eid", "clip_index", "ws", "g", "fl", "v0")

R = "/root/strat1"
GRID = R + "/navrbd_step0/out/inputs/eval_grid_x_feasible.json"
GRID_SHA = "ca289d0f40a8f71043796fe626b26ecaa141853f843acdb1eecbf5648b9ab148"
BANKED = os.environ.get("N1DW_BANKED_RESULT", HERE + "/RESULT_NAVRBD.json")
OUT = os.environ.get("N1DW_READ_OUT", R + "/n1dw/read")
N1DWD = R + "/n1dw/out"


def dpath(arm):
    """dump directory of a read arm name"""
    if arm.startswith("N1R"):
        return f"{R}/navrbd_step0/out/t1/dump_NAVRBD-HO-{arm}_m10_causal_heads"
    zero = arm.endswith("Z")
    base = arm[:-1] if zero else arm
    if base.startswith("N1DW"):
        return f"{N1DWD}/{'t1_zero' if zero else 't1'}/dump_N1DW-HO-{base}{'_zero' if zero else ''}_m10_causal_heads"
    return f"{R}/navrbd/out/{'t1_zero' if zero else 't1'}/dump_NAVRBD-HO-{base}{'_zero' if zero else ''}_m10_causal_heads"


def load_dump_n(d, n):
    fs = sorted(glob.glob(os.path.join(d, "ep*.npz")))
    assert len(fs) == n, (d, len(fs))
    zs = [np.load(f) for f in fs]
    return {k: np.concatenate([z[k] for z in zs]) for k in ("g", "cl", "ol", "fl", "v0", "eid", "clip_index", "ws")}


def labels(X):                                    # read_otband.labels, verbatim
    from tanitad.refs.refc_tactical import factor_from_kinematics
    t = torch.as_tensor(np.asarray(X, np.float64))
    dyaw, dv, v0, v1, _ = FF.maneuver_kinematics(t, DT)
    lat, lon = factor_from_kinematics(dyaw, dv, v0, v1)
    return lat.numpy().astype(np.int64), lon.numpy().astype(np.int64)


def cells_of(x, g):
    """24 cells of a dump vs the GT track g; the two lead-join cells per band are NaN (no lead block exists for held-out windows)"""
    out = {}
    n = len(g)
    for band, (a, b) in BANDS.items():
        r = BL.band_rows(x["cl"], g, a, b)
        for m in METRICS:
            out[f"{band}:{m}"] = np.asarray(r[m], np.float64) if m in r else np.full(n, np.nan)
    return out


# ------------------------------------------------ pure logic (copied from navrbd_read.py)
def check_three_class(lab):
    lab = np.asarray(lab)
    assert lab.min() >= 0 and lab.max() <= 2
    return lab


def kin_acc(g_lab, p_lab):
    return (np.asarray(p_lab) == np.asarray(g_lab)).astype(np.float64)


def paired(a, b, e, nb=NB):
    a, b, e = np.asarray(a, np.float64), np.asarray(b, np.float64), np.asarray(e)
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return {"status": "NOT-COMPUTABLE", "n_windows": 0, "n_episodes": 0}
    r = CI.paired_episode_cluster_bootstrap(a[m], b[m], e[m], n_boot=nb, seed=SEED)
    return {"status": "OK", "delta": float(a[m].mean() - b[m].mean()), "lo": float(r["lo"]), "hi": float(r["hi"]), "separated": bool(r["separated"]),
            "n_windows": int(m.sum()), "n_episodes": int(len(np.unique(e[m])))}


def d1_row(acc_arm, acc_ref, eid, bar):
    r = paired(acc_arm, acc_ref, eid)
    r.update({"bar": float(bar), "gain": bool(r["delta"] >= bar and r["lo"] > 0), "cost": bool(r["delta"] <= -bar and r["hi"] < 0),
              "acc_arm": float(np.mean(acc_arm)), "acc_ref": float(np.mean(acc_ref))})
    return r


def d2_ho(g, pa, pr, eid):
    g, pa, pr, e = (np.asarray(x) for x in (g, pa, pr, eid))
    out = {}
    turn, lk = g != 0, g == 0
    for name, m in (("turn_recall", turn), ("false_turn_share", lk)):
        a, b = (pa != 0)[m].astype(np.float64), (pr != 0)[m].astype(np.float64)
        rr = CI.paired_episode_cluster_bootstrap(a, b, e[m], n_boot=NB, seed=SEED)
        out[name] = {"arm": float(a.mean()), "ref": float(b.mean()), "delta": float(a.mean() - b.mean()), "lo": float(rr["lo"]), "hi": float(rr["hi"]), "separated": bool(rr["separated"])}
    for name, p in (("arm", pa), ("ref", pr)):
        both = turn & (p != 0)
        out[f"side_given_turn_{name}"] = {"value": float((p[both] == g[both]).mean()) if both.any() else None, "n_windows": int(both.sum())}
    return out


def kclip(path):
    n = c = 0
    for ln in open(path, encoding="utf-8"):
        ln = ln.strip()
        if not ln:
            continue
        gn = json.loads(ln).get("grad_norm")
        if gn is None:
            continue
        n += 1
        c += int(float(gn) > 1.0)
    return {"logged_steps": n, "clipped_steps": c, "share": (c / n if n else None), "path": path}


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    banked = json.load(open(BANKED, encoding="utf-8"))
    rep = {"tool": "n1dw_ho_read.py", "card": "PREREG_V8F_N1DW.md sha256 59f2b7f9 + AMENDMENT_N1DW_1 (D1-HO part only)", "evidence_class": "MEASURED (ours; T1-kin dumps, zero training)",
           "estimator": f"taniteval.ci.paired_episode_cluster_bootstrap B={NB} seed 0 ({CI.__file__}); answers 'another draw of EPISODES' only; training-run variance is carried by F_HO and the within-arm seed draws; inference variance does not apply (deterministic planner)",
           "tier": "T1-kin (m=10 causal bucket, self-action open loop)", "limit": "optimistic by construction: route supplied along the driven path", "controls": {}, "problems": []}
    prob = rep["problems"]
    grid_sha = __import__("hashlib").sha256(open(GRID, "rb").read()).hexdigest()
    rep["grid_sha256"] = grid_sha
    if grid_sha != GRID_SHA:
        prob.append("grid sha256 differs")
    ARMS = ["N1R_s0", "N1R_s1", "N1D_s0", "N1D_s1", "N1D_s0Z", "N1D_s1Z"] + [f"N1DW{w}_s{s}{z}" for w in (2, 4) for s in (0, 1) for z in ("", "Z")]
    X = {}
    for a in ARMS:
        d = dpath(a)
        if not os.path.isdir(d):
            prob.append(f"dump missing: {a}")
            continue
        X[a] = load_dump_n(d, HO_EPISODES)
    ref = X["N1R_s0"]
    eid = ref["eid"]
    if len(eid) != HO_WINDOWS or len(np.unique(eid)) != HO_EPISODES:
        prob.append(f"grid count {len(eid)} / {len(np.unique(eid))}")
    for a, x in X.items():
        for k in SAME:
            if not np.array_equal(x[k], ref[k]):
                prob.append(f"window/model-free array differs from N1R_s0: {k} in {a}")
    glat, glon = labels(ref["g"])
    check_three_class(glat)
    rep["dump_dirs"] = {a: dpath(a) for a in X}
    C, plat = {}, {}
    for a, x in X.items():
        cl_lat, cl_lon = labels(x["cl"])
        c = cells_of(x, ref["g"])
        c["TAC-LAT"] = kin_acc(glat, check_three_class(cl_lat))
        c["TAC-LON"] = (cl_lon == glon).astype(np.float64)
        C[a], plat[a] = c, cl_lat
        print(f"cells {a} {time.time() - t0:.0f}s", flush=True)
    acc = {a: C[a]["TAC-LAT"] for a in C}
    # ---------------- K-RE: N1R and N1D recomputed from the dumps equal RESULT_NAVRBD
    bk = banked["D1_HO_each_acc"]
    kre = {"acc": {}, "delta": {}, "pass": True}
    for a in ("N1R_s0", "N1R_s1", "N1D_s0", "N1D_s1", "N1D_s0Z", "N1D_s1Z"):
        kre["acc"][a] = {"live": float(acc[a].mean()), "banked": bk[a], "equal": bool(float(acc[a].mean()) == bk[a])}
        kre["pass"] &= kre["acc"][a]["equal"]
    for s in (0, 1):
        r = d1_row(acc[f"N1D_s{s}"], acc[f"N1R_s{s}"], eid, 0.042364016736401666)
        b = banked["D1_HO"][f"N1D_s{s}-N1R_s{s}"]
        ok = bool(r["delta"] == b["delta"] and round(r["lo"], 4) == round(b["lo"], 4) and round(r["hi"], 4) == round(b["hi"], 4))
        kre["delta"][f"N1D_s{s}-N1R_s{s}"] = {"live": [r["delta"], r["lo"], r["hi"]], "banked": [b["delta"], b["lo"], b["hi"]], "equal": ok}
        kre["pass"] &= ok
    rep["controls"]["K-RE"] = kre
    if not kre["pass"]:
        prob.append("K-RE: N1D - N1R recomputed from the dumps != RESULT_NAVRBD.json")
        rep["K-RE_FAILED_STOP"] = True
    # ---------------- K1, K-CONST
    k1 = paired(acc["N1R_s0"], acc["N1R_s0"], eid)
    rep["controls"]["K1_self_pair_N1R_s0"] = {"delta": k1["delta"], "lo": k1["lo"], "hi": k1["hi"], "pass": bool(k1["delta"] == 0.0 and k1["lo"] == 0.0 and k1["hi"] == 0.0)}
    maj = int(np.bincount(glat, minlength=3).argmax())
    share = float((glat == maj).mean())
    cacc = float((np.full(len(glat), maj) == glat).mean())
    rep["controls"]["K-CONST"] = {"majority_class": CL[maj], "majority_share": share, "constant_accuracy": cacc, "must_read": 0.338215, "pass": bool(cacc == share and round(share, 6) == 0.338215)}
    for k in ("K1_self_pair_N1R_s0", "K-CONST"):
        if not rep["controls"][k]["pass"]:
            prob.append(f"{k} failed")
    if prob:
        rep["status"] = "INVALID (control problems; labels unvalidated)"
    # ---------------- floors and bars
    f_banked = float(banked["floor"]["F_HO_banked"])
    f_live = float(abs(acc["N1R_s1"].mean() - acc["N1R_s0"].mean()))
    f_ho = max(f_banked, f_live)
    seed_draw = {"N1D": float(abs(acc["N1D_s1"].mean() - acc["N1D_s0"].mean()))}
    for w in (2, 4):
        seed_draw[f"N1DW{w}"] = float(abs(acc[f"N1DW{w}_s1"].mean() - acc[f"N1DW{w}_s0"].mean()))
    bar = {w: float(max(BAR_FLOOR, f_ho, seed_draw["N1D"], seed_draw[f"N1DW{w}"])) for w in (2, 4)}
    rep["floor"] = {"F_HO_banked": f_banked, "F_HO_live": f_live, "F_HO_used": f_ho, "seed_draws": seed_draw, "bar_rule": "max(0.0383, F_HO, |N1D_s1-N1D_s0|, |N1DW_w_s1-N1DW_w_s0|)", "bar_by_weight": {f"w{w}": bar[w] for w in bar},
                    "N1D_seed_draw_note": "card estimated ~0.0303 from rounded levels; recomputed exactly here from the banked dumps"}
    # ---------------- D1-HO deciding and beside
    D1, D1D, D2 = {}, {}, {}
    for w in (2, 4):
        for s in (0, 1):
            arm = f"N1DW{w}_s{s}"
            D1[arm] = d1_row(acc[arm], acc[f"N1R_s{s}"], eid, bar[w])
            D1D[arm] = d1_row(acc[arm], acc[f"N1D_s{s}"], eid, bar[w])
            D2[arm] = {"vs_N1R": d2_ho(glat, plat[arm], plat[f"N1R_s{s}"], eid), "vs_N1D": d2_ho(glat, plat[arm], plat[f"N1D_s{s}"], eid)}
    rep["D1_HO_deciding_N1DW-N1R"] = D1
    rep["D1_HO_beside_N1DW-N1D"] = D1D
    rep["D2_HO_beside"] = D2
    rep["D1_HO_each_acc"] = {a: float(v.mean()) for a, v in acc.items()}
    rep["gt_class_share"] = {c: float((glat == i).mean()) for i, c in enumerate(CL)}
    # ---------------- R1 LON / LAT (+ PRIMARY, TAC-LON) beside, no bar
    R1 = {}
    for w in (2, 4):
        for s in (0, 1):
            arm = f"N1DW{w}_s{s}"
            for refname, refarm in (("N1DW-N1R", f"N1R_s{s}"), ("N1DW-N1D", f"N1D_s{s}")):
                rows = {}
                for c in CELLS:
                    r = paired(C[arm][c], C[refarm][c], eid)
                    if r["status"] != "OK":
                        r["reason"] = "no lead block exists for held-out clips (b1_eval_lead_block / join_eval141 cover eval-141 only); n = 0"
                    r["family"] = FAMILY[c.split(":")[-1]]
                    r["higher_is_better"] = c in ("TAC-LAT", "TAC-LON")
                    rows[c] = r
                R1.setdefault(arm, {})[refname] = rows
    rep["R1_LON_LAT_HO_beside_no_bar"] = R1
    # ---------------- Z-USE
    zuse = {}
    for a in [f"N1DW{w}_s{s}" for w in (2, 4) for s in (0, 1)] + ["N1D_s0", "N1D_s1"]:
        r = paired(acc[a], acc[a + "Z"], eid)
        zuse[a] = r
    rep["Z_USE_HO_real_minus_zero"] = zuse
    # ---------------- K-CLIP
    kc = {}
    for w in (2, 4):
        for s in (0, 1):
            kc[f"N1DW{w}_s{s}"] = kclip(f"{R}/n1dw/runs/N1DW{w}_s{s}/train/train_log.jsonl")
    for s in (0, 1):
        p = f"{R}/navrbd/runs/N1D_s{s}/train/train_log.jsonl"
        kc[f"N1D_s{s}"] = kclip(p) if os.path.exists(p) else {"status": "train_log missing", "path": p}
    rep["K-CLIP"] = kc
    # ---------------- outcome rules, D1-HO alone
    gain = {(w, s): D1[f"N1DW{w}_s{s}"]["gain"] for w in (2, 4) for s in (0, 1)}
    cost = {(w, s): D1[f"N1DW{w}_s{s}"]["cost"] for w in (2, 4) for s in (0, 1)}
    dlt = {(w, s): D1[f"N1DW{w}_s{s}"]["delta"] for w in (2, 4) for s in (0, 1)}
    dD = {(w, s): D1D[f"N1DW{w}_s{s}"] for w in (2, 4) for s in (0, 1)}
    PG = "PENDING-GUARD"
    lab = {}
    both = {w: gain[(w, 0)] and gain[(w, 1)] for w in (2, 4)}
    one = {w: (gain[(w, 0)] != gain[(w, 1)]) for w in (2, 4)}
    any_gain = any(gain.values())
    lab["RBW-GAIN / RBW-GAIN-ONE-COST (D1-HO gain at both seeds at a weight)"] = (PG + f" (D1-HO gain at both seeds at weight(s) {[w for w in (2, 4) if both[w]]}; which label depends on the guard; provisional until N1DWx)" if any(both.values())
                                                                                  else "not fired: no weight gains at both seeds on D1-HO")
    lab["RBW-ONE-SEED (gain at exactly one seed at a weight, guard holds)"] = (PG + f" (D1-HO part holds at weight(s) {[w for w in (2, 4) if one[w]]})" if any(one.values()) else "not fired: no weight gains at exactly one seed")
    split = (not any(both.values())) and any(gain[(4, s)] and gain[(2, 1 - s)] for s in (0, 1)) and not (one[4] and one[2] and False)
    lab["RBW-SPLIT (no weight gains at both seeds; w4 gains at one seed and w2 at the other)"] = ("D1-HO condition HOLDS (card gives no guard clause); tie rule: larger mean two-seed N1DW-N1R, within 0.005 -> smaller weight"
                                                                                                    if split else "not fired")
    m = {w: 0.5 * (dlt[(w, 0)] + dlt[(w, 1)]) for w in (2, 4)}
    rep["mean_two_seed_N1DW-N1R"] = {f"w{w}": m[w] for w in m}
    dose = (not any_gain) and all(dD[(4, s)]["separated"] and dD[(4, s)]["delta"] > 0 for s in (0, 1)) and all(dD[(4, s)]["delta"] >= dD[(2, s)]["delta"] for s in (0, 1))
    lab["RBW-DOSE (no gain at either weight; N1DW4-N1D separated >0 at both seeds and >= N1DW2-N1D at both seeds)"] = "FIRED" if dose else "not fired"
    lab["RBW-NO-GAIN (no gain at either weight and RBW-DOSE does not hold)"] = "FIRED (D1-HO alone decides; the probe of section 10 picks the next lever)" if ((not any_gain) and not dose) else "not fired"
    costboth = [w for w in (2, 4) if cost[(w, 0)] and cost[(w, 1)]]
    lab["RBW-COSTS"] = (f"FIRED on the D1-HO part at weight(s) {costboth}" if costboth else PG + " (D1-HO shows no cost at both seeds at any weight; the guard-cell part needs eval-141)")
    fh = {f"w{w}": bool(bar[w] > BAR_FLOOR and all(bar[w] > dlt[(w, s)] for s in (0, 1))) for w in (2, 4)}
    fh_all = bool(all(bar[w] > BAR_FLOOR for w in bar) and all(bar[w] > dlt[(w, s)] for w in (2, 4) for s in (0, 1)))
    lab["RBW-FLOOR-HIGH (bar above 0.0383 and above every observed N1DW - N1R)"] = f"{'FIRED' if fh_all else 'not fired'} (per weight: {fh})"
    attrib = []
    for w in (2, 4):
        for s in (0, 1):
            if gain[(w, s)] and not dD[(w, s)]["separated"]:
                attrib.append(f"N1DW{w}_s{s}: the gain is not attributed to the weight over the channel; N1DWx comes first")
    rep["outcome_D1HO_alone"] = {"labels": lab, "gain_by_arm": {f"N1DW{w}_s{s}": gain[(w, s)] for w in (2, 4) for s in (0, 1)}, "attribution_notes": attrib,
                                 "N1DWx_read": "NOT-APPLICABLE (no N1DWx arm exists; R4 read fires only after a provisional RBW-GAIN)", "strategic_and_guard": PG}
    rep["wall_s"] = round(time.time() - t0, 1)
    json.dump(rep, open(OUT + "/RESULT_N1DW_D1HO.json", "w"), indent=1, default=float)
    # ---------------- text
    L = []
    P = L.append
    P("READ_N1DW_D1HO (MEASURED; T1-kin; D1-HO 5,736 windows / 497 held-out episodes; pod CPU; B 2,000 seed 0 paired episode-cluster bootstrap = 'another draw of EPISODES' only)")
    P("card PREREG_V8F_N1DW.md sha256 59f2b7f9 + Amendment 1; eval-141 guard and STRATEGIC NOT computed here: every label that needs them is PENDING-GUARD")
    P(f"problems: {prob or 'none'}")
    P("")
    P("CONTROLS")
    P(f"  K-RE   : {'PASS' if kre['pass'] else 'FAIL'} (N1R/N1D/N1D-zero accuracies and N1D-N1R deltas recomputed from dumps vs RESULT_NAVRBD.json)")
    for a, v in kre["acc"].items():
        P(f"           {a:8s} live {v['live']:.6f} banked {v['banked']:.6f} equal {v['equal']}")
    P(f"  K1     : N1R_s0 self-pair delta {k1['delta']} lo {k1['lo']} hi {k1['hi']} -> {'PASS' if rep['controls']['K1_self_pair_N1R_s0']['pass'] else 'FAIL'}")
    P(f"  K-CONST: majority {CL[maj]} share {share:.6f}, constant predictor {cacc:.6f} (must read 0.338215) -> {'PASS' if rep['controls']['K-CONST']['pass'] else 'FAIL'}")
    P("")
    P(f"FLOORS: F_HO banked {f_banked:.6f}, live {f_live:.6f}; seed draws |N1D_s1-N1D_s0| {seed_draw['N1D']:.4f}, |N1DW2_s1-N1DW2_s0| {seed_draw['N1DW2']:.4f}, |N1DW4_s1-N1DW4_s0| {seed_draw['N1DW4']:.4f}")
    P(f"  bar w2 = {bar[2]:.6f}   bar w4 = {bar[4]:.6f}   (rule max(0.0383, F_HO, |N1D seed draw|, |N1DW_w seed draw|))")
    P("")
    P("D1-HO accuracy by arm: " + ", ".join(f"{a} {v:.4f}" for a, v in rep["D1_HO_each_acc"].items() if not a.endswith("Z")))
    P("")
    P("DECIDING  D1-HO N1DW - N1R (paired, same windows)       delta [95% CI]            sep  gain(>=bar & lo>0)  cost   | beside N1DW - N1D delta [CI] sep")
    for arm in D1:
        r, q = D1[arm], D1D[arm]
        P(f"  {arm:10s} {r['delta']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}]  sep {r['separated']!s:5s} gain {r['gain']!s:5s} cost {r['cost']!s:5s} bar {r['bar']:.4f} | {q['delta']:+.4f} [{q['lo']:+.4f},{q['hi']:+.4f}] sep {q['separated']}")
    P("")
    P("Z-USE (channel real - zero, D1-HO TAC-LAT)")
    for a, r in zuse.items():
        P(f"  {a:10s} {r['delta']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] sep {r['separated']}")
    P("")
    P("K-CLIP (share of logged steps with grad_norm > 1.0)")
    for a, r in kc.items():
        P(f"  {a:10s} " + (f"{r['clipped_steps']} / {r['logged_steps']} = {r['share']:.3f}" if "share" in r else str(r)))
    P("")
    P("D2-HO beside (turn recall / false-turn share / side-given-turn), N1DW vs N1R")
    for arm, d in D2.items():
        v = d["vs_N1R"]
        P(f"  {arm:10s} recall {v['turn_recall']['arm']:.3f} vs {v['turn_recall']['ref']:.3f} ({v['turn_recall']['delta']:+.3f} sep {v['turn_recall']['separated']}); false-turn {v['false_turn_share']['arm']:.3f} vs {v['false_turn_share']['ref']:.3f} ({v['false_turn_share']['delta']:+.3f} sep {v['false_turn_share']['separated']}); side {v['side_given_turn_arm']['value']:.3f} vs {v['side_given_turn_ref']['value']:.3f}")
    P("")
    P("R1 LONGITUDINAL / LATERAL on D1-HO, no bar (delta [CI], * = separated); N1DW - N1R then N1DW - N1D. dk (headway / time gap) cells: NOT-COMPUTABLE, n = 0 (no lead block for held-out clips)")
    for arm in R1:
        for refname in ("N1DW-N1R", "N1DW-N1D"):
            P(f"  {arm} {refname}")
            for c in CELLS:
                r = R1[arm][refname][c]
                if r["status"] != "OK":
                    P(f"     {c:36s} NOT-COMPUTABLE n=0")
                else:
                    P(f"     {c:36s} [{r['family'][:4]}] {r['delta']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}]{'*' if r['separated'] else ''}")
    P("")
    P("OUTCOME (card section 7, as far as D1-HO alone allows)")
    for k, v in lab.items():
        P(f"  {k}: {v}")
    for a in attrib:
        P("  ATTRIBUTION: " + a)
    P(f"  mean two-seed N1DW - N1R: w2 {m[2]:+.4f}, w4 {m[4]:+.4f}")
    P("  N1DWx read: NOT-APPLICABLE (no N1DWx arm)")
    P(f"wall {rep['wall_s']} s")
    open(OUT + "/READ_N1DW_D1HO.txt", "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L[:40]))
    print("WROTE", OUT, "problems", len(prob), flush=True)
    sys.exit(1 if prob else 0)


if __name__ == "__main__":
    main()
