"""E5 -- the T2 scorer. It REUSES the T1 reader's own functions for every row (nothing is re-implemented):
  rr_lib.rows_of / rr_lib.m1_window            the 10 comps and lateral rows and the M1 crawl share (per window)
  t1kf_read_a1.prep / Ctx / labels             TAC-LAT, TAC-LON (kinematic labeller on executed vs logged path), DK-HW, DK-TG, DK-VIOL
  lead_source.lead_block                       the lead block (state, lead track, length, ego speed) the DK rows are joined against
Loaded BY PINNED PATH (sha256 checked) the way dag2_read.ReadA loads them; the heavy readers bind `taniteval` from the reader tree
(dagt1_tree_b43c7a9a), so run this module in its OWN process (the tests call it through `run_sub`), never next to the E1 policy's tree.

Labels (19, the reader's order): ADE6 .. cross-track, L1, L2, DK-VIOL, DK-HW, DK-TG, TAC-LAT, TAC-LON, S-POS@30/50/80, M1 crawl share.
`own_rows(X, G, ctx)` -> {label: per-window array}. A row whose input is missing is ABSENT from the dict and named in `missing`
with its reason (never filled with zeros). Tier stamp: T2 re-perception rollout; never pooled with T1-kin.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

ART = "E:/Projects/TanitAD-artifacts"
REF = f"{ART}/dagorc_rebase/read/ref"
READERS = f"{ART}/dagreread/reader_src"
TREE = f"{ART}/dagt1_tree_b43c7a9a"
LEAD_BLOCK = f"{REF}/b1_eval_lead_block_k30.npz"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PINS = {f"{REF}/readA_code/rr_lib.py": "c3800878cae80bf3018284528e68ed5f88dcebbab7d1125749942735e6cebc99",
        f"{READERS}/t1kf_read_a1.py": "d16ddd607caf4e05d0bbb667b830299df1f5c1eb0923d995988667afc940e838",
        f"{READERS}/t1k_read.py": "5610114d3e09501347ed0ac65e44790eb2954eb5d5b703af089a64e845dbf5e4",
        f"{READERS}/t1klc_read.py": "9a5623e5f737677e6fe0cf6df41c3588b11282423e27008830a7914253f7a7ca",
        f"{READERS}/lcdiag_common.py": "798089e8abfeb56dfe9e8f6a755448f416be8dfbec948df08638fa997c038a06"}
LABELS = ("ADE6", "speed MAE", "along-track MAE", "accel MAE", "DK-VIOL", "DK-HW", "DK-TG", "heading MAE", "yaw-rate MAE", "curvature MAE", "cross-track MAE",
          "L1 lateral-accel MAE", "L2 curvature MAE (GT>=5.2)", "TAC-LAT", "TAC-LON", "S-POS@30", "S-POS@50", "S-POS@80", "M1 crawl share")
COMPS_KEY = {"ADE6": "ade6_m", "speed MAE": "speed_mae_mps", "along-track MAE": "along_mae_m", "accel MAE": "accel_mae_mps2", "heading MAE": "heading_mae_deg",
             "yaw-rate MAE": "yaw_rate_mae_degps", "curvature MAE": "curvature_mae_1pm", "cross-track MAE": "cross_mae_m", "L1 lateral-accel MAE": "L1_lat_accel_mae_mps2",
             "L2 curvature MAE (GT>=5.2)": "L2_curvature_mae_fast_1pm"}
FAMILY = {"ADE6": "LONGITUDINAL+LATERAL", "speed MAE": "LONGITUDINAL", "along-track MAE": "LONGITUDINAL", "accel MAE": "LONGITUDINAL", "DK-VIOL": "LONGITUDINAL",
          "DK-HW": "LONGITUDINAL", "DK-TG": "LONGITUDINAL", "M1 crawl share": "LONGITUDINAL", "heading MAE": "LATERAL", "yaw-rate MAE": "LATERAL",
          "curvature MAE": "LATERAL", "cross-track MAE": "LATERAL", "L1 lateral-accel MAE": "LATERAL", "L2 curvature MAE (GT>=5.2)": "LATERAL",
          "TAC-LAT": "TACTICAL", "TAC-LON": "TACTICAL", "S-POS@30": "STRATEGIC", "S-POS@50": "STRATEGIC", "S-POS@80": "STRATEGIC"}
TIER = "T2 re-perception rollout (never pooled with T1-kin)"
_H = {}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def init():
    """Load rr_lib and t1kf_read_a1 by pinned path (as dag2_read.ReadA does). -> dict(R=rr_lib, T=t1kf)."""
    if _H:
        return _H
    for p, want in PINS.items():
        got = sha256_file(p)
        if got != want:
            raise RuntimeError(f"REFUSED: {p} is {got}, not the pinned {want}")
    os.environ["RR_READERS"] = READERS
    sys.path.insert(0, f"{REF}/readA_code")
    import rr_lib as R
    if os.path.normpath(R.TREE) != os.path.normpath(TREE):
        raise RuntimeError(f"rr_lib is bound to {R.TREE}, not {TREE}")
    argv0 = sys.argv[:]
    sys.argv = ["t1kf", "--tree", TREE, "--t1k-read", f"{READERS}/t1k_read.py", "--lead-block", LEAD_BLOCK, "--out", "unused.json"]
    try:
        spec = importlib.util.spec_from_file_location("t1kf", f"{READERS}/t1kf_read_a1.py")
        T = importlib.util.module_from_spec(spec)
        sys.modules["t1kf"] = T
        spec.loader.exec_module(T)
    finally:
        sys.argv = argv0
    for m in (R.FF, T.LM, T.DK):
        if os.path.abspath(m.__file__).replace("\\", "/").upper().startswith("G:/"):
            raise RuntimeError("an instrument was imported from G:")
    _H.update(R=R, T=T)
    return _H


def context_from_lead(G, eid, Jd):
    """The reader's `Ctx` for windows with logged paths G [N,30,2], episode (scene) ids `eid` and a joined lead block `Jd` (keys leads, lead_lens, speeds, ...)."""
    T = init()["T"]
    G = np.asarray(G, np.float64)
    hdk = T.LM.distance_keeping(G[:, :T.K], Jd["leads"], Jd["lead_lens"], Jd["speeds"], T.DT)
    return T.Ctx(G, np.asarray(eid), Jd, hdk)


def own_rows(X, G, ctx=None, name="X", spos=None):
    """Per-window arrays of the reader's rows for ONE executed track set X [N,30,2] against the logged paths G. -> (rows, missing).
    `spos`: {'S-POS@30/50/80': [N]} from `t2v8f_lead.spos_from_records` (S - CTRV, ticks >= 1); without it the three rows are NOT COMPUTED, named, with the reason."""
    H = init()
    R, T = H["R"], H["T"]
    X, G = np.asarray(X, np.float64), np.asarray(G, np.float64)
    cx = R.rows_of(X, G)
    rows = {lab: np.asarray(cx[k], np.float64) for lab, k in COMPS_KEY.items()}
    rows["M1 crawl share"] = np.asarray(R.m1_window(X, G)[0], np.float64)
    missing = {}
    if ctx is None:
        for lab in ("TAC-LAT", "TAC-LON", "DK-HW", "DK-TG", "DK-VIOL"):
            missing[lab] = "no lead block / context supplied"
    else:
        P = T.prep(name, X, ctx)
        for lab in ("TAC-LAT", "TAC-LON"):
            rows[lab] = np.asarray(P["ind"][lab], np.float64)
        for lab in ("DK-HW", "DK-TG"):
            rows[lab] = np.asarray(P["abs"][lab], np.float64)
        va = T.LM._pw_get(P["lv"], ("viol", "lead_viol"))
        ra = T.LM._pw_get(P["lv"], ("at_risk", "lead_at_risk"))
        rows["DK-VIOL"] = np.where(ra == 1.0, va, np.nan)
    for lab in ("S-POS@30", "S-POS@50", "S-POS@80"):
        if spos is not None and lab in spos:
            rows[lab] = np.asarray(spos[lab], np.float64)
        else:
            missing[lab] = "S head checkpoints per tick (policy extra s_ckpt / s_pose) were not supplied"
    return rows, missing


def n_valid(rows):
    return {k: int(np.isfinite(v).sum()) for k, v in rows.items()}


def contrast(vx, vy, eid):
    """Paired scene-cluster bootstrap of mean(vx - vy) on windows where both sides are finite: the reader's `paired` (t1k_read.paired: B 2000, seed 0, taniteval/ci.py)."""
    return init()["T"].paired(vx, vy, eid)


# ------------------------------------------------------------------------------------------------ the bootstrap over scenes (X1, X2, BLIND)
M1 = "M1 crawl share"


def pair_check(keys_a, keys_b):
    """Arms are contrasted only on the SAME starts: the (scene, f0) lists must be identical (never positional trust)."""
    if list(keys_a) != list(keys_b):
        raise ValueError("PAIRING GUARD: the two arms' (scene, f0) lists differ; a contrast on misaligned starts is not a contrast")


def scene_contrast(vx, vy, scene_eid, mask=None):
    """Paired scene-cluster bootstrap of mean(vx - vy): `eid` = the SCENE index (the two starts of a scene are one cluster), B 2000, seed 0, taniteval/ci.py (the reader's
    `paired`). `mask` selects windows (e.g. launch scenes for M1); windows where either side is NaN are dropped by the reader's own rule."""
    vx, vy, e = np.asarray(vx, np.float64), np.asarray(vy, np.float64), np.asarray(scene_eid)
    if mask is not None:
        m = np.asarray(mask, bool)
        vx, vy, e = vx[m], vy[m], e[m]
    return contrast(vx, vy, e)


def x_contrasts(rows_t2, rows_p10, rows_rr, scene_eid, mask=None, label=M1, rows_blind=None):
    """X1 = C0(T2) - P10(T2) and X2 = C0(T2) - C0(RR) on one row (default M1), one seed. `rows_blind`: BLIND(T2) given -> also BLIND - C0(T2) (outcome 2's sub-branch).
    -> {'X1','X2'[,'BL']}: delta / lo / hi / separated / n_windows / n_episodes (episodes = scenes)."""
    out = {"X1": scene_contrast(rows_t2[label], rows_p10[label], scene_eid, mask), "X2": scene_contrast(rows_t2[label], rows_rr[label], scene_eid, mask)}
    if rows_blind is not None:
        out["BL"] = scene_contrast(rows_blind[label], rows_t2[label], scene_eid, mask)
    return out


# ------------------------------------------------------------------------------------------------ card section 6, a pure function of the contrasts
H_SEED = {0: 0.0267, 1: 0.0298}                    # half the T1-kin reindexed excess (card section 5)
_SEEDS = (0, 1)


def _need(c, what):
    for k in ("delta", "lo", "hi", "separated"):
        if k not in c:
            raise ValueError(f"outcome(): {what} lacks '{k}'")
    for k in ("delta", "lo", "hi"):
        if not math.isfinite(float(c[k])):
            raise ValueError(f"outcome(): {what}.{k} is not finite (an undecided contrast is never read)")


def outcome(x1, x2, bl, h=None, gates_ok=True):
    """PREREG_T2_CRAWL section 6, in the card's order. x1[s], x2[s]: {delta, lo, hi, separated} for seed s in {0, 1}; bl[s]: BLIND - C0(T2) on M1 (same keys) or None; h: {seed: bar}.
    READING (every clause the card leaves open is fixed here, and listed in HANDOFF_B.md):
      1 CRAWL-CLOSES   at BOTH seeds: x1.hi < h  AND  x2 separated with delta < 0.
      2 CRAWL-PERSISTS at BOTH seeds: x1.lo > 0 AND x1.delta >= h AND x1.delta > |x1_s1 - x1_s0|.  Sub-branch on BLIND - C0(T2) for M1: (b) separated at BOTH seeds, (a) separated at
        neither; one seed only is not in the card, so branch = None with a note (the PI rules).  bl None -> branch None.
      3 MIXED          some bound decides at some seed (x1.hi < h, or x1.lo > 0) but 1 and 2 do not hold: "either outcome at one seed only" and "X1 separated and below h" are both inside it.
      4 UNDERPOWERED   at both seeds x1.hi >= h and x1.lo <= 0: neither bound decides.
      5 GATE-FAIL      any gate failed (gates_ok False): the instrument is fixed first and NO contrast is read (not even validated).
    -> {outcome, branch, seed_gap, notes}."""
    if not gates_ok:
        return {"outcome": "GATE-FAIL", "branch": None, "seed_gap": None, "notes": ["a gate failed: the instrument is fixed first; no contrast is read"]}
    h = h or H_SEED
    for s in _SEEDS:
        _need(x1[s], f"x1[{s}]")
        _need(x2[s], f"x2[{s}]")
        if bl is not None:
            _need(bl[s], f"bl[{s}]")
    seed_gap = abs(x1[1]["delta"] - x1[0]["delta"])
    closes = all(x1[s]["hi"] < h[s] and x2[s]["separated"] and x2[s]["delta"] < 0 for s in _SEEDS)
    persists = all(x1[s]["lo"] > 0 and x1[s]["delta"] >= h[s] and x1[s]["delta"] > seed_gap for s in _SEEDS)
    if closes:
        return {"outcome": "CRAWL-CLOSES", "branch": None, "seed_gap": seed_gap, "notes": []}
    if persists:
        notes, branch = [], None
        if bl is not None:
            sep = [bool(bl[s]["separated"]) for s in _SEEDS]
            if all(sep):
                branch = "b"
            elif not any(sep):
                branch = "a"
            else:
                notes.append("BLIND - C0(T2) is separated at one seed only: the card names no sub-branch for it (PI decision)")
        return {"outcome": "CRAWL-PERSISTS", "branch": branch, "seed_gap": seed_gap, "notes": notes}
    decides = any(x1[s]["hi"] < h[s] or x1[s]["lo"] > 0 for s in _SEEDS)
    return {"outcome": "MIXED" if decides else "UNDERPOWERED", "branch": None, "seed_gap": seed_gap, "notes": []}


# ------------------------------------------------------------------------------------------------ subprocess helper (own process: the readers bind their own tree)
def run_sub(op, **kw):
    """Run `python -I t2v8f_score.py op json` and return its JSON; the scorer never shares a process with the E1 policy's tree."""
    r = subprocess.run([PY, "-I", f"{HERE}/t2v8f_score.py", op, json.dumps(kw)], capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"))
    if r.returncode:
        raise RuntimeError(f"scorer subprocess failed rc={r.returncode}: {r.stderr[-1500:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def _eq(a, b):
    return bool(np.array_equal(np.asarray(a, np.float64), np.asarray(b, np.float64), equal_nan=True))


def op_t4(arr, dump, floor_dump, mut=None):
    """T4 pin: the scorer's own rows on the banked T1-kin dump equal the reader's banked per-window arrays (a_k for the arm, b_k for the P10 floor), bit for bit."""
    R = init()["R"]
    T = init()["T"]
    z = np.load(arr, allow_pickle=True)
    labels = [str(q) for q in z["labels"]]
    d, dref = R.load_dir(dump), R.load_dir(floor_dump)
    G = d["g"]
    X, Y = d["cl"].copy(), dref["fl"].copy()
    if mut == "x_half":
        X = X.copy()
        X[:, :, 0] *= 0.5                                  # mutation: the executed track is wrong in length
    if mut == "m1_vmin":
        R.VMIN = 6.0                                       # mutation: a different crawl threshold (the M1 row alone moves)
    join = np.load(f"{TREE}/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/join_eval141.npz", allow_pickle=False)
    clip = join["clip"].astype(str)[dref["clip_index"]]
    blk = T.DK.load_block(LEAD_BLOCK)
    Jd = T.DK.join_block(blk, clip, 2 * dref["ws"].astype(np.int64), dref["v0"], k=T.K, dt=T.DT)
    cx = context_from_lead(dref["g"], dref["eid"], Jd)
    ra, _ = own_rows(X, G, cx, "RIC0")
    rb, _ = own_rows(Y, G, cx, "P10")
    out = {"n_windows": int(len(G)), "rows": {}}
    for i, lab in enumerate(labels):
        if lab not in ra:
            out["rows"][lab] = "NOT COMPUTED (S-POS)"
            continue
        ea, eb = _eq(ra[lab], z[f"a_{i}"]), _eq(rb[lab], z[f"b_{i}"])
        out["rows"][lab] = {"a_bit_identical": ea, "b_bit_identical": eb,
                            "max_abs_dev_a": float(np.nanmax(np.abs(ra[lab] - z[f"a_{i}"]))) if not ea else 0.0,
                            "max_abs_dev_b": float(np.nanmax(np.abs(rb[lab] - z[f"b_{i}"]))) if not eb else 0.0}
    out["m1_share_pooled_a"] = float(np.nanmean(ra["M1 crawl share"]))
    return out


def op_t3_attr(floor_dump, mut=None):
    """T3 (scorer side): an attribution of P10 against ITSELF reads exactly 0 (delta, lo, hi) on every computed row, on the banked P10 floor tracks."""
    R = init()["R"]
    T = init()["T"]
    dref = R.load_dir(floor_dump)
    join = np.load(f"{TREE}/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/join_eval141.npz", allow_pickle=False)
    clip = join["clip"].astype(str)[dref["clip_index"]]
    Jd = T.DK.join_block(T.DK.load_block(LEAD_BLOCK), clip, 2 * dref["ws"].astype(np.int64), dref["v0"], k=T.K, dt=T.DT)
    cx = context_from_lead(dref["g"], dref["eid"], Jd)
    X, Y = dref["fl"].copy(), dref["fl"].copy()
    if mut == "y_shifted":
        Y[:, :, 0] += 1e-6                                 # mutation: the 'self' arm is not the same track
    ra, _ = own_rows(X, dref["g"], cx, "P10")
    rb, _ = own_rows(Y, dref["g"], cx, "P10")
    worst, per = 0.0, {}
    for lab, v in ra.items():
        r = contrast(v, rb[lab], dref["eid"])
        if r.get("status") != "OK":
            per[lab] = r.get("status")
            continue
        per[lab] = [r["delta"], r["lo"], r["hi"]]
        worst = max(worst, abs(r["delta"]), abs(r["lo"]), abs(r["hi"]))
    return {"n_rows": len(per), "max_abs_delta_lo_hi": worst, "per_row": per}


if __name__ == "__main__":
    op, kw = sys.argv[1], json.loads(sys.argv[2])
    if op == "t4":
        print(json.dumps(op_t4(**kw)))
    elif op == "t3_attr":
        print(json.dumps(op_t3_attr(**kw)))
    elif op in ("dk_pin", "rc_pin", "spos_pin", "x12", "records_smoke"):
        sys.path.insert(0, HERE)                          # python -I drops the script dir; the lead module imports this file, and must get THIS module (not a second copy)
        sys.modules.setdefault("t2v8f_score", sys.modules["__main__"])
        import t2v8f_lead as L
        print(json.dumps(getattr(L, "op_" + op)(**kw)))
    else:
        raise SystemExit(f"unknown op {op}")
