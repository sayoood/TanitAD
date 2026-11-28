#!/usr/bin/env python3
"""VDT-READ: calibration, navtest table read, controls and outcome logic of PREREG_VDAC_TRK.md (H-VDAC-TRK-1; card sha256 099a429f...).  CPU only, no GPU.

Implements the FROZEN card sections 3-8 exactly.  No bar, grid, estimator or control threshold below is a choice of this file; each is the card's.
Reuses (does not re-implement) `refe/safehead_data.agg32` (the one selection arithmetic) and `eval/p5c/p5c_read.py` Boot (paired log-cluster bootstrap,
10,000 draws, seed 20260927, 95 % percentile), and follows the table-lookup + controls pattern of trkmin_read.py / l2next_read.py.

  calibrate  python vdt_read.py calibrate --pk <PK> --dump <hc40_dump.npz> --lab-glob "<dir>/*.lab.npz" --out <cal.json>
                 [--hc40-list <hc40_logs.txt>] [--iou-json <iou.json>] [--navtest-logs <navtest_logs.txt>]
             -> m* per head over the fixed grid (HC40 only; navtest is never read), CAL-NULL, C-SHUF / C-FLIP, IoU-free diagnostics, gain curve.
  read       python vdt_read.py read --pk <PK> --dump <vdt_navtest_dump.npz> --table "<p5c_next>/propscore/full_C1_s*of2.jsonl" --cal <cal.json>
                 --trk-npz <trk_c1.npz> --out-dir <dir> [--g0-json <g0trk_c1.json>] [--dump2 <second_pass.npz>]
             -> READ_VDT.json (arms VT, VT-r, VT-SHUF, VT-0, O-DAC-T; bars P1 P2 M G R X; every control; the PROVISIONAL outcome) + vdt_picks.npz.
  confirm    python vdt_read.py confirm --read <dir>/READ_VDT.json --table ... --csv-vt <harness csv of VT's picks> [--csv-vtr <csv>] [--families <json>] --out-dir <dir>
             -> CTRL-T (harness CSV == table at VT's picks, 1e-6 on every token) + the four families -> VERDICT_VDT.json (the FINAL outcome).

INPUT CONTRACT (what B2's `vdt_dump.py` must write; this file assumes exactly this):
  navtest dump  npz: token [N] str; s_plan [N,64,6] float32 (C1's logits); k_ship [N] int (C1's pick);
                     dk_HT_s0, dk_HT_s1, dk_HT_SHUF [N,64] float (d_k = min over the 41 tracked states x 4 corners of the head's signed distance, metres;
                     out-of-extent corner = +6 as in feed_scalars).  Optional second-pass file: same keys on a subset (determinism).
  HC40 dump     npz: the same keys, plus dk_HT_s0_flip [N,64] (HT-s0's field y-flipped), for tokens of the 40 HC40 logs.
  HC40 labels   one `<log>.lab.npz` per log from sh_label.py: token [n], tr_nc/tr_dac/tr_ddc/tr_ttc/tr_ep/tr_c/tr_pdms [n,64] (proposal axis = the dump's).
  IoU json      {"HT_s0": x, "HT_s1": x, "HT_SHUF": x} drivable IoU on HC40 (x in [0,40) m, mask logit > 0), from the B1/B2 side (this file never sees a field).
  G0-TRK-C1 json {"n_match": 777344, "n_total": 777344, "second_dump_bit_identical": true}.
  Families json {"VT": {"LONGITUDINAL": {comp: {"point","lo","hi","worse_if":"negative"|"positive"}}, "LATERAL": ..., "TACTICAL": ..., "STRATEGIC": ...}, "VT-r": {...}}.
A control whose input is not supplied reads pass=None (PENDING): it blocks every claim of finality and ADOPT, and is never counted as a pass.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------------------------------------------- the card's constants
GRID = (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0)        # section 5: the fixed margin grid, metres
P2_BAR = 0.2316                                          # section 6 P2: 2 * max(F, F_REFE 0.1158)
JUDGED = ("nc", "dac", "ttc", "ep", "c")                 # section 6 G / R sub-scores (DDC is reported, not judged)
M7 = ("nc", "dac", "ttc", "ep", "c", "ddc", "pdms")
HEADS = ("HT_s0", "HT_s1", "HT_SHUF")
N_TOKENS, N_LOGS = 12146, 136
C1_PDMS, C1_PDMS_TOL = 92.5548, 5e-5                     # CTRL-ID
T_TOL = 1e-6                                             # CTRL-T
G0_N = 777344
TOP_K = 8                                                # section 5: top-8 proposals by agg32 for the diagnostics
GAIN_EPS = 1e-9                                          # "positive point gain": above float-sum noise (an exact 0 stays 0)
TIE_EPS = 1e-12                                          # ties on the gain go to the smaller m
SHUF_IOU_MAX = 0.50                                      # C-SHUF
FAMILIES = ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC")
TABLE_SHA256 = {"full_C1_s0of2.jsonl": "9d1239a250c021b89480b0079f2d66ac7b319a3eaee73c80a3deac208b89b42e",
                "full_C1_s1of2.jsonl": "77588849ce5a3080e5ec941f71ba1b52f380fdc04d406849416e8c73c99a99d0"}   # pins as l2next_read.py (same banked table)
CARD_SHA256_PREFIX = "099a429f"
DEFAULT_PK = ("F:/TanitAD_p5cread_tip/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan")
NEXT = {"CAL-NULL": "HB (LoRA trunk copy for the head, P6 IoU +0.031) under its own card, and the agent-check line (D-VDAC-6) for NC",
        "ADVERSE": "HB under its own card if HC40's gain at m* was positive; else the DAC-veto family closes on C1 and the agent check carries the line. No margin is re-chosen on navtest",
        "NULL": "HB, then the agent check",
        "MECHANISM-SMALL": "HB (a better field), with VT-r reported",
        "SINGLE-SEED": "HT-s2, read by these bars unchanged",
        "ADOPT": "VT enters C1's selection (deploying is the PI's call); goal status vs 94.55, then the agent check (NC) on top of VT",
        "INVALID": "fix the failing instrument and re-run that step; no bar moves",
        "Unnamed": "reported as such; the MM names the next arm"}


# ---------------------------------------------------------------------------------------------------------------- reused instruments
def load_pk(pk=None):
    """-> (agg32, Boot, {file: sha256}) from a landed export of the refe-plan PK.  Refuses if Boot is not the card's estimator."""
    pk = pk or os.environ.get("VDT_PK") or DEFAULT_PK
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(pk) / "refe"))
    sys.path.insert(0, str(Path(pk) / "eval" / "p5c"))
    import safehead_data as D                                      # noqa: E402  agg32: the ONE selection arithmetic
    import p5c_read as P                                           # noqa: E402  Boot
    if (P.SEED, P.NBOOT) != (20260927, 10000):
        raise SystemExit(f"p5c_read.Boot is not the card's estimator (seed {P.SEED}, draws {P.NBOOT})")
    shas = {"safehead_data.py": sha256_file(Path(pk) / "refe" / "safehead_data.py"), "p5c_read.py": sha256_file(Path(pk) / "eval" / "p5c" / "p5c_read.py")}
    return D.agg32, P.Boot, shas


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def commit_free_mb() -> float:
    """Free commit (FreeVirtualMemory) in MB on the dev box; +inf off Windows (the gate is a dev-box rule)."""
    if os.name != "nt":
        return float("inf")
    r = subprocess.run(["powershell", "-NoProfile", "-Command", "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1KB,0)"],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip().replace(",", "."))
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------------------------------------------- data
def load_table(pattern: str, check_sha: bool = True):
    """The banked harness-exact navtest table -> ({token: i}, logs [N], tab {metric: [N,64] float64, 0..1}, shas, problems).  `tr` only (tracked sub-scores)."""
    files = sorted(glob.glob(pattern))
    shas = {Path(f).name: sha256_file(f) for f in files}
    probs = []
    if check_sha:
        if set(shas) != set(TABLE_SHA256):
            probs.append(f"table shards {sorted(shas)} != pinned {sorted(TABLE_SHA256)}")
        probs += [f"{k} sha256 {v[:12]} != pinned {TABLE_SHA256[k][:12]}" for k, v in shas.items() if k in TABLE_SHA256 and v != TABLE_SHA256[k]]
    toks, logs, cols = [], [], {m: [] for m in M7}
    for f in files:
        for ln in open(f, encoding="utf-8"):
            if not ln.strip():
                continue
            r = json.loads(ln)
            if "error" in r:
                continue
            toks.append(r["token"]); logs.append(r["log"])
            for m in M7:
                cols[m].append(np.asarray(r["tr"][m], np.float64))
    tab = {m: np.stack(cols[m]) for m in M7}
    return {t: i for i, t in enumerate(toks)}, np.array(logs), tab, shas, probs


def load_dump(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def load_labels(pattern: str):
    """sh_label.py outputs -> (tokens [n], logs [n], tab {metric: [n,64] float64 0..1})."""
    toks, logs, cols = [], [], {m: [] for m in M7}
    for f in sorted(glob.glob(pattern)):
        z = np.load(f, allow_pickle=False)
        lg = Path(f).name.split(".lab.npz")[0]
        t = [str(x) for x in z["token"]]
        toks += t; logs += [lg] * len(t)
        for m in M7:
            cols[m].append(np.asarray(z["tr_" + m], np.float64))
    if not toks:
        return np.array([], str), np.array([], str), {m: np.zeros((0, 64)) for m in M7}
    return np.array(toks), np.array(logs), {m: np.concatenate(cols[m]) for m in M7}


# ---------------------------------------------------------------------------------------------------------------- the rule
def veto_pick(a: np.ndarray, dk: np.ndarray, m: float, k_ship: np.ndarray) -> np.ndarray:
    """VT rule (card 3): veto k if d_k < m; pick = argmax C1 agg32 over the survivors; all vetoed => C1's pick."""
    surv = np.asarray(dk) >= m
    k = np.where(surv, a, -np.inf).argmax(1)
    none = ~surv.any(1)
    k[none] = np.asarray(k_ship)[none]
    return k


def dk_from_tracked_dac(dac: np.ndarray) -> np.ndarray:
    """The TRUE tracked-DAC field (+1 if tr.dac > 0 else -1): with m = 0 the veto is exactly FILTER-ORACLE's F_dac (and O-DAC-T)."""
    return np.where(np.asarray(dac) > 0, 1.0, -1.0)


def _delta(x: np.ndarray, k: np.ndarray, kship: np.ndarray) -> np.ndarray:
    ar = np.arange(len(k))
    return (x[ar, k] - x[ar, kship]) * 100


def _avg_rank(v) -> np.ndarray:
    v = np.asarray(v, float)
    u, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    return (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]


def auc(score, pos) -> float:
    """P(score_pos > score_neg), ties half; nan if one class is empty."""
    pos = np.asarray(pos, bool)
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = _avg_rank(score)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


# ---------------------------------------------------------------------------------------------------------------- calibration (card 5)
def calibrate_head(a, dk, tab, kship) -> dict:
    """HC40 only.  m* = argmax of the mean PDMS gain (x100) of the VT rule vs C1's pick over GRID; ties -> the smaller m; no positive gain => CAL-NULL."""
    ar = np.arange(len(kship))
    curve, best_m, best_g = [], None, None
    zero = ~(tab["dac"] > 0)
    top = np.argsort(-a, axis=1, kind="stable")[:, :TOP_K]
    for m in sorted(GRID):
        k = veto_pick(a, dk, m, kship)
        g = float(_delta(tab["pdms"], k, kship).mean())
        sw = k != kship
        veto = dk < m
        row = {"m": m, "gain_pdms_x100": g, "n_changed": int(sw.sum()), "n_recoveries": int((sw & zero[ar, kship]).sum()), "n_false_moves": int((sw & ~zero[ar, kship]).sum())}
        curve.append(row)
        if best_g is None or g > best_g + TIE_EPS:
            best_m, best_g = m, g
    null = not (best_g > GAIN_EPS)
    # IoU-free diagnostics at m* (or at the best-gain m when CAL-NULL; labelled)
    m_d = best_m
    t_dk, t_zero, t_veto = np.take_along_axis(dk, top, 1), np.take_along_axis(zero, top, 1), np.take_along_axis(dk < m_d, top, 1)
    n_zero, n_pass = int(t_zero.sum()), int((~t_zero).sum())
    diag = {"at_m": m_d, "at_m_is_selected": not null, "n_tokens": int(len(kship)), "top_k": TOP_K,
            "n_top8_dac0": n_zero, "n_top8_dac_pass": n_pass,
            "miss_rate": float((t_zero & ~t_veto).sum() / n_zero) if n_zero else None,
            "false_alarm_rate": float((~t_zero & t_veto).sum() / n_pass) if n_pass else None,
            "auc_neg_dk_dac0_top8": auc(-t_dk.ravel(), t_zero.ravel())}
    return {"curve": curve, "m_star": None if null else best_m, "m_best_gain": best_m, "gain_at_best_m": best_g, "cal_null": null, "diagnostics": diag}


def cmd_calibrate(a) -> int:
    agg32, _, shas = load_pk(a.pk)
    d = load_dump(a.dump)
    ltok, llog, lab = load_labels(a.lab_glob)
    ix = {t: i for i, t in enumerate(ltok)}
    dtok = d["token"].astype(str)
    keep = np.array([i for i, t in enumerate(dtok) if t in ix], int)
    li = np.array([ix[dtok[i]] for i in keep], int)
    n_dump, n_lab = len(dtok), len(ltok)
    tab = {m: lab[m][li] for m in M7}
    S, kship = d["s_plan"][keep], d["k_ship"][keep].astype(int)
    A = agg32(S)
    ctrl = {"CTRL-SEL-HC40": {"agg32_argmax_eq_k_ship": int((A.argmax(1) == kship).sum()), "n": int(len(keep)), "pass": bool((A.argmax(1) == kship).all()) and len(keep) > 0}}
    logs = sorted(set(llog[li]))
    ctrl["CTRL-HC40-SET"] = {"n_logs_labelled": len(logs), "n_dump_tokens": n_dump, "n_labelled_tokens_in_dump": int(len(keep)), "n_label_tokens": n_lab, "pass": None}
    if a.hc40_list:
        want = {x.strip() for x in open(a.hc40_list, encoding="utf-8") if x.strip()}
        ctrl["CTRL-HC40-SET"].update({"list_sha256": sha256_file(a.hc40_list), "n_list": len(want), "labelled_not_in_list": sorted(set(logs) - want)[:5],
                                      "pass": set(logs) <= want and len(want) == 40})
    if a.navtest_logs:
        nl = {x.strip() for x in open(a.navtest_logs, encoding="utf-8") if x.strip()}
        ctrl["CTRL-HC40-DISJOINT"] = {"overlap": sorted(set(logs) & nl)[:5], "pass": not (set(logs) & nl)}
    res = {"card_sha256_prefix": CARD_SHA256_PREFIX, "tier": "T1-class open loop on HC40 (navtrain), harness-exact tracked labels (sh_label.py)", "instruments_sha256": shas,
           "dump_sha256": sha256_file(a.dump), "grid": list(GRID), "n_hc40_tokens": int(len(keep)), "heads": {}, "controls": ctrl}
    for h in HEADS:
        if "dk_" + h in d:
            res["heads"][h] = calibrate_head(A, d["dk_" + h][keep], tab, kship)
    if "HT_s0" in res["heads"] and "dk_HT_s0_flip" in d:                       # C-FLIP: the y-flipped field must gain LESS than the unflipped one, at m*_0
        h0 = res["heads"]["HT_s0"]
        if h0["cal_null"]:
            ctrl["C-FLIP"] = {"pass": None, "why": "HT-s0 is CAL-NULL: no m*_0 to evaluate at (outcome CAL-NULL decides)"}
        else:
            kf = veto_pick(A, d["dk_HT_s0_flip"][keep], h0["m_star"], kship)
            gf = float(_delta(tab["pdms"], kf, kship).mean())
            gu = next(r["gain_pdms_x100"] for r in h0["curve"] if r["m"] == h0["m_star"])
            ctrl["C-FLIP"] = {"gain_unflipped_at_m0": gu, "gain_flipped_at_m0": gf, "pass": bool(gf < gu)}
    else:
        ctrl["C-FLIP"] = {"pass": None, "why": "dk_HT_s0_flip or HT_s0 not supplied"}
    iou = json.load(open(a.iou_json, encoding="utf-8")) if a.iou_json else {}
    res["iou_hc40"] = iou
    ctrl["C-SHUF"] = {"iou_HT_SHUF": iou.get("HT_SHUF"), "max": SHUF_IOU_MAX, "pass": (iou["HT_SHUF"] < SHUF_IOU_MAX) if "HT_SHUF" in iou else None}
    res["controls_failed"] = [k for k, v in ctrl.items() if v["pass"] is False]
    res["controls_pending"] = [k for k, v in ctrl.items() if v["pass"] is None]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1, default=_js)
    for h, r in res["heads"].items():
        print(f"{h}: m* {r['m_star']}  best gain {r['gain_at_best_m']:+.4f} at m {r['m_best_gain']}  CAL-NULL {r['cal_null']}")
    print("controls failed", res["controls_failed"], "pending", res["controls_pending"])
    print("ZZVDT_CAL_DONE")
    return 0


# ---------------------------------------------------------------------------------------------------------------- bars and outcomes (card 6, 7)
def eval_bars(c: dict) -> dict:
    """c = {"vt": {metric: ci}, "M": ci, "vtr": {metric: ci} | None, "shuf": {"pdms": ci} | None}; ci = {"point","lo","hi"}.  Every bar is the card's, as written."""
    vt = c["vt"]
    P1 = vt["pdms"]["lo"] > 0
    P2 = vt["pdms"]["point"] >= P2_BAR
    Mb = c["M"]["hi"] < 0
    adverse_vt = [k for k in JUDGED if vt[k]["hi"] < 0]
    G = not adverse_vt
    out = {"P1": bool(P1), "P2": bool(P2), "M": bool(Mb), "G": bool(G), "adverse_vt": adverse_vt, "vt_pdms_hi": vt["pdms"]["hi"]}
    vtr = c.get("vtr")
    if vtr is None:
        out.update({"R": False, "R_why": "VT-r not read (HT-s1 has no positive HC40 margin, or no replicate supplied): R fails; card silent, MM to confirm", "adverse_vtr": None})
    else:
        adverse_r = [k for k in JUDGED + ("pdms",) if vtr[k]["hi"] < 0]
        out.update({"R": bool(vtr["pdms"]["point"] > 0 and not adverse_r), "adverse_vtr": adverse_r})
    shuf = c.get("shuf")
    out["X_pass_P1"] = None if shuf is None else bool(shuf["pdms"]["lo"] > 0)
    out["X"] = None if shuf is None else (not out["X_pass_P1"])
    return out


def decide(b: dict, *, controls_failed=(), controls_pending=(), cal_null=False, ctrl_t=None, families=None) -> dict:
    """The card's outcomes in precedence order (7).  ctrl_t / families: True | False | None (pending).  Returns {outcome, next_arm, ...}."""
    pend = list(controls_pending) + (["CTRL-T"] if ctrl_t is None else []) + (["FAMILIES"] if families is None else [])

    def done(name, why, **kw):
        return {"outcome": name, "why": why, "next_arm": NEXT[name], "pending": pend, **kw}
    if controls_failed or ctrl_t is False or b.get("X_pass_P1"):
        why = ("failed controls " + ",".join(list(controls_failed) + (["CTRL-T"] if ctrl_t is False else []))) if (controls_failed or ctrl_t is False) else "X: VT-SHUF meets P1 (the rule gains without map information)"
        return done("INVALID", why)
    if cal_null:
        return done("CAL-NULL", "no HC40 margin with a positive point gain for HT-s0; navtest is not read")
    P1, P2, M, G, R = b["P1"], b["P2"], b["M"], b["G"], b["R"]
    if b["vt_pdms_hi"] < 0 or not G:
        return done("ADVERSE", "VT PDMS CI-hi < 0" if b["vt_pdms_hi"] < 0 else f"G fails: separated worse {b['adverse_vt']}")
    if not P1 and not M:
        return done("NULL", "P1 fails and M fails, no harm")
    if M and (not P1 or not P2) and G:
        return done("MECHANISM-SMALL", "M holds but P1 or P2 fails; G holds")
    if P1 and P2 and M and G and not R:
        return done("SINGLE-SEED", "separated, single head seed, not yet a lever claim")
    if P1 and P2 and M and G and R:
        if families is False:
            return done("Unnamed", "P1, P2, M, G, R hold but the families show a separated-worse component on VT")
        if b.get("X") is True and ctrl_t is True and families is True and not controls_pending:
            return done("ADOPT", "P1, P2, M, G, R, CTRL-T hold; X fails P1; four families present, none separated worse")
        return {"outcome": "ADOPT-CANDIDATE", "why": "P1, P2, M, G, R hold; not final until the pending items pass", "next_arm": NEXT["ADOPT"], "pending": pend}
    return done("Unnamed", "no committed outcome matches")


def check_families(fam) -> dict:
    """None -> pending.  Required: the four families for VT and VT-r; none separated worse on VT (VT-r's are reported)."""
    if fam is None:
        return {"ok": None}
    missing, worse, rep = [], [], {}
    for arm in ("VT", "VT-r"):
        a = fam.get(arm) or {}
        for f in FAMILIES:
            if f not in a or not a[f]:
                missing.append(f"{arm}.{f}")
                continue
            for comp, ci in a[f].items():
                if isinstance(ci, dict) and "lo" in ci and "hi" in ci:
                    wi = ci.get("worse_if", "negative")
                    sep = ci["hi"] < 0 if wi == "negative" else ci["lo"] > 0
                    if sep:
                        rep.setdefault(arm, []).append(f"{f}.{comp}")
                        if arm == "VT":
                            worse.append(f"{f}.{comp}")
    return {"ok": (not missing and not worse), "missing": missing, "separated_worse_VT": worse, "separated_worse_by_arm": rep}


# ---------------------------------------------------------------------------------------------------------------- the navtest read
def arm_summary(bt, tab, k, kship, label) -> dict:
    ar = np.arange(len(k))
    d = {m: bt.ci(_delta(tab[m], k, kship)) for m in M7}
    zero_c1, zero_k = ~(tab["dac"][ar, kship] > 0), ~(tab["dac"][ar, k] > 0)
    d["M_dac0_pick_share"] = bt.ci((zero_k.astype(float) - zero_c1.astype(float)) * 100)
    sw = k != kship
    ncz = ~(tab["nc"][ar, k] > 0)
    return {"label": label, "mean_pdms_x100": float(tab["pdms"][ar, k].mean() * 100), "delta_vs_C1": d,
            "n_changed": int(sw.sum()), "n_recoveries_c1_dac0": int((sw & zero_c1).sum()), "n_false_moves_c1_dac_pass": int((sw & ~zero_c1).sum()),
            "n_changed_pdms_better": int((sw & (tab["pdms"][ar, k] > tab["pdms"][ar, kship])).sum()), "n_changed_pdms_worse": int((sw & (tab["pdms"][ar, k] < tab["pdms"][ar, kship])).sum()),
            "nc_zero_pick_share": float(ncz.mean()), "ep_of_pick_mean_x100": float(tab["ep"][ar, k].mean() * 100)}


def cmd_read(a) -> int:
    free = commit_free_mb()
    if not a.no_mem_check and free < 6000:
        raise SystemExit(f"REFUSED: free commit {free:.0f} MB < 6000 MB before loading the navtest table")
    agg32, Boot, shas_i = load_pk(a.pk)
    cal = json.load(open(a.cal, encoding="utf-8"))
    ix, tlogs, tab_all, tshas, probs = load_table(a.table, check_sha=not a.no_sha)
    d = load_dump(a.dump)
    tok = d["token"].astype(str)
    miss = [t for t in tok if t not in ix]
    ti = np.array([ix[t] for t in tok if t in ix], int)
    tab = {m: tab_all[m][ti] for m in M7}
    logs = tlogs[ti]
    n = len(tok)
    ar = np.arange(n)
    S, kship = d["s_plan"], d["k_ship"].astype(int)
    A = agg32(S)
    ctrl = {}
    ctrl["CTRL-SHA"] = {"problems": probs, "table_sha256": tshas, "pass": (not probs) if not a.no_sha else None}
    ctrl["CTRL-SET"] = {"n_tokens": n, "in_table": n - len(miss), "n_logs": len(set(logs)), "pass": n == N_TOKENS and not miss and len(set(logs)) == N_LOGS}
    # CTRL-ID: veto off reproduces C1's pick and the mean; an all-drivable field changes 0 picks, Delta exactly 0.0
    off = A.argmax(1)
    c1_mean = float(tab["pdms"][ar, kship].mean() * 100)
    alld = {}
    for m in GRID:
        k = veto_pick(A, np.full(A.shape, 6.0), m, kship)
        alld[str(m)] = {"changed": int((k != kship).sum()), "delta_sum": float((tab["pdms"][ar, k] - tab["pdms"][ar, kship]).sum())}
    ctrl["CTRL-ID"] = {"veto_off_eq_k_ship": int((off == kship).sum()), "n": n, "c1_table_mean_x100": c1_mean, "all_drivable_by_m": alld,
                       "pass": bool((off == kship).all()) and abs(c1_mean - C1_PDMS) <= C1_PDMS_TOL and all(v["changed"] == 0 and v["delta_sum"] == 0.0 for v in alld.values())}
    # CTRL-PLAN: the dump's logits and picks equal the banked TRK dump bit-for-bit
    if a.trk_npz:
        z = np.load(a.trk_npz, allow_pickle=False)
        zt = z["token"].astype(str)
        pos = {t: i for i, t in enumerate(zt)}
        sel = np.array([pos[t] for t in tok if t in pos], int)
        same_tok = len(sel) == n
        s_eq = same_tok and np.array_equal(S.view(np.uint32), z["s_plan"][sel].astype(np.float32).view(np.uint32))
        k_eq = same_tok and np.array_equal(kship, z["k_ship"][sel].astype(int))
        ctrl["CTRL-PLAN"] = {"tokens_matched": int(len(sel)), "s_plan_bit_identical": bool(s_eq), "k_ship_identical": bool(k_eq), "pass": bool(s_eq and k_eq)}
    else:
        ctrl["CTRL-PLAN"] = {"pass": None, "why": "--trk-npz not supplied"}
    if a.g0_json:
        g = json.load(open(a.g0_json, encoding="utf-8"))
        ctrl["G0-TRK-C1"] = {**g, "pass": bool(g.get("n_match") == G0_N and g.get("n_total") == G0_N and g.get("second_dump_bit_identical") is True)}
    else:
        ctrl["G0-TRK-C1"] = {"pass": None, "why": "--g0-json not supplied"}
    if a.dump2:
        d2 = load_dump(a.dump2)
        t2 = d2["token"].astype(str)
        p1 = {t: i for i, t in enumerate(tok)}
        j = [(i, p1[t]) for i, t in enumerate(t2) if t in p1]
        eq = bool(j) and all(np.array_equal(d2[k][[x for x, _ in j]].view(np.uint8), d[k][[y for _, y in j]].view(np.uint8)) for k in d2 if k.startswith("dk_") or k in ("s_plan",))
        ctrl["CTRL-DET"] = {"n_second_pass_tokens": len(j), "bit_identical": eq, "pass": eq}
    else:
        ctrl["CTRL-DET"] = {"pass": None, "why": "--dump2 not supplied"}
    for k_ in ("C-SHUF", "C-FLIP"):
        ctrl[k_] = cal.get("controls", {}).get(k_, {"pass": None, "why": "not in the calibration json"})
    ctrl["CTRL-CAL"] = {"cal_controls_failed": cal.get("controls_failed", []), "pass": not cal.get("controls_failed")}
    failed = [k for k, v in ctrl.items() if v["pass"] is False]
    pending = [k for k, v in ctrl.items() if v["pass"] is None]
    res = {"card_sha256_prefix": CARD_SHA256_PREFIX, "status": "PROVISIONAL (CTRL-T and the four families pending until `confirm`)",
           "tier": "T1-class open loop, PDMS x 100 by table lookup on tr (harness-exact tracked sub-scores)",
           "estimator": "p5c_read.Boot paired log-cluster, 10,000 draws, seed 20260927, 95 % percentile, 136 logs; answers 'another draw of logs' only (head variance: VT-r)",
           "instruments_sha256": shas_i, "dump_sha256": sha256_file(a.dump), "cal_sha256": sha256_file(a.cal), "controls": ctrl, "controls_failed": failed, "controls_pending": pending,
           "K_ledger": {"reads_this_card": ["VT", "VT-r", "VT-SHUF", "VT-0"], "k_increment": 4, "O-DAC-T": "already reported (FILTER_ORACLE.json), adds none"}, "arms": {}}
    ch = cal["heads"]
    h0, h1 = ch.get("HT_s0", {}), ch.get("HT_s1", {})
    cal_null = bool(h0.get("cal_null", True))
    res["margins"] = {"m_star_0": h0.get("m_star"), "m_star_1": h1.get("m_star")}
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if cal_null or failed:
        b = {"vt_pdms_hi": float("nan"), "P1": False, "P2": False, "M": False, "G": False, "R": False, "X_pass_P1": None, "X": None, "adverse_vt": []}
        res["bars"] = b
        res["outcome"] = decide(b, controls_failed=failed, controls_pending=pending, cal_null=cal_null)
        res["note"] = "navtest table arms not read: " + ("failed controls" if failed else "CAL-NULL (no navtest read)")
        json.dump(res, open(out_dir / "READ_VDT.json", "w", encoding="utf-8"), indent=1, default=_js)
        print("ZZVDT_READ", res["outcome"]["outcome"])
        return 0
    bt = Boot(logs, n_boot=10000, seed=20260927)
    m0 = h0["m_star"]
    picks = {"VT": veto_pick(A, d["dk_HT_s0"], m0, kship), "VT-0": veto_pick(A, d["dk_HT_s0"], 0.0, kship), "VT-SHUF": veto_pick(A, d["dk_HT_SHUF"], m0, kship),
             "O-DAC-T": veto_pick(A, dk_from_tracked_dac(tab["dac"]), 0.0, kship)}
    if h1.get("m_star") is not None and "dk_HT_s1" in d:
        picks["VT-r"] = veto_pick(A, d["dk_HT_s1"], h1["m_star"], kship)
    for name, k in picks.items():
        res["arms"][name] = arm_summary(bt, tab, k, kship, name)
    orc = res["arms"]["O-DAC-T"]["delta_vs_C1"]["pdms"]["point"]
    for name in picks:
        res["arms"][name]["capture_ratio_vs_O-DAC-T"] = (res["arms"][name]["delta_vs_C1"]["pdms"]["point"] / orc) if orc else None
    sub = lambda nm: {m: res["arms"][nm]["delta_vs_C1"][m] for m in M7}      # noqa: E731
    b = eval_bars({"vt": sub("VT"), "M": res["arms"]["VT"]["delta_vs_C1"]["M_dac0_pick_share"], "vtr": sub("VT-r") if "VT-r" in picks else None, "shuf": sub("VT-SHUF")})
    res["bars"] = {**b, "P2_bar": P2_BAR}
    res["outcome"] = decide(b, controls_failed=failed, controls_pending=pending, cal_null=False)
    np.savez(out_dir / "vdt_picks.npz", token=tok, k_ship=kship, **{k.replace("-", "_"): v for k, v in picks.items()})
    json.dump(res, open(out_dir / "READ_VDT.json", "w", encoding="utf-8"), indent=1, default=_js)
    for name in ("VT", "VT-r", "VT-SHUF", "VT-0", "O-DAC-T"):
        if name in res["arms"]:
            for m in ("pdms", "nc", "dac", "ttc", "ep", "c"):
                c = res["arms"][name]["delta_vs_C1"][m]
                print(f"{name:8s} {m:4s} delta {c['point']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]" + ("  SEPARATED" if c["lo"] > 0 or c["hi"] < 0 else ""))
    print("bars", {k: v for k, v in res["bars"].items() if k in ("P1", "P2", "M", "G", "R", "X")}, "| controls failed", failed, "pending", pending)
    print("ZZVDT_READ", res["outcome"]["outcome"])
    return 0


# ---------------------------------------------------------------------------------------------------------------- confirm (CTRL-T + families)
_CSV_COLS = {"score": "pdms", "no_at_fault_collisions": "nc", "drivable_area_compliance": "dac", "driving_direction_compliance": "ddc",
             "ego_progress": "ep", "time_to_collision_within_bound": "ttc", "comfortable": "c"}


def ctrl_t_from_csv(csv_path, tok, k, ix, tab_all) -> dict:
    """The harness CSV (NAVSIM per-token scores) must equal the table at the picks to 1e-6 on EVERY token (score; and every sub-score column the CSV carries)."""
    import csv as _csv
    rows = {r["token"]: r for r in _csv.DictReader(open(csv_path, encoding="utf-8")) if r.get("token") not in (None, "average")}
    mx, miss, cols = 0.0, 0, set()
    for t, kk in zip(tok, k):
        r = rows.get(t)
        if r is None:
            miss += 1
            continue
        for cn, m in _CSV_COLS.items():
            if cn in r and r[cn] not in ("", None):
                cols.add(cn)
                mx = max(mx, abs(float(r[cn]) - float(tab_all[m][ix[t], int(kk)])))
    return {"max_abs": mx, "missing_tokens": miss, "columns_compared": sorted(cols), "pass": bool(miss == 0 and "score" in cols and mx <= T_TOL)}


def cmd_confirm(a) -> int:
    rep = json.load(open(a.read, encoding="utf-8"))
    if not (Path(a.read).parent / "vdt_picks.npz").exists():       # CAL-NULL / failed-control reads wrote no picks: nothing to confirm
        Path(a.out_dir).mkdir(parents=True, exist_ok=True)
        json.dump({**rep, "status": "FINAL-NO-NAVTEST-READ" if rep["outcome"]["outcome"] == "CAL-NULL" else "PROVISIONAL"}, open(Path(a.out_dir) / "VERDICT_VDT.json", "w", encoding="utf-8"), indent=1, default=_js)
        print("ZZVDT_VERDICT", rep["outcome"]["outcome"], "(no picks to confirm)")
        return 0
    pk = np.load(Path(a.read).parent / "vdt_picks.npz", allow_pickle=False)
    ix, _, tab_all, _, _ = load_table(a.table, check_sha=False)
    tok = [str(t) for t in pk["token"]]
    t = {"VT": ctrl_t_from_csv(a.csv_vt, tok, pk["VT"], ix, tab_all)}
    if a.csv_vtr and "VT_r" in pk.files:
        t["VT-r"] = ctrl_t_from_csv(a.csv_vtr, tok, pk["VT_r"], ix, tab_all)
    ctrl = dict(rep["controls"])
    ctrl["CTRL-T"] = {"pass": all(v["pass"] for v in t.values()), **t}
    fam = check_families(json.load(open(a.families, encoding="utf-8")) if a.families else None)
    failed = [k for k, v in ctrl.items() if v["pass"] is False]
    pending = [k for k, v in ctrl.items() if v["pass"] is None]
    b = rep["bars"]
    cal_null = rep["outcome"]["outcome"] == "CAL-NULL"
    o = decide(b, controls_failed=failed, controls_pending=pending, cal_null=cal_null, ctrl_t=ctrl["CTRL-T"]["pass"], families=fam["ok"])
    final = not o["pending"]
    v = {**rep, "controls": ctrl, "families": fam, "outcome": o, "status": "FINAL" if final else "PROVISIONAL (pending: %s)" % ", ".join(o["pending"])}
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    json.dump(v, open(Path(a.out_dir) / "VERDICT_VDT.json", "w", encoding="utf-8"), indent=1, default=_js)
    print("ZZVDT_VERDICT", o["outcome"], v["status"])
    return 0


def _js(x):
    if hasattr(x, "item"):
        return x.item()
    return str(x)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("calibrate")
    c.add_argument("--pk"); c.add_argument("--dump", required=True); c.add_argument("--lab-glob", required=True); c.add_argument("--out", required=True)
    c.add_argument("--hc40-list"); c.add_argument("--iou-json"); c.add_argument("--navtest-logs")
    r = sp.add_parser("read")
    r.add_argument("--pk"); r.add_argument("--dump", required=True); r.add_argument("--table", required=True); r.add_argument("--cal", required=True)
    r.add_argument("--trk-npz"); r.add_argument("--out-dir", required=True); r.add_argument("--g0-json"); r.add_argument("--dump2")
    r.add_argument("--no-sha", action="store_true"); r.add_argument("--no-mem-check", action="store_true")
    f = sp.add_parser("confirm")
    f.add_argument("--read", required=True); f.add_argument("--table", required=True); f.add_argument("--csv-vt", required=True); f.add_argument("--csv-vtr")
    f.add_argument("--families"); f.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    return {"calibrate": cmd_calibrate, "read": cmd_read, "confirm": cmd_confirm}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
