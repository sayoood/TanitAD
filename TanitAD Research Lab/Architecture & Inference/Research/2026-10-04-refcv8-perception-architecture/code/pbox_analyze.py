"""WP-D P-BOX analysis (CPU): every registered metric, bar and must-fail of PREREG_WPD_PROBES.md + PREREG_WPD_A1.md,
from the arms' detection packs (`pbox_arms.py` -> <arm>/eval_packs.pkl, <arm>/train_diag_packs.pkl).

Metrics (EVAL-DIAG, class-agnostic unless stated): AP@{0.5, 1, 2, 4} m raw and after centre-distance NMS r 2 m (WP-C's
F4b, fixed in the prereg); AP@2 m per band (0-20 / 20-40 / 40-60 m); the per-arm TRAIN P = R gate (fitted on the arm's
TRAIN-DIAG packs, never on EVAL) and, at that gate, boxes per detected object (PROBE-0 Q2 definition) and conf_ratio;
presence ECE (15 bins, all scored slots, TP at 2 m); TP centre error longitudinal / lateral by band (PROBE-0 Q5);
per-class AP@2 m (automobile, person).
Estimator: paired episode-cluster bootstrap over the EVAL episodes, B = 1,000, percentile 95 % ("another draw of
EPISODES"); the training-run question is answered by the replicate floor |PB0r - PB0| (H-ESTIM-SEED-1).
Run (dev box): PYTHONPATH=<tip>/stack python pbox_analyze.py --arms-dir <dir with one sub-dir per arm>
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import box_probe0 as P0  # noqa: E402

PKG = Path(__file__).resolve().parents[1]
THRS = [0.5, 1.0, 2.0, 4.0]
BANDS = ["0_20", "20_40", "40_60"]
CLS = {"automobile": 0, "person": 5}
LEVERS = ["PB1", "PB2", "PB3", "PB4", "PB4h", "PB6", "PB7", "PB8"]
DEDUP_ARMS = {"PB1", "PB2", "PB3"}
LOC_ARMS = {"PB3", "PB4", "PB7", "PB8"}
B_BOOT = 1000


def check_trained(d: Path, name: str, allow_no_record=()):
    """Control C-step at the ANALYSIS end: an arm's record must prove its decoder changed. A copy of the VOID no-step
    PB0 (disclosure 8) sat in the dev-box pull dir under the name PB0 and was read once as the control -- it carries no
    C_step field and refuses here."""
    rp = d / name / "record.json"
    if not rp.exists():
        if name in allow_no_record:
            return None
        raise SystemExit(f"{name}: no record.json -- not a banked arm (pass --allow-no-record {name} for a descriptive read)")
    r = json.loads(rp.read_text(encoding="utf-8"))
    c = r.get("C_step_rel_param_change")
    if not (isinstance(c, (int, float)) and c > 0):
        raise SystemExit(f"{name}: record has no positive C_step_rel_param_change ({c!r}) -- an untrained arm, refused")
    return c


def load(d: Path, name: str):
    with open(d / name / "eval_packs.pkl", "rb") as fh:
        ev = pickle.load(fh)
    with open(d / name / "train_diag_packs.pkl", "rb") as fh:
        td = pickle.load(fh)
    return ev, td


def pr_gate(packs):
    rows = sorted(((c, k) for pk in packs for (c, k, *_r) in P0.greedy(pk, 2.0) if k >= 0), key=lambda r: -r[0])
    n_pos = int(sum(int(pk["pos"].sum()) for pk in packs))
    h = np.asarray([k for _, k in rows], float)
    c = np.asarray([c for c, _ in rows], float)
    tp = np.cumsum(h)
    prec = tp / np.arange(1, len(h) + 1)
    rec = tp / max(n_pos, 1)
    d = prec - rec
    # the P = R crossing among rows with at least one TP: if the top-scored row is an FP, prec = rec = 0 there and an
    # unguarded `d <= 0` returns the TOP score as the gate (it did for PB4: gate 0.413 = the max, conf_ratio 0.0006)
    ok = (d <= 0) & (tp > 0)
    k = int(np.argmax(ok)) if ok.any() else len(d) - 1
    return float(c[k])


def per_episode_counts(packs, gate, ep_of_win, n_ep):
    """tp (detected objects), dup, n_conf (slots >= gate not matched to IGNORE), n_pos -- per episode."""
    out = {k: np.zeros(n_ep) for k in ("tp", "dup", "n_conf", "n_pos")}
    for wi, pk in enumerate(packs):
        e = ep_of_win[wi]
        rows = P0.greedy(pk, 2.0, keep=P0.sig(pk["logit"]) >= gate)
        det_g = {g for (c, k, b, s, g) in rows if k == 1}
        posi = np.nonzero(pk["pos"])[0]
        gxy = pk["gt_xy"].astype(np.float64)
        for (c, k, b, s, g) in rows:
            if k >= 0:
                out["n_conf"][e] += 1
            if k != 0 or not posi.size:
                continue
            d = np.sqrt(((gxy[posi] - pk["xy"][s].astype(np.float64)) ** 2).sum(-1))
            j = int(np.argmin(d))
            if d[j] <= 2.0 and int(posi[j]) in det_g:
                out["dup"][e] += 1
        out["tp"][e] += len(det_g)
        out["n_pos"][e] += int(pk["pos"].sum())
    return out


def ece_all(packs):
    p, y = [], []
    for pk in packs:
        for (c, k, b, s, g) in P0.greedy(pk, 2.0):
            if k >= 0:
                p.append(c)
                y.append(k)
    p, y = np.asarray(p), np.asarray(y, float)
    edges = np.linspace(0, 1, 16)
    bi = np.clip(np.digitize(p, edges[1:-1]), 0, 14)
    return float(sum((bi == i).sum() / p.size * abs(y[bi == i].mean() - p[bi == i].mean())
                     for i in range(15) if (bi == i).any()))


def loc_split(packs, gate):
    acc = {b: {"dlon": [], "dlat": []} for b in BANDS}
    for pk in packs:
        for (c, k, b, s, g) in P0.greedy(pk, 4.0, keep=P0.sig(pk["logit"]) >= gate):
            if k != 1 or b is None:
                continue
            d = pk["xy"][s].astype(np.float64) - pk["gt_xy"][g].astype(np.float64)
            acc[b]["dlon"].append(abs(d[0]))
            acc[b]["dlat"].append(abs(d[1]))
    return {b: {"n": len(v["dlon"]), "med_dlon": float(np.median(v["dlon"])) if v["dlon"] else None,
                "med_dlat": float(np.median(v["dlat"])) if v["dlat"] else None} for b, v in acc.items()}


def tables(packs, ep_of_win):
    """APTable per (kind, thr, band) for kind in raw / nms2, plus class tables at 2 m."""
    from tanitad.eval import detection_metrics as det
    T = {}
    for kind in ("raw", "nms2"):
        for thr in THRS:
            rows = [P0.greedy(pk, thr, keep=(P0.nms_keep(pk, 2.0) if kind == "nms2" else None)) for pk in packs]
            for band in [None] + BANDS:
                T[(kind, thr, band)] = P0.APTable(rows, [P0.npos_band(pk, band) for pk in packs], ep_of_win, band)
    for cname, ci in CLS.items():
        rows = [[(c, k, b, s, g) for (c, k, b, s, g) in det.greedy_rows(pk, 2.0, ci)] for pk in packs]
        npos = [int((pk["pos"] & (pk["gt_cls"] == ci)).sum()) for pk in packs]
        T[("cls_" + cname, 2.0, None)] = P0.APTable(rows, npos, ep_of_win)
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms-dir", required=True)
    ap.add_argument("--boot", type=int, default=B_BOOT)
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow-no-record", default="", help="comma list of DESCRIPTIVE pack dirs without a record")
    ap.add_argument("--ref2", default="", help="a second, DESCRIPTIVE reference (e.g. STEP0 = refcv7's decoder)")
    a = ap.parse_args()
    d = Path(a.arms_dir)
    arms = sorted(p.name for p in d.iterdir() if (p / "eval_packs.pkl").exists())
    if "PB0" not in arms:
        raise SystemExit("PB0 (the control) is not banked")
    allow = tuple(x for x in a.allow_no_record.split(",") if x)
    c_step = {n: check_trained(d, n, allow) for n in arms}
    ev0, _ = load(d, "PB0")
    eps = sorted({pk["sha12"] for pk in ev0})
    eix = {e: i for i, e in enumerate(eps)}
    ep_of_win = [eix[pk["sha12"]] for pk in ev0]
    W = P0.boot_weights(len(eps), a.boot)
    res, tabs, cnts = {}, {}, {}
    for arm in arms:
        ev, td = load(d, arm)
        assert [pk["sha12"] for pk in ev] == [pk["sha12"] for pk in ev0], f"{arm}: window order differs"
        T = tables(ev, ep_of_win)
        gate = pr_gate(td)
        cn = per_episode_counts(ev, gate, ep_of_win, len(eps))
        r = {"gate_train_diag": gate,
             "boxes_per_detected_object": float((cn["tp"].sum() + cn["dup"].sum()) / max(cn["tp"].sum(), 1)),
             "conf_ratio": float(cn["n_conf"].sum() / max(cn["n_pos"].sum(), 1)),
             "ece_all": ece_all(ev), "loc": loc_split(ev, gate)}
        for (kind, thr, band), t in T.items():
            r[f"ap{thr:g}_{kind}_{band or 'all'}"] = t.ap()
        res[arm], tabs[arm], cnts[arm] = r, T, cn
        print(arm, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "loc"}, flush=True)

    def bpo_w(cn, w):
        tp = (cn["tp"] * w).sum()
        return float((tp + (cn["dup"] * w).sum()) / max(tp, 1e-9))

    keys = ["ap2_nms2_all", "ap1_raw_all", "ap2_raw_all"] + [f"ap2_nms2_{b}" for b in BANDS]

    def paired_against(ref):
        out = {}
        for arm in arms:
            if arm == ref:
                continue
            pr = {}
            for kk in keys:
                head, kind, bnd = kk.split("_", 2)          # band names themselves contain "_" (0_20)
                thr = float(head[2:])
                band = None if bnd == "all" else bnd
                t1, t0 = tabs[arm][(kind, thr, band)], tabs[ref][(kind, thr, band)]
                dd = [t1.ap(w) - t0.ap(w) for w in W]
                pr[kk] = {"point": t1.ap() - t0.ap(), "ci95": P0.ci(dd)}
            dd = [bpo_w(cnts[arm], w) - bpo_w(cnts[ref], w) for w in W]
            pr["boxes_per_object"] = {"point": res[arm]["boxes_per_detected_object"] -
                                      res[ref]["boxes_per_detected_object"], "ci95": P0.ci(dd)}
            out[arm] = pr
        return out

    paired = paired_against("PB0")
    # descriptive, NOT a registered comparison: every arm against refcv7's decoder unchanged (the overfit caveat, §7.3)
    paired_ref2 = paired_against(a.ref2) if a.ref2 and a.ref2 in arms else None
    floor = ({k: abs(v["point"]) for k, v in paired["PB0r"].items()} if "PB0r" in paired else None)

    def sep(arm, k, sign=+1):
        v = paired[arm][k]
        lo, hi = v["ci95"]
        ci_ok = (lo > 0) if sign > 0 else (hi < 0)
        fl = floor[k] if floor else None
        return ci_ok and (fl is None or abs(v["point"]) > fl), fl

    verdict = {}
    for arm in [x for x in LEVERS if x in paired]:
        v = {}
        b1a = paired[arm]["ap2_nms2_all"]["point"] >= 0.02 and sep(arm, "ap2_nms2_all")[0]
        b1b = (arm in DEDUP_ARMS and paired[arm]["boxes_per_object"]["point"] <= -0.30
               and sep(arm, "boxes_per_object", -1)[0] and paired[arm]["ap2_nms2_all"]["point"] >= -0.005)
        v["bar1"] = bool(b1a or b1b)
        v["bar1_detail"] = {"ap2_nms_route": bool(b1a), "dedup_route": bool(b1b)}
        v["bar2_no_band_drop_gt_0.02"] = all(paired[arm][f"ap2_nms2_{b}"]["point"] >= -0.02 for b in BANDS)
        if arm in LOC_ARMS:
            v["bar3_ap1_raw"] = bool(paired[arm]["ap1_raw_all"]["point"] >= 0.02 and sep(arm, "ap1_raw_all")[0])
        v["PASS"] = bool(v["bar1"] and v["bar2_no_band_drop_gt_0.02"] and v.get("bar3_ap1_raw", True))
        verdict[arm] = v
    if "PB6" in res:
        r6, r0 = res["PB6"], res["PB0"]
        verdict["PB6_dramatic"] = {
            "ap2_raw_ge_0.339": r6["ap2_raw_all"] >= 0.339,
            "ap2_nms_ge_PB0_plus_0.08": r6["ap2_nms2_all"] >= r0["ap2_nms2_all"] + 0.08,
            "ap1_raw_ge_2x_PB0": r6["ap1_raw_all"] >= 2 * r0["ap1_raw_all"],
            "bpo_le_1.3": r6["boxes_per_detected_object"] <= 1.3}
        verdict["PB6_dramatic"]["PASS"] = all(verdict["PB6_dramatic"].values())
    mf = {}
    if "R-pres0z" in res:
        mf["R-pres0z (A1, governs)"] = {"ap2_raw": res["R-pres0z"]["ap2_raw_all"],
                                        "must_be_le": 0.5 * res["PB0"]["ap2_raw_all"],
                                        "failed_as_required": res["R-pres0z"]["ap2_raw_all"]
                                        <= 0.5 * res["PB0"]["ap2_raw_all"]}
    if "R-pres0" in res:
        ok = res["R-pres0"]["ap2_raw_all"] <= 0.5 * res["PB0"]["ap2_raw_all"]
        mf["R-pres0 (registered)"] = {"ap2_raw": res["R-pres0"]["ap2_raw_all"],
                                      "fell_ge_50pct": ok,
                                      "label": None if ok else "ill-posed under warm start (A1 governs)"}
    if "R-memshuf" in res:
        mf["R-memshuf"] = {"ap2_raw": res["R-memshuf"]["ap2_raw_all"],
                           "failed_as_required": res["R-memshuf"]["ap2_raw_all"] <= 0.05}
    if "R-qshuf" in paired:
        hi = paired["R-qshuf"]["ap1_raw_all"]["ci95"][1]
        mf["R-qshuf"] = {"d_ap1_raw_vs_PB0": paired["R-qshuf"]["ap1_raw_all"], "failed_as_required": hi <= 0.01}
    void = [k for k, v in mf.items() if k != "R-pres0 (registered)" and not v.get("failed_as_required", True)]
    out = {"estimator": "paired episode-cluster bootstrap, B=%d, 95 %% percentile; + replicate floor |PB0r-PB0|"
           % a.boot, "arms": res, "paired_vs_PB0": paired, "replicate_floor": floor, "verdict": verdict,
           "paired_vs_ref2_DESCRIPTIVE": ({"ref": a.ref2, "pairs": paired_ref2} if paired_ref2 else None),
           "must_fail": mf, "VOID": bool(void), "void_reasons": void, "C_step_rel_param_change": c_step,
           "n_eval_windows": len(ev0), "n_eval_episodes": len(eps)}
    p = Path(a.out) if a.out else PKG / "raw" / "pbox_result.json"
    p.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print("wrote", p, "VOID" if void else "", flush=True)


if __name__ == "__main__":
    main()
