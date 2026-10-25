"""p2_tol_read.py -- P2 analysis: boundary-tolerant map read, refcv8 EMA-48,852 vs refcv7@50,400, PAIRED on identical S2 windows.

CPU only.  Everything is computed from the two ``p2_<arm>_tol.npz`` (per-window x class x band counts), checked against the BANKED perc8
tables, bootstrapped over episodes with the ``taniteval.ci`` draw stream, and the PRE-REGISTERED verdict (PREREG_P2.md section 2) is applied
mechanically.  Writes ``<out>/P2_TOL.json`` and ``<out>/RESULT.md``.

    python p2_tol_read.py --a-dir <out_a> --a-arm refcv8_ema48852 --b-dir <out_b> --b-arm refcv7_50400 --out <dir> [--n-boot 1000]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p2_tol_lib as T  # noqa: E402
from taniteval import ci  # noqa: E402
from taniteval import map_hires_metrics as MH  # noqa: E402

KS0 = (0,) + T.KS                       # 0, 1, 2, 5, 10
BAND_NAMES = ("0-20m", "20-40m", "40-60m", "60-80m", "80-100m")
CNAME = T.CLASS_NAMES
VERDICT_CLASSES = (2, 3)                # lane, crosswalk
VERDICT_BANDS = (0, 1)
THIN = (2, 3, 4, 5, 6)
STATS = T.stat_names()
BANK = {  # the banked tables C5 / C5b read
    "refcv8": "C:/Users/Admin/refcv8_pf/perc8/raw/A_nosync/perc_map_table.npz",
    "refcv7": "E:/refcv7_eval_kit/perc/step50400_a6text/perc_map_table.npz",
}
MAP_READ_JSON = "C:/Users/Admin/refcv8_pf/perc8/raw/map_read.json"
DIAG_ACC = "E:/refcv7_diag_bin/2026-10-04/eval_final.acc.pt"
MIN_GT_CELLS, MIN_GT_EPISODES = 100_000, 20


# ------------------------------------------------------------------------------------------------------------------ #
def load_tol(dirp, arm):
    z = np.load(Path(dirp) / f"p2_{arm}_tol.npz", allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["eid"] = [str(e) for e in d["eid"]]
    return d


def metrics_from_tot(tot, ks=KS0):
    res = {}
    for k in ks:
        tpp = tot["inter"] if k == 0 else tot[f"tpp{k}"]
        tpg = tot["inter"] if k == 0 else tot[f"tpg{k}"]
        P, R, F, I = T.prf_from_counts(tot["pred"], tot["gt"], tpp, tpg)
        res[k] = {"P": P, "R": R, "F": F, "IoU": I}
    return res


class Booter:
    """Vectorised episode-cluster bootstrap that consumes EXACTLY the ``ci._draws`` stream (seed, n_boot), per-clip sums."""

    def __init__(self, eid, n_boot, seed=0):
        self.uniq, self.idx_by_ep = ci.episode_index(eid)
        e = np.asarray(eid)
        self.clip_of = np.searchsorted(self.uniq, e)
        self.n_in = np.bincount(self.clip_of, minlength=len(self.uniq))
        mult = np.zeros((n_boot, len(self.uniq)), np.float64)
        for i, sel in enumerate(ci._draws(self.uniq, self.idx_by_ep, n_boot, seed)):
            cnt = np.bincount(self.clip_of[sel], minlength=len(self.uniq))
            m, rem = np.divmod(cnt, self.n_in)
            if rem.any():
                raise SystemExit("[p2] a draw that is not whole clips")
            mult[i] = m
        self.mult = mult
        self.n_boot = n_boot

    def clip_sums(self, arr):                              # [N, ...] -> [n_clip, ...]
        out = np.zeros((len(self.uniq),) + arr.shape[1:], np.float64)
        np.add.at(out, self.clip_of, arr.astype(np.float64))
        return out

    def draws(self, cs):                                   # [n_clip, ...] -> [B, ...]
        return np.tensordot(self.mult, cs, axes=(1, 0))


def ci_of(draws, alpha=0.05):
    lo = np.nanpercentile(draws, 100 * alpha / 2, axis=0)
    hi = np.nanpercentile(draws, 100 * (1 - alpha / 2), axis=0)
    return lo, hi


def fnum(x, dp=3):
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{dp}f}"


def cell_str(m, lo, hi, dp=3):
    return f"{fnum(m, dp)} [{fnum(lo, dp)}, {fnum(hi, dp)}]"


def verdict_of(n_gt, n_ep, R5, P5, I5, I0, min_cells=MIN_GT_CELLS, min_eps=MIN_GT_EPISODES):
    if n_gt < min_cells or n_ep < min_eps:
        return "UNDERPOWERED"
    if R5 < 0.20:
        return "ABSENT"
    if R5 >= 0.50 and P5 >= 0.50 and I5 >= 2.0 * I0:
        return "MISPLACED"
    return "MIXED"


def fragile_of(v, R5ci, P5ci, RATci):
    """Annotation only (PREREG section 2): can the 95 % interval of a deciding quantity flip the verdict?"""
    if v == "UNDERPOWERED":
        return False
    if v == "ABSENT":
        return bool(R5ci[1] >= 0.20)
    if v == "MISPLACED":
        return bool(R5ci[0] < 0.50 or P5ci[0] < 0.50 or RATci[0] < 2.0)
    return bool(R5ci[0] < 0.20 or (R5ci[1] >= 0.50 and P5ci[1] >= 0.50 and RATci[1] >= 2.0))


def kstar(Rk_by_k):
    for k in KS0:
        r = Rk_by_k[k]
        if r is not None and np.isfinite(r) and r >= 0.5:
            return str(k)
    return ">10"


# ------------------------------------------------------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-dir", required=True)
    ap.add_argument("--a-arm", default="refcv8_ema48852")
    ap.add_argument("--b-dir", required=True)
    ap.add_argument("--b-arm", default="refcv7_50400")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--prereg", default="C:/Users/Admin/refcv8_pf/perc8tol/PREREG_P2.md")
    ap.add_argument("--skip-c6", action="store_true", help="SMOKE ONLY")
    ap.add_argument("--min-gt-cells", type=int, default=MIN_GT_CELLS, help="PRE-REGISTERED 100000; lowered ONLY by the synthetic plumbing test")
    ap.add_argument("--min-gt-episodes", type=int, default=MIN_GT_EPISODES, help="PRE-REGISTERED 20; lowered ONLY by the synthetic plumbing test")
    ap.add_argument("--no-bank", action="store_true", help="SMOKE ONLY: skip C5/C5b (banked tables cover all windows)")
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    A, B = load_tol(a.a_dir, a.a_arm), load_tol(a.b_dir, a.b_arm)
    recA = json.load(open(Path(a.a_dir) / f"p2_{a.a_arm}.json", encoding="utf-8"))
    recB = json.load(open(Path(a.b_dir) / f"p2_{a.b_arm}.json", encoding="utf-8"))
    N = len(A["eid"])
    ctl = {}

    # ---------------- C7 pairing identity ------------------------------------------------------------------------ #
    ctl["C7_pairing"] = {
        "eid_equal": A["eid"] == B["eid"], "ws_equal": bool(np.array_equal(A["ws"], B["ws"])),
        "gt_equal_elementwise": bool(np.array_equal(A["gt"], B["gt"])),
        "census_equal": bool(all(np.array_equal(A[f"census_{k}"], B[f"census_{k}"]) for k in ("thick", "lat", "lon"))),
        "n_windows": N}
    ctl["C7_pairing"]["pass"] = bool(all(v for k, v in ctl["C7_pairing"].items() if k != "n_windows"))

    # ---------------- C8 identities ------------------------------------------------------------------------------- #
    ev_md5 = {}
    p2dir = Path("C:/Users/Admin/refcv8_pf/perc8tol/raw")
    try:
        mine = dict(l.split()[::-1] for l in open(p2dir / "md5_p2_evalcache.txt", encoding="utf-8") if l.strip())
        mine = {k.lstrip("*"): v for k, v in mine.items()}
        banked = dict(l.split()[::-1] for l in open("C:/Users/Admin/refcv8_pf/perc8/raw/md5_devbox_evalcache.txt", encoding="utf-8") if l.strip())
        banked = {k.lstrip("*"): v for k, v in banked.items()}
        ev_md5 = {"n_mine": len(mine), "n_banked": len(banked), "n_equal": sum(1 for k in banked if mine.get(k) == banked[k])}
    except Exception as e:                                   # noqa: BLE001
        ev_md5 = {"error": repr(e)}
    ctl["C8_identities"] = {
        "refcv8_ckpt_md5": recA["ckpt_md5"], "refcv8_expected": "b4364029b68f29a33eeb4aebbbbd002b",
        "refcv7_ckpt_md5": recB["ckpt_md5"], "refcv7_expected": "b418d0fc4a92a6848c246a6a7c50207b",
        "windows_md5_a": recA["windows_json_md5"], "windows_md5_b": recB["windows_json_md5"],
        "windows_expected": "a3a78dba889e70aa69a05271559794ad", "eval_cache_md5": ev_md5}
    c8 = ctl["C8_identities"]
    c8["pass"] = bool(c8["refcv8_ckpt_md5"] == c8["refcv8_expected"] and c8["refcv7_ckpt_md5"] == c8["refcv7_expected"]
                      and c8["windows_md5_a"] == c8["windows_expected"] == c8["windows_md5_b"]
                      and ev_md5.get("n_banked") == 139 and ev_md5.get("n_equal") == 139)
    ctl["C9_law_frame"] = {"refcv8": recA["controls"]["law_frame"]["pass"], "refcv7": recB["controls"]["law_frame"]["pass"]}
    ctl["C9_law_frame"]["pass"] = bool(ctl["C9_law_frame"]["refcv8"] and ctl["C9_law_frame"]["refcv7"])
    ctl["C4_cheb_k1_vs_EDT"] = {"refcv8_mismatches": recA["controls"]["C4_cheb_k1_vs_EDT_0p2m"]["mismatching_cells"],
                                "refcv7_mismatches": recB["controls"]["C4_cheb_k1_vs_EDT_0p2m"]["mismatching_cells"],
                                "windows": recA["controls"]["C4_cheb_k1_vs_EDT_0p2m"]["windows"],
                                "counts_vs_MH_mismatching_arrays": [recA["controls"]["C0_counts_vs_MH_window_stats"]["mismatching_arrays"],
                                                                    recB["controls"]["C0_counts_vs_MH_window_stats"]["mismatching_arrays"]]}
    ctl["C4_cheb_k1_vs_EDT"]["pass"] = bool(ctl["C4_cheb_k1_vs_EDT"]["refcv8_mismatches"] == 0 and ctl["C4_cheb_k1_vs_EDT"]["refcv7_mismatches"] == 0
                                            and sum(ctl["C4_cheb_k1_vs_EDT"]["counts_vs_MH_mismatching_arrays"]) == 0)
    ctl["C1_C2_real_gt"] = {"C1_gt_as_pred": [recA["controls"]["C1p_gt_as_pred"], recB["controls"]["C1p_gt_as_pred"]],
                            "C2_all_one_class": [recA["controls"]["C2p_all_one_class"], recB["controls"]["C2p_all_one_class"]]}
    ctl["C1_C2_real_gt"]["pass"] = bool(all(x["all_pass"] for x in ctl["C1_C2_real_gt"]["C1_gt_as_pred"] + ctl["C1_C2_real_gt"]["C2_all_one_class"]))

    # ---------------- C5 / C5b vs the banked tables ---------------------------------------------------------------- #
    if not a.no_bank:
        mr = json.load(open(MAP_READ_JSON, encoding="utf-8"))
        c5 = {}
        for tag, tol, bankkey, armname, rec in (("refcv8", A, "refcv8", "refcv8_ema48852", recA), ("refcv7", B, "refcv7", "refcv7_50400", recB)):
            Tb = MH.WindowTable.load(BANK[bankkey])
            bk = Tb.stats["refcv7"]                               # perc_pass keeps the r7 arm key for both
            n_gt_b = np.stack(bk["n_gt"])
            inter_b = np.stack(bk["inter"])
            union_b = np.stack(bk["union"])
            entry = {"banked_table": BANK[bankkey], "n_windows_banked": len(Tb.eid), "n_windows_mine": N,
                     "eid_equal": [str(e) for e in Tb.eid] == A["eid"] if len(Tb.eid) == N else False}
            if len(Tb.eid) == N:
                entry["n_gt_exactly_equal"] = bool(np.array_equal(n_gt_b, tol["gt"]))
                iou_b = inter_b.sum(0) / np.maximum(union_b.sum(0), 1)
                iou_m = tol["inter"].sum(0) / np.maximum(tol["union"].sum(0), 1)
                m = tol["gt"].sum(0) >= MIN_GT_CELLS
                entry["max_abs_iou_diff_cells_with_gt"] = float(np.abs(iou_b - iou_m)[m].max())
                entry["n_cells_compared"] = int(m.sum())
                entry["max_abs_count_diff"] = {k: int(np.abs(np.stack(bk[kb]).sum(0) - tol[ka].sum(0)).max())
                                               for k, (kb, ka) in {"inter": ("inter", "inter"), "n_pred": ("n_pred", "pred"), "union": ("union", "union")}.items()}
                entry["max_rel_count_diff_inter"] = float((np.abs(inter_b.sum(0) - tol["inter"].sum(0)) / np.maximum(inter_b.sum(0), 1)).max())
                entry["exact"] = bool(entry["max_abs_iou_diff_cells_with_gt"] <= 1e-9)
                entry["pass"] = bool(entry["eid_equal"] and entry["n_gt_exactly_equal"] and entry["max_abs_iou_diff_cells_with_gt"] <= 2e-3)
            else:
                entry["pass"] = False
            # C5b: my pooling on the BANKED table == MH.pooled and perc8's map_read.json
            tot = {"pred": np.stack(bk["n_pred"]).sum(0), "gt": n_gt_b.sum(0), "inter": inter_b.sum(0)}
            Pb, Rb, Fb, Ib = T.prf_from_counts(tot["pred"], tot["gt"], tot["inter"], tot["inter"])
            pooled_iou = MH.pooled({k: np.stack(v) for k, v in bk.items()})["iou"]
            mskb = n_gt_b.sum(0) > 0
            d_pool = float(np.abs(Ib - pooled_iou)[mskb].max())
            arms_json = mr["summary"]["arms"][armname]
            bad_json, n_json = 0, 0
            for c in range(8):
                for b in range(5):
                    key = f"{MH.CLASS_NAMES[c]}|{BAND_NAMES[b]}|iou"
                    if key in arms_json and arms_json[key].get("mean") is not None and mskb[c, b]:
                        n_json += 1
                        bad_json += int(round(float(Ib[c, b]), 4) != round(float(arms_json[key]["mean"]), 4))
            ctl.setdefault("C5b_bank_rescore", {})[tag] = {"max_abs_diff_vs_MH_pooled": d_pool, "map_read_json_cells": n_json,
                                                           "map_read_json_mismatches_at_4dp": bad_json,
                                                           "pass": bool(d_pool <= 1e-9 and bad_json == 0)}
            c5[tag] = entry
        ctl["C5_forward_reproduction_k0"] = c5
        ctl["C5b_bank_rescore"]["pass"] = bool(all(v["pass"] for v in ctl["C5b_bank_rescore"].values()))

    # ---------------- pooled point estimates + bootstrap ---------------------------------------------------------- #
    boot = Booter(A["eid"], a.n_boot, seed=0)
    n_cl = len(boot.uniq)
    pt, dr, cs = {}, {}, {}
    for tag, D in (("a", A), ("b", B)):
        cs[tag] = {s: boot.clip_sums(D[s]) for s in STATS}
        tot = {s: D[s].sum(0).astype(np.float64) for s in STATS}
        pt[tag] = metrics_from_tot(tot)
        dtot = {s: boot.draws(cs[tag][s]) for s in STATS}       # [B, 8, 5]
        dr[tag] = metrics_from_tot(dtot)
    gt_eps = {tag: (cs[tag]["gt"] > 0).sum(0) for tag in ("a", "b")}   # [8,5] episodes with GT
    bi = lambda m, k, x: m[k][x]                                        # noqa: E731

    res = {"cells": {}, "n_boot": a.n_boot, "n_clips": n_cl, "n_windows": N}
    for ci_c in range(8):
        for b in range(5):
            key = f"{CNAME[ci_c]}|{BAND_NAMES[b]}"
            cell = {"n_gt_a": int(A["gt"].sum(0)[ci_c, b]), "n_gt_b": int(B["gt"].sum(0)[ci_c, b]),
                    "n_pred_a": int(A["pred"].sum(0)[ci_c, b]), "n_pred_b": int(B["pred"].sum(0)[ci_c, b]),
                    "n_episodes_with_gt": int(gt_eps["a"][ci_c, b]), "k": {}}
            for k in KS0:
                rec = {}
                for met in ("P", "R", "F", "IoU"):
                    ent = {}
                    for tag in ("a", "b"):
                        d = dr[tag][k][met][:, ci_c, b]
                        lo, hi = ci_of(d)
                        ent[tag] = {"mean": float(pt[tag][k][met][ci_c, b]), "lo": float(lo), "hi": float(hi),
                                    "n_nan_draws": int(np.isnan(d).sum())}
                    dd = dr["a"][k][met][:, ci_c, b] - dr["b"][k][met][:, ci_c, b]
                    lo, hi = ci_of(dd)
                    ent["delta"] = {"mean": float(pt["a"][k][met][ci_c, b] - pt["b"][k][met][ci_c, b]), "lo": float(lo), "hi": float(hi),
                                    "separated": bool(lo > 0 or hi < 0), "n_nan_draws": int(np.isnan(dd).sum())}
                    rec[met] = ent
                cell["k"][str(k)] = rec
            # ratio IoU5/IoU0 (per arm) for the fragility annotation
            for tag in ("a", "b"):
                with np.errstate(divide="ignore", invalid="ignore"):
                    rat = dr[tag][5]["IoU"][:, ci_c, b] / dr[tag][0]["IoU"][:, ci_c, b]
                    pr = pt[tag][5]["IoU"][ci_c, b] / pt[tag][0]["IoU"][ci_c, b] if pt[tag][0]["IoU"][ci_c, b] > 0 else float("inf")
                rat = np.where(np.isfinite(rat), rat, np.nan)
                lo, hi = (ci_of(rat) if np.isfinite(rat).any() else (np.nan, np.nan))
                cell[f"iou5_over_iou0_{tag}"] = {"mean": float(pr), "lo": float(lo), "hi": float(hi)}
            cell["k_star_a"] = kstar({k: float(pt["a"][k]["R"][ci_c, b]) for k in KS0})
            cell["k_star_b"] = kstar({k: float(pt["b"][k]["R"][ci_c, b]) for k in KS0})
            res["cells"][key] = cell

    # ---------------- verdicts (PREREG section 2) -------------------------------------------------------------------- #
    verd = {}
    for c in VERDICT_CLASSES:
        for b in VERDICT_BANDS:
            key = f"{CNAME[c]}|{BAND_NAMES[b]}"
            cell = res["cells"][key]
            row = {}
            for tag, nm in (("a", a.a_arm), ("b", a.b_arm)):
                g = cell[f"n_gt_{tag}"]
                R5, P5 = cell["k"]["5"]["R"][tag], cell["k"]["5"]["P"][tag]
                I5, I0 = cell["k"]["5"]["IoU"][tag]["mean"], cell["k"]["0"]["IoU"][tag]["mean"]
                v = verdict_of(g, cell["n_episodes_with_gt"], R5["mean"], P5["mean"], I5, I0, a.min_gt_cells, a.min_gt_episodes)
                rat = cell[f"iou5_over_iou0_{tag}"]
                row[nm] = {"verdict": v, "n_gt": g, "R5": R5, "P5": P5, "IoU0": I0, "IoU5": I5, "IoU5_over_IoU0": rat,
                           "k_star": cell[f"k_star_{tag}"],
                           "ci_fragile": fragile_of(v, (R5["lo"], R5["hi"]), (P5["lo"], P5["hi"]), (rat["lo"], rat["hi"]))}
            verd[key] = row
    res["verdicts"] = verd

    # ---------------- C6 bootstrap-stream control --------------------------------------------------------------------- #
    if not a.skip_c6:
        W = N
        cat = {s: np.concatenate([A[s], B[s]]).astype(np.int64) for s in STATS}
        a_idx = np.arange(W, dtype=np.float64)

        def reducer(metric, k, c, b):
            def red(rows):
                r = np.asarray(rows, dtype=np.int64)
                tot = {s: cat[s][r].sum(axis=0).astype(np.float64) for s in STATS}
                return float(metrics_from_tot(tot, ks=(k,))[k][metric][c, b])
            return red
        cells6 = [("IoU", 5, 2, 0), ("R", 5, 2, 1), ("P", 5, 3, 0), ("IoU", 2, 4, 0), ("IoU", 0, 1, 0), ("F", 10, 5, 0), ("IoU", 1, 2, 0)]
        c6, ok6 = [], True
        for metric, k, c, b in cells6:
            r = ci.paired_episode_cluster_bootstrap(a_idx, a_idx + W, A["eid"], n_boot=a.n_boot, seed=0, reduce=reducer(metric, k, c, b))
            dp = int(r.get("display_dp", ci.DISPLAY_DP))
            ent = res["cells"][f"{CNAME[c]}|{BAND_NAMES[b]}"]["k"][str(k)][metric]["delta"]
            mine = {"delta": round(ent["mean"], dp), "lo": round(ent["lo"], dp), "hi": round(ent["hi"], dp)}
            theirs = {x: r[x] for x in ("delta", "lo", "hi")}
            def _eq(u, v):
                u_nan = u is None or (isinstance(u, float) and math.isnan(u))
                v_nan = v is None or (isinstance(v, float) and math.isnan(v))
                return (u_nan and v_nan) or (not u_nan and not v_nan and u == v)
            same = all(_eq(mine[x], theirs[x]) for x in mine)
            ok6 &= bool(same)
            c6.append({"cell": f"{CNAME[c]}|{BAND_NAMES[b]}|{metric}|k={k}", "mine": mine, "ci_paired_episode_cluster_bootstrap": theirs, "equal": bool(same)})
        ctl["C6_bootstrap_stream"] = {"cells": c6, "pass": bool(ok6)}

    # ---------------- plausibility column (INHERITED): refcv7 diag-bin, other windows ------------------------------- #
    try:
        import torch
        acc = torch.load(DIAG_ACC, map_location="cpu", weights_only=False)["acc"]["pc"]
        dtot = {s: acc[s].sum(0).numpy().astype(np.float64) for s in ("pred", "gt", "inter", "tpp5", "tpg5", "tpp1", "tpg1", "tpp2", "tpg2", "tpp10", "tpg10")}
        diag = {}
        for k in KS0:
            tpp = dtot["inter"] if k == 0 else dtot[f"tpp{k}"]
            tpg = dtot["inter"] if k == 0 else dtot[f"tpg{k}"]
            P, R, F, I = T.prf_from_counts(dtot["pred"], dtot["gt"], tpp, tpg)
            diag[str(k)] = {"P": P.tolist(), "R": R.tolist(), "IoU": I.tolist()}
        res["refcv7_diagbin_plausibility"] = {"source": DIAG_ACC, "evidence": "INHERITED (eval_diag window set, K=8 per episode, ~1,112 windows; NOT the S2 windows; ckpt identity not re-verified)", "k": diag}
    except Exception as e:                                       # noqa: BLE001
        res["refcv7_diagbin_plausibility"] = {"error": repr(e)}

    # ---------------- GT census -------------------------------------------------------------------------------------- #
    census = {}
    for i, c in enumerate(T.CENSUS_CLASSES):
        for b in range(5):
            census[f"{CNAME[c]}|{BAND_NAMES[b]}"] = {kind: T.thickness_summary(A[f"census_{kind}"][i, b]) for kind in ("thick", "lat", "lon")}
    res["census"] = census
    res["controls"] = ctl
    res["inputs"] = {"a": recA, "b": recB, "prereg_sha256": hashlib.sha256(open(a.prereg, "rb").read()).hexdigest() if Path(a.prereg).exists() else None}
    res["wall_s"] = round(time.time() - t0, 1)
    (out / "P2_TOL.json").write_text(MH.to_json(res), encoding="utf-8")
    print(f"[p2-read] wrote {out / 'P2_TOL.json'} in {res['wall_s']} s")
    write_md(res, out / "RESULT.md", a)


# ------------------------------------------------------------------------------------------------------------------ #
def write_md(res, path, a):
    ctl, verd, cells = res["controls"], res["verdicts"], res["cells"]
    A_, B_ = a.a_arm, a.b_arm
    L = []
    w = L.append
    inp = res["inputs"]
    w(f"# P2 -- boundary-tolerant read of the 10 cm map: {A_} vs {B_} (PAIRED, identical S2 windows)\n")
    w("*EvalFlyWheel. Generated by `code/p2_tol_read.py` from the artifacts listed in the manifest. The verdicts are the PRE-REGISTERED mechanical rule "
      f"(`PREREG_P2.md`, sha256 `{inp['prereg_sha256']}`) applied to the pooled point estimates; nothing was tuned after the data.*\n")
    w("| | |\n|---|---|")
    w("| **Evidence class** | MEASURED (ours); every number traces to `P2_TOL.json` and the per-arm `p2_*_tol.npz` |")
    w("| **Tier** | **T0** -- perception output of the forward on the S2 windows; NOT driving performance |")
    w(f"| **Estimator** | episode-cluster bootstrap over {res['n_clips']} map-labelled episodes, the `taniteval.ci` draw stream, seed 0, {res['n_boot']} draws, 95 % percentile; PAIRED = same draws for both arms |")
    w(f"| **n** | {res['n_windows']} map-labelled windows / {res['n_clips']} episodes, identical for both arms (eid + ws sequences compared, C7) |")
    w("| **What the CI answers** | another draw of EPISODES only. ONE training seed per arm (`H-ESTIM-SEED-1`): a separated interval is necessary, not sufficient; no lever claim from any paired difference |")
    w(f"| **Arms** | {A_}: ckpt md5 `{inp['a']['ckpt_md5']}` (step {inp['a']['step']}); {B_}: ckpt md5 `{inp['b']['ckpt_md5']}` (step {inp['b']['step']}); decision rule `{inp['a']['decision_rule']}` (frozen class weights), unshifted pose clock; dev-box GPU, batch 1 |")
    w("| **Tolerance** | Chebyshev distance in 0.1 m cells, k in {0, 1, 2, 5, 10} = {0, 0.1, 0.2, 0.5, 1.0 m}; IoU_k = F_k / (2 - F_k) (k=0 is the exact IoU) |\n")

    w("## 1. Controls\n")
    w("| id | control | read | verdict |\n|---|---|---|---|")
    def pf(x): return "PASS" if x else "**FAIL**"
    w(f"| C1/C2 (synthetic) | GT-as-prediction = 1.0 at every k; all-one-class = 0; analytic strips; diagonal literals; taxicab mutant CAUGHT; vs torch max-pool dilate and the EDT instrument | `logs/test_p2_tol.log` (23 checks) | see log |")
    c12 = ctl["C1_C2_real_gt"]
    w(f"| C1/C2 (real GT) | first {c12['C1_gt_as_pred'][0]['windows']} map windows per arm | GT-as-pred all pass: {[x['all_pass'] for x in c12['C1_gt_as_pred']]}; all-one-class all pass: {[x['all_pass'] for x in c12['C2_all_one_class']]} | {pf(c12['pass'])} |")
    c4 = ctl["C4_cheb_k1_vs_EDT"]
    w(f"| C4 | chessboard k=1 vs the programme's Euclidean EDT (<0.2 m) tp counts, classes 2,3,5, every window; and inter/union/n_gt/n_pred vs `MH.window_stats` | {c4['windows']} windows: mismatching cells refcv8 {c4['refcv8_mismatches']}, refcv7 {c4['refcv7_mismatches']}; count-array mismatches {c4['counts_vs_MH_mismatching_arrays']} | {pf(c4['pass'])} |")
    if "C5_forward_reproduction_k0" in ctl:
        for tag, e in ctl["C5_forward_reproduction_k0"].items():
            if "max_abs_iou_diff_cells_with_gt" in e:
                w(f"| C5 {tag} | forward reproduction at k=0 vs the banked table (`{e['banked_table']}`) | eid equal {e['eid_equal']}; n_gt exactly equal {e['n_gt_exactly_equal']}; max abs IoU diff over {e['n_cells_compared']} cells with GT **{e['max_abs_iou_diff_cells_with_gt']:.2e}** ({'EXACT <= 1e-9' if e['exact'] else 'within the pre-registered 2e-3' if e['pass'] else 'ABOVE 2e-3'}); max abs count diff {e['max_abs_count_diff']}; max rel inter diff {e['max_rel_count_diff_inter']:.2e} | {pf(e['pass'])} |")
            else:
                w(f"| C5 {tag} | window count differs from the banked table | {e} | **FAIL** |")
        for tag, e in ctl["C5b_bank_rescore"].items():
            if tag == "pass":
                continue
            w(f"| C5b {tag} | this analysis' pooling applied to the BANKED table vs `MH.pooled` and `map_read.json` | max abs diff {e['max_abs_diff_vs_MH_pooled']:.1e}; {e['map_read_json_cells']} json cells, {e['map_read_json_mismatches_at_4dp']} mismatches at 4 dp | {pf(e['pass'])} |")
    if "C6_bootstrap_stream" in ctl:
        c6 = ctl["C6_bootstrap_stream"]
        w(f"| C6 | vectorised bootstrap draw stream vs `ci.paired_episode_cluster_bootstrap` | {sum(x['equal'] for x in c6['cells'])}/{len(c6['cells'])} cells equal (delta, lo, hi) | {pf(c6['pass'])} |")
    c7 = ctl["C7_pairing"]
    w(f"| C7 | pairing identity | eid equal {c7['eid_equal']}, ws equal {c7['ws_equal']}, n_gt equal {c7['gt_equal_elementwise']}, GT census equal {c7['census_equal']} | {pf(c7['pass'])} |")
    c8 = ctl["C8_identities"]
    w(f"| C8 | ckpt md5 / S2 md5 / eval-cache | refcv8 `{c8['refcv8_ckpt_md5']}`, refcv7 `{c8['refcv7_ckpt_md5']}`, S2 `{c8['windows_md5_a']}`, eval-cache md5 {c8['eval_cache_md5']} | {pf(c8['pass'])} |")
    c9 = ctl["C9_law_frame"]
    w(f"| C9 | LAW-frame control | refcv8 {c9['refcv8']}, refcv7 {c9['refcv7']} | {pf(c9['pass'])} |")
    allpass = all(v.get("pass", True) for k, v in ctl.items() if isinstance(v, dict) and "pass" in v) and all(v["pass"] for v in ctl.get("C5_forward_reproduction_k0", {}).values())
    w(f"\n**All pre-registered controls: {'PASS' if allpass else 'NOT ALL PASS -- every verdict below is UNVERIFIED'}.**\n")

    w("## 2. Verdict (pre-registered rule, pooled point estimates)\n")
    w("Rule: UNDERPOWERED if n_gt < 100,000 cells or GT in < 20 episodes; ABSENT if R_5 < 0.20; MISPLACED if R_5 >= 0.50 and P_5 >= 0.50 and IoU_5 >= 2 x IoU_0; else MIXED. `fragile` = a 95 % bootstrap bound of a deciding quantity can flip the verdict (annotation only).\n")
    w(f"| class | band | arm | n_gt cells | IoU_0 (exact) | P_5 | R_5 | IoU_5 | IoU_5/IoU_0 | k* (R_k >= 0.5) | **verdict** | fragile |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key, row in verd.items():
        cn, bn = key.split("|")
        for nm, r in row.items():
            w(f"| {cn} | {bn} | {nm} | {r['n_gt']:,} | {r['IoU0']:.3f} | {cell_str(r['P5']['mean'], r['P5']['lo'], r['P5']['hi'])} | {cell_str(r['R5']['mean'], r['R5']['lo'], r['R5']['hi'])} | {r['IoU5']:.3f} | {cell_str(r['IoU5_over_IoU0']['mean'], r['IoU5_over_IoU0']['lo'], r['IoU5_over_IoU0']['hi'], 2)} | {r['k_star']} | **{r['verdict']}** | {'yes' if r['ci_fragile'] else 'no'} |")
    w("")

    w("## 3. Tolerance curves, all bands (P / R / IoU at k = 0, 0.1, 0.2, 0.5, 1.0 m)\n")
    w(f"Cells = pooled value [95 % bootstrap interval] for {A_}; {B_} = point estimate; last column = paired {A_} - {B_} IoU_k with interval and SEP (interval excludes 0; necessary not sufficient). n_gt = GT cells of the class in that band.\n")
    for c in (2, 3, 4, 5, 6):
        w(f"### {CNAME[c]}\n")
        w(f"| band | n_gt | k (m) | P_k {A_} | R_k {A_} | IoU_k {A_} | P_k {B_} | R_k {B_} | IoU_k {B_} | paired dIoU_k |\n|---|---|---|---|---|---|---|---|---|---|")
        for b in range(5):
            cell = cells[f"{CNAME[c]}|{BAND_NAMES[b]}"]
            for k in KS0:
                kk = cell["k"][str(k)]
                d = kk["IoU"]["delta"]
                w(f"| {BAND_NAMES[b] if k == 0 else ''} | {cell['n_gt_a']:,} | {k / 10:g} | {cell_str(kk['P']['a']['mean'], kk['P']['a']['lo'], kk['P']['a']['hi'])} | {cell_str(kk['R']['a']['mean'], kk['R']['a']['lo'], kk['R']['a']['hi'])} | {cell_str(kk['IoU']['a']['mean'], kk['IoU']['a']['lo'], kk['IoU']['a']['hi'])} | {fnum(kk['P']['b']['mean'])} | {fnum(kk['R']['b']['mean'])} | {fnum(kk['IoU']['b']['mean'])} | {cell_str(d['mean'], d['lo'], d['hi'])} {'SEP' if d['separated'] else 'ns'} |")
        w("")
    w("### Context classes (area classes), k = 0 and k = 0.5 m\n")
    w(f"| class | band | IoU_0 {A_} | IoU_5 {A_} | IoU_0 {B_} | IoU_5 {B_} |\n|---|---|---|---|---|---|")
    for c in (0, 1, 7):
        for b in range(5):
            cell = cells[f"{CNAME[c]}|{BAND_NAMES[b]}"]
            g = lambda tag, k: cell["k"][str(k)]["IoU"][tag]["mean"]    # noqa: E731
            w(f"| {CNAME[c]} | {BAND_NAMES[b]} | {fnum(g('a', 0))} | {fnum(g('a', 5))} | {fnum(g('b', 0))} | {fnum(g('b', 5))} |")
    w("")

    w("## 4. GT line-width census (S2 windows, GT only -- identical for both arms)\n")
    w("Thickness of every GT cell = min(maximal lateral run, maximal longitudinal run through it), in 0.1 m cells. `share<=w` = share of GT cells with thickness <= w cells.\n")
    w("| class | band | n cells | median (cells = m) | p10 / p90 (cells) | mean | share<=1 | share<=2 | share<=3 | share<=5 | capped (>=64) |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for c in T.CENSUS_CLASSES:
        for b in range(5):
            s = res["census"][f"{CNAME[c]}|{BAND_NAMES[b]}"]["thick"]
            if not s.get("n_cells"):
                continue
            w(f"| {CNAME[c]} | {BAND_NAMES[b]} | {s['n_cells']:,} | {s['median_cells']} = {s['median_m']:.1f} | {s['p10_cells']} / {s['p90_cells']} | {s['mean_cells']:.2f} | {s['share_le_1']:.3f} | {s['share_le_2']:.3f} | {s['share_le_3']:.3f} | {s['share_le_5']:.3f} | {s['capped_share']:.3f} |")
    w("\nAnalytic exact IoU of two parallel strips of width w cells offset by d cells: IoU_0 = (w - d)+ / (w + d):\n")
    w("| w (cells) | d=0 | d=1 | d=2 | d=3 | d=5 | d=10 |\n|---|---|---|---|---|---|---|")
    for wd in (1, 2, 3, 5, 10):
        w(f"| {wd} | " + " | ".join(f"{T.analytic_iou0_strip(wd, d):.3f}" for d in (0, 1, 2, 3, 5, 10)) + " |")
    w("")
    pl = res.get("refcv7_diagbin_plausibility", {})
    if "k" in pl:
        w("## 5. Plausibility column (INHERITED, other windows): refcv7 from `refcv7_diag_bin/eval_final.acc.pt`, decision `pc`\n")
        w(f"{pl['evidence']}. Same 139 eval episodes, ~1,112 windows (K=8 per episode), NOT the S2 windows -- so it is a sanity read, not a pair.\n")
        w("| class | band | IoU_0 diag | IoU_0 S2 | R_5 diag | R_5 S2 | P_5 diag | P_5 S2 |\n|---|---|---|---|---|---|---|---|")
        for c in (2, 3, 4):
            for b in range(3):
                cell = cells[f"{CNAME[c]}|{BAND_NAMES[b]}"]
                w(f"| {CNAME[c]} | {BAND_NAMES[b]} | {fnum(pl['k']['0']['IoU'][c][b])} | {fnum(cell['k']['0']['IoU']['b']['mean'])} | {fnum(pl['k']['5']['R'][c][b])} | {fnum(cell['k']['5']['R']['b']['mean'])} | {fnum(pl['k']['5']['P'][c][b])} | {fnum(cell['k']['5']['P']['b']['mean'])} |")
        w("")
    readme = Path(path).parent / "READING.md"
    if readme.exists():
        w(readme.read_text(encoding="utf-8"))
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[p2-read] wrote {path}")


if __name__ == "__main__":
    main()
