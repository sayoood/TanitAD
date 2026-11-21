#!/usr/bin/env python3
"""ap_probe_38k.py -- does refcv6's map head SEE the thin classes and lose the argmax, or never learn them?

The discriminating experiment of `TanitAD Research Lab/Architecture & Inference/Research/
2026-09-26-refcv7-map-hires/RESULT.md` §3: a per-class RANKING (average precision of the softmax)
beside the DECISION (argmax IoU), on the windows the boxes video rendered (3 clips x 171), at 0.5 m and
at 10 cm.

Input: the per-cell dump of `render_refcv6_map_video.py --boxes --dump-map` (step 38,000, one model
load): softmax(map_logits) float16 [9, 120, 64], the GT fractions as the label file's uint8, map_seen,
map_valid, and each window's raw GT frame. The 10 cm truth is the label file's own `fine_codes`
[T, 600, 320] (codes 0-7, 255 = not seen), read here by sha12 from the eval kit, identity-checked.

(a) 0.5 m, seen cells (not-seen fraction < 0.5 == map_seen), per class c in 0-7:
    AP(p_c ; cart_frac[c] >= 0.5), AP(p_c ; cart_frac[c] > 0), the base rates, mean p_c on GT-positive
    vs GT-negative cells, argmax IoU (argmax softmax vs argmax cart_frac).
(b) 10 cm: the 0.5 m softmax nearest-upsampled x5 to 600 x 320; AP(p_c ; fine_codes == c) on fine
    cells with code != 255, split by x band 0-20 / 20-40 / 40-60 m (fine rows 0-199 / 200-399 /
    400-599 == coarse rows 0-39 / 40-79 / 80-119). Computed EXACTLY from per-coarse-cell counts:
    the 25 fine cells of one coarse cell share one score, so they are one tie group either way.
AP = sum_k (R_k - R_{k-1}) P_k over DISTINCT score thresholds (sklearn's definition): a constant score
therefore reads AP == base rate exactly.

CONTROLS (must read known values): the constant predictor (each class's frequency) -> AP == base rate
to 3 dp; a shuffled-GT control (the GT of a window from ANOTHER clip) -> AP ~ base rate; the dump's
fractions must equal the label file's `cart_frac` at the dumped raw frame (alignment, exactly 0 cells
differ); `cart_frac` must equal the 5 x 5 block fraction of `fine_codes` (exactly 0 cells differ).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

STAMP = ("refcv6-r101-s0 step 38,000; hybrid: F3 + true label clock from 34,500; trunk C26 "
         "equalisation OFF (declared 43, dropped)")
CLASSES = ("seen, no map class", "drivable", "lane / road line", "crosswalk", "arrow / text",
           "non-drivable edge", "hatched area", "sidewalk / verge")
GT_DIR = Path("D:/refcv6_eval_kit/data/sam3_gt_eval_thor137")
BANDS = (("0-20m", 0, 40), ("20-40m", 40, 80), ("40-60m", 80, 120))    # coarse rows


def ap_weighted(score, npos, nneg):
    """AP over items with (score, n_pos, n_neg); ties at one score form ONE threshold."""
    score = np.asarray(score, np.float64).ravel()
    npos = np.asarray(npos, np.float64).ravel()
    nneg = np.asarray(nneg, np.float64).ravel()
    tot = npos.sum()
    if tot <= 0 or score.size == 0:
        return None
    order = np.argsort(-score, kind="mergesort")
    s = score[order]
    tp = np.cumsum(npos[order])
    fp = np.cumsum(nneg[order])
    last = np.r_[np.nonzero(np.diff(s))[0], s.size - 1]
    tp, fp = tp[last], fp[last]
    rec = tp / tot
    prec = tp / np.maximum(tp + fp, 1e-12)
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def load_dump(dump: Path):
    meta = [json.loads(ln) for ln in open(dump / "meta.jsonl", encoding="utf-8") if ln.strip()]
    P, F, S, V = [], [], [], []
    for f in sorted(glob.glob(str(dump / "chunk_*.npz"))):
        z = np.load(f)
        P.append(z["probs"])
        F.append(z["frac"])
        S.append(z["seen"])
        V.append(z["valid"])
    return meta, np.concatenate(P), np.concatenate(F), np.concatenate(S), np.concatenate(V)


def partner_index(clips: list) -> np.ndarray:
    """For window i, a window of ANOTHER clip: the k-th window of the next clip (cyclic)."""
    order = sorted(set(clips))
    by = {c: [i for i, x in enumerate(clips) if x == c] for c in order}
    out = np.empty(len(clips), dtype=np.int64)
    for ci, c in enumerate(order):
        nxt = by[order[(ci + 1) % len(order)]]
        for k, i in enumerate(by[c]):
            out[i] = nxt[k % len(nxt)]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="boxes_clip", help="the dump set to probe (the rendered windows)")
    a = ap.parse_args(argv)
    t0 = time.time()
    meta, P, Fu, S, V = load_dump(Path(a.dump))
    keep = [i for i, m in enumerate(meta) if a.set in m["set"]]
    if not keep:
        raise SystemExit(f"no dumped window carries set {a.set!r}")
    # ---- LOCATION-PRIOR control, from EVERY dumped window, leave-own-clip-out ------------------ #
    # A per-cell class frequency knows WHERE a class usually is and nothing about THIS scene. It is
    # the floor a scene-blind head would reach, and the honest comparator for the shuffled-GT
    # control, which keeps the location prior (drivable is ahead in every window).
    all_clips = np.asarray([m["clip_sha12"] for m in meta])
    scored_clips = sorted({all_clips[i] for i in keep})

    def _sums(sel):
        """Per-cell sums over the windows ``sel`` (bool [N]): counts for the 0.5 m labels and the
        fine-cell mass for 10 cm (cart_frac u8 == 255 x the 5x5 block fraction, so the class's fine
        positives are frac_c/255 x 25 and the seen fine cells are (255 - frac_notseen)/255 x 25)."""
        s_ = S[sel]
        out = {"seen": s_.sum(0, dtype=np.int64),
               "fine_seen": (255 - Fu[sel, 8].astype(np.int64)).sum(0)}
        for c in range(8):
            f_ = Fu[sel, c]
            out[("thr", c)] = ((f_ >= 128) & s_).sum(0, dtype=np.int64)
            out[("any", c)] = ((f_ > 0) & s_).sum(0, dtype=np.int64)
            out[("fine", c)] = f_.astype(np.int64).sum(0)
        return out
    tot_s = _sums(np.ones(len(all_clips), bool))
    prior_by_clip = {}
    for c12 in scored_clips:
        own = _sums(all_clips == c12)
        s_out = np.maximum(tot_s["seen"] - own["seen"], 1)
        fs_out = np.maximum(tot_s["fine_seen"] - own["fine_seen"], 1)
        prior_by_clip[c12] = {c: {"thr": (tot_s[("thr", c)] - own[("thr", c)]) / s_out,
                                  "any": (tot_s[("any", c)] - own[("any", c)]) / s_out,
                                  "fine": (tot_s[("fine", c)] - own[("fine", c)]) / fs_out}
                              for c in range(8)}
    n_prior_windows = {c12: int((all_clips != c12).sum()) for c12 in scored_clips}
    n_prior_clips = len(set(all_clips.tolist())) - 1
    meta = [meta[i] for i in keep]
    P, Fu, S, V = P[keep].astype(np.float32), Fu[keep], S[keep], V[keep]
    n_w = len(meta)
    PRI = {c: {k: np.stack([prior_by_clip[m["clip_sha12"]][c][k] for m in meta]) for k in ("thr", "any", "fine")}
           for c in range(8)}
    rng = np.random.default_rng(0)
    clips = [m["clip_sha12"] for m in meta]
    # ---- the 10 cm truth, read by sha12, identity + alignment checked ------------------------- #
    fine = np.empty((n_w, 600, 320), np.uint8)
    align_bad = block_bad = 0
    for s12 in sorted(set(clips)):
        z = np.load(GT_DIR / f"{s12}.sam3mapgt.npz", allow_pickle=False)
        stored = (json.loads(str(z["meta_json"])).get("source") or {}).get("clip_sha12")
        if stored != s12:
            raise SystemExit(f"{s12}: GT file stores sha12 {stored!r}")
        fc, cf = z["fine_codes"], z["cart_frac"]
        if fc.shape[1:] != (600, 320):
            raise SystemExit(f"{s12}: fine_codes shape {fc.shape}")
        for i in [i for i, c in enumerate(clips) if c == s12]:
            rf = meta[i].get("map_raw_frame")
            if rf is None:
                raise SystemExit("the dump carries no map_raw_frame -- re-dump with the current tool")
            fine[i] = fc[rf]
            align_bad += int((cf[rf] != Fu[i]).any(axis=0).sum())          # dump vs file cart_frac
            # 5x5 block fraction of the fine codes must equal cart_frac -- the census's own rule
            # (map_res_census.py:46-49: classes 0-7, |fine block fraction - cart| <= 1.5/255)
            f5 = fc[rf].reshape(120, 5, 64, 5)
            for c in range(8):
                frac_fine = (f5 == c).sum(axis=(1, 3)) / 25.0
                block_bad += int((np.abs(frac_fine - cf[rf][c].astype(np.float64) / 255.0)
                                  > 1.5 / 255.0).sum())
    seen = S
    n_seen = int(seen.sum())
    # the spec's "seen" is "not-seen fraction < 0.5"; the dump's mask is the loader's map_seen --
    # assert they are the same cells (u8 < 128 <=> frac < 0.5, since cart_frac is k/25 quantised)
    seen_def_bad = int((S != (Fu[:, 8] < 128)).sum())
    n_seen_not_valid = int((S & ~V).sum())
    part = partner_index(clips)
    res = {"stamp": STAMP, "what": __doc__.split("\n")[0], "n_windows": n_w,
           "n_clips": len(set(clips)), "clips_sha12": sorted(set(clips)),
           "n_seen_cells_0p5m": n_seen, "n_cells_0p5m": int(S.size),
           "n_seen_cells_outside_map_valid": n_seen_not_valid,
           "location_prior": {"what": "per-cell class frequency from every OTHER clip's dumped windows "
                                      "(leave-own-clip-out); knows where a class usually is, nothing "
                                      "about this scene",
                              "n_windows_by_scored_clip": n_prior_windows,
                              "n_other_clips": n_prior_clips},
           "seen_definition_check": {"map_seen_vs_notseen_frac_lt_0.5_cells_differing": seen_def_bad,
                                     "must_read": 0},
           "alignment_control": {"dump_frac_vs_file_cart_frac_cells_differing": align_bad,
                                 "cart_frac_vs_5x5_fine_block_cells_differing": block_bad,
                                 "must_read": 0},
           "ap_definition": "sum_k (R_k - R_{k-1}) P_k over distinct score thresholds (sklearn)",
           "half_m": {}, "ten_cm": {}}
    arg_p = P.argmax(1)
    arg_g = Fu.argmax(1)
    for c, name in enumerate(CLASSES):
        pc = P[:, c]
        f = Fu[:, c]
        pos_thr, pos_any = (f >= 128) & seen, (f > 0) & seen
        s_ = pc[seen]
        yt, ya = pos_thr[seen], pos_any[seen]
        base_t, base_a = float(yt.mean()), float(ya.mean())
        # argmax IoU on seen cells, pooled
        pa, ga = (arg_p == c) & seen, (arg_g == c) & seen
        inter, union = int((pa & ga).sum()), int((pa | ga).sum())
        # controls: constant (frequency) score; GT from another clip's window
        const_t = ap_weighted(np.full(s_.shape, base_t), yt, ~yt)
        seen_p = seen[part]
        s_sh = pc[seen_p]                                 # this window's scores on the partner's cells
        yt_sh = ((Fu[part, c] >= 128) & seen_p)[seen_p]
        ya_sh = ((Fu[part, c] > 0) & seen_p)[seen_p]
        lp_t, lp_a = PRI[c]["thr"][seen], PRI[c]["any"][seen]        # location prior, own clip out
        perm = rng.permutation(s_.size)                             # cell-permuted model scores
        res["half_m"][name] = {
            "n_windows": n_w, "n_cells": int(s_.size),
            "location_prior_ap_frac_ge_0.5": ap_weighted(lp_t, yt, ~yt),
            "location_prior_ap_presence": ap_weighted(lp_a, ya, ~ya),
            "cell_permuted_ap_frac_ge_0.5": ap_weighted(s_[perm], yt, ~yt),
            "ap_frac_ge_0.5": ap_weighted(s_, yt, ~yt), "base_rate_frac_ge_0.5": base_t,
            "n_pos_frac_ge_0.5": int(yt.sum()),
            "ap_presence_frac_gt_0": ap_weighted(s_, ya, ~ya), "base_rate_presence": base_a,
            "n_pos_presence": int(ya.sum()),
            "mean_p_on_pos_frac_ge_0.5": float(s_[yt].mean()) if yt.any() else None,
            "mean_p_on_neg_frac_lt_0.5": float(s_[~yt].mean()) if (~yt).any() else None,
            "argmax_iou": inter / union if union else None, "argmax_union": union,
            "controls": {"constant_ap_frac_ge_0.5": const_t,
                         "constant_equals_base_3dp": (None if const_t is None
                                                      else round(const_t, 3) == round(base_t, 3)),
                         "shuffled_gt_ap_frac_ge_0.5": ap_weighted(s_sh, yt_sh, ~yt_sh),
                         "shuffled_gt_base_frac_ge_0.5": float(yt_sh.mean()),
                         "shuffled_gt_ap_presence": ap_weighted(s_sh, ya_sh, ~ya_sh),
                         "shuffled_gt_base_presence": float(ya_sh.mean())}}
    # ---- 10 cm, via per-coarse-cell counts ------------------------------------------------------ #
    f5 = fine.reshape(n_w, 120, 5, 64, 5)
    seen_f = (f5 != 255).sum(axis=(2, 4)).astype(np.float64)              # [n, 120, 64]
    seen_f_sh = seen_f[part]
    for c, name in enumerate(CLASSES):
        npos = (f5 == c).sum(axis=(2, 4)).astype(np.float64)
        nneg = seen_f - npos
        npos_sh, nneg_sh = npos[part], seen_f_sh - npos[part]
        rows = {}
        for band, r0, r1 in (("all", 0, 120),) + BANDS:
            sl = np.s_[:, r0:r1, :]
            s_, p_, n_ = P[:, c][sl], npos[sl], nneg[sl]
            tot = float(p_.sum() + n_.sum())
            base = float(p_.sum() / tot) if tot else None
            ap_ = ap_weighted(s_, p_, n_)
            const_ = (ap_weighted(np.full(s_.shape, base), p_, n_) if base is not None else None)
            rows[band] = {"n_windows": n_w, "n_fine_cells": int(tot), "n_pos": int(p_.sum()),
                          "ap": ap_, "base_rate": base, "constant_ap": const_,
                          "constant_equals_base_3dp": (None if const_ is None or base is None
                                                       else round(const_, 3) == round(base, 3)),
                          "location_prior_ap": ap_weighted(PRI[c]["fine"][sl], p_, n_),
                          "shuffled_gt_ap": ap_weighted(s_, npos_sh[sl], nneg_sh[sl]),
                          "shuffled_gt_base": (float(npos_sh[sl].sum() / (npos_sh[sl].sum()
                                                                        + nneg_sh[sl].sum()))
                                               if (npos_sh[sl].sum() + nneg_sh[sl].sum()) else None)}
        res["ten_cm"][name] = rows
    res["wall_s"] = round(time.time() - t0, 1)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    # ---- the table the Master Mind asked for ------------------------------------------------- #
    print(STAMP)
    print(f"n windows {n_w} ({res['n_clips']} clips), seen 0.5 m cells {n_seen:,} of {S.size:,} "
          f"({n_seen_not_valid:,} of them outside map_valid); alignment control {align_bad} / "
          f"{block_bad} cells differ (must be 0 / 0); seen-definition check {seen_def_bad} (must be 0)")
    hdr = (f"{'class':20s} {'AP.5>=.5':>9s} {'base':>7s} {'n_pos':>8s} {'AP pres':>8s} {'base':>7s} "
           f"{'n_pos':>8s} {'AP10 0-20':>9s} {'AP10 20-40':>10s} {'AP10 40-60':>10s} {'base10':>7s} "
           f"{'n_pos10':>9s} {'argmaxIoU':>9s} {'p|pos':>6s} {'p|neg':>6s}")
    print(hdr)

    def f3(v):
        return "   -  " if v is None else f"{v:.3f}"
    for name in CLASSES:
        h, t = res["half_m"][name], res["ten_cm"][name]
        print(f"{name:20s} {f3(h['ap_frac_ge_0.5']):>9s} {f3(h['base_rate_frac_ge_0.5']):>7s} "
              f"{h['n_pos_frac_ge_0.5']:>8d} "
              f"{f3(h['ap_presence_frac_gt_0']):>8s} {f3(h['base_rate_presence']):>7s} "
              f"{h['n_pos_presence']:>8d} "
              f"{f3(t['0-20m']['ap']):>9s} {f3(t['20-40m']['ap']):>10s} {f3(t['40-60m']['ap']):>10s} "
              f"{f3(t['all']['base_rate']):>7s} {t['all']['n_pos']:>9d} {f3(h['argmax_iou']):>9s} "
              f"{f3(h['mean_p_on_pos_frac_ge_0.5']):>6s} {f3(h['mean_p_on_neg_frac_lt_0.5']):>6s}")
    print(f"n per row: {n_w} windows; 0.5 m cells scored per class {res['half_m'][CLASSES[0]]['n_cells']:,}; "
          f"10 cm fine cells scored {res['ten_cm'][CLASSES[0]]['all']['n_fine_cells']:,} "
          f"(bands: " + ", ".join(f"{b} {res['ten_cm'][CLASSES[0]][b]['n_fine_cells']:,}"
                                  for b, _, _ in BANDS) + ")")
    print("CONTROLS: constant predictor AP (must == base to 3 dp) | shuffled GT AP vs its base")
    for name in CLASSES:
        h, t = res["half_m"][name], res["ten_cm"][name]
        c_ = h["controls"]
        print(f"{name:20s} 0.5m const {f3(c_['constant_ap_frac_ge_0.5'])} (== base: "
              f"{c_['constant_equals_base_3dp']}) | shuffled {f3(c_['shuffled_gt_ap_frac_ge_0.5'])} vs "
              f"{f3(c_['shuffled_gt_base_frac_ge_0.5'])} ; presence {f3(c_['shuffled_gt_ap_presence'])} vs "
              f"{f3(c_['shuffled_gt_base_presence'])} | 10cm const " + " ".join(
                  f"{b}:{t[b]['constant_equals_base_3dp']}" for b in ("all",) + tuple(x[0] for x in BANDS))
              + f" | 10cm shuffled all {f3(t['all']['shuffled_gt_ap'])} vs {f3(t['all']['shuffled_gt_base'])}")
    print("SCENE-BLIND FLOORS: location prior (own clip out) AP and cell-permuted model AP (~base)")
    for name in CLASSES:
        h, t = res["half_m"][name], res["ten_cm"][name]
        print(f"{name:20s} 0.5m >=.5: model {f3(h['ap_frac_ge_0.5'])} | loc-prior "
              f"{f3(h['location_prior_ap_frac_ge_0.5'])} | cell-perm {f3(h['cell_permuted_ap_frac_ge_0.5'])} "
              f"| base {f3(h['base_rate_frac_ge_0.5'])} || presence: model {f3(h['ap_presence_frac_gt_0'])} "
              f"loc-prior {f3(h['location_prior_ap_presence'])} || 10cm all: model {f3(t['all']['ap'])} "
              f"loc-prior {f3(t['all']['location_prior_ap'])} | by band model/loc-prior " + " ".join(
                  f"{b} {f3(t[b]['ap'])}/{f3(t[b]['location_prior_ap'])}" for b, _, _ in BANDS))
    print(f"wrote {out} ({res['wall_s']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
