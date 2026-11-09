"""Release merge for the per-frame context labels (PI 2026-11-08). Geometry shards + ctxA (every clip, every 0.2 s frame) + ctxB (Qwen at 1 Hz, a selected subset of clips).

Differences from the pilot merge (merge_context.py, kept as it was for the pilot's record):
 * ctxB is optional: a clip without Qwen answers gets every frame from the probe. `vlm_clip` says which is which, `<field>_src` gives the source per frame.
 * Probe training may carry per-clip weights that undo the oversampling of rare clips in the Qwen subset (splits/qwen_clips.json).
 * Temporal treatment is a sticky forward-backward over the per-frame distributions with the switch rate chosen in smooth_eval.py (labels are built offline, so two-sided).
 * Glare is folded into day (0 of the PI's 4 glare frames matched; audit 2026-11-08). View quality is not released (not credible in the pilot, not in the PI's audit).
 * Illumination comes from Qwen plus the probe when ctxB holds v_illum, otherwise from SigLIP zero-shot; the raw folded SigLIP distribution is always kept as illum_img_p.
`country`, `month`, `hour_of_day` are metadata in the geometry columns (every frame). Inference stays vision-only: these are labels, never model inputs.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import extract_context as X
import merge_context as M
import strat_common as C

ILLUM4 = X.ILLUM[:4]
FIELDS = (("weather", X.WEATHER, "v_weather", "z_weather"), ("surface", X.SURFACE, "v_surface", "z_surface"), ("visibility", X.VISIB, "v_visibility", None),
          ("illumination", ILLUM4, "v_illum", "z_illum"))
CTX_COLUMNS = ["luma_mean", "luma_p05", "luma_p95", "sky_luma", "sat_mean", "headlight_frac", "lit_frac", "illum_img_p", "illum_img",
               "weather_p", "weather", "weather_src", "z_weather_p", "surface_p", "surface", "surface_src", "z_surface_p",
               "visibility_p", "visibility", "visibility_src", "illumination_p", "illumination", "illumination_src",
               "vlm_clip", "vlm_frame", "sun_el_deg_est", "sun_band_est", "sun_conf", "illum_agree_sun", "illum_sun_flag", "period_of_day",
               "model_sha_siglip", "model_sha_qwen", "prompt_sha", "smooth_switch"]
ROWS = M.ROWS_PER_CLIP
UNKNOWN_BELOW = 0.4
# PROBE_GRID.json (pilot, 187 train clips, out of fold). The pilot's C = 0.5 is far too strong on unit-length embeddings: rain recall against Qwen 4.7 %, PI rain 1/9, merged rain 0.0 % of frames while Qwen says 6.8 %.
# POST-HOC DEVIATION, logged in CONTEXT_LABELS_CARD.md: the selector (macro-F1 against Qwen) picked class_weight="balanced" for weather (0.637 against 0.572 at C = 5). On the PI's frames balanced is 45/55 and
# unweighted 50/55 (rare classes 14/18 against 13/18, clear 31/37 against 37/37), and the release publishes distributions, which class weighting biases. So: C = 5, unweighted, for every field.
PROBE_C = 5.0
VLM_FRAMES_REPRESENTED = 5   # a 1 Hz Qwen answer stands for the 5 frames of its 0.2 s grid; unweighted it is outvoted 4 to 1 by the probe frames between answers


def fold_glare(z):
    o = z[:, :4].copy()
    o[:, 3] += z[:, 4]
    return o


def zshot(A, zk):
    if zk is None:
        return None
    return fold_glare(A[zk]) if zk == "z_illum" else A[zk]


def and_visibility(B):
    """PROMPT_DECISION.md: a not-good visibility verdict needs both questions to say not-good. The raw answers stay in the ctxB parts."""
    if "v_vis2" not in B or "v_visibility" not in B:
        return B
    a, b = B["v_visibility"], B["v_vis2"]
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    bad = (a.argmax(1) != 0) & (b.argmax(1) != 0)
    good_p = np.where(bad, np.minimum(np.minimum(a[:, 0], b[:, 0]), 0.3), np.maximum(np.maximum(a[:, 0], b[:, 0]), 0.51))
    m = (a[:, 1:] + b[:, 1:]) / 2
    m = m / np.clip(m.sum(1, keepdims=True), 1e-6, None)
    B = dict(B)
    B["v_visibility"] = np.concatenate([good_p[:, None], (1 - good_p[:, None]) * m], 1).astype(np.float32)
    return B


def load_parts(out, clip):
    pa_, pb_ = X.part_path(out, "A", clip), X.part_path(out, "B", clip)
    if not os.path.exists(pa_):
        return None
    return dict(np.load(pa_)), (and_visibility(dict(np.load(pb_))) if os.path.exists(pb_) else None)


def fit_probes(data, train_clips, weights=None):
    """field -> (oof {clip: [n,ncls]} for train clips with Qwen, full model). Clip weights undo the oversampling of the subset selection."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    weights = weights or {}
    res = {}
    for fname, classes, vk, _z in FIELDS:
        Xs, ys, gs, ws = [], [], [], []
        for c in train_clips:
            if c in data and data[c][1] is not None and vk in data[c][1]:
                A, B = data[c]
                F = M.feats(A)[::5][: len(B[vk])]
                v = B[vk][: len(F)]
                ok = np.isfinite(v).all(1)                                  # a non-finite Qwen answer is no answer (1 frame of the 21,000 or so: clip e8e57d1b, 1 Hz frame 3)
                Xs.append(F[ok])
                ys.append(v[ok].argmax(1))
                gs += [c] * int(ok.sum())
                ws += [float(weights.get(c, 1.0))] * int(ok.sum())
        if not Xs:
            continue
        Xa, ya, ga, wa = np.concatenate(Xs), np.concatenate(ys), np.array(gs), np.array(ws, np.float64)
        if len(set(ya.tolist())) < 2:
            res[fname] = ({}, None)
            continue
        full = LogisticRegression(max_iter=1000, C=PROBE_C).fit(Xa, ya, sample_weight=wa)
        oof = {}
        n_groups = len(set(ga.tolist()))
        if n_groups >= 3:
            for tr, te in GroupKFold(min(5, n_groups)).split(Xa, ya, ga):
                if len(set(ya[tr].tolist())) < 2:
                    continue
                m = LogisticRegression(max_iter=1000, C=PROBE_C).fit(Xa[tr], ya[tr], sample_weight=wa[tr])
                for c in set(ga[te].tolist()):
                    oof[c] = M._proba(m, M.feats(data[c][0]), len(classes))
        res[fname] = (oof, full)
    return res


def clip_context(clip, A, B, probes, meta, switch):
    n = len(A["emb"])
    r = {nm: A[f"t1_{nm}"] for nm in ("luma_mean", "luma_p05", "luma_p95", "sky_luma", "sat_mean", "headlight_frac", "lit_frac")}
    z4 = fold_glare(A["z_illum"])
    r["illum_img_p"] = z4
    r["illum_img"] = np.array(ILLUM4)[z4.argmax(1)]
    vlm_clip = B is not None
    vlm_any = np.zeros(n, bool)
    for fname, classes, vk, zk in FIELDS:
        ncls = len(classes)
        has_v = vlm_clip and vk in B
        vlm = np.zeros(n, bool)
        vp = np.zeros((n, ncls), np.float32)
        if has_v:
            idx = np.arange(0, n, 5)[: len(B[vk])]
            ans = B[vk][: len(idx)]
            ok = np.isfinite(ans).all(1)                                    # a non-finite answer is no answer: the frame takes the probe; the forward-backward pass would otherwise spread one NaN over the whole clip
            vlm[idx[ok]] = True
            vp[idx[ok]] = ans[ok]
            vlm_any |= vlm
        if fname in probes and probes[fname][1] is not None:
            oof, full = probes[fname]
            pp = oof.get(clip)
            pp = pp if pp is not None else M._proba(full, M.feats(A), ncls)
            other = "probe"
        elif zk is not None:
            pp = zshot(A, zk)
            other = "zeroshot"
        else:
            pp = np.full((n, ncls), 1.0 / ncls, np.float32)
            other = "none"
        W = np.clip(vp, 1e-3, 1.0) ** VLM_FRAMES_REPRESENTED
        W = W / W.sum(1, keepdims=True)
        E = np.where(vlm[:, None], W, pp)
        P = M.sticky_posterior(E, switch[fname])
        r[fname + "_p"] = P
        lab = np.array(list(classes))[P.argmax(1)]
        r[fname] = np.where(P.max(1) < UNKNOWN_BELOW, "unknown", lab)
        r[fname + "_src"] = np.where(vlm, "vlm", other)
        if fname != "illumination" and zk:
            r["z_" + fname + "_p"] = A[zk]
    r["vlm_clip"] = np.full(n, vlm_clip)
    r["vlm_frame"] = vlm_any
    el = M.sun_el_deg(meta["country"], int(meta["month"]), int(meta["hour_of_day"]))
    sb = M.sun_band(el)
    r["sun_el_deg_est"] = np.full(n, el, np.float32)
    r["sun_band_est"] = np.full(n, sb)
    r["sun_conf"] = np.full(n, "low")
    d = np.array([abs(M.IMG_ORD[x] - M.SUN_ORD[sb]) for x in r["illum_img"]])
    r["illum_agree_sun"] = d <= 1
    r["illum_sun_flag"] = d >= 2
    r["period_of_day"] = np.full(n, M.period_of_day(int(meta["hour_of_day"])))
    return r


MIN_ROWS = 96                                                               # MEASURED 2026-11-09: 9 of 4,659 clips hold 96 or 97 present frames in the source video (geometry: 101 rows, 96 or 97 frame_present); every present frame is labelled


def check_frames(frames_dir):
    """Schema pin of the release: geometry columns then CTX_COLUMNS; 100 or 101 rows per clip, each with a frame; a country on every row; finite distributions."""
    import pandas as pd
    errs = []
    fs = sorted(glob.glob(frames_dir + "/frames_*.parquet"))
    if not fs:
        return ["no frame parquets"]
    for f in fs:
        d = pd.read_parquet(f)
        want = list(C.GEOM_COLUMNS) + CTX_COLUMNS
        if list(d.columns) != want:
            errs.append(f"{os.path.basename(f)}: columns differ, missing {[c for c in want if c not in d.columns]}, extra {[c for c in d.columns if c not in want]}")
            continue
        if not (d.groupby("clip_id").size().between(MIN_ROWS, ROWS).all() and bool(d.frame_present.all())):
            errs.append(f"{os.path.basename(f)}: rows per clip outside {MIN_ROWS} to {ROWS}, or a row without a frame")
        if d.country.isna().any() or (d.country.astype(str).str.len() == 0).any():
            errs.append(f"{os.path.basename(f)}: a row without a country")
        for c in ("weather_p", "surface_p", "visibility_p", "illumination_p", "illum_img_p"):
            a = np.stack(d[c].to_numpy())
            if not np.isfinite(a).all() or abs(a.sum(1) - 1).max() > 1e-2:
                errs.append(f"{os.path.basename(f)}: {c} not a finite distribution")
    return errs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom", default="F:/strat_out/geom")
    ap.add_argument("--out", default="F:/strat_out", help="holds ctxA and ctxB")
    ap.add_argument("--release", default="F:/strat_out/release")
    ap.add_argument("--qwen-clips", default=HERE + "/splits/qwen_clips.json")
    ap.add_argument("--smooth", default=HERE + "/SMOOTH_EVAL.json")
    ap.add_argument("--model-sha", default="F:/strat_out/MODEL_SHA.json")
    ap.add_argument("--clips-table", default=HERE + "/splits/clips_table.parquet")
    a = ap.parse_args(argv)
    import pandas as pd
    tab = pd.read_parquet(a.clips_table)
    msha = json.load(open(a.model_sha)) if os.path.exists(a.model_sha) else {}
    sm = json.load(open(a.smooth))
    chosen = {f: sm[f]["_chosen"] for f in ("weather", "surface", "visibility")}
    # POST-HOC DEVIATION, logged in CONTEXT_LABELS_CARD.md: smooth_eval's rule (written before it ran) picked "raw" for surface, 83.31 % against 82.70 % at switch 0.01. That rule scores
    # against Qwen's 1 Hz answers, which flicker themselves, so it rewards flicker. Raw flips 3.33 % per 0.2 s step (about 3 changes per clip), 0.134 % at 0.01; the PI frames tie on the random 40 (34/37 both).
    deviation = {"surface": 0.01}
    numeric = {f: deviation[f] if f in deviation else float(v) for f, v in chosen.items()}
    switch = {**numeric, "illumination": numeric["weather"]}
    weights = json.load(open(a.qwen_clips)).get("weights", {}) if os.path.exists(a.qwen_clips) else {}
    shards = sorted(glob.glob(f"{a.geom}/geom_*.parquet"))
    clips = []
    for s in shards:
        clips += pq.read_table(s, columns=["clip_id"]).column(0).unique().to_pylist()
    data = {c: p for c in clips if (p := load_parts(a.out, c)) is not None}
    train = tab[tab.split == "train4512"].clip_id.tolist()
    probes = fit_probes(data, train, weights)
    os.makedirs(a.release + "/frames", exist_ok=True)
    clip_rows = []
    for s in shards:
        g = pq.read_table(s)
        have = [c for c in g.column("clip_id").unique().to_pylist() if c in data]
        if not have:
            continue
        g = g.filter(pc.is_in(g.column("clip_id"), pa.array(have)))
        parts = []
        for c in have:
            sub = g.filter(pc.equal(g.column("clip_id"), c))
            meta = {k: sub.column(k)[0].as_py() for k in ("country", "month", "hour_of_day")}
            A, B = data[c]
            if len(sub) != int(A["n_frames"]):
                sub = sub.filter(sub.column("frame_present"))
                assert len(sub) == int(A["n_frames"]), (c, len(sub), int(A["n_frames"]))
            r = clip_context(c, A, B, probes, meta, switch)
            cols = M.to_arrow(r, len(sub))
            cols["model_sha_siglip"] = pa.array([msha.get("siglip2", "NA")] * len(sub))
            cols["model_sha_qwen"] = pa.array([msha.get("qwen25vl3b", "NA")] * len(sub))
            cols["prompt_sha"] = pa.array([X.PROMPT_SHA] * len(sub))
            cols["smooth_switch"] = pa.array([json.dumps(switch, sort_keys=True)] * len(sub))
            order = [n for n in CTX_COLUMNS]
            parts.append(pa.table({**{n: sub.column(n) for n in sub.column_names}, **{n: cols[n] for n in order}}))
            clip_rows.append({"clip_id": c, "vlm_clip": B is not None, "country": meta["country"],
                              **{f"{f}_mode": pd.Series(r[f]).mode().iat[0] for f in ("weather", "surface", "visibility", "illumination")},
                              "period_of_day": r["period_of_day"][0], "sun_band_est": r["sun_band_est"][0], "illum_sun_flag_frac": float(np.mean(r["illum_sun_flag"]))})
        pq.write_table(pa.concat_tables(parts), f"{a.release}/frames/{os.path.basename(s).replace('geom_', 'frames_')}", compression="zstd")
    cl = pd.DataFrame(clip_rows).merge(tab.drop(columns=[c for c in ("country",) if c in tab.columns]), on="clip_id", how="left")
    cl["label_version"] = C.LABEL_VERSION
    os.makedirs(a.release + "/clips", exist_ok=True)
    cl.to_parquet(a.release + "/clips/clips.parquet", index=False)
    errs = check_frames(a.release + "/frames")
    print(json.dumps({"clips_merged": len(clip_rows), "with_qwen": int(cl.vlm_clip.sum()), "countries": int(cl.country.nunique()), "probe_fields": list(probes),
                      "switch": switch, "chosen": chosen, "schema_errors": errs}))
    return 3 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
