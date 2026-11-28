#!/usr/bin/env python3
"""D3 check 5 -- decoded-frame sanity: black / over-exposed / frozen frames, bottom black strip, brightness vs day-night label.

usage: c5_frames.py <train_cache_dir> <eval_cache_dir> <train_manifest.pt> <eval_manifest.pt> <labels_train.jsonl.gz> <labels_eval.jsonl.gz> <out.json> [--ntrain N] [--workers K] [--clips-eval N]

Decodes EVERY frame (the cache stores each raw frame as an independent PNG, codec field read from the payload, not guessed from
the buffer name) of N random train clips and all eval clips.
Literal thresholds (8-bit image physics):
  black        : mean luma (rows above the bottom strip) < 5  AND  p99 < 20
  over-exposed : mean luma > 235  OR  more than 50 % of pixels with max(R,G,B) >= 250
  frozen       : the encoded PNG bytes are IDENTICAL to the previous frame's, or the decoded max abs diff == 0
  near-frozen  : mean abs diff to the previous frame < 0.3 / 255 while the ego speed (manifest pose row r-2) > 3 m/s
  bottom strip : number of consecutive all-zero rows from the bottom (committed census: 30..33 rows, rig B only)
Controls run first on synthetic frames through the SAME function: all-zero -> black, all-255 -> over-exposed, an identical pair -> frozen,
a pair differing by one grey level everywhere (MAD 1.0) -> NOT near-frozen, a gradient of known mean -> that mean.
Ids sha12 only.
"""
import glob, gzip, hashlib, io, json, os, sys, time
import multiprocessing as mp
import numpy as np
import torch
from PIL import Image

sha12 = lambda s: hashlib.sha256(s.encode()).hexdigest()[:12]
T_BLACK_MEAN, T_BLACK_P99, T_OVER_MEAN, T_OVER_FRAC, T_NEARFROZEN, V_MOVING = 5.0, 20.0, 235.0, 0.5, 0.3, 3.0


def strip_rows(a):
    z = a.reshape(a.shape[0], -1).max(1) == 0
    n = 0
    for v in z[::-1]:
        if v:
            n += 1
        else:
            break
    return n


def frame_stats(a, strip):
    u = a[:a.shape[0] - strip] if strip else a
    mx = u.max(2)
    lum = u.mean(2)
    p99 = float(np.percentile(lum[::3, ::3], 99))
    m = float(lum.mean())
    return {"mean": m, "p99": p99, "frac_sat": float((mx >= 250).mean()), "black": bool(m < T_BLACK_MEAN and p99 < T_BLACK_P99),
            "over": bool(m > T_OVER_MEAN or (mx >= 250).mean() > T_OVER_FRAC)}


def controls():
    z = np.zeros((416, 1024, 3), np.uint8)
    w = np.full((416, 1024, 3), 255, np.uint8)
    g = np.tile(np.linspace(0, 255, 1024).astype(np.uint8)[None, :, None], (416, 1, 3))
    a = frame_stats(z, 0); b = frame_stats(w, 0); c = frame_stats(g, 0)
    d1 = np.abs(z.astype(np.int16) - z.astype(np.int16)).max()
    one = np.abs(z.astype(np.int16) - (z + 1).astype(np.int16)).mean()
    ok = a["black"] and not a["over"] and b["over"] and not b["black"] and d1 == 0 and abs(one - 1.0) < 1e-9 and abs(c["mean"] - 127.5) < 1.0 and strip_rows(np.concatenate([w[:384], z[:32]], 0)) == 32
    return {"zero_is_black": a["black"], "white_is_over": b["over"], "gradient_mean_expected_127.5": round(c["mean"], 3), "identical_pair_maxdiff_expected_0": int(d1),
            "plus1_pair_MAD_expected_1": float(one), "strip32_expected_32": strip_rows(np.concatenate([w[:384], z[:32]], 0)), "ALL_OK": bool(ok)}


def proc(args):
    path, vspeed, sid = args
    try:
        d = torch.load(path, map_location="cpu", weights_only=False)
        codec = d.get("codec", "png")
        buf = d["jpeg_buf"].numpy(); ln = d["jpeg_len"].numpy()
        off = np.concatenate([[0], np.cumsum(ln)])
        n = len(ln)
        prev_bytes = None; prev = None
        rec = {"sha12": sid, "codec": codec, "n_frames": int(n), "black": [], "over": [], "frozen": [], "nearfrozen": [], "means": [], "strip": None,
               "n_moving_pairs": 0}
        strips = []
        for i in range(n):
            raw = buf[off[i]:off[i + 1]].tobytes()
            a = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
            if i in (0, n // 2, n - 1):
                strips.append(strip_rows(a))
            if i == 0:
                s = strips[0]
            st = frame_stats(a, s)
            rec["means"].append(st["mean"])
            if st["black"]:
                rec["black"].append(i)
            if st["over"]:
                rec["over"].append(i)
            if prev is not None:
                if raw == prev_bytes:
                    rec["frozen"].append(i)
                else:
                    md = np.abs(a.astype(np.int16) - prev.astype(np.int16))
                    if md.max() == 0:
                        rec["frozen"].append(i)
                    else:
                        p = i - 2
                        v = vspeed[p] if (vspeed is not None and 0 <= p < len(vspeed)) else None
                        if v is not None and v > V_MOVING:
                            rec["n_moving_pairs"] += 1
                            if md.mean() < T_NEARFROZEN:
                                rec["nearfrozen"].append(i)
            prev, prev_bytes = a, raw
        rec["strip"] = int(min(strips)); rec["strip_max"] = int(max(strips))
        rec["mean_of_means"] = float(np.mean(rec["means"])); rec["min_mean"] = float(np.min(rec["means"])); rec["max_mean"] = float(np.max(rec["means"]))
        del rec["means"]
        return rec
    except Exception as e:
        return {"sha12": sid, "error": repr(e)[:150]}


def main():
    a = sys.argv
    tr_dir, ev_dir, tr_man, ev_man, lab_tr, lab_ev, outp = a[1:8]
    ntrain = int(a[a.index("--ntrain") + 1]) if "--ntrain" in a else 300
    nw = int(a[a.index("--workers") + 1]) if "--workers" in a else 4
    nev = int(a[a.index("--clips-eval") + 1]) if "--clips-eval" in a else 10 ** 9
    ctl = controls()
    print("CONTROLS", ctl, flush=True)
    assert ctl["ALL_OK"], ctl
    mt = torch.load(tr_man, map_location="cpu", weights_only=False); me = torch.load(ev_man, map_location="cpu", weights_only=False)
    def speeds(man):
        return {c: p[:, 3].numpy().astype(np.float64) for c, p in zip(man["clip_id"], man["poses"])}
    vt, ve = speeds(mt), speeds(me)
    rng = np.random.default_rng(5)
    tr_ids = list(mt["clip_id"]); ev_ids = list(me["clip_id"])[:nev]
    pick = [tr_ids[i] for i in sorted(rng.choice(len(tr_ids), min(ntrain, len(tr_ids)), replace=False))]
    jobs = [(os.path.join(tr_dir, c + ".v2ep.pt"), vt[c], sha12(c)) for c in pick] + [(os.path.join(ev_dir, c + ".v2ep.pt"), ve[c], sha12(c)) for c in ev_ids]
    split = {sha12(c): "train" for c in pick} | {sha12(c): "eval" for c in ev_ids}
    t0 = time.time()
    res = []
    with mp.Pool(nw) as pool:
        for i, r in enumerate(pool.imap_unordered(proc, jobs, chunksize=1)):
            res.append(r)
            if i % 20 == 0:
                print(i, len(jobs), round(time.time() - t0, 1), flush=True)
    # labels for brightness-vs-label
    strata = {}
    for p in (lab_tr, lab_ev):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for l in f:
                r = json.loads(l); strata[sha12(r["clip_id"])] = (r["strata"]["daynight_clock"], r["strata"]["road_class"], r["strata"]["country"])
    out = {"controls": ctl, "n_jobs": len(jobs), "elapsed_s": round(time.time() - t0, 1), "thresholds_literal": {"black_mean_lt": T_BLACK_MEAN, "black_p99_lt": T_BLACK_P99, "over_mean_gt": T_OVER_MEAN,
           "over_frac_sat_gt": T_OVER_FRAC, "nearfrozen_mad_lt": T_NEARFROZEN, "moving_v_gt": V_MOVING}}
    for sp in ("train", "eval"):
        rs = [r for r in res if split.get(r["sha12"]) == sp]
        ok = [r for r in rs if "error" not in r]
        err = [r for r in rs if "error" in r]
        nf = sum(r["n_frames"] for r in ok)
        def agg(k):
            return {"frames": sum(len(r[k]) for r in ok), "clips": sum(1 for r in ok if r[k]), "examples_sha12": [r["sha12"] for r in ok if r[k]][:3]}
        strips = [r["strip"] for r in ok]
        mm = {"day": [], "night": []}
        for r in ok:
            dn = strata.get(r["sha12"], (None,))[0]
            if dn in mm:
                mm[dn].append(r["mean_of_means"])
        out[sp] = {"n_clips": len(rs), "n_decoded_ok": len(ok), "n_unreadable": len(err), "unreadable_examples": [(r["sha12"], r["error"]) for r in err][:3],
                   "codec_values": sorted({r["codec"] for r in ok}), "n_frames_decoded": nf, "black": agg("black"), "over_exposed": agg("over"), "frozen_identical": agg("frozen"),
                   "near_frozen_while_moving": agg("nearfrozen"), "moving_pairs_tested": sum(r["n_moving_pairs"] for r in ok),
                   "bottom_strip_rows_hist": {str(k): strips.count(k) for k in sorted(set(strips))},
                   "n_strip_gt_43": int(sum(s > 43 for s in strips)), "n_strip_0": int(sum(s == 0 for s in strips)),
                   "strip_inconsistent_within_clip": int(sum(r["strip"] != r["strip_max"] for r in ok)),
                   "mean_luma_by_label": {k: {"n": len(v), "q": [round(float(x), 1) for x in np.quantile(v, [0, .05, .5, .95, 1])] if v else None} for k, v in mm.items()},
                   "day_clips_mean_lt_15": int(sum(m < 15 for m in mm["day"])), "night_clips_mean_gt_100": int(sum(m > 100 for m in mm["night"])),
                   "per_clip": [{"sha12": r["sha12"], "strip": r["strip"], "mean": round(r["mean_of_means"], 1), "n_black": len(r["black"]), "n_over": len(r["over"]), "n_frozen": len(r["frozen"]),
                                 "n_nearfrozen": len(r["nearfrozen"])} for r in ok]}
    json.dump(out, open(outp, "w"), indent=1)
    print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != "per_clip"} if isinstance(v, dict) and "per_clip" in v else v) for k, v in out.items()}, indent=1)[:5000])


if __name__ == "__main__":
    main()
