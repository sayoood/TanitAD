import os, json, numpy as np, pandas as pd, torch, hashlib
EGO = "C:/Users/Admin/tanitad-data/physicalai/labels/egomotion_alpamayo/{}.parquet"
side = {}
for l in open("D:/refcv6_eval_kit/data/refcv6_clip_clock_sidecar.jsonl", encoding="utf-8"):
    if l.strip():
        r = json.loads(l); side[int(r["sid"])] = (float(r["grid_start_s"]), float(r["dt_s"]))
out = {}
for split, mp in (("eval", "D:/refcv6_eval_kit/data/refcv6-b1-416x1024-eval139/_v2manifest.pt"),
                  ("train", "D:/Projects/TanitAD-artifacts/refcv8_audit/D2/train_v2manifest.pt")):
    m = torch.load(mp, map_location="cpu", weights_only=False)
    cids = m["clip_id"]; uids = m["episode_uid"]
    T = [int(p.shape[0]) for p in m["poses"]]
    n_nolog = 0; spans = []; fr_total = 0; fr_ok8 = 0; fr_ok8_gap = 0; maxgap_20_30 = []; post_rate = []
    for i, cid in enumerate(cids):
        p = EGO.format(cid)
        if not os.path.exists(p):
            n_nolog += 1; continue
        ts = pd.read_parquet(p, columns=["timestamp"]).timestamp.to_numpy(np.float64) / 1e6
        ts = ts - ts[0]
        spans.append(ts[-1])
        g0, dt = side.get(int(uids[i]), (0.0, 0.1007))
        k = np.arange(T[i]) + 2
        now = g0 + k * dt
        fr_total += len(now)
        ok8 = now + 8.0 <= ts[-1]
        fr_ok8 += int(ok8.sum())
        # max gap inside [now, now+8]
        gaps = np.diff(ts); mid = ts[:-1]
        mg = np.zeros(len(now))
        for j, t in enumerate(now):
            sel = (ts[1:] > t) & (ts[:-1] < t + 8.0)
            mg[j] = gaps[sel].max() if sel.any() else 99
        fr_ok8_gap += int((ok8 & (mg <= 0.5)).sum())
        s2 = (ts[1:] > 20.0) & (ts[:-1] < 30.0)
        maxgap_20_30.append(gaps[s2].max() if s2.any() else np.nan)
        sel = ts > 20.5
        if sel.sum() > 2: post_rate.append(sel.sum() / (ts[-1] - 20.5))
    out[split] = {"n_clips": len(cids), "n_no_log": n_nolog,
                  "span_q": np.round(np.percentile(spans, [0, 5, 25, 50, 95, 100]), 2).tolist(),
                  "frac_span_ge_30": float(np.mean(np.array(spans) >= 30)), "frac_span_ge_60": float(np.mean(np.array(spans) >= 60)),
                  "frames": fr_total, "frames_ok8": fr_ok8, "frames_ok8_gap_le_0.5s": fr_ok8_gap,
                  "maxgap_20_30_q": np.round(np.nanpercentile(maxgap_20_30, [5, 25, 50, 75, 95, 99]), 3).tolist(),
                  "post20_rate_hz_q": np.round(np.percentile(post_rate, [5, 25, 50, 75, 95]), 2).tolist()}
    print(split, json.dumps(out[split]))
json.dump(out, open("probe_cov.json", "w"), indent=1)
