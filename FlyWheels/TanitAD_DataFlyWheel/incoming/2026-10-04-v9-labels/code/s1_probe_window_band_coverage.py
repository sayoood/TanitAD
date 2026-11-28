"""WP-A stage-1 probe: what fraction of the TRAINER'S windows (D1 truth table, row == ds index) have
the egomotion log covering [NOW, NOW+8 s], by the largest sample gap inside that interval."""
import os, json, hashlib, numpy as np, pandas as pd, torch
EGO = "C:/Users/Admin/tanitad-data/physicalai/labels/egomotion_alpamayo/{}.parquet"
TT = "D:/Projects/TanitAD/TanitAD Research Lab/Data Engineering/Research/2026-10-04-refcv8-data-audit/D1_label_census/tables/label_truth_{}.npz"
MAN = {"eval139": "D:/refcv6_eval_kit/data/refcv6-b1-416x1024-eval139/_v2manifest.pt",
       "train": "D:/Projects/TanitAD-artifacts/refcv8_audit/D2/train_v2manifest.pt"}
res = {}
for split in ("eval139", "train"):
    z = np.load(TT.format(split), allow_pickle=True)
    sha = z["clip_sha12"]; tnow = z["t_now_s"].astype(np.float64)
    m = torch.load(MAN[split], map_location="cpu", weights_only=False)
    cid_by_sha = {hashlib.sha256(c.encode()).hexdigest()[:12]: c for c in m["clip_id"]}
    maxgap = np.full(len(tnow), np.inf); end_ok = np.zeros(len(tnow), bool); in100 = np.zeros(len(tnow), bool)
    for s in np.unique(sha):
        idx = np.nonzero(sha == s)[0]
        ts = pd.read_parquet(EGO.format(cid_by_sha[s.decode() if isinstance(s, bytes) else str(s)]), columns=["timestamp"]).timestamp.to_numpy(np.float64) / 1e6
        ts = ts - ts[0]
        g = np.diff(ts)
        for i in idx:
            a, b = tnow[i], tnow[i] + 8.0
            lo = np.searchsorted(ts, a, "right") - 1; hi = np.searchsorted(ts, b, "left")
            lo = max(lo, 0); hi = min(hi, len(ts) - 1)
            maxgap[i] = g[lo:hi].max() if hi > lo else np.inf
            end_ok[i] = ts[-1] >= b
            in100[i] = b <= 20.0
    n = len(tnow)
    r = {"n_windows": int(n), "log_reaches_now_plus_8": int(end_ok.sum())}
    for thr in (0.25, 0.5, 1.0, 2.0):
        r[f"reach8_and_maxgap_le_{thr}s"] = int((end_ok & (maxgap <= thr)).sum())
    r["band_inside_100Hz_part(now+8<=20s)"] = int(in100.sum())
    r["maxgap_q_on_reach8"] = np.round(np.percentile(maxgap[end_ok], [50, 75, 90, 95, 99]), 3).tolist()
    res[split] = r
    print(split, json.dumps(r))
json.dump(res, open("probe_wincov.json", "w"), indent=1)
