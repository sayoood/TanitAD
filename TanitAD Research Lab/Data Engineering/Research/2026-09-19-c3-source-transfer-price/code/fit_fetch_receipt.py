"""Price the HF -> dev-box pull of the 4,713 train mp4s from a MEASURED receipt.

Input: the per-clip fetch receipt banked by the 2026-09-16 cache package
(`raw/fetch_receipt_eval139.json`): 139 serial `hf_hub_download` calls of
`camera/<clip>.mp4`, each with its byte count and its own wall time.

Why a two-parameter fit and not the aggregate rate: the aggregate 5.345 MB/s
folds a fixed per-request cost (redirect, TLS, CDN first byte) into a byte
rate. The corpus mean clip (13.06 MB) is SMALLER than this sample's (14.64 MB),
so a pure byte rate under-prices the corpus. dl_s = a + MB / BW separates them.

Nothing is downloaded. Output: raw/fetch_fit.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

REC = Path(sys.argv[1])
TRAIN_BYTES = int(sys.argv[2])          # exact, from raw/verify_local_source.json
N_TRAIN = int(sys.argv[3])
OUT = Path(sys.argv[4])

r = json.loads(REC.read_text(encoding="utf-8"))
c = [x for x in r["clips"] if not x.get("cached")]
mb = np.array([x["bytes"] for x in c], float) / 1e6
t = np.array([x["dl_s"] for x in c], float)
A = np.vstack([np.ones_like(mb), mb]).T
a, k = np.linalg.lstsq(A, t, rcond=None)[0]
r2 = 1 - ((t - A @ np.array([a, k])) ** 2).sum() / ((t - t.mean()) ** 2).sum()

rng = np.random.default_rng(0)
boot = []
for _ in range(4000):
    i = rng.integers(0, len(c), len(c))
    ai, ki = np.linalg.lstsq(A[i], t[i], rcond=None)[0]
    boot.append(N_TRAIN * ai + TRAIN_BYTES / 1e6 * ki)     # seconds, serial
boot = np.array(boot)

serial_fit_s = N_TRAIN * a + TRAIN_BYTES / 1e6 * k
serial_agg_s = TRAIN_BYTES / 1e6 / (mb.sum() / t.sum())
res = {
    "receipt": str(REC), "n_clips_fit": len(c),
    "aggregate_mb_s": round(float(mb.sum() / t.sum()), 3),
    "overhead_s_per_request": round(float(a), 3),
    "per_stream_bw_mb_s": round(float(1 / k), 2),
    "fit_r2": round(float(r2), 3),
    "train_bytes": TRAIN_BYTES, "n_train": N_TRAIN,
    "serial_hours_fit": round(serial_fit_s / 3600, 2),
    "serial_hours_fit_ci95": [round(float(np.percentile(boot, 2.5)) / 3600, 2),
                              round(float(np.percentile(boot, 97.5)) / 3600, 2)],
    "serial_hours_aggregate_rate": round(serial_agg_s / 3600, 2),
    "parallel_note": ("UNMEASURED. Parallel streams hide the per-request overhead; the "
                      "link ceiling was never measured, so a k-stream figure is only "
                      "(serial / k) if the link carries k x per-stream bandwidth."),
}
OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps(res, indent=1))
