"""MEASURE the trainer's host-RAM cost per frame for the two banks it holds in memory.

Consumer = refe/train.py (ScorerBank keeps traj/tgt/name per candidate as Python lists;
TargetBank keeps every row dict whole). tracemalloc counts exactly what Python allocated, which is
the admissible number here (RSS on Windows would include the torch import).
"""
import sys
import tracemalloc

PKG = ("D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/"
       "2026-09-20-refe-plan/refe")
sys.path.insert(0, PKG)
import train as T  # noqa: E402
from model import REFeConfig  # noqa: E402

D = sys.argv[1]
cfg = REFeConfig.for_backbone("vits16")
tracemalloc.start()
b0 = tracemalloc.get_traced_memory()[0]
sb = T.ScorerBank(D)
b1 = tracemalloc.get_traced_memory()[0]
tb = T.TargetBank(D, None, cfg, synthetic=True, scorer=sb)
b2 = tracemalloc.get_traced_memory()[0]
nf = len(sb.by)
nr = len(tb.rows)
per_frame_sc = (b1 - b0) / max(nf, 1)
per_row_tb = (b2 - b1) / max(nr, 1)
cand = sb.n_rows / max(nf, 1)
print(f"scorer: {sb.n_rows} candidate rows over {nf} frames ({cand:.1f}/frame) -> "
      f"{(b1-b0)/1e6:.2f} MB = {per_frame_sc/1e3:.1f} KB/frame")
print(f"targets: {nr} rows -> {(b2-b1)/1e6:.2f} MB = {per_row_tb/1e3:.1f} KB/row")
# full-scale arithmetic: rank-0 frames x (1 + aug yield); both banks cover every target row
N0 = 103_288 * (18_135 / 18_179)
for y in (0.71,):
    rows = N0 * (1 + y)
    print(f"FULL SCALE at aug yield {y:.2f}: {rows:,.0f} target rows -> scorer "
          f"{rows*per_frame_sc/2**30:.1f} GiB + targets {rows*per_row_tb/2**30:.1f} GiB = "
          f"{rows*(per_frame_sc+per_row_tb)/2**30:.1f} GiB host RAM (ESTIMATED, linear in rows)")
print(f"ZZRAM_{per_frame_sc/1e3:.1f}KBF_{per_row_tb/1e3:.1f}KBRZZ")
