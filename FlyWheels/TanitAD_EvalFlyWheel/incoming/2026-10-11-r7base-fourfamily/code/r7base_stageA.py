"""r7base stage A (CPU only, no GPU, no Colab): rebuild the registered S6 read from the banked refcv7-50,400
FULL plans (`decisions/epNNN.npz::plan_full_nav_true`, 8 slots = 0.5..6 s) and re-run the battery's own analysis
(refcv3_arm --analyze-only) on S2 and S6 with the TIP instruments and the X7 6 s lead block (k12, dt 0.5).

usage: python r7base_stageA.py <seed> <out_root>
"""
import json
import os
import sys
import time
from pathlib import Path

seed = int(sys.argv[1])
out_root = Path(sys.argv[2])
BAT = Path(__file__).resolve().parent / "battery"
sys.path.insert(0, str(BAT))
sys.stdout.reconfigure(encoding="utf-8")
import tanitad  # noqa: E402
print("tanitad from", tanitad.__file__, flush=True)
import r7_panel as P  # noqa: E402
import taniteval  # noqa: E402
print("taniteval from", taniteval.__file__, flush=True)

BANK = Path("D:/refcv7_eval_kit/battery/step50400_a6text")
LEAD12 = "C:/Users/Admin/qland/landed_hold_x7/artifacts/b1_eval_lead_block_k12_dt0p5.npz"
assert os.path.exists(LEAD12)
P.LEAD_BLOCK = LEAD12
KIT_LABELS = "D:/refcv6_eval_kit/data/v8labels/labels/s2_labels_v8_eval.jsonl.gz"
out_root.mkdir(parents=True, exist_ok=True)
t0 = time.time()
dump = str(BANK / f"dump_s{seed}")

# S6 from the banked full plans (no baselines: the refcv6@38k/refcv4b/refcv5v2 dumps are not this row)
s6d = str(out_root / f"s6_s{seed}")
if not os.path.exists(os.path.join(s6d, "manifest.json")):
    rec = P.build_s6_dump(dump, s6d, baselines={})
    json.dump(rec, open(out_root / f"s6_build_s{seed}.json", "w", encoding="utf-8"), indent=1, default=str)
print("[A] s6 built", round(time.time() - t0, 1), flush=True)
a6 = P.analyze(s6d, str(out_root / f"analysis_s6_s{seed}.json"), KIT_LABELS, n_boot=2000)
print("[A] s6 analysed", round(time.time() - t0, 1), flush=True)
# S2: the banked panel (os + controls on the 2 s grid), re-analysed with the same lead block
p2 = str(BANK / f"panel_s{seed}")
a2 = P.analyze(p2, str(out_root / f"analysis_s2_s{seed}.json"), KIT_LABELS, n_boot=2000)
print("[A] s2 analysed", round(time.time() - t0, 1), flush=True)
print("ZZSTAGEA_DONE_%d" % seed)
