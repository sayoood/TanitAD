import json
from pathlib import Path
p = Path("C:/Users/Admin/refcv8_pf/refcv9_r9pf/code/deviations_r9pf.json")
d = json.loads(p.read_text(encoding="utf-8"))
by = {x["id"]: x for x in d}
by["D10"]["text"] = ("SUPERSEDED by AMENDMENT1 change 4: the PRIMARY eval path is the battery tree's BANKED fb47a58 plan (refcv6_extras nav_true.traj; "
    "bit-equal to the plan8 inside H2CN_DEPLOY8 and RETIME_DEPLOY8 at both seeds, K-INPUTS). The eval bank's traj_own equals it bit for bit at seed 1 "
    "but differs at seed 0 on 4,630 / 4,754 S2 windows (max 35.31 m); R9 compositions on traj_own are banked beside (*_own arrays). "
    "FACTORED_DEPLOY8's plan8 equals NEITHER path at either seed (it is the factored decoder's own native path), so PA-1b compares two decoders "
    "on their own paths, as the sealed s1 defines FACTORED_DEPLOY8. The OSONLY beside arm = the battery plan without re-timing.")
by["D14"]["text"] = ("SUPERSEDED by AMENDMENT1 changes 2 / 3: K-OFF = flags-ON vs flags-OFF on ONE tree (tip + R9 patch + the pinned one-line RNG fix, "
    "F:/refcv9_r9pf_scratch/launch_tree_fix) at inference seeds 0 AND 1; K-EQ = same-input form at seeds 0 AND 1 (both required -- the conservative "
    "reading; the amendment names seeds only for K-OFF). Subset = the first clips in sha12 order until >= 500 windows of the battery's "
    "windows_s{seed}.json, filtered in memory; the tip loader's pose-sync shift is blanked under R8_NO_POSE_SYNC=1 and asserted not attached. "
    "'Every non-R9 output' = every tensor the model returns per call (sha256 of bytes + dtype + shape, all rows and conditions, nav-null calls included); "
    "the K-EQ same inputs = the forward's own traj_preretime, z_tac (lat_head_tac input) and r9_x, plus the scoring prior8 (s_prior agreement reported).")
d += [
 {"id": "D16", "text": "ONE composition per window: the S6 compliant rows are a subset of the S2 rows, and the S6 arm := the S2 composition indexed to "
   "those windows (every S6 input asserted bit-equal to its S2 row). A separate S6-batch composition differs from the S2-batch one by float32 "
   "batch-order rounding only -- 7 / 3,668 rows, <= 7.63e-6 m, 0 decision flips (R9PF_COMPOSE.json, K-S6CONSIST 'separate_batch_composition_beside')."},
 {"id": "D17", "text": "The R9 RNG fix (AMENDMENT1 change 1) is a byte-exact one-line replacement at refc_v3.py line 2125 (CRLF preserved; "
   "code/FIX_R9_RNG.DIFF.txt; the fixed file, the diff, the unchanged refcv9_progress.py and the unfixed reference are pinned in FIX_R9_RNG.sha256). "
   "An earlier sed attempt converted the file's line endings and was discarded unused; the pinned file is the byte-clean one. Every a3 script "
   "re-hashes the amendment seal and every FIX_R9_RNG.sha256 line and refuses otherwise."},
 {"id": "D18", "text": "Process note: two shell commands in this run fed a heredoc AND '</dev/null' to python; on Windows NUL is a console device, "
   "so python opened an interactive REPL that looped on an invalid handle (~64 MB of error text into the harness's own output file). Both "
   "processes were stopped by their task; neither wrote any file in the work dir or scratch, and both edits were re-applied from patch files."},
]
p.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print([x["id"] for x in d])
