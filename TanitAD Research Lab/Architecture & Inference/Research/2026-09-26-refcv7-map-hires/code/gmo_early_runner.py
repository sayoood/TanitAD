"""EARLY, NON-BINDING G-MAP-OVERFIT on Thor (the Master Mind's request, 2026-09-27 ~03:20).

Runs the candidate's OWN harness (stack/scripts/map_hires_overfit.py, unmodified) in-process
so the peak device memory can be read after it, then STAMPS the record the harness wrote:
``"binding": false`` and ``"why"``, and moves it to a NON-canonical name
(g_map_overfit.EARLY_NONBINDING.json) so no consumer of ``g_map_overfit.json`` can take it
as the launch PASS record. ``launch_commit`` is a non-sha string for the same reason.

Usage: gmo_early_runner.py <code dir> <out dir> <weights json> <harness args...>
"""
import hashlib
import json
import sys
import time
from pathlib import Path

CODE, OUT, W = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
EXTRA = sys.argv[4:]
sys.path.insert(0, str(CODE / "stack"))
sys.path.insert(0, str(CODE / "stack" / "scripts"))
import tanitad  # noqa: E402
import torch  # noqa: E402

assert str(CODE) in tanitad.__file__, tanitad.__file__
import map_hires_overfit as M  # noqa: E402

assert str(CODE) in M.__file__, M.__file__
AUDIT = CODE / "TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-map-signal-audit/raw"
WHY = "early read on candidate <base b4a59b9 + NEW-2 blobs>, not the launch commit"
argv = ["--spec", str(AUDIT / "gmo_spec.json"), "--frameset", str(AUDIT / "gmo_frameset.json"),
        "--class-weights", str(W), "--decision-rule", "prior_corrected",
        "--launch-commit", "NONBINDING-EARLY-b4a59b9+NEW2",
        "--launch-argv-sha256", "NONBINDING-EARLY-no-launch-argv",
        "--out", str(OUT), "--device", "cuda"] + EXTRA
t0 = time.time()
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
rc = M.main(argv)
wall = time.time() - t0
peak = (torch.cuda.max_memory_allocated() / 2 ** 30) if torch.cuda.is_available() else None
src = OUT / "g_map_overfit.json"
rec = json.loads(src.read_text(encoding="utf-8"))
rec["binding"] = False
rec["why"] = WHY
import os  # noqa: E402
_shared = os.environ.get("GMO_GPU_SHARED_WITH", "").strip()
if _shared:
    # the Master Mind's rule: s/step and peak memory are INFORMATIVE only when shared
    rec["gpu_shared_with"] = _shared
rec["early_run"] = {
    "harness_exit_code": int(rc), "wall_s": round(wall, 1),
    "peak_cuda_max_memory_allocated_gib": None if peak is None else round(peak, 3),
    "harness_argv": argv,
    "candidate": {"base_commit": "b4a59b982dbc3f9f0e63451c1afc796e97eaa356",
                  "tree": str(CODE), "shipped_tar_md5": "c40478f561c83abcc6196f680f912071",
                  "per_file_md5_manifest": "MD5SUMS.txt (2,828 files, all OK on Thor)"},
    "weights_json_sha256": hashlib.sha256(W.read_bytes()).hexdigest(),
    "runner": "gmo_early_runner.py (stamps only; the harness is the candidate's, unmodified)"}
dst = OUT / "g_map_overfit.EARLY_NONBINDING.json"
dst.write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")
src.unlink()
print(json.dumps({"stamped": str(dst), "binding": False, "harness_exit_code": rc,
                  "peak_gib": rec["early_run"]["peak_cuda_max_memory_allocated_gib"],
                  "wall_s": round(wall, 1)}), flush=True)
