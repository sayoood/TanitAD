"""EARLY, NON-BINDING G-MAP-OVERFIT of the NEW-2 R3 decoder arm (SPEC_REFCV7 §20, A15) on Thor.

Runs the R3 candidate's OWN harness (stack/scripts/map_hires_overfit.py, unmodified) in-process
with the REGISTERED A15 spec (raw/gmo_spec_A15.json: the A12 spec + near_refine_blocks 1 +
must-fail near_block_zeros), then STAMPS the record: ``"binding": false`` + ``"why"``, and
moves it to a NON-canonical name (g_map_overfit_A15.EARLY_NONBINDING.json) so no consumer of
``g_map_overfit.json`` can take it as a launch PASS record. ``launch_commit`` is a non-sha.

Usage: gmo_r3_runner.py <code dir> <out dir> <weights json> [<gpu_shared_with>] -- <harness args>
"""
import hashlib
import json
import sys
import time
from pathlib import Path

i = sys.argv.index("--")
CODE, OUT, W = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
SHARED = sys.argv[4] if i > 4 else ""
EXTRA = sys.argv[i + 1:]
sys.path.insert(0, str(CODE / "stack"))
sys.path.insert(0, str(CODE / "stack" / "scripts"))
import tanitad  # noqa: E402
import torch  # noqa: E402

assert str(CODE) in tanitad.__file__, tanitad.__file__
import map_hires_overfit as M  # noqa: E402

assert str(CODE) in M.__file__, M.__file__
AUD = CODE / "TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-map-signal-audit/raw"
SPEC = CODE / ("TanitAD Research Lab/Architecture & Inference/Research/"
               "2026-09-26-refcv7-map-hires/raw/gmo_spec_A15.json")
A15_MD5 = "20929e21a577a6374f31b3f754ddd4d4"
assert hashlib.md5(SPEC.read_bytes()).hexdigest() == A15_MD5, "the A15 spec is not as registered"
WHY = ("early read of the A15 decoder arm (stacked on A12) on candidate <base 2374cd2 + NEW-2 "
       "R3 blobs>, not the launch commit")
argv = ["--spec", str(SPEC), "--frameset", str(AUD / "gmo_frameset.json"),
        "--class-weights", str(W), "--decision-rule", "prior_corrected",
        "--near-lift-m", "20", "--near-refine-blocks", "1",
        "--launch-commit", "NONBINDING-EARLY-2374cd2+NEW2R3",
        "--launch-argv-sha256", "NONBINDING-EARLY-no-launch-argv",
        "--out", str(OUT), "--device", "cuda"] + EXTRA
# ---- INFORMATIVE diagnostics of MAIN's FINAL logits (never gated): IoU by range from the
# rig and the 0.2 m tolerance P/R/F1 in the 0-20 m band -- the measurements that showed the
# NEW-2 arm's edge error was exact-cell placement (MAIN_long). Computed from the logits the
# harness itself keeps for C1-C3 (the healthy arm's final evaluate, keep_logits=True).
import numpy as np  # noqa: E402
from taniteval import map_hires_metrics as MET  # noqa: E402
from tanitad.models import map_head_hires as H  # noqa: E402

assert str(CODE) in MET.__file__, MET.__file__
DIAG = {}
_orig_eval = M.evaluate
_rows = torch.arange(200, dtype=torch.float64) * 0.1 + 0.05
_cols = -30.0 + (torch.arange(600, dtype=torch.float64) + 0.5) * 0.1
_rng = torch.sqrt(_rows[:, None] ** 2 + _cols[None, :] ** 2)
_BINS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 37)]


def _diag(kept, data, w, rule):
    codes = data["codes"]
    out = {}
    for a, b in _BINS:
        m = torch.zeros(1000, 600, dtype=torch.bool)
        m[:200] = (_rng >= a) & (_rng < b)
        tot, i0 = None, 0
        for lg, lv in kept:
            n = lg.shape[0]
            sig = H.per_class_signal(lg, codes[i0:i0 + n], class_weight=w, lift_valid=lv & m,
                                     decision_rule=rule)
            i0 += n
            part = {k: sig[k].cpu() for k in ("n", "inter", "union", "interraw", "unionraw")}
            tot = part if tot is None else {k: tot[k] + part[k] for k in part}
        out[f"{a}_{b}m"] = {H.CLASS_KEYS[c]: {
            "n": int(tot["n"][c, 0]),
            "iou": (float(tot["inter"][c, 0] / tot["union"][c, 0])
                    if float(tot["union"][c, 0]) else None),
            "iou_raw": (float(tot["interraw"][c, 0] / tot["unionraw"][c, 0])
                        if float(tot["unionraw"][c, 0]) else None)} for c in range(8)}
    stats, i0 = [], 0
    for lg, lv in kept:
        pred = H.decide(lg, rule, class_weight=w).numpy().astype(np.uint8)
        for j in range(lg.shape[0]):
            sc = lv[j].clone()
            sc[200:] = False
            stats.append(MET.window_stats(pred[j], codes[i0 + j].numpy(), valid=sc.numpy()))
        i0 += lg.shape[0]
    st = {k: np.stack([x[k] for x in stats]) for k in stats[0]}
    pl = MET.pooled(st)
    out["tolerance_0.2m_band0_20"] = {H.CLASS_KEYS[c]: {
        "P": float(pl["P"][c][0]), "R": float(pl["R"][c][0]), "F1": float(pl["F1"][c][0]),
        "iou": float(pl["iou"][c][0])} for c in MET.THIN_CLASSES}
    return out


def _evaluate(trunk, branch, data, class_weight, rule, device, batch, arm, keep_logits=False):
    tot, kept = _orig_eval(trunk, branch, data, class_weight, rule, device, batch, arm,
                           keep_logits=keep_logits)
    if keep_logits and arm == "healthy":
        try:
            DIAG["MAIN_final"] = _diag(kept, data, class_weight.detach().cpu(), rule)
            print(json.dumps({"diag": "MAIN_final",
                              "edge_tol": DIAG["MAIN_final"]["tolerance_0.2m_band0_20"]["edge"],
                              "lane_tol": DIAG["MAIN_final"]["tolerance_0.2m_band0_20"]["lane"]}),
                  flush=True)
        except Exception as exc:           # an informative extra never kills the arm
            DIAG["MAIN_final"] = {"error": f"{type(exc).__name__}: {exc}"}
    return tot, kept


M.evaluate = _evaluate
t0 = time.time()
torch.cuda.reset_peak_memory_stats()
rc = M.main(argv)
wall = time.time() - t0
src = OUT / "g_map_overfit.json"
rec = json.loads(src.read_text(encoding="utf-8"))
rec["binding"] = False
rec["why"] = WHY
rec["informative_diagnostics"] = DIAG          # never gated; see _diag
if SHARED:
    rec["gpu_shared_with"] = SHARED               # s/step and peak memory: informative only
rec["early_run"] = {
    "harness_exit_code": int(rc), "wall_s": round(wall, 1),
    "peak_cuda_max_memory_allocated_gib_last_arm": round(
        torch.cuda.max_memory_allocated() / 2 ** 30, 3),
    "harness_argv": argv, "spec_md5": A15_MD5,
    "candidate": {"base_commit": "2374cd2cc36465d73b467ea955f273d459f7b928",
                  "tree": str(CODE), "shipped_tar_md5": "c9dff6147e058dd537be23339464a532",
                  "per_file_md5_manifest": "MD5SUMS_r3.txt (2,853 files, all OK on Thor)"},
    "weights_json_sha256": hashlib.sha256(W.read_bytes()).hexdigest(),
    "runner": "gmo_r3_runner.py (stamps only; the harness is the candidate's, unmodified)"}
dst = OUT / "g_map_overfit_A15.EARLY_NONBINDING.json"
dst.write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")
src.unlink()
print(json.dumps({"stamped": str(dst), "binding": False, "harness_exit_code": rc,
                  "wall_s": round(wall, 1)}), flush=True)
