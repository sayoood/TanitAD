"""Scratch PLUMBING smoke (not a training run): the REAL train() of the b4a59b9 + NEW-2
candidate, 2 steps on CPU, --smoke model, a REAL v2 cache + the REAL /3 GT, so the data stage
(manifest -> window->clip table -> FineMapGTStore @ 100 x 30 -> coverage census -> the fine
target per window -> collate) and D3's first-row check (ga_mh_* declared) run on real files.

The cache is the EVAL139 view: the only one on this box whose clips have /3 GT. Nothing is
kept: the output directory is scratch and its checkpoint is deleted. Clip ids never printed.
Uniform class weights in a scratch JSON (a plumbing smoke, not the launch weights).
Usage: smoke_train_real.py <tree> <out dir>
"""
import json
import sys
import time
import traceback
from pathlib import Path

TREE, OUT = Path(sys.argv[1]), Path(sys.argv[2])
sys.path[:0] = [str(TREE / "stack"), str(TREE / "stack" / "scripts"), str(TREE / "taniteval")]
import tanitad  # noqa: E402

assert str(TREE) in tanitad.__file__, tanitad.__file__
import refc_v3_train as T  # noqa: E402
from tanitad.data.semantic_map_gt_fine import FINE_CLASSES  # noqa: E402
from tanitad.models import map_head_hires as H  # noqa: E402

OUT.mkdir(parents=True, exist_ok=False)
w = OUT / "weights_uniform_SMOKE_ONLY.json"
w.write_bytes(json.dumps({"schema": H.CLASS_WEIGHT_SCHEMA, "classes": list(FINE_CLASSES),
                          "weights": [1.0] * 8, "dry_run": False,
                          "definition_id": "sqrt_mf", "pre_registered": True,
                          "extent": {"x_max_m": 100.0, "y_half_m": 30.0}}).encode("utf-8"))
K = "D:/refcv6_eval_kit/data"
argv = ["--arm", "hier", "--smoke", "--out", str(OUT / "run"), "--steps", "2", "--batch", "1",
        "--device", "cpu", "--save-every", "100", "--log-every", "1", "--workers", "0",
        "--trunk", "timm", "--trunk-name", "resnet18.a1_in1k", "--no-trunk-pretrained",
        "--trunk-mode", "shared", "--trunk-fuse", "concat1x1", "--trunk-frozen-bn", "--trunk-in-channels", "9",
        "--trunk-chunk-ckpt", "2", "--image-hw", "416", "1024",
        "--equalize-bottom-rows", "43", "--u8-batches",
        "--v2-cache", f"{K}/refcv6-b1-416x1024-eval139", "--v2-lru", "1",
        "--allow-eval-clips-in-train",   # BY NAME: a 2-step plumbing smoke, nothing kept
        "--map-hires", "on", "--w-map-hires", "1.0", "--map-hires-class-weights", str(w),
        "--map-gt-root", f"{K}/sam3_gt_v3_eval",
        "--agent-rig-camera", "extrinsics",
        "--agent-rig-extrinsics", f"{K}/refcv6_train_eval139_extrinsics.json",
        "--bev-source", "map_hires_pool", "--bev-coupling"]
t0 = time.time()
status = "COMPLETED"
try:
    T.train(T.build_parser().parse_args(argv))
except SystemExit as e:
    status = "SystemExit: " + str(e)[:1500]
except Exception:
    status = "EXCEPTION:\n" + traceback.format_exc(limit=6)[-3000:]
res = {"status": status, "wall_s": round(time.time() - t0, 1)}
run = OUT / "run"
cfg_p, met_p = run / "config.json", run / "metrics.jsonl"
if cfg_p.is_file():
    cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
    res["grad_reach_logging"] = cfg.get("grad_reach_logging")
    res["config_map_hires_keys"] = sorted(k for k in cfg if "map_hires" in k)
if met_p.is_file():
    rows = [json.loads(x) for x in met_p.read_text(encoding="utf-8").splitlines() if x.strip()]
    res["n_rows"] = len(rows)
    if rows:
        r0 = rows[0]
        res["row0_ga_keys"] = sorted(k for k in r0 if k.startswith("ga_"))
        res["row0_map_hires"] = {k: r0[k] for k in ("map_hires", "n_map_hires_cells",
                                                    "n_map_hires_cells_seen") if k in r0}
        res["row0_n_map_hires_keys"] = sum(1 for k in r0 if k.startswith("map_hires"))
        res["row0_ga_mh"] = {k: r0[k] for k in r0 if k.startswith("ga_mh_")}
for p in run.glob("*.pt"):
    p.unlink()                                  # nothing trained here is kept
print(json.dumps(res, indent=1, default=str))
