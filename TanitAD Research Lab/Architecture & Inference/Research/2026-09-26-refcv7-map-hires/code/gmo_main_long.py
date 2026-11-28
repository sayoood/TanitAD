"""INFORMATIVE diagnostic arm `MAIN_long` (the Master Mind, 2026-09-27 ~04:40): the G-MAP-OVERFIT
MAIN config run to 3,000 steps -- separates "slow" from "cannot" for lane and edge. It moves no
literal and cannot make MAIN pass: MAIN's FAIL at step 1,000 stands.

Uses the candidate harness's OWN functions (load_spec, class_weights_of, load_frames,
run_arm, summarise, per_class_signal) with the identical setup of its main(); the only
changes are `steps` = 3,000 in an in-memory COPY of the spec (the file is untouched) and a
wrapper around the harness's `evaluate` that, at steps 1,000 / 2,000 / 3,000, keeps the logits
and measures IoU by RANGE from the ego (both rules) inside the gated 0-20 m band -- the
measurement the lift-lever choice needs (stride-8 lateral resolution is ~0.33 m per column at
20 m on the 416 x 1024 cylinder, ~0.08 m at 5 m).

Usage: gmo_main_long.py <code dir> <out dir> <weights json> [<gpu_shared_with>]
"""
import hashlib
import json
import math
import sys
import time
from pathlib import Path

CODE, OUT, W = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
SHARED = sys.argv[4] if len(sys.argv) > 4 else ""
sys.path[:0] = [str(CODE / "stack"), str(CODE / "stack" / "scripts")]
import tanitad  # noqa: E402
import torch  # noqa: E402

assert str(CODE) in tanitad.__file__
import map_hires_overfit as M  # noqa: E402
from tanitad.data.semantic_map_gt_fine import MapExtent  # noqa: E402
from tanitad.models import map_head_hires as H  # noqa: E402
from tanitad.models import timm_trunk as TT  # noqa: E402
from tanitad.models.trunk_shapes import frame_for_width  # noqa: E402
sys.path.insert(0, "/home/nvidia/gmo_early_0327/code_t1/taniteval")   # scipy-free metrics
import numpy as np  # noqa: E402
from taniteval import map_hires_metrics as MET  # noqa: E402
assert "code_t1" in MET.__file__, MET.__file__

assert str(CODE) in M.__file__
AUD = CODE / "TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-map-signal-audit/raw"
STEPS_LONG = 3000
DIAG_AT = {10: 1000, 20: 2000, 30: 3000}          # evaluate() call index -> step
RANGE_BINS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 37)]

extent = MapExtent(100.0, 30.0)
spec = M.load_spec(AUD / "gmo_spec.json", class_weights=str(W),
                   decision_rule="prior_corrected", band_keys=extent.band_keys)
spec_long = json.loads(json.dumps(spec))
spec_long["steps"] = STEPS_LONG
tsp = spec.get("trunk") or {}
hw = tuple(tsp.get("image_hw", (416, 1024)))
eq = int(tsp.get("equalize_bottom_rows", 43))
torch.manual_seed(int(spec.get("seed", 0)))
trunk = TT.TimmResNetTrunk(TT.TimmTrunkConfig(
    model_name=str(tsp.get("model", "resnet101.a1_in1k")),
    pretrained=bool(tsp.get("pretrained", True)), frames=3, image_hw=hw,
    frozen_bn=True, equalize_bottom_rows=eq,
    chunk_ckpt=int(tsp.get("chunk_ckpt", 0) or 0), bf16=bool(tsp.get("bf16", False)),
    channels_last=bool(tsp.get("channels_last", False)),
    fold_bn=bool(tsp.get("fold_bn", False))))
tap = trunk.enable_s8_tap()
cw, cw_stamp = M.class_weights_of(spec, extent)
hcfg = H.MapHiresConfig(w_map_hires=1.0, x_max_m=100.0, y_half_m=30.0, grad_ckpt=True,
                        decision_rule=spec["thresholds"]["decision_rule"],
                        class_weights_sha256=cw_stamp.get("sha256") or "")
branch = H.MapHiresBranch(hcfg, d_image=trunk.s8_dim, image_hw=trunk.s8_shape)
fp = {"trunk_init_sha256": M._fingerprint(trunk), "branch_init_sha256": M._fingerprint(branch)}
frame = frame_for_width(int(hw[1]), int(hw[0]))
data = M.load_frames(spec, Path("/home/nvidia/data/refcv6-b1-416x1024-train"),
                     Path("/home/nvidia/data/sam3_gt_v3"),
                     Path("/home/nvidia/data/refcv6_train_eval139_extrinsics.json"),
                     frame, hcfg, eq,
                     cam_ts_dir=Path("/home/nvidia/data/_b1stage416/r0/camera_front_wide"))

# range of every 10 cm cell of the 0-20 m band from the ego (rig origin), metres
rows = torch.arange(200, dtype=torch.float64) * 0.1 + 0.05
cols = -30.0 + (torch.arange(600, dtype=torch.float64) + 0.5) * 0.1
rng = torch.sqrt(rows[:, None] ** 2 + cols[None, :] ** 2)          # [200, 600]
w_eval = cw.clone().to("cuda")
diag = {}
_orig = M.evaluate
_calls = {"k": 0}


def _range_iou(kept, rule_name):
    out = {}
    codes = data["codes"]
    for a, b in RANGE_BINS:
        m = torch.zeros(1000, 600, dtype=torch.bool)
        m[:200] = (rng >= a) & (rng < b)
        tot = None
        i0 = 0
        for lg, lv in kept:
            n = lg.shape[0]
            c = codes[i0:i0 + n]
            i0 += n
            sig = H.per_class_signal(lg, c, class_weight=w_eval.cpu(), lift_valid=lv & m,
                                     decision_rule=rule_name)
            part = {k: sig[k].cpu() for k in ("n", "inter", "union", "interraw", "unionraw")}
            tot = part if tot is None else {k: tot[k] + part[k] for k in part}
        res = {}
        for ci, name in enumerate(H.CLASS_KEYS):
            n = float(tot["n"][ci, 0])
            u, ur = float(tot["union"][ci, 0]), float(tot["unionraw"][ci, 0])
            res[name] = {"n": int(n),
                         "iou": float(tot["inter"][ci, 0]) / u if u > 0 else None,
                         "iou_raw": float(tot["interraw"][ci, 0]) / ur if ur > 0 else None}
        out[f"{a}_{b}m"] = res
    # the 0.2 m tolerance P / R / F1 (SPEC_REFCV7 §6.2 thin-class metric), pooled over the
    # 16 frames, full scored mask, per 20 m band -- band 0 is the gated 0-20 m
    stats = []
    codes = data["codes"]
    i0 = 0
    for lg, lv in kept:
        pred = H.decide(lg, rule_name, class_weight=w_eval.cpu()).numpy().astype(np.uint8)
        for j in range(lg.shape[0]):
            stats.append(MET.window_stats(pred[j], codes[i0 + j].numpy(),
                                          valid=lv[j].numpy()))
        i0 += lg.shape[0]
    st = {k: np.stack([x[k] for x in stats]) for k in stats[0]}
    pl = MET.pooled(st)
    out["tolerance_0.2m_band0_20"] = {
        name: {"P": float(pl["P"][ci][0]), "R": float(pl["R"][ci][0]),
               "F1": float(pl["F1"][ci][0]), "iou": float(pl["iou"][ci][0])}
        for ci, name in enumerate(H.CLASS_KEYS) if ci in MET.THIN_CLASSES}
    return out


def evaluate(trunk_, branch_, data_, class_weight, rule, device, batch, arm, keep_logits=False):
    k = _calls["k"]
    _calls["k"] += 1
    want = k in DIAG_AT
    tot, kept = _orig(trunk_, branch_, data_, class_weight, rule, device, batch, arm,
                      keep_logits=keep_logits or want)
    if k >= 1:
        sm = M.summarise(tot, "0_20", extent.band_keys)
        print(json.dumps({"arm": "MAIN_long", "eval_call": k, "step": min(100 * k, STEPS_LONG),
                          "iou": sm["iou"], "iou_raw": sm["iou_raw"]}), flush=True)
    if want:
        try:
            diag[str(DIAG_AT[k])] = _range_iou(kept, rule)
        except Exception as exc:                       # an informative extra never kills the arm
            import traceback
            diag[str(DIAG_AT[k])] = {"error": f"{type(exc).__name__}: {exc}",
                                     "trace": traceback.format_exc(limit=4)[-1500:]}
            print(json.dumps({"arm": "MAIN_long", "range_diag_error": diag[str(DIAG_AT[k])]}),
                  flush=True)
            return tot, (kept if keep_logits else [])
        print(json.dumps({"arm": "MAIN_long", "range_diag_step": DIAG_AT[k],
                          "lane": {b: v["lane"] for b, v in diag[str(DIAG_AT[k])].items()
                                   if b != "tolerance_0.2m_band0_20"},
                          "edge": {b: v["edge"] for b, v in diag[str(DIAG_AT[k])].items()
                                   if b != "tolerance_0.2m_band0_20"},
                          "tol_f1": diag[str(DIAG_AT[k])]["tolerance_0.2m_band0_20"]}),
              flush=True)
    return tot, (kept if keep_logits else [])


M.evaluate = evaluate
t0 = time.time()
torch.cuda.reset_peak_memory_stats()
res = M.run_arm("healthy", trunk, branch, data, spec_long, cw, "cuda")
res.pop("_logits", None)
rec = {"schema": "tanitad.g_map_overfit_record/1+informative",
       "arm": "MAIN_long", "informative": True, "binding": False,
       "verdict_unchanged": "MAIN FAIL at 1,000 stands",
       "why": "informative diagnostic (the Master Mind, 2026-09-27): separates slow from cannot; "
              "moves no literal; candidate <base b4a59b9 + NEW-2 blobs>, not the launch commit",
       "steps": STEPS_LONG, "prereg_steps": int(spec["steps"]),
       "spec_sha256": hashlib.sha256((AUD / "gmo_spec.json").read_bytes()).hexdigest(),
       "frameset_md5": spec.get("frameset_md5"),
       "harness_sha256": hashlib.sha256(Path(M.__file__).read_bytes()).hexdigest(),
       "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       "class_weights": cw_stamp, "decision_rule": spec["thresholds"]["decision_rule"],
       "extent": extent.as_dict(), "fingerprints": fp, "tap": tap,
       "range_bins_m": RANGE_BINS, "range_diag": diag, "result": res,
       "wall_s": round(time.time() - t0, 1),
       "peak_cuda_max_memory_allocated_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)}
if SHARED:
    rec["gpu_shared_with"] = SHARED
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "g_map_overfit_MAIN_long.INFORMATIVE.json").write_text(
    json.dumps(rec, indent=1, default=float), encoding="utf-8")
print(json.dumps({"written": str(OUT / "g_map_overfit_MAIN_long.INFORMATIVE.json"),
                  "wall_s": rec["wall_s"]}), flush=True)
