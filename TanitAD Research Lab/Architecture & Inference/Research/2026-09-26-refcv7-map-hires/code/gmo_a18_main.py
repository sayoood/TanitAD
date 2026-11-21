"""EARLY, NON-BINDING G-MAP-OVERFIT **MAIN only** under A18 (SPEC_REFCV7 §23, the PI's "Budget to
3,000 steps"; the Master Mind: "MAIN only, 3,000 steps, non-binding"). The registered A18 spec
(raw/gmo_spec_A18.json: the A17.1 spec with steps 3,000 and the decay from step 2,700), A15's
configuration (near lift 20 m + one near refine block, TRAIN sqrt_mf d70dec80).

Uses the candidate harness's OWN functions -- load_spec, class_weights_of, load_frames, run_arm
(the spec's steps and lr_decay are the harness's, not this runner's), controls, verdict -- with
the identical construction of its main(); the fingerprints are ASSERTED equal to A15's record
(the same init). MAIN-only: the must-fail arms are NOT run (they run in the binding run), so the
harness verdict's G_MAP_OVERFIT cannot PASS here and is not the reading -- MAIN's own verdict
(bars, CE ratio, presence, finite loss) and C1-C3 on MAIN's logits are. Informative extras at
steps 1,000 / 2,000 / 3,000: IoU by range and the 0.2 m tolerance P/R/F1 (the A15 runner's
_diag, verbatim).

Usage: gmo_a18_main.py <code dir> <out dir> <weights json>
"""
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

CODE, OUT, W = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path[:0] = [str(CODE / "stack"), str(CODE / "stack" / "scripts")]
import tanitad  # noqa: E402
import torch  # noqa: E402

assert str(CODE) in tanitad.__file__, tanitad.__file__
import map_hires_overfit as M  # noqa: E402
import numpy as np  # noqa: E402
from taniteval import map_hires_metrics as MET  # noqa: E402
from tanitad.data.semantic_map_gt_fine import MapExtent  # noqa: E402
from tanitad.models import map_head_hires as H  # noqa: E402
from tanitad.models import timm_trunk as TT  # noqa: E402
from tanitad.models.trunk_shapes import frame_for_width  # noqa: E402

assert str(CODE) in M.__file__ and str(CODE) in MET.__file__
assert hasattr(M, "lr_multiplier"), "the harness has no lr_decay (NEW-2 R5) -- wrong tree"
AUD = CODE / "TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-map-signal-audit/raw"
SPEC = CODE / ("TanitAD Research Lab/Architecture & Inference/Research/"
               "2026-09-26-refcv7-map-hires/raw/gmo_spec_A18.json")
SPEC_MD5 = "4eda0636f1a41a9b60b59ccb6a4afca3"
assert hashlib.md5(SPEC.read_bytes()).hexdigest() == SPEC_MD5, "the A18 spec is not as registered"
FRAMESET = AUD / "gmo_frameset.json"
W_SHA = "d70dec8087ed73ee6d4129b6fc6e0a97a350f826907462b6b251413ede488b67"
assert hashlib.sha256(W.read_bytes()).hexdigest() == W_SHA, "not the TRAIN sqrt_mf launch weights"
A15_REC = Path("/home/nvidia/nb2r3_2374/gmo_a15/g_map_overfit_A15.EARLY_NONBINDING.json")
a15 = json.loads(A15_REC.read_text(encoding="utf-8"))
DIAG_AT = {10: 1000, 20: 2000, 30: 3000}          # evaluate() call index -> step (eval_every 100)
BINS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 37)]


def utc():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


extent = MapExtent(100.0, 30.0)
spec = M.load_spec(SPEC, class_weights=str(W), decision_rule="prior_corrected",
                   band_keys=extent.band_keys)
assert int(spec["steps"]) == 3000 and int(spec["eval_every"]) == 100
assert spec["lr_decay"] == {"kind": "cosine_to_zero", "start_step": 2700}
assert float(spec["near_lift_m"]) == 20.0 and int(spec["near_refine_blocks"]) == 1
fs_md5 = hashlib.md5(FRAMESET.read_bytes()).hexdigest()
assert fs_md5 == spec["frameset_md5"], (fs_md5, spec["frameset_md5"])
M.lr_multiplier(spec)
tsp = spec.get("trunk") or {}
hw = tuple(tsp.get("image_hw", (416, 1024)))
eq = int(tsp.get("equalize_bottom_rows", 43))
torch.manual_seed(int(spec.get("seed", 0)))                     # == the harness main()'s order
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
                        near_lift_x_m=20.0, near_refine_blocks=1,
                        decision_rule=spec["thresholds"]["decision_rule"],
                        class_weights_sha256=cw_stamp.get("sha256") or "")
branch = H.MapHiresBranch(hcfg, d_image=trunk.s8_dim, image_hw=trunk.s8_shape)
fp = {"trunk_init_sha256": M._fingerprint(trunk), "branch_init_sha256": M._fingerprint(branch),
      "branch_init_sha256_without_near": M._fingerprint_without(branch, "near.", "near_refine."),
      "branch_init_sha256_without_near_refine": M._fingerprint_without(branch, "near_refine.")}
same = {k: fp[k] == a15["fingerprints"].get(k) for k in fp}
print(json.dumps({"utc": utc(), "fingerprints_equal_A15": same}), flush=True)
assert all(same.values()), f"the init is NOT the A15 arm's: {same}"
frame = frame_for_width(int(hw[1]), int(hw[0]))
data = M.load_frames(spec, Path("/home/nvidia/data/refcv6-b1-416x1024-train"),
                     Path("/home/nvidia/data/sam3_gt_v3"),
                     Path("/home/nvidia/data/refcv6_train_eval139_extrinsics.json"),
                     frame, hcfg, eq,
                     cam_ts_dir=Path("/home/nvidia/data/_b1stage416/r0/camera_front_wide"))
assert data["sha12"] == a15["frames_sha12"] and data["raw_frame"] == a15["raw_frames"]

_rows = torch.arange(200, dtype=torch.float64) * 0.1 + 0.05
_cols = -30.0 + (torch.arange(600, dtype=torch.float64) + 0.5) * 0.1
_rng = torch.sqrt(_rows[:, None] ** 2 + _cols[None, :] ** 2)


def _diag(kept, data, w, rule):                    # == gmo_r3_runner._diag (the A15 record's)
    codes = data["codes"]
    out = {}
    for a, b in BINS:
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


diag = {}
_orig = M.evaluate
_calls = {"k": 0}


def evaluate(trunk_, branch_, data_, class_weight, rule, device, batch, arm, keep_logits=False):
    k = _calls["k"]
    _calls["k"] += 1
    want = k in DIAG_AT
    tot, kept = _orig(trunk_, branch_, data_, class_weight, rule, device, batch, arm,
                      keep_logits=keep_logits or want)
    if want:
        try:
            diag[str(DIAG_AT[k])] = _diag(kept, data_, class_weight.detach().cpu(), rule)
            d = diag[str(DIAG_AT[k])]
            print(json.dumps({"arm": "A18_MAIN", "utc": utc(), "range_diag_step": DIAG_AT[k],
                              "edge_tol": d["tolerance_0.2m_band0_20"]["edge"],
                              "lane_tol": d["tolerance_0.2m_band0_20"]["lane"]}), flush=True)
        except Exception as exc:                       # an informative extra never kills the arm
            diag[str(DIAG_AT[k])] = {"error": f"{type(exc).__name__}: {exc}"}
    return tot, (kept if keep_logits else [])


M.evaluate = evaluate
t0, started = time.time(), utc()
torch.cuda.reset_peak_memory_stats()
res = M.run_arm("healthy", trunk, branch, data, spec, cw, "cuda")
ctrl = M.controls(data, res["_logits"], cw, spec["thresholds"]["decision_rule"],
                  spec["thresholds"].get("band", "0_20"), extent.band_keys)
v = M.verdict({"healthy": res}, spec, ctrl, data["axis_guard"])
res.pop("_logits", None)
rec = {"schema": "tanitad.g_map_overfit_record/1+MAIN_ONLY",
       "binding": False, "arm": "A18_MAIN",
       "why": ("early MAIN-only read of A18 (SPEC_REFCV7 §23: 3,000 steps, the decay from 2,700; "
               "A15's configuration) on candidate <tip 2ac0bfb + NEW-2 R5 blobs + the A18 spec>, "
               "code-identical to 37086c3 + R5 (37086c3 changed only SPEC_REFCV7.md); not the "
               "launch commit. The must-fail arms were NOT run (MAIN only, the Master Mind): the "
               "harness's G_MAP_OVERFIT field cannot PASS without them and is not the reading."),
       "launch_commit": "NONBINDING-EARLY-37086c3+NEW2R5+A18",
       "reading": {"MAIN_verdict": v["MAIN"]["verdict"], "MAIN": v["MAIN"],
                   "controls_reproduced": v["controls_reproduced"], "time_guard": v["time_guard"],
                   "must_fail_arms": "NOT RUN (MAIN-only early read)"},
       "verdict_harness_as_computed": v,
       "spec_md5": SPEC_MD5, "spec_sha256": hashlib.sha256(SPEC.read_bytes()).hexdigest(),
       "frameset_md5": fs_md5,
       "harness_sha256": hashlib.sha256(Path(M.__file__).read_bytes()).hexdigest(),
       "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       "class_weights": cw_stamp, "decision_rule": spec["thresholds"]["decision_rule"],
       "extent": extent.as_dict(), "branch_config": hcfg.as_dict(),
       "near_lift_m": 20.0, "near_refine_blocks": 1,
       "optimiser": {"name": "AdamW", "lr": float(spec.get("lr", 1e-3)), "betas": [0.9, 0.999],
                     "eps": 1e-8, "weight_decay": float(spec.get("weight_decay", 0.0)),
                     "batch": int(spec.get("batch", 4)), "steps": int(spec["steps"]),
                     "seed": int(spec.get("seed", 0)), "lr_decay": spec["lr_decay"]},
       "fingerprints": fp, "fingerprints_equal_A15_record": same, "tap": tap,
       "n_frames": len(data["sha12"]), "frames_sha12": data["sha12"],
       "raw_frames": data["raw_frame"], "axis_guard": data["axis_guard"],
       "range_bins_m": BINS, "informative_diagnostics": diag, "results": {"healthy": res},
       "started_utc": started, "ended_utc": utc(), "wall_s": round(time.time() - t0, 1),
       "peak_cuda_max_memory_allocated_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3),
       "candidate": {"base_commit": "2ac0bfb7248076133211c5da5fa21cf86bef9907",
                     "code_identical_to": "37086c3 + NEW-2 R5 blobs",
                     "shipped_tar_md5": "87cecca48cb7399058e4fb7a2c6c1326",
                     "per_file_md5_manifest": "MD5SUMS_r5.txt (2,953 files) + MD5SUMS_r5_post.txt"},
       "weights_json_sha256": W_SHA,
       "a15_record": {"path": str(A15_REC), "md5": hashlib.md5(A15_REC.read_bytes()).hexdigest()}}
OUT.mkdir(parents=True, exist_ok=True)
dst = OUT / "g_map_overfit_A18_MAIN.EARLY_NONBINDING.json"
dst.write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")
print(json.dumps({"written": str(dst), "MAIN": v["MAIN"]["verdict"], "bars": v["MAIN"]["bars"],
                  "controls_reproduced": v["controls_reproduced"], "wall_s": rec["wall_s"]}),
      flush=True)
