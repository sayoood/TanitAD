"""refcv7 NEW-2 -- the loss budget that sets ``--w-map-hires`` (the brief's item 5).

THE RULE, stated before the number: at ImageNet init (refcv7 trains from ImageNet
init, SPEC_REFCV7 §4) the 10 cm term enters the total with the SAME contribution as
the 0.5 m map term::

    w_map_hires = w_map * median_i L_map(i) / median_i L_hires(i)

over the measured windows ``i``, with ``w_map = 1.0`` (refcv6's launch value, kept
by refcv7). Parity at init, not at convergence: the two heads then start with equal
say in the shared trunk's gradient, and neither drowns the other before either has
learned anything.

WHAT IS RUN (CPU, no GPU; real data from the eval kit -- the only SAM3 GT on the dev
box; an init loss is a scale, not a skill, so no eval outcome is tuned on):
* the REAL trunk: ``resnet101.a1_in1k`` ImageNet weights (the local HF cache), K = 3,
  416 x 1024, frozen BN, the C26 equalisation (43 rows) refcv7's FIX-3 makes real;
* per window: the 3 frames of the window's CURRENT stack decoded from the v2ep
  payload, oldest -> newest; the label frame = the newest raw frame;
* the refcv6 0.5 m branch at init (``PerceptionBranch``, w_map > 0) on ``fmap_s16``
  through the clip's own stride-16 lift geometry, ``map_loss_row`` with the
  lift-valid narrowing (the trainer's default);
* the NEW-2 branch at init on the tap's ``fmap_s8`` through the clip's 0.25 m /
  stride-8 geometry, ``map_hires_loss_row`` with the lift-valid narrowing, BOTH
  unweighted and with the DRY-RUN class weights (the launch weights are Thor's).

Clip ids never printed: sha12 only.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--v2-dir", type=Path, required=True)
    ap.add_argument("--extrinsics", type=Path, required=True)
    ap.add_argument("--weights-json", type=Path, required=True,
                    help="class weights (the DRY RUN file; read without the loader's "
                         "dry-run refusal, since this is a scale measurement)")
    ap.add_argument("--n-clips", type=int, default=4)
    ap.add_argument("--frames-per-clip", type=int, default=2)
    ap.add_argument("--equalize-bottom-rows", type=int, default=43)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import tanitad
    sys.path.insert(0, str(Path(tanitad.__file__).resolve().parents[1] / "scripts"))
    import refc_v3_train as T                                      # noqa: E402
    import torchvision.io as tvio
    from tanitad.data import semantic_map_gt as G
    from tanitad.data import semantic_map_gt_fine as F
    from tanitad.data.v2_dataset import stable_episode_id
    from tanitad.models import map_head_hires as H
    from tanitad.models import refcv6_perception_branch as P
    from tanitad.models import timm_trunk as TT
    from tanitad.models.trunk_shapes import FRAME_416x1024

    torch.manual_seed(0)
    t0 = time.time()
    trunk = TT.TimmResNetTrunk(TT.TimmTrunkConfig(
        model_name="resnet101.a1_in1k", pretrained=True, frames=3,
        image_hw=(416, 1024), frozen_bn=True,
        equalize_bottom_rows=int(a.equalize_bottom_rows))).eval()
    tap = trunk.enable_s8_tap()
    pcfg = P.PerceptionBranchConfig(w_map=1.0, w_box3d=0.0)
    pbr = P.PerceptionBranch(pcfg, d_image=trunk.s16_dim, image_hw=trunk.s16_shape).eval()
    hcfg = H.MapHiresConfig(w_map_hires=1.0)
    hbr = H.MapHiresBranch(hcfg, d_image=trunk.s8_dim, image_hw=trunk.s8_shape).eval()
    wd = json.loads(a.weights_json.read_text(encoding="utf-8"))
    cw = torch.tensor(wd["weights"], dtype=torch.float32)
    _single, table = T._read_rig_extrinsics(str(a.extrinsics))
    by12 = {G.sha12(p.name[:-len(".v2ep.pt")]): p for p in a.v2_dir.glob("*.v2ep.pt")}
    gts = sorted(p for p in a.gt_dir.glob("*.sam3mapgt.npz")
                 if p.name[:12] in by12)[:a.n_clips]
    rows = []
    for gp in gts:
        s12 = gp.name[:12]
        v2 = by12[s12]
        cid = v2.name[:-len(".v2ep.pt")]
        if cid not in table:
            continue
        bank16 = P.LiftGeometryBank({cid: table[cid]}, frame=FRAME_416x1024, stride=16,
                                    equalize_bottom_rows=a.equalize_bottom_rows)
        bank8 = H.HiresLiftGeometryBank({cid: table[cid]}, frame=FRAME_416x1024,
                                        cfg=hcfg,
                                        equalize_bottom_rows=a.equalize_bottom_rows)
        eid = [int(stable_episode_id(cid))]
        g16, v16 = bank16.for_episodes(eid)
        g8, v8 = bank8.for_episodes(eid)
        coarse = G.open_path(gp, cid)
        fine = F.open_path_fine(gp, cid)
        d = torch.load(v2, map_location="cpu", weights_only=False)
        offs = torch.cat([torch.zeros(1, dtype=torch.int64),
                          torch.cumsum(d["jpeg_len"].to(torch.int64), 0)])
        T_ = int(d["jpeg_len"].shape[0])
        for f in np.linspace(40, T_ - 20, a.frames_per_clip).astype(int).tolist():
            ims = [tvio.decode_image(d["jpeg_buf"][int(offs[i]):int(offs[i + 1])],
                                     mode=tvio.ImageReadMode.RGB).float() / 255.0
                   for i in (f - 2, f - 1, f)]                    # oldest -> newest
            x = torch.cat(ims, 0)[None]                           # [1, 9, 416, 1024]
            with torch.no_grad():
                xn = trunk.normalise(x)
                s16, _s32, _p = trunk.forward_features(xn, already_normalised=True)
                s8 = trunk.s8_from_normalised(xn, [0])
                po = pbr(s16, g16, v16)
                mf = coarse.read([f])
                lm = P.map_loss_row(po["map_logits"], torch.from_numpy(mf.cart),
                                    torch.from_numpy(mf.seen),
                                    lift_valid=po["map_valid"])
                ho = hbr(s8, g8, v8)
                codes = torch.from_numpy(fine.read([f]).codes)
                lh_u = H.map_hires_loss_row(ho["map_hires_logits"], codes,
                                            lift_valid_025=ho["map_hires_lift_valid"],
                                            with_metrics=False)
                lh_w = H.map_hires_loss_row(ho["map_hires_logits"], codes, class_weight=cw,
                                            lift_valid_025=ho["map_hires_lift_valid"],
                                            with_metrics=False)
            rows.append({"clip_sha12": s12, "raw_frame": int(f),
                         "L_map_05": float(lm["loss"]),
                         "n_map_cells": lm["n_map_cells"],
                         "L_hires_unweighted": float(lh_u["loss"]),
                         "L_hires_weighted_dryrun": float(lh_w["loss"]),
                         "n_hires_cells": lh_u["n_map_hires_cells"]})
            print(json.dumps(rows[-1]), flush=True)
    med = lambda k: statistics.median(r[k] for r in rows)       # noqa: E731
    w_u = 1.0 * med("L_map_05") / med("L_hires_unweighted")
    w_w = 1.0 * med("L_map_05") / med("L_hires_weighted_dryrun")
    rec = {"rule": "w_map_hires = w_map * median L_map_05(init) / median L_hires(init), "
                   "w_map = 1.0",
           "n_windows": len(rows), "n_clips": len({r["clip_sha12"] for r in rows}),
           "median_L_map_05": med("L_map_05"),
           "median_L_hires_unweighted": med("L_hires_unweighted"),
           "median_L_hires_weighted_dryrun": med("L_hires_weighted_dryrun"),
           "reference_ln9": float(np.log(9.0)), "reference_ln8": float(np.log(8.0)),
           "w_map_hires_from_unweighted": w_u,
           "w_map_hires_from_weighted_dryrun": w_w,
           "trunk": {"model": "resnet101.a1_in1k", "pretrained": True, "frames": 3,
                     "image_hw": [416, 1024], "frozen_bn": True,
                     "equalize_bottom_rows": a.equalize_bottom_rows, "tap": tap},
           "class_weights_file": {"name": a.weights_json.name,
                                  "dry_run": bool(wd.get("dry_run"))},
           "data": "eval-kit SAM3 GT + eval v2ep payloads (the only GT on the dev box)",
           "rows": rows, "elapsed_s": round(time.time() - t0, 1)}
    a.out.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in rec.items() if k != "rows"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
