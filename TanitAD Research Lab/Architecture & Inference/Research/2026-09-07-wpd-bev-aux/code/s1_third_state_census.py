"""WP-D s1 — THE THIRD-STATE CENSUS. What does masking occlusion actually cost?

⛔ This is the measurement the WP-D brief demands before the third state is
decided by argument: *"derive visibility/occlusion from the cuboids + ego pose,
or mask occluded cells out of the loss, or model three classes. State the cost of
each and pick one."* The cost of masking is (a) how much of the grid stops being
supervised and (b) how much POSITIVE signal is deleted, and both are numbers, not
opinions.

⛔ CPU ONLY. No GPU, no network, no HuggingFace pull. Zero load on the A40
(training refcv5-v2) and zero on Thor (training refav1).

INPUT — named, with its md5, because an artifact's cost is the cost of the file
its CONSUMER opens:
  ``C:/Users/Admin/tanitad-caches/b1-agent-join-20260906/b1eval_agents.jsonl.xz``
  the B1 EVAL ``obstacle.offline`` join (139 clips / 26,394 labelled frames /
  905,512 boxes). Read here through ``lzma`` + ``json`` DIRECTLY rather than
  through ``train_p8_occupancy.JoinFileReader``, deliberately: a second read path
  is a second probe, and the schema it must satisfy is one documented line
  (``build_obstacle_join.py`` LINE SCHEMA).

⚠️ POISONED-BANK GUARD: every accumulator is asserted NON-ZERO and its mean is
printed. A decode that fails into a pre-allocated buffer leaves a full-size bank
of zeros and the job still exits 0; an all-zero census would read as "occlusion
costs nothing" — a silent false negative.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import sys
import time
from pathlib import Path

import numpy as np


def md5_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--join", required=True)
    ap.add_argument("--stack", required=True)
    ap.add_argument("--every", type=int, default=3, help="record stride")
    ap.add_argument("--n-az", type=int, default=20)
    ap.add_argument("--n-rng", type=int, default=24)
    ap.add_argument("--r-max", type=float, default=60.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sys.path.insert(0, a.stack)
    from tanitad.data import bev_aux as BA
    from tanitad.data import bev_raster as BR

    spec = BA.PolarBEVSpec(n_az=a.n_az, n_rng=a.n_rng, r_max_m=a.r_max)
    join = Path(a.join)
    print(f"[s1] join   {join}", flush=True)
    t_md5 = time.time()
    md5 = md5_of(join)
    print(f"[s1] md5    {md5}  ({time.time() - t_md5:.1f} s)", flush=True)
    print(f"[s1] spec   {spec.to_dict()}", flush=True)

    # accumulators — polar
    n_rec = n_used = n_clear = 0
    n_boxes = 0
    tot_cells = tot_sup = tot_pos_sup = tot_pos_all = tot_shadow = 0
    per_frame_shadow = []
    per_frame_pos = []
    clips = set()
    # accumulators — Cartesian control (a DIFFERENT geometry, reported as such)
    cart_pos = cart_infield = 0
    cart_mask = BR.fov_mask(BR.GRID_DEFAULT)
    n_cart = 0

    t0 = time.time()
    with lzma.open(join, "rt", encoding="utf-8") as fh:
        for line in fh:
            n_rec += 1
            if (n_rec - 1) % a.every:
                continue
            rec = json.loads(line)
            clips.add(rec["clip_id"])
            ags = rec["agents"]
            n_boxes += len(ags)
            if not ags:
                n_clear += 1
            occ, mask = BA.build_target(ags, labelled=True, spec=spec,
                                        occlusion="mask")
            c = BA.target_census(occ, mask)
            tot_cells += c["n_cells"]
            tot_sup += c["n_supervised"]
            tot_pos_sup += c["n_pos_supervised"]
            tot_pos_all += c["n_pos_all"]
            tot_shadow += c["n_ignored"]
            per_frame_shadow.append(c["n_ignored"] / c["n_cells"])
            per_frame_pos.append(c["n_pos_all"] / c["n_cells"])
            n_used += 1
            if n_used % 500 == 0:
                # Cartesian control on every 500th used frame only — it is
                # 7,680 cells at 0.5 m and 16x the cost of the polar target.
                r = BR.rasterize(ags)
                cart_pos += int((r[cart_mask] > 0).sum())
                cart_infield += int(cart_mask.sum())
                n_cart += 1
                print(f"[s1] {n_used} frames  {time.time() - t0:.0f}s  "
                      f"shadow {tot_shadow / max(tot_cells, 1):.4f}  "
                      f"pos_sup {tot_pos_sup / max(tot_sup, 1):.5f}", flush=True)

    # ⚠️ CONTENT ASSERTIONS — the poisoned-bank guard. Each of these would be
    # satisfied by an all-zero bank, and an all-zero bank reads as "occlusion is
    # free", which is exactly the false negative that must not ship silently.
    assert n_used > 0, "no records read — the join did not open"
    assert n_boxes > 0, "zero boxes over the whole census — POISONED BANK"
    assert tot_pos_all > 0, "zero occupied cells — POISONED BANK"
    assert tot_shadow > 0, "zero shadow cells — the occlusion derivation is dead"

    shadow_frac = tot_shadow / tot_cells
    base_sup = tot_pos_sup / tot_sup
    base_all = tot_pos_all / tot_cells
    hidden = tot_pos_all - tot_pos_sup
    res = {
        "join_path": str(join),
        "join_md5": md5,
        "spec": spec.to_dict(),
        "record_stride": a.every,
        "n_records_in_file": n_rec,
        "n_records_censused": n_used,
        "n_clips_seen": len(clips),
        "n_boxes_censused": n_boxes,
        "n_frames_labelled_clear": n_clear,
        "frac_frames_labelled_clear": n_clear / n_used,
        "boxes_per_frame_mean": n_boxes / n_used,
        # --- the third-state cost, which is what this script exists for ------
        "n_cells_total": tot_cells,
        "n_cells_supervised": tot_sup,
        "n_cells_shadowed": tot_shadow,
        "frac_cells_shadowed": shadow_frac,
        "frac_cells_shadowed_per_frame_mean": float(np.mean(per_frame_shadow)),
        "frac_cells_shadowed_per_frame_median":
            float(np.median(per_frame_shadow)),
        "frac_cells_shadowed_per_frame_p90":
            float(np.percentile(per_frame_shadow, 90)),
        "n_pos_all": tot_pos_all,
        "n_pos_supervised": tot_pos_sup,
        "n_pos_hidden_by_mask": hidden,
        "frac_pos_hidden_by_mask": hidden / tot_pos_all,
        # --- the LITERALS the controls must read ------------------------------
        "base_rate_supervised": base_sup,
        "base_rate_all_cells_two_state": base_all,
        "all_zero_accuracy_supervised": 1.0 - base_sup,
        "suggested_pos_weight": (1.0 - base_sup) / base_sup,
        # --- Cartesian control (DIFFERENT geometry — not a replication) -------
        "cart_frames": n_cart,
        "cart_base_rate_infield": (cart_pos / cart_infield) if cart_infield
        else None,
        "elapsed_s": time.time() - t0,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    for k, v in res.items():
        print(f"  {k:42s} {v}")
    print(f"[s1] wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
