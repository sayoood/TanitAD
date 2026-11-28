"""CPU validation of pull_encode_pooled (RA-1a prep). The 4060 is NOT touched (CUDA_VISIBLE_DEVICES is blanked).

    python validate_cpu.py --out ../raw/validation.json [--v416 <416x1024 v2ep of the SAME eval clip>] [--frames 0,19,38,57,76,100]

Tests (every one prints its number; the harness exits non-zero if a PASS bar is missed):
  V0  split assertions   the real sets pass; FOUR deliberate-regression sets (an eval/train leak) must be REFUSED
  V1  pooling            pool_field == an independent avg_pool2d path (fp32); the fp8 pooled tensor is within the
                         e4m3 rounding bound |q - m| <= 2^-4 |m| + 2^-10 of that pool; one cell recomputed by hand
  V2  determinism        the same truncated clip encoded twice (bf16, CPU) gives bit-identical content_md5
  V3  eval-cache match   the full-field output for one EVAL clip vs the EXISTING refav1 eval cache entry
                         (cosine, rel-L2, max|d|, fp8-code equality) + two controls that must read WORSE
                         (shifted frame, other clip). CPU-bf16 vs the cache's CUDA-bf16 differ only by kernel numerics.
  V3b the 416 route      the same comparison for the thor416->256x640 resample route (only with --v416)
  V4  kinematics         kin3 bit-exact vs the banked probe `kin` on every eval window of all 141 clips, p10 == probe.py
                         kin_feats; `a`/`steer` == RefAV1Windows._kin_actions called on the same arrays
Frames are 6 grid frames spread over the clip (the ViT-L forward is ~13 s/frame in bf16 on this CPU).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"                     # the RTX 4060 belongs to another session. NOT "" -- an empty value leaves CUDA visible (MEASURED 2026-10-17)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pull_encode_pooled as PE  # noqa: E402

assert not torch.cuda.is_available(), "CUDA is visible: this harness is CPU-only by contract (set CUDA_VISIBLE_DEVICES=-1)"

FP8_REL, FP8_ABS = 2.0 ** -4, 2.0 ** -10                      # e4m3 round-to-nearest bound: 3 mantissa bits; min subnormal 2^-9
R: dict = {"tests": {}}
FAILS: list[str] = []


def rec(name, ok, **kw):
    R["tests"][name] = {"pass": bool(ok), **kw}
    if not ok:
        FAILS.append(name)
    print(f"[validate] {name}: {'PASS' if ok else 'FAIL'} {json.dumps(kw, default=float)[:600]}", flush=True)


def subset_episode(o: dict, grid_frames: list[int]) -> dict:
    """A v2ep dict holding only the raw frames 2j (j in grid_frames), laid out so encode_episode (which reads
    range(0, len(lens), 2)) sees exactly them. poses/actions are untouched (kinematics are not under test here)."""
    lens, buf = o["jpeg_len"], o["jpeg_buf"]
    offs = torch.cat([torch.zeros(1, dtype=lens.dtype), lens.cumsum(0)])
    chunks, nl = [], []
    for j in grid_frames:
        chunks.append(buf[offs[2 * j]:offs[2 * j + 1]])
        nl += [int(offs[2 * j + 1] - offs[2 * j]), 0]
    o2 = dict(o)
    o2["jpeg_buf"] = torch.cat(chunks)
    o2["jpeg_len"] = torch.tensor(nl, dtype=lens.dtype)
    return o2


def metrics(x: torch.Tensor, ref: torch.Tensor) -> dict:
    x, ref = x.float(), ref.float()
    cos = F.cosine_similarity(x.reshape(x.shape[0], -1), ref.reshape(ref.shape[0], -1), dim=1)
    d = (x - ref).abs()
    return {"cos_mean": float(cos.mean()), "cos_min": float(cos.min()),
            "rel_l2": float((x - ref).norm() / ref.norm()), "max_abs_diff": float(d.max()),
            "ref_absmax": float(ref.abs().max()), "mean_abs_diff": float(d.mean())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", default="0,19,38,57,76,100")
    ap.add_argument("--v416", default=None, help="the 416x1024 v2ep of the SAME eval clip (pulled from Thor)")
    ap.add_argument("--clip", default=None, help="eval clip id (default: first eval clip by sha12)")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--skip-encode", action="store_true", help="V0 + V4 only")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    FR = [int(x) for x in a.frames.split(",")]
    sa = SimpleNamespace(eval_eps=PE.D_EVAL_EPS, eval_cache=PE.D_EVAL_CACHE, eval_labels=PE.D_EVAL_LABELS,
                         train_labels=PE.D_TRAIN_LABELS, listing=PE.D_INDEX, thor=PE.THOR)
    t_all = time.time()

    # ------------------------------------------------------------------------------------------------ V0
    sets = PE.clip_sets(sa)
    rec("V0a_real_sets_pass", True, **{k: v for k, v in sets["checks"].items()})
    tmp = Path(tempfile.mkdtemp(prefix="v0_", dir=str(Path(PE.D_OUT) / "scratch")))
    train_id, eval_ids = sets["train_all"][0], sets["eval"]
    # N1: a TRAIN clip smuggled into the eval v2ep dir (142 names)
    d1 = tmp / "eps"; d1.mkdir()
    for c in eval_ids + [train_id]:
        (d1 / f"{c}.v2ep.pt").write_bytes(b"")
    # N2: the listing is missing one eval clip
    idx = json.load(open(PE.D_INDEX, encoding="utf-8"))["episodes"]
    (tmp / "idx2.json").write_text(json.dumps({"episodes": [c for c in idx if c != eval_ids[0]]}))
    # N3: the train label file lacks one train clip
    import gzip
    tr = PE.read_jsonl_gz(PE.D_TRAIN_LABELS)
    with gzip.open(tmp / "tr3.jsonl.gz", "wt", encoding="utf-8") as fh:
        for r in tr[1:]:
            fh.write(json.dumps(r) + "\n")
    # N4: the eval label file contains a train clip
    ev = PE.read_jsonl_gz(PE.D_EVAL_LABELS)
    with gzip.open(tmp / "ev4.jsonl.gz", "wt", encoding="utf-8") as fh:
        for r in ev + [tr[0]]:
            fh.write(json.dumps(r) + "\n")
    for name, over in (("N1_train_clip_in_eval_dir", {"eval_eps": str(d1)}),
                       ("N2_eval_clip_missing_from_listing", {"listing": str(tmp / "idx2.json")}),
                       ("N3_train_labels_incomplete", {"train_labels": str(tmp / "tr3.jsonl.gz")}),
                       ("N4_train_clip_in_eval_labels", {"eval_labels": str(tmp / "ev4.jsonl.gz")})):
        s2 = SimpleNamespace(**{**vars(sa), **over})
        try:
            PE.clip_sets(s2)
            refused = False
        except SystemExit as e:
            refused = "ZZSPLIT-REFUSED" in str(e)
        rec("V0_" + name + "_must_be_refused", refused)

    # ------------------------------------------------------------------------------------------------ V4 (cheap, first)
    ra_loader = None
    try:
        for p in (PE.REPO / "stack",):
            sys.path.insert(0, str(p))
        from tanitad.data.refav1_loader import RefAV1Windows
        ra_loader = object.__new__(RefAV1Windows)
        ra_loader.dt = 0.2
    except Exception as e:
        rec("V4_loader_import", False, err=repr(e)[:200])
    names = sorted(sets["eval"])                                   # episode_names(cache) order == sorted stems
    kin_max, p10_max, p10_rel, n_win, a_max, st_max, n_eps = 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0
    probe_dir = "C:/Users/Admin/refav1_probe/feat_s4"
    sys.path.insert(0, "C:/Users/Admin/refav1_probe")
    import probe as P                                              # the banked probe, for kin_feats
    for f in sorted(glob.glob(f"{probe_dir}/ep*.npz")):
        z = np.load(f)
        nm = names[int(z["eid"][0])]
        o = torch.load(Path(PE.D_EVAL_EPS) / f"{nm}.v2ep.pt", map_location="cpu", weights_only=False)
        kin = PE.kinematics(o)
        ws = z["ws"].astype(int)
        mine = kin["kin3"][ws].numpy()
        kin_max = max(kin_max, float(np.abs(mine - z["kin"]).max()))
        ref64 = P.kin_feats(z["kin"].astype(np.float64))
        p10_max = max(p10_max, float(np.abs(kin["p10"][ws].numpy() - ref64.astype(np.float32)).max()))     # bit-exact after the cast
        p10_rel = max(p10_rel, float((np.abs(kin["p10"][ws].numpy().astype(np.float64) - ref64) / np.maximum(1.0, np.abs(ref64))).max()))
        n_win += len(ws)
        n_eps += 1
        if ra_loader is not None:
            v_full, kap = o["poses"][:, 3].float(), o["actions"][:, 0].float()
            ak = ra_loader._kin_actions(v_full, kap, 0, kin["v"].shape[0])
            a_max = max(a_max, float((ak[:, 0] - kin["a"]).abs().max()))
            st_max = max(st_max, float((ak[:, 1] - kin["steer"]).abs().max()))
    rec("V4a_kin3_bit_exact_vs_banked_probe_kin", kin_max == 0.0 and n_win > 2000, n_episodes=n_eps, n_windows=n_win,
        max_abs_diff=kin_max)
    rec("V4b_p10_equals_probe_kin_feats", p10_max == 0.0 and p10_rel < 1e-6, max_abs_diff_after_f32_cast=p10_max,
        max_rel_diff_vs_f64=p10_rel, note="probe.py kin_feats on the banked kin, float64, cast to float32")
    rec("V4c_a_and_steer_equal_loader_kin_actions", ra_loader is not None and a_max == 0.0 and st_max == 0.0,
        max_abs_a=a_max, max_abs_steer=st_max, note="RefAV1Windows._kin_actions called on the same arrays")

    # V4d -- units sanity: yaw_rate must follow the kinematic bicycle v*tan(steer)/L with L = 2.9 (the encoder's wheelbase)
    xs, ys = [], []
    for nm in names[:12]:
        o = torch.load(Path(PE.D_EVAL_EPS) / f"{nm}.v2ep.pt", map_location="cpu", weights_only=False)
        k = PE.kinematics(o)
        xs.append((k["v"] * torch.tan(k["steer"]) / PE.L_ENC)[:-1].numpy())
        ys.append(k["yaw_rate"][:-1].numpy())
    x, y = np.concatenate(xs), np.concatenate(ys)
    corr, slope = float(np.corrcoef(x, y)[0, 1]), float(np.polyfit(x, y, 1)[0])
    rec("V4d_units_yaw_rate_vs_bicycle_model", corr > 0.99 and 0.9 < slope < 1.1, n=int(len(x)), corr=corr, slope=slope,
        note="yaw_rate [rad/s] ~ v*tan(steer)/2.9 => steer is a road-wheel ANGLE [rad], L_enc = 2.9 m (12 clips, informational units check)")

    if a.skip_encode:
        return finish(a, t_all)

    # ------------------------------------------------------------------------------------------------ V1/V2/V3
    cid = a.clip or sorted(sets["eval"], key=PE.sha12)[0]
    R["clip_sha12"] = PE.sha12(cid)
    enc = PE.load_encoder_module(PE.D_ENCODER)
    o256 = torch.load(Path(PE.D_EVAL_EPS) / f"{cid}.v2ep.pt", map_location="cpu", weights_only=False)
    sub = subset_episode(o256, FR)
    cache = torch.load(Path(PE.D_EVAL_CACHE) / f"{cid}.pt", map_location="cpu", weights_only=True)   # fp8 [101,640,1024]
    cache_fr = cache[FR]
    other = sorted(sets["eval"], key=PE.sha12)[1] if cid == sorted(sets["eval"], key=PE.sha12)[0] else sorted(sets["eval"], key=PE.sha12)[0]
    other_fr = torch.load(Path(PE.D_EVAL_CACHE) / f"{other}.pt", map_location="cpu", weights_only=True)[FR]

    t = time.time()
    model, mean, std, nsp = PE.build_model("cpu", "bf16")
    f1 = enc.encode_episode(sub, model, mean, std, nsp, "cpu")
    R["secs_bf16_cpu_per_frame"] = round((time.time() - t) / len(FR), 1)
    f2 = enc.encode_episode(sub, model, mean, std, nsp, "cpu")
    # V2
    p1 = PE.payload_from_field(f1, o256, ["4x10", "8x20"], torch.float8_e4m3fn)
    p2 = PE.payload_from_field(f2, o256, ["4x10", "8x20"], torch.float8_e4m3fn)
    rec("V2_reencode_bit_identical", bool(torch.equal(f1, f2)) and PE.content_md5(p1) == PE.content_md5(p2),
        field_max_abs_diff=float((f1 - f2).abs().max()), content_md5=PE.content_md5(p1), device="cpu bf16", n_frames=len(FR))

    # V1 -- an independent pooling path
    T = f1.shape[0]
    grid = f1.reshape(T, 16, 40, 1024).permute(0, 3, 1, 2)
    res1 = {}
    ok1 = True
    for g, (gh, gw) in PE.GRIDS.items():
        m = F.avg_pool2d(grid, kernel_size=(16 // gh, 40 // gw)).permute(0, 2, 3, 1).reshape(T, gh * gw, 1024)
        mine = PE.pool_field(f1, gh, gw)
        q = p1["pool" + g]
        err = (q.float() - m).abs()
        bound = FP8_REL * m.abs() + FP8_ABS
        viol = int((err > bound * (1 + 1e-6)).sum())
        # one cell by hand: cell (r=1, c=3) of the 4x10 grid = tokens rows 4..7, cols 12..15
        if g == "4x10":
            toks = [(4 + i) * 40 + (12 + j) for i in range(4) for j in range(4)]
            hand = f1[:, toks, :].mean(dim=1)
            hand_err = float((mine[:, 1 * 10 + 3] - hand).abs().max())
        else:
            hand_err = None
        res1[g] = {"fp32_pool_vs_avg_pool2d_max_abs": float((mine - m).abs().max()), "fp8_max_abs_err": float(err.max()),
                   "fp8_err_over_bound_max": float((err / bound).max()), "n_bound_violations": viol,
                   "hand_cell_max_abs": hand_err, "pooled_absmax": float(m.abs().max()), "shape": list(q.shape)}
        ok1 &= (res1[g]["fp32_pool_vs_avg_pool2d_max_abs"] < 1e-5 and viol == 0 and (hand_err is None or hand_err < 1e-6))
    # the patch mean = the mean of the 4x10 cells (equal-size cells), to fp16 rounding
    mean_vs_cells = float((p1["mean"].float() - PE.pool_field(f1, 4, 10).mean(dim=1)).abs().max())
    rec("V1_pooled_equals_manual_avg_pool_within_fp8_bound", ok1 and mean_vs_cells < 1e-2, tolerance="|q-m| <= 2^-4|m| + 2^-10",
        per_grid=res1, patch_mean_vs_mean_of_cells_max_abs=mean_vs_cells)

    # V3 -- the geometry is the one refav1 was trained on
    nat = metrics(f1, cache_fr)
    shifted = metrics(f1[:-1], cache[[j + 1 for j in FR[:-1]]])
    wrong_clip = metrics(f1, other_fr)
    code_eq = float((f1.to(torch.float8_e4m3fn).view(torch.uint8) == cache_fr.view(torch.uint8)).float().mean())
    pooled_cache = PE.pool_field(cache_fr.float(), 4, 10)
    pool_nat = metrics(p1["pool4x10"], pooled_cache)
    ok3 = nat["cos_min"] > 0.99 and nat["rel_l2"] < 0.10 and wrong_clip["rel_l2"] > 3 * nat["rel_l2"] and shifted["rel_l2"] > 2 * nat["rel_l2"]
    rec("V3_full_field_matches_refav1_eval_cache", ok3, frames=FR, vs_cache=nat, fp8_code_equal_fraction=code_eq,
        control_shifted_one_grid_frame=shifted, control_other_clip=wrong_clip, pooled4x10_vs_pool_of_cache=pool_nat,
        note="CPU-bf16 vs the cache's CUDA-bf16: same weights/dtype/pixels, different GEMM kernels")
    t = time.time()
    model32, mean32, std32, _ = PE.build_model("cpu", "fp32")
    f32 = enc.encode_episode(sub, model32, mean32, std32, nsp, "cpu")
    R["fp32_control_vs_cache"] = metrics(f32, cache_fr)
    R["bf16_vs_fp32_cpu"] = metrics(f1, f32)
    print("[validate] fp32 control", json.dumps(R["fp32_control_vs_cache"]), round(time.time() - t), "s", flush=True)

    # V3b -- the 416 -> 256x640 route (the 256x640 B1 source on Thor was deleted 2026-10-04..12)
    if a.v416:
        o416 = torch.load(a.v416, map_location="cpu", weights_only=False)
        same_meta = bool(torch.equal(o416["poses"], o256["poses"]) and torch.equal(o416["actions"], o256["actions"])
                         and o416["clip_id"] == o256["clip_id"])
        sub416 = PE.resize416_to_256(subset_episode(o416, FR))
        f416 = enc.encode_episode(sub416, model, mean, std, nsp, "cpu")
        r_cache = metrics(f416, cache_fr)
        r_nat = metrics(f416, f1)
        ok3b = r_cache["cos_min"] > 0.95
        rec("V3b_thor416_resize_route_vs_refav1_eval_cache", ok3b, poses_actions_identical_to_256_v2ep=same_meta,
            vs_cache=r_cache, vs_native256_cpu_bf16=r_nat, pooled4x10_vs_pool_of_cache=metrics(PE.pool_field(f416, 4, 10), pooled_cache),
            bar="cos_min > 0.95 (informational: the route is a documented fallback, not the refav1 pipeline)")
    return finish(a, t_all)


def finish(a, t_all) -> int:
    R["ok"] = not FAILS
    R["failed"] = FAILS
    R["wall_s"] = round(time.time() - t_all)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(R, indent=1, default=float), encoding="utf-8")
    print("ZZVALIDATE-OK" if not FAILS else f"ZZVALIDATE-FAIL {FAILS}", flush=True)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
