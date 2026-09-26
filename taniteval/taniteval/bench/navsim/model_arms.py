"""Our model's NavSim arms -> seam files — PROMOTED from E2 ``code/run_bridge.py`` (the per-token
loop is E2's, unchanged) with two additions the suite needs:

1. the DEVICE comes from the GPU-gap launcher (``ctx.gpu_device()``): CPU unless a gap exists;
   between scenes the loop polls ``should_back_off()`` and moves the model to CPU for the rest of
   the run the moment a training process appears or foreign GPU memory reaches the limit;
2. the checkpoint md5 check is against ``--ckpt-md5`` when the operator names one (E2 hard-coded
   refcv4b's).

Stage-1 tokens of a camera arm are DECLARED CV stand-ins (no stage-1 frames on warmup: E2, 0/192);
such an arm's two-stage EPDMS is UNDEFINED and the summary says so (``n_standin``).
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from . import bridge as B

MODEL_ARMS = {"A1": "A1_ego_cmd", "A2": "A2_vision_pure", "A3": "A3_ego_nocmd", "A4": "A4_blind_ego_cmd",
              "A1NT": "A1NT_ego_cmd_nearest", "A2NT": "A2NT_vision_pure_nearest"}
DEFAULT_BANKS = {"warmup_two_stage": "C:/Users/Admin/tanitad-wt/_s2build/navsim/corpus",
                 "navhard_two_stage": "C:/Users/Admin/tanitad-caches/navsim-frames-navhard-20260920"}

#: ⛔ THE FRAME CACHE IS BOUNDED, AND IT HAS TO BE (W7, MEASURED 2026-09-20 BEFORE THE navhard RUN).
#: The cache was an unbounded ``dict`` keyed by ``scene_token``. On warmup that is 204 scenes =
#: **0.41 GB** and invisible. On navhard it is **5,462 stage-2 tokens mapping to 5,462 DISTINCT
#: scene_tokens (max 1 token per scene, 0 consecutive repeats in the loop's own iteration order)**,
#: so the cache would have grown to **10.83 GB while never once being HIT**, on a box with **7.9 GB
#: available**. The failure would not have looked like a leak: the wrapper's RAM guard aborts an arm
#: below ``--ram-floor-mb``, which reports as RAM_GUARD_ABORT / retryable hours into the run.
#: ⭐ Same family as the ``e_trunk_pooling.py`` dense-window trap in CLAUDE.md — **price the host
#: tensor (N x W x tokens x d x dtype) against FREE RAM, not against total** — and it was invisible
#: on the smaller split, which is exactly how it survived E2.
#: A bound of 64 entries costs 130 MB and loses nothing: the hit rate within an arm is 0 by
#: construction on a 1:1 split, and a miss is a 2 MB ``np.load``.
FRAME_CACHE_MAX = 64


def _rss_gb() -> float:
    """This process's resident set, GB. 0.0 when psutil is unavailable — a missing probe must
    read as "unknown", never as "fine"."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2 ** 30
    except Exception:                                                    # noqa: BLE001
        return 0.0


def times_rel_t0(rec: dict) -> list:
    ts = [int(x) for x in rec["timestamps_us"]]
    return [(t - ts[-1]) / 1e6 for t in ts]


def declared_inputs(arm_key: str) -> list:
    spec = B.ARMS[arm_key]
    cams = {"ST": ["cameras: cam_l0+cam_f0+cam_r0 stitched -> 256x640 cylindrical, t0 frame (ST)"],
            "NT": ["cameras: cam_l0+cam_f0+cam_r0 stitched -> 256x640 cylindrical, nearest-time 2 Hz history (NT)"],
            "BLIND": ["cameras: NONE (constant grey; the deliberate frames-blind regression)"]}[spec["frames"]]
    return cams + list(spec["declared"])


def reuse_seam(name: str, e2_arm: str, src: Path, out_dir: Path, *, ckpt_md5: str, frame_tag: dict,
               n_stage1: int, n_stage2: int, log=print) -> dict:
    """Adopt a PREVIOUS run's model seam instead of recomputing it — or REFUSE.

    ⛔ WHY THIS EXISTS (W7, 2026-09-20). navhard's A1 inference is **3.12 h** and it completed; the
    run then died in the *scoring* step because the wrapper's RAM guard fired at
    ``available memory 1035 MB`` while sibling jobs held the box (the scorer's own RSS was 768 MB —
    it was not the consumer). Re-running inference to redo a step that never depended on it is the
    exact waste ``CLAUDE.md`` records for ``t1_eval.py``: *a 100 %-complete run with a missing last
    step*, whose fix is to **check for the banked dump BEFORE re-running anything**.

    ⛔ AND THE HAZARD IS STALENESS, SO THE GUARD IS AN IDENTITY CHECK, NOT A FILE-EXISTS CHECK. A
    seam silently reused from a different checkpoint, a different arm spec or a different frame
    geometry is a *plausible wrong number* — worse than no result. Every field below must match the
    CURRENT run's inputs, and any mismatch RAISES; nothing here falls back to recomputing quietly,
    because a silent fallback turns a refusal into a 3-hour surprise.
    """
    npz, man_p = Path(src) / f"{name}.npz", Path(src) / f"{name}.manifest.json"
    if not npz.exists() or not man_p.exists():
        raise B.RefusedInput(f"--reuse-seams: {npz.name} / {man_p.name} not both present in {src}")
    man = json.loads(man_p.read_text(encoding="utf-8"))
    want = {"e2_arm": e2_arm, "n_stage1": int(n_stage1), "n_stage2": int(n_stage2)}
    got = {k: man.get(k) for k in want}
    bad = [f"{k}: banked {got[k]!r} != this run {want[k]!r}" for k in want if got[k] != want[k]]
    b_md5 = (man.get("model") or {}).get("ckpt_md5")
    if b_md5 != ckpt_md5:
        bad.append(f"ckpt_md5: banked {b_md5} != this run {ckpt_md5}")
    # ⚠️ NORMALISE BEFORE COMPARING. `frame_tag_check` returns `bank_frame` as a TUPLE in-process
    # and JSON round-trips it to a LIST, so a raw `!=` refuses an IDENTICAL geometry — MEASURED
    # 2026-09-20 on the real banked seam, where the synthetic unit test could not see it because its
    # fixture was written as a list on both sides. ⭐ That is why the guard is exercised against the
    # REAL artifact as well: a comparison whose two sides come from different serialisation paths is
    # the same family as the units trap — equal values, unequal types, a confident wrong answer.
    def _norm(v):
        return [round(x, 6) if isinstance(x, float) else x for x in v] if isinstance(v, (list, tuple)) else v

    b_tag = _norm(((man.get("model") or {}).get("frame_tag") or {}).get("bank_frame"))
    w_tag = _norm(frame_tag.get("bank_frame"))
    if b_tag != w_tag:
        bad.append(f"frame_tag.bank_frame: banked {b_tag} != this run {w_tag}")
    if bad:
        raise B.RefusedInput(f"--reuse-seams REFUSED for {name}: " + " | ".join(bad))
    # ⭐ the SEAM ARTIFACT is re-read and re-counted here, never trusted from the manifest: the
    # manifest is the producer's report about the file, and the file is the thing being adopted.
    z = np.load(npz, allow_pickle=False)
    src_col = [str(x) for x in z["source"]]
    n_model = sum(1 for s in src_col if s != "cv_standin")
    n_standin = len(src_col) - n_model
    if len(src_col) != n_stage1 + n_stage2:
        raise B.RefusedInput(f"--reuse-seams REFUSED for {name}: seam has {len(src_col)} rows, "
                             f"this run needs {n_stage1 + n_stage2}")
    if (n_model, n_standin) != (man.get("n_model_rows"), man.get("n_cv_standin_rows")):
        raise B.RefusedInput(f"--reuse-seams REFUSED for {name}: seam counts ({n_model}, {n_standin}) "
                             f"!= manifest ({man.get('n_model_rows')}, {man.get('n_cv_standin_rows')})")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(npz, out_dir / f"{name}.npz")
    man["reused_from"] = {"dir": str(Path(src)).replace(os.sep, "/"),
                          "why": ("the model seam was computed in full by an earlier run whose SCORING "
                                  "was aborted by the RAM guard; inference never depended on scoring"),
                          "identity_checked": ["e2_arm", "n_stage1", "n_stage2", "ckpt_md5",
                                               "frame_tag.bank_frame", "seam row count", "seam source counts"]}
    man["reused"] = True
    B.json_dump(man, str(out_dir / f"{name}.manifest.json"))
    log(f"[navsim] model arm {name} ({e2_arm}): REUSED banked seam — {n_model} model rows, "
        f"{n_standin} CV stand-ins (identity verified, nothing recomputed)")
    return {k: man[k] for k in ("e2_arm", "n_stage1", "n_stage2", "n_model_rows", "n_cv_standin_rows",
                                "seconds", "device_at_end") if k in man} | {"reused": True}


def run_model_arms(*, arms: dict, doc: dict, ckpt: str, bank: str, out_dir: Path, gpu, threads: int = 6,
                   ckpt_md5: str | None = None, reuse: str | None = None, log=print) -> dict:
    """``arms`` = {suite name: E2 arm key}. Writes ``<out_dir>/<name>.npz`` + ``.manifest.json``.
    ``gpu`` = a :class:`taniteval.bench.gpu_gap.GpuGapLauncher` (already constructed).
    ``reuse`` = a previous run's ``raw/model`` dir whose seams are adopted after an IDENTITY check
    (see :func:`reuse_seam`); arms not found there are computed normally."""
    import torch
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    toks = doc["tokens"]
    s1 = sorted(t for t, r in toks.items() if r["stage"] == 1)
    s2 = sorted(t for t, r in toks.items() if r["stage"] == 2)
    bank_build = json.load(open(os.path.join(bank, "BUILD.json"), encoding="utf-8"))
    tag = B.frame_tag_check(bank_build)
    md5 = B.md5_file(ckpt)
    if ckpt_md5 and md5 != ckpt_md5:
        raise B.RefusedInput(f"checkpoint md5 {md5} != declared {ckpt_md5}")
    # ⭐ REUSE FIRST, and decide BEFORE touching the GPU gate or loading the model: when every arm is
    # adopted there is no inference at all, so acquiring a device and building a 1.29 GB model would
    # be pure cost. (MEASURED: adopting navhard's A1 seam turns 11,232.6 s into a file copy.)
    reused: dict = {}
    todo = dict(arms)
    if reuse:
        for name, arm in list(arms.items()):
            reused[name] = reuse_seam(name, arm, Path(reuse), out_dir, ckpt_md5=md5, frame_tag=tag,
                                      n_stage1=len(s1), n_stage2=len(s2), log=log)
            todo.pop(name)
    if not todo:
        return {"model": {"ckpt": ckpt, "ckpt_md5": md5, "frame_tag": tag,
                          "note": "no inference ran: every model arm was adopted from a banked seam"},
                "arms": reused, "frame_cache": {"note": "no frames were read"}}
    device = gpu.acquire()
    if device == "cpu":
        torch.set_num_threads(int(threads))
    fbank = B.FrameBank.open(bank)
    t_load = time.time()
    model, cfg, targs, prov, arm_mod = B.load_refcv4b(ckpt)         # built on CPU (E2)
    tr = arm_mod.trainer()
    steps, W = int(prov["decoder_steps"]), int(prov["window"])
    if device == "cuda":
        model = model.to("cuda")
        gpu.mark_own_usage()
    import pandas as pd
    blind_grey = int(round(float(np.mean(pd.read_parquet(os.path.join(bank, "frames_provenance.parquet")).mean_px))))
    model_rec = {"ckpt": ckpt, "ckpt_md5": md5, "step": prov["step"], "window_rows": W, "decoder_steps": steps,
                 "n_anchors": prov["n_anchors"], "horizons_steps": prov["horizons"],
                 "state_dict_load": prov["state_dict_load"], "load_s": round(time.time() - t_load, 1),
                 "torch": torch.__version__, "frame_tag": tag, "blind_grey_u8": blind_grey}

    def forward(rows, decl):
        nonlocal model, device
        if device == "cuda" and gpu.should_back_off():
            model = model.to("cpu")
            torch.cuda.empty_cache()
            torch.set_num_threads(int(threads))
            gpu.backed_off()
            device = "cpu"
        if device == "cuda":
            mi = B.model_inputs(rows, decl, device="cuda")
            fr = tr.frames_to_device(rows[None], "cuda")
            with torch.no_grad():
                out = model(fr, steps=steps, **mi["kwargs"])
            traj = out["traj"].float()[0].cpu().numpy().astype(np.float64)
            diag = {"sel_idx": int(out["sel_idx"][0])}
            return {"traj": traj, "diag": diag, "ego": mi["ego"], "nav": mi["nav"]}
        return B.run_model(model, tr, rows, decl, steps)

    from collections import OrderedDict
    frame_cache: "OrderedDict[str, tuple]" = OrderedDict()
    cache_stat = {"hits": 0, "misses": 0, "evictions": 0, "max_entries": int(FRAME_CACHE_MAX),
                  "bytes_per_entry": 4 * 256 * 640 * 3 + 256 * 640,
                  "why_bounded": ("MEASURED 2026-09-20: unbounded, navhard's 5,462 DISTINCT stage-2 "
                                  "scene_tokens would hold 10.83 GB on a box with 7.9 GB available, "
                                  "with a 0 % hit rate (1 token per scene)")}

    def frames_for(scene_token: str):
        if scene_token in frame_cache:
            cache_stat["hits"] += 1
            frame_cache.move_to_end(scene_token)
            return frame_cache[scene_token]
        cache_stat["misses"] += 1
        val = B.FrameBank.load_wide(fbank, scene_token)
        frame_cache[scene_token] = val
        while len(frame_cache) > FRAME_CACHE_MAX:
            frame_cache.popitem(last=False)
            cache_stat["evictions"] += 1
        return val

    result = {"model": model_rec, "arms": dict(reused), "frame_cache": cache_stat}
    for name, arm in todo.items():
        spec = B.ARMS[arm]
        t_arm = time.time()
        recs, fps, srcs, poses_all, knots_all, tok_list = [], [], [], [], [], []
        for tok in s1:
            r = toks[tok]
            if spec["frames"] == "BLIND":
                decl = B.declare(r["ego_statuses"], arm)
                rows = B.pack_frames(np.zeros((4, 256, 640, 3), np.uint8),
                                     B.slot_sources(times_rel_t0(r), "ST", W), blind_grey)
                res = forward(rows, decl)
                poses = B.knots_to_navsim(res["traj"])
                recs.append({"token": tok, "stage": 1, "source": "refcv4b", "declared_values": decl,
                             "ego_block": res["ego"], "nav": res["nav"], "frames": "BLIND(no pixels read)",
                             "knots": res["traj"].tolist(), "diag": res["diag"]})
                srcs.append("refcv4b")
                poses_all.append(poses.astype(np.float32))
                knots_all.append(res["traj"].astype(np.float32))
            else:
                recs.append({"token": tok, "stage": 1, "source": "cv_standin",
                             "why": ("no stage-1 camera frames for this split in the bank — the devkit "
                                     "ConstantVelocityAgent is the DECLARED stand-in so the official script can "
                                     "run; stage-1 rows of this arm are NOT the model")})
                srcs.append("cv_standin")
                poses_all.append(np.full((8, 3), np.nan, np.float32))
                knots_all.append(np.full((8, 2), np.nan, np.float32))
            fps.append(r["fingerprint"])
            tok_list.append(tok)
        for i, tok in enumerate(s2):
            r = toks[tok]
            decl = B.declare(r["ego_statuses"], arm)
            st = r["scene_token"]
            fr, mask, sha = frames_for(st)
            times = times_rel_t0(r)
            if spec["frames"] == "BLIND":
                src_idx = B.slot_sources(times, "ST", W)
                rows = B.pack_frames(fr, src_idx, blind_grey)
            else:
                src_idx = B.slot_sources(times, spec["frames"], W)
                rows = B.pack_frames(fr, src_idx)
            res = forward(rows, decl)
            poses = B.knots_to_navsim(res["traj"])
            recs.append({"token": tok, "stage": 2, "source": "refcv4b", "scene_token": st, "frame_sha16": sha,
                         "frames": spec["frames"], "slot_sources": src_idx,
                         "navsim_frame_times_s": [round(x, 4) for x in times], "declared_values": decl,
                         "ego_block": res["ego"], "nav": res["nav"], "knots": res["traj"].tolist(),
                         "poses": poses.tolist(), "diag": res["diag"]})
            fps.append(r["fingerprint"])
            srcs.append("refcv4b")
            poses_all.append(poses.astype(np.float32))
            knots_all.append(res["traj"].astype(np.float32))
            tok_list.append(tok)
            if (i + 1) % 100 == 0:
                # ⭐ RSS is printed because the bound above is a CLAIM until a number shows it held:
                # an unbounded cache reads as a monotone climb toward the 10.83 GB priced in
                # FRAME_CACHE_MAX's note, and the RAM guard would abort scoring hours later.
                log(f"  [{name}] {i + 1}/{len(s2)} stage-2 scenes, {(time.time() - t_arm) / (i + 1):.2f} s/scene, "
                    f"device={device}, rss={_rss_gb():.2f} GB, cache={len(frame_cache)}/{FRAME_CACHE_MAX} "
                    f"(hit {cache_stat['hits']} / miss {cache_stat['misses']})")
        np.savez(out_dir / f"{name}.npz", token=np.asarray(tok_list), fingerprint=np.asarray(fps),
                 source=np.asarray(srcs), poses=np.stack(poses_all), knots=np.stack(knots_all),
                 sampling=np.asarray([8, 0.5]), arm=np.asarray(arm))
        withheld = [f"{f}[t{k}]" for f in B.FIELDS for k in ("-3", "-2", "-1", "0")
                    if f"{f}[t0]" not in spec["declared"] or k != "0"]
        man = {"arm": name, "e2_arm": arm, "spec": spec, "declared_inputs": list(spec["declared"]),
               "withheld_egostatus_fields": withheld, "frames_construction": spec["frames"],
               "frames_variant": "wide (PHYSICALAI_WIDE120_256x640), sha-verified per scene",
               "conversion": B.knots_to_navsim.__doc__, "nav_map": {str(k): v for k, v in B.NAVSIM_CMD_TO_NAV_NAME.items()},
               "n_stage1": len(s1), "n_stage2": len(s2),
               "n_model_rows": int(sum(1 for s in srcs if s == "refcv4b")),
               "n_cv_standin_rows": int(sum(1 for s in srcs if s == "cv_standin")),
               "seconds": round(time.time() - t_arm, 1), "device_at_end": device, "model": model_rec,
               "frame_cache": dict(cache_stat), "rss_gb_at_end": round(_rss_gb(), 3), "rows": recs}
        B.json_dump(man, str(out_dir / f"{name}.manifest.json"))
        result["arms"][name] = {k: man[k] for k in ("e2_arm", "n_stage1", "n_stage2", "n_model_rows",
                                                    "n_cv_standin_rows", "seconds", "device_at_end",
                                                    "rss_gb_at_end")}
        log(f"[navsim] model arm {name} ({arm}): {man['n_model_rows']} model rows, {man['n_cv_standin_rows']} CV stand-ins, "
            f"{man['seconds']} s")
    gpu.stop()
    return result
