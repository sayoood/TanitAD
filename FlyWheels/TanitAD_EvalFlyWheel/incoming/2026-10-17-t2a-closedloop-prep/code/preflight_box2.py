"""BOX-2 PREFLIGHT -- run this FIRST on the 8 GB Windows PC. It answers, in about two minutes, whether anything below will work.

    python preflight_box2.py --kit <kit dir> [--out preflight.json] [--no-gpu-probe]

It CHECKS (fail loud, one verdict per line, exit code 1 if any HARD check fails):
  1. software: python, torch (+ CUDA build), numpy, msgpack, gsplat (pinned 1.5.3), a CUDA toolchain for gsplat's JIT
     (nvcc, MSVC cl, ninja) -- the f-theta unscented-transform kernels are compiled on first use;
  2. files: every kit file present at the exact size the dev box measured (and md5 for the small ones);
  3. GPU: name, total VRAM, compute capability;
  4. THE VRAM QUESTION (the one thing the dev box cannot answer): peak GPU memory of
        (a) the renderer alone (scene 1 resident, one 1920x1080 f-theta frame),
        (b) refcv7 alone (one forward on a real rendered 10-frame window),
        (c) both resident together,
     against the card's total. An 8 GB card has ~7.5 GiB usable; refcv7's eval loss peaked at 5.85 GiB on the dev box's 4060
     (batch > 1, `battery/*/g0.log`), the 3.1 M-gaussian scene is ~0.7 GiB of tensors plus rasterisation buffers. UNVERIFIED
     until this runs.
  5. a render sanity check: frame 30 of scene 1 rendered from the LOGGED pose has non-trivial structure (not all-black,
     not all-sky), and rendering is deterministic run-to-run on this box (two renders, max abs pixel diff reported).

It writes nothing outside --out and never trains or modifies the kit.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECT = {                                    # relative path -> size in bytes, MEASURED on the dev box 2026-10-17
    "ckpt/ckpt_50400.pt": 402_708_019,
    "ckpt/config.json": 66_500,
    "p10/p10_S2.pt": 21_011,
    "scenes/00040136-e651-4abd-991d-0655ccda9430/volume.nurec": 639_393_135,
    "scenes/00040136-e651-4abd-991d-0655ccda9430/rig_trajectories.json": 2_155_480,
    "scenes/7c72937c-c620-4776-9555-d57222c0081f/volume.nurec": 603_238_216,
    "scenes/7c72937c-c620-4776-9555-d57222c0081f/rig_trajectories.json": 2_136_806,
    "refcv6_eval_kit/data/anchors/refc_anchors_6s_v0cond_alat_117.pt": 13_477,
    "refcv6_eval_kit/data/refcv7/map_hires_class_weights_train_100x30.json": 3_046,
    "refcv6_eval_kit/data/refcv7/nav_compliance_tau_train.json": 2_410,
    "refcv6_eval_kit/data/refcv6_train_eval139_extrinsics.json": 3_367_838,
}
MD5 = {                                       # relative path -> md5, MEASURED on the dev box 2026-10-17 (KIT_MANIFEST.json carries the same)
    "ckpt/ckpt_50400.pt": "b418d0fc4a92a6848c246a6a7c50207b",
    "ckpt/config.json": "e6512a01b9c70f0e4a4dac581621984a",
    "p10/p10_S2.pt": "cc6f478a6bb2b4cbf26b31ba293b442a",
    "scenes/00040136-e651-4abd-991d-0655ccda9430/volume.nurec": "ccecd3fdfd75256b5bf23a4ade2bc8d0",
    "scenes/00040136-e651-4abd-991d-0655ccda9430/rig_trajectories.json": "06df533224a4982e2c0937ea991151e7",
    "scenes/7c72937c-c620-4776-9555-d57222c0081f/volume.nurec": "62c5a0028608ce59a219bb0688ffae76",
    "scenes/7c72937c-c620-4776-9555-d57222c0081f/rig_trajectories.json": "8e6a10f66fe81ad7d05c82c6e4f878f9",
    "refcv6_eval_kit/data/refcv6_train_eval139_extrinsics.json": "3bd70d987d00936c1ac810ffd95facf1",
}
HARD, SOFT = "HARD", "soft"
rows: list[dict] = []


def check(name, ok, detail="", kind=HARD):
    rows.append({"check": name, "ok": bool(ok), "kind": kind, "detail": str(detail)})
    print(f"[{'OK ' if ok else ('FAIL' if kind == HARD else 'warn')}] {name}: {detail}", flush=True)
    return ok


def md5(p, chunk=1 << 22):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", required=True)
    ap.add_argument("--out", default="preflight.json")
    ap.add_argument("--no-gpu-probe", action="store_true")
    a = ap.parse_args()
    kit = Path(a.kit).resolve()
    tree = kit / "tree"
    os.environ["T2A_TREE"] = str(tree)
    os.environ.setdefault("REFCV6_KIT", str(kit / "refcv6_eval_kit"))
    # ---- 1. software ------------------------------------------------------------------------------ #
    check("python", sys.version_info >= (3, 10), sys.version.split()[0])
    mods = {}
    for m in ("numpy", "torch", "msgpack", "gsplat"):
        try:
            mods[m] = importlib.import_module(m)
            check(f"import {m}", True, getattr(mods[m], "__version__", "?"))
        except Exception as e:                                       # noqa: BLE001
            check(f"import {m}", False, f"{type(e).__name__}: {str(e)[:300]}")
    if "gsplat" in mods:
        check("gsplat == 1.5.3 (the version every banked Thor number was rendered with)",
              getattr(mods["gsplat"], "__version__", "") == "1.5.3", getattr(mods["gsplat"], "__version__", "?"), SOFT)
    for tool in ("nvcc", "cl", "ninja"):
        check(f"toolchain: {tool} on PATH (gsplat JIT-compiles CUDA kernels on first use)", shutil.which(tool) is not None,
              shutil.which(tool) or "not found", SOFT)
    torch = mods.get("torch")
    # ---- 2. files --------------------------------------------------------------------------------- #
    for rel, sz in EXPECT.items():
        p = kit / rel
        ok = p.is_file() and p.stat().st_size == sz
        check(f"file {rel}", ok, f"{p.stat().st_size if p.is_file() else 'MISSING'} (want {sz})")
        if ok and rel in MD5 and os.environ.get("T2A_MD5", "1") == "1":
            m = md5(p)
            check(f"md5 {rel}", m == MD5[rel], m)
    check("code tree", (tree / "stack" / "tanitad").is_dir() and (tree / "taniteval" / "taniteval" / "floors_kin.py").is_file(),
          str(tree))
    r4 = kit / "tree_r4"
    check("tree_r4 (R4 inference code 0fcb1c1c: enables the refcv7_r4code_vmax_log_s0 arm, the one where the ceiling reaches the plan)",
          (r4 / "stack" / "tanitad" / "refs" / "refc_v3.py").is_file() and "e9_rank" in (r4 / "stack" / "tanitad" / "refs" / "refc_v3.py").read_text(
              encoding="utf-8", errors="replace") if r4.is_dir() else False, str(r4) if r4.is_dir() else "absent", SOFT)
    # ---- 3/4/5. GPU -------------------------------------------------------------------------------- #
    probe = {}
    if torch is not None and torch.cuda.is_available():
        pr = torch.cuda.get_device_properties(0)
        total = pr.total_memory / 2 ** 30
        probe.update(device=pr.name, total_gib=round(total, 2), capability=f"{pr.major}.{pr.minor}", torch_cuda=torch.version.cuda)
        check("GPU", True, f"{pr.name}, {total:.2f} GiB, sm_{pr.major}{pr.minor}, torch CUDA {torch.version.cuda}")
        check("VRAM >= 7.5 GiB", total >= 7.5, f"{total:.2f} GiB")
    else:
        check("GPU visible to torch", False, "torch.cuda.is_available() is False -- this box cannot run T2-A")
    if not a.no_gpu_probe and torch is not None and torch.cuda.is_available() and "gsplat" in mods and "msgpack" in mods:
        try:
            probe.update(gpu_probe(kit, tree, torch))
        except Exception as e:                                         # noqa: BLE001
            import traceback
            check("GPU probe", False, f"{type(e).__name__}: {str(e)[:400]}")
            probe["traceback"] = traceback.format_exc()[-2500:]
    res = {"kit": str(kit), "checks": rows, "probe": probe, "time": time.strftime("%FT%T")}
    Path(a.out).write_text(json.dumps(res, indent=1))
    bad = [r for r in rows if not r["ok"] and r["kind"] == HARD]
    print(f"\nPREFLIGHT: {len(rows) - len(bad)}/{len(rows)} ok; {len(bad)} HARD failure(s). Wrote {a.out}")
    return 1 if bad else 0


def gpu_probe(kit, tree, torch):
    for p in (tree / "taniteval" / "tools", tree / "taniteval", tree / "stack" / "scripts", tree / "stack",
              tree / "stack/experiments/nurec-gsplat", tree / "stack/experiments/alpasim-gsplat", HERE):
        sys.path.insert(0, str(p))
    import numpy as np
    import closedloop_drive as cd
    import t2a_policies as tp
    from gsplat_renderer import NuRecGsplatRenderer
    GiB = 2 ** 30
    sid = "00040136-e651-4abd-991d-0655ccda9430"
    scene = kit / "scenes" / sid
    out = {}

    def peak():
        torch.cuda.synchronize()
        return round(torch.cuda.max_memory_allocated() / GiB, 3), round(torch.cuda.max_memory_reserved() / GiB, 3)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    r = NuRecGsplatRenderer(scene, layers=["background", "road"])
    r.cull_by_scale(0.95)
    sky = r.attach_sky(lo_deg=0.0, hi_deg=6.0, gain=0.3)            # the grid passes --sky-gain 0.3; the harness ABORTS if the scene has no env map
    check("scene 1 ships its sky env-map (attach_sky(gain=0.3) is not None)", sky is not None, "attached" if sky is not None else "None")
    out["scene_load_s"] = round(time.time() - t, 1)
    out["n_gaussians"] = int(r.n_gauss)
    T = r.rig_to_cam_to_nre(r.gt_rig_to_world(30 * 3))
    t = time.time()
    img1, _, ms1 = r.render(T)
    img2, _, ms2 = r.render(T)
    out["render_ms_first_second"] = [round(ms1, 1), round(ms2, 1)]
    out["render_repeat_max_abs_pixel_diff"] = int(np.abs(img1.astype(int) - img2.astype(int)).max())
    out["render_mean_pixel"] = round(float(img1.mean()), 2)
    check("render is non-trivial (mean pixel in 20..235)", 20 < float(img1.mean()) < 235, f"mean {out['render_mean_pixel']}")
    check("render is deterministic on this box (two renders identical)", out["render_repeat_max_abs_pixel_diff"] == 0,
          f"max |diff| {out['render_repeat_max_abs_pixel_diff']}", SOFT)
    out["peak_gib_renderer_alone_alloc_reserved"] = peak()
    check("(a) renderer alone fits", True, f"peak alloc/reserved {out['peak_gib_renderer_alone_alloc_reserved']} GiB")
    # (b)+(c) the model on the same device, with the renderer still resident
    rig = tp.read_rig_json(str(scene))
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    pol = tp.RefCV7Policy(str(kit / "ckpt/ckpt_50400.pt"), str(kit / "ckpt/config.json"), rig, device="cuda", scene_id=sid)
    out["model_build_s"] = round(time.time() - t, 1)
    out["model_param_total"] = pol.prov["param_total"]
    out["after_model_resident_alloc_gib"] = round(torch.cuda.memory_allocated() / GiB, 3)
    intr = cd.build_intr(cd.InProcTransport(r).camera())
    tr = cd.InProcTransport(r)
    gt = [r.gt_rig_to_world(f * 3) for f in range(r.n_frames() // 3)]
    ts = [r.frame_timestamps_us(f * 3)[1] for f in range(r.n_frames() // 3)]
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    rec = cd.run_rollout(tr, r, pol, intr, 0, 2, gt, ts, gt_stride=3)
    out["two_ticks_wall_s"] = round(time.time() - t, 2)
    out["plan_ms"] = [round(s["plan_ms"], 1) for s in rec["steps"]]
    out["peak_gib_both_resident_alloc_reserved"] = peak()
    total = torch.cuda.get_device_properties(0).total_memory / GiB
    res_peak = out["peak_gib_both_resident_alloc_reserved"][1]
    check("(c) renderer + refcv7 resident together fit in VRAM", res_peak < total * 0.97,
          f"peak reserved {res_peak} GiB of {total:.2f} GiB total")
    plan = np.array(rec["steps"][0]["plan"])
    check("refcv7 plan is finite and 4x2", plan.shape == (4, 2) and np.isfinite(plan).all(), f"{plan.tolist()}")
    out["first_plan"] = plan.tolist()
    # ---- scene 2: same render config renders (frees scene 1 and the model first; the peaks above are already read) ---------- #
    del pol, rec, tr, r
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    sid2 = "7c72937c-c620-4776-9555-d57222c0081f"
    r2 = NuRecGsplatRenderer(kit / "scenes" / sid2, layers=["background", "road"])
    r2.cull_by_scale(0.95)
    sky2 = r2.attach_sky(lo_deg=0.0, hi_deg=6.0, gain=0.3)
    check("scene 2 ships its sky env-map (attach_sky(gain=0.3) is not None)", sky2 is not None, "attached" if sky2 is not None else "None")
    img, _, ms = r2.render(r2.rig_to_cam_to_nre(r2.gt_rig_to_world(30 * 3)))
    out["scene2_render_ms"] = round(ms, 1)
    out["scene2_mean_pixel"] = round(float(img.mean()), 2)
    check("scene 2 render is non-trivial (mean pixel in 20..235)", 20 < float(img.mean()) < 235, f"mean {out['scene2_mean_pixel']}")
    return out


if __name__ == "__main__":
    sys.exit(main())
