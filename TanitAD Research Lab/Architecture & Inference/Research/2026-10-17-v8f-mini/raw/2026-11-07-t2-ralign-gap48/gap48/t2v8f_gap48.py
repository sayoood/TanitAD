"""GAP48 on box 2: the NuRec render gap on the 48 read scenes. DESCRIPTIVE: it measures, it decides nothing.

    python -I t2v8f_gap48.py all --pkg D:/t2v8f/pkg --gap D:/t2v8f/gap48 --kit D:/t2a/box2_kit --hf D:/tanitad/hf --scenes2 D:/t2v8f/scenes2 --scenes3 C:/t2v8f/scenes3 --pull D:/t2v8f/pull_gap48 [--stub]

For each scene of MANIFEST_GAP48.json (the 62 scenes minus the R-SCN exclusions), at every TICK_STEP-th tick (every tick when a scene has fewer than MIN_TICKS_ALL ticks):
  render the front-wide camera at the LOGGED pose in the `objects` condition (the replayed actors attached, drawn at the logged time), read the real mp4 frame of the same camera frame,
  encode both with the in-loop encoder, and bank both pooled token tensors (fp16 [40, 1024]) plus the cosine and the relative L2.
It is stage 1's `gap()` (t2v8f_stage1.py) widened: `build_renderer`, `make_encoder`, `E.cosine`, `E.rel_l2`, `N.grad_ncc` are IMPORTED from the stage-1 package that is already on box 2, not copied.
Two differences from `gap()` on purpose, both recorded in the result: (1) stage 1 built the renderer WITHOUT actors (`objects=False`); GAP48 attaches them (`--condition objects`, the default; `none`
reproduces stage 1). (2) stage 1 read each real frame with `N.read_ref_frame`, which re-opens the mp4 and decodes from frame 0 every time (a 3840x2160 clip: seconds per tick); GAP48 decodes the
mp4 ONCE per scene and applies the same conversion (`rgb24`, then `N.resize_area` when the size differs). The first scene is cross-checked against `N.read_ref_frame` bit for bit (`read_equiv_check`).

Markers in --pull (EXACT names; the chain waits on them): GAP48.running (liveness, rewritten after every scene), preflight.done|.fail, scene_<scene8>.done|.fail, GAP48.done | GAP48.fail.
GAP48.done = every one of the 48 scenes banked. GAP48.fail = preflight refused, or at least one scene did not bank (the other scenes' files stay). A finished scene is skipped on a relaunch (resume).
Files: tokens/<scene8>.pt (torch dict: render and real fp16 [n,40,1024], ticks, cam_frames, cosine, rel_l2, grad_ncc), GAP48_RESULT.json (rewritten after every scene).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
TIER = "T2 render-vs-real gap: descriptive diagnostic (never pooled with T1-kin; decides nothing by itself)"
TICK_STEP, MIN_TICKS_ALL = 10, 20
EXPECTED_SCENES = 48
PINNED_CODE = ("t2v8f_stage1.py", "t2v8f_run.py", "t2v8f_nocv2.py", "t2a_v8f_encoder.py", "t2a_v8f_policy.py", "t2v8f_kit.py", "t2v8f_exec.py")
TOKEN_SHAPE = (40, 1024)
_TEST_RENDER_HOOK = None            # test seam, None in production: f(render_img, real_img) -> image to encode as "the render". A result made with it says so (`test_hook`).
_S1 = {}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def jdump(o, p):
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(o, f, indent=1, default=str)
    os.replace(tmp, p)


def tick_list(n_ticks):
    """Every TICK_STEP-th tick from 0; every tick when the scene has fewer than MIN_TICKS_ALL."""
    n = int(n_ticks)
    return list(range(n)) if n < MIN_TICKS_ALL else list(range(0, n, TICK_STEP))


def ref_frames_onepass(mp4, cam_frames, size_wh):
    """{camera frame: uint8 RGB [H, W, 3] at size_wh} from ONE decode pass. The conversion is `t2v8f_nocv2.read_ref_frame`'s own (decode_frame: `rgb24`; then `resize_area` when the size differs),
    only the file is opened once instead of once per frame. Raises a named RuntimeError when the clip is shorter than the largest requested frame."""
    import av
    import t2v8f_nocv2 as N
    want = sorted(set(int(f) for f in cam_frames))
    got = {}
    wanted = set(want)
    with av.open(str(mp4)) as c:
        c.streams.video[0].thread_type = "AUTO"                                     # threaded H.264 decode gives the same pixels; the first scene's `read_equiv_check` proves it on the box
        for i, vf in enumerate(c.decode(c.streams.video[0])):
            if i in wanted:
                img = vf.to_ndarray(format="rgb24")
                if (img.shape[1], img.shape[0]) != tuple(size_wh):
                    img = N.resize_area(img, size_wh)
                got[i] = img
            if i >= want[-1]:
                break
    miss = [f for f in want if f not in got]
    if miss:
        raise RuntimeError(f"MP4_TOO_SHORT {mp4}: camera frames {miss[:5]} not decoded")
    return got


def s1():
    """The stage-1 module (imported, not copied). Needs --pkg/code on sys.path and the env main() set."""
    if "m" not in _S1:
        import t2v8f_stage1 as m
        _S1["m"] = m
    return _S1["m"]


class Ctx:
    """The three things stage 1's `build_renderer` / `make_encoder` read from its context: `.a.stub`, `.hf`, `.scene_dir(sid)`."""

    def __init__(self, a, manifest):
        self.a = a
        self.pkg, self.gap, self.kit, self.hf = (x.replace("\\", "/") for x in (a.pkg, a.gap, a.kit, a.hf))
        self.pull = a.pull.replace("\\", "/")
        self.roots = {"kit": f"{self.kit}/scenes", "scenes2": a.scenes2.replace("\\", "/"), "scenes3": a.scenes3.replace("\\", "/")}
        self.manifest = manifest
        self.where = {s["scene"]: s["where"] for s in manifest["scenes"]}
        os.makedirs(f"{self.pull}/tokens", exist_ok=True)

    def scene_dir(self, sid):
        return f"{self.roots[self.where[sid]]}/{sid}"

    def mp4(self, sid):
        return f"{self.gap}/mp4/{sid}/camera_front_wide_120fov.mp4"


def set_env(ctx):
    pkg = ctx.pkg
    env = dict(T2A_KIT=ctx.kit, T2V8F_TREE=f"{pkg}/tree", T2V8F_V9_DIR=f"{pkg}/v9", T2V8F_TOOL=f"{pkg}/refs/tool/v8f_t1_eval_ri_lf6.py", T2V8F_SCREEN_TABLE=f"{pkg}/refs/SCREEN_TABLE_2.json",
               HF_HOME=ctx.hf, HF_HUB_OFFLINE="1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    if ctx.a.stub:
        env.update(T2A_STUB="1", CUDA_VISIBLE_DEVICES="-1")
    os.environ.update(env)
    return env


def marker(pull, name, ok, text):
    for ext in ("done", "fail"):
        p = f"{pull}/{name}.{ext}"
        if os.path.exists(p):
            os.replace(p, p + ".prev")
    with open(f"{pull}/{name}.{'done' if ok else 'fail'}", "w", newline="\n") as f:
        f.write(text + "\n")


# ============================================================================================================================ preflight
def preflight(ctx):
    a, m, bad, info = ctx.a, ctx.manifest, [], {}

    def fail(reason):
        bad.append(reason)

    info["python"], info["platform"] = sys.version.split()[0], platform.platform()
    for mod in ("numpy", "torch", "transformers", "PIL", "av"):
        try:
            info[mod] = __import__(mod).__version__ if mod != "PIL" else __import__("PIL").__version__
        except Exception as e:                                                      # noqa: BLE001
            fail(f"IMPORT_FAILED {mod}: {e!r}"[:160])
    if len(m["scenes"]) != EXPECTED_SCENES and not m.get("fixture"):
        fail(f"SCENE_COUNT {len(m['scenes'])} != {EXPECTED_SCENES}")
    if not a.stub:
        for exe in ("cl", "nvcc", "ninja"):
            if shutil.which(exe) is None:
                fail(f"MISSING_TOOLCHAIN {exe} not on PATH (gsplat JIT-compiles on first use; vcvars64 + CUDA 12.8 + ninja)")
        for mod in ("gsplat", "msgpack"):
            try:
                info[mod] = getattr(__import__(mod), "__version__", "ok")
            except Exception as e:                                                  # noqa: BLE001
                fail(f"IMPORT_FAILED {mod}: {e!r}"[:160])
        try:
            import torch
            if not torch.cuda.is_available():
                fail("NO_CUDA_DEVICE torch.cuda.is_available() is False")
            else:
                free, tot = torch.cuda.mem_get_info()
                info["gpu"] = {"name": torch.cuda.get_device_name(0), "free_gib": round(free / 2**30, 2), "total_gib": round(tot / 2**30, 2)}
                if free < 4 * 2**30:
                    fail(f"GPU_MEMORY_LOW {free / 2**30:.2f} GiB free (< 4 GiB)")
        except Exception as e:                                                      # noqa: BLE001
            fail(f"CUDA_PROBE_FAILED {e!r}"[:160])
    else:
        info["gpu"] = "skipped (stub rehearsal)"
    for label, path in (("pull", ctx.pull), ("gap", ctx.gap)):
        try:
            free = shutil.disk_usage(path if os.path.exists(path) else os.path.dirname(path)).free / 2**30
            info[f"free_gib_{label}"] = round(free, 2)
            if free < a.min_free_gb:
                fail(f"DISK_LOW {label} {path}: {free:.1f} GiB free < {a.min_free_gb}")
        except Exception as e:                                                      # noqa: BLE001
            fail(f"DISK_PROBE_FAILED {label} {path}: {e!r}"[:160])
    for e in m["own_files"]:                                                        # the shipped code, job file and (when listed) the clips
        p = f"{ctx.gap}/{e['path']}"
        if not os.path.isfile(p):
            fail(f"MISSING {e['path']}")
        elif os.path.getsize(p) != e["size"]:
            fail(f"SIZE_MISMATCH {e['path']}: {os.path.getsize(p)} != {e['size']}")
        elif sha256_file(p) != e["sha256"]:
            fail(f"HASH_MISMATCH {e['path']}")
    n_pin = 0
    for e in m["pinned_code"]:                                                      # what this job imports from the stage-1 package already on the box
        p = f"{ctx.pkg}/code/{e['file']}"
        if not os.path.isfile(p):
            fail(f"MISSING pkg/code/{e['file']}")
        elif sha256_file(p) != e["sha256"]:
            fail(f"HASH_MISMATCH pkg/code/{e['file']} (not the code this package was built against)")
        else:
            n_pin += 1
    info["pinned_code_ok"] = n_pin
    for label, p in (("tree", f"{ctx.pkg}/tree"), ("v9", f"{ctx.pkg}/v9"), ("tool", os.environ.get("T2V8F_TOOL", "")), ("screen_table", os.environ.get("T2V8F_SCREEN_TABLE", ""))):
        if not os.path.exists(p):
            fail(f"MISSING pkg:{label} {p}")
    for rel in ("tree/stack/experiments/alpasim-gsplat/closedloop_drive.py", "tree/stack/experiments/alpasim-gsplat/gsplat_renderer.py", "tree/stack/experiments/alpasim-gsplat/actor_map.py",
                "t2a/stub_gsplat_renderer.py"):
        if not os.path.isfile(f"{ctx.kit}/{rel}"):
            fail(f"MISSING kit:{rel}")
    n_sc, n_hash = 0, 0
    for sc in m["scenes"]:
        sd = ctx.scene_dir(sc["scene"])
        if not os.path.isdir(sd):
            fail(f"MISSING scene dir {sc['where']}:{sc['scene']}")
            continue
        if not os.path.isfile(f"{sd}/ref_frames.npz"):
            fail(f"MISSING {sc['where']}:{sc['scene']}/ref_frames.npz (the actor falsifier reads it)")
        for f in sc["files"]:
            p = f"{sd}/{f['rel']}"
            if not os.path.isfile(p):
                fail(f"MISSING {sc['where']}:{sc['scene']}/{f['rel']}")
            elif os.path.getsize(p) != f["bytes"]:
                fail(f"SIZE_MISMATCH {sc['where']}:{sc['scene']}/{f['rel']}: {os.path.getsize(p)} != {f['bytes']}")
            elif (f["rel"] != "volume.nurec" or a.hash_volumes):
                n_hash += 1
                if sha256_file(p) != f["sha256"]:
                    fail(f"HASH_MISMATCH {sc['where']}:{sc['scene']}/{f['rel']}")
        n_sc += 1
    info["scenes_present"], info["scene_files_hashed"], info["hash_volumes"] = n_sc, n_hash, bool(a.hash_volumes)
    if not a.stub:
        try:
            sys.path.insert(0, f"{ctx.pkg}/code")
            import t2a_v8f_encoder as E
            snap = E.find_local_snapshot(f"{ctx.hf}/hub")
            info["dinov3_snapshot"] = snap
            dino = s1().DINO_SHA
            if sha256_file(f"{snap}/model.safetensors") != dino:
                fail(f"HASH_MISMATCH dinov3 {snap}/model.safetensors")
        except Exception as e:                                                      # noqa: BLE001
            fail(f"MISSING dinov3 under {ctx.hf}/hub: {e!r}"[:200])
    try:
        import t2v8f_nocv2 as N
        info["mp4_decoders_here"] = N.decoders()
    except Exception as e:                                                          # noqa: BLE001
        fail(f"IMPORT_FAILED t2v8f_nocv2: {e!r}"[:160])
    return {"verdict": "REFUSED" if bad else "PASS", "failures": bad, "info": info}


# ============================================================================================================================ one scene
def measure_scene(ctx, enc, E, sc, first):
    """Render at the logged pose, read the real frame, encode both, bank. -> (result dict, tensors dict). Raises on a scene-level failure (the caller records a .fail)."""
    import numpy as np
    import torch
    import t2v8f_nocv2 as N
    S1 = s1()
    sid = sc["scene"]
    t0 = time.time()
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:                                                               # noqa: BLE001
        pass
    r, tr, intr, stride, info, _R = S1.build_renderer(ctx, sid, objects=(ctx.a.condition == "objects"))
    t_build = time.time() - t0
    if info is not None and info.get("verdict") == "REFUSED":
        raise RuntimeError(f"ACTOR_REFUSED {json.dumps({k: info.get(k) for k in ('n_tracks_json', 'n_layer_tracks', 'n_mapped', 'falsifier_pass_frames')})}")
    n = r.n_frames() // stride
    ticks = tick_list(n)
    frames = [f * stride for f in ticks]
    size = (r.cam.width, r.cam.height)
    t1 = time.time()
    refs = ref_frames_onepass(ctx.mp4(sid), frames, size)
    t_decode = time.time() - t1
    equiv = None
    if first:                                                                       # box-2 proof that the one-pass read is `read_ref_frame`'s read, on the cheapest nonzero frame
        fr = frames[1] if len(frames) > 1 else frames[0]
        old = N.read_ref_frame(ctx.mp4(sid), fr, size)
        equiv = {"scene": sid, "cam_frame": int(fr), "bit_equal": bool(np.array_equal(old, refs[fr]))}
        if not equiv["bit_equal"]:
            raise RuntimeError(f"READ_EQUIV_FAILED the one-pass frame {fr} differs from t2v8f_nocv2.read_ref_frame")
    t2 = time.time()
    rend_tok, real_tok, rows = [], [], []
    for f, fr in zip(ticks, frames):
        ts = r.frame_timestamps_us(fr)[1]
        img = tr.render(r.gt_rig_to_world(fr) @ r.cam.T_sensor_rig, ts)
        ref = refs[fr]
        if _TEST_RENDER_HOOK is not None:
            img = _TEST_RENDER_HOOK(img, ref)
        ta, tb = enc.tokens(img, intr), enc.tokens(ref, intr)
        rend_tok.append(ta.to(torch.float16))
        real_tok.append(tb.to(torch.float16))
        rows.append({"tick": int(f), "cam_frame": int(fr), "cosine": E.cosine(ta.float(), tb.float()), "rel_l2": E.rel_l2(ta.float(), tb.float()), "grad_ncc_pixels": float(N.grad_ncc(img, ref))})
    t_ticks = time.time() - t2
    rt, qt = torch.stack(rend_tok), torch.stack(real_tok)
    if tuple(rt.shape[1:]) != TOKEN_SHAPE or tuple(qt.shape[1:]) != TOKEN_SHAPE:
        raise RuntimeError(f"TOKEN_SHAPE {tuple(rt.shape)} {tuple(qt.shape)}")
    cos = np.array([x["cosine"] for x in rows])
    harness = None if info is None else {k: v for k, v in info.items() if k != "per_track"}
    gpu_peak = None
    try:
        gpu_peak = round(torch.cuda.max_memory_allocated() / 2**30, 3) if torch.cuda.is_available() else None
    except Exception:                                                               # noqa: BLE001
        pass
    del r, tr, refs
    S1.free_gpu()
    res = {"scene": sid, "scene8": sid[:8], "idx": sc["idx"], "where": sc["where"], "n_ticks_total": int(n), "stride": int(stride), "n_pairs": len(rows), "ticks": [x["tick"] for x in rows],
           "cam_frames": [x["cam_frame"] for x in rows], "cosine": [x["cosine"] for x in rows], "rel_l2": [x["rel_l2"] for x in rows], "grad_ncc_pixels": [x["grad_ncc_pixels"] for x in rows],
           "cosine_median": float(np.median(cos)), "cosine_min": float(cos.min()), "cosine_max": float(cos.max()), "condition": ctx.a.condition, "actors": harness, "frame_size_wh": list(size),
           "seconds": {"build_renderer_and_actors": round(t_build, 2), "decode_mp4_once": round(t_decode, 2), "render_and_encode": round(t_ticks, 2), "total": round(time.time() - t0, 2)},
           "gpu_peak_alloc_gib": gpu_peak, "read_equiv_check": equiv}
    tens = {"scene": sid, "ticks": torch.tensor(res["ticks"]), "cam_frames": torch.tensor(res["cam_frames"]), "render": rt, "real": qt, "cosine": torch.tensor(res["cosine"], dtype=torch.float64),
            "rel_l2": torch.tensor(res["rel_l2"], dtype=torch.float64), "grad_ncc_pixels": torch.tensor(res["grad_ncc_pixels"], dtype=torch.float64), "token_dtype": "float16 (exact image of the fp8 e4m3fn tokens)"}
    return res, tens


def resumable(pull, prev_scenes, s8, sid):
    """The earlier launch's record of a scene, only if its marker and token file are there and the file still hashes to the recorded sha256 (verified by content, not by presence)."""
    tok = f"{pull}/tokens/{s8}.pt"
    old = prev_scenes.get(sid)
    if not (old and old.get("status") == "DONE" and os.path.isfile(f"{pull}/scene_{s8}.done") and os.path.isfile(tok) and os.path.getsize(tok) > 0):
        return None
    return old if sha256_file(tok) == old.get("tokens_sha256") else None


# ============================================================================================================================ driver
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("all",))
    ap.add_argument("--pkg", required=True, help="the stage-1 package on the box (D:/t2v8f/pkg): code, tree, v9, refs")
    ap.add_argument("--gap", default=os.path.dirname(HERE), help="this package (D:/t2v8f/gap48): MANIFEST_GAP48.json, mp4/")
    ap.add_argument("--kit", required=True)
    ap.add_argument("--hf", required=True)
    ap.add_argument("--scenes2", required=True)
    ap.add_argument("--scenes3", required=True)
    ap.add_argument("--pull", required=True)
    ap.add_argument("--manifest", default=None, help="default <gap>/MANIFEST_GAP48.json")
    ap.add_argument("--condition", choices=("objects", "none"), default="objects")
    ap.add_argument("--stub", action="store_true", help="CPU rehearsal: stub renderer, stub encoder, no CUDA / DINOv3 checks")
    ap.add_argument("--hash-volumes", dest="hash_volumes", action="store_true", help="also sha256 the large volume.nurec files (26 GB, minutes); default: sizes + every small file's sha256. Stages 4 and 5 hashed them on this box")
    ap.add_argument("--limit-scenes", type=int, default=0, help="rehearsal only: the first N scenes")
    ap.add_argument("--min-free-gb", type=float, default=3.0)
    a = ap.parse_args(argv)
    manifest = json.load(open(a.manifest or f"{a.gap}/MANIFEST_GAP48.json", encoding="utf-8"))
    ctx = Ctx(a, manifest)
    sys.path.insert(0, f"{ctx.pkg}/code")
    set_env(ctx)
    pull = ctx.pull
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    scenes = manifest["scenes"][:a.limit_scenes] if a.limit_scenes else manifest["scenes"]
    res = {"stage": "gap48", "tier": TIER, "started": started, "stub": bool(a.stub), "host": platform.node(), "condition": a.condition, "tick_step": TICK_STEP, "min_ticks_all": MIN_TICKS_ALL,
           "test_hook": _TEST_RENDER_HOOK is not None, "manifest_sha256": sha256_file(a.manifest or f"{a.gap}/MANIFEST_GAP48.json"), "n_scenes_expected": len(scenes),
           "scene_list_sha256": manifest.get("scene_list_sha256"), "steps": {}, "scenes": {}}

    def save():
        res["n_done"] = sum(1 for v in res["scenes"].values() if v["status"] == "DONE")
        res["n_failed"] = sum(1 for v in res["scenes"].values() if v["status"] == "FAIL")
        jdump(res, f"{pull}/GAP48_RESULT.json")

    def live(extra=""):
        with open(f"{pull}/GAP48.running", "w", newline="\n") as f:
            f.write(f"pid={os.getpid()} start={started} done={res.get('n_done', 0)}/{len(scenes)} {extra}\n")

    prev_scenes = {}
    if os.path.isfile(f"{pull}/GAP48_RESULT.json"):
        try:
            prev_scenes = json.load(open(f"{pull}/GAP48_RESULT.json", encoding="utf-8")).get("scenes") or {}
        except Exception:                                                           # noqa: BLE001
            prev_scenes = {}
    for stale in ("GAP48.done", "GAP48.fail"):
        if os.path.exists(f"{pull}/{stale}"):
            os.replace(f"{pull}/{stale}", f"{pull}/{stale}.prev")
    live()
    t0 = time.time()
    try:
        pf = preflight(ctx)
    except Exception as e:                                                          # noqa: BLE001
        pf = {"verdict": "REFUSED", "failures": [f"PREFLIGHT_CRASH {e!r}"[:300]], "info": {}}
    pf["seconds"] = round(time.time() - t0, 1)
    res["steps"]["preflight"] = pf
    save()
    marker(pull, "preflight", pf["verdict"] == "PASS", f"{pf['verdict']} {json.dumps(pf['failures'])[:600]}")
    if pf["verdict"] != "PASS":
        res["finished"], res["overall"] = time.strftime("%Y-%m-%d %H:%M:%S"), "PREFLIGHT_REFUSED"
        save()
        os.remove(f"{pull}/GAP48.running")
        marker(pull, "GAP48", False, f"PREFLIGHT_REFUSED {json.dumps(pf['failures'])[:800]}")
        print(json.dumps({"overall": res["overall"], "failures": pf["failures"][:5]}))
        return 1
    S1 = s1()
    enc, E = S1.make_encoder(ctx)
    first = True
    for sc in scenes:
        sid, s8 = sc["scene"], sc["scene"][:8]
        tok = f"{pull}/tokens/{s8}.pt"
        old = resumable(pull, prev_scenes, s8, sid)
        if old is not None:                                                         # banked by an earlier launch and the file still hashes to what was recorded
            res["scenes"][sid] = old
            if old.get("read_equiv_check"):
                res["read_equiv_check"] = old["read_equiv_check"]
            first = first and not old.get("read_equiv_check")
            continue
        t1 = time.time()
        try:
            r, tens = measure_scene(ctx, enc, E, sc, first)
            import torch
            torch.save(tens, tok + ".tmp")
            os.replace(tok + ".tmp", tok)
            r.update(status="DONE", tokens_file=f"tokens/{s8}.pt", tokens_sha256=sha256_file(tok), tokens_bytes=os.path.getsize(tok))
            res["scenes"][sid] = r
            if r.get("read_equiv_check"):
                res["read_equiv_check"] = r["read_equiv_check"]
            marker(pull, f"scene_{s8}", True, f"DONE n_pairs={r['n_pairs']} cosine_median={r['cosine_median']:.4f} seconds={r['seconds']['total']}")
            first = False
        except Exception as e:                                                      # noqa: BLE001
            res["scenes"][sid] = {"scene": sid, "scene8": s8, "idx": sc["idx"], "where": sc["where"], "status": "FAIL", "reason": repr(e)[:400], "trace": traceback.format_exc()[-1500:],
                                  "seconds": {"total": round(time.time() - t1, 2)}}
            marker(pull, f"scene_{s8}", False, f"FAIL {repr(e)[:500]}")
        try:                                                                        # after the except block, so the dead scene's traceback cannot keep the renderer alive
            S1.free_gpu()
        except Exception:                                                           # noqa: BLE001
            pass
        save()
        live(f"last={s8}|{res['scenes'][sid]['status']}")
    res["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save()
    ok = res["n_failed"] == 0 and res["n_done"] == len(scenes)
    res["overall"] = "ALL_SCENES_BANKED" if ok else f"SCENES_FAILED {res['n_failed']} of {len(scenes)}"
    save()
    os.remove(f"{pull}/GAP48.running")
    marker(pull, "GAP48", ok, f"{res['overall']} n_done={res['n_done']}")
    print(json.dumps({"overall": res["overall"], "n_done": res["n_done"], "n_failed": res["n_failed"]}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
