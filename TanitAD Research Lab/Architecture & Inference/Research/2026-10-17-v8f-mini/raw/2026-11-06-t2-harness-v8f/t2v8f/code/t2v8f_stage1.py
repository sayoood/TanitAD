"""Stage 1 of the v8F-mini T2 re-perception harness on box 2: ONE driver, five jobs in order, strictly one child process at a time (safe mode).

    python -I t2v8f_stage1.py all --pkg PKG --kit D:/t2a/box2_kit --hf D:/tanitad/hf --pull D:/t2v8f/pull [--stub] [--matrix-inproc]

  1 preflight  every path, CUDA, free disk, every sha256 / md5 (refuses with a named reason per failure; later jobs do not run), then an additive install of the
               mp4, the pre-extracted reference frames and sequence_tracks.json into the kit's scene dirs
  2 T5         one C0_s0 start rendered twice (two child processes): plans and frame md5 bit-equal
  3 gap        T1 render gap: frame rendered at the LOGGED pose vs the real mp4 frame, pooled-token cosine and rel-L2. REPORTED beside; no threshold
  4 actor      `objects` draws the lead in 7c72937c (ROI from the lead's projected cuboid); the DECIDING planted no-lead read is condition `empty` (a renderer with no actors
               attached) rendered at the SAME pose and tick as the `objects` read (MM ruling, card 6b). The old planted read (actor clock 30 s past every track) is reported beside, never deciding
  5 plumbing   every arm x {T2, RR} x both scenes x 1 start. Stamped PLUMBING ONLY: never read for H1 or H2

Markers in --pull: STAGE1.running (liveness), <job>.done|.fail per job, STAGE1.done | STAGE1.fail at the end, STAGE1_RESULT.json after every job.
STAGE1.fail = preflight refused, or a job could not run / crashed. A MEASURED negative (T5 differs, actor not drawn) is a finding: STAGE1.done, gate FAIL in the JSON.
Actor thresholds (committed before any box-2 render; UNVERIFIED on real renders): PRESENT = >= 200 changed pixels (> 8 levels) inside the ROI AND >= 1 % of the ROI area,
`objects` frame vs `empty` frame at the same pose and tick (the two differ only by the replayed agents); ABSENT = <= 5 changed pixels inside the ROI. The gate needs PRESENT at
>= 2 of 3 lead ticks, the planted `empty` read ABSENT (vs the objects renderer with its actors switched off) at all 3, and disabled == off exactly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
TIER = "T2 re-perception rollout (never pooled with T1-kin)"
PLUMBING = "PLUMBING ONLY: this rollout is never read for H1 or H2"
SCENES = [("00040136-e651-4abd-991d-0655ccda9430", 0), ("7c72937c-c620-4776-9555-d57222c0081f", 1)]
ARMS = ("C0_s0", "C0_s1", "DAGOACC_s0", "DAGOACC_s1", "P10", "ZERO", "BLIND")
RENDER_FLAGS = ["--cull-scale-quantile", "0.95", "--sky-gain", "0.3"]           # the render flags of every banked Thor number
KIT_REQUIRED = ["tree/stack/experiments/alpasim-gsplat/closedloop_drive.py", "tree/stack/experiments/alpasim-gsplat/gsplat_renderer.py",
                "tree/stack/experiments/alpasim-gsplat/actor_map.py", "tree/stack/experiments/alpasim-gsplat/synth_actor.py", "t2a/stub_gsplat_renderer.py"]
DINO_SHA = "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179"
ROI_PRESENT_PX, ROI_PRESENT_FRAC, ROI_ABSENT_PX = 200, 0.01, 5


def sha256_file(p, algo="sha256"):
    h = hashlib.new(algo)
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def jdump(o, p):
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(o, f, indent=1, default=str)
    os.replace(tmp, p)


class Ctx:
    def __init__(self, a):
        self.a = a
        self.pkg, self.kit, self.hf = a.pkg.replace("\\", "/"), a.kit.replace("\\", "/"), a.hf.replace("\\", "/")
        self.pull = a.pull.replace("\\", "/")
        os.makedirs(self.pull, exist_ok=True)
        self.manifest = json.load(open(f"{self.pkg}/MANIFEST.json", encoding="utf-8"))
        self.res = {"stage": "stage1", "tier": TIER, "purpose": PLUMBING, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "stub": bool(a.stub), "host": platform.node(),
                    "manifest_sha256": sha256_file(f"{self.pkg}/MANIFEST.json"), "steps": {}, "gates": {}}
        self.env = dict(os.environ, T2A_KIT=self.kit, T2V8F_TREE=f"{self.pkg}/tree", T2V8F_V9_DIR=f"{self.pkg}/v9", T2V8F_TOOL=f"{self.pkg}/refs/tool/v8f_t1_eval_ri_lf6.py",
                        T2V8F_SCREEN_TABLE=f"{self.pkg}/refs/SCREEN_TABLE_2.json", HF_HOME=self.hf, HF_HUB_OFFLINE="1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        if a.stub:
            self.env.update(T2A_STUB="1", CUDA_VISIBLE_DEVICES="-1")
        os.environ.update({k: v for k, v in self.env.items() if k.startswith(("T2", "HF_", "CUDA"))})
        self.device = "cpu" if a.stub else "cuda"
        self.encoder = "stub" if a.stub else "dinov3"

    def save(self):
        jdump(self.res, f"{self.pull}/STAGE1_RESULT.json")

    def scene_dir(self, sid):
        return f"{self.kit}/scenes/{sid}"

    def entry_args(self, arm, tcond, agents, out, sid, idx, starts):
        return ["--scene-dir", self.scene_dir(sid), "--arm", arm, "--condition", agents, "--starts", starts, "--steps", "60", "--out", out, "--t2-cond", tcond, "--t2-scene-idx", str(idx),
                "--t2-scene-id", sid, "--t2-encoder", self.encoder, "--t2-device", self.device, "--t2-hub", f"{self.hf}/hub", "--t2-ckpt-root", f"{self.pkg}/ckpt",
                "--t2-blind-mean", f"{self.pkg}/refs/blind_mean_eval141.pt", "--t2-plumbing", *RENDER_FLAGS]

    def run_child(self, argv, log, timeout=7200):
        """Run ONE child to completion (never two at once). Returns the return code (negative = a native crash)."""
        with open(log, "ab") as lf:
            return subprocess.run([sys.executable, "-I", "-u"] + argv, env=self.env, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, timeout=timeout).returncode


def marker(ctx, name, ok, text):
    for ext in ("done", "fail"):
        p = f"{ctx.pull}/{name}.{ext}"
        if os.path.exists(p):
            os.replace(p, p + ".prev")
    with open(f"{ctx.pull}/{name}.{'done' if ok else 'fail'}", "w", newline="\n") as f:
        f.write(text + "\n")


# ============================================================================================================================ 1 preflight
def preflight(ctx):
    a, m, bad, info = ctx.a, ctx.manifest, [], {}

    def fail(reason):
        bad.append(reason)

    info["python"], info["platform"] = sys.version.split()[0], platform.platform()
    for mod in ("numpy", "torch", "transformers", "PIL"):
        try:
            info[mod] = __import__(mod).__version__ if mod != "PIL" else __import__("PIL").__version__
        except Exception as e:                                                      # noqa: BLE001
            fail(f"IMPORT_FAILED {mod}: {e!r}"[:160])
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
    for label, path in (("pkg", ctx.pkg), ("pull", ctx.pull), ("kit", ctx.kit)):
        try:
            free = shutil.disk_usage(path if os.path.exists(path) else os.path.dirname(path)).free / 2**30
            info[f"free_gib_{label}"] = round(free, 2)
            if free < a.min_free_gb:
                fail(f"DISK_LOW {label} {path}: {free:.1f} GiB free < {a.min_free_gb}")
        except Exception as e:                                                      # noqa: BLE001
            fail(f"DISK_PROBE_FAILED {label} {path}: {e!r}"[:160])
    n_ok = 0
    for e in m["files"]:                                                            # every shipped file: present, size, sha256
        p = f"{ctx.pkg}/{e['path']}"
        if not os.path.isfile(p):
            fail(f"MISSING {e['path']}")
        elif os.path.getsize(p) != e["size"]:
            fail(f"SIZE_MISMATCH {e['path']}: {os.path.getsize(p)} != {e['size']}")
        elif sha256_file(p) != e["sha256"]:
            fail(f"HASH_MISMATCH {e['path']}")
        else:
            n_ok += 1
    info["pkg_files_ok"], info["pkg_files"] = n_ok, len(m["files"])
    for rel in KIT_REQUIRED:
        if not os.path.isfile(f"{ctx.kit}/{rel}"):
            fail(f"MISSING kit:{rel}")
    kman = f"{ctx.kit}/KIT_MANIFEST.json"
    if not os.path.isfile(kman):
        fail("MISSING kit:KIT_MANIFEST.json")
    else:
        if not a.stub and m.get("kit_manifest_sha256") and sha256_file(kman) != m["kit_manifest_sha256"]:
            fail("HASH_MISMATCH kit:KIT_MANIFEST.json (not the kit this package was built against)")
        n_kit = 0
        for it in json.load(open(kman, encoding="utf-8"))["items"]:
            if it["kind"] == "scene" or it["dst"].endswith("alpasim-gsplat/closedloop_drive.py") and it["kind"] == "overlay":
                p = f"{ctx.kit}/{it['dst']}"
                if not os.path.isfile(p):
                    fail(f"MISSING kit:{it['dst']}")
                elif os.path.getsize(p) != it["size"] or sha256_file(p, "md5") != it["md5"]:
                    fail(f"HASH_MISMATCH kit:{it['dst']}")
                else:
                    n_kit += 1
        info["kit_items_ok"] = n_kit
    for sid, _ in SCENES:
        if not os.path.isdir(ctx.scene_dir(sid)):
            fail(f"MISSING kit:scenes/{sid}")
    if not a.stub:
        try:
            sys.path.insert(0, HERE)
            import t2a_v8f_encoder as E
            snap = E.find_local_snapshot(f"{ctx.hf}/hub")
            info["dinov3_snapshot"] = snap
            if sha256_file(f"{snap}/model.safetensors") != DINO_SHA:
                fail(f"HASH_MISMATCH dinov3 {snap}/model.safetensors")
        except Exception as e:                                                      # noqa: BLE001
            fail(f"MISSING dinov3 under {ctx.hf}/hub: {e!r}"[:200])
    try:
        sys.path.insert(0, HERE)
        import t2v8f_nocv2 as N
        info["mp4_decoders_here"] = N.decoders()
    except Exception as e:                                                          # noqa: BLE001
        info["mp4_decoders_here"] = f"probe failed: {e!r}"
    return {"verdict": "REFUSED" if bad else "PASS", "failures": bad, "info": info}


def install_refs(ctx):
    """Additive: the mp4, the pre-extracted frames and sequence_tracks.json into the kit's scene dirs. A different file already there -> refuse (never overwrite)."""
    done, bad = [], []
    for sid, _ in SCENES:
        for rel in ("camera_front_wide_120fov.mp4", "ref_frames.npz", "extracted/sequence_tracks.json"):
            src, dst = f"{ctx.pkg}/refs/scenes/{sid}/{rel}", f"{ctx.scene_dir(sid)}/{rel}"
            if os.path.exists(dst):
                (done if sha256_file(dst) == sha256_file(src) else bad).append(dst if sha256_file(dst) == sha256_file(src) else f"EXISTS_DIFFERENT {dst}")
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)
            (done if sha256_file(dst) == sha256_file(src) else bad).append(dst if sha256_file(dst) == sha256_file(src) else f"COPY_MISMATCH {dst}")
    return {"verdict": "REFUSED" if bad else "PASS", "installed_or_identical": len(done), "failures": bad}


# ============================================================================================================================ 2 T5 determinism
def t5(ctx):
    sid, idx = SCENES[1]
    f0 = str(ctx.manifest["plumbing_starts"][sid])
    out = {}
    for tag in ("A", "B"):
        d = f"{ctx.pull}/t5/{tag}"
        os.makedirs(d, exist_ok=True)
        if os.path.exists(f"{d}/records.jsonl"):
            os.replace(f"{d}/records.jsonl", f"{d}/records.jsonl.prev")
        rc = ctx.run_child([f"{HERE}/t2v8f_run.py"] + ctx.entry_args("C0_s0", "T2", "objects", d, sid, idx, f0), f"{ctx.pull}/t5_{tag}.log")
        out[tag] = {"rc": rc}
        if rc == 0 and os.path.exists(f"{d}/records.jsonl"):
            out[tag]["rec"] = json.loads(open(f"{d}/records.jsonl", encoding="utf-8").readline())
    if any(out[t]["rc"] != 0 or "rec" not in out[t] for t in "AB"):
        return {"verdict": "CRASH", "rcs": {t: out[t]["rc"] for t in "AB"}}
    A, B = out["A"]["rec"], out["B"]["rec"]
    import numpy as np
    pa, pb = np.asarray(A["plans"], np.float64), np.asarray(B["plans"], np.float64)
    ma, mb = A["frame_md5"], B["frame_md5"]
    n_diff = sum(x != y for x, y in zip(ma, mb)) + abs(len(ma) - len(mb))
    equal = bool(np.array_equal(pa, pb)) and n_diff == 0 and A["cl"] == B["cl"]
    return {"verdict": "PASS" if equal else "FAIL", "scene": sid, "f0": int(f0), "arm": "C0_s0", "n_frames": len(ma), "frames_differing": int(n_diff), "plans_bit_equal": bool(np.array_equal(pa, pb)),
            "max_abs_plan_diff_m": float(np.abs(pa - pb).max()), "executed_equal": A["cl"] == B["cl"], "rule": "bit-equal plans, executed track and every frame md5"}


# ============================================================================================================================ in-process renderer (gap, actor)
def build_renderer(ctx, sid, objects=False):
    import t2v8f_run as R
    cd = R.setup()
    from gsplat_renderer import NuRecGsplatRenderer
    sd = ctx.scene_dir(sid)
    r = NuRecGsplatRenderer(sd, layers=["background", "road"])
    r.cull_by_scale(0.95)
    r.attach_sky(lo_deg=0.0, hi_deg=6.0, gain=0.3)
    info = None
    if objects:
        import actor_map
        info = actor_map.attach_actors_verified(r, sd)
    tr = cd.InProcTransport(r)
    intr = cd.build_intr(tr.camera())
    stride = int(round(1e5 / (r.frame_timestamps_us(1)[1] - r.frame_timestamps_us(0)[1])))
    return r, tr, intr, stride, info, R


def make_encoder(ctx):
    sys.path.insert(0, HERE)
    import t2a_v8f_encoder as E
    if ctx.a.stub:
        return E.StubEncoder(), E
    return E.InLoopEncoder(E.find_local_snapshot(f"{ctx.hf}/hub"), "bf16", "cuda"), E


def gap(ctx):
    import numpy as np
    import t2v8f_nocv2 as N
    enc, E = make_encoder(ctx)
    per_scene = {}
    for sid, idx in SCENES:
        r, tr, intr, stride, _, _ = build_renderer(ctx, sid)
        n = r.n_frames() // stride
        ticks = ctx.manifest["gap_ticks"][sid]
        rows = []
        for f in ticks:
            fr = f * stride
            ts = r.frame_timestamps_us(fr)[1]
            img = tr.render(r.gt_rig_to_world(fr) @ r.cam.T_sensor_rig, ts)
            ref = N.read_ref_frame(f"{ctx.scene_dir(sid)}/camera_front_wide_120fov.mp4", fr, (r.cam.width, r.cam.height))
            ta, tb = enc.tokens(img, intr), enc.tokens(ref, intr)
            rows.append({"tick": f, "cam_frame": fr, "cosine": E.cosine(ta.float(), tb.float()), "rel_l2": E.rel_l2(ta.float(), tb.float()), "grad_ncc_pixels": N.grad_ncc(img, ref)})
        c, l2 = np.array([x["cosine"] for x in rows]), np.array([x["rel_l2"] for x in rows])
        per_scene[sid] = {"n_ticks": n, "rows": rows, "cosine": {"median": float(np.median(c)), "min": float(c.min()), "max": float(c.max())},
                          "rel_l2": {"median": float(np.median(l2)), "min": float(l2.min()), "max": float(l2.max())}}
        del r, tr
        free_gpu()
    return {"verdict": "REPORTED", "rule": "beside only: no threshold. Encoder-vs-cache floor (G2, one real frame): rel-L2 0.0264, cosine 0.99965", "scenes": per_scene}


# ============================================================================================================================ 4 actor falsifier
def lead_roi(intr, T_sensor_rig, lx, ly, length=4.5, width=1.9, height=1.6):
    """Pixel box (x0, y0, x1, y1) of a cuboid standing on the ground at (lx, ly) in the RIG frame (x forward, y left, z up): its 8 corners through the f-theta model."""
    import numpy as np
    corners = np.array([[lx + dx, ly + dy, z, 1.0] for dx in (-length / 2, length / 2) for dy in (-width / 2, width / 2) for z in (0.0, height)])
    pc = (np.linalg.inv(np.asarray(T_sensor_rig, np.float64)) @ corners.T).T[:, :3]
    rho = np.hypot(pc[:, 0], pc[:, 1])
    th = np.arctan2(rho, pc[:, 2])
    rad = np.array([intr.r_of_theta(t) for t in th])
    u, v = intr.cx + rad * pc[:, 0] / np.maximum(rho, 1e-9), intr.cy + rad * pc[:, 1] / np.maximum(rho, 1e-9)
    x0, x1 = int(max(np.floor(u.min()), 0)), int(min(np.ceil(u.max()), intr.width - 1))
    y0, y1 = int(max(np.floor(v.min()), 0)), int(min(np.ceil(v.max()), intr.height - 1))
    return x0, y0, x1, y1


class ActorDouble:
    """STUB REHEARSAL ONLY: wraps the stub renderer, and while actors are enabled and the actor time lies inside [t_lo, t_hi] paints a white box at `roi`."""

    def __init__(self, inner, roi, t_lo, t_hi):
        self.inner, self.roi, self.t_lo, self.t_hi, self.on = inner, roi, t_lo, t_hi, True

    def __getattr__(self, k):
        return getattr(self.inner, k)

    def set_actors_enabled(self, on):
        self.on = bool(on)

    def render(self, cam_to_nre, tau=0.0, actor_time_us=None, cam_to_nre_end=None):
        img, a, ms = self.inner.render(cam_to_nre, tau=tau, actor_time_us=actor_time_us)
        if self.on and actor_time_us is not None and self.t_lo <= actor_time_us <= self.t_hi:
            x0, y0, x1, y1 = self.roi
            img = img.copy()
            img[y0:y1 + 1, x0:x1 + 1] = 255
        return img, a, ms


def changed(a, b, box):
    import numpy as np
    x0, y0, x1, y1 = box
    d = np.abs(a[y0:y1 + 1, x0:x1 + 1].astype(np.int32) - b[y0:y1 + 1, x0:x1 + 1].astype(np.int32)).max(-1) > 8
    return int(d.sum()), int(d.size)


def free_gpu():
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:                                                               # noqa: BLE001
        pass


def stub_truth_roi(r, lx, ly, length=4.5, width=1.9, height=1.6):
    """STUB REHEARSAL ONLY: where the stub double PAINTS the lead. An independent re-statement of `lead_roi` (the stub camera's own polynomial through numpy's polyval instead of
    `intr.r_of_theta`), so a wrong ROI in `lead_roi` is caught by the gate instead of being painted and read at the same wrong place."""
    import numpy as np
    pts = np.array([[lx + dx, ly + dy, z, 1.0] for dx in (-length / 2, length / 2) for dy in (-width / 2, width / 2) for z in (0.0, height)])
    pc = (np.linalg.inv(np.asarray(r.cam.T_sensor_rig, np.float64)) @ pts.T).T[:, :3]
    rho = np.hypot(pc[:, 0], pc[:, 1])
    rad = np.polynomial.polynomial.polyval(np.arctan2(rho, pc[:, 2]), np.asarray(r.cam.angle_to_pixeldist_poly, np.float64))
    u, v = r.cam.cx + rad * pc[:, 0] / np.maximum(rho, 1e-9), r.cam.cy + rad * pc[:, 1] / np.maximum(rho, 1e-9)
    return (int(max(np.floor(u.min()), 0)), int(max(np.floor(v.min()), 0)), int(min(np.ceil(u.max()), r.cam.width - 1)), int(min(np.ceil(v.max()), r.cam.height - 1)))


def render_empty(ctx, sid, c2ns, ts_us):
    """The `empty` condition: a renderer built exactly like the `objects` one but WITHOUT `attach_actors_verified`, rendered at the SAME camera poses and the same actor times
    as the `objects` reads. Built after the objects renderer is released (one scene on the GPU at a time). Returns one image per pose."""
    r, tr, _, _, _, _ = build_renderer(ctx, sid, objects=False)
    out = [r.render(c2n, actor_time_us=t)[0] for c2n, t in zip(c2ns, ts_us)]
    del r, tr
    free_gpu()
    return out


def actor(ctx):
    import numpy as np
    sid, idx = SCENES[1]
    r, tr, intr, stride, info, R = build_renderer(ctx, sid, objects=not ctx.a.stub)
    n = r.n_frames() // stride
    gt = [r.gt_rig_to_world(f * stride) for f in range(n)]
    ts_us = np.array([r.frame_timestamps_us(f * stride)[1] for f in range(n)], np.float64)
    leads = {int(f): tuple(xy) for f, xy in (ctx.manifest.get("actor_leads", {}).get(sid) or {}).items()}      # chosen on the dev box from the scene's own tracks, shipped in the manifest
    if ctx.a.stub and len(leads) < 3:
        leads = {f: (15.0, 0.0) for f in (60, 90, 120)}
    if len(leads) < 3:
        return {"verdict": "FAIL", "reason": f"NO_LEAD_TICKS: only {len(leads)} lead ticks in the manifest for {sid}"}
    if max(leads) >= n:
        return {"verdict": "FAIL", "reason": f"NO_LEAD_TICKS: lead tick {max(leads)} is outside the {n} ticks of the renderer's scene"}
    reads, c2ns, ticks = {}, [], sorted(leads)
    for f in ticks:
        lx, ly = leads[f]
        roi = lead_roi(intr, r.cam.T_sensor_rig, lx, ly)
        r_use = ActorDouble(r, stub_truth_roi(r, lx, ly), float(ts_us[0]), float(ts_us[-1])) if ctx.a.stub else r
        c2n = tr.r.rig.world_to_nre @ (gt[f] @ r.cam.T_sensor_rig)
        t = float(ts_us[f])
        r_use.set_actors_enabled(False)
        off, _, _ = r_use.render(c2n)
        r_use.set_actors_enabled(True)
        on, _, _ = r_use.render(c2n, actor_time_us=t)
        gone, _, _ = r_use.render(c2n, actor_time_us=float(ts_us[-1]) + 3.0e7)             # BESIDE only: the actor clock 30 s past the end of every track
        r_use.set_actors_enabled(False)
        dis, _, _ = r_use.render(c2n, actor_time_us=t)
        reads[f] = {"roi": roi, "off": off, "on": on, "gone": gone, "dis": dis}
        c2ns.append(np.array(c2n, copy=True))
    harness = None if info is None else {k: v for k, v in info.items() if k != "per_track"}
    nocv2 = R.STATE.get("nocv2")
    del r, tr, r_use
    free_gpu()
    empty = render_empty(ctx, sid, c2ns, [float(ts_us[f]) for f in ticks])                  # the DECIDING planted no-lead read: condition `empty`, same pose and tick
    rows = []
    for f, em in zip(ticks, empty):
        d, roi = reads[f], reads[f]["roi"]
        n_on, area = changed(d["on"], em, roi)
        n_on_off, _ = changed(d["on"], d["off"], roi)
        n_empty, _ = changed(em, d["off"], roi)
        n_gone, _ = changed(d["gone"], d["off"], roi)
        n_dis, _ = changed(d["dis"], d["off"], roi)
        present = n_on >= ROI_PRESENT_PX and n_on >= ROI_PRESENT_FRAC * area
        rows.append({"tick": f, "lead_xy_rig_m": [float(leads[f][0]), float(leads[f][1])], "roi": list(roi), "roi_px": area, "changed_on_vs_empty": n_on, "changed_on_vs_off": n_on_off,
                     "changed_planted_empty_vs_off": n_empty, "changed_disabled": n_dis, "present": bool(present), "planted_absent": n_empty <= ROI_ABSENT_PX,
                     "disabled_exact": n_dis == 0, "beside_clock_past_end_changed": n_gone, "beside_clock_past_end_absent": n_gone <= ROI_ABSENT_PX})
    ok = sum(x["present"] for x in rows) >= 2 and all(x["planted_absent"] and x["disabled_exact"] for x in rows)
    return {"verdict": "PASS" if ok else "FAIL", "scene": sid, "rows": rows, "thresholds": {"present_px": ROI_PRESENT_PX, "present_frac": ROI_PRESENT_FRAC, "absent_px": ROI_ABSENT_PX},
            "deciding_planted_read": "condition empty at the same pose and tick as the objects read; the 30 s clock read is beside and never decides", "harness_falsifier": harness, "nocv2": nocv2}


# ============================================================================================================================ 5 plumbing
def plumbing(ctx):
    cells = {}
    for sid, idx in SCENES:
        f0 = str(ctx.manifest["plumbing_starts"][sid])
        argv = [f"{HERE}/t2v8f_run.py", "matrix", "--scene-dir", ctx.scene_dir(sid), "--scene-idx", str(idx), "--scene-id", sid, "--arms", ",".join(ARMS), "--t2-conds", "T2,RR",
                "--agents", "objects", "--starts", f0, "--out", f"{ctx.pull}/plumbing"] + ctx.entry_args("C0_s0", "T2", "objects", "x", sid, idx, f0)[18:]
        if ctx.a.matrix_inproc:
            argv.append("--inproc")
        rc = ctx.run_child(argv, f"{ctx.pull}/plumbing_{sid[:8]}.log", timeout=4 * 3600)
        for tc in ("T2", "RR"):
            for arm in ARMS:
                d = f"{ctx.pull}/plumbing/{sid[:8]}/objects/{tc}/{arm}"
                cells[f"{sid[:8]}|{tc}|{arm}"] = "DONE" if os.path.exists(f"{d}/DONE") else ("FAIL" if os.path.exists(f"{d}/FAIL") else "ABSENT")
        cells[f"{sid[:8]}|matrix_rc"] = rc
    bad = [k for k, v in cells.items() if v in ("FAIL", "ABSENT")]
    return {"verdict": "PASS" if not bad else "FAIL", "cells": cells, "failed": bad, "stamp": PLUMBING, "score_on_dev_box": "t2v8f_run.py score --root <pull>/plumbing"}


# ============================================================================================================================ driver
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("all",))
    ap.add_argument("--pkg", default=os.path.dirname(HERE))
    ap.add_argument("--kit", required=True)
    ap.add_argument("--hf", required=True)
    ap.add_argument("--pull", required=True)
    ap.add_argument("--stub", action="store_true", help="CPU rehearsal: stub renderer, stub encoder, no CUDA / DINOv3 checks")
    ap.add_argument("--matrix-inproc", action="store_true", help="rehearsal only: run the plumbing cells in one process")
    ap.add_argument("--min-free-gb", type=float, default=10.0)
    a = ap.parse_args(argv)
    ctx = Ctx(a)
    with open(f"{ctx.pull}/STAGE1.running", "w", newline="\n") as f:
        f.write(f"pid={os.getpid()} start={ctx.res['started']}\n")
    for stale in ("STAGE1.done", "STAGE1.fail"):
        if os.path.exists(f"{ctx.pull}/{stale}"):
            os.replace(f"{ctx.pull}/{stale}", f"{ctx.pull}/{stale}.prev")
    crashed = None
    for name, fn in (("preflight", preflight), ("install", install_refs), ("t5", t5), ("gap", gap), ("actor", actor), ("plumbing", plumbing)):
        t0 = time.time()
        try:
            out = fn(ctx)
        except Exception as e:                                                      # noqa: BLE001
            out = {"verdict": "CRASH", "error": repr(e)[:400], "trace": traceback.format_exc()[-1500:]}
        out["seconds"] = round(time.time() - t0, 1)
        ctx.res["steps"][name] = out
        ctx.res["gates"][name] = out["verdict"]
        ctx.save()
        ok = out["verdict"] not in ("REFUSED", "CRASH")
        marker(ctx, name, ok, f"{out['verdict']} {json.dumps(out.get('failures') or out.get('reason') or '')[:600]}")
        if name in ("preflight", "install") and not ok:
            crashed = f"{name} REFUSED: {json.dumps(out.get('failures'))[:800]}"
            break
        if out["verdict"] == "CRASH":
            crashed = crashed or f"{name} CRASH: {out.get('error') or out.get('rcs')}"
    ctx.res["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    ctx.res["overall"] = "INFRASTRUCTURE_FAILURE" if crashed else ("ALL_GATES_PASS" if all(v in ("PASS", "REPORTED") for v in ctx.res["gates"].values()) else "GATE_FAILED")
    ctx.save()
    os.remove(f"{ctx.pull}/STAGE1.running")
    marker(ctx, "STAGE1", crashed is None, (crashed or f"{ctx.res['overall']} " + json.dumps(ctx.res["gates"])))
    print(json.dumps({"overall": ctx.res["overall"], "gates": ctx.res["gates"]}))
    return 1 if crashed else 0


if __name__ == "__main__":
    sys.exit(main())
