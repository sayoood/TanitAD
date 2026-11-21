"""detres_probe.py -- the numerics-sensitivity probe of `eval_box3d_det_ap2_automobile_0_20` (PREREG_DETECTION_RESIDUAL.md, EvalFlyWheel 2026-10-16).

Three phases, ONE script (new file; imports the refcv8 kit and the refcv7 battery modules, edits none of them):
  --phase build   CPU. Build the model + eval dataset exactly as G0 does, collate the 8 in-run batches (128 windows) into an mmap cache on LOCAL disk
                  (`--batch-cache`, default C:/Users/Admin/qland/work/detres/batches). Guarded: a watchdog thread ends THIS process if the host's free
                  commit falls below `--commit-floor-gb`, so a running GPU job of someone else is never put at risk. Needs no GPU, no lock.
  --phase smoke   CPU, tiny (3 windows): runs arms A0 / A1 / A2 / A4 end to end on the cached batch -> code-path check only, no number is a result.
  --phase run     GPU (RTX 4060). REFUSES to start unless env DETRES_LOCK_HELD=1 (set by run_detres_gpu.sh AFTER gpu_lock.py acquire). Runs the arms of
                  `--arms` on the 128 windows, banks the result json after every arm, stops starting new arms when `--budget-s` would be exceeded.

Arms (PREREG): A0 = G0's condition (must read 0.31453); A1 = fp32 eager (wrapper P3: trunk fp32 + NCHW, TF32 off, deterministic); A2:<seed> = trunk.net
parameters x (1+u), u ~ U(-2^-8, 2^-8); A4 = micro 8,8 (in-run backbone chunk shapes); A3 (Inductor) is impossible on this box and is reported as SKIPPED.
Assert on the json, never on the exit code.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
KEY = "eval_box3d_det_ap2_automobile_0_20"
X_INRUN, M_G0, D_STAR = 0.33835, 0.31453, 0.02382
PERM_SHA = "99adbc7ed285236adc1098cf5eb9dbccfb04a2ce3307fb977a292dc4dfbf6975"
CKPT_MD5 = "a32e3b70f46aabbd3ab55f285d03f82d"
G0_JSON = "E:/refcv8_eval_kit/battery/step5000/g0.json"
PKG = Path("C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery")
WORK = Path("C:/Users/Admin/qland/work/detres")


def free_commit_gb() -> float:
    class MS(ctypes.Structure):
        # MEMORYSTATUSEX: dwLength, dwMemoryLoad, ullTotalPhys, ullAvailPhys, ullTotalPageFile, ullAvailPageFile, ullTotalVirtual, ullAvailVirtual, ullAvailExtendedVirtual
        _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("tphys", ctypes.c_ulonglong), ("aphys", ctypes.c_ulonglong),
                    ("tpf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong), ("tvirt", ctypes.c_ulonglong), ("avirt", ctypes.c_ulonglong),
                    ("x", ctypes.c_ulonglong)]
    m = MS()
    m.l = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.apf / 2 ** 30                               # available page file = free COMMIT


def start_commit_watchdog(floor_gb: float, tag: str):
    def run():
        while True:
            f = free_commit_gb()
            if f < floor_gb:
                print(f"[detres] COMMIT WATCHDOG: free commit {f:.2f} GB < floor {floor_gb} GB -> ending THIS process ({tag})", flush=True)
                try:
                    (WORK / "WATCHDOG_KILLED.txt").write_text(f"{time.strftime('%FT%T')} {tag} free commit {f:.2f} GB < {floor_gb}\n")
                finally:
                    os._exit(86)
            time.sleep(1.0)
    threading.Thread(target=run, daemon=True).start()


def md5_file(p, n=1 << 22) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(n), b""):
            h.update(c)
    return h.hexdigest()


def displacement_stats(pa_list, pb_list, ci) -> dict:
    """per-slot (dx, dy, dlogit) of automobile-argmax slots with p>=0.01 in arm A vs arm B (same windows)."""
    import numpy as np
    pool = []
    for pa, pb in zip(pa_list, pb_list):
        p = 1.0 / (1.0 + np.exp(-pa["logit"].astype(np.float64)))
        m = (pa["cls"] == ci) & (p >= 0.01)
        pool.append(np.concatenate([(pb["xy"] - pa["xy"])[m], (pb["logit"] - pa["logit"])[m][:, None]], 1))
    pool = np.concatenate(pool) if pool else np.zeros((0, 3))
    if not len(pool):
        return {"n_slots": 0}
    nx = np.linalg.norm(pool[:, :2], axis=1)
    q = lambda v, p: float(np.quantile(v, p))      # noqa: E731
    return {"n_slots": int(len(pool)), "xy_norm_m": {"p50": q(nx, .5), "p90": q(nx, .9), "p99": q(nx, .99), "max": float(nx.max())},
            "dlogit_abs": {"p50": q(np.abs(pool[:, 2]), .5), "p90": q(np.abs(pool[:, 2]), .9), "p99": q(np.abs(pool[:, 2]), .99)},
            "n_slots_moved_gt_0.1m": int((nx > 0.1).sum())}


def judge_row(G, inrun: dict, row: dict, cls_of: dict) -> dict:
    """The G0 judge's tolerance rule applied to ONE arm row vs the in-run row, for the classes the PREREG names (SMOOTH, DETECTION with support >= 30).
    No A6 phi rescue (none was used in the step-5,000 G0: a6_rescued == [])."""
    out_det, out_smooth, n_det, n_smooth, n_low = [], [], 0, 0, 0
    for k, c in cls_of.items():
        if c not in ("SMOOTH", "DETECTION") or k not in inrun:
            continue
        x, y = inrun.get(k), row.get(k)
        if x is None or y is None or x != x or y != y:
            continue
        x, y = float(x), float(y)
        if c == "DETECTION":
            sup = G.detection_support(k, inrun)
            if sup is not None:
                if sup < G.A2_MIN_SUPPORT:
                    n_low += 1
                    continue
                ok = abs(y - x) <= max(0.02, 2.0 / sup)
            else:
                ok, _d, _t = G.tol_ok(c, k, x, y)
            n_det += 1
            if not ok:
                out_det.append(k)
        else:
            ok, _d, _t = G.tol_ok(c, k, x, y)
            n_smooth += 1
            if not ok:
                out_smooth.append(k)
    return {"n_detection_judged": n_det, "n_smooth_judged": n_smooth, "n_lowsupport_skipped": n_low,
            "detection_out": out_det[:20], "n_detection_out": len(out_det), "smooth_out": out_smooth[:20], "n_smooth_out": len(out_smooth),
            "bar_clean": (set(out_det) <= {KEY}) and not out_smooth}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("build", "smoke", "run", "cpu"), required=True)
    ap.add_argument("--ckpt", default="E:/refcv8_eval_kit/ckpt/ckpt_5000.pt")
    ap.add_argument("--config", default="E:/refcv8_eval_kit/ckpt/config.json")
    ap.add_argument("--metrics", default=None, help="the metrics copy with exactly one eval row at step 5000 (default: pick_metrics)")
    ap.add_argument("--batch-cache", default=str(WORK / "batches"))
    ap.add_argument("--arms", default="A0,A1,A2:1001")
    ap.add_argument("--micro", default="2,2,3,3,3,3")
    ap.add_argument("--budget-s", type=float, default=1140.0, help="GPU wall budget for the arms (the 20-minute rule, minus setup)")
    ap.add_argument("--commit-floor-gb", type=float, default=2.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    device = "cuda" if a.phase == "run" else "cpu"       # build / smoke / cpu are CPU phases
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif os.environ.get("DETRES_LOCK_HELD") != "1":
        print("detres:refused: --phase run needs DETRES_LOCK_HELD=1 (run_detres_gpu.sh sets it after gpu_lock.py acquire)")
        return 2
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    WORK.mkdir(parents=True, exist_ok=True)
    out_p = Path(a.out) if a.out else (PKG / "raw/step5000/detection_residual_probe.json" if a.phase == "run"
                                       else WORK / f"detres_{a.phase}.json")
    t0 = time.time()
    rec: dict = {}
    if out_p.exists() and a.phase != "build" and not a.redo:
        try:
            rec = json.loads(out_p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            rec = {}
    rec.setdefault("arms", {})
    rec.update(tool="detres_probe.py", phase=a.phase, last_started=time.strftime("%FT%T"), prereg="raw/step5000/PREREG_DETECTION_RESIDUAL.md",
               key=KEY, constants={"in_run_X": X_INRUN, "G0_mean_M": M_G0, "D_star": D_STAR, "judge_tol": max(0.02, 2 / 97)})

    def bank():
        rec["wall_s_this_invocation"] = round(time.time() - t0, 1)
        if a.phase == "run" and rec.get("arms"):
            rec["status"] = "GPU_ARMS: " + ", ".join(f"{k}={v.get('status')}" for k, v in rec["arms"].items())
        out_p.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_p.with_suffix(".part")
        tmp.write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
        os.replace(tmp, out_p)

    try:
        if a.phase in ("build", "smoke", "cpu"):
            start_commit_watchdog(a.commit_floor_gb, a.phase)
        import r8_repoint
        L = r8_repoint.apply()
        import numpy as np
        import torch
        import g0_refcv7 as G
        import wrapper_probe_r7 as W
        from microbatch import MicroBatchForward
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
        tr = L.trainer()
        config = L.load_config(a.config)
        rec.update(torch=torch.__version__, device=device, tanitad_file=__import__("tanitad").__file__,
                   gpu=(torch.cuda.get_device_name(0) if device == "cuda" else None), ckpt=a.ckpt, ckpt_md5=md5_file(a.ckpt),
                   config_md5=md5_file(a.config), free_commit_gb_at_start=round(free_commit_gb(), 2),
                   tf32_at_start={"matmul": torch.backends.cuda.matmul.allow_tf32, "cudnn": torch.backends.cudnn.allow_tf32})
        if rec["ckpt_md5"] != CKPT_MD5:
            raise RuntimeError(f"ckpt md5 {rec['ckpt_md5']} != {CKPT_MD5}")
        model, cfg, args, mrec = L.build_model(config, a.ckpt, device)
        sd = mrec["state_dict"]
        rec["model"] = {"step": sd.get("step"), "missing": len(sd.get("missing") or []), "unexpected": len(sd.get("unexpected") or []),
                        "trunk_memory_levers_built": mrec.get("trunk_memory_levers_built"), "mode": mrec.get("mode"),
                        "decoder_steps": mrec.get("decoder_steps"), "sampler": mrec.get("sampler")}
        tf = G.training_flags(model)
        G.set_flags(model, tf["flags"])
        law_idx = int(tr.LAW_AHEAD) - 1
        B, NB = int(args.batch), int(args.eval_batches)
        root = Path(a.batch_cache)
        makers = None
        if a.phase == "build":
            e_ds, _eps, drec = L.build_eval_dataset(model, cfg, args, config, with_perception_targets=True,
                                                    dataset_cls=G.make_g0_dataset_cls(tr, int(tr.LAW_AHEAD)))
            perm = L.inrun_eval_perm(e_ds, NB, B)
            sha = hashlib.sha256(json.dumps(perm).encode()).hexdigest()
            rec["perm_sha256"] = sha
            rec["dataset"] = {k: drec.get(k) for k in ("n_episodes", "n_windows")}
            if sha != PERM_SHA:
                raise RuntimeError(f"perm_sha256 {sha} != the G0 subset {PERM_SHA}")
            t_c = time.time()
            makers = [lambda i=i: G.collate(e_ds, perm[i * B:(i + 1) * B], law_idx) for i in range(NB)]
            cached = G.MmapBatches(root, makers)
            rec["collate_s"] = round(time.time() - t_c, 1)
            rec["batch_cache"] = str(root)
            rec["cache_bytes"] = sum(p.stat().st_size for p in root.glob("batch_*.pt"))
            rec["peak_free_commit_note"] = f"free commit now {free_commit_gb():.2f} GB"
            rec["BUILD_OK"] = True
            bank()
            print("DETRES_BUILD_OK", rec["collate_s"], rec["cache_bytes"], flush=True)
            return 0
        # smoke / run: batches come from the cache ONLY (a missing file must fail loud, never rebuild)
        def _missing(i):
            raise RuntimeError(f"batch_{i}.pt missing in {root}: run --phase build first")
        cached = G.MmapBatches(root, [lambda i=i: _missing(i) for i in range(NB)])
        rec["batch_cache"] = str(root)
        G.patch_frames_to_device(tr)
        mode = getattr(args, "mode", "diffusion")
        abl = bool(getattr(args, "ablate_frames", False))
        from tanitad.models.timm_trunk import TimmResNetTrunk
        trunk = [m for m in model.modules() if isinstance(m, TimmResNetTrunk)][0]
        from tanitad.models.agent_slots import AGENT_CLASSES
        ci = list(AGENT_CLASSES).index("automobile")
        cls_of = {k: v.get("cls") for k, v in json.load(open(G0_JSON, encoding="utf-8"))["verdict_A7"]["terms"].items()}
        inrun = None
        metrics = a.metrics
        if not metrics:
            import subprocess
            metrics = subprocess.run([sys.executable, "-I", str(HERE / "pick_metrics.py"), "E:/refcv8_eval_kit/thor_reads", "5000"],
                                     capture_output=True, text=True).stdout.strip()
        rows = [json.loads(ln) for ln in open(metrics, encoding="utf-8") if ln.strip()]
        ev = [r for r in rows if r.get("step") == 5000 and "eval_loss" in r]
        if len(ev) != 1:
            raise RuntimeError(f"{len(ev)} eval rows at step 5000 in {metrics}")
        inrun = ev[0]
        rec["metrics_copy"] = metrics
        rec["in_run_value"] = inrun.get(KEY)
        if inrun.get(KEY) != X_INRUN:
            raise RuntimeError(f"in-run {KEY} {inrun.get(KEY)} != {X_INRUN}")
        smoke = a.phase == "smoke"
        ref_arm = "C0" if a.phase == "cpu" else "A0"
        if smoke:
            sb = G.sub_batch(cached[0], 3)
            batches = lambda: iter([sb])      # noqa: E731
            sizes0, sizes_a4, B_eff = [1, 2], [3], 3
        else:
            batches = lambda: iter(cached)    # noqa: E731
            sizes0, sizes_a4, B_eff = [int(x) for x in a.micro.split(",")], [8, 8], B      # run and cpu phases: the 8 cached batches
        rec["micro_A0"] = sizes0
        mb = MicroBatchForward(model, sizes0).install()
        bd0 = G.buffer_digest(model)
        bank()
        arms = [x.strip() for x in a.arms.split(",") if x.strip()]
        packs_a0 = None
        t_start = time.time()
        last_wall = 260.0
        gcfg = {"tf32": (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                         torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)}
        # A0's packs are needed as the displacement reference: if A0 is already banked in a previous invocation its npz is on disk
        p_a0 = WORK / f"packs_{ref_arm}.npz"
        for arm in arms:
            if arm in rec["arms"] and rec["arms"][arm].get("status") == "DONE" and not a.redo:
                print(f"[detres] {arm} already DONE", flush=True)
                continue
            if arm.startswith("A3"):
                rec["arms"][arm] = {"status": "SKIPPED", "why": "Inductor needs Triton (not installed) or a C++ compiler (cl/gcc/clang absent); "
                                    "`import triton` -> ModuleNotFoundError on this box"}
                bank()
                continue
            if not smoke and (time.time() - t_start) + last_wall * 1.05 > a.budget_s:
                rec["arms"][arm] = {"status": "NOT_RUN", "why": f"GPU budget {a.budget_s:.0f}s: {time.time() - t_start:.0f}s used, last arm {last_wall:.0f}s"}
                print(f"[detres] {arm} NOT_RUN (budget)", flush=True)
                bank()
                continue
            t_a = time.time()
            ent: dict = {"status": "RUNNING"}
            rec["arms"][arm] = ent
            packs: dict = {}
            undo = []
            try:
                if arm == "A0":
                    pass
                elif arm in ("A1", "C0", ) or arm.startswith("C2:"):
                    lev0 = dict(trunk.memory_levers)
                    bk0 = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.deterministic,
                           torch.backends.cudnn.benchmark)
                    W._set_condition(W.CONDITIONS["P3_fp32_det"], trunk)

                    def undo_a1(lev0=lev0, bk0=bk0):
                        trunk.memory_levers.update(lev0)
                        (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.deterministic,
                         torch.backends.cudnn.benchmark) = bk0
                    undo.append(undo_a1)
                    ent["lever"] = "trunk fp32 + NCHW, cuDNN TF32 off, deterministic (wrapper P3)"
                if arm.startswith(("A2:", "C2:")):
                    seed = int(arm.split(":")[1])
                    gen = torch.Generator(device="cpu").manual_seed(seed)
                    orig = {n: p.detach().clone() for n, p in trunk.net.named_parameters()}
                    num = den = 0.0
                    with torch.no_grad():
                        for n, p in trunk.net.named_parameters():
                            u = (torch.rand(p.shape, generator=gen) * 2.0 - 1.0) * (2.0 ** -8)
                            old = p.detach().clone()
                            p.mul_((1.0 + u).to(device=p.device, dtype=p.dtype))
                            num += float(((p - old).double() ** 2).sum())
                            den += float((old.double() ** 2).sum())

                    def undo_a2(orig=orig):
                        with torch.no_grad():
                            for n, p in trunk.net.named_parameters():
                                p.copy_(orig[n])
                    undo.append(undo_a2)
                    ent["lever"] = f"trunk.net params x (1+u), u~U(-2^-8,2^-8), seed {seed}"
                    ent["perturbation_rel_l2"] = (num / den) ** 0.5
                    ent["n_param_tensors"] = len(orig)
                elif arm == "A4":
                    mb.remove()
                    mb4 = MicroBatchForward(model, sizes_a4).install()
                    ent["lever"] = f"micro {sizes_a4} (backbone sees 24 frames per call = three 8-frame chunks, as in the in-run 16-window batch)"

                    def undo_a4(mb4=mb4):
                        mb4.remove()
                        mb.install()
                    undo.append(undo_a4)
                elif arm not in ("A0", "A1", "C0"):
                    raise ValueError(f"unknown arm {arm}")
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                erow, pb, npk = G.run_eval(tr, model, batches, device, mode, abl, 0, B_eff, packs_out=packs)
                v = erow.get(KEY)
                ent.update(value=v, delta_vs_A0=None, delta_vs_inrun=(None if v is None else round(v - X_INRUN, 5)),
                           inside_judge_tol_of_inrun=(None if v is None else abs(v - X_INRUN) <= max(0.02, 2 / 97)),
                           judge=judge_row(G, inrun, erow, cls_of), n_det_packs=npk,
                           row_selected={k: erow.get(k) for k in (KEY, "eval_box3d_ap2m", "eval_box3d_det_map2_all", "eval_box3d_det_ap2_automobile_all",
                                                                  "eval_box3d_det_ap2_all_0_20", "eval_box3d_det_ap2_all_all", "eval_box3d_centre_err_p50",
                                                                  "eval_traj", "eval_loss")})
                ent["ap_cells"] = {k: erow.get(k) for k in cls_of if "_det_ap" in k and k in erow}      # population view: every AP cell the G0 judged
                if device == "cuda":
                    ent["cuda_max_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
                pk = packs.get("box3d") or []
                if pk and not smoke:
                    an = arm.replace(":", "_")
                    np.savez_compressed(WORK / f"packs_{an}.npz", **G.pack_arrays(pk, f"{an}.box3d"))
                    ent["packs_file"] = str(WORK / f"packs_{an}.npz")
                if arm == ref_arm:
                    if arm == "A0":
                        ent["reproduces_G0"] = (v == M_G0)
                    packs_a0 = pk
                    rec["_a0_value"] = v
                elif packs_a0 is None and p_a0.exists() and not smoke:
                    z = np.load(p_a0, allow_pickle=False)
                    # rebuild minimal packs for the displacement statistic
                    off = z[f"{ref_arm}.box3d__gt_off"]
                    packs_a0 = [{"logit": z[f"{ref_arm}.box3d__logit"][i], "xy": z[f"{ref_arm}.box3d__xy"][i], "cls": z[f"{ref_arm}.box3d__cls"][i]}
                                for i in range(len(off) - 1)]
                if arm != ref_arm and packs_a0 is not None and pk:
                    ent["displacement_vs_A0"] = displacement_stats(packs_a0, pk, ci)
                a0v = (rec["arms"].get(ref_arm) or {}).get("value")
                if arm != ref_arm and a0v is not None and v is not None:
                    ent["delta_vs_A0"] = round(v - a0v, 5)
                ent["buffers_unchanged"] = (G.buffer_digest(model) == bd0)
                ent["status"] = "DONE"
            except BaseException as exc:                      # noqa: BLE001 -- recorded; OOM on A4 is an expected possibility
                ent["status"] = "ERROR"
                ent["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
                ent["traceback_tail"] = traceback.format_exc()[-800:]
                if device == "cuda":
                    torch.cuda.empty_cache()
            finally:
                for u_ in reversed(undo):
                    try:
                        u_()
                    except Exception as exc:                  # noqa: BLE001
                        ent["undo_error"] = f"{type(exc).__name__}: {exc}"
            ent["wall_s"] = round(time.time() - t_a, 1)
            last_wall = ent["wall_s"] if ent["status"] == "DONE" else last_wall
            print(f"[detres] arm {arm}: {ent['status']} value={ent.get('value')} d_A0={ent.get('delta_vs_A0')} ({ent['wall_s']}s)", flush=True)
            bank()
        rec["buffers_unchanged_final"] = (G.buffer_digest(model) == bd0)
        rec["ok"] = True
    except BaseException as exc:                              # noqa: BLE001 -- the json is the result
        rec["ok"] = False
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        rec["traceback_tail"] = traceback.format_exc()[-1500:]
    rec.pop("_a0_value", None)
    bank()
    print(f"detres:{a.phase}:{'ok' if rec.get('ok') or rec.get('BUILD_OK') else 'fail:' + str(rec.get('error'))}")
    return 0 if (rec.get("ok") or rec.get("BUILD_OK")) else 1


if __name__ == "__main__":
    sys.exit(main())
