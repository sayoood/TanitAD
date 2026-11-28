#!/usr/bin/env python3
"""Measure 5 effectiveness run (eval/PREREG_MEASURE5.md §3a, §6): ONE snapshot-015 forward per sample writes the SHARED
trunk cache AND the sample's 64 proposals + logits. GPU only inside the scheduled window.

  navtrain (splits train, val): the inputs are train.TargetBank.__getitem__'s -- the trainer's OWN construction (4
      camera frames from the transferred loose files, ego, goal, the per-sample rig from the locally built calibration,
      equal to the pod's for every log) -- and the numerics are onpolicy_dump.py's (eval, no grad, bf16 autocast + TF32,
      batch 4). Every 16 samples a props chunk in `onpolicy_dump.py --emit-logits`'s line format goes to <m5>/queue
      for the v4 labeller (proposals and logits rounded to 5 dp, as the dump writes them).
  navtest (W3's 200 tokens): REFePlanner.infer, the planner's own path (fp32, per-sample rig from the test DB), exactly
      as stop_candidate_probe.py / slow_copies.py capture the scoring context.
  SHARED CACHE (§6): <cache>/<split>/shard_NNNNN/ holding one .npy per array (a memmap-able container for §6's
      content): key, visual_ctx (the SCORER's K/V, model.py:732), scene_ctx (the TRAJECTORY decoder's K/V, :730-731),
      q0 (queries + ego token, :739), ego, goal, calib, teacher, props, logits. bf16 tensors are stored as their exact
      16-bit patterns (int16, dtype tag "bf16bits"), so nothing is rounded.
  GATES per shard (before it is used): the decoders re-run from the RELOADED arrays reproduce props and logits
      (max |d| <= 1e-3); ANOTHER sample's context must FAIL that (the control). navtest also: props ==
      stop_candidate_dump.npz `traj` and logits == the E-6 table's, bit for bit (the STOP probe's C_a / C_b).

    python eval/m5_forward.py timing --limit 20          (the 20-sample timing gate; writes nothing to the cache)
    python eval/m5_forward.py navtrain --split train
    python eval/m5_forward.py navtest
    python eval/m5_forward.py navtrain --split val
    (--device cpu --no-amp --limit N: a CPU code-path test into a separate --cache-root)
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REFE = os.path.join(os.path.dirname(HERE), "refe")
sys.path.insert(0, HERE)
sys.path.insert(0, REFE)

CKPT = "D:/Projects/TanitAD/data/refe_runs_eval/snap_epoch015.pt"
CKPT_MD5 = "d7c59f4f2fbcbde3e2dec8f67d63a7e7"
CKPT_STEP = 4933
M5 = "D:/Projects/TanitAD/data/refe_m5"
CACHE_ROOT = f"D:/Projects/TanitAD/data/refe_trunk_cache/{CKPT_MD5[:8]}"
NAVTEST_TOKENS = ("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest/raw/"
                  "A1_sub200_tokens.json")
TABLE = "D:/Projects/TanitAD/data/refe_navtest/proptable/sub200_ep015/table.npz"
STOP_DUMP = "D:/Projects/TanitAD/data/refe_navtest/proptable/sub200_ep015/stop_candidate_dump.npz"
SHARD, CHUNK = 64, 16
# 4 = onpolicy_dump.py's batch; M5_FWD_BATCH is the chain's DECLARED fallback when the capped allocator OOMs
BATCH = int(os.environ.get("M5_FWD_BATCH", "4"))


# ------------------------------------------------------------------------------------------ helpers
def past(deadline: str) -> bool:
    """Local HH:MM reached? Clock times before 12:00 count as 'after midnight tonight' on BOTH sides, so a window that
    crosses midnight compares correctly (a plain string compare says "23:30" >= "04:28")."""
    def m(h, mi):
        v = h * 60 + mi
        return v + 1440 if v < 720 else v
    hh, mm = (int(x) for x in deadline.split(":"))
    lt = time.localtime()
    return m(lt.tm_hour, lt.tm_min) >= m(hh, mm)


def vram_cap(device) -> None:
    """M5_VRAM_CAP_MIB (set by eval/m5_gpu_chain.py): cap this process's torch allocator, so an overrun raises an OOM
    instead of the Windows sysmem fallback silently spilling into system RAM (MEASURED on this box 2026-09-27)."""
    cap = os.environ.get("M5_VRAM_CAP_MIB")
    if device.type == "cuda" and cap:
        tot = torch.cuda.get_device_properties(0).total_memory
        frac = min(1.0, max(0.05, float(cap) * 2 ** 20 / tot))
        torch.cuda.set_per_process_memory_fraction(frac, 0)
        print(f"  VRAM cap {float(cap):.0f} MiB = fraction {frac:.3f} of {tot / 2 ** 20:.0f} MiB", flush=True)


def to_store(t: torch.Tensor):
    """(array, tag): bf16 -> its exact 16-bit pattern; anything else -> float32/float64 as is."""
    t = t.detach()
    if t.dtype == torch.bfloat16:
        return t.contiguous().view(torch.int16).cpu().numpy(), "bf16bits"
    if t.dtype == torch.float64:
        return t.cpu().numpy(), "float64"
    return t.float().cpu().numpy(), "float32"


def from_store(a: np.ndarray, tag: str, device) -> torch.Tensor:
    t = torch.from_numpy(np.array(a, copy=True))            # a writable copy of a (possibly memmapped) slice
    return (t.view(torch.bfloat16) if tag == "bf16bits" else t).to(device)


class Capture:
    """Forward pre-hooks: the trajectory decoder's (q0, scene_ctx) and the scoring decoder's context."""

    def __init__(self, model):
        self.on, self.v = False, {}
        self.h = [model.dec[0].register_forward_pre_hook(self._dec),
                  model.score_dec[0].register_forward_pre_hook(self._sdec)]

    def _dec(self, _m, args):
        if self.on:
            self.v["q0"], self.v["scene_ctx"] = args[0], args[1]

    def _sdec(self, _m, args):
        if self.on:
            self.v["visual_ctx"] = args[1]


def rerun(model, q0, scene_ctx, visual_ctx, cfg):
    """The two decoders from cached tensors (model.forward's lines 740-755, same modules)."""
    q = q0
    for blk in model.dec:
        q = blk(q, scene_ctx)
    traj = model.traj_head(q).view(q0.shape[0], cfg.n_proposals, cfg.horizon_steps, cfg.traj_dim)
    return traj, model.score_trajectories(traj, visual_ctx)


def load_model(device):
    import ckpt_io
    from model import REFe, REFeConfig
    cfg = REFeConfig.for_backbone("vitl16")
    model = REFe(cfg)
    meta: dict = {}
    fmt = ckpt_io.load_for_inference(model, CKPT, map_location="cpu", backbone="vitl16", meta_out=meta)
    return model.to(device).eval(), cfg, meta, fmt


def write_shard(sdir: str, arrays: dict, tags: dict, meta: dict) -> None:
    tmp = sdir + ".tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    for k, v in arrays.items():
        np.save(os.path.join(tmp, f"{k}.npy"), v)
    json.dump(dict(meta, tags=tags), open(os.path.join(tmp, "meta.json"), "w", encoding="utf-8"), indent=1)
    if os.path.exists(sdir):
        shutil.rmtree(sdir)
    os.replace(tmp, sdir)


def shard_gate(model, cfg, sdir: str, traj_ref, logits_ref, device, amp: bool) -> dict:
    """Reload the shard and re-run both decoders; another sample's context must FAIL."""
    tags = json.load(open(os.path.join(sdir, "meta.json"), encoding="utf-8"))["tags"]
    L = {k: np.load(os.path.join(sdir, f"{k}.npy"), mmap_mode="r") for k in ("q0", "scene_ctx", "visual_ctx")}
    n = L["q0"].shape[0]
    d_traj = d_log = 0.0
    d_ctrl = 0.0
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
        for b0 in range(0, n, BATCH):
            sl = slice(b0, min(n, b0 + BATCH))
            q0, sc, vc = (from_store(L[k][sl], tags[k], device) for k in ("q0", "scene_ctx", "visual_ctx"))
            tr, lg = rerun(model, q0, sc, vc, cfg)
            d_traj = max(d_traj, float((tr.float().cpu() - traj_ref[sl]).abs().max()))
            d_log = max(d_log, float((lg.float().cpu() - logits_ref[sl]).abs().max()))
            if b0 == 0:
                # CONTROL: another sample's context (rotate the batch axis; a 1-sample batch rotates channels).
                # ⚠ Rolling along the TOKEN axis is NOT a control: the cross-attention is a set operation over the
                # context tokens (positions are already inside them), so a token permutation leaves the logits
                # exactly unchanged -- MEASURED 0.00 on the first CPU test of this gate.
                bad = torch.roll(vc, shifts=1, dims=0) if vc.shape[0] > 1 else torch.roll(vc, shifts=7, dims=2)
                _, lg2 = rerun(model, q0, sc, bad, cfg)
                d_ctrl = float((lg2.float().cpu() - logits_ref[sl]).abs().max())
    return {"max_abs_traj": d_traj, "max_abs_logits": d_log, "swapped_ctx_max_abs_logits": d_ctrl,
            "pass": d_traj <= 1e-3 and d_log <= 1e-3 and d_ctrl > 1e-3}


def props_line(r: dict, traj: np.ndarray, lg: np.ndarray) -> str:
    """onpolicy_dump.py --emit-logits's line, field for field."""
    rec = {"kind": "onpolicy_props", "log_name": r["log_name"], "token": r["token"],
           "step": int(r.get("step", 0)), "rank": int(r.get("rank", 0)), "ckpt_step": CKPT_STEP,
           "teacher": r["traj"], "aug": r.get("aug"),
           "props": [[[round(float(v), 5) for v in p] for p in traj[k]] for k in range(traj.shape[0])]}
    rec["logits"] = [[round(float(v), 5) for v in lg[k]] for k in range(lg.shape[0])]
    return json.dumps(rec) + "\n"


def bank_dir(split: str, limit: int) -> str:
    """A TargetBank directory holding exactly this split's gated rows (the source file's lines, unchanged)."""
    import m5_transfer as TX
    G = json.load(open(os.path.join(M5, "samples_gated.json"), encoding="utf-8"))
    want = [x["key"] for x in G[split]][:limit or None]
    d = os.path.join(M5, f"bank_{split}" + (f"_lim{limit}" if limit else ""))
    os.makedirs(d, exist_ok=True)
    rows = TX.rows_by_key(G["source"], set(want))
    with open(os.path.join(d, "targets_rank0.jsonl"), "w", encoding="utf-8") as f:
        for k in want:
            f.write(json.dumps(rows[k]) + "\n")
    return d


# ------------------------------------------------------------------------------------------ navtrain
def navtrain(a) -> int:
    import train as T
    device = torch.device(a.device)
    vram_cap(device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True            # the trainer's --tf32 (onpolicy_dump.py)
        torch.backends.cudnn.allow_tf32 = True
    model, cfg, meta, fmt = load_model(device)
    cap = Capture(model)
    ds = T.TargetBank(bank_dir(a.split, a.limit), os.path.join(M5, "pixels"), cfg,
                      calib=os.path.join(M5, "calib_table.json"))
    n = len(ds)
    out = os.path.join(a.cache_root, f"navtrain_{a.split}")
    os.makedirs(out, exist_ok=True)
    q = os.path.join(M5, "queue" if not a.test else "queue_test")
    os.makedirs(q, exist_ok=True)
    done = {d for d in os.listdir(out) if d.startswith("shard_") and not d.endswith(".tmp")}
    t_all, n_new = time.time(), 0
    split_code = {"train": 1, "val": 2}[a.split]

    def emit(rows_, tr_np, lg_np, lo, hi, shard_i, chunk_i):
        """One props chunk (deterministic name: a re-run re-emits the same file; the labeller skips done samples)."""
        text = "".join(props_line(rows_[j], tr_np[j], lg_np[j]) for j in range(lo, hi))
        cp = os.path.join(q, f"props_r0_{CKPT_STEP:06d}_{split_code}{shard_i:05d}{chunk_i:02d}.jsonl")
        with open(cp + ".tmp", "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(cp + ".tmp", cp)

    for s0 in range(0, n, SHARD):
        sname = f"shard_{s0 // SHARD:05d}"
        if sname in done and not a.timing:
            continue
        idx = list(range(s0, min(n, s0 + SHARD)))
        acc = {k: [] for k in ("q0", "scene_ctx", "visual_ctx", "traj", "logits")}
        tags, keys, rows = {}, [], []
        sent, n_chunk = 0, 0
        ts = time.time()
        # the next batch's frames decode on a thread while the GPU runs this one (cv2 releases the GIL); the
        # items are EXACTLY ds[i], in order -- only the time they are fetched changes
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(max_workers=1)
        batches = [idx[b0:b0 + BATCH] for b0 in range(0, len(idx), BATCH)]
        fut = pool.submit(lambda ii: [ds[i] for i in ii], batches[0])
        for bi, b0 in enumerate(range(0, len(idx), BATCH)):
            items = fut.result()
            if bi + 1 < len(batches):
                fut = pool.submit(lambda ii: [ds[i] for i in ii], batches[bi + 1])
            img = torch.stack([it[0] for it in items]).to(device, non_blocking=True)
            ego = torch.stack([it[1] for it in items]).to(device, non_blocking=True)
            goal = torch.stack([it[2] for it in items]).to(device, non_blocking=True)
            cal = torch.stack([it[8] for it in items]) if items[0][8].numel() else None
            cap.on = True
            with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=not a.no_amp):
                traj, score = model(img, ego, goal, calib=cal)
            cap.on = False
            for k in ("q0", "scene_ctx", "visual_ctx"):
                arr, tags[k] = to_store(cap.v[k])
                acc[k].append(arr)
            acc["traj"].append(traj.float().cpu())
            acc["logits"].append(score.float().cpu())
            for i in idx[b0:b0 + BATCH]:
                r = ds.rows[i]
                rows.append(r)
                keys.append(f"{r['log_name']}|{r['token']}|{int(r.get('step', 0))}|{int(r.get('rank', 0))}")
            if device.type == "cuda":
                torch.cuda.synchronize()
            # props chunks every CHUNK samples, as soon as they exist (the labellers start at once)
            if not a.timing and len(rows) - sent >= CHUNK:
                tr_np, lg_np = torch.cat(acc["traj"]).numpy(), torch.cat(acc["logits"]).numpy()
                while len(rows) - sent >= CHUNK:
                    emit(rows, tr_np, lg_np, sent, sent + CHUNK, s0 // SHARD, n_chunk)
                    sent, n_chunk = sent + CHUNK, n_chunk + 1
        pool.shutdown(wait=True)
        sec = time.time() - ts
        traj_ref = torch.cat(acc["traj"])
        logits_ref = torch.cat(acc["logits"])
        if not a.timing and len(rows) > sent:
            emit(rows, traj_ref.numpy(), logits_ref.numpy(), sent, len(rows), s0 // SHARD, n_chunk)
        if a.timing:
            print(f"  TIMING {len(idx)} samples in {sec:.1f} s = {sec / len(idx):.3f} s/sample; GPU peak "
                  f"{torch.cuda.max_memory_allocated() / 2**30 if device.type == 'cuda' else 0:.2f} GB; "
                  f"tags {tags}; projected 1,600 train {1600 * sec / len(idx) / 60:.1f} min, 300 val "
                  f"{300 * sec / len(idx) / 60:.1f} min", flush=True)
            print("ZZM5_TIMING_OK")
            return 0
        arrays = {"key": np.array(keys), "q0": np.concatenate(acc["q0"]), "scene_ctx": np.concatenate(acc["scene_ctx"]),
                  "visual_ctx": np.concatenate(acc["visual_ctx"]),
                  "ego": np.stack([np.asarray(r["ego"], np.float32) for r in rows]),
                  "goal": np.stack([np.asarray(r["goal"], np.float32)[:2 * cfg.n_goal_points] for r in rows]),
                  "calib": np.stack([np.asarray(r["_calib"], np.float64) for r in rows]),
                  "teacher": np.stack([np.asarray(r["traj"], np.float32) for r in rows]),
                  "props": traj_ref.numpy(), "logits": logits_ref.numpy()}
        sdir = os.path.join(out, sname)
        write_shard(sdir, arrays, tags, {"split": a.split, "n": len(idx), "sec": round(sec, 1),
                                         "numerics": "bf16 autocast + TF32, eval, no grad" if not a.no_amp else "fp32"})
        g = shard_gate(model, cfg, sdir, traj_ref, logits_ref, device, amp=not a.no_amp)
        m = json.load(open(os.path.join(sdir, "meta.json"), encoding="utf-8"))
        m["gate"] = g
        json.dump(m, open(os.path.join(sdir, "meta.json"), "w", encoding="utf-8"), indent=1)
        n_new += len(idx)
        print(f"  {a.split} {sname}: {len(idx)} samples in {sec:.0f} s ({sec / len(idx):.2f} s/sample); gate "
              f"traj {g['max_abs_traj']:.2e} logits {g['max_abs_logits']:.2e} swapped-ctx {g['swapped_ctx_max_abs_logits']:.2f} "
              f"-> {'PASS' if g['pass'] else 'FAIL'}; local {time.strftime('%H:%M:%S')}", flush=True)
        if not g["pass"]:
            print("ZZM5_FWD_FAIL shard gate")
            return 1
        if a.deadline and past(a.deadline):
            print(f"ZZM5_FWD_STOPPED deadline {a.deadline} reached after {sname}")
            return 2
    write_manifest(out, a.split, cfg, meta, fmt, "navtrain")
    open(os.path.join(q, f"FORWARD_DONE_{a.split}"), "w").close()      # eval/m5_label.py stops its workers on it
    print(f"ZZM5_FWD_OK navtrain {a.split}: {n_new} new samples in {time.time() - t_all:.0f} s")
    return 0


def write_manifest(out, split, cfg, meta, fmt, source):
    import hashlib
    keys = []
    for s in sorted(d for d in os.listdir(out) if d.startswith("shard_") and not d.endswith(".tmp")):
        keys += [str(k) for k in np.load(os.path.join(out, s, "key.npy"))]

    def sha(p):
        return hashlib.sha256(open(p, "rb").read()).hexdigest()
    man = {"checkpoint": CKPT, "checkpoint_md5": CKPT_MD5, "checkpoint_step": CKPT_STEP, "ckpt_format": fmt,
           "ckpt_meta": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in meta.items()},
           "split": split, "source": source, "n": len(keys), "keys": keys, "shard_size": SHARD,
           "config": {k: getattr(cfg, k) for k in ("backbone", "n_cameras", "img_h", "img_w", "patch", "dec_width",
                                                  "n_registers", "n_proposals", "horizon_steps", "traj_dim")},
           "arrays": {"visual_ctx": "[n, 7680, 256] scorer K/V (model.py:732)", "scene_ctx": "[n, 64, 256] trajectory "
                      "decoder K/V (model.py:730-731)", "q0": "[n, 64, 256] queries + ego token (model.py:739)",
                      "ego": "[n, 7]", "goal": "[n, 4]", "calib": "[n, 4, 16] float64", "teacher": "[n, 20, 3]",
                      "props": "[n, 64, 20, 3]", "logits": "[n, 64, 6]"},
           "sources_sha256": {f: sha(os.path.join(REFE, f)) for f in ("model.py", "ckpt_io.py", "load_dinov3.py",
                                                                        "train.py", "planner.py")},
           "builder_sha256": sha(os.path.abspath(__file__)),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
           "at_local": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json.dump(man, open(os.path.join(out, "manifest.json"), "w", encoding="utf-8"), indent=1)


# ------------------------------------------------------------------------------------------ navtest
def navtest(a) -> int:
    import navtrain_scenarios as NS
    import refe_navtest_seam as SEAM
    from planner import REFePlanner
    from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization
    device = torch.device(a.device)
    vram_cap(device)
    T = np.load(TABLE)
    D = np.load(STOP_DUMP, allow_pickle=False)
    tok_order = [str(t) for t in T["token"]]
    exp = json.load(gzip.open(SEAM.W3_EXPORT, "rt", encoding="utf-8"))["tokens"]
    sub = json.load(open(NAVTEST_TOKENS, encoding="utf-8"))
    want = set(sub["tokens"] if isinstance(sub, dict) else sub)
    by_log: dict = {}
    for t in [t for t in exp if t in want]:
        by_log.setdefault(exp[t]["log_name"], []).append(t)
    planner = REFePlanner(checkpoint=CKPT, images_root="D:/Projects/TanitAD/data/refe_navtest/frames",
                          db_dir=SEAM.TEST_DB_DIR, backbone="vitl16", device=a.device, select="best", rule=None)
    model, cfg = planner.model, planner.model.cfg
    if planner.rule != "navsim_v1":
        print("ZZM5_FWD_FAIL planner rule"); return 1
    cap = Capture(model)
    got = {}
    t0 = time.time()
    logs = sorted(by_log.items())
    if a.limit:
        logs = logs[:max(1, a.limit)]
    for log, lt in logs:
        for sc in NS.build_scenarios_for_log(os.path.join(SEAM.TEST_DB_DIR, f"{log}.db"), lt,
                                             history_rows=1, future_rows=80):
            tok = sc._initial_lidar_token
            planner._scenario = sc
            planner.initialize(PlannerInitialization(route_roadblock_ids=sc.get_route_roadblock_ids(),
                                                     mission_goal=sc.get_mission_goal(), map_api=sc.map_api))
            ego = sc.get_ego_state_at_iteration(0)
            img = planner._image_for(ego)
            if img is None:
                continue
            cap.on = True
            traj, score, _k = planner.infer(ego, img)
            cap.on = False
            v = {k: cap.v[k] for k in ("q0", "scene_ctx", "visual_ctx")}
            got[tok] = {"traj": traj.float().reshape(64, 20, 3).cpu(), "logits": score.float().reshape(64, 6).cpu(),
                        **{k: to_store(x) for k, x in v.items()}}
        # progress (read by eval/m5_gpu_chain.py to project the end of this all-or-nothing stage)
        print(f"  NTPROG {len(got)} tokens {time.time() - t0:.1f} s local {time.strftime('%H:%M:%S')}", flush=True)
        if a.deadline and past(a.deadline):
            print(f"ZZM5_FWD_STOPPED deadline {a.deadline} reached in navtest after {len(got)} tokens")
            return 2
    order = [t for t in tok_order if t in got]
    ti = {t: i for i, t in enumerate(tok_order)}
    tr = torch.stack([got[t]["traj"] for t in order])
    lg = torch.stack([got[t]["logits"] for t in order])
    idx = [ti[t] for t in order]
    d_props = float(np.abs(tr.numpy().astype(np.float64) - D["traj"][idx].astype(np.float64)).max())
    d_log = float(np.abs(lg.numpy().astype(np.float64) - T["logits"][idx].astype(np.float64)).max())
    out = os.path.join(a.cache_root, "navtest_sub200")
    os.makedirs(out, exist_ok=True)
    tags = {k: got[order[0]][k][1] for k in ("q0", "scene_ctx", "visual_ctx")}
    for s0 in range(0, len(order), SHARD):
        part = order[s0:s0 + SHARD]
        arrays = {"key": np.array(part), "props": tr[s0:s0 + SHARD].numpy(), "logits": lg[s0:s0 + SHARD].numpy(),
                  "table_index": np.array([ti[t] for t in part])}
        for k in ("q0", "scene_ctx", "visual_ctx"):
            arrays[k] = np.concatenate([got[t][k][0] for t in part])
        sdir = os.path.join(out, f"shard_{s0 // SHARD:05d}")
        write_shard(sdir, arrays, tags, {"split": "navtest_sub200", "n": len(part), "numerics": "fp32 (the planner)"})
        g = shard_gate(model, cfg, sdir, tr[s0:s0 + SHARD], lg[s0:s0 + SHARD], device, amp=False)
        m = json.load(open(os.path.join(sdir, "meta.json"), encoding="utf-8"))
        m["gate"] = g
        json.dump(m, open(os.path.join(sdir, "meta.json"), "w", encoding="utf-8"), indent=1)
        if not g["pass"]:
            print(f"ZZM5_FWD_FAIL navtest shard gate {g}"); return 1
    write_manifest(out, "navtest_sub200", cfg, {"planner_rule": planner.rule}, planner.ckpt_format, "navtest")
    ok = d_props == 0.0 and d_log == 0.0 and len(order) == (200 if not a.limit else len(order))
    print(f"  navtest: {len(order)} tokens in {time.time() - t0:.0f} s; props vs the STOP dump max |d| {d_props}; "
          f"logits vs the E-6 table max |d| {d_log}; tags {tags}")
    print(f"ZZM5_FWD_{'OK' if ok else 'CHECK'} navtest")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("timing", "navtrain", "navtest"))
    ap.add_argument("--split", default="train", choices=("train", "val"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cache-root", default=CACHE_ROOT)
    ap.add_argument("--deadline", default="21:45", help="local HH:MM: stop after the shard that crosses it")
    ap.add_argument("--test", action="store_true", help="props chunks to queue_test (a code-path test)")
    a = ap.parse_args()
    if a.stage == "timing":
        a.timing = True
        a.limit = a.limit or 20
        return navtrain(a)
    a.timing = False
    return navtrain(a) if a.stage == "navtrain" else navtest(a)


if __name__ == "__main__":
    sys.exit(main())
