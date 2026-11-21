#!/usr/bin/env python3
"""The auxiliary drivable-area head's outputs on NAVTEST frames: ONE frozen-trunk forward, any number of head arms.

WHY.  The S1 perception reads P1 (mask IoU in longitudinal bands) and P3 (AUC of the minimum predicted SDF along each proposal's footprint corners, DAC = 0 vs 1) are
registered on navtest frames: P3 on the 1,058 C3 tokens x 64 proposals (raw/2026-10-11-c3-selection-headroom/{table.npz, tokens.json}).  `refe/scorer_finetune.py --aux-drv`
only dumps head outputs on held-out NAVTRAIN frames.  At S1 the trunk is FROZEN (scorer_finetune.py:583-588 `requires_grad_(False)` on every model parameter, then True on
SCORER = score_q_mlp / score_dec / score_head only; the optimisers are optA/optB over the scorer copies (+ H2's feed) at :1281-1284 and the head's own optH at :613; the
preflight gate G1 requires trainable == exactly the scorer, :1097-1099), so every S1 arm's head reads the SAME visual_ctx as the init checkpoint `model_sft4_B_u400.pt`.
One trunk forward over the navtest tokens therefore serves every arm's head (`--ctx-cache` keeps the ctx so a head that exists LATER needs no trunk at all).

WHAT IT RUNS.  The planner's own code path, not a copy of it: `refe_navtest_seam`'s loop (NS.build_scenarios_for_log -> REFePlanner.initialize -> planner._image_for ->
planner.set_driving_command -> planner.infer), the exact invocation `raw/2026-10-11-c3-selection-headroom/c3_run.py` used to produce table.npz: ONE `infer` call per token
(batch 1), rule navsim_v1_lane.  `visual_ctx` is captured by a forward PRE-hook on `model.score_dec[0]` (args[1]) -- the very tensor the scoring decoder reads and the very hook
scorer_finetune.py installs (`cap["sctx"]`, :603) -- so ctx, scorer logits and head outputs all come from ONE forward pass.

IDENTITY (the check that proves the ctx path is the planner's).  The scorer logits of that same forward are compared with `table.npz["logits"]` [1058, 64, 7].  Committed bar:
bit-identical.  `identity.json` carries the verdict, the max abs delta and the worst tokens.

PRECISION.  The planner path is fp32 (no autocast anywhere in planner.py / model.py).  The S1 trainer runs the trunk under `torch.autocast("cuda", bfloat16)`
(scorer_finetune.py fwd), so a trained head saw bf16 ctx.  Default `--head-ctx planner` feeds the heads the planner's fp32 ctx (the S3 deployment path and the only one the
identity check can vouch for); `--head-ctx trainer` re-runs the trunk under the trainer's autocast and applies the head under autocast, and `--measure-trainer-path`
records how far the two ctx differ.  The file records which path produced it (`meta_json`).

OUTPUT CONTRACT (eval/aux_s1_readout.py, "INPUT CONTRACT"):  head_pred_<ARM>.npz with `log`, `token` str [N]; `logit` float32 [N,40,32]; `sdf_m` float32 [N,40,32] (metres: the
head's own `sdf_scale * tanh`, exactly what the trainer's aux eval dumps); `step` int64 [N] = 0; plus `meta_json`.  Rows are in the table.npz token order.

    python eval/aux_navtest_forward.py --heads H2-s0=<aux_drv.pt> H1-s0=<aux_drv.pt> --out <dir> [--ctx-cache <dir>] [--limit-tokens 32]

Exit: 0 ok (also ZZAUXFWD_PARTIAL: one --max-trunk-tokens chunk cached, rerun) | 2 identity FAILED (outputs are still written, flagged) | 4 refused (head / token / cache guard) | 5 GPU busy (nothing loaded).
Venv: C:/Users/Admin/venvs/driverl-eval (torch + CUDA + nuPlan via DZ_ROOT).  GPU: takes the shared dev-box lock exactly as c3_run.py does.
"""
from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PKG, "refe"))
import torch  # noqa: E402
import aux_drivable_head as AUX  # noqa: E402  (the head's own module: the trainer builds exactly this)

TOOL = "aux_navtest_forward/1"
C3_DIR = os.path.join(PKG, "raw", "2026-10-11-c3-selection-headroom")
DEFAULT_TABLE = os.path.join(C3_DIR, "table.npz")
DEFAULT_TOKENS = os.path.join(C3_DIR, "tokens.json")
CKPT = "D:/Projects/TanitAD/data/refe_runs_eval/sft4_peek/model_sft4_B_u400.pt"       # c3_run.py CKPT
CKPT_MD5_PREFIX = "8e382eeb"                                                          # c3_run.py CKPT_MD5_PREFIX
RULE = "navsim_v1_lane"                                                               # c3_run.py RULE
FRAMES = "D:/Projects/TanitAD/data/refe_navtest/frames"
DB_DIR = "D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test"
W3_EXPORT = "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
LOCK = "C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock"
WAIT_IDLE = "D:/Projects/TanitAD-artifacts/_rlfan_work/wait_idle.py"
GPU_MAX_USED_MIB = 3500
OWNER = "aux_navtest_forward"
NCX, NCY = 40, 32
HEAD_FORMAT = "aux_drv/1"
ARM_RE = re.compile(r"^(H1|H2|H1u10k|H1u2k)-s(\d+)$")                    # H1u10k / H1u2k: the S1c head arms (raw/2026-10-16-s1c-build)


class HeadRefused(RuntimeError):
    """A head file the trainer did not write, or one that does not load strictly into the head the trainer builds."""


class GpuBusy(RuntimeError):
    pass


# ======================================================================================================================
# the shared dev-box GPU lock (c3_run.py's protocol; the owner/pid/t body is the one this task's brief specifies)
# ======================================================================================================================
def gpu_used_mib() -> int:
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True)
    try:
        return int(r.stdout.decode("utf-8", "replace").strip().splitlines()[0])
    except (ValueError, IndexError):
        return -1


def pid_alive(pid) -> bool:
    r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"], capture_output=True)      # bytes: the localised "no tasks" line is not cp1252-clean
    return f'"{pid}"' in r.stdout.decode("utf-8", "replace")


def acquire_gpu_lock(lock: str = LOCK, wait_idle: str | None = WAIT_IDLE, wait_timeout_s: float = 240.0, used_fn=gpu_used_mib, settle_s: float = 4.0) -> str:
    """Fail CLOSED: the neighbour's idle waiter must exit 0 within `wait_timeout_s`, the lock must be created with O_EXCL (a lock whose holder pid is alive is never taken; a
    lock WITHOUT our owner marker is never removed, whatever its pid says; our own dead holder's lock is removed only when it is older than 10 min), and memory.used must still read <= GPU_MAX_USED_MIB `settle_s` after taking it.  Returns the lock body."""
    if wait_idle:
        try:
            r = subprocess.run([sys.executable, wait_idle], timeout=wait_timeout_s)
        except subprocess.TimeoutExpired:
            raise GpuBusy(f"wait_idle.py did not report an idle GPU within {wait_timeout_s:.0f} s")
        if r.returncode != 0:
            raise GpuBusy(f"wait_idle.py exited {r.returncode}")
    body = json.dumps({"pid": os.getpid(), "owner": OWNER, "t": time.time()})
    os.makedirs(os.path.dirname(lock), exist_ok=True)
    for attempt in (1, 2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                holder = json.loads(open(lock, encoding="utf-8").read())
                age = time.time() - os.path.getmtime(lock)
            except (OSError, ValueError):
                raise GpuBusy(f"{lock} exists and is unreadable")
            # THE LOCK IS SHARED with other job families that write a DIFFERENT schema (e.g. {"job", "pid" = an MSYS pid, "win_pid_of_locker", "host", "acquired"}); an MSYS pid is invisible to
            # `tasklist`, so a dead-pid test cannot tell such a live job from a dead one.  We may therefore clear ONLY a lock that carries OUR owner marker; anything else is a live job's lock.
            if not isinstance(holder, dict) or holder.get("owner") != OWNER:
                who = (holder.get("job") or holder.get("owner") or "?") if isinstance(holder, dict) else "?"
                raise GpuBusy(f"{lock} is held by a FOREIGN lock (job/owner {who!r}, age {age:.0f} s): not written by {OWNER}, never removed: {holder}")
            if "pid" not in holder or pid_alive(holder["pid"]):
                raise GpuBusy(f"{lock} is held by {holder}")
            if attempt == 2 or age <= 600:
                raise GpuBusy(f"{lock} holder pid {holder['pid']} is gone but the lock is only {age:.0f} s old")
            os.remove(lock)
    else:
        raise GpuBusy("lock not acquired")
    os.write(fd, body.encode("utf-8"))
    os.close(fd)
    time.sleep(settle_s)
    used = used_fn()
    if not 0 <= used <= GPU_MAX_USED_MIB:
        release_gpu_lock(lock, body)
        raise GpuBusy(f"memory.used {used} MiB > {GPU_MAX_USED_MIB} after taking the lock")
    return body


def release_gpu_lock(lock: str, body: str) -> None:
    try:
        cur = json.loads(open(lock, encoding="utf-8").read())
        mine = json.loads(body)
        if cur.get("owner") == mine.get("owner") and cur.get("pid") == mine["pid"]:                  # only the lock THIS call wrote (its owner + pid, read from the body it wrote, so a
            # caller that labels the lock with its own OWNER, e.g. refe/hnscreen_devbox.py, still releases it); a foreign body that happens to carry our pid is not ours
            os.remove(lock)
    except (OSError, ValueError, AttributeError):
        pass


@contextlib.contextmanager
def gpu_lock(enabled: bool = True, **kw):
    if not enabled:
        yield None
        return
    body = acquire_gpu_lock(**kw)
    try:
        yield body
    finally:
        release_gpu_lock(kw.get("lock", LOCK), body)


# ======================================================================================================================
# the ctx tap and the one-token forward
# ======================================================================================================================
class CtxTap:
    """Forward PRE-hook on model.score_dec[0]: `ctx` = args[1], the tensor the scoring decoder reads (= visual_ctx while cfg.detach_scorer_context is on, the
    default).  scorer_finetune.py:603 installs the same hook (`cap["sctx"]`), so the head reads here what it read in training."""

    def __init__(self, model):
        self.ctx = None
        self._h = model.score_dec[0].register_forward_pre_hook(self._hook)

    def _hook(self, mod, args):
        self.ctx = args[1]

    def close(self):
        self._h.remove()


def swap_cam_slots(img: torch.Tensor, i: int, j: int) -> torch.Tensor:
    """DELIBERATE REGRESSION (--mutate-swap-cams): exchange camera slots i and j of img [B, N_cam, 3, H, W] before the trunk."""
    out = img.clone()
    out[:, [i, j]] = img[:, [j, i]]
    return out


def forward_token(planner, ego, img, tap: CtxTap, swap=None):
    """ONE `planner.infer` call -- the planner's own input construction, forward and selection -- with the tap armed.  Returns (scorer logits float32 numpy [M, C], ctx tensor)."""
    if swap is not None:
        img = swap_cam_slots(img, *swap)
    tap.ctx = None
    _traj, score, _k = planner.infer(ego, img)
    if tap.ctx is None:
        raise RuntimeError("the scoring decoder never ran: the ctx tap saw nothing")
    return score.float().cpu().numpy().astype(np.float32), tap.ctx


def identity_report(new: np.ndarray, ref: np.ndarray, tokens) -> dict:
    """Bit-identity of the scorer logits [N, M, C] against a banked table.  PASS only if EVERY value is equal."""
    new, ref = np.asarray(new, np.float32), np.asarray(ref, np.float32)
    if new.shape != ref.shape:
        return {"PASS": False, "n_tokens": int(new.shape[0]), "n_tokens_bit_identical": 0, "max_abs_delta": float("inf"), "shape_mismatch": [list(new.shape), list(ref.shape)]}
    d = np.abs(new.astype(np.float64) - ref.astype(np.float64)).reshape(new.shape[0], -1)
    per_tok = d.max(1) if d.size else np.zeros(0)
    exact = np.array_equal(new, ref)
    order = np.argsort(-per_tok)[:5]
    return {"PASS": bool(exact), "n_tokens": int(new.shape[0]), "n_tokens_bit_identical": int((per_tok == 0).sum()), "n_values": int(d.size),
            "n_values_differ": int((d > 0).sum()), "max_abs_delta": float(per_tok.max()) if per_tok.size else 0.0,
            "median_token_max_delta": float(np.median(per_tok)) if per_tok.size else 0.0,
            "worst_tokens": [(str(tokens[i]), float(per_tok[i])) for i in order]}


# ======================================================================================================================
# the heads
# ======================================================================================================================
def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def parse_arm(name: str) -> tuple:
    m = ARM_RE.match(name)
    if not m:
        raise HeadRefused(f"arm {name!r}: expected H1-s<k>, H2-s<k>, H1u10k-s<k> or H1u2k-s<k> (an N arm has no head)")
    return m.group(1), int(m.group(2))


def load_aux_head(path: str, device: str = "cpu"):
    """The head file as `scorer_finetune.save_aux` writes it ({format aux_drv/1, head, feed, opt, update, meta}) -> (head, info).  The head is built by the trainer's own
    builder from the config in the file and loaded STRICTLY: a missing or unexpected key raises, there is no partial load."""
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if not (isinstance(sd, dict) and sd.get("format") == HEAD_FORMAT):
        raise HeadRefused(f"{path}: not an {HEAD_FORMAT} file (format {sd.get('format') if isinstance(sd, dict) else type(sd).__name__!r})")
    meta = sd.get("meta") or {}
    if "config" not in meta or "head" not in sd:
        raise HeadRefused(f"{path}: no meta.config / head in the file")
    try:
        cfg = AUX.AuxDrivableConfig(**meta["config"])
    except (TypeError, ValueError) as e:
        raise HeadRefused(f"{path}: meta.config does not build an AuxDrivableConfig: {e}")
    if cfg.tap != "visual":
        raise HeadRefused(f"{path}: tap {cfg.tap!r}; this tool taps visual_ctx only (the scene tap is S2's)")
    head = AUX.build_head(cfg)
    head.load_state_dict(sd["head"], strict=True)
    head.to(device).eval()
    info = {"path": os.path.abspath(path), "sha256": sha256_file(path), "update": sd.get("update"), "config": dict(meta["config"]),
            "store_sha256": meta.get("store_sha256"), "frames": meta.get("frames"), "shuffle_seed": meta.get("shuffle_seed"),
            "feed_present": sd.get("feed") is not None, "n_params": int(sum(p.numel() for p in head.parameters()))}
    return head, info


def make_fresh_head_file(path: str, seed: int = 0, depth: int = 2, feed: str = "off") -> str:
    """A freshly initialised, NEVER-TRAINED head saved in the trainer's `save_aux` format (so the loader and the plumbing can be exercised before an S1 head exists)."""
    cfg = AUX.AuxDrivableConfig(tap="visual", depth=depth, feed=feed, w_aux=0.0, seed=seed)
    box = AUX.AuxDrivable(cfg)
    meta = {"config": {k: getattr(cfg, k) for k in cfg.__dataclass_fields__}, "store_sha256": AUX.PINNED_STORE_SHA256, "frames": AUX.PINNED_STORE_KEYS,
            "shuffle_seed": None, "head_file": "aux_drv.pt"}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"format": HEAD_FORMAT, "head": box.head.state_dict(), "feed": box.feed.state_dict() if box.feed is not None else None, "opt": {}, "update": 0,
                "meta": meta}, path)
    return path


@torch.no_grad()
def apply_head(head, ctx: torch.Tensor, autocast: bool = False):
    """ctx [B, P, 256] -> (logit float32 numpy [B, 40, 32], sdf_m float32 numpy [B, 40, 32] metres).  The trainer's aux eval stores `logit.float().cpu()` and the head's own
    `sdf_scale * tanh(.)` -- already metres -- so there is no conversion to apply."""
    ctx = ctx.to(next(head.parameters()).device)
    if autocast:
        with torch.autocast(device_type=ctx.device.type, dtype=torch.bfloat16):
            logit, sdf = head(ctx)
    else:
        logit, sdf = head(ctx)
    return logit.float().cpu().numpy().astype(np.float32), sdf.float().cpu().numpy().astype(np.float32)


def write_head_pred(path: str, logs, tokens, logit: np.ndarray, sdf_m: np.ndarray, meta: dict) -> str:
    n = len(tokens)
    if not (len(logs) == n and logit.shape == (n, NCX, NCY) and sdf_m.shape == logit.shape):
        raise ValueError(f"head_pred shapes: {n} tokens, {len(logs)} logs, logit {logit.shape}, sdf {sdf_m.shape}")
    tmp = path + ".tmp.npz"
    np.savez(tmp, log=np.asarray([str(x) for x in logs]), token=np.asarray([str(x) for x in tokens]), step=np.zeros(n, np.int64),
             logit=np.asarray(logit, np.float32), sdf_m=np.asarray(sdf_m, np.float32), meta_json=np.asarray(json.dumps(meta, sort_keys=True)))
    os.replace(tmp, path)
    return path


# ======================================================================================================================
# ctx cache (fp32, exact): per token a ctx_<token>.npy [P, 256] and the scorer's logits_<token>.npy [M, C] of the SAME forward (the identity check needs them)
# ======================================================================================================================
def cache_path(d: str, token: str) -> str:
    return os.path.join(d, f"ctx_{token}.npy")


def logits_path(d: str, token: str) -> str:
    return os.path.join(d, f"logits_{token}.npy")


def cache_complete(d: str, token: str) -> bool:
    return os.path.exists(cache_path(d, token)) and os.path.exists(logits_path(d, token))


def cache_save(d: str, token: str, ctx: torch.Tensor, logits: np.ndarray) -> None:
    """LOGITS FIRST: a ctx file on disk always has its logits, so a crash mid-chunk never leaves an entry the identity check cannot judge."""
    os.makedirs(d, exist_ok=True)
    for path, arr in ((logits_path(d, token), np.asarray(logits, np.float32)), (cache_path(d, token), ctx.detach().float().cpu().numpy().reshape(-1, ctx.shape[-1]))):
        tmp = path + ".tmp.npy"
        np.save(tmp, arr)
        os.replace(tmp, path)


def cache_load(d: str, token: str):
    """(ctx tensor [1, P, 256] float32, logits float32 [M, C]) or None when the entry is not complete."""
    if not cache_complete(d, token):
        return None
    return torch.from_numpy(np.load(cache_path(d, token)))[None], np.load(logits_path(d, token))


# ======================================================================================================================
# tokens
# ======================================================================================================================
def select_tokens(tokens: list, token_log: dict, limit: int | None) -> list:
    """The first `limit` tokens, taken whole log by log (the logs with the most tokens first) so a smoke run builds as few scenario sets as possible; order = the input order."""
    if not limit or limit >= len(tokens):
        return list(tokens)
    cnt: dict = {}
    for t in tokens:
        cnt[token_log[t]] = cnt.get(token_log[t], 0) + 1
    keep, n = set(), 0
    for lg in sorted(cnt, key=lambda l: (-cnt[l], l)):
        take = [t for t in tokens if token_log[t] == lg]
        for t in take:
            if n >= limit:
                break
            keep.add(t)
            n += 1
        if n >= limit:
            break
    return [t for t in tokens if t in keep]


def log_gpu_mem() -> dict:
    return {"peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3), "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2 ** 30, 3),
            "nvidia_smi_used_mib": gpu_used_mib()}


# ======================================================================================================================
# the driver
# ======================================================================================================================
def run(a) -> int:
    t_start = time.time()
    os.makedirs(a.out, exist_ok=True)
    doc = json.load(open(a.tokens, encoding="utf-8"))
    tab = np.load(a.table, allow_pickle=False)
    table_tokens = [str(t) for t in tab["token"]]
    in_doc = [str(t) for t in (doc["tokens"] if isinstance(doc, dict) else doc)]
    if len(set(in_doc)) != len(in_doc) or set(in_doc) != set(table_tokens) or len(table_tokens) != len(set(table_tokens)):
        print(f"REFUSED: --tokens ({len(in_doc)}) is not the same token SET as --table ({len(table_tokens)}); the identity check needs both")
        return 4
    toks_all = table_tokens                    # the table's order is the canonical order (tokens.json lists the same set in another order)
    token_log = doc["token_log"]
    want = select_tokens(toks_all, token_log, a.limit_tokens)
    tindex = {t: i for i, t in enumerate(table_tokens)}
    ref_logits = np.asarray(tab["logits"], np.float32)[[tindex[t] for t in want]]
    by_log: dict = {}
    for t in want:
        by_log.setdefault(token_log[t], []).append(t)

    arms = {}
    for spec in a.heads or []:
        name, _, path = spec.partition("=")
        try:
            parse_arm(name)
        except HeadRefused as e:
            print(f"REFUSED: {e}")
            return 4
        arms[name] = path
    dev = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    heads, head_info = {}, {}
    try:
        for name, path in sorted(arms.items()):
            heads[name], head_info[name] = load_aux_head(path, dev)
    except HeadRefused as e:
        print(f"REFUSED: {e}")
        return 4
    except (RuntimeError, OSError) as e:                    # a strict-load mismatch is a RuntimeError out of load_state_dict
        print(f"REFUSED: head does not load strictly: {str(e)[:600]}")
        return 4

    ck_md5 = None
    if os.path.abspath(a.ckpt) == os.path.abspath(CKPT):
        h = hashlib.md5()
        with open(a.ckpt, "rb") as f:
            for b in iter(lambda: f.read(1 << 24), b""):
                h.update(b)
        ck_md5 = h.hexdigest()
        if not ck_md5.startswith(CKPT_MD5_PREFIX):
            print(f"REFUSED: {a.ckpt} md5 {ck_md5} does not start with c3_run's {CKPT_MD5_PREFIX}")
            return 4

    if a.ctx_cache and a.head_ctx == "trainer":
        print("REFUSED: --ctx-cache stores the planner's fp32 ctx; --head-ctx trainer needs the autocast trunk, which a cache hit would silently skip")
        return 4
    cache_hit = {t: (a.ctx_cache is not None and cache_complete(a.ctx_cache, t)) for t in want}
    need_trunk = [t for t in want if not cache_hit[t]]

    remaining: list = []
    if a.max_trunk_tokens:
        if not a.ctx_cache:
            print("REFUSED: --max-trunk-tokens needs --ctx-cache (a chunk that is not cached is lost)")
            return 4
        if len(need_trunk) > a.max_trunk_tokens:                  # PARTIAL: compute (and cache) one chunk, whole logs first, then stop so the GPU lock is not held for hours
            run_set = set(select_tokens(need_trunk, token_log, a.max_trunk_tokens))
            remaining = [t for t in need_trunk if t not in run_set]
            need_trunk = [t for t in need_trunk if t in run_set]
            want = list(need_trunk)
            ref_logits = np.asarray(tab["logits"], np.float32)[[tindex[t] for t in want]]
            by_log = {}
            for t in want:
                by_log.setdefault(token_log[t], []).append(t)
    rows: dict = {}                 # token -> {"logits": [M,C], arm -> (logit, sdf)}
    timing = {"logs": {}, "n_trunk": 0}
    trainer_cmp: list = []
    mem = {}
    lock_ctx = gpu_lock(enabled=(dev == "cuda" and not a.no_gpu_lock), wait_timeout_s=a.wait_idle_timeout)     # ANY GPU use, a head-only pass on cached ctx included
    try:
        lock_ctx.__enter__()
    except GpuBusy as e:
        print(f"GPU BUSY (nothing loaded, nothing written): {e}")
        return 5
    try:
        planner = tap = None
        if need_trunk:
            import navtrain_scenarios as NS
            from planner import REFePlanner
            from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization
            import refe_navtest_seam as SEAM
            t_load = time.time()
            planner = REFePlanner(checkpoint=a.ckpt, images_root=a.frames, db_dir=a.db_dir, backbone=a.backbone, device=dev, select="best", rule=RULE)
            tap = CtxTap(planner.model)
            timing["model_load_s"] = round(time.time() - t_load, 1)
            exp = None
            cmd_mode = getattr(planner.model.cfg, "nav_mode", "goal") == "command"
            if cmd_mode or planner.sanitize_goal:
                exp = json.load(gzip.open(W3_EXPORT, "rt", encoding="utf-8"))["tokens"]
            print(f"trunk: {len(need_trunk)} tokens over {len(set(token_log[t] for t in need_trunk))} logs; device {planner.device}; nav_mode {'command' if cmd_mode else 'goal'}", flush=True)

        def sync():
            if dev == "cuda":
                torch.cuda.synchronize()

        def do_heads(tok, ctx_planner, ego, img):
            ctx_use, ac = ctx_planner, False
            if a.head_ctx == "trainer" or a.measure_trainer_path:
                ego_vec, goal = planner.last_inputs
                calib = planner._calib_for(planner._log_hint) if planner.per_sample_calib else None
                tap.ctx = None
                with torch.no_grad(), torch.autocast(device_type=dev, dtype=torch.bfloat16):
                    planner.model(img.to(dev), ego_vec, goal.to(dev), calib=calib)
                ctx_tr = tap.ctx
                if a.measure_trainer_path:
                    c0, c1 = ctx_planner.float(), ctx_tr.float()
                    trainer_cmp.append({"token": tok, "ctx_rel_l2": float((c0 - c1).norm() / c0.norm().clamp_min(1e-30)), "ctx_max_abs": float((c0 - c1).abs().max()),
                                        "ctx_planner_abs_mean": float(c0.abs().mean())})
                if a.head_ctx == "trainer":
                    ctx_use, ac = ctx_tr, True
            out = {}
            for name, hd in heads.items():
                out[name] = apply_head(hd, ctx_use, autocast=ac)
                if a.measure_trainer_path:
                    out[name + "@trainer"] = apply_head(hd, ctx_tr, autocast=True)
            return out

        t_loop = time.time()
        if need_trunk:
            exp_get = (lambda t: exp[t]) if exp is not None else (lambda t: {})
            for li, (lg, lt) in enumerate(sorted(by_log.items())):
                lt = [t for t in lt if t in need_trunk]
                if not lt:
                    continue
                db = os.path.join(a.db_dir, f"{lg}.db")
                t_log0 = time.time()
                t_build = 0.0
                gen = NS.build_scenarios_for_log(db, lt, history_rows=1, future_rows=80)
                got = set()
                while True:
                    tb = time.time()
                    try:
                        sc = next(gen)
                    except StopIteration:
                        t_build += time.time() - tb
                        break
                    t_build += time.time() - tb
                    tok = sc._initial_lidar_token
                    got.add(tok)
                    t0 = time.time()
                    planner._scenario = sc
                    planner.initialize(PlannerInitialization(route_roadblock_ids=sc.get_route_roadblock_ids(), mission_goal=sc.get_mission_goal(), map_api=sc.map_api))
                    ego = sc.get_ego_state_at_iteration(0)
                    t1 = time.time()
                    img = planner._image_for(ego)
                    if img is None:
                        print(f"REFUSED: token {tok}: no frames ({planner.frames.miss_reason})")
                        return 4
                    t2 = time.time()
                    SEAM.set_driving_command(planner, tok, exp_get(tok))
                    sync()
                    t3 = time.time()
                    score, ctx = forward_token(planner, ego, img, tap, swap=(1, 2) if a.mutate_swap_cams else None)
                    sync()
                    t4 = time.time()
                    row = {"logits": score, "heads": do_heads(tok, ctx, ego, img)}
                    sync()
                    t5 = time.time()
                    if a.ctx_cache:
                        cache_save(a.ctx_cache, tok, ctx, score)
                    t6 = time.time()
                    rows[tok] = row
                    for k, v in (("init_route", t1 - t0), ("image", t2 - t1), ("trunk_infer", t4 - t3), ("heads", t5 - t4), ("cache_write", t6 - t5)):
                        timing[k] = timing.get(k, 0.0) + v
                    timing["n_trunk"] += 1
                miss = set(lt) - got
                if miss:
                    print(f"REFUSED: {len(miss)} tokens skipped by the scenario builder in {lg} (first {sorted(miss)[0]})")
                    return 4
                timing["logs"][lg] = {"scenario_build_s": round(t_build, 2), "n": len(lt), "wall_s": round(time.time() - t_log0, 2)}
                print(f"  [{li + 1}/{len(by_log)}] {lg}: {len(lt)} tokens, scenario build {t_build:.1f} s, rows {len(rows)}/{len(need_trunk)}, {time.time() - t_loop:.0f} s", flush=True)
            if dev == "cuda":
                mem = log_gpu_mem()
        # cached tokens: heads only, no trunk
        for t in want:
            if t in rows:
                continue
            ctx, lg_c = cache_load(a.ctx_cache, t)
            rows[t] = {"logits": lg_c, "heads": {n: apply_head(h, ctx) for n, h in heads.items()}}
        timing["loop_s"] = round(time.time() - t_loop, 1)
    finally:
        lock_ctx.__exit__(None, None, None)

    if remaining:
        part = identity_report(np.stack([rows[t]["logits"] for t in want]).astype(np.float32), ref_logits, want)
        part.update({"partial": True, "tokens_in_this_chunk": len(want), "tokens_still_uncached": len(remaining), "mutate_swap_cams": bool(a.mutate_swap_cams)})
        json.dump(part, open(os.path.join(a.out, "identity_partial.json"), "w", encoding="utf-8"), indent=1)
        json.dump({"cached_or_done_in_this_chunk": len(want), "remaining": len(remaining), "wall_s": round(time.time() - t_start, 1), "chunk_identity_pass": part["PASS"]},
                  open(os.path.join(a.out, "progress.json"), "w", encoding="utf-8"), indent=1)
        print(f"ZZAUXFWD_PARTIAL chunk {len(want)} tokens (identity {'PASS' if part['PASS'] else 'FAIL'}), {len(remaining)} still uncached: rerun the same command")
        return 0 if part["PASS"] else 2
    # ---- identity, outputs ----------------------------------------------------------------------------------------
    got_logits = np.stack([rows[t]["logits"] for t in want]).astype(np.float32)
    ident = identity_report(got_logits, ref_logits, want)
    ident.update({"tool": TOOL, "reference": os.path.abspath(a.table), "ckpt": a.ckpt, "ckpt_md5": ck_md5, "rule": RULE, "grouping": "one planner.infer per token (batch 1), tokens visited log by log in sorted order, as c3_run.py -> refe_navtest_seam.py",
                  "mutate_swap_cams": bool(a.mutate_swap_cams), "n_requested": len(want), "from_ctx_cache": int(sum(cache_hit.values()))})
    json.dump(ident, open(os.path.join(a.out, "identity.json"), "w", encoding="utf-8"), indent=1)
    meta_common = {"tool": TOOL, "ckpt": a.ckpt, "ckpt_md5": ck_md5, "frames": a.frames, "rule": RULE, "head_ctx": a.head_ctx, "ctx_dtype": "float32 (planner path; no autocast)"
                   if a.head_ctx == "planner" else "bfloat16 (trainer autocast path)", "tokens_from": os.path.abspath(a.tokens), "n": len(want),
                   "scorer_identity_pass": ident["PASS"], "mutate_swap_cams": bool(a.mutate_swap_cams), "tier": "T0-like: head outputs on teacher-forced navtest frames"}
    for name in sorted(heads):
        lg_ = np.stack([rows[t]["heads"][name][0][0] for t in want])
        sd_ = np.stack([rows[t]["heads"][name][1][0] for t in want])
        write_head_pred(os.path.join(a.out, f"head_pred_{name}.npz"), [token_log[t] for t in want], want, lg_, sd_, dict(meta_common, arm=name, head=head_info[name]))
        if a.measure_trainer_path:
            lg2 = np.stack([rows[t]["heads"][name + "@trainer"][0][0] for t in want])
            sd2 = np.stack([rows[t]["heads"][name + "@trainer"][1][0] for t in want])
            trainer_cmp.append({"arm": name, "n": len(want), "logit_max_abs_delta": float(np.abs(lg_ - lg2).max()), "sdf_m_max_abs_delta": float(np.abs(sd_ - sd2).max()),
                                "mask_sign_disagree_share": float(((lg_ > 0) != (lg2 > 0)).mean()), "logit_abs_mean": float(np.abs(lg_).mean())})
    nt = max(timing["n_trunk"], 1)
    run_doc = {"tool": TOOL, "n_tokens": len(want), "n_logs": len(by_log), "n_trunk_forwards": timing["n_trunk"], "device": dev, "heads": head_info,
               "s_per_token_total": round(timing["loop_s"] / nt, 3) if timing["n_trunk"] else None,
               "timing_sums_s": {k: round(v, 2) for k, v in timing.items() if k not in ("logs", "n_trunk")}, "per_log": timing["logs"],
               "gpu": mem, "trainer_path_comparison": trainer_cmp, "wall_s": round(time.time() - t_start, 1), "identity_pass": ident["PASS"]}
    json.dump(run_doc, open(os.path.join(a.out, "run.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps({"identity": {k: ident[k] for k in ("PASS", "n_tokens", "n_tokens_bit_identical", "max_abs_delta")}, "s_per_token": run_doc["s_per_token_total"], "gpu": mem}, indent=1))
    print(f"ZZAUXFWD_{'OK' if ident['PASS'] else 'IDENTITY_FAIL'} {len(want)}")
    return 0 if ident["PASS"] else 2


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--frames", default=FRAMES)
    ap.add_argument("--db-dir", default=DB_DIR)
    ap.add_argument("--tokens", default=DEFAULT_TOKENS, help="tokens.json (dict with `tokens` and `token_log`); default = the C3 tokens; must be the same SET as --table's (output order = the table's)")
    ap.add_argument("--table", default=DEFAULT_TABLE, help="the banked C3 table.npz: `token` and the scorer `logits` the identity check compares against")
    ap.add_argument("--heads", nargs="*", default=[], help="ARM=path/to/aux_drv.pt (ARM = H1-s<k> | H2-s<k>)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ctx-cache", default=None, help="dir: save/reuse the fp32 visual_ctx per token (7.9 MB each) so a head that exists later needs no trunk forward")
    ap.add_argument("--max-trunk-tokens", type=int, default=None, help="compute and cache at most this many uncached tokens (whole logs first), then stop with ZZAUXFWD_PARTIAL: rerun the same command until none remain; "
                    "bounds how long the shared GPU lock is held. Needs --ctx-cache")
    ap.add_argument("--limit-tokens", type=int, default=None, help="smoke: only this many C3 tokens, whole logs first")
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--backbone", default="vitl16")
    ap.add_argument("--head-ctx", default="planner", choices=("planner", "trainer"))
    ap.add_argument("--measure-trainer-path", action="store_true", help="also run the trunk under the trainer's bf16 autocast and record how far its ctx / head outputs are from the planner's")
    ap.add_argument("--mutate-swap-cams", action="store_true", help="DELIBERATE REGRESSION: swap camera slots 1 and 2 before the trunk (the scorer identity must then FAIL)")
    ap.add_argument("--no-gpu-lock", action="store_true")
    ap.add_argument("--wait-idle-timeout", type=float, default=240.0)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
