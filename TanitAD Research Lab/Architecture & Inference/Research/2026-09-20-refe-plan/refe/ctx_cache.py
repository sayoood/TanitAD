#!/usr/bin/env python3
"""P6-HN-S: the frozen trunk's `visual_ctx` CACHE (bf16 shards + manifest + fan check).  (raw/2026-10-18-p6hn-prereg/PREREG_P6HN_SCREEN.md section 3, gates G15 / I6; SCREEN_NOTES.md item 2.)

WHAT IS CACHED.  The scoring decoder reads `visual_ctx = scene_proj(visual)` ([B, N_cam * P, 256]; 7,680 tokens at 4 cameras, 512 x 960, patch 16; model.py forward), the SECOND call of
`scene_proj` in the forward pass (the first is `scene_ctx = scene_proj(scene)`).  With the trunk frozen and no image augmentation it is a fixed function of (image, calib), so it is computed once per
scene under the trainer's own bf16 autocast, stored LOSSLESSLY as the bf16 tensor the live path produces (3.75 MiB per scene), and reused by every arm, seed and eval.  The same forward also emits the
64-proposal fan (`traj`), cached beside it: that is the I6 reproduction check against RL-S1's banked frozen fan.

THE TAP.  A forward hook on `model.scene_proj` records both outputs; a forward PRE-hook on `model.score_dec[0]` records the tensor the scoring decoder actually reads (`aux_navtest_forward.CtxTap`'s
protocol, `scorer_finetune.py` `cap["sctx"]`).  The capture asserts the decoder's ctx IS the storage of scene_proj's second output (not a copy, not scene_ctx) and that it is bf16 -- a float32 capture
would silently be a different tensor from the one the live path feeds.

UNITS.  One unit per log (`units/<id>.ctx.u16` raw bf16 bits as uint16 [n, P, D]; `units/<id>.fan.f32` float32 [n, K, T, 3]; `units/<id>.json` the sidecar = the DONE marker, written LAST and
atomically, carrying the keys, the two sha256s and the trunk fingerprint).  A unit is skipped on resume only if its sidecar's files re-hash to the recorded sha256 (CONTENT, never presence); a torn
unit (data without a sidecar, or bytes that no longer hash) is rebuilt; a unit built under ANOTHER trunk fingerprint is a refusal (`StaleCache`), never a silent rebuild.  The GPU is taken only through
the landed lock (`eval/aux_navtest_forward.py` `gpu_lock`: wait_idle.py + the O_EXCL devbox_gpu.lock), once per unit, so another job interleaves between units.

THE CONTENT MANIFEST (HNSCREEN-CACHE-CONTENT-FP).  `finalize` ends by SEALING the directory: the sha256 of every file's bytes is computed once, here, and written to `content_manifest.json` (`gate_token.seal_content_manifest`).
The launch gate fingerprints `--ctx-cache` by CONTENT -- it re-hashes every file against that manifest (`gate_token.content_fingerprint`) -- so a same-size byte flip, which the size-listing fingerprint could not see, is refused.
A cache finalised before this existed is sealed with `python refe/ctx_cache.py seal --dir <cache>`; `check --dir <cache>` prints its content fingerprint and any problem.

THE MANIFEST binds the trunk fingerprint (sha256 over every non-scorer tensor of the checkpoint, so a scorer-only fine-tune does NOT invalidate the cache and any trunk/generator/projection change DOES),
the scene-list sha256 and the calib sha256.  `CtxCache(dir, expect_trunk_fingerprint=...)` refuses a cache of another trunk; `verify()` re-hashes every unit.

FAN CHECK (I6).  `check_fan` writes the cached fan in the dump's own record format (`rl_fan_dump.set_record`) and calls the LANDED comparator `rl_fan_dump.compare_sets` against RL-S1's banked sets, with
BOTH registered bounds (the per-set max |delta| and the mean |delta| over x / y).  The values below are the calibration's (INHERITED, `raw/rl_fan/atol_calibration/`): the catastrophe bound is the
rule's 0.5 (CHOSEN_ATOL is null there; the chain takes it from RLS1_FANCMP_ATOL), the mean bound is round 2's CHOSEN_MEAN_ATOL 0.025.  They are NOT defaults of `check_fan`: the caller names them.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
import tempfile
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
for _p in (HERE, os.path.join(PKG, "eval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TOOL = "ctx_cache/1"
OWNER = "ctx_cache"
DTYPE = "bf16"
SCORER = ("score_q_mlp", "score_dec", "score_head")
FAN_ATOL = 0.5                      # INHERITED: raw/rl_fan/atol_calibration (the rule's value, 2 x the 0.25 bf16 floor); a catastrophe screen
FAN_MEAN_ATOL = 0.025               # INHERITED: raw/rl_fan/atol_calibration/round2/verdict_round2.json CHOSEN_MEAN_ATOL (state PASS)


class StaleCache(RuntimeError):
    """a cache (or unit) built under another trunk, or whose bytes no longer hash"""


class CaptureError(RuntimeError):
    """the tap did not see what the live path feeds the scoring decoder"""


# ============================================================================================ hashing
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def trunk_fingerprint(state_dict, exclude=SCORER):
    """sha256 over every tensor of a state dict whose top-level module is NOT a scorer module, in sorted-name order: (name, dtype, shape, bytes).  Works on `model.state_dict()` and on a
    checkpoint's loaded dict alike, so the manifest's value can be compared with the checkpoint's (G15)."""
    h = hashlib.sha256()
    for name in sorted(state_dict):
        if name.split(".")[0] in exclude:
            continue
        t = state_dict[name]
        if not torch.is_tensor(t):
            continue
        t = t.detach().cpu().contiguous()
        h.update(f"{name}|{t.dtype}|{tuple(t.shape)}|".encode())
        h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()


def keys_sha256(keys):
    ks = sorted((str(k[0]), str(k[1]), int(k[2]), int(k[3])) for k in keys)
    return sha256_bytes(json.dumps(ks, separators=(",", ":")).encode("utf-8"))


# ============================================================================================ bf16 <-> raw bits
def to_u16(t):
    if t.dtype != torch.bfloat16:
        raise CaptureError(f"expected a bf16 tensor (the trainer's autocast dtype), got {t.dtype}")
    return t.detach().cpu().contiguous().view(torch.int16).numpy().view(np.uint16)


def from_u16(a):
    return torch.from_numpy(np.ascontiguousarray(a).view(np.int16)).view(torch.bfloat16)


# ============================================================================================ the tap
class VisualCtxTap:
    """records scene_proj's outputs and what score_dec[0] reads; `visual_ctx()` returns the SECOND scene_proj output after asserting it is the decoder's ctx"""

    def __init__(self, model):
        self.outs, self.dec_ctx = [], None
        self._h1 = model.scene_proj.register_forward_hook(lambda m, i, o: self.outs.append(o))
        self._h2 = model.score_dec[0].register_forward_pre_hook(lambda m, args: setattr(self, "dec_ctx", args[1]))

    def reset(self):
        self.outs, self.dec_ctx = [], None

    def visual_ctx(self):
        if len(self.outs) != 2:
            raise CaptureError(f"scene_proj ran {len(self.outs)} times in one forward, the model calls it twice (scene, then visual)")
        v = self.outs[1]
        if self.dec_ctx is None or self.dec_ctx.data_ptr() != v.data_ptr() or self.dec_ctx.shape != v.shape:
            raise CaptureError("the scoring decoder's context is not scene_proj's second output (a copy, or scene_ctx)")
        return v

    def close(self):
        self._h1.remove()
        self._h2.remove()


def capture_batch(model, forward, batch, tap=None):
    """-> (visual_ctx bf16 [B, P, D] on cpu, fan float32 [B, K, T, 3] on cpu).  `forward(batch)` runs the model under the trainer's own autocast and returns the fan `traj`."""
    own = tap is None
    tap = tap or VisualCtxTap(model)
    try:
        tap.reset()
        with torch.no_grad():
            traj = forward(batch)
            ctx = tap.visual_ctx()
            return ctx.detach().cpu().clone(), traj.detach().float().cpu().clone()
    finally:
        if own:
            tap.close()


# ============================================================================================ the landed lock
def landed_gpu_lock(**kw):
    """the dev-box GPU lock of eval/aux_navtest_forward.py (wait_idle.py + the O_EXCL devbox_gpu.lock, memory.used re-checked), imported lazily; its owner label is set to this tool for the call"""
    import aux_navtest_forward as ANF

    @contextlib.contextmanager
    def cm():
        old = ANF.OWNER
        ANF.OWNER = OWNER
        try:
            with ANF.gpu_lock(enabled=True, **kw) as body:
                yield body
        finally:
            ANF.OWNER = old
    return cm()


# ============================================================================================ units
def _paths(out_dir, uid):
    d = os.path.join(out_dir, "units")
    return os.path.join(d, f"{uid}.ctx.u16"), os.path.join(d, f"{uid}.fan.f32"), os.path.join(d, f"{uid}.json")


def _write_atomic(path, data: bytes):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def unit_state(out_dir, uid, keys, trunk_fp):
    """-> 'done' | 'rebuild'.  Raises StaleCache for a sidecar of another trunk.  'done' only if the files re-hash to the sidecar's sha256s (content, not presence) and the keys are the asked ones."""
    ctx_p, fan_p, side_p = _paths(out_dir, uid)
    if not os.path.exists(side_p):
        return "rebuild"
    try:
        side = json.load(open(side_p, encoding="utf-8"))
    except ValueError:
        return "rebuild"
    if side.get("trunk_fingerprint") != trunk_fp:
        raise StaleCache(f"unit {uid} was built under trunk {str(side.get('trunk_fingerprint'))[:12]}, this run is {trunk_fp[:12]}")
    if not (os.path.exists(ctx_p) and os.path.exists(fan_p)):
        return "rebuild"
    if side.get("keys_sha256") != keys_sha256(keys) or sha256_file(ctx_p) != side.get("ctx_sha256") or sha256_file(fan_p) != side.get("fan_sha256"):
        return "rebuild"
    return "done"


def build_units(model, units, load_batch, forward, out_dir, *, trunk_fp, device="cpu", batch=8, lock=None, budget_s=None, clock=time.monotonic, log=print):
    """Build (or resume) the cache.  units: [(unit_id, [(log, token, step, rank), ...])].  Returns {"built": [...], "skipped": [...], "remaining": [...]}.
    The GPU is taken only through `lock` (a zero-arg callable returning a context manager; the landed lock when device is cuda), once per unit.  After `budget_s` seconds no new unit starts."""
    if str(device).startswith("cuda") and lock is None:
        raise ValueError("a cuda device needs the landed GPU lock (lock=landed_gpu_lock): the dev box is shared")
    os.makedirs(os.path.join(out_dir, "units"), exist_ok=True)
    t0 = clock()
    built, skipped, remaining = [], [], []
    tap = VisualCtxTap(model)
    try:
        for uid, keys in units:
            if unit_state(out_dir, uid, keys, trunk_fp) == "done":
                skipped.append(uid)
                continue
            if budget_s is not None and built and clock() - t0 >= budget_s:
                remaining.append(uid)
                continue
            ctx_p, fan_p, side_p = _paths(out_dir, uid)
            if os.path.exists(side_p):
                os.remove(side_p)                                   # the DONE marker goes first: a torn rebuild must never look done
            with (lock() if lock is not None else contextlib.nullcontext()):
                ctxs, fans = [], []
                for i in range(0, len(keys), batch):
                    c, f = capture_batch(model, forward, load_batch(keys[i:i + batch]), tap)
                    ctxs.append(c)
                    fans.append(f)
            ctx = torch.cat(ctxs, 0)
            fan = torch.cat(fans, 0).numpy().astype("<f4")
            ctx_b, fan_b = to_u16(ctx).tobytes(), fan.tobytes()
            _write_atomic(ctx_p, ctx_b)
            _write_atomic(fan_p, fan_b)
            side = {"unit": uid, "n": len(keys), "keys": [list(map(lambda x: x if isinstance(x, str) else int(x), k)) for k in keys], "keys_sha256": keys_sha256(keys),
                    "ctx_shape": list(ctx.shape), "fan_shape": list(fan.shape), "ctx_sha256": sha256_bytes(ctx_b), "fan_sha256": sha256_bytes(fan_b),
                    "dtype": DTYPE, "trunk_fingerprint": trunk_fp}
            _write_atomic(side_p, json.dumps(side, sort_keys=True).encode("utf-8"))        # LAST: the DONE marker
            built.append(uid)
            log(f"ZZCTX_UNIT {uid} n {len(keys)}")
    finally:
        tap.close()
    return {"built": built, "skipped": skipped, "remaining": remaining}


def finalize(out_dir, units, *, trunk_fp, calib_sha256, meta=None):
    """write manifest.json once every unit is done (raises otherwise); the manifest binds the trunk fingerprint, the scene-list sha256 and the calib sha256"""
    ent, allkeys = [], []
    for uid, keys in units:
        if unit_state(out_dir, uid, keys, trunk_fp) != "done":
            raise StaleCache(f"unit {uid} is not done: refusing to finalise a partial cache")
        side = json.load(open(_paths(out_dir, uid)[2], encoding="utf-8"))
        ent.append({k: side[k] for k in ("unit", "n", "keys_sha256", "ctx_shape", "fan_shape", "ctx_sha256", "fan_sha256")})
        allkeys.extend(keys)
    shp = ent[0]["ctx_shape"]
    man = {"tool": TOOL, "dtype": DTYPE, "n_scenes": len(allkeys), "n_tokens": shp[1], "dim": shp[2], "fan_shape": ent[0]["fan_shape"][1:], "trunk_fingerprint": trunk_fp,
           "scene_list_sha256": keys_sha256(allkeys), "calib_sha256": calib_sha256, "units": ent, **(meta or {})}
    _write_atomic(os.path.join(out_dir, "manifest.json"), json.dumps(man, indent=1, sort_keys=True).encode("utf-8"))
    seal(out_dir)                                                  # LAST: the per-file sha256 manifest the launch gate re-checks (it hashes manifest.json too, so it must come after it)
    return man


def seal(out_dir):
    """seal `out_dir` (write content_manifest.json: sha256 of every file's bytes, computed once); -> {"path", "sha256", "n_files", "bytes"}.  Re-sealing rewrites it."""
    import gate_token as GT
    return GT.seal_content_manifest(out_dir)


# ============================================================================================ the reader
class CtxCache:
    def __init__(self, out_dir, expect_trunk_fingerprint=None, verify=False):
        self.dir = out_dir
        self.man = json.load(open(os.path.join(out_dir, "manifest.json"), encoding="utf-8"))
        if expect_trunk_fingerprint is not None and self.man["trunk_fingerprint"] != expect_trunk_fingerprint:
            raise StaleCache(f"cache trunk {self.man['trunk_fingerprint'][:12]} != the checkpoint's {expect_trunk_fingerprint[:12]}")
        self.index, self._units = {}, {}
        for u in self.man["units"]:
            side = json.load(open(_paths(out_dir, u["unit"])[2], encoding="utf-8"))
            self._units[u["unit"]] = side
            for i, k in enumerate(side["keys"]):
                self.index[(k[0], k[1], int(k[2]), int(k[3]))] = (u["unit"], i)
        if verify:
            self.verify()

    def verify(self):
        """re-hash every unit AND the scene list: content, not presence"""
        allkeys = []
        for u in self.man["units"]:
            ctx_p, fan_p, _ = _paths(self.dir, u["unit"])
            if sha256_file(ctx_p) != u["ctx_sha256"] or sha256_file(fan_p) != u["fan_sha256"]:
                raise StaleCache(f"unit {u['unit']}: bytes do not hash to the manifest")
            allkeys.extend(self._units[u["unit"]]["keys"])
        if keys_sha256(allkeys) != self.man["scene_list_sha256"]:
            raise StaleCache("scene list does not hash to the manifest")
        return True

    def _rows(self, keys, which):
        out = []
        for k in keys:
            uid, i = self.index[(str(k[0]), str(k[1]), int(k[2]), int(k[3]))]
            side = self._units[uid]
            if which == "ctx":
                m = np.memmap(_paths(self.dir, uid)[0], dtype=np.uint16, mode="r", shape=tuple(side["ctx_shape"]))
                out.append(np.asarray(m[i]))
            else:
                m = np.memmap(_paths(self.dir, uid)[1], dtype="<f4", mode="r", shape=tuple(side["fan_shape"]))
                out.append(np.asarray(m[i]))
        return out

    def get_ctx(self, keys):
        return from_u16(np.stack(self._rows(keys, "ctx")))

    def get_fan(self, keys):
        return torch.from_numpy(np.stack(self._rows(keys, "fan")).astype(np.float32))


# ============================================================================================ the CLI: seal a finished cache / print its content fingerprint
def main(argv=None):
    import argparse
    import gate_token as GT
    ap = argparse.ArgumentParser(description="seal / check the content manifest of a finished ctx cache")
    ap.add_argument("cmd", choices=("seal", "check"))
    ap.add_argument("--dir", required=True, help="the cache directory (the --ctx-cache value)")
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    if a.cmd == "seal":
        if not os.path.isfile(os.path.join(a.dir, "manifest.json")):
            print(f"REFUSED: {a.dir} has no manifest.json: finalise the cache before sealing it", file=sys.stderr)
            return 2
        try:
            r = seal(a.dir)
        except GT.GateError as e:
            print(f"REFUSED: {e}", file=sys.stderr)
            return 2
        print(f"ZZCTX_SEALED files {r['n_files']} bytes {r['bytes']} sha256 {r['sha256']} seconds {time.perf_counter() - t0:.1f}")
        return 0
    fp = GT.content_fingerprint(a.dir)
    print(f"{GT.fp_key(fp)}  seconds {time.perf_counter() - t0:.1f}")
    for pr in fp.get("problems", []):
        print(f"PROBLEM {pr}", file=sys.stderr)
    return 0 if fp.get("kind") == "dir-content" and not fp.get("n_problems") else 1


# ============================================================================================ the fan check (I6)
def check_fan(cache, ref_paths, ckpt_step, *, atol, mean_atol, tmp_dir=None):
    """The cached fan against RL-S1's banked frozen fan through the LANDED comparator (`rl_fan_dump.compare_sets`), both registered bounds.  ok <=> every cached scene has a
    reference, none is beyond `atol` (per-set max |delta| over x / y / yaw), none beyond `mean_atol` (mean |delta| over x / y)."""
    import rl_fan_dump as RFD
    keys = [tuple(k) for k in cache.index]
    fan = cache.get_fan(keys).numpy().astype(np.float64)
    K, T = fan.shape[1], fan.shape[2]
    tmp_dir = tmp_dir or tempfile.mkdtemp(prefix="ctxfan_")
    path = os.path.join(tmp_dir, "cached_fan_sets.jsonl")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for k, props in zip(keys, fan):
            rec = RFD.set_record({"log_name": k[0], "token": k[1], "step": k[2]}, k[3], ckpt_step, props, "", n_prop=K, horizon=T)
            f.write(json.dumps(rec) + "\n")
    rep = RFD.compare_sets(path, list(ref_paths), ckpt_step, atol=atol, mean_atol=mean_atol)
    n = len(keys)
    ok = bool(rep["reference_keys_matched"] == n and rep["beyond_atol"] == 0 and rep["beyond_mean_atol"] == 0 and rep["reference_lines_disagreeing_with_each_other"] == 0)
    return {"gate": "I6", "ok": ok, "n_cached": n, "n_matched": rep["reference_keys_matched"], "atol": atol, "mean_atol": mean_atol, "beyond_atol": rep["beyond_atol"],
            "beyond_mean_atol": rep["beyond_mean_atol"], "max_abs_delta": rep["max_abs_delta"], "max_mean_abs_delta": rep["max_mean_abs_delta"], "report": rep,
            "evidence": "MEASURED when run on the dev box; tolerances INHERITED from raw/rl_fan/atol_calibration"}


def cache_vs_live(cache, keys, live_ctx):
    """G15's raw field: max |cache - live| over the given scenes (live_ctx bf16 [B, P, D] from a fresh trunk forward).  Identity reads exactly 0.0; a row-shuffled cache must read far above it."""
    c = cache.get_ctx(keys).float()
    return float((c - live_ctx.float()).abs().max())


if __name__ == "__main__":
    sys.exit(main())
