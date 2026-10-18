#!/usr/bin/env python3
"""Tests for refe/ctx_cache.py (P6-HN-S checklist item 2; PREREG_P6HN_SCREEN.md section 3, gates G15 / I6).
Run:  PYTHONIOENCODING=utf-8 python refe/test_ctx_cache.py   ->   ZZTEST_PASS <n>   (prints the mutation table).   CPU fixture only: a tiny stand-in model with the REAL hook contract (scene_proj called
twice, score_dec[0] reading the second output), synthetic images and fans.  No GPU, no checkpoint, no real scene.

WHAT IS PROVEN:
  C  cache == live: the cached bf16 `visual_ctx` of every scene equals a fresh capture bit for bit (identity reads exactly 0.0); a ROW-SHUFFLED cache reads far above it (the control that proves the
     comparison can see a wrong row) and `verify()` refuses it; the tap takes scene_proj's SECOND output and refuses a copy, a single-call model and a non-bf16 forward.
  S  stale trunk: the fingerprint ignores scorer tensors (a scorer-only fine-tune keeps the cache valid) and changes with any trunk tensor; it is the same for `model.state_dict()` and a saved
     checkpoint; a reader of another trunk is REFUSED; resuming into a cache of another trunk is REFUSED (not silently rebuilt).
  R  resume: interrupted-then-resumed == built at once, byte for byte (units and manifest); a corrupted unit is rebuilt by CONTENT, a torn unit (data without its sidecar) is rebuilt, a finished unit is not
     recomputed, a partial cache is not finalised.
  L  the GPU is taken only through the lock, once per unit built (none for a resumed unit); a cuda device without the lock is refused.
  F  the fan check goes through the LANDED comparator with both registered bounds: it passes inside them, fails on one scene beyond the max bound, fails on a mean shift inside the max bound but beyond
     the mean bound, and fails when a cached scene has no reference.
Every mutation below edits ctx_cache.py SOURCE (each snippet must occur EXACTLY once) and must turn >= 1 expected check RED.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import sys
import tempfile
import types

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SRC_PATH = os.path.join(HERE, "ctx_cache.py")
P, D, K, T = 12, 8, 5, 4


def check(bad, name, fn):
    try:
        ok = bool(fn())
    except Exception as e:                                            # noqa: BLE001 -- an exception IS a failed check
        ok = False
        if os.environ.get("CCT_DEBUG"):
            print("  [check raised]", name, type(e).__name__, str(e)[:200])
    if not ok:
        bad.append(name)


def load_module(src, tag):
    mod = types.ModuleType(f"ctx_cache_{tag}")
    mod.__file__ = SRC_PATH
    exec(compile(src, f"ctx_cache_{tag}.py", "exec"), mod.__dict__)    # noqa: S102 -- the module under test, possibly mutated
    return mod


class Blk(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.l = nn.Linear(d, d)

    def forward(self, q, ctx):
        return q + self.l(ctx.mean(1, keepdim=True))


class Fake(nn.Module):
    """the hook contract of REFe.forward: scene_proj(scene) then scene_proj(visual); score_dec[0](q, visual_ctx.detach())"""

    def __init__(self, seed=0, copy_ctx=False, once=False):
        super().__init__()
        torch.manual_seed(seed)
        self.backbone = nn.Linear(6, 16)
        self.scene_proj = nn.Linear(16, D)
        self.traj_head = nn.Linear(D, K * T * 3)
        self.score_dec = nn.ModuleList([Blk(D)])
        self.score_q_mlp = nn.Linear(3, D)
        self.score_head = nn.Linear(D, 6)
        self.copy_ctx, self.once = copy_ctx, once

    def forward(self, img):
        tok = self.backbone(img)
        scene_ctx = self.scene_proj(tok.mean(1, keepdim=True))
        visual_ctx = scene_ctx.expand(-1, P, -1) if self.once else self.scene_proj(tok)
        traj = self.traj_head(scene_ctx.squeeze(1)).view(img.shape[0], K, T, 3)
        q = self.score_q_mlp(traj.detach().mean(2))
        c = visual_ctx.detach().clone() if self.copy_ctx else visual_ctx.detach()
        return traj, self.score_head(self.score_dec[0](q, c))


def load_batch(keys):
    imgs = []
    for k in keys:
        s = int(hashlib.sha256(f"{k[0]}|{k[1]}".encode()).hexdigest()[:8], 16)
        imgs.append(torch.randn(P, 6, generator=torch.Generator().manual_seed(s)))
    return torch.stack(imgs)


def make_forward(model, autocast=True):
    def fwd(batch):
        if autocast:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                return model(batch)[0]
        return model(batch)[0]
    return fwd


def mk_units():
    units, n = [], 0
    for li, cnt in enumerate((3, 2, 2)):
        units.append((f"log{li}", [(f"log{li}", f"t{li}{j}", 0, 0) for j in range(cnt)]))
        n += cnt
    return units


class FakeLock:
    def __init__(self):
        self.enter = self.exit = 0

    def __call__(self):
        outer = self

        class _C:
            def __enter__(s):
                outer.enter += 1

            def __exit__(s, *a):
                outer.exit += 1
                return False
        return _C()


def suite(CC, tag):
    bad: list = []
    root = tempfile.mkdtemp(prefix=f"ccc_{tag}_")
    units = mk_units()
    allkeys = [k for _u, ks in units for k in ks]
    model = Fake().eval()
    fwd = make_forward(model)
    fp = CC.trunk_fingerprint(model.state_dict())

    def build(dirname, mdl=None, fingerprint=None, **kw):
        d = os.path.join(root, dirname)
        m = mdl or model
        return d, CC.build_units(m, units, load_batch, make_forward(m), d, trunk_fp=fingerprint or CC.trunk_fingerprint(m.state_dict()), log=lambda *_: None, **kw)

    dA, rA = build("A")
    manA = CC.finalize(dA, units, trunk_fp=fp, calib_sha256="calib-1")
    cache = CC.CtxCache(dA, expect_trunk_fingerprint=fp, verify=True)

    # ------------------------------------------------------------------ C: cache == live, the shuffled control, the tap
    def live_ctx(keys):
        return CC.capture_batch(model, fwd, load_batch(keys))[0]

    def equals_live():
        diffs = [CC.cache_vs_live(cache, ks, live_ctx(ks)) for _u, ks in units]
        fan_ok = all(torch.equal(cache.get_fan(ks), CC.capture_batch(model, fwd, load_batch(ks))[1]) for _u, ks in units)
        return max(diffs) == 0.0 and fan_ok and cache.get_ctx(allkeys[:2]).dtype == torch.bfloat16 and manA["n_scenes"] == 7 and manA["dtype"] == "bf16"

    check(bad, "C_cache_equals_live_bit_for_bit", equals_live)

    def shuffled_control():
        d = os.path.join(root, "A_shuf")
        shutil.copytree(dA, d)
        ctx_p = CC._paths(d, "log0")[0]
        a = np.fromfile(ctx_p, dtype=np.uint16).reshape(3, P, D).copy()
        a[[0, 1]] = a[[1, 0]]
        a.tofile(ctx_p)
        bad_cache = CC.CtxCache(d)
        ks = units[0][1]
        shuf = CC.cache_vs_live(bad_cache, ks, live_ctx(ks))
        try:
            CC.CtxCache(d, verify=True)
            refused = False
        except CC.StaleCache:
            refused = True
        return shuf > 1e-3 and refused

    check(bad, "C_row_shuffled_cache_reads_far_above_identity_and_is_refused", shuffled_control)

    def tap_contract():
        out = [True, True]
        # a copy: the capture refuses; a model that calls scene_proj once: refuses
        m = Fake(copy_ctx=True).eval()
        try:
            CC.capture_batch(m, make_forward(m), load_batch(units[0][1]))
            out.append(False)
        except CC.CaptureError:
            out.append(True)
        m1 = Fake(once=True).eval()
        try:
            CC.capture_batch(m1, make_forward(m1), load_batch(units[0][1]))
            out.append(False)
        except CC.CaptureError:
            out.append(True)
        return all(out[2:])

    check(bad, "C_tap_takes_scene_proj_second_output_refuses_copy_or_single_call", tap_contract)

    def dtype_refused():
        d = os.path.join(root, "A_f32")
        try:
            CC.build_units(model, units, load_batch, make_forward(model, autocast=False), d, trunk_fp=fp, log=lambda *_: None)
            return False
        except CC.CaptureError:
            return True

    check(bad, "C_non_bf16_forward_refused", dtype_refused)

    # ------------------------------------------------------------------ S: stale trunk
    def fingerprint_scope():
        m2 = Fake().eval()
        with torch.no_grad():
            m2.score_head.weight.add_(1.0)
            m2.score_dec[0].l.weight.add_(1.0)
            m2.score_q_mlp.bias.add_(1.0)
        same_with_scorer_change = CC.trunk_fingerprint(m2.state_dict()) == fp
        m3 = Fake().eval()
        with torch.no_grad():
            m3.backbone.weight[0, 0] += 1e-3
        diff_trunk = CC.trunk_fingerprint(m3.state_dict()) != fp
        m4 = Fake().eval()
        with torch.no_grad():
            m4.scene_proj.bias[0] += 1e-3
        diff_proj = CC.trunk_fingerprint(m4.state_dict()) != fp
        p = os.path.join(root, "ck.pt")
        torch.save(model.state_dict(), p)
        same_ckpt = CC.trunk_fingerprint(torch.load(p, map_location="cpu")) == fp
        return same_with_scorer_change and diff_trunk and diff_proj and same_ckpt

    check(bad, "S_fingerprint_ignores_scorer_changes_with_trunk_and_matches_checkpoint", fingerprint_scope)

    def reader_refuses_other_trunk():
        try:
            CC.CtxCache(dA, expect_trunk_fingerprint="0" * 64)
            return False
        except CC.StaleCache:
            return True

    check(bad, "S_reader_refuses_a_cache_of_another_trunk", reader_refuses_other_trunk)

    def resume_refuses_other_trunk():
        d = os.path.join(root, "A_stale")
        shutil.copytree(dA, d)
        try:
            CC.build_units(model, units, load_batch, fwd, d, trunk_fp="1" * 64, log=lambda *_: None)
            return False
        except CC.StaleCache:
            return True

    check(bad, "S_resume_into_another_trunks_units_is_refused", resume_refuses_other_trunk)

    # ------------------------------------------------------------------ R: resume
    def files_equal(d1, d2):
        names = sorted(os.listdir(os.path.join(d1, "units")))
        if names != sorted(os.listdir(os.path.join(d2, "units"))):
            return False
        return all(open(os.path.join(d1, "units", n), "rb").read() == open(os.path.join(d2, "units", n), "rb").read() for n in names) \
            and open(os.path.join(d1, "manifest.json"), "rb").read() == open(os.path.join(d2, "manifest.json"), "rb").read()

    def resume_identical():
        d, r1 = build("B", budget_s=0.0)                                   # exactly one unit, then the budget stops it
        if not (len(r1["built"]) == 1 and len(r1["remaining"]) == 2):
            return False
        r2 = CC.build_units(model, units, load_batch, fwd, d, trunk_fp=fp, log=lambda *_: None)
        if not (r2["skipped"] == r1["built"] and len(r2["built"]) == 2):
            return False
        CC.finalize(d, units, trunk_fp=fp, calib_sha256="calib-1")
        return files_equal(dA, d)

    check(bad, "R_interrupted_then_resumed_equals_built_at_once_byte_for_byte", resume_identical)

    def corrupted_and_torn_rebuilt():
        d = os.path.join(root, "C")
        shutil.copytree(dA, d)
        ctx_p = CC._paths(d, "log1")[0]
        raw = bytearray(open(ctx_p, "rb").read())
        raw[5] ^= 0xFF
        open(ctx_p, "wb").write(bytes(raw))                                 # corrupted bytes, sidecar intact
        os.remove(CC._paths(d, "log2")[2])                                  # torn: data without its DONE marker
        r = CC.build_units(model, units, load_batch, fwd, d, trunk_fp=fp, log=lambda *_: None)
        CC.finalize(d, units, trunk_fp=fp, calib_sha256="calib-1")
        return sorted(r["built"]) == ["log1", "log2"] and r["skipped"] == ["log0"] and files_equal(dA, d)

    check(bad, "R_corrupted_unit_rebuilt_by_content_and_torn_unit_rebuilt", corrupted_and_torn_rebuilt)

    def partial_not_finalised():
        d, _r = build("D", budget_s=0.0)
        try:
            CC.finalize(d, units, trunk_fp=fp, calib_sha256="x")
            return False
        except CC.StaleCache:
            return True

    check(bad, "R_partial_cache_is_not_finalised", partial_not_finalised)

    def manifest_binds():
        cc = CC.CtxCache(dA, verify=True)
        ok = (manA["trunk_fingerprint"] == fp and manA["calib_sha256"] == "calib-1" and manA["scene_list_sha256"] == CC.keys_sha256(allkeys) and cc.verify())
        d = os.path.join(root, "A_scene")
        shutil.copytree(dA, d)
        sp = CC._paths(d, "log0")[2]
        side = json.load(open(sp, encoding="utf-8"))
        side["keys"][0][1] = "tampered"
        json.dump(side, open(sp, "w", encoding="utf-8"), sort_keys=True)
        try:
            CC.CtxCache(d, verify=True)
            tamper_refused = False
        except CC.StaleCache:
            tamper_refused = True
        return ok and tamper_refused

    check(bad, "R_manifest_binds_trunk_scene_list_calib_and_verify_rehashes", manifest_binds)

    # ------------------------------------------------------------------ L: the lock
    def lock_per_unit():
        lk = FakeLock()
        d, r = build("E", lock=lk)
        again = CC.build_units(model, units, load_batch, fwd, d, trunk_fp=fp, lock=lk, log=lambda *_: None)
        try:
            CC.build_units(model, units, load_batch, fwd, os.path.join(root, "E2"), trunk_fp=fp, device="cuda", log=lambda *_: None)
            refused = False
        except ValueError:
            refused = True
        return lk.enter == 3 and lk.exit == 3 and len(again["built"]) == 0 and refused

    check(bad, "L_lock_taken_once_per_unit_built_none_for_done_and_cuda_without_lock_refused", lock_per_unit)

    # ------------------------------------------------------------------ F: the fan check
    def write_refs(path, noise=0.0, big_on=None, shift=0.0, drop=None, ckpt=10075):
        rng = np.random.default_rng(0)
        import rl_fan_dump as RFD
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for k in allkeys:
                if k == drop:
                    continue
                fan = cache.get_fan([k]).numpy().astype(np.float64)[0]
                fan = fan + rng.normal(0, noise, fan.shape)
                fan[..., :2] += shift
                if k == big_on:
                    fan[0, 0, 0] += 0.6
                rec = RFD.set_record({"log_name": k[0], "token": k[1], "step": k[2]}, k[3], ckpt, fan, "", n_prop=K, horizon=T)
                f.write(json.dumps(rec) + "\n")

    def fan_check():
        pth = os.path.join(root, "ref.jsonl")
        out = []
        write_refs(pth, noise=1e-3)
        r = CC.check_fan(cache, [pth], 10075, atol=CC.FAN_ATOL, mean_atol=CC.FAN_MEAN_ATOL)
        out.append(r["ok"] and r["n_matched"] == 7)
        write_refs(pth, noise=1e-3, big_on=allkeys[2])
        r = CC.check_fan(cache, [pth], 10075, atol=CC.FAN_ATOL, mean_atol=CC.FAN_MEAN_ATOL)
        out.append((not r["ok"]) and r["beyond_atol"] == 1)
        write_refs(pth, shift=0.03)                                           # every set moves 0.03: inside the 0.5 max bound, beyond the 0.025 mean bound
        r = CC.check_fan(cache, [pth], 10075, atol=CC.FAN_ATOL, mean_atol=CC.FAN_MEAN_ATOL)
        out.append((not r["ok"]) and r["beyond_atol"] == 0 and r["beyond_mean_atol"] == 7)
        write_refs(pth, noise=1e-3, drop=allkeys[4])
        r = CC.check_fan(cache, [pth], 10075, atol=CC.FAN_ATOL, mean_atol=CC.FAN_MEAN_ATOL)
        out.append((not r["ok"]) and r["n_matched"] == 6)
        write_refs(pth, noise=1e-3)
        r = CC.check_fan(cache, [pth], 99, atol=CC.FAN_ATOL, mean_atol=CC.FAN_MEAN_ATOL)       # another ckpt_step: nothing matches
        out.append((not r["ok"]) and r["n_matched"] == 0)
        return all(out)

    check(bad, "F_fan_check_through_landed_comparator_both_bounds_and_missing_reference", fan_check)
    return bad


MUTATIONS = [
    ("the tap takes scene_proj's FIRST output", [("        v = self.outs[1]\n", "        v = self.outs[0]\n")], ["C_cache_equals_live_bit_for_bit", "C_tap_takes_scene_proj_second_output_refuses_copy_or_single_call"]),
    ("the tap's storage check dropped", [("        if self.dec_ctx is None or self.dec_ctx.data_ptr() != v.data_ptr() or self.dec_ctx.shape != v.shape:\n", "        if False:\n")],
     ["C_tap_takes_scene_proj_second_output_refuses_copy_or_single_call"]),
    ("bf16 not asserted", [("    if t.dtype != torch.bfloat16:\n", "    if False:\n")], ["C_non_bf16_forward_refused"]),
    ("the fingerprint includes the scorer", [("        if name.split(\".\")[0] in exclude:\n", "        if False:\n")], ["S_fingerprint_ignores_scorer_changes_with_trunk_and_matches_checkpoint"]),
    ("the reader skips the trunk check", [("        if expect_trunk_fingerprint is not None and self.man[\"trunk_fingerprint\"] != expect_trunk_fingerprint:\n", "        if False:\n")],
     ["S_reader_refuses_a_cache_of_another_trunk"]),
    ("a stale sidecar is not refused", [("    if side.get(\"trunk_fingerprint\") != trunk_fp:\n", "    if False:\n")], ["S_resume_into_another_trunks_units_is_refused"]),
    ("resume trusts presence, not content", [("    if side.get(\"keys_sha256\") != keys_sha256(keys) or sha256_file(ctx_p) != side.get(\"ctx_sha256\") or sha256_file(fan_p) != side.get(\"fan_sha256\"):\n", "    if False:\n")],
     ["R_corrupted_unit_rebuilt_by_content_and_torn_unit_rebuilt"]),
    ("finalise without checking the units", [("        if unit_state(out_dir, uid, keys, trunk_fp) != \"done\":\n            raise StaleCache", "        if False:\n            raise StaleCache")], ["R_partial_cache_is_not_finalised"]),
    ("verify() skips the scene list", [("        if keys_sha256(allkeys) != self.man[\"scene_list_sha256\"]:\n", "        if False:\n")], ["R_manifest_binds_trunk_scene_list_calib_and_verify_rehashes"]),
    ("the GPU lock is not taken", [("            with (lock() if lock is not None else contextlib.nullcontext()):\n", "            with contextlib.nullcontext():\n")],
     ["L_lock_taken_once_per_unit_built_none_for_done_and_cuda_without_lock_refused"]),
    ("a cuda device without the lock is allowed", [("    if str(device).startswith(\"cuda\") and lock is None:\n", "    if False:\n")], ["L_lock_taken_once_per_unit_built_none_for_done_and_cuda_without_lock_refused"]),
    ("the fan check ignores the mean bound", [("rep[\"beyond_atol\"] == 0 and rep[\"beyond_mean_atol\"] == 0", "rep[\"beyond_atol\"] == 0")], ["F_fan_check_through_landed_comparator_both_bounds_and_missing_reference"]),
    ("the fan check ignores a missing reference", [("rep[\"reference_keys_matched\"] == n and ", "")], ["F_fan_check_through_landed_comparator_both_bounds_and_missing_reference"]),
    ("a shuffled-row check cannot see a swap (rows read from the first scene)", [("            uid, i = self.index[(str(k[0]), str(k[1]), int(k[2]), int(k[3]))]\n", "            uid, i = self.index[(str(k[0]), str(k[1]), int(k[2]), int(k[3]))]\n            i = 0\n")],
     ["C_cache_equals_live_bit_for_bit"]),
]


def apply_edits(src, edits, name):
    for old, new in edits:
        n = src.count(old)
        if n != 1:
            raise AssertionError(f"mutation {name!r}: snippet occurs {n} times (need exactly 1): {old[:70]!r}")
        src = src.replace(old, new)
    return src


def main():
    src = open(SRC_PATH, encoding="utf-8").read()
    real = suite(load_module(src, "real"), "real")
    assert not real, f"the unmutated ctx_cache.py FAILS its own checks: {real}"
    n, problems = 1, []
    print(f"{'mutation':72s} caught-by (failing checks)")
    for name, edits, expect in MUTATIONS:
        m_src = apply_edits(src, edits, name)
        try:
            ast.parse(m_src)
            failing = suite(load_module(m_src, f"m{n}"), f"m{n}")
        except (Exception, SystemExit) as e:                           # noqa: BLE001 -- a mutated source that breaks the suite's own setup is RED, and said so
            failing = [f"suite setup raised {type(e).__name__}: {str(e)[:80]}"]
        if not failing:
            problems.append(f"mutation {name!r} stayed GREEN: the suite cannot see it")
            continue
        for c in expect:
            if c not in failing and not any(f.startswith("suite setup raised") for f in failing):
                problems.append(f"mutation {name!r}: the check {c} must go RED and stayed green (RED: {failing[:8]})")
        print(f"{name[:72]:72s} {', '.join(failing[:3])}", flush=True)
        n += 1
    if problems:
        print("\n".join("PROBLEM " + p for p in problems))
        print(f"ZZTEST_FAIL {len(problems)} problem(s)")
        return 1
    print(f"ZZTEST_PASS {n}  (1 unmutated suite + {len(MUTATIONS)} RED mutations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
