#!/usr/bin/env python3
"""Self-test of eval/aux_navtest_forward.py.  Run (driverl-eval venv; CPU; no pytest, no GPU, no nuPlan):
    PYTHONIOENCODING=utf-8 C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe eval/test_aux_navtest_forward.py     -> ZZTEST_PASS <n>

Fixtures are a TINY random REFe (width 32, 1 trunk block, 4 proposals, 32 x 64 images, 4 cameras = 32 visual tokens of width 256) and heads built by the trainer's own builder.
Expected values are derived from the construction (an identity, an exact zero, a deliberate perturbation of one float), never from the module under test.  Each RED mutation edits the
module's SOURCE (the snippet must occur exactly once -- a vacuous mutation proves nothing) and the named check must then FAIL.

 (a) a head on CACHED ctx == the same head on the ctx inline (exact), and the tap IS the scene_proj call that produces visual_ctx
 (b) a head whose output layer is zeroed gives logit EXACTLY 0.0 (and the un-zeroed head does not: the probe can read a non-zero)
 (c) two heads with different seeds give different outputs; the same seed reproduces bit for bit; the fixture file == the trainer's own container head
 (d) the npz written round-trips through aux_s1_readout.load_head_pred and, on the real pinned navtest store, passes the readout's guards (and a train key is REFUSED)
 deliberate regressions: swap two camera slots before the trunk => the scorer identity FAILS; strict=False with a missing key => the loader must still refuse
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(HERE, "aux_navtest_forward.py")
PKG = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PKG, "refe"))
import torch  # noqa: E402
import aux_drivable_head as AUX  # noqa: E402
import model as MD  # noqa: E402


def load(src: str, name: str):
    m = types.ModuleType(name)
    m.__file__ = SRC_PATH
    sys.modules[name] = m
    exec(compile(src, SRC_PATH, "exec"), m.__dict__)
    return m


# ------------------------------------------------------------------------------------------------------------------ fixtures
def tiny_model(seed: int = 0):
    torch.manual_seed(seed)
    cfg = MD.REFeConfig(backbone="vits16", img_h=32, img_w=64, width=32, depth=1, heads=2, lora_rank=2, n_registers=2, reg_heads=2, n_proposals=4, horizon_steps=4,
                        dec_depth=1, n_score_components=7)
    return MD.REFe(cfg).eval()


def tiny_inputs(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 4, 3, 32, 64, generator=g), torch.rand(1, 7, generator=g), torch.rand(1, 4, generator=g)


class FakePlanner:
    """Just the part of REFePlanner that `forward_token` uses: `infer(ego, img)` = ONE model call -> (traj [M,T,3], score [M,C], k)."""

    def __init__(self, model, ego_vec, goal):
        self.model, self.ego_vec, self.goal = model, ego_vec, goal

    def infer(self, ego, img):
        with torch.no_grad():
            traj, score = self.model(img, self.ego_vec, self.goal)
        return traj[0], score[0], 0


def tmpdir():
    return tempfile.mkdtemp(prefix="aux_navtest_fwd_test_")


# ------------------------------------------------------------------------------------------------------------------ checks
def chk_tap_is_visual_ctx(M):
    model = tiny_model()
    img, ego, goal = tiny_inputs()
    calls = []
    h = model.scene_proj.register_forward_hook(lambda mod, i, o: calls.append(o.detach()))
    tap = M.CtxTap(model)
    score, ctx = M.forward_token(FakePlanner(model, ego, goal), None, img, tap)
    h.remove()
    tap.close()
    assert len(calls) == 2, len(calls)                           # scene_proj(scene), then scene_proj(visual) == visual_ctx
    assert tuple(ctx.shape) == (1, 32, 256), ctx.shape           # 4 cameras x (32/16 x 64/16 = 8) patches, dec_width 256
    assert torch.equal(ctx, calls[1]), "the tap is not the visual_ctx call"
    assert score.dtype == np.float32 and score.shape == (4, 7)


def chk_cached_equals_inline(M):
    model = tiny_model()
    img, ego, goal = tiny_inputs()
    tap = M.CtxTap(model)
    _, ctx = M.forward_token(FakePlanner(model, ego, goal), None, img, tap)
    d = tmpdir()
    assert tuple(ctx.shape) == (1, 32, 256), ctx.shape          # the visual_ctx, not the score queries
    lg_in = np.arange(28, dtype=np.float32).reshape(4, 7) / 3.0
    assert M.cache_load(d, "tokA") is None
    M.cache_save(d, "tokA", ctx, lg_in)
    back, lg_back = M.cache_load(d, "tokA")
    assert torch.equal(back, ctx), "the fp32 ctx cache is not exact"
    assert np.array_equal(lg_back, lg_in) and lg_back.dtype == np.float32, "the cached scorer logits are not exact"
    assert M.cache_load(d, "nope") is None
    os.remove(M.logits_path(d, "tokA"))
    assert M.cache_load(d, "tokA") is None and not M.cache_complete(d, "tokA"), "a ctx without its logits must count as a MISS (the identity could not be judged)"
    for seed in (0, 1):
        f = M.make_fresh_head_file(os.path.join(d, f"h{seed}", "aux_drv.pt"), seed=seed)
        head, _ = M.load_aux_head(f)
        lg_i, sd_i = M.apply_head(head, ctx)
        lg_c, sd_c = M.apply_head(head, back)
        assert np.array_equal(lg_i, lg_c) and np.array_equal(sd_i, sd_c), f"seed {seed}: cached != inline"
        assert lg_i.shape == (1, 40, 32) and lg_i.dtype == np.float32 and sd_i.dtype == np.float32
        assert np.abs(lg_i).max() > 0, "a fresh head must read a non-zero somewhere"
        assert np.abs(sd_i).max() <= 6.0, "sdf_m is bounded by the head's 6 m tanh"


def chk_zeroed_output_is_exact_zero(M):
    model = tiny_model()
    img, ego, goal = tiny_inputs()
    tap = M.CtxTap(model)
    _, ctx = M.forward_token(FakePlanner(model, ego, goal), None, img, tap)
    d = tmpdir()
    head, _ = M.load_aux_head(M.make_fresh_head_file(os.path.join(d, "aux_drv.pt"), seed=0))
    lg0, _sd0 = M.apply_head(head, ctx)
    assert float(np.abs(lg0).max()) > 0.0, "control: the un-zeroed head must read non-zero (else the zero below proves nothing)"
    with torch.no_grad():
        head.out.weight.zero_()
        head.out.bias.zero_()
    lg, sd = M.apply_head(head, ctx)
    assert bool(np.all(lg == 0.0)), f"zeroed output layer: logit max |.| {np.abs(lg).max()}"
    assert bool(np.all(sd == 0.0)), "zeroed output layer: sdf must be 6*tanh(0) == 0.0"


def chk_sdf_in_metres(M):
    """The head's own output is already metres: a saturated sdf channel reads exactly sdf_scale = 6 m (6 * tanh(20) == 6.0 in float32), the mask channel passes through untouched."""
    model = tiny_model()
    img, ego, goal = tiny_inputs()
    tap = M.CtxTap(model)
    _, ctx = M.forward_token(FakePlanner(model, ego, goal), None, img, tap)
    head, _ = M.load_aux_head(M.make_fresh_head_file(os.path.join(tmpdir(), "aux_drv.pt"), seed=0))
    with torch.no_grad():
        head.out.weight.zero_()
        head.out.bias.copy_(torch.tensor([-3.5, 20.0]))
    lg, sd = M.apply_head(head, ctx)
    assert bool(np.all(sd == 6.0)), f"sdf_m is not the head's metres: {np.unique(sd)[:3]}"
    assert bool(np.all(lg == -3.5)), "the mask logit must be written untouched"


def chk_seeds_differ_and_fixture_is_trainers_head(M):
    model = tiny_model()
    img, ego, goal = tiny_inputs()
    tap = M.CtxTap(model)
    _, ctx = M.forward_token(FakePlanner(model, ego, goal), None, img, tap)
    d = tmpdir()
    out = {}
    for tag, seed in (("a", 0), ("b", 1), ("a2", 0)):
        head, info = M.load_aux_head(M.make_fresh_head_file(os.path.join(d, tag, "aux_drv.pt"), seed=seed))
        out[tag] = M.apply_head(head, ctx)
        assert info["n_params"] == 2440450, info["n_params"]          # SPEC: 2,440,450 parameters at depth 2
    assert float(np.abs(out["a"][0] - out["b"][0]).max()) > 1e-4 and float(np.abs(out["a"][1] - out["b"][1]).max()) > 1e-4, "seeds 0 and 1 give the same head"
    assert np.array_equal(out["a"][0], out["a2"][0]) and np.array_equal(out["a"][1], out["a2"][1]), "the same seed does not reproduce bit for bit"
    # the fixture equals what the trainer's own container builds (AuxDrivable(AuxDrivableConfig(seed)).head), tensor for tensor
    box = AUX.AuxDrivable(AUX.AuxDrivableConfig(tap="visual", depth=2, feed="off", w_aux=0.0, seed=0))
    sd = torch.load(os.path.join(d, "a", "aux_drv.pt"), weights_only=False)
    assert sd["format"] == "aux_drv/1" and set(sd) == {"format", "head", "feed", "opt", "update", "meta"}
    assert set(sd["meta"]) == {"config", "store_sha256", "frames", "shuffle_seed", "head_file"}, "meta keys differ from scorer_finetune.aux_meta()"
    assert set(sd["head"]) == set(box.head.state_dict())
    assert all(torch.equal(sd["head"][k], v) for k, v in box.head.state_dict().items())


def chk_npz_roundtrip_through_readout(M):
    import aux_s1_readout as R
    d = tmpdir()
    rng = np.random.default_rng(3)
    n = 5
    logs = [f"log{i % 2}" for i in range(n)]
    toks = [f"tok{i:02d}" for i in range(n)]
    lg = rng.normal(size=(n, 40, 32)).astype(np.float32)
    sd = (6.0 * np.tanh(rng.normal(size=(n, 40, 32)))).astype(np.float32)
    p = M.write_head_pred(os.path.join(d, "head_pred_H2-s0.npz"), logs, toks, lg, sd, {"arm": "H2-s0"})
    z = np.load(p, allow_pickle=False)
    assert set(z.files) >= {"log", "token", "logit", "sdf_m", "step"} and z["step"].dtype == np.int64 and not z["step"].any()
    assert z["logit"].dtype == np.float32 and z["sdf_m"].dtype == np.float32 and z["logit"].shape == (n, 40, 32)
    got = R.load_head_pred(p)
    assert got["keys"] == list(zip(logs, toks)), got["keys"]
    assert np.array_equal(got["logit"], lg) and np.array_equal(got["sdf"], sd), "load_head_pred did not return the arrays written"
    assert got["dropped_step_nonzero"] == 0 and got["duplicate_keys_identical"] == 0
    for name, a_, b_ in (("n-1 rows", lg[:-1], sd[:-1]), ("32 x 40", lg.reshape(n, 32, 40), sd.reshape(n, 32, 40)), ("sdf shape differs", lg, sd[:, :, :31])):
        try:
            M.write_head_pred(os.path.join(d, "bad.npz"), logs, toks, a_, b_, {})
        except ValueError:
            continue
        raise AssertionError(f"write_head_pred accepted {name}")


def chk_select_tokens(M):
    toks = [f"t{i}" for i in range(10)]
    tl = {t: ("A" if i in (1, 4, 7) else "B" if i in (0, 2) else f"C{i}") for i, t in enumerate(toks)}
    sel = M.select_tokens(toks, tl, 4)
    assert sel == ["t0", "t1", "t4", "t7"], sel                  # log A (3 tokens) whole, then log B up to the limit; the INPUT order is kept
    assert M.select_tokens(toks, tl, None) == toks and M.select_tokens(toks, tl, 10) == toks


def chk_identity_exactness(M):
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(6, 4, 7)).astype(np.float32)
    toks = [f"t{i}" for i in range(6)]
    ok = M.identity_report(ref.copy(), ref, toks)
    assert ok["PASS"] and ok["max_abs_delta"] == 0.0 and ok["n_tokens_bit_identical"] == 6 and ok["n_values_differ"] == 0
    bad = ref.copy()
    bad[3, 2, 5] = np.nextafter(bad[3, 2, 5], np.float32(np.inf))             # ONE value, ONE ulp
    r = M.identity_report(bad, ref, toks)
    assert not r["PASS"], "a single-ulp difference passed the bit-identity bar"
    assert r["n_values_differ"] == 1 and r["n_tokens_bit_identical"] == 5 and r["worst_tokens"][0][0] == "t3"
    assert r["max_abs_delta"] == float(abs(np.float64(bad[3, 2, 5]) - np.float64(ref[3, 2, 5])))
    assert not M.identity_report(ref[:, :3], ref, toks)["PASS"]


def chk_swap_regression_fails_identity(M):
    model = tiny_model()
    ego_vec, goal = tiny_inputs(1)[1:]
    pl = FakePlanner(model, ego_vec, goal)
    tap = M.CtxTap(model)
    imgs = [tiny_inputs(10 + i)[0] for i in range(3)]
    ref = np.stack([M.forward_token(pl, None, im, tap)[0] for im in imgs])
    again = np.stack([M.forward_token(pl, None, im, tap)[0] for im in imgs])
    r0 = M.identity_report(again, ref, ["a", "b", "c"])
    assert r0["PASS"] and r0["max_abs_delta"] == 0.0, "control: the same forward twice must be bit-identical on CPU"
    sw = np.stack([M.forward_token(pl, None, im, tap, swap=(1, 2))[0] for im in imgs])
    r1 = M.identity_report(sw, ref, ["a", "b", "c"])
    assert not r1["PASS"] and r1["max_abs_delta"] > 0.0 and r1["n_tokens_bit_identical"] == 0, f"swapping two camera slots did not break the identity: {r1}"
    x = tiny_inputs(5)[0]
    assert torch.equal(M.swap_cam_slots(M.swap_cam_slots(x, 1, 2), 1, 2), x) and not torch.equal(M.swap_cam_slots(x, 1, 2), x)


def chk_strict_load_refuses(M):
    d = tmpdir()
    f = M.make_fresh_head_file(os.path.join(d, "aux_drv.pt"), seed=0)
    M.load_aux_head(f)                                                         # control: the intact file loads
    sd = torch.load(f, weights_only=False)

    def refuses(mutated, why):
        p = os.path.join(d, "m.pt")
        torch.save(mutated, p)
        try:
            M.load_aux_head(p)
        except (RuntimeError, ValueError, KeyError):                           # HeadRefused is a RuntimeError; a strict mismatch is a RuntimeError
            return
        raise AssertionError(f"the loader accepted {why}")
    miss = {k: v for k, v in sd.items()}
    miss["head"] = {k: v for k, v in sd["head"].items() if k != "out.bias"}
    refuses(miss, "a head file MISSING a key (partial load)")
    extra = dict(sd)
    extra["head"] = dict(sd["head"], bogus=torch.zeros(1))
    refuses(extra, "a head file with an UNEXPECTED key")
    refuses(dict(sd, format="aux_drv/0"), "a wrong format tag")
    scene = dict(sd)
    scene["meta"] = dict(sd["meta"], config=dict(sd["meta"]["config"], tap="scene"))
    refuses(scene, "tap == scene")
    shallow = dict(sd)
    shallow["meta"] = dict(sd["meta"], config=dict(sd["meta"]["config"], depth=1))
    refuses(shallow, "a depth-1 config over depth-2 weights")
    for good in ("H1-s0", "H2-s3", "H1u10k-s0", "H1u2k-s0"):
        assert M.parse_arm(good)
    for bad in ("N-s0", "H3-s0", "H2", "H2-s", "H1u3k-s0"):
        try:
            M.parse_arm(bad)
        except RuntimeError:
            continue
        raise AssertionError(f"arm name {bad!r} accepted")


def chk_gpu_lock_protocol(M):
    d = tmpdir()
    lock = os.path.join(d, "devbox_gpu.lock")
    body = M.acquire_gpu_lock(lock=lock, wait_idle=None, used_fn=lambda: 1000, settle_s=0.0)
    j = json.loads(open(lock, encoding="utf-8").read())
    assert set(j) == {"pid", "owner", "t"} and j["pid"] == os.getpid() and j["owner"] == "aux_navtest_forward"
    try:
        M.acquire_gpu_lock(lock=lock, wait_idle=None, used_fn=lambda: 1000, settle_s=0.0)
    except M.GpuBusy:
        pass
    else:
        raise AssertionError("took the GPU although the lock exists and its pid is alive")
    assert os.path.exists(lock), "a refused acquire must not touch the holder's lock"
    M.release_gpu_lock(lock, body)
    assert not os.path.exists(lock)
    # a lock held by someone else's pid is never released by us
    open(lock, "w", encoding="utf-8").write(json.dumps({"pid": os.getpid() + 7, "owner": "other", "t": 0}))
    M.release_gpu_lock(lock, body)
    assert os.path.exists(lock)
    # a dead holder: removed only when old (> 600 s); a fresh one is respected
    open(lock, "w", encoding="utf-8").write(json.dumps({"pid": 4000001, "owner": "dead", "t": 0}))
    try:
        M.acquire_gpu_lock(lock=lock, wait_idle=None, used_fn=lambda: 1000, settle_s=0.0)
    except M.GpuBusy:
        pass
    else:
        raise AssertionError("removed a FRESH lock of a dead pid")
    old = time.time() - 3600
    os.utime(lock, (old, old))
    b2 = M.acquire_gpu_lock(lock=lock, wait_idle=None, used_fn=lambda: 1000, settle_s=0.0)
    assert json.loads(open(lock, encoding="utf-8").read())["owner"] == "aux_navtest_forward"
    M.release_gpu_lock(lock, b2)
    # memory still held after taking the lock: give it back and refuse
    try:
        M.acquire_gpu_lock(lock=lock, wait_idle=None, used_fn=lambda: 5000, settle_s=0.0)
    except M.GpuBusy:
        assert not os.path.exists(lock), "the lock was kept although memory.used > 3500 MiB"
    else:
        raise AssertionError("took the GPU at 5000 MiB used")
    # the idle waiter failing closed
    bad_wait = os.path.join(d, "never_idle.py")
    open(bad_wait, "w").write("import sys; sys.exit(3)\n")
    try:
        M.acquire_gpu_lock(lock=lock, wait_idle=bad_wait, used_fn=lambda: 1000, settle_s=0.0)
    except M.GpuBusy:
        assert not os.path.exists(lock)
    else:
        raise AssertionError("wait_idle exited 3 and the lock was taken anyway")


CHECKS = [chk_tap_is_visual_ctx, chk_cached_equals_inline, chk_zeroed_output_is_exact_zero, chk_sdf_in_metres, chk_seeds_differ_and_fixture_is_trainers_head, chk_npz_roundtrip_through_readout,
          chk_select_tokens, chk_identity_exactness, chk_swap_regression_fails_identity, chk_strict_load_refuses, chk_gpu_lock_protocol]
BYNAME = {f.__name__: f for f in CHECKS}


def run_checks(M, names=None):
    failing = []
    for f in (CHECKS if names is None else [BYNAME[n] for n in names]):
        try:
            f(M)
        except Exception as e:                                                  # noqa: BLE001 -- a RED mutation may fail in any way
            failing.append(f"{f.__name__}[{type(e).__name__}: {str(e)[:90]}]")
    return failing


# ------------------------------------------------------------------------------------------------------------------ real-store guards (d)
def real_store_checks(M):
    """(d) with the real pinned navtest store: the npz the tool writes passes the readout's guards with --labels-set navtest, and a TRAIN key is refused."""
    import aux_s1_readout as R
    msgs = []
    nav_dir, tr_dir = R.LABEL_DIRS["navtest"], R.LABEL_DIRS["train"]
    if not (os.path.isdir(nav_dir) and os.path.isdir(tr_dir)):
        raise AssertionError(f"the pinned label stores are missing ({nav_dir} / {tr_dir}): (d)'s readout guards cannot be exercised")
    nav, tr = R.open_labels("navtest"), R.open_labels("train")
    nk, tk = sorted(nav.all_keys())[:6], sorted(tr.all_keys())[:1]
    assert len(nk) == 6 and len(tk) == 1
    d = tmpdir()
    rng = np.random.default_rng(1)

    def write(keys, name):
        n = len(keys)
        return M.write_head_pred(os.path.join(d, name), [k[0] for k in keys], [k[1] for k in keys], rng.normal(size=(n, 40, 32)).astype(np.float32),
                                 (6 * np.tanh(rng.normal(size=(n, 40, 32)))).astype(np.float32), {"arm": "H2-s0"})
    ok = write(nk, "head_pred_H2-s0.npz")
    out = os.path.join(d, "p.json")
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = R.main(["perception", "--pred", f"H2-s0={ok}", "--labels-set", "navtest", "--n-boot", "50", "--out", out])
    assert rc == 0 and "ZZAUXS1_PERCEPTION_OK" in buf.getvalue(), (rc, buf.getvalue()[-300:])
    res = json.load(open(out, encoding="utf-8"))
    assert res["labels"]["set"] == "navtest" and "H2-s0" in res["arms"]
    msgs.append("real navtest store: tool npz -> readout perception rc 0")
    bad = write(nk[:3] + tk, "head_pred_H1-s0.npz")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = R.main(["perception", "--pred", f"H1-s0={bad}", "--labels-set", "navtest", "--n-boot", "50", "--out", os.path.join(d, "q.json")])
    assert rc == 3 and "ZZAUXS1_REFUSED" in buf.getvalue(), (rc, buf.getvalue()[-300:])
    msgs.append("real stores: an npz carrying a TRAIN key is REFUSED (rc 3)")
    return msgs


# ------------------------------------------------------------------------------------------------------------------ mutations
MUT = [
    ("strict_false", "guard", "head.load_state_dict(sd[\"head\"], strict=True)", "head.load_state_dict(sd[\"head\"], strict=False)", ["chk_strict_load_refuses"]),
    ("identity_tolerance", "exactness", "exact = np.array_equal(new, ref)", "exact = bool(np.allclose(new, ref, rtol=1e-5, atol=1e-5))", ["chk_identity_exactness"]),
    ("swap_is_noop", "regression", "out[:, [i, j]] = img[:, [j, i]]", "out[:, [i, j]] = img[:, [i, j]]", ["chk_swap_regression_fails_identity"]),
    ("tap_reads_queries", "tap", "self.ctx = args[1]", "self.ctx = args[0]", ["chk_tap_is_visual_ctx", "chk_cached_equals_inline"]),
    ("step_not_zero", "contract", "step=np.zeros(n, np.int64)", "step=np.ones(n, np.int64)", ["chk_npz_roundtrip_through_readout"]),
    ("lock_without_excl", "lock", "os.O_CREAT | os.O_EXCL | os.O_WRONLY", "os.O_CREAT | os.O_WRONLY", ["chk_gpu_lock_protocol"]),
    ("wait_idle_ignored", "lock", "if r.returncode != 0:\n            raise GpuBusy(f\"wait_idle.py exited {r.returncode}\")", "if False:\n            raise GpuBusy(\"x\")", ["chk_gpu_lock_protocol"]),
    ("sdf_not_metres", "contract", "return logit.float().cpu().numpy().astype(np.float32), sdf.float().cpu().numpy().astype(np.float32)",
     "return logit.float().cpu().numpy().astype(np.float32), (sdf / 6.0).float().cpu().numpy().astype(np.float32)", ["chk_sdf_in_metres"]),
]


def main() -> int:
    t0 = time.time()
    src = open(SRC_PATH, encoding="utf-8").read()
    real = load(src, "aux_navtest_forward_real")
    n = 0
    print(f"module under test: {SRC_PATH}")
    print(f"head module: {AUX.__file__}")
    assert os.path.dirname(os.path.abspath(AUX.__file__)).replace("\\", "/").startswith("D:/Projects/TanitAD"), "the head module is not the D: copy"
    assert os.path.dirname(os.path.abspath(MD.__file__)).replace("\\", "/").startswith("D:/Projects/TanitAD"), "the model module is not the D: copy"
    failing = run_checks(real)
    assert not failing, f"the unmutated module FAILS its own checks: {failing}"
    n += len(CHECKS)
    print(f"unmutated module: {len(CHECKS)} checks GREEN")
    extra = real_store_checks(real)
    n += len(extra)
    for e in extra:
        print(f"GREEN {e}")
    print(f"{'mutation':22s} {'class':11s} must go RED -> went RED")
    for i, (name, cls, old, new, expect) in enumerate(MUT):
        cnt = src.count(old)
        assert cnt == 1, f"mutation {name}: the snippet occurs {cnt} times (must be exactly 1) -- a vacuous mutation proves nothing"
        try:
            mod = load(src.replace(old, new), f"aux_navtest_forward_mut{i}")
        except SyntaxError as e:
            raise AssertionError(f"mutation {name} does not compile: {e}")
        red = run_checks(mod, expect)
        red_names = {r.split("[")[0] for r in red}
        missing = [c for c in expect if c not in red_names]
        assert not missing, f"mutation {name} ({cls}) stayed GREEN on {missing}: the suite cannot see it"
        print(f"{name:22s} {cls:11s} {', '.join(sorted(red_names))}")
        n += 1
    print(f"ZZTEST_PASS {n}  ({len(CHECKS)} checks + {len(extra)} real-store + {len(MUT)} RED mutations)  {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
