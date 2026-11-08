"""The refcv9 E2 ROLL harness (`stack/ops/r9e2_read/`): the DEV-only window filter, the per-call STAGING proxy, the env-pin refusals, the INPUTS.json writer and the Thor
shell driver (`r9e2_roll.sh`, proved by a DRY RUN).  E2-ROLL-2, 2026-11-04.

Everything imports the harness's REAL modules (nothing is re-implemented here); the only fakes are the things a dev box cannot have (a model, the r7_roll module, the
trainer's `_bw` / `_bc5` mechanisms for the order / one-shot / batch-size contract).  Each guard has a deliberate-regression (RED) arm that must go RED:

  * DEV filter     RED: the UNFILTERED window list carries HOLD clips (the filter is what removes them); a HOLD clip kept => refuses; a per-clip / total / missing-clip /
                   empty count mismatch => refuses; an identity filter fails the no-HOLD invariant.
  * staging        RED: the bare proxy (no staging) => the fake model raises; stages in the wrong ORDER (_bc5 before _bw) => raises; staged ONCE per window instead of per
                   model call (the proxy makes up to four calls of two batch sizes) => raises; the REAL `_bc5.stage` / `_bw.stage` REFUSE an unstaged batch of a built mechanism.
  * shell driver   RED: removing `--lock-held` / `--no-baselines` / `--require-seal` / the K0 dump / `--lift-valid`, or adding a placeholder `--s6c-k2`, each makes the
                   structural check of the DRY RUN go RED; with the pin guard removed an unset pin is no longer refused (rc != 4).
  * S6c            a null S6c reaches the REAL judge as NOT-COMPUTABLE (clause c) even when every other clause PASSes; a placeholder S6c file would read as evidence.
  * S6c WIRE       (E2-S6C-WIRE, 2026-11-08) the MAPSEL roll of K2 and K0 at seeds 0 and 1 runs inside the one lock hold, through the DEV wrapper, with --own-planes derived from
                   the run's config; the S6c file is per arm per seed and the writer refuses a seed / arm swap or the old single-file flag; the shell STOPS (exit 5) before INPUTS when a
                   score is missing.  RED: per-seed swap; one path on both seeds; a non-DEV windows list or the bare roll (HOLD reaches the roll); banded K2 given `require`; the S6c roll
                   outside the lock hold; the gate removed.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
R9DIR = ROOT / "ops" / "r9e2_read"
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests"), str(R9DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import test_launch_gate_e1rig_e1rec2 as R2  # noqa: E402

import r9_e2_verdict as V  # noqa: E402
import r9e2_inputs as INP  # noqa: E402
import run_battery_r9_e2 as R9  # noqa: E402

BASH = R2.BASH
needs_bash = R2.needs_bash
SH = R9DIR / "r9e2_roll.sh"
ARM_READ = ROOT / "ops" / "r9e1_read" / "r9e1_arm_read.sh"


@pytest.fixture(autouse=True)
def _restore_imports():
    """The import-hook / sys.path tests must not leak into the next test."""
    mp, path, mods = list(sys.meta_path), list(sys.path), set(sys.modules)
    yield
    sys.meta_path[:] = mp
    sys.path[:] = path
    for m in set(sys.modules) - mods:
        sys.modules.pop(m, None)


# ============================================================ (i) the DEV-only window filter
def _world():
    dev = [f"dev-{i}" for i in range(4)]
    hold = [f"hold-{i}" for i in range(3)]
    full = {c: list(range(100, 100 + 7 * (1 + i % 3), 7)) for i, c in enumerate(dev + hold)}

    def blk(cs):
        return {"sha12": [R9.sha12(c) for c in cs], "n_windows_s2": sum(len(full[c]) for c in cs),
                "windows_per_episode": {R9.sha12(c): [len(full[c]), 0] for c in cs}}
    split = {"DEV": blk(dev), "HOLD": blk(hold)}

    def orig(baseline_dump, kit_clip_ids):
        return {c: list(w) for c, w in full.items()}
    return dev, hold, full, split, orig


def _no_hold(out, hold):
    assert not any(R9.sha12(c) in hold for c in out), "a HOLD clip is in the window list"


def test_sha12_is_the_loaders():
    from tanitad.eval import refcv7_loader as L7
    for c in ("dev-0", "0a1b2c3d-0000-4000-8000-123456789abc", "x"):
        assert R9.sha12(c) == L7.sha12(c)


def test_dev_filter_keeps_only_dev_and_counts(tmp_path):
    dev, hold, full, split, orig = _world()
    p = tmp_path / "split.json"
    p.write_text(json.dumps(split), encoding="utf-8")
    d, h, j = R9.load_dev_hold(p)
    assert d == {R9.sha12(c) for c in dev} and h == {R9.sha12(c) for c in hold}
    census = {}
    out = R9.make_dev_window_filter(orig, d, h, j, census)(None, None)
    assert set(out) == set(dev)
    _no_hold(out, h)
    assert all(out[c] == full[c] for c in dev)
    assert census["n_clips_all"] == 7 and census["n_clips_dev"] == 4 and census["n_clips_dropped"] == 3
    assert census["n_windows_dev"] == split["DEV"]["n_windows_s2"]
    assert census["n_windows_dropped"] == split["HOLD"]["n_windows_s2"]


def test_RED_unfiltered_list_carries_hold_and_an_identity_filter_fails_the_invariant():
    dev, hold, full, split, orig = _world()
    h = {R9.sha12(c) for c in hold}
    with pytest.raises(AssertionError):
        _no_hold(orig(None, None), h)                     # the raw list: HOLD clips present => the filter is doing the work
    identity = lambda b, k: orig(b, k)                    # noqa: E731
    with pytest.raises(AssertionError):
        _no_hold(identity(None, None), h)
    _no_hold(R9.make_dev_window_filter(orig, {R9.sha12(c) for c in dev}, h, split)(None, None), h)


def test_RED_hold_clip_surviving_the_filter_refuses():
    dev, hold, full, split, orig = _world()
    d = {R9.sha12(c) for c in dev} | {R9.sha12(hold[0])}          # a corrupted DEV set that includes a HOLD clip (load_dev_hold would refuse it; this is the 2nd line)
    h = {R9.sha12(c) for c in hold}
    with pytest.raises(SystemExit, match="HOLD clip survived"):
        R9.make_dev_window_filter(orig, d, h)(None, None)


def test_RED_count_mismatches_refuse():
    dev, hold, full, split, orig = _world()
    d, h = {R9.sha12(c) for c in dev}, {R9.sha12(c) for c in hold}
    short = lambda b, k: {c: (w[:-1] if c == dev[1] else w) for c, w in orig(b, k).items()}                    # noqa: E731  one window of one clip lost
    with pytest.raises(SystemExit, match="differ from DEV_HOLD_SPLIT"):
        R9.make_dev_window_filter(short, d, h, split)(None, None)
    gone = lambda b, k: {c: w for c, w in orig(b, k).items() if c != dev[2]}                                    # noqa: E731  a DEV clip missing from the kit
    with pytest.raises(SystemExit, match="differ from DEV_HOLD_SPLIT"):
        R9.make_dev_window_filter(gone, d, h, split)(None, None)
    bad_total = json.loads(json.dumps(split))
    bad_total["DEV"]["n_windows_s2"] += 1
    with pytest.raises(SystemExit, match="differ from DEV_HOLD_SPLIT"):
        R9.make_dev_window_filter(orig, d, h, bad_total)(None, None)
    only_hold = lambda b, k: {c: w for c, w in orig(b, k).items() if c in hold}                                 # noqa: E731
    with pytest.raises(SystemExit, match="kept no clip"):
        R9.make_dev_window_filter(only_hold, d, h, split)(None, None)


def test_RED_load_dev_hold_refuses_overlap_and_empty(tmp_path):
    for i, split in enumerate(({"DEV": {"sha12": ["a", "b"]}, "HOLD": {"sha12": ["b", "c"]}},
                               {"DEV": {"sha12": []}, "HOLD": {"sha12": ["c"]}},
                               {"DEV": {"sha12": ["a"]}, "HOLD": {"sha12": []}})):
        p = tmp_path / f"s{i}.json"
        p.write_text(json.dumps(split), encoding="utf-8")
        with pytest.raises(SystemExit, match="not disjoint|empty"):
            R9.load_dev_hold(p)


# ============================================================ (ii) the staging proxy
def _torch():
    import torch
    return torch


class FakeModel:
    """A forward that raises unless BOTH mechanisms staged a batch of exactly its own size (one-shot), like a built pfq + temporal-warp model."""

    def __init__(self, built=True):
        self.staged, self.order, self.calls, self.agent_eps = {}, [], [], []
        self._perception = types.SimpleNamespace(box_pfq=object() if built else None)
        self._r9_bev_warp = object() if built else None

    def modules(self):
        return [self]

    def __call__(self, frames, **kw):
        n = int(frames.shape[0])
        got = dict(self.staged)
        self.staged.clear()                                         # one-shot, like slot_query_select / the warp fuse
        if got.get("_bw") != n or got.get("_bc5") != n:
            raise RuntimeError(f"unstaged forward n={n} staged={got}")
        self.calls.append(n)
        return n


class FakeStage:
    def __init__(self, label):
        self.label = label

    def stage(self, model, batch, device):
        n = int(batch["pose_hist"].shape[0])
        if self.label == "_bc5":
            if "_bw" not in model.staged:
                raise RuntimeError("order: _bc5 staged before _bw")
            assert int(batch["agent_ep"].shape[0]) == n
            model.agent_eps.append(int(batch["agent_ep"][0]))
        model.staged[self.label] = n
        model.order.append(self.label)


class FakeBase:
    """Refcv6Proxy's shape: state in __dict__, reads model.modules() on the RAW model, makes four model calls of two batch sizes per window."""

    def __init__(self, model, ctx, *a, **kw):
        d = self.__dict__
        d["model"], d["ctx"], d["mods"] = model, ctx, list(model.modules())

    def __call__(self, frames, **kw):
        m = self.__dict__["model"]
        return [m(frames), m(frames), m(frames[:1]), m(frames[:1])]


def _ctx(torch, ep_index=2):
    eps = [types.SimpleNamespace(episode_id=1000 + i) for i in range(5)]
    ctx = types.SimpleNamespace(item={"pose_hist": torch.zeros(4, 3), "v_max_ms": torch.tensor(5.0), "name": "not-a-tensor"}, index=(ep_index, 40))
    return eps, ctx


def _stager(eps, mods=None):
    return R9.Stager(mods or [("_bw", FakeStage("_bw")), ("_bc5", FakeStage("_bc5"))], episodes_of=lambda: eps)


def test_item_batch_expands_views_and_skips_non_tensors():
    torch = _torch()
    _, ctx = _ctx(torch)
    b = R9.item_batch(ctx.item, 3)
    assert set(b) == {"pose_hist", "v_max_ms"} and b["pose_hist"].shape == (3, 4, 3) and b["v_max_ms"].shape == (3,)
    assert b["pose_hist"].data_ptr() == ctx.item["pose_hist"].data_ptr()           # a VIEW: no copy of a window's frames


def test_staging_proxy_stages_every_inner_call_in_order_with_the_call_batch_size():
    torch = _torch()
    eps, ctx = _ctx(torch)
    st, model = _stager(eps), FakeModel()
    proxy = R9.make_staging_proxy_cls(FakeBase, st)(model, ctx)
    assert proxy.__dict__["mods"] == [model]                                         # super().__init__ read the RAW model
    assert proxy(torch.zeros(3, 3, 8, 8)) == [3, 3, 1, 1]
    assert model.calls == [3, 3, 1, 1] and model.order == ["_bw", "_bc5"] * 4
    assert model.agent_eps == [1002] * 4                                              # int(ds.episodes[ctx.index[0]].episode_id), the trainer's agent_ep
    c = st.census()
    assert c["n_forwards"] == 4 and c["n_stage_calls"] == {"_bw": 4, "_bc5": 4} and c["batch_sizes"] == {3: 2, 1: 2} and c["order"] == ["_bw", "_bc5"]
    assert c["mechanisms_built"] == {"bc5_pfq_built": True, "bw_built": True}


def test_K0_like_run_passes_with_the_real_noop_stages():
    torch = _torch()
    eps, ctx = _ctx(torch)

    class Plain:                                                                          # a model that built neither mechanism (K0)
        def modules(self):
            return [self]

        def __call__(self, frames, **kw):
            return int(frames.shape[0])
    st = R9.Stager(R9.default_mods(), episodes_of=lambda: eps)                           # the REAL _bw / _bc5 modules: both are no-ops when nothing is built
    assert R9.make_staging_proxy_cls(FakeBase, st)(Plain(), ctx)(torch.zeros(3, 3, 8, 8)) == [3, 3, 1, 1]
    assert st.n_forwards == 4 and st.census()["n_stage_calls"] == {"_bw": 4, "_bc5": 4}
    assert st.census()["mechanisms_built"] == {"bc5_pfq_built": False, "bw_built": False}
    assert [lab for lab, _ in R9.default_mods()] == ["_bw", "_bc5"]                      # the trainer's order (refc_v3_train.py:5920-5922)


def test_RED_bare_proxy_without_staging_raises():
    torch = _torch()
    _, ctx = _ctx(torch)
    with pytest.raises(RuntimeError, match="unstaged forward"):
        FakeBase(FakeModel(), ctx)(torch.zeros(3, 3, 8, 8))


def test_RED_wrong_stage_order_raises():
    torch = _torch()
    eps, ctx = _ctx(torch)
    st = _stager(eps, [("_bc5", FakeStage("_bc5")), ("_bw", FakeStage("_bw"))])
    with pytest.raises(RuntimeError, match="order"):
        R9.make_staging_proxy_cls(FakeBase, st)(FakeModel(), ctx)(torch.zeros(3, 3, 8, 8))


def test_RED_staging_once_per_window_instead_of_per_call_raises():
    torch = _torch()
    eps, ctx = _ctx(torch)

    class OncePerWindow(R9.Stager):
        done = False

        def stage(self, model, ctx, frames):
            if not self.done:
                self.done = True
                super().stage(model, ctx, frames)
    st = OncePerWindow([("_bw", FakeStage("_bw")), ("_bc5", FakeStage("_bc5"))], episodes_of=lambda: eps)
    with pytest.raises(RuntimeError, match="unstaged forward"):                       # the 2nd call (filter-off) finds the one-shot stage consumed
        R9.make_staging_proxy_cls(FakeBase, st)(FakeModel(), ctx)(torch.zeros(3, 3, 8, 8))


def test_RED_real_stages_refuse_an_unstaged_batch_of_a_built_mechanism():
    torch = _torch()
    _, ctx = _ctx(torch)
    pfq = types.SimpleNamespace(_perception=types.SimpleNamespace(box_pfq=object()))
    st = R9.Stager(R9.default_mods(), episodes_of=None)                               # no episode source and the item has no agent_ep => _bc5 REFUSES by delegation
    with pytest.raises(SystemExit, match="agent_ep"):
        st.stage(pfq, ctx, torch.zeros(2, 3, 8, 8))
    from tanitad.train import bev_warp_flag as _bw
    warp = types.SimpleNamespace(_map_hires=types.SimpleNamespace(**{_bw.HEAD_ATTR: object()}))
    with pytest.raises(SystemExit, match="bev-warp"):                                  # a built fuse and a batch with no bw_* keys
        R9.Stager([("_bw", _bw)], episodes_of=None).stage(warp, ctx, torch.zeros(2, 3, 8, 8))


def test_stager_episode_id_sources_and_staged_model_delegation():
    torch = _torch()
    eps, ctx = _ctx(torch)
    assert R9.Stager([], episodes_of=lambda: eps).episode_id(ctx, ctx.item) == 1002
    item = {"agent_ep": torch.tensor([77]), **{k: v for k, v in ctx.item.items()}}
    assert R9.Stager([], episodes_of=None).episode_id(types.SimpleNamespace(index=None), item) == 77
    assert R9.Stager([], episodes_of=None).episode_id(types.SimpleNamespace(index=None), ctx.item) is None
    model = FakeModel()
    sm = R9.StagedModel(model, ctx, _stager(eps))
    assert sm.modules() == [model] and sm._perception is model._perception
    sm.some_flag = 5
    assert model.some_flag == 5                                                       # set on the RAW model (the proxy flips speed_ceiling_filter this way)
    with pytest.raises(RuntimeError, match="without frames"):
        sm()
    with pytest.raises(RuntimeError, match="no window item"):
        R9.Stager([], None).stage(model, types.SimpleNamespace(item=None), torch.zeros(1, 3))


def test_install_staging_captures_the_dataset_and_rebinds_the_proxy():
    torch = _torch()
    eps, ctx = _ctx(torch)
    ds = types.SimpleNamespace(episodes=eps)
    RR = types.SimpleNamespace(L=types.SimpleNamespace(build_eval_dataset=lambda *a, **k: (ds, "eps", {"rec": 1})), Refcv6Proxy=FakeBase)
    st = R9.install_staging(RR, mods=[("_bw", FakeStage("_bw")), ("_bc5", FakeStage("_bc5"))])
    assert RR.R9E2_STAGER is st and RR.Refcv6Proxy is not FakeBase and issubclass(RR.Refcv6Proxy, FakeBase)
    assert RR.L.build_eval_dataset()[2] == {"rec": 1}                                  # the wrapped builder still returns the dataset tuple unchanged
    model = FakeModel()
    RR.Refcv6Proxy(model, ctx)(torch.zeros(2, 3, 8, 8))
    assert model.agent_eps == [1002] * 4


def test_hook_on_import_fires_after_the_module_body_exactly_once(tmp_path):
    (tmp_path / "r9e2_fake_mod.py").write_text("VALUE = 41\n", encoding="utf-8")
    sys.path.insert(0, str(tmp_path))
    seen = []
    R9.hook_on_import("r9e2_fake_mod", lambda m: seen.append(getattr(m, "VALUE", None)))
    import r9e2_fake_mod  # noqa: F401
    assert seen == [41]                                                                # the hook saw the module AFTER its body ran
    R9.hook_on_import("r9e2_fake_mod", lambda m: seen.append("late"))                 # already imported => immediate
    assert seen == [41, "late"]


def test_pin_repoint_asserts_the_env_pins_and_sets_the_remap(tmp_path):
    code = tmp_path / "code"
    code.mkdir()
    tree, kit = tmp_path / "tree", "E:/kit"
    (code / "r8_repoint.py").write_text(f"from pathlib import Path\nLAUNCH_TREE = Path({str(tree)!r})\nEVAL_KIT = {kit!r}\nREMAP_OVERRIDES = Path('devbox.json')\n", encoding="utf-8")
    env = {"PF_R8_CODE": str(code), "R8_LAUNCH_TREE": str(tree), "R8_EVAL_KIT": kit, "R9E2_REMAP": str(tmp_path / "empty.json")}
    RP = R9.pin_repoint(env)
    assert RP.REMAP_OVERRIDES == Path(env["R9E2_REMAP"])
    sys.modules.pop("r8_repoint", None)
    with pytest.raises(SystemExit, match="not the env pins"):                          # RED: r8_repoint read another tree than the pin
        R9.pin_repoint({**env, "R8_LAUNCH_TREE": str(tmp_path / "other")})


# ============================================================ (iii) env pins, entry points
def _full_env():
    return {k: f"/thor/{k.lower()}" for k in R9.ENV_PINS}


@pytest.mark.parametrize("pin", R9.ENV_PINS)
def test_require_env_refuses_each_unset_pin(pin):
    env = _full_env()
    assert R9.require_env(env) == env
    del env[pin]
    with pytest.raises(SystemExit, match=pin):
        R9.require_env(env)
    env[pin] = "  "                                                                    # blank == unset
    with pytest.raises(SystemExit, match=pin):
        R9.require_env(env)


def test_require_env_refuses_the_standalone_pose_clock():
    with pytest.raises(SystemExit, match="R8_NO_POSE_SYNC"):
        R9.require_env({**_full_env(), "R8_NO_POSE_SYNC": "1"})


def test_main_refuses_without_no_baselines_and_without_pins(monkeypatch):
    with pytest.raises(SystemExit, match="--no-baselines"):
        R9.main(["--tag", "t"])
    for k in R9.ENV_PINS:
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(SystemExit, match="unset env pin"):
        R9.main(["--no-baselines", "--tag", "t"])


def _fake_roll_tree(tmp_path):
    (tmp_path / "r7_roll.py").write_text(
        "import types\nclass Refcv6Proxy:\n    def __init__(self, model, ctx, *a, **k):\n        self.__dict__['model'] = model\n"
        "L = types.SimpleNamespace(build_eval_dataset=lambda *a, **k: (types.SimpleNamespace(episodes=[]), None, {}))\n", encoding="utf-8")
    (tmp_path / "fake_camblind.py").write_text(
        "import sys, types\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).resolve().parent))\nimport r7_roll\nimport torch\n"
        "if '--drive' in sys.argv:\n    ctx = types.SimpleNamespace(item={'pose_hist': torch.zeros(4, 3)}, index=(0, 1))\n"
        "    r7_roll.R9E2_STAGER.stage(types.SimpleNamespace(), ctx, torch.zeros(2, 3))\nsys.exit(0)\n", encoding="utf-8")
    return tmp_path / "fake_camblind.py"


def _cb_env(monkeypatch, tmp_path):
    for k in ("R8_LAUNCH_TREE", "PF_R8_CODE", "R8_EVAL_KIT", "L1_PF_CODE", "L1_R8_CODE"):
        monkeypatch.setenv(k, str(tmp_path / k))


def test_camblind_launcher_installs_the_hook_and_demands_a_staged_forward(monkeypatch, tmp_path):
    import camblind_roll_r9_e2 as CB
    script = _fake_roll_tree(tmp_path)
    _cb_env(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", ["x", "--camblind-script", str(script), "--drive"])
    with pytest.raises(SystemExit) as e:
        CB.main()
    assert e.value.code in (0, None)                                                   # the hook rebound r7_roll's proxy and one real forward was staged (K0-like no-ops)
    assert sys.modules["r7_roll"].R9E2_STAGER.n_forwards == 1


def test_RED_camblind_launcher_refuses_an_ok_roll_that_staged_nothing(monkeypatch, tmp_path):
    import camblind_roll_r9_e2 as CB
    script = _fake_roll_tree(tmp_path)
    _cb_env(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", ["x", "--camblind-script", str(script)])
    with pytest.raises(SystemExit, match="nothing was staged"):
        CB.main()
    monkeypatch.setattr(sys, "argv", ["x"])
    with pytest.raises(SystemExit, match="--camblind-script"):
        CB.main()
    monkeypatch.setattr(sys, "argv", ["x", "--camblind-script", str(script)])
    monkeypatch.delenv("L1_PF_CODE")
    with pytest.raises(SystemExit, match="L1_PF_CODE"):
        CB.main()


# ============================================================ INPUTS.json + the REAL judge
MAP = V.MAP_KEY


def _w(p, obj):
    Path(p).write_text(json.dumps(obj), encoding="utf-8")
    return str(p)


def _row(delta, sep, better):
    return {"delta": delta, "lo": delta - 0.01, "hi": delta + 0.01, "separated": sep, "better": better, "n": 100, "n_episodes": 20}


def _write_reads(rd, k2_beats=True):
    """Every judge input EXCEPT S6c: K2 beats K0 on the map primary and box AP, no family separated worse, camera-blind separated worse than K2."""
    rd.mkdir(exist_ok=True)
    _w(rd / "EVAL_K2.json", {"n_windows": 100, "n_clips": 20, "nearfield": {"ci": {MAP: {"point": 0.7 if k2_beats else 0.1}}, "n_files": 5}, "boxes": {"ap2m_all": {"point": 0.6}}})
    _w(rd / "EVAL_K0.json", {"nearfield": {"ci": {MAP: {"point": 0.4}}}, "boxes": {"ap2m_all": {"point": 0.3}}})
    fam = {"rows": {f: {"r": _row(-0.1, False, "lower")} for f in V.FAMILIES}}
    cb = {"rows": {k: _row(-0.2, True, "lower") for k in ("ade_m",) + V.FAMILIES}}
    for s in ("0", "1"):
        _w(rd / f"families_s{s}.json", fam)
        _w(rd / f"camblind_s{s}.json", cb)


def _s6c(d, label, seed, rate, trace_label=None, infer_seed=None):
    """One s6c_dev-shaped flat file (the keys the writer checks + the judge reads): per arm, per inference seed."""
    d.mkdir(exist_ok=True)
    return _w(d / INP.s6c_name(label, seed), {"rate": rate, "n": 900, "infer_seed": seed if infer_seed is None else infer_seed,
                                               "trace": f"/pull/E2_{trace_label or label}_mapsel/trace_s{seed}.npz"})


def _all_s6c(d, k2=(0.03, 0.04), k0=(0.10, 0.12)):
    for s in (0, 1):
        _s6c(d, "K2", s, k2[s])
        _s6c(d, "K0", s, k0[s])
    return d


def test_inputs_writer_matches_the_judges_schema_and_a_missing_s6c_is_null(tmp_path):
    rd = tmp_path / "read"
    _write_reads(rd)
    inp = INP.build_inputs(rd)
    assert set(inp["seeds"]) == {"0", "1"} and all(tuple(r) == INP.SEED_KEYS for r in inp["seeds"].values())
    assert all(r["s6c_k2"] is None and r["s6c_k0"] is None for r in inp["seeds"].values())
    assert inp["not_built"] == ["seed0.s6c_k2", "seed0.s6c_k0", "seed1.s6c_k2", "seed1.s6c_k0"]
    assert inp["seeds"]["1"]["families"].endswith("families_s1.json") and inp["seeds"]["0"]["camblind"].endswith("camblind_s0.json")
    out = tmp_path / "INPUTS.json"
    assert INP.main(["--rd", str(rd), "--out", str(out), "--s6c-dir", str(_all_s6c(tmp_path / "s6c"))]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["not_built"] == []


def test_the_s6c_files_land_per_seed_and_per_arm(tmp_path):
    """The PER-SEED file goes into ITS seed's slot: seed 0 never reads seed 1's file, K2 never reads K0's."""
    rd = tmp_path / "read"
    _write_reads(rd)
    inp = INP.build_inputs(rd, s6c_dir=_all_s6c(tmp_path / "s6c"))
    for s in ("0", "1"):
        assert inp["seeds"][s]["s6c_k2"].replace("\\", "/").endswith(f"S6C_K2_s{s}.json") and inp["seeds"][s]["s6c_k0"].replace("\\", "/").endswith(f"S6C_K0_s{s}.json")
    assert inp["seeds"]["0"]["s6c_k2"] != inp["seeds"]["1"]["s6c_k2"]
    v = V.build(inp)                                                                  # the REAL judge reads the two seeds' two different rates
    assert {c["id"]: c["status"] for c in v["clauses"]}["c"] == "PASS"


def test_a_missing_per_seed_file_leaves_a_null_for_that_seed_only(tmp_path):
    rd = tmp_path / "read"
    _write_reads(rd)
    d = _all_s6c(tmp_path / "s6c")
    (d / INP.s6c_name("K2", 1)).unlink()
    inp = INP.build_inputs(rd, s6c_dir=d)
    assert inp["seeds"]["0"]["s6c_k2"] is not None and inp["seeds"]["1"]["s6c_k2"] is None and inp["not_built"] == ["seed1.s6c_k2"]
    v = V.build(inp)
    assert {c["id"]: c["status"] for c in v["clauses"]}["c"] == "NOT-COMPUTABLE" and v["verdict"] == "NO-GO"


def test_judge_reads_the_null_s6c_as_not_computable_even_when_everything_else_passes(tmp_path):
    rd = tmp_path / "read"
    _write_reads(rd)
    v = V.build(INP.build_inputs(rd))
    st = {c["id"]: c["status"] for c in v["clauses"]}
    assert st == {"a": "PASS", "b": "PASS", "c": "NOT-COMPUTABLE", "d": "PASS", "e": "PASS"}
    assert v["verdict"] == "NO-GO" and [f["id"] for f in v["failing_clauses"]] == ["c"]
    assert v["four_families"]["0"]["STRATEGIC"]["status"] == "PAUSED"


def test_RED_a_placeholder_s6c_would_read_as_evidence_so_the_writer_refuses_one(tmp_path):
    rd = tmp_path / "read"
    _write_reads(rd)
    d = tmp_path / "s6c"
    d.mkdir()
    for lab in ("K2", "K0"):
        for s in (0, 1):
            _w(d / INP.s6c_name(lab, s), {"rate": 0.0})                               # a placeholder carries no infer_seed / trace
    with pytest.raises(SystemExit, match="infer_seed"):
        INP.build_inputs(rd, s6c_dir=d)
    fake = {"seeds": {s: {"eval_k2": str(rd / "EVAL_K2.json"), "eval_k0": str(rd / "EVAL_K0.json"), "s6c_k2": str(d / INP.s6c_name("K2", int(s))),
                          "s6c_k0": str(d / INP.s6c_name("K0", int(s))), "families": str(rd / f"families_s{s}.json"), "camblind": str(rd / f"camblind_s{s}.json")} for s in ("0", "1")}}
    assert {c["id"]: c["status"] for c in V.build(fake)["clauses"]}["c"] != "NOT-COMPUTABLE"     # => the hazard the writer's refusal removes: a made-up number is read as a measurement


def test_RED_one_s6c_path_on_both_seeds_is_refused(tmp_path):
    """The old single-file ``--s6c-k2`` / ``--s6c-k0`` stamped ONE file on both seeds (H-ESTIM-SEED-1): the CLI refuses it, the library function no longer takes it."""
    rd = tmp_path / "read"
    _write_reads(rd)
    one = _s6c(tmp_path / "one", "K2", 0, 0.01)
    for flag in ("--s6c-k2", "--s6c-k0"):
        with pytest.raises(SystemExit, match="ONE file for both inference seeds"):
            INP.main(["--rd", str(rd), "--out", str(tmp_path / "I.json"), flag, one])
    assert not (tmp_path / "I.json").exists()
    with pytest.raises(TypeError):
        INP.build_inputs(rd, s6c_k2=one)


def test_RED_a_seed_swap_or_an_arm_swap_in_the_s6c_dir_is_refused(tmp_path):
    rd = tmp_path / "read"
    _write_reads(rd)
    d = _all_s6c(tmp_path / "s6c")
    _s6c(d, "K2", 1, 0.04, infer_seed=0)                                              # seed 0's score filed in the seed 1 slot
    with pytest.raises(SystemExit, match="seed 1 slot"):
        INP.build_inputs(rd, s6c_dir=d)
    d = _all_s6c(tmp_path / "s6c2")
    _s6c(d, "K0", 0, 0.12, trace_label="K2")                                          # K2's trace scored into K0's slot
    with pytest.raises(SystemExit, match="E2_K0_mapsel"):
        INP.build_inputs(rd, s6c_dir=d)
    with pytest.raises(SystemExit, match="not a directory"):
        INP.build_inputs(rd, s6c_dir=tmp_path / "nope")


def test_RED_inputs_writer_refuses_missing_files_and_wrong_seeds(tmp_path):
    rd = tmp_path / "read"
    _write_reads(rd)
    with pytest.raises(SystemExit, match="seeds 0 AND 1"):
        INP.build_inputs(rd, seeds=["0"])
    (rd / "camblind_s1.json").unlink()
    with pytest.raises(SystemExit, match="camblind_s1.json"):
        INP.build_inputs(rd)
    (rd / "EVAL_K0.json").unlink()
    with pytest.raises(SystemExit, match="EVAL_K0.json"):
        INP.build_inputs(rd)


# ============================================================ the MAPSEL trace count check (r9e2_s6c_tools.py)
def _trace(p, n, final=True, meta_n=None):
    import numpy as np
    np.savez(p, ws=np.arange(n), meta_json=np.array(json.dumps({"final": final, "n_windows": n if meta_n is None else meta_n})))
    return str(p)


def test_trace_final_check_accepts_exactly_the_dev_windows_and_refuses_everything_else(tmp_path):
    import r9e2_s6c_tools as TL
    split = _w(tmp_path / "split.json", {"DEV": {"n_windows_s2": 7}})
    assert TL.check_final(_trace(tmp_path / "ok.npz", 7), split) == (True, "7")
    for name, kw, n in (("partial", dict(final=False), 7), ("with_hold", {}, 12), ("meta_lies", dict(meta_n=12), 7), ("short", {}, 5)):
        ok, why = TL.check_final(_trace(tmp_path / f"{name}.npz", n, **kw), split)                  # RED arms: HOLD windows in the trace, a partial record, a lying meta
        assert ok is False and why, name
    assert TL.check_final(tmp_path / "missing.npz", split)[0] is False
    assert TL.main(["final", str(tmp_path / "with_hold.npz"), split]) == 1 and TL.main(["final", str(tmp_path / "ok.npz"), split]) == 0


# ============================================================ the Thor shell driver, by a DRY RUN
def _sh_pins(text):
    return re.search(r'^PINS="([^"]+)"', text, re.M).group(1).split()


def _dry(script, tmp_path, drop=(), extra=None, env_extra=None):
    out = tmp_path / "out"
    env = {k: f"/thor/{k.lower()}" for k in _sh_pins(SH.read_text(encoding="utf-8"))}
    env.update(R9E2_OUT=out.as_posix(), R9E2_DRY="1", R9E2_PY="/thor/py", R9E2_TREE="/thor/tree")
    for k in drop:
        env.pop(k)
    env.update(extra or {})
    e = dict(os.environ, MSYS_NO_PATHCONV="1", PYTHONDONTWRITEBYTECODE="1")
    for k in list(e):
        if k.startswith(("R9E2_", "L1_", "PF_", "R8_")):
            e.pop(k)
    e.update(env)
    e.update(env_extra or {})
    r = subprocess.run([BASH, Path(script).as_posix()], capture_output=True, text=True, encoding="utf-8", env=e, timeout=120, stdin=subprocess.DEVNULL)
    return r.returncode, r.stdout.replace("\r", ""), r.stderr.replace("\r", ""), out


def _structure(out: str, k2_banded: bool = False, k0_banded: bool = False) -> list:
    """Every structural claim about the dry run -> the list of FAILED claims (empty = all hold)."""
    bad = []
    out = out.replace("\\,", ",")                                  # printf %q escapes the comma of `--infer-seeds 0,1`
    lines = out.splitlines()
    dry = [ln for ln in lines if ln.startswith("[dry] ")]

    def cmds(name):
        return [ln for ln in dry if name in ln and not ln.startswith("[dry] require")]
    marks = [m.group(1) for ln in lines for m in [re.match(r"ZZR9E2ROLL-([A-Z0-9-]+)ZZ", ln)] if m]
    want = ["START", "INPUTS-OK", "LOCK-WAIT", "LOCK-TAKEN", "GPU-CHECK", "ROLL-START", "ROLL-DONE", "ROLL-START", "ROLL-DONE",
            "CAMBLIND-START", "CAMBLIND-DONE", "CAMBLIND-START", "CAMBLIND-DONE", "DUMP-START", "DUMP-DONE", "DUMP-START", "DUMP-DONE",
            "MAPSEL-START", "MAPSEL-DONE", "MAPSEL-START", "MAPSEL-DONE", "MAPSEL-START", "MAPSEL-DONE", "MAPSEL-START", "MAPSEL-DONE",
            "LOCK-RELEASED", "FAMILIES-DONE", "DISTKEEP-DONE", "CAMBLIND-READ-DONE", "FAMILIES-DONE", "DISTKEEP-DONE", "CAMBLIND-READ-DONE",
            "EVAL-DONE", "EVAL-DONE", "S6C-READY", "INPUTS-WRITTEN", "DRY-DONE"]
    if marks != want:
        bad.append(f"marker-order {marks}")
    rolls = cmds("run_battery_r9_e2.py")
    if len(rolls) != 2 or [("E2_K0_pf" in r, "E2_K2_pf" in r) for r in rolls] != [(True, False), (False, True)]:
        bad.append("rolls-K0-then-K2")
    for flag in ("--no-baselines", "--lock-held", "--no-g0", "--infer-seeds 0,1", "--out-root", "model_ema.pt", "config.json"):
        if not all(flag in r for r in rolls):
            bad.append(f"battery{flag}")
    cbs = cmds("camblind_roll_r9_e2.py")
    if len(cbs) != 2:
        bad.append("camblind-count")
    for flag in ("--require-seal", "--os-only", "--camera derange", "--derange-seed 0", "--camblind-script", "--ztac-out", "--dump-dir", "--out-json"):
        if not all(flag in c for c in cbs):
            bad.append(f"camblind{flag}")
    for s, c in zip((0, 1), cbs):
        if f"--seed {s}" not in c or f"E2_K2_pf/windows_s{s}.json" not in c or f"E2_K2_camblind/dump_s{s}" not in c:
            bad.append(f"camblind-seed{s}")
    dumps = cmds("r9_e1_dump.py")
    if len(dumps) != 2 or "dump_K2" not in dumps[0] or "dump_K0" not in dumps[1]:
        bad.append("dumps-K2-then-K0")
    if not all("--require-ema" in d and "--split dev" in d for d in dumps):
        bad.append("dump-flags")
    fams = cmds("r9_e2_families.py")
    if [("--mode families" in f, "--mode camblind" in f) for f in fams] != [(True, False), (False, True)] * 2:
        bad.append("families-modes")
    for f in fams:
        if "--dev-split" not in f or "--a-dump" not in f or "--b-dump" not in f:
            bad.append("families-flags")
    dks = cmds("r9_e2_distkeep.py")                                # LONGITUDINAL distance-keeping, merged into THAT seed's families.json
    if len(dks) != 2 or not all("--merge-into" in d and f"families_s{s}.json" in d.split("--merge-into", 1)[1] and "--lead-block" in d
                                 and f"E2_K2_pf/dump_s{s}" in d and f"E2_K0_pf/dump_s{s}" in d and "--dev-split" in d for s, d in zip((0, 1), dks)):
        bad.append("distkeep-merge")
    evs = cmds("r9_e1_eval.py")
    if len(evs) != 2 or not all("--lift-valid" in e and "--split dev" in e for e in evs):
        bad.append("eval-lift-valid-dev")
    inputs = cmds("r9e2_inputs.py")
    if len(inputs) != 1 or "--s6c-k2" in inputs[0] or "--s6c-k0" in inputs[0] or "--s6c-dir" not in inputs[0]:
        bad.append("inputs-s6c-per-seed-dir")
    ms = cmds("mapsel_roll_r9_e2.py")                              # the S6c roll: K2 then K0, seeds 0 then 1, through the DEV wrapper, windows = THAT arm's DEV windows of THAT seed
    want_ms = [(a, s) for a in ("K2", "K0") for s in (0, 1)]
    def _ms_ok(a, s, m):
        return all(x in m for x in (f"E2_{a}_mapsel/trace_s{s}.npz", f"E2_{a}_mapsel/rast_s{s}", f"--infer-seed {s}", f"/E2_{a}_pf/windows_s{s}.json", "--require-seal", "--os-only",
                                    "--map-gt-root", "--device cuda"))
    if len(ms) != 4 or not all(_ms_ok(a, s, m) for (a, s), m in zip(want_ms, ms)):
        bad.append("mapsel-per-arm-per-seed")
    for (a, s), m in zip(want_ms, ms):
        mode = "skip" if (a == "K2" and k2_banded) or (a == "K0" and k0_banded) else "require"
        if f"--own-planes {mode}" not in m:
            bad.append(f"mapsel-own-planes-{a}-{mode}")
    if any("/mapsel_roll.py" in ln and "mapsel_roll_r9_e2" not in ln for ln in dry if not ln.startswith("[dry] require")):
        bad.append("mapsel-must-go-through-the-dev-wrapper")
    if len(cmds("r9_e2_verdict.py")) != 1:
        bad.append("verdict")
    iflock = next((i for i, ln in enumerate(dry) if "flock -w" in ln), None)
    ifrel = next((i for i, ln in enumerate(dry) if "release the GPU lock" in ln), None)
    gpu = [i for i, ln in enumerate(dry) if any(n in ln for n in ("run_battery_r9_e2.py", "camblind_roll_r9_e2.py", "r9_e1_dump.py", "mapsel_roll_r9_e2.py")) and not ln.startswith("[dry] require")]
    s6req = [i for i, ln in enumerate(dry) if ln.startswith("[dry] require nonempty") and "/S6C_K" in ln]
    inp_i = [i for i, ln in enumerate(dry) if "r9e2_inputs.py" in ln and not ln.startswith("[dry] require")]
    if len(s6req) != 4 or not inp_i or max(s6req) > min(inp_i) or sorted(re.findall(r"S6C_K[02]_s[01]", " ".join(dry[i] for i in s6req))) != ["S6C_K0_s0", "S6C_K0_s1", "S6C_K2_s0", "S6C_K2_s1"]:
        bad.append("s6c-gate-before-inputs")
    cpu = [i for i, ln in enumerate(dry) if any(n in ln for n in ("r9_e2_families.py", "r9_e2_distkeep.py", "r9_e1_eval.py", "r9e2_inputs.py", "r9_e2_verdict.py")) and not ln.startswith("[dry] require")]
    if sum("flock -w" in ln for ln in dry) != 1:
        bad.append("one-lock-hold-around-every-gpu-step (a second lock hold)")
    if iflock is None or ifrel is None or not gpu or not (iflock < min(gpu) and max(gpu) < ifrel < min(cpu)):
        bad.append("one-lock-hold-around-every-gpu-step")
    return bad


def _mutated(tmp_path, old, new):
    text = SH.read_text(encoding="utf-8")
    assert old in text, f"mutation anchor {old!r} not in r9e2_roll.sh"
    d = tmp_path / "mut"
    d.mkdir(exist_ok=True)
    p = d / "r9e2_roll.sh"
    p.write_bytes(text.replace(old, new, 1).encode("utf-8"))
    return p


@needs_bash
def test_dry_run_prints_every_command_in_order_and_touches_nothing(tmp_path):
    before = sorted(p.name for p in tmp_path.iterdir())
    rc, out, err, outdir = _dry(SH, tmp_path)
    assert rc == 0, (out, err)
    assert _structure(out) == [], out
    assert sorted(p.name for p in tmp_path.iterdir()) == before and not outdir.exists()
    assert "[dry] R9E2_DRY=1" in out and "ZZR9E2ROLL-DRY-DONEZZ" in out


@needs_bash
def test_dry_run_refuses_the_old_single_file_s6c_switch_and_gates_on_the_four_per_seed_files(tmp_path):
    rc, out, err, outdir = _dry(SH, tmp_path, extra={"R9E2_S6C_K0": "/thor/s6c_k0.json"})
    assert rc == 4 and "R9E2_S6C_K0" in err and "ZZR9E2ROLL" not in out and not outdir.exists()
    rc, out, _, od = _dry(SH, tmp_path)
    assert rc == 0 and "s6c_k2 null" not in out and f"--s6c-dir {od.as_posix()}/read/s6c" in out
    assert [ln for ln in out.splitlines() if "[dry] require nonempty" in ln and "/S6C_K" in ln].__len__() == 4


def _cfg(tmp_path, name, argv):
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"arm": "hier", "seed": 0, "argv": argv}, indent=1), encoding="utf-8")
    return d.as_posix()


@needs_bash
def test_own_planes_is_derived_from_the_runs_config_not_the_arm_name(tmp_path):
    """`--r9-map-banded` in the config argv (K2M1N20 / K2M3N20) => skip; absent, or only `--r9-map-banded-near-m` => require; K0 is judged by its own config, not K2's."""
    banded = _cfg(tmp_path, "banded", ["--arm", "hier", "--r9-map-banded", "--r9-map-banded-near-m", "10"])
    plain = _cfg(tmp_path, "plain", ["--arm", "hier", "--size", "small"])
    near_only = _cfg(tmp_path, "near", ["--arm", "hier", "--r9-map-banded-near-m", "10"])
    for k2, k0, k2_banded, k0_banded in ((banded, plain, True, False), (plain, plain, False, False), (near_only, plain, False, False), (plain, banded, False, True)):
        rc, out, err = None, None, None
        rc, out, err, _ = _dry(SH, tmp_path, extra={"R9E2_K2_RUN": k2, "R9E2_K0_RUN": k0})
        assert rc == 0, err
        assert _structure(out, k2_banded=k2_banded, k0_banded=k0_banded) == [], (k2, k0, _structure(out, k2_banded=k2_banded, k0_banded=k0_banded))


def _mutate_text(tmp_path, fn):
    text = SH.read_text(encoding="utf-8")
    new = fn(text)
    assert new != text, "the mutation changed nothing"
    d = tmp_path / "mut"
    d.mkdir(exist_ok=True)
    p = d / "r9e2_roll.sh"
    p.write_bytes(new.encode("utf-8"))
    return p


def _banded_dirs(tmp_path):
    return {"R9E2_K2_RUN": _cfg(tmp_path, "k2b", ["--arm", "hier", "--r9-map-banded"]), "R9E2_K0_RUN": _cfg(tmp_path, "k0p", ["--arm", "hier"])}


@needs_bash
def test_the_unmutated_driver_passes_the_banded_structure_check(tmp_path):
    rc, out, err, _ = _dry(SH, tmp_path, extra=_banded_dirs(tmp_path))
    assert rc == 0, err
    assert _structure(out, k2_banded=True) == []


@needs_bash
@pytest.mark.parametrize("name,fn,claim", [
    # banded K2 given `require`: the mode no longer follows the config
    ("banded-given-require", lambda t: t.replace("then echo skip; else echo require; fi", "then echo require; else echo require; fi"), "mapsel-own-planes-K2-skip"),
    # a HOLD-capable path: the roll goes through the bare FlyWheels roll, bypassing the DEV wrapper
    ("bare-roll-no-dev-wrapper", lambda t: t.replace('"$OPS/mapsel_roll_r9_e2.py"', '"$MS_CODE/mapsel_roll.py"'), "mapsel-per-arm-per-seed"),
    # a HOLD-capable path: the roll takes a window list that is not that arm's DEV windows of that seed
    ("non-dev-windows-list", lambda t: t.replace('--windows-json "$OUT/E2_${A}_pf/windows_s$S.json" --device cuda \\\n        --out-json "$MD/roll_s$S.json"', '--windows-json "$OUT/windows_all_s$S.json" --device cuda \\\n        --out-json "$MD/roll_s$S.json"'), "mapsel-per-arm-per-seed"),
    # per-seed swap: seed 1 rolled with seed 0's inference seed
    ("seed-swap", lambda t: t.replace('--infer-seed "$S" --own-planes', '--infer-seed 0 --own-planes'), "mapsel-per-arm-per-seed"),
    # one trace path for both seeds
    ("one-path-both-seeds", lambda t: t.replace('--trace-out "$MD/trace_s$S.npz"', '--trace-out "$MD/trace_s0.npz"'), "mapsel-per-arm-per-seed"),
    # K0 rolled with K2's trace dir (arm swap)
    ("arm-swap", lambda t: t.replace('MD="$OUT/E2_$1_mapsel"', 'MD="$OUT/E2_K2_mapsel"'), "mapsel-per-arm-per-seed"),
    # the S6c roll outside the lock hold: moved after the lock release
    ("roll-outside-lock", lambda t: t.replace('mapsel_arm K2 "$RUN2"\nmapsel_arm K0 "$RUN0"\n', '', 1).replace('ev LOCK-RELEASED', 'mapsel_arm K2 "$RUN2"\nmapsel_arm K0 "$RUN0"\nev LOCK-RELEASED', 1), "one-lock-hold-around-every-gpu-step"),
    # the S6c roll under a second lock hold
    ("second-lock", lambda t: t.replace('mapsel_arm K2 "$RUN2"\nmapsel_arm K0 "$RUN0"\n', 'echo "[dry] flock -w 1 $LOCKF (take_lock)"\nmapsel_arm K2 "$RUN2"\nmapsel_arm K0 "$RUN0"\n', 1), "one-lock-hold-around-every-gpu-step"),
    # the gate removed: INPUTS no longer waits for the four S6c files
    ("gate-removed", lambda t: t.replace('for f in $S6CF; do echo "[dry] require nonempty $f', 'for f in; do echo "[dry] require nonempty $f', 1), "s6c-gate-before-inputs"),
])
def test_RED_each_s6c_wire_mutation_makes_the_structure_check_fail(tmp_path, name, fn, claim):
    rc, out, err, _ = _dry(_mutate_text(tmp_path, fn), tmp_path, extra=_banded_dirs(tmp_path))
    assert rc == 0, err
    bad = _structure(out, k2_banded=True)
    assert any(claim in b for b in bad), (name, claim, bad)


def test_the_real_gate_stops_with_exit_5_before_inputs_and_the_verdict():
    """Text-level pin (a real Thor run is not reproducible here): the AWAIT branch exits 5 BEFORE the INPUTS writer / the judge are reached and writes neither."""
    s = SH.read_text(encoding="utf-8")
    a, i, j = s.index("ev AWAIT-S6C"), s.index('--k0-label K0 --s6c-dir'), s.rindex('"$T/stack/scripts/r9_e2_verdict.py"')
    assert a < i < j and re.search(r"ev AWAIT-S6C[^\n]*\n\s*exit 5", s) and "AWAIT_S6C" in s
    assert "exit 5" not in s[i:], "exit 5 must precede INPUTS / the verdict"


# ---- the dev-box resume script ---------------------------------------------------------------------------------------------------------------------
RESUME = R9DIR / "r9e2_s6c_resume.sh"


def _resume_dry(script, tmp_path, drop=()):
    env = dict(R9E2_RD="/pull/read", R9E2_MS_ROOT="/pull/out", R9E2_SPLIT="/pull/split.json", R9E2_MAPSEL_CODE="/pull/mapsel/code", R9E2_OPS="/pull/ops",
               R9E2_VERDICT="/pull/r9_e2_verdict.py", R9E2_PY="/pull/py", R9E2_DRY="1", MSYS_NO_PATHCONV="1", PYTHONDONTWRITEBYTECODE="1")
    for k in drop:
        env.pop(k)
    e = {k: v for k, v in os.environ.items() if not k.startswith("R9E2_")}
    e.update(env)
    r = subprocess.run([BASH, Path(script).as_posix()], capture_output=True, text=True, encoding="utf-8", env=e, timeout=120, stdin=subprocess.DEVNULL)
    return r.returncode, r.stdout.replace("\r", ""), r.stderr.replace("\r", "")


def _resume_structure(out):
    bad = []
    dry = [ln for ln in out.splitlines() if ln.startswith("[dry] ")]
    sc = [ln for ln in dry if "s6c_dev.py" in ln]
    for (a, s), ln in zip([(a, s) for a in ("K2", "K0") for s in (0, 1)], sc):
        if not all(x in ln for x in (f"--seed {s}", f"E2_{a}_mapsel/trace_s{s}.npz", f"E2_{a}_mapsel/rast_s{s}", f"--flat-out /pull/read/s6c/S6C_{a}_s{s}.json", "--split /pull/split.json", "--mapsel-code")):
            bad.append(f"s6c-{a}-s{s}")
        if "--allow-partial" in ln or "--controls-only" in ln:
            bad.append("dry-run-switch")
    if len(sc) != 4:
        bad.append("four-scorings")
    inp = [i for i, ln in enumerate(dry) if "r9e2_inputs.py" in ln]
    ver = [i for i, ln in enumerate(dry) if "r9_e2_verdict.py" in ln]
    last_sc = max((i for i, ln in enumerate(dry) if "s6c_dev.py" in ln), default=-1)
    if len(inp) != 1 or len(ver) != 1 or not (last_sc < inp[0] < ver[0]) or "--s6c-dir /pull/read/s6c" not in dry[inp[0]]:
        bad.append("inputs-then-verdict-after-all-scorings")
    return bad


@needs_bash
def test_resume_script_scores_four_traces_then_inputs_then_the_verdict(tmp_path):
    rc, out, err = _resume_dry(RESUME, tmp_path)
    assert rc == 0, err
    assert _resume_structure(out) == [], out
    assert "ZZR9E2S6CZZ-DRY-DONE" in out


@needs_bash
@pytest.mark.parametrize("pin", ["R9E2_RD", "R9E2_MS_ROOT", "R9E2_SPLIT", "R9E2_MAPSEL_CODE", "R9E2_OPS", "R9E2_VERDICT"])
def test_resume_script_refuses_each_unset_pin(tmp_path, pin):
    rc, out, err = _resume_dry(RESUME, tmp_path, drop=(pin,))
    assert rc == 4 and pin in err


@needs_bash
@pytest.mark.parametrize("old,new,claim", [
    ('--flat-out "$F"', '--flat-out "$S6D/S6C_${A}_s0.json"', "s6c-K2-s1"),                      # one output path on both seeds
    ('--seed "$S" \\\n', '--seed 0 \\\n', "s6c-K2-s1"),                                           # seed 1 scored as seed 0
    ('T="$MS/E2_${A}_mapsel/trace_s$S.npz"', 'T="$MS/E2_K2_mapsel/trace_s$S.npz"', "s6c-K0-s0"),    # K0 scored from K2's trace
    ('--s6c-dir "$S6D"', '--s6c-k2 /x', "inputs-then-verdict-after-all-scorings"),                # the old single-file switch
    ('for A in K2 K0; do', 'for A in K2; do', "four-scorings"),                                   # a missing arm
])
def test_RED_each_mutation_of_the_resume_script_makes_its_structure_check_fail(tmp_path, old, new, claim):
    text = RESUME.read_text(encoding="utf-8")
    assert old in text, old
    d = tmp_path / "mutr"
    d.mkdir()
    p = d / "r9e2_s6c_resume.sh"
    p.write_bytes(text.replace(old, new, 1).encode("utf-8"))
    rc, out, err = _resume_dry(p, tmp_path)
    assert rc == 0, err
    assert any(claim in b for b in _resume_structure(out)), (claim, _resume_structure(out))


@needs_bash
def test_dry_run_battery_parent_runs_in_the_baseline_cwd_when_one_is_named(tmp_path):
    """r8_battery_patch.apply -> fix_baselines needs the refcv5cmp baseline dumps at RELATIVE paths on Thor: the roll step honours R9E2_BASE_CWD."""
    for extra, want in (({"R9E2_BASE_CWD": "/thor/basecwd"}, "[dry] cd /thor/basecwd"), ({}, "[dry] cd .")):
        rc, out, err, _ = _dry(SH, tmp_path, extra=extra)
        assert rc == 0, err
        dry = [ln for ln in out.splitlines() if ln.startswith("[dry] ")]
        idx = [i for i, ln in enumerate(dry) if "run_battery_r9_e2.py" in ln and not ln.startswith("[dry] require")]
        assert len(idx) == 2 and all(dry[i - 1] == want for i in idx), (want, [dry[i - 1] for i in idx])


@needs_bash
@pytest.mark.parametrize("pin", ["R9E2_K0_RUN", "R9E2_K2_RUN", "R9E2_TREE", "R9E2_PF_CODE", "PF_R8_CODE", "PF_NF_HOLD", "R8_EVAL_KIT", "R9E2_REMAP", "R9E2_SPLIT", "R9E2_S2",
                                 "R9E2_LIFT_VALID", "R9E2_OUT", "L1_PF_CODE", "L1_R8_CODE", "R9E2_LEAD_BLOCK", "R9E2_MAP_GT_ROOT"])
def test_shell_refuses_each_unset_pin(tmp_path, pin):
    assert pin in _sh_pins(SH.read_text(encoding="utf-8"))
    rc, out, err, outdir = _dry(SH, tmp_path, drop=(pin,))
    assert rc == 4 and pin in err and "ZZR9E2ROLL" not in out and not outdir.exists()


@needs_bash
def test_shell_refuses_the_standalone_pose_clock(tmp_path):
    rc, out, err, _ = _dry(SH, tmp_path, env_extra={"R8_NO_POSE_SYNC": "1"})
    assert rc == 4 and "R8_NO_POSE_SYNC" in err


@needs_bash
@pytest.mark.parametrize("old,new,claim", [
    ("--no-g0 --no-baselines \\\n      --tag", "--no-g0 \\\n      --tag", "battery--no-baselines"),
    ("--tag \"$TAG\" --infer-seeds 0,1 --lock-held", "--tag \"$TAG\" --infer-seeds 0,1", "battery--lock-held"),
    ("--camera derange --derange-seed 0 \\\n      --ztac-out \"$CBD/ztac_blind_s$S.npz\" --require-seal --", "--camera derange --derange-seed 0 \\\n      --ztac-out \"$CBD/ztac_blind_s$S.npz\" --", "camblind--require-seal"),
    ("dump_arm K0 \"$RUN0\"\n", "\n", "dumps-K2-then-K0"),
    ("--lift-valid \"$R9E2_LIFT_VALID\" --out", "--out", "eval-lift-valid-dev"),
    ("--k2-label K2 --k0-label K0 --s6c-dir", "--k2-label K2 --k0-label K0 --s6c-k2 /fake --s6c-dir", "inputs-s6c-per-seed-dir"),
    ("--k2-label K2 --k0-label K0 --s6c-dir \"$S6D\"", "--k2-label K2 --k0-label K0", "inputs-s6c-per-seed-dir"),
    ("roll_arm K2 \"$RUN2\"\n", "\n", "rolls-K0-then-K2"),
    ('--out "$RD/distkeep_s$S.json" --merge-into "$RD/families_s$S.json"', '--out "$RD/distkeep_s$S.json"', "distkeep-merge"),
    ('--lead-block "$R9E2_LEAD_BLOCK" ', '', "distkeep-merge"),
])
def test_RED_each_mutation_of_the_driver_makes_the_structure_check_fail(tmp_path, old, new, claim):
    rc, out, err, _ = _dry(_mutated(tmp_path, old, new), tmp_path)
    assert rc == 0, err
    bad = _structure(out)
    assert any(claim in b for b in bad), (claim, bad)


@needs_bash
def test_RED_without_the_pin_guard_an_unset_pin_is_not_refused(tmp_path):
    p = _mutated(tmp_path, 'if [ -n "$MISS" ]; then', 'if [ -n "" ]; then')
    rc, out, err, _ = _dry(p, tmp_path, drop=("R9E2_K0_RUN",))
    assert rc != 4                                                                   # the guard is what produces the refusal


# ---- reachability from the actual caller: every flag the shell passes exists in the callee; every pin a callee needs reaches it ----------
def _flags_of(cmdline):
    return [t for t in cmdline.split() if t.startswith("--")]


@needs_bash
def test_every_flag_the_shell_passes_exists_in_the_callee(tmp_path):
    rc, out, err, _ = _dry(SH, tmp_path)
    assert rc == 0, err
    dry = [ln for ln in out.splitlines() if ln.startswith("[dry] ") and not ln.startswith("[dry] require")]
    callees = {"r9_e1_dump.py": ROOT / "scripts" / "r9_e1_dump.py", "r9_e1_eval.py": ROOT / "scripts" / "r9_e1_eval.py",
               "r9_e2_families.py": ROOT / "scripts" / "r9_e2_families.py", "r9_e2_verdict.py": ROOT / "scripts" / "r9_e2_verdict.py",
               "r9_e2_distkeep.py": ROOT / "scripts" / "r9_e2_distkeep.py",
               "r9e2_inputs.py": R9DIR / "r9e2_inputs.py"}
    cbr9 = ROOT.parent / "FlyWheels" / "TanitAD_EvalFlyWheel" / "incoming" / "2026-11-03-refcv9-camblind" / "code" / "camblind_roll_r9.py"
    if cbr9.is_file():
        callees["camblind_roll_r9_e2.py"] = cbr9
    seen = set()
    for name, src in callees.items():
        text = src.read_text(encoding="utf-8")
        for ln in (x for x in dry if name in x):
            args = ln.split(name, 1)[1]
            if name == "camblind_roll_r9_e2.py":
                args = args.split(" -- ", 1)[0].replace("--camblind-script", "")          # the launcher's own switch is not camblind_roll_r9's
            for f in _flags_of(args):
                seen.add((name, f))
                assert f'"{f}"' in text or f"'{f}'" in text, f"{name} has no {f}"
    assert ("r9_e1_eval.py", "--lift-valid") in seen and ("r9_e1_dump.py", "--require-ema") in seen and ("r9_e2_verdict.py", "--inputs") in seen
    assert ("r9_e2_distkeep.py", "--merge-into") in seen and ("r9_e2_distkeep.py", "--lead-block") in seen


def test_the_pins_the_python_entry_points_need_reach_them_from_the_shell():
    sh = SH.read_text(encoding="utf-8")
    pins = set(_sh_pins(sh))
    for k in R9.ENV_PINS:                                                              # run_battery_r9_e2 / roll_seed_r9_e2
        assert k in pins or f'export {k}="' in sh, f"{k} reaches the staged battery from nowhere"
    for k in ("R8_LAUNCH_TREE", "PF_R8_CODE", "R8_EVAL_KIT", "L1_PF_CODE", "L1_R8_CODE"):    # camblind_roll_r9_e2 + camblind_roll_r9.py
        assert k in pins or f'export {k}="' in sh
    assert "env -u HF_TOKEN" in sh and "-u REFCV6_KIT" in sh                             # the roll env: r8_repoint refuses a conflicting REFCV6_*


def test_the_shells_copied_pins_equal_the_arm_read_scripts():
    a, s = ARM_READ.read_text(encoding="utf-8"), SH.read_text(encoding="utf-8")
    for name in ("SPLIT_SHA_PIN", "GATES_SHA"):
        ma, ms = re.search(name + r'="([0-9a-f]{64})"', a), re.search(name + r'="([0-9a-f]{64})"', s)
        assert ma and ms and ma.group(1) == ms.group(1), name
    for lit in ("/home/nvidia/refcv7_post/thor_gpu.lock", "flock -w \"$1\" 9", "refcv8_det_presence_gates_train.json", "--query-compute-apps=pid"):
        assert lit in a and lit in s, lit


def test_the_driver_never_takes_a_dev_box_default_for_a_path():
    s = SH.read_text(encoding="utf-8")
    body = "\n".join(ln for ln in s.splitlines() if not ln.lstrip().startswith("#"))
    assert not re.search(r"\b[CDEFG]:/(?!Users/Admin/refcv8_pf/refcv9_l1)", body), "a dev-box drive path in the driver"
