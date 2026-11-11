"""PREREG_V8F_SLOWWM: the world-model lever ``--wm-basis {adapter, slow, pca}`` (z1 reads a cached 256-d whitened closed-form basis of the frozen features' token mean instead of the adapter token mean).

Controls of the card (each has a deliberate-regression mutation in ``theory/slowwm/controls_slowwm.py`` that must make the named tests go RED):
1. Off identity: default ``adapter``; the model, its state dict and the loss are the earlier ones (``test_off_*``).
2. Reachability: argparse -> namespace -> config -> model -> z1 -> loss, a non-zero gradient on ``wm.proj`` and an exact zero on the frozen adapter, a cache of the wrong kind refused (``test_reach_*``, ``test_refuses_*``).
4. Echo guard (structural half): the dataset's basis item is a function of the cache file alone and carries no label key (``test_basis_item_*``).
The G-DVB read-back of the BUILT lever is pinned by ``test_gate_*``.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.data import v8f_wm_windows as WW  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
import v8f_mini_train as TR  # noqa: E402

torch.set_num_threads(1)
BASE = ["--synthetic", "--batch", "8", "--d", "256", "--n-q", "2", "--kin-mode", "yaw", "--seed", "5", "--arm", "M2b", "--threads", "1", "--warmup", "2"]
LADDER = ["--hmax", "25", "--wm-only", "--wm-strides", "2", "3", "--wm-rollout-k", "4", "3", "2", "--wm-rollout-stride", "2", "2", "1", "--wm-rollout-w", "1.0", "--wm-ctx", "3", "3", "3", "--wm-windows", "wmonly"]
F = 25


# ------------------------------------------------------------------------------------------------ a planted basis cache for the synthetic world
def write_basis_file(path, kind, seed=0):
    g = np.random.default_rng(seed)
    arr = {"kind": np.array(kind), "mean": g.standard_normal(1024).astype(np.float32), "proj": g.standard_normal((1024, 256)).astype(np.float32) / 32}
    h = hashlib.sha256()
    for k in sorted(arr):
        t = np.ascontiguousarray(arr[k])
        h.update(k.encode() + str(tuple(t.shape)).encode() + str(t.dtype).encode() + t.tobytes())
    np.savez(path, spectrum=np.zeros(256), **arr)
    return h.hexdigest()


def write_cache(root, names, T, kind="slow", seed=0, ok=None, dim=256, sha=None, manifest=True, basis_file=True):
    """Per-clip files in the zbasis format + BASIS_MANIFEST.json (+ basis.npz). Returns the basis sha256."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    bsha = write_basis_file(root / "basis.npz", kind, seed) if basis_file else hashlib.sha256(b"planted").hexdigest()
    g = np.random.default_rng(seed + 1)
    for c in names:
        n = T[c]
        np.savez(root / f"{c}.npz", zbasis=g.standard_normal((n, dim)).astype(np.float32), ok=np.ones(n, bool) if ok is None else ok, t_s=np.arange(n, dtype=np.float32) * 0.2, basis_sha=np.array(sha or bsha))
    if manifest:
        man = {"format": "zbasis-v1", "kind": kind, "dim": 256, "dtype": "float32", "key": "zbasis", "basis_sha256": bsha}
        if basis_file:
            man["basis_file"] = "basis.npz"
        (root / WW.BASIS_MANIFEST).write_text(json.dumps(man), encoding="utf-8")
    return bsha


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("synb")
    w = D.make_synthetic_world(root, n_train=6, n_eval=2, T=60, seed=5)
    J = D.load_join(Path(root) / "join_train.npz")
    store = D.ClipStore(root, "train", lru=8)
    names = w["names"]["train"]
    return {"root": root, "J": J, "store": store, "clips": names, "T": {c: len(store.get(c)["pose"]) for c in names}}


def cfg_of(extra=()):
    a = TR.build_parser().parse_args(["--out", "x", *BASE, *LADDER, *extra])
    return a, TR.make_config(a)


def model_of(extra=(), seed=0):
    a, cfg = cfg_of(extra)
    torch.manual_seed(seed)
    return a, cfg, M.V8FMini(cfg, M.P10Prior.synthetic(0), None, None)


# ------------------------------------------------------------------------------------------------ control 1: off identity
def test_off_default_is_adapter_everywhere():
    assert TR.build_parser().get_default("wm_basis") == "adapter"
    assert M.V8FMiniConfig.__dataclass_fields__["wm_basis"].default == "adapter"
    assert M.WM_BASIS_KINDS == ("adapter", "slow", "pca") and M.WM_BASIS_DIM == 256
    a = TR.build_parser().parse_args(["--out", "x"])
    assert a.wm_basis == "adapter" and TR.basis_cache_problems(a) == []
    assert TR.FLAG_CHECKS["wm_basis"] == "lever:wm_basis"
    assert {act.dest for act in TR.build_parser()._actions if act.dest != "help"} == set(TR.FLAG_CHECKS)


def test_off_build_adds_no_parameter_no_key_and_the_loss_is_the_earlier_one():
    _, cfg0, m0 = model_of()
    _, cfg1, m1 = model_of(["--wm-basis", "slow"])
    assert cfg0.wm_basis == "adapter" and cfg1.wm_basis == "slow"
    s0, s1 = m0.state_dict(), m1.state_dict()
    assert list(s0) == list(s1) and all(torch.equal(s0[k], s1[k]) for k in s0)          # the lever adds no parameter and consumes no RNG
    assert "wm_basis" not in m0.built_levers() and m1.built_levers()["wm_basis"] == "slow"
    g = torch.Generator().manual_seed(3)
    zm, act = torch.randn(2, F, 256, generator=g), torch.randn(2, F, 2, generator=g)
    torch.manual_seed(9)
    t_new = m0.wm_only_loss({"zmean": zm, "act": act})["terms"]
    torch.manual_seed(9)
    ref = m0.wm.stack.loss(m0.wm._z1_post(zm), m0.wm.scale_actions(act)[:, :-1])           # the earlier spelling: proj of the token mean, then the stack
    assert float(t_new["wm"]) == float(ref["total"])
    for k, v in ref.items():
        if k != "total":
            assert float(t_new[f"wm/{k}"]) == float(v), k


def test_off_adapter_model_refuses_a_basis_batch_and_a_basis_model_refuses_the_adapter_paths():
    _, _, m0 = model_of()
    _, _, m1 = model_of(["--wm-basis", "slow"])
    u = torch.randn(2, F, 256)
    act = torch.randn(2, F, 2)
    with pytest.raises(ValueError):
        m0.wm.z1_from_basis(u)
    with pytest.raises(ValueError):
        m1.wm.z1_from_mean(u)
    with pytest.raises(ValueError):
        m1.wm.z1(u[:, :, None, :].expand(-1, -1, 40, -1))
    with pytest.raises(ValueError):
        m1.wm_only_loss({"zmean": u, "act": act})
    with pytest.raises(ValueError):
        m1.wm_only_loss({"pool": torch.randn(2, F, 1024), "act": act})
    assert m0.wm.built_basis()["basis_path_refused"] is True and m1.wm.built_basis()["mean_path_refused"] is True


def test_config_validation_refuses_a_lever_that_builds_nothing():
    with pytest.raises(ValueError):
        dataclasses.replace(cfg_of()[1], wm_basis="nope")
    with pytest.raises(ValueError):
        dataclasses.replace(cfg_of()[1], wm="off", wm_basis="slow", wm_rollout_k=(1, 1, 1), wm_ctx=(3, 3, 3), wm_k=1)
    a, cfg = cfg_of()
    with pytest.raises(ValueError):
        dataclasses.replace(cfg, d=32, wm_basis="pca")


# ------------------------------------------------------------------------------------------------ control 2: reachability
@pytest.mark.parametrize("kind", ["slow", "pca"])
def test_reach_argparse_to_namespace_to_config_to_model(kind):
    a, cfg, m = model_of(["--wm-basis", kind])
    assert a.wm_basis == kind and cfg.wm_basis == kind and m.cfg.wm_basis == kind and m.wm.wm_basis == kind
    bb = m.wm.built_basis()
    assert bb == {"mode": kind, "in_dim": 256, "mean_path_refused": True, "token_path_refused": True, "basis_path_refused": False, "basis_path_is_proj_tail": True}


@pytest.mark.parametrize("kind", ["slow", "pca"])
def test_reach_z1_to_loss_and_a_nonzero_gradient_on_proj_with_exact_zero_on_the_frozen_adapter(kind):
    _, _, m = model_of(["--wm-basis", kind])
    g = torch.Generator().manual_seed(4)
    u, act = torch.randn(2, F, 256, generator=g), torch.randn(2, F, 2, generator=g)
    z1 = m.wm.z1_from_basis(u)
    assert z1.shape[:2] == (2, F)
    torch.manual_seed(1)
    out = m.wm_only_loss({"zbasis": u, "act": act})
    out["terms"]["total"].backward()
    gp = sum(float(p.grad.abs().sum()) for n, p in m.named_parameters() if n.startswith("wm.proj.") and p.grad is not None)
    assert gp > 0.0, "no gradient reached wm.proj"
    gw = float(m.wm.proj[1].weight.grad.abs().sum())
    assert gw > 0.0
    ctrl = sum(float(p.grad.abs().sum()) for n, p in m.named_parameters() if p.grad is not None and not n.startswith("wm."))
    assert ctrl == 0.0                                                      # the control reads exactly zero: nothing outside the WM receives a gradient
    # the batch key decides the input: a different u gives a different loss (not u * 2: proj starts with a LayerNorm over the 256 coordinates, which removes a frame's overall scale)
    torch.manual_seed(1)
    out2 = m.wm_only_loss({"zbasis": u.flip(-1), "act": act})
    assert float(out2["terms"]["wm"]) != float(out["terms"]["wm"])


def test_reach_the_zbasis_key_is_what_the_loss_reads_not_zmean():
    _, _, m = model_of(["--wm-basis", "slow"])
    u, act = torch.randn(2, F, 256), torch.randn(2, F, 2)
    torch.manual_seed(1)
    a_ = float(m.wm_only_loss({"zbasis": u, "act": act})["terms"]["wm"])
    torch.manual_seed(1)
    b_ = float(m.wm_only_loss({"zbasis": u, "zmean": torch.zeros_like(u), "act": act})["terms"]["wm"])
    assert a_ == b_                                                         # zbasis has priority; a stray zmean cannot reroute a basis model


# ------------------------------------------------------------------------------------------------ the dataset: manifest checks
def test_dataset_serves_zbasis_and_collates_float32(world, tmp_path):
    J, store, names, T = world["J"], world["store"], world["clips"], world["T"]
    wins = WW.wm_windows(store, J, names, F=F)
    sha = write_cache(tmp_path / "c", names, T, "slow")
    ds = WW.V8FMiniWMWindows(store, J, wins, F=F, zcache=str(tmp_path / "c"), basis="slow")
    assert ds.basis == "slow" and ds.basis_manifest["basis_sha256"] == sha and ds.zcache_missing(names) == []
    it = ds.item(*wins[5])
    assert "zbasis" in it and "zmean" not in it and it["zbasis"].shape == (F, 256) and it["zbasis"].dtype == np.float32
    b = WW.collate_wm([ds.item(*wins[3]), ds.item(*wins[40])])
    assert b["zbasis"].shape == (2, F, 256) and b["zbasis"].dtype == torch.float32 and "zmean" not in b and "pool" not in b
    assert len(ds.ok_windows()) == len(wins)


def test_refuses_a_cache_of_the_wrong_kind_missing_manifest_wrong_dim_or_wrong_sha(world, tmp_path):
    J, store, names, T = world["J"], world["store"], world["clips"], world["T"]
    wins = WW.wm_windows(store, J, names, F=F)
    write_cache(tmp_path / "pca", names, T, "pca")
    with pytest.raises(ValueError, match="kind"):
        WW.V8FMiniWMWindows(store, J, wins, F=F, zcache=str(tmp_path / "pca"), basis="slow")
    write_cache(tmp_path / "nomf", names, T, "slow", manifest=False)
    with pytest.raises(ValueError, match="does not exist"):
        WW.V8FMiniWMWindows(store, J, wins, F=F, zcache=str(tmp_path / "nomf"), basis="slow")
    write_cache(tmp_path / "sha", names, T, "slow", sha="0" * 64)
    ds = WW.V8FMiniWMWindows(store, J, wins, F=F, zcache=str(tmp_path / "sha"), basis="slow")
    with pytest.raises(ValueError, match="built from basis"):
        ds.item(*wins[0])
    write_cache(tmp_path / "dim", names, T, "slow", dim=128)
    ds = WW.V8FMiniWMWindows(store, J, wins, F=F, zcache=str(tmp_path / "dim"), basis="slow")
    with pytest.raises(ValueError, match="shape"):
        ds.item(*wins[0])
    with pytest.raises(ValueError):
        WW.V8FMiniWMWindows(store, J, wins, F=F, basis="slow")              # no cache directory at all
    # an ADAPTER zmean cache under --wm-basis slow: no manifest, refused
    (tmp_path / "mean").mkdir()
    for c in names:
        np.savez(tmp_path / "mean" / f"{c}.npz", zmean=np.zeros((T[c], 256), np.float32), ok=np.ones(T[c], bool), t_s=np.zeros(T[c], np.float32))
    with pytest.raises(ValueError, match="does not exist"):
        WW.V8FMiniWMWindows(store, J, wins, F=F, zcache=str(tmp_path / "mean"), basis="slow")
    # a BASIS cache handed to the adapter dataset: a loud KeyError on zmean
    ds = WW.V8FMiniWMWindows(store, J, wins, F=F, zcache=str(tmp_path / "pca"), basis="adapter")
    with pytest.raises(KeyError):
        ds.item(*wins[0])


def test_basis_item_is_a_function_of_the_cache_file_alone_and_carries_no_label_key(world, tmp_path):
    J, store, names, T = world["J"], world["store"], world["clips"], world["T"]
    wins = WW.wm_windows(store, J, names, F=F)
    write_cache(tmp_path / "c", names, T, "slow")
    ds = WW.V8FMiniWMWindows(store, J, wins, F=F, zcache=str(tmp_path / "c"), basis="slow")
    it = ds.item(*wins[7])
    assert set(it) == {"zbasis", "act", "t", "ci"} and set(WW.WM_BATCH_KEYS_ZBASIS) == set(it)
    with np.load(tmp_path / "c" / f"{names[0]}.npz") as z:
        assert set(z.files) == {"zbasis", "ok", "t_s", "basis_sha"}


# ------------------------------------------------------------------------------------------------ the trainer: static refusals and the G-DVB read-back
def parse(extra):
    return TR.build_parser().parse_args(["--out", "x", *BASE, *LADDER, *extra])


def test_trainer_refuses_before_any_build(world, tmp_path):
    names, T = world["clips"], world["T"]
    write_cache(tmp_path / "pca", names, T, "pca")
    write_cache(tmp_path / "slow", names, T, "slow")
    assert TR.basis_cache_problems(parse(["--wm-basis", "slow", "--wm-zcache", str(tmp_path / "slow")])) == []
    bad = TR.basis_cache_problems(parse(["--wm-basis", "slow", "--wm-zcache", str(tmp_path / "pca")]))           # control 2's mutation: the wrong cache under slow
    assert bad and "kind" in bad[0]
    assert TR.basis_cache_problems(parse(["--wm-basis", "slow"]))                                                  # no cache directory
    a = parse(["--wm-basis", "slow", "--wm-zcache", str(tmp_path / "slow")])
    a.wm_windows = "train"
    assert TR.basis_cache_problems(a)                                                                              # declared without wmonly windows
    assert TR.basis_cache_problems(parse(["--wm-zcache", str(tmp_path / "slow")]))                                 # adapter + a basis cache
    with pytest.raises(SystemExit) as e:
        TR.main(["--out", str(tmp_path / "o"), *BASE, *LADDER, "--wm-basis", "slow", "--wm-zcache", str(tmp_path / "pca")])
    assert "ZZDVB-REFUSEDZZ" in str(e.value.code)
    # a basis.npz that does not hash to the manifest's sha
    write_basis_file(tmp_path / "slow" / "basis.npz", "slow", seed=99)
    assert any("hashes to" in p for p in TR.basis_cache_problems(parse(["--wm-basis", "slow", "--wm-zcache", str(tmp_path / "slow")])))


def test_the_sha_formula_of_the_trainer_is_the_documented_one(tmp_path):
    sha = write_basis_file(tmp_path / "b.npz", "pca", seed=3)
    assert TR.basis_sha256_of_file(str(tmp_path / "b.npz")) == sha


@pytest.fixture(scope="module")
def init(tmp_path_factory):
    root = tmp_path_factory.mktemp("slowwm")
    assert TR.main(["--out", str(root / "m2b"), *BASE, "--log-every", "1", "--ckpt-every", "100", "--steps", "2"]) == 0
    w = D.make_synthetic_world(root / "w", n_train=6, n_eval=2, T=60, seed=5)
    names = w["names"]["train"]
    st = D.ClipStore(root / "w", "train", lru=8)
    T = {c: len(st.get(c)["pose"]) for c in names}
    sha = {k: write_cache(root / f"zb_{k}", names, T, k, seed=i) for i, k in enumerate(("slow", "pca"))}
    return {"root": root, "ck": root / "m2b" / "ckpt_last.pt", "sha": sha}


def gate(init, name, kind, extra=(), cache=None, ck=True):
    argv = ["--out", str(init["root"] / name), *BASE, *(["--init-ckpt", str(init["ck"])] if ck else []), *LADDER, "--wm-basis", kind, "--wm-zcache", cache or str(init["root"] / f"zb_{kind}"), *extra, "--selfcheck-only"]
    try:
        return TR.main(argv)
    except SystemExit as e:
        return e.code


def sc_of(init, name):
    return json.loads((init["root"] / name / "selfcheck.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("kind", ["slow", "pca"])
def test_gate_passes_and_reads_the_built_lever_back(init, kind):
    assert gate(init, f"g_{kind}", kind) == 0
    rb = sc_of(init, f"g_{kind}")["levers"]["wm_basis_readback"]
    assert rb["declared"] == kind == rb["config"] == rb["dataset_basis"] == rb["kind"] == rb["built"]["mode"]
    assert rb["basis_sha256"] == init["sha"][kind] and rb["dim"] == 256 and rb["dataset_zbasis_shape"] == [25, 256] and rb["cache_problems"] == []
    assert rb["built"]["in_dim"] == 256 and rb["built"]["mean_path_refused"] is True and rb["built"]["basis_path_is_proj_tail"] is True


def test_gate_default_argv_records_no_basis_key(init):
    assert TR.main(["--out", str(init["root"] / "dflt"), *BASE, "--selfcheck-only"]) == 0
    assert "wm_basis_readback" not in sc_of(init, "dflt")["levers"]


def test_gate_refuses_a_lever_the_built_model_does_not_carry(init, monkeypatch):
    real = TR.make_config
    monkeypatch.setattr(TR, "make_config", lambda a, *r, **k: dataclasses.replace(real(a, *r, **k), wm_basis="adapter"))     # the ONE line that applies --wm-basis is deleted
    rc = gate(init, "g_unbuilt", "slow")
    assert rc != 0


def test_gate_refuses_a_cache_of_the_other_kind(init):
    rc = gate(init, "g_wrong", "slow", cache=str(init["root"] / "zb_pca"))
    assert rc != 0 and "ZZDVB-REFUSEDZZ" in str(rc)


def test_gate_refuses_a_basis_declared_without_the_windows_that_read_it(init):
    argv = ["--out", str(init["root"] / "g_nowin"), *BASE, "--init-ckpt", str(init["ck"]), "--wm-basis", "slow", "--wm-zcache", str(init["root"] / "zb_slow"), "--selfcheck-only"]
    with pytest.raises(SystemExit) as e:
        TR.main(argv)
    assert "ZZDVB-REFUSEDZZ" in str(e.value.code)


def test_a_wmonly_run_on_a_basis_cache_trains_proj_and_leaves_everything_else_bit_equal(init):
    out = init["root"] / "run_slow"
    argv = ["--out", str(out), *BASE, "--init-ckpt", str(init["ck"]), *LADDER, "--wm-basis", "slow", "--wm-zcache", str(init["root"] / "zb_slow"), "--steps", "3", "--log-every", "1", "--ckpt-every", "100"]
    assert TR.main(argv) == 0
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert s["done"] is True and s["wm_only"] is True
    log = [json.loads(ln) for ln in (out / "train_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert log and all(np.isfinite(r["wm"]) for r in log)
    a = torch.load(init["ck"], map_location="cpu", weights_only=False)["model"]
    ck1 = torch.load(out / "ckpt_last.pt", map_location="cpu", weights_only=False)
    b = ck1["model"]
    assert ck1["cfg"]["wm_basis"] == "slow"
    rest = [n for n in a if not n.startswith("wm.")]
    assert rest and all(torch.equal(a[n], b[n]) for n in rest)                                  # the adapter, readers, heads and planner are bit-equal
    sc = sc_of(init, "run_slow")
    assert sc["planner_identity_probe"]["pass"] is True and sc["levers"]["wm_basis_readback"]["cache_problems"] == []
