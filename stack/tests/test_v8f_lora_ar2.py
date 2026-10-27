"""PREREG_V8F_LORA Amendment 2: the box-2 chain for the SAME-DEVICE identity control and the AR-2 smoke -- the instruments' tests. CPU only; nothing here touches box 2.

Run: CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=<repo>/stack;<repo>/taniteval pytest -q stack/tests/test_v8f_lora_ar2.py

THE CLASS THIS FILE EXISTS FOR (CLAUDE.md, "built, tested, and unreachable from its caller"): the first identity control compared against REFERENCES THAT WERE NOT THE PRODUCTION CACHES
(`raw/2026-10-25-lora/identity_control.py:38-41`, `:212`) and its numbers were quoted as route evidence for days (R68). The control that replaces it is only worth anything if
  * it REFUSES a reference that is not a production-cache row (source / dtype / md5 / frame counts) BEFORE it compares anything,
  * a reading that is not the arm's own numerics on the builder's own device CANNOT pass (cpu, a frame subset, autocast, fewer than 2 + 2 clips),
  * a changed preprocessing constant FAILS it (and a "mutant caught" on a non-qualifying reading proves nothing),
  * the queue stage that runs it runs the line it PRINTS, with the declared clips, and the chain that arms it stops unless it PASSED.
Sections (test names carry them): `driver` (pure logic on synthetic tensors; then the whole `run()` on a cache the BUILDER'S OWN functions made), `reader`, `smoke` (the edits to
`v8f_lora_smoke.py`), `queue` (CRLF, old stages byte-identical, dry run, real execution against fakes), `chain` (order, guards, dry run).
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from tanitad.models import v8f_lora_encoder as E  # noqa: E402

torch.set_num_threads(2)
PKG = REPO / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-v8f-mini"
CODE = PKG / "code"
DRIVER_P, READ_P, SMOKE_P = CODE / "v8f_lora_identity_dev.py", CODE / "v8f_lora_ar2_read.py", CODE / "v8f_lora_smoke.py"
QUEUE = CODE / "v8f_mini_box2_queue.cmd"
CHAIN = CODE / "box2_ops" / "chain_v8f_lora_ar2.sh"
CHAIN_DSCALE = CODE / "box2_ops" / "chain_v8f_dscale.sh"
GUARD = CODE / "box2_ops" / "v8f_guard.sh"
PREREG = REPO / "Project Steering" / "PREREG_V8F_LORA.md"
PROD_TRAIN_MANIFEST = Path("C:/Users/Admin/box2_ship/logs/v8f_dscale/manifest_train.jsonl")      # box 2's production manifests, pulled by earlier chains (read-only)
PROD_EVAL_MANIFEST = Path("C:/Users/Admin/box2_ship/logs/v8f_grid/manifest_eval.jsonl")

TRAIN_IDS = ["8f387977-4086-466d-bba4-b04f8cdc0743", "bf5ec76e-e8b5-4f0d-a9cc-4fe1687d3d3a", "0e1526b3-d2b6-46b2-95fa-2903bb34e2ec", "83e32d0f-ed19-4116-8c1c-9e4c843de993"]
EVAL_IDS = ["dca157a0-20a9-464b-a74c-084b678e7be1", "e6cb7cd4-8349-4b9d-9674-e3ebf789e611"]

needs_pkg = pytest.mark.skipif(not (DRIVER_P.exists() and READ_P.exists()), reason="the v8F-mini package is not in this checkout")
needs_queue = pytest.mark.skipif(not QUEUE.exists(), reason="the v8F-mini package is not in this checkout")
needs_chain = pytest.mark.skipif(not CHAIN.exists(), reason="the v8F-mini package is not in this checkout")
win = pytest.mark.skipif(sys.platform != "win32", reason="the queue is a Windows batch file")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


DR = _load("v8f_lora_identity_dev_under_test", DRIVER_P) if DRIVER_P.exists() else None
RD = _load("v8f_lora_ar2_read_under_test", READ_P) if READ_P.exists() else None
_H: dict = {}


def lora_helpers():
    """The tiny-encoder helpers of test_v8f_lora.py (loaded, not copied)."""
    if "m" not in _H:
        try:
            _H["m"] = _load("test_v8f_lora_for_ar2", Path(__file__).resolve().parent / "test_v8f_lora.py")
        except BaseException as e:                                       # noqa: BLE001  (pytest.skip.Exception is a BaseException)
            pytest.skip(f"test_v8f_lora helpers unavailable: {e}")
    return _H["m"]


# =================================================================================================================================================
# DRIVER: the pure logic
# =================================================================================================================================================
def _fp8(n=3, seed=0, scale=2.0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, 40, 1024, generator=g) * scale).to(torch.float8_e4m3fn)


@needs_pkg
def test_driver_compare_an_identical_cache_reads_exactly_zero_and_a_changed_one_is_seen():
    a = _fp8()
    r = DR.compare_pooled(a.clone(), a)
    assert r["rel_l2"] == 0.0 and r["max_abs"] == 0.0 and r["frac_bitwise_equal"] == 1.0 and r["n_values"] == a.numel()
    scaled = (a.float() * 1.05).to(torch.float8_e4m3fn)                                  # a 5 % scaled copy: the size of the device / route noise this control exists to see
    r = DR.compare_pooled(scaled, a)
    assert 0.01 < r["rel_l2"] < 0.2 and r["frac_bitwise_equal"] < 0.9
    one = a.clone()
    u = one.view(torch.uint8)
    u[0, 0, 0] = int(u[0, 0, 0]) ^ 1                                                   # ONE fp8 byte, its lowest mantissa bit
    r = DR.compare_pooled(one, a)
    assert 0.0 < r["rel_l2"] < 1e-3 and r["max_abs"] > 0 and abs(r["frac_bitwise_equal"] - (1 - 1 / a.numel())) < 1e-12
    nan = a.clone()
    nan.view(torch.uint8)[0, 0, 0] = 0x7F                                                 # the e4m3fn NaN
    assert DR.compare_pooled(nan, a)["rel_l2"] == float("inf"), "a non-finite reading is never a pass"
    with pytest.raises(ValueError):
        DR.compare_pooled(a.float(), a)
    with pytest.raises(ValueError):
        DR.compare_pooled(a[:2], a)


def _pc(train, eval_):
    d = {f"t{i}": {"split": "train", "rel_l2": v} for i, v in enumerate(train)}
    d.update({f"e{i}": {"split": "eval", "rel_l2": v} for i, v in enumerate(eval_)})
    return d


GOOD = dict(device_type="cuda", all_frames=True, autocast=False)


@needs_pkg
@pytest.mark.parametrize("name,per,scope,verdict,why", [
    ("identity", _pc([0.0, 0.0], [0.0, 0.0]), GOOD, "PASS", None),
    ("at the bar", _pc([0.0, 0.0], [0.0, 1e-3]), GOOD, "PASS", None),
    ("just over the bar", _pc([0.0, 0.0], [0.0, 1.0000001e-3]), GOOD, "FAIL", "rel-L2 >"),
    ("one train clip over", _pc([2e-2, 0.0], [0.0, 0.0]), GOOD, "FAIL", "rel-L2 >"),
    ("nan", _pc([0.0, float("nan")], [0.0, 0.0]), GOOD, "FAIL", "rel-L2 >"),
    ("cpu", _pc([0.0, 0.0], [0.0, 0.0]), dict(GOOD, device_type="cpu"), "FAIL", "SAME-DEVICE"),
    ("a frame subset", _pc([0.0, 0.0], [0.0, 0.0]), dict(GOOD, all_frames=False), "FAIL", "subset"),
    ("autocast", _pc([0.0, 0.0], [0.0, 0.0]), dict(GOOD, autocast=True), "FAIL", "autocast"),
    ("one train clip", _pc([0.0], [0.0, 0.0]), GOOD, "FAIL", "Amendment 2 needs"),
    ("one eval clip", _pc([0.0, 0.0], [0.0]), GOOD, "FAIL", "Amendment 2 needs"),
    ("six clips", _pc([0.0] * 4, [0.0] * 2), GOOD, "PASS", None),
])
def test_driver_decide_passes_iff_every_clip_is_within_tau_and_the_reading_qualifies(name, per, scope, verdict, why):
    d = DR.decide(per, scope=scope)
    assert d["verdict"] == verdict, (name, d)
    assert d["qualifying"] == (name in ("identity", "at the bar", "just over the bar", "one train clip over", "nan", "six clips"))
    assert d["numbers_pass_tau"] == (name not in ("just over the bar", "one train clip over", "nan"))
    if why:
        assert any(why in r for r in d["fail_reasons"]), (name, d["fail_reasons"])
    else:
        assert d["fail_reasons"] == []
    assert d["worst_rel_l2"] is not None


@needs_pkg
def test_driver_decide_an_empty_reading_and_a_custom_tau():
    assert DR.decide({}, scope=GOOD)["verdict"] == "FAIL"
    per = _pc([5e-4, 0.0], [0.0, 0.0])
    assert DR.decide(per, scope=GOOD)["verdict"] == "PASS" and DR.decide(per, tau=1e-4, scope=GOOD)["verdict"] == "FAIL"


def _row(clip="c-1", **kv):
    r = {"clip": clip, "sha12": "x", "file": f"clips/{clip}.pt", "bytes": 1, "md5": "m", "content_md5": "c", "n_frames": 101, "t_ep": 201, "dtype": "float8_e4m3fn", "enc_dtype": "bf16",
         "device": "cuda", "source": "local256", "grids": ["4x10", "8x20"], "shapes": {"pool4x10": [101, 40, 1024]}, "pipeline": "ra1a-pooled-v1"}
    r.update(kv)
    return r


@needs_pkg
@pytest.mark.parametrize("kv,frag", [
    ({"source": "thor416"}, "source"), ({"source": "thor416-resize"}, "source"), ({"enc_dtype": "fp32"}, "enc_dtype"), ({"dtype": "float16"}, "dtype"), ({"device": "cpu"}, "device"),
    ({"grids": ["8x20"]}, "grids"), ({"shapes": {"pool4x10": [101, 160, 1024]}}, "shapes"), ({"n_frames": 100}, "n_frames"), ({"md5": None}, "md5"), ({"content_md5": ""}, "content_md5"),
    ({"clip": "other"}, "row is for clip"), ({"t_ep": None}, "t_ep"), ({"file": None}, "file")])
def test_driver_production_row_assertion_refuses_a_reference_that_is_not_the_production_cache(kv, frag):
    assert DR.production_problems(_row(), "c-1") == []
    bad = DR.production_problems(_row(**kv), "c-1")
    assert bad and any(frag in b for b in bad), bad


@needs_pkg
def test_driver_constants_are_amendment_2_and_the_builders():
    assert DR.TAU_DEVICE == 1e-3 and (DR.MIN_TRAIN, DR.MIN_EVAL) == (2, 2) and DR.REQUIRED_DEVICE_TYPE == "cuda" and DR.EXPECTED_TRAINABLE == 3_145_728 == 32 * 2048 * 2 * 24
    assert DR.PRODUCTION_ROW == {"source": "local256", "enc_dtype": "bf16", "dtype": "float8_e4m3fn"} and DR.GRID == "4x10"
    assert DR.build_parser().parse_args(["--cache-root", "x", "--out", "y"]).tau == 1e-3
    src = DRIVER_P.read_text(encoding="utf-8")
    assert "torch.autocast" not in src, "Amendment 2 pins the builder's pure-bf16 numerics: the driver never autocasts"
    assert 'dtype=torch.bfloat16' in src and "verify_sha=True" in src and "r=32, alpha=32.0, targets=(\"q\", \"v\")" in src
    assert "inference_mode" in src and "pool_field(f, 4, 10).to(torch.float8_e4m3fn)" in src, "pool in fp32 on the CPU, THEN quantise: exactly payload_from_field"
    ref_src = Path(E.builder_paths()[1]).read_text(encoding="utf-8") if E.builder_paths()[1].is_file() else None
    if ref_src is not None:
        assert "range(0, frames.shape[0], 8)" in ref_src and DR.BUILDER_BATCH == 8, "the builder encodes in batches of 8 from frame 0: the control reproduces that composition"
    assert set(DR.MUTANTS.values()) <= {("PIXEL_SCALE", 256.0), ("IMAGE_MEAN", (0.486, 0.456, 0.406)), ("IMAGE_STD", (0.229, 0.224, 0.226))}
    for attr, _ in DR.MUTANTS.values():
        assert hasattr(E, attr), f"the mirror has no {attr} to mutate"
    if PREREG.exists():
        t = PREREG.read_text(encoding="utf-8")
        assert "τ_device = 1e-3" in t and "SAME-DEVICE" in t and "≥ 2 train and ≥ 2 eval clips" in t


@needs_pkg
def test_driver_spread_and_cli_arguments():
    assert DR.spread(101, 0) == list(range(101)) and DR.spread(101, 101) == list(range(101)) and DR.spread(101, 500) == list(range(101))
    s = DR.spread(101, 8)
    assert len(s) == 8 and s[0] == 0 and s[-1] == 100 and s == sorted(set(s))
    a = DR.build_parser().parse_args(["--cache-root", "x", "--out", "y", "--mutant"])
    assert a.mutant == DR.DEFAULT_MUTANT
    assert DR.build_parser().parse_args(["--cache-root", "x", "--out", "y", "--mutant", "IMAGE_MEAN0_0.485_to_0.486"]).mutant == "IMAGE_MEAN0_0.485_to_0.486"
    assert DR.build_parser().parse_args(["--cache-root", "x", "--out", "y"]).mutant is None


# =================================================================================================================================================
# DRIVER: the whole run() on a cache the BUILDER'S OWN functions made (a 2-layer d = 1024 bf16 ViT at the real 256x640 geometry; CPU)
# =================================================================================================================================================
N_FR = 3
ENC_KW = dict(hidden=1024, layers=2, dtype=torch.bfloat16, r=4, alpha=4.0, bb_seed=3)


def _make_enc():
    return lora_helpers().make_encoder(**ENC_KW)


def _write_cache_clip(cache, split, clip, o, enc, perturb=None, row_kv=None):
    """The production cache entry for ``clip`` made by the BUILDER'S functions (``encode_episode`` + ``payload_from_field``), a manifest row that describes it."""
    pep, ship = E.load_builder()
    f = ship.encode_episode(o, copy.deepcopy(enc.merge()), enc.mean, enc.std, 5, "cpu")
    pay = pep.payload_from_field(f, o, ["4x10"], torch.float8_e4m3fn)
    if perturb is not None:
        pay["pool4x10"] = perturb(pay["pool4x10"])
    d = Path(cache) / split / "clips"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{clip}.pt"
    torch.save(pay, p)
    row = {"clip": clip, "sha12": hashlib.sha256(clip.encode()).hexdigest()[:12], "file": f"clips/{clip}.pt", "bytes": p.stat().st_size, "md5": hashlib.md5(p.read_bytes()).hexdigest(),
           "content_md5": pep.content_md5(pay), "n_frames": N_FR, "t_ep": 2 * N_FR - 1, "dtype": "float8_e4m3fn", "enc_dtype": "bf16", "device": "cuda", "source": "local256",
           "grids": ["4x10", "8x20"], "shapes": {"pool4x10": [N_FR, 40, 1024]}, "pipeline": "ra1a-pooled-v1"}
    row.update(row_kv or {})
    return row


def _write_world(root, perturb_clip=None, perturb=None, row_kv=None):
    H = lora_helpers()
    enc = _make_enc()
    root = Path(root)
    names = {"train": ["syn-tr-a", "syn-tr-b"], "eval": ["syn-ev-a", "syn-ev-b"]}
    rows = {"train": [], "eval": []}
    i = 0
    for split, cl in names.items():
        for c in cl:
            o = H.v2ep_from_frames(H.smooth_frames(N_FR, (256, 640), seed=i), c)
            fd = root / "frames" / split
            fd.mkdir(parents=True, exist_ok=True)
            torch.save(o, fd / f"{c}.v2ep.pt")
            rows[split].append(_write_cache_clip(root / "cache", split, c, o, enc, perturb if c == perturb_clip else None, (row_kv or {}).get(c)))
            i += 1
    for split in rows:
        (root / "cache" / split / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows[split]) + "\n", encoding="utf-8")
    return {"root": root, "cache": root / "cache", "frames": root / "frames", **{k: v for k, v in names.items()}}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return _write_world(tmp_path_factory.mktemp("ar2w"))


@pytest.fixture
def cpu_is_qualifying(monkeypatch):
    """The PASS path needs a qualifying device; the tests have only a CPU, so the REQUIRED device type is moved for them (and ONLY for them)."""
    monkeypatch.setattr(DR, "REQUIRED_DEVICE_TYPE", "cpu")
    monkeypatch.setattr(DR, "EXPECTED_TRAINABLE", 4 * 2048 * 2 * 2)
    yield


def _args(w, tmp, *extra, train=None, eval_=None, cache=None):
    return DR.build_parser().parse_args(["--cache-root", str(cache or w["cache"]), "--train-clips", *(train or w["train"]), "--eval-clips", *(eval_ or w["eval"]),
                                         "--train-frames", str(w["frames"] / "train"), "--eval-frames", str(w["frames"] / "eval"), "--device", "cpu", "--out", str(tmp / "x.json"), *extra])


class Spy:
    def __init__(self):
        self.n = 0

    def __call__(self, dev):
        self.n += 1
        return _make_enc()


@needs_pkg
def test_driver_run_identity_passes_on_a_cache_the_builders_own_functions_made(world, tmp_path, cpu_is_qualifying):
    spy = Spy()
    res, rc = DR.run(_args(world, tmp_path), build=spy)
    assert rc == 0 and res["verdict"] == "PASS" and res["qualifying"] and res["numbers_pass_tau"], res.get("fail_reasons")
    assert spy.n == 1 and res["worst_rel_l2"] == 0.0 and res["dtype"] == "torch.bfloat16" and res["autocast"] is False and res["batch"] == 8
    assert set(res["clips"]) == set(world["train"] + world["eval"])
    for c, v in res["clips"].items():
        assert v["rel_l2"] == 0.0 and v["max_abs"] == 0.0 and v["frac_bitwise_equal"] == 1.0 and v["n_compared"] == v["n_frames"] == N_FR, c
        assert v["manifest_row_matched"]["source"] == "local256" and v["manifest_row_matched"]["enc_dtype"] == "bf16" and v["stored_md5"] == v["manifest_row_matched"]["md5"]
        assert re.fullmatch(r"[0-9a-f]{64}", v["frames_sha256"])
    assert res["builder_unchanged"] is True and res["builder_encode_episode_sha256"] == E.BUILDER_ENCODE_EPISODE_SHA256 and re.fullmatch(r"[0-9a-f]{64}", res["driver_sha256"])
    assert res["driver_sha256"] == hashlib.sha256(DRIVER_P.read_bytes()).hexdigest()


@needs_pkg
def test_driver_run_a_perturbed_stored_cache_with_a_consistent_row_fails(tmp_path, cpu_is_qualifying):
    """The stored tokens differ by ~5 % (the size of a device / route gap) and the row describes them faithfully (md5 and all): the control must FAIL on that clip, name it, and exit 1."""
    w = _write_world(tmp_path / "w", perturb_clip="syn-ev-b", perturb=lambda p: (p.float() * 1.05).to(torch.float8_e4m3fn))
    res, rc = DR.run(_args(w, tmp_path), build=Spy())
    assert rc == 1 and res["verdict"] == "FAIL" and res["numbers_pass_tau"] is False and res["qualifying"] is True
    assert res["worst_clip"] == "syn-ev-b" and res["worst_rel_l2"] > 0.01
    assert res["clips"]["syn-ev-a"]["rel_l2"] == 0.0 and res["clips"]["syn-tr-a"]["rel_l2"] == 0.0, "only the perturbed clip moves"
    assert any("syn-ev-b"[:8] in r for r in res["fail_reasons"])


@needs_pkg
def test_driver_run_refuses_a_row_that_is_not_a_production_row_before_the_model_is_built(tmp_path, cpu_is_qualifying):
    w = _write_world(tmp_path / "w", row_kv={"syn-tr-b": {"source": "thor416-resize", "enc_dtype": "fp32"}})
    spy = Spy()
    res, rc = DR.run(_args(w, tmp_path), build=spy)
    assert rc == 2 and res["verdict"] == "REFUSED" and "not a production-cache row" in res["refused"] and "source" in res["refused"] and spy.n == 0, "nothing was compared, no model was built"


@needs_pkg
def test_driver_run_refuses_a_stored_file_that_is_not_the_one_the_row_describes(world, tmp_path, cpu_is_qualifying):
    cache = tmp_path / "cache"
    shutil.copytree(world["cache"], cache)
    p = cache / "train" / "clips" / "syn-tr-a.pt"
    pay = torch.load(p, weights_only=True)
    pay["pool4x10"] = (pay["pool4x10"].float() * 1.5).to(torch.float8_e4m3fn)
    torch.save(pay, p)                                                                      # the file changed, the row still carries the OLD md5
    spy = Spy()
    res, rc = DR.run(_args(world, tmp_path, cache=cache), build=spy)
    assert rc == 2 and "not the cache the row describes" in res["refused"] and spy.n == 0
    (cache / "train" / "clips" / "syn-tr-b.pt").unlink()
    res, rc = DR.run(_args(world, tmp_path, train=["syn-tr-b"], cache=cache), build=spy)
    assert rc == 2 and "does not exist" in res["refused"] and spy.n == 0, "no stored tokens => no reference => refused (no other reference is used)"
    res, rc = DR.run(_args(world, tmp_path, train=["syn-nope"], cache=cache), build=spy)
    assert rc == 2 and "has no row" in res["refused"]


@needs_pkg
def test_driver_run_refuses_frames_that_are_not_the_ones_the_cache_was_encoded_from(world, tmp_path, cpu_is_qualifying):
    H = lora_helpers()
    fr = tmp_path / "frames"
    shutil.copytree(world["frames"], fr)
    long = H.v2ep_from_frames(H.smooth_frames(N_FR + 2, (256, 640), seed=9), "syn-tr-a")      # another episode length
    torch.save(long, fr / "train" / "syn-tr-a.v2ep.pt")
    w = dict(world, frames=fr)
    spy = Spy()
    res, rc = DR.run(_args(w, tmp_path), build=spy)
    assert rc == 2 and "not the frames the cache was encoded from" in res["refused"] and spy.n == 0
    torch.save(H.v2ep_from_frames(H.smooth_frames(N_FR, (256, 640), seed=0), "syn-other"), fr / "train" / "syn-tr-a.v2ep.pt")
    res, rc = DR.run(_args(w, tmp_path), build=spy)
    assert rc == 2 and "clip_id" in res["refused"] and spy.n == 0
    (fr / "train" / "syn-tr-a.v2ep.pt").unlink()
    res, rc = DR.run(_args(w, tmp_path), build=spy)
    assert rc == 2 and "frames file" in res["refused"] and spy.n == 0


@needs_pkg
def test_driver_run_on_cpu_is_refused_unless_labelled_and_then_never_qualifies(world, tmp_path):
    """Amendment 2 withdrew the CPU tolerance: CPU vs the GPU-built cache reads rel-L2 0.03 by construction. Without the labelled flag the run is REFUSED; with it the numbers are
    reported but the verdict is FAIL whatever they say."""
    spy = Spy()
    res, rc = DR.run(_args(world, tmp_path), build=spy)
    assert rc == 2 and "SAME-DEVICE" in res["refused"] and spy.n == 0
    monkey = pytest.MonkeyPatch()
    monkey.setattr(DR, "EXPECTED_TRAINABLE", 4 * 2048 * 2 * 2)
    try:
        res, rc = DR.run(_args(world, tmp_path, "--allow-non-cuda"), build=spy)
    finally:
        monkey.undo()
    assert rc == 1 and res["verdict"] == "FAIL" and res["numbers_pass_tau"] is True and res["qualifying"] is False and res["worst_rel_l2"] == 0.0
    assert any("SAME-DEVICE" in r for r in res["fail_reasons"])


@needs_pkg
def test_driver_run_a_frame_subset_never_qualifies(world, tmp_path, cpu_is_qualifying):
    res, rc = DR.run(_args(world, tmp_path, "--frames-per-clip", "2"), build=Spy())
    assert rc == 1 and res["verdict"] == "FAIL" and res["qualifying"] is False and res["scope"]["all_frames"] is False
    assert all(v["n_compared"] == 2 and v["n_frames"] == N_FR for v in res["clips"].values()) and any("subset" in r for r in res["fail_reasons"])


@needs_pkg
def test_driver_run_with_fewer_than_two_and_two_clips_never_qualifies(world, tmp_path, cpu_is_qualifying):
    res, rc = DR.run(_args(world, tmp_path, train=["syn-tr-a"]), build=Spy())
    assert rc == 1 and res["verdict"] == "FAIL" and res["qualifying"] is False and any("Amendment 2 needs" in r for r in res["fail_reasons"])


@needs_pkg
@pytest.mark.parametrize("name", sorted(["PIXEL_SCALE_255_to_256", "IMAGE_MEAN0_0.485_to_0.486", "IMAGE_STD2_0.225_to_0.226"]))
def test_driver_mutant_a_changed_preprocessing_constant_fails_and_is_restored(world, tmp_path, cpu_is_qualifying, name):
    before = (E.PIXEL_SCALE, E.IMAGE_MEAN, E.IMAGE_STD)
    spy = Spy()
    res, rc = DR.run(_args(world, tmp_path, "--mutant", name), build=spy)
    assert res["mutant"] == name and res["verdict"] == "FAIL" and res["numbers_pass_tau"] is False and res["qualifying"] is True, res.get("fail_reasons")
    assert res["mutant_caught"] is True and rc == 0, "rc 0 = the mutant was CAUGHT"
    assert res["worst_rel_l2"] > DR.TAU_DEVICE and res["expected_verdict"] == "FAIL"
    assert (E.PIXEL_SCALE, E.IMAGE_MEAN, E.IMAGE_STD) == before, "the mutation is undone"
    ok, rc2 = DR.run(_args(world, tmp_path), build=spy)
    assert rc2 == 0 and ok["verdict"] == "PASS", "and the unmutated control still passes afterwards (the encoder was rebuilt, the constants restored)"


@needs_pkg
def test_driver_mutant_that_changes_nothing_is_not_caught(world, tmp_path, cpu_is_qualifying, monkeypatch):
    monkeypatch.setattr(DR, "apply_mutant", lambda enc, name, device: (lambda: None))              # the mutation is a no-op: the instrument must say it is blind
    res, rc = DR.run(_args(world, tmp_path, "--mutant"), build=Spy())
    assert res["numbers_pass_tau"] is True and res["mutant_caught"] is False and rc == 1


@needs_pkg
def test_driver_mutant_caught_on_a_non_qualifying_device_proves_nothing(world, tmp_path, monkeypatch):
    monkeypatch.setattr(DR, "EXPECTED_TRAINABLE", 4 * 2048 * 2 * 2)
    res, rc = DR.run(_args(world, tmp_path, "--mutant", "--allow-non-cuda"), build=Spy())
    assert res["numbers_pass_tau"] is False and res["qualifying"] is False and res["mutant_caught"] is False and rc == 1


@needs_pkg
def test_driver_manifest_only_asserts_the_rows_and_never_builds_a_model(world, tmp_path):
    spy = Spy()
    res, rc = DR.run(_args(world, tmp_path, "--manifest-only"), build=spy)
    assert rc == 0 and res["verdict"] == "MANIFEST-OK" and res["n_clips"] == 4 and res["stored_files_checked"] == 4 and spy.n == 0
    cache = tmp_path / "bad"
    shutil.copytree(world["cache"], cache)
    rows = [json.loads(x) for x in (cache / "eval" / "manifest.jsonl").read_text().splitlines()]
    rows[0]["source"] = "thor416"
    (cache / "eval" / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    res, rc = DR.run(_args(world, tmp_path, "--manifest-only", cache=cache), build=spy)
    assert rc == 2 and "source" in res["refused"] and spy.n == 0
    # the files need not exist in manifest-only mode (the chain runs it on the manifests it pulled, without the 21 MB clips)
    nofiles = tmp_path / "nofiles"
    shutil.copytree(world["cache"], nofiles, ignore=shutil.ignore_patterns("clips"))
    res, rc = DR.run(_args(world, tmp_path, "--manifest-only", cache=nofiles), build=spy)
    assert rc == 0 and res["stored_files_checked"] == 0


@needs_pkg
def test_driver_main_writes_its_json_even_on_a_refusal(world, tmp_path, capsys):
    out = tmp_path / "o" / "r.json"
    rc = DR.main(["--cache-root", str(tmp_path / "nowhere"), "--train-clips", "a", "--eval-clips", "b", "--train-frames", "x", "--eval-frames", "y", "--out", str(out)])
    assert rc == 2 and json.loads(out.read_text())["verdict"] == "REFUSED" and "REFUSED" in capsys.readouterr().out
    assert DR.main(["--cache-root", "z", "--train-clips", "a", "--out", str(tmp_path / "q.json")]) == 2, "--train-clips needs --train-frames"
    rc = DR.main(["--cache-root", str(world["cache"]), "--train-clips", *world["train"], "--eval-clips", *world["eval"], "--manifest-only", "--out", str(tmp_path / "m.json")])
    assert rc == 0 and json.loads((tmp_path / "m.json").read_text())["verdict"] == "MANIFEST-OK"


# =================================================================================================================================================
# READER
# =================================================================================================================================================
def _ident(**kv):
    clips = {c: {"split": "train" if c in TRAIN_IDS else "eval", "rel_l2": 0.0, "frac_bitwise_equal": 1.0, "n_frames": 101, "n_compared": 101,
                 "manifest_row_matched": {"source": "local256", "enc_dtype": "bf16"}} for c in TRAIN_IDS + EVAL_IDS}
    d = {"verdict": "PASS", "qualifying": True, "numbers_pass_tau": True, "mutant": None, "tau_device": 1e-3, "clips": clips, "worst_rel_l2": 0.0, "worst_clip": TRAIN_IDS[0],
         "weights_sha256": E.BUILDER_MODEL_SHA256, "builder_model_sha256": E.BUILDER_MODEL_SHA256, "builder_unchanged": True, "driver_sha256": "d" * 64, "cuda": "NVIDIA GeForce RTX 4060"}
    d.update(kv)
    return d


def _mut(**kv):
    m = _ident(verdict="FAIL", numbers_pass_tau=False, mutant="PIXEL_SCALE_255_to_256", mutant_caught=True, worst_rel_l2=2.1e-2)
    m.update(kv)
    return m


@needs_pkg
def test_reader_identity_passes_only_a_qualifying_pass_with_a_caught_mutant_on_the_declared_clips():
    j = RD.judge_identity(_ident(), _mut(), TRAIN_IDS, EVAL_IDS, "d" * 64)
    assert j["ok"] and j["problems"] == [] and j["mutant_caught"] is True and j["worst_rel_l2"] == 0.0 and j["mutant_worst_rel_l2"] == 2.1e-2
    bad = {
        "did not PASS": (_ident(verdict="FAIL", fail_reasons=["rel-L2 > 0.001 on 1 of 6 clips"], numbers_pass_tau=False), _mut()),
        "not qualifying": (_ident(qualifying=False, qualifying_reasons=["device 'cpu'"]), _mut()),
        "is a mutant run": (_ident(mutant="PIXEL_SCALE_255_to_256"), _mut()),
        "covered": (_ident(clips={c: v for c, v in list(_ident()["clips"].items())[:5]}), _mut()),
        "tau_device": (_ident(tau_device=1e-2), _mut()),
        "weights' sha256": (_ident(weights_sha256="0" * 64), _mut()),
        "pinned against": (_ident(builder_unchanged=False), _mut()),
        "production row": (_ident(clips={c: dict(v, manifest_row_matched={"source": "thor416", "enc_dtype": "bf16"}) for c, v in _ident()["clips"].items()}), _mut()),
        "frames compared": (_ident(clips={c: dict(v, n_compared=8) for c, v in _ident()["clips"].items()}), _mut()),
        "ran driver": (_ident(driver_sha256="e" * 64), _mut()),
        "no mutant reading": (_ident(), None),
        "NOT caught": (_ident(), _mut(mutant_caught=False, verdict="PASS", numbers_pass_tau=True)),
        "not a mutant run": (_ident(), _mut(mutant=None)),
        "did not cover": (_ident(), _mut(clips={})),
    }
    for frag, (r, m) in bad.items():
        j = RD.judge_identity(r, m, TRAIN_IDS, EVAL_IDS, "d" * 64)
        assert not j["ok"] and any(frag in p for p in j["problems"]), (frag, j["problems"])


def _smoke_rep(**kv):
    rep = {"what": "PREREG_V8F_LORA AR-2 smoke", "amp": False, "device": "cuda", "cuda": "NVIDIA GeForce RTX 4060", "cpu_fp32_mechanics_proof": False, "rc": 0,
           "trainer_argv": ["--arm", "M2b", "--encoder", "lora", "--enc-ckpt"],
           "selfcheck": {"levers": {"encoder": {"backbone_dtype": "torch.bfloat16", "checkpointing": True, "lora_trainable": 3_145_728}}},
           "verdict": {"AR2_gradient_and_finite_checks_pass": True, "n_steps_run": 200, "frozen_dinov3_control_exactly_zero": True},
           "identity_probe_tau_device": {"tau_device": 1e-3, "rel_l2": 4e-4, "pass": True},
           "measured_batch1": {"step_wall_s_median": 4.0, "iter_s_median": 4.1, "n_steps_in_medians": 195, "unique_frames_per_step_median": 14.0, "phase1_encode_nograd_s_median": 0.6,
                               "heads_fwd_bwd_s_median": 0.1, "phase2_encode_grad_s_median": 3.2, "peak_alloc_gib_max": 5.5, "peak_reserved_gib_max": 6.5, "loss_first_to_last": [9.0, 7.5]},
           "projection_ESTIMATED": {"hours_one_seed_ESTIMATED": 120.0, "steps": 19164, "step_s_b64_ESTIMATED": 22.5, "U_unique_frames_per_window_b64": 11.0,
                                    "census_scope": {"smoke_trainer_argv_has_max_clips": True, "census_and_heads_clips": 4512}, "assumes_micro_batch_frames": 16,
                                    "t_heads_b64": {"peak_alloc_gib_max": 2.5}}}
    rep.update(kv)
    return rep


def _sc(**kv):
    return {"info": dict({"n_clips": 4, "train_sha256": RD.list_sha256(TRAIN_IDS), "frames_missing": 0}, **kv)}


@needs_pkg
def test_reader_smoke_reports_step_time_peak_memory_and_the_projected_hours_and_decides_the_venue():
    steps = [{"step_wall_s": 4.0}, {"step_wall_s": 4.2}]
    s = RD.read_smoke(_smoke_rep(), steps, TRAIN_IDS, _sc())
    assert s["ok"], s["problems"]
    assert s["step_time_batch1_s"]["median"] == 4.0 and s["peak_memory_gib"]["allocated_max"] == 5.5 and s["peak_memory_gib"]["reserved_max"] == 6.5
    assert s["projection_ESTIMATED"]["hours_one_seed"] == 120.0 and "ESTIMATED" in json.dumps(s["evidence_class"]) and s["venue_rule_proof5"].startswith("A40")
    assert s["projection_ESTIMATED"]["assumes_micro_batch_frames"] == 16 and s["peak_memory_gib"]["heads_alone_at_batch_64_allocated"] == 2.5
    assert RD.venue_of(36.0, True).startswith("BOX 2") and RD.venue_of(36.0001, True).startswith("A40") and RD.venue_of(1.0, False).startswith("n/a") and RD.venue_of(None, True).startswith("n/a")
    line = RD.summary_line(RD.judge_identity(_ident(), _mut(), TRAIN_IDS, EVAL_IDS), s)
    assert "projected 120.0 h/seed (ESTIMATED)" in line and "step 4.00 s" in line and "5.50 GiB alloc / 6.50 reserved" in line and "identity PASS" in line
    pr = _smoke_rep()["projection_ESTIMATED"]
    assert RD.read_smoke(_smoke_rep(projection_ESTIMATED=dict(pr, hours_one_seed_ESTIMATED=30.0)), steps, TRAIN_IDS, _sc())["venue_rule_proof5"].startswith("BOX 2")


@needs_pkg
def test_reader_smoke_refuses_every_way_the_smoke_can_be_the_wrong_experiment():
    steps = [{"step_wall_s": 4.0}]
    cases = {
        "autocast was ON": _smoke_rep(amp=True),
        "autocast was ON ": _smoke_rep(trainer_argv=["--amp"]),
        "AR-2 gradient": _smoke_rep(verdict={"AR2_gradient_and_finite_checks_pass": False, "n_steps_run": 200, "frozen_dinov3_control_exactly_zero": False}),
        "rc 1": _smoke_rep(rc=1),
        "CPU mechanics": _smoke_rep(cpu_fp32_mechanics_proof=True),
        "not a cuda device": _smoke_rep(device="cpu"),
        "199 steps": _smoke_rep(verdict={"AR2_gradient_and_finite_checks_pass": True, "n_steps_run": 199}),
        "built encoder is not bf16": _smoke_rep(selfcheck={"levers": {"encoder": {"backbone_dtype": "torch.float32", "checkpointing": True, "lora_trainable": 3_145_728}}}),
        "gradient checkpointing": _smoke_rep(selfcheck={"levers": {"encoder": {"backbone_dtype": "torch.bfloat16", "checkpointing": False, "lora_trainable": 3_145_728}}}),
        "no projection": _smoke_rep(projection_ESTIMATED={}),
        "TRAINING path's identity probe misses tau_device": _smoke_rep(identity_probe_tau_device={"tau_device": 1e-3, "rel_l2": 2.4e-2, "pass": False}),
        "no training-path identity probe": _smoke_rep(identity_probe_tau_device=None),
    }
    for frag, rep in cases.items():
        s = RD.read_smoke(rep, steps, TRAIN_IDS, _sc())
        assert not s["ok"] and any(frag.strip() in p for p in s["problems"]), (frag, s["problems"])
    assert any("trained on 3 clips" in p for p in RD.read_smoke(_smoke_rep(), steps, TRAIN_IDS, _sc(n_clips=3))["problems"])
    assert any("trained on" in p for p in RD.read_smoke(_smoke_rep(), steps, TRAIN_IDS, _sc(train_sha256="0" * 64))["problems"])
    assert any("had no frames" in p for p in RD.read_smoke(_smoke_rep(), steps, TRAIN_IDS, _sc(frames_missing=2))["problems"])
    assert any("no selfcheck.json" in p for p in RD.read_smoke(_smoke_rep(), steps, TRAIN_IDS, None)["problems"])


def _syn_rows(n, tag):
    return {f"{tag}-{i:05d}": {"clip": f"{tag}-{i:05d}", "source": "local256", "enc_dtype": "bf16"} for i in range(n)}


@needs_pkg
def test_reader_check_declared_the_declared_clips_are_the_first_of_the_manifest_in_sha12_order():
    tr, ev = _syn_rows(4572, "t"), _syn_rows(141, "e")
    want = RD.first_train_clips(tr, 4)
    order = sorted(tr, key=RD.sha12)
    assert want == order[:4] and not set(order[540:600]) & set(want)
    ex = set(order[540:600])
    assert RD.first_train_clips(tr, 600) == [c for c in order if c not in ex][:600] and len(set(RD.first_train_clips(tr, 600)) & ex) == 0, "the RA-1 holdout is withheld"
    evs = sorted(ev, key=RD.sha12)[:2]
    assert RD.check_declared(tr, ev, want, evs, 4) == []
    assert any("not the first 4" in b for b in RD.check_declared(tr, ev, want[::-1], evs, 4)), "order matters: --max-clips takes the first N"
    assert any("not the first 4" in b for b in RD.check_declared(tr, ev, want[:3] + [order[4]], evs, 4))
    assert any("eval clips" in b for b in RD.check_declared(tr, ev, want, evs[::-1], 4))
    assert any("4,572" in b for b in RD.check_declared(dict(list(tr.items())[:-1]), ev, want, evs, 4))
    assert any("141" in b for b in RD.check_declared(tr, dict(list(ev.items())[:-1]), want, evs, 4))
    tr2 = dict(tr)
    tr2[want[0]] = dict(tr[want[0]], source="thor416")
    assert any("source" in b for b in RD.check_declared(tr2, ev, want, evs, 4))
    assert len(RD.list_sha256(want)) == 64 and RD.list_sha256(want) == RD.list_sha256(want[::-1]), "train_sha256 is over the sha12-sorted list"


@needs_pkg
@pytest.mark.skipif(not (PROD_TRAIN_MANIFEST.exists() and PROD_EVAL_MANIFEST.exists()), reason="box 2's production manifests are not on this disk")
def test_reader_check_declared_on_the_real_production_manifests_holds_the_queue_and_chain_lists():
    tr, ev = RD.read_rows(PROD_TRAIN_MANIFEST), RD.read_rows(PROD_EVAL_MANIFEST)
    assert RD.check_declared(tr, ev, TRAIN_IDS, EVAL_IDS, 4) == [], "the declared clips ARE the first 4 train (holdout withheld) and first 2 eval clips of the production manifests"
    assert (len(tr), len(ev)) == (4572, 141) and {(r["source"], r["enc_dtype"], r["device"]) for r in list(tr.values()) + list(ev.values())} == {("local256", "bf16", "cuda")}
    # the list the queue and the chain carry
    q = QUEUE.read_text(encoding="utf-8").replace("\r\n", "\n") if QUEUE.exists() else ""
    if q:
        assert re.search(r'set "LORA_TRAIN=([^"]+)"', q).group(1).split() == TRAIN_IDS and re.search(r'set "LORA_EVAL=([^"]+)"', q).group(1).split() == EVAL_IDS
    if CHAIN.exists():
        c = CHAIN.read_text(encoding="utf-8")
        assert re.search(r'^TRAIN_CLIPS="([^"]+)"', c, re.M).group(1).split() == TRAIN_IDS and re.search(r'^EVAL_CLIPS="([^"]+)"', c, re.M).group(1).split() == EVAL_IDS


@needs_pkg
def test_reader_frames_equal_is_tensor_by_tensor(tmp_path):
    H = lora_helpers()
    o = H.v2ep_from_frames(H.smooth_frames(3, (64, 160), seed=1), "c")
    torch.save(o, tmp_path / "a.pt")
    torch.save(copy.deepcopy(o), tmp_path / "b.pt")
    assert RD.frames_equal(tmp_path / "a.pt", tmp_path / "b.pt") == []
    o2 = copy.deepcopy(o)
    o2["jpeg_buf"][5] = (int(o2["jpeg_buf"][5]) + 1) % 256
    torch.save(o2, tmp_path / "c.pt")
    assert RD.frames_equal(tmp_path / "a.pt", tmp_path / "c.pt") == ["tensor jpeg_buf differs"]
    o3 = dict(o)
    o3["clip_id"] = "other"
    torch.save(o3, tmp_path / "d.pt")
    assert any("clip_id" in x for x in RD.frames_equal(tmp_path / "a.pt", tmp_path / "d.pt"))
    del o3["poses"]
    torch.save(o3, tmp_path / "e.pt")
    assert any("keys differ" in x for x in RD.frames_equal(tmp_path / "a.pt", tmp_path / "e.pt"))


@needs_pkg
def test_reader_cli_identity_and_smoke_from_files(tmp_path, capsys):
    d = tmp_path / "id"
    d.mkdir()
    (d / "identity_lora.json").write_text(json.dumps(_ident()))
    (d / "identity_lora_mutant.json").write_text(json.dumps(_mut()))
    s = tmp_path / "sm"
    (s / "run").mkdir(parents=True)
    (s / "smoke_report.json").write_text(json.dumps(_smoke_rep()))
    (s / "smoke_steps.jsonl").write_text("\n".join(json.dumps({"step_wall_s": 4.0}) for _ in range(3)) + "\n")
    (s / "run" / "selfcheck.json").write_text(json.dumps(_sc()))
    argv = ["--identity", str(d), "--smoke", str(s), "--train-clips", *TRAIN_IDS, "--eval-clips", *EVAL_IDS, "--out", str(tmp_path / "r.json")]
    assert RD.main(argv) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0].startswith("identity PASS") and out.strip().endswith("OK")
    assert json.loads((tmp_path / "r.json").read_text())["ok"] is True
    (d / "identity_lora_mutant.json").write_text(json.dumps(_mut(mutant_caught=False)))
    assert RD.main(argv) == 1 and "NOT caught" in capsys.readouterr().out
    (d / "identity_lora.json").unlink()
    assert RD.main(argv) == 2, "a missing file is a refusal, not a traceback"


# =================================================================================================================================================
# SMOKE: the edits to v8f_lora_smoke.py (autocast opt-in, the census on the real clip list, tau_device, reserved memory)
# =================================================================================================================================================
@pytest.fixture
def smoke_env(monkeypatch):
    H = lora_helpers()
    if not SMOKE_P.is_file():
        pytest.skip("the smoke script is not in this checkout")
    pep, _ = E.load_builder()
    monkeypatch.setattr(pep, "GH", 4)
    monkeypatch.setattr(pep, "GW", 10)
    monkeypatch.setattr(H.D, "FRAME_HW", H.TINY_HW)
    monkeypatch.setattr(H.TR, "build_encoder_raw", H._tiny_raw)
    spec = importlib.util.spec_from_file_location("v8f_lora_smoke_ar2_t", SMOKE_P)
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    return H, S


def test_smoke_strip_flag_removes_the_flag_and_its_value_only(tmp_path):
    spec = importlib.util.spec_from_file_location("v8f_lora_smoke_ar2_s", SMOKE_P)
    if not SMOKE_P.is_file():
        pytest.skip("the smoke script is not in this checkout")
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    assert S.strip_flag(["--a", "1", "--max-clips", "4", "--b", "2"], "--max-clips") == ["--a", "1", "--b", "2"]
    assert S.strip_flag(["--max-clips=4", "--b"], "--max-clips") == ["--b"] and S.strip_flag(["--b"], "--max-clips") == ["--b"]
    assert S.strip_flag(["--max-clips", "4"], "--max-clips") == [] and S.strip_flag(["--max-clips-x", "4"], "--max-clips") == ["--max-clips-x", "4"]


def test_smoke_amp_is_off_by_default_the_census_uses_the_whole_clip_list_and_the_report_carries_tau_device(smoke_env, tmp_path):
    H, S = smoke_env
    src = SMOKE_P.read_text(encoding="utf-8")
    assert '*(["--amp"] if a.amp else [])' in src and '"--enc-ckpt", "--amp"' not in src, "the smoke must not append --amp unconditionally"
    frames = H.make_frames_root(tmp_path / "frames", [f"syn-train-{i:03d}" for i in range(6)])
    targv = ["--synthetic", "--d", "32", "--n-q", "2", "--warmup", "2", "--threads", "1", "--kin-mode", "yaw", "--seed", "5", "--arm", "M2b", "--lora-r", "4",
             "--lora-alpha", "4", "--frames-root", str(frames), "--batch", "64", "--max-clips", "3"]
    out = tmp_path / "smoke"
    rc = S.main(["--steps", "3", "--out", str(out), "--warmup", "1", "--census-batches", "2", "--heads-b64-steps", "1", "--", *targv])
    rep = json.load(open(out / "smoke_report.json", encoding="utf-8"))
    assert rc == 0 and rep["amp"] is False and "--amp" not in rep["trainer_argv"] and "--max-clips" in rep["trainer_argv"]
    sc = json.load(open(out / "run" / "selfcheck.json", encoding="utf-8"))
    assert sc["info"]["n_clips"] == 3 and sc["info"]["n_clips_all"] == 6, "the smoke TRAINED on the declared 3 clips"
    pr = rep["projection_ESTIMATED"]
    assert pr["census_scope"]["smoke_trainer_argv_has_max_clips"] is True and pr["census_scope"]["census_and_heads_clips"] == 6 and pr["census_scope"]["census_clips_all"] == 6, \
        "but the PROJECTION's census and heads-at-64 ran on the whole list: 3 clips would share most frames of a batch of 64"
    assert pr["assumes_micro_batch_frames"] == 16 and pr["census"]["n_windows_all_clips"] > 0
    ip = rep["identity_probe_tau_device"]
    assert ip["tau_device"] == 1e-3 and set(ip) >= {"rel_l2", "pass", "device", "scope"} and isinstance(ip["pass"], bool)
    steps = [json.loads(x) for x in open(out / "smoke_steps.jsonl", encoding="utf-8")]
    assert "peak_reserved_gib" in steps[0] and steps[0]["peak_reserved_gib"] is None, "None on a CPU run, a number on box 2"
    assert "peak_reserved_gib_max" in rep["measured_batch1"]
    out2 = tmp_path / "smoke2"
    S.main(["--steps", "3", "--out", str(out2), "--warmup", "1", "--no-projection", "--amp", "--", *targv])
    rep2 = json.load(open(out2 / "smoke_report.json", encoding="utf-8"))
    assert rep2["amp"] is True and "--amp" in rep2["trainer_argv"], "--amp is opt-in and recorded"


# =================================================================================================================================================
# QUEUE: identitylora / smokelora
# =================================================================================================================================================
LSTAGES = ("identitylora", "smokelora")
# every label of the queue as it stood at tip 99d59527 (before this edit), with the sha256[:16] of its text ("\n" + the lines up to the next label): the edit may ADD labels and extend
# the usage line, never touch these
PRE_EXISTING = {
    "d_train": "653a25ac6e43be79", "d_eval": "b64a718bcf22e4ca", "prep": "8f8749398e785a74", "gate": "01fa4b4d415d3057", "train": "c3867e94dae8b06b", "eval": "8d3f96a2ec519c6d", "tac": "8357d58b9565d87f",
    "tacone": "d05a9be75c4de2bd", "tacgrp": "7cae083fcfb5a237", "tacgone": "cdbb6e94327d04af", "tacgrp2": "efb5d1a2944b99fa", "tacgtwo": "14ccef2e21c86f83", "tacabs": "60aa7130ee682938",
    "tacabsarm": "0ed52c566b9de578", "gatewc": "487c836142548e93", "trainwc": "5565fa65d285f47f", "evalwc": "ccb014965cc92aa5", "gatepgt": "3ca89871f8da1c43", "trainpgt": "49fe41ddd1622379",
    "evalpgt": "a04620ebdcfbb068", "evalpgt_1": "effe5bbee55d4771", "evalpgt_2": "33cdd72c074d8c6a", "evalpgt_3": "a7e30c81d582343c", "evalpgt_4": "22636d9d007850fa", "gridbase": "56299fbe24cec6f7",
    "gategrid": "e3250ab756539587", "traingrid": "87cf143655b55573", "evalgrid": "399406480951bc1d", "evalgrid_1": "74aaef5e39bbaca5", "evalgrid_2": "f242aa25ecd51bd6", "evalgrid_3": "210db6bc5c411a0c",
    "evalgrid_4": "22636d9d007850fa", "gatehist": "93a341e35cc9a0b1", "trainhist": "344fcc45b1624ead", "evalhist": "413108b37c6fe068", "evalhist_1": "03081f7da50e8739", "evalhist_2": "168db150abcf3ce1",
    "evalhist_3": "d9d586a4304f2d24", "evalhist_4": "22636d9d007850fa", "gatedscale": "0d91b533582f0ea6", "traindscale": "7c5f99d4e8e0c096", "evaldscale": "080e32b4ee8c2ac6",
    "evaldscale_1": "f59695b83d08d77a", "evaldscale_2": "813ee4390fc783ce", "evaldscale_3": "fdba053d0afad97d", "evaldscale_4": "22636d9d007850fa", "gatearm": "b79d2a733b86fe61",
    "floors": "2b7f345ee3e86bcc", "compare": "b41e1a1c859c59c3"}
OLD_STAGES = ("gate", "train", "eval", "floors", "compare", "tac", "tacgrp", "tacgrp2", "tacabs", "gatearm", "tacabsarm", "gatewc", "trainwc", "evalwc", "gatepgt", "trainpgt", "evalpgt",
              "gategrid", "traingrid", "evalgrid", "gatehist", "trainhist", "evalhist", "gatedscale", "traindscale", "evaldscale")


def _queue_text():
    return QUEUE.read_text(encoding="utf-8").replace("\r\n", "\n")


def _blocks():
    parts = re.split(r"(?m)^(:[A-Za-z0-9_]+)$", _queue_text())
    return {parts[i][1:]: parts[i + 1] for i in range(1, len(parts), 2)}


def _block(label):
    return _queue_text().split(f"\n:{label}\n", 1)[1].split("\nexit /b 0\n", 1)[0].split("\n")


def _code(lines):
    return [x for x in lines if x.strip() and not x.lower().startswith("rem ")]


def _argv_of(line):
    return [t.strip('"') for t in re.findall(r'"[^"]*"|\S+', line.strip())]


def _dry(args, env_extra=None, clean_paths=False, cwd=None):
    env = dict(os.environ, V8F_DRYRUN="1")
    if clean_paths:
        for k in ("REPO", "RUNSM", "CACHEFULL", "V9", "PY", "LORAFRAMES"):
            env.pop(k, None)
    env.update(env_extra or {})
    r = subprocess.run(f'cmd /c ""{QUEUE}" {args}"', env=env, capture_output=True, text=True, timeout=60, encoding="utf-8", cwd=cwd)
    return r.returncode, [x for x in r.stdout.splitlines() if ".py" in x]


@needs_queue
def test_queue_ar2_the_queue_is_crlf_the_stages_are_dispatched_and_the_usage_line_names_them():
    raw = QUEUE.read_bytes()
    assert raw.count(b"\r\n") == raw.count(b"\n") > 600, "a .cmd with bare LF endings misparses labels and `exit /b`"
    assert raw.count(b"\r") == raw.count(b"\r\n") and raw.endswith(b"\r\n"), "every line CRLF, no bare CR"
    q = _queue_text()
    for st in LSTAGES:
        assert f'if /i "%~1"=="{st}" goto :{st}' in q, "a stage nobody dispatches never runs"
        assert q.count(f"\n:{st}\n") == 1
    assert q.count("\n:loraenv\n") == 1
    usage = next(x for x in q.split("\n") if x.startswith("echo usage:"))
    assert usage.endswith("^|gatedscale^|traindscale SEED^|evaldscale SEED^|identitylora^|smokelora"), "the new stages are appended; DSCALE's usage text is intact"


@needs_queue
def test_queue_ar2_every_existing_stage_is_byte_identical():
    """The edit only ADDS labels: the text of every label that existed before hashes as it did, the dispatch of each old stage is still there, and none mentions the new stages."""
    blocks, q = _blocks(), _queue_text()
    for label, h in PRE_EXISTING.items():
        assert label in blocks, f"the stage {label} was removed"
        assert hashlib.sha256(("\n" + blocks[label]).encode()).hexdigest()[:16] == h, f"the stage {label} was edited"
        for new in ("lora", "LORA", "identity", "smokelora"):
            assert new not in blocks[label], (label, new)
    # PREREG_V8F_KIND later added its own four labels (test_v8f_kind.py pins them and the byte-identity of every label here); this edit's three are still the only others
    mixo = {"gatemixo", "trainmixo", "evalmixo"} | {f"evalmixo_{i}" for i in range(1, 8)}      # PREREG_V8F_MIXO appended these (test_v8f_mixo.py pins them); widened here, nothing else
    flat = {"gateflat", "trainflat", "evalflat"} | {f"evalflat_{i}" for i in range(1, 5)}      # PREREG_V8F_FLAT appended these (test_v8f_flat.py pins them); widened here, nothing else
    oacc = {"waccoacc", "oaccwacc", "gateoacc", "trainoacc", "evaloacc", "fitoacc"}          # PREREG_V8F_OACC appended these (test_v8f_oacc.py pins them); widened here, nothing else
    orc = {"orcarm", "gateorc", "trainorc", "evalorc"}          # PREREG_V8F_ORECOVER appended these (test_v8f_orc_chain.py pins them); widened here, nothing else
    dag = {"dagname", "prepdag", "dagarm", "dagarm_dry", "dagarm_have", "bankdag", "gatedag", "traindag", "evaldag", "dag2arm", "dag2arm_dry", "dag2arm_have", "bankdag2"}      # PREREG_V8F_DAGGER / PREREG_V8F_DAGGER2 appended these after ORC's (widened here, nothing else)
    dag |= {"dagoaccarm", "dagoaccarm_dry", "dagoaccarm_have", "dagoaccwacc", "dagoaccwacc_dry", "dagoaccwacc_have", "waccdag"}      # PREREG_V8F_DAG_OACC, after DAGGER2's
    dag |= {"flatdagarm", "flatdagarm_dry", "flatdagarm_have", "evalflatdag", "evalflatdag_1", "evalflatdag_2", "bankflat"}      # PREREG_V8F_FLATDAG, after DAG-OACC's
    dag |= {"dagorcarm"}      # PREREG_V8F_DAG_ORC, after FLAT-DAG's
    assert set(blocks) - {"kind", "kindseed", "kindown", "kindorc"} - mixo - flat - oacc - orc - dag == set(PRE_EXISTING) | {"loraenv", "identitylora", "smokelora"}, "the only new labels are the three of this edit"
    for st in OLD_STAGES:
        assert f'if /i "%~1"=="{st}" goto :' in q, st
    labels = set(blocks)
    disp = re.findall(r'(?m)^if /i "%~1"=="(\w+)" goto :(\w+)$', q)
    assert len(disp) >= len(OLD_STAGES) + len(LSTAGES) and {a for a, _ in disp} >= set(OLD_STAGES) | set(LSTAGES)
    assert [(a, b) for a, b in disp if b not in labels] == [], "a dispatch line whose target label does not exist (the stage would die with 'label not found')"
    assert all(a == b or a in ("gate", "train", "eval") or True for a, b in disp)


@needs_queue
def test_queue_ar2_runs_the_command_it_prints_static_and_no_ampersand_enters_a_command_string():
    """A dry run can only print: the real line is checked statically -- the driver / smoke appear ONCE per command, inside the one string, and the real run executes that string."""
    code = _code(_block("identitylora"))
    ids = [x for x in code if x.startswith('set "ID')]
    assert len(ids) == 2 and [x for x in code if "%LORAID%" in x] == ids, "the driver appears only inside the two strings"
    assert sum(x.strip().startswith("%ID0% > ") for x in code) == 1 and sum(x.strip().startswith("%ID1% > ") for x in code) == 1, "each string is executed exactly once"
    assert "if defined V8F_DRYRUN (echo %ID0% & echo %ID1% & exit /b 0)" in code
    assert ids[0].count("--mutant") == 0 and ids[1].count("--mutant") == 1 and _argv_of(ids[0].replace('set "ID0=', "")) != _argv_of(ids[1].replace('set "ID1=', ""))
    assert ids[1].replace(" --mutant", "").replace("ID1", "ID0").replace("identity_lora_mutant", "identity_lora") == ids[0], "the mutant run is the control with --mutant added and nothing else"
    code = _code(_block("smokelora"))
    sm = [x for x in code if x.startswith('set "SM=')]
    assert len(sm) == 1 and [x for x in code if "%LORASMOKE%" in x] == sm and sum(x.strip().startswith("%SM% > ") for x in code) == 1
    assert "if defined V8F_DRYRUN (echo %SM% & exit /b 0)" in code
    for x in ids + sm:
        assert "&" not in x and "Architecture" not in x, "the repo path carries an ampersand: it must never enter a command string (it splits the line, MEASURED)"
    loraenv = "\n".join(_code(_block("loraenv")))
    assert 'set "LORAID=v8f_lora_identity_dev.py"' in loraenv and 'set "LORASMOKE=v8f_lora_smoke.py"' in loraenv, "named relative to the code directory mk_v8f_job.py runs the queue from"
    assert "if not exist \"%LORAID%\"" in loraenv and "if defined V8F_DRYRUN exit /b 0" in loraenv
    for label in LSTAGES:
        body = _block(label)
        assert not any(x.rstrip().endswith("(") for x in body), f"{label}: a multi-line parenthesised block (an `exit /b 1` inside one returns rc 0 from cmd /c, MEASURED by the PGT build)"
    assert "--amp" not in "\n".join(code), "Amendment 2 pins autocast OFF: no --amp in the smoke stage"
    assert 'if exist "%OUT%\\smoke_report.json" (echo [skip] smokelora & exit /b 0)' in code and any("REFUSED: %OUT% exists without a smoke_report.json" in x for x in code)


def _without_flag(line, drop):
    t, out, i = [x for x in re.findall(r'"[^"]*"|\S+', line)], [], 0
    while i < len(t):
        if t[i].strip('"') in drop:
            i += 1
            continue
        out.append(t[i])
        i += 1
    return " ".join(out)


@needs_queue
@win
def test_queue_ar2_stages_dry_the_declared_clips_the_controls_and_B_s_argv(tmp_path):
    env = dict(REPO=str(REPO), RUNSM=str(tmp_path / "runsm"), CACHEFULL=str(tmp_path / "cf"), V9=str(tmp_path / "v9"), LORAFRAMES=str(tmp_path / "fr"))
    rc, ln = _dry("identitylora", env)
    assert rc == 0 and len(ln) == 2, ln
    a0, a1 = _argv_of(ln[0]), _argv_of(ln[1])
    assert a0[1] == a1[1] == "v8f_lora_identity_dev.py", "named relative to the code directory"
    assert a0[a0.index("--train-clips") + 1:a0.index("--train-clips") + 5] == TRAIN_IDS and a0[a0.index("--eval-clips") + 1:a0.index("--eval-clips") + 3] == EVAL_IDS
    assert a0[a0.index("--cache-root") + 1].endswith("cf") and a0[a0.index("--device") + 1] == "cuda" and "--mutant" not in a0 and a1.count("--mutant") == 1
    assert a0[a0.index("--train-frames") + 1] == str(tmp_path / "fr" / "train") and a0[a0.index("--eval-frames") + 1] == str(tmp_path / "fr" / "eval")
    assert a0[a0.index("--out") + 1].endswith("identity_lora.json") and a1[a1.index("--out") + 1].endswith("identity_lora_mutant.json")
    assert "--allow-non-cuda" not in ln[0] and "--frames-per-clip" not in ln[0] and "--manifest-only" not in ln[0], "the qualifying reading: cuda, every frame"
    rc, sm = _dry("smokelora", env)
    assert rc == 0 and len(sm) == 1, sm
    t = _argv_of(sm[0])
    assert t[1] == "v8f_lora_smoke.py" and t[t.index("--steps") + 1] == "200" and t.count("--max-clips") == 1 and t[t.index("--max-clips") + 1] == "4" == str(len(TRAIN_IDS))
    assert t[t.index("--frames-root") + 1] == str(tmp_path / "fr" / "train") and "--amp" not in t and "--out" not in t[t.index("--"):], "the smoke owns --out; no --amp"
    assert t[t.index("--arm") + 1] == "M2b" and t[t.index("--seed") + 1] == "0" and t[t.index("--grid") + 1] == "pool4x10" and t.count("--exclude") == 1 and t[t.index("--exclude") + 1] == "540:600"
    # the trainer argv is B's: everything after --frames-root's value is the train stage's `%A%`
    b_train = _argv_of(_dry("train M2b 0", env)[1][0])
    tail_b = b_train[b_train.index("--out") + 2:]
    tail_s = t[t.index("--frames-root") + 2:]
    assert tail_s == tail_b and "--device" in tail_s and "--batch" in tail_s and tail_s[tail_s.index("--batch") + 1] == "64"
    for no in ("--steps-mult 3", "--w-cons", "--p-gt", "--data-frac", "--hist", "--hmax", "--window-start", "--lora"):
        assert no not in sm[0], f"the smoke line carries only B's flags + --max-clips / --frames-root: {no}"
    for args in ("train M2b 0", "traindscale 0", "gatedscale", "evaldscale 0", "trainhist 0"):
        rc, other = _dry(args, env)
        assert rc == 0 and other and all("v8f_lora" not in x.lower() and "identity_dev" not in x.lower() for x in other), args


@needs_queue
@win
def test_queue_ar2_default_paths_are_box_2_s(tmp_path):
    rc, ln = _dry("identitylora", clean_paths=True)
    assert rc == 0 and len(ln) == 2
    assert 'D:\\tanitad\\data\\lora_ar2_frames\\train' in ln[0] and '"D:\\tanitad\\cache_full"' in ln[0] and 'D:\\tanitad\\runs\\v8f_mini_full\\lora_ar2\\identity\\identity_lora.json' in ln[0]
    rc, sm = _dry("smokelora", clean_paths=True)
    assert rc == 0 and 'D:\\tanitad\\runs\\v8f_mini_full\\lora_ar2\\smoke' in sm[0] and "--cache-dir D:\\tanitad\\cache_full" in sm[0]


@pytest.fixture
def qtmp(tmp_path):
    """The fake tree mirrors the queue's long repo-relative path (HJRAW), and Windows refuses a directory over 248 characters: under a long --basetemp the tree goes to the system temp."""
    if len(str(tmp_path)) <= 60:
        yield tmp_path
        return
    d = Path(tempfile.mkdtemp(prefix="ar2q"))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _fake_tree(tmp):
    """A repo whose scripts are FAKES that record the argv they are handed (and write the files the next step waits for); the fake CODE directory is the queue's cwd."""
    repo = tmp / "repo"
    code = tmp / "code"
    code.mkdir(parents=True)
    rec = tmp / "argv.jsonl"
    (code / "v8f_lora_identity_dev.py").write_text(
        "import json, os, sys\n"
        f"open({str(rec)!r}, 'a').write(json.dumps({{'who': 'id', 'argv': sys.argv[1:]}}) + '\\n')\n"
        "a = sys.argv; out = a[a.index('--out') + 1]\n"
        "if os.environ.get('FAKE_ID') == 'crash':\n"
        "    sys.exit(1)\n"
        "if os.environ.get('FAKE_ID') == 'crash_mutant' and '--mutant' in a:\n"
        "    sys.exit(1)\n"
        "os.makedirs(os.path.dirname(out), exist_ok=True)\n"
        "open(out, 'w').write(json.dumps({'verdict': 'FAIL' if os.environ.get('FAKE_ID') == 'fail' else 'PASS'}))\n"
        "sys.exit(1 if os.environ.get('FAKE_ID') == 'fail' else 0)\n", encoding="utf-8")
    (code / "v8f_lora_smoke.py").write_text(
        "import json, os, sys\n"
        f"open({str(rec)!r}, 'a').write(json.dumps({{'who': 'smoke', 'argv': sys.argv[1:]}}) + '\\n')\n"
        "a = sys.argv; out = a[a.index('--out') + 1]\n"
        "os.makedirs(out, exist_ok=True)\n"
        "if os.environ.get('FAKE_SMOKE') != 'nowrite':\n"
        "    open(os.path.join(out, 'smoke_report.json'), 'w').write('{}')\n"
        "sys.exit(1 if os.environ.get('FAKE_SMOKE') == 'failverdict' else 0)\n", encoding="utf-8")
    hj = repo / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-refav1-hj-mini" / "raw" / "p10_yaw4512"
    hj.mkdir(parents=True)
    (hj / "p10_prior.pt").write_bytes(b"p")
    (hj / "p10_eval_pred6.npy").write_bytes(b"p")
    cf = tmp / "cf"
    for sub in ("train", "eval"):
        (cf / sub).mkdir(parents=True)
        (cf / sub / "manifest.jsonl").write_text("{}\n", encoding="utf-8")
    (cf / "native_tracks.pt").write_bytes(b"t")
    (tmp / "runsm").mkdir()
    return repo, code, rec


@needs_queue
@win
def test_queue_ar2_real_lines_execute_exactly_the_printed_commands_and_notice_failures(qtmp):
    repo, code, rec = _fake_tree(qtmp)
    runsm = qtmp / "runsm"
    base = dict(os.environ, REPO=str(repo), PY=sys.executable, RUNSM=str(runsm), CACHEFULL=str(qtmp / "cf"), V9=str(qtmp / "v9"), LORAFRAMES=str(qtmp / "fr"))
    base.pop("V8F_DRYRUN", None)

    def stage(args, cwd=code, **extra):
        r = subprocess.run(f'cmd /c ""{QUEUE}" {args}"', env=dict(base, **extra), capture_output=True, text=True, timeout=120, encoding="utf-8", cwd=cwd)
        return r.returncode, r.stdout

    def calls():
        return [json.loads(x) for x in rec.read_text(encoding="utf-8").splitlines()] if rec.exists() else []

    def printed(args):
        r = subprocess.run(f'cmd /c ""{QUEUE}" {args}"', env=dict(base, V8F_DRYRUN="1"), capture_output=True, text=True, timeout=60, encoding="utf-8", cwd=code)
        return [_argv_of(x)[2:] for x in r.stdout.splitlines() if ".py" in x]
    want_id, want_sm = printed("identitylora"), printed("smokelora")
    assert len(want_id) == 2 and len(want_sm) == 1
    # the stages refuse to run from anywhere but the code directory (the scripts are named relative to it)
    rc, out = stage("identitylora", cwd=qtmp)
    assert rc == 2 and "REFUSED: v8f_lora_identity_dev.py is not in the current directory" in out and not calls()
    # a driver that crashes writes no json => [FAIL]
    rc, out = stage("identitylora", FAKE_ID="crash")
    assert rc == 1 and "[FAIL] identitylora wrote no json" in out
    rc, out = stage("identitylora", FAKE_ID="crash_mutant")
    assert rc == 1 and "[FAIL] identitylora mutant wrote no json" in out
    # a driver that FAILS the control still wrote its reading: that is a READING (the chain judges it), not a stage failure
    rc, out = stage("identitylora", FAKE_ID="fail")
    assert rc == 0 and "[ok] identitylora reading written" in out and "[ok] identitylora mutant reading written" in out
    idd = runsm / "lora_ar2" / "identity"
    assert json.loads((idd / "identity_lora.json").read_text())["verdict"] == "FAIL" and (idd / "identity_lora_mutant.json").exists()
    rec.write_text("", encoding="utf-8")
    # a finished stage is skipped; with one json missing it re-runs both (the stale one is deleted first)
    rc, out = stage("identitylora")
    assert rc == 0 and "[skip] identitylora" in out and not calls()
    (idd / "identity_lora_mutant.json").unlink()
    rc, out = stage("identitylora")
    assert rc == 0 and [c["argv"] for c in calls()] == want_id and [c["who"] for c in calls()] == ["id", "id"], "the REAL lines hand the driver the printed commands, control first"
    assert json.loads((idd / "identity_lora.json").read_text())["verdict"] == "PASS", "the stale reading was replaced"
    rec.write_text("", encoding="utf-8")
    # the smoke
    rc, out = stage("smokelora", FAKE_SMOKE="nowrite")
    assert rc == 1 and "[FAIL] smokelora wrote no smoke_report.json" in out
    rc, out = stage("smokelora")
    assert rc == 2 and "REFUSED:" in out and "without a smoke_report.json" in out, "the partial run dir from the failed attempt is refused, not resumed"
    shutil.rmtree(runsm / "lora_ar2" / "smoke")
    rec.write_text("", encoding="utf-8")
    rc, out = stage("smokelora", FAKE_SMOKE="failverdict")
    assert rc == 0 and "[ok] smokelora report written" in out, "a smoke whose verdict failed still wrote its report: the CHAIN reads it"
    assert [c["argv"] for c in calls()] == want_sm and calls()[0]["who"] == "smoke", "the REAL smokelora line hands the smoke the printed command"
    a = calls()[0]["argv"]
    assert a[a.index("--max-clips") + 1] == "4" and "--amp" not in a
    rc, out = stage("smokelora")
    assert rc == 0 and "[skip] smokelora" in out and len(calls()) == 1


# =================================================================================================================================================
# CHAIN: chain_v8f_lora_ar2.sh (needs box 2 and ssh, so it cannot RUN here: its order, its guards and its --dry-run are pinned)
# =================================================================================================================================================
def _git_bash():
    for p in (os.environ.get("GIT_BASH"), "C:/Program Files/Git/bin/bash.exe", "C:/Program Files (x86)/Git/bin/bash.exe"):
        if p and Path(p).exists():
            return p
    return None


@needs_chain
def test_chain_ar2_order_predecessor_and_what_it_never_does():
    c = CHAIN.read_text(encoding="utf-8")
    assert b"\r\n" not in CHAIN.read_bytes(), "a bash script with CRLF endings does not run"
    code = "\n".join(x for x in c.splitlines() if not x.lstrip().startswith("#"))
    assert "AFTER_OUT=${AFTER_OUT:-$L/chain_v8f_dscale.out}" in code and "AFTER_DONE=${AFTER_DONE:-ZZV8FMINI-DONE}" in code and "AFTER_STOP=${AFTER_STOP:-ZZV8FMINI-STOP}" in code
    assert code.count("queue identitylora ") == 1 and code.count("queue smokelora ") == 1 and "queue train" not in code and "traindscale" not in code and "trainhist" not in code,         "instruments only: no training stage"
    amp = [x.strip() for x in code.splitlines() if "--amp" in x]
    assert len(amp) == 2 and all("grep" in x and "stop " in x for x in amp), f"--amp appears only inside the two guard greps that STOP when it is present: {amp}"
    assert "ssh tanitad-box2" not in code.split("--dry-run")[1].split("exit 0\nfi")[0], "the dry run touches no box"
    order = ["waiting for the predecessor chain", "BOX2_BUSY.txt", "fetch -q origin", "archive --format=tar", 'stop "box 2 $f != tip $T"', "pull box 2's cache train manifest", "--check-declared",
             "--manifest-only", "devbuild_cmd \"$(cygpath -m $X)\" \"$STAGE_TRAIN\"", "--compare-frames", "ship train frames", "launch telemetry_long", "v8f_guard.sh > ", "queue identitylora ", "--identity \"$EM/identity\" --train-clips",
             "queue smokelora ", "--smoke \"$EM/smoke\"", 'finish "$S1"']
    real = c[c.index('waiting for the predecessor chain ($AFTER_OUT)'):]                 # past the --dry-run block, which prints the same commands in its own order
    idx = [real.index(x) for x in order]
    assert idx == sorted(idx) and len(set(idx)) == len(idx), dict(zip(order, idx))
    # the identity control is judged BEFORE the smoke is launched: a FAIL (or a mutant that was not caught) ends the chain with the numbers
    i_read = c.index('--identity "$EM/identity" --train-clips')
    assert i_read < c.index("queue smokelora ") and "|| stop \"the same-device identity control did not PASS" in c[i_read:c.index("queue smokelora ")]
    assert "--expect-driver-sha $DRV_SHA" in c and "DRV_SHA=$(sha256sum" in c
    # frames: the PRODUCTION builder, from the tip archive, checked against the banked frames, <= 8 clips
    assert "pod_build_b1_epcache.py --out $3 --root $2 --workers 4 --clips $4" in c and "TANITAD_STACK=$1/stack" in c and "CUDA_VISIBLE_DEVICES=-1" in c
    assert len((re.search(r'^TRAIN_CLIPS="([^"]+)"', c, re.M).group(1) + " " + re.search(r'^EVAL_CLIPS="([^"]+)"', c, re.M).group(1)).split()) == 6 <= 8
    assert "--banked \"$BANKED_TRAIN\" \"$BANKED_EVAL\"" in c and 'stop "the rebuilt frames differ from the banked production frames (compare-frames)"' in c
    assert 'rsha "$RFR/train/$c.v2ep.pt" SHA256' in c and 'rsha "$RFR/eval/$c.v2ep.pt" SHA256' in c, "every shipped frame file is sha256-checked on the far side"
    assert 'stop "the tip\'s smoke script still appends --amp (Amendment 2 pins it OFF)"' in c
    assert 'grep -c -- "--max-clips $N_TRAIN "' in c and 'stop "the queue\'s smokelora line carries --amp' in c
    assert "logs/v8f_lora_ar2" not in code and "E=$L/v8f_lora_ar2" in c, "outputs are pulled into logs/v8f_lora_ar2/"
    for f in ("smoke_report.json", "smoke_steps.jsonl", "run/selfcheck.json", "identity_lora.json", "identity_lora_mutant.json"):
        assert f in c, f
    assert 'grep ZZGUARD-STOP $L/v8floraar2_guard.out 2>/dev/null && stop "guard stop"' in c and c.count('stop "guard stop"') == 2


@needs_chain
@pytest.mark.skipif(not CHAIN_DSCALE.exists(), reason="chain_v8f_dscale.sh is not in this checkout")
def test_chain_ar2_keeps_dscale_s_guards_unchanged():
    """Every guard line of the DSCALE chain that carries no DSCALE-specific name is in this chain verbatim: the busy marker, the boot check, the temperature guard's start and its
    freshness test, the STOP / DONE markers, waitjob and the wait's stop semantics, the polling of the predecessor. (The two POLL constants are the training wait's and are sized
    to the training; this chain's jobs are minutes long and carry their own.)"""
    d, c = CHAIN_DSCALE.read_text(encoding="utf-8").splitlines(), CHAIN.read_text(encoding="utf-8")
    keys = ("BOX2_BUSY", "v8f_guard.sh", "fresh telemetry", "ZZV8FMINI-STOP", "ZZV8FMINI-DONE", "waitjob.sh", "ZZGUARD-STOP", "ZZJOB-END", "telemetry", "BOOT0", "sleep 300")
    guard_lines = [x.strip() for x in d if not x.lstrip().startswith("#") and any(k in x for k in keys) and not any(w in x.lower() for w in ("dscale", "hist", "grid", "chain_v8f", "q gate"))]
    assert len(guard_lines) >= 15, len(guard_lines)
    missing = [x for x in guard_lines if x not in c]
    assert not missing, f"DSCALE's guard lines missing from the AR-2 chain: {missing}"
    for x in ("boot() {", "launch() {", "clean() {", "rsha() {", "stop() {"):
        da = next(l for l in d if l.startswith(x))
        assert da in c.splitlines(), f"{x} differs from DSCALE's"
    # the waitrc function: DSCALE's text with the guard log's name changed
    w = re.search(r"\nwaitrc\(\) \{.*?\n\}\n", CHAIN_DSCALE.read_text(encoding="utf-8"), re.S).group(0).replace("v8fdscale_guard", "v8floraar2_guard")
    assert w in c, "waitrc() is DSCALE's, with this chain's guard log"
    g = GUARD.read_text(encoding="utf-8") if GUARD.exists() else ""
    assert not g or "ZZV8FMINI" in g and "(DONE|STOP)" in g, "the shared guard already knows this chain's markers"


@needs_chain
def test_chain_ar2_finish_releases_the_box_before_it_announces_done():
    c = CHAIN.read_text(encoding="utf-8")
    m = re.search(r"\nfinish\(\) \{.*?\n\}\n", c, re.S)
    assert m, "no finish() function in the chain"
    body = m.group(0)
    for need in ("telemetry.stop", "rm -f $B/BOX2_BUSY.txt", 'echo "" > $CUR', "ZZV8FMINI-DONE", "exit 0"):
        assert need in body, f"finish() lost {need!r}"
    assert body.index("rm -f $B/BOX2_BUSY.txt") < body.index("ZZV8FMINI-DONE") < body.index("exit 0")
    assert 'stop() { echo "ZZV8FMINI-STOP $*"; echo "" > $CUR; exit 1; }' in c


@needs_chain
@needs_queue
def test_chain_ar2_syntax_and_dry_run_prints_every_command(tmp_path):
    bash = _git_bash()
    if bash is None or sys.platform != "win32":
        pytest.skip("Git Bash on Windows is needed to run the chain's --dry-run")
    r = subprocess.run([bash, "-n", str(CHAIN)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert r.returncode == 0, r.stderr
    env = dict(os.environ, V8F_QUEUE=str(QUEUE), TMPDIR=tmp_path.as_posix())
    env.pop("V8F_DRYRUN", None)
    r = subprocess.run([bash, str(CHAIN), "--dry-run"], env=env, capture_output=True, text=True, timeout=180, encoding="utf-8", stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stderr
    out = r.stdout.splitlines()
    heads = [x for x in out if x.startswith("== ")]
    assert heads[0].startswith("== dev-side build, train (4 clips)") and heads[1].startswith("== dev-side build, eval (2 clips)")
    pos = {h: out.index(h) for h in heads}
    assert pos["== identitylora"] < pos["== smokelora"], "the identity control, THEN the smoke"
    assert [h for h in heads if h in ("== identitylora", "== smokelora")] == ["== identitylora", "== smokelora"]
    # the queue's own lines, at box 2's default paths, are what the chain prints for its two stages
    want = []
    for st in LSTAGES:
        want += _dry(st, clean_paths=True)[1]
    got = [x for x in out if "v8f_lora_identity_dev.py" in x and x.startswith('"D:') or "v8f_lora_smoke.py" in x and x.startswith('"D:')]
    assert got == want and len(want) == 3
    # the dev-side builder is the PRODUCTION builder, CPU only, one command per split, on the declared ids
    builds = [x for x in out if "pod_build_b1_epcache.py" in x]
    assert len(builds) == 2 and all("CUDA_VISIBLE_DEVICES=-1" in b and "--workers 4" in b and "/stack/scripts/pod_build_b1_epcache.py" in b for b in builds)
    assert "refav1-train600-256_stage" in builds[0] and "refav1-eval141-256rb_stage" in builds[1]
    # every declared clip is shipped, once, to the far side's frames dir
    ships = [x for x in out if x.startswith("scp ")]
    assert len(ships) == 6 and len({s.split()[-2] for s in ships}) == 6 and all("tanitad-box2:D:/tanitad/data/lora_ar2_frames/" in s for s in ships)
    for cid in TRAIN_IDS + EVAL_IDS:
        assert sum(cid in s for s in ships) == 1
    reads = [x for x in out if "v8f_lora_ar2_read.py" in x or "v8f_lora_identity_dev.py\"" in x]
    assert any("--compare-frames" in x for x in reads) and any("--check-declared" in x for x in reads) and any("--manifest-only" in x for x in reads)
    assert sum("--identity" in x for x in reads) == 2 and any("--smoke" in x for x in reads)
    assert all(x.split(" ")[0] == "PYTHONIOENCODING=utf-8" and '"<TIPDIR>/TanitAD Research Lab/Architecture & Inference/' in x for x in reads), "the script path is QUOTED (spaces, ampersand)"
    assert "== pull identity: identity_lora.json identity_lora_mutant.json identity_lora.log identity_lora_mutant.log" in out
    assert any(x.startswith("== pull smoke: smoke_report.json smoke_steps.jsonl") and "run/selfcheck.json" in x for x in out)
    assert not any("train " in x and "v8f_mini_train.py" in x and "--selfcheck-only" not in x and "v8f_lora_smoke" not in x for x in out), "no training stage is printed"


@needs_chain
def test_chain_ar2_load_bearing_lines_are_present():
    """Every line whose loss would let a wrong run through is pinned. Substrings, not whole lines: line ends are not load-bearing."""
    c = CHAIN.read_text(encoding="utf-8")
    lines = (
        r"""for st in identitylora smokelora loraenv; do grep -q "^:$st\$" $QT || stop "the tip's queue has no $st stage"; done""",
        r"""git -C $MIRROR show "$T:stack/scripts/v8f_mini_train.py" | grep -q -- '"--encoder"' || stop "the tip's trainer has no --encoder flag""",
        r"""grep -q -- '"--frames-root"' || stop "the tip's trainer has no --frames-root flag""",
        r"""grep -q -- '"--max-clips"' || stop "the tip's trainer has no --max-clips flag""",
        r"""grep -q -- '"--amp", action="store_true"' || stop "the tip's smoke script still appends --amp (Amendment 2 pins it OFF)""",
        r"""git -C $MIRROR cat-file -e "$T:$B1" || stop "the tip has no B1 lead block""",
        r"""stack/tanitad/models/v8f_lora_encoder.py stack/scripts/dinov3_fp8_encode_ship.py "$HJ/code/pull_encode_pooled.py" """,
        r"""[ "$l" = "$(rsha "D:/tanitad/repo/$f" SHA256)" ] || stop "box 2 $f != tip $T""",
        r"""$SCP tanitad-box2:D:/tanitad/cache_full/train/manifest.jsonl $E/cache/train/manifest.jsonl || stop "pull box 2's cache train manifest""",
        r"""$SCP tanitad-box2:D:/tanitad/cache_full/eval/manifest.jsonl $E/cache/eval/manifest.jsonl || stop "pull box 2's cache eval manifest""",
        r"""|| stop "the declared clips are not the manifest's first (check-declared)""",
        r"""|| stop "a declared manifest row is not a production row (the driver's --manifest-only)""",
        r"""for c in $TRAIN_CLIPS $EVAL_CLIPS; do echo "$ILINES" | grep -q -- "$c" || stop "the queue's identitylora line does not carry $c"; done""",
        r"""echo "$SLINE" | grep -q -- "--exclude 540:600" || stop "the queue's smokelora line lacks B's --exclude 540:600""",
        r"""|| { tail -5 $FR/build_train.log; stop "the production builder failed (train)"; }""",
        r"""|| { tail -5 $FR/build_eval.log; stop "the production builder failed (eval)"; }""",
        r"""[ "$(sha256sum $FR/train/$c.v2ep.pt | cut -c1-64)" = "$(rsha "$RFR/train/$c.v2ep.pt" SHA256)" ] || stop "box 2's train frames $c != the local build (sha256)""",
        r"""[ "$(sha256sum $FR/eval/$c.v2ep.pt | cut -c1-64)" = "$(rsha "$RFR/eval/$c.v2ep.pt" SHA256)" ] || stop "box 2's eval frames $c != the local build (sha256)""",
        r"""[[ $t2 == 20* && $t2 != "$t1" ]] || stop "no fresh telemetry""",
        r"""grep -qF "'^(ZZV8FMINI|" $B/v8f_guard.sh || stop "v8f_guard.sh does not know this chain's markers""",
        r"""V8F_CHAIN_OUT=$L/chain_v8f_lora_ar2.out V8F_CURRENT=$CUR bash $B/v8f_guard.sh""",
        r"""CUR=$L/v8floraar2_current.txt""",
        r"""[ -f $B/BOX2_BUSY.txt ] && { echo "ZZV8FMINI-STOP busy marker present""",
        r"""until grep -q "^$AFTER_DONE" "$AFTER_OUT" 2>/dev/null; do""",
        r"""grep -q "^$AFTER_STOP" "$AFTER_OUT" 2>/dev/null && { echo "ZZV8FMINI-STOP the predecessor stopped""",
        r"""BOOT0=$(boot); [ -n "$BOOT0" ] || stop "box 2 unreachable""",
        r"""|| { cat $E/ar2_read.out; stop "the AR-2 smoke did not pass its checks: see $E/ar2_read.json"; }""",
        r"""HOLDOUT_SHA=f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9""",
        r"""echo "ZZV8FMINI-DONE $(date '+%m-%d %H:%M') tip $T: $1 (boot unchanged $BOOT0)""")
    for ln in lines:
        assert ln in c, f"missing from the chain: {ln!r}"


@needs_queue
def test_the_guard_can_kill_every_python_script_the_ar2_stages_launch():
    """Box 2 is the PI's machine: a TEMPERATURE stop must be able to kill the identity driver and the smoke mid-run, not only log it.
    v8f_guard.sh's kill_ours selects processes by one -match regex; every .py the identitylora / smokelora stages run must match it,
    and the control (a script the guard must NOT touch) must not."""
    import re
    guard = (PKG / "code" / "box2_ops" / "v8f_guard.sh").read_text(encoding="utf-8")
    pats = re.findall(r"-match '([^']+)'", guard)
    assert len(pats) == 1, pats
    rx = re.compile(pats[0])
    q = (PKG / "code" / "v8f_mini_box2_queue.cmd").read_text(encoding="utf-8")
    var_py = dict(re.findall(r'set "([A-Z0-9_]+)=[^"\r\n]*?([A-Za-z0-9_]+)\.py"', q))      # `set "LORAID=v8f_lora_identity_dev.py"`
    launched = set()
    for st in ("identitylora", "smokelora"):
        m = re.search(rf"^:{st}\r?$(.*?)(?=^:[a-z0-9]+\r?$|\Z)", q, re.M | re.S)
        assert m, st
        launched |= set(re.findall(r"([A-Za-z0-9_]+)\.py", m.group(1)))                          # named directly
        launched |= {var_py[v] for v in re.findall(r"%([A-Z0-9_]+)%", m.group(1)) if v in var_py}   # named through a variable
    assert {"v8f_lora_identity_dev", "v8f_lora_smoke"} <= launched, launched
    for s in launched:
        assert rx.search(s + ".py"), f"the guard cannot kill {s}.py"
    assert not rx.search("land_hold.py") and not rx.search("ProtonVPN.py"), "the guard must touch only our jobs"
