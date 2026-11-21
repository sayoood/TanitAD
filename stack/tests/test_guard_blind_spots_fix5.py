"""⛔⛔ FIX-5 — the two guard blind spots the A16 audit's Q6 constructed and found GREEN.

`…/2026-09-26-refcv6-frozen-trunk-audit/code/q6_guards.py`:

* **G2 R3** -- ``timm_trunk._assert_pretrained_loaded`` read the STEM only, so a trunk with the
  ImageNet stem and a RANDOM ``layer4`` (a partial load) passed. Every stage is now fingerprinted
  (sha256[:16] of the first conv of layer1..layer4, MEASURED on the cached checkpoints and banked
  in `…/2026-09-26-declared-vs-built/raw/stage_fingerprints.json`).
* **G1 R3** -- ``refcv6_max_speed.read_speed_max_sidecar_v6`` SKIPPED its label-md5 check when the
  ``.meta.json`` was absent, so a sidecar from another label release read clean. With a
  ``label_md5`` named, the meta is now REQUIRED.

Each regression arm is the audit's own construction and must go RED; each GREEN control must read.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tanitad.models import timm_trunk as tt  # noqa: E402
from tanitad.refs import refcv6_max_speed as v6ms  # noqa: E402

timm = pytest.importorskip("timm")

#: the MEASURED fingerprints (2026-09-26, `raw/stage_fingerprints.json`), as literals
RESNET34_STAGES = {"layer1.0.conv1": "f6daa1fb2b054df6", "layer2.0.conv1": "3c633611cfc77356",
                   "layer3.0.conv1": "237623d9929e51c1", "layer4.0.conv1": "b0e7e9ad75f71c5d"}


def _cached_resnet34():
    try:
        return timm.create_model("resnet34.a1_in1k", pretrained=True, features_only=True,
                                 out_indices=(3, 4))
    except Exception as e:                                   # offline and not cached
        pytest.skip(f"resnet34.a1_in1k weights not in the local HF cache ({type(e).__name__}); "
                    f"a SKIP is not a pass -- run where the checkpoint is cached")


# ---- G2: the pretrained-weights check reads EVERY stage ----------------------------------- #
def test_the_pinned_fingerprints_are_the_MEASURED_literals():
    assert tt._PINNED_STAGE_SHA16["resnet34.a1_in1k"] == RESNET34_STAGES
    assert set(tt._PINNED_STAGE_SHA16["resnet101.a1_in1k"]) == set(RESNET34_STAGES)


def test_G2_GREEN_the_real_checkpoint_passes_and_fingerprints_as_measured():
    net = _cached_resnet34()
    assert tt.stage_fingerprints(net) == RESNET34_STAGES
    tt._assert_pretrained_loaded(net, "resnet34.a1_in1k")


def test_G2_R3_stem_intact_layer4_RANDOM_now_goes_RED():
    """The audit's construction exactly: the stem as downloaded, layer4 re-initialised."""
    net = _cached_resnet34()
    torch.manual_seed(0)
    for m in net.layer4.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
    with pytest.raises(RuntimeError, match="PARTIAL load"):
        tt._assert_pretrained_loaded(net, "resnet34.a1_in1k")


def test_G2_R1_a_random_init_still_goes_RED():
    net = timm.create_model("resnet34.a1_in1k", pretrained=False, features_only=True,
                            out_indices=(3, 4))
    with pytest.raises(RuntimeError, match="NOT loaded"):
        tt._assert_pretrained_loaded(net, "resnet34.a1_in1k")


def test_G2_every_stage_is_checked_not_only_layer4(monkeypatch):
    """Mechanism, no download: pin a random resnet18's OWN fingerprints under a fake name, then
    re-initialise ONE stage at a time -- each must go RED on its own."""
    net = timm.create_model("resnet18", pretrained=False, features_only=True,
                            out_indices=(3, 4))
    stem = float(net.conv1.weight.detach().abs().sum())
    monkeypatch.setitem(tt._PINNED_STATS, "fake.r18", stem)
    monkeypatch.setitem(tt._PINNED_STAGE_SHA16, "fake.r18", tt.stage_fingerprints(net))
    tt._assert_pretrained_loaded(net, "fake.r18")                       # the GREEN control
    assert sorted(tt.stage_fingerprints(net)) == ["layer1.0.conv1", "layer2.0.conv1",
                                                  "layer3.0.conv1", "layer4.0.conv1"]
    for stage in ("layer1", "layer2", "layer3", "layer4"):
        conv = getattr(net, stage)[0].conv1
        saved = conv.weight.detach().clone()
        with torch.no_grad():
            conv.weight.mul_(1.0001)                                     # the smallest change
        with pytest.raises(RuntimeError, match="PARTIAL load"):
            tt._assert_pretrained_loaded(net, "fake.r18")
        with torch.no_grad():
            conv.weight.copy_(saved)
    tt._assert_pretrained_loaded(net, "fake.r18")


# ---- G1: the sidecar md5 check REFUSES when its meta is absent ----------------------------- #
def _row(sid, v=11.19, b=1):
    oh = [0.0] * v6ms.N_SPEED_MAX_BINS_V6
    oh[b] = 1.0
    return {"sid": sid, "v_hi_ms": v, "bin": b, "one_hot": oh,
            "limit_kmh": v6ms.SPEED_MAX_STEPS_KMH_V6[b], "over_ceiling": False, "valid": 1}


def _sidecar(meta=None) -> str:
    p = Path(tempfile.mkdtemp()) / "s.jsonl"
    p.write_text(json.dumps(_row(1)) + "\n", encoding="utf-8")
    if meta is not None:
        Path(str(p) + ".meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return str(p)


def test_G1_R3_meta_ABSENT_with_a_named_blob_now_goes_RED():
    with pytest.raises(SystemExit, match="ABSENT"):
        v6ms.read_speed_max_sidecar_v6(_sidecar(), label_md5="0" * 32)


def test_G1_meta_without_source_md5_goes_RED():
    with pytest.raises(SystemExit, match="missing `source_md5`"):
        v6ms.read_speed_max_sidecar_v6(_sidecar({"schema": "x"}), label_md5="0" * 32)


def test_G1_GREEN_controls_matching_meta_reads_and_no_named_blob_is_unchanged():
    by_sid, _ = v6ms.read_speed_max_sidecar_v6(_sidecar({"source_md5": "0" * 32}),
                                               label_md5="0" * 32)
    assert by_sid[1] == (pytest.approx(11.19), 1.0)
    by_sid, _ = v6ms.read_speed_max_sidecar_v6(_sidecar())             # no md5 asked for
    assert by_sid[1][1] == 1.0
