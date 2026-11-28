"""⛔ C26 — THE RIG-CORRELATED BLACK STRIP IS EQUALIZED AT THE TRUNK AND UNOBSERVED IN THE LIFT.

⭐⭐ WHY THIS FILE EXISTS. MEASURED 2026-09-23 over all 4,713 clips of the 416×1024 B1 corpus
cache on Thor (`_b1_416_strip_census_full.json`, 0 decode errors): **2,721 clips (57.73 %)**
carry **26–43** fully-black bottom rows (median 31) and **1,992 carry exactly zero** — bimodal,
with nothing in between. On eval-139 an arbitrary label painted on as that strip is **0.899 ±
0.046 balanced-accuracy decodable from an 8×16 thumbnail**, against controls at chance. A strip
whose presence identifies the rig is a shortcut (`calib.py` retraction class **C26**: *"a
rig-correlated BLACK region is still a rig-correlated signal, and this model eats shortcuts"*).

⭐ THE FIX HAS TWO HALVES, AND ONE WITHOUT THE OTHER IS A NEW DEFECT.
* the TRUNK zeroes the bottom N rows of every frame before normalisation — train and eval
  alike, so the region is constant and carries no rig information;
* the LIFT marks the same rows UNOBSERVED. Its `observed` mask existed and was never passed
  (2026-09-22 review); without it a zeroed row reads as observed BLACK ROAD, and every such
  cell sits in the 3.75–9.75 m headway band.

⛔ EVERY EXPECTATION IS A LITERAL OR ANALYTIC: the normalised value of black is
`(0 - mean) / std` from ImageNet's published statistics, not from the module under test.
⛔ CPU only.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

torch = pytest.importorskip("torch")

from tanitad.models import timm_trunk as TT  # noqa: E402

#: ⛔ LITERALS. N is the MEASURED corpus maximum; H x W is the SPEC §12 geometry.
N_ROWS = 43
H, W = 416, 1024
#: ImageNet's published statistics (torchvision / timm), NOT read from the trunk
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _trunk(n: int):
    cfg = TT.TimmTrunkConfig(image_hw=(H, W), equalize_bottom_rows=n,
                             verify_imagenet_stats=False)
    t = TT.TimmResNetTrunk.__new__(TT.TimmResNetTrunk)
    torch.nn.Module.__init__(t)
    t.cfg = cfg
    t.k = 1
    t.register_buffer("_mean", torch.tensor(IMAGENET_MEAN).reshape(1, 3, 1, 1))
    t.register_buffer("_std", torch.tensor(IMAGENET_STD).reshape(1, 3, 1, 1))
    t.norm_calls = 0
    return t


def test_the_strip_rows_become_EXACTLY_normalised_black() -> None:
    """Rows [H-N, H) must equal (0 - mean) / std per channel -- black, then normalised."""
    torch.manual_seed(0)
    x = torch.rand(2, 3, H, W)
    y = _trunk(N_ROWS).normalise(x)
    black = torch.tensor([(0.0 - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)])
    got = y[:, :, -N_ROWS:, :]
    assert torch.allclose(got, black.reshape(1, 3, 1, 1).expand_as(got), atol=1e-6)


def test_the_rows_ABOVE_the_strip_are_untouched() -> None:
    """⭐ THE CONTROL: equalization must not leak upward. Row H-N-1 and above normalise
    exactly as they would with the lever off."""
    torch.manual_seed(1)
    x = torch.rand(2, 3, H, W)
    on = _trunk(N_ROWS).normalise(x)
    off = _trunk(0).normalise(x)
    assert torch.equal(on[:, :, :-N_ROWS, :], off[:, :, :-N_ROWS, :])


def test_zero_rows_is_BIT_IDENTICAL_to_every_existing_arm() -> None:
    """⛔ The default must change nothing: same tensor, and the input is not mutated."""
    torch.manual_seed(2)
    x = torch.rand(2, 3, H, W)
    x0 = x.clone()
    y = _trunk(0).normalise(x)
    ref = (x0 - torch.tensor(IMAGENET_MEAN).reshape(1, 3, 1, 1)) / \
        torch.tensor(IMAGENET_STD).reshape(1, 3, 1, 1)
    assert torch.equal(y, ref)
    assert torch.equal(x, x0), "normalise mutated its input in place"


def test_equalization_does_not_mutate_the_callers_tensor() -> None:
    """The batch tensor is shared with loss code that reads it; an in-place write would
    zero the strip in places nothing asked for."""
    torch.manual_seed(3)
    x = torch.rand(1, 3, H, W)
    x0 = x.clone()
    _trunk(N_ROWS).normalise(x)
    assert torch.equal(x, x0)


def test_the_lift_marks_the_SAME_rows_unobserved() -> None:
    """⛔ THE SECOND HALF. With N rows equalized, the lift must lose exactly the cells whose
    nearest pixel falls in those rows -- and must lose NONE when N is 0 (the control)."""
    from tanitad.data import calib
    from tanitad.data.rig_projection import RigCamera
    from tanitad.models import refcv6_perception_branch as P
    from tanitad.models import trunk_shapes as TS
    frame = TS.FRAME_416x1024
    cam = RigCamera.nominal(frame, height_m=1.5) if hasattr(RigCamera, "nominal") \
        else None
    if cam is None:
        pytest.skip("no nominal rig camera constructor")
    b0 = P.LiftGeometryBank({"clipA": cam}, frame=frame, stride=16)
    bN = P.LiftGeometryBank({"clipA": cam}, frame=frame, stride=16,
                            equalize_bottom_rows=N_ROWS)
    from tanitad.data.v2_dataset import stable_episode_id
    eid = int(stable_episode_id("clipA"))
    _, v0 = b0.geometry(eid)
    _, vN = bN.geometry(eid)
    n0, nN = int(v0.sum()), int(vN.sum())
    assert nN < n0, f"equalizing {N_ROWS} rows removed no lift cells ({n0} -> {nN})"
    assert bool((vN & ~v0).sum() == 0), "equalization ADDED valid cells"


def test_the_trainer_flag_REACHES_the_built_trunk() -> None:
    """⛔ A FLAG THAT PARSES AND REACHES NOTHING is the defect this programme keeps paying
    for (`305debd`). The flag must arrive on the constructed trunk's config."""
    here = os.path.dirname(os.path.abspath(__file__))
    tp = os.path.join(here, "..", "scripts", "refc_v3_train.py")
    spec = importlib.util.spec_from_file_location("rt_eq", tp)
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)
    a = T.build_parser().parse_args(["--arm", "hier", "--out", "z", "--trunk", "timm",
                                     "--equalize-bottom-rows", str(N_ROWS)])
    cfg = T._pin_trainer_cfg(T.v3.refc_v3_smoke_config(True), a)
    assert int(cfg.core.encoder.trunk_equalize_bottom_rows) == N_ROWS
    cfg.core.encoder.in_channels = 3
    cfg.core.encoder.trunk_pretrained = False
    torch.manual_seed(0)
    m = T.v3.RefCV3Model(cfg)
    assert int(m.core.encoder.cfg.equalize_bottom_rows) == N_ROWS
