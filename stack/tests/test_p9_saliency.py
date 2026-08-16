"""P9 probe-gradient saliency — the instrument CPU-smoked, its guards proved
able to fire.

The discipline inherited from the X4/O6 line: every stamped bound is
verified to be a GRAPH FACT (not an assertion), every guard is watched fire
on a fixture built to trip it, and every "not computable" is a stated record.
The full pass runs at a checkpoint (S-W -> S-T pause list); everything here
is the tiny synthetic stack — machinery proof, quotable NONE.
"""
from __future__ import annotations

import json
import math
import sys
import zlib
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_STACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_STACK))
sys.path.insert(0, str(_STACK / "scripts"))

import probe_saliency_p9 as P9  # noqa: E402
from tanitad.config import (EncoderConfig, PredictorConfig,  # noqa: E402
                            ReadoutConfig)
from tanitad.data.calib import CanonicalFrame  # noqa: E402
from tanitad.models.v6 import V6Config, V6Stack  # noqa: E402


def _tiny_stack(seed: int = 0) -> V6Stack:
    torch.manual_seed(seed)
    return P9.freeze(V6Stack(V6Config(
        encoder=EncoderConfig(in_channels=3, image_size=64, patch_size=16,
                              d_model=32, depth=1, n_heads=2),
        readout=ReadoutConfig(grid=2, d_readout=16),
        predictor=PredictorConfig(d_model=32, depth=1, n_heads=2, window=4,
                                  horizons=(1,)),
        d_tac=32, d_str=16, d_goal_embed=8, adapter_hidden=32,
        f_hidden_tac=32, f_hidden_str=16, f_blocks=1, n_candidates=2)))


def _frames(stack, n=3, seed=1):
    g = torch.Generator().manual_seed(seed)
    c, (h, w) = stack.cfg.encoder.in_channels, stack.cfg.encoder.image_hw()
    return torch.rand(n, stack.cfg.predictor.window, c, h, w, generator=g)


# =========================================================================== #
# 1. the structural-support stamp is a VERIFIED graph fact
# =========================================================================== #
def test_speed_saliency_is_exactly_zero_outside_the_declared_support():
    """`encode_window` encodes frames independently, so the speed readout
    (z[-2], z[-1]) can reach ONLY the last two frames — and the energy on the
    others must be EXACTLY zero, not merely small."""
    stack = _tiny_stack()
    frames = _frames(stack)
    t = P9.builtin_targets(stack)[0]
    assert t.support_frames == [2, 3]
    sal, y = P9.saliency(stack, frames, t)
    assert sal.shape == frames.shape[:2] + frames.shape[3:]
    assert y.shape == (frames.shape[0],)
    rep = P9.frame_energy_report(sal, t.support_frames)
    assert rep["support_violation"] is False
    assert rep["energy_by_frame"][0] == 0.0
    assert rep["energy_by_frame"][1] == 0.0
    assert rep["energy_by_frame"][2] > 0.0
    assert rep["energy_by_frame"][3] > 0.0
    assert "GRAPH FACT" in rep["support_note"]


def test_support_guard_FIRES_on_a_misdeclared_support():
    """⭐ THE GUARD, WATCHED FAIL. Declare the speed target's support as the
    last frame only — the real graph reaches z[-2] too, so the verification
    must flip support_violation instead of trusting the declaration."""
    stack = _tiny_stack()
    t = P9.builtin_targets(stack)[0]
    sal, _ = P9.saliency(stack, _frames(stack), t)
    rep = P9.frame_energy_report(sal, [3])            # WRONG on purpose
    assert rep["support_violation"] is True
    assert rep["energy_outside_support"] > 0.0


def test_a_last_frame_only_target_reaches_only_the_last_frame():
    """The converse control: a ridge-style readout of z[-1] alone must have
    support exactly {W-1} — proving the per-frame independence cuts BOTH
    ways, not just for the pair readout."""
    stack = _tiny_stack()
    d = stack.cfg.d_op
    g = torch.Generator().manual_seed(3)
    w_vec = torch.randn(d, generator=g)
    t = P9.TargetSpec("ridge_test", lambda z: z[:, -1] @ w_vec, [3],
                      "test ridge", "ridge")
    sal, _ = P9.saliency(stack, _frames(stack), t)
    rep = P9.frame_energy_report(sal, [3])
    assert rep["support_violation"] is False
    assert rep["last_frame_share"] == pytest.approx(1.0)


# =========================================================================== #
# 2. the FOV split — the fov_mask predicate on columns, vacuity stamped
# =========================================================================== #
def test_column_azimuth_is_linear_for_cylindrical_and_atan_for_pinhole():
    cyl = CanonicalFrame.from_hfov(120.0, 64, 64, "cylindrical")
    az = P9.column_azimuth_rad(cyl, 64)
    assert az[0] == pytest.approx(-math.radians(60) * (1 - 1 / 64), rel=1e-6)
    d1 = float(az[1] - az[0])
    d2 = float(az[-1] - az[-2])
    assert d1 == pytest.approx(d2, rel=1e-9)          # linear in column
    pin = CanonicalFrame.from_hfov(120.0, 64, 64, "pinhole")
    azp = P9.column_azimuth_rad(pin, 64)
    assert float(azp.abs().max()) < math.radians(60)  # atan compresses
    # atan's angular step SHRINKS toward the edges (d atan = f/(f^2+x^2))
    assert float(azp[-1] - azp[-2]) < float(azp[32] - azp[31])


def test_fov_split_is_stamped_vacuous_when_the_mask_covers_the_frame():
    """⛔ the rank_ceiling lesson in image clothes: a 120-deg frame against
    the 60-deg-half-angle mask has EVERY column inside — in_share = 1.0 BY
    CONSTRUCTION and the record must say so."""
    frame = CanonicalFrame.from_hfov(120.0, 64, 64, "cylindrical")
    sal = torch.ones(2, 4, 64, 64)
    r = P9.fov_split(sal, frame, math.radians(60.0))
    assert r["vacuous_by_construction"] is True
    assert r["in_fov_energy_share"] == pytest.approx(1.0)
    assert "BY CONSTRUCTION" in r["vacuous_note"]


def test_fov_split_is_informative_with_a_narrower_mask():
    """Uniform saliency on a cylindrical 120-deg frame against a 30-deg
    half-angle: exactly the middle half of the columns is inside."""
    frame = CanonicalFrame.from_hfov(120.0, 64, 64, "cylindrical")
    sal = torch.ones(1, 4, 8, 64)
    r = P9.fov_split(sal, frame, math.radians(30.0))
    assert r["vacuous_by_construction"] is False
    assert r["in_fov_energy_share"] == pytest.approx(0.5, abs=0.02)
    assert r["out_fov_energy_share"] == pytest.approx(0.5, abs=0.02)
    assert r["n_cols_in"] == 32


def test_fov_split_names_the_p4_predicate_identity():
    frame = CanonicalFrame.from_hfov(120.0, 64, 64, "cylindrical")
    r = P9.fov_split(torch.ones(1, 4, 8, 64), frame, math.radians(60.0))
    assert "fov_mask" in r["predicate"] and "visibility_occ" in r["predicate"]


def test_default_mask_half_angle_is_fov_masks_own_default():
    """The predicate constant must BE bev_raster.fov_mask's default (60 deg),
    not a number that happens to equal it — P4 §1.1 pinned the two defaults
    bit-identical, so this asserts against the function's own signature."""
    import inspect

    from tanitad.data.bev_raster import fov_mask
    default = inspect.signature(fov_mask).parameters["half_angle_rad"].default
    assert math.radians(P9.FOV_MASK_DEFAULT_HALF_ANGLE_DEG) \
        == pytest.approx(default, abs=0.0)


# =========================================================================== #
# 3. lead grouping — the P4 lesson carried in the record
# =========================================================================== #
def test_lead_groups_split_by_the_agent_centre_predicate():
    e = torch.tensor([1.0, 2.0, 3.0, 4.0])
    lf = torch.tensor([0.9, 0.8, 0.7, 0.6])
    # two leads ahead (|az| small), two beside/behind (|az| > 60 deg)
    lx = torch.tensor([20.0, 10.0, 1.0, -5.0])
    ly = torch.tensor([0.0, 2.0, 8.0, 1.0])
    r = P9.lead_groups(e, lf, lx, ly, math.radians(60.0))
    assert r["in_fov"]["n"] == 2 and r["out_of_fov"]["n"] == 2
    assert r["in_fov"]["mean_energy"] == pytest.approx(1.5)
    assert r["out_of_fov"]["mean_energy"] == pytest.approx(3.5)
    assert "MEANINGFUL" in r["p4_note"] and "Never drop" in r["p4_note"]


def test_lead_groups_reports_an_empty_group_instead_of_imputing():
    r = P9.lead_groups(torch.tensor([1.0]), torch.tensor([0.5]),
                       torch.tensor([20.0]), torch.tensor([0.0]),
                       math.radians(60.0))
    assert r["out_of_fov"]["n"] == 0
    assert "reported, not imputed" in r["out_of_fov"]["note"]


# =========================================================================== #
# 4. probe-arrays refusals — every guard watched fire
# =========================================================================== #
def _dump(tmp_path, probes, ckpt="ck.pt"):
    p = tmp_path / "probe_arrays.pt"
    torch.save({"ckpt": ckpt, "probes": probes}, p)
    return str(p)


def test_ridge_targets_refuse_an_unverified_probe_family(tmp_path):
    path = _dump(tmp_path, {"speed": {"w": torch.randn(64), "b": 0.1}},
                 ckpt="SOME_OTHER_MODEL.pt")
    targets, refusals = P9.ridge_targets(path, "my_ckpt.pt", 64,
                                         accept_family=False, window=4)
    assert targets == []
    assert "family unverified" in refusals[0]["reason"]


def test_ridge_targets_accept_a_matching_family_and_refuse_dim_mismatch(
        tmp_path):
    path = _dump(tmp_path, {
        "speed": {"w": torch.randn(64), "b": 0.1},
        "lead_gap": {"w": torch.randn(64), "b": 2.0},
        "wrong_dim": {"w": torch.randn(32), "b": 0.0},
        "zero_dir": {"w": torch.zeros(64), "b": 0.0},
    }, ckpt="/pods/ck.pt")
    targets, refusals = P9.ridge_targets(path, "/local/ck.pt", 64,
                                         accept_family=False, window=4)
    names = {t.name for t in targets}
    assert names == {"ridge_speed", "ridge_lead_gap"}
    lead = next(t for t in targets if t.name == "ridge_lead_gap")
    assert lead.family == "lead-state" and lead.support_frames == [3]
    reasons = " | ".join(r["reason"] for r in refusals)
    assert "dim mismatch" in reasons and "degenerate" in reasons


def test_accept_probe_family_overrides_a_missing_meta(tmp_path):
    path = _dump(tmp_path, {"speed": {"w": torch.randn(64), "b": 0.0}},
                 ckpt="")
    targets, _ = P9.ridge_targets(path, None, 64, accept_family=True,
                                  window=4)
    assert [t.name for t in targets] == ["ridge_speed"]


# =========================================================================== #
# 5. end-to-end synthetic smoke — the runnable-at-a-checkpoint contract
# =========================================================================== #
def test_end_to_end_synthetic_smoke(tmp_path):
    s = P9.main(["--synthetic", "--out", str(tmp_path), "--n-windows", "3",
                 "--seed", "0"])
    assert s["quotable"].startswith("NONE")
    assert s["tier"].startswith("T0-DIAGNOSTIC")
    assert set(s["targets"]) == {"speed", "yaw"}
    for t in s["targets"].values():
        assert t["support_violation"] is False
        assert t["degenerate"] is False
        assert t["fov_split"]["vacuous_by_construction"] is True
    nc = " | ".join(r["reason"] for r in s["not_computable"])
    assert "lead-state" in json.dumps(s["not_computable"])
    assert "not computable at this checkpoint" in nc
    on_disk = json.loads((tmp_path / "p9_summary.json").read_text())
    assert on_disk["targets"]["speed"]["energy_total"] > 0
    assert (tmp_path / "saliency_speed.pt").exists()
    assert (tmp_path / "saliency_speed.png").exists()


def test_png_writer_emits_a_valid_grayscale_png(tmp_path):
    p = tmp_path / "m.png"
    P9.write_png_gray(p, torch.rand(16, 24))
    raw = p.read_bytes()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    assert raw[12:16] == b"IHDR"
    w, h = int.from_bytes(raw[16:20], "big"), int.from_bytes(raw[20:24],
                                                             "big")
    assert (w, h) == (24, 16)
    start = raw.index(b"IDAT") + 4
    end = raw.index(b"IEND") - 8
    rows = zlib.decompress(raw[start:end])
    assert len(rows) == 16 * (24 + 1)                 # filter byte per row


def test_v6config_roundtrips_through_the_ckpt_dict():
    stack = _tiny_stack()
    cfg2 = P9.v6config_from_dict(json.loads(json.dumps(
        stack.cfg.to_dict())))
    assert cfg2.d_op == stack.cfg.d_op
    assert cfg2.d_tac == stack.cfg.d_tac
    assert cfg2.predictor.window == stack.cfg.predictor.window
    assert cfg2.encoder.image_hw() == stack.cfg.encoder.image_hw()
    # the rebuilt config builds a stack whose state_dict keys MATCH — the
    # property load_v6_stack's strict load depends on
    assert set(V6Stack(cfg2).state_dict()) == set(stack.state_dict())


def test_ckpt_loader_refuses_shape_and_container_garbage(tmp_path):
    bad = tmp_path / "bad.pt"
    torch.save({"something_else": 1}, bad)
    with pytest.raises(SystemExit, match="neither a v6 trainer checkpoint"):
        P9.load_v6_stack(str(bad))
    noconf = tmp_path / "noconf.pt"
    torch.save({"stack": {}, "config": {}}, noconf)
    with pytest.raises(SystemExit, match="no v6_config"):
        P9.load_v6_stack(str(noconf))


def test_windows_shape_mismatch_is_refused(tmp_path):
    blob = tmp_path / "w.pt"
    torch.save({"frames": torch.rand(2, 3, 3, 32, 32)}, blob)
    with pytest.raises(SystemExit, match="do not match"):
        P9.main(["--synthetic", "--windows-pt", str(blob),
                 "--out", str(tmp_path / "o")])
