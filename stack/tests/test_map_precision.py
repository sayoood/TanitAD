"""``--w-map-fp`` -- the 10 cm map head's PRECISION term in the near-field corridor (refcv9 warm start, PI 2026-10-27).

    L_fp = mean over the GT-DRIVABLE cells of the corridor (x in [0, 20) m, |y| < 3 m, by cell centre) of
           (p_sidewalk + p_edge), p = the head's softmax over the 8 classes;  added as w * L_fp.

Every expectation is a LITERAL authored here:
  1. the class ids (1 / 5 / 7) against the label NAMES, the corridor rows / cols on the refcv8 extent (1000 x 600) and on
     the CI rig's (600 x 320), by hand from the cell-centre rule;
  2. ANALYTIC values: drivable everywhere -> exactly 0; sidewalk p = 1 -> exactly 1; a sidewalk p = 1 OUTSIDE the
     corridor, or on a cell that is NOT GT-drivable, never contributes (exactly 0); k of 2^n cells -> exactly k / 2^n;
     MUTATION: the corridor crop removed -> the out-of-corridor case reads 1, not 0 (RED);
  3. the argv refusals, the G-DVB lever both ways, the weight gates (REFC_WEIGHT_GATES + G-LIVE's LIVE_WEIGHT_RULES);
  4. the trainer's loss on the map rig: absent weight -> no key (the tip's graph); w > 0 -> the key, its cell count
     (2,323 = the fixture's GT-drivable corridor cells, counted by hand below) and the total moved by w * L_fp.
The flags-absent bit identity against the PRISTINE tip is the stepid digest (evidence, not this file).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tanitad.data.semantic_map_gt_fine import FINE_CLASSES  # noqa: E402
from tanitad.train import declared_vs_built as D  # noqa: E402
from tanitad.train import map_precision as MP  # noqa: E402

#: the refcv8 10 cm extent (100 m ahead x +-30 m at 0.1 m) and the CI rig's (EXTENT_V2: 60 m x +-16 m)
R8_HW, R8_X, R8_Y = (1000, 600), 100.0, 30.0
RIG_HW, RIG_X, RIG_Y = (600, 320), 60.0, 16.0
#: by hand: rows with centre (i + 0.5) * 0.1 in [0, 20) -> i = 0..199; cols with |-y_half + (j + 0.5) * 0.1| < 3 ->
#: refcv8: j + 0.5 in (270, 330) -> j = 270..329; rig: j + 0.5 in (130, 190) -> j = 130..189
R8_ROWS, R8_COLS = (0, 200), (270, 330)
RIG_ROWS, RIG_COLS = (0, 200), (130, 190)
BIG = 1000.0          # a logit gap whose softmax tail is EXACTLY 0.0 in float32 (exp(-1000) underflows)


# =========================================================================== #
# 1. the class ids and the corridor                                           #
# =========================================================================== #
def test_the_class_ids_are_read_from_the_label_names():
    assert (MP.DRIVABLE, MP.EDGE, MP.SIDEWALK) == (1, 5, 7)
    assert FINE_CLASSES[1] == "drivable" and FINE_CLASSES[5] == "non-drivable edge"
    assert FINE_CLASSES[7] == "sidewalk / verge" and len(FINE_CLASSES) == 8
    assert MP.CORRIDOR_X_M == (0.0, 20.0) and MP.CORRIDOR_Y_HALF_M == 3.0


def test_the_corridor_rows_and_cols_on_the_refcv8_and_the_rig_extents():
    assert MP.corridor_ranges(R8_HW, x_max_m=R8_X, y_half_m=R8_Y) == (R8_ROWS, R8_COLS)
    assert MP.corridor_ranges(RIG_HW, x_max_m=RIG_X, y_half_m=RIG_Y) == (RIG_ROWS, RIG_COLS)
    m = MP.corridor_mask(R8_HW, x_max_m=R8_X, y_half_m=R8_Y)
    assert int(m.sum()) == 12_000 and bool(m[0, 270]) and bool(m[199, 329])
    assert not bool(m[200, 300]) and not bool(m[0, 269]) and not bool(m[0, 330])
    with pytest.raises(ValueError, match="not the extent"):
        MP.corridor_ranges((999, 600), x_max_m=R8_X, y_half_m=R8_Y)


# =========================================================================== #
# 2. analytic values and the mutation                                         #
# =========================================================================== #
def _logits(cls: int, hw=RIG_HW, b=1):
    """Every cell's softmax is EXACTLY one-hot on `cls`."""
    lg = torch.full((b, 8, *hw), -BIG)
    lg[:, cls] = 0.0
    return lg


def _codes(code: int, hw=RIG_HW, b=1):
    return torch.full((b, *hw), code, dtype=torch.uint8)


def _fp(lg, codes):
    return MP.map_fp_loss(lg, codes, x_max_m=RIG_X, y_half_m=RIG_Y)


def test_ANALYTIC_drivable_everywhere_is_exactly_0_and_sidewalk_everywhere_is_exactly_1():
    r = _fp(_logits(MP.DRIVABLE), _codes(MP.DRIVABLE))
    assert float(r["loss"]) == 0.0 and r["n_cells"] == 12_000
    r = _fp(_logits(MP.SIDEWALK), _codes(MP.DRIVABLE))
    assert float(r["loss"]) == 1.0 and r["n_cells"] == 12_000
    r = _fp(_logits(MP.EDGE, b=2), _codes(MP.DRIVABLE, b=2))
    assert float(r["loss"]) == 1.0 and r["n_cells"] == 24_000


def test_ANALYTIC_cells_outside_the_corridor_or_not_drivable_never_contribute():
    # sidewalk p = 1 everywhere OUTSIDE the corridor, drivable p = 1 inside; GT drivable everywhere
    lg = _logits(MP.SIDEWALK)
    lg[:, :, RIG_ROWS[0]:RIG_ROWS[1], RIG_COLS[0]:RIG_COLS[1]] = _logits(
        MP.DRIVABLE, hw=(200, 60))[:, :, :, :]
    assert float(_fp(lg, _codes(MP.DRIVABLE))["loss"]) == 0.0
    # sidewalk p = 1 everywhere, but GT is sidewalk / unseen in the corridor: no GT-drivable cell -> exact 0, n 0
    for code in (MP.SIDEWALK, 255, MP.EDGE, 0):
        r = _fp(_logits(MP.SIDEWALK), _codes(code))
        assert float(r["loss"]) == 0.0 and r["n_cells"] == 0, code
    # k of 2^n GT-drivable cells flagged sidewalk -> exactly k / 2^n; the non-drivable flagged cells add nothing
    codes = _codes(255)
    codes[0, 10:18, 140:148] = MP.DRIVABLE                       # 64 GT-drivable corridor cells
    lg = _logits(MP.SIDEWALK)                                    # every cell flagged sidewalk ...
    lg[0, :, 10:18, 140:148] = _logits(MP.DRIVABLE, hw=(8, 8))[0]
    lg[0, :, 10, 140:145] = _logits(MP.SIDEWALK, hw=(1, 5))[0, :, 0]   # ... except the drivable block, 5 of whose
    # cells are flagged; every other cell of the grid is flagged too but is not GT-drivable
    r = _fp(lg, codes)
    assert float(r["loss"]) == 5 / 64 and r["n_cells"] == 64
    # the gradient reaches the corridor logits only
    lg = torch.zeros(1, 8, *RIG_HW, requires_grad=True)
    _fp(lg, _codes(MP.DRIVABLE))["loss"].backward()
    g = lg.grad.abs().sum(dim=1)[0]
    assert float(g[RIG_ROWS[1]:].sum()) == 0.0 and float(g[:, :RIG_COLS[0]].sum()) == 0.0
    assert float(g[:, RIG_COLS[1]:].sum()) == 0.0 and float(g[0, 150]) > 0.0


def test_ANALYTIC_no_drivable_cell_is_an_exact_zero_ATTACHED_to_the_graph():
    lg = torch.zeros(2, 8, *RIG_HW, requires_grad=True)
    r = _fp(lg, _codes(255, b=2))
    assert float(r["loss"]) == 0.0 and r["n_cells"] == 0 and r["loss"].requires_grad
    r["loss"].backward()
    assert float(lg.grad.abs().sum()) == 0.0


def test_MUTATION_the_corridor_crop_removed_goes_RED(monkeypatch):
    """The out-of-corridor case of the analytic test, with `corridor_ranges` returning the WHOLE grid: the cells the
    term must never read are read, and the exact 0 becomes 0.9375 (= 1 - 12,000 / 192,000) -- RED."""
    lg = _logits(MP.SIDEWALK)
    lg[:, :, RIG_ROWS[0]:RIG_ROWS[1], RIG_COLS[0]:RIG_COLS[1]] = _logits(MP.DRIVABLE, hw=(200, 60))
    assert float(_fp(lg, _codes(MP.DRIVABLE))["loss"]) == 0.0
    monkeypatch.setattr(MP, "corridor_ranges", lambda out_hw, **_k: ((0, int(out_hw[0])), (0, int(out_hw[1]))))
    r = _fp(lg, _codes(MP.DRIVABLE))
    assert r["n_cells"] == 192_000 and float(r["loss"]) == 0.9375
    assert float(r["loss"]) != 0.0


# =========================================================================== #
# 3. argv, G-DVB, the weight gates                                            #
# =========================================================================== #
def _a(**over):
    a = argparse.Namespace(w_map_fp=0.5, map_hires="on", w_map_hires=1.0, r8_train_only_map=False,
                           r9_train_only=None)
    for k, v in over.items():
        setattr(a, k, v)
    return a


def test_the_argv_refusals():
    MP.pin_args(_a())
    MP.pin_args(_a(w_map_fp=0.0, map_hires="off", w_map_hires=0.0))           # off: nothing is checked
    MP.pin_args(_a(r9_train_only="planner,map"))
    MP.pin_args(_a(r9_train_only="map"))
    MP.pin_args(_a(r8_train_only_map=True))
    for bad, needle in ((_a(w_map_fp=-0.1), "finite value >= 0"), (_a(w_map_fp=float("nan")), "finite value >= 0"),
                        (_a(map_hires="off"), "--map-hires is not `on`"),
                        (_a(w_map_hires=0.0), "--w-map-hires is 0"),
                        (_a(r9_train_only="planner"), "not the map head")):
        with pytest.raises(SystemExit, match=needle.replace("(", r"\(")):
            MP.pin_args(bad)


def test_GDVB_the_lever_reads_the_weight_the_loss_reads():
    import refc_v3_train  # noqa: F401  -- the trainer registers its levers at import
    assert D.REGISTRY["w_map_fp"].kind == "loss"
    m = torch.nn.Module()
    m._map_hires = torch.nn.Linear(1, 1)
    assert MP.dvb_check(m, _a(w_map_fp=0.0)) == []
    assert "_w_map_fp" in str(MP.dvb_check(m, _a())[0])                          # declared, not built: RED
    setattr(m, MP.MODEL_ATTR, 0.5)
    assert MP.dvb_check(m, _a()) == []
    assert "does not declare" in str(MP.dvb_check(m, _a(w_map_fp=0.0))[0])        # built, not declared: RED
    assert len(MP.dvb_check(m, _a(w_map_fp=0.25))) == 1                           # a different weight: RED
    m._map_hires = None
    assert "no 10 cm head" in str(MP.dvb_check(m, _a())[0])


def test_the_weight_gates_of_the_trainer_and_of_G_LIVE():
    import launch_gate as LG
    import refc_v3_train as T
    assert "w_map_fp" in T.REFC_WEIGHT_GATES
    rule = LG.LIVE_WEIGHT_RULES["w_map_fp"]
    assert rule[1] == ("map_fp",) and rule[2] == 1.0
    assert rule[0](_a()) is True
    assert rule[0](_a(w_map_fp=0.0)) is False and rule[0](_a(map_hires="off")) is False
    assert rule[0](_a(w_map_hires=0.0)) is False


# =========================================================================== #
# 4. the trainer's loss on the map rig                                        #
# =========================================================================== #
def test_RIG_the_term_in_the_trainers_loss_absent_weight_no_key_live_weight_the_key():
    pytest.importorskip("timm")
    import refc_v3_train as T
    import test_map_only_ft as R
    model, cfg = R._build()
    batch = R._batch(cfg)
    # the fixture's GT-drivable corridor cells, counted by hand: rows 0-199 x cols 130-189 of `map_fine` == 1
    assert int((batch["map_fine"][:, 0:200, 130:190] == 1).sum()) == 2323
    model.train()
    l0 = T.compute_losses_v3(model, batch, "cpu", mode="diffusion")
    assert not hasattr(model, MP.MODEL_ATTR)
    assert MP.LOSS_KEY not in l0 and MP.COUNT_KEY not in l0                   # w = 0: the tip's keys
    model2, cfg2 = R._build()
    setattr(model2, MP.MODEL_ATTR, 0.5)
    model2.train()
    l1 = T.compute_losses_v3(model2, R._batch(cfg2), "cpu", mode="diffusion")
    assert l1[MP.COUNT_KEY] == 2323.0
    assert l1[MP.LOSS_KEY].requires_grad and 0.0 < float(l1[MP.LOSS_KEY]) < 1.0
    assert float(l1["map_hires"]) == float(l0["map_hires"])                   # the CE is untouched
    assert float(l1["loss"] - l0["loss"]) == pytest.approx(0.5 * float(l1[MP.LOSS_KEY]), rel=1e-4)
