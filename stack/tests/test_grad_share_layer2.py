"""refcv9 GS-LAYER2 -- the ``layer2`` parameter group of the in-run gradient-share instrument (`tanitad/train/grad_share.py`).

WHY. E0j's design number (the banded map takes 0.1459 of LAYER2 at w_map = 1, 0.0659 of the trunk; w* 0.5107 / 1.018) was
measured by ``TanitAD Research Lab/.../2026-10-27-perception-redesign/e0/pgrad_gram.py`` on the group
``stage_layer2``. ``grad_share.GROUP_PREFIXES`` had trunk / bev025 / bev_pool only, so refcv9-full's share alarm read
``gs_trunk_proj_map_hires`` -- the nearest CONTAINING quantity, not the one the floor was designed on. The group is now there.

WHAT E0j CALLED ``layer2``. ``pgrad_gram.py`` has no literal ``layer2``: lines 204-205 file every ``core.encoder.*`` parameter by
``grad_conflict.default_group_of`` (the backbone's own stage token), and ``e0j_read.py:38-39`` reads its decision group
``layer2`` as ``stage_layer2``. On the timm ResNet trunk that is the name prefix ``core.encoder.net.layer2.``.

The pgrad_gram group logic is INLINE in its ``main()`` (not an importable function), so `_pgrad_gram_layer2` below REIMPLEMENTS
lines 197-205 from the quoted source; ``gram_core.share_from_gram`` (pure numpy, importable) is the E0j share formula, and
`test_the_inline_share_formula_is_gram_cores` pins the inline copy to it whenever the file is in the tree.

Pinned here (CPU, the tiny REAL RefCV3Model of ``test_map_only_ft._build``: timm resnet18 @ 64x64 + the 10 cm branch):
  * EQUALITY: ``gs_layer2_proj_*`` / ``gs_layer2_norm_total`` equal E0j's formula on the SAME gradients, to 1e-6;
  * NON-ZERO + a CONTROL that reads EXACTLY 0.0 (a 0.0-weighted map term);
  * every group is read on its OWN members (layer2 lies inside trunk and trunk does not move);
  * OFF: every pre-existing key is bit-identical with and without the group;
  * DELIBERATE REGRESSIONS: no ``layer2`` line / a wrong prefix / the tip's first-match ``break`` each go RED.
"""
from __future__ import annotations

import importlib.util
import math
import struct
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
np = pytest.importorskip("numpy")
ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import refc_v3_train as T  # noqa: E402
import test_map_only_ft as R  # noqa: E402
from tanitad.train import grad_conflict as gcf  # noqa: E402
from tanitad.train import grad_share as GS  # noqa: E402

E0 = (ROOT.parent / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-27-perception-redesign" / "e0")
# non-unit, so a missing multiply shows; a POWER OF TWO on purpose: scaling a float32 loss by it is exact, so the 1e-6
# equality below measures grad_share against E0j's formula and not the float32 rounding of `w * L` (MEASURED here: 2.0 reads
# <= 8e-9 off E0j, 2.5 reads 3.6e-7 off -- both inside 1e-6, the second with a 3x margin only)
W_MAP = 2.0
OLD3 = {"trunk": ("core.encoder.",), "bev025": ("_map_hires.lift.", "_map_hires.encoder."),
        "bev_pool": ("_perception.bev_pool.",)}                    # the tip's GROUP_PREFIXES, verbatim
TIMING_KEYS = {"gs_measure_s", "gs_pass_s_max"}                    # wall clock: the only keys that cannot be bit-equal


# =========================================================================== #
# the rig + E0j's definitions                                                  #
# =========================================================================== #
@pytest.fixture(scope="module")
def rig():
    model, cfg = R._build()
    model._w_map_hires = W_MAP
    model.train()
    torch.manual_seed(7)
    losses = T.compute_losses_v3(model, R._batch(cfg), "cpu", log_metrics=False)
    assert losses["map_hires"].requires_grad and losses["traj"].requires_grad
    return model, losses


def _pgrad_gram_layer2(model):
    """``pgrad_gram.py`` lines 197-205, reimplemented from the quoted source: the ``core.encoder.*`` parameters whose
    ``default_group_of`` is ``stage_layer2`` (pgrad_gram first re-enables ``requires_grad`` on its prefixes; the trainer's
    own state is the one grad_share reads, and on this build every encoder parameter is already trainable)."""
    pre = "core.encoder."
    return [(n, p) for n, p in model.named_parameters()
            if p.requires_grad and n.startswith(pre) and gcf.default_group_of(n[len(pre):]) == "stage_layer2"]


def _share_inline(gm, w):
    """``e0/gram_core.py:29-36`` -- s_i(w) = w_i (Gm w)_i / (w^T Gm w)."""
    gm = np.asarray(gm, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    return w * (gm @ w) / float(w @ gm @ w)


def _vec(t, params):
    gr = torch.autograd.grad(t, params, retain_graph=True, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1).double() for g, p in zip(gr, params)])


def _read(model, losses, prefixes=None):
    groups = GS.param_groups(model) if prefixes is None else GS.param_groups(model, prefixes)
    terms = GS.term_tensors(losses, GS.weights_of(model, T.TRAJ_WEIGHT))
    return GS.measure(terms, losses["loss"], groups)


def _equality_problems(model, losses, prefixes=None):
    """Everything that must hold for the layer2 reading to be E0j's quantity; [] = holds. Used by the GREEN test AND by
    the deliberate-regression tests (a defect must make this non-empty)."""
    bad = []
    groups = GS.param_groups(model) if prefixes is None else GS.param_groups(model, prefixes)
    want = [p for _, p in _pgrad_gram_layer2(model)]
    if "layer2" not in groups:
        return ["no `layer2` group"]
    if [id(p) for p in groups["layer2"]] != [id(p) for p in want]:
        bad.append("layer2 members differ from pgrad_gram's stage_layer2")
        return bad
    w = GS.weights_of(model, T.TRAJ_WEIGHT)
    terms = GS.term_tensors(losses, w)
    row = GS.measure(terms, losses["loss"], groups)
    # E0j's formula on the SAME gradients: the three terms are traj, map_hires and `rest` (= total - named; weight 1), all
    # as UNWEIGHTED gradient vectors over pgrad_gram's layer2 set, G(w) = sum_j w_j g_j == the total's gradient
    # (`rest` is the trainer's own tensor `total - named`, so its gradient is not rebuilt by a float32 CANCELLATION
    #  `g_tot - w g_traj - w g_map`, which would put ~3e-7 of rounding noise into a 0.35 share and say nothing about layer2)
    g_traj, g_map, g_rest = _vec(losses["traj"], want), _vec(losses["map_hires"], want), _vec(terms["rest"], want)
    vecs = {"traj": g_traj, "map_hires": g_map, "rest": g_rest}
    names = ("traj", "map_hires", "rest")
    gm = np.array([[float(torch.dot(vecs[a], vecs[b])) for b in names] for a in names])
    wv = np.array([w["traj"], w["map_hires"], 1.0])
    s = _share_inline(gm, wv)
    for i, t in enumerate(names):
        got = row.get(f"gs_layer2_proj_{t}")
        if got is None or abs(got - float(s[i])) > 1e-6:
            bad.append(f"proj_{t}: grad_share {got} vs E0j {float(s[i])}")
    nt = math.sqrt(float(wv @ gm @ wv))
    if abs(row.get("gs_layer2_norm_total", float("nan")) - nt) > 1e-6 * max(1.0, nt):
        bad.append(f"norm_total: grad_share {row.get('gs_layer2_norm_total')} vs E0j {nt}")
    return bad


# =========================================================================== #
# item 1 -- WHAT layer2 is                                                     #
# =========================================================================== #
def test_layer2_is_exactly_pgrad_grams_stage_layer2_set(rig):
    model, _ = rig
    ref = _pgrad_gram_layer2(model)
    groups = GS.param_groups(model)
    assert "layer2" in groups
    assert [id(p) for p in groups["layer2"]] == [id(p) for _, p in ref]
    # LITERALS (MEASURED, this build = timm resnet18): 15 tensors / 525,568 parameters; the first three names
    assert len(ref) == 15 and sum(p.numel() for _, p in ref) == 525568
    assert [n for n, _ in ref][:3] == ["core.encoder.net.layer2.0.conv1.weight", "core.encoder.net.layer2.0.bn1.weight",
                                       "core.encoder.net.layer2.0.bn1.bias"]
    # and the PREFIX is the whole definition: nothing outside it carries the stage token, nothing inside it is missing
    names = [n for n, _ in model.named_parameters()]
    assert {n for n, _ in ref} == {n for n in names if n.startswith("core.encoder.net.layer2.")}
    assert [n for n in names if "layer2" in n and not n.startswith("core.encoder.net.layer2.")] == []
    assert GS.GROUP_PREFIXES["layer2"] == ("core.encoder.net.layer2.",)


def test_the_resnet101_trunk_layer2_prefix_is_E0js_1219584_parameter_stage_layer2():
    """The E1 arms' trunk is `resnet101.a1_in1k` (`stack/ops/runs.d/r9e1_C0.argv.json`). E0j recorded 1,219,584 parameters in
    `stage_layer2` (`raw/E0J_RESULT.json` line 22). Same prefix, same count, on the REAL architecture (random init, CPU)."""
    import timm
    net = timm.create_model("resnet101.a1_in1k", pretrained=False, num_classes=0)
    full = [("core.encoder.net." + n, p) for n, p in net.named_parameters()]
    by_prefix = [(n, p) for n, p in full if n.startswith(GS.GROUP_PREFIXES["layer2"])]
    by_e0j = [(n, p) for n, p in full if gcf.default_group_of(n[len("core.encoder."):]) == "stage_layer2"]
    assert [n for n, _ in by_prefix] == [n for n, _ in by_e0j]
    assert sum(p.numel() for _, p in by_prefix) == 1219584


# =========================================================================== #
# item 2 -- two groups for one parameter                                       #
# =========================================================================== #
def test_layer2_lies_inside_trunk_and_each_group_keeps_its_own_members(rig):
    model, _ = rig
    g = GS.param_groups(model)
    assert list(sorted(g)) == ["bev025", "bev_pool", "layer2", "trunk"]
    trunk_ids = [id(p) for p in g["trunk"]]
    assert set(id(p) for p in g["layer2"]) < set(trunk_ids)                 # a PROPER subset: nested, not a replacement
    # trunk is exactly what the tip's first-match grouping gave it: every trainable core.encoder.* parameter, in order
    assert trunk_ids == [id(p) for n, p in model.named_parameters() if p.requires_grad and n.startswith("core.encoder.")]
    assert [id(p) for p in g["bev025"]] == [id(p) for n, p in model.named_parameters()
                                            if p.requires_grad and n.startswith(OLD3["bev025"])]
    assert [id(p) for p in g["bev_pool"]] == [id(p) for n, p in model.named_parameters()
                                              if p.requires_grad and n.startswith(OLD3["bev_pool"])]
    assert len(g["trunk"]) == 64 and len(g["bev025"]) == 18 and len(g["bev_pool"]) == 3 and len(g["layer2"]) == 15


def test_each_group_is_read_on_its_own_params(rig):
    """||g_total||^2 of the layer2 group is the sum over ITS OWN 15 tensors only (independent recomputation), is smaller than
    the trunk's (a subset of its coordinates), and the linearity control closes in both groups."""
    model, losses = rig
    row = _read(model, losses)
    own = [p for _, p in _pgrad_gram_layer2(model)]
    g_tot = _vec(losses["loss"], own)
    assert row["gs_layer2_norm_total"] == pytest.approx(float(g_tot.norm()), rel=1e-9)
    assert 0.0 < row["gs_layer2_norm_total"] < row["gs_trunk_norm_total"]
    assert row["gs_layer2_lin_rel_err"] <= 1e-6 and row["gs_trunk_lin_rel_err"] <= 1e-6
    for grp in ("layer2", "trunk"):                                          # shares close to 1 in EACH group
        assert sum(v for k, v in row.items() if k.startswith(f"gs_{grp}_proj_")) == pytest.approx(1.0, abs=1e-6)


# =========================================================================== #
# EQUALITY (the analytic check), NON-ZERO, CONTROL                             #
# =========================================================================== #
def test_inline_share_formula_is_gram_cores():
    f = E0 / "gram_core.py"
    if not f.is_file():
        pytest.skip(f"{f} is not in this tree: the inline copy of s_i(w) stays unpinned to its source")
    spec = importlib.util.spec_from_file_location("gs_layer2_gram_core", f)
    gc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gc)
    rng = np.random.default_rng(0)
    for _ in range(5):
        a = rng.normal(size=(3, 7))
        gm = a @ a.T
        w = rng.uniform(0.1, 3.0, size=3)
        assert np.array_equal(_share_inline(gm, w), gc.share_from_gram(gm, w))


def test_EQUALITY_the_layer2_reading_is_E0js_formula_on_the_same_gradients(rig):
    model, losses = rig
    assert _equality_problems(model, losses) == []
    # NOT a relabelled trunk: the two groups read DIFFERENT map shares on this batch (E0j: 0.1459 vs 0.0659 on Thor)
    row = _read(model, losses)
    assert abs(row["gs_layer2_proj_map_hires"] - row["gs_trunk_proj_map_hires"]) > 1e-4


def test_NONZERO_when_the_map_has_weight_and_the_CONTROL_reads_exactly_zero_without_it(rig):
    model, losses = rig
    row = _read(model, losses)
    assert row["gs_layer2_proj_map_hires"] > 0.0 and row["gs_layer2_norm_map_hires"] > 0.0
    # CONTROL: the SAME reading with the map term at weight 0.0 (attached, so a key exists) -> EXACTLY 0.0 in layer2 AND trunk
    w_traj = GS.weights_of(model, T.TRAJ_WEIGHT)["traj"]
    terms = {"traj": w_traj * losses["traj"], "map_hires": 0.0 * losses["map_hires"]}
    groups = GS.param_groups(model)
    ctl = GS.measure(terms, terms["traj"] + terms["map_hires"], groups)
    for grp in ("layer2", "trunk"):
        assert ctl[f"gs_{grp}_proj_map_hires"] == 0.0 and ctl[f"gs_{grp}_norm_map_hires"] == 0.0
        assert ctl[f"gs_{grp}_norm_total"] > 0.0                             # the 0.0 is a measurement, not a 0/0
        assert ctl[f"gs_{grp}_proj_traj"] == pytest.approx(1.0, abs=1e-9)
    # and the TRAINER's own path (term_tensors) drops a weight-0 term entirely: there is no key to misread as a share
    assert "map_hires" not in GS.term_tensors(losses, {"traj": w_traj, "map_hires": 0.0})


# =========================================================================== #
# OFF -- the groups that existed before read what they read before             #
# =========================================================================== #
def test_the_three_original_group_prefixes_are_unchanged():
    assert {k: v for k, v in GS.GROUP_PREFIXES.items() if k != "layer2"} == OLD3
    assert list(GS.GROUP_PREFIXES) == ["trunk", "bev025", "bev_pool", "layer2"]          # layer2 appended, nothing reordered


def test_OFF_every_pre_existing_key_is_bit_identical_and_the_new_keys_are_exactly_layer2(rig):
    model, losses = rig
    old = _read(model, losses, OLD3)                                          # the reading as the tip makes it
    new = _read(model, losses)
    shared = set(old) - TIMING_KEYS
    # bit-for-bit by the IEEE-754 pattern, not `==`: a group with no gradient reads NaN (0/0), and NaN != NaN
    diff = [k for k in shared if struct.pack("<d", old[k]) != struct.pack("<d", new[k])]
    assert shared and diff == [], diff
    assert math.isnan(old["gs_bev_pool_proj_traj"]) and math.isnan(new["gs_bev_pool_proj_traj"])   # MEASURED: bev_pool has no gradient on this rig
    assert set(new) - set(old) == {f"gs_layer2_{kind}_{t}" for kind in ("proj", "norm") for t in ("traj", "map_hires", "rest")} | {
        "gs_layer2_norm_total", "gs_layer2_lin_rel_err"}
    assert old["gs_n_passes"] == new["gs_n_passes"] == 4.0                    # no extra autograd PASS: layer2 adds entries, not passes


# =========================================================================== #
# DELIBERATE REGRESSIONS -- each must turn the equality check RED               #
# =========================================================================== #
def test_DELIBERATE_REGRESSION_deleting_the_layer2_line_goes_RED(rig):
    model, losses = rig
    no_layer2 = {k: v for k, v in GS.GROUP_PREFIXES.items() if k != "layer2"}
    assert _equality_problems(model, losses, no_layer2) == ["no `layer2` group"]


def test_DELIBERATE_REGRESSION_a_wrong_prefix_goes_RED(rig):
    model, losses = rig
    wrong = dict(GS.GROUP_PREFIXES, layer2=("core.encoder.net.layer3.",))
    assert _equality_problems(model, losses, wrong) != []


def test_DELIBERATE_REGRESSION_the_tips_first_match_break_leaves_layer2_empty(rig):
    """The tip's `param_groups` stops at the first matching prefix; `trunk` matches first, so `layer2` (listed after it) gets
    NOTHING -- the group would be declared and unreachable. This is the line that had to go."""
    model, _ = rig
    g: dict = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        for name, pre in GS.GROUP_PREFIXES.items():
            if n.startswith(pre):
                g.setdefault(name, []).append(p)
                break
    assert "layer2" not in g and len(g["trunk"]) == 64


# =========================================================================== #
# REACHABILITY from the caller (built, tested and unreachable is a class)      #
# =========================================================================== #
def test_the_trainer_builds_its_groups_with_the_default_prefixes_and_merges_every_key():
    src = (ROOT / "scripts" / "refc_v3_train.py").read_text(encoding="utf-8")
    assert "_gs_groups = _gshare.param_groups(model) if _gs_every > 0 else {}" in src      # defaults -> includes layer2
    assert "row.update(_gs_row)" in src                                       # every gs_* key of the reading reaches the row
    assert src.count("_gshare.param_groups(") == 1                            # one call site: no caller overrides the prefixes
    assert src.count("_gshare.measure(") == 1 and src.count("_gshare.fp32_replay(") == 1
