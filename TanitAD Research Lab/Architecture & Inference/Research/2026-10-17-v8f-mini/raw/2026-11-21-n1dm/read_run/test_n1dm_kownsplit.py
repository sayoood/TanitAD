"""test_n1dm_kownsplit.py -- tests of n1dm_kownsplit.py (K-OWN-SPLIT, PREREG_V8F_N1DM 2.3 iii) on the tree's synthetic world (no data, CPU, 1 thread).
Run: python -I -m pytest -q -p no:cacheprovider test_n1dm_kownsplit.py     (PYTHONPATH = tree/stack;tree/taniteval;tree/stack/scripts;tree/stack/tests ; tanitad must resolve inside the tree)
T1   the record is VALID on a real built route model, modes and cond: every control passes
T2   K-ID / K-MASK / K-UNIF / K-INFER read their known values (0.0 / the has-mask / the same class share and a gap of exactly 0.0 / 0.0)
T3   a model with no route head is a REFUSAL (no number), and so is code_mix kind
T4   PLANTED SHORTCUT: a stand-in model that is right only where it reads the GT code gives own acc 0.0, GT acc 1.0, gap exactly +1.0, n_pair_rows = the has&ok rows
T5   MUTATIONS (each a source edit of the tool that must turn a NAMED test RED): GT_P -> 0.0 (K-MASK FAILS, status INVALID); GT_P -> NATURAL_P (K-MASK FAILS); the pair mask `ok & has` -> `ok` (T4's row count RED)
"""
import copy
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent
TREE = HERE / "tree"
for p in (TREE / "stack" / "tests", TREE / "stack" / "scripts", TREE / "taniteval", TREE / "stack", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import tanitad  # noqa: E402
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
import v8f_mini_train as TR  # noqa: E402
from test_v8f_n1dh import build, set_route_emb  # noqa: E402
from test_v8f_n1dw import load_J, make_batch, write_nav_npz, write_rbd_npz  # noqa: E402
import n1dm_kownsplit as K  # noqa: E402

assert Path(tanitad.__file__).resolve().is_relative_to(TREE.resolve()), tanitad.__file__
CI = K._ci_module(str(TREE).replace("\\", "/"))


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("kownworld")
    w = D.make_synthetic_world(root, n_train=6, n_eval=2, T=60, seed=0)
    J = D.load_join(Path(root) / "join_train.npz")
    return {"root": root, "rbd": write_rbd_npz(root / "rbd.npz", J), "nav": write_nav_npz(root / "nav.npz", J), **w}


def probe(world, n=48, seed=0, has_frac=0.5):
    """a collated batch of n windows with the loop's own route labels; lab_t.has set on a seeded half so the GT arm differs from the OWN arm on SOME rows only; episode ids = clip index"""
    b, J, _ = make_batch(world, n=n, seed=seed, with_rb=False)
    y, ok = TR.route_stage(b)
    b = dict(b)
    lt = dict(b["lab_t"])
    keep = torch.as_tensor(np.random.RandomState(seed).rand(n) < has_frac)
    lt["has"] = lt["has"].bool() & keep
    b["lab_t"] = lt
    return b, y.numpy(), ok.numpy(), b["ci"].numpy()


def model_of(world, mode):
    b, J, _ = make_batch(world, n=4, with_rb=False)
    m = build(J, route=True, aux_route=mode)
    with torch.no_grad():
        m.route_head.weight.copy_(torch.randn(m.route_head.weight.shape, generator=torch.Generator().manual_seed(5)) * 0.2)
        m.route_head.bias.copy_(torch.randn(m.route_head.bias.shape, generator=torch.Generator().manual_seed(6)) * 0.3)
    if m.route_emb is not None:
        set_route_emb(m)
    if m.route_br is not None:
        for i, br in enumerate(m.route_br):
            with torch.no_grad():
                br.weight.copy_(torch.randn(br.weight.shape, generator=torch.Generator().manual_seed(20 + i)) * 0.05)
    return m


@pytest.mark.parametrize("mode", ["cond", "modes"])
def test_t1_t2_the_record_is_valid_and_every_control_reads_its_known_value(world, mode):
    m = model_of(world, mode)
    b, y, ok, eid = probe(world)
    r = K.read_split(m, [b], y, ok, eid, CI, n_boot=100, perm_tol=0.3)
    assert r["status"] == "VALID", r["controls"]
    C = r["controls"]
    assert C["K-ID"]["max_abs_logit_diff_two_own_runs"] == 0.0 and C["K-ID"]["paired_delta"] == 0.0
    has = b["lab_t"]["has"].bool().numpy()
    assert 0 < has.sum() < len(has), "the probe must carry both has and not-has rows"
    assert C["K-MASK"]["gt_arm_own_equals_not_has"] and C["K-MASK"]["own_arm_all_true"] and C["K-MASK"]["n_has"] == int(has.sum())
    assert C["K-UNIF"]["gap_gt_minus_own"] == 0.0 and C["K-UNIF"]["argmax_class_counts_own"] == C["K-UNIF"]["argmax_class_counts_gt"] and sum(1 for c in C["K-UNIF"]["argmax_class_counts_own"] if c) == 1
    assert C["K-INFER"]["max_abs_logit_diff_own_arm_vs_forward_infer"] == 0.0
    assert r["n_pair_rows"] == int((ok & has).sum()) and r["n_has"] == int(has.sum())
    assert r["natural_draw"]["gt_code_rows"]["n_rows"] + r["natural_draw"]["own_code_rows"]["n_rows"] == int(ok.sum())
    assert r["max_abs_logit_diff_gt_vs_own_on_has_rows"] > 0.0, "the GT code must reach the route head's input h on has rows (else the diagnostic is vacuous)"
    assert r["max_abs_logit_diff_gt_vs_own"] == r["max_abs_logit_diff_gt_vs_own_on_has_rows"], "rows without a T label read T's own code in both arms: identical logits there"


def test_t3_no_route_head_and_code_mix_kind_are_refusals_never_numbers(world):
    b, J, _ = make_batch(world, n=8, with_rb=False)
    m0 = build(J, route=False)
    r = K.read_split(m0, [b], np.zeros(8, int), np.ones(8, bool), np.arange(8), CI, n_boot=20)
    assert r["status"].startswith("REFUSED") and "gt_minus_own" not in r and "arms" not in r
    fake = types.SimpleNamespace(route_head=object(), cfg=types.SimpleNamespace(code_mix="kind"))
    r2 = K.read_split(fake, [b], np.zeros(8, int), np.ones(8, bool), np.arange(8), CI, n_boot=20)
    assert r2["status"].startswith("REFUSED") and "code_mix" in r2["status"]


class Planted(torch.nn.Module):
    """a stand-in with the model surface the tool uses: its route logits point at the LABEL only where the row reads the GT code, and at a wrong class where it reads its own code (a planted shortcut of size exactly 1.0)"""

    def __init__(self, y):
        super().__init__()
        self.y = torch.as_tensor(y).long()
        self.route_head = torch.nn.Linear(2, 3)
        self.cfg = types.SimpleNamespace(code_mix="whole", aux_route="cond", o_interface="full", detach_upper=True)

    def _code_gt(self, lab):
        return torch.ones(lab["has"].shape[0], 4)

    def _forward(self, b, p_gt, use_labels, gen=None, code_override=None):
        has = b["lab_t"]["has"].bool()
        B = has.shape[0]
        use_t = (torch.rand(B, generator=gen) < p_gt) & has if (use_labels and p_gt > 0) else torch.zeros(B, dtype=torch.bool)
        if code_override is not None:
            use_t = has & (code_override[:, 0] == 1.0)
        right = torch.nn.functional.one_hot(self.y, 3).float() * 5.0
        wrong = torch.nn.functional.one_hot((self.y + 1) % 3, 3).float() * 5.0
        return {"route_logits": torch.where(use_t[:, None], right, wrong), "route_own": ~use_t, "code": torch.zeros(B, 4)}

    def forward_infer(self, b):
        return self._forward(b, 0.0, False)


def test_t4_a_planted_shortcut_reads_a_gap_of_exactly_one(world):
    b, y, ok, eid = probe(world, n=48, has_frac=0.5)
    ok = np.ones_like(ok)
    pm = Planted(y)
    r = K.read_split(pm, [b], y, ok, eid, CI, n_boot=100, perm_tol=0.35)
    has = b["lab_t"]["has"].bool().numpy()
    assert r["n_pair_rows"] == int(has.sum()) and 0 < r["n_pair_rows"] < len(y)
    assert r["arms"]["own"]["acc"] == 0.0 and r["arms"]["gt"]["acc"] == 1.0
    assert r["gt_minus_own"]["delta"] == 1.0 and r["gt_minus_own"]["separated"] is True
    assert r["controls"]["K-MASK"]["pass"] and r["controls"]["K-ID"]["pass"]
    assert r["arms"]["tonly"]["acc"] == 1.0, "the isolated T-code arm reads the same planted shortcut"


def _mutant(old, new):
    src = (HERE / "n1dm_kownsplit.py").read_text(encoding="utf-8")
    assert src.count(old) >= 1, old
    mod = types.ModuleType("kown_mut")
    mod.__file__ = str(HERE / "n1dm_kownsplit.py")
    exec(compile(src.replace(old, new), mod.__file__, "exec"), mod.__dict__)
    return mod


@pytest.mark.parametrize("old,new", [("GT_P = 1.0 ", "GT_P = 0.0 "), ("GT_P = 1.0 ", "GT_P = 0.2 ")])
def test_t5_mutation_gt_arm_probability_turns_kmask_red(world, old, new):
    m = model_of(world, "modes")
    b, y, ok, eid = probe(world)
    mod = _mutant(old, new)
    r = mod.read_split(m, [b], y, ok, eid, CI, n_boot=50, perm_tol=0.3)
    assert r["status"].startswith("INVALID") and "K-MASK" in r["status"], r["status"]
    assert r["controls"]["K-MASK"]["pass"] is False


def test_t5_mutation_pair_mask_without_has_turns_the_planted_row_count_red(world):
    b, y, ok, eid = probe(world, n=48, has_frac=0.5)
    ok = np.ones_like(ok)
    mod = _mutant("m = ok & has", "m = ok")
    r = mod.read_split(Planted(y), [b], y, ok, eid, CI, n_boot=50, perm_tol=0.5)
    has = b["lab_t"]["has"].bool().numpy()
    assert r["n_pair_rows"] != int(has.sum()), "the mutant counts rows without a T label: T4's row-count assertion would be RED"
    assert r["gt_minus_own"]["delta"] != 1.0, "and the planted gap is diluted: T4's gap assertion would be RED"


def test_t6_slicing_the_probe_changes_no_argmax_and_no_row_count(world, monkeypatch):
    m = model_of(world, "modes")
    b, y, ok, eid = probe(world, n=48)
    whole = K.run_arms(m, [b])
    monkeypatch.setattr(K, "SLICE", 16)
    cut = K.run_arms(m, [b])
    assert cut["own"].shape == whole["own"].shape == (48, 3) and np.array_equal(cut["has"], whole["has"])
    for k in ("own", "gt", "tonly", "infer"):
        assert np.abs(cut[k] - whole[k]).max() < 1e-4, k
    assert np.array_equal(cut["own"].argmax(-1), whole["own"].argmax(-1)) and np.array_equal(cut["gt_own_mask"], ~whole["has"])
