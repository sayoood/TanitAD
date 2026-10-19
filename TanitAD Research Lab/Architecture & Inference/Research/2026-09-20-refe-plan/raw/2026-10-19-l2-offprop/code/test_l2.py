#!/usr/bin/env python3
"""L2-OFFPROP tests: SYNTHETIC data only (no real label is read), CPU only.   pytest test_l2.py -q     (CUDA_VISIBLE_DEVICES=-1)

Each control the draft names has a test that CAN FAIL:
  C-ID    zero steps -> P2's scores bit for bit; A1 equals L1's own read
  C-REACH planted feature -> A1-OOF >= 0.98 while the SAME world without the plant stays low
  C-LEAK  STRUCTURAL (MM addendum 2026-10-19): `oof_scores` asserts train/held-out disjointness (tokens, logs, and the TrainSet's own index arrays); RED mutations (one held-out token injected, a
          whole fold, a held-out row routed into the pair / in-fan index arrays) must each make it raise; there is no switch that turns it off.  (The statistical test - clean vs leaky A1 on
          shuffled labels - was UNINFORMATIVE for a linear 257-parameter head and is gone.)
  G-LIVE  non-zero grad on the EP row, EXACTLY 0.0 on frozen rows, and the mutated loss reads non-zero (the probe can read a non-zero); the H0 (w_pair = 0) and prox levers REACH the head
  + reachability of every Recipe flag, the GLOBAL H-null permutation (MM ruling E1), arm H0, the pooled-A1 / H0-in-fan plumbing of the read (E2), the seal guard (RED on a draft / wrong sha /
    missing gate / a gate earned on a different recipe or code)
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import l2lib  # noqa: E402
import run_l2  # noqa: E402
import synthetic  # noqa: E402
from l2lib import stepA  # noqa: E402

try:
    import torch
    torch.set_num_threads(2)
except Exception:  # pragma: no cover
    pass

REC = l2lib.Recipe(epochs=12)        # a short recipe: the tests check mechanics and signs, not the final numbers


def _chosen_recipe() -> "l2lib.Recipe":
    """the SELECTED recipe (RECIPE_SELECTION.md): $L2_RECIPE, else the artifact copy, else the staged raw/ copy.  No silent fallback to the source default."""
    cands = [os.environ.get("L2_RECIPE"), "D:/Projects/TanitAD-artifacts/l2_offprop/recipe_chosen.json", os.path.join(HERE, "..", "raw", "recipe_chosen.json")]
    for c in cands:
        if c and os.path.exists(c):
            return l2lib.Recipe.from_json(c)
    pytest.fail("the selected recipe file (recipe_chosen.json) was not found: set L2_RECIPE")


CHOSEN = _chosen_recipe()


@pytest.fixture(scope="module")
def Dsig():
    return synthetic.make_data(n_tokens=72, n_logs=24, seed=3, gamma=0.4, rho=0.4)


@pytest.fixture(scope="module")
def Dnull():
    return synthetic.make_data(n_tokens=72, n_logs=24, seed=3, gamma=0.0, rho=0.4)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- folds
def test_folds_rank_mod_5_over_sorted_names():
    names = [f"log{i:02d}" for i in range(24)][::-1]                 # given in REVERSED order: the fold must not depend on it
    f = l2lib.fold_of_logs(names)
    by = dict(zip(names, f))
    assert [by[f"log{i:02d}"] for i in range(24)] == [i % 5 for i in range(24)]
    assert np.bincount(f).tolist() == [5, 5, 5, 5, 4]


def test_every_token_in_exactly_one_fold_and_log_not_split(Dsig):
    D = Dsig
    assert set(np.unique(D.fold)) == set(range(5))
    for l in range(D.n_logs):
        assert len(set(D.fold[D.log_idx == l])) == 1


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- the training set
def test_trainset_pairs_are_the_a1_pairs_and_index_the_right_items(Dsig):
    D = Dsig
    toks = np.nonzero(D.fold != 1)[0]
    ts = l2lib.make_trainset(D, toks)
    n_a, s_a = ts.ia // 64, ts.ia % 64
    n_p, q_p = ts.ip // D.Q, ts.ip % D.Q
    assert np.array_equal(n_a, n_p) and set(np.unique(n_a)) <= set(toks.tolist())
    assert np.array_equal(D.anchor[n_p, q_p], s_a)                    # the anchor slot of job q
    diff = D.t_p[n_p, q_p] - D.t_in[n_a, s_a]
    assert (np.abs(diff) >= l2lib.THR_PAIR - 1e-12).all() and np.array_equal(np.sign(diff), ts.y)
    # and it is EVERY such pair: count by brute force
    full = np.abs(D.t_p[toks] - D.t_a[toks]) >= l2lib.THR_PAIR - 1e-12
    assert len(ts.ia) == int(full.sum())
    assert len(ts.fan_idx) == 64 * len(toks) and np.allclose(ts.fan_t, D.t_in[toks].reshape(-1))


def test_pair_set_equals_L1_A1_denominator(Dsig):
    """the training pairs over ALL tokens are exactly the pairs in L1's A1 denominator (so `A1` and the loss use one definition)"""
    D = Dsig
    ts = l2lib.make_trainset(D, np.arange(D.N))
    terms = stepA.admission_terms(D.P0)
    assert len(ts.ia) == int(terms["A1"][1].sum())


def test_the_default_null_is_the_global_shuffle(Dsig):
    """MM ruling E1: the H-null is the GLOBAL shuffle; the draft's within-(token, anchor) permutation is withdrawn and exists only as an explicit diagnostic"""
    D = Dsig
    assert l2lib.NULL_GROUP_DEFAULT == "global"
    assert l2lib.ArmSpec("n", "Hnull", null_rep=100).null_group == "global"
    g = l2lib.permute_pert_labels(D, 100)
    assert np.array_equal(g, l2lib.permute_pert_labels(D, 100, "global"))
    assert np.allclose(np.sort(g.reshape(-1)), np.sort(D.t_p.reshape(-1)))             # a permutation of ALL labels
    moved_groups = sum(not np.allclose(np.sort(g[n, D.anchor[n] == a]), np.sort(D.t_p[n, D.anchor[n] == a])) for n in range(D.N) for a in np.unique(D.anchor[n]))
    assert moved_groups > 0.9 * D.N * 16                                               # it breaks the (token, anchor) group marginals; the anchor-null never does
    from run_l2 import all_cells, null_cell_name
    names = [c for c in all_cells() if c.startswith("Hnull")]
    assert len(names) == 10 and all("-global_" in c for c in names) and null_cell_name(100, 0) == "Hnull-global_r100_s0"


def test_hnull_anchor_group_permutation_is_a_diagnostic_that_permutes_within_token_anchor_groups_only(Dsig):
    D = Dsig
    perm = l2lib.permute_pert_labels(D, 100, "anchor")
    changed = 0
    for n in range(D.N):
        for a in np.unique(D.anchor[n]):
            q = np.nonzero(D.anchor[n] == a)[0]
            assert len(q) == 6
            assert np.allclose(np.sort(perm[n, q]), np.sort(D.t_p[n, q]))          # marginals intact inside the group
            changed += int(not np.array_equal(perm[n, q], D.t_p[n, q]))
    assert changed > 0.5 * D.N * 16 * 0.5
    assert np.allclose(np.sort(perm.reshape(-1)), np.sort(D.t_p.reshape(-1)))     # and globally
    assert np.array_equal(perm, l2lib.permute_pert_labels(D, 100, "anchor"))       # pure function of the seed
    assert not np.array_equal(perm, l2lib.permute_pert_labels(D, 101, "anchor"))


def test_hnull_trainset_differs_from_h_and_keeps_marginals(Dsig):
    D = Dsig
    toks = np.arange(D.N)
    ts_h = l2lib.make_trainset(D, toks)
    ts_n = l2lib.make_trainset(D, toks, l2lib.permute_pert_labels(D, 100))
    assert not np.array_equal(ts_h.y, ts_n.y[:len(ts_h.y)]) or len(ts_h.y) != len(ts_n.y)
    assert np.array_equal(ts_h.fan_idx, ts_n.fan_idx) and np.allclose(ts_h.fan_t, ts_n.fan_t)    # the in-fan BCE labels are untouched


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- C-LEAK
def test_assert_no_leak_is_red_on_overlap(Dsig):
    D = Dsig
    te = np.nonzero(D.fold == 2)[0]
    tr = np.nonzero(D.fold != 2)[0]
    l2lib.assert_no_leak(D, tr, te)
    with pytest.raises(RuntimeError, match="C-LEAK"):
        l2lib.assert_no_leak(D, np.append(tr, te[0]), te)                       # ONE held-out token in training
    other_token_same_log = np.nonzero((D.log_idx == D.log_idx[te[0]]) & (np.arange(D.N) != te[0]))[0]
    if len(other_token_same_log):
        with pytest.raises(RuntimeError, match="C-LEAK"):
            l2lib.assert_no_leak(D, np.append(tr, other_token_same_log[0]), te[te != other_token_same_log[0]])   # a LOG shared across the split


def test_oof_scores_trip_on_the_leak_mutation(Dsig):
    with pytest.raises(RuntimeError, match="C-LEAK"):
        l2lib.oof_scores(Dsig, l2lib.ArmSpec("leak", "H", seed=0, leak_fold=1), REC)


def test_oof_scores_trip_on_ONE_injected_held_out_token_in_every_fold(Dsig):
    """the RED mutation the MM addendum names: inject ONE held-out token into training => the assertion fires (before any optimiser step)"""
    for f in range(l2lib.N_FOLDS):
        with pytest.raises(RuntimeError, match="C-LEAK"):
            l2lib.fold_split(Dsig, f, l2lib.ArmSpec("m", "H", leak_fold=f, leak_n=1))
    with pytest.raises(RuntimeError, match="C-LEAK"):
        l2lib.oof_scores(Dsig, l2lib.ArmSpec("m", "H0", seed=0, leak_fold=3, leak_n=1), REC)       # H0, H-null and the planted arm share the code path
    with pytest.raises(RuntimeError, match="C-LEAK"):
        l2lib.oof_scores(Dsig, l2lib.ArmSpec("m", "Hnull", seed=0, null_rep=100, leak_fold=0, leak_n=1), REC)
    with pytest.raises(RuntimeError, match="C-LEAK"):
        l2lib.oof_scores(Dsig, l2lib.ArmSpec("m", "planted", seed=0, leak_fold=4, leak_n=1), REC)


def test_clean_split_records_a_structural_check_per_fold_and_nothing_can_switch_it_off(Dsig):
    assert "check_leak" not in l2lib.ArmSpec.__dataclass_fields__                            # the old off-switch is gone: the check fails closed
    o = l2lib.oof_scores(Dsig, l2lib.ArmSpec("c", "H", seed=0), l2lib.Recipe(epochs=2))
    for f in o["diag"]["folds"]:
        assert f["leak_check"]["tokens_overlap"] == 0 and f["leak_check"]["logs_overlap"] == 0
        assert f["leak_check"]["trainset"]["held_out_tokens_referenced"] == 0
        assert f["leak_check"]["trainset"]["tokens_used_by_trainset"] == f["n_train_tokens"]


def test_trainset_level_check_catches_a_held_out_row_routed_past_clean_token_lists(Dsig, monkeypatch):
    """C-LEAK reach: the token LISTS are clean but `make_trainset` emitted an index into a held-out token - the check on the arrays that FEED the optimiser must catch it"""
    D = Dsig
    orig = l2lib.make_trainset
    held = int(np.nonzero(D.fold == 1)[0][0])

    def into_pairs(D_, tokens, t=None):
        ts = orig(D_, tokens, t)
        ts.ip = ts.ip.copy()
        ts.ip[0] = held * D_.Q
        return ts

    def into_fan(D_, tokens, t=None):
        ts = orig(D_, tokens, t)
        ts.fan_idx = ts.fan_idx.copy()
        ts.fan_idx[0] = held * l2lib.N_ANCHOR_SLOTS
        return ts

    def into_anchor(D_, tokens, t=None):
        ts = orig(D_, tokens, t)
        ts.ia = ts.ia.copy()
        ts.ia[0] = held * l2lib.N_ANCHOR_SLOTS + 3
        return ts
    for bad in (into_pairs, into_fan, into_anchor):
        monkeypatch.setattr(l2lib, "make_trainset", bad)
        with pytest.raises(RuntimeError, match="C-LEAK"):
            l2lib.fold_split(D, 1, l2lib.ArmSpec("m", "H"))
    monkeypatch.setattr(l2lib, "make_trainset", orig)
    l2lib.fold_split(D, 1, l2lib.ArmSpec("m", "H"))                                           # and the unmutated path is clean again


def test_oof_scores_requires_every_token_scored_exactly_once(Dsig, monkeypatch):
    orig = l2lib.fold_split

    def drops_one(D, f, arm, t=None):
        tr, te, ts, chk = orig(D, f, arm, t)
        return tr, (te[:-1] if f == 3 else te), ts, chk                                         # one token never scored by an out-of-fold model
    monkeypatch.setattr(l2lib, "fold_split", drops_one)
    with pytest.raises(RuntimeError, match="scored exactly once"):
        l2lib.oof_scores(Dsig, l2lib.ArmSpec("m", "H", seed=0), l2lib.Recipe(epochs=1))


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- the head and G-LIVE
def _flat(D):
    return D.tok_in.reshape(-1, D.D), D.tok_p.reshape(-1, D.D)


def test_zero_steps_return_p2_bit_for_bit(Dsig):
    D = Dsig
    Xi, Xp = _flat(D)
    ts = l2lib.make_trainset(D, np.nonzero(D.fold != 0)[0])
    fit = l2lib.fit_head(Xi, Xp, ts, D.W6, D.b6, 0, REC, steps=0)
    assert fit["steps"] == 0 and np.array_equal(fit["W"], D.W6) and np.array_equal(fit["b"], D.b6)


def test_training_moves_only_the_ep_row_and_is_seeded(Dsig):
    D = Dsig
    Xi, Xp = _flat(D)
    ts = l2lib.make_trainset(D, np.nonzero(D.fold != 0)[0])
    f0 = l2lib.fit_head(Xi, Xp, ts, D.W6, D.b6, 0, REC, steps=25)
    f0b = l2lib.fit_head(Xi, Xp, ts, D.W6, D.b6, 0, REC, steps=25)
    f1 = l2lib.fit_head(Xi, Xp, ts, D.W6, D.b6, 1, REC, steps=25)
    frozen = [r for r in range(6) if r != l2lib.EP]
    for f in (f0, f1):
        assert np.array_equal(f["W"][frozen], D.W6[frozen]) and np.array_equal(f["b"][frozen], D.b6[frozen])      # bit-identical
        assert not np.array_equal(f["W"][l2lib.EP], D.W6[l2lib.EP])                                              # and the EP row moved
    assert np.array_equal(f0["W"], f0b["W"])                                                                      # same seed, same head
    assert not np.array_equal(f0["W"][l2lib.EP], f1["W"][l2lib.EP])                                              # the seed matters


def test_g_live_passes_and_its_control_reads_nonzero(Dsig):
    D = Dsig
    Xi, Xp = _flat(D)
    ts = l2lib.make_trainset(D, np.nonzero(D.fold != 0)[0])
    g = l2lib.g_live(Xi, Xp, ts, D.W6, D.b6, REC)
    assert g["pass"] and g["control_mutation_reads_nonzero_on_frozen"]
    assert g["h0_lever_reaches"] and g["prox_lever_reaches"]
    assert all(r["w_pair0_grad_equals_bce_only_bitwise"] and r["w_pair1_grad_differs_from_w_pair0"] for r in g["h0"])
    assert all(r["zero_at_init"] and r["reaches_ep_row_off_init"] for r in g["prox"])
    clean = [r for r in g["batches"] if not r["mutated"]]
    assert all(r["frozen_W_grad_abs_max"] == 0.0 and r["frozen_b_grad_abs_max"] == 0.0 for r in clean)
    assert all(r["ep_W_grad_abs_sum"] > 0 and r["ep_b_grad_abs"] > 0 and r["finite"] for r in clean)


def test_g_live_mutation_goes_red_inside_training(Dsig):
    D = Dsig
    Xi, Xp = _flat(D)
    ts = l2lib.make_trainset(D, np.nonzero(D.fold != 0)[0])
    with pytest.raises(RuntimeError, match="FROZEN row"):
        l2lib.fit_head(Xi, Xp, ts, D.W6, D.b6, 0, REC, steps=3, loss_mutation="leak_other_row")


def test_every_recipe_flag_reaches_the_optimiser(Dsig):
    """the CLAUDE.md class 'built, tested, unreachable from its caller': each Recipe field must change the trained head"""
    D = Dsig
    Xi, Xp = _flat(D)
    ts = l2lib.make_trainset(D, np.nonzero(D.fold != 0)[0])
    base = l2lib.fit_head(Xi, Xp, ts, D.W6, D.b6, 0, l2lib.Recipe(epochs=4))["W"][l2lib.EP]
    same = l2lib.fit_head(Xi, Xp, ts, D.W6, D.b6, 0, l2lib.Recipe(epochs=4))["W"][l2lib.EP]
    assert np.array_equal(base, same)
    for field, val in (("lr", 1e-4), ("epochs", 2), ("batch", 128), ("lam_bce", 3.0), ("prox", 0.3)):
        w = l2lib.fit_head(Xi, Xp, ts, D.W6, D.b6, 0, l2lib.Recipe(**{**dict(epochs=4), field: val}))["W"][l2lib.EP]
        assert not np.array_equal(w, base), f"Recipe.{field} did not reach the trained head"
    w0 = l2lib.fit_head(Xi, Xp, ts, D.W6, D.b6, 0, l2lib.Recipe(epochs=4, lr=0.0))["W"][l2lib.EP]
    assert np.array_equal(w0, D.W6[l2lib.EP])                                      # lr = 0 is a no-op
    # the proximal weight pulls toward P2's head: a large one leaves the head closer to P2's than none does
    far = lambda w: float(np.linalg.norm(w - D.W6[l2lib.EP]))
    w_free = l2lib.fit_head(Xi, Xp, ts, D.W6, D.b6, 0, l2lib.Recipe(epochs=6, lr=2e-3))["W"][l2lib.EP]
    w_prox = l2lib.fit_head(Xi, Xp, ts, D.W6, D.b6, 0, l2lib.Recipe(epochs=6, lr=2e-3, prox=5.0))["W"][l2lib.EP]
    assert far(w_prox) < far(w_free)
    # an unknown recipe field is refused (a silently ignored lever is the 'built, tested, unreachable' class)
    with pytest.raises(ValueError, match="unknown fields"):
        import tempfile
        pth = os.path.join(tempfile.mkdtemp(), "r.json")
        json.dump({"recipe": {"lr": 1e-3, "epochs": 3, "proximal": 1.0}}, open(pth, "w"))
        l2lib.Recipe.from_json(pth)


def test_arm_h0_pairwise_term_is_dead_and_everything_else_is_the_same_recipe(Dsig):
    """H0 = the SAME recipe with the pairwise weight 0: flipping every pair label must not change the trained head AT ALL (bit-identical), while it does change H's"""
    D = Dsig
    Xi, Xp = _flat(D)
    ts = l2lib.make_trainset(D, np.nonzero(D.fold != 0)[0])
    flipped = l2lib.TrainSet(ts.ia, ts.ip, -ts.y, ts.fan_idx, ts.fan_t, ts.tokens)
    rec = l2lib.Recipe(epochs=4, lr=2e-3, prox=0.05)
    a = l2lib.fit_head(Xi, Xp, ts, D.W6, D.b6, 0, rec, w_pair=0.0)
    b = l2lib.fit_head(Xi, Xp, flipped, D.W6, D.b6, 0, rec, w_pair=0.0)
    assert np.array_equal(a["W"], b["W"]) and a["steps"] == b["steps"] > 0               # pair labels are irrelevant at w_pair = 0
    h = l2lib.fit_head(Xi, Xp, ts, D.W6, D.b6, 0, rec)
    hf = l2lib.fit_head(Xi, Xp, flipped, D.W6, D.b6, 0, rec)
    assert not np.array_equal(h["W"], hf["W"]) and not np.array_equal(h["W"], a["W"])    # ... and at w_pair = 1 they are not
    assert h["steps"] == a["steps"]                                                       # the SAME minibatch stream: H0 takes as many steps as H
    # through oof_scores: H0 is not H, and its zero-step arm is P2 bit for bit
    o_h = l2lib.oof_scores(D, l2lib.ArmSpec("h", "H", seed=0), rec)
    o_h0 = l2lib.oof_scores(D, l2lib.ArmSpec("h0", "H0", seed=0), rec)
    assert not np.array_equal(o_h["ep_p"], o_h0["ep_p"])
    p2 = l2lib.oof_scores(D, l2lib.ArmSpec("p2", "P2"), rec)
    z = l2lib.oof_scores(D, l2lib.ArmSpec("z", "H0", seed=1, steps=0), rec)
    assert np.array_equal(z["ep_in"], p2["ep_in"]) and np.array_equal(z["ep_p"], p2["ep_p"])


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- C-ID, C-REACH, the read
def test_c_id_zero_steps_equals_p2_and_l1s_own_read(Dsig):
    D = Dsig
    p2 = l2lib.oof_scores(D, l2lib.ArmSpec("P2", "P2"), REC)
    h0 = l2lib.oof_scores(D, l2lib.ArmSpec("H0", "H", seed=0, steps=0), REC)
    assert np.array_equal(h0["ep_in"], p2["ep_in"]) and np.array_equal(h0["ep_p"], p2["ep_p"])               # bit for bit
    ev = l2lib.evaluate_arm(D, h0["ep_in"], h0["ep_p"])
    # synthetic banked EP column := ep_apply of the same head, so L1's analyse on the banked logits must be reproduced EXACTLY
    P, li, _ = stepA.assemble_pairs(D.bundle, D.perts, D.scores, D.labels, D.log_names)
    ref = stepA.analyse(P, li, D.n_logs, l2lib.BOOT_B, l2lib.BOOT_SEED)
    for tag in ("all", "lon", "lat"):
        for k in ("A1", "A2", "A3", "A4", "A5"):
            for f in ("point", "lo", "hi", "den"):
                assert np.isclose(ev[tag][k][f], ref[tag][k][f], rtol=0, atol=1e-12, equal_nan=True), (tag, k, f)   # A2 (TTC) has no pairs in the synthetic world: NaN == NaN


def test_c_reach_planted_feature_gives_a1_oof_ge_098_while_the_unplanted_world_stays_low(Dnull):
    D = Dnull                                           # gamma = 0: the token carries NO within-slot ordering
    pl = l2lib.oof_scores(D, l2lib.ArmSpec("pl", "planted", seed=0), REC)
    a1_pl = l2lib.evaluate_arm(D, pl["ep_in"], pl["ep_p"])["all"]["A1"]["point"]
    un = l2lib.oof_scores(D, l2lib.ArmSpec("un", "H", seed=0), REC)
    a1_un = l2lib.evaluate_arm(D, un["ep_in"], un["ep_p"])["all"]["A1"]["point"]
    assert a1_pl >= 0.98, a1_pl
    assert a1_un < 0.80, a1_un                          # no plant, no signal: the read path does NOT manufacture ordering


def test_h_learns_a_planted_within_slot_signal_and_hnull_does_not(Dsig):
    D = Dsig
    p2 = l2lib.oof_scores(D, l2lib.ArmSpec("P2", "P2"), REC)
    ev2 = l2lib.evaluate_arm(D, p2["ep_in"], p2["ep_p"])
    h = l2lib.oof_scores(D, l2lib.ArmSpec("H", "H", seed=0), REC)
    evh = l2lib.evaluate_arm(D, h["ep_in"], h["ep_p"], l2lib.ref_terms_of(ev2))
    n = l2lib.oof_scores(D, l2lib.ArmSpec("N", "Hnull", seed=0, null_rep=100, null_group="global"), REC)
    evn = l2lib.evaluate_arm(D, n["ep_in"], n["ep_p"])
    a1_2, a1_h, a1_n = ev2["all"]["A1"]["point"], evh["all"]["A1"]["point"], evn["all"]["A1"]["point"]
    assert a1_h > a1_2 + 0.05, (a1_2, a1_h)
    assert evh["all"]["A1"]["lo"] > a1_n, (evh["all"]["A1"], a1_n)
    assert a1_n < a1_2 + 0.03, (a1_2, a1_n)             # the globally-shuffled null does not manufacture concordance
    assert evh["A1_minus_P2_paired"]["lo"] > 0


def test_documented_caveat_the_drafts_anchor_group_null_keeps_between_group_signal():
    """ESCALATION 1 (READY.md): permuting the EP labels only WITHIN each (token, anchor) group keeps the between-group part of the label (here: the EP ceiling of a high-EP anchor, shared by its six
    perturbations), which a head can learn from the token.  On this synthetic world the draft's null therefore lands ABOVE the global-shuffle null (and above P2), so the draft's VOID rule can fire
    for a reason that is not an instrument fault.  This test pins the measurement; if it ever stops holding, the escalation text must be revisited."""
    D = synthetic.make_data(n_tokens=72, n_logs=24, seed=3, gamma=0.4, rho=0.4)

    def a1(group):
        o = l2lib.oof_scores(D, l2lib.ArmSpec("n", "Hnull", seed=0, null_rep=100, null_group=group), REC)
        return l2lib.evaluate_arm(D, o["ep_in"], o["ep_p"])["all"]["A1"]["point"]

    assert a1("anchor") > a1("global") + 0.03


def test_infan_top8_concordance_matches_a_brute_force_selrank_implementation(Dsig):
    D = Dsig
    p2 = l2lib.oof_scores(D, l2lib.ArmSpec("P2", "P2"), REC)
    num, den = l2lib.infan_top8_terms(D, p2["ep_in"])
    nb = db = 0.0
    for i in range(D.N):
        idx = D.top8[i]
        for a in idx:
            for b in idx:
                if D.t_in[i, a] - D.t_in[i, b] > 0.05:
                    db += 1
                    d = p2["ep_in"][i, a] - p2["ep_in"][i, b]
                    nb += 1.0 if d > 0 else (0.5 if d == 0 else 0.0)
    assert num.sum() == nb and den.sum() == db and db > 0


def test_speed_scale_pairs_partition_the_lon_a1_denominator(Dsig):
    D = Dsig
    p2 = l2lib.oof_scores(D, l2lib.ArmSpec("P2", "P2"), REC)
    ev = l2lib.evaluate_arm(D, p2["ep_in"], p2["ep_p"])
    assert sum(v["den"] for v in ev["lon_A1_by_speed_scale"].values()) == ev["lon"]["A1"]["den"]
    assert ev["strategic"].startswith("PAUSED")


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- the seal guard
_GATE_DIRS = iter(range(10_000))


def _recipe_file(tmp_path, name="recipe.json", **kw):
    p = tmp_path / name
    p.write_text(json.dumps({"recipe": l2lib.Recipe(**kw).as_dict()}))
    return str(p)


def _gates(tmp_path, recipe, ok=True, skip=None, stamp_recipe=None, stamp_code=None):
    """a gates dir whose PASS files carry the stamp the real gates write.  stamp_recipe / stamp_code override it (a gate earned on a different recipe / different code)"""
    g = tmp_path / f"gates{next(_GATE_DIRS)}"                    # a fresh directory per call: a stale PASS file must not leak between cases
    g.mkdir()
    st = l2lib.gate_stamp(recipe)
    if stamp_recipe is not None:
        st["recipe_sha256"] = stamp_recipe
    if stamp_code is not None:
        st["code_sha256"] = stamp_code
    for name in run_l2.REQUIRED_GATES:
        if name != skip:
            (g / f"{name}.json").write_text(json.dumps({"pass": ok, **st}))
    return str(g)


def test_seal_guard_refuses_a_draft_a_wrong_sha_a_missing_gate_and_a_gate_earned_on_another_recipe(tmp_path):
    rec = _recipe_file(tmp_path)
    draft = tmp_path / "DRAFT.md"
    draft.write_text("# DRAFT PREREG L2-OFFPROP\n\n**Status: DRAFT, nothing run.**\n")
    sha = l2lib.sha256_file(str(draft))
    with pytest.raises(SystemExit, match="SEAL REFUSED"):
        run_l2.check_seal(str(draft), sha, _gates(tmp_path, rec), rec)                       # the draft marker
    sealed = tmp_path / "SEALED.md"
    sealed.write_text("# PREREG L2-OFFPROP\n\n**Status: SEALED 2026-10-19.**\n")
    sha2 = l2lib.sha256_file(str(sealed))
    with pytest.raises(SystemExit, match="SEAL REFUSED"):
        run_l2.check_seal(str(sealed), "0" * 64, _gates(tmp_path, rec), rec)                 # wrong sha
    with pytest.raises(SystemExit, match="SEAL REFUSED"):
        run_l2.check_seal(str(sealed), sha2, _gates(tmp_path, rec, skip="reach"), rec)       # a gate without a PASS
    with pytest.raises(SystemExit, match="SEAL REFUSED"):
        run_l2.check_seal(str(sealed), sha2, _gates(tmp_path, rec, ok=False), rec)           # gates that FAILED
    with pytest.raises(SystemExit, match="different recipe"):
        run_l2.check_seal(str(sealed), sha2, _gates(tmp_path, rec, stamp_recipe="0" * 64), rec)   # PASSes earned on ANOTHER recipe file (the chain's C-REACH was bound to a recipe later replaced)
    with pytest.raises(SystemExit, match="different recipe"):
        run_l2.check_seal(str(sealed), sha2, _gates(tmp_path, rec, stamp_code={"l2lib.py": "0" * 64}), rec)   # ... or on different code
    other = _recipe_file(tmp_path, "other.json", lr=3e-3)
    with pytest.raises(SystemExit, match="different recipe"):
        run_l2.check_seal(str(sealed), sha2, _gates(tmp_path, rec), other)                   # the run is pointed at a recipe the gates never saw
    with pytest.raises(SystemExit, match="SEAL REFUSED"):
        run_l2.check_seal(str(sealed), sha2, _gates(tmp_path, rec), str(tmp_path / "nope.json"))   # no recipe file at all
    assert run_l2.check_seal(str(sealed), sha2, _gates(tmp_path, rec), rec)["prereg_sha256"] == sha2   # and a clean seal passes


def test_cfeat_is_not_recipe_bound_but_is_code_bound(tmp_path):
    rec = _recipe_file(tmp_path)
    sealed = tmp_path / "S.md"
    sealed.write_text("# PREREG\n\n**Status: SEALED.**\n")
    sha = l2lib.sha256_file(str(sealed))
    g = _gates(tmp_path, rec)
    cf = json.load(open(os.path.join(g, "cfeat.json")))
    cf["recipe_sha256"] = None                                                                # C-FEAT never reads the recipe
    json.dump(cf, open(os.path.join(g, "cfeat.json"), "w"))
    assert run_l2.check_seal(str(sealed), sha, g, rec)["prereg_sha256"] == sha
    cf["code_sha256"] = {"l2lib.py": "0" * 64}
    json.dump(cf, open(os.path.join(g, "cfeat.json"), "w"))
    with pytest.raises(SystemExit, match="different recipe file or different code"):
        run_l2.check_seal(str(sealed), sha, g, rec)


def test_real_run_cells_are_the_arms_of_the_amended_prereg():
    cells = run_l2.all_cells()
    assert len(cells) == 1 + 2 + 2 + 2 * 5                                           # P2, H x2, H0 x2, GLOBAL H-null 5 replicates x 2 seeds
    assert cells["H_s0"].kind == "H" and cells["H_s1"].seed == 1
    assert cells["H0_s0"].kind == "H0" and cells["H0_s1"].seed == 1
    assert sorted({c.null_rep for c in cells.values() if c.kind == "Hnull"}) == [100, 101, 102, 103, 104]
    assert {c.seed for c in cells.values() if c.kind == "Hnull"} == {0, 1}
    assert {c.null_group for c in cells.values() if c.kind == "Hnull"} == {"global"}
    diag = run_l2.all_cells(("anchor",))
    assert {c.null_group for c in diag.values() if c.kind == "Hnull"} == {"anchor"} and len(diag) == 15            # the withdrawn definition is reachable only by asking for it


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- the read: pooled A1 (E2), H0 in-fan clause
def _fake_ev(a1=0.85, lo=0.8, a4=0.02, a5=0.01, infan=0.9, lon=0.5, lat=0.4):
    ci = lambda v: {"point": v, "lo": v - 0.05, "hi": v + 0.05}
    return {"all": {"A1": {"point": a1, "lo": lo, "hi": a1 + 0.05}, "A4": ci(a4), "A5": ci(a5)}, "lon": {"A1": ci(lon)}, "lat": {"A1": ci(lat)}, "infan_top8_ep_concordance": ci(infan)}


def _fake_run(h_infan=(0.9, 0.9), h0_infan=(0.9, 0.9), p2_infan=0.95, null_points=0.6):
    ev = {"P2": _fake_ev(a1=0.685, lo=0.6, infan=p2_infan)}
    for s in (0, 1):
        ev[f"H_s{s}"] = _fake_ev(infan=h_infan[s])
        ev[f"H0_s{s}"] = _fake_ev(a1=0.7, lo=0.65, infan=h0_infan[s])
        for r in l2lib.NULL_REPS:
            ev[run_l2.null_cell_name(r, s)] = _fake_ev(a1=null_points, lo=0.5)
    return ev


def test_read_uses_the_pooled_a1_the_h0_infan_at_the_same_seed_and_never_gates_on_p2(tmp_path):
    ev = _fake_run()
    res = run_l2.outcome_from_ev(ev, True, "global", "all", "any_seed")
    assert res["outcome"] == "L2-ADMIT"
    assert all(abs(r["a1_point"] - 0.85) < 1e-12 for r in res["inputs"]["reads"].values())              # the POOLED quantity, not lon (0.5) or lat (0.4)
    assert res["inputs"]["a1_lon_lat"]["H_s0"] == {"lon": 0.5, "lat": 0.4}                              # ... with lon / lat reported beside it
    # P2's in-fan is far above H's but H matches H0: NOT gated on P2
    res = run_l2.outcome_from_ev(_fake_run(h_infan=(0.6, 0.6), h0_infan=(0.6, 0.6), p2_infan=0.95), True, "global", "all", "any_seed")
    assert res["outcome"] == "L2-ADMIT" and res["inputs"]["reads"][0]["infan_p2"] == 0.95
    # the H0 comparison is per seed: seed 1's H0 is read for seed 1 (a mutation that read seed 0's H0 for both would map this wrongly)
    res = run_l2.outcome_from_ev(_fake_run(h_infan=(0.9, 0.80), h0_infan=(0.7, 0.9)), True, "global", "all", "any_seed")
    assert res["outcome"] == "L2-PARTIAL" and res["per_seed"][0]["infan_ok"] and not res["per_seed"][1]["infan_ok"]
    # a non-global null group is flagged as a diagnostic and never as the sealed outcome
    ev2 = _fake_run()
    for s in (0, 1):
        for r in l2lib.NULL_REPS:
            ev2[run_l2.null_cell_name(r, s, "anchor")] = _fake_ev(a1=0.9, lo=0.5)
    resd = run_l2.outcome_from_ev(ev2, True, "anchor", "all", "any_seed")
    assert resd["diagnostic_null_group"] is True and resd["outcome"] == "VOID"                          # the anchor-null above P2 + 0.05 voids ONLY the diagnostic read
    assert run_l2.outcome_from_ev(ev2, True, "global", "all", "any_seed")["diagnostic_null_group"] is False


def test_summarise_a1_is_the_pooled_quantity():
    ev = _fake_ev(a1=0.7, lon=0.2, lat=0.1)
    s = l2lib.summarise(ev)
    assert s["a1_point"] == 0.7 and s["a1_lon"] == 0.2 and s["a1_lat"] == 0.1


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- the semi-synthetic world used for the design
def test_semi_synthetic_hits_its_two_calibration_targets_and_reads_no_real_label():
    base = synthetic.make_data(n_tokens=72, n_logs=24, seed=9, gamma=0.8, rho=0.8)          # stands in for the real tokens: only its features and structure are used
    S = synthetic.semi_synthetic(base, beta=0.0, alpha=0.4, seed=1)
    p2 = l2lib.oof_scores(S, l2lib.ArmSpec("P2", "P2"), REC)
    ev = l2lib.evaluate_arm(S, p2["ep_in"], p2["ep_p"])
    assert abs(ev["infan_top8_ep_concordance"]["point"] - 0.889) < 0.012, ev["infan_top8_ep_concordance"]["point"]
    assert abs(ev["all"]["A1"]["point"] - 0.685) < 0.012, ev["all"]["A1"]["point"]
    assert not np.array_equal(S.t_in, base.t_in) and np.array_equal(S.tok_in, base.tok_in) and np.array_equal(S.tok_p, base.tok_p)     # labels replaced, tokens untouched
    # MM ruling 2026-10-19 (MM_RULING_L2_BUILD.md, sealing note): the planted-signal clause that stood here (72-token world, beta 3,
    # "H gains > +0.02 over P2") is MOVED, not loosened. Under the selected in-fan-weighted recipe it measured +0.0044, a property of
    # a world 5.5x smaller than the run (57 training tokens per fold). The SAME +0.02 bar is asserted at the REAL shape (400 tokens,
    # 24 logs, a weaker beta 0.5 signal) by test_selected_recipe_finds_a_planted_signal_on_REAL_tokens (measured +0.045), which
    # is the binding sensitivity check for the configuration that actually runs.


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- reachability of the EP score into the read
def test_the_arms_ep_score_reaches_the_aggregate_and_nothing_else_moves(Dsig):
    """the retrained EP score must enter the V1_W aggregate (A3/A4/A5) - a read that ignored the arm's EP would 'pass' C-ID for the wrong reason - while the frozen heads' reads (A2, DAC) stay fixed"""
    D = Dsig
    p2 = l2lib.oof_scores(D, l2lib.ArmSpec("P2", "P2"), REC)
    e0 = l2lib.evaluate_arm(D, p2["ep_in"], p2["ep_p"])
    rng = np.random.default_rng(0)
    e1 = l2lib.evaluate_arm(D, p2["ep_in"] + rng.normal(0, 1.0, p2["ep_in"].shape), p2["ep_p"] + rng.normal(0, 1.0, p2["ep_p"].shape))       # an arbitrary other EP score
    assert e1["all"]["A1"]["point"] != e0["all"]["A1"]["point"]
    assert e1["all"]["A3"]["point"] != e0["all"]["A3"]["point"]                                  # the aggregate saw the new EP
    assert e1["all"]["A4"]["point"] != e0["all"]["A4"]["point"] or e1["all"]["A5"]["point"] != e0["all"]["A5"]["point"]
    for k in ("point", "lo", "hi", "den"):
        a, b = e0["all"]["A2"][k], e1["all"]["A2"][k]
        assert (np.isnan(a) and np.isnan(b)) or a == b                                           # TTC head is frozen: A2 is unchanged by ANY EP score
    same = lambda x, y: all((np.isnan(x[k]) and np.isnan(y[k])) or x[k] == y[k] for k in x)          # NaN == NaN (the synthetic DAC head has no pairs)
    assert same(e0["lat_DAC_concordance"], e1["lat_DAC_concordance"]) and e0["tactical"] == e1["tactical"]


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- the production commands, end to end, on SYNTHETIC data
def test_run_and_read_end_to_end_on_synthetic_data(tmp_path, monkeypatch):
    """`run_l2.py run` + `read` as the Master Mind will call them, on a small synthetic Data (no real label, no real file): all 15 cells are produced (P2, H x2, H0 x2, 10 GLOBAL-null), the read maps an outcome (not INCOMPLETE),
    in-fan(H) is compared with in-fan(H0), and a re-run resumes (ZZSKIP).  Guards the 'built, tested, unreachable from its caller' class for the run path itself."""
    D = synthetic.make_data(n_tokens=48, n_logs=24, seed=4, gamma=0.5, rho=0.4)
    monkeypatch.setattr(l2lib, "load_real", lambda *a, **k: D)
    rec = _recipe_file(tmp_path, epochs=2, lr=2e-3, prox=0.1)
    prereg = tmp_path / "PREREG.md"
    prereg.write_text("# PREREG\n\n**Status: SEALED 2026-10-19.**\n")
    sha = l2lib.sha256_file(str(prereg))
    gates = _gates(tmp_path, rec)
    feats = tmp_path / "feats.npz"
    feats.write_bytes(b"not read: load_real is replaced")
    out = tmp_path / "run"
    common = ["--prereg", str(prereg), "--prereg-sha256", sha, "--gates-dir", gates, "--feats", str(feats), "--out-dir", str(out), "--recipe", rec]
    assert run_l2.main(["run", *common, "--threads", "1"]) == 0
    arms = sorted(p.stem for p in (out / "arms").glob("*.npz"))
    assert len(arms) == 15 and "H0_s0" in arms and "H0_s1" in arms and "Hnull-global_r104_s1" in arms and not any("anchor" in a or "token" in a for a in arms)
    mt = os.path.getmtime(out / "arms" / "H_s0.npz")
    assert run_l2.main(["run", *common, "--threads", "1", "--cells", "H_s0"]) == 0                       # resumable: the finished cell is not recomputed
    assert os.path.getmtime(out / "arms" / "H_s0.npz") == mt
    assert run_l2.main(["read", *common]) == 0
    res = json.load(open(out / "L2_RESULT.json"))
    assert res["outcome"]["outcome"] in ("L2-ADMIT", "L2-PARTIAL", "L2-FLAT", "VOID") and not res["cells_missing"]
    assert res["outcome"]["diagnostic_null_group"] is False and "infan_h0" in res["outcome"]["inputs"]["reads"]["0"]
    md = open(out / "L2_RESULT.md", encoding="utf-8").read()
    assert "in-fan(H) >= in-fan(H0) - 0.02" in md and "P2's in-fan is REPORTED, never gated" in md and "STRATEGIC: PAUSED" in md
    # the same arrays, an incomplete run: no verdict
    (out / "arms" / "Hnull-global_r103_s1.npz").unlink()
    assert run_l2.main(["read", *common]) == 0
    assert json.load(open(out / "L2_RESULT.json"))["outcome"]["outcome"] == "INCOMPLETE"
    # and a stale gate (earned on another recipe) refuses the run before any cell is trained
    other = _recipe_file(tmp_path, "other.json", epochs=2, lr=1e-3)
    with pytest.raises(SystemExit, match="SEAL REFUSED"):
        run_l2.main(["run", "--prereg", str(prereg), "--prereg-sha256", sha, "--gates-dir", gates, "--feats", str(feats), "--out-dir", str(tmp_path / "run2"), "--recipe", other])
    assert not (tmp_path / "run2").exists()


def test_selected_recipe_finds_a_planted_signal_on_REAL_tokens():
    """REGRESSION test (not a selection criterion): on the REAL exported tokens with SYNTHETIC labels (sigA, a planted within-slot signal of beta 0.5) the SELECTED recipe's H gains >= +0.02 A1-OOF over P2
    (MEASURED at selection: +0.045).  The unit-world test above (72 tokens, beta 3) is a different, harsher question for an in-fan-weighted recipe; see RECIPE_SELECTION.md."""
    feats = "D:/Projects/TanitAD-artifacts/l2_offprop/feats.npz"
    if not os.path.exists(feats):
        pytest.skip(f"REAL tokens {feats} not found on this machine: the same number is in raw/sweep_semi.json (sigA a1_gain_over_p2)")
    real = l2lib.load_real(feats)
    S = synthetic.semi_synthetic(real, beta=0.5, alpha=0.25, seed=21)
    p2 = l2lib.oof_scores(S, l2lib.ArmSpec("P2", "P2"), CHOSEN)
    h = l2lib.oof_scores(S, l2lib.ArmSpec("H", "H", seed=0), CHOSEN)
    a2 = l2lib.evaluate_arm(S, p2["ep_in"], p2["ep_p"])["all"]["A1"]["point"]
    ah = l2lib.evaluate_arm(S, h["ep_in"], h["ep_p"])["all"]["A1"]["point"]
    assert ah > a2 + 0.02, (a2, ah)


def test_gate_stamp_is_line_ending_insensitive(tmp_path):
    """a CRLF checkout of the same recipe file must not make a gate PASS look stale (autocrlf on Windows)"""
    rec = tmp_path / "recipe_lf.json"
    rec.write_bytes(json.dumps({"recipe": l2lib.Recipe().as_dict()}, indent=1).encode())          # multi-line, LF only
    rec = str(rec)
    crlf = tmp_path / "recipe_crlf.json"
    crlf.write_bytes(open(rec, "rb").read().replace(b"\n", b"\r\n"))
    assert open(rec, "rb").read() != crlf.read_bytes() and b"\r\n" in crlf.read_bytes()
    assert l2lib.gate_stamp(rec)["recipe_sha256"] == l2lib.gate_stamp(str(crlf))["recipe_sha256"]
    other = _recipe_file(tmp_path, "o.json", lr=3e-3)
    assert l2lib.gate_stamp(rec)["recipe_sha256"] != l2lib.gate_stamp(other)["recipe_sha256"]        # and a real change still shows
