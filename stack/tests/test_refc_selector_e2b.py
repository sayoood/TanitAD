"""``E-DDA-2b`` driver — the metrics' analytic values, the argparse-derived
stamp, and the OFF-path proof done the HARD way.

⛔ THE OFF-PATH PROOF IS THE POINT OF THIS FILE, AND A NAIVE VERSION IS
VACUOUS. ``WP-B``'s removability proof was **green with its head deliberately
corrupted**, because a zero-init gate multiplied the whole branch away. So here
every new path is **deliberately corrupted** and the pair is asserted:

* with the knob OFF the comparison must STILL PASS — the corruption cannot
  reach the default path;
* with the knob ON the comparison must FAIL — the corruption is reachable, so
  the OFF result is a fact about the gate and not about an inert branch.

A proof that only shows the first half proves nothing.
"""

from __future__ import annotations

import importlib
import os
import sys

import pytest
import torch

_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

E = importlib.import_module("refc_selector_e2b")
from tanitad.refs import refc_selector_aug as AUG          # noqa: E402


# --------------------------------------------------------------------------- #
# 1. the metrics' ANALYTIC values                                              #
# --------------------------------------------------------------------------- #
def test_rank_auc_of_a_constant_scorer_is_EXACTLY_the_no_information_value():
    q = torch.rand(6, 20)
    for c in (0.0, 1.0, -3.5):
        auc, n = E.rank_auc(torch.full_like(q, c), q)
        assert auc == E.RANK_AUC_NO_INFORMATION == 0.5, f"const {c} read {auc}"
        assert n > 0


def test_rank_auc_is_exactly_1_and_exactly_0_at_the_two_extremes():
    q = torch.tensor([[0.1, 0.5, 0.9, 0.3]])
    assert E.rank_auc(q, q)[0] == 1.0
    assert E.rank_auc(-q, q)[0] == 0.0


def test_ap_of_a_constant_scorer_is_EXACTLY_the_base_rate():
    """ANALYTIC: one tie group of N holding P positives ⇒ AP = P/N exactly."""
    for n, p in ((10, 3), (7, 7), (100, 1), (64, 41)):
        y = torch.cat([torch.ones(p), torch.zeros(n - p)])
        y = y[torch.randperm(n)]
        ap, got_n, base = E.average_precision(torch.zeros(n), y)
        assert got_n == n
        assert base == p / n
        assert ap == pytest.approx(p / n, abs=0.0, rel=1e-12), (
            f"tie-group AP read {ap} against a base rate of {p / n}")


def test_MUTATION_the_naive_AP_misreads_a_constant_arm_IN_BOTH_DIRECTIONS():
    """⛔ The tie rule is the measurement. Breaking ties by array index makes a
    CONSTANT scorer read something other than its base rate — ABOVE it when the
    positives happen to sit early, BELOW when they sit late. That direction
    dependence is why the failure is so hard to spot in a table.

    MEASURED elsewhere in this programme: +22.8 % on one rig, +0.881 % on
    another; MEASURED here on this rig: -0.0066 on the ``const`` arm.
    """
    n, p = 20, 6
    y_early = torch.cat([torch.ones(p), torch.zeros(n - p)])
    y_late = torch.cat([torch.zeros(n - p), torch.ones(p)])
    s = torch.zeros(n)
    base = p / n

    assert E.average_precision(s, y_early)[0] == pytest.approx(base, rel=1e-12)
    assert E.average_precision(s, y_late)[0] == pytest.approx(base, rel=1e-12)

    assert E._ap_naive(s, y_early) > base, "naive AP must inflate here"
    assert E._ap_naive(s, y_late) < base, "naive AP must deflate here"


def test_ap_refuses_a_shape_it_cannot_mean():
    with pytest.raises(ValueError):
        E.average_precision(torch.zeros(2, 3), torch.zeros(2, 3))


# --------------------------------------------------------------------------- #
# 2. the rig's ANALYTIC cross-check                                            #
# --------------------------------------------------------------------------- #
def test_the_straight_arm_reads_curvature_EXACTLY_zero():
    """The one self-check statement with no discretisation error, as a literal.
    A sign flip or an axis swap breaks it first."""
    rig = E.build_rig(E.RigSpec(n_windows=2))
    chk = E.rig_self_check(rig)
    assert chk["straight_arm_present"] is True
    assert chk["straight_arm_kappa_max"] == 0.0
    assert chk["sign_agreement"] == 1.0
    assert chk["rank_agreement_monotone"] is True


def test_the_curved_arms_agree_within_the_CHORD_discretisation_and_say_so():
    """⚠️ NOT exact, and the tolerance carries its cause. MEASURED 2026-09-10:
    rel err <= 3.73 % at kappa*ds = 0.302 rad."""
    chk = E.rig_self_check(E.build_rig(E.RigSpec(n_windows=2)))
    assert chk["kappa_rel_err_max"] < 0.05
    assert 0.25 < chk["kappa_ds_rad"] < 0.35


def test_an_EVEN_curvature_grid_is_refused_because_it_omits_kappa_zero():
    with pytest.raises(ValueError, match="ODD"):
        E.build_arcs(E.RigSpec(n_curv=8))


def test_a_fan_whose_candidates_REVERSE_is_refused():
    """MEASURED 2026-09-10: an a=-2.0 arm at v0=10 over 6 s crosses zero speed
    at t=5 s; the recovered curvature then reads -1.2862 against a built
    -0.0600 on 16/32 candidates and looks like a broken instrument."""
    with pytest.raises(ValueError, match="REVERSES"):
        E.build_arcs(E.RigSpec(end_speed_frac=-1.0))


# --------------------------------------------------------------------------- #
# 3. the stamp is DERIVED FROM ARGPARSE, in both directions                    #
# --------------------------------------------------------------------------- #
def _stamp_for(argv: list[str]) -> tuple[dict, list[str]]:
    parser = E.build_parser()
    args = parser.parse_args(argv)
    acfg = AUG.AugmentConfig(n_aug=args.n_aug, std_min=args.std_min,
                             std_max=args.std_max,
                             foreign_frac=args.foreign_frac)
    specs = E.term_specs(parser, args, acfg)
    from tanitad import effective_weights as EW
    rows = EW.classify_all(specs, EW.explicit_dests(parser, argv))
    st = EW.stamp(rows, where="test", explicit_source="argv")
    st["knob_dests"] = E.knob_dests(parser)
    st["values"] = {d: getattr(args, d) for d in E.knob_dests(parser)}
    return st, E.knob_dests(parser)


def test_EVERY_argparse_knob_appears_in_the_stamp_and_nothing_else_does():
    """⛔ The ``--wp-index`` guard: 3 of 6 knobs were parsed, stamped and inert,
    and only an argparse-DERIVED list found it. Equality in BOTH directions, so
    a knob cannot be forgotten and a stale row cannot survive a rename."""
    st, dests = _stamp_for([])
    assert len(dests) == len(set(dests)), "a dest is served by two actions"
    assert {r["term"] for r in st["terms"]} == set(dests)
    assert set(st["values"]) == set(dests)
    assert st["n_terms"] == len(dests)


def test_the_stamp_carries_the_LITERAL_value_of_every_knob():
    st, _ = _stamp_for(["--n-aug", "3", "--foreign-frac", "0.02",
                        "--self-attn", "off"])
    assert st["values"]["n_aug"] == 3
    assert st["values"]["foreign_frac"] == 0.02
    assert st["values"]["self_attn"] == "off"


def test_explicitness_comes_from_ARGV_not_from_the_value():
    st, _ = _stamp_for(["--n-aug", "0"])          # typed, and equal to the default
    row = next(r for r in st["terms"] if r["term"] == "n_aug")
    assert row["explicit"] is True
    st2, _ = _stamp_for([])
    row2 = next(r for r in st2["terms"] if r["term"] == "n_aug")
    assert row2["explicit"] is False


def test_MUTATION_a_dropped_stamp_row_is_CAUGHT():
    """If the equality assert cannot fail, the argparse derivation proves
    nothing."""
    st, dests = _stamp_for([])
    st["terms"] = [r for r in st["terms"] if r["term"] != "top_k"]
    assert {r["term"] for r in st["terms"]} != set(dests)


def test_an_INERT_foreign_fraction_is_flagged_by_needs_not_silently_stamped():
    st, _ = _stamp_for(["--foreign-frac", "0.0001", "--foreign-bank-curv", "3",
                        "--foreign-bank-acc", "2"])
    row = next(r for r in st["terms"] if r["term"] == "foreign_frac")
    assert row["missing"] is not None and "INERT" in row["missing"]
    assert row["status"] == "NO_GRAPH"


def test_a_top_k_at_or_above_the_fan_is_flagged_as_an_inert_prune():
    st, _ = _stamp_for(["--top-k", "999"])
    row = next(r for r in st["terms"] if r["term"] == "top_k")
    assert row["missing"] is not None and "prune never fires" in row["missing"]


# --------------------------------------------------------------------------- #
# 4. THE OFF-PATH PROOF, WITH THE CORRUPTION THAT MAKES IT NON-VACUOUS         #
# --------------------------------------------------------------------------- #
def _score(argv: list[str]) -> torch.Tensor:
    parser = E.build_parser()
    args = parser.parse_args(argv)
    spec = E.RigSpec(n_windows=args.n_windows, n_curv=args.n_curv,
                     n_acc=args.n_acc, n_steps=args.n_steps, dt=args.dt,
                     v0=args.v0, seed=args.seed)
    rig = E.build_rig(spec)
    acfg = AUG.AugmentConfig(
        n_aug=args.n_aug, std_min=args.std_min, std_max=args.std_max,
        foreign_frac=args.foreign_frac,
        foreign_train_only=not bool(args.foreign_at_eval))
    tcfg = E.T.TargetConfig(dt=args.dt)
    b = spec.n_windows
    perm = torch.randperm(b, generator=torch.Generator().manual_seed(args.seed))
    n_fit = max(int(args.fit_frac * b), 1)
    fit, sco = perm[:n_fit], perm[n_fit:]
    return E.run_selector(rig, args, tcfg, acfg, fit, sco,
                          progress_only=False)["score"]


_FAST = ["--n-windows", "6", "--n-curv", "3", "--n-acc", "2", "--steps", "4",
         "--batch", "2", "--top-k", "4"]


def test_the_default_path_is_BITWISE_unchanged_when_augmentation_is_OFF():
    a = _score(_FAST)
    b = _score(_FAST)
    assert torch.equal(a, b), "the OFF path is not even deterministic"


def test_CORRUPTION_the_augmentation_branch_cannot_reach_the_OFF_path(monkeypatch):
    """⛔ THE PAIR. Corrupt ``add_mul_noise`` beyond recognition, then:
    OFF must be UNCHANGED (the gate holds) and ON must CHANGE (the corruption
    is reachable, so the OFF result is not a fact about an inert branch)."""
    clean_off = _score(_FAST)
    clean_on = _score(_FAST + ["--n-aug", "2"])

    real = AUG.add_mul_noise

    def poisoned(cand, cfg=None, *, generator=None):
        # ⭐ CORRUPT THE BRANCH, NOT THE FUNCTION — and with NO CONDITIONAL, so
        # the corruption is not itself gated. The augmented block is
        # `out[:, n:]`; when `--n-aug 0` that slice is EMPTY, so this is a no-op
        # BECAUSE THE BRANCH DOES NOT EXIST, which is the thing being proved.
        #
        # ⚠️ MEASURED 2026-09-10: my first version wrote `out * 1e6`, which
        # corrupts the identity path too — the OFF assertion then failed and the
        # "proof" would have been about my wrapper, not about the flag. The test
        # caught my test.
        out, origin = real(cand, cfg, generator=generator)
        n = cand.shape[1]
        return torch.cat([out[:, :n], out[:, n:] * 1e6], dim=1), origin

    monkeypatch.setattr(AUG, "add_mul_noise", poisoned)

    assert torch.equal(_score(_FAST), clean_off), (
        "the corrupted augmentation reached the OFF path — the flag does not "
        "gate it")
    assert not torch.equal(_score(_FAST + ["--n-aug", "2"]), clean_on), (
        "⛔ VACUOUS PROOF: corrupting the augmentation changed NOTHING with the "
        "flag ON, so the OFF comparison above says nothing about the gate. "
        "This is the WP-B failure — a branch multiplied away by something else")


def test_CORRUPTION_the_foreign_bank_cannot_reach_the_OFF_path(monkeypatch):
    clean_off = _score(_FAST)
    on_argv = _FAST + ["--foreign-frac", "0.5", "--foreign-bank-curv", "5",
                       "--foreign-bank-acc", "2"]
    clean_on = _score(on_argv)

    real = AUG.mix_foreign_bank

    def poisoned(cand, bank, cfg=None, *, origin=None, generator=None):
        out, org = real(cand, bank * 0.0 + 999.0, cfg, origin=origin,
                        generator=generator)
        return out, org

    monkeypatch.setattr(AUG, "mix_foreign_bank", poisoned)

    assert torch.equal(_score(_FAST), clean_off), (
        "the corrupted bank reached the OFF path")
    assert not torch.equal(_score(on_argv), clean_on), (
        "⛔ VACUOUS PROOF: poisoning the bank changed nothing with the bank ON")


def test_the_ON_path_actually_differs_from_the_OFF_path():
    """The third leg: OFF vs ON must differ at all, or 'OFF is unchanged' is
    trivially true and the knob does nothing."""
    assert not torch.equal(_score(_FAST), _score(_FAST + ["--n-aug", "2"]))


# --------------------------------------------------------------------------- #
# 5. end to end: the controls read their known values                          #
# --------------------------------------------------------------------------- #
def test_the_dump_source_is_BLOCKED_not_approximated():
    assert E.main(["--source", "dump"]) == 3


def test_the_const_arm_reads_its_known_value_EXACTLY(capsys):
    """⚠️ Only RIG knobs are passed. Typing a selector knob on an arm that
    builds no selector is a DISCARDED weight and the driver refuses the launch
    — which is the guard working, and the next test pins it."""
    rig_only = ["--n-windows", "8", "--n-curv", "3", "--n-acc", "2"]
    assert E.main(["--arm", "const"] + rig_only) == 0
    out = capsys.readouterr().out
    assert "rank_auc=0.500000" in out, (
        "the no-information control did not read its known value EXACTLY")


def test_a_selector_knob_typed_on_a_control_arm_REFUSES_the_launch(capsys):
    """⛔ `--steps 6 --arm const` advertises a knob the arm discards. The run
    record would say `steps: 6` for an arm that never trains. Refused."""
    rc = E.main(["--arm", "const", "--n-windows", "8", "--n-curv", "3",
                 "--n-acc", "2", "--steps", "6"])
    assert rc == 2
    assert "REFUSED" in capsys.readouterr().out
    rc2 = E.main(["--arm", "const", "--n-windows", "8", "--n-curv", "3",
                  "--n-acc", "2", "--steps", "6",
                  "--allow-discarded-weights"])
    assert rc2 == 0, "the acknowledgement flag must let an honest launch through"


def test_the_deliberate_regression_does_NOT_improve_the_contact_readout(capsys):
    """⛔ `H-DDA-7`'s pre-registered regression arm. A progress-only head selects
    for distance covered, so its tie-group AP on the no-contact label must NOT
    beat the base rate. If it does, the contact readout is blind and any panel
    built on it is VOID."""
    fast = ["--n-windows", "10", "--n-curv", "5", "--n-acc", "3", "--steps",
            "40", "--batch", "3", "--top-k", "8"]
    assert E.main(["--arm", "progress_only"] + fast) == 0
    out = capsys.readouterr().out
    line = [l for l in out.splitlines() if "AP(tie-group)" in l][-1]
    ap = float(line.split("AP(tie-group)=")[1].split()[0])
    base = float(line.split("base_rate=")[1].split()[0])
    assert ap <= base + 1e-9, (
        f"the progress-only regression arm read AP {ap:.6f} ABOVE the base "
        f"rate {base:.6f} — the contact readout cannot see it, panel VOID")

def test_an_INERT_foreign_at_eval_is_REFUSED_before_the_run(capsys):
    """⭐ AN INERT KNOB I SHIPPED AND THEN CAUGHT IN MY OWN DRIVER.

    MEASURED 2026-09-10: the first version of this script scored the NATIVE fan
    at eval unconditionally, so `--foreign-at-eval` was parsed, stamped into the
    run record, and could not change a single number -- the `--wp-index` failure
    (3 of 6 knobs parsed, stamped and inert) reproduced inside the very package
    that cites it. The fix is `--eval-aug` plus a `needs` row, and this pins the
    refusal.
    """
    rc = E.main(["--arm", "selector", "--steps", "3", "--n-windows", "6",
                 "--n-curv", "3", "--n-acc", "2", "--top-k", "4",
                 "--foreign-frac", "0.5", "--foreign-bank-curv", "5",
                 "--foreign-bank-acc", "2", "--foreign-at-eval"])
    assert rc == 2
    out = capsys.readouterr().out
    assert "INERT" in out and "REFUSED" in out


def test_eval_aug_with_no_augmentation_is_ALSO_refused_as_inert():
    rc = E.main(["--arm", "selector", "--steps", "3", "--n-windows", "6",
                 "--n-curv", "3", "--n-acc", "2", "--top-k", "4",
                 "--eval-aug"])
    assert rc == 2


def test_the_emittable_guard_is_LIVE_on_the_eval_path(monkeypatch):
    """⛔ A guard that is never CALLED is the same as no guard. Force the pick
    onto a foreign candidate and require the driver to refuse."""
    import tanitad.refs.refc_selector as S

    real_forward = S.SubMetricSelector.forward

    def picks_the_last(self, *a, **kw):
        out = real_forward(self, *a, **kw)
        out["sel_idx_selector"] = torch.full_like(out["sel_idx_selector"],
                                                  out["score"].shape[1] - 1)
        return out

    monkeypatch.setattr(S.SubMetricSelector, "forward", picks_the_last)
    with pytest.raises(ValueError, match="FOREIGN"):
        E.main(["--arm", "selector", "--steps", "3", "--n-windows", "6",
                "--n-curv", "3", "--n-acc", "2", "--top-k", "4",
                "--n-aug", "1", "--foreign-frac", "0.5",
                "--foreign-bank-curv", "5", "--foreign-bank-acc", "2",
                "--foreign-at-eval", "--eval-aug"])


def test_the_eval_fan_is_NATIVE_by_default_so_arms_stay_comparable():
    """Every arm must score the same candidates unless --eval-aug says so."""
    parser = E.build_parser()
    assert parser.parse_args([]).eval_aug is False
    assert parser.parse_args([]).foreign_at_eval is False
