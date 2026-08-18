"""F-10 / catalog S3 — THE DOMAIN-STRATIFIED TRAINING MIX: both directions.

THE SPEC, quoted (two independent locations, established BEFORE a line of the
implementation was written):

  * ``…/2026-08-07-hierarchical-wm-redesign/V6_TRAINING_MEASURES.md:81`` —
    *"S3 | domain-stratified training mix (geographic/domain diversity beats
    volume — arXiv 2607.04500) | the S1 scaling-ladder data-mix arm folds in
    here | cross-domain P-battery deltas reported per stratum"*
  * ``…/2026-08-16-diagram-conformance/DIAGRAM_CONFORMANCE.md:69`` —
    *"domain-diverse mix (catalog S3) | NOT BUILT | no domain-stratified
    sampling in `train()` (episode draw is uniform / O4-weighted only). Needs
    the VLM/scena strata as a SAMPLER input — which is admissible for the data
    MIX (it is not a model input) but must be declared. Fix F-10"*, and
    ``:215`` — *"F-10 | P3 | S3 domain-stratified mix — VLM strata as SAMPLER
    input (admissible: data mix, not a model input; declare it)."*

⛔ THE TWO CENTRAL FACTS THIS FILE PINS.

**(1) A PER-WINDOW DOMAIN WEIGHT IS EXACTLY A NO-OP.**
``InteractionSampler`` draws EPISODES uniformly and consults its ``weights``
only INSIDE the drawn episode. A domain label is an episode property, hence
constant within an episode, and a constant through ``torch.multinomial`` is
uniform. The obvious implementation — "put the domain weight in the sampler's
weight vector" — therefore changes NOTHING about the quantity F-10 exists to
change, and it does so silently. Same class as C115 (a loss over ``z_tac``'s
non-existent temporal extent). Demonstrated here on the shared sampler, then
shown absent from the shipped one.

**(2) THE DIVERSITY OBJECTIVE IS MAXIMAL AT TWO OPPOSITE DEGENERATE INPUTS.**
A "perfectly balanced" reading is achieved, at EVERY temperature, both when
there is exactly ONE stratum and when EVERY EPISODE IS ITS OWN STRATUM — and
both are exactly the uniform draw. A balance metric cannot tell either of them
from a working mix. Same class as C119, where the naive interaction entropy read
0.9649 on an empty road against 0.2500 on a dense one. Both are refused, not
warned about, because the score cannot see them.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from tanitad.models.v6 import (  # noqa: E402
    DOMAIN_MIX_CONTROL_MIN_N, DOMAIN_MIX_MAX_AMPLIFICATION,
    DOMAIN_MIX_MIN_STRATUM_EPISODES, DomainMix, InteractionSampler,
    StratifiedEpisodeSampler, V6Config, V6Stack, domain_mix_control)

# ---------------------------------------------------------------------------
# synthetic corpora — the stratum shapes the guards are calibrated against
# ---------------------------------------------------------------------------

#: The realistic shape at the parity-corpus SIZE (2376 episodes). The numbers in
#: `DOMAIN_MIX_MAX_AMPLIFICATION`'s docstring are MEASURED on exactly these.
REALISTIC = {"urban_day": 1200, "urban_night": 500, "highway_day": 400,
             "rain": 180, "snow": 60, "dusk": 36}
COARSE_3 = {"urban": 1800, "highway": 400, "rural": 176}
FINE_10 = {f"s{i}": n for i, n in
           enumerate([900, 500, 300, 200, 150, 120, 90, 60, 40, 16])}
LONG_TAIL_12 = {f"s{i}": n for i, n in
                enumerate([1500, 300, 180, 120, 90, 60, 40, 30, 20, 16, 12, 8])}


def _strata(sizes: dict) -> list[str]:
    out: list[str] = []
    for k, v in sizes.items():
        out += [k] * v
    return out


def _index(n_ep: int, per_ep: int = 20) -> list[tuple[int, int]]:
    return [(e, t) for e in range(n_ep) for t in range(per_ep)]


def _draw(sampler, n: int = 4000, bs: int = 8) -> list[int]:
    out: list[int] = []
    for _ in range(n // bs):
        out += sampler(bs)
    return out


# ===========================================================================
# (1) THE ARCHITECTURAL FINDING — a per-window domain weight is a NO-OP
# ===========================================================================

def test_a_per_window_domain_weight_is_EXACTLY_a_noop_on_the_episode_mix():
    """⛔ THE FINDING THAT DETERMINED THE IMPLEMENTATION.

    A domain-balanced weight expressed per WINDOW gives the BIT-IDENTICAL draw
    sequence to an all-ones weight, and leaves the achieved domain share at the
    corpus proportion. This is not an approximation — the sequences are equal
    element-for-element, because the episode draw never reads ``weights`` and a
    constant vector through ``multinomial`` is uniform.
    """
    index = _index(6)
    dom = {0: "urban", 1: "urban", 2: "urban", 3: "urban",
           4: "highway", 5: "highway"}
    # the "balanced" intent: urban has 4 eps, highway 2, so each domain wants
    # half the mass -> episode weight 1/4 vs 1/2.
    wmap = {"urban": 0.25, "highway": 0.5}
    w_dom = torch.tensor([wmap[dom[e]] for e, _ in index], dtype=torch.float32)

    def run(w, seed=1234):
        g = torch.Generator().manual_seed(seed)
        return _draw(InteractionSampler(index, w, eps_per_batch=4,
                                        generator=g))

    a = run(w_dom)
    b = run(torch.ones(len(index)))
    assert a == b, "a per-window domain weight moved the draw — it must not"

    share = collections.Counter(dom[index[i][0]] for i in a)
    frac = share["urban"] / len(a)
    # the CORPUS proportion is 4/6 = 0.6667; the balanced TARGET is 0.5.
    assert abs(frac - 4 / 6) < 0.05, frac
    assert abs(frac - 0.5) > 0.10, (
        "the per-window weight must NOT have reached the balanced target")


def test_the_stratified_sampler_DOES_move_the_episode_mix():
    """The paired opposite of the finding above — the positive control.

    Without this, "the no-op test passes" is equally consistent with a sampler
    that cannot move the mix at all.
    """
    index = _index(6)
    dom = {0: "urban", 1: "urban", 2: "urban", 3: "urban",
           4: "highway", 5: "highway"}
    labels = [dom[e] for e in range(6)]
    ep_w = DomainMix(tau=1.0, min_stratum_episodes=2).episode_weights(labels)
    g = torch.Generator().manual_seed(1234)
    s = StratifiedEpisodeSampler(index, torch.ones(len(index)),
                                 {i: float(w) for i, w in
                                  enumerate(ep_w.tolist())},
                                 eps_per_batch=4, generator=g)
    got = _draw(s)
    share = collections.Counter(dom[index[i][0]] for i in got)
    frac = share["urban"] / len(got)
    assert abs(frac - 0.5) < 0.05, (
        f"the stratified sampler did not reach the balanced target: {frac}")


def test_the_shared_InteractionSampler_contract_was_NOT_weakened():
    """F-7's lesson: do not edit a shared module to fit a new cell into it.

    ``InteractionSampler``'s guarantee is *"episodes are drawn uniformly so no
    episode is starved"*, and F-10 is precisely the arm that breaks it — so
    F-10 SUBCLASSES rather than edits. Pinned against the source: the base
    class still draws with ``randint``.
    """
    src = (ROOT / "tanitad" / "models" / "v6.py").read_text(encoding="utf-8")
    body = src.split("class InteractionSampler", 1)[1].split(
        "\n# ====", 1)[0]
    assert "torch.randint(len(self.ep_ids)" in body, (
        "InteractionSampler's uniform episode draw was edited")
    assert "Episodes are drawn uniformly so no episode is starved" in body
    assert issubclass(StratifiedEpisodeSampler, InteractionSampler)


def test_the_window_weights_are_inherited_UNTOUCHED_so_O4_T3_compose():
    """F-10 acts on EPISODES, O4/T3 on WINDOWS — genuinely different axes.

    That is why F-10 is not refused alongside ``--o4-alpha`` the way
    ``--t3-scores`` is (T3 and O4 are two levers on ONE axis).
    """
    index = _index(4, per_ep=10)
    win_w = torch.zeros(len(index))
    # inside every episode, make ONLY window 0 drawable
    for i, (_e, t) in enumerate(index):
        if t == 0:
            win_w[i] = 1.0
    labels = ["a", "a", "b", "b"]
    ep_w = DomainMix(tau=1.0, min_stratum_episodes=2).episode_weights(labels)
    g = torch.Generator().manual_seed(7)
    s = StratifiedEpisodeSampler(index, win_w,
                                 {i: float(w) for i, w in
                                  enumerate(ep_w.tolist())},
                                 eps_per_batch=4, generator=g)
    got = _draw(s, n=800)
    assert all(index[i][1] == 0 for i in got), (
        "the inherited window weighting was not respected")


# ===========================================================================
# (2) THE DEGENERATE STRATIFICATIONS — the C119 shape
# ===========================================================================

def test_ONE_stratum_is_EXACTLY_uniform_at_every_tau():
    """Demonstrated on the arithmetic before it is refused."""
    labels = ["only"] * 64
    for tau in (0.0, 0.25, 0.5, 0.75, 1.0):
        mix = DomainMix(tau=tau, min_stratum_episodes=1)
        sizes = {"only": 64}
        qs = {k: float(v) ** (1.0 - tau) for k, v in sizes.items()}
        z = sum(qs.values())
        w = torch.tensor([qs[s] / (sizes[s] * z) for s in labels])
        assert torch.allclose(w, torch.full_like(w, 1 / 64), atol=1e-7)
        with pytest.raises(ValueError, match="one-stratum mix is EXACTLY"):
            mix.episode_weights(labels)


def test_every_episode_its_OWN_stratum_is_ALSO_uniform_and_is_refused():
    """The OPPOSITE degenerate input, and it lands in the same place."""
    labels = [f"ep{i}" for i in range(64)]
    for tau in (0.0, 0.5, 1.0):
        with pytest.raises(ValueError, match="every "
                                             "episode is its own stratum"):
            DomainMix(tau=tau, min_stratum_episodes=1).episode_weights(labels)


def test_BOTH_degenerate_stratifications_score_PERFECTLY_BALANCED():
    """⛔ THE C119 DEMONSTRATION — why these are refusals, not warnings.

    The naive "diversity" reading (max stratum share − min stratum share, or
    equivalently the KL to the uniform stratum distribution) is EXACTLY ZERO on
    both degenerate inputs at tau = 1. The score is at its optimum on precisely
    the two inputs that make it meaningless, so no threshold on it can catch
    them.
    """
    def balance_gap(labels: list[str], tau: float = 1.0) -> float:
        sizes: dict = {}
        for s in labels:
            sizes[s] = sizes.get(s, 0) + 1
        qs = {k: float(v) ** (1.0 - tau) for k, v in sizes.items()}
        z = sum(qs.values())
        share = {k: qs[k] / z for k in sizes}
        return max(share.values()) - min(share.values())

    assert balance_gap(["only"] * 64) == pytest.approx(0.0, abs=1e-12)
    assert balance_gap([f"ep{i}" for i in range(64)]) == pytest.approx(
        0.0, abs=1e-12)
    # and a WORKING mix scores identically well, which is the point
    assert balance_gap(_strata(REALISTIC)) == pytest.approx(0.0, abs=1e-12)


def test_the_amplification_ceiling_refuses_the_long_tail_shape():
    """MEASURED calibration, re-derived here rather than asserted from prose."""
    labels = _strata(LONG_TAIL_12)
    assert len(labels) == 2376
    loose = DomainMix(tau=1.0, max_amplification=1e9,
                      min_stratum_episodes=1).report(labels)
    assert loose["max_amplification"] == pytest.approx(24.75, abs=0.01)
    assert loose["n_eff_frac"] == pytest.approx(0.1427, abs=0.001)
    with pytest.raises(ValueError, match="above the"):
        DomainMix(tau=1.0).episode_weights(labels)


def test_min_stratum_and_amplification_are_NOT_redundant_guards():
    """⚠️ The long-tail shape's smallest stratum is EXACTLY 8, so it PASSES
    ``min_stratum_episodes`` and is caught only by the amplification ceiling.
    Two guards, and the measurement says both are load-bearing."""
    labels = _strata(LONG_TAIL_12)
    assert min(LONG_TAIL_12.values()) == DOMAIN_MIX_MIN_STRATUM_EPISODES
    # min_stratum alone does NOT refuse it
    DomainMix(tau=1.0, max_amplification=1e9).episode_weights(labels)
    # the ceiling does
    with pytest.raises(ValueError, match="above the"):
        DomainMix(tau=1.0).episode_weights(labels)


def test_the_three_admissible_shapes_pass_the_default_ceiling():
    """The ceiling must ADMIT the shapes the programme plausibly has, or it is
    a lever nobody can pull."""
    for sizes in (REALISTIC, COARSE_3, FINE_10):
        labels = _strata(sizes)
        assert len(labels) == 2376
        DomainMix(tau=1.0).episode_weights(labels)


@pytest.mark.parametrize("sizes,amp,n_eff", [
    (REALISTIC, 11.00, 650.6),
    (COARSE_3, 4.50, 1030.1),
    (FINE_10, 14.85, 705.6),
    (LONG_TAIL_12, 24.75, 339.0),
])
def test_the_MEASURED_calibration_table_reproduces(sizes, amp, n_eff):
    """The table in ``DOMAIN_MIX_MAX_AMPLIFICATION``'s docstring, pinned.

    A constant justified by a measurement whose measurement is not re-runnable
    is a constant justified by prose.
    """
    rep = DomainMix(tau=1.0, max_amplification=1e9,
                    min_stratum_episodes=1).report(_strata(sizes))
    assert rep["max_amplification"] == pytest.approx(amp, abs=0.01)
    assert rep["n_eff_episodes"] == pytest.approx(n_eff, abs=0.1)


# ===========================================================================
# PARITY — the mix reweights, it never re-selects
# ===========================================================================

def test_an_UNLABELLED_episode_is_REFUSED_not_dropped():
    """⛔ Dropping it is a corpus RE-SELECTION, and a stratum-share report
    would not show it."""
    labels = _strata({"a": 40, "b": 40})
    labels[7] = None
    with pytest.raises(ValueError, match="carry NO stratum label"):
        DomainMix(tau=1.0).episode_weights(labels)
    labels[7] = "   "
    with pytest.raises(ValueError, match="carry NO stratum label"):
        DomainMix(tau=1.0).episode_weights(labels)


def test_every_episode_keeps_STRICTLY_POSITIVE_probability_at_full_balance():
    """The parity invariant at the weight level: all 2376 stay reachable."""
    labels = _strata(REALISTIC)
    w = DomainMix(tau=1.0).episode_weights(labels)
    assert w.numel() == 2376
    assert float(w.min()) > 0.0
    assert float(w.sum()) == pytest.approx(1.0, abs=1e-6)


def test_the_rarest_stratums_episodes_are_actually_DRAWN_at_full_balance():
    """The parity invariant at the DRAW level, not only the weight level."""
    sizes = {"big": 40, "small": 10}
    labels = _strata(sizes)
    index = _index(50, per_ep=4)
    ep_w = DomainMix(tau=1.0).episode_weights(labels)
    g = torch.Generator().manual_seed(3)
    s = StratifiedEpisodeSampler(index, torch.ones(len(index)),
                                 {i: float(w) for i, w in
                                  enumerate(ep_w.tolist())},
                                 eps_per_batch=4, generator=g)
    got = _draw(s, n=4000)
    seen = {index[i][0] for i in got}
    assert len(seen) == 50, f"only {len(seen)} of 50 episodes were reachable"


# ===========================================================================
# TAU SEMANTICS and the n_eff price
# ===========================================================================

def test_tau_0_is_UNIFORM_over_episodes_and_n_eff_is_exactly_n():
    labels = _strata(REALISTIC)
    rep = DomainMix(tau=0.0).report(labels)
    w = DomainMix(tau=0.0).episode_weights(labels)
    assert torch.allclose(w, torch.full_like(w, 1 / 2376), atol=1e-9)
    assert rep["n_eff_episodes"] == pytest.approx(2376.0, abs=0.01)
    assert rep["n_eff_frac"] == pytest.approx(1.0, abs=1e-4)
    assert rep["max_amplification"] == pytest.approx(1.0, abs=1e-4)
    # and at tau=0 the DRAWN stratum share is the CORPUS share
    # (1e-5: the drawn share sums float32 weights, the corpus share is exact)
    for k in REALISTIC:
        assert rep["stratum_share_drawn"][k] == pytest.approx(
            rep["stratum_share_corpus"][k], abs=1e-5)


def test_tau_1_gives_every_stratum_an_EQUAL_share():
    rep = DomainMix(tau=1.0).report(_strata(REALISTIC))
    for k in REALISTIC:
        assert rep["stratum_share_drawn"][k] == pytest.approx(1 / 6, abs=1e-6)


def test_n_eff_COLLAPSES_as_tau_balances_and_that_is_the_price():
    """⭐ The catalog row claims *diversity beats volume*. ``n_eff_episodes`` is
    the volume, and it is monotone in tau."""
    labels = _strata(REALISTIC)
    prev = None
    for tau in (0.0, 0.25, 0.5, 0.75, 1.0):
        rep = DomainMix(tau=tau, max_amplification=1e9).report(labels)
        if prev is not None:
            assert rep["n_eff_episodes"] < prev
        prev = rep["n_eff_episodes"]
    assert prev == pytest.approx(650.6, abs=0.1)
    # full balance on the realistic shape costs 73 % of the effective corpus
    assert 1 - 650.6 / 2376 == pytest.approx(0.726, abs=0.01)


@pytest.mark.parametrize("tau", [-0.01, 1.01, 2.0, -1.0])
def test_tau_outside_0_1_is_refused(tau):
    with pytest.raises(ValueError, match="tau must be in"):
        DomainMix(tau=tau)


def test_a_legal_DomainMix_construction_and_report_raise_NOTHING():
    """The paired opposite of every refusal above (C95/C97)."""
    labels = _strata(REALISTIC)
    mix = DomainMix(tau=0.5)
    w = mix.episode_weights(labels)
    rep = mix.report(labels)
    assert w.numel() == 2376 and rep["n_strata"] == 6
    assert rep["_reads"]
    json.dumps(rep)  # the run row must be serialisable


# ===========================================================================
# THE CONTROL — informativeness, and it refuses below n
# ===========================================================================

def test_domain_mix_control_REFUSES_below_min_n_and_returns_NO_ratio():
    m = torch.arange(16, dtype=torch.float32)
    labels = ["a"] * 8 + ["b"] * 8
    out = domain_mix_control(m, labels)
    assert out["verdict"] == "REFUSED_TOO_FEW"
    assert out["ratio"] is None, "a refused control must expose NO number"
    assert out["min_n"] == DOMAIN_MIX_CONTROL_MIN_N == 32


def test_domain_mix_control_flags_an_UNINFORMATIVE_stratification():
    """⛔ The failure a balance metric cannot see: strata that CUT ACROSS the
    quantity they are supposed to diversify. The strata ARE balanced; they just
    carry no information about the metric."""
    g = torch.Generator().manual_seed(0)
    m = torch.randn(200, generator=g)
    labels = ["a" if i % 2 == 0 else "b" for i in range(200)]
    out = domain_mix_control(m, labels)
    assert out["verdict"] == "UNINFORMATIVE", out
    assert out["ratio"] < 0.1
    # and the mix over these labels is perfectly balanced, which is the point
    rep = DomainMix(tau=1.0).report(labels)
    assert rep["stratum_share_drawn"]["a"] == pytest.approx(0.5, abs=1e-6)


def test_domain_mix_control_says_OK_on_a_genuinely_separating_stratification():
    """The paired positive control."""
    g = torch.Generator().manual_seed(0)
    m = torch.cat([torch.randn(100, generator=g),
                   torch.randn(100, generator=g) + 6.0])
    labels = ["slow"] * 100 + ["fast"] * 100
    out = domain_mix_control(m, labels)
    assert out["verdict"] == "OK", out
    assert out["ratio"] > 1.0


def test_domain_mix_control_reports_SEM_per_stratum():
    g = torch.Generator().manual_seed(1)
    m = torch.randn(120, generator=g)
    labels = ["a"] * 60 + ["b"] * 60
    out = domain_mix_control(m, labels)
    assert set(out["stratum_sem"]) == {"a", "b"}
    assert all(v is not None and v > 0 for v in out["stratum_sem"].values())
    assert out["stratum_n"] == {"a": 60, "b": 60}


def test_domain_mix_control_names_the_one_stratum_case():
    out = domain_mix_control(torch.randn(64), ["only"] * 64)
    assert out["verdict"] == "DEGENERATE_ONE_STRATUM"
    assert out["ratio"] is None


def test_domain_mix_control_names_the_zero_within_spread_case():
    m = torch.tensor([1.0] * 32 + [5.0] * 32)
    labels = ["a"] * 32 + ["b"] * 32
    out = domain_mix_control(m, labels)
    assert out["verdict"] == "DEGENERATE_ZERO_WITHIN"
    assert out["ratio"] is None, (
        "a zero within-spread makes the ratio UNDEFINED, not infinite")


def test_domain_mix_control_refuses_a_length_mismatch():
    with pytest.raises(ValueError, match="align 1:1"):
        domain_mix_control(torch.randn(40), ["a"] * 39)


# ===========================================================================
# THE ARTIFACT LOADER — every refusal, and its paired acceptance
# ===========================================================================

def _stable(i: int) -> int:
    return (1 << 40) + i


class _Ep:
    def __init__(self, eid: int):
        self.episode_id = eid


def _episodes(n: int = 40):
    return [_Ep(_stable(i)) for i in range(n)]


def _artifact(tmp_path: Path, *, n: int = 40, schema="domain-strata-v1",
              provenance=None, keys=None, drop=0) -> Path:
    strata = {}
    for i in range(n - drop):
        k = _stable(i) if keys is None else keys(i)
        strata[str(k)] = "urban" if i % 2 == 0 else "highway"
    blob = {"schema": schema, "strata": strata}
    if provenance is not None:
        blob["provenance"] = provenance
    p = tmp_path / "strata.json"
    p.write_text(json.dumps(blob), encoding="utf-8")
    return p


def _load(path, episodes):
    from train_v6_staged import load_domain_strata
    return load_domain_strata(path, episodes=episodes)


def test_the_loader_ACCEPTS_a_well_formed_declared_artifact(tmp_path):
    """The paired opposite of every refusal below."""
    p = _artifact(tmp_path, provenance={"source": "vlm-scena-v1",
                                        "built": "2026-08-18"})
    labels, prov = _load(p, _episodes())
    assert len(labels) == 40
    assert set(labels) == {"urban", "highway"}
    assert prov["source"] == "vlm-scena-v1"


def test_the_loader_refuses_an_UNDECLARED_artifact(tmp_path):
    """⛔ THE ADMISSIBILITY GUARD. A label-derived SAMPLER input is admissible
    only as a DECLARED data mix (DIAGRAM_CONFORMANCE.md:69)."""
    p = _artifact(tmp_path, provenance=None)
    with pytest.raises(SystemExit, match="NO 'provenance' stamp"):
        _load(p, _episodes())
    p = _artifact(tmp_path, provenance={})
    with pytest.raises(SystemExit, match="NO 'provenance' stamp"):
        _load(p, _episodes())


def test_the_loader_refuses_a_LEGACY_16bit_join_key(tmp_path):
    """⛔ The legacy id collides on 69 of 2400 train clips, so a join through it
    would put episodes in ANOTHER SCENE'S domain while the stratum shares still
    looked correct."""
    p = _artifact(tmp_path, provenance={"source": "x"}, keys=lambda i: 1000 + i)
    with pytest.raises(SystemExit, match="LEGACY 16-bit episode ids"):
        _load(p, _episodes())


def test_the_loader_refuses_a_legacy_id_on_the_EPISODE_side(tmp_path):
    p = _artifact(tmp_path, provenance={"source": "x"})
    eps = _episodes()
    eps[3] = _Ep(4242)
    with pytest.raises(SystemExit, match="LEGACY 16-bit id"):
        _load(p, eps)


def test_the_loader_refuses_an_UNLABELLED_episode_rather_than_dropping_it(
        tmp_path):
    """⛔ Dropping it re-selects the corpus, and a stratum-share report would
    show a beautiful mix over the labelled subset."""
    p = _artifact(tmp_path, provenance={"source": "x"}, drop=5)
    with pytest.raises(SystemExit, match="UNLABELLED"):
        _load(p, _episodes())


def test_the_loader_refuses_a_wrong_or_missing_schema(tmp_path):
    p = _artifact(tmp_path, schema="something-else",
                  provenance={"source": "x"})
    with pytest.raises(SystemExit, match="is not a domain-strata-v1"):
        _load(p, _episodes())


def test_the_loader_refuses_unreadable_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit, match="not readable JSON"):
        _load(p, _episodes())


def test_the_loader_refuses_an_empty_strata_map(tmp_path):
    p = tmp_path / "empty.json"
    p.write_text(json.dumps({"schema": "domain-strata-v1",
                             "provenance": {"s": 1}, "strata": {}}),
                 encoding="utf-8")
    with pytest.raises(SystemExit, match="no non-empty 'strata' map"):
        _load(p, _episodes())


# ===========================================================================
# THE SAMPLER's own refusals
# ===========================================================================

def test_the_sampler_refuses_an_episode_with_no_mix_weight():
    index = _index(4)
    with pytest.raises(ValueError, match="NO mix weight"):
        StratifiedEpisodeSampler(index, torch.ones(len(index)),
                                 {0: 0.25, 1: 0.25, 2: 0.5})


def test_the_sampler_refuses_a_zero_or_negative_episode_weight():
    index = _index(4)
    with pytest.raises(ValueError, match="strictly reachable"):
        StratifiedEpisodeSampler(index, torch.ones(len(index)),
                                 {0: 0.5, 1: 0.5, 2: 0.0, 3: 0.5})


def test_the_sampler_inherits_the_index_weight_alignment_check():
    with pytest.raises(ValueError, match="must align 1:1"):
        StratifiedEpisodeSampler(_index(4), torch.ones(3),
                                 {i: 0.25 for i in range(4)})


def test_a_legal_sampler_construction_and_draw_raise_NOTHING():
    index = _index(6)
    g = torch.Generator().manual_seed(0)
    s = StratifiedEpisodeSampler(index, torch.ones(len(index)),
                                 {i: 1 / 6 for i in range(6)},
                                 eps_per_batch=3, generator=g)
    got = s(8)
    assert len(got) == 8 and all(0 <= i < len(index) for i in got)


# ===========================================================================
# STAGE CONTRACT — zero keys, so nothing to adjudicate
# ===========================================================================

def test_F10_needs_no_STAGE_MAY_INTRODUCE_entry():
    """A zero-parameter cell adds no state_dict key, so the allowlist has
    nothing to adjudicate and must NOT grow. (The F-8/F-9/F-11 case, not the
    F-7 one, which added five keys and needed ``S-T``.)"""
    from train_v6_staged import STAGE_MAY_INTRODUCE
    for stage, allowed in STAGE_MAY_INTRODUCE.items():
        assert not any("domain" in x for x in allowed), (stage, allowed)
    assert STAGE_MAY_INTRODUCE["S-W"] == ()
    assert STAGE_MAY_INTRODUCE["S-J"] == ()


def test_the_06b8782_class_does_not_apply_to_F10():
    """Commit ``06b8782`` changed what S-J trains by APPENDING to
    ``MODULE_GROUPS`` without touching S-J's declaring line. F-10 introduces no
    parameter to assign to a group at all, so there is no mechanism by which it
    could move a stage's trainable set."""
    from tanitad.models.v6 import MODULE_GROUPS
    # MODULE_GROUPS is the tuple of GROUP NAMES; V6Stack._GROUP_PREFIXES maps a
    # state_dict prefix to one of them. F-10 adds to NEITHER, because it owns
    # no parameter to assign.
    assert "domain" not in MODULE_GROUPS and "mix" not in MODULE_GROUPS
    assert not any("domain" in p or "mix" in p
                   for p, _g in V6Stack._GROUP_PREFIXES)
    # the incumbent group set, pinned so an append is visible here too
    assert MODULE_GROUPS == ("encoder", "readout", "predictor_op", "layer_tac",
                             "layer_str", "planner", "aux", "interp")


def test_F10_is_not_a_LOSS_and_touches_no_weight():
    """A sampling mix is a schedule over the DATA. It must not appear in the
    loss-weight dataclass at all — an entry there would make it look like a
    term a stage could zero."""
    from train_v6_staged import V6LossWeights
    fields = set(V6LossWeights().__dataclass_fields__)
    assert not any("domain" in f for f in fields), fields


def test_the_mix_holds_no_nn_Module_and_no_tensor_parameter():
    mix = DomainMix(tau=1.0)
    assert not isinstance(mix, torch.nn.Module)
    assert not any(isinstance(v, torch.nn.Parameter)
                   for v in vars(mix).values())


@pytest.mark.slow
def test_default_build_is_untouched_at_the_production_geometry():
    """87,893,449 / 405 — the numbers a broken strict resume would kill.

    MEASURED by BUILDING through the real ``build_stack_from_args`` launch
    path, with F-10 both off and on.
    """
    from train_v6_staged import build_parser, build_stack_from_args

    def counts(*extra):
        a = build_parser().parse_args(
            ["--stage", "S-S", "--out", "unused"] + list(extra))
        torch.manual_seed(0)
        s = build_stack_from_args(a)
        return sum(p.numel() for p in s.parameters()), len(s.state_dict())

    assert counts() == (87_893_449, 405)
    assert counts("--domain-strata", "/tmp/x.json") == (87_893_449, 405)
    assert counts("--domain-strata", "/tmp/x.json",
                  "--domain-tau", "0") == (87_893_449, 405)
    # and against a hand-built default config, so the two paths agree
    torch.manual_seed(0)
    f = V6Stack(V6Config())
    assert (sum(p.numel() for p in f.parameters()),
            len(f.state_dict())) == (87_893_449, 405)


# ===========================================================================
# PREFLIGHT — refusals in milliseconds, and the legal launch raises nothing
# ===========================================================================

def _pf(*extra):
    from train_v6_staged import build_parser, preflight
    a = build_parser().parse_args(
        ["--stage", "S-S", "--out", "unused"] + list(extra))
    return preflight(a)


def test_preflight_refuses_domain_knobs_without_the_artifact():
    probs = _pf("--domain-tau", "0.5")
    assert any("without --domain-strata" in p for p in probs), probs


def test_preflight_refuses_tau_out_of_range(tmp_path):
    p = tmp_path / "s.json"
    p.write_text("{}", encoding="utf-8")
    probs = _pf("--domain-strata", str(p), "--domain-tau", "1.5")
    assert any("must be in [0, 1]" in x for x in probs), probs


def test_preflight_refuses_a_missing_artifact_path():
    probs = _pf("--domain-strata", "/definitely/not/here.json")
    assert any("does not exist" in p for p in probs), probs
    assert any("NO SCORE PRODUCER SHIPS WITH F-10" in p for p in probs), probs


def test_preflight_refuses_a_sub_one_amplification_ceiling(tmp_path):
    p = tmp_path / "s.json"
    p.write_text("{}", encoding="utf-8")
    probs = _pf("--domain-strata", str(p), "--domain-max-amp", "0.5")
    assert any("must be >= 1" in x for x in probs), probs


def test_preflight_PASSES_a_legal_F10_launch(tmp_path):
    """The paired opposite: a well-formed F-10 launch raises NOTHING."""
    p = _artifact(tmp_path, provenance={"source": "x"})
    probs = _pf("--domain-strata", str(p), "--domain-tau", "0.5")
    assert not [x for x in probs if "domain" in x], probs


def test_preflight_is_SILENT_about_F10_when_the_cell_is_off():
    assert not [p for p in _pf() if "domain" in p]


# ===========================================================================
# THE train() WIRING — order pinned against the source, composition executed
# ===========================================================================

_TRAINER = ROOT / "scripts" / "train_v6_staged.py"


def test_the_F10_swap_sits_AFTER_T3_and_BEFORE_the_T5_pair_block():
    """⛔ ORDER IS LOAD-BEARING and no log would show it wrong.

    F-10 must wrap whatever WINDOW weighting is in force, so it comes after
    O4/T3 construct theirs; and it must come before T5's ``sample.weights``
    mutation, which would otherwise be applied to an object F-10 then replaces
    — silently discarding the tail-window exclusion. Same idiom F-9 uses to pin
    its curriculum refresh ahead of the draw.
    """
    src = _TRAINER.read_text(encoding="utf-8")
    i_o4 = src.index("# ---- O4: interaction-weighted sampling")
    i_t3 = src.index("# ---- F-9 / catalog T3: the INTERACTION CURRICULUM")
    i_f10 = src.index("# ---- F-10 / catalog S3: the DOMAIN-STRATIFIED MIX")
    i_t5 = src.index("# ---- F-8 / T5: the CONSECUTIVE-WINDOW PAIR index")
    assert i_o4 < i_t3 < i_f10 < i_t5, (i_o4, i_t3, i_f10, i_t5)


def test_the_F10_block_reads_the_window_weights_DEFENSIVELY():
    """``make_sampler`` (the --o4-alpha 0 control arm) is a plain CLOSURE with
    no ``.weights``. Reading it unguarded would AttributeError on exactly the
    arm most likely to be run first, and only once a corpus was mounted."""
    src = _TRAINER.read_text(encoding="utf-8")
    blk = src.split("# ---- F-10 / catalog S3", 1)[1].split("# ---- F-8 / T5",
                                                            1)[0]
    assert 'getattr(sample, "weights", None)' in blk
    assert "sample.weights," not in blk, (
        "the F-10 block reads sample.weights directly — it must not")


def test_the_run_row_carries_the_mix_provenance_AND_its_price():
    """A declaration that lives in a console line nobody re-reads is the
    'please merge in a README' failure in a different costume."""
    src = _TRAINER.read_text(encoding="utf-8")
    assert 'cfg_json["domain_mix"] = dmixlog' in src
    rep = DomainMix(tau=0.5).report(_strata(REALISTIC))
    assert "n_eff_episodes" in rep and "stratum_share_drawn" in rep


def test_the_train_block_composition_runs_end_to_end(tmp_path):
    """The loader -> mix -> stratified sampler chain, executed in the order
    ``train()`` runs it, against a synthetic episode list. The corpus mount is
    what --dry-run cannot provide; this pins everything either side of it."""
    from train_v6_staged import load_domain_strata
    n = 40
    p = _artifact(tmp_path, n=n, provenance={"source": "vlm-scena-v1"})
    eps = _episodes(n)
    labels, prov = load_domain_strata(p, episodes=eps)
    mix = DomainMix(tau=1.0, min_stratum_episodes=8)
    ep_w = mix.episode_weights(labels)
    rep = mix.report(labels)
    index = _index(n, per_ep=5)
    g = torch.Generator().manual_seed(0)
    s = StratifiedEpisodeSampler(index, torch.ones(len(index)),
                                 {i: float(w) for i, w in
                                  enumerate(ep_w.tolist())},
                                 eps_per_batch=4, generator=g)
    got = _draw(s, n=2000)
    share = collections.Counter(labels[index[i][0]] for i in got)
    assert abs(share["urban"] / len(got) - 0.5) < 0.05
    dmixlog = {"domain_mix": "active", "provenance": prov, **rep}
    json.dumps(dmixlog)  # exactly what config.json receives
    assert dmixlog["n_strata"] == 2 and dmixlog["provenance"]["source"]


def test_the_REAL_strata_census_is_only_8_percent_of_the_parity_corpus():
    """⛔ THE ESCALATION, pinned as arithmetic.

    The only domain strata that exist for `physicalai-train-e438721ae894` cover
    **201 of 2376 episodes = 8.46 %** (aug120 road_type). A fully balanced mix
    on that shape puts **75 % of every batch on 8.46 % of the corpus** and
    leaves **10.2 %** of the effective volume. The mechanism is correct; the
    ARTIFACT is what makes it a sane arm, and it does not exist yet.
    """
    labelled = {"urban": 129, "highway": 38, "rural": 32, "unclear": 2}
    assert sum(labelled.values()) == 201
    assert round(201 / 2376 * 100, 2) == 8.46
    # (a) with `unclear` (n=2) kept, the min-stratum guard REFUSES
    a = {"unknown": 2376 - 201, **labelled}
    with pytest.raises(ValueError, match="fewer than"):
        DomainMix(tau=1.0).episode_weights(_strata(a))
    loose = DomainMix(tau=1.0, max_amplification=1e9,
                      min_stratum_episodes=1).report(_strata(a))
    assert loose["max_amplification"] == pytest.approx(237.6, abs=0.1)
    assert loose["n_eff_frac"] == pytest.approx(0.0186, abs=0.001)
    # (b) merged into `unknown`, it is ACCEPTED — and this is the price
    b = {"unknown": 2177, "urban": 129, "highway": 38, "rural": 32}
    rep = DomainMix(tau=1.0).report(_strata(b))
    assert rep["max_amplification"] == pytest.approx(18.56, abs=0.01)
    assert rep["n_eff_episodes"] == pytest.approx(243.2, abs=0.2)
    on_labelled = 1.0 - rep["stratum_share_drawn"]["unknown"]
    assert on_labelled == pytest.approx(0.75, abs=1e-4)
