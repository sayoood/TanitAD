"""⛔ THE THREE LIVE `overlapping_holdout_se` EMISSION PATHS, CLOSED.

PI ruling 2026-08-28: `overlapping_holdout_se` is FORBIDDEN as a decision-grade
estimator. It is not only 1.11-3.10x too narrow (median 1.50x) — it BIASES THE
POINT ESTIMATE, bidirectionally: -6.67 % to +11.69 % on headline `ade_0_2s`
across the 27 banked window dumps, up to x3.3 on hierarchy seams, and up to
x-4.15 INCLUDING A SIGN FLIP on paired deltas. A number computed with it can be
wrong in sign, so a gate decided on it can be wrong in verdict.

The full closeout, the number-movement ledger and what remains open live in
`products/P7-TanitEval/ESTIMATOR_CLOSEOUT.md`. This file is the mechanical half.

WHY EACH TEST HAS A DELIBERATE-REGRESSION ARM
---------------------------------------------
A guard never shown to FAIL proves nothing. Every closure below is paired with
a fixture that re-introduces the exact defect and asserts the guard catches it —
because three of this programme's instruments were later found to be
structurally unable to report the answer they were cited for (`df` on a pod, the
`_files_under` skip that scanned 373 of 373 files as zero, and the shape
detector that had no caller at all).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from taniteval import bench, report, runner
from taniteval import gate_guard as gg

REPO = Path(__file__).resolve().parents[2]
N_WP = 4
METRICS = ("ade_0_2s", "fde@2s", "miss_rate@2m", "tms_openloop")


def _win(n_ep=10, per=9, seed=0):
    """Windows with real per-EPISODE structure.

    The between-cluster variance is what the overlapping-holdout block
    under-counts and the episode-cluster bootstrap does not, so a fixture
    without it would make the two estimators agree and every assertion below
    would pass vacuously."""
    g = torch.Generator().manual_seed(seed)
    n = n_ep * per
    eid = [i // per for i in range(n)]
    off = torch.rand(n_ep, generator=g) * 2.0
    base = torch.stack([off[e].expand(N_WP * 2) for e in eid]).reshape(n, N_WP, 2)
    return {"pred": base + 0.1 * torch.rand(n, N_WP, 2, generator=g),
            "gt": torch.zeros(n, N_WP, 2), "cv": base * 1.3,
            "speed": torch.rand(n, generator=g) * 10,
            "head_deg": torch.randn(n, generator=g) * 5,
            "eid": eid, "wp_steps": (5, 10, 15, 20)}


@pytest.fixture(scope="module")
def res():
    return bench.run(_win(), n_boot=400)


# =========================================================================== #
# SITE 1 — bench.run emitted the legacy block TWICE, the second time bare      #
# =========================================================================== #
class TestSite1BenchBareHeldoutKey:
    """`bench.py:303` wrote `"heldout": legacy_block` — the SAME dict as the
    quarantined `LEGACY_BLOCK`, under a key carrying no verdict word and so
    invisible to `gate_guard`. Five consumers read it as the arm's headline."""

    def test_no_number_is_quotable_from_the_bare_key(self, res):
        for side in ("model", "cv"):
            for m, node in res[bench.HELDOUT_ALIAS][side].items():
                assert not {"mean", "ci95", "std"} & set(node), (
                    f"heldout.{side}.{m} still exposes a number: {node}")

    def test_the_numbers_survive_bit_identical_under_the_labelled_key(self, res):
        """History is CLOSED, not deleted — published figures stay traceable."""
        from taniteval.bench import _agg, _suite
        from tanitad.eval.gates import split_by_episode
        w = _win()
        splits = [split_by_episode(w["eid"], 0.2, s) for s in range(0, 8)]
        expect = _agg([_suite(w["pred"][va], w["gt"][va]) for _t, va in splits])
        got = res[bench.LEGACY_BLOCK]["model"]
        for m in expect:
            assert got[m]["mean"] == pytest.approx(expect[m]["mean"], abs=1e-12)
            assert got[m]["ci95"] == pytest.approx(expect[m]["ci95"], abs=1e-12)

    def test_bench_run_now_calls_the_policy_guard(self, res):
        """`bench.run` was the ONLY block emitter that never ran
        `assert_no_deprecated_estimator`. closedloop / hierarchy / planner_p2 /
        driving / corridor / lateral / strategic_probes all did."""
        assert bench.assert_no_deprecated_estimator(res) is True

    def test_the_guard_REFUSES_a_block_that_leaks_the_deprecated_estimator(self):
        """⭐ DELIBERATE REGRESSION: put the legacy numbers back under a
        non-quarantined key and require the guard to refuse the block."""
        leaked = {"cluster_bootstrap": {"model": {}},
                  "headline": {"ade_0_2s": {
                      "mean": 0.4522, "ci95": 0.0312, "lo": 0.42, "hi": 0.48,
                      "estimator": bench.DEPRECATED_ESTIMATOR}}}
        with pytest.raises(ValueError, match="overlapping_holdout_se"):
            bench.assert_no_deprecated_estimator(leaked)

    def test_the_guard_REFUSES_an_interval_with_no_named_estimator(self):
        """The unlabelled-interval half of the same policy."""
        with pytest.raises(ValueError, match="without a named estimator"):
            bench.assert_no_deprecated_estimator(
                {"headline": {"ade_0_2s": {"mean": 0.45, "lo": 0.4, "hi": 0.5}}})

    def test_the_run_gate_tripwire_is_STILL_ARMED_on_the_tombstone(self, res):
        """⭐ The half that is easy to break while fixing the other half.

        `run_gate._deprecated_present` searches the literal `("heldout",
        "model")` path and accepts a node on `"mean" in n` OR
        `n["estimator"] == DEPRECATED_ESTIMATOR`. Dropping the numbers leaves
        the SECOND branch firing — which is the branch written for exactly this
        shape. Deleting the key instead would have silently disarmed the gate's
        own refusal, i.e. traded one silent failure for a worse one."""
        rg = _load_run_gate()
        assert rg._deprecated_present(res, ("ade_0_2s",)) is True
        only_dep = {"full_set": res["full_set"],
                    "heldout": res[bench.HELDOUT_ALIAS]}
        with pytest.raises(SystemExit, match="DEPRECATED"):
            rg._read_eval_metric(only_dep, "ade_0_2s")

    def test_a_tombstone_shaped_node_is_not_mistaken_for_a_reading(self):
        """⭐ DELIBERATE REGRESSION on the tombstone itself: if a future edit
        puts `mean` back, `_deprecated_present` must still fire (it does — the
        first branch) and `_read_eval_metric` must still refuse. The tombstone
        is a belt on top of that brace, never a replacement for it."""
        rg = _load_run_gate()
        regressed = {"full_set": {"model": {"ade_0_2s": 0.4271}},
                     "heldout": {"model": {"ade_0_2s": {
                         "mean": 0.4522, "ci95": 0.0312,
                         "estimator": "overlapping_holdout_se"}}}}
        assert rg._deprecated_present(regressed, ("ade_0_2s",)) is True
        with pytest.raises(SystemExit):
            rg._read_eval_metric(regressed, "ade_0_2s")


def _load_by_path(name, path):
    """Import a module that is not on a package path (``taniteval/tests`` has
    no ``__init__.py``, and ``stack/scripts`` is a script directory)."""
    import importlib.util
    import sys
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_run_gate():
    return _load_by_path("run_gate", REPO / "stack" / "scripts" / "run_gate.py")


def _load_guard_test():
    return _load_by_path(
        "_guard_scope_probe",
        REPO / "taniteval" / "tests" / "test_no_jack_in_gates.py")


# =========================================================================== #
# SITE 1b — runner.regression FELL BACK to the biased estimator                #
# =========================================================================== #
class TestSite1bRegressionGateRefuses:
    """`runner.py:409-410` did `block, src = d["heldout"]["model"],
    "heldout(DEPRECATED)"` — a labelled but still SILENT substitution. The
    golden file would then hold a mean-of-split-means for some arms and a
    full-set mean for others, and the ±8 % tolerance is compared across a
    -6.67 %…+11.69 % estimator gap. That is not a weaker comparison; it is not
    a comparison."""

    @staticmethod
    def _deprecated_only_result():
        return {"heldout": {"model": {m: {
            "mean": 0.4522, "ci95": 0.0312,
            "estimator": "overlapping_holdout_se", "deprecated": True}
            for m in ("ade_0_2s", "fde@2s", "miss_rate@2m")}}}

    @staticmethod
    def _primary_result(mean=0.4271):
        return {"cluster_bootstrap": {"model": {m: {
            "mean": mean, "lo": mean - 0.06, "hi": mean + 0.06,
            "ci95": 0.06, "estimator": "episode_cluster_bootstrap"}
            for m in ("ade_0_2s", "fde@2s", "miss_rate@2m")}}}

    def _run(self, tmp_path, files, monkeypatch, **kw):
        for name, doc in files.items():
            (tmp_path / f"{name}.json").write_text(json.dumps(doc))
        monkeypatch.setattr(runner, "RES", tmp_path)
        return runner.regression(**kw)

    def test_it_REFUSES_instead_of_falling_back(self, tmp_path, monkeypatch,
                                                capsys):
        ok = self._run(tmp_path, {"arm_a": self._deprecated_only_result()},
                       monkeypatch)
        assert ok is False, "the gate must FAIL, not substitute"
        out = capsys.readouterr().out
        assert "REFUSED" in out and "arm_a" in out
        assert "overlapping_holdout_se" in out
        assert not (tmp_path / "golden.json").exists(), \
            "a refused arm must never reach golden.json"

    def test_a_tombstoned_artifact_reaches_the_SAME_refusal(self, tmp_path,
                                                            monkeypatch, capsys):
        """⭐ The subtle one. Post-closeout artifacts carry a `heldout` key with
        NO `mean`. A refusal keyed on `"mean"` would let those fall through as
        "no block at all" and be silently DROPPED from the gate instead of
        named — swapping a loud wrong answer for a quiet missing one."""
        tomb = {"heldout": bench._alias_tombstone(
            {"model": {m: {} for m in ("ade_0_2s", "fde@2s", "miss_rate@2m")}})}
        ok = self._run(tmp_path, {"arm_t": tomb}, monkeypatch)
        assert ok is False
        assert "arm_t" in capsys.readouterr().out

    def test_a_primary_only_arm_passes_normally(self, tmp_path, monkeypatch):
        assert self._run(tmp_path, {"arm_ok": self._primary_result()},
                         monkeypatch, update=True) is True
        assert json.loads((tmp_path / "golden.json").read_text())["arm_ok"][
            "ade_0_2s"] == 0.4271

    def test_allow_missing_downgrades_but_never_substitutes(self, tmp_path,
                                                            monkeypatch):
        """The escape hatch must not restore the old behaviour: the arm is
        still SKIPPED, never read off the deprecated block."""
        ok = self._run(tmp_path,
                       {"arm_ok": self._primary_result(),
                        "arm_bad": self._deprecated_only_result()},
                       monkeypatch, update=True, allow_missing=True)
        golden = json.loads((tmp_path / "golden.json").read_text())
        assert ok is True
        assert "arm_ok" in golden and "arm_bad" not in golden


# =========================================================================== #
# SITE 1c — the leaderboard was ORDERED on the biased estimator                #
# =========================================================================== #
class TestSite1cLeaderboardOrdering:
    """`report.py:72` sorted on `heldout.model.ade_0_2s.mean`. Because the bias
    is BIDIRECTIONAL the ordering itself was partly an estimator artifact —
    two arms can swap places without either model changing."""

    @staticmethod
    def _arm(name, primary, legacy):
        """One leaderboard row. The legacy means are deliberately far from any
        rendered CI bound so "is this string on the page?" cannot be satisfied
        by an unrelated number — the fixture must be able to tell the two
        estimators apart by their printed digits alone."""
        return {"model": {"name": name}, "n_windows": 881, "ckpt_step": 30000,
                "cluster_bootstrap": {
                    "model": {m: {"mean": primary, "lo": primary - 0.011,
                                  "hi": primary + 0.011, "ci95": 0.011,
                                  "estimator": "episode_cluster_bootstrap"}
                              for m in (*METRICS, "rmse")},
                    "cv": {m: {"mean": 0.8377, "lo": 0.8, "hi": 0.88,
                               "ci95": 0.04,
                               "estimator": "episode_cluster_bootstrap"}
                           for m in (*METRICS, "rmse")}},
                "heldout": {"model": {m: {
                    "mean": legacy, "ci95": 0.0312,
                    "estimator": "overlapping_holdout_se"}
                    for m in (*METRICS, "rmse")},
                    "cv": {m: {"mean": 0.83, "ci95": 0.03,
                               "estimator": "overlapping_holdout_se"}
                           for m in (*METRICS, "rmse")}}}

    def test_the_order_follows_the_primary_not_the_legacy_estimator(self):
        """⭐ DELIBERATE REGRESSION, built as an ORDER INVERSION: arm A is
        better on the primary and worse on the legacy block. A leaderboard
        still keyed on the old number would put B first. This is the failure in
        miniature — the ranking moved without either model changing, which is
        exactly what the bidirectional point-estimate bias does across arms."""
        rows = report._lb_rows({
            "a": self._arm("ARM-A", primary=0.401, legacy=0.911),
            "b": self._arm("ARM-B", primary=0.452, legacy=0.822)})
        assert rows.index("ARM-A") < rows.index("ARM-B"), \
            "the leaderboard is still ordered on overlapping_holdout_se"
        assert "0.401" in rows and "0.452" in rows
        assert "0.911" not in rows and "0.822" not in rows, \
            "a legacy number is still being displayed"

    def test_an_arm_with_only_the_legacy_block_is_marked_not_backfilled(self):
        legacy_only = self._arm("ARM-OLD", primary=0.401, legacy=0.911)
        legacy_only.pop("cluster_bootstrap")
        rows = report._lb_rows({"old": legacy_only})
        assert "NO PRIMARY INTERVAL" in rows
        assert "0.911" not in rows, "the deprecated value was back-filled"

    def test_a_block_that_does_not_NAME_the_primary_estimator_is_rejected(self):
        """⭐ DELIBERATE REGRESSION: a `cluster_bootstrap`-shaped block whose
        nodes carry the DEPRECATED estimator name must not be promoted just
        because it sits under the right key."""
        forged = self._arm("ARM-FORGED", primary=0.401, legacy=0.911)
        for node in forged["cluster_bootstrap"]["model"].values():
            node["estimator"] = "overlapping_holdout_se"
        assert report._primary(forged) is None
        assert "NO PRIMARY INTERVAL" in report._lb_rows({"f": forged})


# =========================================================================== #
# SITE 2 — recompute_ci.py, an unguarded standalone caller                     #
# =========================================================================== #
class TestSite2RecomputeCiIsInScope:
    """`taniteval/recompute_ci.py:91` called `ci.overlapping_holdout_se` from
    OUTSIDE `ENFORCED_ROOTS`, which covered `taniteval/taniteval` and
    `taniteval/tools` but not the `taniteval/` top level where ~35 standalone
    drivers live. It was unguarded by accident, not by decision.

    It is legal — it is the instrument that MEASURES the defect, and refusing
    to compute the BEFORE column would destroy the evidence that the AFTER
    column is needed. What it was missing is a name that says so."""

    def test_the_top_level_is_now_inside_the_enforced_scope(self):
        T = _load_guard_test()
        roots = {(Path(r).resolve(), p) for r, p in T.ENFORCED_ROOTS}
        assert ((REPO / "taniteval").resolve(), "*.py") in roots, \
            "the taniteval/ top level is not scanned — recompute_ci.py and " \
            "~35 sibling drivers are unguarded again"
        # and the scan must actually SEE this file, not merely name its parent
        seen = {Path(f).resolve() for r, p in T.ENFORCED_ROOTS
                for f in gg._files_under(r, p, T.SKIP)}
        assert (REPO / "taniteval" / "recompute_ci.py").resolve() in seen

    def test_the_reproduction_declares_its_estimator_in_its_own_name(self):
        src = (REPO / "taniteval" / "recompute_ci.py").read_text(
            encoding="utf-8")
        assert "def reproduce_overlapping_holdout_published(" in src
        assert "def naive_published(" not in src, (
            "the old name announced nothing — the same defect as "
            "driving_diagnostic.mean_ci: right arithmetic, invisible name")
        assert gg.is_declared_estimator_name(
            "reproduce_overlapping_holdout_published")

    def test_it_carries_no_verdict_key_and_so_passes_the_name_guard(self):
        src = (REPO / "taniteval" / "recompute_ci.py").read_text(
            encoding="utf-8")
        assert gg.scan_source(src, "recompute_ci.py") == []
        assert gg.scan_source_shapes(src, "recompute_ci.py") == []

    def test_the_guard_WOULD_fire_if_it_ever_decided_something(self):
        """⭐ DELIBERATE REGRESSION: the reason bringing this file into scope is
        worth doing. Route its BEFORE column into a verdict and the guard must
        refuse it."""
        regressed = '''
def main(win):
    rep = reproduce_overlapping_holdout_published(win)
    return {"G1_pass": bool(rep[0] < 0.4522)}
'''
        assert gg.scan_source(regressed, "recompute_ci.py"), \
            "the name guard did not fire on a verdict from the BEFORE column"

    def test_it_reads_the_published_value_from_EITHER_artifact_vintage(self,
                                                                       tmp_path):
        """The ledger tool must keep reproducing history across the closeout:
        pre-2026-08-28 files carry the numbers under `heldout`, post files
        carry them under `legacy_overlapping_holdout_se`."""
        src = (REPO / "taniteval" / "recompute_ci.py").read_text(
            encoding="utf-8")
        assert "legacy_overlapping_holdout_se" in src
        assert 'd.get("heldout", {})' in src


# =========================================================================== #
# SITE 3 — driving_diagnostic.mean_ci, the UNNAMED CLONE                       #
# =========================================================================== #
class TestSite3UnnamedClone:
    """`stack/scripts/driving_diagnostic.py:167-177` re-typed
    `1.96 * std / sqrt(n)` over per-split means under a name no grep for
    `jack` or `overlapping_holdout` would return, with NO `estimator` field,
    reachable from six live callers (the brief named two).

    ⚠️ The arithmetic is preserved VERBATIM including ddof=1 — delegating to
    `ci.overlapping_holdout_se` (ddof=0) would rescale every published D-number
    by sqrt((n-1)/n). A reproduction that changes the number is not one."""

    @staticmethod
    def _dd():
        import driving_diagnostic as dd
        return dd

    def test_the_output_now_self_labels(self):
        node = self._dd().mean_ci([1.0, 2.0, 3.0, 4.0])
        assert node["estimator"] == "overlapping_holdout_se"
        assert node["deprecated"] is True

    def test_the_numbers_are_bit_identical_to_the_pre_closeout_clone(self):
        def pre_closeout(vals):                    # the removed body, verbatim
            n = len(vals)
            m = sum(vals) / n
            std = (sum((v - m) ** 2 for v in vals) / max(1, n - 1)) ** 0.5
            return {"mean": round(m, 4),
                    "ci95": round(1.96 * std / n ** 0.5, 4),
                    "std": round(std, 4), "n_splits": n,
                    "per_split": [round(v, 4) for v in vals]}
        for vals in ([1.0, 2.0, 3.0], [0.41, 0.47, 0.44, 0.52, 0.39, 0.48,
                                       0.45, 0.43]):
            got, want = self._dd().mean_ci(vals), pre_closeout(vals)
            for k in want:
                assert got[k] == want[k], f"{k} moved: {got[k]} != {want[k]}"

    def test_every_caller_keeps_working_through_the_alias(self):
        dd = self._dd()
        assert dd.mean_ci is dd.overlapping_holdout_mean_ci
        agg = dd.agg_metric_dicts([{"ade_0_2s": 1.0}, {"ade_0_2s": 2.0}])
        assert agg["ade_0_2s"]["estimator"] == "overlapping_holdout_se"

    def test_the_labelled_output_is_now_REFUSED_by_the_policy_guard(self):
        """⭐ The teeth. Before the closeout `mean_ci`'s output carried no
        estimator at all, so `driving.assert_no_deprecated_estimator` walked
        straight past it. Now any block containing it is refused."""
        from taniteval import driving as drv
        block = {"decode_ladder": self._dd().mean_ci([1.0, 2.0, 3.0])}
        with pytest.raises(ValueError, match="overlapping_holdout_se"):
            drv.assert_no_deprecated_estimator(block)

    def test_both_spellings_are_in_the_name_guards_ban_list(self):
        assert {"mean_ci", "overlapping_holdout_mean_ci"} <= gg.BANNED_CALLS

    def test_the_shape_guard_still_catches_the_class(self):
        """⭐ DELIBERATE REGRESSION. The rename makes THIS function a declared
        reproduction, which the shape detector exempts by design. The detector
        must still fire on the same arithmetic under a fresh innocent name —
        otherwise the fix would have disarmed the guard rather than used it."""
        reintroduced = '''
def summarise_holdouts(vals):
    n = len(vals)
    m = sum(vals) / n
    std = (sum((v - m) ** 2 for v in vals) / max(1, n - 1)) ** 0.5
    return {"mean": m, "ci95": 1.96 * std / n ** 0.5}
'''
        assert gg.scan_source_shapes(reintroduced, "fresh_clone.py")

    def test_a_dispersion_laundered_through_an_innocent_NAME_is_caught(self):
        """⭐ A HOLE THIS WORK FOUND (2026-08-28), not one it was sent to fix.

        The shape detector was still name-keyed in its NUMERATOR: `spread =
        np.std(v)` then `1.96 * spread / sqrt(n)` escaped, because `spread` is
        not in `_DISPERSION_NAME_RE`. Found by writing the deliberate-
        regression arm this brief demanded. `_dispersion_names` now follows the
        data there too."""
        laundered = '''
def summarise_splits(vals):
    n = len(vals)
    spread = np.std(vals)
    alias = spread
    return {"mean": float(np.mean(vals)),
            "ci95": 1.959964 * alias / np.sqrt(n)}
'''
        assert gg.scan_source_shapes(laundered, "laundered.py")

    def test_the_mean_of_a_dispersion_free_expression_is_NOT_flagged(self):
        """False-positive control: `_dispersion_names` must not swallow every
        local name. A quantile times a plain mean is not an SE."""
        innocent = '''
def scale(vals):
    n = len(vals)
    centre = np.mean(vals)
    return 1.96 * centre / np.sqrt(n)
'''
        assert gg.scan_source_shapes(innocent, "innocent.py") == []
