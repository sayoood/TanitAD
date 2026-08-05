"""``tools/eval_four_families.py`` — the contract, pinned.

⛔ WHY THIS TOOL AND THIS TEST EXIST. ``stack/scripts/eval_flagship_v4.py`` gates
its full metric path on ``is_v4 = isinstance(ck, dict) and ("head" in ck)``. A
v1-shaped flagship checkpoint has keys ``grounding``/``model``/``opt``/``step``
and no ``head``, so on such a checkpoint that script can ONLY run
``MODE_A_canary_only_validation`` — it never emits per-window ``pred``/``gt`` and
so cannot produce a single one of the four binding families. That structural
limit is easy to mistake for "the eval ran", because MODE-A exits 0 and prints a
number. These tests pin the properties that make the replacement admissible.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest
import torch

TOOL = Path(__file__).resolve().parents[1] / "tools" / "eval_four_families.py"


def _tree():
    return ast.parse(TOOL.read_text())


def test_the_tool_exists_and_parses():
    assert TOOL.exists(), f"{TOOL} is the four-family entry point"
    _tree()


def test_it_runs_BOTH_passes_because_neither_alone_can_fill_four_families():
    """The fidelity pass gives LONGITUDINAL/LATERAL; only the hierarchy pass
    traverses the decision heads. A tool that called one of them would report two
    families UNAVAILABLE and look like a completed eval."""
    src = TOOL.read_text()
    assert "rollout.collect(" in src
    assert "hierarchy.run(" in src
    assert "all_families(" in src


def test_hierarchy_is_not_left_at_its_40_episode_default():
    """``hierarchy.run``'s ``max_eps`` defaults to 40. On a 290-episode corpus
    that silently scores 14 % of it — the same class of defect as the ``--episodes``
    default that returned a 17 %-optimistic number on this very corpus."""
    src = TOOL.read_text()
    assert "max_eps=len(eps)" in src, (
        "hierarchy.run must be told the real episode count, not left at 40")


def test_episodes_flag_is_recorded_in_the_output_not_just_honoured():
    """A truncated denominator that is not in the record is indistinguishable
    from a full run once the log scrolls away."""
    src = TOOL.read_text()
    for k in ("episodes_scored", "episodes_available", "episodes_flag"):
        assert f'"{k}"' in src, f"the output record must carry {k}"


def test_the_banned_estimator_is_named_as_banned_and_never_called():
    src = TOOL.read_text()
    assert "overlapping_holdout_se" in src, (
        "the record must say which estimator is NOT used — a bare 'CI' invites "
        "the reader to assume the old one")
    tree = _tree()
    called = {n.func.attr for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert "overlapping_holdout_se" not in called
    assert {"bootstrap_metrics", "episode_cluster_bootstrap"} & called, (
        "intervals must come from the episode-cluster bootstrap")


def test_ci_components_reuse_four_families_own_geometry():
    """⛔ A second implementation of the geometry here would let the interval and
    the point estimate drift apart silently — the exact failure the estimator
    rule exists to prevent."""
    src = TOOL.read_text()
    assert "ff._seq_geometry(" in src, (
        "per-window CI components must come from four_families' own "
        "_seq_geometry, not a re-derivation")


def test_out_refuses_a_directory():
    """MEASURED failure on this corpus: --out given a directory raised
    IsADirectoryError only AFTER the whole scoring pass had run."""
    src = TOOL.read_text()
    assert "os.path.isdir(a.out)" in src
    i_guard = src.index("os.path.isdir(a.out)")
    i_load = src.index("loaders.load(")
    assert i_guard < i_load, "the guard must fire BEFORE the expensive work"


def test_skip_hierarchy_says_the_result_is_inadmissible():
    src = TOOL.read_text()
    seg = src[src.index("--skip-hierarchy"):src.index("--skip-hierarchy") + 400]
    assert "UNAVAILABLE" in seg and "NOT admissible" in seg


def test_vision_only_and_binding_rules_travel_with_the_record():
    """The record has to carry the reading rule, because the number outlives the
    conversation that produced it. ``route_acc_nav`` near 1.0 is an ECHO of the
    model's own input — v1 MEASURED exactly 1.0000 — and a reader with only the
    JSON must be told that."""
    src = TOOL.read_text()
    assert '"_vision_only"' in src and "route_acc_follow" in src
    assert '"_binding"' in src and "WORK ITEM" in src


def test_corpus_identity_is_recorded_so_arms_are_not_cross_compared_blindly():
    src = TOOL.read_text()
    assert '"corpus"' in src and '"corpus_key"' in src


# --- the numeric contract the tool depends on, on synthetic windows --------- #

def _win(n=24, h=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    gt = torch.cumsum(torch.rand(n, h, 2, generator=g) * 0.3, dim=1)
    pred = gt + torch.randn(n, h, 2, generator=g) * 0.05
    return {"pred": pred[:, [4, 9, 14, 19]], "gt": gt[:, [4, 9, 14, 19]],
            "pred_dense": pred, "gt_dense": gt,
            "wp_steps": [5, 10, 15, 20], "dense_steps": list(range(1, h + 1)),
            "dt_s": 0.1, "eid": [i // 6 for i in range(n)]}


def test_per_window_components_reduce_to_the_family_point_estimates():
    """The interval and the family scalar must be the SAME number's spread and
    centre. If these two ever disagree the tool is reporting a CI around a
    quantity it is not printing."""
    from taniteval import four_families as ff
    w = _win()
    fam = ff.all_families(w)
    P = ff._seq_geometry(w["pred_dense"], 0.1)
    G = ff._seq_geometry(w["gt_dense"], 0.1)
    sp = (P["speed"] - G["speed"]).abs()
    assert fam["longitudinal"]["speed_mae_mps"] == pytest.approx(
        float(sp.mean()), abs=1e-4)
    # the tool's per-window component is the per-window mean; its mean over
    # windows equals the pooled mean because every window has the same H
    assert float(sp.mean(1).mean()) == pytest.approx(float(sp.mean()), abs=1e-6)


def test_a_window_with_no_valid_step_is_dropped_not_counted_as_zero_error():
    """A stopped vehicle has no path tangent. Counting its heading error as 0
    would make a corpus of stopped windows look perfectly steered."""
    from taniteval import four_families as ff
    w = _win()
    w["pred_dense"][:4] = 0.0                      # 4 windows: no motion at all
    w["gt_dense"][:4] = 0.0
    P = ff._seq_geometry(w["pred_dense"], 0.1)
    G = ff._seq_geometry(w["gt_dense"], 0.1)
    both = P["valid"] & G["valid"]
    assert int((~both[:4]).sum()) == both.shape[1] * 4
    fam = ff.all_families(w)
    assert fam["lateral"]["excluded_below_min_ds"] >= both.shape[1] * 4


@pytest.mark.skipif(not TOOL.exists(), reason="tool missing")
def test_cli_help_works_without_a_gpu():
    """--help must not import torch.cuda or a checkpoint. A tool whose --help
    needs a GPU cannot be inspected on the box you are debugging from."""
    r = subprocess.run([sys.executable, str(TOOL), "--help"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-800:]
    assert "--corpus" in r.stdout and "--skip-hierarchy" in r.stdout
