"""``driving.py`` must emit a gate-readable ``cluster_bootstrap`` block.

The ⭐ v4 gate bug: a driving eval JSON carried its episode-cluster bootstrap
only under ``headline``, so ``run_gate._read_eval_metric`` found no
``cluster_bootstrap`` block and silently fell back to the DEPRECATED
``overlapping_holdout_se`` (1.28-2.06x too narrow). ``tier0`` now re-exposes the
headline intervals in the ``{"model": {...}}`` shape the gate reads.

Pinned here:
  * the block exists, is shaped for the gate, and every node names the primary
    estimator;
  * its ADE@2s interval REPRODUCES ``CI_RECOMPUTE_2026-07-20.json`` for
    flagship-30k ([0.3675, 0.4871]) — the recomputation the task requires;
  * the deprecated-estimator guard still passes on the enlarged block;
  * end to end, ``run_gate`` reads the emitted block and resolves ``miss_at_2m``.

CPU-only, runs against the committed ``results/windows_flagship-30k.pt``.
Standalone: ``python taniteval/tests/test_driving_gate_block.py``.
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))          # taniteval/
sys.path.insert(0, "/root/taniteval")              # pod layout

from taniteval import driving as D  # noqa: E402

PIN = _HERE.parents[1] / "results" / f"windows_{D.SANITY_ARM}.pt"
# Project Steering/CI_RECOMPUTE_2026-07-20.json, arm flagship-30k:
#   full_set_mean 0.4271 · boot_lo 0.3675 · boot_hi 0.4871
CI_RECOMPUTE = (0.4271, 0.3675, 0.4871)

_CACHE = {}


def _block(n_boot=2000):
    if not PIN.exists():
        raise AssertionError(f"{PIN} missing — the committed dump is required")
    if n_boot not in _CACHE:
        _CACHE[n_boot] = D.from_windows(PIN, n_boot=n_boot, arm=D.SANITY_ARM)
    return _CACHE[n_boot]


def test_cluster_bootstrap_block_is_present_and_shaped_for_the_gate():
    out = _block()
    assert out["primary_ci"] == "episode_cluster_bootstrap"
    model = out["cluster_bootstrap"]["model"]
    for k in ("ade_0_2s", "miss_2m", "speed_mae_mps", "lat_abs_2s_m"):
        assert k in model, k
        assert model[k]["estimator"] == "episode_cluster_bootstrap", k
        assert "mean" in model[k] and "lo" in model[k] and "hi" in model[k], k


def test_recomputed_ade_ci_matches_ci_recompute():
    """The whole point: the interval the gate will read reproduces the published
    episode-cluster bootstrap to 4 decimals."""
    mean, lo, hi = CI_RECOMPUTE
    node = _block()["cluster_bootstrap"]["model"]["ade_0_2s"]
    assert node["mean"] == mean and node["lo"] == lo and node["hi"] == hi, node


def test_cluster_block_mirrors_headline_exactly():
    out = _block()
    for k, v in out["cluster_bootstrap"]["model"].items():
        assert v == out["headline"][k], k


def test_enlarged_block_still_passes_the_deprecated_estimator_guard():
    """Adding the block must not introduce a forbidden estimator anywhere."""
    assert D.assert_no_deprecated_estimator(_block()) is True


def test_run_gate_reads_the_emitted_block_end_to_end():
    """Integration: driving emits -> run_gate reads, incl. the miss alias.
    Skips loudly if the stack scripts are not importable (pod layout)."""
    try:
        sys.path.insert(0, str(_HERE.parents[2] / "stack" / "scripts"))
        sys.path.insert(0, "/root/TanitAD/stack/scripts")
        import run_gate as rg
    except Exception as e:                                         # noqa: BLE001
        print(f"  (run_gate integration UNVERIFIED here: {type(e).__name__}: {e})")
        return
    out = _block()
    val, prov = rg._read_eval_metric(out, "ade_0_2s")
    assert val == CI_RECOMPUTE[0]
    assert "episode_cluster_bootstrap" in prov
    assert f"CI [{CI_RECOMPUTE[1]}, {CI_RECOMPUTE[2]}]" in prov, prov
    # the card's canonical miss name resolves to driving's miss_2m
    mval, mprov = rg._read_eval_metric(out, "miss_at_2m")
    assert mval == out["headline"]["miss_2m"]["mean"]
    assert "miss_2m" in mprov


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    bad = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:                                    # noqa: BLE001
            bad += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"==== {len(fns) - bad}/{len(fns)} passed ====")
    sys.exit(1 if bad else 0)
