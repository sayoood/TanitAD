"""The eval window dump's CONTRACT: rows the paired bootstrap can actually consume.

⛔ WHY IT EXISTS. `taniteval.ci.paired_episode_cluster_bootstrap(a, b, eid, ...)` is the
ONLY admissible interval in this programme, and until 2026-09-19 no refcv6 artifact could
feed it: `compute_losses_v3` ALREADY REDUCES OVER THE BATCH and the eval then means those
batch scalars, so `metrics.jsonl` carried 0 non-scalar values and no id. An aggregate-only
dump cannot be bootstrapped BY ANY RESCORE.

⭐ THE ID WAS ALREADY THERE, and the spec that said otherwise was wrong. `_contract.py:138`
puts `"episode_id": ep.episode_id` in EVERY item; it is `stable_episode_id(clip_id)`,
63-bit precisely so it survives torch's default int64 collate, and
`test_refc_v3_u8_batches.py`'s pinned key set contains it. The earlier reading ("the batch
carries no identifier") came from grepping for keys the TRAINER READS rather than keys the
DATASET EMITS — the trainer simply never read it. ⇒ the fix needed no dataset change and
no batch-contract change, hence no parity risk.

This file pins the FORMAT and the estimator round-trip. The live wiring is proven by a real
run, banked under `raw/` in the W-BOOTSTRAP package.
"""
from __future__ import annotations

import json

import pytest

from taniteval.ci import paired_episode_cluster_bootstrap


def _rows(n_ep: int = 5, per_ep: int = 4) -> list[dict]:
    """A dump in the shape the trainer writes: one row per WINDOW, each carrying its
    `episode_id` so windows can be CLUSTERED by episode."""
    out = []
    for e in range(n_ep):
        for w in range(per_ep):
            out.append({"step": 1, "episode_id": 10_000_000_000 + e,
                        "loss": 100.0 + e + 0.1 * w, "traj": 5.0 + 0.01 * w})
    return out


def test_a_dump_row_carries_an_id_and_scalars():
    """⛔ Both halves are required. Scalars alone cannot be clustered; an id alone has
    nothing to bootstrap."""
    r = _rows()[0]
    assert isinstance(r["episode_id"], int)
    assert set(r) >= {"step", "episode_id", "loss", "traj"}
    json.dumps(r)                       # it must survive the JSONL round trip


def test_the_bootstrap_ACCEPTS_the_dump_shape():
    """The reachability claim, as a test: rows -> (a, b, eid) -> a finite interval."""
    rows = _rows()
    a = [r["traj"] for r in rows]
    eid = [r["episode_id"] for r in rows]
    b = [x * 1.10 for x in a]
    out = paired_episode_cluster_bootstrap(a, b, eid, n_boot=500, seed=0)
    assert out["n_windows"] == len(rows)
    assert out["n_episodes"] == len({*eid})
    assert out["lo"] <= out["delta"] <= out["hi"]


def test_MUTATION_an_arm_against_ITSELF_must_contain_zero():
    """⛔ THE DISCRIMINATING CONTROL, and the reason this file is not ceremony.

    A harness that reports a run as "separated" from ITSELF is measuring its own bug — and
    that is precisely the failure `H-ESTIM-SEED-1` is about, where two runs differing in
    NOTHING cleared the same estimator on 6 of 42 cells. If this ever goes red, no interval
    from this path may be quoted.
    """
    rows = _rows()
    a = [r["traj"] for r in rows]
    eid = [r["episode_id"] for r in rows]
    out = paired_episode_cluster_bootstrap(a, a, eid, n_boot=500, seed=0)
    assert out["lo"] <= 0.0 <= out["hi"], out
    assert out["separated"] is False, out


def test_clustering_is_BY_EPISODE_not_by_window():
    """⚠️ The estimator's whole question is "would another draw of EPISODES say this?".
    Feeding it a distinct id per window would silently turn it into a window bootstrap —
    narrower, and answering something nobody asked."""
    rows = _rows(n_ep=5, per_ep=4)
    a = [r["traj"] for r in rows]
    by_ep = paired_episode_cluster_bootstrap(
        a, [x * 1.1 for x in a], [r["episode_id"] for r in rows], n_boot=500, seed=0)
    by_win = paired_episode_cluster_bootstrap(
        a, [x * 1.1 for x in a], list(range(len(rows))), n_boot=500, seed=0)
    assert by_ep["n_episodes"] == 5
    assert by_win["n_episodes"] == len(rows)      # the wrong thing, shown to differ


@pytest.mark.parametrize("field", ["n_windows", "n_episodes"])
def test_n_is_always_STATED(field):
    """⛔ A read over fewer windows than a gate's floor must be reported as underpowered,
    never quoted bare. The estimator carries both counts; nothing downstream may drop them."""
    rows = _rows()
    out = paired_episode_cluster_bootstrap(
        [r["traj"] for r in rows], [r["loss"] for r in rows],
        [r["episode_id"] for r in rows], n_boot=200, seed=0)
    assert field in out and out[field] > 0


def test_the_trainer_EXPOSES_the_flag_and_it_is_OFF_by_default():
    """⛔ PARITY. The dump is a SEPARATE opt-in batch-1 pass; with the flag unset the
    in-training monitor and every banked comparison are untouched."""
    import argparse
    import pathlib
    import re
    src = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "refc_v3_train.py"
    text = src.read_text(encoding="utf-8", errors="replace")
    m = re.search(r'ap\.add_argument\("--eval-window-dump",\s*default=(\w+)', text)
    assert m, "--eval-window-dump is not declared in the trainer"
    assert m.group(1) == "None", f"the dump must default to OFF, got {m.group(1)}"
    assert isinstance(argparse.ArgumentParser, type)      # import-sanity, not a claim
