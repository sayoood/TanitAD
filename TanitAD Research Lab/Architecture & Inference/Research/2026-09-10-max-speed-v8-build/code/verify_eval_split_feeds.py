"""⛔ THE EVAL SPLIT'S OWN `enable_max_speed`, RUN ON THE REAL v8 EVAL BLOB.

⭐ WHY THIS EXISTS AS A SEPARATE PROBE. The 20-step smoke uses `--synth-episodes`
and therefore has **no `--eval-cache`**, so `eval_max_speed_stats` is never
computed and `e_ds.enable_max_speed(...)` is **never called**. The smoke is green
and the eval path is untested — which is exactly the shape of defect this
programme keeps paying for: *a thing that is built and stamped and never actually
exercised.*

⛔ AND THE EVAL SPLIT IS THE HALF THAT DECIDES WHETHER THE ARM IS EVALUABLE AT
ALL. The previous supplier died precisely here: 31/4,572 train but **0/147 eval**,
so the ON and OFF arms would receive identical input on every eval window and the
lever would be unmeasurable. A train-only check cannot see that.

⇒ This drives the trainer's REAL `V3Dataset.enable_max_speed` — unbound, over a
`SimpleNamespace` self, the rig `stack/tests/test_max_speed_wiring.py` already
uses — against the REAL v8 eval blob loaded through the REAL `load_v7_labels`, and
asserts the census it produces. Nothing is stubbed but the frame store.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "stack"))

from tanitad.data import v7_labels as v7l                     # noqa: E402
from tanitad.data.v2_dataset import stable_episode_id         # noqa: E402
from tanitad.refs import max_speed_input as msi               # noqa: E402


def _trainer():
    p = REPO / "stack" / "scripts" / "refc_v3_train.py"
    spec = importlib.util.spec_from_file_location("refc_v3_train_for_eval", str(p))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["refc_v3_train_for_eval"] = mod
    spec.loader.exec_module(mod)
    return mod


def probe(blob: Path, split: str) -> dict:
    tr = _trainer()
    labels, manifest = v7l.load_v7_labels(blob, allow_oracle_nav=True)

    # the dataset surface `enable_max_speed` actually touches -- and ONLY that,
    # so a change that starts touching more fails loudly here.
    ds = types.SimpleNamespace()
    ds.v7_by_sid = {stable_episode_id(l.clip_id): l for l in labels}
    ds.episodes = [types.SimpleNamespace(episode_id=sid) for sid in ds.v7_by_sid]
    # one window per episode, the same shape the real index has
    ds.index = [(i, 0) for i in range(len(ds.episodes))]

    rep = tr.V3Dataset.enable_max_speed(ds, manifest, msi.DEFAULT_MODE)

    fed = [v for v in ds._max_speed_by_sid.values() if v[1] > 0.5]
    vals = sorted(v[0] for v in fed)
    checks = {
        "n_records_matches_blob": rep["n_clips"] == len(labels),
        # ⛔ THE ONE THAT DECIDES EVALUABILITY.
        "every_window_receives_a_ceiling": (
            rep["n_windows_with_ceiling"] == rep["n_windows"] > 0),
        "window_ceiling_frac_is_1": rep["window_ceiling_frac"] == 1.0,
        "every_clip_carries_a_block": rep["n_clips_with_block"] == len(labels),
        "units_declared_m_s": rep["control_units"] == "m_s",
        "provenance_is_ego_future": rep["provenance"].startswith("ego-future"),
        "ladder_is_the_pinned_one": rep["steps_kmh"] == list(
            msi.POSTED_LIMIT_STEPS_KMH),
        "label_md5_recorded": len(rep["label_md5"]) == 32,
        "oracle_stamp_recorded": rep["allow_oracle_nav"] is True,
        # ⛔ the values shipped are RAW m/s, not pre-quantized -- if the loader
        # ever starts quantizing, the ladder would be applied twice.
        "values_are_RAW_not_on_the_ladder": any(
            v not in msi.POSTED_LIMIT_STEPS_MS for v in vals),
        # ⭐ SAME-BREATH VACUITY CONTROL: a probe over an empty index would
        # satisfy several checks above and mean nothing.
        "CONTROL_more_than_100_windows": rep["n_windows"] > 100,
    }
    return {"split": split, "blob": str(blob), "label_md5": rep["label_md5"],
            "n_clips": rep["n_clips"], "n_windows": rep["n_windows"],
            "n_windows_with_ceiling": rep["n_windows_with_ceiling"],
            "window_ceiling_frac": rep["window_ceiling_frac"],
            "n_clips_over_ceiling": rep["n_clips_over_ceiling"],
            "v_min_ms": vals[0] if vals else None,
            "v_max_ms": vals[-1] if vals else None,
            "checks": checks, "PASS": all(checks.values())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eval-blob", required=True)
    ap.add_argument("--train-blob", required=True)
    ap.add_argument("--report")
    a = ap.parse_args(argv)
    rep = {"instrument": "verify_eval_split_feeds.py",
           "why": ("the 20-step smoke has no --eval-cache, so the eval split's "
                   "enable_max_speed is never called by it"),
           "splits": {"eval": probe(Path(a.eval_blob), "eval"),
                      "train": probe(Path(a.train_blob), "train")}}
    rep["OVERALL_PASS"] = all(s["PASS"] for s in rep["splits"].values())
    txt = json.dumps(rep, indent=1)
    print(txt)
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(txt, encoding="utf-8")
    for k, v in rep["splits"].items():
        bad = [c for c, ok in v["checks"].items() if not ok]
        print(f"{k:6s} {'PASS' if v['PASS'] else 'FAIL'}"
              + (f"  failed: {bad}" if bad else ""))
    print(f"\nOVERALL: {'PASS' if rep['OVERALL_PASS'] else 'FAIL'}")
    return 0 if rep["OVERALL_PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
