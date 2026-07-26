#!/usr/bin/env python3
"""Is the frozen-WM-planner family SUSPECT, or just UNSTAMPED?

`2026-07-23-frozen-wm-learned-planner/artifacts/*.json` publish `paired` blocks
with intervals and `separated` verdicts but name **no estimator anywhere in the
JSON**. Under the program rule *"never quote an interval without its
estimator"* that is inadmissible — but it is NOT the same defect as
`overlapping_holdout_se`, and the difference decides whether these numbers get
retracted or merely re-stamped.

Reading `run.py:104-134` / `run40.py:60-81`, the scripts vendored their own
`ep_bootstrap` / `paired_boot`: resample EPISODES with replacement, recompute
the mean on the resampled windows, take the 2.5/97.5 percentiles, paired form
takes both arms on the same resampled episodes. That is the episode-cluster
bootstrap. But *reading* code is INHERITED evidence, and per the operating
standard a claim of this weight has to be MEASURED.

So: the per-window dumps (`perwin_*.pt`) are committed. This re-derives the
published intervals from those dumps using `taniteval.ci` — the canonical
implementation — and checks whether the numbers agree. Agreement converts
"looks like the right estimator" into "IS the right estimator, measured", and
the family becomes a labelling fix rather than a recompute.

Run:  OMP_NUM_THREADS=8 PYTHONPATH=/root/taniteval python3 verify_frozenwm_estimator.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/root/taniteval")

from taniteval import ci as _ci                      # noqa: E402

D = Path("/root/cl_rerun_20260726/frozenwm")
OUT = Path("/root/cl_rerun_20260726")
K = 4          # 4 waypoint knots


def ade(wp, gt):
    return torch.linalg.norm(wp - gt, dim=-1).mean(dim=1)


def main():
    d = torch.load(D / "perwin_40ep.pt", weights_only=False)
    pub = json.loads((D / "results_40ep.json").read_text(encoding="utf-8"))
    print("dump keys:", list(d) if isinstance(d, dict) else type(d).__name__)
    if isinstance(d, dict):
        for k, v in d.items():
            print("   ", k, getattr(v, "shape", None), getattr(v, "dtype", None))
    print("\npublished:", json.dumps(pub, indent=1)[:1500])

    rep = {"_what": "re-derive the frozen-WM-planner intervals from the "
                    "committed per-window dumps using taniteval.ci, to decide "
                    "whether that family is SUSPECT or merely UNSTAMPED",
           "_source_read": "run.py:104-134 / run40.py:60-81 vendor an "
                           "episode-cluster bootstrap (episode resampling, "
                           "percentile CI, paired form on shared episodes)",
           "dump_keys": list(d) if isinstance(d, dict) else str(type(d)),
           "checks": {}}

    # The dumps store per-window ADE arrays keyed by arm, plus `eid`.
    eid = d.get("eid")
    if eid is not None:
        eid = [str(int(x)) for x in np.asarray(eid).reshape(-1)]
        for name, blk in pub.get("paired", {}).items():
            # names look like "W_minus_oracle"
            if "_minus_" not in name:
                continue
            a, b = name.split("_minus_")
            # the dump stores the learned arm's ADE as `W_a` (and its FDE as
            # `W_f`), while the JSON names the arm plain `W`.
            alias = {"W": "W_a"}

            def _key(nm):
                return next((k for k in d if k.lower() in
                             (nm.lower(), alias.get(nm, "\0").lower())), None)
            ka, kb = _key(a), _key(b)
            if ka is None or kb is None:
                rep["checks"][name] = {"skipped": f"no per-window arrays for "
                                                  f"{a!r}/{b!r} in the dump"}
                continue
            va = np.asarray(d[ka], dtype=float).reshape(-1)
            vb = np.asarray(d[kb], dtype=float).reshape(-1)
            got = _ci.paired_episode_cluster_bootstrap(va, vb, eid,
                                                       n_boot=2000, seed=0)
            pd_ = blk["delta"] if isinstance(blk, dict) else blk[0]
            pci = blk["ci"] if isinstance(blk, dict) else blk[1:3]
            rep["checks"][name] = {
                "published_delta": pd_, "published_ci": list(pci),
                "recomputed_delta": got["delta"],
                "recomputed_ci": [got["lo"], got["hi"]],
                "recomputed_estimator": got["estimator"],
                "delta_matches": bool(abs(pd_ - got["delta"]) < 1e-3),
                "ci_close": bool(abs(pci[0] - got["lo"]) < 0.05
                                 and abs(pci[1] - got["hi"]) < 0.05)}
    (OUT / "frozenwm_estimator_verification.json").write_text(
        json.dumps(rep, indent=2, default=str), encoding="utf-8")
    print("\n" + json.dumps(rep["checks"], indent=2))


if __name__ == "__main__":
    main()
