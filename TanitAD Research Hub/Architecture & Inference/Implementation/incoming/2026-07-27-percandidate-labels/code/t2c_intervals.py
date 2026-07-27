"""Episode-cluster bootstrap intervals on every T2 headline rate.

Estimator named in advance and never varied: ``taniteval.ci``
episode-cluster bootstrap, B = 2000, resampling unit = the CLIP (the episode
cluster on this corpus). ``overlapping_holdout_se`` is NEVER used — it is not a
jackknife, it biases the point estimate bidirectionally (-6.67 % to +11.69 %
over 27 arms) and it has flipped the sign of a paired delta.

Paired form is used for every contrast, because both arms live on the same
windows and the shared per-window difficulty cancels inside each draw.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO / "taniteval"))
from taniteval import ci as _ci  # noqa: E402

B = 2000


def progress_matrix(D, shape):
    """[W, C] progress, from either dump layout (full or compacted)."""
    if "progress" in D:
        return np.asarray(D["progress"], np.float64)
    p = np.asarray(D["progress_per_candidate"], np.float64)
    return np.broadcast_to(p[None] if p.ndim == 1 else p, shape)


def pdms_lite(D, NCF, TF, CO, shape):
    """PDM's weights, DAC dropped. Recomputed rather than stored."""
    P = progress_matrix(D, shape)
    pmax = np.maximum(P.max(1, keepdims=True), 1e-6)
    EP = np.where(P <= 0.0, 0.0, np.clip(P / pmax, 0.0, 1.0))
    return (~NCF) * (5.0 * EP + 5.0 * (~TF) + 2.0 * CO) / 12.0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    D = torch.load(a.dump, map_location="cpu", weights_only=False)
    eid = list(D["alias"])
    FE = np.asarray(D["fan_err"], np.float64)
    NCF = np.asarray(D["nc_fault"], bool)
    TF = np.asarray(D["ttc_flag"], bool)
    CO = np.asarray(D["comfort_ok"], bool)
    PDMS = pdms_lite(D, NCF, TF, CO, FE.shape)

    def boot(perwin, name):
        r = _ci.episode_cluster_bootstrap(perwin, eid, n_boot=B)
        return dict(name=name, mean=round(float(r["mean"]), 5),
                    lo=round(float(r["lo"]), 5), hi=round(float(r["hi"]), 5),
                    ci95=round(float(r["ci95"]), 5),
                    n_windows=int(r["n_windows"]),
                    n_episodes=int(r["n_episodes"]))

    rates = [
        boot(NCF.mean(1), "NC at-fault rate over all candidates"),
        boot(TF.mean(1), "TTC infraction rate over all candidates"),
        boot((~CO).mean(1), "comfort violation rate over all candidates"),
        boot((NCF.any(1) & ~NCF.all(1)).astype(float),
             "frac windows where the NC label VARIES across the fan"),
        boot((TF.any(1) & ~TF.all(1)).astype(float),
             "frac windows where the TTC label VARIES across the fan"),
    ]

    rng = np.random.default_rng(0)
    W, C = FE.shape
    picks = dict(rule_pdms_lite=PDMS.argmax(1),
                 random=rng.integers(0, C, size=W),
                 oracle_in_fan=FE.argmin(1))
    arms = {k: dict(
        ade=boot(np.take_along_axis(FE, p[:, None], 1)[:, 0], f"{k} ade_0_2s"),
        pdms=boot(np.take_along_axis(PDMS, p[:, None], 1)[:, 0], f"{k} PDMS-lite"),
        nc=boot(np.take_along_axis(NCF, p[:, None], 1)[:, 0].astype(float),
                f"{k} at-fault collision rate"))
        for k, p in picks.items()}

    def paired(k1, k2, mat):
        return _ci.paired_episode_cluster_bootstrap(
            np.take_along_axis(mat, picks[k1][:, None], 1)[:, 0].astype(float),
            np.take_along_axis(mat, picks[k2][:, None], 1)[:, 0].astype(float),
            eid, n_boot=B)

    res = dict(
        estimator="episode-cluster bootstrap (taniteval.ci), B=2000, "
                  "unit = clip; paired form for every contrast; "
                  "overlapping_holdout_se NEVER used",
        rates=rates, arms=arms,
        paired_contrasts=dict(
            collision_rate_rule_minus_random=paired(
                "rule_pdms_lite", "random", NCF),
            collision_rate_rule_minus_ade_oracle=paired(
                "rule_pdms_lite", "oracle_in_fan", NCF),
            collision_rate_ade_oracle_minus_random=paired(
                "oracle_in_fan", "random", NCF),
            ade_rule_minus_ade_oracle=paired(
                "rule_pdms_lite", "oracle_in_fan", FE)))
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
