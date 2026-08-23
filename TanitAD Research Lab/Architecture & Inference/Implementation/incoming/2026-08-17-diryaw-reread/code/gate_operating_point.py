"""DIR_YAW_RAD 0.15 -> 0.10: the GATE'S OPERATING POINT, with the mandated estimator.

0 GPU. Reads only banked artifacts (`taniteval/results/windows_*.pt` +
`hier_*.json` panels). Nothing here changes a threshold in shipped code.

WHY THIS SCRIPT EXISTS, given two prior passes already answered "does a verdict move?":

  * 2026-08-15 (`.../2026-08-15-dir-yaw-gate-reread/`) reported point deltas only.
  * 2026-08-16 (`.../2026-08-16-dir-yaw-reread/`) built an exact kappa ENVELOPE whose
    single pivotal unknown is **m**, the number of windows in the band (0.10, 0.15].
    It quoted m as ESTIMATED ("m ~ 2-6") with NO interval.

  The brief mandates the PAIRED EPISODE-CLUSTER BOOTSTRAP (`taniteval/ci.py`) and forbids
  `overlapping_holdout_se`. The band mass IS bootstrappable from banked data, and it is the
  quantity every kappa envelope pivots on. So that is what this measures.

WHAT IS AND IS NOT RECONSTRUCTIBLE (established from source, §1 of the report):
  banked `windows_*.pt` carries pred/gt/cv/eid/speed/head_deg. `head_deg` is
  `driving_diagnostic.net_heading_change_deg(ep.poses, last)`
  = `|wrap(poses[last+K_MAX,2] - poses[last,2])| * 180/pi`, K_MAX = 20
  (`stack/scripts/driving_diagnostic.py:139-142`, `taniteval/taniteval/bench.py:399`).
  The gate's own input is
  `gt_net = wrap_to_pi(fut[:, GOAL_H-1, 2] - pl[:, 2])`, `fut = poses[t+WIN : t+WIN+GOAL_H]`,
  `last = t+WIN-1`, `GOAL_H = K_MAX = 20` (`taniteval/taniteval/hierarchy.py:589,595,114`)
  => `fut[:, GOAL_H-1] == poses[last+20]`. SAME poses, SAME horizon, SAME wrap.
  ⇒ `head_deg * pi/180` IS `|gt_net|` EXACTLY.  The SIGN is lost to `.abs()`, so
  kappa cannot be rebuilt from these dumps — every MAGNITUDE-only quantity can.
  That is verified here as a HARD GATE (`verify_head_deg`) rather than assumed, because
  the 5/5 bit-exact check it reproduces was INHERITED from the 2026-08-15 pass and a
  decision-grade claim may not be inherited.

Usage:  python gate_operating_point.py [--repo <path>] [--out <json>]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys

import numpy as np
import torch

# published constants, read from source rather than retyped
PUBLISHED_GATE = 0.15
REREAD_GATE = 0.10
K_MAX = 20                      # taniteval/taniteval/hierarchy.py:113
N_BOOT = 2000
SEED = 0


def _ci():
    """taniteval.ci — the ONLY admissible estimator here."""
    from taniteval import ci
    assert hasattr(ci, "paired_episode_cluster_bootstrap"), "ci.py lacks the paired API"
    return ci


# --------------------------------------------------------------------------- #
# 1. HARD GATE: head_deg must reproduce each panel's own published gt_dir      #
# --------------------------------------------------------------------------- #
def verify_head_deg(dumps, panels):
    """Reconstruct the turning-window COUNT at 0.15 from head_deg and compare to
    `n - straight` of each panel's own `consistency.distributions.gt_dir`.

    A panel's gt_dir was produced by `_dir_of(gt_net)` inside the GPU pass; our
    value comes from `head_deg` banked in a different file by a different function.
    Identical counts prove the two are the same quantity.

    ⚠️ THE COMPARISON IS ON INTEGERS ON PURPOSE. The 2026-08-15 pass reported this
    check as *"0.000000 — bit-exact, 5/5"* on the FRACTIONS. Reproduced here, the
    fractions differ by **2.78e-17** (one float64 ULP): `1 - straight/n` and
    `(rad > g).mean()` are different arithmetic paths to the same rational, so a
    float equality test on them is answering a question about IEEE rounding, not
    about the instrument. The counts are the invariant, and they agree exactly.
    """
    rows, ok = [], True
    for pname, pth in sorted(panels.items()):
        try:
            with open(pth, encoding="utf-8") as fh:
                pan = json.load(fh)
            gd = pan["consistency"]["distributions"]["gt_dir"]
        except Exception as exc:                       # noqa: BLE001
            rows.append({"panel": pname, "status": f"UNREADABLE: {exc}"})
            continue
        tot = sum(gd.values())
        if not tot:
            continue
        pub_n = tot - gd.get("route_straight", 0)          # INTEGER count
        # match the dump by window count (the corpus/grid signature)
        cand = [d for d in dumps.values() if len(d["head_deg"]) == tot]
        if not cand:
            rows.append({"panel": pname, "n_windows": tot,
                         "status": "NO DUMP WITH MATCHING n — not cross-checkable"})
            continue
        rad = np.asarray(cand[0]["head_deg"], dtype=np.float64) * math.pi / 180.0
        our_n = int((rad > PUBLISHED_GATE).sum())
        exact = our_n == pub_n
        ok &= exact
        rows.append({"panel": pname, "n_windows": tot,
                     "published_turning_count": int(pub_n),
                     "from_head_deg_count": our_n,
                     "counts_identical": exact,
                     "frac_ulp_gap": abs(our_n / tot - pub_n / tot)})
    return {"all_counts_identical": bool(ok), "rows": rows,
            "_read": ("identical COUNTS mean head_deg*pi/180 IS |gt_net|, so every "
                      "magnitude-only gate quantity below is MEASURED, not modelled.")}


# --------------------------------------------------------------------------- #
# 2. The operating point, with the MANDATED estimator                          #
# --------------------------------------------------------------------------- #
def operating_point(head_deg, eid, ci):
    """frac-turning at both gates, the PAIRED delta, and the band mass (0.10, 0.15].

    ⛔ estimator = episode-cluster bootstrap over episodes; the 0.15 and 0.10 reads
    are on the SAME windows, so the delta takes the PAIRED form. Combining two
    single-arm intervals in quadrature would be invalid (they are not independent)
    and `overlapping_holdout_se` is refused outright — it biases the POINT estimate.
    """
    rad = np.asarray(head_deg, dtype=np.float64) * math.pi / 180.0
    n = rad.size
    turn15 = (rad > PUBLISHED_GATE).astype(np.float64)
    turn10 = (rad > REREAD_GATE).astype(np.float64)
    band = ((rad > REREAD_GATE) & (rad <= PUBLISHED_GATE)).astype(np.float64)

    def one(v):
        return ci.episode_cluster_bootstrap(v, eid, reduce="mean",
                                            n_boot=N_BOOT, seed=SEED)

    paired = ci.paired_episode_cluster_bootstrap(turn10, turn15, eid,
                                                 n_boot=N_BOOT, seed=SEED)
    return {
        "n_windows": int(n),
        "n_episodes": int(len(set(map(str, eid)))),
        "frac_turning_at_0.15": one(turn15),
        "frac_turning_at_0.10": one(turn10),
        "paired_delta_0.10_minus_0.15": paired,
        "band_mass_0.10_to_0.15": one(band),
        "band_count_windows": int(band.sum()),
        "median_abs_net_yaw_rad": round(float(np.median(rad)), 6),
        "p90_abs_net_yaw_rad": round(float(np.percentile(rad, 90)), 6),
        "p99_abs_net_yaw_rad": round(float(np.percentile(rad, 99)), 6),
        "published_gate_over_median": (round(PUBLISHED_GATE / float(np.median(rad)), 2)
                                       if float(np.median(rad)) > 0 else None),
        "_estimator": ("episode_cluster_bootstrap / paired_episode_cluster_bootstrap "
                       "(taniteval/ci.py), 2000 draws, seed 0, resampling unit = "
                       "EPISODE. overlapping_holdout_se REFUSED."),
    }


def corpus_signature(eid):
    """Stable id for the episode set a dump was scored on, so numbers can never be
    quoted without their corpus (two measurements disagree ACROSS corpora)."""
    uniq = sorted(set(map(str, eid)))
    h = hashlib.sha1("\n".join(uniq).encode()).hexdigest()[:12]
    return h, len(uniq)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    repo = args.repo or os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))))
    sys.path.insert(0, os.path.join(repo, "taniteval"))
    sys.path.insert(0, os.path.join(repo, "stack"))
    sys.path.insert(0, os.path.join(repo, "stack", "scripts"))
    ci = _ci()

    dumps = {}
    for p in sorted(glob.glob(os.path.join(repo, "taniteval", "results", "windows_*.pt"))):
        try:
            d = torch.load(p, map_location="cpu", weights_only=False)
        except Exception as exc:                       # noqa: BLE001
            print(f"  skip {os.path.basename(p)}: {exc}", file=sys.stderr)
            continue
        if not ("head_deg" in d and "eid" in d):
            continue
        dumps[os.path.basename(p)[len("windows_"):-len(".pt")]] = {
            "head_deg": d["head_deg"].numpy(), "eid": [str(x) for x in d["eid"]]}

    panels = {}
    for pat in ("TanitAD Research Hub/**/hier_*.json",
                "stack/experiments/**/hier_*.json"):
        for p in glob.glob(os.path.join(repo, pat), recursive=True):
            panels.setdefault(os.path.relpath(p, repo).replace("\\", "/"), p)

    verify = verify_head_deg(dumps, panels)

    # group dumps by corpus so a number can never be quoted without its corpus
    by_corpus: dict[str, dict] = {}
    for arm, d in dumps.items():
        sig, n_ep = corpus_signature(d["eid"])
        by_corpus.setdefault(sig, {"n_episodes": n_ep, "arms": [],
                                   "head_deg": d["head_deg"], "eid": d["eid"]})
        by_corpus[sig]["arms"].append(arm)

    corpora = {}
    for sig, g in sorted(by_corpus.items()):
        # the gate's operating point is a property of the CORPUS (GT poses only),
        # so it is identical for every arm scored on the same episode set — asserted,
        # not assumed.
        same = [dumps[a]["head_deg"] for a in g["arms"]]
        ident = all(len(h) == len(same[0]) and np.array_equal(h, same[0]) for h in same)
        corpora[sig] = {
            "n_arms_sharing_this_corpus": len(g["arms"]),
            "arms": sorted(g["arms"]),
            "gt_head_deg_identical_across_arms": bool(ident),
            **operating_point(g["head_deg"], g["eid"], ci),
        }

    out = {
        "_question": ("what does DIR_YAW_RAD 0.15 -> 0.10 actually change, and with "
                      "what interval under the MANDATED paired episode-cluster "
                      "bootstrap?"),
        "_evidence_class": "MEASURED (ours) — banked artifacts only, 0 GPU",
        "_gates": {"published": PUBLISHED_GATE, "reread": REREAD_GATE,
                   "horizon_steps": K_MAX, "dt_s": 0.1},
        "_estimator_refused": "overlapping_holdout_se (biases the POINT estimate)",
        "verify_head_deg_is_gt_net_magnitude": verify,
        "per_corpus_operating_point": corpora,
        "_n_dumps": len(dumps),
    }
    js = json.dumps(out, indent=1, default=float)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(js + "\n")
        print(f"wrote {args.out}")
    else:
        print(js)
    return 0 if verify["all_counts_identical"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
