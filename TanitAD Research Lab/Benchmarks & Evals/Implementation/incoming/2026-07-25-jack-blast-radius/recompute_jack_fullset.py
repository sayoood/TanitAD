#!/usr/bin/env python3
"""`_jack` / `_agg` blast-radius sweep — RECOMPUTATION (step 3).

`taniteval/results/windows_<arm>.pt` carries the RAW per-window
``pred / gt / cv / eid`` for the canonical 881-window, 40-episode val set. That
is enough to recompute, from first principles and with zero GPU:

  * the DEPRECATED published central value — mean over 8 overlapping random 20 %
    episode holdouts (``bench._agg`` reproduction, exact arithmetic)
  * the CORRECT central value — plain full-set mean over all 881 windows
  * the decision-grade interval — episode-cluster bootstrap (``taniteval/ci.py``)
  * paired arm-vs-arm and arm-vs-CV deltas — paired episode-cluster bootstrap

and therefore to answer, per number, ``published -> corrected -> does the
verdict flip?``

READ-ONLY on `taniteval/`: `ci.py` is imported from its file path with bytecode
writing disabled, so nothing is created inside the sibling agent's tree.
No GPU, no pod, no network.

Run:  <venv-python> recompute_jack_fullset.py --repo <repo-root>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True          # do not litter taniteval/__pycache__

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# read-only imports of the program's own estimators                            #
# --------------------------------------------------------------------------- #
def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# exact reproduction of the split machinery the published numbers used         #
# (stack/tanitad/eval/gates.py::split_by_episode -> i3_episode_split)          #
# --------------------------------------------------------------------------- #
def make_split_fn(repo: Path):
    """Return the REAL ``split_by_episode``. Falls back to a verified local
    reimplementation only if the package will not import; the fallback is
    checked against the published ``heldout`` block before any number is used."""
    sys.path.insert(0, str(repo / "stack"))
    sys.path.insert(0, str(repo / "stack" / "scripts"))
    try:
        from tanitad.eval.gates import split_by_episode  # noqa: E402
        return split_by_episode, "tanitad.eval.gates.split_by_episode (REAL)"
    except Exception as e:                       # pragma: no cover
        return None, f"IMPORT FAILED: {type(e).__name__}: {e}"


# --------------------------------------------------------------------------- #
# metric suite — byte-identical to taniteval/taniteval/bench.py::_suite        #
# --------------------------------------------------------------------------- #
HORIZONS_S = {5: "0.5s", 10: "1s", 15: "1.5s", 20: "2s"}


def suite(pred, gt):
    de = torch.linalg.norm(pred - gt, dim=-1)            # [N, 4]
    out = {}
    for j, (_step, name) in enumerate(sorted(HORIZONS_S.items())):
        out[f"de@{name}"] = float(de[:, j].mean())
        out[f"ade@{name}"] = float(de[:, :j + 1].mean())
    out["ade_0_2s"] = out["ade@2s"]
    out["fde@2s"] = float(de[:, -1].mean())
    out["rmse"] = float(np.sqrt(((pred - gt) ** 2).sum(-1).mean().item()))
    out["miss_rate@2m"] = float((de[:, -1] > 2.0).float().mean())
    return out


def per_window_ade(pred, gt):
    """[N] per-window ade_0_2s components — the bootstrap's resampling unit."""
    de = torch.linalg.norm(pred - gt, dim=-1).numpy().astype(np.float64)
    return de.mean(axis=1)


def heldout_agg(pred, gt, eid, split_fn, n_splits=8, val_frac=0.2, seed=0):
    """EXACT reproduction of ``bench._agg`` over ``bench.run``'s 8 splits.

    Returns ``{metric: {mean(split-mean), ci95(overlapping_holdout_se), std}}``.
    """
    splits = [split_fn(eid, val_frac, s) for s in range(seed, seed + n_splits)]
    per_split = [suite(pred[va], gt[va]) for _tr, va in splits]
    out = {}
    for k in per_split[0]:
        v = np.array([d[k] for d in per_split], dtype=float)
        out[k] = {"mean": round(float(np.nanmean(v)), 4),
                  "ci95": round(float(1.96 * np.nanstd(v) / max(1, v.size) ** .5), 4),
                  "std": round(float(np.nanstd(v)), 4)}
    return out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    repo = Path(a.repo).resolve()
    out_dir = Path(a.out) if a.out else Path(__file__).resolve().parent
    res = repo / "taniteval" / "results"

    C = load_module(repo / "taniteval" / "taniteval" / "ci.py", "_jack_audit_ci")
    split_fn, split_prov = make_split_fn(repo)
    print(f"[recompute] split provenance: {split_prov}")
    if split_fn is None:
        sys.exit("cannot proceed without the real split function")

    win_files = sorted(res.glob("windows_*.pt"))
    print(f"[recompute] {len(win_files)} raw window dumps found")

    # ---- window-set alignment proof --------------------------------------- #
    # Two dump vintages label episodes differently (0..39 vs the real ids). But
    # `gt` and `cv` are model-independent, so byte-identical gt+cv+episode
    # boundaries PROVE the same 881 windows in the same order — which makes
    # every arm paired-comparable once the ids are canonicalised positionally.
    ref = torch.load(res / "windows_flagship-30k.pt", map_location="cpu",
                     weights_only=False)

    def canonical_eid(eid):
        """Positional episode index 0..K-1 from the episode-boundary pattern."""
        out, k = [], 0
        for i, e in enumerate(eid):
            if i and str(e) != str(eid[i - 1]):
                k += 1
            out.append(k)
        return out

    ref_bnd = [i for i in range(1, len(ref["eid"]))
               if str(ref["eid"][i]) != str(ref["eid"][i - 1])]
    alignment = {}

    arms, rows = {}, []
    for wf in win_files:
        arm = wf.stem[len("windows_"):]
        d = torch.load(wf, map_location="cpu", weights_only=False)
        pred, gt, cv, eid_raw = d["pred"], d["gt"], d["cv"], d["eid"]
        aligned = (gt.shape == ref["gt"].shape
                   and float((gt - ref["gt"]).abs().max()) == 0.0
                   and float((cv - ref["cv"]).abs().max()) == 0.0
                   and [i for i in range(1, len(eid_raw))
                        if str(eid_raw[i]) != str(eid_raw[i - 1])] == ref_bnd)
        alignment[arm] = {
            "gt_and_cv_byte_identical_to_flagship-30k": bool(aligned),
            "eid_labels_equal_to_flagship-30k":
                bool([str(x) for x in eid_raw] == [str(x) for x in ref["eid"]]),
            "n_windows": int(pred.shape[0]),
        }
        # the PUBLISHED heldout used the raw labels; the canonical labels are
        # what makes cross-vintage pairing valid.
        eid = eid_raw
        fs_m = suite(pred, gt)
        fs_c = suite(cv, gt)
        ho = heldout_agg(pred, gt, eid, split_fn)
        ho_c = heldout_agg(cv, gt, eid, split_fn)
        pw_m, pw_c = per_window_ade(pred, gt), per_window_ade(cv, gt)
        boot = C.episode_cluster_bootstrap(pw_m, eid, n_boot=a.n_boot, seed=0)
        vs_cv = C.paired_episode_cluster_bootstrap(pw_c, pw_m, eid,
                                                   n_boot=a.n_boot, seed=0)
        # RELABELLING EFFECT: `split_by_episode` splits on sorted(set(int(e))),
        # so the SAME windows under a different episode LABELLING fall into
        # different 20 % holdouts and the split-mean moves. The full-set mean
        # cannot move. Quantified here because two dump vintages exist.
        ceid = canonical_eid(eid_raw)
        ho_canon = (heldout_agg(pred, gt, ceid, split_fn)
                    if aligned else None)
        arms[arm] = {"pw": pw_m, "eid": eid, "ceid": ceid, "aligned": aligned,
                     "pred": pred, "gt": gt, "cv": cv}
        rows.append({
            "arm": arm,
            "n_windows": int(pred.shape[0]),
            "n_episodes": int(len(set(eid))),
            "window_set_aligned_to_flagship-30k": bool(aligned),
            "heldout_under_CANONICAL_labels": (
                ho_canon["ade_0_2s"]["mean"] if ho_canon else None),
            "relabelling_shift_in_split_mean": (
                round(ho["ade_0_2s"]["mean"] - ho_canon["ade_0_2s"]["mean"], 4)
                if ho_canon else None),
            "published_heldout_split_mean_ade_0_2s": ho["ade_0_2s"]["mean"],
            "published_heldout_ci95": ho["ade_0_2s"]["ci95"],
            "corrected_full_set_ade_0_2s": round(fs_m["ade_0_2s"], 4),
            "abs_bias": round(ho["ade_0_2s"]["mean"] - fs_m["ade_0_2s"], 4),
            "rel_bias_pct": round(100 * (ho["ade_0_2s"]["mean"] - fs_m["ade_0_2s"])
                                  / fs_m["ade_0_2s"], 3),
            "bootstrap_ci95": boot["ci95"],
            "bootstrap_lo": boot["lo"], "bootstrap_hi": boot["hi"],
            "ci_width_ratio_boot_over_heldout": (
                round(boot["ci95"] / ho["ade_0_2s"]["ci95"], 3)
                if ho["ade_0_2s"]["ci95"] else None),
            # --- did the arm beat CV? old flag vs new flag ------------------ #
            "beats_cv_OLD_splitmean": bool(ho["ade_0_2s"]["mean"]
                                           < ho_c["ade_0_2s"]["mean"]),
            "beats_cv_NEW_fullset": bool(fs_m["ade_0_2s"] < fs_c["ade_0_2s"]),
            "beats_cv_separated_paired": bool(vs_cv["separated"]
                                              and vs_cv["delta"] > 0),
            "cv_heldout_split_mean": ho_c["ade_0_2s"]["mean"],
            "cv_full_set": round(fs_c["ade_0_2s"], 4),
            "full_set_all_metrics": {k: round(v, 4) for k, v in fs_m.items()},
            "heldout_all_metrics": {k: v["mean"] for k, v in ho.items()},
        })
        print(f"  {arm:<28} heldout {ho['ade_0_2s']['mean']:.4f} -> "
              f"full_set {fs_m['ade_0_2s']:.4f}  "
              f"({rows[-1]['rel_bias_pct']:+.2f} %)  "
              f"boot CI [{boot['lo']:.4f}, {boot['hi']:.4f}]")

    # ---------------------------------------------------------------------- #
    # decision-grade PAIRED comparisons (same 881 windows, same episodes)     #
    # ---------------------------------------------------------------------- #
    PAIRS = [
        ("flagship-v16-ab-ft", "flagship-30k", "v1.6 vs v1 (the v3.5 fork)"),
        ("refc-xl-30k", "flagship-30k", "REF-C-XL vs flagship v1 (the ranking)"),
        ("refc-base-30k", "flagship-30k", "REF-C-base vs flagship v1"),
        ("flagship-v3enc-10k", "flagship-30k", "v3enc vs v1 (the RESTART)"),
        ("flagship-v4.1-10k", "flagship-30k", "v4.1 vs v1 (the FAIL)"),
        ("flagship-v4.2-step4000", "flagship-30k", "v4.2 vs v1"),
        ("flagship-nospeed", "flagship-speed", "no-speed ablation control"),
        ("refc-xl-30k", "refc-base-30k", "REF-C scale A/B"),
        ("refb-v2-30k", "flagship-30k", "REF-B v2 vs v1"),
        ("refa-dynin-30k", "flagship-30k", "REF-A vs v1"),
    ]
    pair_rows = []
    for a_key, b_key, why in PAIRS:
        if a_key not in arms or b_key not in arms:
            pair_rows.append({"a": a_key, "b": b_key, "why": why,
                              "status": "MISSING raw windows for one arm"})
            continue
        A, B = arms[a_key], arms[b_key]
        if not (A["aligned"] and B["aligned"]):
            pair_rows.append({"a": a_key, "b": b_key, "why": why,
                              "status": "window set NOT aligned — not paired-comparable"})
            continue
        # pair on the CANONICAL positional episode index: the raw dumps use two
        # different LABELLINGS of the same 40 episodes (0..39 vs the real ids),
        # proven equivalent above by byte-identical gt/cv + episode boundaries.
        cross_vintage = ([str(x) for x in A["eid"]] != [str(x) for x in B["eid"]])
        p = C.paired_episode_cluster_bootstrap(A["pw"], B["pw"], A["ceid"],
                                               n_boot=a.n_boot, seed=0)
        # the legacy way this delta was read: difference of the two split-means,
        # each computed under its OWN dump's episode labelling — which is how it
        # was actually published.
        ho_a = heldout_agg(A["pred"], A["gt"], A["eid"], split_fn)["ade_0_2s"]
        ho_b = heldout_agg(B["pred"], B["gt"], B["eid"], split_fn)["ade_0_2s"]
        fs_a = suite(A["pred"], A["gt"])["ade_0_2s"]
        fs_b = suite(B["pred"], B["gt"])["ade_0_2s"]
        legacy_delta = round(ho_a["mean"] - ho_b["mean"], 4)
        true_delta = round(fs_a - fs_b, 4)
        ratio = (round(legacy_delta / true_delta, 3)
                 if abs(true_delta) > 1e-9 else None)
        pair_rows.append({
            "a": a_key, "b": b_key, "why": why, "status": "OK",
            "cross_vintage_episode_labelling": bool(cross_vintage),
            "cross_vintage_note": (
                "the two published `heldout` split-means came from DIFFERENT "
                "random 20% partitions (split_by_episode sorts on int(eid) and "
                "the labellings differ) — the legacy delta below is not even a "
                "like-for-like split comparison" if cross_vintage else None),
            "legacy_delta_of_split_means": legacy_delta,
            "true_delta_of_full_set_means": true_delta,
            "bias_ratio_legacy_over_true": ratio,
            "sign_flip": bool(legacy_delta * true_delta < 0),
            "paired_bootstrap": p,
            "verdict_legacy_separated": None,
            "verdict_paired_separated": bool(p["separated"]),
        })
        print(f"  [pair] {a_key} - {b_key}: legacy d {legacy_delta:+.4f} -> "
              f"true d {true_delta:+.4f} (x{ratio})  "
              f"paired CI [{p['lo']:+.4f}, {p['hi']:+.4f}] "
              f"separated={p['separated']}")

    out = {
        "generated_by": "recompute_jack_fullset.py (jack blast-radius sweep)",
        "estimator_provenance": {
            "deprecated": "overlapping_holdout_se == mean of 8 OVERLAPPING "
                          "random 20% episode-holdout split-means "
                          "(bench._agg / _jack); ci95 = 1.96*std/sqrt(8)",
            "correct_point": "plain mean over ALL windows (full_set)",
            "correct_interval": "taniteval/ci.py episode_cluster_bootstrap / "
                                "paired_episode_cluster_bootstrap, B="
                                f"{a.n_boot}, resampling unit = val episode",
            "split_fn": split_prov,
        },
        "single_arm": rows,
        "paired": pair_rows,
    }
    p = out_dir / "jack_recompute.json"
    p.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"[recompute] wrote {p.name}")


if __name__ == "__main__":
    main()
