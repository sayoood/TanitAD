#!/usr/bin/env python3
"""`_jack` / `_agg` blast-radius sweep — ARTIFACT FINGERPRINTING (step 2).

Walks every committed result JSON in the repo and classifies it by which
interval/central-value construction it carries:

  ``heldout``            the DEPRECATED block — mean over 8 OVERLAPPING random
                         20 % episode holdouts (``bench._agg`` / ``_jack``,
                         estimator ``overlapping_holdout_se``). Its ``mean`` is
                         a **mean-of-split-means**, NOT the full-set mean.
  ``full_set``           plain mean over all windows (the correct central value)
  ``cluster_bootstrap``  the 2026-07-20 replacement (``taniteval/ci.py``)
  ``legacy_*``           post-migration modules that keep the old block under an
                         explicitly legacy key

KEY MEASUREMENT (no re-run needed): any artifact carrying BOTH ``heldout`` and
``full_set`` lets us read the point-estimate bias DIRECTLY —
``heldout.mean - full_set.value`` for the same metric. That is the exact defect
the 07-21 REF-C-XL retraction (class C6) found once and never generalised.

READ-ONLY. Writes only its own inventory next to itself.
Run:  <venv-python> sweep_jack_artifacts.py --repo <repo-root>
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

# --------------------------------------------------------------------------- #
# fingerprints                                                                 #
# --------------------------------------------------------------------------- #
DEPRECATED_NAME = "overlapping_holdout_se"
NEW_NAME = "episode_cluster_bootstrap"

# Keys whose VALUE dict is a deprecated split-mean aggregate block. Matched by
# SUFFIX because emitters prefix them (`grounded_rollout_heldout`,
# `legacy_overlapping_holdout_se`, ...) — a set of exact names missed two
# in-repo gate artifacts on the first pass.
HELDOUT_SUFFIXES = ("heldout", "legacy_overlapping_holdout_se", "jackknife",
                    "heldout_split", "heldout_8split")
FULLSET_SUFFIXES = ("full_set", "fullset", "full_set_mean")
BOOT_KEYS = {"cluster_bootstrap", "bootstrap", "primary", "headline",
             "episode_cluster_bootstrap"}

# metrics we try to align between the heldout and full_set blocks
ALIGN_METRICS = ["ade_0_2s", "ade@2s", "ade@1s", "ade@0.5s", "ade@1.5s",
                 "de@2s", "de@1s", "fde@2s", "rmse", "miss_rate@2m",
                 "tms_openloop"]

VERDICT_KEY_RE = re.compile(
    r"(verdict|separated|beats_|passes?|pass_|fail|gate|decision|significant)",
    re.I)


def walk_items(obj, path=""):
    """Yield ``(json_path, key, value)`` for every dict entry, recursively."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}/{k}"
            yield p, k, v
            yield from walk_items(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk_items(v, f"{path}[{i}]")


def is_agg_block(v):
    """True iff ``v`` looks like a ``_agg``/``_jack`` metric dict.

    Two pre-migration shapes exist and NEITHER carried an estimator label
    (verified against ``git show a91bef8:taniteval/taniteval/{bench,closedloop,
    hierarchy}.py``):
      ``_agg``  -> {mean, ci95, std}                  (bench, closedloop)
      ``_jack`` -> {mean, ci95, n, separated}         (hierarchy, closedloop,
                                                      planner_p2)
    Both are mean-of-split-means. Post-migration nodes are self-labelling.
    """
    if not isinstance(v, dict):
        return False
    if v.get("estimator") == DEPRECATED_NAME or v.get("deprecated") is True:
        return True
    if v.get("estimator") is not None:          # labelled as something else
        return False
    ks = set(v)
    return ({"mean", "ci95", "std"} <= ks          # _agg shape
            or {"mean", "ci95", "n", "separated"} <= ks)   # _jack shape


def _stem(key, suffixes):
    """`grounded_rollout_heldout` -> `grounded_rollout`; `heldout` -> ``."""
    for s in sorted(suffixes, key=len, reverse=True):
        if key.endswith(s):
            return key[: -len(s)].rstrip("_")
    return key


def scalarize(v):
    """A metric entry -> its central value (handles both scalar and dict form)."""
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        for k in ("mean", "value", "delta", "point"):
            if isinstance(v.get(k), (int, float)):
                return float(v[k])
    return None


def fingerprint(path: Path, repo: Path):
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return {"file": str(path), "error": f"{type(e).__name__}: {e}"}
    try:
        doc = json.loads(raw)
    except Exception as e:
        return {"file": str(path), "error": f"unparseable: {type(e).__name__}"}

    rec = {
        "file": str(path.relative_to(repo)).replace("\\", "/"),
        "bytes": path.stat().st_size,
        "has_heldout_block": False,
        "has_fullset_block": False,
        "has_bootstrap_block": False,
        "n_deprecated_metric_dicts": 0,
        "n_bootstrap_metric_dicts": 0,
        "mentions_deprecated_name": DEPRECATED_NAME in raw,
        "mentions_new_name": NEW_NAME in raw,
        "refuses_deprecated": False,
        "heldout_paths": [],
        "fullset_paths": [],
        "verdict_keys": [],
        "n_windows": None,
        "n_episodes": None,
        "arm": None,
    }
    for jp, k, v in walk_items(doc):
        # A `*heldout` key only counts if it actually CONTAINS split-mean metric
        # dicts. Several artifacts use `_heldout` to name a held-out DATA SPLIT
        # (`in_rigA_heldout`) — that is a different concept and was a false
        # positive on the first pass.
        if (k.endswith(HELDOUT_SUFFIXES) and isinstance(v, dict)
                and any(is_agg_block(x) for _p, _k, x in walk_items(v))):
            rec["has_heldout_block"] = True
            rec["heldout_paths"].append(jp)
        if k.endswith(FULLSET_SUFFIXES) and isinstance(v, dict):
            rec["has_fullset_block"] = True
            rec["fullset_paths"].append(jp)
        if k in BOOT_KEYS and isinstance(v, dict):
            rec["has_bootstrap_block"] = True
        if isinstance(v, dict):
            if is_agg_block(v):
                rec["n_deprecated_metric_dicts"] += 1
            if v.get("estimator") in (NEW_NAME, "paired_" + NEW_NAME):
                rec["n_bootstrap_metric_dicts"] += 1
        if k in ("deprecated_and_refused", "refused_estimator") \
                and v == DEPRECATED_NAME:
            rec["refuses_deprecated"] = True
        if k in ("n_windows",) and isinstance(v, int) and rec["n_windows"] is None:
            rec["n_windows"] = v
        if k in ("n_episodes",) and isinstance(v, int) and rec["n_episodes"] is None:
            rec["n_episodes"] = v
        if k in ("arm", "key", "model_key") and isinstance(v, str) and not rec["arm"]:
            rec["arm"] = v[:60]
        if VERDICT_KEY_RE.search(k) and isinstance(v, (bool, str)) \
                and len(rec["verdict_keys"]) < 24:
            rec["verdict_keys"].append(f"{jp}={v}"[:120])

    # ---- the direct bias read: heldout split-mean vs full_set mean --------- #
    deltas = []
    for hp in rec["heldout_paths"]:
        for fp in rec["fullset_paths"]:
            # Pair only the SAME quantity's two blocks. Emitters name them
            # either `<parent>/heldout` + `<parent>/full_set` or, flat,
            # `<stem>_heldout` + `<stem>_full_set`. Without the stem check a
            # model heldout gets differenced against a CV full_set and reports
            # a meaningless 600 % "bias" (measured on refa4b_gate_30k.json).
            if hp.rsplit("/", 1)[0] != fp.rsplit("/", 1)[0]:
                continue
            hs = _stem(hp.rsplit("/", 1)[-1], HELDOUT_SUFFIXES)
            fs_ = _stem(fp.rsplit("/", 1)[-1], FULLSET_SUFFIXES)
            if hs != fs_:
                continue
            hb, fb = resolve(doc, hp), resolve(doc, fp)
            if not (isinstance(hb, dict) and isinstance(fb, dict)):
                continue
            for sub in set(hb) & set(fb):        # e.g. "model" / "cv"
                h_sub, f_sub = hb[sub], fb[sub]
                if not (isinstance(h_sub, dict) and isinstance(f_sub, dict)):
                    # flat shape: hb/fb ARE the metric dicts
                    h_sub, f_sub, sub = hb, fb, "(flat)"
                for m in ALIGN_METRICS:
                    if m in h_sub and m in f_sub:
                        hv, fv = scalarize(h_sub[m]), scalarize(f_sub[m])
                        if hv is None or fv is None:
                            continue
                        deltas.append({
                            "block": sub, "metric": m,
                            "heldout_split_mean": round(hv, 6),
                            "full_set_mean": round(fv, 6),
                            "abs_bias": round(hv - fv, 6),
                            "rel_bias_pct": (round(100.0 * (hv - fv) / fv, 4)
                                             if fv else None)})
                if sub == "(flat)":
                    break
    # dedupe
    seen, uniq = set(), []
    for d in deltas:
        key = (d["block"], d["metric"], d["heldout_split_mean"], d["full_set_mean"])
        if key not in seen:
            seen.add(key)
            uniq.append(d)
    rec["heldout_vs_fullset"] = uniq
    # the headline number specifically — max over all metrics is dominated by
    # de@0.5s, whose denominator is ~0.07 m and inflates the percentage.
    head = [d for d in uniq if d["metric"] == "ade_0_2s"
            and d["block"] in ("model", "(flat)", "grounded_rollout")]
    if head:
        h = max(head, key=lambda d: abs(d["abs_bias"]))
        rec["ade_0_2s_published_heldout"] = h["heldout_split_mean"]
        rec["ade_0_2s_corrected_full_set"] = h["full_set_mean"]
        rec["ade_0_2s_abs_bias"] = h["abs_bias"]
        rec["ade_0_2s_rel_bias_pct"] = h["rel_bias_pct"]
    rec["max_abs_bias"] = (max(abs(d["abs_bias"]) for d in uniq) if uniq else None)
    rec["max_rel_bias_pct"] = (max((abs(d["rel_bias_pct"]) for d in uniq
                                    if d["rel_bias_pct"] is not None),
                                   default=None) if uniq else None)

    # ---- verdict ---------------------------------------------------------- #
    through_biased = (rec["has_heldout_block"]
                      or rec["n_deprecated_metric_dicts"] > 0)
    if not through_biased:
        rec["verdict"] = "SAFE"
        rec["verdict_why"] = ("no split-mean / overlapping_holdout_se block "
                              "present in this artifact")
    elif uniq:
        rec["verdict"] = "CORRECTED"
        rec["verdict_why"] = ("carries BOTH blocks -> the biased split-mean and "
                              "the correct full-set mean are both readable here")
    else:
        rec["verdict"] = "SUSPECT"
        rec["verdict_why"] = ("carries a split-mean block with NO full_set "
                              "counterpart -> not recomputable from this file")
    return rec


def resolve(doc, jpath):
    cur = doc
    for part in [p for p in jpath.split("/") if p]:
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    repo = Path(a.repo).resolve()
    out_dir = Path(a.out) if a.out else Path(__file__).resolve().parent

    roots = [repo / "taniteval" / "results",
             repo / "TanitAD Research Hub",
             repo / "Project Steering",
             repo / "Benchmarks & Eval",
             repo / "stack",
             repo / "DataEng"]
    files = []
    for r in roots:
        if not r.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(r):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                if fn.lower().endswith(".json"):
                    files.append(Path(dirpath) / fn)
    files = sorted(set(files))
    print(f"[sweep] scanning {len(files)} JSON artifacts under {repo}")

    recs = [fingerprint(f, repo) for f in files]

    jl = out_dir / "jack_artifact_inventory.jsonl"
    with jl.open("w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    cols = ["file", "verdict", "arm", "n_windows", "n_episodes",
            "has_heldout_block", "has_fullset_block", "has_bootstrap_block",
            "n_deprecated_metric_dicts", "n_bootstrap_metric_dicts",
            "mentions_deprecated_name", "refuses_deprecated",
            "ade_0_2s_published_heldout", "ade_0_2s_corrected_full_set",
            "ade_0_2s_abs_bias", "ade_0_2s_rel_bias_pct",
            "max_abs_bias", "max_rel_bias_pct", "verdict_why"]
    cv = out_dir / "jack_artifact_inventory.csv"
    with cv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in recs:
            if "error" in r:
                r = {**r, "verdict": "UNPARSEABLE", "verdict_why": r["error"]}
            w.writerow(r)

    from collections import Counter
    c = Counter(r.get("verdict", "UNPARSEABLE") for r in recs)
    print("[sweep] verdicts:", dict(c))
    hits = [r for r in recs if r.get("heldout_vs_fullset")]
    hits.sort(key=lambda r: -(r.get("max_rel_bias_pct") or 0))
    print(f"[sweep] {len(hits)} artifacts carry BOTH blocks — top bias:")
    for r in hits[:40]:
        print(f"  {r['max_rel_bias_pct']:>8.3f}%  {r['max_abs_bias']:>9.4f}  "
              f"{r['file']}")
    print(f"[sweep] wrote {jl.name} + {cv.name}")


if __name__ == "__main__":
    main()
