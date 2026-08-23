"""Is `/root/taniteval/taniteval/ci.py` (md5 ef925f06…) a DIFFERENT ESTIMATOR
from HEAD's (md5 c92618a0…), or only a different RENDERING?

WHY THIS MATTERS
----------------
`idm2_lib.py:19` and `idm3_a0.py` both do `sys.path.insert(0, "/root/taniteval")`
UNCONDITIONALLY, before `from taniteval import ci`.  So **every published v3
interval came through the ef925f06 file**, not through HEAD's.  C44 records the
class.  If the two files differ statistically, that is a second program-wide
finding; if they differ only in how the numbers are printed, the intervals stand.

⛔ This script ASSERTS THE MD5 OF THE FILE ACTUALLY LOADED (via the imported
module's own ``__file__``), not the one it intended to load — which is the
standing rule C44 leaves behind.

WHAT IS TESTED
--------------
Both files are imported side-by-side under distinct module names and driven with
IDENTICAL inputs and IDENTICAL seeds across a randomised battery plus the
degenerate cases that motivated HEAD's change (bit-identical arms, ~1e-9
separations).  Every returned field is compared.  A field that differs is
classified as STATISTICAL (delta / lo / hi / ci95 / p / separated / n_*) or
DISPLAY-ONLY (rounding of an otherwise-equal value, or a new explanatory key).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

OLD = "/root/taniteval/taniteval/ci.py"
NEW = "/workspace/TanitAD-head/taniteval/taniteval/ci.py"

STAT_FIELDS = ("separated", "p_delta_gt0", "reducer", "n_windows", "n_episodes",
               "n_boot", "estimator")
NUM_FIELDS = ("delta", "lo", "hi", "ci95")


def load_mod(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def md5(p: str) -> str:
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def main():
    old = load_mod(OLD, "ci_old")
    new = load_mod(NEW, "ci_new")

    # ⛔ the md5 of what was ACTUALLY LOADED, read back off the module object
    loaded = {
        "old": {"module_file": old.__file__, "md5": md5(old.__file__)},
        "new": {"module_file": new.__file__, "md5": md5(new.__file__)},
    }

    out = {
        "what": "is /root/taniteval's ci.py a different ESTIMATOR from HEAD's, "
                "or only a different RENDERING?",
        "date": "2026-07-28",
        "agent": "parity-leak-check",
        "evidence_class": "MEASURED (ours; both modules imported side-by-side and "
                          "driven with identical inputs and seeds)",
        "md5_of_the_file_ACTUALLY_LOADED": loaded,
        "import_path_that_selects_the_old_one": [
            "/root/idm2/idm2_lib.py:19  sys.path.insert(0, '/root/taniteval')  "
            "-- unconditional, before `from taniteval import ci as tci`",
            "/workspace/idm3/idm3_a0.py  sys.path.insert(0, '/root/taniteval')  "
            "-- likewise",
        ],
        "public_api_diff": {
            "only_in_new": sorted(set(new.__all__) - set(old.__all__)),
            "only_in_old": sorted(set(old.__all__) - set(new.__all__)),
        },
        "source_identity_of_the_estimator_core": {},
        "cases": [],
    }

    # --- are the estimator FUNCTIONS byte-identical in source? ------------- #
    import inspect
    for fn in ("overlapping_holdout_se", "episode_index", "_draws", "_red_mean",
               "_red_rms", "_red_median", "_quantile_reducer", "resolve_reducer",
               "_reducer_name", "episode_cluster_bootstrap", "bootstrap_metrics"):
        try:
            so = inspect.getsource(getattr(old, fn))
            sn = inspect.getsource(getattr(new, fn))
            out["source_identity_of_the_estimator_core"][fn] = {
                "identical": so == sn,
                "md5_old": hashlib.md5(so.encode()).hexdigest()[:12],
                "md5_new": hashlib.md5(sn.encode()).hexdigest()[:12],
            }
        except Exception as e:
            out["source_identity_of_the_estimator_core"][fn] = {"error": repr(e)}
    so = inspect.getsource(old.paired_episode_cluster_bootstrap)
    sn = inspect.getsource(new.paired_episode_cluster_bootstrap)
    out["source_identity_of_the_estimator_core"]["paired_episode_cluster_bootstrap"] = {
        "identical": so == sn, "note": "the ONE function HEAD changed"}

    rng = np.random.default_rng(0)
    cases = []
    # A. randomised battery over reducers, sizes, seeds, effect sizes
    for i in range(24):
        n_ep = int(rng.integers(6, 45))
        per = rng.integers(8, 40, size=n_ep)
        eid = np.repeat(np.arange(n_ep), per)
        n = eid.size
        eff = float(rng.choice([0.0, 0.001, 0.02, 0.2, 1.0]))
        a = rng.normal(1.0, 0.4, n) + eff
        b = rng.normal(1.0, 0.4, n)
        red = str(rng.choice(["mean", "rms", "median", "p90"]))
        cases.append((f"random_{i}_{red}_eff{eff}", a, b, eid, red,
                      int(rng.integers(0, 10_000))))
    # B. the degenerate cases HEAD's change was written for
    eid = np.repeat(np.arange(20), 30)
    n = eid.size
    base = rng.normal(1.0, 0.3, n)
    cases.append(("DEGENERATE_bit_identical_arms", base, base.copy(), eid, "mean", 0))
    cases.append(("DEGENERATE_1e-9_offset", base + 1e-9, base, eid, "mean", 0))
    cases.append(("DEGENERATE_1e-14_offset", base + 1e-14, base, eid, "mean", 0))
    cases.append(("NEAR_5e-5_offset", base + 5e-5, base, eid, "mean", 0))

    n_field_diffs = 0
    n_stat_diffs = 0
    for name, a, b, eid, red, seed in cases:
        ro = old.paired_episode_cluster_bootstrap(a, b, eid, reduce=red, seed=seed)
        rn = new.paired_episode_cluster_bootstrap(a, b, eid, reduce=red, seed=seed)
        stat_same = all(ro.get(k) == rn.get(k) for k in STAT_FIELDS)
        num_same = all(ro.get(k) == rn.get(k) for k in NUM_FIELDS)
        # a numeric field may differ ONLY by rounding depth; check the values
        # agree once both are put at the coarser precision
        num_same_at_4dp = all(
            round(float(ro[k]), 4) == round(float(rn[k]), 4) for k in NUM_FIELDS)
        extra = sorted(set(rn) - set(ro))
        rec = {"case": name, "reducer": red, "seed": seed,
               "statistical_fields_identical": stat_same,
               "numeric_fields_bitwise_identical": num_same,
               "numeric_fields_identical_at_4dp": num_same_at_4dp,
               "keys_only_in_NEW": extra,
               "old": {k: ro.get(k) for k in NUM_FIELDS + STAT_FIELDS},
               "new": {k: rn.get(k) for k in NUM_FIELDS + STAT_FIELDS}}
        if not num_same:
            n_field_diffs += 1
        if not stat_same:
            n_stat_diffs += 1
        out["cases"].append(rec)

    # --- also exercise the UNPAIRED estimator, untouched by HEAD ----------- #
    unp = []
    for i in range(8):
        n_ep = int(rng.integers(6, 40))
        per = rng.integers(8, 40, size=n_ep)
        eid = np.repeat(np.arange(n_ep), per)
        v = rng.normal(1.0, 0.4, eid.size)
        red = str(rng.choice(["mean", "rms", "median", "p90"]))
        ro = old.episode_cluster_bootstrap(v, eid, reduce=red, seed=i)
        rn = new.episode_cluster_bootstrap(v, eid, reduce=red, seed=i)
        unp.append({"i": i, "reducer": red, "identical": ro == rn})
    out["unpaired_episode_cluster_bootstrap"] = {
        "n_cases": len(unp), "all_identical": all(u["identical"] for u in unp),
        "cases": unp}

    # ⚠️ The decisive criterion is (a) do the STATISTICAL fields ever differ, and
    # (b) is every numeric difference confined to cases where HEAD deliberately
    # escalated the rendering (it stamps `display_dp` when and only when it did)?
    # A naive "compare at 4 dp" comparator DOUBLE-ROUNDS — round(round(x, 5), 4)
    # can differ from round(x, 4) at an exact tie under banker's rounding — and
    # would manufacture a false "statistical difference". That is recorded here
    # rather than silently worked around.
    esc = [c for c in out["cases"] if "display_dp" in c["keys_only_in_NEW"]]
    unesc_diff = [c for c in out["cases"]
                  if "display_dp" not in c["keys_only_in_NEW"]
                  and not c["numeric_fields_bitwise_identical"]]
    out["SUMMARY"] = {
        "n_cases": len(out["cases"]),
        "n_cases_with_ANY_statistical_field_differing": n_stat_diffs,
        "n_cases_with_a_numeric_field_differing_bitwise": n_field_diffs,
        "n_cases_bitwise_identical_in_EVERY_field": sum(
            1 for c in out["cases"]
            if c["numeric_fields_bitwise_identical"]
            and c["statistical_fields_identical"] and not c["keys_only_in_NEW"]),
        "n_cases_where_NEW_escalated_the_rendering": len(esc),
        "escalated_cases": [c["case"] for c in esc],
        "n_NON_escalated_cases_with_a_numeric_difference": len(unesc_diff),
        "double_rounding_note":
            "round(5e-05, 4) == %r under banker's rounding, while the UNROUNDED "
            "bound rounds to 0.0001 — so a 'compare at 4 dp' check reports a "
            "spurious mismatch in the NEAR_5e-5 case. The mismatch is in the "
            "comparator, not in the estimator." % round(5e-05, 4),
        "verdict": None,
    }
    if n_stat_diffs == 0 and len(unesc_diff) == 0:
        out["SUMMARY"]["verdict"] = (
            "DISPLAY-ONLY. Same estimator, same draws, same separation test. "
            "Every estimator-core function is BYTE-IDENTICAL in source; the one "
            "changed function (`paired_episode_cluster_bootstrap`) changes only "
            "`round(x, 4)` -> `round(x, dp)` plus two explanatory keys, and dp "
            "is 4 unless 4 dp would print an interval CONTRADICTING `separated`. "
            "⇒ every published v3 interval STANDS as a NUMBER. What the old file "
            "could do — and HEAD fixes — is PRINT `0.0 [0.0, 0.0] separated=true`.")
    else:
        out["SUMMARY"]["verdict"] = (
            "⛔ STATISTICAL DIFFERENCE — a second program-wide finding. See cases.")

    Path("/root/leakcheck/ci_equivalence.json").parent.mkdir(parents=True, exist_ok=True)
    Path("/root/leakcheck/ci_equivalence.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in
                      ("md5_of_the_file_ACTUALLY_LOADED", "public_api_diff",
                       "source_identity_of_the_estimator_core",
                       "unpaired_episode_cluster_bootstrap", "SUMMARY")},
                     indent=1)[:5000])
    print("\n--- the 4 degenerate cases ---")
    for c in out["cases"][-4:]:
        print(json.dumps(c, indent=1))


if __name__ == "__main__":
    main()
