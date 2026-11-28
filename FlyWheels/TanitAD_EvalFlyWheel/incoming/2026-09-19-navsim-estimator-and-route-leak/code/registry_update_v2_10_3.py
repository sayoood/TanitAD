#!/usr/bin/env python3
"""W2 (E3) registry 2.10.2 -> 2.10.3: state the POST-SETTLEMENT form of the estimator gate
in the registry itself, and kill the 2.9.0 reading.

⛔ WHY (MEASURED 2026-09-20). E1 reported that `GATE_estimator_cluster_unit` "still reads
*until the unit is settled and PRE-REGISTERED, the ONLY admissible interval is
{status: UNAVAILABLE}*" and therefore blocked its valid navhard interval (76 log clusters,
EPDMS x100 11.4816, CI [8.25, 14.50]). I checked at source instead of acting on the report:

  * the WORKING-TREE registry is 2.10.2 and already admits that interval;
  * the sentence E1 quoted is at **HEAD, registry 2.9.0** (`git show HEAD:...`), because this
    stream's registry update is STAGED AND NOT COMMITTED — agents never commit;
  * E1's artifact therefore down-declared a real interval into
    `estimator.interval.summary_interval` and set `estimator.cluster_unit` to
    `{status: UNAVAILABLE, n: 76, reason: "log_name"}` — the UNIT NAME in the reason field.

⇒ The gate did not need loosening. What it needed was (a) to SAY its post-settlement form in
the registry, so a reader cannot re-derive the 2.9.0 reading, and (b) to catch the two shapes
E1's artifact actually has, both of which passed silently as "an honest refusal":
a FALSE REFUSAL holding an admissible interval, and a one-token refusal reason.

Run: PYTHONIOENCODING=utf-8 <venv>/python.exe code/registry_update_v2_10_3.py
"""
import json
import sys

P = r"D:\Projects\TanitAD\products\P7-TanitEval\CRITERIA_REGISTRY.json"

POST_SETTLEMENT = {
    "PASS": ("a `navsim_log_cluster_bootstrap` (or its paired form) over >= min_clusters and "
             "<= the split's `log_name` count, carrying the protocol's official aggregation, "
             "n_boot >= n_boot_min and numeric lo/hi. MEASURED EXAMPLE (E1, 2026-09-20): navhard "
             "CV, 76 log clusters, EPDMS x100 = 11.4816, CI [8.25, 14.50] — this PASSES."),
    "REFUSED": ("{status: UNAVAILABLE, reason, n} with a REAL reason — a WORK ITEM, never a pass. "
                "warmup_two_stage is the standing case: 7 log_names < the RG-14 floor of 8."),
    "FAIL": [
        "scene-token / mapping-key / episode-cluster / overlapping_holdout_se resampling",
        "n_clusters < min_clusters (the RG-14 floor of 8)",
        "n_clusters > the split's log_name count — a unit finer than log_name was resampled",
        "an unregistered aggregation, or one that is not this protocol's official aggregation",
        "n_boot < n_boot_min",
        "a bare point estimate with no interval block, or no `estimator.cluster_unit` at all",
        "an interval block with no numeric lo/hi",
        ("FALSE REFUSAL: {status: UNAVAILABLE} declared while an ADMISSIBLE interval sits "
         "nested inside the same block (e.g. under `summary_interval`) — MEASURED on E1's "
         "navhard CV artifact 2026-09-20"),
        ("a `cluster_unit` REFUSAL whose `reason` is a single token — the unit name stuffed "
         "into the reason field (same artifact: reason='log_name')"),
    ],
}

SUPERSEDES = (
    "⛔ registry 2.9.0's `admissible_until_settled` — *'until the unit is settled and "
    "PRE-REGISTERED, the only admissible interval is {status:UNAVAILABLE}'* — is DEAD from "
    "2.10.0. MEASURED 2026-09-20: E1 built its navhard artifact against the 2.9.0 copy AT HEAD "
    "(`git show HEAD:products/P7-TanitEval/CRITERIA_REGISTRY.json` reads version 2.9.0) because "
    "this gate's update is STAGED AND NOT COMMITTED, and consequently down-declared a valid "
    "76-cluster interval. ⭐ THE GENERAL RULE THAT COST: an artifact is built against the "
    "registry a process can READ — a staged gate is not in force for anyone until it is "
    "COMMITTED, and 'the working tree says otherwise' is not a defence a second agent can use."
)

FALSE_REFUSAL = (
    "An artifact that declares UNAVAILABLE while HOLDING an admissible interval FAILS. The gate "
    "exists so that a NavSim number carries its interval; an artifact that HAS one and hides it "
    "is misreporting, not declining — and a false refusal reads exactly like an honest one, so "
    "it must be detected rather than trusted. Implementation: "
    "`tools/criteria_check.py::_admissible_interval_hiding_in`, which validates the nested block "
    "with the SAME validator the declared interval goes through (`_navsim_interval_ok`), so the "
    "gate cannot admit a block in one place and reject the identical block in the other."
)


def main() -> int:
    with open(P, encoding="utf-8") as f:
        reg = json.load(f)
    if reg.get("version") != "2.10.2":
        print(f"expected 2.10.2, found {reg.get('version')!r} — refusing", file=sys.stderr)
        return 2
    g = reg["benchmarks"]["navsim"]["GATE_estimator_cluster_unit"]
    g["post_settlement_form"] = POST_SETTLEMENT
    g["supersedes"] = SUPERSEDES
    g["false_refusal_rule"] = FALSE_REFUSAL
    g["status"] = ("SETTLED 2026-09-19 (was UNRESOLVED through registry 2.9.0); "
                   "post-settlement form stated in the registry 2026-09-20 (2.10.3)")
    reg["version"] = "2.10.3"
    hist = reg.setdefault("_version_history", {})
    hist["2.10.3"] = ("2026-09-20 (W2/E3): GATE_estimator_cluster_unit states its POST-SETTLEMENT "
                      "form, supersedes the 2.9.0 text explicitly, and FAILS a false refusal "
                      "(UNAVAILABLE while holding an admissible interval) and a one-token "
                      "cluster_unit refusal reason. No rule was loosened: the working-tree gate "
                      "had admitted a valid log-cluster interval since 2.10.0.")
    with open(P, "w", encoding="utf-8", newline="\n") as f:
        json.dump(reg, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("registry -> 2.10.3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
