"""Non-parity (S3's first pass) vs PARITY (this decision-grade re-run).

Honours the pre-registration: nothing is re-tuned, nothing is re-banded. This
only reads the two JSON sets and reports what MOVED. A material move is itself
a finding and is reported as one -- the parity number is not quietly adopted.
"""
import json, sys
from pathlib import Path

OLD = Path(r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\TanitAD Research Hub"
           r"\Architecture & Inference\Implementation\incoming\2026-07-26-4brain-s3")
NEW = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
OUT = Path(r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\TanitAD Research Hub"
           r"\Architecture & Inference\Implementation\incoming"
           r"\2026-07-26-s3-decision-grade")
TAG_OLD, TAG_NEW = "primary", "parity_primary"


def J(base, kind, tag):
    return json.loads((base / f"s3_{kind}_{tag}.json").read_text())


ocov, ncov = J(OLD, "coverage", TAG_OLD), J(NEW, "coverage", TAG_NEW)
oopt, nopt = J(OLD, "option_set", TAG_OLD), J(NEW, "option_set", TAG_NEW)
opw, npw = J(OLD, "power", TAG_OLD), J(NEW, "power", TAG_NEW)
ofw, nfw = J(OLD, "blind_baseline", TAG_OLD), J(NEW, "blind_baseline", TAG_NEW)

R = {"artifact": "S3 decision-grade re-run: NON-PARITY vs PARITY",
     "date": "2026-07-26",
     "non_parity": {"train": "physicalai-train-14231cd29c74 (400 eps, dev box)",
                    "val": "physicalai-val-bb543bdf7836 (100 eps, dev box)",
                    "byte_level_overlap_with_parity_train": "58/400 = 14.5% (train), "
                                                            "14/100 = 14.0% (val)"},
     "parity": {"train": "physicalai-train-e438721ae894 (2376 eps) -- PARITY=YES",
                "val": "physicalai-val-0c5f7dac3b11 (600 eps) -- PARITY=YES, "
                       "0/600 byte-level overlap with the parity train"},
     "changed_between_runs": "ONLY --train-cache / --test-cache. Zero code change, "
                             "no re-banding, no re-tuning. Band edges remain the "
                             "spec's own 2/5/10 s; A_MAN=0.5, DV_MIN=1.5 unchanged.",
     }


def d(a, b):
    try:
        return round(b - a, 4)
    except Exception:
        return None


# ---------------- coverage ----------------
cov = {}
for ax in ("lat", "lon"):
    o, n = ocov[f"val_{ax}"], ncov[f"val_{ax}"]
    cov[ax] = {k: {"non_parity": o[k], "parity": n[k], "delta": d(o[k], n[k])}
               for k in ("n_windows_total", "n_after_M1_observable_horizon",
                         "n_after_M1_M4_moving", "n_admissible",
                         "coverage_of_all_windows", "coverage_of_M1M4_windows",
                         "majority_rate", "event_rate", "episode_yield_rate",
                         "n_episodes_total", "n_episodes_with_decision_point",
                         "n_episodes_with_an_EVENT")}
    cov[ax]["majority_class"] = {"non_parity": o["majority_class"],
                                 "parity": n["majority_class"]}
    cov[ax]["class_balance"] = {
        c: {"non_parity": o["class_balance"][c], "parity": n["class_balance"][c],
            "delta": d(o["class_balance"][c], n["class_balance"][c])}
        for c in o["class_balance"]}
    cov[ax]["ttm_mean_s"] = {
        "non_parity": o["ttm_distribution_event_windows"].get("mean"),
        "parity": n["ttm_distribution_event_windows"].get("mean")}
    cov[ax]["ttm_deciles_s"] = {
        "non_parity": o["ttm_distribution_event_windows"].get("deciles_s"),
        "parity": n["ttm_distribution_event_windows"].get("deciles_s")}
R["coverage_and_class_balance_VAL"] = cov

# ---------------- power ----------------
pw = {}
for ax in ("lat", "lon"):
    o, n = opw[ax], npw[ax]
    pw[ax] = {
        "non_parity_measured_on": o["measured_on_test_cache"],
        "PARITY_measured_on": n["measured_on_test_cache"],
        "yield_delta": d(o["measured_on_test_cache"]["episode_yield_rate"],
                         n["measured_on_test_cache"]["episode_yield_rate"]),
        "S3_ESTIMATED_600ep_projection_from_the_100ep_cache":
            o["projections"]["training_pod_val_600ep"],
        "MEASURED_on_the_real_600ep_val": {
            "n_val_episodes": n["measured_on_test_cache"]["n_episodes"],
            "clusters_with_decision_point":
                n["measured_on_test_cache"]["clusters_with_decision_point"],
            "clusters_with_an_event":
                n["measured_on_test_cache"]["clusters_with_an_event"]},
        "meets_two_arm_bar_200": bool(
            n["measured_on_test_cache"]["clusters_with_decision_point"] >= 200),
        "meets_single_arm_bar_40": bool(
            n["measured_on_test_cache"]["clusters_with_decision_point"] >= 40),
        "re_projection_to_the_published_40ep_set_using_the_PARITY_yield":
            n["projections"]["published_eval_val_40ep"],
    }
R["power"] = pw
R["strata_val_PARITY"] = npw["strata_val"]
R["strata_val_non_parity"] = opw["strata_val"]

# ---------------- option set ----------------
os_ = {}
for ax in ("lat", "lon"):
    o, n = oopt[f"val_{ax}"], nopt[f"val_{ax}"]
    os_[ax] = {
        "PRIMARY_spec_edges": {
            "edges_s": n["PRIMARY_spec_edges"]["edges_s"],
            "balance_non_parity": o["PRIMARY_spec_edges"]["balance"],
            "balance_parity": n["PRIMARY_spec_edges"]["balance"],
            "majority_rate_non_parity": o["PRIMARY_spec_edges"]["majority_rate"],
            "majority_rate_parity": n["PRIMARY_spec_edges"]["majority_rate"],
            "qwk_of_majority_baseline": 0.0},
        "SECONDARY_equal_mass_quartiles": {
            "edges_non_parity": (o["SECONDARY_equal_mass_quartiles"] or {}).get("edges_s"),
            "edges_parity": (n["SECONDARY_equal_mass_quartiles"] or {}).get("edges_s"),
            "majority_non_parity": (o["SECONDARY_equal_mass_quartiles"] or {}).get("majority_rate"),
            "majority_parity": (n["SECONDARY_equal_mass_quartiles"] or {}).get("majority_rate")},
        "mae_median_constant": {
            "non_parity": o["mae_median_constant_baseline"],
            "parity": n["mae_median_constant_baseline"]},
    }
R["option_set"] = os_

# ---------------- firewall / skill bars ----------------
fw = {}
for ax in ("lat", "lon"):
    o, n = ofw[ax], nfw[ax]
    arms = {}
    for a in ("B1_sensor_only", "B2_plus_route", "B3_FULL_CONDITIONING",
              "B4_plus_clock"):
        oc, nc = o["arms"][a]["blind_qwk_ci"], n["arms"][a]["blind_qwk_ci"]
        arms[a] = {
            "non_parity": {"qwk": o["arms"][a]["blind"]["qwk"],
                           "ci": [oc["lo"], oc["hi"]],
                           "estimator": oc["estimator"], "n_boot": oc["n_boot"],
                           "n_episodes": oc["n_episodes"]},
            "parity": {"qwk": n["arms"][a]["blind"]["qwk"],
                       "ci": [nc["lo"], nc["hi"]],
                       "estimator": nc["estimator"], "n_boot": nc["n_boot"],
                       "n_episodes": nc["n_episodes"]},
            "delta": d(o["arms"][a]["blind"]["qwk"], n["arms"][a]["blind"]["qwk"]),
            "per_band_recall_parity": n["arms"][a]["blind"]["per_band_recall"],
        }
    fw[ax] = {
        "arms": arms,
        "operative_blind_floor": {"non_parity": o["operative_blind_floor"],
                                  "parity": n["operative_blind_floor"]},
        "R1_circular": {"non_parity": o["verdict_R1_circular"],
                        "parity": n["verdict_R1_circular"]},
        "paired_leak_B2_minus_B1": {"non_parity": o["paired_leak_B2_minus_B1"],
                                    "parity": n["paired_leak_B2_minus_B1"]},
        "paired_leak_B3_minus_B1": {"non_parity": o["paired_leak_B3_minus_B1"],
                                    "parity": n["paired_leak_B3_minus_B1"]},
        "paired_clock_B4_minus_B3": {"non_parity": o["paired_clock_B4_minus_B3"],
                                     "parity": n["paired_clock_B4_minus_B3"]},
        "SKILL_BARS": {
            "S3_as_specified": {
                "non_parity": o["pre_registered_skill_bars"]["S3_as_specified_route_and_vt_GIVEN"],
                "PARITY": n["pre_registered_skill_bars"]["S3_as_specified_route_and_vt_GIVEN"]},
            "S3_W_withheld": {
                "non_parity": o["pre_registered_skill_bars"]["S3_W_withheld_conditioning_pixels_and_v0_only"],
                "PARITY": n["pre_registered_skill_bars"]["S3_W_withheld_conditioning_pixels_and_v0_only"]},
        },
    }
R["firewall_and_skill_bars"] = fw

(OUT / "s3_parity_vs_nonparity.json").write_text(json.dumps(R, indent=2))

# ------------------- console -------------------
print("=" * 96)
print("POWER  (the ceiling question, now MEASURED on the real 600-episode val)")
print("=" * 96)
for ax in ("lat", "lon"):
    p = pw[ax]
    m, e = p["MEASURED_on_the_real_600ep_val"], p["S3_ESTIMATED_600ep_projection_from_the_100ep_cache"]
    print(f"  {ax}: S3 PROJECTED {e['projected_clusters_with_decision_point']} clusters "
          f"-> MEASURED {m['clusters_with_decision_point']} of {m['n_val_episodes']} eps"
          f"   >=200? {p['meets_two_arm_bar_200']}   (events: "
          f"proj {e['projected_clusters_with_an_event']} -> meas {m['clusters_with_an_event']})")
    print(f"        yield non-parity {p['non_parity_measured_on']['episode_yield_rate']} "
          f"-> parity {p['PARITY_measured_on']['episode_yield_rate']}  (delta {p['yield_delta']})")
    print(f"        re-projected onto the published 40-ep set: "
          f"{p['re_projection_to_the_published_40ep_set_using_the_PARITY_yield']}")

print()
print("=" * 96)
print("SKILL BARS  (S3 as specified  vs  S3-W withheld)")
print("=" * 96)
for ax in ("lat", "lon"):
    s = fw[ax]["SKILL_BARS"]
    print(f"  {ax}:  S3   non-parity {s['S3_as_specified']['non_parity']:.4f}"
          f"  ->  PARITY {s['S3_as_specified']['PARITY']:.4f}"
          f"   (delta {d(s['S3_as_specified']['non_parity'], s['S3_as_specified']['PARITY']):+.4f})")
    print(f"       S3-W non-parity {s['S3_W_withheld']['non_parity']:.4f}"
          f"  ->  PARITY {s['S3_W_withheld']['PARITY']:.4f}"
          f"   (delta {d(s['S3_W_withheld']['non_parity'], s['S3_W_withheld']['PARITY']):+.4f})")

print()
print("=" * 96)
print("FIREWALL ARMS  (episode_cluster_bootstrap B=2000)")
print("=" * 96)
for ax in ("lat", "lon"):
    print(f"  --- {ax}")
    for a, v in fw[ax]["arms"].items():
        o, n = v["non_parity"], v["parity"]
        print(f"    {a:<24} non-par {o['qwk']:+.4f} [{o['ci'][0]:+.4f},{o['ci'][1]:+.4f}]"
              f"   PARITY {n['qwk']:+.4f} [{n['ci'][0]:+.4f},{n['ci'][1]:+.4f}]  "
              f"(d {v['delta']:+.4f})")
    for k in ("paired_leak_B2_minus_B1", "paired_leak_B3_minus_B1",
              "paired_clock_B4_minus_B3"):
        o, n = fw[ax][k]["non_parity"], fw[ax][k]["parity"]
        print(f"    {k:<24} non-par {o['delta']:+.4f} [{o['lo']:+.4f},{o['hi']:+.4f}] "
              f"sep={o['separated']}   PARITY {n['delta']:+.4f} "
              f"[{n['lo']:+.4f},{n['hi']:+.4f}] sep={n['separated']}")
    print(f"    R1 REFUSED? non-par {fw[ax]['R1_circular']['non_parity']['REFUSED']}"
          f"   PARITY {fw[ax]['R1_circular']['parity']['REFUSED']}")

print()
print("=" * 96)
print("COVERAGE / CLASS BALANCE (val)")
print("=" * 96)
for ax in ("lat", "lon"):
    c = cov[ax]
    print(f"  --- {ax}   admissible {c['n_admissible']['non_parity']} -> "
          f"{c['n_admissible']['parity']}   coverage "
          f"{c['coverage_of_all_windows']['non_parity']} -> "
          f"{c['coverage_of_all_windows']['parity']}   majority "
          f"{c['majority_rate']['non_parity']} -> {c['majority_rate']['parity']}"
          f"  ({c['majority_class']['non_parity']} -> {c['majority_class']['parity']})")
    for cl, v in c["class_balance"].items():
        print(f"       {cl:<8} {v['non_parity']:.4f} -> {v['parity']:.4f}  ({v['delta']:+.4f})")
    print(f"       ttm mean {c['ttm_mean_s']['non_parity']} -> {c['ttm_mean_s']['parity']} s")

print()
print("STRATA (parity val, per axis, episode clusters):")
for ax in ("lat", "lon"):
    for s, v in npw["strata_val"][ax].items():
        print(f"  {ax} {s:<10} clusters={v['n_episode_clusters']:<5} "
              f"windows={v['n_windows']:<7} >=40:{v['meets_single_arm_bar_40']!s:<5} "
              f">=200:{v['meets_two_arm_bar_200']!s:<5} median_ttm={v['median_ttm_s']}")
print("\nwrote", OUT / "s3_parity_vs_nonparity.json")
