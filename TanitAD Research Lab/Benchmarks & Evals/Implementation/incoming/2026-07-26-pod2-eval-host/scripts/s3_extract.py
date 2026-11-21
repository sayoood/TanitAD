"""Pull the power / stratum numbers out of the pod2 S3 run and diff vs pod3's."""
import json
import sys

new = json.load(open(sys.argv[1]))            # pod2 s3_power_pod2_parity_primary.json
old = sys.argv[2] if len(sys.argv) > 2 else None

print("provenance:", json.dumps(new["provenance"], indent=1)[:700])
print()
for axis in ("lat", "lon"):
    b = new[axis]
    print(f"== {axis}: keys {list(b.keys())}")
    print("   measured_on_test_cache:", json.dumps(b.get("measured_on_test_cache"))[:400])
    print("   train_cache_clusters  :", json.dumps(b.get("train_cache_clusters"))[:200])
    print("   bars                  :", json.dumps(b.get("bars"))[:300])
print()
for side in ("strata_val", "strata_train"):
    print(f"== {side}")
    for axis, strata in new[side].items():
        for name, v in strata.items():
            print(f"   {axis:<4} {name:<12} win={v.get('n_windows'):>7} "
                  f"clusters={v.get('n_episode_clusters'):>4} "
                  f">=40:{v.get('meets_single_arm_bar_40')} "
                  f">=200:{v.get('meets_two_arm_bar_200')} "
                  f"majority={v.get('majority_rate')}")

if old:
    o = json.load(open(old))
    print("\n== DIFF vs pod3 run (s3_power_parity_primary.json)")
    for side in ("strata_val", "strata_train"):
        for axis, strata in new[side].items():
            for name, v in strata.items():
                ov = o.get(side, {}).get(axis, {}).get(name, {})
                a, b_ = ov.get("n_episode_clusters"), v.get("n_episode_clusters")
                aw, bw = ov.get("n_windows"), v.get("n_windows")
                flag = "MATCH" if (a == b_ and aw == bw) else "DIFF"
                print(f"   {side:<13} {axis:<4} {name:<12} pod3 {aw}/{a}  pod2 {bw}/{b_}  {flag}")
    for axis in ("lat", "lon"):
        print(f"   {axis} measured_on_test_cache pod3:",
              json.dumps(o[axis].get("measured_on_test_cache"))[:200])
        print(f"   {axis} measured_on_test_cache pod2:",
              json.dumps(new[axis].get("measured_on_test_cache"))[:200])
