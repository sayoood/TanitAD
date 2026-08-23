import json, sys
r = json.load(open(sys.argv[1]))
print("=" * 78)
print("LABEL:", r["label"])
print("=" * 78)
a = r["THE_ANSWER"]
print("VERDICT:", a["verdict"])
print("  n_val:", a["n_val_episodes"], " n_train:", a["n_train_episodes"])
print("  overlap by poses_sha256 :", a["n_overlap_by_poses_sha256"])
print("  overlap by frames_sha256:", a["n_overlap_by_frames_sha256"])
print("  frac of val episodes:", round(a["frac_of_val_episodes"], 4))
print()
print("--- CONTROLS ---")
c = r["controls"]
print(" ALL_CONTROLS_PASS:", c["ALL_CONTROLS_PASS"])
print(" SELF   passes:", c["SELF"]["passes"], c["SELF"]["per_family"])
print(" SPIKE  passes:", c["SPIKE"]["passes"], "k=", c["SPIKE"]["k_injected"],
      "found_poses=", c["SPIKE"]["n_found_by_poses_sha256"],
      "found_frames=", c["SPIKE"]["n_found_by_frames_sha256"],
      "ids=", c["SPIKE"]["identities_recovered"])
print(" MUTANT passes:", c["MUTANT"]["passes"], c["MUTANT"]["n_matched_poses_sha256"],
      c["MUTANT"]["n_matched_poses_xy_sha256"])
print()
print("--- HASH FAMILY AGREEMENT ---")
for k, v in r["hash_family_agreement"].items():
    print("  %s: %s" % (k, v))
print()
print("--- NAME-DERIVED CROSS-CHECKS (not evidence) ---")
for k, v in r["name_derived_cross_checks"].items():
    print("  %s: %s" % (k, v))
print()
print("--- WITHIN-CACHE DUPLICATES ---")
for k, v in r["within_cache_duplicates"].items():
    print(" ", k)
    for f, vv in v.items():
        print("    %s: distinct=%s/%s dupgroups=%s" %
              (f, vv["n_distinct"], vv["n_episodes"], vv["n_duplicate_groups"]))
        if vv["n_duplicate_groups"]:
            print("      groups:", vv["duplicate_groups"][:5])
print()
print("--- PARTIAL / SHIFTED FRAME CONTAINMENT ---")
p = r["partial_frame_containment"]
print(" ", dict((k, v) for k, v in p.items() if k != "top20"))
if p.get("available"):
    for row in p["top20"][:6]:
        print("   ", row)
print()
if a["n_overlap_by_poses_sha256"]:
    print("--- LEAK DETAIL (first 8) ---")
    for d in a["detail"][:8]:
        print("  ", dict((k, d[k]) for k in ("val_tag", "train_tags", "poses_bitwise_equal",
                                             "episode_id_agrees", "T", "v_max", "path_len_m")))
