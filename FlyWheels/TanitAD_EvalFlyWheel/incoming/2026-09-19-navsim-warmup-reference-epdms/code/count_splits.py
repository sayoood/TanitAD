import yaml, json
base = "C:/Users/Admin/navsim/devkit/navsim/planning/script/config/common/train_test_split/"
out = {}
for split in ("warmup_two_stage", "navhard_two_stage"):
    tts = yaml.safe_load(open(base + split + ".yaml", encoding="utf-8"))
    sf = yaml.safe_load(open(base + "scene_filter/" + split + ".yaml", encoding="utf-8"))
    m = tts["reactive_all_mapping"]
    s1_from_map = set(); s2_now = set(); s2_prev = set(); pair_counts = []
    for orig, prev, pairs in m:
        s1_from_map |= {orig, prev}
        pair_counts.append(len(pairs))
        for p in pairs:
            s2_now.add(p[0]); s2_prev.add(p[1])
    rec = {
        "data_split": tts.get("data_split"),
        "n_mapping_entries": len(m),
        "stage1_tokens_from_mapping": len(s1_from_map),
        "stage2_now_tokens": len(s2_now), "stage2_prev_tokens": len(s2_prev),
        "stage2_tokens_from_mapping": len(s2_now | s2_prev),
        "stage2_pairs_total": sum(pair_counts),
        "pairs_per_entry_min_max": (min(pair_counts), max(pair_counts)),
        "sf_keys": sorted(sf.keys()),
        "sf_log_names": len(sf.get("log_names") or []),
        "sf_tokens": len(sf.get("tokens") or []),
        "sf_reactive_synthetic_initial_tokens": len(sf.get("reactive_synthetic_initial_tokens") or []),
        "sf_synthetic_scene_tokens": (None if sf.get("synthetic_scene_tokens") is None else len(sf["synthetic_scene_tokens"])),
        "sf_non_reactive": (None if sf.get("non_reactive_synthetic_initial_tokens") is None else len(sf["non_reactive_synthetic_initial_tokens"])),
        "stage1_map_eq_sf_tokens": s1_from_map == set(sf.get("tokens") or []),
        "stage2_map_eq_sf_reactive": (s2_now | s2_prev) == set(sf.get("reactive_synthetic_initial_tokens") or []),
        "tok_len_stage1": sorted({len(t) for t in s1_from_map}), "tok_len_stage2": sorted({len(t) for t in (s2_now | s2_prev)}),
        "num_history_frames": sf.get("num_history_frames"), "num_future_frames": sf.get("num_future_frames"),
        "frame_interval": sf.get("frame_interval"), "include_synthetic_scenes": sf.get("include_synthetic_scenes"),
    }
    out[split] = rec
print(json.dumps(out, indent=1))
json.dump(out, open("split_counts.json", "w"), indent=1)
