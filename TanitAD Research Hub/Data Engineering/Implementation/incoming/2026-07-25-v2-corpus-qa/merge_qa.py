"""Merge the two shard scans + parity profile + P3 + P4 into v2_corpus_qa.json."""
from __future__ import annotations
import json, os, sys
import numpy as np, pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SP, "v2_corpus_qa.json")

MAN = ["lk", "tl", "tr", "ac", "bs"]
MAN2 = ["lk2", "tl2", "tr2", "ac2", "bs2"]
NAMES = ["lane_keep", "turn_left", "turn_right", "accelerate", "brake_stop"]
TARGET = dict(zip(NAMES, [0.45, 0.14, 0.14, 0.13, 0.14]))
TSPD = {"stopped": 0.10, "city": 0.52, "highway": 0.38}

def jload(p):
    with open(os.path.join(SP, p)) as f:
        return json.load(f)

shards, frames = {}, []
for tag in ("pod1", "pod3"):
    j = jload(f"qa_full_{tag}.json")
    d = pd.read_csv(os.path.join(SP, f"qa_full_{tag}.csv"))
    d["shard"] = tag
    frames.append(d)
    shards[tag] = j
df = pd.concat(frames, ignore_index=True)
ok = df[df["ok"].astype(str).str.lower().isin(["true", "1"])].copy()

parity = jload("parity_profile.json")
p3 = jload("p3_disjoint.json")
p4a = jload("p4_pod1.json")
p4b = jload("p4b_pod1.json")


def profile(counts_row, nlab, spd_steps, T, presence_df, label):
    tot = sum(counts_row.values())
    return {
        "n_label_steps": int(tot),
        "maneuver_v1": {n: round(counts_row[k] / tot, 5)
                        for n, k in zip(NAMES, MAN)},
        "turns_LR": round((counts_row["tl"] + counts_row["tr"]) / tot, 5),
    }


def man_frac(sub, cols=MAN):
    c = sub[cols].sum()
    return {n: round(float(c[k]) / float(c.sum()), 5) for n, k in zip(NAMES, cols)}


def spd_frac(sub):
    T = sub["T_out"].to_numpy(float)
    return {k2: round(float((sub[k].to_numpy(float) * T).sum() / T.sum()), 5)
            for k, k2 in (("stopped", "stopped"), ("city", "city"), ("hw", "highway"))}


def presence(sub):
    return {k: round(float(sub[k].mean()), 5)
            for k in ("has_turn", "junction", "has_stop", "has_brake")} | {
        "net_head_gt45": round(float((sub["net_head"] > 45).mean()), 5),
        "net_head_gt90": round(float((sub["net_head"] > 90).mean()), 5)}


# ---- parity, measured, from its own epcache --------------------------------
pc = parity["man_v1_counts"]
ptot = sum(pc.values())
parity_man = {n: round(pc[k] / ptot, 5) for n, k in zip(NAMES, MAN)}
pspd = parity["speed_steps"]
pst = sum(pspd.values())
parity_spd = {"stopped": round(pspd["stopped"] / pst, 5),
              "city": round(pspd["city"] / pst, 5),
              "highway": round(pspd["hw"] / pst, 5)}

R = {
    "corpus_key": "physicalai-v2bal-4b7eeeac222d",
    "generated_utc": pd.Timestamp.utcnow().isoformat(),
    "evidence_class": "MEASURED (this scan; artifacts listed in the manifest)",
    "labeler": {"module": "stack/scripts/refb_labels.py (repo md5 "
                          "6632348b4c90f6d88ae106d1a1f7a275)",
                "function": "maneuver_labels (v1 kinematic)",
                **shards["pod1"]["labeler_thresholds"]},

    # ---------------- P1 INTEGRITY -----------------------------------------
    "P1_integrity": {
        "clips_scanned": int(len(df)),
        "clips_loadable_ok": int(len(ok)),
        "clips_bad": int(len(df) - len(ok)),
        "bad_clip_ids": (shards["pod1"]["bad_clips"] + shards["pod3"]["bad_clips"]),
        "jpeg_frames_decoded": int(ok["n_decoded"].sum()),
        "jpeg_frames_failed": int(ok["n_bad_frames"].sum()) if "n_bad_frames" in ok else 0,
        "total_raw_frames": int(ok["n_raw"].sum()),
        "total_stacked_frames": int(ok["T_out"].sum()),
        "total_label_steps": int(ok["nlab"].sum()),
        "hours_stacked": round(float(ok["T_out"].sum()) / 10 / 3600, 3),
        "hours_raw": round(float(ok["n_raw"].sum()) / 10 / 3600, 3),
        "bytes_total_gb": round(float(ok["bytes"].sum()) / 1024 ** 3, 2),
        "mb_per_clip_mean": round(float(ok["bytes"].mean()) / 1e6, 3),
        "nonfinite_pose_values": int(ok["n_nan_poses"].sum()),
        "nonfinite_action_values": int(ok["n_nan_actions"].sum()),
        "near_constant_frames_sampled": int(ok["n_const_frames_sampled"].sum()),
        "image_sizes": sorted(set(int(x) for x in ok["image_size"].unique())),
        "n_stacks": sorted(set(int(x) for x in ok["n_stack"].unique())),
        "jpeg_qualities": sorted(set(int(x) for x in ok["quality"].unique())),
        "T_out_min": int(ok["T_out"].min()), "T_out_max": int(ok["T_out"].max()),
        "T_out_mean": round(float(ok["T_out"].mean()), 3),
        "T_out_lt_190": int((ok["T_out"] < 190).sum()),
        "clips_with_zero_motion": int((ok["dist_m"] < 20).sum()),
        "scan_seconds": {t: shards[t]["scan_seconds"] for t in shards},
    },

    # ---------------- P2 DISTRIBUTION --------------------------------------
    "P2_distribution": {
        "achieved_v2_maneuver_v1labeler": man_frac(ok),
        "achieved_v2_turns_LR": round(
            float(ok[["tl", "tr"]].sum().sum()) / float(ok[MAN].sum().sum()), 5),
        "target_maneuver": TARGET,
        "target_turns_LR": 0.28,
        "selection_projected_maneuver": p3["selection_projected_maneuver"],
        "parity_measured_maneuver": parity_man,
        "parity_turns_LR": round(parity_man["turn_left"] + parity_man["turn_right"], 5),
        "achieved_v2_maneuver_v2labeler": man_frac(ok, MAN2),
        "parity_maneuver_v2labeler": {
            n: round(parity["man_v2_counts"][k] / sum(parity["man_v2_counts"].values()), 5)
            for n, k in zip(NAMES, MAN2)},
        "achieved_v2_speed_stepweighted": spd_frac(ok),
        "target_speed": TSPD,
        "selection_projected_speed": p3["selection_projected_speed_stepweighted"],
        "parity_measured_speed": parity_spd,
        "achieved_v2_presence": presence(ok),
        "selection_projected_presence": {
            **p3["selection_projected_presence"],
            "net_head_gt45": p3["selection_projected_net_head_gt45"],
            "net_head_gt90": p3["selection_projected_net_head_gt90"]},
        "parity_measured_presence": {**parity["presence"],
                                     "net_head_gt45": parity["net_head_gt45"],
                                     "net_head_gt90": parity["net_head_gt90"]},
        "parity_hours": parity["hours_stacked"],
        "parity_episodes": parity["ok"],
        "per_shard_maneuver": {t: man_frac(ok[ok.shard == t]) for t in shards},
        "per_shard_turns_LR": {
            t: round(float(ok[ok.shard == t][["tl", "tr"]].sum().sum())
                     / float(ok[ok.shard == t][MAN].sum().sum()), 5) for t in shards},
    },

    # ---------------- P3 CONSOLIDATION -------------------------------------
    "P3_consolidation": {
        "pod1_built": p3["pod1_built"], "pod3_built": p3["pod3_built"],
        "union_built": p3["union_built"],
        "clip_id_overlap": p3["overlap_pod1_pod3"],
        "selected_missing_from_both": p3["n_selected_missing"],
        "built_not_in_selection": p3["n_built_not_selected"],
        "corpus_key_recomputed_from_built_union": p3["corpus_key_from_built_union"],
        "corpus_key_expected": p3["corpus_key_expected"],
        "corpus_key_match": p3["corpus_key_from_built_union"] == p3["corpus_key_expected"],
        "chunks_spanning_both_pods": p3["pod_by_chunk_split_chunks"],
        "double_built_with_differing_bytes": 0,
        "double_built_note": "zero clip-id overlap between shards, so no clip can "
                             "be double-built; within a shard the filename IS the "
                             "clip_id, so duplicates are impossible.",
        "episode_id_unique_over_corpus": int(ok["episode_id"].nunique()),
        "episode_id_collisions": int(len(ok) - ok["episode_id"].nunique()),
        "episode_id_max_multiplicity": int(ok["episode_id"].value_counts().max()),
        "parity_episode_id_unique": parity["episode_id_unique"],
        "parity_episode_id_collisions": parity["ok"] - parity["episode_id_unique"],
    },

    # ---------------- P4 TRAIN READINESS -----------------------------------
    "P4_train_readiness": {
        "shard_smoked": "pod1 (4,953 clips) -- the two shards are on separate "
                        "single-attach volumes, so no single node can mount all "
                        "9,000 until consolidation",
        "builder_vs_loader_equivalence": {
            "clips_checked": p4b["A_builder_vs_loader"]["checked"],
            "frames_byte_identical": p4b["A_builder_vs_loader"]["frames_identical"],
            "poses_identical": p4b["A_builder_vs_loader"]["poses_identical"],
            "actions_identical": p4b["A_builder_vs_loader"]["actions_identical"],
            "mismatches": p4b["A_builder_vs_loader"]["mismatch"]},
        "providers_built": p4a["B_providers"]["n_providers"],
        "manifest_build_seconds_cold": p4a["B_providers"]["build_seconds"],
        "manifest_build_seconds_warm": p4b["manifest_seconds_warm"],
        "manifest_rss_gb": p4a["B_providers"]["rss_after_manifest_gb"],
        "frames_shape": p4a["B_providers"]["frames_shape_0"],
        "window": p4a["B_providers"]["window"],
        "max_horizon": p4a["B_providers"]["max_horizon"],
        "n_windows_pod1_shard": p4a["B_providers"]["n_windows"],
        "n_windows_full_corpus_est": None,
        "window_contract_keys": p4a["C_window_contract"]["keys"],
        "window_contract_n_keys": p4a["C_window_contract"]["n_keys"],
        "window_contract_spec": p4a["C_window_contract"]["spec"],
        "batch_shapes": p4a["D_throughput"]["batch_shapes"],
        "throughput": p4b["D_scaling"],
        "float_frames_mb_per_window": p4b["window_bytes_est_mb"],
    },
}

# ---- projection-vs-built, PER CLIP: does the egomotion-only selection proxy
# ---- predict what the camera-timestamped build actually contains?
SEL = (r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\TanitAD Research Hub"
       r"\Data Engineering\Implementation\incoming\2026-07-24-v2-corpus-50h-balanced"
       r"\r0_selection_v2.parquet")
sel = pd.read_parquet(SEL)
sel["clip_id"] = sel["clip_id"].astype(str)
m = ok.merge(sel, on="clip_id", suffixes=("_built", "_proj"))
tb = (m["tl_built"] + m["tr_built"]) / m["nlab_built"]
tp = (m["tl_proj"] + m["tr_proj"]) / m["nlab_proj"]
R["P2_distribution"]["projection_vs_built_perclip"] = {
    "clips_matched": int(len(m)),
    "turnfrac_mean_built": round(float(tb.mean()), 5),
    "turnfrac_mean_projected": round(float(tp.mean()), 5),
    "turnfrac_mean_abs_diff": round(float((tb - tp).abs().mean()), 5),
    "turnfrac_pearson_r": round(float(np.corrcoef(tb, tp)[0, 1]), 4),
    "T_out_equal_projected": int((m["T_out"] == m["nlab_proj"] + 20).sum()),
    "T_out_mean_built": round(float(m["T_out"].mean()), 3),
    "T_out_mean_projected": round(float((m["nlab_proj"] + 20).mean()), 3),
    "meanv_mean_abs_diff": round(float((m["mean_v_built"] - m["mean_v_proj"]).abs().mean()), 4),
    "junction_agreement": round(float((m["junction_built"] == m["junction_proj"]).mean()), 5),
    "has_turn_agreement": round(float((m["has_turn_built"] == m["has_turn_proj"]).mean()), 5),
}

# full-corpus window estimate: windows = sum(T_out - (window + max_horizon) + 1)
W, MH = R["P4_train_readiness"]["window"], R["P4_train_readiness"]["max_horizon"]
R["P4_train_readiness"]["n_windows_full_corpus_est"] = int(
    (ok["T_out"] - (W + MH) + 1).clip(lower=0).sum())

# ---- cross-build validation + outliers -------------------------------------
xb = jload("crossbuild_poses3.json")
R["P1_integrity"]["crossbuild_vs_parity_epcache"] = {
    "method": "clips present in BOTH the trusted 13.13 h parity epcache and the "
              "v2 cache, keyed on the full clip_id resolved via the stored "
              "episode_id's UNIQUE 4-char prefix among the 3,000 discovered "
              "parity clips (the ep_NNNNN file index is NOT the discovery index)",
    "clips_compared": xb["compared"], "T_equal": xb["T_equal"],
    "poses_bit_identical": xb["poses_bit_identical"],
    "maneuver_histogram_identical": xb["maneuver_hist_identical"],
    "max_abs_dxy_m": xb["max_abs_dxy"], "max_abs_dyaw_rad": xb["max_abs_dyaw"],
    "max_abs_dv_ms": xb["max_abs_dv"],
    "note": "pod1 shard only (the parity epcache lives on pod1)"}
R["P1_integrity"]["outliers"] = jload("outliers.json")
R["pod_code_state"] = {
    "evidence": "MEASURED 2026-07-25 by md5sum / grep -c on both pods",
    "git_head_both_pods": "0f93b98 (dirty)",
    "train_flagship4b_has_v2_cache_flag": {"tanitad-pod": False, "tanitad-pod3": False},
    "tanitad_data_v2_dataset_py_present": {"tanitad-pod": False, "tanitad-pod3": False},
    "refb_labels_lines": {"repo": 1300, "tanitad-pod": 474, "tanitad-pod3": 745},
    "refb_labels_md5": {"repo": "6632348b4c90f6d88ae106d1a1f7a275",
                        "tanitad-pod": "43197feda8fc5c12474a293d8ae0cc67",
                        "tanitad-pod3": "80b490b0e60f2040d593e87283f262a5"},
    "drifted_vs_repo_pod1": ["scripts/train_flagship4b.py", "scripts/refb_labels.py",
                             "tanitad/data/mixing.py", "tanitad/data/toy_driving.py"],
    "matching_repo_pod1": ["scripts/refb_train.py", "tanitad/data/comma2k19.py"],
    "consequence": "a --v2-cache launch on either pod's current checkout fails "
                   "(unrecognized flag + missing module); the QA used a "
                   "repo-synced SHADOW stack at /workspace/tmp/qa_stack"}
R["corpus_write_audit"] = {
    "v2ep_pt_files_modified": 0,
    "probe": "find <cache> -name *.v2ep.pt -newermt 2026-07-24T22:50 -> 0",
    "files_created_by_this_QA": ["_v2manifest.pt (26.6 MB, pod1) -- the loader's "
                                 "own index sidecar, written by build_v2_providers; "
                                 "additive, auto-rebuilt if the clip set changes"]}

json.dump(R, open(OUT, "w"), indent=2)
print(json.dumps({k: v for k, v in R.items()
                  if k not in ("P4_train_readiness",)}, indent=2)[:9000])
print("WROTE", OUT)
