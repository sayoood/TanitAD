#!/usr/bin/env python3
"""p1pb_admissibility.py -- THEORY_CRAWL_AMEND_2 section 2: the admissibility of the RA-1 holdout set, checked BEFORE any score.
No window of the holdout set is scored, no feature is read, no model is run here.  CPU only.   python -I p1pb_admissibility.py"""
import hashlib
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np

ART = "E:/Projects/TanitAD-artifacts"
MC = f"{ART}/dagland/motion_c"
OUT = f"{MC}/theory/wsp_read/P1PB_ADMISSIBILITY.json"


def sha12(c):
    return hashlib.sha256(c.encode()).hexdigest()[:12]


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


res = {"checks": {}}
ck = res["checks"]
# --------------------------------------------------------------- A1 the holdout list: the RA-1 selection holdout, clips 540:600 of the sha12 order of the 4,572
man = [json.loads(ln) for ln in open(f"{MC}/t2_scenes/raw/box2_cache_full_train_manifest.jsonl", encoding="utf-8") if ln.strip()]
clips_all = [m["clip"] for m in man]
assert all(sha12(m["clip"]) == m["sha12"] for m in man)
ho = sorted(clips_all, key=sha12)[540:600]
ho_sha = hashlib.sha256("\n".join(ho).encode()).hexdigest()
train_list = json.load(open(f"{MC}/t2_scenes/raw/TRAIN_CLIPS_4512.json"))["train_clips_4512"]
tr_sha = hashlib.sha256("\n".join(sorted(train_list, key=sha12)).encode()).hexdigest()
REC_HO = "f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9"
REC_TR = "37f4f9ee468e0872134074a82d63c2915458596fdbcf3d0fcf5b09f1fe0d8cc1"
a1 = {"source": "PREREG_V8F_MINI.md Amendment 4 (clips 540:600 of the sha12 order of the 4,572 cache_full TRAIN clips; flag --exclude 540:600; v8f_mini_train.py lines 2383-2387) applied to box 2's "
                "cache_full train manifest (copy: t2_scenes/raw/box2_cache_full_train_manifest.jsonl, 4,572 rows)",
      "n_manifest": len(clips_all), "n_unique": len(set(clips_all)), "n_holdout": len(ho), "holdout_sha256": ho_sha, "recorded_holdout_sha256": REC_HO, "sha_equal": ho_sha == REC_HO,
      "train_list_sha256": tr_sha, "recorded_train_sha256": REC_TR, "train_sha_equal": tr_sha == REC_TR,
      "train_file_equals_manifest_minus_holdout": set(train_list) == set(clips_all) - set(ho), "holdout_train_overlap": len(set(ho) & set(train_list))}
a1["pass"] = bool(ho_sha == REC_HO and tr_sha == REC_TR and len(ho) == 60 and a1["holdout_train_overlap"] == 0 and a1["train_file_equals_manifest_minus_holdout"])
ck["A1_holdout_list"] = a1
json.dump(ho, open(f"{MC}/theory/wsp_read/P1PB_HOLDOUT_60.json", "w"), indent=0)

# --------------------------------------------------------------- A2 C0's training clip list (and its ancestors)
runs = {}
for nm, p in (("C0_s0", f"{ART}/v8f-recover/M2b_s0_recft_C0"), ("C0_s1", f"{ART}/v8f-recover/M2b_s1_recft_C0"), ("dagoacc_s0", f"{ART}/v8f-dagoacc/M2b_s0_dagoacc"),
              ("dagoacc_s1", f"{ART}/v8f-dagoacc/M2b_s1_dagoacc"), ("M2b_s0", f"{ART}/sfresh_inputs/inputs/M2b_s0"), ("M2b_s1", f"{ART}/sfresh_inputs/inputs/M2b_s1")):
    s = json.load(open(f"{p}/summary.json"))
    a = json.load(open(f"{p}/argv.json"))["argv"]
    r = {"summary": f"{p}/summary.json", "n_clips": s.get("n_clips"), "exclude": s.get("exclude"), "excluded_sha256": s.get("excluded_sha256"), "train_sha256": s.get("train_sha256"),
         "train_window_list_sha256": s.get("train_window_list_sha256"), "argv_exclude": a[a.index("--exclude") + 1] if "--exclude" in a else None,
         "init_ckpt_argv": a[a.index("--init-ckpt") + 1] if "--init-ckpt" in a else None}
    r["pass"] = bool(s.get("n_clips") == 4512 and s.get("exclude") == "540:600" and s.get("excluded_sha256") == REC_HO and s.get("train_sha256") == REC_TR and r["argv_exclude"] == "540:600")
    runs[nm] = r
ck["A2_C0_training_clip_list"] = {"runs": runs, "chain": "M2b (from scratch) -> dagoacc (init M2b) -> recft_C0 (init dagoacc); every stage trained with --exclude 540:600",
                                  "note": "ckpt files read by P1' and here: v8f-recover/M2b_s{0,1}_recft_C0/ckpt_last.pt (the same files as the T1-kin dumps' ckpt field)",
                                  "pass": bool(all(r["pass"] for r in runs.values()))}

# --------------------------------------------------------------- A3 the go/stay probe's training windows
W = np.load(f"{ART}/dagland/trigger_probe/raw/windows_train.npz", allow_pickle=False)
wclips = [str(c) for c in W["clips"]]
ci, tt = W["clip_i"], W["t"]
dig = hashlib.sha256("\n".join(f"{wclips[int(c)]}:{int(t)}" for c, t in zip(ci, tt)).encode("utf-8")).hexdigest()
WL = "e144adcd9b6273f2b44ad7dfa162931b90e159f7b3911e0893b25777b2ae8168"
hos = set(ho)
used = {}
for nm in ("pmove", "pstop"):
    idx = np.load(f"{ART}/dagland/trigger_probe/raw/sample_{nm}.npz")["idx"]
    cl_used = np.unique(ci[idx])
    used[nm] = {"n_windows": int(len(idx)), "n_clips": int(len(cl_used)), "holdout_clips_among": int(sum(wclips[int(c)] in hos for c in cl_used))}
a3 = {"windows_train_npz_n_windows": int(len(ci)), "windows_train_n_clips": len(wclips), "windows_train_holdout_clips": int(len(set(wclips) & hos)),
      "windows_train_clip_set_equals_train_4512": set(wclips) == set(train_list), "window_list_digest": dig, "C0_train_window_list_sha256": WL, "digest_equal_to_C0_window_list": dig == WL,
      "sampled_rows_used_by_the_probe": used, "windows_train_episode_sha12_hits_on_holdout": int(len(set(W["episode_sha12"].tolist()) & {sha12(c) for c in ho})),
      "probe_models_artifact": "wsp_read/P1P_PROBE_MODELS.npz (refit on the sample_pmove.npz rows, P1P_REFIT_R1.json)"}
a3["pass"] = bool(dig == WL and a3["windows_train_holdout_clips"] == 0 and all(v["holdout_clips_among"] == 0 for v in used.values())
                  and a3["windows_train_episode_sha12_hits_on_holdout"] == 0 and a3["windows_train_clip_set_equals_train_4512"])
ck["A3_probe_training_windows"] = a3

# --------------------------------------------------------------- A4 disjoint from eval-141 (the pooled read is a union of two disjoint clip sets)
ev = np.load(f"{ART}/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/join_eval141.npz", allow_pickle=False)
evc = [str(c) for c in ev["clip"]]
ck["A4_disjoint_from_eval141"] = {"n_eval_clips": len(evc), "overlap": int(len(set(evc) & hos)), "pass": bool(len(set(evc) & hos) == 0)}

# --------------------------------------------------------------- A5 the frozen features of the holdout clips on the dev box
CACHE = "F:/TanitAD-scratch/speedread/cache/train"
mfB = {m["clip"]: m for m in man}
bad = []
for c in ho:
    p = f"{CACHE}/clips/{c}.pt"
    if not os.path.exists(p) or os.path.getsize(p) != mfB[c]["bytes"] or md5_file(p) != mfB[c]["md5"]:
        bad.append(c)
route_keys = ("dtype", "enc_dtype", "device", "source", "mean_dtype", "grids")
evm = [json.loads(ln) for ln in open(f"{ART}/sfresh_inputs/box2/eval_manifest.jsonl", encoding="utf-8") if ln.strip()]
route_ev = sorted({tuple(str(r.get(k)) for k in route_keys) for r in evm})
route_ho = sorted({tuple(str(mfB[c].get(k)) for k in route_keys) for c in ho})
ck["A5_features_on_dev_box"] = {"cache_dir": f"{CACHE}/clips (the cache the probe's own V_pmove was extracted from: trigger_probe/extract_v.py CACHE)",
                                "n_holdout_files_present_and_md5_equal_to_box2_manifest": len(ho) - len(bad), "bad": bad, "route_keys": list(route_keys),
                                "eval141_route": [list(x) for x in route_ev], "holdout_route": [list(x) for x in route_ho], "same_route_as_eval141": route_ev == route_ho,
                                "n_frames_total": int(sum(mfB[c]["n_frames"] for c in ho))}
ck["A5_features_on_dev_box"]["pass"] = bool(not bad and route_ev == route_ho)

res["holdout_clips_file"] = f"{MC}/theory/wsp_read/P1PB_HOLDOUT_60.json"
res["holdout_sha256"] = ho_sha
res["admissible"] = bool(all(v["pass"] for v in ck.values()))
res["note"] = ("Evidence class MEASURED (ours, dev-box CPU, files named above; sha and md5 recomputed here). No window of the holdout set was scored, no feature read for scoring, "
               "no model run before this file was written.")
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({k: v.get("pass") for k, v in ck.items()}), "admissible:", res["admissible"])
