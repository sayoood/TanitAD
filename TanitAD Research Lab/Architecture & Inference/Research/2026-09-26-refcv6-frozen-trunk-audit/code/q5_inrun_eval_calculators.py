"""Q5 (advisory class E) -- what refcv6's IN-RUN EVAL writes, how it aggregates, what state it
carries. Evidence: the live run's own metrics.jsonl (read-only scp from Thor, 4,030 rows) and the
trainer's own dataset class over the eval-139 cache (no model, no GPU).

THE AGGREGATOR (refc_v3_train.py:8269-8300): for 8 fixed batches of 16 windows,
    acc[k] += float(v) for every 0-dim tensor / python scalar v that compute_losses_v3 returns
    eval_<k> = acc[k] / nb_e            (nb_e = number of batches, ALWAYS 8)
so (i) a key a batch does NOT emit is divided by 8 anyway, (ii) a COUNTED ZERO (`sum / max(n,1)`
with n = 0) enters as a perfect 0.0, and (iii) every batch weighs 1/8 whatever its n. None of
these is wrong by itself; each is only admissible if you know how many of the 8 fixed batches
carried a zero-n. This instrument COUNTS that, per label-derived term, on the exact subset:
    perm = randperm(len(e_ds), Generator().manual_seed(12345))[:128]   (refc_v3_train.py:7561-7563)
It also reads the stateful parts off the live log.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

C.bootstrap()
import torch  # noqa: E402

SCRATCH = C.SCRATCH
METRICS = SCRATCH / "refcv6-r101-s0_metrics.jsonl"


def part1_log() -> dict:
    rows = [json.loads(x) for x in METRICS.read_text(encoding="utf-8").splitlines() if x.strip()]
    ev = [r for r in rows if "eval_loss" in r]
    tr = [r for r in rows if "loss" in r]
    err = [r for r in rows if "eval_error" in r]
    cd = [r for r in rows if any(k.startswith("cd_") for k in r) and "loss" not in r]
    nan = [(r["step"], k) for r in ev for k, v in r.items()
           if isinstance(v, float) and (math.isnan(v) or math.isinf(v))]
    tk = set(k for r in tr for k in r)
    ek = set(k[5:] for r in ev for k in r if k.startswith("eval_"))
    # the dedup counters are ACCUMULATED "since the run log last read and reset it"
    # (timm_trunk.py:638-642); an in-run eval runs the trunk between two log rows.
    post = [(r["step"], r.get("trunk_frame_slots"), r.get("trunk_frames_computed"))
            for r in tr if r["step"] % 500 == 50 and r["step"] >= 1000]
    norm = [(r["step"], r.get("trunk_frame_slots"), r.get("trunk_frames_computed"))
            for r in tr if r["step"] % 500 != 50 and r["step"] >= 1000]
    post_vals = sorted(set((a, b) for _, a, b in post))
    norm_vals = sorted(set((a, b) for _, a, b in norm))
    last = max(r["step"] for r in rows)
    return {"n_rows": len(rows), "last_step": last, "n_train_rows": len(tr), "n_eval_rows": len(ev),
            "n_eval_error_rows": len(err), "n_conflict_own_rows": len(cd),
            "eval_nan_or_inf_cells": len(nan),
            "eval_windows_values": sorted(set(r.get("eval_windows") for r in ev)),
            "eval_batches_values": sorted(set(r.get("eval_batches") for r in ev)),
            "n_eval_metric_keys": len(ek),
            "eval_metric_keys": sorted(ek),
            "train_keys_not_in_eval": sorted(tk - ek),
            "eval_keys_not_in_train": sorted(ek - tk),
            "cascade_key_rows_train": sum(1 for r in tr if "cascade" in r),
            "cascade_key_rows_eval": sum(1 for r in ev if "eval_cascade" in r),
            "dedup_counter_rows_after_an_eval": {"n": len(post), "distinct_(slots,frames)": post_vals},
            "dedup_counter_rows_other": {"n": len(norm), "distinct_(slots,frames)": norm_vals}}


def part2_fixed_subset(cfg: dict) -> dict:
    T = C.trainer_module()
    from tanitad.data import v7_labels as v7l
    from tanitad.data.v2_dataset import build_v2_providers, stable_episode_id
    e_eps = build_v2_providers([str(C.KIT / "data/refcv6-b1-416x1024-eval139")], lru_size=2,
                               verbose=False)
    e_ds = T.V3Dataset(e_eps, window=8, max_horizon=20, channels=9)
    labs, man = v7l.load_v7_labels(str(C.KIT / "data/v8labels/labels/s2_labels_v8_eval.jsonl.gz"),
                                   allow_oracle_nav=True)
    by_sid = {stable_episode_id(l.clip_id): l for l in labs}
    n_ds = len(e_ds)
    perm = torch.randperm(n_ds, generator=torch.Generator().manual_seed(12345))[:128].tolist()
    # map GT: the eval split's two clips with no SAM3 file, from the run's own stamp
    no_map = set((cfg["refcv6_perception"]["map_gt_stats"]["eval"].get("reasons") or {}).keys())
    ep_sha = {}
    import hashlib
    for i, ep in enumerate(e_eps):
        ep_sha[i] = None
    # clip id per provider index, from the manifest (same order build_v2_providers used)
    m = torch.load(str(C.KIT / "data/refcv6-b1-416x1024-eval139/_v2manifest.pt"),
                   map_location="cpu", weights_only=False)
    for i in range(len(e_eps)):
        ep_sha[i] = C.sha12(m["clip_id"][i])
        assert int(e_eps[i].episode_id) == stable_episode_id(m["clip_id"][i])
    batches = []
    for b in range(8):
        win = perm[b * 16:(b + 1) * 16]
        n_band = n_map = 0
        eps = set()
        for w in win:
            e_i, t = e_ds.index[w]
            eps.add(e_i)
            lab = by_sid.get(int(e_eps[e_i].episode_id))
            if lab is not None and v7l.tactical_class_ids(lab, (t + 7) * 0.1)[0] != v7l.IGNORE_ID:
                n_band += 1
            if ep_sha[e_i] not in no_map:
                n_map += 1
        batches.append({"batch": b, "n_tactical_in_band": n_band, "n_map_labelled": n_map,
                        "n_distinct_episodes": len(eps)})
    k_band = sum(1 for x in batches if x["n_tactical_in_band"] > 0)
    k_map = sum(1 for x in batches if x["n_map_labelled"] > 0)
    return {"len_e_ds": n_ds, "config_eval_n_windows": cfg["agent_join_stats"]["eval"]["n_windows"],
            "perm_first8_sha": hashlib.sha256(json.dumps(perm[:8]).encode()).hexdigest()[:12],
            "n_windows_in_subset": len(perm),
            "n_distinct_episodes_in_subset": len({e_ds.index[w][0] for w in perm}),
            "per_batch": batches,
            "batches_with_zero_tactical_rows": 8 - k_band,
            "batches_with_zero_map_rows": 8 - k_map,
            "multiplicative_bias_on_eval_lat_lon_ce_terms": k_band / 8.0,
            "note": ("a batch with n = 0 contributes a COUNTED 0.0 (refcv6_tactical.py:803 "
                     "`sum / keep.sum().clamp_min(1)`), and eval_<k> divides by all 8 batches, "
                     "so eval_<k> = (k/8) x (mean over the k supervised batches)")}


def main():
    C.ram_guard("q5_inrun_eval_calculators (light job; the brief's 8 GB floor applies to every job)")
    cfg = C.load_config()
    out = {"what": "Q5: the in-run eval's calculators -- what they write, how they aggregate, "
                   "what state they carry",
           "evidence_class": "MEASURED (live metrics.jsonl scp'd read-only; trainer's V3Dataset)",
           "part1_live_log": part1_log(), "part2_fixed_eval_subset": part2_fixed_subset(cfg)}
    p1, p2 = out["part1_live_log"], out["part2_fixed_eval_subset"]
    print(json.dumps({k: v for k, v in p1.items() if k not in ("eval_metric_keys",)}, indent=1))
    print(json.dumps(p2, indent=1))
    C.write_json("q5_inrun_eval_calculators.json", out)


if __name__ == "__main__":
    main()
