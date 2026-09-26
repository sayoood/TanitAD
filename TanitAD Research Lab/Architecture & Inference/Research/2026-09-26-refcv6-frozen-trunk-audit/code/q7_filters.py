"""Q7 (advisory class G) -- every upstream FILTER on refcv6's data path: what it removes, and
whether the removal CORRELATES with anything.

"A uniform 25 % loss is a smaller problem than a biased 25 % loss, and only the correlation test
distinguishes them." The attributes tested are the v8 label release's own per-clip STRATA
(country, day/night, road class, stop-and-go), the ego-future SPEED BAND, the nav command and the
tactical lat/lon tokens -- none of them is an input to any filter below, so a difference is a bias
and not a tautology.

CLIP-LEVEL (the train VIEW, `_VIEW_RECORD.json` read-only from Thor, md5 e1eabe49...):
  cache 4,713 -> labels_v8_train 4,572 -> agent_join 4,427 -> map_gt 4,369
  removed: not_v8_train 141 | no_agent_join 145 | no_validated_map 58
EVAL: 147 v8 eval records -> 139 eval-view clips.

WINDOW-LEVEL (within kept clips; every count from the trainer's own classes or the run's stamp):
  enumeration range(T - window - max_horizon) (refb_train.py:116), the 6 s future MASK, the
  tactical band, the 2-D agent join's per-frame coverage (eval split: real JoinFileReader;
  train split: from the run's stamp), the box-3-D visible filter and the map lift-valid mask
  (from the live metrics.jsonl, every logged batch).

Statistics: for a categorical attribute, a permutation test of the total-variation distance
between the removed and the kept distributions (10,000 permutations, seed 0); for a numeric one,
a permutation test of the difference in means. n is printed beside every p.
"""
from __future__ import annotations

import gzip
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

C.bootstrap()
import numpy as np  # noqa: E402

SCRATCH = C.SCRATCH
VIEW = SCRATCH / "refcv6-b1-416x1024-train___VIEW_RECORD.json"
TRAIN_LAB = Path("C:/Users/Admin/tanitad-wt/_s2build/release/v8/s2_labels_v8_train.jsonl.gz")
EVAL_LAB = C.KIT / "data/v8labels/labels/s2_labels_v8_eval.jsonl.gz"
METRICS = SCRATCH / "refcv6-r101-s0_metrics.jsonl"
N_PERM = 10000


def attrs(r: dict) -> dict:
    s = r.get("strata") or {}
    g = ((r.get("g_tac") or {}).get("goals") or {}).get("SPEED_BAND") or {}
    return {"country": s.get("country"), "daynight": s.get("daynight_clock"),
            "road_class": s.get("road_class"),
            "stop_launch": s.get("has_stop_launch_20s"),
            "stopped_frac": s.get("stopped_frame_frac_20s"),
            "v_hi_ms": g.get("v_hi_ms"),
            "nav": (r.get("nav_command") or {}).get("token"),
            "tac_lat": (r.get("a_tac") or {}).get("lat"),
            "tac_lon": (r.get("a_tac") or {}).get("lon")}


def perm_test(removed: list, kept: list, numeric: bool, rng) -> dict:
    """Permutation test, VECTORISED (integer-coded categories; numpy)."""
    removed = [x for x in removed if x is not None and not (isinstance(x, float) and math.isnan(x))]
    kept = [x for x in kept if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if len(removed) < 3 or len(kept) < 3:
        return {"n_removed": len(removed), "n_kept": len(kept), "skipped": "n < 3"}
    nr, nk = len(removed), len(kept)
    n = nr + nk
    if numeric:
        pool = np.asarray(removed + kept, dtype=np.float64)
        obs = abs(pool[:nr].mean() - pool[nr:].mean())
        tot = pool.sum()
        ge = 0
        for _ in range(N_PERM // 1000):
            idx = np.argsort(rng.random((1000, n)), axis=1)[:, :nr]
            sr = pool[idx].sum(axis=1)
            st = np.abs(sr / nr - (tot - sr) / nk)
            ge += int((st >= obs - 1e-12).sum())
        out = {"stat": "abs mean diff", "mean_removed": round(float(pool[:nr].mean()), 4),
               "mean_kept": round(float(pool[nr:].mean()), 4)}
    else:
        cats = sorted(set(map(str, removed + kept)))
        code = {c: i for i, c in enumerate(cats)}
        pool = np.asarray([code[str(x)] for x in removed + kept], dtype=np.int64)
        K = len(cats)
        tot = np.bincount(pool, minlength=K).astype(np.float64)

        def stat_rows(sel):             # sel [P, nr] indices into pool
            P = sel.shape[0]
            cr = np.zeros((P, K))
            np.add.at(cr, (np.repeat(np.arange(P), nr), pool[sel].ravel()), 1.0)
            ck = tot[None, :] - cr
            return 0.5 * np.abs(cr / nr - ck / nk).sum(axis=1)
        obs = float(stat_rows(np.arange(nr)[None, :])[0])
        ge = 0
        for _ in range(N_PERM // 1000):
            idx = np.argsort(rng.random((1000, n)), axis=1)[:, :nr]
            ge += int((stat_rows(idx) >= obs - 1e-12).sum())
        cr = np.bincount(pool[:nr], minlength=K) / nr
        ck = np.bincount(pool[nr:], minlength=K) / nk
        order = np.argsort(-np.abs(cr - ck))[:4]
        out = {"stat": "TVD", "largest_share_gaps_(removed,kept)":
               {cats[i]: [round(float(cr[i]), 4), round(float(ck[i]), 4)] for i in order}}
    out.update({"n_removed": nr, "n_kept": nk, "observed": round(float(obs), 5),
                "p_perm": round((ge + 1) / (N_PERM + 1), 5), "n_perm": N_PERM})
    return out


def clip_level() -> dict:
    view = json.loads(VIEW.read_text(encoding="utf-8"))
    labs = [json.loads(x) for x in gzip.open(TRAIN_LAB, "rt", encoding="utf-8")]
    by = {C.sha12(r["clip_id"]): attrs(r) for r in labs}
    rem = view["removed_sha12"]
    removed_all = set(rem.get("no_agent_join", [])) | set(rem.get("no_validated_map", []))
    kept = [s for s in by if s not in removed_all]
    rng = np.random.default_rng(0)
    res = {"stages": view["stages"], "removed_counts": view["removed"],
           "n_train_labels": len(by), "n_kept_with_label": len(kept),
           "removed_lists_in_labels": {k: sum(1 for s in v if s in by) for k, v in rem.items()},
           "tests": {}}
    for fname, lst in (("no_agent_join", rem.get("no_agent_join", [])),
                       ("no_validated_map", rem.get("no_validated_map", [])),
                       ("both_filters", sorted(removed_all))):
        rr = [by[s] for s in lst if s in by]
        kk = [by[s] for s in kept]
        res["tests"][fname] = {a: perm_test([x[a] for x in rr], [x[a] for x in kk],
                                            a in ("stopped_frac", "v_hi_ms"), rng)
                               for a in ("country", "daynight", "road_class", "stop_launch",
                                         "stopped_frac", "v_hi_ms", "nav", "tac_lat", "tac_lon")}
    return res


def eval_clip_level() -> dict:
    import torch
    labs = [json.loads(x) for x in gzip.open(EVAL_LAB, "rt", encoding="utf-8")]
    m = torch.load(str(C.KIT / "data/refcv6-b1-416x1024-eval139/_v2manifest.pt"),
                   map_location="cpu", weights_only=False)
    in_view = {C.sha12(c) for c in m["clip_id"]}
    by = {C.sha12(r["clip_id"]): attrs(r) for r in labs}
    rem = [s for s in by if s not in in_view]
    kept = [s for s in by if s in in_view]
    rng = np.random.default_rng(1)
    return {"n_eval_labels": len(by), "n_eval_view": len(in_view), "n_removed": len(rem),
            "removed_sha12": sorted(rem),
            "tests": {a: perm_test([by[s][a] for s in rem], [by[s][a] for s in kept],
                                   a in ("stopped_frac", "v_hi_ms"), rng)
                      for a in ("country", "daynight", "road_class", "v_hi_ms", "nav")}}


def window_level(cfg: dict) -> dict:
    import torch
    m = torch.load(str(SCRATCH / "refcv6-b1-416x1024-train___v2manifest.pt"), map_location="cpu",
                   weights_only=False)
    W, MAXH, H6 = 8, 20, 60
    T = np.array([int(x) for x in m["T_out"]])
    n_win = np.maximum(T - W - MAXH, 0)
    # a window's 6 s future is MASKED where NOW + k > T - 1 (refc_v3_train.py:2979-2981)
    now_last = T - MAXH - 2
    masked = np.array([sum(1 for r in range(W - 1, nl + 1) if r + H6 > t - 1)
                       for t, nl in zip(T, now_last)])
    rows = [json.loads(x) for x in METRICS.read_text(encoding="utf-8").splitlines() if x.strip()]
    tr = [r for r in rows if "loss" in r and "box3d_n_target_prefilter" in r]
    pre = sum(r["box3d_n_target_prefilter"] for r in tr)
    vis = sum(r["box3d_n_target_visible"] for r in tr)
    seen = sum(r.get("n_map_cells_seen", 0) for r in tr)
    unobs = sum(r.get("n_map_cells_unobserved", 0) for r in tr)
    ags = cfg["agent_join_stats"]
    return {
        "enumeration": {"rule": "range(T - window - max_horizon), window 8, max_horizon 20",
                        "n_windows": int(n_win.sum()), "config_n_windows": ags["train"]["n_windows"],
                        "rows_per_clip": int(np.median(T)),
                        "now_rows_never_used_per_clip": int(np.median(T - (n_win))),
                        "frac_rows_never_a_now": float(1 - n_win.sum() / T.sum())},
        "future_mask_6s": {"windows_with_masked_future_slots": int(masked.sum()),
                           "frac": float(masked.sum() / n_win.sum()),
                           "why": "enumeration keeps max_horizon 20 rows for REF-B parity; the 60-row "
                                  "(6 s) future is fetched per item and MASKED past the clip end"},
        "tactical_band": {"windows_supervised_trainer_clock": 179129,
                          "frac": 179129 / int(n_win.sum()),
                          "source": "raw/q4d_label_offset_true_clock.json (the trainer's own v7_labels)"},
        "agent_join_frames": {"train": {"n_windows": ags["train"]["n_windows"],
                                        "n_labelled": ags["train"]["n_windows_labelled"],
                                        "frac_unlabelled": 1 - ags["train"]["frac_windows_labelled"]},
                              "eval": {"n_windows": ags["eval"]["n_windows"],
                                       "n_labelled": ags["eval"]["n_windows_labelled"],
                                       "frac_unlabelled": 1 - ags["eval"]["frac_windows_labelled"]}},
        "box3d_visible_filter": {"n_logged_batches": len(tr), "targets_prefilter": int(pre),
                                 "targets_visible": int(vis),
                                 "frac_removed": float(1 - vis / pre) if pre else None},
        "map_lift_valid_mask": {"cells_seen": int(seen), "cells_unobserved_excluded": int(unobs),
                                "frac_excluded": float(unobs / seen) if seen else None},
        "max_speed_clamp": cfg["refcv6_max_speed"]["train"]["n_over_ceiling"],
        "agent_class_out_of_vocab_boxes": cfg["seams"]["agent_cls_weight"]["out_of_vocabulary"],
    }


def main():
    C.ram_guard("q7_filters (light job; the brief's 8 GB floor applies to every job)")
    cfg = C.load_config()
    out = {"what": "Q7: upstream filters on refcv6's data path -- counts and correlation",
           "evidence_class": "MEASURED (ours) over the run's own view record, labels and log",
           "clip_level_train": clip_level(), "clip_level_eval": eval_clip_level(),
           "window_level": window_level(cfg)}
    print(json.dumps({k: v for k, v in out["clip_level_train"].items() if k != "tests"}, indent=1))
    for f, tests in out["clip_level_train"]["tests"].items():
        for a, r in tests.items():
            print(f, a, r)
    print(json.dumps(out["clip_level_eval"], indent=1)[:3000])
    print(json.dumps(out["window_level"], indent=1))
    C.write_json("q7_filters.json", out)


if __name__ == "__main__":
    main()
