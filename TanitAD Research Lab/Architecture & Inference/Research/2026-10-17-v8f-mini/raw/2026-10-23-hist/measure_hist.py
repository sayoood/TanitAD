"""PREREG_V8F_HIST, "What else moves with the lever" (D1-D4): MEASURED from the code and the banked inputs, CPU only.

  D1  the built parameter delta, from the REAL M2b configs at hist 4 and hist 10 (a CPU build of V8FMini): expected exactly +1,536, all in `adapter`.
  D2  the initialisation: which weights are drawn identically across hist 4 / 10 under one torch seed, and which come from a different point of the RNG stream.
  D3  the held-age clamp near a clip's start (training only) AND the training window set / step budget, from box 2's train manifest (the clip lengths):
        `V8FMiniWindows.item` clamps T's and S's held ages by room = max(t - (hist - 1), 0)                       (v8f_mini_data.py:475-477)
        `train_windows` starts at t = hist - 1                                                                   (v8f_mini_data.py:505-513)
        `build_data` calls it with hist=cfg.hist, and the trainer sizes S = ceil(n_windows_all_clips / batch), total = S x steps-mult   (v8f_mini_train.py:338-339, :405-406)
      The PREREG did not state the last two: at hist 10 the t = 3..8 windows of every clip are NOT trained on, and the step budget falls with them. FINDING F-1.
  D4  the eval padding counts on raw/eval_grid_windows.json: real frames per window at hist 10 = min(t + 1, 10) (V8FMiniWindows.item: ok = idx >= 0).

  python measure_hist.py [--manifest <box2 train manifest.jsonl>] [--out measure_hist.json]
UNVERIFIED is said, per item, where an input is not on this disk."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
PKG = HERE.parents[2]
REPO = HERE.parents[6]
for p in (REPO / "stack", REPO / "taniteval"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
DEFAULT_MANIFEST = "C:/Users/Admin/box2_ship/logs/v8f_grid/manifest_train.jsonl"     # box 2's train manifest, pulled by the GRID chain (4,572 clips)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def clamp_ages(t, a_t, a_s, hist, max_age_t=2, max_age_s=6):
    """The held-age clamp of V8FMiniWindows.item (v8f_mini_data.py:475-477), restated: room = t - (hist - 1), at least 0; a_t <= room, a_t <= max_age_t; a_s nested above a_t."""
    room = max(t - (hist - 1), 0)
    a_t = int(min(a_t, room, max_age_t))
    a_s = int(max(a_t, min(a_s, room, max_age_s)))
    return a_t, a_s


def age_draws(max_age_t=2, max_age_s=6):
    """The sampler's draws (ChunkedSampler.epoch, v8f_mini_data.py:572-573): a_t uniform 0..max_age_t, a_s = a_t + uniform 0..max_age_s - max_age_t. Equiprobable (a_t, a_s) pairs."""
    return [(a, a + k) for a in range(max_age_t + 1) for k in range(max_age_s - max_age_t + 1)]


def d1():
    from tanitad.models import v8f_mini as M
    sys.path.insert(0, str(REPO / "stack" / "scripts"))
    import v8f_mini_train as TR
    out = {"build": "CPU build of V8FMini from the trainer's own make_config(argv) with the REAL M2b flags (d 256, kin-mode yaw)"}
    built = {}
    for h in (4, 10):
        a = TR.build_parser().parse_args(["--out", "x", "--arm", "M2b", "--hist", str(h), "--kin-mode", "yaw"])
        cfg = TR.make_config(a)
        model = M.V8FMini(cfg, M.P10Prior.synthetic(0), None, None)
        lv = model.built_levers()
        built[h] = {"hist": cfg.hist, "hmax": cfg.hmax, "d": cfg.d, "n_params_trainable": lv["n_params_trainable"], "params": lv["params"], "time_shape": list(model.time.shape)}
    banked = json.loads((PKG / "raw" / "full" / "runs" / "M2b_s0" / "summary.json").read_text(encoding="utf-8"))["levers"]
    out["built"] = built
    out["banked_m2b_s0"] = {"n_params_trainable": banked["n_params_trainable"], "params": banked["params"]}
    out["hist4_equals_banked"] = bool(built[4]["params"] == banked["params"] and built[4]["n_params_trainable"] == banked["n_params_trainable"])
    delta = {k: built[10]["params"][k] - built[4]["params"][k] for k in built[4]["params"]}
    out["delta_by_group"] = delta
    out["delta_total"] = built[10]["n_params_trainable"] - built[4]["n_params_trainable"]
    out["expected"] = {"total": 6 * 256, "adapter": 6 * 256, "others": 0}
    out["pass"] = bool(out["hist4_equals_banked"] and out["delta_total"] == 1536 and delta["adapter"] == 1536 and all(v == 0 for k, v in delta.items() if k != "adapter")
                       and built[4]["hmax"] == built[10]["hmax"] == 16)
    return out


def d2():
    import torch
    from tanitad.models import v8f_mini as M
    sys.path.insert(0, str(REPO / "stack" / "scripts"))
    import v8f_mini_train as TR
    sd = {}
    for h in (4, 10):
        a = TR.build_parser().parse_args(["--out", "x", "--arm", "M2b", "--hist", str(h), "--kin-mode", "yaw"])
        torch.manual_seed(0)
        sd[h] = M.V8FMini(TR.make_config(a), M.P10Prior.synthetic(0), None, None).state_dict()
    same, differ = [], []
    for k in sd[4]:
        if k == "time":
            continue
        (same if torch.equal(sd[4][k], sd[10][k]) else differ).append(k)
    t4, t10 = sd[4]["time"], sd[10]["time"]
    is_const = lambda x: bool(x.numel() > 0 and (x.float() == x.float().flatten()[0]).all())       # ones / zeros / a zero-init layer: no RNG involved
    same_random = [k for k in same if not is_const(sd[4][k])]
    differ_const = [k for k in differ if is_const(sd[4][k]) and is_const(sd[10][k])]
    import re
    explained = lambda k: bool(k in ("pos", "ck_nom", "ck_scale", "basis") or k.startswith(("prior.", "adapter.")) or re.fullmatch(r"wm\.stack\.predictors\.\d+\.blocks\.\d+\.mod\.1\.bias", k))
    return {"seed": 0, "n_tensors_identical": len(same), "n_tensors_different": len(differ),
            "identical_and_not_constant": same_random, "identical_not_constant_unexplained": [k for k in same_random if not explained(k)],
            "different_but_constant_both": differ_const,
            "time_first_4_rows_identical_to_the_hist4_table": bool(torch.equal(t10[:4], t4)),
            "reading": "an identical tensor is (a) drawn BEFORE `time` (adapter, pos), (b) a buffer (ck_nom, ck_scale, basis), (c) the P10 prior (built from its own synthetic(0) generator before the model), (d) an AdaLN bias set to a structured constant "
                       "(hjepa.py:357-360, gates = 1, no RNG), or (e) a plain constant (LayerNorm ones/zeros, zero-init heads). Every other tensor, i.e. every RANDOM weight built after `time` (conditioning embeddings, "
                       "the readers' queries, attention and MLP weights, the heads, the WM), sits at a point of the RNG stream shifted by the six extra time rows and differs. The hist lever is therefore also an initialisation change, "
                       "like a different init seed. The training-seed floor F spans init variation, so the criterion already prices it; this design cannot separate it from the lever"}


def verify_clamp_restatement():
    """`clamp_ages` above RESTATES V8FMiniWindows.item's two clamp lines; prove it equals the real function on a synthetic world (T = 60, 6 clips), every t from 3 to 29 and every
    sampler draw, at hist 4 and hist 10. Anything but zero mismatches makes the D3 counts below inadmissible."""
    import tempfile
    from tanitad.data import v8f_mini_data as D
    root = Path(tempfile.mkdtemp(prefix="v8fhist_d3_"))                # left behind on purpose (nothing is deleted); it is a few KB of synthetic clips
    w = D.make_synthetic_world(root, n_train=6, n_eval=2, T=60, seed=0)
    J = D.load_join(root / "join_train.npz")
    store = D.ClipStore(root, "train", lru=8, kin_mode="yaw", tracks=D.load_native_tracks(root / "native_tracks.pt"))
    n = bad = 0
    for h in (4, 10):
        ds = D.V8FMiniWindows(store, J, [], hist=h, hmax=16)
        for t in range(3, 30):
            for (a, b) in age_draws():
                it = ds.item(0, t, a, b)
                n += 1
                bad += (it["a_t"], it["a_s"]) != clamp_ages(t, a, b, h)
    return {"world": str(root), "n_checked": n, "n_mismatched": int(bad), "pass": bad == 0 and n == 2 * 27 * 15}


def d3(manifest):
    from tanitad.data import v8f_mini_data as D
    out = {"formulas": {"windows_per_clip": "len(range(hist - 1, T - HORIZON)) with HORIZON 30 (train_windows)", "S": "ceil(n_windows_all_clips / batch)", "steps": "S x steps_mult"}}
    draws = age_draws()
    out["age_draws"] = {"n_equiprobable_pairs": len(draws), "pairs": draws,
                        "source": "ChunkedSampler.epoch (v8f_mini_data.py:572-573): a_t uniform 0..2, a_s = a_t + uniform 0..4"}
    out["clamp_restatement_check"] = verify_clamp_restatement()
    if not out["clamp_restatement_check"]["pass"]:
        raise SystemExit(f"measure_hist: the restated clamp differs from V8FMiniWindows.item: {out['clamp_restatement_check']}")
    per_t = {}
    for t in range(3, 18):
        rows = {}
        for h in (4, 10):
            if t < h - 1:
                rows[f"hist{h}"] = None
                continue
            ch = sum(1 for (a, b) in draws if clamp_ages(t, a, b, h) != (a, b))
            rows[f"hist{h}"] = {"room": max(t - (h - 1), 0), "draws_clamped": ch, "of": len(draws)}
        both = [t >= 9]
        rows["changed_draws"] = None if not both[0] else sum(1 for (a, b) in draws if clamp_ages(t, a, b, 4) != clamp_ages(t, a, b, 10))
        per_t[str(t)] = rows
    out["per_t"] = per_t
    mp = Path(manifest)
    if not mp.exists():
        out["manifest"] = f"UNVERIFIED: {manifest} is not on this disk; the formulas above give the per-clip counts, and the totals need the clip lengths"
        return out
    ents = {}
    with open(mp, encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                d = json.loads(ln)
                ents[d["clip"]] = (d["sha12"], int(d["n_frames"]))
    out["manifest"] = {"path": str(mp), "sha256": sha256(mp), "n_clips": len(ents)}
    order = sorted(ents, key=D.sha12)                                       # the trainer's own order (v8f_mini_train.py:312)
    assert all(ents[c][0] == D.sha12(c) for c in order), "the manifest's sha12 column is not D.sha12(clip)"
    clips = [c for i, c in enumerate(order) if not (540 <= i < 600)]        # --exclude 540:600
    T = np.array([ents[c][1] for c in clips], np.int64)
    out["n_clips_after_exclude"] = int(len(clips))
    out["clip_length"] = {"min": int(T.min()), "max": int(T.max()), "n_101": int((T == 101).sum()), "n_lt_101": int((T < 101).sum()), "n_lt_45": int((T < 45).sum()), "n_lt_40": int((T < 40).sum())}
    H = D.HORIZON
    wins = lambda h: np.maximum(T - H - (h - 1), 0)                         # len(range(h - 1, T - H))
    n4, n10 = int(wins(4).sum()), int(wins(10).sum())
    out["banked_b"] = {"n_windows_all_clips": 306566, "S": 4791, "steps": 19164, "batch": 64, "steps_mult": 4}
    out["b_reproduced"] = {"n_windows": n4, "S": math.ceil(n4 / 64), "steps": 4 * math.ceil(n4 / 64)}
    out["b_reproduced"]["equals_banked"] = bool(n4 == 306566 and out["b_reproduced"]["S"] == 4791 and out["b_reproduced"]["steps"] == 19164)
    S10 = math.ceil(n10 / 64)
    out["h_hist10"] = {"n_windows": n10, "S": S10, "steps": 4 * S10, "windows_lost_vs_b": n4 - n10, "windows_lost_per_clip_t_3_to_8": 6, "n_clips_losing_all_six": int((wins(4) - wins(10) == 6).sum()),
                       "steps_vs_b": {"b": 19164, "h": 4 * S10, "ratio": 4 * S10 / 19164, "short_by": 19164 - 4 * S10},
                       "epoch_fraction": "H sees the same 4 passes over a window set that is 8.8 % smaller (its windows are a subset of B's: t >= 9)"}
    # the clamp, on the windows that exist at hist 10 and on the ones that exist only at hist 4
    t_lo, t_hi = 9, 14
    n_lo = int(sum(max(0, min(t_hi, int(x) - H - 1) - t_lo + 1) for x in T))       # windows with 9 <= t <= 14 and t < T - 30
    per_win_changed = {t: sum(1 for (a, b) in draws if clamp_ages(t, a, b, 4) != clamp_ages(t, a, b, 10)) for t in range(t_lo, t_hi + 1)}
    n_t = {t: int(sum(1 for x in T if t <= int(x) - H - 1)) for t in range(3, 15)}
    exp_draws_changed_per_epoch = sum(n_t[t] * per_win_changed[t] / len(draws) for t in range(t_lo, t_hi + 1))
    only_b = {t: n_t[t] for t in range(3, 9)}
    out["d3_clamp"] = {
        "windows_in_both_whose_sampled_ages_can_clamp_differently (9 <= t <= 14)": n_lo,
        "of_which_per_t": {str(t): n_t[t] for t in range(t_lo, t_hi + 1)},
        "draws_changed_per_t_of_15": {str(t): per_win_changed[t] for t in range(t_lo, t_hi + 1)},
        "expected_changed_draws_per_epoch": exp_draws_changed_per_epoch,
        "share_of_h_windows_per_epoch": exp_draws_changed_per_epoch / n10,
        "windows_only_in_b_t_3_to_8": {str(t): v for t, v in only_b.items()},
        "windows_only_in_b_total": int(sum(only_b.values())),
        "note": "at hist 4 the t = 3..8 windows are themselves clamped (room 0..5); at hist 10 they are not trained on at all (their history would be padded)",
    }
    return out


def d4():
    g = json.loads((PKG / "raw" / "eval_grid_windows.json").read_text(encoding="utf-8"))
    t = np.array([tt for c in g["order"] for tt in g["windows"][c]], np.int64)
    real = lambda h: np.minimum(t + 1, h)                                       # frames with idx = t - 15 + p >= 0 among the h the readers see (V8FMiniWindows.item: okf = idx >= 0)
    out = {"source": "raw/eval_grid_windows.json", "n_windows": int(t.size), "n_episodes": len(g["order"]), "sha256": sha256(PKG / "raw" / "eval_grid_windows.json")}
    for h in (4, 10):
        r = real(h)
        out[f"hist{h}"] = {"real_frames_histogram": {str(int(v)): int((r == v).sum()) for v in np.unique(r)}, "n_fully_real": int((r == h).sum()), "n_padded": int((r < h).sum())}
    out["t_counts"] = {"t3": int((t == 3).sum()), "t7": int((t == 7).sum()), "t_ge_11": int((t >= 11).sum()), "t_lt_11": int((t < 11).sum()), "t_ge_9": int((t >= 9).sum())}
    out["prereg_expects"] = {"t3": [141, "4 real frames, 6 padded"], "t7": [141, "8 real, 2 padded"], "fully_real": 2117}
    h10 = out["hist10"]["real_frames_histogram"]
    out["pass"] = bool(out["t_counts"]["t3"] == 141 and out["t_counts"]["t7"] == 141 and h10 == {"4": 141, "8": 141, "10": 2117} and out["hist4"]["n_padded"] == 0
                       and out["t_counts"]["t_ge_11"] == out["t_counts"]["t_ge_9"] == 2117)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--out", default=str(HERE.with_suffix(".json")))
    a = ap.parse_args(argv)
    res = {"schema": "v8f-hist-measure-v1", "prereg": "PREREG_V8F_HIST.md, what else moves with the lever", "D1": d1(), "D2": d2(), "D3": d3(a.manifest), "D4": d4()}
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
    d3r = res["D3"]
    print("D1 params:", "PASS" if res["D1"]["pass"] else "FAIL", "delta", res["D1"]["delta_total"], res["D1"]["delta_by_group"])
    print("D2 init: unexplained identical-nonconstant:", res["D2"]["identical_not_constant_unexplained"], "| identical tensors", res["D2"]["n_tensors_identical"], "different", res["D2"]["n_tensors_different"], "| identical but NOT constant:", res["D2"]["identical_and_not_constant"], "| time rows 0-3 identical:", res["D2"]["time_first_4_rows_identical_to_the_hist4_table"])
    if isinstance(d3r.get("manifest"), dict):
        print("D3 B reproduced:", d3r["b_reproduced"], "| H hist10:", d3r["h_hist10"])
        print("D3 clamp:", {k: v for k, v in d3r["d3_clamp"].items() if k != "of_which_per_t"})
    else:
        print("D3", d3r["manifest"])
    print("D3 clamp restatement vs the real item():", res["D3"]["clamp_restatement_check"])
    print("D4 padding:", "PASS" if res["D4"]["pass"] else "FAIL", res["D4"]["hist10"], res["D4"]["hist4"], res["D4"]["t_counts"])
    ok = res["D1"]["pass"] and res["D4"]["pass"] and (not isinstance(d3r.get("manifest"), dict) or d3r["b_reproduced"]["equals_banked"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
