#!/usr/bin/env python3
"""vis1_train4572_controls.py -- the controls for the merged VIS-1 sidecar (refcv8, 4,572 train + 139 eval clips).

  (a) every OLD clip's entry (split, f-theta mask, frame arrays, row arrays) in the NEW sidecar equals the OLD
      sidecar's, exactly (dtype + values); n compared / n differing.
  (b) the UNMODIFIED builder re-run (vis1_subbuild_driver.py) on OLD clips -- 100 train + all 139 eval -- reproduces
      the old sidecar's stored entries exactly: proves the builder / calib / extrinsics / joins used for the ADD clips
      are the ones that built the old file. The three NAMED clips are the first three of the sorted control sample.
  (c) the clip set == the 4,572 train sha12s UNION the 139 eval sha12s, read from the two manifests, split labels
      included, and train/eval disjoint.
  (d) the 58 joined ADD clips: frames > 0, rows > 0, finite, ordered counts (n_vis <= n_img <= n_full), non-degenerate
      (non-zero vis_frac), against an INDEPENDENT derivation of "joined" from the 2-D join shard files; and the 145
      unjoined: 0 frames, 0 rows, mask True, split train.
  (e) the trainer's OWN ``VIS1Sidecar.require_clips`` on the new npz with the full 4,572 + 139 id list does not
      raise -- plus NEGATIVE controls (an absent id, a flipped mask) that must raise, so the check is not vacuous.
  (f) the trainer-side lookup, end to end: for every frame of every joined ADD clip, the trainer's own
      ``V3Dataset._agent_item`` block is joined to the new sidecar through ``vis1_block_for_rows`` (the function
      ``_vis1_item`` calls) -- no refusal, and ``known`` == the in-scope rows with a 3-D label. For the 145 unjoined
      clips ``_agent_item`` returns agent_label False (so ``_vis1_item`` never looks up a frame).

sha12 only in every output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))

import merge_vis1_sidecar as M  # noqa: E402  (per_clip_entries / load_npz / sha12)
from tanitad.data import vis1 as V  # noqa: E402

CLIP_RE = re.compile(r'^\{\s*"clip_id"\s*:\s*"([^"]+)"')
ENTRY_KEYS = ("split", "mask_ftheta", "frame_f", "frame_n_zh", "frame_n_valid", "frame_cnt") + M.ROW_KEYS


def log(m):
    print(f"[vis1-ctl {time.strftime('%H:%M:%S')}] {m}", flush=True)


def entry_diff(x: dict, y: dict) -> list:
    bad = []
    for k in ENTRY_KEYS:
        a, b = x[k], y[k]
        if isinstance(a, np.ndarray):
            if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b):
                bad.append(k)
        elif a != b:
            bad.append(k)
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--sub", required=True)
    ap.add_argument("--select-detail", required=True)
    ap.add_argument("--train-cache", required=True)
    ap.add_argument("--eval-cache", required=True)
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--n-shards", type=int, default=24)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import torch
    res = {"tool": "vis1_train4572_controls.py", "tool_md5": M.md5_file(HERE), "argv": sys.argv[1:],
           "new_sha256": V.file_sha256(a.new), "old_sha256": V.file_sha256(a.old), "sub_sha256": V.file_sha256(a.sub)}
    det = json.load(open(a.select_detail, encoding="utf-8"))
    add, ctl, ev_sel = list(det["add"]), list(det["ctl_old_train"]), list(det["eval"])
    named3 = list(det["ctl_old_train_named3"])
    old_a, _ = M.load_npz(a.old)
    new_a, new_meta = M.load_npz(a.new)
    sub_a, _ = M.load_npz(a.sub)
    old_pc, new_pc, sub_pc = (M.per_clip_entries(x) for x in (old_a, new_a, sub_a))

    # ---- (a) old entries unchanged in the new sidecar -------------------------------------------------- #
    differing = [s for s in old_pc if s not in new_pc or entry_diff(old_pc[s], new_pc[s])]
    res["a_old_entries_unchanged"] = {"n_compared": len(old_pc), "n_differing": len(differing),
                                      "differing_sample": differing[:5], "PASS": len(differing) == 0}
    log(f"(a) old entries: compared {len(old_pc)}, differing {len(differing)}")

    # ---- (b) builder re-run on OLD clips reproduces the stored entries --------------------------------- #
    sub_old = [s for s in sub_pc if s in old_pc]
    b_bad = [s for s in sub_old if entry_diff(old_pc[s], sub_pc[s])]
    n_frames_b = int(sum(len(sub_pc[s]["frame_f"]) for s in sub_old))
    n_rows_b = int(sum(len(sub_pc[s]["row"]) for s in sub_old))
    named = {s: {"frames": int(len(sub_pc[s]["frame_f"])), "rows": int(len(sub_pc[s]["row"])),
                 "equal": s not in b_bad} for s in named3}
    exp = set(ctl) | set(ev_sel)
    res["b_builder_reproduces_old"] = {
        "n_old_clips_rebuilt": len(sub_old), "expected": len(exp), "set_ok": set(sub_old) == exp,
        "n_train": int(sum(1 for s in sub_old if old_pc[s]["split"] == "train")),
        "n_eval": int(sum(1 for s in sub_old if old_pc[s]["split"] == "eval")),
        "n_frames_compared": n_frames_b, "n_rows_compared": n_rows_b, "n_differing": len(b_bad),
        "differing_sample": b_bad[:5], "named3": named,
        "PASS": bool(len(b_bad) == 0 and set(sub_old) == exp and all(v["equal"] for v in named.values()))}
    log(f"(b) rebuilt old clips: {len(sub_old)} ({n_frames_b} frames, {n_rows_b} rows), differing {len(b_bad)}, "
        f"named3 {named}")

    # ---- (c) clip set == manifests --------------------------------------------------------------------- #
    def cl(cache):
        man = torch.load(Path(cache) / "_v2manifest.pt", map_location="cpu", weights_only=False)
        return [str(c) for c in man["clip_id"]], man
    tr_c, tr_man = cl(a.train_cache)
    ev_c, ev_man = cl(a.eval_cache)
    tr12, ev12 = {M.sha12(c): c for c in tr_c}, {M.sha12(c): c for c in ev_c}
    got_tr = {s for s, p in zip(new_a["clip_sha12"].tolist(), new_a["clip_split"].tolist()) if p == "train"}
    got_ev = {s for s, p in zip(new_a["clip_sha12"].tolist(), new_a["clip_split"].tolist()) if p == "eval"}
    res["c_clip_set"] = {"n_clips": int(len(new_a["clip_sha12"])), "n_train": len(got_tr), "n_eval": len(got_ev),
                         "train_manifest": len(tr12), "eval_manifest": len(ev12),
                         "train_equal": got_tr == set(tr12), "eval_equal": got_ev == set(ev12),
                         "train_eval_disjoint": not (set(tr12) & set(ev12)),
                         "sorted_unique": bool((new_a["clip_sha12"][:-1] < new_a["clip_sha12"][1:]).all()),
                         "n_add": len(add), "add_in_train_manifest": all(s in tr12 for s in add),
                         "add_all_present_split_train": all(new_pc[s]["split"] == "train" for s in add)}
    res["c_clip_set"]["PASS"] = bool(
        res["c_clip_set"]["train_equal"] and res["c_clip_set"]["eval_equal"] and
        res["c_clip_set"]["train_eval_disjoint"] and res["c_clip_set"]["n_clips"] == 4572 + 139 and
        res["c_clip_set"]["sorted_unique"] and res["c_clip_set"]["add_all_present_split_train"])
    log(f"(c) clip set: {res['c_clip_set']}")

    # ---- (d) the ADD clips ------------------------------------------------------------------------------ #
    cid_of = {**tr12, **ev12}
    joined_ref = set()                                  # independent derivation: which ADD clips have 2-D join lines
    add_set = set(add)
    for f in sorted(Path(a.shard_dir).glob("j2d_*.jsonl")):
        seen = set()
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                m = CLIP_RE.match(line)
                if m:
                    seen.add(M.sha12(m.group(1)))
        joined_ref |= (seen & add_set)
    with_frames = {s for s in add if len(new_pc[s]["frame_f"]) > 0}
    no_frames = add_set - with_frames
    d = {"n_add": len(add), "joined_from_2d_shards": len(joined_ref), "with_frames": len(with_frames),
         "joined_set_equals_frames_set": joined_ref == with_frames, "zero_frame": len(no_frames)}
    stats, bad_vals, zero_rows = [], [], []
    for s in sorted(with_frames):
        c = new_pc[s]
        nf, ni, nv = c["n_full"].astype(np.int64), c["n_img"].astype(np.int64), c["n_vis"].astype(np.int64)
        ok = bool(len(c["frame_f"]) > 0 and np.isfinite(c["cx"]).all() and np.isfinite(c["cy"]).all()
                  and (ni <= nf).all() and (nv <= ni).all() and (nv >= 0).all() and (c["frame_f"] >= 0).all())
        vf = nv[nf > 0] / nf[nf > 0] if (nf > 0).any() else np.zeros(0)
        stats.append((s, int(len(c["frame_f"])), int(len(c["row"])), float(vf.mean()) if len(vf) else float("nan"),
                      float((nf > 0).mean()) if len(nf) else float("nan")))
        if not ok:
            bad_vals.append(s)
        if len(c["row"]) == 0:
            zero_rows.append(s)
    allrows = np.concatenate([new_pc[s]["n_full"] for s in with_frames]).astype(np.int64)
    allvis = np.concatenate([new_pc[s]["n_vis"] for s in with_frames]).astype(np.int64)
    oldn = np.concatenate([old_pc[s]["n_full"] for s in old_pc]).astype(np.int64)
    oldv = np.concatenate([old_pc[s]["n_vis"] for s in old_pc]).astype(np.int64)
    vf_new = allvis[allrows > 0] / allrows[allrows > 0]
    vf_old = oldv[oldn > 0] / oldn[oldn > 0]
    d.update({"frames_total": int(sum(x[1] for x in stats)), "rows_total": int(sum(x[2] for x in stats)),
              "frames_per_clip_min_med_max": [int(min(x[1] for x in stats)), float(np.median([x[1] for x in stats])),
                                              int(max(x[1] for x in stats))],
              "rows_with_n_full_gt0_frac_ADD": float((allrows > 0).mean()), "rows_with_n_full_gt0_frac_OLD": float((oldn > 0).mean()),
              "mean_vis_frac_ADD": float(vf_new.mean()), "mean_vis_frac_OLD": float(vf_old.mean()),
              "n_clips_bad_values": len(bad_vals), "bad_sample": bad_vals[:5], "n_joined_clips_with_zero_stored_rows": len(zero_rows),
              "sample_clip_stats(sha12,frames,rows,mean_vis_frac,frac_nfull_gt0)": [list(x) for x in stats[:5]]})
    zf_ok = all(len(new_pc[s]["frame_f"]) == 0 and len(new_pc[s]["row"]) == 0 and new_pc[s]["mask_ftheta"]
                and new_pc[s]["split"] == "train" for s in no_frames)
    d["zero_frame_entries_ok(0 frames, 0 rows, mask True, split train)"] = zf_ok
    d["PASS"] = bool(len(with_frames) == 58 and len(no_frames) == 145 and joined_ref == with_frames
                     and not bad_vals and zf_ok and np.isfinite(vf_new.mean()) and vf_new.mean() > 0.0
                     and (allrows > 0).any())
    res["d_add_clips"] = d
    log(f"(d) ADD: with_frames {len(with_frames)}, zero_frame {len(no_frames)}, joined==frames {joined_ref == with_frames}, "
        f"mean vis_frac ADD {vf_new.mean():.4f} OLD {vf_old.mean():.4f}")

    # ---- (e) the trainer's own require_clips ----------------------------------------------------------- #
    sc = V.VIS1Sidecar(a.new)
    ids = list(tr12) + list(ev12)
    e = {"n_ids": len(ids)}
    try:
        sc.require_clips(ids, where="controls train+eval")
        e["require_clips_full_list"] = "no raise"
    except SystemExit as ex:
        e["require_clips_full_list"] = f"RAISED {str(ex)[:200]}"
    try:
        sc.require_clips(ids + ["000000000000"], where="negative control (absent id)")
        e["negative_absent_id"] = "DID NOT RAISE (check is vacuous)"
    except SystemExit:
        e["negative_absent_id"] = "raised (as required)"
    sc2 = V.VIS1Sidecar(a.new)
    sc2.a = {k: v.copy() if k == "clip_mask_ftheta" else v for k, v in sc2.a.items()}
    sc2.a["clip_mask_ftheta"][sc2.clip_index[add[0]]] = False
    try:
        sc2.require_clips(ids, where="negative control (flipped mask)")
        e["negative_flipped_mask"] = "DID NOT RAISE (check is vacuous)"
    except SystemExit:
        e["negative_flipped_mask"] = "raised (as required)"
    e["meta_n_stack"] = new_meta.get("n_stack")
    e["PASS"] = bool(e["require_clips_full_list"] == "no raise" and e["negative_absent_id"].startswith("raised")
                     and e["negative_flipped_mask"].startswith("raised") and int(new_meta.get("n_stack", -1)) == 3)
    res["e_require_clips"] = e
    log(f"(e) {e}")

    # ---- (f) trainer-side lookup, end to end ------------------------------------------------------------ #
    sys.path.insert(0, str(HERE.parent))
    import precompute_vis1_sidecar as B
    B.trainer()
    uid_of = {}
    for man, cs in ((tr_man, tr_c), (ev_man, ev_c)):
        for c, u in zip(cs, [int(x) for x in man["episode_uid"]]):
            uid_of[c] = u
    sel_clips = sorted({cid_of[s] for s in add + ctl} | {c for c in ev_c}, key=B.sha12)
    n_sh = max(1, min(int(a.n_shards), len(sel_clips)))
    shard_clips = [sel_clips[k::n_sh] for k in range(n_sh)]
    f_res = {"n_joined_frames_checked": 0, "n_rows_known": 0, "n_rows_expected_known": 0, "refusals": [],
             "n_unjoined_probes": 0, "unjoined_with_label": []}
    pad_seen = set()
    for k in range(n_sh):
        cs_add = [c for c in shard_clips[k] if B.sha12(c) in add_set]
        if not cs_add:
            continue
        ctx = B.ShardCtx(str(Path(a.shard_dir) / f"j2d_{k:03d}.jsonl"), str(Path(a.shard_dir) / f"j3d_{k:03d}.jsonl"),
                         {uid_of[c]: c for c in shard_clips[k]}, 3)
        pad = int(ctx.ds.agent_pad)
        pad_seen.add(pad)
        for cid in cs_add:
            s12 = B.sha12(cid)
            uid = uid_of[cid]
            if s12 in with_frames:
                for f in [f for f in ctx.frames_by_clip.get(cid, []) if f >= 0]:
                    rows = ctx.rows(uid, f)
                    if rows is None:
                        f_res["refusals"].append([s12, f, "NO_LABEL for a joined frame"])
                        continue
                    box = rows["box"].numpy()
                    inscope = V.in_store_scope(box[:, 0], box[:, 1]) if len(box) else np.zeros(0, bool)
                    exp_known = int((rows["zh"].numpy() & inscope).sum())
                    try:
                        nfull, nvis, known = V.vis1_block_for_rows(
                            sc, s12, f, box=box, track_ids=rows["tids"], zh_mask=rows["zh"].numpy(), pad=pad)
                    except SystemExit as ex:
                        f_res["refusals"].append([s12, f, str(ex)[:160]])
                        continue
                    f_res["n_joined_frames_checked"] += 1
                    f_res["n_rows_known"] += int(known.sum())
                    f_res["n_rows_expected_known"] += exp_known
                    if int(known.sum()) != exp_known:
                        f_res["refusals"].append([s12, f, f"known {int(known.sum())} != expected {exp_known}"])
            else:
                for f in (7, 50, 120):
                    item = ctx.item_fn(ctx.ds, types.SimpleNamespace(episode_id=int(uid)), int(f))
                    f_res["n_unjoined_probes"] += 1
                    if bool(item["agent_label"]):
                        f_res["unjoined_with_label"].append([s12, f])
    f_res["pad_seen"] = sorted(pad_seen)
    f_res["PASS"] = bool(not f_res["refusals"] and not f_res["unjoined_with_label"]
                         and f_res["n_joined_frames_checked"] == d["frames_total"]
                         and f_res["n_rows_known"] == f_res["n_rows_expected_known"] and f_res["n_rows_known"] > 0
                         and f_res["n_unjoined_probes"] == 3 * len(no_frames))
    f_res["refusals"] = f_res["refusals"][:10]
    res["f_trainer_lookup"] = f_res
    log(f"(f) {f_res}")

    res["ALL_PASS"] = bool(all(res[k]["PASS"] for k in res if isinstance(res[k], dict) and "PASS" in res[k]))
    Path(a.out).write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    log(f"ALL_PASS = {res['ALL_PASS']}  ->  {a.out}")
    return 0 if res["ALL_PASS"] else 3


if __name__ == "__main__":
    sys.exit(main())
