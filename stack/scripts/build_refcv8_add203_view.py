#!/usr/bin/env python3
"""Build the refcv8 train view = the refcv6 4,369-clip view + the 203 clips the PI ruled in (2026-10-12).

THOR-SIDE, METADATA ONLY. No frame is decoded and no existing data file is opened for writing:
  * the new view is a NEW directory of symlinks into the parent B1 cache (same recipe as the
    old view: absolute target `<parent>/<clip_id>.v2ep.pt`);
  * its `_v2manifest.pt` is the OLD manifest's entries (carried bit-for-bit) MERGED with the
    entries `v2_dataset._scan_meta` yields for the 203 new files -- the same function
    `load_or_build_manifest` calls, which mmap-reads only each file's small pose/action storages.
    That is ~203 small reads instead of re-reading all 4,572 (the audit priced a full rebuild).

CONTROLS (the build REFUSES when any fails):
  C1  3 OLD clips re-scanned with `_scan_meta` must reproduce the OLD manifest's entries
      EXACTLY (torch.equal on poses/actions, == on every scalar) -- the merge uses the same
      function that produced the old entries.
  C2  the new manifest's `files` list must equal `v2_dataset._list_clips(new_view)` (sorted
      glob of the new view) -- the trainer's own freshness test (`man.get("files") == files`).
  C3  the 4,369 old entries in the new manifest are the old manifest's objects (index-mapped).
  C4  window-count control: `sum(T_out) - n_clips * c` must reproduce the OLD n_windows
      (746,946, refcv7 config) with an INTEGER c, which is then applied to the new set.
DISJOINTNESS: new view n eval139 listing = 0 and n v8-eval labels = 0 (positive checks).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import sys
import time


def sha12(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def md5(p: str) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-view", required=True)
    ap.add_argument("--parent", required=True)
    ap.add_argument("--new-view", required=True)
    ap.add_argument("--add-ids", required=True)
    ap.add_argument("--eval-view", required=True, help="the eval139 view dir (listing only)")
    ap.add_argument("--train-labels", required=True)
    ap.add_argument("--eval-labels", required=True)
    ap.add_argument("--stack", required=True, help="repo stack/ dir (for tanitad.data.v2_dataset)")
    ap.add_argument("--old-n-windows", type=int, default=746946)
    ap.add_argument("--report", required=True)
    a = ap.parse_args(argv)
    sys.path.insert(0, a.stack)
    import torch
    from tanitad.data import v2_dataset as V

    t0 = time.time()
    old_view, parent, new_view = a.old_view, a.parent, a.new_view
    if os.path.exists(new_view):
        raise SystemExit(f"REFUSE: {new_view} already exists (every output is a NEW path)")
    rep: dict = {"schema": "tanitad.refcv8_add203_view_report/1", "controls": {}}

    # ---- inputs -------------------------------------------------------------------------
    add_ids = [x.strip() for x in open(a.add_ids) if x.strip()]
    assert len(add_ids) == 203 and len(set(add_ids)) == 203, "ADD list is not 203 distinct ids"
    old_links = {}
    for fn in sorted(os.listdir(old_view)):
        if fn.endswith(".v2ep.pt"):
            p = os.path.join(old_view, fn)
            assert os.path.islink(p), f"old view entry {fn} is not a symlink"
            old_links[fn[:-len(".v2ep.pt")]] = os.readlink(p)
    assert len(old_links) == 4369, len(old_links)
    for cid, tgt in old_links.items():
        assert tgt == f"{parent}/{cid}.v2ep.pt", (cid, tgt)
    parent_ids = {fn[:-len(".v2ep.pt")] for fn in os.listdir(parent) if fn.endswith(".v2ep.pt")}
    assert len(parent_ids) == 4713, len(parent_ids)
    assert not (set(add_ids) & set(old_links)), "ADD overlaps the old view"
    missing = [c for c in add_ids if c not in parent_ids]
    assert not missing, f"ADD clips absent from the parent cache: {missing[:3]}"
    for c in add_ids:
        assert os.path.isfile(f"{parent}/{c}.v2ep.pt"), c
    eval_ids = {fn[:-len(".v2ep.pt")] for fn in os.listdir(a.eval_view) if fn.endswith(".v2ep.pt")}
    assert len(eval_ids) == 139
    assert not (set(add_ids) & eval_ids) and not (set(old_links) & eval_ids), "eval139 overlap"
    tr_lab = {json.loads(l)["clip_id"] for l in gzip.open(a.train_labels, "rt", encoding="utf-8")}
    ev_lab = {json.loads(l)["clip_id"] for l in gzip.open(a.eval_labels, "rt", encoding="utf-8")}
    assert len(tr_lab) == 4572 and len(ev_lab) == 147 and not (tr_lab & ev_lab)
    assert set(add_ids) <= tr_lab and not (set(add_ids) & ev_lab)
    assert set(old_links) <= tr_lab
    new_ids = sorted(set(old_links) | set(add_ids))
    assert len(new_ids) == 4572 and set(new_ids) == tr_lab, "new view != v8 train label set"
    assert not (set(new_ids) & ev_lab) and not (set(new_ids) & eval_ids)
    rep["inputs"] = {"n_old": 4369, "n_add": 203, "n_new": len(new_ids),
                     "parent_n": len(parent_ids), "eval139_listing_n": len(eval_ids),
                     "v8_train_labels_n": len(tr_lab), "v8_eval_labels_n": len(ev_lab),
                     "new_eq_v8_train_label_set": True,
                     "new_n_eval139_listing": 0, "new_n_v8_eval_labels": 0}

    # ---- THE PARITY INGEST GATE (tests/test_build_parity_guard.py; parity.py sec. 10c) ----------
    # The eval139 / v8-eval checks above are this view's OWN disjointness. The gate asks the programme-wide
    # question by per-clip sha256: is any of these 4,572 inside the 40-episode DEPLOYED VAL? It runs BEFORE
    # the first byte of the view is written and REFUSES on any overlap. (Added 2026-10-12 after the launch
    # gate's G-SUITE-PINNED run on a0b9faf failed `test_every_derived_corpus_writer_is_gated_or_classified`
    # on this module; the 4,572 view already built on Thor was re-checked through the same call: 0 inside.)
    from tanitad.data import parity
    parity.require_ingest_gate("build_refcv8_add203_view")
    _kept, rep["parity_gate"] = parity.guard_corpus_build(
        new_ids, label="build_refcv8_add203_view -> " + new_view, role="train", mode="refuse")
    assert sorted(_kept) == new_ids, "parity gate changed the clip set in refuse mode"

    # ---- old manifest + scan the 203 + controls -----------------------------------------
    old_man_path = os.path.join(old_view, V.MANIFEST_NAME)
    old_md5 = md5(old_man_path)
    old = torch.load(old_man_path, map_location="cpu", weights_only=False)
    assert old["version"] == V.MANIFEST_VERSION, (old["version"], V.MANIFEST_VERSION)
    old_files = old["files"]
    assert len(old_files) == 4369 and old_files == [c + ".v2ep.pt" for c in sorted(old_links)]
    keys = list(old.keys())
    per_clip_keys = [k for k in keys if isinstance(old[k], list) and len(old[k]) == 4369]
    assert set(per_clip_keys) == {"files", "poses", "actions", "episode_id", "n_stack", "image_size",
                                  "image_h", "image_w", "T_out", "clip_id", "episode_uid"}, per_clip_keys
    # C1 control: re-scan 3 old clips with the SAME function
    rnd = random.Random(0)
    ctl_idx = sorted(rnd.sample(range(4369), 3))
    c1 = []
    for i in ctl_idx:
        r = V._scan_meta(os.path.join(parent, old_files[i]))
        okc = (torch.equal(r[0], old["poses"][i]) and torch.equal(r[1], old["actions"][i])
               and r[2] == old["episode_id"][i] and r[3] == old["n_stack"][i]
               and r[4] == old["image_size"][i] and r[5] == old["clip_id"][i]
               and r[6] == old["image_h"][i] and r[7] == old["image_w"][i]
               and int(r[0].shape[0]) == old["T_out"][i]
               and V.stable_episode_id(r[5]) == old["episode_uid"][i])
        c1.append({"clip": old_files[i][:8], "identical": bool(okc)})
        assert okc, f"C1 control FAILED for {old_files[i]}"
    rep["controls"]["C1_rescan_old_clips_equal_old_manifest"] = c1

    new_entries = {}
    for cid in sorted(add_ids):
        poses, actions, eid, n_stack, S, clip_id, H, W = V._scan_meta(f"{parent}/{cid}.v2ep.pt")
        assert clip_id == cid, (clip_id, cid)
        new_entries[cid + ".v2ep.pt"] = (poses, actions, eid, n_stack, S, clip_id, H, W,
                                          int(poses.shape[0]), V.stable_episode_id(clip_id))
    rep["scan_203_s"] = round(time.time() - t0, 1)

    # ---- merge -------------------------------------------------------------------------
    files_new = [c + ".v2ep.pt" for c in new_ids]
    oidx = {f: i for i, f in enumerate(old_files)}
    man = {k: [] for k in keys}
    man["version"] = old["version"]
    for f in files_new:
        if f in oidx:
            i = oidx[f]
            for k in per_clip_keys:
                man[k].append(old[k][i])
        else:
            p, ac, eid, ns, S, cid, H, W, T, uid = new_entries[f]
            row = {"files": f, "poses": p, "actions": ac, "episode_id": eid, "n_stack": ns,
                   "image_size": S, "image_h": H, "image_w": W, "T_out": T, "clip_id": cid,
                   "episode_uid": uid}
            for k in per_clip_keys:
                man[k].append(row[k])
    for k in keys:
        if k not in per_clip_keys and k != "version":
            man[k] = old[k]
    pos = {f: j for j, f in enumerate(files_new)}
    for f, i in oidx.items():
        j = pos[f]
        assert man["poses"][j] is old["poses"][i] and man["actions"][j] is old["actions"][i]
        assert man["clip_id"][j] == old["clip_id"][i] and man["T_out"][j] == old["T_out"][i]
        assert man["episode_uid"][j] == old["episode_uid"][i]
    rep["controls"]["C3_old_entries_carried_bit_for_bit"] = True
    uids = man["episode_uid"]
    assert len(set(uids)) == len(uids) == 4572, "episode_uid collision in the new manifest"

    # C4 window count
    s_old = sum(old["T_out"])
    c_off = (s_old - a.old_n_windows) / 4369
    assert abs(c_off - round(c_off)) < 1e-9, f"window-count control not integer: {c_off}"
    c_off = int(round(c_off))
    n_new = sum(man["T_out"]) - 4572 * c_off
    n_old_re = s_old - 4369 * c_off
    rep["windows"] = {"offset_c_per_clip": c_off, "old_n_windows_reproduced": n_old_re,
                      "new_n_windows": n_new, "add_n_windows": n_new - n_old_re,
                      "windows_per_add_clip": round((n_new - n_old_re) / 203, 4),
                      "one_epoch_steps_b16_drop_last": n_new // 16,
                      "dropped_per_epoch": n_new - 16 * (n_new // 16),
                      "min_T_out_add": min(v[8] for v in new_entries.values())}
    rep["controls"]["C4_window_count_reproduces_old_746946"] = (n_old_re == a.old_n_windows)
    assert min(man["T_out"]) - c_off >= 1

    # ---- write the NEW view ------------------------------------------------------------
    os.makedirs(new_view)
    for c in new_ids:
        os.symlink(f"{parent}/{c}.v2ep.pt", os.path.join(new_view, c + ".v2ep.pt"))
    assert V._list_clips(new_view) == man["files"], "C2: manifest files != sorted glob of the new view"
    rep["controls"]["C2_manifest_files_equal_trainer_list_clips"] = True
    mp = os.path.join(new_view, V.MANIFEST_NAME)
    torch.save(man, mp + ".tmp")
    os.replace(mp + ".tmp", mp)
    chk = torch.load(mp, map_location="cpu", weights_only=False)
    assert chk["files"] == man["files"] and len(chk["poses"]) == 4572 and chk["version"] == V.MANIFEST_VERSION
    for f in files_new:
        assert os.path.isfile(os.path.join(new_view, f)), f
    rep["controls"]["every_symlink_target_resolves"] = True

    # ---- _VIEW_RECORD.json (same keys as the old record + the delta) --------------------
    old_rec = json.load(open(os.path.join(old_view, "_VIEW_RECORD.json"), encoding="utf-8"))
    not_v8 = sorted(parent_ids - tr_lab)
    rec = {
        "view": new_view, "parent": parent, "n_view": 4572,
        "stages": {"cache": 4713, "labels_v8_train": 4572,
                   "agent_join": "NOT A FILTER in this view (4,427 of the 4,572 have a join; the other 145 are NO_LABEL for agents)",
                   "map_gt": "NOT A FILTER in this view (4,369 of the 4,572 have validated SAM3 map GT; the other 203 are map-unlabelled)"},
        "removed": {"not_v8_train": len(not_v8), "no_agent_join": 0, "no_validated_map": 0},
        "removed_sha12": {"not_v8_train": sorted(sha12(c) for c in not_v8)},
        "added_vs_previous_view": {
            "previous_view": old_view, "previous_view_n": 4369,
            "previous_record_md5": md5(os.path.join(old_view, "_VIEW_RECORD.json")),
            "previous_manifest_md5": old_md5,
            "n_added": 203,
            "added_groups": {"no_agent_join": 145, "no_validated_map": 58},
            "added_sha12_no_agent_join": old_rec["removed_sha12"]["no_agent_join"],
            "added_sha12_no_validated_map": old_rec["removed_sha12"]["no_validated_map"],
            "ruling": "PI 2026-10-12: one full pass, add the missing clips, never the val split",
            "add_list_md5": md5(a.add_ids),
        },
        "join": old_rec.get("join"), "labels": old_rec.get("labels"),
        "map_gt_revision": old_rec.get("map_gt_revision"),
        "disjoint_from_eval139": True, "built_by": "stack/scripts/build_refcv8_add203_view.py",
        "parity_gate": rep["parity_gate"],
    }
    with open(os.path.join(new_view, "_VIEW_RECORD.json"), "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    rep["outputs"] = {
        "new_view": new_view, "n_symlinks": len(new_ids),
        "manifest_md5": md5(mp), "manifest_bytes": os.path.getsize(mp),
        "view_record_md5": md5(os.path.join(new_view, "_VIEW_RECORD.json")),
        "old_manifest_md5_unchanged": md5(old_man_path) == old_md5 == "3c9f8bc8583938ff60c57d4d97fcd8c0"}
    assert rep["outputs"]["old_manifest_md5_unchanged"]
    rep["wall_s"] = round(time.time() - t0, 1)
    with open(a.report, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    print("ADD203_VIEW_DONE " + json.dumps({"n": 4572, "n_windows": n_new,
                                            "manifest_md5": rep["outputs"]["manifest_md5"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
