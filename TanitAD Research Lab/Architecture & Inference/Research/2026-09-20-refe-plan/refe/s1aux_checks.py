#!/usr/bin/env python3
"""Checks and small tools for the S1 pod chain `refe/s1aux_pod.sh` (eval/S1_POD_RUNBOOK.md).  Spec: eval/PREREG_AUX_DRIVABLE_S1.md.

Every subcommand prints ONE final marker line (`ZZS1AUX_<X>_OK ...` or `ZZS1AUX_<X>_REFUSED <reason>`) and exits 0 / 1.  The chain judges the marker AND the exit code, never the exit code alone.

  tree           the shipped code tree (refe/ code/ splits/ under --pkg) equals SHIP_MANIFEST.json, file for file (the launch gate re-measures the same tree: a drift is caught HERE, not after hours)
  store          the aux label store verifies by its OWN sha check: AuxLabelSource (pinned store_sha256 + key count) + verify_all (every shard's sha256), mismatch must be 0
  data-check     the `data` event of a --preflight / run log: the S1 data is the audited data (train sets, held-out sets and logs, no train row in a held-out log, teacher-lane floor)
  run-check      a FINISHED arm: the `start` and `done` events read updates == 400, the evals/dumps exist, a head arm carries its head files and an N arm carries none
  manifest-row   append one row (token id, argv sha256, commit, key id) to the arms manifest; the chain writes it BEFORE the arm's first training step (PREREG_AUX_DRIVABLE_S1 section 2)
  remote-verify  the pushed files, by the REMOTE listing: LFS sha256 (or the git blob id of a small file) equals the local file's; never the push's return code
  extract        a .tar / .tar.zst into a directory, every member validated BEFORE it is written (p2r_sync.safe_extract: no absolute path, no '..', files and dirs only, no chown)
  fetch          one file of an HF repo into a local dir (p2r_sync.default_fetch)

The constants below are the AUDITED values (raw/2026-10-13-aux-s1-data-audit/S1_DATA_AUDIT.md, MEASURED by emulating the trainer's own draw on the banked rows): a data event that
differs means the arm would train on other data than the one the registration was written for, which is a refusal, not a footnote.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

UPDATES = 400                      # PREREG_AUX_DRIVABLE_S1 section 4 rule 1: "a run whose start event does not read updates == 400 is not an S1 arm"
TRAIN_SETS = 174912                # tr_idx: bank rows whose key is in op.by, log not held out, PDM-labelled (audit section 1)
HELDOUT_SETS = 3137
HELDOUT_LOGS = 24
TLANE_MIN_ROWS = 2109              # SFT-4's train teacher-lane count at launch (PREREG_SFT4R)
STORE_KEYS = 103139
MODEL_B_MIN_BYTES = 1_200_000_000  # model_sft_B.pt is a full 1,288,48x,xxx-byte checkpoint


def say(marker, *rest):
    print(" ".join([marker, *[str(x) for x in rest]]), flush=True)


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln.startswith("{"):
                continue
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
    return rows


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def git_blob_id(path):
    """The git object id of a file's content: what the Hub lists for a file that is NOT stored in LFS."""
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


# ------------------------------------------------------------------------------------------------------------- tree
def tree_check(pkg):
    import gate_token as GT
    mp = Path(pkg) / GT.SHIP_MANIFEST
    if not mp.is_file():
        return False, f"{GT.SHIP_MANIFEST} is missing under {pkg}"
    man = GT.read_json(mp)
    got = GT.tree_manifest(pkg)
    want = man.get("files") or {}
    if got != want:
        d = GT.tree_diff(want, got)
        return False, f"the shipped tree differs from {GT.SHIP_MANIFEST} (commit {str(man.get('commit'))[:12]}): {json.dumps(d)[:600]}"
    if GT.tree_digest(got) != man.get("tree_sha256"):
        return False, "tree_sha256 of the manifest does not hash its own files map"
    return True, f"{len(got)} files, commit {str(man.get('commit'))[:12]}, tree {str(man.get('tree_sha256'))[:12]}"


# ------------------------------------------------------------------------------------------------------------- store
def store_check(store):
    import aux_drivable_head as AUX
    try:
        src = AUX.AuxLabelSource(store)                     # refuses a store whose store_sha256 / key count / schema is not the pinned one
        v = src.verify_all()                                # sha256 of EVERY shard against the manifest
    except Exception as e:                                  # noqa: BLE001 -- a refusal
        return False, f"{type(e).__name__}: {str(e)[:300]}"
    keys = src.manifest.get("counts", {}).get("stored_keys")
    if v["mismatch"] != 0 or keys != STORE_KEYS:
        return False, f"verify_all mismatch {v['mismatch']} (first {v['first_bad']}), stored_keys {keys} != {STORE_KEYS}"
    return True, f"{v['files']} shards, {keys} keys, store_sha256 {str(src.manifest.get('store_sha256'))[:16]}"


# ------------------------------------------------------------------------------------------------------------- data event
def data_problems(ev):
    """Problems of a `data` event against the audited S1 data.  [] = it is the audited data."""
    bad = []
    if not ev:
        return ["no `data` event"]
    lab = ev.get("lane_labels") or {}
    for k, want in (("train_sets", TRAIN_SETS), ("heldout_sets", HELDOUT_SETS), ("heldout_logs", HELDOUT_LOGS), ("train_rows_in_heldout_logs", 0)):
        if ev.get(k) != want:
            bad.append(f"{k} {ev.get(k)} != {want}")
    if not (isinstance(lab.get("teacher_lane_train"), int) and lab["teacher_lane_train"] >= TLANE_MIN_ROWS):
        bad.append(f"teacher_lane_train {lab.get('teacher_lane_train')} < {TLANE_MIN_ROWS} (the train-side lane loss would be masked)")
    for k in ("teacher_lane_heldout", "pdm_train", "pdm_heldout"):
        if not (isinstance(lab.get(k), int) and lab[k] > 0):
            bad.append(f"lane_labels.{k} {lab.get(k)} loaded no labels")
    return bad


def data_check(jsonl):
    ev = [r for r in read_jsonl(jsonl) if r.get("event") == "data"]
    if len(ev) != 1:
        return False, f"{len(ev)} data events in {jsonl}, want 1"
    bad = data_problems(ev[0])
    if bad:
        return False, "; ".join(bad)[:600]
    lab = ev[0]["lane_labels"]
    return True, f"train_sets {ev[0]['train_sets']} heldout_sets {ev[0]['heldout_sets']} logs {ev[0]['heldout_logs']} tlane_train {lab['teacher_lane_train']}"


# ------------------------------------------------------------------------------------------------------------- finished arm
def arm_class(arm):
    return arm.split("-")[0]


def run_problems(d, arm):
    """Problems of a FINISHED arm directory `d` (sft.jsonl, model_sft_B.pt, dump/, aux_drv.pt).  [] = a valid S1 arm."""
    d = Path(d)
    cls = arm_class(arm)
    bad = []
    lp = d / "sft.jsonl"
    if not lp.is_file():
        return [f"{lp} missing"]
    rows = read_jsonl(lp)
    by = {}
    for r in rows:
        by.setdefault(r.get("event"), []).append(r)
    st, dn = by.get("start", []), by.get("done", [])
    if len(st) != 1 or st[0].get("updates") != UPDATES:
        bad.append(f"start events {[s.get('updates') for s in st]}, want exactly one reading updates == {UPDATES} (not an S1 arm)")
    if len(dn) != 1 or dn[0].get("updates") != UPDATES:
        bad.append(f"done events {[x.get('updates') for x in dn]}, want exactly one reading updates == {UPDATES}")
    tags = {r.get("tag") for r in by.get("eval", [])}
    if not {"step0", f"update{UPDATES}"} <= tags:
        bad.append(f"eval tags {sorted(map(str, tags))} lack step0 / update{UPDATES}")
    bad += [f"data: {p}" for p in data_problems((by.get("data") or [None])[0])]
    m = d / "model_sft_B.pt"
    if not m.is_file() or m.stat().st_size < MODEL_B_MIN_BYTES:
        bad.append(f"model_sft_B.pt missing or smaller than {MODEL_B_MIN_BYTES} B")
    dump = d / "dump"
    for tag in ("step0", f"update{UPDATES}"):
        if not glob.glob(str(dump / f"scores_{tag}_*.npz")):
            bad.append(f"dump/scores_{tag}_*.npz missing (the paired bootstrap's per-set logits)")
    head_files = glob.glob(str(dump / "head_pred_*.npz"))
    train_rows = [r for r in by.get("train", []) if isinstance(r, dict)]
    if cls == "N":
        if (d / "aux_drv.pt").exists() or head_files or by.get("aux_eval") or any("aux_bce" in r for r in train_rows):
            bad.append("an N arm carries head artifacts (aux_drv.pt / head_pred_*.npz / aux_eval / aux_bce): the no-head replicate must differ from the head arms by the head alone")
    else:
        if not (d / "aux_drv.pt").is_file():
            bad.append("aux_drv.pt missing on a head arm")
        atags = {r.get("tag") for r in by.get("aux_eval", [])}
        if not {"step0", f"update{UPDATES}"} <= atags:
            bad.append(f"aux_eval tags {sorted(map(str, atags))} lack step0 / update{UPDATES}")
        for tag in ("step0", f"update{UPDATES}"):
            if not any(tag in os.path.basename(h) for h in head_files):
                bad.append(f"dump/head_pred_*{tag}*.npz missing (the P1 / P2 / P3 readout's input)")
        ab = [r.get("aux_bce") for r in train_rows if "aux_bce" in r]
        if not ab or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in ab):
            bad.append(f"aux_bce on train rows: {len(ab)} rows, finite={all(isinstance(x, (int, float)) and math.isfinite(x) for x in ab)}")
    for r in train_rows:
        for k in ("bceA", "lossB"):
            if k in r and not (isinstance(r[k], (int, float)) and math.isfinite(r[k])):
                bad.append(f"non-finite {k} in a train row (K2)")
                break
    return bad


def run_check(d, arm, out_json=None):
    bad = run_problems(d, arm)
    if out_json:
        Path(out_json).write_text(json.dumps({"arm": arm, "dir": str(d), "problems": bad, "ok": not bad, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1), encoding="utf-8")
    return (not bad), ("; ".join(bad)[:700] if bad else f"{arm}: updates {UPDATES}, evals + dumps present" + (", head files present" if arm_class(arm) != "N" else ", no head artifact"))


# ------------------------------------------------------------------------------------------------------------- manifest row
def manifest_row(arm, event, out, token=None, k=None, note=None):
    row = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "arm": arm, "event": event}
    if token:
        tok = json.loads(Path(token).read_text(encoding="utf-8"))
        b = tok.get("binding") or {}
        row.update({"token_file": os.path.basename(token), "token_sha256": sha256_file(token), "token_verdict": tok.get("verdict"), "profile": b.get("profile"),
                    "commit": b.get("commit"), "tree_sha256": b.get("tree_sha256"), "argv_sha256": b.get("argv_sha256"), "key_id": tok.get("key_id"),
                    "token_created_utc": tok.get("created_utc")})
    if k is not None:
        row["restart_k"] = int(k)
    if note:
        row["note"] = note
    with open(out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    return row


# ------------------------------------------------------------------------------------------------------------- remote verification
def verify_remote(api, repo, repo_type, pairs):
    """pairs = [(remote_path, local_path)].  Problems (empty = every file is on the Hub with the local content): LFS sha256, or the git blob id of a file that is not in LFS."""
    tree = {}
    for it in api.list_repo_tree(repo, recursive=True, repo_type=repo_type):
        tree[it.path] = it
    bad = []
    for rp, lp in pairs:
        it = tree.get(rp)
        if it is None:
            bad.append(f"missing_remote:{rp}")
            continue
        lfs = getattr(it, "lfs", None)
        sha = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs is not None else None
        if sha is not None:
            if sha != sha256_file(lp):
                bad.append(f"sha256_mismatch:{rp}")
        else:
            if getattr(it, "blob_id", None) != git_blob_id(lp):
                bad.append(f"blob_id_mismatch:{rp}")
    return bad


# ------------------------------------------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("tree"); p.add_argument("--pkg", required=True)
    p = sub.add_parser("store"); p.add_argument("--store", required=True)
    p = sub.add_parser("data-check"); p.add_argument("--jsonl", required=True)
    p = sub.add_parser("run-check"); p.add_argument("--dir", required=True); p.add_argument("--arm", required=True); p.add_argument("--json")
    p = sub.add_parser("manifest-row"); p.add_argument("--arm", required=True); p.add_argument("--event", required=True); p.add_argument("--out", required=True)
    p.add_argument("--token"); p.add_argument("--k"); p.add_argument("--note")
    p = sub.add_parser("remote-verify"); p.add_argument("--repo", required=True); p.add_argument("--repo-type", default="model", choices=["model", "dataset"])
    p.add_argument("--prefix", default=""); p.add_argument("--files", nargs="+", required=True); p.add_argument("--names", nargs="+")
    p = sub.add_parser("extract"); p.add_argument("--archive", required=True); p.add_argument("--dest", required=True); p.add_argument("--roots", default="")
    p = sub.add_parser("fetch"); p.add_argument("--repo", required=True); p.add_argument("--repo-type", default="model", choices=["model", "dataset"])
    p.add_argument("--path", required=True); p.add_argument("--local-dir", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "tree":
        ok, why = tree_check(a.pkg)
        say("ZZS1AUX_TREE_OK" if ok else "ZZS1AUX_TREE_REFUSED", why)
    elif a.cmd == "store":
        ok, why = store_check(a.store)
        say("ZZS1AUX_STORE_OK" if ok else "ZZS1AUX_STORE_REFUSED", why)
    elif a.cmd == "data-check":
        ok, why = data_check(a.jsonl)
        say("ZZS1AUX_DATA_OK" if ok else "ZZS1AUX_DATA_REFUSED", why)
    elif a.cmd == "run-check":
        ok, why = run_check(a.dir, a.arm, a.json)
        say("ZZS1AUX_RUN_OK" if ok else "ZZS1AUX_RUN_REFUSED", why)
    elif a.cmd == "manifest-row":
        row = manifest_row(a.arm, a.event, a.out, a.token, a.k, a.note)
        say("ZZS1AUX_MANIFEST_OK", row["arm"], row["event"], row.get("argv_sha256", "-"))
        ok = True
    elif a.cmd == "remote-verify":
        import hf_egress as E
        names = a.names or [os.path.basename(f) for f in a.files]
        if len(names) != len(a.files):
            say("ZZS1AUX_REMOTE_REFUSED", f"{len(names)} names for {len(a.files)} files")
            return 1
        pairs = [((a.prefix.strip("/") + "/" + n) if a.prefix else n, f) for n, f in zip(names, a.files)]
        try:
            bad = verify_remote(E.make_api(), a.repo, a.repo_type, pairs)
        except Exception as e:                                # noqa: BLE001 -- an unreadable listing verifies nothing
            bad = [f"exception:{type(e).__name__}:{E._sanitize(e)[:200]}"]
        ok = not bad
        say("ZZS1AUX_REMOTE_OK" if ok else "ZZS1AUX_REMOTE_REFUSED", f"{len(pairs)} files" if ok else ",".join(bad)[:600])
    elif a.cmd == "extract":
        import p2r_sync as S
        try:
            n, nb = S.safe_extract(a.archive, a.dest, {r for r in a.roots.split(",") if r})
            ok, why = True, f"{n} files {nb} B"
        except Exception as e:                                # noqa: BLE001
            ok, why = False, f"{type(e).__name__}: {str(e)[:300]}"
        say("ZZS1AUX_EXTRACT_OK" if ok else "ZZS1AUX_EXTRACT_REFUSED", why)
    else:
        import p2r_sync as S
        try:
            p_ = S.default_fetch(a.repo, a.repo_type, a.path, a.local_dir)
            ok, why = True, f"{p_} {Path(p_).stat().st_size} B"
        except Exception as e:                                # noqa: BLE001
            ok, why = False, f"{type(e).__name__}: {str(e)[:300]}"
        say("ZZS1AUX_FETCH_OK" if ok else "ZZS1AUX_FETCH_REFUSED", why)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
