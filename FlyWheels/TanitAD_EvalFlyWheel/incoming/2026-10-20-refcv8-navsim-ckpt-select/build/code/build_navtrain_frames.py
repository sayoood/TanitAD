#!/usr/bin/env python3
"""build_navtrain_frames.py -- the navtrain 416x1024 frame bank for the D-R8-CKPT-RULE sample (TANITAD VENV, CPU; torch + the E2 stitch).

PORT of W3's ``build_navtest_frames.py`` (shard-streaming bank builder) as driven by refcv6's ``build_navtest416.py`` (re-pointed to the
416 x 1024 frame).  What is UNCHANGED, on purpose: the stitch (E2's ``build_scene``, loaded through refcv6's ``frames416.py`` and configured
for ``FRAME_416x1024`` -- never re-derived), the bank layout (``frames_sNN.npy`` u8 ``[n_frames, 416, 1024, 3]`` + ONE ``index.json`` whose
``tokens[t] = {shard, file, rows, log_name, rig_key, observed_frac, mean_px, sha256[:16]}``), the per-token sha256 of the stacked
``[4, 416, 1024, 3]`` array, and KB2 (``mean_px >= 1.0``).  The bridge reads the result with ``refcv6_bridge.BankNavtest416`` UNCHANGED
(``run_bridge8.py --bank1-kind navtest``); the test suite loads a bank built here through that very class.

What is NEW (navtrain differs from navtest, and ``run_bridge8`` hard-refuses a missing frame ONLY for split ``navtest`` -- GROUNDWORK B7):

* the shards are OpenScene ``navtrain_current_N.tgz`` + ``navtrain_history_N.tgz`` (members ``<root>/<log>/<CAM>/<file>.jpg``), read as streams;
* **KB3 + B7: EVERY token of the sample needs all 12 jpgs (4 history frames x {CAM_L0, CAM_F0, CAM_R0}).  If ANY is absent the build
  REFUSES (exit 3, ``REFUSED.json``, no ``index.json``).**  There is no "partial token", no skipped token and no stand-in: on navtrain a
  missing frame would otherwise silently become a declared CV stand-in row in ``run_bridge8``;
* a frame that fails to stitch, an all-black frame (KB2), an inputs export that does not cover the sample, two shard members with the
  same path but different bytes -> the same hard refusal;
* after writing, the bank is RE-READ the way ``BankNavtest416.load`` reads it and every token's sha256 is re-derived (a bank that does not
  round-trip is deleted from view: ``DONE.json`` is written last and only on success).

    python build_navtrain_frames.py --sample NAVTRAIN_SAMPLE_TOKENS.json --inputs navtrain_sample_inputs.json.gz --shards-dir <dir> \
        --logs-root <navsim_logs/trainval> --bank <out>/frame_bank --frames416 <.../refcv6-standard-tests/navsim/code/frames416.py>

Markers on stdout: ``ZZBANK_DONE tokens=<n> frames=<k>`` or ``ZZBANK_REFUSED_<why>``.  ``R8BANK_MUTATION`` (tests only) reintroduces one
defect class; ``main`` refuses while it is set.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import sys
import tarfile
import time

import numpy as np

import navtrain_common as N

MUTATION = os.environ.get("R8BANK_MUTATION", "")
MUTATIONS = ("missing_frame_standin", "kb3_t0_only", "zero_frame_ok", "dup_content_ignored", "partial_on_stitch_failure", "rows_shifted")
CAMS_LOWER = ("cam_l0", "cam_f0", "cam_r0")
H, W = 416, 1024


class Refused(Exception):
    """A hard refusal.  ``why`` is the opaque marker suffix; ``detail`` goes to REFUSED.json."""

    def __init__(self, why: str, detail: dict):
        super().__init__(why)
        self.why, self.detail = why, detail


def token_rels(rec: dict) -> list:
    """The 12 jpg rels (4 frames x 3 cameras) a token needs, from the inputs export record (W3's ``token_paths``)."""
    if MUTATION == "kb3_t0_only":
        return [str(rec["cams"][c][-1]).replace("\\", "/") for c in CAMS_LOWER]
    return [str(p).replace("\\", "/") for c in CAMS_LOWER for p in rec["cams"][c]]


def extract_needed(sources: list, need: dict, scratch: pathlib.Path) -> tuple:
    """Stream every shard once; copy ONLY the needed jpgs into ``scratch/<log>/<CAM>/<file>``.  Returns ``(extracted rels, stats)``.
    Two members with the same rel but different bytes are a refusal (a corrupt or mixed download)."""
    extracted, stats = {}, []
    for src in sources:
        n_seen = n_new = n_dup = 0
        if os.path.isdir(src):
            base = src.replace("\\", "/").rstrip("/")
            items = ((N.member_rel(os.path.join(dp, fn).replace("\\", "/")[len(base) + 1:], strip_root=False), os.path.join(dp, fn))
                     for dp, _d, fns in os.walk(src) for fn in fns)
            readers = ((rel, (lambda p=p: open(p, "rb").read())) for rel, p in items if rel)
        else:
            readers = _tar_readers(src)
        for rel, read in readers:
            n_seen += 1
            if rel not in need:
                continue
            data = read()
            dst = scratch / rel
            if rel in extracted:
                n_dup += 1
                if hashlib.sha256(data).hexdigest() != extracted[rel] and MUTATION != "dup_content_ignored":
                    raise Refused("duplicate_member_content_differs", {"rel": rel, "source": src})
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
            extracted[rel] = hashlib.sha256(data).hexdigest()
            n_new += 1
        stats.append({"source": src.replace("\\", "/"), "cam_jpgs_seen": n_seen, "needed_extracted": n_new, "needed_duplicates": n_dup})
    return extracted, stats


def _tar_readers(path: str):
    with tarfile.open(path, "r|*") as tf:
        for m in tf:
            if not m.isfile():
                continue
            rel = N.member_rel(m.name, strip_root=True)
            if rel:
                yield rel, (lambda f=tf.extractfile(m): f.read())


def build_bank(inputs_doc: dict, sample: dict, sources: list, logs_root: str, bank_dir: str, scratch_dir: str, e2, *,
               keep_scratch: bool = False, progress=print) -> dict:
    """The whole build.  Raises :class:`Refused` on any gap; returns the BUILD_REPORT dict on success."""
    t0 = time.time()
    if sample.get("status") != "FIXED":
        raise Refused("sample_not_fixed", {"status": sample.get("status")})
    toks = sorted(sample["tokens"])
    if N.sha256_tokens(toks) != sample.get("sha256_sorted_tokens") or len(toks) != sample.get("count"):
        raise Refused("sample_sha256_does_not_rederive", {"count": sample.get("count"), "len": len(toks)})
    recs = inputs_doc["tokens"]
    absent = [t for t in toks if t not in recs]
    if absent:
        raise Refused("inputs_do_not_cover_the_sample", {"n_absent": len(absent), "first": absent[:20]})
    if inputs_doc.get("split") != "navtrain":
        raise Refused("inputs_split_is_not_navtrain", {"split": inputs_doc.get("split")})
    bank, scratch = pathlib.Path(bank_dir), pathlib.Path(scratch_dir)
    if (bank / "DONE.json").exists():
        raise Refused("bank_already_done", {"bank": str(bank)})
    need: dict = {}
    for t in toks:
        for rel in token_rels(recs[t]):
            need.setdefault(rel, set()).add(t)
    scratch.mkdir(parents=True, exist_ok=True)
    extracted, ext_stats = extract_needed(sources, need, scratch)
    # ---- KB3 / B7: every needed jpg, or REFUSE.  No partial token, no skip, no stand-in. ----
    missing = sorted(set(need) - set(extracted))
    if missing and MUTATION != "missing_frame_standin":
        bad_tokens = sorted({t for r in missing for t in need[r]})
        raise Refused("missing_frames", {"n_missing_jpgs": len(missing), "n_tokens_affected": len(bad_tokens), "missing_first50": missing[:50],
                                         "tokens_first50": bad_tokens[:50], "rule": "KB3: 4 history frames x {CAM_L0, CAM_F0, CAM_R0} per token"})
    if missing:                                              # mutation: silently drop the affected tokens instead of refusing
        gone = {t for r in missing for t in need[r]}
        toks = [t for t in toks if t not in gone]
    # ---- unique frames, stitched once each ----
    frame_list, frame_row = [], {}
    for t in toks:
        for ft in recs[t]["frame_tokens"]:
            if ft not in frame_row:
                frame_row[ft] = len(frame_list)
                frame_list.append((recs[t]["log_name"], ft))
    bank.mkdir(parents=True, exist_ok=True)
    (bank / "rigs").mkdir(exist_ok=True)
    out_npy = bank / "frames_s00.npy"
    arr = np.lib.format.open_memmap(out_npy, mode="w+", dtype=np.uint8, shape=(max(len(frame_list), 1), H, W, 3))
    log_cache, rig_cache, rigs, frame_meta, fails = {}, {}, {}, {}, []

    def log_frames(ln):
        if ln not in log_cache:
            log_cache.clear()                                # one log pickle in RAM at a time
            fr, _stub = N.load_log_pickle(os.path.join(logs_root, f"{ln}.pkl"))
            log_cache[ln] = ({f["token"]: i for i, f in enumerate(fr)}, fr)
        return log_cache[ln]

    t1 = time.time()
    for k, (ln, ft) in enumerate(frame_list):
        try:
            idx, fr = log_frames(ln)
            cd = fr[idx[ft]]["cams"]
            one, src, rk, obs = e2.build_scene([cd], scratch, rig_cache)
            if one.shape != (1, H, W, 3) or one.dtype != np.uint8:
                raise ValueError(f"stitch returned {one.dtype}{one.shape}")
            mpx = float(one.mean())
            if mpx < 1.0 and MUTATION != "zero_frame_ok":
                raise ValueError(f"mean_px {mpx:.3f} < 1.0 (an all-black frame, KB2)")
            arr[k] = one[0]
            frame_meta[ft] = {"row": k, "rig_key": rk, "observed_frac": round(float(obs), 6), "mean_px": round(mpx, 2)}
            if rk not in rigs:
                np.save(bank / "rigs" / f"{rk}.src.npy", src)
                rigs[rk] = {"observed_frac": round(float((src >= 0).mean()), 6)}
        except Exception as e:                               # noqa: BLE001  (any frame failure is recorded, then refused below)
            fails.append({"frame_token": ft, "log": ln, "reason": f"{type(e).__name__}: {str(e)[:200]}"})
        if (k + 1) % 500 == 0:
            progress(f"[bank] {k + 1}/{len(frame_list)} frames, {(time.time() - t1) / (k + 1):.3f} s/frame")
    arr.flush()
    del arr
    if fails and MUTATION != "partial_on_stitch_failure":
        _clean(bank)
        raise Refused("frame_failed_to_stitch", {"n_failed": len(fails), "first": fails[:20]})
    index = {"frame": {"h": e2.FRAME.height, "w": e2.FRAME.width, "f_ref": e2.FRAME.f_ref, "projection": e2.FRAME.projection},
             "tokens": {}, "shards": {}, "rigs": rigs}
    mm = np.load(out_npy, mmap_mode="r")
    built = []
    for t in toks:
        fts = recs[t]["frame_tokens"]
        if not all(ft in frame_meta for ft in fts):
            continue                                         # only reachable under partial_on_stitch_failure
        rows = [frame_meta[ft]["row"] for ft in fts]
        if MUTATION == "rows_shifted":
            rows = [min(r + 1, mm.shape[0] - 1) for r in rows]
        stack = np.ascontiguousarray(mm[rows])
        index["tokens"][t] = {"shard": 0, "file": out_npy.name, "rows": rows, "log_name": recs[t]["log_name"],
                              "rig_key": frame_meta[fts[-1]]["rig_key"], "observed_frac": min(frame_meta[ft]["observed_frac"] for ft in fts),
                              "mean_px": round(float(stack.mean()), 2), "sha256": hashlib.sha256(stack.tobytes()).hexdigest()[:16]}
        built.append(t)
    index["shards"]["0"] = {"n_tokens_built": len(built), "n_frames": len(frame_list), "bank_file": out_npy.name, "sources": ext_stats}
    del mm
    json.dump(index, open(bank / "index.json", "w", encoding="utf-8"))
    rep = {"tool": "build_navtrain_frames.py", "sample_sha256_sorted_tokens": sample["sha256_sorted_tokens"], "n_sample_tokens": len(toks),
           "n_tokens_built": len(built), "n_frames": len(frame_list), "n_needed_jpgs": len(need), "n_jpgs_extracted": len(extracted),
           "n_frame_failures": len(fails), "sources": ext_stats, "bank_file": str(out_npy), "bank_bytes": os.path.getsize(out_npy),
           "frame": index["frame"], "wall_s": round(time.time() - t0, 1), "stitch_s": round(time.time() - t1, 1),
           "kb3": "every token: 4 history frames x {CAM_L0, CAM_F0, CAM_R0} extracted, else the build refuses (GROUNDWORK B7)"}
    if not keep_scratch:
        shutil.rmtree(scratch, ignore_errors=True)
        rep["scratch_deleted"] = not scratch.exists()
    # ---- round trip: read the bank the way BankNavtest416.load reads it; every token must re-derive ----
    problems = verify_bank(bank_dir, toks)
    if problems:
        raise Refused("bank_does_not_round_trip", {"problems": problems[:20], "n_problems": len(problems)})
    json.dump(rep, open(bank / "BUILD_REPORT.json", "w", encoding="utf-8"), indent=1)
    json.dump({"done": True, "n_tokens": len(built), "n_frames": len(frame_list), "index_sha256": N.sha256_file(str(bank / "index.json"))},
              open(bank / "DONE.json", "w", encoding="utf-8"))
    return rep


def verify_bank(bank_dir: str, tokens: list) -> list:
    """Re-read ``bank_dir`` exactly as ``refcv6_bridge.BankNavtest416`` does (``np.load(mmap)`` -> ``[rows]`` -> sha256[:16]); a list of problems."""
    problems = []
    index = json.load(open(os.path.join(bank_dir, "index.json"), encoding="utf-8"))
    prov = index["tokens"]
    if set(prov) != set(tokens):
        problems.append(f"token set: {len(set(prov) ^ set(tokens))} differ")
    mm = None
    for t in tokens:
        r = prov.get(t)
        if r is None:
            continue
        if mm is None:
            mm = np.load(os.path.join(bank_dir, r["file"]), mmap_mode="r")
        a = np.ascontiguousarray(mm[r["rows"]])
        if hashlib.sha256(a.tobytes()).hexdigest()[:16] != r["sha256"]:
            problems.append(f"{t}: sha256 does not re-derive")
        elif a.dtype != np.uint8 or a.shape != (4, H, W, 3) or float(a.mean()) < 1.0:
            problems.append(f"{t}: {a.dtype}{a.shape} mean {a.mean():.3f}")
    return problems


def _clean(bank: pathlib.Path):
    for p in ("index.json", "frames_s00.npy"):
        try:
            (bank / p).unlink()
        except OSError:
            pass


def load_stitcher(frames416_path: str):
    """E2's stitch configured for FRAME_416x1024, through refcv6's ``frames416.py`` (needs TANITAD_REPO + the stack on sys.path + torch)."""
    spec = importlib.util.spec_from_file_location("frames416", frames416_path)
    f4 = importlib.util.module_from_spec(spec)
    sys.modules["frames416"] = f4
    spec.loader.exec_module(f4)
    f4.assert_frame(f4.FRAME_416x1024)
    f4.configure(f4.FRAME_416x1024)
    return f4.E2BF


def main(argv=None) -> int:
    if MUTATION:
        sys.exit("[build_navtrain_frames] R8BANK_MUTATION is set -- a mutation-test environment never builds a bank")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--inputs", required=True, help="export_navtrain_inputs.py output (.json.gz)")
    ap.add_argument("--shards-dir", default="")
    ap.add_argument("--shard", action="append", default=[])
    ap.add_argument("--logs-root", required=True, help="the OpenScene trainval pickles (navsim_logs/trainval)")
    ap.add_argument("--bank", required=True)
    ap.add_argument("--scratch", default="")
    ap.add_argument("--frames416", required=True, help="refcv6 suite code/frames416.py (loads E2's build_frames.py)")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--keep-scratch", action="store_true")
    a = ap.parse_args(argv)
    import torch
    torch.set_num_threads(a.threads)
    sample = json.load(open(a.sample, encoding="utf-8"))
    opener = gzip.open if a.inputs.endswith(".gz") else open
    doc = json.load(opener(a.inputs, "rt", encoding="utf-8"))
    sources = list(a.shard) + (N.discover_shards(a.shards_dir) if a.shards_dir else [])
    if not sources:
        print("ZZBANK_REFUSED_no_shard_source", flush=True)
        return 3
    e2 = load_stitcher(a.frames416)
    scratch = a.scratch or os.path.join(a.bank, "_scratch")
    try:
        rep = build_bank(doc, sample, sources, a.logs_root, a.bank, scratch, e2, keep_scratch=a.keep_scratch)
    except Refused as r:
        os.makedirs(a.bank, exist_ok=True)
        json.dump({"refused": r.why, "detail": r.detail}, open(os.path.join(a.bank, "REFUSED.json"), "w", encoding="utf-8"), indent=1)
        print(f"ZZBANK_REFUSED_{r.why}", flush=True)
        return 3
    print(f"ZZBANK_DONE tokens={rep['n_tokens_built']} frames={rep['n_frames']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
