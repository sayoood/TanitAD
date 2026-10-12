#!/usr/bin/env python3
"""RL-S1 priority 1: the FROZEN-FAN dumper -- a FIXED checkpoint's 64 proposals over a GIVEN key list, written DIRECTLY as `onpolicy_set` records.

    python refe/rl_fan_dump.py --ckpt <model_final.pt> --keys rl_train_scenes.tsv rl_val_scenes.tsv --partition-manifest <manifest.json> \\
        --targets <train_grow> --images <navtrain_pixels> --calib <calib_table.json> --out <dir> [--shard I/N] [--ranks 0] [--batch 4]

WHY A SIBLING OF `onpolicy_dump.py` AND NOT AN EDIT OF IT.  `onpolicy_dump.py` is called by `code/op_pipeline.sh` and is the producer behind the live
on-policy chain (the watcher at its line 104, the `--ranks 0,1` loop, the backpressure on `onpolicy_label.py`).  Four things RL-S1 needs would change
its behaviour for those callers: a key list (F1), a FIXED checkpoint file instead of a live run dir (F2), a different OUTPUT KIND (F3: the old one writes
`onpolicy_props` queue chunks which only the teacher simulator `onpolicy_label.py` turns into `onpolicy_set`; 10.2 s per set = 144 worker-hours for
51,000), and a `targets` placeholder (F4).  This file is additive: the old dumper is untouched, byte for byte.

NUMERICS ARE THE OLD DUMPER'S, LINE FOR LINE (the point of the bit-identity requirement): the trainer's own input construction
(`train.TargetBank.__getitem__`: images, ego, goal, per-sample rig), `torch.set_num_threads(1)`, TF32 on, eval mode, no grad, bf16 autocast, batch 4, the
`calib` argument only when the rig tensor is non-empty, `traj.float().cpu()`, and the SAME two-step rounding as dump -> `onpolicy_label.py`
(5 decimals, then 5 for x / y and 6 for yaw).  `eval/rl_fan_identity.py` runs this against the old dumper and against the banked step-10,075 sets.

THE OUTPUT -- one file per shard, `<out>/onpolicy_rlfan_<tag>_s<i>of<n>.jsonl` (the `onpolicy_*.jsonl` glob that `onpolicy_relabel_pdm.py`,
`rl_terms_label.py` and `train.OnPolicyBank` all read), one line per (scene, rank):
    {"kind": "onpolicy_set", "label_version": 0, "comfort_source": "none", "log_name", "token", "step", "rank", "ckpt_step", "aug",
     "traj": [M][T][2], "yaw": [M][T], "targets": [{}] * M, "teacher_targets": {}, "labeller": "rl_fan_dump/1", "lean": true, "fan_ckpt_sha256": "<16 hex>"}
The key fields come FIRST (the PDM labeller's `--served-only` reads the head and tail of a line only).  `targets` are EMPTY PLACEHOLDERS:
`ScorerBank.components({})` is defined (a perfect all-ones row) and `OnPolicyBank._apply_pdm_labels` then OVERWRITES all six components, so these sets are
meaningful ONLY with PDM labels: a trainer MUST run with `--require-pdm` / `pdm_only` (a flag whose default disables it has to be NAMED in the argv).
`label_version` 0 so a lean set never supersedes a teacher-labelled set of the same key and checkpoint.  Records carry no timestamp: the output is a pure
function of (checkpoint, keys, bank, pixels), so a re-run reproduces the file byte for byte.

RL-S1 RUNS IT TWICE, once per side of the partition (`--keys rl_train_scenes.tsv --out <sets_train>` and `--keys rl_val_scenes.tsv --out <sets_val>`), because
`scorer_finetune.py` reads `--onpolicy` and `--heldout` as two directories and `--heldout-logs` as the val side's log list (`cut -f2 rl_val_scenes.tsv | sort -u`).
Both lists may also be given together (one dir) when a single pass is wanted; the partition manifest check then covers both files.

SHARDING is by LOG (`zlib.crc32(log) % n`, the same function `rl_terms_label.shard_of` uses), so dump shard i and label shard i hold the same logs and
the shards are disjoint and exhaustive.  Scenes are dumped in sorted (log, token) order, so batch composition is deterministic (and the numerics do not
depend on it: `eval/rl_fan_identity.py` measures that).

EXIT CODES: 0 ok - 3 scene(s) with no bank row / unresolved image under --pixels (written to <out>/unavailable_scenes_<tag>.tsv; see
`refe/rl_pool_available.py` for the one admissible remedy) - 4 refused (bad arguments, checkpoint / manifest mismatch) - 5 `--compare-sets` found a set
that is NOT bit-identical (default), or -- with `--compare-atol X` -- a set whose max |delta| over every x / y / yaw value exceeds X.
Markers: `ZZRLFAN_PLAN <n>`, `ZZRLFAN_DONE <written> <skipped_done> <missing>`, and with `--compare-atol`: `ZZRLFAN_ATOL_PASS <matched> <max_delta> <atol>` or
`ZZRLFAN_ATOL_FAIL <matched> <beyond> <max_delta> <atol>`.

`--compare-atol X` (RL-S1 pod gate, 2026-10-12): bit-identity holds only at matched batching on ONE GPU.  `eval/rl_fan_identity.py` MEASURED (RTX 4060) that BOTH
dumpers move by up to 0.25 when the batch grouping changes, and that dev box vs the A40's banked step-10,075 sets differ by max abs 0.25 on every key; a
bit-identical gate would REFUSE a correct dumper on the pod.  With X the gate is an ABSOLUTE tolerance in trajectory units (metres for x / y, radians for yaw, one
number for all three), judged per set as `max |delta| <= X`.  X is chosen from the measured floor and the deliberate-regression arms in
`raw/rl_fan/atol_calibration/`, never assumed.  Default (no flag): today's bit-identical gate, unchanged.
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import hashlib
import io
import json
import math
import os
import sys
import time
import zlib
from pathlib import Path

TOOL = "rl_fan_dump/1"
LABEL_VERSION = 0
RANK_DEFAULT = "0"
EXIT_OK, EXIT_MISSING, EXIT_REFUSED, EXIT_MISMATCH = 0, 3, 4, 5

HERE = Path(__file__).resolve().parent


# ------------------------------------------------------------------------------------------------------------- key lists
def _open_text(path):
    if str(path).endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", newline="")
    return open(path, encoding="utf-8", newline="")


def sha256_bytes(path) -> str:
    """sha256 of a file's bytes (a .gz is hashed DECOMPRESSED, so it compares with the plain file's digest)."""
    h = hashlib.sha256()
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def read_scenes(path):
    """A `token<TAB>log` TSV (rl_partition.py's output) -> [(token, log)].  A malformed or duplicated row is an error, never skipped."""
    out, seen = [], set()
    with _open_text(path) as f:
        head = f.readline().rstrip("\r\n").split("\t")
        if head[:2] != ["token", "log"]:
            raise ValueError(f"{path}: first line must be the header 'token<TAB>log', got {head!r}")
        for n, ln in enumerate(f, 2):
            ln = ln.rstrip("\r\n")
            if not ln:
                continue
            p = ln.split("\t")
            if len(p) < 2 or not p[0] or not p[1]:
                raise ValueError(f"{path}:{n}: need token<TAB>log, got {ln!r}")
            if p[0] in seen:
                raise ValueError(f"{path}:{n}: token {p[0]} listed twice (a token belongs to exactly one scene)")
            seen.add(p[0])
            out.append((p[0], p[1]))
    return out


def verify_scene_files(paths, manifest_path):
    """Every key file must be byte-identical to the one the frozen partition manifest names (sha256).  Returns {partition_id, files: {name: sha}}."""
    m = json.load(open(manifest_path, encoding="utf-8"))
    if m.get("verify") != "clean":
        raise ValueError(f"{manifest_path}: the partition manifest's own verify is {m.get('verify')!r}, not 'clean'")
    want = {m["train"]["file"]: m["train"]["sha256"], m["val"]["file"]: m["val"]["sha256"]}
    got = {}
    for p in paths:
        name = os.path.basename(str(p))
        name = name[:-3] if name.endswith(".gz") else name
        if name not in want:
            raise ValueError(f"{p}: {name} is not one of the partition manifest's files {sorted(want)}")
        sha = sha256_bytes(p)
        if sha != want[name]:
            raise ValueError(f"{p}: sha256 {sha[:16]} != the manifest's {want[name][:16]} -- this is NOT the frozen partition")
        got[name] = sha
    return {"partition_id": m.get("partition_id"), "files": got}


# ------------------------------------------------------------------------------------------------------------- sharding / plan
def shard_of(log: str, n: int) -> int:
    """Log-disjoint, deterministic sharding.  Identical to rl_terms_label.shard_of (a test pins that), so dump shard i and label shard i see the same logs."""
    return zlib.crc32(log.encode("utf-8")) % n


def parse_shard(s: str):
    try:
        i, n = (int(x) for x in s.split("/"))
    except Exception:
        raise ValueError(f"--shard {s!r}: need I/N")
    if not (n >= 1 and 0 <= i < n):
        raise ValueError(f"--shard {s}: need 0 <= I < N")
    return i, n


def plan(rows, scenes, ranks, shard):
    """rows: the bank's rows (dicts with log_name / token / rank).  -> (indices, missing, log_mismatch), indices in sorted (log, token, rank) order.

    missing:      (token, log, rank) with no bank row at all
    log_mismatch: (token, log, rank) whose token IS in the bank, under a different log -- a defect in the key list, never a coverage gap
    """
    si, sn = shard
    by, logs_of = {}, collections.defaultdict(set)
    for i, r in enumerate(rows):
        k = (r.get("log_name", ""), r.get("token", ""), int(r.get("rank", 0)))
        if k in by:
            raise ValueError(f"the bank holds {k} twice (row {by[k]} and {i}); refusing to pick one")
        by[k] = i
        logs_of[r.get("token", "")].add(r.get("log_name", ""))
    out, missing, mism = [], [], []
    for token, log in sorted(set(scenes), key=lambda s: (s[1], s[0])):
        if shard_of(log, sn) != si:
            continue
        for rk in ranks:
            i = by.get((log, token, rk))
            if i is not None:
                out.append(i)
            elif token in logs_of and log not in logs_of[token]:
                mism.append((token, log, rk))
            else:
                missing.append((token, log, rk))
    return out, missing, mism


# ------------------------------------------------------------------------------------------------------------- the record
def _r5(v) -> float:
    return round(float(v), 5)


def set_record(row, rank, ckpt_step, props, fan_sha16="", n_prop=64, horizon=20):
    """props: array-like [M][T][3] (x, y, yaw) -> the `onpolicy_set` line (a dict).  The rounding is dump -> label's, literally."""
    if len(props) != n_prop or any(len(p) != horizon for p in props):
        raise ValueError(f"{row.get('log_name')} {row.get('token')}: fan is {len(props)} x {[len(p) for p in props][:2]}.. , need {n_prop} x {horizon}")
    traj = [[[round(_r5(p[0]), 5), round(_r5(p[1]), 5)] for p in prop] for prop in props]
    yaw = [[round(_r5(p[2]), 6) for p in prop] for prop in props]
    return {"kind": "onpolicy_set", "label_version": LABEL_VERSION, "comfort_source": "none",
            "log_name": row["log_name"], "token": row["token"], "step": int(row.get("step", 0)), "rank": int(rank),
            "ckpt_step": int(ckpt_step), "aug": row.get("aug"),
            "traj": traj, "yaw": yaw, "targets": [{} for _ in range(n_prop)], "teacher_targets": {},
            "labeller": TOOL, "lean": True, "fan_ckpt_sha256": fan_sha16}


def key_of(rec):
    return (rec["log_name"], rec["token"], int(rec["step"]), int(rec.get("rank", 0)))


# ------------------------------------------------------------------------------------------------------------- resume
def read_done(path, ckpt_step):
    """Keys already written to this shard file.  A torn last line is ignored; a line of ANOTHER checkpoint step is an error (never mix fans)."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8", newline="") as f:
        for ln in f:
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if r.get("kind") != "onpolicy_set":
                continue
            if int(r["ckpt_step"]) != int(ckpt_step):
                raise ValueError(f"{path} already holds sets of ckpt_step {r['ckpt_step']}, this run is {ckpt_step}: use a fresh --out")
            done.add(key_of(r))
    return done


def _terminate_torn_line(path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                with open(path, "ab") as g:
                    g.write(b"\n")


# ------------------------------------------------------------------------------------------------------------- model + forward (the old dumper's, verbatim)
def ckpt_step_of(path, meta):
    """FULL checkpoints carry meta.step; PARTIAL ones carry state.step."""
    if isinstance(meta.get("step"), int):
        return int(meta["step"])
    import torch
    sd = torch.load(path, map_location="cpu", weights_only=False)
    return int(sd["state"]["step"])


def load_model(ckpt, backbone, device, *, build=None, loader=None):
    """-> (model on `device`, eval mode, meta dict).  `build()` / `loader(model, path, meta_out)` are the test seams; the defaults are
    REFe(cfg) and ckpt_io.load_for_inference -- the same pair planner.py and scorer_finetune.py use."""
    if build is None or loader is None:
        sys.path.insert(0, str(HERE))
        import ckpt_io
        from model import REFe, REFeConfig
        cfg = ckpt_io.config_for_checkpoint(REFeConfig.for_backbone(backbone), ckpt)
        build = build or (lambda: REFe(cfg))
        loader = loader or (lambda m, p, meta_out: ckpt_io.load_for_inference(m, p, map_location="cpu", backbone=backbone, meta_out=meta_out))
    model = build()
    meta: dict = {}
    loader(model, ckpt, meta)
    return model.to(device).eval(), meta


def forward_batch(model, items, device, amp=True):
    """items: ds[i] tuples -> traj [B, M, T, 3] float32 numpy.  onpolicy_dump.py lines 128-136."""
    import torch
    img = torch.stack([it[0] for it in items]).to(device, non_blocking=True)
    ego = torch.stack([it[1] for it in items]).to(device, non_blocking=True)
    goal = torch.stack([it[2] for it in items]).to(device, non_blocking=True)
    cal = torch.stack([it[8] for it in items]) if items[0][8].numel() else None
    if amp:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            traj, _score = model(img, ego, goal, calib=cal)
    else:
        with torch.no_grad():
            traj, _score = model(img, ego, goal, calib=cal)
    return traj.float().cpu().numpy()


def iter_items(ds, idxs, workers):
    """ds[i] for i in idxs, in order.  workers > 0 prefetches with a bounded thread pool (cv2 decode releases the GIL); the VALUES are identical."""
    if workers <= 0:
        for i in idxs:
            yield ds[i]
        return
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(workers) as ex:
        q, it = collections.deque(), iter(idxs)
        for _ in range(workers * 2):
            i = next(it, None)
            if i is None:
                break
            q.append(ex.submit(ds.__getitem__, i))
        while q:
            item = q.popleft().result()
            i = next(it, None)
            if i is not None:
                q.append(ex.submit(ds.__getitem__, i))
            yield item


def dump(model, ds, order, *, out_file, done, ckpt_step, n_prop, horizon, batch, chunk, device, amp, workers, fan_sha16, log=print):
    """Dump `order` (row indices) not already in `done`, appending whole chunks to out_file.  -> stats dict."""
    todo = [i for i in order if (ds.rows[i]["log_name"], ds.rows[i]["token"], int(ds.rows[i].get("step", 0)), int(ds.rows[i].get("rank", 0))) not in done]
    st = {"todo": len(todo), "written": 0, "sec_forward": 0.0, "sec_total": 0.0}
    _terminate_torn_line(out_file)
    t_all = time.time()
    buf, t_chunk = [], time.time()
    stream = iter_items(ds, todo, workers)
    n = 0
    with open(out_file, "a", encoding="utf-8", newline="\n") as fh:
        while n < len(todo):
            idx = todo[n:n + batch]
            items = [next(stream) for _ in idx]
            t0 = time.time()
            traj = forward_batch(model, items, device, amp)
            st["sec_forward"] += time.time() - t0
            if traj.shape[0] != len(idx) or traj.shape[1:] != (n_prop, horizon, 3):
                raise RuntimeError(f"model returned {traj.shape}, need [{len(idx)}, {n_prop}, {horizon}, 3]")
            for j, i in enumerate(idx):
                row = ds.rows[i]
                buf.append(json.dumps(set_record(row, int(row.get("rank", 0)), ckpt_step, traj[j], fan_sha16, n_prop, horizon)) + "\n")
            n += len(idx)
            if len(buf) >= chunk or n >= len(todo):
                fh.write("".join(buf))
                fh.flush()
                os.fsync(fh.fileno())
                st["written"] += len(buf)
                log(f"  {n:,}/{len(todo):,} sets  ({(time.time() - t_chunk) / max(len(buf), 1):.3f} s/set this chunk)")
                buf, t_chunk = [], time.time()
    st["sec_total"] = time.time() - t_all
    return st


# ------------------------------------------------------------------------------------------------------------- bit-identity against existing sets
def set_delta_strict(a, b):
    """-> (d_xy, d_yaw): max |delta| over x / y values and over yaw values of two (traj, yaw) pairs.  Unlike the legacy loop in `compare_sets` (zip truncates), a SHAPE
    mismatch (a missing proposal / step) or a NON-FINITE value on either side is `inf`: a fan with a dropped proposal, or a NaN, must never read as 'within tolerance'
    (`max(0.0, nan)` is 0.0 in Python, which is how a NaN fan would otherwise pass)."""
    (ta, ya), (tb, yb) = a, b
    if len(ta) != len(tb) or len(ya) != len(yb):
        return math.inf, math.inf
    dxy = dyaw = 0.0
    for pa, pb in zip(ta, tb):
        if len(pa) != len(pb):
            return math.inf, math.inf
        for qa, qb in zip(pa, pb):
            if len(qa) != len(qb):
                return math.inf, math.inf
            for u, v in zip(qa, qb):
                d = abs(u - v)
                dxy = max(dxy, d if math.isfinite(d) else math.inf)
    for pa, pb in zip(ya, yb):
        if len(pa) != len(pb):
            return math.inf, math.inf
        for u, v in zip(pa, pb):
            d = abs(u - v)
            dyaw = max(dyaw, d if math.isfinite(d) else math.inf)
    return dxy, dyaw


def quantile_nearest_rank(sorted_vals, q):
    """Nearest-rank quantile (the smallest value with at least a fraction q of the sample at or below it); q in (0, 1].  Empty -> 0.0."""
    if not sorted_vals:
        return 0.0
    return sorted_vals[max(0, math.ceil(q * len(sorted_vals)) - 1)]


def compare_sets(new_path, ref_paths, ckpt_step, atol=None):
    """Compare this run's sets with EXISTING sets of the same keys and ckpt_step (e.g. the step-10,075 sets of the main run).  Exact list equality of traj
    and yaw (both sides are 5/6-decimal rounded floats).  -> dict with counts, max |delta| over unequal keys, and the first unequal keys.
    Every reference line of a key must also agree with the other reference lines of that key (a relabelled set keeps its proposals).

    atol=None: the result is exactly the bit-identity report above (no extra keys).  atol=X (a finite float >= 0): the report ALSO carries `atol`, `within_atol`
    (sets with max |delta| <= X, bit-identical ones included), `beyond_atol` (> X, strict; a shape mismatch or non-finite value is always beyond), `max_abs_delta`
    (over every matched set), `max_abs_delta_xy` / `max_abs_delta_yaw`, `delta_quantiles` (nearest-rank p50 / p90 / p99 / max of the per-set max delta over ALL
    matched sets) and `first_beyond_atol`.  `not_identical` and the reference-disagreement count keep their exact meaning."""
    new = {}
    with open(new_path, encoding="utf-8", newline="") as f:
        for ln in f:
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if r.get("kind") == "onpolicy_set" and int(r["ckpt_step"]) == int(ckpt_step):
                new[key_of(r)] = (r["traj"], r["yaw"])
    tokens = {k[1] for k in new}
    marker = f'"ckpt_step": {int(ckpt_step)}'.encode()
    ref = {}
    ref_lines = 0
    ref_disagree = []
    for p in ref_paths:
        with open(p, "rb") as f:
            for ln in f:
                if marker not in ln:
                    continue
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if r.get("kind") != "onpolicy_set" or int(r["ckpt_step"]) != int(ckpt_step) or r["token"] not in tokens:
                    continue
                k = key_of(r)
                if k not in new:
                    continue
                ref_lines += 1
                v = (r["traj"], r["yaw"])
                if k in ref and ref[k] != v:
                    ref_disagree.append(k)
                ref.setdefault(k, v)
    n_equal, unequal, max_abs = 0, [], 0.0
    per_set = []                                                        # atol mode: (key, max |delta| over x / y / yaw, d_xy, d_yaw) for EVERY matched set
    for k, v in ref.items():
        if new[k] == v:
            n_equal += 1
            if atol is not None:
                per_set.append((k, 0.0, 0.0, 0.0))
            continue
        d = 0.0
        for (a_t, a_y), (b_t, b_y) in ((new[k], v),):
            for pa, pb in zip(a_t, b_t):
                for qa, qb in zip(pa, pb):
                    d = max(d, abs(qa[0] - qb[0]), abs(qa[1] - qb[1]))
            for pa, pb in zip(a_y, b_y):
                for qa, qb in zip(pa, pb):
                    d = max(d, abs(qa - qb))
        unequal.append((k, d))
        max_abs = max(max_abs, d)
        if atol is not None:
            sxy, syaw = set_delta_strict(new[k], v)
            per_set.append((k, max(sxy, syaw), sxy, syaw))
    out = {"new_sets": len(new), "reference_keys_matched": len(ref), "reference_lines_read": ref_lines, "bit_identical": n_equal,
           "not_identical": len(unequal), "max_abs_delta_over_unequal": max_abs, "first_unequal": [[list(k), d] for k, d in unequal[:5]],
           "reference_lines_disagreeing_with_each_other": len(ref_disagree)}
    if atol is not None:
        beyond = sorted(((k, d) for k, d, _, _ in per_set if d > atol), key=lambda kd: -kd[1])
        ds = sorted(d for _, d, _, _ in per_set)
        out.update({"atol": atol, "within_atol": len(per_set) - len(beyond), "beyond_atol": len(beyond),
                    "max_abs_delta": ds[-1] if ds else 0.0,
                    "max_abs_delta_xy": max((x for _, _, x, _ in per_set), default=0.0), "max_abs_delta_yaw": max((y for _, _, _, y in per_set), default=0.0),
                    "delta_quantiles": {"p50": quantile_nearest_rank(ds, 0.5), "p90": quantile_nearest_rank(ds, 0.9), "p99": quantile_nearest_rank(ds, 0.99),
                                        "max": ds[-1] if ds else 0.0},
                    "first_beyond_atol": [[list(k), d] for k, d in beyond[:5]]})
    return out


# ------------------------------------------------------------------------------------------------------------- CLI
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", required=True, help="a FIXED checkpoint file (FULL model_final.pt or PARTIAL ckpt_last.pt); never a run dir, never reloaded")
    ap.add_argument("--ckpt-step", type=int, default=None, help="the sets' ckpt_step; default = the file's own (meta.step / state.step); a given value must AGREE with the file")
    ap.add_argument("--expect-ckpt-sha256", default="", help="refuse unless the checkpoint FILE has this sha256 (model_final.pt: MODEL_REGISTRY)")
    ap.add_argument("--keys", nargs="+", required=True, help="token<TAB>log scene lists (rl_partition.py output)")
    ap.add_argument("--partition-manifest", default="", help="the frozen partition manifest: every --keys file must match its sha256")
    ap.add_argument("--targets", required=True, help="the training bank dir (targets_rank*.jsonl): train_grow")
    ap.add_argument("--images", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="fan", help="file-name tag; one --out dir per fan")
    ap.add_argument("--backbone", default="vitl16")
    ap.add_argument("--shard", default="0/1", help="I/N: scenes whose LOG hashes to I (crc32 % N)")
    ap.add_argument("--ranks", default=RANK_DEFAULT, help="bank ranks to dump; the RL fan is rank 0 (the plain view)")
    ap.add_argument("--batch", type=int, default=4, help="4 = onpolicy_dump.py's batch; the numerics were measured at 4")
    ap.add_argument("--chunk", type=int, default=64, help="sets per fsync'd write")
    ap.add_argument("--workers", type=int, default=0, help="threads that decode the next samples while the GPU runs (0 = the old dumper's serial loop)")
    ap.add_argument("--threads", type=int, default=1, help="torch.set_num_threads (the old dumper: 1)")
    ap.add_argument("--limit", type=int, default=0, help="dump only the first N planned sets (smoke)")
    ap.add_argument("--plan-only", action="store_true", help="resolve the bank + pixels and report coverage; no model, no GPU")
    ap.add_argument("--allow-missing", action="store_true", help="exit 0 although some scenes have no bank row / image (they are still listed)")
    ap.add_argument("--compare-sets", nargs="*", default=None,
                    help="existing onpolicy_*.jsonl files or dirs: after the dump, EVERY dumped key that also has a set there at this ckpt_step must be bit-identical (exit 5 otherwise)")
    ap.add_argument("--compare-atol", type=float, default=None,
                    help="with --compare-sets: judge each set by max |delta| over every x / y / yaw value <= ATOL (absolute, trajectory units) instead of bit-identity; exit 5 iff any set "
                         "is beyond it.  Default None = the bit-identical gate, unchanged.  Choose ATOL from raw/rl_fan/atol_calibration/ (a measured floor and regression arms), never by hand")
    ap.add_argument("--compare-min-matched", type=int, default=None,
                    help="with --compare-sets: REFUSE (exit 4) when fewer than this many dumped keys have a reference set (a gate that compares 3 of 512 sets proves little); "
                         "matching NOTHING is always refused")
    ap.add_argument("--device", default="cuda")
    return ap


def run(a, *, build=None, loader=None, ds_factory=None):
    """The whole job.  `build` / `loader` / `ds_factory` are the test seams (defaults: REFe, ckpt_io.load_for_inference, train.TargetBank)."""
    try:
        shard = parse_shard(a.shard)
        ranks = [int(x) for x in a.ranks.split(",") if x != ""]
        if not ranks:
            raise ValueError("--ranks is empty")
        key_files = list(a.keys)
        part = verify_scene_files(key_files, a.partition_manifest) if a.partition_manifest else None
        scenes = []
        for p in key_files:
            scenes += read_scenes(p)
        if len({t for t, _ in scenes}) != len(scenes):
            raise ValueError("a token appears in two key files (train and val must be disjoint)")
        if (a.compare_atol is not None or a.compare_min_matched is not None) and a.compare_sets is None:
            raise ValueError("--compare-atol / --compare-min-matched need --compare-sets (a tolerance with nothing to compare against would be silently ignored)")
        if a.compare_atol is not None and not (math.isfinite(a.compare_atol) and a.compare_atol >= 0):
            raise ValueError(f"--compare-atol {a.compare_atol!r}: need a finite number >= 0")
        if a.compare_min_matched is not None and a.compare_min_matched < 1:
            raise ValueError(f"--compare-min-matched {a.compare_min_matched}: need >= 1")
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as e:
        print(f"REFUSED: {e}")
        return EXIT_REFUSED
    os.makedirs(a.out, exist_ok=True)
    sn_tag = f"{a.tag}_s{shard[0]}of{shard[1]}"
    out_file = os.path.join(a.out, f"onpolicy_rlfan_{sn_tag}.jsonl")
    man_file = os.path.join(a.out, f"rl_fan_dump_{sn_tag}.manifest.json")
    unavail_file = os.path.join(a.out, f"unavailable_scenes_{sn_tag}.tsv")
    t_start = time.time()

    # ---- the bank, with the trainer's own image / rig resolution
    sys.path.insert(0, str(HERE))
    if ds_factory is None:
        from model import REFeConfig
        import train as T
        cfg = REFeConfig.for_backbone(a.backbone)
        n_prop, horizon = cfg.n_proposals, cfg.horizon_steps
        # max_missing_frac=1.0: bank rows OUTSIDE the key list are none of this tool's business; the key list's own gaps are plan()'s `missing` (exit 3)
        try:
            ds = T.TargetBank(a.targets, a.images, cfg, calib=a.calib, max_missing_frac=1.0)
        except RuntimeError as e:
            print(f"REFUSED: {e}")
            return EXIT_REFUSED
    else:
        ds, n_prop, horizon = ds_factory(a)
    idx, missing, mism = plan(ds.rows, scenes, ranks, shard)
    if mism:
        print(f"REFUSED: {len(mism)} scene(s) name a token that the bank holds under a DIFFERENT log, e.g. {mism[:2]} -- the key list does not belong to this bank")
        return EXIT_REFUSED
    in_shard = [s for s in set(scenes) if shard_of(s[1], shard[1]) == shard[0]]
    print(f"  keys: {len(scenes):,} scenes in the lists, {len(in_shard):,} in shard {a.shard}; ranks {ranks}; planned {len(idx):,} sets; "
          f"{len(missing):,} with no bank row / unresolved image", flush=True)
    print(f"ZZRLFAN_PLAN {len(idx)}", flush=True)
    if missing:
        with open(unavail_file, "w", encoding="utf-8", newline="\n") as f:
            f.write("token\tlog\trank\n" + "".join(f"{t}\t{l}\t{r}\n" for t, l, r in missing))
        print(f"  unavailable scenes -> {unavail_file}", flush=True)
    if a.limit:
        idx = idx[:a.limit]
    manifest = {"tool": TOOL, "tag": a.tag, "shard": a.shard, "ranks": ranks, "partition": part, "key_files": {os.path.basename(p): sha256_bytes(p) for p in key_files},
                "scenes_in_lists": len(scenes), "scenes_in_shard": len(in_shard), "planned": len(idx), "unavailable": len(missing),
                "bank_rows_dropped_unresolved_image": getattr(ds, "n_dropped_missing", None), "bank_rows_dropped_no_rig": getattr(ds, "n_dropped_calib", None),
                "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "argv": sys.argv[1:], "plan_only": bool(a.plan_only)}
    if a.plan_only:
        manifest["finished"] = True
        _write_json(man_file, manifest)
        print(f"ZZRLFAN_DONE 0 0 {len(missing)}", flush=True)
        return EXIT_MISSING if (missing and not a.allow_missing) else EXIT_OK

    # ---- the model: the FILE named by --ckpt, loaded once, never reloaded
    import torch
    torch.set_num_threads(a.threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    ck_sha = sha256_bytes(a.ckpt)
    if a.expect_ckpt_sha256 and ck_sha != a.expect_ckpt_sha256.lower():
        print(f"REFUSED: {a.ckpt} has sha256 {ck_sha[:16]}.., expected {a.expect_ckpt_sha256[:16]}..")
        return EXIT_REFUSED
    model, meta = load_model(a.ckpt, a.backbone, a.device, build=build, loader=loader)
    step = ckpt_step_of(a.ckpt, meta)
    if a.ckpt_step is not None and a.ckpt_step != step:
        print(f"REFUSED: --ckpt-step {a.ckpt_step} but {a.ckpt} is step {step}")
        return EXIT_REFUSED
    print(f"  model: {a.ckpt} step {step} sha256 {ck_sha[:16]} (per_sample_calib {meta.get('per_sample_calib')})", flush=True)
    manifest.update({"ckpt": os.path.abspath(a.ckpt), "ckpt_sha256": ck_sha, "ckpt_step": step, "ckpt_meta": {k: meta.get(k) for k in ("backbone", "step", "epoch", "frozen_sha256", "per_sample_calib")},
                     "torch": torch.__version__, "device": a.device,
                     "gpu": torch.cuda.get_device_name(0) if a.device.startswith("cuda") and torch.cuda.is_available() else None})

    done = read_done(out_file, step)
    print(f"  {len(done):,} sets already in {os.path.basename(out_file)}", flush=True)
    amp = a.device.startswith("cuda")
    st = dump(model, ds, idx, out_file=out_file, done=done, ckpt_step=step, n_prop=n_prop, horizon=horizon,
              batch=a.batch, chunk=a.chunk, device=a.device, amp=amp, workers=a.workers, fan_sha16=ck_sha[:16])
    manifest.update({"already_done": len(done), "written": st["written"], "sec_total": round(st["sec_total"], 2), "sec_forward": round(st["sec_forward"], 2),
                     "s_per_set_total": round(st["sec_total"] / max(st["written"], 1), 4), "s_per_set_forward": round(st["sec_forward"] / max(st["written"], 1), 4),
                     "startup_sec": round(time.time() - t_start - st["sec_total"], 2), "peak_gpu_gb": (round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if amp and torch.cuda.is_available() else None)})
    rc = EXIT_OK
    if a.compare_sets is not None:
        refs = []
        for p in a.compare_sets:
            refs += sorted(glob.glob(os.path.join(p, "onpolicy_*.jsonl"))) if os.path.isdir(p) else [p]
        cmp_ = compare_sets(out_file, refs, step, atol=a.compare_atol)
        manifest["compare_sets"] = cmp_
        print("  compare-sets: " + json.dumps(cmp_), flush=True)
        if cmp_["reference_keys_matched"] == 0:
            print("REFUSED: --compare-sets matched NO key -- a gate that compares nothing proves nothing")
            rc = EXIT_REFUSED
        elif a.compare_min_matched is not None and cmp_["reference_keys_matched"] < a.compare_min_matched:
            print(f"REFUSED: --compare-sets matched {cmp_['reference_keys_matched']} key(s), fewer than --compare-min-matched {a.compare_min_matched}")
            rc = EXIT_REFUSED
        elif a.compare_atol is not None:
            if cmp_["beyond_atol"] or cmp_["reference_lines_disagreeing_with_each_other"]:
                rc = EXIT_MISMATCH
                print(f"ZZRLFAN_ATOL_FAIL {cmp_['reference_keys_matched']} {cmp_['beyond_atol']} {cmp_['max_abs_delta']:.6g} {a.compare_atol:g}", flush=True)
            else:
                print(f"ZZRLFAN_ATOL_PASS {cmp_['reference_keys_matched']} {cmp_['max_abs_delta']:.6g} {a.compare_atol:g}", flush=True)
        elif cmp_["not_identical"] or cmp_["reference_lines_disagreeing_with_each_other"]:
            rc = EXIT_MISMATCH
    manifest["finished"] = True
    _write_json(man_file, manifest)
    print(f"ZZRLFAN_DONE {st['written']} {len(done)} {len(missing)}", flush=True)
    if rc != EXIT_OK:
        return rc
    return EXIT_MISSING if (missing and not a.allow_missing) else EXIT_OK


def _write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def main(argv=None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
