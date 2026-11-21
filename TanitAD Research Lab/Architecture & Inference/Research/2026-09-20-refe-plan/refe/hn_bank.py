#!/usr/bin/env python3
"""P6-HN-S: the offset hard-negative BANK (builder + file + SY yield) and the hard-negative LOSS TERM, in ONE module that both trainers import.
(raw/2026-10-18-p6hn-prereg/PREREG_P6HN_SCREEN.md sections 2, 4.3, 5; PREREG_P6HN.draft.amendmentA.md; SCREEN_NOTES.md checklist item 1.)

WHAT IT IS.  Two things that must stay in one file: the bank (so the label path cannot drift from the fan's) and `hn_term` (so the screen exercises exactly the
term the full P6-HN arm would; amendment A makes the draft adopt it).  Nothing here touches the trainer, the gate or the launch profile (checklist items 3, 4).

THE BANK (section 2).  Per anchor (a scene's teacher plan, 20 poses x (x, y, yaw), 4 s):
  * STATIONARY rule: path length (sum of consecutive xy steps STARTING FROM THE ORIGIN) < 2.0 m  ->  no offsets, counted (PREREG_HN_CENSUS 3.6).
  * 8 offsets: pose k shifted by delta along the anchor's unit LEFT-normal at pose k, delta in {-0.60,-0.45,-0.30,-0.15,+0.15,+0.30,+0.45,+0.60} m.  The normal is
    (-sin yaw, cos yaw) from the stored yaw, else from a central-difference tangent.  Heading and time (the pose index) are kept.  NO MAP GEOMETRY is read to build or
    choose a negative (gate I2 scans this file's call graph for a map API, with a control that must find one in the labeller).
  * 9 labelled trajectories per scene (anchor + 8 offsets), labelled by `onpolicy_relabel_pdm.label_log` ITSELF (resolved by reference in `labeller()`, never a copy;
    gate I1 verifies it bit for bit on the fan with `label_identity_check`), label_version 5, A7 repair ON (inside the labeller).
  * hard negative = (NC == 0 or DAC == 0) and PDMS(offset) == 0 and PDMS(anchor) > 0.  NC = 0.5 is not one.  Sub-types: 1 DAC-only, 2 NC-only, 3 both.
    `hard_negative_rule` is the ONE definition: the builder (SY yield), the loader and the S-DAC filter all call it, and the rule is re-derived from the stored labels at
    load time, so a stored flag can never disagree with the labels.
  * the bank file is a deterministic container (header JSON + raw arrays, no timestamps), keyed (log, token, step, rank) + ckpt_step, with a sibling manifest
    {partition_id, label_version, delta grid, fan fingerprint, sha256, counts}.  `load_bank` REFUSES a bank whose partition id, label_version or fan fingerprint differs
    from the run's, or whose bytes changed after minting.

THE TERM (section 2.4, the MASKED TWO-CALL FORM).  The scoring decoder self-attends across the candidate set (model.py CrossBlock.self_attn), so a plain append would
move every fan row's score.  So: the fan call is the trainer's existing call (UNCHANGED, bit-identical with or without a bank); `hn_term` makes ONE more call over
[fan ; hn] in which fan rows attend only to fan rows (so their hidden states equal the fan call's) and each hn row attends to the 64 fan rows and to itself and to nothing
else.  Each hn row is therefore scored against the deployed fan's own states, as a 65th candidate; only hn rows are kept.
    L_hn = sum over valid hn rows, six PDM components, of BCE(logit, target) / (B * K * 6)         (K = fan size = 64)
so one hard negative weighs exactly what one fan proposal weighs, and no free weight is added.  The lane output (component 7) gets NO loss from hn rows.  Padding to 8 uses
a mask; masked rows contribute exactly 0 (their trajectories and targets are replaced BEFORE the call, so even NaN junk in a pad slot changes nothing).
    ⚠️ Two precisions on the draft's and the prereg's wording, both found while implementing (reported to the Master Mind in the hand-back):
      (1) the draft's "a bank row that duplicates fan row j adds exactly row j's share of bce_fan_B" holds ONLY on a scorer with no set-level self-attention: with it, the
          duplicate is scored against 64 fan states PLUS ITSELF, so its logit differs from row j's.  It is tested on a scorer with the self-attention neutralised
          (test_hn_bank.py L_dup_identity); on the real scorer the exact statement is the per-layer oracle below.
      (2) "each hn row is scored as a 65th candidate in its real set" is exact for ONE decoder layer (equal to the last row of an independent 65-candidate call).  At depth > 1 a
          65-candidate call would let the fan rows see the hn row at layer 1, which is the contamination this form exists to avoid and which never happens at selection;
          here the hn row reads the fan's UNPERTURBED per-layer states (the deployed fan call's), which is the situation selection faces.  Tested against an independent
          per-layer oracle (L_equiv_oracle) and, at depth 1, against the real 65-candidate call (L_equiv65_depth1).
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import hashlib
import json
import math
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

TOOL = "hn_bank/1"
SCHEMA = "hn_bank/1"
MAGIC = b"HNBANK1\n"
DELTAS = (-0.60, -0.45, -0.30, -0.15, 0.15, 0.30, 0.45, 0.60)
N_OFF = len(DELTAS)                  # 8 = max_per_sample
N_COMP = 6                           # the six PDM components (NC, DAC, EP, TTC, C, DDC); the lane output (7th) never gets a loss from an hn row
N_LAB = 7                            # + PDMS
LABEL_VERSION = 5                    # the fan labels the screen trains on (PREREG_P6HN_SCREEN 2.3)
STATIONARY_M = 2.0
SY_MIN_SHARE = 0.10                  # SY: share of RL-train scenes with >= 1 hard negative; < 0.10 => SCREEN-SPARSE
N_BOOT, SEED = 10000, 20260927
STATUS_LABELLED, STATUS_STATIONARY, STATUS_LABEL_FAILED = 0, 1, 2
SUBTYPE_NAMES = {1: "DAC-only", 2: "NC-only", 3: "both"}
HN_MASK_DESC = "hn->fan+self"
HN_WEIGHT_DESC = "1/(B*64*6) per element"


class BankMismatch(RuntimeError):
    """A bank that is not the run's: another partition / label_version / fan fingerprint, changed bytes, or a key the bank does not hold."""


# ======================================================================================================== geometry
def _wrap(a):
    return (np.asarray(a, np.float64) + np.pi) % (2 * np.pi) - np.pi


def path_length(anchor) -> float:
    """sum of consecutive xy steps STARTING FROM THE ORIGIN (PREREG_HN_CENSUS 3.6)"""
    xy = np.asarray(anchor, np.float64)[:, :2]
    pts = np.concatenate([np.zeros((1, 2)), xy], 0)
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=-1).sum())


def _central_tangent(xy):
    """unit tangent of an xy polyline by central differences (one-sided at the ends); a zero-length tangent carries the nearest valid one (never NaN)"""
    xy = np.asarray(xy, np.float64)
    d = np.zeros_like(xy)
    d[1:-1] = xy[2:] - xy[:-2]
    d[0] = xy[1] - xy[0]
    d[-1] = xy[-1] - xy[-2]
    n = np.linalg.norm(d, axis=-1)
    ok = n > 1e-9
    if not ok.any():
        raise ValueError("a polyline without a single non-degenerate step has no tangent")
    idx = np.where(ok, np.arange(len(n)), -1)
    fwd = np.maximum.accumulate(idx)                                    # last valid index at or before k
    first = int(np.argmax(ok))
    fwd = np.where(fwd < 0, first, fwd)
    u = d[fwd] / n[fwd][:, None]
    return u


def anchor_array(traj, yaw=None):
    """-> [T, 3] float64 (x, y, yaw).  traj [T, 2] with `yaw` [T], or traj [T, 3].  With no usable yaw, the central-difference tangent heading is filled in."""
    t = np.asarray(traj, np.float64)
    if t.ndim != 2 or t.shape[1] not in (2, 3):
        raise ValueError(f"an anchor is [T, 2 or 3], got {t.shape}")
    if t.shape[1] == 3:
        a = t.copy()
    elif yaw is not None:
        a = np.concatenate([t, np.asarray(yaw, np.float64)[:, None]], 1)
    else:
        a = np.concatenate([t, np.zeros((len(t), 1))], 1)
        a[:, 2] = np.nan
    if not np.isfinite(a[:, 2]).all():
        u = _central_tangent(a[:, :2])
        a[:, 2] = np.arctan2(u[:, 1], u[:, 0])
    return a


def left_normals(anchor):
    """[T, 2] unit LEFT-normals.  From the stored yaw (a [T,3] anchor with finite yaw), else from a central-difference tangent."""
    a = np.asarray(anchor, np.float64)
    if a.shape[1] == 3 and np.isfinite(a[:, 2]).all():
        yaw = a[:, 2]
        return np.stack([-np.sin(yaw), np.cos(yaw)], -1)
    u = _central_tangent(a[:, :2])
    return np.stack([-u[:, 1], u[:, 0]], -1)


def make_offsets(anchor, deltas=DELTAS):
    """[T, 3] anchor -> [len(deltas), T, 3]: pose k shifted by delta along the left-normal at pose k; heading (and the pose index = time) kept."""
    a = anchor_array(anchor)
    n = left_normals(a)
    out = np.empty((len(deltas),) + a.shape, np.float64)
    for i, d in enumerate(deltas):
        out[i, :, :2] = a[:, :2] + float(d) * n
        out[i, :, 2] = a[:, 2]
    return out


def _store(x):
    """the values the scorer is FED and the labeller LABELS are the same numbers: xy to 5 d.p., yaw to 6 d.p. (the fan dump's rounding), then float32"""
    x = np.asarray(x, np.float64).copy()
    x[..., :2] = np.round(x[..., :2], 5)
    x[..., 2] = np.round(x[..., 2], 6)
    return x.astype(np.float32)


# ======================================================================================================== labels (through label_log itself)
def labeller():
    """`onpolicy_relabel_pdm.label_log`, BY REFERENCE (checklist item 1: 'never a copy').  Imported lazily: the module is numpy-only at import time."""
    import onpolicy_relabel_pdm as ORP
    return ORP.label_log


def build_label_row(meta, traj, label_version):
    """ONE label-job row, the format label_log reads.  Shared by the bank builder and by `label_identity_check` (I1), so the two cannot be formatted differently."""
    t = np.asarray(traj, np.float64)
    return {"log_name": meta["log_name"], "token": meta["token"], "step": int(meta["step"]), "rank": int(meta.get("rank", 0)),
            "ckpt_step": int(meta["ckpt_step"]), "label_version": int(label_version),
            "traj": t[..., :2].tolist(), "yaw": t[..., 2].tolist()}


def label_sets(rows, label_fn=None, mapper=None):
    """rows -> ({key: {col: [M] floats}}, counters).  Groups rows by log (label_log builds one scenario list per log), calls label_fn(job) for each, parses the lines.
    A returned line whose label_version differs from the row's, or whose key was not asked for, is an error (never read from the wrong labels)."""
    label_fn = label_fn or labeller()
    mapper = mapper or map
    by_log = collections.OrderedDict()
    for r in rows:
        by_log.setdefault(r["log_name"], []).append(r)
    jobs = [(lg, rs) for lg, rs in by_log.items()]
    want = {(r["log_name"], r["token"], int(r["step"]), int(r["rank"])): r for r in rows}
    got, failed, skipped, errs = {}, 0, 0, []
    for out, nf, ns, err in mapper(label_fn, jobs):
        failed += int(nf)
        skipped += int(ns)
        if err:
            errs.append(err)
        for ln in out:
            rec = json.loads(ln)
            k = (rec["key"][0], rec["key"][1], int(rec["key"][2]), int(rec["key"][3]))
            if k not in want:
                raise BankMismatch(f"the labeller returned a set that was not asked for: {k}")
            if int(rec["label_version"]) != int(want[k]["label_version"]):
                raise BankMismatch(f"label_version: asked {want[k]['label_version']}, the labeller returned {rec['label_version']} for {k}")
            if int(rec["ckpt_step"]) != int(want[k]["ckpt_step"]):
                raise BankMismatch(f"ckpt_step: asked {want[k]['ckpt_step']}, the labeller returned {rec['ckpt_step']} for {k}")
            got[k] = {c: rec[c] for c in ("nc", "dac", "ep", "ttc", "c", "ddc", "pdms")}
    return got, {"label_failed": failed, "label_skipped": skipped, "label_errors": errs, "requested": len(rows), "returned": len(got)}


def lab_matrix(cols):
    """{col: [M]} -> [M, 7] float32 in the order (nc, dac, ep, ttc, c, ddc, pdms)"""
    return np.stack([np.asarray(cols[c], np.float64) for c in ("nc", "dac", "ep", "ttc", "c", "ddc", "pdms")], -1).astype(np.float32)


def label_identity_check(fan_rows, banked, label_fn=None, label_version=LABEL_VERSION):
    """I1: the builder's labelling call, fed a set's own 64 fan proposals, reproduces RL-S1's banked PDM labels BIT FOR BIT.
    fan_rows: label-job rows (log_name, token, step, rank, ckpt_step, traj [64,T,2], yaw [64,T]); banked: {key: {nc, dac, ep, ttc, c, ddc, pdms: [64]}}."""
    rows = []
    for r in fan_rows:                                                  # the SAME row formatter the bank builder uses, so a defect in it is visible here
        arr = np.concatenate([np.asarray(r["traj"], np.float64), np.asarray(r["yaw"], np.float64)[..., None]], -1)
        rows.append(build_label_row(r, arr, label_version))
    got, cnt = label_sets(rows, label_fn)
    bad = []
    for k, cols in got.items():
        ref = banked.get(k)
        if ref is None or any(list(cols[c]) != list(ref[c]) for c in ("nc", "dac", "ep", "ttc", "c", "ddc", "pdms")):
            bad.append(list(k))
    n = len(rows)
    return {"n_requested": n, "n_labelled": len(got), "n_identical": len(got) - len(bad), "mismatching_keys": bad[:5],
            "ok": bool(n > 0 and len(got) == n and not bad)}


# ======================================================================================================== the hard-negative rule (ONE definition)
def hard_negative_rule(lab9):
    """lab9 [..., 9, 7] (slot 0 = the anchor; columns nc, dac, ep, ttc, c, ddc, pdms) -> (hn [..., 8] bool, sub [..., 8] int8: 0 none, 1 DAC-only, 2 NC-only, 3 both).
    hard negative <=> (NC == 0 or DAC == 0) and PDMS(offset) == 0 and PDMS(anchor) > 0.  NaN labels (unlabelled rows) are never a hard negative."""
    lab9 = np.asarray(lab9)
    nc, dac, pdms = lab9[..., 1:, 0], lab9[..., 1:, 1], lab9[..., 1:, 6]
    anchor_ok = lab9[..., 0, 6] > 0
    nc0 = nc == 0
    dac0 = dac == 0
    hn = (nc0 | dac0) & (pdms == 0) & anchor_ok[..., None]
    sub = np.where(hn, dac0.astype(np.int8) + 2 * nc0.astype(np.int8), 0).astype(np.int8)
    return hn, sub


# ======================================================================================================== the bank
def _sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def keys_fingerprint(keys):
    """sha256 over the sorted (log, token, step, rank, ckpt_step) list: the scene-list identity"""
    ks = sorted((str(k[0]), str(k[1]), int(k[2]), int(k[3]), int(k[4]) if len(k) > 4 else 0) for k in keys)
    return _sha256_bytes(json.dumps(ks, separators=(",", ":")).encode("utf-8"))


def write_container(path, keys, arrays):
    """deterministic: header JSON (sorted keys, no timestamps) + raw C-order array bytes.  Returns the sha256 of the file."""
    order = ["status", "traj9", "lab9"]
    meta, off = [], 0
    for name in order:
        a = np.ascontiguousarray(arrays[name])
        meta.append({"name": name, "dtype": a.dtype.str, "shape": list(a.shape), "offset": off, "nbytes": int(a.nbytes)})
        off += int(a.nbytes)
    header = json.dumps({"schema": SCHEMA, "keys": [list(k) for k in keys], "arrays": meta}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<Q", len(header)))
        f.write(header)
        for name in order:
            f.write(np.ascontiguousarray(arrays[name]).tobytes())
    return sha256_file(path)


def read_container(path):
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:len(MAGIC)] != MAGIC:
        raise BankMismatch(f"{path}: not a {SCHEMA} bank (bad magic)")
    (hl,) = struct.unpack("<Q", raw[len(MAGIC):len(MAGIC) + 8])
    h0 = len(MAGIC) + 8
    header = json.loads(raw[h0:h0 + hl].decode("utf-8"))
    base = h0 + hl
    arrays = {}
    for m in header["arrays"]:
        arrays[m["name"]] = np.frombuffer(raw, dtype=np.dtype(m["dtype"]), count=int(np.prod(m["shape"])) if m["shape"] else 1, offset=base + m["offset"]).reshape(m["shape"]).copy()
    return [tuple(k) for k in header["keys"]], arrays


class HNBank:
    """The loaded bank.  `batch(keys)` serves hn_xy [B, 8, T, 3], hn_tg [B, 8, 6], hn_m [B, 8] for the trainer; the rule is re-derived from the stored labels."""

    def __init__(self, keys, arrays, manifest):
        self.keys = list(keys)
        self.status = arrays["status"]
        self.traj9 = arrays["traj9"]
        self.lab9 = arrays["lab9"]
        self.manifest = manifest
        self.index = {tuple(k[:4]): i for i, k in enumerate(self.keys)}
        if len(self.index) != len(self.keys):
            raise BankMismatch("duplicate (log, token, step, rank) keys in the bank")
        self.hn, self.sub = hard_negative_rule(self.lab9)
        self.hn &= (self.status == STATUS_LABELLED)[:, None]
        self.sub = np.where(self.hn, self.sub, 0).astype(np.int8)

    def __len__(self):
        return len(self.keys)

    def hn_mask(self, subtypes=None):
        """[N, 8] bool; `subtypes` (e.g. (1,) for the S-DAC arm: DAC-only hard negatives) restricts the rows that count"""
        if subtypes is None:
            return self.hn.copy()
        return self.hn & np.isin(self.sub, list(subtypes))

    def n_rows(self, subtypes=None):
        return int(self.hn_mask(subtypes).sum())

    def scenes_with_hn(self, subtypes=None):
        return int(self.hn_mask(subtypes).any(1).sum())

    def hn_block(self, arm="B", subtypes=None):
        """the trainer's `hn` data-event block (G-DVB compares it with the argv)"""
        return {"sha256": self.manifest["sha256"], "rows": self.n_rows(subtypes), "scenes_with_hn": self.scenes_with_hn(subtypes), "max_per_sample": N_OFF, "arm": arm,
                "comps": N_COMP, "weight": HN_WEIGHT_DESC, "mask": HN_MASK_DESC}

    def batch(self, keys, ckpt_steps=None, subtypes=None, strict=True):
        """keys: [(log, token, step, rank)] -> dict of torch tensors.  A key the bank does not hold is an ERROR when strict (a scene without a bank row is a partition bug,
        not an empty row); `ckpt_steps` (same length) must equal the bank's, or the join is refused (a bank for another fan)."""
        idx = []
        for j, k in enumerate(keys):
            k = (str(k[0]), str(k[1]), int(k[2]), int(k[3]))
            i = self.index.get(k)
            if i is None:
                if strict:
                    raise BankMismatch(f"key {k} is not in the bank")
                idx.append(-1)
                continue
            if ckpt_steps is not None and int(ckpt_steps[j]) != int(self.keys[i][4]):
                raise BankMismatch(f"key {k}: bank ckpt_step {self.keys[i][4]} != served {ckpt_steps[j]}")
            idx.append(i)
        idx = np.asarray(idx)
        ok = idx >= 0
        ii = np.where(ok, idx, 0)
        m = self.hn_mask(subtypes)[ii] & ok[:, None]
        xy = self.traj9[ii, 1:]
        tg = np.nan_to_num(self.lab9[ii, 1:, :N_COMP], nan=0.0)
        return {"hn_xy": torch.from_numpy(np.ascontiguousarray(xy)), "hn_tg": torch.from_numpy(np.ascontiguousarray(tg, dtype=np.float32)),
                "hn_m": torch.from_numpy(m), "hn_sub": torch.from_numpy(np.where(m, self.sub[ii], 0))}


def load_bank(path, manifest_path=None, expect=None):
    """Load and VERIFY.  The manifest's sha256 must equal the file's (a bank whose bytes changed after minting is refused); `expect` = {partition_id, label_version,
    fan_fingerprint}: every key given must equal the manifest's, or the bank is refused."""
    manifest_path = manifest_path or (str(path) + ".manifest.json")
    with open(manifest_path, encoding="utf-8") as f:
        man = json.load(f)
    got = sha256_file(path)
    if got != man.get("sha256"):
        raise BankMismatch(f"{path}: sha256 {got[:16]} != manifest {str(man.get('sha256'))[:16]} (changed after minting)")
    for k, v in (expect or {}).items():
        if man.get(k) != v:
            raise BankMismatch(f"bank {k} = {man.get(k)!r}, the run needs {v!r}")
    keys, arrays = read_container(path)
    if len(keys) != man.get("n_rows"):
        raise BankMismatch(f"{path}: {len(keys)} keys vs manifest n_rows {man.get('n_rows')}")
    return HNBank(keys, arrays, man)


def build_bank(anchors, *, partition_id, fan_fingerprint, label_fn=None, mapper=None, label_version=LABEL_VERSION, deltas=DELTAS):
    """anchors: iterable of dicts {log_name, token, step, rank, ckpt_step, traj [T,2|3], yaw [T] (optional)} -> (keys, arrays, manifest-without-sha).
    Every anchor yields exactly one row: LABELLED, STATIONARY (path < 2.0 m: nothing is sent to the labeller) or LABEL_FAILED (the labeller dropped it)."""
    anchors = list(anchors)
    keys, status, traj9, lab9, rows_to_label, pos = [], [], [], [], [], {}
    for r in anchors:
        a = _store(anchor_array(r["traj"], r.get("yaw")))
        key = (r["log_name"], r["token"], int(r["step"]), int(r.get("rank", 0)), int(r["ckpt_step"]))
        keys.append(key)
        t9 = np.empty((1 + len(deltas),) + a.shape, np.float32)
        t9[0] = a
        if path_length(a) < STATIONARY_M:
            t9[1:] = a
            status.append(STATUS_STATIONARY)
            lab9.append(np.full((1 + len(deltas), N_LAB), np.nan, np.float32))
        else:
            t9[1:] = _store(make_offsets(a.astype(np.float64), deltas))
            status.append(STATUS_LABEL_FAILED)               # until the labeller returns it
            lab9.append(np.full((1 + len(deltas), N_LAB), np.nan, np.float32))
            pos[key[:4]] = len(keys) - 1
            rows_to_label.append(build_label_row(r, t9.astype(np.float64), label_version))
        traj9.append(t9)
    got, cnt = label_sets(rows_to_label, label_fn, mapper) if rows_to_label else ({}, {"label_failed": 0, "label_skipped": 0, "label_errors": [], "requested": 0, "returned": 0})
    for k, cols in got.items():
        m = lab_matrix(cols)
        if m.shape != (1 + len(deltas), N_LAB):
            raise BankMismatch(f"{k}: the labeller returned {m.shape}, need {(1 + len(deltas), N_LAB)}")
        i = pos[k]
        lab9[i] = m
        status[i] = STATUS_LABELLED
    arrays = {"status": np.asarray(status, np.int8), "traj9": np.stack(traj9), "lab9": np.stack(lab9)}
    man = {"tool": TOOL, "schema": SCHEMA, "partition_id": partition_id, "label_version": int(label_version), "delta_grid": [float(d) for d in deltas],
           "fan_fingerprint": fan_fingerprint, "n_rows": len(keys), "keys_sha256": keys_fingerprint(keys), "max_per_sample": len(deltas), "comps": N_COMP,
           "stationary_m": STATIONARY_M, "label_counters": {k: v for k, v in cnt.items() if k != "label_errors"}, "label_errors": cnt["label_errors"][:5],
           "labeller": "onpolicy_relabel_pdm.label_log"}
    return keys, arrays, man


def mint_bank(anchors, path, **kw):
    """build + write the bank file and its sibling manifest; returns the loaded bank"""
    keys, arrays, man = build_bank(anchors, **kw)
    man["sha256"] = write_container(path, keys, arrays)
    b = HNBank(keys, arrays, man)
    man["counts"] = {"labelled": int((arrays["status"] == STATUS_LABELLED).sum()), "stationary": int((arrays["status"] == STATUS_STATIONARY).sum()),
                     "label_failed": int((arrays["status"] == STATUS_LABEL_FAILED).sum()), "hard_negative_rows": b.n_rows(), "scenes_with_hn": b.scenes_with_hn()}
    with open(str(path) + ".manifest.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(man, f, indent=1, sort_keys=True)
    return load_bank(path, expect={"partition_id": kw["partition_id"], "label_version": kw.get("label_version", LABEL_VERSION), "fan_fingerprint": kw["fan_fingerprint"]})


# ======================================================================================================== SY: the yield report
def _log_boot_ratio(num, den, log_idx, n=N_BOOT, seed=SEED):
    """ratio-of-sums log-cluster bootstrap (the draws proxy_eval.boot makes): (point, lo, hi)"""
    num, den = np.asarray(num, np.float64), np.asarray(den, np.float64)
    G = int(log_idx.max()) + 1
    sn, sd = np.bincount(log_idx, num, G), np.bincount(log_idx, den, G)
    rng = np.random.default_rng(seed)
    est = np.empty(n)
    for b in range(n):
        pick = rng.integers(0, G, G)
        cnt = np.bincount(pick, minlength=G)
        d = float(cnt @ sd)
        est[b] = float(cnt @ sn) / d if d > 0 else float("nan")
    pt = float(sn.sum() / sd.sum()) if sd.sum() > 0 else float("nan")
    return pt, float(np.nanpercentile(est, 2.5)), float(np.nanpercentile(est, 97.5))


def yield_report(bank, n_boot=N_BOOT, seed=SEED, min_share=SY_MIN_SHARE):
    """SY: the share of the bank's scenes (RL-train) with >= 1 hard negative, by a log-cluster bootstrap over the bank's logs.  It reads SUPPLY, never an effect.
    `sparse` (=> SCREEN-SPARSE, no cache, no pass) is judged on the POINT estimate (< 0.10); the interval is reported beside it.  The denominator is EVERY scene in the
    bank (stationary and label-failed ones count as 'no hard negative'); the labelled-only share is reported too."""
    N = len(bank)
    logs = sorted({k[0] for k in bank.keys})
    lid = {lg: i for i, lg in enumerate(logs)}
    li = np.asarray([lid[k[0]] for k in bank.keys])
    has = bank.hn.any(1).astype(np.float64)
    pt, lo, hi = _log_boot_ratio(has, np.ones(N), li, n_boot, seed)
    lab = bank.status == STATUS_LABELLED
    nl = int(lab.sum())
    per_slot = []
    for i, d in enumerate(DELTAS):
        per_slot.append({"delta_m": d, "side": "left" if d > 0 else "right", "hn_rate_per_labelled_scene": float(bank.hn[lab, i].mean()) if nl else None})
    sub = {SUBTYPE_NAMES[c]: int((bank.sub == c).sum()) for c in (1, 2, 3)}
    nhn = int(bank.hn.sum())
    return {"tool": TOOL, "gate": "SY", "n_scenes": N, "n_logs": len(logs), "n_labelled": nl, "n_stationary": int((bank.status == STATUS_STATIONARY).sum()),
            "n_label_failed": int((bank.status == STATUS_LABEL_FAILED).sum()), "scenes_with_hn": int(has.sum()),
            "share_with_hn": pt, "share_ci95": [lo, hi], "share_with_hn_among_labelled": float(bank.hn[lab].any(1).mean()) if nl else None,
            "estimator": f"log-cluster bootstrap, ratio of sums, {n_boot} resamples, seed {seed}, percentile 2.5/97.5; denominator = every scene in the bank",
            "min_share": min_share, "sparse": bool(pt < min_share), "hard_negative_rows": nhn, "rate_per_delta_and_side": per_slot, "subtype_rows": sub,
            "anchors_with_pdms_zero": int(((bank.lab9[lab, 0, 6]) == 0).sum()), "anchors_with_pdms_zero_share_of_labelled": float((bank.lab9[lab, 0, 6] == 0).mean()) if nl else None,
            "evidence": "MEASURED on the bank's own labels (supply only; no model, no effect)"}


# ======================================================================================================== the term
class HNOut(collections.namedtuple("HNOut", "loss n_hn logits")):
    """loss: scalar float32; n_hn: number of VALID hard-negative rows that entered it (D1's count); logits: [B, 8, C] float32 (diagnostics, masked rows are junk)"""


def hn_attn_mask(K, H, device=None):
    """bool [K+H, K+H], True = NOT allowed to attend (nn.MultiheadAttention's convention).  Row i attends to column j unless j is an hn column other than i itself:
    fan rows see the fan only (their states equal the fan call's); an hn row sees the K fan rows and itself and nothing else."""
    i = torch.arange(K + H, device=device)[:, None]
    j = torch.arange(K + H, device=device)[None, :]
    blocked = (j >= K) & (j != i)
    return blocked


class _MaskedSelfAttn(torch.nn.Module):
    """a temporary stand-in for CrossBlock.self_attn that passes `attn_mask` to the REAL nn.MultiheadAttention, so the REAL CrossBlock.forward runs (no copy of its body)"""

    def __init__(self, inner, blocked):
        super().__init__()
        self.inner = inner
        self.blocked = blocked

    def forward(self, q, k, v, **kw):
        return self.inner(q, k, v, attn_mask=self.blocked, **kw)


@contextlib.contextmanager
def _masked_blocks(blocks, blocked):
    orig = [b.self_attn for b in blocks]
    try:
        for b, inner in zip(blocks, orig):
            b.self_attn = _MaskedSelfAttn(inner, blocked)
        yield
    finally:
        for b, inner in zip(blocks, orig):
            b.self_attn = inner


def score_set(scorer, trajs, sctx, blocked=None):
    """The scoring decoder on a candidate set, optionally with a self-attention mask.  With blocked=None this is `REFe.score_trajectories` line for line (pinned in
    test_hn_bank.py: equal bit for bit, and the source of score_trajectories is hash-pinned there so the two cannot drift apart)."""
    s = scorer.score_q_mlp(trajs.flatten(2))
    if blocked is None:
        for blk in scorer.score_dec:
            s = blk(s, sctx)
    else:
        with _masked_blocks(list(scorer.score_dec), blocked):
            for blk in scorer.score_dec:
                s = blk(s, sctx)
    return scorer.score_head(s)


def hn_scores(scorer, fan_xy, hn_xy, hn_m, sctx):
    """the SECOND, masked call over [fan ; hn] -> hn logits [B, 8, C].  Pad slots are replaced by zeros first, so nothing in a pad slot can reach a valid row or a gradient."""
    B, K = fan_xy.shape[0], fan_xy.shape[1]
    H = hn_xy.shape[1]
    keep = hn_m.bool()
    xy = torch.where(keep[..., None, None], hn_xy.to(fan_xy.dtype), torch.zeros_like(hn_xy, dtype=fan_xy.dtype))
    both = torch.cat([fan_xy, xy], dim=1)
    blocked = hn_attn_mask(K, H, device=fan_xy.device)
    return score_set(scorer, both, sctx, blocked)[:, K:]


def hn_term(scorer, fan_xy, hn_xy, hn_tg, hn_m, sctx):
    """L_hn = sum over valid hn rows, six components, BCE(logit, target) / (B * K * 6).  `scorer` has score_q_mlp / score_dec / score_head (arm B's ScorerCopy, or the
    model itself).  Returns HNOut.  Arm A never calls this; with an all-False mask the loss is exactly 0.0 and every gradient exactly 0.0."""
    B, K = fan_xy.shape[0], fan_xy.shape[1]
    keep = hn_m.bool()
    logits = hn_scores(scorer, fan_xy, hn_xy, keep, sctx).float()
    tg = torch.where(keep[..., None], hn_tg.to(torch.float32), torch.zeros_like(hn_tg, dtype=torch.float32))
    bce = F.binary_cross_entropy_with_logits(logits[..., :N_COMP], tg[..., :N_COMP], reduction="none")
    bce = torch.where(keep[..., None], bce, torch.zeros_like(bce))
    loss = bce.sum() / float(B * K * N_COMP)
    return HNOut(loss, keep.sum(), logits)


def probe_fields(scorer_b, scorer_a, fan_xy, hn_xy, hn_tg, hn_m, sctx):
    """RAW fields for the launch gate's G14 (re-judged from these by judge_g14, never from an 'ok'): the section-4.3 gradient table from the term ALONE, plus the fan-call
    identity.  Clears and uses .grad of both scorers."""
    for m in (scorer_a, scorer_b):
        m.zero_grad(set_to_none=True)
    out = hn_term(scorer_b, fan_xy, hn_xy, hn_tg, hn_m, sctx)
    out.loss.backward()

    def gabs(mod, sl=None):
        tot, any_grad = 0.0, False
        for p in mod.parameters():
            if p.grad is None:
                continue
            any_grad = True
            tot += float(p.grad.abs().sum() if sl is None else p.grad[sl].abs().sum())
        return tot, any_grad

    rows = {}
    for name in ("score_q_mlp", "score_dec"):
        rows[name] = gabs(getattr(scorer_b, name))[0]
    head = scorer_b.score_head
    rows["score_head_rows_0_5"] = float(head.weight.grad[:N_COMP].abs().sum() + head.bias.grad[:N_COMP].abs().sum())
    rows["score_head_row_6_lane"] = float(head.weight.grad[N_COMP:].abs().sum() + head.bias.grad[N_COMP:].abs().sum()) if head.weight.shape[0] > N_COMP else 0.0
    rows["armA_all"] = sum(float(p.grad.abs().sum()) for p in scorer_a.parameters() if p.grad is not None)
    finite = all(bool(torch.isfinite(p.grad).all()) for p in scorer_b.parameters() if p.grad is not None)
    with torch.no_grad():
        one = score_set(scorer_b, fan_xy, sctx)
        two = score_set(scorer_b, fan_xy, sctx)            # the fan call is the same call; the term never touches it
    return {"grad_abs_sum": rows, "grads_finite": finite, "loss": float(out.loss), "n_hn": int(out.n_hn), "fan_two_call_equals_one_call": bool(torch.equal(one, two))}


# ======================================================================================================== CLI (yield only; the build needs the real anchors + labeller)
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    y = sub.add_parser("yield", help="SY report of a minted bank")
    y.add_argument("--bank", required=True)
    y.add_argument("--partition-id", default=None)
    y.add_argument("--label-version", type=int, default=LABEL_VERSION)
    y.add_argument("--fan-fingerprint", default=None)
    y.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    exp = {"label_version": a.label_version}
    if a.partition_id:
        exp["partition_id"] = a.partition_id
    if a.fan_fingerprint:
        exp["fan_fingerprint"] = a.fan_fingerprint
    bank = load_bank(a.bank, expect=exp)
    rep = yield_report(bank)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rep, f, indent=1, sort_keys=True)
    print(f"ZZHNBANK_SY share {rep['share_with_hn']:.4f} ci {rep['share_ci95'][0]:.4f} {rep['share_ci95'][1]:.4f} sparse {rep['sparse']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
