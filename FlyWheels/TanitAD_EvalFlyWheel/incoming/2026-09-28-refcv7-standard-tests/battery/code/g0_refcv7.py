"""GATE G0 (SPEC.md §2) -- the refcv7 loader reproduces refcv7's OWN in-run eval row.

    python g0_refcv7.py --ckpt D:/refcv7_eval_kit/ckpt/ckpt_1500.pt \
        --config D:/refcv7_eval_kit/ckpt/config.json --metrics <metrics.jsonl copy> \
        --seeds 0,1,2,3,4,5,6,7 --out <g0.json>

The trainer's own `compute_losses_v3`, `_eval_row_from_acc` and `detection_metrics.summarise`
(`refc_v3_train.py:9658-9714`) on the loader's eval dataset over the SAME 128 windows in the SAME 8
batches of 16 (`inrun_eval_perm`). Recorded departures: (1) the forward is micro-batched INSIDE the
single model call (`microbatch.py`); (2) only the LAW frame of `future_frames` reaches the device;
(3) `--trunk-compile` is off (loader); (4) the DDIM draw is seeded per inference seed; (5) the
parameters carry their TRAINING requires_grad flags (the in-run eval's state), restored from the
built model's `declare_grad_unreachable` declarations -- the loader leaves them all False.

Ported from the refcv6 package's `reproduce_inrun_eval.py` (2026-09-23 battery), refcv7 changes only.
Every artifact is asserted by the CALLER on the JSON this writes, never on this process's exit code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tanitad.eval import refcv7_loader as L  # noqa: E402

L.bootstrap()
import numpy as np  # noqa: E402
import torch  # noqa: E402
from microbatch import MicroBatchForward  # noqa: E402

# --------------------------------------------------------------------------------------- #
# term classes -- SPEC §2, by NAME, written before any value was read                      #
# --------------------------------------------------------------------------------------- #
EXCLUDED_EXACT = {"eval_goal_gate_grad", "eval_step"}
COUNT_EXACT = {"eval_batches", "eval_windows", "eval_slot_valid_frac", "eval_tac_label_rows",
               "eval_tac_label_v7", "eval_nav_injected", "eval_ego_injected",
               "eval_box3d_visible_filter"}
MODEL_DEP_COUNT_SUFFIX = ("_n_conf", "_tp@gate", "_n_ignore_masked_slots", "_n_presence_exempt")
DET_TOKENS = ("_prec@gate", "_rec@gate", "_f1@gate", "_ap2m", "_auroc_", "_cls_acc_tp",
              "_det_ap", "_det_map", "_conf_ratio", "_centre_err_p50")
MATCHED_BASE = {"agent_centre", "agent_cls", "agent_presence", "agent_size", "agent_yaw",
                "box3d", "box3d_centre", "box3d_cls", "box3d_h", "box3d_occ", "box3d_presence",
                "box3d_presence_focal", "box3d_rates", "box3d_size", "box3d_yaw", "box3d_z",
                "box3d_vis1"}
#: t(0.995, K-1) keyed by K (the number of seeds). K = 24 added by SPEC AMENDMENT A5: t(0.995, 23) = 2.807.
T_995 = {2: 63.657, 3: 9.925, 4: 5.841, 5: 4.604, 6: 4.032, 7: 3.707, 8: 3.499, 9: 3.355,
         10: 3.250, 12: 3.106, 16: 2.947, 24: 2.807}
#: the seeds G0 was REGISTERED with (SPEC §2): "as registered" and A2 are always judged on these only
REGISTERED_SEEDS = tuple(range(8))
#: SPEC AMENDMENT A5: the STOCHASTIC class is judged on K >= 24 seeds at milestones
A5_MIN_SEEDS = 24

# --------------------------------------------------------------------------------------- #
# SPEC AMENDMENT A6 -- DRAFT until `raw/SPEC_SHA256_AMENDMENT_A6.txt` exists (the text is      #
# `raw/AMENDMENT_A6_DRAFT.md`). Until then `verdict_A6` is COMPUTED AND REPORTED, never the gate.  #
# --------------------------------------------------------------------------------------- #
#: A6 item 1 -- terms whose loss TARGET is a hard threshold of the model's OWN output, by NAME from
#: source: `tacv6_goal_conf_bce` = BCE(goal_conf, 1[sigmoid(goal_logits) >= 0.5] == y), weighted by
#: goal_w * class_mask and normalised per batch by sum(w) (refcv6_tactical.py:823-833). Deterministic in
#: the inference seed, but DISCONTINUOUS in the numerics: one validity logit crossing 0 moves the batch
#: value by +-c_i * w_i / sum(w) in one step (c_i = that cell's confidence logit; softplus(c)-softplus(-c)
#: = c exactly). On 128 windows the eval row averages only ~271 supervised cells (7..55 per batch).
THRESHOLD_TARGET = {"eval_tacv6_goal_conf_bce"}
#: A6's floor multiplier: SPEC §2's own convention for a measured floor (wrapper clause: Wrapper <= 3*Floor)
A6_K = 3.0
#: the arm A6 measures the numerics floor with: seed 0 under the wrapper clause's P3 settings
A6_ARM = "fp32_s0"
A6_REGISTRATION = HERE.parent / "raw" / "SPEC_SHA256_AMENDMENT_A6.txt"


def a6_registration(started: str | None = None) -> dict:
    """Is A6 REGISTERED for a G0 that started at `started` (ISO, local)? Registered iff the
    registration file exists AND was written before the G0 started (its mtime, recorded)."""
    p = A6_REGISTRATION
    if not p.exists():
        return {"registered": False, "why": f"{p.name} absent: A6 is a DRAFT (reported, never the gate)"}
    mt = time.strftime("%FT%T", time.localtime(p.stat().st_mtime))
    if started is not None and mt > started[:19]:
        return {"registered": False, "file_mtime": mt, "g0_started": started,
                "why": "registered AFTER this G0 started: A6 is POST HOC for this checkpoint"}
    return {"registered": True, "file_mtime": mt, "g0_started": started,
            "text": p.read_text(encoding="utf-8", errors="replace")[:500]}


# --------------------------------------------------------------------------------------- #
# SPEC AMENDMENT A7 -- REGISTERED by the Master Mind 2026-10-04T17:11:57+02:00 (SPEC.md "AMENDMENT A7";   #
# sha256 in `raw/SPEC_SHA256_AMENDMENT_A7.txt`). It binds only a G0 that STARTS after that file was      #
# written. A7.1 (A2's low-support rule under every amendment) is `A2_LOWSUPPORT_AMENDS`, below.          #
#   A7.2  the DISCRETE-SMALL-N population guard (`a7_lowsupport_guard`, called by `judge`) and the      #
#         M5 power probe (`ClassColumnSwap` here, judged by `a7_m5_probe`);                              #
#   A7.3  the detection packs of inference seed 0 and fp32_s0, banked (`bank_detection_packs`);          #
#   A7.4  the inference-seed draw reported as ONE draw (`seed_group_report`, `seed_draw_correlation`).   #
# A7.3 and A7.4 are REPORTED, NEVER GATING; M5 is reported and never VOIDs G0.                           #
# --------------------------------------------------------------------------------------- #
A7_REGISTRATION = HERE.parent / "raw" / "SPEC_SHA256_AMENDMENT_A7.txt"
#: A7 inherits A6 items 1-6 and 8 UNCHANGED (SPEC A7 "Unchanged"): the A6 floor machinery (phi, the
#: THRESHOLD_TARGET interval, the fp32_s0 arm) is in force under A7 as well
A6_AMENDS = ("A6", "A7")
#: A7.2 item 2: a member "moved" iff |delta| > 0.02 (A2's DETECTION tolerance); the class FAILS iff
#: N_in > FACTOR * N_num + SLACK. Constants chosen with the 30,000 / 50,400 counts visible (A7 "Calibration
#: disclosure"); they bind only checkpoints whose data did not exist at registration.
A7_MOVE_TOL = 0.02
A7_GUARD_FACTOR = 2
A7_GUARD_SLACK = 5
#: A7.2 item 3: the two classes whose logit columns M5 swaps, and the blind spot an undetected M5 NAMES
M5_CLASSES = ("bus", "heavy_truck")
M5_BLIND_SPOT = "a rare-class index swap is invisible to G0"
#: A7.3: the sibling file the packs are banked into (`<g0 stem>.packs.npz`)
A7_PACKS_SUFFIX = ".packs.npz"
A7_PACK_FIELDS = ("logit", "xy", "cls", "gt_xy", "gt_cls", "gt_pos", "gt_ign")


def a7_registration(started: str | None = None) -> dict:
    """Is A7 REGISTERED for a G0 that started at `started` (ISO, local)? Mirrors `a6_registration`: registered iff
    the registration file exists AND was written before the G0 started (its mtime, recorded). In addition A6 must
    itself be registered for that G0: A7 inherits A6 items 1-6 and 8 unchanged, so it can never gate on rules A6
    has not yet put in force."""
    p = A7_REGISTRATION
    if not p.exists():
        return {"registered": False, "why": f"{p.name} absent: A7 is not registered (reported, never the gate)"}
    mt = time.strftime("%FT%T", time.localtime(p.stat().st_mtime))
    if started is not None and mt > started[:19]:
        return {"registered": False, "file_mtime": mt, "g0_started": started,
                "why": "registered AFTER this G0 started: A7 is POST HOC for this checkpoint"}
    r6 = a6_registration(started)
    if not r6["registered"]:
        return {"registered": False, "file_mtime": mt, "g0_started": started,
                "why": f"A6 is not registered for this G0 ({r6.get('why')}): A7 inherits A6 items 1-6 and 8 "
                       f"and cannot gate before them"}
    return {"registered": True, "file_mtime": mt, "g0_started": started,
            "text": p.read_text(encoding="utf-8", errors="replace")[:500]}


# ----------------------------------------------------------------------------------- #
# A7.2 item 3 -- the M5 power probe: a rare-class INDEX SWAP in both slot heads         #
# ----------------------------------------------------------------------------------- #
def m5_class_indices(classes=None) -> dict:
    """{class name: column} for M5's two classes, DERIVED from the vocabulary the slot heads' class channels and the
    GT labels share: `agent_slots.AGENT_CLASSES` (== `bev_raster.ALL_CLASSES`, imported never re-listed;
    `targets_from_join` indexes the GT class as `enumerate(AGENT_CLASSES)` and `detection_metrics` keys its
    per-class cells by the same tuple). Never a hand-written index: this is the one place a wrong guess would make
    the probe swap two columns that are not bus and heavy_truck. `classes` overrides the vocabulary (tests)."""
    if classes is None:
        from tanitad.models import agent_slots as _as
        classes = _as.AGENT_CLASSES
    classes = tuple(classes)
    if len(set(classes)) != len(classes):
        raise ValueError(f"M5: the class vocabulary has duplicates: {classes}")
    gone = [c for c in M5_CLASSES if c not in classes]
    if gone:
        raise ValueError(f"M5: {gone} not in the class vocabulary {classes}")
    return {c: classes.index(c) for c in M5_CLASSES}


def m5_cls_slice() -> slice:
    """The `cls` channel slice of the slot head's output row (`agent_slots.SLOT_SLICES["cls"]`), checked to be
    exactly as wide as the vocabulary: the class-logit column of class c is `slice.start + index(c)`."""
    from tanitad.models import agent_slots as _as
    sl = _as.SLOT_SLICES["cls"]
    if sl.stop - sl.start != len(_as.AGENT_CLASSES):
        raise ValueError(f"M5: the cls slice {sl} is not as wide as the {len(_as.AGENT_CLASSES)}-class vocabulary")
    return sl


class ClassColumnSwap:
    """SPEC A7.2 item 3 (M5): swap the `bus` and `heavy_truck` class-logit columns in BOTH slot heads, then RESTORE.

    The class logits of `AgentSlotDecoder` / `Box3DSlotDecoder` (the agent head `model.core.agent_head`, the box3d
    head `model._perception.box_dec`; the trainer's own accessors, `refc_v3_train._slot_refine_block`) are
    `raw[..., SLOT_SLICES['cls']]` of ONE `nn.Linear` named `head` that every decoder layer's output goes through, so
    exchanging the two output ROWS of its weight and bias exchanges those two logit columns at every layer and leaves
    every class-agnostic channel (presence, box, yaw, rates, occlusion) and every other class column bit-identical.
    That is the effect of a vocabulary-order loader defect: right weights, wrong class index.

    `apply()` snapshots both heads' weight and bias first; `restore()` swaps back (a row exchange is its own
    inverse) and ASSERTS bit-exactness against the snapshot -- if it is not exact the snapshot is copied back, so no
    later arm can run on a damaged model, and `restore()` returns False (the caller records it; the judge then reads
    M5 as NOT EVALUABLE, never as detected). A head reachable under two names is swapped ONCE."""

    def __init__(self, heads: dict, columns: dict, cls_start: int):
        if set(columns) != set(M5_CLASSES):
            raise ValueError(f"M5 columns must name exactly {M5_CLASSES}, got {sorted(columns)}")
        self.columns = {c: int(columns[c]) for c in M5_CLASSES}
        self.cls_start = int(cls_start)
        self.heads = dict(heads)
        self._unique, seen = [], set()
        for name, h in self.heads.items():
            if id(h) not in seen:
                seen.add(id(h))
                self._unique.append((name, h))
        self.same_module = len(self._unique) != len(self.heads)
        self._snap = None
        self.took_effect = False

    @staticmethod
    def find_heads(model) -> dict:
        """{'agent': core.agent_head, 'box3d': _perception.box_dec} -- BOTH must exist (the trainer's accessors)."""
        agent = getattr(getattr(model, "core", model), "agent_head", None)
        br = getattr(model, "_perception", None)
        box = getattr(br, "box_dec", None) if br is not None else None
        heads = {"agent": agent, "box3d": box}
        gone = [n for n, h in heads.items() if h is None]
        if gone:
            raise ValueError(f"M5 swaps BOTH slot heads; this model has no {gone}")
        return heads

    @classmethod
    def for_model(cls, model, classes=None) -> "ClassColumnSwap":
        return cls(cls.find_heads(model), m5_class_indices(classes), m5_cls_slice().start)

    def _rows(self) -> tuple:
        return (self.cls_start + self.columns["bus"], self.cls_start + self.columns["heavy_truck"])

    def _check(self):
        a, b = self._rows()
        for name, h in self._unique:
            lin = getattr(h, "head", None)
            if not isinstance(lin, torch.nn.Linear) or lin.bias is None:
                raise ValueError(f"M5: slot head {name!r} has no nn.Linear `.head` with a bias")
            if lin.weight.shape[0] <= max(a, b):
                raise ValueError(f"M5: slot head {name!r} has {lin.weight.shape[0]} output rows, rows {a},{b} asked")

    def _swap(self):
        a, b = self._rows()
        with torch.no_grad():
            for _name, h in self._unique:
                for t in (h.head.weight, h.head.bias):
                    t[[a, b]] = t[[b, a]]          # the right-hand side is evaluated (copied) first

    def _put_back(self):
        with torch.no_grad():
            for name, h in self._unique:
                w0, b0 = self._snap[name]
                h.head.weight.copy_(w0)
                h.head.bias.copy_(b0)

    def apply(self) -> "ClassColumnSwap":
        if self._snap is not None:
            raise RuntimeError("M5: apply() twice without restore()")
        self._check()
        self._snap = {name: (h.head.weight.detach().clone(), h.head.bias.detach().clone())
                      for name, h in self._unique}
        try:
            self._swap()
        except Exception:
            self._put_back()
            self._snap = None
            raise
        a, b = self._rows()
        ok = a != b
        for name, h in self._unique:
            w0, b0 = self._snap[name]
            w1, b1 = h.head.weight.detach(), h.head.bias.detach()
            ok = ok and bool(torch.equal(w1[a], w0[b]) and torch.equal(w1[b], w0[a])
                             and torch.equal(b1[a], b0[b]) and torch.equal(b1[b], b0[a]))
        self.took_effect = bool(ok)
        return self

    def restore(self) -> bool:
        """-> True iff the heads are BIT-IDENTICAL to the snapshot after swapping back (else forced back by copy)."""
        if self._snap is None:
            raise RuntimeError("M5: restore() before apply()")
        self._swap()
        exact = all(torch.equal(h.head.weight.detach(), self._snap[n][0])
                    and torch.equal(h.head.bias.detach(), self._snap[n][1]) for n, h in self._unique)
        if not exact:
            self._put_back()
        self._snap = None
        self.took_effect = False
        return bool(exact)

    def describe(self) -> dict:
        a, b = self._rows()
        return {"classes": dict(self.columns), "cls_start": self.cls_start, "param_rows": [a, b],
                "same_module_under_two_names": self.same_module,
                "heads": {n: {"type": type(h).__name__, "weight_shape": list(h.head.weight.shape),
                              "rows_swapped": [a, b]} for n, h in self._unique}}


# ----------------------------------------------------------------------------------- #
# A7.3 -- the detection packs banked (reported, never gating)                           #
# ----------------------------------------------------------------------------------- #
def pack_arrays(packs: list, prefix: str) -> dict:
    """The registered fields of a list of `detection_metrics.window_packs` packs as npz arrays under `prefix__*`:
    per slot the presence `logit` [W, N], `xy` [W, N, 2] and the class argmax `cls` [W, N]; the GT `gt_xy`, `gt_cls`,
    `gt_pos`, `gt_ign` concatenated over windows with the offsets `gt_off` [W + 1]; and the window keys `batch`
    (the eval batch the pack came from) and `ep_sha12` (sha12 of the pack's `ep` field -- the trainer's stable
    episode id; the raw value is NEVER stored; '' if the pack carries none). The row order is the pack order
    (batch-major), which is the window order whenever every window has an agent label."""
    n = len(packs)
    f32, i16 = np.float32, np.int16
    gt_n = [int(len(np.asarray(p["gt_cls"]))) for p in packs]
    off = np.concatenate([[0], np.cumsum(gt_n)]).astype(np.int64)

    def cat(field, dtype, tail=()):
        parts = [np.asarray(p[field], dtype).reshape((-1, *tail)) for p in packs]
        return np.concatenate(parts) if parts else np.zeros((0, *tail), dtype)

    ep = [("" if p.get("ep") is None else L.sha12(str(p["ep"]))) for p in packs]
    return {f"{prefix}__logit": np.stack([np.asarray(p["logit"], f32) for p in packs]),
            f"{prefix}__xy": np.stack([np.asarray(p["xy"], f32) for p in packs]),
            f"{prefix}__cls": np.stack([np.asarray(p["cls"], i16) for p in packs]),
            f"{prefix}__gt_off": off,
            f"{prefix}__gt_xy": cat("gt_xy", f32, (2,)),
            f"{prefix}__gt_cls": cat("gt_cls", i16),
            f"{prefix}__gt_pos": cat("pos", bool),
            f"{prefix}__gt_ign": cat("ign", bool),
            f"{prefix}__batch": np.asarray([int(p.get("_batch", -1)) for p in packs], np.int32),
            f"{prefix}__ep_sha12": np.asarray(ep, dtype="U12") if n else np.zeros(0, "U12")}


def unpack_detection_packs(path, arm: str, head: str) -> list:
    """Read one (arm, head) back from a banked `.npz` -> a list of dicts with the registered fields."""
    with np.load(str(path), allow_pickle=False) as z:
        pre = f"{arm}__{head}"
        off = z[f"{pre}__gt_off"]
        out = []
        for i in range(len(off) - 1):
            lo, hi = int(off[i]), int(off[i + 1])
            out.append({"logit": z[f"{pre}__logit"][i], "xy": z[f"{pre}__xy"][i], "cls": z[f"{pre}__cls"][i],
                        "gt_xy": z[f"{pre}__gt_xy"][lo:hi], "gt_cls": z[f"{pre}__gt_cls"][lo:hi],
                        "gt_pos": z[f"{pre}__gt_pos"][lo:hi], "gt_ign": z[f"{pre}__gt_ign"][lo:hi],
                        "batch": int(z[f"{pre}__batch"][i]), "ep_sha12": str(z[f"{pre}__ep_sha12"][i])})
        return out


def bank_detection_packs(arms: dict, inrun: dict, heads, path, not_run: dict | None = None) -> dict:
    """SPEC A7.3: bank the per-window detection packs of inference seed 0 (`s0`) and of `fp32_s0`, both slot heads,
    into the sibling `.npz` at `path`. REPORTED, NEVER GATING -- and NEVER SILENT: every (arm, head) cell is named
    in the report with its status. `BANKED` = packs written and their count equals the in-run row's own
    `eval_{head}_n_windows`; `COUNT_MISMATCH` = written, but the count differs (both numbers printed); `MISSING` =
    the arm did not run (`not_run[arm]` says why) or ran and produced no pack for that head; `ERROR` = building the
    arrays raised. `gaps` lists every cell that is not BANKED."""
    rep = {"path": str(path), "heads": list(heads), "arms": {}, "gaps": [], "npz": None, "status": None,
           "gating": False}
    if not heads:
        rep["status"] = "NO DETECTION HEADS (nothing to bank)"
        return rep
    arrays = {}
    for arm in ("s0", "fp32_s0"):
        for hd in heads:
            packs = list((arms.get(arm) or {}).get(hd) or [])
            exp = inrun.get(f"eval_{hd}_n_windows")
            cell = {"n_packs": len(packs), "expected_n_windows_in_run_row": exp}
            if (not_run or {}).get(arm):
                cell.update(status="MISSING", why=str(not_run[arm]))
            elif not packs:
                cell.update(status="MISSING", why="the arm ran but produced no detection pack for this head")
            else:
                try:
                    arrays.update(pack_arrays(packs, f"{arm}__{hd}"))
                    cell["status"] = "BANKED"
                    if not _isnull(exp) and exp is not None and int(round(float(exp))) != len(packs):
                        cell.update(status="COUNT_MISMATCH",
                                    why=f"{len(packs)} packs banked, the in-run row counts {exp} windows")
                except Exception as exc:                    # noqa: BLE001 -- reported, never silent
                    cell.update(status="ERROR", why=f"{type(exc).__name__}: {str(exc)[:200]}")
            rep["arms"][f"{arm}.{hd}"] = cell
            if cell["status"] != "BANKED":
                rep["gaps"].append({"cell": f"{arm}.{hd}", "status": cell["status"], "why": cell.get("why")})
    if arrays:
        meta = {"schema": "g0-a7-packs/1", "fields": list(A7_PACK_FIELDS), "arms": ["s0", "fp32_s0"],
                "heads": list(heads), "spec": "SPEC.md AMENDMENT A7.3",
                "ep_sha12": "sha12 of str(pack['ep']) (the trainer's stable episode id); no raw clip id is stored",
                "cells": {k: v["n_packs"] for k, v in rep["arms"].items()}}
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(p.name + ".part")
            with open(tmp, "wb") as fh:
                np.savez_compressed(fh, meta=np.asarray(json.dumps(meta)), **arrays)
            tmp.replace(p)
            rep["npz"] = {"path": str(p), "bytes": p.stat().st_size, "sha256": L.sha256_file(p),
                          "keys": len(arrays) + 1}
        except Exception as exc:                            # noqa: BLE001 -- reported, never silent
            rep["npz"] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    rep["status"] = ("ALL BANKED" if (not rep["gaps"] and rep["npz"] and "error" not in rep["npz"])
                     else "GAPS: " + ", ".join(f"{g['cell']}={g['status']}" for g in rep["gaps"])
                     + ("; npz NOT WRITTEN" if not rep["npz"] or "error" in rep["npz"] else ""))
    return rep


# ----------------------------------------------------------------------------------- #
# A7.4 -- the inference-seed draw, reported as ONE draw (diagnostic, never gating)      #
# ----------------------------------------------------------------------------------- #
def seed_group_report(by_seed: dict) -> dict:
    """The seed-group F test, seeds 0-7 vs 8-23, on `eval_traj` -- `seed_group_check.py`'s own test, unchanged (the
    form of `PREREG_SEED_GROUP_CHECK_50400.md`): the row mean over the 8 batches of each seed's per-batch `traj`
    (full precision, with the 5-dp `eval_traj` as a rounding control), the one-sided F test var(8-23)/var(0-7) at
    df (15, 7), and Levene (median-centred). Verdict A / B / INCONCLUSIVE as the prereg defines them. Needs seeds
    0..23. It never changes a G0 verdict."""
    out = {"term": "eval_traj", "prereg": "raw/PREREG_SEED_GROUP_CHECK_50400.md", "gating": False}
    seeds = sorted(int(s) for s in by_seed)
    if seeds != list(range(24)):
        return {**out, "status": "NOT COMPUTED", "why": f"needs inference seeds 0..23, this run has {len(seeds)}"}
    try:
        import seed_group_check as SGC
        bs = {str(int(k)): v for k, v in by_seed.items()}
        rm = SGC._row_means_g0({"by_seed": bs}, seeds)
        tf = SGC.test([rm[s]["row_mean_full"] for s in range(8)], [rm[s]["row_mean_full"] for s in range(8, 24)])
        t5 = SGC.test([rm[s]["row_5dp"] for s in range(8)], [rm[s]["row_5dp"] for s in range(8, 24)])
    except ZeroDivisionError:
        return {**out, "status": "NOT COMPUTABLE", "why": "zero variance among the seeds 0-7 row means"}
    except Exception as exc:                                # noqa: BLE001 -- a diagnostic never breaks G0
        return {**out, "status": "ERROR", "why": f"{type(exc).__name__}: {str(exc)[:200]}"}
    verdict = tf["verdict"] if tf["verdict"] == t5["verdict"] else f"DISAGREE full={tf['verdict']} 5dp={t5['verdict']}"
    return {**out, "status": "COMPUTED", "VERDICT": verdict, "test_full": tf, "test_5dp": t5,
            "rounding_control_max_abs": max(abs(v["row_mean_full"] - v["row_5dp"]) for v in rm.values())}


def traj_cells(by_seed: dict) -> dict:
    """{seed: [per-batch `traj`, full precision]} from a G0 artifact's (or run's) `by_seed`."""
    out = {}
    for s, v in by_seed.items():
        vals = [b.get("traj") for b in (v.get("per_batch") or [])]
        if vals and all(isinstance(x, (int, float)) for x in vals):
            out[int(s)] = [float(x) for x in vals]
    return out


def traj_cells_from_diag(diag: dict) -> dict:
    """{seed: per-batch traj} from a `g0_diag_r7.py` artifact (arms `seed{s}`, key `per_batch_traj`)."""
    out = {}
    for name, a in (diag.get("arms") or {}).items():
        if name.startswith("seed") and name[4:].isdigit() and a.get("per_batch_traj"):
            out[int(name[4:])] = [float(x) for x in a["per_batch_traj"]]
    return out


def _nan_none(x):
    return None if (x is None or (isinstance(x, float) and x != x)) else float(x)


def seed_draw_correlation(cur: dict, earlier: dict) -> dict:
    """A7.4: the per-(seed, batch) correlation of `eval_traj` DEVIATIONS between two G0 artifacts on the same
    windows. Deviation = the cell's `traj` minus the mean over that artifact's OWN seeds at the same batch (the
    batch/window effect removed, the DDIM draw left). Pearson r over every (seed, batch) cell both artifacts hold
    (n = common seeds x batches; 24 x 8 = 192 for two 24-seed artifacts), plus r over the per-seed row means. A
    high r says the draw is a function of the seed VALUE and is reused at every checkpoint."""
    from scipy import stats

    def dev(c):
        nbs = {len(v) for v in c.values()}
        if len(nbs) != 1:
            raise ValueError("seeds with unequal batch counts")
        nb = nbs.pop()
        mb = [statistics.fmean(c[s][b] for s in c) for b in range(nb)]
        return {s: [c[s][b] - mb[b] for b in range(nb)] for s in c}, nb
    dc, nbc = dev(cur)
    de, nbe = dev(earlier)
    if nbc != nbe:
        return {"status": "NOT COMPARABLE", "why": f"{nbc} vs {nbe} batches per seed"}
    common = sorted(set(dc) & set(de))
    if len(common) < 3:
        return {"status": "NOT COMPUTABLE", "why": f"only {len(common)} common inference seeds"}
    x = [dc[s][b] for s in common for b in range(nbc)]
    y = [de[s][b] for s in common for b in range(nbc)]
    import warnings
    with warnings.catch_warnings():                      # a constant array is a NULL correlation, reported as None
        warnings.simplefilter("ignore")
        r, p = stats.pearsonr(x, y)
        rs, ps = stats.pearsonr([statistics.fmean(cur[s]) for s in common],
                                [statistics.fmean(earlier[s]) for s in common])
    return {"status": "COMPUTED", "n_cells": len(x), "n_seeds_common": len(common), "n_batches": nbc,
            "r_cell": _nan_none(r), "p_cell": _nan_none(p), "r_seed_row_mean": _nan_none(rs),
            "p_seed_row_mean": _nan_none(ps),
            "deviation": "traj(seed, batch) - mean over the artifact's own seeds at that batch"}


def seed_draw_reports(rec: dict, by_seed: dict, earlier_arg: str | None = None, root=None) -> dict:
    """A7.4 item 2: the correlation against every EARLIER G0 artifact that exists. Default discovery:
    `raw/step*/g0.json` of this package; an artifact is used only if it is on the SAME 128 windows
    (`perm_sha256`) and is not this checkpoint (`ckpt_md5`); every skipped artifact is listed with the reason.
    `earlier_arg` (comma-separated paths) replaces the discovery. None found -> status NO EARLIER G0 (not an error)."""
    out = {"gating": False, "pairs": [], "skipped": []}
    if earlier_arg:
        paths = [Path(p.strip()) for p in earlier_arg.split(",") if p.strip()]
    else:
        paths = sorted((Path(root) if root else HERE.parent / "raw").glob("step*/g0.json"))
    cur = traj_cells(by_seed)
    for p in paths:
        try:
            g = json.load(open(p, encoding="utf-8"))
            if g.get("ckpt_md5") == rec.get("ckpt_md5"):
                out["skipped"].append({"path": str(p), "why": "the same checkpoint"})
                continue
            if g.get("perm_sha256") != rec.get("perm_sha256"):
                out["skipped"].append({"path": str(p), "why": "not the same 128 windows (perm_sha256 differs)"})
                continue
            res = seed_draw_correlation(cur, traj_cells(g.get("by_seed") or {}))
            out["pairs"].append({"earlier": str(p), "earlier_step": g.get("step"),
                                 "earlier_ckpt_md5": g.get("ckpt_md5"), **res})
        except Exception as exc:                            # noqa: BLE001 -- a diagnostic never breaks G0
            out["skipped"].append({"path": str(p), "why": f"{type(exc).__name__}: {str(exc)[:200]}"})
    out["status"] = "COMPUTED" if out["pairs"] else "NO EARLIER G0 (comparable) FOUND"
    return out


def a7_reports(rec, by_seed, inrun, tr, seeds, packs_s0, packs_fp32, out_p, earlier_arg, no_a6) -> dict:
    """SPEC A7.3 + A7.4 for one G0 run: the packs, the seed-group test, the cross-checkpoint draw correlation.
    Every part is REPORTED, never gating, and a failure inside any part is recorded in the artifact instead of
    raised (a diagnostic can never cost the G0 its artifact)."""
    rep = {"gating": False,
           "inference_seed_floor": "the battery's inference-seed floor (seeds 0 and 1) is ONE draw of the DDIM noise, "
                                   "reused at every checkpoint (SPEC A7.4): quote it as one draw"}
    heads = tuple(getattr(getattr(tr, "_det_metrics", None), "HEADS", ()) or ())
    not_run = {}
    if 0 not in [int(s) for s in seeds]:
        not_run["s0"] = "inference seed 0 was not run (--seeds)"
    arm = (rec.get("a6") or {}).get(A6_ARM) or {}
    if no_a6:
        not_run["fp32_s0"] = "--no-a6: the fp32_s0 arm did not run"
    elif not arm.get("row"):
        not_run["fp32_s0"] = f"the fp32_s0 arm did not complete: {str(arm.get('raised'))[:200]}"
    try:
        rep["packs"] = bank_detection_packs({"s0": packs_s0, "fp32_s0": packs_fp32}, inrun, heads,
                                            Path(out_p).with_suffix(A7_PACKS_SUFFIX), not_run)
    except Exception as exc:                                # noqa: BLE001
        rep["packs"] = {"status": f"ERROR {type(exc).__name__}: {str(exc)[:200]}", "gating": False}
    print(f"[g0] A7.3 detection packs: {rep['packs'].get('status')}", flush=True)
    rep["seed_group"] = seed_group_report(by_seed)
    sg = rep["seed_group"]
    print(f"[g0] A7.4 seed-group (seeds 0-7 vs 8-23, eval_traj): {sg.get('status')} "
          f"{sg.get('VERDICT') or sg.get('why') or ''}", flush=True)
    try:
        rep["seed_draw_correlation"] = seed_draw_reports(rec, by_seed, earlier_arg)
    except Exception as exc:                                # noqa: BLE001
        rep["seed_draw_correlation"] = {"status": f"ERROR {type(exc).__name__}: {str(exc)[:200]}", "gating": False}
    print(f"[g0] A7.4 seed-draw correlation: {rep['seed_draw_correlation'].get('status')}", flush=True)
    return rep



class TacCellCapture:
    """Records, per `compute_losses_v3` call, exactly what `tactical_behaviour_losses` received:
    validity logits, confidence logits, goal_y, goal_w, the class mask (A6's per-cell evidence).
    Installed on the trainer module's `v6tac` (the name `compute_losses_v3` calls through)."""

    def __init__(self, tr):
        self.tr = tr
        self.calls = []
        self.errors = []
        self._orig = None

    def install(self):
        orig = self.tr.v6tac.tactical_behaviour_losses
        self._orig = orig

        def tbl(out, **kw):
            total, tele = orig(out, **kw)
            # ⛔ the RECORDING can never break the forward: an error here is logged and A6 then fails
            # closed (missing cells), while the gate's own replay is untouched
            try:
                cm = kw.get("goal_class_mask")
                self.calls.append({
                    "goal_logits": out["goal_logits"].detach().float().cpu().tolist(),
                    "goal_conf": out["goal_conf"].detach().float().cpu().tolist(),
                    "goal_y": kw["goal_y"].detach().float().cpu().tolist(),
                    "goal_w": kw["goal_w"].detach().float().cpu().tolist(),
                    "class_mask": (None if cm is None else (cm.detach().float().cpu().tolist()
                                                             if torch.is_tensor(cm) else list(cm))),
                    "tac_goal_conf_bce": float(tele["tac_goal_conf_bce"])})
            except Exception as exc:                    # noqa: BLE001 -- recorded, fail closed
                self.errors.append(f"{type(exc).__name__}: {str(exc)[:200]}")
            return total, tele
        self.tr.v6tac.tactical_behaviour_losses = tbl
        return self

    def remove(self):
        if self._orig is not None:
            self.tr.v6tac.tactical_behaviour_losses = self._orig
            self._orig = None


def _cells_flat(call: dict):
    """-> list of (l, c, y, cw) for every cell of one batch, cw = goal_w * class_mask (the weight the
    trainer's conf term applies, refcv6_tactical.py:826-830), and sum(cw)."""
    m = call.get("class_mask")
    out, s = [], 0.0
    for lr, cr, yr, wr in zip(call["goal_logits"], call["goal_conf"], call["goal_y"], call["goal_w"]):
        for j, (l, c, y, w) in enumerate(zip(lr, cr, yr, wr)):
            cw = float(w) * (1.0 if m is None else float(m[j]))
            out.append((float(l), float(c), float(y), cw))
            s += cw
    return out, s


def _valid(l: float) -> bool:
    """`sigmoid(l) >= 0.5` as the trainer evaluates it in float32 (refcv6_tactical.py:824)."""
    if l < -80.0:
        return False
    return float(np.float32(1.0 / (1.0 + math.exp(-l)))) >= 0.5


def _correct(l: float, y: float) -> float:
    """refcv6_tactical.py:824-825: (sigmoid(l) >= 0.5) == y, as 0/1 (a soft y is never 'correct')."""
    return 1.0 if y in (0.0, 1.0) and (_valid(l) == (y == 1.0)) else 0.0


def conf_bce_from_cells(call: dict) -> float:
    """The batch's `tac_goal_conf_bce`, recomputed from the captured cells in float64 (a CONTROL: it
    must equal the trainer's own value to float32 precision)."""
    cells, s = _cells_flat(call)
    tot = 0.0
    for l, c, y, cw in cells:
        if cw <= 0:
            continue
        correct = _correct(l, y)
        # BCE-with-logits(c, t) = softplus(c) - t*c  (numerically stable form)
        sp = max(c, 0.0) + math.log1p(math.exp(-abs(c)))
        tot += cw * (sp - correct * c)
    return tot / max(s, 1.0)


def threshold_interval(calls_s0: list, calls_alt: list, k: float = A6_K) -> dict:
    """SPEC A6 item 2 (pure). The values of the 8-batch MEAN of `tac_goal_conf_bce` reachable from the
    replay (`calls_s0`) by flipping the validity decision of any UNDECIDABLE supervised cell, i.e. one
    whose replay logit |l| <= k * delta, with delta = max |l_s0 - l_alt| over supervised cells (the
    largest validity-logit move the numerics-only arm `calls_alt` produced on this checkpoint).
    Flipping cell i changes its batch value by (2*correct_i - 1) * c_i * cw_i / sum(cw_b): a correct
    cell (target 1, term softplus(-c)) becomes incorrect (target 0, term softplus(c)), a change of
    softplus(c) - softplus(-c) = +c; the reverse flip is -c. A soft y (never 'correct') cannot flip."""
    if len(calls_s0) != len(calls_alt) or not calls_s0:
        raise ValueError(f"A6: {len(calls_s0)} replay batches vs {len(calls_alt)} arm batches")
    nb = len(calls_s0)
    delta = 0.0
    flat = []
    for a, b in zip(calls_s0, calls_alt):
        ca, sa = _cells_flat(a)
        cb, _ = _cells_flat(b)
        if len(ca) != len(cb):
            raise ValueError("A6: replay and arm batches have different cell counts")
        flat.append((ca, sa))
        for (la, _c, _y, w), (lb, _c2, _y2, _w2) in zip(ca, cb):
            if w > 0:
                delta = max(delta, abs(la - lb))
    thr = k * delta
    t0 = sum(conf_bce_from_cells(c) for c in calls_s0) / nb
    lo = hi = t0
    und, flipped_alt = [], 0
    for bi, ((cells, s), b) in enumerate(zip(flat, calls_alt)):
        cb, _ = _cells_flat(b)
        for ci, ((l, c, y, w), (lb, _c, _y, _w)) in enumerate(zip(cells, cb)):
            if w <= 0:
                continue
            if _valid(l) != _valid(lb):
                flipped_alt += 1
            if abs(l) <= thr and y in (0.0, 1.0):
                correct = _correct(l, y)
                d = (2.0 * correct - 1.0) * c * w / max(s, 1.0) / nb
                lo += min(0.0, d)
                hi += max(0.0, d)
                und.append({"batch": bi, "cell": ci, "logit": l, "conf": c, "delta_mean": d})
    n_sup = sum(1 for cells, _s in flat for (_l, _c, _y, w) in cells if w > 0)
    return {"t_replay": t0, "lo": lo, "hi": hi, "delta_logit_max": delta, "k": k, "threshold": thr,
            "n_supervised": n_sup, "n_undecidable": len(und), "n_flipped_by_arm": flipped_alt,
            "undecidable": und[:40]}


def term_class(k: str) -> str:
    """SPEC §2's table. ⚠️ A standalone `n` / `npos` TOKEN marks a count (a raw substring test
    matches `lon_...`; the refcv6 G0 found that bug before its first output)."""
    if k in EXCLUDED_EXACT or "_calib_" in k:
        return "EXCLUDED"
    s = k[len("eval_"):] if k.startswith("eval_") else k
    if k.endswith("_conf_ratio_alarm"):
        return "DERIVED"                      # judged through conf_ratio (SPEC §2)
    if k.endswith(MODEL_DEP_COUNT_SUFFIX):
        return "DETECTION"
    toks = s.split("_")
    if (k in COUNT_EXACT or "n" in toks or "npos" in toks or s.startswith("agent_rows_")
            or s.startswith("n_map_hires_cells")):
        return "COUNT"
    if any(t in k for t in DET_TOKENS) and (s.startswith("agent_") or s.startswith("box3d_")):
        return "DETECTION"
    if (s in MATCHED_BASE or s.startswith("agent_presence_layer") or s.startswith("agent_layer")
            or s.startswith("box3d_presence_layer") or s.startswith("box3d_layer")):
        return "MATCHED"
    for pre in ("map_hires_inter_", "map_hires_union_", "map_hires_interraw_",
                "map_hires_unionraw_"):
        if s.startswith(pre):
            return "MAP10_COUNTS"
    if s.startswith("map_hires_iou_") or s.startswith("map_hires_iouraw_"):
        return "MAP10_IOU"
    return "SMOOTH_OR_STOCHASTIC"


# --------------------------------------------------------------------------------------- #
# the LAW-only future frame (compute_losses_v3 reads fut_frames[:, LAW_AHEAD-1] only, :4685) #
# --------------------------------------------------------------------------------------- #
class LawOnly:
    def __init__(self, u8_frame, law_idx):
        self.u8 = u8_frame
        self.law_idx = int(law_idx)


class LawOnlyDevice:
    def __init__(self, dev_frame, law_idx):
        self.x = dev_frame
        self.law_idx = law_idx

    def __getitem__(self, key):
        if not (isinstance(key, tuple) and len(key) == 2 and key[0] == slice(None)
                and int(key[1]) == self.law_idx):
            raise KeyError(f"LawOnlyDevice holds only [:, {self.law_idx}], asked {key}")
        return self.x


_PATCHED = {}


def patch_frames_to_device(tr):
    if "f2d" in _PATCHED:
        return
    orig = tr.frames_to_device

    def f(x, device):
        if isinstance(x, LawOnly):
            return LawOnlyDevice(orig(x.u8, device), x.law_idx)
        return orig(x, device)
    tr.frames_to_device = f
    _PATCHED["f2d"] = orig


def make_g0_dataset_cls(tr, law_ahead: int):
    class G0Windows(tr.V3Dataset):
        def _window_u8(self, i: int) -> dict:          # refb_train.py:220-231, future cut to LAW
            e_i, t = self.index[i]
            ep = self.episodes[e_i]
            w = self.window
            return {
                "frames": ep.frames[t:t + w],
                "actions": ep.actions[t:t + w],
                "future_frames": ep.frames[t + w:t + w + law_ahead],
                "future_actions": ep.actions[t + w:t + w + self.max_horizon],
                "future_poses": ep.poses[t + w:t + w + self.max_horizon],
                "pose_last": ep.poses[t + w - 1],
                "episode_id": ep.episode_id,
            }
    return G0Windows


def collate(e_ds, idx_list, law_idx):
    import torch.utils.data as tud
    items = [e_ds[i] for i in idx_list]
    eb = tud.default_collate(items)
    ff = eb["future_frames"]
    eb["future_frames"] = LawOnly(ff[:, law_idx].clone(), law_idx)
    del ff
    return eb


class MmapBatches:
    """SPEC AMENDMENT A5 item 5: the collated batches written ONCE to disk, read back through a
    file-backed mmap on every use (indexing or iteration). A batch file that already exists is reused
    (same checkpoint-md5-keyed directory), so a crashed G0 re-collates nothing. `LawOnly` is stored as
    its uint8 tensor + index and rebuilt on load. The content is bit-identical to the old in-RAM list
    (`g0_diag_r7.py`'s `s0` arm reproduces G0's stored seed-0 row through the same path)."""

    def __init__(self, root: Path, makers):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.paths = []
        for i, make in enumerate(makers):
            p = self.root / f"batch_{i}.pt"
            if not p.exists():
                eb = make()
                ser = {k: ({"__lawonly__": v.u8, "law_idx": v.law_idx} if isinstance(v, LawOnly) else v)
                       for k, v in eb.items()}
                torch.save(ser, str(p) + ".part")
                Path(str(p) + ".part").replace(p)
                del eb, ser
            self.paths.append(p)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        d = torch.load(str(self.paths[i]), map_location="cpu", weights_only=False, mmap=True)
        return {k: (LawOnly(v["__lawonly__"], v["law_idx"])
                    if isinstance(v, dict) and "__lawonly__" in v else v) for k, v in d.items()}

    def __iter__(self):
        return (self[i] for i in range(len(self.paths)))

    def remove(self):
        for p in self.paths:
            try:
                p.unlink()
            except OSError:
                pass
        try:
            self.root.rmdir()
        except OSError:
            pass


class RamBatches:
    """The pre-A5 form (2026-10-04 lever, memory not numbers): the collated batches held in RAM, with
    MmapBatches' interface. Selected ONLY by `--batch-cache ram` or env `REFCV7_G0_BATCH_CACHE=ram` --
    for a host whose free DISK is short (D: 7.7 GB on 2026-10-04) while its free COMMIT is not. The
    tensors are the same objects `collate` returns; G0 at step 1,500 ran this way."""

    def __init__(self, makers):
        self.root = "RAM"
        self.items = [make() for make in makers]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    def __iter__(self):
        return iter(self.items)

    def remove(self):
        self.items = []


def make_batches(spec, default_root: Path, makers):
    """`spec` None -> MmapBatches at `default_root`; 'ram' -> RamBatches; else MmapBatches at `spec`.
    The env `REFCV7_G0_BATCH_CACHE` is read when `spec` is None (so a chain can select it unchanged)."""
    spec = spec or os.environ.get("REFCV7_G0_BATCH_CACHE") or None
    if spec == "ram":
        return RamBatches(makers)
    return MmapBatches(Path(spec) if spec else default_root, makers)


def sub_batch(eb, n, rows=None):
    rows = list(range(n)) if rows is None else rows
    o = {}
    for k, v in eb.items():
        if torch.is_tensor(v) and v.dim() >= 1:
            o[k] = v[rows]
        elif isinstance(v, LawOnly):
            o[k] = LawOnly(v.u8[rows], v.law_idx)
        elif isinstance(v, (list, tuple)):
            o[k] = type(v)(v[i] for i in rows)
        else:
            o[k] = v
    return o


# --------------------------------------------------------------------------------------- #
# requires_grad AS TRAINED (SPEC §2)                                                        #
# --------------------------------------------------------------------------------------- #
def training_flags(model) -> dict:
    """{param name: requires_grad as the TRAINER holds it}: True, except parameters under a module
    that carries `_gradreach.GRAD_UNREACHABLE_FLAG` (`declare_grad_unreachable` froze them)."""
    from tanitad.models import _gradreach as _gr
    dead = _gr.grad_unreachable_prefixes(model)
    out = {}
    for n, _p in model.named_parameters():
        out[n] = not any(n == pre or n.startswith(pre + ".") for pre in dead)
    return {"flags": out, "unreachable_prefixes": dead}


def set_flags(model, flags: dict):
    for n, p in model.named_parameters():
        p.requires_grad_(bool(flags[n]))


def buffer_digest(model) -> str:
    h = hashlib.sha256()
    for n, b in model.named_buffers():
        if b is None:
            continue
        h.update(n.encode())
        h.update(b.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------------------- #
# the in-run eval, replayed                                                                 #
# --------------------------------------------------------------------------------------- #
def scalar_items(el: dict) -> dict:
    out = {}
    for k, v in el.items():
        if torch.is_tensor(v) and v.ndim == 0:
            out[k] = float(v.detach())
        elif isinstance(v, (int, float, bool)) and not str(k).startswith("_"):
            out[k] = float(v)
    return out


def run_eval(tr, model, batches, device, mode, abl, seed, batch_size, packs_out=None):
    """refc_v3_train.py:9658-9747 with the RNG seeded per inference seed.

    `packs_out` (SPEC A7.3, optional): a dict that is filled with {head: [pack, ...]} -- shallow copies of the
    `detection_metrics.window_packs` output with the batch index added as `_batch`. The pooled `det_packs` the
    row is summarised from are the ORIGINAL objects, untouched, so capturing cannot change a number."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    acc, nb, per_batch, det_packs = {}, 0, [], {}
    with torch.no_grad():
        for eb in batches():
            el = tr.compute_losses_v3(model, eb, device, mode=mode, ablate_frames=abl)
            row = {}
            for k, v in el.items():
                if torch.is_tensor(v) and v.ndim == 0:
                    acc[k] = acc.get(k, 0.0) + float(v.detach())
                    row[k] = float(v.detach())
                elif isinstance(v, (int, float, bool)):
                    acc[k] = acc.get(k, 0.0) + float(v)
                    row[k] = float(v)
            for hd in tr._det_metrics.HEADS:
                if el.get(f"_det_pack_{hd}"):
                    det_packs.setdefault(hd, []).extend(el[f"_det_pack_{hd}"])
                    if packs_out is not None:
                        packs_out.setdefault(hd, []).extend(
                            {**pk, "_batch": nb} for pk in el[f"_det_pack_{hd}"])
            per_batch.append(row)
            nb += 1
            del el
    erow = tr._eval_row_from_acc(acc, nb, model)
    for hd, pk in det_packs.items():
        for dk, dv in tr._det_metrics.summarise(pk, hd).items():
            erow[dk] = (None if (isinstance(dv, float) and dv != dv) else round(float(dv), 5))
    erow.update(eval_batches=nb, eval_windows=nb * int(batch_size))
    return erow, per_batch, {hd: len(v) for hd, v in det_packs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--micro", default="2,2,3,3,3,3")
    ap.add_argument("--mutations", default="m1,m2,m4,m5",
                    help="m1 (must be detected, else VOID), m2 / m4 / m5 power probes. m5 = SPEC A7.2: swap the bus "
                         "and heavy_truck class-logit columns in BOTH slot heads at inference seed 0, then restore")
    ap.add_argument("--earlier-g0", default=None,
                    help="SPEC A7.4 (diagnostic, never gating): comma-separated EARLIER G0 artifacts whose per-(seed, "
                         "batch) eval_traj deviations are correlated with this run's. Default: every raw/step*/g0.json "
                         "of this package on the SAME 128 windows (perm_sha256) with another checkpoint md5")
    ap.add_argument("--diagnostic-arms", default="eps0",
                    help="REPORTED, never judged (2026-10-04 diagnosis): 'eps0' = seed-0 replay with the "
                         "DDIM draw zeroed -- tells whether the in-run row sits at the noise-free decode")
    ap.add_argument("--skip-wrapper-control", action="store_true")
    ap.add_argument("--n-batches", type=int, default=0, help="PROBE ONLY: fewer batches")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-cache", default=None,
                    help="dir for the collated batches (SPEC A5 item 5; default next to --out)")
    ap.add_argument("--keep-batch-cache", action="store_true")
    ap.add_argument("--no-a6", action="store_true",
                    help="skip SPEC A6's measured numerics floor (the fp32_s0 arm + the per-cell capture "
                         "of the tactical conf term); without it G0-A6 is NOT EVALUABLE")
    a = ap.parse_args()
    t_all = time.time()
    tr = L.trainer()
    device = "cuda"
    spec = HERE.parent / "SPEC.md"
    rec = {"tool": "g0_refcv7.py", "spec_sha256": hashlib.sha256(spec.read_bytes()).hexdigest()
           if spec.exists() else None, "ckpt": a.ckpt, "ckpt_md5": L.md5_file(a.ckpt),
           "config": a.config, "config_md5": L.md5_file(a.config), "micro": a.micro,
           "seeds": a.seeds, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
           "tanitad_file": __import__("tanitad").__file__, "started": time.strftime("%FT%T"),
           "tf32": {"matmul": torch.backends.cuda.matmul.allow_tf32,
                    "cudnn": torch.backends.cudnn.allow_tf32}}
    out_p = Path(a.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    def bank():
        json.dump(rec, open(out_p.with_suffix(".partial.json"), "w", encoding="utf-8"), indent=1,
                  default=str)

    config = L.load_config(a.config)
    model, cfg, args, mrec = L.build_model(config, a.ckpt, device)
    rec["model"] = {k: mrec.get(k) for k in ("state_dict", "param_breakdown",
                                             "anchor_file_vs_ckpt_buffers", "declared_vs_built",
                                             "stamp_checks", "trunk_memory_levers_built",
                                             "trunk_memory_levers_run", "departures", "mode",
                                             "decoder_steps", "sampler", "build_s", "argv_remap")}
    step = int(a.step if a.step is not None else mrec["state_dict"]["step"])
    rec["step"] = step
    print(f"[g0] model built in {mrec['build_s']} s; strict missing="
          f"{len(mrec['state_dict']['missing'])} unexpected={len(mrec['state_dict']['unexpected'])} "
          f"step={step}; param_breakdown equal={mrec['param_breakdown']['equal']}", flush=True)
    # ---- requires_grad as trained ---------------------------------------------------- #
    tf = training_flags(model)
    loader_flags = {n: bool(p.requires_grad) for n, p in model.named_parameters()}
    rec["requires_grad"] = {"loader_true": sum(loader_flags.values()),
                            "trained_true": sum(tf["flags"].values()),
                            "n_params_tensors": len(tf["flags"]),
                            "unreachable_prefixes": tf["unreachable_prefixes"]}
    # ---- the in-run row -------------------------------------------------------------- #
    rows = [json.loads(ln) for ln in open(a.metrics, encoding="utf-8") if ln.strip()]
    ev = [r for r in rows if r.get("step") == step and "eval_loss" in r]
    if len(ev) != 1:
        raise SystemExit(f"[g0] {len(ev)} eval rows at step {step} in {a.metrics}")
    inrun = ev[0]
    rec["inrun_row_n_keys"] = len(inrun)
    # ---- eval dataset + the fixed subset --------------------------------------------- #
    t_ds = time.time()
    law_idx = int(tr.LAW_AHEAD) - 1
    e_ds, e_eps, drec = L.build_eval_dataset(model, cfg, args, config, with_perception_targets=True,
                                             dataset_cls=make_g0_dataset_cls(tr, int(tr.LAW_AHEAD)))
    rec["dataset"] = {k: drec.get(k) for k in ("n_episodes", "n_windows", "label_clock", "labels",
                                               "nav", "max_speed_v6", "agent_join", "map_fine",
                                               "join3d", "vis1", "camera_coverage")}
    B = int(args.batch)
    NB = int(a.n_batches or args.eval_batches)
    perm = L.inrun_eval_perm(e_ds, int(args.eval_batches), B)
    rec["perm_sha256"] = hashlib.sha256(json.dumps(perm).encode()).hexdigest()
    rec["n_batches"] = NB
    if a.n_batches:
        rec["PROBE_ONLY_fewer_batches"] = True
    print(f"[g0] eval dataset: {drec['n_episodes']} episodes -> {drec['n_windows']} windows "
          f"({time.time() - t_ds:.0f}s); subset {len(perm)}; batches {NB}", flush=True)
    t_c = time.time()
    # SPEC AMENDMENT A5 item 5 (memory, not numbers): the collated batches live on DISK and are read
    # back through a file-backed mmap (no commit charge) instead of ~3.9 GB of RAM; bit-identical.
    # 2026-10-04: `--batch-cache ram` (or env REFCV7_G0_BATCH_CACHE=ram) keeps them in RAM instead --
    # for a host short of DISK rather than commit; the tensors are identical either way
    cached = make_batches(a.batch_cache, out_p.parent / f"g0_batch_cache_{rec['ckpt_md5'][:8]}",
                          [lambda i=i: collate(e_ds, perm[i * B:(i + 1) * B], law_idx)
                           for i in range(NB)])
    rec["batch_cache"] = str(cached.root)
    rec["collate_s"] = round(time.time() - t_c, 1)
    print(f"[g0] collated {NB} batches in {rec['collate_s']} s (disk cache {cached.root})", flush=True)

    def batches():
        return iter(cached)
    patch_frames_to_device(tr)
    mode = getattr(args, "mode", "diffusion")
    abl = bool(getattr(args, "ablate_frames", False))
    sizes = [int(x) for x in a.micro.split(",")]
    set_flags(model, tf["flags"])
    bd0 = buffer_digest(model)
    torch.cuda.reset_peak_memory_stats()
    bank()
    # ---- requires_grad control: one 2-window forward, eps zeroed, both flag states ----- #
    orig_randn_like = torch.randn_like
    try:
        torch.randn_like = lambda x, *aa, **kk: torch.zeros_like(x)
        sb = sub_batch(cached[0], 2)
        res = {}
        for nm, fl in (("trained_flags", tf["flags"]), ("loader_flags", loader_flags)):
            set_flags(model, fl)
            torch.manual_seed(0)
            with torch.no_grad():
                res[nm] = scalar_items(tr.compute_losses_v3(model, sb, device, mode=mode,
                                                            ablate_frames=abl))
        set_flags(model, tf["flags"])
        diff = {k: (res["trained_flags"][k], res["loader_flags"].get(k))
                for k in res["trained_flags"] if res["trained_flags"][k] != res["loader_flags"].get(k)}
        rec["requires_grad_control"] = {"batch": 2, "ddim_eps": "zeros", "n_terms":
                                        len(res["trained_flags"]), "n_differ": len(diff),
                                        "differ_first20": dict(list(diff.items())[:20]),
                                        "max_rel_all": max((abs(a_ - b_) / max(abs(a_), 1e-12)
                                                            for a_, b_ in diff.values()
                                                            if b_ is not None), default=0.0),
                                        "differ_all_rel": {k: abs(a_ - b_) / max(abs(a_), 1e-12)
                                                           for k, (a_, b_) in diff.items()
                                                           if b_ is not None}}
        print(f"[g0] requires_grad control: {len(diff)} of {len(res['trained_flags'])} terms "
              f"differ between trained and loader flags (dev box)", flush=True)
    finally:
        torch.randn_like = orig_randn_like
    bank()
    # ---- the seeds -------------------------------------------------------------------- #
    mb = MicroBatchForward(model, sizes).install()
    seeds = [int(s) for s in a.seeds.split(",") if s.strip() != ""]
    by_seed = {}
    rec["a6"] = {"status": "skipped (--no-a6)"} if a.no_a6 else {"cells": {}}
    packs_s0, packs_fp32 = {}, {}            # SPEC A7.3: the detection packs of inference seed 0 / fp32_s0
    for s in seeds:
        t_s = time.time()
        # SPEC A6 (draft): seed 0 also records the tactical conf term's per-cell inputs (read-only hook)
        cap = TacCellCapture(tr).install() if (s == 0 and not a.no_a6) else None
        try:
            erow, pb, npk = run_eval(tr, model, batches, device, mode, abl, s, B,
                                     packs_out=(packs_s0 if s == 0 else None))
        finally:
            if cap is not None:
                cap.remove()
        if cap is not None:
            rec["a6"]["cells"]["s0"] = cap.calls
            rec["a6"]["capture_errors_s0"] = cap.errors
        dg = buffer_digest(model)
        by_seed[s] = {"row": erow, "per_batch": pb, "det_packs": npk,
                      "wall_s": round(time.time() - t_s, 1), "buffers_unchanged": dg == bd0}
        print(f"[g0] seed {s}: eval_loss {erow.get('eval_loss')} eval_traj {erow.get('eval_traj')} "
              f"({time.time() - t_s:.0f}s, peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB,"
              f" buffers unchanged {dg == bd0})", flush=True)
        rec["by_seed"] = {str(k): v for k, v in by_seed.items()}
        bank()
    rec["merge_rules_flagged"] = mb.flagged()
    rec["merge_rules_all"] = mb.rules
    rec["cuda_max_memory_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    # ---- mutations (M1 must FAIL; M2 / M4 power probes) ------------------------------ #
    muts = [m.strip() for m in a.mutations.split(",") if m.strip()]
    rec["mutations"] = {}
    from tanitad.models.timm_trunk import TimmResNetTrunk
    trunk = [m for m in model.modules() if isinstance(m, TimmResNetTrunk)][0]
    for m in muts:
        t_m = time.time()
        info = {}
        try:
            if m == "m1":
                old = int(getattr(trunk.cfg, "equalize_bottom_rows", 0) or 0)
                calls0 = int(getattr(trunk, "equalize_calls", 0))
                object.__setattr__(trunk.cfg, "equalize_bottom_rows", 0)
                info = {"what": "trunk bottom-row equalisation DROPPED (FIX-3 defect, "
                                "D-REFCV6-EQUALIZE-DROPPED)", "equalize_rows_as_built": old}
                erow_m, pb_m, _ = run_eval(tr, model, batches, device, mode, abl, 0, B)
                info["equalize_calls_during_mutation"] = int(getattr(trunk, "equalize_calls", 0)) - calls0
                object.__setattr__(trunk.cfg, "equalize_bottom_rows", old)
            elif m == "m2":
                br = model._map_hires
                old = str(br.cfg.decision_rule)
                object.__setattr__(br.cfg, "decision_rule", "raw")
                info = {"what": "map decision rule prior_corrected -> raw (the declared rule "
                                "dropped)", "rule_as_built": old}
                erow_m, pb_m, _ = run_eval(tr, model, batches, device, mode, abl, 0, B)
                object.__setattr__(br.cfg, "decision_rule", old)
            elif m == "m4":
                dec = model.core.decoder
                old = dec.residual_prior
                from tanitad.models import kinematic_prior as _kp
                dec.residual_prior = _kp.RESIDUAL_PRIOR_OFF
                info = {"what": "NEW-1 residual prior OFF at eval", "prior_as_built": str(old)}
                try:
                    erow_m, pb_m, _ = run_eval(tr, model, batches, device, mode, abl, 0, B)
                finally:
                    dec.residual_prior = old
            elif m == "m5":
                # SPEC A7.2 item 3: a rare-class INDEX SWAP (bus <-> heavy_truck) in BOTH slot heads at eval, then
                # restored. The columns are derived from the vocabulary, never written by hand (m5_class_indices).
                sw = ClassColumnSwap.for_model(model).apply()
                info = {"what": "bus <-> heavy_truck class-logit columns swapped in BOTH slot heads (SPEC A7.2 M5; "
                                "a vocabulary-order loader defect)", "swap": sw.describe(),
                        "swap_took_effect": bool(sw.took_effect)}
                try:
                    erow_m, pb_m, _ = run_eval(tr, model, batches, device, mode, abl, 0, B)
                finally:
                    info["restoration_bit_exact"] = sw.restore()
                    if not info["restoration_bit_exact"]:
                        print("[g0] M5: THE SLOT HEADS WERE NOT RESTORED BIT-EXACTLY by the swap-back (forced back from "
                              "the snapshot); M5 will read NOT EVALUABLE", flush=True)
            else:
                raise SystemExit(f"unknown mutation {m}")
            info.update(seed=0, row=erow_m, wall_s=round(time.time() - t_m, 1))
        except SystemExit:
            raise
        except Exception as exc:                        # noqa: BLE001 -- a loud failure IS a detection
            info.update(raised=f"{type(exc).__name__}: {str(exc)[:400]}")
        rec["mutations"][m] = info
        print(f"[g0] mutation {m}: {info.get('raised') or 'ran'} ({info.get('wall_s')} s)", flush=True)
        bank()
    # ---- diagnostic arms: REPORTED, NEVER JUDGED (not mutations; no verdict reads them) ------- #
    rec["diagnostic_arms"] = {}
    for dname in [x.strip() for x in a.diagnostic_arms.split(",") if x.strip()]:
        t_d = time.time()
        if dname != "eps0":
            rec["diagnostic_arms"][dname] = {"raised": "unknown diagnostic arm"}
            continue
        _orig = torch.randn_like
        torch.randn_like = lambda x, *aa, **kk: torch.zeros_like(x)
        try:
            erow_d, pb_d, _ = run_eval(tr, model, batches, device, mode, abl, 0, B)
            rec["diagnostic_arms"][dname] = {
                "what": "seed-0 replay with the DDIM eps zeroed (refc.py:2802); REPORTED, never judged",
                "row": erow_d, "per_batch_traj": [r.get("traj") for r in pb_d],
                "wall_s": round(time.time() - t_d, 1)}
        except Exception as exc:                        # noqa: BLE001 -- diagnostic only
            rec["diagnostic_arms"][dname] = {"raised": f"{type(exc).__name__}: {str(exc)[:300]}"}
        finally:
            torch.randn_like = _orig
        print(f"[g0] diagnostic {dname}: eval_traj "
              f"{(rec['diagnostic_arms'][dname].get('row') or {}).get('eval_traj')} "
              f"(in-run {inrun.get('eval_traj')})", flush=True)
        bank()
    # ---- SPEC A6 (draft): the MEASURED numerics floor -- seed 0 under the wrapper clause's P3
    # settings (trunk fp32 + NCHW, cuDNN TF32 off, deterministic). Same weights, flags, windows, code:
    # a numerics-only lever. Its row gives every SMOOTH term's floor phi = |fp32 - s0|; its captured
    # cells give the validity-logit floor of the THRESHOLD-TARGET term. Settings restored after.
    if not a.no_a6:
        t_a = time.time()
        lev0 = dict(trunk.memory_levers)
        bk0 = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32,
               torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
        cap = TacCellCapture(tr)
        try:
            import wrapper_probe_r7 as W
            W._set_condition(W.CONDITIONS["P3_fp32_det"], trunk)
            cap.install()
            erow_a, pb_a, _ = run_eval(tr, model, batches, device, mode, abl, 0, B, packs_out=packs_fp32)
            rec["a6"][A6_ARM] = {"what": "seed 0, trunk fp32 + NCHW, cuDNN TF32 off, deterministic "
                                         "(wrapper P3); SPEC A6's numerics floor",
                                 "row": erow_a, "per_batch": pb_a, "wall_s": round(time.time() - t_a, 1)}
            rec["a6"]["cells"][A6_ARM] = cap.calls
            rec["a6"]["capture_errors"] = cap.errors
        except Exception as exc:                        # noqa: BLE001 -- A6 then NOT EVALUABLE (fail closed)
            rec["a6"][A6_ARM] = {"raised": f"{type(exc).__name__}: {str(exc)[:300]}"}
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            cap.remove()
            trunk.memory_levers.update(lev0)
            (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32,
             torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark) = bk0
        print(f"[g0] A6 arm {A6_ARM}: "
              f"{rec['a6'][A6_ARM].get('raised') or 'ran'} ({time.time() - t_a:.0f} s)", flush=True)
        bank()
    info_bd = buffer_digest(model)
    rec["buffers_unchanged_after_all"] = info_bd == bd0
    mb.remove()
    # ---- wrapper control (SPEC §2, the refcv6 A2 form) --------------------------------- #
    if not a.skip_wrapper_control:
        import wrapper_probe_r7 as W
        rec["wrapper_control"] = W.probe(tr, model, cached[0], device, mode, abl, trunk,
                                         sub_batch_fn=sub_batch)
        print(f"[g0] wrapper clause: {rec['wrapper_control'].get('clause')} "
              f"({rec['wrapper_control'].get('clause_condition')})", flush=True)
    # SPEC AMENDMENT A5 item 3: as registered (seeds 0..7) FIRST, then A2 (seeds 0..7), then A5 (all
    # seeds, >= 24) -- the gate. With fewer than 24 seeds there is no A5 verdict and A2 gates (pre-A5).
    reg = {s: v for s, v in by_seed.items() if int(s) in REGISTERED_SEEDS}
    rec["verdict_as_registered"] = judge(inrun, reg, rec)
    rec["verdict_A2"] = judge(inrun, reg, rec, amend="A2")
    if len(by_seed) >= A5_MIN_SEEDS:
        rec["verdict"] = judge(inrun, by_seed, rec, amend="A5")      # SPEC AMENDMENT A5 gates milestones
    else:
        rec["verdict"] = rec["verdict_A2"]
    # SPEC AMENDMENT A6 (DRAFT until registered): computed and reported whenever its arm ran; it GATES
    # only if `raw/SPEC_SHA256_AMENDMENT_A6.txt` was written BEFORE this G0 started (a6_registration)
    if len(by_seed) >= A5_MIN_SEEDS and not a.no_a6:
        rec["verdict_A5"] = rec["verdict"]
        try:
            rec["verdict_A6"] = judge(inrun, by_seed, rec, amend="A6")
            reg6 = a6_registration(rec["started"])
        except Exception as exc:                        # noqa: BLE001 -- never lose the G0 artifact
            rec["verdict_A6"] = {"G0": "ERROR", "amendment": "A6",
                                 "reasons": [f"A6 judge raised {type(exc).__name__}: {str(exc)[:300]}"]}
            reg6 = {"registered": False, "why": "the A6 judge raised; A5 stays the gate"}
        rec["verdict_A6"]["registration"] = reg6
        if reg6["registered"]:
            rec["verdict"] = rec["verdict_A6"]
    # SPEC AMENDMENT A7 (REGISTERED 2026-10-04T17:11:57+02:00): A6 + the DISCRETE-SMALL-N population guard + the M5
    # probe. Computed on every 24-seed G0 and REPORTED; it is THE GATE only for a G0 that STARTED after
    # `raw/SPEC_SHA256_AMENDMENT_A7.txt` was written (a7_registration). A7 fails CLOSED: a registered A7 whose
    # judge raised, or whose numerics arm is missing, is not a PASS.
    if len(by_seed) >= A5_MIN_SEEDS:
        reg7 = a7_registration(rec["started"])
        try:
            rec["verdict_A7"] = judge(inrun, by_seed, rec, amend="A7")
        except Exception as exc:                        # noqa: BLE001 -- never lose the G0 artifact
            rec["verdict_A7"] = {"G0": "ERROR", "amendment": "A7", "n_terms": 0, "by_class_counts": {},
                                 "medians": {}, "mutation_detection": {},
                                 "reasons": [f"A7 judge raised {type(exc).__name__}: {str(exc)[:300]}"]}
        rec["verdict_A7"]["registration"] = reg7
        if reg7["registered"]:
            rec["verdict"] = rec["verdict_A7"]
    rec["a7"] = a7_reports(rec, by_seed, inrun, tr, seeds, packs_s0, packs_fp32, out_p, a.earlier_g0, a.no_a6)
    rec["wall_s"] = round(time.time() - t_all, 1)
    rec["finished"] = time.strftime("%FT%T")
    json.dump(rec, open(out_p, "w", encoding="utf-8"), indent=1, default=str)
    if not a.keep_batch_cache:
        cached.remove()                      # derived, re-creatable; D: is short of space
    v = rec["verdict"]
    print(json.dumps({"G0_as_registered": rec["verdict_as_registered"]["G0"],
                      "reasons_as_registered": rec["verdict_as_registered"]["reasons"][:10],
                      **{k: v[k] for k in ("G0", "amendment", "reasons", "n_terms", "by_class_counts",
                                           "medians", "mutation_detection")}},
                     indent=1, default=str)[:6000])
    print(f"[g0] wrote {a.out} ({rec['wall_s']} s)")


# --------------------------------------------------------------------------------------- #
# the verdict                                                                               #
# --------------------------------------------------------------------------------------- #
def _isnull(v):
    return v is None or (isinstance(v, float) and v != v)


def tol_ok(cls: str, k: str, x: float, y: float) -> tuple[bool, float, str]:
    """(ok, deviation, tol string) for a DETERMINISTIC comparison of reproduction y vs in-run x."""
    d = abs(y - x)
    rel = d / max(abs(x), 1e-12)
    if cls == "COUNT":
        exact_key = k.startswith("eval_map_hires_") or k.startswith("eval_n_map_hires")
        tol = 1e-6 * max(1.0, abs(x)) if exact_key else 1e-5 + 1e-9
        return d <= tol, d, "exact (5 dp)"
    if cls == "DETECTION":
        if k.endswith(MODEL_DEP_COUNT_SUFFIX):
            return (rel <= 0.05 or d <= 2.0), rel, "rel<=5% or abs<=2"
        if k.endswith("_conf_ratio") or k.endswith("_centre_err_p50"):
            return rel <= 0.05, rel, "rel<=5%"
        return d <= 0.02, d, "abs<=0.02"
    if cls == "MATCHED":
        return rel <= 0.15, rel, "rel<=15%"
    if cls == "MAP10_COUNTS":
        return d <= max(0.02 * abs(x), 25.0), rel, "abs<=max(2%|x|, 25 cells)"
    if cls == "MAP10_IOU":
        return d <= 0.01, d, "abs<=0.01"
    if cls == "SMOOTH":
        ok = (d <= 1e-3) if abs(x) < 0.1 else (rel <= 0.01)
        return ok, rel, "abs<=1e-3 (|x|<0.1) else rel<=1%"
    raise ValueError(cls)


#: SPEC AMENDMENT A2 (registered 2026-09-28 before any milestone number): a DETECTION key computed on
#: fewer than A2_MIN_SUPPORT ground-truth events is REPORTED with its n and never gates (one event there
#: moves the metric by more than any fixed tolerance: the step-1,500 G0's only OUT term was an AP on
#: n_pos = 1); at or above it the tolerance is max(0.02, 2/n) -- "two discrete events".
A2_MIN_SUPPORT = 30
#: every amendment whose registered text keeps A2's low-support rule (A5 item 2, A6 item 7) -- pinned by
#: test_g0_a6_lowsupport.py so a new amendment cannot silently drop it again
A2_LOWSUPPORT_AMENDS = ("A2", "A5", "A6", "A7")
_BANDS = r"(all|0_20|20_40|40_60)"


def detection_support(k: str, inrun: dict):
    """The number of GT events a per-class / per-band DETECTION key is computed on, read from the
    in-run row's own count keys; None for a POOLED key (it keeps the registered tolerance)."""
    import re
    s = k[len("eval_"):] if k.startswith("eval_") else k
    m = re.match(r"(agent|box3d)_det_ap[0-9p]+_(.+)_" + _BANDS + r"$", s)
    if m and m.group(2) != "all":
        return float(inrun.get(f"eval_{m.group(1)}_det_npos_{m.group(2)}_{m.group(3)}") or 0.0)
    m = re.match(r"(agent|box3d)_det_map[0-9p]+_" + _BANDS + r"$", s)
    if m:
        pre = f"eval_{m.group(1)}_det_npos_"
        suf = f"_{m.group(2)}"
        ns = [float(v) for kk, v in inrun.items() if kk.startswith(pre) and kk.endswith(suf)
              and not kk.startswith(pre + "all_") and v]
        return min(ns) if ns else 0.0
    m = re.match(r"(agent|box3d)_rec@gate_(.+)$", s)
    if m:
        return float(inrun.get(f"eval_{m.group(1)}_npos_{m.group(2)}") or 0.0)
    return None


def a6_context(rec, by_seed) -> dict:
    """SPEC A6's measured inputs: phi_k = |fp32_s0 - s0| per term, and the THRESHOLD-TARGET interval
    from the per-cell captures, with its two CONTROLS (the cells must reproduce the trainer's own batch
    values; the interval's centre must equal the replay's logged seed-0 value). Any gap -> `error`
    (A6 then fails CLOSED: a term it cannot evaluate is OUT, never OK)."""
    a6 = rec.get("a6") or {}
    s0 = (by_seed.get(0) or by_seed.get("0") or {}).get("row")
    arm = (a6.get(A6_ARM) or {}).get("row")
    if not s0 or not arm:
        return {"error": f"A6 needs seed 0 and the {A6_ARM} row (have s0={bool(s0)}, "
                         f"arm={bool(arm)}; arm record: {str(a6.get(A6_ARM))[:200]})"}
    phi = {}
    for k, v in arm.items():
        w = s0.get(k)
        if isinstance(v, (int, float)) and isinstance(w, (int, float)) and not _isnull(v) \
                and not _isnull(w):
            phi[k] = abs(float(v) - float(w))
    out = {"phi": phi}
    cells = a6.get("cells") or {}
    if cells.get("s0") and cells.get(A6_ARM):
        try:
            iv = threshold_interval(cells["s0"], cells[A6_ARM])
            ctl = [abs(conf_bce_from_cells(cl) - float(cl["tac_goal_conf_bce"])) for cl in cells["s0"]]
            iv["control_cells_vs_trainer_max_abs"] = max(ctl)
            logged = s0.get("eval_tacv6_goal_conf_bce")
            iv["control_centre_vs_logged_abs"] = (None if logged is None
                                                  else abs(iv["t_replay"] - float(logged)))
            iv["controls_ok"] = (max(ctl) <= 1e-5 and logged is not None
                                 and abs(iv["t_replay"] - float(logged)) <= 1e-5)
            out["interval"] = iv
        except Exception as exc:                        # noqa: BLE001 -- fail closed
            out["interval_error"] = f"{type(exc).__name__}: {exc}"
    else:
        out["interval_error"] = "per-cell captures missing"
    return out


def _a7_moved(a, b) -> bool:
    """SPEC A7.2 item 2: 'moved' = |a - b| > 0.02. The difference is rounded to 9 dp first, so floating-point noise
    in a subtraction of two 5-dp row values cannot decide a boundary (0.5 - 0.48 is 0.020000000000000018 in float
    arithmetic and exactly 0.02 as written; the registered rule is the written one)."""
    return round(abs(float(a) - float(b)), 9) > A7_MOVE_TOL


def a7_lowsupport_guard(res: dict, by_seed: dict, rec: dict) -> dict:
    """SPEC A7.2 items 1-2: the DISCRETE-SMALL-N population guard.

    Members = exactly A2's low-support DETECTION keys (support n < 30, n read from the in-run row's own
    `*_det_npos_*` / `*_npos_*` keys): the terms `judge` classed DETECTION_LOWSUPPORT. Each member is REPORTED with
    its n, in-run value, replay mean, seed-0 value and `fp32_s0` value, and is never gated on its own. The
    population gates:
        N_in  = members with |in-run - replay mean| > 0.02
        N_num = members with |fp32_s0 - seed 0|     > 0.02      (fp32_s0 = the A6 numerics arm)
        the class FAILS iff  N_in > 2 * N_num + 5.
    The guard is FAIL-CLOSED: a missing `fp32_s0` row (or one lacking a member key, or a missing seed-0 row) makes
    the class NOT EVALUABLE, and a NOT EVALUABLE class FAILS. (An `fp32_s0` value that is null where seed 0's is
    not counts as a move: a definedness flip is a numerics effect.)"""
    members = sorted(k for k, r in res.items() if r.get("cls") == "DETECTION_LOWSUPPORT")
    s0 = (by_seed.get(0) or by_seed.get("0") or {}).get("row")
    arm = ((rec.get("a6") or {}).get(A6_ARM) or {}).get("row")
    recs, n_in = [], 0
    for k in members:
        r = res[k]
        mv = _a7_moved(r["inrun"], r["mean"])
        n_in += bool(mv)
        v0 = None if (not s0 or _isnull(s0.get(k))) else float(s0[k])
        recs.append({"key": k, "n": r.get("support_n"), "inrun": float(r["inrun"]), "replay_mean": float(r["mean"]),
                     "seed0": v0, "fp32_s0": None, "dev_inrun": abs(float(r["inrun"]) - float(r["mean"])),
                     "dev_numerics": None, "moved_in": bool(mv), "moved_numerics": None})
    why = None
    if not arm:
        why = "the fp32_s0 row is missing (the A6 numerics arm did not run, or did not complete)"
    elif s0 is None:
        why = "the seed-0 row is missing"
    else:
        absent = [m["key"] for m in recs if m["key"] not in arm]
        no_s0 = [m["key"] for m in recs if m["seed0"] is None]
        if absent:
            why = f"the fp32_s0 row lacks {len(absent)} of the {len(recs)} members (e.g. {absent[:3]})"
        elif no_s0:
            why = f"the seed-0 row lacks {len(no_s0)} of the {len(recs)} members (e.g. {no_s0[:3]})"
    out = {"rule": f"FAIL iff N_in > {A7_GUARD_FACTOR} * N_num + {A7_GUARD_SLACK}; moved = |delta| > {A7_MOVE_TOL}",
           "tol": A7_MOVE_TOL, "factor": A7_GUARD_FACTOR, "slack": A7_GUARD_SLACK,
           "n_members": len(recs), "N_in": n_in, "N_num": None, "bound": None, "why": why, "members": recs}
    if why is not None:
        out["status"] = "NOT EVALUABLE"
        return out
    n_num = 0
    for m in recs:
        f = arm[m["key"]]
        fp = None if _isnull(f) else float(f)
        mv = True if fp is None else _a7_moved(fp, m["seed0"])
        m["fp32_s0"] = fp
        m["dev_numerics"] = None if fp is None else abs(fp - m["seed0"])
        m["moved_numerics"] = bool(mv)
        n_num += bool(mv)
    out["N_num"] = n_num
    out["bound"] = A7_GUARD_FACTOR * n_num + A7_GUARD_SLACK
    out["status"] = "FAIL" if n_in > out["bound"] else "PASS"
    return out


def a7_m5_probe(info, guard: dict, res: dict, moved) -> dict:
    """SPEC A7.2 item 3: the M5 power probe, judged. REPORTED, NEVER GATING, NEVER VOID.

    `info` = rec['mutations']['m5']: the seed-0 row with the bus / heavy_truck class-logit columns swapped in both
    slot heads (and whether the restoration was bit-exact). DETECTED iff
        N_M5 > 2 * N_num + 5            (N_M5 = low-support members with |M5 row - seed 0| > 0.02; the same
                                         members and the same N_num as the guard)
        OR any gating DETECTION term (n >= 30, or pooled) moves outside its tolerance.
    UNDETECTED => the blind spot is NAMED: 'a rare-class index swap is invisible to G0'. If M5 did not run, raised,
    was not restored bit-exactly, or the numerics arm needed for N_num is missing, it is NOT RUN / NOT EVALUABLE
    (a gap that is reported -- never read as detected)."""
    base = {"never_gates": True, "never_voids": True,
            "rule": f"DETECTED iff N_M5 > {A7_GUARD_FACTOR} * N_num + {A7_GUARD_SLACK} OR any gating DETECTION term "
                    f"moves outside its tolerance",
            "N_M5": None, "N_num": guard.get("N_num"), "bound": guard.get("bound"), "detected": None,
            "blind_spot": None}
    if not info:
        return {**base, "status": "NOT RUN", "why": "no m5 mutation record (the arm was not run)"}
    base.update(swap=info.get("swap"), swap_took_effect=info.get("swap_took_effect"),
                restoration_bit_exact=info.get("restoration_bit_exact"))
    if info.get("raised"):
        return {**base, "status": "NOT EVALUABLE", "why": f"M5 raised: {str(info['raised'])[:300]}"}
    if info.get("restoration_bit_exact") is not True:
        return {**base, "status": "NOT EVALUABLE",
                "why": "the restoration of the slot heads was not shown bit-exact"}
    row = info.get("row") or {}
    members = guard.get("members") or []
    if any(m["seed0"] is None for m in members):
        return {**base, "status": "NOT EVALUABLE", "why": "a member has no seed-0 value to compare the M5 row with"}
    n_m5, moved_keys, absent = 0, [], 0
    for m in members:
        y = row.get(m["key"], "__MISSING__")
        if y == "__MISSING__":
            absent += 1                       # not produced: cannot be claimed as a move
            continue
        mv = True if _isnull(y) else _a7_moved(y, m["seed0"])
        if mv:
            n_m5 += 1
            moved_keys.append(m["key"])
    gating_out, other_out = [], 0
    for k, r in res.items():
        v = moved(k, r, row.get(k))
        if not v:
            continue
        if r.get("cls") == "DETECTION":
            gating_out.append({"term": k, "inrun": r["inrun"], "m5": row.get(k)})
        else:
            other_out += 1
    n_num, bound = guard.get("N_num"), guard.get("bound")
    clause_pop = None if bound is None else n_m5 > bound
    clause_gate = bool(gating_out)
    out = {**base, "N_M5": n_m5, "n_members": len(members), "n_members_absent_from_m5_row": absent,
           "members_moved_first40": moved_keys[:40], "clause_population": clause_pop,
           "clause_gating_detection": clause_gate, "gating_detection_out": gating_out[:20],
           "n_gating_detection_out": len(gating_out), "n_other_terms_out_informational": other_out}
    if clause_pop or clause_gate:
        out.update(status="DETECTED", detected=True)
    elif clause_pop is None:
        out.update(status="NOT EVALUABLE", detected=None,
                   why="the population clause needs N_num (the fp32_s0 numerics arm), which is missing")
    else:
        out.update(status="UNDETECTED", detected=False, blind_spot=M5_BLIND_SPOT)
    return out


def judge(inrun, by_seed, rec, amend: str | None = None):
    seeds = sorted(by_seed)
    K = len(seeds)
    terms = sorted(k for k in inrun if k.startswith("eval_"))
    res, reasons, cls_count = {}, [], {}
    a6 = a6_context(rec, by_seed) if amend in A6_AMENDS else None      # A7 inherits A6 items 1-6, 8 UNCHANGED
    if a6 is not None and a6.get("error"):
        reasons.append(f"A6 NOT EVALUABLE: {a6['error']}")
        a6 = {"phi": {}, "interval_error": a6["error"]}
    for k in terms:
        c = term_class(k)
        x = inrun[k]
        vals = [by_seed[s]["row"].get(k, "__MISSING__") for s in seeds]
        r = {"inrun": x, "class": c}
        if c in ("EXCLUDED", "DERIVED"):
            r["verdict"] = c
            res[k] = r
            cls_count[c] = cls_count.get(c, 0) + 1
            continue
        if any(v == "__MISSING__" for v in vals):
            r["verdict"] = "MISSING"
            reasons.append(f"{k}: not produced by the reproduction")
            res[k] = r
            continue
        if _isnull(x) or any(_isnull(v) for v in vals):
            ok = _isnull(x) and all(_isnull(v) for v in vals)
            r.update(seed_values=[None if _isnull(v) else v for v in vals], cls=c + "/UNDEFINED",
                     verdict="OK" if ok else "OUT")
            if not ok:
                reasons.append(f"{k} [UNDEFINED rule] in-run {x} vs seeds {r['seed_values'][:3]}")
            res[k] = r
            cls_count["UNDEFINED"] = cls_count.get("UNDEFINED", 0) + 1
            continue
        vals = [float(v) for v in vals]
        mean = statistics.fmean(vals)
        sd = statistics.stdev(vals) if K >= 2 else 0.0
        spread = (max(vals) - min(vals)) / max(abs(mean), 1e-12)
        if c == "SMOOTH_OR_STOCHASTIC":
            c = "STOCHASTIC" if spread > 1e-5 else "SMOOTH"
        if a6 is not None and k in THRESHOLD_TARGET and spread <= 1e-5:
            # SPEC A6 item 1 (by name, from source). A seed-VARYING member stays STOCHASTIC (A5's PI).
            c = "THRESHOLD_TARGET"
        r.update(mean=mean, sd=sd, rel_spread=spread, cls=c, seed_values=vals)
        x = float(x)
        if c == "THRESHOLD_TARGET":
            # SPEC A6 item 2: the in-run value must be REACHABLE from the replay by flipping only
            # undecidable validity decisions, widened by the registered SMOOTH tolerance
            iv = a6.get("interval")
            tolc = 0.01 * abs(x) if abs(x) >= 0.1 else 1e-3
            if not iv or not iv.get("controls_ok"):
                ok = False
                r.update(tol="A6 interval NOT EVALUABLE (fail closed)",
                         a6_interval_error=a6.get("interval_error"),
                         a6_controls={kk: (iv or {}).get(kk) for kk in (
                             "control_cells_vs_trainer_max_abs", "control_centre_vs_logged_abs",
                             "controls_ok")})
            else:
                lo, hi = iv["lo"] - tolc, iv["hi"] + tolc
                ok = lo <= x <= hi
                r.update(a6_lo=lo, a6_hi=hi, a6_t_replay=iv["t_replay"],
                         a6_delta_logit_max=iv["delta_logit_max"], a6_n_undecidable=iv["n_undecidable"],
                         a6_n_flipped_by_arm=iv["n_flipped_by_arm"], dev=abs(mean - x),
                         tol=f"A6 flip interval [{lo:.5f}, {hi:.5f}] (k={A6_K}, +SMOOTH tol)")
            r["verdict"] = "OK" if ok else "OUT"
            if not ok:
                reasons.append(f"{k} [THRESHOLD_TARGET] in-run {x} outside the A6 interval "
                               f"({r.get('tol')})")
            cls_count[c] = cls_count.get(c, 0) + 1
            res[k] = r
            continue
        if c == "STOCHASTIC":
            if K not in T_995:
                raise ValueError(f"no t(0.995, {K - 1}) registered; add it to T_995 (never default)")
            t = T_995[K]
            half = t * sd * math.sqrt(1 + 1 / K) + 0.01 * abs(mean)
            ok = (mean - half) <= x <= (mean + half)
            r.update(pi_lo=mean - half, pi_hi=mean + half, tol=f"99% PI t={t} +1% |mean|")
        else:
            if spread > 1e-5 and c in ("COUNT", "DETECTION", "MAP10_COUNTS", "MAP10_IOU", "MATCHED"):
                r["note"] = "seed spread > 1e-5 in a deterministic class: judged on the seed mean"
            ok, dev, tol = tol_ok(c, k, x, mean)
            # A6 item 7 (registered 2026-10-04T08:17:13+02:00) keeps "A2's low-support rule" UNCHANGED; this
            # tuple omitted "A6" until 2026-10-04 ~17:30 (G0 50,400 diagnosis, raw/g0diag_step50400/): the judge
            # gated every low-support DETECTION cell at abs 0.02 under A6, against the registered text.
            sup = detection_support(k, inrun) if (amend in A2_LOWSUPPORT_AMENDS and c == "DETECTION") else None
            if sup is not None:
                r["support_n"] = sup
                if sup < A2_MIN_SUPPORT:
                    r.update(dev=abs(mean - x), tol=f"REPORTED (support n={sup:g} < {A2_MIN_SUPPORT})",
                             cls="DETECTION_LOWSUPPORT", verdict="REPORTED")
                    cls_count["DETECTION_LOWSUPPORT"] = cls_count.get("DETECTION_LOWSUPPORT", 0) + 1
                    res[k] = r
                    continue
                t2 = max(0.02, 2.0 / sup)
                ok, dev, tol = abs(mean - x) <= t2, abs(mean - x), f"abs<=max(0.02, 2/n)={t2:.4f}"
            r.update(dev=dev, tol=tol)
            if a6 is not None and c == "SMOOTH":
                # SPEC A6 item 3: a SMOOTH term's tolerance is max(registered, k * phi), phi MEASURED
                phi = a6["phi"].get(k)
                r["a6_phi"] = phi
                if not ok and phi is not None and abs(mean - x) <= A6_K * phi:
                    ok = True
                    r["a6_rescued"] = True
                    r["tol"] = f"{tol} OR A6 |dev|<={A6_K:g}*phi={A6_K * phi:.3g}"
            if c in ("SMOOTH", "MATCHED", "MAP10_COUNTS") and abs(x) > 0:
                r["rel_dev"] = abs(mean - x) / abs(x)
            if c == "DETECTION" and not k.endswith(MODEL_DEP_COUNT_SUFFIX) \
                    and not k.endswith(("_conf_ratio", "_centre_err_p50")):
                r["abs_dev01"] = abs(mean - x)
        r["verdict"] = "OK" if ok else "OUT"
        if not ok:
            reasons.append(f"{k} [{c}] in-run {x} vs seeds mean {mean:.6g} (sd {sd:.3g})")
        cls_count[c] = cls_count.get(c, 0) + 1
        res[k] = r
    med = {}
    sm = [v["rel_dev"] for v in res.values() if v.get("cls") == "SMOOTH" and "rel_dev" in v
          and abs(float(v["inrun"])) >= 0.1]
    mt = [v["rel_dev"] for v in res.values() if v.get("cls") == "MATCHED" and "rel_dev" in v]
    mc = [v["rel_dev"] for v in res.values() if v.get("cls") == "MAP10_COUNTS" and "rel_dev" in v
          and abs(float(v["inrun"])) >= 1000]
    dt = [v["abs_dev01"] for v in res.values() if v.get("cls") == "DETECTION" and "abs_dev01" in v]
    med["SMOOTH"] = statistics.median(sm) if sm else None
    med["MATCHED"] = statistics.median(mt) if mt else None
    med["MAP10_COUNTS"] = statistics.median(mc) if mc else None
    med["DETECTION_abs01"] = statistics.median(dt) if dt else None
    smooth_med_bar = 0.002
    if a6 is not None:
        # SPEC A6 item 4: the SMOOTH class-median bar is max(0.2 %, k * the median of phi/|x|) over the
        # SAME keys (|x| >= 0.1); THRESHOLD_TARGET keys are not SMOOTH under A6
        ph = [v["a6_phi"] / abs(float(v["inrun"])) for v in res.values()
              if v.get("cls") == "SMOOTH" and v.get("a6_phi") is not None
              and abs(float(v["inrun"])) >= 0.1]
        med["SMOOTH_phi_rel"] = statistics.median(ph) if ph else None
        if ph:
            smooth_med_bar = max(0.002, A6_K * statistics.median(ph))
        med["SMOOTH_bar"] = smooth_med_bar
    if med["SMOOTH"] is not None and med["SMOOTH"] > smooth_med_bar:
        reasons.append(f"SMOOTH median rel dev {med['SMOOTH']:.4f} > {smooth_med_bar:.4f}")
    if med["MATCHED"] is not None and med["MATCHED"] > 0.05:
        reasons.append(f"MATCHED median rel dev {med['MATCHED']:.4f} > 0.05")
    if med["MAP10_COUNTS"] is not None and med["MAP10_COUNTS"] > 0.002:
        reasons.append(f"MAP10 median rel dev {med['MAP10_COUNTS']:.4f} > 0.002")
    if med["DETECTION_abs01"] is not None and med["DETECTION_abs01"] > 0.005:
        reasons.append(f"DETECTION median abs dev {med['DETECTION_abs01']:.4f} > 0.005")
    md = rec["model"]
    if md["state_dict"]["missing"] or md["state_dict"]["unexpected"]:
        reasons.append("strict load not clean")
    if not md["param_breakdown"]["equal"]:
        reasons.append("param_breakdown differs from config.json")
    anc = md.get("anchor_file_vs_ckpt_buffers") or {}
    anc_bad = [k for k, v in anc.items() if isinstance(v, dict) and v.get("max_abs_diff") != 0.0]
    if anc_bad:
        reasons.append(f"anchor file != ckpt anchor buffers: {anc_bad}")
    if (md.get("declared_vs_built") or {}).get("mismatches"):
        reasons.append("loader G-DVB mismatches")
    if not all(v.get("buffers_unchanged") for v in by_seed.values()):
        reasons.append("model buffers changed during an eval pass")
    wc = rec.get("wrapper_control")
    if wc is None:
        reasons.append("wrapper control not run")
    elif wc.get("clause") != "PASS":
        reasons.append(f"wrapper clause {wc.get('clause')}: {wc.get('reasons')}")
    # SPEC A7.2 -- the DISCRETE-SMALL-N population guard (gates ONLY under A7). It is a function looked up
    # through the module namespace on purpose: the test suite's deliberate-regression arm replaces it.
    a7g = None
    if amend == "A7":
        a7g = a7_lowsupport_guard(res, by_seed, rec)
        if a7g["status"] == "FAIL":
            reasons.append(f"A7.2 DISCRETE-SMALL-N guard: N_in={a7g['N_in']} > 2*N_num+{A7_GUARD_SLACK}="
                           f"{a7g['bound']} (N_num={a7g['N_num']}, {a7g['n_members']} low-support members)")
        elif a7g["status"] != "PASS":
            reasons.append(f"A7.2 DISCRETE-SMALL-N guard NOT EVALUABLE (fails closed): {a7g['why']}")
    # mutation detection: >= 1 non-COUNT term outside its class tolerance vs the IN-RUN value
    def _mutation_bad(k, r, y):
        """None = the term is not eligible for mutation detection; else True iff the MUTATED value `y` lies outside
        the term's tolerance. This is the per-term rule the loop below has always applied (SPEC A2 item 2, A6
        item 5); it is a function so that SPEC A7's M5 clause "any gating DETECTION term moves outside its
        tolerance" reads the SAME rule rather than a re-implementation of it."""
        c = r.get("cls", "")
        # ⭐ a term already OUT under the unmutated reproduction cannot be MOVED outside by a
        # mutation (SPEC §2's "move ... outside"); counting it inflated M1 (corrected 2026-09-28,
        # recorded in SPEC AMENDMENT A2)
        if c in ("COUNT",) or r.get("verdict") != "OK" or c.endswith("/UNDEFINED"):
            return None
        x = r["inrun"]
        if _isnull(y) or _isnull(x):
            return None
        if c == "STOCHASTIC":
            return not (r["pi_lo"] <= float(y) <= r["pi_hi"])
        if c == "THRESHOLD_TARGET":                 # SPEC A6: outside the flip interval
            return not (r["a6_lo"] <= float(y) <= r["a6_hi"])
        if r.get("support_n") is not None:
            return abs(float(y) - float(x)) > max(0.02, 2.0 / float(r["support_n"]))
        bad = not tol_ok(c, k, float(x), float(y))[0]
        if bad and a6 is not None and c == "SMOOTH" and r.get("a6_phi") is not None:
            # SPEC A6 item 5: the SAME per-term tolerance, max(registered, k * phi)
            bad = abs(float(y) - float(x)) > A6_K * float(r["a6_phi"])
        return bad

    det = {}
    for m, info in (rec.get("mutations") or {}).items():
        if m == "m5":
            continue                      # SPEC A7.2 power probe: judged by its own rule, only under A7 (below)
        if info.get("raised"):
            det[m] = {"detected": True, "how": "raised", "n_terms_out": None}
            continue
        row = info.get("row") or {}
        outs = []
        for k, r in res.items():
            if _mutation_bad(k, r, row.get(k)):
                outs.append({"term": k, "class": r.get("cls", ""), "inrun": r["inrun"], "mutated": row.get(k)})
        det[m] = {"detected": bool(outs), "n_terms_out": len(outs), "first10": outs[:10]}
    m1 = det.get("m1")
    if m1 is None:
        reasons.append("M1 not run")
    blind = [m for m, d in det.items() if not d["detected"]]
    only_m1 = bool(m1) and not m1["detected"]
    if only_m1:
        reasons.append("M1 (trunk equalisation dropped) stayed inside every tolerance: the gate "
                       "has no power -> VOID")
    m5 = None
    if amend == "A7":
        # SPEC A7.2 item 3: REPORTED, never a reason, never VOID. It is added to `det` AFTER `blind` / `only_m1`
        # are computed, so no M1 / VOID logic can read it.
        m5 = a7_m5_probe((rec.get("mutations") or {}).get("m5"), a7g, res, _mutation_bad)
        det["m5"] = {"detected": m5["detected"], "n_terms_out": m5.get("N_M5"), "how": m5["status"]}
    g0 = "PASS" if not reasons else "FAIL"
    if only_m1 and all(r.startswith("M1") for r in reasons):
        g0 = "VOID"
    out = {"G0": g0, "amendment": amend or "as registered", "reasons": reasons,
           "n_terms": len(res), "by_class_counts": cls_count,
           "medians": med, "mutation_detection": det,
           "blind_spots_named": [f"{m}: not detected by G0 (a wiring defect this gate cannot see)"
                                 for m in blind if m != "m1"],
           "terms": res}
    if amend == "A7":
        out["a7_lowsupport"] = a7g
        out["m5"] = m5
        if m5["status"] == "UNDETECTED":
            out["blind_spots_named"].append(f"m5: {M5_BLIND_SPOT}")
    if a6 is not None:
        # SPEC A6 item 6: every term the floor rescued is NAMED with its numbers -- never silent
        out["a6_rescued"] = [{"term": k, "inrun": r["inrun"], "replay": r.get("mean"),
                              "phi": r.get("a6_phi"), "rel_dev": r.get("rel_dev")}
                             for k, r in res.items() if r.get("a6_rescued")]
        out["a6_threshold_terms"] = {k: {kk: r.get(kk) for kk in (
            "inrun", "mean", "a6_lo", "a6_hi", "a6_t_replay", "a6_delta_logit_max", "a6_n_undecidable",
            "a6_n_flipped_by_arm", "verdict")} for k, r in res.items()
            if r.get("cls") == "THRESHOLD_TARGET"}
        out["a6_interval"] = {kk: vv for kk, vv in (a6.get("interval") or {}).items()
                              if kk != "undecidable"}
        out["a6_interval_undecidable"] = (a6.get("interval") or {}).get("undecidable")
    return out


if __name__ == "__main__":
    main()
