#!/usr/bin/env python3
"""Derive the leaderboard INPUT record for refcv3 @40,284 with the RETR-2026-09-26-YAWMASK correction applied.

⛔ WHY (Master Mind decision, 2026-09-26, batch 15). The refcv3 LATERAL yaw-rate cells of the banked ARM record were
computed over a shared cell that scored standstill jitter (tangent-less steps). The yaw-rate-mask package re-derived
exactly those paired cells (``…/2026-09-26-yaw-rate-mask/raw/claims/refcv3_40284_arm.result.json``, ``yaw_fixed``; list
layout ``FIELDS`` at ``…/code/rederive_claims.py:53``). The correction had reached the published page only as a HAND
EDIT inside the generated section, and a rebuild from inputs would have undone it. This tool writes a NEW record: the
banked ARM with ONLY those cells replaced, so the correction lives in the inputs. The banked record is history and is
never modified.

Guards — each REFUSES rather than guesses:
* the banked ARM's git blob must equal the blob the mask package re-derived from (its ``banked_blob``);
* every replaced cell's CURRENT values must equal the mask package's ``yaw_banked`` view, field for field: the fix is
  applied to the record it was computed from, or not at all;
* fields the mask package did not re-derive (``ci95``, ``p_delta_gt0``) are NULLED, never left stale beside a new delta
  (the banked ``p_delta_gt0`` 1.0 would sit beside a masked delta of −0.0100);
* per-arm (unpaired) yaw-rate values elsewhere in the record were NOT re-derived: they are LISTED as stale in the
  provenance (the leaderboard does not read them), never silently carried as if current.

    python taniteval/tools/derive_yawmask_record.py --check     # re-derive in memory; exit 1 if the banked output differs
    python taniteval/tools/derive_yawmask_record.py --write     # (re)write the derived record
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARM = "taniteval/results/refcv3-40284-openloop.ARM.json"
MASK = ("TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-26-yaw-rate-mask/raw/claims/"
        "refcv3_40284_arm.result.json")
OUT = "taniteval/results/refcv3-40284-openloop.ARM.yawmask.json"
RETR = "RETR-2026-09-26-YAWMASK"
#: the mask package's list layout, verbatim from ``rederive_claims.py:53`` — a literal, never re-derived here
FIELDS = ("delta", "lo", "hi", "separated", "n_windows", "n_episodes", "n_dropped_nonfinite")
NULLED = ("ci95", "p_delta_gt0")
METRIC = "LAT_yaw_rate_mae_radps"


class DeriveError(RuntimeError):
    pass


def lf_bytes(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


def git_blob_id(path: Path) -> str:
    """The id git stores for this file (LF-normalised, as ``git hash-object`` with filters reports it)."""
    b = lf_bytes(path)
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def pointer_of(key: str) -> list:
    """``"paired_os_minus_ha0 | lateral | LAT_yaw_rate_mae_radps"`` -> the cell's path inside the ARM record."""
    pair, fam, metric = [x.strip() for x in key.split("|")]
    if fam != "lateral" or metric != METRIC:
        raise DeriveError(f"unexpected mask key {key!r}: only the lateral {METRIC} cells are re-derived")
    return ["refcv3", "families_paired", pair, "families", "lateral", metric]


def _get(doc, path):
    for k in path:
        doc = doc[k]
    return doc


def stale_yaw_pointers(doc) -> list:
    """Every OTHER yaw-RATE value in the record (per-arm, unpaired) — present, not re-derived by the mask package."""
    out = []

    def walk(o, p):
        if isinstance(o, dict):
            for k, v in o.items():
                q = p + [k]
                if "yaw_rate" in k.lower() and not (len(q) >= 2 and q[0] == "refcv3" and q[1] == "families_paired"):
                    out.append("/" + "/".join(q))
                walk(v, q)
    walk(doc, [])
    return sorted(out)


def derive(repo: Path = REPO) -> dict:
    arm_p, mask_p = repo / ARM, repo / MASK
    arm = json.loads(arm_p.read_text(encoding="utf-8"))
    mask = json.loads(mask_p.read_text(encoding="utf-8"))
    blob = git_blob_id(arm_p)
    if mask.get("banked_blob") != blob:
        raise DeriveError(f"the banked ARM is blob {blob}, the mask package re-derived blob {mask.get('banked_blob')} — "
                          "refusing to apply a fix computed on a different record")
    if mask.get("banked_record") != ARM:
        raise DeriveError(f"the mask result names {mask.get('banked_record')!r}, not {ARM!r}")
    fixed, banked = mask["yaw_fixed"], mask["yaw_banked"]
    if set(fixed) != set(banked) or len(fixed) != mask.get("n_yaw_cells"):
        raise DeriveError("yaw_fixed / yaw_banked / n_yaw_cells disagree about which cells were re-derived")
    out = copy.deepcopy(arm)
    changed = []
    for key in sorted(fixed):
        path = pointer_of(key)
        cell = _get(out, path)
        now = [cell.get(f) for f in FIELDS]
        if now != banked[key]:
            raise DeriveError(f"{key}: the record holds {now}, the mask package's banked view is {banked[key]} — "
                              "the fix would be applied to a cell it was not computed from")
        new = dict(zip(FIELDS, fixed[key]))
        cell["_unmasked"] = {f: cell.get(f) for f in FIELDS}
        cell.update(new)
        for f in NULLED:
            cell[f] = None
        cell["_masked_by"] = RETR
        changed.append("/" + "/".join(path))
    out["_derived"] = {
        "schema": "leaderboard-derived-input/1",
        "what": f"refcv3 @40,284 ARM record with {RETR} applied to its {len(changed)} paired yaw-rate cells",
        "retraction": RETR,
        "from": {"path": ARM, "git_blob": blob, "sha256_lf": hashlib.sha256(lf_bytes(arm_p)).hexdigest()},
        "mask_result": {"path": MASK, "git_blob": git_blob_id(mask_p), "sha256_lf": hashlib.sha256(lf_bytes(mask_p)).hexdigest(),
                        "list_layout": list(FIELDS), "list_layout_source": "…/2026-09-26-yaw-rate-mask/code/rederive_claims.py:53"},
        "changed": changed,
        "fields_replaced": list(FIELDS),
        "fields_nulled_not_rederived": list(NULLED),
        "unmasked_values_kept_in": "each changed cell's `_unmasked`",
        "stale_not_rederived": stale_yaw_pointers(arm),
        "stale_note": ("per-arm (unpaired) yaw-rate values were not re-derived by the mask package; they are stale under "
                       "the mask and the leaderboard does not read them. `net_yaw_err_rad` is a different metric."),
        "tool": "taniteval/tools/derive_yawmask_record.py",
        "decided_by": "Master Mind, 2026-09-26 (batch 15): derive a new input; never modify the banked record",
    }
    return out


def dump(doc: dict) -> str:
    return json.dumps(doc, indent=1, ensure_ascii=False) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    doc = derive()
    out = REPO / OUT
    if a.write:
        out.write_text(dump(doc), encoding="utf-8", newline="\n")
        print(json.dumps({"written": OUT, "changed": doc["_derived"]["changed"]}, indent=1))
        return 0
    if not out.exists():
        print(f"MISSING {OUT}")
        return 1
    same = json.loads(out.read_text(encoding="utf-8")) == doc
    print("MATCH" if same else f"DIFFERS: {OUT} is not the derivation of its inputs")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
