"""feed_parity_pf.py -- SHIP-R8 AMENDMENT 3 clause (1b) FEED PARITY, asserted on the ROLL ARTIFACTS, derived INDEPENDENTLY of the overlay.

NEW file (EvalFlyWheel 2026-10-20, hold_r8pf).  numpy only (no torch, no stack import): it shares NO code with ``navfeed_overlay`` /
``refcv8_train.scale_nav`` / ``v9_labels.row_for_now``, so it cannot be green because of the defect it checks for.

For every window of a per-frame (``perframe_full``) dump -- every scored window of every surface is a dump window -- it asserts:
  T  TOKEN    decisions.nav_cmd == the v9 per-frame token, looked up DIRECTLY in the v9 release npz at (sid, k = ws + RAW_OFFSET) with the positive key
              assertion ``row__k == k``; RAW_OFFSET is read from the dump manifest's own ``grid.ws_is`` text ("RAW frame = ws + 2"), never assumed.
              The v9 -> nav_cmd map is the LITERAL {0: 0, 1: 1, 2: 2} (refc_v3_train.py:4686, pinned by tests/test_refcv8_v9_wiring.py:396).
  C  CLIP     decisions.nav_cmd_clip == the per-clip token of an INDEPENDENT clip-fed dump of the same windows (decisions.nav_cmd there), keyed by
              (clip sha12, ws); the reference must be constant within every clip (it is the clip-level token) and hold EXACTLY the same window set.
  K  KNOWN    decisions.r8_nav_known == the release's ``row__nav_args_valid`` at the same row.
  A  ARGS     decisions.r8_nav_fed is non-zero EXACTLY where r8_nav_known, and equals an independent numpy re-implementation of the training scaling
              (log1p(min(d, 1000) / 50) for d_next / d_end / lookahead, dyaw / 90, side clamped to [-1, 1], times known, then the known bit) of the
              release's raw columns, to <= 1e-5.
  I  IDENTITY sha256(manifest clip_id)[:12] == the release's ``clip__sha12`` for that sid (the join key is the trainer's stable episode id).
Any unreadable artifact, a missing key or an empty window set reads INCONCLUSIVE -- never PASS.  Output: one JSON (``--out``), sha12 / episode
index keyed only (no raw clip id ever leaves this script).

    python feed_parity_pf.py --pf-dump <dir> --clip-dump <dir> --v9 <npz> [--v9-md5 <md5>] --out <json> [--label <name>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

V9_TO_NAV_CMD = {0: 0, 1: 1, 2: 2}       # LITERAL (refc_v3_train.py:4686)
NAV_D_CLIP_M = 1000.0                    # LITERAL (refcv8_train.py:69)
ARGS_TOL = 1e-5
KEYS = ("ws", "nav_cmd", "nav_cmd_clip", "r8_nav_known", "r8_nav_fed")


def md5_file(p) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha12(s) -> str:
    return hashlib.sha256(str(s).encode()).hexdigest()[:12]


def raw_offset_of(man: dict) -> int:
    m = re.search(r"RAW frame = ws \+ (\d+)", str((man.get("grid") or {}).get("ws_is", "")))
    if not m:
        raise ValueError(f"the dump manifest does not state the RAW offset: {(man.get('grid') or {}).get('ws_is')!r}")
    return int(m.group(1))


def scale_nav_np(d_next, d_end, dyaw, side, look, valid) -> np.ndarray:
    """Independent numpy statement of the trainer's scaling (SPEC: refcv8_train.scale_nav docstring); unknown rows are exactly zeros."""
    def lg(d):
        return math.log1p(min(max(float(d), 0.0), NAV_D_CLIP_M) / 50.0)
    k = 1.0 if valid else 0.0
    fz = lambda v: float(v) if (valid and np.isfinite(v)) else 0.0      # noqa: E731  (window_inputs: args encoded only when valid)
    side_v = float(side) if valid else 0.0
    look_v = float(np.clip(look, 0.0, 1000.0))
    row = [lg(fz(d_next)) * k, lg(fz(d_end)) * k, fz(dyaw) / 90.0 * k, max(-1.0, min(1.0, side_v)) * k, lg(look_v) * k, k]
    return np.asarray(row, np.float64)


class V9:
    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.sid = z["clip__sid"].astype(np.int64)
        self.sha = z["clip__sha12"]
        self.row0, self.nr, self.k0 = z["clip__row0"].astype(np.int64), z["clip__n_rows"].astype(np.int64), z["clip__k0"].astype(np.int64)
        self.k = z["row__k"].astype(np.int64)
        self.tok = z["row__nav_token"]
        self.valid = z["row__nav_args_valid"]
        self.cols = {c: z[f"row__{c}"] for c in ("nav_d_next_m", "nav_d_end_m", "nav_dyaw_next_deg", "nav_side_next", "nav_lookahead_m")}
        self.by_sid = {int(s): i for i, s in enumerate(self.sid)}

    def row(self, sid: int, k: int) -> int:
        i = self.by_sid[int(sid)]
        j = int(k) - int(self.k0[i])
        if not 0 <= j < int(self.nr[i]):
            raise ValueError(f"frame {k} outside clip index {i}")
        r = int(self.row0[i]) + j
        if int(self.k[r]) != int(k):
            raise ValueError(f"release row {r} holds k={int(self.k[r])}, expected {k}")
        return r

    def expected(self, sid: int, k: int) -> dict:
        r = self.row(sid, k)
        t = self.tok[r]
        if not np.isfinite(t):
            raise ValueError("non-finite v9 token")
        v = bool(int(self.valid[r]))
        c = self.cols
        return {"token": V9_TO_NAV_CMD[int(t)], "known": v,
                "fed": scale_nav_np(c["nav_d_next_m"][r], c["nav_d_end_m"][r], c["nav_dyaw_next_deg"][r], c["nav_side_next"][r],
                                    c["nav_lookahead_m"][r], v)}


def load_dump(d) -> tuple[dict, dict]:
    """{(sha12, ws): {key: value}} for every window, + {sha12: (episode_index, sid)}."""
    d = Path(d)
    man = json.load(open(d / "manifest.json", encoding="utf-8"))
    out, ids = {}, {}
    for e in man["episodes"]:
        s12 = sha12(e["clip_id"])
        ids[s12] = (int(e["episode_index"]), int(e["episode_id"]))
        z = np.load(d / "decisions" / f"ep{int(e['file_index']):03d}.npz", allow_pickle=False)
        n = len(z["ws"])
        for i, w in enumerate(z["ws"].astype(np.int64)):
            out[(s12, int(w))] = {k: (z[k][i] if (k in z.files and z[k].shape[0] == n) else None) for k in KEYS}
    return man, out, ids


def check(pf_dump, clip_dump, v9_path, v9_md5=None, label="", subset_ok=False) -> dict:
    r = {"tool": "feed_parity_pf.py", "label": label, "clause": "SHIP-R8 AMENDMENT 3 (1b) FEED PARITY", "status": "INCONCLUSIVE",
         "pf_dump": str(pf_dump), "clip_dump": str(clip_dump), "v9": str(v9_path), "checks": {}, "counts": {}}
    try:
        if v9_md5:
            got = md5_file(v9_path)
            r["v9_md5"] = got
            if got != v9_md5:
                r["error"] = f"v9 md5 {got} != expected {v9_md5}"
                return r
        man, P, ids = load_dump(pf_dump)
        _mc, C, _ids_c = load_dump(clip_dump)
        off = raw_offset_of(man)
        r["raw_offset_from_manifest"] = off
        r["manifest_nav_feed_mode"] = (man.get("nav_feed") or {}).get("mode")
        if not P:
            r["error"] = "the per-frame dump holds no windows"
            return r
        miss = sorted({k for w in P.values() for k, v in w.items() if v is None})
        if miss:
            r["error"] = f"per-frame decisions lack per-window key(s) {miss}"
            return r
        v9 = V9(v9_path)
        bad = {"T": [], "C": [], "K": [], "A": [], "I": []}
        n_diff = n_known = 0
        tok_by_clip: dict = {}
        clip_tok_by_clip: dict = {}
        for s12, (ei, sid) in ids.items():
            if str(v9.sha[v9.by_sid[sid]]) != s12:
                bad["I"].append({"episode_index": ei})
        for (s12, w), x in sorted(P.items()):
            ei, sid = ids[s12]
            e = v9.expected(sid, w + off)
            tok, ctok, kn = int(x["nav_cmd"]), int(x["nav_cmd_clip"]), bool(x["r8_nav_known"])
            fed = np.asarray(x["r8_nav_fed"], np.float64).reshape(-1)
            tok_by_clip.setdefault(s12, set()).add(e["token"])
            if tok != e["token"]:
                bad["T"].append({"episode_index": ei, "ws": w, "fed": tok, "v9": e["token"]})
            ref = C.get((s12, w))
            if ref is None or ref["nav_cmd"] is None:
                bad["C"].append({"episode_index": ei, "ws": w, "why": "window absent from the clip-fed reference"})
            else:
                rt = int(ref["nav_cmd"])
                clip_tok_by_clip.setdefault(s12, set()).add(rt)
                if ctok != rt:
                    bad["C"].append({"episode_index": ei, "ws": w, "nav_cmd_clip": ctok, "reference": rt})
            if kn != e["known"]:
                bad["K"].append({"episode_index": ei, "ws": w, "sidecar": kn, "release": e["known"]})
            nz = bool(np.abs(fed).sum() > 0)
            if nz != kn or fed.shape != (6,) or float(np.max(np.abs(fed - e["fed"]))) > ARGS_TOL:
                bad["A"].append({"episode_index": ei, "ws": w, "nonzero": nz, "known": kn,
                                 "max_abs_vs_independent": (float(np.max(np.abs(fed - e["fed"]))) if fed.shape == (6,) else None)})
            n_diff += int(tok != ctok)
            n_known += int(kn)
        wp, wc = set(P), set(C)
        r["checks"] = {
            "T_fed_token_equals_v9_per_frame_token": not bad["T"],
            "C_nav_cmd_clip_equals_clip_fed_reference": not bad["C"],
            "C_reference_constant_within_every_clip": all(len(v) == 1 for v in clip_tok_by_clip.values()) and bool(clip_tok_by_clip),
            "C_same_window_set_as_clip_fed_reference": (wp <= wc) if subset_ok else (wp == wc),
            "K_known_equals_release_args_valid": not bad["K"],
            "A_r8_nav_nonzero_exactly_where_known_and_equals_independent_scaling": not bad["A"],
            "I_clip_identity_sha12_matches_release": not bad["I"],
            "manifest_mode_is_perframe_full": r["manifest_nav_feed_mode"] == "perframe_full",
        }
        r["counts"] = {"n_windows": len(P), "n_episodes": len(ids), "n_windows_clip_reference": len(wc),
                       "n_windows_only_in_pf": len(wp - wc), "n_windows_only_in_reference": len(wc - wp),
                       "n_token_differs_from_clip": n_diff, "n_args_known": n_known,
                       "n_clips_whose_v9_token_changes_within_clip": int(sum(1 for v in tok_by_clip.values() if len(v) > 1)),
                       "n_fail": {k: len(v) for k, v in bad.items()}}
        r["first_failures"] = {k: v[:5] for k, v in bad.items() if v}
        r["status"] = "PASS" if all(r["checks"].values()) else "FAIL"
    except Exception as exc:      # noqa: BLE001 -- recorded; an unreadable artifact is INCONCLUSIVE, never PASS
        r["status"] = "INCONCLUSIVE"
        r["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    return r


def known_value_rows(pf_dump, v9_path, n=3) -> list:
    """Deterministic known-value sample for the report: the first window (sorted by episode index, ws) of the first ``n`` clips whose v9 token
    differs from the fed clip token somewhere -- sha12 / episode index / ws, the sidecar values and the release values, side by side."""
    man, P, ids = load_dump(pf_dump)
    off = raw_offset_of(man)
    v9 = V9(v9_path)
    rows, seen = [], set()
    for (s12, w), x in sorted(P.items(), key=lambda kv: (ids[kv[0][0]][0], kv[0][1])):
        if s12 in seen or int(x["nav_cmd"]) == int(x["nav_cmd_clip"]):
            continue
        ei, sid = ids[s12]
        e = v9.expected(sid, w + off)
        rows.append({"sha12": s12, "episode_index": ei, "ws": w, "k": w + off, "sidecar_nav_cmd": int(x["nav_cmd"]),
                     "sidecar_nav_cmd_clip": int(x["nav_cmd_clip"]), "release_token": e["token"], "sidecar_known": bool(x["r8_nav_known"]),
                     "release_known": e["known"], "sidecar_r8_nav_fed": [round(float(v), 6) for v in np.asarray(x["r8_nav_fed"]).reshape(-1)],
                     "independent_scaling": [round(float(v), 6) for v in e["fed"]]})
        seen.add(s12)
        if len(rows) >= n:
            break
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pf-dump", required=True)
    ap.add_argument("--clip-dump", required=True)
    ap.add_argument("--v9", required=True)
    ap.add_argument("--v9-md5", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--known-values", type=int, default=3)
    ap.add_argument("--subset-ok-smoke-only", action="store_true", help="SMOKE ONLY: the per-frame windows need only be a SUBSET of the reference")
    a = ap.parse_args(argv)
    r = check(a.pf_dump, a.clip_dump, a.v9, a.v9_md5, a.label, subset_ok=a.subset_ok_smoke_only)
    r["subset_ok_smoke_only"] = bool(a.subset_ok_smoke_only)
    if r["status"] != "INCONCLUSIVE" and a.known_values:
        try:
            r["known_value_windows"] = known_value_rows(a.pf_dump, a.v9, a.known_values)
        except Exception as exc:      # noqa: BLE001
            r["known_value_windows_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    json.dump(r, open(a.out, "w", encoding="utf-8"), indent=1, default=str)
    print(json.dumps({"status": r["status"], "label": a.label, "counts": r.get("counts"), "failed": [k for k, v in r.get("checks", {}).items() if not v],
                      "error": r.get("error")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
