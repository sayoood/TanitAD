"""``--reuse-floors <run dir>`` — adopt a previous run's FULLY-SCORED floor arms (CV, STOP) instead of
re-scoring them, after an identity check that REFUSES on any mismatch.

⛔ WHY (W7, 2026-09-21, the orchestrator's lever, MEASURED). The navhard scoring of refcv4b died TWICE
in its first scored arm at ~80 % — attempt 0: STOP at 4,915/5,912 agent calls; attempt 1: CV at
stage-two scenario 4,294/5,462 — both times on ``E1_RAM_GUARD_ABORT`` with the box's available memory
at 948–1,750 MB while OTHER sessions' jobs held it (the scorer's own RSS was 768–774 MB). CV and STOP
on navhard were ALREADY scored completely, 5,912/5,912, in ``…/20260920T082848Z-navsim_v2-none-06e257``
(CV == the official leaderboard 0.11481608441648 exactly). Re-scoring them is ~3 h of pure exposure to
the killer that has now struck twice. W3 did precisely this on navtest ("floors are the banked full-
split CSVs restricted to the same tokens — no rescoring").

⛔ THE MANDATORY-FLOORS RULE IS SATISFIED, NOT SKIPPED. STOP and CV are mandatory on every NavSim run
because the paired comparison needs them ON THE SAME TOKENS. A floor adopted from a run that scored the
IDENTICAL token set, with the IDENTICAL devkit, patch set and metric cache, IS a floor on the same
tokens — its per-token rows are what a re-score would reproduce, since the scorer is deterministic given
the plan (STOP's plan is all-zero; CV is the devkit's own agent). The adoption is RECORDED in the
artifact with its source run; it is never a silent skip.

⛔ WHAT IS CHECKED, AND WHY EACH ONE (any mismatch RAISES; there is no fall-back to re-scoring):
  1. the SOURCE arm really completed: ``status PASS``, ``rc 0``, ``log_successful == expected``,
     ``log_failed == 0``, ``csv_valid_rows == expected`` — a floor that never finished is not a floor;
  2. the same SPLIT and PROTOCOL;
  3. the same DEVKIT: the source run's recorded sha == this run's measured sha;
  4. the same PATCH SET, by name AND raw-bytes blob — ⚠ a patch present in one run and not the other
     changes which scenes can be scored at all (the IDM degenerate-path patch decides 166 tokens);
  5. the same METRIC CACHE path, and the same cache identity at preflight (metadata rows / tokens);
  6. the IDENTICAL TOKEN SET, BY VALUE, WITH STAGE LABELS: every token of the source CSV, its stage
     read from WHICH stage columns are populated, equals this run's split-yaml assignment — 5,912
     tokens, 450 stage-1, 5,462 stage-2, not a count match but a set match;
  7. the same AGENT INPUTS (export content sha256) — the scene set the scorer was driven over;
  8. for a SEAM floor (STOP), the seam this run builds is BYTE-EQUAL in tokens and poses to the source
     run's seam — STOP is deterministic, so a difference would mean the arm itself changed.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

#: ⭐ ECHO joined 2026-09-21 (W7): on attempt 2 ECHO PASSED (5,912/5,912, 4,663 s) and A1 was then
#: exposed alone. ECHO's plan is a DETERMINISTIC function of the export's t0 ego block (ha0_ext via
#: `kinematic_goal_extrapolation`), independent of the checkpoint — so a PASSED ECHO from an attempt
#: that later failed on another arm is exactly as reusable as CV/STOP, under the same identity check
#: PLUS its seam byte-equal to this run's (the check that STOP already carries).
REUSABLE_FLOORS = ("CV", "STOP", "ECHO")
SEAM_FLOORS = ("STOP", "ECHO")
_SUMMARY_ROWS = ("extended_pdm_score_stage_one", "extended_pdm_score_stage_two",
                 "extended_pdm_score_combined", "average_all_frames")


class FloorReuseRefused(RuntimeError):
    """An adoption the identity check would not allow — never downgraded to a warning."""


def _load(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _patch_key(patches) -> list:
    """(name, raw-bytes blob) per patch; a patch with no file (packaging-only) keys on its name."""
    return sorted((p.get("name"), p.get("raw_bytes_git_blob_now")) for p in (patches or []))


def _csv_token_stages(csv_path: Path) -> dict:
    """token -> 1 | 2, read from WHICH stage columns the devkit populated (not from any label)."""
    import pandas as pd
    d = pd.read_csv(csv_path, index_col=0)
    d = d[~d["token"].isin(_SUMMARY_ROWS)]
    s1 = d["ego_progress_stage_one"].notna()
    return {str(t): (1 if a else 2) for t, a in zip(d["token"], s1)}


def check_and_adopt(*, arm: str, src_run: Path, run, prof, split_yaml: dict, devkit_sha: str,
                    patches: list, preflight: dict, export_sha256: str | None,
                    new_seam: Path | None, log=print, kind: str = "floor") -> dict:
    """Verify, copy ``raw/<arm>/`` from ``src_run`` into this run, and return the ``score_arm``-shaped
    record the summary pipeline consumes. Raises :class:`FloorReuseRefused` on ANY mismatch.

    ``kind="model"`` (W7 2026-09-21): a SCORED MODEL ARM, adopted to RE-DERIVE a completed run's
    artifacts/summary after an artifact-builder fix, WITHOUT re-scoring. ⛔ The plan identity is then
    the load-bearing check, so it is MANDATORY: this run's seam (``raw/model/<arm>.npz``, itself adopted
    by the identity-checked ``--reuse-seams``) must be BYTE-EQUAL to the source run's — same plan, same
    tokens, same devkit, patches, cache and inputs ⇒ the scorer's rows are the same rows."""
    if kind == "model":
        pass
    elif arm not in REUSABLE_FLOORS:
        raise FloorReuseRefused(f"{arm} is not a reusable floor (only {REUSABLE_FLOORS})")
    src_run = Path(src_run)
    bad: list = []
    sbr = _load(src_run / "bench_run.json")
    src_counts = _load(src_run / "raw" / arm / f"{arm}.counts.json")
    expected = int(prof.n_stage1) + int(prof.n_stage2)
    # 1 — the source arm COMPLETED
    for k, want in (("status", "PASS"), ("rc", 0), ("log_successful", expected), ("log_failed", 0),
                    ("csv_valid_rows", expected), ("expected_tokens", expected)):
        if src_counts.get(k) != want:
            bad.append(f"source {arm}.{k} = {src_counts.get(k)!r}, need {want!r}")
    # 2 — split + protocol
    if (sbr.get("split") or {}).get("name") != prof.name:
        bad.append(f"source split {(sbr.get('split') or {}).get('name')!r} != {prof.name!r}")
    if sbr.get("protocol") != prof.protocol:
        bad.append(f"source protocol {sbr.get('protocol')!r} != {prof.protocol!r}")
    # 3 — devkit sha
    s_sha = (sbr.get("devkit") or {}).get("sha")
    if s_sha != devkit_sha:
        bad.append(f"devkit sha: source {s_sha} != this run {devkit_sha}")
    # 4 — patch set by name AND raw-bytes blob
    s_p, n_p = _patch_key((sbr.get("devkit") or {}).get("patches")), _patch_key(patches)
    if s_p != n_p:
        only_s = [x for x in s_p if x not in n_p]
        only_n = [x for x in n_p if x not in s_p]
        bad.append(f"patch set differs: only in source {only_s}; only in this run {only_n}")
    # 5 — metric cache path + identity
    if str(src_counts.get("cache")) != str(prof.cache).replace(os.sep, "/"):
        bad.append(f"metric cache: source {src_counts.get('cache')} != this run {prof.cache}")
    s_pre = _load(src_run / "raw" / "preflight.json") if (src_run / "raw" / "preflight.json").exists() else {}
    for k in ("n_metadata_rows", "n_tokens"):
        sv, nv = (s_pre.get("cache") or {}).get(k), (preflight.get("cache") or {}).get(k)
        if sv != nv:
            bad.append(f"metric cache {k}: source {sv} != this run {nv}")
    # 6 — the IDENTICAL token set, by value, with stage labels
    src_csv = src_run / "scores" / f"{arm}.csv"
    got = _csv_token_stages(src_csv)
    want = {str(t): 1 for t in split_yaml["stage_one"]} | {str(t): 2 for t in split_yaml["stage_two"]}
    if got != want:
        miss = sorted(set(want) - set(got))[:3]
        extra = sorted(set(got) - set(want))[:3]
        flip = sorted(t for t in set(got) & set(want) if got[t] != want[t])[:3]
        bad.append(f"token set differs: {len(got)} vs {len(want)}; missing {miss}; extra {extra}; "
                   f"stage flipped {flip}")
    n1 = sum(1 for v in got.values() if v == 1)
    # 7 — the same agent inputs
    s_exp = src_run / "raw" / "export_record.json"
    s_sha256 = _load(s_exp).get("sha256") if s_exp.exists() else None
    if export_sha256 is not None and s_sha256 != export_sha256:
        bad.append(f"agent-input export sha256: source {s_sha256} != this run {export_sha256}")
    # 8 — a SEAM floor's plan must be byte-equal (STOP and ECHO are deterministic)
    seam_eq = None
    if arm in SEAM_FLOORS or kind == "model":
        s_seam = src_run / "raw" / ("model" if kind == "model" else "seams") / f"{arm}.npz"
        if new_seam is None or not Path(new_seam).exists() or not s_seam.exists():
            bad.append(f"{arm} seam missing on one side; its plan cannot be compared")
        else:
            a, b = np.load(s_seam, allow_pickle=False), np.load(new_seam, allow_pickle=False)
            # ⛔ BYTE-equal means BYTES. MEASURED 2026-09-21: `np.array_equal` on A1's poses returned False
            # for three files with the IDENTICAL sha256 (dd7814b7e74e8e85), because the 450 CV-stand-in rows
            # are all-NaN and NaN != NaN under IEEE-754 — a FALSE REFUSAL of a byte-identical plan. A check
            # named "byte-equal" that compares VALUES differs from its name exactly on NaN (and on -0.0).
            seam_eq = bool(all(a[k].dtype == b[k].dtype and a[k].shape == b[k].shape
                               and a[k].tobytes() == b[k].tobytes() for k in ("token", "poses")))
            if not seam_eq:
                bad.append(f"{arm} seam differs from the source run's (tokens or poses) — the arm itself changed")
    if bad:
        raise FloorReuseRefused(f"--reuse-floors REFUSED for {arm}: " + " | ".join(bad))

    # ---- adopt: copy the arm's raw directory, then write the record ----------------------
    dst = run.p(f"raw/{arm}")
    dst.mkdir(parents=True, exist_ok=True)
    copied = []
    for f in sorted((src_run / "raw" / arm).iterdir()):
        if f.is_file():
            shutil.copyfile(f, dst / f.name)
            copied.append(f.name)
    csv_dst = dst / f"{arm}.devkit.csv"
    h_src = hashlib.sha256(src_csv.read_bytes()).hexdigest()
    h_dst = hashlib.sha256(csv_dst.read_bytes()).hexdigest()
    if h_src != h_dst:
        # the banked scores/<arm>.csv and raw/<arm>/<arm>.devkit.csv must be the same bytes
        raise FloorReuseRefused(f"{arm}: scores/{arm}.csv ({h_src[:12]}) != raw/{arm}/{arm}.devkit.csv "
                                f"({h_dst[:12]}) in the SOURCE run — its own record is inconsistent")
    prov = {"reused": True, "kind": kind, "source_run": src_run.name,
            "source_run_dir": str(src_run).replace(os.sep, "/"),
            "rule": ("RE-DERIVATION, NOT A NEW MEASUREMENT: this model arm's per-token rows are adopted, byte-"
                     "identical, from the run that SCORED them; the plan (seam) is byte-equal, so the scorer's "
                     "output is the same output. Adopted only to rebuild artifacts/summary after a builder fix."
                     if kind == "model" else
                     "the mandatory-floors rule is SATISFIED by a floor scored on the SAME TOKENS: this "
                     "arm's per-token rows are adopted, byte-identical, from a run that scored the "
                     "identical token set with the identical devkit, patch set, metric cache and agent "
                     "inputs. It was NOT re-scored in this run, and that is recorded, never skipped."),
            "why": ("the killer that aborted refcv4b's navhard scoring twice (E1_RAM_GUARD_ABORT at "
                    "948-1,750 MB available while other sessions' jobs held the box) struck both times at "
                    "~80 % of the FIRST scored arm; re-scoring two floors that already exist was ~3 h of "
                    "pure exposure to it"),
            "checked": {"source_arm_completed": {k: src_counts.get(k) for k in
                                                 ("status", "rc", "log_successful", "log_failed",
                                                  "csv_valid_rows", "expected_tokens")},
                        "split": prof.name, "protocol": prof.protocol, "devkit_sha": devkit_sha,
                        "patch_set": [x[0] for x in n_p], "metric_cache": str(prof.cache).replace(os.sep, "/"),
                        "token_set": {"n": len(got), "n_stage1": n1, "n_stage2": len(got) - n1,
                                      "equal_by_value_with_stage_labels": True},
                        "export_sha256": export_sha256, "seam_byte_equal": seam_eq},
            "scores_csv_sha256": h_src, "files_copied": copied}
    (dst / f"{arm}.reused.json").write_text(json.dumps(prov, indent=1), encoding="utf-8")
    rep = dict(src_counts)
    rep.update({"csv": str(csv_dst), "reused_from": prov, "status": "PASS", "retryable": False,
                "wall_s": 0.0, "wall_s_in_source_run": src_counts.get("wall_s")})
    (dst / f"{arm}.counts.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    log(f"[navsim] {arm}: REUSED from {src_run.name} — {len(got)} tokens ({n1} stage-1) identical by value, "
        f"devkit {devkit_sha[:7]}, {len(n_p)} patches, cache + export identical"
        + ("" if seam_eq is None else f", {arm} seam byte-equal={seam_eq}") + " (not re-scored)")
    return rep
