"""FAR-SIDE COMPLETION AUDIT of the SAM3 perception backfill.

⛔ WHY A THIRD CENSUS EXISTS. Two already do (`hf_census.py` for the v1 prefix,
`hf_v2_census.py` for v2) and both are trustworthy. This one exists because
neither answers the question that was actually asked — *how many of the 115
clips are COMPLETE* — and because an INHERITED claim ("77/115 complete; clip
`24b6948f` stores `live: false` against `{road: 2, sky: 0}`") had to be settled
by CONTENT rather than by quoting a report. It audits BOTH prefixes in one
pass, which is the only way the supersession question can be answered at all.

============================ THE COMPLETION CRITERION =========================

Stated BEFORE measuring, and deliberately NOT a file count -- C77 banked 115
well-formed records holding ZERO detections and passed a file-count check.

A record R for fixture clip C is COMPLETE iff ALL FOUR hold:

  P1 IDENTITY   R exists under the prefix, is non-zero-byte, parses as JSON,
                and R["clip_id"] == C.
                (a record filed under the wrong name is not evidence about C)

  P2 CONTROL PRESENT
                R["liveness"] is a mapping carrying a non-empty
                R["liveness"]["n_det"].
                ph0_sam3.py:1164-1173 -- `liveness_probe` returns
                `rec = {"concepts": cs, "n_det": out}`. Absent => the record
                predates the C77 positive control, and for it an empty scene
                and a dead engine are INDISTINGUISHABLE.

  P3 CONTROL LIVE
                is_live(R["liveness"]) is True, i.e. ANY of road/sky > 0.
                ph0_sam3.py:955-978 (`is_live`) -- the `any` rule, and the
                verdict is recomputed here, never read from a stored flag.

  P4 ZERO ERRORS
                No error anywhere in R. Checked in the PAYLOAD (every
                frames[*].det[*] / frames[*].scene[*] carrying an "error" key,
                plus liveness.errors) AND cross-checked against the summary
                keys R["n_err_total"] / R["err_kinds"] -- because a summary key
                is itself a cache and C77 is the story of a cache believed.

⛔ `n_det_total > 0` IS DELIBERATELY **NOT** IN THE PREDICATE.
   Zero AGENT detections is a legitimate ABSTENTION -- an empty road has no car
   and no pedestrian (ph0_sam3.py:146-152, and main()'s banked `_note`). That
   is precisely what the road/sky control exists to separate from a dead
   engine. Adding it would mark 6 provably-empty scenes incomplete -- and
   that, MEASURED below, is exactly where the inherited "77" came from.

   The audit therefore reports `n_det_total` per clip and in aggregate, but
   scores completion on P1-P4. Both numbers are published.

V2-ONLY ADDITIONAL GATE (reported separately, never folded into P1-P4):
  P5 schema_version >= 2   and   engine.confidence_threshold == 0.25
  -- the v2 run manifest's own stated rule. A v1-floor record is present,
  non-empty, error-free and live while being the WRONG record: a detection
  floor is invisible in the payload, it shows up only as rows that are not
  there.

usage:  python sam3_audit.py [--out FILE] [--prefix P]... [--limit N]
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import json
import os
import re
import sys
import time

REPO = "Sayood/tanitad-ph0-aug120"
WT = ("G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/.claude/worktrees/"
      "interesting-tharp-463cf3")
KEYS = "G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/Keys.txt"
FIXTURE = f"{WT}/colab/fixtures/sam3_backfill_expected.json"
PREFIXES = ["sam3_backfill/", "sam3_backfill_v2/"]

#: Names a DERIVED liveness verdict has been stored under, historically. The
#: schema no longer stores any of them (ph0_sam3.py:1164 "NO `live` BOOLEAN IS
#: STORED"); this list is what the audit HUNTS FOR, because residue on the far
#: side is exactly the open question.
DERIVED_FIELDS = ("live", "all_fired", "alive", "is_live")

# --------------------------------------------------------------------------- #
# Drive I/O. MEASURED 2026-08-23: the G: mount serves cached metadata while
# FAILING content reads with OSError errno 22; the same read succeeds seconds
# later. Every Drive read is therefore retried. Nothing here ever prints a
# token: `_token` returns it to the HF client and to nowhere else.
# --------------------------------------------------------------------------- #


def _read_bytes(path: str, attempts: int = 40) -> bytes:
    last = None
    for i in range(attempts):
        try:
            with open(path, "rb") as fh:
                return fh.read()
        except OSError as e:
            last = e
            time.sleep(0.25 * (1 + i % 8))
    raise OSError(f"drive read failed after {attempts} attempts: {path} "
                  f"({last})")


def _token() -> str:
    """Read IN PLACE from the git-ignored Keys.txt. Never printed, never argv."""
    m = re.findall(r"hf_[A-Za-z0-9]+",
                   _read_bytes(KEYS).decode("utf-8", errors="replace"))
    if not m:
        raise SystemExit("no HF token found in Keys.txt")
    return max(m, key=len)


def is_live(liveness) -> bool:
    """P3. Byte-identical rule to ph0_sam3.is_live (ph0_sam3.py:977-978).

    Reads the COUNTS, never a stored flag -- the flag is a cache of a rule that
    already changed once mid-corpus (`all` -> `any`)."""
    nd = (liveness or {}).get("n_det") or {}
    return any(int(v) > 0 for v in nd.values())


def record_errors(rec: dict) -> collections.Counter:
    """P4, from the PAYLOAD. Full error strings, not the leading token that
    `err_kinds` truncates to."""
    errs: collections.Counter = collections.Counter()
    for fr in (rec.get("frames") or {}).values():
        for key in ("det", "scene"):
            for d in fr.get(key) or []:
                if isinstance(d, dict) and "error" in d:
                    errs[str(d["error"])] += 1
    for concept, e in ((rec.get("liveness") or {}).get("errors") or {}).items():
        errs[f"[liveness:{concept}] {e}"] += 1
    return errs


def audit_record(cid: str, rec: dict, size: int) -> dict:
    """Evaluate P1-P5 on ONE record and return the per-clip row."""
    row: dict = {"clip_id": cid, "bytes": size}
    reasons: list[str] = []

    # ---- P1 identity ------------------------------------------------------
    if size == 0:
        reasons.append("P1_zero_byte")
    if rec.get("clip_id") != cid:
        reasons.append(f"P1_clip_id_mismatch({rec.get('clip_id')!r})")

    # ---- payload quantities ----------------------------------------------
    row["n_frames_run"] = int(rec.get("n_frames_run") or 0)
    row["n_det_total"] = int(rec.get("n_det_total") or 0)
    row["n_scene_det_total"] = int(rec.get("n_scene_det_total") or 0)
    row["per_concept_hits"] = {k: int(v) for k, v
                               in (rec.get("per_concept_hits") or {}).items()}
    row["per_scene_hits"] = {k: int(v) for k, v
                             in (rec.get("per_scene_hits") or {}).items()}
    row["schema_version"] = rec.get("schema_version")
    row["conf_threshold"] = (rec.get("engine") or {}).get(
        "confidence_threshold")

    # ---- P4 errors: payload is authoritative, summary is cross-checked ----
    errs = record_errors(rec)
    row["n_err_payload"] = int(sum(errs.values()))
    row["err_strings"] = dict(errs.most_common())
    row["n_err_summary"] = (None if rec.get("n_err_total") is None
                            else int(rec["n_err_total"]))
    row["err_kinds_summary"] = dict(rec.get("err_kinds") or {})
    row["summary_matches_payload"] = (row["n_err_summary"] is None
                                      or row["n_err_summary"]
                                      == row["n_err_payload"])
    if row["n_err_payload"] or (row["n_err_summary"] or 0):
        reasons.append("P4_errors")

    # ---- P2/P3 the positive control --------------------------------------
    lv = rec.get("liveness")
    counts = {k: int(v) for k, v in ((lv or {}).get("n_det") or {}).items()}
    row["liveness_present"] = lv is not None
    row["liveness_n_det"] = counts
    row["liveness_frame_idx"] = (lv or {}).get("frame_idx")
    if lv is None or not counts:
        reasons.append("P2_no_control")
        row["is_live_recomputed"] = None
    else:
        alive = is_live(lv)
        row["is_live_recomputed"] = alive
        row["control_partial"] = alive and not all(v > 0
                                                   for v in counts.values())
        if not alive:
            reasons.append("P3_control_dead")

    # ---- the STORED derived verdict: residue hunt (the 24b6948f question) --
    stored = {f: lv[f] for f in DERIVED_FIELDS
              if isinstance(lv, dict) and f in lv}
    row["stored_derived_fields"] = stored
    row["liveness_consistent"] = None
    if stored and row["is_live_recomputed"] is not None:
        # CONSISTENT iff every stored verdict equals the recomputed one.
        row["liveness_consistent"] = all(
            bool(v) == row["is_live_recomputed"] for v in stored.values())

    # ---- P5, v2 gate, reported but NOT folded into complete ---------------
    row["p5_schema_ok"] = (row["schema_version"] is not None
                           and int(row["schema_version"]) >= 2)
    row["p5_conf_ok"] = (row["conf_threshold"] is not None
                         and abs(float(row["conf_threshold"]) - 0.25) < 1e-9)

    row["incomplete_reasons"] = reasons
    row["complete"] = not reasons
    return row


def audit_prefix(api, hf_hub_download, tok, prefix: str, want: list[str],
                 limit: int | None = None) -> dict:
    info = api.dataset_info(REPO, files_metadata=True)
    far = {f.rfilename: (f.size or 0) for f in info.siblings
           if f.rfilename.startswith(prefix)}
    runs = sorted(rf for rf in far if "/_runs/" in rf)
    recs = sorted(rf for rf in far
                  if rf.endswith(".json") and "/_runs/" not in rf)
    if limit:
        recs = recs[:limit]
    print(f"[{prefix}] {len(recs)} records + {len(runs)} run manifests",
          flush=True)
    if not recs:
        return {"prefix": prefix, "n_records": 0,
                "note": "PREFIX EMPTY / ABSENT on the far side"}

    rows, seen = [], set()
    for i, rf in enumerate(recs):
        cid = rf[len(prefix):-len(".json")]
        seen.add(cid)
        if far[rf] == 0:
            rows.append({"clip_id": cid, "bytes": 0, "complete": False,
                         "incomplete_reasons": ["P1_zero_byte"]})
            continue
        for attempt in range(4):
            try:
                p = hf_hub_download(REPO, rf, repo_type="dataset", token=tok,
                                    force_download=True)
                with open(p, "rb") as fh:
                    rec = json.loads(fh.read().decode("utf-8", "replace"))
                break
            except Exception as e:                              # noqa: BLE001
                if attempt == 3:
                    rows.append({"clip_id": cid, "bytes": far[rf],
                                 "complete": False,
                                 "incomplete_reasons":
                                     [f"P1_unreadable({type(e).__name__})"]})
                    rec = None
                    break
                time.sleep(2 * (attempt + 1))
        if rec is None:
            continue
        rows.append(audit_record(cid, rec, far[rf]))
        if (i + 1) % 25 == 0:
            print(f"[{prefix}] {i+1}/{len(recs)} read", flush=True)

    # ------------------------------ aggregate ------------------------------
    per_c, per_s, errs, err_kinds = (collections.Counter(),
                                     collections.Counter(),
                                     collections.Counter(),
                                     collections.Counter())
    for r in rows:
        per_c.update(r.get("per_concept_hits") or {})
        per_s.update(r.get("per_scene_hits") or {})
        errs.update(r.get("err_strings") or {})
        err_kinds.update(r.get("err_kinds_summary") or {})

    complete = [r for r in rows if r.get("complete")]
    incomplete = [r for r in rows if not r.get("complete")]
    reason_hist: collections.Counter = collections.Counter()
    for r in incomplete:
        for why in r.get("incomplete_reasons") or []:
            reason_hist[why.split("(")[0]] += 1

    stored_any = [r for r in rows if r.get("stored_derived_fields")]
    disagree = [r for r in stored_any if r.get("liveness_consistent") is False]
    partial = [r for r in rows if r.get("control_partial")]
    zero_det = [r for r in rows if r.get("liveness_present")
                and not r.get("n_det_total")]

    missing = sorted(set(want) - seen)
    return {
        "prefix": prefix,
        "n_files_listed": len(far),
        "n_records": len(recs),
        "n_run_manifests": len(runs),
        "run_manifests": runs,
        "n_zero_byte": sum(1 for r in rows if r.get("bytes") == 0),
        "fixture_coverage": f"{len(seen & set(want))}/{len(want)}",
        "missing_vs_fixture": missing,
        "n_missing_vs_fixture": len(missing),
        "extra_not_in_fixture": sorted(seen - set(want)),
        "n_frames_run_total": sum(r.get("n_frames_run") or 0 for r in rows),
        "n_det_total": sum(r.get("n_det_total") or 0 for r in rows),
        "n_scene_det_total": sum(r.get("n_scene_det_total") or 0
                                 for r in rows),
        "per_concept_totals": dict(per_c.most_common()),
        "per_scene_totals": dict(per_s.most_common()),
        # ⛔ THE ERROR CENSUS -- full strings from the payload, and the
        # summary's own truncated kinds beside it so a divergence is visible.
        "error_census_payload": dict(errs.most_common()),
        "n_err_payload_total": int(sum(errs.values())),
        "error_kinds_summary": dict(err_kinds.most_common()),
        "n_records_summary_disagrees_with_payload":
            sum(1 for r in rows if r.get("summary_matches_payload") is False),
        "liveness": {
            "control_present": sum(1 for r in rows
                                   if r.get("liveness_present")),
            "control_absent": sum(1 for r in rows
                                  if r.get("liveness_present") is False),
            "live": sum(1 for r in rows
                        if r.get("is_live_recomputed") is True),
            "dead": sum(1 for r in rows
                        if r.get("is_live_recomputed") is False),
            "partial_one_control_occluded": len(partial),
        },
        "partial_control_clips": [{"clip_id": r["clip_id"],
                                   "n_det": r["liveness_n_det"],
                                   "n_det_total": r["n_det_total"]}
                                  for r in partial],
        # ⭐ THE RESIDUE HUNT: does ANY record still store the derived verdict,
        # and does it agree with the counts it summarises?
        "liveness_consistency": {
            "records_storing_a_derived_field": len(stored_any),
            "stored_field_names": sorted({k for r in stored_any
                                          for k in r["stored_derived_fields"]}),
            "agreeing": sum(1 for r in stored_any
                            if r.get("liveness_consistent") is True),
            "disagreeing": len(disagree),
            "disagreeing_clips": [
                {"clip_id": r["clip_id"],
                 "stored": r["stored_derived_fields"],
                 "recomputed": r["is_live_recomputed"],
                 "n_det": r["liveness_n_det"]} for r in disagree],
        },
        "zero_det_with_live_control": [
            {"clip_id": r["clip_id"], "liveness_n_det": r["liveness_n_det"],
             "n_scene_det_total": r.get("n_scene_det_total")}
            for r in zero_det if r.get("is_live_recomputed")],
        "zero_det_with_dead_control": [
            {"clip_id": r["clip_id"], "liveness_n_det": r["liveness_n_det"]}
            for r in zero_det if r.get("is_live_recomputed") is False],
        "p5_v2_gate": {
            "wrong_schema": sum(1 for r in rows if not r.get("p5_schema_ok")),
            "wrong_conf": sum(1 for r in rows if not r.get("p5_conf_ok"))},
        # ------------------------- THE ANSWER ------------------------------
        "completion": {
            "predicate": "P1 identity AND P2 control present AND "
                         "P3 control live AND P4 zero errors "
                         "(n_det_total>0 deliberately EXCLUDED)",
            "n_expected": len(want),
            "n_complete": len(complete),
            "n_incomplete": len(incomplete) + len(missing),
            "n_incomplete_records_present": len(incomplete),
            "n_incomplete_missing_entirely": len(missing),
            "incomplete_reason_histogram": dict(reason_hist.most_common()),
            "incomplete_clips": [{"clip_id": r["clip_id"],
                                  "reasons": r.get("incomplete_reasons")}
                                 for r in incomplete],
            # published beside it, NOT the criterion -- this is the number the
            # inherited "77/115" claim was actually reporting.
            "n_complete_AND_nonzero_det": sum(1 for r in complete
                                              if r.get("n_det_total")),
            "n_complete_but_zero_det": sum(1 for r in complete
                                           if not r.get("n_det_total")),
        },
        "per_clip": sorted(rows, key=lambda r: r["clip_id"]),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("sam3_audit")
    ap.add_argument("--prefix", action="append", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)

    try:                                   # the dev box sits behind a TLS proxy
        import truststore
        truststore.inject_into_ssl()
    except Exception:                                            # noqa: BLE001
        print("[warn] truststore unavailable; TLS may fail", flush=True)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    from huggingface_hub import HfApi, hf_hub_download

    want = json.loads(_read_bytes(FIXTURE).decode("utf-8"))["clips"]
    if len(want) != 115:
        print(f"[warn] fixture holds {len(want)} clips, expected 115")
    tok = _token()
    api = HfApi(token=tok)

    out = {
        "class": "MEASURED",
        "generated_utc": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "auditor": "sam3_audit.py (2026-08-23-sam3-completion-audit)",
        "repo": REPO,
        "fixture": {"path": FIXTURE, "n_clips": len(want)},
        "criterion": (__doc__ or "").split(
            "============================ THE COMPLETION CRITERION"
            " =========================")[-1].strip(),
        "prefixes": {},
    }
    for pref in (a.prefix or PREFIXES):
        out["prefixes"][pref] = audit_prefix(api, hf_hub_download, tok, pref,
                                             want, a.limit)

    txt = json.dumps(out, indent=1, ensure_ascii=False)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "wb") as fh:
            fh.write(txt.encode("utf-8"))
        print("wrote", a.out)
    for pref, rep in out["prefixes"].items():
        c = rep.get("completion") or {}
        print(f"\n===== {pref}")
        print(f"  records          {rep.get('n_records')}  "
              f"coverage {rep.get('fixture_coverage')}")
        print(f"  n_det_total      {rep.get('n_det_total')}  "
              f"scene {rep.get('n_scene_det_total')}")
        print(f"  errors (payload) {rep.get('n_err_payload_total')}  "
              f"kinds {len(rep.get('error_census_payload') or {})}")
        print(f"  liveness         {rep.get('liveness')}")
        print(f"  stored-flag residue "
              f"{rep.get('liveness_consistency', {}).get('records_storing_a_derived_field')}"
              f"  disagreeing "
              f"{rep.get('liveness_consistency', {}).get('disagreeing')}")
        print(f"  COMPLETE         {c.get('n_complete')}/{c.get('n_expected')}"
              f"   INCOMPLETE {c.get('n_incomplete')}")
        print(f"  reasons          {c.get('incomplete_reason_histogram')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
