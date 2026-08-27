#!/usr/bin/env python3
"""CRITERIA COMPLETENESS CHECK — diff an eval artifact against the binding registry.

WHY THIS EXISTS (TANITAD_PROGRAMME.md §2, EvalFlyWheel charter §5):

    "Criteria completeness is enforced by machinery, not memory — we measurably
     failed to keep them complete and constant by hand."

Three ADE-only reports went out after the four-families rule was made binding.
A rule in prose decays; C126 measured exactly that decay, where a correction
living only in a report was re-counted by every later census because **no prose
correction can reach a glob**. So the rule ships as a tool plus a test.

THE THREE STATES — the whole point of this instrument
-----------------------------------------------------
    PRESENT   the metric is in the artifact
    REFUSED   the artifact's `refused` block names the criterion AND gives a
              reason. ADMISSIBLE — and it becomes a WORK ITEM, visibly.
    ABSENT    neither. ⛔ VIOLATION — the silent omission the doctrine forbids.

The distinction between REFUSED and ABSENT is the instrument's reason to exist.
An eval that cannot compute headway because no lead-agent state exists is doing
the right thing when it SAYS so with its n and its reason. An eval that simply
never mentions the tactical family is hiding a gap, and reads at a glance
exactly like an eval that has no gap. This tool makes those two look different.

Usage
-----
    python tools/criteria_check.py taniteval/results/driving_flagship-30k.json
    python tools/criteria_check.py --all taniteval/results/
    python tools/criteria_check.py --all taniteval/results/ --json report.json
    python tools/criteria_check.py <artifact> --strict     # exit 1 on ABSENT

Exit codes: 0 = no violations (or non-strict), 1 = ABSENT criteria under --strict,
2 = the artifact could not be read/parsed.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "products" / "P7-TanitEval" / "CRITERIA_REGISTRY.json"

PRESENT, REFUSED, PARTIAL, ABSENT = "PRESENT", "REFUSED", "PARTIAL", "ABSENT"

# The mount holding this repo is a Google Drive filesystem that intermittently
# returns OSError on a file that is genuinely there (MEASURED 2026-08-23: 71 of
# 74 result JSONs failed a first read; 67 succeeded within 8 retries). A census
# that silently skipped those files would under-report coverage, which is the
# exact failure class this tool exists to prevent — so reads retry, and a file
# that never opens is reported as UNREADABLE rather than dropped.
_RETRIES = 8
_BACKOFF_S = 0.15


def _read_json(path: Path) -> dict | None:
    for attempt in range(_RETRIES):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except OSError:
            time.sleep(_BACKOFF_S * (attempt + 1))
        except json.JSONDecodeError:
            return None
    return None


def _dig(obj, dotted: str):
    """Resolve a dotted path. Returns (found, value).

    ⛔ A KEY MAY ITSELF CONTAIN A DOT, and splitting naively makes such a field
    unreachable — which this instrument then reports as ABSENT. MEASURED
    2026-08-23: the harness emits `n_excluded_goal_below_0.5m` inside
    `goal_setting`; a path ending in that name resolves to nothing under a plain
    `.split(".")`, and the criterion reads as a silent omission.

    That is precisely the C133-b failure mode — a resolution bug and a real gap
    are the same observation — so resolution tries the LONGEST literal key at
    each level before falling back to splitting. Exact keys always win.
    """
    def walk(cur, path: str):
        if not path:
            return True, cur
        if not isinstance(cur, dict):
            return False, None
        # Exact match on the whole remaining path wins (handles dotted names).
        if path in cur:
            return True, cur[path]
        # Otherwise try progressively shorter prefixes, longest first, so a key
        # containing a dot is still reachable when more path follows it.
        candidates = [k for k in cur if path.startswith(k + ".")]
        for key in sorted(candidates, key=len, reverse=True):
            ok, val = walk(cur[key], path[len(key) + 1:])
            if ok:
                return True, val
        return False, None

    return walk(obj, dotted)


def _refused_block(artifact: dict) -> dict:
    r = artifact.get("refused")
    return r if isinstance(r, dict) else {}


IN_SCOPE, OUT_OF_SCOPE, UNKNOWN_SCOPE = "IN_SCOPE", "OUT_OF_SCOPE", "UNKNOWN_SCOPE"


def _matches(artifact: dict, rule: dict) -> bool:
    kind, value = rule.get("kind"), rule.get("value")
    if kind == "key_present":
        found, val = _dig(artifact, value)
        return found and val is not None
    if kind == "block_prefix":
        blk = artifact.get("block")
        return isinstance(blk, str) and blk.startswith(value)
    return False


def scope_of(artifact: dict, registry: dict) -> tuple[str, str]:
    """Decide whether the criteria apply AT ALL, before evaluating any of them.

    Scoring a profiling artifact against driving criteria manufactures a violation
    count that is pure noise. In-scope markers are checked FIRST so an artifact
    that is genuinely a driving eval is never excused by also carrying a profiling
    block. An artifact matching neither is UNKNOWN — surfaced for a human, never
    counted as compliant and never silently dropped.
    """
    app = registry.get("applicability", {})
    for rule in app.get("in_scope_if_any", []):
        if _matches(artifact, rule):
            return IN_SCOPE, f"matched {rule.get('kind')}={rule.get('value')!r}"
    for rule in app.get("out_of_scope_if_any", []):
        if _matches(artifact, rule):
            return OUT_OF_SCOPE, rule.get("reason", f"matched {rule.get('value')!r}")
    return UNKNOWN_SCOPE, "matches no in-scope or out-of-scope marker"


# A criterion can be declined in two idioms, and BOTH are honest:
#   1. a top-level `refused` block naming it            (taniteval.driving style)
#   2. the criterion's own value being a status object  (four_families style):
#        {"status": "UNAVAILABLE", "reason": "...", "n": 6844}
# Missing the second idiom is not cosmetic: it would read an explicit, reasoned
# refusal as a silent omission and report the most disciplined artifacts in the
# repo as the least compliant.
_DECLINED = {"UNAVAILABLE", "REFUSED", "N/A", "NA", "NOT_APPLICABLE", "BLOCKED"}


def _status_of(val) -> tuple[str, str] | None:
    """If val is an inline status object, return its (state, detail); else None."""
    if not isinstance(val, dict):
        return None
    status = str(val.get("status", "")).strip().upper()
    if not status:
        return None
    if status in _DECLINED:
        reason = str(val.get("reason", "")).strip()
        n = val.get("n", val.get("n_windows_it_would_have_had"))
        if reason:
            tail = f" (n={n})" if n is not None else ""
            return REFUSED, reason[:200] + tail
        return ABSENT, f"status={status} but NO reason given"
    return PRESENT, f"status={status}"


def classify(artifact: dict, crit: dict) -> tuple[str, str]:
    """Classify one criterion against one artifact. Returns (state, detail)."""
    for key in crit.get("keys", []):
        found, val = _dig(artifact, key)
        if not found or val is None:
            continue
        inline = _status_of(val)
        if inline is not None:
            state, detail = inline
            return state, f"{key}: {detail}"
        return PRESENT, key

    # REFUSED is only admissible WITH a reason. A refused entry whose reason is
    # empty is not an acknowledgement, it is a shrug — treat it as ABSENT.
    refused = _refused_block(artifact)
    for name in crit.get("refused_as", []):
        if name in refused:
            reason = str(refused[name] or "").strip()
            if reason:
                return REFUSED, reason
            return ABSENT, f"listed in `refused` as {name!r} but with NO reason given"

    for key in crit.get("partial_keys", []):
        found, val = _dig(artifact, key)
        if found and val is not None:
            return PARTIAL, f"{key} only (weaker form of the criterion)"

    return ABSENT, "no key present and not refused"


def resolve_tier(artifact: dict, registry: dict) -> tuple[str | None, str | None]:
    """Return (tier_id, raw_value). tier_id is None when unstamped."""
    tiers = registry.get("tiers", {})
    for path in tiers.get("key_paths", []):
        found, val = _dig(artifact, path)
        if found and isinstance(val, str):
            for tid, spec in tiers.get("values", {}).items():
                if val in spec.get("aliases", []) or val == tid:
                    return tid, val
            return None, val          # stamped, but not a tier we recognise
    return None, None


def audit_keys(artifacts: list[dict], registry: dict) -> dict:
    """Which criteria resolve in NO artifact at all?

    ⛔ A key that resolves nowhere is far more likely a REGISTRY TYPO than a
    universal absence — and the two are indistinguishable in a census, because
    both print as "0 present". MEASURED 2026-08-23: `tac.confusion` looked for
    `confusion` while the emitter writes `confusion_gt_rows_pred_cols`
    (four_families.py:563). The census reported "0/135, silently absent
    everywhere" and it reached the PI as the programme's surviving eval gap. It
    was a spelling mistake.

    So UNRESOLVABLE is reported SEPARATELY from ABSENT. A finding needs a key
    that at least one real artifact is known to answer.
    """
    unresolvable, resolved = [], {}
    for fam, spec in registry.get("families", {}).items():
        for crit in spec.get("criteria", []):
            hits = {}
            for key in crit.get("keys", []) + crit.get("partial_keys", []):
                n = sum(1 for d in artifacts
                        if _dig(d, key)[0] and _dig(d, key)[1] is not None)
                if n:
                    hits[key] = n
            if hits:
                resolved[crit["id"]] = hits
            elif not crit.get("not_yet_emitted"):
                unresolvable.append({
                    "family": fam, "id": crit["id"],
                    "keys_tried": crit.get("keys", []),
                    "verdict": ("resolves in NO artifact — read the EMITTER and fix "
                                "the key, or set not_yet_emitted:true if the metric "
                                "genuinely does not exist yet")})
    return {"unresolvable": unresolvable, "resolved": resolved}


def check_artifact(artifact: dict, registry: dict) -> dict:
    scope, scope_why = scope_of(artifact, registry)
    if scope != IN_SCOPE:
        return {"scope": scope, "scope_why": scope_why, "tier": None, "tier_raw": None,
                "tier_is_driving_performance": False, "families": {}, "hygiene": [],
                "leak_guards": [], "n_violations": 0, "n_work_items": 0,
                "violations": [], "work_items": []}

    families = {}
    for fam, spec in registry.get("families", {}).items():
        rows = []
        for crit in spec.get("criteria", []):
            state, detail = classify(artifact, crit)
            rows.append({"id": crit["id"], "label": crit.get("label", crit["id"]),
                         "required": crit.get("required", True),
                         "state": state, "detail": detail})
        families[fam] = rows

    hygiene = []
    tier_id, tier_raw = resolve_tier(artifact, registry)
    for crit in registry.get("artifact_hygiene", {}).get("criteria", []):
        cid = crit["id"]
        if cid == "hyg.tier_stamp":
            if tier_id:
                state, detail = PRESENT, f"{tier_id} ({tier_raw})"
            elif tier_raw:
                state, detail = ABSENT, f"unrecognised tier value {tier_raw!r}"
            else:
                state, detail = ABSENT, "no tier marker at any known key path"
        elif cid == "hyg.no_forbidden_estimator":
            state, detail = _check_forbidden_estimator(artifact)
        else:
            state, detail = classify(artifact, crit)
        hygiene.append({"id": cid, "label": crit.get("label", cid),
                        "required": crit.get("required", True),
                        "state": state, "detail": detail})

    leaks = []
    for guard in registry.get("leak_guards", {}).get("guards", []):
        state, detail = classify(artifact, guard)
        leaks.append({"id": guard["id"], "label": guard.get("rule", guard["id"]),
                      "required": guard.get("required", True),
                      "state": state, "detail": detail})

    all_rows = [r for rows in families.values() for r in rows] + hygiene + leaks
    violations = [r for r in all_rows if r["state"] == ABSENT and r["required"]]
    work_items = [r for r in all_rows if r["state"] in (REFUSED, PARTIAL)]

    return {
        "scope": scope, "scope_why": scope_why,
        "tier": tier_id, "tier_raw": tier_raw,
        "tier_is_driving_performance": bool(
            registry.get("tiers", {}).get("values", {})
            .get(tier_id or "", {}).get("is_driving_performance", False)),
        "families": families, "hygiene": hygiene, "leak_guards": leaks,
        "n_violations": len(violations), "n_work_items": len(work_items),
        "violations": violations, "work_items": work_items,
    }


def _check_forbidden_estimator(artifact: dict) -> tuple[str, str]:
    """`overlapping_holdout_se` is forbidden as a DECISION-GRADE interval, but is
    allowed to appear under an explicitly deprecated/refused label — that is how
    published figures stay traceable. So find it, then judge its CONTEXT."""
    hits = []

    def walk(o, path=""):
        if isinstance(o, dict):
            for k, v in o.items():
                p = f"{path}.{k}" if path else k
                # Report the hit only at the LEAF that carries the token. A parent
                # dict stringifies to include its children, so recording parents
                # too produced a hit whose context was then truncated away — and a
                # truncated disavowal reads exactly like a use.
                if isinstance(v, (str, int, float, bool)) or v is None:
                    if "overlapping_holdout" in str(k) or "overlapping_holdout" in str(v):
                        hits.append((p, str(v)))
                elif "overlapping_holdout" in str(k):
                    hits.append((p, json.dumps(v, ensure_ascii=False)))
                walk(v, p)
        elif isinstance(o, list):
            for i, v in enumerate(o):
                if isinstance(v, (str, int, float, bool)):
                    if "overlapping_holdout" in str(v):
                        hits.append((f"{path}[{i}]", str(v)))
                walk(v, f"{path}[{i}]")

    walk(artifact)
    if not hits:
        return PRESENT, "not referenced"

    # Naming the forbidden estimator in order to DISAVOW it is good practice, not
    # a violation — the best-disciplined artifacts in the repo say "overlapping_
    # holdout_se is NOT used" verbatim. Matching the bare token would flag exactly
    # those, so judge the surrounding text, normalised (an earlier list of literal
    # tokens missed "NOT used" because it only carried "not_used").
    def _norm(s: str) -> str:
        return "".join(c if c.isalnum() else " " for c in s.lower())

    exonerating = ("deprecated", "refused", "legacy", "not used", "never used",
                   "forbidden", "must not", "no longer", "is not", "banned")
    bad = []
    for path, val in hits:
        ctx = _norm(f"{path} {val}")
        if not any(e in ctx for e in exonerating):
            bad.append((path, val))
    if bad:
        return ABSENT, f"referenced with no disavowal/deprecation context at: {bad[0][0]}"
    return PRESENT, f"named only to disavow or deprecate it ({hits[0][0]})"


# --------------------------------------------------------------- rendering ---
_MARK = {PRESENT: "  ok  ", REFUSED: " WORK ", PARTIAL: " PART ", ABSENT: "MISSING"}


def render(name: str, res: dict, verbose: bool = True) -> str:
    out = [f"=== {name} ==="]
    if res.get("scope") != IN_SCOPE:
        out.append(f"{res.get('scope')}: {res.get('scope_why')}")
        out.append("the driving criteria do not apply to this artifact — not scored.")
        return "\n".join(out)
    tier = res["tier"] or "UNSTAMPED"
    drive = "driving-performance tier" if res["tier_is_driving_performance"] \
        else "NOT a driving-performance tier"
    out.append(f"tier: {tier}  ({drive})")
    if res["tier"] == "T0":
        out.append("      T0 is a world-model diagnostic. It is NEVER driving performance.")
    out.append("")
    for fam, rows in res["families"].items():
        states = [r["state"] for r in rows]
        n_ok = sum(s == PRESENT for s in states)
        out.append(f"{fam}  [{n_ok}/{len(rows)} present]")
        for r in rows:
            if verbose or r["state"] != PRESENT:
                out.append(f"   [{_MARK[r['state']]}] {r['label']}")
                if r["state"] != PRESENT:
                    out.append(f"            -> {r['detail']}")
        out.append("")
    for title, rows in (("ARTIFACT HYGIENE", res["hygiene"]),
                        ("LEAK GUARDS", res["leak_guards"])):
        out.append(title)
        for r in rows:
            if verbose or r["state"] != PRESENT:
                out.append(f"   [{_MARK[r['state']]}] {r['label']}")
                if r["state"] != PRESENT:
                    out.append(f"            -> {r['detail']}")
        out.append("")
    out.append(f"VIOLATIONS (silently absent, required): {res['n_violations']}")
    out.append(f"WORK ITEMS (refused with a reason, or partial): {res['n_work_items']}")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("target", help="an eval JSON, or a directory with --all")
    ap.add_argument("--all", action="store_true", help="check every *.json directly under target")
    ap.add_argument("--recursive", action="store_true",
                    help="recurse into subdirectories. USE THIS for a repo-wide census: "
                         "a census run over ONE directory reports that directory's gaps, "
                         "not the repo's, and stating the former as the latter is the "
                         "'absence at one location is not absence' failure (C-EVAL-1).")
    ap.add_argument("--registry", default=str(REGISTRY))
    ap.add_argument("--json", dest="json_out", help="write the full report as JSON")
    ap.add_argument("--strict", action="store_true", help="exit 1 if any required criterion is ABSENT")
    ap.add_argument("--quiet", action="store_true", help="only show non-PRESENT rows")
    args = ap.parse_args(argv)

    registry = _read_json(Path(args.registry))
    if registry is None:
        print(f"FATAL: cannot read registry at {args.registry}", file=sys.stderr)
        return 2

    if args.recursive:
        targets = sorted(glob.glob(os.path.join(args.target, "**", "*.json"),
                                   recursive=True))
    elif args.all:
        targets = sorted(glob.glob(os.path.join(args.target, "*.json")))
    else:
        targets = [args.target]
    if not targets:
        print(f"FATAL: no JSON found at {args.target}", file=sys.stderr)
        return 2

    report, unreadable, violating, in_scope_arts = {}, [], 0, []
    for t in targets:
        art = _read_json(Path(t))
        if art is None or not isinstance(art, dict):
            unreadable.append(t)
            continue
        res = check_artifact(art, registry)
        if res.get("scope") == IN_SCOPE:
            in_scope_arts.append(art)
        report[os.path.basename(t)] = res
        violating += bool(res["n_violations"])
        if not (args.all or args.recursive):
            print(render(os.path.basename(t), res, verbose=not args.quiet))

    if args.all or args.recursive:
        print(f"SCOPE OF THIS CENSUS: {args.target}"
              f"{' (recursive)' if args.recursive else ' (NON-recursive — this directory only)'}")
        print("A finding here is a finding ABOUT THIS SCOPE. Do not restate it as a "
              "repo-wide absence without re-running recursively.\n")

        # ⛔ SELF-CHECK BEFORE REPORTING. A criterion nothing answers is a typo far
        # more often than a gap, and both print as "0 present". Say so up front, so
        # a misconfiguration can never again be read out as a programme finding.
        ak = audit_keys(in_scope_arts, registry)
        if ak["unresolvable"]:
            print("⛔ REGISTRY SELF-CHECK FAILED — these criteria resolve in NO "
                  "artifact. Treat as a CONFIG ERROR, not as a finding:")
            for u in ak["unresolvable"]:
                print(f"   {u['family']}/{u['id']}: tried {u['keys_tried']}")
            print()
        else:
            print("registry self-check: every criterion resolves in >=1 artifact.\n")

        print(_render_census(report, unreadable, registry))

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"registry_version": registry.get("version"),
                        "unreadable": unreadable, "artifacts": report},
                       indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json_out}")

    if unreadable:
        print(f"\nUNREADABLE ({len(unreadable)}) — reported, not silently skipped:",
              file=sys.stderr)
        for u in unreadable[:10]:
            print(f"   {u}", file=sys.stderr)

    return 1 if (args.strict and violating) else 0


def _render_census(report: dict, unreadable: list, registry: dict) -> str:
    """The programme-level view: which criteria are missing ACROSS the corpus.
    A per-file view cannot show that a whole family is absent everywhere."""
    total = len(report)
    scoped = {k: v for k, v in report.items() if v.get("scope") == IN_SCOPE}
    out_of = [k for k, v in report.items() if v.get("scope") == OUT_OF_SCOPE]
    unknown = [k for k, v in report.items() if v.get("scope") == UNKNOWN_SCOPE]
    n = len(scoped)

    out = [f"CRITERIA CENSUS (registry v{registry.get('version')})", "=" * 62,
           f"{total} artifacts read -> {n} in scope (driving evals), "
           f"{len(out_of)} out of scope, {len(unknown)} UNKNOWN", ""]
    if unknown:
        out.append("UNKNOWN SCOPE — a human must classify these; they are NOT counted "
                   "as compliant:")
        for u in unknown:
            out.append(f"   {u}")
        out.append("")
    if not n:
        return "\n".join(out + ["no in-scope artifacts"])

    tiers = {}
    for res in scoped.values():
        tiers[res["tier"] or "UNSTAMPED"] = tiers.get(res["tier"] or "UNSTAMPED", 0) + 1
    out.append("TIER STAMPS  (in-scope artifacts only)")
    for t, c in sorted(tiers.items(), key=lambda kv: -kv[1]):
        flag = "  <- unstamped: a number with no tier is not quotable" if t == "UNSTAMPED" else ""
        out.append(f"   {c:4d}  {t}{flag}")
    out.append("")

    for fam in registry.get("families", {}):
        out.append(f"{fam}")
        agg = {}
        for res in scoped.values():
            for r in res["families"].get(fam, []):
                d = agg.setdefault(r["id"], {"label": r["label"], PRESENT: 0,
                                             REFUSED: 0, PARTIAL: 0, ABSENT: 0})
                d[r["state"]] += 1
        for cid, d in agg.items():
            bits = [f"{d[PRESENT]} present"]
            if d[REFUSED]:
                bits.append(f"{d[REFUSED]} refused")
            if d[PARTIAL]:
                bits.append(f"{d[PARTIAL]} partial")
            if d[ABSENT]:
                bits.append(f"{d[ABSENT]} MISSING")
            mark = "  <-- silently absent everywhere" if d[ABSENT] == n else ""
            out.append(f"   {d['label']:<52} {', '.join(bits)}{mark}")
        out.append("")

    tot_v = sum(r["n_violations"] for r in scoped.values())
    tot_w = sum(r["n_work_items"] for r in scoped.values())
    out.append(f"TOTAL violations (silent omissions), over {n} in-scope artifacts: {tot_v}")
    out.append(f"TOTAL work items (refused with reason / partial): {tot_w}")
    if unreadable:
        out.append(f"UNREADABLE artifacts: {len(unreadable)} (see stderr)")
    return "\n".join(out)


if __name__ == "__main__":
    sys.exit(main())
