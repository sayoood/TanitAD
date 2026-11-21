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

BENCHMARKS (registry >= 2.10.0)
-------------------------------
An artifact that CLAIMS NavSim (`benchmark.navsim`, or `benchmark.navsim_v1`) is also
checked against `benchmarks.navsim`: its five criteria AND its four BLOCKING gates
(estimator unit, ego-status enforcement, modality label, cross-protocol). ⛔ MEASURED
2026-09-19 by two independent streams (E1, E2): before 2.10.0 this tool evaluated NONE of
that block, so every blocking NavSim gate was unenforced and every NavSim artifact passed.
Any artifact that carries a protocol TAG must carry one the registry knows
(`registered_protocol_tags` — the union of the registry's tag blocks). A gate FAILURE is
reported in the ABSENT column with a detail starting ``FAIL —``, the same convention
`hyg.no_forbidden_estimator` uses for a live use of the forbidden estimator.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
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
#: A nuScenes open-loop planning record that KNOWS what it is: reported, NEVER scored against the
#: TanitAD criteria and NEVER counted as compliant (see :func:`nuscenes_planning_state`).
EXTERNAL_ONLY = "EXTERNAL_ONLY"


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


# ------------------------------------------------- nuScenes open-loop planning (W6) --- #
# nuScenes OPEN-LOOP PLANNING IS NOT A TANITAD CRITERION. Register `H-EVAL-6` SUPPORTED, and the
# PI-approved portfolio marks it SKIP claim-bearing (`D-BENCH-PORT`, 2026-08-29): there is no
# official protocol; the L2 averaging convention alone moves ONE checkpoint 0.72 -> 1.22 m and
# FLIPS the UniAD/VAD ranking; the ground-truth HUMAN trajectory scores 0.36-0.96 % collision; and
# the field's "high-level command" is the GT future thresholded at +-2 m - our own route-echo
# defect, published as SOTA (`products/P7-TanitEval/benchmarks/NUSCENES_PROTOCOL.md`).
#
# The instrument has to tell THREE things apart, and a scope alone cannot:
#   * a nuScenes planning record declaring `claim_bearing: false`  -> EXTERNAL_ONLY (reported,
#     never scored, never counted as compliant);
#   * the SAME record claiming something (`claim_bearing` true or absent) -> VIOLATION;
#   * nuScenes planning numbers sitting INSIDE an in-scope TanitAD driving artifact -> VIOLATION,
#     which is the misuse this row exists for - an EXTERNAL_ONLY scope ALONE would excuse it,
#     because the artifact would simply stop being scored.
NUSCENES_PLANNING_KEYS = ("benchmark.nuscenes.planning", "benchmark.nuscenes.planning.protocol_tag")
NUSCENES_PLANNING_SCHEMAS = ("taniteval.nuscenes_planning/1",)
NUSC_GATE_ID = "nusc.plan.not_a_criterion"
NUSC_GATE_LABEL = "nuScenes open-loop planning is never a TanitAD criterion (H-EVAL-6)"


def nuscenes_planning_state(artifact: dict, registry: dict) -> tuple:
    """``("none" | "external_only" | "misused", detail)`` for one artifact."""
    hit = None
    for k in NUSCENES_PLANNING_KEYS:
        found, val = _dig(artifact, k)
        if found and val is not None:
            hit = k
            break
    if hit is None and str(artifact.get("schema", "")) in NUSCENES_PLANNING_SCHEMAS:
        hit = "schema=" + str(artifact.get("schema"))
    if hit is None:
        # W2's second detection mechanism: the PROTOCOL TAG. A record can be a nuScenes
        # open-loop number while carrying no planning block at all — a suite summary is
        # exactly that shape — and one detector is one detector.
        tags = [t for _, t in artifact_protocol_tags(artifact, registry)
                if t.startswith("nuScenes_OL_")]
        if tags:
            hit = f"protocol tag {tags[0]}"
    if hit is None:
        return "none", ""
    cb = artifact.get("claim_bearing", None)
    if cb is not False:
        return "misused", (
            f"{FAIL_}carries {hit} but claim_bearing is {cb!r} - it must be exactly false. "
            f"nuScenes open-loop planning is INADMISSIBLE as a TanitAD criterion (H-EVAL-6 "
            f"SUPPORTED; D-BENCH-PORT SKIP claim-bearing): a cited external-comparability row only.")
    scope, why = scope_of(artifact, registry)
    if scope == IN_SCOPE:
        return "misused", (
            f"{FAIL_}carries {hit} INSIDE an in-scope TanitAD driving artifact ({why}) - a "
            f"nuScenes open-loop number may never stand in a TanitAD criterion artifact, whatever "
            f"its claim_bearing flag says (H-EVAL-6). Keep it in its own external record.")
    return "external_only", f"nuScenes open-loop planning ({hit}); claim_bearing false"


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
    """Classify one criterion against one artifact. Returns (state, detail).

    ⛔ A KEY THAT IS PRESENT BUT **NULL** IS NOT A REFUSAL, AND SAYING SO IS THE POINT.
    MEASURED 2026-09-19 (W1, the NavSim STOP floor): `four_families.lateral` returned
    `None` for heading / yaw-rate / curvature on a stationary plan — the terms are genuinely
    UNDEFINED there — and this function read the nulls exactly as it reads MISSING keys, so
    three binding criteria were reported as silent omissions of an instrument that had in
    fact declined. The states stay four; what changes is that a null is now DIAGNOSED
    ("present but null") instead of being indistinguishable from an absent key, because the
    fix differs: a missing key needs an emitter, a null needs a REFUSAL with a reason and an
    n (`four_families.lateral` emits one since 2026-09-20).
    """
    null_keys = []
    for key in crit.get("keys", []):
        found, val = _dig(artifact, key)
        if found and val is None:
            null_keys.append(key)
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

    if null_keys:
        return ABSENT, (
            f"{null_keys[0]} is present but NULL — a null is not a refusal. It carries no reason "
            f"and no n, and reads exactly like a missing key, so an instrument that honestly "
            f"DECLINED is reported as a silent omission. A term that is UNDEFINED must emit "
            f"{{status: UNAVAILABLE, reason, n}} (four_families.lateral does, since 2026-09-20); "
            f"a 0.0 would be worse — it reads as perfect agreement.")
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
    nus, nus_why = nuscenes_planning_state(artifact, registry)
    if nus == "external_only":
        return {"scope": EXTERNAL_ONLY, "scope_why": nus_why, "tier": None, "tier_raw": None,
                "tier_is_driving_performance": False, "families": {}, "hygiene": [],
                "leak_guards": [],
                "benchmarks": [_row(NUSC_GATE_ID, NUSC_GATE_LABEL, PRESENT, nus_why)],
                "n_violations": 0, "n_work_items": 0, "violations": [], "work_items": []}
    misuse = [_row(NUSC_GATE_ID, NUSC_GATE_LABEL, ABSENT, nus_why)] if nus == "misused" else []
    scope, scope_why = scope_of(artifact, registry)
    if scope != IN_SCOPE:
        return {"scope": scope, "scope_why": scope_why, "tier": None, "tier_raw": None,
                "tier_is_driving_performance": False, "families": {}, "hygiene": [],
                "leak_guards": [], "benchmarks": misuse, "n_violations": len(misuse),
                "n_work_items": 0, "violations": list(misuse), "work_items": []}

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

    benchmarks = check_benchmarks(artifact, registry) + misuse

    all_rows = [r for rows in families.values() for r in rows] + hygiene + leaks + benchmarks
    violations = [r for r in all_rows if r["state"] == ABSENT and r["required"]]
    work_items = [r for r in all_rows if r["state"] in (REFUSED, PARTIAL)]

    return {
        "scope": scope, "scope_why": scope_why,
        "tier": tier_id, "tier_raw": tier_raw,
        "tier_is_driving_performance": bool(
            registry.get("tiers", {}).get("values", {})
            .get(tier_id or "", {}).get("is_driving_performance", False)),
        "families": families, "hygiene": hygiene, "leak_guards": leaks,
        "benchmarks": benchmarks,
        "n_violations": len(violations), "n_work_items": len(work_items),
        "violations": violations, "work_items": work_items,
    }


# ============================================================ benchmarks ===
# ⛔ WHY THIS BLOCK EXISTS. MEASURED 2026-09-19 by E1 and, independently, by E2:
# this tool iterated families / hygiene / leak guards ONLY, so `benchmarks.navsim` —
# five criteria and four BLOCKING gates — was never evaluated and every NavSim artifact
# passed. Both streams had to hand-write their own gate evaluators. A gate that the
# machinery never evaluates is a note; this block makes the registry's gates machinery.
FAIL_ = "FAIL — "
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_EVIDENCE_FILE = re.compile(r"[\w./\\-]+\.(?:json|jsonl|py|txt|md|csv|npz|log|pkl)\b")


def registered_protocol_tags(registry: dict) -> dict:
    """``{tag: source_block}`` — the UNION of the registry's protocol-tag blocks.

    Each tag lives in ITS OWN block (NavSim's closed set, nuScenes' planning tags, the
    internal tier's protocols) and is never merged into another's; the union is
    COMPUTED here so no copy can drift. The suite schema must equal this set (W1's test).
    A listed source that does not resolve is a registry defect and raises.
    """
    out = {}
    for src in registry.get("protocol_tags", {}).get("sources", []):
        found, val = _dig(registry, src)
        if not found or val is None:
            raise KeyError(f"protocol_tags source {src!r} does not resolve in the registry")
        for tag in (val.keys() if isinstance(val, dict) else val):
            out.setdefault(str(tag), src)
    return out


def artifact_protocol_tags(artifact: dict, registry: dict) -> list:
    """``[(key_path, tag)]`` for every protocol tag the artifact declares."""
    pt = registry.get("protocol_tags", {})
    paths = list(pt.get("key_paths", []))
    schema = artifact.get("schema")
    if isinstance(schema, str) and schema.startswith(pt.get("suite_record_schema_prefix",
                                                            "\0")):
        paths += pt.get("suite_record_key_paths", [])
    hits = []
    for p in paths:
        found, val = _dig(artifact, p)
        if found and isinstance(val, str) and val.strip():
            hits.append((p, val.strip()))
    return hits


def check_protocol_tag(artifact: dict, registry: dict):
    """(state, detail), or None when the artifact declares no tag at all."""
    hits = artifact_protocol_tags(artifact, registry)
    if not hits:
        return None
    known = registered_protocol_tags(registry)
    bad = [(p, t) for p, t in hits if t not in known]
    if bad:
        return ABSENT, (f"{FAIL_}protocol tag {bad[0][1]!r} at {bad[0][0]} is not registered "
                        f"(the union of {len(known)} tags in protocol_tags.sources). An "
                        f"unregistered tag is a number with no protocol.")
    return PRESENT, "; ".join(f"{t} ({known[t]})" for _, t in hits)


def _declined(val) -> bool:
    return isinstance(val, dict) and str(val.get("status", "")).strip().upper() in _DECLINED


def _row(cid, label, state, detail, required=True):
    return {"id": cid, "label": label, "required": required, "state": state, "detail": detail}


def _navsim_view(artifact: dict):
    """The artifact with its NavSim block at `benchmark.navsim` (the v1 alias mapped), or
    None when it claims no NavSim."""
    b = artifact.get("benchmark")
    if not isinstance(b, dict):
        return None
    if isinstance(b.get("navsim"), dict):
        return artifact
    if isinstance(b.get("navsim_v1"), dict):
        view = dict(artifact)
        view["benchmark"] = dict(b)
        view["benchmark"]["navsim"] = b["navsim_v1"]
        return view
    return None


def _navsim_interval_ok(iv, g: dict, tag, want: str) -> tuple:
    """(ok, detail) for ONE candidate interval block — the POST-SETTLEMENT form of the gate.

    ⭐ The declared interval AND the false-refusal detector below both run through THIS
    validator, so the gate cannot admit a block in one place and reject the identical block
    in the other. (A check derived twice is a check that can disagree with itself.)
    """
    if not isinstance(iv, dict):
        return False, f"estimator.interval is a {type(iv).__name__}, not a block"
    est = str(iv.get("estimator", ""))
    if est not in g.get("estimators_admissible", []):
        return False, (f"interval estimator {est!r} is not admissible for NavSim "
                       f"({g.get('estimators_admissible')}) — scene-token / episode-cluster / "
                       f"overlapping_holdout_se intervals FAIL")
    for k in ("resample_unit", "cluster_unit"):
        if iv.get(k, want) != want:
            return False, f"interval {k}={iv.get(k)!r}, not {want!r}"
    n = iv.get("n_clusters")
    lo_ = int(g.get("min_clusters", 8))
    if not isinstance(n, int) or isinstance(n, bool) or n < lo_:
        return False, f"n_clusters={n!r} < the floor {lo_} (RG-14)"
    mx_map = g.get("max_clusters_by_protocol", {})
    if tag in mx_map:
        mx = mx_map[tag]
        if mx is None:
            return False, (f"{tag} cannot carry an interval — its log identities are "
                           f"not observable")
        if n > mx:
            return False, (f"n_clusters={n} exceeds {tag}'s {mx} logs — a unit finer "
                           f"than log_name was resampled")
    agg = iv.get("aggregation")
    agg_map = g.get("aggregation_by_protocol", {})
    if agg not in set(agg_map.values()):
        return False, f"interval aggregation {agg!r} is not a registered official one"
    if tag in agg_map and agg_map[tag] != agg:
        return False, (f"aggregation {agg!r} is not {tag}'s official one "
                       f"({agg_map[tag]!r}) — a bootstrap of the wrong aggregate")
    if int(iv.get("n_boot", 0) or 0) < int(g.get("n_boot_min", 2000)):
        return False, f"n_boot={iv.get('n_boot')!r} < {g.get('n_boot_min', 2000)}"
    lohi = ("delta", "lo", "hi") if "delta" in iv else ("lo", "hi")
    if not all(isinstance(iv.get(k), (int, float)) for k in lohi):
        return False, f"the interval block carries no numeric {lohi}"
    return True, f"PASS — {est} over {n} {want}s ({agg})"


def _admissible_interval_hiding_in(blk, g: dict, tag, want: str, _depth: int = 0):
    """The key path of an ADMISSIBLE interval nested inside a DECLINED block, or None.

    ⛔ MEASURED 2026-09-20 on E1's real navhard CV artifact: it declared
    `{status: UNAVAILABLE, reason: "a numeric interval exists in summary.json; the v2.9.0
    registry gate admits only UNAVAILABLE here until W2 updates it"}` while carrying the
    real log-cluster interval (76 clusters, [0.0825, 0.145]) under `summary_interval`. The
    gate had admitted that interval since registry 2.10.0 — the artifact was built against
    the **2.9.0 copy at HEAD**, because this stream's registry update is staged and not yet
    committed. A false refusal reads exactly like an honest one, so it is detected here and
    FAILS: the gate exists so a NavSim number carries its interval, and an artifact that HAS
    one and hides it is misreporting, not declining.
    """
    if _depth > 3 or not isinstance(blk, dict):
        return None
    for k, v in blk.items():
        if k in ("status", "reason", "n") or not isinstance(v, dict):
            continue
        if _navsim_interval_ok(v, g, tag, want)[0]:
            return k
        deep = _admissible_interval_hiding_in(v, g, tag, want, _depth + 1)
        if deep:
            return f"{k}.{deep}"
    return None


def _gate_navsim_estimator(art: dict, g: dict) -> tuple:
    want = g.get("cluster_unit", "log_name")
    fu, unit = _dig(art, "estimator.cluster_unit")
    fi, iv = _dig(art, "estimator.interval")
    _, tag = _dig(art, "protocol.navsim_protocol")
    if not fu or unit is None:
        return ABSENT, ("estimator.cluster_unit not declared — the gate requires it "
                        f"(registered unit: {want!r})")
    if isinstance(unit, dict):
        st = _status_of(unit)
        if st is None or st[0] != REFUSED:
            return ABSENT, f"{FAIL_}estimator.cluster_unit is an object without a reasoned refusal"
        why = str(unit.get("reason", "")).strip()
        if len(why.split()) < 2:
            # ⛔ MEASURED on E1's navhard artifact: `{status: UNAVAILABLE, n: 76,
            # reason: "log_name"}` — the UNIT NAME stuffed into the reason field. A single
            # token cannot explain anything, and the refusal shape then hides a unit that is
            # in fact settled and correct.
            return ABSENT, (f"{FAIL_}estimator.cluster_unit is a REFUSAL whose `reason` is the "
                            f"single token {why!r} — a unit name is not a reason. The unit is "
                            f"SETTLED: declare the string {want!r}.")
        unit_refused = True
    elif isinstance(unit, str):
        if unit != want:
            return ABSENT, (f"{FAIL_}estimator.cluster_unit={unit!r}; the registered unit is "
                            f"{want!r} (scene tokens overlap within a log — SPEC §2)")
        unit_refused = False
    else:
        return ABSENT, f"{FAIL_}estimator.cluster_unit is a {type(unit).__name__}"
    if not fi or iv is None:
        return ABSENT, "estimator.interval absent — a NavSim number with no interval block"
    if _declined(iv):
        hid = _admissible_interval_hiding_in(iv, g, tag, want)
        if hid:
            return ABSENT, (f"{FAIL_}estimator.interval declares {str(iv.get('status'))!r} while "
                            f"HOLDING an admissible interval at `estimator.interval.{hid}` — a "
                            f"FALSE REFUSAL. A log-cluster interval over >= {g.get('min_clusters', 8)} "
                            f"log_names has been ADMITTED since registry 2.10.0 (SETTLED "
                            f"2026-09-19); the pre-settlement rule that admitted only "
                            f"{{status: UNAVAILABLE}} is DEAD. Promote it to `estimator.interval`.")
        if str(iv.get("reason", "")).strip():
            return REFUSED, f"interval refused: {str(iv['reason'])[:160]} (n={iv.get('n')})"
        return ABSENT, f"{FAIL_}the interval is declined with NO reason"
    ok, detail = _navsim_interval_ok(iv, g, tag, want)
    if not ok:
        return ABSENT, FAIL_ + detail
    if unit_refused:
        return ABSENT, f"{FAIL_}a numeric interval beside a REFUSED cluster_unit is incoherent"
    return PRESENT, detail


def _evidence_ok(ev) -> bool:
    if isinstance(ev, (dict, list)):
        return len(ev) > 0
    if isinstance(ev, str):
        return bool(_EVIDENCE_FILE.search(ev))
    return False


def _gate_navsim_ego(art: dict, g: dict) -> tuple:
    f, blk = _dig(art, "protocol.ego_status_enforcement")
    if not f or blk is None:
        return ABSENT, "protocol.ego_status_enforcement absent — no mechanism, no declaration"
    if not isinstance(blk, dict):
        return ABSENT, (f"{FAIL_}a bare {type(blk).__name__} is an assertion, not enforcement "
                        f"('we did not use it' FAILS)")
    _, vo = _dig(art, "protocol.vision_only")
    if vo is True and blk.get("vision_only_claimed") is False:
        return ABSENT, (f"{FAIL_}protocol.vision_only is true while the enforcement block denies "
                        f"a vision-only claim — incoherent; a privileged or camera-free arm is "
                        f"not vision-only, and the leaderboard would print the wrong one")
    claimed = bool(blk.get("vision_only_claimed")) or vo is True
    mech = blk.get("mechanism")
    if isinstance(mech, str) and mech.strip() and _evidence_ok(blk.get("evidence")):
        return PRESENT, "PASS — mechanism + structured evidence declared"
    if claimed:
        return ABSENT, (f"{FAIL_}a vision-only claim without a mechanism AND structured evidence "
                        f"is an assertion, not enforcement")
    if str(blk.get("reason", "")).strip():
        return REFUSED, (f"declared not applicable (no vision-only claim): "
                         f"{str(blk['reason'])[:160]}")
    return ABSENT, f"{FAIL_}neither mechanism+evidence nor a reasoned not-applicable"


def _gate_navsim_modality(art: dict, g: dict) -> tuple:
    missing, bad = [], []
    for k in g.get("keys", ["protocol.sensor_set", "protocol.setting"]):
        f, v = _dig(art, k)
        if not f or v is None:
            missing.append(k)
        elif not (isinstance(v, str) and v.strip()):
            bad.append(k)
    if bad:
        return ABSENT, f"{FAIL_}{bad} must be non-empty strings — a refusal is not a label"
    if missing:
        return ABSENT, f"{missing} absent — the server records no modality, so the label is ours"
    return PRESENT, "PASS — sensor_set and setting declared"


def _gate_navsim_cross_protocol(art: dict, g: dict) -> tuple:
    f1, tag = _dig(art, "protocol.navsim_protocol")
    if not f1 or tag is None:
        return ABSENT, "protocol.navsim_protocol absent — an untagged number merges protocols"
    if tag not in g.get("closed_set", []):
        return ABSENT, f"{FAIL_}{tag!r} is not a NavSim protocol ({g.get('closed_set')})"
    f2, sha = _dig(art, "protocol.devkit_sha")
    if not f2 or sha is None:
        return ABSENT, "protocol.devkit_sha absent — the SHA is part of the column"
    if not (isinstance(sha, str) and _SHA40.match(sha)):
        return ABSENT, f"{FAIL_}devkit_sha {sha!r} is not a full 40-hex SHA"
    pins = g.get("devkit_pins", {}).get(tag)
    if pins is not None and sha not in pins:
        return ABSENT, (f"{FAIL_}devkit_sha {sha[:10]}… is not registered for {tag} "
                        f"({[p[:10] for p in pins]}) — e.g. v1 PDMS from the v2 scorer")
    fv, var = _dig(art, "benchmark.navsim.variant")
    want = "PDMS_v1" if tag.startswith("PDMS_v1") else "EPDMS_v2"
    if fv and isinstance(var, str) and not var.startswith(want):
        return ABSENT, f"{FAIL_}benchmark.navsim.variant {var!r} contradicts {tag!r}"
    return PRESENT, f"PASS — {tag} @ {sha[:10]}"


_NAVSIM_GATES = {
    "navsim.estimator_unit": _gate_navsim_estimator,
    "navsim.ego_enforcement": _gate_navsim_ego,
    "navsim.modality_label": _gate_navsim_modality,
    "navsim.cross_protocol": _gate_navsim_cross_protocol,
}


def _command_consumed(art: dict) -> bool:
    for p in ("benchmark.navsim.ego_inputs.inputs.driving_command.used",
              "protocol.inference_inputs.inputs.driving_command.used"):
        f, v = _dig(art, p)
        if f and v is True:
            return True
    return False


def check_navsim(artifact: dict, registry: dict) -> list:
    spec = registry.get("benchmarks", {}).get("navsim")
    art = _navsim_view(artifact)
    if art is None or not isinstance(spec, dict):
        return []
    rows = []
    for crit in spec.get("criteria", []):
        state, detail = classify(art, crit)
        cid = crit["id"]
        if cid == "navsim.score" and state == PRESENT:
            _, sc = _dig(art, "benchmark.navsim.score")
            if isinstance(sc, dict) and sc.get("column") not in (None, "score"):
                state, detail = ABSENT, (f"{FAIL_}score read from column {sc.get('column')!r} "
                                         f"— the column trap; read `score`, never `pdm_score`")
        if cid == "navsim.route_leak_check" and _command_consumed(art):
            _, rl = _dig(art, "benchmark.navsim.route_leak_check")
            if not (isinstance(rl, dict) and rl.get("verdict") and not _declined(rl)):
                state, detail = ABSENT, (f"{FAIL_}this arm CONSUMES driving_command but its "
                                         f"route_leak_check carries no settled verdict "
                                         f"(ROUTE_LEAK_VERDICT: {spec.get('ROUTE_LEAK_VERDICT', {}).get('verdict')})")
        rows.append(_row(cid, crit.get("label", cid), state, detail, crit.get("required", True)))
    for key in sorted(k for k in spec if k.startswith("GATE_")):
        g = spec[key]
        fn = _NAVSIM_GATES.get(g.get("id"))
        if fn is None:
            rows.append(_row(g.get("id", key), key, ABSENT,
                             f"{FAIL_}gate {g.get('id')!r} has no evaluator in criteria_check.py "
                             f"— a registered gate the machinery cannot evaluate is a note"))
            continue
        state, detail = fn(art, g)
        rows.append(_row(g["id"], f"GATE {g['id']}", state, detail,
                         bool(g.get("blocking", True))))
    return rows


def check_benchmarks(artifact: dict, registry: dict) -> list:
    """Every benchmark row for one IN-SCOPE artifact: the protocol-tag check (when the
    artifact declares a tag) and the NavSim block (when it claims NavSim)."""
    rows = []
    tag = check_protocol_tag(artifact, registry)
    if tag is not None:
        rows.append(_row("protocol.tag_registered", "protocol tag is a registered one", *tag))
    rows += check_navsim(artifact, registry)
    return rows


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
    #
    # ⚠️ FOUR RECURRENCES OF ONE CLASS, so the CLASS is fixed here rather than a
    # fifth literal. MEASURED 2026-09-05 across the banked refav1 records, which
    # disavow the estimator in four different wordings:
    #     "overlapping_holdout_se is NOT used anywhere"      -> caught by "not used"
    #     "overlapping_holdout_se is never used"             -> caught by "never used"
    #     "⛔ NOT overlapping_holdout_se, which is …"        -> MISSED
    #     "— NEVER overlapping_holdout_se"                  -> MISSED
    # The two misses share a shape the literal list cannot express: the token is
    # NEGATED DIRECTLY, with no "used" anywhere and no "is not" (the text reads
    # "which is anti-conservative"). Adding literals one at a time is what let the
    # same defect return three times, so the direct negation is matched as a
    # pattern. It cannot exonerate a live use: a leaf that really carries
    # `estimator: overlapping_holdout_se` has no negation before the token, and the
    # judgement is per-LEAF, so a disavowing sibling never covers for it.
    def _norm(s: str) -> str:
        return "".join(c if c.isalnum() else " " for c in s.lower())

    exonerating = ("deprecated", "refused", "legacy", "not used", "never used",
                   "forbidden", "must not", "no longer", "is not", "banned")
    # the token negated directly: "NOT overlapping_holdout_se", "NEVER …", "no …"
    negated = re.compile(r"\b(?:not|never|no|nor|neither|without|excluding)"
                         r"\s+overlapping\s+holdout")
    bad = []
    for path, val in hits:
        ctx = _norm(f"{path} {val}")
        if not any(e in ctx for e in exonerating) and not negated.search(ctx):
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
                        ("LEAK GUARDS", res["leak_guards"]),
                        ("BENCHMARK GATES", res.get("benchmarks", []))):
        if title == "BENCHMARK GATES" and not rows:
            continue
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
    external = [k for k, v in report.items() if v.get("scope") == EXTERNAL_ONLY]
    n = len(scoped)

    out = [f"CRITERIA CENSUS (registry v{registry.get('version')})", "=" * 62,
           f"{total} artifacts read -> {n} in scope (driving evals), "
           f"{len(external)} EXTERNAL-ONLY (nuScenes OL: never a criterion), "
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

    bench = {}
    for res in scoped.values():
        for r in res.get("benchmarks", []):
            d = bench.setdefault(r["id"], {PRESENT: 0, REFUSED: 0, PARTIAL: 0, ABSENT: 0})
            d[r["state"]] += 1
    if bench:
        out.append("BENCHMARK GATES (artifacts that claim a benchmark or carry a protocol tag)")
        for cid, d in sorted(bench.items()):
            out.append(f"   {cid:<52} {d[PRESENT]} pass, {d[REFUSED]} refused, "
                       f"{d[ABSENT]} FAIL/MISSING")
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
