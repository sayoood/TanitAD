"""TACTICAL LABEL CENSUS -- every v7.x tactical field x whether it reaches training.

⛔ WHY: on 2026-09-06 a GT traffic-light colour was found present in the label
blob (779/4,572 records) and invisible to every supervised target. A RED<->GREEN
mutation over all 739 coloured entries moved 0 supervised records while a
same-breath control mutation of ``a_tac.lon`` moved 779. The gap was structural,
not a zero weight -- there is no tactical-GOAL head at all.

This script makes that auditable for EVERY tactical field rather than for the one
field somebody happened to ask about. It answers, per field:

  IN BLOB    -- present in the banked record, as <n>/<total>
  SURFACED   -- does ``v7_labels.load_v7_labels`` expose it on ``V7Label``, and where
  SUPERVISED -- is it (or its vocabulary) a class of one of ``v7_labels.HEADS``

⚠️ SURFACED-as-`audit` is NOT supervision. ``v7_labels`` declares ``audit``
*"audit-only, NEVER a training input"*, so an audit-only field is unreachable by
any loss no matter what weight a config sets.

Usage:
    python stack/scripts/tactical_label_census.py <s2_labels_v7.x.jsonl[.gz]>

Output is ASCII-only (the dev box is cp1252 and a non-ASCII print() has already
truncated one banked artifact into a file that read complete).
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import gzip
import json
import pathlib
import sys


def _load(path: pathlib.Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _flat_tactical_fields(recs: list[dict]) -> dict[str, int]:
    """Every dotted tactical field path -> how many records carry it."""
    n = collections.Counter()
    for r in recs:
        a = r.get("a_tac") or {}
        for k in a:
            n["a_tac.%s" % k] += 1
        g = r.get("g_tac") or {}
        for k in g:
            n["g_tac.%s" % k] += 1
        for tok, sub in (g.get("goals") or {}).items():
            n["g_tac.goals.<TOKEN>"] += 0          # ensure the row exists
            n["g_tac.goals.%s" % tok] += 1
            if isinstance(sub, dict):
                for kk in sub:
                    n["g_tac.goals.<TOKEN>.%s" % kk] += 1
    return dict(n)


def _surface_map() -> dict[str, str]:
    """Where load_v7_labels puts each tactical field. Derived from the module,
    not from memory of it: the V7Label field names are read off the dataclass."""
    from tanitad.data import v7_labels as V7L
    fields = {f.name for f in dataclasses.fields(V7L.V7Label)}
    m = {
        "a_tac.lat": "V7Label.tac_lat" if "tac_lat" in fields else "DROPPED",
        "a_tac.lon": "V7Label.tac_lon" if "tac_lon" in fields else "DROPPED",
        "a_tac.lat_args": "V7Label.audit['a_tac_args']",
        "a_tac.lon_args": "V7Label.audit['a_tac_args']",
        "a_tac.truncated": "V7Label.audit['a_tac_truncated']",
        "a_tac.serves_goals": "V7Label.audit['serves_goals']",
        "g_tac.anchor": "V7Label.tac_anchor" if "tac_anchor" in fields else "DROPPED",
        "g_tac.violations": "V7Label.audit['g_tac_violations']",
        "g_tac.goals": "V7Label.audit['goal_flags']  (AUDIT ONLY)",
    }
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("blob", type=pathlib.Path)
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args(argv)

    from tanitad.data import v7_labels as V7L
    from tanitad.models import vocab_v7 as V7

    recs = _load(a.blob)
    total = len(recs)
    fields = _flat_tactical_fields(recs)
    surf = _surface_map()

    supervised_classes = set()
    for toks in V7L.HEADS.values():
        supervised_classes |= set(toks)

    print("=" * 78)
    print("TACTICAL LABEL CENSUS")
    print("=" * 78)
    print("blob        : %s" % a.blob)
    print("n_records   : %d" % total)
    schemas = collections.Counter(r.get("schema_version") for r in recs)
    vocabs = collections.Counter(r.get("vocab") for r in recs)
    print("schema      : %s" % dict(schemas))
    print("vocab       : %s" % dict(vocabs))
    print("supervised heads: %s" % {k: len(v) for k, v in V7L.HEADS.items()})
    print()

    print("-" * 78)
    print("%-38s %-11s %-26s" % ("TACTICAL FIELD", "IN BLOB", "SURFACED BY LOADER"))
    print("-" * 78)
    struct = [k for k in sorted(fields) if not k.startswith("g_tac.goals.")]
    for k in struct:
        where = surf.get(k, "-- not surfaced --")
        print("%-38s %-11s %-26s" % (k, "%d/%d" % (fields[k], total), where))
    print()

    print("-" * 78)
    print("%-34s %-11s %-9s %-18s" % ("TACTICAL GOAL TOKEN", "IN BLOB",
                                      "IN VOCAB", "SUPERVISED?"))
    print("-" * 78)
    goal_rows = sorted(((k.split(".", 2)[2], v) for k, v in fields.items()
                        if k.startswith("g_tac.goals.")
                        and "<TOKEN>" not in k),
                       key=lambda kv: -kv[1])
    for tok, n in goal_rows:
        in_vocab = "yes" if tok in V7.TACTICAL_GOAL_TOKENS_V7 else "NOT IN v7"
        # ⚠️ A GOAL TOKEN IS NEVER SUPERVISED AS A GOAL -- no head is sized on
        # TACTICAL_GOAL_TOKENS_V7. A few goal names (TURN_L, LANE_CHANGE_R, ...)
        # also exist as tac_lat ACTION classes. Printing plain "yes" there would
        # be true of the STRING and false of the GOAL -- the reader would
        # conclude the goal is supervised. Say which it is.
        sup = ("NO (name also a tac_lat ACTION class)"
               if tok in supervised_classes else "NO")
        print("%-34s %-11s %-9s %-18s"
              % (tok, "%d/%d" % (n, total), in_vocab, sup))
    print()
    print("  NOTE read the last column carefully: NO tactical GOAL is supervised as")
    print("    a goal. Rows marked '(name also a tac_lat ACTION class)' share a")
    print("    STRING with the lateral-action vocabulary; the GOAL is still")
    print("    unsupervised. No head is sized on TACTICAL_GOAL_TOKENS_V7.")
    print()

    tl = [(t, n) for t, n in goal_rows if "TRAFFIC_LIGHT" in t]
    tl_any = sum(1 for r in recs
                 if any("TRAFFIC_LIGHT" in k
                        for k in ((r.get("g_tac") or {}).get("goals") or {})))
    print("-" * 78)
    print("TRAFFIC LIGHT (the worked example)")
    print("-" * 78)
    print("  any TRAFFIC_LIGHT_* goal : %d/%d records" % (tl_any, total))
    for t, n in sorted(tl):
        print("    %-32s %d/%d" % (t, n, total))
    states = collections.Counter()
    prov = collections.Counter()
    disputed = collections.Counter()
    tbasis = collections.Counter()
    for r in recs:
        for tok, sub in ((r.get("g_tac") or {}).get("goals") or {}).items():
            if "TRAFFIC_LIGHT" in tok and isinstance(sub, dict):
                states[sub.get("state")] += 1
                prov[sub.get("provenance")] += 1
                disputed[sub.get("disputed")] += 1
                tbasis[sub.get("time_basis")] += 1
    print("  state       : %s" % dict(states))
    print("  provenance  : %s" % dict(prov))
    print("  disputed    : %s" % dict(disputed))
    print("  time_basis  : %s" % dict(tbasis))
    print("  SUPERVISED  : NO -- no head in v7_labels.HEADS carries these classes")
    print()

    # ---- the band ---------------------------------------------------------
    print("-" * 78)
    print("BAND COVERAGE (why the GT is absent at most scored frames)")
    print("-" * 78)
    t0 = collections.Counter(r.get("t0_s") for r in recs)
    clips = collections.Counter(r["clip_id"] for r in recs)
    print("  distinct clips %d ; max records per clip %d"
          % (len(clips), max(clips.values())))
    print("  t0_s          : %s" % dict(t0.most_common(3)))
    cov = tot = cov_av = tot_av = 0.0
    for r in recs:
        h = r.get("horizon") or {}
        span = float(h.get("recording_span_s") or 0.0)
        av = float(h.get("available_s") or 0.0)
        tb = (r.get("bands") or {}).get("tactical_s") or [2.0, 6.0]
        width = max(0.0, tb[1] - tb[0])
        if span > 0:
            cov += width
            tot += span
        if av > 0:
            cov_av += min(width, av)
            tot_av += av
    if tot:
        print("  tactical band vs FULL RECORDING : %.1f s / %.1f s = %.2f %%"
              % (cov, tot, 100.0 * cov / tot))
    if tot_av:
        print("  tactical band vs USABLE HORIZON : %.1f s / %.1f s = %.2f %%"
              % (cov_av, tot_av, 100.0 * cov_av / tot_av))
    print("  => a scored frame outside that band has NO v7 tactical GT.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
