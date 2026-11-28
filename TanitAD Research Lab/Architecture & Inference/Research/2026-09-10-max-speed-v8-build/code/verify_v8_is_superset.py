"""⛔⛔ THE SINGLE-LEVER PRECONDITION NOBODY WOULD THINK TO CHECK.

⭐ THE PROBLEM, STATED PLAINLY. `--max-speed-input` needs the v8 blob. If the ON
arm reads v8 and the OFF arm reads v7.2, the two namespaces differ in **THREE**
parsed keys — `max_speed_input`, `v7_labels`, `eval_labels` — and the panel is
**not single-lever**. It would be the `--v2` conflation again: *ten levers on two
axes, result non-attributable.*

⇒ **BOTH arms must read the v8 blob**, and the OFF arm simply does not pass the
flag. ⛔ That is only sound if v8 is a **PURE SUPERSET** of v7.2 for every field
the OFF path reads — otherwise swapping the blob silently changes the OFF arm's
tactical/nav supervision and the "control" is a different experiment.

⛔ THIS IS ASSERTED, NOT ASSUMED. Every record is compared key by key against its
v7.2 original:

  * the clip set and its ORDER must be identical (parity is sacred);
  * every key present in v7.2 must be present in v8 and **byte-equal**, EXCEPT
    the two this build deliberately changed (`release`, `_provenance`);
  * `_provenance` must be a superset — every v7.2 sub-key byte-equal, plus
    exactly one addition (`speed_max_input`);
  * `release` must change v7.2 -> v8.0 and nothing else may;
  * the ONLY new top-level key may be `speed_max_input`;
  * `schema_version` and `vocab` must be UNCHANGED, or `load_v7_labels` refuses.

⚠️ NOTE THE DIRECTION OF THE TEST. *"v8 differs from v7.2"* is not the question —
of course it does, that is the point. The question is whether it differs **only**
where this build intended, which is the descendant check, not a difference check.
A revert also differs.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

#: ⛔ LITERALS. The exact set of top-level keys this build is permitted to touch.
ADDED_KEYS = {"speed_max_input"}
CHANGED_KEYS = {"release", "_provenance"}
FROZEN_KEYS = {"schema_version", "vocab"}       # load_v7_labels refuses a change
ADDED_PROVENANCE_KEYS = {"speed_max_input"}


def _load(p: Path) -> list[dict]:
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def verify(old_p: Path, new_p: Path) -> dict:
    old, new = _load(old_p), _load(new_p)
    n = len(old)
    problems: list[str] = []

    if len(new) != n:
        problems.append(f"record count {len(old)} -> {len(new)}")
    if [r.get("clip_id") for r in old] != [r.get("clip_id") for r in new]:
        problems.append("the clip sequence changed (parity violation)")

    n_equal_keys = 0
    n_added = n_release = n_prov = n_frozen_ok = 0
    unexpected_added: set[str] = set()
    unexpected_changed: set[str] = set()
    prov_unexpected: set[str] = set()

    for a, b in zip(old, new):
        added = set(b) - set(a)
        removed = set(a) - set(b)
        if removed:
            problems.append(f"{a.get('clip_id')}: keys REMOVED {sorted(removed)}")
        unexpected_added |= (added - ADDED_KEYS)
        n_added += int(added == ADDED_KEYS)

        for k in FROZEN_KEYS:
            n_frozen_ok += int(a.get(k) == b.get(k))

        for k in set(a):
            if k in CHANGED_KEYS:
                continue
            if a[k] == b.get(k):
                n_equal_keys += 1
            else:
                unexpected_changed.add(k)

        n_release += int(a.get("release") == "v7.2" and b.get("release") == "v8.0")

        pa = a.get("_provenance") or {}
        pb = b.get("_provenance") or {}
        prov_added = set(pb) - set(pa)
        prov_unexpected |= (prov_added - ADDED_PROVENANCE_KEYS)
        same_sub = all(pa[k] == pb.get(k) for k in pa)
        n_prov += int(prov_added == ADDED_PROVENANCE_KEYS and same_sub)

    if unexpected_added:
        problems.append(f"unexpected NEW top-level keys: {sorted(unexpected_added)}")
    if unexpected_changed:
        problems.append(f"unexpected CHANGED keys: {sorted(unexpected_changed)}")
    if prov_unexpected:
        problems.append(f"unexpected _provenance keys: {sorted(prov_unexpected)}")

    return {
        "old": str(old_p), "new": str(new_p), "n_records": n,
        # ⭐ the same-breath VACUITY control: a comparison over records with no
        # shared keys would report "nothing changed" and mean nothing.
        "CONTROL_shared_keys_per_record": len(set(old[0]) & set(new[0])) if n else 0,
        "CONTROL_v72_keys_per_record": len(old[0]) if n else 0,
        "n_records_adding_exactly_speed_max_input": n_added,
        "n_records_release_v72_to_v80": n_release,
        "n_records_provenance_is_a_superset": n_prov,
        "n_frozen_field_checks_passed": n_frozen_ok,
        "n_frozen_field_checks_expected": 2 * n,
        "n_untouched_key_comparisons_EQUAL": n_equal_keys,
        "problems": problems,
        "VERDICT_pure_superset": not problems and n_added == n
                                 and n_release == n and n_prov == n
                                 and n_frozen_ok == 2 * n,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="OLD=NEW pairs, one per split")
    ap.add_argument("--report")
    a = ap.parse_args(argv)
    rep = {"instrument": "verify_v8_is_superset.py",
           "why": ("both arms of the E16 panel must read the SAME blob, or the "
                   "panel is not single-lever"),
           "splits": {}}
    for pair in a.pairs:
        old, new = pair.split("=", 1)
        rep["splits"][Path(new).name] = verify(Path(old), Path(new))
    txt = json.dumps(rep, indent=1)
    print(txt)
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(txt, encoding="utf-8")
    ok = all(s["VERDICT_pure_superset"] for s in rep["splits"].values())
    print(f"\nOVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
