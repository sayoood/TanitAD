"""Mint the 2026-09-16 absence-is-negative SIDECAR. ⛔ Never touches the blob.

The PI ruled that a ``vlm-cot`` token the caption did not mention is FALSE, not
unknown. This writes that ruling down as a file: one row per clip, one bit per
CoT token, plus a stamped policy record carrying the ruling verbatim, its date,
the token list, the rule, the source blob's md5 and the census before and after.

⛔⛔ THE LABEL BLOB IS TRACKED AND HASHED AND IS NOT MODIFIED. The policy lives
BESIDE the corpus, not inside it, for the reason ``v7_labels`` already learned
the hard way: six copies of that blob exist under three roots with differing
md5s, and a policy baked into one of them would be invisible in the other five.
A sidecar names the md5 it was built over, and the loader REFUSES to apply it to
any other — so the corpus and the policy either travel together or the run
fails.

⛔ CLIP IDS ARE WRITTEN AS ``sha256(clip_id)[:12]``, never in the clear: this
artifact is committed. The algorithm is DECLARED in the meta (join_meta's M18
rule -- a digest that does not say what it is taken over is a number, not a
verification), and the builder REFUSES on a digest collision rather than letting
two clips share a row.

ASCII only in the printed output: this box is cp1252.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

from tanitad.data.v7_labels import (COT_ABSENCE_POLICY_ID, COT_ABSENCE_PRECEDENT,
                                    COT_ABSENCE_RULING, COT_ABSENCE_RULING_DATE,
                                    COT_ABSENCE_RULING_WIDENED,
                                    COT_SIDECAR_DIGEST_ALGO, COT_SIDECAR_SCHEMA,
                                    TAC_GOAL_TOKENS, clip_sha12,
                                    cot_backed_tokens, goal_supervision_census,
                                    load_cot_negative_sidecar, load_v7_labels,
                                    assert_sidecar_matches_presence)

RULE = ("For every token in `tokens`, a clip is POSITIVE when the token is in "
        "its `g_tac.goals` and NEGATIVE otherwise. There is no ignore state "
        "for these tokens under this policy. Tokens NOT in `tokens` are "
        "untouched and keep the provenance-derived policy, so the geometry "
        "tokens are unaffected.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fn-measurements", default=None,
                    help="the false-negative analysis json; its headline rates "
                         "are carried in the meta so the sidecar never travels "
                         "without the cost of the ruling attached.")
    a = ap.parse_args()

    labels, man = load_v7_labels(a.labels)
    tokens = tuple(t for t in TAC_GOAL_TOKENS if t in cot_backed_tokens())
    if not tokens:
        raise SystemExit("[build] ⛔ this blob emits no vlm-cot token — there is "
                         "nothing for this policy to decide, and writing an "
                         "empty sidecar would be a permission to do nothing "
                         "wearing the name of a ruling.")
    print(f"[build] blob md5={man.md5} n={man.n_records} "
          f"cot tokens={len(tokens)}: {list(tokens)}", flush=True)

    before = goal_supervision_census(labels)

    rows: dict[str, str] = {}
    for lb in labels:
        d = clip_sha12(lb.clip_id)
        if d in rows:
            raise SystemExit(
                f"[build] ⛔ sha12 collision on {d} — two clips would share one "
                f"row and one of them would get the other's negatives. Widen "
                f"the digest rather than dropping a clip.")
        g = lb.tac_goals or frozenset()
        rows[d] = "".join("1" if t in g else "0" for t in tokens)

    meta = {
        "policy": COT_ABSENCE_POLICY_ID,
        "ruling": COT_ABSENCE_RULING,
        "ruling_widened": COT_ABSENCE_RULING_WIDENED,
        "ruling_date": COT_ABSENCE_RULING_DATE,
        "ruled_by": "PI",
        "precedent_overridden": COT_ABSENCE_PRECEDENT,
        "rule": RULE,
        "tokens": list(tokens),
        "source_blob": Path(a.labels).name,
        "source_blob_md5": man.md5,
        "source_blob_records": man.n_records,
        "source_blob_schema": man.schema_version,
        "source_blob_vocab": man.vocab,
        "digest_algorithm": COT_SIDECAR_DIGEST_ALGO,
        "digest_scope": "the clip_id string, utf-8, truncated to 12 hex chars",
        "_evidence_class": "MEASURED (ours; the bits are a transform of the "
                           "blob's own g_tac.goals, no new labelling)",
        "counts_before": {t: {"pos": before[t]["pos"], "neg": before[t]["neg"],
                              "ignored": before[t]["ignored"]}
                          for t in TAC_GOAL_TOKENS},
    }
    doc = {"schema": COT_SIDECAR_SCHEMA, "meta": meta, "clips": rows}

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # write once WITHOUT counts_after, load it, then compute the after-census
    # THROUGH the loader -- so the numbers in the meta are the numbers the
    # consumer will actually get, not a second implementation of them.
    with gzip.open(out, "wt", encoding="utf-8") as fh:
        json.dump(doc, fh)
    sc, stamped = load_cot_negative_sidecar(out, man)
    assert_sidecar_matches_presence(labels, sc)
    after = goal_supervision_census(labels, negatives="cot-absence-negative",
                                    sidecar=sc)
    meta["counts_after"] = {t: {"pos": after[t]["pos"], "neg": after[t]["neg"],
                                "ignored": after[t]["ignored"]}
                            for t in TAC_GOAL_TOKENS}
    if a.fn_measurements:
        fn = json.loads(Path(a.fn_measurements).read_text(encoding="utf-8"))
        meta["false_negative_measurement"] = {
            "_read": "the measured cost of this ruling; see "
                     "TanitAD Research Lab/Data Engineering/Research/"
                     "2026-09-16-flywheel-negatives/RESULT.md",
            "window_s": fn.get("window_s"),
            "traffic_light_box_channel": fn.get("traffic_light_box_channel"),
            "pipeline_control": fn.get("pipeline_control"),
        }
    with gzip.open(out, "wt", encoding="utf-8") as fh:
        json.dump(doc, fh)

    # and re-read the FINAL bytes, so what is reported is what was written
    sc2, _ = load_cot_negative_sidecar(out, man)
    assert_sidecar_matches_presence(labels, sc2)
    print(f"[build] wrote {out} md5={sc2.md5} "
          f"({out.stat().st_size} bytes, {len(rows)} clips)", flush=True)
    print(f"{'token':30s} {'pos':>6s} {'neg->':>7s} {'neg':>6s} "
          f"{'ign->':>7s} {'ign':>5s}")
    for t in TAC_GOAL_TOKENS:
        b, af = before[t], after[t]
        print(f"{t:30s} {af['pos']:6d} {b['neg']:7d} {af['neg']:6d} "
              f"{b['ignored']:7d} {af['ignored']:5d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
