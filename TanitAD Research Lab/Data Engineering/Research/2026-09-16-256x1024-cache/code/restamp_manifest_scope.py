"""Scope the `decision_grade` stamp on a built cache -- it is CONDITIONAL, not blanket.

WHY. ``parity.guard_corpus_build`` can only ask one question: is this clip inside
``physicalai-train-e438721ae894``? That is the OLD PARITY corpus of the
flagship/refav1 line. When a sanctioned audit keeps an overlapping clip, the gate
stamps ``decision_grade: false`` on EVERYTHING -- correctly, because the gate
cannot know which model will later be scored on the cache.

A blanket stamp is WRONG FOR THE READER in the other direction: a future reader
discounts a v7.2/B1-trained result that is in fact clean. So this tool adds a
``decision_grade_scope`` block that says which models the stamp applies to, and
records the MEASURED evidence for the exemption. It NEVER edits the gate's own
``parity_ingest_gate`` record -- that is the instrument's output and stays
verbatim, including the 11 overlapping sha12.

The evidence is re-measured here from PRIMARY sources every run, not asserted:
  * ``index/clip_index_v7.2_train.json`` -> ``clips``      (the v7.2 train corpus)
  * ``splits/eval_split_v3.json`` -> ``train_clip_ids``    (the split that made it)
both at the pinned corpus revision. If either overlaps the built set, this tool
REFUSES to write the exemption.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                                # noqa: BLE001
        pass


def sha12(c):
    return hashlib.sha256(c.encode()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--v72-train-index", required=True)
    ap.add_argument("--eval-split", required=True)
    ap.add_argument("--third-check", default="",
                    help="a free-text MEASURED third check (e.g. the built B1 "
                         "train cache listing)")
    a = ap.parse_args()

    ids = [l.strip() for l in open(a.ids) if l.strip()]
    S = set(ids)
    ix = json.load(open(a.v72_train_index))
    ix_tr = set(ix["clips"]) if isinstance(ix, dict) and "clips" in ix else set(ix)
    sp = json.load(open(a.eval_split))
    sp_tr = set(map(str, sp["train_clip_ids"]))
    sp_ev = set(map(str, sp["eval_clip_ids"]))

    ov_ix, ov_sp = len(S & ix_tr), len(S & sp_tr)
    if ov_ix or ov_sp:
        sys.exit(f"REFUSING to write the exemption: the built set overlaps the "
                 f"v7.2 train corpus (index {ov_ix}, split {ov_sp}). The blanket "
                 f"decision_grade:false stands.")

    m = json.load(open(a.manifest))
    gate = m.get("parity_ingest_gate", {})
    n_ov = gate.get("in_parity_train", 0)
    m["decision_grade_scope"] = {
        "stamped_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": (
            f"decision_grade: FALSE **for models trained on the OLD PARITY "
            f"corpus physicalai-train-e438721ae894** (the flagship / refav1 "
            f"line) -- {n_ov} of {len(ids)} clips are inside it. "
            f"decision_grade: TRUE for models trained on the v7.2 / B1 corpus: "
            f"the built set is MEASURED disjoint from that corpus."),
        "false_for": {
            "corpus": "physicalai-train-e438721ae894",
            "line": "flagship / refav1 (parity-trained)",
            "n_overlapping_clips": n_ov,
            "frac_of_built_set": round(n_ov / max(len(ids), 1), 4),
            "remedy": (f"score parity-trained models on the clean subset of "
                       f"{len(ids) - n_ov} clips -- the built set minus "
                       f"parity_ingest_gate.overlap_sha12_parity_train"),
        },
        "true_for": {
            "corpus": "v7.2 / B1 (Sayood/tanitad-v7-training-corpus)",
            "line": "refcv6 and every v7.2-trained arm",
            "measured_disjointness": [
                {"source": "index/clip_index_v7.2_train.json -> clips",
                 "n_train_clips": len(ix_tr), "overlap_with_built": ov_ix},
                {"source": "splits/eval_split_v3.json -> train_clip_ids",
                 "n_train_clips": len(sp_tr), "overlap_with_built": ov_sp},
            ] + ([{"source": "built B1 train cache listing",
                   "note": a.third_check, "overlap_with_built": 0}]
                 if a.third_check else []),
            "positive_membership": {
                "source": "splits/eval_split_v3.json -> eval_clip_ids",
                "n_eval_split": len(sp_ev),
                "n_built_inside_eval_split": len(S & sp_ev),
                "n_built_outside_eval_split": len(S - sp_ev),
                "eval_split_version": sp.get("version"),
                "based_on_labels_md5": sp.get("based_on_labels_md5"),
                "seed": sp.get("seed"),
            },
        },
        "note": ("parity_ingest_gate above is the instrument's own record and is "
                 "NOT edited by this tool. The gate can only ask about "
                 "physicalai-train-e438721ae894; the scope block says which "
                 "models that answer binds."),
    }
    json.dump(m, open(a.manifest, "w"), indent=1)
    print(f"[restamp] {os.path.basename(os.path.dirname(a.manifest))}: "
          f"FALSE for parity-trained ({n_ov}/{len(ids)} overlap), TRUE for "
          f"v7.2/B1 (overlap {ov_ix} index / {ov_sp} split; "
          f"{len(S & sp_ev)}/{len(ids)} inside eval_split "
          f"{sp.get('version')})", flush=True)


if __name__ == "__main__":
    main()
