"""P5 - the ``--agent-cls-weight`` corpus-line guard, tested by MUTATION.

The guard (``agent_slots.load_cls_class_weight:339-343``) refuses when the
artifact's declared ``corpus_line`` differs from ``expect_corpus_line``. The
expectation is supplied by ``refc_v3_train.CLS_WEIGHT_CHOICES:4966-4970``::

    "train2400": (CLS_WEIGHTS_TRAIN2400, CORPUS_LINE_PARITY)
    "b1":        (CLS_WEIGHTS_B1,        CORPUS_LINE_B1)

so the expectation is selected by the SAME key that selects the file. Three arms
decide whether that guard can go red on the operator error it names:

  A  b1 / b1              -> must LOAD
  B  train2400/train2400  -> must LOAD.  ** This is the arm that matters.**
     Nothing in this call knows which corpus the ARM trains on, so a refcv6 B1
     run launched with `--agent-cls-weight train2400` reaches the loss with the
     PARITY vector and no refusal anywhere.
  C  b1 artifact / parity expectation (the mutation) -> must REFUSE, proving the
     guard is not simply inert.

Arm C is the deliberate-regression the advisory's class F asks for: without it,
arm A passing proves only that a string equals itself.

It also prints the two vectors' element-wise ratio, so "the vectors are not
interchangeable" is a number rather than an assertion.

Run (CPU, read-only):
  PYTHONPATH=D:/Projects/TanitAD/stack python p5_cls_weight_guard_regression.py
"""
from __future__ import annotations

import json
import pathlib
import sys

from tanitad.models import agent_slots as A

HERE = pathlib.Path(__file__).resolve().parents[1]
OUT = HERE / "raw" / "p5_cls_weight_guard.json"


def _try(name, expect):
    try:
        vec, st = A.load_cls_class_weight(name, expect_corpus_line=expect)
        return {"outcome": "LOADED", "corpus_line": st["corpus_line"],
                "digest": st["digest"],
                "imbalance": st.get("imbalance_majority_to_rarest"),
                "weights": st["weights"],
                "counts": st.get("counts"),
                "out_of_vocabulary": st.get("out_of_vocabulary")}
    except SystemExit as e:
        # ASCII only: a decorative character in a printed string raised
        # UnicodeEncodeError on this cp1252 console AFTER every arm had run --
        # the advisory's class-F item 5, reproduced in this very probe on its
        # first run. The message is kept, transliterated.
        return {"outcome": "REFUSED",
                "message": str(e).encode("ascii", "replace").decode()[:400]}


def main() -> int:
    res = {"_evidence_class": "MEASURED (ours; this file + raw/p5_cls_weight_guard.json)",
           "guard": "agent_slots.load_cls_class_weight:339-343",
           "table": "refc_v3_train.CLS_WEIGHT_CHOICES:4966-4970",
           "arms": {}}
    res["arms"]["A_b1_with_b1_expectation"] = _try(
        A.CLS_WEIGHTS_B1, A.CORPUS_LINE_B1)
    res["arms"]["B_train2400_with_its_own_expectation_THE_OPERATOR_ERROR"] = _try(
        A.CLS_WEIGHTS_TRAIN2400, A.CORPUS_LINE_PARITY)
    res["arms"]["C_MUTATION_b1_artifact_parity_expectation"] = _try(
        A.CLS_WEIGHTS_B1, A.CORPUS_LINE_PARITY)

    a = res["arms"]["A_b1_with_b1_expectation"]
    b = res["arms"]["B_train2400_with_its_own_expectation_THE_OPERATOR_ERROR"]
    c = res["arms"]["C_MUTATION_b1_artifact_parity_expectation"]
    ratios = None
    if a["outcome"] == "LOADED" and b["outcome"] == "LOADED":
        ratios = {k: round(b["weights"][k] / a["weights"][k], 4)
                  for k in a["weights"]}
    res["vector_ratio_train2400_over_b1"] = ratios
    res["max_abs_log_ratio_class"] = (
        max(ratios.items(), key=lambda kv: abs(kv[1] - 1.0)) if ratios else None)
    res["verdict"] = {
        "guard_can_go_red": c["outcome"] == "REFUSED",
        "guard_blocks_the_operator_error": b["outcome"] == "REFUSED",
        "reading": "arm C REFUSED and arm B LOADED means the guard is live but "
                   "CANNOT see the error it names: the expectation is selected "
                   "by the same flag value that selects the file, so a B1 arm "
                   "launched with --agent-cls-weight train2400 trains on the "
                   "PARITY vector with no refusal. The arm's own corpus is "
                   "never an input to this check.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "arms"}, indent=1))
    for k, v in res["arms"].items():
        print(k, "->", v["outcome"],
              v.get("corpus_line", v.get("message", ""))[:120])
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
