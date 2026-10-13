#!/usr/bin/env python3
"""A label dir that is NAMED but loads nothing must REFUSE (exit 4), never mask silently -- scorer_finetune.py, after the held-out index.

Why (S1 input audit 2026-10-13, raw/2026-10-13-aux-s1-data-audit/S1_DATA_AUDIT.md): with `--teacher-lane-train <missing dir>` train.py's bank still reports
n_cols 7 and the train-side lane loss is fully masked with no error; with `--require-pdm --pdm-labels-heldout <missing dir>` the held-out index is empty and every
evaluation reads n = 0. Both are "built, tested, unreachable from its caller" (CLAUDE.md): the run looks healthy and tests nothing.

Runs the REAL main() on CPU through test_rl_epo.smoke (stub model + stub banks), with banks that mimic the real failure: the dir is named, the bank loads 0 rows.
  GREEN controls: the unmodified stubs preflight with rc 0 (the harness works and the guard does not fire on loaded labels).
  RED cases     : train teacher lane empty (n_cols still 7), held-out PDM empty, held-out PDM loaded but disjoint from the sets (ho_idx empty) -> rc 4 + REFUSED.
  MUTATION      : the guard deleted from the source -> the empty-teacher-lane case no longer refuses (proves this test can see the guard's absence).
Prints ZZTEST_PASS <n>.
"""
from __future__ import annotations

import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_rl_epo as TR  # noqa: E402

SRC = open(os.path.join(HERE, "scorer_finetune.py"), encoding="utf-8").read()
GUARD = "    if empty:\n        print(f\"REFUSED: {' '.join(empty)} named but loaded 0 labels"
BASE_BANK = TR.StubBank


class TrainTlaneEmpty(BASE_BANK):
    """the real train.py behaviour for a missing teacher-lane dir: n_cols stays 7, nothing loaded"""
    def __init__(self, path, *a, **k):
        super().__init__(path, *a, **k)
        if "train" in path and k.get("teacher_lane_labels"):
            self.teacher_lane, self.n_tlane = {}, 0


class HeldPdmEmpty(BASE_BANK):
    def __init__(self, path, *a, **k):
        super().__init__(path, *a, **k)
        if "train" not in path:
            self.pdm_keys, self.n_pdm = set(), 0


class HeldPdmDisjoint(BASE_BANK):
    """labels load (n_pdm > 0) but none matches a held-out set: --require-pdm leaves ho_idx empty"""
    def __init__(self, path, *a, **k):
        super().__init__(path, *a, **k)
        if "train" not in path:
            self.pdm_keys, self.n_pdm = {("nolog", "notok", -1, 0)}, 1


def run(bank, src=SRC, native7=True, tag="x"):
    TR.StubBank = bank
    try:
        with tempfile.TemporaryDirectory() as td:
            return TR.smoke(src, td, epo=False, native7=native7, preflight_first=True, tag=tag)
    finally:
        TR.StubBank = BASE_BANK


def main() -> int:
    assert SRC.count(GUARD) == 1, "the guard snippet must occur exactly once (else the mutation below is vacuous)"
    checks = []

    def t(name, ok):
        checks.append((name, bool(ok)))
        print(f"{'PASS' if ok else 'FAIL'}  {name}")

    g7 = run(BASE_BANK, native7=True, tag="g7")
    t("green/lane_head_with_loaded_labels_preflights_rc0", g7.get("pre") == 0 and "REFUSED" not in g7.get("out_pre", ""))
    g6 = run(BASE_BANK, native7=False, tag="g6")
    t("green/no_lane_flags_preflights_rc0", g6.get("pre") == 0 and "REFUSED" not in g6.get("out_pre", ""))

    r = run(TrainTlaneEmpty, tag="r_tl")
    t("red/train_teacher_lane_named_but_empty_refuses", r.get("pre") == 4 and "REFUSED: --teacher-lane-train named but loaded 0 labels" in r.get("out_pre", "")
      and r.get("pass_json") is None)
    r = run(HeldPdmEmpty, native7=False, tag="r_pdm")
    t("red/heldout_pdm_named_but_empty_refuses", r.get("pre") == 4 and "REFUSED: --pdm-labels-heldout named but loaded 0 labels" in r.get("out_pre", ""))
    r = run(HeldPdmDisjoint, native7=False, tag="r_dis")
    t("red/heldout_index_empty_refuses", r.get("pre") == 4 and "held-out sets survive the filters" in r.get("out_pre", ""))

    msrc = SRC.replace(GUARD, "    if False:\n        print(f\"REFUSED: {' '.join(empty)} named but loaded 0 labels")
    m = run(TrainTlaneEmpty, src=msrc, tag="m_tl")
    t("mutation/guard_removed_the_empty_teacher_lane_is_NOT_refused", m.get("pre") != 4 and "named but loaded 0 labels" not in m.get("out_pre", ""))

    bad = [n for n, ok in checks if not ok]
    if bad:
        print("ZZTEST_FAIL", len(bad), bad)
        return 1
    print(f"ZZTEST_PASS {len(checks)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
