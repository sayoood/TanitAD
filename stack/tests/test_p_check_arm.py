"""The per-arm gate (`stack/scripts/p_check_arm.py`) — the thing that decides whether an ~11 h
arm counts, and what makes `p_runner` resumable.

⛔ It had NO tests until 2026-09-20, which is the wrong state for a checker that stands between a
finished arm and `move_aside`. Every expectation here is a literal; the fixtures are hand-written
metrics files so the answers are known in advance.

⛔ The case that motivated the file: an UNPARSEABLE line must be COUNTED and must REFUSE, never
skipped. MEASURED in the pinned tree: every metrics write is a COMPLETE line followed immediately by `log.flush()` (`refc_v3_train.py` :7180-7181, and four sibling sites 7093/7098, 7261/7263, 7273/7274, 7324/7328; the file is opened once in append mode at :6958). ⇒ A hard kill therefore loses the final ROW, not half of one: killed between rows, or between `write()` and `flush()`, the file PARSES PERFECTLY and is simply SHORT. The only partial-line window is inside the flush syscall itself. ⛔ SO FOR A SIGKILL OR A cgroup-OOM — this box's documented trainer death — `n_unparseable_metrics_lines` is STRUCTURALLY BLIND, and the STEP-COUNT check is what fires. The unparseable counter's real triggers are power loss, a filesystem fault, or a different writer that does not flush per row. A bare `except: continue` would let the checker read an EARLIER row as
"the last one" and pass an arm whose artifact never fully parsed.

⚠️ A7-IN-s0, the arm paused on 2026-09-20, is the CLEAN counter-example and was VERIFIED rather than assumed: 107 lines, **0 unparseable**, file ends with a newline, last row step 1,070 = 107 x 10. ⚠️ I had written that this arm's kill DID truncate it; that was FALSE — a plausible mechanism asserted about a specific file without opening it. Why it stayed clean is NOT established: the ordered kill (trainer last, by explicit PID) and a per-row flush are both candidates and I have separated neither.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT.parent / "taniteval"))

import p_check_arm as PC                                    # noqa: E402

GOOD_ROW = {"step": 5000, "eval_box3d_centre": 13.92835, "eval_box3d_n_matched": 38.012,
            "eval_box3d_n_target": 38.012, "eval_box3d_size": 1.35, "eval_box3d_yaw": 0.81}


def _arm(tmp: Path, *, seed=1, steps=5000, rows=None, extra_lines=()):
    run = tmp / "run"
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(
        {"argv": ["--arm", "hier", "--seed", str(seed), "--steps", str(steps)]}),
        encoding="utf-8")
    lines = [json.dumps(r) for r in (rows if rows is not None else [GOOD_ROW])]
    lines += list(extra_lines)
    (run / "metrics.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tmp


def test_a_clean_arm_reads_VALID(tmp_path):
    rep = PC.check(_arm(tmp_path), seed=1, steps=5000)
    assert rep["status"] == "VALID" and rep["reasons"] == []
    assert rep["headline_eval_box3d_centre"] == 13.92835
    assert rep["identity_matched_equals_target"] is True
    assert rep["n_unparseable_metrics_lines"] == 0


def test_a_WRONG_SEED_is_INVALID(tmp_path):
    rep = PC.check(_arm(tmp_path, seed=0), seed=1, steps=5000)
    assert rep["status"] == "INVALID"
    assert any("seed mismatch" in r for r in rep["reasons"])


def test_a_WRONG_STEP_COUNT_is_INVALID(tmp_path):
    rep = PC.check(_arm(tmp_path, steps=2000), seed=1, steps=5000)
    assert rep["status"] == "INVALID"
    assert any("steps mismatch" in r for r in rep["reasons"])


def test_an_arm_that_never_EVALUATED_is_INVALID(tmp_path):
    rep = PC.check(_arm(tmp_path, rows=[{"step": 10, "loss": 1.0}]), seed=1, steps=5000)
    assert rep["status"] == "INVALID"
    assert any("never completed an eval" in r for r in rep["reasons"])


def test_a_BROKEN_label_density_identity_is_INVALID(tmp_path):
    row = dict(GOOD_ROW, eval_box3d_n_matched=30.0)          # matched != target
    rep = PC.check(_arm(tmp_path, rows=[row]), seed=1, steps=5000)
    assert rep["status"] == "INVALID"
    assert any("identity BROKEN" in r for r in rep["reasons"])


def test_a_NON_FINITE_headline_is_INVALID(tmp_path):
    rep = PC.check(_arm(tmp_path, rows=[dict(GOOD_ROW, eval_box3d_centre=float("nan"))]),
                   seed=1, steps=5000)
    assert rep["status"] == "INVALID"
    assert any("not a finite number" in r for r in rep["reasons"])


# ------------------------------------------------------------- the counted skip
def test_an_UNPARSEABLE_line_is_COUNTED_and_REFUSES(tmp_path):
    """⛔ The case a bare `except: continue` would swallow: a truncated final line."""
    rep = PC.check(_arm(tmp_path, extra_lines=['{"step": 5000, "eval_box3d_cen']),
                   seed=1, steps=5000)
    assert rep["n_unparseable_metrics_lines"] == 1
    assert rep["status"] == "INVALID"
    assert any("did not fully parse" in r for r in rep["reasons"])


def test_the_LAST_row_carrying_the_headline_wins_not_the_last_line(tmp_path):
    rows = [dict(GOOD_ROW, step=1000, eval_box3d_centre=20.0),
            dict(GOOD_ROW, step=5000, eval_box3d_centre=13.92835),
            {"step": 5001, "loss": 0.5}]                      # a TRAIN row after the eval
    rep = PC.check(_arm(tmp_path, rows=rows), seed=1, steps=5000)
    assert rep["status"] == "VALID" and rep["final_step"] == 5000
    assert rep["headline_eval_box3d_centre"] == 13.92835


def test_blank_lines_are_NOT_counted_as_unparseable(tmp_path):
    rep = PC.check(_arm(tmp_path, extra_lines=["", "   "]), seed=1, steps=5000)
    assert rep["n_unparseable_metrics_lines"] == 0 and rep["status"] == "VALID"


def test_a_TRUNCATED_RUN_missing_its_final_row_is_INVALID(tmp_path):
    """⛔ THE OOM / SIGKILL SIGNATURE, and the unparseable counter CANNOT SEE IT.

    The trainer writes each row and flushes immediately, so a hard kill loses the final ROW and
    leaves a file that parses perfectly and is merely short. The step-count check is the guard
    that fires; this test exists because the guard that catches the realistic failure had none.
    """
    rep = PC.check(_arm(tmp_path, rows=[dict(GOOD_ROW, step=4000, eval_box3d_centre=15.0)],
                        steps=5000), seed=1, steps=5000)
    assert rep["n_unparseable_metrics_lines"] == 0, (
        "the file PARSES — the unparseable counter is structurally blind to a missing row")
    assert rep["status"] == "INVALID"
    assert any("final eval at step 4000, expected 5000" in r for r in rep["reasons"])
