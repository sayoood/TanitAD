"""Move `tac_goal_bce` from NO_CONSUMER to CONSUMED, and flip the test that
pinned the gap so it now pins the CLOSURE -- two-sided, same-breath control.

⛔ The status is not edited to make a test pass: `assert_consumers` MEASURED
"tac_goal_bce": "CONSUMED" on the patched tree, from its own detector, which is
a derivation independent of this literal.
"""
from __future__ import annotations

import ast
import io
import sys

SRC, DST = sys.argv[1], sys.argv[2]
with io.open(SRC, "rb") as fh:
    _raw = fh.read()
CRLF, LF = _raw.count(b"\r\n"), _raw.count(b"\n")
NEWLINE = "\r\n" if CRLF * 2 > LF else "\n"
print(f"source newline: CRLF={CRLF} LF={LF} -> {NEWLINE!r}")
s = _raw.decode("utf-8").replace("\r\n", "\n")

EDITS: list[tuple[str, str, str]] = []

A1 = '''    # ⛔ THE THREE GAPS. Each is a finding, and each has its own test below.
    "tac_goal_bce": "NO_CONSUMER",       # D-TACGOAL-1 — still open
    "tac_action_ce_v6": "NO_CONSUMER",   # v6 lands the ids, applies no CE
    "str_ce_refc": "NO_CONSUMER",        # refc has no strategic TOKEN head
}'''
B1 = '''    # ⭐⭐ WAS A GAP, CLOSED 2026-09-09 (D-TACGOAL-TRAINER-SEAM-OPEN).
    # `refc_v3_train.py` now calls `tac_goal_head.tac_goal_loss` on
    # `out["tac_goal_logits"]` behind `--w-tac-goal`, and a real backward puts
    # grad_abs_sum 3.602122873067856 on the head's two tensors (MEASURED at
    # smoke width; Research/2026-09-09-tacgoal-wiring/RESULT.md).
    # ⛔ This literal is NOT edited to make a test pass: `assert_consumers`
    # measures "CONSUMED" from its own detector, which is a derivation
    # independent of this map.
    # ⚠️ CONSUMED means A CONSUMER EXISTS, never that an ARM BOUGHT IT.
    # `--w-tac-goal` defaults to 0.0 and no live recipe passes it, so the 22
    # tokens are REACHABLE, not yet TRAINED. Reading this row as "the tokens
    # are a training signal in the live run" is the D-TACGOAL claim one step
    # too far -- the arm-level fact lives in config.json's effective-weights
    # block, which now carries a `--w-tac-goal` row for exactly this reason.
    "tac_goal_bce": "CONSUMED",
    # ⛔ THE TWO REMAINING GAPS. Each is a finding, and each has its own test.
    "tac_action_ce_v6": "NO_CONSUMER",   # v6 lands the ids, applies no CE
    "str_ce_refc": "NO_CONSUMER",        # refc has no strategic TOKEN head
}'''
EDITS.append(("C1 status literal", A1, B1))

A2 = '''def test_D_TACGOAL_1_is_STILL_OPEN_no_trainer_calls_the_goal_loss() -> None:
    """⛔⛔ THE HEADLINE. The head, the loss and the emitter all EXIST; the
    trainer calls none of them, so the 22 tactical goal tokens reach no
    gradient.

    ⭐ Both halves are asserted POSITIVELY, with a same-breath control, because
    an absence found through a channel that could not read is not an absence
    (``rg`` under-reports on the G: mount and still exits 0).
    """
    trainer = _read("stack/scripts/refc_v3_train.py")
    # -- the same-breath control: this file WAS read
    assert "def compute_losses_v3(" in trainer, _msg(
        "the control marker is missing from refc_v3_train.py — the read is "
        "not trustworthy, so the absences below prove nothing")
    for sym in ("tac_goal_loss", "TacGoalEmitter", "tac_goal_logits"):
        assert sym not in trainer, _msg(
            f"⭐ GOOD NEWS: refc_v3_train.py now mentions {sym!r} — "
            f"D-TACGOAL-1 may be closing. Re-run the census "
            f"(stack/scripts/v7_vocab_reach_census.py) and update the claim "
            f"register and RESULT.md: the 22 tactical goal tokens would move "
            f"from AUDIT_OR_METRIC_ONLY to TRAINING_SIGNAL.")
    # -- and the pieces that DO exist, so 'absent' cannot be read as 'deleted'
    head = _read("stack/tanitad/refs/tac_goal_head.py")
    assert "def tac_goal_loss(" in head and "class TacGoalTokenHead(" in head, \\
        _msg("tac_goal_head.py lost the loss or the head — the gap is no "
             "longer 'wiring missing', it is 'the parts are gone'")
    model = _read("stack/tanitad/refs/refc_v3.py")
    assert 'cache["tac_goal_logits"] = self.tac_goal_tok_head(z_tac)' in model, \\
        _msg("refc_v3.py no longer computes tac_goal_logits — the head is not "
             "even forward-run; re-derive the census before quoting it")'''
B2 = '''def test_D_TACGOAL_TRAINER_SEAM_IS_CLOSED_the_trainer_calls_the_goal_loss(
) -> None:
    """⭐⭐ THE HEADLINE, INVERTED 2026-09-09. This test used to assert the seam
    was OPEN -- ``assert "tac_goal_loss" not in trainer`` -- and it was RIGHT
    for three days: the head, the loss and the emitter all existed and the
    trainer called none of them, so refcv5-v2 trained 11,286 parameters for
    40,284 steps at ``grad_abs_sum`` exactly 0.

    It now asserts the OPPOSITE, and it is still two-sided: if the call is ever
    removed again, this goes RED and names what to re-derive.

    ⭐ Every half is a POSITIVE assertion with a same-breath control, because an
    absence found through a channel that could not read is not an absence
    (``rg`` under-reports on the G: mount and still exits 0). ⛔ The control
    matters MORE in this direction: a file that failed to read yields ``sym not
    in trainer`` = True for every symbol, so the OLD form of this test would
    have passed on an unreadable file. The new form fails on one.
    """
    trainer = _read("stack/scripts/refc_v3_train.py")
    # -- the same-breath control: this file WAS read
    assert "def compute_losses_v3(" in trainer, _msg(
        "the control marker is missing from refc_v3_train.py — the read is "
        "not trustworthy, so the assertions below prove nothing")
    # -- the CALL, the FLAG and the TARGET, each named explicitly
    for sym, what in (
            ("_tac_goal_head.tac_goal_loss(", "the loss CALL"),
            ('ap.add_argument("--w-tac-goal"', "the weight FLAG"),
            ('"flag": "--w-tac-goal"', "the effective-weight GATE row"),
            ('item["tac_goal_y"]', "the dataset TARGET"),
            ("tac_goal_logits", "the head's logits"),
    ):
        assert sym in trainer, _msg(
            f"⛔ refc_v3_train.py no longer contains {what} ({sym!r}) — "
            f"D-TACGOAL-TRAINER-SEAM-OPEN HAS RE-OPENED. The 22 tactical "
            f"goal tokens would fall back from TRAINING_SIGNAL to "
            f"AUDIT_OR_METRIC_ONLY. Re-run the census "
            f"(stack/scripts/v7_vocab_reach_census.py) and update the claim "
            f"register and RESULT.md.")
    # -- the term must be ABSENT at weight zero, not multiplied by zero: that
    #    is what keeps the OFF path bit-identical to the pre-wiring trainer.
    assert "if _w_tg > 0.0:" in trainer, _msg(
        "the tac_goal term is no longer gated on a positive weight — if it "
        "is now multiplied by zero instead of skipped, the default arm's "
        "autograd graph differs from the pre-wiring trainer's and the "
        "bit-identity proof in RESULT.md no longer holds.")
    # -- and the pieces it depends on, so 'present' cannot be read as 'whole'
    head = _read("stack/tanitad/refs/tac_goal_head.py")
    assert "def tac_goal_loss(" in head and "class TacGoalTokenHead(" in head, \\
        _msg("tac_goal_head.py lost the loss or the head — the trainer's call "
             "now points at nothing")
    model = _read("stack/tanitad/refs/refc_v3.py")
    assert 'cache["tac_goal_logits"] = self.tac_goal_tok_head(z_tac)' in model, \\
        _msg("refc_v3.py no longer computes tac_goal_logits — the head is not "
             "even forward-run; re-derive the census before quoting it")'''
EDITS.append(("C2 flip the headline test", A2, B2))

for tag, a, b in EDITS:
    n = s.count(a)
    if n != 1:
        raise SystemExit(f"⛔ ANCHOR NOT UNIQUE for {tag}: {n} occurrences")
    s = s.replace(a, b, 1)
    print(f"applied {tag}")

ast.parse(s)
assert s.count('"tac_goal_bce": "CONSUMED"') == 1
assert "STILL_OPEN_no_trainer_calls" not in s

with io.open(DST, "wb") as fh:
    fh.write(s.replace("\n", NEWLINE).encode("utf-8"))
print(f"WROTE {DST} lines={s.count(chr(10))}")
