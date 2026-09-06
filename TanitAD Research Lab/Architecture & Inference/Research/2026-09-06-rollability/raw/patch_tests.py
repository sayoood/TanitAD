"""D-ROLL-1 -- minimal, assertion-preserving updates to the two sibling tests
that pinned the UNCONDITIONAL construction. ASCII-only prints.

Every existing assertion is kept byte-for-byte; only the BUILD states the lever
it is pricing. One control is STRENGTHENED (kin3 now asks for the head and is
still refused, so it tests the vocabulary rather than the default).
"""
import hashlib
import io
import sys

REPO = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"

EDITS = [
    # ---------------------------------------------------------------- v3 ----
    (r"stack\tests\test_refc_v3.py", [
        ("""    assert refc_v3_xl_config().d_tac == refc_v3_small_config().d_tac
    assert param_breakdown_v3(RefCV3Model(refc_v3_small_config()))["total"] \\
        == 62_930_419 + NAV + STR + 5_130 + TACGOAL
    assert param_breakdown_v3(RefCV3Model(refc_v3_xl_config()))["total"] \\
        == 217_760_775 + NAV + STR + 5_130 + TACGOAL
    # and it carries its OWN ledger line rather than hiding inside `tac_heads`,
    # because the arm's whole question is what the goal-SET head buys
    assert param_breakdown_v3(RefCV3Model(refc_v3_small_config()))[
        "tac_goal_tok_head"] == TACGOAL
""",
         """    assert refc_v3_xl_config().d_tac == refc_v3_small_config().d_tac
    # \u2b50 D-ROLL-1 (2026-09-06): the goal-SET head is OPT-IN, so a rung that
    # PRICES it must ASK for it. Building it on the v7 vocabulary alone put
    # 11,286 params into every rebuild of a checkpoint trained before the head
    # existed -- params no recorded `param_breakdown` names -- and made
    # refcv4b@40284, three refcv3 checkpoints and the LIVE refcv5 A40 run
    # unrollable. Every assertion here is unchanged; the BUILD now states the
    # lever it is pricing.
    _tg = lambda c: dataclasses.replace(c, tac_goal_tok_head=True)  # noqa: E731
    assert param_breakdown_v3(RefCV3Model(_tg(refc_v3_small_config())))["total"] \\
        == 62_930_419 + NAV + STR + 5_130 + TACGOAL
    assert param_breakdown_v3(RefCV3Model(_tg(refc_v3_xl_config())))["total"] \\
        == 217_760_775 + NAV + STR + 5_130 + TACGOAL
    # and it carries its OWN ledger line rather than hiding inside `tac_heads`,
    # because the arm's whole question is what the goal-SET head buys
    assert param_breakdown_v3(RefCV3Model(_tg(refc_v3_small_config())))[
        "tac_goal_tok_head"] == TACGOAL
    # \u26d4 CONTROL, same breath: the DEFAULT build -- v7.0 vocabulary, flag OFF
    # -- is the one a banked checkpoint rebuilds through, and it must carry
    # NEITHER the head nor the ledger line. That is the rollability contract,
    # and without this line the rung pins above would pass again under the
    # unconditional construction that broke every checkpoint.
    assert "tac_goal_tok_head" not in param_breakdown_v3(
        RefCV3Model(refc_v3_small_config()))
"""),
        ("""    base = param_breakdown_v3(RefCV3Model(refc_v3_sized_config("base",
                                                              hier=True)))
""",
         """    base = param_breakdown_v3(RefCV3Model(_tg(refc_v3_sized_config(
        "base", hier=True))))
"""),
    ]),
    # ------------------------------------------------------------- wiring ----
    (r"stack\tests\test_tac_goal_wiring.py", [
        ("""    cfg.tac_vocab_version = "v7.0"          # the flag `--v7-labels` pins
    return v3.RefCV3Model(cfg), cfg
""",
         """    cfg.tac_vocab_version = "v7.0"          # the flag `--v7-labels` pins
    # \u2b50 D-ROLL-1 (2026-09-06): the head is OPT-IN. A v7 vocabulary is
    # NECESSARY but no longer SUFFICIENT -- building it on the vocabulary alone
    # added 11,286 params to every rebuild of a checkpoint trained before the
    # head existed and made refcv4b, three refcv3 checkpoints and the LIVE
    # refcv5 A40 run unrollable/unresumable. Every assertion in this file is
    # unchanged; the BUILD states the lever.
    cfg.tac_goal_tok_head = True
    return v3.RefCV3Model(cfg), cfg
"""),
        ("""    cfg.tac_vocab_version = "kin3"          # no --v7-labels on the command line
    m = v3.RefCV3Model(cfg)
    assert m.tac_goal_tok_head is None
""",
         """    cfg.tac_vocab_version = "kin3"          # no --v7-labels on the command line
    # \u2b50 D-ROLL-1: ASK FOR IT ANYWAY. With the opt-in flag the vocabulary
    # refusal would otherwise be indistinguishable from the flag's default, and
    # a control that can pass for two reasons discriminates neither. Setting it
    # True keeps this a test of the VOCABULARY.
    cfg.tac_goal_tok_head = True
    m = v3.RefCV3Model(cfg)
    assert m.tac_goal_tok_head is None
"""),
    ]),
]

fail = 0
for rel, subs in EDITS:
    path = REPO + "\\" + rel
    with io.open(path, encoding="utf-8", newline="") as fh:
        t = fh.read()
    before = t
    # LINE ENDINGS ARE PART OF THE ANCHOR. `test_refc_v3.py` is CRLF and
    # `test_tac_goal_wiring.py` is LF; an LF anchor silently matches 0 times in
    # the CRLF file, which reads exactly like "the code is not there".
    crlf = "\r\n" in t
    for old, new in subs:
        if crlf:
            old = old.replace("\n", "\r\n")
            new = new.replace("\n", "\r\n")
        n = t.count(old)
        if n != 1:
            print("ANCHOR-NOT-UNIQUE %s count=%d :: %r" % (rel, n, old[:70]))
            fail += 1
            continue
        t = t.replace(old, new)
    if t == before:
        print("NO-CHANGE", rel)
        continue
    with io.open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(t)
    h = hashlib.md5(io.open(path, "rb").read()).hexdigest()
    print("PATCHED %s md5=%s marker_D-ROLL-1=%d" % (rel, h, t.count("D-ROLL-1")))

sys.exit(1 if fail else 0)
