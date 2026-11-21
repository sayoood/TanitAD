"""Apply the D-REFAV1-CCOS-EVAL test/tier fixes to BOTH the G: repo and the off-Drive
mirror, by exact string replacement, with retries on the flapping mount and a
content check afterwards (an exit code on G: is not evidence).

Why each edit exists (all MEASURED 2026-09-05 on the dev box, torch 2.11.0+cu128):
  * test_cost_ccos.py::test_c  -- the identity control read 5.96e-08 (d=64),
    -2.38e-07 (4096) and -8.34e-07 (65,536): a float32 cosine of two IDENTICAL
    vectors is not exactly 1 because ``sqrt(x.x)**2 != x.x`` and the 65,536-term
    reduction is not exact. The shipped ``cos`` has the same property (1.3e-06,
    test_cost_chord.py::test_a3). Only the CHORD is exact (Sterbenz). Asserting
    ``== 0.0`` was asserting a property float32 cannot have.
  * test_cost_ccos.py::test_m  -- end-to-end, ``baseline_costs['cv']`` under ccos
    read 1.0531 on the tiny model: the cv row is scored inside the 4-row
    ``base_stack`` while ``z_ref`` is rolled with batch 1, so the two fields differ
    by batch-composition ulps and cv's centred vector is NOISE, not the zero vector.
    Its cosine against ``g - z_ref`` is then O(1/sqrt(D)) -- 0.053 at D = 64 here.
    The exact 1.0 of test_f holds only for bit-identical fields. The per-window
    value on the real model is BANKED by refav1_arm (``basecost_cv_cl``), never
    assumed.
  * test_refav1_arm.py -- ``ha0_ext`` (the echo control) joined the default arm
    list; the expected sets/tiers did not know it.
  * t1_eval.py DEFAULT_TIERS -- ``ha0_ext`` is T1 like ``ha``/``ha0`` (no recorded
    future: a0 is the BACKWARD difference at t0, kappa0 is the recorded curvature AT
    t0). Without it the bare t1_eval CLI refuses every new dump after the rollout.
"""
import sys, time, os

REPO = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
MIRROR = r"C:\Users\Admin\tanitad-wt"

EDITS = {
    r"stack\tests\test_cost_ccos.py": [
        (
'''def test_c_identity_control_is_exactly_zero(d):
    """A candidate whose centred rollout IS the goal's must pay EXACTLY 0.
    KNOWN VALUE: 0.0. (⚠️ the shipped `cos` FAILS its own version of this at
    1.3e-06 in float32 — `test_cost_chord.py::test_a3_*`.)"""
    e, u, _ = _basis(d, dtype=torch.float32)
    r = (FIELD_NORM * e)[None, None]
    zt = (r + 3.0 * u[None, None])
    gg = (r + 3.0 * u[None, None])
    v = _goal_term(zt, gg, "ccos", r)
    assert float(v.abs().max()) == 0.0, f"d={d}: got {float(v[0]):.6e}"
''',
'''def test_c_identity_control_is_exactly_zero(d):
    """A candidate whose centred rollout IS the goal's must pay 0 — to float32
    cosine precision. KNOWN VALUE: 0.0, tolerance 2e-6.

    ⚠️ MEASURED 2026-09-05 (torch 2.11.0+cu128, CPU): this read **5.96e-08**
    (d=64), **-2.38e-07** (4096) and **-8.34e-07** (65,536) — NOT exactly 0.
    A float32 cosine of two IDENTICAL vectors is not exactly 1: ``F.cosine_
    similarity`` normalises by ``sqrt(x.x)`` and ``sqrt(s)**2 != s`` in
    float32, and the 65,536-term reduction is itself inexact. The shipped
    ``cos`` has the SAME property (1.3e-06, `test_cost_chord.py::test_a3_*`);
    only the CHORD reads its identity control exactly (Sterbenz). The first
    version of this test asserted ``== 0.0`` — a property a float32 cosine
    cannot have — and would have failed on every machine. The tolerance is the
    one `test_d`/`test_e` already use for the SAME function.
    """
    e, u, _ = _basis(d, dtype=torch.float32)
    r = (FIELD_NORM * e)[None, None]
    zt = (r + 3.0 * u[None, None])
    gg = (r + 3.0 * u[None, None])
    v = float(_goal_term(zt, gg, "ccos", r).abs().max())
    assert v <= 2e-6, f"d={d}: got {v:.6e}"
    # and the SHIPPED branch's own identity error on the same fields, so the
    # comparison is like with like: ccos is no worse than cos here.
    v_cos = float(_goal_term(zt, gg, "cos").abs().max())
    assert v <= max(2e-6, 4.0 * v_cos), (
        f"d={d}: ccos identity error {v:.3e} vs cos's own {v_cos:.3e}")
'''),
        (
'''    ⚠️ `pytest.approx`, not `==`: `baseline_costs` comes from `icem_plan`'s
    SECOND `cost_fn(base_stack)` call, and MEASURED (`RESULT.md` §6b) the same
    control block scores 1-2 float32 ulp differently between batch compositions.
    """
    m = _model(seed=5)
    feats, v0, nav = _window(m.cfg, seed=5)
    pc = _pc(m.cfg)
    with torch.no_grad():
        a = m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=pc, cost_metric="cos")
        b = m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=pc, cost_metric="ccos")
    assert b.cost_metric == "ccos"
    assert a.goal_source == b.goal_source == "tactical_imagined", (
        "no goal on this window — the test measures nothing")
    for k in ("cv", "hold_v0"):
        assert float(b.baseline_costs[k]) == pytest.approx(1.0, abs=1e-4), (
            f"{k} under ccos = {b.baseline_costs[k]!r}, expected exactly the "
            f"no-information value 1.0")
        assert float(a.baseline_costs[k]) < 1e-3, (
''',
'''    ⚠️ NOT ``== 1.0`` AND NOT ``approx(1.0, 1e-4)`` — MEASURED 2026-09-05:
    ``baseline_costs['cv']`` under ccos read **1.0531** on this tiny model.
    `baseline_costs` comes from `icem_plan`'s SECOND `cost_fn(base_stack)`
    call, where cv is ROW 0 of a 4-row batch, while ``z_ref`` is rolled with
    batch 1. The two fields differ by batch-composition ulps, so cv's centred
    vector is a NOISE vector, not the zero vector, and its cosine against
    ``g - z_ref`` is O(1/sqrt(D)): +-0.125 at this model's D = 64, +-0.004 at
    the real D = 65,536. The exact 1.0 of `test_f` holds only for bit-identical
    fields. ⇒ the KNOWN VALUE here is "1.0 up to that noise", and the
    decision-relevant assertion is that the floor moved from ~1e-07 to ~1.0
    — SEVEN orders of magnitude — not its last digits. The per-window value on
    the real checkpoint is BANKED by `refav1_arm.py` (``basecost_cv_cl``), so
    it is measured, never assumed.
    """
    m = _model(seed=5)
    feats, v0, nav = _window(m.cfg, seed=5)
    pc = _pc(m.cfg)
    with torch.no_grad():
        a = m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=pc, cost_metric="cos")
        b = m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=pc, cost_metric="ccos")
    assert b.cost_metric == "ccos"
    assert a.goal_source == b.goal_source == "tactical_imagined", (
        "no goal on this window — the test measures nothing")
    D = m.cfg.tac_queries * m.cfg.d_state
    noise = 4.0 / D ** 0.5                     # 4 sigma of an O(1/sqrt(D)) cosine
    for k in ("cv", "hold_v0"):
        v = float(b.baseline_costs[k])
        assert abs(v - 1.0) <= noise, (
            f"{k} under ccos = {v!r}, expected the no-information value 1.0 up "
            f"to batch-composition noise {noise:.3f} (D={D})")
        assert v > 0.5, f"{k} under ccos = {v!r}: the floor did not move to ~1.0"
        assert float(a.baseline_costs[k]) < 1e-3, (
'''),
    ],
    r"stack\tests\test_refav1_arm.py": [
        (
'''    assert set(rec["arms"]) == {"cl", "ha", "ha0", "ol", "cl_navshuf",
                                "cl_oraclegoal"}
    assert rec["tiers"] == {"cl": "T1", "ha": "T1", "ha0": "T1", "ol": "T0",
                            "cl_navshuf": "T1", "cl_oraclegoal": "T0"}
''',
'''    # `ha0_ext` (2026-09-05, D-REFAV1-CCOS-EVAL): the ECHO control — constant
    # (a0, kappa0) of the MEASURED t0 state, T1 like `ha` / `ha0` (no recorded
    # future). ADDITIVE, like `ha0` before it.
    assert set(rec["arms"]) == {"cl", "ha", "ha0", "ha0_ext", "ol", "cl_navshuf",
                                "cl_oraclegoal"}
    assert rec["tiers"] == {"cl": "T1", "ha": "T1", "ha0": "T1", "ha0_ext": "T1",
                            "ol": "T0", "cl_navshuf": "T1", "cl_oraclegoal": "T0"}
'''),
    ],
    r"taniteval\tools\t1_eval.py": [
        (
'''    "ha0": "T1",
    "ol": "T0", "o16": "T0", "o6": "T0",
}
''',
'''    "ha0": "T1",
    # ⭐ ha0_ext (2026-09-05, D-REFAV1-CCOS-EVAL): the ECHO control of
    # stack/tanitad/eval/echo_gate.py in refav1 form — constant (a0, kappa0) of
    # the MEASURED t0 state held for K steps; a0 is the BACKWARD difference at
    # t0 and kappa0 the recorded curvature AT t0, so nothing after t0 enters.
    # T1 for the same reason as `ha` / `ha0`. Without this line the bare
    # t1_eval CLI refused every refav1 dump AFTER the rollout (the same
    # after-the-expensive-part failure D-HA0-TIER closed for `ha0`).
    "ha0_ext": "T1",
    "ol": "T0", "o16": "T0", "o6": "T0",
}
'''),
    ],
}


def rd(p):
    for i in range(3):
        try:
            with open(p, encoding="utf-8", newline="") as fh:
                return fh.read()
        except OSError as e:
            print("  retry read", i, p, e); time.sleep(6)
    raise SystemExit("cannot read " + p)


def wr(p, s):
    for i in range(12):
        try:
            with open(p, "w", encoding="utf-8", newline="") as fh:
                fh.write(s)
            if rd(p) == s:
                return
        except OSError as e:
            print("  retry write", i, p, e); time.sleep(6)
    raise SystemExit("cannot write " + p)


ok = True
pending = []
ROOTS = [MIRROR, REPO] if "--mirror-only" not in sys.argv else [MIRROR]
for rel, edits in EDITS.items():
    for root in ROOTS:
        p = os.path.join(root, rel)
        try:
            s = rd(p)
        except SystemExit:
            print("PENDING (mount down):", p); pending.append(rel); continue
        crlf = "\r\n" in s
        s_lf = s.replace("\r\n", "\n")
        for old, new in edits:
            n = s_lf.count(old)
            if n == 1:
                s_lf = s_lf.replace(old, new)
            elif new in s_lf and old not in s_lf:
                print("  already applied:", rel, root[:2])
            else:
                print("FAIL: old text found", n, "times in", p); ok = False
        out = s_lf.replace("\n", "\r\n") if crlf else s_lf
        if out != s:
            wr(p, out)
        got = rd(p).replace("\r\n", "\n")
        for _, new in edits:
            if new not in got:
                print("FAIL verify", p); ok = False
        print(("OK   " if ok else "FAIL ") + p + ("  (crlf)" if crlf else "  (lf)"))
print("ALL_APPLIED_AND_VERIFIED" if ok else "SOME_FAILED", "PENDING:", pending)
sys.exit(0 if ok else 1)
