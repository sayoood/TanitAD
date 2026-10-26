"""PREREG_V8F_OACC Amendment 2 (post-data, CONTROL ONLY): the B-vs-X built-lever control compared two trainer RECORD FORMATS.
MEASURED 2026-10-26 on the real records: B (banked M2b, trained 2026-10-17) writes `residual_zero_init` AFTER training in summary.json (False) and
BEFORE it in selfcheck.json (True); the current trainer writes it before training in both, and records w_tac / w_cons inside `w`, which B's
older record lacks. B's own checkpoint cfg reads w_tac 1.0, w_cons 0.5 at both seeds, X's the same; X's head_o.2 weight is non-zero (|.|_1 164.4).
Fix: `residual_zero_init` joins lever_skip (the expected before/after difference test_v8f_grid / test_v8f_hist already drop); `w` is compared KEY
BY KEY, and a key B's record lacks must equal B's BUILT config. python apply_oacc_a2.py <tree>"""
import io
import sys

T = sys.argv[1].rstrip("/") + "/"
PK = T + "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"


def rep(p, pairs):
    s = io.open(p, encoding="utf-8", newline="").read()
    for a, b in pairs:
        assert s.count(a) == 1, (p, s.count(a), a[:100])
        s = s.replace(a, b)
    assert "\r" not in s
    io.open(p, "w", encoding="utf-8", newline="").write(s)


rep(PK + "code/v8f_oacc_read.py", [
    ('lever_skip=("w_acc", "o_interface", "code_mix", "params", "n_params_trainable")) -> list:',
     'lever_skip=("w_acc", "o_interface", "code_mix", "params", "n_params_trainable", "residual_zero_init", "w")) -> list:'),
    ('''        diff = sorted(k for k in set(lg) & set(lb) if k not in lever_skip and lg[k] != lb[k])
        if diff:
            bad.append(f"{tag}: {w}.json built levers {diff} differ from B_s{s}'s")''',
     '''        diff = sorted(k for k in set(lg) & set(lb) if k not in lever_skip and lg[k] != lb[k])
        if diff:
            bad.append(f"{tag}: {w}.json built levers {diff} differ from B_s{s}'s")
        # PREREG_V8F_OACC Amendment 2: `w` key by key; a weight B's older record does not carry must equal B's BUILT config (W_B_BUILT)
        wb, wx = lb.get("w") or {}, lg.get("w") or {}
        w_bad = sorted(k for k in set(wb) | set(wx) if wx.get(k) != (wb[k] if k in wb else W_B_BUILT.get(k, "absent from B's config")))
        if w_bad:
            bad.append(f"{tag}: {w}.json built loss weights {w_bad} differ from B_s{s}'s (a weight B's record lacks must equal B's built config {W_B_BUILT})")'''),
    ('def check_x_run(',
     "#: B's BUILT loss weights that its older record does not list (MEASURED 2026-10-26: the cfg of the banked M2b_s0 / M2b_s1 checkpoints; PREREG_V8F_OACC Amendment 2)\n"
     'W_B_BUILT = {"w_tac": 1.0, "w_cons": 0.5}\n\n\n'
     'def check_x_run('),
])
rep(T + "stack/tests/test_v8f_oacc.py", [
    ('''@needs_reader
def test_check_x_run_names_each_clause_of_the_token_by_its_own_message():''',
     '''@needs_reader
def test_check_x_run_amendment_2_record_formats():
    """PREREG_V8F_OACC Amendment 2: B's older record (residual_zero_init after training, `w` without w_tac / w_cons) against X's current one is
    ONE lever; any real loss-weight difference is still refused, by its name."""
    sm, argv, sc, rsm, rargv = _gate_records()
    new_w = {"w_pen": 0.5, "w_pen_self": 0.25, "w_wm": 0.1, "w_tac": 1.0, "w_cons": 0.5}
    for rec in (sm, sc):
        rec["levers"] = dict(rec["levers"], w=dict(new_w), residual_zero_init=True)
    rsm["levers"] = dict(rsm["levers"], w={"w_pen": 0.5, "w_pen_self": 0.25, "w_wm": 0.1}, residual_zero_init=False)
    assert _no_crash(RD.check_x_run, 0, sm, argv, sc, rsm, rargv, 42.9311, holdout="H") == []
    for key, val in (("w_tac", 2.0), ("w_cons", 0.25), ("w_wm", 0.2), ("w_pen", None), ("w_new", 1.0)):
        w2 = dict(new_w)
        if val is None:
            w2.pop(key)
        else:
            w2[key] = val
        sm2 = dict(sm, levers=dict(sm["levers"], w=w2))
        got = _no_crash(RD.check_x_run, 0, sm2, argv, sc, rsm, rargv, 42.9311, holdout="H")
        assert got and any("loss weights" in g and key in g for g in got), (key, got)
    rsm2 = dict(rsm, levers=dict(rsm["levers"], w={"w_pen": 0.5, "w_pen_self": 0.25, "w_wm": 0.1, "w_tac": 2.0}))   # B's record carries w_tac: compared as recorded
    assert any("w_tac" in g for g in _no_crash(RD.check_x_run, 0, sm, argv, sc, rsm2, rargv, 42.9311, holdout="H"))


@needs_reader
def test_check_x_run_names_each_clause_of_the_token_by_its_own_message():'''),
])
rep(PK + "raw/2026-10-25-oacc/mutate_oacc.py", [
    ('''    off(READER, "other built levers may differ from B's", "if diff:", "reader"),''',
     '''    off(READER, "other built levers may differ from B's", "if diff:", "reader"),
    off(READER, "loss weights may differ from B's (Amendment 2)", "if w_bad:", "reader"),'''),
])
print("ok")
