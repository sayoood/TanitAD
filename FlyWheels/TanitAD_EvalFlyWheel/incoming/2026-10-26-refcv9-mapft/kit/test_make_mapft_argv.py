"""Unit tests for make_mapft_argv.py -- PREREG_MAPFT s1 budget rule.

Every expected value below is a LITERAL worked by hand from the sealed text
    N = min(10000, floor(0.85 * (21600 - smoke_wall_s - 1800) / s_step)),  N < 1000 -> STOP
    eval_every = max(500, N // 4),  ema_decay = round(1 - 10 / N, 5),  warmup 200, lr 1e-4
never as an expression over the code under test (CLAUDE.md: "a cross-check must be derived independently").
The MUTANT tests reintroduce the classic implementation errors and require that the literal cases DISTINGUISH each one
from the correct rule -- a literal set that cannot tell a mutant apart would be decorative.

Run:  python -m pytest kit/test_make_mapft_argv.py -q --basetemp=F:/refcv9_mapft_scratch/pt_kit      (no torch needed)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_mapft_argv as M  # noqa: E402

BASE = HERE / "refcv8_launch_argv_base.json"

# (s_step, smoke_wall_s) -> (N, eval_every, ema_decay)   -- hand-worked, see the comments
CASES = [
    # avail = 21600-600-1800 = 19200; 0.85*19200 = 16320; /3.0 = 5440
    ("3.0", "600", 5440, 1360, 0.99816),
    # avail = 19500; 0.85*19500 = 16575; /1.0 = 16575 -> capped at 10000
    ("1.0", "300", 10000, 2500, 0.999),
    # 16320 / 16.0 = 1020 -> launches; N//4 = 255 < 500 so the floor 500 binds
    ("16.0", "600", 1020, 500, 0.9902),
    # exactly 1000 (16320 / 16.32 = 1000 exactly): N < 1000 is FALSE -> launches
    ("16.32", "600", 1000, 500, 0.99),
]
STOPS = [
    ("17.0", "600", 960),      # 16320 / 17 = 960
    ("16.33", "600", 999),     # 16320 / 16.33 = 999.38 -> 999
    ("2.0", "20000", 0),       # the smoke alone ate the budget: available_s < 0 -> N = 0
]


@pytest.mark.parametrize("s,w,n,ev,ema", CASES)
def test_launch_cases(s, w, n, ev, ema):
    b = M.derive_budget(s, w)
    assert b["stop"] is False
    assert b["N"] == n
    assert b["steps"] == n
    assert b["eval_every"] == ev
    assert b["ema_decay"] == ema
    assert b["warmup"] == 200 and b["lr"] == "1e-4"


@pytest.mark.parametrize("s,w,n", STOPS)
def test_stop_cases(s, w, n):
    b = M.derive_budget(s, w)
    assert b["stop"] is True and b["N"] == n and "steps" not in b


def test_cap_is_flagged_and_exact_integer_not_lost_to_float():
    assert M.derive_budget("1.0", "300")["capped"] is True
    assert M.derive_budget("3.0", "600")["capped"] is False
    b = M.derive_budget("16.32", "600")      # exact integer 1000: a double may read 999.99999 -> floor 999 -> STOP
    assert b["n_floor_exact"] == 1000 and b["N"] == 1000 and b["stop"] is False


def test_ema_tie_is_pythons_literal_expression():
    # N=3200: 1 - 10/3200 = 0.996875 is an exact decimal tie at 5 dp; the seal's text is the Python expression
    # round(1 - 10 / N, 5), which yields 0.99687 (the double sits just under the tie). Pinned so a "fix" is deliberate.
    assert round(1 - 10 / 3200, 5) == 0.99687
    # derive for an N of 3200: avail 19200 -> 16320/s = 3200 -> s = 5.1
    b = M.derive_budget("5.1", "600")
    assert b["N"] == 3200 and b["ema_decay"] == 0.99687 and b["eval_every"] == 800


def test_bad_inputs_refused():
    for s, w in [("0", "600"), ("-1", "600"), ("abc", "600"), ("3.0", "-5")]:
        with pytest.raises(SystemExit):
            M.derive_budget(s, w)


# ----------------------------------------------------------------------------- mutants: each must be told apart
def _mutant_no_085(s, w):
    return math.floor((21600 - w - 1800) / s)


def _mutant_no_reserve(s, w):
    return math.floor(0.85 * (21600 - w) / s)


def _mutant_no_smoke(s, w):
    return math.floor(0.85 * (21600 - 1800) / s)


def _mutant_no_cap(s, w):
    return math.floor(0.85 * (21600 - w - 1800) / s)


def _mutant_5h(s, w):
    return math.floor(0.85 * (18000 - w - 1800) / s)


@pytest.mark.parametrize("mut", [_mutant_no_085, _mutant_no_reserve, _mutant_no_smoke, _mutant_5h])
def test_literal_cases_distinguish_each_mutant(mut):
    differs = False
    for s, w, n, _ev, _ema in CASES:
        got = min(10000, mut(float(s), float(w)))
        differs |= (got != n)
    assert differs, f"{mut.__name__} agrees with every literal case: the cases cannot detect it"


def test_literal_cases_distinguish_no_cap():
    s, w, n, *_ = CASES[1]
    assert _mutant_no_cap(float(s), float(w)) == 16575 != n


def test_stop_threshold_is_strict_less_than():
    # a `<=` mutant would STOP at exactly 1000; the literal says 1000 launches
    assert M.derive_budget("16.32", "600")["stop"] is False
    assert M.derive_budget("16.33", "600")["stop"] is True


# ----------------------------------------------------------------------------- the smoke reader
def _write_rows(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for st, el in rows:
            fh.write(json.dumps({"step": st, "elapsed_s": el, "loss": 1.0}) + "\n")
        fh.write('{"step": 60, "elapsed_s": 1')          # a torn final line: skipped, counted


def test_smoke_reader_literal(tmp_path):
    p = tmp_path / "metrics.jsonl"
    # elapsed_s is CUMULATIVE. rows >= step 20: (20, 71.0) .. (50, 160.9) -> (160.9 - 71.0) / 30 = 899/300 s/step
    _write_rows(p, [(10, 41.3), (20, 71.0), (30, 100.9), (40, 131.2), (50, 160.9)])
    r = M.read_smoke_metrics(p)
    from fractions import Fraction
    assert r["s_step"] == Fraction(899, 300)
    assert r["rows_used"] == [20, 30, 40, 50]
    assert r["n_unparseable_lines"] == 1
    assert r["warn_unstable"] is False
    # and the budget for smoke_wall_s=420: avail 19380; 0.85*19380 = 16473; /(899/300) = 5497.1 -> 5497
    b = M.derive_budget(r["s_step"], "420")
    assert b["N"] == 5497 and b["eval_every"] == 1374 and b["ema_decay"] == 0.99818


def test_smoke_reader_does_not_divide_cumulative_by_step(tmp_path):
    # the wrong reading (elapsed_s[last] / step[last]) would give 160.9/50 = 3.218; the right one is 2.9967
    p = tmp_path / "m.jsonl"
    _write_rows(p, [(10, 41.3), (20, 71.0), (30, 100.9), (40, 131.2), (50, 160.9)])
    r = M.read_smoke_metrics(p)
    assert abs(r["s_step_float"] - 2.99667) < 1e-4 and abs(r["s_step_float"] - 3.218) > 0.2


def test_smoke_reader_refuses_malformed(tmp_path):
    p = tmp_path / "a.jsonl"
    _write_rows(p, [(10, 41.3)])
    with pytest.raises(SystemExit):
        M.read_smoke_metrics(p)                       # fewer than 2 usable rows
    p2 = tmp_path / "b.jsonl"
    _write_rows(p2, [(20, 71.0), (30, 70.0), (40, 80.0)])
    with pytest.raises(SystemExit):
        M.read_smoke_metrics(p2)                      # elapsed_s going backwards


def test_smoke_reader_flags_unstable(tmp_path):
    p = tmp_path / "c.jsonl"
    _write_rows(p, [(20, 70.0), (30, 100.0), (40, 160.0), (50, 190.0)])   # 3.0 / 6.0 / 3.0 s/step
    r = M.read_smoke_metrics(p)
    assert r["warn_unstable"] is True and abs(r["interval_spread"] - 1.0) < 1e-9


# ----------------------------------------------------------------------------- argv algebra (the real base)
@pytest.fixture(scope="module")
def base():
    return M.load_base(BASE)


def _g(argv, flag):
    return M.flag_val(argv, flag)


def test_base_is_refcv8_launch_argv(base):
    assert len(base) == 210
    assert M.sha_compact(base) == "188aa439258d28e712e9a1b4a8ebda982d5f068b4515022f54ba1172e18fd956"


def test_wrong_base_refused(tmp_path, base):
    bad = list(base)
    i = bad.index("--steps")
    bad[i + 1] = "48851"
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"argv": bad}), encoding="utf-8")
    with pytest.raises(SystemExit):
        M.load_base(p)


def test_full_argv_literals(base):
    b = M.derive_budget("3.0", "600")
    av, edits = M.build_full(base, b)
    # 210 - 6 removed flags x 2 tokens + (--r8-train-only-map = 1) + (--conflict-detector off = 2) = 201
    assert len(av) == 201
    assert _g(av, "--steps") == ["5440"]
    assert _g(av, "--warmup") == ["200"]
    assert _g(av, "--lr") == ["1e-4"]
    assert _g(av, "--eval-every") == ["1360"]
    assert _g(av, "--ema-decay") == ["0.99816"]
    assert _g(av, "--r8-train-only-map") == []
    assert _g(av, "--conflict-detector") == ["off"]
    assert _g(av, "--init-from") == ["/home/nvidia/r8ns/ckpt/ckpt_ema_48852.pt"]
    assert _g(av, "--out") == ["/home/nvidia/r8ns/mapft/runs/refcv8-mapft-s0"]
    assert _g(av, "--join-defect-masks") == [
        "/home/nvidia/r8ns/mapft/launch_tree/stack/tanitad/configs/refcv8_join_label_defects_train4572.json"]
    for gone in ("--b1-guard-ref-ap2m", "--b1-guard-step", "--grad-share-every", "--grad-share-terms",
                 "--grad-share-at", "--conflict-every"):
        assert _g(av, gone) is None, gone
    # everything else is refcv8's, token for token
    keep = {"--batch": ["16"], "--seed": ["0"], "--w-map-hires": ["1.0"], "--eval-batches": ["8"],
            "--r8-alloc-emit-start": ["1000"], "--save-every": ["500"], "--log-every": ["50"], "--opt": ["dd"],
            "--v2-cache": ["/home/nvidia/data/refcv8-b1-416x1024-train-4572"],
            "--map-hires-class-weights": ["/home/nvidia/data/refcv7/map_hires_class_weights_train_100x30.json"]}
    for f, v in keep.items():
        assert _g(av, f) == v, f
    changed = {f for f, v in M.groups(base) if _g(av, f) is not None and _g(av, f) != v}
    assert changed == {"--init-from", "--join-defect-masks", "--out", "--steps", "--warmup", "--eval-every",
                       "--ema-decay"}, changed


def test_smoke_argv_literals(base):
    av, _ = M.build_smoke(base)
    assert _g(av, "--steps") == ["50"] and _g(av, "--log-every") == ["10"] and _g(av, "--eval-every") == ["0"]
    assert _g(av, "--save-every") == ["50"] and _g(av, "--warmup") == ["10"]
    assert _g(av, "--out") == ["/home/nvidia/r8ns/mapft/runs/refcv8-mapft-smoke50"]
    assert _g(av, "--r8-train-only-map") == [] and _g(av, "--conflict-detector") == ["off"]
    assert _g(av, "--batch") == ["16"]                      # timing is only meaningful at the launch batch


def test_stop_builds_nothing(base):
    b = M.derive_budget("17.0", "600")
    with pytest.raises(SystemExit):
        M.build_full(base, b)


def test_audit_catches_a_tampered_budget(base):
    b = M.derive_budget("3.0", "600")
    av, _ = M.build_full(base, b)
    assert M.audit_argv(av, b) == []
    i = av.index("--steps")
    bad = list(av)
    bad[i + 1] = "6000"                                    # a hand-edited "round" budget
    assert any("--steps" in x for x in M.audit_argv(bad, b))
    bad2 = [t for t in av]
    j = bad2.index("--conflict-detector")
    bad2[j + 1] = "auto"
    assert any("conflict-detector" in x for x in M.audit_argv(bad2, b))


def test_cli_stop_exit_code_and_marker(tmp_path, capsys):
    out = tmp_path / "full.json"
    rc = M.main(["build", "--s-step", "17.0", "--smoke-wall-s", "600", "--base", str(BASE), "--out", str(out)])
    assert rc == 3 and not out.exists()
    assert "ZZMAPFT-STOPZZ" in capsys.readouterr().out
    assert (tmp_path / "full_STOP.json").exists()


def test_cli_build_roundtrip(tmp_path, capsys):
    out = tmp_path / "full.json"
    rc = M.main(["build", "--s-step", "3.0", "--smoke-wall-s", "600", "--base", str(BASE), "--out", str(out)])
    assert rc == 0
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["argv_sha256"] == M.sha_compact(d["argv"]) and d["budget"]["N"] == 5440
    assert "ZZMAPFT-BUILT-5440ZZ" in capsys.readouterr().out
    assert M.main(["audit", "--s-step", "3.0", "--smoke-wall-s", "600", "--argv", str(out)]) == 0
    assert M.main(["audit", "--s-step", "3.5", "--smoke-wall-s", "600", "--argv", str(out)]) == 2


def test_cli_build_from_smoke_metrics_roundtrip(tmp_path, capsys):
    # the path the Thor script takes: s_step from the smoke's metrics.jsonl (Fractions in the record must serialise)
    m = tmp_path / "metrics.jsonl"
    _write_rows(m, [(10, 41.3), (20, 71.0), (30, 100.9), (40, 131.2), (50, 160.9)])
    out = tmp_path / "full.json"
    rc = M.main(["build", "--from-smoke-metrics", str(m), "--smoke-wall-s", "420", "--base", str(BASE), "--out", str(out)])
    assert rc == 0
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["budget"]["N"] == 5497 and d["smoke"]["s_step"] == "899/300" and d["smoke"]["rows_used"] == [20, 30, 40, 50]
    assert _g(d["argv"], "--steps") == ["5497"] and _g(d["argv"], "--eval-every") == ["1374"]
    assert _g(d["argv"], "--ema-decay") == ["0.99818"]
    assert M.main(["audit", "--from-smoke-metrics", str(m), "--smoke-wall-s", "420", "--argv", str(out)]) == 0
    # a different wall (600 s) changes N: 0.85*(21600-600-1800)/(899/300) = 5446.1 -> 5446 != 5497 -> the audit must FAIL
    assert M.main(["audit", "--from-smoke-metrics", str(m), "--smoke-wall-s", "600", "--argv", str(out)]) == 2
