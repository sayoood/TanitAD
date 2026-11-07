"""K2-WCOST -- G-LIVE refuses a K2 argv whose W-MAP-1 weight (`--w-r8-drivable`) is not 0.1 / R, R = the W-COST ratio the trainer prints at the first
labelled step of G-LIVE's OWN smoke (refc_v3_train.py:11856-11862, REFCV9_DESIGN 2.7 W-COST).

  the check     `launch_gate.wcost_reasons(prof, launch_argv, lines)` -- entered ONLY for refcv9-e2 / refcv9-full / refcv9-full-d1 AND an argv carrying `--r8-critic-drivable`
  the reader    `launch_gate.wcost_print_shim` installed by `run_smoke` as `refc_v3_train.print` (G-LIVE runs the trainer IN PROCESS and kept its stdout nowhere)
  the stamp     `r9_e2_make_argv.stamp_w_cost` (`main` only: `variant_doc` and `regen_k0` are untouched -> the committed files still regenerate byte-equal)

Every guard has a deliberate-regression arm (a mutation of launch_gate.py, listed in the K2-WCOST HANDOFF) that must turn the named test RED.
"""
from __future__ import annotations

import json
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import launch_gate as LG  # noqa: E402
import r9_e2_make_argv as MK  # noqa: E402

RUNS = ROOT / "ops" / "runs.d"
FAMILY = ("refcv9-e2", "refcv9-full", "refcv9-full-d1")
VARIANTS = ("K2", "K2M1N20", "K2M3", "K2M3N20")
K0 = json.loads((RUNS / "r9e2_K0.argv.json").read_text(encoding="utf-8"))["argv"]
K2_ARGV = {v: json.loads((RUNS / f"r9e2_{v}.argv.json").read_text(encoding="utf-8"))["argv"] for v in VARIANTS}
TRAINER = (ROOT / "scripts" / "refc_v3_train.py").read_bytes().decode("utf-8").replace("\r\n", "\n")
LAUNCH_GATE_SRC = (ROOT / "scripts" / "launch_gate.py").read_bytes().decode("utf-8").replace("\r\n", "\n")

FMT = ("[v3] refcv9 W-COST at step %d: critic-BCE / trajectory-loss = %.4f  (critic BCE %.4f, traj %.4f, n_drv %d, --w-r8-drivable %.4g; "
       "target weighted ratio ~0.1 => w = 0.1 / ratio)")


FMT_S = FMT.replace("= %.4f", "= %s")


def line(R, step=0):
    """A W-COST line in the trainer's own format (R is the text printed, e.g. '2.5000')."""
    return FMT_S % (step, R, 0.6931, 0.2772, 120, 0.1)


def judge(R, w, prof="refcv9-e2", variant="K2", lines=None):
    argv = LG.set_flag(K2_ARGV[variant], "--w-r8-drivable", [str(w)])
    return LG.wcost_reasons(LG.PROFILES[prof], argv, [line(R)] if lines is None else lines)


# ============================================================== the trainer's line and the gate's reader agree ==
def test_the_reader_regex_matches_the_trainers_own_format_string_and_the_trainer_still_prints_it():
    # the format literal is pinned in the TRAINER's source (a reworded print would silently blind the check)
    assert '"[v3] refcv9 W-COST at step %d: critic-BCE / trajectory-loss = %.4f  (critic BCE %.4f, traj %.4f, n_drv %d, --w-r8-drivable %.4g; "' in TRAINER
    assert 'not getattr(model, "_r9_wcost_printed", False)' in TRAINER and 'float(losses.get("n_drv", 0.0)) > 0.0' in TRAINER      # once, at the first LABELLED step
    m = LG.WCOST_LINE_RE.search(FMT % (7, 2.5, 0.6931, 0.2772, 120, 0.04))
    assert m and m.group(1) == "7" and float(m.group(2)) == 2.5
    assert LG.WCOST_LINE_KEY in FMT % (0, 1.0, 1.0, 1.0, 1, 0.1)


def test_the_weight_text_is_exactly_what_the_composer_writes():
    for R in (0.05, 0.37, 0.9999, 1.0, 1.7, 2.5, 3.0, 7.77, 12.5, 49.0):
        assert LG.wcost_weight_text(R) == repr(MK.w_from_ratio(R)), R
    assert LG.wcost_weight_text(2.5) == "0.04" and LG.wcost_weight_text(0.5) == "0.2"


# ============================================================== the check: PASS / FAIL ==
@pytest.mark.parametrize("v", VARIANTS)
def test_R_2_5_with_w_0_04_PASSES(v):
    r, d = judge("2.5000", 0.04, variant=v)
    assert r == [], r
    assert d["on"] is True and d["R"] == 2.5 and d["step"] == 0 and d["w_argv"] == 0.04 and d["w_want"] == "0.04" and d["deviation"] < 1e-9


@pytest.mark.parametrize("v", VARIANTS)
@pytest.mark.parametrize("prof", FAMILY)
def test_R_2_5_with_the_placeholder_0_1_FAILS_and_the_message_names_R_the_argv_weight_and_0_04(v, prof):
    r, d = judge("2.5000", 0.1, prof=prof, variant=v)
    assert len(r) == 1, r
    msg = r[0]
    assert msg.startswith("G-LIVE W-COST:") and "--w-r8-drivable 0.1 is not 0.1 / R" in msg
    assert "R = 2.5" in msg and "write --w-r8-drivable 0.04" in msg and "--w-cost-ratio 2.5" in msg
    assert d["w_want"] == "0.04" and d["w_argv"] == 0.1


def test_the_committed_provisional_K2_files_are_refused_whenever_R_is_far_from_1_and_pass_only_inside_the_tolerance_band():
    # the placeholder has w * R / 0.1 == R: it is admitted ONLY for R in [0.95, 1.05], where 0.1 IS the rule's value to within the tolerance
    for v in VARIANTS:
        assert LG.flag_values(K2_ARGV[v], "--w-r8-drivable") == ["0.1"]
        for R in ("0.2000", "0.5000", "0.9000", "1.1000", "2.5000", "10.0000"):
            r, _ = LG.wcost_reasons(LG.PROFILES["refcv9-e2"], K2_ARGV[v], [line(R)])
            assert len(r) == 1 and ("write --w-r8-drivable %s" % LG.wcost_weight_text(float(R))) in r[0], (v, R, r)
        for R in ("0.9500", "1.0000", "1.0500"):
            assert LG.wcost_reasons(LG.PROFILES["refcv9-e2"], K2_ARGV[v], [line(R)])[0] == [], (v, R)


@pytest.mark.parametrize("lines", [None, []])
def test_no_W_COST_line_with_the_critic_ON_FAILS_never_passes_silently(lines):
    argv = LG.set_flag(K2_ARGV["K2"], "--w-r8-drivable", ["0.04"])
    for prof in FAMILY:
        r, d = LG.wcost_reasons(LG.PROFILES[prof], argv, lines)
        assert r == ["G-LIVE W-COST: the critic never saw a labelled step in the smoke (no '[v3] refcv9 W-COST at step N: critic-BCE / trajectory-loss = R' line) -- "
                     "R is unmeasurable, so --w-r8-drivable cannot be held to 0.1 / R; a missing reading is a FAIL, never a pass"], r
        assert d["on"] is True and d["n_lines"] == 0


@pytest.mark.parametrize("v", VARIANTS)
def test_the_line_does_not_have_to_be_the_first_thing_the_smoke_printed_but_the_first_W_COST_line_decides(v):
    argv = LG.set_flag(K2_ARGV[v], "--w-r8-drivable", ["0.04"])
    assert LG.wcost_reasons(LG.PROFILES["refcv9-e2"], argv, [line("2.5000", step=3), line("9.0000", step=9)])[0] == []
    r, d = LG.wcost_reasons(LG.PROFILES["refcv9-e2"], argv, [line("9.0000", step=3), line("2.5000", step=9)])
    assert len(r) == 1 and "R = 9" in r[0] and d["step"] == 3 and d["n_lines"] == 2


# ============================================================== critic OFF: not entered ==
@pytest.mark.parametrize("prof", FAMILY)
@pytest.mark.parametrize("lines", [None, [], [line("2.5000")], ["W-COST garbage"]])
def test_K0_without_the_critic_is_NOT_ENTERED_whatever_the_smoke_printed(prof, lines):
    assert LG.has_flag(K0, "--r8-critic-drivable") is False and LG.has_flag(K0, "--w-r8-drivable") is False
    assert LG.wcost_in_scope(LG.PROFILES[prof], K0) is False
    assert LG.wcost_reasons(LG.PROFILES[prof], K0, lines) == ([], {})


@pytest.mark.parametrize("prof", ["refcv8", "refcv9-warm", "refcv9-e1rig", "refcv8-nt", "refcv8-mapft", "v8f"])
def test_every_other_profile_is_not_entered_even_with_the_critic_in_the_argv(prof):
    argv = LG.set_flag(K2_ARGV["K2"], "--w-r8-drivable", ["0.1"])
    assert LG.has_flag(argv, "--r8-critic-drivable")
    assert LG.wcost_in_scope(LG.PROFILES[prof], argv) is False
    assert LG.wcost_reasons(LG.PROFILES[prof], argv, None) == ([], {})
    assert LG.wcost_reasons(LG.PROFILES[prof], argv, [line("2.5000")]) == ([], {})


def test_the_family_list_is_exactly_the_profiles_built_by_the_full_family_builder():
    assert set(LG._R9F_WCOST_PROFILES) == set(FAMILY)
    assert {n for n, p in LG.PROFILES.items() if p.get("lr_schedule_rule")} == set(LG._R9F_WCOST_PROFILES)     # the builder's marker: a 4th member must join the list


# ============================================================== TOL: the edge and the arithmetic ==
def test_TOL_is_0_05_relative_and_the_edge_is_INCLUSIVE():
    assert LG.WCOST_TOL == 0.05 and LG.WCOST_TARGET == 0.1
    # |w * R / 0.1 - 1| == 0.05 exactly (to float noise) PASSES, a hair beyond FAILS -- on both sides, and for a non-trivial w
    for w, R in (("0.1", "1.0500"), ("0.1", "0.9500"), ("0.04", "2.6250"), ("0.04", "2.3750")):
        assert judge(R, w)[0] == [], (w, R)
    for w, R in (("0.1", "1.0501"), ("0.1", "0.9499"), ("0.04", "2.6260"), ("0.04", "2.3740")):
        r, d = judge(R, w)
        assert len(r) == 1 and d["deviation"] > 0.05, (w, R)


def test_the_composers_own_4_significant_digit_weight_always_passes_over_a_wide_range_of_R():
    for k in range(1, 400):
        R = round(0.013 * k * k / 40.0 + 0.01 * k, 4) or 0.0001
        w = LG.wcost_weight_text(R)
        r, d = judge("%.4f" % R, w)
        assert r == [] and d["deviation"] < 6e-4, (R, w, d)


@pytest.mark.parametrize("R", ["0.0000", "-1.0000", "nan", "inf"])
def test_an_unusable_ratio_is_a_FAIL_not_a_pass(R):
    r, _ = judge(R, 0.04)
    assert len(r) == 1 and "carries no finite positive ratio R" in r[0]


def test_a_line_with_the_key_but_no_parsable_ratio_is_a_FAIL():
    r, _ = LG.wcost_reasons(LG.PROFILES["refcv9-e2"], LG.set_flag(K2_ARGV["K2"], "--w-r8-drivable", ["0.04"]), ["[v3] refcv9 W-COST at step 0: oops"])
    assert len(r) == 1 and "carries no finite positive ratio R" in r[0]


@pytest.mark.parametrize("w", ["0", "0.0", "-0.04", "nan", "inf", "abc"])
def test_a_missing_or_nonpositive_weight_is_a_FAIL_naming_the_value_to_write(w):
    r, _ = judge("2.5000", w)
    assert len(r) == 1 and "--w-r8-drivable 0.04" in r[0] and "never supervised" in r[0]
    no_w = LG.set_flag(K2_ARGV["K2"], "--w-r8-drivable", None)
    r, _ = LG.wcost_reasons(LG.PROFILES["refcv9-e2"], no_w, [line("2.5000")])
    assert len(r) == 1 and "--w-r8-drivable 0.04" in r[0]


# ============================================================== the shim: records, passes through, restores ==
def test_the_shim_records_only_W_COST_lines_prints_everything_unchanged_and_never_raises(capsys):
    sink = []
    p = LG.wcost_print_shim(sink)
    p("[v3] step 1", flush=True)
    p(line("2.5000"), flush=True)
    p("a", "b", sep="|")

    class Bad:
        def __str__(self):
            raise RuntimeError("boom")
    with pytest.raises(RuntimeError):
        p(Bad())                                          # the observer swallows ITS failure; the real print then raises as it always did
    out = capsys.readouterr().out
    assert sink == [line("2.5000")] and "[v3] step 1\n" in out and line("2.5000") in out and "a|b\n" in out


FAKE_TRAINER_SRC = '''
class ResumableEpochSampler:
    SCHEME = "fake"
    def __init__(self):
        self.epoch, self.skip_batches, self.n, self.batch = 0, 0, 0, 1
    def __iter__(self):
        return iter(())
def build_optimizer(model, args):
    return None
def compute_losses_v3(model, batch, device, *a, **k):
    return {}
def next_train_batch(it, dl, sampler, dpos):
    return None
SAW_SHIM = []
def main(argv):
    SAW_SHIM.append("print" in globals())
    print("[v3] some other line", flush=True)
    print(WCOST_FMT % (0, 2.5, 0.6931, 0.2772, 120, 0.1), flush=True)
'''


def _fake_trainer():
    T = types.ModuleType("fake_refc_v3_train")
    T.__file__ = "fake_refc_v3_train.py"
    T.WCOST_FMT = FMT
    exec(compile(FAKE_TRAINER_SRC, "fake_refc_v3_train.py", "exec"), T.__dict__)      # the fake's `print` resolves through ITS module globals, exactly as the real trainer's does
    return T


def _ctx(profile, argv):
    return LG.Ctx(profile=profile, tree="", commit="", argv=list(argv), out_dir="", path_map=[], tree_sha256="", argv_sha256="", options={})


@pytest.mark.parametrize("prof", FAMILY)
def test_REACHABLE_run_smoke_records_the_trainers_own_W_COST_line_for_a_critic_argv_and_the_judge_reads_it(prof, capsys):
    pytest.importorskip("torch")
    argv = LG.set_flag(K2_ARGV["K2"], "--w-r8-drivable", ["0.04"])
    T = _fake_trainer()
    rec = LG.run_smoke(_ctx(prof, argv), T, list(argv), snapshot="final")
    assert T.SAW_SHIM == [True] and "print" not in T.__dict__                       # installed while main ran, restored after
    assert len(rec["wcost_lines"]) == 1 and rec["wcost_lines"][0] == FMT % (0, 2.5, 0.6931, 0.2772, 120, 0.1)
    assert FMT % (0, 2.5, 0.6931, 0.2772, 120, 0.1) in capsys.readouterr().out      # and the trainer's print still reached stdout
    assert LG.wcost_reasons(LG.PROFILES[prof], argv, rec.get("wcost_lines"))[0] == []
    bad = LG.set_flag(argv, "--w-r8-drivable", ["0.1"])
    r, _ = LG.wcost_reasons(LG.PROFILES[prof], bad, rec.get("wcost_lines"))
    assert len(r) == 1 and "write --w-r8-drivable 0.04" in r[0]


def test_REACHABLE_run_smoke_installs_nothing_for_K0_and_adds_no_key():
    pytest.importorskip("torch")
    T = _fake_trainer()
    rec = LG.run_smoke(_ctx("refcv9-e2", K0), T, list(K0), snapshot="final")
    assert T.SAW_SHIM == [False] and "wcost_lines" not in rec and "print" not in T.__dict__


def test_REACHABLE_the_G_LIVE_call_site_passes_the_smokes_lines_to_the_judge_and_its_reasons_are_G_LIVEs():
    # the G-LIVE function itself needs a real trainer smoke (no GPU / no data here): pin the three hops by source, once each
    assert LAUNCH_GATE_SRC.count('wcr_, wcd_ = wcost_reasons(ctx.prof, ctx.argv, run1.get("wcost_lines"))') == 1
    i = LAUNCH_GATE_SRC.index('wcr_, wcd_ = wcost_reasons(ctx.prof, ctx.argv, run1.get("wcost_lines"))')
    tail = LAUNCH_GATE_SRC[i:i + 400]
    assert "if wcd_:" in tail and 'det["wcost"] = wcd_' in tail and "reasons += wcr_" in tail
    j = LAUNCH_GATE_SRC.index("fpr_, fpd_ = full_profile_live_reasons(")
    assert j < i < j + 600                                                              # in the same G-LIVE block as the other full-family rows
    assert LAUNCH_GATE_SRC.count('pat.set(T, "print", wcost_print_shim(rec["wcost_lines"]))') == 1
    k = LAUNCH_GATE_SRC.index('pat.set(T, "print", wcost_print_shim(rec["wcost_lines"]))')
    assert LAUNCH_GATE_SRC.index("if wcost_in_scope(ctx.prof, ctx.argv):") < k < LAUNCH_GATE_SRC.index("T.main(list(argv))", k) < k + 600


# ============================================================== the composer stamps the measurement ==
def _k0_paths():
    return str(RUNS / "r9e2_K0.argv.json"), str(RUNS / "refcv8-r101-s0.argv.json")


def test_the_composer_stamps_ratio_and_source_with_w_cost_ratio_and_the_gate_accepts_the_file_it_wrote(tmp_path):
    out = tmp_path / "k2.json"
    assert MK.main(["--variant", "K2", "--k0", _k0_paths()[0], "--w-cost-ratio", "2.5", "--w-cost-source", "C:/runs/k2_explore/smoke.log", "--write", str(out)]) == 0
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["w_cost_ratio"] == 2.5 and d["w_cost_source"] == "C:/runs/k2_explore/smoke.log" and "w_cost" not in d
    assert LG.flag_values(d["argv"], "--w-r8-drivable") == ["0.04"]
    assert LG.wcost_reasons(LG.PROFILES["refcv9-e2"], d["argv"], [line("2.5000")])[0] == []                  # the regenerated file PASSES the W-COST row of its own smoke
    assert LG.wcost_reasons(LG.PROFILES["refcv9-e2"], d["argv"], [line("5.0000")])[0] != []                  # ...and not another smoke's


def test_without_a_ratio_the_composer_stamps_provisional_and_the_file_is_refused_by_the_gate_when_R_is_far_from_1(tmp_path):
    out = tmp_path / "k2p.json"
    assert MK.main(["--variant", "K2", "--k0", _k0_paths()[0], "--w-r8-drivable", "0.1", "--provisional", "--write", str(out)]) == 0
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["w_cost"] == "provisional" and "w_cost_ratio" not in d and "w_cost_source" not in d
    r, _ = LG.wcost_reasons(LG.PROFILES["refcv9-e2"], d["argv"], [line("2.5000")])
    assert len(r) == 1 and "write --w-r8-drivable 0.04" in r[0]                                              # an explore smoke on the provisional file reports R in its FAIL


def test_the_stamp_is_applied_by_main_only_so_the_committed_files_regenerate_byte_equal_and_K0_is_untouched():
    k0b = (RUNS / "r9e2_K0.argv.json").read_bytes()
    for v in VARIANTS:
        d = json.loads((RUNS / f"r9e2_{v}.argv.json").read_text(encoding="utf-8"))
        w = float(LG.flag_values(d["argv"], "--w-r8-drivable")[0])
        got = MK.dump(MK.variant_doc(json.loads(k0b.decode("utf-8")), k0b, v, w, provisional=True,
                                     w_source="operator-supplied PROVISIONAL placeholder (no W-COST reading exists yet)")).encode("utf-8")
        assert got == (RUNS / f"r9e2_{v}.argv.json").read_bytes() and "w_cost" not in d
    k0_path, base_path = _k0_paths()
    assert MK.main(["--check-k0", "--base", base_path, "--k0", k0_path]) == 0                                # K0 regenerates BYTE-EQUAL
    assert MK.dump(MK.regen_k0(json.loads(Path(base_path).read_text(encoding="utf-8")))).encode("utf-8") == k0b


def test_the_stamp_refuses_a_ratio_without_a_source_a_source_without_a_ratio_and_a_bad_ratio(tmp_path):
    k0 = _k0_paths()[0]
    for extra in (["--w-cost-ratio", "2.5"], ["--w-r8-drivable", "0.1", "--w-cost-source", "x.log"], ["--w-cost-ratio", "2.5", "--w-cost-source", "  "]):
        with pytest.raises(SystemExit):
            MK.main(["--variant", "K2", "--k0", k0, "--write", str(tmp_path / "x.json"), *extra])
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(SystemExit):
            MK.stamp_w_cost({}, bad, "x.log")
    assert not (tmp_path / "x.json").exists()
    assert MK.stamp_w_cost({"a": 1}, None, None) == {"a": 1, "w_cost": "provisional"}
    assert MK.stamp_w_cost({}, 2.5, "x.log") == {"w_cost_ratio": 2.5, "w_cost_source": "x.log"}


def test_the_message_is_the_one_the_runbook_relies_on_R_then_the_composer_command():
    r, _ = judge("2.5000", 0.1)
    m = re.search(r"write --w-r8-drivable (\S+) \(= 0\.1 / R at 4 significant digits; r9_e2_make_argv\.py --w-cost-ratio (\S+)\)", r[0])
    assert m and m.group(1) == "0.04" and float(m.group(2)) == 2.5
