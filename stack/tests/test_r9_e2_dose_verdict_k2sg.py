"""Tests for `r9_e2_dose_verdict.py --arm K2SG` (PREREG_K2SG_2026-11-19.md sections 2-4: the map->trunk stop-gradient arm).  CPU only, no scipy.

  (a) the K2DC / K2DC02 documents are byte-identical to the tip module's for a fixed synthetic input (golden sha256 of the JSON + md; optionally also compared live to the module
      named by R9_DOSE_TIP_MODULE);
  (b) all trunk readings exactly 0.0 and tac_v6 non-zero => valid;  (c) one reading 1e-12 => OP-VOID naming the step;  (d) no gs rows => OP-VOID;  (e) tac_v6 0.0 or missing => OP-VOID;
  (f) the argv guard REFUSES a doubled flag, 0.25, a share target;  (g) the real landed r9e2_K2SG.argv.json is accepted;
  (h) MUTANTS of the reader's own source (the exact-zero identity made a tolerance, the control dropped, the no-rows check dropped, ...): each must turn a check RED.
      By hand: R9_DOSE_UNDER_TEST=<mutated copy> pytest stack/tests/test_r9_e2_dose_verdict_k2sg.py -k "not mutant".
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np  # noqa: F401  (the dump helper needs it)
import pytest

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parents[0] / "scripts"
RUNS_D = HERE.parents[0] / "ops" / "runs.d"
for p in (str(SCRIPTS), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
SRC = Path(os.environ.get("R9_DOSE_UNDER_TEST") or SCRIPTS / "r9_e2_dose_verdict.py")


def _load(path, name="_dose_sg_under_test"):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


D = _load(SRC)


# ---------------------------------------------------------------------------------------------------------------- #
# (a) the K2DC / K2DC02 outputs did not move                                                                         #
# ---------------------------------------------------------------------------------------------------------------- #
def _row(delta, sep=False):
    return {"delta": delta, "lo": delta - 0.001, "hi": delta + 0.001, "separated": sep, "better": "lower", "n": 100, "n_episodes": 10, "a_mean": 0.1, "b_mean": 0.1 - delta}


def _seed():
    rows = {"LAT_cross_mae_m": _row(-0.004), "LAT_cross_abs_final_m": _row(-0.004)}
    fam = {"LONGITUDINAL": {"LON_x": _row(0.001)}, "LATERAL": dict(rows), "TACTICAL": {"TAC_x": _row(0.0)}, "STRATEGIC": {"status": "NOT-SUPPLIED", "rows": {}}}
    return {"dc_vs_k0": {"LAT_cross_mae_m": _row(0.001), "LAT_cross_abs_final_m": _row(0.0)}, "dc_vs_k2": rows, "jitter_dlag_dc_vs_k2": _row(-0.0005),
            "dc_vs_k0_by_family": fam, "dc_vs_k2_by_family": fam}


def _outputs(m):
    import hashlib
    nums = {"0": _seed(), "1": _seed()}
    void = {"valid": True, "status": "HELD", "rule": "r", "n_inside_band": 2, "n_probes_step_ge_3000": 3}
    arm = {"arm": "K2DC02", "argv_json": "x.json", "target": 0.05, "bounds": [0.1, 25.0], "share_band": [0.025, 0.075]}
    res = {}
    for name, a in (("K2DC", None), ("K2DC02", arm)):
        doc = m.build(nums, void, None, a)
        res[name] = hashlib.sha256((json.dumps(doc, indent=1, sort_keys=True, default=float) + "\x00" + m.md(doc)).encode("utf-8")).hexdigest()
    return res


# sha256 of the module's JSON + md for the input above.  RE-PINNED when STRATEGIC went PAUSED -> NOT-SUPPLIED / REPORT-ONLY (PI 2026-11-16): the previous pins were
# K2DC 0c2aa628...d52ce, K2DC02 cd726326...aa5aa (tip 7decfcc9 / 16ef6ff1).  A one-off check showed the OLD and NEW modules agree byte-for-byte once the STRATEGIC keys / lines are removed.
GOLDEN = {"K2DC": "fb52752d2fadc7913c51090105eaa87e51c7689cb8139895781732d334221a78", "K2DC02": "32c90fab9a5a3e1fc5e2e3cb8ab65c08ce094a812bbab4d514ef16f89d6c0aaf"}


def test_k2dc_and_k2dc02_outputs_are_unchanged_golden():
    assert _outputs(D) == GOLDEN


@pytest.mark.skipif(not os.environ.get("R9_DOSE_TIP_MODULE"), reason="R9_DOSE_TIP_MODULE (the tip's r9_e2_dose_verdict.py from an export) not set")
def test_k2dc_and_k2dc02_outputs_equal_the_tip_module_live():
    tip = _load(os.environ["R9_DOSE_TIP_MODULE"], "_dose_tip")
    import copy

    def stripped(m):                       # everything except the STRATEGIC report (intentionally changed 2026-11-16)
        nums = {"0": _seed(), "1": _seed()}
        void = {"valid": True, "status": "HELD", "rule": "r", "n_inside_band": 2, "n_probes_step_ge_3000": 3}
        doc = copy.deepcopy(m.build(copy.deepcopy(nums), void, None, None))
        for k in ("STRATEGIC", "families_note"):
            doc.pop(k)
        for s in doc["seeds"].values():
            s["dc_vs_k0_by_family"].pop("STRATEGIC", None)
            s["dc_vs_k2_by_family"].pop("STRATEGIC", None)
        return json.dumps(doc, sort_keys=True, default=float), [x for x in m.md(m.build(copy.deepcopy(nums), void, None, None)).split("\n") if x.strip() and "STRATEGIC" not in x]
    assert stripped(D) == stripped(tip)


# ---------------------------------------------------------------------------------------------------------------- #
# the validity identity (PREREG_K2SG section 2)                                                                      #
# ---------------------------------------------------------------------------------------------------------------- #
def gs_row(step, trunk=0.0, tac=0.31, **kw):
    r = {"step": step, "loss": 1.0}
    if trunk is not None:
        r["gs_trunk_proj_map_hires"] = trunk
    if tac is not None:
        r["gs_trunk_proj_tac_v6"] = tac
    r.update(kw)
    return r


def write_metrics(tmp_path, rows, name="metrics.jsonl"):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


GOOD = [gs_row(1000, gs_layer2_proj_map_hires=0.0, gs_bev025_proj_map_hires=0.02, gs_bev_pool_proj_map_hires=0.0), gs_row(3000), gs_row(5000)]


def test_b_all_zero_with_nonzero_control_is_valid(tmp_path):
    v = D.void_check_sg(write_metrics(tmp_path, GOOD + [{"step": 250, "loss": 2.0}]))        # a row without gs keys is not a reading
    assert v["valid"] is True and v["status"] == "VALID" and v["n_rows"] == 3 and v["n_trunk_nonzero"] == 0 and v["n_control_bad"] == 0
    info = v["informational_not_in_rule"]
    assert info["gs_layer2_proj_map_hires"]["n_exactly_zero"] == 1 and info["gs_bev025_proj_map_hires"]["n_exactly_zero"] == 0       # informational, counted, never ruled on
    assert info["gs_bev_pool_proj_map_hires"]["n_rows"] == 3


def test_b2_negative_zero_and_int_zero_are_exactly_zero(tmp_path):
    assert D.void_check_sg(write_metrics(tmp_path, [gs_row(1000, trunk=-0.0), gs_row(3000, trunk=0)]))["valid"] is True


def test_c_one_tiny_nonzero_reading_is_op_void_naming_its_step(tmp_path):
    v = D.void_check_sg(write_metrics(tmp_path, [gs_row(1000), gs_row(3000, trunk=1e-12), gs_row(5000)]))
    assert v["valid"] is False and v["status"] == "OP-VOID" and v["trunk_nonzero"] == [{"step": 3000, "value": 1e-12}]


def test_c2_nan_and_string_readings_are_not_zero(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text('{"step": 1000, "gs_trunk_proj_map_hires": NaN, "gs_trunk_proj_tac_v6": 0.3}\n{"step": 3000, "gs_trunk_proj_map_hires": "0.0", "gs_trunk_proj_tac_v6": 0.3}\n', encoding="utf-8")
    v = D.void_check_sg(p)
    assert v["valid"] is False and v["n_trunk_nonzero"] == 1                                   # NaN is non-finite => non-zero; the string "0.0" parses to 0.0 (a log writer's choice)


def test_d_no_gs_rows_is_op_void(tmp_path):
    v = D.void_check_sg(write_metrics(tmp_path, [{"step": 250, "loss": 1.0}, {"step": 500, "loss": 0.9}]))
    assert v["valid"] is False and v["status"] == "OP-VOID" and v["n_rows"] == 0
    only_step0 = D.void_check_sg(write_metrics(tmp_path, [gs_row(0)], "m0.jsonl"))              # step >= 1 only: a step-0 reading is not a reading
    assert only_step0["valid"] is False and only_step0["n_rows"] == 0


@pytest.mark.parametrize("bad", [0.0, None], ids=["control_zero", "control_missing"])
def test_e_control_zero_or_missing_is_op_void(tmp_path, bad):
    v = D.void_check_sg(write_metrics(tmp_path, [gs_row(1000), gs_row(3000, tac=bad), gs_row(5000)]))
    assert v["valid"] is False and v["status"] == "OP-VOID" and v["n_trunk_nonzero"] == 0 and v["control_bad"] == [{"step": 3000, "value": bad}]


def test_missing_or_unreadable_metrics_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        D.void_check_sg(None)
    with pytest.raises(SystemExit):
        D.void_check_sg(tmp_path / "nope.jsonl")


# ---------------------------------------------------------------------------------------------------------------- #
# the argv guard                                                                                                     #
# ---------------------------------------------------------------------------------------------------------------- #
def argv_json(tmp_path, extra, name="a.argv.json"):
    p = tmp_path / name
    p.write_text(json.dumps({"schema": "r9e2-argv-v1", "argv": ["--arm", "hier", "--seed", "0", *extra]}), encoding="utf-8")
    return p


def test_f_argv_guard(tmp_path):
    ok = D.arm_sg("K2SG", argv_json(tmp_path, ["--map-hires-trunk-grad-scale", "0.0"]))
    assert ok["arm"] == "K2SG" and ok["kind"] == "stop-gradient" and ok["trunk_grad_scale"] == 0.0
    assert D.arm_sg("K2SG", argv_json(tmp_path, ["--map-hires-trunk-grad-scale", "0"], "b.json"))["trunk_grad_scale"] == 0.0
    refused = {"twice": ["--map-hires-trunk-grad-scale", "0.0", "--map-hires-trunk-grad-scale", "0.0"],
               "0.25": ["--map-hires-trunk-grad-scale", "0.25"],
               "absent": [],
               "no value": ["--map-hires-trunk-grad-scale"],
               "not a number": ["--map-hires-trunk-grad-scale", "off"],
               "share target": ["--map-hires-trunk-grad-scale", "0.0", "--map-hires-share-target", "0.1"],
               "equals form": ["--map-hires-trunk-grad-scale=0.0"],
               "1e-12": ["--map-hires-trunk-grad-scale", "1e-12"]}
    for i, (label, extra) in enumerate(refused.items()):
        with pytest.raises(SystemExit, match="REFUSED"):
            D.arm_sg("K2SG", argv_json(tmp_path, extra, f"r{i}.json"))
    with pytest.raises(SystemExit):
        D.arm_sg("K2SG", tmp_path / "missing.json")


def test_g_the_landed_k2sg_argv_is_accepted():
    p = RUNS_D / "r9e2_K2SG.argv.json"
    assert p.is_file(), p
    arm = D.arm_sg("K2SG")                                                                      # the default path, read from stack/ops/runs.d
    assert arm["trunk_grad_scale"] == 0.0 and Path(arm["argv_json"]).resolve() == p.resolve()
    toks = json.loads(p.read_text(encoding="utf-8"))["argv"]
    assert toks.count("--map-hires-trunk-grad-scale") == 1 and "--map-hires-share-target" not in toks


# ---------------------------------------------------------------------------------------------------------------- #
# the decision and the wording                                                                                       #
# ---------------------------------------------------------------------------------------------------------------- #
def _nums():
    return {"0": _seed(), "1": _seed()}


def test_build_valid_void_and_wording():
    arm = {"arm": "K2SG", "kind": "stop-gradient", "argv_json": "x", "trunk_grad_scale": 0.0}
    ok = D.build(_nums(), {"valid": True, "status": "VALID"}, None, arm)
    assert ok["outcome"] == "SUPPORTED" and ok["arm"]["arm"] == "K2SG" and "op_validity" in ok and "dose_validity" not in ok
    assert ok["prereg"].endswith("PREREG_K2SG_2026-11-19.md") and ok["STRATEGIC"]["status"] == "NOT-SUPPLIED"
    assert "K2DC" not in json.dumps({k: v for k, v in ok.items() if k not in ("seeds", "arm")})   # the arm label is K2SG everywhere (the `arm.note` legitimately cites PREREG_K2DC's constants)
    assert "K2DC" not in D.md(ok)
    void = D.build(_nums(), {"valid": False, "status": "OP-VOID"}, None, arm)
    assert void["outcome"] == "OP-VOID" and void["outcome_if_valid"] == "SUPPORTED"
    assert D.EXIT["OP-VOID"] == 3
    refuted = {"0": _seed(), "1": _seed()}
    for s in refuted.values():
        s["dc_vs_k2"] = {"LAT_cross_mae_m": _row(0.0002), "LAT_cross_abs_final_m": _row(-0.0003)}
    r = D.build(refuted, {"valid": True}, None, arm)
    assert r["outcome"] == "REFUTED" and "ELIMINATED as the lateral cause" in r["reason"] and "--r9-hist-supervise-geom" in r["reason"] and "L2-SP" in r["reason"]
    inc = {"0": _seed(), "1": _seed()}
    for s in inc.values():
        s["dc_vs_k2"] = {"LAT_cross_mae_m": _row(0.002), "LAT_cross_abs_final_m": _row(0.002)}
    assert "seed + 1" in D.build(inc, {"valid": True}, None, arm)["reason"]


# ---------------------------------------------------------------------------------------------------------------- #
# end to end on synthetic dumps (the real reader)                                                                    #
# ---------------------------------------------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def fam_mod():
    import test_r9_e2_families as T
    return T


def _pf(T, root, noise):
    for s, sd in zip((0, 1), (11, 12)):
        T.write_dump(root / f"dump_s{s}", T.DEV_CLIPS + T.HOLD_CLIPS, noise, sd, sparse=True)
    return root


def _split(T, tmp_path):
    p = tmp_path / "DEV_HOLD_SPLIT.json"
    p.write_text(json.dumps({"DEV": {"sha12": [T.sha12(c) for c in T.DEV_CLIPS]}, "HOLD": {"sha12": [T.sha12(c) for c in T.HOLD_CLIPS]}}), encoding="utf-8")
    return p


def _cli(fam_mod, tmp_path, metrics_rows, argv_extra=()):
    T = fam_mod
    pf = {n: _pf(T, tmp_path / n, z) for n, z in (("k2sg", 0.010), ("k2", 0.040), ("k0", 0.010))}
    m = write_metrics(tmp_path, metrics_rows)
    out = tmp_path / "out"
    rc = D.main(["--arm", "K2SG", "--k2dc-pf", str(pf["k2sg"]), "--k2-pf", str(pf["k2"]), "--k0-pf", str(pf["k0"]), "--dev-split", str(_split(T, tmp_path)), "--metrics", str(m),
                 "--boot", "200", "--out-dir", str(out), *argv_extra])
    return rc, out


def test_cli_k2sg_valid_supported(fam_mod, tmp_path):
    rc, out = _cli(fam_mod, tmp_path, GOOD)
    assert rc == 0
    doc = json.loads((out / "DOSE_VERDICT.json").read_text(encoding="utf-8"))
    assert doc["outcome"] == "SUPPORTED" and doc["op_validity"]["status"] == "VALID" and doc["arm"]["arm"] == "K2SG"
    text = (out / "DOSE_VERDICT.md").read_text(encoding="utf-8")
    assert "K2SG" in text and "K2DC" not in text and "STRATEGIC" in text and "op_validity" in text


def test_cli_k2sg_invalid_is_op_void_exit_3(fam_mod, tmp_path):
    rc, out = _cli(fam_mod, tmp_path, [gs_row(1000), gs_row(3000, trunk=1e-12)])
    assert rc == 3
    doc = json.loads((out / "DOSE_VERDICT.json").read_text(encoding="utf-8"))
    assert doc["outcome"] == "OP-VOID" and doc["outcome_if_valid"] == "SUPPORTED" and doc["op_validity"]["trunk_nonzero"][0]["step"] == 3000


def test_cli_k2sg_refuses_a_bad_argv_and_a_missing_metrics(fam_mod, tmp_path):
    bad = argv_json(tmp_path, ["--map-hires-trunk-grad-scale", "0.25"], "bad.json")
    rc, _ = _cli(fam_mod, tmp_path, GOOD, ("--argv-json", str(bad)))
    assert rc == 4
    with pytest.raises(SystemExit):
        D.void_check_sg(None)


# ---------------------------------------------------------------------------------------------------------------- #
# (h) deliberate-regression mutants of the reader's own source                                                        #
# ---------------------------------------------------------------------------------------------------------------- #
def _suite(mod, tmp_path):
    """The validity + argv checks that must hold on a correct reader; raises AssertionError / SystemExit-expectation failures on a wrong one."""
    def vc(rows, name):
        return mod.void_check_sg(write_metrics(tmp_path, rows, name))
    assert vc(GOOD, "g.jsonl")["valid"] is True
    c = vc([gs_row(1000), gs_row(3000, trunk=1e-12)], "c.jsonl")
    assert c["valid"] is False and c["trunk_nonzero"][0]["step"] == 3000                           # (c)
    assert vc([{"step": 250, "loss": 1.0}], "d.jsonl")["valid"] is False                          # (d)
    assert vc([gs_row(0)], "s0.jsonl")["valid"] is False                                          # a step-0 reading is not a reading: no rows => OP-VOID
    assert vc([gs_row(1000, tac=0.0)], "e0.jsonl")["valid"] is False                              # (e) zero control
    assert vc([gs_row(1000, tac=None)], "e1.jsonl")["valid"] is False                             # (e) missing control
    assert vc([gs_row(1000), gs_row(3000, trunk=0.5)], "big.jsonl")["valid"] is False
    with pytest.raises(SystemExit):
        mod.arm_sg("K2SG", argv_json(tmp_path, ["--map-hires-trunk-grad-scale", "0.25"], "m1.json"))
    with pytest.raises(SystemExit):
        mod.arm_sg("K2SG", argv_json(tmp_path, ["--map-hires-trunk-grad-scale", "0.0", "--map-hires-share-target", "0.1"], "m2.json"))
    with pytest.raises(SystemExit):
        mod.arm_sg("K2SG", argv_json(tmp_path, ["--map-hires-trunk-grad-scale", "0.0", "--map-hires-trunk-grad-scale", "0.0"], "m3.json"))


MUTANTS = {
    "exact zero made a tolerance (1e-6)": ("if v is None or v != 0.0:", "if v is None or abs(v) > 1e-6:"),
    "control check dropped": ("if c is None or c == 0.0:", "if False:"),
    "control zero allowed": ("if c is None or c == 0.0:", "if c is None:"),
    "no-rows check dropped": ("if not rows and not nonzero:", "if False:"),
    "step-0 rows counted": ("        if st < 1:\n            continue\n", ""),
    "scale 0.25 accepted": ("    if v != 0.0:\n        _fail(f\"{p}: {SG_FLAG} = {v!r}, exactly 0.0 is required\")", "    if v > 0.5:\n        _fail(f\"{p}: x\")"),
    "share target allowed": ("    if SG_SHARE_FLAG in toks:", "    if False:"),
    "duplicate flag allowed": ("    if n != 1:", "    if n < 1:"),
}


def test_pristine_reader_passes_the_suite(tmp_path):
    _suite(_load(SRC, "_dose_sg_pristine"), tmp_path)


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_mutant_is_caught(name, tmp_path):
    src = (SCRIPTS / "r9_e2_dose_verdict.py").read_text(encoding="utf-8")
    old, new = MUTANTS[name]
    assert src.count(old) == 1, f"the mutation site for {name!r} is not unique / not there"
    p = tmp_path / "mutant_dose_sg.py"
    p.write_text(src.replace(old, new), encoding="utf-8")
    mut = _load(p, "_dose_sg_mutant")
    with pytest.raises((AssertionError, SystemExit, pytest.fail.Exception)):
        _suite(mut, tmp_path)
    _suite(_load(SCRIPTS / "r9_e2_dose_verdict.py", "_dose_sg_pristine2"), tmp_path)           # and the pristine source passes the same suite
