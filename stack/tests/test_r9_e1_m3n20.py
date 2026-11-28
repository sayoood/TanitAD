"""E1-M3N20 (PREREG_M3, stamped 2026-11-07 01:31:44Z, md5 8e497d34): arm M3 = BASE + the map's OWN 128-channel BEV encoder at ``--w-map-hires 1.0``, where BASE is
M1 (near band 0-10 m) -> arm **M3**, or M1N20 (near band 0-20 m, near lift 20) -> arm **M3N20**. Both are built BEFORE M1N20 is read; only the selected one trains.

The class this file exists for is CLAUDE.md's "built, tested, and UNREACHABLE FROM ITS CALLER": every lever below already existed (``--r9-map-own-enc`` = C2,
``--r9-map-banded-near-m`` = E1-NEAR20); what did NOT exist is their COMBINATION as an arm, M3's committed argv, or a verdict row whose reference is right. So:

  (1) the arm: the committed argv files are the TOOL's output and differ from their neighbours in exactly the lever tokens;
  (2) the gate admits both and nothing else (its own rules + its own /2-banded record, held to near lift 20 AND the own-encoder stamp);
  (3) the consumers of an arm-name list follow (record-kind table, bind script/template, harness, box replay);
  (4) the verdict row: decisive ref M1 / M1N20, CARRY vs C0, M1's metrics, the PREREG's committed readings, a CLI end to end;
  (5) REACHABILITY from the generated argv, through the trainer's own parser: a 2 x 2 (own encoder x near band) of built models, then one backward whose gradient is
      NON-ZERO on the own encoder and on near rows 100-199, against controls that read EXACTLY 0.0 (and the argvs that must not build them).
"""
from __future__ import annotations

import dataclasses as dc
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import launch_gate as LG  # noqa: E402
import r9_e1_make_argv as MK  # noqa: E402
import r9_e1_verdict as V  # noqa: E402
import refc_v3_train as T  # noqa: E402
import test_launch_gate_e1rig_own_record as OR  # noqa: E402
import test_map_only_ft as R  # noqa: E402
import test_r9_e1_verdict as TV  # noqa: E402
from tanitad.data.rig_projection import RigCamera  # noqa: E402
from tanitad.data.semantic_map_gt_fine import EXTENT_V2  # noqa: E402
from tanitad.models import map_banded as MB  # noqa: E402
from tanitad.models import map_head_hires as H  # noqa: E402
from tanitad.models.trunk_shapes import frame_for_width  # noqa: E402
from tanitad.refs import refc_v3 as v3  # noqa: E402
from tanitad.train import map_banded_flag as MBF  # noqa: E402
from tanitad.train import map_own_enc_flag as MOE  # noqa: E402

RUNS = ROOT / "ops" / "runs.d"
RIG = LG.PROFILES["refcv9-e1rig"]
BD, MO = RIG["map_overfit_banded"], RIG["map_overfit"]
COMMIT = "ef" * 20


def _file(arm):
    return json.loads((RUNS / f"r9e1_{arm}.argv.json").read_text(encoding="utf-8"))


def _argv(arm):
    return list(_file(arm)["argv"])


C0 = _argv("C0")
ARMS = ("M1", "M3", "M1N20", "M3N20")
OWN = {"M1": False, "M3": True, "M1N20": False, "M3N20": True}
NEAR = {"M1": 10, "M3": 10, "M1N20": 20, "M3N20": 20}
ROWS = {10: 100, 20: 200}


def _diff(a, b):
    da, db = dict(LG.flag_pairs(a)), dict(LG.flag_pairs(b))
    return {f for f in set(da) | set(db) if da.get(f) != db.get(f)}


# =========================================================================== #
# (1) the arm                                                                  #
# =========================================================================== #
def test_the_arm_tables_and_the_weight_rule():
    assert MK.OWN_ENC_ARMS == ("M3", "M3N20") and set(MK.NEAR20_ARMS) == {"M1N20", "M3N20"} and MK.MAP_ARMS == {"M1": 1.0, "M2a": 0.511, "M2b": 1.018}
    assert "w_map_hires" in MK.NEAR20_ARMS["M1N20"] and "w_map_hires" not in MK.NEAR20_ARMS["M3N20"]       # M3N20 takes M3's weight: operator-supplied
    assert MK.banded_near_lift_m("M3N20") == "20" and MK.banded_near_lift_m("M3") == "10" and MK.banded_near_lift_m("M1") == "10"
    assert MK.fixed_map_weight("M1N20") == 1.0 and MK.fixed_map_weight("M3") is None and MK.fixed_map_weight("M3N20") is None
    for arm in ("M3", "M3N20"):
        with pytest.raises(SystemExit, match="needs --w-map-hires"):
            MK.map_arm_argv(C0, arm)                                                    # the weight is REQUIRED
        with pytest.raises(SystemExit, match="must be > 0"):
            MK.map_arm_argv(C0, arm, w=0.0)
    for arm in ("M1", "M2a", "M2b", "M1N20"):
        with pytest.raises(SystemExit, match="M3 only"):
            MK.map_arm_argv(C0, arm, w=0.5)                                            # the fixed-weight arms refuse a weight


@pytest.mark.parametrize("arm", ["M3", "M3N20"])
def test_the_committed_files_are_the_tools_output_with_w_1_and_its_lever_block(arm):
    f = _file(arm)
    assert f["arm"] == f"r9e1_{arm}" and f["argv"] == MK.map_arm_argv(C0, arm, w=1.0) and f["seed"] == 0 and f["derived_from"] == "r9e1_C0"
    assert f["lever"]["own_enc_flag"] == "--r9-map-own-enc" and f["lever"]["w_map_hires"] == 1.0 and "no M2 winner" in f["lever"]["w_map_hires_source"]
    assert f["lever"]["near_lift_m"] == float(NEAR[arm])
    raw = (RUNS / f"r9e1_{arm}.argv.json").read_bytes()
    assert b"\r" not in raw and raw.endswith(b"}\n")
    d = dict(LG.flag_pairs(f["argv"]))
    assert d["--w-map-hires"] == ["1.0"] and d["--seed"] == ["0"] and d["--out"] == [f"/home/nvidia/r8ns/r9e1/runs/r9e1-{arm}"]
    assert d["--map-hires-near-lift-m"] == [str(NEAR[arm])] and d["--map-hires-near-refine-blocks"] == ["1"]
    toks = [t for t in f["argv"] if t.startswith("--")]
    assert toks.index("--r9-map-own-enc") == toks.index("--r9-map-banded") + 1                    # the own encoder right after the banded flag
    assert ("--r9-map-banded-near-m" in f["argv"]) == (arm == "M3N20")


def test_M3N20_differs_from_each_neighbour_in_exactly_its_lever_tokens():
    m1, m3, m1n20, m3n20 = (_argv(a) for a in ARMS)
    assert _diff(m3n20, m3) == {"--r9-map-banded-near-m", "--map-hires-near-lift-m", "--out"}              # M3 -> M3N20: the near band, nothing else
    assert _diff(m3n20, m1n20) == {"--r9-map-own-enc", "--out"}                                          # M1N20 -> M3N20: the own encoder, nothing else
    assert _diff(m3, m1) == {"--r9-map-own-enc", "--out"}                                                # M1 -> M3: the own encoder, nothing else
    assert _diff(m3n20, m1) == {"--r9-map-own-enc", "--r9-map-banded-near-m", "--map-hires-near-lift-m", "--out"}
    assert _diff(m3n20, C0) == {"--r9-map-banded", "--r9-map-own-enc", "--r9-map-banded-near-m", "--out"}      # C0 has lift 20 and w 1.0 already
    strip = lambda a, drop: [t for i, t in enumerate(a) if t not in drop and not (i and a[i - 1] in drop and a[i - 1] != "--r9-map-own-enc")]    # noqa: E731
    assert strip(m3n20, ("--r9-map-banded-near-m",)).count("--r9-map-own-enc") == 1


@pytest.mark.parametrize("arm", ["M3", "M3N20"])
def test_post_parse_pins_accept_the_arm_and_refuse_every_wrong_variant(arm):
    argv = _argv(arm)
    MK.check_map_arm_pins(argv, arm)                                                  # the trainer's own argparse + banded + near-end + own-encoder pins
    other = "M3N20" if arm == "M3" else "M3"
    with pytest.raises(SystemExit):
        MK.check_map_arm_pins(argv, other)                                            # the near band is the arm's: 10 vs 20, both ways
    no_own = [t for t in argv if t != "--r9-map-own-enc"]
    with pytest.raises(SystemExit, match="own-enc"):
        MK.check_map_arm_pins(no_own, arm)                                            # the own encoder dropped
    with pytest.raises(SystemExit, match="own-enc"):
        MK.check_map_arm_pins(_argv("M1N20" if arm == "M3N20" else "M1"), arm)         # the BASE offered as the arm
    with pytest.raises(SystemExit):
        MK.check_map_arm_pins([t for t in argv if t != "--r9-map-banded"], arm)        # the own encoder without the banded head


def test_cli_writes_the_committed_bytes_and_M3N20_needs_the_weight(tmp_path, monkeypatch):
    monkeypatch.setenv("MSYS_NO_PATHCONV", "1")
    c0f = RUNS / "r9e1_C0.argv.json"
    for arm in ("M3", "M3N20"):
        out = tmp_path / f"r9e1_{arm}.argv.json"
        with pytest.raises(SystemExit):
            MK.main(["--base", str(c0f), "--map-arm", arm, "--write", str(out)])
        assert MK.main(["--base", str(c0f), "--map-arm", arm, "--w-map-hires", "1.0", "--write", str(out)]) == 0
        assert out.read_bytes() == (RUNS / f"r9e1_{arm}.argv.json").read_bytes()


def test_UNSET_identity_the_other_arm_files_are_the_tools_output_unchanged():
    for name in ("M1", "M2a", "M2b", "M1N20"):
        assert MK.map_arm_argv(C0, name) == _argv(name)                                # M1 / M2a / M2b / M1N20: byte-equal to the tip's
    assert MK.map_arm_argv(C0, "M3", w=0.511)[MK.map_arm_argv(C0, "M3", w=0.511).index("--r9-map-banded") + 1] == "--r9-map-own-enc"


# =========================================================================== #
# (2) the gate                                                                 #
# =========================================================================== #
def _rec(argv, *, own, near, **over):
    """A /2-banded record as `map_hires_overfit.py --banded` writes it for this build: the built branch config / parameter breakdown are the REAL branch's."""
    cfg = H.MapHiresConfig(w_map_hires=1.0, x_max_m=100.0, y_half_m=30.0, near_lift_x_m=float(near), near_refine_blocks=1, banded=True,
                           banded_near_m=float(near), own_enc=own)
    br = H.MapHiresBranch(cfg, d_image=512, image_hw=(52, 128))
    gate_rows = {10: {"near": 100, "mid": 40, "far": 0}, 20: {"near": 200, "mid": 0, "far": 0}}[near]        # literals, not the gate's own table
    kw = dict(near_lift_m=float(near), bands=[{"name": n, "gate_rows": r} for n, r in gate_rows.items()], branch_config=cfg.as_dict())
    if own:
        kw.update(own_enc=True, own_enc_source="launch_argv", branch_params=br.param_breakdown())
    kw.update(over)
    return OR._record(argv, **kw), br


def _judge(tmp_path, rec, argv):
    return OR._judge(tmp_path, rec, argv)


def test_GREEN_M3N20_and_M3_are_admitted_by_the_argv_rules_and_their_own_record_closes_the_item(tmp_path):
    for arm in ("M3", "M3N20"):
        refusals, pend = LG.profile_argv_rules(RIG, _argv(arm), {}, [])
        assert [r for r in refusals if "not readable on this host" not in r and "missing on this host" not in r] == [] and pend == [], arm
    on, det = LG.required_on_reasons(RIG, _argv("M3N20"))
    assert det["required_values"]["--map-hires-near-lift-m"] == "20" and not [r for r in on if r.startswith(("--map-hires-near-lift-m", "--w-map-hires"))]
    assert LG.required_on_reasons(RIG, _argv("M3"))[1]["required_values"]["--map-hires-near-lift-m"] == "10"
    t20 = LG.banded_table_for(BD, _argv("M3N20"), MO)
    assert t20["near_lift_m"] == 20.0 and t20["gate_rows"] == {"near": 200, "mid": 0, "far": 0} and t20["own_enc"] == BD["own_enc"]
    assert LG.banded_table_for(BD, _argv("M3"), MO) is BD
    for arm, near in (("M3N20", 20), ("M3", 10)):
        argv = _argv(arm)
        rec, br = _rec(argv, own=True, near=near)
        assert rec["branch_params"]["map_encoder"] == 1_624_832 and br.cfg.d_map_enc == 128 and br.banded.geom[0].rows == ROWS[near]
        assert _judge(tmp_path, rec, argv) == [], (arm, _judge(tmp_path, rec, argv))
        assert OR._closes(tmp_path, rec, argv) == (True, [])
        got = LG.own_records_closed(RIG, argv, {"map_overfit_record": OR._write(tmp_path, rec)}, COMMIT, LG.argv_sha256(argv))
        assert list(got) == ["E1-MAP-OWN-RECORD"] and LG.open_item_reasons(RIG, argv, closed=got) == []


def test_RED_M3N20_refuses_a_near_10_record_a_record_without_the_own_enc_stamp_and_another_argvs_record(tmp_path):
    n20 = _argv("M3N20")

    def why(rec, argv=n20):
        reasons = _judge(tmp_path, rec, argv)
        assert reasons, "the defective record passed the banded judge"
        assert not OR._closes(tmp_path, rec, argv)[0]
        return " | ".join(reasons)

    near10, _ = _rec(n20, own=True, near=10)                                          # a record built at near 10 (the M3 table), bound to the M3N20 argv
    assert "near_lift_m" in why(near10) and "gated on" in why(near10)
    no_own, _ = _rec(n20, own=False, near=20)                                         # a 20 m banded record WITHOUT the own encoder / stamp (M1N20's record)
    assert "own_enc" in why(no_own)
    good, _ = _rec(n20, own=True, near=20)
    assert "own_enc" in why({k: v for k, v in good.items() if k != "own_enc"}) and "own_enc" in why(dict(good, own_enc=False))
    assert "128" in why(dict(good, branch_config=dict(good["branch_config"], d_map_enc=64)))
    assert "own-encoder parameters" in why(dict(good, branch_params=dict(good["branch_params"], map_encoder=0)))
    m3 = _argv("M3")
    m3_rec, _ = _rec(m3, own=True, near=10)
    assert "bound to argv" in why(good, argv=m3) and "bound to argv" in why(m3_rec, argv=n20)         # M3N20's record is not M3's, and vice versa
    m1n20_rec, _ = _rec(_argv("M1N20"), own=False, near=20)
    assert "bound to argv" in why(m1n20_rec, argv=n20)                                                # M1N20's record offered for M3N20's argv: another sha
    assert _judge(tmp_path, good, n20) == []                                                          # and the right one still passes


def test_RED_the_gate_admits_nothing_else_a_near_or_weight_variant_is_refused():
    base = _argv("M3N20")

    def set_(argv, flag, val):
        out = list(argv)
        out[out.index(flag) + 1] = val
        return out

    reasons = lambda a: [r for r in LG.required_on_reasons(RIG, a)[0] if r.startswith(("--map-hires-near-lift-m", "--w-map-hires"))]       # noqa: E731
    assert reasons(base) == []
    assert any("--map-hires-near-lift-m" in r for r in reasons(set_(base, "--map-hires-near-lift-m", "10")))     # near 20 with lift 10
    assert any("--w-map-hires" in r for r in reasons(set_(base, "--w-map-hires", "0.7")))                          # a free weight
    m3 = _argv("M3")
    assert any("--map-hires-near-lift-m" in r for r in reasons(set_(m3, "--map-hires-near-lift-m", "20")))       # M3 with lift 20 and no near flag
    flag15 = set_(base, "--r9-map-banded-near-m", "15")
    assert any("--map-hires-near-lift-m" in r for r in reasons(flag15)) or LG.banded_near_m_of(flag15) == 15.0   # near-m 15 is no admitted layout


# =========================================================================== #
# (3) the consumers of an arm-name list                                        #
# =========================================================================== #
def test_record_kind_table_gains_M3_and_M3N20():
    import test_r9gate3_record_kind as RK
    assert RK.EXPECTED["M3"] == ("map-banded", "box") and RK.EXPECTED["M3N20"] == ("map-banded", "box") and RK.EXPECTED["M1N20"] == ("map-banded", "box")


_BASH = shutil.which("bash")
_A18 = ROOT.parent / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-09-26-refcv7-map-hires" / "raw" / "gmo_spec_A18.json"
_HASH = "4e36c721d0e1c8befe3371b73f92e4a455ff4732"


def _bind_dry(tmp_path, kind, arm, inputs=None):
    env = dict(os.environ, PYTHON=sys.executable, MSYS_NO_PATHCONV="1", PYTHONDONTWRITEBYTECODE="1")
    cmd = [_BASH, (ROOT / "ops" / "r9e1_bind.sh").as_posix(), "--dry-run", kind, arm, "--commit", _HASH, "--root", (tmp_path / "root").as_posix(),
           "--code", ROOT.parent.as_posix()] + (["--inputs", inputs.as_posix()] if inputs else [])
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env, timeout=600)
    return r.stdout.replace("\r", ""), tmp_path / "root" / f"ovf_{arm}_{_HASH[:12]}" / f"{kind}.args"


@pytest.mark.skipif(not _BASH, reason="needs bash")
@pytest.mark.parametrize("arm", ["M3", "M3N20"])
def test_bind_dry_run_takes_the_box_kind_and_refuses_the_inherited_map_kind_for_the_banded_argv(tmp_path, arm):
    out2, _ = _bind_dry(tmp_path / "b", "box", arm)
    assert "ZZR9E1BIND-DRYOKZZ" in out2 and "REFUSED" not in out2, out2
    out3, _ = _bind_dry(tmp_path / "c", "map", arm)
    assert "use map-banded" in out3                                                  # the kind must match what the argv builds


@pytest.mark.parametrize("arm", ["M3", "M3N20"])
def test_the_bind_template_renders_the_arms_near_end_and_launch_argv_with_the_gates_own_readers(tmp_path, arm):
    """No bash / A18 spec needed: the template render of r9e1_bind.sh (`refcv8_assemble_argv.py --render-template`) with the values the script fills in."""
    argv_path = RUNS / f"r9e1_{arm}.argv.json"
    nl = "%g" % LG.banded_near_m_of(LG.load_argv_file(argv_path))
    assert nl == str(NEAR[arm])
    sha = LG.argv_sha256(LG.load_argv_file(argv_path))
    args_out, roots_out = tmp_path / "a.args", tmp_path / "a.roots"
    cmd = [sys.executable, str(ROOT / "scripts" / "refcv8_assemble_argv.py"), "--render-template", str(ROOT / "scripts" / "r9e1_bind_map-banded.args.tpl"),
           "--sub", "CODE=/code", "--sub", "COMMIT=" + _HASH, "--sub", "ARGV=" + argv_path.as_posix(), "--sub", "ARGV_SHA=" + sha, "--sub", "OUT=/out",
           "--sub", "INPUTS=/inputs", "--sub", "NEAR_LIFT=" + nl, "--args-out", str(args_out), "--data-roots-out", str(roots_out)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=300)
    assert "RENDERED" in r.stdout, r.stdout + r.stderr
    toks = args_out.read_text(encoding="utf-8").replace("\r", "").split("\n")
    assert toks[toks.index("--near-lift-m") + 1] == str(NEAR[arm]) and toks[toks.index("--launch-argv") + 1].endswith(f"r9e1_{arm}.argv.json")
    assert toks[toks.index("--launch-argv-sha256") + 1] == sha and "--banded" in toks and not any("{" in t and "}" in t for t in toks)
    import map_hires_overfit as O
    facts = O.launch_argv_facts(argv_path, sha)                                       # the harness reads the SAME argv: own encoder + near end together
    assert facts["own_enc"] is True and facts["banded"] is True and O.argv_banded_near_m(facts["argv"]) == float(NEAR[arm])


@pytest.mark.skipif(not _BASH or not _A18.is_file(), reason="needs bash and the registered A18 spec in the tree (UNVERIFIED here: the tree has no TanitAD Research Lab/Architecture & Inference)")
@pytest.mark.parametrize("arm", ["M3", "M3N20"])
def test_bind_dry_run_of_the_banded_kind_renders_the_argvs_near_end(tmp_path, arm):
    out, args = _bind_dry(tmp_path, "map-banded", arm, inputs=_A18.parent)
    assert "ZZR9E1BIND-DRYOKZZ" in out and "REFUSED" not in out, out
    assert f"banded near band 0-{NEAR[arm]} m" in out
    toks = args.read_text(encoding="utf-8").replace("\r", "").split("\n")
    assert toks[toks.index("--near-lift-m") + 1] == str(NEAR[arm]) and toks[toks.index("--launch-argv") + 1].endswith(f"r9e1_{arm}.argv.json")
    sha = LG.argv_sha256(_argv(arm))
    assert toks[toks.index("--launch-argv-sha256") + 1] == sha                       # the record binds THIS arm's argv, the own-encoder flag included


def test_the_harness_follows_the_argv_own_encoder_and_near_end_together():
    import map_hires_overfit as O
    f = {a: O.launch_argv_facts(RUNS / f"r9e1_{a}.argv.json") for a in ARMS}
    for a in ARMS:
        assert f[a]["own_enc"] is OWN[a] and f[a]["banded"] is True and O.argv_banded_near_m(f[a]["argv"]) == float(NEAR[a])
        assert O.resolve_own_enc(False, f[a])[0] is OWN[a]
        O.check_near_vs_argv(float(NEAR[a]), f[a])
        with pytest.raises(SystemExit, match="does not train"):
            O.check_near_vs_argv(10.0 if NEAR[a] == 20 else 20.0, f[a])
    with pytest.raises(SystemExit, match="does not carry"):
        O.resolve_own_enc(True, f["M1N20"])                                          # --own-enc against an argv without the flag: a conflict


def test_the_box_replay_builds_the_launch_models_near_band_and_own_encoder():
    src = (ROOT / "scripts" / "g_box_overfit.py").read_text(encoding="utf-8")
    i = src.index("own_enc=bool(tr._moe.is_on(a))")
    assert "banded_near_m" in src[i - 400:i + 900]                                   # one call builds both levers from the argv's args


# =========================================================================== #
# (4) the verdict row                                                          #
# =========================================================================== #
def test_verdict_rows_decisive_ref_base_carry_ref_C0_and_M1_metrics():
    m1, m1n20 = V.ARMS["M1"], V.ARMS["M1N20"]
    for arm, ref, flags in (("M3", "M1", ("--r9-map-banded", "--r9-map-own-enc")),
                            ("M3N20", "M1N20", ("--r9-map-banded", "--r9-map-own-enc", "--r9-map-banded-near-m"))):
        a = V.ARMS[arm]
        assert a["kind"] == "lever" and a["ref"] == ref and a["allowed"] == (ref, "C0") and a["argv_flags"] == flags
        assert V.resolve_ref(arm, None) == ref and V.resolve_ref(arm, "C0") == "C0"
        for bad in ("M2a", "M2b", "M1" if arm == "M3N20" else "M1N20"):
            with pytest.raises(V.Refused):
                V.resolve_ref(arm, bad)
        shape = lambda ms: [(m["id"].split(".", 1)[1], m["role"], m["clause3"], tuple(m["keys"])) for m in ms]       # noqa: E731
        assert shape(a["metrics"]) == shape(m1n20["metrics"][:1]) + [("guard.ap2", "guard", True, ("ap2m_all",))] + shape(m1n20["metrics"][1:])
        assert a["metrics"][0]["keys"] == (V.MAP_PRIMARY_KEY,) and a["metrics"][0]["role"] == "primary"
        beside = [m for m in a["metrics"] if not m["clause3"]]
        assert len(beside) == 17 and all(m["role"] == "beside" for m in beside) and {V.D3_LANE_KEY, V.D3_BOUNDARY_KEY} <= {m["keys"][0] for m in beside}
        assert all(m["id"].startswith(arm + ".") for m in a["metrics"])
        for reading in ("KEEP/KEEP", "KEEP/DROP", "DROP clause 1-2", "DROP clause 3", "Cfp", "64-channel lift"):
            assert reading in a["next_lever"], (arm, reading)
        assert "CARRY" in " ".join(a["ambiguities"]) and "DECISIVE" in " ".join(a["ambiguities"])
    assert "M2 winner" not in " ".join(V.ARMS["M3"]["ambiguities"]).replace("no M2 winner", "")


@pytest.fixture(scope="module")
def mrig(tmp_path_factory):
    r = TV.MaskedRig(tmp_path_factory.mktemp("m3rig"))
    r.arm("M1", {"miss": [7, 8, 9]}, profile="C0")                     # M1 / M1N20 stand in as BASE: C0 boxes and C0 full-layout map
    r.arm("M1N20", {"miss": [7, 8, 9]}, profile="C0")
    r.arm("M3", {"miss": [7, 8, 9]}, profile="M1")                     # the arm: better map (lv 5-20 m 0.75 vs C0's 0.353), same boxes
    r.arm("M3N20", {"miss": [7, 8, 9]}, profile="M1")
    return r


def test_CLI_end_to_end_M3_and_M3N20_read_vs_their_BASE_and_vs_C0(mrig):
    for arm, base in (("M3", "M1"), ("M3N20", "M1N20")):
        assert V.resolve_ref(arm, None) == base
        for ref in (base, "C0"):                                         # DECISIVE (vs BASE) and CARRY (vs C0): the same rule on the same row
            rc, v, md = mrig.verdict(arm, ref, extra=("--map-beside-d3",))
            assert rc == 0 and v["status"] == "OK" and v["ref"] == ref and v["kind"] == "lever", (arm, ref, v.get("problems"))
            assert [x["id"] for x in v["reads"]][0] == f"{arm}.primary" and v["mapping"]["argv_flags"] == list(V.ARMS[arm]["argv_flags"])
            assert v["verdict"] == "KEEP", (arm, ref, v["verdict"], v.get("failing_clauses"))
            r0 = v["reads"][0]
            assert r0["arm_point"] == pytest.approx(60 / 80, abs=1e-9) and r0["ref_point"] == pytest.approx(60 / 170, abs=1e-9) and r0["separated"] is True
            assert v["clause3"]["vetoes"] == []
            d3 = {x["key"]: x for x in v["beside_pedgefit"]["rows"] if x["key"] in (V.D3_LANE_KEY, V.D3_BOUNDARY_KEY)}
            assert set(d3) == {V.D3_LANE_KEY, V.D3_BOUNDARY_KEY} and all(x["role"] == "beside" for x in d3.values())   # the D3 keys are beside rows, never clauses
    for arm, bad in (("M3", "M1N20"), ("M3N20", "M1")):                 # a reference outside the row's accepted set is refused by the CLI too
        with pytest.raises(V.Refused):
            V.resolve_ref(arm, bad)
        rc, v, _ = mrig.verdict(arm, bad)
        assert rc == 2 and "cannot be read against" in " ".join(v["problems"])


# =========================================================================== #
# (5) reachability from the generated argv, through the trainer's own parser   #
# =========================================================================== #
def _args_of(arm):
    """The trainer's OWN parse of the generated argv + the two flags' own refusals (the same calls `refc_v3_train.main` makes)."""
    args = T.build_parser().parse_args(_argv(arm))
    MBF.pin_args(args)
    MOE.pin_args(args)
    return args


def _cfg_from_args(args, **tiny):
    """`MapHiresConfig` from the parsed args with the trainer's own expressions (refc_v3_train.py ~10032: `banded=_mbf.is_on`, `banded_near_m=_mbf.near_m`,
    `own_enc=_moe.is_on`, `near_lift_x_m=declared_near_lift_m`, `near_refine_blocks=declared_near_refine_blocks`); ``tiny`` shrinks widths only."""
    return H.MapHiresConfig(w_map_hires=float(args.w_map_hires), x_max_m=float(EXTENT_V2.x_max_m), y_half_m=float(EXTENT_V2.y_half_m),
                            near_lift_x_m=float(H.declared_near_lift_m(args)), near_refine_blocks=int(H.declared_near_refine_blocks(args)),
                            banded=bool(MBF.is_on(args)), banded_near_m=float(MBF.near_m(args)), own_enc=bool(MOE.is_on(args)), **tiny)


def test_the_trainer_builds_its_config_from_exactly_these_expressions():
    src = (ROOT / "scripts" / "refc_v3_train.py").read_text(encoding="utf-8")
    for expr in ("banded=bool(_mbf.is_on(args))", "banded_near_m=float(_mbf.near_m(args))", "own_enc=bool(_moe.is_on(args))",
                 "near_lift_x_m=float(_mhr.declared_near_lift_m(args))", "near_refine_blocks=int(_mhr.declared_near_refine_blocks(args))"):
        assert src.count(expr) == 1, expr


@pytest.mark.parametrize("arm", ARMS)
def test_REAL_WIDTH_the_arms_argv_builds_the_own_encoder_and_the_near_band_it_declares(arm):
    """The 2 x 2: own encoder (M3, M3N20) x near band (M1N20, M3N20). Each argv builds exactly what it declares and nothing of the other lever."""
    args = _args_of(arm)
    assert MOE.is_on(args) is OWN[arm] and MBF.near_m(args) == float(NEAR[arm])
    cfg = _cfg_from_args(args)
    br = H.MapHiresBranch(cfg, d_image=512, image_hw=(52, 128))
    assert br.banded.geom[0].rows == ROWS[NEAR[arm]] and br.banded.near_m == float(NEAR[arm]) and br.cfg.near_rows == ROWS[NEAR[arm]]
    pb = br.param_breakdown()
    if OWN[arm]:
        assert br.cfg.d_map_enc == 128 and br.lift.d_out == 128 and br.map_encoder is not None
        assert pb["map_encoder"] == 1_624_832 and MOE.stamp(br)["map_encoder_params"] == 1_624_832 and MOE.stamp(br)["lift_width"] == 128
        assert [n for n, _ in br.named_parameters() if n.startswith("map_encoder.")]
    else:
        assert getattr(br, "map_encoder", None) is None and br.lift.d_out == 64 and pb.get("map_encoder", 0) == 0     # the control: no own-encoder parameters
        assert not [n for n, _ in br.named_parameters() if n.startswith("map_encoder.")]
    assert br.cfg.near_rows == ROWS[NEAR[arm]]                                                                            # rows 100-199 exist only under 20 m


def _tiny_model(arm):
    args = _args_of(arm)
    cfg_h = _cfg_from_args(args, d_lift=16, d_model=16, d_up=16, dilations=(1, 2), grad_ckpt=True, d_map_enc=32)
    torch.manual_seed(0)
    cfg = v3.refc_v3_smoke_config(hier=True)
    cfg.core.encoder = dc.replace(cfg.core.encoder, trunk="timm", trunk_name="resnet18.a1_in1k", trunk_pretrained=False, in_channels=9, image_size=64,
                                  image_width=None, trunk_frozen_bn=True)
    model = v3.RefCV3Model(cfg)
    model.core.encoder.enable_s8_tap()
    model._map_hires = H.build_map_hires_branch(model, cfg_h)
    model._w_map_hires = 1.0
    setattr(model, MBF.MODEL_ATTR, True)
    h, wd = model.cfg.core.encoder.image_hw()
    frame = frame_for_width(int(wd), int(h))
    model._lift_bank_hires = H.HiresLiftGeometryBank({R.CLIP: RigCamera.nominal(frame, height_m=1.5, x_m=1.5)}, frame=frame, cfg=cfg_h)
    model._map_hires_class_weight = torch.ones(8)
    model._map_lift_valid_mask = True
    model._w_map, model._w_box3d = 0.0, 0.0
    g = torch.Generator().manual_seed(11)
    with torch.no_grad():            # a TRAINED near lift / refine (zero-initialised at build: their grads would read 0)
        for n, p in model.named_parameters():
            if n.startswith(("_map_hires.near.", "_map_hires.near_refine.")):
                p.add_(0.02 * torch.randn(p.shape, generator=g))
    return model, cfg


def _grads(model, cfg):
    torch.manual_seed(0)
    model.train()
    model.zero_grad(set_to_none=True)
    losses = T.compute_losses_v3(model, R._batch(cfg), "cpu", mode="diffusion")
    losses["loss"].backward()
    return losses, {n: (None if p.grad is None else float(p.grad.abs().sum())) for n, p in model.named_parameters()}


def test_TRACE_M3N20_one_backward_reaches_the_own_encoder_and_near_rows_100_199_with_zero_controls(monkeypatch):
    model, cfg = _tiny_model("M3N20")
    br = model._map_hires
    assert br.cfg.d_map_enc == 32 and br.banded.geom[0].rows == 200
    own = [n for n, _ in model.named_parameters() if n.startswith("_map_hires.map_encoder.")] + ["_map_hires.refine.inp.0.weight"]
    near = [n for n, _ in model.named_parameters() if n.startswith(("_map_hires.banded.head_near", "_map_hires.near.", "_map_hires.near_refine."))]
    assert own and near
    # PROBE A: every cell observed -> the OWN ENCODER's parameters (and the widened lift rows) are NON-ZERO
    _, g = _grads(model, cfg)
    assert [n for n in own if not g[n]] == [], [n for n in own if not g[n]]
    assert g["_map_hires.lift.proj.weight"] is not None and float(model._map_hires.lift.proj.weight.grad[16:].abs().sum()) > 0.0
    # PROBE B: ONLY near rows 100-199 (10-20 m: the rows that exist only under --r9-map-banded-near-m 20) supervised -> the near path is NON-ZERO
    real, mask = MB.unobs_masks, {"on": True}

    def only_rows_100_199(v, geom):
        out = [torch.ones_like(m) for m in real(v, geom)]
        if mask["on"]:
            out[0][:, 100:200] = False
        return out
    monkeypatch.setattr(MB, "unobs_masks", only_rows_100_199)
    monkeypatch.setattr(MB, "MAXENT_W", 0.0)
    losses, g = _grads(model, cfg)
    assert float(losses["map_banded_n_near"]) > 0 and float(losses["map_banded_n_mid"]) == 0 and float(losses["map_banded_n_far"]) == 0
    assert [n for n in near if not g[n]] == [], [n for n in near if not g[n]]
    assert [n for n in own if not g[n]] == [], [n for n in own if not g[n]]            # the own encoder feeds the near rows too (it is the map's one BEV encoder)
    # CONTROL: the same rows unobserved too -> every one of those parameters reads EXACTLY 0.0 (the probe works at all)
    mask["on"] = False
    losses, g = _grads(model, cfg)
    assert float(losses["map_hires"]) == 0.0
    assert [n for n in own + near if g[n] not in (None, 0.0)] == []


def test_MUT_an_own_encoder_detached_from_the_banded_head_turns_the_M3N20_nonzero_check_red(monkeypatch):
    real = H.MapHiresBranch._forward_banded

    def detach(self, feats, fmap_s8, grid, valid, near_source, near_block_zeros, map_feats=None):
        return real(self, feats, fmap_s8, grid, valid, near_source, near_block_zeros, None if map_feats is None else map_feats.detach())
    monkeypatch.setattr(H.MapHiresBranch, "_forward_banded", detach)
    model, cfg = _tiny_model("M3N20")
    own = [n for n, _ in model.named_parameters() if n.startswith("_map_hires.map_encoder.")]
    _, g = _grads(model, cfg)
    assert [n for n in own if not g[n]] == own                                           # every own-encoder parameter now reads 0 / None: the TRACE test would be RED


def test_CONTROL_M1s_argv_builds_no_own_encoder_and_no_rows_100_199():
    model, _ = _tiny_model("M1")
    br = model._map_hires
    assert getattr(br, "map_encoder", None) is None
    assert not [n for n, _ in model.named_parameters() if n.startswith("_map_hires.map_encoder.")]
    assert br.banded.geom[0].rows == 100 and br.cfg.near_rows == 100 and br.lift.d_out == 16
    m3, _ = _tiny_model("M3")
    assert m3._map_hires.banded.geom[0].rows == 100 and m3._map_hires.map_encoder is not None
    n20, _ = _tiny_model("M1N20")
    assert n20._map_hires.banded.geom[0].rows == 200 and getattr(n20._map_hires, "map_encoder", None) is None


def test_the_declared_vs_built_registry_reads_both_levers_off_the_built_M3N20_model():
    from tanitad.train import declared_vs_built as D
    model, _ = _tiny_model("M3N20")
    args = _args_of("M3N20")
    bad = [m for m in D.check(model, args) if "banded_near_m" in str(m) or "r9_map_own_enc" in str(m) or "own_enc" in str(m)]
    assert bad == [], bad
    wrong, _ = _tiny_model("M3")                                                          # M3's model against M3N20's args: BOTH levers disagree
    bad = [str(m) for m in D.check(wrong, args)]
    assert any("near" in m for m in bad), bad
