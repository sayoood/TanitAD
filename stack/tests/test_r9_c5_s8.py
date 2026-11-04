"""refcv9 C5-S8 (BRIEF_C5_S8): B4's box-head levers classified as OWN-HEAD levers end to end, and lever 3 (`--r9-box-s8-horizon`).

DELIVERABLE 1 -- `--r9-box-layers N` is an OWN-head lever (it changes the INFERENCE box head: 6 decoder layers), so:
  * `launch_gate.E1_BOX_OWN_FLAGS` names it and every map derived from it follows (the E1-BOX-OWN-RECORD row, the open items, `_R9F_ATTACH`);
  * `_R9F_ATTACH` maps it to `_bc5.attach(model, args)` -- NOT B3's `_bbf.attach` (B3's `is_any` does not cover it, so on a layers-only argv `_bbf.attach` is a
    no-op and its needle would prove nothing about the depth);
  * `g_box_overfit.B3_FLAGS` / `b3_record_fields` / `strip_*` read a VALUED flag by its HEAD (`--r9-box-layers 6` and `--r9-box-layers=6`);
  * the HARNESS REPLAY BUILD (`TrainerAdapter._build_model`) builds the SAME depth as the trainer -- else G-DVB refuses it (the B3 defect of 2026-11-03);
  * `r9e1_bind.sh` already classifies by `flag_pairs` NAMES (value-safe): pinned here by running the real script's dry run.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import types
from pathlib import Path

import pytest
import torch

STACK = Path(__file__).resolve().parents[1]
for _p in (str(STACK), str(STACK / "scripts"), str(STACK / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import launch_gate as LG                                           # noqa: E402
import test_g_box_overfit_near_lift as NL                          # noqa: E402  (its tiny rig, inputs, adapter helpers)
import test_g_box_overfit_rig as RG                                # noqa: E402  (the E1 rig argv at tiny size)
import test_launch_gate_e1rig_e1rec2 as R2                         # noqa: E402  (the bind-script dry-run helper)

GBO_PATH = STACK / "scripts" / "g_box_overfit.py"
TRAINER_PATH = STACK / "scripts" / "refc_v3_train.py"
RUNS = STACK / "ops" / "runs.d"
LAYERS = "--r9-box-layers"
RIG = LG.PROFILES["refcv9-e1rig"]
#: the one line of the harness replay that builds the depth (deleting it = the B3 defect again)
C5_ATTACH_LINE = "        tr._bc5.attach(m, a)\n"


def _bash_tree_argv(extra):
    return json.loads((RUNS / "r9e1_B3.argv.json").read_text(encoding="utf-8"))["argv"] + list(extra)


@pytest.fixture(scope="module")
def rigs(tmp_path_factory):
    d = tmp_path_factory.mktemp("c5s8_rig")
    NL._synth_inputs(d)
    base = RG._rig_argv(d)
    argvs = {"off": base, "l6": base + [LAYERS, "6"], "l6eq": base + [LAYERS + "=6"], "b3l6": base + ["--r9-box-quality", "--r9-box-dn", "--r9-box-class-sqrt", LAYERS, "6"]}
    T = NL._load("refc_v3_train_c5s8_ref", TRAINER_PATH)
    refs = {k: LG.run_trainer_until(T, v, "model")["model"] for k, v in argvs.items()}      # the REFERENCES: the real train()
    G = NL._load("g_box_overfit_c5s8_t", GBO_PATH)
    return types.SimpleNamespace(d=d, T=T, G=G, argvs=argvs, ref=refs)


def _mutated(tmp_path: Path, name: str, edits):
    src = GBO_PATH.read_text(encoding="utf-8")
    for old, new in edits:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    p = tmp_path / f"{name}.py"
    p.write_text(src, encoding="utf-8", newline="\n")
    return NL._load(f"{name}_t", p)


def _digests(m):
    return LG.state_digests(m.state_dict())


# ====================================================================================================================== #
# 1. the classification: ONE fact in the gate, the harness, the open items                                               #
# ====================================================================================================================== #
def test_layers_is_an_OWN_head_lever_pinned_across_gate_and_harness(rigs):
    G = rigs.G
    assert LAYERS in LG.E1_BOX_OWN_FLAGS and LAYERS not in LG.E1_BOX_AUX_FLAGS and LAYERS not in LG.E1_T1_FLAGS
    assert tuple(G.B3_FLAGS) == LG.E1_BOX_OWN_FLAGS                                    # the gate's tuple IS the harness's (the pin tests said so for B3's four)
    assert LG.E1_BOX_OWN_FLAGS[:4] == ("--r9-box-quality", "--r9-box-dn", "--r9-box-class-sqrt", "--r9-box-inloss-nms")      # B3's four, unchanged and in order
    assert LAYERS in LG.E1_OWN_RECORDS["E1-BOX-OWN-RECORD"]["flags"]                    # the record the row closes covers it
    assert not set(LG.E1_BOX_OWN_FLAGS) & set(LG.E1_BOX_AUX_FLAGS)


@pytest.mark.parametrize("argv", [[LAYERS, "6"], [LAYERS + "=6"]], ids=["spaced", "equals"])
def test_a_layers_argv_opens_the_OWN_record_item_and_NOT_the_unknown_lever_item(argv):
    items = {it["id"]: it for it in RIG["open_items"]}
    assert LG.open_item_hits(items["E1-BOX-OWN-RECORD"], argv) == [LAYERS]               # the flag NAME, not a value token
    assert LG.open_item_hits(items["E1-BOX-UNKNOWN-LEVER"], argv) == []                  # classified: not "unknown"
    row = LG.own_record_row(RIG, argv, "G-BOX-OVERFIT")
    assert row is not None and row["item"] == "E1-BOX-OWN-RECORD"                      # the box check demands the lever's OWN /2-b3 record
    assert LG.own_record_row(RIG, ["--r9-box-depth-aux"], "G-BOX-OVERFIT") is None      # an aux flag still does not


def test_an_UNKNOWN_box_flag_next_to_layers_is_still_refused_everywhere(rigs):
    items = {it["id"]: it for it in RIG["open_items"]}
    argv = [LAYERS, "6", "--r9-box-layerz", "6"]
    assert LG.open_item_hits(items["E1-BOX-UNKNOWN-LEVER"], argv) == ["--r9-box-layerz"]
    for bad in (["--r9-box-layerz", "6"], ["--r9-box-layers-v2=6"], [LAYERS, "6", "--r9-box-foo"]):
        with pytest.raises(SystemExit, match="unknown --r9-box-"):
            rigs.G.b3_record_fields(bad)


def test_R9F_ATTACH_maps_layers_to_the_c5_attach_not_to_B3s(rigs):
    assert LG._R9F_ATTACH[LAYERS] == "_bc5.attach(model, args)"
    for f in LG.E1_BOX_OWN_FLAGS[:4]:
        assert LG._R9F_ATTACH[f] == "_bbf.attach(model, args)"                           # B3's four: unchanged
    src = TRAINER_PATH.read_text(encoding="utf-8")
    prof = LG.PROFILES["refcv9-e2"]
    assert prof["lever_attach_needles"][LAYERS] == "_bc5.attach(model, args)"
    ok, det = LG.lever_attach_reasons(prof, [LAYERS, "6"], src)
    assert ok == [] and det["checked"] == {LAYERS: True}
    # RED: the c5 attach hop deleted from a COPY of the trainer -> the reachability gate names the flag
    bad_src = src.replace("_bc5.attach(model, args)", "pass")
    assert bad_src != src
    red, det2 = LG.lever_attach_reasons(prof, [LAYERS, "6"], bad_src)
    assert len(red) == 1 and LAYERS in red[0] and "_bc5.attach(model, args)" in red[0] and det2["checked"] == {LAYERS: False}
    # the B3 needle is NOT a proof for layers: deleting `_bbf.attach` leaves a layers-only argv green and a B3 argv red
    no_bbf = src.replace("_bbf.attach(model, args)", "pass")
    assert LG.lever_attach_reasons(prof, [LAYERS, "6"], no_bbf)[0] == []
    assert len(LG.lever_attach_reasons(prof, ["--r9-box-dn"], no_bbf)[0]) == 1


# ====================================================================================================================== #
# 2. the harness' record fields, and the valued flag through every strip / membership test                               #
# ====================================================================================================================== #
@pytest.mark.parametrize("argv", [[LAYERS, "6"], [LAYERS + "=6"]], ids=["spaced", "equals"])
def test_b3_record_fields_stamps_a_valued_layers_flag_by_name(rigs, argv):
    G = rigs.G
    r = G.b3_record_fields(["--w-box3d", "1.0"] + argv)
    assert r["schema"] == G.B3_SCHEMA and r["b3"]["flags"] == [LAYERS]
    r2 = G.b3_record_fields(["--r9-box-quality"] + argv + ["--r9-box-dn"])
    assert r2["b3"]["flags"] == ["--r9-box-quality", "--r9-box-dn", LAYERS]              # B3_FLAGS order
    assert G.b3_record_fields(["--w-box3d", "1.0"]) == {}                              # no own-head flag: the record is untouched


def test_the_value_token_survives_every_strip_and_the_identity_changes_with_the_depth(rigs):
    G = rigs.G
    av = ["--w-box3d", "1.0", LAYERS, "6", "--r9-box-persp-aux", "--w-box-persp", "0.1", "--r9-bev-warp", "--trunk-compile"]
    build, rec = G.harness_build_argv(av)
    assert build == ["--w-box3d", "1.0", LAYERS, "6"]                                   # aux + T1 + compile stripped; the layers flag AND its value kept
    assert rec["aux_flags_not_built"] == ["--r9-box-persp-aux"] and rec["bev_warp_flags_not_built"] == ["--r9-bev-warp"]
    assert G.strip_aux([LAYERS, "6", "--r9-box-persp-aux"])[0] == [LAYERS, "6"]
    assert G.strip_t1([LAYERS, "6", "--r9-bev-warp-lags", "1", "2"])[0] == [LAYERS, "6"]
    h = {"other_args": {}, "audit_inputs": {}, "roots": {}}
    i6 = G.identity_block(["--w-box3d", "1.0", LAYERS, "6"], h)
    i4 = G.identity_block(["--w-box3d", "1.0", LAYERS, "4"], h)
    i0 = G.identity_block(["--w-box3d", "1.0"], h)
    assert len({i6["identity_sha256"], i4["identity_sha256"], i0["identity_sha256"]}) == 3      # a depth never reuses another depth's (or the default's) record
    assert LAYERS in i6["effective_build_argv"] and "6" in i6["effective_build_argv"]


def _rec_for(argv, stamp_flags):
    return {"schema": "tanitad.g_box_overfit_record/2-b3", "tool": "g_box_overfit.py", "RESULT": "PASS",
            "launch_argv_sha256": LG.argv_sha256(argv), "b3": {"flags": stamp_flags}, "arms": {}}


def test_judge_box_own_record_reads_the_layers_flag_set(tmp_path):
    argv = ["--w-box3d", "1.0", LAYERS, "6"]
    row = dict(LG.E1_OWN_RECORDS["E1-BOX-OWN-RECORD"])

    def reasons(stamp):
        p = tmp_path / "rec.json"
        p.write_text(json.dumps(_rec_for(argv, stamp)), encoding="utf-8")
        r, det = LG.judge_box_own_record(row, str(p), LG.argv_sha256(argv), argv)
        return [x for x in r if "was made for" in x or "stamps no flag set" in x], det

    ok, det = reasons([LAYERS])
    assert ok == [] and det["flags"] == {"argv": [LAYERS], "record_stamp": [LAYERS]}
    bad, _ = reasons(["--r9-box-quality"])                                             # a record made for B3's quality never covers the depth
    assert len(bad) == 1 and "was made for ['--r9-box-quality']" in bad[0] and f"carries ['{LAYERS}']" in bad[0]
    assert len(reasons([])[0]) == 1                                                    # no stamped flag set


# ====================================================================================================================== #
# 3. the HARNESS REPLAY BUILD builds the trainer's depth                                                                  #
# ====================================================================================================================== #
@pytest.mark.parametrize("kind", ["l6", "l6eq"])
def test_the_harness_replay_IS_the_depth_6_model_train_builds(rigs, kind):
    m, ad = NL._harness_model(rigs.G, rigs.T, rigs.argvs[kind])
    ref, T = rigs.ref[kind], rigs.T
    assert ad.dvb_mismatches == []                                                       # G-DVB clean
    assert len(m._perception.box_dec.blocks.layers) == len(ref._perception.box_dec.blocks.layers) == 6
    assert m._perception.box_dec.blocks.num_layers == 6
    assert sorted(set(ref.state_dict()) ^ set(m.state_dict())) == []
    dt, dh = _digests(ref), _digests(m)
    assert [k for k in dt if dt[k] != dh[k]] == []                                       # same seed, same build order: the harness' init IS the launch init
    assert any(k.startswith("_perception.box_dec.blocks.layers.5.") for k in m.state_dict())
    args = T.build_parser().parse_args(rigs.argvs[kind])
    assert ad.c5_stamp == T._bc5.stamp(ref, args) and ad.c5_stamp["layers"]["depth"] == 6 and ad.b3_stamp is None
    assert ad.c5_stamp["layers"]["box_decoder_params"] == int(m._perception.box_dec.n_params)
    f = rigs.G.b3_build_fields(ad.b3_stamp, ad.c5_stamp)
    assert f == {"built_c5": ad.c5_stamp} and "built" not in f                          # a layers-only argv: no B3 `built`


def test_RED_the_replay_without_the_c5_attach_is_refused_by_G_DVB_with_the_depth_mismatch(rigs, tmp_path):
    G0 = _mutated(tmp_path, "gbo_no_c5_attach", [(C5_ATTACH_LINE, "")])
    with pytest.raises(SystemExit) as e:
        NL._harness_model(G0, rigs.T, rigs.argvs["l6"])
    lines = str(e.value).splitlines()
    assert lines[0] == "[gbo] the replayed build is NOT the declared launch model:"
    assert len(lines) == 2 and "r9_box_layers: declared 6 but BUILT 3" in lines[1] and "box_dec.blocks.layers" in lines[1], lines


def test_the_combined_B3_plus_layers_argv_builds_both_in_the_trainers_order(rigs):
    m, ad = NL._harness_model(rigs.G, rigs.T, rigs.argvs["b3l6"])
    ref = rigs.ref["b3l6"]
    assert ad.dvb_mismatches == [] and len(m._perception.box_dec.blocks.layers) == 6 and m._r9_box_b3.dn
    dt, dh = _digests(ref), _digests(m)
    assert sorted(dt) == sorted(dh) and [k for k in dt if dt[k] != dh[k]] == []
    assert ad.b3_stamp is not None and ad.c5_stamp is not None
    f = rigs.G.b3_build_fields(ad.b3_stamp, ad.c5_stamp)
    assert set(f) == {"built", "built_c5"}


def test_OFF_path_is_bit_identical_with_and_without_the_c5_attach_line(rigs, tmp_path):
    G0 = _mutated(tmp_path, "gbo_off_no_c5", [(C5_ATTACH_LINE, "")])
    m_new, ad_new = NL._harness_model(rigs.G, rigs.T, rigs.argvs["off"])
    m_old, ad_old = NL._harness_model(G0, rigs.T, rigs.argvs["off"])
    ref = rigs.ref["off"]
    assert ad_new.dvb_mismatches == ad_old.dvb_mismatches == []
    dn, do_, dt = _digests(m_new), _digests(m_old), _digests(ref)
    assert sorted(dn) == sorted(do_) == sorted(dt)
    assert [k for k in dn if dn[k] != do_[k]] == [] and [k for k in dn if dn[k] != dt[k]] == []
    assert len(m_new._perception.box_dec.blocks.layers) == 3
    assert ad_new.c5_stamp is None and rigs.G.b3_build_fields(ad_new.b3_stamp, ad_new.c5_stamp) == {}      # the record is untouched


# ====================================================================================================================== #
# 4. the bind script: r9e1_bind.sh classifies by flag NAMES, so a valued flag lands in OWN                                #
# ====================================================================================================================== #
def _code_tree(tmp_path: Path, argv) -> Path:
    code = tmp_path / "code"
    shutil.copytree(STACK / "scripts", code / "stack" / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
    (code / "stack" / "ops" / "runs.d").mkdir(parents=True)
    (code / "stack" / "ops" / "runs.d" / "r9e1_X.argv.json").write_text(json.dumps({"argv": argv}), encoding="utf-8")
    return code


@R2.needs_bash
@pytest.mark.parametrize("extra", [[LAYERS, "6"], [LAYERS + "=6"]], ids=["spaced", "equals"])
def test_bind_dry_run_puts_layers_in_OWN_box_b3_accepts_and_box_refuses(tmp_path, extra):
    argv = _bash_tree_argv(extra)
    out, root = R2._bind(tmp_path / "ok", "box-b3", "X", code=_code_tree(tmp_path / "ok", argv))
    assert "ZZR9E1BIND-DRYOKZZ" in out and "REFUSED" not in out, out
    out2, _ = R2._bind(tmp_path / "wrong", "box", "X", code=_code_tree(tmp_path / "wrong", argv))
    assert "ZZR9E1BIND-REFUSEDZZ" in out2 and "own-head --r9-box-* lever" in out2 and "use box-b3" in out2 and LAYERS in out2, out2
    # a layers-ONLY argv (no B3 flag) is an own-head arm too: kind box-b3, never the inherited kind
    only = [t for t in argv if t not in ("--r9-box-quality", "--r9-box-dn", "--r9-box-class-sqrt")]
    out3, _ = R2._bind(tmp_path / "only", "box-b3", "X", code=_code_tree(tmp_path / "only", only))
    assert "ZZR9E1BIND-DRYOKZZ" in out3, out3


@R2.needs_bash
def test_RED_bind_dry_run_still_refuses_an_unknown_box_flag_beside_layers(tmp_path):
    argv = _bash_tree_argv([LAYERS, "6", "--r9-box-foo"])
    for kind in ("box", "box-b3"):
        out, root = R2._bind(tmp_path / kind, kind, "X", code=_code_tree(tmp_path / kind, argv))
        assert "ZZR9E1BIND-REFUSEDZZ" in out and "unknown box lever (--r9-box-foo)" in out and "DRYOK" not in out, out
