"""The launch gate (SPEC_REFCV7 section 2): the token binding and every refusal path.

CPU only, no model build, no data: each test constructs its inputs and states its expectation as a
LITERAL. The hashes below were computed with plain `hashlib` on the spelled-out byte strings, never
with the code under test, so a change to the gate's hashing DEFINITION goes red here.

The supervisor is exercised END TO END in bash (a `flock` shim stands in on hosts without one):
it must REFUSE a free-form TRAIN_CMD, a manifest that disagrees with the gated argv, and a token
whose argv / HMAC does not match -- and it must run a fake trainer to completion on a valid token.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import launch_gate as LG  # noqa: E402

GATE_PY = ROOT / "scripts" / "launch_gate.py"
SUP_SH = ROOT / "ops" / "sup_refcv7.sh"


# ------------------------------------------------------------------------------------------ #
# the binding primitives -- literal expectations                                               #
# ------------------------------------------------------------------------------------------ #
def test_argv_sha256_is_the_ORDERED_compact_json_array():
    assert LG.argv_sha256(["--a", "1", "--out", "/o"]) == \
        "5ba5a3c41c5af201b0e2f4629cf1494ccfcf2f6544020b37bd1c2561262f9e6c"
    # order is meaning: the same tokens in another order are another launch
    assert LG.argv_sha256(["--out", "/o", "--a", "1"]) == \
        "1c03b9600cdb1c31bbc54f97e1d3b849cc8e8e90f34fc360c648016efac2adfd"


def test_tree_digest_definition_and_cache_exclusion(tmp_path):
    (tmp_path / "stack" / "__pycache__").mkdir(parents=True)
    (tmp_path / "taniteval").mkdir()
    (tmp_path / "stack" / "a.py").write_bytes(b"x\n")
    (tmp_path / "taniteval" / "b.txt").write_bytes(b"y")
    (tmp_path / "stack" / "__pycache__" / "a.cpython-313.pyc").write_bytes(b"junk")
    (tmp_path / "stack" / "c.pyc").write_bytes(b"junk")
    man = LG.tree_manifest(tmp_path)
    assert sorted(man) == ["stack/a.py", "taniteval/b.txt"]
    assert LG.tree_digest(man) == \
        "5d062eb992a918321f420ae1d3b95403d21e6f0e10e55dfff2277dbade10fd6b"


def test_tree_manifest_refuses_a_tree_without_stack(tmp_path):
    (tmp_path / "taniteval").mkdir()
    with pytest.raises(LG.GateError):
        LG.tree_manifest(tmp_path)


def test_fingerprints_file_dir_missing(tmp_path):
    f = tmp_path / "f.bin"
    f.write_bytes(b"x\n")
    fp = LG.fingerprint(f)
    assert fp["kind"] == "file" and fp["bytes"] == 2
    assert fp["sha256"] == "73cb3858a687a8494ca3323053016282f3dad39d42cf62ca4e79dda2aac7d9ac"
    d = tmp_path / "d"
    (d / "sub").mkdir(parents=True)
    (d / "z.txt").write_bytes(b"zz")
    (d / "sub" / "c.bin").write_bytes(b"ccc")
    (d / "_v2manifest.pt").write_bytes(b"rewritten on first use")      # excluded by design
    fd = LG.fingerprint(d)
    assert (fd["kind"], fd["n_files"], fd["bytes"]) == ("dir", 2, 5)
    assert fd["listing_sha256"] == \
        "854a0ed1f8e8443566074746dc7ca978b9d36ad4bc663074f1ca3b282c6450f6"
    assert LG.fingerprint(tmp_path / "nope") == {"kind": "missing"}


def test_tensor_digest_survives_a_strided_batch_1_slice():
    """MEASURED 2026-09-26 on the G-EVAL probe (batch 1): shape [1] with stride 12 counts as
    contiguous, so `.contiguous().reshape(-1).view(uint8)` raised and G-EVAL crashed."""
    import torch
    t = torch.arange(24.).reshape(2, 12)[:1, 3]
    assert t.stride() == (12,) and t.is_contiguous()
    assert LG.tensor_digest(t) == \
        "72b230f7fac0e6860c64d8e6f76d7c2850f7fcb241e5456a66c5ee379aeb6fff"
    assert LG.tensor_digest(torch.tensor([3.0])) == LG.tensor_digest(t)


def test_data_inputs_are_derived_from_the_argv_outputs_excluded():
    argv = ["--arm", "hier", "--anchors", "/d/a.pt", "--image-hw", "416", "1024",
            "--v2-cache", "/d/cache", "--lr", "1e-4", "--out", "/runs/x",
            "--eval-window-dump", "/runs/x/w.jsonl"]
    got = LG.data_inputs(argv, LG.PROFILES["refcv7"])
    assert got == [("--anchors", "/d/a.pt"), ("--v2-cache", "/d/cache")]


def test_set_flag_replaces_every_occurrence_and_removes():
    argv = ["--steps", "50400", "--x", "--steps", "7", "--out", "/o"]
    assert LG.set_flag(argv, "--steps", ["30"]) == ["--steps", "30", "--x", "--out", "/o"]
    assert LG.set_flag(argv, "--x", None) == ["--steps", "50400", "--steps", "7", "--out", "/o"]
    assert LG.set_flag(["--a"], "--b", ["1"]) == ["--a", "--b", "1"]


def test_path_map_longest_prefix():
    pm = LG.parse_path_map(["/home/nvidia/data=D:/kit/data", "/home/nvidia=D:/other"])
    assert LG.map_path("/home/nvidia/data/anchors/a.pt", pm) == "D:/kit/data/anchors/a.pt"
    assert LG.map_path("/home/nvidia/x", pm) == "D:/other/x"
    assert LG.map_path("/home/nvidiaX/x", pm) == "/home/nvidiaX/x"


# ------------------------------------------------------------------------------------------ #
# the key                                                                                      #
# ------------------------------------------------------------------------------------------ #
def test_key_is_created_once_and_reused(tmp_path):
    kp = tmp_path / "keys" / "k.key"
    k1 = LG.load_key(kp, create=True)
    k2 = LG.load_key(kp, create=True)
    assert len(k1) == 32 and k1 == k2
    assert len(kp.read_text(encoding="ascii").strip()) == 64


def test_key_is_refused_inside_the_code_tree_and_never_silently_created(tmp_path):
    tree = tmp_path / "tree"
    (tree / "stack").mkdir(parents=True)
    with pytest.raises(LG.GateError):
        LG.load_key(tree / "stack" / "k.key", create=True, forbid_under=[tree])
    with pytest.raises(LG.GateError):
        LG.load_key(tmp_path / "absent.key", create=False)


# ------------------------------------------------------------------------------------------ #
# the token: finalize -> verify, and every refusal                                             #
# ------------------------------------------------------------------------------------------ #
def _mini_tree(root: Path) -> Path:
    (root / "stack" / "scripts").mkdir(parents=True)
    (root / "taniteval").mkdir()
    (root / "stack" / "scripts" / "x.py").write_text("print(1)\n", encoding="utf-8")
    (root / "taniteval" / "t.py").write_text("T = 1\n", encoding="utf-8")
    return root


def _ctx(tmp_path: Path, argv=None, arm=None) -> LG.Ctx:
    tree = _mini_tree(tmp_path / "tree")
    data = tmp_path / "data.bin"
    if not data.exists():
        data.write_bytes(b"payload")
    argv = argv or ["--arm", "hier", "--anchors", data.as_posix(), "--steps", "10",
                    "--out", (tmp_path / "run").as_posix()]
    return LG.Ctx(profile="refc", tree=str(tree), commit="ab" * 20, argv=list(argv),
                  out_dir=str(tmp_path / "gate"), path_map=[],
                  tree_sha256=LG.tree_digest(LG.tree_manifest(tree)),
                  argv_sha256=LG.argv_sha256(argv), options={}, arm=arm)


def _evidence(ctx: LG.Ctx, check: str, status: str = "PASS", **over) -> dict:
    ev = {"schema": LG.GATE_SCHEMA, "kind": "evidence", "check": check, "status": status,
          "reasons": [] if status == "PASS" else [f"{check} said {status}"],
          "binding": ctx.binding(), "inputs_read": {}, "host": {"node": "test"},
          "arm": None, "started_utc": "2026-09-26T00:00:00Z",
          "finished_utc": "2026-09-26T00:00:01Z", "details": {}}
    ev.update(over)
    return ev


def _write_all(ctx: LG.Ctx, **status_by_check) -> None:
    for c in LG.BASE_CHECKS:
        LG.write_evidence(ctx.out_dir, _evidence(ctx, c, status_by_check.get(c, "PASS")))


def _pass_token(tmp_path):
    ctx = _ctx(tmp_path)
    key = LG.load_key(tmp_path / "k.key")
    _write_all(ctx)
    verdict, path, tok = LG.finalize(ctx, key)
    return ctx, key, verdict, path, tok


def test_all_seven_bound_PASS_evidences_make_a_signed_PASS_that_verifies(tmp_path):
    ctx, key, verdict, path, tok = _pass_token(tmp_path)
    assert verdict == "PASS" and path.name == f"PASS_{'ab' * 6}.json"
    assert set(tok["checks"]) == set(LG.BASE_CHECKS) and len(LG.BASE_CHECKS) == 7
    assert tok["binding"]["argv"] == ctx.argv
    assert tok["binding"]["data"]["--anchors"]["sha256"] == \
        "239f59ed55e737c77147cf55ad0c1b030b6d7ee748a7426952f9b852d5a935e5"
    ok, reasons, _ = LG.verify_token(path, ctx.argv, ctx.tree, key=key)
    assert ok and reasons == []


def test_REFUSED_when_the_argv_changed_after_the_pass(tmp_path):
    ctx, key, _, path, _ = _pass_token(tmp_path)
    changed = LG.set_flag(ctx.argv, "--steps", ["11"])
    ok, reasons, _ = LG.verify_token(path, changed, ctx.tree, key=key)
    assert not ok and any("argv differs" in r for r in reasons)


def test_REFUSED_when_the_token_is_hand_edited(tmp_path):
    ctx, key, _, path, tok = _pass_token(tmp_path)
    tok["checks"]["G-LIVE"]["status"] = "PASS (edited)"
    path.write_text(json.dumps(tok), encoding="utf-8")
    ok, reasons, _ = LG.verify_token(path, ctx.argv, ctx.tree, key=key)
    assert not ok and any("HMAC" in r for r in reasons)


def test_REFUSED_when_the_argv_AND_the_token_are_edited_consistently(tmp_path):
    """The attack the HMAC exists for: someone changes the argv and patches the token to match."""
    ctx, key, _, path, tok = _pass_token(tmp_path)
    changed = LG.set_flag(ctx.argv, "--steps", ["11"])
    tok["binding"]["argv"] = changed
    tok["binding"]["argv_sha256"] = LG.argv_sha256(changed)
    path.write_text(json.dumps(tok), encoding="utf-8")
    ok, reasons, _ = LG.verify_token(path, changed, ctx.tree, key=key)
    assert not ok and any("HMAC" in r for r in reasons)


def test_REFUSED_under_another_hosts_key(tmp_path):
    ctx, _, _, path, _ = _pass_token(tmp_path)
    other = LG.load_key(tmp_path / "other.key")
    ok, reasons, _ = LG.verify_token(path, ctx.argv, ctx.tree, key=other)
    assert not ok and any("HMAC" in r for r in reasons)


def test_REFUSED_when_one_code_file_changed(tmp_path):
    ctx, key, _, path, _ = _pass_token(tmp_path)
    (Path(ctx.tree) / "stack" / "scripts" / "x.py").write_text("print(2)\n", encoding="utf-8")
    ok, reasons, _ = LG.verify_token(path, ctx.argv, ctx.tree, key=key)
    assert not ok and any("code tree differs" in r for r in reasons)


def test_REFUSED_when_a_data_input_changed(tmp_path):
    ctx, key, _, path, _ = _pass_token(tmp_path)
    (tmp_path / "data.bin").write_bytes(b"payload, rebuilt")
    ok, reasons, _ = LG.verify_token(path, ctx.argv, ctx.tree, key=key)
    assert not ok and any("data differs" in r for r in reasons)


def test_REFUSED_on_another_commit(tmp_path):
    ctx, key, _, path, _ = _pass_token(tmp_path)
    ok, reasons, _ = LG.verify_token(path, ctx.argv, ctx.tree, key=key, commit="cd" * 20)
    assert not ok and any("commit" in r for r in reasons)


def test_one_FAIL_makes_a_FAIL_token_and_DEMOTES_the_old_PASS(tmp_path):
    ctx, key, _, pass_path, _ = _pass_token(tmp_path)
    LG.write_evidence(ctx.out_dir, _evidence(ctx, "G-LIVE", "FAIL",
                                             finished_utc="2026-09-26T00:00:09Z"))
    verdict, path, tok = LG.finalize(ctx, key)
    assert verdict == "FAIL" and path.name == f"FAIL_{'ab' * 6}.json"
    assert not pass_path.exists()                     # no stale PASS left to be picked up
    assert list((Path(ctx.out_dir) / "superseded").glob("PASS_*.json"))
    ok, reasons, _ = LG.verify_token(path, ctx.argv, ctx.tree, key=key)
    assert not ok and any("not PASS" in r for r in reasons)


def test_missing_evidence_is_INCOMPLETE_never_PASS(tmp_path):
    ctx = _ctx(tmp_path)
    key = LG.load_key(tmp_path / "k.key")
    for c in LG.BASE_CHECKS:
        if c != "G-SUITE":
            LG.write_evidence(ctx.out_dir, _evidence(ctx, c))
    verdict, _, tok = LG.finalize(ctx, key)
    assert verdict == "INCOMPLETE" and tok["checks"]["G-SUITE"]["status"] == "MISSING"


def test_ERROR_evidence_is_a_FAIL(tmp_path):
    ctx = _ctx(tmp_path)
    key = LG.load_key(tmp_path / "k.key")
    _write_all(ctx, **{"G-EVAL": "ERROR"})
    assert LG.finalize(ctx, key)[0] == "FAIL"


def test_evidence_bound_to_another_argv_is_not_usable(tmp_path):
    ctx = _ctx(tmp_path)
    key = LG.load_key(tmp_path / "k.key")
    _write_all(ctx)
    ev = _evidence(ctx, "G-CKPT")
    ev["binding"] = {**ev["binding"], "argv_sha256": "0" * 64}
    LG.write_evidence(ctx.out_dir, ev)                # overwrites the good one
    verdict, _, tok = LG.finalize(ctx, key)
    assert verdict == "INCOMPLETE"
    assert tok["checks"]["G-CKPT"]["rejected"][0]["why"] == ["binding differs"]


def test_evidence_that_read_OTHER_data_is_not_usable(tmp_path):
    ctx = _ctx(tmp_path)
    key = LG.load_key(tmp_path / "k.key")
    _write_all(ctx)
    ev = _evidence(ctx, "G-DVB", inputs_read={"--anchors": {"kind": "file", "sha256": "1" * 64}})
    LG.write_evidence(ctx.out_dir, ev)
    verdict, _, tok = LG.finalize(ctx, key)
    assert verdict == "INCOMPLETE"
    assert "input --anchors read with a different fingerprint" in \
        tok["checks"]["G-DVB"]["rejected"][0]["why"]


def test_an_ARM_can_never_produce_a_PASS(tmp_path):
    ctx = _ctx(tmp_path)
    key = LG.load_key(tmp_path / "k.key")
    _write_all(ctx)
    ev = _evidence(ctx, "G-HYG", arm="undeclared_equalize")
    LG.write_evidence(ctx.out_dir, ev)
    assert LG.finalize(ctx, key)[0] == "INCOMPLETE"   # the arm evidence is rejected
    ctx_arm = dataclasses.replace(ctx, arm="no_cascade_passthrough")
    _write_all(ctx)
    assert LG.finalize(ctx_arm, key)[0] == "FAIL"     # and an arm context refuses outright


def test_PI_DECISION_is_its_own_verdict_and_never_verifies(tmp_path):
    ctx = _ctx(tmp_path)
    key = LG.load_key(tmp_path / "k.key")
    _write_all(ctx, **{"G-LIVE": "PI-DECISION"})
    verdict, path, _ = LG.finalize(ctx, key)
    assert verdict == "PI-DECISION" and path.name.startswith("PI-DECISION_")
    ok, reasons, _ = LG.verify_token(path, ctx.argv, ctx.tree, key=key)
    assert not ok and any("not PASS" in r for r in reasons)


# ------------------------------------------------------------------------------------------ #
# G-HYG                                                                                        #
# ------------------------------------------------------------------------------------------ #
def _fake_hygiene(strict: bool):
    """A stand-in for tanitad.train.config_hygiene with the documented API."""
    m = types.SimpleNamespace()

    class Undeclared(AttributeError):
        pass

    def is_strict(o):
        return strict

    def undeclared(root, path="cfg"):
        out = []
        for p, o in LG._dataclass_instances(root):
            for k in vars(o):
                if k not in {f.name for f in dataclasses.fields(o)}:
                    out.append((p, type(o).__qualname__, k))
        return out
    m.is_strict, m.undeclared_attributes, m.UndeclaredConfigAttribute = is_strict, undeclared, \
        Undeclared
    return m


@dataclasses.dataclass
class _Enc:
    equalize_bottom_rows: int = 0


@dataclasses.dataclass
class _Core:
    encoder: _Enc = dataclasses.field(default_factory=_Enc)


@dataclasses.dataclass
class _Cfg:
    core: _Core = dataclasses.field(default_factory=_Core)


@dataclasses.dataclass
class _StrictEnc:
    equalize_bottom_rows: int = 0

    def __setattr__(self, k, v):
        if k not in {f.name for f in dataclasses.fields(self)}:
            raise AttributeError(k)
        object.__setattr__(self, k, v)


@dataclasses.dataclass
class _StrictCfg:
    enc: _StrictEnc = dataclasses.field(default_factory=_StrictEnc)

    def __setattr__(self, k, v):
        if k not in {f.name for f in dataclasses.fields(self)}:
            raise AttributeError(k)
        object.__setattr__(self, k, v)


def test_G_HYG_open_config_classes_FAIL_on_the_active_probe():
    reasons, det = LG.judge_hygiene(_fake_hygiene(strict=False), _Cfg())
    assert len(det["probe_assignments_accepted"]) == 3
    assert any("do NOT refuse" in r for r in reasons)
    assert any("SUCCEEDED on 3" in r for r in reasons)


def test_G_HYG_the_historical_undeclared_attribute_FAILS():
    cfg = _Cfg()
    cfg.core.encoder.trunk_equalize_bottom_rows = 43   # the refc_v3_train.py:381 pattern
    reasons, det = LG.judge_hygiene(_fake_hygiene(strict=True), cfg)
    assert det["undeclared_attributes"] == [["cfg.core.encoder", "_Enc",
                                              "trunk_equalize_bottom_rows"]]
    assert reasons and "UNDECLARED" in reasons[0]


def test_G_HYG_strict_classes_and_a_clean_tree_PASS():
    reasons, det = LG.judge_hygiene(_fake_hygiene(strict=True), _StrictCfg())
    assert reasons == [] and det["probe_assignments_accepted"] == []


def test_a_missing_module_is_reported_absent_and_the_arm_blocks_a_present_one():
    ctx = types.SimpleNamespace(arm=None)
    assert LG.import_optional(ctx, "tanitad.train.no_such_module_xyz", "x")[0] is None
    ctx.arm = "missing_colorsys_module"
    mod, why = LG.import_optional(ctx, "colorsys", "missing_colorsys_module")
    assert mod is None and "gate arm" in why


# ------------------------------------------------------------------------------------------ #
# G-LIVE (pure verdict)                                                                        #
# ------------------------------------------------------------------------------------------ #
def _steps(n, keys=("loss", "traj", "cls", "cascade"), bad=()):
    return [{"keys": sorted(keys), "nonfinite": sorted(bad),
             "vals": {k: float(i + 1) for k in keys}} for i in range(n)]


def _terms(f3=True):
    t = [{"id": "planner", "keys": ["loss", "traj", "cls"], "min_frac": 1.0, "why": "p"}]
    if f3:
        t.append({"id": "f3_per_layer", "keys": ["cascade"], "min_frac": 1.0, "why": "F3"})
    return t


def _live(steps, terms=None, rows=None, grads=None, **kw):
    n = len(steps)
    rows = rows if rows is not None else [{"step": n, **steps[-1]["vals"]}]
    grads = grads or {"n_trainable_params": 3, "dead_groups": [], "nonfinite_params": []}
    return LG.judge_live(terms if terms is not None else _terms(), kw.pop("problems", []),
                         steps, expect_steps=n, summary={"done": True, "step": n},
                         train_rows=rows, grads=grads, prior=kw.pop("prior", None),
                         exit_msg=kw.pop("exit_msg", None),
                         admitted_dead=kw.pop("admitted_dead", None))


def test_G_LIVE_the_F3_defect_as_shipped_FAILS_naming_cascade():
    reasons, _ = _live(_steps(30, keys=("loss", "traj", "cls")))
    assert any("`cascade`" in r and "ABSENT from all 30" in r for r in reasons)


def test_G_LIVE_passes_when_every_declared_term_is_present_and_finite():
    reasons, det = _live(_steps(30))
    assert reasons == [] and det["terms"][-1]["present_steps"] == 30


def test_G_LIVE_non_finite_and_record_drift_FAIL():
    assert any("non-finite" in r for r in _live(_steps(30, bad=("cascade",)))[0])
    s = _steps(30)
    row = {"step": 30, "loss": 30.0, "traj": 30.0, "cls": 30.0}           # cascade computed,
    assert any("ABSENT from the metrics.jsonl row" in r                   # not logged
               for r in _live(s, rows=[row])[0])
    row_bad = {"step": 30, "loss": 30.0, "traj": 30.0, "cls": 30.0, "cascade": 29.0}
    assert any(r.startswith("control:") for r in _live(s, rows=[row_bad])[0])


def test_G_LIVE_a_dead_leaf_module_FAILS():
    g = {"n_trainable_params": 3, "nonfinite_params": [],
         "dead_groups": [{"group": "core.decoder.cascade.control_heads.0", "numel": 10}]}
    reasons, _ = _live(_steps(30), grads=g)
    assert any("core.decoder.cascade.control_heads.0" in r for r in reasons)


# ---- refcv7 (PI 2026-09-27 ~20:55): the 10 dead groups of the launch smoke, admitted by FLAG ---- #
_R7_DEAD_TEN = ("core.strategic.gru", "core.strategic.proj", "nav_to_str", "str_goal_head",
                "gstr_embed", "gstr_film", "core.decoder.ctx_to_cond", "core.route_head",
                "core.decoder.lat_to_anchor", "core.decoder.lon_to_anchor")


def _r7_admitted(argv):
    """exactly as the G-LIVE call site filters the profile table"""
    return {g: why for g, flag, why in LG.PROFILES["refcv7"]["live_dead_admitted"]
            if LG.has_flag(argv, flag)}


def test_G_LIVE_refcv7_profile_admits_EXACTLY_the_ten_measured_dead_groups():
    rows = LG.PROFILES["refcv7"]["live_dead_admitted"]
    assert tuple(sorted(g for g, _f, _w in rows)) == tuple(sorted(_R7_DEAD_TEN))
    assert {g: f for g, f, _w in rows} == {
        "core.strategic.gru": "--no-strategic", "core.strategic.proj": "--no-strategic",
        "nav_to_str": "--no-strategic", "str_goal_head": "--no-strategic",
        "gstr_embed": "--no-strategic", "gstr_film": "--no-strategic",
        "core.decoder.ctx_to_cond": "--no-strategic", "core.route_head": "--no-strategic",
        "core.decoder.lat_to_anchor": "--graft-tac8-prior",
        "core.decoder.lon_to_anchor": "--graft-tac8-prior"}
    assert all(w for _g, _f, w in rows)                     # every row carries its reason
    assert "live_dead_admitted" not in LG.PROFILES["refcv6"]   # other profiles are unchanged


def test_G_LIVE_the_ten_dead_groups_PASS_on_the_launch_flags():
    dead = [{"group": g, "numel": 7} for g in _R7_DEAD_TEN]
    reasons, det = _live(_steps(30), grads={"n_trainable_params": 20, "dead_groups": dead,
                                            "nonfinite_params": []},
                         admitted_dead=_r7_admitted(["--no-strategic", "--graft-tac8-prior"]))
    assert not any("ZERO gradient" in r for r in reasons)
    assert sorted(det["dead_groups_admitted"]) == sorted(_R7_DEAD_TEN)


def test_G_LIVE_RED_admission_is_void_without_its_flag():
    dead = [{"group": g, "numel": 7} for g in _R7_DEAD_TEN]
    grads = {"n_trainable_params": 20, "dead_groups": dead, "nonfinite_params": []}
    r1, _ = _live(_steps(30), grads=grads, admitted_dead=_r7_admitted(["--graft-tac8-prior"]))
    assert any("8 declared-trainable leaf module(s) received ZERO gradient" in r for r in r1)
    r2, _ = _live(_steps(30), grads=grads, admitted_dead=_r7_admitted(["--no-strategic"]))
    assert any("2 declared-trainable leaf module(s) received ZERO gradient" in r for r in r2)


def test_G_LIVE_RED_any_OTHER_dead_group_still_FAILS():
    dead = [{"group": g, "numel": 7} for g in _R7_DEAD_TEN] + [
        {"group": "tac_decoder_v6.lat_head", "numel": 257}]
    reasons, _ = _live(_steps(30), grads={"n_trainable_params": 21, "dead_groups": dead,
                                          "nonfinite_params": []},
                       admitted_dead=_r7_admitted(["--no-strategic", "--graft-tac8-prior"]))
    assert any("1 declared-trainable leaf module(s) received ZERO gradient" in r
               and "tac_decoder_v6.lat_head" in r for r in reasons)


def test_the_MODEL_job_runs_CPU_only_on_every_host_and_the_smoke_does_not(tmp_path, monkeypatch):
    """MEASURED 2026-09-27 on Thor: G-EVAL crashed on 'tensors on two devices' because the model
    job built the trainer's model on CUDA while its identity probe runs on CPU."""
    seen = {}

    class _P:
        def __init__(self, cmd, env=None, **kw):
            seen[cmd[cmd.index("--checks") + 1]] = dict(env or {})
        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(LG.subprocess, "Popen", _P)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    ctx = LG.Ctx(profile="refcv7", tree=str(tmp_path), commit="0" * 40, argv=["--x"],
                 out_dir=str(tmp_path), path_map=[], tree_sha256="0" * 64,
                 argv_sha256="0" * 64, options={}, arm=None)
    for job, checks in (("model", ["G-HYG", "G-DVB", "G-EVAL"]), ("smoke", ["G-LIVE", "G-CKPT"])):
        try:
            LG.run_job(ctx, tmp_path / "ctx.json", job, checks, "python")
        except Exception:                                   # noqa: BLE001 -- only the env matters
            pass
    assert seen["G-HYG,G-DVB,G-EVAL"]["CUDA_VISIBLE_DEVICES"] == ""
    assert seen["G-LIVE,G-CKPT"].get("CUDA_VISIBLE_DEVICES") != ""


def test_the_MODEL_job_drops_trunk_compile_on_every_host(tmp_path):
    """MEASURED 2026-09-27 on Thor: a CPU-only model job that kept --trunk-compile crashed G-EVAL in
    Inductor's CPU C++ build. The drop must not depend on --cpu-only."""
    ctx = LG.Ctx(profile="refcv7", tree=str(tmp_path), commit="0" * 40,
                 argv=["--arm", "hier", "--trunk-compile", "--batch", "16"],
                 out_dir=str(tmp_path), path_map=[], tree_sha256="0" * 64,
                 argv_sha256="0" * 64, options={}, arm=None)
    out, rec = LG._sentinel_argv(ctx, (), tmp_path)
    assert "--trunk-compile" not in out
    assert out == ["--arm", "hier", "--batch", "16"]
    assert "--trunk-compile" in rec["dropped"]


def test_G_EVAL_compares_under_the_loaders_eval_settings_and_restores_the_flags():
    """MEASURED 2026-09-27 on Thor: requires_grad left as trained made 74 of 188 outputs differ between
    two IDENTICAL models; the comparison must run with the loader's eval-time flags, then restore."""
    import inspect
    src = inspect.getsource(LG.job_model)
    i_off = src.index("p_.requires_grad_(False)")
    i_call = src.index("eval_identity(ctx, T, model, args, scratch, probe_out)")
    i_restore = src.index("p_.requires_grad_(f_)")
    assert i_off < i_call < i_restore
    assert "finally:" in src[i_call:i_restore]


def test_smoke_runs_with_the_TRAINER_argv_as_the_process_argv():
    """MEASURED 2026-09-27: the trainer stamps config.json['argv'] = sys.argv[1:]; in-process it
    recorded the gate's own 'check --ctx ...' and G-EVAL crashed. The smoke must set sys.argv."""
    import inspect
    src = inspect.getsource(LG.run_smoke)
    assert 'pat.set(sys, "argv"' in src
    i_set, i_main = src.index('pat.set(sys, "argv"'), src.index("T.main(list(argv))")
    assert i_set < i_main                                    # set BEFORE the trainer runs


def test_G_LIVE_a_constant_loss_is_a_disconnected_graph():
    s = [{"keys": ["cascade", "cls", "loss", "traj"], "nonfinite": [],
          "vals": {"loss": 1.0, "traj": 1.0, "cls": 1.0, "cascade": 1.0}} for _ in range(5)]
    assert any("CONSTANT" in r for r in _live(s)[0])


def test_G_LIVE_the_residual_prior_zero_where_v0_positive_FAILS():
    p = {"found": True, "key": "residual_prior", "n_rows_v0_pos": 32,
         "n_rows_v0_pos_prior_zero": 5, "n_nonfinite": 0}
    assert any("ZERO on 5 of 32" in r for r in _live(_steps(30), prior=p)[0])
    p2 = {"found": False, "looked_for": ["residual_prior"], "out_keys_sample": ["traj"]}
    assert any("no prior tensor" in r for r in _live(_steps(30), prior=p2)[0])


def test_G_LIVE_rules_are_EXHAUSTIVE_over_the_trainer_registries():
    """A weight flag or diffusion flag the trainer knows and the gate does not is a FAIL."""
    from tanitad.models import refcv6_diffusion as rv6
    fields = {f.name for f in dataclasses.fields(rv6.DiffusionFlags)}
    assert fields <= set(LG.LIVE_FLAG_RULES), sorted(fields - set(LG.LIVE_FLAG_RULES))
    fake = types.SimpleNamespace(REFC_WEIGHT_GATES={"w_new_head": {}}, _rv6=rv6,
                                 refcv6_flags_from_args=lambda a: None)
    terms, problems = LG.declared_terms(fake, types.SimpleNamespace())
    assert problems == ["the trainer registers weight `w_new_head` and G-LIVE has no rule for it"]


def test_G_LIVE_derives_cascade_from_the_F3_flag():
    from tanitad.models import refcv6_diffusion as rv6
    fake = types.SimpleNamespace(REFC_WEIGHT_GATES={}, _rv6=rv6,
                                 refcv6_flags_from_args=lambda a: rv6.DiffusionFlags(
                                     f3_per_layer=True))
    terms, problems = LG.declared_terms(fake, types.SimpleNamespace())
    assert problems == [] and ["cascade"] in [t["keys"] for t in terms]


# ------------------------------------------------------------------------------------------ #
# G-CLOCK (pure arithmetic)                                                                    #
# ------------------------------------------------------------------------------------------ #
def _clip_windows(key, g0=0.113, dt=0.1006667, ns=3, w=8, source="sidecar", legacy=False):
    out = []
    for t in range(0, 120, 7):
        r = t + w - 1
        t_true = g0 + (r + ns - 1) * dt
        out.append((key, r * 0.1 if legacy else t_true, t_true, r * 0.1, source))
    return out


def test_G_CLOCK_the_true_clock_passes_and_the_historical_one_does_not():
    res = LG.judge_clock_windows(_clip_windows("clipA"), 0.05, n_clips=1, cap=0.01)
    assert (res["n_windows"], res["n_violations"], res["n_unverified_clips"]) == (18, 0, 0)
    assert res["legacy_control_violations"] == 18                # r = 80: 8.3677 vs 8.0
    legacy = LG.judge_clock_windows(_clip_windows("clipA", legacy=True), 0.05, n_clips=1,
                                    cap=0.01)
    assert (legacy["n_violations"], legacy["n_violating_clips"]) == (18, 1)
    assert legacy["worst_abs_s"] == 0.398338                     # r = 126: 12.9983 vs 12.6


def test_G_CLOCK_unverified_clips_are_admitted_only_up_to_the_cap():
    """FIX-2's G3 rule (fixes agent): 22 of 4,369 train clips on a fallback clock pass 1 %."""
    fb = [w for k in range(22) for w in _clip_windows(f"fb{k}", source="pose_dt", legacy=True)]
    ok = LG.judge_clock_windows(_clip_windows("m0") + fb, 0.05, n_clips=4369, cap=0.01)
    assert (ok["n_violations"], ok["n_unverified_clips"], ok["cap_ok"]) == (0, 22, True)
    assert ok["fallback_worst_abs_s_vs_reference"] == 0.398338   # stated, never hidden
    # the eval split's 3 of 139 (2.16 %) is above the cap -- decision E2, not the gate's
    ev = [w for k in range(3) for w in _clip_windows(f"e{k}", source="nominal_dt")]
    bad = LG.judge_clock_windows(ev, 0.05, n_clips=139, cap=0.01)
    assert (bad["n_unverified_clips"], bad["cap_ok"]) == (3, False)
    # a fallback clip with NO reference is unmeasured -- reported None, never an exact 0.0
    nr = LG.judge_clock_windows([("f", 1.0, None, 1.0, "nominal_dt")], 0.05, n_clips=1, cap=1.0)
    assert nr["fallback_worst_abs_s_vs_reference"] is None
    assert nr["n_fallback_clips_with_reference"] == 0
    # MEASURED but with no independent reference counts as unverified too
    unref = LG.judge_clock_windows([("u", 1.0, None, 1.0, "sidecar")], 0.05, n_clips=1, cap=0.01)
    assert (unref["n_measured_but_unreferenced"], unref["cap_ok"]) == (1, False)


def test_G_CLOCK_the_gate_never_raises_the_cap():
    p = LG.PROFILES["refcv7"]
    fake_t = types.SimpleNamespace(LABEL_CLOCK_MAX_UNVERIFIED_FRAC=0.01)
    cap, reasons, _ = LG.clock_cap(p, fake_t, ["--label-clock-max-unverified", "0.03"])
    assert cap == 0.01 and reasons and "does not raise the cap" in reasons[0]
    assert LG.clock_cap(p, fake_t, ["--label-clock-max-unverified", "0.005"])[1] == []
    loose_t = types.SimpleNamespace(LABEL_CLOCK_MAX_UNVERIFIED_FRAC=0.05)
    assert LG.clock_cap(p, loose_t, [])[0] == 0.01      # a looser trainer constant is not taken


# ------------------------------------------------------------------------------------------ #
# FIX-4 / FIX-5 launch-prep rules (coordinator 2026-09-26)                                     #
# ------------------------------------------------------------------------------------------ #
def test_FIX4_nav_compliance_needs_a_recorded_tau_from_the_train_split(tmp_path):
    p = LG.PROFILES["refcv7"]
    argv = ["--graft-nav-compliance", "--nav-compliance-tau-rad", "0.105",
            "--v2-cache", "/d/train", "--v7-labels", "/d/train_labels.jsonl.gz"]
    assert "without a RECORDED tau" in LG.nav_tau_reasons(p, argv, None)[0][0]
    rec = tmp_path / "tau.json"
    rec.write_text(json.dumps({"status": "OK", "tau": 0.105, "split_v2_cache": "/d/train",
                               "labels": "/d/train_labels.jsonl.gz"}), encoding="utf-8")
    assert LG.nav_tau_reasons(p, argv, str(rec))[0] == []
    other = LG.set_flag(argv, "--nav-compliance-tau-rad", ["0.2"])
    assert any("not the recorded tau" in r for r in LG.nav_tau_reasons(p, other, str(rec))[0])
    evalsplit = LG.set_flag(argv, "--v2-cache", ["/d/eval139"])
    assert any("derived on split_v2_cache" in r
               for r in LG.nav_tau_reasons(p, evalsplit, str(rec))[0])
    assert LG.nav_tau_reasons(p, ["--arm", "hier"], None) == ([], {"on": False})


def test_FIX5_a_required_sidecar_meta_that_is_absent_is_a_finding(tmp_path):
    side = tmp_path / "speed_max_train.jsonl"
    side.write_text("{}\n", encoding="utf-8")
    argv = ["--speed-max-sidecar-v6", side.as_posix(), "--out", "/o"]
    man = LG.data_manifest(argv, LG.PROFILES["refcv7"], [])
    assert man["--speed-max-sidecar-v6:meta"]["kind"] == "missing"
    probs = LG.data_manifest_problems(man)
    assert len(probs) == 1 and probs[0].startswith("FIX-5:")
    Path(str(side) + ".meta.json").write_text('{"source_md5": "x"}', encoding="utf-8")
    man2 = LG.data_manifest(argv, LG.PROFILES["refcv7"], [])
    assert man2["--speed-max-sidecar-v6:meta"]["kind"] == "file"
    assert LG.data_manifest_problems(man2) == []


REQ3 = ["--graft-tac8-prior", "--graft-nav-compliance", "--speed-ceiling-filter"]
PRIOR = ["--residual-prior", "ha0_ext_pose", "--ego-history"]      # SPEC_REFCV7 10 (A5)
#: NEW-2 as SPEC_REFCV7 A6/A7/A8 fix it, and the NEW-2 builder's flag table (BUILD.md sec. 10)
MAP10 = ["--map-hires", "on", "--w-map-hires", "1.0", "--bev-source", "map_hires_pool",
         "--map-hires-decision-rule", "prior_corrected", "--map-hires-x-max-m", "100",
         "--map-hires-y-half-m", "30", "--map-hires-grad-ckpt", "on",
         "--bev-planner-crop-m", "60", "16", "--w-map", "0",
         # MAP-LIFT closed (SPEC_REFCV7 17 A12 + 20 A15 + 23 A18): the map path's two flags
         "--map-hires-near-lift-m", "20", "--map-hires-near-refine-blocks", "1"]
#: the box head as SPEC_REFCV7 A9 + A14.1 fix it (the BOX-HEAD open item closed 2026-09-27, 28d8365)
BOXA9 = ["--slot-presence-loss", "focal", "--slot-presence-prior", "0.01", "--slot-deep-supervision",
         "--slot-vis1", "--vis1-sidecar", "/home/nvidia/data/refcv7/vis1_sidecar_refcv6b1_train4369_eval139.npz",
         "--slot-query-select", "learned_ref"]


@pytest.mark.parametrize("missing", REQ3)
def test_SPEC7_each_of_the_three_selection_mechanisms_is_REQUIRED_ON(missing):
    argv = ["--arm", "hier", *PRIOR, *MAP10, *BOXA9, *[f for f in REQ3 if f != missing]]
    reasons, _ = LG.required_on_reasons(LG.PROFILES["refcv7"], argv)
    assert reasons == [f"{missing} is REQUIRED ON for a refcv7 launch (SPEC_REFCV7 7, PI ruling "
                       f"E1) and the argv does not pass it"]
    assert LG.required_on_reasons(LG.PROFILES["refcv7"],
                                  ["--arm", "hier", *PRIOR, *MAP10, *BOXA9, *REQ3])[0] == []
    assert LG.required_on_reasons(LG.PROFILES["refcv6"], ["--arm", "hier"])[0] == []


@pytest.mark.parametrize("hires", [[], ["--map-hires", "off"]])
def test_SPEC8_the_10cm_map_head_is_REQUIRED_for_refcv7(hires):
    rest = MAP10[2:]                                   # every NEW-2 lever but the switch
    reasons, _ = LG.required_on_reasons(LG.PROFILES["refcv7"],
                                        ["--arm", "hier", *PRIOR, *REQ3, *BOXA9, *rest, *hires])
    assert len(reasons) == 1 and reasons[0].startswith("--map-hires on is REQUIRED")


@pytest.mark.parametrize("flag,bad", [
    ("--bev-source", ["s16_lift"]), ("--map-hires-decision-rule", ["raw"]),
    ("--map-hires-x-max-m", ["60"]), ("--map-hires-y-half-m", ["16"]),
    ("--map-hires-grad-ckpt", ["off"]), ("--bev-planner-crop-m", ["100", "30"]),
    ("--w-map", ["1.0"]), ("--w-map-hires", ["0"])])
def test_SPEC_A6_A7_A8_every_NEW2_lever_has_ONE_admissible_value(flag, bad):
    """A6 (one lift: every consumer reads the pooled BEV; the 0.5 m head removed), A7 (100 x
    +-30 m, grad checkpointing, the 60 x +-16 m planner crop), A4 (the prior-corrected rule): a
    refcv7 argv carrying any other value is REFUSED -- and so is one that leaves it unwritten."""
    good = ["--arm", "hier", *PRIOR, *MAP10, *BOXA9, *REQ3]
    assert LG.required_on_reasons(LG.PROFILES["refcv7"], good)[0] == []
    r, _ = LG.required_on_reasons(LG.PROFILES["refcv7"], LG.set_flag(good, flag, bad))
    assert len(r) == 1 and r[0].startswith(flag), r
    r, _ = LG.required_on_reasons(LG.PROFILES["refcv7"], LG.set_flag(good, flag, None))
    assert len(r) == 1 and r[0].startswith(flag), r


def test_SPEC7_required_on_reads_the_BUILT_state_through_the_frozen_helper(tmp_path):
    """The fixes agent's frozen API (landed ab436ee): check_refcv7_required(model, args, *,
    tau_file=None) -> list[Mismatch]. The banked tau file is handed through when it exists."""
    argv = ["--arm", "hier", *PRIOR, *MAP10, *BOXA9, *REQ3]
    seen = {}

    def good(model, args, *, tau_file=None):
        seen["tau_file"] = tau_file
        return []
    lst = ("graft_tac8_prior", "graft_nav_compliance", "speed_ceiling_filter")
    ok = types.SimpleNamespace(REFCV7_REQUIRED_ON=lst, check_refcv7_required=good)
    tau = tmp_path / "tau.json"
    tau.write_text('{"tau": 0.18}', encoding="utf-8")
    assert LG.required_on_reasons(LG.PROFILES["refcv7"], argv, ok, object(), object(),
                                  tau_file=str(tau))[0] == []
    assert seen["tau_file"] == str(tau)
    LG.required_on_reasons(LG.PROFILES["refcv7"], argv, ok, object(), object(),
                           tau_file=str(tmp_path / "absent.json"))
    assert seen["tau_file"] is None                          # a missing file is not handed on
    off = types.SimpleNamespace(REFCV7_REQUIRED_ON=lst, check_refcv7_required=lambda m, a, *,
                                tau_file=None: ["speed_ceiling_filter built False"])
    r, _ = LG.required_on_reasons(LG.PROFILES["refcv7"], argv, off, object(), object())
    assert r == ["REQUIRED-ON: 1 mechanism(s) not BUILT and ON: speed_ceiling_filter built False"]
    old_name = types.SimpleNamespace(REFCV7_REQUIRED_ON=lst,
                                     check_required_on=lambda m, a: [])  # the pre-freeze name
    r, _ = LG.required_on_reasons(LG.PROFILES["refcv7"], argv, old_name, object(), object())
    assert r == ["declared_vs_built has no `check_refcv7_required` (the fixes agent's frozen "
                 "REQUIRED-ON API) -- BUILT-and-ON cannot be read"]
    drift = types.SimpleNamespace(REFCV7_REQUIRED_ON=lst,
                                  check_refcv7_required=lambda m, a: [])  # no tau_file kwarg
    r, _ = LG.required_on_reasons(LG.PROFILES["refcv7"], argv, drift, object(), object())
    assert r and "API drift" in r[0]
    r, _ = LG.required_on_reasons(LG.PROFILES["refcv7"], argv, types.SimpleNamespace(),
                                  object(), object())
    assert any("publishes no REFCV7_REQUIRED_ON" in x for x in r)


#: a data root that exists on NO host. MEASURED 2026-09-26: with the real `/home/nvidia/data`
#: these tests read Thor's REAL label file there (its sha256 0cb3c502... is, as it should be, the
#: banked record's) instead of the fixture -- a test must not depend on which host runs it
_H = "/gate-test-host/data"


def _banked_tau(tmp_path, labels_bytes=b"s2 labels v8\n", tau=0.18063741505146028):
    """The BANKED record's schema (ab436ee, written on Thor by derive_navc_tau.py)."""
    import hashlib
    lab = tmp_path / "s2_labels_v8_train.jsonl.gz"
    lab.write_bytes(labels_bytes)
    rec = tmp_path / "nav_compliance_tau_train.json"
    rec.write_text(json.dumps({
        "tau": tau, "status": "OK", "unit": "rad",
        "inputs": {"manifest": {"path": f"{_H}/refcv6-b1-416x1024-train/_v2manifest.pt",
                                "sha256": "50772f31cbdb312bff069d1762f9c08257bd149852f4ba4318b9d2787529ae8a"},
                   "labels": {"path": f"{_H}/v8labels/labels/s2_labels_v8_train.jsonl.gz",
                              "sha256": hashlib.sha256(b"s2 labels v8\n").hexdigest()}}}),
        encoding="utf-8")
    pmap = [(f"{_H}/v8labels/labels/s2_labels_v8_train.jsonl.gz", lab.as_posix())]
    return rec, pmap


def test_FIX4_the_BANKED_tau_record_binds_the_split_by_path_AND_label_content(tmp_path):
    p = LG.PROFILES["refcv7"]
    argv = ["--graft-nav-compliance", "--nav-compliance-tau-rad", "0.18063741505146028",
            "--v2-cache", f"{_H}/refcv6-b1-416x1024-train",
            "--v7-labels", f"{_H}/v8labels/labels/s2_labels_v8_train.jsonl.gz"]
    rec, pmap = _banked_tau(tmp_path)
    r, d = LG.nav_tau_reasons(p, argv, str(rec), pmap)
    assert r == [] and d["schema"].startswith("banked")
    r, _ = LG.nav_tau_reasons(p, LG.set_flag(argv, "--nav-compliance-tau-rad", ["0.18"]),
                              str(rec), pmap)
    assert r == ["--nav-compliance-tau-rad ['0.18'] is not the recorded tau 0.18063741505146028"]
    r, _ = LG.nav_tau_reasons(p, LG.set_flag(argv, "--v2-cache", [f"{_H}/eval139"]),
                              str(rec), pmap)
    assert len(r) == 1 and "not inside this arm's --v2-cache" in r[0]
    (tmp_path / "x").mkdir()
    rec2, pmap2 = _banked_tau(tmp_path / "x", labels_bytes=b"other labels\n")
    r, _ = LG.nav_tau_reasons(p, argv, str(rec2), pmap2)
    assert len(r) == 1 and "are not the labels the tau was derived on" in r[0]
    r, _ = LG.nav_tau_reasons(p, argv, str(rec), [])            # labels not readable here
    assert len(r) == 1 and "cannot be checked" in r[0]


def test_SPEC7_ceiling_mask_is_inference_only():
    p = LG.PROFILES["refcv7"]
    argv = ["--speed-ceiling-filter", "--eval-cache", "/d/e"]
    good = [{"training": False, "clipped": 0.12, "rows_empty": 0}]
    assert LG.judge_ceiling(p, argv, good, eval_in_argv=True)[0] == []
    bad = good + [{"training": True, "clipped": 0.1, "rows_empty": 0}]
    r, d = LG.judge_ceiling(p, argv, bad, eval_in_argv=True)
    assert d["n_applied_in_training"] == 1 and "ACTIVE in 1 TRAINING" in r[0]
    r, _ = LG.judge_ceiling(p, argv, [], eval_in_argv=True)
    assert r == ["the ceiling mask was never ACTIVE in an eval-mode forward -- the filter the "
                 "argv declares did not act at inference"]
    assert LG.judge_ceiling(p, ["--arm", "hier"], bad, eval_in_argv=True)[0] == []   # inert


def test_SPEC7_the_tau_FILE_stamp_path_sha256_tau_must_be_in_config_json(tmp_path):
    """Keyed on the batch-2 stamp: a {path, sha256, tau} node anywhere in config.json."""
    rec = tmp_path / "tau.json"
    rec.write_text('{"tau": 0.18063741505146028}', encoding="utf-8")
    sha = LG.sha256_file(rec)
    stamp = {"path": "/x/tau.json", "sha256": sha, "tau": 0.18063741505146028}
    assert LG.tau_config_reasons(str(rec), {"selection": {"nav_compliance_tau_file": stamp}})[0] == []
    r, _ = LG.tau_config_reasons(str(rec), {"selection": {"nav_tau_sha256": sha}})   # no stamp node
    assert r and "carries no {path, sha256, tau} stamp" in r[0]
    r, _ = LG.tau_config_reasons(str(rec), {"s": dict(stamp, tau=0.18)})             # another tau
    assert r and "carries no {path, sha256, tau} stamp" in r[0]


def test_SPEC7_the_tau_file_flag_must_name_the_banked_record(tmp_path):
    import argparse
    p = LG.PROFILES["refcv7"]
    rec = tmp_path / "tau.json"
    rec.write_text('{"tau": 0.18}', encoding="utf-8")
    other = tmp_path / "other.json"
    other.write_text('{"tau": 0.18, "x": 1}', encoding="utf-8")
    new = argparse.ArgumentParser()
    new.add_argument("--nav-compliance-tau-file")
    old = argparse.ArgumentParser()
    argv = ["--graft-nav-compliance", "--nav-compliance-tau-file", rec.as_posix()]
    assert LG.tau_file_flag_reasons(p, argv, new, str(rec))[0] == []
    r, _ = LG.tau_file_flag_reasons(p, argv, old, str(rec))
    assert r and "the trainer has no --nav-compliance-tau-file" in r[0]
    r, _ = LG.tau_file_flag_reasons(p, ["--graft-nav-compliance", "--nav-compliance-tau-file",
                                        other.as_posix()], new, str(rec))
    assert r and "is not the banked tau record" in r[0]
    r, _ = LG.tau_file_flag_reasons(p, ["--graft-nav-compliance"], new, str(rec))
    assert r and "without --nav-compliance-tau-file" in r[0]
    assert LG.tau_file_flag_reasons(p, ["--arm", "hier"], new, str(rec)) == ([], {"on": False})


def test_G_DVB_calls_the_frozen_API_with_the_drivort_forbid():
    """The fixes agent's API: check(model, args, parser, forbid_kinds=...). A refcv7 gate adds
    ("drivort",) -- train() does not."""
    seen = {}

    def check(model, args, parser, *, forbid_kinds=()):
        seen["forbid"] = forbid_kinds
        seen["parser"] = parser
        return []
    D = types.SimpleNamespace(check=check, coverage=lambda p: [], REGISTRY={"x": 1})
    model = types.SimpleNamespace(modules=lambda: iter(()))
    args = types.SimpleNamespace(equalize_bottom_rows=0)
    reasons, det = LG.judge_dvb(D, model, args, "PARSER", [], forbid_kinds=("drivort",))
    assert reasons == [] and seen == {"forbid": ("drivort",), "parser": "PARSER"}
    old = types.SimpleNamespace(check=lambda model, args: [], coverage=lambda p: [])
    reasons, _ = LG.judge_dvb(old, model, args, "PARSER", [], forbid_kinds=("drivort",))
    assert reasons and "API drift" in reasons[0]
    reasons, _ = LG.judge_dvb(None, model, args, "PARSER", [], "No module named x")
    assert reasons and "not yet landed" in reasons[0]


# ------------------------------------------------------------------------------------------ #
# SPEC_REFCV7 6.1 -- DrivoR-T levers must be OFF                                                #
# ------------------------------------------------------------------------------------------ #
@pytest.mark.parametrize("extra", [["--refcv7"], ["--w-r7-wta", "0"], ["--w-r7-scorer", "1.0"],
                                   ["--r7-nav-tau-rad", "0.2"], ["--drivort"],
                                   ["--w-drivort-wta", "1"]])
def test_refcv7_REFUSES_every_DrivoR_T_flag_today_and_after_the_rename(extra):
    argv = ["--arm", "hier", "--residual-prior", *extra, "--out", "/o"]
    reasons, det = LG.forbidden_lever_reasons(LG.PROFILES["refcv7"], argv)
    assert reasons and extra[0] in det["argv_hits"]


def test_refcv7_clean_argv_and_the_built_object():
    p = LG.PROFILES["refcv7"]
    assert LG.forbidden_lever_reasons(p, ["--arm", "hier", "--residual-prior"])[0] == []
    built = types.SimpleNamespace(refcv7_wta=object(), refcv7_scorer=None)
    reasons, det = LG.forbidden_lever_reasons(p, ["--arm", "hier"], built)
    assert det["built_hits"] == ["refcv7_wta"] and reasons
    # the generic refc family forbids nothing
    assert LG.forbidden_lever_reasons(LG.PROFILES["refc"], ["--refcv7"])[0] == []


# ------------------------------------------------------------------------------------------ #
# SPEC_REFCV7 6.2 -- NEW-2, inert when off                                                      #
# ------------------------------------------------------------------------------------------ #
def test_NEW2_is_keyed_on_the_switch():
    p = LG.PROFILES["refcv7"]
    assert LG.hires_on(p, ["--map-hires", "on"]) is True
    assert LG.hires_on(p, ["--map-hires", "off"]) is False
    assert LG.hires_on(p, ["--arm", "hier"]) is False


def test_NEW2_output_shapes_and_the_fmap_s8_passthrough():
    import torch
    p = LG.PROFILES["refcv7"]
    # A7: the 10 cm grid is 1000 x 600 (100 m x +-30 m), the 0.25 m lift 400 x 240
    good = {"fmap_s8": torch.zeros(1, 512, 52, 128),
            "perception": {"map_hires_logits": torch.zeros(1, 8, 1000, 600),
                           "map_hires_bev": torch.zeros(1, 64, 400, 240)}}
    assert LG.hires_output_reasons(p, good, (416, 1024))[0] == []
    bad = {k: v for k, v in good.items() if k != "fmap_s8"}
    assert any("`fmap_s8` is ABSENT" in r for r in LG.hires_output_reasons(p, bad, (416, 1024))[0])
    old60 = {**good, "perception": {"map_hires_logits": torch.zeros(1, 8, 600, 320),
                                    "map_hires_bev": torch.zeros(1, 64, 240, 128)}}
    assert any("are not (1000, 600)" in r
               for r in LG.hires_output_reasons(p, old60, (416, 1024))[0])


def test_NEW2_cost_gate_PI_DECISION_above_plus_25_percent():
    p = LG.PROFILES["refcv7"]
    times = [0.0] + [9.0 * i for i in range(1, 31)]               # 9.0 s/step flat
    run = {"fwd_train_calls": 30, "fmap_hits": 30, "hires_first": ([], {}),
           "step_times": times, "cuda_max_mem_gb": 50.0}
    grads = {"groups": [{"group": "core.perception.map_hires_head", "numel": 5,
                         "params": 1, "nz_params": 1}]}
    reasons, needs_pi, det = LG.judge_new2_live(p, run, {}, [], grads, cuda=True, approval=None)
    assert needs_pi and det["cost"]["s_per_step_mean"] == 9.0 and det["cost"]["ratio"] == 1.4062
    _, needs_pi2, det2 = LG.judge_new2_live(p, run, {}, [], grads, cuda=True,
                                            approval={"max_s_per_step": 9.5})
    assert not needs_pi2 and "approval" in det2["cost"]["verdict"]
    fast = {**run, "step_times": [0.0] + [7.5 * i for i in range(1, 31)]}
    assert LG.judge_new2_live(p, fast, {}, [], grads, cuda=True, approval=None)[1] is False
    assert any("needs the Thor smoke" in r
               for r in LG.judge_new2_live(p, run, {}, [], grads, cuda=False, approval=None)[0])


def test_step_cost_skips_the_warmup():
    c = LG.step_cost([0.0, 100.0, 101.0, 103.0, 105.0], warmup=1)
    assert (c["n_intervals"], c["n_used"], c["s_per_step_mean"]) == (4, 3, 1.6667)


# ------------------------------------------------------------------------------------------ #
# the supervisor, end to end in bash                                                          #
# ------------------------------------------------------------------------------------------ #
def _bash() -> str | None:
    if os.name == "nt":
        p = Path("C:/Program Files/Git/bin/bash.exe")      # never WSL's bash.exe
        return str(p) if p.is_file() else None
    return shutil.which("bash")


FAKE_TRAINER = '''import json, sys, pathlib
a = sys.argv[1:]
out = pathlib.Path(a[a.index("--out") + 1]); n = int(a[a.index("--steps") + 1])
out.mkdir(parents=True, exist_ok=True)
with open(out / "metrics.jsonl", "a", encoding="utf-8") as fh:
    for s in range(1, n + 1):
        fh.write(json.dumps({"step": s, "loss": 1.0 / s}) + "\\n")
(out / "summary.json").write_text(json.dumps({"done": True, "step": n}), encoding="utf-8")
'''


def _sup_rig(tmp_path: Path, *, extra_env: str = "", tamper: str | None = None,
             caller_env: dict | None = None):
    bash = _bash()
    if bash is None:
        pytest.skip("no bash on this host")
    tree = _mini_tree(tmp_path / "code")
    (tree / "stack" / "scripts" / "launch_gate.py").write_bytes(GATE_PY.read_bytes())
    (tree / "stack" / "scripts" / "refc_v3_train.py").write_text(FAKE_TRAINER, encoding="utf-8")
    run = tmp_path / "run"
    argv = ["--arm", "hier", "--steps", "3", "--out", run.as_posix()]
    ctx = LG.Ctx(profile="refc", tree=str(tree), commit="ab" * 20, argv=argv,
                 out_dir=str(tmp_path / "gate"), path_map=[],
                 tree_sha256=LG.tree_digest(LG.tree_manifest(tree)),
                 argv_sha256=LG.argv_sha256(argv), options={})
    key_file = tmp_path / "k.key"
    key = LG.load_key(key_file)
    _write_all(ctx)
    verdict, tok_path, _ = LG.finalize(ctx, key)
    assert verdict == "PASS"
    argv_file = tmp_path / "argv.json"
    argv_file.write_text(json.dumps(argv if tamper != "argv" else
                                    LG.set_flag(argv, "--steps", ["4"])), encoding="utf-8")
    if tamper == "token":
        t = json.loads(tok_path.read_text(encoding="utf-8"))
        t["reasons"] = ["edited"]
        tok_path.write_text(json.dumps(t), encoding="utf-8")
    runs_d = tmp_path / "runs.d"
    runs_d.mkdir()
    py = Path(sys.executable).as_posix()
    (runs_d / "arm1.env").write_text(
        f'CODE="{tree.as_posix()}"\nPYTHON="{py}"\nGATE_TOKEN="{tok_path.as_posix()}"\n'
        f'GATE_ARGV_FILE="{argv_file.as_posix()}"\nGATE_KEY_FILE="{key_file.as_posix()}"\n'
        f'POLL_S=1\nMAX_RELAUNCH=1\n{extra_env}', encoding="utf-8")
    shim = tmp_path / "bin"
    shim.mkdir()
    (shim / "flock").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8", newline="\n")
    os.chmod(shim / "flock", 0o755)
    env = dict(os.environ, MSYS_NO_PATHCONV="1", **(caller_env or {}))
    env.pop(LG.ARM_ENV, None)
    path_sep = ":"
    shim_posix = shim.as_posix()
    if os.name == "nt":
        shim_posix = "/" + shim_posix[0].lower() + shim_posix[2:]
    cmd = [bash, "-c", f'export PATH="{shim_posix}{path_sep}$PATH"; '
                       f'bash "{SUP_SH.as_posix()}" arm1 "{runs_d.as_posix()}"; echo "SUPRC=$?"']
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=300)
    rc = int(p.stdout.strip().rsplit("SUPRC=", 1)[-1].split()[0])
    return rc, p.stdout + p.stderr, run


def test_supervisor_RUNS_the_trainer_on_a_valid_token_and_writes_its_done_marker(tmp_path):
    rc, out, run = _sup_rig(tmp_path)
    assert rc == 0, out[-3000:]
    assert json.loads((run / "summary.json").read_text(encoding="utf-8"))["done"] is True
    v = json.loads((run / "gate_verify_1.json").read_text(encoding="utf-8"))
    assert v["verdict"] == "MATCH"
    assert json.loads((run / "gate_exec_1.json").read_text(encoding="utf-8"))["verdict"] == "MATCH"


def test_supervisor_REFUSES_when_the_argv_changed_after_the_pass(tmp_path):
    rc, out, run = _sup_rig(tmp_path, tamper="argv")
    assert rc == 5, out[-3000:]
    assert not (run / "metrics.jsonl").exists()          # the trainer never started
    assert not (run / "gate_exec_1.json").exists()       # refused BEFORE exec: layer 1 alone
    v = json.loads((run / "gate_verify_1.json").read_text(encoding="utf-8"))
    assert v["verdict"] == "REFUSED" and any("argv differs" in r for r in v["reasons"])


def test_supervisor_REFUSES_a_hand_edited_token(tmp_path):
    rc, out, run = _sup_rig(tmp_path, tamper="token")
    assert rc == 5, out[-3000:]
    v = json.loads((run / "gate_verify_1.json").read_text(encoding="utf-8"))
    assert any("HMAC" in r for r in v["reasons"])


def test_supervisor_takes_its_variables_from_the_MANIFEST_never_the_callers_environment(
        tmp_path):
    """MEASURED 2026-09-27 on Thor: an evidence runner's exported `STEPS=unit,...` was read as the
    manifest's step count. What the caller exports never reaches the manifest's variables."""
    rc, out, run = _sup_rig(tmp_path, caller_env={"STEPS": "999", "TRAIN_CMD": "python x.py",
                                                  "OUT_DIR": "/elsewhere"})
    assert rc == 0, out[-3000:]
    assert json.loads((run / "summary.json").read_text(encoding="utf-8"))["done"] is True


def test_supervisor_REFUSES_a_free_form_TRAIN_CMD(tmp_path):
    rc, out, _ = _sup_rig(tmp_path, extra_env='TRAIN_CMD="python train.py"\n')
    assert rc == 5 and "TRAIN_CMD" in out


def test_supervisor_REFUSES_a_manifest_that_disagrees_with_the_gated_argv(tmp_path):
    rc, out, _ = _sup_rig(tmp_path, extra_env="STEPS=50400\n")
    assert rc == 5 and "--steps 3" in out


def test_every_child_the_supervisor_spawns_closes_the_lock_fd():
    """Rule 1, with its mutation arm: removing ONE `200>&-` must turn this red."""
    text = SUP_SH.read_text(encoding="utf-8")
    assert _lock_fd_violations(text) == []
    mutated = text.replace('sleep "$POLL_S" 200>&-', 'sleep "$POLL_S"', 1)
    assert mutated != text and _lock_fd_violations(mutated) == ['sleep "$POLL_S"']


def _lock_fd_violations(text: str) -> list[str]:
    after = text.split('exec 200>"$LOCK"', 1)[1]
    spawn = ("nohup ", "sleep ", '"$PYTHON" ', "rm -f ", "tee -a", "grep -Ec", "date -u")
    bad = []
    for ln in after.splitlines():
        s = ln.split("#", 1)[0].strip() if not ln.lstrip().startswith("#") else ""
        if any(k in s for k in spawn) and "200>&-" not in s:
            bad.append(s)
    return bad


# ------------------------------------------------------------------------------------------ #
# G-MAP (coordinator 2026-09-26): per-class signal, per-class logging, the overfit record       #
# ------------------------------------------------------------------------------------------ #
def _class_stats(zero_class=None, cells=500):
    st = {"_hooked": 30}
    for c in range(8):
        g = 0.0 if c == zero_class else 1.5
        st[str(c)] = {"cells": cells, "grad_all": g, "grad_own": g, "nonfinite": False}
    return st


def test_G_MAP_the_registered_spelling_is_literal():
    """LOGGING_SPEC_MAP10 sec. 1: CLASS_KEYS in code order 0..7, BAND_KEYS, and the key forms."""
    h = LG.PROFILES["refcv7"]["map_hires"]
    assert h["classes"] == ("nocls", "drivable", "lane", "crosswalk", "arrow", "edge", "hatched",
                            "sidewalk")
    assert h["bands"] == ("0_20", "20_40", "40_60", "60_80", "80_100")        # A7: 100 m
    keys = LG.map_expected_eval_keys(LG.PROFILES["refcv7"])
    # A8 item 3: the 40-key Watch contract (IoU) + the 40 loss shares beside it
    assert len(keys) == 80 and keys[0] == "map_hires_iou_nocls_0_20"
    assert keys[40] == "map_hires_lshare_nocls_0_20" and keys[-1] == "map_hires_lshare_sidewalk_80_100"


def test_G_MAP_every_present_class_carries_signal_and_a_zeroed_weight_FAILS():
    p = LG.PROFILES["refcv7"]
    assert LG.judge_map_classes(p, _class_stats())[0] == []
    r, _ = LG.judge_map_classes(p, _class_stats(zero_class=2))
    assert r == ["G-MAP: class 'lane' has 500 labelled cells in the smoke and ZERO loss "
                 "contribution (no gradient at its cells)"]
    # a class ABSENT from the smoke (fewer than M cells) is reported, not failed
    few = _class_stats(zero_class=6)
    few["6"]["cells"] = 10
    r, d = LG.judge_map_classes(p, few)
    assert r == [] and d["per_class"]["hatched"]["present"] is False
    assert LG.judge_map_classes(p, {"_hooked": 0})[0][0].startswith("G-MAP: the per-class probe")


_CLS = ("nocls", "drivable", "lane", "crosswalk", "arrow", "edge", "hatched", "sidewalk")
_BANDS = ("0_20", "20_40", "40_60", "60_80", "80_100")


def _eval_row(classes=_CLS, counts=True, raw=True, n=40.0):
    """A literal eval row in the registered spelling (LOGGING_SPEC_MAP10 sec. 2)."""
    r = {"step": 200}
    for c in classes:
        for b in _BANDS:
            r[f"eval_map_hires_iou_{c}_{b}"] = 0.5
            r[f"eval_map_hires_lshare_{c}_{b}"] = 0.125
            r[f"eval_map_hires_n_{c}_{b}"] = n
            if counts:
                r[f"eval_map_hires_inter_{c}_{b}"] = 2.0
                r[f"eval_map_hires_union_{c}_{b}"] = 4.0
            if raw:
                r[f"eval_map_hires_interraw_{c}_{b}"] = 2.0
                r[f"eval_map_hires_unionraw_{c}_{b}"] = 4.0
    return r


def test_G_MAP_eval_row_all_8_classes_x_3_bands_and_RL1_RL2_FAIL():
    p = LG.PROFILES["refcv7"]
    assert LG.map_eval_key_reasons(p, [_eval_row()])[0] == []
    # RL1: the refcv6 pattern -- drivable only
    r, d = LG.map_eval_key_reasons(p, [_eval_row(classes=("drivable",))])
    assert d["n_missing"] == 70 and "logs 10 of the 80" in r[0]
    # RL2: per-batch IoUs, no COUNTS
    r, d = LG.map_eval_key_reasons(p, [_eval_row(counts=False)])
    assert d["n_counts_bad"] == 80 and "COUNT keys are absent" in r[0]
    # weighted loss: the RAW-rule counts are required too; unweighted: not
    assert LG.map_eval_key_reasons(p, [_eval_row(raw=False)], weighted=False)[0] == []
    assert LG.map_eval_key_reasons(p, [_eval_row(raw=False)])[1]["n_counts_bad"] == 80
    # item 3: a class with no labelled cell in 0-20 m is a FAIL, not a waiver
    row = _eval_row()
    row["eval_map_hires_n_arrow_0_20"] = 0.0
    r, d = LG.map_eval_key_reasons(p, [row])
    assert d["classes_without_cells_in_gated_band"] == ["arrow"] and "not waived" in r[0]
    assert "wrote no in-run eval row" in LG.map_eval_key_reasons(p, [])[0][0]


def test_G_MAP_trainer_declared_list():
    p = LG.PROFILES["refcv7"]
    keys = LG.map_expected_eval_keys(p)
    t_ok = types.SimpleNamespace(MAP_HIRES_EVAL_KEYS=tuple(keys))
    assert LG.map_eval_list_reasons(p, t_ok)[0] == []
    t_drv = types.SimpleNamespace(MAP_HIRES_EVAL_KEYS=tuple(k for k in keys if "_drivable_" in k))
    assert "omits 70 of the 80" in LG.map_eval_list_reasons(p, t_drv)[0][0]
    assert "declares no per-class" in LG.map_eval_list_reasons(p, types.SimpleNamespace())[0][0]


def _landed_map_module(h):
    """A stand-in for `tanitad.models.map_head_hires` as landed in cef9709 (the spellings)."""
    def band_keys_for_rows(n):
        return tuple(f"{a}_{min(a + 20, n // 10)}" for a in range(0, n // 10, 20))
    return types.SimpleNamespace(
        CLASS_KEYS=tuple(h["classes"]), BAND_KEYS=("0_20", "20_40", "40_60"),
        band_keys_for_rows=band_keys_for_rows,
        per_class_key=lambda s, c, b, prefix="": f"{prefix}map_hires_{s}_{h['classes'][c]}_{b}")


def test_G_MAP_eval_keys_from_the_trainers_OWN_row_builder_on_the_BUILT_model():
    """NEW-2 as landed (cef9709) declares no list: the gate runs the trainer's own eval-row
    builder on the built model -- a branch built at the /2 extent emits 3 bands, not 5."""
    p = LG.PROFILES["refcv7"]
    h = p["map_hires"]

    def builder(acc, nb_e, model):                   # `_eval_row_from_acc` + derived_per_class
        row = {f"eval_{k}": v for k, v in acc.items()}
        for c in h["classes"]:
            for b in model.bands:
                for s in ("iou", "iouraw", "lshare"):
                    row[f"eval_map_hires_{s}_{c}_{b}"] = None
        return row
    T = types.SimpleNamespace(_eval_row_from_acc=builder)
    assert LG.map_eval_list_reasons(p, T, model=types.SimpleNamespace(bands=h["bands"]))[0] == []
    r = LG.map_eval_list_reasons(p, T, model=types.SimpleNamespace(bands=h["bands"][:3]))[0]
    assert len(r) == 1 and "eval-row builder omits 32 of the 80" in r[0]
    assert "declares no per-class" in LG.map_eval_list_reasons(p, T, model=None)[0][0]


def test_G_MAP_spelling_is_checked_at_the_DECLARED_extent_not_the_2_constant(monkeypatch):
    p = LG.PROFILES["refcv7"]
    h = p["map_hires"]
    M = _landed_map_module(h)
    monkeypatch.setitem(sys.modules, "tanitad.models.map_head_hires", M)
    r, d = LG.map_spelling_reasons(p)
    assert r == [] and d["bands_at_extent"] == ["0_20", "20_40", "40_60", "60_80", "80_100"]
    assert d["per_class_key_agrees"] is True
    drift = types.SimpleNamespace(**{**vars(M), "per_class_key": lambda s, c, b, prefix="":
                                     f"{prefix}map10_{s}_{h['classes'][c]}_{b}"})
    monkeypatch.setitem(sys.modules, "tanitad.models.map_head_hires", drift)
    assert "per_class_key spells the IoU keys" in LG.map_spelling_reasons(p)[0][0]
    old = types.SimpleNamespace(**{k: v for k, v in vars(M).items() if k != "band_keys_for_rows"})
    monkeypatch.setitem(sys.modules, "tanitad.models.map_head_hires", old)
    assert "are not the registered" in LG.map_spelling_reasons(p)[0][0]


def test_G_MAP_train_rows_loss_contributions_sum_to_the_loss():
    p = LG.PROFILES["refcv7"]
    lc = {f"map_hires_lc_{c}_{b}": 0.05 for c in _CLS for b in _BANDS}      # 40 x 0.05 = 2.0
    assert LG.map_train_row_reasons(p, [{"step": 50, "map_hires": 2.0, **lc}])[0] == []
    r, _ = LG.map_train_row_reasons(p, [{"step": 50, "map_hires": 2.1, **lc}])
    assert r and "do not sum to `map_hires`" in r[0]
    r, _ = LG.map_train_row_reasons(p, [{"step": 50, "map_hires": 1.2, "map_hires_lc_lane_0_20": 1.2}])
    assert r and "lack the per-class loss contributions" in r[0]
    assert "never reached the log" in LG.map_train_row_reasons(p, [{"step": 50, "loss": 1.0}])[0][0]


def test_G_MAP_decision_rule_must_be_declared_and_the_watch_carries_the_series(tmp_path):
    p = LG.PROFILES["refcv7"]
    assert LG.map_decision_rule_reasons(p, {"map_hires": {"decision_rule": "prior_corrected"}})[0] == []
    assert "must be DECLARED" in LG.map_decision_rule_reasons(p, {"map_hires": {}})[0][0]
    assert "cannot be seen" in LG.map_watch_reasons(p, None)[0][0]
    w = tmp_path / "watch.html"
    series = " ".join(f"eval_map_hires_iou_{c}_{b}" for c in _CLS for b in _BANDS)
    w.write_text(f"<html>{series} <div>Thin-class alarm</div></html>", encoding="utf-8")
    assert LG.map_watch_reasons(p, str(w))[0] == []
    w.write_text(f"<html>{series}</html>", encoding="utf-8")                # RL3: no alarm tile
    assert LG.map_watch_reasons(p, str(w))[0] == ["G-MAP: the Watch carries no thin-class alarm tile"]


#: the PREREG_G_MAP_OVERFIT literals, spelled out here (never read from the profile under test)
_GMO_BARS = {"nocls": 0.85, "drivable": 0.85, "sidewalk": 0.85, "lane": 0.5, "crosswalk": 0.5,
             "arrow": 0.5, "edge": 0.5, "hatched": 0.5}
_GMO_FRAMES = [("10497f0d664b", 10), ("10497f0d664b", 110), ("10497f0d664b", 130),
               ("10497f0d664b", 150), ("05c575ed45be", 50), ("05c575ed45be", 85),
               ("05c575ed45be", 105), ("05c575ed45be", 130), ("104d79ae052d", 10),
               ("104d79ae052d", 90), ("104d79ae052d", 115), ("104d79ae052d", 135),
               ("0dbd6c9cd776", 15), ("0dbd6c9cd776", 35), ("0dbd6c9cd776", 60),
               ("0dbd6c9cd776", 90)]
_THIN = ["lane", "crosswalk", "arrow", "edge", "hatched"]


def _gmo_record(commit, argv_sha, cw_sha, **over):
    """A record in the NEW-2 harness's schema (`map_hires_overfit.py` main(): schema
    `tanitad.g_map_overfit_record/1`, `launch_commit` / `launch_argv_sha256`, the verdict with
    MAIN / regression_arms / controls / time_guard, per-class `results.healthy.final`)."""
    rec = {"schema": "tanitad.g_map_overfit_record/1",
           "launch_commit": commit, "launch_argv_sha256": argv_sha,
           "class_weights": {"sha256": cw_sha, "definition_id": "sqrt_mf"},
           "decision_rule": "prior_corrected", "extent": {"x_max_m": 100.0, "y_half_m": 30.0},
           "frames_sha12": [f for f, _ in _GMO_FRAMES], "raw_frames": [r for _, r in _GMO_FRAMES],
           "frameset_md5": "4eafa03c2b6a6e6d6336be1d78acb91d",
           # SPEC_REFCV7 23 (A18): the protocol the record RAN (the harness's own fields)
           "spec_sha256": "5cb4f6fc4a8c15660237f541f077bfb334337dd287f674f8de02b7d414077c58",
           "optimiser": {"name": "AdamW", "lr": 0.001, "betas": [0.9, 0.999], "eps": 1e-08,
                         "weight_decay": 0.0, "batch": 4, "steps": 3000, "seed": 0,
                         "lr_decay": {"kind": "cosine_to_zero", "start_step": 2700}},
           "near_lift_m": 20.0, "near_refine_blocks": 1,
           "results": {"healthy": {"final": {"n": {c: 5000 for c in _GMO_BARS},
                                             "iou": {c: b + 0.05 for c, b in _GMO_BARS.items()},
                                             "ce_mean": {c: 0.2 for c in _GMO_BARS}},
                                   "step0": {"ce_mean": {c: 2.0 for c in _GMO_BARS}},
                                   "loss_finite_every_step": True}},
           "verdict": {"band": "0_20", "decision_rule": "prior_corrected",
                       "MAIN": {"presence_short": {}, "inconclusive": False,
                                "bars": {c: True for c in _GMO_BARS},
                                "ce_ratio_ok": {c: True for c in _GMO_BARS},
                                "loss_finite_every_step": True, "verdict": "PASS"},
                       "regression_arms": {
                           "lane_w0": {"ran": True, "must_fail": ["lane"], "failed": ["lane"],
                                       "rule": "any", "failed_as_required": True},
                           "s8_zeros": {"ran": True, "must_fail": _THIN, "failed": list(_THIN),
                                        "rule": "all", "failed_as_required": True},
                           "near_block_zeros": {"ran": True, "must_fail": ["edge"],
                                                "failed": ["edge"], "rule": "any",
                                                "failed_as_required": True}},
                       "controls": {"C1_constant_drivable": {"reproduced": True},
                                    "C2_gt_as_logits": {"reproduced": True},
                                    "C3_rule_identity_w_ones": {"reproduced": True}},
                       "controls_reproduced": True,
                       "time_guard": {"kinds": ["time_1ms"], "time_1ms_on_every_clip": True},
                       "G_MAP_OVERFIT": "PASS"}}
    rec.update(over)
    return rec


def test_G_MAP_OVERFIT_record_is_a_launch_prerequisite_bound_to_the_launch(tmp_path):
    p = LG.PROFILES["refcv7"]
    commit, argv_sha, cw_sha = "ab" * 20, "cd" * 32, "ef" * 32
    rec = tmp_path / "g_map_overfit.json"
    rec.write_text(json.dumps(_gmo_record(commit, argv_sha, cw_sha)), encoding="utf-8")
    assert LG.judge_map_overfit(p, str(rec), commit, argv_sha=argv_sha,
                                class_weights_sha256=cw_sha)[0] == []
    assert "no overfit PASS record" in LG.judge_map_overfit(p, None, commit)[0][0]
    # SPEC_REFCV7 A11: the CODE is bound by the closure record (judge_closure), so the record's
    # own `launch_commit` is informative -- another commit is NOT a refusal here
    r, _ = LG.judge_map_overfit(p, str(rec), "12" * 20, argv_sha=argv_sha,
                                class_weights_sha256=cw_sha)
    assert r == []
    r, _ = LG.judge_map_overfit(p, str(rec), commit, argv_sha="00" * 32,
                                class_weights_sha256=cw_sha)
    assert len(r) == 1 and "not the launch argv" in r[0]
    r, _ = LG.judge_map_overfit(p, str(rec), commit, argv_sha=argv_sha,
                                class_weights_sha256="99" * 32)
    assert len(r) == 1 and "the launch argv declares 9999" in r[0]
    # a record run WITHOUT --launch-commit / --launch-argv-sha256 (the harness default ""),
    # and one without the 1 ms time guard: named FAILs, never a pass
    today = _gmo_record(commit, argv_sha, cw_sha, launch_commit="", launch_argv_sha256="")
    today["verdict"]["time_guard"] = {"kinds": ["pose_5cm"], "time_1ms_on_every_clip": False}
    rec.write_text(json.dumps(today), encoding="utf-8")
    r, _ = LG.judge_map_overfit(p, str(rec), commit, argv_sha=argv_sha, class_weights_sha256=cw_sha)
    assert sum("bound to" in x for x in r) == 1 and any("1 ms label-time guard" in x for x in r)


def test_G_MAP_OVERFIT_every_class_present_at_its_registered_bar_and_the_must_fail_arms(tmp_path):
    p = LG.PROFILES["refcv7"]
    commit, argv_sha, cw_sha = "ab" * 20, "cd" * 32, "ef" * 32
    rec = tmp_path / "g_map_overfit.json"

    def judge(r_):
        rec.write_text(json.dumps(r_), encoding="utf-8")
        return LG.judge_map_overfit(p, str(rec), commit, argv_sha=argv_sha,
                                    class_weights_sha256=cw_sha)[0]
    few = _gmo_record(commit, argv_sha, cw_sha)
    few["results"]["healthy"]["final"]["n"]["arrow"] = 999                      # presence floor
    assert judge(few) == ["G-MAP-OVERFIT: class 'arrow' has 999 scored cells < 1000 -- "
                          "INCONCLUSIVE => FAIL (prereg sec. 6.1)"]
    low = _gmo_record(commit, argv_sha, cw_sha)
    low["results"]["healthy"]["final"]["iou"]["edge"] = 0.4    # below the REGISTERED 0.5, even
    low["verdict"]["MAIN"]["bars"]["edge"] = True               # if the record says it passed
    assert judge(low) == ["G-MAP-OVERFIT: class 'edge' IoU 0.4 < the registered bar 0.5"]
    ext = _gmo_record(commit, argv_sha, cw_sha, extent={"x_max_m": 60.0, "y_half_m": 16.0})
    assert len(judge(ext)) == 2 and "SPEC_REFCV7 12" in judge(ext)[0]
    ce = _gmo_record(commit, argv_sha, cw_sha)
    ce["verdict"]["MAIN"]["ce_ratio_ok"]["hatched"] = False
    assert judge(ce) == ["G-MAP-OVERFIT: the per-class CE did not fall to <= 0.5x its step-0 "
                         "value for ['hatched'] (prereg sec. 6.4)"]
    s8 = _gmo_record(commit, argv_sha, cw_sha)
    s8["verdict"]["regression_arms"]["s8_zeros"]["failed"] = ["lane", "crosswalk", "arrow", "edge"]
    r = judge(s8)
    assert len(r) == 1 and "'s8_zeros' must FAIL ALL of" in r[0]
    lw = _gmo_record(commit, argv_sha, cw_sha)
    lw["verdict"]["regression_arms"]["lane_w0"] = {"ran": True, "failed": []}
    assert "'lane_w0' did not FAIL ['lane']" in judge(lw)[0]
    dry = _gmo_record(commit, argv_sha, cw_sha, class_weights={"dry_run": True, "sha256": cw_sha})
    assert "not the launch weights" in judge(dry)[0]
    fr = _gmo_record(commit, argv_sha, cw_sha, raw_frames=[r_ for _, r_ in _GMO_FRAMES][:-1] + [91])
    assert "not the FROZEN 16-frame set" in judge(fr)[0]
    bad = _gmo_record(commit, argv_sha, cw_sha)
    bad["verdict"]["G_MAP_OVERFIT"] = "FAIL"
    assert judge(bad) == ["G-MAP-OVERFIT: the record's verdict is 'FAIL', not PASS"]


def test_G_MAP_OVERFIT_joins_the_required_checks_only_with_the_10cm_head(tmp_path):
    ctx = _ctx(tmp_path, argv=["--arm", "hier", "--map-hires", "on", "--out", "/o"])
    ctx = dataclasses.replace(ctx, profile="refcv7")
    key = LG.load_key(tmp_path / "k.key")
    _write_all(ctx)
    verdict, _, tok = LG.finalize(ctx, key)
    assert tok["checks"]["G-MAP-OVERFIT"]["status"] == "MISSING"
    off = _ctx(tmp_path / "off")
    _write_all(off)
    assert "G-MAP-OVERFIT" not in LG.finalize(off, key)[2]["checks"]



def test_exec_REFUSES_on_its_own_and_never_starts_the_trainer(tmp_path):
    """Layer 2 alone: `launch_gate.py exec` re-verifies immediately before it would exec."""
    ctx, key, _, _, _ = _pass_token(tmp_path)
    (Path(ctx.tree) / "stack" / "scripts" / "launch_gate.py").write_bytes(GATE_PY.read_bytes())
    ctx2 = dataclasses.replace(ctx, tree_sha256=LG.tree_digest(LG.tree_manifest(ctx.tree)),
                               out_dir=str(tmp_path / "gate2"))
    _write_all(ctx2)
    _, tok2, _ = LG.finalize(ctx2, key)
    good = tmp_path / "argv_good.json"
    good.write_text(json.dumps(ctx.argv), encoding="utf-8")
    bad = tmp_path / "argv_bad.json"
    bad.write_text(json.dumps(LG.set_flag(ctx.argv, "--steps", ["11"])), encoding="utf-8")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    env.pop(LG.ARM_ENV, None)

    def ex(argv_file, js):
        return subprocess.run([sys.executable, str(GATE_PY), "exec", "--token", str(tok2),
                               "--argv-file", str(argv_file), "--tree", ctx.tree,
                               "--key-file", str(tmp_path / "k.key"), "--json", str(js),
                               "--dry-run"], env=env, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=300)
    p = ex(bad, tmp_path / "e_bad.json")
    assert p.returncode == 5 and "ZZGATE-EXEC-DRYRUNZZ" not in p.stdout
    assert json.loads((tmp_path / "e_bad.json").read_text(encoding="utf-8"))["verdict"] == \
        "REFUSED"
    p = ex(good, tmp_path / "e_good.json")
    assert p.returncode == 0 and "ZZGATE-EXEC-DRYRUNZZ" in p.stdout, p.stdout[-800:]


def _ga_rows(steps, le, keys, drop_nonfinal=False, zero=None):
    rows = []
    for s in steps:
        r = {"step": s, "loss": 1.0}
        if not (drop_nonfinal and s != steps[-1]):
            r.update({k: (0.0 if k == zero else 0.5) for k in keys})
        rows.append(r)
    return rows


_GROUPS = {"trunk": {"exists": True, "trainable_params": 10},
           "planner": {"exists": True, "trainable_params": 5},
           "lift": {"exists": True, "trainable_params": 0},            # built, frozen: not required
           "box_decoder": {"exists": False}}


def test_G_LIVE_ga_reach_every_logged_row_every_trainable_group():
    p = LG.PROFILES["refcv7"]
    keys = ["ga_trunk", "ga_planner"]
    ok = _ga_rows([50, 100], 50, keys)
    r, d = LG.judge_ga_reach(p, ok, log_every=50, final_step=100, groups=_GROUPS, ga_on=True,
                             hires=False)
    assert r == [] and d["required_groups"] == ["planner", "trunk"]
    # the refcv6 off-by-one: ga_* only in the FINAL row -> the periodic row at 50 has none
    r, _ = LG.judge_ga_reach(p, _ga_rows([50, 100], 50, keys, drop_nonfinal=True), log_every=50,
                             final_step=100, groups=_GROUPS, ga_on=True, hires=False)
    assert len(r) == 1 and r[0].startswith("G-LIVE ga-reach: 1 of 2 logged rows lack")
    # a trainable group whose reach is 0.0 received no gradient
    r, _ = LG.judge_ga_reach(p, _ga_rows([50, 100], 50, keys, zero="ga_planner"), log_every=50,
                             final_step=100, groups=_GROUPS, ga_on=True, hires=False)
    assert len(r) == 1 and "ga_planner = 0.0" in r[0]
    # a smoke shorter than 2 x log_every cannot show a periodic row
    r, _ = LG.judge_ga_reach(p, _ga_rows([30], 50, keys), log_every=50, final_step=30,
                             groups=_GROUPS, ga_on=True, hires=False)
    assert any("< 2 x log_every" in x for x in r)
    # NEW-2: the four ga_mh_* parts are required in every logged row too
    r, _ = LG.judge_ga_reach(p, ok, log_every=50, final_step=100, groups=_GROUPS, ga_on=True,
                             hires=True)
    assert r and "ga_mh_lift" in r[0]
    # an arm with neither seam logs nothing and is not asked to
    assert LG.judge_ga_reach(p, [], log_every=50, final_step=100, groups={}, ga_on=False,
                             hires=False)[0] == []


def test_G_LIVE_the_ga_off_by_one_arm_drops_ga_keys_from_every_row_but_the_last():
    j = LG._GaFinalRowOnlyJson(json, 100)
    assert json.loads(j.dumps({"step": 50, "loss": 1.0, "ga_trunk": 0.5})) == \
        {"step": 50, "loss": 1.0}
    assert json.loads(j.dumps({"step": 100, "loss": 1.0, "ga_trunk": 0.5}))["ga_trunk"] == 0.5
    assert json.loads(j.dumps({"step": 50, "conflict": 0.1, "ga_x": 1})) == \
        {"step": 50, "conflict": 0.1, "ga_x": 1}                          # not a training row
    assert j.loads("[1]") == [1]                                         # the rest is json


def test_NEW1_residual_prior_off_is_not_NEW1_and_only_ha0_ext_pose_is_refcv7():
    """`--residual-prior` takes a MODE and defaults to `off` (the refcv6 decoder bit for bit);
    SPEC_REFCV7 10 (A5): refcv7's prior is `ha0_ext_pose` -- `off`, `ha0_ext` and `cv_yawrate`
    are REFUSED by name."""
    p = LG.PROFILES["refcv7"]
    assert LG.residual_prior_on(["--residual-prior", "ha0_ext"]) is True
    assert LG.residual_prior_on(["--residual-prior", "off"]) is False
    assert LG.residual_prior_on(["--arm", "hier"]) is False
    for mode in ("off", "ha0_ext", "cv_yawrate"):
        r = LG.forbidden_value_reasons(p, ["--residual-prior", mode])
        assert r and r[0].startswith(f"--residual-prior ['{mode}'] is REFUSED for a refcv7 "
                                     f"launch (refused values: ['off', 'ha0_ext', 'cv_yawrate']")
    assert LG.forbidden_value_reasons(p, ["--residual-prior", "ha0_ext_pose"]) == []
    assert LG.required_flag_reasons(p, ["--residual-prior", "ha0_ext_pose", "--slot-deep-supervision",
                                        "--slot-vis1"]) == [
        "profile refcv7 requires --ego-history in the launch argv (SPEC: NEW-1 is part of this "
        "arm)"]


# ------------------------------------------------------------------------------------------ #
# finalize RE-DERIVES the profile's argv rules; an open PI decision is PI-DECISION                 #
# ------------------------------------------------------------------------------------------ #
def test_a_standalone_finalize_cannot_mint_a_token_that_run_would_refuse(tmp_path):
    ctx = dataclasses.replace(_ctx(tmp_path, argv=["--arm", "hier", "--out", "/o"]),
                              profile="refcv7")
    key = LG.load_key(tmp_path / "k.key")
    _write_all(ctx)                                   # seven synthetic PASS evidences
    verdict, path, tok = LG.finalize(ctx, key)        # no profile_refusals handed in
    assert verdict == "FAIL" and path.name.startswith("FAIL_")
    rs = " || ".join(tok["reasons"])
    assert "requires --residual-prior" in rs and "REQUIRED ON" in rs and "--map-hires on" in rs


def _refcv7_argv_ctx(tmp_path, mode="ha0_ext_pose"):
    data = tmp_path / "data"
    (data / "refcv6-b1-416x1024-train").mkdir(parents=True)
    (data / "refcv6-b1-416x1024-train" / "ep_0.v2ep.pt").write_bytes(b"x")
    (data / "v8labels" / "labels").mkdir(parents=True)
    (data / "refcv7").mkdir(parents=True)                 # the box head's VIS-1 sidecar (a data input)
    (data / "refcv7" / "vis1_sidecar_refcv6b1_train4369_eval139.npz").write_bytes(b"x")
    rec, _ = _banked_tau(data / "v8labels" / "labels")
    argv = ["--arm", "hier", "--residual-prior", mode, "--ego-history", *MAP10, *BOXA9,
            "--graft-tac8-prior", "--graft-nav-compliance",
            "--nav-compliance-tau-rad", "0.18063741505146028", "--speed-ceiling-filter",
            "--v2-cache", f"{_H}/refcv6-b1-416x1024-train",
            "--v7-labels", f"{_H}/v8labels/labels/s2_labels_v8_train.jsonl.gz",
            "--out", "/o"]
    tree = _mini_tree(tmp_path / "tree")
    ctx = LG.Ctx(profile="refcv7", tree=str(tree), commit="ab" * 20, argv=argv,
                 out_dir=str(tmp_path / "gate"),
                 # the box head's sidecar is REQUIRED at its launch path (a required value): map that host too
                 path_map=[[_H, data.as_posix()], ["/home/nvidia/data", data.as_posix()]],
                 tree_sha256=LG.tree_digest(LG.tree_manifest(tree)),
                 argv_sha256=LG.argv_sha256(argv),
                 options={"nav_tau_record": str(rec),
                          "gate_inputs": {"gate:nav-tau-record": str(rec)}}, arm=None)
    for c in LG.CHECKS:                               # BASE + G-MAP-OVERFIT (the 10 cm head is on)
        LG.write_evidence(ctx.out_dir, _evidence(ctx, c))
    return ctx, rec


def test_NEW1_the_ruled_prior_PASSES_the_refused_ones_FAIL_and_the_tau_sha_is_in_the_token(
        tmp_path, monkeypatch):
    """SPEC_REFCV7 10 (A5): ha0_ext_pose + --ego-history. And the coordinator: the banked tau
    record's sha256 is IN the PASS token. (The profile's OPEN ITEMS are closed here: they have
    their own test.)"""
    monkeypatch.setitem(LG.PROFILES, "refcv7", dict(LG.PROFILES["refcv7"], open_items=()))
    ctx, rec = _refcv7_argv_ctx(tmp_path)
    key = LG.load_key(tmp_path / "k.key")
    verdict, path, tok = LG.finalize(ctx, key)
    assert verdict == "PASS", tok["reasons"]
    assert tok["binding"]["data"]["gate:nav-tau-record"]["sha256"] == LG.sha256_file(rec)
    pm = [tuple(x) for x in ctx.path_map]
    assert LG.verify_token(path, ctx.argv, ctx.tree, key=key, path_map=pm)[0]
    for mode in ("ha0_ext", "cv_yawrate", "off"):
        c2, _ = _refcv7_argv_ctx(tmp_path / mode, mode=mode)
        verdict, _, tok = LG.finalize(c2, key)
        assert verdict == "FAIL" and any(r.startswith(f"--residual-prior ['{mode}'] is REFUSED")
                                         for r in tok["reasons"]), tok["reasons"]
    c3, _ = _refcv7_argv_ctx(tmp_path / "noeh")
    c3 = dataclasses.replace(c3, argv=LG.set_flag(c3.argv, "--ego-history", None))
    c3 = dataclasses.replace(c3, argv_sha256=LG.argv_sha256(c3.argv))
    for c in LG.CHECKS:
        LG.write_evidence(c3.out_dir, _evidence(c3, c))
    verdict, _, tok = LG.finalize(c3, key)
    assert verdict == "FAIL" and any("requires --ego-history" in r for r in tok["reasons"])


def test_an_OPEN_value_decision_makes_the_token_PI_DECISION_until_it_is_recorded(tmp_path,
                                                                              monkeypatch):
    ctx, _ = _refcv7_argv_ctx(tmp_path)
    key = LG.load_key(tmp_path / "k.key")
    p = dict(LG.PROFILES["refcv7"], open_items=())          # OPEN ITEMS: their own test
    monkeypatch.setitem(LG.PROFILES, "refcv7", dict(
        p, pi_pending_values={"--arm": "which arm (a test-only open decision)"}))
    verdict, path, tok = LG.finalize(ctx, key)
    assert verdict == "PI-DECISION" and not LG.verify_token(path, ctx.argv, ctx.tree, key=key)[0]
    assert tok["reasons"][-1] == ("PI decision pending: --arm hier: which arm (a test-only open "
                                  "decision)")
    monkeypatch.setitem(LG.PROFILES, "refcv7", dict(
        p, pi_pending_values={"--arm": "which arm"},
        required_values={**p["required_values"], "--arm": "hier"}))      # the ruling, recorded
    assert LG.finalize(ctx, key)[0] == "PASS"


# ------------------------------------------------------------------------------------------ #
# G-SUITE: a SKIP is not a pass when its reason says the test could not run here                #
# ------------------------------------------------------------------------------------------ #
def test_G_SUITE_a_skip_for_a_missing_HF_cache_is_NOT_a_pass(tmp_path):
    xml = tmp_path / "junit.xml"
    xml.write_text(
        '<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest">'
        '<testcase classname="tests.test_a" name="test_ok" time="0.1"/>'
        '<testcase classname="tests.test_guard_blind_spots_fix5" name="test_stage_fp" time="0">'
        '<skipped type="pytest.skip" message="resnet34.a1_in1k weights not in the local HF cache '
        '(OSError); a SKIP is not a pass -- run where the checkpoint is cached">x</skipped>'
        '</testcase>'
        '<testcase classname="tests.test_b" name="test_eval139" time="0">'
        '<skipped type="pytest.skip" message="the eval-139 cache is not on this box">y</skipped>'
        '</testcase>'
        '<testcase classname="tests.test_c" name="test_bad" time="0"><failure message="boom"/>'
        '</testcase></testsuite></testsuites>', encoding="utf-8")
    res = LG._parse_junit(xml)
    assert res["n_tests"] == 4 and res["n_skipped"] == 2 and res["failed"] == ["tests.test_c::test_bad"]
    snp = LG.skip_not_pass(res, LG.PROFILES["refcv7"]["suite_skip_not_pass"])
    assert list(snp) == ["tests.test_guard_blind_spots_fix5::test_stage_fp"]


# ------------------------------------------------------------------------------------------ #
# SPEC 2's G-DVB arm "unwire one selection term"; NEW-1's owner check in G-LIVE                 #
# ------------------------------------------------------------------------------------------ #
def test_the_unwire_arm_removes_a_BUILT_selection_term_or_reports_that_it_could_not():
    dec = types.SimpleNamespace(navc_gate=object(), tac8_lat_to_anchor=object())
    model = types.SimpleNamespace(core=types.SimpleNamespace(decoder=dec))
    assert LG._arm_unwire_selection(model) == {"unwired": "core.decoder.navc_gate",
                                               "declared_by": "--graft-nav-compliance"}
    assert dec.navc_gate is None and dec.tac8_lat_to_anchor is not None
    assert LG._arm_unwire_selection(model)["unwired"] == "core.decoder.tac8_lat_to_anchor"
    assert LG._arm_unwire_selection(model)["unwired"] is None


def test_G_LIVE_NEW1_the_owners_plan_check_is_judged_when_the_prior_is_on():
    base = {"found": True, "key": "residual_prior_path", "n_rows_v0_pos": 32,
            "n_rows_v0_pos_prior_zero": 0, "n_nonfinite": 0}
    good = dict(base, owner_plan_check={"calls": 30, "not_ok": 0, "rows_v_gt_0": 64,
                                        "rows_v_gt_0_prior_nonzero": 64, "errors": []})
    assert _live(_steps(30), prior=good)[0] == []
    bad = dict(base, owner_plan_check={"calls": 30, "not_ok": 2, "rows_v_gt_0": 64,
                                       "rows_v_gt_0_prior_nonzero": 60, "errors": []})
    r = _live(_steps(30), prior=bad)[0]
    assert any("NOT ok on 2 of 30" in x for x in r) and any("non-zero on 60 of 64" in x for x in r)
    gone = dict(base, owner_plan_check={"unavailable": "ModuleNotFoundError: kinematic_prior"})
    assert any("plan_check is unavailable" in x for x in _live(_steps(30), prior=gone)[0])
    never = dict(base, owner_plan_check={"calls": 0, "errors": ["TypeError: x"]})
    assert any("never completed" in x and "TypeError" in x
               for x in _live(_steps(30), prior=never)[0])


# ------------------------------------------------------------------------------------------ #
# G-SUITE, consume mode (Master Mind 2026-09-27): the Thor verdict BOUND to commit + tree, the   #
# PINNED environment list, and G-SUITE-PINNED on the dev box                                   #
# ------------------------------------------------------------------------------------------ #
def _verdict(pre=("tests.test_a::t1", "tests.test_a::t2"), new=(), reg=(), nojunit=()):
    """gate_verdict.py's verdict.json shape (qland/work/thorgate_n1/gate_verdict.py)."""
    return {"cand": {"pass": 100, "fail": len(pre) + len(new) + len(reg), "skip": 3, "n": 105},
            "tip": {"pass": 100, "fail": len(pre), "skip": 3, "n": 103},
            "regressions": [[t, "boom"] for t in reg], "new_fail": [[t, "boom"] for t in new],
            "pre_existing_fail": list(pre), "tip_skip_cand_fail": [], "missing_on_cand": [],
            "rc_without_junit": list(nojunit), "verdict": "PASS"}


def test_G_SUITE_a_verdict_is_BOUND_to_the_commit_and_the_tree_it_ran_on(tmp_path):
    tree = _mini_tree(tmp_path / "cand")
    vj = tmp_path / "verdict.json"
    vj.write_text(json.dumps(_verdict()), encoding="utf-8")
    b = LG.bind_suite_verdict(vj, tree, "ab" * 20)
    tsha = LG.tree_digest(LG.tree_manifest(tree))
    assert b["schema"] == "tanitad.launch_gate.suite_bound/1" and b["cand_tree_sha256"] == tsha
    pin = {"ids": ["tests.test_a::t1", "tests.test_a::t2", "tests.test_b::t9"]}
    r, d = LG.judge_suite_verdict(b, "ab" * 20, tsha, pin)
    assert r == [] and d["n_pinned_now_passing"] == 1      # a pinned test that now PASSES is fine
    # a stale / other-commit verdict, a verdict of another tree, an unbound verdict: refused
    assert any("not the launch commit" in x
               for x in LG.judge_suite_verdict(b, "cd" * 20, tsha, pin)[0])
    assert any("not the launched tree" in x
               for x in LG.judge_suite_verdict(b, "ab" * 20, "0" * 64, pin)[0])
    assert any("not bound" in x
               for x in LG.judge_suite_verdict({"verdict": _verdict()}, "ab" * 20, tsha, pin)[0])
    assert "no Thor full-suite verdict" in LG.judge_suite_verdict(None, "ab" * 20, tsha, pin)[0][0]
    with pytest.raises(LG.GateError):
        LG.bind_suite_verdict(vj, tree, "not-a-sha")


def test_G_SUITE_a_NEW_failure_outside_the_pinned_list_FAILS(tmp_path):
    pin = {"ids": ["tests.test_a::t1", "tests.test_a::t2"]}
    cases = ((_verdict(new=("tests.test_c::t3",)), "OUTSIDE the pinned"),
             (_verdict(reg=("tests.test_d::t4",)), "OUTSIDE the pinned"),
             (_verdict(nojunit=[["CAND", "tests/test_e.py", 2, "ImportError"]]), "crashed"))
    for v, sig in cases:
        b = {"schema": "tanitad.launch_gate.suite_bound/1", "commit": "ab" * 20,
             "cand_tree_sha256": "f" * 64, "verdict": v}
        r, _ = LG.judge_suite_verdict(b, "ab" * 20, "f" * 64, pin)
        assert any(sig in x for x in r), r


def test_G_SUITE_the_pinned_env_list_is_content_bound_by_the_profile(tmp_path):
    tree = _mini_tree(tmp_path / "t")
    rel, want = LG.PROFILES["refcv7"]["suite_env_failures"]
    assert rel == "stack/ops/launch_gate_thor_env_failures.json"
    assert want == "fc1214e5978383c694af7dbfc6c26f44758fcf8a93c4386700efb792072b4ed4"
    assert "absent from the tree" in LG.env_failure_pin(LG.PROFILES["refcv7"], tree)[1]
    f = tree / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text('{"ids": [], "files": []}', encoding="utf-8")
    assert "the profile pins" in LG.env_failure_pin(LG.PROFILES["refcv7"], tree)[1]
    rec, prob = LG.env_failure_pin(LG.PROFILES["refcv7"], ROOT.parent)     # the real file
    assert prob is None and rec["n_ids"] == 165 and rec["n_files"] == 33 and rec["n_flaky"] == 1
    assert rec["flaky"][0]["id"] == ("tests.test_flagship_v4::"
                                     "test_graft_tau_to_zero_makes_the_class_posterior_one_hot")
    assert rec["flaky"][0]["record_sha256"] == (
        "91016fd69b9835605dff45c7acfcc708c6ae4cc2a27661c8fd7cc10fe4bacab7")


def test_G_SUITE_a_FLAKY_failure_is_admitted_ONLY_with_its_record_cited(tmp_path):
    fid = "tests.test_f::t_flaky"
    rec = tmp_path / "rec.json"
    rec.write_text("{}", encoding="utf-8")
    import hashlib
    sha = hashlib.sha256(b"{}").hexdigest()
    b = {"schema": "tanitad.launch_gate.suite_bound/1", "commit": "ab" * 20,
         "cand_tree_sha256": "f" * 64, "verdict": _verdict(new=(fid,))}
    env = ["tests.test_a::t1", "tests.test_a::t2"]          # _verdict's pre-existing pair
    cited = {"ids": env, "flaky": [{"id": fid, "record": "rec.json", "record_sha256": sha}]}
    r, d = LG.judge_suite_verdict(b, "ab" * 20, "f" * 64, cited, tmp_path)
    assert r == [] and d["flaky_failing"] == [fid] and d["flaky"][fid].endswith("re-read")
    uncited = {"ids": env, "flaky": [{"id": fid}]}
    r, _ = LG.judge_suite_verdict(b, "ab" * 20, "f" * 64, uncited, tmp_path)
    assert r and "OUTSIDE the pinned" in r[0]
    forged = {"ids": env, "flaky": [{"id": fid, "record": "rec.json", "record_sha256": "0" * 64}]}
    r, d = LG.judge_suite_verdict(b, "ab" * 20, "f" * 64, forged, tmp_path)
    assert r and d["flaky"][fid].startswith("NOT admitted: the cited record hashes")
    # the dev-box half admits the same flaky failure, and nothing else
    files = [{"sub": "stack", "file": "tests/test_f.py"}]
    res = {"stack/tests/test_f.py": {"rc": 1, "n_tests": 3, "failed": [fid], "skipped": {}}}
    assert LG.judge_pinned_run(files, res, (), {fid})[0] == []
    assert "1 test(s) FAIL" in LG.judge_pinned_run(files, res, (), set())[0][0]


def test_G_SUITE_PINNED_every_test_passes_or_skips_for_a_REGISTERED_reason():
    files = [{"sub": "stack", "file": "tests/test_refcv6_trunk.py"}]
    key = "stack/tests/test_refcv6_trunk.py"
    ok = {key: {"rc": 0, "n_tests": 18, "failed": [], "n_skipped": 0, "skipped": {}}}
    assert LG.judge_pinned_run(files, ok, ())[0] == []
    skip = {key: {"rc": 0, "n_tests": 18, "failed": [], "n_skipped": 1,
                  "skipped": {"t::a": "resnet101 weights not in the local HF cache"}}}
    r, d = LG.judge_pinned_run(files, skip, ())
    assert d["n_unregistered_skips"] == 1 and "not registered" in r[0]
    assert LG.judge_pinned_run(files, skip, (r"not in the local HF cache",))[0] == []
    fail = {key: {"rc": 1, "n_tests": 18, "failed": ["t::b"], "n_skipped": 0, "skipped": {}}}
    assert "1 test(s) FAIL" in LG.judge_pinned_run(files, fail, ())[0][0]
    crash = {key: {"rc": 2, "tail": "ERROR collecting"}}
    assert "produced no junit" in LG.judge_pinned_run(files, crash, ())[0][0]


def test_ram_rule_start_needs_3_consecutive_samples(monkeypatch):
    seq = iter([8.0, 7.0, 8.0, 8.1, 8.2, 9.0])
    monkeypatch.setattr(LG, "ram_available_gb", lambda: next(seq))
    monkeypatch.setattr(LG.time, "sleep", lambda s: None)
    ok, av = LG._wait_ram(7.5, 3600, samples=3, gap_s=0)
    assert ok and av == 8.2                     # 8.0 / (7.0 resets) / 8.0 8.1 8.2 -> start


# ------------------------------------------------------------------------------------------ #
# G-HYG: the static import closure (scipy is ABSENT from Thor's training venv)                   #
# ------------------------------------------------------------------------------------------ #
def _closure_tree(root: Path, body: str) -> Path:
    (root / "stack" / "scripts").mkdir(parents=True)
    (root / "stack" / "tanitad").mkdir(parents=True)
    (root / "taniteval").mkdir()
    (root / "stack" / "tanitad" / "__init__.py").write_text("", encoding="utf-8")
    (root / "stack" / "tanitad" / "m.py").write_text(body, encoding="utf-8")
    (root / "stack" / "scripts" / "train.py").write_text("import json\nfrom tanitad import m\n",
                                                          encoding="utf-8")
    return root


def test_G_HYG_import_closure_eager_missing_FAILS_lazy_and_guarded_are_reported(tmp_path):
    body = ("import numpy\nimport nosuchmod_eager\n"
            "def f():\n    import nosuchmod_lazy\n"
            "try:\n    import nosuchmod_opt\nexcept ImportError:\n    pass\n"
            "def g():\n    from scipy import stats\n")
    tree = _closure_tree(tmp_path / "t", body)
    clo = LG.import_closure(tree, "stack/scripts/train.py")
    assert clo["n_files"] == 3                         # train.py, tanitad/__init__.py, tanitad/m.py
    assert sorted(clo["third_party"]) == ["nosuchmod_eager", "nosuchmod_lazy", "nosuchmod_opt",
                                          "numpy", "scipy"]
    r, d = LG.judge_import_closure(clo, ban=("scipy",))
    assert sorted(d["missing_fail"]) == ["nosuchmod_eager", "scipy"]      # scipy: STRICT, even lazy
    assert sorted(d["missing_lazy_only"]) == ["nosuchmod_lazy"]
    assert sorted(d["missing_guarded_only"]) == ["nosuchmod_opt"]
    assert len(r) == 2 and r[0].startswith("G-HYG import closure: `nosuchmod_eager` is NOT")
    # the regression arm's mechanism: a MODULE-level dependency absent from the venv
    r, _ = LG.judge_import_closure(clo, ban=("numpy", "scipy"))
    assert any("`numpy` is NOT importable" in x for x in r)


# ------------------------------------------------------------------------------------------ #
# A9 / A10: G-LIVE-PRES, G-BOX-OVERFIT; A8: the class-weight stamp; G-EVAL's run record         #
# ------------------------------------------------------------------------------------------ #
def test_G_LIVE_PRES_deep_supervision_aux_layers_never_count_as_a_head():
    """A9 R2: the earlier layers' decodes ride under `aux`; the gate reads the LAST layer only, so
    two aux entries cannot stand in for a missing slot head."""
    torch = pytest.importorskip("torch")
    lp = LG.PROFILES["refcv7"]["live_pres"]
    lo, hi = torch.full((4, 100), -5.0), torch.full((4, 100), 5.0)
    out = {"box3d": {"presence_logit": lo, "aux": [{"presence_logit": hi}, {"presence_logit": hi}]},
           "agent_slots": {"aux": [{"presence_logit": lo}]}}
    heads, aux = LG.live_pres_fracs(LG._named_tensors(out), lp)
    assert sorted(heads) == ["out.box3d.presence_logit"]
    assert sorted(aux) == ["out.agent_slots.aux[0].presence_logit", "out.box3d.aux[0].presence_logit",
                           "out.box3d.aux[1].presence_logit"]
    r = LG.judge_live_pres(LG.PROFILES["refcv7"], heads, aux)[0]
    assert len(r) == 1 and r[0].startswith("G-LIVE-PRES: 1 presence head(s) found")


def test_G_LIVE_PRES_both_slot_heads_below_half_at_the_end_of_the_smoke():
    p = LG.PROFILES["refcv7"]
    ok = {"out.agent_slots.presence_logit": {"frac": 0.08, "n": 1600},
          "out.box3d.presence_logit": {"frac": 0.11, "n": 4800}}
    r, d = LG.judge_live_pres(p, ok)
    assert r == [] and d["g_live_count"].startswith("NOT APPLICABLE")
    red = dict(ok, **{"out.box3d.presence_logit": {"frac": 0.85, "n": 4800}})   # refcv6: 71-99 %
    assert LG.judge_live_pres(p, red)[0] == [
        "G-LIVE-PRES: out.box3d.presence_logit: 0.850 of 4800 slots have sigma >= 0.5 at the end "
        "of the smoke (must be < 0.5; refcv6: 71-99 of 100)"]
    one = {"out.agent_slots.presence_logit": {"frac": 0.08, "n": 1600}}
    assert "1 presence head(s) found" in LG.judge_live_pres(p, one)[0][0]
    assert LG.judge_live_pres(LG.PROFILES["refcv6"], red) == ([], {})      # generic: not checked


_GBO_MAIN = {"ap2m": 0.95, "prec": 0.93, "rec": 0.94, "n_conf": 117, "centre_p50_m": 0.21,
             "size_p50_m": 0.18, "z_p50_m": 0.07, "cls_acc": 0.97, "presence": 0.12}


#: SPEC_REFCV7 A13: the LAUNCH optimiser as the harness records it (the built -- PEAK -- groups)
_GBO_OPT = {"class": "AdamW", "clip": 10.0,
            "groups": [{"name": None, "lr": 5e-05, "weight_decay": 1e-04, "betas": [0.9, 0.999], "eps": 1e-08,
                        "n_tensors": 316},
                       {"name": None, "lr": 1e-04, "weight_decay": 1e-04, "betas": [0.9, 0.999], "eps": 1e-08,
                        "n_tensors": 491}]}
#: SPEC_REFCV7 A17: the schedule literal the harness writes
_GBO_SCHED = {"rule": "A17", "hold_peak_through_step": 1799, "decay": "cosine", "decay_from_step": 1800,
              "decay_to_step": 2000, "final_factor": 0.0}


def _gbo_arm(final, *, finite=True, last_step=2000, verdict="PASS"):
    """One arm in the box harness's schema (`g_box_overfit.run_arm`): log rows from step 0, the
    criteria (only criterion 6's finiteness flag is READ by the gate), the arm's own verdict, its
    optimiser as built (A13)."""
    r0 = {"step": 0, "ap2m": 0.01, "prec": 0.0, "rec": 0.0, "n_conf": 0, "centre_p50_m": 5.0,
          "size_p50_m": 2.0, "z_p50_m": 1.0, "cls_acc": 0.1, "presence": 1.0}
    return {"rows": [r0, {"step": last_step, **final}], "verdict": verdict,
            "criteria": {"6": {"value": [finite, 1.0, final["presence"], final["presence"]]}},
            "optimizer": json.loads(json.dumps(_GBO_OPT)), "lr_schedule": dict(_GBO_SCHED)}


def _gbo_record(argv_sha, **over):
    """The box-head harness's record (`g_box_overfit.py` main(), as it stands in the box-head
    package -- PROVISIONAL until it lands)."""
    rec = {"tool": "g_box_overfit.py", "binding": True, "commit": "ab" * 20, "RESULT": "PASS",
           "prereg_md5": "594c71196cc5bbd527fee40b2cb0e3f1",
           "frameset_md5": "b291404c36f83c3e397e3b90367e8e7b", "steps": 2000,
           "literals": {"batch": 4, "seed": 0, "n_pos": 113, "n_ign": 77, "clip": 10.0,
                        "lr_schedule": dict(_GBO_SCHED)},
           "launch_argv_sha256": argv_sha, "dvb_mismatches": [],
           "C1_C2": {"C1": {"pass": True}, "C2": {"pass": True}}, "C3": {"pass": True},
           "C4_step0": {"step": 0},
           "arms": {"main": _gbo_arm(_GBO_MAIN),
                    "memory_zeros": _gbo_arm(dict(_GBO_MAIN, ap2m=0.12),
                                             verdict="failed as required"),
                    "presence_w0": _gbo_arm(dict(_GBO_MAIN, prec=0.31, rec=0.52, n_conf=0),
                                            verdict="failed as required")}}
    rec.update(over)
    return rec


def test_G_BOX_OVERFIT_the_registered_literals_and_the_must_fail_arms(tmp_path):
    """PREREG_G_BOX_OVERFIT sec. 5-6 re-judged from the FINAL row values against the gate's own
    literals: the record's pass flags never decide."""
    p = LG.PROFILES["refcv7"]
    commit, argv = "ab" * 20, ["--arm", "hier", "--out", "/o"]
    argv_sha = LG.argv_sha256(argv)
    f = tmp_path / "g_box_overfit.json"

    def judge(rec):
        f.write_text(json.dumps(rec), encoding="utf-8")
        return LG.judge_box_overfit(p, str(f), commit, argv_sha=argv_sha, argv=argv)[0]
    assert judge(_gbo_record(argv_sha)) == []
    assert "no box overfit PASS record" in LG.judge_box_overfit(p, None, commit)[0][0]
    low = _gbo_record(argv_sha)
    low["arms"]["main"]["rows"][-1]["z_p50_m"] = 0.2
    assert judge(low) == ["G-BOX-OVERFIT: main fails prereg criterion 4 ['centre_p50_m', "
                          "'size_p50_m', 'z_p50_m'] at step 2000: {'centre_p50_m': 0.21, "
                          "'size_p50_m': 0.18, 'z_p50_m': 0.2}"]
    claims = _gbo_record(argv_sha)                  # PASS flags everywhere, AP below the bar
    claims["arms"]["main"]["rows"][-1]["ap2m"] = 0.85
    claims["arms"]["main"]["criteria"]["1"] = {"pass": True}
    assert judge(claims) == ["G-BOX-OVERFIT: main fails prereg criterion 1 ['ap2m'] at step "
                             "2000: {'ap2m': 0.85}"]
    void = _gbo_record(argv_sha)
    void["arms"]["memory_zeros"]["rows"][-1]["ap2m"] = 0.93
    assert judge(void) == ["G-BOX-OVERFIT: must-fail arm 'memory_zeros' must FAIL prereg "
                           "criteria ['1'] and passes ['1'] -- the result is VOID (prereg sec. 7)"]
    half = _gbo_record(argv_sha)                    # presence_w0 fails 2 but hits the count
    half["arms"]["presence_w0"]["rows"][-1]["n_conf"] = 110
    assert judge(half) == ["G-BOX-OVERFIT: must-fail arm 'presence_w0' must FAIL prereg "
                           "criteria ['2', '3'] and passes ['3'] -- the result is VOID (prereg "
                           "sec. 7)"]
    c3 = _gbo_record(argv_sha, literals={"batch": 4, "seed": 0, "n_pos": 113, "n_ign": 28,
                                         "lr_schedule": dict(_GBO_SCHED)})  # the prereg's pre-A10 count
    assert judge(c3) == ["G-BOX-OVERFIT: literal n_ign = 28, the prereg (with A10's "
                         "reconciliation) fixes 77"]
    nan = _gbo_record(argv_sha)
    nan["arms"]["main"]["criteria"]["6"]["value"][0] = False
    assert judge(nan) == ["G-BOX-OVERFIT: main fails prereg criterion 6 ['presence_ratio'] at "
                          "step 2000: {'presence_ratio': 0.12} (loss not finite)"]
    short = _gbo_record(argv_sha)
    short["arms"]["main"]["rows"][-1]["step"] = 1500
    assert judge(short) == ["G-BOX-OVERFIT: arm 'main' ends at step 1500, not 2000"]
    ctl = _gbo_record(argv_sha, C1_C2={"C1": {"pass": True}, "C2": {"pass": False}})
    assert judge(ctl) == ["G-BOX-OVERFIT: control C2 does not read its known value "
                          "({'pass': False})"]
    no4 = _gbo_record(argv_sha)
    del no4["C4_step0"]
    assert judge(no4) == ["G-BOX-OVERFIT: C4 (step 0, the launch init) is not reported"]
    # SPEC_REFCV7 A11: the commit is the CLOSURE's job -- another commit is not a refusal here
    assert judge(_gbo_record(argv_sha, commit="12" * 20)) == []
    # the argv: the gate's definition, or the harness's own json.dumps form of the SAME list
    hform = hashlib.sha256(json.dumps(argv).encode()).hexdigest()
    assert judge(_gbo_record(hform)) == []
    assert judge(_gbo_record("00" * 32)) == ["G-BOX-OVERFIT: the record is bound to argv "
                                             "000000000000, not the launch argv "
                                             f"{argv_sha[:12]}"]
    # SPEC_REFCV7 A13 + A17 -- RED arms: the EARLY runs' optimiser (every tensor at 2e-4, no weight decay, no
    # clip); the A13 CONSTANT schedule; an arm that records no optimiser
    early = _gbo_record(argv_sha)
    early["arms"]["main"]["optimizer"] = {"class": "AdamW", "clip": None,
                                          "groups": [{"lr": 2e-4, "weight_decay": 0.0}]}
    r = judge(early)
    assert len(r) == 1 and r[0].startswith("G-BOX-OVERFIT: arm 'main' trained with 'AdamW' groups [0.0002]"), r
    const = _gbo_record(argv_sha)
    const["literals"]["lr_schedule"] = {"rule": "A13", "decay": "none"}
    r = judge(const)
    assert len(r) == 1 and r[0].startswith("G-BOX-OVERFIT: literal lr_schedule = {'rule': 'A13'"), r
    noopt = _gbo_record(argv_sha)
    del noopt["arms"]["presence_w0"]["optimizer"]
    assert judge(noopt) == ["G-BOX-OVERFIT: arm 'presence_w0' records no optimizer spec (SPEC_REFCV7 A13)"]


def test_A8_the_class_weight_stamp_in_config_json():
    p = LG.PROFILES["refcv7"]
    good = {"map_hires": {"class_weights": {"definition_id": "sqrt_mf", "pre_registered": True,
                                            "sha256": "ab" * 32}}}
    assert LG.class_weights_stamp_reasons(p, good, "ab" * 32)[0] == []
    mf = {"map_hires": {"class_weights": {"definition_id": "mf", "pre_registered": False,
                                          "sha256": "cd" * 32}}}
    assert len(LG.class_weights_stamp_reasons(p, mf, "ab" * 32)[0]) == 3
    assert "carries no" in LG.class_weights_stamp_reasons(p, {}, "ab" * 32)[0][0]


def test_G_EVAL_rebuilds_from_THIS_launchs_run_record(tmp_path):
    ctx = _ctx(tmp_path)
    assert LG.latest_smoke_config(ctx) is None
    d = Path(ctx.out_dir) / "smoke" / "20260927T000000"
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"argv": ctx.argv}), encoding="utf-8")
    assert LG.latest_smoke_config(ctx) == d / "config.json"
    # the view ignores what does not enter the model -- and sees everything that does
    a = ["--arm", "hier", "--steps", "10", "--out", "/x", "--equalize-bottom-rows", "43"]
    assert LG._argv_model_view(a) == LG._argv_model_view(LG.set_flag(a, "--steps", ["100"]))
    assert LG._argv_model_view(a) != LG._argv_model_view(
        LG.set_flag(a, "--equalize-bottom-rows", ["0"]))


def test_the_suite_bind_CLI_writes_a_bound_verdict(tmp_path):
    tree = _mini_tree(tmp_path / "cand")
    vj = tmp_path / "verdict.json"
    vj.write_text(json.dumps(_verdict()), encoding="utf-8")
    out = tmp_path / "bound.json"
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    p = subprocess.run([sys.executable, str(GATE_PY), "suite-bind", "--verdict", str(vj),
                        "--cand-tree", str(tree), "--commit", "ab" * 20, "--out", str(out)],
                       env=env, capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert p.returncode == 0 and "ZZSUITE-BOUND-abababababab" in p.stdout, p.stderr[-400:]
    b = json.loads(out.read_text(encoding="utf-8"))
    assert b["commit"] == "ab" * 20 and b["verdict"]["verdict"] == "PASS"


def test_the_grad_unreachable_probe_gets_a_callable_that_RUNS_the_backward(monkeypatch):
    """Batch 3's contract (declared_vs_built.probe_grad_unreachable(model, backward)): the callable
    runs forward + loss + .backward(). A callable that only RETURNS the loss leaves every .grad
    None -- the probe's positive control would read "no backward ran"."""
    import torch
    model = torch.nn.Linear(3, 1)
    T = types.SimpleNamespace(compute_losses_v3=lambda m, b, dev: {"loss": m(b["x"]).sum()})
    seen = {}

    def probe(m, backward):
        backward()
        seen["grad"] = m.weight.grad is not None
        return []
    D = types.SimpleNamespace(probe_grad_unreachable=probe)
    real = LG.importlib.import_module
    monkeypatch.setattr(LG.importlib, "import_module",
                        lambda n: D if n == "tanitad.train.declared_vs_built" else real(n))
    ctx = types.SimpleNamespace()
    out = LG._probe_grad_unreachable(ctx, T, model, {"x": torch.ones(2, 3)})
    assert out == {"present": True, "result": []} and seen["grad"] is True
    D2 = types.SimpleNamespace(probe_grad_unreachable=lambda m, b: ["dead: core.decoder.x"])
    monkeypatch.setattr(LG.importlib, "import_module", lambda n: D2)
    out = LG._probe_grad_unreachable(ctx, T, model, {"x": torch.ones(2, 3)})
    assert LG.judge_grad_unreachable_probe(out)[0] == [
        "G-LIVE: probe_grad_unreachable reports 1 problem(s): ['dead: core.decoder.x']"]


# ------------------------------------------------------------------------------------------ #
# SPEC_REFCV7 A11: the overfit PASS records bind to a CODE CLOSURE, recorded by closure_run.py  #
# ------------------------------------------------------------------------------------------ #
CLOSURE_PY = ROOT / "scripts" / "closure_run.py"
import closure_run as CR  # noqa: E402

_HARNESS = '''import json
import sys
from pathlib import Path

from tanitad import mini_head
try:
    from tanitad import opt_guarded          # ABSENT at the run: a guarded module-level import
except ImportError:
    opt_guarded = None


def lazy():
    try:
        from tanitad import opt_lazy         # ABSENT at the run: a guarded LAZY import
    except ImportError:
        return None
    return opt_lazy


def main():
    a = sys.argv[1:]
    out = Path(a[a.index("--out") + 1])
    payload = Path(a[a.index("--frames") + 1]).read_bytes()
    lazy()
    if "--crash" in a:
        raise RuntimeError("boom mid-run")
    from tanitad import lazy_dep             # a lazy import that RUNS: it is in sys.modules
    src = Path(__file__).with_name("exec_me.py")
    with open(src, encoding="utf-8") as fh:  # tree code run through exec: never in sys.modules
        exec(compile(fh.read(), str(src), "exec"), {})
    out.write_text(json.dumps({"G_X": "PASS", "v": mini_head.VALUE + lazy_dep.LAZY,
                               "n": len(payload)}), encoding="utf-8")
    if "--exit3" in a:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
'''


def _harness_tree(root: Path) -> Path:
    (root / "stack" / "tanitad").mkdir(parents=True)
    (root / "stack" / "scripts").mkdir(parents=True)
    (root / "taniteval").mkdir()
    (root / "data").mkdir()
    (root / "stack" / "tanitad" / "__init__.py").write_bytes(b"")
    (root / "stack" / "tanitad" / "mini_head.py").write_bytes(b"VALUE = 1\n")
    (root / "stack" / "tanitad" / "lazy_dep.py").write_bytes(b"LAZY = 7\n")
    (root / "stack" / "scripts" / "harness_x.py").write_bytes(_HARNESS.encode())
    (root / "stack" / "scripts" / "exec_me.py").write_bytes(b"EXECUTED = True\n")
    (root / "data" / "frames.bin").write_bytes(b"frames-16")
    (root / "data" / "unread.bin").write_bytes(b"never opened")
    return root


def _closure_run(root: Path, name: str, *extra_harness_args: str, binding: bool = True):
    clo, res = root.parent / f"{name}.closure.json", root.parent / f"{name}.pass.json"
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(PYTHONPATH=str(root / "stack"), PYTHONDONTWRITEBYTECODE="1",
               PYTHONIOENCODING="utf-8")
    env.pop(LG.ARM_ENV, None)
    cmd = [sys.executable, str(CLOSURE_PY), "--out", str(clo), "--result", str(res),
           "--data-root", str(root / "data"), "--tree", str(root),
           *(["--binding"] if binding else []), "--",
           str(root / "stack" / "scripts" / "harness_x.py"), "--out", str(res),
           "--frames", str(root / "data" / "frames.bin"), *extra_harness_args]
    p = subprocess.run(cmd, cwd=str(root), env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=300)
    return p, clo, res


def test_closure_digest_is_ONE_definition_in_the_wrapper_and_the_gate():
    args = ([["stack/tanitad/a.py", "cd" * 20]], [], ["tanitad.x"], [["/d/f", "ab" * 32]])
    lit = "d2ae69721523b04d75741bd9aacad12afade85eb1d5713dae9b3c859ca7d7a26"
    assert CR.closure_digest(*args) == lit
    assert LG._closure_digest(*args) == lit


def test_closure_run_records_the_imported_tree_modules_FROM_THE_PROCESS(tmp_path):
    """SPEC_REFCV7 A11: every module in sys.modules inside the run's tree -- a lazy import that
    ran included -- read from the process at exit, never from a hand list; every file OPENED under
    a data root; the names looked for and not found; the PASS JSON by sha256."""
    root = _harness_tree(tmp_path / "t")
    p, clo, res = _closure_run(root, "ok")
    assert p.returncode == 0, p.stderr[-2000:]
    c = json.loads(clo.read_text(encoding="utf-8"))
    assert (c["schema"], c["binding"], c["exit_status"], c["script"]) == (
        "tanitad.launch_gate.closure/1", True, 0, "stack/scripts/harness_x.py")
    mods = dict(c["modules"])
    assert sorted(mods) == ["stack/scripts/exec_me.py", "stack/scripts/harness_x.py",
                            "stack/tanitad/__init__.py", "stack/tanitad/lazy_dep.py",
                            "stack/tanitad/mini_head.py"]
    assert mods["stack/tanitad/mini_head.py"] == "b15b1b0797d64623135bb2d04a67bae078fa44c2"
    assert mods["stack/scripts/exec_me.py"] == "2c1316b28329c53141fa8779969873013e175c2d"
    assert {"tanitad.opt_guarded", "tanitad.opt_lazy"} <= set(c["absent"])
    assert c["data"] == [[str((root / "data" / "frames.bin").resolve()),
                          "fb43a9e836367a7c374f0d8852055235b973c398102e91ed0939b8756feda08f"]]
    assert c["result"]["sha256"] == LG.sha256_file(res)
    assert (c["outside_tree"], c["children"], c["probed"]) == ([], [], [])
    assert c["closure_sha256"] == LG._closure_digest(c["modules"], c["probed"], c["absent"],
                                                     c["data"])


def test_closure_run_reads_the_modules_from_SYS_MODULES_not_from_a_list(tmp_path, monkeypatch):
    """The sys.modules half on its own (no audit hook in this process): a module loaded BY PATH
    under a made-up name -- the box harness's `_load_by_path` -- is recorded by its FILE."""
    import importlib.util
    root = _harness_tree(tmp_path / "t")
    f = root / "stack" / "tanitad" / "probe_mod.py"
    f.write_bytes(b"P = 1\n")
    spec = importlib.util.spec_from_file_location("zz_closure_probe_mod", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setitem(sys.modules, "zz_closure_probe_mod", mod)
    a = types.SimpleNamespace(tree=str(root), out=str(tmp_path / "c.json"), data=[],
                              data_root=[], result=None, binding=True, commit=None)
    mods = CR.Recorder(a, root / "stack" / "scripts" / "harness_x.py", ["h"])._modules()[0]
    assert ["stack/tanitad/probe_mod.py", "21d3457a54686309521105eec8314b82526c8c7b"] in mods


def test_closure_run_a_CRASHED_harness_still_writes_its_record_with_the_exit_status(tmp_path):
    root = _harness_tree(tmp_path / "t")
    p, clo, _ = _closure_run(root, "crash", "--crash")
    assert p.returncode != 0 and clo.is_file()
    c = json.loads(clo.read_text(encoding="utf-8"))
    assert (c["exit_status"], c["error"]) == (1, "RuntimeError: boom mid-run")
    assert c["result"]["sha256"] is None                     # the harness never wrote its JSON
    p3, clo3, _ = _closure_run(root, "exit3", "--exit3")
    assert p3.returncode == 3
    assert json.loads(clo3.read_text(encoding="utf-8"))["exit_status"] == 3


def test_closure_binds_a_PASS_and_REFUSES_each_red_arm_of_SPEC_A11(tmp_path):
    """The four registered red arms -- one blob differs, a non-binding early record, an imported
    but unrecorded module, a crashed harness -- and the holes around them: a module the run looked
    for and did not find that the launch tree now provides, an edited record, another run's PASS
    JSON, a changed data file, a shadowing import."""
    root = _harness_tree(tmp_path / "t")
    h = "stack/scripts/harness_x.py"
    _, clo, res = _closure_run(root, "ok")

    def judge(c=None, r=None, **kw):
        return LG.judge_closure("G-X", str(c or clo), str(r or res), root, h, **kw)[0]
    assert judge() == []
    mh = root / "stack" / "tanitad" / "mini_head.py"
    mh.write_bytes(b"VALUE = 2\n")                                        # 1. one blob differs
    assert judge() == ["G-X: the launch commit differs from the harness run's code: 1 changed "
                       "blob(s) ['stack/tanitad/mini_head.py'], 0 absent at the launch [] -- "
                       "the PASS does not describe this code"]
    mh.write_bytes(b"VALUE = 1\n")
    _, clo_e, res_e = _closure_run(root, "early", binding=False)          # 2. an early record
    assert judge(clo_e, res_e) == ["G-X: the closure is NOT binding (an early run) -- only a run "
                                   "made with closure_run.py --binding can be consumed"]
    c = json.loads(clo.read_text(encoding="utf-8"))                      # 3. imported, unrecorded
    c["modules"] = [m for m in c["modules"] if m[0] != "stack/tanitad/mini_head.py"]
    c["closure_sha256"] = LG._closure_digest(c["modules"], c["probed"], c["absent"], c["data"])
    hand = tmp_path / "hand.closure.json"
    hand.write_text(json.dumps(c), encoding="utf-8")
    assert judge(hand) == ["G-X: the record lacks 1 file(s) the harness imports at module level "
                           "on the launch tree (['stack/tanitad/mini_head.py']) -- an imported "
                           "but unrecorded module"]
    _, clo_c, res_c = _closure_run(root, "crash", "--crash")              # 4. a crashed harness
    r = judge(clo_c, res_c)
    assert r[0] == ("G-X: the wrapped harness exited 1 (RuntimeError: boom mid-run) -- only a "
                    "completed run binds") and len(r) == 2              # + its PASS JSON is absent
    lazy = root / "stack" / "tanitad" / "opt_lazy.py"                     # a guarded LAZY import
    lazy.write_bytes(b"X = 1\n")                                          # now finds a module
    assert judge() == ["G-X: the launch tree provides 1 module(s) the run looked for and did NOT "
                       "find (['tanitad.opt_lazy']) -- the launch code takes an import path the "
                       "PASS never ran"]
    lazy.unlink()
    (root / "stack" / "tanitad" / "opt_guarded.py").write_bytes(b"Y = 1\n")   # module level: both
    r = judge()                                                                # rules see it
    assert len(r) == 2 and "['tanitad.opt_guarded']" in r[0] and "opt_guarded.py" in r[1]
    (root / "stack" / "tanitad" / "opt_guarded.py").unlink()
    ed = json.loads(clo.read_text(encoding="utf-8"))                      # an edited record
    ed["modules"][0][1] = "00" * 20
    edf = tmp_path / "edited.closure.json"
    edf.write_text(json.dumps(ed), encoding="utf-8")
    assert "do not re-hash to closure_sha256" in judge(edf)[0]
    other = tmp_path / "other.pass.json"                                  # another run's JSON
    other.write_text('{"G_X": "PASS"}', encoding="utf-8")
    assert judge(r=other)[0].startswith("G-X: the closure names a PASS JSON ")
    (root / "data" / "frames.bin").write_bytes(b"frames-17")              # a changed data file
    assert "1 data file(s) the run read are absent or changed" in judge()[0]
    (root / "data" / "frames.bin").write_bytes(b"frames-16")
    sh = json.loads(clo.read_text(encoding="utf-8"))                      # a shadowing import
    sh["outside_tree"] = [{"module": "tanitad", "file": "/elsewhere/tanitad/__init__.py"}]
    shf = tmp_path / "shadow.closure.json"
    shf.write_text(json.dumps(sh), encoding="utf-8")
    assert "imported from OUTSIDE the run's tree" in judge(shf)[0]
    assert "no closure record" in LG.judge_closure("G-X", None, str(res), root, h)[0][0]
    assert judge() == []                                                  # restored: PASS again


def test_closure_a_torch_timm_CUDA_cuDNN_mismatch_FAILS(tmp_path):
    """Master Mind 2026-09-27: an environment mismatch FAILS, it does not warn."""
    root = _harness_tree(tmp_path / "t")
    _, clo, res = _closure_run(root, "ok")
    c = json.loads(clo.read_text(encoding="utf-8"))
    c["env"].update(torch="2.8.0", timm="1.0.19", cuda="12.8", cudnn=91002)
    clo.write_text(json.dumps(c), encoding="utf-8")
    host = {"torch": "2.8.0", "timm": "1.0.19", "cuda": "12.8", "cudnn": 91002}
    h = "stack/scripts/harness_x.py"
    assert LG.judge_closure("G-X", str(clo), str(res), root, h, host_env=host)[0] == []
    r = LG.judge_closure("G-X", str(clo), str(res), root, h, host_env=dict(host, cudnn=90100))[0]
    assert r == ["G-X: the run's environment differs from this host's {'cudnn': (91002, 90100)} "
                 "-- a changed venv means the binding runs are re-done"]


def test_G_MAP_OVERFIT_job_needs_its_closure_and_its_PASS_states_the_isolation_scope(
        tmp_path, monkeypatch):
    """The job consumes the (PASS JSON, closure) PAIR; its PASS text says what the overfit
    certifies -- the head in isolation -- and whose job the wiring is (Master Mind 2026-09-27)."""
    root = _harness_tree(tmp_path / "t")
    _, clo, res = _closure_run(root, "ok")
    c = json.loads(clo.read_text(encoding="utf-8"))
    monkeypatch.setattr(LG, "_host_env", lambda: {k: c["env"].get(k) for k in
                                                  ("torch", "timm", "cuda", "cudnn")})
    p = LG.PROFILES["refcv7"]
    monkeypatch.setitem(LG.PROFILES, "refcv7", dict(
        p, map_overfit=dict(p["map_overfit"], harness="stack/scripts/harness_x.py")))
    monkeypatch.setattr(LG, "judge_map_overfit", lambda *a, **k: ([], {}))   # its own tests
    ctx, _ = _refcv7_argv_ctx(tmp_path / "c")
    ctx = dataclasses.replace(ctx, tree=str(root), options=dict(
        ctx.options, map_overfit_record=str(res), map_overfit_closure=str(clo)))
    ev = LG._job_map_overfit(ctx)["G-MAP-OVERFIT"]
    assert ev["status"] == "PASS", ev["reasons"]
    assert ev["reasons"] == ["G-MAP-OVERFIT PASS covers the head IN ISOLATION (trunk + branch on "
                             "the frozen frames); the launch model's WIRING is G-LIVE's and "
                             "G-DVB's job at the launch commit"]
    assert set(ev["inputs_read"]) == {"gate:map-overfit-record", "gate:map-overfit-closure"}
    no = dataclasses.replace(ctx, options=dict(ctx.options, map_overfit_closure=None))
    ev = LG._job_map_overfit(no)["G-MAP-OVERFIT"]
    assert ev["status"] == "FAIL" and "no closure record" in ev["reasons"][0]
    monkeypatch.setattr(LG, "_host_env", lambda: {"torch": "0.0-other", "timm": None,
                                                  "cuda": None, "cudnn": None})
    ev = LG._job_map_overfit(ctx)["G-MAP-OVERFIT"]
    assert ev["status"] == "FAIL" and "environment differs" in ev["reasons"][0]


# ------------------------------------------------------------------------------------------ #
# the profile's OPEN ITEMS: an undecided flag set can never mint a PASS                        #
# ------------------------------------------------------------------------------------------ #
def test_the_profile_OPEN_ITEMS_block_a_PASS_until_each_is_closed(tmp_path, monkeypatch):
    """Master Mind 2026-09-27: an undecided part of the flag set is an OPEN ITEM; while one is open
    the verdict is INCOMPLETE with the item NAMED. Both refcv7 items are CLOSED (BOX-HEAD: the box
    head landed, 28d8365; MAP-LIFT: SPEC_REFCV7 23, A18 -- its flags are required values, pinned in
    the MAP-LIFT tests below), so the mechanism is held on a SYNTHETIC item."""
    p = LG.PROFILES["refcv7"]
    assert p["open_items"] == ()                     # the refcv7 launch flag set is DECIDED
    ctx, _ = _refcv7_argv_ctx(tmp_path)
    key = LG.load_key(tmp_path / "k.key")
    assert LG.finalize(ctx, key)[0] == "PASS"
    it = {"id": "SYNTH", "what": "an undecided lever", "owner": "the test"}
    monkeypatch.setitem(LG.PROFILES, "refcv7", dict(p, open_items=(it,)))
    verdict, path, tok = LG.finalize(ctx, key)
    assert verdict == "INCOMPLETE" and not LG.verify_token(path, ctx.argv, ctx.tree, key=key)[0]
    assert tok["reasons"] == ["OPEN ITEM SYNTH: an undecided lever (owner: the test)"]


def test_the_BOX_HEAD_item_is_closed_by_its_flags_as_REQUIRED_values():
    """The box head's flags, as LITERALS: a refcv7 argv carrying any other value -- or leaving one out -- is
    REFUSED (the closed BOX-HEAD item)."""
    p = LG.PROFILES["refcv7"]
    rv = p["required_values"]
    assert {k: rv[k] for k in ("--slot-presence-loss", "--slot-presence-prior", "--slot-query-select",
                               "--vis1-sidecar")} == {
        "--slot-presence-loss": "focal", "--slot-presence-prior": "0.01", "--slot-query-select": "learned_ref",
        "--vis1-sidecar": "/home/nvidia/data/refcv7/vis1_sidecar_refcv6b1_train4369_eval139.npz"}
    assert {"--slot-deep-supervision", "--slot-vis1"} <= set(p["required_flags"])
    assert p["box_required"] == {"module": "tanitad.train.box_head_guard", "fn": "check_refcv7_box_required"}
    good = ["--arm", "hier", *PRIOR, *MAP10, *BOXA9, *REQ3]
    assert LG.required_on_reasons(p, good)[0] == [] and LG.required_flag_reasons(p, good) == []
    for flag, bad in (("--slot-presence-loss", ["bce"]), ("--slot-presence-prior", ["0.05"]),
                      ("--slot-query-select", ["heatmap"]), ("--slot-query-select", ["learned"]),
                      ("--vis1-sidecar", ["/elsewhere/sidecar.npz"])):
        r, _ = LG.required_on_reasons(p, LG.set_flag(good, flag, bad))
        assert len(r) == 1 and r[0].startswith(flag), r
        r, _ = LG.required_on_reasons(p, LG.set_flag(good, flag, None))
        assert len(r) == 1 and r[0].startswith(flag), r
    for flag in ("--slot-deep-supervision", "--slot-vis1"):
        r = LG.required_flag_reasons(p, LG.set_flag(good, flag, None))
        assert len(r) == 1 and flag in r[0] and "A9" in r[0], r


def test_G_DVB_calls_the_box_head_guard_and_names_its_mismatches(monkeypatch):
    """The BUILT half of the closed item: G-DVB runs `box_head_guard.check_refcv7_box_required(model, args)`."""
    import sys as _sys
    import types as _types
    p = LG.PROFILES["refcv7"]
    fake = _types.ModuleType("tanitad.train.box_head_guard")
    calls = []

    def ok(model, args):
        calls.append((model, args))
        return []
    fake.check_refcv7_box_required = ok
    monkeypatch.setitem(_sys.modules, "tanitad.train.box_head_guard", fake)
    m, a = object(), object()
    assert LG.box_required_reasons(p, m, a)[0] == [] and calls == [(m, a)]
    # RED arms: a mismatch is named; no model; the module without the function; a profile without the field
    fake.check_refcv7_box_required = lambda model, args: ["--slot-query-select: declared learned_ref, built learned"]
    r = LG.box_required_reasons(p, m, a)[0]
    assert r == ["BOX-HEAD: 1 requirement(s) not met (SPEC_REFCV7 A9/A14.1): --slot-query-select: declared "
                 "learned_ref, built learned"]
    assert "no built model" in LG.box_required_reasons(p, None, a)[0][0]
    del fake.check_refcv7_box_required
    assert "has no `check_refcv7_box_required`" in LG.box_required_reasons(p, m, a)[0][0]
    assert LG.box_required_reasons(LG.PROFILES["refcv6"], m, a) == ([], {"box_required": None})


# ------------------------------------------------------------------------------------------ #
# G-SUITE-PINNED: a THROWAWAY repo inside the archive, and no git redirection in the tests     #
# ------------------------------------------------------------------------------------------ #
def _git(*a, cwd=None, env=None):
    return subprocess.run(["git", *a], cwd=cwd, env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", check=True).stdout.strip()


def test_G_SUITE_PINNED_git_reads_go_to_a_THROWAWAY_repo_never_the_shared_one(tmp_path,
                                                                              monkeypatch):
    """Master Mind 2026-09-27: tanitad-push's SHARED INDEX carries real staged deletions and some
    pinned tests run `git init` / `git add` / `git diff --cached`. The archive gets its own repo
    (objects borrowed read-only through `alternates`), and a GIT_DIR in the caller's environment
    must reach neither that repo's setup nor the tests."""
    if shutil.which("git") is None:
        pytest.skip("git is not on PATH")
    clean = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    src = tmp_path / "src"
    src.mkdir()
    _git("init", "-q", cwd=src, env=clean)
    (src / "stack").mkdir()
    (src / "stack" / "a.py").write_bytes(b"A = 1\n")
    _git("add", "stack/a.py", cwd=src, env=clean)
    _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "c", cwd=src,
         env=clean)
    sha = _git("rev-parse", "HEAD", cwd=src, env=clean)
    decoy = tmp_path / "decoy"                     # stands in for tanitad-push: must not move
    decoy.mkdir()
    _git("init", "-q", cwd=decoy, env=clean)
    (decoy / "d.txt").write_bytes(b"d\n")
    _git("add", "d.txt", cwd=decoy, env=clean)
    idx0 = (decoy / ".git" / "index").read_bytes()
    dest = tmp_path / "archive"
    (dest / "stack").mkdir(parents=True)
    (dest / "stack" / "a.py").write_bytes(b"A = 1\n")
    monkeypatch.setenv("GIT_DIR", str(decoy / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(decoy))
    got = LG.throwaway_repo(dest, str(src / ".git"), sha)
    assert got["head"] == sha
    assert (decoy / ".git" / "index").read_bytes() == idx0
    assert _git("status", "--porcelain", cwd=dest, env=clean) == ""       # archive == HEAD
    assert (dest / ".git" / "objects" / "info" / "alternates").read_text(
        encoding="utf-8").strip() == (src / ".git").resolve().joinpath("objects").as_posix()
    env = LG.pinned_env(dest, {"omp": 4, "cpu_only": True},
                        {"GIT_DIR": "x", "GIT_WORK_TREE": "y", "GIT_INDEX_FILE": "z",
                         "MKL_NUM_THREADS": "4", "PATH": "p", LG.ARM_ENV: "arm"})
    assert sorted(env) == ["CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "PATH", "PYTHONDONTWRITEBYTECODE",
                           "PYTHONIOENCODING", "PYTHONPATH", "SECRET_SCAN_FULL"]


def test_an_ABORT_is_VERIFIED_the_child_tree_is_gone_or_the_evidence_says_it_is_not(
        monkeypatch):
    """MEASURED 2026-09-27: the RAM-floor abort reported "ABORTED" while its child ran on for 8
    minutes and overwrote the ERROR (a taskkill that could not start under memory pressure). With
    the tree kill made a no-op, `_stop_child` must still stop the child through its fallback and
    say so step by step -- `stopped` is what it OBSERVED, never what it asked for. (No psutil:
    Thor's training venv has none, and a skipped test certifies nothing.)"""
    import time
    kw = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
          else {"start_new_session": True})
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"],
                         stdin=subprocess.DEVNULL, **kw)
    try:
        time.sleep(2)
        assert p.poll() is None, "the child exited on its own -- nothing was tested"
        monkeypatch.setattr(LG, "_kill_tree", lambda pid: None)     # cannot start, like 03:53Z
        stop = LG._stop_child(p, wait_s=5)
        assert stop["stopped"] is True and p.poll() is not None, stop
        assert stop["steps"][0] == "tree: STILL RUNNING after 5 s", stop
        assert stop["steps"][1].startswith("direct: exited"), stop
    finally:
        if p.poll() is None:
            p.kill()


def test_G_SUITE_PINNED_the_refcv7_registry_admits_exactly_the_five_measured_skips():
    """The refcv7 registry, against the five skips MEASURED on the 3cca805 rehearsal (literal ids
    and reasons, copied from the evidence): all admitted; the same reasons on OTHER tests are not."""
    p = LG.PROFILES["refcv7"]
    skipped = {
        "tests.test_anchor_prefilter::test_prefilter_is_BIT_EXACT_on_every_candidate_it_decodes":
            "Windows-CPU BLAS picks shape-dependent gemm tilings, so the batched and the subset "
            "decode differ by ULPs (measured max 1.9e-06 on the dev box, 2026-08-15).",
        "tests.test_anchor_prefilter::test_replicates_the_measured_survivor_counts_on_the_canonical_val":
            "anchor fixture absent: refc_anchors_small64.pt",
        "tests.test_kingate_contract::test_gate_k_1_must_be_reachable_so_the_identity_control_can_actually_run":
            "the T1 gate hook is not written yet",
        "tests.test_kingate_contract::test_the_zero_flag_path_must_be_declared_bit_identical":
            "the T1 gate hook is not written yet",
        "tests.test_refa_v1_precision::test_cuda_bf16_autocast_tiny_step_has_finite_loss_and_grad_norm":
            "no CUDA device"}
    files = [{"sub": "stack", "file": "tests/test_x.py"}]
    ok = {"stack/tests/test_x.py": {"rc": 0, "n_tests": 9, "failed": [], "skipped": skipped}}
    assert LG.judge_pinned_run(files, ok, p["suite_pinned_skip_ok"])[1]["unregistered_skips"] == {}
    other = {"stack/tests/test_x.py": {"rc": 0, "n_tests": 9, "failed": [], "skipped": {
        "tests.test_refcv6_trunk::test_something_new": "no CUDA device",
        "tests.test_resim::test_x": "anchor fixture absent: refc_anchors_small64.pt"}}}
    assert len(LG.judge_pinned_run(files, other, p["suite_pinned_skip_ok"])[1][
        "unregistered_skips"]) == 2


def test_G_SUITE_PINNED_a_registered_skip_admits_ITS_test_only():
    """A skip registry entry is a (test id, reason) PAIR: the same reason on another test is
    still an unregistered skip (a skip is not a pass)."""
    files = [{"sub": "stack", "file": "tests/test_x.py"}]
    res = {"stack/tests/test_x.py": {"rc": 0, "n_tests": 3, "failed": [], "skipped": {
        "tests.test_x::test_a": "no CUDA device", "tests.test_x::test_b": "no CUDA device"}}}
    ok = [(r"tests\.test_x::test_a", r"^no CUDA device$")]
    r, d = LG.judge_pinned_run(files, res, ok)
    assert list(d["unregistered_skips"]) == ["tests.test_x::test_b"] and len(r) == 1
    assert LG.judge_pinned_run(files, res, [r"^no CUDA device$"])[0] == []     # legacy form



# ------------------------------------------------------------------------------------------ #
# MAP-LIFT is CLOSED (SPEC_REFCV7 17 A12 + 20 A15 + 23 A18): the map path's two flags are      #
# required values, the canonical argv carries them, and G-MAP-OVERFIT binds the A18 protocol   #
# ------------------------------------------------------------------------------------------ #
_A18_SPEC = (ROOT.parent / "TanitAD Research Lab" / "Architecture & Inference" / "Research"
             / "2026-09-26-refcv7-map-hires" / "raw" / "gmo_spec_A18.json")
_CANON_ARGV = ROOT / "ops" / "runs.d" / "refcv7-r101-s0.argv.json"


def test_MAP_LIFT_is_closed_and_its_flags_are_required_values():
    p = LG.PROFILES["refcv7"]
    assert "MAP-LIFT" not in [it["id"] for it in p["open_items"]]
    rv = p["required_values"]
    assert rv["--map-hires-near-lift-m"] == "20" and rv["--map-hires-near-refine-blocks"] == "1"
    for f in ("--map-hires-near-lift-m", "--map-hires-near-refine-blocks"):
        assert "SPEC_REFCV7" in p["required_values_why"][f]


@pytest.mark.skipif(not _CANON_ARGV.is_file(), reason="the canonical argv is not in this checkout")
def test_MAP_LIFT_the_canonical_argv_carries_the_map_path_and_a_missing_flag_is_refused():
    d = json.loads(_CANON_ARGV.read_text(encoding="utf-8"))
    argv = d["argv"]
    assert "todo_map_lift" not in d
    ch = {c["flag"]: c for c in d["changes_vs_refcv6"]}
    assert ch["--map-hires-near-lift-m"]["refcv7"] == ["20"]
    assert ch["--map-hires-near-refine-blocks"]["refcv7"] == ["1"]
    assert "SPEC_REFCV7 17" in ch["--map-hires-near-lift-m"]["why"]
    assert "SPEC_REFCV7 20" in ch["--map-hires-near-refine-blocks"]["why"]
    p = LG.PROFILES["refcv7"]
    reasons, _ = LG.required_on_reasons(p, argv)
    assert not [r for r in reasons if "--map-hires-near" in r], reasons
    # red arms: each flag dropped, and each at another value, is a NAMED refusal
    for f, other in (("--map-hires-near-lift-m", "10"), ("--map-hires-near-refine-blocks", "2")):
        i = argv.index(f)
        dropped = argv[:i] + argv[i + 2:]
        r, _ = LG.required_on_reasons(p, dropped)
        assert [x for x in r if x.startswith(f)] == [
            f"{f} {p['required_values'][f]} is REQUIRED for a refcv7 launch "
            f"({p['required_values_why'][f]}) and the argv has None"]
        wrong = argv[:i + 1] + [other] + argv[i + 2:]
        r, _ = LG.required_on_reasons(p, wrong)
        assert [x for x in r if x.startswith(f)] == [
            f"{f} {p['required_values'][f]} is REQUIRED for a refcv7 launch "
            f"({p['required_values_why'][f]}) and the argv has ['{other}']"]


@pytest.mark.skipif(not _A18_SPEC.is_file(), reason="the A18 spec is not in this checkout")
def test_G_MAP_OVERFIT_protocol_literals_are_the_A18_spec_itself():
    """The profile's protocol literals are not free-standing: they are the registered A18 spec's
    own (sha256, steps, decay, map path) -- a drift of either side is RED."""
    import hashlib
    mo = LG.PROFILES["refcv7"]["map_overfit"]
    b = _A18_SPEC.read_bytes()
    assert hashlib.sha256(b).hexdigest() == mo["spec_sha256"]
    s = json.loads(b.decode("utf-8"))
    assert s["steps"] == mo["protocol"]["steps"] == 3000
    assert s["lr_decay"] == mo["protocol"]["lr_decay"] == {"kind": "cosine_to_zero",
                                                           "start_step": 2700}
    assert s["lr"] == mo["protocol"]["lr"] == 0.001 and s["batch"] == mo["protocol"]["batch"] == 4
    assert s["seed"] == mo["protocol"]["seed"] == 0
    assert s["near_lift_m"] == mo["near_lift_m"] == 20.0
    assert s["near_refine_blocks"] == mo["near_refine_blocks"] == 1
    assert {k: tuple(v) for k, v in s["must_fail"].items() if k != "s8_zeros"} \
        == dict(mo["must_fail_any"])
    assert tuple(s["must_fail"]["s8_zeros"]) == mo["must_fail_all"]["s8_zeros"]


def test_G_MAP_OVERFIT_refuses_a_record_that_did_not_run_the_A18_protocol(tmp_path):
    """⛔ The box judge's trap (the prereg's lr kept after the protocol moved), checked for the map:
    the 1,000-step protocol, the prereg lr, no decay, the A17.1 decay, another spec, another map
    path and a missing near_block_zeros arm are each a NAMED refusal."""
    p = LG.PROFILES["refcv7"]
    commit, argv_sha, cw_sha = "ab" * 20, "cd" * 32, "ef" * 32
    rec = tmp_path / "g_map_overfit.json"

    def judge(r_):
        rec.write_text(json.dumps(r_), encoding="utf-8")
        return LG.judge_map_overfit(p, str(rec), commit, argv_sha=argv_sha,
                                    class_weights_sha256=cw_sha)[0]

    assert judge(_gmo_record(commit, argv_sha, cw_sha)) == []                  # A18 as run
    T_ = "G-MAP-OVERFIT: the record's optimiser"

    def opt(**kw):
        r_ = _gmo_record(commit, argv_sha, cw_sha)
        r_["optimiser"] = dict(r_["optimiser"], **kw)
        return r_
    old = opt(steps=1000, lr_decay=None)                                       # the prereg's
    assert judge(old) == [
        f"{T_} steps is 1000, not the registered 3000 (SPEC_REFCV7 23, A18)",
        f"{T_} lr_decay is None, not the registered "
        "{'kind': 'cosine_to_zero', 'start_step': 2700} (SPEC_REFCV7 23, A18)"]
    assert judge(opt(lr=2e-4)) == [
        f"{T_} lr is 0.0002, not the registered 0.001 (SPEC_REFCV7 23, A18)"]
    assert judge(opt(lr_decay={"kind": "cosine_to_zero", "start_step": 900})) == [
        f"{T_} lr_decay is {{'kind': 'cosine_to_zero', 'start_step': 900}}, not the registered "
        "{'kind': 'cosine_to_zero', 'start_step': 2700} (SPEC_REFCV7 23, A18)"]
    assert judge(opt(steps=True)) == [
        f"{T_} steps is True, not the registered 3000 (SPEC_REFCV7 23, A18)"]
    a171 = _gmo_record(commit, argv_sha, cw_sha, spec_sha256="5c" * 32)
    assert judge(a171) == ["G-MAP-OVERFIT: the record ran spec 5c5c5c5c5c5c5c5c..., not the "
                           "registered A18 spec 5cb4f6fc4a8c1566... (SPEC_REFCV7 23) or the A19 "
                           "MAIN-only spec 56dea067fdef45e9... (SPEC_REFCV7 24)"]
    nb = _gmo_record(commit, argv_sha, cw_sha, near_refine_blocks=2)
    assert judge(nb) == ["G-MAP-OVERFIT: the record ran near_refine_blocks=2, not the launch "
                         "map path's 1 (SPEC_REFCV7 17 / 20)"]
    nl = _gmo_record(commit, argv_sha, cw_sha)
    del nl["near_lift_m"]
    assert judge(nl) == ["G-MAP-OVERFIT: the record ran near_lift_m=None, not the launch map "
                         "path's 20.0 (SPEC_REFCV7 17 / 20)"]
    # an ABSENT must-fail arm: excused under the profile's A19 policy (SPEC_REFCV7 24) ...
    nbz = _gmo_record(commit, argv_sha, cw_sha)
    del nbz["verdict"]["regression_arms"]["near_block_zeros"]
    assert judge(nbz) == []
    # ... and, without the policy, the pre-A19 FAIL (every arm must run)
    p = dict(p, overfit_main_only=None)
    assert judge(nbz) == ["G-MAP-OVERFIT: must-fail arm 'near_block_zeros' did not FAIL "
                          "['edge'] (not run)"]


#: the Master Mind 2026-09-27: the launch argv LIST is FINAL -- the box A17 argv (150 tokens) with
#: `--map-hires-near-lift-m 20 --map-hires-near-refine-blocks 1` right after `--map-hires-grad-ckpt
#: on`. The binding runs bind THIS sha; the file's metadata may change, its list may not.
#: refcv7 RESTART (restart-options package 2026-09-28): the FINAL list with --conflict-every
#: 10 -> 50 (an instrument cadence; training numerics bit-identical). Launch list: 6402d33d.
_FINAL_ARGV_SHA256 = "e46ad3eb4099264754ba97846c0f3f5e8447ce61e21124dc980e2eb64464e9c8"


@pytest.mark.skipif(not _CANON_ARGV.is_file(), reason="the canonical argv is not in this checkout")
def test_the_canonical_argv_is_the_FINAL_list_and_its_gate_sha_is_the_pinned_literal():
    argv = json.loads(_CANON_ARGV.read_text(encoding="utf-8"))["argv"]
    assert len(argv) == 154
    assert LG.argv_sha256(argv) == _FINAL_ARGV_SHA256
    i = argv.index("--map-hires-grad-ckpt")
    assert argv[i:i + 6] == ["--map-hires-grad-ckpt", "on", "--map-hires-near-lift-m", "20",
                             "--map-hires-near-refine-blocks", "1"]
    # red arm: the same list without the map path is ANOTHER launch (another sha)
    assert LG.argv_sha256(argv[:i + 2] + argv[i + 6:]) != _FINAL_ARGV_SHA256


def _bom_tree(tmp_path):
    """A trainer whose closure holds a package whose `__init__.py` starts with a UTF-8 BOM (as
    `tanitad/eval/__init__.py` does) and imports a third-party name and its own submodule."""
    bom = "﻿".encode("utf-8")
    tree = tmp_path / "t"
    (tree / "stack" / "scripts").mkdir(parents=True)
    (tree / "stack" / "bompkg").mkdir(parents=True)
    (tree / "stack" / "scripts" / "train.py").write_text("import bompkg\n", encoding="utf-8")
    (tree / "stack" / "bompkg" / "__init__.py").write_bytes(
        bom + b"import numpy\nfrom bompkg import sub\n")
    (tree / "stack" / "bompkg" / "sub.py").write_text("import scipy\n", encoding="utf-8")
    return tree, bom


def test_G_HYG_import_closure_reads_a_BOM_source_as_python_does(tmp_path):
    """The Master Mind's dry run of the launch config (2026-09-27): "G-HYG import closure crashed:
    SyntaxError: invalid non-printable character U+FEFF (__init__.py, line 1)". Python's import
    accepts a BOM; `ast.parse` of a plain utf-8 decode does not -- so the closure reads utf-8-sig."""
    import ast
    tree, bom = _bom_tree(tmp_path)
    with pytest.raises(SyntaxError):                       # the mechanism, literally (red arm)
        ast.parse((bom + b"import os\n").decode("utf-8"))
    assert ast.parse((bom + b"import os\n").decode("utf-8-sig")).body   # the fix, literally
    clo = LG.import_closure(tree, "stack/scripts/train.py")
    assert clo["n_files"] == 3 and sorted(clo["third_party"]) == ["numpy", "scipy"]


def test_the_static_eager_closure_follows_a_BOM_package_instead_of_skipping_it(tmp_path):
    """The same read in `static_eager_own_files` swallowed the SyntaxError and SKIPPED the file, so a
    BOM'd package's own imports were never required of a closure record (a silent under-count)."""
    tree, _ = _bom_tree(tmp_path)
    assert LG.static_eager_own_files(tree, "stack/scripts/train.py") == [
        "stack/bompkg/__init__.py", "stack/bompkg/sub.py", "stack/scripts/train.py"]



# ------------------------------------------------------------------------------------------ #
# SPEC_REFCV7 24 (A19, the PI 2026-09-27): the BINDING overfit runs are MAIN-only -- an ABSENT #
# must-fail arm is excused, MAIN is re-judged from its OWN literals, a VOID is still a FAIL    #
# ------------------------------------------------------------------------------------------ #
_A19_MAP_SPEC_SHA256 = "56dea067fdef45e9b905f5741b562e59ad037396067cdbe43278c37f62be4f2a"
_A19_MAP_SPEC = (ROOT.parent / "TanitAD Research Lab" / "Architecture & Inference" / "Research"
                 / "2026-09-26-refcv7-map-hires" / "raw" / "gmo_spec_A19_MAP_MAIN.json")


def _gmo_main_only(commit, argv_sha, cw_sha, *, harness_verdict="FAIL"):
    """What the harness writes for the A19 MAIN-only binding: `--arms healthy` on the A19 spec -- no
    must-fail result, no regression row; `harness_verdict` is whatever the harness says (it is NOT
    read under A19)."""
    r_ = _gmo_record(commit, argv_sha, cw_sha, spec_sha256=_A19_MAP_SPEC_SHA256)
    r_["verdict"]["regression_arms"] = {}
    r_["verdict"]["G_MAP_OVERFIT"] = harness_verdict
    return r_


def _judge_map(tmp_path, p=None):
    commit, argv_sha, cw_sha = "ab" * 20, "cd" * 32, "ef" * 32
    rec = tmp_path / "g_map_overfit.json"

    def judge(r_, prof=None, full=False):
        rec.write_text(json.dumps(r_), encoding="utf-8")
        out = LG.judge_map_overfit(prof or p or LG.PROFILES["refcv7"], str(rec), commit,
                                   argv_sha=argv_sha, class_weights_sha256=cw_sha)
        return out if full else out[0]
    return judge, (commit, argv_sha, cw_sha)


def test_A19_the_refcv7_profile_names_the_policy_its_source_and_the_inherited_evidence():
    a = LG.PROFILES["refcv7"]["overfit_main_only"]
    assert a["policy"] == "A19_MAIN_ONLY_BINDING" and a["source"] == "SPEC_REFCV7 24 (A19, landed 36cc332)"
    assert a["map"]["spec_sha256"] == _A19_MAP_SPEC_SHA256
    assert "near_block_zeros read edge 0.199" in a["map"]["inherited"]
    assert a["map"]["evidence"][0].endswith("raw/gmo_early/g_map_overfit_A171.EARLY_NONBINDING.json")
    assert "memory_zeros was NOT measured at full scale on the launch head" in a["box"]["inherited"]
    assert "the PI accepted this knowingly (A19 Q2)" in a["box"]["inherited"]
    assert LG.PROFILES["refcv6"]["overfit_main_only"] is None                  # pre-A19 elsewhere


def test_A19_map_a_MAIN_only_record_PASSES_and_the_harness_verdict_does_not_decide(tmp_path):
    judge, k = _judge_map(tmp_path)
    for hv in ("FAIL", "PASS"):                  # what the harness says is NOT read under A19
        r, d = judge(_gmo_main_only(*k, harness_verdict=hv), full=True)
        assert r == [], r
        assert d["a19"]["applied"] is True
        assert d["a19"]["absent_must_fail_arms"] == ["lane_w0", "near_block_zeros", "s8_zeros"]
        assert d["must_fail"]["lane_w0"] == ("NOT RUN -- excused under SPEC_REFCV7 24 (A19, "
                                             "landed 36cc332); inherited")
        assert d["a19"]["statement"].startswith(
            "MAIN-ONLY binding under SPEC_REFCV7 24 (A19, landed 36cc332): the must-fail arm(s) "
            "['lane_w0', 'near_block_zeros', 's8_zeros'] were NOT RUN -- the map must-fail arms "
            "are INHERITED from A17.1")


def test_A19_map_MAIN_is_re_judged_from_its_OWN_numbers(tmp_path):
    """The harness says PASS everywhere; the record's own numbers miss -- each is a named FAIL (a
    judge that trusted the verdict would pass all of them)."""
    judge, k = _judge_map(tmp_path)
    edge = _gmo_main_only(*k, harness_verdict="PASS")
    edge["results"]["healthy"]["final"]["iou"]["edge"] = 0.49
    assert judge(edge) == ["G-MAP-OVERFIT: class 'edge' IoU 0.49 < the registered bar 0.5"]
    few = _gmo_main_only(*k, harness_verdict="PASS")
    few["results"]["healthy"]["final"]["n"]["arrow"] = 999
    assert judge(few) == ["G-MAP-OVERFIT: class 'arrow' has 999 scored cells < 1000 -- "
                          "INCONCLUSIVE => FAIL (prereg sec. 6.1)"]
    ce = _gmo_main_only(*k, harness_verdict="PASS")          # 1.2 > 0.5 x 2.0; the flag says ok
    ce["results"]["healthy"]["final"]["ce_mean"]["hatched"] = 1.2
    assert judge(ce) == ["G-MAP-OVERFIT: the per-class CE did not fall to <= 0.5x its step-0 "
                         "value for ['hatched'] (prereg sec. 6.4)"]
    nan = _gmo_main_only(*k, harness_verdict="PASS")
    nan["results"]["healthy"]["loss_finite_every_step"] = False
    assert judge(nan) == ["G-MAP-OVERFIT: the MAIN loss is not recorded finite at every step "
                          "(sec. 6.5)"]
    c2 = _gmo_main_only(*k, harness_verdict="PASS")
    c2["verdict"]["controls"]["C2_gt_as_logits"]["reproduced"] = False
    assert judge(c2) == ["G-MAP-OVERFIT: control C2 is not recorded as reproduced "
                         "({'reproduced': False})"]
    tg = _gmo_main_only(*k, harness_verdict="PASS")
    tg["verdict"]["time_guard"] = {"kinds": ["pose_5cm"], "time_1ms_on_every_clip": False}
    assert len(judge(tg)) == 1 and "1 ms label-time guard" in judge(tg)[0]
    a18 = _gmo_main_only(*k)                                  # MAIN-only needs a registered spec
    a18["spec_sha256"] = "5c" * 32
    assert "or the A19 MAIN-only spec 56dea067fdef45e9... (SPEC_REFCV7 24)" in judge(a18)[0]


def test_A19_map_a_must_fail_arm_that_RAN_and_PASSED_is_still_a_FAIL(tmp_path):
    """A19 excuses ABSENCE, never a VOID: an arm that ran is judged from its own result against the
    gate's literal bars (the harness's regression row cannot vouch for it)."""
    judge, k = _judge_map(tmp_path)
    lw = _gmo_main_only(*k)
    lw["results"]["lane_w0"] = {"final": {"iou": {c: b + 0.05 for c, b in _GMO_BARS.items()}}}
    assert judge(lw) == [
        "G-MAP-OVERFIT: must-fail arm 'lane_w0' did not FAIL ['lane'] ({'ran': True, 'failed': "
        "[], 'from': 'results'}) -- VOID: A19 excuses absence, never a VOID"]
    s8 = _gmo_main_only(*k)
    s8_iou = {c: 0.0 for c in _GMO_BARS}
    s8_iou["hatched"] = 0.6                                   # one thin class passes
    s8["results"]["s8_zeros"] = {"final": {"iou": s8_iou}}
    r = judge(s8)
    assert len(r) == 1 and r[0].startswith("G-MAP-OVERFIT: must-fail arm 's8_zeros' must FAIL ALL "
                                           "of ['lane', 'crosswalk', 'arrow', 'edge', 'hatched']")
    assert r[0].endswith("-- VOID: A19 excuses absence, never a VOID")
    ok = _gmo_main_only(*k)                                   # an arm that RAN and FAILED: fine
    ok["results"]["near_block_zeros"] = {"final": {"iou": dict({c: 0.9 for c in _GMO_BARS},
                                                                edge=0.2)}}
    assert judge(ok) == []


def test_A19_removed_from_the_profile_a_MAIN_only_record_FAILS_as_before(tmp_path):
    p = dict(LG.PROFILES["refcv7"], overfit_main_only=None)
    judge, k = _judge_map(tmp_path, p)
    assert judge(_gmo_main_only(*k)) == [
        "G-MAP-OVERFIT: the record's verdict is 'FAIL', not PASS",
        "G-MAP-OVERFIT: the record ran spec 56dea067fdef45e9..., not the registered A18 spec "
        "5cb4f6fc4a8c1566... (SPEC_REFCV7 23)",
        "G-MAP-OVERFIT: must-fail arm 'lane_w0' did not FAIL ['lane'] (not run)",
        "G-MAP-OVERFIT: must-fail arm 'near_block_zeros' did not FAIL ['edge'] (not run)",
        "G-MAP-OVERFIT: must-fail arm 's8_zeros' must FAIL ALL of ['lane', 'crosswalk', 'arrow', "
        "'edge', 'hatched'] (not run) -- a thin class passing without image information means "
        "the harness scores something else"]


@pytest.mark.skipif(not _A19_MAP_SPEC.is_file(), reason="the A19 map spec is not in this checkout")
def test_A19_the_map_spec_is_the_A18_spec_minus_its_must_fail_rows():
    import hashlib
    b = _A19_MAP_SPEC.read_bytes()
    assert hashlib.sha256(b).hexdigest() == _A19_MAP_SPEC_SHA256
    a19 = json.loads(b.decode("utf-8"))
    a18p = _A19_MAP_SPEC.with_name("gmo_spec_A18.json")
    assert "must_fail" not in a19 and "must_fail_all" not in a19
    assert a19["steps"] == 3000 and a19["lr_decay"] == {"kind": "cosine_to_zero", "start_step": 2700}
    if a18p.is_file():
        a18 = json.loads(a18p.read_text(encoding="utf-8"))
        diff = sorted(k for k in set(a18) | set(a19) if a18.get(k) != a19.get(k))
        assert diff == ["a19_inherited_must_fail_record", "amends", "must_fail", "must_fail_all",
                        "registered"], diff


def _box_main_only(argv_sha):
    """The box harness for `--arms main` (g_box_overfit main()): no must-fail arm, RESULT FAIL."""
    r_ = _gbo_record(argv_sha, RESULT="FAIL")
    del r_["arms"]["memory_zeros"], r_["arms"]["presence_w0"]
    return r_


def test_A19_box_a_MAIN_only_record_PASSES_is_re_judged_and_a_VOID_is_still_a_FAIL(tmp_path):
    p = LG.PROFILES["refcv7"]
    commit, argv = "ab" * 20, ["--arm", "hier", "--out", "/o"]
    argv_sha = LG.argv_sha256(argv)
    f = tmp_path / "g_box_overfit.json"

    def judge(rec, prof=p, full=False):
        f.write_text(json.dumps(rec), encoding="utf-8")
        out = LG.judge_box_overfit(prof, str(f), commit, argv_sha=argv_sha, argv=argv)
        return out if full else out[0]
    r, d = judge(_box_main_only(argv_sha), full=True)
    assert r == [] and d["a19"]["applied"] is True
    assert d["a19"]["absent_must_fail_arms"] == ["memory_zeros", "presence_w0"]
    assert "memory_zeros was NOT measured at full scale on the launch head" in d["a19"]["statement"]
    good = _box_main_only(argv_sha)
    good["RESULT"] = "PASS"                                    # RESULT is not read either way
    assert judge(good) == []
    low = _box_main_only(argv_sha)                             # main's rows decide, not its flags
    low["RESULT"] = "PASS"
    low["arms"]["main"]["rows"][-1]["ap2m"] = 0.85
    assert judge(low) == ["G-BOX-OVERFIT: main fails prereg criterion 1 ['ap2m'] at step 2000: "
                          "{'ap2m': 0.85}"]
    sched = _box_main_only(argv_sha)
    sched["literals"]["lr_schedule"] = {"rule": "A13", "decay": "none"}
    assert judge(sched)[0].startswith("G-BOX-OVERFIT: literal lr_schedule = {'rule': 'A13'")
    ign = _box_main_only(argv_sha)
    ign["literals"]["n_ign"] = 28
    assert judge(ign) == ["G-BOX-OVERFIT: literal n_ign = 28, the prereg (with A10's "
                          "reconciliation) fixes 77"]
    void = _box_main_only(argv_sha)                            # an arm that RAN and passed
    void["arms"]["memory_zeros"] = _gbo_arm(dict(_GBO_MAIN, ap2m=0.93), verdict="VOID")
    assert judge(void) == ["G-BOX-OVERFIT: must-fail arm 'memory_zeros' must FAIL prereg "
                           "criteria ['1'] and passes ['1'] -- the result is VOID (prereg sec. 7)"]
    vres = _box_main_only(argv_sha)
    vres["RESULT"] = "VOID"
    assert judge(vres) == ["G-BOX-OVERFIT: the record's RESULT is 'VOID' -- a must-fail arm passed "
                           "(A19 excuses absence, never a VOID)"]
    # the profile's A19 field removed: the pre-A19 FAIL
    assert judge(_box_main_only(argv_sha), prof=dict(p, overfit_main_only=None)) == [
        "G-BOX-OVERFIT: the record's RESULT is 'FAIL', not PASS",
        "G-BOX-OVERFIT: must-fail arm 'memory_zeros' did not run (no rows)",
        "G-BOX-OVERFIT: must-fail arm 'presence_w0' did not run (no rows)"]


def test_A19_the_PASS_text_says_what_the_token_does_not_cover(tmp_path, monkeypatch):
    """The evidence's PASS reason carries A19's statement when (and only when) A19 applied."""
    monkeypatch.setattr(LG, "judge_closure", lambda *a, **k: ([], {}))
    ctx, _ = _refcv7_argv_ctx(tmp_path)
    for job, check, jfn in ((LG._job_map_overfit, "G-MAP-OVERFIT", "judge_map_overfit"),
                            (LG._job_box_overfit, "G-BOX-OVERFIT", "judge_box_overfit")):
        monkeypatch.setattr(LG, jfn, lambda *a, **k: ([], {"a19": {"applied": True,
                                                                   "statement": "S19"}}))
        ev = job(ctx)[check]
        assert ev["status"] == "PASS" and ev["reasons"] == [f"{check} PASS {LG.OVERFIT_SCOPE}; S19"]
        monkeypatch.setattr(LG, jfn, lambda *a, **k: ([], {"a19": {"applied": False}}))
        assert job(ctx)[check]["reasons"] == [f"{check} PASS {LG.OVERFIT_SCOPE}"]


#: the refcv7 eval loader (package `…/2026-09-27-refcv7-eval-loader/`, the loader agent's
#: integration diff): G-EVAL's DEFAULT for refcv7; refcv6 keeps its battery loader
def test_the_refcv7_profile_evaluates_with_its_OWN_loader_by_default():
    assert LG.PROFILES["refcv7"]["eval_loader"] == "stack/tanitad/eval/refcv7_loader.py"
    assert LG.PROFILES["refcv6"]["eval_loader"] == LG._EVAL_LOADER_REL        # red arm: refcv6's
    assert LG.PROFILES["refcv6"]["eval_loader"] != LG.PROFILES["refcv7"]["eval_loader"]
    loader = ROOT / "tanitad" / "eval" / "refcv7_loader.py"
    if loader.is_file():                        # it lands in the same gate: resolves in the tree
        assert LG._resolve_repo_rel(ROOT.parent, LG.PROFILES["refcv7"]["eval_loader"]) == loader
