"""refcv9 E1-REUSE -- an arm binds to an EXISTING PASSING overfit record when the harness invocation it would run is IDENTICAL to the base's.

THE MM'S DESIGN DECISION (pinned, 207ebd): G-MAP-OVERFIT / G-BOX-OVERFIT bind the WHOLE launch-argv sha256, so every arm ran both harnesses (1,993 s map /
2,294 s box on the one Thor GPU). A harness that does not BUILD the arm's lever (the training-only aux heads, E1-REC3; T1, E1-REC4: the harness STRIPS those flags)
computes exactly what the BASE arm's run computed. The harness's OWN ``--identity-only`` (argv parsing + strip + data roots, then exit BEFORE any model / data / GPU)
gives that identity; ``launch_gate.judge_overfit_reuse`` re-derives BOTH identities on the base's recorded invocation and requires them equal.

Pinned here:
 (a) the identities: B1 / B2 / B2c / T1 / T1s box == C0's; B3 / B3p / M1 / C1 box differ; the map identity is one for every non-banded arm; a banded argv is refused;
 (b) ONE formula in the two harnesses and the gate (and the same key tuple);
 (c) the static pin the hash rests on: ``args.out`` is read ONCE, inside ``refc_v3_train.train()`` -- which the harnesses never call (with a mutation: a second read
     is caught);
 (d) the identity mode is the REAL run's own front half: nothing past it is reachable, its block equals the pure function's, and its stamps equal the fields a run writes;
 (e) the RED arms through the real judge, on a synthetic base record + closure; GREEN: B1 / T1 bind to C0.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TREE = ROOT.parent
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import launch_gate as LG  # noqa: E402
import test_launch_gate as TG  # noqa: E402
import test_launch_gate_e1rig_e1rec2 as R2  # noqa: E402

G = pytest.importorskip("g_box_overfit")
O = pytest.importorskip("map_hires_overfit")
torch = pytest.importorskip("torch")

RIG = LG.PROFILES["refcv9-e1rig"]
RUNS = ROOT / "ops" / "runs.d"
COMMIT = "ab" * 20


def _host_class_weights_sha() -> str:
    """The class-weight sha the gate's G-MAP-OVERFIT job (`_job_map_overfit`) declares ON THIS HOST: the rig argv's
    --map-hires-class-weights file, hashed only when it exists here. On Thor it EXISTS (MEASURED 2026-11-03: d70dec8087ed73ee, the same
    path for every r9e1 arm), so a synthetic base record stamping a fake sha is refused on its class weights before the judge reaches the
    reuse rule under test; on the dev box it is absent and the gate skips the comparison. Stamp what the gate will compare."""
    v = LG.flag_values(LG.load_argv_file(RUNS / "r9e1_C0.argv.json"), RIG["map_hires"]["class_weights_flag"])
    return LG.sha256_file(v[0]) if v and Path(v[0]).is_file() else "ef" * 32


CW_SHA = _host_class_weights_sha()
SAME_AS_C0 = ("B1", "B2", "B2c", "T1", "T1s")             # aux (E1-REC3) + T1 (E1-REC4): flags the harness STRIPS
ALL_ARMS = ("C0", "C1", "B1", "B2", "B2c", "B3", "B3p", "T1", "T1s", "M1")
NEEDS_ITS_OWN = ("C1", "B3", "B3p", "M1")


def _argv_file(name) -> Path:
    return RUNS / f"r9e1_{name}.argv.json"


def _argv(name):
    return json.loads(_argv_file(name).read_text(encoding="utf-8"))["argv"]


def _audit(d: Path) -> Path:
    """The box harness's audit dir (prereg + frame set), a copy of the shipped one: the identity hashes its CONTENT, never its location."""
    dst = d / "audit"
    if not dst.exists():
        shutil.copytree(ROOT / "ops" / "refcv8_box_audit" / "raw", dst / "raw")
    return dst


def _box_tokens(argv_file, audit, out, *, commit=COMMIT, arms="main", candidate="x", device="cuda"):
    return ["--launch-argv", str(argv_file), "--audit-dir", str(audit), "--out", str(out), "--device", device, "--binding", "--commit", commit,
            "--candidate", candidate, "--arms", arms]


def _map_tokens(argv_file, out, *, commit=COMMIT, with_argv=False):
    sha = LG.argv_sha256(LG.load_argv_file(argv_file))
    t = ["--spec", "/in/gmo_spec_A19_MAP_MAIN.json", "--frameset", "/in/gmo_frameset.json", "--class-weights", "/data/map_hires_class_weights.json",
         "--decision-rule", "prior_corrected", "--near-lift-m", "20", "--near-refine-blocks", "1", "--v2-cache", "/data/refcv6-b1-416x1024-train",
         "--gt-root", "/data/sam3_gt_v3", "--extrinsics", "/data/extrinsics.json", "--cam-ts-dir", "/data/camera_front_wide",
         "--launch-commit", commit, "--launch-argv-sha256", sha, "--arms", "healthy", "--out", str(out), "--device", "cuda"]
    return t + (["--launch-argv", str(argv_file)] if with_argv else [])


def _identity(mod, tokens, out: Path) -> dict:
    assert mod.main(list(tokens) + ["--identity-only", str(out)]) == 0
    return json.loads(out.read_text(encoding="utf-8"))


def _box_id(name, d: Path, **kw):
    return _identity(G, _box_tokens(_argv_file(name), _audit(d), d / "rec.json", **kw), d / f"{name}.box.identity.json")


def _map_id(name, d: Path, **kw):
    return _identity(O, _map_tokens(_argv_file(name), d / "map_out", **kw), d / f"{name}.map.identity.json")


# ====================================================================================== (a) the identities ==
@pytest.fixture(scope="module")
def ids(tmp_path_factory):
    d = tmp_path_factory.mktemp("ids")
    box = {n: _box_id(n, d) for n in ALL_ARMS}
    mp = {n: _map_id(n, d, with_argv=(n in ("C0", "T1", "T1s"))) for n in ALL_ARMS if n != "M1"}
    return box, mp


def test_a_the_aux_and_T1_arms_have_the_BASE_arms_box_identity_and_every_own_head_arm_has_its_own(ids):
    box, _ = ids
    want = box["C0"]["identity_sha256"]
    for n in SAME_AS_C0:
        assert box[n]["identity_sha256"] == want, n
        assert {k: box[n][k] for k in G.IDENTITY_KEYS} == {k: box["C0"][k] for k in G.IDENTITY_KEYS}
    for n in NEEDS_ITS_OWN:
        assert box[n]["identity_sha256"] != want, n
    assert box["B3"]["identity_sha256"] != box["B3p"]["identity_sha256"]
    # the per-arm stamps ride along but are NOT hashed: they are what each arm's binding carries
    assert box["B1"]["stamps"][G.AUX_STAMP] == ["--r9-box-depth-aux"] and box["C0"]["stamps"][G.AUX_STAMP] == []
    assert box["T1s"]["stamps"][G.T1_STAMP] == ["--r9-bev-warp", "--r9-bev-warp-shuffle"] and box["B1"]["stamps"][G.T1_STAMP] == []
    assert box["C0"]["launch_argv_sha256"] != box["B1"]["launch_argv_sha256"] and box["C0"]["effective_build_argv"] == box["B1"]["effective_build_argv"]
    assert any(f in box["B1"]["effective_build_argv_exact"] for f in ("--r9-box-depth-aux", "--r9-box-persp-aux")) is False   # the harness's own strip


def test_a_the_map_identity_is_ONE_for_every_non_banded_arm_with_or_without_the_launch_argv_and_a_banded_argv_is_refused(ids, tmp_path):
    _, mp = ids
    assert len({b["identity_sha256"] for b in mp.values()}) == 1 and mp["C0"]["build_facts"] == {"banded": False, "own_enc": False}
    assert mp["T1"]["stamps"][O.T1_STAMP] == ["--r9-bev-warp"] and mp["B3"]["stamps"][O.T1_STAMP] == []
    for name in ("M1", "M2a", "M2b"):                                      # a non-banded invocation of a banded argv: the harness REFUSES, as a run does
        with pytest.raises(SystemExit, match="BANDED head"):
            _map_id(name, tmp_path, with_argv=True)


@pytest.mark.parametrize("name", ["M1", "M2a", "M2b"])
def test_a_a_banded_argv_has_a_banded_map_identity_that_is_not_the_inherited_one(name, tmp_path, ids):
    toks = _map_tokens(_argv_file(name), tmp_path / "o", with_argv=True) + ["--banded"]
    blk = _identity(O, toks, tmp_path / "b.json")
    assert blk["build_facts"]["banded"] is True and blk["identity_sha256"] != ids[1]["C0"]["identity_sha256"]


def test_a_the_identity_hashes_what_the_harness_computes_and_never_where_it_wrote_or_read(tmp_path):
    base = _box_id("C0", tmp_path / "a")
    d = tmp_path / "b"
    d.mkdir()
    other = _identity(G, _box_tokens(_argv_file("C0"), _audit(d), d / "elsewhere.json", candidate="a different label"), d / "i.json")
    assert other["identity_sha256"] == base["identity_sha256"]                   # --out, --candidate, the audit-dir LOCATION: references
    for kw in ({"arms": "main,memory_zeros"}, {"device": "cpu"}, {"commit": "cd" * 20}):   # what the run computes / the code it binds: inputs
        dd = tmp_path / ("c_" + "_".join(kw))
        dd.mkdir()
        assert _box_id("C0", dd, **kw)["identity_sha256"] != base["identity_sha256"], kw
    red = tmp_path / "red"
    red.mkdir()
    pre = _audit(red) / "raw" / "PREREG_G_BOX_OVERFIT.md"
    pre.write_bytes(pre.read_bytes() + b"x")                                       # one byte of the audit CONTENT
    assert _box_id("C0", red)["identity_sha256"] != base["identity_sha256"]
    assert base["audit_inputs"]["prereg_md5"] == hashlib.md5((ROOT / "ops/refcv8_box_audit/raw/PREREG_G_BOX_OVERFIT.md").read_bytes()).hexdigest()


# ===================================================================================== (b) one formula ==
def test_b_the_three_identity_formulas_and_the_key_tuples_are_ONE(ids):
    box, mp = ids
    assert G.IDENTITY_KEYS == O.IDENTITY_KEYS == LG.REUSE_IDENTITY_KEYS == ("schema", "tool", "effective_build_argv", "build_facts", "data_roots", "other_args",
                                                                            "audit_inputs")
    assert G.IDENTITY_SCHEMA == O.IDENTITY_SCHEMA == "tanitad.overfit_identity/1"
    for blk in (box["C0"], box["B3"], mp["C0"]):
        core = {k: blk[k] for k in G.IDENTITY_KEYS}
        assert G.identity_sha256(core) == O.identity_sha256(core) == LG.overfit_identity_sha256(core) == blk["identity_sha256"]
        core2 = dict(core, other_args=dict(core["other_args"], zz=1))                  # a changed input moves ALL THREE together
        assert len({G.identity_sha256(core2), O.identity_sha256(core2), LG.overfit_identity_sha256(core2)}) == 1
        assert G.identity_sha256(core2) != blk["identity_sha256"]


# ================================================================== (c) args.out is read once, in train() ==
def _out_reads(src: str) -> list[tuple[str, int]]:
    """[(enclosing top-level function, lineno)] of every `<name>.out` attribute LOAD in the source."""
    tree = ast.parse(src)
    hits: list[tuple[str, int]] = []
    for top in tree.body:
        for n in ast.walk(top):
            if isinstance(n, ast.Attribute) and n.attr == "out" and isinstance(n.ctx, ast.Load) and isinstance(n.value, ast.Name) and n.value.id in ("args", "a"):
                hits.append((getattr(top, "name", "<module>"), n.lineno))
    return hits


TRAINER = ROOT / "scripts" / "refc_v3_train.py"


def test_c_args_out_is_read_exactly_once_and_inside_train_which_neither_harness_calls():
    reads = _out_reads(TRAINER.read_text(encoding="utf-8"))
    assert [f for f, _ in reads] == ["train"], reads                                # the one read the hash excludes `--out` on
    for harness in (G, O):
        src = Path(harness.__file__).read_text(encoding="utf-8")
        calls = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "train"
                 and isinstance(n.func.value, ast.Name) and n.func.value.id in ("tr", "T", "trainer", "refc_v3_train")]
        assert calls == [], (harness.__name__, [c.lineno for c in calls])
    for sub in ("models", "refs", "train", "data"):                                 # the library code the build path imports reads no `args.out` either
        for p in (ROOT / "tanitad" / sub).rglob("*.py"):
            assert [h for h in _out_reads(p.read_text(encoding="utf-8")) if h[0] != "main"] == [], p      # a CLI `main()` is not on the build path


def test_c_MUTATION_a_second_read_of_args_out_is_caught():
    src = TRAINER.read_text(encoding="utf-8")
    assert len(_out_reads(src)) == 1
    assert len(_out_reads(src + "\n\ndef build_something(args):\n    return args.out\n")) == 2                       # another function reads it
    assert [f for f, _ in _out_reads(src + "\n\ndef build_something(args):\n    return args.out\n")][-1] == "build_something"
    assert len(_out_reads(src.replace("out_dir = Path(args.out)", "out_dir = Path(args.out_dir_x)", 1))) == 0            # the pin tracks the read itself


# ====================================== (d) the identity mode is the REAL run's own front half, nothing more ==
def test_d_the_identity_mode_exits_before_any_model_data_or_gpu(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("past the identity exit")
    for nm in ("_load_by_path", "load_vendored_loader", "TrainerAdapter", "control_c3"):
        monkeypatch.setattr(G, nm, boom)
    for nm in ("main_banded", "load_spec"):
        monkeypatch.setattr(O, nm, boom)
    assert _box_id("T1", tmp_path)["identity_sha256"] and _map_id("T1", tmp_path, with_argv=True)["identity_sha256"]
    assert not (tmp_path / "rec.json").exists() and not (tmp_path / "map_out").exists()          # no record, no run directory


def test_d_the_box_block_equals_the_pure_function_and_its_stamps_equal_the_fields_a_run_writes(tmp_path):
    for name in ("C0", "B1", "T1s", "B3"):
        d = tmp_path / name
        d.mkdir()
        blk = _box_id(name, d)
        launch_argv, argv_rec = G.load_launch_argv(_argv_file(name))
        ns = argparse.Namespace(arms="main", steps=G.STEPS, device="cuda", binding=True, commit=COMMIT, check_build=False, frameset_profile="refcv8", one_frame=False)
        audit = {"prereg_md5": hashlib.md5((d / "audit/raw/PREREG_G_BOX_OVERFIT.md").read_bytes()).hexdigest(),
                 "frameset_md5": hashlib.md5((d / "audit/raw/visibility/gbo_frameset.json").read_bytes()).hexdigest(), "addendum_md5": None}
        want = G.identity_block(launch_argv, G.identity_harness_part(ns, audit))
        assert {k: v for k, v in blk.items() if k != "harness_tokens"} == json.loads(json.dumps(want, default=str)), name
        build, tc = G.harness_build_argv(launch_argv)                                     # what a real run writes into its record
        run_fields = {**argv_rec, **G.b3_record_fields(launch_argv), **tc}
        assert blk["stamps"][G.AUX_STAMP] == list(run_fields.get("aux_flags_not_built", []))
        assert blk["stamps"][G.T1_STAMP] == list(run_fields.get("bev_warp_flags_not_built", []))
        assert blk["effective_build_argv_exact"] == build and blk["launch_argv_sha256"] == run_fields["launch_argv_sha256"]
        assert "trunk_compile_in_launch_argv" in run_fields and "--trunk-compile" not in blk["effective_build_argv"]
        assert blk["harness_tokens"][:2] == ["--launch-argv", str(_argv_file(name))]


def test_d_the_map_block_equals_the_pure_function_and_a_wrong_argv_sha_is_refused_as_the_run_refuses(tmp_path):
    for name in ("C0", "T1"):
        blk = _map_id(name, tmp_path, with_argv=True)
        ns = argparse.Namespace(banded=False, arms="healthy", decision_rule="prior_corrected", near_lift_m=20.0, near_refine_blocks=1, x_max_m=O.EXTENT_REFCV7.x_max_m,
                                y_half_m=O.EXTENT_REFCV7.y_half_m, grad_ckpt="on", device="cuda", launch_commit=COMMIT, own_enc=False,
                                spec=Path("/in/gmo_spec_A19_MAP_MAIN.json"), frameset=Path("/in/gmo_frameset.json"), class_weights="/data/map_hires_class_weights.json",
                                v2_cache=Path("/data/refcv6-b1-416x1024-train"), gt_root=Path("/data/sam3_gt_v3"), extrinsics=Path("/data/extrinsics.json"),
                                cam_ts_dir=Path("/data/camera_front_wide"))
        want = O.identity_block(LG.load_argv_file(_argv_file(name)), O.identity_harness_part(ns))
        assert {k: v for k, v in blk.items() if k != "harness_tokens"} == json.loads(json.dumps(want, default=str)), name
    bad = [t if t != LG.argv_sha256(_argv("T1")) else "0" * 64 for t in _map_tokens(_argv_file("T1"), tmp_path / "o", with_argv=True)]
    with pytest.raises(SystemExit):
        O.main(bad + ["--identity-only", str(tmp_path / "bad.json")])
    assert not (tmp_path / "bad.json").exists()


# ============================================================ (e) the RED arms through the real judge ==
class Base:
    """A synthetic BASE arm: its record (judged by the gate's own judge), its closure (the harness invocation it ACTUALLY ran) and where they live."""

    def __init__(self, tmp_path, kind, name="C0", commit=COMMIT, **rec_over):
        self.kind, self.name, self.commit = kind, name, commit
        self.check = {"box": "G-BOX-OVERFIT", "map": "G-MAP-OVERFIT"}[kind]
        self.dir = tmp_path / f"base_{kind}_{name}"
        self.dir.mkdir(parents=True)
        sha = LG.argv_sha256(_argv(name))
        if kind == "box":
            rec = TG._box_main_only(sha)
            rec["commit"] = commit
            self.tokens = _box_tokens(_argv_file(name), _audit(self.dir), self.dir / "gbo_binding.json", commit=commit)
            self.script, self.mod = "stack/scripts/g_box_overfit.py", G
        else:
            rec = TG._gmo_main_only(commit, sha, CW_SHA)
            self.tokens = _map_tokens(_argv_file(name), self.dir / "run_map", commit=commit)
            self.script, self.mod = "stack/scripts/map_hires_overfit.py", O
        rec.update(rec_over)
        self.rec = self.dir / "record.json"
        self.rec.write_text(json.dumps(rec), encoding="utf-8")
        cargv = [self.script, *self.tokens]
        self.clo = self.dir / "closure.json"
        self.clo.write_text(json.dumps({"argv": cargv, "argv_sha256": hashlib.sha256(json.dumps(cargv, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()}),
                            encoding="utf-8")
        self.identity = self._identity(self.dir / "base.identity.json", _argv_file(name), is_base=True)

    def _identity(self, out, argv_file, is_base=False):
        toks = list(self.tokens)
        if self.kind == "box":
            toks[toks.index("--launch-argv") + 1] = str(argv_file)
        else:
            toks[toks.index("--launch-argv-sha256") + 1] = LG.argv_sha256(LG.load_argv_file(argv_file))
            if not is_base:
                toks += ["--launch-argv", str(argv_file)]
        return _identity(self.mod, toks, Path(out))

    def bind(self, tmp_path, this_arm, *, argv_file=None, this_identity=None, commit=None):
        """r9e1_bind.sh's own writer + verifier (`write_reuse_binding` = the gate's judge). -> (reasons, details, binding path)."""
        af = Path(argv_file or _argv_file(this_arm))
        out = tmp_path / f"bind_{this_arm}_{self.kind}_{len(list(tmp_path.glob('bind_*')))}"
        out.mkdir(parents=True)
        tid = this_identity or str(out / "this.identity.json")
        if not this_identity:
            self._identity(tid, af)
        clo = shutil.copy2(self.clo, out / f"{self.kind}_closure.json")
        reasons, det = LG.write_reuse_binding("refcv9-e1rig", self.check, this_arm, self.name, str(af), str(_argv_file(self.name)), str(self.rec), str(self.clo),
                                              tid, str(self.dir / "base.identity.json"), commit or self.commit, str(out / "reuse_binding.json"), str(TREE),
                                              closure_option=str(clo))
        return reasons, det, out / "reuse_binding.json"

    def judge(self, binding, argv, *, work):
        return LG.judge_overfit_reuse(RIG, self.check, str(binding), self.commit, argv_sha=LG.argv_sha256(argv), argv=argv, tree=TREE, workdir=work,
                                      closure_option=str(Path(binding).parent / f"{self.kind}_closure.json"))


def _joined(reasons):
    return " | ".join(reasons)


@pytest.mark.parametrize("kind", ["box", "map"])
@pytest.mark.parametrize("arm", ["B1", "T1"])
def test_e_GREEN_B1_and_T1_bind_to_C0s_record_and_the_gates_judge_accepts_the_written_binding(tmp_path, kind, arm):
    base = Base(tmp_path, kind)
    reasons, det, bp = base.bind(tmp_path, arm)
    assert reasons == [] and bp.is_file(), reasons
    b = json.loads(bp.read_text(encoding="utf-8"))
    assert b["schema"] == LG.REUSE_SCHEMA and b["base_arm"] == "C0" and b["this_arm"] == arm and b["kind"] == kind
    assert b["this_identity_sha256"] == b["base_identity_sha256"] == det["identity"]["this"] == det["identity"]["base"]
    assert b["this_argv_sha256"] == LG.argv_sha256(_argv(arm)) != b["base_argv_sha256"]
    assert b["this_stamps"][LG.E1_BOX_AUX_STAMP] == [f for f in _argv(arm) if f in LG.E1_BOX_AUX_FLAGS]
    assert b["this_stamps"][LG.E1_T1_STAMP] == [f for f in _argv(arm) if f in LG.E1_T1_FLAGS]
    r2, d2, rec, clo = base.judge(bp, _argv(arm), work=tmp_path / "w")                      # the judge, again, from the file alone
    assert r2 == [] and Path(rec).samefile(base.rec) and Path(clo).samefile(base.clo)
    assert (arm == "B1" and kind == "box") == bool(d2.get("aux_heads_not_certified")) and (arm == "T1") == bool(d2.get("bev_warp_not_certified"))


def test_e_RED_one_build_argv_token_different_is_refused_and_no_binding_is_left(tmp_path):
    base = Base(tmp_path, "box")
    af = tmp_path / "b1_seed7.argv.json"
    af.write_text(json.dumps(LG.set_flag(_argv("B1"), "--seed", ["7"])), encoding="utf-8")
    reasons, _, bp = base.bind(tmp_path, "B1", argv_file=af)
    assert "NOT identical to the base's (differs in ['effective_build_argv'" in _joined(reasons) and not bp.exists()
    af2 = tmp_path / "b1_lr.argv.json"                                                      # an argv that changes the build in another flag
    af2.write_text(json.dumps(LG.set_flag(_argv("B1"), "--r9-box-quality", [])), encoding="utf-8")
    assert "OWN record is required" in _joined(base.bind(tmp_path, "B1", argv_file=af2)[0])


def test_e_RED_an_own_head_box_argv_cannot_take_a_reuse_binding_of_kind_box(tmp_path):
    base = Base(tmp_path, "box")
    for arm in ("B3", "B3p"):
        reasons, _, bp = base.bind(tmp_path, arm)
        txt = _joined(reasons)
        assert "OWN record is required" in txt and "NOT identical" in txt and not bp.exists(), (arm, txt)


def test_e_RED_a_banded_argv_cannot_take_a_reuse_binding_of_kind_map(tmp_path):
    base = Base(tmp_path, "map")
    d = tmp_path / "idm1"
    d.mkdir()
    ident = _map_id("C0", d, with_argv=True)                                                 # M1 has no non-banded identity: offer C0's
    (d / "m1.identity.json").write_text(json.dumps(ident), encoding="utf-8")
    reasons, _, bp = base.bind(tmp_path, "M1", this_identity=str(d / "m1.identity.json"))
    txt = _joined(reasons)
    assert not bp.exists() and "REFUSED the this argv in --identity-only" in txt and "BANDED head" in txt


def test_e_RED_a_base_record_at_another_commit_is_refused(tmp_path):
    base = Base(tmp_path, "box")
    reasons, _, bp = base.bind(tmp_path, "B1", commit="cd" * 20)
    assert not bp.exists() and f"the base record was made at commit {COMMIT[:12]}, the binding is at {('cd' * 20)[:12]}" in _joined(reasons)
    mbase = Base(tmp_path, "map")                                                            # the map record binds the commit in its own judge too
    reasons, _, bp = mbase.bind(tmp_path, "B1", commit="cd" * 20)
    assert not bp.exists() and any("base record" in r for r in reasons)


def test_e_RED_a_base_record_that_FAILS_its_own_judge_is_refused(tmp_path):
    base = Base(tmp_path, "box")
    rec = json.loads(base.rec.read_text(encoding="utf-8"))
    rec["arms"]["main"]["rows"][-1]["ap2m"] = 0.85                                          # main fails the pre-registered bar
    base.rec.write_text(json.dumps(rec), encoding="utf-8")
    reasons, _, bp = base.bind(tmp_path, "B1")
    assert not bp.exists() and any("main fails prereg criterion 1" in r and "[base record, re-judged for the base argv]" in r for r in reasons)
    mbase = Base(tmp_path, "map", launch_argv_sha256="0" * 64)                              # a map record bound to ANOTHER argv
    reasons, _, bp = mbase.bind(tmp_path, "B1")
    assert not bp.exists() and any("[base record, re-judged for the base argv]" in r for r in reasons)


@pytest.mark.parametrize("what", ["base_record", "base_closure"])
@pytest.mark.parametrize("kind", ["box", "map"])
def test_e_RED_the_base_record_or_closure_edited_after_the_binding_FAILS(tmp_path, kind, what):
    base = Base(tmp_path, kind)
    reasons, _, bp = base.bind(tmp_path, "B1")
    assert reasons == []
    assert base.judge(bp, _argv("B1"), work=tmp_path / "w0")[0] == []
    target = base.rec if what == "base_record" else base.clo
    target.write_bytes(target.read_bytes() + b" ")                                           # ONE byte
    txt = _joined(base.judge(bp, _argv("B1"), work=tmp_path / "w1")[0])
    assert f"the {what} " in txt and "edited after the binding" in txt


def test_e_RED_a_stamp_that_is_not_THIS_argvs_set_FAILS_and_a_binding_offered_to_another_arm_FAILS(tmp_path):
    base = Base(tmp_path, "box")
    reasons, _, bp = base.bind(tmp_path, "B1")
    assert reasons == []
    b = json.loads(bp.read_text(encoding="utf-8"))
    for stamp, bad in ((LG.E1_BOX_AUX_STAMP, []), (LG.E1_BOX_AUX_STAMP, ["--r9-box-persp-aux"]), (LG.E1_BOX_AUX_STAMP, ["--r9-box-depth-aux", "--r9-box-depth-aux"])):
        c = json.loads(json.dumps(b))
        c["this_stamps"][stamp] = bad
        bp.write_text(json.dumps(c), encoding="utf-8")
        assert f"the binding's `{stamp}` is {bad!r}" in _joined(base.judge(bp, _argv("B1"), work=tmp_path / "w2")[0]), bad
    c = json.loads(json.dumps(b))
    c["this_stamps"][LG.E1_T1_STAMP] = ["--r9-bev-warp"]                                     # a T1 stamp on an argv with no T1 flag
    bp.write_text(json.dumps(c), encoding="utf-8")
    assert f"the binding's `{LG.E1_T1_STAMP}` is ['--r9-bev-warp']" in _joined(base.judge(bp, _argv("B1"), work=tmp_path / "w3")[0])
    bp.write_text(json.dumps(b), encoding="utf-8")                                           # control: the untouched binding passes again
    assert base.judge(bp, _argv("B1"), work=tmp_path / "w4")[0] == []
    other = _joined(base.judge(bp, _argv("B2"), work=tmp_path / "w5")[0])                    # B1's binding for B2's launch: sha AND stamp differ
    assert "the binding is for argv" in other and "not the launch argv" in other and "the binding's `aux_heads_not_certified`" in other


def test_e_RED_a_foreign_schema_or_kind_or_missing_base_is_refused(tmp_path):
    base = Base(tmp_path, "box")
    reasons, _, bp = base.bind(tmp_path, "B1")
    b = json.loads(bp.read_text(encoding="utf-8"))
    assert "no tanitad.overfit_reuse/1 binding" in _joined(LG.judge_overfit_reuse(RIG, "G-BOX-OVERFIT", str(tmp_path / "absent.json"), COMMIT, argv_sha="0" * 64,
                                                                                  argv=_argv("B1"), tree=TREE, workdir=tmp_path / "w")[0])
    c = dict(b, kind="map")
    bp.write_text(json.dumps(c), encoding="utf-8")
    assert "the binding is of kind 'map'" in _joined(base.judge(bp, _argv("B1"), work=tmp_path / "wk")[0])
    c = dict(b, base_record=dict(b["base_record"], relpath="nope.json", abspath=str(tmp_path / "nope.json")))
    bp.write_text(json.dumps(c), encoding="utf-8")
    assert "does not exist here" in _joined(base.judge(bp, _argv("B1"), work=tmp_path / "wm")[0])
    # a profile-keyed check outside map / box has no reuse kind at all
    assert LG.REUSE_KINDS == {"G-MAP-OVERFIT": "map", "G-BOX-OVERFIT": "box"}
    assert LG.judge_overfit_reuse(RIG, "G-BOX-OVERFIT-B3", str(bp), COMMIT, argv_sha="0" * 64, argv=_argv("B1"), tree=TREE, workdir=tmp_path / "wn")[0]


def test_e_the_job_reads_a_reuse_binding_through_the_same_record_option_and_a_plain_record_is_unchanged(tmp_path):
    base = Base(tmp_path, "box")
    reasons, det, bp = base.bind(tmp_path, "B1")
    assert reasons == []
    clo = bp.parent / "box_closure.json"

    def ctx(argv, rec, out):
        return LG.Ctx(profile="refcv9-e1rig", tree=str(TREE), commit=COMMIT, argv=list(argv), out_dir=str(out), path_map=[], tree_sha256="0" * 64,
                      argv_sha256=LG.argv_sha256(argv), options={"box_overfit_record": str(rec), "box_overfit_closure": str(clo)})
    ev = LG._job_box_overfit(ctx(_argv("B1"), bp, tmp_path / "j1"))["G-BOX-OVERFIT"]
    assert ev["details"]["base_arm"] == "C0" and ev["details"]["this_arm"] == "B1"
    assert ev["details"]["identity"]["this"] == ev["details"]["identity"]["base"] and "gate:box-overfit-reuse-binding" in ev["inputs_read"]
    assert not [r for r in ev["reasons"] if "REUSE" in r and "binding" in r and "no tanitad" in r]
    ev2 = LG._job_box_overfit(ctx(_argv("C0"), base.rec, tmp_path / "j2"))["G-BOX-OVERFIT"]   # a plain run record: no reuse key anywhere
    assert "reuse_binding" not in ev2["details"] and "gate:box-overfit-reuse-binding" not in ev2["inputs_read"]


# =============================================================================== the bind script's dry run ==
@R2.needs_bash
@pytest.mark.parametrize("kind", ["box", "map"])
@pytest.mark.parametrize("arm", ["B1", "T1s"])
def test_bind_dry_reuse_accepts_the_aux_and_T1_arms_for_the_inherited_kinds(tmp_path, kind, arm):
    out, root = R2._bind(tmp_path, kind, arm, "--reuse-from", "C0")
    assert "ZZR9E1BIND-DRYREUSE-OKZZ" in out and "REFUSED" not in out and "harness invocation IDENTICAL to C0's" in out, out


@R2.needs_bash
@pytest.mark.parametrize("args,why", [
    (("box", "B3", "--reuse-from", "C0"), "own-head --r9-box-* lever"),                       # B3 is box-b3's, before reuse is even reached
    (("box", "M1", "--reuse-from", "C0"), "NOT identical to C0's"),                           # an argv that builds another head
    (("box", "C1", "--reuse-from", "C0"), "NOT identical to C0's"),                           # another seed
    (("box-b3", "B3", "--reuse-from", "C0"), "covers the kinds map and box only"),
    (("box", "C0", "--reuse-from", "C0"), "names the arm itself"),
    (("box", "B1", "--reuse-from", "NoSuchArm"), "no base rig argv"),
    (("box", "B1", "--reuse-from", "C0;rm"), "must be the NAME of a base arm")])
def test_bind_dry_reuse_refuses_each_red_arm_with_its_reason(tmp_path, args, why):
    out, root = R2._bind(tmp_path, *args)
    assert "ZZR9E1BIND-REFUSEDZZ" in out and why in out and "DRYREUSE" not in out, out


# ======================================= the Thor gate scripts pick a reuse binding in place of a run record ==
GATE3 = ROOT / "ops" / "r9e1_gate3"
THOR_GATE, REC_EXPLORE = GATE3 / "r9gate3_thor_gate.sh", GATE3 / "r9gate3_rec_explore.sh"
PICK_BEGIN, PICK_END = "# >>> E1-REUSE record pick", "# <<< E1-REUSE record pick"


def _pick_block(path: Path) -> str:
    src = path.read_text(encoding="utf-8")
    assert "\r" not in src and src.count(PICK_BEGIN) == 1 and src.count(PICK_END) == 1, path
    return src[src.index(PICK_BEGIN):src.index(PICK_END) + len(PICK_END)]


def test_gate3_both_scripts_carry_the_SAME_pick_block_once_before_the_gate_reads_a_record():
    assert _pick_block(THOR_GATE) == _pick_block(REC_EXPLORE)
    for path, first_use in ((THOR_GATE, 'for f in $MREC $MCLO $BREC $BCLO'), (REC_EXPLORE, "launch_gate.py run")):
        src = path.read_text(encoding="utf-8")
        i, defs = src.index(PICK_BEGIN), [src.index(x) for x in ("MREC=$OV/run_map/g_map_overfit.json", "BREC=$OV/run_box/gbo_binding.json", "refuse() {")]
        assert all(d < i for d in defs) and i < src.index(first_use), path.name                   # picked AFTER the defaults, BEFORE any read
        banded = src.index("MREC=$OV/run_map-banded/g_map_overfit.json")
        assert banded < i                                                                          # a banded arm's default is set before the pick
        assert src.count("$MREC") >= 2 and "--map-overfit-record $MREC" in src and "--box-overfit-record $BREC" in src   # the gate still reads the PICKED path


def _run_pick(tmp_path, *, map_run=False, map_reuse=False, box_run=False, box_reuse=False, banded=False, block=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    ov = tmp_path / "ovf"
    mrec = ov / ("run_map-banded/g_map_overfit.json" if banded else "run_map/g_map_overfit.json")
    brec, mru, bru = ov / "run_box/gbo_binding.json", ov / "run_map/reuse_binding.json", ov / "run_box/reuse_binding.json"
    for flag, p in ((map_run, mrec), (box_run, brec), (map_reuse, mru), (box_reuse, bru)):
        if flag:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("{}", encoding="utf-8")
    script = tmp_path / "pick.sh"
    script.write_text("set -u\nOV=" + ov.as_posix() + "\nMREC=" + mrec.as_posix() + "\nBREC=" + brec.as_posix() + "\n"
                      'refuse() { echo "REFUSED: $*"; exit 10; }\n' + (block or _pick_block(THOR_GATE)) + '\necho "PICK-MREC=$MREC"\necho "PICK-BREC=$BREC"\n', encoding="utf-8")
    r = R2.subprocess.run([R2.BASH, script.as_posix()], capture_output=True, text=True, timeout=120)
    out = r.stdout.replace("\r", "")
    return r.returncode, out, {"map_run": mrec.as_posix(), "box_run": brec.as_posix(), "map_reuse": mru.as_posix(), "box_reuse": bru.as_posix()}


@R2.needs_bash
def test_gate3_pick_reuse_only_picks_the_reuse_binding_run_only_picks_the_run_record(tmp_path):
    rc, out, p = _run_pick(tmp_path / "a", map_reuse=True, box_reuse=True)                       # reuse only (both kinds)
    assert rc == 0 and f"PICK-MREC={p['map_reuse']}" in out and f"PICK-BREC={p['box_reuse']}" in out, out
    rc, out, p = _run_pick(tmp_path / "b", map_run=True, box_run=True)                           # run record only: UNCHANGED from before
    assert rc == 0 and f"PICK-MREC={p['map_run']}" in out and f"PICK-BREC={p['box_run']}" in out, out
    rc, out, p = _run_pick(tmp_path / "c", map_reuse=True, box_run=True)                         # per kind: map reuses, box ran
    assert rc == 0 and f"PICK-MREC={p['map_reuse']}" in out and f"PICK-BREC={p['box_run']}" in out, out
    rc, out, p = _run_pick(tmp_path / "d", map_run=True, box_reuse=True)
    assert rc == 0 and f"PICK-MREC={p['map_run']}" in out and f"PICK-BREC={p['box_reuse']}" in out, out
    rc, out, p = _run_pick(tmp_path / "e", map_run=True, box_run=True, banded=True)              # a banded arm keeps its run_map-banded default
    assert rc == 0 and "run_map-banded/g_map_overfit.json" in out.split("PICK-MREC=")[1].split("\n")[0], out


@R2.needs_bash
@pytest.mark.parametrize("kw,kind", [({"map_run": True, "map_reuse": True, "box_run": True}, "map"), ({"map_run": True, "box_run": True, "box_reuse": True}, "box"),
                                     ({"map_run": True, "map_reuse": True, "box_run": True, "box_reuse": True}, "map")])
def test_gate3_pick_REFUSES_when_a_run_record_AND_a_reuse_binding_both_exist(tmp_path, kw, kind):
    rc, out, _ = _run_pick(tmp_path, **kw)
    assert rc == 10 and out.startswith("REFUSED: both a run record") and f"for the {kind} kind: ambiguous" in out and "PICK-" not in out, out


@R2.needs_bash
@pytest.mark.parametrize("kw,kind", [({}, "map"), ({"map_run": True}, "box"), ({"box_reuse": True}, "map"), ({"map_reuse": True}, "box")])
def test_gate3_pick_REFUSES_when_neither_exists_for_a_kind(tmp_path, kw, kind):
    rc, out, _ = _run_pick(tmp_path, **kw)
    assert rc == 10 and out.startswith("REFUSED: neither a run record") and f"for the {kind} kind" in out and "PICK-" not in out, out


@R2.needs_bash
def test_gate3_pick_MUTATIONS_each_removed_clause_makes_a_scenario_go_wrong(tmp_path):
    good = _pick_block(THOR_GATE)
    no_ambiguity = good.replace('if [ -e "$_ru" ] && [ -e "$_run" ]; then refuse', 'if false; then refuse')
    rc, out, _ = _run_pick(tmp_path / "m1", map_run=True, map_reuse=True, box_run=True, block=no_ambiguity)
    assert rc == 0                                                                               # the mutant silently prefers one: the REFUSAL test above would go RED
    no_neither = good.replace('elif [ ! -e "$_run" ]; then refuse', 'elif false; then refuse')
    rc, out, _ = _run_pick(tmp_path / "m2", box_run=True, block=no_neither)
    assert rc == 0                                                                               # the mutant lets a missing record through to the gate
    no_pick = good.replace('if [ "$_k" = map ]; then MREC="$_ru"; else BREC="$_ru"; fi', ':')
    rc, out, p = _run_pick(tmp_path / "m3", map_reuse=True, box_reuse=True, block=no_pick)
    assert f"PICK-MREC={p['map_reuse']}" not in out                                              # the mutant never picks the binding: the reuse-only test goes RED
    assert _run_pick(tmp_path / "g", map_reuse=True, box_reuse=True, block=good)[0] == 0         # control: the shipped block is the one that passes
