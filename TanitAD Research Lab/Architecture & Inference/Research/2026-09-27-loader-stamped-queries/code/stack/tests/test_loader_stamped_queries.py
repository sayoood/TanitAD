"""I3 (refcv7 box head, SPEC_REFCV7 A9 R4): a PRE-A9 record rebuilds at ITS OWN query count.

WHAT MOVED. ``agent_slots.N_QUERIES_DEFAULT`` went 100 -> 300, and with it BOTH query counts a recorded
refc run is rebuilt at: the trainer's ``--agent-queries`` default (the agent head) and
``PerceptionBranchConfig.n_queries`` (the box head). A record whose argv never named ``--agent-queries``
(refcv6-r101-s0 included) would be rebuilt at 300 and die on its strict load. The rule that fixes it lives
in ``taniteval/tools/refcv3_arm.py`` (the agent head via ``refc_v3_train.agent_queries_as_trained`` in
``rebuild_config``; the box head via the ``refcv6_perception.n_queries`` stamp in
``rebuild_perception_branch``).

WHAT I3 NAMED, AND WHAT THE SOURCE SAYS (read at 37086c3, 2026-09-27):

* ``stack/experiments/alpasim-gsplat/closedloop_drive.py`` does NOT re-parse argv itself.
  ``RefCV3Policy.__init__`` calls ``refcv3_arm.load_model``, so it rebuilds through the rule above.
  Pinned here BEHAVIOURALLY, through the policy's own constructor.
* ``taniteval/tools/seam_probe.py`` rebuilds NO model at all. Its only ``build_parser().parse_args`` is
  its OWN CLI, and it reads emission dumps. There is nothing for the rule to act on, and a
  mutation-tested census pins that.
* ``taniteval/taniteval/bench/cli.py`` is pinned in ``taniteval/tests/test_bench_stamped_queries.py``,
  next to the bench's own tests.

Every expectation is a LITERAL (100, 300). The deliberate-regression arm rebuilds the tree as it was
BEFORE the rule. Every ``refc_v3_train.py`` loaded under it lacks ``agent_queries_as_trained``, so the
recorded argv is rebuilt at the parser default. It must REFUSE, through the same entry points that
pass above.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

ROOT = Path(__file__).resolve().parents[2]                           # <repo>
TOOLS = ROOT / "taniteval" / "tools"
EXP = ROOT / "stack" / "experiments" / "alpasim-gsplat"
for _p in (ROOT / "stack", ROOT / "stack" / "scripts", ROOT / "taniteval", TOOLS, EXP):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

for _f in (TOOLS / "refcv3_arm.py", TOOLS / "seam_probe.py", EXP / "closedloop_drive.py"):
    if not _f.is_file():
        pytest.skip(f"NO_TREE: {_f} is not in this checkout", allow_module_level=True)

from tanitad.models import refcv6_perception_branch as PB           # noqa: E402
from tanitad.refs import refc_v3 as v3                               # noqa: E402

#: The smallest argv that builds BOTH slot heads through the trainer's own parser and pin. The agent head
#: is ``--agents head``. The box head is ``--w-box3d > 0``, which needs ``--trunk timm`` and ``--agent-join``.
#: The argv is SILENT on ``--agent-queries``, as refcv6-r101-s0's is, so the parser default decides. The
#: geometry (9 channels, 256x640, the v3 horizons) is the one ``RefCV3Policy`` asserts before it plans.
ARGV = ["--arm", "hier", "--smoke", "--trunk", "timm", "--trunk-name", "resnet18.a1_in1k",
        "--no-trunk-pretrained", "--trunk-in-channels", "9", "--image-hw", "256", "640",
        "--agents", "head", "--agent-join", "join.jsonl", "--w-agent", "1.0", "--w-box3d", "1.0",
        "--out", "x"]
#: Stamp fields that A9 (R1-R3), A14 (query select), A6 (bev source) and A7 (planner crop) added. They are
#: absent from every pre-A9 record, so a faithful pre-A9 record omits them.
A9_AGENT_KEYS = ("presence_loss", "presence_prior", "deep_supervision", "vis1")
A9_PERCEPTION_KEYS = A9_AGENT_KEYS + ("query_select", "bev_source", "planner_crop_m")

_T = None


def _trainer():
    """The trainer by path, under a name no loader uses (it only BUILDS the records here)."""
    global _T
    if _T is None:
        spec = importlib.util.spec_from_file_location(
            "refc_v3_train_for_i3_records", str(ROOT / "stack" / "scripts" / "refc_v3_train.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _T = mod
    return _T


def write_record(run: Path, *, queries: int, pre_a9: bool, param_breakdown: bool = True) -> Path:
    """One run directory (``ckpt.pt`` + ``config.json``), built the way ``refc_v3_train.train`` builds
    it, when the trainer's query default was ``queries``."""
    T = _trainer()
    args = T.build_parser().parse_args(ARGV)
    args.agent_queries = queries                    # the parser default the run was trained under
    cfg = T._pin_trainer_cfg(v3.refc_v3_smoke_config(True), args)
    torch.manual_seed(0)
    model = v3.RefCV3Model(cfg)
    pcfg = PB.PerceptionBranchConfig(w_map=0.0, w_box3d=1.0, n_queries=queries)
    model._perception = PB.build_perception_branch(model, pcfg)
    agents, perc = dict(cfg.core.agents.as_dict()), dict(pcfg.as_dict())
    if pre_a9:
        for k in A9_AGENT_KEYS:
            agents.pop(k, None)
        for k in A9_PERCEPTION_KEYS:
            perc.pop(k, None)
        agents["n_queries_default_upstream"] = queries
    perc["branch_params"] = model._perception.param_breakdown()
    config = {"argv": list(ARGV), "arm": "hier",
              "image_hw": list(cfg.core.encoder.image_hw()),
              "horizons": list(cfg.core.trajectory.horizons),
              "seams": {"agents": agents}, "refcv6_perception": perc}
    if param_breakdown:
        config["param_breakdown"] = {k: int(v) for k, v in v3.param_breakdown_v3(model).items()}
    run.mkdir(parents=True, exist_ok=True)
    torch.save({"step": 7, "model": model.state_dict(), "opt": {}}, run / "ckpt.pt")
    (run / "config.json").write_text(json.dumps(config, default=str), encoding="utf-8")
    return run / "ckpt.pt"


@pytest.fixture(scope="module")
def records(tmp_path_factory):
    root = tmp_path_factory.mktemp("i3_records")
    out = {"pre": write_record(root / "pre", queries=100, pre_a9=True),
           "pre_nopb": write_record(root / "pre_nopb", queries=100, pre_a9=True, param_breakdown=False),
           "post": write_record(root / "post", queries=300, pre_a9=False)}
    # the box head's half of the rule: the SAME pre-A9 weights, with a stamp that does not carry its count
    nobq = root / "pre_nobq"
    nobq.mkdir()
    shutil.copyfile(out["pre_nopb"], nobq / "ckpt.pt")
    c = json.loads((out["pre_nopb"].parent / "config.json").read_text(encoding="utf-8"))
    del c["refcv6_perception"]["n_queries"]
    (nobq / "config.json").write_text(json.dumps(c), encoding="utf-8")
    out["pre_nobq"] = nobq / "ckpt.pt"
    return out


# --------------------------------------------------------------------------------------------------------- #
# the entry points: each returns (model, provenance) through the loader's OWN call path                     #
# --------------------------------------------------------------------------------------------------------- #
def _arm():
    """``refcv3_arm`` by path: the rule every named loader's model path runs through."""
    spec = importlib.util.spec_from_file_location("refcv3_arm_i3_under_test", str(TOOLS / "refcv3_arm.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["refcv3_arm_i3_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def via_refcv3_arm(ck: Path):
    model, _cfg, _targs, prov = _arm().load_model(str(ck), None, "cpu", False)
    return model, prov


def via_closedloop(ck: Path):
    import closedloop_drive as cd
    pol = cd.RefCV3Policy(str(ck), device="cpu")
    return pol.model, pol.prov


ENTRIES = {"refcv3_arm.load_model": via_refcv3_arm,
           "closedloop_drive.RefCV3Policy": via_closedloop}


def _counts(model) -> tuple:
    """(agent head, box head) query counts as BUILT: the loaded query tables, not a config field."""
    return (int(model.core.agent_head.queries.shape[1]),
            int(model._perception.box_dec.queries.shape[1]))


def _strict(prov) -> dict:
    sd = prov["state_dict_load"]
    return {k: list(sd[k]) for k in ("missing_keys", "unexpected_keys", "tolerated_inert_buffers")}


STRICT = {"missing_keys": [], "unexpected_keys": [], "tolerated_inert_buffers": []}


@pytest.mark.parametrize("entry", list(ENTRIES))
def test_a_PRE_A9_record_rebuilds_at_100_and_loads_STRICT(records, entry):
    model, prov = ENTRIES[entry](records["pre"])
    assert _strict(prov) == STRICT
    assert _counts(model) == (100, 100)
    assert "agent_queries -> 100 (config.json seams.agents.queries (argv silent))" in prov["rebuilt_from"]


@pytest.mark.parametrize("entry", list(ENTRIES))
def test_a_POST_A9_record_rebuilds_at_300_and_loads_STRICT(records, entry):
    model, prov = ENTRIES[entry](records["post"])
    assert _strict(prov) == STRICT
    assert _counts(model) == (300, 300)
    assert "agent_queries ->" not in prov["rebuilt_from"]          # the stamp IS the default: no override


# --------------------------------------------------------------------------------------------------------- #
# ⛔ THE DELIBERATE-REGRESSION ARM: the same entry points on the tree as it was before the rule              #
# --------------------------------------------------------------------------------------------------------- #
#: every sys.modules name a stripped trainer (or an arm copy holding one) can be registered under here
_ARM_MODULES = ("refc_v3_train_for_arm", "refcv3_arm", "refcv3_arm_i3_under_test")


@pytest.fixture
def pre_i3_tree(monkeypatch):
    """Every ``refc_v3_train.py`` loaded from here on lacks ``agent_queries_as_trained``. That is the tree
    before the rule: ``rebuild_config``'s ``getattr(tr, "agent_queries_as_trained", None)`` reads None,
    and the recorded argv is rebuilt at the PARSER DEFAULT.

    Each arm-module copy that is already loaded is made to re-resolve its trainer through the hook. All of
    it is restored afterwards, so no stripped trainer outlives the test.

    ⚠️ The copies are selected by ``__file__`` read from the module ``__dict__``, never by ``hasattr``. On a
    lazy namespace module like ``torch.ops``, ``hasattr(mod, "_TRAINER")`` is True: the lookup CREATES the
    attribute. MEASURED: a ``hasattr`` selector patched ``torch.ops``.
    """
    real = importlib.util.spec_from_file_location

    def hooked(name, location=None, *a, **kw):
        spec = real(name, location, *a, **kw)
        if spec is not None and location is not None and Path(str(location)).name == "refc_v3_train.py":
            run = spec.loader.exec_module

            def exec_module(mod, _run=run):
                _run(mod)
                mod.__dict__.pop("agent_queries_as_trained", None)
            spec.loader.exec_module = exec_module
        return spec

    saved = {k: sys.modules.get(k) for k in _ARM_MODULES}
    monkeypatch.setattr(importlib.util, "spec_from_file_location", hooked)
    for mod in list(sys.modules.values()):
        d = getattr(mod, "__dict__", None) or {}
        if str(d.get("__file__") or "").endswith("refcv3_arm.py") and "_TRAINER" in d:
            monkeypatch.setattr(mod, "_TRAINER", None)
    yield
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


@pytest.mark.parametrize("entry", list(ENTRIES))
def test_RED_ARM_without_the_rule_the_pre_A9_record_is_REFUSED_by_the_strict_load(records, pre_i3_tree, entry):
    with pytest.raises(SystemExit) as ei:
        ENTRIES[entry](records["pre_nopb"])
    msg = str(ei.value)
    assert "DISAGREE ON SHAPE" in msg
    assert "size mismatch for core.agent_head.queries" in msg
    assert "torch.Size([1, 100," in msg and "torch.Size([1, 300," in msg


@pytest.mark.parametrize("entry", list(ENTRIES))
def test_RED_ARM_with_param_breakdown_stamped_the_cross_check_refuses_first(records, pre_i3_tree, entry):
    with pytest.raises(SystemExit) as ei:
        ENTRIES[entry](records["pre"])
    assert "config.json CONTRADICTS the rebuilt model" in str(ei.value)
    assert "param_breakdown" in str(ei.value)


@pytest.mark.parametrize("entry", list(ENTRIES))
def test_RED_ARM_a_box_stamp_without_n_queries_rebuilds_300_and_is_REFUSED(records, entry):
    """The box head's half: ``rebuild_perception_branch`` reads the count from the STAMP. Without it the
    branch is rebuilt at the dataclass default and the strict load refuses. The agent head (stamped)
    is not the mismatch."""
    with pytest.raises(SystemExit) as ei:
        ENTRIES[entry](records["pre_nobq"])
    msg = str(ei.value)
    assert "DISAGREE ON SHAPE" in msg and "size mismatch for _perception.box_dec.queries" in msg
    assert "core.agent_head" not in msg


# --------------------------------------------------------------------------------------------------------- #
# seam_probe.py: there is NO model rebuild for the rule to act on                                           #
# --------------------------------------------------------------------------------------------------------- #
REBUILD_MODULES = frozenset({"refc_v3_train", "refcv3_arm", "refcv6_loader", "refc_v3"})
REBUILD_CALLS = frozenset({"load_state_dict", "_pin_trainer_cfg", "load_model", "build_model",
                           "RefCV3Model", "load_refcv4b"})
REBUILD_FILES = ("refc_v3_train.py", "refcv3_arm.py", "refcv6_loader.py")


def model_rebuild_sites(src: str) -> set:
    """Every place a module could rebuild a REF-C model from a record. That means importing the trainer, an
    arm loader or the model module, calling something that builds or loads one, or naming one of their files
    as a path (a positional call argument or a ``/`` operand, never help text)."""
    out = set()
    for node in ast.walk(ast.parse(src)):
        operands = []
        if isinstance(node, ast.Import):
            out |= {f"import {a.name}" for a in node.names if a.name.split(".")[-1] in REBUILD_MODULES}
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod.split(".")[-1] in REBUILD_MODULES or any(a.name in REBUILD_MODULES for a in node.names):
                out.add(f"from {mod} import")
        elif isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if name in REBUILD_CALLS:
                out.add(f"call {name}")
            operands = list(node.args)
        elif isinstance(node, ast.BinOp):
            operands = [node.left, node.right]
        for c in operands:
            if isinstance(c, ast.Constant) and isinstance(c.value, str) and c.value.endswith(REBUILD_FILES):
                out.add(f"path {c.value}")
    return out


#: one mutation per rebuild form the census must see
MUTATIONS = {
    "import + load_model": ("\nimport refcv3_arm\n\n\ndef _m(ck):\n    return refcv3_arm.load_model(ck)\n",
                            {"import refcv3_arm", "call load_model"}),
    "trainer by path": ("\nimport importlib.util\nimport os\n_s = importlib.util.spec_from_file_location("
                        "'t', os.path.join('stack', 'scripts', 'refc_v3_train.py'))\n",
                        {"path refc_v3_train.py"}),
    "raw state-dict load": ("\n\ndef _m2(model, sd):\n    model.load_state_dict(sd)\n",
                            {"call load_state_dict"}),
}


def test_seam_probe_rebuilds_NO_model_and_the_census_would_see_one():
    src = (TOOLS / "seam_probe.py").read_text(encoding="utf-8")
    # control: the file really was READ (an unreadable file must not pass as an empty census)
    assert len(src) > 20000 and "def run_probe" in src and "def load_dump" in src
    assert model_rebuild_sites(src) == set(), (
        "seam_probe.py now rebuilds a model. If it re-parses a RECORDED refc argv, rebuild through "
        "refcv3_arm.load_model (the stamped-query rule), never through the parser default")
    for name, (mut, want) in MUTATIONS.items():                       # RED ARMS
        assert model_rebuild_sites(src + mut) == want, name
