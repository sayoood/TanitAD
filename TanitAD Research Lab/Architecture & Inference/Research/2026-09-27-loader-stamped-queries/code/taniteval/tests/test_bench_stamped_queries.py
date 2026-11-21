"""I3 (refcv7 box head, SPEC_REFCV7 A9 R4) for the BENCH: a PRE-A9 record rebuilds at ITS OWN query count.

``agent_slots.N_QUERIES_DEFAULT`` went 100 -> 300. That moved the trainer's ``--agent-queries`` default (the
agent head) and ``PerceptionBranchConfig.n_queries`` (the box head). A recorded argv that never named
``--agent-queries`` (refcv6-r101-s0 included) would be rebuilt at 300 and die on its strict load. The rule
lives in ``taniteval/tools/refcv3_arm.py`` (``rebuild_config`` + ``rebuild_perception_branch``).

WHAT THE SOURCE SAYS (read at 37086c3, 2026-09-27). ``taniteval/taniteval/bench/cli.py`` rebuilds no model:
its ``build_parser().parse_args`` is its OWN CLI. Every model the suite loads in-process comes through ONE
function, ``bench/navsim/bridge.py::load_refcv4b``, which runs ``refcv3_arm.load_model``. The callers are
``navsim/model_arms.py`` (navsim_v2) and ``adapters/nuscenes_planning.py::refc_forward`` (nuscenes_ol).
``internal_t1`` runs ``refcv3_arm.py`` as a subprocess; ``test_bench_suite_internal_t1.py`` pins that tool.

So this file pins the bench's model path BEHAVIOURALLY through ``load_refcv4b``. A pre-A9 record rebuilds at
100 and loads strict, and a post-A9 record rebuilds at 300. The deliberate-regression arm is the tree before
the rule (every ``refc_v3_train.py`` loaded lacks ``agent_queries_as_trained``), and it must REFUSE. A
mutation-tested census pins that ``cli.py`` itself rebuilds nothing. Every expectation is a LITERAL.
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
for _p in (ROOT / "stack", ROOT / "stack" / "scripts", ROOT / "taniteval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
if not (ROOT / "taniteval" / "tools" / "refcv3_arm.py").is_file():
    pytest.skip("NO_TREE: taniteval/tools/refcv3_arm.py is not in this checkout", allow_module_level=True)

from taniteval.bench.navsim import bridge as B                       # noqa: E402
from tanitad.models import refcv6_perception_branch as PB           # noqa: E402
from tanitad.refs import refc_v3 as v3                               # noqa: E402

CLI = ROOT / "taniteval" / "taniteval" / "bench" / "cli.py"

#: The smallest argv that builds BOTH slot heads through the trainer's own parser and pin. It is SILENT on
#: ``--agent-queries``, as refcv6-r101-s0's is. Its v3 horizons are the ones ``load_refcv4b`` checks
#: against ``KNOT_T_S``.
ARGV = ["--arm", "hier", "--smoke", "--trunk", "timm", "--trunk-name", "resnet18.a1_in1k",
        "--no-trunk-pretrained", "--trunk-in-channels", "9", "--image-hw", "256", "640",
        "--agents", "head", "--agent-join", "join.jsonl", "--w-agent", "1.0", "--w-box3d", "1.0",
        "--out", "x"]
#: stamp fields added by A9 (R1-R3), A14, A6 and A7. A faithful pre-A9 record omits them.
A9_AGENT_KEYS = ("presence_loss", "presence_prior", "deep_supervision", "vis1")
A9_PERCEPTION_KEYS = A9_AGENT_KEYS + ("query_select", "bev_source", "planner_crop_m")

_T = None


def _trainer():
    global _T
    if _T is None:
        spec = importlib.util.spec_from_file_location(
            "refc_v3_train_for_i3_bench_records", str(ROOT / "stack" / "scripts" / "refc_v3_train.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _T = mod
    return _T


def write_record(run: Path, *, queries: int, pre_a9: bool, param_breakdown: bool = True) -> Path:
    """One run directory (``ckpt.pt`` + ``config.json``), built the way ``refc_v3_train.train`` builds
    it, when the trainer's query default was ``queries``."""
    T = _trainer()
    args = T.build_parser().parse_args(ARGV)
    args.agent_queries = queries
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
    root = tmp_path_factory.mktemp("i3_bench_records")
    out = {"pre": write_record(root / "pre", queries=100, pre_a9=True),
           "pre_nopb": write_record(root / "pre_nopb", queries=100, pre_a9=True, param_breakdown=False),
           "post": write_record(root / "post", queries=300, pre_a9=False)}
    nobq = root / "pre_nobq"
    nobq.mkdir()
    shutil.copyfile(out["pre_nopb"], nobq / "ckpt.pt")
    c = json.loads((out["pre_nopb"].parent / "config.json").read_text(encoding="utf-8"))
    del c["refcv6_perception"]["n_queries"]
    (nobq / "config.json").write_text(json.dumps(c), encoding="utf-8")
    out["pre_nobq"] = nobq / "ckpt.pt"
    return out


def via_bench(ck: Path):
    """The bench's ONE in-process model loader."""
    model, _cfg, _targs, prov, _mod = B.load_refcv4b(str(ck))
    return model, prov


def _counts(model) -> tuple:
    return (int(model.core.agent_head.queries.shape[1]),
            int(model._perception.box_dec.queries.shape[1]))


def _strict(prov) -> dict:
    sd = prov["state_dict_load"]
    return {k: list(sd[k]) for k in ("missing_keys", "unexpected_keys", "tolerated_inert_buffers")}


STRICT = {"missing_keys": [], "unexpected_keys": [], "tolerated_inert_buffers": []}


def test_the_bench_loader_rebuilds_a_PRE_A9_record_at_100_and_loads_STRICT(records):
    model, prov = via_bench(records["pre"])
    assert _strict(prov) == STRICT
    assert _counts(model) == (100, 100)
    assert "agent_queries -> 100 (config.json seams.agents.queries (argv silent))" in prov["rebuilt_from"]


def test_the_bench_loader_rebuilds_a_POST_A9_record_at_300_and_loads_STRICT(records):
    model, prov = via_bench(records["post"])
    assert _strict(prov) == STRICT
    assert _counts(model) == (300, 300)
    assert "agent_queries ->" not in prov["rebuilt_from"]


# --------------------------------------------------------------------------------------------------------- #
# ⛔ THE DELIBERATE-REGRESSION ARM: the same loader on the tree as it was before the rule                    #
# --------------------------------------------------------------------------------------------------------- #
_ARM_MODULES = ("refc_v3_train_for_arm", "refcv3_arm_for_navsim")


@pytest.fixture
def pre_i3_tree(monkeypatch):
    """Every ``refc_v3_train.py`` loaded from here on lacks ``agent_queries_as_trained``, so the recorded
    argv is rebuilt at the PARSER DEFAULT. ``load_refcv4b`` loads ``refcv3_arm`` (and so the trainer)
    afresh on every call, so the hook reaches it. Everything is restored afterwards."""
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
    yield
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


def test_RED_ARM_without_the_rule_the_bench_loader_REFUSES_the_pre_A9_record(records, pre_i3_tree):
    with pytest.raises(SystemExit) as ei:
        via_bench(records["pre_nopb"])
    msg = str(ei.value)
    assert "DISAGREE ON SHAPE" in msg
    assert "size mismatch for core.agent_head.queries" in msg
    assert "torch.Size([1, 100," in msg and "torch.Size([1, 300," in msg


def test_RED_ARM_with_param_breakdown_stamped_the_cross_check_refuses_first(records, pre_i3_tree):
    with pytest.raises(SystemExit) as ei:
        via_bench(records["pre"])
    assert "config.json CONTRADICTS the rebuilt model" in str(ei.value)
    assert "param_breakdown" in str(ei.value)


def test_RED_ARM_a_box_stamp_without_n_queries_rebuilds_300_and_is_REFUSED(records):
    with pytest.raises(SystemExit) as ei:
        via_bench(records["pre_nobq"])
    msg = str(ei.value)
    assert "DISAGREE ON SHAPE" in msg and "size mismatch for _perception.box_dec.queries" in msg
    assert "core.agent_head" not in msg


# --------------------------------------------------------------------------------------------------------- #
# bench/cli.py: it rebuilds NO model itself                                                                 #
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


MUTATIONS = {
    "import + load_model": ("\nimport refcv3_arm\n\n\ndef _m(ck):\n    return refcv3_arm.load_model(ck)\n",
                            {"import refcv3_arm", "call load_model"}),
    "trainer by path": ("\nimport importlib.util\nimport os\n_s = importlib.util.spec_from_file_location("
                        "'t', os.path.join('stack', 'scripts', 'refc_v3_train.py'))\n",
                        {"path refc_v3_train.py"}),
    "bridge call": ("\n\ndef _m3(ck):\n    from .navsim import bridge\n    return bridge.load_refcv4b(ck)\n",
                    {"call load_refcv4b"}),
}


def test_bench_cli_rebuilds_NO_model_and_the_census_would_see_one():
    src = CLI.read_text(encoding="utf-8")
    # control: the file really was READ, and its one refcv3_arm.py mention is help TEXT
    assert len(src) > 10000 and "def run_benchmark_cmd" in src and "refcv3_arm.py" in src
    assert model_rebuild_sites(src) == set(), (
        "bench/cli.py now rebuilds a model. If it re-parses a RECORDED refc argv, load it through "
        "bridge.load_refcv4b / refcv3_arm.load_model (the stamped-query rule), never the parser default")
    for name, (mut, want) in MUTATIONS.items():                       # RED ARMS
        assert model_rebuild_sites(src + mut) == want, name
