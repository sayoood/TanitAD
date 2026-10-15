"""The refcv8 launch argv carries ITS OWN header (2026-10-15).

MEASURED after the refcv8-r101-s0 launch gate: `refcv8_assemble_argv.py` wrote `dict(doc, argv=...)`, i.e. the BASE
smoke file's header with the argv swapped in. The committed launch argv (48,852 steps, `--w-r8-listwise 0.0`, out
`refcv8-r101-s0`) therefore said `"arm": "refcv8-wpb-smoke"`, "A SMOKE, not a launch", and listed
`--w-r8-listwise 1.0` under `changes_vs_refcv7`. The gate never read the header (it binds the argv list, argv_sha256),
so nothing failed. The header was simply false. The assembler now writes its own header (`launch_doc`). The
committed file's header was rewritten with its argv list byte-identical.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import refcv8_assemble_argv as A  # noqa: E402

RUNS = ROOT / "ops" / "runs.d"
#: the launch gate's binding for refcv8-r101-s0 (ctx.json argv_sha256, commit fb47a583763f) -- a LITERAL
LAUNCH_ARGV_SHA = "188aa439258d28e712e9a1b4a8ebda982d5f068b4515022f54ba1172e18fd956"


def _doc(name: str) -> dict:
    return json.loads((RUNS / name).read_text(encoding="utf-8"))


def _sha(argv: list[str]) -> str:
    return hashlib.sha256(json.dumps(argv, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def test_the_committed_launch_argv_is_the_launched_one_and_its_header_says_so():
    d = _doc("refcv8-r101-s0.argv.json")
    assert _sha(d["argv"]) == LAUNCH_ARGV_SHA                      # the header fix did not touch the argv
    assert d["arm"] == "refcv8-r101-s0" and d["stage"] == "launch"
    assert "changes_vs_refcv7" not in d and "SMOKE" not in d["what"].upper()
    assert d["derived_from"] == {"file": "stack/ops/runs.d/refcv8-wpb-smoke.argv.json", "arm": "refcv8-wpb-smoke",
                                 "n_tokens": 200}


def test_the_set_lines_replayed_over_the_base_ARE_the_launch_argv():
    """Independent of the assembler's run: the base FILE plus the SET lines from the launch assembly's own log must
    reproduce the argv token for token. Every override is therefore named, and every other token is the base's."""
    d = _doc("refcv8-r101-s0.argv.json")
    ps = A.pairs(_doc("refcv8-wpb-smoke.argv.json")["argv"])
    assert len(d["set_lines"]) == 29
    for ln in d["set_lines"]:
        flag, *vals = ln.split("  <- ")[0].split()
        ps = A.set_flag(ps, flag, vals)
    assert A.flat(ps) == d["argv"]
    # control: dropping any one SET line breaks the reproduction (the check can fail)
    ps2 = A.pairs(_doc("refcv8-wpb-smoke.argv.json")["argv"])
    for ln in d["set_lines"][1:]:
        flag, *vals = ln.split("  <- ")[0].split()
        ps2 = A.set_flag(ps2, flag, vals)
    assert A.flat(ps2) != d["argv"]


def test_launch_doc_writes_its_own_header_and_carries_nothing_descriptive_from_the_base():
    base = {"schema": "tanitad.launch_argv/1", "arm": "smoke", "what": "A SMOKE, not a launch",
            "changes_vs_refcv7": [{"flag": "--w-r8-listwise", "value": ["1.0"]}], "argv": ["--a", "1"]}
    out = A.launch_doc(base, ["--a", "2"], ["--a 2  <- why"], base_path="b.json", run_out="/r/runs/arm-x/",
                       stage="launch")
    assert out["arm"] == "arm-x" and out["stage"] == "launch" and out["argv"] == ["--a", "2"]
    assert "changes_vs_refcv7" not in out and "SMOKE" not in out["what"].upper()
    assert out["derived_from"] == {"file": "b.json", "arm": "smoke", "n_tokens": 2}
    assert out["set_lines"] == ["--a 2  <- why"]


def test_main_writes_through_launch_doc():
    """Mutation: restoring `dict(doc, argv=flat(ps))` turns this RED."""
    src = inspect.getsource(A.main)
    assert "launch_doc(doc, flat(ps), why" in src and "dict(doc, argv=" not in src
