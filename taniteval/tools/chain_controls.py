#!/usr/bin/env python3
"""chain_controls.py -- a chain stage may only GATE on a control that some stage PRODUCES.

THE DEFECT THIS CLOSES (free-point navtest read, 2026-10-18, RESULT s5).  The chain's `checks`
stage gated on a control file `<work>/D3.json`.  The check was correct; its input was never
produced: `fp_smoke.py --swap-cams` wrote `<work>/smoke_swap/smoke_result.json`, the swap smoke was
not a stage of the chain, and a human `cp` was the only link between the two names.  The chain
therefore finished `ZZCHAIN_DONE INVALID` with D3 reported *missing* -- only at the FINAL read,
after every expensive stage had run.  Same class as "built, tested, and unreachable from its
caller" (CLAUDE.md): a mechanism that exists and is right but is not reached.

THE RULE.  A chain declares, per stage, what it PRODUCES and what it GATES ON.  Before any stage
runs, `require_producers(plan)` refuses (raises `ChainPlanError`) unless every gated control has:

  * a producer in a stage EARLIER than the gating stage (a producer that runs after its consumer is
    no producer), or
  * an explicit `external` declaration naming who makes it before the chain starts -- and, when a
    `work` directory is given, the file must already EXIST (nothing in the chain will make it), or
  * an explicit `arriving` declaration naming who delivers it WHILE the chain runs (a pod pull): no
    startup existence check, but `verify_gates_present(stage, work)` refuses at the stage that needs
    it if it never arrived.

Two more checks close the ways a declaration can still lie:

  * SOURCE-SILENT: a producer stage that names a `script` must have a script that mentions the
    control's file name.  (A declared producer whose code never names the file is the original bug
    with a label on it.  A stage with no `script` is an in-process action and is not source-checked;
    `verify_stage_outputs` covers it at run time.)
  * `verify_stage_outputs(stage, work)` after each stage: every declared output must exist and be
    non-empty, so "stage OK" can never mean "stage ran and wrote nothing".

AND A DRIFT TRIPWIRE.  A declaration is only as good as its upkeep: a developer who adds a new gate
to the checking script and not to the plan has re-created the bug.  `scan_reads` finds the
work-relative files a script READS and WRITES (AST: f-strings / os.path.join rooted at `work`,
one level of variable indirection) and `check_declared_reads` flags a read the plan does not
declare.  It is a LOWER BOUND -- it finds reads that exist, it cannot prove there are no others
(a path built from a loop variable comes back as UNRESOLVED, listed, never silently dropped).

HOW A CHAIN ADOPTS IT (a dozen lines):

    import chain_controls as CC
    plan = CC.ChainPlan(
        stages=(CC.Stage("picks",  produces=("picks_report.json", "points.npz"), script="fp_picks.py"),
                CC.Stage("checks", gates_on=("picks_report.json", "D3.json"),
                         produces=("gates_summary.json",), script="fp_checks.py")),
        external={"IC1.json": "fp_capture.py check-a9, run before the chain"})
    CC.require_producers(plan, work=W, script_root=HERE)        # BEFORE any stage, before --wait-for-pull
    ...
    for st in plan.stages:                                      # after each stage returns 0:
        CC.raise_on(CC.verify_stage_outputs(st, W), f"stage {st.name} wrote less than it declared")

or from a shell-driven chain:  `python chain_controls.py check plan.json --work W`  (exit 2 = refused).

Stdlib only; ASCII only (the dev box console is cp1252).  Tests:
`taniteval/tests/test_chain_controls.py` (known answers, controls, and a mutation per check).
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

__all__ = ["Stage", "ChainPlan", "ChainPlanError", "check_plan", "require_producers", "raise_on",
           "verify_stage_outputs", "verify_gates_present", "scan_reads", "check_declared_reads", "plan_from_dict", "main"]

WORK_NAMES = ("work", "w", "W", "wd", "work_dir")
_READ_FUNCS = {"load", "exists", "isfile", "getsize", "read_text", "read_bytes"}
_WRITE_FUNCS = {"save", "savez", "savez_compressed", "dump", "write_text", "write_bytes"}


class ChainPlanError(Exception):
    """The chain plan is not startable: at least one gated control has no producer."""


@dataclass(frozen=True)
class Stage:
    """One stage of a chain.  Paths are relative to the chain's work directory, '/' separated; '*' is a glob."""
    name: str
    produces: tuple = ()
    gates_on: tuple = ()
    script: str | None = None
    optional_reads: tuple = ()          # read by the script but absent-tolerant by design (declared, not gated)


@dataclass(frozen=True)
class ChainPlan:
    stages: tuple
    external: Mapping[str, str] = field(default_factory=dict)   # control -> who made it BEFORE the chain starts (must exist now)
    arriving: Mapping[str, str] = field(default_factory=dict)   # control -> who delivers it WHILE the chain runs (e.g. a pull)


def _covers(produced: str, wanted: str) -> bool:
    """Does a declared product `produced` (maybe a glob) cover the control `wanted` (maybe a glob)?"""
    return produced == wanted or fnmatch.fnmatchcase(wanted, produced)


def _literal_pieces(pattern: str) -> list:
    base = pattern.replace("\\", "/").rsplit("/", 1)[-1]
    return [p for p in base.split("*") if p]


def _script_text(script: str | None, script_root: str | None):
    """(text or None, problem or None)."""
    if not script:
        return None, None
    path = script if os.path.isabs(script) or not script_root else os.path.join(script_root, script)
    if not os.path.isfile(path):
        return None, f"script {script!r} not found at {path!r}; its declared products cannot be verified (fail closed)"
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read(), None


def _exists_in(work: str, pattern: str) -> bool:
    if "*" not in pattern and "?" not in pattern:
        return os.path.isfile(os.path.join(work, pattern))
    head = pattern.replace("\\", "/").rsplit("/", 1)
    folder = os.path.join(work, head[0]) if len(head) == 2 else work
    return os.path.isdir(folder) and any(fnmatch.fnmatchcase(n, head[-1]) for n in os.listdir(folder))


def check_plan(plan: ChainPlan, work: str | None = None, *, verify_source: bool = True,
               script_root: str | None = None) -> list:
    """Every reason this plan cannot start.  An empty list means every gated control has a producer."""
    problems = []
    names = [s.name for s in plan.stages]
    for n in sorted({n for n in names if names.count(n) > 1}):
        problems.append(f"DUPLICATE-STAGE: stage name {n!r} is declared more than once")
    if not plan.stages:
        problems.append("EMPTY-PLAN: no stages declared")
    for ctl, who in plan.external.items():
        if not str(who).strip():
            problems.append(f"EXTERNAL-UNNAMED: external control {ctl!r} names no producer")
        elif work is not None and not _exists_in(work, ctl):
            problems.append(f"EXTERNAL-ABSENT: control {ctl!r} is declared external ({who}) but does not exist in "
                            f"{work!r} -- nothing in this chain will make it")
    for ctl, who in plan.arriving.items():
        if not str(who).strip():
            problems.append(f"ARRIVING-UNNAMED: arriving control {ctl!r} names no producer")
    texts = {}
    for idx, st in enumerate(plan.stages):
        if not st.name:
            problems.append(f"UNNAMED-STAGE: stage #{idx} has no name")
        for ctl in st.gates_on:
            earlier = [(j, s) for j, s in enumerate(plan.stages[:idx]) if any(_covers(p, ctl) for p in s.produces)]
            if earlier:
                if verify_source:
                    for _j, s in earlier:
                        if s.name not in texts:
                            texts[s.name] = _script_text(s.script, script_root)
                        text, prob = texts[s.name]
                        if prob:
                            problems.append(f"SCRIPT-UNREADABLE: stage {s.name!r}: {prob}")
                        elif text is not None:
                            decl = [p for p in s.produces if _covers(p, ctl)]
                            for p in decl:
                                miss = [piece for piece in _literal_pieces(p) if piece not in text]
                                if miss:
                                    problems.append(
                                        f"SOURCE-SILENT: stage {s.name!r} declares it produces {p!r} (gated on by stage "
                                        f"{st.name!r}) but its script {s.script!r} never mentions {miss}: a declared "
                                        f"producer that does not name the file is the original bug with a label on it")
                continue
            if any(_covers(e, ctl) for e in plan.external) or any(_covers(e, ctl) for e in plan.arriving):
                continue
            later = [s.name for s in plan.stages[idx:] if any(_covers(p, ctl) for p in s.produces)]
            if later:
                problems.append(f"LATE-PRODUCER: stage {st.name!r} gates on {ctl!r}, which is produced only by "
                                f"{later[0]!r} -- at or after the stage that needs it")
            else:
                problems.append(f"NO-PRODUCER: stage {st.name!r} gates on {ctl!r} and NO stage produces it, and it is "
                                f"not declared external. It would be reported 'missing' only at the final read")
    # one stage's `produces` must not collide with another stage's: two writers of one control hide which one ran
    seen = {}
    for st in plan.stages:
        for p in st.produces:
            if p in seen and seen[p] != st.name:
                problems.append(f"DOUBLE-PRODUCER: {p!r} is produced by both {seen[p]!r} and {st.name!r}")
            seen.setdefault(p, st.name)
    return problems


def raise_on(problems: Sequence[str], headline: str = "chain plan refused") -> None:
    if problems:
        raise ChainPlanError(headline + ":\n  - " + "\n  - ".join(problems))


def require_producers(plan: ChainPlan, work: str | None = None, *, verify_source: bool = True,
                      script_root: str | None = None) -> dict:
    """Refuse at STARTUP unless every gated control has a producer.  Returns the ledger control -> producer."""
    raise_on(check_plan(plan, work, verify_source=verify_source, script_root=script_root),
             "chain plan refused at startup (a gated control has no producer)")
    ledger = {}
    for idx, st in enumerate(plan.stages):
        for ctl in st.gates_on:
            who = next((s.name for s in plan.stages[:idx] if any(_covers(p, ctl) for p in s.produces)), None)
            ledger[ctl] = who or next((("external: " if e in plan.external else "arriving: ") + v
                                       for e, v in list(plan.external.items()) + list(plan.arriving.items())
                                       if _covers(e, ctl)), "?")
    return ledger


def verify_stage_outputs(stage: Stage, work: str) -> list:
    """After a stage returned 0: every declared product exists and is non-empty (a glob needs >= 1 match)."""
    out = []
    for p in stage.produces:
        if "*" in p or "?" in p:
            if not _exists_in(work, p):
                out.append(f"stage {stage.name!r} declared {p!r} but no file matches in {work!r}")
        else:
            fp = os.path.join(work, p)
            if not os.path.isfile(fp):
                out.append(f"stage {stage.name!r} declared {p!r} but it does not exist in {work!r}")
            elif os.path.getsize(fp) == 0:
                out.append(f"stage {stage.name!r} declared {p!r} but it is EMPTY")
    return out


def verify_gates_present(stage: Stage, work: str) -> list:
    """Just BEFORE a stage runs: every control it gates on exists now.  Catches an `arriving` control that never
    arrived (a pull that stopped) at the stage that needs it, not at the final read."""
    return [f"stage {stage.name!r} gates on {c!r} but it is not in {work!r} yet" for c in stage.gates_on
            if not _exists_in(work, c)]


# --------------------------------------------------------------------------------------------------
# drift tripwire: which work-relative files does a script read / write?
# --------------------------------------------------------------------------------------------------
def _rel_from_joined(node: ast.AST, roots: Iterable[str]):
    """f"{work}/a_{x}.json" -> 'a_*.json'; os.path.join(work, 'a.json') -> 'a.json'; else None."""
    roots = tuple(roots)
    if isinstance(node, ast.JoinedStr) and node.values:
        first = node.values[0]
        if not (isinstance(first, ast.FormattedValue) and isinstance(first.value, ast.Name) and first.value.id in roots):
            return None
        out = []
        for v in node.values[1:]:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                out.append(v.value)
            else:
                out.append("*")
        s = "".join(out)
        return s[1:] if s.startswith("/") else None
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "join"
            and node.args and isinstance(node.args[0], ast.Name) and node.args[0].id in roots):
        parts = []
        for a in node.args[1:]:
            parts.append(a.value if isinstance(a, ast.Constant) and isinstance(a.value, str) else "*")
        return "/".join(parts) if parts else None
    return None


def _is_write_mode(call: ast.Call) -> bool:
    mode = None
    if len(call.args) > 1 and isinstance(call.args[1], ast.Constant):
        mode = call.args[1].value
    for k in call.keywords:
        if k.arg == "mode" and isinstance(k.value, ast.Constant):
            mode = k.value.value
    return isinstance(mode, str) and any(c in mode for c in "wax")


def _scope_nodes(scope: ast.AST):
    """Nodes that belong to ``scope`` itself: do not descend into a nested function (its own scope)."""
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        n = stack.pop()
        yield n
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        stack.extend(ast.iter_child_nodes(n))


def scan_reads(source: str, roots: Iterable[str] = WORK_NAMES) -> dict:
    """{'reads': set, 'writes': set, 'unresolved': set} of work-relative paths a script touches (a LOWER BOUND).

    Recognised: ``open(p[, mode])`` (a mode with w/a/x is a WRITE), ``np.load`` / ``os.path.exists`` / ``isfile`` /
    ``getsize`` / ``read_text`` (READ), ``np.save`` / ``savez`` / ``write_text`` (WRITE), where ``p`` is an f-string rooted
    at a work-dir name (``f"{work}/a_{x}.json"`` -> ``a_*.json``), ``os.path.join(work, "a.json")``, or a local variable
    assigned one of those in the same function.  A path that is nothing but a wildcard (``*.json``) is UNRESOLVED."""
    tree = ast.parse(source)
    reads, writes, unresolved = set(), set(), set()
    scopes = [tree] + [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for scope in scopes:
        alias = {}
        for stmt in _scope_nodes(scope):
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                rel = _rel_from_joined(stmt.value, roots)
                if rel is not None:
                    alias[stmt.targets[0].id] = rel
        for node in _scope_nodes(scope):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else (fn.id if isinstance(fn, ast.Name) else None)
            if name is None:
                continue
            arg = node.args[0]
            rel = _rel_from_joined(arg, roots)
            if rel is None and isinstance(arg, ast.Name):
                rel = alias.get(arg.id)
            if rel is None:
                continue
            if name == "open":
                (writes if _is_write_mode(node) else reads).add(rel)
            elif name in _READ_FUNCS:
                reads.add(rel)
            elif name in _WRITE_FUNCS:
                writes.add(rel)
    for coll in (reads, writes):
        for r in list(coll):
            if os.path.splitext(r.rsplit("/", 1)[-1])[0] == "*":     # '*.json' names nothing: report, never drop
                coll.discard(r)
                unresolved.add(r)
    return {"reads": reads, "writes": writes, "unresolved": unresolved}


def check_declared_reads(stage: Stage, source: str, roots: Iterable[str] = WORK_NAMES) -> list:
    """Reads in the stage's script that its declaration does not cover (not gated, not optional, not its own product)."""
    scan = scan_reads(source, roots)
    covered = tuple(stage.gates_on) + tuple(stage.optional_reads) + tuple(stage.produces)
    return [f"stage {stage.name!r}: its script reads {r!r} but the plan declares neither a gate nor an optional read "
            f"for it -- a control the plan does not know about cannot have its producer checked"
            for r in sorted(scan["reads"]) if not any(_covers(c, r) or _covers(r, c) for c in covered)]


# --------------------------------------------------------------------------------------------------
# JSON plan (for shell-driven chains) and the CLI
# --------------------------------------------------------------------------------------------------
def plan_from_dict(d: Mapping) -> ChainPlan:
    stages = tuple(Stage(name=s["name"], produces=tuple(s.get("produces", ())), gates_on=tuple(s.get("gates_on", ())),
                         script=s.get("script"), optional_reads=tuple(s.get("optional_reads", ())))
                   for s in d.get("stages", ()))
    return ChainPlan(stages=stages, external=dict(d.get("external", {})), arriving=dict(d.get("arriving", {})))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="refuse a chain plan whose gated controls have no producer")
    ap.add_argument("cmd", choices=("check",))
    ap.add_argument("plan", help="JSON: {stages: [{name, produces, gates_on, script, optional_reads}], "
                                 "external: {ctl: who}, arriving: {ctl: who}}")
    ap.add_argument("--work", default=None, help="work dir: external controls must already exist in it")
    ap.add_argument("--script-root", default=None, help="directory the stage scripts are relative to (default: the plan file's)")
    ap.add_argument("--no-source-check", action="store_true")
    a = ap.parse_args(argv)
    with open(a.plan, encoding="utf-8") as fh:
        plan = plan_from_dict(json.load(fh))
    root = a.script_root or os.path.dirname(os.path.abspath(a.plan))
    probs = check_plan(plan, a.work, verify_source=not a.no_source_check, script_root=root)
    if probs:
        print("CHAIN_PLAN_REFUSED", file=sys.stderr)
        for p in probs:
            print("  - " + p, file=sys.stderr)
        return 2
    print("CHAIN_PLAN_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
