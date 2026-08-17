"""⛔ THE RUNBOOK IS UNDER TEST — because nothing else was testing it.

WHY THIS FILE EXISTS
====================
`V6_GO_PACKAGE.md` §2 is the document an operator opens **at 3 a.m., under time
pressure, and does not re-derive**. That makes it the highest-blast-radius
artifact in the programme, and until today **nothing executed it**. It went
stale in eleven separate ways while the suite stayed green, and two of those
would have been launch-fatal:

  * §2.0 told you to md5 **three** files. The trainer's real script-level import
    closure is **four** (``train_stage_a`` and ``stage_a_probes`` are top-level
    imports at ``train_v6_staged.py:114`` / ``:117``). Shipping three gives you
    ``ModuleNotFound: train_stage_a`` on the pod. ⚠️ And that list was **wrong
    when it was written**, not merely stale — the imports were already there at
    the commit the row cites. One dependency was measured and the CLOSURE was
    assumed, which is `absence found at ONE location is not absence` with the
    polarity flipped: **presence found at one location read as the whole set.**
  * every launch line named ``/workspace/experiments/v6-SW-30k``; the live run
    is ``~/experiments/v6F-SW-30k``, so the whole ``--init-from`` chain pointed
    at a directory that does not exist.

⭐ THE DURABLE FIX IS NOT "SOMEONE RE-READS IT". It is that the runbook stopped
being a second source of truth: §2's launch lines are now the **output of
`scripts/v6_chain.py commands`**, and :func:`test_runbook_launch_lines_are_
exactly_what_v6_chain_emits` fails the suite the moment the two disagree. A
flag rename, a changed default, a renamed run directory or a new Thor constant
now breaks a test instead of an operator.

⛔ NOT A REGEX — AND THAT IS THE POINT
======================================
A regex guard written for this class matched **its own documentation at 176:0**
(it "found" the flags it was documenting and never looked at the code). So
nothing here pattern-matches flag names out of prose. Instead:

  1. Commands are lexed with :mod:`shlex` — a real POSIX lexer, not a pattern.
  2. Flags are validated by feeding the parsed argv to the trainer's **actual**
     ``build_parser()`` (built exactly the way ``main()`` builds it, including
     the ``--i-know-this-is-the-control-arm`` flag that ``main`` adds and
     ``build_parser`` does not — miss that and a *correct* runbook reads stale).
     argparse itself is the authority; an unknown flag raises SystemExit.
  3. Admissibility is checked with the trainer's own ``preflight()``.
  4. The file list in STEP ZERO is compared against an **AST-computed** import
     closure, so a new ``import`` in the trainer breaks this test.
  5. :func:`test_the_extractor_cannot_match_its_own_documentation` pins that the
     extractor reads *commands* and not the prose tables that discuss flags —
     the 176:0 failure, made unrepeatable.

The one thing that is a substring check is "does the doc NAME the probes the
code REQUIRES" (:func:`test_runbook_names_every_probe_S_S_requires`), and its
polarity is safe: adding a probe in code forces a doc edit, never the reverse.
"""
from __future__ import annotations

import ast
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

_STACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_STACK))
sys.path.insert(0, str(_STACK / "scripts"))

import v6_chain as C                                               # noqa: E402
import train_v6_staged as T                                        # noqa: E402

_REPO = _STACK.parent
RUNBOOK = (_REPO / "TanitAD Research Hub" / "Architecture & Inference"
           / "Implementation" / "incoming" / "2026-08-07-hierarchical-wm-redesign"
           / "V6_GO_PACKAGE.md")

TRAINER_TOKEN = "scripts/train_v6_staged.py"

#: ⛔ THE ANCESTOR'S OWN RECORD, banked in the repo and md5-identical to the live
#: run's (`2cb83239cc19d13b9bc7a49a27459b82`, verified on Thor 2026-08-17).
#: Since E1 the chain CARRIES the model geometry out of this file rather than
#: emitting a geometry-free line — MEASURED: without it the stack built at the
#: trainer's defaults, 87.93 M against a 336.54 M checkpoint. On the pod the
#: real `<root>/v6F-SW-30k/config.json` is found first and wins; off-box (here,
#: and on the dev box where these commands are generated) this banked copy is
#: what makes the line reproducible.
GEOMETRY_FROM = (_REPO / "TanitAD Research Hub" / "Architecture & Inference"
                 / "Implementation" / "incoming" / "2026-08-17-st-launch-fixes"
                 / "raw" / "v6F-SW-30k.config.json")

#: ⛔ The config §2.2's block was generated with. It is pinned HERE and quoted in
#: the runbook's own "regenerate rather than edit" recipe, so the two cannot
#: drift silently: change one and this test fails.
THOR_CFG = dict(
    root="/root/experiments",
    workdir="/root/TanitAD/stack",
    train_cache="/root/data/physicalai-train-e438721ae894-w120-256x640cyl",
    val_cache="/root/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl",
    geometry_from=str(GEOMETRY_FROM),
)


# ============================================================================
# extraction — markdown fences, then a REAL LEXER. No pattern touches a flag.
# ============================================================================

def _fenced_bash_blocks(md: str) -> list[str]:
    """Every ```bash fenced block, joined across backslash continuations.

    Fence detection is structural markdown, not a semantic pattern: it asks
    "where does the code block start" and never "what does the code say".
    """
    blocks, cur, inside = [], [], False
    for line in md.splitlines():
        if line.startswith("```"):
            if inside:
                blocks.append("\n".join(cur))
                cur, inside = [], False
            elif line[3:].strip().lower() in ("bash", "sh", "shell"):
                inside = True
            continue
        if inside:
            cur.append(line)
    if inside:                                  # unterminated fence
        blocks.append("\n".join(cur))
    # join `\`-continued lines so shlex sees one command
    return [b.replace("\\\n", " ") for b in blocks]


def _commands(md: str) -> list[list[str]]:
    """Lex every fenced line into tokens with :mod:`shlex`."""
    out = []
    for block in _fenced_bash_blocks(md):
        for line in block.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                out.append(shlex.split(line, comments=True))
            except ValueError:                  # unbalanced quotes in prose
                continue
    return out


def _trainer_argvs(md: str) -> list[list[str]]:
    """The trainer's OWN argv out of each launch line.

    Slices between the trainer path and the first shell redirect, so the
    surrounding ``mkdir -p … && cd … && ENV=… setsid nohup python3 -u`` prologue
    and the ``> train.out 2>&1 < /dev/null &`` epilogue are excluded.
    """
    argvs = []
    for toks in _commands(md):
        if TRAINER_TOKEN not in toks:
            continue
        i = toks.index(TRAINER_TOKEN) + 1
        argv = []
        for t in toks[i:]:
            if t in (">", ">>", "2>&1", "<", "&", "&&", "|", ";"):
                break
            argv.append(t)
        argvs.append(argv)
    return argvs


def _parser_like_main():
    """The parser ``main()`` actually parses with.

    ⚠️ ``build_parser()`` alone is NOT the trainer's CLI: ``main()`` adds
    ``--i-know-this-is-the-control-arm`` (``dest="control_arm_ack"``). A guard
    that checked only ``build_parser`` would report a CORRECT runbook as stale
    — the same shape as the defect that made that flag inert for weeks.
    """
    ap = T.build_parser()
    ap.add_argument("--i-know-this-is-the-control-arm", action="store_true",
                    dest="control_arm_ack", default=False)
    return ap


@pytest.fixture(scope="module")
def md() -> str:
    assert RUNBOOK.exists(), (
        f"the runbook moved: {RUNBOOK}. This test is the thing that keeps it "
        f"honest, so a move must update the path here, not delete the test.")
    return RUNBOOK.read_text(encoding="utf-8")


# ============================================================================
# 1. every flag in the runbook is a flag the trainer really has
# ============================================================================

def test_the_runbook_contains_launch_lines_at_all(md):
    """A runbook whose commands vanished passes every other test vacuously."""
    argvs = _trainer_argvs(md)
    assert len(argvs) >= 5, (
        f"expected the STEP-ZERO dry-run plus four stage launch lines; found "
        f"{len(argvs)}. If §2 was restructured, fix the extractor — do not "
        f"let this test pass on an empty set.")


def test_every_runbook_flag_is_a_real_trainer_flag(md):
    """⭐ THE FLAG-RENAME GUARD. argparse is the authority, not a pattern."""
    ap = _parser_like_main()
    for argv in _trainer_argvs(md):
        try:
            ap.parse_args(argv)
        except SystemExit as e:                 # argparse exits 2 on error
            pytest.fail(
                f"the runbook's launch line no longer parses against "
                f"train_v6_staged.build_parser():\n  {' '.join(argv)}\n"
                f"  argparse exited {e.code}. A flag was renamed or removed and "
                f"the runbook still tells an operator to pass it. Regenerate §2 "
                f"with `v6_chain.py commands`.")


def test_every_runbook_launch_line_passes_the_trainers_own_preflight(md):
    """⛔ The refusals that exist to fire BEFORE a GPU-day is spent.

    Not a re-implementation: ``preflight`` is imported and run. It covers the
    S-W ``--selector`` refusal, the S-S ``--w-select`` refusal, ``--v2-cache``
    being required for a real run, ``--o5-k <= --plan-steps``, and
    ``--init-from`` being mandatory for every staged stage.

    ⚠️ ``preflight`` is run UNDER THE ENVIRONMENT THE RUNBOOK'S OWN LINE
    DECLARES, not this process's. Since E5 the seam-dump import is a startup
    refusal, and the launch line carries `PYTHONPATH=<stack>:<taniteval>` — so
    evaluating it with only `stack/` on the path would fail a line that is
    CORRECT. That is a scope error of exactly the kind this file exists to
    prevent, so the PYTHONPATH is read out of the command instead of assumed,
    and a `--dump-seam-plan` line that does NOT declare a taniteval root is a
    failure in its own right.
    """
    ap = _parser_like_main()
    for cmd_toks in _commands(md):
        if TRAINER_TOKEN not in cmd_toks:
            continue
        i = cmd_toks.index(TRAINER_TOKEN) + 1
        argv = []
        for t in cmd_toks[i:]:                  # stop at the shell epilogue
            if t in (">", ">>", "2>&1", "<", "&", "&&", "|", ";"):
                break
            argv.append(t)
        a = ap.parse_args(argv)
        if a.dry_run:
            continue                            # STEP ZERO is executed below
        pp = next((t.split("=", 1)[1] for t in cmd_toks
                   if t.startswith("PYTHONPATH=")), "")
        roots = [p for p in pp.split(":") if p]
        if a.dump_seam_plan:
            assert any(p.rstrip("/").endswith("taniteval") for p in roots), (
                f"the line sets --dump-seam-plan but its PYTHONPATH ({pp!r}) "
                f"names no taniteval root — the dump would bank NOTHING while "
                f"the run looked healthy, first attempt ~1.8 h in (E5).")
        added = [p for p in roots if p not in sys.path]
        sys.path[:0] = [str(_REPO / "taniteval")] if added else []
        try:
            problems = T.preflight(a)
        finally:
            if added:
                sys.path.pop(0)
        assert problems == [], (
            f"the runbook tells an operator to run a command the trainer's own "
            f"preflight REFUSES:\n  {' '.join(argv)}\n  " + "\n  ".join(problems))


# ============================================================================
# 2. ⭐ the runbook is a RENDERING of the chain, not a second source of truth
# ============================================================================

def test_runbook_launch_lines_are_exactly_what_v6_chain_emits(md):
    """⭐ THE TEST THAT MAKES STALENESS IMPOSSIBLE RATHER THAN UNLIKELY.

    Every stage line in §2.2 must be argv-identical to
    ``v6_chain.launch_line()``. This pins, in one assertion and forever:
    ``--batch 8`` (Thor saturates at 8), ``--v2-lru 6`` (the trainer spawns ZERO
    dataloader workers, so the 8.6 GB/worker rule does not bind and the LRU is
    the real knob), ``--n-candidates`` on every step including S-W (defect D4's
    shape mismatch bypasses ``load_stage_init``'s adjudication entirely), the
    ``v6F-*`` run-directory names, ``setsid``/``-u``/``< /dev/null``, and the
    ``--init-from`` + ``--prev-gate`` pair on every stage above S-W.
    """
    cfg = C.ChainConfig(**THOR_CFG)
    plan = C.build_plan(cfg)
    expected = {s.stage: shlex.split(C.launch_line(s, cfg, plan))
                for s in plan}

    found = {}
    for toks in _commands(md):
        if TRAINER_TOKEN not in toks:
            continue
        i = toks.index(TRAINER_TOKEN) + 1
        if "--stage" not in toks[i:]:
            continue
        stage = toks[i:][toks[i:].index("--stage") + 1]
        if stage in expected:
            found[stage] = toks

    missing = sorted(set(expected) - set(found))
    assert not missing, (
        f"§2.2 no longer carries a launch line for {missing}. Regenerate with "
        f"`v6_chain.py commands`.")

    for stage, want in expected.items():
        got = found[stage]
        assert got == want, (
            f"⛔ the runbook's {stage} line has DRIFTED from v6_chain.py.\n"
            f"  runbook : {' '.join(got)}\n"
            f"  chain   : {' '.join(want)}\n"
            f"  ⇒ the runbook is a RENDERING of the chain, never an edit of it. "
            f"Regenerate §2.2:\n"
            f"     python3 scripts/v6_chain.py commands "
            f"--root {THOR_CFG['root']} --workdir {THOR_CFG['workdir']} "
            f"--train-cache {THOR_CFG['train_cache']} "
            f"--val-cache {THOR_CFG['val_cache']}")


def test_the_pinned_thor_config_is_the_chains_own_default_shape(md):
    """The runbook's "regenerate rather than edit" recipe must be runnable.

    If someone renames a `v6_chain.py` CLI flag, the recipe printed in §2 stops
    working — and an operator would discover that instead of a test.
    """
    ap = C.build_parser()
    argv = ["commands",
            "--root", THOR_CFG["root"], "--workdir", THOR_CFG["workdir"],
            "--train-cache", THOR_CFG["train_cache"],
            "--val-cache", THOR_CFG["val_cache"],
            "--geometry-from", THOR_CFG["geometry_from"]]
    a = ap.parse_args(argv)                     # SystemExit here = renamed flag
    assert a.cmd == "commands"
    assert C._cfg_from_args(a).root == THOR_CFG["root"]
    assert C._cfg_from_args(a).geometry_from == THOR_CFG["geometry_from"]
    assert GEOMETRY_FROM.exists(), GEOMETRY_FROM


def test_thor_constants_in_the_lines_are_the_committed_ones(md):
    """Values are read from PARSED ARGV, never scraped out of prose."""
    ap = _parser_like_main()
    seen = 0
    for argv in _trainer_argvs(md):
        a = ap.parse_args(argv)
        if a.dry_run:
            continue
        seen += 1
        assert a.batch == C.THOR_BATCH, (
            f"--batch {a.batch} in the runbook vs THOR_BATCH {C.THOR_BATCH}. "
            f"Thor's 20 SMs saturate at 8: throughput is FLAT at 12.3-14.1 "
            f"windows/s across a 6x batch range, so 16 (the A40 instinct, and "
            f"the trainer's default) buys nothing and only costs memory.")
        assert a.v2_lru == C.THOR_V2_LRU, (
            f"--v2-lru {a.v2_lru} vs THOR_V2_LRU {C.THOR_V2_LRU}. ⚠️ The "
            f"'8.6 GB per dataloader worker' rule does NOT bind this trainer — "
            f"train() collates synchronously in the main process and never "
            f"builds a DataLoader, so there are ZERO workers to tune. --v2-lru "
            f"is the real host-RAM knob and its trainer default is "
            f"{_trainer_default('--v2-lru')}.")
        assert a.n_candidates == C.ChainConfig().n_candidates, (
            "--n-candidates must be a LADDER-WIDE constant emitted on every "
            "step including S-W: a per-stage difference dies in --init-from "
            "with a shape mismatch that load_state_dict(strict=False) RAISES "
            "on, so STAGE_MAY_INTRODUCE's adjudication never sees it (D4).")
    assert seen == 4, f"expected 4 real launch lines, parsed {seen}"


def _trainer_default(flag: str):
    for act in T.build_parser()._actions:
        if flag in act.option_strings:
            return act.default
    raise AssertionError(f"{flag} is not a trainer flag any more")


# ============================================================================
# 3. STEP ZERO — EXECUTED, not read
# ============================================================================

def test_step_zero_dry_run_actually_runs(md, tmp_path):
    """⭐ The runbook's own preflight, executed as a subprocess on CPU.

    A runbook command that has not been run is a hypothesis. This one is cheap
    (no corpus, no GPU, seconds), so there is no excuse for it being untested.
    """
    argvs = [a for a in _trainer_argvs(md)
             if "--dry-run" in a and "--stage" in a]
    assert argvs, "§2.0(e)'s dry-run command disappeared from the runbook"
    argv = list(argvs[0])
    out = str(tmp_path / "v6-dryrun").replace("\\", "/")
    argv[argv.index("--out") + 1] = out          # the only pod-path substitution
    if "--device" not in argv:
        argv += ["--device", "cpu"]

    proc = subprocess.run(
        [sys.executable, "-u", str(_STACK / "scripts" / "train_v6_staged.py")]
        + argv,
        cwd=str(_STACK), capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=900)
    assert proc.returncode == 0, (
        f"§2.0(e) — the command the runbook tells an operator to run BEFORE "
        f"committing GPU-days — exits {proc.returncode}.\n"
        f"argv: {' '.join(argv)}\n--- stdout ---\n{proc.stdout[-3000:]}\n"
        f"--- stderr ---\n{proc.stderr[-3000:]}")
    assert (Path(out) / "dry_run.json").exists()
    # the gate a dry-run writes is INCONCLUSIVE and stamped, so it can never
    # license a real launch
    gate = Path(out) / "stage_gate.json"
    assert gate.exists(), "the dry-run no longer writes stage_gate.json"


# ============================================================================
# 4. ⭐ STEP ZERO's md5 LIST vs the AST-computed import closure
# ============================================================================

def _module_level_imports(tree: ast.Module) -> list[str]:
    """Top-level module names imported when the file is EXECUTED as a module.

    ⛔ Deliberately NOT ``ast.walk``. ``ast.walk`` also visits imports inside
    function bodies, which are resolved lazily and are NOT needed for the file
    to import — using it here over-counted this trainer's closure by 12 files
    (``train_flagship_v4``, ``refb_train``, … are all function-local). An
    over-broad closure is not a safe error: it would make §2.0(a) list a dozen
    files nobody needs, and a list nobody believes is a list nobody follows.
    Descends into module-level ``if`` / ``try`` guards, never into ``def`` /
    ``class``.
    """
    out: list[str] = []
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Import):
            out += [al.name.split(".")[0] for al in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.append(node.module.split(".")[0])
        elif isinstance(node, (ast.If, ast.Try)):
            stack += list(node.body) + list(node.orelse)
            stack += list(getattr(node, "finalbody", []))
            for h in getattr(node, "handlers", []):
                stack += list(h.body)
    return out


def _script_import_closure(entry: str) -> set[str]:
    """Every ``scripts/*.py`` the entry point needs AT IMPORT TIME, transitively.

    Static, so it works on a box with no torch and breaks the moment someone
    adds a top-level import to the trainer.

    ⚠️ Cross-checked against a RUNTIME measurement (mine 2026-08-16): importing
    ``train_v6_staged`` and filtering ``sys.modules`` by ``__file__`` parent gave
    exactly these four names, so the static walk is neither over- nor
    under-counting.
    """
    scripts = _STACK / "scripts"
    seen, todo = set(), [entry]
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        p = scripts / f"{name}.py"
        if not p.exists():
            continue
        seen.add(name)
        for m in _module_level_imports(ast.parse(p.read_text(encoding="utf-8"))):
            if m not in seen and (scripts / f"{m}.py").exists():
                todo.append(m)
    return seen


def test_the_static_closure_matches_the_runtime_one():
    """⛔ Absence found at ONE location is not absence — and neither is presence.

    The static walk is the guard; this pins it against the number that was
    MEASURED by actually importing the trainer, so a future refactor cannot
    quietly turn the guard into a different question.
    """
    assert _script_import_closure("train_v6_staged") == {
        "train_v6_staged", "train_stage_a", "stage_a_probes",
        "train_v58f_unicycle_head"}, (
        "the trainer's import-time closure changed. That is not a test failure "
        "to silence — it is a change to what must be FILE-SHIPPED to a pod. "
        "Update §2.0(a) of the runbook and this assertion together.")


def test_step_zero_md5_list_is_the_real_import_closure(md):
    """⛔ THE DEFECT THAT SHIPPED: 'THREE files, not two' was never true.

    ``train_stage_a`` and ``stage_a_probes`` were already top-level imports at
    the very commit the old row cited. Ship the three files the runbook named
    and the pod dies with ``ModuleNotFound: train_stage_a`` — after the transfer,
    after the md5 verification, at the moment of launch.
    """
    closure = _script_import_closure("train_v6_staged")
    assert "train_stage_a" in closure and "stage_a_probes" in closure, (
        "the trainer's import closure changed; re-derive §2.0(a) rather than "
        "trusting this assertion's example names")

    listed: set[str] = set()
    for toks in _commands(md):
        if not toks or toks[0] != "md5sum":
            continue
        for t in toks[1:]:
            if t.endswith(".py") and "/scripts/" in t.replace("\\", "/"):
                listed.add(Path(t).stem)

    missing = sorted(closure - listed)
    assert not missing, (
        f"⛔ §2.0(a) tells the operator to md5 a set of files that does NOT "
        f"cover the trainer's import closure. Missing: {missing}.\n"
        f"  A file-ship following that list dies at import on the pod, and a "
        f"STALE copy of any of them does not fail loudly — it changes the "
        f"emission envelope.\n"
        f"  ⚠️ This is `presence found at ONE location read as the whole set`. "
        f"Enumerate the closure; do not measure one dependency and generalise.")


# ============================================================================
# 5. the FIRED pre-registrations the runbook must not contradict
# ============================================================================

def test_no_runbook_launch_line_builds_a_selector_while_SEL1_is_refused(md):
    """⛔ SEL-1 is REFUSED (E-WC2, 2026-08-16). Computed from parsed argv."""
    ap = _parser_like_main()
    for argv in _trainer_argvs(md):
        a = ap.parse_args(argv)
        assert a.selector == "none", (
            f"the runbook tells an operator to launch --selector {a.selector!r} "
            f"while SEL-1 stands REFUSED (σ/ADE {C.SEL1_ADMISSION['sigma_over_ade']} "
            f"vs a pre-registered refusal line of "
            f"{C.SEL1_ADMISSION['refused_at_sigma_over_ade']}; the CI's LOWER "
            f"bound is 2.48x the threshold). assert_selector_admissible would "
            f"refuse it at launch — the runbook must not send anyone into that.")
        assert not a.w_select, (
            "--w-select must be 0 while SEL-1 is REFUSED; a selection loss with "
            "no scorer is how a selector silently never trains.")


def test_runbook_quotes_the_pre_registered_SW_thresholds(md):
    """The reopening thresholds were committed in code BEFORE the dump exists.

    A runbook that quotes a *different* number would let the decision be made
    after seeing the measurement, which is what a pre-registration exists to
    prevent.
    """
    for key in ("funded_at_or_below_m", "refused_above_m"):
        val = C.SW_LATENT_ADMISSION[key]
        assert f"{val}" in md, (
            f"§2.6.1 no longer quotes SW_LATENT_ADMISSION[{key!r}] = {val}. "
            f"The thresholds live in code (v6_chain.py) and the doc renders "
            f"them; if code changed, the pre-registration was broken, not the "
            f"doc.")
    assert C.SW_LATENT_ADMISSION["refused_above_m"] > \
        C.SW_LATENT_ADMISSION["funded_at_or_below_m"]


def test_runbook_names_every_probe_S_S_requires(md):
    """⛔ S-S requires THREE probes, not one.

    Substring containment, deliberately, and the polarity is the safe one: add a
    required probe in code and the doc MUST be edited. It can never pass by the
    doc teaching the test what to look for, because the names come from
    ``STAGE_GATE_SPEC``.
    """
    required = T.STAGE_GATE_SPEC["S-S"]["required"]
    assert len(required) >= 3, (
        "STAGE_GATE_SPEC['S-S']['required'] shrank — if the revalidations were "
        "downgraded to `reported`, an S-S gate that omits them would read PASS "
        "instead of INCONCLUSIVE, which is exactly what a silent carry-forward "
        "of S-T's stale selector certificate looks like.")
    for probe in required:
        assert probe in md, (
            f"the runbook never mentions S-S's REQUIRED probe {probe!r}. An "
            f"operator would plan one gate probe and discover two more after "
            f"~2.5 GPU-days.")


def test_runbook_carries_thors_only_admissible_memory_probe(md):
    """`mem_get_info` / `free` / `tegrastats` / `VmRSS` all misreport on
    unified memory, in BOTH directions."""
    assert C.THOR_MEMORY_PROBE in md, (
        f"§2.3 no longer names {C.THOR_MEMORY_PROBE} as the only admissible "
        f"probe on Thor. A probe that reports the wrong scope is worse than no "
        f"probe, because it looks like an answer.")


# ============================================================================
# 6. ⭐ the guard may not match its own documentation (the 176:0 failure)
# ============================================================================

def test_the_extractor_cannot_match_its_own_documentation(md):
    """⛔ A regex guard for this class once matched its own docs 176:0.

    §2.5's control-arm TABLE names ``--per-layer-encoders``, ``--o5-mode
    endpoint``, ``--no-isolate-planner`` and ``--i-know-this-is-the-control-arm``
    in prose. A pattern-matching guard would "find" those, report the runbook
    as covered, and never look at a command. The lexer-based extractor must
    return ZERO commands from prose, and every command it does return must be a
    real trainer invocation.
    """
    assert "--per-layer-encoders" in md and "--i-know-this-is-the-control-arm" in md, (
        "§2.5's control-arm table is gone; this test's premise needs rewriting")

    for argv in _trainer_argvs(md):
        assert argv and argv[0].startswith("--"), (
            f"the extractor returned something that is not a trainer argv: "
            f"{argv[:4]}")
        assert "--stage" in argv, (
            f"a launch line with no --stage was extracted: {' '.join(argv)}")

    # The prose-only flags must appear in NO extracted command. ⚠️ `--uplink`
    # LEFT this set on 2026-08-17: since E1 the chain CARRIES the ancestor's
    # geometry explicitly (defaults included, so the line's meaning cannot move
    # when a default does), and `--uplink stopgrad` is now genuinely in every
    # launch line. The four that remain are CONTROL-ARM flags the chain never
    # emits, which is what makes them a valid leak canary.
    prose_only = {"--per-layer-encoders", "--no-isolate-planner",
                  "--no-isolate-uplink", "--i-know-this-is-the-control-arm"}
    for argv in _trainer_argvs(md):
        leaked = prose_only.intersection(argv)
        assert not leaked, (
            f"{sorted(leaked)} came out of a markdown TABLE, not a command — "
            f"the extractor is matching documentation, which is the 176:0 "
            f"failure this test exists to make unrepeatable.")


def test_this_test_would_actually_fail_on_a_stale_runbook(md, tmp_path):
    """⭐ A GUARD NOBODY HAS SEEN FIRE IS A GUARD NOBODY KNOWS ABOUT.

    Feeds the extractor the PRE-2026-08-16 line (``--batch 16``, the
    ``v6-SW-30k`` directory, no ``--v2-lru``, no ``--n-candidates``) and asserts
    it is rejected. Without this, every assertion above could be vacuously true.
    """
    stale = (
        "```bash\n"
        "mkdir -p /workspace/experiments/v6-ST-10k && "
        "cd /workspace/TanitAD/stack && "
        "PYTHONPATH=/workspace/TanitAD/stack OMP_NUM_THREADS=6 "
        "nohup python3 scripts/train_v6_staged.py --stage S-T "
        "--prev-gate /workspace/experiments/v6-SW-30k/stage_gate.json "
        "--init-from /workspace/experiments/v6-SW-30k/ckpt.pt "
        "--max-horizon 60 --out /workspace/experiments/v6-ST-10k "
        "--steps 10000 --batch 16 --lr 1e-4 --w-t1 1.0 "
        "> /workspace/experiments/v6-ST-10k/train.out 2>&1 &\n"
        "```\n")
    argvs = _trainer_argvs(stale)
    assert len(argvs) == 1, "the extractor failed to read the stale line at all"
    a = _parser_like_main().parse_args(argvs[0])
    assert a.batch != C.THOR_BATCH, (
        "the stale fixture no longer differs from the committed constant — "
        "update the fixture, do not weaken the test")
    assert a.v2_lru != C.THOR_V2_LRU
    cfg = C.ChainConfig(**THOR_CFG)
    plan = C.build_plan(cfg)
    want = shlex.split(C.launch_line(C.step_by_key(plan, "S-T"), cfg, plan))
    assert argvs[0] != [t for t in want if t.startswith("-")], \
        "the stale line must not compare equal to the chain's emission"


def test_the_import_closure_check_would_fail_on_the_old_three_file_list():
    """The old §2.0(a) list, replayed. It must be rejected."""
    old = ("```bash\n"
           "md5sum /workspace/TanitAD/stack/tanitad/models/v6.py \\\n"
           "       /workspace/TanitAD/stack/scripts/train_v6_staged.py \\\n"
           "       /workspace/TanitAD/stack/scripts/train_v58f_unicycle_head.py\n"
           "```\n")
    listed = set()
    for toks in _commands(old):
        if toks and toks[0] == "md5sum":
            listed |= {Path(t).stem for t in toks[1:]
                       if t.endswith(".py") and "/scripts/" in t}
    closure = _script_import_closure("train_v6_staged")
    assert closure - listed, (
        "the old three-file list would now pass — which would mean the trainer "
        "no longer imports train_stage_a/stage_a_probes. Re-derive §2.0(a).")
