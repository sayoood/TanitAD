"""PREREG_V8F_ORECOVER on box 2 (Amendments 1, 2 and 3): the queue stages `gateorc` / `trainorc` / `evalorc` ARM SEED and the chain `code/box2_ops/chain_v8f_orc.sh`. CPU, no network, nothing reaches box 2.

WHAT THIS FILE PROVES (each item has a deliberate-regression arm: `raw/2026-10-26-orcchain/mutate_orcchain.py` breaks the line that makes it true and shows these tests RED):
  (A) the queue   four new labels and nothing else; every one of the 79 older labels byte-identical (sha256 pins of 98713259) and 41 other stages' dry-run lines byte-identical; ORC-0 is ORC's argv MINUS the five
                  ego-perturbation LAW flags (Amendment 3), with `--ego-perturb-p 0` and its own `--out`, at both seeds; every lever of Amendment 1 NAMED once with its registered value (ORC-0: none of the five); the init path is the BANKED M2b_s<seed> checkpoint; the gate line is the training line with
                  `--out` / `--selfcheck-only` changed; refusals; the line the stage PRINTS is the line it RUNS (fake trainer, also under a 60-char REPO + 70-char tmp dir: every line far under cmd's 8,191)
  (B) the chain   LF only, `bash -n`, static order (refusal, wait, atomic marker, refresh, checkpoint sha, ALL FOUR gates, token, the four runs in the PREREG's order, nothing conditional), the predecessor REFUSAL without
                  AFTER_OUT and its wait semantics RUN end to end on a stubbed box, `--dry-run` printing the exact commands / shas / token schema and shipping nothing
  (C) the gate    the chain's assertions are RUN against selfcheck.json files written by the REAL trainer (synthetic data, the registered fine-tune argv): a good gate passes, each single-field corruption is named
  (D) the token   written by the chain only after those assertions, ACCEPTED by `orc_read.g_token` against the real run's argv.json, bound to the TRAINING argv (not the gate's), refused on every way it can be wrong
  (E) the run     the post-run record check, and the perturbation-reached control agreeing with `orc_read.g_pert`
AMENDMENT 3 closes the blocker this build found (the trainer's gate refused ORC-0 with the law flags): ORC-0's gates PASS on the REAL trainer, unpatched, at both seeds (`test_orc0_gates_pass_*`), and a law flag
named on ORC-0 is refused (`test_a_law_flag_*`: by the trainer's gate when its value is not the default, by the chain's assertions whenever it is named at all)."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-v8f-mini"
CODE = PKG / "code"
QUEUE = Path(os.environ.get("ORC_CHAIN_QUEUE", str(CODE / "v8f_mini_box2_queue.cmd")))       # the env overrides are the mutation harness's door (raw/2026-10-26-orcchain/mutate_orcchain.py)
CHAIN = Path(os.environ.get("ORC_CHAIN_SCRIPT", str(CODE / "box2_ops" / "chain_v8f_orc.sh")))
READ_P = PKG / "raw" / "2026-10-26-orcread" / "orc_read.py"
FULL = PKG / "raw" / "full"
needs_queue = pytest.mark.skipif(not QUEUE.exists(), reason="the v8F-mini package is not in this checkout")
needs_chain = pytest.mark.skipif(not CHAIN.exists(), reason="the v8F-mini package is not in this checkout")
needs_reader = pytest.mark.skipif(not READ_P.exists(), reason="raw/2026-10-26-orcread (the reader) is not in this checkout")
needs_banked = pytest.mark.skipif(not all((FULL / "runs" / f"M2b_s{s}" / "summary.json").exists() for s in (0, 1)), reason="raw/full (the banked M2b records) is not in this checkout")
win = pytest.mark.skipif(sys.platform != "win32", reason="the queue is a Windows batch file")

CKPT_SHA = {0: "601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8", 1: "6104fb15dc21d524b6ed53be632b8d53ec5413526ac587734c22684c7e50c2e6"}
HOLD = "f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9"
BOX2 = dict(REPO=r"D:\tanitad\repo", PY=r"D:\tanitad\venv\Scripts\python.exe", CACHEFULL=r"D:\tanitad\cache_full", V9=r"D:\tanitad\data\v9", RUNSM=r"D:\tanitad\runs\v8f_mini_full")
ORC_LABELS = {"orcarm", "gateorc", "trainorc", "evalorc"}
ORC_STAGES = ("gateorc", "trainorc", "evalorc")
# Labels appended AFTER the ORC edit by later, insertion-only edits, each declared with its PREREG. They are tested by their own builds
# (raw/2026-10-26-dagger-queue/, raw/2026-10-27-dagger2-queue/, raw/2026-10-27-oaccflat-queue/, raw/2026-10-27-dag-orc-queue/); here they only have to sit after the ORC region and leave it untouched.
# A label that is in none of PRE_LABELS, ORC_LABELS and LATER_LABELS still turns the test below RED: a new edit declares itself here.
LATER_LABELS = {
    "dagname": "PREREG_V8F_DAGGER", "prepdag": "PREREG_V8F_DAGGER", "dagarm": "PREREG_V8F_DAGGER", "dagarm_dry": "PREREG_V8F_DAGGER",
    "dagarm_have": "PREREG_V8F_DAGGER", "bankdag": "PREREG_V8F_DAGGER", "gatedag": "PREREG_V8F_DAGGER", "traindag": "PREREG_V8F_DAGGER",
    "evaldag": "PREREG_V8F_DAGGER",
    "dag2arm": "PREREG_V8F_DAGGER2", "dag2arm_dry": "PREREG_V8F_DAGGER2", "dag2arm_have": "PREREG_V8F_DAGGER2", "bankdag2": "PREREG_V8F_DAGGER2",
    "dagoaccarm": "PREREG_V8F_DAG_OACC", "dagoaccarm_dry": "PREREG_V8F_DAG_OACC", "dagoaccarm_have": "PREREG_V8F_DAG_OACC", "dagoaccwacc": "PREREG_V8F_DAG_OACC", "dagoaccwacc_dry": "PREREG_V8F_DAG_OACC", "dagoaccwacc_have": "PREREG_V8F_DAG_OACC", "waccdag": "PREREG_V8F_DAG_OACC",
    "flatdagarm": "PREREG_V8F_FLATDAG", "flatdagarm_dry": "PREREG_V8F_FLATDAG", "flatdagarm_have": "PREREG_V8F_FLATDAG", "evalflatdag": "PREREG_V8F_FLATDAG", "evalflatdag_1": "PREREG_V8F_FLATDAG", "evalflatdag_2": "PREREG_V8F_FLATDAG", "bankflat": "PREREG_V8F_FLATDAG",
    "dagorcarm": "PREREG_V8F_DAG_ORC",
}
LATER_FIRST = "dagname"                                                       # the first label after the ORC region
LATER_HEADER = tuple(f"rem   v8f_mini_box2_queue.cmd {st} " for st in ("bankdag", "gatedag", "traindag", "evaldag", "bankdag2")) \
    + ("rem   gatedag / traindag / evaldag also take ARM dag2 ", "rem   gatedag / traindag / evaldag also take ARM dagoacc ", "rem   gatedag / traindag / evaldag also take ARM flatdag ",
       "rem   v8f_mini_box2_queue.cmd waccdag ", "rem   v8f_mini_box2_queue.cmd bankflat ",
       "rem   gatedag / traindag / evaldag also take ARM dagorc ")           # the later edits' usage lines in the header
LAW = ("--ego-perturb-vscale", "--ego-perturb-vmax", "--ego-perturb-adrop", "--ego-perturb-yawrate-std", "--ego-perturb-seed")     # PREREG Amendment 3: named by ORC, by NONE of ORC-0
RUNS = (("orc", 0), ("orc0", 0), ("orc", 1), ("orc0", 1))                  # the PREREG's order (section 5): ORC s0, ORC-0 s0, ORC s1, ORC-0 s1
# sha256[:16] of "\n" + the text of each label of the queue AT 98713259 (the programme's pin convention) and of the 41 other stages' dry-run output (lines with .py) under the fixed environment of
# raw/2026-10-26-orcchain/gen_pins.py (REPO Z:\r, PY Z:\py.exe, RUNSM Z:\runsm, CACHEFULL Z:\cf, V9 Z:\v9, WACC 0.5): this edit appended after the last label and edited none
PRE_LABELS = {
    "d_train": "653a25ac6e43be79", "d_eval": "b64a718bcf22e4ca", "prep": "8f8749398e785a74", "gate": "01fa4b4d415d3057", "train": "c3867e94dae8b06b",
    "eval": "8d3f96a2ec519c6d", "tac": "8357d58b9565d87f", "tacone": "d05a9be75c4de2bd", "tacgrp": "7cae083fcfb5a237", "tacgone": "cdbb6e94327d04af",
    "tacgrp2": "efb5d1a2944b99fa", "tacgtwo": "14ccef2e21c86f83", "tacabs": "60aa7130ee682938", "tacabsarm": "0ed52c566b9de578", "gatewc": "487c836142548e93",
    "trainwc": "5565fa65d285f47f", "evalwc": "ccb014965cc92aa5", "gatepgt": "3ca89871f8da1c43", "trainpgt": "49fe41ddd1622379", "evalpgt": "a04620ebdcfbb068",
    "evalpgt_1": "effe5bbee55d4771", "evalpgt_2": "33cdd72c074d8c6a", "evalpgt_3": "a7e30c81d582343c", "evalpgt_4": "22636d9d007850fa", "gridbase": "56299fbe24cec6f7",
    "gategrid": "e3250ab756539587", "traingrid": "87cf143655b55573", "evalgrid": "399406480951bc1d", "evalgrid_1": "74aaef5e39bbaca5", "evalgrid_2": "f242aa25ecd51bd6",
    "evalgrid_3": "210db6bc5c411a0c", "evalgrid_4": "22636d9d007850fa", "gatehist": "93a341e35cc9a0b1", "trainhist": "344fcc45b1624ead", "evalhist": "413108b37c6fe068",
    "evalhist_1": "03081f7da50e8739", "evalhist_2": "168db150abcf3ce1", "evalhist_3": "d9d586a4304f2d24", "evalhist_4": "22636d9d007850fa",
    "gatedscale": "0d91b533582f0ea6", "traindscale": "7c5f99d4e8e0c096", "evaldscale": "080e32b4ee8c2ac6", "evaldscale_1": "f59695b83d08d77a",
    "evaldscale_2": "813ee4390fc783ce", "evaldscale_3": "fdba053d0afad97d", "evaldscale_4": "22636d9d007850fa", "loraenv": "adacc37155199a86",
    "identitylora": "ee8f28bc42b410c9", "smokelora": "af9acad644a32d72", "gatearm": "b79d2a733b86fe61", "floors": "2b7f345ee3e86bcc", "compare": "b41e1a1c859c59c3",
    "kind": "d1445aa3e2135915", "kindseed": "9c13c94705b73355", "kindown": "df45fbd31218aea9", "kindorc": "e7051ae3f70c3a4f", "gatemixo": "b2ea5430b698741c",
    "trainmixo": "419a7db33d684844", "evalmixo": "7acc0975f2aa9ef3", "evalmixo_1": "1cc5381d64b27142", "evalmixo_2": "a70ba66191aed3cf",
    "evalmixo_3": "5d326de83e5d37b1", "evalmixo_4": "0cde88e556e0b0e4", "evalmixo_5": "479e39bc5b10065b", "evalmixo_6": "77271eba1510a820",
    "evalmixo_7": "901e965d7b940ca4", "gateflat": "cec296bd862eedcd", "trainflat": "9e28a04f9c79af1e", "evalflat": "024f6a7e06886322", "evalflat_1": "2f90baa2f624c148",
    "evalflat_2": "f3c16ba44e719907", "evalflat_3": "ea291c4013c2b747", "evalflat_4": "901e965d7b940ca4", "waccoacc": "9b7d270c845087bb", "oaccwacc": "4a310e6bb75049e0",
    "gateoacc": "dd5b553616e77133", "trainoacc": "3ed09e112ca497a6", "evaloacc": "289bb5bf587e0a5c", "fitoacc": "4b9db406766c1422",
}

PRE_DRY = {
    "gate": "d558d46dc0719440", "train M2b 0": "3883c04e634b5ce7", "train M2b 1 pool8x20": "2e3491a3d4425d6d", "eval M2b 0": "af6931c5b19136f6",
    "floors": "e7aa39b6dbb78ba8", "compare": "652ff3adbf032df5", "tac": "2ae2a21592b53aeb", "tacgrp": "2586a181f6ba7a36", "tacgrp2": "a8215849f17435c5",
    "tacabs": "904963cfe0f32d7c", "gatearm M2b": "ab89df3cf37ebd3c", "tacabsarm M2b 0": "db9dacd04c293b0f", "gatewc": "ef4844f0dff4d358",
    "trainwc 0": "c86801474df21c49", "evalwc 0": "996cbde23a72689f", "gatepgt": "fee5d3ad9416077c", "trainpgt 0": "a3ef3b30ba37b9a1", "evalpgt 0": "89ff9f49e0480b54",
    "gategrid M2b": "8823ba7765b671a4", "traingrid 0 M2b": "d66f6b09e3a283fb", "evalgrid 0 M2b": "abcbdda740a7e791", "gatehist": "59d196bc411a16b0",
    "trainhist 1": "6d1be9592dfb73df", "evalhist 0": "12dbd689bd9e8f83", "gatedscale": "3512ebf055906f87", "traindscale 0": "862142ce1c9ee957",
    "evaldscale 0": "2733552ecd030597", "kind": "1e4b503123b76cf2", "gatemixo": "1c1095d715c47e7f", "trainmixo 0": "620195022872e796", "trainmixo 1": "1643287d799417ef",
    "evalmixo 0": "ee4be0db7734aca4", "gateflat": "ac03432f743a934b", "trainflat 1": "d7bef299d43c11ad", "evalflat 0": "213540357745a313",
    "waccoacc": "50d451260a76d231", "gateoacc": "5dc9dc1fa1709e49", "trainoacc 0": "7f58815720137d1d", "trainoacc 1": "05100e261920ea72",
    "evaloacc 1": "c1332ad1667dc5f5", "fitoacc 0": "dcd50e9e41c3cfbe",
}
PRE_PREAMBLE = "ea812008cc9fe5bd"          # the text before the first label (header, environment, dispatch table, usage), with this edit's 5 header lines and 3 dispatch lines removed


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def _queue_text():
    return QUEUE.read_text(encoding="utf-8").replace("\r\n", "\n")


def _blocks():
    parts = re.split(r"(?m)^(:[A-Za-z0-9_]+)$", _queue_text())
    return {parts[i][1:]: parts[i + 1] for i in range(1, len(parts), 2)}


def _orc_region():
    """ORC's own region: from :orcarm up to the first label a later edit appended (LATER_FIRST)."""
    return _queue_text().split("\n:orcarm\n", 1)[1].split(f"\n:{LATER_FIRST}\n", 1)[0]


def _block(label):
    return _queue_text().split(f"\n:{label}\n", 1)[1].split("\nexit /b 0\n", 1)[0].split("\n")


def _code(lines):
    return [x for x in lines if x.strip() and not x.lower().startswith("rem ")]


def _sha16(s):
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def _argv_of(line):
    return [t.strip('"') for t in re.findall(r'"[^"]*"|\S+', line.strip())]


def _train_argv(line):
    t = _argv_of(line)
    return t[next(k for k, x in enumerate(t) if x.lower().endswith("v8f_mini_train.py")) + 1:]


def _without(tokens, drop):
    out, i = [], 0
    while i < len(tokens):
        if tokens[i] in drop:
            i += 2
            continue
        out.append(tokens[i])
        i += 1
    return out


def _drop_flags(tokens, drop):
    """tokens without the flags in ``drop`` AND all the values that follow each of them (up to the next `--flag`): vscale carries two."""
    out, k = [], 0
    while k < len(tokens):
        if tokens[k] in drop:
            k += 1
            while k < len(tokens) and not tokens[k].startswith("--"):
                k += 1
            continue
        out.append(tokens[k])
        k += 1
    return out


def _flags(tokens):
    o, k = {}, 0
    while k < len(tokens):
        if tokens[k].startswith("--"):
            j = k + 1
            while j < len(tokens) and not tokens[j].startswith("--"):
                j += 1
            o.setdefault(tokens[k], []).append(tokens[k + 1:j])
            k = j
        else:
            k += 1
    return o


def _env(base):
    e = {k: v for k, v in os.environ.items() if k.upper() not in ("V8F_DRYRUN", "WACC", "REPO", "PY", "RUNSM", "CACHEFULL", "V9", "FAKE_TRAIN", "FAKE_FAIL_TAG")}
    e.update(base)
    return e


def _dry(args, env=None):
    r = subprocess.run(f'cmd /c ""{QUEUE}" {args}"', env=_env({"V8F_DRYRUN": "1", **(BOX2 if env is None else env)}), capture_output=True, text=True, timeout=60, encoding="utf-8", stdin=subprocess.DEVNULL)
    return r.returncode, [x for x in r.stdout.splitlines() if ".py" in x], r.stdout


def _dry1(args, env=None):
    rc, ln, out = _dry(args, env)
    assert rc == 0 and len(ln) == 1, f"`{args}` should print exactly one command line, got rc {rc}, lines {ln}, out {out[:300]}"
    return ln[0]


def _registered(arm, seed):
    """The fine-tune of Amendments 1 and 3 as the flags that follow M2b's own (%A%), in the order the queue writes them: ORC names the five law flags after --ego-perturb-p, ORC-0 names NONE of them."""
    return ["--init-ckpt", rf"D:\tanitad\runs\v8f_mini_full\M2b_s{seed}\ckpt_last.pt", "--steps", "3000", "--lr", "1e-4", "--warmup", "100", "--p-gt-start", "0.2", "--p-gt-end", "0.2",
            "--ego-perturb-p", "0.3" if arm == "orc" else "0",
            *(["--ego-perturb-vscale", "0.5", "1.0", "--ego-perturb-vmax", "8.0", "--ego-perturb-adrop", "1.5", "--ego-perturb-yawrate-std", "2.0", "--ego-perturb-seed", str(seed)] if arm == "orc" else [])]


# =============================================================================================================================================
# (A) THE QUEUE
# =============================================================================================================================================
@needs_queue
def test_queue_is_crlf_every_older_label_is_byte_identical_and_only_the_four_orc_labels_are_added():
    raw = QUEUE.read_bytes()
    assert raw.count(b"\r\n") == raw.count(b"\n") > 1000 and raw.count(b"\r") == raw.count(b"\r\n") and raw.endswith(b"\r\n"), "every line CRLF, no bare CR"
    q, blocks = _queue_text(), _blocks()
    assert len(PRE_LABELS) == 79
    for label, h in PRE_LABELS.items():
        assert label in blocks, f"the stage {label} was removed"
        assert _sha16("\n" + blocks[label]) == h, f"the stage {label} was edited"
    assert len(ORC_LABELS) == 4 and ORC_LABELS <= set(blocks), "the four labels of this edit are present"
    assert set(blocks) == set(PRE_LABELS) | ORC_LABELS | set(LATER_LABELS), \
        f"a label that is neither older, ORC's nor a declared later edit: {sorted(set(blocks) - set(PRE_LABELS) - ORC_LABELS - set(LATER_LABELS))}"
    for st in ORC_STAGES:
        assert q.count(f'if /i "%~1"=="{st}" goto :{st}\n') == 1, "a stage nobody dispatches never runs"
        assert q.count(f"\n:{st}\n") == 1
    assert q.count("\n:orcarm\n") == 1 and "goto :orcarm" not in q and q.count("call :orcarm %~2 %~3 || exit /b 2") == 3, "orcarm is a helper: called by the three stages, never dispatched"
    disp = re.findall(r'(?m)^if /i "%~1"=="(\w+)" goto :(\w+)$', q)
    assert [(a, b) for a, b in disp if b not in set(blocks)] == [], "a dispatch line whose target label does not exist"
    # the preamble (header, environment, dispatch table, usage line) is the old one plus exactly this edit's lines
    pre = q.split("\n:d_train\n")[0].split("\n")
    i = next(k for k, x in enumerate(pre) if x.startswith("rem   v8f_mini_box2_queue.cmd fitoacc SEED"))
    assert pre[i + 1] == "rem" and pre[i + 2].startswith("rem   v8f_mini_box2_queue.cmd gateorc ARM SEED") and pre[i + 4].startswith("rem   v8f_mini_box2_queue.cmd trainorc ARM SEED") \
        and pre[i + 5].startswith("rem   v8f_mini_box2_queue.cmd evalorc ARM SEED"), "the header documents the three stages"
    rest = pre[:i + 1] + pre[i + 6:]
    rest = [x for x in rest if not (x.startswith('if /i "%~1"==') and any(s in x for s in ORC_STAGES))]
    # the later edits' own preamble lines: their usage lines (DAGGER also added one "rem" separator after them) and their dispatch lines
    k = next((n for n, x in enumerate(rest) if x.startswith(LATER_HEADER)), None)
    if k is not None:
        m = k
        while m < len(rest) and rest[m].startswith(LATER_HEADER):
            m += 1
        assert rest[k - 1] == "rem" and rest[m] == "rem", "the later usage lines sit as one block between two separators"
        rest = rest[:k] + rest[m + 1:]
    later_disp = {f'if /i "%~1"=="{st}" goto :{st}' for st in LATER_LABELS}
    rest = [x for x in rest if x not in later_disp]
    assert not any(x.startswith(LATER_HEADER) for x in rest), "a later usage line outside the block"
    assert _sha16("\n".join(rest)) == PRE_PREAMBLE, "the preamble changed beyond this edit's own lines (the usage line, the environment and the dispatch of every older stage are untouched)"
    assert q.split("\n:orcarm\n")[0].endswith("\n:fitoacc\n" + q.split("\n:fitoacc\n", 1)[1].split("\n:orcarm\n")[0]) and q.split("\n:orcarm\n")[0].endswith("echo [ok] HX weights M2b-oacc_s%~2\nexit /b 0"), \
        "the new stages start right after the last old label, which keeps its text"
    assert ("\n:orcarm\n" + _orc_region()).rstrip().endswith("echo [ok] eval %TG%\nexit /b 0"), "ORC's region ends with evalorc"
    tail = re.findall(r"(?m)^:([A-Za-z0-9_]+)$", q.split("\n:evalorc\n", 1)[1])
    assert tail and tail[0] == LATER_FIRST and set(tail) == set(LATER_LABELS), "every label after ORC's region is a declared later edit"


@needs_queue
def test_queue_orc_stages_are_one_flat_command_string_each_that_the_dry_run_prints_and_the_real_run_executes():
    """A dry run can only print: the real line is checked statically -- the trainer / eval appear ONCE per stage, inside the one string, the dry-run echo comes BEFORE any skip, and the real run executes that string."""
    for label, var, want in (("gateorc", "GX", 'set "GX="%PY%" "%TRAIN%" --arm M2b --seed %ORCSEED% --grid pool4x10 --out "%RUNSM%\\gate_M2b-%ORCS%_s%ORCSEED%" %A% %ORCF% --selfcheck-only"'),
                             ("trainorc", "TX", 'set "TX="%PY%" "%TRAIN%" --arm M2b --seed %ORCSEED% --grid pool4x10 --out "%OUT%" %A% %ORCF%"')):
        code = _code(_block(label))
        assert [x for x in code if "%TRAIN%" in x] == [want], label
        assert f"if defined V8F_DRYRUN (echo %{var}% & exit /b 0)" in code and sum(x.startswith(f"%{var}% ") for x in code) == 1, label
        assert code.index(f"if defined V8F_DRYRUN (echo %{var}% & exit /b 0)") < min([k for k, x in enumerate(code) if x.startswith(("if exist", f"%{var}% ", "if not exist"))]), "the dry run prints before it skips or runs"
        assert "call :orcarm %~2 %~3 || exit /b 2" in code and code.index("call :orcarm %~2 %~3 || exit /b 2") < code.index(next(x for x in code if x.startswith(f'set "{var}='))), "the arm is decided before the string is built"
        assert not any(("pool8x20" in x or "--hist" in x or "--w-" in x or "--hmax" in x or "--window-start" in x or "--code-mix" in x or "--o-interface" in x or "--ego-perturb" in x) for x in code), \
            f"{label}: the levers live in ORCF (one place), not in the stage"
    code = _code(_block("evalorc"))
    assert [x for x in code if "%EVAL%" in x] == ['set "E0="%PY%" "%EVAL%" run --ckpt "%OUT%\\ckpt_last.pt" --tag %TG% %EC% %ER% --project --families"']
    assert "if defined V8F_DRYRUN (echo %E0% & exit /b 0)" in code and sum(x.strip().startswith("%E0% >> ") for x in code) == 1
    assert 'set "OUT=%RUNSM%\\M2b_s%ORCSEED%_%ORCS%"' in code and 'set "TG=M2b-%ORCS%_s%ORCSEED%"' in code and "--tac" not in " ".join(code)
    arm = [x for x in _code(_block("orcarm"))]
    f = next(x for x in arm if x.startswith('set "ORCF='))
    assert f == ('set "ORCF=--init-ckpt "%RUNSM%\\M2b_s%~2\\ckpt_last.pt" --steps 3000 --lr 1e-4 --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2 --ego-perturb-p %ORCP%%ORCL%"'), \
        "the fine-tune's levers are written ONCE, here: --ego-perturb-p, then %ORCL% (the five law flags for ORC, NOTHING for ORC-0: Amendment 3)"
    lw = 'if /i "%~1"=="orc" set "ORCL= --ego-perturb-vscale 0.5 1.0 --ego-perturb-vmax 8.0 --ego-perturb-adrop 1.5 --ego-perturb-yawrate-std 2.0 --ego-perturb-seed %~2"'
    assert arm.count('set "ORCL="') == 1 and arm.count(lw) == 1 and arm.index('set "ORCL="') < arm.index(lw) < arm.index(f), "ORCL is emptied, then set for ORC ONLY, then read by ORCF"
    assert all(sum(x.count(fl) for x in _code(_orc_region().split("\n"))) == 1 for fl in LAW), "each law flag is written once in ORC's region, inside ORC's ORCL"
    assert not any(fl in x for fl in LAW for x in arm if x != lw), "no other line of the helper names a law flag (ORC-0 reaches none)"
    assert '(set "ORCP=0.3" & set "ORCS=orc")' in " ".join(x.split("if /i \"%~1\"==\"orc\" ")[-1] for x in arm if x.startswith('if /i "%~1"=="orc" '))
    assert 'if /i "%~1"=="orc0" (set "ORCP=0" & set "ORCS=orc0")' in arm, "p differs between the arms HERE and nowhere else"
    assert sum(x.count("--ego-perturb-p") for x in _code(_orc_region().split("\n"))) == 1, "--ego-perturb-p appears once in the CODE of ORC's region"
    for label in ("orcarm", *ORC_STAGES):
        body = _block(label)
        assert not any(x.rstrip().endswith("(") for x in body), f"{label}: a multi-line parenthesised block (an `exit /b 1` inside one returns rc 0 from `cmd /c`, MEASURED by the PGT build)"
    for label in ORC_STAGES:
        assert sum(1 for x in _block(label) if x.startswith("if errorlevel 1 (echo [FAIL]") and x.rstrip().endswith("exit /b 1)")) >= 1, label


@needs_queue
@win
@pytest.mark.parametrize("seed", [0, 1])
def test_dry_run_orc_argv_is_the_orc0_argv_plus_the_five_law_flags_with_p_0_3_vs_0_and_out_differing(seed):
    """PREREG Amendment 3: ORC argv == ORC-0 argv + the five law flags; the ONLY other differences are --ego-perturb-p (0.3 vs 0) and --out."""
    o, o0 = _train_argv(_dry1(f"trainorc orc {seed}")), _train_argv(_dry1(f"trainorc orc0 {seed}"))
    fo, f0 = _flags(o), _flags(o0)
    assert _drop_flags(o, {"--out", "--ego-perturb-p", *LAW}) == _drop_flags(o0, {"--out", "--ego-perturb-p"}), "ORC minus (law flags, p, out) == ORC-0 minus (p, out): nothing else differs (a second lever is a confound)"
    assert [k for k in LAW if k in f0] == [] and all(fo[k] for k in LAW) and all(len(fo[k]) == 1 for k in LAW), "ORC names each law flag once; ORC-0 names none of them"
    assert fo["--ego-perturb-p"] == [["0.3"]] and f0["--ego-perturb-p"] == [["0"]], "always NAMED (the flag's default disables the lever), 0.3 for ORC and 0 for ORC-0"
    assert [k for k in fo if k.startswith("--ego-perturb")] == ["--ego-perturb-p", *LAW] and [k for k in f0 if k.startswith("--ego-perturb")] == ["--ego-perturb-p"]
    assert o0[-2:] == ["--ego-perturb-p", "0"], "ORC-0's argv ENDS with --ego-perturb-p 0: nothing of the family follows"
    assert fo["--out"][0][0].endswith(f"M2b_s{seed}_orc") and f0["--out"][0][0].endswith(f"M2b_s{seed}_orc0") and o != o0
    gate_o, gate_0 = _train_argv(_dry1(f"gateorc orc {seed}")), _train_argv(_dry1(f"gateorc orc0 {seed}"))
    assert _drop_flags(gate_o, {"--out", "--ego-perturb-p", *LAW}) == _drop_flags(gate_0, {"--out", "--ego-perturb-p"}) and _flags(gate_o)["--ego-perturb-p"] == [["0.3"]] and _flags(gate_0)["--ego-perturb-p"] == [["0"]]
    assert [k for k in LAW if k in _flags(gate_0)] == [] and all(k in _flags(gate_o) for k in LAW), "the gates carry the same flags as their runs"


@needs_queue
@win
@pytest.mark.parametrize("arm,seed", RUNS)
def test_dry_run_every_lever_is_named_once_with_its_registered_value_and_nothing_else_moved(arm, seed):
    line = _dry1(f"trainorc {arm} {seed}")
    t = _train_argv(line)
    fl = _flags(t)
    want = _flags(_registered(arm, seed))
    for k, v in want.items():
        assert fl.get(k) == v, f"{k}: {fl.get(k)} != the registered {v} (exactly once, with its registered value)"
    assert fl["--arm"] == [["M2b"]] and fl["--seed"] == [[str(seed)]] and fl["--grid"] == [["pool4x10"]], "M2b's training argv, at this seed: the --seed is the seed of the M2b checkpoint being fine-tuned"
    assert all((k in fl) == (arm == "orc") for k in LAW), "ORC names all five law flags, ORC-0 NONE (Amendment 3: the trainer's gate refuses a law flag set while p is 0)"
    assert fl["--init-ckpt"] == [[rf"D:\tanitad\runs\v8f_mini_full\M2b_s{seed}\ckpt_last.pt"]], "the init is the BANKED M2b_s<seed> checkpoint of THIS seed (the chain asserts its sha256)"
    for no in ("--code-mix", "--w-acc", "--hist", "--hmax", "--window-start", "--w-tac", "--w-cons", "--w-traj", "--w-pen", "--w-wm", "--o-interface", "--data-frac", "--max-clips", "--selfcheck-only", "--resume", "--no-resume", "--synthetic"):
        assert no not in fl, f"ONE lever: {no} is not on the line"
    # M2b's argv exactly as the existing trainmixo stage builds it, minus --code-mix, plus the fine-tune
    mixo = _train_argv(_dry1(f"trainmixo {seed}"))
    base = _without(_without(mixo, {"--out"}), {"--code-mix"})
    assert _without(t, {"--out"}) == base + _registered(arm, seed), "the argv = trainmixo's minus `--code-mix kind` plus the registered fine-tune flags, in this order"
    assert all(len(x) < 2000 for x in [line]), "the printed line is far below cmd.exe's 8,191-character limit"


@needs_queue
@win
@pytest.mark.parametrize("arm,seed", RUNS)
def test_dry_run_gate_line_is_the_training_line_with_out_and_selfcheck_only_changed_and_eval_reads_this_runs_checkpoint(arm, seed):
    gate, train, ev = _train_argv(_dry1(f"gateorc {arm} {seed}")), _train_argv(_dry1(f"trainorc {arm} {seed}")), _dry1(f"evalorc {arm} {seed}")
    assert gate.count("--selfcheck-only") == 1 and "--selfcheck-only" not in train and gate[-1] == "--selfcheck-only"
    assert _without(gate[:-1], {"--out"}) == _without(train, {"--out"}), "the gate's argv is the run's argv with --out and --selfcheck-only changed and NOTHING else"
    assert _flags(gate)["--out"][0][0].endswith(f"gate_M2b-{arm}_s{seed}") and _flags(train)["--out"][0][0].endswith(f"M2b_s{seed}_{arm}")
    t = _argv_of(ev)
    assert t[t.index("--tag") + 1] == f"M2b-{arm}_s{seed}" and t[t.index("--ckpt") + 1].endswith(f"M2b_s{seed}_{arm}\\ckpt_last.pt") and "--project" in t and "--families" in t
    assert f"\\M2b_s{seed}\\" not in ev and "--tac" not in t, "THIS run's checkpoint, never B's; the standard eval"


@needs_queue
@win
def test_queue_orc_refuses_a_bad_arm_a_missing_arm_or_seed_and_a_third_seed():
    for args in ("trainorc", "gateorc", "evalorc", "trainorc orc", "trainorc orc0", "trainorc foo 0", "trainorc ORCX 0", "trainorc orc 2", "trainorc orc -1", "trainorc 0 orc", "evalorc orc0", "gateorc orc 7"):
        rc, ln, out = _dry(args)
        assert rc == 2 and ln == [] and "REFUSED" in out, (args, rc, out)
    assert _dry1("trainorc ORC 0") == _dry1("trainorc orc 0"), "the arm is case-insensitive like every dispatch in this file"


@needs_queue
@win
def test_other_stages_dry_run_lines_are_byte_unchanged():
    env = dict(REPO=r"Z:\r", PY=r"Z:\py.exe", RUNSM=r"Z:\runsm", CACHEFULL=r"Z:\cf", V9=r"Z:\v9", WACC="0.5")
    for st, h in PRE_DRY.items():
        rc, ln, out = _dry(st, env)
        assert rc == 0 and ln and _sha16("\n".join(ln)) == h, f"`{st}`: the dry-run output changed (rc {rc})"
    assert len(PRE_DRY) == 41


def _fake_repo(root, repo_name="repo"):
    """A repo whose trainer / eval are FAKES that record the argv they are handed (and make the files the next stage waits for): the queue's own `prep` and stages run unchanged."""
    repo = root / repo_name
    sc = repo / "stack" / "scripts"
    sc.mkdir(parents=True)
    rec = root / "argv.jsonl"
    (sc / "v8f_mini_train.py").write_text(
        "import json, os, sys\n"
        f"open({str(rec)!r}, 'a').write(json.dumps({{'who': 'train', 'argv': sys.argv[1:]}}) + '\\n')\n"
        "a = sys.argv; out = a[a.index('--out') + 1]\n"
        "os.makedirs(out, exist_ok=True)\n"
        "if os.environ.get('FAKE_TRAIN') == 'fail':\n"
        "    sys.exit(3)\n"
        "if '--selfcheck-only' not in a and os.environ.get('FAKE_TRAIN') != 'nosummary':\n"
        "    open(os.path.join(out, 'summary.json'), 'w').write('{}')\n", encoding="utf-8")
    (sc / "v8f_mini_eval.py").write_text(
        "import json, os, sys\n"
        f"open({str(rec)!r}, 'a').write(json.dumps({{'who': 'eval', 'argv': sys.argv[1:]}}) + '\\n')\n"
        "a = sys.argv; d = a[a.index('--out-dir') + 1]; t = a[a.index('--tag') + 1]\n"
        "if os.environ.get('FAKE_FAIL_TAG') == t:\n"
        "    sys.exit(1)\n"
        "os.makedirs(d, exist_ok=True)\n"
        "open(os.path.join(d, 'eval_' + t + '.npz'), 'wb').write(b'x')\n", encoding="utf-8")
    hj = repo / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-refav1-hj-mini" / "raw" / "p10_yaw4512"
    hj.mkdir(parents=True)
    (hj / "p10_prior.pt").write_bytes(b"p")
    (hj / "p10_eval_pred6.npy").write_bytes(b"p")
    cf = root / "cf"
    for sub in ("train", "eval"):
        (cf / sub).mkdir(parents=True)
        (cf / sub / "manifest.jsonl").write_text("{}\n", encoding="utf-8")
    (cf / "native_tracks.pt").write_bytes(b"t")
    (root / "runsm").mkdir()
    return repo, rec


@needs_queue
@win
def test_queue_orc_real_lines_execute_exactly_the_printed_commands(tmp_path):
    """The REAL gate / train / eval stages hand the (fake) trainer and eval exactly the argv the dry run printed, skip what is finished, and notice failures."""
    root = tmp_path
    repo, rec = _fake_repo(root, "repo")
    env = _env(dict(REPO=str(repo), PY=sys.executable, RUNSM=str(root / "runsm"), CACHEFULL=str(root / "cf"), V9=str(root / "v9")))
    runsm = root / "runsm"

    def stage(args, **extra):
        r = subprocess.run(f'cmd /c ""{QUEUE}" {args}"', env=dict(env, **extra), capture_output=True, text=True, timeout=180, encoding="utf-8", stdin=subprocess.DEVNULL)
        return r.returncode, r.stdout

    def printed(args):
        r = subprocess.run(f'cmd /c ""{QUEUE}" {args}"', env=dict(env, V8F_DRYRUN="1"), capture_output=True, text=True, timeout=60, encoding="utf-8", stdin=subprocess.DEVNULL)
        assert r.returncode == 0, r.stdout
        return [x for x in r.stdout.splitlines() if ".py" in x]

    def recs():
        return [json.loads(x) for x in rec.read_text(encoding="utf-8").splitlines()] if rec.exists() else []
    for arm, seed in (("orc", 1), ("orc0", 0)):
        lines = {st: printed(f"{st} {arm} {seed}") for st in ORC_STAGES}
        assert all(len(v) == 1 and len(v[0]) < 8191 - 400 for v in lines.values()), {k: len(v[0]) for k, v in lines.items()}      # + the redirection the real run appends
        n = len(recs())
        rc, out = stage(f"gateorc {arm} {seed}")
        got = recs()
        assert rc == 0 and f"[ok] gate M2b-{arm}_s{seed}" in out and len(got) == n + 1 and got[-1]["who"] == "train" and got[-1]["argv"] == _argv_of(lines["gateorc"][0])[2:] and "--selfcheck-only" in got[-1]["argv"], (rc, out)
        assert not (runsm / f"M2b_s{seed}_{arm}" / "summary.json").exists(), "a gate trains nothing"
        rc, out = stage(f"evalorc {arm} {seed}")
        assert rc == 2 and "has not finished training" in out and len(recs()) == n + 1, "an eval before the training is REFUSED, nothing runs"
        rc, out = stage(f"trainorc {arm} {seed}", FAKE_TRAIN="fail")
        assert rc == 1 and f"[FAIL] train M2b_s{seed}_{arm} - see" in out
        rc, out = stage(f"trainorc {arm} {seed}", FAKE_TRAIN="nosummary")
        assert rc == 1 and "exited 0 without summary.json" in out
        n = len(recs())
        rc, out = stage(f"trainorc {arm} {seed}")
        got = recs()
        assert rc == 0 and len(got) == n + 1 and got[-1]["argv"] == _argv_of(lines["trainorc"][0])[2:], "the REAL trainorc line hands the trainer the printed command"
        a = got[-1]["argv"]
        assert a[a.index("--init-ckpt") + 1].endswith(f"M2b_s{seed}\\ckpt_last.pt") and a[a.index("--ego-perturb-p") + 1] == ("0.3" if arm == "orc" else "0") and a[a.index("--seed") + 1] == str(seed)
        assert (runsm / f"M2b_s{seed}_{arm}" / "summary.json").exists() and not (runsm / f"M2b_s{seed}").exists(), "the run has its own directory; B's is never created or touched"
        rc, out = stage(f"trainorc {arm} {seed}")
        assert rc == 0 and f"[skip] M2b_s{seed}_{arm}" in out and len(recs()) == n + 1, "a finished run is not re-trained"
        rc, out = stage(f"evalorc {arm} {seed}", FAKE_FAIL_TAG=f"M2b-{arm}_s{seed}")
        assert rc == 1 and f"[FAIL] eval M2b-{arm}_s{seed} - see" in out
        n = len(recs())
        rc, out = stage(f"evalorc {arm} {seed}")
        got = recs()
        assert rc == 0 and len(got) == n + 1 and got[-1]["who"] == "eval" and got[-1]["argv"] == _argv_of(lines["evalorc"][0])[2:]
        assert (runsm / "eval" / f"eval_M2b-{arm}_s{seed}.npz").exists()
        rc, out = stage(f"evalorc {arm} {seed}")
        assert rc == 0 and f"[skip] eval M2b-{arm}_s{seed}" in out and len(recs()) == n + 1


@needs_queue
@win
@pytest.mark.parametrize("arm,seed", RUNS)
def test_dry_run_under_a_60_char_repo_and_70_char_directories_keeps_every_line_far_below_the_cmd_limit(arm, seed):
    """cmd.exe cuts a line at 8,191 characters AFTER expansion. REPO is a 60-character path; RUNSM, CACHEFULL and V9 are 70-character ones (the queue repeats them ~14 times per command): the longest printed line,
    plus the redirection the real run appends to it, stays below half the limit, and cmd itself parsed the echo of the whole line."""
    env = dict(REPO="D:\\" + "r" * 57, PY="D:\\" + "p" * 67 + ".exe", RUNSM="E:\\" + "t" * 67, CACHEFULL="E:\\" + "c" * 67, V9="E:\\" + "v" * 67)
    assert len(env["REPO"]) == 60 and len(env["RUNSM"]) == len(env["CACHEFULL"]) == len(env["V9"]) == 70
    for st in ORC_STAGES:
        rc, ln, out = _dry(f"{st} {arm} {seed}", env)
        assert rc == 0 and len(ln) == 1, (st, rc, out[:300])
        n = len(ln[0]) + len(' >> "' + env["RUNSM"] + f'\\M2b_s{seed}_{arm}\\train.log" 2>&1')
        assert ln[0].count(env["RUNSM"]) >= 3 and ln[0].count(env["V9"]) >= 3 and n < 8191 // 2, f"{st}: {n} characters (limit 8,191; this is the longest line of the stage)"
    t = _train_argv(_dry1(f"trainorc {arm} {seed}", env))
    assert t[-1] == (str(seed) if arm == "orc" else "0") and _flags(t)["--init-ckpt"] == [[env["RUNSM"] + f"\\M2b_s{seed}\\ckpt_last.pt"]], "the long paths are the ones in the argv"


# =============================================================================================================================================
# (B) THE CHAIN: static facts, the predecessor wait RUN on a stubbed box, --dry-run
# =============================================================================================================================================
def _git_bash():
    for p in (os.environ.get("GIT_BASH"), "C:/Program Files/Git/bin/bash.exe", "C:/Program Files (x86)/Git/bin/bash.exe"):
        if p and Path(p).exists():
            return p
    return None


def _chain():
    c = CHAIN.read_text(encoding="utf-8")
    return c, "\n".join(x for x in c.splitlines() if not x.lstrip().startswith("#"))


def _at(text, needle, start=0):
    assert needle in text[start:], f"the anchor is missing: {needle!r}"
    return text.index(needle, start)


@needs_chain
def test_chain_is_lf_only_bash_syntax_clean_and_pins_the_banked_checkpoints():
    raw = CHAIN.read_bytes()
    assert b"\r" not in raw, "a bash script with CR bytes does not run"
    raw.decode("utf-8")
    bash = _git_bash()
    if bash is not None and sys.platform == "win32":
        r = subprocess.run([bash, "-n", str(CHAIN)], capture_output=True, text=True, encoding="utf-8", timeout=60)
        assert r.returncode == 0, r.stderr
    c, _ = _chain()
    m = re.search(r"CKPT_SHA=\((\w{64}) (\w{64})\)", c)
    assert m and (m.group(1), m.group(2)) == (CKPT_SHA[0], CKPT_SHA[1])
    if (FULL / "MANIFEST.md").exists():
        man = (FULL / "MANIFEST.md").read_text(encoding="utf-8")
        assert f"`{CKPT_SHA[0]}`" in man and f"`{CKPT_SHA[1]}`" in man, "the chain's pins are the banked runs' (raw/full/MANIFEST.md)"
    if READ_P.exists():
        R = _load("orc_read_for_chain_pins", READ_P)
        assert R.BANKED_M2B_SHA == CKPT_SHA, "the reader pins the same two values"


@needs_chain
def test_chain_static_order_refusal_wait_atomic_marker_refresh_checkpoint_sha_all_four_gates_then_the_four_runs_nothing_conditional():
    c, code = _chain()
    assert 'RUNS=("orc 0" "orc0 0" "orc 1" "orc0 1")' in c, "the PREREG's order (section 5): ORC s0, ORC-0 s0, ORC s1, ORC-0 s1"
    # (1) the refusal comes first: before any directory is made, before the box is asked anything
    ref = _at(c, 'if [ -z "${AFTER_OUT:-}" ]; then')
    assert ref < _at(c, "boot() {") and ref < _at(c, "mkdir -p $E/runs $E/lines") and ref < _at(c, "gate_one() {")
    assert "AFTER_OUT=${AFTER_OUT:-" not in code and "AFTER_OUT=$L" not in code, "NO default predecessor log: a default path can already hold an OLD log's end marker"
    # (2) the main flow, in order
    main = c[_at(c, "waiting for the predecessor chain"):]
    order = ["until grep -q", "( set -C;", "BOOT0=$(boot)", "mkdir -p $E/runs $E/lines", "fetch -q origin", "ls-remote origin $BR", 'ckptsha_b "before the gates"', 'cp "$L/v8f_mini/runs/M2b_s$s/summary.json"',
             "launch telemetry_long", "v8f_guard.sh > ", 'for run in "${RUNS[@]}"; do set -- $run; gate_one $1 $2; done', 'for run in "${RUNS[@]}"; do set -- $run; runone $1 $2; done',
             'ckptsha_b "after the last run"', 'finish "PREREG_V8F_ORECOVER']
    idx = [_at(main, x) for x in order]
    assert idx == sorted(idx) and len(set(idx)) == len(idx), dict(zip(order, idx))
    # (3) inside gate_one: the gate, its pull, THEN the assertions, THEN the token (a token exists only after the assertions passed)
    g = c[_at(c, "gate_one() {"):_at(c, "runone() {")]
    go = ['queue "gate_$g" $GATE_POLLS gateorc $arm $s', 'pull "$gd/selfcheck.json"', "<<'EOF' || stop \"gate ORC assertions $g\"", "<<'EOF' || stop \"write the gate token $g\""]
    gi = [_at(g, x) for x in go]
    assert gi == sorted(gi), dict(zip(go, gi))
    # (4) inside runone: train -> sha -> eval -> sha -> pull -> the run record
    r = c[_at(c, "runone() {"):_at(c, "waiting for the predecessor chain")]
    ro = ['queue "t_$g" $TRAIN_POLLS trainorc $arm $s', "ckptsha_x $arm $s trained", 'queue "e_$g" $EVAL_POLLS evalorc $arm $s', "ckptsha_x $arm $s after_evals", 'pull "$g/$f" "$ART/$g"', "sha256sum",
          "<<'EOF' || stop \"run ORC record $g\"", "ZZORC-CONTROL-FAIL", "grep ZZGUARD-STOP"]
    ri = [_at(r, x) for x in ro]
    assert ri == sorted(ri), dict(zip(ro, ri))
    # (5) nothing is conditional on a read: one finish in the flow, the loops are not inside an if, no seed-1-only branch
    assert code.count('finish "PREREG_V8F_ORECOVER') == 1 and "s1_required" not in code and "DECISION" not in code and "CONTINUE" not in code
    assert not re.search(r"(?m)^\s*(if|\[\s).*(gate_one|runone)", main), "no run is skipped by a test"
    called = set(re.findall(r"\bqueue (?:\"[^\"]+\"|\w+) (?:\$\w+|\d+) (\w+)", c))
    assert called == set(ORC_STAGES), called
    if QUEUE.exists():
        assert called <= set(re.findall(r'(?m)^if /i "%~1"=="(\w+)" goto', _queue_text()))


@needs_chain
def test_chain_load_bearing_lines_are_present():
    c, code = _chain()
    for needle in ('( set -C; echo "v8F-mini PREREG_V8F_ORECOVER', "BOX2_BUSY.txt", 'grep -qF "\'^(ZZV8FMINI|" $B/v8f_guard.sh', "V8F_CHAIN_OUT=$L/chain_v8f_orc.out", 'echo "ZZV8FMINI-DONE ', 'echo "ZZV8FMINI-STOP ',
                   '[ "$g" = "${CKPT_SHA[$s]}" ]', '[ "$h" = "$h0" ] || stop', 'launch_gate_token.json', 'SHA256SUMS.txt', "ZZORC-CONTROL-FAIL", 'git -C $MIRROR cat-file -e "$T^{commit}"',
                   "git -C $MIRROR -c core.autocrlf=false archive", "sha256sum | cut -c1-64)", "md5", 'rsha "D:/tanitad/data/v9/'):
        assert needle in c, needle
    # finish and stop RELEASE the box before the marker is announced / the chain exits
    fin = c[_at(c, "finish() {"):_at(c, "gate_one() {")]
    assert _at(fin, "rm -f $B/BOX2_BUSY.txt") < _at(fin, 'echo "ZZV8FMINI-DONE ') < _at(fin, "exit 0")
    assert re.search(r'stop\(\) \{ echo "ZZV8FMINI-STOP \$\*"; echo "" > \$CUR; rm -f \$B/BOX2_BUSY.txt; exit 1; \}', c)
    # the key files are sha-checked against the tip, the v9 releases md5-checked, and the lever's presence in the tip is a pre-flight
    for needle in ('"--init-ckpt"', '"--ego-perturb-p"', "ego:ego_perturb_p", "data:init_ckpt", "ego_perturb_probe(model, probe, cfg, ego)", "b = ego.apply(b, step)", '"ego_perturb_n": ego_tot', "^class LRSchedule", "^class EgoPerturb"):
        assert needle in c, f"the pre-flight on the tip does not look for {needle}"
    assert "gateorc trainorc evalorc orcarm" in c and "orc_read.py" in c
    assert "ckpt_sha256.json" in c and "E=$L/v8f_orc" in c and "chain_v8f_orc.out" in c


def _copy_chain(tmp_path, mods=()):
    """The chain with its box-2 directory moved into tmp, ssh / scp turned into `false`, the mirror pointed nowhere and (optionally) other substitutions. Each substitution is asserted: a chain whose text moved
    must not fall through to the real box or the real mirror."""
    bash = _git_bash()
    if bash is None or sys.platform != "win32":
        pytest.skip("Git Bash on Windows is needed to run the chain")
    ship = tmp_path / "box2_ship"
    (ship / "logs").mkdir(parents=True)
    src = CHAIN.read_text(encoding="utf-8")
    for old, new in (("B=/c/Users/Admin/box2_ship;", f"B={ship.as_posix()};"), ('SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"', 'SSH="false"'),
                     ('SCP="scp -q -o BatchMode=yes -o ConnectTimeout=20"', 'SCP="false"'), ("MIRROR=/c/Users/Admin/tanitad-push; BR=agent/arch-inf-20260803\n", f"MIRROR={(tmp_path / 'no_mirror').as_posix()}; BR=none\n"), *mods):
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    code = "\n".join(x for x in src.splitlines() if not x.lstrip().startswith("#"))
    assert not re.search(r"(?<![\w$])(ssh|scp)\s", code), "no literal ssh / scp command is left: every call goes through $SSH / $SCP, which are `false` here"
    script = tmp_path / "chain_under_test.sh"
    script.write_bytes(src.encode("utf-8"))
    return bash, script, ship


def _run_chain(tmp_path, bash, script, extra_env=None, args=()):
    env = _env({"TMPDIR": tmp_path.as_posix(), "ORC_ART": (tmp_path / "art").as_posix()})
    for k in ("AFTER_OUT", "AFTER_DONE", "AFTER_STOP", "AFTER_GUARD"):
        env.pop(k, None)
    env.update(extra_env or {})
    r = subprocess.run([bash, str(script), *args], env=env, capture_output=True, text=True, timeout=120, encoding="utf-8", stdin=subprocess.DEVNULL)
    return r.returncode, r.stdout


@needs_chain
@win
def test_chain_refuses_to_start_without_after_out_and_a_default_is_not_an_option(tmp_path):
    bash, script, ship = _copy_chain(tmp_path)
    rc, out = _run_chain(tmp_path, bash, script)
    assert rc == 1 and out.startswith("ZZV8FMINI-STOP AFTER_OUT is not set") and "OLD log" in out, out
    assert not (ship / "BOX2_BUSY.txt").exists() and not (ship / "logs" / "v8f_orc").exists() and not (tmp_path / "art").exists(), "a refusal touches nothing"
    rc, out = _run_chain(tmp_path, bash, script, {"AFTER_OUT": ""})
    assert rc == 1 and "AFTER_OUT is not set" in out, "an EMPTY AFTER_OUT is not a predecessor either"
    # the artifact root must be reachable (a chain that trains for hours and then has nowhere to put it)
    rc, out = _run_chain(tmp_path, bash, script, {"AFTER_OUT": (tmp_path / "x.out").as_posix(), "ORC_ART": (tmp_path / "no_such_parent" / "art").as_posix()})
    assert rc == 1 and "nowhere to go" in out and not (ship / "BOX2_BUSY.txt").exists(), out


@needs_chain
@win
def test_chain_wait_loop_runs_end_to_end_waits_proceeds_on_done_stops_on_stop_and_takes_and_releases_the_marker(tmp_path):
    bash, script, ship = _copy_chain(tmp_path, [("  sleep 300\n", "  { echo WAITING; exit 77; }\n")])        # the wait would sleep 300 s: here it reports and leaves, rc 77
    pred = tmp_path / "pred.out"

    def run(lines, **env):
        pred.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
        (ship / "BOX2_BUSY.txt").unlink(missing_ok=True)
        return _run_chain(tmp_path, bash, script, {"AFTER_OUT": pred.as_posix(), **env})
    rc, out = run(["ZZV8FMINI-DONE 10-26 08:00 tip abc: done"])
    assert rc == 1 and "ZZV8FMINI-STOP box 2 unreachable" in out and "WAITING" not in out, out
    assert not (ship / "BOX2_BUSY.txt").exists(), "a chain that stops releases the marker it took"
    assert not (ship / "logs" / "v8f_orc").exists(), "the stubbed box was unreachable before anything was made"
    rc, out = run(["some log line", "waiting for ZZV8FMINI-DONE"])
    assert rc == 77 and "WAITING" in out, "no end marker yet: the chain WAITS (a marker quoted inside a line does not count: the test is anchored)"
    rc, out = run(["  ZZV8FMINI-DONE indented is not an end marker"])
    assert rc == 77 and "WAITING" in out
    rc, out = run(["ZZV8FMINI-STOP v8foacc10260603_t_M2b_s0_oacc rc=1"])
    assert rc == 1 and "ZZV8FMINI-STOP the predecessor stopped: ZZV8FMINI-STOP v8foacc10260603_t_M2b_s0_oacc rc=1" in out and "box 2 unreachable" not in out, out
    assert not (ship / "BOX2_BUSY.txt").exists(), "a chain that stopped while waiting never took the marker"
    rc, out = run(["ZZV8FMINI-DONE ok"], AFTER_DONE="MYDONE")
    assert rc == 77 and "WAITING" in out, "AFTER_DONE overrides the marker it waits for"
    rc, out = run(["MYDONE ok"], AFTER_DONE="MYDONE")
    assert rc == 1 and "box 2 unreachable" in out
    rc, out = run(["MYSTOP x"], AFTER_STOP="MYSTOP")
    assert rc == 1 and "the predecessor stopped: MYSTOP x" in out
    # a busy marker that is already there stops the chain, and is not removed (it is somebody else's)
    pred.write_bytes(b"ZZV8FMINI-DONE ok\n")
    (ship / "BOX2_BUSY.txt").write_text("another chain", encoding="utf-8")
    rc, out = _run_chain(tmp_path, bash, script, {"AFTER_OUT": pred.as_posix()})
    assert rc == 1 and "busy marker present: another chain" in out and (ship / "BOX2_BUSY.txt").read_text(encoding="utf-8") == "another chain", out


@needs_chain
@win
def test_chain_dry_run_prints_every_command_the_shas_and_the_token_schema_and_ships_nothing(tmp_path):
    bash = _git_bash()
    if bash is None:
        pytest.skip("Git Bash on Windows is needed to run the chain's --dry-run")
    env = _env({"V8F_QUEUE": str(QUEUE), "TMPDIR": tmp_path.as_posix(), "ORC_ART": (tmp_path / "art").as_posix()})
    for k in ("AFTER_OUT", "AFTER_DONE", "AFTER_STOP"):
        env.pop(k, None)
    r = subprocess.run([bash, str(CHAIN), "--dry-run"], env=env, capture_output=True, text=True, timeout=180, encoding="utf-8", stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stdout + r.stderr
    out = r.stdout.splitlines()
    assert not (tmp_path / "art").exists(), "a dry run makes nothing"
    assert [x[3:].split(":")[0] for x in out if x.startswith("== M2b_s")] == [f"M2b_s{s}_{a}" for a, s in RUNS], "the four runs, in the PREREG's order"
    assert "wait for <UNSET: a real run REFUSES to start without AFTER_OUT>: ^ZZV8FMINI-DONE (stop on ^ZZV8FMINI-STOP)" in out
    for s in (0, 1):
        assert f"M2b_s{s} D:/tanitad/runs/v8f_mini_full/M2b_s{s}/ckpt_last.pt {CKPT_SHA[s]}" in out
    want = []                                       # the chain's printed lines ARE the queue's own dry-run lines at box 2's paths
    for a, s in RUNS:
        for st in ORC_STAGES:
            want.append(_dry1(f"{st} {a} {s}"))
    assert [x for x in out if x.startswith('"') and ".py" in x] == want and len(want) == 12
    assert sum(x.endswith(": SAME") for x in out if x.startswith("-- gate argv")) == 4
    shas = [x.split(": ")[1] for x in out if x.startswith("-- token argv_sha256")]
    assert len(shas) == 4 and len(set(shas)) == 4
    if READ_P.exists():
        R = _load("orc_read_for_chain_dry", READ_P)
        for (a, s), sha in zip(RUNS, shas):
            assert sha == R.gate_argv_sha256(_train_argv(_dry1(f"trainorc {a} {s}"))), "the printed token binding is the reader's own function of the TRAINING argv"
            assert sha != R.gate_argv_sha256(_train_argv(_dry1(f"gateorc {a} {s}")))
    tok = next(x for x in out if x.startswith('{"kind": "launch-gate-token"'))
    for need in ('"verdict": "PASS"', '"binding"', '"commit"', '"argv_sha256"', '"data_sha256"', "THE TRAINING ARGV"):
        assert need in tok, need
    r2 = subprocess.run([bash, str(CHAIN), "--dry-run"], env=dict(env, AFTER_OUT="/x/pred.out", AFTER_DONE="D1"), capture_output=True, text=True, timeout=180, encoding="utf-8", stdin=subprocess.DEVNULL)
    assert "wait for /x/pred.out: ^D1 (stop on ^ZZV8FMINI-STOP)" in r2.stdout.splitlines()
    # the dry-run block can reach neither box 2 nor the job machinery
    c, code = _chain()
    dry = code[_at(code, 'if [ "${1:-}" = "--dry-run" ]; then'):_at(code, "# ---- the predecessor is NAMED")] if "# ---- the predecessor is NAMED" in code else c[_at(c, 'if [ "${1:-}" = "--dry-run" ]; then'):_at(c, "# ---- the predecessor is NAMED")]
    for bad in ("$SCP", "$SSH", "ssh ", "scp ", "mk_v8f_job", "BOX2_BUSY"):
        assert bad not in dry, f"the dry-run block mentions {bad}"
    assert not re.search(r"(^|[;&|{(]\s*)(launch|queue|waitrc|pull|boot)\s", dry, re.M), "the dry-run block CALLS a box-2 helper (a message that names one is fine)"


# =============================================================================================================================================
# (C) THE GATE ASSERTIONS and (D) THE TOKEN, run against the REAL trainer's own records (synthetic data, the registered fine-tune argv)
# =============================================================================================================================================
SYN_A = ["--synthetic", "--batch", "8", "--d", "32", "--n-q", "2", "--log-every", "1", "--ckpt-every", "100", "--threads", "1", "--kin-mode", "yaw"]


def _ft(ck, arm, seed, steps="3000"):
    return ["--init-ckpt", str(ck), "--steps", steps, "--lr", "1e-4", "--warmup", "100", "--p-gt-start", "0.2", "--p-gt-end", "0.2", "--ego-perturb-p", "0.3" if arm == "orc" else "0",
            *(["--ego-perturb-vscale", "0.5", "1.0", "--ego-perturb-vmax", "8.0", "--ego-perturb-adrop", "1.5", "--ego-perturb-yawrate-std", "2.0", "--ego-perturb-seed", str(seed)] if arm == "orc" else [])]


def _argv(out, ck, arm, seed, steps="3000", gate=False):
    return ["--arm", "M2b", "--seed", str(seed), "--grid", "pool4x10", "--out", str(out), *SYN_A, *_ft(ck, arm, seed, steps), *(["--selfcheck-only"] if gate else [])]


def _line(argv, script="v8f_mini_train.py"):
    q = lambda t: f'"{t}"' if (" " in t or "\\" in t) else t               # noqa: E731
    return f'"D:\\tanitad\\venv\\Scripts\\python.exe" "D:\\tanitad\\repo\\stack\\scripts\\{script}" ' + " ".join(q(t) for t in argv)


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    """The REAL trainer (v8f_mini_train.main, synthetic data) writes the records the chain asserts: for each seed one source checkpoint `M2b_s<seed>/ckpt_last.pt`, then for ORC and ORC-0 the registered fine-tune's
    GATE (`--selfcheck-only`, steps 3000) and, for seed 0, a 6-step TRAINING run. NOTHING is patched: ORC-0 is the Amendment 3 argv (`--ego-perturb-p 0`, none of the five law flags), which the trainer's own gate
    ACCEPTS; `test_orc0_gates_pass_*` pins that, and `test_a_law_flag_*` pins that it refuses the old shape."""
    pytest.importorskip("torch")
    for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    import torch
    torch.set_num_threads(1)
    import v8f_mini_train as TR
    from tanitad.data import v8f_mini_data as D
    root = tmp_path_factory.mktemp("orc_synth")
    out = {"root": root, "gate": {}, "run": {}, "ck": {}, "sha": {}, "TR": TR, "D": D}
    for seed in (0, 1):
        src = root / "runsm" / f"M2b_s{seed}"
        assert TR.main(["--arm", "M2b", "--seed", str(seed), "--grid", "pool4x10", "--out", str(src), *SYN_A, "--steps", "1", "--ckpt-every", "1"]) == 0
        ck = src / "ckpt_last.pt"
        out["ck"][seed], out["sha"][seed] = ck, hashlib.sha256(ck.read_bytes()).hexdigest()
        for arm in ("orc", "orc0"):
            gate = root / "runsm" / f"gate_M2b-{arm}_s{seed}"
            assert TR.main(_argv(gate, ck, arm, seed, gate=True)) == 0, f"the trainer's own gate must ACCEPT the registered {arm} argv at seed {seed}"
            out["gate"][(arm, seed)] = json.loads((gate / "selfcheck.json").read_text(encoding="utf-8"))
            if seed == 0:
                run = root / "runsm" / f"M2b_s{seed}_{arm}"
                assert TR.main(_argv(run, ck, arm, seed, steps="6")) == 0
                out["run"][(arm, seed)] = run
    return out


B_SUMMARY = {s: json.loads((FULL / "runs" / f"M2b_s{s}" / "summary.json").read_text(encoding="utf-8")) if (FULL / "runs" / f"M2b_s{s}" / "summary.json").exists() else None for s in (0, 1)}


def _heredoc(pattern):
    c = CHAIN.read_text(encoding="utf-8")
    m = re.search(pattern, c, re.S)
    assert m, f"the chain has no python heredoc matching {pattern!r}"
    return m.group(1)


GATE_BODY = lambda: _heredoc(r"<<'EOF' \|\| stop \"gate ORC assertions[^\"]*\"\n(.*?)\nEOF\n")          # noqa: E731
TOKEN_BODY = lambda: _heredoc(r"<<'EOF' \|\| stop \"write the gate token[^\"]*\"\n(.*?)\nEOF\n")      # noqa: E731
RECORD_BODY = lambda: _heredoc(r"<<'EOF' \|\| stop \"run ORC record[^\"]*\"\n(.*?)\nEOF\n")           # noqa: E731
REACH_BODY = lambda: _heredoc(r"<<'EOF' \|\| echo \"ZZORC-CONTROL-FAIL[^\"]*\"\n(.*?)\nEOF\n")        # noqa: E731


def _py(body, args, timeout=60):
    r = subprocess.run([sys.executable, "-", *map(str, args)], input=body, capture_output=True, text=True, encoding="utf-8", timeout=timeout)
    return r.returncode, r.stdout + r.stderr


def good_selfcheck(synth, arm, seed):
    """The real trainer's gate record with the production NUMBERS laid over what the synthetic world cannot know (clip counts, B's data fingerprint, parameter counts, the banked init sha256 and path)."""
    sc = copy.deepcopy(synth["gate"][(arm, seed)])
    b = B_SUMMARY[seed]
    i, lv = sc["info"], sc["levers"]
    i.update(n_clips=4512, n_excluded=60, n_excluded_declared=60, excluded_in_train=0, excluded_sha256=HOLD, train_sha256=b["train_sha256"], n_windows_all_clips=b["n_windows"], n_windows=b["n_windows"])
    lv["params"], lv["n_params_trainable"] = dict(b["levers"]["params"]), b["levers"]["n_params_trainable"]
    lv["w"].update({k: b["levers"]["w"][k] for k in ("w_pen", "w_pen_self", "w_wm")})
    lv["init_ckpt"].update(sha256=CKPT_SHA[seed], path=rf"D:\tanitad\runs\v8f_mini_full\M2b_s{seed}\ckpt_last.pt")
    i["init_ckpt"] = copy.deepcopy(lv["init_ckpt"])
    return sc


def _lines(arm, seed):
    return _dry1(f"gateorc {arm} {seed}"), _dry1(f"trainorc {arm} {seed}")


def _gate(tmp_path, sc, arm="orc", seed=0, gl=None, tl=None, banked=None, steps="3000", windows="306566", nclips="4512", hold=HOLD):
    g, t = _lines(arm, seed)
    for name, txt in (("sc.json", json.dumps(sc)), ("b.json", json.dumps(B_SUMMARY[seed])), ("g.txt", gl if gl is not None else g), ("t.txt", tl if tl is not None else t)):
        (tmp_path / name).write_text(txt, encoding="utf-8")
    return _py(GATE_BODY(), [tmp_path / "sc.json", hold, tmp_path / "b.json", tmp_path / "g.txt", tmp_path / "t.txt", arm, seed, "0.3" if arm == "orc" else "0", banked or CKPT_SHA[seed], steps, windows, nclips])


def _set(path, value):
    def f(sc):
        d = sc
        for k in path[:-1]:
            d = d[k]
        if value is _DEL:
            d.pop(path[-1])
        else:
            d[path[-1]] = value
    return f


_DEL = object()
ORC_GATE_CASES = [
    ("declared_vs_built", _set(("declared_vs_built",), "FAIL"), "declared_vs_built"),
    ("live not pass", _set(("live", "pass"), False), "live"),
    ("prior control moved", _set(("live", "control_prior_grad_sum"), 1.0), "live"),
    ("a weight dead", _set(("live", "dead_params"), ["head_o.2.weight"]), "G-LIVE dead set"),
    ("a weight declared dead", _set(("live", "declared_dead"), ["e_code.weight"]), "G-LIVE dead set"),
    ("no ego probe", _set(("ego_perturb_probe",), _DEL), "ego_perturb_probe None"),
    ("ego probe not pass", _set(("ego_perturb_probe", "pass"), False), "ego_perturb_probe"),
    ("ego probe says off for ORC", _set(("ego_perturb_probe", "ego_perturb"), "off"), "the ego probe ran for p"),
    ("ego probe ran for another p", _set(("ego_perturb_probe", "p_declared"), 0.4), "the ego probe ran for p 0.4"),
    ("no eligible window in the probe", _set(("ego_perturb_probe", "n_selected_at_p1"), 0), "saw no eligible window"),
    ("the perturbed batch moves no loss", _set(("ego_perturb_probe", "loss_diff"), 0.0), "does not reach the loss"),
    ("the perturbed batch moves no gradient", _set(("ego_perturb_probe", "grad_diff_all"), 0.0), "does not reach the loss"),
    ("repeat loss control", _set(("ego_perturb_probe", "control_repeat_loss_diff"), 1e-9), "control_repeat_loss_diff"),
    ("repeat grad control", _set(("ego_perturb_probe", "control_repeat_grad_diff"), 1e-9), "control_repeat_grad_diff"),
    ("p0 loss control", _set(("ego_perturb_probe", "control_p0_loss_diff"), 1e-9), "control_p0_loss_diff"),
    ("p0 grad control", _set(("ego_perturb_probe", "control_p0_grad_diff"), 1e-9), "control_p0_grad_diff"),
    ("prior grad control", _set(("ego_perturb_probe", "control_prior_grad_diff"), 1e-9), "control_prior_grad_diff"),
    ("p0 is not the identity", _set(("ego_perturb_probe", "control_p0_same_object"), False), "p = 0 is not the identity"),
    ("wm auxiliary moved", _set(("ego_perturb_probe", "control_wm_grad_diff"), 1e-9), "world-model auxiliary"),
    ("G-LIVE on the perturbed batch fails", _set(("ego_perturb_probe", "live_perturbed", "pass"), False), "PERTURBED batch"),
    ("perturbed prior control moved", _set(("ego_perturb_probe", "live_perturbed", "control_prior_grad_sum"), 1.0), "PERTURBED batch"),
    ("built p", _set(("levers", "ego_perturb", "ego_perturb_p"), 0.0), "built ego_perturb ego_perturb_p"),
    ("built vscale", _set(("levers", "ego_perturb", "ego_perturb_vscale"), [0.6, 1.0]), "ego_perturb_vscale"),
    ("built vmax", _set(("levers", "ego_perturb", "ego_perturb_vmax"), 6.0), "ego_perturb_vmax"),
    ("built adrop", _set(("levers", "ego_perturb", "ego_perturb_adrop"), 1.0), "ego_perturb_adrop"),
    ("built yawrate std", _set(("levers", "ego_perturb", "ego_perturb_yawrate_std"), 0.0), "ego_perturb_yawrate_std"),
    ("built perturbation seed", _set(("levers", "ego_perturb", "ego_perturb_seed"), 1), "ego_perturb_seed"),
    ("built kin mode", _set(("levers", "ego_perturb", "kin_mode"), "steer"), "built ego_perturb kin_mode"),
    ("ego levers unrecorded", _set(("levers", "ego_perturb"), _DEL), "built ego_perturb ego_perturb_p"),
    ("init is not the banked checkpoint", _set(("levers", "init_ckpt", "sha256"), "0" * 64), "does not start from the banked"),
    ("info.init_ckpt disagrees", _set(("info", "init_ckpt", "n_tensors"), 1), "disagree"),
    ("init not applied", _set(("levers", "init_ckpt", "applied"), False), "not applied / read back"),
    ("init read back with mismatches", _set(("levers", "init_ckpt", "n_mismatch_after_load"), 3), "not applied / read back"),
    ("init path of the other seed", _set(("levers", "init_ckpt", "path"), r"D:\tanitad\runs\v8f_mini_full\M2b_s1\ckpt_last.pt"), "is not .../M2b_s0/ckpt_last.pt"),
    ("steps", _set(("steps_total",), 9000), "steps 9000"),
    ("warmup", _set(("levers", "lr_schedule", "warmup"), 200), "built lr schedule"),
    ("peak lr", _set(("levers", "lr_schedule", "lr"), 3e-4), "built lr schedule"),
    ("schedule total", _set(("levers", "lr_schedule", "total"), 3001), "built lr schedule"),
    ("p_gt start", _set(("levers", "p_gt_start"), 1.0), "built p_gt"),
    ("p_gt end", _set(("levers", "p_gt_end"), 0.0), "built p_gt"),
    ("n_clips", _set(("info", "n_clips"), 4000), "clips / holdout counts"),
    ("n_excluded", _set(("info", "n_excluded"), 59), "clips / holdout counts"),
    ("excluded_in_train", _set(("info", "excluded_in_train"), 1), "clips / holdout counts"),
    ("holdout", _set(("info", "excluded_sha256"), "0" * 64), "not the RA-1 holdout"),
    ("train list", _set(("info", "train_sha256"), "d" * 64), "TRAIN list"),
    ("kin_mode", _set(("info", "kin_mode"), "steer"), "kin_mode"),
    ("wm", _set(("levers", "wm"), "a"), "levers wm"),
    ("consistency", _set(("levers", "consistency"), False), "levers wm"),
    ("cons_absent", _set(("levers", "cons_absent"), "zero"), "levers wm"),
    ("o_interface", _set(("levers", "o_interface"), "cut"), "o_interface"),
    ("code_mix", _set(("levers", "code_mix"), "kind"), "code_mix"),
    ("w_acc", _set(("levers", "w_acc"), 0.3), "w_acc"),
    ("w_pen", _set(("levers", "w", "w_pen"), 0.6), "w_pen"),
    ("w_wm", _set(("levers", "w", "w_wm"), 2.0), "w_wm"),
    ("w_tac", _set(("levers", "w", "w_tac"), 2.0), "w_tac / w_cons"),
    ("w_cons", _set(("levers", "w", "w_cons"), 1.0), "w_tac / w_cons"),
    ("window set is not B's", _set(("info", "n_windows_all_clips"), 279494), "the window set is not B's"),
    ("hist", _set(("levers", "hist"), 10), "built hist 10 != 4"),
    ("hmax", _set(("levers", "hmax"), 10), "built hmax 10 != 16"),
    ("window_start", _set(("levers", "window_start"), 9), "built window_start 9 != 3"),
    ("window list starts at 9", _set(("info", "window_t_min"), 9), "window start (asked"),
    ("one parameter too many", lambda s: s["levers"].__setitem__("n_params_trainable", s["levers"]["n_params_trainable"] + 1), "n_params_trainable"),
    ("a group grew", lambda s: s["levers"]["params"].__setitem__("T", s["levers"]["params"]["T"] + 5), "parameter group T"),
    ("lru unrecorded", _set(("info", "lru"), _DEL), "info.lru"),
]


@needs_chain
@needs_banked
@win
def test_gate_assertions_pass_the_registered_gate_of_every_run_and_refuse_every_single_field_corruption_of_orc(synth, tmp_path):
    """The chain's gate block is RUN against selfcheck.json records written by the REAL trainer: all four runs pass; each single-field corruption of an ORC record exits 1 and names the broken field."""
    for arm, seed in RUNS:
        sc = good_selfcheck(synth, arm, seed)
        rc, out = _gate(tmp_path, sc, arm, seed)
        assert rc == 0 and "ORC gate: PASS" in out and f"{arm} s{seed}" in out and f"init {CKPT_SHA[seed][:12]}" in out, (arm, seed, out)
    for name, mod, msg in ORC_GATE_CASES:
        sc = good_selfcheck(synth, "orc", 0)
        mod(sc)
        rc, out = _gate(tmp_path, sc)
        assert rc != 0 and msg in out, (name, rc, out[-600:])
    # the numbers the chain holds are the ones compared with
    rc, out = _gate(tmp_path, good_selfcheck(synth, "orc", 0), steps="17472")
    assert rc == 1 and "steps 3000 != the PREREG's 17472" in out
    rc, out = _gate(tmp_path, good_selfcheck(synth, "orc", 0), windows="306565")
    assert rc == 1 and "the PREREG's 306565" in out
    rc, out = _gate(tmp_path, good_selfcheck(synth, "orc", 0), banked=CKPT_SHA[1])
    assert rc == 1 and "does not start from the banked" in out, "the other seed's checkpoint is not this run's base"
    rc, out = _gate(tmp_path, good_selfcheck(synth, "orc", 0), hold="1" * 64)
    assert rc == 1 and "not the RA-1 holdout" in out


@needs_chain
@needs_banked
@win
def test_gate_assertions_for_orc0_read_the_probe_as_off_and_refuse_a_perturbation_that_leaked_into_the_control(synth, tmp_path):
    for seed in (0, 1):
        rc, out = _gate(tmp_path, good_selfcheck(synth, "orc0", seed), "orc0", seed)
        assert rc == 0 and "ORC gate: PASS" in out
    sc = good_selfcheck(synth, "orc0", 0)
    sc["ego_perturb_probe"] = copy.deepcopy(good_selfcheck(synth, "orc", 0)["ego_perturb_probe"])
    rc, out = _gate(tmp_path, sc, "orc0", 0)
    assert rc == 1 and "should read 'off'" in out, out
    sc = good_selfcheck(synth, "orc0", 0)
    sc["levers"]["ego_perturb"]["ego_perturb_p"] = 0.3
    rc, out = _gate(tmp_path, sc, "orc0", 0)
    assert rc == 1 and "built ego_perturb ego_perturb_p 0.3 != 0.0" in out, out
    sc = good_selfcheck(synth, "orc0", 0)
    rc, out = _gate(tmp_path, sc, "orc0", 0, tl=_lines("orc", 0)[1], gl=_lines("orc", 0)[0])
    assert rc == 1 and "does not carry --ego-perturb-p 0 exactly once" in out and "names --ego-perturb-yawrate-std on ORC-0" in out, "ORC-0's gate record with ORC's lines (the arms swapped) is refused"
    # Amendment 3: the BUILT law values of ORC-0 are the trainer's DEFAULTS (vmax 8.0 = ORC's, so the eligible denominator is the same); a built value moved is refused, each of the five
    for seed in (0, 1):
        sc = good_selfcheck(synth, "orc0", seed)
        ep = sc["levers"]["ego_perturb"]
        assert (ep["ego_perturb_vscale"], ep["ego_perturb_vmax"], ep["ego_perturb_adrop"], ep["ego_perturb_yawrate_std"], ep["ego_perturb_seed"]) == ([0.5, 1.0], 8.0, 1.5, 0.0, 0), "the REAL built values are the defaults"
        for k, v in (("ego_perturb_vscale", [0.6, 1.0]), ("ego_perturb_vmax", 6.0), ("ego_perturb_adrop", 1.0), ("ego_perturb_yawrate_std", 2.0), ("ego_perturb_seed", seed + 1)):
            bad = copy.deepcopy(sc)
            bad["levers"]["ego_perturb"][k] = v
            rc, out = _gate(tmp_path, bad, "orc0", seed)
            assert rc == 1 and f"built ego_perturb {k}" in out, (k, out)
    # ... and a law flag NAMED on either ORC-0 line is refused whatever its value (the trainer's guard refuses only non-default values; the chain closes the rest)
    g0, t0 = _lines("orc0", 0)
    for fl, val in (("--ego-perturb-vscale", "0.5 1.0"), ("--ego-perturb-vmax", "8.0"), ("--ego-perturb-adrop", "1.5"), ("--ego-perturb-yawrate-std", "2.0"), ("--ego-perturb-seed", "0")):
        sc = good_selfcheck(synth, "orc0", 0)
        rc, out = _gate(tmp_path, sc, "orc0", 0, gl=g0.replace(" --selfcheck-only", f" {fl} {val} --selfcheck-only"), tl=t0 + f" {fl} {val}")
        assert rc == 1 and f"names {fl} on ORC-0" in out, (fl, out[-500:])
        rc, out = _gate(tmp_path, sc, "orc0", 0, tl=t0 + f" {fl} {val}")
        assert rc == 1 and f"names {fl} on ORC-0" in out
    sc = good_selfcheck(synth, "orc", 0)
    rc, out = _gate(tmp_path, sc, "orc", 0, tl=_lines("orc0", 0)[1], gl=_lines("orc0", 0)[0])
    assert rc == 1 and "does not carry --ego-perturb-p 0.3 exactly once" in out


def _drop(line, flag, n=1):
    t = line.split(" ")
    k = t.index(flag)
    return " ".join(t[:k] + t[k + 1 + n:])


@needs_chain
@needs_banked
@win
def test_gate_assertions_refuse_every_way_the_queues_printed_lines_can_be_wrong(synth, tmp_path):
    sc = good_selfcheck(synth, "orc", 0)
    g, t = _lines("orc", 0)
    cases = (("no --init-ckpt", g, _drop(t, "--init-ckpt"), "--init-ckpt"),
             ("init of the other seed", g, t.replace(r"M2b_s0\ckpt_last.pt", r"M2b_s1\ckpt_last.pt"), "is not .../M2b_s0/ckpt_last.pt"),
             ("p other", g, t.replace("--ego-perturb-p 0.3", "--ego-perturb-p 0.31"), "does not carry --ego-perturb-p 0.3 exactly once"),
             ("p missing", g, _drop(t, "--ego-perturb-p"), "does not carry --ego-perturb-p"),
             ("p twice", g, t + " --ego-perturb-p 0.3", "does not carry --ego-perturb-p 0.3 exactly once"),
             ("yaw std other", g, t.replace("--ego-perturb-yawrate-std 2.0", "--ego-perturb-yawrate-std 3.0"), "--ego-perturb-yawrate-std"),
             ("a law flag missing on ORC: yaw std", g, _drop(t, "--ego-perturb-yawrate-std"), "does not carry --ego-perturb-yawrate-std 2.0 exactly once"),
             ("a law flag missing on ORC: vscale", g, _drop(t, "--ego-perturb-vscale", 2), "does not carry --ego-perturb-vscale 0.5 1.0 exactly once"),
             ("a law flag missing on ORC: vmax", g, _drop(t, "--ego-perturb-vmax"), "does not carry --ego-perturb-vmax 8.0 exactly once"),
             ("a law flag missing on ORC: adrop", g, _drop(t, "--ego-perturb-adrop"), "does not carry --ego-perturb-adrop 1.5 exactly once"),
             ("a law flag missing on ORC: seed", g, _drop(t, "--ego-perturb-seed"), "does not carry --ego-perturb-seed 0 exactly once"),
             ("vscale other", g, t.replace("--ego-perturb-vscale 0.5 1.0", "--ego-perturb-vscale 0.5 0.9"), "--ego-perturb-vscale"),
             ("perturbation seed other", g, t.replace("--ego-perturb-seed 0", "--ego-perturb-seed 1"), "--ego-perturb-seed"),
             ("steps missing", g, _drop(t, "--steps"), "--steps"), ("lr other", g, t.replace("--lr 1e-4", "--lr 3e-4"), "--lr"), ("warmup other", g, t.replace("--warmup 100", "--warmup 200"), "--warmup"),
             ("p_gt not held", g, t.replace("--p-gt-start 0.2", "--p-gt-start 1.0"), "--p-gt-start"), ("seed other", g, t.replace("--seed 0", "--seed 1"), "--seed"),
             ("grid other", g, t.replace("--grid pool4x10", "--grid pool8x20"), "--grid"), ("arm other", g, t.replace("--arm M2b", "--arm M1"), "--arm"),
             ("a code-mix lever beside", g, t + " --code-mix kind", "sets --code-mix"), ("a w-acc lever beside", g, t + " --w-acc 0.3", "sets --w-acc"), ("a hist lever beside", g, t + " --hist 10", "sets --hist"),
             ("the train line carries --selfcheck-only", g, t + " --selfcheck-only", "TRAIN line carries --selfcheck-only"),
             ("the gate line has no --selfcheck-only", g.replace(" --selfcheck-only", ""), t, "--selfcheck-only exactly once"),
             ("the gate line differs from the train line", g.replace("--lr 1e-4", "--lr 2e-4"), t, "--lr"),
             ("the gate line drops a flag", _drop(g, "--warmup"), t, "--warmup"))
    for name, gl, tl, msg in cases:
        rc, out = _gate(tmp_path, sc, gl=gl, tl=tl)
        assert rc == 1 and msg in out, (name, rc, out[-500:])
    # the gate/train identity is its own clause (the lines each carry the registered flags)
    rc, out = _gate(tmp_path, sc, gl=g + " --exclude 0:1", tl=t)
    assert rc == 1 and "the gate argv is not the training argv" in out, out


def _gate_argv_sha256_of_chain():
    """The chain's own `gate_argv_sha256` (the token heredoc), exec'd out of the heredoc text."""
    body = TOKEN_BODY()
    src = re.search(r"(def gate_argv_sha256\(argv\):.*?\n    return [^\n]+\n)", body, re.S).group(1)
    ns = {"hashlib": hashlib, "json": json}
    exec(src, ns)                                                                  # noqa: S102
    return ns["gate_argv_sha256"]


@needs_chain
@needs_reader
def test_the_chains_token_hash_is_the_readers_gate_argv_sha256_on_every_kind_of_argv():
    R = _load("orc_read_for_hash", READ_P)
    f = _gate_argv_sha256_of_chain()
    argvs = [["--arm", "M2b", "--seed", "0"], [], ["--out", r"D:\tanitad\runs\x y\z", "--ego-perturb-vscale", "0.5", "1.0"], ["--a", "ü", "--b", 'q"uote'], _train_argv(_dry1("trainorc orc 1")) if sys.platform == "win32" and QUEUE.exists() else ["--x"]]
    for a in argvs:
        assert f(a) == R.gate_argv_sha256(a), a
    assert f(argvs[0]) != f(argvs[0][::-1]), "the hash is of the ORDERED array"


def _token_inputs(tmp_path, sc, gl, tl, commit="a" * 40, run="M2b_s0_orc"):
    (tmp_path / "sc.json").write_text(json.dumps(sc), encoding="utf-8")
    (tmp_path / "g.txt").write_text(gl, encoding="utf-8")
    (tmp_path / "t.txt").write_text(tl, encoding="utf-8")
    keys = json.dumps({"stack/scripts/v8f_mini_train.py": "b" * 64})
    return [tmp_path / "sc.json", tmp_path / "tok.json", commit, "c" * 64, keys, tmp_path / "g.txt", tmp_path / "t.txt", run]


@needs_chain
@needs_reader
@win
def test_the_token_is_accepted_by_the_readers_g_token_against_the_real_runs_argv_json_and_binds_the_training_argv_not_the_gates(synth, tmp_path):
    R = _load("orc_read_for_token", READ_P)
    for arm in ("orc", "orc0"):
        run = synth["run"][(arm, 0)]
        sc = synth["gate"][(arm, 0)]                                              # the REAL gate record of this synthetic fine-tune; its argv_sha is the real gate argv's
        gate_argv = _argv(run.parent / f"gate_M2b-{arm}_s0", synth["ck"][0], arm, 0, gate=True)
        train_argv = json.loads((run / "argv.json").read_text(encoding="utf-8"))["argv"]
        assert train_argv == _argv(run, synth["ck"][0], arm, 0, steps="6"), "the run's argv.json is the argv this test built (sys.argv[1:])"
        args = _token_inputs(tmp_path, sc, _line(gate_argv), _line(train_argv), run=f"M2b_s0_{arm}")
        rc, out = _py(TOKEN_BODY(), args)
        assert rc == 0 and "gate token written" in out, out
        tok = json.loads((tmp_path / "tok.json").read_text(encoding="utf-8"))
        assert tok["kind"] == "launch-gate-token" and tok["verdict"] == "PASS" and tok["binding"]["commit"] == "a" * 40 and tok["binding"]["data_sha256"] == sc["info"]["train_sha256"]
        assert tok["binding"]["argv_sha256"] == R.gate_argv_sha256(train_argv) != R.gate_argv_sha256(gate_argv), "bound to the TRAINING argv, not the gate's"
        assert tok["gate"]["argv_sha256"] == R.gate_argv_sha256(gate_argv) and tok["gate"]["argv_sha16"] == sc["argv_sha"] and tok["init_ckpt_sha256"] == sc["levers"]["init_ckpt"]["sha256"]
        r = R.train_records({"train_dir": str(run)}, None)                        # the reader's own loader on the real run directory
        ok, det = R.g_token(f"{arm}_s0", r, tok)
        assert ok is True, det
        assert det["commit"] == "a" * 40 and det["argv_sha256"] == tok["binding"]["argv_sha256"]
        # each way the token can fail the reader: another argv, another commit, no data fingerprint, a verdict that is not PASS, another kind
        ok, _ = R.g_token("t", {"argv": gate_argv}, tok)
        assert ok is False, "the gate's argv is not the run's argv"
        for mod in (lambda t: t["binding"].__setitem__("commit", "a" * 39), lambda t: t["binding"].__setitem__("data_sha256", "x"), lambda t: t.__setitem__("verdict", "FAIL"),
                    lambda t: t.__setitem__("kind", "other"), lambda t: t["binding"].__setitem__("argv_sha256", "0" * 64)):
            bad = copy.deepcopy(tok)
            mod(bad)
            assert R.g_token("t", r, bad)[0] is False
    # the token is never written for an argv the gate did not run, a gate that did not pass, or a malformed commit / data fingerprint
    sc = synth["gate"][("orc", 0)]
    gate_argv = _argv(synth["run"][("orc", 0)].parent / "gate_M2b-orc_s0", synth["ck"][0], "orc", 0, gate=True)
    train_argv = _argv(synth["run"][("orc", 0)], synth["ck"][0], "orc", 0, steps="6")
    for name, kw, msg in (("another gate argv", dict(gl=_line(gate_argv[:-1] + ["--p-gt-end", "0.3", "--selfcheck-only"])), "the gate ran an argv other than the queue's printed gate line"),
                          ("gate not PASS", dict(sc=dict(sc, declared_vs_built="FAIL")), "not a PASS"), ("short commit", dict(commit="a" * 39), "not 40-hex"),
                          ("non-hex commit", dict(commit="g" * 40), "not 40-hex"), ("data fingerprint", dict(sc=dict(sc, info=dict(sc["info"], train_sha256="ab"))), "data_sha256"),
                          ("selfcheck-only in the training line", dict(tl=_line(train_argv + ["--selfcheck-only"])), "carries --selfcheck-only")):
        args = _token_inputs(tmp_path, kw.get("sc", sc), kw.get("gl", _line(gate_argv)), kw.get("tl", _line(train_argv)), commit=kw.get("commit", "a" * 40))
        (tmp_path / "tok.json").unlink(missing_ok=True)
        rc, out = _py(TOKEN_BODY(), args)
        assert rc == 1 and msg in out and not (tmp_path / "tok.json").exists(), (name, rc, out)


# =============================================================================================================================================
# (E) THE RUN RECORD and the PERTURBATION-REACHED control
# =============================================================================================================================================
def _rec_run(synth, arm, tmp_path, mod=None, steps="6", banked=None, line_mod=None, argv_mod=None):
    run = synth["run"][(arm, 0)]
    d = tmp_path / f"rec_{arm}"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir()
    for f in ("argv.json", "summary.json", "selfcheck.json"):
        shutil.copy(run / f, d / f)
    argv = _argv(run, synth["ck"][0], arm, 0, steps="6")
    if argv_mod:
        a = json.loads((d / "argv.json").read_text(encoding="utf-8"))
        argv_mod(a)
        (d / "argv.json").write_text(json.dumps(a), encoding="utf-8")
    if mod:
        mod(d)
    (tmp_path / "t.txt").write_text(_line(argv_mod and argv or argv) if not line_mod else line_mod(_line(argv)), encoding="utf-8")
    return _py(RECORD_BODY(), [d, tmp_path / "t.txt", arm, 0, "0.3" if arm == "orc" else "0", steps, banked or synth["sha"][0]])


def _edit_json(name, fn):
    def f(d):
        p = d / name
        x = json.loads(p.read_text(encoding="utf-8"))
        fn(x)
        p.write_text(json.dumps(x), encoding="utf-8")
    return f


@needs_chain
@win
def test_run_record_assertions_pass_a_real_run_and_refuse_a_token_that_binds_an_argv_that_did_not_run(synth, tmp_path):
    for arm in ("orc", "orc0"):
        rc, out = _rec_run(synth, arm, tmp_path)
        assert rc == 0 and "ORC run record: PASS" in out, (arm, out)
    cases = (("argv.json is not the planned argv", dict(argv_mod=lambda a: a["argv"].__setitem__(a["argv"].index("--lr") + 1, "3e-4")), "argv.json is not the planned training argv"),
             ("the planned line is not the run's", dict(line_mod=lambda l: l.replace("--ego-perturb-p 0.3", "--ego-perturb-p 0.31")), "argv.json is not the planned training argv"),
             ("argv.json sha", dict(mod=_edit_json("argv.json", lambda x: x.__setitem__("sha", "0" * 16))), "argv.json sha"),
             ("summary argv_sha", dict(mod=_edit_json("summary.json", lambda x: x.__setitem__("argv_sha", "0" * 16))), "summary.json argv_sha"),
             ("not done", dict(mod=_edit_json("summary.json", lambda x: x.__setitem__("done", False))), "done is not true"),
             ("steps", dict(steps="3000"), "summary.json steps 6 != 3000"),
             ("arm", dict(mod=_edit_json("summary.json", lambda x: x.__setitem__("arm", "M1"))), "arm / seed"),
             ("seed", dict(mod=_edit_json("summary.json", lambda x: x.__setitem__("seed", 1))), "arm / seed"),
             ("selfcheck not pass", dict(mod=_edit_json("selfcheck.json", lambda x: x.__setitem__("declared_vs_built", "FAIL"))), "selfcheck.json is not a PASS"),
             ("init is not banked", dict(banked="0" * 64), "init_ckpt.sha256"),
             ("built p", dict(mod=_edit_json("summary.json", lambda x: x["levers"]["ego_perturb"].__setitem__("ego_perturb_p", 0.1))), "built ego_perturb_p"))
    for name, kw, msg in cases:
        rc, out = _rec_run(synth, "orc", tmp_path, **kw)
        assert rc == 1 and msg in out, (name, rc, out)


@needs_chain
@needs_reader
def test_the_perturbation_reached_control_agrees_with_the_readers_own_on_a_grid_and_on_the_real_summary(synth, tmp_path):
    R = _load("orc_read_for_reach", READ_P)
    sm_path = tmp_path / "summary.json"
    cases = [(arm, n, e) for arm in ("orc", "orc0") for n, e in ((0, 0), (0, 1000), (1, 1000), (280, 1000), (281, 1000), (300, 1000), (319, 1000), (320, 1000), (321, 1000), (500, 1000), (1000, 1000), (1001, 1000), (7, 23), (3000, 10000), (2799, 10000), (2800, 10000), (3200, 10000), (3201, 10000))]
    for arm, n, e in cases:
        sm_path.write_text(json.dumps({"ego_perturb_n": n, "ego_perturb_eligible": e}), encoding="utf-8")
        rc, out = _py(REACH_BODY(), [sm_path, arm, 0])
        ok, _ = R.g_pert("t", "ORC" if arm == "orc" else "ORC0", {"sources": [("summary", {"ego_perturb_n": n, "ego_perturb_eligible": e, "levers": {"ego_perturb": {"ego_perturb_p": 0.3 if arm == "orc" else 0.0}}})]},
                         "ego_perturb_n", "ego_perturb_eligible", "ego_perturb.ego_perturb_p")
        assert (rc == 0) == ok, (arm, n, e, rc, out, ok)
        assert ("PASS" in out) == ok
    for arm in ("orc", "orc0"):
        run = synth["run"][(arm, 0)]
        r = R.train_records({"train_dir": str(run)}, None)
        ok, det = R.g_pert("t", "ORC" if arm == "orc" else "ORC0", r, "ego_perturb_n", "ego_perturb_eligible", "ego_perturb.ego_perturb_p")
        rc, out = _py(REACH_BODY(), [run / "summary.json", arm, 0])
        assert ok is True and rc == 0 and "PASS" in out, (arm, det, out)
        ok, det = R.g_init("t", 0, r, synth["sha"][0], "init_ckpt.sha256")
        assert ok is True, ("the reader finds the init sha in the REAL record under its default key", det)


# =============================================================================================================================================
# AMENDMENT 3: the trainer's OWN gate accepts ORC-0 (it refused the earlier shape) and refuses a law flag on it
# =============================================================================================================================================
@pytest.mark.parametrize("seed", [0, 1])
def test_orc0_gates_pass_on_the_real_trainer_unpatched_at_both_seeds_with_the_probe_off_and_the_law_at_its_defaults(synth, seed):
    """The trainer is NOT patched anywhere in this file (the fixture ran `TR.main` on the Amendment 3 argv): the gate record it wrote is a PASS, the ego probe reads `off`, nothing of the law is moved from its default."""
    TR = synth["TR"]
    out = synth["root"] / f"orc0_gate_again_s{seed}"
    assert TR.main(_argv(out, synth["ck"][seed], "orc0", seed, gate=True)) == 0, "the trainer's own gate ACCEPTS ORC-0's registered argv"
    sc = json.loads((out / "selfcheck.json").read_text(encoding="utf-8"))
    assert sc["declared_vs_built"] == "PASS" and sc["ego_perturb_probe"]["pass"] is True and sc["ego_perturb_probe"]["ego_perturb"] == "off"
    ep = sc["levers"]["ego_perturb"]
    assert ep["ego_perturb_p"] == 0.0 and ep["kin_mode"] == "yaw"
    D = synth["D"]
    d = D.EGO_PERTURB_DEFAULTS
    assert (ep["ego_perturb_vscale"], ep["ego_perturb_vmax"], ep["ego_perturb_adrop"], ep["ego_perturb_yawrate_std"], ep["ego_perturb_seed"]) == (list(d["vscale"]), d["vmax"], d["adrop"], d["yawrate_std_deg"], d["seed"])
    assert d["vmax"] == 8.0 and ep["ego_perturb_vmax"] == synth["gate"][("orc", seed)]["levers"]["ego_perturb"]["ego_perturb_vmax"], "vmax's default IS ORC's registered 8.0: the eligible denominator is identical"
    orc = synth["gate"][("orc", seed)]
    assert orc["ego_perturb_probe"].get("ego_perturb") != "off" and orc["levers"]["ego_perturb"]["ego_perturb_p"] == 0.3, "the control is ORC's gate with the perturbation OFF, not the same gate"


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("flag,value", [("--ego-perturb-vscale", ["0.6", "1.0"]), ("--ego-perturb-vmax", ["6.0"]), ("--ego-perturb-adrop", ["1.0"]), ("--ego-perturb-yawrate-std", ["2.0"]), ("--ego-perturb-seed", ["1"])])
def test_a_law_flag_named_on_orc0_with_a_non_default_value_is_refused_by_the_trainers_own_gate(synth, seed, flag, value):
    """The guard Amendment 3 keeps unchanged: a law flag set (to a value other than the default) while --ego-perturb-p is 0 is 'a lever that is declared and does nothing' and the trainer REFUSES it. (yawrate-std 2.0 and
    seed 1 are the two registered values that tripped it before Amendment 3.)"""
    TR = synth["TR"]
    out = synth["root"] / f"orc0_law_{flag.replace('--ego-perturb-', '')}_s{seed}"
    with pytest.raises(SystemExit) as e:
        TR.main(_argv(out, synth["ck"][seed], "orc0", seed, gate=True) + [flag, *value])
    assert "are set while ego_perturb_p is 0" in str(e.value) and flag[2:].replace("-", "_") in str(e.value), str(e.value)[:300]


def test_a_law_flag_named_on_orc0_at_its_default_value_passes_the_trainers_gate_which_is_why_the_chain_refuses_the_naming(synth):
    """HONEST SCOPE of the previous test: the trainer's guard compares VALUES with the defaults, so `--ego-perturb-vscale 0.5 1.0`, `--ego-perturb-vmax 8.0`, `--ego-perturb-adrop 1.5` and `--ego-perturb-seed 0` named on ORC-0
    pass it. The queue never writes them on ORC-0 and the chain's gate assertions (test_gate_assertions_for_orc0_*) refuse the NAMING itself, so ORC-0 stays 'p = 0 and nothing else of the family'."""
    TR = synth["TR"]
    for k, (flag, value) in enumerate((("--ego-perturb-vscale", ["0.5", "1.0"]), ("--ego-perturb-vmax", ["8.0"]), ("--ego-perturb-adrop", ["1.5"]), ("--ego-perturb-seed", ["0"]))):
        out = synth["root"] / f"orc0_default_law_{k}"
        assert TR.main(_argv(out, synth["ck"][0], "orc0", 0, gate=True) + [flag, *value]) == 0, flag

