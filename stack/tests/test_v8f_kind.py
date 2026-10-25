"""PREREG_V8F_KIND: which KIND of T's constraint cells is worth ADE6 to O (eval only, zero training). CPU only.

What is pinned, and the control that must go red for each (the mutants live in raw/2026-10-25-kind/mutate_kind.py, output beside it):
  * THE PARTITION. The six kinds (`v8f_mini.CODE_KINDS`) hold 5 + 3 + 5 + 2 + 9 + 2 = 26 cells, each cell in exactly one kind, their union IS the `constraints`
    composite index for index, they equal the PREREG's own table (parsed from the markdown), their names collide with no group / composite, and every old group,
    composite and spec returns what it returned before (hashed against the tip's output).
  * THE EVAL. `v8f_mini_eval.py run --tac oracle --tac-groups <spec>` already routes the spec through `code_group_index` (one call, no table of its own): a kind, a
    sum of kinds and a leave-one-kind-out spec run as they are, replace exactly the kind's cells with the GT code and leave T's own code everywhere else.
  * THE QUEUE. The `kind` stage adds four labels and touches no old one (every old label hashes as at the tip; the file stays CRLF); its dry run prints the 30 eval commands.
  * THE CHAIN. Eval only, after DSCALE's marker; the checkpoint sha256 is asserted before and after; check-inputs runs before any box-2 eval; --dry-run prints the 30 evals.
  * THE READER (code/v8f_kind_read.py). The committed table reaches every row (ONE KIND CARRIES / DIFFUSE / INTERACTIVE / MIXED), the SPEED side-read both ways, and every
    refusal is rc 2: an identity control that misses +0 [0, 0] (own and all), a partition that is not one, a window set / checkpoint / manipulation that is not the banked
    one, a missing eval, a missing family. The numbers come from real files on disk and the real `main` on a synthetic straight-road world (test_v8f_dk.World).
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tanitad.models import v8f_mini as M  # noqa: E402

PKG = REPO / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-v8f-mini"
CODE = PKG / "code"
READER = CODE / "v8f_kind_read.py"
QUEUE = CODE / "v8f_mini_box2_queue.cmd"
CHAIN = CODE / "box2_ops" / "chain_v8f_kind.sh"
GUARD = CODE / "box2_ops" / "v8f_guard.sh"
PREREG = REPO / "Project Steering" / "PREREG_V8F_KIND.md"
EVAL_PY = REPO / "stack" / "scripts" / "v8f_mini_eval.py"
torch.set_num_threads(1)

KINDS = ("speed", "angle", "tte", "timegap", "geom", "leadgap")
KIND_SIZES = {"speed": 5, "angle": 3, "tte": 5, "timegap": 2, "geom": 9, "leadgap": 2}
C0 = M.N_GOAL + M.N_LAT + M.N_LON


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


needs_pkg = pytest.mark.skipif(not READER.exists(), reason="the v8F-mini package (code/v8f_kind_read.py) is not in this checkout")
win = pytest.mark.skipif(sys.platform != "win32", reason="the queue is a Windows batch file")
R = _load("v8f_kind_read_under_test", READER) if READER.exists() else None            # importable without the package (the lander's stack-only probe)
_TAC = _load("test_v8f_tac_for_kind", Path(__file__).resolve().parent / "test_v8f_tac.py")
world = _TAC.world                                                                      # the eval-level fixtures are test_v8f_tac's
_DK = _load("test_v8f_dk_for_kind", Path(__file__).resolve().parent / "test_v8f_dk.py") if READER.exists() else None
_PGT = _load("test_v8f_pgt_for_kind", Path(__file__).resolve().parent / "test_v8f_pgt.py") if READER.exists() else None


# =================================================================================================================================================
# THE PARTITION
# =================================================================================================================================================
def test_kinds_partition_the_26_constraint_cells_and_their_union_is_constraints():
    assert tuple(M.CODE_KINDS) == KINDS
    assert {k: len(v) for k, v in M.CODE_KINDS.items()} == KIND_SIZES and sum(KIND_SIZES.values()) == len(M.CONS_NAMES) == M.N_CONS == 26
    flat = [i for k in KINDS for i in M.CODE_KINDS[k]]
    assert len(flat) == len(set(flat)) == 26, "each cell is in exactly one kind (disjoint)"
    assert sorted(flat) == M.code_group_index("constraints") == list(range(C0, M.CODE_DIM)), "the union IS the constraints composite, index for index"
    names = [M.CONS_NAMES[i - C0] for i in flat]
    assert sorted(names) == sorted(M.CONS_NAMES), "every constraint cell is named by exactly one kind"
    for k in KINDS:                                                                   # each kind's tuple is the cells' CODE indices, by CI[name]
        assert list(M.CODE_KINDS[k]) == [C0 + M.CI[M.CONS_NAMES[i - C0]] for i in M.CODE_KINDS[k]]
        assert M.code_group_index(k) == sorted(M.CODE_KINDS[k]) and len(M.code_group_index(f"all-{k}")) == M.CODE_DIM - KIND_SIZES[k]
    assert sorted(i for k in KINDS for i in M.code_group_index(k)) == M.code_group_index("constraints"), "the six single-kind specs tile the constraints"
    # the six kinds, summed, are the constraints (as a spec); each kind plus its leave-one-out partition the 60-dim code
    assert M.code_group_index("+".join(KINDS)) == M.code_group_index("constraints")
    for k in KINDS:
        assert sorted(M.code_group_index(k) + M.code_group_index(f"all-{k}")) == list(range(M.CODE_DIM))


@pytest.mark.skipif(not PREREG.exists(), reason="PREREG_V8F_KIND.md is not in this checkout")
def test_kinds_are_the_prereg_tables_cells_parsed_from_the_markdown():
    txt = PREREG.read_text(encoding="utf-8")
    sec = txt.split("## The partition", 1)[1].split("The union of the six kinds", 1)[0]
    rows = re.findall(r"^\| `(\w+)`(?: \([^)]*\))? \| (.+?) \|$", sec, re.M)
    table = {k: tuple(re.findall(r"`(\w+)`", cells)) for k, cells in rows}
    assert tuple(table) == KINDS, table
    for k in KINDS:
        assert tuple(M.CONS_NAMES[i - C0] for i in M.CODE_KINDS[k]) == table[k] or sorted(M.CONS_NAMES[i - C0] for i in M.CODE_KINDS[k]) == sorted(table[k]), k
    assert sum(len(v) for v in table.values()) == 26
    if R is not None:
        assert {k: tuple(v) for k, v in R.KIND_CELLS.items()} == table, "the reader restates the PREREG's table by name"


def test_kind_names_collide_with_no_group_composite_or_keyword():
    assert not set(M.CODE_KINDS) & set(M.CODE_GROUPS), "a kind named like a group would shadow it"
    assert not set(M.CODE_KINDS) & set(M.CODE_COMPOSITES)
    assert not set(M.CODE_KINDS) & {"all"} and not any(k.startswith("all-") or "+" in k for k in M.CODE_KINDS)


# sha256[:32] of json.dumps({spec: code_group_index(spec)}, sort_keys=True) for the specs below, taken at tip c7d40467 BEFORE the kinds existed
OLD_SPECS = ("goal", "latdec", "londec", "latcons", "lonspeed", "stoplead", "decisions", "constraints", "all", "all-goal", "all-latdec", "all-londec", "all-latcons",
             "all-lonspeed", "all-stoplead", "lonspeed+londec", "goal+londec", "all-decisions", "latcons+stoplead", "decisions+latcons")
OLD_SPECS_SHA = "486a1e4dc40c8daef76c61b57db5f490"
OLD_GROUP_SHA = {"goal": "763ad76a33bb", "latdec": "439f994edfe8", "londec": "06ab1df75f66", "latcons": "71392ea5cf2f", "lonspeed": "9d5364c6d159", "stoplead": "3bc31d91ecd1"}


def test_the_old_groups_composites_and_every_old_spec_return_what_they_returned_at_the_tip():
    got = hashlib.sha256(json.dumps({s: M.code_group_index(s) for s in OLD_SPECS}, sort_keys=True).encode()).hexdigest()[:32]
    assert got == OLD_SPECS_SHA, "an existing spec's output changed"
    assert {g: hashlib.sha256(json.dumps(list(v)).encode()).hexdigest()[:12] for g, v in M.CODE_GROUPS.items()} == OLD_GROUP_SHA
    assert tuple(M.CODE_GROUPS) == ("goal", "latdec", "londec", "latcons", "lonspeed", "stoplead")
    assert M.CODE_COMPOSITES == {"decisions": ("goal", "latdec", "londec"), "constraints": ("latcons", "lonspeed", "stoplead")}


def test_code_group_index_takes_a_kind_wherever_it_takes_a_group():
    spd = [C0 + M.CI[n] for n in ("lon_v_target_ms", "v_a_ms", "v_end_ms", "v_lo_ms", "v_hi_ms")]
    assert M.code_group_index("speed") == sorted(spd) == [46, 56, 57, 58, 59]
    assert M.code_group_index("speed+angle") == sorted(spd + [C0 + M.CI[n] for n in ("lat_theta_deg", "turn_dyaw_deg", "turn_exit_psi_deg")])
    assert M.code_group_index("all-speed") == [i for i in range(M.CODE_DIM) if i not in spd] and len(M.code_group_index("all-speed")) == 55
    assert M.code_group_index("speed+londec") == sorted(spd + list(M.CODE_GROUPS["londec"])), "a kind and a group mix"
    assert M.code_group_index("goal+tte") == sorted(list(M.CODE_GROUPS["goal"]) + list(M.CODE_KINDS["tte"]))
    assert M.code_group_index("all-speed+angle") == [i for i in range(M.CODE_DIM) if i not in spd + list(M.CODE_KINDS["angle"])]
    assert M.code_group_index("all-constraints") == M.code_group_index("decisions") and M.code_group_index("all-tte") != M.code_group_index("all-geom")
    for bad in ("speedy", "speed,angle", "all-speedy", "Speed"):
        with pytest.raises(ValueError, match="unknown code group"):
            M.code_group_index(bad)
    with pytest.raises(ValueError, match="selects nothing"):
        M.code_group_index("all-all")
    with pytest.raises(ValueError, match="unknown code group.*geom"):
        M.code_group_index("nokind")                                                  # the refusal names the kinds too


# =================================================================================================================================================
# THE EVAL: --tac-groups already routes through code_group_index
# =================================================================================================================================================
def test_eval_routes_tac_groups_through_code_group_index_and_nowhere_else():
    """The routing finding, pinned statically: in the eval, `a.tac_groups` is read in exactly three places -- the one guard that compares it with "all", the ONE call
    `M.code_group_index(a.tac_groups)` and the manipulation record -- so no table of group names exists on the eval side to extend, and a spec the model accepts is a spec
    the eval runs. (The functional proof is the next test.)"""
    src = EVAL_PY.read_text(encoding="utf-8")
    tree = ast.parse(src)
    uses = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == "tac_groups" and isinstance(n.value, ast.Name) and n.value.id == "a"]
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "code_group_index"]
    assert len(calls) == 1 and ast.unparse(calls[0]) == "M.code_group_index(a.tac_groups)"
    where = sorted(src.splitlines()[n.lineno - 1].strip() for n in uses)
    assert len(uses) == 3 and where[0].startswith('"groups": a.tac_groups') and where[1].startswith('grp_idx = M.code_group_index(a.tac_groups)')         and where[2].startswith('if a.tac_groups != "all" and a.tac != "oracle":'), where
    assert not re.search(r'"(speed|angle|tte|timegap|geom|leadgap)"', src.split("def cmd_run", 1)[1].split("def beta_block", 1)[0]), "no kind name is hard-wired in the eval"
    assert "CODE_GROUPS" not in src and "CODE_COMPOSITES" not in src and "CODE_KINDS" not in src, "the eval holds no table of its own"


def _gt_cells_differ(z_a, z_b, cells):
    return float(np.abs(z_a["tac_code"][:, cells] - z_b["tac_code"][:, cells]).max())


def test_eval_accepts_kind_specs_and_replaces_exactly_the_kinds_cells(world, tmp_path):
    EV = _load("v8f_mini_eval_for_kind", EVAL_PY)
    ck = _TAC._ckpt(world, tmp_path)
    out = tmp_path / "ev"
    specs = {"full": "all", "sp": "speed", "nosp": "all-speed", "an": "angle", "spang": "speed+angle", **{f"k_{k}": k for k in KINDS if k not in ("speed", "angle")},
             "union": "constraints"}
    for t, spec in specs.items():
        assert _TAC._run(EV, world, ck, out, f"c-{t}", "--tac", "oracle", "--tac-groups", spec, "--tac-alpha", "1") == 0, spec
    z = {t: np.load(out / f"eval_c-{t}.npz") for t in specs}
    man = {t: json.load(open(out / f"eval_c-{t}.json", encoding="utf-8"))["tac_manipulation"] for t in specs}
    for t, spec in specs.items():
        assert man[t]["mode"] == "oracle" and man[t]["groups"] == spec and man[t]["alpha"] == 1.0
        assert man[t]["n_code_dims_replaced"] == len(M.code_group_index(spec)), spec
    assert [man[f"k_{k}"]["n_code_dims_replaced"] for k in ("tte", "timegap", "geom", "leadgap")] == [5, 2, 9, 2]
    S, A = sorted(M.CODE_KINDS["speed"]), sorted(M.CODE_KINDS["angle"])
    rest = [i for i in range(M.CODE_DIM) if i not in S + A]
    assert _gt_cells_differ(z["full"], z["nosp"], S) > 1e-3, "the world gives T a speed-cell error, or a zero below proves nothing"
    # GT where the spec says GT ...
    assert np.array_equal(z["sp"]["tac_code"][:, S], z["full"]["tac_code"][:, S]), "speed: the GT code in the five speed cells"
    assert np.array_equal(z["nosp"]["tac_code"][:, [i for i in range(M.CODE_DIM) if i not in S]], z["full"]["tac_code"][:, [i for i in range(M.CODE_DIM) if i not in S]]), \
        "all-speed: the GT code everywhere EXCEPT the speed cells"
    assert np.array_equal(z["spang"]["tac_code"][:, S + A], z["full"]["tac_code"][:, S + A]), "speed+angle: GT in the eight cells"
    # ... and T's OWN code everywhere else (T's own speed cells come from the angle run, T's own rest from the speed run, and so on: two runs that agree off their cells)
    assert np.array_equal(z["nosp"]["tac_code"][:, S], z["an"]["tac_code"][:, S]), "all-speed holds T's own code in the speed cells"
    assert np.array_equal(z["sp"]["tac_code"][:, rest], z["an"]["tac_code"][:, rest]), "speed leaves T's own code outside the speed cells"
    assert np.array_equal(z["spang"]["tac_code"][:, rest], z["sp"]["tac_code"][:, rest]), "speed+angle leaves T's own code in the other 52 dims"
    assert np.array_equal(z["union"]["tac_code"][:, C0:], z["full"]["tac_code"][:, C0:]) and np.array_equal(z["union"]["tac_code"][:, :C0], z["an"]["tac_code"][:, :C0]), \
        "the union: GT in the 26 constraint cells, T's decisions"
    # each kind's single oracle writes the GT into exactly its own cells
    for k in KINDS:
        c = sorted(M.CODE_KINDS[k])
        zk = z["sp" if k == "speed" else "an" if k == "angle" else f"k_{k}"]
        assert np.array_equal(zk["tac_code"][:, c], z["full"]["tac_code"][:, c]), k
    assert np.abs(z["sp"]["pred"] - z["nosp"]["pred"]).max() > 1e-4, "the speed cells move O's plan"
    with pytest.raises(SystemExit, match="unknown code group"):
        _TAC._run(EV, world, ck, out, "bad", "--tac", "oracle", "--tac-groups", "speedy")
    with pytest.raises(SystemExit, match="REFUSED"):
        _TAC._run(EV, world, ck, out, "bad", "--tac", "shuffle", "--tac-groups", "speed")


# =================================================================================================================================================
# THE QUEUE
# =================================================================================================================================================
# every label of the queue as it stood at tip c7d40467, with sha256[:16] of its text ("\n" + the lines up to the next label)
PRE_EXISTING = {
    "d_train": "653a25ac6e43be79", "d_eval": "b64a718bcf22e4ca", "prep": "8f8749398e785a74", "gate": "01fa4b4d415d3057", "train": "c3867e94dae8b06b", "eval": "8d3f96a2ec519c6d",
    "tac": "8357d58b9565d87f", "tacone": "d05a9be75c4de2bd", "tacgrp": "7cae083fcfb5a237", "tacgone": "cdbb6e94327d04af", "tacgrp2": "efb5d1a2944b99fa",
    "tacgtwo": "14ccef2e21c86f83", "tacabs": "60aa7130ee682938", "tacabsarm": "0ed52c566b9de578", "gatewc": "487c836142548e93", "trainwc": "5565fa65d285f47f",
    "evalwc": "ccb014965cc92aa5", "gatepgt": "3ca89871f8da1c43", "trainpgt": "49fe41ddd1622379", "evalpgt": "a04620ebdcfbb068", "evalpgt_1": "effe5bbee55d4771",
    "evalpgt_2": "33cdd72c074d8c6a", "evalpgt_3": "a7e30c81d582343c", "evalpgt_4": "22636d9d007850fa", "gridbase": "56299fbe24cec6f7", "gategrid": "e3250ab756539587",
    "traingrid": "87cf143655b55573", "evalgrid": "399406480951bc1d", "evalgrid_1": "74aaef5e39bbaca5", "evalgrid_2": "f242aa25ecd51bd6", "evalgrid_3": "210db6bc5c411a0c",
    "evalgrid_4": "22636d9d007850fa", "gatehist": "93a341e35cc9a0b1", "trainhist": "344fcc45b1624ead", "evalhist": "413108b37c6fe068", "evalhist_1": "03081f7da50e8739",
    "evalhist_2": "168db150abcf3ce1", "evalhist_3": "d9d586a4304f2d24", "evalhist_4": "22636d9d007850fa", "gatedscale": "0d91b533582f0ea6", "traindscale": "7c5f99d4e8e0c096",
    "evaldscale": "080e32b4ee8c2ac6", "evaldscale_1": "f59695b83d08d77a", "evaldscale_2": "813ee4390fc783ce", "evaldscale_3": "fdba053d0afad97d",
    "evaldscale_4": "22636d9d007850fa", "loraenv": "adacc37155199a86", "identitylora": "ee8f28bc42b410c9", "smokelora": "af9acad644a32d72", "gatearm": "b79d2a733b86fe61",
    "floors": "2b7f345ee3e86bcc", "compare": "b41e1a1c859c59c3"}
KIND_LABELS = ("kind", "kindseed", "kindown", "kindorc")
MIXO_LABELS = {"gatemixo", "trainmixo", "evalmixo"} | {f"evalmixo_{i}" for i in range(1, 8)}      # PREREG_V8F_MIXO appended these (test_v8f_mixo.py pins them); widened here, nothing else
FLAT_LABELS = {"gateflat", "trainflat", "evalflat"} | {f"evalflat_{i}" for i in range(1, 5)}       # PREREG_V8F_FLAT appended these (test_v8f_flat.py pins them); widened here, nothing else
OLD_DISPATCH = ("gate", "train", "eval", "floors", "compare", "tac", "tacgrp", "tacgrp2", "tacabs", "gatearm", "tacabsarm", "gatewc", "trainwc", "evalwc", "gatepgt", "trainpgt",
                "evalpgt", "gategrid", "traingrid", "evalgrid", "gatehist", "trainhist", "evalhist", "gatedscale", "traindscale", "evaldscale", "identitylora", "smokelora")
needs_queue = pytest.mark.skipif(not QUEUE.exists(), reason="the v8F-mini package is not in this checkout")


def _queue_text():
    return QUEUE.read_text(encoding="utf-8").replace("\r\n", "\n")


def _blocks():
    parts = re.split(r"(?m)^(:[A-Za-z0-9_]+)$", _queue_text())
    return {parts[i][1:]: parts[i + 1] for i in range(1, len(parts), 2)}


def _block(label):
    return _queue_text().split(f"\n:{label}\n", 1)[1].split("\nexit /b 0\n", 1)[0].split("\n")


def _code(lines):
    return [x for x in lines if x.strip() and not x.lower().startswith("rem ")]


def _dry(args, clean_paths=True):
    env = dict(os.environ, V8F_DRYRUN="1")
    if clean_paths:
        for k in ("REPO", "RUNSM", "CACHEFULL", "V9", "PY", "LORAFRAMES"):
            env.pop(k, None)
    r = subprocess.run(f'cmd /c ""{QUEUE}" {args}"', env=env, capture_output=True, text=True, timeout=60, encoding="utf-8")
    return r.returncode, [x for x in r.stdout.splitlines() if "v8f_mini_eval.py" in x]


@needs_queue
def test_queue_is_crlf_dispatches_kind_and_every_existing_stage_is_byte_identical():
    raw = QUEUE.read_bytes()
    assert raw.count(b"\r\n") == raw.count(b"\n") > 600 and raw.count(b"\r") == raw.count(b"\r\n") and raw.endswith(b"\r\n"), "every line CRLF, no bare CR"
    q, blocks = _queue_text(), _blocks()
    for label, h in PRE_EXISTING.items():
        assert label in blocks, f"the stage {label} was removed"
        assert hashlib.sha256(("\n" + blocks[label]).encode()).hexdigest()[:16] == h, f"the stage {label} was edited"
        assert "kind" not in blocks[label].lower().replace("kinds", ""), f"{label} mentions the new stage"
    assert set(blocks) - MIXO_LABELS - FLAT_LABELS == set(PRE_EXISTING) | set(KIND_LABELS), "the only new labels are the four of this edit (and PREREG_V8F_MIXO's and PREREG_V8F_FLAT's, pinned in test_v8f_mixo.py / test_v8f_flat.py)"
    assert q.count('if /i "%~1"=="kind" goto :kind\n') == 1
    target = {"train": "d_train", "eval": "d_eval"}                                    # `train` and `eval` have always dispatched to :d_train / :d_eval
    for st in OLD_DISPATCH:
        assert f'if /i "%~1"=="{st}" goto :{target.get(st, st)}\n' in q, st
    labels = set(blocks)
    disp = re.findall(r'(?m)^if /i "%~1"=="(\w+)" goto :(\w+)$', q)
    assert [(a, b) for a, b in disp if b not in labels] == [], "a dispatch line whose target label does not exist"
    usage = next(x for x in q.split("\n") if x.startswith("echo usage:"))
    assert "^|tacgrp2^|kind^|tacabs^|" in usage and usage.endswith("^|gatedscale^|traindscale SEED^|evaldscale SEED^|identitylora^|smokelora"), "the usage line names kind; its end is untouched"
    assert "^|gatehist^|trainhist SEED^|evalhist SEED^|gatedscale^|traindscale SEED^|evaldscale SEED" in q


@needs_queue
@win
def test_queue_kind_stage_dry_run_prints_the_thirty_evals_in_order():
    rc, ln = _dry("kind")
    assert rc == 0 and len(ln) == 30 == 2 * (2 + 6 + 6 + 1), "two seeds x (own + all + 6 single + 6 leave-one-out + union)"
    def _tac(x):
        m = re.search(r"--tac oracle --tac-groups (\S+) --tac-alpha (\S+)", x)
        return m.groups() if m else (None, None)
    got = [(re.search(r"--tag (\S+)", x).group(1), _tac(x)) for x in ln]
    per_seed = ([("kind-own", None)] + [("kind-all", ("all", "1"))] + [(f"kind-only-{k}", (k, "1")) for k in KINDS] + [(f"kind-loo-{k}", (f"all-{k}", "1")) for k in KINDS]
                + [("kind-union", ("constraints", "1"))])
    want = [(f"M2b_s{s}-{n}", g if g else (None, None)) for s in (0, 1) for n, g in per_seed]
    assert got == want, got
    own = [x for x in ln if "-kind-own " in x]
    assert len(own) == 2 and all("--tac" not in x and x.rstrip().endswith("--project --families") for x in own), "own is the standard eval: --tac off, the banked eval's flags"
    assert all("--project" not in x and "--families" not in x for x in ln if "-kind-own " not in x)
    assert all(f"M2b_s{s}\\ckpt_last.pt" in x for s in (0, 1) for x in ln if f"M2b_s{s}-kind" in x), "each seed evaluates its own checkpoint"
    assert all("--kin-mode yaw" in x and "--tracks" in x and "--out-dir" in x and "--data-frac" not in x and "--steps" not in x for x in ln)
    assert _dry("tacgrp2")[0] == 0 and len(_dry("tacgrp2")[1]) == 40, "the old stage still prints its 40 evals"


@needs_queue
def test_queue_kind_runs_the_command_it_prints_and_skips_only_what_is_finished_static():
    for label in ("kindown", "kindorc"):
        code = _code(_block(label))
        ek = [x for x in code if x.startswith('set "EK=')]
        assert len(ek) == 1 and ek[0].count("%EVAL%") == 1 and [x for x in code if "%EVAL%" in x] == ek, f"{label}: the eval script appears once, inside the one string"
        assert "if defined V8F_DRYRUN (echo %EK% & exit /b 0)" in code
        assert sum(x.strip().startswith("%EK% >> ") for x in code) == 1, f"{label}: EK is executed exactly once"
        assert '--ckpt "%OUT%\\ckpt_last.pt"' in ek[0] and "%EC% %ER%" in ek[0]
        assert not any(x.rstrip().endswith("(") for x in _block(label)), f"{label}: a multi-line parenthesised block"
        assert 'if exist "%EV%\\eval_%TG%.npz" (echo [skip] eval %TG% & exit /b 0)' in code, "a finished eval is not repeated"
        assert any(x.startswith("if errorlevel 1 (echo [FAIL]") and x.rstrip().endswith("exit /b 1)") for x in code)
        assert not any("data-frac" in x or "p-gt" in x or "--w-" in x for x in code), "an eval stage: no training flag"
    assert 'set "TG=%~1_s%~2-kind-own"' in _code(_block("kindown")) and 'set "TG=%~1_s%~2-kind-%~4"' in _code(_block("kindorc"))
    assert "--tac oracle --tac-groups %~3 --tac-alpha 1" in next(x for x in _code(_block("kindorc")) if x.startswith('set "EK='))
    seed = _code(_block("kindseed"))
    assert sum(x.startswith("for %%K in (speed angle tte timegap geom leadgap)") for x in seed) == 2
    assert all(x.rstrip().endswith("|| exit /b 1") for x in seed if x.startswith(("call", "for")))
    top = _code(_block("kind"))
    assert "for %%S in (0 1) do call :kindseed %%S || exit /b 1" in top and "call :prep || exit /b 2" in top


# =================================================================================================================================================
# THE CHAIN
# =================================================================================================================================================
def _git_bash():
    for p in (os.environ.get("GIT_BASH"), "C:/Program Files/Git/bin/bash.exe", "C:/Program Files (x86)/Git/bin/bash.exe"):
        if p and Path(p).exists():
            return p
    return None


needs_chain = pytest.mark.skipif(not CHAIN.exists(), reason="the v8F-mini package is not in this checkout")


@needs_chain
def test_chain_static_order_the_wait_the_checkpoint_assertion_and_eval_only():
    c = CHAIN.read_text(encoding="utf-8")
    assert b"\r\n" not in CHAIN.read_bytes(), "a bash script with CRLF endings does not run"
    code = "\n".join(x for x in c.splitlines() if not x.lstrip().startswith("#"))
    for banned in ("traindscale", "trainhist", "gatedscale", "--stage", "--no-families", "queue t_", "ZZV8FTAC", "train "):
        assert banned not in code, f"an eval-only chain: {banned!r}"
    order = ["waiting for the predecessor chain", "( set -C;", "fetch -q origin", 'ckptsha "before the evals"', 'v8f_kind_read.py" --check-inputs', "launch telemetry_long",
             "v8f_guard.sh > ", "queue kindrun $EVAL_POLLS kind", 'ckptsha "after the evals"', 'pull "eval/eval_M2b_s$s-$suf.$x"', '|| stop "kind read"', 'finish "$D1 |']
    idx = [c.index(x) for x in order]
    assert idx == sorted(idx) and len(set(idx)) == len(idx), dict(zip(order, idx))
    assert c.index("floors_perwindow.npz") < c.index('v8f_kind_read.py" --check-inputs') < c.index("launch telemetry_long"), "B's side is refused while box 2 is still idle and untouched"
    assert "AFTER_OUT=${AFTER_OUT:-$L/chain_v8f_dscale.out}" in c and 'AFTER_DONE=${AFTER_DONE:-ZZV8FMINI-DONE}' in c and "ZZV8FMINI-STOP" in c and "chain_v8f_kind.out" in c
    assert 'grep -qF "\'^(ZZV8FMINI|" $B/v8f_guard.sh' in c and "ZZV8FMINI" in GUARD.read_text(encoding="utf-8")
    assert "( set -C;" in c and "BOX2_BUSY.txt" in c, "the busy marker is taken atomically"
    # the checkpoint assertion: both shas, before AND after; the chain's pair is the reader's pair
    shas = re.search(r"^CKPT_SHA=\((\w{64}) (\w{64})\)", c, re.M).groups()
    assert dict(zip(("M2b_s0", "M2b_s1"), shas)) == R.BANKED_CKPT_SHA == {"M2b_s0": "601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8",
                                                                          "M2b_s1": "6104fb15dc21d524b6ed53be632b8d53ec5413526ac587734c22684c7e50c2e6"}
    assert '|| stop "the reader cannot use the M2b side (check-inputs)"' in c and '|| stop "kind read"' in c, "a refused input or a refused read stops the chain"
    assert "for x in json npz log; do pull \"eval/eval_M2b_s$s-$suf.$x\"" in c, "the json, the npz and the log of every eval are pulled"
    assert c.count('ckptsha "') == 2 and 'rsha "$R/M2b_s$s/ckpt_last.pt" SHA256' in c and '!= the banked run' in c
    assert "ckpt_sha256.json" in c and "--ckpt-sha" in c
    # the queue, the suffixes and the markers
    assert [x for x in re.findall(r'queue (\w+) \$EVAL_POLLS (\w+)', c)] == [("kindrun", "kind")]
    m = re.search(r'^SUFFIXES="kind-own kind-all"; for k in \$KINDS; do SUFFIXES="\$SUFFIXES kind-only-\$k"; done; for k in \$KINDS; do SUFFIXES="\$SUFFIXES kind-loo-\$k"; done; '
                  r'SUFFIXES="\$SUFFIXES kind-union"$', c, re.M)
    assert m and 'KINDS="speed angle tte timegap geom leadgap"' in c and R.arm_suffixes() == ["kind-own", "kind-all"] + [f"kind-only-{k}" for k in KINDS] + \
        [f"kind-loo-{k}" for k in KINDS] + ["kind-union"]
    for st in ("kind", "kindseed", "kindown", "kindorc"):
        assert f'"^:$st\\$"' in c or f"^:$st\\$" in c
    for f in ("v8f_kind_read.py", "v8f_pgt_read.py", "v8f_dk_read.py", "v8f_wc_read.py", "v8f_tac_groups_read.py", "v8f_mini_full_report.py"):
        assert f in c, f"the chain insists the tip has {f}"
    assert "^CODE_KINDS = {" in c and "M.code_group_index(a.tac_groups)" in c
    for x in ("$L/v8f_mini/runs/M2b_s$s/$f", "$L/v8f_cz/eval/$f", "$L/v8f_tacgrp2/eval/$f", "floors_perwindow.npz"):
        assert x in c, x
    arch = next(x for x in c.splitlines() if "archive --format=tar" in x)
    assert all(x in arch for x in ("stack", "taniteval", '"$HJ"', '"$PK"', '"$B1"')), "the reader and the lead block ride in the archive"
    finish = c.split("finish() {", 1)[1].split("\n}\n", 1)[0]
    assert finish.index("rm -f $B/BOX2_BUSY.txt") < finish.index("ZZV8FMINI-DONE"), "the box is released before DONE is announced"
    assert 'stop() { echo "ZZV8FMINI-STOP $*"' in c
    polls = int(re.search(r"^EVAL_POLLS=(\d+)", c, re.M).group(1))
    assert polls * 30 / 3600 >= 1.0, "the wait covers the PREREG's hour"


@needs_chain
def test_chain_keeps_dscale_s_guard_lines_verbatim():
    ds = CODE / "box2_ops" / "chain_v8f_dscale.sh"
    if not ds.exists():
        pytest.skip("chain_v8f_dscale.sh is not in this checkout")
    d, c = ds.read_text(encoding="utf-8").splitlines(), CHAIN.read_text(encoding="utf-8")
    keys = ("fresh telemetry", "waitjob.sh", "ZZGUARD-STOP", "ZZJOB-END", "BOOT0", "sleep 300", "launch()", "boot()", "clean()", "tl()", "telemetry.stop")
    lines = [x.strip() for x in d if not x.lstrip().startswith("#") and any(k in x for k in keys) and not any(w in x.lower() for w in ("dscale", "chain_v8f", "telemetry_v8f", "$e/"))]
    assert len(lines) >= 8
    assert [x for x in lines if x not in c] == [], "DSCALE's guard / polling / telemetry lines must be in this chain verbatim"


@needs_chain
def test_chain_syntax_and_dry_run_prints_the_thirty_queue_lines(tmp_path):
    bash = _git_bash()
    if bash is None or sys.platform != "win32":
        pytest.skip("Git Bash on Windows is needed to run the chain's --dry-run")
    r = subprocess.run([bash, "-n", str(CHAIN)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert r.returncode == 0, r.stderr
    env = dict(os.environ, V8F_QUEUE=str(QUEUE), TMPDIR=tmp_path.as_posix())
    for k in ("V8F_DRYRUN", "REPO", "RUNSM", "CACHEFULL", "V9", "PY"):
        env.pop(k, None)
    r = subprocess.run([bash, str(CHAIN), "--dry-run"], env=env, capture_output=True, text=True, timeout=180, encoding="utf-8", stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stderr
    out = r.stdout.splitlines()
    assert [x for x in out if x.startswith("== ")] == ["== predecessor", "== checkpoint sha256 (asserted on box 2 before and after the evals)", "== kind", "== read"]
    evals = [x for x in out if "v8f_mini_eval.py" in x]
    assert evals == _dry("kind")[1] and len(evals) == 30, "the chain's printed lines ARE the queue's own dry-run lines at box 2's default paths"
    assert any(x.startswith("M2b_s0 D:/tanitad/runs/v8f_mini_full/M2b_s0/ckpt_last.pt 601cb435") for x in out)
    assert any("v8f_kind_read.py --eval-dir" in x and "--ckpt-sha" in x and "--out" in x for x in out)
    assert any("^ZZV8FMINI-DONE" in x and "chain_v8f_dscale.out" in x for x in out), "the predecessor is DSCALE's chain"


# =================================================================================================================================================
# THE READER: the committed table and its refusals
# =================================================================================================================================================
N_WIN, N_EP = 72, 12
SEED_GAP = 0.12075                                                                       # F: M2b_s1's own ADE6 sits this far above s0's (2F = 0.2415)


def _gains(each=0.05, **over):
    return {k: over.get(k, each) for k in KINDS}


def _strata_labels(w, layout="few"):
    """The synthetic world's two label columns per window (the T-tick row's `curve` / `turn_is_turn`). 12 episodes x 6 windows (wi = window index in the episode).
    few:  JUNCTION in 4 episodes (8 windows), CURVE in 2 episodes (2 windows)  -> both strata under 10 episodes, NO-TURN over all 12.
    many: JUNCTION in 11 episodes (11 windows), CURVE in 4 episodes (4 windows) -> JUNCTION over the 10-episode floor, CURVE under it."""
    wi = np.tile(np.arange(w.n_win), w.n_ep)
    if layout == "many":
        return ((w.eid >= 8) & (wi == 4)).astype(np.int8), ((w.eid <= 10) & (wi == 3)).astype(np.int8)
    return (np.isin(w.eid, (4, 5)) & (wi == 4)).astype(np.int8), ((w.eid < 4) & np.isin(wi, (3, 4))).astype(np.int8)


def _write_strata(path, w, fb, cu, tit, **over):
    """A strata file in the exact format of mk_window_strata.py. The stratum is written HERE, independently of the reader's own `stratum_codes`:
    JUNCTION (0) = turn_is_turn on a window with a v9 row; CURVE (1) = curve on one; NO-TURN (2) = every other window."""
    has = ~np.asarray(fb, bool)
    s = np.full(w.n, 2, np.int8)
    s[(cu == 1) & has] = 1
    s[(tit == 1) & has] = 0
    arrs = dict(stratum=s, curve=cu, turn_is_turn=tit, has=has, eid=w.eid, t=w.t, a_t=np.zeros(w.n, np.int64), rowT=np.arange(w.n),
                provenance=np.array(json.dumps({"release_md5": R.RELEASE_MD5, "definition": "synthetic test strata"})))
    arrs.update(over)
    np.savez(path, **arrs)
    return path


def _kind_world(d, g0=None, g1=None, u0=1.0, u1=1.0, loo=0.05, noise=0.01, own_dev=0.0, all_dev=0.0, drop=None, bad_man=None, ckpt_bad=None, no_fam=None, eid_swap=False,
                fb_diff=False, sha=None, no_json=None, lateral_x=0.0, shift_x=None, layout="few", strata_over=None):
    """Real files on disk for the 15 + 2 evals per seed on a straight-road world whose ADE6 is the y offset of the plan. g0 / g1: {kind: gain} per seed (a scalar or one value per
    window); u: the union's gain; loo: all-k minus all. `own_dev` / `all_dev` break an identity control; `drop` omits a file; `bad_man` = (name, field, value); `ckpt_bad` = name
    scored on another checkpoint; `no_fam` = (name, family) removes a family from a json; `shift_x` = {arm suffix: metres} moves an arm's plan forward (LAMBDA); `layout` = the
    strata layout (_strata_labels); `strata_over` = keys of the strata file to overwrite (a wrong strata file)."""
    d = Path(d)
    ev = d / "eval"
    ev.mkdir(parents=True, exist_ok=True)
    w = _DK.World(d)
    n, eid, gt = w.n, w.eid, w.gt
    rng = np.random.default_rng(11)
    z0 = np.clip(rng.normal(0, 1, n), -2, 2)
    fb = np.zeros(n, bool)
    fb[:3] = True
    g = {0: g0 or _gains(), 1: g1 or _gains()}
    u = {0: u0, 1: u1}

    def write(name, ade, meta, own=False, banked=False):
        ade = np.asarray(ade, np.float64)
        pred = gt.copy()
        pred[:, :, 1] += ade[:, None]
        key = name.split("-", 1)[1] if "-" in name else ""                       # the banked std eval is T's own code, the banked -tacorc eval is the full oracle: a shift of an arm is the shift of its banked twin
        sx = (shift_x or {}).get({"": "kind-own", "tacorc": "kind-all"}.get(key, key), 0.0)
        if sx:                                                                     # a forward shift of the whole plan; the y offset is re-solved so the ADE stays `ade`
            pred[:, :, 0] += sx
            pred[:, :, 1] = gt[:, :, 1] + np.sqrt(np.maximum(ade ** 2 - sx ** 2, 0.0))[:, None]
        pred = pred.astype(np.float32)
        a6 = np.linalg.norm(pred.astype(np.float64) - gt, axis=-1).mean(1)
        e = eid[::-1].copy() if (eid_swap and name == "M2b_s0-kind-loo-tte") else eid
        arrs = dict(pred=pred, ade6=a6, eid=e)
        if not own:
            arrs["tac_oracle_fallback"] = (~fb if fb_diff and name == "M2b_s1-kind-all" else fb)
        if drop != name:
            np.savez(ev / f"eval_{name}.npz", **arrs)
        if banked or no_json == name:
            return
        raw = {"traj": {"ade6_m": float(a6.mean())}, "tactical": {"acc": 0.5}, "strategic": {"pos_err": 1.0}}
        if no_fam and no_fam[0] == name:
            raw.pop(no_fam[1])
        seed = name.split("_s")[1][0]
        js = {"tag": name, "ckpt": f"D:\\tanitad\\runs\\v8f_mini_full\\M2b_s{seed}\\ckpt_last.pt", "tac": "pred" if own else "oracle", "raw": raw}
        if ckpt_bad == name:
            js["ckpt"] = f"D:\\tanitad\\runs\\v8f_mini_full\\M2b_s{1 - int(seed)}\\ckpt_last.pt"
        if not own:
            spec = R.spec_of(name.split("-", 1)[1])
            js["tac_manipulation"] = {"mode": "oracle", "groups": spec, "n_code_dims_replaced": len(M.code_group_index(spec)), "alpha": 1.0}
            if bad_man and bad_man[0] == name:
                js["tac_manipulation"][bad_man[1]] = bad_man[2]
        json.dump(js, open(ev / f"eval_{name}.json", "w", encoding="utf-8"))

    for s in (0, 1):
        c = f"M2b_s{s}"
        zz = np.random.default_rng(100 + s).normal(0, 1, n)
        own = 1.3 + 0.2 * np.clip(zz, -2, 2) + 0.2 * z0 + (SEED_GAP if s else 0.0)
        full = own - 1.1 + 0.0 * z0
        arms = {"kind-own": own + (own_dev if s == 0 else 0.0)}
        arms["kind-all"] = np.where(fb, own, full) + (all_dev if s == 0 else 0.0)
        for k in KINDS:
            arms[f"kind-only-{k}"] = np.where(fb, own, own - g[s][k] + rng.normal(0, noise, n))
            arms[f"kind-loo-{k}"] = np.where(fb, own, full + loo + rng.normal(0, noise, n))
        arms["kind-union"] = np.where(fb, own, own - u[s] + rng.normal(0, noise, n))
        for suf, ade in arms.items():
            write(f"{c}-{suf}", np.maximum(ade, 0.05), None, own=(suf == "kind-own"))
        # the banked controls: the same windows the re-runs reproduce
        write(c, np.maximum(own, 0.05), None, banked=True)
        write(f"{c}-tacorc", np.maximum(np.where(fb, own, full), 0.05), None, banked=True)
    w.write_gt()
    w.write_block()
    sj = d / "ckpt_sha256.json"
    sj.write_text(json.dumps(sha or R.BANKED_CKPT_SHA), encoding="utf-8")
    cu, tit = _strata_labels(w, layout)
    st = _write_strata(d / "strata.npz", w, fb, cu, tit, **(strata_over or {}))
    return {"d": d, "w": w, "eval": ev, "out": d / "kind.json", "lead": np.array([s == "LEAD" for s in w.states]), "strata": st, "fb": fb, "curve": cu, "tit": tit,
            "args": ["--eval-dir", str(ev), "--ckpt-sha", str(sj), "--out", str(d / "kind.json"), "--n-boot", "300", "--dk-n-boot", "20", "--n-windows", str(N_WIN),
                     "--n-episodes", str(N_EP), "--lead-block", str(d / "lead.npz"), "--gt-npz", str(d / "gt.npz"), "--strata", str(st)]}


@pytest.fixture
def no_known(monkeypatch):
    """F = 0.1207 m is the REAL banked premise (`known_values`); a synthetic world cannot hold it. One test runs the real check."""
    monkeypatch.setattr(R, "known_values", lambda F: {"checked": False, "why": "synthetic test input", "F": F})


def _go(f, extra=("--no-families",)):
    rc = R.main([*f["args"], *extra])
    return rc, (json.loads(f["out"].read_text(encoding="utf-8")) if rc == 0 and f["out"].exists() else None)


def _w(x):
    return np.mean(x)


def _paired_mean(f, a, b, s=0):
    za, zb = np.load(f["eval"] / f"eval_M2b_s{s}-{a}.npz"), np.load(f["eval"] / f"eval_M2b_s{s}-{b}.npz")
    return float(za["ade6"].mean() - zb["ade6"].mean())


@needs_pkg
def test_reader_constants_are_the_prereg_s(tmp_path):
    assert R.BAR_2F == 0.2415 and R.ADD_BAR == 0.8 and (R.SPEED_HI, R.SPEED_LO) == (0.20, 0.05) and R.B == 2000 and (R.EVAL_WINDOWS, R.EVAL_EPISODES) == (2399, 141)
    assert R.KINDS == KINDS and R.SEEDS == (0, 1) and R.ARM == "M2b" and R.K_DK == 10 and R.IDX_2S == 9
    assert R.OUTCOMES == ("ONE KIND CARRIES", "DIFFUSE", "INTERACTIVE", "MIXED")
    assert R.FAMILIES == ("longitudinal", "lateral", "tactical", "strategic")
    assert len(R.arm_suffixes()) == 15 and len(set(R.arm_suffixes())) == 15
    assert R.spec_of("kind-own") is None and R.spec_of("kind-all") == "all" and R.spec_of("kind-union") == "constraints"
    assert R.spec_of("kind-only-tte") == "tte" and R.spec_of("kind-loo-tte") == "all-tte"
    if PREREG.exists():                                                                # the committed next-step cells are the PREREG's, verbatim
        txt = PREREG.read_text(encoding="utf-8")
        assert "2F = 0.2415 m" in txt and "A ≥ 0.8" in txt and "B = 2,000" in txt
        for outcome, nxt in R.NEXT_TEXT.items():
            assert f"**{outcome}**" in txt and nxt in txt, outcome
    assert R.PGT.KNOWN["F"] == 0.1207, "2F = 0.2415 is the PREREG's rounding of 2 x 0.1207"
    assert R.PGT.IDX_2S == R.IDX_2S, "LAMBDA's horizon is PGT's"


def _decide_rows(g0, g1, A0=None, A1=None, sep=True):
    def row(g, A):
        return {"gain": {k: {"point": g.get(k, 0.0), "separated": sep, "lo": g.get(k, 0.0) - 0.01, "hi": g.get(k, 0.0) + 0.01} for k in KINDS}, "A": A}
    return {0: row(g0, A0), 1: row(g1, A1)}


@needs_pkg
def test_decide_reaches_every_row_of_the_committed_table_and_its_boundaries():
    d = R.decide
    assert d(_decide_rows({"speed": 0.5}, {"speed": 0.5}, 0.9, 0.3))["outcome"] == "ONE KIND CARRIES", "a carrier wins whatever A reads"
    assert d(_decide_rows({"speed": 0.5}, {"speed": 0.5}, 0.9, 0.3))["carriers"] == ["speed"]
    assert d(_decide_rows({"speed": 0.5, "geom": 0.3}, {"speed": 0.4, "geom": 0.25}, 1.0, 1.0))["carriers"] == ["speed", "geom"], "more than one kind can carry"
    assert d(_decide_rows({}, {}, 0.8, 0.8))["outcome"] == "DIFFUSE", "A = 0.8 exactly is DIFFUSE (A >= 0.8)"
    assert d(_decide_rows({}, {}, 0.7999, 0.9))["outcome"] == "MIXED", "one seed below the bar: the seeds disagree"
    assert d(_decide_rows({}, {}, 0.79, 0.4))["outcome"] == "INTERACTIVE"
    assert d(_decide_rows({}, {}, 1.2, 1.1))["outcome"] == "DIFFUSE"
    assert d(_decide_rows({}, {}, None, 0.9))["outcome"] == "MIXED" and d(_decide_rows({}, {}, None, None))["outcome"] == "MIXED", "A undefined: no branch"
    assert d(_decide_rows({}, {}, -0.5, 0.2))["outcome"] == "INTERACTIVE", "a negative A is below the bar"
    # the 2F bar: >= 0.2415 clears, just under does not; separated is also needed; the sign matters
    assert d(_decide_rows({"tte": 0.2415}, {"tte": 0.2415}, 0.9, 0.9))["outcome"] == "ONE KIND CARRIES"
    assert d(_decide_rows({"tte": 0.24149}, {"tte": 0.2415}, 0.9, 0.9))["outcome"] == "DIFFUSE"
    assert d(_decide_rows({"tte": 0.9}, {"tte": 0.9}, 0.3, 0.3, sep=False))["outcome"] == "INTERACTIVE", "a gain that is not separated carries nothing"
    assert d(_decide_rows({"tte": -0.9}, {"tte": -0.9}, 0.3, 0.3))["outcome"] == "INTERACTIVE", "a separated HARM is not a gain"
    # one seed is not a carrier: the explicit condition decides (DIFFUSE here), and the one-seed kind is named
    r = d(_decide_rows({"speed": 0.6}, {"speed": 0.1}, 0.9, 0.9))
    assert r["outcome"] == "DIFFUSE" and r["carriers"] == [] and r["kinds_clearing_at_one_seed_only"] == {"speed": ["s0"]}
    assert d(_decide_rows({"speed": 0.6}, {"geom": 0.6}, 0.9, 0.9))["outcome"] == "DIFFUSE", "different kinds at the two seeds is no carrier"
    assert d(_decide_rows({"speed": 0.6}, {"speed": 0.1}, 0.9, 0.3))["outcome"] == "MIXED"
    # the SPEED side-read
    s = R.speed_side_read(_decide_rows({"speed": 0.20}, {"speed": 0.35}))
    assert s["read"] == "ASSOCIATION-RIGHT" and s["decides"] == "nothing alone"
    assert R.speed_side_read(_decide_rows({"speed": 0.05}, {"speed": 0.0}))["read"] == "DOSE-MODEL-RIGHT"
    assert R.speed_side_read(_decide_rows({"speed": 0.20}, {"speed": 0.199}))["read"] == "NEITHER"
    assert R.speed_side_read(_decide_rows({"speed": 0.0501}, {"speed": 0.0}))["read"] == "NEITHER"
    assert R.speed_side_read(_decide_rows({"speed": 0.25}, {"speed": 0.04}))["read"] == "NEITHER", "both seeds are needed"


@needs_pkg
def test_reader_end_to_end_one_kind_carries_and_every_number_is_the_independent_recomputation(tmp_path, no_known, capsys):
    f = _kind_world(tmp_path / "one", g0=_gains(speed=0.6), g1=_gains(speed=0.55))
    rc, r = _go(f, extra=())                                                          # families ON: the paired LON / LAT blocks too
    assert rc == 0, capsys.readouterr().err
    assert r["DECISION"] == "ONE KIND CARRIES" and r["decision"]["carriers"] == ["speed"] and r["VERDICT"] == r["DECISION"]
    for s in (0, 1):
        row = r["rows"][f"s{s}"]
        for k in KINDS:                                                               # gain_k = ADE6(own) - ADE6(only-k), necessity_k = ADE6(loo-k) - ADE6(all): recomputed from the files
            assert row["gain"][k]["point"] == pytest.approx(_paired_mean(f, "kind-own", f"kind-only-{k}", s), abs=1e-12), (s, k)
            assert row["necessity"][k]["point"] == pytest.approx(_paired_mean(f, f"kind-loo-{k}", "kind-all", s), abs=1e-12), (s, k)
        gu = _paired_mean(f, "kind-own", "kind-union", s)
        assert row["gain_union"]["point"] == pytest.approx(gu, abs=1e-12)
        assert row["A"] == pytest.approx(sum(_paired_mean(f, "kind-own", f"kind-only-{k}", s) for k in KINDS) / gu, rel=1e-12)
        assert row["A_interval"]["lo"] < row["A"] < row["A_interval"]["hi"] and "descriptive" in row["A_interval"]["estimator"]
        assert row["gain"]["speed"]["point"] > 0.5 and all(row["gain"][k]["point"] < 0.1 for k in KINDS if k != "speed")
    F = abs(float(np.load(f["eval"] / "eval_M2b_s0.npz")["ade6"].mean()) - float(np.load(f["eval"] / "eval_M2b_s1.npz")["ade6"].mean()))      # the seed floor, from the banked files
    assert r["bar_2F_m"] == 0.2415 and r["F_measured"] == pytest.approx(F, abs=1e-12) and r["two_F_measured"] == pytest.approx(2 * F, abs=1e-12)
    assert r["bar_2F_m"] != r["two_F_measured"], "the committed bar is NOT recomputed from the inputs (a synthetic world's F is not the banked 0.1207)"
    assert r["speed_side_read"]["read"] == "ASSOCIATION-RIGHT"
    assert all(v["per_window_equal"] and v["delta"] == 0.0 and v["lo"] == 0.0 and v["hi"] == 0.0 for s in ("s0", "s1") for v in r["identity_control"][s].values())
    assert r["checkpoint_sha256"] == R.BANKED_CKPT_SHA and r["families_skipped"] is False
    assert "speed -> T's speed variance" in r["next"] and "ONE KIND CARRIES" in r["reading"] and "CONFOUND" not in r["next"], "the confound caveat is for the two O-side rows"
    assert r["partition"]["n_cells"] == KIND_SIZES and r["partition"]["union_is_constraints"] is True and r["confound"].startswith("O was trained on WHOLE codes")
    assert "T0" in r["tier"] and "EPISODES" in r["estimator"] and r["n_windows"] == N_WIN and r["n_episodes"] == N_EP


@needs_pkg
def test_reader_end_to_end_beside_every_arm_carries_the_four_families_and_lambda(tmp_path, no_known):
    f = _kind_world(tmp_path / "bes", shift_x={"kind-only-speed": 0.5, "kind-own": 0.3})
    rc, r = _go(f, extra=())
    assert rc == 0
    for s in ("s0", "s1"):
        assert set(r["beside"][s]) == set(R.arm_suffixes()), "every oracle arm and both controls"
        for suf, blk in r["beside"][s].items():
            for fam in R.FAMILIES:
                assert blk[fam], (s, suf, fam)
            assert blk["longitudinal"]["distance_keeping_k10"]["arm"] and blk["lateral"]["paired_vs_own"] and blk["longitudinal"]["paired_vs_own"]
            assert blk["tactical"] == {"acc": 0.5} and blk["strategic"] == {"pos_err": 1.0}, "read from the arm's own json (strategic is NOT paused)"
            if suf != "kind-own":
                assert blk["longitudinal"]["distance_keeping_k10"]["pair_vs_own"]
        assert r["beside"][s]["kind-all"]["lambda"]["mean_lead_m"] == 0.0, "LAMBDA of the full oracle against itself is exactly 0"
        assert r["beside"][s]["kind-all"]["lambda"]["paired_vs_full_oracle"]["delta"] == 0.0
        lam = r["beside"][s]["kind-only-speed"]["lambda"]
        # ANALYTIC: this arm's plan is 0.5 m ahead of the full oracle's at every step, on every LEAD window
        assert lam["mean_lead_m"] == pytest.approx(0.5, abs=1e-6) and lam["n_lead_windows"] == int(f["lead"].sum())
        assert lam["paired_vs_full_oracle"]["separated"] and lam["paired_vs_full_oracle"]["point"] == pytest.approx(0.5, abs=1e-6)
        assert r["beside"][s]["kind-only-angle"]["lambda"]["mean_lead_m"] == pytest.approx(0.0, abs=1e-9)
        # LAMBDA is read against the FULL ORACLE, not against T's own code: T's own plan is 0.3 m ahead of it, the speed arm 0.5 m
        assert r["beside"][s]["kind-own"]["lambda"]["mean_lead_m"] == pytest.approx(0.3, abs=1e-6)
    assert r["strategic_family"].startswith("NOT PAUSED") and r["n_lead_windows_k10"] == int(f["lead"].sum())
    # with the paired blocks skipped the lateral family says so, and the read says so
    rc, r2 = _go(f)
    assert rc == 0 and r2["families_skipped"] is True and "SKIPPED" in r2["beside"]["s0"]["kind-own"]["lateral"]["status"]


@needs_pkg
def test_reader_end_to_end_diffuse_interactive_mixed_and_undefined(tmp_path, no_known):
    f = _kind_world(tmp_path / "dif", g0=_gains(0.2), g1=_gains(0.2))
    rc, r = _go(f)
    assert rc == 0 and r["DECISION"] == "DIFFUSE" and r["decision"]["carriers"] == [] and all(a >= 0.8 for a in r["decision"]["A"]), r["reading"]
    assert "CELL-WISE mixed codes" in r["next"] and "CONFOUND" in r["next"] and "LOWER bound" in r["next"]
    f = _kind_world(tmp_path / "int", g0=_gains(0.05), g1=_gains(0.05))
    rc, r = _go(f)
    assert rc == 0 and r["DECISION"] == "INTERACTIVE" and all(a < 0.8 for a in r["decision"]["A"]) and "deprioritised" in r["next"] and "CONFOUND" in r["next"], r["reading"]
    f = _kind_world(tmp_path / "mix", g0=_gains(0.2), g1=_gains(0.05))
    rc, r = _go(f)
    assert rc == 0 and r["DECISION"] == "MIXED" and r["decision"]["A"][0] >= 0.8 > r["decision"]["A"][1] and "no branch from this PREREG" in r["next"], r["reading"]
    assert r["decision"]["per_seed_row"] == {"s0": "DIFFUSE", "s1": "INTERACTIVE"}
    f = _kind_world(tmp_path / "und", g0=_gains(0.2), g1=_gains(0.2), u1=-0.3)
    rc, r = _go(f)
    assert rc == 0 and r["DECISION"] == "MIXED" and r["rows"]["s1"]["A"] is None and r["rows"]["s1"]["A_interval"]["status"] == "NOT-DEFINED" and "undefined" in r["reading"], r["reading"]
    assert r["decision"]["per_seed_row"]["s1"] == "A UNDEFINED"
    f = _kind_world(tmp_path / "one", g0=_gains(speed=0.6, geom=0.1), g1=_gains(speed=0.0, geom=0.1), u0=1.0, u1=1.0)
    rc, r = _go(f)
    assert rc == 0 and r["decision"]["carriers"] == [] and r["decision"]["kinds_clearing_at_one_seed_only"] == {"speed": ["s0"]}, "a kind that clears at one seed is not a carrier"
    assert r["DECISION"] in ("INTERACTIVE", "DIFFUSE", "MIXED") and r["speed_side_read"]["read"] == "NEITHER"
    f = _kind_world(tmp_path / "dose", g0=_gains(0.03), g1=_gains(0.02))
    rc, r = _go(f)
    assert rc == 0 and r["speed_side_read"]["read"] == "DOSE-MODEL-RIGHT"


@needs_pkg
def test_reader_a_gain_inside_noise_does_not_carry_even_beyond_the_bar(tmp_path, no_known, capsys):
    """Separated is part of the bar: a mean gain beyond 2F whose paired episode-cluster interval spans zero carries nothing. The speed gain is +1.2 m in five episodes, -0.6 m in
    three and +0.05 m in the other four at BOTH seeds; the premise (point >= 2F, interval spans 0) is asserted from the reader's own output, so the test cannot pass vacuously."""
    n = N_WIN
    ep = np.repeat(np.arange(N_EP), n // N_EP)
    pattern = np.where(ep < 5, 1.2, np.where(ep < 8, -0.6, 0.05))
    f = _kind_world(tmp_path / "ns", g0=_gains(0.05, speed=pattern), g1=_gains(0.05, speed=pattern))
    rc, r = _go(f)
    assert rc == 0, capsys.readouterr().err
    for s in ("s0", "s1"):
        g = r["rows"][s]["gain"]["speed"]
        assert g["point"] >= R.BAR_2F and g["lo"] <= 0.0 <= g["hi"] and g["separated"] is False, (s, g)       # the premise
    assert r["decision"]["carriers"] == [] and r["decision"]["kinds_clearing_at_one_seed_only"] == {}, "a gain inside noise carries nothing, at either seed"
    assert r["DECISION"] != "ONE KIND CARRIES"


@needs_pkg
def test_reader_refuses_every_way_the_inputs_can_be_wrong_with_rc2(tmp_path, capsys, no_known):
    s0, s1 = "M2b_s0", "M2b_s1"
    cases = (("own", dict(own_dev=0.003), "IDENTITY CONTROL FAILED: eval_M2b_s0-kind-own - eval_M2b_s0 reads"),
             ("all", dict(all_dev=0.003), "IDENTITY CONTROL FAILED: eval_M2b_s0-kind-all - eval_M2b_s0-tacorc reads"),
             ("miss", dict(drop=f"{s1}-kind-loo-geom"), f"eval_{s1}-kind-loo-geom.npz is missing"),
             ("missbanked", dict(drop=f"{s0}-tacorc"), f"eval_{s0}-tacorc.npz is missing"),
             ("missstd", dict(drop=s1), f"eval_{s1}.npz is missing"),
             ("nojson", dict(no_json=f"{s0}-kind-only-tte"), f"eval_{s0}-kind-only-tte.json is missing"),
             ("eid", dict(eid_swap=True), "is not on the banked window list (eid differs)"),
             ("fb", dict(fb_diff=True), "disagree about which windows have no v9 row"),
             ("dims", dict(bad_man=(f"{s0}-kind-only-speed", "n_code_dims_replaced", 8)), "5 / 1.0"),
             ("groups", dict(bad_man=(f"{s1}-kind-loo-angle", "groups", "all-speed")), "expected oracle / all-angle / 57 / 1.0"),
             ("alpha", dict(bad_man=(f"{s0}-kind-union", "alpha", 0.5)), "alpha 0.5, expected oracle / constraints / 26 / 1.0"),
             ("mode", dict(bad_man=(f"{s0}-kind-all", "mode", "absmask")), "absmask / all / 60"),
             ("ckpt", dict(ckpt_bad=f"{s0}-kind-only-geom"), "scored 'D:\\\\tanitad\\\\runs\\\\v8f_mini_full\\\\M2b_s1\\\\ckpt_last.pt', not <runs>/M2b_s0/ckpt_last.pt"),
             ("notactical", dict(no_fam=(f"{s0}-kind-loo-tte", "tactical")), "carries no tactical family (raw.tactical)"),
             ("nostrategic", dict(no_fam=(f"{s1}-kind-own", "strategic")), "eval_M2b_s1-kind-own.json carries no strategic family (raw.strategic)"),
             ("shaw", dict(sha={s0: "0" * 64, s1: R.BANKED_CKPT_SHA[s1]}), "M2b_s0 has sha256 0000000000000000..., the banked run's is 601cb43526731786"),
             ("shamiss", dict(sha={s0: R.BANKED_CKPT_SHA[s0]}), "M2b_s1 has sha256 None"))
    for name, kw, msg in cases:
        f = _kind_world(tmp_path / name, **kw)
        assert _go(f)[0] == 2, name
        err = capsys.readouterr().err
        assert "KIND READ REFUSED" in err or "KIND READ INPUT ERROR" in err, (name, err)
        assert msg in err, (name, err)
    f = _kind_world(tmp_path / "counts")
    assert _go(f, extra=("--no-families", "--n-windows", "2399", "--n-episodes", "141"))[0] == 2
    assert "the evals hold 72 windows over 12 episodes, the committed eval-141 grid has 2399 over 141" in capsys.readouterr().err
    f = _kind_world(tmp_path / "nosha")
    assert R.main([a for i, a in enumerate(f["args"]) if a != "--ckpt-sha" and f["args"][i - 1] != "--ckpt-sha"] + ["--no-families"]) == 2
    assert "--ckpt-sha is required" in capsys.readouterr().err


@needs_pkg
def test_reader_refuses_a_partition_that_is_not_one_and_a_model_that_moved_a_cell(tmp_path, capsys, no_known, monkeypatch):
    f = _kind_world(tmp_path / "ok")
    assert _go(f)[0] == 0, "the unmutated world reads"
    moved = {k: list(v) for k, v in M.CODE_KINDS.items()}
    moved["speed"].remove(C0 + M.CI["v_end_ms"])
    moved["angle"].append(C0 + M.CI["v_end_ms"])                                      # still a partition of the 26 -- but not the PREREG's
    dropped = {k: list(v) for k, v in M.CODE_KINDS.items()}
    dropped["tte"].pop()
    with monkeypatch.context() as mp:                                                 # each patch is undone by its own context: `no_known` stays in force
        mp.setattr(M, "CODE_KINDS", {k: tuple(v) for k, v in moved.items()})
        assert _go(f)[0] == 2 and "CODE_KINDS['speed']" in capsys.readouterr().err
    with monkeypatch.context() as mp:
        mp.setattr(M, "CODE_KINDS", {k: tuple(v) for k, v in dropped.items()})
        assert _go(f)[0] == 2 and "CODE_KINDS['tte']" in capsys.readouterr().err
    with monkeypatch.context() as mp:                                                 # the PREREG's own table, corrupted in the reader: not a partition of the 26 cells
        mp.setattr(R, "KIND_CELLS", {**R.KIND_CELLS, "geom": R.KIND_CELLS["geom"][:-1] + ("lead_gap_m",)})
        assert _go(f)[0] == 2 and "not a partition of the 26 constraint cells" in capsys.readouterr().err
    with monkeypatch.context() as mp:                                                 # the six kinds are right, but the `constraints` composite is not their union
        mp.setattr(M, "CODE_COMPOSITES", {**M.CODE_COMPOSITES, "constraints": ("latcons", "lonspeed")})
        assert _go(f)[0] == 2 and "the union of the six kinds is not the `constraints` composite index for index" in capsys.readouterr().err
    real = M.code_group_index
    with monkeypatch.context() as mp:                                                 # a kind that the model's code_group_index resolves differently from the kind's cells
        mp.setattr(M, "code_group_index", lambda spec: real("goal") if spec == "geom" else real(spec))
        assert _go(f)[0] == 2 and "code_group_index('geom') does not return the kind's cells" in capsys.readouterr().err
    assert _go(f)[0] == 0, "every patch was undone and the world reads again"


@needs_pkg
def test_reader_known_value_control_refuses_a_world_whose_F_is_not_the_banked_one(tmp_path, capsys):
    f = _kind_world(tmp_path / "f")
    F = abs(float(np.load(f["eval"] / "eval_M2b_s0.npz")["ade6"].mean()) - float(np.load(f["eval"] / "eval_M2b_s1.npz")["ade6"].mean()))
    assert abs(F - 0.1207) > 6e-5, "premise: the synthetic world's seed floor is not the PREREG's 0.1207 m to the control's tolerance"
    assert _go(f)[0] == 2 and "not the inputs the PREREG was written against" in capsys.readouterr().err
    assert R.known_values(0.1207)["checked"] is True and R.known_values(0.12073)["checked"] is True
    with pytest.raises(SystemExit):
        R.known_values(0.1300)
    assert "F reads 0.13000 m" in capsys.readouterr().err


@needs_pkg
def test_reader_identity_instrument_is_exact_zero_when_the_rerun_is_the_banked_eval(tmp_path, no_known):
    f = _kind_world(tmp_path / "id")
    rc, r = _go(f)
    assert rc == 0
    for s in ("s0", "s1"):
        assert set(r["identity_control"][s]) == {"own_vs_banked_eval", "all_vs_banked_tacorc"}
        assert all(v["point"] == 0.0 and v["per_window_equal"] for v in r["identity_control"][s].values())
    # ... and a one-bit deviation in a single window of the re-run is NOT +0 [0, 0]: the control is sharp
    p = f["eval"] / "eval_M2b_s1-kind-all.npz"
    z = dict(np.load(p))
    z["ade6"] = z["ade6"].copy()
    z["ade6"][5] = np.nextafter(z["ade6"][5], 10.0)
    np.savez(p, **z)
    assert _go(f)[0] == 2


@needs_pkg
def test_reader_check_inputs_validates_the_banked_side_alone(tmp_path, capsys, monkeypatch):
    f = _PGT._pgt_files(tmp_path / "chk")
    monkeypatch.setattr(R.PGT, "known_values", lambda *a, **k: {"checked": False})
    monkeypatch.setattr(R, "known_values", lambda F: {"checked": False})
    out = tmp_path / "chk" / "inputs.json"
    base = ["--check-inputs", "--eval-dir", str(f["eval"]), "--runs-dir", str(f["runs"]), "--out", str(out), "--n-boot", "300", "--dk-n-boot", "60", "--no-families",
            "--n-windows", str(N_WIN), "--n-episodes", str(N_EP), "--lead-block", str(f["d"] / "lead.npz"), "--p10-pred", str(f["d"] / "p10.npy"), "--gt-npz", str(f["d"] / "gt.npz"),
            "--no-strata"]
    assert R.main(base) == 0
    msg = capsys.readouterr().out
    assert "KIND INPUTS OK" in msg and "partition 26 = 5 + 3 + 5 + 2 + 9 + 2" in msg and "expects 30 evals (15 per seed)" in msg
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["expected_evals"] == 30 and res["expected_suffixes"] == R.arm_suffixes() and res["partition"]["union_is_constraints"] is True
    assert res["banked"]["identity_control"]["s0"]["mdk"]["delta"] == 0.0 and res["n_windows"] == N_WIN
    # the committed counts are checked on the banked side too; a corrupt banked run stops the chain while box 2 is idle
    assert R.main([*base, "--n-windows", "2399", "--n-episodes", "141"]) == 2                              # a later flag wins: the committed grid size
    assert "committed eval-141 grid" in capsys.readouterr().err
    ru = Path(f["runs"]) / "M2b_s1" / "summary.json"
    keep = ru.read_text(encoding="utf-8")
    ru.write_text("{", encoding="utf-8")
    assert R.main(base) == 2
    ru.write_text(keep, encoding="utf-8")
    (Path(f["eval"]) / "eval_M2b_s0-tacorc.npz").rename(Path(f["eval"]) / "moved.npz")
    assert R.main(base) == 2 and "eval_M2b_s0-tacorc.npz is missing" in capsys.readouterr().err
    assert R.main(["--check-inputs", "--eval-dir", str(f["eval"]), "--out", str(out)]) == 2 and "--runs-dir is required" in capsys.readouterr().err


@needs_pkg
def test_reader_the_estimator_is_the_paired_episode_cluster_bootstrap_and_a_seed_replicate_is_not_claimed(tmp_path, no_known):
    f = _kind_world(tmp_path / "est")
    rc, r = _go(f)
    assert rc == 0 and r["rows"]["s0"]["gain"]["speed"]["estimator"].startswith("paired_") and r["rows"]["s0"]["gain"]["speed"]["n_boot"] == 300
    assert r["rows"]["s0"]["gain"]["speed"]["n_episodes"] == N_EP and "another draw of EPISODES" in r["estimator"] and "H-ESTIM-SEED-1" in r["estimator"]
    assert R.B == 2000, "the production default is the PREREG's B = 2,000"
    import taniteval.ci as CI
    assert R.PGT.paired is not CI.paired_episode_cluster_bootstrap and "pf(" in Path(R.PGT.__file__).read_text(encoding="utf-8")


# =================================================================================================================================================
# AMENDMENT 1: the window strata (beside, deciding nothing)
# =================================================================================================================================================
STRATA_FILE = PKG / "raw" / "2026-10-25-kind" / "window_strata_eval141.npz"
STRATA_BUILDER = PKG / "raw" / "2026-10-25-kind" / "mk_window_strata.py"
V9_BUILDER = REPO / "stack" / "scripts" / "build_v9_labels.py"
RELEASE = Path("C:/Users/Admin/box2_ship/v9ship/v9_labels_eval139.npz")
needs_strata = pytest.mark.skipif(not STRATA_FILE.exists(), reason="the v8F-mini package (raw/2026-10-25-kind) is not in this checkout")
STRATA_STRATUM_SHA = "560dac3f59e22871f94451365cf06c2d5c64bb33fbb8071542a180bb45c2b7e9"
STRATA_COUNTS, STRATA_EPISODES = {"JUNCTION": 298, "CURVE": 140, "NO-TURN": 1961}, {"JUNCTION": 34, "CURVE": 17, "NO-TURN": 140}


def _same(a, b):
    """json-level equality of two reads."""
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def _with_strata(f, path, out):
    a = list(f["args"])
    a[a.index("--strata") + 1] = str(path)
    a[a.index("--out") + 1] = str(out)
    return {**f, "args": a, "out": Path(out)}


@needs_pkg
@needs_strata
def test_strata_file_is_the_eval_grids_and_is_pinned_to_its_inputs():
    z = np.load(STRATA_FILE, allow_pickle=False)
    s, prov = z["stratum"], json.loads(str(z["provenance"]))
    assert R.STRATA == ("JUNCTION", "CURVE", "NO-TURN") and R.MIN_STRATUM_EPISODES == 10
    assert Path(R.STRATA_DEFAULT).resolve() == STRATA_FILE.resolve(), "the reader's default strata file is the one that rides in the PK archive of the chain"
    assert s.shape == (2399,) and s.dtype == np.int8 and hashlib.sha256(np.ascontiguousarray(s).tobytes()).hexdigest() == STRATA_STRATUM_SHA
    assert {nm: int((s == i).sum()) for i, nm in enumerate(R.STRATA)} == STRATA_COUNTS == prov["counts"]
    assert {nm: int(np.unique(z["eid"][s == i]).size) for i, nm in enumerate(R.STRATA)} == STRATA_EPISODES == prov["episodes"]
    assert prov["release_md5"] == R.RELEASE_MD5 == "6b5c7f207cffc3b7eb3cd527fd433599" and prov["n_both_curve_and_turn"] == 0 and prov["n_no_v9_row"] == int((~z["has"]).sum()) == 34
    # the stratum is a function of the two label columns and the v9-row presence (the reader's restatement)
    assert np.array_equal(s, R.stratum_codes(z["curve"], z["turn_is_turn"], z["has"]))
    assert not ((z["curve"] == 1) & (z["turn_is_turn"] == 1)).any(), "curve = int(big and not is_junc): never both"
    # the windows ARE the eval grid's, the T-tick row is the evaluator's, and the file was built from THIS join and THIS grid
    grid = json.loads((PKG / "raw" / "eval_grid_windows.json").read_text(encoding="utf-8"))
    ci = np.array([k for k, c in enumerate(grid["order"]) for _ in grid["windows"][c]])
    tt = np.array([t for c in grid["order"] for t in grid["windows"][c]])
    assert np.array_equal(z["eid"], ci) and np.array_equal(z["t"], tt)
    J = dict(np.load(PKG / "raw" / "join_eval141.npz", allow_pickle=True))
    i = np.arange(len(ci))
    a_t = np.minimum(np.minimum(i % 3, np.maximum(tt - 3, 0)), 2)
    assert np.array_equal(z["a_t"], a_t) and np.array_equal(z["rowT"], J["off"][ci] + tt - a_t) and np.array_equal(z["has"], J["has"][z["rowT"]])
    assert "curve" not in J and "turn_is_turn" not in J, "the shipped join does not carry the two columns: they come from the v9 release (mk_window_strata.py)"
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()                              # noqa: E731
    assert prov["join_sha256"] == sha(PKG / "raw" / "join_eval141.npz") and prov["grid_sha256"] == sha(PKG / "raw" / "eval_grid_windows.json"), "rebuild the strata when the join or grid moves"
    audit = PKG / "raw" / "2026-10-25-t-attrib" / "table.npz"
    if audit.exists():                                                                              # the banked audit paired the same rows
        T = np.load(audit, allow_pickle=True)
        assert all(np.array_equal(z[k], T[kk]) for k, kk in (("eid", "ci"), ("t", "tt"), ("a_t", "a_t"), ("rowT", "rowT"), ("has", "has")))


@needs_pkg
@needs_strata
def test_strata_definitions_are_the_amendments_and_the_builders():
    if V9_BUILDER.exists():                                                                         # CURVE = curve = 1; the builder writes it as `big and not is_junc`
        src = V9_BUILDER.read_text(encoding="utf-8")
        assert "is_junc = bool(big and dom is not None and dom.is_turn)" in src and 'col("curve", 0, np.int8)[i] = int(big and not is_junc)' in src
        assert 'col("turn_is_turn", 0, np.int8)[i] = int(dom.is_turn)' in src
    B = _load("mk_window_strata_for_test", STRATA_BUILDER)
    rng = np.random.default_rng(0)
    cu, ti = rng.integers(0, 2, 500).astype(np.int8), rng.integers(0, 2, 500).astype(np.int8)
    has = rng.random(500) < 0.9
    assert np.array_equal(B.stratum_of(cu, ti, has), R.stratum_codes(cu, ti, has)) and tuple(B.STRATA) == R.STRATA
    s = R.stratum_codes(np.array([1, 0, 0, 1, 1, 0]), np.array([0, 1, 0, 1, 0, 0]), np.array([True, True, True, True, False, False]))
    assert s.tolist() == [1, 0, 2, 0, 2, 2], "CURVE, JUNCTION, NO-TURN, JUNCTION wins a tie, a window with no v9 row is NO-TURN, the rest NO-TURN"
    if PREREG.exists():
        txt = PREREG.read_text(encoding="utf-8")
        assert "## Amendment 1" in txt and "fewer than 10 episodes" in txt and "turn_is_turn" in txt and "`curve` = 1" in txt


@pytest.mark.skipif(not RELEASE.exists(), reason="the v9 eval139 release is on the dev box only")
@needs_strata
def test_strata_file_is_reproduced_by_its_builder_from_the_release(tmp_path):
    B = _load("mk_window_strata_for_repro", STRATA_BUILDER)
    out = tmp_path / "again.npz"
    assert B.main(["--release", str(RELEASE), "--out", str(out)]) == 0
    a, b = np.load(STRATA_FILE, allow_pickle=False), np.load(out, allow_pickle=False)
    for k in ("stratum", "curve", "turn_is_turn", "has", "eid", "t", "a_t", "rowT"):
        assert np.array_equal(a[k], b[k]), k


@needs_pkg
def test_strata_numbers_are_the_independent_recomputation_and_tile_the_pooled_gain(tmp_path, no_known, capsys):
    from taniteval import ci as CI
    for layout, floor in (("few", {"JUNCTION": False, "CURVE": False, "NO-TURN": True}), ("many", {"JUNCTION": True, "CURVE": False, "NO-TURN": True})):
        f = _kind_world(tmp_path / layout, g0=_gains(0.2, speed=0.5), g1=_gains(0.1, speed=0.4), layout=layout)
        rc, r = _go(f)
        assert rc == 0, capsys.readouterr().err
        z, eid = np.load(f["strata"]), f["w"].eid
        st = r["strata"]
        assert st["decides"].startswith("nothing") and set(st["strata"]) == set(R.STRATA) and "predicted-goal input stays on the list" in st["read_with"]
        assert sum(st["strata"][nm]["n_windows"] for nm in R.STRATA) == N_WIN, "the three strata tile the windows"
        for i, nm in enumerate(R.STRATA):
            m = z["stratum"] == i
            blk = st["strata"][nm]
            assert blk["n_windows"] == int(m.sum()) and blk["n_episodes"] == int(np.unique(eid[m]).size) and blk["n_windows_without_v9_row"] == int(f["fb"][m].sum())
            assert (blk["interval"] == "computed") == floor[nm] == (blk["n_episodes"] >= 10), (layout, nm, blk["n_episodes"])
            for s in (0, 1):
                ev = lambda n: np.load(f["eval"] / f"eval_M2b_s{s}-{n}.npz")["ade6"]               # noqa: E731
                own = ev("kind-own")
                row = blk["seeds"][f"s{s}"]
                for k, g in (*[(k, row["gain"][k]) for k in KINDS], ("union", row["gain_union"])):
                    arm = ev("kind-union" if k == "union" else f"kind-only-{k}")
                    assert g["point"] == pytest.approx(float((own[m] - arm[m]).mean()), abs=1e-12), (layout, nm, s, k)
                    if floor[nm]:                                                                  # the paired episode-cluster bootstrap WITHIN the stratum, B = 300 here
                        want = CI.paired_episode_cluster_bootstrap(own[m], arm[m], eid[m], 300, 0)
                        assert (g["delta"], g["lo"], g["hi"], g["separated"]) == (want["delta"], want["lo"], want["hi"], want["separated"]), (layout, nm, s, k)
                    else:                                                                          # fewer than 10 episodes: n and NO interval
                        assert set(g) == {"point", "interval"} and g["interval"].startswith("NONE: fewer than 10 episodes"), (layout, nm, s, k, g)
        for s in (0, 1):                                                                           # the contributions sum to the deciding pooled gain, exactly
            for k in (*KINDS, "union"):
                c = st["contribution_to_pooled_gain_m"]["seeds"][f"s{s}"][k]
                pooled = r["rows"][f"s{s}"]["gain_union"] if k == "union" else r["rows"][f"s{s}"]["gain"][k]
                assert c["pooled_gain"] == pytest.approx(pooled["point"], abs=1e-12)
                assert sum(c[nm] for nm in R.STRATA) == pytest.approx(c["pooled_gain"], abs=1e-12), (layout, s, k)
        assert st["labels"]["release_md5"] == R.RELEASE_MD5 and len(st["labels"]["stratum_sha256"]) == 64


@needs_pkg
def test_strata_never_change_the_outcome_permuting_the_labels_leaves_the_decision_identical(tmp_path, no_known, capsys):
    """The stratum read is BESIDE: permute which windows are JUNCTION / CURVE (the label columns, among the windows that have a v9 row) and the whole deciding read is
    identical, key for key; only the `strata` block moves. A missing stratum read (--no-strata) changes nothing either."""
    f = _kind_world(tmp_path / "p", g0=_gains(0.1, speed=0.6), g1=_gains(0.1, speed=0.55), layout="many")
    rc, base = _go(f)
    assert rc == 0, capsys.readouterr().err
    assert base["DECISION"] == "ONE KIND CARRIES"
    moved = 0
    for seed in (1, 2, 3):
        idx = np.where(~f["fb"])[0]
        perm = np.random.default_rng(seed).permutation(idx)
        cu, tit = f["curve"].copy(), f["tit"].copy()
        cu[idx], tit[idx] = f["curve"][perm], f["tit"][perm]
        g = _with_strata(f, _write_strata(tmp_path / f"perm{seed}.npz", f["w"], f["fb"], cu, tit), tmp_path / f"perm{seed}.json")
        rc, r = _go(g)
        assert rc == 0, capsys.readouterr().err
        for key in base:
            if key != "strata":
                assert _same(r[key], base[key]), f"the stratum labels moved `{key}`"
        moved += int(not _same(r["strata"], base["strata"]))
    assert moved >= 2, "the permutation did move the stratum read (otherwise the test pins nothing)"
    h = _with_strata(f, f["strata"], tmp_path / "nos.json")
    rc, r = _go(h, extra=("--no-families", "--no-strata"))
    assert rc == 0 and r["strata"]["status"].startswith("SKIPPED")
    assert all(_same(r[k], base[k]) for k in base if k != "strata")


@needs_pkg
def test_strata_file_that_is_not_the_eval_grids_is_refused_with_rc2(tmp_path, capsys, no_known):
    w0 = _kind_world(tmp_path / "w0")
    w = w0["w"]
    eid_bad = w.eid.copy()
    eid_bad[[0, -1]] = eid_bad[[-1, 0]]
    s_ok = np.load(w0["strata"])["stratum"]
    s_wrong = s_ok.copy()
    s_wrong[np.where(s_ok == 2)[0][0]] = 1                                  # a NO-TURN window marked CURVE with no curve label behind it
    cases = (("eid", dict(eid=eid_bad), "episode index differs"),
             ("tick", dict(t=w.t + 1), "tick differs"),
             ("labels", dict(stratum=s_wrong), "not JUNCTION (turn_is_turn = 1) / CURVE (curve = 1) / NO-TURN"),
             ("code", dict(stratum=np.full(w.n, 7, np.int8)), "outside {0 JUNCTION, 1 CURVE, 2 NO-TURN}"),
             ("has", dict(has=np.ones(w.n, bool)), "T-tick row pairing is not the evaluator's"),
             ("short", dict(stratum=s_ok[:-1]), "holds 71 windows, the evals hold 72"),
             ("release", dict(provenance=np.array(json.dumps({"release_md5": "0" * 32}))), "was built from release 000000"),
             ("prov", dict(provenance=np.array("{")), "provenance is not json"))
    for name, over, msg in cases:
        f = _kind_world(tmp_path / name, strata_over=over)
        assert _go(f)[0] == 2, name
        err = capsys.readouterr().err
        assert "KIND READ REFUSED" in err and msg in err, (name, err)
    f = _kind_world(tmp_path / "gone")
    f["strata"].unlink()
    assert _go(f)[0] == 2 and "does not exist" in capsys.readouterr().err
    f = _kind_world(tmp_path / "nokey")
    z = {k: v for k, v in np.load(f["strata"]).items() if k != "curve"}
    np.savez(f["strata"], **z)
    assert _go(f)[0] == 2 and "is missing ['curve']" in capsys.readouterr().err
    f = _kind_world(tmp_path / "ok")
    assert _go(f)[0] == 0, "the unmutated world reads"


@needs_pkg
def test_strata_check_inputs_refuses_a_strata_file_that_is_not_the_banked_grids(tmp_path, capsys, monkeypatch):
    f = _PGT._pgt_files(tmp_path / "chk")
    monkeypatch.setattr(R.PGT, "known_values", lambda *a, **k: {"checked": False})
    monkeypatch.setattr(R, "known_values", lambda F: {"checked": False})
    gt = np.load(Path(f["d"]) / "gt.npz", allow_pickle=True)
    n = int(gt["eid"].size)
    fb = np.zeros(n, bool)
    fb[:5] = True
    for s in (0, 1):                                                       # the banked -tacorc evals carry the oracle fallback mask (as the real ones do)
        p = Path(f["eval"]) / f"eval_M2b_s{s}-tacorc.npz"
        z = dict(np.load(p))
        z["tac_oracle_fallback"] = fb
        np.savez(p, **z)
    w = type("W", (), {"n": n, "eid": np.asarray(gt["eid"]), "t": np.asarray(gt["frame"]) // 2})()
    wi = np.tile(np.arange(n // N_EP), N_EP)
    cu, tit = np.zeros(n, np.int8), ((w.eid < 11) & (wi == 3)).astype(np.int8)
    ok = _write_strata(tmp_path / "ok.npz", w, fb, cu, tit)
    out = tmp_path / "chk" / "inputs.json"
    base = ["--check-inputs", "--eval-dir", str(f["eval"]), "--runs-dir", str(f["runs"]), "--out", str(out), "--n-boot", "300", "--dk-n-boot", "60", "--no-families",
            "--n-windows", str(N_WIN), "--n-episodes", str(N_EP), "--lead-block", str(f["d"] / "lead.npz"), "--p10-pred", str(f["d"] / "p10.npy"), "--gt-npz", str(f["d"] / "gt.npz")]
    assert R.main([*base, "--strata", str(ok)]) == 0
    msg = capsys.readouterr().out
    assert "strata JUNCTION 10 windows/10 episodes, CURVE 0 windows/0 episodes, NO-TURN 62 windows/12 episodes" in msg
    res = json.loads(out.read_text(encoding="utf-8"))["strata"]
    assert res["counts"] == {"JUNCTION": 10, "CURVE": 0, "NO-TURN": 62} and res["n_without_v9_row"] == 5 and res["release_md5"] == R.RELEASE_MD5
    bad = _write_strata(tmp_path / "bad.npz", w, fb, cu, tit, has=(~fb) | (np.arange(n) == 0))        # window 0 has no label: the stratum is unchanged, the presence is not the evaluator's
    assert R.main([*base, "--strata", str(bad)]) == 2 and "T-tick row pairing is not the evaluator's" in capsys.readouterr().err
    assert R.main([*base, "--strata", str(tmp_path / "nope.npz")]) == 2 and "does not exist" in capsys.readouterr().err
