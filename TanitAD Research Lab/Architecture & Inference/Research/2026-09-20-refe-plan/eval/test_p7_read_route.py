#!/usr/bin/env python3
"""P7-READ-ROUTE (2026-11-12): gap 3 (PKG from __file__), B2 (the `p6` comparator pin), B3 (the `pick20` dump). CPU only, no GPU, no navtest scoring.
   python eval/test_p7_read_route.py  ->  ZZP7ROUTE PASS n/n
Every claim has a known-value control and a mutation arm that must go RED (a mutation is the OLD behaviour put back into a temp copy of the real source)."""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage3_full as S3  # noqa: E402

DATA = Path(os.environ.get("REFE_NAVTEST_DIR", "E:/Projects/TanitAD/data/refe_navtest"))
OLD_E_PKG = 'Path("E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan")'
res = []


def check(name):
    def d(fn):
        try:
            fn(); res.append((True, name)); print("PASS", name)
        except Exception as e:  # noqa: BLE001
            res.append((False, name)); print("FAIL", name, "<--", type(e).__name__, str(e)[:300])
        return fn
    return d


def must_fail(fn, *a, **k):
    try:
        fn(*a, **k)
    except (AssertionError, SystemExit, RuntimeError, ValueError, SyntaxError):
        return
    raise AssertionError("the mutation was NOT detected (the check stayed green)")


def _clean_env():
    os.environ.pop("TANITAD_REFE_PKG", None)


# ------------------------------------------------------------------------------------------------ 1. PKG from __file__
def _export(td: Path, mutate=None) -> Path:
    """a minimal tip export in a temp dir: <td>/pk/eval/{stage2_sft,stage3_full,eval_checkpoint,harness_pins,proxy_eval}.py + harness/score_e.py; `mutate(name, text)` edits a copy"""
    pk = td / "pk"
    (pk / "eval" / "harness").mkdir(parents=True)
    for n in ("stage2_sft", "stage3_full", "eval_checkpoint", "harness_pins", "proxy_eval"):
        t = (HERE / f"{n}.py").read_text(encoding="utf-8")
        if mutate:
            t = mutate(n, t)
        (pk / "eval" / f"{n}.py").write_text(t, encoding="utf-8", newline="\n")
    t = (HERE / "harness" / "score_e.py").read_text(encoding="utf-8")
    if mutate:
        t = mutate("score_e", t)
    (pk / "eval" / "harness" / "score_e.py").write_text(t, encoding="utf-8", newline="\n")
    return pk


def _load(path: Path, name: str):
    old = list(sys.path)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m
    finally:
        sys.path[:] = old


def _score_e_pkg(path: Path) -> str:
    """score_e.py runs main() at import; execute only its prologue (up to the `spec =` line) and read PKG"""
    src = path.read_text(encoding="utf-8")
    head = src[: src.index("spec = importlib")]
    g = {"__file__": str(path), "__name__": "score_e_prologue"}
    exec(compile(head, str(path), "exec"), g)  # noqa: S102
    return g["PKG"]


def resolves_own_pkg(pk: Path) -> None:
    """every route module of the export at `pk` resolves ITS OWN package; raises AssertionError otherwise"""
    _clean_env()
    s2 = _load(pk / "eval" / "stage2_sft.py", f"s2_{abs(hash(str(pk)))}")
    assert Path(s2.PKG).resolve() == pk.resolve(), (s2.PKG, pk)
    assert Path(s2.HARNESS).resolve() == (pk / "eval" / "harness").resolve()
    s3 = _load(pk / "eval" / "stage3_full.py", f"s3_{abs(hash(str(pk)))}")
    assert Path(s3.PKG).resolve() == pk.resolve(), s3.PKG
    assert Path(s3.S2_RAW).resolve() == (pk / "raw" / "2026-10-05-sft-stage2").resolve(), s3.S2_RAW     # stage 3 reads where stage 2 (this export) wrote
    assert Path(_score_e_pkg(pk / "eval" / "harness" / "score_e.py")).resolve() == pk.resolve()


@check("P1 a tip export in a temp dir resolves its OWN PKG in stage2_sft, stage3_full (PKG, S2_RAW) and harness/score_e.py")
def t_own_pkg():
    with tempfile.TemporaryDirectory() as td:
        resolves_own_pkg(_export(Path(td)))


@check("P1 RED: stage2_sft.py with the old hard-coded E: PKG put back is detected")
def t_red_stage2():
    with tempfile.TemporaryDirectory() as td:
        def mut(n, t):
            if n != "stage2_sft":
                return t
            t2 = re.sub(r'PKG = Path\(os\.environ\.get\("TANITAD_REFE_PKG"\) or Path\(__file__\)\.resolve\(\)\.parent\.parent\)', "PKG = " + OLD_E_PKG, t)
            assert t2 != t, "the mutation did not apply"
            return t2
        must_fail(resolves_own_pkg, _export(Path(td), mut))


@check("P1 RED: stage3_full.py's S2_RAW pointing back at the E: tree is detected")
def t_red_stage3():
    with tempfile.TemporaryDirectory() as td:
        def mut(n, t):
            if n != "stage3_full":
                return t
            t2 = t.replace('S2_RAW = PKG / "raw" / "2026-10-05-sft-stage2"', 'S2_RAW = Path("E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-05-sft-stage2")')
            assert t2 != t
            return t2
        must_fail(resolves_own_pkg, _export(Path(td), mut))


@check("P1 RED: harness/score_e.py with the hard-coded E: PKG put back is detected")
def t_red_score_e():
    with tempfile.TemporaryDirectory() as td:
        def mut(n, t):
            if n != "score_e":
                return t
            t2 = re.sub(r'PKG = \(os\.environ\.get.*\n', 'PKG = "E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"\n', t)
            assert t2 != t
            return t2
        must_fail(resolves_own_pkg, _export(Path(td), mut))


@check("P1 $TANITAD_REFE_PKG overrides (the one escape hatch), in all three modules")
def t_env_override():
    with tempfile.TemporaryDirectory() as td:
        pk = _export(Path(td))
        other = Path(td) / "other"
        other.mkdir()
        os.environ["TANITAD_REFE_PKG"] = str(other)
        try:
            s2 = _load(pk / "eval" / "stage2_sft.py", "s2_env")
            s3 = _load(pk / "eval" / "stage3_full.py", "s3_env")
            assert Path(s2.PKG) == other and Path(s3.PKG) == other and Path(_score_e_pkg(pk / "eval" / "harness" / "score_e.py")) == other
        finally:
            _clean_env()


@check("P1 no route module of the REAL tree still hard-codes the E: package path (grep over the eval route files)")
def t_no_hardcode():
    bad = []
    for rel in ("stage2_sft.py", "stage3_full.py", "harness/score_e.py", "command_adherence.py", "score_navtest_refe.py", "proxy_eval.py", "instruments/transp_score.py", "instruments/families_run.py"):
        t = (HERE / rel).read_text(encoding="utf-8")
        if "E:/Projects/TanitAD/TanitAD Research Lab" in re.sub(r"#.*", "", t):
            bad.append(rel)
    assert not bad, bad


@check("P1 the md5 pin of stage2_sft.py in stage3_full equals the file (the pin and the file move as a pair)")
def t_md5_pin():
    assert S3.md5_file(S3.STAGE2_SFT) == S3.STAGE2_SFT_MD5, (S3.md5_file(S3.STAGE2_SFT), S3.STAGE2_SFT_MD5)


# ------------------------------------------------------------------------------------------------ 2. B2: the p6 comparator
@check("B2 `p6` is a BASELINES entry and `--base p6` / `--also p6` are argparse choices")
def t_p6_choice():
    assert "p6" in S3.BASELINES
    r = subprocess.run([sys.executable, str(HERE / "stage3_full.py"), "--help"], capture_output=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0 and re.search(r"--base \{[^}]*\bp6\b[^}]*\}", r.stdout), r.stdout[:400]


@check("B2 pins hold on the REAL P6 files (E: data dir + F: families), the PDMS banked value is the mean of the csv, and the pins are the md5 of those files")
def t_p6_pins_real():
    b = S3.pick_baseline("p6")
    assert b["pin_ok"], (b["csv"], b["seam"], b.get("csv_missing"), b.get("seam_missing"))
    assert b["families"] is not None, b.get("families_missing")
    import proxy_eval as PE
    rows = PE.read_csv(b["csv"])
    mean = float(np.mean([100.0 * float(r["score"]) for r in rows.values()]))
    assert round(mean, 4) == S3.BASELINES["p6"]["pdms_x100"] == 91.1965, mean


@check("B2 level control: the P6 csv's registered level CI equals F:/TanitAD_p6read_work/stage3_p6_full/level_ci.json (91.1965 [90.4102, 91.9835]) to 1e-9")
def t_p6_level():
    import selection_quality as SQ
    import gzip
    import json
    exp = json.load(gzip.open(S3.EXPORT, "rt", encoding="utf-8"))["tokens"]
    tl = {t: v["log_name"] for t, v in exp.items()}
    r = SQ.level_from_csv(S3.pick_baseline("p6")["csv"], tl)["level_x100"]
    want = (91.19647788427798, 90.41022246390116, 91.98351862523724)         # MEASURED, F:/TanitAD_p6read_work/stage3_p6_full/level_ci.json "PDMS"
    got = (r["mean"], r["ci95"][0], r["ci95"][1])
    assert all(abs(g - w) < 1e-9 for g, w in zip(got, want)), (got, want)


@check("B2 the banked copies alone (no E: data dir, no F: file) satisfy the p6 pins: the pin survives the drives being absent")
def t_p6_bank_only():
    orig = copy.deepcopy(S3.BASELINES["p6"])
    try:
        for k in ("csv", "seam", "report", "families"):
            S3.BASELINES["p6"][k] = [p for p in orig[k] if str(S3.BANK) in str(p)]
            assert S3.BASELINES["p6"][k], k
        b = S3.pick_baseline("p6")
        assert b["pin_ok"] and b["families"] is not None, (b.get("csv_missing"), b.get("seam_missing"), b.get("families_missing"))
        assert str(S3.BANK) in str(b["csv"]) and str(S3.BANK) in str(b["seam"]) and str(S3.BANK) in str(b["families"])
    finally:
        S3.BASELINES["p6"] = orig


@check("B2 EVERY comparator (u400, p2, p6, deployed) resolves csv + seam + families from the IN-REPO tree alone (no E: data dir, no F: file): a tip export on F: can run any --base/--also")
def t_all_in_repo_only():
    orig = copy.deepcopy(S3.BASELINES)
    try:
        for key, ent in S3.BASELINES.items():
            for k in ("csv", "seam", "report", "families"):
                ent[k] = [p for p in orig[key][k] if str(S3.PKG) in str(p)]
                assert ent[k], (key, k)
            b = S3.pick_baseline(key)
            assert b["pin_ok"] and b["families"] is not None, (key, b.get("csv_missing"), b.get("seam_missing"), b.get("families_missing"))
    finally:
        S3.BASELINES.clear()
        S3.BASELINES.update(orig)


@check("B2 RED: a wrong csv md5, a wrong seam md5 and a wrong families md5 each make the pin fail (a pin outside the tuple is refused)")
def t_p6_red():
    orig = copy.deepcopy(S3.BASELINES["p6"])
    try:
        S3.BASELINES["p6"]["csv_md5"] = (S3.BASELINES["u400"]["csv_md5"],)
        assert S3.pick_baseline("p6")["pin_ok"] is False
        S3.BASELINES["p6"]["csv_md5"] = orig["csv_md5"]
        S3.BASELINES["p6"]["seam_md5"] = "0" * 32
        assert S3.pick_baseline("p6")["pin_ok"] is False
        S3.BASELINES["p6"]["seam_md5"] = orig["seam_md5"]
        S3.BASELINES["p6"]["families_md5"] = "0" * 32
        b = S3.pick_baseline("p6")
        assert b["families"] is None and b.get("families_missing"), "a wrong families md5 must leave the families unresolved"
    finally:
        S3.BASELINES["p6"] = orig
    assert S3.pick_baseline("p6")["pin_ok"]


@check("B2 the p6 entry cannot be mistaken for another arm: its csv/seam md5 differ from u400's, p2's and deployed's")
def t_p6_distinct():
    p = S3.BASELINES["p6"]
    for k in ("u400", "p2", "deployed"):
        o = S3.BASELINES[k]
        assert not (S3._md5_set(p["csv_md5"]) & S3._md5_set(o["csv_md5"])) and not (S3._md5_set(p["seam_md5"]) & S3._md5_set(o["seam_md5"])), k


# ------------------------------------------------------------------------------------------------ 3. B3: pick20
SEAM_PY = HERE / "refe_navtest_seam.py"


def _seam_mod():
    spec = importlib.util.spec_from_file_location("refe_navtest_seam_route", SEAM_PY)
    m = importlib.util.module_from_spec(spec)
    old = list(sys.path)
    try:
        spec.loader.exec_module(m)
    finally:
        sys.path[:] = old
    return m


def _synthetic(n=7, seed=3):
    rng = np.random.default_rng(seed)
    g = np.cumsum(rng.normal(1.0, 0.3, (n, 20, 1)), axis=1)
    p = np.concatenate([g, np.cumsum(rng.normal(0, 0.1, (n, 20, 1)), axis=1), rng.normal(0, 0.2, (n, 20, 1))], axis=2).astype(np.float32)
    return p


@check("B3 control: seam row == float32(to_navsim(pick20)) bit for bit on synthetic plans; pick20_equals_seam says True")
def t_pick20_control():
    m = _seam_mod()
    p20 = _synthetic()
    seam = np.stack([m.to_navsim(x).astype(np.float32) for x in p20])            # exactly how the seam builds its row (to_navsim of the float32 executed plan)
    assert m.pick20_equals_seam(p20, seam)


@check("B3 RED: one ulp in one pose, a swapped row, a wrong shape, a length mismatch each make the control False")
def t_pick20_red():
    m = _seam_mod()
    p20 = _synthetic()
    seam = np.stack([m.to_navsim(x).astype(np.float32) for x in p20])
    bad = p20.copy()
    bad[3, 7, 0] = np.nextafter(bad[3, 7, 0], np.float32(1e9))
    assert not m.pick20_equals_seam(bad, seam)
    assert not m.pick20_equals_seam(p20[::-1].copy(), seam)
    assert not m.pick20_equals_seam(p20[:, :19], seam)
    assert not m.pick20_equals_seam(p20[:-1], seam)
    assert not m.pick20_equals_seam(p20, seam[::-1].copy())


def _savez_keys(src: str):
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "savez" and any(k.arg == "proposals" for k in node.keywords):
            return [k.arg for k in node.keywords]
    raise AssertionError("no proposals savez found")


ORIG_KEYS = ["token", "fingerprint", "proposals", "logits", "pick", "select", "rule", "repair_last_heading", "ckpt", "sampling"]      # the tip's dump (refe_navtest_seam.py:354-357)


def pick20_wired(src: str) -> None:
    keys = _savez_keys(src)
    assert "pick20" in keys, keys
    assert [k for k in keys if k != "pick20"] == ORIG_KEYS, keys                  # every existing key kept, same names, same order
    assert "dump_pick20.append(ex20.astype(np.float32))" in src
    assert "poses = to_navsim(ex20)" in src                                        # the executed plan the seam row uses IS what pick20 stores
    assert "pick20_equals_seam(pick20," in src and "assert pick20_equals_seam" in src      # the dump fails loud if the control fails


@check("B3 wiring (AST): the dump's np.savez keeps all 10 existing keys in order and adds pick20; pick20 is the executed plan that feeds `poses`; the control is asserted at dump time")
def t_pick20_wired():
    pick20_wired(SEAM_PY.read_text(encoding="utf-8"))


@check("B3 RED: dropping the `pick20=` key, or feeding it something other than the executed plan, or dropping the assert, is detected")
def t_pick20_wiring_red():
    s = SEAM_PY.read_text(encoding="utf-8")
    for old, new in (("pick20=pick20, ", ""), ("dump_pick20.append(ex20.astype(np.float32))", "dump_pick20.append(np.zeros((20, 3), np.float32))"),
                     ("assert pick20_equals_seam(", "assert True or pick20_equals_seam(")):
        assert old in s, old
        must_fail(pick20_wired, s.replace(old, new, 1))


@check("B3 default output unchanged: with the dump OFF nothing of pick20 runs (the append and the assert sit inside `if a.dump_proposals`)")
def t_pick20_default_off():
    s = SEAM_PY.read_text(encoding="utf-8")
    i = s.index("dump_pick20.append(")
    ctx = s[max(0, i - 700): i]
    assert "if a.dump_proposals:" in ctx, ctx[-300:]
    j = s.index("assert pick20_equals_seam(")
    assert "if a.dump_proposals:" in s[max(0, j - 700): j]
    assert s.count("np.savez(a.out,") == 1 and "pick20" not in s[s.index("np.savez(a.out,"): s.index("np.savez(a.out,") + 300]        # the SEAM npz itself is untouched


if __name__ == "__main__":
    n_ok = sum(1 for ok, _ in res if ok)
    print(f"ZZP7ROUTE {'PASS' if n_ok == len(res) else 'FAIL'} {n_ok}/{len(res)}")
    sys.exit(0 if n_ok == len(res) else 1)
