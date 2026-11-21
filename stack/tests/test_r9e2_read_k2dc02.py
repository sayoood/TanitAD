"""E2 read launcher for arm K2DC02 (stack/ops/r9e2_read/e2read_launch_k2dc02.sh) and the dose verdict's `--arm` option.  CPU only, no scipy, no GPU, no ssh.

  * the K2DC02 launcher differs from the K2DC one ONLY in the intended lines (compared after K2DC02 -> K2DC, so a stray K2DC left in an executable line shows up as a missing change);
  * `bash -n` passes;
  * the guards are RUN (their lines are extracted and executed in bash): a config with the 0.02 target and the 0.1,25 bounds passes, K2DC's own config (0.10 / 1,25) is refused,
    a missing K0 seed directory is refused;
  * DELIBERATE REGRESSION: a copy of the launcher with the 0.02 guard line removed lets K2DC's config through, so the refusal check goes RED on it;
  * the launcher's literals equal the tokens of stack/ops/runs.d/r9e2_K2DC02.argv.json (the single source), and the dose verdict READS target and bounds from that json;
  * the dose verdict's default path (no --arm, K2DC) is unchanged.
  * (e) the roll pin is r9e2_roll.sh 38161299 (1c48ee8f) and R9E2_CKPT_NAME other than model_ema.pt is refused (guard RUN, mutation checked).
"""
import difflib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "ops" / "r9e2_read"
NEW = OPS / "e2read_launch_k2dc02.sh"
OLD = OPS / "e2read_launch_k2dc.sh"
ARGV02 = ROOT / "ops" / "runs.d" / "r9e2_K2DC02.argv.json"
DOSE = ROOT / "scripts" / "r9_e2_dose_verdict.py"


def _bash():
    for c in ("C:/Program Files/Git/usr/bin/bash.exe", "C:/Program Files/Git/bin/bash.exe"):      # never System32's WSL bash
        if Path(c).exists():
            return c
    w = shutil.which("bash")
    if w and "system32" not in w.lower():
        return w
    pytest.skip("no usable bash")


def _text(p):
    return p.read_bytes().decode("utf-8")


def _norm(t):
    return t.replace("K2DC02", "K2DC").replace("k2dc02", "k2dc")


# r9e2_roll.sh at 1c48ee8f (an ancestor of the K2DC02 train commit e454e96c); the K2DC pin b4d141b9 is r9e2_roll.sh at 2efcf06f
ROLL_PIN_LINE = "WRAP_BLOB=fcf2e5c342309c3ce76d2782619d5145b11ff29a; ROLL_BLOB=3816129936e810cece67dfb620b3813a435a99cb"


def _ckpt_guard(text):
    keep = [ln for ln in text.splitlines() if ln.startswith('case "${R9E2_CKPT_NAME')]
    assert len(keep) == 1, keep
    return keep[0]


def test_roll_pin_and_ckpt_name_guard(tmp_path):
    """The roll pin is r9e2_roll.sh 38161299; its only change over b4d141b9 is R9E2_CKPT_NAME (default model_ema.pt). The guard admits only that default.
    MUTATE: without the guard line, ckpt.pt passes (RED for the mutant)."""
    t = _text(NEW)
    assert t.count(ROLL_PIN_LINE) == 1
    g = _ckpt_guard(t)

    def run(lines, val):
        env_line = "" if val is None else "export R9E2_CKPT_NAME=%s\n" % val
        p = tmp_path / "ck.sh"
        p.write_bytes(('set -u\ndie() { echo "ZZE2READ-REFUSED $*"; exit 3; }\n%s%s\necho GUARD-PASSED\n' % (env_line, "\n".join(lines))).encode("utf-8"))
        env = {k: v for k, v in os.environ.items() if k != "R9E2_CKPT_NAME"}
        r = subprocess.run([_bash(), str(p)], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
        return r.returncode, r.stdout

    assert run([g], None) == (0, "GUARD-PASSED\n")
    assert run([g], "model_ema.pt")[0] == 0
    rc, out = run([g], "ckpt.pt")
    assert rc == 3 and "R9E2_CKPT_NAME='ckpt.pt'" in out, out
    assert run([], "ckpt.pt")[0] == 0                         # the mutant (guard removed) lets the raw checkpoint through


def _toks():
    argv = json.loads(ARGV02.read_text(encoding="utf-8"))["argv"]
    return argv[argv.index("--map-hires-share-target") + 1], argv[argv.index("--map-hires-w-bounds") + 1]


# ---------------------------------------------------------------------------------------------------------------- #
def test_launcher_has_lf_endings_and_valid_syntax(tmp_path):
    assert b"\r" not in NEW.read_bytes()
    r = subprocess.run([_bash(), "-n", str(NEW)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stderr


def test_launcher_differs_from_k2dc_only_in_the_intended_lines():
    a, b = _norm(_text(OLD)).splitlines(), _norm(_text(NEW)).splitlines()
    removed, added = [], []
    for ln in difflib.unified_diff(a, b, lineterm="", n=0):
        if ln.startswith(("---", "+++", "@@")):
            continue
        (removed if ln.startswith("-") else added).append(ln[1:])
    # removed: exactly the K0 seed-source line's old OLD= and the old marker-check line; nothing else of the executable code
    assert [r for r in removed if not r.startswith("#")] == [
        'ARM=K2DC; OLD=$R/runs/E2_READ_K2M1N20; OUT=$R/runs/E2_READ_$ARM; K0=$R/runs/r9e2-K0; K2=$R/runs/r9e2-$ARM',
        "WRAP_BLOB=fcf2e5c342309c3ce76d2782619d5145b11ff29a; ROLL_BLOB=b4d141b9a540901fb44e3c3f644d11ff8e1535e5",
        next(r for r in removed if r.startswith("for m in")),
    ]
    assert all(r.startswith("for m in 'ROLL-DONE K0'") for r in removed if r.startswith("for m in"))
    code_added = [x for x in added if not x.startswith("#")]
    assert len(code_added) == 8, code_added
    assert code_added[0].startswith("ARM=K2DC; OLD=${K0_SEED_SRC:-$R/runs/E2_READ_K2DC}; OUT=")
    # (e) the roll re-pin (wrapper pin unchanged) and the one checkpoint-name guard
    assert code_added.count(ROLL_PIN_LINE) == 1
    assert sum(x.startswith('case "${R9E2_CKPT_NAME:-model_ema.pt}" in model_ema.pt) ;;') for x in code_added) == 1
    assert sum(x.startswith("CFG1=") for x in code_added) == 1
    assert sum(x.startswith('case "$CFG1"') for x in code_added) == 2
    assert sum(x.startswith("for d in $OLD ") for x in code_added) == 1
    assert sum(x.startswith("for m in 'ROLL-(DONE|SKIP) K0'") and "grep -qE" in x for x in code_added) == 1


def test_every_executable_k2dc_mention_is_k2dc02():
    for i, ln in enumerate(_text(NEW).splitlines(), 1):
        if ln.lstrip().startswith("#") or ln.startswith("ARM=K2DC02; OLD=${K0_SEED_SRC:-$R/runs/E2_READ_K2DC};"):      # the one intended K2DC: the default K0 seed source
            continue
        assert not re.search(r"K2DC(?!02)", ln), f"line {i} still names K2DC: {ln[:120]}"
    t = _text(NEW)
    assert r"arms \[K2DC02\]" in t and "' K2DC02-DONE '" in t and "ARM=K2DC02;" in t and "K0_SEED_SRC:-$R/runs/E2_READ_K2DC}" in t


def test_launcher_literals_equal_the_argv_json_tokens():
    target, bounds = _toks()
    assert (target, bounds) == ("0.02", "0.1,25")
    t = _text(NEW)
    assert f'"--map-hires-share-target","{target}"' in t and f'"--map-hires-w-bounds","{bounds}"' in t


# ---------------------------------------------------------------------------------------------------------------- #
# the guards are RUN                                                                                                #
# ---------------------------------------------------------------------------------------------------------------- #
def _guard_lines(text):
    keep = [ln for ln in text.splitlines() if ln.startswith(("CFG1=", 'case "$CFG1"', "for d in $OLD "))]
    assert len(keep) == 4, keep
    return keep


def _run_guards(text, tmp_path, cfg_tokens, make_seed=True):
    k2 = tmp_path / "k2"
    k2.mkdir(exist_ok=True)
    (k2 / "config.json").write_text(json.dumps({"argv": cfg_tokens}, indent=1), encoding="utf-8")
    old = tmp_path / "seed_src"
    if make_seed:
        for d in ("E2_K0_pf", "E2_K0_mapsel", "read/dump_K0"):
            (old / d).mkdir(parents=True, exist_ok=True)
    script = 'die() { echo "ZZE2READ-REFUSED $*"; exit 3; }\nK2=%s\nOLD=%s\n%s\necho GUARDS-PASSED\n' % (k2.as_posix(), old.as_posix(), "\n".join(_guard_lines(text)))
    p = tmp_path / "guards.sh"
    p.write_bytes(script.encode("utf-8"))
    r = subprocess.run([_bash(), str(p)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.returncode, r.stdout


K2DC02_CFG = ["--map-hires-share-target", "0.02", "--map-hires-share-every", "250", "--map-hires-w-bounds", "0.1,25"]
K2DC_CFG = ["--map-hires-share-target", "0.10", "--map-hires-share-every", "250", "--map-hires-w-bounds", "1,25"]


def test_guards_accept_k2dc02_and_refuse_k2dc_config_and_missing_seed(tmp_path):
    t = _text(NEW)
    (tmp_path / "a").mkdir()
    rc, out = _run_guards(t, tmp_path / "a", K2DC02_CFG)
    assert rc == 0 and "GUARDS-PASSED" in out, out
    (tmp_path / "b").mkdir()
    rc, out = _run_guards(t, tmp_path / "b", K2DC_CFG)
    assert rc == 3 and "0.02" in out, out
    (tmp_path / "c").mkdir()
    rc, out = _run_guards(t, tmp_path / "c", ["--map-hires-share-target", "0.02", "--map-hires-w-bounds", "1,25"])           # right target, K2DC's floor
    assert rc == 3 and "0.1,25" in out, out
    (tmp_path / "d").mkdir()
    rc, out = _run_guards(t, tmp_path / "d", K2DC02_CFG, make_seed=False)
    assert rc == 3 and "K0 seed source directory missing" in out, out
    (tmp_path / "e").mkdir()
    rc, out = _run_guards(t, tmp_path / "e", ["--map-hires-share-target", "0.020", "--map-hires-w-bounds", "0.1,25"])        # the token is compared as written in the argv
    assert rc == 3, out


def test_deliberate_regression_without_the_0p02_guard_is_caught(tmp_path):
    t = _text(NEW)
    lines = t.splitlines()
    mutant = "\n".join(ln for ln in lines if '"--map-hires-share-target","0.02"' not in ln)
    assert mutant != t and mutant.count('"--map-hires-share-target","0.02"') == 0
    with pytest.raises(AssertionError):                       # _guard_lines' own census goes red: the guard line is gone
        _guard_lines(mutant)
    # and the BEHAVIOUR: with only that guard line dropped from the executed set, K2DC's 0.10 config would pass the target check
    keep = [ln for ln in _guard_lines(t) if '"0.02"' not in ln]
    (tmp_path / "m").mkdir()
    k2 = tmp_path / "m" / "k2"
    k2.mkdir()
    (k2 / "config.json").write_text(json.dumps({"argv": ["--map-hires-share-target", "0.10", "--map-hires-w-bounds", "0.1,25"]}), encoding="utf-8")
    old = tmp_path / "m" / "seed"
    for d in ("E2_K0_pf", "E2_K0_mapsel", "read/dump_K0"):
        (old / d).mkdir(parents=True)
    p = tmp_path / "m" / "g.sh"
    p.write_bytes(('die() { echo "REFUSED $*"; exit 3; }\nK2=%s\nOLD=%s\n%s\necho GUARDS-PASSED\n' % (k2.as_posix(), old.as_posix(), "\n".join(keep))).encode("utf-8"))
    r = subprocess.run([_bash(), str(p)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0 and "GUARDS-PASSED" in r.stdout        # the mutant is blind to a 0.10 config; the pristine guard refuses it (previous test)


# ---------------------------------------------------------------------------------------------------------------- #
# the dose verdict                                                                                                  #
# ---------------------------------------------------------------------------------------------------------------- #
def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def D():
    sys.path.insert(0, str(ROOT / "scripts"))
    return _load(DOSE, "_dose_k2dc02")


def test_dose_default_path_is_k2dc_and_unchanged(D, tmp_path):
    assert D.ARM_DEFAULT == "K2DC" and D.arm_dose() is None and D.arm_dose("K2DC") is None
    assert D.SHARE_BAND == (0.05, 0.15) and D.BAR_M == 0.0025 and D.BAND_M == 0.0010
    nums = {s: {"dc_vs_k0": {"LAT_cross_mae_m": {"delta": 0.001, "lo": 0, "hi": 0.002, "separated": False}, "LAT_cross_abs_final_m": {"delta": 0.0, "lo": 0, "hi": 0, "separated": False}},
                "dc_vs_k2": {"LAT_cross_mae_m": {"delta": -0.004, "lo": -0.005, "hi": -0.003, "separated": True}, "LAT_cross_abs_final_m": {"delta": -0.004, "lo": -0.005, "hi": -0.003, "separated": True}},
                "jitter_dlag_dc_vs_k2": {"delta": -0.0005, "lo": -0.001, "hi": 0.0, "separated": False}} for s in ("0", "1")}
    doc = D.build(nums, {"valid": True, "status": "HELD"})
    assert "arm" not in doc and doc["thresholds"]["share_band"] == [0.05, 0.15] and doc["outcome"] == "SUPPORTED"
    assert sorted(doc) == sorted(["schema", "class", "prereg", "outcome", "outcome_if_valid", "reason", "per_seed", "dose_validity", "thresholds", "resolutions", "tier", "windows",
                                  "estimator", "families_note", "STRATEGIC", "seeds"])
    m = tmp_path / "m.jsonl"
    m.write_text("\n".join(json.dumps({"step": s, "dc_share_read": v, "dc_w_eff": 3.0}) for s, v in ((3000, 0.09), (3250, 0.11))) + "\n", encoding="utf-8")
    v = D.void_check(m)
    assert v["valid"] is True and v["band"] == [0.05, 0.15] and "w_bounds" not in v and v["rule"].endswith("inside [0.05, 0.15]")
    assert D.void_check(m, D.SHARE_BAND) == v
    with pytest.raises(SystemExit):
        D.arm_dose("K2DC", str(ARGV02))                          # an argv json belongs to a derived arm only
    with pytest.raises(SystemExit):
        D.arm_dose("K2DC03")


def test_dose_k2dc02_reads_target_and_bounds_from_its_argv_json(D, tmp_path):
    argv = json.loads(ARGV02.read_text(encoding="utf-8"))["argv"]
    want_t, want_b = float(argv[argv.index("--map-hires-share-target") + 1]), [float(x) for x in argv[argv.index("--map-hires-w-bounds") + 1].split(",")]
    a = D.arm_dose("K2DC02")
    assert a["arm"] == "K2DC02" and a["target"] == want_t == 0.02 and a["bounds"] == want_b == [0.1, 25.0] and a["share_band"] == [0.01, 0.03]
    # a different json moves the numbers: nothing is typed in the verdict script
    alt = tmp_path / "alt.argv.json"
    alt.write_text(json.dumps({"argv": ["--map-hires-share-target", "0.04", "--map-hires-w-bounds", "0.2,10"]}), encoding="utf-8")
    b = D.arm_dose("K2DC02", str(alt))
    assert b["target"] == 0.04 and b["bounds"] == [0.2, 10.0] and b["share_band"] == [0.02, 0.06]
    bad = tmp_path / "bad.argv.json"
    bad.write_text(json.dumps({"argv": ["--steps", "1"]}), encoding="utf-8")
    with pytest.raises(SystemExit):
        D.arm_dose("K2DC02", str(bad))
    src = DOSE.read_text(encoding="utf-8")
    body = src.split("def arm_dose")[1].split("def _row")[0]
    assert "toks.index(" in body and not re.search(r"target\s*=\s*0\.\d", body) and not re.search(r"\"0\.1,25\"", body)      # read from the json, not typed


def test_dose_k2dc02_validity_band_follows_its_target(D, tmp_path):
    a = D.arm_dose("K2DC02")
    m = tmp_path / "m.jsonl"
    rows = [(3000, 0.02, 0.1), (3250, 0.025, 25.0), (3500, 0.021, 7.0), (250, 0.5, 25.0)]
    m.write_text("\n".join(json.dumps({"step": s, "dc_share_read": v, "dc_w_eff": w}) for s, v, w in rows) + "\n", encoding="utf-8")
    v = D.void_check(m, tuple(a["share_band"]), tuple(a["bounds"]))
    assert v["valid"] is True and v["n_inside_band"] == 3 and v["band"] == [0.01, 0.03]
    assert v["n_late_w_eff_probes"] == 3 and v["n_late_at_lower_bound"] == 1 and v["n_late_at_upper_bound"] == 1
    k2dc_like = tmp_path / "k.jsonl"                         # K2DC's 0.10 share is OUTSIDE K2DC02's band: DOSE-VOID for this arm
    k2dc_like.write_text("\n".join(json.dumps({"step": s, "dc_share_read": 0.10}) for s in (3000, 3250)) + "\n", encoding="utf-8")
    assert D.void_check(k2dc_like, tuple(a["share_band"]))["valid"] is False and D.void_check(k2dc_like)["valid"] is True
    nums = {s: {"dc_vs_k0": {"LAT_cross_mae_m": {"delta": 0.001, "lo": 0, "hi": 0.002, "separated": False}, "LAT_cross_abs_final_m": {"delta": 0.0, "lo": 0, "hi": 0, "separated": False}},
                "dc_vs_k2": {"LAT_cross_mae_m": {"delta": -0.004, "lo": -0.005, "hi": -0.003, "separated": True}, "LAT_cross_abs_final_m": {"delta": -0.004, "lo": -0.005, "hi": -0.003, "separated": True}},
                "jitter_dlag_dc_vs_k2": {"delta": -0.0005, "lo": -0.001, "hi": 0.0, "separated": False}} for s in ("0", "1")}
    doc = D.build(nums, v, None, a)
    assert doc["arm"]["arm"] == "K2DC02" and doc["thresholds"]["share_band"] == [0.01, 0.03] and doc["outcome"] == "SUPPORTED" and "PREREG_K2DC02_2026-11-18" in doc["arm"]["note"] and "UNVERIFIED" not in doc["arm"]["note"]
    assert "doc.get('arm', {}).get('arm', ARM_DEFAULT)} dose hypothesis" in DOSE.read_text(encoding="utf-8")        # md() title follows the arm (md's table needs full family rows; the CLI path covers it)
