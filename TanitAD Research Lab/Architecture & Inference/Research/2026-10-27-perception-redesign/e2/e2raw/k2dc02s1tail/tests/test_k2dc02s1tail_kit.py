"""Tests of the K2DC02s1 tail-read kit (PREREG_K2DC02S1_2026-11-20.md sec. 2 to 5). Kept in the kit, not in stack/tests.

Run (one pytest process at a time, >= 5.5 GB commit free; PYTHONPATH not needed):
    C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q <this file>
What is pinned:
  * every derived script differs from its k2dc02tail source only in an enumerated set of lines (compared after K2DC02s1 -> K2DC02); the source is read from git, and the test SKIPS if the mirror is absent;
  * the kept guards (tail tool blob, identity control, deliberate-regression arm, seed guard, roll/env blobs, GPU lock) are still in the text; no comparator is re-rolled; the reproduction roll is K0t at seed 0 only;
  * the pin check (Thor, read only) passes on a synthetic tree and REFUSES on every mutant (changed byte, missing file, extra file, edited manifest, wrong line count, foreign prefix, diverging K0t copies);
  * the combined reading (sec. 4 + 5) on synthetic inputs, with mutants that must go RED (a SUPPORTED that does not clear the floor reads UNRESOLVED; the floor's own label never makes a reading);
  * the reproduction check band, the dose-tool arm guard (which, at the PREREG's tip 48c797f3, REFUSES: that tool does not know the K2DC02s1 arm), bash -n of every .sh, LF endings and repo path length."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

KIT = Path(__file__).resolve().parents[1]
SCRIPTS, DEVBOX = KIT / "scripts", KIT / "devbox"
GD = "C:/Users/Admin/tanitad-push/.git"
SRC_COMMIT = "48c797f3"
SRC_DIR = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-27-perception-redesign/e2/e2raw/k2dc02tail"
PIN_SHA = "3d367ad922e4b250c1782b6204dad7b56ce9c0d83eae5955fe1959c5512f311c"


def _bash() -> str:
    for c in ("C:/Program Files/Git/usr/bin/bash.exe", "C:/Program Files/Git/bin/bash.exe"):      # never System32's WSL bash
        if Path(c).exists():
            return c
    b = shutil.which("bash")
    if not b:
        pytest.skip("no bash")
    return b


def _run(script, *args, env=None, cwd=None):
    import os
    e = dict(os.environ)
    e.update(env or {})
    return subprocess.run([_bash(), str(script), *args], capture_output=True, text=True, encoding="utf-8", errors="replace", env=e, cwd=cwd, timeout=300)


def _src(rel: str) -> str:
    r = subprocess.run(["git", "--git-dir", GD, "show", f"{SRC_COMMIT}:{SRC_DIR}/{rel}"], capture_output=True)
    if r.returncode != 0:
        pytest.skip(f"cannot read {rel} at {SRC_COMMIT} from the mirror")
    return r.stdout.decode("utf-8")


def _norm(t: str) -> str:
    return t.replace("K2DC02S1TAIL", "K2DC02TAIL").replace("k2dc02s1tail", "k2dc02tail").replace("K2DC02s1", "K2DC02")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


combined = _load(DEVBOX / "k2dc02s1tail_combined.py", "k2s_combined")
repro = _load(DEVBOX / "k2dc02s1tail_repro_check.py", "k2s_repro")
ALL_SH = sorted(list(SCRIPTS.glob("*.sh")) + list(DEVBOX.glob("*.sh")))
ALL_FILES = sorted(p for p in KIT.rglob("*") if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts)


# ---------------------------------------------------------------------------------------------------------------- #
# hygiene
# ---------------------------------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("p", ALL_SH, ids=lambda p: p.name)
def test_bash_syntax(p):
    r = subprocess.run([_bash(), "-n", str(p)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_lf_only_and_repo_path_length():
    for p in ALL_FILES:
        assert b"\r" not in p.read_bytes(), f"CR in {p.name}"
        s = p.as_posix()
        rel = s[s.index("TanitAD Research Lab"):]
        assert len(rel) <= 200, f"{len(rel)} chars: {rel}"


def test_every_script_name_is_k2dc02s1tail():
    for p in ALL_FILES:
        assert p.name.startswith("k2dc02s1tail_") or p.name.startswith("test_k2dc02s1tail"), p.name


def test_python_modules_run_nothing_on_import():
    src = (DEVBOX / "k2dc02s1tail_fam_table.py").read_text(encoding="utf-8") + (DEVBOX / "k2dc02s1tail_sanitize_navcomp.py").read_text(encoding="utf-8")
    assert src.count('if __name__ == "__main__":') == 2
    for n in ("k2dc02s1tail_combined.py", "k2dc02s1tail_repro_check.py", "k2dc02s1tail_strategic_navcomp.py"):
        assert 'if __name__ == "__main__":' in (DEVBOX / n).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------------------------------------------- #
# derived scripts differ from their sources only in listed ways (after K2DC02s1 -> K2DC02)
# ---------------------------------------------------------------------------------------------------------------- #
ALLOWED = {
    ("scripts/k2dc02tail_build.sh", "scripts/k2dc02s1tail_build.sh"): [
        "# k2dc02tail_build.sh -- Thor, CPU only.",                                  # header (arm / PREREG name)
        "R=/home/nvidia/r8ns/r9e2; RT=$R/read_tree_e454e96c6a84_k2dc; PY=",        # the tree is read_tree_<train commit>_k2dc
        "BANKED_PROV=$W/inputs/K2DC.model_ema_tail5k6k.pt.provenance.json",         # read from the K2DC02 tail's inputs/ (W0)
        "mkdir -p $W $W/inputs; ev()",                                              # DRY-aware ev, no mkdir in DRY
        'ev START "tool $TOOL tree $RT out $OUTD"',                                 # + dry flag, moved after the guards
    ],
    ("scripts/k2dc02tail_chain.sh", "scripts/k2dc02s1tail_chain.sh"): [
        "m() { echo", "bash $S/k2dc02tail_build.sh <", "bash $S/k2dc02tail_stageA.sh <", "m A-DONE",
    ],
    ("scripts/k2dc02tail_chain_all.sh", "scripts/k2dc02s1tail_chain_all.sh"): [
        "m() { echo", "bash $S/k2dc02tail_cmp_reroll.sh", "grep -q 'ZZK2DC02TAILCMP-DONEZZ'", 'm "ALLSTEP CMP-DONE"',    # the comparator re-roll is replaced by the pin check + one-seed roll
    ],
    ("scripts/k2dc02tail_stageA.sh", "scripts/k2dc02s1tail_stageA.sh"): [
        "# k2dc02tail_stageA.sh -- K2DC02-TAIL STAGE A", 'ev START "K2DC02-TAIL stage A',                          # header / start line (arm, PREREG name)
        "R=/home/nvidia/r8ns/r9e2; RT=$R/read_tree_e454e96c6a84_k2dc; F1=", "mkdir -p $W; ST=$W/STATUS.txt", "ev() { echo",
        "$PY -I -c 'import json,sys", "json.dump(m, open(sys.argv[2]",                  # the fit-remap python moved into REMAPPY (same text), so DRY can compare it on a pipe
    ],
    ("scripts/k2dc02tail_stageB_launch.sh", "scripts/k2dc02s1tail_stageB_launch.sh"): [
        "#   e2/e2raw/PREREG_K2DC02TAIL", "#   READ TREE =", "#     K0_CMP (default", "#     $R/runs/E2TAIL_K2DC/stageB/E2_K0_pf was deleted", "#     Produce it with",       # header: what the K2DC02 tail assumed
        "R=/home/nvidia/r8ns/r9e2; W=$R/runs/E2TAIL_K2DC02; A=", "CMP=${K0_CMP:-$W/cmp}",                    # pinned CMP / K0SEED, no override
        "TREE=$R/read_tree_e454e96c6a84_k2dc; ENVF", "for f in r9e2_dev_filter.json",
        "cp -al $CMP/E2_K0_pf", 'echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) SEEDED K0 slot',                     # a COPY re-hashed against the pin, never a hard link
    ],
    ("devbox/k2dc02tail_common.sh", "devbox/k2dc02s1tail_common.sh"): [
        "# PREREG_K2DC02TAIL_2026-11-20.md + AMENDMENT1", "#   K2T_TIP", "#   K2T_W ", "TIP=${K2T_TIP:-387616b3", 'E2TREAD="$E2P/e2/e2raw/e2tail/stageB/read"', "  local paths=(",
    ],
    ("devbox/k2dc02tail_s6c_pull_score.sh", "devbox/k2dc02s1tail_s6c_pull_score.sh"): [
        "# k2dc02tail_s6c_pull_score.sh [DRY]", "#   1 pulls", "[ \"$(th \"grep -c ' DONE ' $W/CMP_STATUS.txt\")\"", "for p in $W/cmp/E2_K0_pf", "NEED=$(th",
        'mkdir -p "$MS" "$P/cmp"', "pull cmp_k0", "pull cmp_k2dc",
    ],
    ("devbox/e2_strategic_navcomp.py", "devbox/k2dc02s1tail_strategic_navcomp.py"): [],
}


@pytest.mark.parametrize("pair", list(ALLOWED), ids=lambda p: p[1].split("/")[-1])
def test_derived_differs_only_in_listed_ways(pair):
    s_rel, d_rel = pair
    src = _src(s_rel).split("\n")
    der = _norm((KIT / d_rel).read_text(encoding="utf-8")).split("\n")
    dset = set(der)
    removed = [l for l in src if l not in dset]
    prefixes = ALLOWED[pair]
    for l in removed:
        assert any(l.startswith(p) for p in prefixes), f"{d_rel}: a source line is gone and is not on the list: {l[:140]!r}"
    for p in prefixes:
        assert any(l.startswith(p) for l in removed), f"{d_rel}: stale allowance (no source line starts with it): {p!r}"
    kept = len(src) - len(removed)
    assert kept >= 0.5 * len(src), f"{d_rel}: only {kept} of {len(src)} source lines survive"


def test_strategic_navcomp_is_the_source_verbatim():
    assert _norm((DEVBOX / "k2dc02s1tail_strategic_navcomp.py").read_text(encoding="utf-8")) == _src("devbox/e2_strategic_navcomp.py")


def test_fam_table_and_sanitize_behave_like_their_sources(tmp_path):
    ev = {"four_families": {"0": {"LATERAL": {"status": "PASS", "rows": {"LAT_cross_mae_m": {"delta": 0.0123, "lo": 0.001, "hi": 0.02, "better": "lower", "separated_worse": True, "n": 100, "n_episodes": 70}}},
                                   "TACTICAL": {"status": "n/a"}}}}
    (tmp_path / "v.json").write_text(json.dumps(ev), encoding="utf-8")
    (tmp_path / "src_fam.py").write_text(_src("devbox/fam_table.py"), encoding="utf-8")
    a = subprocess.run([sys.executable, "-I", str(tmp_path / "src_fam.py"), str(tmp_path / "v.json")], capture_output=True, text=True).stdout
    b = subprocess.run([sys.executable, "-I", str(DEVBOX / "k2dc02s1tail_fam_table.py"), str(tmp_path / "v.json")], capture_output=True, text=True).stdout
    assert a and a.replace("K2DC02", "K2DC02s1") == b
    (tmp_path / "src_san.py").write_text(_src("devbox/sanitize_navcomp.py"), encoding="utf-8")
    u = "-".join(["12345678", "1234", "1234", "1234", "123456789abc"])      # built at run time: the lander refuses id-shaped strings in landed files
    for tag, script in (("a", tmp_path / "src_san.py"), ("b", DEVBOX / "k2dc02s1tail_sanitize_navcomp.py")):
        d = tmp_path / f"in_{tag}"
        d.mkdir()
        (d / "navcomp_K0t_s0.json").write_text(json.dumps({"time_base_control": {"per_clip": [{"clip": u, "x": 1}]}}), encoding="utf-8")
        (d / "STRATEGIC_NAVCOMP.md").write_text("ok", encoding="utf-8")
        r = subprocess.run([sys.executable, "-I", str(script), str(d), str(tmp_path / f"out_{tag}")], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
    ja, jb = (json.loads((tmp_path / f"out_{t}" / "navcomp_K0t_s0.json").read_text(encoding="utf-8")) for t in "ab")
    assert ja == jb and ja["time_base_control"]["per_clip"][0]["clip"] == "clip#0"
    assert u not in (tmp_path / "out_b" / "navcomp_K0t_s0.json").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------------------------------------------- #
# kept guards, no comparator re-roll, one-seed reproduction roll, DRY structure
# ---------------------------------------------------------------------------------------------------------------- #
def _t(name):
    return (KIT / name).read_text(encoding="utf-8")


def _code(name):
    """the script without its comment-only lines (header comments legitimately name what was removed)"""
    return "\n".join(l for l in _t(name).split("\n") if not l.lstrip().startswith("#"))


def test_tail_build_keeps_its_guards():
    b = _t("scripts/k2dc02s1tail_build.sh")
    for must in ("TOOL_BLOB=e456ff67be132e9090f688dde4b8e9043ea43d5d", "--ema-a $K2DC02s1/ckpt_ema_5000.pt --ema-b $K2DC02s1/model_ema.pt", "IDCTL_TENSOR_EQUAL", "torch.equal", "IDCTL-PASS-TENSOR",
                 "0.9998", '"--seed","1"', "--r9-hist-supervise-geom", " K2DC02s1-DONE ", "W_tail = (EMA_6000 - 0.81872 * EMA_5000) / 0.18128", "never overwritten"):
        assert must in b, must
    assert b.index("die NO-RUN") < b.index("TC=${K2S_TRAIN_COMMIT") < b.index("TOOL=$RT/stack/scripts/ema_tail_weights.py")           # the missing-run guard is the FIRST guard
    assert b.index("[ $DRY = 1 ]") < b.index("rm -rf \"${SC:?}\"")                            # dry exits before the scratch rebuild
    assert "runs/r9e2-K2DC02s1" in b and "$W0/inputs/K2DC.model_ema_tail5k6k.pt.provenance.json" in b
    ca = _t("scripts/k2dc02s1tail_chain_all.sh")
    assert "IDCTL_TENSOR_EQUAL=1 BANKED_PROV_SHA_PIN=9065c97666dfad6d9f259a7124f96cd50aa6dfe88ea0fd04356f016f09f72f8a" in ca


def test_stage_a_keeps_the_regression_arm_and_tail_guard():
    a = _t("scripts/k2dc02s1tail_stageA.sh")
    for must in ("REGRESSION-NOT-RED", "REGRESSION-CONTROL", "REGRESSION-RED", "DELIBERATE REGRESSION", 'int(d.get("step",-1))==6000', 'd.get("ckpt_md5")==h', "$RAWRUN/model_ema.pt $RUN/$CK $RG",
                 "LOCK=/home/nvidia/refcv7_post/thor_gpu.lock", "flock -w 43200", "GPU-BUSY", "RAWRUN=$R/runs/r9e2-K2DC02s1"):
        assert must in a, must
    assert a.index("die NO-RUN") < a.index("for f in \"$FT\"")                                # first guard before any input
    assert a.index("ev DRY-OK") < a.index("gpu_dump $A $FD")                                  # dry exits before the first GPU stage call
    assert '"$RB" = 1' in a and '"$RG2" = 0' in a


def test_stage_b_uses_the_pinned_k0t_and_cannot_be_overridden():
    b = _code("scripts/k2dc02s1tail_stageB_launch.sh")
    assert "K0_CMP" not in b and "K0_SEED:-" not in b and "K0_CMP:-" not in b
    assert "CMP=$W0/cmp" in b and "PINMAN=$W0/K2DC02S1_CMP_PIN.sha256" in b
    assert "k2dc02s1tail_pin_check.sh" in b and b.index("k2dc02s1tail_pin_check.sh") < b.index("cp -a $CMP/E2_K0_pf")
    assert "cp -al $CMP" not in b and "cp -a $CMP/E2_K0_pf $B/E2_K0_pf && grep ' cmp/E2_K0_pf/' $PINMAN" in b      # a copy, re-hashed against the pin
    assert "ROLL_BLOB=3816129936e810cece67dfb620b3813a435a99cb" in b and "d4bdcf07cf337758ee0770f72091a23e6bf4eb34" in b
    assert b.index("guard 1") < b.index("TREE/stack")


def test_no_comparator_is_re_rolled_and_the_repro_roll_is_k0t_seed0_only():
    allsrc = "\n".join(_code(p.relative_to(KIT).as_posix()) for p in ALL_FILES if p.suffix == ".sh")
    assert "cmp_reroll" not in allsrc and "cmp_k2dc/E2_K2_pf --infer" not in allsrc
    r = _t("scripts/k2dc02s1tail_repro_roll.sh")
    assert r.count("--tag E2_K0_pf") == 1 and "E2_K2_pf" not in r.replace("E2_K2_pf ", "")        # only K0t is rolled
    assert "--infer-seeds 0 " in r and "--infer-seeds 0,1" not in r
    for must in ("--no-g0 --no-baselines", "--lock-held", "OMP_NUM_THREADS=6", "flock -w 43200", "GPU-NOT-EMPTY", "thor_gpu.lock", "ROLL_BLOB=3816129936e810cece67dfb620b3813a435a99cb",
                 "d4bdcf07cf337758ee0770f72091a23e6bf4eb34", 'OUTROOT=$W/repro', 'source $ENVF K2DC02s1', "never overwritten"):
        assert must in r, must
    assert r.index("die NO-RUN") < r.index("[ ${#TC} = 12 ]") < r.index("ROLL-BLOB")
    assert r.index("ev DRY-OK") < r.index("flock -w 43200") and r.index("ev DRY-OK") < r.index("mkdir -p $W\n")           # no lock, no write in DRY
    assert "runs/E2TAIL_K2DC02/" not in _code("scripts/k2dc02s1tail_repro_roll.sh").replace("runs/E2TAIL_K2DC02s1", "")                  # never names the pinned root as a target


def test_every_thor_script_has_a_dry_mode_and_writes_nothing_in_it():
    for n in ("build", "stageA", "repro_roll", "stageB_launch", "chain", "chain_all"):
        t = _t(f"scripts/k2dc02s1tail_{n}.sh")
        assert "K2S_DRY" in t or "LAUNCH_DRY" in t, n
    for n in ("build", "stageA", "repro_roll"):
        t = _t(f"scripts/k2dc02s1tail_{n}.sh")
        assert re.search(r'ev\(\) \{ \[ \$DRY = 1 \] \|\| echo', t), f"{n}: ev() writes its status file in DRY"
    assert "ZZK2DC02S1TAILALL-DRY-SWEEP-DONEZZ" in _t("scripts/k2dc02s1tail_chain_all.sh")


@pytest.mark.parametrize("name,needle", [("build", "NO-RUN"), ("stageA", "NO-RUN"), ("repro_roll", "NO-RUN"), ("stageB_launch", "guard 1")])
def test_dry_refuses_first_at_the_missing_run(name, needle):
    r = _run(SCRIPTS / f"k2dc02s1tail_{name}.sh", "DRY")
    assert r.returncode == 3, (r.stdout, r.stderr)
    assert needle in r.stdout and "K2DC02s1 has not trained" in r.stdout
    assert len(r.stdout.strip().split("\n")) == 1                                              # the FIRST guard, and nothing before it


def test_chain_all_dry_is_a_sweep_that_names_the_first_refusal():
    r = _run(SCRIPTS / "k2dc02s1tail_chain_all.sh", "DRY")
    assert r.returncode == 0, (r.stdout, r.stderr)
    out = r.stdout
    for tag in ("pin-check", "build", "stageA", "repro-roll", "stageB"):
        assert f"=== [dry] {tag} rc " in out, tag
    assert "DRY-SWEEP-DONEZZ first refusal: pin-check" in out                                   # locally the pin root does not exist: the pin check is the first refusal


# ---------------------------------------------------------------------------------------------------------------- #
# the pin check
# ---------------------------------------------------------------------------------------------------------------- #
DIRS = ("cmp/E2_K0_pf", "cmp_k2dc/E2_K2_pf", "stageB/E2_K0_pf", "stageB/E2_K2_pf")


def _make_pin(root: Path, k0_diverge=False, foreign=False):
    lines = []
    for d in DIRS:
        for i in range(3):
            p = root / d / f"f{i}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            body = f"{d}-{i}".encode() if not (k0_diverge and d == "stageB/E2_K0_pf" and i == 1) else b"diverged"
            if d.endswith("E2_K0_pf") and not k0_diverge:
                body = f"K0-{i}".encode()
            p.write_bytes(body)
            lines.append(f"{hashlib.sha256(body).hexdigest()}  {d}/f{i}.json")
    if foreign:
        lines.append(f"{'0' * 64}  other/dir/x.json")
    man = "\n".join(lines) + "\n"
    (root / "K2DC02S1_CMP_PIN.sha256").write_bytes(man.encode())
    return hashlib.sha256(man.encode()).hexdigest(), len(lines)


def _pin(root, sha, n):
    return _run(SCRIPTS / "k2dc02s1tail_pin_check.sh", root.as_posix(), sha, str(n))


def _tree(root):
    return sorted((str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file())


def test_pin_check_passes_and_writes_nothing(tmp_path):
    sha, n = _make_pin(tmp_path)
    before = _tree(tmp_path)
    r = _pin(tmp_path, sha, n)
    assert r.returncode == 0 and "ZZK2DC02S1TAILPIN-OKZZ" in r.stdout, (r.stdout, r.stderr)
    assert _tree(tmp_path) == before


@pytest.mark.parametrize("mutant", ["byte", "missing", "extra", "manifest_edit", "nlines", "foreign", "k0_diverge", "no_manifest"])
def test_pin_check_refuses_every_mutant(tmp_path, mutant):
    sha, n = _make_pin(tmp_path, k0_diverge=(mutant == "k0_diverge"), foreign=(mutant == "foreign"))
    if mutant == "foreign":                                                                       # the manifest is self-consistent, so only the prefix rule can refuse it
        sha = hashlib.sha256((tmp_path / "K2DC02S1_CMP_PIN.sha256").read_bytes()).hexdigest()
    if mutant == "k0_diverge":
        sha = hashlib.sha256((tmp_path / "K2DC02S1_CMP_PIN.sha256").read_bytes()).hexdigest()
    if mutant == "byte":
        (tmp_path / "cmp_k2dc/E2_K2_pf/f1.json").write_bytes(b"changed")
    elif mutant == "missing":
        (tmp_path / "stageB/E2_K2_pf/f0.json").unlink()
    elif mutant == "extra":
        (tmp_path / "cmp/E2_K0_pf/unlisted.json").write_bytes(b"x")
    elif mutant == "manifest_edit":
        m = tmp_path / "K2DC02S1_CMP_PIN.sha256"
        m.write_bytes(m.read_bytes().replace(b"f0.json", b"f9.json", 1))
    elif mutant == "nlines":
        n += 1
    elif mutant == "no_manifest":
        (tmp_path / "K2DC02S1_CMP_PIN.sha256").unlink()
    r = _pin(tmp_path, sha, n)
    assert r.returncode == 3 and "ZZK2DC02S1TAILPIN-REFUSED" in r.stdout and "ZZK2DC02S1TAILPIN-OKZZ" not in r.stdout, (mutant, r.stdout, r.stderr)


def test_pin_check_defaults_are_the_preregs():
    t = _t("scripts/k2dc02s1tail_pin_check.sh")
    assert PIN_SHA in t and "NL=${3:-3524}" in t and "runs/E2TAIL_K2DC02}" in t
    r = _run(SCRIPTS / "k2dc02s1tail_pin_check.sh", "C:/does/not/exist")
    assert r.returncode == 3 and "no manifest" in r.stdout
    assert PIN_SHA in _t("devbox/k2dc02s1tail_common.sh")


# ---------------------------------------------------------------------------------------------------------------- #
# the combined reading (sec. 4 + 5)
# ---------------------------------------------------------------------------------------------------------------- #
def _doc(outcome, c0, c1, arm="K2DC02s1"):
    def seed(c):
        return {"dc_vs_k2": {"LAT_cross_mae_m": {"delta": c, "lo": c - 0.002, "hi": c + 0.002}, "LAT_cross_abs_final_m": {"delta": c / 2, "lo": -0.01, "hi": 0.01}},
                "dc_vs_k2_by_family": {"LONGITUDINAL": {"LON_accel_mae": {"delta": 0.01, "lo": -0.01, "hi": 0.03, "separated": False}},
                                       "LATERAL": {"LAT_cross_mae_m": {"delta": c, "lo": c - 0.002, "hi": c + 0.002, "separated": True}},
                                       "TACTICAL": {"TAC_traj_lat_correct": {"delta": -0.002, "lo": -0.01, "hi": 0.006, "separated": False}}},
                "jitter_dlag_dc_vs_k2": {"delta": -0.001, "lo": -0.004, "hi": 0.002}}
    return {"arm": {"arm": arm}, "outcome": outcome, "seeds": {"0": seed(c0), "1": seed(c1)}}


def test_replicated_needs_supported_and_the_floor_cleared_at_both_seeds():
    r = combined.combine(_doc("SUPPORTED", -0.004, -0.0035), _doc("INCONCLUSIVE", 0.001, -0.002))
    assert r["reading"] == "DOSE-REPLICATED" and r["floor_admissibility"]["clears_both"] is True


def test_mutant_supported_that_does_not_clear_the_floor_reads_unresolved():
    for floor in (_doc("REFUTED", 0.001, -0.0036), _doc("INCONCLUSIVE", -0.004, 0.0), _doc("INCONCLUSIVE", 0.0041, 0.0)):         # seed 1 exceeds / seed 0 ties / seed 0 exceeds
        r = combined.combine(_doc("SUPPORTED", -0.004, -0.0035), floor)
        assert r["reading"] == "DOSE-UNRESOLVED", (r["reading"], r["floor_admissibility"])
        assert r["reading"] != "DOSE-REPLICATED"


def test_refuted_is_eliminated_whatever_the_floor_says():
    r = combined.combine(_doc("REFUTED", 0.0004, -0.0006), _doc("SUPPORTED", -0.009, -0.009))
    assert r["reading"] == "DOSE-ELIMINATED"


def test_the_floor_tool_label_never_makes_a_reading():
    r = combined.combine(_doc("INCONCLUSIVE", -0.004, -0.004), _doc("REFUTED", 0.0, 0.0))
    assert r["reading"] == "DOSE-UNRESOLVED"
    r2 = combined.combine(_doc("INCONCLUSIVE", -0.004, -0.004), _doc("SUPPORTED", 0.0, 0.0))
    assert r2["reading"] == "DOSE-UNRESOLVED"
    assert combined.floor_block(_doc("REFUTED", 0.0, 0.0))["tool_label_not_a_verdict"] == "REFUTED"


def test_inconclusive_is_unresolved():
    assert combined.combine(_doc("INCONCLUSIVE", -0.004, 0.001), _doc("INCONCLUSIVE", 0.0, 0.0))["reading"] == "DOSE-UNRESOLVED"


@pytest.mark.parametrize("outcome", ["DOSE-VOID", "SUPPORTED (UNCHECKED-VOID: the in-training share condition was not read)", "REFUTED (UNCHECKED-VOID: x)"])
def test_void_outcomes_decide_nothing(outcome):
    r = combined.combine(_doc(outcome, -0.004, -0.004), _doc("INCONCLUSIVE", 0.0, 0.0))
    assert r["reading"] == "VOID" and r["voids"]


@pytest.mark.parametrize("flag", ["tail_void", "repro_void"])
def test_tail_and_repro_void_beat_a_supported_that_clears(flag):
    r = combined.combine(_doc("SUPPORTED", -0.004, -0.004), _doc("INCONCLUSIVE", 0.0, 0.0), **{flag: True})
    assert r["reading"] == "VOID" and ("TAIL-VOID" if flag == "tail_void" else "REPRO-VOID") in r["voids"]


def test_wrong_arm_or_missing_row_is_refused():
    with pytest.raises(combined.Refused):
        combined.combine(_doc("SUPPORTED", -0.004, -0.004, arm="K2DC02"), _doc("INCONCLUSIVE", 0.0, 0.0))
    d = _doc("SUPPORTED", -0.004, -0.004)
    del d["seeds"]["1"]["dc_vs_k2"]["LAT_cross_mae_m"]
    with pytest.raises(combined.Refused):
        combined.combine(d, _doc("INCONCLUSIVE", 0.0, 0.0))
    with pytest.raises(combined.Refused):
        combined.combine(_doc("MAYBE", 0.0, 0.0), _doc("INCONCLUSIVE", 0.0, 0.0))


def test_ema_mode_is_reported_beside_and_never_a_section_5_label():
    r = combined.combine(_doc("SUPPORTED", -0.004, -0.004), _doc("INCONCLUSIVE", 0.0, 0.0), mode="ema")
    assert r["reading"] == "EMA-REPORTED-BESIDE" and r["outcome_base"] == "SUPPORTED" and r["floor_admissibility"]["clears_both"] is True
    assert combined.combine(_doc("DOSE-VOID", 0.0, 0.0), _doc("INCONCLUSIVE", 0.0, 0.0), mode="ema")["reading"] == "VOID"


def test_main_writes_the_reading_with_the_floor_rows_and_exit_codes(tmp_path):
    for name, doc in (("dose", _doc("SUPPORTED", -0.004, -0.0035)), ("floor", _doc("INCONCLUSIVE", 0.001, -0.002)), ("floor_bad", _doc("INCONCLUSIVE", 0.001, -0.0037))):
        (tmp_path / name).mkdir()
        (tmp_path / name / "DOSE_VERDICT.json").write_text(json.dumps(doc), encoding="utf-8")
    rc = combined.main(["--mode", "tail", "--dose-dir", str(tmp_path / "dose"), "--floor-dir", str(tmp_path / "floor"), "--out-dir", str(tmp_path / "o1")])
    j = json.loads((tmp_path / "o1" / "COMBINED_READING.json").read_text(encoding="utf-8"))
    md = (tmp_path / "o1" / "COMBINED_READING.md").read_text(encoding="utf-8")
    assert rc == 0 and j["reading"] == "DOSE-REPLICATED" and "LONGITUDINAL" in md and "TACTICAL" in md and "never a lever verdict" in md and j["floor"]["seeds"]["1"]["dc_vs_k2_by_family"]
    rc = combined.main(["--mode", "tail", "--dose-dir", str(tmp_path / "dose"), "--floor-dir", str(tmp_path / "floor_bad"), "--out-dir", str(tmp_path / "o2")])
    assert rc == 0 and json.loads((tmp_path / "o2" / "COMBINED_READING.json").read_text(encoding="utf-8"))["reading"] == "DOSE-UNRESOLVED"
    rc = combined.main(["--mode", "tail", "--dose-dir", str(tmp_path / "dose"), "--floor-dir", str(tmp_path / "floor"), "--out-dir", str(tmp_path / "o3"), "--repro-void"])
    assert rc == 5
    rc = combined.main(["--mode", "tail", "--dose-dir", str(tmp_path / "nodir"), "--floor-dir", str(tmp_path / "floor"), "--out-dir", str(tmp_path / "o4")])
    assert rc == 4


# ---------------------------------------------------------------------------------------------------------------- #
# the reproduction check (sec. 2 item 6)
# ---------------------------------------------------------------------------------------------------------------- #
def _fam(delta, sep=False):
    return {"rows": {"LATERAL": {"LAT_cross_mae_m": {"delta": delta, "separated": sep, "n": 10}, "LAT_heading_deg": {"delta": 0.05, "separated": False, "n": 10}},
                     "LONGITUDINAL": {"LON_accel_mae": {"delta": 0.01, "separated": False, "n": 10}}, "TACTICAL": {}}}


def _repro(pinned_delta, reroll_delta):
    return repro.compare(repro.family_rows({"0": _fam(pinned_delta)}), repro.family_rows({"0": _fam(reroll_delta)}))


@pytest.mark.parametrize("shift,want", [(0.0, "REPRO-PASS"), (0.0009, "REPRO-PASS"), (-0.0009, "REPRO-PASS"), (0.001, "REPRO-PASS"), (0.0011, "REPRO-VOID"), (-0.002, "REPRO-VOID")])
def test_repro_band_is_plus_minus_1_mm_at_seed_0(shift, want):
    assert _repro(0.0126, 0.0126 + shift)["verdict"] == want


def test_repro_missing_row_or_seed_voids():
    d = _fam(0.0126)
    del d["rows"]["LATERAL"]["LAT_cross_mae_m"]
    assert repro.compare(repro.family_rows({"0": _fam(0.0126)}), repro.family_rows({"0": d}))["verdict"] == "REPRO-VOID"
    assert repro.compare(repro.family_rows({"0": _fam(0.0126)}), repro.family_rows({}))["verdict"] == "REPRO-VOID"
    assert repro.SEEDS == ("0",)


def test_repro_main_and_selftest(tmp_path):
    for t, v in (("pinned", 0.0126), ("reroll", 0.0130)):
        (tmp_path / t).mkdir()
        (tmp_path / t / "families_s0.json").write_text(json.dumps(_fam(v)), encoding="utf-8")
    assert repro.main(["--pinned-dir", str(tmp_path / "pinned"), "--reroll-dir", str(tmp_path / "reroll"), "--out", str(tmp_path / "o.json")]) == 0
    (tmp_path / "reroll" / "families_s0.json").write_text(json.dumps(_fam(0.0150)), encoding="utf-8")
    assert repro.main(["--pinned-dir", str(tmp_path / "pinned"), "--reroll-dir", str(tmp_path / "reroll")]) == 5
    assert repro.main(["--selftest", str(tmp_path / "pinned")]) == 0


# ---------------------------------------------------------------------------------------------------------------- #
# the dose-tool arm guard and the verdict scripts' invocations
# ---------------------------------------------------------------------------------------------------------------- #
def _guard(tmp_path, tool_text, with_argv=True):
    c = tmp_path / "code"
    (c / "stack/scripts").mkdir(parents=True)
    (c / "stack/ops/runs.d").mkdir(parents=True)
    (c / "stack/scripts/r9_e2_dose_verdict.py").write_text(tool_text, encoding="utf-8")
    if with_argv:
        (c / "stack/ops/runs.d/r9e2_K2DC02s1.argv.json").write_text("{}", encoding="utf-8")
    script = tmp_path / "g.sh"
    script.write_bytes(f'. "{(DEVBOX / "k2dc02s1tail_common.sh").as_posix()}"\nC="{c.as_posix()}"\ndose_tool_guard\n'.encode())
    return _run(script, env={"K2T_D": (tmp_path / "d").as_posix()})


def test_dose_tool_guard_accepts_a_tool_that_knows_the_arm(tmp_path):
    r = _guard(tmp_path, 'ARM_ARGV = {"K2DC02": "r9e2_K2DC02.argv.json", "K2DC02s1": "r9e2_K2DC02s1.argv.json"}      # arms\n')
    assert r.returncode == 0, (r.stdout, r.stderr)


def test_dose_tool_guard_refuses_without_the_arm_or_its_argv(tmp_path):
    r = _guard(tmp_path, 'ARM_ARGV = {"K2DC02": "r9e2_K2DC02.argv.json"}      # arms\n')
    assert r.returncode == 1 and "no K2DC02s1 in ARM_ARGV" in r.stdout
    (tmp_path / "b").mkdir()
    r = _guard(tmp_path / "b", 'ARM_ARGV = {"K2DC02": "r9e2_K2DC02.argv.json", "K2DC02s1": "x"}\n', with_argv=False)
    assert r.returncode == 1 and "argv.json" in r.stdout


def test_the_tool_at_the_preregs_tip_does_not_know_the_arm(tmp_path):
    """DOCUMENTS THE BLOCKER: at 48c797f3 r9_e2_dose_verdict.py has ARM_ARGV = {K2DC02} only, so `--arm K2DC02s1` is refused ('unknown --arm'). This test flips when the MM lands the arm."""
    t = subprocess.run(["git", "--git-dir", GD, "show", f"{SRC_COMMIT}:stack/scripts/r9_e2_dose_verdict.py"], capture_output=True)
    if t.returncode != 0:
        pytest.skip("mirror absent")
    r = _guard(tmp_path, t.stdout.decode("utf-8"))
    assert r.returncode == 1 and "REFUSED" in r.stdout


def test_the_kits_default_tip_carries_the_arm_and_the_guard_passes(tmp_path):
    """MM 11-20: the arm entry landed at 20179573; the kit's default K2T_TIP is that commit, and the guard PASSES on the tool exported from it."""
    common = (DEVBOX / "k2dc02s1tail_common.sh").read_text(encoding="utf-8")
    assert "TIP=${K2T_TIP:-20179573828bfd78b7c28ba64cbba52fe26fea74}" in common
    t = subprocess.run(["git", "--git-dir", GD, "show", "20179573828bfd78b7c28ba64cbba52fe26fea74:stack/scripts/r9_e2_dose_verdict.py"], capture_output=True)
    if t.returncode != 0:
        pytest.skip("mirror absent")
    r = _guard(tmp_path, t.stdout.decode("utf-8"))
    assert r.returncode == 0, (r.stdout, r.stderr)


def test_dose_invocations_and_the_floor_slot():
    v = _code("devbox/k2dc02s1tail_verdicts.sh")
    assert v.count("--arm K2DC02s1") == 2 and not re.search(r"--arm K2DC02(?!s1)", v)
    assert '--k2dc-pf "$(W2 "$PFK2")" --k2-pf "$(W2 "$PFK2DC")" --k0-pf "$(W2 "$PFK0")"' in v                       # dose: K2DCt in K2's slot, K0t as K0
    assert '--k2dc-pf "$(W2 "$PFK2")" --k2-pf "$(W2 "$PFK2DC02")" --k0-pf "$(W2 "$PFK0")"' in v                     # floor: K2DC02t in K2's slot
    assert '--k2-read "$(W2 "$C/$E2TREAD")"' in v and '--k2-read "$(W2 "$C/$K2DC02TREAD")"' in v
    assert "--mode tail" in v and "--repro-void" in v and "--tail-void" in v and "DOSETOOL" in v and v.index("DOSETOOL") < v.index("mkdir -p \"$V/repro")
    assert v.index("floor_verdict.log") < v.index("k2dc02s1tail_combined.py")
    assert '"K2DC02s1t:0=' in v and '"K2DC02t:0=' in v and '"K2DCt:0=' in v and '"K0t:0=' in v and "--ref K0t" in v                  # STRATEGIC arms
    assert "k2dc02s1tail_sanitize_navcomp.py" in v and "REGRESSION-RED" in v and "IDCTL-PASS-TENSOR" in v
    assert 'for pair in "pinned:$PFK0" "reroll:$PFREPRO"' in v and "S=0" in v                                              # reproduction at inference seed 0 only


def test_ema_verdict_script_reads_what_the_ema_launcher_produces():
    e = _code("devbox/k2dc02s1tail_ema_verdicts.sh")
    assert e.count("--arm K2DC02s1") == 2 and not re.search(r"--arm K2DC02(?!s1)", e)
    assert "T_S1=$RUNS/E2_READ_K2DC02s1" in e and "T_K2DC=$RUNS/E2_READ_K2DC;" in e and "T_K2DC02=$RUNS/E2_READ_K2DC02" in e
    assert '--k2dc-pf "$(W2 "$PF_S1")" --k2-pf "$(W2 "$PF_K2DC")" --k0-pf "$(W2 "$PF_K0")"' in e                       # dose: K2DC (EMA) in K2's slot
    assert '--k2dc-pf "$(W2 "$PF_S1")" --k2-pf "$(W2 "$PF_K2DC02")" --k0-pf "$(W2 "$PF_K0")"' in e                    # floor: K2DC02 (EMA) in K2's slot
    assert '--k2-read "$(W2 "$RD_K2DC")"' in e and '--k2-read "$(W2 "$RD_K2DC02")"' in e and "--mode ema" in e
    assert "DOSETOOL" in e and "PF_K0=$P/s1/E2_K0_pf" in e
    assert "r9e2-K2DC02s1/metrics.jsonl" in e


def test_s6c_step_checks_the_pin_before_pulling_and_the_tail_md5():
    s = _t("devbox/k2dc02s1tail_s6c_pull_score.sh")
    assert s.index("THORPIN") < s.index("pull stageB") and s.index("pull repro") < s.index("PINSHA") < s.index("PINPULLSHA")
    assert "$PIN_SHA" in s and "TAILGUARD-MD5" in s and "tail_md5.txt" in s
    assert "cmp_reroll" not in s and "CMP_STATUS" not in s


# ---------------------------------------------------------------------------------------------------------------- #
# the seed guard counts INSIDE argv / argv_effective (MM 4a3d05, 2026-11-22). The 1f14f8d8 trainer writes the argv twice in config.json
# (argv + argv_effective, equal), so the old whole-file count read flag x2 / pair x2 and refused the valid K2DC02s1 tail read at
# 17:12:39Z (ABORT-SEED). Same defect and same form as the EMA read launchers' fix (8b581767).
# ---------------------------------------------------------------------------------------------------------------- #
SEED_ARM, SEED_VAL = "K2DC02s1", "1"
OLD_SEED_GUARD = 'NS=$(printf \'%s\' "$CFG1" | grep -o \'"--seed"\' | wc -l); NV=$(printf \'%s\' "$CFG1" | grep -o \'"--seed","1"\' | wc -l); [ "$NS" = 1 ] && [ "$NV" = 1 ] || die SEED "config.json must carry --seed 1 exactly once (flag x$NS, pair x$NV): not the replicate argv"'


def _seed_guard_lines(text):
    ls = [l for l in text.split("\n") if l.startswith("CFG1=$(tr -d") or l.startswith("NS=$(printf")]
    assert len(ls) == 2 and ls[0].startswith("CFG1="), ls
    return ls


def _run_seed_guard(tmp_path, cfg, lines=None):
    lines = lines or _seed_guard_lines(_t("scripts/k2dc02s1tail_build.sh"))
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    (run / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    sh = tmp_path / "seed_guard.sh"
    body = [SEED_ARM + "=" + run.as_posix(), 'die() { echo "ABORT-$1 ${*:2}"; exit 3; }'] + lines + ["echo GUARD-OK"]
    sh.write_bytes(("\n".join(body) + "\n").encode("utf-8"))
    return _run(sh)


def _argv(seed=None, extra=()):
    a = ["--out", "runs/r9e2-" + SEED_ARM, "--seed", seed or SEED_VAL, "--w-map-hires", "25.0", "--map-hires-w-bounds", "0.1,25", *extra]
    return a


def test_seed_guard_accepts_the_trainers_real_layout_argv_and_argv_effective(tmp_path):
    r = _run_seed_guard(tmp_path, {"arm": SEED_ARM, "seed": int(SEED_VAL), "argv": _argv(), "argv_effective": _argv()})
    assert r.returncode == 0 and "GUARD-OK" in r.stdout, r.stdout + r.stderr


def test_seed_guard_accepts_argv_alone(tmp_path):
    r = _run_seed_guard(tmp_path, {"arm": SEED_ARM, "argv": _argv()})
    assert r.returncode == 0 and "GUARD-OK" in r.stdout, r.stdout + r.stderr


@pytest.mark.parametrize("cfg_kind", ["argv_wrong_seed", "effective_wrong_seed", "argv_seed_twice", "effective_missing_seed", "two_effective_keys"])
def test_seed_guard_refuses_every_wrong_config(tmp_path, cfg_kind):
    other = "1" if SEED_VAL == "0" else "0"
    good = _argv()
    cfg = {
        "argv_wrong_seed": {"argv": _argv(other), "argv_effective": _argv(other)},
        "effective_wrong_seed": {"argv": good, "argv_effective": _argv(other)},
        "argv_seed_twice": {"argv": good + ["--seed", SEED_VAL], "argv_effective": good},
        "effective_missing_seed": {"argv": good, "argv_effective": [t for t in good if t not in ("--seed",)][:2]},
    }.get(cfg_kind)
    if cfg is None:      # two argv_effective keys cannot come from json.dumps: write the text by hand
        run = tmp_path / "run"
        run.mkdir(exist_ok=True)
        lst = json.dumps(good)
        (run / "config.json").write_text('{"argv": %s, "argv_effective": %s, "x": {"argv_effective": %s}}' % (lst, lst, lst), encoding="utf-8")
        sh = tmp_path / "seed_guard.sh"
        body = [SEED_ARM + "=" + run.as_posix(), 'die() { echo "ABORT-$1 ${*:2}"; exit 3; }'] + _seed_guard_lines(_t("scripts/k2dc02s1tail_build.sh")) + ["echo GUARD-OK"]
        sh.write_bytes(("\n".join(body) + "\n").encode("utf-8"))
        r = _run(sh)
    else:
        r = _run_seed_guard(tmp_path, cfg)
    assert r.returncode == 3 and "ABORT-SEED" in r.stdout and "GUARD-OK" not in r.stdout, (cfg_kind, r.stdout + r.stderr)


def test_the_old_whole_file_seed_guard_refuses_the_real_layout(tmp_path):
    """the deliberate-regression arm: the pre-fix line, run on the trainer's real layout, must reproduce the 17:12:39Z refusal"""
    cfg_line = _seed_guard_lines(_t("scripts/k2dc02s1tail_build.sh"))[0]
    r = _run_seed_guard(tmp_path, {"arm": SEED_ARM, "argv": _argv(), "argv_effective": _argv()}, lines=[cfg_line, OLD_SEED_GUARD])
    assert r.returncode == 3 and "ABORT-SEED" in r.stdout and "flag x2" in r.stdout, r.stdout + r.stderr
    assert OLD_SEED_GUARD not in _t("scripts/k2dc02s1tail_build.sh")
