"""Tests of the K2DC02nh tail-read kit (PREREG_K2DC02NH_2026-11-20.md sec. 2 to 4). Kept in the kit, not in stack/tests. Mirrors test_k2dc02s1tail_kit.py (the template kit).

Run (one pytest process at a time, >= 5.5 GB commit free) from a tree that holds the kit, stack/ and the e2raw PREREG files (a fresh `git archive` export):
    PYTHONPATH="<export>/stack;<export>" C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q -p no:cacheprovider <this file>
What is pinned:
  * every derived script keeps most of its k2dc02s1tail source lines (compared after K2DC02nh -> K2DC02s1; the source is read from git, and the test SKIPS if the mirror is absent); the python helpers that were not changed are the source verbatim;
  * the kept guards (tail tool blob, identity control, deliberate-regression arm, roll/env blobs, GPU lock, pin check) are still in the text, and the nh guards (seed 0, NO --r9-hist-supervise-geom, share target 0.02, ' K2DC02nh-DONE ') are there;
  * no leftover 's1' / 'K2DC02s1' token outside comments / strings that cite the template, its PREREG or its pin manifest;
  * the lever rule (sec. 3) on synthetic inputs: its constants equal the PREREG's (-2.5 mm, +-1 mm) and the pinned dose tool's, it agrees with the dose tool's own `decide` over a grid, and mutants of every clause go RED;
  * the sec. 4 readings (SUPPORTED items 1/2, REFUTED, INCONCLUSIVE, VOID) and LEVER_READING.json's `reading` field; bash -n of every .sh, LF endings and repo path length."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import itertools
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

KIT = Path(__file__).resolve().parents[1]
SCRIPTS, DEVBOX = KIT / "scripts", KIT / "devbox"
E2RAW = KIT.parent
GD = "C:/Users/Admin/tanitad-push/.git"
SRC_COMMIT = "e30e7b6e910fd85febb9ab342c784f200ceb429b"
SRC_DIR = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-27-perception-redesign/e2/e2raw/k2dc02s1tail"
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


def _git_show(commit_path: str) -> bytes:
    r = subprocess.run(["git", "--git-dir", GD, "show", commit_path], capture_output=True)
    if r.returncode != 0:
        pytest.skip(f"cannot read {commit_path} from the mirror")
    return r.stdout


def _src(rel: str) -> str:
    return _git_show(f"{SRC_COMMIT}:{SRC_DIR}/{rel}").decode("utf-8")


def _norm(t: str) -> str:
    """derived text -> the template's spelling"""
    return t.replace("K2DC02NH", "K2DC02S1").replace("K2DC02nh", "K2DC02s1").replace("k2dc02nh", "k2dc02s1")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


lever = _load(DEVBOX / "k2dc02nhtail_lever.py", "k2n_lever")
repro = _load(DEVBOX / "k2dc02nhtail_repro_check.py", "k2n_repro")
ALL_SH = sorted(list(SCRIPTS.glob("*.sh")) + list(DEVBOX.glob("*.sh")))
ALL_FILES = sorted(p for p in KIT.rglob("*") if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts)


def _repo_root():
    for p in KIT.parents:
        if (p / "stack" / "scripts" / "r9_e2_dose_verdict.py").is_file():
            return p
    return None


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


def test_every_file_is_named_k2dc02nhtail():
    for p in ALL_FILES:
        assert p.name.startswith("k2dc02nhtail_") or p.name.startswith("test_k2dc02nhtail"), p.name


def test_python_modules_run_nothing_on_import():
    src = (DEVBOX / "k2dc02nhtail_fam_table.py").read_text(encoding="utf-8") + (DEVBOX / "k2dc02nhtail_sanitize_navcomp.py").read_text(encoding="utf-8")
    assert src.count('if __name__ == "__main__":') == 2
    for n in ("k2dc02nhtail_lever.py", "k2dc02nhtail_repro_check.py", "k2dc02nhtail_strategic_navcomp.py"):
        assert 'if __name__ == "__main__":' in (DEVBOX / n).read_text(encoding="utf-8")


def test_the_arm_is_named_only_as_nh_markers_are_renamed():
    for p in ALL_SH:
        t = p.read_text(encoding="utf-8")
        assert "ZZK2DC02s1" not in t and "ZZK2DC02S1" not in t, p.name
    assert "ZZK2DC02nhT-S6C-DONEZZ" in (DEVBOX / "k2dc02nhtail_s6c_pull_score.sh").read_text(encoding="utf-8")
    assert "ZZK2DC02nhT-VERD-DONEZZ" in (DEVBOX / "k2dc02nhtail_verdicts.sh").read_text(encoding="utf-8")
    assert "ZZK2DC02NHTAILPIN-OKZZ" in (SCRIPTS / "k2dc02nhtail_pin_check.sh").read_text(encoding="utf-8")
    assert "runs/E2TAIL_K2DC02nh" in (SCRIPTS / "k2dc02nhtail_build.sh").read_text(encoding="utf-8")
    assert "/f/refcv9_e3/k2dc02nhtail" in (DEVBOX / "k2dc02nhtail_common.sh").read_text(encoding="utf-8")


ALLOW_LINE = re.compile(r"PREREG_K2DC02S1|K2DC02S1_CMP_PIN|k2dc02s1tail_|template|Template")
TOKEN = re.compile(r"K2DC02s1|k2dc02s1|K2DC02S1|\bs1\b|\bT_S1\b|\bRD_S1\b|\bPF_S1\b")


def test_no_leftover_s1_token_outside_citations():
    bad = []
    for p in ALL_FILES:
        if p.name.startswith("test_"):
            continue
        for i, l in enumerate(p.read_text(encoding="utf-8").split("\n"), 1):
            if TOKEN.search(l) and not ALLOW_LINE.search(l):
                bad.append(f"{p.name}:{i}: {l[:120]}")
    assert not bad, "\n".join(bad)


# ---------------------------------------------------------------------------------------------------------------- #
# derived files keep their template lines (after K2DC02nh -> K2DC02s1)
# ---------------------------------------------------------------------------------------------------------------- #
SURVIVE = {                      # derived file -> minimum fraction of the template's lines that survive unchanged (the rewritten verdict scripts and the common file are bounded lower)
    "scripts/k2dc02nhtail_build.sh": 0.88, "scripts/k2dc02nhtail_chain.sh": 0.80, "scripts/k2dc02nhtail_chain_all.sh": 0.80, "scripts/k2dc02nhtail_pin_check.sh": 0.90,
    "scripts/k2dc02nhtail_repro_roll.sh": 0.90, "scripts/k2dc02nhtail_stageA.sh": 0.90, "scripts/k2dc02nhtail_stageB_launch.sh": 0.85,
    "devbox/k2dc02nhtail_s6c_pull_score.sh": 0.80, "devbox/k2dc02nhtail_common.sh": 0.45, "devbox/k2dc02nhtail_verdicts.sh": 0.45, "devbox/k2dc02nhtail_ema_verdicts.sh": 0.30,
    "devbox/k2dc02nhtail_repro_check.py": 0.90,
}


@pytest.mark.parametrize("rel", list(SURVIVE), ids=lambda r: r.split("/")[-1])
def test_derived_keeps_most_template_lines(rel):
    src = _src(rel.replace("k2dc02nhtail_", "k2dc02s1tail_")).split("\n")
    dset = set(_norm((KIT / rel).read_text(encoding="utf-8")).split("\n"))
    kept = sum(1 for l in src if l in dset)
    assert kept >= SURVIVE[rel] * len(src), f"{rel}: only {kept} of {len(src)} template lines survive (need {SURVIVE[rel]:.0%})"


def test_untouched_helpers_are_the_template_verbatim(tmp_path):
    assert _norm((DEVBOX / "k2dc02nhtail_strategic_navcomp.py").read_text(encoding="utf-8")) == _src("devbox/k2dc02s1tail_strategic_navcomp.py")
    ev = {"four_families": {"0": {"LATERAL": {"status": "PASS", "rows": {"LAT_cross_mae_m": {"delta": 0.0123, "lo": 0.001, "hi": 0.02, "better": "lower", "separated_worse": True, "n": 100, "n_episodes": 5}}},
                                  "TACTICAL": {"status": "n/a"}}}}
    (tmp_path / "v.json").write_text(json.dumps(ev), encoding="utf-8")
    (tmp_path / "src_fam.py").write_text(_src("devbox/k2dc02s1tail_fam_table.py"), encoding="utf-8")
    a = subprocess.run([sys.executable, "-I", str(tmp_path / "src_fam.py"), str(tmp_path / "v.json")], capture_output=True, text=True).stdout
    b = subprocess.run([sys.executable, "-I", str(DEVBOX / "k2dc02nhtail_fam_table.py"), str(tmp_path / "v.json")], capture_output=True, text=True).stdout
    assert a and _norm(b) == a
    (tmp_path / "src_san.py").write_text(_src("devbox/k2dc02s1tail_sanitize_navcomp.py"), encoding="utf-8")
    u = "-".join(["12345678", "1234", "1234", "1234", "123456789abc"])      # built at run time: the lander refuses id-shaped strings in landed files
    for tag, script in (("a", tmp_path / "src_san.py"), ("b", DEVBOX / "k2dc02nhtail_sanitize_navcomp.py")):
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
# kept guards, the nh config guards, no comparator re-roll, one-seed reproduction roll, DRY structure
# ---------------------------------------------------------------------------------------------------------------- #
def _t(name):
    return (KIT / name).read_text(encoding="utf-8")


def _code(name):
    """the script without its comment-only lines (header comments legitimately name what was removed)"""
    return "\n".join(l for l in _t(name).split("\n") if not l.lstrip().startswith("#"))


def test_tail_build_keeps_its_guards_and_adds_the_nh_config_guards():
    b = _t("scripts/k2dc02nhtail_build.sh")
    for must in ("TOOL_BLOB=e456ff67be132e9090f688dde4b8e9043ea43d5d", "--ema-a $K2DC02nh/ckpt_ema_5000.pt --ema-b $K2DC02nh/model_ema.pt", "IDCTL_TENSOR_EQUAL", "torch.equal", "IDCTL-PASS-TENSOR",
                 "0.9998", " K2DC02nh-DONE ", "arms \\[K2DC02nh\\]", "W_tail = (EMA_6000 - 0.81872 * EMA_5000) / 0.18128", "never overwritten"):
        assert must in b, must
    # the nh config guards: seed 0 exactly once, NO --r9-hist-supervise-geom (K2DC02 minus that flag), share target 0.02
    assert '"--seed","0"' in b and '"--seed","1"' not in b
    assert "die HSG-PRESENT" in b and "*'\"--r9-hist-supervise-geom\"'*) die HSG-PRESENT" in b and "die NO-HSG" not in b
    assert "--map-hires-share-target\",\"0.02\"" in b
    assert b.index("die NO-RUN") < b.index("TC=${K2S_TRAIN_COMMIT") < b.index("TOOL=$RT/stack/scripts/ema_tail_weights.py")           # the missing-run guard is the FIRST guard
    assert b.index("[ $DRY = 1 ]") < b.index("rm -rf \"${SC:?}\"")                            # dry exits before the scratch rebuild
    assert "runs/r9e2-K2DC02nh" in b and "$W0/inputs/K2DC.model_ema_tail5k6k.pt.provenance.json" in b
    ca = _t("scripts/k2dc02nhtail_chain_all.sh")
    assert "IDCTL_TENSOR_EQUAL=1 BANKED_PROV_SHA_PIN=9065c97666dfad6d9f259a7124f96cd50aa6dfe88ea0fd04356f016f09f72f8a" in ca


def test_nh_guards_agree_with_the_ema_launcher_and_the_argv_json():
    """the three config guards of the tail build are the ones e2read_launch_k2dc02nh.sh has; the nh argv json (when the tree holds it) has no HSG token, seed 0, share target 0.02."""
    root = _repo_root()
    if root is None:
        pytest.skip("no repo root above the kit")
    la = (root / "stack/ops/r9e2_read/e2read_launch_k2dc02nh.sh").read_text(encoding="utf-8")
    assert "r9-hist-supervise-geom" in la and '"--seed","0"' in la and '"--map-hires-share-target","0.02"' in la
    a = json.loads((root / "stack/ops/runs.d/r9e2_K2DC02nh.argv.json").read_text(encoding="utf-8"))["argv"]
    assert "--r9-hist-supervise-geom" not in a and a[a.index("--seed") + 1] == "0" and a[a.index("--map-hires-share-target") + 1] == "0.02" and a.count("--seed") == 1
    k = json.loads((root / "stack/ops/runs.d/r9e2_K2DC02.argv.json").read_text(encoding="utf-8"))["argv"]
    # the arm is K2DC02's argv minus that one token, with --out naming the arm (test_r9e2_k2dc02nh_argv.py: the parsed diff is exactly out + r9_hist_supervise_geom)
    km = [t for t in k if t != "--r9-hist-supervise-geom"]
    assert "--r9-hist-supervise-geom" in k and len(km) == len(a) and k.count("--out") == a.count("--out") == 1
    io = a.index("--out") + 1
    assert [x for i, x in enumerate(km) if i != io] == [x for i, x in enumerate(a) if i != io]
    assert km[io] == "/home/nvidia/r8ns/r9e2/runs/r9e2-K2DC02" and a[io] == "/home/nvidia/r8ns/r9e2/runs/r9e2-K2DC02nh"


def test_stage_a_keeps_the_regression_arm_and_tail_guard():
    a = _t("scripts/k2dc02nhtail_stageA.sh")
    for must in ("REGRESSION-NOT-RED", "REGRESSION-CONTROL", "REGRESSION-RED", "DELIBERATE REGRESSION", 'int(d.get("step",-1))==6000', 'd.get("ckpt_md5")==h', "$RAWRUN/model_ema.pt $RUN/$CK $RG",
                 "LOCK=/home/nvidia/refcv7_post/thor_gpu.lock", "flock -w 43200", "GPU-BUSY", "RAWRUN=$R/runs/r9e2-K2DC02nh"):
        assert must in a, must
    assert a.index("die NO-RUN") < a.index("for f in \"$FT\"")                                # first guard before any input
    assert a.index("ev DRY-OK") < a.index("gpu_dump $A $FD")                                  # dry exits before the first GPU stage call
    assert '"$RB" = 1' in a and '"$RG2" = 0' in a


def test_stage_b_uses_the_pinned_k0t_and_cannot_be_overridden():
    b = _code("scripts/k2dc02nhtail_stageB_launch.sh")
    assert "K0_CMP" not in b and "K0_SEED:-" not in b and "K0_CMP:-" not in b
    assert "CMP=$W0/cmp" in b and "PINMAN=$W0/K2DC02S1_CMP_PIN.sha256" in b
    assert "k2dc02nhtail_pin_check.sh" in b and b.index("k2dc02nhtail_pin_check.sh") < b.index("cp -a $CMP/E2_K0_pf")
    assert "cp -al $CMP" not in b and "cp -a $CMP/E2_K0_pf $B/E2_K0_pf && grep ' cmp/E2_K0_pf/' $PINMAN" in b      # a copy, re-hashed against the pin
    assert "ROLL_BLOB=3816129936e810cece67dfb620b3813a435a99cb" in b and "d4bdcf07cf337758ee0770f72091a23e6bf4eb34" in b
    assert b.index("guard 1") < b.index("TREE/stack")


def test_no_comparator_is_re_rolled_and_the_repro_roll_is_k0t_seed0_only():
    allsrc = "\n".join(_code(p.relative_to(KIT).as_posix()) for p in ALL_FILES if p.suffix == ".sh")
    assert "cmp_reroll" not in allsrc
    r = _t("scripts/k2dc02nhtail_repro_roll.sh")
    assert r.count("--tag E2_K0_pf") == 1 and "E2_K2_pf" not in r.replace("E2_K2_pf ", "")        # only K0t is rolled
    assert "--infer-seeds 0 " in r and "--infer-seeds 0,1" not in r
    for must in ("--no-g0 --no-baselines", "--lock-held", "OMP_NUM_THREADS=6", "flock -w 43200", "GPU-NOT-EMPTY", "thor_gpu.lock", "ROLL_BLOB=3816129936e810cece67dfb620b3813a435a99cb",
                 "d4bdcf07cf337758ee0770f72091a23e6bf4eb34", 'OUTROOT=$W/repro', 'source $ENVF K2DC02nh', "never overwritten"):
        assert must in r, must
    assert r.index("die NO-RUN") < r.index("[ ${#TC} = 12 ]") < r.index("ROLL-BLOB")
    assert r.index("ev DRY-OK") < r.index("flock -w 43200") and r.index("ev DRY-OK") < r.index("mkdir -p $W\n")           # no lock, no write in DRY
    assert "runs/E2TAIL_K2DC02/" not in _code("scripts/k2dc02nhtail_repro_roll.sh").replace("runs/E2TAIL_K2DC02nh", "")                  # never names the pinned root as a target


def test_every_thor_script_has_a_dry_mode_and_writes_nothing_in_it():
    for n in ("build", "stageA", "repro_roll", "stageB_launch", "chain", "chain_all"):
        t = _t(f"scripts/k2dc02nhtail_{n}.sh")
        assert "K2S_DRY" in t or "LAUNCH_DRY" in t, n
    for n in ("build", "stageA", "repro_roll"):
        t = _t(f"scripts/k2dc02nhtail_{n}.sh")
        assert re.search(r'ev\(\) \{ \[ \$DRY = 1 \] \|\| echo', t), f"{n}: ev() writes its status file in DRY"
    assert "ZZK2DC02NHTAILALL-DRY-SWEEP-DONEZZ" in _t("scripts/k2dc02nhtail_chain_all.sh")


@pytest.mark.parametrize("name,needle", [("build", "NO-RUN"), ("stageA", "NO-RUN"), ("repro_roll", "NO-RUN"), ("stageB_launch", "guard 1")])
def test_dry_refuses_first_at_the_missing_run(name, needle):
    r = _run(SCRIPTS / f"k2dc02nhtail_{name}.sh", "DRY")
    assert r.returncode == 3, (r.stdout, r.stderr)
    assert needle in r.stdout and "K2DC02nh has not trained" in r.stdout
    assert len(r.stdout.strip().split("\n")) == 1                                              # the FIRST guard, and nothing before it


def test_chain_all_dry_is_a_sweep_that_names_the_first_refusal():
    r = _run(SCRIPTS / "k2dc02nhtail_chain_all.sh", "DRY")
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
    return _run(SCRIPTS / "k2dc02nhtail_pin_check.sh", root.as_posix(), sha, str(n))


def _tree(root):
    return sorted((str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file())


def test_pin_check_passes_and_writes_nothing(tmp_path):
    sha, n = _make_pin(tmp_path)
    before = _tree(tmp_path)
    r = _pin(tmp_path, sha, n)
    assert r.returncode == 0 and "ZZK2DC02NHTAILPIN-OKZZ" in r.stdout, (r.stdout, r.stderr)
    assert _tree(tmp_path) == before


@pytest.mark.parametrize("mutant", ["byte", "missing", "extra", "manifest_edit", "nlines", "foreign", "k0_diverge", "no_manifest"])
def test_pin_check_refuses_every_mutant(tmp_path, mutant):
    sha, n = _make_pin(tmp_path, k0_diverge=(mutant == "k0_diverge"), foreign=(mutant == "foreign"))
    if mutant in ("foreign", "k0_diverge"):                                                    # the manifest is self-consistent, so only the prefix / K0-copy rule can refuse it
        sha = hashlib.sha256((tmp_path / "K2DC02S1_CMP_PIN.sha256").read_bytes()).hexdigest()
    if mutant == "byte":
        (tmp_path / "stageB/E2_K2_pf/f1.json").write_bytes(b"changed")                        # K2DC02t: the comparator in K2's slot
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
    assert r.returncode == 3 and "ZZK2DC02NHTAILPIN-REFUSED" in r.stdout and "ZZK2DC02NHTAILPIN-OKZZ" not in r.stdout, (mutant, r.stdout, r.stderr)


def test_pin_check_defaults_are_the_pinned_ones():
    t = _t("scripts/k2dc02nhtail_pin_check.sh")
    assert PIN_SHA in t and "NL=${3:-3524}" in t and "runs/E2TAIL_K2DC02}" in t
    r = _run(SCRIPTS / "k2dc02nhtail_pin_check.sh", "C:/does/not/exist")
    assert r.returncode == 3 and "no manifest" in r.stdout
    assert PIN_SHA in _t("devbox/k2dc02nhtail_common.sh")


# ---------------------------------------------------------------------------------------------------------------- #
# the lever rule (PREREG_K2DC02NH sec. 3) and the sec. 4 readings
# ---------------------------------------------------------------------------------------------------------------- #
def _seed(nh_k0_delta=0.0, nh_k0_sep=False, cross=-0.004, final=-0.002, jit=-0.001):
    """one inference seed's numbers: dc = K2DC02nh_t, k0 = K0_t, k2 = K2DC02_t (the dose tool's `seed_numbers` keys)."""
    row = lambda d, sep=False: {"delta": d, "lo": d - 0.002, "hi": d + 0.002, "separated": sep}      # noqa: E731
    return {"dc_vs_k0": {"LAT_cross_mae_m": row(nh_k0_delta, nh_k0_sep)}, "dc_vs_k2": {"LAT_cross_mae_m": row(cross), "LAT_cross_abs_final_m": row(final)},
            "jitter_dlag_dc_vs_k2": {"delta": jit, "lo": jit - 0.003, "hi": jit + 0.003, "separated": False},
            "dc_vs_k2_by_family": {"LATERAL": {"LAT_cross_mae_m": row(cross)}}}


def _nums(a=None, b=None):
    return {"0": a or _seed(), "1": b or _seed()}


def _ev(pt, lo, hi):
    return {"point": pt, "lo": lo, "hi": hi}


def _eval(prim=0.47, p010=0.50, w=0.02):
    return {"nearfield": {"ci": {"nf_swedge_prec_lv_5_20": _ev(prim, prim - w, prim + w), "nf_prec_0_10": _ev(p010, p010 - w, p010 + w)}}}


GO, NOGO_A, NOGO_D = ({"verdict": "GO", "failing_clauses": []}, {"verdict": "NO-GO", "failing_clauses": [{"id": "a"}]}, {"verdict": "NO-GO", "failing_clauses": [{"id": "d"}]})


def _comb(nums, nh=None, k2=None, e2=None, other=None, **kw):
    return lever.combine(nums, nh or _eval(), k2 or _eval(), e2, other, **kw)


def test_lever_constants_equal_the_preregs_and_the_dose_tools():
    assert lever.BAR_M == 0.0025 and lever.BAND_M == 0.0010 and lever.LAT_DIST_ROWS == ("LAT_cross_mae_m", "LAT_cross_abs_final_m") and lever.SEEDS == ("0", "1")
    prereg = E2RAW / "PREREG_K2DC02NH_2026-11-20.md"
    if not prereg.is_file():
        pytest.skip("PREREG_K2DC02NH is not next to the kit")
    p = prereg.read_text(encoding="utf-8")
    assert "≤ −2.5 mm" in p and "within ±1 mm" in p and "both seeds 0 and 1" in p and "same constants as the dose rule" in p
    assert "`LAT_cross_mae_m` and `LAT_cross_abs_final_m` both within" in p
    root = _repo_root()
    if root is None:
        pytest.skip("no stack/ above the kit")
    tool = (root / "stack/scripts/r9_e2_dose_verdict.py").read_text(encoding="utf-8")
    assert re.search(r"^BAR_M = 0\.0025\b", tool, re.M) and re.search(r"^BAND_M = 0\.0010\b", tool, re.M)
    assert re.search(r'^LAT_DIST_ROWS = \("LAT_cross_mae_m", "LAT_cross_abs_final_m"\)', tool, re.M)


def test_supported_at_both_seeds():
    r = _comb(_nums(), e2=GO)
    assert r["rule_outcome"] == "SUPPORTED" and r["reading"] == "SUPPORTED" and r["sec4_item"] == 1


@pytest.mark.parametrize("cross,want", [(-0.0025, "SUPPORTED"), (-0.0024, "INCONCLUSIVE"), (-0.0026, "SUPPORTED"), (0.0, "REFUTED"), (0.0010, "REFUTED"), (-0.0010, "REFUTED"), (-0.0011, "INCONCLUSIVE")])
def test_the_bar_and_the_band_are_inclusive_and_exactly_at_the_preregs_numbers(cross, want):
    s = _seed(cross=cross, final=0.0 if want == "REFUTED" else -0.002)
    assert lever.decide({"0": s, "1": copy.deepcopy(s)})["rule_outcome"] == want


def test_both_seeds_are_required_for_every_outcome():
    good, bad = _seed(), _seed(cross=-0.001, final=-0.002)
    assert lever.decide({"0": good, "1": bad})["rule_outcome"] == "INCONCLUSIVE"
    assert lever.decide({"0": bad, "1": good})["rule_outcome"] == "INCONCLUSIVE"
    ref = _seed(cross=0.0, final=0.0)
    assert lever.decide({"0": ref, "1": good})["rule_outcome"] == "INCONCLUSIVE"
    with pytest.raises(lever.Refused):
        lever.decide({"0": good})


def test_mutant_each_supported_clause_alone_breaks_supported():
    assert lever.decide(_nums(_seed(nh_k0_delta=0.003, nh_k0_sep=True), _seed()))["rule_outcome"] == "INCONCLUSIVE"       # nh separated worse than K0
    assert lever.decide(_nums(_seed(nh_k0_delta=0.003, nh_k0_sep=False), _seed(nh_k0_delta=0.003, nh_k0_sep=False)))["rule_outcome"] == "SUPPORTED"     # worse but NOT separated: still not worse
    assert lever.decide(_nums(_seed(nh_k0_delta=-0.003, nh_k0_sep=True), _seed(nh_k0_delta=-0.003, nh_k0_sep=True)))["rule_outcome"] == "SUPPORTED"      # separated BETTER is fine
    assert lever.decide(_nums(_seed(jit=0.0), _seed()))["rule_outcome"] == "INCONCLUSIVE"                                   # jitter delta must be < 0, 0.0 does not clear
    assert lever.decide(_nums(_seed(jit=0.001), _seed(jit=0.001)))["rule_outcome"] == "INCONCLUSIVE"
    assert lever.decide(_nums(_seed(cross=-0.002), _seed(cross=-0.002)))["rule_outcome"] == "INCONCLUSIVE"                  # bar missed


def test_mutant_refuted_needs_both_lateral_distance_rows_inside_one_mm():
    both = _seed(cross=0.0005, final=-0.0009)
    assert lever.decide({"0": both, "1": copy.deepcopy(both)})["rule_outcome"] == "REFUTED"
    out_final = _seed(cross=0.0005, final=0.0012)
    assert lever.decide({"0": out_final, "1": copy.deepcopy(out_final)})["rule_outcome"] == "INCONCLUSIVE"
    out_cross = _seed(cross=0.0012, final=0.0)
    assert lever.decide({"0": out_cross, "1": copy.deepcopy(out_cross)})["rule_outcome"] == "INCONCLUSIVE"
    # REFUTED does not look at the K0 row or the jitter row
    r = _seed(cross=0.0, final=0.0, nh_k0_delta=0.01, nh_k0_sep=True, jit=0.01)
    assert lever.decide({"0": r, "1": copy.deepcopy(r)})["rule_outcome"] == "REFUTED"


def test_mutating_the_bar_constant_flips_the_outcome(monkeypatch):
    s = _seed(cross=-0.003)
    assert lever.decide({"0": s, "1": copy.deepcopy(s)})["rule_outcome"] == "SUPPORTED"
    monkeypatch.setattr(lever, "BAR_M", 0.005)                                                  # the mutation: a bar 2x too strict
    assert lever.decide({"0": s, "1": copy.deepcopy(s)})["rule_outcome"] == "INCONCLUSIVE"
    monkeypatch.setattr(lever, "BAR_M", 0.0025)
    monkeypatch.setattr(lever, "BAND_M", 0.0001)
    z = _seed(cross=0.0005, final=0.0005)
    assert lever.decide({"0": z, "1": copy.deepcopy(z)})["rule_outcome"] == "INCONCLUSIVE"     # a narrower band turns the REFUTED into INCONCLUSIVE


def test_the_lever_rule_equals_the_dose_tools_decide_over_a_grid():
    root = _repo_root()
    if root is None:
        pytest.skip("no stack/ above the kit")
    D = _load(root / "stack/scripts/r9_e2_dose_verdict.py", "k2n_dose_tool")
    assert (D.BAR_M, D.BAND_M, D.EPS) == (lever.BAR_M, lever.BAND_M, lever.EPS)
    vals = (-0.006, -0.0025, -0.0024, -0.001, 0.0, 0.001, 0.0011)
    n = 0
    for c0, c1, f0, f1, j0, w1 in itertools.product(vals, vals, (-0.002, 0.0, 0.0011), (-0.002, 0.0), (-0.001, 0.0, 0.001), (False, True)):
        per = {"0": _seed(cross=c0, final=f0, jit=j0), "1": _seed(cross=c1, final=f1, jit=-0.001, nh_k0_delta=0.002, nh_k0_sep=w1)}
        keys = ("dc_vs_k0", "dc_vs_k2", "jitter_dlag_dc_vs_k2")
        sub = {s: {k: per[s][k] for k in keys} for s in per}
        try:
            ref = D.decide(sub)["outcome_if_valid"]
        except SystemExit:                                                                      # the tool refuses a contradiction; so must the lever rule
            with pytest.raises(lever.Refused):
                lever.decide(sub)
            continue
        assert lever.decide(sub)["rule_outcome"] == ref, (c0, c1, f0, f1, j0, w1)
        n += 1
    assert n > 1000


def test_sec4_readings_are_exactly_the_four_headings():
    assert lever.READINGS == ("SUPPORTED", "REFUTED", "INCONCLUSIVE", "VOID")
    seen = {_comb(_nums(), e2=GO)["reading"], _comb(_nums(_seed(cross=0.0, final=0.0), _seed(cross=0.0, final=0.0)))["reading"],
            _comb(_nums(_seed(cross=-0.001), _seed()))["reading"], _comb(_nums(), tail_void=True)["reading"]}
    assert seen == set(lever.READINGS)


def test_item_1_needs_go_on_either_read_and_item_2_the_tradeoff():
    assert _comb(_nums(), e2=GO)["sec4_item"] == 1
    assert _comb(_nums(), e2=NOGO_D, other=GO)["sec4_item"] == 1                                # GO on the OTHER read counts ("either read")
    r = _comb(_nums(), e2=NOGO_A)                                                               # (a) fails on the read
    assert r["reading"] == "SUPPORTED" and r["sec4_item"] == 2
    r = _comb(_nums(), nh=_eval(p010=0.40), k2=_eval(p010=0.60), e2=NOGO_D)                     # 0-10 m precision: nh interval wholly below K2DC02's
    assert r["sec4_item"] == 2 and r["map_tradeoff"]["nf_prec_0_10"]["separated_worse"] is True
    r = _comb(_nums(), nh=_eval(p010=0.49), k2=_eval(p010=0.50), e2=NOGO_D)                     # overlapping intervals are not "separated"
    assert r["sec4_item"] == "unspecified" and r["map_tradeoff"]["nf_prec_0_10"]["separated_worse"] is False
    assert _comb(_nums(), e2=GO, nh=_eval(p010=0.40), k2=_eval(p010=0.60))["sec4_item"] == 1  # item 1 first: GO outranks the trade-off flag


def test_refuted_inconclusive_and_void_items():
    z = _seed(cross=0.0, final=0.0)
    r = _comb({"0": z, "1": copy.deepcopy(z)})
    assert r["reading"] == "REFUTED" and r["sec4_item"] == 3 and "module-reset" in r["why"]
    r = _comb(_nums(_seed(cross=-0.001), _seed()))
    assert r["reading"] == "INCONCLUSIVE" and r["sec4_item"] == 4 and "replicate" in r["why"]
    for kw, name in (({"tail_void": True}, "TAIL-VOID"), ({"repro_void": True}, "REPRO-VOID"), ({"zerograd_void": True}, "ZEROGRAD-VOID")):
        r = _comb(_nums(), e2=GO, **kw)                                                         # a VOID beats a SUPPORTED that would be GO
        assert r["reading"] == "VOID" and r["sec4_item"] == 5 and name in r["voids"]
    assert _comb({"0": z, "1": copy.deepcopy(z)}, tail_void=True)["reading"] == "VOID"        # and a REFUTED


def test_ema_mode_is_reported_beside_and_never_a_lever_label():
    r = _comb(_nums(), e2=GO, mode="ema")
    assert r["reading"] == "EMA-REPORTED-BESIDE" and r["rule_outcome"] == "SUPPORTED" and "TAIL outcome decides" in r["why"]


def test_wrong_inputs_are_refused():
    d = _nums()
    del d["1"]["dc_vs_k2"]["LAT_cross_abs_final_m"]
    with pytest.raises(lever.Refused):
        _comb(d)
    with pytest.raises(lever.Refused):
        _comb(_nums(), nh={"nearfield": {"ci": {}}})


def test_main_writes_the_reading_and_exit_codes(tmp_path):
    (tmp_path / "nh").mkdir()
    (tmp_path / "k2").mkdir()
    (tmp_path / "nh" / "EVAL_K2.json").write_text(json.dumps(_eval()), encoding="utf-8")
    (tmp_path / "nh" / "E2_VERDICT.json").write_text(json.dumps(GO), encoding="utf-8")
    (tmp_path / "k2" / "EVAL_K2.json").write_text(json.dumps(_eval()), encoding="utf-8")
    (tmp_path / "n.json").write_text(json.dumps(_nums()), encoding="utf-8")
    common = ["--nh-read", str(tmp_path / "nh"), "--k2dc02-read", str(tmp_path / "k2"), "--numbers-json", str(tmp_path / "n.json")]
    assert lever.main(["--mode", "tail", *common, "--out-dir", str(tmp_path / "o1")]) == 0
    j = json.loads((tmp_path / "o1" / "LEVER_READING.json").read_text(encoding="utf-8"))
    md = (tmp_path / "o1" / "LEVER_READING.md").read_text(encoding="utf-8")
    assert j["reading"] == "SUPPORTED" and j["sec4_item"] == 1 and "seeds" in j and "STRATEGIC" in md and "Map trade-off" in md and j["constants"]["bar_m"] == -0.0025 and j["tier"].startswith("T1") and "episode" in j["estimator"]
    assert lever.main(["--mode", "tail", *common, "--out-dir", str(tmp_path / "o2"), "--repro-void"]) == 5
    assert json.loads((tmp_path / "o2" / "LEVER_READING.json").read_text(encoding="utf-8"))["reading"] == "VOID"
    assert lever.main(["--mode", "ema", *common, "--out-dir", str(tmp_path / "o3")]) == 0
    assert lever.main(["--mode", "tail", "--nh-read", str(tmp_path / "none"), "--k2dc02-read", str(tmp_path / "k2"), "--numbers-json", str(tmp_path / "n.json"), "--out-dir", str(tmp_path / "o4")]) == 4


# ---------------------------------------------------------------------------------------------------------------- #
# the reproduction check
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
# the lever-tool guard and the verdict scripts' invocations
# ---------------------------------------------------------------------------------------------------------------- #
TOOL_OK = "BAR_M = 0.0025              # x\nBAND_M = 0.0010             # y\ndef seed_numbers(a):\n    pass\ndef decide(p):\n    pass\n"


def _guard(tmp_path, tool_text, argv_tokens=("--arm", "hier"), with_argv=True):
    c = tmp_path / "code"
    (c / "stack/scripts").mkdir(parents=True)
    (c / "stack/ops/runs.d").mkdir(parents=True)
    (c / "stack/scripts/r9_e2_dose_verdict.py").write_text(tool_text, encoding="utf-8")
    if with_argv:
        (c / "stack/ops/runs.d/r9e2_K2DC02nh.argv.json").write_text(json.dumps({"argv": list(argv_tokens)}), encoding="utf-8")
    script = tmp_path / "g.sh"
    script.write_bytes(f'. "{(DEVBOX / "k2dc02nhtail_common.sh").as_posix()}"\nC="{c.as_posix()}"\nlever_tool_guard\n'.encode())
    return _run(script, env={"K2T_D": (tmp_path / "d").as_posix()})


def test_lever_tool_guard_accepts_and_refuses(tmp_path):
    assert _guard(tmp_path, TOOL_OK).returncode == 0
    for i, (txt, kw, why) in enumerate(((TOOL_OK.replace("0.0025", "0.0030"), {}, "lacks BAR_M"), (TOOL_OK.replace("def decide", "def decide_x"), {}, "lacks BAR_M"),
                                         (TOOL_OK, {"with_argv": False}, "argv.json"), (TOOL_OK, {"argv_tokens": ("--r9-hist-supervise-geom",)}, "that is K2DC02"))):
        d = tmp_path / f"m{i}"
        d.mkdir()
        r = _guard(d, txt, **kw)
        assert r.returncode == 1 and why in r.stdout, (i, r.stdout)


def test_the_kits_default_tip_carries_what_the_guard_needs(tmp_path):
    common = (DEVBOX / "k2dc02nhtail_common.sh").read_text(encoding="utf-8")
    assert f"TIP=${{K2T_TIP:-{SRC_COMMIT}}}" in common
    t = _git_show(f"{SRC_COMMIT}:stack/scripts/r9_e2_dose_verdict.py").decode("utf-8")
    a = _git_show(f"{SRC_COMMIT}:stack/ops/runs.d/r9e2_K2DC02nh.argv.json").decode("utf-8")
    r = _guard(tmp_path, t, argv_tokens=tuple(json.loads(a)["argv"]))
    assert r.returncode == 0, (r.stdout, r.stderr)


def test_verdict_script_invokes_the_lever_rule_with_the_right_slots():
    v = _code("devbox/k2dc02nhtail_verdicts.sh")
    assert "r9_e2_dose_verdict.py" not in v and "--arm K2DC02" not in v and "k2dc02nhtail_combined" not in v and "floor_verdict" not in v
    assert '--mode tail --nh-pf "$(W2 "$PFNH")" --k2dc02-pf "$(W2 "$PFK2DC02")" --k0-pf "$(W2 "$PFK0")"' in v                    # K2DC02nht vs K2DC02t (K2's slot) and K0t
    assert '--nh-read "$(W2 "$RD")" --k2dc02-read "$(W2 "$C/$K2DC02TREAD")"' in v and '--out-dir "$(W2 "$V/lever")"' in v
    assert "PFNH=$MS/E2_K2_pf; PFK0=$P/pin/cmp/E2_K0_pf; PFK2DC02=$P/pin/stageB/E2_K2_pf" in v and "K2DCt" not in v
    assert "--tail-void" in v and "--repro-void" in v and "--zerograd-void" in v and "LEVERTOOL" in v and v.index("LEVERTOOL") < v.index("mkdir -p \"$V/repro")
    assert "--other-e2-verdict" in v and "K2T_EMA_E2V" in v
    assert '"K2DC02nht:0=' in v and '"K2DC02t:0=' in v and '"K0t:0=' in v and "--ref K0t" in v                              # STRATEGIC arms, reported in full
    assert "k2dc02nhtail_sanitize_navcomp.py" in v and "REGRESSION-RED" in v and "IDCTL-PASS-TENSOR" in v and "ZEROGRAD_FAIL.json" in v
    assert 'for pair in "pinned:$PFK0" "reroll:$PFREPRO"' in v and "S=0" in v and '--a-dump "$(W2 "$PFK2DC02/dump_s$S")"' in v           # reproduction at inference seed 0 only, anchored on K2DC02t


def test_ema_verdict_script_reads_what_the_ema_launcher_produces():
    e = _code("devbox/k2dc02nhtail_ema_verdicts.sh")
    assert "--arm K2DC02" not in e and "r9_e2_dose_verdict.py" not in e
    assert "T_NH=$RUNS/E2_READ_K2DC02nh" in e and "T_K2DC02=$RUNS/E2_READ_K2DC02" in e
    assert '--mode ema --nh-pf "$(W2 "$P/nh/E2_K2_pf")" --k2dc02-pf "$(W2 "$P/k2dc02/E2_K2_pf")" --k0-pf "$(W2 "$P/nh/E2_K0_pf")"' in e
    assert '--nh-read "$(W2 "$RD_NH")" --k2dc02-read "$(W2 "$RD_K2DC02")"' in e and "LEVERTOOL" in e
    launcher = _src_launcher()
    assert "runs/E2_READ_$ARM" in launcher or "E2_READ_$ARM" in launcher


def _src_launcher():
    root = _repo_root()
    if root is not None and (root / "stack/ops/r9e2_read/e2read_launch_k2dc02nh.sh").is_file():
        return (root / "stack/ops/r9e2_read/e2read_launch_k2dc02nh.sh").read_text(encoding="utf-8")
    return _git_show(f"{SRC_COMMIT}:stack/ops/r9e2_read/e2read_launch_k2dc02nh.sh").decode("utf-8")


def test_s6c_step_checks_the_pin_before_pulling_and_pulls_only_the_two_comparators():
    s = _t("devbox/k2dc02nhtail_s6c_pull_score.sh")
    assert s.index("THORPIN") < s.index("pull stageB") and s.index("pull repro") < s.index("PINSHA") < s.index("PINPULLSHA")
    assert "$PIN_SHA" in s and "TAILGUARD-MD5" in s and "tail_md5.txt" in s
    assert "cmp_k2dc" not in s and "cmp/E2_K0_pf stageB/E2_K2_pf" in s and "cmp_reroll" not in s and "CMP_STATUS" not in s


# ---------------------------------------------------------------------------------------------------------------- #
# the seed guard counts INSIDE argv / argv_effective (MM 4a3d05, 2026-11-22). The 1f14f8d8 trainer writes the argv twice in config.json
# (argv + argv_effective, equal), so the old whole-file count read flag x2 / pair x2 and refused the valid K2DC02s1 tail read at
# 17:12:39Z (ABORT-SEED). Same defect and same form as the EMA read launchers' fix (8b581767).
# ---------------------------------------------------------------------------------------------------------------- #
SEED_ARM, SEED_VAL = "K2DC02nh", "0"
OLD_SEED_GUARD = 'NS=$(printf \'%s\' "$CFG1" | grep -o \'"--seed"\' | wc -l); NV=$(printf \'%s\' "$CFG1" | grep -o \'"--seed","0"\' | wc -l); [ "$NS" = 1 ] && [ "$NV" = 1 ] || die SEED "config.json must carry --seed 0 exactly once (flag x$NS, pair x$NV): not the K2DC02nh argv"'


def _seed_guard_lines(text):
    ls = [l for l in text.split("\n") if l.startswith("CFG1=$(tr -d") or l.startswith("NS=$(printf")]
    assert len(ls) == 2 and ls[0].startswith("CFG1="), ls
    return ls


def _run_seed_guard(tmp_path, cfg, lines=None):
    lines = lines or _seed_guard_lines(_t("scripts/k2dc02nhtail_build.sh"))
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
        body = [SEED_ARM + "=" + run.as_posix(), 'die() { echo "ABORT-$1 ${*:2}"; exit 3; }'] + _seed_guard_lines(_t("scripts/k2dc02nhtail_build.sh")) + ["echo GUARD-OK"]
        sh.write_bytes(("\n".join(body) + "\n").encode("utf-8"))
        r = _run(sh)
    else:
        r = _run_seed_guard(tmp_path, cfg)
    assert r.returncode == 3 and "ABORT-SEED" in r.stdout and "GUARD-OK" not in r.stdout, (cfg_kind, r.stdout + r.stderr)


def test_the_old_whole_file_seed_guard_refuses_the_real_layout(tmp_path):
    """the deliberate-regression arm: the pre-fix line, run on the trainer's real layout, must reproduce the 17:12:39Z refusal"""
    cfg_line = _seed_guard_lines(_t("scripts/k2dc02nhtail_build.sh"))[0]
    r = _run_seed_guard(tmp_path, {"arm": SEED_ARM, "argv": _argv(), "argv_effective": _argv()}, lines=[cfg_line, OLD_SEED_GUARD])
    assert r.returncode == 3 and "ABORT-SEED" in r.stdout and "flag x2" in r.stdout, r.stdout + r.stderr
    assert OLD_SEED_GUARD not in _t("scripts/k2dc02nhtail_build.sh")
