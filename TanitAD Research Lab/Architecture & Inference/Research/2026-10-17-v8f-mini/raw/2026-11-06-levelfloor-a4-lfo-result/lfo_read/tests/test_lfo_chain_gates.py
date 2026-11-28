"""test_lfo_chain_gates.py -- the pre-check (lfo_precheck.py) and the chain's gates for arm LFO075: floor-key layouts and CEILING labels in the rollout check, absent / mismatching checkpoints, the dry-run (4 rollouts in priority order, no
marker written), the chain's refusals on a COPY of the package (pins, patched-tool pin), and the marker discipline (a chain that dies leaves LFO.fail). Env overrides: LFO_PRECHECK_PATH, LFO_CHAIN_PATH."""
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
BASH = "C:/Program Files/Git/bin/bash.exe"
CHAIN = os.environ.get("LFO_CHAIN_PATH", f"{PK}/lfo_chain.sh")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


P = _load("lfo_precheck_under_test", os.environ.get("LFO_PRECHECK_PATH", f"{PK}/code/lfo_precheck.py"))
LLR0 = "E:/Projects/TanitAD-artifacts/v8f-recover/M2b_s0_recft_LLR/ckpt_last.pt"


def test_precheck_real_checkpoints_pass():
    r = P.precheck()
    assert r["PASS"] is True and not r["absent"] and not r["failures"]


def test_absent_checkpoint_is_reported_absent_not_failed(tmp_path):
    r = P.precheck(rdb=str(tmp_path))
    assert r["PASS"] is False and len(r["absent"]) == 4


def test_checkpoint_sha_mismatch_fails(tmp_path):
    for s in (0, 1):
        for run in ("C0", "LLR"):
            d = tmp_path / f"M2b_s{s}_recft_{run}"
            os.makedirs(d)
            (d / "ckpt_last.pt").write_bytes(b"not the banked checkpoint")
            (d / "SHA256SUMS.txt").write_text(hashlib.sha256(b"not the banked checkpoint").hexdigest() + " *ckpt_last.pt\n", encoding="utf-8")
    r = P.precheck(rdb=str(tmp_path))
    assert r["PASS"] is False and not r["absent"] and len(r["failures"]) >= 4


def _mk(tmp_path, kap=0.75, sha=None, src="logged", label="CEILING", flags="--smooth sg72 --scene reindexed --level-floor on --floor-kappa 0.75 --floor-ckpt X --floor-vhat-source logged", vis="NO. CEILING", c0=None):
    d = tmp_path / "out" / "RILFO075_s0"
    os.makedirs(d, exist_ok=True)
    hdr = {"level_floor": "on", "floor_kappa": kap, "floor_ckpt": LLR0, "floor_ckpt_sha256": sha or P.llr_sha_of(0), "floor_vhat_source": src, "level_floor_detail": {"vision_only": vis}}
    if label:
        hdr["label"] = label
    json.dump({"t1v8f": hdr}, open(d / "t1v8f_RILFO075_s0.json", "w", encoding="utf-8"))
    json.dump({"ckpt_sha256": c0 or P.C0_SHA[0], "flags": flags}, open(tmp_path / "out" / "done_RILFO075_s0.txt", "w", encoding="utf-8"))
    return P.check_rollout(str(tmp_path / "out"), "RILFO075_s0")[0]


def test_rollout_check_refuses_each_deviation_and_demands_the_ceiling_labels(tmp_path):
    assert all(_mk(tmp_path).values())
    assert _mk(tmp_path, kap=0.5)["floor_kappa_is_0_75"] is False
    assert _mk(tmp_path, sha="0" * 64)["floor_ckpt_sha256_is_the_LLR_sha"] is False
    assert _mk(tmp_path, src="heads")["header_floor_vhat_source_is_logged"] is False
    assert _mk(tmp_path, label=None)["header_label_is_CEILING"] is False
    assert _mk(tmp_path, vis="v_hat, presence and gap_hat come from the floor model")["detail_says_not_vision_only"] is False
    assert _mk(tmp_path, flags="--smooth sg72 --scene reindexed --level-floor on --floor-kappa 0.75 --floor-ckpt X")["done_flags_name_the_logged_source"] is False
    assert _mk(tmp_path, c0="0" * 64)["planner_ckpt_sha256_is_C0"] is False
    with pytest.raises(ValueError):
        P.parse_tag("RILF075_s0")                                                                  # an LF tag is not an LFO tag
    with pytest.raises(ValueError):
        P.parse_tag("RILFO050_s0")


def test_dry_run_lists_four_rollouts_in_priority_order_and_writes_no_marker():
    before = sorted(os.listdir(f"{PK}/raw"))
    r = subprocess.run([BASH, CHAIN, "--dry-run"], capture_output=True, text=True, cwd=PK)
    assert r.returncode == 0, r.stderr
    rows = [l.split("\t") for l in r.stdout.splitlines() if l.count("\t") == 3]
    assert rows == [["RILFO075_s0", "0", "reindexed", "off"], ["RILFO075_s1", "1", "reindexed", "off"], ["RILFO075C_s0", "0", "reindexed", "kbox"], ["RILFO075C_s1", "1", "reindexed", "kbox"]]
    assert sorted(os.listdir(f"{PK}/raw")) == before
    assert "card sha256 8bfbc386 (want 8bfbc386)" in r.stdout and "--floor-vhat-source logged" in r.stdout


def _copy_pkg(tmp):
    d = tmp / "pk"
    for sub in ("code", "tests"):
        shutil.copytree(f"{PK}/{sub}", d / sub, ignore=shutil.ignore_patterns("__pycache__"))
    for f in ("lfo_chain.sh", "lfo_chain.cmd"):
        shutil.copy2(CHAIN if f.endswith(".sh") else f"{PK}/{f}", d / f)
    for sub in ("logs", "raw"):
        os.makedirs(d / sub)
    lines = []
    for root, _, files in os.walk(d):
        for f in files:
            p = os.path.join(root, f).replace("\\", "/")
            rel = p[len(str(d).replace("\\", "/")) + 1:]
            if rel.startswith(("code/", "tests/")) or rel.startswith("lfo_chain"):
                lines.append(f"{rel} sha256 {hashlib.sha256(open(p, 'rb').read()).hexdigest()}")
    mc = "E:/Projects/TanitAD-artifacts/dagland/motion_c"
    for k, f in (("a4_card", f"{mc}/a4_prereg/PREREG_LEVELFLOOR_A4.md"), ("lf_read_py", f"{mc}/lf_read/code/lf_read.py"), ("lf_precheck_py", f"{mc}/lf_read/code/lf_precheck.py"),
                 ("llr_read_py", f"{mc}/llr_read/code/llr_read.py"), ("lf_read_json", f"{mc}/lf_read/raw/LF_READ.json")):
        lines.append(f"{k} sha256 {hashlib.sha256(open(f, 'rb').read()).hexdigest()}")
    (d / "PINS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return d


def _run_chain(d, bp, extra_env=None):
    env = dict(os.environ, LFO_PK=str(d).replace("\\", "/"), LFO_PKW=str(d).replace("\\", "/"), LFO_BUILD_PINS_FILE=bp, **(extra_env or {}))
    r = subprocess.run([BASH, str(d / "lfo_chain.sh")], capture_output=True, text=True, env=env, cwd=str(d))
    fail = open(d / "raw" / "LFO.fail", encoding="utf-8").read() if (d / "raw" / "LFO.fail").exists() else ""
    return r.returncode, fail


def test_chain_refuses_without_the_patched_tool_pin(tmp_path):
    d = _copy_pkg(tmp_path)
    rc, fail = _run_chain(d, str(tmp_path / "absent_PINS.txt"))
    assert rc == 2 and "lfo_build/PINS.txt is absent" in fail and "step 'gate0'" in fail
    (tmp_path / "keyless.txt").write_text("# nothing\n", encoding="utf-8")
    rc, fail = _run_chain(d, str(tmp_path / "keyless.txt").replace("\\", "/"))
    assert rc == 2 and "no patched_tool key" in fail


def test_chain_refuses_a_tool_that_differs_from_the_pinned_patched_tool(tmp_path):
    tool = "F:/TanitAD-scratch/motion_c/lfo_read/tree/stack/scripts/v8f_t1_eval_ri.py"
    if not os.path.isfile(tool):
        pytest.skip("the LFO tree's tool is not built yet")
    d = _copy_pkg(tmp_path)
    (tmp_path / "wrong.txt").write_text("patched_tool=" + "c" * 64 + "\nlevel_floor=" + "d" * 64 + "\n", encoding="utf-8")
    rc, fail = _run_chain(d, str(tmp_path / "wrong.txt").replace("\\", "/"))
    assert rc == 2 and "tree tool sha" in fail and "patched_tool" in fail


def test_chain_refuses_an_operator_that_differs_from_its_pin(tmp_path):
    tool = "F:/TanitAD-scratch/motion_c/lfo_read/tree/stack/scripts/v8f_t1_eval_ri.py"
    if not os.path.isfile(tool):
        pytest.skip("the LFO tree's tool is not built yet")
    d = _copy_pkg(tmp_path)
    good = hashlib.sha256(open(tool, "rb").read()).hexdigest()
    (tmp_path / "wrongop.txt").write_text(f"patched_tool={good}\nlevel_floor=" + "d" * 64 + "\n", encoding="utf-8")
    rc, fail = _run_chain(d, str(tmp_path / "wrongop.txt").replace("\\", "/"))
    assert rc == 2 and "tree level_floor.py sha" in fail


def test_chain_refuses_a_pinned_file_that_differs(tmp_path):
    d = _copy_pkg(tmp_path)
    (d / "code" / "lfo_read.py").write_text((d / "code" / "lfo_read.py").read_text(encoding="utf-8") + "\n# edited after pinning\n", encoding="utf-8", newline="\n")
    rc, fail = _run_chain(d, str(tmp_path / "x.txt"))
    assert rc == 2 and "code/lfo_read.py sha" in fail and "PINS.txt" in fail


def test_a_chain_that_dies_leaves_a_fail_marker_naming_the_step(tmp_path):
    """The EXIT trap: a TERM during a gate leaves raw/LFO.fail with the step; the running marker says exited."""
    d = _copy_pkg(tmp_path)
    sh = (d / "lfo_chain.sh").read_text(encoding="utf-8")
    sh = sh.replace('STEP=gate0\n', 'STEP=gate0\nkill -TERM $$\nsleep 5\n', 1)               # the chain signals itself at the start of gate 0
    (d / "lfo_chain.sh").write_text(sh, encoding="utf-8", newline="\n")
    rc, fail = _run_chain(d, str(tmp_path / "x.txt"))
    assert rc == 143 and "step 'gate0'" in fail and "exited without a done marker" in fail
    assert open(d / "raw" / "LFO.running", encoding="utf-8").read().startswith("exited")


def test_the_chain_names_the_logged_source_in_every_rollout_argv_and_never_reads_logged_or_hold_now():
    sh = open(CHAIN, encoding="utf-8").read()
    assert '--floor-vhat-source logged "$@" < /dev/null' in sh and 'local ffl="--level-floor on --floor-kappa $KAPPA --floor-ckpt $RDBW/M2b_s${s}_recft_LLR/ckpt_last.pt --floor-vhat-source logged"' in sh
    assert "--scene reindexed" in sh and "--scene logged" not in sh and "hold_now\" ]" not in sh
    assert 'printf \'%s\\t%s\\t%s\\t%s\\n\' "$1" "$2" "$3" "$4"' in sh                                  # four rollouts only
    assert "trap on_exit EXIT" in sh and "trap 'exit 143' TERM INT HUP" in sh
