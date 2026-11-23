"""Tests of make_2x_script.py and of the generated pod_dscale_lead_2x.sh (card PREREG_V8F_PROOF_ADD123). CPU only; no GPU, no pod.
The banked source is SRC_SH (default: ../../pod_dscale_lead_run.sh). The stages are exercised (a) with DRY=1 (every command printed, no file created) and (b) for real against fakes of the
interpreter and of flock (the smoke bar, the time limit, the arguments of the training and extract commands, the done file).

Run:  PYCAP_MB=3000 bash F:/TanitAD-scratch/v8proof/ops/pycap.sh -m pytest test_2x_script.py -q -p no:cacheprovider --basetemp <NEW folder>
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import pytest

HERE = Path(__file__).resolve().parent
SRC_SH = Path(os.environ.get("SRC_SH", str(HERE.parent.parent / "pod_dscale_lead_run.sh")))
GEN_SH = HERE / "pod_dscale_lead_2x.sh"
BASH = shutil.which("bash")
REALPY = os.environ.get("REALPY", "/c/Users/Admin/venvs/tanitad/Scripts/python.exe")   # on a pod: its own interpreter
ARMS = ["TK16_s0", "TK16_s1", "TK16_s2", "TK16_s3"]
BANNED = ["vo" + "id", "super" + "seded", "retr" + "acted", "for" + "merly", "corr" + "ection", "with" + "drawn", "no lon" + "ger", "closed " + "loop"]


def load_gen():
    spec = importlib.util.spec_from_file_location("make_2x_script_under_test", str(HERE / "make_2x_script.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["make_2x_script_under_test"] = m
    spec.loader.exec_module(m)
    return m


G = load_gen()


def src_text():
    return SRC_SH.read_bytes().decode("utf-8")


def must_fail(fn):
    try:
        fn()
    except AssertionError:
        return
    raise AssertionError("a planted wrong statement was accepted: the check is blind")


def _assert(x):
    assert x


# ------------------------------------------------------------------------------------------------ the generator
@pytest.mark.parametrize("i", range(len(G.REPLACES)))
def test_generator_stops_when_an_anchor_is_missing_or_doubled(i):
    old = G.REPLACES[i][0]
    text = src_text()
    assert text.count(old) == 1
    with pytest.raises(SystemExit) as e:
        G.apply_replaces(text.replace(old, "", 1))                              # missing
    assert e.value.code == 3
    with pytest.raises(SystemExit) as e:
        G.apply_replaces(text + "\n" + old)                                      # doubled
    assert e.value.code == 3


def test_generator_output_is_the_text_on_disk_and_nothing_is_overwritten(tmp_path):
    want = G.build_text(src_text())
    assert GEN_SH.read_bytes().decode("utf-8") == want
    assert "\r" not in want
    out = tmp_path / "x2.sh"
    assert G.main(["--src", str(SRC_SH), "--out", str(out)]) == 0 and out.read_text(encoding="utf-8") == want
    assert G.main(["--src", str(SRC_SH), "--out", str(out)]) == 3              # an existing output is never overwritten
    with pytest.raises(SystemExit):
        G.build_text(src_text().replace("\n", "\r\n"))
    must_fail(lambda: _assert(want == src_text()))


def test_generated_text_holds_the_card_points_and_none_of_the_banned_ones():
    t = GEN_SH.read_text(encoding="utf-8")
    assert t.count('FOLDERS="runQ"') == 1
    assert "--ck-steps 19164,28746" in t and "tok_probe_u_ck.py" in t
    assert 'timeout "$TRAIN_LIMIT_S"' in t and "TRAIN_LIMIT_S:-3300" in t and "TRAIN-ABOVE-TIME-LIMIT" in t
    assert "MAX_H:-0.9" in t and "ESTIMATE-ABOVE-0.9H" in t and "(s/n)*b/3600" in t
    assert "DSCALE_2X_DONE.txt" in t and "DSCALE_2X_STOP.txt" in t and "dscale_2x" in t and "TREE_PINS_2X.txt" in t
    assert "ckpt_step${tag#s}.pt" in t and "ckpt_step19164" in t and "ckpt_step28746" in t and "last s19164 s28746" in t and "_s19164" in t and "_s28746" in t
    assert "DSCALE_LEAD_" not in t and "ESTIMATE-ABOVE-6H" not in t and not re.search(r"run(C|U|U2|CSH|USH)\b", t)
    assert not re.search(r"(^|[;&|(]|\s)(rm|mv)\s", t.replace("\n", " \n "))
    low = t.lower()
    for w in BANNED:
        assert w not in low
    for f in HERE.glob("*"):
        if f.suffix in (".py", ".sh", ".md", ".txt") and f.name not in ("helper_pytest.out",):
            ll = f.read_text(encoding="utf-8", errors="replace").lower()
            for w in BANNED:
                assert w not in ll, (f.name, w)
    must_fail(lambda: _assert("rm -rf" in t or "DSCALE_LEAD_" in t))


def test_bash_syntax_is_valid():
    assert BASH, "no bash found"
    r = subprocess.run([BASH, "-n", GEN_SH.as_posix()], capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    bad = subprocess.run([BASH, "-n", "-c", "if then fi ]]"], capture_output=True, text=True, encoding="utf-8")
    assert bad.returncode != 0


# ------------------------------------------------------------------------------------------------ DRY runs
def base_env(tmp, kit=None, out=None):
    env = dict(os.environ)
    env.update({"OUT": Path(out).as_posix() if out else (tmp / "out_dry").as_posix(), "TREE": (tmp / "tree").as_posix(), "P10": (tmp / "p10.pt").as_posix(), "TRACKS": (tmp / "tracks.pt").as_posix(),
                "PY": "python-not-run", "DRY": "1"})
    if kit:
        env["KIT"] = Path(kit).as_posix()
    return env


def sh(script, stage, env, cwd):
    return subprocess.run([BASH, Path(script).as_posix(), stage], env=env, cwd=str(cwd), capture_output=True, text=True, encoding="utf-8")


def snapshot(root):
    return sorted(str(p.relative_to(root)) for p in Path(root).rglob("*"))


@pytest.mark.parametrize("stage", ["precheck", "cpucheck", "smoke", "run", "extract", "checklog"])
def test_dry_run_prints_commands_and_creates_no_file(stage, tmp_path):
    env = base_env(tmp_path)
    before = snapshot(tmp_path)
    r = sh(GEN_SH, stage, env, tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    r.stdout = r.stdout.replace("\\", "")                                          # printf %q escapes commas and spaces
    assert "DRY:" in r.stdout
    assert not Path(env["OUT"]).exists() and snapshot(tmp_path) == before
    if stage == "smoke":
        assert "x 38328 steps" in r.stdout and "0.9" in r.stdout and "TK16_s3" in r.stdout
    if stage == "run":
        assert "tok_probe_u_ck.py" in r.stdout and "--ck-steps 19164,28746" in r.stdout and "timeout 3300" in r.stdout
        assert "--arms TK16_s0,TK16_s1,TK16_s2,TK16_s3" in r.stdout and "--steps 38328" in r.stdout and "--expect-steps 38328" in r.stdout
        assert "--clip-list" not in r.stdout.split("flock", 1)[1] and "--shuffle-gap-labels" not in r.stdout
        assert r.stdout.count("flock") == 1
    if stage == "extract":
        assert r.stdout.count("flock") == 12
        for n in ("19164", "28746"):
            assert "ckpt_step%s.pt" % n in r.stdout and "states/runQ_s%s" % n in r.stdout
        assert "states/runQ " in r.stdout or "states/runQ'" in r.stdout or r.stdout.count("--run-dir") == 4
    r2 = sh(GEN_SH, stage, {**env, "OUT": ""}, tmp_path)                           # a missing OUT is refused, nothing runs
    assert r2.returncode == 3
    must_fail(lambda: _assert(re.search(r"(^|\s)(rm|mv)\s", r.stdout) is None and "DRY:" not in r.stdout))
    assert re.search(r"(^|\s)(rm|mv)\s", r.stdout) is None


# ------------------------------------------------------------------------------------------------ real runs against fakes
FAKE_PY = r"""#!/usr/bin/env bash
case "$1" in -c|-) exec "$REALPY" "$@";; esac
echo "$*" >> "$FAKE_LOG"
case "$1" in *dscale_evalview.py) exit 0;; esac
out=""; prev=""
for a in "$@"; do [ "$prev" = --out ] && out=$a; prev=$a; done
case "$2" in
  train)
    [ -n "${FAKE_SLEEP:-}" ] && sleep "$FAKE_SLEEP"
    mkdir -p "$out"
    echo "{\"seconds\": ${FAKE_SECONDS:-10}}" > "$out/TRAIN_DONE.json"
    case "$1" in *tok_probe_u_ck.py)
      echo '{}' > "$out/CK_RECORD.json"
      for arm in TK16_s0 TK16_s1 TK16_s2 TK16_s3; do mkdir -p "$out/$arm"; for c in ckpt_last ckpt_step19164 ckpt_step28746; do echo "$arm $c" > "$out/$arm/$c.pt"; done; done;;
    esac
    exit 0;;
  check-log) echo '{}'; exit 0;;
esac
exit 0
"""
FAKE_FLOCK = "#!/usr/bin/env bash\nshift\nexec \"$@\"\n"
TREE_FILES = ["stack/scripts/v8f_mini_train.py", "stack/tanitad/data/v8f_mini_data.py", "stack/tanitad/models/v8f_mini.py"]
KIT_FILES = ["tok_probe_u.py", "tok_probe_u_ck.py", "dscale_precheck.py", "dscale_clockcheck.py", "dscale_tokens_same.py", "STUDY_EVAL_MD5.tsv", "STUDY_TRAIN_MD5.tsv", "dscale_evalview.py"]


def world(tmp):
    kit, tree, bin_ = tmp / "kit", tmp / "tree", tmp / "bin"
    for d in (kit, bin_, tmp / "cache" / "train"):
        d.mkdir(parents=True)
    shutil.copy(GEN_SH, kit / GEN_SH.name)
    lines = []
    for rel in TREE_FILES:
        p = tree / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("tree " + rel + "\n", encoding="utf-8")
        lines.append("%s tree:%s" % (hashlib.sha256(p.read_bytes()).hexdigest(), rel))
    for rel in KIT_FILES:
        (kit / rel).write_text("kit " + rel + "\n", encoding="utf-8")
        lines.append("%s kit:%s" % (hashlib.sha256((kit / rel).read_bytes()).hexdigest(), rel))
    (tmp / "pins.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    (tmp / "cache" / "train" / "manifest.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp / "cg_limit").write_text(str(100 * 1073741824), encoding="utf-8")
    (tmp / "cg_stat").write_text("total_rss 0\n", encoding="utf-8")
    (bin_ / "fakepy").write_text(FAKE_PY, encoding="utf-8", newline="\n")
    (bin_ / "flock").write_text(FAKE_FLOCK, encoding="utf-8", newline="\n")
    for _f in ("fakepy", "flock"):                                             # on Linux a stand-in program needs its run bit (the lead test run is on a pod)
        os.chmod(str(bin_ / _f), 0o755)
    return kit


def real_env(tmp, kit, out, **extra):
    env = dict(os.environ)
    env.update({"KIT": kit.as_posix(), "OUT": out.as_posix(), "TREE": (tmp / "tree").as_posix(), "P10": "p10", "TRACKS": "tracks", "PY": (tmp / "bin" / "fakepy").as_posix(), "REALPY": REALPY,
                "FAKE_LOG": (tmp / ("fake_%s.log" % out.name)).as_posix(), "PINS": (tmp / "pins.txt").as_posix(), "CACHE": (tmp / "cache").as_posix(), "CG_LIMIT": (tmp / "cg_limit").as_posix(),
                "CG_STAT": (tmp / "cg_stat").as_posix(), "PATH": str(tmp / "bin") + os.pathsep + env.get("PATH", ""), "DRY": "0", "LOCK": (tmp / "lock").as_posix()})
    env.update(extra)
    return env


def prime(out, *markers, eval_manifest=True):
    (out / "markers").mkdir(parents=True)
    for m in markers:
        (out / "markers" / (m + ".done")).write_text("t\n", encoding="utf-8")
    if eval_manifest:
        (out / "eval_view" / "eval").mkdir(parents=True)
        (out / "eval_view" / "eval" / "manifest.jsonl").write_text('{"clip": "c1"}\n{"clip": "c2"}\n', encoding="utf-8")


def test_smoke_above_the_bar_stops_with_exit_4_and_below_it_passes(tmp_path):
    kit = world(tmp_path)
    script = kit / GEN_SH.name
    over, under = tmp_path / "out_over", tmp_path / "out_under"
    prime(over, "cpucheck")
    prime(under, "cpucheck")
    r = sh(script, "smoke", real_env(tmp_path, kit, over, FAKE_SECONDS="100"), tmp_path)           # 100 s / 200 x 38328 / 3600 = 5.32 h
    assert r.returncode == 4, r.stdout + r.stderr
    stop = (over / "DSCALE_2X_STOP.txt").read_text(encoding="utf-8")
    assert "ESTIMATE-ABOVE-0.9H" in stop and not (over / "markers" / "smoke.done").exists()
    assert json.loads((over / "SMOKE_ESTIMATE.json").read_text(encoding="utf-8"))["projected_gpu_h_upper_bound"] == pytest.approx(5.32, abs=0.01)
    r = sh(script, "smoke", real_env(tmp_path, kit, under, FAKE_SECONDS="10"), tmp_path)           # 0.53 h
    assert r.returncode == 0, r.stdout + r.stderr
    assert (under / "markers" / "smoke.done").exists() and not (under / "DSCALE_2X_STOP.txt").exists()
    assert json.loads((under / "SMOKE_ESTIMATE.json").read_text(encoding="utf-8"))["projected_gpu_h_upper_bound"] == pytest.approx(0.53, abs=0.01)
    assert (under / "dscale_2x.log").exists() and not (under / "dscale_lead.log").exists()
    # the smoke trains four arms of the new folder with the BANKED tool
    log = Path(real_env(tmp_path, kit, under)["FAKE_LOG"]).read_text(encoding="utf-8")
    assert "tok_probe_u.py train" in log and "--arms TK16_s0,TK16_s1,TK16_s2,TK16_s3" in log and "--steps 200" in log and "tok_probe_u_ck" not in log
    # a second start on a finished stop file is refused
    r = sh(script, "smoke", real_env(tmp_path, kit, over, FAKE_SECONDS="10"), tmp_path)
    assert r.returncode == 3
    must_fail(lambda: _assert((over / "markers" / "smoke.done").exists()))


def test_the_time_limit_stops_the_run_with_its_reason(tmp_path):
    kit = world(tmp_path)
    out = tmp_path / "out_slow"
    prime(out, "cpucheck", "smoke")
    r = sh(kit / GEN_SH.name, "run", real_env(tmp_path, kit, out, TRAIN_LIMIT_S="1", FAKE_SLEEP="6"), tmp_path)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "TRAIN-ABOVE-TIME-LIMIT" in (out / "DSCALE_2X_STOP.txt").read_text(encoding="utf-8")
    assert not (out / "markers" / "train_runQ.done").exists() and not (out / "runQ" / "TRAIN_DONE.json").exists()


def test_run_extract_checklog_end_to_end_on_fakes(tmp_path):
    kit = world(tmp_path)
    script = kit / GEN_SH.name
    out = tmp_path / "out_ok"
    prime(out, "cpucheck", "smoke")
    env = real_env(tmp_path, kit, out)
    r = sh(script, "run", env, tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (out / "markers" / "train_runQ.done").exists() and (out / "runQ" / "CK_RECORD.json").exists()
    log = Path(env["FAKE_LOG"]).read_text(encoding="utf-8").splitlines()
    train = [x for x in log if "train" in x.split(" ")[1:2]]
    assert len(train) == 1 and "tok_probe_u_ck.py" in train[0] and "--ck-steps 19164,28746" in train[0] and "--steps 38328" in train[0] and "--expect-steps 38328" in train[0]
    assert "--clip-list" not in train[0] and "--shuffle-gap-labels" not in train[0] and "--arms TK16_s0,TK16_s1,TK16_s2,TK16_s3" in train[0]
    r = sh(script, "extract", env, tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    ex = [x for x in Path(env["FAKE_LOG"]).read_text(encoding="utf-8").splitlines() if x.split(" ")[1:2] == ["extract"]]
    assert len(ex) == 12
    for arm in ARMS:
        mine = [x for x in ex if "--arm %s " % arm in x]
        assert len(mine) == 3
        assert sum(1 for x in mine if "--run-dir" in x and "--states-dir %s/states/runQ " % out.as_posix() in x) == 1
        assert sum(1 for x in mine if "%s/runQ/%s/ckpt_step19164.pt" % (out.as_posix(), arm) in x and "states/runQ_s19164 " in x) == 1
        assert sum(1 for x in mine if "%s/runQ/%s/ckpt_step28746.pt" % (out.as_posix(), arm) in x and "states/runQ_s28746 " in x) == 1
    r = sh(script, "checklog", env, tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    done = (out / "DSCALE_2X_DONE.txt").read_text(encoding="utf-8").splitlines()
    assert done[0].startswith("DSCALE_2X_DONE ") and not (out / "DSCALE_LEAD_DONE.txt").exists()
    assert any(x.startswith("tok_probe_u.py sha256: ") for x in done) and any(x.startswith("tok_probe_u_ck.py sha256: ") for x in done)
    ck = [x for x in done if x.startswith("ckpt ")]
    assert len(ck) == 12
    for arm in ARMS:
        for c in ("ckpt_last", "ckpt_step19164", "ckpt_step28746"):
            want = hashlib.sha256((out / "runQ" / arm / (c + ".pt")).read_bytes()).hexdigest()
            assert "ckpt runQ/%s/%s %s" % (arm, c, want) in ck
    r = sh(script, "checklog", env, tmp_path)                                          # a finished folder is never overwritten
    assert r.returncode == 3
    must_fail(lambda: _assert(len(ck) == 4))
