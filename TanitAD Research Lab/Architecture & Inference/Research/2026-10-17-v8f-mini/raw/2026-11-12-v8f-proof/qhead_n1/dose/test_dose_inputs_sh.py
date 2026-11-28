"""Tests of pod_qhead_dose_inputs.sh: DRY=1 prints every command in order and creates nothing; fail-closed refusals and markers; the slot wait (GPU count, container memory); a full run against
stub tools; no process-name test, no rm, no mv; no banked file is touched."""
import os
import re
import subprocess

import pytest

import dose_testlib as TL  # noqa: E402

BASH = TL.BASH
SCRIPT = os.path.join(TL.SRC, "pod_qhead_dose_inputs.sh")
pytestmark = pytest.mark.skipif(BASH is None, reason="no bash on this box")

FAKE_PY = r"""#!/usr/bin/env bash
# stub python: acts on the tool named by $1; --out value written as the real tools would
tool=$(basename "$1"); out=""; prev=""
for a in "$@"; do [ "$prev" = "--out" ] && out=$a; prev=$a; done
echo "stub $tool" >> "$STUB_LOG"
case "$tool" in
  pod_dose_layout.py) echo x > "$out"; echo '{}' > "$out.json";;
  extract_states.py) [ -n "${STUB_FAIL_EXTRACT:-}" ] && { echo boom; exit 1; }
     echo x > "$out"; printf '{\n "ckpt_sha256": "%s",\n "lo": 0,\n "hi": %s,\n "n_clips": %s\n}\n' "${STUB_SHA:-aaaa}" "${STUB_HI:-4572}" "${STUB_NCLIPS:-4572}" > "$out.json";;
  pod_lead_apply.py) echo x > "$out"; echo '{}' > "$out.json"; echo x > "$out.head.npz";;
esac
echo "done $tool"
exit 0
"""


def sh(p):
    return TL.sh(p)


def run(env_extra, tmp_path, timeout=120):
    env = dict(os.environ)
    env.update(env_extra)
    return subprocess.run([BASH, SCRIPT], env=env, capture_output=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=sh(tmp_path))


def fake_pod(tmp_path, gpu_pids=0, anon_gb=1, max_gb=60, banked_sha="aaaa"):
    d = tmp_path / "pod"
    (d / "bin").mkdir(parents=True, exist_ok=True)
    (d / "bin" / "nvidia-smi").write_text("#!/usr/bin/env bash\n" + "".join("echo %d\n" % (1000 + i) for i in range(gpu_pids)) + "exit 0\n", encoding="utf-8", newline="\n")
    (d / "memory.max").write_text(str(max_gb * 1073741824), encoding="utf-8")
    (d / "memory.stat").write_text("anon %d\nfile 1\n" % (anon_gb * 1073741824), encoding="utf-8")
    P = d / "probe"
    (P / "states").mkdir(parents=True, exist_ok=True)
    for f in ("extract_states.py", "pod_lead_apply.py"):
        (P / f).write_text("# stub\n", encoding="utf-8")
    T = d / "qhead_dose" / "tools"
    T.mkdir(parents=True, exist_ok=True)
    (T / "pod_dose_layout.py").write_text("# stub\n", encoding="utf-8")
    (d / "cache" / "train" / "clips").mkdir(parents=True, exist_ok=True)
    for arm in ("LSD_s0", "LSD_s1"):
        (d / "runs" / "leadsup_b" / arm).mkdir(parents=True, exist_ok=True)
        (d / "runs" / "leadsup_b" / arm / "ckpt_last.pt").write_bytes(b"ckpt")
        (P / "states" / (arm + "_train.npz.json")).write_text('{\n "ckpt_sha256": "%s",\n "lo": 600\n}\n' % banked_sha, encoding="utf-8")
        (P / "states" / (arm + "_train.npz")).write_bytes(b"BANKED")
    py = d / "fakepy"
    py.write_text(FAKE_PY, encoding="utf-8", newline="\n")
    env = {"QD_W": sh(d), "QD_CACHE": sh(d / "cache"), "QD_PY": sh(py), "QD_NICE": "", "QD_CG_MAX": sh(d / "memory.max"), "QD_CG_STAT": sh(d / "memory.stat"),
           "PATH": sh(d / "bin") + os.pathsep + os.environ["PATH"], "QD_PROBE_GAP_S": "1", "QD_POLL_S": "1", "QD_MAX_WAIT_S": "2", "STUB_LOG": sh(d / "stub.log")}
    return d, env


def test_syntax_lf_and_forbidden_commands():
    r = subprocess.run([BASH, "-n", SCRIPT], capture_output=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    raw = open(SCRIPT, "rb").read()
    assert b"\r" not in raw
    code = "\n".join(l for l in raw.decode("utf-8").splitlines() if not l.lstrip().startswith("#"))
    for bad in ("pgrep", "pkill", "killall", "ps -", "ps aux", "pidof"):
        assert bad not in code, bad
    assert not re.search(r"(^|[;&|\s])(rm|mv|rmdir|shred)(\s|$)", code), "rm or mv in the script"
    assert "--query-compute-apps=pid" in code and "memory.max" in code and "memory.stat" in code and "set -o noclobber" in code


def test_dry_run_order_and_content_and_nothing_created(tmp_path):
    d, env = fake_pod(tmp_path)
    env["DRY"] = "1"
    before = sorted(str(p) for p in d.rglob("*"))
    r = run(env, tmp_path)
    assert r.returncode == 0, r.stderr + r.stdout
    out = r.stdout
    assert sorted(str(p) for p in d.rglob("*")) == before, "DRY must create nothing"
    pos = [out.find(s) for s in ("pod_dose_layout.py", "extract_states.py --tree", "pod_lead_apply.py --tree")]
    assert all(p >= 0 for p in pos)
    ex = [m.start() for m in re.finditer(r"extract_states\.py --tree \S+ --ckpt \S+/(LSD_s[01])/ckpt_last\.pt --cache \S+ --split train --lo 0 --hi 4572 --out \S+/(LSD_s[01])_dose_train\.npz", out)]
    ap = [m.start() for m in re.finditer(r"pod_lead_apply\.py --tree \S+ --ckpt \S+/(LSD_s[01])/ckpt_last\.pt --states \S+_dose_train\.npz --out \S+_dose_train_lead\.npz", out)]
    assert len(ex) == 2 and len(ap) == 2
    assert out.find("pod_dose_layout.py") < ex[0] < ap[0] < ex[1] < ap[1]
    assert out.count("DRY: slot wait before") == 1 + 2 + 2
    assert "DRY: slot wait before layout" in out and "DRY: slot wait before states LSD_s1" in out and "DRY: slot wait before lead read LSD_s0" in out
    assert "--lo 0 --hi 4572 --out" in out and "train_layout_0_4572.npz" in out
    assert "leadsup_b/LSD_s0/ckpt_last.pt" in out and "leadsup_b/LSD_s1/ckpt_last.pt" in out
    assert "ZZ QHEAD-DOSE INPUTS DONE" in out
    assert "/probe/states/LSD_s0_train.npz " not in out and "--out " + sh(d / "probe") not in out, "a banked file name must not appear as an output"


@pytest.mark.parametrize("marker", ["QHEAD_DOSE_INPUTS_STARTED.txt", "QHEAD_DOSE_INPUTS_DONE.txt", "QHEAD_DOSE_INPUTS_STOP.txt"])
def test_markers_refuse_a_second_start(tmp_path, marker):
    d, env = fake_pod(tmp_path)
    (d / "qhead_dose" / "run").mkdir(parents=True)
    (d / "qhead_dose" / "run" / marker).write_text("x")
    r = run(env, tmp_path)
    assert r.returncode == 2 and "REFUSED" in r.stdout and marker in r.stdout


def test_missing_tool_stops_closed_with_a_marker_and_last_line(tmp_path):
    d, env = fake_pod(tmp_path)
    os.replace(d / "qhead_dose" / "tools" / "pod_dose_layout.py", d / "qhead_dose" / "tools" / "pod_dose_layout.py.moved")
    r = run(env, tmp_path)
    assert r.returncode == 4
    stop = (d / "qhead_dose" / "run" / "QHEAD_DOSE_INPUTS_STOP.txt").read_text()
    assert "pod_dose_layout.py" in stop
    assert r.stdout.strip().splitlines()[-1].endswith("ZZ QHEAD-DOSE INPUTS STOP: missing %s/qhead_dose/tools/pod_dose_layout.py" % sh(d))


def test_missing_checkpoint_stops_closed(tmp_path):
    d, env = fake_pod(tmp_path)
    os.replace(d / "runs" / "leadsup_b" / "LSD_s1" / "ckpt_last.pt", d / "runs" / "leadsup_b" / "LSD_s1" / "ckpt_last.pt.moved")
    r = run(env, tmp_path)
    assert r.returncode == 4 and "LSD_s1/ckpt_last.pt" in (d / "qhead_dose" / "run" / "QHEAD_DOSE_INPUTS_STOP.txt").read_text()


def test_slot_wait_refuses_a_fourth_gpu_job(tmp_path):
    d, env = fake_pod(tmp_path, gpu_pids=3)
    r = run(env, tmp_path)
    assert r.returncode == 4, r.stdout[-600:]
    s = (d / "qhead_dose" / "run" / "QHEAD_DOSE_INPUTS_STOP.txt").read_text()
    assert "never a fourth GPU job" in s and "layout" in s
    assert not (d / "stub.log").exists(), "no tool may run without a slot"


def test_slot_wait_needs_28_gb_of_the_container_free(tmp_path):
    d, env = fake_pod(tmp_path, gpu_pids=0, anon_gb=40, max_gb=60)          # 20 GB free
    r = run(env, tmp_path)
    assert r.returncode == 4 and "never a fourth GPU job" in (d / "qhead_dose" / "run" / "QHEAD_DOSE_INPUTS_STOP.txt").read_text()


def test_full_run_with_stub_tools_writes_new_names_only_and_the_last_line(tmp_path):
    d, env = fake_pod(tmp_path, gpu_pids=2, anon_gb=10)
    r = run(env, tmp_path)
    assert r.returncode == 0, r.stdout[-800:] + r.stderr
    assert r.stdout.strip().splitlines()[-1].endswith("ZZ QHEAD-DOSE INPUTS DONE 2")
    assert (d / "qhead_dose" / "run" / "QHEAD_DOSE_INPUTS_DONE.txt").read_text().strip() == "2"
    st = d / "qhead_dose" / "states"
    for arm in ("LSD_s0", "LSD_s1"):
        for suf in ("_dose_train.npz", "_dose_train.npz.json", "_dose_train_lead.npz", "_dose_train_lead.npz.json", "_dose_train_lead.npz.head.npz"):
            assert (st / (arm + suf)).exists(), arm + suf
        assert (d / "probe" / "states" / (arm + "_train.npz")).read_bytes() == b"BANKED"
    assert (d / "qhead_dose" / "layout" / "train_layout_0_4572.npz.json").exists()
    order = (d / "stub.log").read_text().split()
    tools = [t for t in order if t.endswith(".py")]
    assert tools == ["pod_dose_layout.py", "extract_states.py", "pod_lead_apply.py", "extract_states.py", "pod_lead_apply.py"]
    log = (d / "qhead_dose" / "run" / "dose_inputs.log").read_text()
    assert log.count("slot free on two probes for") == 5
    r2 = run(env, tmp_path)
    assert r2.returncode == 2 and "REFUSED" in r2.stdout


def test_extract_args_are_appended_to_the_extraction_command_only(tmp_path):
    d, env = fake_pod(tmp_path, gpu_pids=0)
    env["DRY"] = "1"
    env["QD_EXTRACT_ARGS"] = "--device cuda"
    r = run(env, tmp_path)
    assert r.returncode == 0
    lines = [l for l in r.stdout.splitlines() if l.startswith("DRY: ") and "extract_states.py" in l]
    assert len(lines) == 2 and all("--device cuda" in l for l in lines)
    assert not [l for l in r.stdout.splitlines() if ("pod_lead_apply.py" in l or "pod_dose_layout.py" in l) and "--device cuda" in l]


def test_new_record_with_the_wrong_clip_count_stops(tmp_path):
    d, env = fake_pod(tmp_path)
    env["STUB_NCLIPS"] = "1200"
    r = run(env, tmp_path)
    assert r.returncode == 4 and "does not say lo 0, hi 4572, n_clips 4572" in (d / "qhead_dose" / "run" / "QHEAD_DOSE_INPUTS_STOP.txt").read_text()


def test_checkpoint_sha_difference_between_new_and_banked_record_stops(tmp_path):
    d, env = fake_pod(tmp_path, banked_sha="bbbb")
    r = run(env, tmp_path)
    assert r.returncode == 4 and "checkpoint sha256" in (d / "qhead_dose" / "run" / "QHEAD_DOSE_INPUTS_STOP.txt").read_text()


def test_tool_failure_stops_and_a_partial_state_file_is_never_overwritten(tmp_path):
    d, env = fake_pod(tmp_path)
    env["STUB_FAIL_EXTRACT"] = "1"
    r = run(env, tmp_path)
    assert r.returncode == 4 and "states LSD_s0 failed" in (d / "qhead_dose" / "run" / "QHEAD_DOSE_INPUTS_STOP.txt").read_text()
    # a restart (marker moved by the lead) meets a state file without a record: stop, do not overwrite
    st = d / "qhead_dose" / "states" / "LSD_s0_dose_train.npz"
    st.write_bytes(b"PARTIAL")
    for m in ("QHEAD_DOSE_INPUTS_STARTED.txt", "QHEAD_DOSE_INPUTS_STOP.txt"):
        os.replace(d / "qhead_dose" / "run" / m, d / "qhead_dose" / "run" / (m + ".old"))
    env.pop("STUB_FAIL_EXTRACT")
    r = run(env, tmp_path)
    assert r.returncode == 4 and "without its record" in (d / "qhead_dose" / "run" / "QHEAD_DOSE_INPUTS_STOP.txt").read_text()
    assert st.read_bytes() == b"PARTIAL"
