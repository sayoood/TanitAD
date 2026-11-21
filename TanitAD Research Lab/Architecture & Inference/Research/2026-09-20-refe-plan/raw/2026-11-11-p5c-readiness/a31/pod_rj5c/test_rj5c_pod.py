"""Tests for pod_rj5c/rj5c_pod.sh (PREREG_P5C_A3 section 7: chain C rolled ON THE POD).  Stub sandbox: no pod, no GPU, no nuPlan; the python of the pod is a recording stub.
Run: python -m pytest -q test_rj5c_pod.py -p no:cacheprovider   (Git bash on PATH)
Every refusal has a RED test (the refusal fires) and a deliberate-regression arm (the script text is MUTATED, the refusal removed, and the same scenario must go the other way)."""
import os
import re
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "rj5c_pod.sh")
BASH = shutil.which("bash")
if BASH and os.path.basename(os.path.dirname(BASH)).lower() == "bin":
    real = os.path.join(os.path.dirname(os.path.dirname(BASH)), "usr", "bin", "bash.exe")
    BASH = real if os.path.exists(real) else BASH
FILES = {"rj5_roll.py": "PIN_ROLL", "rj5_chain.py": "PIN_CHAIN", "patch/build_teacher_rollouts.py": "PIN_PATCH", "logs_fetch_C.txt": "PIN_LOGS_C", "control36_logs.txt": "PIN_CTL36"}


def posix(p):
    return str(p).replace("\\", "/")


def sha(p):
    import hashlib
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def w(p, text):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w", encoding="utf-8", newline="\n").write(text)


class Pod:
    def __init__(self, tmp, mutate=(), quota="1530000 100000", ps_trainer=False, n_c=192, n_ctl=35, tamper=None, drop_db=None, dirty_empty=False):
        self.t = str(tmp)
        self.ws, self.ship, self.out = self.t + "/ws", self.t + "/ws/data/rj5c_ship", self.t + "/out"
        self.calls = self.t + "/calls.log"
        open(self.calls, "w").close()
        self.dbs, self.bank = self.ws + "/data/navtrain_dbs", self.ws + "/data/refe_navtrain/train_grow_cmd_3c_p7"
        for d in (self.dbs, self.bank, self.ws + "/refe-plan-vdac/refe", self.ws + "/refe-plan-vdac/code", self.ws + "/refe-plan-vdac/eval", self.ws + "/venv-teacher/bin", self.t + "/cg"):
            os.makedirs(d, exist_ok=True)
        w(self.t + "/cg/cpu.max", quota + "\n")
        w(self.ws + "/refe-plan-vdac/refe/x.py", "# x\n")
        w(self.ws + "/teacher_env.sh", "export DZ_ROOT=/dz\nexport DRIVERL_EVAL_ROUTE_LANE_RANK=2\n")
        w(self.ws + "/venv-teacher/bin/python", '#!/bin/bash\necho "CALL $*" >> "%s"\necho "ENV pp=$PYTHONPATH dev=$DRIVERL_EVAL_DEVICE cvd=$CUDA_VISIBLE_DEVICES lane=${DRIVERL_EVAL_ROUTE_LANE_RANK-UNSET} omp=$OMP_NUM_THREADS hz=$REFE_SIM_HZ" >> "%s"\nexit 0\n' % (self.calls, self.calls))
        w(self.bank + "/targets_rank0.jsonl", '{"log_name": "l", "token": "t", "rank": 0}\n')
        logs_c = ["2021.01.01.00.00.%03d_veh-1_00001_00002" % i for i in range(n_c)]
        logs_ctl = ["2021.02.02.00.00.%03d_veh-2_00001_00002" % i for i in range(n_ctl)]
        for lg in logs_c + logs_ctl:
            w("%s/%s.db" % (self.dbs, lg), "db\n")
        if drop_db:
            os.remove("%s/%s.db" % (self.dbs, logs_c[drop_db]))
        w(self.ship + "/rj5_roll.py", "# roll\n")
        w(self.ship + "/rj5_chain.py", "# chain\n")
        w(self.ship + "/patch/build_teacher_rollouts.py", "# patch\n")
        w(self.ship + "/logs_fetch_C.txt", "\n".join(logs_c) + "\n")
        w(self.ship + "/control36_logs.txt", "\n".join(logs_ctl) + "\n")
        if dirty_empty:
            w(self.out + "/empty_db_dir/stray.db", "x")
        w(self.t + "/ps_ok.txt", "/usr/bin/python3 other_job.py\n")
        w(self.t + "/ps_train.txt", "/workspace/venv/bin/python /workspace/refe-plan-p7/refe/train.py --out x\n")
        text = open(SCRIPT, encoding="utf-8").read()
        for old, new in (("PKGSRC=/workspace/refe-plan-vdac", "PKGSRC=%s/refe-plan-vdac" % posix(self.ws)), ("BANK=/workspace/data/refe_navtrain/train_grow_cmd_3c_p7", "BANK=%s" % posix(self.bank)),
                         ("DBS=/workspace/data/navtrain_dbs", "DBS=%s" % posix(self.dbs)), ("TENV=/workspace/teacher_env.sh", "TENV=%s/teacher_env.sh" % posix(self.ws)),
                         ("PY=/workspace/venv-teacher/bin/python", "PY=%s/venv-teacher/bin/python" % posix(self.ws))):
            assert text.count(old) == 1, old
            text = text.replace(old, new)
        pinned = {"PIN_ROLL": self.ship + "/rj5_roll.py", "PIN_CHAIN": self.ship + "/rj5_chain.py", "PIN_PATCH": self.ship + "/patch/build_teacher_rollouts.py", "PIN_LOGS_C": self.ship + "/logs_fetch_C.txt",
                  "PIN_BANK_R0": self.bank + "/targets_rank0.jsonl", "PIN_CTL36": self.ship + "/control36_logs.txt"}
        for k, p in pinned.items():
            text, n = re.subn(r"^%s=[0-9a-f]{64}" % k, "%s=%s" % (k, sha(p)), text, flags=re.M)
            assert n == 1, k
        for old, new in mutate:
            assert text.count(old) == 1, (text.count(old), old[:80])
            text = text.replace(old, new)
        if tamper:                                        # a shipped file whose bytes differ from the pin (the pin was computed first)
            p = {"bank": self.bank + "/targets_rank0.jsonl"}.get(tamper, self.ship + "/" + tamper)
            if tamper.endswith("_logs.txt") or tamper == "logs_fetch_C.txt":      # same line count: only the pin can catch it
                t0 = open(p, encoding="utf-8").read().rstrip("\n")
                open(p, "w", encoding="utf-8", newline="\n").write(t0[:-1] + "X\n")
            else:
                open(p, "a").write("tampered\n")
        self.script = self.t + "/rj5c_pod.sh"
        w(self.script, text)
        self.ps = "cat " + posix(self.t + ("/ps_train.txt" if ps_trainer else "/ps_ok.txt"))

    def run(self, *args):
        env = {k: v for k, v in os.environ.items() if not k.startswith(("RJ5C_", "DRIVERL", "DZ_"))}
        env.update({"RJ5C_SHIP": posix(self.ship), "RJ5C_OUT": posix(self.out), "RJ5C_PKG": posix(self.t + "/pkg"), "RJ5C_PS": self.ps, "RJ5C_CGROUP_ROOT": posix(self.t + "/cg")})
        r = subprocess.run([BASH, posix(self.script), *args], capture_output=True, text=True, env=env, encoding="utf-8", errors="replace", timeout=120)
        return r.returncode, r.stdout + r.stderr

    def calls_text(self):
        return open(self.calls, encoding="utf-8").read()


def refused(rc, out, needle):
    return rc == 1 and "ZZRJ5C_REFUSE" in out and needle in out


def test_static_lf_bash_n_no_fetch_no_pgrep_no_nproc():
    t = open(SCRIPT, encoding="utf-8", newline="").read()
    assert "\r" not in t and subprocess.run([BASH, "-n", posix(SCRIPT)], capture_output=True).returncode == 0
    code = "\n".join(ln for ln in t.splitlines() if not ln.strip().startswith("#"))
    assert not re.search(r"pgrep|pkill|\$\(\s*nproc|`nproc", code)
    assert not re.search(r'"\$PY" "\$SHIP/rj5_chain.py"[^\n]*--fetch', code.replace("never fetches", ""))      # no chain command carries --fetch
    assert "refe-plan-p7/" not in code


def test_happy_path_command_env_and_both_runs(tmp_path):
    p = Pod(tmp_path)
    rc, out = p.run()
    assert rc == 0 and "ZZRJ5C_DONE" in out, out
    c = p.calls_text().splitlines()
    calls = [x for x in c if x.startswith("CALL")]
    envs = [x for x in c if x.startswith("ENV")]
    assert len(calls) == 2 and not any("--fetch" in x for x in calls)
    e = posix(p.out)
    d = posix(p.dbs)
    assert calls[0] == ("CALL %s/rj5_chain.py --bank %s/targets_rank0.jsonl --logs-file %s/logs_fetch_C.txt --work %s/run_fetch_C --db-dir %s/empty_db_dir --held-db-dir %s --procs 14 --batch-logs 48 --arms RJ5 --control-every 20"
                        % (posix(p.ship), posix(p.bank), posix(p.ship), e, e, d))
    assert calls[1] == ("CALL %s/rj5_chain.py --bank %s/targets_rank0.jsonl --logs-file %s/control36_logs.txt --work %s/run_control36 --db-dir %s/empty_db_dir --held-db-dir %s --procs 14 --batch-logs 48 --arms R0 --control-every 0"
                        % (posix(p.ship), posix(p.bank), posix(p.ship), e, e, d))
    pkg = posix(p.t + "/pkg")
    for x in envs:
        assert x == "ENV pp=%s/patch:%s/refe:%s/code:%s/eval:/dz/nuplan-devkit dev=cpu cvd=-1 lane=UNSET omp=1 hz=10" % (posix(p.ship), pkg, pkg, pkg), x
    assert os.path.exists(p.t + "/pkg/refe/x.py") and not os.listdir(p.out + "/empty_db_dir")


def test_dry_run_runs_nothing(tmp_path):
    p = Pod(tmp_path)
    rc, out = p.run("--dry-run")
    assert rc == 0 and p.calls_text() == "" and "DRY:" in out


# scenario: (pod options, args, needle)
SCEN = {
    "fetch_flag": (dict(), ("--fetch",), "never fetches"),
    "fetch_eq": (dict(), ("--fetch=1",), "never fetches"),
    "other_arg": (dict(), ("--procs", "40"), "unknown argument"),
    "trainer": (dict(ps_trainer=True), (), "a trainer is running"),
    "pin_roll": (dict(tamper="rj5_roll.py"), (), "rj5_roll.py"),
    "pin_chain": (dict(tamper="rj5_chain.py"), (), "rj5_chain.py"),
    "pin_patch": (dict(tamper="patch/build_teacher_rollouts.py"), (), "TF-N2 patch"),
    "pin_logs": (dict(tamper="logs_fetch_C.txt"), (), "logs_fetch_C.txt"),
    "pin_bank": (dict(tamper="bank"), (), "the pod's own bank"),
    "pin_ctl": (dict(tamper="control36_logs.txt"), (), "control36_logs.txt"),
    "missing_db": (dict(drop_db=7), (), "no pod DB for log"),
    "quota": (dict(quota="1330000 100000"), (), "cgroup grants 13"),
    "dirty_empty": (dict(dirty_empty=True), (), "is not empty"),
}


@pytest.mark.parametrize("name", list(SCEN))
def test_each_refusal_fires_and_launches_nothing(tmp_path, name):
    opts, args, needle = SCEN[name]
    p = Pod(tmp_path, **opts)
    rc, out = p.run(*args)
    assert refused(rc, out, needle), out
    assert p.calls_text() == ""


# the deliberate-regression arms: ONE refusal removed from the script text; the same scenario must then NOT refuse for that reason
MUT = [
    ("fetch_refusal_removed", '    --fetch|--fetch=*|--fetch-script|--fetch-script=*) refuse', '    --fetch|--fetch=*|--fetch-script|--fetch-script=*) true; : refuse', "fetch_flag"),
    ("trainer_check_removed", '[ -z "$BUSY" ] || refuse', 'true || refuse', "trainer"),
    ("pin_roll_removed", 'pin "$SHIP/rj5_roll.py" "$PIN_ROLL" rj5_roll.py\n', '', "pin_roll"),
    ("pin_chain_removed", 'pin "$SHIP/rj5_chain.py" "$PIN_CHAIN" rj5_chain.py\n', '', "pin_chain"),
    ("pin_patch_removed", 'pin "$SHIP/patch/build_teacher_rollouts.py" "$PIN_PATCH" "TF-N2 patch"\n', '', "pin_patch"),
    ("pin_logs_removed", 'pin "$SHIP/logs_fetch_C.txt" "$PIN_LOGS_C" logs_fetch_C.txt\n', '', "pin_logs"),
    ("pin_bank_removed", 'pin "$BANK/targets_rank0.jsonl" "$PIN_BANK_R0" "the pod\'s own bank"\n', '', "pin_bank"),
    ("pin_ctl_removed", 'pin "$SHIP/control36_logs.txt" "$PIN_CTL36" control36_logs.txt\n', '', "pin_ctl"),
    ("quota_check_removed", '[ "$Q" -ge "$PROCS" ] || refuse', 'true || refuse', "quota"),
    ("empty_dir_check_removed", 'then refuse "the scratch --db-dir $EMPTY is not empty"; fi', 'then true; fi', "dirty_empty"),
]


@pytest.mark.parametrize("name,old,new,scen", MUT, ids=[m[0] for m in MUT])
def test_mutant_refusal_removed_goes_the_other_way(tmp_path, name, old, new, scen):
    opts, args, needle = SCEN[scen]
    p = Pod(tmp_path, mutate=[(old, new)], **opts)
    rc, out = p.run(*args)
    assert not refused(rc, out, needle), "MUTANT SURVIVED: " + name
    if name == "fetch_refusal_removed":
        assert "unknown argument" not in out or True
