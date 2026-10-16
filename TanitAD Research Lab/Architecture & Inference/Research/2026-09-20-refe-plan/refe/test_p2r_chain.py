#!/usr/bin/env python3
"""Tests for the P2 re-run chain: refe/p2r_pod.sh, p2r_sync.py, p2r_gate_prep.py (eval/PREREG_P2_AMENDMENT_1.md).

No pod, no GPU, no network. Three layers:
  1. `check_chain(text)` -- static properties of the chain that must hold: no SFT-4 dependency and no wait loop, the
     training arguments byte-identical to p2_chain.sh, HF_TOKEN never echoed, every child launched with fd 200 closed,
     the stages in order, the gate PARSED, egress present periodically AND at the end, the labels banked before training.
  2. MUTATIONS of the chain text: each property is broken in a copy and `check_chain` must say so. A check that cannot
     go RED is not a check.
  3. Behaviour of the helpers (gate_pass under bash; p2r_sync restore / extraction / tick / final / watch and
     p2r_gate_prep against a fake HF API), each in both directions.

Run: python refe/test_p2r_chain.py   -> ZZTEST_PASS <n>   (needs bash on PATH; Git Bash on the dev box)
Any skipped check prints ZZTEST_SKIP <reason> and is NOT counted.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hf_egress as E  # noqa: E402
import p2r_carve as CV  # noqa: E402
import p2r_gate_prep as GP  # noqa: E402
import p2r_sync as S  # noqa: E402

CHAIN = HERE / "p2r_pod.sh"
OLD = HERE / "p2_chain.sh"
ARGS_RX = re.compile(r'^ARGS="(.*)"$', re.M)
CHILD_RX = re.compile(r'"\$PY" +(train\.py|p2r_sync\.py|p2_gate\.py|p2r_gate_prep\.py|pdm_label_complete\.py|launch_gate\.py|'
                      r'onpolicy_relabel_pdm\.py)|"\$PY" +"\$PKG/refe/launch_gate\.py"|bash +"\$PKG/code/|apt-get|uv pip|^sleep |\bsleep +\d+ |\bdd +if=')
ORDER = ["source \"$WORK/teacher_env.sh\"", "p2r_sync.py watch", "STAGES=weights", "p2r_sync.py restore", "p2r_sync.py recover",
         "p2r_carve.py --sets", "STAGES=dbs",
         "PT._setup(); print('ZZPDM_SETUP_OK')", "onpolicy_relabel_pdm.py --sets", "p2r_sync.py pdm-push", "p2_gate.py --init-from", "launch_gate.py run --profile refe_p2",
         "train.py $ARGS --preflight", "train.py $ARGS >>", "p2r_sync.py final"]
TOKEN_OK_RX = re.compile(r"^token_ok\(\) \{\n.*?^\}\n", re.S | re.M)


def logical_commands(text):
    """Backslash-joined lines, heredoc bodies dropped, an open `( ... )` subshell merged into one command."""
    out, cur, depth, heredoc = [], "", 0, None
    raw = text.split("\n")
    joined, buf = [], ""
    for ln in raw:
        if ln.endswith("\\"):
            buf += ln[:-1] + " "
            continue
        joined.append(buf + ln)
        buf = ""
    for ln in joined:
        if heredoc:
            if ln == heredoc:
                heredoc = None
            continue
        m = re.search(r"<<'?(\w+)'?\s*$", ln)
        if m:
            heredoc = m.group(1)
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        cur = (cur + " " + s).strip() if cur else s
        depth += s.count("(") - s.count(")")
        if depth <= 0:
            out.append(cur)
            cur, depth = "", 0
    if cur:
        out.append(cur)
    return out


def token_line_ok(ln):
    s = ln.strip()
    return (s.startswith("#") or "env -u HF_TOKEN" in s
            or s == '[ -n "${HF_TOKEN:-}" ] || refuse "HF_TOKEN is not set in the environment"')


def check_chain(text, ref_args):
    v = []
    if "\r" in text:
        v.append("CRLF")
    if re.search(r"refe_sft4|RUN EXIT|p2_gate_runner|^\s*until\b", text, re.M):
        v.append("SFT4_OR_WAIT_DEPENDENCY")
    m = ARGS_RX.search(text)
    if not m or m.group(1) != ref_args:
        v.append("ARGS_CHANGED")
    for ln in text.split("\n"):
        if "HF_TOKEN" in ln and not token_line_ok(ln):
            v.append("HF_TOKEN_USE:" + ln.strip()[:60])
    if re.search(r"^\s*set\s+-[a-wyz]*x|\bbash\s+-x\b", text, re.M):
        v.append("XTRACE")
    if '[ -n "${HF_TOKEN:-}" ] || refuse' not in text:
        v.append("NO_TOKEN_ASSERT")
    for c in logical_commands(text):
        if CHILD_RX.search(c) and "200>&-" not in c:
            v.append("FD200:" + c[:70])
    pos = [text.find(k) for k in ORDER]
    for k, p in zip(ORDER, pos):
        if p < 0:
            v.append("MISSING:" + k)
    if all(p >= 0 for p in pos) and pos != sorted(pos):
        bad = [ORDER[i] for i in range(1, len(ORDER)) if pos[i] < pos[i - 1]]
        v.append("ORDER:" + ",".join(bad))
    if 'gate_pass "$P2/gate/gate.json"' not in text:
        v.append("GATE_NOT_CHECKED")
    if re.search(r"grep[^\n]*verdict", text):
        v.append("GATE_GREPPED")
    if '"^PREFLIGHT_OK"' not in text:
        v.append("NO_PREFLIGHT_CHECK")
    w = re.search(r'"\$PY" p2r_sync\.py watch.*?200>&- &', text, re.S)
    if not w or "--interval-min" not in w.group(0) or "--heavy-min" not in w.group(0):
        v.append("NO_PERIODIC_EGRESS")
    if "p2r_sync.py final" not in text or "ZZP2R_FINAL_OK" not in text:
        v.append("NO_FINAL_EGRESS")
    elif not re.search(r'exit 1\s*$', text.strip(), re.M) or not text.rstrip().endswith("exit 1"):
        v.append("FINAL_FAILURE_NOT_PROPAGATED")
    if not re.search(r'elif \[ -f "\$PDMLOG" \] && "\$PY" pdm_label_complete\.py', text):
        v.append("RESTORED_LABELS_NOT_AUDITED")
    if "ZZEGRESS_OK p2r_pdm_labels" not in text:
        v.append("LABELS_NOT_BANKED_CHECK")
    # the 24 SFT held-out logs OUT before the relabel and the trainer see the sets (MEASURED 2026-10-11: 177,836 = 174,912 + 2,924)
    cv = re.search(r'"\$PY" p2r_carve\.py --sets "\$SETS" .*?--expect "\$\{P2R_EXPECT_SERVED:-174912\}"', text, re.S)
    if not cv or not re.search(r'grep -q "\^ZZCARVE_OK" "\$P2/carve\.log" \|\| refuse', text) or "stage_mark carve" not in text:
        v.append("CARVE_NOT_CHECKED")
    # the labeller must prove it can start before ~10 h of relabel (2026-10-11: navsim missing, every set failed)
    if not re.search(r'grep -q "\^ZZPDM_SETUP_OK" "\$P2/pdm_setup\.log" \|\| refuse', text):
        v.append("PDM_SETUP_NOT_CHECKED")
    # every file the chain refuses to start without must EXIST in the package (2026-10-11: a mangled continuation turned the list
    # into a literal "\n" token, `bash -n` passed, and the pod refused "$PKG/n missing")
    fl = re.search(r'for f in (.*?); do\n  \[ -f "\$PKG/\$f" \] \|\| refuse', text, re.S)
    if not fl:
        v.append("REQUIRED_FILES_LOOP_MISSING")
    else:
        for f in fl.group(1).replace("\\\n", " ").split():
            if not (HERE.parent / f).is_file():
                v.append("REQUIRED_FILE_ABSENT:" + f)
    v += check_launch_gate(text)
    return v


def check_launch_gate(text):
    """THE BINDING LAUNCH-GATE RULE (CLAUDE.md, PI 2026-09-26): no train.py starts without a PASS token that VERIFIES for the exact argv, tree and data."""
    v = []
    run_pos = text.find("launch_gate.py run --profile refe_p2")
    if run_pos < 0:
        v.append("NO_LAUNCH_GATE_STAGE")
    if "stage_mark launch_gate" not in text:
        v.append("LAUNCH_GATE_STAGE_NOT_MARKED")
    am = ARGS_RX.search(text)
    if am and run_pos >= 0 and am.start() > run_pos:
        v.append("ARGS_AFTER_GATE")             # the token must be minted over the very string train.py is later given
    if len(ARGS_RX.findall(text)) != 1:
        v.append("ARGS_DEFINED_TWICE")
    m = TOKEN_OK_RX.search(text)
    if not m:
        v.append("NO_TOKEN_OK_FUNCTION")
    else:
        body = m.group(0)
        for needle in ('launch_gate.py" verify', '--profile refe_p2', '--args "$ARGS"', '--pkg "$PKG"', '--token "$tok"'):
            if needle not in body:
                v.append("TOKEN_OK_NOT_VERIFYING:" + needle)
        if re.search(r"\bgrep\b|^\s*return 0|\btrue\b", body, re.M):
            v.append("TOKEN_OK_NOT_VERIFYING:grep_or_constant")
    tp = text.find('train.py $ARGS --preflight')
    if "token_ok || refuse" not in text[max(0, tp - 260):tp]:
        v.append("PREFLIGHT_WITHOUT_TOKEN")
    lp = text.find('train.py $ARGS >>')
    if "token_ok || refuse" not in text[max(0, lp - 260):lp]:
        v.append("TRAIN_WITHOUT_TOKEN")
    if text.count("token_ok ||") < 4:
        v.append("TOKEN_CHECKS_MISSING")        # after the stage, again before it is marked done, before the preflight, before every launch
    for ln in text.split("\n"):
        if not ln.strip().startswith("#") and "grep" in ln and ("ZZGATE-" in ln or "launch_gate.log" in ln):
            v.append("TOKEN_GREPPED:" + ln.strip()[:60])
    if "SHIP_MANIFEST.json" not in text:
        v.append("SHIP_MANIFEST_NOT_REQUIRED")
    if not re.search(r'^export TANITAD_GATE_KEY_FILE="\$P2/', text, re.M):
        v.append("KEY_NOT_OUTSIDE_TREE")
    return v


def tar_owner_violations(text):
    """Every `tar -x` must carry --no-same-owner: as root, tar restores the archive's owner, the 2026-10-11 pod's NFS refuses the
    chown, and the non-zero exit is indistinguishable from a corrupt archive (pixel shards marked failed and deleted)."""
    return [ln.strip()[:60] for ln in text.split("\n") if re.search(r"\btar -x", ln) and not ln.strip().startswith("#")
            and "--no-same-owner" not in ln]


def drop_line(text, needle):
    return "\n".join(ln for ln in text.split("\n") if needle not in ln)


def mutations(text):
    """(name, mutated text, violation prefix that must appear)."""
    out = [
        ("delete the final push", drop_line(text, "p2r_sync.py final"), "NO_FINAL_EGRESS"),
        ("delete the periodic watcher", re.sub(r'"\$PY" p2r_sync\.py watch.*?200>&- &\n', "", text, flags=re.S),
         "NO_PERIODIC_EGRESS"),
        ("delete the gate check", drop_line(text, 'gate_pass "$P2/gate/gate.json"'), "GATE_NOT_CHECKED"),
        ("grep the gate verdict instead of parsing it",
         text.replace('gate_pass "$P2/gate/gate.json" 2>> $L', "grep -q '\"verdict\": \"PASS\"' $P2/gate/gate.json"),
         "GATE_GREPPED"),
        ("delete the preflight check", drop_line(text, '"^PREFLIGHT_OK"'), "NO_PREFLIGHT_CHECK"),
        ("train child keeps fd 200", text.replace('"$PY" train.py $ARGS >> "$P2/train.log" 2>&1 200>&-',
                                                  '"$PY" train.py $ARGS >> "$P2/train.log" 2>&1'), "FD200"),
        ("echo the token", text.replace('log "env ok"', 'log "env ok token=$HF_TOKEN"'), "HF_TOKEN_USE"),
        ("xtrace on", text.replace("set -u\n", "set -ux\n", 1), "XTRACE"),
        ("reinstate an SFT-4 wait",
         text.replace("log \"env ok\"", "until grep -q \"RUN EXIT\" /workspace/data/refe_sft4/chain.log; do sleep 120; done"),
         "SFT4_OR_WAIT_DEPENDENCY"),
        ("change a training argument", text.replace("--lr 5e-5 --amp", "--lr 1e-4 --amp"), "ARGS_CHANGED"),
        ("drop the token assertion", drop_line(text, '[ -n "${HF_TOKEN:-}" ] || refuse'), "NO_TOKEN_ASSERT"),
        ("drop the label-banking check", drop_line(text, "ZZEGRESS_OK p2r_pdm_labels"), "LABELS_NOT_BANKED_CHECK"),
        ("the 2026-10-11 mangled continuation", text.replace("splits/navtrain.yaml \\\n", "splits/navtrain.yaml \\n"),
         "REQUIRED_FILE_ABSENT:\\n"),
        ("require a file the package does not ship", text.replace("refe/p2r_carve.py splits/", "refe/p2r_carve_gone.py splits/"),
         "REQUIRED_FILE_ABSENT:refe/p2r_carve_gone.py"),
        ("drop the PDM-labeller setup check", drop_line(text, 'grep -q "^ZZPDM_SETUP_OK"'), "PDM_SETUP_NOT_CHECKED"),
        ("drop the carve check", drop_line(text, 'grep -q "^ZZCARVE_OK"'), "CARVE_NOT_CHECKED"),
        ("carve without the expected count", text.replace(' \\\n    --expect "${P2R_EXPECT_SERVED:-174912}" --json', ' \\\n    --json'),
         "CARVE_NOT_CHECKED"),
        ("delete the carve stage", re.sub(r"(?s)# ---- 3c\..*?stop_after carve\n", "", text), "MISSING:p2r_carve.py"),
        ("final failure swallowed", text.rstrip()[: -len("exit 1")] + "exit 0\n", "FINAL_FAILURE_NOT_PROPAGATED"),
    ]
    lg = re.search(r"# ---- 8b\..*?stop_after launch_gate\n", text, re.S)
    assert lg, "the launch-gate block moved"
    argl = ARGS_RX.search(text).group(0) + "\n"
    out += [
        ("delete the launch-gate stage", text.replace(lg.group(0), ""), "NO_LAUNCH_GATE_STAGE"),
        ("train.py without the token check", drop_line(text, "train.py is NOT started"), "TRAIN_WITHOUT_TOKEN"),
        ("preflight without the token check", drop_line(text, "the preflight and the trainer do not start"), "PREFLIGHT_WITHOUT_TOKEN"),
        ("token_ok always true", re.sub(r'(?s)(token_ok\(\) \{\n).*?(\n\}\n)', r'\1  return 0\2', text, count=1), "TOKEN_OK_NOT_VERIFYING"),
        ("token_ok greps the gate log", re.sub(r'(?s)(token_ok\(\) \{\n).*?(\n\}\n)', r'\1  grep -q ZZGATE-PASS "$P2/launch_gate.log"\2', text, count=1),
         "TOKEN_OK_NOT_VERIFYING"),
        ("token_ok verifies another argv", text.replace('--args "$ARGS" --pkg "$PKG" \\', '--args "--lr 1" --pkg "$PKG" \\'), "TOKEN_OK_NOT_VERIFYING"),
        ("token_ok forgets the profile", text.replace("verify --profile refe_p2 --token", "verify --token"), "TOKEN_OK_NOT_VERIFYING"),
        ("ARGS defined after the gate", text.replace(argl, "").replace("# ---- 9. preflight", argl + "# ---- 9. preflight", 1), "ARGS_AFTER_GATE"),
        ("ARGS defined twice", text.replace(argl, argl + argl, 1), "ARGS_DEFINED_TWICE"),
        ("the gate log is grepped for PASS", text.replace('  token_ok || refuse "the launch gate did not mint',
                                                       '  grep -q "^ZZGATE-PASS-" "$P2/launch_gate.log" || refuse "x"\n  token_ok || refuse "the launch gate did not mint'),
         "TOKEN_GREPPED"),
        ("the manifest is not required", text.replace(" SHIP_MANIFEST.json \\", " \\"), "SHIP_MANIFEST_NOT_REQUIRED"),
        ("the key lives in the shipped tree", text.replace('export TANITAD_GATE_KEY_FILE="$P2/launch_gate.key"', 'export TANITAD_GATE_KEY_FILE="$PKG/launch_gate.key"'),
         "KEY_NOT_OUTSIDE_TREE"),
        ("the gate child keeps fd 200", text.replace('> "$P2/launch_gate.log" 2>&1 200>&-', '> "$P2/launch_gate.log" 2>&1'), "FD200"),
        ("the stage is never marked", drop_line(text, "stage_mark launch_gate"), "LAUNCH_GATE_STAGE_NOT_MARKED"),
        ("train before the gate", text.replace(lg.group(0), "").replace("# ---- 9. preflight", "# ---- 9. preflight", 1) + "\n" + lg.group(0), "ORDER"),
    ]
    out.append(("trust restored labels without the audit",
                text.replace('elif [ -f "$PDMLOG" ] && "$PY" pdm_label_complete.py', 'elif [ -f "$PDMLOG" ] && true'),
                "RESTORED_LABELS_NOT_AUDITED"))
    blk = re.search(r"# ---- 7\..*?\nfi\n", text, re.S)
    assert blk, "the label-banking block moved"
    late = text.replace(blk.group(0), "") + "\n" + blk.group(0)
    out.append(("bank the labels AFTER training", late, "ORDER"))
    blk2 = re.search(r"# ---- 3\. .*?\nfi\n", text, re.S)
    assert blk2, "the restore block moved"
    out.append(("remove the restore stage", text.replace(blk2.group(0), ""), "MISSING"))
    blk3 = re.search(r"# ---- 3b\..*?\nfi\n", text, re.S)
    assert blk3, "the recover block moved"
    out.append(("remove the recover stage", text.replace(blk3.group(0), ""), "MISSING"))
    return out


# ------------------------------------------------------------------ behavioural helpers
def sh(helpers_path, script, env=None):
    py = sys.executable.replace("\\", "/")
    e = dict(os.environ, **(env or {}))
    r = subprocess.run([shutil.which("bash"), "-c", f'PY_C="{py}"; . "$0"; {script}', helpers_path],
                       capture_output=True, text=True, env=e)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def md5(b):
    return hashlib.md5(b).hexdigest()


def sha(b):
    return hashlib.sha256(b).hexdigest()


class FakeApi:
    """HF in memory. repos[repo][path] = bytes. `corrupt_sha` lies about model_final's LFS sha256; `deny` refuses commits."""

    def __init__(self, corrupt_sha=False, deny=False):
        self.repos, self.commits, self.corrupt_sha, self.deny = {}, [], corrupt_sha, deny

    def whoami(self):
        return {"name": "Sayood"}

    def create_repo(self, repo, private, repo_type, exist_ok):
        if self.deny:
            raise PermissionError("403 Forbidden")
        self.repos.setdefault(repo, {})

    def repo_info(self, repo, repo_type, expand=None):
        return SimpleNamespace(private=True, used_storage=0)

    def create_commit(self, repo, operations, commit_message, repo_type):
        self.commits.append([op.path_in_repo for op in operations])
        for op in operations:
            self.repos[repo][op.path_in_repo] = Path(op.path_or_fileobj).read_bytes()

    def list_repo_tree(self, repo, recursive, repo_type):
        for p, b in sorted(self.repos.get(repo, {}).items()):
            s = sha(b)
            if self.corrupt_sha and p.endswith("model_final.pt"):
                s = "0" * 64
            yield SimpleNamespace(path=p, size=len(b), lfs={"size": len(b), "sha256": s})

    def list_models(self, author, expand):
        return []

    list_datasets = list_spaces = list_models


def make_tar(path, members):
    with tarfile.open(path, "w") as tf:
        for name, data in members.items():
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))


def carve_checks(tmp):
    n = 0

    def row(log, i, ck, pad=0):
        return json.dumps({"kind": "onpolicy_set", "log_name": log, "token": f"t{i}", "step": i, "rank": 0, "ckpt_step": ck,
                           "label_version": 3, "traj": [[0, 0]] * pad}) + "\n"
    sets, stash, ho = tmp / "cv_sets", tmp / "cv_full", tmp / "cv_ho.txt"
    sets.mkdir()
    ho.write_text("H1\nH2\n", encoding="utf-8")
    # 3 training logs x 2 steps, a superseded version of one key, 2 held-out logs (one row too long for the head parse's window
    # to see its log_name -- it must still go OUT)
    late = '{"traj": ' + json.dumps([[0, 0]] * 400) + ', "kind": "onpolicy_set", "log_name": "H2", "token": "t9", "step": 9, "rank": 0, "ckpt_step": 5, "label_version": 3}\n'
    (sets / "onpolicy_r0_w0.jsonl").write_text(row("A", 0, 5) + row("A", 1, 5) + row("H1", 2, 5) + row("A", 0, 3), encoding="utf-8")
    (sets / "onpolicy_r0_w1.jsonl").write_text(row("B", 3, 5) + row("B", 4, 5) + late + row("C", 5, 5) + row("C", 6, 5), encoding="utf-8")
    full_before = {p.name: p.read_bytes() for p in sets.iterdir()}
    assert CV.log_name_of(late) == "H2"; n += 1                                  # the JSON fallback places the long row
    rep, why = CV.carve(str(sets), str(stash), str(ho), expect=6)
    assert why is None, why; n += 1
    assert (rep["served_full"], rep["served_in_heldout_logs"], rep["served_train"], rep["dropped_rows"]) == (8, 2, 6, 2), rep; n += 1
    assert {p.name: p.read_bytes() for p in stash.iterdir()} == full_before; n += 1          # the whole bank, byte for byte
    left = [json.loads(x)["log_name"] for f in sorted(sets.glob("onpolicy_*.jsonl")) for x in f.read_text().splitlines()]
    assert sorted(set(left)) == ["A", "B", "C"] and len(left) == 7; n += 1           # the superseded row stays: served_keys decides
    assert not list(sets.glob(".carve_*")); n += 1                                   # no part file left behind
    after = {p.name: p.read_bytes() for p in sets.glob("onpolicy_*.jsonl")}
    rep2, why2 = CV.carve(str(sets), str(stash), str(ho), expect=6)                  # re-run: rebuilt from the stash, identical
    assert why2 is None and {p.name: p.read_bytes() for p in sets.glob("onpolicy_*.jsonl")} == after; n += 1
    # an interrupted carve (original moved, carved copy not yet in place) is finished, not carved twice
    (sets / "onpolicy_r0_w1.jsonl").unlink()
    assert CV.carve(str(sets), str(stash), str(ho), expect=6)[1] is None and (sets / "onpolicy_r0_w1.jsonl").read_bytes() == after["onpolicy_r0_w1.jsonl"]; n += 1
    # refusals: the wrong expected count (the pod's 177,836-vs-174,912 case), an empty held-out list, a parse that misses held-out rows
    assert CV.carve(str(sets), str(stash), str(ho), expect=8)[1] == "served_train_6_expected_8"; n += 1
    empty = tmp / "cv_empty.txt"
    empty.write_text("\n", encoding="utf-8")
    assert CV.carve(str(sets), str(stash), str(empty), expect=6)[1] == "empty_heldout_list"; n += 1
    real = CV.log_name_of
    CV.log_name_of = lambda line: None                                               # MUTANT: nothing is ever held out
    try:
        assert CV.carve(str(sets), str(stash), str(ho), expect=6)[1] == "heldout_rows_survived"; n += 1
    finally:
        CV.log_name_of = real
    assert CV.carve(str(sets), str(stash), str(ho), expect=6)[1] is None; n += 1     # and the real parse restores the carve
    return n


def main() -> int:
    n = 0
    text = CHAIN.read_text(encoding="utf-8")
    ref_args = ARGS_RX.search(OLD.read_text(encoding="utf-8")).group(1)
    for f in ("p2r_pod.sh", "p2r_sync.py", "p2r_gate_prep.py"):
        assert b"\r" not in (HERE / f).read_bytes(), f + " must stay LF"
        n += 1
    # ---- 1. the real chain satisfies every property; the reference arguments are the previous chain's, byte for byte
    assert check_chain(text, ref_args) == [], check_chain(text, ref_args); n += 1
    assert ARGS_RX.search(text).group(1) == ref_args; n += 1
    # ---- 2. every property goes RED when broken
    for name, mut, want in mutations(text):
        got = check_chain(mut, ref_args)
        assert any(g.startswith(want) for g in got), f"mutation '{name}' did not raise {want}: {got}"
        n += 1
    # ---- 3a. gate_pass, extracted verbatim, both directions
    m = re.search(r"^gate_pass\(\).*?^PYEOF$", text, re.S | re.M)
    assert m, "gate_pass is no longer in the chain"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        h = tmp / "helpers.sh"
        h.write_text(m.group(0) + "\n}\n", encoding="utf-8", newline="\n")
        g = lambda gates, verdict="PASS": (tmp / "g.json").write_text(json.dumps({"verdict": verdict, "gates": gates}))  # noqa: E731
        full = {"G1_a": {"ok": True}, "G2_b": {"ok": True}, "G3_pdm_only": {"ok": True}, "G4_real": {"ok": True}}
        call = lambda: sh(str(h).replace("\\", "/"), f'gate_pass "{str(tmp / "g.json").replace(chr(92), "/")}"')[0]  # noqa: E731
        g(full); assert call() == 0; n += 1
        g(full, "FAIL"); assert call() != 0; n += 1
        g({**full, "G3_pdm_only": {"ok": False}}); assert call() != 0; n += 1       # the mutation arm: G3 inert -> ok false
        g({k: v for k, v in full.items() if not k.startswith("G4")}); assert call() != 0; n += 1   # --skip-trainer
        g({**full, "G2_b": {}}); assert call() != 0; n += 1                          # ok key absent is not a pass
        g({}); assert call() != 0; n += 1
        (tmp / "g.json").write_text("not json"); assert call() != 0; n += 1
        # ---- 3a2. token_ok, extracted verbatim, against a token minted by the gate's own mint(): matching argv -> 0; every other case -> refused
        import gate_token as GT
        m3 = TOKEN_OK_RX.search(text)
        assert m3, "token_ok is no longer in the chain"
        pkgroot = HERE.parent
        key = GT.load_key(tmp / "k" / "gate.key", forbid_under=[pkgroot])
        real = ARGS_RX.search(text).group(1)
        assert "/workspace/data/" in real, "the chain's data root moved: update the absent-root rewrite below"
        real = real.replace("/workspace/data/", "/zz_absent_test_root/data/")   # the token LOGIC on an absent data root, on EVERY host: on the pod the real paths exist and each mint/verify fingerprinted the real 413k-file listing + the 1.29 GB init ckpt (1,290 s pod vs ~10 s dev box). R40/BACKLOG 2026-10-16
        argv = GT.argv_from_string(real)
        lgdir = tmp / "lg"

        def mint_token(verdict_checks, argv_=argv):
            shutil.rmtree(lgdir, ignore_errors=True)
            return GT.mint(profile="refe_p2", commit="a" * 40, tree_man=GT.tree_manifest(pkgroot), argv=argv_, trainer="refe/train.py",
                           data=GT.data_manifest(argv_, ("--targets", "--scorer-targets", "--images", "--calib", "--init-from",
                                                         "--onpolicy-targets", "--pdm-labels")), checks=verdict_checks, reasons=[], key=key, out_dir=lgdir)
        h3 = tmp / "helpers3.sh"
        h3.write_text(m3.group(0), encoding="utf-8", newline="\n")
        p2dir = tmp / "p2dir"
        p2dir.mkdir(exist_ok=True)

        def call_token_ok(args_str, keyfile=None):
            env = {"PY": sys.executable.replace("\\", "/"), "PKG": str(pkgroot).replace("\\", "/"), "LG": str(lgdir).replace("\\", "/"),
                   "P2": str(p2dir).replace("\\", "/"), "ARGS": args_str, "TANITAD_GATE_KEY_FILE": str(keyfile or (tmp / "k" / "gate.key")).replace("\\", "/"),
                   "PYTHONIOENCODING": "utf-8"}
            return sh(str(h3).replace("\\", "/"), "token_ok", env)[0]
        mint_token({"G-HYG": {"status": "PASS"}})
        assert call_token_ok(real) == 0; n += 1                                                       # the matching argv: the trainer may start
        assert call_token_ok(real.replace("--lr 5e-5", "--lr 1e-4")) != 0; n += 1                    # a changed lever
        assert call_token_ok(real + " --resume") != 0; n += 1                                         # an extra flag
        assert call_token_ok(real, keyfile=tmp / "k" / "other.key") != 0; n += 1                      # another host's key (absent)
        tokp = next(lgdir.glob("PASS_*.json"))
        tok = json.loads(tokp.read_text(encoding="utf-8"))
        tok["binding"]["argv_sha256"] = "0" * 64                                                       # an edited token
        tokp.write_text(json.dumps(tok), encoding="utf-8")
        assert call_token_ok(real) != 0; n += 1
        mint_token({"G-HYG": {"status": "FAIL"}})                                                      # a FAIL token is a file called FAIL_*.json: no PASS token exists
        assert not list(lgdir.glob("PASS_*.json")) and call_token_ok(real) != 0; n += 1
        shutil.rmtree(lgdir, ignore_errors=True)
        assert call_token_ok(real) != 0; n += 1                                                       # no token at all
        # ---- 3b. served_count, extracted verbatim, on a tiny sets dir (skipped, loudly, if the module cannot import here)
        m2 = re.search(r"^served_count\(\).*?^PYEOF$", text, re.S | re.M)
        assert m2
        h2 = tmp / "helpers2.sh"
        h2.write_text(m2.group(0) + "\n}\n", encoding="utf-8", newline="\n")
        sets = tmp / "sets"
        sets.mkdir()
        rows = [{"kind": "onpolicy_set", "log_name": "L", "token": "t", "step": s, "rank": 0, "ckpt_step": c}
                for s, c in ((1, 10), (1, 20), (2, 10))]
        (sets / "onpolicy_a.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        rc, out, err = sh(str(h2).replace("\\", "/"), f'served_count "{str(sets).replace(chr(92), "/")}"',
                          {"PKG": str(HERE.parent).replace("\\", "/")})
        if rc == 0:
            assert out == "2", out; n += 1          # keys (1) and (2): the newest ckpt per key
        else:
            print("ZZTEST_SKIP served_count: " + (err.splitlines() or ["?"])[-1][:200])
        # ---- 3c. member validation and extraction
        for name, ok in (("train_grow/a.json", True), ("./train_grow/b/c.json", True), ("train_grow", True),
                         ("../escape.txt", False), ("/abs/x", False), ("train_grow/../../x", False),
                         ("other_root/a", False), ("C:/win/x", False)):
            ti = tarfile.TarInfo(name)
            ti.type = tarfile.DIRTYPE if name == "train_grow" else tarfile.REGTYPE
            assert S.member_verdict(ti, {"train_grow"})[0] is ok, name; n += 1
        link = tarfile.TarInfo("train_grow/ln")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        assert not S.member_verdict(link, {"train_grow"})[0]; n += 1
        good, evil = tmp / "good.tar", tmp / "evil.tar"
        make_tar(good, {"train_grow/calib_table.json": b"{}", "train_grow/x/y.txt": b"yy"})
        make_tar(evil, {"train_grow/ok.txt": b"1", "../escape.txt": b"2"})
        d1 = tmp / "d1"
        assert S.safe_extract(good, d1, {"train_grow"}) == (2, 4) and (d1 / "train_grow/x/y.txt").read_bytes() == b"yy"; n += 1
        d2 = tmp / "d2" / "inner"
        try:
            S.safe_extract(evil, d2, {"train_grow"})
            raise AssertionError("an escaping member was extracted")
        except ValueError as e:
            assert "unsafe_path" in str(e); n += 1
        assert not (tmp / "d2" / "escape.txt").exists(); n += 1
        # ---- 3d. pick_remote
        sizes = {"a/model_final.pt": 50, "b/other.bin": 50, "c/model_final.pt": 51}
        assert S.pick_remote(sizes, "model_final", 50, [".pt"]) == ["a/model_final.pt"]; n += 1
        assert S.pick_remote({"zz.bin": 50}, "model_final", 50, [".pt"]) == ["zz.bin"]; n += 1   # size-only fallback
        assert S.pick_remote(sizes, "model_final", 99, [".pt"]) == []; n += 1
        # ---- 3e. restore against the fake API: success, then a tampered download, then a missing file
        E._add_op = lambda p, f: SimpleNamespace(path_in_repo=p, path_or_fileobj=str(f))
        mf = b"M" * 64
        tg, st = tmp / "tg.tar", tmp / "st.tar"
        make_tar(tg, {"train_grow/calib_table.json": b"{}", "train_grow/scorer_targets_r0.jsonl": b"1\n",
                      "train_grow/targets_rank0.jsonl": b"2\n"})
        # the REAL on-policy bank's four roots (MEASURED 2026-10-11 on the pod; a sets/-only fixture let the pod refuse)
        make_tar(st, {"sets/onpolicy_r0_w0.jsonl": b"3\n", "sets/pad.bin": b"p" * 12000,      # a different size from tg
                      "logs/supervisor.log": b"l", "pids/label_r0_w0.pid": b"1", "smoke_archive/onpolicy_r0_smoke.jsonl": b"s"})
        assert S.EXTRACT["onpolicy_sets_logs_full"][1] == {"sets", "logs", "pids", "smoke_archive"}; n += 1
        assert not S.member_verdict(tarfile.TarInfo("evil/x"), S.EXTRACT["onpolicy_sets_logs_full"][1])[0]; n += 1
        blobs = {"model_final.pt": mf, "refe_train_grow.tar": tg.read_bytes(), "onpolicy_sets_logs_full.tar": st.read_bytes()}
        pins = {"model_final": (md5(mf), len(mf), [".pt"]),
                "refe_train_grow": (md5(blobs["refe_train_grow.tar"]), len(blobs["refe_train_grow.tar"]), [".tar"]),
                "onpolicy_sets_logs_full": (md5(blobs["onpolicy_sets_logs_full.tar"]),
                                            len(blobs["onpolicy_sets_logs_full.tar"]), [".tar"])}

        def fake_api(tamper=None, omit=None):
            api = FakeApi()
            api.repos["m"] = {"model_final.pt": mf, "snapshots/s1.pt": b"s"}
            api.repos["d"] = {k: v for k, v in blobs.items() if k != "model_final.pt"}
            if omit:
                api.repos["d"].pop(omit)
            return api

        def fetcher(tamper=None):
            def fetch(repo, rtype, path, local_dir):
                data = (fake.repos[repo][path])
                if tamper and path.endswith(tamper):
                    data = bytes([data[0] ^ 1]) + data[1:]                  # same size, one flipped bit
                p = Path(local_dir) / path
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(data)
                return p
            return fetch

        fake = fake_api()
        w1 = tmp / "w1"
        ok, why = S.restore(fake, w1, fetch=fetcher(), pins=pins, model_repo="m", data_repo="d")
        assert ok, why
        assert (w1 / S.INIT_REL).read_bytes() == mf and (w1 / "refe_navtrain/train_grow/calib_table.json").is_file()
        assert (w1 / "refe_navtrain/onpolicy/sets/onpolicy_r0_w0.jsonl").is_file(); n += 3
        assert (w1 / "refe_navtrain/onpolicy/logs/supervisor.log").is_file(); n += 1
        ok, why = S.restore(fake, w1, fetch=fetcher(), pins=pins, model_repo="m", data_repo="d")   # idempotent
        assert ok, why; n += 1
        w2 = tmp / "w2"
        ok, why = S.restore(fake, w2, fetch=fetcher(tamper=".tar"), pins=pins, model_repo="m", data_repo="d")
        assert not ok and why.startswith("md5_mismatch:refe_train_grow"), why; n += 1
        assert not (w2 / "refe_navtrain/train_grow").exists(); n += 1                   # nothing extracted from a bad file
        w3 = tmp / "w3"
        ok, why = S.restore(fake_api(omit="refe_train_grow.tar"), w3, fetch=fetcher(), pins=pins, model_repo="m",
                            data_repo="d")
        assert not ok and why.startswith("remote_not_unique:refe_train_grow"), why; n += 1
        # ---- 3f. tick / final / watch
        run, logs, ob = tmp / "run", tmp / "logs", tmp / "ob"
        (logs / "gate").mkdir(parents=True)
        run.mkdir()
        (run / "metrics.jsonl").write_text('{"a":1}\n')
        (run / "ckpt_last.pt").write_bytes(b"c" * 100)
        (run / "snap_epoch001.pt").write_bytes(b"s" * 50)
        (logs / "chain.log").write_text("start\n")
        (logs / "gate" / "gate.json").write_text('{"verdict":"PASS"}')
        a = SimpleNamespace(run=str(run), logs=str(logs), outbox=str(ob), repo="out", chain_log=str(tmp / "elsewhere.log"),
                            max_push_bytes=1e9, tail=[], stop_file=str(tmp / "stop"), interval_min=0.2, heavy_min=0.0)
        api = FakeApi()
        assert S.tick(api, a, 1, False)[0]
        assert set(api.repos["out"]) == {"run/metrics.jsonl", "logs/chain.log", "logs/gate/gate.json"}; n += 1   # no checkpoints
        assert S.tick(api, a, 2, True)[0] and "run/ckpt_last.pt" in api.repos["out"] and "run/snap_epoch001.pt" in api.repos["out"]; n += 1
        k = len(api.commits)
        assert S.tick(api, a, 3, True) == (True, "nothing_changed") and len(api.commits) == k; n += 1
        with open(run / "metrics.jsonl", "a") as fh:
            fh.write('{"a":2}\n')
        assert S.tick(api, a, 4, True)[0] and api.commits[-1] == ["run/metrics.jsonl"]; n += 1                  # only the change
        (run / "ckpt_last.pt").write_bytes(b"d" * 100)                       # SAME size, new content
        os.utime(run / "ckpt_last.pt", ns=(1, 1_700_000_000_000_000_000))
        assert S.tick(api, a, 5, True)[0] and api.repos["out"]["run/ckpt_last.pt"] == b"d" * 100; n += 1
        assert api.commits[-1] == ["run/ckpt_last.pt"]; n += 1
        # final: refuses before done, refuses a disagreeing summary, passes with a matching remote sha, fails on a lying one
        ok, why = S.final(FakeApi(), a)
        assert not ok and why.startswith("summary_unreadable"), why; n += 1
        fin = b"F" * 80
        (run / "model_final.pt").write_bytes(fin)
        (run / "summary.json").write_text(json.dumps({"done": False}))
        assert S.final(FakeApi(), a)[1] == "summary_not_done"; n += 1
        (run / "summary.json").write_text(json.dumps({"done": True, "model_final_sha256": "ab" * 32}))
        assert S.final(FakeApi(), a)[1] == "local_sha_disagrees_with_summary"; n += 1
        (run / "summary.json").write_text(json.dumps({"done": True, "model_final_sha256": sha(fin)}))
        good_api = FakeApi()
        ok, why = S.final(good_api, a)
        assert ok and "sha256-verified" in why, why
        assert good_api.repos["out"]["run/model_final.pt"] == fin and "run/ckpt_last.pt" in good_api.repos["out"]; n += 2
        ok, why = S.final(FakeApi(corrupt_sha=True), a)
        assert not ok and why.startswith("remote_sha_mismatch"), why; n += 1                                    # the regression arm
        ok, why = S.final(FakeApi(deny=True), a)
        assert not ok and why.startswith("push_failed"), why; n += 1
        # watch: five failed pushes in a row are LOGGED and never raise or stop the run
        calls = {"n": 0}
        real_sleep = S.time.sleep

        def fake_sleep(_s):
            calls["n"] += 1
            if calls["n"] >= 18:
                Path(a.stop_file).write_text("x")
        S.time.sleep = fake_sleep
        a.outbox = str(tmp / "ob_watch")                     # fresh state: every file is "changed", so every tick pushes
        try:
            assert S.watch(FakeApi(deny=True), a) == 0
        finally:
            S.time.sleep = real_sleep
        assert open(a.chain_log, encoding="utf-8").read().count("ZZSYNC_DEGRADED") == 1; n += 2
        # pdm-push: the labels go off the pod as ONE tar + a MANIFEST whose sha256 is the tar's, under the tag the chain greps
        lab_dir = tmp / "labels"
        lab_dir.mkdir()
        (lab_dir / "pdm_r0.jsonl").write_text('{"key":[1]}\n' * 3)
        (lab_dir / "pdm_r1.jsonl").write_text('{"key":[2]}\n' * 2)
        pa = SimpleNamespace(labels=str(lab_dir), repo="out", chain_log=str(tmp / "pdm_chain.log"),
                             outbox=str(tmp / "ob_pdm"), max_push_bytes=1e9)
        papi = FakeApi()
        ok, why = S.pdm_push(papi, pa)
        assert ok, why
        man = json.loads(papi.repos["out"]["pdm_labels/pdm_train_labels.MANIFEST.json"])
        assert man["tar_sha256"] == sha(papi.repos["out"]["pdm_labels/pdm_train_labels.tar.gz"]) and man["n_files"] == 2; n += 2
        assert " ZZEGRESS_OK p2r_pdm_labels " in open(pa.chain_log, encoding="utf-8").read(); n += 1
        (lab_dir / "pdm_r0.jsonl").unlink(); (lab_dir / "pdm_r1.jsonl").unlink()
        assert S.pdm_push(FakeApi(), pa) == (False, "no_pdm_files"); n += 1
        os.environ.pop("HF_TOKEN", None)
        assert S.main(["tick", "--run", "r", "--logs", "l", "--chain-log", "c", "--outbox", "o"]) == 2; n += 1   # no token: refuse
        # recover: a NEW pod picks up what the lost one pushed -- labels (+ the relabel log), ckpt_last.pt and the run files
        rapi = FakeApi()
        assert S.recover(rapi, tmp / "rw0", tmp / "rr0", repo="out", fetch=lambda *x: None) == (True, "labels=no ckpt=no"); n += 1   # first run
        pdm_log = tmp / "pdm_train.log"
        pdm_log.write_text("served sets: 5\nZZRELABEL_PDM_DONE 5 0 0\n")
        lab_dir.mkdir(exist_ok=True)
        (lab_dir / "pdm_r0.jsonl").write_text('{"key":[1]}\n' * 3)
        pa2 = SimpleNamespace(labels=str(lab_dir), log=str(pdm_log), repo="out", chain_log=str(tmp / "rec.log"),
                              outbox=str(tmp / "ob_rec"), max_push_bytes=1e9)
        assert S.pdm_push(rapi, pa2)[0]
        ra = SimpleNamespace(run=str(run), logs=str(logs), outbox=str(tmp / "ob_rec_run"), repo="out",
                             chain_log=str(tmp / "rec.log"), max_push_bytes=1e9, tail=[])
        assert S.tick(rapi, ra, 1, True)[0]

        def rfetch(repo, rtype, path, local_dir, tamper=None):
            data = rapi.repos[repo][path]
            if tamper and path.endswith(tamper):
                data = bytes([data[0] ^ 1]) + data[1:]
            p = Path(local_dir) / path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
            return p
        w, r_ = tmp / "rw1", tmp / "rr1"
        ok, why = S.recover(rapi, w, r_, repo="out", fetch=rfetch)
        assert (ok, why) == (True, "labels=yes ckpt=yes"), why
        assert (w / "refe_sft2/train_pdm/pdm_r0.jsonl").read_bytes() == (lab_dir / "pdm_r0.jsonl").read_bytes(); n += 1
        assert (w / "refe_sft2/pdm_train.log").read_text() == pdm_log.read_text(); n += 1      # the audit can be re-run
        assert (r_ / "ckpt_last.pt").read_bytes() == (run / "ckpt_last.pt").read_bytes() and (r_ / "metrics.jsonl").is_file(); n += 2
        ok, why = S.recover(rapi, w, r_, repo="out", fetch=lambda *x: None)             # idempotent: nothing to fetch again
        assert (ok, why) == (True, "labels=no ckpt=no"), why; n += 1
        ok, why = S.recover(rapi, tmp / "rw2", tmp / "rr2", repo="out", fetch=lambda *a_: rfetch(*a_, tamper="tar.gz"))
        assert not ok and why == "labels_tar_mismatch", why; n += 1                   # a corrupted download is a FAILURE
        ok, why = S.recover(rapi, tmp / "rw3", tmp / "rr3", repo="out", fetch=lambda *a_: rfetch(*a_, tamper="ckpt_last.pt"))
        assert not ok and why == "run_file_mismatch:run/ckpt_last.pt", why; n += 1
        del rapi.repos["out"]["pdm_labels/pdm_train.log"]                              # labels without their log: recompute
        ok, why = S.recover(rapi, tmp / "rw4", tmp / "rr4", repo="out", fetch=rfetch)
        assert (ok, why) == (True, "labels=no ckpt=yes"), why; n += 1
        # ---- 3g. p2r_gate_prep: labelled sets + WITHHELD sets, so G3's mutation is not inert
        gs, gp = tmp / "gsets", tmp / "gpdm"
        gs.mkdir(); gp.mkdir()

        def srow(i, ck):
            return json.dumps({"kind": "onpolicy_set", "log_name": f"L{i}", "token": f"t{i}", "step": i, "rank": 0,
                               "ckpt_step": ck, "traj": [[0, 0]]}) + "\n"
        (gs / "onpolicy_a.jsonl").write_text("".join(srow(i, 5) for i in range(8)) + srow(0, 9), encoding="utf-8")
        (gp / "pdm_a.jsonl").write_text("".join(json.dumps({"key": [f"L{i}", f"t{i}", i, 0], "ckpt_step": 5 if i else 9,
                                                              "nc": [1]}) + "\n" for i in range(6)), encoding="utf-8")
        o_s, o_p = tmp / "o_sets", tmp / "o_pdm"
        rc = GP.main(["--sets", str(gs), "--pdm", str(gp), "--out-sets", str(o_s), "--out-pdm", str(o_p),
                      "--n-labelled", "3", "--n-unlabelled", "2"])
        assert rc == 0; n += 1
        lab = {tuple(json.loads(x)["key"][:3]) for x in (o_p / "pdm_gate.jsonl").read_text().splitlines()}
        assert lab == {("L2", "t2", 2), ("L3", "t3", 3), ("L4", "t4", 4)}; n += 1
        kept = [json.loads(x) for x in (o_s / "onpolicy_gate.jsonl").read_text().splitlines()]
        unl = {(r["log_name"], r["token"], r["step"]) for r in kept} - lab
        assert unl == {("L0", "t0", 0), ("L1", "t1", 1)}; n += 1       # the withheld sets: first in the stream, no gate label
        full = {tuple(json.loads(x)["key"][:3]) for x in (gp / "pdm_a.jsonl").read_text().splitlines()}
        assert unl <= full; n += 1                                       # ... although the REAL label dir does hold them
        assert sum(1 for r in kept if r["log_name"] == "L0") == 2; n += 1       # every version of a withheld key is kept
        # a SMALL bank in which every set has a label (the sandbox case): sets are withheld first, so G3 still gets some
        gp2 = tmp / "gpdm_all"
        gp2.mkdir()
        (gp2 / "pdm_a.jsonl").write_text("".join(json.dumps({"key": [f"L{i}", f"t{i}", i, 0], "ckpt_step": 5,
                                                              "nc": [1]}) + "\n" for i in range(8)), encoding="utf-8")
        assert GP.main(["--sets", str(gs), "--pdm", str(gp2), "--out-sets", str(tmp / "x_s"), "--out-pdm", str(tmp / "x_p"),
                        "--n-labelled", "8", "--n-unlabelled", "2"]) == 0; n += 1
        assert len((tmp / "x_p" / "pdm_gate.jsonl").read_text().splitlines()) == 6; n += 1      # 8 labels - the 2 withheld
        # one set only: nothing can be both withheld and labelled -> refuse (G3's mutation arm would be inert)
        g1, p1 = tmp / "g1", tmp / "p1"
        g1.mkdir(); p1.mkdir()
        (g1 / "onpolicy_a.jsonl").write_text(srow(0, 5), encoding="utf-8")
        (p1 / "pdm_a.jsonl").write_text(json.dumps({"key": ["L0", "t0", 0, 0], "ckpt_step": 5, "nc": [1]}) + "\n", encoding="utf-8")
        assert GP.main(["--sets", str(g1), "--pdm", str(p1), "--out-sets", str(tmp / "y_s"), "--out-pdm", str(tmp / "y_p"),
                        "--n-labelled", "3", "--n-unlabelled", "1"]) == 1; n += 1
        # ---- 3i. tar extraction on NFS as root: --no-same-owner on every pod script that extracts, and the check goes RED without it
        for f in ("code/pod_dataprep.sh", "refe/p6_pod.sh", "code/pod_teacher_env.sh", "refe/p2r_pod.sh"):
            src = (HERE.parent / f).read_text(encoding="utf-8")
            assert tar_owner_violations(src) == [], (f, tar_owner_violations(src)); n += 1
        # NAVSIM v1.1 at its pinned commit is installed by the env stage (the labeller imports it; 2026-10-11 it was missing)
        env_sh = (HERE.parent / "code/pod_teacher_env.sh").read_text(encoding="utf-8")
        assert 'NAVSIM_COMMIT="3e8291bfa89ff247231e0227778840cd0a036896"' in env_sh; n += 1
        assert re.search(r'uv pip install -q --python "\$PY" --no-deps -e "\$NAVSIM_DIR" \|\| fail', env_sh); n += 1
        dp = (HERE.parent / "code/pod_dataprep.sh").read_text(encoding="utf-8")
        assert tar_owner_violations(dp.replace(" --no-same-owner", "")); n += 1
        # ---- 3h. p2r_carve: the held-out logs OUT, in place, re-runnable, and refused when the count is not the expected one
        n += carve_checks(tmp)
    print(f"ZZTEST_PASS {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
