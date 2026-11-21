"""Control-flow tests for snap40k_r8.sh against a FAKE "Thor" (a directory with a train.log and a ckpt.pt; stub ssh / scp).  NEW file.

    python -m pytest -q -p no:cacheprovider test_snap40k_r8.py      (needs Git-for-Windows `sh`)

The fake ssh runs the remote command LOCALLY with `sh -c`, so the real probe / stat / md5sum pipeline is exercised; the fake scp is a `cp` that can be told to corrupt the copy, touch
the source mid-copy, or let the next ckpt line appear mid-copy.  Everything is asserted on ARTIFACTS (the destination file, its md5 sidecar, the opaque markers); nothing here reads an
exit code as evidence.  Each guard has its REGRESSION arm.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SH = shutil.which("sh") or "C:/Program Files/Git/bin/sh.exe"
pytestmark = pytest.mark.skipif(not Path(SH).exists() and not shutil.which("sh"), reason="no sh")
PAYLOAD = b"rolling-ckpt-payload-" * 1000


def msys(p: Path) -> str:
    s = str(p).replace("\\", "/")
    return "/" + s[0].lower() + s[2:] if len(s) > 1 and s[1] == ":" else s


FAKE_SSH = '#!/bin/sh\nif [ "$FAKE_SSH_DEAD" = "1" ]; then exit 255; fi\nexec sh -c "$*"\n'
FAKE_SCP = r'''#!/bin/sh
# usage: fake_scp.sh <host:/path> <dest>
src="${1#*:}"; dest="$2"
cp "$src" "$dest"
[ "$FAKE_SCP_CORRUPT" = "1" ] && printf 'X' >> "$dest"
[ "$FAKE_SCP_FLIP" = "1" ] && printf 'Z' | dd of="$dest" bs=1 seek=7 count=1 conv=notrunc 2>/dev/null
[ "$FAKE_SCP_TOUCH_REMOTE" = "1" ] && touch -d "@$(( $(stat -c %Y "$src") + 7 ))" "$src"
[ -n "$FAKE_SCP_APPEND_LOG" ] && echo "[v3:hier] ckpt step $FAKE_SCP_APPEND_LOG -> ckpt.pt" >> "$(dirname "$src")/train.log"
exit 0
'''


@pytest.fixture()
def fake(tmp_path):
    root = tmp_path / "fk"
    thor = root / "thor"
    thor.mkdir(parents=True)
    (thor / "ckpt.pt").write_bytes(PAYLOAD)
    (thor / "train.log").write_text("[v3:hier] ckpt step 38000 -> ckpt.pt\n[v3:eval] step 38000 loss 1\n[v3:hier] ckpt step 39500 -> ckpt.pt\n", encoding="utf-8")
    (root / "fake_ssh.sh").write_text(FAKE_SSH, encoding="utf-8", newline="\n")
    (root / "fake_scp.sh").write_text(FAKE_SCP, encoding="utf-8", newline="\n")
    kit = root / "kit"
    (kit / "ckpt").mkdir(parents=True)
    return root, thor, kit


def run_snap(fake, wait=True, **extra):
    root, thor, kit = fake
    base = {"SNAP_EXPECT_SIZE": str(len(PAYLOAD))}
    base.update(extra)
    env = dict(os.environ, SNAP_SSH=f"sh {msys(root / 'fake_ssh.sh')}", SNAP_SCP=f"sh {msys(root / 'fake_scp.sh')}",
               SNAP_RDIR=msys(thor), SNAP_SRC_SPEC=f"fakehost:{msys(thor)}/ckpt.pt", SNAP_KIT=msys(kit), SNAP_POLL_S="1", SNAP_MAX_WAIT_S="40",
               SNAP_MIN_FREE_KB="1", SNAP_MAX_SSH_FAIL="3", **base)
    cmd = [SH, str(HERE / "snap40k_r8.sh")]
    if wait:
        return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)


def markers(kit: Path):
    p = kit / "chain" / "snap40k.log"
    return [ln.split()[0] for ln in p.read_text(encoding="utf-8").splitlines() if ln.startswith("ZZS40")] if p.exists() else []


def log_text(kit: Path):
    p = kit / "chain" / "snap40k.log"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def append(thor: Path, step: int):
    with open(thor / "train.log", "a", encoding="utf-8") as f:
        f.write(f"[v3:hier] ckpt step {step} -> ckpt.pt\n")


def test_waits_then_snapshots_when_the_40000_line_appears_and_verifies_by_md5(fake):
    root, thor, kit = fake
    pr = run_snap(fake, wait=False)
    time.sleep(2.5)
    assert markers(kit)[:1] == ["ZZS40STARTZZ"] and "ZZS40SEENZZ" not in markers(kit)                       # 39,500 is not 40,000: still waiting
    append(thor, 40000)
    pr.wait(timeout=60)
    dest = kit / "ckpt" / "ckpt_full_40000.pt"
    assert dest.read_bytes() == PAYLOAD and not (kit / "ckpt" / "ckpt_full_40000.pt.part").exists()
    assert (kit / "ckpt" / "ckpt_full_40000.pt.md5").read_text().split()[0] == hashlib.md5(PAYLOAD).hexdigest()
    assert '"step": 40000' in (kit / "ckpt" / "ckpt_full_40000.pt.provenance.json").read_text()
    m = markers(kit)
    assert m[0] == "ZZS40STARTZZ" and "ZZS40SEENZZ" in m and m[-1] == "ZZS40DONEZZ"


def test_a_later_ckpt_line_aborts_loudly_and_copies_nothing(fake):
    root, thor, kit = fake
    append(thor, 40000)
    append(thor, 40500)                                                                                    # REGRESSION arm: the 40,000 state is already overwritten
    p = run_snap(fake)
    assert "ZZS40ABORTLATERZZ" in markers(kit) and "ZZS40SEENZZ" not in markers(kit)
    assert not (kit / "ckpt" / "ckpt_full_40000.pt").exists()


def test_corrupted_copy_is_refused_by_the_md5_comparison(fake):
    root, thor, kit = fake
    append(thor, 40000)
    p = run_snap(fake, FAKE_SCP_FLIP="1")                                                                    # same size, one byte different
    assert "ZZS40MD5MISMATCHZZ" in markers(kit) and not (kit / "ckpt" / "ckpt_full_40000.pt").exists()
    assert not (kit / "ckpt" / "ckpt_full_40000.pt.part").exists()                                           # no half-file left under a plausible name


def test_truncated_copy_is_refused_by_the_size_check(fake):
    root, thor, kit = fake
    append(thor, 40000)
    p = run_snap(fake, FAKE_SCP_CORRUPT="1")                                                                 # one extra byte
    assert ("ZZS40SIZEMISMATCHZZ" in markers(kit)) and not (kit / "ckpt" / "ckpt_full_40000.pt").exists()


def test_remote_file_rewritten_during_the_copy_is_refused(fake):
    root, thor, kit = fake
    append(thor, 40000)
    p = run_snap(fake, FAKE_SCP_TOUCH_REMOTE="1")                                                            # the trainer started rewriting ckpt.pt (mtime moved)
    assert "ZZS40CHANGEDZZ" in markers(kit) and not (kit / "ckpt" / "ckpt_full_40000.pt").exists()


def test_a_ckpt_line_that_appears_during_the_copy_is_too_late(fake):
    root, thor, kit = fake
    append(thor, 40000)
    p = run_snap(fake, FAKE_SCP_APPEND_LOG="40500")
    assert "ZZS40TOOLATEZZ" in markers(kit) and not (kit / "ckpt" / "ckpt_full_40000.pt").exists()


def test_wrong_remote_size_aborts_before_copying(fake):
    root, thor, kit = fake
    append(thor, 40000)
    p = run_snap(fake, SNAP_EXPECT_SIZE=str(len(PAYLOAD) + 1))                                                # a partial / different file
    assert "ZZS40BADSIZEZZ" in markers(kit) and not (kit / "ckpt" / "ckpt_full_40000.pt").exists()


def test_dead_ssh_is_loud_not_silent(fake):
    root, thor, kit = fake
    p = run_snap(fake, FAKE_SSH_DEAD="1")
    m = markers(kit)
    assert "ZZS40SSHFAILZZ" in m and m[-1] == "ZZS40SSHDEADZZ" and not (kit / "ckpt" / "ckpt_full_40000.pt").exists()


def test_already_snapshotted_is_a_noop(fake):
    root, thor, kit = fake
    (kit / "ckpt" / "ckpt_full_40000.pt").write_bytes(b"x")
    (kit / "ckpt" / "ckpt_full_40000.pt.md5").write_text("d41d8cd98f00b204e9800998ecf8427e  ckpt_full_40000.pt\n", encoding="utf-8")
    p = run_snap(fake, FAKE_SSH_DEAD="1")                                                                    # must not even look at Thor
    assert markers(kit)[-1] == "ZZS40ALREADYZZ" and (kit / "ckpt" / "ckpt_full_40000.pt").read_bytes() == b"x"


def test_the_probe_output_never_contains_the_searched_text(fake):
    """The CLAUDE.md trap: a client-side filter whose pattern is in the echoed command matches its own echo.  The remote prints ONLY ZZ<n>ZZ."""
    root, thor, kit = fake
    probe = (HERE / "snap40k_r8.sh").read_text(encoding="utf-8").split("PROBE=\"", 1)[1].split("\"\n", 1)[0]
    probe = probe.replace("$RDIR", msys(thor)).replace("\\$", "$")
    out = subprocess.run([SH, msys(root / "fake_ssh.sh"), probe], capture_output=True, text=True, timeout=60).stdout.strip()
    assert out == "ZZ39500ZZ" and "ckpt step" not in out


def test_script_reads_thor_only_and_never_writes_there():
    """Static guard on the READ-ONLY rule: the only remote commands are grep/cut/sort/tail/stat/md5sum; no python, no pgrep/pkill, no rm/mv/cp/touch/tee/redirect on the remote side."""
    txt = (HERE / "snap40k_r8.sh").read_text(encoding="utf-8")
    remote_lines = [ln for ln in txt.splitlines() if "$THOR" in ln or ln.startswith("PROBE=")]
    assert remote_lines
    for ln in remote_lines:
        for bad in ("pgrep", "pkill", "python", " rm ", " mv ", " cp ", "touch", "tee ", "kill", "> $RDIR", ">$RDIR"):
            assert bad not in ln, (bad, ln)
    assert "ssh -n" in txt                                                                                    # -n always
