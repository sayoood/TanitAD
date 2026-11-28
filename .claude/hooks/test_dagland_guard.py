"""Tests for dagland_guard.py: the 8 recorded stray-file commands must be REFUSED, the real append must PASS.

Run: C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q E:/Projects/TanitAD/.claude/hooks/test_dagland_guard.py
The "live" cases read the real dagland folder (os.path.exists only, nothing is written there).
"""
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dagland_guard as G  # noqa: E402

D = "E:/Projects/TanitAD-artifacts/dagland"
LIVE = D + "/LOOP_STATE_P6_2026-11-06.md"

# Reconstructed from memory/no-rm-in-state-appends.md (one per recorded incident pattern).
RECORDED = [
    f"cat >> {D}/LOOP_STATE_2026-11-04.md <<'EOF'\nEOF\nrm -f {D}/LOOP_STATE_2026-11-04.md",           # 11-04 01:06 / ~17:00
    f"cat >> {D}/LOOP_STATE_2026-11-04_b.md <<'EOF'\nEOF",                                             # 11-04 21:27 empty heredoc, new name
    f"cat >> {D}/LOOP_STATE_P6_2026-11-09.md <<'EOF'\nEOF\nrm -f {D}/LOOP_STATE_P6_2026-11-09.md",      # 11-09 12:16Z
    f"ssh pod 'rm -f /workspace/x.tar'; cat >> {D}/LOOP_STATE_P6_2026-11-09_note_never.md </dev/null", # 11-09 ~15:00Z
    f"cat >> {D}/LOOP_STATE_P6_2026-11-10_x.md </dev/null; mv {D}/LOOP_STATE_P6_2026-11-10_x.md /tmp",  # 11-09/10 night (x4)
    f"cat >> {D}/LOOP_STATE_P6_2026-11-10.md.x 2>/dev/null; true",                                    # 11-10 03:13
    f"cat >> {D}/LOOP_STATE_P6_2026-11-10.md <<'EOF'\nEOF\nrm -f {D}/LOOP_STATE_P6_2026-11-10.md",      # 11-10 10:55Z
    f"cat >> {D}/LOOP_STATE_P6_2026-11-10.md.never 2>/dev/null </dev/null; true",                     # 11-10 11:17Z
    f"cat >> {D}/LOOP_STATE_P6_2026-11-10_unused_never 2>/dev/null </dev/null; true",                 # 11-10 13:52Z
]

REAL_APPEND = (f"cat >> {LIVE} <<'EOF'\n\n## 2026-11-11 block\n- never rm a stray; moved "
               f"{D}/LOOP_STATE_P6_2026-11-10.md.x aside (mv by hand)\n- Remove-Item was not used\nEOF")


@pytest.mark.skipif(not os.path.exists(LIVE), reason="live dagland file not on this box")
@pytest.mark.parametrize("cmd", RECORDED)
def test_recorded_commands_refused(cmd):
    block, why = G.decide(cmd)
    assert block, cmd
    assert why.startswith("dagland guard:")


@pytest.mark.skipif(not os.path.exists(LIVE), reason="live dagland file not on this box")
def test_real_append_passes_even_with_rm_words_in_body():
    assert G.decide(REAL_APPEND) == (False, "")


@pytest.mark.skipif(not os.path.exists(LIVE), reason="live dagland file not on this box")
@pytest.mark.parametrize("cmd", [
    f"tail -n 5 {LIVE}",
    f"ls {D}/",
    f"ls -la '{D}'",
    f"grep -c x {D}/LOOP_STATE_P6_2026-11-0*.md",
    f'Get-Content "{LIVE}" -Tail 3',
    "git status; ls F:/tanitad_scratch",
    "rm -f F:/tanitad_scratch/x.tmp",
])
def test_reads_and_unrelated_commands_pass(cmd):
    assert G.decide(cmd) == (False, ""), cmd


@pytest.mark.skipif(not os.path.exists(LIVE), reason="live dagland file not on this box")
@pytest.mark.parametrize("cmd", [
    f"touch {D}/NEW_THING.md",
    f'Add-Content -Path "{D}/new.md" -Value x',
    f"Remove-Item {LIVE}",
    f"rm {LIVE}",
    f"mv {LIVE} {LIVE}.bak",
    f"cp {LIVE} {D}/copy.md",
    f"echo x | tee {D}/LOOP_STATE_$(date +%F).md",
    f"python -c \"import os; os.remove('{LIVE}')\"",
    f"cat >> /e/Projects/TanitAD-artifacts/dagland/LOOP_STATE_P6_2026-11-12.md <<'EOF'\nx\nEOF",
    f"cat >> {LIVE} <<'EOF'\nok\nEOF\nrm -f F:/tanitad_scratch/x",
])
def test_other_creates_and_deletes_refused(cmd):
    assert G.decide(cmd)[0], cmd


@pytest.mark.skipif(not os.path.exists(LIVE), reason="live dagland file not on this box")
def test_gitbash_drive_path_of_live_file_passes():
    assert G.decide("tail -n 1 /e/Projects/TanitAD-artifacts/dagland/LOOP_STATE_P6_2026-11-06.md") == (False, "")


def test_powershell_herestring_body_is_ignored(tmp_path, monkeypatch):
    d = tmp_path / "x" / "dagland"
    d.mkdir(parents=True)
    (d / "LIVE.md").write_text("", encoding="utf-8")
    monkeypatch.setattr(G, "DAGLAND", str(d))
    cmd = f"Add-Content -Path '{d.as_posix()}/LIVE.md' -Value @'\nrm dagland/NOPE.md\n'@"
    assert G.decide(cmd) == (False, "")
    assert G.decide(f"Add-Content -Path '{d.as_posix()}/NOPE.md' -Value x")[0]


def test_relative_path_resolves_against_guard_dir(tmp_path, monkeypatch):
    d = tmp_path / "dagland"
    d.mkdir()
    (d / "LIVE.md").write_text("", encoding="utf-8")
    monkeypatch.setattr(G, "DAGLAND", str(d))
    assert G.decide("cat >> dagland/LIVE.md <<'EOF'\nx\nEOF") == (False, "")
    assert G.decide("cat >> dagland/LIVE.md.x </dev/null")[0]


def test_no_dagland_fast_path():
    assert G.decide("rm -rf F:/tanitad_scratch/p6sel") == (False, "")
    assert G.decide("") == (False, "")


def _run(payload: dict) -> int:
    p = subprocess.run([sys.executable, os.path.join(HERE, "dagland_guard.py")],
                       input=json.dumps(payload).encode("utf-8"), capture_output=True)
    return p.returncode


@pytest.mark.skipif(not os.path.exists(LIVE), reason="live dagland file not on this box")
def test_end_to_end_exit_codes():
    assert _run({"tool_name": "Bash", "tool_input": {"command": RECORDED[5]}}) == 2
    assert _run({"tool_name": "Bash", "tool_input": {"command": REAL_APPEND}}) == 0
    assert _run({"tool_name": "PowerShell", "tool_input": {"command": f"Remove-Item {LIVE}"}}) == 2
    assert _run({"tool_name": "Bash", "tool_input": {"command": "ls"}}) == 0


def test_bad_input_allows():
    p = subprocess.run([sys.executable, os.path.join(HERE, "dagland_guard.py")], input=b"not json",
                       capture_output=True)
    assert p.returncode == 0 and b"internal error" in p.stderr
