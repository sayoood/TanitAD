"""stream_archive must not deadlock on the unread tail (or the stderr) of `git archive`.

MEASURED 2026-10-22 05:13-05:18 Berlin on the dev box (Windows): the G-SUITE-PINNED archive step
extracted 100 % of the commit (43,203 files / 10,225,715,486 B) and then sat at 0 CPU for >30 min.
`git archive` pads its tar to a 10,240-B record; `tarfile` (mode "r|") stops at the end-of-archive
block, the padding stays in the pipe, git blocks writing it and the gate blocks in
`p.stderr.read()` waiting for git to exit. Draining the 10,240 B from outside (all zero) released
both. Whether it bites depends on the tail residue against the pipe buffer, so it passed on other
commits by luck, and on Linux (64 KiB pipe) it never shows. The symmetric hazard: if git writes more
than a pipe buffer to STDERR while the gate reads stdout, git blocks on stderr and the gate blocks
on stdout.

How this test reproduces it deterministically, on any platform: `subprocess.Popen` AS SEEN BY the
gate module is swapped for a shim that launches `sys.executable -c <child>` instead of git (a PATH
shim would NOT work -- on Windows CreateProcess appends only `.exe`, so a `git.bat` is ignored and
the real git answers). The child writes a small valid tar and then PAD (512 KiB) zero bytes, far
above any pipe buffer on any OS, and optionally STDERR_BYTES (256 KiB) of noise on stderr.

  * FIX arms: `launch_gate.stream_archive` must finish inside a hard timeout, extract byte-exact,
    write the completion marker and report `trailing_pad_bytes`; non-zero bytes after the
    end-of-archive block must raise GateError.
  * DELIBERATE-REGRESSION arms: `_stream_archive_prefix` is the PRE-FIX body, copied verbatim from
    commit 947b05f2 (blob 9d9d5e20). Run against the SAME child it must NOT finish -- the
    test reproduces the deadlock, so the fix arms cannot be green against a child that never blocked.
    Only the child this test spawned is ever killed (the Popen objects are kept).
Expectations are literals or built independently of the code under test.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import launch_gate as LG  # noqa: E402

# names the verbatim pre-fix body resolves in THIS module's globals (see `_prefix_fn`)
GateError = LG.GateError
utc_now = LG.utc_now
write_json = LG.write_json
_ARCHIVE_DONE = LG._ARCHIVE_DONE

COMMIT = "0123456789abcdef0123456789abcdef01234567"
PAD = 512 * 1024            # >> any OS pipe buffer (Windows 4-64 KiB, Linux 64 KiB)
STDERR_BYTES = 256 * 1024   # >> any OS pipe buffer
FIX_TIMEOUT_S = 60          # the fixed function finishes in well under a second; this is a hard cap
HANG_WAIT_S = 20            # how long the pre-fix body is given to (not) finish

# literal expectations
FILES = {
    "stack/a.py": b"A = 1\n" * 50,
    "stack/sub/b.bin": bytes(range(256)) * 40,
    "README.md": b"hello\n",
}
EXPECT_BYTES = 10546        # 300 + 10240 + 6
SYMLINKS = ["stack/link"]


def _make_tar() -> bytes:
    """A small `git archive`-shaped tar: dir entries, files, one symlink, end-of-archive + record pad."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tf:
        for d in ("stack", "stack/sub"):
            ti = tarfile.TarInfo(d)
            ti.type = tarfile.DIRTYPE
            ti.mode = 0o755
            tf.addfile(ti)
        for name, data in FILES.items():
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            ti.mode = 0o644
            tf.addfile(ti, io.BytesIO(data))
        ln = tarfile.TarInfo("stack/link")
        ln.type = tarfile.SYMTYPE
        ln.linkname = "a.py"
        tf.addfile(ln)
    return buf.getvalue()


_CHILD = r"""
import sys
tar_path, pad, nz_at, err_n, err_first = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
out, err = sys.stdout.buffer, sys.stderr.buffer
def noise():
    if err_n:
        err.write(b"E" * err_n)
        err.flush()
if err_first == "1":
    noise()
out.write(open(tar_path, "rb").read())
out.flush()
tail = bytearray(pad)
if nz_at >= 0:
    tail[nz_at] = 0x41
out.write(bytes(tail))
out.flush()
if err_first != "1":
    noise()
sys.exit(0)
"""


# --- the PRE-FIX body, verbatim from commit 947b05f2 (blob 9d9d5e20), renamed ------------------- #
def _stream_archive_prefix(git_dir: str, commit: str, dest: Path) -> dict:
    """`git -c core.autocrlf=false archive <commit>` of the WHOLE commit, STREAMED member by
    member into `dest` (never held in memory: the full tree is ~6.6 GB). A `.gate_archive_complete`
    marker is written LAST; a directory without it is a partial extraction and is refused."""
    dest.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, GIT_DIR=str(git_dir))
    p = subprocess.Popen(["git", "-c", "core.autocrlf=false", "archive", "--format=tar", commit],
                         env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    n = nbytes = 0
    skipped = []
    with tarfile.open(fileobj=p.stdout, mode="r|") as tf:
        for m in tf:
            if m.name.startswith(("/", "..")) or ".." in Path(m.name).parts:
                raise GateError(f"refusing archive member {m.name!r}")
            target = dest / m.name
            if m.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not m.isfile():
                skipped.append(m.name)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(m)
            with open(target, "wb") as fh:
                shutil.copyfileobj(src, fh, 1 << 20)
            n += 1
            nbytes += int(m.size)
    err = p.stderr.read().decode(errors="replace")
    rc = p.wait()
    if rc != 0:
        raise GateError(f"git archive {commit[:12]} failed rc={rc}: {err[-400:]}")
    rec = {"commit": commit, "files": n, "bytes": nbytes, "skipped_non_files": skipped[:20],
           "written_utc": utc_now()}
    write_json(dest / _ARCHIVE_DONE, rec)
    return rec
# ------------------------------------------------------------------------------------------------ #


@pytest.fixture
def children():
    """Every child process THIS test spawned; killed at teardown (never anyone else's)."""
    spawned: list[subprocess.Popen] = []
    yield spawned
    for p in spawned:
        if p.poll() is None:
            p.kill()
        try:
            p.wait(timeout=10)
        except Exception:  # pragma: no cover - a child that will not die
            pass
        for s in (p.stdout, p.stderr):
            try:
                if s:
                    s.close()
            except Exception:
                pass


def _shim(tmp_path: Path, children: list, *, pad: int, nz_at: int = -1, err_n: int = 0,
          err_first: bool = False) -> types.SimpleNamespace:
    """A stand-in for the `subprocess` module whose Popen launches the fake `git archive` child."""
    tar = _make_tar()
    assert len(tar) % 10240 == 0, "the child's tar must end on a record boundary"
    tar_path = tmp_path / "child.tar"
    tar_path.write_bytes(tar)
    real_popen = subprocess.Popen

    def popen(argv, **kw):
        assert argv[0] == "git" and "archive" in argv and argv[-1] == COMMIT, argv  # the call we replace
        p = real_popen([sys.executable, "-I", "-c", _CHILD, str(tar_path), str(pad), str(nz_at),
                        str(err_n), "1" if err_first else "0"], **kw)
        children.append(p)
        return p

    ns = types.SimpleNamespace(**{k: getattr(subprocess, k) for k in dir(subprocess)
                                  if not k.startswith("__")})
    ns.Popen = popen
    return ns


def _in_thread(fn, *args) -> tuple[threading.Thread, dict]:
    box: dict = {}

    def target():
        try:
            box["rec"] = fn(*args)
        except BaseException as e:  # noqa: BLE001 - the test inspects it
            box["exc"] = e
    t = threading.Thread(target=target, daemon=True)
    t.start()
    return t, box


def _prefix_fn(shim):
    """The verbatim pre-fix function, with its `subprocess` global pointed at the shim."""
    ns = dict(globals())
    ns["subprocess"] = shim
    return types.FunctionType(_stream_archive_prefix.__code__, ns, "_stream_archive_prefix")


def _assert_extracted(dest: Path) -> None:
    for name, data in FILES.items():
        got = (dest / name).read_bytes()
        assert got == data, f"{name}: extracted bytes differ"


# the scenario ids say what each one stresses
SCENARIOS = {
    "stdout-tail-only": dict(pad=PAD),
    "stdout-tail+stderr-after": dict(pad=PAD, err_n=STDERR_BYTES),
    "stderr-flood-FIRST": dict(pad=PAD, err_n=STDERR_BYTES, err_first=True),
    "one-record-tail-10240": dict(pad=10240),
}


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_fixed_stream_archive_finishes_extracts_and_marks(name, tmp_path, monkeypatch, children):
    sc = SCENARIOS[name]
    monkeypatch.setattr(LG, "subprocess", _shim(tmp_path, children, **sc))
    dest = tmp_path / "out" / "f0123456789"
    t, box = _in_thread(LG.stream_archive, "unused-git-dir", COMMIT, dest)
    t.join(timeout=FIX_TIMEOUT_S)
    assert not t.is_alive(), f"stream_archive did not finish in {FIX_TIMEOUT_S}s -- the pipe deadlock"
    assert "exc" not in box, f"raised {box.get('exc')!r}"
    rec = box["rec"]
    _assert_extracted(dest)
    assert rec["commit"] == COMMIT and rec["files"] == 3 and rec["bytes"] == EXPECT_BYTES
    assert rec["skipped_non_files"] == SYMLINKS
    assert 0 <= rec["trailing_pad_bytes"] <= sc["pad"]
    if sc["pad"] >= PAD:
        # the padding really was left in the pipe by tarfile and then drained by the fix
        assert rec["trailing_pad_bytes"] > 0
    marker = dest / LG._ARCHIVE_DONE
    assert marker.is_file()
    assert json.loads(marker.read_text(encoding="utf-8")) == json.loads(json.dumps(rec))
    assert len(children) == 1 and children[0].returncode == 0   # the child ran to a clean exit


@pytest.mark.parametrize("nz_at", [0, PAD // 2, PAD - 1], ids=["first", "middle", "last"])
def test_nonzero_trailing_bytes_raise_gateerror(nz_at, tmp_path, monkeypatch, children):
    monkeypatch.setattr(LG, "subprocess", _shim(tmp_path, children, pad=PAD, nz_at=nz_at))
    dest = tmp_path / "out" / "f0123456789"
    t, box = _in_thread(LG.stream_archive, "unused-git-dir", COMMIT, dest)
    t.join(timeout=FIX_TIMEOUT_S)
    assert not t.is_alive(), "stream_archive hung on non-zero trailing bytes"
    assert isinstance(box.get("exc"), LG.GateError), f"expected GateError, got {box!r}"
    assert "non-zero bytes after the end-of-archive" in str(box["exc"])
    assert not (dest / LG._ARCHIVE_DONE).exists(), "a stream that stopped early must NOT be marked complete"
    children[0].wait(timeout=30)               # the fix killed its own child: it must be gone
    assert children[0].poll() is not None


# (scenario, what the stuck thread dies with once ITS child is killed: where it was stuck decides it --
#  after the tar was read, rc != 0 -> GateError; before a single header byte, tarfile's own ReadError)
_PREFIX_DEATH = [("stdout-tail-only", GateError),
                 ("one-record-tail-10240", GateError),     # EXACTLY the production tail (one 10,240-B record)
                 ("stderr-flood-FIRST", tarfile.ReadError)]


# The one-record arm reproduces ONLY where the pipe buffer is smaller than one 10,240-B record (Windows' anonymous pipe).
# Linux's 64 KiB pipe absorbs it: MEASURED on Thor 2026-10-22 (suite_22f9863a8b33), the pre-fix body FINISHED in 0.0 s.
# The 512 KiB tail and the stderr flood still deadlock the pre-fix body on every platform.
_WINDOWS_PIPE_ONLY = {"one-record-tail-10240"}


@pytest.mark.parametrize("name,dies_with", [
    pytest.param(n, e, id=n, marks=pytest.mark.skipif(
        n in _WINDOWS_PIPE_ONLY and sys.platform != "win32",
        reason="a 64 KiB POSIX pipe absorbs one 10,240-B record; the production deadlock is a Windows-pipe signature"))
    for n, e in _PREFIX_DEATH])
def test_DELIBERATE_REGRESSION_prefix_body_deadlocks(name, dies_with, tmp_path, children):
    """The PRE-FIX body, same child: it must hang. If it ever finishes, the fix arms above prove nothing."""
    sc = SCENARIOS[name]
    shim = _shim(tmp_path, children, **sc)
    dest = tmp_path / "out" / "f0123456789"
    t, box = _in_thread(_prefix_fn(shim), "unused-git-dir", COMMIT, dest)
    t0 = time.time()
    t.join(timeout=HANG_WAIT_S)
    waited = time.time() - t0
    try:
        assert t.is_alive(), (f"pre-fix body FINISHED after {waited:.1f}s ({box!r}) -- this machine does "
                              f"not reproduce the deadlock for {name}; the fix arms are not discriminating")
        assert len(children) == 1
        child = children[0]
        assert child.poll() is None, "the child already exited, so the hang is not the pipe deadlock"
        assert not (dest / LG._ARCHIVE_DONE).exists()
        if name != "stderr-flood-FIRST":
            # the measured symptom: extraction 100 % complete, marker absent, both processes idle
            _assert_extracted(dest)
    finally:
        for p in children:                      # ONLY the child this test spawned
            if p.poll() is None:
                p.kill()
        t.join(timeout=30)                      # killing it must release the stuck thread
    assert not t.is_alive(), "cleanup failed: the pre-fix thread is still stuck after the child was killed"
    assert isinstance(box.get("exc"), dies_with), box   # the thread ended BECAUSE its child was killed
