"""Fail-closed network guard for the RECOVER-NEXT dev-box tests. install() makes any socket connect / name lookup, any process spawn that names a network tool (ssh, scp, sftp, rsync, curl, wget, ...) or the host
tanitad-box2, and every os.system / os.popen / os.exec* / os.spawn* raise RuntimeError('NETWORK-BLOCKED ...') BEFORE anything leaves the process. Nothing in the tests is allowed to reach a remote."""
import os, re, socket, subprocess

BLOCKED_EXE = {"ssh", "scp", "sftp", "rsync", "curl", "wget", "ftp", "telnet", "nc", "ncat", "ping", "nslookup", "tracert", "plink", "pscp"}
BLOCKED_TEXT = ("tanitad-box2", "192.168.178.171")
TRIPPED = []


def blocked_cmd(args) -> bool:
    parts = list(args) if isinstance(args, (list, tuple)) else str(args).split()
    exe = re.sub(r"\.(exe|cmd|bat)$", "", os.path.basename(str(parts[0])).lower()) if parts else ""
    flat = " ".join(map(str, parts)).lower()
    return exe in BLOCKED_EXE or any(t in flat for t in BLOCKED_TEXT)


def _deny(what):
    TRIPPED.append(what)
    raise RuntimeError(f"NETWORK-BLOCKED {what}")


def _connect(self, *a, **k): _deny("socket.connect")
def _connect_ex(self, *a, **k): _deny("socket.connect_ex")
def _create_connection(*a, **k): _deny("socket.create_connection")
def _getaddrinfo(*a, **k): _deny("socket.getaddrinfo")
def _gethostbyname(*a, **k): _deny("socket.gethostbyname")
_orig_popen_init = subprocess.Popen.__init__


def _popen_init(self, args, *a, **k):
    if blocked_cmd(args):
        _deny(f"subprocess {args!r}")
    return _orig_popen_init(self, args, *a, **k)


def _deny_os(*a, **k): _deny("os.system/popen/exec/spawn")


def install():
    socket.socket.connect, socket.socket.connect_ex = _connect, _connect_ex
    socket.create_connection, socket.getaddrinfo, socket.gethostbyname = _create_connection, _getaddrinfo, _gethostbyname
    subprocess.Popen.__init__ = _popen_init
    for n in ("system", "popen", "execv", "execve", "execvp", "execl", "execlp", "spawnv", "spawnve", "spawnl", "spawnlp", "spawnvp"):
        if hasattr(os, n):
            setattr(os, n, _deny_os)


def installed() -> bool:
    return (socket.socket.connect is _connect and socket.create_connection is _create_connection and socket.getaddrinfo is _getaddrinfo
            and subprocess.Popen.__init__ is _popen_init and os.system is _deny_os)
