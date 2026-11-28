"""sitecustomize_bash.py : copy to <dir>/sitecustomize.py and put <dir> on PYTHONPATH.  EVERY python process started under it (the test AND the nested runs of launch_gate's G-SUITE, which a
bashshim.py wrapper around the outermost process cannot reach) maps a BARE `bash` in a Popen argument list to Git's POSIX bash.  Why: on this box CreateProcess resolves a bare `bash` to
C:\Windows\System32\bash.exe (the WSL launcher) before PATH, even when the test is started from Git bash; see bashshim.py.  Nothing else changes."""
import subprocess

_GIT_BASH = r"C:\Program Files\Git\usr\bin\bash.exe"
_orig = subprocess.Popen.__init__


def _init(self, args, *a, **kw):
    if isinstance(args, (list, tuple)) and args and args[0] == "bash":
        args = [_GIT_BASH, *args[1:]]
    _orig(self, args, *a, **kw)


subprocess.Popen.__init__ = _init
