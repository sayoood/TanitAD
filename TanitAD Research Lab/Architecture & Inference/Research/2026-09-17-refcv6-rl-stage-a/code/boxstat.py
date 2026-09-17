"""Print `<gpu_used_MiB> <host_free_GB>` — one line, two integers, or exit non-zero.

⛔ Written as a file rather than inlined in the shell because generating shell from a
heredoc ate the `\\r` escapes in a `tr -d` and silently produced a broken command that
`bash -n` still accepted. A syntax check is not a semantics check.

⭐ It REFUSES rather than guessing. A probe that cannot read its quantity must not return
a number a caller will read as "clear" — the same rule as the 40-char blob assertion.
"""
from __future__ import annotations

import ctypes
import subprocess
import sys


class _MEMSTAT(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def host_free_gb() -> int:
    m = _MEMSTAT()
    m.dwLength = ctypes.sizeof(_MEMSTAT)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
        raise OSError("GlobalMemoryStatusEx failed")
    return int(m.ullAvailPhys / (1024 ** 3))


def gpu_used_mib() -> int:
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits"], capture_output=True)
    t = r.stdout.decode("utf-8", "replace").strip().splitlines()
    if r.returncode != 0 or not t:
        raise OSError("nvidia-smi gave no reading")
    return int(t[0].strip())


def main() -> int:
    try:
        print(f"{gpu_used_mib()} {host_free_gb()}")
    except Exception as e:                       # noqa: BLE001 — any failure is INCONCLUSIVE
        print(f"PROBE-FAILED {type(e).__name__}: {e}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
