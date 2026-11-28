"""Self-test of gpu_lock.py on a TEMP lock file (never touches the real devbox_gpu.lock):
    python test_gpu_lock.py            -> prints one JSON verdict, exit 0 iff all pass
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_lock as G  # noqa: E402


def main() -> int:
    d = Path(tempfile.mkdtemp())
    L = d / "devbox_gpu.lock"
    me = os.getpid()
    r = {}
    r["1_acquire_when_free"] = G.acquire(L, me, "t", "j", 3, 0.2) == 0 and json.loads(L.read_text())["pid"] == me
    r["2_live_holder_is_waited_on_then_timeout"] = G.acquire(L, me + 1, "t2", "j", 1.5, 0.3) == 2 and json.loads(L.read_text())["pid"] == me
    r["3_release_refused_for_another_pid"] = G.release(L, me + 12345) == 3 and L.exists()
    r["4_release_own"] = G.release(L, me, "t") == 0 and not L.exists()
    # LOCK-STEAL-2 (2026-10-18): only a lock carrying the acquirer's OWN owner marker may be judged stale.  Tests 5 and 6 used to pin the old
    # behaviour with a FOREIGN owner ("dead" / "preboot") being renamed aside; the fixtures now carry the acquirer's marker ("t"), and 7 / 8 pin the inverse.
    L.write_text(json.dumps({"pid": 999999, "owner": "t", "job": "x", "t": time.time(), "acquired": "now"}))
    r["5_dead_pid_lock_renamed_aside_not_deleted"] = (G.acquire(L, me, "t", "j", 3, 0.2) == 0
                                                       and len(list(d.glob("devbox_gpu.lock.stale_*"))) == 1)
    G.release(L, me, "t")
    L.write_text(json.dumps({"pid": me, "owner": "t", "job": "x", "t": G.boot_time() - 3600, "acquired": "old"}))
    r["6_pre_boot_lock_stale_even_if_pid_alive"] = (G.acquire(L, me, "t", "j", 3, 0.2) == 0
                                                     and len(list(d.glob("devbox_gpu.lock.stale_*"))) == 2)
    G.release(L, me, "t")
    body = json.dumps({"pid": 999999, "owner": "dead", "job": "x", "t": time.time(), "acquired": "now"})
    L.write_text(body)
    r["7_foreign_owner_dead_pid_lock_is_NOT_renamed"] = (G.acquire(L, me, "t", "j", 1.0, 0.2) == 2 and L.read_text() == body
                                                         and len(list(d.glob("devbox_gpu.lock.stale_*"))) == 2)
    body = json.dumps({"pid": me, "owner": "preboot", "job": "x", "t": G.boot_time() - 3600, "acquired": "old"})
    L.write_text(body)
    r["8_foreign_owner_pre_boot_lock_is_NOT_renamed"] = (G.acquire(L, me, "t", "j", 1.0, 0.2) == 2 and L.read_text() == body
                                                         and len(list(d.glob("devbox_gpu.lock.stale_*"))) == 2)
    ok = all(r.values())
    print(json.dumps({"ok": ok, "tests": r}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
