"""Patch env-gated thread caps into the scale-up copy of yt_pilot_common.py.
8 parallel workers each spawned ~228 threads (PyAV thread_type=AUTO + OpenCV grabbing
all 96 cores) -> 200+ runnable threads, 84k ctx-switch/s, CPU 81% IDLE (thrash).
Caps: PyAV stream.thread_count + cv2.setNumThreads via YT_DECODE_THREADS / YT_CV_THREADS.
Env-gated: unset -> original behavior (pilot copy stays byte-identical when env unset)."""
import sys
p = sys.argv[1] if len(sys.argv) > 1 else "/workspace/tmp/yt_scaleup/scripts/yt_pilot_common.py"
s = open(p).read()

old1 = '        stream.thread_type = "AUTO"'
new1 = ('        stream.thread_type = "AUTO"\n'
        '        _tc = int(os.environ.get("YT_DECODE_THREADS", "0") or 0)\n'
        '        if _tc > 0:\n'
        '            stream.thread_type = "FRAME"\n'
        '            stream.thread_count = _tc')

old2 = '        import cv2\n        self.detect_every'
new2 = ('        import cv2\n'
        '        _ct = int(os.environ.get("YT_CV_THREADS", "0") or 0)\n'
        '        if _ct > 0:\n'
        '            cv2.setNumThreads(_ct)\n'
        '        self.detect_every')

assert old1 in s, "decode thread_type anchor not found"
assert old2 in s, "cv2 import anchor not found"
assert "YT_DECODE_THREADS" not in s, "already patched"
s = s.replace(old1, new1, 1).replace(old2, new2, 1)
open(p, "w").write(s)
print("PATCHED", p)
