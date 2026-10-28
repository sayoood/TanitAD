"""Regenerates golden/golden_speedread_modes.json from the PINNED SPEEDREAD instrument (golden/v8f_mini_eval_speedread_fd5ce0be.py, sha256 fd5ce0be...), never from the instrument under test.
The golden is what the SPEEDREAD code did on the tiny synthetic fixture of test_v8f_speedsub.py BEFORE PREREG_V8F_SPEEDSUB touched the file. Run:
    CUDA_VISIBLE_DEVICES=-1 python make_golden_speedread.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("SPEEDSUB_NO_AUTOLOAD", "1")
import test_v8f_speedsub as T  # noqa: E402

if __name__ == "__main__":
    g = T.golden_payload(T.ORIG)
    out = os.path.join(HERE, "golden", "golden_speedread_modes.json")
    json.dump(g, open(out, "w", encoding="utf-8", newline="\n"), indent=0, sort_keys=True)
    print("wrote", out, len(json.dumps(g)), "bytes")
