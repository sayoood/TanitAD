#!/usr/bin/env python3
"""One online-labeller worker process (SPEC_P6 D1), speaking length-free pickle frames on stdin/stdout. Launched by online_labeller._SubprocPool when the trainer
and the NAVSIM stack live in DIFFERENT interpreters (`--label-python`). `multiprocessing` cannot do that: its spawn hands the parent's sys.path to the child, which
then imports the parent's (wrong-ABI) numpy. Frames: (meta, props) in; the `_job` result dict (or {"error": text}) out. EOF on stdin ends the worker."""
import os
import pickle
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def main() -> int:
    out = sys.stdout.buffer
    sys.stdout = sys.stderr                        # nothing a library prints may corrupt the frame stream
    inp = sys.stdin.buffer
    import online_labeller as OL
    OL._pdm_init(sys.argv[1])
    while True:
        try:
            meta, shape, raw = pickle.load(inp)
            import numpy as np
            props = np.frombuffer(raw, dtype=np.float64).reshape(shape).copy()
        except EOFError:
            return 0
        try:
            res = OL._job("pdm", meta, props)
            if res.get("labels") is not None:                      # raw bytes back (see online_labeller._SubprocPool)
                res["labels_shape"] = tuple(res["labels"].shape)
                res["labels"] = np.ascontiguousarray(res["labels"], dtype=np.float32).tobytes()
        except Exception:                          # noqa: BLE001 -- reported to the parent, which counts it and guards the rate
            res = {"error": traceback.format_exc()[-600:]}
        pickle.dump(res, out, protocol=4)
        out.flush()


if __name__ == "__main__":
    sys.exit(main())
