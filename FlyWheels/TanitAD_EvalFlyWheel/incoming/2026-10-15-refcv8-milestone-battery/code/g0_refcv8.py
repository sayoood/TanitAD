"""g0_refcv8.py -- GATE G0 (battery SPEC section 2, amendments A1-A7) on a refcv8 checkpoint.

    python g0_refcv8.py --ckpt E:/refcv8_eval_kit/ckpt/ckpt_5000.pt --config E:/refcv8_eval_kit/ckpt/config.json \
        --metrics E:/refcv8_eval_kit/thor_reads/metrics_<ts>.jsonl --seeds 0,1,...,23 --out <g0.json>

A THIN wrapper: it re-points the environment (`r8_repoint.configure_env`), wraps `L.build_eval_dataset` so the eval
dataset carries the run's pose-sync sidecar (`r8_repoint.patch_loader`, the one behavioural patch), then runs the
refcv7 battery's `g0_refcv7.main()` UNCHANGED -- same judge, same A7 rules, same M1/M2/M4/M5 mutations, same
24-seed class. Every flag of `g0_refcv7.py` is accepted as is.

Besides `--out` it writes `<out stem>.r8_repoint.json` (the re-pointing record: env, remap overrides, pose-sync stats),
also on a failed / interrupted run, so a missing G0 artifact still leaves the evidence of what was attempted.
Assert on the JSON this writes, never on the exit code of this process.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r8_repoint  # noqa: E402

L = r8_repoint.apply()                      # env -> loader import -> pose-sync wrapper (BEFORE g0_refcv7 imports L)
import g0_refcv7 as G  # noqa: E402  (the refcv7 battery's judge, from the launch tree's battery/code)


def _out_path(argv: list[str]) -> Path | None:
    for i, a in enumerate(argv):
        if a == "--out" and i + 1 < len(argv):
            return Path(argv[i + 1])
        if a.startswith("--out="):
            return Path(a.split("=", 1)[1])
    return None


def main() -> None:
    out = _out_path(sys.argv[1:])
    t0 = time.strftime("%FT%T")
    status = "started"
    try:
        G.main()
        status = "finished"
    except BaseException as exc:            # noqa: BLE001 -- recorded, then re-raised
        status = f"raised {type(exc).__name__}: {str(exc)[:300]}"
        raise
    finally:
        if out is not None:
            rec = r8_repoint.describe(L)
            rec.update(g0_started=t0, g0_wrapper_status=status, g0_wrapper_finished=time.strftime("%FT%T"),
                       g0_module=str(Path(G.__file__)), argv=sys.argv[1:])
            try:
                out.parent.mkdir(parents=True, exist_ok=True)
                json.dump(rec, open(out.with_suffix(".r8_repoint.json"), "w", encoding="utf-8"), indent=1,
                          default=str)
            except OSError as exc:           # never mask the real outcome
                print(f"[g0_refcv8] could not write the re-pointing record: {exc}", flush=True)


if __name__ == "__main__":
    main()
