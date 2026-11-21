"""RED/GREEN: remove the eval-side wiring, one guard at a time, and show it FAIL.

⛔ A GUARD THAT CANNOT FAIL IS WORSE THAN NONE — it buys the confidence without
the coverage. So this script MUTATES THE SHIPPED SOURCE in place (backing it up
and verifying the restore by sha256), re-runs the suite, and records exactly
which tests died. A test that has only ever been seen passing proves nothing.

FIVE mutations, each a realistic way the fix is lost. The first two are the
EXACT pre-2026-07-28 state of the world on the eval side; the last three are the
ways it decays after landing.

  1. drop_frame_arg     — `build_v2_val_episodes` stops passing `frame=` to the
                          loader. The evaluator reads the un-sliced parent while
                          claiming the sub-frame. THE PRE-FIX STATE.
  2. drop_parent_arg    — the geometry binding is no longer told the parent, so
                          "declared but never applied" goes silent.
  3. drop_frame_check   — `assert_eval_frame_matches_run` is never called: the
                          `ego=` failure (trained sliced, scored unsliced)
                          becomes reachable by omission again.
  4. drop_frame_to_cfg  — `_eval_cfg(frame)` stops applying the frame, so the
                          encoder is not sized for what it is fed.
  5. drop_cache_binding — the run's CACHE frame is no longer compared, so a
                          same-size slice of a DIFFERENT field passes.

Usage:
  python redgreen_eval.py --stack <path to stack/> --out <json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

EVALER = "scripts/eval_flagship_v4.py"
PARITY = "tanitad/data/parity.py"
TESTS = "tests/test_eval_v2_frame.py"

MUTATIONS = {
    "drop_frame_arg": {
        "file": EVALER,
        "old": """    eps = build_v2_providers(dirs, lru_size=int(getattr(a, "v2_lru", 64)),
                             frame=slice_frame, verbose=verbose)""",
        "new": """    eps = build_v2_providers(dirs, lru_size=int(getattr(a, "v2_lru", 64)),
                             verbose=verbose)""",
        "why": "the evaluator scores the un-sliced parent — THE PRE-FIX STATE",
    },
    "drop_parent_arg": {
        "file": EVALER,
        "old": """    binding = parity.assert_v2_geometry_matches(
        rec, train_frame, label="--v2-val-cache", providers=eps,
        parent=cache_frame)""",
        "new": """    binding = parity.assert_v2_geometry_matches(
        rec, train_frame, label="--v2-val-cache", providers=eps)""",
        "why": "the binding can no longer tell a slice from a mismatch",
    },
    "drop_frame_check": {
        "file": EVALER,
        "old": """    frame_check = parity.assert_eval_frame_matches_run(
        run_cfg, model_frame, label="--ckpt vs eval frame",
        cache_frame=(cache_frame if use_v2 else None))""",
        "new": """    frame_check = {"checked": False, "note": "DISABLED"}""",
        "why": "the `ego=` failure becomes reachable by omission again",
    },
    "drop_frame_to_cfg": {
        "file": EVALER,
        "old": """    if frame is not None:
        from tanitad.geometry import apply_frame
        apply_frame(cfg, frame)
    return cfg""",
        "new": """    return cfg""",
        "why": "the encoder is not sized for the frame the eval reads",
    },
    "drop_cache_binding": {
        "file": PARITY,
        "old": """    if cache_frame is not None and isinstance(tcache, dict):
        tc = CanonicalFrame.from_dict(tcache)
        if tc != cache_frame:""",
        "new": """    if False and cache_frame is not None and isinstance(tcache, dict):
        tc = CanonicalFrame.from_dict(tcache)
        if tc != cache_frame:""",
        "why": "a same-size slice of a DIFFERENT field passes silently",
    },
}


def _run(stack: Path, tests: str = TESTS) -> dict:
    p = subprocess.run([sys.executable, "-m", "pytest", tests, "-q",
                        "--no-header", "-p", "no:cacheprovider"],
                       cwd=str(stack), capture_output=True, text=True)
    tail = p.stdout.strip().splitlines()
    failed = sorted({ln.split("::")[1].split(" ")[0]
                     for ln in tail if ln.startswith("FAILED ")})
    return {"exit": p.returncode, "summary": tail[-1] if tail else "",
            "n_failed": len(failed), "failed_tests": failed,
            "first_error_lines": [ln.strip() for ln in tail
                                  if ln.lstrip().startswith("E ")][:6]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    stack = Path(a.stack).resolve()
    out: dict = {"stack": str(stack), "tests": TESTS,
                 "evidence_class": "MEASURED (ours; artifact = this JSON)"}

    out["green_before"] = _run(stack)
    print("GREEN before:", out["green_before"]["summary"], flush=True)
    assert out["green_before"]["exit"] == 0, "suite is not green to begin with"

    out["mutations"] = {}
    for name, m in MUTATIONS.items():
        f = stack / m["file"]
        original = f.read_text(encoding="utf-8")
        digest = hashlib.sha256(original.encode("utf-8")).hexdigest()
        assert m["old"] in original, f"{name}: anchor not found (source moved)"
        try:
            f.write_text(original.replace(m["old"], m["new"], 1),
                         encoding="utf-8")
            res = _run(stack)
        finally:
            f.write_text(original, encoding="utf-8")
            assert hashlib.sha256(
                f.read_text(encoding="utf-8").encode("utf-8")).hexdigest() \
                == digest, f"{name}: RESTORE FAILED"
        res.update({"why": m["why"], "file": m["file"],
                    "restored_sha256_matches": True,
                    "sha256_of_shipped_source": digest})
        out["mutations"][name] = res
        print(f"RED [{name}]: {res['summary']}  killed={res['n_failed']}: "
              f"{res['failed_tests']}", flush=True)
        assert res["exit"] != 0, f"{name} did NOT fail — the guard is INERT"

    out["green_after"] = _run(stack)
    print("GREEN after :", out["green_after"]["summary"], flush=True)
    assert out["green_after"]["exit"] == 0, "restore did not re-green the suite"
    out["ALL_GUARDS_CAN_FAIL"] = True
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
