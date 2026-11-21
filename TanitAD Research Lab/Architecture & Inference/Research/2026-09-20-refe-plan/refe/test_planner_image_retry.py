"""REFePlanner._image_for: one retry on a transient frame-load failure, and a NAMED miss reason.

MEASURED 2026-11-19 (P5-C read, attempt 1): a transient load failure under low dev-box commit memory lost
1 of 12,146 navtest tokens after 5 h, and the old bare `except Exception: return None` left the seam's miss
reason EMPTY ("no frames: "). This test executes the method's REAL source text (ast-extracted from
planner.py, so no nuPlan / model import is needed) against a 4-camera zip container, with:
  T1 a failure on the first read only           -> frames returned, miss_reason untouched
  T2 a failure on every read                    -> None, miss_reason names the exception and "2 attempts"
  T3 cv2.imdecode returning None on every read  -> None, miss_reason names imdecode
  T4 a healthy container                        -> frames returned, shape [1, 4, 3, H, W], RGB order kept
and MUTATION arms that must turn a check RED: no retry (range(2) -> range(1)); the miss-reason line deleted.
Prints ZZTEST_PASS <n> or raises.  Run: <python with numpy + cv2> test_planner_image_retry.py
"""
from __future__ import annotations

import ast
import os
import sys
import tempfile
import types
import zipfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PLANNER = os.path.join(HERE, "planner.py")
N = 0


def ok(cond, msg):
    global N
    if not cond:
        raise AssertionError(msg)
    N += 1


def method_source(text: str) -> str:
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "REFePlanner":
            for f in node.body:
                if isinstance(f, ast.FunctionDef) and f.name == "_image_for":
                    lines = text.splitlines()[f.lineno - 1:f.end_lineno]
                    ind = len(lines[0]) - len(lines[0].lstrip())
                    return "\n".join(l[ind:] for l in lines)
    raise AssertionError("REFePlanner._image_for not found in planner.py")


def build(src: str):
    fake_torch = types.SimpleNamespace(from_numpy=lambda a: a, Tensor=object)
    ns = {"np": np, "torch": fake_torch}
    exec(compile("from __future__ import annotations\n" + src, PLANNER + "::_image_for", "exec"), ns)
    return ns["_image_for"]


class Frames:
    def __init__(self, paths):
        self.paths, self.miss_reason = paths, ""

    def resolve(self, log_name, t_us):
        return list(self.paths)


def make_self(paths):
    cfg = types.SimpleNamespace(img_w=32, img_h=16, n_cameras=4, cameras=["F0", "L0", "R0", "B0"])
    return types.SimpleNamespace(_log_hint="log_x", frames=Frames(paths), cfg=cfg)


def ego():
    return types.SimpleNamespace(time_point=types.SimpleNamespace(time_us=123))


def container(tmp):
    paths = []
    for k, cam in enumerate(["F0", "L0", "R0", "B0"]):
        img = np.zeros((20, 40, 3), np.uint8)
        img[:, :, 0] = 10 * (k + 1)                    # BGR blue channel carries the camera index
        img[:, :, 2] = 200
        okk, enc = cv2.imencode(".png", img)
        assert okk
        zp = os.path.join(tmp, f"log_x_CAM_{cam}.zip")
        with zipfile.ZipFile(zp, "w") as z:
            z.writestr("frame_000123.png", enc.tobytes())
        paths.append(f"{zp}::frame_000123.png")
    return paths


class FlakyZip:
    """zipfile.ZipFile stand-in that raises MemoryError on the first `fail` reads, then delegates."""
    def __init__(self, fail):
        self.left, self.real = fail, zipfile.ZipFile

    def __call__(self, *a, **k):
        if self.left > 0:
            self.left -= 1
            raise MemoryError("Unable to allocate 6.0 MiB (simulated)")
        return self.real(*a, **k)


def run(fn, paths, flaky=None, decode_none=False):
    s = make_self(paths)
    real_zip, real_dec = zipfile.ZipFile, cv2.imdecode
    try:
        if flaky is not None:
            zipfile.ZipFile = flaky
        if decode_none:
            cv2.imdecode = lambda *a, **k: None
        out = fn(s, ego())
    finally:
        zipfile.ZipFile, cv2.imdecode = real_zip, real_dec
    return out, s.frames.miss_reason


def checks(fn):
    with tempfile.TemporaryDirectory() as tmp:
        paths = container(tmp)
        out, why = run(fn, paths)                                         # T4
        ok(out is not None and tuple(out.shape) == (1, 4, 3, 16, 32), f"T4 shape {None if out is None else out.shape}")
        ok(abs(float(out[0, 2, 2].mean()) - 30 / 255) < 1e-6, "T4 RGB order: camera index must sit in channel 2 (BGR->RGB)")
        ok(why == "", f"T4 miss_reason must stay empty, got {why!r}")
        out, why = run(fn, paths, flaky=FlakyZip(1))                      # T1
        ok(out is not None, f"T1 one transient failure must be retried, got None ({why!r})")
        ok(why == "", f"T1 miss_reason must stay empty after a successful retry, got {why!r}")
        out, why = run(fn, paths, flaky=FlakyZip(10 ** 6))                # T2
        ok(out is None, "T2 a persistent failure must return None")
        ok("MemoryError" in why and "2 attempts" in why, f"T2 miss_reason must name the cause, got {why!r}")
        out, why = run(fn, paths, decode_none=True)                       # T3
        ok(out is None and "imdecode" in why, f"T3 got {why!r}")


def main():
    text = open(PLANNER, encoding="utf-8").read()
    src = method_source(text)
    checks(build(src))
    mutations = {
        "no retry": ("for _attempt in range(2):", "for _attempt in range(1):"),
        "unnamed miss": ('self.frames.miss_reason = f"frame load failed after 2 attempts ({member}): {err}"[:300]', "pass"),
    }
    for name, (a, b) in mutations.items():
        ok(src.count(a) == 1, f"mutation target for {name!r} must occur exactly once")
        try:
            checks(build(src.replace(a, b)))
        except AssertionError:
            ok(True, name)
            print(f"  mutation {name!r}: RED (caught)")
            continue
        raise AssertionError(f"mutation {name!r} SURVIVED")
    print(f"ZZTEST_PASS {N}")


if __name__ == "__main__":
    sys.exit(main())
