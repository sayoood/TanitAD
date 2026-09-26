# `cv_fill_poly` is pixel-exact with real OpenCV 4.5.4 — a guard that had never run, now lit

**Package** `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-26-fillpoly-opencv-reference/`
**Owner** EvalFlyWheel orchestrator · **Date** 2026-09-26 · **Evidence** MEASURED unless stated.

## Why

Every nuScenes collision grid is drawn by `cv_fill_poly` (`taniteval/adapters/nuscenes_planning.py`), a
port of OpenCV 4.5.4's `drawing.cpp` — the main venv has no OpenCV. Its only parity test **skipped on this
box for its whole life** (*"NOT RUN: OpenCV not installed in this venv"*). The floors reproducing PARA-Drive
to a few percent vouched for it only indirectly.

## How the reference was made

* **Environment** (built by the Master Mind; PI approval relayed from its session: *"you can download opencv
  on D:"*): `D:/venvs/opencv-ref-454`, Python 3.9.25 from the box's existing uv CPython, **opencv-python-headless
  4.5.4.60 + numpy 1.26.4 only**, installed OFFLINE (`--no-index --no-deps`).
  **Re-verified independently here:** `cv2 == 4.5.4`; build info *"General configuration for OpenCV 4.5.4"*;
  **torch not importable**; `pip list` = exactly those two plus pip/setuptools; both wheels' sha256 **and** sizes
  recomputed and equal to PyPI's published digests (opencv `a1f9d41c6afe86fd…` 35,024,208 B; numpy
  `3373d5d70a5fe74a…` 15,814,633 B).
* **4.5.4, not the latest (5.0.0.93):** a raster is only a reference if it comes from the routine the port
  claims to reproduce.
* **Vertices generated in the MAIN venv and banked** (`code/gen_polygons.py`): numpy does not guarantee a
  `Generator`'s stream across versions (1.26 there, 2.5.1 here), so no test depends on two RNGs agreeing.
* **Drawn in the OpenCV env** (`code/render_reference.py`), which ⛔ **asserts `cv2.__version__ == "4.5.4"` before
  drawing anything** and records its own provenance (build-info head, wheel sha256s recomputed from the files).

## The reference — `taniteval/tests/fixtures/fillpoly_opencv454/`

| group | cases | grid | filled px | cv2 raised |
|---|---|---|---|---|
| `quads` — the old test's exact cases (seed 0, `integers(-20,60,(4,2))`) | 200 | 40×40 | 91,489 | 0 |
| `boxes` — rotated agent boxes through the builders' own `np.round((c−BX+DX/2)/DX)` | 300 | 200×200 | 16,976 | 0 |
| `adversarial` — degenerate, concave, self-intersecting, boundary, off-grid, huge, reversed winding | 20 | 64×64 | 15,178 | 0 |

`fillpoly_cv2_454_reference.npz` sha256 `08e0d3e0f384a5060eb1ac2e11e8428369b458a2636f6edf108a4a86262de5ca`
(recomputed independently — matches the receipt).

## ⭐ Result

**520 / 520 cases pixel-exact. 0 differing pixels.** The port that draws every collision grid is correct —
now measured directly, not inferred from downstream agreement.

## Tests

* `taniteval/tests/test_fillpoly_opencv454_reference.py` (new, 10): the npz is the banked bytes; the receipt
  proves its own version and wheel sha256s; per-group counts, grid shapes and **filled-pixel literals** (so an
  all-zero reference could not pass); exact parity per group; and two arms that **must fail** — every quad
  shifted one pixel right must be caught on >95 % of non-empty cases, and a single flipped reference pixel must
  be caught.
* `test_nuscenes_planning.py::test_fillpoly_parity_with_real_cv2` **no longer skips**: it compares its own
  200 quads to the banked rasters, and first requires its regenerated quads to EQUAL the banked vertices — so a
  future change in numpy's random stream is reported plainly instead of silently comparing different polygons.
  ⭐ Proven to touch ONLY that function versus the true tip `32fea93` (70 functions both sides, none added or
  removed, everything outside it identical).
* **82 passed, 0 skipped** across the nuScenes suites (previously 1 skip — this one).

⚠️ Only the BINARY `.npz` is byte-hashed: text files convert LF↔CRLF on checkout here, so hashing
`polygons.json` would fail elsewhere for a reason unrelated to the reference.

## Manifest

| artifact | where |
|---|---|
| `code/gen_polygons.py`, `code/render_reference.py`, this `RESULT.md`, `LANDING_READY.txt` | repo, staged |
| `taniteval/tests/fixtures/fillpoly_opencv454/{fillpoly_cv2_454_reference.npz, polygons.json, RECEIPT.json}` | repo, staged |
| `taniteval/tests/test_fillpoly_opencv454_reference.py`, `taniteval/tests/test_nuscenes_planning.py` | repo, staged |
| the throwaway env + wheels | **off-repo** `D:/venvs/opencv-ref-454/`, `D:/venvs/opencv-ref-454_wheels/` — no longer needed by any test; deletable |
