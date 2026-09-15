#!/usr/bin/env python3
"""Mutation runs: re-introduce each defect the new tests exist to catch, run the two test
files, and record which tests go RED. A guard that stays green under its own defect is
not a guard. Writes ``../raw/mutation_runs.json``.

Each mutation is a tiny pytest plugin that monkeypatches the module under test at
configure time (the source files are never edited). Usage:
    PYTHONPATH=<worktree>/stack python mutation_runs.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
STACK = Path(os.environ.get("PYTHONPATH", "").split(os.pathsep)[0] or ".").resolve()
TESTS = ["tests/test_semantic_map_gt.py", "tests/test_bev_lift.py"]

MUTATIONS = {
    "M0_control_no_mutation": "",
    "M1_identity_check_removed": """
        import tanitad.data.semantic_map_gt as G
        _orig = G._check_meta
        def _no_identity(meta, s12):
            meta = dict(meta); meta["source"] = dict(meta.get("source") or {}, clip_sha12=s12)
            return _orig(meta, s12)
        G._check_meta = _no_identity
    """,
    "M2_time_check_disabled": """
        import tanitad.data.semantic_map_gt as G
        G.ClipMapGT.check_times = lambda self, idx, t, tol_us=1000.0: 0.0
    """,
    "M3_negative_index_wraps": """
        import numpy as np
        import tanitad.data.semantic_map_gt as G
        _orig = G.ClipMapGT._index
        def _wrap(self, frame_idx):
            a = np.asarray(frame_idx)
            if np.issubdtype(a.dtype, np.integer) and a.dtype != np.bool_:
                a = np.where(a < 0, a + self.n_frames, a)
            return _orig(self, a)
        G.ClipMapGT._index = _wrap
    """,
    "M4_seen_threshold_off_by_one": """
        import numpy as np
        import tanitad.data.semantic_map_gt as G
        _orig = G.ClipMapGT.read
        def _read(self, frame_idx, frame_t_us=None, *, tol_us=1000.0):
            m = _orig(self, frame_idx, frame_t_us, tol_us=tol_us)
            u8 = self.cart_u8()[m.frame_idx][:, 8]
            return G.MapFrames(cart=m.cart, seen=u8 <= 128, t_img_us=m.t_img_us, frame_idx=m.frame_idx)
        G.ClipMapGT.read = _read
    """,
    "M5_pose_check_disabled": """
        import tanitad.data.semantic_map_gt as G
        G.ClipMapGT.check_pose_alignment = lambda self, poses, **kw: {"verdict": "ALIGNED"}
    """,
    "M6_lift_mirrored_y": """
        import torch
        import tanitad.models.bev_lift as L
        _orig = L.project_rig_points
        def _mirror(p, cam):
            return _orig(p * torch.tensor([1.0, -1.0, 1.0], dtype=p.dtype), cam)
        L.project_rig_points = _mirror
    """,
    "M7_symmetric_tiling_feature_offset": """
        import tanitad.models.bev_lift as L
        L.REFC_FEATURE_CENTRE_OFFSET_PX = 7.5
        _orig = L.pixel_to_feature_grid
        def _sym(col, row, frame, stride, centre_offset_px=None):
            return _orig(col, row, frame, stride, (stride - 1) / 2.0)
        L.pixel_to_feature_grid = _sym
    """,
    "M8_validity_rows_only": """
        import tanitad.models.bev_lift as L
        _orig = L.project_rig_points
        def _rows_only(p, cam):
            r = _orig(p, cam)
            r["valid"] = r["in_rows"]
            return r
        L.project_rig_points = _rows_only
    """,
}


def run(name: str, body: str) -> dict:
    with tempfile.TemporaryDirectory() as td:
        mod = f"mut_{name.lower()}"
        (Path(td) / f"{mod}.py").write_text(
            "def pytest_configure(config):\n" + textwrap.indent(textwrap.dedent(body) or "pass", "    ") + "\n",
            encoding="utf-8")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([td, str(STACK)]), PYTHONUTF8="1")
        cmd = [sys.executable, "-m", "pytest", *TESTS, "-q", "-p", "no:cacheprovider", "-p", mod, "-rfE"]
        p = subprocess.run(cmd, cwd=STACK, env=env, capture_output=True, text=True, encoding="utf-8")
    txt = p.stdout + p.stderr
    failed = sorted(set(re.findall(r"^FAILED (\S+)", txt, flags=re.M)))
    summ = re.findall(r"^=*\s*(\d+ (?:failed|passed)[^\n]*?) in [\d.]+s", txt, flags=re.M)
    return {"exit": p.returncode, "summary": summ[-1] if summ else txt[-300:],
            "failed": [f.split("::", 1)[1].split(" ")[0] for f in failed]}


def main() -> int:
    out = {}
    for name, body in MUTATIONS.items():
        out[name] = run(name, body)
        print(f"{name}: {out[name]['summary']} | RED: {out[name]['failed']}", flush=True)
    (HERE.parent / "raw" / "mutation_runs.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
