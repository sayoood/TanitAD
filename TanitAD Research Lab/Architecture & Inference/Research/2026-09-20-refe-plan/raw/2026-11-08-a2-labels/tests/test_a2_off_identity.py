"""OFF-identity on REAL nuPlan states (heavy; skipped unless A2_SCENES points at a build_scenes.py output).  The proof the brief asks for:
with the extension OFF, labels are bit-identical to the pinned tip (d5ec25ec), through the real worker path online_labeller._job, in separate processes, seeded fans.
  A2_SCENES=F:/a2_scratch/scenes A2_IDENT_WORK=F:/a2_scratch/idw python tests/run_tests.py test_a2_off_identity.py      (needs make_merged.sh output at F:/a2_scratch/merged_refe)
Deliberate-regression arm: the same comparison on a PERTURBED copy of the tip labels must report NOT equal."""
import json
import os
import subprocess

import numpy as np
import pytest

SCENES = os.environ.get("A2_SCENES")
WORK = os.environ.get("A2_IDENT_WORK", "F:/a2_scratch/idw")
A2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_off_is_bit_identical_to_tip_through_job():
    if not SCENES:
        pytest.importorskip("a2_scenes_not_configured")           # skipped
    subprocess.run(["bash", os.path.join(A2, "measure", "run_identity.sh"), SCENES, WORK], check=True)
    res = json.load(open(os.path.join(WORK, "job_identity_result.json")))
    assert res["n_scenes"] >= 8
    assert res["tip_vs_overlay_off_bytes_equal"] and res["overlay_off_vs_on_labels_bytes_equal"] and res["on_vs_on_two_processes_bytes_equal"]
    assert res["off_has_no_behaviour_arrays"] and res["on_has_behaviour_arrays"] == res["n_scenes"]
    a, b = np.load(os.path.join(WORK, "job_tip_off.npz")), np.load(os.path.join(WORK, "job_ov_off.npz"))
    k = sorted(a.files)[0]
    bad = b[k].copy()
    bad[0, 0] = np.nextafter(bad[0, 0], np.float32(2.0))               # regression arm: one ulp of one label must break the byte comparison
    assert a[k].tobytes() != bad.tobytes()
