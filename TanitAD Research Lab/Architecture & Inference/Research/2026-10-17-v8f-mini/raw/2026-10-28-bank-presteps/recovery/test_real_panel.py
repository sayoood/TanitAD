"""pytest on REAL bank data (the self-contained panel written by recovery_run.py s0 -> ../raw/real_panel.npz): 1,000 identity windows from the real
logged tracks and 400 real off_lat / off_head records of bank_s0 (V1 labeller, FROZEN_AMENDMENT_1 construction).
Run: C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q test_real_panel.py   (cwd = this directory)
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import labeller as L  # noqa: E402

PANEL = HERE.parent / "raw" / "real_panel.npz"
_spec = importlib.util.spec_from_file_location("floors_kin_tip", HERE.parent / "src_tip" / "floors_kin.py")
FK = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FK)

pytestmark = pytest.mark.skipif(not PANEL.exists(), reason="run recovery_run.py s0 first (writes raw/real_panel.npz)")


@pytest.fixture(scope="module")
def panel():
    z = np.load(PANEL)
    ents = []
    for i, (kind, c, T, r) in enumerate(z["meta"]):
        s = int(z["starts"][i])
        n = int(T - r)
        g, v = z["G"][s:s + n], z["V"][s:s + n]
        ox, oy, oyaw, vown = z["owns"][i]
        ents.append((int(kind), g, v, (ox, oy, oyaw), float(vown)))
    return ents


def test_identity_bit_for_bit_on_1000_real_windows(panel):
    ident = [e for e in panel if e[0] == 0]
    assert len(ident) == 1000
    for _, g, v, own, vown in ident:
        res = L.recovery_label(g, v, 0, own, vown)
        assert res["status"] == L.STATUS_IDENTITY
        assert np.array_equal(res["label_own"], L.human_label(g, 0))
        stack = FK.gt_waypoints(np.concatenate([g, v[:, None]], 1), [0], np.arange(1, 31))[0]
        assert np.array_equal(res["label_own"], stack)                   # equals the stack's own gt_waypoints, zero tolerance


def test_real_off_route_panel_gate_G_REC(panel):
    off = [e for e in panel if e[0] == 1]
    assert len(off) >= 300
    n_emit = viol = 0
    dr_viol = nc_viol = nc_n = 0
    for _, g, v, own, vown in off:
        res = L.recovery_label(g, v, 0, own, vown)
        if res["label_own"] is not None:
            n_emit += 1
            viol += L.gate_violation_count(res["label_own"], vown, res["k_rej"] + 1)
        dr = L.label_ignore_offset(g, v, 0, own, vown)
        if dr["label_own"] is not None:
            dr_viol += L.gate_violation_count(dr["label_own"], vown, 6) > 0
        nc = L.recovery_label(g, v, 0, own, vown, clip=False, check=False)
        if nc["label_own"] is not None:
            nc_n += 1
            nc_viol += L.gate_violation_count(nc["label_own"], vown, nc["k_rej"] + 1) > 0
    assert n_emit >= 20, n_emit                                          # the gate is exercised on real emitted labels
    assert viol == 0                                                     # G-REC: exactly zero violations among emitted labels
    n_dr = sum(1 for e in off if L.label_ignore_offset(e[1], e[2], 0, e[3], e[4])["label_own"] is not None)
    assert dr_viol >= 0.9 * n_dr                                         # the offset-ignoring arm FAILS the gate on nearly every real off-route record
    assert nc_n > 0 and nc_viol >= 0.3 * nc_n                            # the no-clip / unchecked arm is caught


def test_real_labeller_is_deterministic(panel):
    off = [e for e in panel if e[0] == 1][:50]
    for _, g, v, own, vown in off:
        a, b = L.recovery_label(g, v, 0, own, vown), L.recovery_label(g, v, 0, own, vown)
        assert a["status"] == b["status"]
        if a["label_own"] is not None:
            assert np.array_equal(a["label_own"], b["label_own"])
