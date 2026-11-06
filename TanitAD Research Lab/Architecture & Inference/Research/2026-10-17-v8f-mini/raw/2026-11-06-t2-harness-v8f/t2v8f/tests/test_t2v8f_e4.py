"""E4 gates (Agent A): nav equals the T1 tool's join column, the scene construction equals the v9 one, BLIND mean, DAG-OACC init, starts.
Mutations are tests that must go RED."""
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
import numpy as np  # noqa: E402
import t2v8f_kit as K  # noqa: E402

JOIN = "E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/join_eval141.npz"
EGO = "C:/Users/Admin/tanitad-data/physicalai/labels/egomotion_alpamayo/{}.parquet"
NCLIP = 6


def _clips():
    B = K.load_v9()
    J = np.load(JOIN, allow_pickle=True)
    for ci in range(NCLIP):
        cid, o, n = str(J["clip"][ci]), int(J["off"][ci]), int(J["n"][ci])
        yield B, B.load_log(EGO.format(cid)), J["has"][o:o + n], J["now_s"][o:o + n], J["nav"][o:o + n]


def _native_equal():
    """Bit-for-bit equality of `nav_at` on the native log with the join column, on the rows where the release has a row. -> (n_equal, n_rows, n_nonfollow)"""
    eq = tot = nf = 0
    for B, log, has, now, nav in _clips():
        tr = B.track10(log)
        geo = (log, tr, B.turn_table(log, tr, None), float(tr.s[-1]))
        for j in np.nonzero(has)[0]:
            tot += 1
            eq += bool(np.array_equal(K.nav_at(geo, float(now[j])), nav[j]))
            nf += int(nav[j][:3].argmax() > 0)
    return eq, tot, nf


def test_nav_equals_join_on_banked_windows():
    eq, tot, nf = _native_equal()
    print("native-log nav: equal", eq, "of", tot, "rows; non-follow rows", nf)
    assert tot == 600 and eq == tot and nf > 0          # the control must contain turns, else equality could hold trivially


def test_nav_mutations_red():
    B = K.load_v9()
    old = B.LITERALS["nav_h_s"]
    try:
        B.LITERALS["nav_h_s"] = 1.0                       # M1: announce horizon 6 s -> 1 s
        e1 = _native_equal()
    finally:
        B.LITERALS["nav_h_s"] = old
    L, R = B.NAV_TURN_L, B.NAV_TURN_R
    try:
        B.NAV_TURN_L, B.NAV_TURN_R = R, L                 # M2: left and right swapped
        e2 = _native_equal()
    finally:
        B.NAV_TURN_L, B.NAV_TURN_R = L, R
    print("M1 nav_h_s=1:", e1, " M2 L/R swap:", e2)
    assert e1[0] < e1[1] and e2[0] < e2[1]
    assert _native_equal()[0] == 600                      # restored


def test_scene_construction_equals_v9_on_banked_clip():
    """A scene has only a 10 Hz rig log. Build one from a banked clip (the builder's own 10 Hz track, speed by central difference), run `route_geometry` +
    `nav_at` at the release rows' physical times and compare with the join: d_next and valid bit-identical, token equal except where the speed channel
    moves the 6 s * v threshold."""
    tok_eq = tot = v_eq = 0
    d_diff = 0.0
    for B, log, has, now, nav in _clips():
        g = np.arange(0.0, log.t_end + 1e-9, 0.1)
        x, y, yaw, v = B.sample(log, g)
        sp = np.gradient(np.hypot(np.diff(x, prepend=x[0]), np.diff(y, prepend=y[0])).cumsum(), g)
        geo = K.route_geometry(g, x, y, yaw, sp)
        rows = np.stack([K.nav_at(geo, float(now[j])) for j in np.nonzero(has)[0]])
        ref = nav[has]
        tok_eq += int((rows[:, :3].argmax(1) == ref[:, :3].argmax(1)).sum())
        v_eq += int((rows[:, 4] == ref[:, 4]).sum())
        d_diff = max(d_diff, float(np.abs(rows[:, 3] - ref[:, 3]).max()))
        tot += len(ref)
    print("scene construction: token equal", tok_eq, "of", tot, "valid equal", v_eq, "max |d_next/100| diff", d_diff)
    assert d_diff == 0.0 and v_eq == tot and tok_eq >= tot - 2


def test_blind_mean_banked():
    m, n, nc = K.blind_mean("E:/Projects/TanitAD-artifacts/sfresh_inputs/cache/eval/clips")
    assert tuple(m.shape) == (40, 1024) and str(m.dtype) == "torch.float16" and nc == 141 and n > 10000
    assert bool(np.isfinite(m.float().numpy()).all())
    import hashlib
    print("BLIND mean (eval-141 cache, NOT train): frames", n, "clips", nc, "sha256", hashlib.sha256(m.numpy().tobytes()).hexdigest())


def test_dagoacc_init_and_starts():
    c = K.arm_ckpts()
    assert all(c[a] for a in ("C0_s0", "C0_s1", "DAGOACC_s0", "DAGOACC_s1")), c
    assert c["DAGOACC_s0"].endswith("v8f-dagoacc/M2b_s0_dagoacc/ckpt_last.pt")
    assert K.init_ckpt_from_argv(f"{K.ART}/v8f-recover/M2b_s0_recft_C0/argv.json", root="C:/nonexistent") is None     # absent -> 'beside, not built'
    tab = json.load(open(K.SCREEN_TABLE))["scenes"]
    sel = [e for e in tab if e.get("selected")]
    st = K.all_starts(tab)
    ln = [st[e["scene"]] for e in sel]
    assert len(sel) == 62 and st == K.all_starts(tab)
    # card 6b (2026-11-06) clamps a start to f0 >= 18: the three scenes that had no start (launch_17/_25/_38) keep one; the full count/clamp gate is tests/test_t2v8f_run.py
    assert all(s for e, s in zip(sel, ln) if e["slot"].startswith("launch"))
    assert all(18 <= f <= e["n_ticks"] - 61 for e, s in zip(sel, ln) for f in s)
    print("starts: launch", sum(len(s) for e, s in zip(sel, ln) if e["slot"].startswith("launch")),
          "guard", sum(len(s) for e, s in zip(sel, ln) if e["slot"].startswith("guard")))
