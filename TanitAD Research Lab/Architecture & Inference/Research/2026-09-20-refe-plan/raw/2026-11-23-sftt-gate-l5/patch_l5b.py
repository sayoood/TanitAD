"""MM 2026-11-23: G-LIVE L5 fix (SFTT gate 2, 15:49Z), from commit 95b3d174's gate_checks.py + test_scorerft_profile.py.
1. NAVSIM's pdm_score on the pod (navsim 3e8291b) returns a PDMResults DATACLASS, not a one-row DataFrame: `.iloc[0]` raised AttributeError, so L5 never ran.
   -> `_pdm_subscores(res, names)` reads either form.
2. MEASURED on the pod with (1) applied (l5probe, 8 tokens x 6 sub-scores): exactly ONE difference, ep 0.9281340780250414 (NAVSIM, float64) vs the label
   0.9281340837478638 == float32(NAVSIM) EXACTLY. The labeller STORES float32 labels, so a float64 identity at tol 1e-9 can never hold.
   -> `identity_diff` compares NAVSIM's value cast to the LABEL ARRAY's dtype (identity at the storage precision); tol stays 1e-9 (so it is exact for float32).
   The facts record the dtype compared at and the raw float64 max diff.
usage: patch_l5b.py <gate_checks in> <gate_checks out> <test in> <test out>"""
import sys
gi, go, ti, to = sys.argv[1:5]
g = open(gi, encoding="utf-8", newline="").read()
t = open(ti, encoding="utf-8", newline="").read()


def rep(src, old, new, what):
    assert src.count(old) == 1, ("anchor not unique", what, src.count(old))
    return src.replace(old, new)


HELPERS = '''def _pdm_subscores(res, names: dict) -> dict:
    """NAVSIM's pdm_score result -> {sub-score name: value}.  navsim 3e8291b (the pod) returns a PDMResults DATACLASS; other versions a one-row DataFrame (SFTT gate 2,
    2026-11-23: `.iloc[0]` on the dataclass raised AttributeError, so L5 never compared a label)."""
    if hasattr(res, "iloc"):
        row = res.iloc[0]
        return {nm: float(row[nm]) for nm in names.values()}
    return {nm: float(getattr(res, nm)) for nm in names.values()}


def identity_diff(navsim_value, label_value, label_dtype) -> float:
    """|NAVSIM - label| at the LABEL's storage precision: NAVSIM's value (clipped to [0, 1]) is cast to the label array's dtype first.  MEASURED 2026-11-23 (l5probe, pod):
    the only difference over 8 tokens x 6 sub-scores was ep 0.9281340780250414 (float64) vs the label 0.9281340837478638 == float32(NAVSIM) EXACTLY: the labeller stores
    float32, so a float64 comparison at 1e-9 can never hold, and the identity is exact at float32."""
    return abs(float(np.asarray(np.clip(float(navsim_value), 0.0, 1.0), dtype=label_dtype)) - float(label_value))


def harness_identity_facts(cap, n_tokens: int = 8) -> dict:
'''
g = rep(g, "def harness_identity_facts(cap, n_tokens: int = 8) -> dict:\n", HELPERS, "def")
g = rep(g, "    worst, n = 0.0, 0\n", "    worst, worst64, n, diffs, dts = 0.0, 0.0, 0, [], set()\n", "init")
g = rep(g, '            row = pdm_score(mc, Trajectory(np.asarray(plan8, dtype=np.float32), S["samp8"]), S["samp"], S["sim"], S["scorer"]).iloc[0]\n',
        '            row = _pdm_subscores(pdm_score(mc, Trajectory(np.asarray(plan8, dtype=np.float32), S["samp8"]), S["samp"], S["sim"], S["scorer"]), names)\n'
        '            lab_dt = np.asarray(lab0).dtype\n            dts.add(str(lab_dt))\n', "row")
g = rep(g, "                worst = max(worst, abs(float(np.clip(row[names[c]], 0.0, 1.0)) - float(lab0[k])))\n",
        "                dk = identity_diff(row[names[c]], lab0[k], lab_dt)\n"
        "                worst = max(worst, dk)\n"
        "                worst64 = max(worst64, abs(float(np.clip(row[names[c]], 0.0, 1.0)) - float(lab0[k])))\n"
        "                if dk > 0.0:\n"
        "                    diffs.append({\"token\": meta.get(\"token\"), \"component\": c, \"navsim\": float(row[names[c]]), \"label\": float(lab0[k])})\n", "worst")
g = rep(g, '    return {"ran": True, "tokens": n, "max_abs_diff": worst}\n',
        '    return {"ran": True, "tokens": n, "max_abs_diff": worst, "compared_at_dtype": sorted(dts), "max_abs_diff_float64": worst64, "diffs": diffs[:24]}\n', "ret")
assert "\r\n" not in g

CHK = '''def chk_l5_identity_dtype():
    """G-LIVE L5 (SFTT gate 2, 2026-11-23): the sub-scores are read from a PDMResults DATACLASS or a DataFrame row, and the identity is judged at the LABEL's dtype.
    Numbers MEASURED on the pod (l5probe): ep NAVSIM 0.9281340780250414 vs label 0.9281340837478638 == float32(NAVSIM)."""
    import dataclasses
    names = {"nc": "no_at_fault_collisions", "dac": "drivable_area_compliance", "ep": "ego_progress", "ttc": "time_to_collision_within_bound", "c": "comfort", "ddc": "driving_direction_compliance"}

    @dataclasses.dataclass
    class FakePDMResults:                                   # the navsim 3e8291b return type (fields as navsim/common/dataclasses.py:557)
        no_at_fault_collisions: float = 1.0
        drivable_area_compliance: float = 1.0
        ego_progress: float = 0.9281340780250414
        time_to_collision_within_bound: float = 1.0
        comfort: float = 1.0
        driving_direction_compliance: float = 1.0
        score: float = 0.97

    class FakeFrame:                                        # the one-row DataFrame form: .iloc[0][col]
        class _I:
            def __getitem__(self, i):
                return dict(dataclasses.asdict(FakePDMResults()))
        iloc = _I()
    sd = GC._pdm_subscores(FakePDMResults(), names)
    ok(sd["ego_progress"] == 0.9281340780250414 and set(sd) == set(names.values()), "L5: a PDMResults dataclass is read by attribute")
    sf = GC._pdm_subscores(FakeFrame(), names)
    ok(sf == sd, "L5: a one-row DataFrame form is read by column, identically")
    nav, lab32 = 0.9281340780250414, float(np.float32(0.9281340780250414))
    ok(lab32 == 0.9281340837478638, "L5: the measured label IS float32(NAVSIM) (the pod's numbers)")
    d32 = GC.identity_diff(nav, lab32, np.float32)
    ok(d32 == 0.0 and GC.judge_harness_identity({"ran": True, "tokens": 8, "max_abs_diff": d32}) == [], "L5: a float32 label equal to float32(NAVSIM) is the identity (PASS)")
    ulp = float(np.nextafter(np.float32(lab32), np.float32(2.0)))
    d1 = GC.identity_diff(nav, ulp, np.float32)
    ok(d1 > 1e-9 and GC.judge_harness_identity({"ran": True, "tokens": 8, "max_abs_diff": d1}) != [], "L5: a float32 label ONE ulp away FAILS")
    d64 = GC.identity_diff(nav, lab32, np.float64)
    ok(d64 > 1e-9 and GC.judge_harness_identity({"ran": True, "tokens": 8, "max_abs_diff": d64}) != [], "L5: float64 labels are compared at float64 (5.7e-9 > 1e-9 FAILS)")
    ok(GC.identity_diff(nav, nav, np.float64) == 0.0, "L5: an exact float64 label is the identity")
    ok(GC.identity_diff(1.25, 1.0, np.float32) == 0.0 and GC.identity_diff(-0.5, 0.0, np.float32) == 0.0, "L5: NAVSIM's value is clipped to [0, 1] first")


'''
t = rep(t, "CHECKS = {", CHK + "CHECKS = {", "chk")
t = rep(t, '"field_reach_active": chk_field_reach_active_sample, ', '"field_reach_active": chk_field_reach_active_sample, "l5_identity_dtype": chk_l5_identity_dtype, ', "reg")
MUT = ('    ("gate_checks.py", "    return abs(float(np.asarray(np.clip(float(navsim_value), 0.0, 1.0), dtype=label_dtype)) - float(label_value))", '
       '"    return abs(float(np.clip(float(navsim_value), 0.0, 1.0)) - float(label_value))", "l5_identity_dtype"),   # compare at float64 regardless of the label dtype (the gate-2 FAIL)\n'
       '    ("gate_checks.py", "    if hasattr(res, \\"iloc\\"):\\n        row = res.iloc[0]", "    if True:\\n        row = res.iloc[0]", "l5_identity_dtype"),   # always .iloc (the gate-2 AttributeError)\n')
anchor = '    ("gate_checks.py", "    out[\\"active_index\\"] = idx[active[0]]'
assert t.count(anchor) == 1, ("mut anchor", t.count(anchor))
t = t.replace(anchor, MUT + anchor)
assert "\r\n" not in t
open(go, "w", encoding="utf-8", newline="\n").write(g)
open(to, "w", encoding="utf-8", newline="\n").write(t)
print("ZZPATCH_L5B_OK")
