"""Tests for field_read.py: each rule branch, each control, and the deliberate-regression arms (swap arms, drop a token, perturb one PDMS,
tamper a seam) that must go RED. Synthetic data only; no NAVSIM, no GPU. Run the WHOLE file (module fixtures), never -k."""
import csv
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import field_read as FR      # noqa: E402
import p5c_read as PR        # noqa: E402

NB = 400
COLS = {"nc": "no_at_fault_collisions", "dac": "drivable_area_compliance", "ddc": "driving_direction_compliance", "ep": "ego_progress",
        "ttc": "time_to_collision_within_bound", "c": "comfort", "pdms": "score"}


def mk_rows(n_logs=24, per=25, seed=0, f_dac_fix=0.0, f_ep_shift=0.0, f_pdms_shift=0.0, names=("F", "C")):
    """rows dict like transp_score's. C has ~12 % DAC zeros; F repairs `f_dac_fix` of them (pdms rises with the repair)."""
    rng = np.random.default_rng(seed)
    rows = {}
    for li in range(n_logs):
        for k in range(per):
            t = f"{li:03d}{k:03d}"
            base_dac = (rng.random() > 0.12)
            ep = float(np.clip(rng.normal(0.7, 0.1), 0.05, 1))
            arms = {}
            for nm in names:
                dac = 1.0 if base_dac else (1.0 if (nm == names[0] and rng.random() < f_dac_fix) else 0.0)
                e = float(np.clip(ep + (f_ep_shift if nm == names[0] else 0.0), 0.0, 1))
                pdms = dac * e + (f_pdms_shift if nm == names[0] else 0.0)
                tr = {"nc": 1.0, "dac": dac, "ddc": 1.0, "ttc": 1.0, "ep": e, "c": 1.0, "pdms": pdms}
                arms[nm] = {"tracked": dict(tr), "transparent": dict(tr, pdms=pdms * 0.99)}
            rows[t] = {"token": t, "log": f"log{li:03d}", "arms": arms}
    return rows


def mk_geo(rows, names=("F", "C"), f_along=1.0, f_spread=1.0, seed=1):
    rng = np.random.default_rng(seed)
    toks = sorted(rows)
    a = rng.normal(8.0, 1.5, len(toks))
    s = rng.normal(5.0, 0.7, len(toks))
    return {names[0]: {"along4": a * f_along, "spread4": s * f_spread}, names[1]: {"along4": a.copy(), "spread4": s.copy()}}


def prep(rows, names=("F", "C")):
    P = PR.per_token(rows, list(names))
    return P, P["logs"]


def fam_pass():
    return {"status": "PASS", "adverse": [], "components_compared": 3}


def read(rows, geo, fam=None, controls=True, rep=None, names=("F", "C")):
    P, logs = prep(rows, names)
    c = FR.core(P, logs, names[0], names[1], geo, NB)
    c.pop("_boot")
    return c, FR.decide(c, fam or fam_pass(), controls, rep)


# ------------------------------------------------------------------------------------------------------ rule branches
def test_adopt_shaped_prints_single_seed_never_adopt():
    rows = mk_rows(f_dac_fix=0.9, seed=3)
    c, v = read(rows, mk_geo(rows))
    assert c["E1"]["tracked"]["lo"] > 0 and c["E2"]["dac_zero_share_diff_pp"]["hi"] < 0
    assert v["verdict"] == "ADOPT-SHAPED" and v["wording"] == FR.SINGLE_SEED
    assert v["verdict"] != "ADOPT"


def test_adopt_only_with_replicates_and_floor():
    rows = mk_rows(f_dac_fix=0.9, seed=3)
    c, _ = read(rows, mk_geo(rows))
    good = {"fs_minus_cs_e1": {"point": 1.0}, "fs_minus_cs_dz": {"point": -1.0}, "floor": 0.0}
    assert read(rows, mk_geo(rows), rep=good)[1]["verdict"] == "ADOPT"
    assert read(rows, mk_geo(rows), rep=dict(good, floor=99.0))[1]["verdict"] == "NOT PROVEN"          # E1 lo not above the seed floor
    assert read(rows, mk_geo(rows), rep=dict(good, fs_minus_cs_e1={"point": -0.1}))[1]["verdict"] == "NOT PROVEN"     # replicate sign not reproduced
    assert read(rows, mk_geo(rows), rep=dict(good, fs_minus_cs_dz={"point": 0.2}))[1]["verdict"] == "NOT PROVEN"


def test_size_label_large_vs_small():
    rows = mk_rows(f_dac_fix=1.0, f_pdms_shift=0.0, seed=4)
    c, v = read(rows, mk_geo(rows), rep={"fs_minus_cs_e1": {"point": 1.0}, "fs_minus_cs_dz": {"point": -1.0}, "floor": 0.0})
    assert c["E2"]["relative_cut_of_dac_zero_share"] <= -0.30
    assert v["verdict"] == "ADOPT"
    assert v["size_label"] == ("LARGE" if c["E1"]["tracked"]["point"] >= 1.0 else "SMALL")


def test_swapped_arms_flip_to_refuted_RED():
    """deliberate regression: the same data read with F and C swapped must NOT keep the ADOPT-shaped verdict."""
    rows = mk_rows(f_dac_fix=0.9, seed=3)
    P, logs = prep(rows)
    geo = mk_geo(rows)
    c = FR.core(P, logs, "C", "F", geo, NB)       # arms swapped
    c.pop("_boot")
    v = FR.decide(c, fam_pass(), True)
    assert v["verdict"] == "REFUTED"
    assert c["K4"]["fired"] is True


def test_refuted_by_dac_zero_share_rise():
    rows = mk_rows(seed=5)
    for r in rows.values():                          # F breaks DAC on ~10 % of tokens
        pass
    rng = np.random.default_rng(9)
    for t, r in rows.items():
        if rng.random() < 0.10:
            r["arms"]["F"]["tracked"]["dac"] = 0.0
            r["arms"]["F"]["tracked"]["pdms"] = 0.0
    c, v = read(rows, mk_geo(rows))
    assert v["verdict"] == "REFUTED"


def test_trade_when_adverse_check_fires():
    rows = mk_rows(f_dac_fix=0.9, seed=3)
    c, v = read(rows, mk_geo(rows, f_along=0.9))     # picked plan travels 10 % less far: A2
    assert c["A2"]["adverse"] and not c["A1"]["adverse"]
    assert v["verdict"] == "TRADE"
    c, v = read(rows, mk_geo(rows, f_spread=0.7))    # fan collapses: A3
    assert c["A3"]["adverse"] and v["verdict"] == "TRADE"


def test_a1_needs_both_separation_and_point():
    rows = mk_rows(f_dac_fix=0.9, f_ep_shift=-0.2, seed=3)
    c, _ = read(rows, mk_geo(rows))
    assert c["A1"]["ci"]["hi"] < 0 and c["A1"]["adverse"] == (c["A1"]["ci"]["point"] <= -1.0)
    rows = mk_rows(f_dac_fix=0.9, f_ep_shift=-0.2, seed=3)
    for r in rows.values():                           # small EP loss: separated but point > -1.0
        r["arms"]["F"]["tracked"]["ep"] = r["arms"]["C"]["tracked"]["ep"] - 0.002
    c, _ = read(rows, mk_geo(rows))
    assert c["A1"]["adverse"] is False


def test_family_guard_blocks_adopt_and_notevaluated_is_incomplete():
    rows = mk_rows(f_dac_fix=0.9, seed=3)
    bad = {"status": "FAIL", "adverse": [{"component": "cross_mae_m"}], "components_compared": 3}
    assert read(rows, mk_geo(rows), fam=bad)[1]["verdict"] == "NOT PROVEN"
    ne = {"status": "NOT-EVALUATED", "reason": "x", "adverse": [], "components_compared": 0}
    assert read(rows, mk_geo(rows), fam=ne)[1]["verdict"] == "INCOMPLETE"


def test_not_proven_when_no_effect():
    rows = mk_rows(seed=6)
    c, v = read(rows, mk_geo(rows))
    assert v["verdict"] == "NOT PROVEN"


def test_failed_control_gives_invalid_never_a_verdict():
    rows = mk_rows(f_dac_fix=0.9, seed=3)
    assert read(rows, mk_geo(rows), controls=False)[1]["verdict"] == "INVALID"
    assert read(rows, mk_geo(rows), controls=None)[1]["verdict"] == "INVALID"


# ------------------------------------------------------------------------------------------------------ controls
def test_ctrl_id_exact_zero_and_red_arm():
    rows = mk_rows(f_dac_fix=0.9, seed=3)
    P, logs = prep(rows)
    geo = mk_geo(rows)
    r = FR.ctrl_id(P, logs, "F", geo, NB)
    assert r["pass"] and r["nonzero_deltas"] == [] and r["verdict"] == "NOT PROVEN"
    # RED arm: the same detector must see a perturbation (compare F with a copy whose one PDMS moved by 1e-12)
    P2 = {k: v for k, v in P.items()}
    P2["Fp"] = {s: {m: np.array(P["F"][s][m]) for m in PR.METRICS} for s in PR.SCORERS}
    P2["Fp"]["tracked"]["pdms"][0] += 1e-12
    c = FR.core(P2, logs, "Fp", "F", dict(geo, Fp=geo["F"]), NB)
    c.pop("_boot")
    assert FR.zero_exact({"x": c["E1"]["tracked"]}) != []


def test_ctrl_set_detects_dropped_token_and_order_and_population():
    toks = [f"t{i}" for i in range(10)]
    assert FR.ctrl_set({"a": toks, "b": list(toks)})["pass"]
    assert not FR.ctrl_set({"a": toks, "b": toks[:-1]})["pass"]                # dropped token
    assert not FR.ctrl_set({"a": toks, "b": toks[::-1]})["pass"]               # same set, other order
    assert not FR.ctrl_set({"a": toks, "b": list(toks)}, n_expected=12146)["pass"]
    assert not FR.ctrl_set({"a": toks, "b": list(toks)}, None, ["l1"] * 10, 136)["pass"]


def test_per_token_refuses_a_dropped_token_in_one_arm():
    rows = mk_rows(n_logs=4, per=5)
    t = sorted(rows)[3]
    del rows[t]["arms"]["F"]
    with pytest.raises(PR.Refused):
        PR.per_token(rows, ["F", "C"])


def write_csv(path, rows, arm, perturb_first=0.0):
    toks = sorted(rows)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["token"] + list(COLS.values()))
        w.writeheader()
        for i, t in enumerate(toks):
            tr = rows[t]["arms"][arm]["tracked"]
            d = {"token": t, **{COLS[m]: tr[m] for m in COLS}}
            if i == 0:
                d["score"] = tr["pdms"] + perturb_first
            w.writerow(d)


def test_ctrl_csv_pass_and_one_perturbed_pdms_RED(tmp_path):
    pytest.importorskip("p5c_read_prep")
    rows = mk_rows(n_logs=5, per=6)
    p = str(tmp_path / "h.csv")
    write_csv(p, rows, "F")
    assert FR.ctrl_csv(rows, "F", p, sorted(rows))["pass"]
    write_csv(p, rows, "F", perturb_first=2e-9)                  # one PDMS off by 2e-9 > the 1e-9 bar
    r = FR.ctrl_csv(rows, "F", p, sorted(rows))
    assert not r["pass"] and r["n_over_bar"] == 1
    write_csv(p, rows, "F", perturb_first=5e-10)                 # inside the bar
    assert FR.ctrl_csv(rows, "F", p, sorted(rows))["pass"]


def write_seam(tmp_path, name, poses, pick=None, nprop=64):
    toks = np.array([f"t{i:04d}" for i in range(len(poses))])
    sp = str(tmp_path / f"{name}.npz")
    dp = str(tmp_path / f"{name}.proposals.npz")
    np.savez(sp, token=toks, fingerprint=toks, poses=poses.astype(np.float32), sampling=np.array([8.0, 0.5]), arm=np.array(name))
    prop = np.repeat(poses[:, None].astype(np.float32), nprop, axis=1)
    prop[:, 1:, 7, 1] += np.arange(1, nprop)[None, :] * 0.1          # a fan
    pk = np.zeros(len(poses), dtype=np.int64) if pick is None else pick
    np.savez(dp, token=toks, fingerprint=toks, proposals=prop, logits=np.zeros((len(poses), nprop, 6), np.float32), pick=pk, pick20=np.zeros((len(poses), 20, 3), np.float32),
             sampling=np.array([8.0, 0.5]))
    return sp, dp, [str(t) for t in toks]


def test_seam_geometry_and_ctrl_seam_ledger_tamper_RED(tmp_path):
    rng = np.random.default_rng(0)
    poses = rng.normal(0, 1, (12, 8, 3))
    poses[:, :, 0] = np.cumsum(np.abs(poses[:, :, 0]), axis=1)
    sp, dp, toks = write_seam(tmp_path, "armF", poses)
    g = FR.seam_geometry(sp, dp, toks)
    assert g["pick_is_seam_plan"]
    assert np.allclose(g["along4"], poses[:, 7, 0].astype(np.float32))
    assert (g["spread4"] > 0).all()
    led = str(tmp_path / "ledger.json")
    r1 = FR.ctrl_seam({"F": g}, led)
    assert r1["pass"] and r1["ledger"].startswith("recorded")
    assert FR.ctrl_seam({"F": g}, led)["pass"]                                  # re-check passes on the same bytes
    poses2 = poses.copy()
    poses2[0, 7, 0] += 1.0                                                      # one pose changes: the seam file's bytes change
    sp2, dp2, _ = write_seam(tmp_path, "armF", poses2)
    g2 = FR.seam_geometry(sp2, dp2, toks)
    r2 = FR.ctrl_seam({"F": g2}, led)
    assert not r2["pass"] and any("ledger" in x for x in r2["fails"])
    # the file changing between hashing and the re-check at read time
    g3 = dict(g, seam_sha256="0" * 64)
    assert not FR.ctrl_seam({"F": g3})["pass"]


def test_seam_dump_pick_must_be_the_seam_plan(tmp_path):
    rng = np.random.default_rng(1)
    poses = rng.normal(0, 1, (6, 8, 3))
    sp, dp, toks = write_seam(tmp_path, "armX", poses, pick=np.full(6, 5, dtype=np.int64))      # pick points at a different proposal
    g = FR.seam_geometry(sp, dp, toks)
    assert not g["pick_is_seam_plan"] and not FR.ctrl_seam({"X": g})["pass"]


def test_seam_refuses_wrong_sampling(tmp_path):
    poses = np.zeros((3, 8, 3))
    sp, dp, toks = write_seam(tmp_path, "arm0", poses)
    z = dict(np.load(sp))
    z["sampling"] = np.array([8.0, 0.25])
    np.savez(sp, **z)
    with pytest.raises(PR.Refused):
        FR.seam_geometry(sp, dp, toks)


def test_convert_report_binds_rows_to_the_seam(tmp_path):
    poses = np.random.default_rng(2).normal(0, 1, (4, 8, 3))
    sp, dp, toks = write_seam(tmp_path, "armF", poses)
    g = FR.seam_geometry(sp, dp, toks)
    cr = str(tmp_path / "cr.json")
    json.dump({"arms": {"F": {"seam_sha256": g["seam_sha256"]}}}, open(cr, "w"))
    assert FR.ctrl_seam({"F": g}, None, cr)["pass"]
    json.dump({"arms": {"F": {"seam_sha256": "1" * 64}}}, open(cr, "w"))
    assert not FR.ctrl_seam({"F": g}, None, cr)["pass"]


# ------------------------------------------------------------------------------------------------------ estimands and blocks
def test_ratio_ci_exactly_one_for_identical_arrays_and_directional():
    P, logs = prep(mk_rows(seed=7))
    bt = PR.Boot(logs, NB)
    v = np.random.default_rng(0).normal(8, 1, len(logs))
    r = FR.ratio_ci(bt, v, v)
    assert r["point"] == r["lo"] == r["hi"] == 1.0
    r = FR.ratio_ci(bt, v * 0.9, v)
    assert r["hi"] < 0.91 and r["lo"] > 0.89


def test_fan_spread_value():
    poses = np.zeros((1, 8, 3))
    poses[0, 7, 0] = 1.0
    prop = np.zeros((1, 64, 8, 3))
    prop[0, :, 7, 0] = np.arange(64)                           # points 0..63 on a line: mean pairwise distance = (n+1)/3
    e = prop[:, :, 7, :2]
    dd = np.sqrt(((e[:, :, None, :] - e[:, None, :, :]) ** 2).sum(-1))
    assert abs(dd.sum((1, 2))[0] / (64 * 63) - 65 / 3) < 1e-9


def fam_json(adverse=False, a="C", b="F"):
    def ent(lo, hi, pt):
        return {"status": "OK", "A": {"point": 1, "lo": 0.9, "hi": 1.1}, "B": {"point": 1, "lo": 0.9, "hi": 1.1}, "diff_B_minus_A": {"point": pt, "lo": lo, "hi": hi}}
    lat = {"cross_mae_m": ent(0.01, 0.2, 0.1) if adverse else ent(-0.1, 0.1, 0.0), "heading_mae_deg": ent(-0.1, 0.1, 0.0)}
    return {"label_a": a, "label_b": b, "estimator": "x", "frames": {"executed_tracked": {"families": {"longitudinal": {"along_mae_m": ent(-0.1, 0.1, 0.0)}, "lateral": lat, "tactical": {}}}}}


def test_family_guard_polarity_labels_and_missing():
    assert FR.family_guard(fam_json())["status"] == "PASS"
    g = FR.family_guard(fam_json(adverse=True))
    assert g["status"] == "FAIL" and g["adverse"][0]["component"] == "cross_mae_m"
    assert FR.family_guard(fam_json(a="C0", b="C1"))["status"] == "NOT-EVALUATED"        # wrong labels: the diff direction is unproven
    assert FR.family_guard(None)["status"] == "NOT-EVALUATED"


def test_strategic_never_paused_and_block_refuses_paused_json():
    assert "PAUSED" not in json.dumps(FR.strategic_block({"instrument": "s", "arms": {"F": {}}, "paired_vs_base": {}}, 5))
    with pytest.raises(PR.Refused):
        FR.strategic_block({"arms": {}, "note": "PAUSED"}, 5)
    assert FR.strategic_block(None, 5)["status"] == "NOT SUPPLIED"


def test_relabel_tactical_renames_and_drops_paused_stub():
    t = {"kind": "P5C_TACTICAL", "strategic": {"status": "PAUSED"}, "command_adherence": {"status": "COMPUTED", "arms": {"C1": {}, "C0": {}, "base": {}},
                                                                                          "paired_C1_minus_x": {"C1-C0": {}, "C1-base": {}}}, "selection_quality": {"C1": {}, "C0": {}}}
    r = FR.relabel_tactical(t)
    s = json.dumps(r)
    assert "PAUSED" not in s and '"C1"' not in s and '"C0"' not in s and "C1-C0" not in s
    assert "F-C" in s and "paired_F_minus_x" in r["command_adherence"]


def test_m1_block_and_selection_quality():
    rows = mk_rows(n_logs=10, per=10, seed=8)
    P, logs = prep(rows)
    n = len(logs)
    rng = np.random.default_rng(0)

    def T(shift):
        dac = (rng.random((n, 64)) > 0.3 + shift).astype(float)
        pd = rng.random((n, 64)) * 100
        pick = rng.integers(0, 64, n)
        return {"dac_tracked": dac, "dac_raw": dac, "pdms": pd, "pick": pick}
    TF, TC = T(-0.2), T(0.0)
    toks = P["tokens"]
    m = FR.m1_block(toks, logs, set(toks[:40]), TF, TC, NB)
    assert m["dac_tracked"]["F_minus_C_pp"]["point"] > 10 and "NOT rates" in m["population"]
    with pytest.raises(PR.Refused):
        FR.m1_block(toks, logs, {"nope"}, TF, TC, NB)
    s = FR.selection_block(TF, TC, NB, logs)
    assert set(s["F_minus_C"]) == {"actual", "random", "best_of_64"}
    assert set(s["actual_minus_random"]) == {"F", "C"}
    assert s["levels"]["F"]["random"]["point"] == pytest.approx(TF["pdms"].mean())


def test_m2_routing_flag_raw_up_tracked_not():
    rows = mk_rows(f_dac_fix=0.0, seed=10)
    rng = np.random.default_rng(2)
    for r in rows.values():                          # F's raw (transparent) DAC rises on 6 % of tokens, tracked unchanged
        if r["arms"]["C"]["transparent"]["dac"] == 0.0 and rng.random() < 0.9:
            r["arms"]["F"]["transparent"]["dac"] = 1.0
    P, logs = prep(rows)
    c = FR.core(P, logs, "F", "C", mk_geo(rows), NB)
    assert c["M2"]["raw_improves_tracked_does_not"] is True and "tracker-aware" in c["M2"]["routing"]


def test_dry_stamp_constant_has_em_dash():
    assert FR.DRY_STAMP == "DRY RUN — NOT A FIELD VERDICT"


def test_text_report_never_prints_paused_and_carries_tier():
    rows = mk_rows(f_dac_fix=0.9, seed=3)
    c, v = read(rows, mk_geo(rows))
    guard = FR.family_guard(fam_json())
    res = {"verdict": v, "tier": FR.TIER, "estimator": {"kind": "k", "draws": NB, "seed": FR.SEED, "answers": FR.ANSWERS}, "n_tokens": 1, "n_logs": 1, "arms": {},
           "controls": [{"control": "CTRL-SET", "pass": True}], "controls_all_pass": True, "endpoints": {k: c[k] for k in ("E1", "E2", "E3", "M2")},
           "six_subscores": c["six_subscores_tracked_and_transparent"], "adverse": {k: c[k] for k in ("A1", "A2", "A3")}, "K4": c["K4"],
           "M1": {"status": "NOT COMPUTED", "reason": "x"}, "families": FR.families_block(fam_json(), None, FR.strategic_block(None, 1), None, guard)}
    t = FR.text_report(res)
    assert "T1" in t and "PAUSED" not in t and "STRATEGIC" in t and "never pooled" in t


# ------------------------------------------------------------------------------------------------------ v2: M1 in the card's form (field_rawdac rows)
import field_rawdac as RD    # noqa: E402


def mk_rawdac_world(seed=21, n_logs=10, per=10):
    """tokens, logs, c3 set (first 40), F/C tables (tracked DAC 1.0 = compliant) and consistent field_rawdac rows (sim == tracked violation)."""
    rows = mk_rows(n_logs=n_logs, per=per, seed=seed)
    P, logs = prep(rows)
    toks = P["tokens"]
    n = len(toks)
    rng = np.random.default_rng(seed)

    def arm(shift):
        dac = (rng.random((n, 64)) > 0.3 + shift).astype(float)
        pick = rng.integers(0, 64, n)
        return {"dac_tracked": dac, "dac_raw": dac.copy(), "pdms": rng.random((n, 64)) * 100, "pick": pick}
    TF, TC = arm(-0.2), arm(0.0)
    c3 = set(toks[:40])

    def rd_rows(T):
        out = {}
        for i, t in enumerate(toks):
            if t in c3:
                sim = (T["dac_tracked"][i] == 0.0).astype(float)
                raw = np.where(rng.random(64) < 0.05, 1.0 - sim, sim)           # raw disagrees with sim on ~5 % of proposals
                out[t] = {"token": t, "log": logs[i], "pick": int(T["pick"][i]), "raw": raw.tolist(), "sim": sim.tolist()}
        return out
    return toks, logs, c3, TF, TC, rd_rows(TF), rd_rows(TC)


def test_rawdac_m1_columns_labelled_and_move_exactly_one_over_64_on_a_flipped_proposal_RED():
    toks, logs, c3, TF, TC, rf, rc = mk_rawdac_world()
    RF = FR.rawdac_arrays(rf, toks, c3, TF, TF["pick"], "F")
    RC = FR.rawdac_arrays(rc, toks, c3, TC, TC["pick"], "C")
    m = FR.m1_block(toks, logs, c3, TF, TC, NB, RF, RC)
    for k in ("dac_tracked", "raw_navsim_dac", "simulated_navsim_dac", "raw_transparent"):
        assert "F_minus_C_pp" in m[k], k
    assert m["raw_transparent"] is m["dac_raw"] and "M2_c3_pick_crosscheck" in m and "navsim_dac_violation(mode='raw')" in m["raw_navsim_dac_mapping"]
    base = m["raw_navsim_dac"]["F_minus_C_pp"]["point"]
    t0 = sorted(c3)[0]
    rf2 = {t: dict(r) for t, r in rf.items()}
    flag = list(rf2[t0]["raw"])
    flag[5] = 1.0 - flag[5]                                  # one proposal's raw flag flips
    rf2[t0]["raw"] = flag
    RF2 = FR.rawdac_arrays(rf2, toks, c3, TF, TF["pick"], "F")
    m2 = FR.m1_block(toks, logs, c3, TF, TC, NB, RF2, RC)
    moved = m2["raw_navsim_dac"]["F_minus_C_pp"]["point"] - base
    expect = (100.0 / 64.0) / len(c3) * (1.0 if rf[t0]["raw"][5] == 1.0 else -1.0)      # a violation removed raises the compliant share
    assert moved == pytest.approx(expect, abs=1e-9)
    assert m2["simulated_navsim_dac"]["F_minus_C_pp"]["point"] == m["simulated_navsim_dac"]["F_minus_C_pp"]["point"]      # sim and transparent untouched
    assert m2["raw_transparent"]["F_minus_C_pp"]["point"] == m["raw_transparent"]["F_minus_C_pp"]["point"]


def test_rawdac_swapped_files_and_dropped_c3_token_RED():
    toks, logs, c3, TF, TC, rf, rc = mk_rawdac_world(seed=22)
    with pytest.raises(PR.Refused, match="fail their controls"):     # F's rows read against C's table: CTRL-SIM (and the pick) fail
        FR.rawdac_arrays(rc, toks, c3, TF, TF["pick"], "F")
    RF = FR.rawdac_arrays(rf, toks, c3, TF, TF["pick"], "F")
    RC = FR.rawdac_arrays(rc, toks, c3, TC, TC["pick"], "C")
    a = FR.m1_block(toks, logs, c3, TF, TC, NB, RF, RC)["raw_navsim_dac"]["F_minus_C_pp"]["point"]
    b = FR.m1_block(toks, logs, c3, TC, TF, NB, RC, RF)["raw_navsim_dac"]["F_minus_C_pp"]["point"]
    assert a == pytest.approx(-b, abs=1e-9) and abs(a) > 1.0            # swapping the arms flips the sign of a non-trivial effect
    drop = dict(rf)
    drop.pop(sorted(c3)[3])
    with pytest.raises(PR.Refused, match="lack 1 of the 40 C3 tokens"):
        FR.rawdac_arrays(drop, toks, c3, TF, TF["pick"], "F")
    with pytest.raises(PR.Refused, match="both arms or neither"):
        FR.m1_block(toks, logs, c3, TF, TC, NB, RF, None)
    short = {t: dict(r, raw=r["raw"][:63]) for t, r in rf.items()}
    with pytest.raises(PR.Refused, match="without 64"):
        FR.rawdac_arrays(short, toks, c3, TF, TF["pick"], "F")


def test_rawdac_ctrl_sim_red_roll_drops_and_ctrl_pick_catches_wrong_pick():
    toks, logs, c3, TF, TC, rf, rc = mk_rawdac_world(seed=23)
    table = {t: {"tr": {"dac": TF["dac_tracked"][toks.index(t)]}} for t in rf}
    ok = RD.ctrl_sim(rf, table)
    assert ok["pass"] and ok["per_proposal_agreement"] == 1.0 and ok["n_tokens"] == 40 and ok["n_proposals"] == 40 * 64
    rolled = {t: dict(r, sim=list(np.roll(r["sim"], 1))) for t, r in rf.items()}       # the RED arm: the proposal axis rolled by one
    bad = RD.ctrl_sim(rolled, table)
    assert bad["per_token_agreement"] < ok["per_token_agreement"] and not bad["pass"]
    picks = {t: int(r["pick"]) for t, r in rf.items()}
    assert RD.ctrl_pick(rf, picks)["pass"]
    picks[sorted(picks)[0]] = (picks[sorted(picks)[0]] + 1) % 64
    assert not RD.ctrl_pick(rf, picks)["pass"]
    with pytest.raises(ValueError):
        RD.ctrl_sim(rf, {}, None)


def test_load_rawdac_binds_to_the_dump_sha_and_refuses_red_headers(tmp_path):
    dump = tmp_path / "d.npz"
    np.savez(dump, token=np.array(["a"]), pick=np.array([0]), proposals=np.zeros((1, 64, 8, 3), np.float32))
    rows = tmp_path / "rawdac_F_s0of1.jsonl"
    rows.write_text(json.dumps({"token": "a", "log": "l", "pick": 0, "raw": [0.0] * 64, "sim": [0.0] * 64}) + "\n", encoding="utf-8")
    hp = tmp_path / "rawdac_F_s0of1.jsonl.header.s0of1.json"
    with pytest.raises(PR.Refused, match="no .header"):
        FR.load_rawdac([str(rows)], str(dump), "F")
    hp.write_text(json.dumps({"proposals_sha256": FR.sha256_file(str(dump)), "red": False}), encoding="utf-8")
    assert set(FR.load_rawdac([str(rows)], str(dump), "F")) == {"a"}
    hp.write_text(json.dumps({"proposals_sha256": "0" * 64, "red": False}), encoding="utf-8")
    with pytest.raises(PR.Refused, match="sha256"):
        FR.load_rawdac([str(rows)], str(dump), "F")
    hp.write_text(json.dumps({"proposals_sha256": FR.sha256_file(str(dump)), "red": True}), encoding="utf-8")
    with pytest.raises(PR.Refused, match="RED"):
        FR.load_rawdac([str(rows)], str(dump), "F")


# ---- v3 (2026-11-20) POLARITY: min_ttc_s / headway_min_m / time_gap_min_s are higher-is-safer, corrected INSIDE field_read
def _old_fam_direction(m):
    """the inherited rule verbatim (an OLD harness export of p5c_read): the three margins fall through to 'lower'."""
    m = m.lower()
    if "within_" in m or "accuracy" in m or "kappa" in m:
        return "higher"
    if "bias" in m:
        return "magnitude"
    if "progress_ratio" in m:
        return "closer_to_1"
    return "lower"


def _margin_json(lo, hi, pt, k="min_ttc_s"):
    def ent(lo_, hi_, pt_):
        return {"status": "OK", "A": {"point": 1, "lo": 0.9, "hi": 1.1}, "B": {"point": 1, "lo": 0.9, "hi": 1.1}, "diff_B_minus_A": {"point": pt_, "lo": lo_, "hi": hi_}}
    return {"label_a": "C", "label_b": "F", "estimator": "x", "frames": {"executed_tracked": {"families": {
        "longitudinal": {k: ent(lo, hi, pt), "along_mae_m": ent(-0.1, 0.1, 0.0)}, "lateral": {"cross_mae_m": ent(-0.1, 0.1, 0.0)}, "tactical": {}}}}}


@pytest.mark.parametrize("k", ["min_ttc_s", "headway_min_m", "time_gap_min_s"])
def test_polarity_margin_rise_is_not_adverse_fall_is(k):
    g = FR.family_guard(_margin_json(0.157, 0.436, 0.297, k))          # the banked P5-C C1-vs-C0 min_ttc_s numbers: a separated RISE
    assert g["status"] == "PASS" and g["adverse"] == []
    g = FR.family_guard(_margin_json(-0.436, -0.157, -0.297, k))
    assert g["status"] == "FAIL" and [a["component"] for a in g["adverse"]] == [k] and g["adverse"][0]["direction"] == "higher"


def test_polarity_correction_holds_on_an_old_harness_and_prints_both_readings(monkeypatch):
    monkeypatch.setattr(PR, "fam_direction", _old_fam_direction)        # what FIELD_READ_P5C_DIR=F:/TanitAD_p5cread_tip imports: no margin rule
    j = _margin_json(0.157, 0.436, 0.297)
    assert PR.entry_adverse("min_ttc_s", j["frames"]["executed_tracked"]["families"]["longitudinal"]["min_ttc_s"]) is True      # the inherited reading is the defect
    g = FR.family_guard(j)
    assert g["status"] == "PASS"                                        # the corrected guard
    (c,) = g["polarity_corrections"]
    assert c["component"] == "min_ttc_s" and c["inherited_reading"] == {"direction": "lower", "adverse": True} and c["corrected_reading"] == {"direction": "higher", "adverse": False}
    assert "higher-is-safer" in c["reason"] or "bigger = safer" in c["reason"]
    gi = g["guard_status_under_inherited_rule"]
    assert gi["status"] == "FAIL" and gi["flipped_by_correction"] is True and gi["adverse_inherited_only"] == ["longitudinal.min_ttc_s"]


def test_polarity_no_correction_listed_when_nothing_changes():
    g = FR.family_guard(fam_json())
    assert g["polarity_corrections"] == [] and g["guard_status_under_inherited_rule"]["flipped_by_correction"] is False


def test_polarity_matcher_agrees_with_p5c_read_when_it_carries_one():
    if hasattr(PR, "is_safety_margin"):
        assert tuple(FR.SAFETY_MARGIN_HIGHER) == tuple(PR.SAFETY_MARGIN_HIGHER)
        for k in ("min_ttc_s", "longitudinal.min_ttc_s", "mean_headway_min_m", "lead_viol", "speed_mae_mps", "target_speed_acc.within_1.0_mps"):
            assert FR.is_safety_margin(k) == PR.is_safety_margin(k)


def test_RED_polarity_revert_is_caught(monkeypatch):
    """deliberate regression: drop the correction (empty margin tuple) on an old harness -> the rise is flagged adverse, which the tests above forbid."""
    monkeypatch.setattr(PR, "fam_direction", _old_fam_direction)
    monkeypatch.setattr(FR, "SAFETY_MARGIN_HIGHER", ())
    g = FR.family_guard(_margin_json(0.157, 0.436, 0.297))
    assert g["status"] == "FAIL"                                        # the defect is back; test_polarity_margin_rise_is_not_adverse_fall_is would be RED here
