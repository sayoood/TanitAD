"""The A7 gate + verdict code, tested on SYNTHETIC panels whose answers are known in advance.

⛔ Nothing here is an A7 result. These are fabricated arm directories with ANALYTIC targets
(E, F and R computed by hand from literal values) so that the code which will read ~7 GPU-hours
of real arms is proven before that GPU is spent -- the "analysis-time failure after the paid
rollout" trap (CLAUDE.md, t1_eval `selgap`).

Run: python -m pytest -q test_a7_verdict_code.py   (PYTHONPATH must reach taniteval + torch)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import a7_analyze as AN          # noqa: E402
import a7_check_arm as CK        # noqa: E402

N_CH = 50


def _argv(arm, pre, seed, d):
    return ["--arm", "hier", "--size", "tiny", "--out", str(d / "run"), "--device", "cuda",
            "--seed", str(seed), "--steps", "2000", "--batch", "2", "--workers", "0",
            "--trunk", "timm", "--trunk-name", "resnet34.a1_in1k", "--trunk-in-channels", "9",
            "--trunk-pretrained" if pre else "--no-trunk-pretrained",
            "--trunk-chunk-ckpt", "1", "--trunk-frozen-bn",
            "--trunk-bn-recalib", "256", "--trunk-bn-recalib-seed", "0",
            "--eval-every", "2000", "--eval-batches", "500",
            "--eval-window-dump", str(d / "eval_windows.jsonl")]


def _make_arm(panel, arm, pre, seed, traj_level, stats_shift=0.0, frac_pattern=True):
    d = panel / arm
    (d / "run").mkdir(parents=True)
    g = torch.Generator().manual_seed(1)       # SAME windows for every arm
    base = torch.rand(1000, generator=g).double()
    rows, fr = [], []
    for i in range(1000):
        frac = 0.5 if (frac_pattern and i % 7 == 0) else 1.0
        fr.append(frac)
        rows.append({"step": 2000, "episode_id": 10_000 + i // 25,
                     "traj": float(traj_level + 0.1 * base[i]), "slot_valid_frac": frac})
    agg = CK._rebuild_eval_traj(rows)          # the aggregate the trainer would have written
    ev = {"step": 2000, "eval_traj": agg, "eval_windows": 1000, "eval_loss": 10 * agg,
          "eval_lon": agg, "eval_lon_tac": agg, "eval_lat": agg, "eval_lat_tac": agg,
          "eval_tac_v6": agg, "eval_goal_tac": agg, "eval_anchor_acc": 1.0 / agg,
          "eval_route": agg}
    (d / "run" / "metrics.jsonl").write_text(json.dumps({"step": 1990, "loss": 1.0}) + "\n"
                                             + json.dumps(ev) + "\n", encoding="utf-8")
    (d / "eval_windows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                                          encoding="utf-8")
    start = {"mean": torch.linspace(-1, 1, N_CH).double() + (0.0 if pre else 0.7) + stats_shift,
             "var": torch.linspace(0.5, 2, N_CH).double() * (1.0 if pre else 3.0)
             * (1 + stats_shift), "n_bn": 36}
    torch.save({"start": start, "before": start, "window_idx": list(range(256))},
               d / "run" / "bn_recalib_stats.pt")
    stale = {"bn_staleness_var": 0.01 if pre else 0.02, "bn_staleness_mean": 0.01,
             "n_channels": N_CH}
    fin = {"step": 2000, "freeze_held": True, "staleness": stale, "staleness_error": None}
    (d / "run" / "bn_recalib.json").write_text(json.dumps(fin), encoding="utf-8")
    stamp = {"n_windows": 256, "seed": 0, "windows_sha12": "abcdefabcdef", "n_batches": 128,
             "n_images": 3072, "n_bn": 36, "stats_sha12": "x" * 12, "changed": True,
             "var_median_before": 1.0 if not pre else 0.3, "var_median_after": 0.2,
             "chunk_bypassed": True, "final": fin}
    (d / "run" / "config.json").write_text(json.dumps(
        {"argv": _argv(arm, pre, seed, d), "trunk_bn_recalib": stamp}), encoding="utf-8")
    (d / "run" / "summary.json").write_text(json.dumps({"done": True, "step": 2000}),
                                            encoding="utf-8")
    return d


def _panel(tmp_path, levels, shift_in_s1=0.0):
    p = tmp_path / "panel"
    for arm, pre, seed in (("A7-IN-s0", 1, 0), ("A7-IN-s1", 1, 1),
                           ("A7-RND-s0", 0, 0), ("A7-RND-s1", 0, 1)):
        d = _make_arm(p, arm, pre, seed, levels[arm],
                      stats_shift=shift_in_s1 if arm == "A7-IN-s1" else 0.0)
        rep = CK.check(d, pre, seed)
        (d / "a7_arm_check.json").write_text(json.dumps(rep, default=str), encoding="utf-8")
    return p


def test_the_gate_passes_a_well_formed_arm_and_rebuilds_exactly(tmp_path):
    d = _make_arm(tmp_path, "A7-RND-s0", 0, 0, 1.0)
    rep = CK.check(d, 0, 0)
    assert rep["status"] == "VALID", rep["failures"]
    assert rep["facts"]["dump_rebuild"]["rel"] < 1e-12
    # the plain mean is NOT the aggregate when futures are partial (A7.8 item 3)
    assert rep["facts"]["dump_rebuild"]["plain_row_mean"] != pytest.approx(
        rep["facts"]["dump_rebuild"]["aggregate"], rel=1e-6)


@pytest.mark.parametrize("breakage, needle", [
    ("freeze", "MOVED"), ("seed", "--seed"), ("rows", "dump rows"), ("identity", "identity")])
def test_the_gate_REFUSES_each_defect(tmp_path, breakage, needle):
    d = _make_arm(tmp_path, "A7-RND-s0", 0, 0, 1.0)
    if breakage == "freeze":
        j = json.loads((d / "run" / "bn_recalib.json").read_text(encoding="utf-8"))
        j["freeze_held"] = False
        (d / "run" / "bn_recalib.json").write_text(json.dumps(j), encoding="utf-8")
        rep = CK.check(d, 0, 0)
    elif breakage == "seed":
        rep = CK.check(d, 0, 1)
    elif breakage == "rows":
        lines = (d / "eval_windows.jsonl").read_text(encoding="utf-8").splitlines()[:999]
        (d / "eval_windows.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        rep = CK.check(d, 0, 0)
    else:   # a RND arm that did not start at the identity: the confound was not present
        c = json.loads((d / "run" / "config.json").read_text(encoding="utf-8"))
        c["trunk_bn_recalib"]["var_median_before"] = 0.9
        (d / "run" / "config.json").write_text(json.dumps(c), encoding="utf-8")
        rep = CK.check(d, 0, 0)
    assert rep["status"] == "INVALID"
    assert any(needle in f for f in rep["failures"]), rep["failures"]


def test_an_unreadable_artifact_is_INCONCLUSIVE_never_valid(tmp_path):
    d = _make_arm(tmp_path, "A7-IN-s0", 1, 0, 1.0)
    (d / "run" / "metrics.jsonl").write_bytes(b"\xff\xfe not json")
    assert CK.check(d, 1, 0)["status"] == "INCONCLUSIVE"


# ------------------------------------------------------------ the verdict, analytic targets
# window traj = level + 0.1*u, identical u across arms, so every arm's slot-weighted aggregate
# is level + c for ONE constant c: E and F are then exact functions of the literal levels.
def test_SUPPORTS_with_hand_computed_R(tmp_path):
    lv = {"A7-IN-s0": 1.00, "A7-IN-s1": 1.10, "A7-RND-s0": 1.50, "A7-RND-s1": 1.60}
    res = AN.analyze(_panel(tmp_path, lv))
    h = res["headline_A7_6"]
    assert h["E"] == pytest.approx(0.5, abs=1e-9)        # (1.55 - 1.05)
    assert h["F"] == pytest.approx(0.1, abs=1e-9)        # max(0.10, 0.10)
    assert h["R"] == pytest.approx(5.0, rel=1e-6)
    assert h["verdict"] == "SUPPORTS" and h["all_four_cross_pairs_favour_IN"]
    assert res["identity_control_A7_3a"]["passed"] is True
    b = res["bootstrap_supporting_A7_8"]["s0"]["slot_weighted_ratio"]
    assert b["delta"] == pytest.approx(0.5, abs=1e-3) and b["separated"] is True
    assert res["families_losses_not_metrics"]["TACTICAL"]["eval_anchor_acc"]["E"] > 0, (
        "an ACCURACY must be sign-flipped so positive still means ImageNet better")


def test_UNDERPOWERED_is_never_called_refuted(tmp_path):
    lv = {"A7-IN-s0": 1.00, "A7-IN-s1": 1.40, "A7-RND-s0": 1.20, "A7-RND-s1": 1.30}
    h = AN.analyze(_panel(tmp_path, lv))["headline_A7_6"]
    assert h["E"] == pytest.approx(0.05, abs=1e-9) and h["F"] == pytest.approx(0.4, abs=1e-9)
    assert h["R"] == pytest.approx(0.125, rel=1e-6)
    assert h["verdict"] == "UNDERPOWERED (not REFUTED)"


def test_RANDOM_AHEAD(tmp_path):
    lv = {"A7-IN-s0": 1.50, "A7-IN-s1": 1.60, "A7-RND-s0": 1.00, "A7-RND-s1": 1.10}
    h = AN.analyze(_panel(tmp_path, lv))["headline_A7_6"]
    assert h["R"] == pytest.approx(-5.0, rel=1e-6) and h["verdict"] == "RANDOM AHEAD"


def test_VOID_when_the_ImageNet_arms_recalibrate_differently(tmp_path):
    lv = {"A7-IN-s0": 1.00, "A7-IN-s1": 1.10, "A7-RND-s0": 1.50, "A7-RND-s1": 1.60}
    res = AN.analyze(_panel(tmp_path, lv, shift_in_s1=0.2))
    assert res["identity_control_A7_3a"]["passed"] is False
    assert res["headline_A7_6"]["verdict"] == "VOID"


def test_BANKING_sha12s_every_episode_id_and_refuses_a_UUID(tmp_path):
    """The Master Mind's banking list, with clip identity only as sha12 (CLAUDE/MM rule)."""
    import gzip
    import hashlib
    import a7_bank as BK
    lv = {"A7-IN-s0": 1.00, "A7-IN-s1": 1.10, "A7-RND-s0": 1.50, "A7-RND-s1": 1.60}
    p = _panel(tmp_path, lv)
    for arm in lv:
        (p / arm / "train.log").write_text("[v3] step 2000 loss 1.0\n", encoding="utf-8")
    raw = tmp_path / "raw"
    rep = BK.bank(p, raw)
    assert not rep["refused_uuid"]
    with gzip.open(raw / "A7-IN-s0" / "eval_windows.sha12.jsonl.gz", "rt",
                   encoding="utf-8") as f:
        rows = [json.loads(x) for x in f if x.strip()]
    assert len(rows) == 1000 and all("episode_id" not in r for r in rows)
    # the literal: episode 10000's sha12, computed here independently of the banker
    assert rows[0]["episode_sha12"] == hashlib.sha256(b"10000").hexdigest()[:12]
    assert len({r["episode_sha12"] for r in rows}) == 40       # clusters preserved: 1000 / 25
    # a UUID in a text artifact is REFUSED, not banked. ⚠️ Assembled at runtime: a literal
    # UUID in this SOURCE would itself trip the repo's clip-id guard (tools/clipid_scan.py).
    fake = "-".join(("0123abcd", "4567", "89ab", "cdef", "0123456789ab"))
    (p / "A7-RND-s1" / "train.log").write_text("clip %s\n" % fake, encoding="utf-8")
    rep2 = BK.bank(p, tmp_path / "raw2")
    assert any(s.endswith("train.log") for s in rep2["refused_uuid"])
    assert not (tmp_path / "raw2" / "A7-RND-s1" / "train.log").exists()


def test_VOID_when_argv_differs_beyond_the_allowed_flags(tmp_path):
    lv = {"A7-IN-s0": 1.00, "A7-IN-s1": 1.10, "A7-RND-s0": 1.50, "A7-RND-s1": 1.60}
    p = _panel(tmp_path, lv)
    c = json.loads((p / "A7-RND-s1" / "run" / "config.json").read_text(encoding="utf-8"))
    c["argv"][c["argv"].index("--batch") + 1] = "4"
    (p / "A7-RND-s1" / "run" / "config.json").write_text(json.dumps(c), encoding="utf-8")
    res = AN.analyze(p)
    assert res["headline_A7_6"]["verdict"] == "VOID"
    assert res["argv_audit_identical_except_allowed"]["A7-RND-s1"] is False
