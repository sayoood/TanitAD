"""test_gapmarginage.py -- card ADD119 (GAPMARGINAGE10K): the generated queue, reader and driver against their generator, the tables of the card, the two name routines on
random payloads, and the queue's dry run. The kit of card ADD117 is taken from ../gapmargin (same relative place in the pod stage). No pod result file is opened."""
import io
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
BASH = shutil.which("bash")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "read"))
import make_gapmarginage_kit as MK  # noqa: E402
import read_gapmarginage10k as RG  # noqa: E402

P = "GAPMARGINAGE10K-"


def rd(p):
    return io.open(p, "rb").read().replace(b"\r\n", b"\n").decode("utf-8")


def test_generated_files_equal_the_generator_output(tmp_path):
    texts = MK.make_all()
    for key, (_, rel) in MK.BUILD.items():
        assert rd(os.path.join(HERE, rel)) == texts[key], key
    assert MK.main(["--out-dir", str(tmp_path)]) == 0
    with pytest.raises(FileExistsError):
        MK.main(["--out-dir", str(tmp_path)])


def test_generator_refuses_a_changed_source(monkeypatch):
    src = dict(MK.SOURCES)
    src["reader"] = (src["reader"][0], "0" * 16)
    monkeypatch.setattr(MK, "SOURCES", src)
    with pytest.raises(SystemExit):
        MK.make_all()


def test_tables_of_the_card():
    assert RG.CARD == "PREREG_V8F_PROOF_ADD119" and RG.PRE == "GAPMARGINAGE10K" and RG.SETTINGS == ("K30", "K60") and RG.MARGIN_ARMS == ("A90", "A80") and RG.CLOSE_TOL == 0.0045
    assert RG.NEW_ARMS == {"K30": ("A90", "A80"), "K60": ("A90", "A80")}
    assert RG.SPEC["K30"]["P"] == 30 and RG.SPEC["K30"]["grid_file"] == "grid_p30.json" and RG.SPEC["K60"]["P"] == 60 and RG.SPEC["K60"]["grid_file"] == "grid_p60.json"
    assert all(RG.SPEC[S]["readout"] == "K" and RG.SPEC[S]["every"] == 5 for S in RG.SETTINGS)
    assert RG.SPEC["K30"]["runs"] == {"A90": "RQ50KF5P30Hb20gA90", "A80": "RQ50KF5P30Hb20gA80", "GEc": "RQ50KF5P30Hb20gGEc", "RLc": "RLcKF5P30Hb20", "RR": "RQ50KF5P30Hb20"}
    assert RG.SPEC["K60"]["runs"] == {"A90": "RQ50KF5P60Hb20gA90", "A80": "RQ50KF5P60Hb20gA80", "GEc": "RQ50KF5P60Hb20gGEc", "RLc": "RLcKF5P60Hb20", "RR": "RQ50KF5P60Hb20"}
    assert RG.EXPECTED == {"K30": {"GEc": (0.0062, 0.0839, 0.5154), "RR": (0.0954, 0.1339, 0.4954), "RLc": (0.0000, 0.0714, 0.4908)},
                           "K60": {"GEc": (0.0137, 0.0738, 0.6438), "RR": (0.1096, 0.1230, 0.6575), "RLc": (0.0137, 0.0410, 0.6301)}}
    assert RG.DONE_RE.match("ZZ GAPMARGINAGE10K DONE 2026-11-22T17:00:00Z all steps done") and not RG.DONE_RE.match("ZZ GAPMARGIN10K DONE 2026-11-22T17:00:00Z all steps done")


def payload(rng):
    st = {}
    for S in RG.SETTINGS:
        pairs = {}
        for pr in RG.PAIRS:
            pairs[pr] = {}
            for mk in "UCF":
                lo = float(rng.choice([-0.03, -0.01, 0.0, 0.004]))
                hi = lo + float(rng.choice([0.0, 0.008, 0.02, 0.05]))
                pairs[pr][mk] = {"interval": [lo, hi], "pooled": 0.5 * (lo + hi)}
        lv = {a: {"C": 0.0, "U": float(rng.choice([0.03, 0.0365, 0.041, 0.0411, 0.06])), "F": 0.2} for a in RG.ARMS}
        lv["RLc"]["U"] = 0.0365
        st[S] = {"pairs": pairs, "levels": lv}
    return {"settings": st}


def test_the_two_name_routines_agree_on_random_payloads():
    rng = np.random.default_rng(11)
    seen = set()
    for _ in range(400):
        pl = payload(rng)
        a, b = RG.names_of(pl), RG.derive_all(pl)
        assert a == b
        assert all(v.startswith(P) for v in a.values())
        seen |= {v.split("-", 2)[2] for k, v in a.items() if k.endswith("_OVERALL")}
    assert seen == {"MARGIN-DOES-NOT-CLOSE-UNSAFE", "MARGIN-CLOSES-UNSAFE-AT-A-PRICE", "MARGIN-CLOSES-UNSAFE-AT-NO-SEPARATED-PRICE"}


def test_closing_needs_the_lower_interval_and_the_bound():
    rng = np.random.default_rng(1)
    pl = payload(rng)
    for S in RG.SETTINGS:
        for pr in RG.PAIRS:
            for mk in "UCF":
                pl["settings"][S]["pairs"][pr][mk]["interval"] = [-0.01, 0.01]
        for a in RG.MARGIN_ARMS:
            pl["settings"][S]["levels"][a]["U"] = 0.05
    assert RG.names_of(pl)["K30_OVERALL"] == P + "K30-MARGIN-DOES-NOT-CLOSE-UNSAFE"
    pl["settings"]["K30"]["pairs"]["A90-VS-GEc"]["U"]["interval"] = [-0.03, -0.01]
    assert RG.names_of(pl)["K30_OVERALL"] == P + "K30-MARGIN-DOES-NOT-CLOSE-UNSAFE" and RG.names_of(pl)["K30_A90_UNSAFE"] == P + "K30-A90-LOWERS-UNSAFE"      # lowered, level above the bound
    pl["settings"]["K30"]["levels"]["A90"]["U"] = 0.0365 + 0.0045
    assert RG.names_of(pl)["K30_OVERALL"] == P + "K30-MARGIN-CLOSES-UNSAFE-AT-NO-SEPARATED-PRICE"
    pl["settings"]["K30"]["pairs"]["A90-VS-GEc"]["F"]["interval"] = [0.01, 0.03]
    assert RG.names_of(pl)["K30_OVERALL"] == P + "K30-MARGIN-CLOSES-UNSAFE-AT-A-PRICE" and RG.names_of(pl)["K60_OVERALL"] == P + "K60-MARGIN-DOES-NOT-CLOSE-UNSAFE"


def test_queue_dry_run_prints_the_eight_rollouts(tmp_path):
    env = dict(os.environ)
    env.update({"DRY": "1", "GA_ROOT": str(tmp_path / "root")})
    r = subprocess.run([BASH, os.path.join(HERE, "queue", "pod_gapmarginage10k_queue.sh").replace(os.sep, "/")], env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    assert not os.path.exists(str(tmp_path / "root"))
    cmds = [x for x in r.stdout.splitlines() if x.startswith("CMD: DISTRULE_DIR")]
    assert len(cmds) == 8, r.stdout[-800:] + r.stderr[-400:]
    want = [(arm, pre, k) for pre in ("30", "60") for arm in ("A90", "A80") for k in (0, 1)]
    for x, (arm, pre, k) in zip(cmds, want):
        assert "--vis-gap-file /workspace/v8proof/gapmargin/gapfiles/VGAPLE_%s.npz --vis-closing" % arm in x
        assert "/ckpt/UE_s%d/A/ckpt_last.pt" % k in x and ("--preroll-steps %s" % pre) in x and ("grid_p%s.json" % pre) in x and "--rule-every 5" in x and "--form RR" in x
        assert ("I10RQ50KF5P%sHb20g%s" % (pre, arm)) in x and "eval_grid_windows.json" not in x
    assert "DRY DONE" in r.stdout
