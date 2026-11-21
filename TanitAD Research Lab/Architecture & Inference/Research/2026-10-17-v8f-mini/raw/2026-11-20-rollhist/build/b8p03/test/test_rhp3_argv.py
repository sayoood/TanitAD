"""Tests of rhp3_argv.py on the REAL pod argv files (pulled read-only from the finished RH / B0 arms) and the stock pair check. CPU only."""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
B = os.path.dirname(HERE)
sys.path.insert(0, B)
sys.path.insert(0, os.path.join(HERE, "ref", "tree", "stack"))
sys.path.insert(0, os.path.join(HERE, "ref", "tree", "stack", "scripts"))
import rhp3_argv as A  # noqa: E402
import rollhist_pair_check as P  # noqa: E402


def load(n):
    return A.load(os.path.join(B, "pod_ro", n))


def sha(s):
    import hashlib
    return hashlib.sha256(open(os.path.join(B, "pod_ro", f"bank_s{s}.json"), "rb").read()).hexdigest()


@pytest.mark.parametrize("s", [0, 1])
def test_real_rh_argv_gives_rhp3_with_exactly_two_changes(s):
    rh = load(f"argv_RH_s{s}.json")
    new, bad = A.build(rh, s, "/root/rollhist", bank_sha=sha(s))
    assert bad == [], bad
    f, g = A.flags(new), A.flags(rh)
    assert sorted(k for k in set(f) | set(g) if f.get(k) != g.get(k)) == ["--dagger-p", "--out"]
    assert f["--dagger-p"] == ["0.3"] and f["--out"] == [f"/root/rollhist/runs/RHp3_s{s}/train"]
    assert f["--dagger-bank"] == g["--dagger-bank"] and f["--dagger-bank-sha256"] == g["--dagger-bank-sha256"] == [sha(s)]
    assert f["--init-ckpt"] == g["--init-ckpt"] and f["--steps"] == ["3000"] and f["--seed"] == [str(s)] and f["--dagger-seed"] == g["--dagger-seed"]


@pytest.mark.parametrize("s", [0, 1])
def test_stock_pair_check_passes_rhp3_against_the_existing_b0(s):
    new, bad = A.build(load(f"argv_RH_s{s}.json"), s, "/root/rollhist", bank_sha=sha(s))
    assert P.pair_problems(new, load(f"argv_B0_s{s}.json"), bank_sha256=sha(s)) == []


def test_stock_pair_check_does_not_pin_the_dose_which_is_why_the_study_checker_does():
    new, _ = A.build(load("argv_RH_s0.json"), 0, "/root/rollhist", bank_sha=sha(0))
    new = A.setf(new, "--dagger-p", ["0.5"])
    assert P.pair_problems(new, load("argv_B0_s0.json"), bank_sha256=sha(0)) == []              # the stock check accepts any p > 0 (documented, not changed)


def test_refusals():
    rh = load("argv_RH_s0.json")
    for mut, key in ((A.setf(rh, "--dagger-p", ["0.2"]), "dagger-p"), (A.setf(rh, "--dagger-bank", ["/elsewhere/bank"]), "dagger-bank"), (A.setf(rh, "--out", ["/root/rollhist/runs/other/train"]), "out"),
                     (A.setf(rh, "--dagger-bank-readout", ["causal_g1"]), "declaration")):
        _, bad = A.build(mut, 0, "/root/rollhist", bank_sha=sha(0))
        assert bad and any(key in b for b in bad), (key, bad)
    _, bad = A.build(rh, 0, "/root/rollhist", bank_sha="0" * 64)                                # the bank changed since RH trained on it
    assert bad and any("bank" in b for b in bad)
    _, bad = A.build(rh, 1, "/root/rollhist", bank_sha=sha(1))                                  # seed-0 argv under seed 1: the bank path / out disagree
    assert bad


def test_render_is_idempotent_and_refuses_a_differing_existing_file(tmp_path, monkeypatch):
    R = tmp_path / "rh"
    for s in (0, 1):
        (R / "runs" / f"RH_s{s}").mkdir(parents=True)
        (R / "runs" / f"B0_s{s}").mkdir(parents=True)
        (R / "bank" / f"s{s}").mkdir(parents=True)
        open(R / "bank" / f"s{s}" / "bank.json", "wb").write(open(os.path.join(B, "pod_ro", f"bank_s{s}.json"), "rb").read())
        rh = [str(x).replace("/root/rollhist", str(R).replace("\\", "/")) for x in load(f"argv_RH_s{s}.json")]
        json.dump({"argv": rh}, open(R / "runs" / f"RH_s{s}" / "argv.json", "w"))
        json.dump({"argv": load(f"argv_B0_s{s}.json")}, open(R / "runs" / f"B0_s{s}" / "argv.json", "w"))
    monkeypatch.setattr(A, "R", str(R).replace("\\", "/"))
    assert A.main(["render"]) == 0                                    # the sha in the real argv equals the real bank.json: the same bank
    first = open(R / "runs" / "RHp3_s0" / "argv.json").read()
    assert A.main(["render"]) == 0 and open(R / "runs" / "RHp3_s0" / "argv.json").read() == first
    open(R / "runs" / "RHp3_s0" / "argv.json", "w").write(first.replace("0.3", "0.5"))
    assert A.main(["render"]) == 1                                    # never silently rewritten
