"""Literal tests for ``make_score_jobs.py``: seven jobs, labels and paths that ``select_ckpt_r8.spec_from_layout`` will find."""
from __future__ import annotations

import json
import os

import make_score_jobs as J
import select_ckpt_r8 as SEL

ROOT = "/home/nvidia/r8ns/navtrain_sel"
SAMPLE = "/home/nvidia/r8ns/navtrain/NAVTRAIN_SAMPLE_TOKENS.json"
DRIVER = "/home/nvidia/r8ns/code/navtrain/navtrain_cache_score.py"

EXPECTED = [
    ("r8sel_ema40000_s0", ["--seam", f"{ROOT}/ema40000/bridge/seam_R8_RT.npz"]),
    ("r8sel_ema40000_s1", ["--seam", f"{ROOT}/ema40000/bridge/seam_R8_RT_s1.npz"]),
    ("r8sel_ema48852_s0", ["--seam", f"{ROOT}/ema48852/bridge/seam_R8_RT.npz"]),
    ("r8sel_ema48852_s1", ["--seam", f"{ROOT}/ema48852/bridge/seam_R8_RT_s1.npz"]),
    ("r8sel_ref_PRIOR_ha0p", ["--seam", f"{ROOT}/ema48852/bridge/seam_PRIOR_ha0p.npz"]),
    ("r8sel_ref_STOP", ["--official", "STOP"]),
    ("r8sel_ref_CV", ["--official", "CV"]),
]


def test_seven_jobs_with_the_literal_labels_and_seams():
    jobs = J.make_jobs(ROOT, SAMPLE, DRIVER)
    assert [(j["name"], j["argv"][-2:]) for j in jobs] == [(lab, ex[-2:]) for lab, ex in EXPECTED]


def test_the_job_argv_is_the_score_command_of_the_driver():
    j = J.make_jobs(ROOT, SAMPLE, DRIVER)[0]
    assert j["argv"] == ["python3", DRIVER, "score", "--profile", "thor", "--sample", SAMPLE, "--out", ROOT, "--label", "r8sel_ema40000_s0",
                         "--seam", f"{ROOT}/ema40000/bridge/seam_R8_RT.npz"]
    assert j["cwd"] == ROOT and j["log"] == f"{ROOT}/pool_logs/r8sel_ema40000_s0.log"


def test_every_job_lands_where_the_decision_script_looks():
    spec = SEL.spec_from_layout(ROOT, SAMPLE)
    want = {a["csv"] for a in spec["arms"].values()} | {r["csv"] for r in spec["references"].values()}
    got = {f"{ROOT}/scores/{j['name']}/{j['name']}.csv" for j in J.make_jobs(ROOT, SAMPLE, DRIVER)}
    assert got == want and len(got) == 7
    seams = {a["seam"] for a in spec["arms"].values()} | {spec["references"]["PRIOR_ha0p"]["seam"]}
    assert {j["argv"][-1] for j in J.make_jobs(ROOT, SAMPLE, DRIVER) if "--seam" in j["argv"]} == seams


def test_missing_seams_refuse_and_write_nothing(tmp_path, capsys):
    out = tmp_path / "jobs.jsonl"
    rc = J.run(["--layout-root", str(tmp_path), "--sample", SAMPLE, "--driver", DRIVER, "--out", str(out)])
    assert rc == 2 and not out.exists()
    assert "ZZJOBS_REFUSED_missing_seams n=5" in capsys.readouterr().out


def test_with_every_seam_present_the_file_has_seven_json_lines(tmp_path, capsys):
    for ck in ("ema40000", "ema48852"):
        os.makedirs(tmp_path / ck / "bridge")
        for n in ("R8_RT", "R8_RT_s1", "PRIOR_ha0p"):
            (tmp_path / ck / "bridge" / f"seam_{n}.npz").write_bytes(b"x")
    out = tmp_path / "jobs.jsonl"
    rc = J.run(["--layout-root", str(tmp_path), "--sample", SAMPLE, "--driver", DRIVER, "--out", str(out)])
    assert rc == 0 and "ZZJOBS_DONE n=7" in capsys.readouterr().out
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert [j["name"] for j in lines] == [e[0] for e in EXPECTED]
