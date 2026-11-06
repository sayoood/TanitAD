"""Rule R-SCN (AMENDMENT_T2_SCENE_REFUSAL.md, frozen 2167a7eb): the exclusion builder and the re-pinned crawl read, each with RED arms. Synthetic data only."""
import copy
import hashlib
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
import read_stage2 as R          # noqa: E402
import scene_exclusions as X     # noqa: E402

DEC = [f"{tc}|{a}" for tc in ("T2", "RR") for a in ("C0_s0", "C0_s1", "P10", "ZERO", "BLIND")]
BES = [f"{tc}|{a}" for tc in ("T2", "RR") for a in ("DAGOACC_s0", "DAGOACC_s1")]
SC = {"aaaaaaaa": 0, "bbbbbbbb": 1, "cccccccc": 2, "dddddddd": 3}            # bbbbbbbb ACTOR_REFUSED (5 tracks), cccccccc NO_DYNAMIC_LAYER


def c(d, lo, hi):
    return {"delta": d, "lo": lo, "hi": hi, "separated": lo > 0 or hi < 0}


def refusal(n_tracks):
    return X.REFUSED + json.dumps({"n_tracks_json": n_tracks, "n_layer_tracks": 3, "n_mapped": 3, "falsifier_pass_frames": 0, "verdict": "REFUSED"})


KEYERR = ["Traceback (most recent call last):", '  File "actor_map.py", line 77, in layer_track_ranges',
          """KeyError: "no timestamps_us_ranges for layer 'dynamic_rigids'; probed: []" """.strip()]


def make(tmp_path, n_tracks_file=5, done_in_refused=False, extra_tb=False, other_reason=False):
    pull, src = tmp_path / "pull", []
    for s8, idx in SC.items():
        bad = s8 in ("bbbbbbbb", "cccccccc")
        for cell in DEC + BES:
            tc, arm = cell.split("|")
            d = pull / "rollouts" / s8 / "objects" / tc / arm
            d.mkdir(parents=True)
            if bad and not (done_in_refused and cell == "T2|P10"):
                (d / "FAIL").write_text("rc=1 records=0/2 executor=base\n")
            else:
                (d / "DONE").write_text("rc=0 records=2/2 executor=base\n")
        tr = tmp_path / f"{s8}_tracks.json"
        tr.write_text(json.dumps({"c0": {"tracks_data": {"tracks_id": [str(i) for i in range(n_tracks_file if s8 == "bbbbbbbb" else 9)]}}}))
        src.append({"idx": idx, "scene": s8 + "-x", "rel": "extracted/sequence_tracks.json", "source": str(tr), "sha256": hashlib.sha256(tr.read_bytes()).hexdigest()})
    failed = {p: [f"{s8}|{cell}" for s8 in ("bbbbbbbb", "cccccccc") for cell in cells] for p, cells in (("deciding", DEC), ("beside", BES))}
    logs = {}
    for p, cells in (("deciding", DEC), ("beside", BES)):
        lines = ["an earlier run", "Traceback (most recent call last):", "FileNotFoundError: old run"]          # lines 1-3 belong to an earlier run
        lines.append("Loading weights:   0%\r 50%\r100%")                                                     # a bare CR must not shift the line count
        for s8 in ("aaaaaaaa", "bbbbbbbb", "cccccccc", "dddddddd"):
            for _ in cells:
                if s8 == "bbbbbbbb":
                    lines.append(refusal(5))
                elif s8 == "cccccccc":
                    lines += (["Traceback (most recent call last):", "RuntimeError: boom"] if other_reason else KEYERR)
                else:
                    lines.append("12:00:00.000 INFO: ok")
        if extra_tb and p == "deciding":
            lines += ["Traceback (most recent call last):", "ValueError: stray"]
        f = tmp_path / f"rollouts_{p}.log"
        f.write_bytes(("\n".join(lines) + "\n").encode())
        logs[p] = (str(f), 4)
    result = {"stage": "stage2", "manifest_sha256": "m" * 64, "n_scenes_manifest": 4, "gates": {"preflight": "PASS", "gate": "PASS", "rollouts": "INCOMPLETE"},
              "overall": "INCOMPLETE",
              "steps": {"rollouts": {"state": {"deciding": {"done": 20, "skipped_done": 0, "failed": failed["deciding"]},
                                                "beside": {"done": 8, "skipped_done": 0, "failed": failed["beside"]}}, "passes_skipped_disk": []}}}
    return result, str(pull), logs, src


def pooled(on_idx=(0, 3)):
    g = {"pair_keys": [[i, f] for i in on_idx for f in (20, 40)]}
    return {"groups": {"C0_s0|T2|objects": g, "P10|T2|objects": copy.deepcopy(g)},
            "contrasts": {f"seed{s}": {"X1": c(0.06, 0.03, 0.09), "X2": c(0.0, -0.02, 0.02), "BL": c(0.0, -0.01, 0.01)} for s in (0, 1)}}


def test_builder_attributes_both_reasons(tmp_path):
    r = X.build(*make(tmp_path))
    assert r["verdict"] == "ATTRIBUTED", r["problems"]
    assert {k: v["reason"] for k, v in r["excluded"].items()} == {"bbbbbbbb": "ACTOR_REFUSED", "cccccccc": "NO_DYNAMIC_LAYER"}
    assert r["excluded"]["bbbbbbbb"]["tracks_file_count"] == 5 and r["logs"]["deciding"]["n_events"] == 20


@pytest.mark.parametrize("kw,needle", [({"extra_tb": True}, "terminal events"), ({"done_in_refused": True}, "not every cell is FAIL"),
                                       ({"n_tracks_file": 7}, "n_tracks_json 5 != 7"), ({"other_reason": True}, "reasons")])
def test_builder_red_arms(tmp_path, kw, needle):
    r = X.build(*make(tmp_path, **kw))
    assert r["verdict"] == "UNATTRIBUTED" and any(needle in p for p in r["problems"]), r["problems"]


def test_builder_sha_guard(tmp_path):
    res, pull, logs, src = make(tmp_path)
    src[1]["sha256"] = "0" * 64
    r = X.build(res, pull, logs, src)
    assert r["verdict"] == "UNATTRIBUTED" and any("sha256 differs" in p for p in r["problems"])


def test_read_with_exclusions_reads(tmp_path):
    res, pull, logs, src = make(tmp_path)
    exc = X.build(res, pull, logs, src)
    out = R.read(pooled(), res, exc)
    assert out["outcome"] != "NOT-READ", out.get("reason")
    assert out["deciding_required"] == 20 and out["excluded"] == {"bbbbbbbb": "ACTOR_REFUSED", "cccccccc": "NO_DYNAMIC_LAYER"}


def test_read_without_exclusions_keeps_first_pin(tmp_path):
    res, pull, logs, src = make(tmp_path)
    assert R.read(pooled(), res)["outcome"] == "NOT-READ"                        # failures and no rule given: the first pin's behaviour
    full = {"gates": {"preflight": "PASS", "gate": "PASS"}, "steps": {"rollouts": {"state": {"deciding": {"done": 420, "skipped_done": 0, "failed": []}}}}}
    assert R.read(pooled(), full)["outcome"] != "NOT-READ"


def _mut(tmp_path, f):
    res, pull, logs, src = make(tmp_path)
    exc = X.build(res, pull, logs, src)
    p = pooled()
    f(res, exc, p)
    return R.read(p, res, exc)


@pytest.mark.parametrize("name,f", [
    ("failed_outside", lambda res, exc, p: res["steps"]["rollouts"]["state"]["deciding"]["failed"].append("aaaaaaaa|T2|P10")),
    ("partial_scene", lambda res, exc, p: res["steps"]["rollouts"]["state"]["deciding"]["failed"].pop()),
    ("unattributed", lambda res, exc, p: exc.update(verdict="UNATTRIBUTED")),
    ("other_stage", lambda res, exc, p: exc.update(manifest_sha256="z" * 64)),
    ("pairs_on_excluded", lambda res, exc, p: p["groups"]["P10|T2|objects"]["pair_keys"].append([1, 20])),
    ("scene_missing", lambda res, exc, p: p["groups"].update({"C0_s0|T2|objects": {"pair_keys": [[0, 20]]}})),
    ("short_done", lambda res, exc, p: res["steps"]["rollouts"]["state"]["deciding"].update(done=19)),
])
def test_read_red_arms(tmp_path, name, f):
    assert _mut(tmp_path, f)["outcome"] == "NOT-READ", name


def test_read_names_the_problem(tmp_path):
    out = _mut(tmp_path, lambda res, exc, p: res["steps"]["rollouts"]["state"]["deciding"]["failed"].append("aaaaaaaa|T2|P10"))
    assert out["outcome"] == "NOT-READ" and "outside the excluded scenes" in out["reason"], out.get("reason")
