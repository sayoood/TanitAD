#!/usr/bin/env python3
"""Self-test of eval/s1_read.py (the S1 end-to-end read driver). Run (driverl-eval venv; no pytest):
    PYTHONIOENCODING=utf-8 C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe eval/test_s1_read.py     -> ZZTEST_PASS <n>

Style of eval/test_aux_s1_readout.py: expected values are DERIVED BY HAND from the fixtures below (never from the module under test); each RED mutation edits the module's SOURCE
(the snippet must occur exactly once: a vacuous mutation proves nothing) and the named check(s) must then FAIL. Real-data checks (the three banked N arms in the dev-box cache) run when
the cache is present and are SKIPPED, with the reason printed, when it is not.

FIXTURE (arms). An arm's evidence is rebuilt from its label: argv (class, seed, --out, --dump-scores), the token binding (argv_sha256 = sha256 of the compact JSON of the argv), a CUMULATIVE
arms_manifest.jsonl (the pod's chain appends one row per arm, so N-s1's file holds N-s0's row and its own), run_check, a PASS preflight (head arms: gate G12 with the raw fields), and the
trainer's sft.jsonl events. Hand-derived numbers: K6 bar = 0.9 x bce_declared; with bce_declared = 1.0 the bar is exactly 0.9. K3: s_per_update 10.0 (N) vs 10.4 (H) is a rise of 4 %.
FIXTURE (promotion). Three seeds; PDMS classification labels, P3 decisions and K-decisions are set directly, so each condition of S1 prereg 4.3 is hit alone.
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import math
import os
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(HERE, "s1_read.py")


def _eval_dir():
    for c in (os.environ.get("S1_EVAL_DIR"), HERE, os.path.join(os.path.dirname(os.path.dirname(HERE)), "eval"), os.path.join(os.path.dirname(HERE), "eval")):
        if c and os.path.isfile(os.path.join(c, "aux_s1_readout.py")):
            return os.path.abspath(c)
    raise SystemExit("cannot find eval/aux_s1_readout.py: set S1_EVAL_DIR")


EVAL = _eval_dir()
sys.path.insert(0, EVAL)
sys.path.insert(0, os.path.join(os.path.dirname(EVAL), "refe"))
import numpy as np  # noqa: E402
import aux_s1_readout as S  # noqa: E402
import rl_val_eval as RV  # noqa: E402

CACHE = os.environ.get("S1_CACHE", "D:/Projects/TanitAD-artifacts/s1_read_cache")


def load(src: str, name: str):
    m = types.ModuleType(name)
    m.__file__ = SRC_PATH
    sys.modules[name] = m
    exec(compile(src, SRC_PATH, "exec"), m.__dict__)
    return m


# ------------------------------------------------------------------------------------------------------------------ fixture: arms
def argv_for(cls, seed, arm, drop=(), add=()):
    a = ["--ckpt", "/w/u400.pt", "--targets", "/w/t", "--images", "/w/i", "--workers", "3", "--lr", "1e-4", "--max-updates", "400", "--eval-every", "400"]
    if cls != "N":
        a += ["--aux-drv", "--w-aux-drv", "0.1", "--aux-drv-lr", "2e-4", "--aux-drv-labels", "/w/ls"]
    if cls == "H2":
        a += ["--aux-drv-feed", "on"]
    a += ["--dump-scores", f"/w/runs/{arm}/dump", "--seed", str(seed), "--out", f"/w/runs/{arm}"]
    return [x for x in a if x not in drop] + list(add)


G12_OK = {"H1": {"ok": True, "reference_replay_identical": True, "scorer_params_bit_identical_2steps": True, "identity_probe_reads_one_ulp": True,
                 "score_to_head_grads_all_none": True, "bce_near_ln2": True, "bce_declared": 1.0, "head_retained_gb": 0.5},
          "H2": {"ok": True, "feed_zero_init": True, "feed_step0_equal_base": True, "feed_probe_reads_difference": True,
                 "score_to_head_grads_all_none": True, "bce_near_ln2": True, "bce_declared": 1.0, "head_retained_gb": 0.5}}


def make_ev(m, arm, label_as=None, manifest_arms=None, g12=None, sec_per_update=10.0, aux_rows=None, gpu=5.0):
    """The evidence of `arm`. `label_as` is only used by tests that PRETEND this evidence belongs under another label (the caller passes it to check_arm_evidence)."""
    cls, seed = m.parse_arm(arm)
    argv = argv_for(cls, seed, arm)
    tsha = "tok" + arm
    sha = m.argv_sha256(argv)
    rows = [{"arm": a, "event": "token_verified", "argv_sha256": sha if a == arm else "x", "token_sha256": tsha if a == arm else "y"} for a in (manifest_arms or [arm])]
    sft = [{"event": "data", "train_sets": 174912, "heldout_sets": 3137, "heldout_logs": 24, "train_rows_in_heldout_logs": 0},
           {"event": "start", "updates": 400}, {"event": "eval", "tag": "step0"},
           {"event": "score_dump", "tag": "step0", "files": {"B": f"/w/runs/{arm}/dump/scores_step0_B.npz"}}]
    for u in range(25, 401, 25):
        r = {"event": "train", "update": u, "s_per_update": sec_per_update}
        if cls != "N":
            r["aux_bce"] = 0.5
        sft.append(r)
    sft += [{"event": "eval", "tag": "update400"}, {"event": "score_dump", "tag": "update400", "files": {"B": f"/w/runs/{arm}/dump/scores_update400_B.npz"}}, {"event": "done", "updates": 400}]
    if cls != "N":
        sft.append({"event": "aux_eval", "tag": "update400"})
    pf = {"verdict": "PASS", "gpu_mem_gb": gpu, "gates": {}}
    if cls != "N":
        pf["gates"][m.G12] = dict(G12_OK[cls], **(g12 or {}))
    return {"manifest": rows, "run_check": {"arm": arm, "ok": True, "problems": []}, "preflight": pf, "sft": sft, "token_verify": {"verdict": "MATCH"},
            "token": {"verdict": "PASS", "binding": {"profile": "refe_aux_s1", "argv": argv, "argv_sha256": sha}}, "token_file": "PASS_ab.json", "token_sha256": tsha}


def codes(problems):
    return {p.split(":")[0] for p in problems}


# ------------------------------------------------------------------------------------------------------------------ checks: pins and gating
def chk_pins(m):
    assert (m.N_BOOT, m.SEED, m.K2_IOU_MIN, m.P3_FLOOR, m.P3_MARGIN, m.K6_FACTOR, m.K6_FRAC, m.K3_TIME, m.K3_GB, m.N_UPDATES, m.RULE) == \
        (10000, 20260927, 0.80, 0.90, 0.05, 0.9, 0.10, 0.03, 1.0, 400, "navsim_v1")
    assert m.PAIRS_NEEDED == 2 and m.PAIRS_NEEDED_PROVISIONAL == 3 and m.SEEDS == (0, 1, 2) and len(m.REGISTERED) == 9
    sel = m.cmd_selection("py", "/e", {"N-s0": "a.npz"}, "/pdm", "o.json")
    assert sel[sel.index("--n-boot") + 1] == "10000" and sel[sel.index("--rule") + 1] == "navsim_v1" and "_harness" in sel, sel
    nav = m.cmd_perception_navtest("py", "/e", "/pkg", {"H2-s0": "a"}, {"H2-s0": "b"}, "o")
    ho = m.cmd_perception_heldout("py", "/e", {"H2-s0": "a"}, {"H2-s0": "b"}, "/logs", "o")
    for c in (nav, ho):
        assert c[c.index("--n-boot") + 1] == "10000" and "--pred-init" in c, c
    assert nav[nav.index("--labels-set") + 1] == "navtest" and ho[ho.index("--labels-set") + 1] == "train"
    fwd = m.cmd_forward("py", "/e", {"H2-s0": "h.pt"}, "/o", "/ctx")
    assert fwd[fwd.index("--device") + 1] == "cpu" and "--no-gpu-lock" in fwd and fwd[fwd.index("--ctx-cache") + 1] == "/ctx"        # the floors are CPU passes: same device
    m.check_pins_against_harness(S)                                                                                     # the real harness agrees
    fake = types.SimpleNamespace(N_BOOT=1000, SEED=S.SEED, K2_IOU_MIN=S.K2_IOU_MIN, P3_AUC_FLOOR=S.P3_AUC_FLOOR, P3_AUC_MARGIN=S.P3_AUC_MARGIN, TOOL=S.TOOL)
    try:
        m.check_pins_against_harness(fake)
    except m.EvidenceRefused:
        return
    raise AssertionError("pin drift against the harness was not refused")


def chk_harness_json_pins(m):
    ok = {"tool": "aux_s1_readout/1", "estimator": "paired log-cluster bootstrap, 10000 resamples, 95 %, seed 20260927; x"}
    m.check_harness_json(ok, "t")
    for bad in (dict(ok, estimator=ok["estimator"].replace("10000", "1000")), dict(ok, estimator=ok["estimator"].replace("20260927", "1")), dict(ok, tool="other/1")):
        try:
            m.check_harness_json(bad, "t")
        except m.EvidenceRefused:
            continue
        raise AssertionError(f"accepted {bad}")


def chk_reads_table(m):
    present = ["N-s0", "N-s1", "N-s2", "H1-s0", "H1-s1", "H1-s2", "H2-s0", "H2-s1"]
    t = {r["id"]: r for r in m.reads_table(present)}
    assert t["PROMOTE_H2"]["arms_missing"] == ["H2-s2"] and t["PROMOTE_H1"]["arms_missing"] == [] and t["S1_OVERALL"]["arms_missing"] == ["H2-s2"]
    assert t["SEL_H2_s2"]["arms_needed"] == ["H2-s2", "N-s0", "N-s1", "N-s2"] and t["SEL_H2_s2"]["arms_missing"] == ["H2-s2"]
    assert t["K3_H2_s1"]["arms_needed"] == ["H2-s1", "N-s1"] and t["SEED_FLOOR"]["arms_missing"] == []
    t0 = {r["id"]: r for r in m.reads_table([])}
    assert all(r["status"] == "INCOMPLETE" for r in t0.values()) and t0["S1_OVERALL"]["arms_missing"] == list(m.REGISTERED)
    p = ["N-s0", "N-s1"]
    t1 = {r["id"]: r for r in m.reads_table(p)}
    assert t1["SEED_FLOOR"]["arms_missing"] == ["N-s2"] and t1["SEED_FLOOR"]["status"] == "INCOMPLETE"
    assert m.missing_arms(["N-s0"]) == [a for a in m.REGISTERED if a != "N-s0"]


LABELS = ("K2_FIRES", "K2_CLEARED", "H2_EVALUATED", "H2_PROVISIONAL", "H2_KILLED", "P3_AT_FLOOR", "H2_PROMOTED", "H2_NOT_PROMOTED", "H1_PROMOTED", "H1_NOT_PROMOTED", "K6_FIRES", "K3_FIRES")


def chk_incomplete_gate(m):
    full = list(m.REGISTERED)
    for present in (full[:-1], full[:3], [], [a for a in full if a != "N-s1"]):
        r = m.decide(present, {}, {}, None, None, None, {})
        assert r["status"] == "INCOMPLETE" and r["H2"] is None and r["H1"] is None, r["status"]
        assert r["missing_arms"] == [a for a in full if a not in present]
        text = json.dumps({k: v for k, v in r.items() if k != "reads"})                       # the reads' clause TEXT names labels as references; the outcome fields must not
        assert not any(lab in text for lab in LABELS), [lab for lab in LABELS if lab in text]
        assert all(set(x) == {"id", "clause", "arms_needed", "arms_missing", "status"} and x["status"] in ("INCOMPLETE", "READY") for x in r["reads"])


def chk_refused_gate(m):
    full = list(m.REGISTERED)
    r = m.decide([a for a in full if a != "N-s1"], {"N-s1": ["ID_ARGV_SEED: x"]}, {}, None, None, None, {})
    assert r["status"] == "REFUSED" and r["refused_arms"] == {"N-s1": ["ID_ARGV_SEED: x"]} and "H2" not in r
    assert not any(lab in json.dumps({k: v for k, v in r.items() if k != "reads"}) for lab in LABELS)


# ------------------------------------------------------------------------------------------------------------------ checks: arm identity
def chk_evidence_control(m):
    for arm in m.REGISTERED:
        P, W = m.check_arm_evidence(arm, make_ev(m, arm, manifest_arms=["N-s0", arm] if arm != "N-s0" else None), {})
        assert P == [], (arm, P)


def chk_swap_manifest(m):
    ev = make_ev(m, "N-s1", manifest_arms=["N-s0", "N-s1"])
    c = codes(m.check_arm_evidence("N-s2", ev, {})[0])                                       # N-s1's evidence filed under N-s2
    assert "ID_MANIFEST_NO_ROW" in c and "ID_RUNCHECK" in c, c
    ev2 = make_ev(m, "N-s0", manifest_arms=["N-s0", "N-s1"])                                  # last manifest row names a DIFFERENT arm
    assert "ID_MANIFEST_LAST_ROW" in codes(m.check_arm_evidence("N-s0", ev2, {})[0])


def chk_swap_argv(m):
    ev = make_ev(m, "N-s1")
    c = codes(m.check_arm_evidence("N-s2", ev, {})[0])
    assert {"ID_ARGV_SEED", "ID_ARGV_OUT", "ID_ARGV_DUMP"} <= c, c
    assert "ID_ARGV_CLASS" in codes(m.check_arm_evidence("H2-s0", make_ev(m, "H1-s0"), {})[0])        # an H1 run filed as H2
    assert "ID_ARGV_CLASS" in codes(m.check_arm_evidence("N-s0", make_ev(m, "H1-s0"), {})[0])         # a head run filed as N
    assert "ID_ARGV_CLASS" in codes(m.check_arm_evidence("H1-s0", make_ev(m, "N-s0"), {})[0])         # an N run filed as H1
    ev3 = make_ev(m, "H2-s1")
    ev3["token"]["binding"]["argv"] = argv_for("H2", 1, "H2-s1", add=["--lr", "5e-4"])
    assert "ID_ARGV_SHA" in codes(m.check_arm_evidence("H2-s1", ev3, {})[0])                            # an argv edited after the sha was bound


def chk_registered_values(m):
    ev = make_ev(m, "H2-s0")
    ev["token"]["binding"]["argv"] = argv_for("H2", 0, "H2-s0", add=["--max-updates", "800"])
    ev["token"]["binding"]["argv_sha256"] = m.argv_sha256(ev["token"]["binding"]["argv"])
    ev["manifest"][0]["argv_sha256"] = ev["token"]["binding"]["argv_sha256"]
    assert "ID_ARGV_UPDATES" in codes(m.check_arm_evidence("H2-s0", ev, {})[0])
    ev = make_ev(m, "H1-s0")
    ev["sft"] = [r for r in ev["sft"] if r.get("event") != "done"]
    assert "ID_SFT_DONE" in codes(m.check_arm_evidence("H1-s0", ev, {})[0])
    ev = make_ev(m, "N-s0")
    ev["sft"][0]["heldout_logs"] = 23
    assert "ID_SFT_DATA" in codes(m.check_arm_evidence("N-s0", ev, {})[0])
    ev = make_ev(m, "N-s0")
    ev["sft"].append({"event": "aux_eval", "tag": "x"})
    assert "ID_N_HAS_AUX_ROWS" in codes(m.check_arm_evidence("N-s0", ev, {})[0])
    ev = make_ev(m, "H2-s0")
    ev["sft"] = [r for r in ev["sft"] if r.get("event") != "aux_eval" and "aux_bce" not in r]
    assert "ID_HEAD_NO_AUX_ROWS" in codes(m.check_arm_evidence("H2-s0", ev, {})[0])
    ev = make_ev(m, "N-s0")
    assert "ID_N_HAS_HEAD_FILES" in codes(m.check_arm_evidence("N-s0", ev, {"arms/N-s0/aux_drv.pt": {}})[0])


def chk_k5_problems(m):
    for cls, field in (("H2", "feed_probe_reads_difference"), ("H1", "reference_replay_identical"), ("H1", "scorer_params_bit_identical_2steps"), ("H2", "feed_zero_init")):
        arm = f"{cls}-s0"
        P, _ = m.check_arm_evidence(arm, make_ev(m, arm, g12={field: False}), {})
        assert "K5_FIELD_NOT_TRUE" in codes(P) and any(field in p for p in P), (arm, field, P)
    P, _ = m.check_arm_evidence("H1-s0", make_ev(m, "H1-s0", g12={"ok": False}), {})
    assert "K5_G12_NOT_OK" in codes(P)
    ev = make_ev(m, "H2-s0")
    del ev["preflight"]["gates"][m.G12]
    assert "K5_G12_MISSING" in codes(m.check_arm_evidence("H2-s0", ev, {})[0])
    ev = make_ev(m, "N-s0")
    ev["preflight"]["gates"][m.G12] = {"ok": True}
    assert "ID_N_HAS_G12" in codes(m.check_arm_evidence("N-s0", ev, {})[0])


def chk_cross_arm(m):
    evs = {a: make_ev(m, a) for a in ("N-s0", "N-s1", "H2-s0", "H1-s0")}
    shas = {"N-s0": "a", "N-s1": "b", "H2-s0": "c", "H1-s0": "a"}                    # H1-s0 bit-identical to N-s0 is EXPECTED (no path from the head to the scorer)
    P, W = m.check_cross_arm(evs, shas, {})
    assert P == {} and W == [], (P, W)
    P, _ = m.check_cross_arm(evs, {**shas, "N-s1": "a"}, {})
    assert "DUP_SCORES" in codes(P["N-s1"]), P
    P, _ = m.check_cross_arm(evs, {**shas, "H2-s0": "a"}, {})
    assert "H2_SCORES_EQUAL_N" in codes(P["H2-s0"]), P
    evs2 = dict(evs)
    evs2["H2-s0"] = make_ev(m, "H2-s0")
    evs2["H2-s0"]["token"]["binding"]["argv"] = argv_for("H2", 0, "H2-s0", add=["--lr", "9e-4"])
    P, _ = m.check_cross_arm(evs2, shas, {})
    assert any("PARITY_ARGV_CORE" in p for ps in P.values() for p in ps), P
    _, W = m.check_cross_arm(evs, shas, {"arms/N-s0/dump/scores_step0_B.npz": {"sha256": "p"}, "arms/N-s1/dump/scores_step0_B.npz": {"sha256": "q"}})
    assert W and "STEP0_NOT_COMMON" in W[0]


# ------------------------------------------------------------------------------------------------------------------ checks: fetch
def chk_fetch(m):
    with tempfile.TemporaryDirectory() as d:
        repo, cache = os.path.join(d, "repo"), os.path.join(d, "cache")
        os.makedirs(os.path.join(repo, "arms", "N-s0"))
        small = os.path.join(repo, "arms", "N-s0", "run_check.json")
        big = os.path.join(repo, "arms", "N-s0", "scores.npz")
        open(small, "wb").write(b'{"ok": true}\n')
        open(big, "wb").write(os.urandom(6000))
        rem = m.LocalRemote(repo)
        lst = rem.listing()
        r = m.fetch_verified(rem, lst, "arms/N-s0/scores.npz", cache)
        assert r["cached"] is False and r["verified_against"] == "lfs_sha256" and r["sha256"] == m.sha256_file(big) and r["size"] == 6000
        r = m.fetch_verified(rem, lst, "arms/N-s0/run_check.json", cache)
        assert r["verified_against"] == "git_blob_sha1" and r["size"] == 13
        r2 = m.fetch_verified(rem, lst, "arms/N-s0/scores.npz", cache)
        assert r2["cached"] is True
        # a corrupt CACHED copy is re-downloaded, not trusted
        cp = os.path.join(cache, "arms", "N-s0", "scores.npz")
        raw = bytearray(open(cp, "rb").read())
        raw[10] ^= 0xFF
        open(cp, "wb").write(bytes(raw))
        r3 = m.fetch_verified(rem, lst, "arms/N-s0/scores.npz", cache)
        assert r3["cached"] is False and m.sha256_file(cp) == m.sha256_file(big)
        # a listing whose sha256 is wrong: refused and nothing left behind
        for rel, lie, why in (("arms/N-s0/scores.npz", {"sha256": "0" * 64}, "sha256"), ("arms/N-s0/scores.npz", {"size": 5999}, "size"), ("arms/N-s0/run_check.json", {"blob_id": "1" * 40}, "blob")):
            shutil_rm(os.path.join(cache, *rel.split("/")))
            lied = m.LocalRemote(repo, lies={rel: lie}).listing()
            try:
                m.fetch_verified(rem, lied, rel, cache)
            except m.FetchRefused:
                assert not os.path.exists(os.path.join(cache, *rel.split("/"))) and not os.path.exists(os.path.join(cache, *rel.split("/")) + ".part"), why
                continue
            raise AssertionError(f"a {why} mismatch was accepted")
        try:
            m.fetch_verified(rem, {"arms/N-s0/run_check.json": {"size": 13, "sha256": None, "blob_id": None}}, "arms/N-s0/run_check.json", cache + "2")
        except m.FetchRefused:
            pass
        else:
            raise AssertionError("a listing row with nothing to verify against was accepted")
        try:
            m.fetch_verified(rem, lst, "arms/N-s0/absent.json", cache)
        except m.FetchRefused:
            pass
        else:
            raise AssertionError("a path absent from the listing was fetched")


def shutil_rm(p):
    if os.path.exists(p):
        os.remove(p)


def chk_arm_paths(m):
    lst = {f"arms/H2-s0/{f}": {} for f in m.ARM_SMALL + m.ARM_DUMP + m.ARM_HEAD}
    lst["arms/H2-s0/PASS_ab12.json"] = {}
    want, absent = m.arm_paths("H2-s0", lst)
    assert absent == [] and "arms/H2-s0/aux_drv.pt" in want and "arms/H2-s0/PASS_ab12.json" in want and not any("model_sft" in w or "scorer_states" in w for w in want)
    del lst["arms/H2-s0/aux_drv.pt"]
    del lst["arms/H2-s0/run_check.json"]
    _, absent = m.arm_paths("H2-s0", lst)
    assert sorted(absent) == ["arms/H2-s0/aux_drv.pt", "arms/H2-s0/run_check.json"]                      # a PARTIAL push is not an arm
    nl = {f"arms/N-s0/{f}": {} for f in m.ARM_SMALL + m.ARM_DUMP}
    nl["arms/N-s0/PASS_ab12.json"] = nl["arms/N-s0/PASS_cd34.json"] = {}
    w, absent = m.arm_paths("N-s0", nl)
    assert "arms/N-s0/aux_drv.pt" not in w and absent and "PASS_<id>.json (found 2)" in absent[0]
    assert m.discover_arms({"arms/H2-s0/run_check.json": {}, "arms/N-s2/run_check.json": {}, "arms/N-s1/x": {}}) == ["N-s2", "H2-s0"]


# ------------------------------------------------------------------------------------------------------------------ checks: decision rules
def chk_p3_floor(m):
    nan = float("nan")
    cases = [((0.95, 0.80, 0.02), "H2_EVALUATED"), ((0.95, 0.80, 0.0), "P3_AT_FLOOR"), ((0.95, 0.80, -0.01), "P3_AT_FLOOR"), ((0.95, 0.80, nan), "P3_AT_FLOOR"),
             ((0.95, 0.80, None), "FLOOR_UNREAD"), ((0.85, 0.80, 0.05), "H2_PROVISIONAL"), ((0.85, 0.80, -0.1), "P3_AT_FLOOR"), ((0.85, 0.80, None), "FLOOR_UNREAD"),
             ((0.70, 0.80, 0.3), "H2_KILLED"), ((0.70, 0.80, -0.1), "H2_KILLED"), ((0.70, 0.80, None), "H2_KILLED"), ((0.80, 0.80, 0.1), "H2_PROVISIONAL"),
             ((0.91, 0.87, 0.1), "H2_PROVISIONAL"), ((0.92, 0.87, 0.1), "H2_EVALUATED"), ((0.90, 0.80, 0.1), "H2_EVALUATED"), ((0.8999, 0.80, 0.1), "H2_PROVISIONAL")]
    for args, want in cases:
        got = m.p3_decide(*args)
        assert got == want, (args, got, want)


def chk_p3_matches_harness(m):
    nan = float("nan")
    for h in (0.60, 0.79, 0.80, 0.84, 0.899, 0.90, 0.93, 0.97):
        for s in (0.70, 0.80, 0.87, 0.95):
            for lo in (None, nan, -0.2, 0.0, 1e-9, 0.2):
                assert m.p3_decide(h, s, lo) == S.p3_decision(h, s, lo)["decision"], (h, s, lo)


def chk_classify_matches_harness(m):
    fl = {"floor": {"PDMS": 0.35, "DAC": 0.25}, "measured": True, "n_n_seeds": 3}
    capped = {"floor": {"PDMS": 0.35}, "measured": False, "n_n_seeds": 2}
    for ci in ([-3.0, -1.0], [-1.0, 0.5], [0.1, 0.9], [0.4, 2.0], [0.36, 1.0], [0.35, 1.0], [0.0, 1.0], [-1.0, 0.0]):
        for f in (fl, capped, None):
            for q in ("PDMS", "DAC"):
                r = {"ci95": ci}
                assert m.classify(r, f, q) == S.classify(r, f, q), (ci, f, q)
    assert m.classify({"ci95": [0.36, 1.0]}, fl, "PDMS") == "separated and above the seed floor"
    assert m.classify({"ci95": [0.35, 1.0]}, fl, "PDMS") == "separated but within the seed floor"
    assert m.classify({"ci95": [0.4, 2.0]}, capped, "PDMS") == "separated, single seed, not yet a lever claim"


def chk_k2(m):
    def arm(iou, label=None):
        return {"p1": {"all": {"0-40": {"est": iou}}}, "k2": {"decision": label or ("K2_FIRES" if iou < 0.80 else "K2_CLEARED")}}
    assert m.k2_from(arm(0.79), None)["decision"] == "K2_FIRES"
    assert m.k2_from(arm(0.80), None)["decision"] == "K2_CLEARED"                                    # the bar is a strict <
    assert m.k2_from(arm(0.95), None)["decision"] == "K2_CLEARED"
    r = m.k2_from(arm(0.85), arm(0.70))                                                              # they disagree: the NAVTEST P1 governs
    assert r["decision"] == "K2_CLEARED" and r["navtest_and_ho_agree"] is False and r["source"].startswith("navtest")
    r = m.k2_from(None, arm(0.70))
    assert r["decision"] == "K2_FIRES" and "P1-HO" in r["source"]
    assert m.k2_from(None, None)["decision"] == "K2_UNREAD"
    assert m.k2_from(arm(float("nan"), "K2_UNDEFINED"), None)["decision"] == "K2_UNDEFINED"
    try:
        m.k2_from(arm(0.85, "K2_FIRES"), None)
    except m.EvidenceRefused:
        return
    raise AssertionError("a harness K2 label that disagrees with the re-derived one was accepted")


def chk_k6(m):
    pf = {"gates": {m.G12: {"bce_declared": 1.0}}}

    def rows(r25, r50):
        return [{"event": "train", "update": 25, "aux_bce": r25}, {"event": "train", "update": 50, "aux_bce": r50}, {"event": "train", "update": 75, "aux_bce": 0.1}]
    assert m.read_k6(rows(0.95, 0.5), pf)["decision"] == "K6_CLEARED"                                # update 25 is before 10 % of 400 steps: not the row
    assert m.read_k6(rows(0.2, 0.9), pf)["decision"] == "K6_FIRES"                                   # 0.9 >= 0.9 x 1.0: the bar is >=
    assert m.read_k6(rows(0.2, 0.89), pf)["decision"] == "K6_CLEARED"
    assert m.read_k6(rows(0.2, 0.95), pf)["decision"] == "K6_FIRES"
    r = m.read_k6(rows(0.2, 0.89), pf)
    assert r["row_update"] == 50 and abs(r["bar"] - 0.9) < 1e-12 and r["sensitive_to_ambiguity"] is True                  # vs ln 2 x 0.9 = 0.624: 0.89 fires there
    assert m.read_k6([{"event": "train", "update": 25, "aux_bce": 0.5}], pf)["decision"] == "K6_UNREAD"
    assert m.read_k6(rows(0.2, 0.9), {"gates": {}})["decision"] == "K6_UNREAD"


def chk_k3(m):
    def pf(gb, ret):
        return {"gpu_mem_gb": gb, "gates": {m.G12: {"head_retained_gb": ret}}}

    def rows(s):
        return [{"event": "train", "update": 25, "s_per_update": 99.0}, {"event": "train", "update": 400, "s_per_update": s}]
    n = (pf(5.0, None), rows(10.0))
    assert m.read_k3(pf(5.4, 0.5), n[0], rows(10.2), n[1])["decision"] == "K3_CLEARED"
    r = m.read_k3(pf(5.4, 1.2), n[0], rows(10.2), n[1])
    assert r["decision"] == "K3_FIRES" and r["limb_head_retained_gt_1gb"] and not r["limb_time_gt_3pct"] and not r["limb_gpu_delta_gt_1gb"]
    r = m.read_k3(pf(6.5, 0.5), n[0], rows(10.2), n[1])
    assert r["decision"] == "K3_FIRES" and r["limb_gpu_delta_gt_1gb"] and abs(r["gpu_mem_gb_delta_vs_N"] - 1.5) < 1e-9
    r = m.read_k3(pf(5.4, 0.5), n[0], rows(10.4), n[1])
    assert r["decision"] == "K3_FIRES" and r["limb_time_gt_3pct"] and abs(r["step_time_rise_vs_N"] - 0.04) < 1e-9 and r["update_rows"] == [400, 400]
    assert m.read_k3({"gates": {}}, {}, [], [])["decision"] == "K3_UNREAD"


E, PR, KL = "H2_EVALUATED", "H2_PROVISIONAL", "H2_KILLED"
ABOVE, WITHIN, NOTSEP, ADV = "separated and above the seed floor", "separated but within the seed floor", "not_separated", "adverse"


def h2_inputs(p3=(E, E, E), pdms=(ABOVE, ABOVE, ABOVE), dac=(NOTSEP, NOTSEP, NOTSEP), k2=("K2_CLEARED",) * 3, k3=("K3_CLEARED",) * 3, k6=("K6_CLEARED",) * 3, floor=True):
    return dict(p3=dict(enumerate(p3)), cls={s: {"PDMS": pdms[s], "DAC": dac[s]} for s in range(3)}, k2=dict(enumerate(k2)), k3=dict(enumerate(k3)), k6=dict(enumerate(k6)), floor_measured=floor)


def chk_promote_h2(m):
    P = m.promote_h2
    r = P(**h2_inputs())
    assert r["outcome"] == "H2_PROMOTED_TO_S2" and r["regime"] == "H2_EVALUATED" and r["conditions"]["2_pdms_pairs"]["needed"] == 2
    assert "capacity not excluded" in r["stamps"][0] and "never an ADOPT" in r["stamps"][1] and "C-shuffle" in r["if_promoted"]
    assert P(**h2_inputs(pdms=(ABOVE, ABOVE, WITHIN)))["outcome"] == "H2_PROMOTED_TO_S2"             # 2 of 3 suffices under EVALUATED
    assert P(**h2_inputs(pdms=(ABOVE, WITHIN, NOTSEP)))["outcome"] == "H2_NOT_PROMOTED"
    r = P(**h2_inputs(p3=(E, PR, E), pdms=(ABOVE, ABOVE, WITHIN)))                                     # PROVISIONAL: 3 of 3 needed
    assert r["outcome"] == "H2_NOT_PROMOTED" and r["regime"] == "H2_PROVISIONAL" and r["conditions"]["2_pdms_pairs"]["needed"] == 3 and not r["conditions"]["2_pdms_pairs"]["ok"]
    assert P(**h2_inputs(p3=(E, PR, E)))["outcome"] == "H2_PROMOTED_TO_S2"
    assert P(**h2_inputs(dac=(NOTSEP, ADV, NOTSEP)))["outcome"] == "H2_NOT_PROMOTED"                  # one adverse DAC kills the promotion
    assert P(**h2_inputs(pdms=(ABOVE, ABOVE, ADV)))["outcome"] == "H2_NOT_PROMOTED"
    assert P(**h2_inputs(k2=("K2_CLEARED", "K2_FIRES", "K2_CLEARED")))["outcome"] == "H2_NOT_PROMOTED"
    assert P(**h2_inputs(k2=("K2_CLEARED", "K2_UNREAD", "K2_CLEARED")))["outcome"] == "H2_NOT_PROMOTED"
    assert P(**h2_inputs(k3=("K3_CLEARED", "K3_CLEARED", "K3_FIRES")))["outcome"] == "H2_NOT_PROMOTED"
    assert P(**h2_inputs(k6=("K6_FIRES", "K6_CLEARED", "K6_CLEARED")))["outcome"] == "H2_NOT_PROMOTED"
    for bad in ("P3_AT_FLOOR", KL, "P3_NOT_READ", "FLOOR_UNREAD"):                                     # one seed that is not EVALUATED / PROVISIONAL blocks (AMB-1: all three)
        r = P(**h2_inputs(p3=(E, bad, E)))
        assert r["outcome"] == "H2_NOT_PROMOTED" and not r["conditions"]["1_p3"]["ok"], bad
    assert P(**h2_inputs(p3=(E, "P3_AT_FLOOR", E)), c1_mode="majority")["outcome"] == "H2_PROMOTED_TO_S2"        # the AMB-1 alternative differs
    assert P(**h2_inputs(floor=False))["outcome"] == "H2_NOT_PROMOTED"                                # fewer than three N runs: capped, never a promotion
    assert P(**h2_inputs(pdms=("separated, single seed, not yet a lever claim",) * 3))["outcome"] == "H2_NOT_PROMOTED"


def chk_promote_h1(m):
    ok = {0: "K2_CLEARED", 1: "K2_CLEARED", 2: "K2_CLEARED"}
    k6 = {0: "K6_CLEARED", 1: "K6_CLEARED", 2: "K6_CLEARED"}
    assert m.promote_h1(ok, k6, False)["outcome"] == "H1_PROMOTED_TO_S2"
    assert m.promote_h1({**ok, 2: "K2_FIRES"}, k6, False)["outcome"] == "H1_NOT_PROMOTED"
    assert m.promote_h1(ok, {**k6, 1: "K6_FIRES"}, False)["outcome"] == "H1_NOT_PROMOTED"
    assert m.promote_h1({**ok, 1: "K2_UNREAD"}, k6, False)["outcome"] == "H1_NOT_PROMOTED"
    r = m.promote_h1(ok, k6, True)
    assert r["outcome"] == "H1_IDENTITY_DEFECT" and "PROMOTED" not in json.dumps(r)
    assert m.promote_h1({**ok, 2: "K2_FIRES"}, k6, False, seeds=[0])["outcome"] == "H1_PROMOTED_TO_S2"        # the AMB-2 alternative differs


def chk_h1_identity(m):
    z = {"mean": 0.0, "ci95": [0.0, 0.0]}
    zero = {"paired_vs_n": {"quantities": {q: dict(z) for q in ("NC", "DAC", "EP", "TTC", "C", "DDC", "PDMS", "R")}}}
    r = m.read_h1_identity(zero, "a", "a", None)
    assert r["defect"] is False and r["selection_identical"] and r["warning"] is None
    r = m.read_h1_identity(zero, "a", "b", True)
    assert r["defect"] is False and r["warning"] is None
    r = m.read_h1_identity(zero, "a", "b", False)
    assert r["defect"] is False and "bitwise" in r["warning"]                                          # scorer not bit-identical but selection identical: a warning (AMB-7)
    bad = copy.deepcopy(zero)
    bad["paired_vs_n"]["quantities"]["EP"] = {"mean": 0.01, "ci95": [0.0, 0.02]}
    assert m.read_h1_identity(bad, "a", "b", False)["defect"] is True
    bad2 = copy.deepcopy(zero)
    bad2["paired_vs_n"]["quantities"]["PDMS"]["ci95"] = [-0.001, 0.0]
    assert m.read_h1_identity(bad2, "a", "a", None)["defect"] is True


# ------------------------------------------------------------------------------------------------------------------ checks: a COMPLETE synthetic read through decide()
QS = ("NC", "DAC", "EP", "TTC", "C", "DDC", "PDMS", "R")


def synth(m, h2_pdms_lo=(0.6, 0.6, 0.6), h2_p3=((0.95, 0.80, 0.05),) * 3, h1_zero=True, iou=0.85, p3_missing=False, est=None):
    floor = {"floor": {"PDMS": 0.35, "DAC": 0.25, "EP": 0.5, "NC": 0.2, "TTC": 0.2}, "measured": True, "n_n_seeds": 3}

    def pair(lo, hi):
        qs = {q: {"mean": (lo + hi) / 2, "ci95": [lo, hi], "better": 1, "worse": 0, "tied": 0} for q in QS}
        qs["DAC"] = {"mean": 0.0, "ci95": [-0.5, 0.5], "better": 0, "worse": 0, "tied": 1}
        return qs
    pairs = {}
    for s in range(3):
        z = {q: {"mean": 0.0, "ci95": [0.0, 0.0], "better": 0, "worse": 0, "tied": 9} for q in QS}
        for c in ("H1", "H2"):
            qs = z if (c == "H1" and h1_zero) else pair(h2_pdms_lo[s], h2_pdms_lo[s] + 1.0)
            pairs[f"{c}-s{s}-minus-N-s{s}"] = {"n_sets": 3137, "n_logs": 24, "paired_vs_n": {"quantities": qs, "primary": qs["PDMS"]},
                                              "classification": {q: m.classify(qs[q], floor, q) for q in ("PDMS", "DAC")},
                                              "arms": {"head": {"PDMS_x100": 90.0, "fan_random_x100": 75.0, "fan_best_x100": 98.0, "skill": 0.65},
                                                       "n": {"PDMS_x100": 90.0, "fan_random_x100": 75.0, "fan_best_x100": 98.0, "skill": 0.65}},
                                              "selfcheck": {"selfpair": {"ok": True}, "swap": {"ok": True}}}
    est = est or "paired log-cluster bootstrap, 10000 resamples, 95 %, seed 20260927; x"
    sel = {"tool": "aux_s1_readout/1", "estimator": est, "seed_floor": floor, "pairs": pairs, "unpaired_head_arms": []}
    nav_arms = {}
    for s in range(3):
        for c in ("H1", "H2"):
            ah, asc, lo = h2_p3[s]
            lab = m.p3_decide(ah, asc, lo)
            nav_arms[f"{c}-s{s}"] = {"p1": {"all": {"0-40": {"est": iou}}}, "k2": {"decision": "K2_CLEARED" if iou >= 0.8 else "K2_FIRES"},
                                     "p3": {"decision": {"decision": lab, "auc_head": ah, "auc_scorer": asc, "evaluate_threshold": max(0.9, asc + 0.05), "floor_lower_bound": lo}}}
            if p3_missing:
                nav_arms[f"{c}-s{s}"]["p3"] = {"status": "NOT_COMPUTED"}
    nav = {"tool": "aux_s1_readout/1", "estimator": est, "arms": nav_arms}
    ho = {"tool": "aux_s1_readout/1", "estimator": est, "arms": {a: {"p1": {"all": {"0-40": {"est": iou}}}, "k2": {"decision": "K2_CLEARED"}} for a in nav_arms}}
    evid = {a: {"preflight": make_ev(m, a)["preflight"], "sft": make_ev(m, a)["sft"]} for a in m.REGISTERED}
    h1id = {s: {"sha_h1": "a", "sha_n": "a", "arrays_equal": None} for s in range(3)}
    return sel, nav, ho, evid, h1id


def chk_decide_complete(m):
    sel, nav, ho, evid, h1id = synth(m)
    r = m.decide(list(m.REGISTERED), {}, evid, sel, nav, ho, h1id)
    assert r["status"] == "READ" and r["H2"]["outcome"] == "H2_PROMOTED_TO_S2" and r["H1"]["outcome"] == "H1_PROMOTED_TO_S2", (r["status"], r["H2"], r["H1"])
    assert r["H2"]["regime"] == "H2_EVALUATED" and r["H2"]["conditions"]["2_pdms_pairs"]["needed"] == 2 and r["H2"]["conditions"]["2_pdms_pairs"]["n_above_floor"] == 3
    assert all(x["status"] == "READ" for x in r["reads"]) and r["seed_floor"]["measured"] and r["families"]["STRATEGIC"]["status"] == "PAUSED"
    assert r["per_arm"]["p3"]["H1-s0"]["decision"] == "NOT_DECIDED_FOR_H1" and r["per_arm"]["p3"]["H1-s0"]["reported_label"] == "H2_EVALUATED"
    assert r["per_arm"]["p3"]["H2-s0"]["decision"] == "H2_EVALUATED" and r["per_arm"]["k2"]["H2-s0"]["source"].startswith("navtest")
    # the PROVISIONAL band (head 0.85 between the scorer 0.80 and 0.90) with a positive floor: PROMOTED only with 3 of 3 pairs above the floor
    selp, navp, hop, evp, h1p = synth(m, h2_p3=((0.85, 0.80, 0.05),) * 3)
    rp = m.decide(list(m.REGISTERED), {}, evp, selp, navp, hop, h1p)
    assert rp["H2"]["regime"] == "H2_PROVISIONAL" and rp["H2"]["conditions"]["2_pdms_pairs"]["needed"] == 3 and rp["H2"]["outcome"] == "H2_PROMOTED_TO_S2"
    selq, navq, hoq, evq, h1q = synth(m, h2_p3=((0.85, 0.80, 0.05),) * 3, h2_pdms_lo=(0.6, 0.6, 0.2))
    assert m.decide(list(m.REGISTERED), {}, evq, selq, navq, hoq, h1q)["H2"]["outcome"] == "H2_NOT_PROMOTED"
    # a head that ends AT its floor (lower bound <= 0) is P3_AT_FLOOR on every seed: not promoted, whatever the selection says
    self_, navf, hof, evf, h1f = synth(m, h2_p3=((0.95, 0.80, -0.02),) * 3)
    rf = m.decide(list(m.REGISTERED), {}, evf, self_, navf, hof, h1f)
    assert rf["per_arm"]["p3"]["H2-s1"]["decision"] == "P3_AT_FLOOR" and rf["H2"]["outcome"] == "H2_NOT_PROMOTED"
    assert r["H2"]["sensitivity_AMB-1"]["differs"] is False and r["H1"]["sensitivity_AMB-2"]["differs"] is False
    # PDMS lower bound 0.6 vs floor 0.35 => above; with lower bounds 0.2 (within the floor) on 2 seeds there is no promotion
    sel2, nav2, ho2, evid2, h1id2 = synth(m, h2_pdms_lo=(0.2, 0.2, 0.6))
    r2 = m.decide(list(m.REGISTERED), {}, evid2, sel2, nav2, ho2, h1id2)
    assert r2["H2"]["outcome"] == "H2_NOT_PROMOTED" and r2["classification"]["H2-s0"]["PDMS"] == WITHIN
    # an H1 arm that is not selection-identical to its N is a DEFECT, reported instead of a promotion
    sel3, nav3, ho3, evid3, h1id3 = synth(m, h1_zero=False)
    r3 = m.decide(list(m.REGISTERED), {}, evid3, sel3, nav3, ho3, h1id3)
    assert r3["H1"]["outcome"] == "H1_IDENTITY_DEFECT" and r3["H2"]["h1_identity_defect_open"] is True
    # P3 unread (no navtest forward): the perception gate is unread, H2 is not promoted, K2 falls back to P1-HO
    sel4, nav4, ho4, evid4, h1id4 = synth(m, p3_missing=True)
    r4 = m.decide(list(m.REGISTERED), {}, evid4, sel4, nav4, ho4, h1id4)
    assert r4["per_arm"]["p3"]["H2-s0"]["decision"] == "P3_NOT_READ" and r4["H2"]["outcome"] == "H2_NOT_PROMOTED"
    r5 = m.decide(list(m.REGISTERED), {}, evid4, sel4, None, ho4, h1id4)
    assert r5["per_arm"]["k2"]["H2-s0"]["source"].startswith("P1-HO") and r5["H2"]["outcome"] == "H2_NOT_PROMOTED"


def chk_decide_rederivation(m):
    sel, nav, ho, evid, h1id = synth(m)
    bad = copy.deepcopy(nav)
    bad["arms"]["H2-s1"]["p3"]["decision"]["decision"] = "H2_EVALUATED" if nav["arms"]["H2-s1"]["p3"]["decision"]["decision"] != "H2_EVALUATED" else "P3_AT_FLOOR"
    try:
        m.decide(list(m.REGISTERED), {}, evid, sel, bad, ho, h1id)
    except m.EvidenceRefused as e:
        assert "P3 decision" in str(e)
    else:
        raise AssertionError("a P3 label that disagrees with the re-derived decision was accepted")
    bad = copy.deepcopy(sel)
    bad["pairs"]["H2-s2-minus-N-s2"]["classification"]["PDMS"] = "adverse"
    try:
        m.decide(list(m.REGISTERED), {}, evid, bad, nav, ho, h1id)
    except m.EvidenceRefused as e:
        assert "classification" in str(e)
    else:
        raise AssertionError("a classification that disagrees with the re-derived one was accepted")
    sel_e, nav_e, ho_e, evid_e, h1id_e = synth(m, est="paired log-cluster bootstrap, 1000 resamples, 95 %, seed 20260927")
    try:
        m.decide(list(m.REGISTERED), {}, evid_e, sel_e, nav_e, ho_e, h1id_e)
    except m.EvidenceRefused:
        pass
    else:
        raise AssertionError("a harness output with 1000 resamples was accepted")
    try:
        m.decide(list(m.REGISTERED), {}, evid, None, nav, ho, h1id)
    except m.EvidenceRefused:
        return
    raise AssertionError("a complete set without the selection read was accepted")


def chk_families_and_md(m):
    f = m.families_block(None)
    assert f["STRATEGIC"]["status"] == "PAUSED" and "item 25" in f["STRATEGIC"]["ruling"] and "2026-09-19" in f["STRATEGIC"]["ruling"]
    for k in ("LONGITUDINAL", "LATERAL", "TACTICAL"):
        assert f[k]["measured_at_S1"] and f[k]["not_computed"], k
    assert "T0-like" in f["tier"] and "10000 resamples" in f["estimator"]
    ids = [a["id"] for a in m.AMBIGUITIES]
    assert ids == [f"AMB-{i}" for i in range(1, 10)] and all(a["used"] and a["alt"] and a["text"] for a in m.AMBIGUITIES)
    rd = {"outcome": m.decide([], {}, {}, None, None, None, {}), "arms": {"present": []}, "families": f, "timings_s": {"total": 1.0}}
    md = m.render_md(rd)
    assert "INCOMPLETE" in md and "PAUSED" in md and "AMB-1" in md and "T0-like" in md and "never driving performance" in md


# ------------------------------------------------------------------------------------------------------------------ checks: the load_scores shim
def chk_loader_shim(m):
    m.check_loader_equivalence(RV)
    bad_rows = lambda p: {k: dict(v, scores=v["scores"] + 1.0) for k, v in RV.load_scores(p).items()}            # noqa: E731
    def view(p):                                                                                                  # rows that are VIEWS of one array (what the original pins per row)
        arr, o = np.load(p)["scores"], RV.load_scores(p)
        return {k: dict(o[k], scores=arr[i]) for i, k in enumerate(o)}
    for fn in (bad_rows, view):
        try:
            m.check_loader_equivalence(RV, load_fn=fn)
        except m.EvidenceRefused:
            continue
        raise AssertionError("a loader that does not reproduce rl_val_eval.load_scores was accepted")
    with tempfile.TemporaryDirectory() as d:                                                                      # the shim returns OWNED rows and the same dict
        p = os.path.join(d, "s.npz")
        keys = [("L1", "a", 0, 0), ("L1", "b", 0, 0), ("L2", "c", 1, 1)]
        sc = np.arange(3 * 4 * 2, dtype=np.float64).reshape(3, 4, 2)
        RV.write_scores_npz(p, keys, [7, 8, 9], sc)
        got = m.load_scores_copy(p, RV)
        assert sorted(got) == sorted(keys) and got[("L2", "c", 1, 1)]["ckpt"] == 9 and got[("L1", "b", 0, 0)]["scores"].tolist() == sc[1].tolist()
        assert all(v["scores"].base is None for v in got.values())
        RV.write_scores_npz(p, keys[:1] + keys[:1], [1, 1], sc[:2])
        try:
            m.load_scores_copy(p, RV)
        except ValueError:
            return
        raise AssertionError("a duplicate key was accepted")


def chk_safe_tar(m):
    import tarfile
    with tempfile.TemporaryDirectory() as d:
        good, evil = os.path.join(d, "g.tar"), os.path.join(d, "e.tar")
        f = os.path.join(d, "x.txt")
        open(f, "w").write("hi")
        with tarfile.open(good, "w") as t:
            t.add(f, arcname="heldout_pdm/x.txt")
        with tarfile.open(evil, "w") as t:
            t.add(f, arcname="../escape.txt")
        names = m.safe_extract_tar(good, os.path.join(d, "out"))
        assert names == ["heldout_pdm/x.txt"] and os.path.isfile(os.path.join(d, "out", "heldout_pdm", "x.txt"))
        try:
            m.safe_extract_tar(evil, os.path.join(d, "out2"))
        except m.FetchRefused:
            assert not os.path.exists(os.path.join(d, "escape.txt"))
            return
        raise AssertionError("a path-traversal member was extracted")


def chk_plumbing_check(m):
    """plumbing_check wires the arm's OWN head file and ITS seed's floor, and returns status only (no value). The subprocess is stubbed: this proves the argument assembly."""
    calls = []

    def stub(cmd, log_path):
        calls.append(cmd)
        return {"rc": 0, "seconds": 0.0, "log": os.path.basename(log_path), "tail": []}
    m.run_cmd = stub
    m.floor_inputs = lambda floor_dir, pkg: {0: "/f/s0.npz", 1: "/f/s1.npz", 2: "/f/s2.npz"}
    with tempfile.TemporaryDirectory() as d:
        a = types.SimpleNamespace(python="py", cache="/c", ctx_cache="/ctx", floor_dir="/f")
        assert m.plumbing_check(a, "N-s0", ["N-s0"], "/e", "/p", d)["status"] == "SKIPPED"                  # an N arm has no head
        assert m.plumbing_check(a, "H2-s1", ["N-s0"], "/e", "/p", d)["status"] == "SKIPPED" and calls == []   # not a clean arm: nothing runs
        r = m.plumbing_check(a, "H2-s1", ["H2-s1"], "/e", "/p", d)
        assert len(calls) == 2 and "forward" in r and "perception_navtest" in r and "values" in r
        fwd, per = calls
        assert "H2-s1=" + os.path.join("/c", "arms", "H2-s1", "aux_drv.pt") in fwd and fwd[fwd.index("--device") + 1] == "cpu"
        assert "H2-s1=/f/s1.npz" in per and per[per.index("--pred-init") + 1] == "H2-s1=/f/s1.npz", per          # the floor of seed 1, not seed 0
        assert not any(isinstance(v, float) for v in json.dumps(r).split()) and "auc" not in json.dumps(r).lower()


def chk_argv_helpers(m):
    a = ["--seed", "1", "--aux-drv", "--aux-drv-feed", "on", "--lane-head", "--lr", "1e-4"]
    fl = m.argv_flags(a)
    assert fl == {"--seed": "1", "--aux-drv": "", "--aux-drv-feed": "on", "--lane-head": "", "--lr": "1e-4"}
    assert m.argv_class(fl) == "H2" and m.argv_class(m.argv_flags(["--aux-drv"])) == "H1" and m.argv_class(m.argv_flags(["--lr", "1"])) == "N"
    assert m.argv_class(m.argv_flags(["--aux-drv", "--aux-drv-feed", "off"])) == "H1"
    core1 = m.argv_core(m.argv_flags(argv_for("N", 0, "N-s0")))
    assert core1 == m.argv_core(m.argv_flags(argv_for("N", 2, "N-s2"))) == m.argv_core(m.argv_flags(argv_for("H2", 1, "H2-s1")))
    import hashlib
    assert m.argv_sha256(["--a", "1"]) == hashlib.sha256(b'["--a","1"]').hexdigest()


CHECKS = [chk_pins, chk_harness_json_pins, chk_reads_table, chk_incomplete_gate, chk_refused_gate, chk_evidence_control, chk_swap_manifest, chk_swap_argv, chk_registered_values,
          chk_k5_problems, chk_plumbing_check, chk_cross_arm, chk_fetch, chk_arm_paths, chk_p3_floor, chk_p3_matches_harness, chk_classify_matches_harness, chk_k2, chk_k6, chk_k3, chk_promote_h2,
          chk_promote_h1, chk_h1_identity, chk_decide_complete, chk_decide_rederivation, chk_families_and_md, chk_loader_shim, chk_safe_tar, chk_argv_helpers]
BYNAME = {f.__name__: f for f in CHECKS}


def run_checks(mod, names=None):
    failing = []
    for f in (CHECKS if names is None else [BYNAME[n] for n in names]):
        try:
            f(mod)
        except Exception as e:                                                   # noqa: BLE001
            failing.append(f"{f.__name__}[{type(e).__name__}]")
    return failing


# ------------------------------------------------------------------------------------------------------------------ real-data checks (the three banked N arms)
def real_checks(m) -> list:
    out = []
    arms = [a for a in ("N-s0", "N-s1", "N-s2") if os.path.isfile(os.path.join(CACHE, "arms", a, "run_check.json"))]
    if len(arms) < 3:
        print(f"SKIP real-data checks: the cache {CACHE} does not hold the three N arms (run s1_read.py read --arms N-s0 N-s1 N-s2 first)")
        return out
    evs = {a: m.load_evidence(a, CACHE) for a in arms}
    for a in arms:
        P, _ = m.check_arm_evidence(a, evs[a], {})
        assert P == [], (a, P)
        assert m.argv_sha256(evs[a]["token"]["binding"]["argv"]) == evs[a]["token"]["binding"]["argv_sha256"]
    out.append("real: the three banked N arms pass check_arm_evidence under their own labels")
    shas = {a: m.sha256_file(os.path.join(CACHE, "arms", a, "dump", "scores_update400_B.npz")) for a in arms}
    P, _ = m.check_cross_arm(evs, shas, None)
    assert P == {}, P
    out.append("real: check_cross_arm (argv parity, distinct score dumps) clean on the three N arms")
    for true, wrong in (("N-s1", "N-s2"), ("N-s2", "N-s0"), ("N-s0", "N-s1")):                      # the REAL evidence of one arm filed under another label
        c = codes(m.check_arm_evidence(wrong, evs[true], {})[0])
        assert {"ID_ARGV_SEED", "ID_ARGV_OUT", "ID_ARGV_DUMP", "ID_RUNCHECK"} <= c, (true, wrong, c)
    for wrong in ("H1-s0", "H2-s0"):
        assert "ID_ARGV_CLASS" in codes(m.check_arm_evidence(wrong, evs["N-s0"], {})[0])
    out.append("real: N-s1's evidence under the labels N-s2 / N-s0 / ... and N-s0's under H1-s0 / H2-s0 are all refused")
    for a in arms:
        s = {"N-s0": 0, "N-s1": 1, "N-s2": 2}[a]
        assert evs[a]["sft"][0]["heldout_logs"] == 24
        assert m.argv_flags(evs[a]["token"]["binding"]["argv"])["--seed"] == str(s)
    out.append("real: argv --seed equals the label seed on all three")
    return out


# ------------------------------------------------------------------------------------------------------------------ RED mutations
CLASSES = {"pin", "gate", "identity", "fetch", "rule", "estimator", "shim", "parity"}
M = [
    ("n_boot_1000", "pin", "N_BOOT = 10000", "N_BOOT = 1000", ["chk_pins"]),
    ("seed_changed", "pin", "SEED = 20260927", "SEED = 20260928", ["chk_pins"]),
    ("rule_navsim_v1_lane", "pin", 'RULE = "navsim_v1"', 'RULE = "navsim_v1_lane"', ["chk_pins"]),
    ("k2_bar_070", "pin", "K2_IOU_MIN = 0.80", "K2_IOU_MIN = 0.70", ["chk_pins", "chk_k2"]),
    ("k6_factor_08", "pin", "K6_FACTOR, K6_FRAC = 0.9, 0.10", "K6_FACTOR, K6_FRAC = 0.8, 0.10", ["chk_pins", "chk_k6"]),
    ("selection_cmd_no_nboot", "pin", '"--rule", RULE, "--n-boot", str(N_BOOT), "--out", out]', '"--rule", RULE, "--out", out]', ["chk_pins"]),
    ("harness_estimator_unchecked", "estimator", 'if f"{N_BOOT} resamples" not in est or f"seed {SEED}" not in est:', "if False:", ["chk_harness_json_pins", "chk_decide_rederivation"]),
    ("harness_tool_unchecked", "estimator", 'if js.get("tool") != HARNESS_TOOL:', "if False:", ["chk_harness_json_pins"]),
    ("incomplete_gate_off", "gate", '    if miss:\n        return {"status": "INCOMPLETE"', '    if False:\n        return {"status": "INCOMPLETE"', ["chk_incomplete_gate"]),
    ("refused_gate_off", "gate", '    if refused:\n        return {"status": "REFUSED"', '    if False:\n        return {"status": "REFUSED"', ["chk_refused_gate"]),
    ("reads_table_sel_without_n", "gate", 'rows.append((f"SEL_{c}_s{s}", "sel_pair", [f"{c}-s{s}"] + N))', 'rows.append((f"SEL_{c}_s{s}", "sel_pair", [f"{c}-s{s}"]))', ["chk_reads_table"]),
    ("reads_table_k3_without_n", "gate", 'rows.append((f"K3_{c}_s{s}", "k3", [f"{c}-s{s}", f"N-s{s}"]))', 'rows.append((f"K3_{c}_s{s}", "k3", [f"{c}-s{s}"]))', ["chk_reads_table"]),
    ("swap_seed_unchecked", "identity", 'if fl.get("--seed") != str(seed):', "if False:", ["chk_swap_argv"]),
    ("swap_class_unchecked", "identity", "if argv_class(fl) != cls:", "if False:", ["chk_swap_argv"]),
    ("swap_out_unchecked", "identity", "if out != arm:", "if False:", ["chk_swap_argv"]),
    ("swap_manifest_row_unchecked", "identity", "    if not rows:\n", "    if False:\n", ["chk_swap_manifest"]),
    ("swap_manifest_last_unchecked", "identity", 'elif ev["manifest"][-1].get("arm") != arm:', "elif False:", ["chk_swap_manifest"]),
    ("swap_runcheck_unchecked", "identity", 'if rc.get("arm") != arm or rc.get("ok") is not True or rc.get("problems"):', "if False:", ["chk_swap_manifest"]),
    ("argv_sha_unchecked", "identity", "if argv_sha256(argv) != b.get(\"argv_sha256\"):", "if False:", ["chk_swap_argv"]),
    ("updates_unchecked", "identity", 'if fl.get("--max-updates") != str(N_UPDATES) or fl.get("--eval-every") != str(N_UPDATES):', "if False:", ["chk_registered_values"]),
    ("k5_fields_unchecked", "identity", "bad = [f for f in K5_FIELDS[cls] if g12.get(f) is not True]", "bad = []", ["chk_k5_problems"]),
    ("dup_scores_unchecked", "parity", "if shas[a] == shas[b]:", "if False:", ["chk_cross_arm"]),
    ("h2_equals_n_unchecked", "parity", 'if a.startswith("H2-") and', "if False and", ["chk_cross_arm"]),
    ("argv_core_unchecked", "parity", "if c != cores[ref_arm]:", "if False:", ["chk_cross_arm"]),
    ("fetch_sha_unchecked", "fetch", 'if sha != meta["sha256"]:', "if False:", ["chk_fetch"]),
    ("fetch_size_unchecked", "fetch", 'if size != meta.get("size"):', "if False:", ["chk_fetch"]),
    ("fetch_blob_unchecked", "fetch", 'if b != meta["blob_id"]:', "if False:", ["chk_fetch"]),
    ("fetch_cache_trusted", "fetch", '            row = verify_file(dest, meta)\n            row.update({"path": rel, "cached": True})',
     '            row = {"size": os.path.getsize(dest), "sha256": "x", "verified_against": "cache"}\n            row.update({"path": rel, "cached": True})', ["chk_fetch"]),
    ("fetch_unverifiable_accepted", "fetch", 'raise FetchRefused(f"{path}: the listing carries neither an LFS sha256 nor a git blob id: nothing to verify against")',
     'return {"size": size, "sha256": sha, "verified_against": "none", "expected": None}', ["chk_fetch"]),
    ("tar_traversal_allowed", "fetch", 'if os.path.isabs(n) or ".." in n.replace("\\\\", "/").split("/") or not (m.isfile() or m.isdir()):', "if False:", ["chk_safe_tar"]),
    ("partial_push_is_an_arm", "fetch", "    absent = [w for w in want if w not in listing]\n", "    absent = []\n", ["chk_arm_paths"]),
    ("p3_floor_dropped", "rule", "    if not (floor_lo > 0.0):\n        return \"P3_AT_FLOOR\"", "    if False:\n        return \"P3_AT_FLOOR\"", ["chk_p3_floor", "chk_p3_matches_harness"]),
    ("p3_floor_ge_zero", "rule", "if not (floor_lo > 0.0):", "if not (floor_lo >= 0.0):", ["chk_p3_floor", "chk_p3_matches_harness"]),
    ("p3_unread_is_a_pass", "rule", '    if floor_lo is None:\n        return "FLOOR_UNREAD"\n', "", ["chk_p3_floor", "chk_p3_matches_harness"]),
    ("p3_margin_dropped", "rule", "thr = max(P3_FLOOR, auc_scorer + P3_MARGIN)", "thr = P3_FLOOR", ["chk_p3_floor", "chk_p3_matches_harness"]),
    ("p3_kill_boundary", "rule", "elif auc_head < auc_scorer:", "elif auc_head <= auc_scorer:", ["chk_p3_floor", "chk_p3_matches_harness"]),
    ("classify_floor_ignored", "rule", 'lo > floor["floor"].get(q, float("inf"))', "lo > 0.0", ["chk_classify_matches_harness"]),
    ("k2_boundary", "rule", '"K2_FIRES" if iou < K2_IOU_MIN else "K2_CLEARED"', '"K2_FIRES" if iou <= K2_IOU_MIN else "K2_CLEARED"', ["chk_k2"]),
    ("k2_navtest_not_governing", "rule", "    if n is not None:\n        out = dict(n, source", "    if False:\n        out = dict(n, source", ["chk_k2"]),
    ("k2_harness_label_unchecked", "rule", 'if nav_arm.get("k2", {}).get("decision") not in (None, n["decision"]):', "if False:", ["chk_k2"]),
    ("k6_strict_gt", "rule", 'fires = row["aux_bce"] >= K6_FACTOR * bce0', 'fires = row["aux_bce"] > K6_FACTOR * bce0', ["chk_k6"]),
    ("k6_first_row_not_10pct", "rule", 'cand = [r for r in train if r["update"] >= need]', "cand = train", ["chk_k6"]),
    ("k3_mem_limit_2gb", "rule", "mem = bool(ret is not None and ret > K3_GB)", "mem = bool(ret is not None and ret > 2 * K3_GB)", ["chk_k3"]),
    ("k3_time_limit_30pct", "rule", "tim = bool(rise is not None and rise > K3_TIME)", "tim = bool(rise is not None and rise > 0.3)", ["chk_k3"]),
    ("k3_gpu_delta_unread", "rule", "gpu = bool(dgb is not None and dgb > K3_GB)", "gpu = False", ["chk_k3"]),
    ("promo_provisional_needs_two", "rule", "n_above >= need", "n_above >= PAIRS_NEEDED", ["chk_promote_h2"]),
    ("promo_floor_unmeasured_ok", "rule", "c2 = c1 and floor_measured and n_above", "c2 = c1 and n_above", ["chk_promote_h2"]),
    ("promo_adverse_ignored", "rule", "c3 = not adverse", "c3 = True", ["chk_promote_h2"]),
    ("promo_kill_conditions_ignored", "rule", "c4 = not fired", "c4 = True", ["chk_promote_h2"]),
    ("promo_p3_majority_only", "rule", 'c1 = (n_ok == len(seeds)) if c1_mode == "all" else (n_ok >= 2)', "c1 = (n_ok >= 2)", ["chk_promote_h2"]),
    ("h1_identity_branch_off", "rule", '    if identity_defect:\n        return {"outcome": "H1_IDENTITY_DEFECT"', '    if False:\n        return {"outcome": "H1_IDENTITY_DEFECT"', ["chk_promote_h1", "chk_decide_complete"]),
    ("h1_kill_ignored", "rule", '    bad = sorted(f"{s}:{n}" for n, d in (("K2", k2), ("K6", k6))', '    bad = [] and sorted(f"{s}:{n}" for n, d in (("K2", k2), ("K6", k6))', ["chk_promote_h1"]),
    ("h1_defect_never", "rule", '"defect": not sel_ident,', '"defect": False,', ["chk_h1_identity"]),
    ("p3_rederive_off", "rule", 'if local != d["decision"]:', "if False:", ["chk_decide_rederivation"]),
    ("classification_rederive_off", "rule", 'if cls[f"{c}-s{s}"] != pair["classification"]:', "if False:", ["chk_decide_rederivation"]),
    ("strategic_not_paused", "rule", '"STRATEGIC": {"status": "PAUSED"', '"STRATEGIC": {"status": "PASS"', ["chk_families_and_md"]),
    ("plumbing_wrong_seed_floor", "rule", "{arm: fl[seed]}", "{arm: fl[0]}", ["chk_plumbing_check"]),
    ("shim_row_is_a_view", "shim", 'np.array(a["scores"][i], np.float64)', 'np.asarray(a["scores"][i], np.float64)', ["chk_loader_shim"]),
    ("shim_off_by_one_row", "shim", 'np.array(a["scores"][i], np.float64)', 'np.array(a["scores"][i - 1], np.float64)', ["chk_loader_shim"]),
    ("shim_duplicates_allowed", "shim", '        if k in out:\n            raise ValueError(f"duplicate scores for set {RV.kstr(k)} in {path}")\n        out[k] = {"ckpt": int(a', '        out[k] = {"ckpt": int(a', ["chk_loader_shim"]),
]


def main() -> int:
    src = open(SRC_PATH, encoding="utf-8").read()
    n = 0
    real = load(src, "s1_read_real")
    failing = run_checks(real)
    assert not failing, f"the unmutated module FAILS its own checks: {failing}"
    n += len(CHECKS)
    print(f"unmutated module: {len(CHECKS)} checks GREEN")
    extra = real_checks(real)
    n += len(extra)
    for e in extra:
        print(f"GREEN {e}")
    print(f"{'mutation':32s} {'class':9s} must go RED -> went RED")
    for i, (name, cls, old, new, expect) in enumerate(M):
        cnt = src.count(old)
        assert cnt == 1, f"mutation {name}: the snippet occurs {cnt} times (must be exactly 1): a vacuous mutation proves nothing"
        try:
            mod = load(src.replace(old, new), f"s1_read_mut{i}")
        except SyntaxError as e:
            raise AssertionError(f"mutation {name} does not compile: {e}")
        red = run_checks(mod, expect)
        red_names = {r.split("[")[0] for r in red}
        missing = [c for c in expect if c not in red_names]
        assert not missing, f"mutation {name} ({cls}) stayed GREEN on {missing}: the suite cannot see it"
        print(f"{name:32s} {cls:9s} {', '.join(red)}")
        n += 1
    assert {c for _, c, *_ in M} >= CLASSES, "a required mutation class is missing"
    n += 1
    print(f"ZZTEST_PASS {n}  ({len(CHECKS)} checks + {len(extra)} real-data + {len(M)} RED mutations + coverage)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
