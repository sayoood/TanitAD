"""Tests of SEARCH-NAVTEST-PIPE.  CPU only.  Every gate has a RED arm; mutation_check.py disables each `@GUARD` line in turn and this suite must then fail.

    cd code && CUDA_VISIBLE_DEVICES=-1 python -m pytest test_navpipe.py -q -p no:cacheprovider --basetemp=<scratch>

Planted data only for the chain-level tests (a fake runner stands in for the GPU / search / harness children); the REAL banked Step B / carve-B / navtest-seam files are read (never modified, no navtest score computed)
for: the token file and chunk plan, the banked seam's coverage, the banked CSV's formula, canonical merging, and the analysis stage's exact reproduction of Step B's banked P2HEAD read.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import shutil
import sys

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import navpipe_core as NC  # noqa: E402
NC.add_code_paths(spc=True)                         # the repo-relative sibling packages (SPC_STEPB_CODE / L1_CODE / SPC_CODE override); no dev-box path is named here
STEPB_CODE, L1_CODE = NC.STEPB_CODE, NC.L1_CODE
import navpipe_chain as CH  # noqa: E402
import navpipe_launch as NL  # noqa: E402
import navpipe_analyse as NA  # noqa: E402
import navpipe_bridge as NB  # noqa: E402
import navpipe_gpu_stage as GS  # noqa: E402
import make_navtest_bundle as MB  # noqa: E402
from navpipe_core import Refusal  # noqa: E402

REAL = {"tokens": CH.NAVTEST_TOKENS, "seam": CH.BANKED_SEAM, "csv": CH.BANKED_CSV}
needs_real = pytest.mark.skipif(not all(os.path.exists(p) for p in REAL.values()), reason="banked navtest files absent")
needs_banked_b = pytest.mark.skipif(not all(os.path.exists(p) for p in (CH.BANKED_SEARCH, CH.BANKED_LABELS, CH.BANKED_RESULT, CH.SRC_BUNDLE)), reason="banked Step B files absent")


def raises(cls, fn, *a, **k):
    with pytest.raises(Refusal) as e:
        fn(*a, **k)
    assert e.value.cls == cls, (e.value.cls, e.value.reason)
    return e.value


# ====================================================================================================== chunk plan
def planted_doc(n_logs=4, per_log=(5, 6, 4, 5)):
    toks, tl = [], {}
    for j in range(n_logs):
        for i in range(per_log[j]):
            t = hashlib.sha1(f"t{j}-{i}".encode()).hexdigest()[:16]
            toks.append(t)
            tl[t] = f"2021.01.0{j}.00.00.00_veh-1_{j:05d}_{j + 1:05d}"
    return {"rule": "planted", "tokens": toks, "token_log": tl}


def test_plan_partitions_whole_logs_and_is_deterministic():
    d = planted_doc()
    a, b = NC.plan_chunks(d, 11), NC.plan_chunks(d, 11)
    assert a == b and [c["n"] for c in a] == [11, 9]
    seen = [lg for c in a for lg in c["logs"]]
    assert sorted(seen) == sorted(set(d["token_log"].values())) and len(seen) == len(set(seen))
    for c in a:
        ts = NC.chunk_tokens(d, c)
        assert ts == sorted(ts) and NC.tokens_sha256(ts) == c["tokens_sha256"] and c["n"] <= 11


def test_plan_refuses_a_log_over_the_cap_and_duplicate_tokens():
    d = planted_doc()
    raises("gate", NC.plan_chunks, d, 5)
    d2 = {**d, "tokens": d["tokens"] + [d["tokens"][0]]}
    raises("gate", NC.plan_chunks, d2, 11)


def test_plan_is_recomputed_never_trusted():
    d = planted_doc()
    plan = NC.plan_doc(d, "x" * 64, 11)
    NC.verify_plan(plan, d, "x" * 64)
    bad = json.loads(json.dumps(plan))
    bad["chunks"][0]["logs"] = bad["chunks"][0]["logs"][:-1]
    raises("gate", NC.verify_plan, bad, d, "x" * 64)
    t = json.loads(json.dumps(plan["chunks"][0]))
    t["tokens_sha256"] = "0" * 64
    raises("gate", NC.chunk_tokens, d, t)


@needs_real
def test_real_navtest_plan_fits_the_budgets():
    doc = json.load(open(REAL["tokens"], encoding="utf-8"))
    plan = NC.plan_doc(doc, NC.sha256_file(REAL["tokens"]), 1000)
    ch = plan["chunks"]
    assert plan["n_tokens"] == 12146 and plan["n_logs"] == 136 and sum(c["n"] for c in ch) == 12146
    assert max(c["n"] for c in ch) <= 1000 and 12 <= len(ch) <= 16
    worst = max(NC.estimate_override_chars(NC.chunk_tokens(doc, c), c["logs"]) for c in ch)
    assert worst < NC.HARNESS_ARG_BUDGET, worst                          # the largest chunk's Hydra override fits the Windows command line
    assert max(c["n"] for c in ch) * NC.CTX_BYTES_PER_TOKEN < 4.0 * 2 ** 30   # the per-chunk ctx peak


PLANT_PREREG = "658ba1576af175a9a345692e284b1515604a6065279b921b2a4ca6dead6529d1"       # the sealed SEARCH-P2-CONFIRM text's sha256 (the C1 chain writes it into chain_state.json at SEAL_OK)


def c1_doc(**kw):
    """a planted RESULT_SPC.json with the keys `spc_analyse.analyse_c1` writes and the licence reads (the shape is pinned against the real code/banked data in test_navpipe_licence.py)"""
    d = {"read": "C1", "arm": "P2HEAD (no head substitution)", "outcome": "O1_CONFIRMED", "n_tokens": 905, "n_logs": 24,
         "endpoint_dPDMS_points": {"point": 0.3273, "lo": 0.0565, "hi": 0.7216}, "safety_points": {"nc": 0.0, "dac": 0.17, "ttc": 0.25, "c": 0.0},
         "inert": {"inert": False, "frac_moved": 0.165, "n": 2715, "thr_m": 0.5, "min_frac": 0.05}, "moved_share": 0.34}
    d.update(kw)
    return d


class C1:
    """a planted C1 work dir: RESULT_SPC.json + the chain_state.json beside it; .path/.sha are what the operator types"""

    def __init__(self, tmp, doc=None, state=None, name="c1", raw=None):
        self.dir = os.path.join(str(tmp), name)
        self.doc = doc
        if doc is None and raw is None:
            self.doc = c1_doc()
        elif doc is None:
            try:
                self.doc = json.loads(raw)
            except ValueError:
                self.doc = None
            self.doc = self.doc if isinstance(self.doc, dict) else None
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, "RESULT_SPC.json")
        if raw is not None:
            open(self.path, "w", encoding="utf-8", newline="\n").write(raw)
        else:
            json.dump(doc if doc is not None else c1_doc(), open(self.path, "w", encoding="utf-8", newline="\n"), indent=1)       # allow_nan: a NaN round-trips as the literal NaN
        if state is not False:
            json.dump(state if state is not None else {"stage": "LABEL_DONE", "done": list(NC.C1_STAGES) + ["SEARCH_DONE"], "prereg_sha256": PLANT_PREREG}, open(os.path.join(self.dir, "chain_state.json"), "w"))
        self.sha = NC.sha256_file(self.path)


# ====================================================================================================== G-SEAL (a) + the C1 licence

DOSE_STUB_SRC = """\"\"\"TEST DOUBLE of the dose package's dose_chain.py AT REVISION 1 for planted worlds: the c2_config table (MM_RULING_DOSE.md DQ4) as the dose chain writes it, with "moved share" = `inert.frac_moved` (MM_RULING_C2FINAL.md point 1).
Production code CALLS the real file by path; only these tests carry a copy, and test_navpipe_select.py compares this copy with the real file over a grid.\"\"\"
DOSE_STAGES = ("SEAL_OK", "AMEND_SEAL_OK", "DATA_OK", "C1_DONE", "BUNDLE_OK", "SUBID_PASS", "REACH_PASS", "SEARCH_DONE", "SELFREP_PASS", "LABEL_DONE")
SAFETY = ("nc", "dac", "ttc", "c")


def frac_moved_of(read):
    try:
        return float(read["inert"]["frac_moved"])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def licensed(read, strict=False):
    ok = ("O1_CONFIRMED",) if strict else ("O1_CONFIRMED", "O2_DIRECTIONAL")
    try:
        safety = all(float(read["safety_points"][k]) >= -0.25 for k in SAFETY)
    except (KeyError, TypeError, ValueError):
        return False
    return bool(read.get("outcome") in ok and safety and frac_moved_of(read) >= 0.05)


def c2_config(c1_read, dose_read, contrast_lo):
    c1, dose_any, dose_strict = licensed(c1_read), licensed(dose_read), licensed(dose_read, strict=True)
    try:
        lo_pos = bool(float(contrast_lo) > 0.0)
    except (TypeError, ValueError):
        lo_pos = False
    if c1 and not dose_any:
        cfg, row = "C8", 1
    elif c1 and dose_any:
        cfg, row = ("O4" if lo_pos else "C8"), 3
    elif dose_strict:
        cfg, row = "O4", 2
    else:
        cfg, row = None, 4
    return {"config": cfg, "row": row, "why": "stub row %d" % row, "c1_licensed": c1, "dose_licensed": dose_any, "dose_o1_strict": dose_strict, "contrast_lower_ci_gt_0": lo_pos}
"""

# the PRE-REVISION-1 semantics (the sealed-then-defective dose_chain.py: "moved share" = the file's `moved_share`), kept just to plant a dose chain that disagrees with C2's own licence terms (G-LICENCE-ROW-AGREES)
DOSE_STUB_LEGACY_SRC = DOSE_STUB_SRC.replace('return bool(read.get("outcome") in ok and safety and frac_moved_of(read) >= 0.05)', 'return bool(read.get("outcome") in ok and safety and float(read.get("moved_share") or 0.0) >= 0.05)')
assert DOSE_STUB_LEGACY_SRC != DOSE_STUB_SRC
R1_STATUS = "Status: **SEALED** 2026-10-20T05:53:59+02:00 (planted REVISION 1)"


def plant_dose_package(root, src, status=R1_STATUS, pin=None):
    """a planted dose package `<root>/code/dose_chain.py` + `<root>/AMENDMENT_DOSE_R1.md` (+ its .sha256), laid out like the real `.../dose/`: the sealed Revision 1 text pins `code:dose_chain.py` to the file (or to `pin`)"""
    os.makedirs(os.path.join(root, "code"), exist_ok=True)
    path = os.path.join(root, "code", "dose_chain.py")
    open(path, "w", encoding="utf-8", newline="\n").write(src)
    text = "# AMENDMENT_DOSE_R1 (planted)\n\n" + status + "\n\n<!-- PINS-BEGIN -->\n```\ncode:dose_chain.py " + (pin or NC.sha_lf(path)) + "\n```\n<!-- PINS-END -->\n"
    open(os.path.join(root, NC.DOSE_R1_AMENDMENT), "w", encoding="utf-8", newline="\n").write(text)
    open(os.path.join(root, NC.DOSE_R1_SIDECAR), "w").write(hashlib.sha256(text.encode()).hexdigest() + "  " + NC.DOSE_R1_AMENDMENT + "\n")
    return path


EVID = {}                                           # amendment path -> the Evidence make_seal planted beside it (the dose-chain double, the bound dose result, the fixture record)


class Evidence:
    def __init__(self):
        self.kw = {}


def dose_doc(c1, stub, **kw):
    """a planted RESULT_DOSE.json (the keys `dose_chain.analyse_dose` writes and C2 reads), bound to `c1`; by default NOT licensed, so c2_config reads row 1 (C8) for a licensed C1"""
    c1_outcome = (c1.doc or {}).get("outcome")
    d = {"read": "DOSE", "arm": "P2HEAD search, O4 config (M 256, elites 32, widen 2.0), no head substitution", "outcome": "O3_NULL", "n_tokens": 905, "n_logs": 24, "c1_outcome": c1_outcome,
         "safety_points": {"nc": 0.0, "dac": 0.0, "ttc": 0.0, "c": 0.0}, "moved_share": 0.5, "inert": {"inert": False, "frac_moved": 0.3, "n": 2715, "thr_m": 0.5, "min_frac": 0.05},
         "contrast_O4_minus_C8": {"pooled": {"point": 0.1, "lo": -0.2, "hi": 0.4}},
         "c1": {"outcome": c1_outcome, "result_sha256": c1.sha, "bundle_sha256": "b" * 64, "config_c8_sha256": NC.CONFIG_C8_SHA256, "config_o4_sha256": NC.CONFIG_O4_SHA256}}
    d.update(kw)
    if "c2_config" not in d:
        keys = ("outcome", "safety_points", "inert", "moved_share")
        r = stub.c2_config({k: (c1.doc or {}).get(k) for k in keys}, {k: d.get(k) for k in keys}, d["contrast_O4_minus_C8"]["pooled"]["lo"])
        d["c2_config"] = {"config": r["config"], "row": r["row"], "why": r["why"]}
    return d


def plant_evidence(tmp, c1, dose=None, dose_state=None, fixture=None, name="dose", dose_kw=None, dose_mut=None, fixture_mut=None, dose_src=None):
    ev = Evidence()
    ev.stub_path = plant_dose_package(os.path.join(tmp, "dosepkg"), dose_src or DOSE_STUB_SRC)          # the double sits in a planted dose package SEALED AT REVISION 1, as the real one does
    ev.stub = NC.load_dose_chain(ev.stub_path, NC.sha_lf(ev.stub_path))
    ev.dir = os.path.join(tmp, name)
    os.makedirs(ev.dir, exist_ok=True)
    ev.dose_path = os.path.join(ev.dir, "RESULT_DOSE.json")
    doc = dose if dose is not None else dose_doc(c1, ev.stub, **(dose_kw or {}))
    if dose_mut:
        dose_mut(doc)
    json.dump(doc, open(ev.dose_path, "w", encoding="utf-8", newline="\n"), indent=1, allow_nan=True)
    if dose_state is not False:
        json.dump(dose_state if dose_state is not None else {"stage": "LABEL_DONE", "done": list(ev.stub.DOSE_STAGES), "prereg_sha256": PLANT_PREREG}, open(os.path.join(ev.dir, "chain_state.json"), "w"))
    ev.dose_sha = NC.sha256_file(ev.dose_path)
    rec = NC.make_fixture_record("IDENTICAL", HERE, NC.CARVE_B_SHA256, "p" * 64, None, None, "c" * 64)
    rec.update(fixture or {})
    if fixture_mut:
        fixture_mut(rec)
    ev.fixture_path = os.path.join(tmp, "FIXTURE_RECORD.json")
    json.dump(rec, open(ev.fixture_path, "w", encoding="utf-8", newline="\n"), indent=1, sort_keys=True)
    ev.kw = {"config": NC.CONFIG_C8, "dose_result": ev.dose_path, "dose_result_sha256": ev.dose_sha, "fixture_record": ev.fixture_path, "dose_chain": ev.stub_path}
    return ev


def check_c2_seal(am, sha, c1_path, c1_sha, **over):
    """NC.check_c2_seal with the evidence make_seal planted beside `am` (config C8, the bound dose result, the dose-chain double, the fixture record); `over` replaces any of them (None removes it)"""
    ev = EVID.get(os.path.abspath(am)) if am else None
    kw = dict(ev.kw) if ev else {}
    kw.update(over)
    return NC.check_c2_seal(am, sha, c1_path, c1_sha, **kw)


def make_seal(tmp, status="Status: **SEALED 2026-10-20**", drop=(), extra_pins=None, c1=None, tamper_pin=None, dose=None, dose_state=None, fixture=None, dose_kw=None, dose_mut=None, fixture_mut=None, dose_src=None):
    tmp = str(tmp)
    ck = os.path.join(tmp, "ck.pt")
    open(ck, "wb").write(b"ckpt")
    c1 = c1 if c1 is not None else C1(tmp)
    ev = plant_evidence(tmp, c1, dose=dose, dose_state=dose_state, fixture=fixture, dose_kw=dose_kw, dose_mut=dose_mut, fixture_mut=fixture_mut, dose_src=dose_src)
    pins = {"token_file_sha256": "a" * 64, "chunk_plan_sha256": "b" * 64, "ckpt_sha256": NC.sha256_file(ck), "config_c8_sha256": NC.CONFIG_C8_SHA256, "config_o4_sha256": NC.CONFIG_O4_SHA256, "head_s0_sha256": "d" * 64,
            "banked_seam_sha256": "e" * 64, "banked_csv_sha256": "f" * 64, "c1_prereg_sha256": PLANT_PREREG, "licence_ruling_sha256": NC.LICENCE_RULING_SHA256, "c2amend_ruling_sha256": NC.C2AMEND_RULING_SHA256,
            "inference_cost_status": "UNMEASURED", "bar_dpdms_points": "0.10", "bar_safety_points": "-0.25", "harness_script_sha256": "1" * 64, "seam_module_sha256": "2" * 64,
            "dose_chain_sha256": NC.sha_lf(ev.stub_path), "fixture_record_sha256": NC.sha256_file(ev.fixture_path)}
    for f in NC.OPERATIVE:
        pins[f"code:{f}"] = NC.sha_lf(os.path.join(HERE, f))
    pins.update(extra_pins or {})
    for k in drop:
        pins.pop(k, None)
    if tamper_pin:
        pins[tamper_pin] = "9" * 64
    body = "# C2 amendment (planted)\n\n" + status + "\n\nText.\n\n<!-- PINS-BEGIN -->\n" + "\n".join(f"{k} {v}" for k, v in pins.items()) + "\n<!-- PINS-END -->\n"
    am = os.path.join(tmp, "AMEND.md")
    open(am, "w", encoding="utf-8", newline="\n").write(body)
    sha = hashlib.sha256(body.encode()).hexdigest()
    open(os.path.join(tmp, "AMEND.sha256"), "w").write(sha + "  AMEND.md\n")
    c1.ev = ev
    EVID[os.path.abspath(am)] = ev
    return am, sha, c1, pins


def test_seal_passes_on_a_complete_sealed_amendment(tmp_path):
    am, sha, c1, pins = make_seal(tmp_path)
    r = check_c2_seal(am, sha, c1.path, c1.sha)
    assert r["amendment_sha256"] == sha and r["pins"]["bar_dpdms_points"] == "0.10"
    assert r["licence"]["outcome"] == "O1_CONFIRMED" and r["licence"]["c1_result_sha256"] == c1.sha and r["licence"]["moved_share_inert_frac_moved"] == 0.165
    assert r["pins"]["config_name"] == "C8" and r["pins"]["config_sha256"] == NC.CONFIG_C8_SHA256 and r["selection"]["row"] == 1 and r["fixture"]["verdict"] == "IDENTICAL"
    assert r["pins"]["config_name"] == "C8" and r["pins"]["config_sha256"] == NC.CONFIG_C8_SHA256 and r["selection"]["row"] == 1 and r["fixture"]["verdict"] == "IDENTICAL"
    assert "admission_record_sha256" not in NC.REQUIRED_PINS and "admission" not in r                    # EP-ADMISSION no longer licenses this arm (MM_RULING_C2_LICENCE.md)


SEAL_REASON = {"no_args": "SEAL_MISSING_ARGS", "no_c1_args": "LICENCE_MISSING_ARGS", "no_amend": "SEAL_AMENDMENT_MISSING", "no_sidecar": "SEAL_SIDECAR_MISSING", "typed_sha_wrong": "SEAL_SHA_MISMATCH",
               "edited_after_seal": "SEAL_SHA_MISMATCH", "draft": "SEAL_NOT_SEALED", "pin_missing": "SEAL_PIN_MISSING", "c1_prereg_pin_missing": "SEAL_PIN_MISSING", "licence_ruling_pin_missing": "SEAL_PIN_MISSING",
               "operative_unpinned": "SEAL_PIN_MISMATCH", "code_pin_wrong": "SEAL_PIN_MISMATCH", "cost_unstated": "SEAL_INFERENCE_COST_UNSTATED", "licence_ruling_pin_wrong": "SEAL_LICENCE_RULING_PIN",
               "amend_ruling_pin_missing": "SEAL_PIN_MISSING", "amend_ruling_pin_wrong": "SEAL_AMEND_RULING_PIN", "config_c8_pin_wrong": "SEAL_CONFIG_PIN", "config_o4_pin_wrong": "SEAL_CONFIG_PIN",
               "dose_chain_pin_missing": "SEAL_PIN_MISSING", "fixture_record_pin_missing": "SEAL_PIN_MISSING"}


@pytest.mark.parametrize("case", list(SEAL_REASON))
def test_seal_refuses(tmp_path, case):
    kw = {}
    if case == "draft":
        kw["status"] = "Status: **DRAFT**"
    if case == "pin_missing":
        kw["drop"] = ("ckpt_sha256",)
    if case == "c1_prereg_pin_missing":
        kw["drop"] = ("c1_prereg_sha256",)
    if case == "licence_ruling_pin_missing":
        kw["drop"] = ("licence_ruling_sha256",)
    if case == "operative_unpinned":
        kw["drop"] = ("code:navpipe_chain.py",)
    if case == "code_pin_wrong":
        kw["tamper_pin"] = "code:navpipe_core.py"
    if case == "cost_unstated":
        kw["extra_pins"] = {"inference_cost_status": "FINE"}
    if case == "licence_ruling_pin_wrong":
        kw["tamper_pin"] = "licence_ruling_sha256"
    if case == "amend_ruling_pin_missing":
        kw["drop"] = ("c2amend_ruling_sha256",)
    if case == "amend_ruling_pin_wrong":
        kw["tamper_pin"] = "c2amend_ruling_sha256"
    if case == "config_c8_pin_wrong":
        kw["tamper_pin"] = "config_c8_sha256"
    if case == "config_o4_pin_wrong":
        kw["tamper_pin"] = "config_o4_sha256"
    if case == "dose_chain_pin_missing":
        kw["drop"] = ("dose_chain_sha256",)
    if case == "fixture_record_pin_missing":
        kw["drop"] = ("fixture_record_sha256",)
    am, sha, c1, pins = make_seal(tmp_path, **kw)
    if case == "no_args":
        e = raises("seal", check_c2_seal, None, None, None, None)
    elif case == "no_c1_args":                                                                   # a sealed amendment alone is no licence
        e = raises("seal", check_c2_seal, am, sha, None, None)
    elif case == "no_amend":
        e = raises("seal", check_c2_seal, str(tmp_path / "nope.md"), sha, c1.path, c1.sha)
    elif case == "no_sidecar":
        os.remove(str(tmp_path / "AMEND.sha256"))
        e = raises("seal", check_c2_seal, am, sha, c1.path, c1.sha)
    elif case == "typed_sha_wrong":
        e = raises("seal", check_c2_seal, am, "0" * 64, c1.path, c1.sha)
    elif case == "edited_after_seal":
        open(am, "a").write("\nedited\n")
        e = raises("seal", check_c2_seal, am, sha, c1.path, c1.sha)
    else:
        e = raises("seal", check_c2_seal, am, sha, c1.path, c1.sha)
    assert e.code == 5
    assert e.reason == SEAL_REASON[case], (case, e.reason)                                      # the REASON, not just the class: a later check must not be able to stand in for a removed one


def test_chain_refuses_before_opening_any_data_without_the_seal(tmp_path):
    called = []
    cfg = CH.Cfg(str(tmp_path / "w"), mode="c2", token_file=str(tmp_path / "does_not_exist.json"), amendment=None)
    ch = CH.Chain(cfg, runner=lambda *a, **k: called.append(a) or 0, out=io.StringIO())
    assert ch.run() == 5 and not called
    out = ch.out.getvalue()
    assert "ZZNAVPIPE_STOP SEAL_MISSING_ARGS" in out and "DATA_OK" not in out and not os.path.exists(str(tmp_path / "w" / "chunks"))


# ====================================================================================================== G-DISK (b)
def test_disk_guard():
    assert NC.check_disk(".", 6.0, 0, lambda p: 6.5) == 6.5
    raises("gate", NC.check_disk, ".", 6.0, 0, lambda p: 5.9)
    raises("gate", NC.check_disk, ".", 6.0, 1 * 2 ** 30, lambda p: 6.9)            # 6.9 free - 1 GiB about to be written = 5.9 < 6
    assert NC.check_disk(".", 6.0, 1 * 2 ** 30, lambda p: 7.1) == 7.1


# ====================================================================================================== G-IDENT (c)
def fan(token, n_prop=64):
    rs = np.random.RandomState(int(token[:8], 16))
    f = (rs.randn(n_prop, 20, 3) * 0.7).astype(np.float32)
    f[..., 0] = np.cumsum(np.abs(f[..., 0]) + 0.3, axis=1)
    return f, int(rs.randint(0, n_prop))


def banked_for(tokens):
    poses, fps = {}, {}
    for t in tokens:
        f, k = fan(t)
        poses[t] = NC.exec_to_poses(NC.repair_last_heading(f[k])[None])[0]
        fps[t] = hashlib.sha1(t.encode()).hexdigest()
    return {"poses": poses, "fp": fps, "arm": "planted"}


def planted_bundle(tokens):
    raw = np.stack([fan(t)[0] for t in tokens])
    pick = np.array([fan(t)[1] for t in tokens], np.int64)
    return raw, pick


def test_identity_passes_and_each_red_arm_fails():
    toks = sorted(planted_doc()["tokens"])[:8]
    banked = banked_for(toks)
    raw, pick = planted_bundle(toks)
    r = NC.check_pick_identity(toks, raw, pick, banked)
    assert r["ok"] and r["n_bad"] == 0
    b2 = {**banked, "poses": {k: v.copy() for k, v in banked["poses"].items()}}
    b2["poses"][toks[3]][0, 0] = np.nextafter(b2["poses"][toks[3]][0, 0], np.float32(1e9))                 # ONE ulp
    r = NC.check_pick_identity(toks, raw, pick, b2)
    assert not r["ok"] and r["bad_examples"] == [toks[3]] and r["max_abs_diff_m"] > 0
    p2 = pick.copy()
    p2[5] = (p2[5] + 1) % 64                                                                               # a different proposal is the pick
    assert not NC.check_pick_identity(toks, raw, p2, banked)["ok"]
    b3 = {**banked, "poses": {k: v for k, v in banked["poses"].items() if k != toks[0]}}
    r = NC.check_pick_identity(toks, raw, pick, b3)
    assert not r["ok"] and r["n_absent_from_banked"] == 1
    assert not NC.check_pick_identity([], raw[:0], pick[:0], banked)["ok"]                                 # an empty chunk is not an identity
    raw2 = raw.copy()
    raw2[2, pick[2], 4, 0] += np.float32(1e-3)                                                             # one coordinate of the pick moved, on a pose NAVSIM's 0.5 s grid uses (t = 1.0 s)
    assert not NC.check_pick_identity(toks, raw2, pick, banked)["ok"]
    raw3 = raw.copy()
    raw3[2, pick[2], 5, 0] += np.float32(1e-3)                                                             # ... and on native pose 5 (t = 1.2 s), which NO output pose uses
    assert NC.check_pick_identity(toks, raw3, pick, banked)["ok"], "DOCUMENTED LIMIT: the seam banks 8 poses; native poses 0,3,5,8,10,11,13,15,16,18 are invisible to the identity"


def test_identity_is_bytewise_so_signed_zero_is_a_difference():
    toks = sorted(planted_doc()["tokens"])[:2]
    banked = banked_for(toks)
    raw, pick = planted_bundle(toks)
    raw[0, pick[0], :, 1] = 0.0                                                    # the pick runs dead straight: every y of the converted row is +0.0
    got = NC.exec_to_poses(NC.repair_last_heading(raw[:, :][np.arange(2), pick])[:1])[0]
    assert (got[:, 1] == 0).all() and not np.signbit(got[:, 1]).any()
    b = {**banked, "poses": {k: v.copy() for k, v in banked["poses"].items()}}
    b["poses"][toks[0]] = got.copy()
    b["poses"][toks[1]] = banked["poses"][toks[1]]
    assert NC.check_pick_identity(toks, raw, pick, b)["ok"]
    b["poses"][toks[0]][:, 1] = np.float32(-0.0)                                   # the SAME numbers up to the sign of zero
    assert np.array_equal(b["poses"][toks[0]], got)                                # numpy's array_equal calls them equal ...
    r = NC.check_pick_identity(toks, raw, pick, b)
    assert not r["ok"] and r["bad_examples"] == [toks[0]]                          # ... the gate (bytes) does not


def test_repair_and_the_searchs_identity_arm_are_the_executed_plan():
    """the T1 identity arm: tts_cem.composite(raw_pick, 0, 0) (what the search calls the pick's exec) == planner.executed (repair of the pick), BYTE for byte"""
    import tts_cem as T
    for t in sorted(planted_doc()["tokens"])[:6]:
        f, k = fan(t)
        assert T.composite(f[k], 0.0, 0.0)[0].tobytes() == NC.repair_last_heading(f[k]).tobytes()


def test_vendored_seam_module_is_pinned_and_passes_its_own_selftest(tmp_path, capsys):
    m = NC.seam_module()
    assert m.selftest() == 0                                                                               # SPEC E-1's analytic bars
    bad = tmp_path / "refe_navtest_seam.py"
    shutil.copy(os.path.join(PKG, "ref", "refe_navtest_seam.py"), bad)
    open(bad, "a").write("\n# edited\n")
    NC._SEAM = None
    raises("gate", NC.seam_module, str(bad))
    NC._SEAM = None
    assert NC.seam_module() is not None


@needs_real
def test_banked_seam_covers_every_navtest_token_with_a_fingerprint():
    doc = json.load(open(REAL["tokens"], encoding="utf-8"))
    b = NC.load_banked_seam(REAL["seam"])
    assert set(b["poses"]) == set(doc["tokens"]) and len(b["fp"]) == 12146
    assert all(v.shape == (8, 3) and v.dtype == np.float32 for v in list(b["poses"].values())[:50])


@needs_real
def test_harness_csv_reader_reproduces_the_registry_p2_point_and_enforces_the_formula(tmp_path):
    doc = json.load(open(REAL["tokens"], encoding="utf-8"))
    toks = sorted(doc["tokens"])
    sub, pdms = NC.read_harness_csv(REAL["csv"], toks)
    assert abs(100 * pdms.mean() - 88.82) < 0.005 and sub.shape == (12146, 6)                               # MODEL_REGISTRY 14.2: P2-s0 88.82 (read from the BANKED csv; nothing is scored here)
    rows = open(REAL["csv"], encoding="utf-8").read().splitlines()
    bad = tmp_path / "bad.csv"
    cells = rows[1].split(",")
    cells[-1] = str(float(cells[-1]) + 1e-6)
    open(bad, "w", newline="\n").write("\n".join([rows[0], ",".join(cells)] + rows[2:]) + "\n")
    t = rows[1].split(",")[1]
    raises("score", NC.read_harness_csv, str(bad), [t])                                                     # PDMS != NC*DAC*(5EP+5TTC+2C)/12
    raises("score", NC.read_harness_csv, REAL["csv"], ["ffffffffffffffff"])                                 # a missing token
    inv = tmp_path / "inv.csv"
    open(inv, "w", newline="\n").write(rows[0] + "\n" + rows[1].replace(",True,", ",False,", 1) + "\n")
    raises("score", NC.read_harness_csv, str(inv), [t])


# ====================================================================================================== T1 route pieces
def f_scores(poses):
    """a planted, deterministic 'harness': six sub-scores and the PDMS from the 8 poses (sensitive to a 0.5 m shift in x and to a lateral offset)"""
    p = np.asarray(poses, np.float64)
    ep = np.clip(0.3 + p[:, -1, 0] / 60.0, 0, 1)
    dac = np.clip(1.0 - np.abs(p[:, :, 1]).max(1) / 30.0, 0, 1)
    nc = np.where(p[:, -1, 0] < -50, 0.5, 1.0)
    sub = np.stack([nc, dac, ep, np.ones(len(p)), np.ones(len(p)), np.ones(len(p))], 1)
    return sub, nc * dac * (5 * ep + 5 + 2) / 12.0


def write_csv(path, tokens, sub, pdms):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(",token,valid," + ",".join(NC.CSV_COLS) + ",score\n")
        for i, t in enumerate(tokens):
            f.write(f"{i},{t},True," + ",".join(repr(float(x)) for x in sub[i]) + f",{repr(float(pdms[i]))}\n")
        f.write(f"{len(tokens)},average,True," + ",".join(["0"] * 7) + "\n")
    return path


def test_t1_planted_identity_and_its_red_arms():
    toks = sorted(planted_doc()["tokens"])[:6]
    raw, pick = planted_bundle(toks)
    pp = NC.exec_to_poses(NC.repair_last_heading(raw[np.arange(6), pick]))
    bsub, bp = f_scores(pp)
    ok = NC.t1_planted_identity(toks, pp, f_scores(pp), (bsub, bp), f_scores)
    assert ok["ok"] and ok["max_abs_diff"] == 0.0 and ok["control_tokens_changed"] > 0
    off_rows = (lambda s, p: (s, p + 1e-4))(*f_scores(pp))                                                          # the re-scored pick rows carry a label-path offset
    off = NC.t1_planted_identity(toks, pp, off_rows, (bsub, bp), f_scores)
    assert not off["ok"] and not off["identity_ok"] and off["control_ok"]
    blind = NC.t1_planted_identity(toks, pp, (bsub, bp), (bsub, bp), lambda q: (bsub, bp))                          # a route that cannot see a shift
    assert not blind["ok"] and not blind["control_ok"] and blind["identity_ok"]
    assert not NC.t1_planted_identity([], pp[:0], (bsub[:0], bp[:0]), (bsub[:0], bp[:0]), f_scores)["ok"]


def test_seam_writer_and_token_file(tmp_path):
    toks = sorted(planted_doc()["tokens"])[:4]
    banked = banked_for(toks)
    p = NC.write_seam(str(tmp_path / "s.npz"), toks, np.stack([banked["poses"][t] for t in toks]), banked["fp"], "x")
    z = np.load(p)
    assert z["poses"].shape == (4, 8, 3) and z["poses"].dtype == np.float32 and list(z["sampling"]) == [8.0, 0.5] and [str(t) for t in z["token"]] == toks
    raises("score", NC.write_seam, str(tmp_path / "s2.npz"), toks, np.zeros((3, 8, 3)), banked["fp"], "x")
    raises("score", NC.write_seam, str(tmp_path / "s3.npz"), toks, np.full((4, 8, 3), np.nan), banked["fp"], "x")
    raises("score", NC.write_seam, str(tmp_path / "s4.npz"), toks, np.zeros((4, 8, 3)), {}, "x")
    d = planted_doc()
    NC.write_tokens_file(str(tmp_path / "t.json"), toks, d["token_log"])
    big = ["%016x" % i for i in range(3000)]
    raises("score", NC.write_tokens_file, str(tmp_path / "big.json"), big, {t: "2021.01.01.00.00.00_veh-1_00000_00001" for t in big})


# ====================================================================================================== G-STATE, G-BANK (d)
def bank_world(tmp, cid=0, identity_ok=True, t1id_ok=True):
    tmp = str(tmp)
    os.makedirs(tmp, exist_ok=True)
    st = NC.State(os.path.join(tmp, "state.json"), "plan1")
    files = {}
    for k in ("bundle", "search", "scores", "identity"):
        files[k] = os.path.join(tmp, f"{k}.bin")
        open(files[k], "wb").write(k.encode() * 10)
    ctx = os.path.join(tmp, "ctx.npy")
    open(ctx, "wb").write(b"ctx" * 100)
    cj = os.path.join(tmp, "ctx.json")
    json.dump({"ctx_sha256": NC.sha256_file(ctx), "n": 3, "tokens_sha256": "x"}, open(cj, "w"))
    for s, kw in (("BUNDLED", {"identity_ok": identity_ok}), ("GATED", {}), ("SEARCHED", {}), ("SCORED", {"t1id_ok": t1id_ok})):
        st.advance(cid, s, **kw)
    return st, files, ctx, cj, os.path.join(tmp, "manifest.json")


def test_state_is_monotone_atomic_and_bound_to_its_plan(tmp_path):
    st = NC.State(str(tmp_path / "s.json"), "P")
    st.advance(0, "BUNDLED")
    raises("gate", st.advance, 0, "SEARCHED")                                       # a skip
    raises("gate", st.advance, 0, "NEW")                                            # a regress
    st2 = NC.State(str(tmp_path / "s.json"), "P")
    assert st2.status(0) == "BUNDLED"                                               # resumable
    raises("gate", NC.State, str(tmp_path / "s.json"), "OTHER")                     # another plan's state is refused
    assert not [f for f in os.listdir(tmp_path) if ".tmp" in f]


def test_bank_then_delete_happy_path(tmp_path):
    st, files, ctx, cj, man = bank_world(tmp_path)
    NC.bank_chunk(st, 0, files, cj, man)
    st.advance(0, "BANKED")
    assert NC.verify_banked(st, 0) == []
    NC.delete_ctx(st, 0, ctx)
    assert not os.path.exists(ctx) and st.status(0) == "DELETED"
    assert all(os.path.exists(p) for p in files.values())                           # only the ctx is deleted


def test_delete_red_arms(tmp_path):
    # not banked yet
    st, files, ctx, cj, man = bank_world(tmp_path / "a")
    raises("gate", NC.delete_ctx, st, 0, ctx)
    assert os.path.exists(ctx)
    # banked, then one banked output changes on disk
    st, files, ctx, cj, man = bank_world(tmp_path / "b")
    NC.bank_chunk(st, 0, files, cj, man)
    st.advance(0, "BANKED")
    open(files["search"], "ab").write(b"x")
    raises("gate", NC.delete_ctx, st, 0, ctx)
    assert os.path.exists(ctx)
    # banked, then one banked output disappears
    st, files, ctx, cj, man = bank_world(tmp_path / "c")
    NC.bank_chunk(st, 0, files, cj, man)
    st.advance(0, "BANKED")
    os.remove(files["scores"])
    raises("gate", NC.delete_ctx, st, 0, ctx)
    assert os.path.exists(ctx)
    # the manifest itself changed
    st, files, ctx, cj, man = bank_world(tmp_path / "d")
    NC.bank_chunk(st, 0, files, cj, man)
    st.advance(0, "BANKED")
    open(man, "ab").write(b" ")
    raises("gate", NC.delete_ctx, st, 0, ctx)
    # identity / T1-identity records not ok
    for kw in ({"identity_ok": False}, {"t1id_ok": False}):
        st, files, ctx, cj, man = bank_world(tmp_path / ("e" + str(len(kw)) + list(kw)[0]), **kw)
        NC.bank_chunk(st, 0, files, cj, man)
        st.advance(0, "BANKED")
        raises("gate", NC.delete_ctx, st, 0, ctx)
        assert os.path.exists(ctx)


def test_bank_refuses_an_incomplete_or_empty_chunk(tmp_path):
    st, files, ctx, cj, man = bank_world(tmp_path / "a")
    part = {k: v for k, v in files.items() if k != "scores"}
    raises("gate", NC.bank_chunk, st, 0, part, cj, man)
    open(files["search"], "wb").close()
    raises("gate", NC.bank_chunk, st, 0, files, cj, man)


# ====================================================================================================== merging and the analysis stage, on the REAL banked carve-B arrays
@needs_banked_b
def test_canonical_merge_of_shuffled_chunks_is_the_unchunked_array():
    so = {k: v for k, v in np.load(CH.BANKED_SEARCH, allow_pickle=False).items()}
    n = len(so["token"])
    rs = np.random.RandomState(7)
    perm = rs.permutation(n)
    cuts = [0, 90, 184, 289, 345, n]
    parts = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        ix = perm[a:b]
        parts.append({k: (v[ix] if k not in ("meta", "seeds") else v) for k, v in so.items()})
    m = NC.canonical_merge(parts)
    order = np.argsort(np.array([str(t) for t in so["token"]]))
    for k in so:
        if k in ("meta", "seeds"):
            continue
        assert np.asarray(m[k]).tobytes() == np.asarray(so[k])[order].tobytes(), k
    dup = [parts[0], parts[0]]
    raises("analysis", NC.canonical_merge, dup)


@needs_banked_b
def test_analysis_stage_reproduces_step_b_banked_p2head_read_exactly(tmp_path):
    """the pipeline's analysis stage (c1 rule = spc_analyse.analyse_c1 UNCHANGED) on the banked carve-B inputs, merged from SHUFFLED chunks in canonical order"""
    import analyse_stepB as AB  # noqa: F401
    b = {k: v for k, v in np.load(os.path.join(CH.SRC_BUNDLE, "bundle.npz"), allow_pickle=False).items()}
    so = {k: v for k, v in np.load(CH.BANKED_SEARCH, allow_pickle=False).items()}
    lb = {k: v for k, v in np.load(CH.BANKED_LABELS, allow_pickle=False).items()}
    rs = np.random.RandomState(3)
    perm = rs.permutation(len(so["token"]))
    cuts = [0, 90, 184, 289, 345, len(perm)]
    S = [{k: (v[perm[a:z]] if k not in ("meta", "seeds") else v) for k, v in so.items()} for a, z in zip(cuts[:-1], cuts[1:])]
    L = [{k: (v[perm[a:z]] if k not in ("sub_names", "identity", "search_sha256") else v) for k, v in lb.items()} for a, z in zip(cuts[:-1], cuts[1:])]
    ms, ml = NC.canonical_merge(S), NC.canonical_merge(L)
    ms["bundle_index"] = np.arange(len(ms["token"]), dtype=np.int64)
    ml["bundle_index"] = ms["bundle_index"]
    ix = {str(t): i for i, t in enumerate(b["token"])}
    assert [str(t) for t in ms["token"]] == [str(t) for t in b["token"]]                      # canonical order == the original bundle's (sorted) order
    for name, d in (("bundle", b), ("search", ms), ("scores", ml)):
        np.savez(tmp_path / f"{name}.npz", **d)
    out = tmp_path / "res.json"
    assert NA.main(["--bundle", str(tmp_path / "bundle.npz"), "--search", str(tmp_path / "search.npz"), "--labels", str(tmp_path / "scores.npz"), "--rule", "c1", "--out", str(out)]) == 0
    res = json.load(open(out))
    banked = json.load(open(CH.BANKED_RESULT))["arms"]["P2HEAD"]
    for k in ("endpoint_dPDMS_points", "per_seed_points", "sub_score_deltas_points", "families", "inert", "moved_share"):
        assert json.dumps(res[k], sort_keys=True) == json.dumps(banked[k], sort_keys=True), k
    e = res["endpoint_dPDMS_points"]
    assert round(e["point"], 4) == 0.3273 and round(e["lo"], 4) == 0.0565 and round(e["hi"], 4) == 0.7216
    assert res["outcome"] == "O1_CONFIRMED"


@needs_banked_b
def test_fixture_row_compare_sees_a_changed_row(tmp_path):
    ok = CH.fixture_row_compare(CH.BANKED_SEARCH, CH.BANKED_LABELS, CH.BANKED_SEARCH, CH.BANKED_LABELS)
    assert ok["ok"] and ok["n_tokens"] == 400
    so = {k: v.copy() for k, v in np.load(CH.BANKED_SEARCH, allow_pickle=False).items()}
    so["exec"][17, 1, 3, 0] = np.nextafter(so["exec"][17, 1, 3, 0], np.float32(1e9))        # ONE ulp in one search output
    np.savez(tmp_path / "s.npz", **so)
    r = CH.fixture_row_compare(str(tmp_path / "s.npz"), CH.BANKED_LABELS, CH.BANKED_SEARCH, CH.BANKED_LABELS)
    assert not r["ok"] and "exec" in r["arrays_differing"]
    lb = {k: v.copy() for k, v in np.load(CH.BANKED_LABELS, allow_pickle=False).items()}
    lb["pdms"][5, 0] += 1e-12
    np.savez(tmp_path / "l.npz", **lb)
    r = CH.fixture_row_compare(CH.BANKED_SEARCH, str(tmp_path / "l.npz"), CH.BANKED_SEARCH, CH.BANKED_LABELS)
    assert not r["ok"] and "labels.pdms" in r["arrays_differing"]


# ====================================================================================================== the C2 decision-rule terms (DQ3), each with a RED arm
def test_c2_terms_each_term_has_a_red_arm():
    good = dict(dp={"point": 0.2, "lo": 0.01, "hi": 0.4}, sr=0.1, sf={"nc": 0.0, "dac": 0.0, "ttc": -0.1, "c": 0.0}, bars={"bar_dpdms_points": 0.10, "bar_safety_points": -0.25})
    assert NA.c2_terms(good["dp"], good["sr"], good["sf"], good["bars"])["rule_met"]
    assert not NA.c2_terms({**good["dp"], "lo": 0.0}, good["sr"], good["sf"], good["bars"])["rule_met"]              # lower CI must be > 0
    assert not NA.c2_terms({**good["dp"], "point": 0.09}, good["sr"], good["sf"], good["bars"])["rule_met"]          # point >= the +0.10 bar
    assert NA.c2_terms({**good["dp"], "point": 0.10}, 0.05, good["sf"], good["bars"])["rule_met"]                    # at the bar exactly: met
    assert not NA.c2_terms(good["dp"], 0.2, good["sf"], good["bars"])["rule_met"]                                    # point must exceed the 3-seed range
    for k in ("nc", "dac", "ttc", "c"):
        assert not NA.c2_terms(good["dp"], good["sr"], {**good["sf"], k: -0.26}, good["bars"])["rule_met"], k       # each safety point >= -0.25
        assert NA.c2_terms(good["dp"], good["sr"], {**good["sf"], k: -0.25}, good["bars"])["rule_met"], k
    raises_exc = pytest.raises(KeyError)
    with raises_exc:
        NA.c2_terms(good["dp"], good["sr"], good["sf"], {})                                                              # no default bar exists
    with pytest.raises(SystemExit):
        NA.main(["--bundle", "x", "--search", "y", "--labels", "z", "--rule", "c2", "--out", "o"])                       # c2 needs the sealed bars


# ====================================================================================================== launcher, bridge, GPU stage (pure parts)
def test_search_ranges_split_on_shard_boundaries_and_never_overlap():
    assert CH.search_ranges(105, 2) == [(0, 60), (60, 105)]
    assert CH.search_ranges(105, 1) == [(0, 105)]
    assert CH.search_ranges(11, 2) == [(0, 11)]
    for n in (21, 40, 41, 90, 94, 1000):
        r = CH.search_ranges(n, 2)
        assert r[0][0] == 0 and r[-1][1] == n and all(a[1] == b[0] for a, b in zip(r, r[1:])) and all(x[0] % 20 == 0 for x in r)


def test_launcher_data_clause_comes_from_the_plan_and_the_plan_must_hash(tmp_path):
    d = planted_doc()
    plan = NC.plan_doc(d, "t" * 64, 11)
    pf = tmp_path / "plan.json"
    json.dump(plan, open(pf, "w"))
    assert NL.chunk_clause(str(pf), plan["plan_sha256"], 1) == plan["chunks"][1]["tokens_sha256"]
    with pytest.raises(SystemExit):
        NL.chunk_clause(str(pf), "0" * 64, 1)
    with pytest.raises(SystemExit):
        NL.chunk_clause(str(pf), plan["plan_sha256"], 9)
    edited = json.loads(json.dumps(plan))
    edited["chunks"][0]["tokens_sha256"] = "0" * 64                                                                      # an edited plan keeps its old plan_sha256 field
    json.dump(edited, open(pf, "w"))
    with pytest.raises(SystemExit):
        NL.chunk_clause(str(pf), plan["plan_sha256"], 0)


def test_launcher_refuses_a_head_and_any_arm_but_p2head(tmp_path):
    d = planted_doc()
    plan = NC.plan_doc(d, "t" * 64, 11)
    pf = tmp_path / "plan.json"
    json.dump(plan, open(pf, "w"))
    base = ["--plan-file", str(pf), "--plan-sha256", plan["plan_sha256"], "--chunk", "0"]
    with pytest.raises(SystemExit, match="NO head substitution"):
        NL.cmd_search(base + ["--head", "h.npz", "--arm", "P2HEAD"])
    with pytest.raises(SystemExit, match="NO head substitution"):
        NL.cmd_search(base + ["--head-sha256=abc", "--arm", "P2HEAD"])
    full = base + ["--bundle", "b", "--ctx", "c", "--ckpt", "k", "--ckpt-sha256", "s", "--refe-dir", "r", "--config", "cfg", "--config-sha256", "s", "--gate", "g", "--out-dir", "o"]
    with pytest.raises(SystemExit, match="P2HEAD only"):
        NL.cmd_search(full + ["--arm", "L2"])


def test_make_plan_shape():
    d = planted_doc()
    p = MB.make_plan(d, "D:/db", "D:/frames")
    assert set(p) == {"tokens", "token_log", "db", "token_root"} and len(p["db"]) == 4 and set(p["token_root"].values()) == {"D:/frames"} and all(v.endswith(".db") for v in p["db"].values())


def forwards(toks, banked, shift_tokens=()):
    calls = {}
    for t in toks:
        f, k = fan(t)
        if t in shift_tokens:
            f = f.copy()
            f[k, :, 0] += np.float32(0.01)
        calls[t] = (f, np.zeros((64, 6), np.float32), k)
    return calls


def test_bridge_judge_pass_and_every_red_arm():
    toks = sorted(planted_doc()["tokens"])[:3]
    banked = banked_for(toks)
    ident = {"rc": 0, "stdout": "ZZBUNDLE_OK 3 1 s", "fired": 0, "calls": forwards(toks, banked)}
    red = {"rc": 1, "stdout": "ZZBUNDLE_FAIL missing 0 bad 3", "fired": 3, "calls": forwards(toks, banked, shift_tokens=toks)}
    assert NB.judge(ident, red, toks, banked)["pass"]
    assert not NB.judge({**ident, "rc": 1}, red, toks, banked)["pass"]                                                    # identity: the builder exited non-zero
    assert not NB.judge({**ident, "calls": forwards(toks, banked, shift_tokens=toks[:1])}, red, toks, banked)["pass"]      # identity: one token differs from the bank
    assert not NB.judge(ident, {**red, "fired": 2}, toks, banked)["pass"]                                                  # RED: the perturbation never reached the forward on a token
    assert not NB.judge(ident, {**red, "rc": 0, "stdout": "ZZBUNDLE_OK 3"}, toks, banked)["pass"]                          # RED: the builder's gate stayed silent
    assert not NB.judge(ident, {**red, "calls": forwards(toks, banked)}, toks, banked)["pass"]                             # RED: a perturbed forward still equals the bank (blind comparison)
    miss = {**ident, "calls": {k: v for k, v in ident["calls"].items() if k != toks[0]}}
    assert not NB.judge(miss, red, toks, banked)["pass"]


def test_gpu_stage_stops_at_the_first_failing_step():
    class A:  # noqa: D401
        bridge_sample, chunk_sample, plan, ckpt, banked_seam, out_dir, n_expected = "b.json", "c.json", "p.json", "k.pt", "s.npz", "o", 7
    steps = GS.plan(A, py="py")
    assert [s[0] for s in steps] == ["BRIDGE", "BUNDLE"]

    class R:
        def __init__(self, rc, out):
            self.returncode, self.stdout, self.stderr = rc, out, ""
    buf = io.StringIO()
    assert GS.run(steps, runner=lambda argv: R(0, "ZZBRIDGE_PASS\n") if "navpipe_bridge.py" in " ".join(argv) else R(0, "ZZBUNDLE_OK 7 12 s\n"), out=buf) == 0 and "ZZSTAGE_DONE" in buf.getvalue()
    buf = io.StringIO()
    assert GS.run(steps, runner=lambda argv: R(0, "ZZBRIDGE_FAIL x\n"), out=buf) == 3 and "ZZSTAGE_FAIL BRIDGE" in buf.getvalue() and "BUNDLE" not in buf.getvalue().replace("ZZSTAGE_FAIL BRIDGE", "")
    buf = io.StringIO()
    assert GS.run(steps, runner=lambda argv: R(0, "ZZBRIDGE_PASS\n") if "navpipe_bridge.py" in " ".join(argv) else R(0, "ZZBUNDLE_OK 6 12 s\n"), out=buf) == 3       # wrong count
    buf = io.StringIO()
    assert GS.run(steps, runner=lambda argv: R(1, "ZZBRIDGE_PASS\n"), out=buf) == 3                                                                                  # marker present, rc != 0


# ====================================================================================================== the planted CHAIN (c2): wiring of every gate through the caller
class World:
    def __init__(self, tmp, harness_bias=0.0, break_pick=None, bad_clause=False, free=50.0, blind_control=False, bias_tokens=None):
        self.tmp = str(tmp)
        os.makedirs(self.tmp, exist_ok=True)
        self.doc = planted_doc()
        self.tokens = sorted(self.doc["tokens"])
        self.banked = banked_for(self.tokens)
        self.break_pick, self.bad_clause, self.harness_bias, self.free = break_pick, bad_clause, harness_bias, free
        self.blind_control, self.bias_tokens = blind_control, bias_tokens
        self.calls = []
        self.token_file = os.path.join(self.tmp, "tokens.json")
        json.dump(self.doc, open(self.token_file, "w"))
        self.bseam = os.path.join(self.tmp, "seam.npz")
        np.savez(self.bseam, token=np.array(self.tokens), fingerprint=np.array([self.banked["fp"][t] for t in self.tokens]), poses=np.stack([self.banked["poses"][t] for t in self.tokens]), sampling=NC.SEAM_SAMPLING,
                 arm=np.array("planted"))
        sub, p = f_scores(np.stack([self.banked["poses"][t] for t in self.tokens]))
        self.bcsv = write_csv(os.path.join(self.tmp, "banked.csv"), self.tokens, sub, p)
        self.ckpt = os.path.join(self.tmp, "ck.pt")
        open(self.ckpt, "wb").write(b"ckpt")
        self.config = os.path.join(self.tmp, "cfg.json")                 # a COPY of the real C8 file (same bytes, so its sha256 is the pinned constant); tests that write to `w.config` must never touch the tracked file
        shutil.copyfile(NC.CONFIG_C8, self.config)
        self.head = os.path.join(self.tmp, "head.npz")
        open(self.head, "wb").write(b"head")
        self.plan = NC.plan_doc(self.doc, NC.sha256_file(self.token_file), 11)
        self.harness = os.path.join(PKG, "ref", "score_navtest_refe.py")
        pins = {"token_file_sha256": NC.sha256_file(self.token_file), "chunk_plan_sha256": self.plan["plan_sha256"], "ckpt_sha256": NC.sha256_file(self.ckpt),
                "head_s0_sha256": NC.sha256_file(self.head), "banked_seam_sha256": NC.sha256_file(self.bseam), "banked_csv_sha256": NC.sha256_file(self.bcsv), "harness_script_sha256": NC.sha_lf(self.harness),
                "seam_module_sha256": NC.sha_lf(os.path.join(PKG, "ref", "refe_navtest_seam.py")), "inference_cost_status": "UNMEASURED", "bar_dpdms_points": "0.10", "bar_safety_points": "-0.25"}
        am, sha, c1, _ = make_seal(self.tmp, extra_pins=pins)
        self.am, self.sha, self.c1 = am, sha, c1
        self.ev = EVID[os.path.abspath(am)]
        self.marker = os.path.join(self.tmp, "ONE_RUN.json")

    def cfg(self, **kw):
        return CH.Cfg(os.path.join(self.tmp, "work"), mode="c2", amendment=self.am, amendment_sha256=self.sha, c1_result=self.c1.path, c1_result_sha256=self.c1.sha, cap=11, expect_tokens=(20, 4), token_file=self.token_file,
                      banked_seam=self.bseam, banked_csv=self.bcsv, ckpt=self.ckpt, head_s0=self.head, harness=self.harness, one_run_marker=self.marker, **{**self.ev.kw, "config": self.config, **kw})

    # ---- the fake children
    def runner(self, argv, env, log_path, cwd=None):
        self.calls.append(os.path.basename([a for a in argv if a.endswith(".py")][-1]) if any(a.endswith(".py") for a in argv) else argv[0])
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        joined = " ".join(argv)
        out = io.StringIO()
        rc = 0
        if "navpipe_gpu_stage.py" in joined:
            rc = self.gpu(argv, out)
        elif "navpipe_launch.py" in joined and " gate " in joined:
            rc = self.gate(argv)
        elif "navpipe_launch.py" in joined and " search " in joined:
            rc = self.search(argv)
        elif "navpipe_launch.py" in joined and " merge " in joined:
            import stepB
            with contextlib.redirect_stdout(out):
                rc = stepB.cmd_merge(argv[argv.index("merge") + 1:])
        elif "navpipe_analyse.py" in joined:
            with contextlib.redirect_stdout(out):
                rc = NA.main(argv[[k for k, a in enumerate(argv) if a.endswith("navpipe_analyse.py")][0] + 1:])
        else:
            raise AssertionError("unexpected child: " + joined[:200])
        open(log_path, "a", encoding="utf-8").write(out.getvalue())
        return rc

    def arg(self, argv, k):
        return argv[argv.index(k) + 1]

    def gpu(self, argv, out):
        sample = json.load(open(self.arg(argv, "--chunk-sample")))
        toks = sorted(sample["tokens"])
        d = os.path.join(self.arg(argv, "--out-dir"), "bundle")
        os.makedirs(d, exist_ok=True)
        raw, pick = planted_bundle(toks)
        if self.break_pick in toks:
            pick[toks.index(self.break_pick)] = (pick[toks.index(self.break_pick)] + 1) % 64
        n = len(toks)
        np.savez(os.path.join(d, "bundle.npz"), token=np.array(toks), log=np.array([self.doc["token_log"][t] for t in toks]), traj_raw=raw, logits=np.zeros((n, 64, 6), np.float32), pick=pick)
        np.save(os.path.join(d, "ctx.npy"), np.zeros((n, 4, 4), np.uint16))
        json.dump({"n": n, "ctx_file": "ctx.npy", "ctx_sha256": NC.sha256_file(os.path.join(d, "ctx.npy")), "tokens_sha256": NC.tokens_sha256(toks)}, open(os.path.join(d, "ctx.json"), "w"))
        out.write(f"ZZBUNDLE_OK {n} 1 s\nZZSTAGE_DONE\n")
        return 0

    def gate(self, argv):
        d = self.arg(argv, "--out-dir")
        os.makedirs(d, exist_ok=True)
        json.dump({"schema": "tanitad.g_reach_stepb/1", "verdict": "PASS", "bundle_sha256": NC.sha256_file(self.arg(argv, "--bundle")), "ckpt_sha256": self.arg(argv, "--ckpt-sha256"),
                   "config_sha256": self.arg(argv, "--config-sha256")}, open(os.path.join(d, "G-REACH-STEPB_PASS_x.json"), "w"))
        return 0

    def search(self, argv):
        b = np.load(self.arg(argv, "--bundle"), allow_pickle=False)
        lo, hi = int(self.arg(argv, "--start")), int(self.arg(argv, "--end"))
        toks = [str(t) for t in b["token"]]
        n = hi - lo
        S = 3
        rs = np.random.RandomState(11)
        pick = np.asarray(b["pick"], np.int64)[lo:hi]
        pex = np.stack([NC.repair_last_heading(b["traj_raw"][i, pick[i - lo]]) for i in range(lo, hi)])
        ex = np.repeat(pex[:, None], S, 1).astype(np.float32)
        moved = np.zeros((n, S), bool)
        for i in range(n):
            h = int(toks[lo + i][:6], 16)
            for s in range(S):
                if (h + s) % 3 == 0:
                    ex[i, s, :, 1] += np.float32(0.3 * (s + 1))
                    moved[i, s] = True
        clause = hashlib.sha256("\n".join(sorted(toks)).encode()).hexdigest()
        meta = {"arm": "P2HEAD", "head_sha256": None, "ckpt_sha256": self.arg(argv, "--ckpt-sha256"), "config_sha256": self.arg(argv, "--config-sha256"), "bundle_sha256": NC.sha256_file(self.arg(argv, "--bundle")),
                "data_clause": {"carve_sha256": ("0" * 64 if self.bad_clause else clause)}}
        d = self.arg(argv, "--out-dir")
        os.makedirs(d, exist_ok=True)
        np.savez(os.path.join(d, f"shard_P2HEAD_{lo:04d}.npz"), meta=np.array(json.dumps(meta)), token=np.array(toks[lo:hi]), bundle_index=np.arange(lo, hi, dtype=np.int64), pick=pick, seeds=np.array([1, 2, 3]),
                 exec=ex, scorer_in=ex, theta=np.zeros((n, S, 2)), moved=moved, hist_best_agg=np.zeros((n, S, 5)), hist_mu=np.zeros((n, S, 5, 2)), pred_logit_out=np.zeros((n, S, 6)),
                 pred_logit_pick=np.zeros((n, 6)), arm_argmax=pick.copy(), pick_exec=pex.astype(np.float32), t_search_s=np.ones((n, S)), t_token_s=np.ones(n))
        return 0

    def harness_fn(self, label, seam_path, tokens_file):
        z = np.load(seam_path)
        toks = [str(t) for t in z["token"]]
        sub, p = f_scores(z["poses"])
        sub = sub.copy()
        if "_pick" in label and self.harness_bias:                         # a route whose re-scored picks do not reproduce the banked CSV (bias on all tokens, or only on `bias_tokens`)
            hit = np.array([self.bias_tokens is None or t in self.bias_tokens for t in toks])
            p = p + self.harness_bias * hit
            sub[:, 2] += self.harness_bias * hit
        if "_ctl" in label and self.blind_control:                         # a route that cannot see a 0.5 m shift
            sub, p = f_scores(z["poses"] - np.array([0.5, 0.0, 0.0], np.float32))
        path = os.path.join(self.tmp, "h_" + label + ".csv")
        write_csv(path, toks, sub, nc_formula(sub))
        self.labels = getattr(self, "labels", []) + [label]
        return path

    def chain(self, **kw):
        cfg = self.cfg(**{k: v for k, v in kw.items() if k in ("procs", "prefetch", "max_chunks")})
        return CH.Chain(cfg, runner=self.runner, out=io.StringIO(), ram_fn=lambda: 99.0, free_fn=lambda p: self.free, harness_fn=self.harness_fn)


def nc_formula(sub):
    return sub[:, 0] * sub[:, 1] * (5 * sub[:, 2] + 5 * sub[:, 3] + 2 * sub[:, 4]) / 12.0


def test_chain_happy_path_deletes_every_ctx_only_after_banking_and_equals_the_unchunked_analysis(tmp_path):
    w = World(tmp_path)
    ch = w.chain(procs=2)
    assert ch.run() == 0, ch.out.getvalue()
    out = ch.out.getvalue()
    for m in ("SEAL_OK", "DATA_OK", "DISK_OK", "BUNDLE_OK 0", "SEARCH_DONE 0", "SCORE_DONE 0", "BANKED 0", "CHUNK_DONE 0", "CHUNK_DONE 1", "ANALYSIS_DONE", "DONE", "LICENCE_OK O1_CONFIRMED", "LICENCE_BASIS"):
        assert "ZZNAVPIPE_" + m in out, m
    st = NC.State(os.path.join(w.tmp, "work", "navpipe_state.json"), w.plan["plan_sha256"])
    for cid in (0, 1):
        assert st.status(cid) == "DELETED"
        assert not os.path.exists(os.path.join(w.tmp, "work", "chunks", f"c{cid:02d}", "bundle", "ctx.npy"))
        assert os.path.exists(os.path.join(w.tmp, "work", "chunks", f"c{cid:02d}", "bundle", "bundle.npz"))
        assert NC.verify_banked(st, cid) == []
    assert w.calls.count("navpipe_gpu_stage.py") == 2
    res = json.load(open(os.path.join(w.tmp, "work", "final", "RESULT_NAVPIPE.json")))
    assert res["read"] == "C2" and res["n_tokens"] == 20 and res["decision_rule_terms"]["bar_dpdms_points"] == 0.10 and "outcome_names" in res
    # chunked == unchunked: the same analysis on ONE global bundle in sorted order
    import analyse_stepB as AB
    toks = w.tokens
    raw, pick = planted_bundle(toks)
    sub_b, p_b = f_scores(np.stack([w.banked["poses"][t] for t in toks]))
    gb = {"token": np.array(toks), "log": np.array([w.doc["token_log"][t] for t in toks]), "pick": pick, "sub": np.zeros((20, 64, 6)), "pdms": np.zeros((20, 64))}
    gb["sub"][np.arange(20), pick], gb["pdms"][np.arange(20), pick] = sub_b, p_b
    shards = []
    argv = ["x"]
    sdir = os.path.join(w.tmp, "global")
    os.makedirs(sdir)
    np.savez(os.path.join(sdir, "bundle.npz"), token=gb["token"], log=gb["log"], traj_raw=raw, pick=pick, logits=np.zeros((20, 64, 6), np.float32))
    w.bad_clause = False
    w.search(["--bundle", os.path.join(sdir, "bundle.npz"), "--start", "0", "--end", "20", "--ckpt-sha256", NC.sha256_file(w.ckpt), "--config-sha256", NC.sha_lf(w.config), "--out-dir", sdir])
    so = {k: v for k, v in np.load(os.path.join(sdir, "shard_P2HEAD_0000.npz"), allow_pickle=False).items()}
    exp = np.stack([f_scores(NC.exec_to_poses(so["exec"][:, s])) [1] for s in range(3)], 1)
    subs = np.stack([f_scores(NC.exec_to_poses(so["exec"][:, s]))[0] for s in range(3)], 1)
    lab = {"token": so["token"], "pdms": exp, "sub": subs}
    ref = NA.analyse_c2(gb, so, lab, {"bar_dpdms_points": 0.10, "bar_safety_points": -0.25})
    for k in ("endpoint_dPDMS_points", "per_seed_points", "sub_score_deltas_points", "families", "inert", "moved_share"):
        assert json.dumps(res[k], sort_keys=True) == json.dumps(ref[k], sort_keys=True, default=str), k


def test_chain_red_identity_stops_and_keeps_the_ctx(tmp_path):
    w = World(tmp_path, break_pick=None)
    toks = NC.chunk_tokens(w.doc, w.plan["chunks"][1])
    w.break_pick = toks[2]
    ch = w.chain()
    assert ch.run() == 3
    out = ch.out.getvalue()
    assert "ZZNAVPIPE_STOP G_IDENT_NOT_PASSED" in out and "CHUNK_DONE 0" in out and "CHUNK_DONE 1" not in out and "ANALYSIS_DONE" not in out
    st = NC.State(os.path.join(w.tmp, "work", "navpipe_state.json"), w.plan["plan_sha256"])
    assert st.status(0) == "DELETED" and st.status(1) == "NEW"
    assert os.path.exists(os.path.join(w.tmp, "work", "chunks", "c01", "bundle", "ctx.npy"))        # the evidence of the failure is kept


def test_child_crash_is_retried_but_a_real_failure_is_not(tmp_path):
    seq = iter([3221225477, 0])
    ch = CH.Chain(CH.Cfg(str(tmp_path / "w")), runner=lambda *a, **k: next(seq), out=io.StringIO())
    ch.retry_sleep_s = 0
    assert ch.child(["x"], "l.log") == 0 and "CHILD_CRASH_RETRY" in ch.out.getvalue()
    seq2 = iter([1, 0])
    ch2 = CH.Chain(CH.Cfg(str(tmp_path / "w2")), runner=lambda *a, **k: next(seq2), out=io.StringIO())
    ch2.retry_sleep_s = 0
    assert ch2.child(["x"], "l.log") == 1                                                       # rc 1 is a failure, never retried
    always = iter([3221225477] * 5)
    ch3 = CH.Chain(CH.Cfg(str(tmp_path / "w3")), runner=lambda *a, **k: next(always), out=io.StringIO())
    ch3.retry_sleep_s = 0
    assert ch3.child(["x"], "l.log") == 3221225477                                              # bounded: 3 attempts, then the failure surfaces


def test_chain_red_disk_guard_stops_before_any_child(tmp_path):
    w = World(tmp_path, free=5.0)
    ch = w.chain()
    assert ch.run() == 3 and "ZZNAVPIPE_STOP DISK_LOW" in ch.out.getvalue() and not w.calls


def test_chain_red_disk_guard_counts_the_chunks_own_ctx(tmp_path):
    w = World(tmp_path, free=6.0 + 0.5 * 11 * NC.CTX_BYTES_PER_TOKEN / 2 ** 30)                     # enough for half the chunk's ctx only
    ch = w.chain()
    assert ch.run() == 3 and "DISK_LOW" in ch.out.getvalue() and "navpipe_gpu_stage.py" not in w.calls


def test_chain_red_pick_rescore_that_differs_from_the_banked_csv_stops_and_keeps_the_ctx(tmp_path):
    w = World(tmp_path, harness_bias=1e-3)                                    # the route's re-scored picks are 1e-3 off P2's banked rows
    ch = w.chain()
    out = ""
    assert ch.run() == 7
    out = ch.out.getvalue()
    assert "ZZNAVPIPE_PICK_RESCORE_MISMATCH 0" in out and "ZZNAVPIPE_STOP G_PICK_RESCORE_DIFFERS_FROM_BANKED" in out and "SCORE_DONE" not in out
    st = NC.State(os.path.join(w.tmp, "work", "navpipe_state.json"), w.plan["plan_sha256"])
    assert st.status(0) == "SEARCHED" and os.path.exists(os.path.join(w.tmp, "work", "chunks", "c00", "bundle", "ctx.npy"))


def test_chain_red_wrong_data_clause_in_the_search_meta(tmp_path):
    w = World(tmp_path, bad_clause=True)
    ch = w.chain()
    assert ch.run() == 6 and "SEARCH_MERGE_INVALID" in ch.out.getvalue()


def test_chain_is_resumable_and_does_not_redo_finished_chunks(tmp_path):
    w = World(tmp_path)
    ch = w.chain(max_chunks=1)
    assert ch.run() == 0 and "PARTIAL" in ch.out.getvalue()
    assert w.calls.count("navpipe_gpu_stage.py") == 1
    ch2 = w.chain()
    assert ch2.run() == 0
    assert w.calls.count("navpipe_gpu_stage.py") == 2                                           # chunk 0 was not rebuilt
    assert "CHUNK_DONE 1" in ch2.out.getvalue() and "CHUNK_DONE 0" not in ch2.out.getvalue()


def test_chain_prefetch_gives_the_same_result_and_degrades_when_disk_is_tight(tmp_path):
    a = World(tmp_path / "a")
    ca = a.chain(prefetch=1)
    assert ca.run() == 0
    b = World(tmp_path / "b")
    cb = b.chain(prefetch=0)
    assert cb.run() == 0
    ra = json.load(open(os.path.join(a.tmp, "work", "final", "RESULT_NAVPIPE.json")))
    rb = json.load(open(os.path.join(b.tmp, "work", "final", "RESULT_NAVPIPE.json")))
    assert ra["endpoint_dPDMS_points"] == rb["endpoint_dPDMS_points"]
    t = World(tmp_path / "c", free=6.0 + 1.5 * 11 * NC.CTX_BYTES_PER_TOKEN / 2 ** 30)           # room for one ctx, not two: prefetch must fall back to sequential, not crash
    ct = t.chain(prefetch=1)
    assert ct.run() == 0


@needs_real
def test_fixture_mode_refuses_a_navtest_token(tmp_path, monkeypatch):
    nav = json.load(open(CH.NAVTEST_TOKENS, encoding="utf-8"))["tokens"][:3]
    cb = tmp_path / "carve.json"
    json.dump({"tokens": nav}, open(cb, "w"))
    monkeypatch.setattr(CH, "CARVE_B", str(cb))
    monkeypatch.setattr(CH, "CARVE_B_SHA", NC.tokens_sha256(sorted(nav)))
    ch = CH.Chain(CH.Cfg(str(tmp_path / "w"), mode="fixture"), runner=lambda *a, **k: 0, out=io.StringIO())
    assert ch.run() == 5 and "FIXTURE_REFUSED_NAVTEST_TOKEN" in ch.out.getvalue()


def test_harness_pin_and_operative_files_are_pinned():
    assert NC.sha_lf(CH.HARNESS) == CH.HARNESS_SHA256_LF
    for f in NC.OPERATIVE:
        assert os.path.isfile(os.path.join(HERE, f)), f
