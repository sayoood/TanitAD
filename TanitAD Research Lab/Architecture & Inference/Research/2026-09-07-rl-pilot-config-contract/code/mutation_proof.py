#!/usr/bin/env python3
"""MUTATION PROOF for the rl_pilot_refc21 CONFIG CONTRACT (2026-09-07).

⛔ EVERY EXPECTATION BELOW IS HARD-CODED. Not one case asks "did the code refuse
when there was a mismatch" — that is an expected value of whatever the code
does, and it stays green while the code is wrong. Each case states, as a
literal, the case name, the verdict, the counts and the substrings the operator
must be able to read, and fails if the code produces anything else.

⭐ THE POINT OF T0. It is not enough to show the new guard fires; the claim is
that the TWO GUARDS ALREADY THERE CANNOT fire. T0 measures that directly.

⛔ ZERO GPU: every model is built and every checkpoint is loaded on CPU.
"""
from __future__ import annotations

import contextlib
import dataclasses as dc
import importlib.util
import io
import json
import os
import sys
import traceback

os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO = r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD"
PILOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    REPO, "stack", "scripts", "rl_pilot_refc21.py")
SCRATCH = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(SCRATCH, "mutwork")
REAL_CKPT = r"C:/Users/Admin/tanitad-data/models/refc-base-30k/ckpt.pt"

sys.path.insert(0, os.path.join(REPO, "stack"))
import torch                                                   # noqa: E402
from tanitad.refs import refc                                  # noqa: E402
from tanitad.rl import PostTrainConfig                         # noqa: E402

spec = importlib.util.spec_from_file_location("pilot_under_test", PILOT)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)

RESULTS: list[dict] = []


def check(case, label, got, want):
    ok = got == want
    RESULTS[-1]["checks"].append(
        {"label": label, "want": want, "got": got, "pass": ok})
    print(f"    [{'PASS' if ok else 'FAIL'}] {label}: want {want!r} got {got!r}")
    return ok


def check_in(case, label, needle, hay):
    ok = needle in hay
    RESULTS[-1]["checks"].append(
        {"label": label, "want_substring": needle, "pass": ok})
    print(f"    [{'PASS' if ok else 'FAIL'}] {label}: substring {needle!r}")
    return ok


def begin(name, what):
    print(f"\n=== {name} — {what}")
    RESULTS.append({"case": name, "what": what, "checks": []})


def write_fixture(path, sd, cfg_obj=None, step=1234):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    blob = {"model": sd, "step": step}
    if cfg_obj is not None:
        blob["cfg"] = cfg_obj
    torch.save(blob, path)


def run_load(ckpt, out):
    """(returned_or_None, exception_or_None, stdout)."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            m = pilot.load_model(ckpt, "cpu", out)
        return m, None, buf.getvalue()
    except BaseException as exc:                       # SystemExit included
        return None, exc, buf.getvalue()


def stamp_of(out):
    with open(os.path.join(out, "config_contract.json"), encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# T0 — THE CONTROL: the two existing guards are blind to `v0_conditioned`      #
# --------------------------------------------------------------------------- #
begin("T0", "EXPECT_PARAMS and strict=True cannot see anchors.v0_conditioned")
m_false = refc.RefCModel(refc.refc_config())
_c = refc.refc_config()
_c.anchors.v0_conditioned = True
m_true = refc.RefCModel(_c)
p_false = sum(p.numel() for p in m_false.parameters())
p_true = sum(p.numel() for p in m_true.parameters())
check("T0", "param count with v0_conditioned=False", p_false, 104_191_577)
check("T0", "param count with v0_conditioned=True", p_true, 104_191_577)
check("T0", "state_dict key list identical (names AND order)",
      list(m_false.state_dict()) == list(m_true.state_dict()), True)
check("T0", "n state_dict entries", len(m_false.state_dict()), 488)
check("T0", "every tensor shape identical",
      {k: tuple(v.shape) for k, v in m_false.state_dict().items()}
      == {k: tuple(v.shape) for k, v in m_true.state_dict().items()}, True)
try:
    m_true.load_state_dict(m_false.state_dict(), strict=True)
    xload = "OK"
except Exception as exc:                                          # noqa: BLE001
    xload = f"{type(exc).__name__}"
check("T0", "strict=True cross-load False-weights -> True-model", xload, "OK")
SD = m_false.state_dict()
del m_true, m_false, _c

# ⛔ NO ORPHANS. A consequence written for a field that does not exist is a
# consequence that can never fire, and it would read in review as coverage the
# guard does not have. Every key must be a real leaf of the built config.
_leaves = pilot._cfg_leaves(refc.refc_config())
check("T0", "consequence entries recorded", len(pilot.CFG_CONSEQUENCE), 34)
check("T0", "consequence keys that are not real config leaves",
      [k for k in pilot.CFG_CONSEQUENCE if k not in _leaves], [])
check("T0", "config leaves in total", len(_leaves), 96)

FULL_CFG = dc.asdict(refc.refc_config())

# --------------------------------------------------------------------------- #
# T1 — DISAGREE must REFUSE, and the message must be operator-legible          #
# --------------------------------------------------------------------------- #
begin("T1", "ckpt cfg says v0_conditioned=True, run would use False -> REFUSE")
cfg_bad = json.loads(json.dumps(FULL_CFG))
cfg_bad["anchors"]["v0_conditioned"] = True
fx = os.path.join(WORK, "t1", "ckpt.pt")
out = os.path.join(WORK, "t1", "out")
write_fixture(fx, SD, cfg_bad)
mdl, exc, sout = run_load(fx, out)
check("T1", "model returned", mdl is None, True)
check("T1", "exception type", type(exc).__name__, "SystemExit")
msg = str(exc)
check_in("T1", "message says the contract refused",
         "CONFIG CONTRACT REFUSED", msg)
check_in("T1", "message names the field", "anchors.v0_conditioned", msg)
check_in("T1", "message gives the checkpoint's value",
         "checkpoint says True", msg)
check_in("T1", "message gives the value this run would use",
         "this run would use False", msg)
check_in("T1", "message says the shape guards cannot see it",
         "changes no tensor name and no parameter count", msg)
check_in("T1", "consequence cites the 10 m/s reference roll",
         "offered at the 10 m/s reference INSTEAD OF the window's own measured "
         "speed", msg)
check_in("T1", "consequence cites the MEASURED vocabulary gap", "0.3773", msg)
check_in("T1", "consequence cites the v0-conditioned figure", "0.2610", msg)
check_in("T1", "consequence names the real hazard in words",
         "ACTION SPACE THE CHECKPOINT WAS NEVER TRAINED IN", msg)
st = stamp_of(out)
check("T1", "stamp case", st["case"], "DISAGREE")
check("T1", "stamp verdict", st["verdict"], "REFUSED")
check("T1", "stamp disagreement fields", sorted(st["disagreements"]),
      ["anchors.v0_conditioned"])
check("T1", "stamp records ckpt value", st["disagreements"]
      ["anchors.v0_conditioned"]["ckpt"], True)
check("T1", "stamp records built value", st["disagreements"]
      ["anchors.v0_conditioned"]["built"], False)
os.remove(fx)

# --------------------------------------------------------------------------- #
# T2 — AGREE must PASS. Not optional: a guard that refuses everything gets     #
#      deleted rather than fixed.                                             #
# --------------------------------------------------------------------------- #
begin("T2", "ckpt cfg identical to the rebuilt config -> PASS, and STAMPED")
fx = os.path.join(WORK, "t2", "ckpt.pt")
out = os.path.join(WORK, "t2", "out")
write_fixture(fx, SD, json.loads(json.dumps(FULL_CFG)))
mdl, exc, sout = run_load(fx, out)
check("T2", "no exception", None if exc is None else type(exc).__name__, None)
check("T2", "a model was returned", mdl is not None, True)
st = stamp_of(out)
check("T2", "stamp case", st["case"], "AGREE")
check("T2", "stamp verdict", st["verdict"], "PROCEEDING")
check("T2", "fields compared", len(st["compared"]), 96)
check("T2", "disagreements", st["disagreements"], {})
check("T2", "nothing left assumed", st["assumed_unverified"], {})
check("T2", "v0_conditioned WAS among the compared fields",
      st["compared"]["anchors.v0_conditioned"], False)
check("T2", "provenance", st["cfg_provenance"], "ckpt['cfg']")
check_in("T2", "stdout announces the AGREE case",
         "CONFIG CONTRACT: case AGREE", sout)
check_in("T2", "stdout reports 96/96 compared",
         "96/96 config fields COMPARED and AGREEING", sout)
check_in("T2", "the weights still loaded", "cold start loaded: 104,191,577", sout)
del mdl
os.remove(fx)

# --------------------------------------------------------------------------- #
# T3 — ABSENT must PROCEED **WITH THE STAMP PRESENT**                          #
# --------------------------------------------------------------------------- #
begin("T3", "ckpt with no config at all -> proceed, assumption STAMPED")
fx = os.path.join(WORK, "t3", "ckpt.pt")
out = os.path.join(WORK, "t3", "out")
write_fixture(fx, SD, None)
mdl, exc, sout = run_load(fx, out)
check("T3", "no exception", None if exc is None else type(exc).__name__, None)
check("T3", "a model was returned", mdl is not None, True)
check("T3", "the stamp FILE exists",
      os.path.exists(os.path.join(out, "config_contract.json")), True)
st = stamp_of(out)
check("T3", "stamp case", st["case"], "ABSENT")
check("T3", "stamp verdict", st["verdict"],
      "PROCEEDING ON UNVERIFIED DEFAULTS")
check("T3", "nothing was compared", st["compared"], {})
check("T3", "every field is recorded as assumed",
      len(st["assumed_unverified"]), 96)
check("T3", "the assumed v0_conditioned value is written down",
      st["assumed_unverified_behaviour_critical"]["anchors.v0_conditioned"],
      False)
check("T3", "the assumed reference speed is written down",
      st["assumed_unverified_behaviour_critical"]["anchors.ref_speed_ms"], 10.0)
check("T3", "n behaviour-critical assumptions recorded",
      len(st["assumed_unverified_behaviour_critical"]), 34)
check_in("T3", "stdout announces the ABSENT case",
         "CONFIG CONTRACT: case ABSENT", sout)
check_in("T3", "stdout states the assumption explicitly",
         "ASSUMED anchors.v0_conditioned = False  (UNVERIFIED)", sout)
check_in("T3", "stdout warns what an unverified assumption costs",
         "post-trains on a model configuration it never confirmed", sout)
check("T3", "weight evidence corroborates the assumption",
      st["v0_conditioned_weight_evidence"]["verdict"], "CORROBORATED")
check("T3", "weight evidence state", st["v0_conditioned_weight_evidence"]
      ["state"], "ALL_ZERO")
del mdl
os.remove(fx)

# --------------------------------------------------------------------------- #
# T4 — a `cfg` key that is NOT a model config must NOT be mistaken for one     #
# --------------------------------------------------------------------------- #
begin("T4", "ckpt['cfg'] is a PostTrainConfig (the pilot's OWN ckpt_after.pt "
            "shape) -> treated as ABSENT, with the reason recorded")
ptc = PostTrainConfig(method="grpo", group_size=4, steps=1, batch=1, lr=1e-5,
                      seed=0, dt=0.5, decoder_steps=2).to_dict()
fx = os.path.join(WORK, "t4", "ckpt.pt")
out = os.path.join(WORK, "t4", "out")
write_fixture(fx, SD, ptc)
mdl, exc, sout = run_load(fx, out)
check("T4", "no exception", None if exc is None else type(exc).__name__, None)
st = stamp_of(out)
check("T4", "stamp case", st["case"], "ABSENT")
check("T4", "provenance is None, not the RL config",
      st["cfg_provenance"], None)
check("T4", "one source was rejected", len(st["sources_rejected"]), 1)
check_in("T4", "the rejection names the source",
         "ckpt['cfg']", st["sources_rejected"][0])
check_in("T4", "the rejection says why",
         "NONE of them RefCConfig fields", st["sources_rejected"][0])
del mdl
os.remove(fx)

# --------------------------------------------------------------------------- #
# T5 — the WEIGHTS may contradict an assumed value; that must refuse           #
# --------------------------------------------------------------------------- #
begin("T5", "run would use v0_conditioned=True but the weights carry an "
            "all-zero control vocabulary -> REFUSE")
fx = os.path.join(WORK, "t5", "ckpt.pt")
out = os.path.join(WORK, "t5", "out")
write_fixture(fx, SD, None)
cfg_true = refc.refc_config()
cfg_true.anchors.v0_conditioned = True
ck5 = torch.load(fx, map_location="cpu", weights_only=False)
buf = io.StringIO()
exc5 = None
try:
    with contextlib.redirect_stdout(buf):
        pilot.assert_config_contract(cfg_true, ck5, fx, out)
except BaseException as e:                                        # noqa: BLE001
    exc5 = e
check("T5", "exception type", type(exc5).__name__, "SystemExit")
m5 = str(exc5)
check_in("T5", "message refuses", "CONFIG CONTRACT REFUSED", m5)
check_in("T5", "message says the weights are the evidence",
         "the CHECKPOINT'S OWN WEIGHTS say that is impossible", m5)
check_in("T5", "message says every anchor collapses",
         "EVERY anchor collapses to 'do nothing'", m5)
check_in("T5", "message cites refc.py's own words",
         "a plausible-looking WRONG experiment", m5)
st = stamp_of(out)
check("T5", "stamp verdict", st["verdict"], "REFUSED (v0 weight evidence)")
check("T5", "weight evidence verdict",
      st["v0_conditioned_weight_evidence"]["verdict"], "CONTRADICTED")
del ck5
os.remove(fx)

# --------------------------------------------------------------------------- #
# T6 — THE LIVE PATH: the pilot's actual cold start                            #
# --------------------------------------------------------------------------- #
begin("T6", "the REAL cold start refc-diffusion-base-v21-30k, loaded and looked at")
out = os.path.join(WORK, "t6", "out")
mdl, exc, sout = run_load(REAL_CKPT, out)
st = stamp_of(out)
check("T6", "stamp case", st["case"], "PARTIAL")
check("T6", "stamp verdict", st["verdict"], "PROCEEDING")
check("T6", "provenance is the sidecar next to the ckpt",
      st["cfg_provenance"].startswith("sidecar "), True)
check("T6", "checkpoint step", st["ckpt_step"], 29999)
check("T6", "fields the checkpoint DOES carry, all agreeing",
      len(st["compared"]), 38)
check("T6", "disagreements", st["disagreements"], {})
check("T6", "fields the checkpoint does NOT carry -> assumed",
      len(st["assumed_unverified"]), 58)
check("T6", "v0_conditioned is one of the ASSUMED, not the compared",
      "anchors.v0_conditioned" in st["assumed_unverified"], True)
check("T6", "and it is assumed False",
      st["assumed_unverified"]["anchors.v0_conditioned"], False)
check("T6", "weight evidence for that assumption",
      st["v0_conditioned_weight_evidence"]["verdict"], "CORROBORATED")
check("T6", "weight evidence state", st["v0_conditioned_weight_evidence"]
      ["state"], "BUFFER_ABSENT")
check_in("T6", "stdout states the assumption",
         "ASSUMED anchors.v0_conditioned = False  (UNVERIFIED)", sout)
# ⛔ SEPARATE, PRE-EXISTING DEFECT — recorded, not fixed here.
check("T6", "the run then dies in the PRE-EXISTING strict load",
      type(exc).__name__ if exc else None, "RuntimeError")
check_in("T6", "and it dies on the buffer this ckpt predates",
         'Missing key(s) in state_dict: "decoder.anchor_controls"',
         str(exc) if exc else "")

# --------------------------------------------------------------------------- #
n_pass = sum(1 for r in RESULTS for c in r["checks"] if c["pass"])
n_all = sum(len(r["checks"]) for r in RESULTS)
print(f"\n================ {n_pass}/{n_all} checks PASS ================")
for r in RESULTS:
    bad = [c["label"] for c in r["checks"] if not c["pass"]]
    print(f"  {r['case']}: {len(r['checks']) - len(bad)}/{len(r['checks'])}"
          + (f"   FAILED: {bad}" if bad else ""))
with open(os.path.join(SCRATCH, "mutation_proof.json"), "w",
          encoding="utf-8") as fh:
    json.dump({"n_pass": n_pass, "n_checks": n_all, "cases": RESULTS}, fh,
              indent=1, default=str)
sys.exit(0 if n_pass == n_all else 1)
