#!/usr/bin/env python
"""ckpt120_reach.py -- K-REACH-e of PREREG_V8F_CKPT120: the POST-TRAINING reachability gate for the new parameters.

Study wrapper over ``zerograd_gate.analyze`` (Adam-state method: a parameter with NO optimizer state never received a gradient; one whose state is exactly zero only ever received exact-zero gradients).
``zerograd_gate.py`` itself is untouched. This wrapper SCOPES the verdict to the CKPT120 parameters: every ``s_far_out.*`` key and (for an arm with nav_s) every ``e_nav_s.*`` key of the checkpoint's model state dict must be TRAINED.
A whole-model FAIL on some other deliberately frozen parameter is not this gate's business; the new heads are.

VERDICTS (a missing or unreadable result is never a PASS)
  PASS          every expected key is uniquely named, TRAINED, and both controls read as required
  HEAD-DEAD     an expected key has no optimizer state (case a) or only exact-zero state (case b), or is in the offender rows: build defect, nothing is read   (exit 1)
  INCONCLUSIVE  an expected key cannot be uniquely named (shape alignment ambiguous), the checkpoint has no optimizer state, or a control could not be evaluated   (exit 2)

CONTROLS (the same breath, in the JSON)
  positive   >= 1 uniquely named TRAINED parameter matches ``--control`` (default ``head_s\\..*``: S's own head, which the far head sits on): excludes a probe that reads nothing.
  negative   the checkpoint is MUTATED IN MEMORY: the optimizer-state entry of the id named ``s_far_out.weight`` (or ``e_nav_s.weight``) is deleted and the gate re-run; it MUST read HEAD-DEAD. A gate that cannot see a
             planted dead head is not evidence about the real one.
  expected   ``--expect-nav-s`` says whether the arm has nav_s: ``e_nav_s.*`` keys must exist iff it does (an arm that lost its token is not a PASS).

usage: python ckpt120_reach.py --ckpt CKPT --out reach.json [--expect-nav-s on|off] [--control REGEX] [--param-names FILE.json]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import zerograd_gate as Z  # noqa: E402

SCHEMA = "ckpt120-reach-gate-v1"
PREFIXES = {"far": "s_far_out.", "nav_s": "e_nav_s."}
DEFAULT_CONTROL = r"head_s\..*"
EXIT = {"PASS": 0, "HEAD-DEAD": 1, "INCONCLUSIVE": 2}


def expected_keys(sd, expect_nav_s):
    """({kind: [keys]}, problems): the CKPT120 keys of the model state dict; the far head must exist; e_nav_s exists iff the arm has nav_s."""
    out, bad = {}, []
    out["far"] = sorted(k for k in sd if k.startswith(PREFIXES["far"]))
    out["nav_s"] = sorted(k for k in sd if k.startswith(PREFIXES["nav_s"]))
    if not out["far"]:
        bad.append("no s_far_out.* key in the checkpoint's model state dict")
    if expect_nav_s and not out["nav_s"]:
        bad.append("--expect-nav-s on but the checkpoint has no e_nav_s.* key")
    if not expect_nav_s and out["nav_s"]:
        bad.append("--expect-nav-s off but the checkpoint carries e_nav_s.* keys")
    return out, bad


def analyze(ck, expect_nav_s=False, control=DEFAULT_CONTROL, param_names=None, _mutate=None):
    """The scoped verdict of one loaded checkpoint dict. ``_mutate`` (the negative control) is a model key whose optimizer-state entry is deleted first."""
    import torch  # noqa: F401
    mk, sd = Z._pick(ck, Z._MODEL_KEYS)
    if sd is None:
        return {"verdict": "INCONCLUSIVE", "reasons": ["no model state dict in the checkpoint"]}
    keys, problems = expected_keys(sd, expect_nav_s)
    rep = {"schema": SCHEMA, "reasons": list(problems), "expected": {k: v for k, v in keys.items()}}
    if problems and not keys["far"]:
        rep["verdict"] = "INCONCLUSIVE"
        return rep
    want = keys["far"] + keys["nav_s"]
    work = ck
    if _mutate is not None:
        work = copy.copy(ck)
        ok, opt = Z._pick(ck, Z._OPT_KEYS)
        opt2 = copy.copy(opt)
        opt2["state"] = dict(opt["state"])
        victim = None
        for i, nm in _names_by_id(ck, param_names).items():       # the id whose unique name is the target: its state entry is deleted
            if nm == [_mutate] and i in opt["state"]:
                victim = i
                break
        if victim is None:
            return {**rep, "verdict": "INCONCLUSIVE", "reasons": rep["reasons"] + [f"negative control: could not locate the optimizer id of {_mutate} to delete"]}
        opt2["state"] = {k: v for k, v in opt["state"].items() if k != victim}
        work[ok] = opt2
        rep["mutated_key"] = _mutate
    try:
        z = Z.analyze(work, controls=[], param_names=param_names)
    except Z.GateError as e:
        return {**rep, "verdict": "INCONCLUSIVE", "reasons": rep["reasons"] + [f"zerograd_gate could not evaluate: {e}"]}
    rep["alignment"] = z["alignment"]
    rep["optimizer"] = z["optimizer"]
    named_off = {}
    for r in z["offenders"] + z.get("allowed", []):
        for n in r["names"]:
            named_off.setdefault(n, []).append({"case": r["case"], "ambiguous": r["ambiguous"], "id": r["id"]})
    dead = {k: named_off[k] for k in want if k in named_off}
    # the trained side: every expected key must be uniquely named to a TRAINED id. A trained id is any registered id that is not an offender row; its candidate names come from the same alignment.
    unique_trained, unresolved = [], []
    ids = [i for g in (Z._pick(work, Z._OPT_KEYS)[1])["param_groups"] for i in g["params"]]
    offender_ids = {r["id"] for r in z["offenders"] + z.get("allowed", [])}
    trained_ids = [i for i in ids if (int(i) if isinstance(i, int) else i) not in offender_ids]
    names_of = _names_by_id(work, param_names)
    owner = {}
    for i in trained_ids:
        for n in names_of.get(i, []):
            owner.setdefault(n, []).append((i, len(names_of.get(i, []))))
    for k in want:
        if k in dead:
            continue
        o = owner.get(k, [])
        if len(o) == 1 and o[0][1] == 1:
            unique_trained.append(k)
        else:
            unresolved.append(k)
    rep["dead"], rep["trained_unique"], rep["unresolved"] = dead, unique_trained, unresolved
    if dead:
        rep["verdict"] = "HEAD-DEAD"
        rep["reasons"].append(f"{len(dead)} expected key(s) have no gradient history: {sorted(dead)[:4]}")
    elif unresolved:
        rep["verdict"] = "INCONCLUSIVE"
        rep["reasons"].append(f"{len(unresolved)} expected key(s) cannot be uniquely named to a trained optimizer id (pass --param-names or fix the checkpoint): {unresolved[:4]}")
    elif problems:
        rep["verdict"] = "INCONCLUSIVE"
    else:
        rep["verdict"] = "PASS"
    trained_set = {n for n, o in owner.items() if len(o) == 1 and o[0][1] == 1}
    rep["_trained_set"] = sorted(trained_set)
    return rep


def _names_by_id(ck, param_names):
    """{optimizer id: [candidate names]} by the gate's own alignment (exact with ``param_names`` / the checkpoint's own list, else the shape alignment)."""
    mk, sd = Z._pick(ck, Z._MODEL_KEYS)
    ok, opt = Z._pick(ck, Z._OPT_KEYS)
    state = opt["state"]
    ids_by_group = [list(g["params"]) for g in opt["param_groups"]]
    ids = [i for g in ids_by_group for i in g]
    entries = [(str(n), tuple(t.shape)) for n, t in sd.items() if hasattr(t, "is_floating_point") and (t.is_floating_point() or t.is_complex()) and not any(re.fullmatch(x, str(n)) for x in Z.DEFAULT_EXCLUDE_ENTRIES)]
    pn = param_names if param_names is not None else ck.get("param_names")
    id_shape = {i: Z._state_shape(state.get(i)) for i in ids}
    try:
        res = Z.resolve_names(ids_by_group, id_shape, entries, pn)
    except Z.GateError:
        res = Z.resolve_names(ids_by_group, id_shape, entries, None)
    return res["names"] or {}


def gate(ck, expect_nav_s=False, control=DEFAULT_CONTROL, param_names=None):
    """The full gate: the scoped verdict + the positive control + the negative control (a planted dead far head must read HEAD-DEAD). Returns the report dict; ``verdict`` is the exit class."""
    rep = analyze(ck, expect_nav_s, control, param_names)
    ts = rep.pop("_trained_set", [])
    c = {"positive": {"pattern": control, "trained_matches": sum(1 for n in ts if re.fullmatch(control, n))}}
    c["positive"]["ok"] = c["positive"]["trained_matches"] > 0
    if rep["verdict"] != "PASS":
        rep["controls"] = c
        return rep
    target = "s_far_out.weight"
    neg = analyze(ck, expect_nav_s, control, param_names, _mutate=target)
    neg.pop("_trained_set", None)
    c["negative"] = {"mutated": target, "verdict": neg["verdict"], "ok": neg["verdict"] == "HEAD-DEAD"}
    rep["controls"] = c
    if not (c["positive"]["ok"] and c["negative"]["ok"]):
        rep["verdict"] = "INCONCLUSIVE"
        rep["reasons"].append(f"a control did not read as required: {c}")
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description="K-REACH-e: every s_far_out.* / e_nav_s.* parameter must have trained (see module docstring).")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True, help="the verdict JSON, written atomically; its absence is the evidence that the gate did not run")
    ap.add_argument("--expect-nav-s", choices=("on", "off"), default="off")
    ap.add_argument("--control", default=DEFAULT_CONTROL)
    ap.add_argument("--param-names", default=None)
    a = ap.parse_args(argv)
    try:
        ck, how = Z.load_ckpt(a.ckpt)
        pn = json.load(open(a.param_names, encoding="utf-8")) if a.param_names else None
        rep = gate(ck, a.expect_nav_s == "on", a.control, pn)
        rep["ckpt"], rep["load"] = a.ckpt.replace("\\", "/"), how
    except Exception as e:                                              # noqa: BLE001
        rep = {"schema": SCHEMA, "verdict": "INCONCLUSIVE", "reasons": [f"{type(e).__name__}: {e}"]}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1, ensure_ascii=True)
    os.replace(tmp, a.out)
    print(f"K-REACH-e {rep['verdict']}: " + "; ".join(rep.get("reasons", [])))
    return EXIT[rep["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
