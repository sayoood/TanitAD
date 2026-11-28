#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the launch gate's DELIBERATE-REGRESSION ARMS and MEASURE whether each one fails the gate.

    python stack/scripts/launch_gate_arms.py --tree <tree> --argv-file <argv.json> \
        --out-dir <dir> [--arms a,b,...] [--profile refc] -- <extra `launch_gate.py run` args>

A guard is certified only by REINTRODUCING its defect and watching it fail (CLAUDE.md,
"guards need mutation, not inspection"). Each arm below re-installs a defect MEASURED on refcv6
(or named by SPEC_REFCV7 section 2/6) through the real gate CLI (`launch_gate.py run --arm X`),
never by editing a file, and is graded against a CONTROL run of the same checks without it:

  CAUGHT          the arm's check is FAIL/ERROR AND a reason carries the arm's signature AND that
                  signature is ABSENT from the control's reasons (the arm introduced it)
  MISCREDITED     red, but not via the arm's signature -- something else holds the line (or the
                  control was already red for the same reason: the arm cannot be read here)
  ESCAPED         the check PASSED with the defect installed: the guard is decorative
  NOT-APPLICABLE  the arm's mechanism does not exist on this tree / argv (stated, with why)

Token arms (the PASS token and the supervisor's verify) run against a PASS token minted over the
REAL binding of this tree + argv with a synthetic all-PASS evidence set -- the token mechanics
are what they test; the evidence is labelled synthetic in the result.

⛔ The verdict is `arms_result.json`; an arm that could not be graded is never counted CAUGHT.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import launch_gate as LG  # noqa: E402

#: name -> (checks, signature regex, profile override, argv edit, applicability note)
ARMS: dict[str, dict] = {
    "no_cascade_passthrough": {
        "checks": ["G-LIVE"], "sig": r"layer_u0_hat|`cascade`",
        "doc": "D-REFCV6-F3-WHITELIST: the per-stage keys leave the decoder pass-through "
               "(RefCModel.DECODER_PASSTHROUGH when the tree has it; else popped from the "
               "forward's output); with 82c2331's refusal the trainer stops at step 1"},
    "no_cascade_silent": {
        "checks": ["G-LIVE"], "sig": r"`cascade`.*ABSENT",
        "doc": "the same, and the loss SKIPS silently -- exactly as shipped for 34,500 steps"},
    "undeclared_equalize": {
        "checks": ["G-HYG"], "sig": r"UNDECLARED|UndeclaredConfigAttribute|NOT a declared field",
        "doc": "the D-REFCV6-EQUALIZE-DROPPED pattern: the pin sets a lever as an UNDECLARED "
               "attribute of the encoder config"},
    "drop_fix3_field": {
        "checks": ["G-DVB"], "sig": r"equalis|equaliz|trunk_equalize",
        "requires_flag": "--equalize-bottom-rows",
        "doc": "FIX-3's field is dropped by a dataclasses.replace rebuild (the historical drop)",
        "needs": "a declared --equalize-bottom-rows on a timm trunk (FIX-3): with nothing "
                 "declared, dropping the field changes nothing"},
    "venv_lacks_module": {
        "checks": ["G-HYG"], "sig": r"import closure: `pandas` is NOT importable",
        "doc": "a module the launch path imports at MODULE level is absent from the launch venv "
               "(the MEASURED Thor state for scipy, 2026-09-27)"},
    "box_overfit_missing": {
        "checks": ["G-BOX-OVERFIT"], "sig": r"no box overfit PASS record", "profile": "refcv7",
        "doc": "G-BOX-OVERFIT: the prerequisite record is missing (SPEC_REFCV7 A9/A10)"},
    "presence_saturated": {
        "checks": ["G-LIVE"], "sig": r"G-LIVE-PRES: .* of .* slots", "profile": "refcv7",
        "doc": "A9/A10 15.2: every slot's presence logit pushed to +20 -- the refcv6 pattern "
               "(71-99 of 100 slots confident at the 0.5 gate)",
        "needs": "a model with BOTH slot heads (--agents head and --w-box3d > 0)"},
    "missing_hygiene_module": {
        "checks": ["G-HYG"], "sig": r"config_hygiene is not importable",
        "doc": "tanitad.train.config_hygiene cannot be imported"},
    "missing_dvb_module": {
        "checks": ["G-DVB"], "sig": r"declared_vs_built is not importable",
        "doc": "tanitad.train.declared_vs_built cannot be imported"},
    "loader_skips_pin": {
        "checks": ["G-EVAL"],
        "sig": r"strict load|state_dict|param_breakdown|outputs differ|eval loader REFUSED",
        "doc": "the eval loader builds without the trainer's _pin_trainer_cfg (it then builds "
               "another model, or cannot build at all -- both must be red)"},
    "legacy_label_clock": {
        "checks": ["G-CLOCK"], "sig": r"from the independent reference|trainer's own G3 REFUSES",
        "doc": "V3Dataset.legacy_label_clock: the historical (t + w - 1) * 0.1 label clock"},
    "drivort_flag": {
        "checks": ["G-DVB"], "sig": r"DrivoR-T|drivort", "profile": "refcv7",
        "argv_add": ["--refcv7"],
        "doc": "SPEC_REFCV7 6.1: a refcv7 launch passing DrivoR-T's --refcv7"},
    "ceiling_active_in_training": {
        "checks": ["G-LIVE"], "sig": r"ceiling mask was ACTIVE in \d+ TRAINING", "profile": "refcv7",
        "doc": "SPEC_REFCV7 7: the decoder runs its eval branch during a training step, so the "
               "inference-only max-speed ceiling mask acts on training",
        "needs": "--speed-ceiling-filter in the argv (the refcv7 selection set)"},
    "required_on_missing": {
        "checks": ["G-DVB"], "sig": r"REQUIRED ON", "profile": "refcv7",
        "argv_remove": ["--speed-ceiling-filter"],
        "doc": "SPEC_REFCV7 7: a refcv7 argv missing one of the three selection mechanisms"},
    "tau_differs": {
        "checks": ["G-DVB"], "sig": r"is not the recorded tau", "profile": "refcv7",
        "argv_scale": ("--nav-compliance-tau-rad", 1.5),
        "doc": "SPEC_REFCV7 7: --nav-compliance-tau-rad differs from the banked derivation"},
    "tau_file_missing": {
        "checks": ["G-DVB"], "sig": r"without a RECORDED tau|does not exist", "profile": "refcv7",
        "gate_opt_set": ("--nav-tau-record", "__gate_arm_missing_tau__.json"),
        "requires_flag": "--graft-nav-compliance",
        "doc": "SPEC_REFCV7 7: the banked tau file is missing"},
    "map_class_weight_zero": {
        "checks": ["G-LIVE"], "sig": r"G-MAP: class .* ZERO loss contribution", "profile": "refcv7",
        "class_weight_zero": 3,
        "doc": "G-MAP: one class weight of the declared 10 cm class-weight JSON forced to 0",
        "needs": "--map-hires on and --map-hires-class-weights <json> in the argv"},
    "map_drivable_only_logging": {
        "checks": ["G-LIVE", "G-DVB"], "sig": r"G-MAP: the (in-run eval logs|trainer's)",
        "profile": "refcv7",
        "doc": "G-MAP: the eval (and the trainer's declared list) log map IoU for drivable only",
        "needs": "--map-hires on in the argv"},
    "map_drivable_only_static": {
        "checks": ["G-DVB"], "sig": r"G-MAP: the trainer's (eval-row builder omits|\w+ omits)",
        "profile": "refcv7",
        "doc": "G-MAP (static half, no smoke): the trainer's eval-row builder / declared list "
               "carries map IoU for drivable only -- run on the REAL trainer and the BUILT model",
        "needs": "--map-hires on in the argv"},
    "map_overfit_missing": {
        "checks": ["G-MAP-OVERFIT"], "sig": r"no overfit PASS record", "profile": "refcv7",
        "doc": "G-MAP-OVERFIT: the prerequisite record is missing",
        "needs": "--map-hires on in the argv"},
    "no_fmap_s8_passthrough": {
        "checks": ["G-LIVE"], "sig": r"fmap_s8", "profile": "refcv7",
        "doc": "NEW-2: `fmap_s8` leaves RefCModel.DECODER_PASSTHROUGH",
        "needs": "--map-hires on in the argv (the NEW-2 branch)"},
    "ga_off_by_one": {
        "checks": ["G-LIVE"], "sig": r"G-LIVE ga-reach: \d+ of \d+ logged rows lack",
        "doc": "the refcv6 gradient-reach trigger re-installed (filled before `step += 1`, "
               "written after): `ga_*` keys reach only the FINAL row -- MEASURED 0 keys in "
               "refcv6's 4,621 rows (map-signal audit)",
        "needs": "a model with the perception branch or the tactical decoder (a reach log "
                 "exists to break)"},
    "map_iou_not_counts": {
        "checks": ["G-LIVE"], "sig": r"COUNT keys are absent", "profile": "refcv7",
        "doc": "G-MAP RL2 (LOGGING_SPEC_MAP10 sec. 6): the eval row carries per-batch IoUs, not "
               "the argmax counts the pooled IoU is built from",
        "needs": "--map-hires on in the argv"},
    "residual_prior_ha0_ext": {
        "checks": ["G-DVB"], "sig": r"--residual-prior \['ha0_ext'\] is REFUSED",
        "profile": "refcv7", "argv_set": ("--residual-prior", ["ha0_ext"]),
        "doc": "SPEC_REFCV7 10 (A5): refcv7's prior is ha0_ext_pose; the echo's steer-read "
               "`ha0_ext` must be REFUSED"},
    "residual_prior_cv_yawrate": {
        "checks": ["G-DVB"], "sig": r"--residual-prior \['cv_yawrate'\] is REFUSED",
        "profile": "refcv7", "argv_set": ("--residual-prior", ["cv_yawrate"]),
        "doc": "SPEC_REFCV7 10 (A5): `cv_yawrate` (no acceleration) must be REFUSED"},
    "unwire_selection_term": {
        "checks": ["G-DVB"],
        "sig": r"selection mechanism `(nav_compliance|tac8_prior)` \(built\)", "profile": "refcv7",
        "requires_flag": "--graft-nav-compliance",
        "doc": "SPEC_REFCV7 2 (G-DVB's own arm, 'unwire one selection term'): the model is BORN "
               "without its built nav-compliance gate while the argv still declares "
               "--graft-nav-compliance -- D-REFCV6-CONFIG-BUILD re-installed",
        "needs": "a declared --graft-nav-compliance on a tree that builds it (FIX-4, ab436ee)"},
}
TOKEN_ARMS = ("token_argv_changed", "token_hand_edited", "token_argv_and_token_edited",
              "token_other_key", "token_tree_changed", "token_data_changed")


def _zero_class_weight(argv: list[str], k: int, out: Path) -> list[str] | None:
    """A copy of the declared 10 cm class-weight JSON with class k's weight forced to 0, and the
    argv re-pointed at it. None when the argv or the file's shape does not allow it."""
    flag = LG.PROFILES["refcv7"]["map_hires"]["class_weights_flag"]
    v = LG.flag_values(argv, flag)
    if not v or not Path(v[0]).is_file():
        return None
    w = json.loads(Path(v[0]).read_text(encoding="utf-8"))
    if isinstance(w, list):
        w[k] = 0.0
    elif isinstance(w, dict) and isinstance(w.get("weights"), list):
        w["weights"][k] = 0.0
    elif isinstance(w, dict) and len(w) > k and all(isinstance(x, (int, float))
                                                    for x in w.values()):
        w[list(w)[k]] = 0.0
    else:
        return None
    cp = out / f"class_weights_zero_{k}.json"
    cp.write_text(json.dumps(w), encoding="utf-8")
    return LG.set_flag(argv, flag, [cp.as_posix()])


def run_gate(tree: Path, argv_file: Path, out: Path, checks: list[str], profile: str,
             extra: list[str], arm: str | None) -> dict:
    cmd = [sys.executable, str(tree / "stack" / "scripts" / "launch_gate.py"), "run",
           "--profile", profile, "--checks", ",".join(checks), "--tree", str(tree),
           "--commit", "0" * 40, "--argv-file", str(argv_file), "--out-dir", str(out), *extra]
    if arm:
        cmd += ["--arm", arm]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    env.pop(LG.ARM_ENV, None)
    t0 = time.time()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    evs = {}
    for c in checks:
        ep = out / "evidence" / f"{c}.json"
        evs[c] = LG.read_json(ep) if ep.is_file() else None
    return {"rc": p.returncode, "elapsed_s": round(time.time() - t0, 1), "evidence": evs,
            "tail": p.stdout.splitlines()[-6:]}


def grade(arm: dict, ctl: dict, res: dict) -> tuple[str, str]:
    sig = re.compile(arm["sig"])
    for c in arm["checks"]:
        ev = res["evidence"].get(c)
        if ev is None:
            return "UNGRADED", f"no evidence for {c} (rc {res['rc']})"
        if ev.get("arm") is None:
            return "UNGRADED", f"{c} evidence does not record the arm -- not this run's"
        reasons = " || ".join(ev.get("reasons") or [])
        if ev["status"] == "PASS":
            return "ESCAPED", f"{c} PASSED with the defect installed"
        ctl_ev = (ctl["evidence"] or {}).get(c) or {}
        ctl_reasons = " || ".join(ctl_ev.get("reasons") or [])
        if sig.search(reasons) and not sig.search(ctl_reasons):
            return "CAUGHT", f"{c} {ev['status']}: " + next(
                r for r in ev["reasons"] if sig.search(r))[:300]
        if sig.search(reasons):
            return "MISCREDITED", (f"{c} is red with the signature, but the CONTROL already "
                                   f"carried it -- the arm cannot be read on this tree")
        return "MISCREDITED", f"{c} {ev['status']} without the signature: {reasons[:300]}"
    return "UNGRADED", "no checks"


def token_arms(tree: Path, argv_file: Path, out: Path, profile: str,
               alt_tree: Path | None = None) -> dict:
    """Mint a PASS over the REAL binding of `tree` + argv (synthetic all-PASS evidence, labelled
    so), then tamper one bound thing at a time; `verify` must REFUSE each, for the right reason."""
    import dataclasses
    import shutil
    argv = LG.load_argv_file(argv_file)
    wd = out / "token"
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True)
    ctx = LG.Ctx(profile=profile, tree=str(tree), commit="0" * 40, argv=argv,
                 out_dir=str(wd / "gate"), path_map=[],
                 tree_sha256=LG.tree_digest(LG.tree_manifest(tree)),
                 argv_sha256=LG.argv_sha256(argv), options={})
    key = LG.load_key(wd / "k.key")
    other = LG.load_key(wd / "other.key")

    def mint(c: LG.Ctx) -> Path:
        for chk in LG.CHECKS:
            ev = LG.finish_evidence(LG.new_evidence(c, chk), "PASS")
            ev["details"]["synthetic"] = "token-arm evidence: the TOKEN mechanics are under test"
            LG.write_evidence(c.out_dir, ev)
        verdict, path, _ = LG.finalize(c, key)
        if verdict != "PASS":
            raise SystemExit(f"token arms: the synthetic mint did not PASS ({verdict})")
        return path

    tok_path = mint(ctx)
    tok = json.loads(tok_path.read_text(encoding="utf-8"))
    rec = {"binding": {k: tok["binding"][k] for k in ("tree_sha256", "argv_sha256",
                                                      "data_sha256")}}
    ok, why, _ = LG.verify_token(tok_path, argv, tree, key=key)
    rec["control_verify"] = "MATCH" if ok else f"REFUSED {why}"
    cases: dict[str, tuple] = {}
    changed = LG.set_flag(argv, "--seed", ["1"])
    cases["token_argv_changed"] = LG.verify_token(tok_path, changed, tree, key=key)[:2] + (
        "argv differs",)
    t2 = json.loads(json.dumps(tok))
    t2["reasons"] = ["edited by hand"]
    p2 = wd / "edited.json"
    p2.write_text(json.dumps(t2), encoding="utf-8")
    cases["token_hand_edited"] = LG.verify_token(p2, argv, tree, key=key)[:2] + ("HMAC",)
    t3 = json.loads(json.dumps(tok))
    t3["binding"]["argv"] = changed
    t3["binding"]["argv_sha256"] = LG.argv_sha256(changed)
    p3 = wd / "argv_and_token.json"
    p3.write_text(json.dumps(t3), encoding="utf-8")
    cases["token_argv_and_token_edited"] = LG.verify_token(p3, changed, tree, key=key)[:2] + (
        "HMAC",)
    cases["token_other_key"] = LG.verify_token(tok_path, argv, tree, key=other)[:2] + ("HMAC",)
    if alt_tree is not None:
        cases["token_tree_changed"] = LG.verify_token(tok_path, argv, alt_tree, key=key)[:2] + (
            "code tree differs",)
        rec["alt_tree"] = str(alt_tree)
    else:
        cases["token_tree_changed"] = (None, ["no --alt-tree given"], "")
    data = [(f, p) for f, p in LG.data_inputs(argv, LG.PROFILES[profile]) if Path(p).is_file()]
    if data:
        flag, p0 = data[0]
        cp = wd / ("copy_" + Path(p0).name)
        cp.write_bytes(Path(p0).read_bytes())
        moved = LG.set_flag(argv, flag.split("[")[0], [cp.as_posix()])
        ctx2 = dataclasses.replace(ctx, argv=moved, argv_sha256=LG.argv_sha256(moved),
                                   out_dir=str(wd / "gate2"))
        tp2 = mint(ctx2)
        cp.write_bytes(cp.read_bytes() + b"\x00")            # the data changes AFTER the pass
        cases["token_data_changed"] = LG.verify_token(tp2, moved, tree, key=key)[:2] + (
            "data differs",)
        rec["data_arm_flag"] = flag
    else:
        cases["token_data_changed"] = (None, ["the argv reads no data file"], "")
    for name, (ok, why, sig) in cases.items():
        if ok is None:
            rec[name] = {"verdict": "NOT-APPLICABLE", "why": why}
        elif ok:
            rec[name] = {"verdict": "ESCAPED", "why": "verify returned MATCH"}
        else:
            rec[name] = {"verdict": "CAUGHT" if any(sig in r for r in why) else "MISCREDITED",
                         "why": why[:3]}
    return rec


def main(argv=None) -> int:
    LG._harden_streams()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tree", required=True)
    ap.add_argument("--argv-file", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--profile", default="refc")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--token-arms", action="store_true")
    ap.add_argument("--control-evidence", action="append", default=[],
                    help="CHECK=<evidence dir> of a completed run to use as that check's control")
    ap.add_argument("--alt-tree", default=None,
                    help="a second real tree (e.g. tip + fixes) for the tree-change arm")
    ap.add_argument("extra", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    extra = [x for x in a.extra if x != "--"]
    tree, out = Path(a.tree).resolve(), Path(a.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    argv_launch = LG.load_argv_file(a.argv_file)
    result = {"tree": str(tree), "tree_sha256": LG.tree_digest(LG.tree_manifest(tree)),
              "argv_file": str(a.argv_file), "argv_sha256": LG.argv_sha256(argv_launch),
              "started_utc": LG.utc_now(), "arms": {}, "controls": {}}
    names = [x for x in a.arms.split(",") if x]
    control_ev = dict(x.split("=", 1) for x in a.control_evidence)
    for name in names:
        arm = ARMS[name]
        prof = arm.get("profile", a.profile)
        if name in ("no_fmap_s8_passthrough", "map_class_weight_zero", "map_drivable_only_logging",
                    "map_drivable_only_static",
                    "map_overfit_missing", "map_iou_not_counts") and not LG.hires_on(
                        LG.PROFILES[prof], argv_launch):
            result["arms"][name] = {"verdict": "NOT-APPLICABLE", "why": arm["needs"]}
            continue
        req = arm.get("requires_flag") or ("--speed-ceiling-filter"
                                           if name == "ceiling_active_in_training" else None)
        if req and not LG.has_flag(argv_launch, req):
            result["arms"][name] = {"verdict": "NOT-APPLICABLE",
                                    "why": arm.get("needs") or f"{req} is not in the argv"}
            continue
        argv_file = Path(a.argv_file)
        arm_argv = list(argv_launch)
        if arm.get("argv_add"):
            arm_argv = arm_argv + arm["argv_add"]
        for f in arm.get("argv_remove", ()):
            if not LG.has_flag(arm_argv, f):
                arm_argv = None
                break
            arm_argv = LG.set_flag(arm_argv, f, None)
        if arm_argv is not None and arm.get("argv_set"):
            f, vals = arm["argv_set"]
            arm_argv = LG.set_flag(arm_argv, f, list(vals))
        if arm_argv is not None and arm.get("argv_scale"):
            f, k = arm["argv_scale"]
            v = LG.flag_values(arm_argv, f)
            arm_argv = LG.set_flag(arm_argv, f, [repr(float(v[0]) * k)]) if v else None
        if arm_argv is not None and arm.get("class_weight_zero") is not None:
            arm_argv = _zero_class_weight(arm_argv, int(arm["class_weight_zero"]), out)
        if arm_argv is None:
            result["arms"][name] = {"verdict": "NOT-APPLICABLE",
                                    "why": "the argv lacks the flag this arm edits"}
            continue
        if arm_argv != argv_launch:
            argv_file = out / f"argv_{name}.json"
            argv_file.write_text(json.dumps(arm_argv), encoding="utf-8")
        arm_extra = list(extra)
        if arm.get("gate_opt_set"):
            o, v = arm["gate_opt_set"]
            while o in arm_extra:
                i = arm_extra.index(o)
                del arm_extra[i:i + 2]
            arm_extra += [o, str(out / v)]
        ckey = f"{prof}:{','.join(arm['checks'])}:{argv_file.name}"
        reuse = [control_ev.get(c) for c in arm["checks"]]
        if ckey not in result["controls"] and all(reuse):
            # a completed run's evidence stands in for the control (a G-EVAL control costs three
            # full CPU forwards); it must be bound to the SAME binding as this arm's runs
            evs = {c: LG.read_json(Path(control_ev[c]) / f"{c}.json") for c in arm["checks"]}
            result["controls"][ckey] = {"reused_from": {c: control_ev[c] for c in arm["checks"]},
                                        "status": {k: v.get("status") for k, v in evs.items()},
                                        "reasons": {k: v.get("reasons") for k, v in evs.items()},
                                        "_ev": {"evidence": evs}}
        if ckey not in result["controls"]:
            c = run_gate(tree, Path(a.argv_file), out / f"control_{prof}_{'_'.join(arm['checks'])}",
                         arm["checks"], prof, extra, None)
            result["controls"][ckey] = {k: v for k, v in c.items() if k != "evidence"} | {
                "status": {k: (v or {}).get("status") for k, v in c["evidence"].items()},
                "reasons": {k: (v or {}).get("reasons") for k, v in c["evidence"].items()}}
            result["controls"][ckey]["_ev"] = c
        ctl = result["controls"][ckey]["_ev"]
        r = run_gate(tree, argv_file, out / f"arm_{name}", arm["checks"], prof, arm_extra, name)
        verdict, why = grade(arm, ctl, r)
        if verdict == "MISCREDITED" and arm.get("needs") and name == "drop_fix3_field":
            ev = r["evidence"].get("G-DVB") or {}
            probe = (ev.get("details") or {}).get("gate_probe_equalize") or {}
            if probe.get("trunk") is None:
                verdict, why = "NOT-APPLICABLE", arm["needs"]
        if name == "ga_off_by_one" and verdict != "CAUGHT":
            cev = (ctl.get("evidence") or {}).get("G-LIVE") or {}
            gr = (cev.get("details") or {}).get("ga_reach") or {}
            if not gr.get("ga_on") and not ((cev.get("details") or {}).get("map_hires") or
                                            {}).get("on"):
                verdict, why = "NOT-APPLICABLE", f"{arm['needs']} (control: ga_on False)"
        if name == "unwire_selection_term":
            # the arm must have REMOVED something, or it tested nothing (never counted CAUGHT)
            ev = r["evidence"].get("G-DVB") or {}
            un = (ev.get("details") or {}).get("arm_unwire") or {}
            if un.get("unwired") is None:
                verdict, why = "NOT-APPLICABLE", f"{arm['needs']} ({un.get('why', 'no record')})"
            else:
                why = f"[unwired {un['unwired']}] {why}"
        result["arms"][name] = {"verdict": verdict, "why": why, "doc": arm["doc"],
                                "checks": arm["checks"], "profile": prof,
                                "elapsed_s": r["elapsed_s"],
                                "status": {k: (v or {}).get("status")
                                           for k, v in r["evidence"].items()}}
        LG._say(f"arm {name}: {verdict} -- {why[:200]}")
    for v in result["controls"].values():
        v.pop("_ev", None)
    if a.token_arms:
        result["token_arms"] = token_arms(tree, Path(a.argv_file), out, a.profile,
                                          Path(a.alt_tree) if a.alt_tree else None)
        for k in TOKEN_ARMS:
            LG._say(f"token arm {k}: {result['token_arms'][k]['verdict']}")
    result["finished_utc"] = LG.utc_now()
    counts: dict[str, int] = {}
    for v in list(result["arms"].values()) + [v for k, v in result.get("token_arms", {}).items()
                                              if k in TOKEN_ARMS]:
        counts[v["verdict"]] = counts.get(v["verdict"], 0) + 1
    result["counts"] = counts
    LG.write_json(out / "arms_result.json", LG.scrub(result))
    LG._say(f"arms: {counts} -> {out / 'arms_result.json'}")
    bad = counts.get("ESCAPED", 0) + counts.get("UNGRADED", 0)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
