#!/usr/bin/env python3
"""THE REFe LAUNCH GATE: no training step runs until this mints a PASS token for the EXACT (commit, code tree, argv, data) being launched.
CLAUDE.md, PI 2026-09-26 (BINDING).  Route, contract and every regression arm: eval/REFE_LAUNCH_GATE.md.

    # DEV BOX -- ship the exact landed bytes (writes SHIP_MANIFEST.json; refuses a file that is not its blob at the commit)
    python refe/launch_gate.py ship-manifest --pkg <package root> --commit <40-hex> --out <package root>/SHIP_MANIFEST.json
    # POD -- run every check on the REAL argv and REAL data, then mint (the chain does this; a token is never hand-written)
    python launch_gate.py run --profile refe_p2 --pkg /workspace/refe-plan --args "$ARGS" --out-dir <dir> --gate-sets <d> --gate-pdm <d> \
        --gate-bank <d> --p2-gate-json <gate.json>
    # POD -- the chain, before every train.py launch:  exits 0 (ZZGATE-MATCHZZ) or 5 (ZZGATE-REFUSEDZZ)
    python launch_gate.py verify --token <PASS_*.json> --args "$ARGS" --pkg /workspace/refe-plan --profile refe_p2
    # anywhere, instant -- what a profile refuses before a single check runs (the P6 profile refuses today)
    python launch_gate.py refusals --profile refe_p6 --args "<the P6 argv>"

Token: schema `tanitad.launch_gate/1` (gate_token.py), HMAC-SHA256 with a pod-local key; PASS only if every required check has usable PASS
evidence bound to the same (profile, commit, tree_sha256, argv_sha256) whose `inputs_read` fingerprints equal the launch data manifest.
REHEARSAL evidence (tiny config, synthetic images, CPU) never counts toward a token.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_token as GT  # noqa: E402

CHECKS = ("G-HYG", "G-DVB", "G-LIVE", "G-CLOCK", "G-EVAL", "G-CKPT", "G-SUITE", "G-P2GATE", "G-LIVE-LAB", "G-LIVE-MEM", "G-EGR")
SUITE = [{"script": "test_p2r_chain.py", "token": "ZZTEST_PASS"}, {"script": "test_launch_gate.py", "args": ["--static", "--no-banked"], "token": "ZZLG_STATIC_PASS"}]
FORBIDDEN = ("--synthetic", "--overfit", "--overfit-n", "--overfit-thresh", "--spread", "--break-optimizer", "--halt-after-steps",
             "--declare-change", "--preflight", "--cpu", "--n-cameras", "--no-undistort", "--no-cosine", "--weights")
P2_DATA = ("--targets", "--scorer-targets", "--images", "--calib", "--init-from", "--onpolicy-targets", "--pdm-labels")
P6_TERMS = ("traj_L1", "score", "pos_L1", "yaw_L1", "bce_nc", "bce_dac", "bce_ep", "bce_ttc", "bce_c", "bce_ddc")
_BASE = dict(trainer="refe/train.py", live_terms=("traj_L1", "score"), ckpt_steps=4, smoke_steps=2, smoke_accum=2, suite=SUITE,
             forbidden_flags=FORBIDDEN, output_flags=("--out",), mintable=True, require_commit=True, rehearsal=False,
             checks=("G-HYG", "G-DVB", "G-LIVE", "G-EVAL", "G-CKPT", "G-SUITE"), na_ok=("G-CLOCK",),
             clock_na="P2 has no label clock: its labels are the PDM relabel of a banked on-policy bank, stamped by the relabeller's ckpt_step")

PROFILES = {
    # PREREG_P2 / eval/PREREG_P2_AMENDMENT_1.md: the joint fine-tune p2r_pod.sh launches.  required_flags are its design, asserted not trusted.
    "refe_p2": {**_BASE, "name": "refe_p2", "data_flags": P2_DATA, "checks": _BASE["checks"] + ("G-P2GATE",),
                "required_flags": {"--scorer-mode": "onpolicy", "--pdm-only": None, "--scorer-sees-trunk": None, "--epochs": "3",
                                   "--epoch-unit": "scenes", "--batch": "4", "--accum": "64", "--lr": "5e-5", "--amp": "bf16"},
                "provisional_flags": ()},
    # SPEC_P6_PAPER_FAITHFUL.md section 5.  ⛔ REFUSES TODAY: --yaw-wrap / --nav-mode / --labeller are D5 / D6 / D1 and are NOT built, and the
    # trainer's metrics carry no pos / yaw / per-component BCE terms.  Flag and term NAMES are PROVISIONAL until the implementer lands them.
    "refe_p6": {**_BASE, "name": "refe_p6", "data_flags": ("--targets", "--images", "--calib", "--metric-cache"),
                "suite": SUITE + [{"script": "test_p6_levers.py", "args": ["--fast"], "token": "ZZP6_PASS"}],
                # G-LIVE-LAB / G-LIVE-MEM / G-EGR are POD-RUN: the gate consumes their instruments' output (--gate-lab-json / --gate-mem-metrics / --gate-egr-json)
                "checks": ("G-HYG", "G-DVB", "G-LIVE", "G-CLOCK", "G-EVAL", "G-CKPT", "G-SUITE", "G-LIVE-LAB", "G-LIVE-MEM", "G-EGR"), "na_ok": (),
                "live_terms": P6_TERMS,
                "forbidden_flags": FORBIDDEN + ("--grow", "--grow-scenes", "--onpolicy-targets", "--init-from", "--pdm-labels", "--pdm-only",
                                                "--scorer-mode", "--label-backend"),
                # --yaw-wrap must be NAMED in the argv; its VALUE is the PI's open ruling P6-Y (on = wrapped, today's behaviour; off = the paper's literal L1).
                # TODO(Master Mind): once the PI rules P6-Y, replace None with "on" or "off" so the gate asserts the ruling, not just the presence.
                "required_flags": {"--scorer-sees-trunk": None, "--yaw-wrap": None, "--nav-mode": "command", "--labeller": "online",
                                   "--epochs": "25", "--batch": "4", "--accum": "64", "--lr": "2e-4", "--weight-decay": "0.01", "--amp": "bf16",
                                   "--score-w": "0.1"},
                "provisional_flags": (),
                # SPEC_P6 section 0.2 wrote 10,063 steps (= ceil(25 x 103,037 / 256)); the trainer's own definition (train.py: epochs x ceil(epoch_len / (batch x
                # accum)), re-derived independently by G-DVB) gives 25 x 403 = 10,075 -- the final run's measured step count. The profile registers the TRAINER's.
                "expect_schedule": {"epochs": 25, "steps": 10075, "epoch_len": 103037, "n_tuples": 189888}},
    # the dev-box rehearsal: P2's levers on the tiny bank.  Exercises every instrument; its evidence is REHEARSAL and can never mint.
    "refe_rehearsal": {**_BASE, "name": "refe_rehearsal", "data_flags": ("--targets", "--scorer-targets", "--init-from", "--onpolicy-targets",
                                                                          "--pdm-labels"),
                       "forbidden_flags": tuple(f for f in FORBIDDEN if f not in ("--synthetic", "--cpu", "--n-cameras")), "mintable": False,
                       "require_commit": False, "rehearsal": True, "required_flags": {"--scorer-mode": "onpolicy", "--scorer-sees-trunk": None},
                       "provisional_flags": (), "eval_cfg_allowed": ("n_cameras", "cameras")},
}


def parse_args_string(s: str) -> list:
    return GT.argv_from_string(s)


def flag_present(argv, flag):
    return flag in argv


def profile_refusals(prof: dict, argv: list) -> list:
    """Reasons the profile refuses this argv BEFORE any check runs.  (a) a forbidden flag; (b) a required lever missing or different;
    (c) a lever the profile DECLARES that the trainer does not BUILD (flag absent from the parser) -- a launch blocker, never a footnote;
    (d) a declared loss term the trainer never emits."""
    import re
    import train as T
    out = []
    parser = T.build_parser()
    known = {o for a in parser._actions for o in a.option_strings}
    for f in prof.get("forbidden_flags", ()):
        if f in argv:
            out.append(f"argv carries {f}, which {prof['name']} FORBIDS in a launch")
    for f, want in prof.get("required_flags", {}).items():
        if f not in argv:
            out.append(f"{prof['name']} requires {f}{'' if want is None else ' ' + want}; the argv does not carry it")
            continue
        got = GT.flag_values(argv, f)
        if want is not None:
            try:
                same = got and float(got[0]) == float(want)
            except ValueError:
                same = got and got[0] == want
            if not same:
                out.append(f"{prof['name']} requires {f} {want}; the argv has {f} {' '.join(got or [])}")
    for f in sorted(set(prof.get("provisional_flags", ())) | {f for f in prof.get("required_flags", {}) if f not in known}):
        if f not in known:
            out.append(f"DECLARED, NOT BUILT: {prof['name']} declares {f} but the trainer's parser has no such flag "
                       f"(a config that declares a lever the model does not build is a launch blocker)")
    src = (HERE / "train.py").read_text(encoding="utf-8")
    for t in prof.get("live_terms", ()):
        if not re.search(r'["\']' + re.escape(t) + r'["\']', src):
            out.append(f"DECLARED, NOT BUILT: {prof['name']} requires the loss term {t!r}; train.py never emits it")
    return out


# ============================================================================================ orchestration
def derived_argvs(prof: dict, argv: list, work: Path, gate_sets=None, gate_pdm=None, gate_bank=None) -> tuple:
    """(build_argv, smoke_argv).  BUILD keeps every launch lever (the schedule is observed) and points the on-policy side at the gate's small
    sets; SMOKE additionally swaps the bank for the carved gate bank, cuts the schedule to `smoke_steps`, and zeroes the workers."""
    from gate_checks import _drop, _set
    b = list(argv)
    b = _drop(b, "--resume")
    if gate_sets:
        b = _set(b, "--onpolicy-targets", gate_sets)
    if gate_pdm:
        b = _set(b, "--pdm-labels", gate_pdm)
    b = _set(b, "--out", str(work / "build_run"))
    s = list(b)
    if gate_bank:
        s = _set(_set(s, "--targets", gate_bank), "--scorer-targets", gate_bank)
    for flag in ("--epochs", "--grow-scenes"):
        if flag in s:
            i = s.index(flag)
            del s[i:i + 2]
    s = _set(s, "--steps", str(prof["smoke_steps"]))
    s = _set(s, "--accum", str(min(int((GT.flag_values(s, "--accum") or ["1"])[0]), prof["smoke_accum"])))
    s = _set(s, "--workers", "0")
    s = _set(s, "--log-every", "1")
    s = _set(s, "--ckpt-every-min", "999")
    if prof.get("rehearsal"):
        s = _drop(s, "--compile")
    s = _set(s, "--out", str(work / "smoke_run"))
    return b, s


def write_evidence(out_dir: Path, ev: dict, binding: dict, data: dict) -> Path:
    ev = dict(ev)
    ev["binding"] = binding
    ev["inputs_read"] = {k: v for k, v in data.items()}
    return GT.write_json(out_dir / f"evidence_{ev['check']}.json", ev)


def gate_inputs_of(p2_gate_json=None, lab_json=None, mem_metrics=None, egr_json=None) -> dict:
    return {k: v for k, v in (("gate:p2_gate_json", p2_gate_json), ("gate:lab_json", lab_json), ("gate:mem_metrics", mem_metrics),
                              ("gate:egr_json", egr_json)) if v}


def run_checks(prof: dict, argv: list, pkg: Path, out_dir: Path, *, only=None, arm=None, gate_sets=None, gate_pdm=None, gate_bank=None,
               p2_gate_json=None, commit=None, lab_json=None, mem_metrics=None, egr_json=None) -> dict:
    """Run every check of the profile (or `only`), writing evidence files.  Returns {check: status}."""
    import gate_checks as C
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "work"
    tree = GT.tree_manifest(pkg)
    binding = GT.binding_of(prof["name"], commit, GT.tree_digest(tree), argv)
    gate_inputs = gate_inputs_of(p2_gate_json, lab_json, mem_metrics, egr_json)
    data = GT.data_manifest(argv, prof["data_flags"], gate_inputs)
    rehearsal = bool(prof.get("rehearsal"))
    want = [c for c in (prof["checks"] + tuple(prof.get("na_ok", ()))) if (only is None or c in only)]
    status: dict = {}
    refusals = profile_refusals(prof, argv)

    def emit(ev):
        write_evidence(out_dir, ev, binding, data)
        status[ev["check"]] = ev["status"]
        print(f"  {ev['check']:9s} {ev['status']:5s} {('; '.join(ev['reasons']))[:200]}", flush=True)
    if refusals:
        for c in want:
            emit(C.evidence(c, "FAIL", ["profile refusal: " + r for r in refusals], {"refusals": refusals}, arm=arm, rehearsal=rehearsal))
        return status
    b_argv, s_argv = derived_argvs(prof, argv, work, gate_sets, gate_pdm, gate_bank)
    need_b = any(c in want for c in ("G-DVB", "G-CLOCK"))
    need_s = any(c in want for c in ("G-LIVE", "G-EVAL", "G-DVB", "G-LIVE-LAB"))
    b = C.instrumented_run(b_argv, mode="build", work=work / "build", arm=arm) if need_b else None
    if b is not None:
        _free(b)                      # the BUILD model is not needed again (a second ViT-L would sit on the pod's GPU through the smoke)
    s = C.instrumented_run(s_argv, mode="smoke", work=work / "smoke", arm=arm) if need_s else None
    prof2 = dict(prof)
    if "G-HYG" in want:
        emit(C.check_hyg(arm=arm, rehearsal=rehearsal))
    if "G-DVB" in want:
        emit(C.check_dvb(b_argv, b, s, prof2, arm=arm, rehearsal=rehearsal))
    if "G-LIVE" in want:
        emit(C.check_live(s_argv, s, prof2, arm=arm, rehearsal=rehearsal))
    if "G-CLOCK" in want:
        emit(C.check_clock(b_argv, b, prof2, arm=arm, rehearsal=rehearsal))
    if "G-EVAL" in want:
        emit(C.check_eval(s_argv, s, prof2, arm=arm, rehearsal=rehearsal))
    if "G-CKPT" in want:
        emit(C.check_ckpt(s_argv, work / "ckpt", prof2, arm=arm, rehearsal=rehearsal))
    if "G-SUITE" in want:
        emit(C.check_suite(prof2, arm=arm, rehearsal=rehearsal))
    if "G-P2GATE" in want:
        emit(consume_p2_gate(p2_gate_json, arm=arm, rehearsal=rehearsal))
    if "G-LIVE-LAB" in want:
        emit(C.check_live_lab(lab_json, s, argv, arm=arm, rehearsal=rehearsal))
    if "G-LIVE-MEM" in want:
        emit(C.check_live_mem(mem_metrics, arm=arm, rehearsal=rehearsal))
    if "G-EGR" in want:
        emit(C.check_egr(egr_json, arm=arm, rehearsal=rehearsal))
    return status


def _free(cap) -> None:
    import gc
    cap.model = None
    cap.first_batch = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:                                                        # noqa: BLE001 -- best effort
        pass


def consume_p2_gate(path, arm=None, rehearsal=False) -> dict:
    """p2_gate.py's own output (G1 warm start == the planner's load; G2 the analytic pair, incl. the detached arm that must read EXACTLY 0.0;
    G3 pdm-only; G4 the real trainer smoke) is a SUB-CHECK, not duplicated: parsed, never grepped, all four present and ok.  Its file's sha256 is
    bound into the token (`gate:p2_gate_json`)."""
    import gate_checks as C
    if not path or not Path(path).is_file():
        return C.evidence("G-P2GATE", "FAIL", [f"p2_gate.py output missing: {path}"], {}, arm=arm, rehearsal=rehearsal)
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    g = d.get("gates", {})
    bad = [p for p in ("G1", "G2", "G3", "G4") if not any(k.startswith(p + "_") and v.get("ok") is True for k, v in g.items())]
    reasons = []
    if d.get("verdict") != "PASS" or bad:
        reasons.append(f"p2_gate.py is not PASS: verdict={d.get('verdict')} missing_or_not_ok={bad}")
    return C.verdict("G-P2GATE", reasons, {"gates": {k: v.get("ok") for k, v in g.items()}, "file_sha256": GT.sha256_file(path)}, arm=arm, rehearsal=rehearsal)


def collect(out_dir: Path, check: str):
    p = out_dir / f"evidence_{check}.json"
    return GT.read_json(p) if p.is_file() else None


def finalize(prof: dict, argv: list, pkg: Path, out_dir: Path, key: bytes, *, p2_gate_json=None, commit=None, lab_json=None, mem_metrics=None,
             egr_json=None):
    """Mint the token from the evidence ON DISK.  Every required check must have usable PASS evidence bound to this launch."""
    tree = GT.tree_manifest(pkg)
    gate_inputs = gate_inputs_of(p2_gate_json, lab_json, mem_metrics, egr_json)
    data = GT.data_manifest(argv, prof["data_flags"], gate_inputs)
    binding = GT.binding_of(prof["name"], commit, GT.tree_digest(tree), argv)
    reasons = profile_refusals(prof, argv) + GT.data_problems(data)
    manifest = GT.read_json(pkg / GT.SHIP_MANIFEST) if (pkg / GT.SHIP_MANIFEST).is_file() else None
    if manifest is not None:
        commit = manifest.get("commit")
        binding = GT.binding_of(prof["name"], commit, GT.tree_digest(tree), argv)
        if manifest.get("files") != tree:
            reasons.append(f"the pod's code tree is not the shipped tree: {GT.tree_diff(manifest.get('files') or {}, tree)}")
    if prof.get("require_commit") and not commit:
        reasons.append("no commit is bound: ship SHIP_MANIFEST.json (launch_gate.py ship-manifest --commit <landed 40-hex>) with the package")
    if not prof.get("mintable", True):
        reasons.append(f"profile {prof['name']} is a REHEARSAL profile: it can never mint a token")
    if os.environ.get("TANITAD_GATE_ARM"):
        reasons.append(f"a regression arm is active ({os.environ['TANITAD_GATE_ARM']}): an arm run can never PASS")
    checks: dict = {}
    for c in prof["checks"] + tuple(prof.get("na_ok", ())):
        ev = collect(out_dir, c)
        if ev is None:
            checks[c] = {"status": "MISSING"}
            if c in prof["checks"]:
                reasons.append(f"{c}: no evidence file")
            continue
        why = []
        if ev.get("binding") != binding:
            why.append("binding differs")
        if ev.get("arm"):
            why.append(f"produced under regression arm {ev['arm']}")
        if ev.get("rehearsal"):
            why.append("REHEARSAL evidence never counts toward a token")
        for f, fp in (ev.get("inputs_read") or {}).items():
            mine = data.get(f)
            if mine is None or GT.fp_key(mine) != GT.fp_key(fp):
                why.append(f"input {f} was read with a different fingerprint")
        ok_status = ("PASS",) if c in prof["checks"] else ("PASS", "NA")
        status = ev.get("status") if not why else "REJECTED"
        checks[c] = {"status": status if status in ok_status or status == "REJECTED" else status, "evidence_sha256": GT.sha256_file(out_dir / f"evidence_{c}.json"),
                     "finished_utc": ev.get("finished_utc"), "reasons": (ev.get("reasons") or [])[:6] + why}
        if status not in ok_status:
            reasons.append(f"{c} {status}: " + " | ".join((ev.get("reasons") or [])[:2] + why))
        if c in prof.get("na_ok", ()) and ev.get("status") == "NA":
            checks[c]["status"] = "PASS"
            checks[c]["na_reason"] = (ev.get("reasons") or [""])[0]
    return GT.mint(profile=prof["name"], commit=commit, tree_man=tree, argv=argv, trainer=prof["trainer"], data=data, checks=checks,
                   reasons=reasons, key=key, out_dir=out_dir, gate_inputs=gate_inputs, gate_self_sha=GT.sha256_file(HERE / "launch_gate.py"))


# ============================================================================================ CLI
def _argv_of(a) -> list:
    if getattr(a, "args_file", None):
        return GT.read_json(a.args_file)
    if getattr(a, "args", None) is not None:
        return parse_args_string(a.args)
    raise SystemExit("--args or --args-file is required")


def cmd_run(a) -> int:
    prof = PROFILES[a.profile]
    argv = _argv_of(a)
    pkg, out = Path(a.pkg), Path(a.out_dir)
    key = GT.load_key(a.key_file, create=True, forbid_under=[pkg])
    man = GT.read_json(pkg / GT.SHIP_MANIFEST) if (pkg / GT.SHIP_MANIFEST).is_file() else None
    commit = man.get("commit") if man else a.commit
    arm = os.environ.get("TANITAD_GATE_ARM")
    if a.gate_bank and a.gate_sets and not Path(a.gate_bank).exists():
        import gate_fixture as F
        src = (GT.flag_values(argv, "--targets") or [None])[0]
        print("  carved the smoke bank:", F.carve_gate_bank(src, a.gate_sets, a.gate_bank), flush=True)
    run_checks(prof, argv, pkg, out, only=set(a.only.split(",")) if a.only else None, arm=arm, gate_sets=a.gate_sets, gate_pdm=a.gate_pdm,
               gate_bank=a.gate_bank, p2_gate_json=a.p2_gate_json, commit=commit, lab_json=a.gate_lab_json, mem_metrics=a.gate_mem_metrics,
               egr_json=a.gate_egr_json)
    verdict, path, tok = finalize(prof, argv, pkg, out, key, p2_gate_json=a.p2_gate_json, commit=commit, lab_json=a.gate_lab_json,
                                  mem_metrics=a.gate_mem_metrics, egr_json=a.gate_egr_json)
    for r in tok["reasons"][:8]:
        print("  REASON:", r[:300])
    print(f"verdict {verdict} -> {path}", flush=True)
    print(f"ZZGATE-{verdict}-{(tok['binding']['commit'] or tok['binding']['tree_sha256'])[:12]}ZZ", flush=True)
    return 0 if verdict == "PASS" else 1


def cmd_verify(a) -> int:
    prof = PROFILES.get(a.profile) if a.profile else None
    try:
        key = GT.load_key(a.key_file, create=False, forbid_under=[a.pkg])
        ok, reasons, info = GT.verify_token(a.token, _argv_of(a), a.pkg, key=key, profile=a.profile, data_flags=prof["data_flags"] if prof else (),
                                            commit=a.commit)
    except Exception as e:                                                   # noqa: BLE001 -- refuse, never pass
        ok, reasons, info = False, [f"{type(e).__name__}: {e}"], {}
    for r in reasons:
        print("REFUSED:", r[:400], file=sys.stderr, flush=True)
    if a.json:
        GT.write_json(a.json, {"verdict": "MATCH" if ok else "REFUSED", "reasons": reasons, "info": info, "utc": GT.utc_now()})
    print(f"ZZGATE-{'MATCH' if ok else 'REFUSED'}ZZ", flush=True)
    return 0 if ok else 5


def cmd_ship(a) -> int:
    man = GT.ship_manifest(a.pkg, commit=a.commit, git_dir=a.git_dir, repo_prefix=a.repo_prefix)
    GT.write_json(a.out, man)
    print(f"ZZSHIP_OK {len(man['files'])} files tree {man['tree_sha256'][:16]} commit {a.commit and a.commit[:12]}", flush=True)
    return 0


def cmd_refusals(a) -> int:
    rs = profile_refusals(PROFILES[a.profile], _argv_of(a))
    for r in rs:
        print("REFUSED:", r)
    print(f"ZZGATE-{'REFUSED' if rs else 'CLEAN'}ZZ")
    return 5 if rs else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "verify", "refusals"):
        p = sub.add_parser(name)
        p.add_argument("--profile", required=(name != "verify"), choices=sorted(PROFILES))
        p.add_argument("--args")
        p.add_argument("--args-file")
        p.add_argument("--key-file")
        p.add_argument("--pkg", required=(name != "refusals"))
        if name == "run":
            p.add_argument("--out-dir", required=True)
            p.add_argument("--only")
            p.add_argument("--commit")
            p.add_argument("--gate-sets")
            p.add_argument("--gate-pdm")
            p.add_argument("--gate-bank")
            p.add_argument("--p2-gate-json")
            p.add_argument("--gate-lab-json", help="refe_p6: diag_p6_labeller_identity.py --out (G-LIVE-LAB)")
            p.add_argument("--gate-mem-metrics", help="refe_p6: metrics.jsonl of a >= 200-micro-batch run on the real card (G-LIVE-MEM)")
            p.add_argument("--gate-egr-json", help="refe_p6: hf_egress.py gate's g_egr_result.json (G-EGR)")
        if name == "verify":
            p.add_argument("--token", required=True)
            p.add_argument("--commit")
            p.add_argument("--json")
    p = sub.add_parser("ship-manifest")
    p.add_argument("--pkg", required=True)
    p.add_argument("--commit")
    p.add_argument("--git-dir")
    p.add_argument("--repo-prefix", default="")
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"run": cmd_run, "verify": cmd_verify, "refusals": cmd_refusals, "ship-manifest": cmd_ship}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
