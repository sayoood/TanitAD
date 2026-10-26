"""ship_r8_hsgate_verdict.py -- PREREG_HSGATE (2331e1f3...) bar G3: a COPY of ship_r8_sub/ship_r8_4c_verdict.py (left untouched; built by
make_hsgate_verdict.py, diff ship_r8_hsgate_verdict.DIFF.txt) with the arm HSGATE8 added.  Clause (1) for HSGATE8 = Amendment 4's clause-1 function
(UNEDITED, run on the --hsgate forward) AND the PREREG_HSGATE seal re-hashed now == the forward's record AND the gate file sha256 now ==
REGISTERED_HSGATE.sha256 == the forward's record AND K_REG_HSGATE, K_FIT_HSGATE, K_ID0, K_ID1, K_EVALFEAT, K_LANDED, K_GATE_SURF, K_TRACE AND the A5 input
traces re-hashed now == the forward's record.  (1b) = Amendment 4's (1b) AND Amendment 5's (1b).  Writes only under refcv9_hsgate.

ship_r8_4c_verdict.py -- SHIP-R8 AMENDMENT 4c: a COPY of ship_r8_sub_verdict.py (left untouched: it is pinned by MANIFEST_4B) with the arm H2CNL_DEPLOY8
added (diff: ship_r8_4c_verdict.DIFF.txt).  Clause (1) for H2CNL_DEPLOY8 = Amendment 4's clause-1 function (UNEDITED, run on the --h2cnl forward) AND the 4c
seal re-hashed now == the forward's record AND the H2cn-L stage file's sha256 now == REGISTERED_H2CNL_STAGE.sha256 == the forward's record AND K_REG_H2CNL
(registration precedes the first eval-side output) AND K_FREEZE_H2CNL (the freeze controls: K-H2CN-REPRO, K-FIT, K-RELOAD, K-DETERM, K-DISJ, K-JOIN, K-MUT)
AND K_JOIN_EVAL.  The mutation arm H2CNL_MUT8 is read BESIDE with the same clause-1 form against REGISTERED_H2CNL_MUT_STAGE.sha256.  An --h2cnl forward is
never read into out/ship_r8 or out_4b/ship_r8.

ship_r8_sub_verdict.py -- SHIP-R8 clauses (1)-(7) + (1b) applied MECHANICALLY to the Amendment-4 / Amendment-5 arm.  Pure json in, json + md out.
NEW file (EvalFlyWheel 2026-10-26).  Clauses (1b, A4 form) and (2)-(7) are the REGISTERED verdict's own functions -- ship_r8_verdict_pf.py and its pinned base
copy _ship_r8_verdict_base.py (sha256 re-checked at import by ship_r8_verdict_pf.load_base) -- imported, never edited, with the arm name swapped
(B.ARM, the clause_e `arm` argument).  Clause (1) is REPLACED as each amendment writes it:

  AMENDMENT 4 (H2CN_DEPLOY8): the frozen stage file's sha256 == the registered one (re-hashed now) AND the registration precedes the first eval-side stage
      output (the forward's gate record vs its first-output time) AND the fit used no eval139 / train4572 clip (fresh-clip disjointness, asserted at the
      freeze and re-asserted at the forward) AND the H2CN_PERSURFACE identity (S2 H2cn - H0 ADE == -0.0130 / -0.0136 within 1e-4) AND the RETIME_DEPLOY8
      identity (re-read == the registered SHIP_R8_step48852ema_pf.json cells EXACTLY: forward blocks + clauses (2)-(5) evidence + the TAC cells).
      (1b) unchanged (same rolls): the registered FEED_PARITY files re-checked by sha256 against the registered verdict's inputs, then its clause1b.
  AMENDMENT 5 (FACTORED_DEPLOY8): the registered table's sha256 checks (ship_r8_verdict_pf.clause1_pf, unchanged) AND the identity controls
      (FACTORED with P_ss := P_os == OSONLY_DEPLOY8 bit for bit; RETIME_DEPLOY8 == the registered cells) AND both input traces' sha256 (re-hashed now ==
      the forward's record [== --trace-pins when given]) + final flag + 4,754 windows + hook counters.  (1b): the os-only rolls' own manifests read
      perframe_full (not shown -> INCOMPLETE).
OVERALL (both amendments): FAIL if any clause FAILs; else INCOMPLETE if any is INCOMPLETE (a missing clause is INCOMPLETE); else PASS.

    python ship_r8_sub_verdict.py --arm H2CN_DEPLOY8 --forward <FORWARD_H2CN_*.json> --arms-dir <out>/arms/h2cn
    python ship_r8_sub_verdict.py --arm FACTORED_DEPLOY8 --forward <FORWARD_FACTORED_SCORED_*.json> --arms-dir <out>/arms/factored_scored
    python ship_r8_sub_verdict.py --identity-only --forward <FORWARD_FACTORED_*.json> --arms-dir <out>/arms/factored     # controls only, no scored arm
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUBDIR = Path("C:/Users/Admin/refcv8_pf/ship_r8_sub")                # READ-ONLY
sys.path.insert(0, str(SUBDIR))
sys.path.insert(0, str(HERE))
import ship_r8_sub_common as C  # noqa: E402
import hsgate_common as HG  # noqa: E402

sys.path.insert(0, str(C.PF_HOLD / "code"))
import ship_r8_verdict_pf as VPF  # noqa: E402   (loads + pins the base copy)

B = VPF.B
SCRIPT_SHA = C.sha256_file(__file__)
TREE = "step48852ema_pf"
REG_SHIP = C.PF_HOLD / "raw" / "ship_r8" / f"SHIP_R8_{TREE}.json"
REQUIRED = ("1", "1b", "2", "3", "4", "5", "6", "7")
CL1_TEXT = {"H2CN_DEPLOY8": ("(1) [AMENDMENT 4] the frozen H2cn stage's sha256 == the registered one; the registration precedes the first eval-side stage output; "
                             "the fit used no eval139 / train4572 clip; H2CN_PERSURFACE and RETIME_DEPLOY8 identities hold"),
            "H2CN_CAP8": ("(1) [AMENDMENT 4b] everything Amendment 4's clause 1 states (frozen stage sha256 == registered; registration precedes the first eval-side output; "
                          "fresh-clip disjoint; H2CN_PERSURFACE and RETIME_DEPLOY8 identities) AND the Amendment 4b seal == its file (re-hashed now) AND K-CAP-ID, K-CAP-COMPLY, K-CAP-MONO"),
            "FACTORED_DEPLOY8": ("(1) [AMENDMENT 5] the registered table's sha256 checks (unchanged) AND the identity controls (P_ss := P_os == OSONLY bit for bit; "
                                 "RETIME_DEPLOY8 == registered) AND both input traces' sha256 + final + 4,754 windows + hook counters")}
CL1_TEXT["H2CNL_DEPLOY8"] = ("(1) [AMENDMENT 4c] everything Amendment 4's clause 1 states (H2cn stage sha256 == registered; registration precedes the first eval-side "
                             "output; fresh-clip disjoint; H2CN_PERSURFACE and RETIME_DEPLOY8 identities) AND the Amendment 4c seal == its file (re-hashed now) AND the "
                             "H2cn-L stage sha256 == REGISTERED_H2CNL_STAGE.sha256 (re-hashed now) and its registration precedes the first eval-side output AND the "
                             "freeze controls (K-H2CN-REPRO, K-JOIN, K-FIT, K-RELOAD, K-DETERM, K-DISJ, K-MUT) AND K-JOIN on the eval rows")
CL1_TEXT["H2CNL_MUT8"] = CL1_TEXT["H2CNL_DEPLOY8"].replace("H2cn-L stage sha256 == REGISTERED_H2CNL_STAGE", "H2cn-L MUTATION stage sha256 == REGISTERED_H2CNL_MUT_STAGE")
DEVIATIONS_4C = [
    "P1-1 mutation construction (not specified by the amendment): the 6 LON6-softmax columns of the FIT rows are permuted by a seeded (seed 0) cross-episode "
    "derangement (h2cnl_freeze.cross_episode_derangement): a bijection, every row receives a DIFFERENT episode's columns, column marginals preserved exactly",
    "P1-2 K-FIT for the L / MUT stages has no earlier PASSED fit to equal (Amendment 4 compared against H2CN_RESULT.json): replaced by literals -- n_fresh == the "
    "registered fit's, n_params == 68100 / 68358 (H2cn's + 6 x 128), wd in the grid, 1 <= epochs <= 200",
    "P1-3 K-H2CN-REPRO for the mutation stage is shown by base-array digest equality (X[:, :10], Z, H, eid == the arrays whose 6-column-dropped fit reproduced "
    "the registered H2cn stage bit-for-bit), not by a third identical refit",
    "P1-4 h2cnl_freeze.py carries a DEV-ONLY --smoke-dir mode (3 epochs, outputs outside the package); used once, outputs only in the agent scratch dir",
    "P1-5 README_SHIP_R8_SUB.md and ship_r8_sub_common.py were NOT edited (both pinned by MANIFEST_4B)",
    "P1-6 h2cnl_freeze.DIFF.txt is large because of the new controls; the fit change itself is one call (fit_surface: X := concat[X, softmax(lon6)])",
    "P2-7 the forward / verdict / reproduction are COPIES (ship_r8_4c_forward.py, ship_r8_4c_verdict.py, read_4c.py; diffs beside) because the originals are "
    "pinned by MANIFEST_4B; ship_r8_sub_tac.py is used unchanged",
    "P2-8 the eval-side K-JOIN adds two cross-checks beyond the amendment (extras v0 == ctx v0; extras nav_true.traj == plan8 bit for bit) and a mutation "
    "(join on the NEXT window) that must read RED by those cross-checks alone",
    "P2-9 RT.forward's transient work dir lives on F:/refcv9_4c_scratch (C: nearly full) and is removed at the end; every output is under out_4c/",
    "P2-10 read_4c.py normalises backslashes BEFORE the out/ token (read_4b did it last) and skips the npz meta_json `tool` field (the producer is the 4c copy)",
    "P2-11 the mutation stage H2CNL_MUT8 is fed the TRUE eval-side LON6 softmax (only its FIT rows were deranged)",
    "P2-12 L1 / L2 'both inference seeds' is read as a conjunction (seed 0 AND seed 1)",
    "P2-13 bars_4c.py's verdict WORDING was edited after its first run: the first BARS_4C.json read 'VOID ... fix the instrument', which misnamed an "
    "ATTRIBUTION-control failure (K-MUT-L1) as an instrument failure; the edit only separates the two cases and names the amendment's branches -- every "
    "cell, bar flag and control is the same mechanical computation (the first file was overwritten)",
]
CL1B_A5 = "(1b) [AMENDMENT 5] FEED PARITY: the os-only rolls use the per-frame feed (perframe_full), asserted from the rolls' own manifests; not shown -> INCOMPLETE"
CL1_TEXT["HSGATE8"] = ("(1) [PREREG_HSGATE] everything Amendment 4's clause 1 states (H2cn stage sha256 == registered; registration precedes the first eval-side "
                       "output; fresh-clip disjoint; H2CN_PERSURFACE and RETIME_DEPLOY8 identities) AND the PREREG_HSGATE seal == its file (re-hashed now) AND the "
                       "gate sha256 == REGISTERED_HSGATE.sha256 (re-hashed now) and its registration precedes the first eval-side output AND K_FIT_HSGATE, K_ID0, "
                       "K_ID1, K_EVALFEAT, K_LANDED, K_GATE_SURF, K_TRACE AND the Amendment 5 input traces unchanged since the forward recorded them")
CL1B_HSG = "(1b) [PREREG_HSGATE] FEED PARITY of BOTH halves: Amendment 4's (1b) (the registered FEED_PARITY of the battery rolls) AND Amendment 5's (1b) (the os-only and SELFSHARP rolls' own manifests perframe_full)"
from deviations_hsgate import DEVIATIONS as DEVIATIONS_HSG  # noqa: E402   (confirmed by the Master Mind in the PHASE 2 GO)


def exact_equal(a, b, path: str = "") -> list:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b), key=str):
            out += [f"{path}/{k}: one side only"] if (k not in a or k not in b) else exact_equal(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += exact_equal(x, y, f"{path}[{i}]")
        return out
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return []
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def load(p):
    return B._load(p)


def clauses_on(arm: str, fd, tac) -> list:
    """The registered clauses (2)-(5) read on `arm` (B.ARM swapped for the call, restored after)."""
    old = B.ARM
    B.ARM = arm
    try:
        cs = [B.clause_e(2, "S2", fd, "minus_ha0_ext_pose", arm), B.clause_e(3, "S6", fd, "minus_ha0_ext_pose", arm), B.clause4(fd), B.clause5(tac)]
    finally:
        B.ARM = old
    # LABEL ONLY: the registered base's clause-5 PASS reason is a hard-coded string naming RETIME_DEPLOY8 (_ship_r8_verdict_base.py:198).  The cells it judged
    # are the `tac` file's, whose clause-5 arm the base itself requires to equal B.ARM (= `arm`, line 184).  The status and evidence are untouched.
    c5 = cs[3]
    if arm != "RETIME_DEPLOY8" and "RETIME_DEPLOY8" in str(c5.get("reason", "")):
        c5.setdefault("evidence", {})["reason_text_registered_verbatim"] = c5["reason"]
        c5["reason"] = c5["reason"].replace("RETIME_DEPLOY8", arm) + f" [arm label substituted; the registered reason string is hard-coded; tac_file_arm == {arm} enforced by the base]"
    return cs


def registered_inputs(reg_ship: dict) -> dict:
    """The registered verdict's own inputs, each re-hashed against the sha256 the registered verdict recorded."""
    out = {}
    for k in ("pr", "pp", "tf", "parity_s0", "parity_s1", "deploy_sha", "frozen", "provenance", "gate", "registered", "fd8", "tac"):
        p = reg_ship["inputs"][k]
        got = C.sha256_file(p) if Path(p).exists() else None
        out[k] = {"path": p, "sha256_now": got, "sha256_registered": reg_ship["inputs_sha256"].get(k), "equal": bool(got is not None and got == reg_ship["inputs_sha256"].get(k))}
    return out


def retime_identity(fd, tac_retime_rec, reg_ship) -> dict:
    """RETIME_DEPLOY8 re-read through the sub forward + the TAC device == the registered SHIP_R8 cells (clauses 2-5 evidence) EXACTLY."""
    tac_r = load((tac_retime_rec or {}).get("tac_file")) if tac_retime_rec else None
    mine = clauses_on("RETIME_DEPLOY8", fd, tac_r)
    reg = {str(c["id"]): c for c in reg_ship["clauses"]}
    rows = {}
    for c in mine:
        r = reg.get(str(c["id"]))
        mm = ["registered clause absent"] if r is None else exact_equal({"status": c["status"], "evidence": c["evidence"]}, {"status": r["status"], "evidence": r["evidence"]})
        rows[str(c["id"])] = {"status_mine": c["status"], "status_registered": (r or {}).get("status"), "n_mismatches": len(mm), "first_mismatches": mm[:5]}
    fwd = ((fd or {}).get("controls") or {}).get("K_RETIME_identity_vs_registered_forward_deploy8")
    tacid = ((tac_retime_rec or {}).get("identity_vs_registered_TAC") or {}).get("pass")
    ok = bool(all(v["n_mismatches"] == 0 for v in rows.values()) and fwd is True and tacid is True)
    return {"pass": ok, "clauses_2_5_vs_registered": rows, "forward_blocks_and_bars_exact": fwd, "tac_cells_controls_exact": tacid,
            "known": "None means the input is absent (INCOMPLETE), False means shown different (FAIL)"}


def _status(bad, unk):
    return "FAIL" if bad else ("INCOMPLETE" if unk else "PASS")


def clause1_a4(fd, rid) -> dict:
    ctl = (fd or {}).get("controls") or {}
    bad, unk = [], []
    reg_tok = C.read_seal(C.WORK / "REGISTERED_H2CN_STAGE.sha256")
    now = C.sha256_file(C.STAGE_PT) if C.STAGE_PT.exists() else ""
    rec = (fd or {}).get("h2cn_stage") or {}
    ev = {"stage_sha256_now": now, "registered_sha256": reg_tok, "forward_stage_sha256": rec.get("sha256"), "forward_registered_sha256": rec.get("registered")}
    if fd is None:
        unk.append("the forward file is absent")
    if len(reg_tok) != 64 or len(now) != 64:
        unk.append("the registered stage sha256 or the stage file is not readable as 64 hex")
    elif not (now == reg_tok == rec.get("sha256") == rec.get("registered")):
        bad.append("the frozen stage file / forward record / registration disagree on the sha256")
    kreg = ctl.get("K_REG_detail") or {}
    g = kreg.get("gate") or {}
    ev["registration_gate_passed_epoch"], ev["first_eval_output_epoch"] = g.get("passed_epoch"), kreg.get("first_eval_output_epoch")
    ev["registration_file_mtime_epoch"] = g.get("registration_file_mtime_epoch")
    for nm, key in (("K_REG (registration precedes the first eval-side output)", "K_REG"), ("K_DISJ (fresh-clip disjointness)", "K_DISJ"),
                    ("K_PERSURF (H2CN_PERSURFACE S2 identity within 1e-4)", "K_PERSURF_identity_S2_vs_N1"),
                    ("the battery's own forward controls ALL_PASS", "ALL_PASS")):
        v = ctl.get(key)
        ev[key] = v
        if v is False:
            bad.append(nm + " is false")
        elif v is not True:
            unk.append(nm + " not shown")
    ev["K_PERSURF_cells"] = (ctl.get("K_PERSURF_detail") or {}).get("cells_H2cn_minus_H0_S2_compliant_ADE")
    ev["RETIME_identity"] = rid
    if rid["forward_blocks_and_bars_exact"] is False or rid["tac_cells_controls_exact"] is False or any(r["n_mismatches"] for r in rid["clauses_2_5_vs_registered"].values()):
        bad.append("RETIME_DEPLOY8 re-read != the registered cells")
    elif not rid["pass"]:
        unk.append("RETIME_DEPLOY8 identity not shown")
    if fd is not None and (B._boot_ok(fd) is not True or sorted((fd.get("seeds") or {}).keys()) != ["0", "1"]):
        bad.append("the forward is not B 2000 on inference seeds 0 and 1")
    st = _status(bad, unk)
    return {"id": 1, "text": CL1_TEXT["H2CN_DEPLOY8"], "status": st, "reason": "; ".join(bad + unk) if (bad or unk) else
            "stage sha256 == registered (re-hashed now); registration precedes the first eval-side output; fresh-clip disjoint; H2CN_PERSURFACE and RETIME_DEPLOY8 identities hold; B 2000, seeds 0 and 1",
            "evidence": ev}


def clause1_4b(fd, rid) -> dict:
    """AMENDMENT 4b clause 1 = Amendment 4's clause 1 (unchanged function, run on the cap8 forward) AND the 4b seal re-hashed now == the forward's record AND the three
    K-CAP controls.  Amendment 4's own conjunction is NOT edited; its status/reason are carried into this one."""
    base = clause1_a4(fd, rid)
    ctl = (fd or {}).get("controls") or {}
    bad, unk = [], []
    if base["status"] == "FAIL":
        bad.append("Amendment 4 clause-1 conjunction: " + base["reason"])
    elif base["status"] != "PASS":
        unk.append("Amendment 4 clause-1 conjunction: " + base["reason"])
    now = C.check_amendment_4b_seal()                                    # SystemExit if the sealed text is not the file (never a silent pass)
    rec = ((fd or {}).get("amendments") or {}).get("4b") or {}
    ev = dict(base["evidence"])
    ev.update(amendment_4b_seal_now=now, amendment_4b_seal_forward_record=rec, forward_is_cap8_run=(fd or {}).get("cap8"))
    if fd is None:
        unk.append("the forward file is absent")
    elif (fd or {}).get("cap8") is not True:
        bad.append("the forward is not a --cap8 run")
    if not rec:
        unk.append("the forward carries no Amendment 4b seal record")
    elif rec.get("sha256") != now["sha256"]:
        bad.append("the Amendment 4b file hashed now != the sha256 the forward recorded")
    for nm, key in (("K-CAP-ID (lim := +inf -> bit-equal H2CN_DEPLOY8)", "K_CAP_ID"), ("K-CAP-COMPLY (compliance-row exceed_rate_strict == 0 on every set)", "K_CAP_COMPLY"),
                    ("K-CAP-MONO (s_cap non-decreasing)", "K_CAP_MONO")):
        v = ctl.get(key)
        ev[key] = v
        if v is False:
            bad.append(nm + " is false")
        elif v is not True:
            unk.append(nm + " not shown")
    kd = ctl.get("K_CAP_detail") or {}
    ev["K_CAP_per_surface_seed"] = kd.get("per_surface_seed")
    ev["K_CAP_compliance_of_H2CN_CAP8"] = kd.get("compliance_of_H2CN_CAP8")
    st = _status(bad, unk)
    return {"id": 1, "text": CL1_TEXT["H2CN_CAP8"], "status": st, "reason": "; ".join(bad + unk) if (bad or unk) else
            "Amendment 4's clause-1 conjunction holds on the cap8 forward; the Amendment 4b seal == its file (re-hashed now) and == the forward's record; "
            "K-CAP-ID, K-CAP-COMPLY, K-CAP-MONO all true", "evidence": ev}


def clause1_4c(fd, rid, arm: str = "H2CNL_DEPLOY8") -> dict:
    """AMENDMENT 4c clause 1 (see the module docstring).  Amendment 4's own conjunction is NOT edited; its status/reason are carried into this one."""
    import h2cnl_freeze as HL
    base = clause1_a4(fd, rid)
    ctl = (fd or {}).get("controls") or {}
    bad, unk = [], []
    if base["status"] == "FAIL":
        bad.append("Amendment 4 clause-1 conjunction: " + base["reason"])
    elif base["status"] != "PASS":
        unk.append("Amendment 4 clause-1 conjunction: " + base["reason"])
    now = HL.check_amendment_4c_seal()                                   # SystemExit if the sealed text is not the file (never a silent pass)
    rec = ((fd or {}).get("amendments") or {}).get("4c") or {}
    mut = arm == "H2CNL_MUT8"
    regf = C.WORK / ("REGISTERED_H2CNL_MUT_STAGE.sha256" if mut else "REGISTERED_H2CNL_STAGE.sha256")
    ptf = HL.MUT_PT if mut else HL.STAGE_L_PT
    reg_tok = C.read_seal(regf)
    sha_now = C.sha256_file(ptf) if ptf.exists() else ""
    frec = (fd or {}).get("h2cnl_mut_stage" if mut else "h2cnl_stage") or {}
    ev = dict(base["evidence"])
    ev.update(amendment_4c_seal_now=now, amendment_4c_seal_forward_record=rec, forward_is_h2cnl_run=(fd or {}).get("h2cnl"),
              h2cnl_stage_sha256_now=sha_now, h2cnl_registered_sha256=reg_tok, h2cnl_forward_stage_record=frec)
    if fd is None:
        unk.append("the forward file is absent")
    elif (fd or {}).get("h2cnl") is not True:
        bad.append("the forward is not an --h2cnl run")
    if not rec:
        unk.append("the forward carries no Amendment 4c seal record")
    elif rec.get("sha256") != now["sha256"]:
        bad.append("the Amendment 4c file hashed now != the sha256 the forward recorded")
    if len(reg_tok) != 64 or len(sha_now) != 64:
        unk.append(f"the registered {arm} stage sha256 or the stage file is not readable as 64 hex")
    elif not (sha_now == reg_tok == frec.get("sha256") == frec.get("registered")):
        bad.append(f"the {arm} stage file / forward record / registration disagree on the sha256")
    kreg = "K_REG_H2CNL_MUT" if mut else "K_REG_H2CNL"
    for nm, key in ((f"{kreg} (registration precedes the first eval-side output)", kreg), ("K_FREEZE_H2CNL (freeze controls ALL_PASS, stage sha256s match)", "K_FREEZE_H2CNL"),
                    ("K_JOIN_EVAL (every eval row joins exactly one extras row of the same forward; mutation RED)", "K_JOIN_EVAL")):
        v = ctl.get(key)
        ev[key] = v
        if v is False:
            bad.append(nm + " is false")
        elif v is not True:
            unk.append(nm + " not shown")
    ev["K_REG_H2CNL_detail"] = (ctl.get("K_REG_H2CNL_detail") or {}).get(arm)
    ev["K_FREEZE_H2CNL_detail"] = ctl.get("K_FREEZE_H2CNL_detail")
    st = _status(bad, unk)
    return {"id": 1, "text": CL1_TEXT[arm], "status": st, "reason": "; ".join(bad + unk) if (bad or unk) else
            f"Amendment 4's clause-1 conjunction holds on the --h2cnl forward; the Amendment 4c seal == its file (re-hashed now) and == the forward's record; "
            f"the {arm} stage sha256 == registered (re-hashed now); {kreg}, K_FREEZE_H2CNL, K_JOIN_EVAL all true", "evidence": ev}


def clause1b_a4(reg_ship) -> dict:
    ri = registered_inputs(reg_ship)
    par = {s: load(ri[f"parity_s{s}"]["path"]) for s in ("0", "1")}
    c = VPF.clause1b(par, TREE, VPF.expected_windows(TREE))
    c["evidence"]["sha256_recheck_vs_registered_verdict"] = {k: ri[k] for k in ("parity_s0", "parity_s1")}
    if not (ri["parity_s0"]["equal"] and ri["parity_s1"]["equal"]):
        c.update(status="INCOMPLETE", reason="the FEED_PARITY files no longer hash to the registered verdict's record: the registered parity cannot be shown | " + c["reason"])
    else:
        c["reason"] = "unchanged (same rolls): the registered FEED_PARITY files hash to the registered verdict's record | " + c["reason"]
    return c


def clause1_a5(fd, rid, reg_ship, pins) -> dict:
    ri = registered_inputs(reg_ship)
    reg_sha = VPF.read_registered(ri["registered"]["path"])
    tf = load(ri["tf"]["path"])
    prov, gate = load(ri["provenance"]["path"]), load(ri["gate"]["path"])
    reg = VPF.registration(reg_sha, C.sha256_file(ri["frozen"]["path"]), prov, gate, tf, VPF.roll_mtimes(TREE))
    dsha = Path(ri["deploy_sha"]["path"]).read_text().strip()
    c = VPF.clause1_pf(fd, tf, dsha, reg)                                   # the registered table's sha256 checks, UNCHANGED (B.FROZEN_SHA256 := registered)
    base_status = c["status"]
    ctl = (fd or {}).get("controls") or {}
    bad, unk = [], []
    if base_status == "FAIL":
        bad.append("registered table checks: " + c["reason"])
    elif base_status != "PASS":
        unk.append("registered table checks: " + c["reason"])
    for nm, key in (("K_SSOS (P_ss := P_os == OSONLY bit for bit)", "K_SSOS_identity_bit_for_bit"), ("K_TRACE (final / 4,754 / counters)", "K_TRACE")):
        v = ctl.get(key)
        if v is False:
            bad.append(nm + " is false")
        elif v is not True:
            unk.append(nm + " not shown")
    if rid["forward_blocks_and_bars_exact"] is False or rid["tac_cells_controls_exact"] is False or any(r["n_mismatches"] for r in rid["clauses_2_5_vs_registered"].values()):
        bad.append("RETIME_DEPLOY8 re-read != the registered cells")
    elif not rid["pass"]:
        unk.append("RETIME_DEPLOY8 identity not shown")
    trs = {}
    for s, rec in ((ctl.get("K_TRACE_detail") or {}).get("inputs") or {}).items():
        for nm in ("l1_base", "selfsharp"):
            r = rec.get(nm) or {}
            f = r.get("file")
            now = C.sha256_file(f) if f and Path(f).exists() else None
            pin = ((pins or {}).get(f"s{s}") or {}).get(nm)
            trs[f"s{s}|{nm}"] = {"sha256_forward_record": r.get("sha256"), "sha256_now": now, "sha256_pin": pin}
            if r.get("sha256") is None or now is None:
                unk.append(f"s{s} {nm}: sha256 not recorded / file unreadable")
            elif now != r["sha256"] or (pin is not None and pin != now):
                bad.append(f"s{s} {nm}: the trace changed since it was recorded")
    if not trs:
        unk.append("no input-trace record in the forward")
    c.update(id=1, text=CL1_TEXT["FACTORED_DEPLOY8"], status=_status(bad, unk), reason="; ".join(bad + unk) if (bad or unk) else
             "registered table checks PASS; P_ss := P_os reproduces OSONLY bit for bit; RETIME_DEPLOY8 == registered; both traces final, 4,754 windows, all forced, none unarmed, batch 1, sha256 unchanged")
    c["evidence"].update(registered_table_clause1_status=base_status, RETIME_identity=rid, traces=trs, trace_checks=(ctl.get("K_TRACE_detail") or {}).get("checks"),
                         trace_pins_file_given=pins is not None)
    return c


def clause1b_a5(fd) -> dict:
    d = ((fd or {}).get("controls") or {}).get("K_FEED_detail")
    if not d:
        return {"id": "1b", "text": CL1B_A5, "status": "INCOMPLETE", "reason": "the forward carries no roll-manifest feed record", "evidence": {}}
    bad, unk = [], []
    for s, rolls in d.items():
        for nm, r in rolls.items():
            if r.get("manifest_nav_feed_mode") is None or r.get("roll_pf_nav_feed") is None:
                unk.append(f"s{s} {nm}: feed mode not stated")
            elif not r.get("perframe_full"):
                bad.append(f"s{s} {nm}: manifest {r.get('manifest_nav_feed_mode')} / roll {r.get('roll_pf_nav_feed')} ok={r.get('roll_pf_ok')} n={r.get('manifest_n_windows')}")
    if not any("selfsharp" in rolls for rolls in d.values()):
        unk.append("the SELFSHARP rolls' manifests were not read (identity-only run)")
    return {"id": "1b", "text": CL1B_A5, "status": _status(bad, unk), "reason": "; ".join(bad + unk) if (bad or unk) else
            "both seeds: the L1 os-only and the SELFSHARP rolls' own manifests read nav_feed perframe_full (roll pf ok, 4,754 windows)", "evidence": d}


def clause1_hsgate(fd, rid) -> dict:
    """PREREG_HSGATE clause 1 (see the module docstring).  Amendment 4's own conjunction is NOT edited; its status/reason are carried into this one."""
    base = clause1_a4(fd, rid)
    ctl = (fd or {}).get("controls") or {}
    bad, unk = [], []
    if base["status"] == "FAIL":
        bad.append("Amendment 4 clause-1 conjunction: " + base["reason"])
    elif base["status"] != "PASS":
        unk.append("Amendment 4 clause-1 conjunction: " + base["reason"])
    now = HG.check_prereg()                                             # SystemExit if the sealed text is not the file (never a silent pass)
    rec = ((fd or {}).get("amendments") or {}).get("PREREG_HSGATE") or {}
    hrec = (fd or {}).get("hsgate") or {}
    gfile, regf = HG.WORK / "HSGATE_FROZEN.json", HG.WORK / "REGISTERED_HSGATE.sha256"
    reg_tok = C.read_seal(regf)
    gate_now = C.sha256_file(gfile) if gfile.exists() else ""
    ev = dict(base["evidence"])
    ev.update(prereg_seal_now=now, prereg_seal_forward_record=rec, gate_sha256_now=gate_now, gate_registered_sha256=reg_tok, gate_forward_record=hrec.get("gate_sha256"))
    if fd is None:
        unk.append("the forward file is absent")
    elif not hrec:
        bad.append("the forward is not an --hsgate run")
    if not rec:
        unk.append("the forward carries no PREREG_HSGATE seal record")
    elif rec.get("sha256") != now["sha256"]:
        bad.append("PREREG_HSGATE hashed now != the sha256 the forward recorded")
    if len(reg_tok) != 64 or len(gate_now) != 64:
        unk.append("the registered gate sha256 or the gate file is not readable as 64 hex")
    elif not (gate_now == reg_tok == hrec.get("gate_sha256") == hrec.get("registered")):
        bad.append("the gate file / forward record / registration disagree on the sha256")
    for nm, key in (("K_REG_HSGATE (gate registration precedes the first eval-side output)", "K_REG_HSGATE"), ("K_FIT_HSGATE (fit controls ALL_PASS)", "K_FIT_HSGATE"),
                    ("K_ID0 (g := 0 reproduces H2CN_DEPLOY8 bit for bit)", "K_ID0"), ("K_ID1 (g := 1 reproduces FACTORED_DEPLOY8 bit for bit)", "K_ID1"),
                    ("K_EVALFEAT (eval features from the K-LABSRC extras keys; joins exact, mutation RED)", "K_EVALFEAT"),
                    ("K_LANDED (H2CN / FACTORED positions == the landed packages)", "K_LANDED"), ("K_GATE_SURF (S2 / S6 decisions agree)", "K_GATE_SURF"),
                    ("K_TRACE (Amendment 5 traces final / 4,754 / counters)", "K_TRACE")):
        v = ctl.get(key)
        ev[key] = v
        if v is False:
            bad.append(nm + " is false")
        elif v is not True:
            unk.append(nm + " not shown")
    trs = {}
    for s, r0 in ((ctl.get("K_TRACE_detail") or {}).get("inputs") or {}).items():
        for nm in ("l1_base", "selfsharp"):
            r = r0.get(nm) or {}
            f = r.get("file")
            nw = C.sha256_file(f) if f and Path(f).exists() else None
            trs[f"s{s}|{nm}"] = {"sha256_forward_record": r.get("sha256"), "sha256_now": nw}
            if r.get("sha256") is None or nw is None:
                unk.append(f"s{s} {nm}: sha256 not recorded / file unreadable")
            elif nw != r["sha256"]:
                bad.append(f"s{s} {nm}: the trace changed since it was recorded")
    if not trs:
        unk.append("no input-trace record in the forward")
    ev["traces"] = trs
    ev["K_ID_detail"] = ctl.get("K_ID_detail")
    ev["K_REG_HSGATE_detail"] = ctl.get("K_REG_HSGATE_detail")
    st = _status(bad, unk)
    return {"id": 1, "text": CL1_TEXT["HSGATE8"], "status": st, "reason": "; ".join(bad + unk) if (bad or unk) else
            "Amendment 4's clause-1 conjunction holds on the --hsgate forward; PREREG_HSGATE seal == its file (re-hashed now) and == the forward's record; the gate "
            "sha256 == registered (re-hashed now); K_REG_HSGATE, K_FIT_HSGATE, K_ID0, K_ID1, K_EVALFEAT, K_LANDED, K_GATE_SURF, K_TRACE all true; traces unchanged",
            "evidence": ev}


def clause1b_hsgate(reg_ship, fd) -> dict:
    a, b = clause1b_a4(reg_ship), clause1b_a5(fd)
    sts = (a["status"], b["status"])
    st = "FAIL" if "FAIL" in sts else ("INCOMPLETE" if "INCOMPLETE" in sts else "PASS")
    return {"id": "1b", "text": CL1B_HSG, "status": st, "reason": f"A4 (1b): {a['status']} -- {a['reason']} || A5 (1b): {b['status']} -- {b['reason']}",
            "evidence": {"A4_1b": a, "A5_1b": b}}


def overall(clauses) -> str:
    by = {str(c.get("id")): c.get("status") for c in clauses}
    if any(i not in by or by[i] not in ("PASS", "FAIL", "INCOMPLETE", "N/A") for i in REQUIRED):
        return "FAIL" if "FAIL" in by.values() else "INCOMPLETE"
    st = [by[i] for i in REQUIRED]
    return "FAIL" if "FAIL" in st else ("INCOMPLETE" if "INCOMPLETE" in st else "PASS")


def nr_rows(fd, arm) -> dict:
    """The per-row FZR-NR readings of `arm` (compliant set) -- to name WHICH surface a clause-(4) failure sits on (Amendment 4 s5)."""
    b = ((fd or {}).get("bars") or {}).get(f"compliant|{arm}") or {}
    m = (b.get("FZR-NR") or {}).get("metrics") or {}
    return {k: v.get("verdict", v.get("status")) for k, v in m.items()}


def fmt(c) -> str:
    return B._f(c)


def render(v: dict) -> str:
    L = [f"OVERALL: {v['overall']} -- SHIP-R8 on {v['arm']} (AMENDMENT {v['amendment']}), tree {TREE}", "", f"# SHIP_R8_{v['arm']}", "",
         f"tier: {v['tier']}", f"estimator: {v['estimator']}", "one training seed; T1 open loop (not driving performance)", "",
         "| clause | status | reading |", "|---|---|---|"]
    for c in v["clauses"]:
        L.append(f"| {c['text']} | **{c['status']}** | {c['reason']} |")
    L += ["", "## evidence (scored arm)", ""]
    for c in v["clauses"]:
        if str(c["id"]) in ("2", "3"):
            L.append(f"* clause ({c['id']}) {v['arm']} - ha0_ext_pose: " + " | ".join(f"{s}: {fmt(x)}" for s, x in (c["evidence"].get("cells") or {}).items()))
        if str(c["id"]) == "4":
            L.append(f"* FZR-NR per family {c['evidence'].get('FZR-NR_per_family')}; FZR-LAT HOLDS {c['evidence'].get('FZR-LAT_HOLDS')} "
                     f"(separated worse at any seed: {c['evidence'].get('FZR-LAT_separated_worse_at_any_seed')})")
        if str(c["id"]) == "5" and c["evidence"].get("cells"):
            L += ["", "| TAC cell (arm - refcv7-50,400; worse+separated = CI upper < 0) | delta [lo, hi] |", "|---|---|"] + [f"| {n} | {fmt(x)} |" for n, x in c["evidence"]["cells"].items()]
    L += ["", "## beside (never scored)", ""]
    for arm, b in v["beside"]["arms"].items():
        L.append(f"* {arm}: " + "; ".join(f"({k}) {x['status']}" for k, x in b["clauses"].items()) +
                 (f" | E1 " + " | ".join(f"{s}: {fmt(x)}" for s, x in (b.get("E1") or {}).items()) if b.get("E1") else "") +
                 (f" | E3 " + " | ".join(f"{s}: {fmt(x)}" for s, x in (b.get("E3") or {}).items()) if b.get("E3") else ""))
    if v["beside"].get("running_max_changed_windows"):
        L.append(f"* running-maximum changed windows per (surface, seed): {v['beside']['running_max_changed_windows']}")
    if v["beside"].get("clamp_touched_windows"):
        L.append(f"* windows changed by the 4b clamp per (surface, seed): {v['beside']['clamp_touched_windows']}")
    if v.get("next_step"):
        L += ["", f"NEXT (committed in the amendment): {v['next_step']}"]
    if v.get("disclosed_deviations"):
        L += ["", "## disclosed deviations (accepted by the Master Mind 2026-10-26 for P1-1..P1-6)", ""] + [f"* {d}" for d in v["disclosed_deviations"]]
    L += ["", f"inputs sha256: {json.dumps(v['inputs_sha256'])}", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("H2CN_DEPLOY8", "H2CN_CAP8", "FACTORED_DEPLOY8", "H2CNL_DEPLOY8", "HSGATE8"))
    ap.add_argument("--identity-only", action="store_true")
    ap.add_argument("--forward", required=True)
    ap.add_argument("--arms-dir", required=True)
    ap.add_argument("--trace-pins", default=None)
    ap.add_argument("--out-dir", default=str(C.WORK / "out" / "ship_r8"))
    a = ap.parse_args()
    seals = C.check_amendment_seals()
    reg_ship = json.loads(REG_SHIP.read_text(encoding="utf-8"))
    ri = registered_inputs(reg_ship)
    fd = load(a.forward)
    ad = Path(a.arms_dir)
    tac_ret_rec = load(ad / "RETIME_DEPLOY8" / f"TAC_RECORD_{TREE}.json")
    rid = retime_identity(fd, tac_ret_rec, reg_ship)
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    if a.identity_only:
        ctl = (fd or {}).get("controls") or {}
        rec = {"tool": "ship_r8_sub_verdict.py --identity-only", "generated": time.strftime("%FT%T%z"), "mode": (fd or {}).get("mode"), "forward": a.forward,
               "forward_sha256": C.sha256_file(a.forward), "RETIME_identity": rid,
               "forward_controls": {k: v for k, v in ctl.items() if isinstance(v, bool)}, "registered_inputs_recheck": ri, "amendments": seals,
               "scored_arm_computed": False, "note": "controls only; no SHIP-R8 clause is read on a scored arm in this record"}
        rec["ALL_IDENTITIES_PASS"] = bool(rid["pass"] and ctl.get("ALL_PASS") is True)
        p = od / f"IDENTITY_{(fd or {}).get('mode', 'x')}_{TREE}.json"
        p.write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
        chk = json.loads(p.read_text(encoding="utf-8"))
        print(f"IDENTITY {p} ALL_IDENTITIES_PASS={chk['ALL_IDENTITIES_PASS']} RETIME={rid['pass']}", flush=True)
        print(f"ZZSUBIDENT_{'PASS' if chk['ALL_IDENTITIES_PASS'] else 'FAIL'}ZZ", flush=True)
        return 0
    if not a.arm:
        ap.error("--arm is required unless --identity-only")
    arm = a.arm
    amend = {"H2CN_DEPLOY8": "4", "H2CN_CAP8": "4b", "FACTORED_DEPLOY8": "5", "H2CNL_DEPLOY8": "4c", "HSGATE8": "HSG"}[arm]
    h2 = amend in ("4", "4b", "4c", "HSG")
    if amend == "HSG":
        HG.check_prereg()                                                 # refuses before anything is read
        if fd is None or not fd.get("hsgate"):
            raise SystemExit("[hsgate-verdict] HSGATE8 needs an --hsgate forward")
        rod = od.resolve()
        if not (rod == HG.WORK.resolve() or HG.WORK.resolve() in rod.parents):
            raise SystemExit(f"[hsgate-verdict] REFUSING: --out-dir {od} is outside {HG.WORK}")
    if fd is not None and fd.get("mode") != ("h2cn" if h2 else "factored"):
        raise SystemExit(f"[sub-verdict] the forward file is mode {fd.get('mode')!r}, the arm {arm} needs Amendment {amend}'s forward")
    if amend == "4b":
        C.check_amendment_4b_seal()                                       # refuses before anything is read
    if amend == "4c":
        import h2cnl_freeze as HL
        HL.check_amendment_4c_seal()                                      # refuses before anything is read
    if fd is not None and fd.get("h2cnl") and od.resolve() in ((C.WORK / "out" / "ship_r8").resolve(), (C.WORK / "out_4b" / "ship_r8").resolve()):
        raise SystemExit("[sub-verdict] REFUSING: an --h2cnl forward must be read into a NEW --out-dir (the landed out/ship_r8 and out_4b/ship_r8 files are never overwritten)")
    if fd is not None and fd.get("cap8") and od.resolve() == (C.WORK / "out" / "ship_r8").resolve():
        raise SystemExit("[sub-verdict] REFUSING: a --cap8 forward must be read into a SEPARATE --out-dir (the landed out/ship_r8 files are never overwritten)")
    tac = load(ad / arm / f"TAC_{TREE}.json")
    pr, pp = load(ri["pr"]["path"]), load(ri["pp"]["path"])
    pins = load(a.trace_pins) if a.trace_pins else None
    c1 = clause1_hsgate(fd, rid) if amend == "HSG" else (clause1_4b(fd, rid) if amend == "4b" else (clause1_a4(fd, rid) if amend == "4" else (clause1_4c(fd, rid, arm) if amend == "4c" else clause1_a5(fd, rid, reg_ship, pins))))
    c1b = clause1b_hsgate(reg_ship, fd) if amend == "HSG" else (clause1b_a4(reg_ship) if h2 else clause1b_a5(fd))
    c25 = clauses_on(arm, fd, tac)
    c6, c7 = B.clause6(pr), B.clause7(pp)
    for c_, k in ((c6, "pr"), (c7, "pp")):
        c_["evidence"]["sha256_recheck_vs_registered_verdict"] = ri[k]
        if not ri[k]["equal"] and c_["status"] in ("PASS", "N/A"):
            c_.update(status="INCOMPLETE", reason=f"the registered {k} file no longer hashes to the registered verdict's record | " + c_["reason"])
    clauses = [c1, c1b] + c25 + [c6, c7]
    ov = overall(clauses)
    beside_arms = {"4": ("H2CN_NEAR", "H2CN_PERSURFACE", "RETIME_DEPLOY8"), "4b": ("H2CN_DEPLOY8", "H2CN_NEAR", "H2CN_PERSURFACE", "RETIME_DEPLOY8"),
                   "5": ("OSONLY_DEPLOY8", "RETIME_DEPLOY8"), "4c": ("H2CN_DEPLOY8", "H2CNL_MUT8", "H2CN_NEAR", "H2CN_PERSURFACE", "RETIME_DEPLOY8"),
                   "HSG": ("H2CN_DEPLOY8", "FACTORED_DEPLOY8", "RETIME_DEPLOY8")}[amend]
    bes = {"arms": {}}
    for b_arm in beside_arms:
        cs = clauses_on(b_arm, fd, load(ad / b_arm / f"TAC_{TREE}.json"))
        bes["arms"][b_arm] = {"clauses": {str(x["id"]): {"status": x["status"], "reason": x["reason"]} for x in cs},
                              "E1": cs[0]["evidence"].get("cells"), "E3": cs[1]["evidence"].get("cells"), "FZR_NR_rows": nr_rows(fd, b_arm)}
    bes["scored_arm_FZR_NR_rows"] = nr_rows(fd, arm)
    bes["compliance_row"] = (fd or {}).get("compliance_row")
    if h2:
        bes["running_max_changed_windows"] = {k: v.get("running_max_changed_windows_H2CN_DEPLOY8") for k, v in ((fd or {}).get("per_window_extra") or {}).items()}
    if amend == "4c":
        bes["arms"]["H2CNL_MUT8"]["clauses"]["1"] = {k: v for k, v in clause1_4c(fd, rid, "H2CNL_MUT8").items() if k in ("status", "reason")}
        bes["h2cnl_per_window"] = {k: {kk: vv for kk, vv in (v.get("h2cnl") or {}).items() if kk not in ("join", "join_mutation_next_window")}
                                   for k, v in ((fd or {}).get("per_window_extra") or {}).items()}
    if amend == "HSG":
        bes["hsgate_F_rate"] = ((fd or {}).get("hsgate") or {}).get("F_rate")
        bes["hsgate_per_window"] = {k: {kk: vv for kk, vv in (v.get("hsgate") or {}).items() if kk not in ("join", "join_mutation_next_window", "landed")}
                                    for k, v in ((fd or {}).get("per_window_extra") or {}).items()}
    if amend == "4b":
        bes["clamp_touched_windows"] = {k: {kk: (v.get("cap8") or {}).get(kk) for kk in ("n_changed_by_clamp", "share_changed_by_clamp", "n_changed_by_clamp_compliant",
                                                                                         "share_changed_by_clamp_compliant", "n_windows", "n_compliant", "n_nonfinite_lim")}
                                        for k, v in ((fd or {}).get("per_window_extra") or {}).items()}
    nxt = None
    if amend == "4":
        if ov == "PASS":
            nxt = "PASS -> H2cn ships as refcv8's stage (the PI's (a)); registry + GOALS_AND_CLAIMS updated; NavSim only with a separate go."
        elif ov == "FAIL":
            fails = [str(c["id"]) for c in clauses if c["status"] == "FAIL"]
            nr_fail = [k for k, vv in bes["scored_arm_FZR_NR_rows"].items() if str(vv).startswith("FAIL")]
            c4ev, c5ev = clauses[4]["evidence"], clauses[5]["evidence"]
            s6_only = bool(fails and set(fails) <= {"3", "4", "5"}
                           and ("4" not in fails or all(k.startswith("S6:") for k in nr_fail))          # FZR-LAT is an S6 lateral row by construction
                           and ("5" not in fails or all(n.startswith("S6|") for n in (c5ev.get("separated_worse_cells") or []))))
            bes["next_step_reading_rows"] = {"FZR_NR_failing_rows": nr_fail, "FZR_LAT_HOLDS": c4ev.get("FZR-LAT_HOLDS"),
                                             "TAC_separated_worse_cells": c5ev.get("separated_worse_cells")}
            near_holds = all(bes["arms"]["H2CN_NEAR"]["clauses"].get(i, {}).get("status") == "PASS" for i in fails)
            nxt = ("FAIL of H2CN_DEPLOY8 (reported first); it fails only on S6 clauses while H2CN_NEAR holds them -> a sealed AMENDMENT 4b making H2CN_NEAR the arm is the next step"
                   if (s6_only and near_holds) else "FAIL -> the shipped RETIME stage stays; the failing clause(s) " + ", ".join(fails) + " name the next lever")
            bes["next_step_reading"] = {"failing_clauses": fails, "s6_only": bool(s6_only), "H2CN_NEAR_holds_them": bool(near_holds)}
    elif amend == "4b":
        fails = [str(c["id"]) for c in clauses if c["status"] == "FAIL"]
        nxt = {"PASS": "PASS -> the PI chooses the shipped stage between H2CN_DEPLOY8 (Amendment 4 PASS, ~3x RETIME's ceiling exceedance) and H2CN_CAP8 (0 exceedance by construction); "
                       "the Master Mind's recommendation is H2CN_CAP8. NavSim only with a separate go.",
               "FAIL": "FAIL -> clause(s) " + ", ".join(fails) + " named; H2CN_DEPLOY8 stands as Amendment 4 registered it, with its compliance row put to the PI.",
               "INCOMPLETE": "INCOMPLETE -> a clause could not be shown (see the reasons); H2CN_DEPLOY8 stands as Amendment 4 registered it."}.get(ov)
    elif amend == "HSG":
        fails = [str(c["id"]) for c in clauses if c["status"] == "FAIL"]
        nxt = ("this verdict is bar G3 of PREREG_HSGATE (SHIP-R8 OVERALL " + ov + (": clause(s) " + ", ".join(fails) + " named" if fails else "") +
               "); the committed next step reads G1 + G2 + G3 + G4 together -> result/HSGATE_VERDICT.json (hsgate_bars.py)")
    elif amend == "4c":
        fails = [str(c["id"]) for c in clauses if c["status"] == "FAIL"]
        nxt = ("this verdict is bar L3 of Amendment 4c (SHIP-R8 OVERALL " + ov + (": clause(s) " + ", ".join(fails) + " named" if fails else "") +
               "); the committed next step reads L1 + L2 + L3 together -> BARS_4C.json (bars_4c.py)")
    else:
        nxt = {"PASS": "PASS -> the factored decode is a refcv8 deliverable candidate; build the in-model implementation (two planner passes on one trunk forward) "
                       "and its bit-equal control before anything ships; NavSim only with a separate go.",
               "FAIL": "FAIL -> the failing clause names the lever; the decode stays a refcv9 design input (H-R9-SSGUARD-1)."}.get(ov)
    v = {"tool": "ship_r8_hsgate_verdict.py", "generated": time.strftime("%FT%T%z"), "tree": TREE, "arm": arm, "amendment": amend, "overall": ov, "clauses": clauses,
         "per_clause": {str(c["id"]): c["status"] for c in clauses}, "beside": bes, "next_step": nxt, "amendments": seals,
         "tier": "T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)",
         "estimator": "paired episode-cluster bootstrap B 2000, bootstrap seed 0 (pair_read_r8.paired_cell -> taniteval.ci), over EPISODES with the models FIXED: "
                      "answers 'another draw of episodes' only -- not another training run (one training seed), not another inference run (seeds 0 and 1 reported separately)",
         "registered_verdict": {"file": str(REG_SHIP), "sha256": C.sha256_file(REG_SHIP), "base_copy_sha256": VPF.BASE_SHA256},
         "inputs": {"forward": a.forward, "tac_arm": str(ad / arm / f"TAC_{TREE}.json"), "tac_retime_record": str(ad / "RETIME_DEPLOY8" / f"TAC_RECORD_{TREE}.json"),
                    "trace_pins": a.trace_pins, **{("tac_registered_RETIME_DEPLOY8" if k == "tac" else k): r["path"] for k, r in ri.items()}},
         "script_sha256": SCRIPT_SHA}
    if amend == "4c":
        v["disclosed_deviations"] = DEVIATIONS_4C
    if amend == "HSG":
        v["disclosed_deviations"] = DEVIATIONS_HSG
    v["inputs_sha256"] = {k: (C.sha256_file(p) if p and Path(p).exists() else None) for k, p in v["inputs"].items()}
    jp = od / f"SHIP_R8_{arm}.json"
    jp.write_text(json.dumps(v, indent=1, default=str) + "\n", encoding="utf-8")
    (od / f"SHIP_R8_{arm}.md").write_text(render(v), encoding="utf-8")
    chk = json.loads(jp.read_text(encoding="utf-8"))                          # assert on the ARTIFACT
    assert chk["overall"] in ("PASS", "FAIL", "INCOMPLETE") and [str(c["id"]) for c in chk["clauses"]] == list(REQUIRED)
    print(f"OVERALL {chk['overall']} {arm} per_clause={chk['per_clause']}", flush=True)
    print(f"ZZSUBVERDICT_{chk['overall']}ZZ {arm}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
