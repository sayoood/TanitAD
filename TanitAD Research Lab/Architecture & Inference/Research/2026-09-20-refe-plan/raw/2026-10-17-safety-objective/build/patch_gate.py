#!/usr/bin/env python3
"""patch_gate.py <PK dir>: refe/gate_checks.py (blob 14e32f1b...) and refe/launch_gate.py (blob e6e000ca...) -> the SAFEOBJ gate (SPEC_SAFEOBJ 5):
G-DVB entries for the five --epo-con flags (all `built`) + the data event's `epo_con` block, gate G13 (required iff --epo AND --epo-con != off, re-judged from raw fields), the profile
`refe_rls1d` (arms R0d, R1d, R1o; per-arm registered values and must-name lists), and the refusal of any --epo-con other than off on every older profile.
Every existing mutation anchor of test_rl_r0c.py / test_rl_s1b.py keeps occurring exactly once (the new text is added BESIDE the anchored lines, never inside them)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import patchlib as PL  # noqa: E402

GC_BASE = "14e32f1b1f2712d6093c9fa07e7c0cd1a470aa4d"
LG_BASE = "e6e000cab54c8ead215b51a1c6110d293451302d"

# =============================================================================================================================================== gate_checks.py
GC_DVB_ANCHOR = "    \"epo_ref\": \"built\",                # R0c: the KL anchor's reference policy; observed through the data event's epo.ref = scorer_finetune.epo_reference_built (what the selector RETURNS)\n"
GC_DVB_NEW = GC_DVB_ANCHOR + '''    # SAFEOBJ (--epo-con g5; SPEC_SAFEOBJ 5.1): all `built` -- observed through the data event's `epo_con` block (SCORER_EPO_CON_KEYS), which EpoDual.built_facts reads off the object the run holds
    "epo_con": "built", "epo_con_eps": "built", "epo_con_lr": "built", "epo_con_max": "built", "epo_con_lam0": "built",
'''

GC_KEYS_ANCHOR = "SCORER_NAMESPACE_FUNCS = (\"main\", \"epo_refusals\", \"_g_probe\", \"aux_s1_refusals\")"
GC_KEYS_NEW = '''SCORER_EPO_CON_KEYS = ("mode", "comps", "cols", "eps", "lr", "max", "lam0", "ref")     # the keys of the data event's `epo_con` block (SAFEOBJ; every --epo run writes it, an OFF run included)
SCORER_EPO_CON_COMPS = {"g5": ("ttc", "nc")}          # what each mode is REGISTERED to constrain (PREREG_SAFEOBJ 2): the gate's own statement, independent of the trainer's table
SCORER_EPO_CON_DEFAULTS = {"eps": 0.0, "lr": 4.0, "max": 10.0, "lam0": 0.0}          # the registered numbers; a trainer that predates SAFEOBJ cannot have built any other
''' + GC_KEYS_ANCHOR

GC_GATES_ANCHOR = '"G10": "teacher_lane_in_bank", "G11": "epo_live_and_identity", "G12": "aux_drv_live_and_identity"}\n'
GC_GATES_NEW = '"G10": "teacher_lane_in_bank", "G11": "epo_live_and_identity", "G12": "aux_drv_live_and_identity", "G13": "epo_con_live_and_identity"}\n'

GC_REQ_ANCHOR = "def scorer_required_gates(ns) -> list:\n"
GC_JUDGE13 = '''def judge_g13(g13, ns) -> list:
    """G13 (scorer_finetune --preflight, --epo-con g5) re-judged from its RAW FIELDS against the argv's namespace -- never from its own `ok`.  [] = proved.
    The L_con-only backward is finite and non-zero on every scorer module of arm B; the lambda-0 and switch-off controls read EXACTLY 0.0 (value and gradient); arm A reads EXACTLY 0.0 from L_con
    while its own BCE gradient (the same probe) reads non-zero; at step 0 the two policies coincide; the dual's synthetic sequences land on the closed form EXACTLY (0.3125 at the registered eta 4,
    lambda0 0), floor at 0.0 and cap at lam_max; the EPO_CON block round-trips bit-equal; no regression switch was set."""
    if not isinstance(g13, dict):
        return ["G13 is not a dict"]
    r = []
    mods = set(SCORER_GROUPS)
    fin = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)          # noqa: E731
    if g13.get("ok") is not True:
        r.append("G13 reports ok != true")
    if g13.get("mutate"):
        r.append(f"G13 was produced under regression switch(es) {g13['mutate']}: a regression arm can never count")
    on, lam0, off = (g13.get(k) or {} for k in ("per_module_on", "per_module_lam0", "per_module_off"))
    if set(on) != mods or not all(fin(x) and x > 0.0 for x in on.values()):
        r.append(f"G13: the L_con-only backward is not finite and non-zero on every scorer module {sorted(mods)} of arm B: {on}")
    if not (lam0 and off) or any(x != 0.0 for d in (lam0, off) for x in d.values()) or g13.get("value_lam0") != 0.0 or g13.get("value_off") != 0.0:
        r.append("G13: the zero controls (lambda 0 / --epo-con off) do not read EXACTLY 0.0 (value and gradient) -- the probe is not proved able to read zero")
    a_reads = [g13.get(k) or {} for k in ("A_per_module_on", "A_per_module_lam0", "A_per_module_off")]
    if any(set(d) != mods or any(x != 0.0 for x in d.values()) for d in a_reads):
        r.append(f"G13: arm A received a gradient from L_con (it must read EXACTLY 0.0 on every module): {a_reads[0]}")
    a_ctrl = g13.get("A_bce_control") or {}
    if set(a_ctrl) != mods or not all(fin(x) and x > 0.0 for x in a_ctrl.values()):
        r.append(f"G13: arm A's own BCE gradient does not read non-zero in the same probe (a zero on A would be a property of the probe, not of the term): {a_ctrl}")
    want = list(SCORER_EPO_CON_COMPS.get(getattr(ns, "epo_con", "off"), ()))
    gap, tol = g13.get("c_gap") or {}, g13.get("c_gap_tol")
    if sorted(gap) != sorted(want) or not fin(tol) or not all(fin(v) and v <= tol for v in gap.values()):
        r.append(f"G13: the step-0 identity |c_A - c_B| <= {tol} does not hold on {want}: {gap}")
    dr = g13.get("dual_reach") or {}
    if sorted(dr.get("comps") or ()) != sorted(want):
        r.append(f"G13: the dual probe covers {dr.get('comps')}, not the declared constraint {want}")
    for k, flag in (("eta", "epo_con_lr"), ("lam0", "epo_con_lam0"), ("lam_max", "epo_con_max")):
        if not (fin(dr.get(k)) and _close(dr.get(k), getattr(ns, flag, None))):
            r.append(f"G13: the dual probe ran with {k} {dr.get(k)!r}, not the argv's --{flag.replace('_', '-')} {getattr(ns, flag, None)!r}")
    if all(fin(dr.get(k)) for k in ("eta", "lam0", "lam_max", "viol")) and isinstance(dr.get("steps"), int) and dr["steps"] >= 1 and dr["viol"] > 0.0:
        exp = dr["lam0"]
        for _ in range(dr["steps"]):
            exp = min(dr["lam_max"], max(0.0, exp + dr["eta"] * dr["viol"]))
        if (dr["eta"], dr["lam0"], dr["lam_max"], dr["viol"], dr["steps"]) == (4.0, 0.0, 10.0, 2.0 ** -7, 10) and exp != 0.3125:
            r.append(f"G13: the closed form of the registered dual sequence is {exp}, not 0.3125 (the judge's own arithmetic is off)")
        for c in want:
            if (dr.get("got") or {}).get(c) != exp:
                r.append(f"G13: dual ascent on {c}: {(dr.get('got') or {}).get(c)!r} != the closed form {exp} -- the dual step does not reach lambda")
            if (dr.get("floor_got") or {}).get(c) != 0.0:
                r.append(f"G13: a negative violation did not floor lambda_{c} at EXACTLY 0.0: {(dr.get('floor_got') or {}).get(c)!r}")
            if (dr.get("cap_got") or {}).get(c) != dr["lam_max"]:
                r.append(f"G13: a large violation did not cap lambda_{c} at lam_max {dr['lam_max']} EXACTLY: {(dr.get('cap_got') or {}).get(c)!r}")
    else:
        r.append("G13: the dual probe's raw fields are missing or malformed")
    ck = g13.get("checkpoint") or {}
    if ck.get("block_equal") is not True or ck.get("restored_equal") is not True or not {"lam", "update", "cap_hits"} <= set(ck.get("keys") or ()):
        r.append(f"G13: the EPO_CON checkpoint block does not round-trip bit-equal with lam/update/cap_hits: {ck}")
    return r


'''
GC_REQ_NEW = GC_JUDGE13 + GC_REQ_ANCHOR

GC_REQDOC_ANCHOR = "    G11 iff --epo, G12 iff --aux-drv).\"\"\"\n"
GC_REQDOC_NEW = "    G11 iff --epo, G12 iff --aux-drv, G13 iff --epo AND --epo-con != off).\"\"\"\n"

GC_REQ2_ANCHOR = "    if ns.epo:\n        req.append(\"G11\")\n"
GC_REQ2_NEW = GC_REQ2_ANCHOR + "    if ns.epo and getattr(ns, \"epo_con\", \"off\") != \"off\":\n        req.append(\"G13\")\n"

GC_ROWS_SIG_ANCHOR = "def scorer_dvb_rows(ns, data: dict) -> list:\n"
GC_ROWS_SIG_NEW = "def scorer_dvb_rows(ns, data: dict, profile=None) -> list:\n"

GC_ROWS_ANCHOR = "    add(\"aux_drv\", bool(ns.aux_drv), \"aux_drv\" in data, \"the data event carries an `aux_drv` block iff --aux-drv\")\n"
GC_ROWS_NEW = '''    if ns.epo:                                       # SAFEOBJ: the constraint, compared with the BUILT facts of the EpoDual the run holds (never an echo of the namespace)
        import rl_reward                             # the pinned component order: one source of truth with the trainer
        mode = getattr(ns, "epo_con", "off")
        comps = list(SCORER_EPO_CON_COMPS.get(mode, ()))
        blk = data.get("epo_con")
        legacy = blk is None and mode == "off" and not (profile or {}).get("epo_con_block")
        why = " (legacy trainer: no epo_con block, so no constraint can have been built)" if legacy else ""
        if legacy:
            blk = {"mode": "off", "comps": [], "cols": [], "ref": "armA_detached", **SCORER_EPO_CON_DEFAULTS}
        blk = blk or {}
        for k in SCORER_EPO_CON_KEYS:
            if k == "mode":
                add("epo_con", mode, blk.get("mode"), "data event: epo_con.mode (derived from the BUILT comps tuple)" + why)
            elif k == "comps":
                add("epo_con", comps, blk.get("comps"), "data event: epo_con.comps (the constrained sub-scores)" + why)
            elif k == "cols":
                add("epo_con", [rl_reward.COMPONENTS.index(c) for c in comps], blk.get("cols"), "data event: epo_con.cols (the label columns, from rl_reward.COMPONENTS)" + why)
            elif k == "ref":
                add("epo_con", "armA_detached", blk.get("ref"), "data event: epo_con.ref (what epo_con_reference RETURNS: arm A's logits, detached)" + why)
            else:
                add("epo_con_" + k, getattr(ns, "epo_con_" + k), blk.get(k), f"data event: epo_con.{k}" + why)
''' + GC_ROWS_ANCHOR

GC_CALL_ANCHOR = "        rows = scorer_dvb_rows(ns, data)\n"
GC_CALL_NEW = "        rows = scorer_dvb_rows(ns, data, profile)\n"

GC_EDITS = [
    (GC_DVB_ANCHOR, GC_DVB_NEW),
    (GC_KEYS_ANCHOR, GC_KEYS_NEW),
    (GC_GATES_ANCHOR, GC_GATES_NEW),
    (GC_REQ_ANCHOR, GC_REQ_NEW),
    (GC_REQDOC_ANCHOR, GC_REQDOC_NEW),
    (GC_REQ2_ANCHOR, GC_REQ2_NEW),
    (GC_ROWS_SIG_ANCHOR, GC_ROWS_SIG_NEW),
    (GC_ROWS_ANCHOR, GC_ROWS_NEW),
    (GC_CALL_ANCHOR, GC_CALL_NEW),
]

# =============================================================================================================================================== launch_gate.py
LG_CONST_ANCHOR = 'RLS1C_SUITE = RLS1B_SUITE + [{"script": "test_rl_r0c.py", "token": "ZZTEST_PASS"}]\n'
LG_CONST_NEW = LG_CONST_ANCHOR + '''# RL-S1d / SAFEOBJ (raw/2026-10-17-safety-objective/PREREG_SAFEOBJ.md + SPEC_SAFEOBJ.md; needs the PI's budget decision before any launch).  ONE lever moves against R0: constrained EPO,
# `--epo-con g5` (L_con = sum_k lambda_k (c^k_A - eps - c^k_B) on TTC and NC, lambda by dual ascent).  The argv is R0's banked argv (sha256 24451b62...) plus the five --epo-con* flags.
# The switch's parser default is `off`, which would silently reproduce R0: so R0d / R1d NAME all five flags and R1o (the matched lever-off control on the SAME trainer bytes) NAMES `--epo-con off`;
# every older profile registers `--epo-con off` and so refuses any other value.  Order R0d, R1d, R1o (a K3 kill stops the chain; R1o's K3 is OBSERVE-ONLY, rls1_checks.py).
RLS1D_ARM_SEEDS = {"R0d": 0, "R1d": 1, "R1o": 1}
RLS1D_ARM_CLASS = {"R0d": "R", "R1d": "R", "R1o": "R"}
RLS1D_CON_ON = {"epo_con": "g5", "epo_con_eps": 0.0, "epo_con_lr": 4.0, "epo_con_max": 10.0, "epo_con_lam0": 0.0}          # PREREG_SAFEOBJ 2: eps 0, eta 4, lambda_max 10, lambda0 0
RLS1D_CON_OFF = {"epo_con": "off"}
RLS1D_ARM_EPO_REGISTERED = {"R0d": RLS1D_CON_ON, "R1d": RLS1D_CON_ON, "R1o": RLS1D_CON_OFF}
RLS1D_CON_FLAGS = ("--epo-con", "--epo-con-eps", "--epo-con-lr", "--epo-con-max", "--epo-con-lam0")
RLS1D_ARM_MUST_NAME = {"R0d": RLS1D_CON_FLAGS, "R1d": RLS1D_CON_FLAGS, "R1o": ("--epo-con",)}
RLS1D_CON_TERMS = ("con", "con_lam_ttc", "con_lam_nc", "con_g_ttc", "con_g_nc")                # the trainer's metric rows must emit these on a constrained arm (R1o: none, and their presence there is a FAIL)
RLS1D_ARM_CON_TERMS = {"R0d": RLS1D_CON_TERMS, "R1d": RLS1D_CON_TERMS, "R1o": ()}
RLS1D_SUITE = RLS1C_SUITE + [{"script": "test_rl_safeobj.py", "token": "ZZTEST_PASS"}]
'''

LG_AUX_ANCHOR = '"--epo-l", "--epo-lam-l",\n                   "--epo-terms-train", "--epo-terms-heldout", "--epo-ref")\n'
LG_AUX_NEW = ('"--epo-l", "--epo-lam-l", "--epo-con", "--epo-con-eps", "--epo-con-lr", "--epo-con-max", "--epo-con-lam0",\n'
              '                   "--epo-terms-train", "--epo-terms-heldout", "--epo-ref")\n')

LG_PROFILE_ANCHOR = '                          "epo_registered": RLS1C_EPO_REGISTERED, "must_name": RLS1C_MUST_NAME}\n'
LG_PROFILE_NEW = LG_PROFILE_ANCHOR + '''# RL-S1d = refe_rls1 with ITS arm table (R0d, R1d, R1o), the registered --epo-con values PER ARM (arm_epo_registered), the flags each arm must NAME (arm_must_name) and the extra metric columns a
# constrained arm must emit (arm_con_terms).  `epo_con_block`: the data event MUST carry the `epo_con` block (G-DVB cannot read a legacy absence as "off" here: R1o's OFF is itself observed).
PROFILES["refe_rls1d"] = {**PROFILES["refe_rls1"], "name": "refe_rls1d", "suite": RLS1D_SUITE, "arm_class": RLS1D_ARM_CLASS, "arm_seeds": RLS1D_ARM_SEEDS,
                          "arm_epo_registered": RLS1D_ARM_EPO_REGISTERED, "arm_must_name": RLS1D_ARM_MUST_NAME, "arm_con_terms": RLS1D_ARM_CON_TERMS, "epo_con_block": True}
'''

LG_REFUSE_ANCHOR = '    out += [f"scorer_finetune refuses this argv: {m}" for m in mod.epo_refusals(ns, parser)]\n'
LG_REFUSE_NEW = '    out += [f"scorer_finetune refuses this argv: {m}" for m in mod.epo_refusals(ns, parser) + mod.epo_con_refusals(ns, parser)]          # SAFEOBJ: epo_con_refusals beside epo_refusals\n'

LG_MUSTNAME_ANCHOR = '        if cls == "R?":\n'
LG_MUSTNAME_NEW = '''        for f in prof.get("arm_must_name", {}).get(declared, ()):                  # SAFEOBJ: the flags THIS arm must CARRY (R0d / R1d: the five --epo-con*; R1o: --epo-con off, whose default would hide the control)
            if f not in argv:
                out.append(f"{name} requires {f} NAMED for {who} (a flag whose default reproduces the old arm must be named, or the arm did not test the lever); the argv does not carry it")
''' + LG_MUSTNAME_ANCHOR

LG_REG_ANCHOR = "        if cls == \"R2\":\n            reg.update(RLS1_R2_REGISTERED)\n"
LG_REG_NEW = LG_REG_ANCHOR + "        reg = {\"epo_con\": \"off\", **reg, **prof.get(\"arm_epo_registered\", {}).get(declared, {})}          # SAFEOBJ: every profile registers --epo-con off unless its arm table says otherwise\n"

LG_TERMS_ANCHOR = "    return out, (RLS1_EPO_TERMS if cls != \"C\" else ())\n"
LG_TERMS_NEW = ("    con_terms = tuple(prof.get(\"arm_con_terms\", {}).get(declared, ())) if cls != \"C\" else ()          # SAFEOBJ: a constrained arm's rows must also emit the con* columns\n"
                "    return out, ((RLS1_EPO_TERMS + con_terms) if cls != \"C\" else ())\n")

LG_CONSUME_ANCHOR = "    g12 = next((v for k, v in gates.items() if prefix[k] == \"G12\"), None)\n"
LG_CONSUME_NEW = ("    g13 = next((v for k, v in gates.items() if prefix[k] == \"G13\"), None)\n"
                  "    if g13 is not None:                                      # G13 (--epo-con): re-judged from its raw fields too (SAFEOBJ)\n"
                  "        reasons += C.judge_g13(g13, ns)\n" + LG_CONSUME_ANCHOR)

LG_CHOICES_ANCHOR = "choices=sorted(set(RLS1_ARM_CLASS) | set(RLS1B_ARM_CLASS) | set(RLS1B_NEEDS_PI) | set(RLS1C_ARM_CLASS))"
LG_CHOICES_NEW = "choices=sorted(set(RLS1_ARM_CLASS) | set(RLS1D_ARM_CLASS) | set(RLS1B_ARM_CLASS) | set(RLS1B_NEEDS_PI) | set(RLS1C_ARM_CLASS))"

LG_HELP_ANCHOR = "refe_rls1c: R0c; refe_aux_s1: N-s0..H2-s2;"
LG_HELP_NEW = "refe_rls1c: R0c; refe_rls1d: R0d R1d R1o; refe_aux_s1: N-s0..H2-s2;"

LG_MSG_ANCHOR = "(refe_rls1, refe_rls1b, refe_rls1c, refe_aux_s1, refe_aux_s1c)"
LG_MSG_NEW = "(refe_rls1, refe_rls1b, refe_rls1c, refe_rls1d, refe_aux_s1, refe_aux_s1c)"

LG_EDITS = [
    (LG_CONST_ANCHOR, LG_CONST_NEW),
    (LG_AUX_ANCHOR, LG_AUX_NEW),
    (LG_PROFILE_ANCHOR, LG_PROFILE_NEW),
    (LG_REFUSE_ANCHOR, LG_REFUSE_NEW),
    (LG_MUSTNAME_ANCHOR, LG_MUSTNAME_NEW),
    (LG_REG_ANCHOR, LG_REG_NEW),
    (LG_TERMS_ANCHOR, LG_TERMS_NEW),
    (LG_CONSUME_ANCHOR, LG_CONSUME_NEW),
    (LG_CHOICES_ANCHOR, LG_CHOICES_NEW),
    (LG_HELP_ANCHOR, LG_HELP_NEW),
    (LG_MSG_ANCHOR, LG_MSG_NEW),
]


def main():
    pk = PL.pk_arg()
    for fn, edits, base in (("gate_checks.py", GC_EDITS, GC_BASE), ("launch_gate.py", LG_EDITS, LG_BASE)):
        r = PL.patch_file(os.path.join(pk, "refe", fn), edits, expect_before=base)
        print({k: v for k, v in r.items() if k != "reversals"})


if __name__ == "__main__":
    main()
