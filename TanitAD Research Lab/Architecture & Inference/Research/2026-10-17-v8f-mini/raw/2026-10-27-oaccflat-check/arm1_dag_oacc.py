#!/usr/bin/env python3
"""ARM 1 -- DAG-OACC = DAG_s's training argv VERBATIM + `--w-acc 49.6969`.  Does the REAL trainer's in-process launch gate (v8f_mini_train.py ... --selfcheck-only, CPU, synthetic world,
REAL model dimensions, REAL M2b_s0 checkpoint as --init-ckpt, a bank made by the REAL generator from that checkpoint) accept it, with BOTH levers live at once?

    CUDA_VISIBLE_DEVICES=-1 python arm1_dag_oacc.py            # full run
    CG_MUTANT=acc_line python arm1_dag_oacc.py                # a mutant must turn it RED (see mutants.py)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cg_lib as L  # noqa: E402
from cg_lib import case, res  # noqa: E402

TR, torch = L.TR, L.torch
S = int(__import__("os").environ.get("CG_SEED", "0"))
OUT = L.ROOT / "out" / (f"arm1_s{S}" + (("__MUT_" + L.MUT_RECORD["mutant"]) if L.MUT_RECORD else ""))
WORLD = L.WORK / f"world_s{S}"
BANK = L.WORK / f"bank_m2b_s{S}"


def main():
    L.WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    ck = L.M2B_CKPT[S]
    sha_ck = L.sha256_file(ck)
    case(f"init checkpoint on the dev box is the banked M2b_s{S} (sha256 {L.SHA_M2B[S][:8]}...)", sha_ck == L.SHA_M2B[S], sha_ck[:12])
    world = L.make_world(WORLD, seed=S)
    # ---- the bank: the REAL generator, from the REAL M2b_s0 weights (its header names that checkpoint's sha256: K-B2)
    if (BANK / "bank.json").exists():
        sha = L.sha256_file(BANK / "bank.json")
        case("bank (cached from an earlier run in work/): bank.json present", True, f"sha256 {sha[:16]}")
    else:
        _, sha, log, rc, secs = L.make_bank(world, ck, BANK)
        (L.WORK / f"bank_m2b_s{S}.log").write_text(log, encoding="utf-8")
        case("bank tool: 2 shards through the real v8f_dagger_bank.py + merge, generator = REAL M2b_s0 checkpoint, rc 0", rc == 0, f"rc {rc} {secs:.0f}s {log[-250:]!r}")
        if rc != 0:
            return L.finish(f"arm1_s{S}")
    man = json.loads((BANK / "bank.json").read_text(encoding="utf-8"))
    prov = json.loads((BANK / "PROVENANCE.json").read_text(encoding="utf-8"))
    case("bank header names the REAL M2b_s0 checkpoint (K-B2 side), coverage clears 0.5", man["common"]["ckpt_sha256"] == L.SHA_M2B[S] and not man["partial"] and not man["coverage"]["blocked"],
         {"bank.json common.ckpt_sha256": man["common"]["ckpt_sha256"][:12], "controls_PASS": {k: v.get("PASS") for k, v in (prov.get("controls") or {}).items()}, "coverage": man["coverage"]})

    # ---- the two argvs, derived mechanically from DAG_s0's REAL argv.json
    real = L.DAG_ARGV[S]
    assert dict(L.pairs(real))['--dagger-seed'] == str(S) and dict(L.pairs(real))['--seed'] == str(S)
    a_treat = L.to_gate_argv(real, OUT / "dag_oacc", init_ckpt=ck, bank=BANK, bank_sha=sha, extra=[("--w-acc", repr(L.W_ACC))])
    a_ctrl = L.to_gate_argv(real, OUT / "dag_ctrl", init_ckpt=ck, bank=BANK, bank_sha=sha)
    bad, added = L.carried_verbatim(real, a_treat, allowed_extra={"--w-acc"})
    ft_has_wacc = dict(L.pairs(a_treat)).get("--w-acc") == repr(L.W_ACC) and a_treat.count("--w-acc") == 1
    case("DAG-OACC gate argv = DAG's real argv token-for-token except the declared data/path re-pointing, plus ONLY --w-acc", (not bad and added == [] and ft_has_wacc), {"bad": bad, "added": added})
    bad, added = L.carried_verbatim(real, a_ctrl)
    case("control gate argv = DAG's real argv token-for-token except the declared data/path re-pointing (NO --w-acc)", not bad and not added, {"bad": bad, "added": added})
    ft, fc = dict(L.pairs(a_treat)), dict(L.pairs(a_ctrl))
    diff = sorted(k for k in set(ft) | set(fc) if ft.get(k) != fc.get(k))
    case("one-lever rule on the two gate argvs: they differ in --w-acc and --out ONLY", diff == ["--out", "--w-acc"], diff)
    need = ["--init-ckpt", "--steps", "--lr", "--warmup", "--p-gt-start", "--p-gt-end", "--ego-perturb-p", "--dagger-p", "--dagger-bank", "--dagger-bank-sha256", "--dagger-seed", "--w-acc"]
    case("every DAG lever + --w-acc is NAMED on the treatment argv (a flag whose default disables it must be named)", all(k in ft for k in need) and ft["--dagger-p"] == "0.1" and ft["--ego-perturb-p"] == "0", {k: ft.get(k) for k in need})

    # ---- THE REAL GATE: treatment
    L.COMPOSED.clear()
    rc, sc = L.gate(OUT / "dag_oacc", a_treat)
    case("REAL gate accepts DAG-OACC (rc 0, declared_vs_built PASS)", rc == 0 and sc and sc.get("declared_vs_built") == "PASS", f"rc {str(rc)[:600]}")
    comp_t = dict(L.COMPOSED)
    rc0, sc0 = L.gate(OUT / "dag_ctrl", a_ctrl)
    case("REAL gate accepts the control (DAG, no --w-acc) (rc 0, declared_vs_built PASS)", rc0 == 0 and sc0 and sc0.get("declared_vs_built") == "PASS", f"rc {str(rc0)[:600]}")
    comp_c = dict(L.COMPOSED)
    (OUT / "composed_probe_treatment.json").write_text(json.dumps(comp_t, indent=1), encoding="utf-8")
    (OUT / "composed_probe_control.json").write_text(json.dumps(comp_c, indent=1), encoding="utf-8")
    if not (sc and sc0):
        return L.finish(f"arm1_s{S}")

    lv, lv0 = sc["levers"], sc0["levers"]
    ap, ap0 = sc["acc_probe"], sc0["acc_probe"]
    dp = sc["dagger_probe"]
    # G-DVB: both lever families listed, with the argv's values
    case("G-DVB (selfcheck levers) lists w_acc = 49.6969 and the dagger_* levers with the argv's values",
         lv.get("w_acc") == L.W_ACC and lv.get("dagger_p") == 0.1 and lv.get("dagger_bank_sha256") == sha and lv.get("dagger_bank_sha256_declared") == sha
         and lv["dagger"].get("dagger_seed") == S and lv["ego_perturb"].get("ego_perturb_p") == 0.0,
         {"w_acc": lv.get("w_acc"), "dagger_p": lv.get("dagger_p"), "dagger": {k: (str(v)[:24] if isinstance(v, str) else v) for k, v in lv["dagger"].items()}, "ego_perturb": lv["ego_perturb"]})
    ic = lv["init_ckpt"]
    case("init is the REAL M2b_s0 checkpoint: sha256 601cb435..., applied, 0 tensors differ after the strict load", ic.get("sha256") == L.SHA_M2B[S] and ic.get("applied") is True and ic.get("n_mismatch_after_load") == 0 and ic.get("source_step") == 19164,
         {k: ic.get(k) for k in ("sha256", "applied", "n_mismatch_after_load", "source_step", "n_tensors")})
    case("control: levers.w_acc == 0.0 and dagger levers identical to the treatment's (only w_acc moved)",
         lv0.get("w_acc") == 0.0 and all(lv.get(k) == lv0.get(k) for k in ("dagger_p", "dagger_bank_sha256", "dagger_bank_sha256_declared", "dagger", "ego_perturb", "init_ckpt", "lr_schedule", "o_interface", "code_mix", "w", "params", "n_params_trainable")),
         {k: (lv.get(k) == lv0.get(k)) for k in ("dagger_p", "dagger_bank_sha256", "dagger", "ego_perturb", "init_ckpt", "lr_schedule", "o_interface", "code_mix", "w", "params")})

    # C-O2 (OACC's G-LIVE entry): the term is finite and non-zero, reaches the total with the declared weight, and O's head gets a gradient from it
    case("acc_probe PASS at the declared weight; acc finite and > 0", ap.get("pass") is True and ap["w_acc_declared"] == L.W_ACC and ap["acc"] > 0.0, {k: ap.get(k) for k in ("w_acc_declared", "acc", "total_w", "total_0")})
    case("acc_probe: total(w) - total(0) == w x acc (the weight reaches the loss the trainer backprops)", abs(ap["contribution"] - ap["w_x_acc"]) <= 1e-3 * ap["w_x_acc"] + 1e-5, {k: ap.get(k) for k in ("contribution", "w_x_acc")})
    case("acc_probe: O's head (head_o) gets a NON-ZERO gradient from the term; controls (repeat, frozen prior, WM) read EXACTLY 0.0",
         ap["head_o_grad_diff"] > 0.0 and ap["control_repeat_total_diff"] == 0.0 and ap["control_repeat_grad_diff"] == 0.0 and ap["control_prior_grad_diff"] == 0.0 and ap["control_wm_grad_diff"] == 0.0,
         {k: ap.get(k) for k in ("head_o_grad_diff", "read_o_grad_diff", "control_repeat_total_diff", "control_repeat_grad_diff", "control_prior_grad_diff", "control_wm_grad_diff")})
    # the DAgger half, in the SAME gate run
    case("dagger_probe PASS in the same gate run: swapped batch moves loss and gradient; p = 0 control EXACTLY 0.0",
         dp.get("pass") is True and dp["loss_diff"] != 0.0 and dp["grad_diff_all"] > 0.0 and dp["control_p0_loss_diff"] == 0.0 and dp["control_p0_grad_diff"] == 0.0 and dp["control_p0_same_object"] is True,
         {k: dp.get(k) for k in ("p_declared", "n_selected_at_p1", "loss_diff", "grad_diff_all", "control_p0_loss_diff", "control_p0_grad_diff")})
    kb = sc["info"].get("dagger") or {}
    case("K-B2 (bank <-> init checkpoint), K-B3 (train-only) and COVERAGE >= 0.5 were read and passed", bool(kb) and kb["kb2"].get("init_ckpt_sha256") == L.SHA_M2B[S] and kb["coverage"]["blocked"] is False, {"coverage": kb.get("coverage"), "kb2": kb.get("kb2")})

    # COMPOSED: the swap at the declared p AND the acc term, on the same batch
    share = comp_t.get("share_at_declared_p")
    case("swap share at the DECLARED p: 0.1 +- 0.02 (mean n_selected / n_eligible over 400 steps of the loop's own dag.apply schedule)", share is not None and abs(share - 0.1) <= 0.02, {"share": share, "eligible_draws": comp_t.get("share_n_eligible_draws")})
    case("COMPOSED: on the SWAPPED batch the acc term is finite, > 0, differs from the logged batch's, and total(w) - total(0) == w x acc",
         comp_t["acc_on_swapped"] > 0.0 and comp_t["acc_on_swapped"] != comp_t["acc_on_logged"] and abs(comp_t["contribution_swapped"] - comp_t["w_x_acc_swapped"]) <= 1e-3 * comp_t["w_x_acc_swapped"] + 1e-5,
         {k: comp_t.get(k) for k in ("acc_on_swapped", "acc_on_logged", "contribution_swapped", "w_x_acc_swapped")})
    case("COMPOSED: on the SWAPPED batch O's head gets a non-zero gradient from the term; controls EXACTLY 0.0", comp_t["head_o_grad_from_term_on_swapped"] > 0.0 and comp_t["control_repeat_total_diff"] == 0.0 and comp_t["control_repeat_grad_diff"] == 0.0 and comp_t["control_prior_grad_diff"] == 0.0,
         {k: comp_t.get(k) for k in ("head_o_grad_from_term_on_swapped", "head_o_grad_from_term_on_logged", "control_repeat_total_diff", "control_repeat_grad_diff", "control_prior_grad_diff", "swap_moves_total")})
    case("COMPOSED: at the BUILT weight the term contributes w x acc (> 0) in the treatment", comp_t["contribution_at_BUILT_weight"] > 0.0 and comp_t["head_o_grad_diff_at_BUILT_weight"] > 0.0, {k: comp_t.get(k) for k in ("contribution_at_BUILT_weight", "head_o_grad_diff_at_BUILT_weight")})
    # the CONTROL: without --w-acc the term's contribution at the BUILT weight is exactly 0
    case("CONTROL (no --w-acc): built w_acc 0.0; the term's contribution at the BUILT weight and its gradient on head_o are EXACTLY 0.0",
         comp_c["built_w_acc"] == 0.0 and comp_c["contribution_at_BUILT_weight"] == 0.0 and comp_c["head_o_grad_diff_at_BUILT_weight"] == 0.0 and ap0.get("declared_zero_total_reproduces_exactly") is True and ap0["w_acc_declared"] == 0.0,
         {k: comp_c.get(k) for k in ("built_w_acc", "contribution_at_BUILT_weight", "head_o_grad_diff_at_BUILT_weight")} | {"declared_zero_total_reproduces_exactly": ap0.get("declared_zero_total_reproduces_exactly")})
    case("CONTROL: the swap is just as live without --w-acc (share and dagger_probe unchanged in kind)", sc0["dagger_probe"].get("pass") is True and abs(comp_c["share_at_declared_p"] - share) < 1e-12, {"share_ctrl": comp_c["share_at_declared_p"], "share_treat": share})
    # a negative weight is refused by the real gate (negative control of the lever's guard, reachable from this caller)
    a_neg = L.to_gate_argv(real, OUT / "dag_oacc_neg", init_ckpt=ck, bank=BANK, bank_sha=sha, extra=[("--w-acc", "-1.0")])
    rcn, scn = L.gate(OUT / "dag_oacc_neg", a_neg)
    case("REFUSED by the real gate: --w-acc -1.0 on the DAG argv", isinstance(rcn, str) and ("ZZDVB-REFUSEDZZ" in rcn or "w_acc must be a finite number >= 0" in rcn), str(rcn)[:200])

    # ---- Amendment 2 / one-lever from RECORDS: see arm1_pair_records.py (separate: it needs no torch graph)
    (OUT / "selfcheck_treatment.json").write_text(json.dumps(sc, indent=1, default=str), encoding="utf-8")
    (OUT / "selfcheck_control.json").write_text(json.dumps(sc0, indent=1, default=str), encoding="utf-8")
    return L.finish(f"arm1_s{S}")


if __name__ == "__main__":
    try:
        rc_main = main()
    except Exception as e:  # noqa: BLE001  an aborted script is a FAILED check, never a silent pass
        import traceback
        case(f"script aborted: {type(e).__name__}: {str(e)[:300]}", False, traceback.format_exc()[-600:])
        rc_main = L.finish(f"arm1_s{S}")
    sys.exit(rc_main)
