#!/usr/bin/env python3
"""ARM 2 -- FLAT-DAG = FLAT_s (`M2b_s{S}_flat.pt`, o_interface cut) fine-tuned with DAG's recipe (`--init-ckpt` FLAT_s, 3,000 steps, lr 1e-4, warmup 100, p_gt 0.2, ego-perturb 0, `--dagger-p 0.1` + a bank + sha + seed S),
keeping FLAT's `--o-interface cut`.  One REAL gate run (v8f_mini_train.py --selfcheck-only, CPU, synthetic world at the REAL dimensions, REAL FLAT_s checkpoint as --init-ckpt AND as the bank generator's --ckpt) must show
the cut AND the swap, both live.

    CUDA_VISIBLE_DEVICES=-1 python arm2_flat_dag.py           # S = CG_SEED (default 0)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cg_lib as L  # noqa: E402
from cg_lib import case  # noqa: E402

TR, torch, res = L.TR, L.torch, L.res
import v8f_dagger_bank as GEN  # noqa: E402  (the REAL bank generator module, imported only to call its load_ckpt / train_list)

S = int(__import__("os").environ.get("CG_SEED", "0"))
OUT = L.ROOT / "out" / (f"arm2_s{S}" + (("__MUT_" + L.MUT_RECORD["mutant"]) if L.MUT_RECORD else ""))
WORLD = L.WORK / f"world_s{S}"
BANK_F = L.WORK / f"bank_flat_s{S}"      # generator = FLAT_s
BANK_M = L.WORK / f"bank_m2b_s{S}"       # generator = M2b_s (the stand-in; made by arm1_dag_oacc.py)
CUTG = {}
_orig_oip = TR.o_interface_probe


def oip_plus(model, batch, cfg, shift=2.5):
    r = _orig_oip(model, batch, cfg, shift)
    CUTG["probe_batch"] = batch
    CUTG["logged"] = L.grad_probe_cut(model, batch, cfg)
    return r


TR.o_interface_probe = oip_plus
_dpp = TR.dagger_probe


def dpp2(model, batch, cfg, dag):
    r = _dpp(model, batch, cfg, dag)                  # = cg_lib's dagger_probe_plus (trainer's probe + composed_probe)
    if dag.active:
        CUTG["swapped"] = L.grad_probe_cut(model, dag.apply(batch, dag.PROBE_STEP, p=1.0), cfg)
    return r


TR.dagger_probe = dpp2


def is_refused(rc):
    return isinstance(rc, str) and ("ZZDVB-REFUSEDZZ" in rc or rc.startswith("EXC "))


def main():
    L.WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    fck, mck = L.FLAT_CKPT[S], L.M2B_CKPT[S]
    sha_f = L.sha256_file(fck)
    case(f"FLAT checkpoint on the dev box is the banked M2b_s{S}_flat (sha256 {L.SHA_FLAT_PREFIX[S]}...)", sha_f.startswith(L.SHA_FLAT_PREFIX[S]), sha_f[:12])
    ck_cfg = torch.load(fck, map_location="cpu", weights_only=False)["cfg"]
    case("FLAT checkpoint's own cfg reads o_interface: cut (and the M2b one has no such key)", ck_cfg.get("o_interface") == "cut" and "o_interface" not in torch.load(mck, map_location="cpu", weights_only=False)["cfg"], ck_cfg.get("o_interface"))
    world = L.make_world(WORLD, seed=S)

    # ---------------------------------------------------------------- (b) the bank generator accepts the FLAT checkpoint and FLAT's argv
    flat_doc = L.ROOT / f"flat_runs/flat_s{S}_argv.json"
    ns, _ = GEN.train_list(argparse.Namespace(m2b_argv=str(flat_doc), cache_dir="", tracks="", join_train=""))
    case("(b) generator's train_list() parses FLAT's REAL banked argv.json (--o-interface cut included) as --m2b-argv", ns.o_interface == "cut" and ns.arm == "M2b" and ns.synthetic is False, {"o_interface": ns.o_interface, "arm": ns.arm})
    gmodel, gcfg, gck = GEN.load_ckpt(fck, torch.device("cpu"))
    case("(b) generator's load_ckpt() builds the FLAT checkpoint as a CUT model (cfg.o_interface == cut) and loads it strictly", gcfg.o_interface == "cut" and gck["arm"] == "M2b", gcfg.o_interface)
    if (BANK_F / "bank.json").exists():
        shaF = L.sha256_file(BANK_F / "bank.json")
        case("(b) FLAT bank (cached from an earlier run in work/): bank.json present", True, shaF[:16])
        rcb = 0
    else:
        _, shaF, log, rcb, secs = L.make_bank(world, fck, BANK_F, extra_stub=["--o-interface", "cut"])
        (L.WORK / f"bank_flat_s{S}.log").write_text(log, encoding="utf-8")
        case("(b) bank generator --ckpt FLAT_s (cut), --m2b-argv a FLAT argv with --o-interface cut: 2 shards + merge, rc 0", rcb == 0, f"rc {rcb} {secs:.0f}s {log[-250:]!r}")
    if rcb != 0:
        return L.finish(f"arm2_s{S}")
    manF = json.loads((BANK_F / "bank.json").read_text(encoding="utf-8"))
    hdr = json.loads((BANK_F / "PROVENANCE.json").read_text(encoding="utf-8"))
    case("(b) C-F1: the FLAT bank's bank.json header names the FLAT checkpoint's sha256 (common.ckpt_sha256), is not PARTIAL, coverage clears 0.5, controls K-B4 / K-B5 recorded True",
         manF["common"]["ckpt_sha256"] == sha_f and not manF["partial"] and not manF["coverage"]["blocked"] and set(hdr.get("controls") or {}) == {"K-B4", "K-B5"} and all(v.get("PASS") is True for v in hdr["controls"].values()),
         {"common.ckpt_sha256": manF["common"]["ckpt_sha256"][:12], "coverage": manF["coverage"], "controls_PASS": {k: v.get("PASS") for k, v in (hdr.get("controls") or {}).items()}})

    # ---------------------------------------------------------------- the argvs: FLAT's real argv + DAG's recipe tail
    dag_tail = L.DAG_ARGV[S][L.DAG_ARGV[S].index("--init-ckpt"):]
    real = list(L.FLAT_ARGV[S]) + dag_tail
    fd = L.to_gate_argv(real, OUT / "flat_dag", init_ckpt=fck, bank=BANK_F, bank_sha=shaF)
    bad, added = L.carried_verbatim(real, fd)
    case("FLAT-DAG gate argv = FLAT's real argv + DAG's recipe tail, token-for-token, except the declared data/path re-pointing", not bad and not added, {"bad": bad, "added": added})
    shaM0 = L.sha256_file(BANK_M / "bank.json") if (BANK_M / "bank.json").exists() else "0" * 64
    dag_gate = L.to_gate_argv(L.DAG_ARGV[S], OUT / "dag_ref", init_ckpt=mck, bank=BANK_M, bank_sha=shaM0)       # DAG's own argv mapped onto the same synthetic world (its bank = the M2b-generated one)
    fdag, ffd = dict(L.pairs(dag_gate)), dict(L.pairs(fd))
    flag_diff = sorted(set(ffd) ^ set(fdag))
    val_diff = sorted(k for k in set(ffd) & set(fdag) if ffd[k] != fdag[k])
    case("pair vs DAG (both mapped onto the same world): FLAT-DAG adds exactly `--o-interface cut`; values differ only in --out / --init-ckpt (FLAT_s) / --dagger-bank / --dagger-bank-sha256 (FLAT's bank)",
         flag_diff == ["--o-interface"] and ffd["--o-interface"] == "cut" and val_diff == ["--dagger-bank", "--dagger-bank-sha256", "--init-ckpt", "--out"], {"flags": flag_diff, "values": val_diff})
    need = ["--o-interface", "--init-ckpt", "--steps", "--lr", "--warmup", "--p-gt-start", "--p-gt-end", "--ego-perturb-p", "--dagger-p", "--dagger-bank", "--dagger-bank-sha256", "--dagger-seed"]
    case("every lever incl. `--o-interface cut` is NAMED on the FLAT-DAG argv", all(k in dict(L.pairs(fd)) for k in need), {k: dict(L.pairs(fd)).get(k) for k in ("--o-interface", "--dagger-p", "--dagger-seed")})

    # ---------------------------------------------------------------- G1: the REAL gate on FLAT-DAG
    CUTG.clear()
    L.COMPOSED.clear()
    rc, sc = L.gate(OUT / "flat_dag", fd)
    case("REAL gate accepts FLAT-DAG (rc 0, declared_vs_built PASS)", rc == 0 and sc and sc.get("declared_vs_built") == "PASS", f"rc {str(rc)[:700]}")
    cg_t, comp_t = {k: v for k, v in CUTG.items() if k != "probe_batch"}, dict(L.COMPOSED)
    probe_batch = CUTG.get("probe_batch")
    if sc:
        lv, oi, dp = sc["levers"], sc["o_interface_probe"], sc["dagger_probe"]
        ic = lv["init_ckpt"]
        case("init is the REAL FLAT_s checkpoint: sha256 matches, applied, 0 tensors differ after the strict load", ic.get("sha256") == sha_f and ic.get("applied") and ic.get("n_mismatch_after_load") == 0, {k: ic.get(k) for k in ("sha256", "applied", "n_mismatch_after_load", "source_step")})
        case("G-DVB lists o_interface = cut AND dagger_p 0.1, dagger_seed, the bank sha256 (declared == built)", lv.get("o_interface") == "cut" and lv.get("dagger_p") == 0.1 and lv["dagger"].get("dagger_seed") == S and lv.get("dagger_bank_sha256") == shaF == lv.get("dagger_bank_sha256_declared"),
             {"o_interface": lv.get("o_interface"), "dagger_p": lv.get("dagger_p"), "seed": lv["dagger"].get("dagger_seed"), "bank": str(lv.get("dagger_bank_sha256"))[:16]})
        perts = oi["perturbations"]
        case("CUT, the trainer's own probe: O invariant to S and T -- traj60 bit-for-bit under donor / GT / random code and shifted S checkpoints; O is handed the constants (code 0, ck = nominal, ok 0)",
             oi["pass"] is True and oi["o_interface"] == "cut" and all(p["max_abs_dtraj60"] == 0.0 and p["o_read_code_abs_max"] == 0.0 and p["o_read_ck_minus_nominal_abs_max"] == 0.0 and p["o_read_ck_ok_abs_max"] == 0.0 for p in perts.values()) and len(perts) == 4,
             {n: p["max_abs_dtraj60"] for n, p in perts.items()})
        case("CUT, T still reads S (goal logits move with S's checkpoints): the cut leaves S->T intact", perts["shifted_checkpoints"]["t_reads_s_max_abs_dgoal_logits"] > 0.0, perts["shifted_checkpoints"]["t_reads_s_max_abs_dgoal_logits"])
        cl = cg_t["logged"]
        case("CUT, gradient (my probe, beside the trainer's): the O-PLAN loss gives e_code.weight EXACTLY 0.0 and S / T groups EXACTLY 0.0; O's own groups > 0", cl["e_code.weight"] == 0.0 and cl["S_groups(read_s,head_s)"] == 0.0 and cl["T_groups(read_t,trunk_t,h_goal)"] == 0.0 and cl["O_groups(read_o,head_o)"] > 0.0, cl)
        cs = cg_t.get("swapped")
        case("CUT + SWAP composed: on the SWAPPED batch the same gradient readings hold (e_code.weight, S, T exactly 0.0; O > 0)", bool(cs) and cs["e_code.weight"] == 0.0 and cs["S_groups(read_s,head_s)"] == 0.0 and cs["T_groups(read_t,trunk_t,h_goal)"] == 0.0 and cs["O_groups(read_o,head_o)"] > 0.0, cs)
        case("SWAP reached, trainer's own probe: swapped batch moves loss and gradient; p = 0 control EXACTLY 0.0", dp.get("pass") is True and dp["loss_diff"] != 0.0 and dp["grad_diff_all"] > 0.0 and dp["control_p0_loss_diff"] == 0.0 and dp["control_p0_grad_diff"] == 0.0 and dp["control_p0_same_object"] is True,
             {k: dp.get(k) for k in ("p_declared", "n_selected_at_p1", "loss_diff", "grad_diff_all", "control_p0_loss_diff", "control_p0_grad_diff")})
        case("SWAP reached: share 0.1 +- 0.02 at the declared p (400 steps of the loop's own dag.apply schedule)", comp_t.get("share_at_declared_p") is not None and abs(comp_t["share_at_declared_p"] - 0.1) <= 0.02, {"share": comp_t.get("share_at_declared_p"), "draws": comp_t.get("share_n_eligible_draws")})
        kb = sc["info"].get("dagger") or {}
        case("K-B2 ties the bank to the INIT checkpoint (= FLAT_s): read, equal, and COVERAGE >= 0.5", kb.get("kb2", {}).get("ckpt_sha256") == sha_f == kb.get("kb2", {}).get("init_ckpt_sha256") and kb["coverage"]["blocked"] is False, {"kb2": kb.get("kb2"), "coverage": kb.get("coverage")})
        case("live_check on the SWAPPED batch under the cut: pass (declared-dead set is the cut's O-input weights, nothing else)", dp["live_swapped"]["pass"] is True, {"terms_finite": True, "groups_grad": dp["live_swapped"]["groups_grad"], "dead": sc["live"].get("dead_params"), "declared_dead": sc["live"].get("declared_dead")})

    # ---------------------------------------------------------------- bank <-> checkpoint: does the trainer tie the bank to the generator checkpoint? (a)
    if BANK_M.exists() and (BANK_M / "bank.json").exists():
        shaM = L.sha256_file(BANK_M / "bank.json")
        fd_m = L.to_gate_argv(real, OUT / "flat_dag_m2bbank", init_ckpt=fck, bank=BANK_M, bank_sha=shaM)
        rcm, scm = L.gate(OUT / "flat_dag_m2bbank", fd_m)
        case("(a) NEGATIVE: FLAT-DAG on a bank generated by M2b_s (the iteration-1 stand-in) is REFUSED by the real gate with K-B2", is_refused(rcm) and "K-B2" in str(rcm), str(rcm)[:330])
    else:
        case("(a) NEGATIVE: needs work/bank_m2b_s{S} (run arm1_dag_oacc.py first)", False, "missing")

    # ---------------------------------------------------------------- the footgun: the cut dropped from the argv
    fd_nocut = L.to_gate_argv(real, OUT / "flat_dag_nocut", init_ckpt=fck, bank=BANK_F, bank_sha=shaF, drop_flags={"--o-interface"})
    CUTG.clear()
    L.COMPOSED.clear()
    rcn, scn = L.gate(OUT / "flat_dag_nocut", fd_nocut)
    nocut_accepted = (rcn == 0 and scn is not None and scn.get("declared_vs_built") == "PASS")
    res["finding_F1_nocut_argv_accepted_by_gate"] = bool(nocut_accepted)
    res["finding_F1_detail"] = {"rc": str(rcn)[:300], "built_o_interface": (scn or {}).get("levers", {}).get("o_interface"), "init_ckpt_cfg_o_interface": ck_cfg.get("o_interface"),
                                "grad_readings_full_built": {k: v for k, v in CUTG.items() if k != "probe_batch"}}
    case("CONTROL for the cut reading: the SAME argv WITHOUT `--o-interface cut` builds `full` and the gradient readings flip (e_code.weight > 0, S / T groups > 0) -- the probe can see the cut",
         scn is not None and scn["levers"].get("o_interface") == "full" and CUTG["logged"]["e_code.weight"] > 0.0 and CUTG["logged"]["S_groups(read_s,head_s)"] > 0.0 and CUTG["logged"]["T_groups(read_t,trunk_t,h_goal)"] > 0.0, CUTG.get("logged"))
    print(f"FINDING F1: gate {'ACCEPTS' if nocut_accepted else 'REFUSES'} a FLAT-DAG argv with no --o-interface cut (checkpoint cfg says cut)  rc={str(rcn)[:200]}", flush=True)
    case("(documented, not a pass criterion) the gate refuses a FLAT-DAG argv that drops the cut", True, "ACCEPTED (hole F1)" if nocut_accepted else "REFUSED")

    # ---------------------------------------------------------------- FLAT-DAG-0 (the PREREG's pair arm): same argv with --dagger-p 0 and no bank flags
    d0 = L.to_gate_argv(real, OUT / "flat_dag0", init_ckpt=fck, bank=BANK_F, bank_sha=shaF, drop_flags={"--dagger-bank", "--dagger-bank-sha256", "--dagger-seed"})
    d0[d0.index("--dagger-p") + 1] = "0"
    CUTG.clear()
    rc0, sc0 = L.gate(OUT / "flat_dag0", d0)
    case("FLAT-DAG-0 (--dagger-p 0, no bank flags): REAL gate accepts; cut live; dagger off", rc0 == 0 and sc0 and sc0["levers"].get("o_interface") == "cut" and sc0["dagger_probe"].get("dagger") == "off" and CUTG["logged"]["e_code.weight"] == 0.0, {"rc": str(rc0)[:200], "dagger_probe": (sc0 or {}).get("dagger_probe")})
    if sc:
        (OUT / "selfcheck_flat_dag.json").write_text(json.dumps(sc, indent=1, default=str), encoding="utf-8")
        (OUT / "cut_gradient_readings.json").write_text(json.dumps({"flat_dag_logged": cg_t.get("logged"), "flat_dag_swapped": cg_t.get("swapped"), "composed_probe_flat_dag": comp_t, "nocut_full_control": res.get("finding_F1_detail")}, indent=1, default=str), encoding="utf-8")
    if probe_batch is not None:
        # the generator-loaded FLAT model IS the cut policy: the trainer's own cut probe, run on the model the bank generator builds
        try:
            r = TR.o_interface_probe.__wrapped__(gmodel, probe_batch, gcfg) if hasattr(TR.o_interface_probe, "__wrapped__") else _orig_oip(gmodel, probe_batch, gcfg)
            case("(b) the model the BANK GENERATOR builds from the FLAT checkpoint passes the trainer's cut probe (traj60 invariant to S/T): the bank is rolled out by the CUT policy", r.get("pass") is True and r["o_interface"] == "cut",
                 {n: p["max_abs_dtraj60"] for n, p in r["perturbations"].items()})
        except AssertionError as e:
            case("(b) generator-built FLAT model passes the cut probe", False, str(e)[:300])
    return L.finish(f"arm2_s{S}")


if __name__ == "__main__":
    try:
        rc_main = main()
    except Exception as e:  # noqa: BLE001  an aborted script is a FAILED check, never a silent pass
        import traceback
        case(f"script aborted: {type(e).__name__}: {str(e)[:300]}", False, traceback.format_exc()[-600:])
        rc_main = L.finish(f"arm2_s{S}")
    sys.exit(rc_main)
