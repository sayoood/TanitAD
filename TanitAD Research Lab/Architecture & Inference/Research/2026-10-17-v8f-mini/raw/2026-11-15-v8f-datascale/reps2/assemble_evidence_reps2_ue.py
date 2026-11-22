"""Assemble the evidence JSON of an arm for ``dscale10k_check.py mint`` (pod). usage: assemble_evidence.py ARM [ARM ...]   (UE_s2 | UE_s3, REPS2: a fork of assemble_evidence_reps1_ue.py with the init record gate/init_ue_s<k>.json; DSCALE10K block 3 lineage: DSCALE10K block 3, derived from assemble_evidence_s1.py: the init record is gate/init_b3.json, the reach probe gate/probe2/<ARM>.A.json)
Reads the item records in /root/dscale10k/gate/ (each written by gate_items.py / resume_smoke.py / the probes), writes /root/dscale10k/gate/evidence_<ARM>.json:
  hyg {arm, base}   G-HYG   built cfg + trainable parameter count of the arm against B_s0's (init.json)
  reach             G-LIVE  the reach table (B arms: the corpus batch + the TL-positive control; U: corpus / 10k-only / mixed + control)
  clock             G-CLOCK the per-clip residual lists (union arms)
  G-EVAL, G-CKPT, G-SUITE   {"pass": bool, ...}: pass only if the item AND every one of its deliberate-regression arms read as required
  c_init            C-INIT  report
  reports           C-CACHE, C-SESS, the item records' sha256, the block-1 projection
Refuses (exit 1) when an item record is missing, failed, or one of its regression arms did not read RED."""
import hashlib
import json
import os
import sys

D = "/root/dscale10k"
G = D + "/gate"


def L(p):
    return json.load(open(p))


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def item(n):
    p = f"{G}/{n}.json"
    if not os.path.isfile(p):
        raise SystemExit(f"EVIDENCE-REFUSED: {p} is missing")
    d = L(p)
    if not d.get("pass"):
        raise SystemExit(f"EVIDENCE-REFUSED: item {n} does not pass: {d.get('problems')} regression {[r['red'] for r in d.get('regression', [])]}")
    return d, sha(p)


def projection():
    pr = L(D + "/PILOT_REPORT.json")
    p = pr["projection_block1"]
    ratio = 111.0 / 358.0                                                 # pilot/bank200_4sh.log "total wall_s 111" over pilot/bank200.log "[13583/13583 windows] 358s" (same 200 clips, ckpt, argv)
    sa, sb = p["stage_A_h"], p["stage_B_h"]
    bank = {k: v * ratio for k, v in p["bank_h"].items()}
    a = 2 * sa["B"] + 2 * sa["U"]                                         # B_s0, B_rep, U_s0, R_s0 stage A
    b = 2 * sb["B"] + sb["U"]
    k = 2 * bank["B_4512_clips"] + bank["U_14512_clips"]
    return {"evidence_class": "ESTIMATED from PILOT_REPORT.json (600-step pilots, 200-clip bank) with the 4-shard factor 111/358 = %.3f (MEASURED, pilot/bank200_4sh.log and pilot/bank200.log)" % ratio,
            "stage_A_h": round(a, 3), "stage_B_h": round(b, 3), "banks_h": round(k, 3), "bank_h_each": {kk: round(v, 3) for kk, v in bank.items()}, "block1_wall_h": round(a + b + k, 3),
            "block1_wall_h_with_1_shard_bank": p["block1_wall_h"], "stop_rule_10h": "OK (below 10 h)" if a + b + k < 10 else "STOP"}


def main():
    ev = {}
    init, init_sha = item("init")
    clock, clock_sha = item("clock")
    evl, evl_sha = item("eval")
    cp, cp_sha = item("ckptproc")
    rs, rs_sha = item("resume")
    su, su_sha = item("suite")
    cc, cc_sha = item("cachecheck")
    sb, sb_sha = item("stageb")
    se, se_sha = item("sess")
    gckpt = {"pass": True, "ckptproc_sha256": cp_sha, "resume_sha256": rs_sha, "resumed_equals_straight_bitwise": rs["record"].get("resumed_equals_straight_bitwise"),
             "n_tensors_differ_resumed_vs_straight": rs["record"].get("n_tensors_differ_resumed_vs_straight")}
    geval = {"pass": True, "eval_sha256": evl_sha, "B": {k: evl["record"]["B"][k] for k in ("strict_load_ok", "cfg_equal_built", "n_params", "n_windows", "n_episodes", "grid_sha256")}}
    gsuite = {"pass": True, "suite_sha256": su_sha, "mode": su["record"]["mode"], "tree_manifest_sha256": su["record"]["tree_manifest_sha256"], "n_regressions": su["record"]["n_regressions"]}
    for arm in sys.argv[1:]:
        assert arm in ("UE_s2", "UE_s3"), "assemble_evidence_reps2_ue.py serves UE_s2 / UE_s3"
        init_r = item("init_" + arm.lower())
        ini, ini_sha = init_r
        probe = f"{G}/probe2/{arm}.A.json"
        if not os.path.isfile(probe):
            raise SystemExit(f"EVIDENCE-REFUSED: {probe} is missing")
        pj = L(probe)
        # REPS2 guard (the checker does not bind evidence to an arm: a dry mint with UE_s1's evidence read PASS for UE_s2): the probe and the init record must be THIS arm's own
        if os.path.basename(pj.get("argv", "")) != f"{arm}.A.json":
            raise SystemExit(f"EVIDENCE-REFUSED: the reach probe {probe} was run on {pj.get('argv')!r}, not on argv/{arm}.A.json")
        if os.path.basename(ini["record"][arm].get("argv", "")) != f"{arm}.A.json" or int(ini["record"][arm].get("seed", -1)) != int(arm[-1]):
            raise SystemExit(f"EVIDENCE-REFUSED: the init record of {arm} names argv {ini['record'][arm].get('argv')!r} / seed {ini['record'][arm].get('seed')!r}")
        kinds =("corpus", "tenk", "mixed") if arm in ("UE_s2", "UE_s3") else ("corpus",)      # MM 2026-11-17: R_s0 trains on the union set too (REQUIRED_UNION)
        out = {"arm": arm, "hyg": {"arm": ini["record"][arm], "base": ini["record"]["B_s0"]},
               "reach": {k: pj[k] for k in kinds + ("control",)}, "reach_probe": probe, "reach_probe_sha256": sha(probe),
               "G-EVAL": geval, "G-CKPT": gckpt, "G-SUITE": gsuite, "c_init": ini["record"]["c_init_report"],
               "reports": {"C-CACHE": {k: cc["record"][k] for k in ("n", "n_equal", "bytes", "wall_s", "manifest_sha256", "actual_sha256", "pass")}, "C-SESS": se["record"], "stage_b_gate": {"B_s0.B": sb["record"]["B_s0.B"]["G-DAGGER"]["dagger_probe"], "bank_pilot": sb["record"]["B_s0.B"]["C-BANK"], "mutation": sb["record"]["mutation"], "U_s0.B_smoke": sb["record"]["U_s0.B_smoke"]},
                           "item_sha256": {"init": ini_sha, "clock": clock_sha, "eval": evl_sha, "ckptproc": cp_sha, "resume": rs_sha, "suite": su_sha, "cachecheck": cc_sha, "sess": se_sha, "stageb": sb_sha},
                           "projection_block1": projection(), "pilot_report_sha256": sha(D + "/PILOT_REPORT.json")}}
        if arm in ("UE_s2", "UE_s3"):                                               # MM 2026-11-17: the same G-CLOCK evidence (same 10k clips)
            out["clock"] = {"tenk": clock["record"]["tenk"], "corpus": clock["record"]["corpus"]}
        udry = D + "/dry/U_s0.A.dry.json"                                 # MM ruling 2026-11-17: the pair's step count = 4 x S_U of the UNION dry record (U_s0's), for UE_s0..UE_s3 and for BL_s0
        out["ue_steps"] = 4 * int(L(udry)["S_U"])
        out["ue_steps_source"] = {"path": udry, "sha256": sha(udry), "S_U": int(L(udry)["S_U"])}
        json.dump(out, open(f"{G}/evidence_{arm}.json", "w"), indent=1, sort_keys=True)
        print("evidence", arm, sha(f"{G}/evidence_{arm}.json")[:12])


if __name__ == "__main__":
    main()
