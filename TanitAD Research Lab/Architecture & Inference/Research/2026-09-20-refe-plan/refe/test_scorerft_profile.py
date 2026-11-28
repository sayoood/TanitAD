#!/usr/bin/env python3
"""SCORER-FT launch-gate PROFILES and ROWS (DESIGN_SCORER_FT section 3; PI go 2026-11-20): `refe_scorerft_t` (T), `refe_scorerft_c` (C), `refe_scorerft_t_s1` (Ts), `refe_scorerft_c_s1` (Cs), and the G-DVB / G-LIVE rows
`rank_dvb_rows`, `rank_live_rows` (L1-L5), the routing in `check_live` and the generalised (h).

  python test_scorerft_profile.py            everything, incl. the source-mutation harness
  python test_scorerft_profile.py --fast     no mutation harness
  python test_scorerft_profile.py --check N  one named check (what the harness runs on a mutated copy of this tree)

CPU only, no scipy, stdlib + numpy + torch + the gate modules.  Prints ZZSCORERFTPROF_PASS <n>.  Asserts:
  * the four profiles exist, share the recipe, and NO existing profile moved (every pre-existing profile's whole-profile fingerprint equals its pin below: the MEASURED 2026-11-07 trap);
  * argv(T) - argv(C) = {--rank-w, --rank-heads, --rank-ep-gap, --rank-gate-topk, --out};  argv(Ts) - argv(T) = {--seed, --out};  argv(Cs) - argv(C) = {--seed, --out};  C names `--rank-w 0.0`;
  * the profiles REFUSE while the base sha256 or the weight is unfilled; with the pins filled the template argv is clean; each lever the card names is refused when it moves; REFE_RANK_MUTATE in the environment is refused;
  * the rows (pure functions of a capture) judge a CLEAN synthetic ranking term PASS and turn RED on each deliberate-regression arm of DESIGN 3.4 that is observable without the trainer (R-DROP, R-UNSAFE, R-SHUF, R-LEAK, R-HEAD,
    R-CTRL), with a numpy reference independent of the term;
  * G-LIVE's (h) keys on `traj_label_required`, and the scorerft arms are routed to the same live rows as C1.
INTEGRATION_PENDING (printed, not a failure): the checks that need the built trainer half (train.py's four flags, rank_loss.RankLoss) are re-run by `chk_trainer_integration` once `--rank-w` is in the parser."""
from __future__ import annotations

import hashlib
import inspect
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import gate_token as GT  # noqa: E402
import launch_gate as L  # noqa: E402
import gate_checks as GC  # noqa: E402

N = 0
HEX_BASE = "ab" * 32
W_FIX = 0.25                                                       # a fixture rank_w for the numeric gradient captures below (the MM's real one is the weight rule's output)
# the weight the TEMPLATE argv and the filled profile carry: the registered SCORERFT_RANK_W once the MM fills it (the profile then REQUIRES that exact --rank-w,
# so a fixture value would be refused), else the fixture. MM 2026-11-21: with W_FIX here this file went RED as soon as the pin was filled, and it is in
# SCORERFT_SUITE_EXTRA, so T's G-SUITE would have refused T (MEASURED on a scratch copy with SCORERFT_RANK_W = 0.5).
W_ARGV = float(L.SCORERFT_RANK_W) if L.SCORERFT_RANK_W is not None else W_FIX
# ARM P / Ps carry THEIR OWN registered weight SCORERFT_P_RANK_W (the pod weight rule on P's own argv): once the MM fills it the P profiles REQUIRE that exact --rank-w, so the P template argv and the filled P profile
# must use it, not T's (MEASURED 2026-11-25 on a scratch copy with SCORERFT_P_RANK_W = 0.0625: "refe_scorerft_p requires --rank-w 0.0625; the argv has --rank-w 0.03125", i.e. P's G-SUITE would have refused P).
WP_ARGV = float(L.SCORERFT_P_RANK_W) if L.SCORERFT_P_RANK_W is not None else W_ARGV
ARMS = ("T", "C", "Ts", "Cs")
NAMES = {"T": "refe_scorerft_t", "C": "refe_scorerft_c", "Ts": "refe_scorerft_t_s1", "Cs": "refe_scorerft_c_s1"}
P_NAMES = {"P": "refe_scorerft_p", "Ps": "refe_scorerft_p_s1"}                # O3's decision-level arm (head `pick`) and its seed-1 replicate
# whole-profile fingerprints sha256(repr(sorted(items)))[:16] of EVERY profile that existed at 36b92a2b (the 2026-11-07 trap: a launch_gate.py edit can move them, and other tests pin them BY VALUE)
FP_PINS = {"refe_p2": "627341286aea3506", "refe_p6": "80fa4d7ec757ad46", "refe_field": "5ddc49053a99fa78", "refe_field_ctrl": "0b6281b1bfa996ed", "refe_rehearsal": "821cd1ef3bf55d4e",
           "refe_rls1": "bce7e25ee8718c08", "refe_aux_s1": "f95f48727b62ccc6", "refe_p7": "94cca167ce121a94", "refe_p5c": "ed995961b67113bd", "refe_p5c_c0": "65b0a2e8fc15b36e",
           "refe_p5c_c0r": "e89c54c4fa046afd", "refe_field_p7": "6950f70944a0fbda", "refe_field_p7_ctrl": "3dc54a1a4de1a7d7", "refe_field_p7_ctrl_s1": "84134e3ebb2d3f93",
           "refe_field_p7_s1": "b502f0275a4862d8", "refe_p2rep": "44ca107f53aceee8", "refe_sft4r": "8972e3c348206b94", "refe_scorer1": "88d02fb7b34d4fd8", "refe_scorer1p6": "e74d64617f514d35",
           "refe_s1z_p6": "5c3bff457aa3bf1e", "refe_s1zsm_p6": "86380fbb786daa75", "refe_rep_p6": "dd348f44c9638f1a", "refe_rep_s1z_p6": "2356afd4efeeb52a", "refe_rep_s1zsm_p6": "27fe236a2fd68ca4",
           "refe_dose_s1z_p6": "9085f9e5cf59c016", "refe_div_p6": "e51f9374a8f29816", "refe_rls1b": "5d6bfbc10152ae96", "refe_rls1c": "26214cf954099d51", "refe_rls1d": "fcc2585f2a5d9b94",
           "refe_rls1e": "54d96289b19c3cd6", "refe_hnscreen": "d30126bbb5a32957", "refe_aux_s1c": "dcdf2dbd2a674f1c", "refe_vdac_p6": "d4e05641baf8a48e", "refe_p5c_c1r": "4a76a09574fbc144"}          # + refe_p5c_c1r (C1's seed-1 replicate, PI go 2026-11-20): pinned by value too


def ok(cond, msg=""):
    global N
    assert cond, msg
    N += 1


def filled(arm: str) -> dict:
    """the profile with every MM-filled pin given a (fixture) value: what the gate sees once the real objects exist."""
    p = L.scorerft_profile(arm)
    p["init_from_sha256"] = HEX_BASE
    if L.SCORERFT_ARM_RANK[arm]:
        p["rank_w"] = W_ARGV
    return p


def targv(arm: str) -> list:
    return L.scorerft_argv(arm, rank_w=W_ARGV if L.SCORERFT_ARM_RANK[arm] else None)


def refusals(prof: dict, argv: list) -> str:
    return " || ".join(L.profile_refusals(prof, argv))


def flagset(argv):
    return {f: v for f, v in GT.flag_pairs(argv)}


def delta(a: list, b: list) -> set:
    fa, fb = flagset(a), flagset(b)
    return {f for f in set(fa) | set(fb) if fa.get(f) != fb.get(f)}


# ------------------------------------------------------------------------------------------------ profile checks
def chk_registry():
    for arm, nm in NAMES.items():
        ok(nm in L.PROFILES and L.PROFILES[nm]["name"] == nm and L.PROFILES[nm]["scorerft_rules"] is True and L.PROFILES[nm]["scorerft_arm"] == arm, nm)
        p = L.PROFILES[nm]
        ok(not p.get("p5c_rules") and not p.get("p7field_rules") and "p5c_arm" not in p, nm)
        ok(p["traj_label_required"] is True and p["required_flags"]["--steps"] == "1000" and p["required_flags"]["--epochs"] == "0" and "--traj-label-table" in p["required_flags"], nm)
        ok(p["expect_schedule"]["steps"] == 1000 and p["expect_schedule"]["epochs"] == 0 and p["expect_schedule"]["epoch_len"] == L.P7_SCENES, nm)
        ok(p["traj_label_sha256"] == L.P5C_LABEL_TABLE_SHA256 and p["traj_label_rank_policy"] == "rank0", nm)
        ok(any(e.get("script") == "test_rank_loss.py" and e["token"] == "ZZRANK_PASS" for e in p["suite"]), nm)
        ok(any(e.get("script") == "test_scorerft_profile.py" and e["token"] == "ZZSCORERFTPROF_PASS" for e in p["suite"]), nm)
        ok("REFE_RANK_MUTATE" in p["forbidden_env"], nm)
    for nm in ("refe_scorerft_t", "refe_scorerft_t_s1"):
        ok(L.PROFILES[nm]["live_terms"] == L.P6_TERMS + ("rank_L",) and L.PROFILES[nm]["required_flags"]["--rank-heads"] == "ep,gate", nm)
    for nm in ("refe_scorerft_c", "refe_scorerft_c_s1"):                                           # a control that declared its term FAILED G-LIVE (field_L ABSENT, C, 2026-11-19)
        ok("rank_L" not in L.PROFILES[nm]["live_terms"] and L.PROFILES[nm]["required_flags"]["--rank-w"] == "0.0" and "--rank-heads" not in L.PROFILES[nm]["required_flags"], nm)
    ok(L.PROFILES["refe_scorerft_t"]["required_flags"]["--seed"] == "0" and L.PROFILES["refe_scorerft_t_s1"]["required_flags"]["--seed"] == "1", "seeds")
    ok(L.SCORERFT_BASE_SHA256 is None or re.fullmatch(r"[0-9a-f]{64}", L.SCORERFT_BASE_SHA256), "pin shape")
    for k in ("rank_w", "rank_heads", "rank_ep_gap", "rank_gate_topk"):
        ok(GC.DVB_KINDS.get(k) == "built", k)


def chk_existing_unchanged():
    """every profile that existed at 36b92a2b has its pinned whole-profile fingerprint; no other profile gained a scorerft key; the new ones are exactly the four."""
    for nm, fp in FP_PINS.items():
        got = hashlib.sha256(repr(sorted(L.PROFILES[nm].items())).encode()).hexdigest()[:16]
        ok(got == fp, f"{nm}: fingerprint moved {fp} -> {got}")
    SH_NAMES = {"refe_safehead_red", "refe_shmlp_s0", "refe_shmlp_s1", "refe_shmlp_red", "refe_ephead_s0", "refe_ephead_s1", "refe_ephead_red", "refe_ds_sh_s0", "refe_ds_sh_s1", "refe_ds_sh_red", "refe_ds_ep_s0", "refe_ds_ep_s1", "refe_ds_ep_red", "refe_safehead_s0", "refe_safehead_s1"}   # PREREG_SAFEHEAD 2026-11-27: three new profiles; no existing fingerprint moves; PREREG_SHMLP/EPHEAD 2026-11-27: six more  # + the 6 DATA-SCALE profiles (PREREG_DATASCALE, 2026-11-28)
    ok(set(L.PROFILES) == set(FP_PINS) | set(NAMES.values()) | set(P_NAMES.values()) | SH_NAMES, sorted(set(L.PROFILES) ^ (set(FP_PINS) | set(NAMES.values()) | set(P_NAMES.values()) | SH_NAMES)))
    for nm in FP_PINS:
        ok(not any(k.startswith(("scorerft", "rank_")) or k == "traj_label_required" for k in L.PROFILES[nm]), nm)


def chk_argv_delta():
    T, C, Ts, Cs = (targv(a) for a in ARMS)
    ok(delta(T, C) == {"--rank-w", "--rank-heads", "--rank-ep-gap", "--rank-gate-topk", "--out"}, delta(T, C))
    ok(delta(Ts, T) == {"--seed", "--out"}, delta(Ts, T))
    ok(delta(Cs, C) == {"--seed", "--out"}, delta(Cs, C))
    ok(flagset(C)["--rank-w"] == ["0.0"] and "--rank-heads" not in flagset(C), "C names --rank-w 0.0 explicitly and no head")
    ok(flagset(T)["--rank-heads"] == ["ep,gate"] and float(flagset(T)["--rank-w"][0]) == W_ARGV and flagset(T)["--rank-gate-topk"] == ["8"], "T rule")
    c1 = flagset(L.p5c_argv("C1"))
    for a_, v_ in zip(ARMS, (T, C, Ts, Cs)):
        d = delta(v_, L.p5c_argv("C1"))
        ok(d == {"--init-from", "--steps", "--out", "--rank-w"} | ({"--rank-heads", "--rank-ep-gap", "--rank-gate-topk"} if L.SCORERFT_ARM_RANK[a_] else set()) | ({"--seed"} if a_ in ("Ts", "Cs") else set()), (a_, d))
        ok(flagset(v_)["--traj-label-table"] == c1["--traj-label-table"] and flagset(v_)["--score-w"] == ["0.1"] and "--scorer-sees-trunk" in flagset(v_), a_)
        ok(flagset(v_)["--steps"] == ["1000"] and flagset(v_)["--init-from"] == [L.SCORERFT_DEFAULT_INIT], a_)
    try:
        L.scorerft_argv("T")                                                                      # no template without the weight
        ok(L.SCORERFT_RANK_W is not None, "unfilled weight must raise")
    except ValueError:
        ok(L.SCORERFT_RANK_W is None)


def _real_rank_loss():
    """The REAL refe/rank_loss.py (the suite's other checks install a fake `rank_loss` module in sys.modules; this one is loaded by path under its own name)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("rank_loss_real_for_pick", str(HERE / "rank_loss.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def chk_p_arms():
    """ARM P (O3: the pick-anchored decision-level term, head `pick`): argv(P) - argv(C) = {--rank-w, --rank-heads, --out} and NOTHING else (no gap / top-k, which the trainer refuses without ep / gate); argv(Ps) - argv(P) =
    {--seed, --out}; the weight pin SCORERFT_P_RANK_W refuses while unfilled; with the pins filled the template is clean; every lever the card names is refused when it moves; the control C is unchanged by the new arms."""
    P_, Ps_ = (L.scorerft_argv(a, rank_w=WP_ARGV) for a in ("P", "Ps"))
    C_, Cs_ = targv("C"), targv("Cs")
    ok(delta(P_, C_) == {"--rank-w", "--rank-heads", "--out"}, delta(P_, C_))
    ok(delta(Ps_, P_) == {"--seed", "--out"}, delta(Ps_, P_))
    ok(delta(Ps_, Cs_) == {"--rank-w", "--rank-heads", "--out"}, delta(Ps_, Cs_))
    fP = flagset(P_)
    ok(fP["--rank-heads"] == ["pick"] and float(fP["--rank-w"][0]) == WP_ARGV and "--rank-ep-gap" not in fP and "--rank-gate-topk" not in fP, fP)
    ok(flagset(P_)["--seed"] == ["0"] and flagset(Ps_)["--seed"] == ["1"] and flagset(P_)["--init-from"] == [L.SCORERFT_DEFAULT_INIT] and flagset(P_)["--steps"] == ["1000"], "P recipe = C's")
    ok(L.SCORERFT_ARM_NAME["P"] == "refe_scorerft_p" and L.SCORERFT_ARM_NAME["Ps"] == "refe_scorerft_p_s1" and flagset(P_)["--out"] != flagset(C_)["--out"], "names / out")
    for a_, nm in P_NAMES.items():
        p = L.PROFILES[nm]
        ok(p["name"] == nm and p["scorerft_rules"] is True and p["scorerft_arm"] == a_ and p["rank_heads"] == "pick" and "rank_L" in p["live_terms"], nm)
        ok(p["required_flags"]["--rank-heads"] == "pick" and "--rank-ep-gap" not in p["required_flags"] and "--rank-gate-topk" not in p["required_flags"], nm)
        ok(p.get("rank_census") == L.SCORERFT_P_RANK_CENSUS and p["rank_census"] != L.SCORERFT_RANK_CENSUS and p["traj_label_required"] is True and "REFE_RANK_MUTATE" in p["forbidden_env"], nm)   # P carries ITS OWN census pin (2026-11-25), never T's
        ok(p["required_flags"]["--seed"] == ("0" if a_ == "P" else "1"), nm)
        ok(any(e.get("script") == "test_rank_loss.py" and e["token"] == "ZZRANK_PASS" for e in p["suite"]), nm)
        ok(sum(1 for e in p["suite"] if e.get("script") == "test_clip_probe.py") == 1 and {"script": "test_clip_probe.py", "token": "ZZCLIPPROBE_PASS"} in p["suite"], nm + ": P / Ps carry test_clip_probe.py in G-SUITE")
    for nm in list(NAMES.values()):                                                                # ... and ONLY P / Ps: T / C / Ts / Cs are untouched
        ok(all(e.get("script") != "test_clip_probe.py" for e in L.PROFILES[nm]["suite"]), nm + ": no test_clip_probe.py")
    ok(L.PROFILES["refe_scorerft_p"]["suite"] == L.PROFILES["refe_scorerft_p_s1"]["suite"] and L.SCORERFT_P_SUITE_EXTRA == [{"script": "test_clip_probe.py", "token": "ZZCLIPPROBE_PASS"}]
       and L.PROFILES["refe_scorerft_p"]["suite"][:-1] == L.PROFILES["refe_scorerft_c"]["suite"], "P suite = C's suite + the probe test (last)")
    try:
        L.scorerft_argv("P")                                                                       # no template without the weight pin
        ok(L.SCORERFT_P_RANK_W is not None, "unfilled P weight must raise")
    except ValueError:
        ok(L.SCORERFT_P_RANK_W is None)
    p0 = L.scorerft_profile("P")                                                                   # the profile as committed: the weight pin decides
    p0["init_from_sha256"] = HEX_BASE
    if L.SCORERFT_P_RANK_W is None:
        ok(p0["rank_w"] is None and "finite float" in refusals(p0, P_) and "SCORERFT_P_RANK_W" in refusals(p0, P_), "an unfilled P weight REFUSES the argv: " + refusals(p0, P_))
    for a_ in ("P", "Ps"):
        argv = L.scorerft_argv(a_, rank_w=WP_ARGV)
        pf = L.scorerft_profile(a_)
        pf["init_from_sha256"] = HEX_BASE
        pf["rank_w"] = WP_ARGV
        ok(refusals(pf, argv) == "", f"{a_}: the template with the pins filled must be clean: {refusals(pf, argv)}")
        for nm, bad in (("heads ep,gate", mv(argv, "--rank-heads", "ep,gate")), ("heads ep,pick", mv(argv, "--rank-heads", "ep,pick")), ("rank-w moved", mv(argv, "--rank-w", repr(WP_ARGV + 0.5))),
                        ("steps", mv(argv, "--steps", "2000")), ("no heads", drop(argv, "--rank-heads")), ("no rank-w (R-CTRL)", drop(argv, "--rank-w")),
                        ("gap named", argv + ["--rank-ep-gap", "0.05"]), ("topk named", argv + ["--rank-gate-topk", "8"]), ("seed", mv(argv, "--seed", "1" if a_ == "P" else "0")), ("extra", argv + ["--w-field", "2.0"])):
            ok(refusals(pf, bad) != "", f"{a_} must refuse: {nm}")
        pw = dict(pf, rank_w=0.0)
        ok("must be > 0" in refusals(pw, argv), "P at weight 0 refuses")
        ph = dict(pf, rank_heads="ep,gate")
        ok("not the card's" in refusals(ph, argv), "P registering T's heads refuses")
    ok(all(delta(targv(a_), L.p5c_argv("C1")) == ({"--init-from", "--steps", "--out", "--rank-w"} | ({"--rank-heads", "--rank-ep-gap", "--rank-gate-topk"} if L.SCORERFT_ARM_RANK[a_] else set())
                                                    | ({"--seed"} if a_ in ("Ts", "Cs") else set())) for a_ in ARMS), "T / C / Ts / Cs are unchanged by the new arms")


def chk_pick_rows():
    """The gate's own rows with the `pick` head, against the REAL rank_loss.RankLoss: (1) the gate's numpy reference == the trainer's term, VALUE and PAIR SET, on float32 logits (the trainer's dtype); (2) L3 on a stand-in model: `pick`
    reaches score_head rows NC, DAC, EP, TTC AND C and EXACTLY 0.0 on DDC, on the trajectory side and under the weight-0 control; a gate that thought `pick` trains only NC/DAC/EP is RED, and so is a term that stops reaching C."""
    import torch
    RLr = _real_rank_loss()
    for seed in range(4):
        s_, y = make_sets(seed=20 + seed, B=3, M=16)
        msk = np.ones(3, np.float32)
        term = RLr.RankLoss("pick", ep_gap=0.02, gate_topk=8)
        v = term.term(torch.from_numpy(s_).requires_grad_(True), torch.from_numpy(y), torch.from_numpy(msk), 3)
        ref = GC.rank_reference(s_, y, msk, 3, "pick", 0.02, 8)
        ok(ref["pairs"]["pick"] > 0 and ref["pairs"]["pick"] == term.last["pairs"]["pick"], (seed, ref["pairs"], term.last))
        ok(abs(float(v) - ref["L"]) <= 1e-6 * max(1.0, abs(ref["L"])), (seed, float(v), ref["L"]))
        P = term.pairs(torch.from_numpy(s_), torch.from_numpy(y))
        mine = sorted(tuple(int(t) for t in idx) for idx in P["pick"][0].nonzero().tolist())
        ok(mine == ref["pair_sets"]["pick"], seed)
        ok(GC.judge_rank_audit([{"logits": s_, "labels": y, "msk": msk, "batch": 3, "heads": "pick", "ep_gap": 0.02, "gate_topk": 8, "value": float(v), "pairs": dict(term.last["pairs"])}], None)[0] == [], seed)
    old_mod = sys.modules.get("rank_loss")
    sys.modules["rank_loss"] = RLr                                                                   # the probe (`import rank_loss as RL` in gate_checks.rank_reach) now builds the REAL term
    try:
        done = False
        for seed in range(6):
            cap = reach_cap(heads="pick", seed=seed)
            fr = GC.rank_reach(cap, {}, None)
            if not fr["declared"]["loss"] > 0.0:
                continue                                                                             # the stand-in model's own pick happened to be the labels' best: no pair on that set
            done = True
            rows = fr["declared"]["score_head_rows"]
            ok(all(rows[h] > 0.0 for h in ("nc", "dac", "ep", "ttc", "c")) and rows["ddc"] == 0.0, rows)
            g = fr["declared"]["groups"]
            ok(g["traj_head"]["abs_sum"] == 0.0 and g["dec"]["abs_sum"] == 0.0 and g["queries"]["abs_sum"] == 0.0 and g["backbone.lora"]["abs_sum"] > 0 and g["scene_proj"]["abs_sum"] > 0, g)
            ok(all(v["abs_sum"] == 0.0 for v in fr["zero"]["groups"].values()) and all(v == 0.0 for v in fr["zero"]["score_head_rows"].values()), "the weight-0 control reads exactly 0.0")
            ok(GC.judge_rank_reach(fr, "pick") == [], GC.judge_rank_reach(fr, "pick"))
            ok(any("row TTC" in x or "row C" in x for x in GC.judge_rank_reach(fr, "ep,gate")), "a pick capture judged as ep,gate (TTC / C must be 0.0) is RED")
            fr_bad = {**fr, "declared": {**fr["declared"], "score_head_rows": {**rows, "c": 0.0}}}
            ok(any("row C" in x for x in GC.judge_rank_reach(fr_bad, "pick")), "pick: C reading zero is RED")
            break
        ok(done, "no seed gave the stand-in model a non-oracle pick")
    finally:
        if old_mod is None:
            sys.modules.pop("rank_loss", None)
        else:
            sys.modules["rank_loss"] = old_mod


def chk_pins_refuse():
    for arm in ARMS:
        argv = targv(arm)
        p0 = L.scorerft_profile(arm)
        p0["init_from_sha256"] = None
        ok("SCORERFT_BASE_SHA256" in refusals(p0, argv), arm)                                      # an unfilled base pin refuses
        p1 = filled(arm)
        ok(refusals(p1, argv) == "", f"{arm}: the template argv with the pins filled must be clean: {refusals(p1, argv)}")
        p2 = filled(arm)
        p2["traj_label_sha256"] = "cd" * 32
        ok("is not C1's" in refusals(p2, argv), arm)
        p3 = filled(arm)
        p3["traj_label_rank_policy"] = "all"
        ok("rank policy" in refusals(p3, argv), arm)
    pw = L.scorerft_profile("T")
    pw["init_from_sha256"] = HEX_BASE
    pw["rank_w"] = None
    ok("rank_w" in refusals(pw, targv("T")) and "finite float" in refusals(pw, targv("T")), "an unfilled weight refuses")
    pw["rank_w"] = 0.0
    ok("must be > 0" in refusals(pw, targv("T")), "T at weight 0 refuses")
    pc = filled("C")
    pc["rank_w"] = 0.1
    ok("EXACTLY 0.0" in refusals(pc, targv("C")), "C at a weight refuses")
    pr = filled("T")
    pr["rank_gate_topk"] = 16
    ok("not the card's" in refusals(pr, targv("T")), "the registered rule is the card's")
    pe = filled("T")
    pe["expect_schedule"] = dict(pe["expect_schedule"], steps=2000)
    ok("not the card's" in refusals(pe, targv("T")), "the schedule is 1,000 steps")
    pt = filled("T")
    pt["traj_label_required"] = False
    ok("traj_label_required" in refusals(pt, targv("T")), "(h) must key on the profile")


def mv(argv: list, flag: str, val: str) -> list:
    a = list(argv)
    a[a.index(flag) + 1] = val
    return a


def drop(argv: list, flag: str) -> list:
    a = list(argv)
    i = a.index(flag)
    del a[i:i + 2]
    return a


def chk_levers():
    pT, pC = filled("T"), filled("C")
    T, C = targv("T"), targv("C")
    for nm, bad in (("steps", mv(T, "--steps", "2000")), ("seed", mv(T, "--seed", "1")), ("rank-w", mv(T, "--rank-w", repr(W_ARGV + 0.5))), ("heads", mv(T, "--rank-heads", "ep")),
                    ("gap", mv(T, "--rank-ep-gap", "0.05")), ("topk", mv(T, "--rank-gate-topk", "16")), ("lr", mv(T, "--lr", "1e-4")), ("score-w", mv(T, "--score-w", "0.2")),
                    ("batch", mv(T, "--batch", "2")), ("nav", mv(T, "--nav-mode", "goal")), ("no rank-w (R-CTRL)", drop(T, "--rank-w")), ("no heads", drop(T, "--rank-heads")),
                    ("no table", drop(T, "--traj-label-table")), ("no init", drop(T, "--init-from")), ("extra flag", T + ["--field-labels", "/x"]), ("w-field", mv(T, "--w-field", "2.0"))):
        ok(refusals(pT, bad) != "", f"T must refuse: {nm}")
    for nm, bad in (("rank-w on C", mv(C, "--rank-w", "0.5")), ("no rank-w on C", drop(C, "--rank-w")), ("heads on C", C + ["--rank-heads", "ep,gate"]), ("seed on C", mv(C, "--seed", "1"))):
        ok(refusals(pC, bad) != "", f"C must refuse: {nm}")
    ok(refusals(pT, T + ["--rank-w", "0.25"]) != "", "a lever carried twice")
    ok(refusals(pT, T + ["--synthetic"]) != "", "a forbidden flag")
    old = os.environ.get("REFE_RANK_MUTATE")
    os.environ["REFE_RANK_MUTATE"] = "drop"
    try:
        ok("REFE_RANK_MUTATE" in refusals(pT, T) and "REFE_RANK_MUTATE" in refusals(pC, C), "the regression-arm environment switch is refused in a launch")
    finally:
        if old is None:
            os.environ.pop("REFE_RANK_MUTATE")
        else:
            os.environ["REFE_RANK_MUTATE"] = old
    ok(refusals(pT, T) == "" and refusals(pC, C) == "", "env restored")
    # R-CTRL: the control declares rank_L / the treatment omits it
    pc = filled("C")
    pc["live_terms"] = pc["live_terms"] + ("rank_L",)
    ok("declares rank_L live" in refusals(pc, C), "R-CTRL: C declaring rank_L")
    pt = filled("T")
    pt["live_terms"] = L.P6_TERMS
    ok("omit rank_L" in refusals(pt, T), "R-CTRL: T not declaring rank_L")


def chk_fresh_out():
    with tempfile.TemporaryDirectory(prefix="sft_out_") as td:
        (Path(td) / "ckpt_last.pt").write_bytes(b"x")
        bad = mv(targv("T"), "--out", td)
        ok("starts FRESH" in refusals(filled("T"), bad), "a seeded checkpoint is refused")
        ok("starts FRESH" not in refusals(filled("T"), mv(targv("T"), "--out", td + "/new")), "a fresh --out is accepted")


def chk_live_wiring():
    src = (HERE / "gate_checks.py").read_text(encoding="utf-8")
    ok('if profile.get("scorerft_rules"):' in src and "rank_live_rows(cap, profile, arm)" in src and "p5c_live_rows(cap, build_cap, profile)" in src, "check_live routes scorerft_rules")
    ok('profile["traj_label_required"] if "traj_label_required" in profile else profile.get("p5c_arm") in ("C1", "C1r")' in src, "(h) keys on the profile")
    ok('profile.get("scorerft_rules")))' in src, "G-CLOCK requires the bank content pin")
    lg = (HERE / "launch_gate.py").read_text(encoding="utf-8")
    ok('if prof.get("scorerft_rules"):\n        out += scorerft_refusals(prof, argv)' in lg, "profile_refusals routes scorerft_rules")
    ok('prof.get("scorerft_rules"))' in lg, "run_checks builds the BUILD capture for G-LIVE (f)")


def chk_h_generalised():
    """(h): a profile that says `traj_label_required` needs the table whatever the arm's NAME; a profile without the key behaves as before (C1 needs one, C0 / field arms must not carry one)."""
    ds = SimpleNamespace(traj_label=None, traj_label_sha256=None, traj_label_cov=None)
    cap = SimpleNamespace(args=SimpleNamespace(init_from=None), bank_objs=[ds], model=None, step0=None)
    r, _ = GC.p5c_live_rows(cap, SimpleNamespace(step0=None), {"scorerft_arm": "T", "traj_label_required": True, "init_from_sha256": HEX_BASE})
    ok(any("carries no label table" in x for x in r), r)
    ds2 = SimpleNamespace(traj_label=object(), traj_label_sha256="zz", traj_label_cov=None)
    cap2 = SimpleNamespace(args=SimpleNamespace(init_from=None), bank_objs=[ds2], model=None, step0=None)
    r, _ = GC.p5c_live_rows(cap2, SimpleNamespace(step0=None), {"p5c_arm": "C0", "init_from_sha256": HEX_BASE})
    ok(any("must not carry a label table" in x for x in r), "C0 unchanged: still must not carry a table")
    r, _ = GC.p5c_live_rows(cap2, SimpleNamespace(step0=None), {"p7field_arm": "C", "init_from_sha256": HEX_BASE})
    ok(any("must not carry a label table" in x for x in r), "the field arms unchanged")
    r, _ = GC.p5c_live_rows(cap2, SimpleNamespace(step0=None), {"scorerft_arm": "C", "traj_label_required": True, "init_from_sha256": HEX_BASE})
    ok(not any("must not carry a label table" in x for x in r), "a scorerft control carries the table and is NOT refused for it")


# ------------------------------------------------------------------------------------------------ the rows, on synthetic captures
def route_calls(seeds=(20, 21, 22), real=None):
    """Captured-call dicts as the gate's recorder stores them (logits / labels / msk / batch / value / pairs), built from the REAL term on float32 logits."""
    import torch
    calls = []
    for sd in seeds:
        s_, y = make_sets(seed=sd, B=3, M=16)
        msk = np.ones(3, np.float32)
        term = (real or _real_rank_loss()).RankLoss("pick", ep_gap=0.02, gate_topk=16)
        v = term.term(torch.from_numpy(s_), torch.from_numpy(y), torch.from_numpy(msk), 3)
        calls.append({"heads": "pick", "ep_gap": 0.02, "gate_topk": 16, "batch": 3, "logits": s_, "labels": y, "msk": msk, "value": float(v), "pairs": dict(term.last["pairs"])})
    return calls


def chk_route_real():
    """CTRL-ROUTE (b) (card 4.4): on real labelled sets, the pick-only term's gradient on the LOGITS is EXACTLY 0.0 on every label-inconsistent factor, non-zero on live ones, exactly 0.0 outside the pair roles and on DDC; the count of
    label-1 picks is > 0 or the check is REFUSED; `REFE_RANK_MUTATE=pick_noroute` turns it RED, both directly and through `rank_live_rows` (the gate)."""
    RLr = _real_rank_loss()
    old_mod, old_env = sys.modules.get("rank_loss"), os.environ.get("REFE_RANK_MUTATE")
    sys.modules["rank_loss"] = RLr
    try:
        os.environ.pop("REFE_RANK_MUTATE", None)
        calls = route_calls(real=RLr)
        fr = GC.rank_route_facts(calls)
        ok(fr["ran"] and fr["n_pick_nc1"] > 0 and fr["n_pick_dac1"] > 0 and fr["dead_cells"] > 0 and fr["live_cells"] > 0 and fr["live_nonzero"] == fr["live_cells"], fr)
        ok(fr["dead_nonzero"] == 0 and fr["pick_nc_nonzero"] == 0 and fr["pick_dac_nonzero"] == 0 and fr["other_nonzero"] == 0 and fr["other_cells"] > 0, fr)
        ok(GC.judge_rank_route(fr) == [], GC.judge_rank_route(fr))
        os.environ["REFE_RANK_MUTATE"] = "pick_noroute"                                              # the deliberate-regression arm: the routing removed, every factor live
        try:
            frm = GC.rank_route_facts(calls)
        finally:
            os.environ.pop("REFE_RANK_MUTATE", None)
        rm = GC.judge_rank_route(frm)
        ok(frm["dead_nonzero"] > 0 and frm["pick_nc_nonzero"] > 0 and frm["pick_dac_nonzero"] > 0 and any("label-inconsistent" in x for x in rm), (frm, rm))
        ok(frm["live_nonzero"] == frm["live_cells"] and frm["other_nonzero"] == 0, "the unrouted term differs ONLY on the dead cells")
        ok(GC.judge_rank_route({**fr, "live_nonzero": 0}) != [], "a routing that zeroes every cell is RED")
        ok(GC.judge_rank_route({**fr, "other_nonzero": 1}) != [], "a gradient outside the pair roles / on DDC is RED")
        ok(GC.judge_rank_route({**fr, "n_pick_nc1": 0}) != [] and "VACUOUS" in GC.judge_rank_route({**fr, "n_pick_nc1": 0})[0], "no label-1 pick: REFUSED as vacuous")
        ok(GC.judge_rank_route({"ran": False}) != [], "an unrun probe is not a pass")
        lab0 = [dict(c, labels=np.concatenate([np.zeros_like(c["labels"][..., :1]), c["labels"][..., 1:]], -1)) for c in calls]
        fr0 = GC.rank_route_facts(lab0)
        ok(fr0["n_pick_nc1"] == 0 and any("VACUOUS" in x for x in GC.judge_rank_route(fr0)), ("NC label 0 everywhere", fr0))
        # through the gate: rank_live_rows on a P capture
        def live(env):
            cap = reach_cap(heads="pick", seed=1)
            cap.metrics = [{"event": "rank"}] + [{"step": i, "traj_L1": 1.0, "rank_L": 0.4, "rank_pairs_pick": 4} for i in range(2)]
            cap.rank_calls = [dict(c) for c in calls]
            cap.lab_submits = []
            if env:
                os.environ["REFE_RANK_MUTATE"] = env
            try:
                return GC.rank_live_rows(cap, {}, None)
            finally:
                os.environ.pop("REFE_RANK_MUTATE", None)
        r, d = live(None)
        ok(not any("CTRL-ROUTE" in x for x in r) and d["route"]["ran"] and d["route"]["dead_nonzero"] == 0, (r, d.get("route")))
        r2, d2 = live("pick_noroute")
        ok(any("CTRL-ROUTE" in x and "label-inconsistent" in x for x in r2) and d2["route"]["dead_nonzero"] > 0, r2)
    finally:
        if old_mod is None:
            sys.modules.pop("rank_loss", None)
        else:
            sys.modules["rank_loss"] = old_mod
        if old_env is None:
            os.environ.pop("REFE_RANK_MUTATE", None)
        else:
            os.environ["REFE_RANK_MUTATE"] = old_env


def make_sets(seed=7, B=3, M=16):
    rng = np.random.default_rng(seed)
    y = np.zeros((B, M, 6), np.float32)
    y[..., 0] = (rng.random((B, M)) > 0.25)
    y[..., 1] = (rng.random((B, M)) > 0.2)
    y[..., 2] = rng.random((B, M))
    y[..., 3] = (rng.random((B, M)) > 0.4)
    y[..., 4] = (rng.random((B, M)) > 0.1)
    y[..., 5] = 1.0
    s = rng.normal(size=(B, M, 6)).astype(np.float32)
    return s, y


def torch_term(logits, labels, msk, batch, heads, ep_gap, gate_topk, mutate=None):
    """A torch stand-in for the TRAINER's term (DESIGN 2.2) with the env-mutation switches of 3.4: the system under test for the rows."""
    import torch
    F = torch.nn.functional
    s, y = logits.float(), labels.float()
    B, M, _ = s.shape
    safe = (y[..., 0] > 0.5) & (y[..., 1] > 0.5)
    if mutate == "unsafe_ep":
        safe_ep = torch.ones_like(safe)
    else:
        safe_ep = safe
    if mutate == "shuffle":
        y = y.roll(1, dims=1)
        safe, safe_ep = (y[..., 0] > 0.5) & (y[..., 1] > 0.5), (y[..., 0] > 0.5) & (y[..., 1] > 0.5)
    pdms = y[..., 0] * y[..., 1] * (5 * y[..., 2] + 5 * y[..., 3] + 2 * y[..., 4]) / 12
    p = torch.sigmoid(s.detach())
    agg = p[..., 0] * p[..., 1] * (5 * p[..., 2] + 5 * p[..., 3] + 2 * p[..., 4]) / 12
    top = torch.zeros_like(safe)
    ti = torch.topk(agg, int(gate_topk), dim=1).indices
    top.scatter_(1, ti, True)
    hs = {h.strip() for h in heads.split(",") if h.strip()}
    total, pairs = s.new_zeros(()), {}
    for b in range(B):
        for h in ("ep", "gate", "ttc"):
            if h not in hs:
                continue
            if h == "ep":
                m_ = safe_ep[b][:, None] & safe_ep[b][None, :] & ((y[b, :, 2][:, None] - y[b, :, 2][None, :]) >= ep_gap)
                a = (5 / 12) * (y[b, :, 2][:, None] - y[b, :, 2][None, :])
                z = s[b, :, 2][:, None] - s[b, :, 2][None, :]
            elif h == "gate":
                m_ = safe[b][:, None] & (~safe[b])[None, :] & top[b][None, :] & (pdms[b][:, None] > pdms[b][None, :])
                a = pdms[b][:, None] - pdms[b][None, :]
                zz = F.logsigmoid(s[b, :, 0]) + F.logsigmoid(s[b, :, 3 if mutate == "head_swap" else 1])
                z = zz[:, None] - zz[None, :]
            else:
                m_ = safe[b][:, None] & safe[b][None, :] & (y[b, :, 3][:, None] == 1) & (y[b, :, 3][None, :] == 0)
                a = torch.full((M, M), 5 / 12)
                z = s[b, :, 3][:, None] - s[b, :, 3][None, :]
            pairs[h] = pairs.get(h, 0) + int(m_.sum())
            if m_.any():
                af = torch.where(m_, a, torch.zeros_like(a))
                total = total + msk[b] * (af * F.softplus(-z)).sum() / af.sum() / float(batch)
    out = total
    if mutate == "zero":
        out = s.new_zeros(())                                             # a term that always returns exactly 0.0 (no graph)
    if mutate == "drop":
        out = total.detach()                                              # computed, never reaches the loss
    return out, {"pairs": pairs}


class FakeRank:
    mutate = None

    def __init__(self, heads="ep,gate", ep_gap=0.02, gate_topk=8):
        self.heads, self.ep_gap, self.gate_topk = heads, ep_gap, gate_topk

    def facts(self):
        return {"heads": self.heads, "ep_gap": self.ep_gap, "gate_topk": self.gate_topk}

    def term(self, score, labels, msk, batch):
        return torch_term(score, labels, msk, batch, self.heads, self.ep_gap, self.gate_topk, FakeRank.mutate)


def make_model(leaky=False):
    import torch
    nn = torch.nn

    class Fake(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.backbone = nn.Module()
            self.backbone.lo = nn.Module()
            self.backbone.lo.A = nn.Parameter(torch.ones(1, 8) * 0.5)
            self.scene_proj = nn.Linear(8, 8)
            self.queries = nn.Parameter(torch.randn(16, 8) * 0.1)
            self.dec = nn.Linear(8, 8)
            self.traj_head = nn.Linear(8, 4)
            self.score_q_mlp = nn.Sequential(nn.Linear(4, 8))
            self.score_dec = nn.Linear(8, 8)
            self.score_head = nn.Linear(8, 6)

        def forward(self, img, ego, goal, calib=None, detach_scorer=True):
            vis = self.scene_proj(img * self.backbone.lo.A)
            h = self.dec(img[:, None, :] + self.queries[None])
            traj = self.traj_head(h)
            q = self.score_q_mlp((traj if (leaky or not detach_scorer) else traj.detach()))
            s = self.score_head(torch.relu(q + self.score_dec(vis)[:, None, :]))
            return traj, s
    return Fake()


def install_fake_rank_module():
    mod = types.ModuleType("rank_loss")
    mod.RankLoss = FakeRank
    sys.modules["rank_loss"] = mod


def reach_cap(heads="ep,gate", w=W_FIX, leaky=False, seed=3):
    import torch
    m = make_model(leaky)
    s_, y = make_sets(seed=seed, B=1, M=16)
    img = torch.randn(2, 8)
    cap = SimpleNamespace(model=m, first_batch=(img, torch.zeros(2, 5), torch.zeros(2, 2), None), rank_first=[{"labels": y, "msk": np.ones(1, np.float32)}],
                          args=SimpleNamespace(rank_w=w, rank_heads=heads, rank_ep_gap=0.02, rank_gate_topk=8))
    return cap


def chk_reach_and_regressions():
    """L3 on a tiny stand-in model: the CLEAN term passes; R-DROP (detached), R-HEAD (the gate reads the wrong logit) and R-LEAK (traj not detached) each turn it RED; the weight-0 control reads exactly 0.0."""
    install_fake_rank_module()
    FakeRank.mutate = None
    cap = reach_cap()
    fr = GC.rank_reach(cap, {}, None)
    ok(fr["ran"] and fr["labelled"] == 1, fr)
    ok(GC.judge_rank_reach(fr, "ep,gate") == [], GC.judge_rank_reach(fr, "ep,gate"))
    rows = fr["declared"]["score_head_rows"]
    ok(rows["nc"] > 0 and rows["dac"] > 0 and rows["ep"] > 0 and rows["ttc"] == 0.0 and rows["c"] == 0.0 and rows["ddc"] == 0.0, rows)
    ok(all(v["abs_sum"] == 0.0 for v in fr["zero"]["groups"].values()) and all(v == 0.0 for v in fr["zero"]["score_head_rows"].values()), "the weight-0 control reads exactly 0.0")
    ok(fr["declared"]["groups"]["traj_head"]["abs_sum"] == 0.0 and fr["declared"]["groups"]["dec"]["abs_sum"] == 0.0 and fr["declared"]["groups"]["queries"]["abs_sum"] == 0.0, "trajectory side exactly 0.0")
    ok(fr["declared"]["groups"]["backbone.lora"]["abs_sum"] > 0 and fr["declared"]["groups"]["scene_proj"]["abs_sum"] > 0, "the trunk is reached")
    for mut, frag in (("drop", "does not reach a head it trains"), ("head_swap", "row TTC")):
        FakeRank.mutate = mut
        try:
            r = GC.judge_rank_reach(GC.rank_reach(reach_cap(), {}, None), "ep,gate")
            ok(r != [] and any(frag in x for x in r), (mut, r))                                   # R-DROP / R-HEAD: RED
        finally:
            FakeRank.mutate = None
    r = GC.judge_rank_reach(GC.rank_reach(reach_cap(leaky=True), {}, None), "ep,gate")
    ok(any("trajectory side" in x for x in r), ("R-LEAK", r))                                    # R-LEAK: RED on the trajectory zero control
    cap_ttc = reach_cap(heads="ep,gate,ttc")
    r = GC.judge_rank_reach(GC.rank_reach(cap_ttc, {}, None), "ep,gate,ttc")
    ok(r == [], ("ttc named: its row must be non-zero", r))
    cap_ttc2 = reach_cap(heads="ep,gate")
    fr2 = GC.rank_reach(cap_ttc2, {}, None)
    ok(GC.judge_rank_reach(fr2, "ep,gate,ttc") != [], "a TTC head the term does not train but the arm names is RED")
    ok(GC.judge_rank_reach({"ran": False}, "ep,gate") != [] and GC.judge_rank_reach({"ran": True, "labelled": 0}, "ep,gate") != [], "an unrun / unlabelled probe is not a pass")


def audit_call(mutate, seed=11, heads="ep,gate", gap=0.02, topk=8):
    import torch
    s_, y = make_sets(seed=seed)
    msk = np.ones(s_.shape[0], np.float32)
    FakeRank.mutate = mutate
    try:
        out, st = torch_term(torch.from_numpy(s_), torch.from_numpy(y), torch.from_numpy(msk), s_.shape[0], heads, gap, topk, mutate)
    finally:
        FakeRank.mutate = None
    return {"logits": s_, "labels": y, "msk": msk, "batch": int(s_.shape[0]), "heads": heads, "ep_gap": gap, "gate_topk": topk, "value": float(out), "pairs": st["pairs"]}


def chk_audit():
    ref_ok, _ = GC.judge_rank_audit([audit_call(None)], None)
    ok(ref_ok == [], ref_ok)
    c = audit_call(None)
    ref = GC.rank_reference(c["logits"], c["labels"], c["msk"], c["batch"], c["heads"], c["ep_gap"], c["gate_topk"])
    ok(ref["pairs"]["ep"] > 0 and ref["pairs"]["gate"] > 0 and ref["L"] > 0, ref["pairs"])
    ok(abs(ref["L"] - c["value"]) <= 1e-6 * abs(c["value"]), (ref["L"], c["value"]))
    for mut in ("unsafe_ep", "shuffle"):                                                            # R-UNSAFE, R-SHUF: RED by value and by the pair counts
        r, _ = GC.judge_rank_audit([audit_call(mut)], None)
        ok(r != [], mut)
    ok(GC.judge_rank_audit([audit_call("head_swap")], None)[0] != [], "R-HEAD also moves the loss value")
    c2 = audit_call(None)
    c2["value"] *= 1.001
    ok(GC.judge_rank_audit([c2], None)[0] != [], "a 0.1 % value error is caught (tol 1e-6)")
    c3 = audit_call(None)
    c3["pairs"] = dict(c3["pairs"], ep=c3["pairs"]["ep"] + 1)
    ok(GC.judge_rank_audit([c3], None)[0] != [], "a wrong pair count is caught")
    ok(GC.judge_rank_audit([], None)[0] != [], "no captured call is not a pass")
    # analytic identities of the reference itself: a bar below everything / above everything
    s_, y = make_sets()
    msk = np.ones(3, np.float32)
    full = GC.rank_reference(s_, y, msk, 3, "ep,gate", 0.02, 16)
    wide = GC.rank_reference(s_, y, msk, 3, "ep,gate", 0.02, 10 ** 6)
    ok(full["L"] == wide["L"] and full["pair_sets"] == wide["pair_sets"], "topk >= M is the unrestricted reference")
    none = GC.rank_reference(s_, y, msk, 3, "ep", 1.5, 8)
    ok(none["L"] == 0.0 and none["pairs"]["ep"] == 0, "a gap above everything: zero pairs, exactly 0.0")
    part = GC.rank_reference(s_, y, msk, 3, "ep", 0.5, 8)
    ok(0 < part["pairs"]["ep"] < GC.rank_reference(s_, y, msk, 3, "ep", 0.02, 8)["pairs"]["ep"], "a gap in the middle: partial")
    ok(GC.rank_reference(s_, y, np.zeros(3, np.float32), 3, "ep,gate", 0.02, 8)["L"] == 0.0, "an unlabelled set (mask 0) contributes exactly 0")


def dvb_cap(w=W_FIX, accum=2, steps=2, **over):
    n = steps * accum
    a = SimpleNamespace(rank_w=w, rank_heads="ep,gate" if w else "", rank_ep_gap=0.02, rank_gate_topk=8)
    sa = SimpleNamespace(steps=steps, accum=accum)
    if w:
        calls = [{"heads": "ep,gate", "ep_gap": 0.02, "gate_topk": 8} for _ in range(n)]
        grads = [w / accum] * n
        metrics = [{"event": "rank", "heads": ["ep", "gate"], "ep_gap": 0.02, "gate_topk": 8, "aggregate": "navsim_v1", "reduce": "set-wmean/const-B", "mutate": None}] + [
            {"step": i, "traj_L1": 1.0, "rank_w": w, "rank_L": 0.5, "rank_pairs_ep": 5, "rank_pairs_gate": 3} for i in range(steps)]
    else:
        calls, grads, metrics = [], [], [{"step": i, "traj_L1": 1.0} for i in range(steps)]
    s = SimpleNamespace(args=sa, rank_calls=calls, rank_grads=grads, metrics=metrics)
    for k, v in over.items():
        setattr(s, k, v)
    return a, s


def dvb_ok(a, s):
    return all(r["ok"] for r in GC.rank_dvb_rows(a, None, s)), [r for r in GC.rank_dvb_rows(a, None, s) if not r["ok"]]


def chk_dvb_rows():
    a, s = dvb_cap()
    good, bad = dvb_ok(a, s)
    ok(good, bad)
    a, s = dvb_cap(rank_grads=[])
    ok(not dvb_ok(a, s)[0], "R-DROP: the hook never fires")
    a, s = dvb_cap(rank_grads=[0.5 * W_FIX / 2] * 4)
    ok(not dvb_ok(a, s)[0], "the multiplier wired off by a factor reads the factor")
    a, s = dvb_cap(rank_grads=[W_FIX / 2] * 3)
    ok(not dvb_ok(a, s)[0], "a term that fired on only some micro-batches")
    a, s = dvb_cap(rank_calls=[{"heads": "ep,gate", "ep_gap": 0.02, "gate_topk": 8}] * 3)
    ok(not dvb_ok(a, s)[0], "the term is called once per micro-batch")
    a, s = dvb_cap()
    s.metrics[0]["heads"] = ["ep"]
    ok(not dvb_ok(a, s)[0], "the event's heads must equal the argv's")
    a, s = dvb_cap()
    s.rank_calls[0]["gate_topk"] = 4
    ok(not dvb_ok(a, s)[0], "the kwargs the term was called with must equal the argv's")
    a, s = dvb_cap()
    s.metrics[0]["mutate"] = "drop"
    ok(not dvb_ok(a, s)[0], "a term built under REFE_RANK_MUTATE can never count")
    a, s = dvb_cap()
    s.metrics = [m for m in s.metrics if m.get("event") != "rank"]
    ok(not dvb_ok(a, s)[0], "no `rank` start event")
    a, s = dvb_cap()
    s.metrics[1]["rank_w"] = 0.5
    ok(not dvb_ok(a, s)[0], "the logged rank_w must be the argv's on every row")
    a, s = dvb_cap(w=0.0)
    ok(dvb_ok(a, s)[0], "the control: nothing fires, nothing is logged")
    a, s = dvb_cap(w=0.0)
    s.rank_grads = [0.1]
    ok(not dvb_ok(a, s)[0], "R-CTRL: a control whose hook fires")
    a, s = dvb_cap(w=0.0)
    s.metrics[0]["rank_L"] = 0.3
    ok(not dvb_ok(a, s)[0], "R-CTRL: a control that logs rank_L")
    ok(GC.rank_dvb_rows(SimpleNamespace(rank_w=0.0), None, None) == [], "no smoke capture, no rows")


def chk_live_rows_control_and_t():
    install_fake_rank_module()
    FakeRank.mutate = None
    capC = SimpleNamespace(args=SimpleNamespace(rank_w=0.0, rank_heads=""), metrics=[{"step": 0, "traj_L1": 1.0}], rank_calls=[])
    ok(GC.rank_live_rows(capC, {}, None)[0] == [], "control passes")
    capC.metrics[0]["rank_L"] = 0.1
    ok(GC.rank_live_rows(capC, {}, None)[0] != [], "R-CTRL: control logging rank_L is RED")
    cap = reach_cap()
    cap.metrics = [{"event": "rank"}] + [{"step": i, "traj_L1": 1.0, "rank_L": 0.4, "rank_pairs_ep": 4, "rank_pairs_gate": 2} for i in range(2)]
    cap.rank_calls = [audit_call(None)]
    cap.lab_submits = []
    r, d = GC.rank_live_rows(cap, {}, None)
    ok(len(r) == 1 and "L5" in r[0], r)                                                              # only L5 (NAVSIM-bound, pod-only) is unproved on a dev box; L1-L4 pass
    cap.rank_traj_reach = {"n": 2, "n_none": 0, "abs_sum": 0.5}
    ok(any("R-LEAK" in x for x in GC.rank_live_rows(cap, {}, None)[0]), "R-LEAK: the score the term got reaches traj_head")
    cap.rank_traj_reach = {"n": 2, "n_none": 2, "abs_sum": 0.0}
    cap.metrics[1]["rank_pairs_gate"] = 0
    cap.metrics[2]["rank_pairs_gate"] = 0
    ok(any("L2" in x and "gate" in x for x in GC.rank_live_rows(cap, {}, None)[0]), "a head with zero pairs is structurally dead")
    cap.metrics[1].pop("rank_L")
    ok(any("L1" in x for x in GC.rank_live_rows(cap, {}, None)[0]), "rank_L absent from a row")
    ok(GC.judge_harness_identity({"ran": True, "tokens": 8, "max_abs_diff": 0.0}) == [] and GC.judge_harness_identity({"ran": True, "tokens": 8, "max_abs_diff": 1e-6}) != []
       and GC.judge_harness_identity({"ran": False, "why": "x"}) != [] and GC.judge_harness_identity({"ran": True, "tokens": 0, "max_abs_diff": 0.0}) != [], "L5 judge")


def chk_trainer_integration():
    """Re-run once the trainer half has landed: train.py's parser carries the four flags (default OFF), the registry == the parser (stale / unknown), and the real `rank_loss.RankLoss` passes L3 / L4 on the stand-in model."""
    import train as T
    dests = {a.dest for a in T.build_parser()._actions if a.dest != "help"}
    have = {"rank_w", "rank_heads", "rank_ep_gap", "rank_gate_topk"} & dests
    if not have:
        print("  INTEGRATION_PENDING: train.py has none of --rank-w/--rank-heads/--rank-ep-gap/--rank-gate-topk (the trainer half is not in this tree): the registry<->parser identity, the real term's L3/L4 and the live G-DVB/G-LIVE run are NOT proved here", flush=True)
        return
    ok(have == {"rank_w", "rank_heads", "rank_ep_gap", "rank_gate_topk"}, have)
    d = T.build_parser().parse_args([])
    ok(d.rank_w == 0.0 and d.rank_heads == "" and d.rank_ep_gap == 0.02 and d.rank_gate_topk == 8, "defaults OFF")
    ok(set(GC.DVB_KINDS) == dests, sorted(set(GC.DVB_KINDS) ^ dests))
    import importlib
    sys.modules.pop("rank_loss", None)
    RL = importlib.import_module("rank_loss")
    ok(hasattr(RL, "RankLoss"), "rank_loss.RankLoss")
    FakeRank.mutate = None
    cap = reach_cap()
    fr = GC.rank_reach(cap, {}, None)
    ok(GC.judge_rank_reach(fr, "ep,gate") == [], GC.judge_rank_reach(fr, "ep,gate"))
    import torch
    c = audit_call(None)
    out = RL.RankLoss(heads="ep,gate", ep_gap=0.02, gate_topk=8).term(torch.from_numpy(c["logits"]), torch.from_numpy(c["labels"]), torch.from_numpy(c["msk"]), c["batch"])
    out = out[0] if isinstance(out, tuple) else out
    ref = GC.rank_reference(c["logits"], c["labels"], c["msk"], c["batch"], "ep,gate", 0.02, 8)
    ok(abs(float(out) - ref["L"]) <= 1e-6 * abs(ref["L"]), (float(out), ref["L"]))


def chk_no_scipy():
    src = (HERE / "test_scorerft_profile.py").read_text(encoding="utf-8")
    ok(not re.search(r"^\s*(import|from)\s+scipy", src, re.M))
    for m in re.finditer(r"subprocess\.run\(", src):
        depth, j = 0, m.end() - 1
        for k in range(j, len(src)):
            depth += src[k] == "("
            depth -= src[k] == ")"
            if depth == 0:
                break
        call = src[j:k + 1]
        ok("text=True" not in call or "encoding=" in call, "subprocess.run with text=True and no encoding=")


def chk_census_l2():
    """SFTT launch gate G-LIVE L2 (MM 2026-11-23): a head the smoke leaves at ZERO pairs passes iff the registered, hash-pinned census proves pairs > 0 under the DECLARED rule.  Fixtures are built in a tmp dir and the pin
    is the fixture's own sha (profile['rank_census']); the imported rank_loss is a fake module whose __file__ is a fixture."""
    import json
    install_fake_rank_module()
    FakeRank.mutate = None
    pr = L.scorerft_profile("T")
    ok(pr.get("rank_census") == L.SCORERFT_RANK_CENSUS and all(len(L.SCORERFT_RANK_CENSUS[k]) == 64 for k in ("sha256", "rank_loss_sha256")), "the profile of a ranking arm carries the pin")
    ok("rank_census" not in L.scorerft_profile("C") and "rank_census" not in L.scorerft_profile("Cs"), "a control arm carries no census")
    ok(L.SCORERFT_RANK_CENSUS["sha256"].startswith("a9f13c78") and L.SCORERFT_RANK_CENSUS["rank_loss_sha256"].startswith("c0e027d2"), "the pins are the MM's")
    INIT = "/workspace/data/refe_runs/p5c_c1/model_final.pt"
    base = {"heads": ["ep", "gate"], "ep_gap": 0.02, "gate_topk": 8, "label_backend": "pdm", "NOT_A_DECISION": False, "init_from": INIT, "rule": {"status": "SELECT", "rank_w": W_FIX},
            "pairs_by_head": {"ep": 100, "gate": 5}, "sets_with_pairs_by_head": {"ep": 20, "gate": 2}, "labelled_sets": 32}
    td = Path(tempfile.mkdtemp(prefix="sftcen_"))
    try:
        rl = td / "rank_loss.py"
        rl.write_bytes(b"# fixture rank_loss\r\nx = 1\r\n")
        rl_sha = hashlib.sha256(b"# fixture rank_loss\nx = 1\n").hexdigest()
        rl2 = td / "rank_loss_other.py"
        rl2.write_bytes(b"# a different rank_loss\n")

        def case(over=None, rank_loss_file=rl, pin_over=None, write=True, smoke_gate=0, drop_msk=False):
            d = json.loads(json.dumps(base))
            d.update(over or {})
            path = td / f"census_{len(list(td.iterdir()))}.json"
            if write:
                path.write_text(json.dumps(d), encoding="utf-8")
            pin = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if write else "00" * 32, "rank_loss_sha256": rl_sha}
            pin.update(pin_over or {})
            profile = {"rank_census": pin}
            sys.modules["rank_loss"].__file__ = str(rank_loss_file)
            cap = reach_cap()
            cap.args.init_from = INIT
            cap.metrics = [{"event": "rank"}] + [{"step": i, "traj_L1": 1.0, "rank_L": 0.4, "rank_pairs_ep": 4, "rank_pairs_gate": smoke_gate} for i in range(2)]
            cap.rank_calls = [audit_call(None)]
            if drop_msk:                                       # a 4th call: the trainer keeps the mask of the first 3 calls only
                cap.rank_calls += [dict(cap.rank_calls[0]) for _ in range(3)]
                for c_ in cap.rank_calls[3:]:
                    c_.pop("msk")
            cap.lab_submits = []
            r, det = GC.rank_live_rows(cap, profile, None)
            return [x for x in r if x.startswith("L2")], [x for x in r if not x.startswith("L2")], det

        # PASS (a): zero smoke gate pairs + a valid census
        l2, rest, det = case()
        ok(l2 == [] and len(rest) == 1 and "L5" in rest[0], (l2, rest))
        ok(det["census"]["verdict"]["gate"] == {"source": "census", "ok": True} and det["census"]["verdict"]["ep"] == {"source": "smoke", "ok": True}, det["census"]["verdict"])
        ok(det["census"]["n_smoke_sets"] == 3 and abs(det["census"]["p_zero_smoke"]["gate"] - (1 - 2 / 32) ** 3) < 1e-12, det["census"])
        ok(det["census"]["facts"]["sha256"] and det["census"]["facts"]["rank_loss_sha256"] == rl_sha, "the facts record both shas")
        ok(case(drop_msk=True)[2]["census"]["n_smoke_sets"] is None and case(drop_msk=True)[2]["census"]["p_zero_smoke"]["gate"] is None, "an unrecoverable smoke set count is None, never invented")
        # PASS (b): smoke pairs > 0 and NO census file: the old path, untouched
        l2, rest, det = case(write=False, smoke_gate=3)
        ok(l2 == [] and "census" not in det and det["pairs_logged"]["gate"] == 6, (l2, det.get("census")))
        # FAIL (c)-(k): each ONE reason, the original L2 text verbatim + the census clause
        fails = {"c_no_census_file": dict(write=False), "d_sha_mismatch": dict(pin_over={"sha256": "ab" * 32}), "e_census_gate_pairs_zero": dict(over={"pairs_by_head": {"ep": 100, "gate": 0}}),
                 "e2_census_gate_sets_zero": dict(over={"sets_with_pairs_by_head": {"ep": 20, "gate": 0}}), "f_gate_topk_16": dict(over={"gate_topk": 16}), "g_init_from_differs": dict(over={"init_from": "/x/other.pt"}),
                 "h_rank_loss_sha_differs": dict(rank_loss_file=rl2), "i_not_a_decision": dict(over={"NOT_A_DECISION": True}), "j_synthetic": dict(over={"label_backend": "synthetic"}),
                 "k_heads_ep_only": dict(over={"heads": ["ep"]}), "l_ep_gap": dict(over={"ep_gap": 0.05}), "m_rank_w": dict(over={"rule": {"status": "SELECT", "rank_w": 0.5}}),
                 "n_pin_rank_loss_sha": dict(pin_over={"rank_loss_sha256": "cd" * 32}), "o_no_pin": dict(pin_over={"sha256": ""})}
        for nm, kw in fails.items():
            l2, rest, det = case(**kw)
            ok(len(l2) == 1 and l2[0].startswith("L2 head 'gate' formed ZERO pairs over the whole smoke (structurally dead: the 'built, tested, unreachable' class); the registered census does not prove it: "), (nm, l2))
            ok(det["census"]["verdict"]["gate"] == {"source": "census", "ok": False}, nm)
        # judge / facts never raise
        ok(GC.rank_census_facts({}) ["ran"] is False and GC.rank_census_facts(None)["ran"] is False and GC.rank_census_facts({"rank_census": {"path": str(td / "nope.json")}})["ran"] is False, "facts: never raise")
        (td / "bad.json").write_text("{not json", encoding="utf-8")
        ok(GC.rank_census_facts({"rank_census": {"path": str(td / "bad.json")}})["ran"] is False, "facts: a corrupt file")
        ok(GC.judge_census_head("gate", {"ran": False, "why": "x"}, SimpleNamespace(), {}) != [], "judge: an unrun census")
    finally:
        shutil.rmtree(td, ignore_errors=True)


def chk_field_reach_active_sample():
    """SFTT launch gate 1b (MM 2026-11-23, Fs): `field_reach` probes the FIRST sample whose hinge loss is > 0, not idx[0].  A fake `field_loss` (hinge on a per-sample scalar 'sdf') stands in for the real one."""
    import torch
    mod = types.ModuleType("field_loss")

    def field_loss(traj, sdf, margin=0.0, interp=None):                    # sum_b relu(margin - s_b) * mean(traj_b): exactly 0 (loss AND gradient) where s_b >= margin
        s = sdf.reshape(sdf.shape[0], -1)[:, 0]
        return (torch.relu(margin - s) * traj.reshape(traj.shape[0], -1).mean(1)).sum() / traj.shape[0]
    mod.field_loss = field_loss
    old = sys.modules.get("field_loss")
    sys.modules["field_loss"] = mod
    try:
        def fcap(s_vals, valid=None):
            B = len(s_vals)
            sdf = torch.tensor(s_vals, dtype=torch.float32).reshape(B, 1, 1)
            return SimpleNamespace(model=make_model(), first_batch=(torch.randn(B, 8), torch.zeros(B, 5), torch.zeros(B, 2), None), field_first=(sdf, valid),
                                   args=SimpleNamespace(w_field=2.0, field_margin=0.5, field_interp="5hz"))
        fr = GC.field_reach(fcap([5.0, 5.0, 0.1, 0.1]), None)                      # index 0 inactive (sdf 5 >= margin), index 2 the first active
        ok(fr["active_index"] == 2 and fr["n_active"] == 2 and fr["losses"][0] == 0.0 and fr["losses"][1] == 0.0 and fr["losses"][2] > 0.0, fr.get("losses"))
        ok(fr["declared"]["loss"] != 0.0 and fr["declared"]["groups"]["traj_head"]["abs_sum"] > 0.0 and fr["declared"]["groups"]["queries"]["abs_sum"] > 0.0, "an active sample reaches the groups")
        ok(all(v["abs_sum"] == 0.0 for v in fr["zero"]["groups"].values()), "the weight-0 control reads exactly 0.0")
        rs = GC.judge_field_reach(fr, 2.0)
        ok(not any("EXACTLY zero on 'traj_head'" in x or "EXACTLY zero on 'queries'" in x or "UNPROVED" in x for x in rs), rs)
        fr0 = GC.field_reach(fcap([0.1, 5.0, 5.0, 5.0]), None)                     # the old path: index 0 active
        ok(fr0["active_index"] == 0 and fr0["n_active"] == 1 and fr0["declared"]["groups"]["traj_head"]["abs_sum"] > 0.0, "index 0 active: unchanged")
        fr_n = GC.field_reach(fcap([5.0, 5.0, 5.0, 5.0]), None)                    # no labelled sample is active
        ok(fr_n["ran"] and fr_n["labelled"] == 4 and fr_n["n_active"] == 0 and "declared" not in fr_n, fr_n)
        rn = GC.judge_field_reach(fr_n, 2.0)
        ok(len(rn) == 1 and "UNPROVED" in rn[0] and "every hinge loss is exactly 0.0" in rn[0] and "does not reach" not in rn[0], rn)
        frv = GC.field_reach(fcap([0.1, 5.0, 0.1, 0.1], valid=torch.tensor([False, True, True, True])), None)   # an unlabelled sample is skipped; first ACTIVE labelled = 2
        ok(frv["labelled"] == 3 and frv["active_index"] == 2, frv.get("active_index"))
        ok(GC.judge_field_reach({"ran": True, "labelled": 0}, 2.0) != [] and GC.judge_field_reach({"ran": False}, 2.0) != [], "unchanged: unlabelled / not run")
        # 1c (MM 2026-11-24): G-LIVE's evidence form of EVERY result shape -- the Fs gate crashed in check_live on 1b's `losses` list, a caller these lines never reached
        for f_ in (fr, fr0, fr_n, frv, {"ran": False}, {"ran": True, "labelled": 0}):
            d_ = GC.field_reach_detail(f_)
            ok(set(d_) == set(f_) and all(d_[k] == f_[k] for k in f_ if not isinstance(f_[k], dict)), d_)
        d_ = GC.field_reach_detail(fr)
        ok(set(d_["declared"]) == {"loss", "groups"} and set(d_["zero"]) == {"loss", "groups"} and d_["losses"] == fr["losses"] and d_["n_active"] == 2 and d_["active_index"] == 2, d_)
        ok("det[\"field_reach\"] = field_reach_detail(fr)" in inspect.getsource(GC.check_live), "check_live builds its field_reach evidence through field_reach_detail")
    finally:
        if old is None:
            sys.modules.pop("field_loss", None)
        else:
            sys.modules["field_loss"] = old


def chk_l5_identity_dtype():
    """G-LIVE L5 (SFTT gate 2, 2026-11-23): the sub-scores are read from a PDMResults DATACLASS or a DataFrame row, and the identity is judged at the LABEL's dtype.
    Numbers MEASURED on the pod (l5probe): ep NAVSIM 0.9281340780250414 vs label 0.9281340837478638 == float32(NAVSIM)."""
    import dataclasses
    names = {"nc": "no_at_fault_collisions", "dac": "drivable_area_compliance", "ep": "ego_progress", "ttc": "time_to_collision_within_bound", "c": "comfort", "ddc": "driving_direction_compliance"}

    @dataclasses.dataclass
    class FakePDMResults:                                   # the navsim 3e8291b return type (fields as navsim/common/dataclasses.py:557)
        no_at_fault_collisions: float = 1.0
        drivable_area_compliance: float = 1.0
        ego_progress: float = 0.9281340780250414
        time_to_collision_within_bound: float = 1.0
        comfort: float = 1.0
        driving_direction_compliance: float = 1.0
        score: float = 0.97

    class FakeFrame:                                        # the one-row DataFrame form: .iloc[0][col]
        class _I:
            def __getitem__(self, i):
                return dict(dataclasses.asdict(FakePDMResults()))
        iloc = _I()
    sd = GC._pdm_subscores(FakePDMResults(), names)
    ok(sd["ego_progress"] == 0.9281340780250414 and set(sd) == set(names.values()), "L5: a PDMResults dataclass is read by attribute")
    sf = GC._pdm_subscores(FakeFrame(), names)
    ok(sf == sd, "L5: a one-row DataFrame form is read by column, identically")
    nav, lab32 = 0.9281340780250414, float(np.float32(0.9281340780250414))
    ok(lab32 == 0.9281340837478638, "L5: the measured label IS float32(NAVSIM) (the pod's numbers)")
    d32 = GC.identity_diff(nav, lab32, np.float32)
    ok(d32 == 0.0 and GC.judge_harness_identity({"ran": True, "tokens": 8, "max_abs_diff": d32}) == [], "L5: a float32 label equal to float32(NAVSIM) is the identity (PASS)")
    ulp = float(np.nextafter(np.float32(lab32), np.float32(2.0)))
    d1 = GC.identity_diff(nav, ulp, np.float32)
    ok(d1 > 1e-9 and GC.judge_harness_identity({"ran": True, "tokens": 8, "max_abs_diff": d1}) != [], "L5: a float32 label ONE ulp away FAILS")
    d64 = GC.identity_diff(nav, lab32, np.float64)
    ok(d64 > 1e-9 and GC.judge_harness_identity({"ran": True, "tokens": 8, "max_abs_diff": d64}) != [], "L5: float64 labels are compared at float64 (5.7e-9 > 1e-9 FAILS)")
    ok(GC.identity_diff(nav, nav, np.float64) == 0.0, "L5: an exact float64 label is the identity")
    ok(GC.identity_diff(1.25, 1.0, np.float32) == 0.0 and GC.identity_diff(-0.5, 0.0, np.float32) == 0.0, "L5: NAVSIM's value is clipped to [0, 1] first")


def reach_cap_multi(y, msk=None, heads="ep,gate", w=W_FIX):
    """a capture whose first batch has len(y) samples (the pod's batch is 4), labels `y` (B, M, 6) and per-sample mask `msk`."""
    import torch
    B = y.shape[0]
    img = torch.randn(B, 8)
    return SimpleNamespace(model=make_model(False), first_batch=(img, torch.zeros(B, 5), torch.zeros(B, 2), None),
                           rank_first=[{"labels": y, "msk": np.ones(B, np.float32) if msk is None else np.asarray(msk, np.float32)}],
                           args=SimpleNamespace(rank_w=w, rank_heads=heads, rank_ep_gap=0.02, rank_gate_topk=8))


def _pairless(y, i):
    """sample i's labels made identical on every proposal: no EP gap, no safe / unsafe split, no proposal PDMS-better than another: the term forms NO pair (exactly 0.0, no gradient)."""
    y = y.copy()
    y[i] = 1.0
    return y


def chk_reach_scan():
    """MM 2026-11-25 (the pod gate FAIL of ecac87ed98de, L3 read exactly zero on every row): `rank_reach` probes the FIRST labelled sample whose declared-weight term loss is non-zero, not sample 0.
    sample 0 pairless + sample k paired => passes on k (sample_index / scanned recorded); all pairless => FAIL; a masked sample is not scanned; R-DROP, an always-0 term and R-HEAD stay RED."""
    install_fake_rank_module()
    FakeRank.mutate = None
    _, y = make_sets(seed=5, B=4, M=16)
    f0 = GC.rank_reach(reach_cap_multi(y), {}, None)
    ok(f0["sample_index"] == 0 and f0["scanned"] == 1 and f0["found"] is True and GC.judge_rank_reach(f0, "ep,gate") == [], ("paired sample 0 is probed as before", f0["sample_index"], f0["scanned"]))
    y1 = _pairless(y, 0)
    fr = GC.rank_reach(reach_cap_multi(y1), {}, None)
    ok(fr["sample_index"] == 1 and fr["scanned"] == 2 and fr["found"] is True and fr["labelled"] == 1, (fr["sample_index"], fr["scanned"]))
    ok(GC.judge_rank_reach(fr, "ep,gate") == [], GC.judge_rank_reach(fr, "ep,gate"))
    ok(all(v["abs_sum"] == 0.0 for v in fr["zero"]["groups"].values()), "the weight-0 control (on the SAME sample) reads exactly 0.0")
    y2 = _pairless(_pairless(y1, 1), 2)
    fr2 = GC.rank_reach(reach_cap_multi(y2), {}, None)
    ok(fr2["sample_index"] == 3 and fr2["scanned"] == 4 and GC.judge_rank_reach(fr2, "ep,gate") == [], (fr2["sample_index"], fr2["scanned"]))
    y_all = y.copy()
    for i in range(4):
        y_all = _pairless(y_all, i)
    fa = GC.rank_reach(reach_cap_multi(y_all), {}, None)
    ok(fa["found"] is False and fa["sample_index"] == 0 and fa["scanned"] == 4, (fa["found"], fa["sample_index"], fa["scanned"]))
    ra = GC.judge_rank_reach(fa, "ep,gate")
    ok(any("EXACTLY zero" in x for x in ra), ("all pairless => FAIL", ra))
    fm = GC.rank_reach(reach_cap_multi(y, msk=[0, 1, 1, 1]), {}, None)
    ok(fm["sample_index"] == 1 and fm["scanned"] == 1 and fm["labelled"] == 1, ("a masked sample is not scanned", fm["sample_index"], fm["scanned"]))
    fn = GC.rank_reach(reach_cap_multi(y, msk=[0, 0, 0, 0]), {}, None)
    ok(fn["labelled"] == 0 and GC.judge_rank_reach(fn, "ep,gate") != [], "no labelled sample is not a pass")
    for mut, frag in (("drop", "does not reach a head it trains"), ("zero", "EXACTLY zero"), ("head_swap", "row TTC")):
        FakeRank.mutate = mut
        try:
            r = GC.judge_rank_reach(GC.rank_reach(reach_cap_multi(y1), {}, None), "ep,gate")
            ok(r != [] and any(frag in x for x in r), (mut, "stays RED with a pairless sample 0", r))
        finally:
            FakeRank.mutate = None
    cap_l = reach_cap_multi(y1)
    cap_l.model = make_model(True)
    ok(any("trajectory side" in x for x in GC.judge_rank_reach(GC.rank_reach(cap_l, {}, None), "ep,gate")), "R-LEAK stays RED on the scanned sample")
    # the REAL term, head `pick`: sample 0 constant labels (nothing is PDMS-better than the scorer's pick: no pair), the others random
    RLr = _real_rank_loss()
    old_mod = sys.modules.get("rank_loss")
    sys.modules["rank_loss"] = RLr
    try:
        done = False
        for seed in range(8):
            _, yp = make_sets(seed=40 + seed, B=4, M=16)
            frp = GC.rank_reach(reach_cap_multi(_pairless(yp, 0), heads="pick"), {}, None)
            if not frp["found"]:
                continue
            done = True
            ok(frp["sample_index"] >= 1 and GC.judge_rank_reach(frp, "pick") == [], (seed, frp["sample_index"], GC.judge_rank_reach(frp, "pick")))
            break
        ok(done, "no seed gave the stand-in model a non-oracle pick on a sample >= 1")
        fpa = GC.rank_reach(reach_cap_multi(_pairless(_pairless(_pairless(_pairless(yp, 0), 1), 2), 3), heads="pick"), {}, None)
        ok(fpa["found"] is False and fpa["forced"] is True and GC.judge_rank_reach(fpa, "pick") == [], ("pick: every sample pairless -> the FORCED pair probe runs and proves the reach", GC.judge_rank_reach(fpa, "pick")))
        fpe = GC.rank_reach(reach_cap_multi(_pairless(_pairless(_pairless(_pairless(y, 0), 1), 2), 3), heads="ep,gate"), {}, None)
        ok(fpe["found"] is False and fpe["forced"] is False and GC.judge_rank_reach(fpe, "ep,gate") != [], "a DENSE head set is never forced: every sample pairless stays FAIL")
    finally:
        if old_mod is None:
            sys.modules.pop("rank_loss", None)
        else:
            sys.modules["rank_loss"] = old_mod
    # the capture detail keeps the new keys (rank_live_rows copies every non-scalar value into a loss / groups dict)
    install_fake_rank_module()
    cap = reach_cap_multi(y1)
    cap.metrics = [{"event": "rank"}] + [{"step": i, "traj_L1": 1.0, "rank_L": 0.4, "rank_pairs_ep": 4, "rank_pairs_gate": 1} for i in range(2)]
    cap.rank_calls = [audit_call(None)]
    cap.lab_submits = []
    _, det = GC.rank_live_rows(cap, {}, None)
    ok(det["rank_reach"]["sample_index"] == 1 and det["rank_reach"]["scanned"] == 2 and det["rank_reach"]["found"] is True, det["rank_reach"])


def _wrapped_rank_module(kind):
    """the REAL rank_loss module with a RankLoss whose `term` is broken in the way `kind` names (the forced-pair probe's RED arms): 'drop' = a graph-less value (R-DROP), 'zero' = a term that is always exactly 0.0 with a graph,
    'head' = the aggregate reads the DDC logit where it must read C (R-HEAD)."""
    RLr = _real_rank_loss()
    mod = types.ModuleType("rank_loss")
    mod.navsim_v1, mod.pick_index, mod.HEADS = RLr.navsim_v1, RLr.pick_index, RLr.HEADS

    class Broken(RLr.RankLoss):
        def term(self, score, labels, msk, batch):
            if kind == "head":
                score = score[..., [0, 1, 2, 3, 5, 4]]
            r = super().term(score, labels, msk, batch)
            if kind == "drop":
                return r.detach()
            if kind == "zero":
                return score.sum() * 0.0
            return r
    mod.RankLoss = Broken
    return mod


def chk_forced_pick():
    """MM 2026-11-25 (99ddf53d1461: G-LIVE L3 read exactly zero because no scanned sample formed a PICK pair, while the trainer's own call 0 had 41): when the real scan finds no pair and `pick` is declared, `rank_reach`
    rewrites the labels of the first labelled sample so a pair MUST exist and proves the reach on that.  (1) pairless real scan -> forced, j* != the pick, every expected row non-zero, DDC / trajectory side / weight-0 control
    exactly 0.0; (2) a real pair on sample 0 -> no forcing, as before; (3) the forced probe with a dead term is RED: R-DROP, an always-0 term, R-HEAD, R-LEAK."""
    old_mod = sys.modules.get("rank_loss")
    RLr = _real_rank_loss()
    sys.modules["rank_loss"] = RLr
    try:
        _, y = make_sets(seed=77, B=4, M=16)
        yall = y
        for i in range(4):
            yall = _pairless(yall, i)
        fr = GC.rank_reach(reach_cap_multi(yall, heads="pick"), {}, None)
        ok(fr["found"] is False and fr["forced"] is True and fr["sample_index"] == 0 and fr["scanned"] == 4, (fr["found"], fr["forced"], fr["sample_index"], fr["scanned"]))
        ok(fr["forced_j"] is not None and fr["forced_pick"] is not None and fr["forced_j"] != fr["forced_pick"], (fr["forced_j"], fr["forced_pick"]))
        ok(fr["declared"]["loss"] > 0.0 and len(fr["scan"]) == 4 and all(r["pdms_pick"] == r["pdms_max"] for r in fr["scan"]), fr["scan"])
        rows = fr["declared"]["score_head_rows"]
        ok(all(rows[h] > 0.0 for h in ("nc", "dac", "ep", "ttc", "c")) and rows["ddc"] == 0.0, rows)
        g = fr["declared"]["groups"]
        ok(g["traj_head"]["abs_sum"] == 0.0 and g["backbone.lora"]["abs_sum"] > 0 and g["scene_proj"]["abs_sum"] > 0 and g["score_dec"]["abs_sum"] > 0, g)
        ok(all(v["abs_sum"] == 0.0 for v in fr["zero"]["groups"].values()) and all(v == 0.0 for v in fr["zero"]["score_head_rows"].values()), "the weight-0 control (on the SAME forced labels) reads exactly 0.0")
        ok(GC.judge_rank_reach(fr, "pick") == [], GC.judge_rank_reach(fr, "pick"))
        lab, j, pk = GC.forced_pick_labels(__import__("torch").zeros(1, 5, 6))                     # all-tied scores: argmin == argmax == 0, the forced j* is moved off the pick
        ok(j != pk and float(lab[0, j].min()) == 1.0 and float(lab.sum()) == 6.0, (j, pk))
        # (2) a real pair on sample 0 -> the real scan is used, nothing is forced
        done = False
        for seed in range(8):
            _, yp = make_sets(seed=40 + seed, B=4, M=16)
            f2 = GC.rank_reach(reach_cap_multi(yp, heads="pick"), {}, None)
            if f2["found"]:
                done = True
                ok(f2["forced"] is False and f2["forced_j"] is None and f2["scanned"] >= 1 and GC.judge_rank_reach(f2, "pick") == [], (seed, f2["forced"], f2["scanned"]))
                break
        ok(done, "no seed gave a real pick pair")
        # (3) the RED arms under forcing
        for kind, frag in (("drop", "EXACTLY zero"), ("zero", "EXACTLY zero"), ("head", "row C")):
            sys.modules["rank_loss"] = _wrapped_rank_module(kind)
            try:
                fk = GC.rank_reach(reach_cap_multi(yall, heads="pick"), {}, None)
                r = GC.judge_rank_reach(fk, "pick")
                ok(fk["forced"] is True and r != [] and any(frag in x for x in r), (kind, fk["forced"], r))
            finally:
                sys.modules["rank_loss"] = RLr
        cap_l = reach_cap_multi(yall, heads="pick")
        cap_l.model = make_model(True)
        r = GC.judge_rank_reach(GC.rank_reach(cap_l, {}, None), "pick")
        ok(any("trajectory side" in x for x in r), ("R-LEAK stays RED under forcing", r))
    finally:
        if old_mod is None:
            sys.modules.pop("rank_loss", None)
        else:
            sys.modules["rank_loss"] = old_mod


P_CENSUS_FIX = {"heads": ["pick"], "ep_gap": 0.02, "gate_topk": 8, "label_backend": "pdm", "NOT_A_DECISION": False, "init_from": L.SCORERFT_DEFAULT_INIT, "rule": {"status": "SELECT", "rank_w": 0.125},
                "pairs_by_head": {"pick": 212}, "sets_with_pairs_by_head": {"pick": 13}, "labelled_sets": 32}       # the fields of the pod's weight_rule_rank_p.json that L2 judges (MEASURED 2026-11-25)


def _p_args(arm="P"):
    """the args the gate's L2 sees for arm P: the REAL trainer parser on P's REAL argv (so --rank-ep-gap / --rank-gate-topk are the trainer's DEFAULTS, which P's argv does not name)."""
    import train as T
    return T.build_parser().parse_args(L.scorerft_argv(arm, rank_w=0.125))


def _p_cases(td, facts_json, rl_file, pin_base):
    """[(name, reasons)] for the census fixture mutations; `pin_base` = the pin of P (path / sha256 are replaced per case)."""
    import json
    sys.modules["rank_loss"].__file__ = str(rl_file)
    a = _p_args()
    out = []

    def run(name, over=None, pin_over=None, args_over=None):
        d = json.loads(json.dumps(facts_json))
        d.update(over or {})
        path = Path(td) / f"pc_{len(out)}.json"
        path.write_text(json.dumps(d), encoding="utf-8")
        pin = dict(pin_base, path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        pin.update(pin_over or {})
        prof = {"rank_census": pin}
        aa = SimpleNamespace(**{**vars(a), **(args_over or {})})
        out.append((name, GC.judge_census_head("pick", GC.rank_census_facts(prof), aa, prof)))
    run("valid")
    run("wrong_sha", pin_over={"sha256": "ab" * 32})
    run("wrong_heads", over={"heads": ["ep", "gate"]})
    run("heads_empty", over={"heads": []})
    run("wrong_rank_w", over={"rule": {"status": "SELECT", "rank_w": 0.0625}})
    run("wrong_init", over={"init_from": "/x/other.pt"})
    run("wrong_ep_gap", over={"ep_gap": 0.05})
    run("wrong_topk", over={"gate_topk": 16})
    run("wrong_backend", over={"label_backend": "synthetic"})
    run("not_a_decision", over={"NOT_A_DECISION": True})
    run("wrong_rank_loss_sha", pin_over={"rank_loss_sha256": "cd" * 32})
    run("pairs_zero", over={"pairs_by_head": {"pick": 0}})
    run("sets_zero", over={"sets_with_pairs_by_head": {"pick": 0}})
    run("args_w_differs", args_over={"rank_w": 0.5})
    run("args_init_differs", args_over={"init_from": "/y/z.pt"})
    return out


def chk_p_census():
    """MM 2026-11-25 (pod gate FAIL D1): P / Ps register their OWN census pin, `judge_census_head("pick", rank_census_facts(profile), args, profile)` returns [] on the pod's pulled census under P's real argv, and EACH
    condition goes RED when mutated (sha, heads, rank_w, init, ep_gap, top-k, backend, NOT_A_DECISION, rank_loss sha, zero pairs / sets, the declared args).  The trainer's parser DEFAULTS for the two flags P's argv does not
    name equal the census's 0.02 / 8 (the census is judged against args.rank_ep_gap / args.rank_gate_topk)."""
    import json
    pin = L.SCORERFT_P_RANK_CENSUS
    ok(set(pin) == {"path", "sha256", "rank_loss_sha256"} and all(len(pin[k]) == 64 and pin[k] == pin[k].lower() for k in ("sha256", "rank_loss_sha256")), pin)
    ok(pin["sha256"] == "ef1c36bf4071f8a667432a2198e531c1611c9cd29f9eaa0706a6af166f8a4e1f" and pin["path"] == "/workspace/data/refe_p5c/scorerft/weight_rule_rank_p.json", "the pins are the MM's (the pod's pulled census)")
    for nm, arm in P_NAMES.items():
        ok(L.PROFILES[arm]["rank_census"] == pin and L.scorerft_profile(nm)["rank_census"] == pin, nm)
    for arm in ("T", "Ts"):
        ok(L.scorerft_profile(arm)["rank_census"] == L.SCORERFT_RANK_CENSUS, "T / Ts keep their own pin")
    ok("rank_census" not in L.scorerft_profile("C") and "rank_census" not in L.scorerft_profile("Cs"), "controls carry no census")
    ok(pin["rank_loss_sha256"] == hashlib.sha256((HERE / "rank_loss.py").read_bytes().replace(b"\r", b"")).hexdigest(), "P's pinned rank_loss sha256 is THIS tree's rank_loss.py (CR stripped): T's pin (c0e027d2) is the pre-pick file and would fail here")
    a = _p_args()
    ok(a.rank_heads == "pick" and a.rank_ep_gap == 0.02 and a.rank_gate_topk == 8 and a.init_from == L.SCORERFT_DEFAULT_INIT and a.rank_w == 0.125, (a.rank_heads, a.rank_ep_gap, a.rank_gate_topk))
    ok("--rank-ep-gap" not in L.scorerft_argv("P") and "--rank-gate-topk" not in L.scorerft_argv("P"), "P's argv names neither: the trainer defaults are what the gate compares")
    old_mod = sys.modules.get("rank_loss")
    sys.modules["rank_loss"] = types.ModuleType("rank_loss")
    td = Path(tempfile.mkdtemp(prefix="pcen_"))
    try:
        res = _p_cases(td, P_CENSUS_FIX, HERE / "rank_loss.py", dict(pin))
        ok(res[0][0] == "valid" and res[0][1] == [], res[0])
        for nm, r in res[1:]:
            ok(len(r) == 1 and r[0].startswith("L2 head 'pick' formed ZERO pairs over the whole smoke (structurally dead: the 'built, tested, unreachable' class); the registered census does not prove it: "), (nm, r))
        real = Path(os.environ.get("SCORERFT_P_CENSUS_LOCAL", pin["path"]))
        if real.is_file():
            ok(hashlib.sha256(real.read_bytes()).hexdigest() == pin["sha256"], "the real census file is the pinned bytes")
            full = json.loads(real.read_text(encoding="utf-8"))
            sys.modules["rank_loss"].__file__ = str(HERE / "rank_loss.py")
            prof = {"rank_census": dict(pin, path=str(real))}
            ok(GC.judge_census_head("pick", GC.rank_census_facts(prof), a, prof) == [], "the REAL census passes L2 for P's real argv")
            for nm, mod in (("heads", lambda d: d.update(heads=["ep"])), ("rank_w", lambda d: d["rule"].update(rank_w=0.5)), ("init", lambda d: d.update(init_from="/x")), ("pairs", lambda d: d.update(pairs_by_head={"pick": 0}))):
                d2 = json.loads(json.dumps(full))
                mod(d2)
                q = td / f"real_mut_{nm}.json"
                q.write_text(json.dumps(d2), encoding="utf-8")
                prof2 = {"rank_census": dict(pin, path=str(q), sha256=hashlib.sha256(q.read_bytes()).hexdigest())}
                ok(GC.judge_census_head("pick", GC.rank_census_facts(prof2), a, prof2) != [], f"the real census with {nm} mutated is RED")
            prof3 = {"rank_census": dict(pin, path=str(real), sha256="00" * 32)}
            ok(GC.judge_census_head("pick", GC.rank_census_facts(prof3), a, prof3) != [], "the real census under a wrong pinned sha is RED")
            print("  p_census: the REAL census file was judged", flush=True)
        else:
            print("  p_census: the real census file is absent here (not the pod): judged on the fixture only", flush=True)
        # the gate row end to end: zero smoke pick pairs (the trainer logs the LAST rank call of a step) + the census
        install_fake_rank_module()
        sys.modules["rank_loss"].__file__ = str(HERE / "rank_loss.py")
        path = td / "row.json"
        path.write_text(json.dumps(P_CENSUS_FIX), encoding="utf-8")
        prof = {"rank_census": dict(pin, path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())}
        cap = reach_cap(heads="pick")
        cap.args.init_from = L.SCORERFT_DEFAULT_INIT
        cap.args.rank_w = 0.125
        cap.metrics = [{"event": "rank"}] + [{"step": i, "traj_L1": 1.0, "rank_L": 0.4, "rank_pairs_pick": 0} for i in range(2)]
        cap.rank_calls = [audit_call(None, heads="pick")]
        cap.lab_submits = []
        r, det = GC.rank_live_rows(cap, prof, None)
        ok([x for x in r if x.startswith("L2")] == [] and det["census"]["verdict"]["pick"] == {"source": "census", "ok": True}, (r, det.get("census")))
        r2, det2 = GC.rank_live_rows(cap, {}, None)
        ok(any(x.startswith("L2 head 'pick'") and "no registered census" in x for x in r2), "without a registered census a zero-pick smoke is refused (the 2026-11-25 FAIL)")
    finally:
        shutil.rmtree(td, ignore_errors=True)
        if old_mod is None:
            sys.modules.pop("rank_loss", None)
        else:
            sys.modules["rank_loss"] = old_mod


CHECKS = {"registry": chk_registry, "existing_unchanged": chk_existing_unchanged, "argv_delta": chk_argv_delta, "p_arms": chk_p_arms, "pick_rows": chk_pick_rows, "route_real": chk_route_real, "pins_refuse": chk_pins_refuse, "levers": chk_levers, "fresh_out": chk_fresh_out,
          "live_wiring": chk_live_wiring, "h_generalised": chk_h_generalised, "reach_and_regressions": chk_reach_and_regressions, "audit": chk_audit, "dvb_rows": chk_dvb_rows,
          "live_rows": chk_live_rows_control_and_t, "census_l2": chk_census_l2, "p_census": chk_p_census, "reach_scan": chk_reach_scan, "forced_pick": chk_forced_pick, "field_reach_active": chk_field_reach_active_sample, "l5_identity_dtype": chk_l5_identity_dtype, "trainer_integration": chk_trainer_integration, "no_scipy": chk_no_scipy}

# (file, old, new, the check that must go RED): ONE emitting line of a COPY of this tree is neutralised.
MUTATIONS = [
    ("launch_gate.py", "    if prof.get(\"scorerft_rules\"):\n        out += scorerft_refusals(prof, argv)\n", "", "levers"),                                         # the rules are not hooked into profile_refusals
    ("launch_gate.py", "    if not _p5c_hex64(prof.get(\"init_from_sha256\")):\n        out.append(f\"{name}: C1's final", "    if False:\n        out.append(f\"{name}: C1's final", "pins_refuse"),   # an unfilled base pin no longer refuses
    ("launch_gate.py", "    elif ranking and not rw > 0.0:\n", "    elif False:\n", "pins_refuse"),                                                                    # a ranking arm at weight 0 is accepted
    ("launch_gate.py", "    elif not ranking and rw != 0.0:\n", "    elif False:\n", "pins_refuse"),                                                                  # a control at a weight is accepted
    ("launch_gate.py", "    for ev in tuple(prof.get(\"forbidden_env\", ())) + SCORERFT_ENV_FORBIDDEN:\n", "    for ev in ():\n", "levers"),                           # REFE_RANK_MUTATE is accepted in a launch
    ("launch_gate.py", "    if not ranking and \"rank_L\" in terms:\n", "    if False:\n", "levers"),                                                                # a control declaring rank_L is accepted (R-CTRL)
    ("launch_gate.py", "    for f in sorted(set(got) - set(want) - {\"--resume\"}):\n        out.append(f\"{name}: the argv carries {f} {' '.join(got[f][0])}, which arm {arm}'s argv does not (the arms",
     "    for f in []:\n        out.append(f\"{name}: the argv carries {f} {' '.join(got[f][0])}, which arm {arm}'s argv does not (the arms", "levers"),                  # an extra lever is accepted
    ("launch_gate.py", "    a[a.index(\"--seed\") + 1] = str(SCORERFT_ARM_SEED[arm])\n", "", "argv_delta"),                                                          # the replicates no longer differ by the seed
    ("launch_gate.py", "        rk = [\"--rank-w\", \"0.0\"]\n", "        rk = []\n", "argv_delta"),
    ("launch_gate.py", "        if arm in SCORERFT_P_ARMS:\n            rk = [\"--rank-w\", repr(float(w)), \"--rank-heads\", SCORERFT_P_RANK_HEADS]\n", "        if False:\n            rk = []\n", "p_arms"),                 # P gets T's head set and gap / top-k
    ("launch_gate.py", "    if ranking and arm in SCORERFT_P_ARMS:\n        req.update({\"--rank-heads\": SCORERFT_P_RANK_HEADS})", "    if False:\n        req.update({})", "p_arms"),                    # P's profile no longer requires --rank-heads pick
    ("launch_gate.py", "(SCORERFT_ARM_HEADS[arm], SCORERFT_RANK_EP_GAP, SCORERFT_RANK_GATE_TOPK):\n", "(SCORERFT_RANK_HEADS, SCORERFT_RANK_EP_GAP, SCORERFT_RANK_GATE_TOPK):\n", "p_arms"),   # the registered-rule check forgets the per-arm heads
    ("gate_checks.py", " | ({\"ttc\", \"c\"} if \"pick\" in _rank_heads_set(heads) else set())", "", "pick_rows"),                                                           # the reach judge forgets that `pick` trains TTC and C
    ("gate_checks.py", "                        if not (j == pick_k and pdms[i] > pdms[j]):\n", "                        if not (j == pick_k and pdms[i] >= pdms[j]):\n", "pick_rows"),   # the reference pairs ties                                                                   # C no longer NAMES --rank-w 0.0
    ("gate_checks.py", "    if profile.get(\"scorerft_rules\"):                                   # SCORER-FT arms", "    if False:                                   # SCORER-FT arms", "live_wiring"),   # the scorerft arms are not routed to the live rows
    ("gate_checks.py", "    needs_table = profile[\"traj_label_required\"] if \"traj_label_required\" in profile else profile.get(\"p5c_arm\") in (\"C1\", \"C1r\")", "    needs_table = profile.get(\"p5c_arm\") in (\"C1\", \"C1r\")", "h_generalised"),
    ("gate_checks.py", "        ok=bool(calls) and len(mult) == len(calls) and all(math.isclose(m_, w, rel_tol=1e-5) for m_ in mult))", "        ok=True)", "dvb_rows"),      # the hook row no longer judges the multiplier
    ("gate_checks.py", "        elif not math.isclose(v, ref[\"L\"], rel_tol=RANK_REL_TOL, abs_tol=1e-12):", "        elif False:", "audit"),                                       # the L4 value comparison is gone
    ("gate_checks.py", "        if x is not None and x[\"abs_sum\"] != 0.0:\n            out.append(f\"L3 rank-only backward reaches the trajectory side", "        if False:\n            out.append(f\"L3 rank-only backward reaches the trajectory side", "reach_and_regressions"),
    ("gate_checks.py", "        if rows.get(h, 0.0) != 0.0:\n            out.append(f\"L3 rank-only backward reaches score_head row", "        if False:\n            out.append(f\"L3 rank-only backward reaches score_head row", "reach_and_regressions"),
    ("gate_checks.py", "    c = census or {}\n    if not c.get(\"ran\"):", "    return []\n    c = census or {}\n    if not c.get(\"ran\"):", "census_l2"),                        # judge_census_head returns [] unconditionally (L2 amendment)
    ("gate_checks.py", "    for i in labelled_idx:                                           # the FIRST labelled sample whose declared-weight term loss is non-zero\n", "    for i in labelled_idx[:1]:                                           # the FIRST labelled sample whose declared-weight term loss is non-zero\n", "reach_scan"),   # the probe is pinned to sample 0 again (the pod gate FAIL, L3)
    ("gate_checks.py", "        if not force and float(loss.detach()) == 0.0:\n            return None\n", "        if False:\n            return None\n", "reach_scan"),                                               # a pairless sample is probed, not skipped
    ("gate_checks.py", "        forced = \"pick\" in _rank_heads_set(a.rank_heads)", "        forced = False", "forced_pick"),                                  # the forced-pair probe is switched off again (the 99ddf53d FAIL, L3)
    ("gate_checks.py", "        if j == pk:                                                  # all-tied scores", "        if False:                                                    # all-tied scores", "forced_pick"),   # j* may equal the pick: no pair
    ("launch_gate.py", "        prof[\"rank_census\"] = dict(SCORERFT_P_RANK_CENSUS if arm in SCORERFT_P_ARMS else SCORERFT_RANK_CENSUS)", "        prof[\"rank_census\"] = dict(SCORERFT_RANK_CENSUS)", "p_census"),   # P carries T's census pin again
    ("launch_gate.py", "    if ranking:\n        prof[\"rank_census\"]", "    if ranking and arm not in SCORERFT_P_ARMS:\n        prof[\"rank_census\"]", "p_census"),   # P carries no census again (the 2026-11-25 FAIL)
    ("gate_checks.py", "    return abs(float(np.asarray(np.clip(float(navsim_value), 0.0, 1.0), dtype=label_dtype)) - float(label_value))", "    return abs(float(np.clip(float(navsim_value), 0.0, 1.0)) - float(label_value))", "l5_identity_dtype"),   # compare at float64 regardless of the label dtype (the gate-2 FAIL)
    ("gate_checks.py", "    if hasattr(res, \"iloc\"):\n        row = res.iloc[0]", "    if True:\n        row = res.iloc[0]", "l5_identity_dtype"),   # always .iloc (the gate-2 AttributeError)
    ("gate_checks.py", "    out[\"active_index\"] = idx[active[0]]\n    sl = slice(idx[active[0]], idx[active[0]] + 1)", "    out[\"active_index\"] = idx[0]\n    sl = slice(idx[0], idx[0] + 1)", "field_reach_active"),   # always probe idx[0] (the Fs defect)
    ("gate_checks.py", "    return {k: ({\"loss\": v[\"loss\"], \"groups\": v[\"groups\"]} if isinstance(v, dict) else v) for k, v in fr.items()}\n",
     "    return {k: (v if k in (\"ran\", \"labelled\") else {\"loss\": v[\"loss\"], \"groups\": v[\"groups\"]}) for k, v in fr.items()}\n", "field_reach_active"),   # 1c: the 1b-era form (TypeError on `losses`)
    ("gate_checks.py", "        det[\"field_reach\"] = field_reach_detail(fr)\n",
     "        det[\"field_reach\"] = {k: (v if k in (\"ran\", \"labelled\") else {\"loss\": v[\"loss\"], \"groups\": v[\"groups\"]}) for k, v in fr.items()}\n", "field_reach_active"),   # 1c: check_live bypasses the helper (the Fs crash)
    ("gate_checks.py", "    if fr[\"dead_nonzero\"] > 0 or fr[\"pick_nc_nonzero\"] > 0 or fr[\"pick_dac_nonzero\"] > 0:\n", "    if False:\n", "route_real"),                              # the judge forgets the label-inconsistent cells
    ("gate_checks.py", "                                out[\"dead_nonzero\"] += 1\n", "                                out[\"dead_nonzero\"] += 0\n", "route_real"),                          # the facts never count a dead cell that reads non-zero
    ("gate_checks.py", "    if \"pick\" in heads:                                                                       # CTRL-ROUTE (b)", "    if False:                                                                       # CTRL-ROUTE (b)", "route_real"),   # the gate no longer runs CTRL-ROUTE
    ("gate_checks.py", "    if not fr[\"n_pick_nc1\"] > 0:\n", "    if False:\n", "route_real"),                                                                               # a vacuous probe is accepted
    ("gate_checks.py", "                        live = (d_ < 0.0) if role[m] == 1 else (d_ > 0.0)", "                        live = (d_ > 0.0) if role[m] == 1 else (d_ < 0.0)", "route_real"),         # the expectation's directions flipped
]


def run_mutations() -> None:
    for fn, old, new, check in MUTATIONS:
        td = Path(tempfile.mkdtemp(prefix="sftmut_"))
        try:
            for f in HERE.glob("*.py"):
                shutil.copyfile(f, td / f.name)
            for f in HERE.glob("p7_pod.sh"):
                shutil.copyfile(f, td / f.name)
            src = (td / fn).read_text(encoding="utf-8")
            ok(src.count(old) == 1, f"mutation target not found exactly once in {fn}: {old[:70]!r}")
            (td / fn).write_text(src.replace(old, new), encoding="utf-8", newline="\n")
            env = dict(os.environ, PYTHONIOENCODING="utf-8")
            r = subprocess.run([sys.executable, str(td / "test_scorerft_profile.py"), "--check", check], cwd=str(td), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
            ok(r.returncode != 0, f"MUTATION SURVIVED: {fn}: {old[:60]!r} -> check {check} still passed")
            print(f"  mutation RED as required: {fn} [{old.strip()[:56]}] -> {check}", flush=True)
        finally:
            shutil.rmtree(td, ignore_errors=True)


def main() -> int:
    if "--check" in sys.argv:
        name = sys.argv[sys.argv.index("--check") + 1]
        CHECKS[name]()
        print(f"CHECK_OK {name} {N}")
        return 0
    for name, fn in CHECKS.items():
        fn()
        print(f"  {name}: ok ({N})", flush=True)
    if "--fast" not in sys.argv:
        run_mutations()
    print(f"ZZSCORERFTPROF_PASS {N}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
