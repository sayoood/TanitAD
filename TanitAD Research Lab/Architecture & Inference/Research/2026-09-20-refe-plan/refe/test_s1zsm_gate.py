#!/usr/bin/env python3
"""S1ZSM launch-gate tests (CPU, no data; PREREG_S1ZSM_P6 section 2).  Prints ZZTEST_PASS.

  G1 the legal argv of the one arm: no refusal; every --twin-* lever AND --twin-speed-neutral NAMED; no lane flag, no pair flag, no EP in the heads
  G2 refusals: the DEFAULT-OFF FLAG LEFT OUT OF THE ARGV (card section 2, regression arm 4) -- refused twice (required flag + the pin); every other S1Z refusal still holds (no --twin-w, --twin-w 0, wrong heads,
     radius, top-K, seed, out dir, lane / pair flags); S1Z's own argv is refused by this profile and this profile's argv is refused by refe_s1z_p6
  G3 FINGERPRINTS: ALL 21 digests of the tip's profiles are unmoved (pinned here, MEASURED at the tip 0c19b8fd); the new profile is a deep copy sharing no mutable object, stable, same data pin / data_expect
  G4 a navtest log or token in the training data is REFUSED (argv paths and the trainer's audit counts -> G-DVB expect rows RED)
  G5 judge_g18 (extended) from the raw fields of the REAL TwinLoss: control PASS; the band removed, the balancing removed, the radius removed, the top-K removed, L_twin dropped, a head with no kept pair, an out-of-band
     pair, an unbalanced head, moved built values, S1ZSM fields without the flag -- each RED with the expected reason
  G6 G-DVB: the new dest is `built` and observed against the BUILT `twin` block (12 rows), a flipped fact is RED, a built-ON block under an argv without the flag is RED; a trainer flag WITHOUT a registry entry is
     refused by the gate itself (unknown_flags); the gate's constants equal the trainer's
  G7 the scripts: bash -n; MODE=argv of s1zsm_pod.sh equals launch_gate.s1zsm_argv (in a SANDBOX workspace, never the literal /workspace); a calibrate file that is not speed-neutral (S1Z's) is REFUSED (rc 4), a
     speed-neutral one adds exactly --twin-w; the chain: the card sha, the profile, FIVE mutants each required to fail the preflight AND read G18 ok false, terminal lines, own dirs, no P7 pause, the SAME wait functions
     as s1z_chain.sh, the pair_weight refusal at the precheck; the pause script: the cmdline checks, the EXIT-trap resume, HOLD file, the resume command identical to rep_pause_p7.sh; ship_s1zsm.py parses
"""
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch  # noqa: E402
import gate_checks as C  # noqa: E402
import gate_token as GT  # noqa: E402
import launch_gate as L  # noqa: E402
import scorer_finetune as S  # noqa: E402

PROF = L.PROFILES["refe_s1zsm_p6"]
SHA = {L.SCORER1P6_CKPT: L.SCORER1P6_CKPT_SHA256}
for _k, _f in (("--navtest-logs", L.S1ZSM_ROOT + "/navtest_ids/navtest_logs.json"), ("--navtest-tokens", L.S1ZSM_ROOT + "/navtest_ids/navtest_tokens.txt.gz")):
    SHA[_f] = L.SCORER1_NAVTEST_SHA256[_k]


def fake_sha(path):
    if path in SHA:
        return SHA[path]
    raise FileNotFoundError(path)


L.sha256_of_file = fake_sha


def refusals(argv, prof=PROF):
    return L.profile_refusals(prof, argv)


def has(rs, needle):
    return any(needle in r for r in rs)


def swap(argv, flag, *vals):
    out, i, done = [], 0, False
    while i < len(argv):
        if argv[i] == flag:
            out += [flag] + list(vals)
            done = True
            i += 1
            while i < len(argv) and not argv[i].startswith("--"):
                i += 1
            continue
        out.append(argv[i])
        i += 1
    return out if done else out + [flag] + list(vals)


def drop(argv, flag):
    out, i = [], 0
    while i < len(argv):
        if argv[i] == flag:
            i += 1
            while i < len(argv) and not argv[i].startswith("--"):
                i += 1
            continue
        out.append(argv[i])
        i += 1
    return out


ARM = L.s1zsm_argv("S1ZSM", 0.37)

# ---- G1 the legal argv
assert refusals(ARM) == [], refusals(ARM)
for f, v in (("--twin-heads", "NC,DAC,TTC"), ("--twin-ade", "1.0"), ("--twin-topk", "8"), ("--twin-w", "0.37"), ("--seed", "0"), ("--max-updates", "400")):
    assert ARM[ARM.index(f) + 1] == v, (f, ARM[ARM.index(f) + 1])
assert "--twin-speed-neutral" in ARM and ARM[ARM.index("--twin-speed-neutral") + 1].startswith("--"), "the store_true flag takes no value"
assert "--twin-w" not in L.s1zsm_argv("S1ZSM") and "--twin-w" in ARM
assert not any("lane" in t for t in ARM) and not any(t.startswith("--pair") for t in ARM), "no lane flag, no pair flag"
assert ARM[ARM.index("--out") + 1] == L.S1ZSM_ROOT + "/s1zsm" and len({L.S1ZSM_ROOT, L.S1Z_ROOT, L.SCORER1P6_ROOT}) == 3 and L.S1ZSM_PKG not in (L.S1Z_PKG, L.SCORER1P6_PKG)
assert L.S1ZSM_CARD_SHA256 == hashlib.sha256(open(os.path.join(HERE, "PREREG_S1ZSM_P6.md"), "rb").read()).hexdigest() if os.path.exists(os.path.join(HERE, "PREREG_S1ZSM_P6.md")) else True
print("G1 ok")

# ---- G2 refusals
rs = refusals(drop(ARM, "--twin-speed-neutral"))                    # the default-off flag left out of the argv
assert has(rs, "requires --twin-speed-neutral") and has(rs, "PINS --twin-speed-neutral"), rs
assert has(refusals(swap(ARM, "--seed", "1")), "--seed")
assert has(refusals(ARM + ["--lane-head"]), "--lane-head") and has(refusals(ARM + ["--teacher-lane-train", "/x"]), "teacher-lane")
assert refusals(ARM + ["--pair-heads", "NC"]) and refusals(ARM + ["--pair-w", "0.3"]) and refusals(ARM + ["--pair-mean", "head"])
assert has(refusals(ARM + ["--pair-calibrate"]), "--pair-calibrate")
assert has(refusals(drop(ARM, "--twin-w")), "--twin-w")
assert has(refusals(swap(ARM, "--twin-w", "0")), "--twin-w") and has(refusals(swap(ARM, "--twin-w", "nan")), "--twin-w") and has(refusals(swap(ARM, "--twin-w", "-0.1")), "--twin-w")
for f in ("--twin-heads", "--twin-ade", "--twin-topk"):
    assert has(refusals(drop(ARM, f)), f), (f, refusals(drop(ARM, f)))
assert refusals(swap(ARM, "--twin-heads", "NC,DAC,TTC,EP")) and refusals(swap(ARM, "--twin-heads", "NC,DAC"))
assert refusals(swap(ARM, "--twin-ade", "2.0")) and refusals(swap(ARM, "--twin-ade", "0")) and refusals(swap(ARM, "--twin-topk", "16")) and refusals(swap(ARM, "--twin-topk", "0"))
assert refusals(ARM + ["--twin-speed-neutral"]), "the flag twice is refused (the pin carries it once)"
SHA[L.SCORER1P6_CKPT] = "0" * 64
assert has(refusals(ARM), "registered P6 checkpoint")
SHA[L.SCORER1P6_CKPT] = L.SCORER1P6_CKPT_SHA256
assert refusals(swap(ARM, "--out", L.SCORER1P6_ROOT + "/c0")) and refusals(swap(ARM, "--out", L.S1Z_ROOT + "/s1z")) and refusals(swap(ARM, "--out", L.S1ZSM_ROOT + "/s1z"))
# S1Z's argv under this profile, and this argv under S1Z's profile
for _f in (L.S1Z_ROOT + "/navtest_ids/navtest_logs.json", L.S1Z_ROOT + "/navtest_ids/navtest_tokens.txt.gz"):
    SHA[_f] = L.SCORER1_NAVTEST_SHA256["--navtest-logs" if _f.endswith(".json") else "--navtest-tokens"]
assert refusals(L.s1z_argv("S1Z", 0.37)), "S1Z's own argv is not this arm"
assert L.profile_refusals(L.PROFILES["refe_s1z_p6"], L.s1z_argv("S1Z", 0.37)) == []
assert L.profile_refusals(L.PROFILES["refe_s1z_p6"], ARM), "this arm's argv is not S1Z's (a flag S1Z does not pin, another root)"
assert refusals(ARM) == []
print("G2 ok")

# ---- G3 fingerprints


def fp(n):
    return hashlib.sha256(repr(sorted(L.PROFILES[n].items())).encode()).hexdigest()


TIP_FP = {"refe_p2": "627341286aea3506", "refe_p6": "80fa4d7ec757ad46", "refe_field": "5ddc49053a99fa78", "refe_field_ctrl": "0b6281b1bfa996ed", "refe_rehearsal": "821cd1ef3bf55d4e",
          "refe_rls1": "bce7e25ee8718c08", "refe_aux_s1": "f95f48727b62ccc6", "refe_p7": "94cca167ce121a94", "refe_p2rep": "44ca107f53aceee8", "refe_sft4r": "8972e3c348206b94",
          "refe_scorer1": "88d02fb7b34d4fd8", "refe_scorer1p6": "e74d64617f514d35", "refe_s1z_p6": "5c3bff457aa3bf1e", "refe_rep_p6": "dd348f44c9638f1a", "refe_rep_s1z_p6": "2356afd4efeeb52a",
          "refe_rls1b": "5d6bfbc10152ae96", "refe_rls1c": "26214cf954099d51", "refe_rls1d": "fcc2585f2a5d9b94", "refe_rls1e": "54d96289b19c3cd6", "refe_hnscreen": "d30126bbb5a32957",
          "refe_aux_s1c": "dcdf2dbd2a674f1c"}          # MEASURED at the tip 0c19b8fd (sha256 of repr(sorted(profile.items())), first 16 hex): the 21 profiles that exist before S1ZSM
moved = [n for n, want in TIP_FP.items() if not fp(n).startswith(want)]
assert not moved, f"existing profile fingerprints moved: {moved}"
assert set(L.PROFILES) - set(TIP_FP) == {"refe_s1zsm_p6"}, set(L.PROFILES) - set(TIP_FP)
assert fp("refe_s1zsm_p6") == fp("refe_s1zsm_p6") and "function" not in repr(sorted(PROF.items())) and "0x" not in repr(sorted(PROF.items())), "the digest of the new profile must be stable"
base = L.PROFILES["refe_s1z_p6"]
assert PROF is not base and PROF["recipe"] is not base["recipe"] and PROF["required_flags"] is not base["required_flags"] and PROF["arm_seed"] is not base["arm_seed"] and PROF["twin_spec"] is not base["twin_spec"]
assert PROF["data_expect"] == base["data_expect"] and PROF["data_pin"] == L.SCORER1P6_DATA_PIN and PROF["ckpt_sha256"] == base["ckpt_sha256"] and PROF["navtest_sha256"] == base["navtest_sha256"]
assert PROF["recipe"] == base["recipe"] and PROF["data_expect"]["nav_mode"] == "command" and PROF["twin_spec"] == base["twin_spec"] and PROF["sn_spec"] == {"dv": 0.15, "seed": 20261107}
assert "--twin-speed-neutral" in PROF["required_flags"] and "--twin-speed-neutral" not in base["required_flags"] and "sn_spec" not in base
assert base["arm_seed"] == L.S1Z_ARM_SEED and "S1ZSM" not in L.S1Z_ARM_SEED and not any("s1zsm" in x["script"] for x in base["suite"]), "the S1Z profile and suite are untouched"
assert [x["script"] for x in PROF["suite"]][:len(base["suite"])] == [x["script"] for x in base["suite"]] and len(PROF["suite"]) == len(base["suite"]) + 4
print("G3 ok (21 existing digests unmoved; refe_s1zsm_p6 = %s...)" % fp("refe_s1zsm_p6")[:16])

# ---- G4 navtest in the training data
assert has(refusals(swap(ARM, "--onpolicy", "/workspace/data/navtest/sets")), "NAVTEST IS NEVER TRAINING DATA")
assert has(refusals(swap(ARM, "--pdm-labels-train", "/workspace/data/navtest_pdm")), "NAVTEST IS NEVER TRAINING DATA")
assert refusals(drop(ARM, "--navtest-logs")) and refusals(drop(ARM, "--navtest-tokens"))
ns, err = C.parse_scorer_argv(S.build_parser(), ARM)
assert not err and ns.twin_speed_neutral is True, err
data_ok = {"nav_mode": "command", "heldout_logs": 24, "train_rows_in_heldout_logs": 0, "navtest": {k.split(".", 1)[1]: v for k, v in L.SCORER1_NAVTEST_EXPECT.items()}}
rows = [r for r in C.scorer_expect_rows(PROF, ns, data_ok) if r["flag"].startswith("data.navtest")]
assert len(rows) == 6 and all(r["ok"] for r in rows), rows
for key in ("train_rows_in_navtest_logs", "train_rows_in_navtest_tokens", "bank_sets_in_navtest_logs", "bank_sets_in_navtest_tokens"):
    bad = json.loads(json.dumps(data_ok))
    bad["navtest"][key] = 3
    rr = [r for r in C.scorer_expect_rows(PROF, ns, bad) if r["flag"] == "data.navtest." + key]
    assert len(rr) == 1 and rr[0]["ok"] is False, (key, rr)
print("G4 ok")

# ---- G5 judge_g18 (extended) from the raw fields of the REAL TwinLoss
HEADS = ["NC", "DAC", "TTC"]
T_, SPD = 20, [10.0, 11.0, 10.5, 14.0, 9.0]


def one_set():
    """the hand set of test_s1zsm_loss: v0 = 10, NC hard negatives p2 (in band both), p3 (out of band), p4 (in band of p0)"""
    xy = torch.tensor([[[v * 0.2 * (t + 1), 0.1 * k] for t in range(T_)] for k, v in enumerate(SPD)], dtype=torch.float32)
    lab = lambda nc, dac, ttc: [nc, dac, 0.5, ttc, 1.0, 1.0]                                    # noqa: E731
    y = torch.tensor([lab(1, 1, 1), lab(1, 1, 1), lab(0, 1, 1), lab(0, 1, 1), lab(0, 1, 1)])
    return torch.cat([xy, torch.zeros(len(SPD), T_, 1)], -1)[None], y[None]


def real_g18(mutate=None, drop_keys=()):
    for e in S.TWIN_MUTATE_ENVS + S.TWIN_SN_MUTATE_ENVS:
        os.environ.pop(e, None)
    if mutate in ("noradius", "notopk"):
        os.environ["REFE_TWIN_MUTATE_" + mutate.upper()] = "1"
    if mutate in ("noband", "nosnbalance"):
        os.environ["REFE_TWIN_MUTATE_" + {"noband": "NOBAND", "nosnbalance": "NOSNBALANCE"}[mutate]] = "1"
    try:
        tl = S.TwinLoss(["NC"], 1000.0, 8, speed_neutral=True)
        cxy, y = one_set()
        counts = {}
        for _ in range(8):
            tl.term(torch.randn(1, len(SPD), 6), y, cxy, torch.ones(1, len(SPD)), torch.tensor([10.0]))
            for h_, v_ in tl.last.items():
                c_ = counts.setdefault(h_, {k: 0 for k in v_})
                for k, x in v_.items():
                    c_[k] += x
        # the DAC / TTC heads have no hard negative in this tiny set: give them the same real counts (the gate reads counts, the loss class produced them)
        one = counts["NC"]
        per_head = {h: dict(one) for h in HEADS}
        share = {h: {**dict(one), "share_with_pair": 1.0} for h in HEADS}
        bank = {h: {"sets": 100, "sets_with_clean": 90, "sets_with_pair": 70, "share_with_pair": 0.7} for h in HEADS}
        drop_ = mutate == "droploss"
        g = {"loss_grad_delta_w1_minus_w0": {m: (0.0 if drop_ else 1.5e-3) for m in C.SCORER_GROUPS}, "control_w0_term_alone_grad_abs": {m: 0.0 for m in C.SCORER_GROUPS},
             "controls_exactly_zero": True, "micro_batch_pairs": per_head, "share_micro": share, "micro_batches": 16, "bank_share": bank, "bank_stride": 5, "heads_built": list(HEADS),
             "ade_built": 1.0 if mutate != "noradius" else 1e6, "topk_built": 8 if mutate != "notopk" else 10 ** 6, "radius_filter_built": mutate != "noradius", "mining_built": mutate != "notopk",
             "mutate_noradius": mutate == "noradius", "mutate_notopk": mutate == "notopk", "mutate_drop_loss": drop_,
             "loss_grad_noise_w0_vs_w0": {m: (0.0 if m == "score_head" else 1e-6) for m in C.SCORER_GROUPS}, "noise_factor": S.TWIN_NOISE_FACTOR,
             "probe_batch": {"found": True, "index": 3, "searched": 4, "search_limit": S.TWIN_PROBE_SEARCH_MAX, "pairs_per_head": {h: {"pairs": 2} for h in HEADS}, "reason": None},
             "speed_neutral": True, "sn_dv_built": tl.sn_dv_built, "sn_band_filter_built": bool(tl.sn_band_filter), "sn_balance_built": bool(tl.sn_balance), "sn_seed_built": tl.sn_seed,
             "sn_probe_batch_ok": mutate not in ("noband", "nosnbalance"), "sn_live_ok": mutate not in ("noband", "nosnbalance"), "sn_ok": mutate not in ("noband", "nosnbalance"),
             "mutate_noband": mutate == "noband", "mutate_nosnbalance": mutate == "nosnbalance", "ok": not mutate}
        for k in drop_keys:
            g.pop(k)
        return g
    finally:
        for e in S.TWIN_MUTATE_ENVS + S.TWIN_SN_MUTATE_ENVS:
            os.environ.pop(e, None)


def judge(g, nsx=None):
    return C.judge_g18(g, nsx if nsx is not None else ns)


g0 = real_g18()
assert g0["micro_batch_pairs"]["NC"]["out_of_band"] == 0 and g0["micro_batch_pairs"]["NC"]["unbalanced_sets"] == 0 and g0["micro_batch_pairs"]["NC"]["kept_F"] == g0["micro_batch_pairs"]["NC"]["kept_S"] > 0
assert judge(g0) == [], judge(g0)
nb, nsb = judge(real_g18("noband")), judge(real_g18("nosnbalance"))
assert any("OUTSIDE the speed band" in r for r in nb) and any("sn_dv_built" in r for r in nb) and any("regression arm (REFE_TWIN_MUTATE_NOBAND" in r for r in nb), nb
assert any("not F/S balanced" in r for r in nsb) and any("balancing is not built" in r for r in nsb) and any("regression arm (REFE_TWIN_MUTATE_NOBAND" in r for r in nsb), nsb
nr, nk, dl = judge(real_g18("noradius")), judge(real_g18("notopk")), judge(real_g18("droploss"))
assert any("radius filter is missing" in r for r in nr) and any("top-K mining is missing" in r for r in nk) and any("not live IN the loss" in r for r in dl)
g = real_g18()
g["micro_batch_pairs"]["DAC"].update(pairs=0, kept_F=0, kept_S=0)
assert any("head DAC has no kept pair in the probe" in r for r in judge(g)) and any("LAUNCH BLOCKER" in r and "DAC" in r for r in judge(g)), judge(g)
g = real_g18()
g["share_micro"]["TTC"].update(pairs=0, kept_F=0, kept_S=0, share_with_pair=0.0)
assert any("head TTC has share 0" in r for r in judge(g))
g = real_g18()
g["share_micro"]["NC"]["out_of_band"] = 1
assert any("OUTSIDE the speed band in share_micro" in r for r in judge(g))
g = real_g18()
g["micro_batch_pairs"]["NC"]["kept_S"] += 1
assert any("not F/S balanced in micro_batch_pairs" in r for r in judge(g))
g = real_g18(drop_keys=("sn_ok",))
assert any("sn_ok" in r for r in judge(g))
g = real_g18()
g["sn_seed_built"] = 1
assert any("sn_seed_built" in r for r in judge(g))
g = real_g18()
g["sn_dv_built"] = 0.2
assert any("sn_dv_built" in r for r in judge(g))
g = real_g18()
g["speed_neutral"] = None
assert any("the flag did not reach the built loss" in r for r in judge(g))
g = real_g18()
del g["share_micro"]["NC"]["band_pairs"]
assert any("carries no speed-neutral counts" in r for r in judge(g))
# S1Z (the flag NOT in the argv) is judged exactly as before; S1ZSM fields in a record whose argv lacks the flag are RED
ns_s1z, _e = C.parse_scorer_argv(S.build_parser(), drop(ARM, "--twin-speed-neutral"))
assert ns_s1z.twin_speed_neutral is False
assert any("S1ZSM fields but the argv does not name" in r for r in judge(real_g18(), ns_s1z))
s1z_rec = {k: v for k, v in real_g18().items() if not (k.startswith("sn_") or k in ("speed_neutral", "mutate_noband", "mutate_nosnbalance"))}
for h in HEADS:
    for w in ("micro_batch_pairs", "share_micro"):
        s1z_rec[w][h] = {k: v for k, v in s1z_rec[w][h].items() if k not in S.TWIN_SN_KEYS}
assert judge(s1z_rec, ns_s1z) == [], judge(s1z_rec, ns_s1z)
g = real_g18()
assert C.G18_NOISE_FACTOR == S.TWIN_NOISE_FACTOR == 10.0
print("G5 ok")

# ---- G6 G-DVB and the obliged gates
assert C.DVB_KINDS_SCORER["twin_speed_neutral"] == "built" and "twin_speed_neutral" not in C.SCORER_TWIN_DESTS
assert C.G18_SN_DV == S.TWIN_SN_DV == L.S1ZSM_SN_SPEC["dv"] == 0.15 and C.G18_SN_SEED == S.TWIN_SN_SEED == L.S1ZSM_SN_SPEC["seed"] == 20261107
data = {"twin": S.TwinLoss(HEADS, 1.0, 8, speed_neutral=True).built_facts(0.37)}
rows = [r for r in C.scorer_dvb_rows(ns, data, PROF) if r["dest"].startswith("twin")]
assert len(rows) == 12 and all(r["ok"] for r in rows), [r for r in rows if not r["ok"]]
for key, val in (("ade", 1e6), ("topk", 16), ("heads", ["NC", "DAC"]), ("w", 0.5), ("radius_filter", False), ("mining", False), ("speed_neutral", False), ("sn_dv", 1e6), ("sn_dv", 0.2),
                 ("sn_band_filter", False), ("sn_balance", False), ("sn_seed", 7)):
    bad = json.loads(json.dumps(data))
    bad["twin"][key] = val
    assert any(not r["ok"] for r in C.scorer_dvb_rows(ns, bad, PROF) if r["dest"].startswith("twin")), key
bad = json.loads(json.dumps(data))
del bad["twin"]["speed_neutral"]
assert any(not r["ok"] and r["dest"] == "twin_speed_neutral" for r in C.scorer_dvb_rows(ns, bad, PROF)), "the flag in the argv, no speed_neutral in the built block: RED"
assert any(not r["ok"] for r in C.scorer_dvb_rows(ns, {}, PROF) if r["dest"].startswith("twin")), "no twin block in the data event: RED"
rows_x = C.scorer_dvb_rows(ns_s1z, data, PROF)                                       # built ON, argv without the flag
assert any(not r["ok"] and r["dest"] == "twin_speed_neutral" for r in rows_x), "built speed-neutral under an argv that leaves the flag out: RED"
data_s1z = {"twin": S.TwinLoss(HEADS, 1.0, 8).built_facts(0.37)}
assert len([r for r in C.scorer_dvb_rows(ns_s1z, data_s1z, PROF) if r["dest"].startswith("twin")]) == 7, "an S1Z argv keeps its 7 twin rows"
req = C.scorer_required_gates(ns)
assert "G18" in req and "G16" not in req, req
# "a new trainer flag without a G-DVB entry is refused by the gate itself": with the registry entry removed the gate reports the flag as unknown
saved = dict(C.DVB_KINDS_SCORER)
try:
    del C.DVB_KINDS_SCORER["twin_speed_neutral"]
    ev = C.check_dvb_scorer(ARM, None, PROF)
    assert "twin_speed_neutral" in ev["details"]["unknown_flags"] and ev["status"] == "FAIL" and any("twin_speed_neutral" in r for r in ev["reasons"]), ev["reasons"]
finally:
    C.DVB_KINDS_SCORER.clear()
    C.DVB_KINDS_SCORER.update(saved)
ev = C.check_dvb_scorer(ARM, None, PROF)
assert "twin_speed_neutral" not in ev["details"]["unknown_flags"] and ev["details"]["unknown_flags"] == [] and "twin_speed_neutral" not in ev["details"]["stale_registry"]
print("G6 ok")

# ---- G7 the scripts
bash = None
for cand in ("C:/Program Files/Git/bin/bash.exe", "C:/Program Files/Git/usr/bin/bash.exe"):
    if os.path.exists(cand):
        bash = cand
        break
bash = bash or shutil.which("bash")
pod, chain, chain_old = os.path.join(HERE, "s1zsm_pod.sh"), os.path.join(HERE, "s1zsm_chain.sh"), os.path.join(HERE, "s1z_chain.sh")
pause, pause_old, ship = os.path.join(HERE, "pause_p7_s1zsm.sh"), os.path.join(HERE, "rep_pause_p7.sh"), os.path.join(HERE, "ship_s1zsm.py")
for f in (pod, chain, chain_old, pause, pause_old):
    assert os.path.exists(f), f
HAVE_SHIP = os.path.exists(ship)          # ship_s1zsm.py is a Windows-side tool kept OUTSIDE refe/ (as ship_s1z.py was): checked when it sits beside the tests
ctext = open(chain, "rb").read().decode("utf-8")
for f in (pod, chain, pause) + ((ship,) if HAVE_SHIP else ()):
    assert b"\r" not in open(f, "rb").read(), ("LF only", f)
joined = "\n".join(ln for ln in ctext.split("\n") if not ln.lstrip().startswith("#"))
assert "STAGE0_READING" not in joined and "P6_SPEED_READING" not in joined and "lane" not in joined.lower(), "no human gate, no lane"
assert "refe-plan-p6" not in joined and "refe-plan-p7" not in joined and "refe-plan-scorer1p6" not in joined and "refe-plan-s1z/" not in joined and "$DATA/refe_scorer1p6" not in joined and "$DATA/refe_s1z/" not in joined, "own dirs only"
assert "--profile refe_s1zsm_p6" in joined and "refe-plan-s1zsm" in joined and "$DATA/refe_s1zsm" in joined and L.S1ZSM_CARD_SHA256 in ctext and L.SCORER1P6_CKPT_SHA256 in ctext
assert "--pair-calibrate" not in joined, "--pair-calibrate is in no launch argv"
assert re.search(r"for ARMN in S1ZSM; do", joined) and "ZZS1ZSM_CHAIN_DONE" in joined and "ZZS1ZSM_CHAIN_FAIL" in joined and "REFUSED" in joined and '--prefix "s1zsm"' in joined
assert "ZZS1Z_" not in ctext and "S1Z_HF" not in ctext, "no S1Z marker or env name left in the S1ZSM chain"
assert re.search(r"for MUT in noband nosnbalance droploss noradius notopk; do", joined) and "mutate_$MUT" in joined and "gate_is $OUT/mutate_$MUT.log G18 ok false" in joined
assert 'grep -q "^ZZSFT_PREFLIGHT_FAIL" $OUT/mutate_$MUT.log' in joined
assert not re.search(r"\bkill -|SIGSTOP|SIGCONT|pkill|taskkill|ZZP7|P7_PAUSE|pause_p7|resume_p7|lane_pause|lane_restart|refe_p7/", joined), "the chain neither pauses nor resumes P7 (it only READS the P7 chain lock)"
assert "S1ZSM_STOP_AFTER" in ctext and "stop_after inputs" in joined and "S1ZSM_HF_TOKEN_FILE" in joined
# the pair_weight refusal: at the precheck (before the inputs) AND after the measurement
i_pre, i_inp = joined.index("speed_neutral"), joined.index("STAGE=inputs")
assert i_pre < i_inp and 'refuse "$PWF was not measured with --twin-speed-neutral' in joined and 'd["speed_neutral"] is True' in joined and "ZZPAIR_CALIBRATED" in joined


def fn_body(text, name):
    m1 = re.search(r"^%s\(\) \{.*$" % re.escape(name), text, re.M)                           # the opening line (. does not cross a newline here)
    if m1 and "; }" in m1.group(0):                                                          # a one-line function (its closing brace sits on the same line)
        return m1.group(0)
    m = re.search(r"^%s\(\) \{.*?^\}" % re.escape(name), text, re.S | re.M)
    assert m, name
    return m.group(0)


old_text = open(chain_old, "rb").read().decode("utf-8")
norm = lambda t: t.replace("S1Z_", "S1ZSM_").replace("ZZS1Z_", "ZZS1ZSM_")          # noqa: E731
for name in ("gpu_busy", "lock_held", "wait_clear"):
    assert norm(fn_body(old_text, name)) == fn_body(ctext, name), name                       # the wait for no chain lock and an idle GPU is the S1Z function, byte for byte (env names renamed)
assert fn_body(old_text, "others_hold").replace("refe_scorer1p6; do", "refe_scorer1p6 refe_s1z refe_rep; do") == fn_body(ctext, "others_hold"), "others_hold differs by exactly the two added locks"
pt = open(pause, "rb").read().decode("utf-8")
po = open(pause_old, "rb").read().decode("utf-8")
for needle in ("p7_pod.sh' || early", "train.py' || early", "trap 'resume_p7' EXIT", "P7_RESUMED", "P7_HELD", "PAUSE_STOPPED_EARLY", "HOLD_P7", "refe_s1zsm/pause.log" if False else "OUT=/workspace/data/refe_s1zsm; L=$OUT/pause.log",
               "s1zsm_chain.sh", "S1ZSM_HF_TOKEN_FILE", "timeout 21600"):
    assert needle in pt, needle
resume_line = [ln for ln in po.split("\n") if "p7_pod.sh >>" in ln]
assert len(resume_line) == 1 and resume_line[0] in pt.split("\n"), "the P7 resume command is rep_pause_p7.sh's, byte for byte"
for needle in ("kill -TERM $CHAIN; kill -TERM $TR", "kill -0 $TR 2>/dev/null && { say \"trainer still alive after 60 s, SIGKILL\"; kill -9 $TR; sleep 2; }", "desc() {", "nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l"):
    assert needle in pt and needle in po, needle
assert "rep_chain" not in pt and "refe_rep" not in pt.replace("refe_s1zsm", "")
src_ship = open(ship, encoding="utf-8").read() if HAVE_SHIP else None
if HAVE_SHIP:
    ast.parse(src_ship)
assert not HAVE_SHIP or 'staged["refe/s1zsm_chain.sh"]' in src_ship and "refe_s1zsm_p6" in src_ship and "s1zsm_pkg.tar" in src_ship and L.S1ZSM_CARD_SHA256 in src_ship and "s1z_pkg.tar" not in src_ship
if bash:
    for sh in (pod, chain, pause):
        r = subprocess.run([bash, "-n", sh], capture_output=True, encoding="utf-8")
        assert r.returncode == 0, (sh, r.stderr[:300])

    def gp(toks):
        return sorted((f, tuple(v)) for f, v in GT.flag_pairs(toks))

    def pod_argv(ws, extra_env=None, mode="argv"):
        env = dict(os.environ, ARM="S1ZSM", MODE=mode, S1ZSM_WS=ws)
        env.update(extra_env or {})
        return subprocess.run([bash, pod], env=env, capture_output=True, encoding="utf-8")

    with tempfile.TemporaryDirectory(dir=os.environ.get("SCORER1_TMP") or None) as td:      # a SANDBOX workspace: the literal /workspace holds real calibrate files on the pod
        ws = td.replace(chr(92), "/") + "/ws"
        os.makedirs(ws)
        r = pod_argv(ws)
        assert r.returncode == 0, r.stderr[:300]
        out = [t.replace(ws, "/workspace") for t in r.stdout.strip().split()]
        assert gp(out) == gp(L.s1zsm_argv("S1ZSM", twin_w=None)), set(gp(out)) ^ set(gp(L.s1zsm_argv("S1ZSM", twin_w=None)))
        assert "--pair-calibrate" not in out and "--twin-speed-neutral" in out and "--twin-w" not in out
        cal = os.path.join(ws, "data", "refe_s1zsm", "calibrate_s1zsm")
        os.makedirs(cal)
        # S1Z's number (measured for the loss WITHOUT the speed-neutral step) is refused
        json.dump({"twin_w": 0.026321, "lever": "twin"}, open(os.path.join(cal, "pair_weight.json"), "w"))
        r = pod_argv(ws)
        assert r.returncode == 4 and "REFUSED" in r.stderr and "--twin-speed-neutral" in r.stderr and r.stdout.strip() == "", (r.returncode, r.stderr[:300])
        for bad in ({"speed_neutral": False}, {"sn_dv": 0.2}, {"sn_balance": False}, {"sn_seed": 1}, {"lever": "pair"}):
            d_ = {"twin_w": 0.37, "lever": "twin", "speed_neutral": True, "sn_dv": 0.15, "sn_balance": True, "sn_seed": 20261107}
            d_.update(bad)
            json.dump(d_, open(os.path.join(cal, "pair_weight.json"), "w"))
            assert pod_argv(ws).returncode == 4, bad
        json.dump({"twin_w": 0.37, "lever": "twin", "speed_neutral": True, "sn_dv": 0.15, "sn_balance": True, "sn_seed": 20261107}, open(os.path.join(cal, "pair_weight.json"), "w"))
        r = pod_argv(ws)
        out = [t.replace(ws, "/workspace") for t in r.stdout.strip().split()]
        assert r.returncode == 0 and "--twin-w" in out and out[out.index("--twin-w") + 1] == "0.37", (r.returncode, r.stderr[:200], out[-12:])
        assert gp(out) == gp(L.s1zsm_argv("S1ZSM", twin_w=0.37)) and L.profile_refusals(PROF, out) == [], "the pod script's argv with lambda_z is a legal launch argv"
        # MODE=calibrate does not refuse a stale file (it re-measures): it gets as far as the missing teacher_env.sh (rc != 4, no REFUSED)
        json.dump({"twin_w": 0.026321}, open(os.path.join(cal, "pair_weight.json"), "w"))
        r = pod_argv(ws, mode="calibrate")
        assert r.returncode != 4 and "REFUSED" not in r.stderr, (r.returncode, r.stderr[:200])
        r = subprocess.run([bash, pod], env=dict(os.environ, ARM="S1Z", MODE="argv", S1ZSM_WS=ws), capture_output=True, encoding="utf-8")
        assert r.returncode == 2, "an unknown ARM is refused"
    print("G7 ok")
else:
    print("G7 SKIPPED (no bash)")
print("ZZTEST_PASS")
