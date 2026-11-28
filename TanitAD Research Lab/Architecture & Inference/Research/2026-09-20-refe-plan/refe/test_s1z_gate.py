#!/usr/bin/env python3
"""S1Z launch-gate tests (CPU, no data; PREREG_S1Z_P6 section 2).  Prints ZZTEST_PASS.

  G1 the legal argv of the one arm: no refusal; every --twin-* lever NAMED; no lane flag, no pair flag, no EP in the heads
  G2 refusals: a wrong seed / checkpoint sha / out dir; a lane flag; a --pair-* flag or --pair-calibrate; and the DEFAULT-DISABLED TERM MISSING FROM THE ARGV (no --twin-w, --twin-w 0, no --twin-heads,
     no --twin-ade, no --twin-topk) -- each refused; wrong heads (EP), radius or top-K refused
  G3 FINGERPRINTS: refe_scorer1p6's own digest is the tip's (pinned here), the 17 older digests are pinned by test_scorer1p6_gate.py (run with this one); the new profile is a distinct dict sharing no
     mutable object, its digest is stable (no function in it), and it reads the SAME data pin and data_expect as refe_scorer1p6
  G4 a navtest log or token in the training data is REFUSED: a data path naming navtest (argv), and the trainer's own audit counts (data event) -> the G-DVB expect rows go RED
  G5 judge_g18 from the raw fields of the REAL TwinLoss: control PASS; the radius filter removed, the top-K mining removed, L_twin dropped from the loss, a head with share 0, a non-zero control, a moved
     built value -- each RED with the expected reason
  G6 G-DVB: the four --twin-* dests are `built`, observed against the BUILT `twin` block, a flipped value is RED; G18 is obliged iff --twin-w > 0 (and G16 is not)
  G7 the pod scripts: bash -n; MODE=argv of s1z_pod.sh equals launch_gate.s1z_argv (a calibrate file adds exactly --twin-w); the chain has no human gate / no lane / its own dirs, waits for no chain lock and an
     idle GPU with the SAME functions as scorer1p6_chain.sh (one extra lock name), does not pause or resume P7, banks under the prefix s1z, ends with ZZS1Z_CHAIN_DONE
"""
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

PROF = L.PROFILES["refe_s1z_p6"]
SHA = {L.SCORER1P6_CKPT: L.SCORER1P6_CKPT_SHA256}
for _k, _f in (("--navtest-logs", L.S1Z_ROOT + "/navtest_ids/navtest_logs.json"), ("--navtest-tokens", L.S1Z_ROOT + "/navtest_ids/navtest_tokens.txt.gz")):
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


ARM = L.s1z_argv("S1Z", 0.37)

# ---- G1 the legal argv
assert refusals(ARM) == [], refusals(ARM)
for f, v in (("--twin-heads", "NC,DAC,TTC"), ("--twin-ade", "1.0"), ("--twin-topk", "8"), ("--twin-w", "0.37"), ("--seed", "0"), ("--max-updates", "400")):
    assert ARM[ARM.index(f) + 1] == v, (f, ARM[ARM.index(f) + 1])
assert "--twin-w" not in L.s1z_argv("S1Z") and "--twin-w" in ARM
assert not any("lane" in t for t in ARM) and not any(t.startswith("--pair") for t in ARM), "no lane flag, no pair flag"
assert ARM[ARM.index("--out") + 1] == L.S1Z_ROOT + "/s1z" and L.S1Z_ROOT != L.SCORER1P6_ROOT and L.S1Z_PKG != L.SCORER1P6_PKG
print("G1 ok")

# ---- G2 refusals
assert has(refusals(swap(ARM, "--seed", "1")), "--seed")
assert has(refusals(ARM + ["--lane-head"]), "--lane-head") and has(refusals(ARM + ["--teacher-lane-train", "/x"]), "teacher-lane")
assert refusals(ARM + ["--pair-heads", "NC"]) and refusals(ARM + ["--pair-w", "0.3"]) and refusals(ARM + ["--pair-mean", "head"])
assert has(refusals(ARM + ["--pair-calibrate"]), "--pair-calibrate")
# the default-disabled term missing from the argv (the card's fourth regression arm)
rs = refusals(drop(ARM, "--twin-w"))
assert has(rs, "--twin-w"), rs
assert has(refusals(swap(ARM, "--twin-w", "0")), "> 0") or has(refusals(swap(ARM, "--twin-w", "0")), "--twin-w"), refusals(swap(ARM, "--twin-w", "0"))
assert has(refusals(swap(ARM, "--twin-w", "nan")), "--twin-w") and has(refusals(swap(ARM, "--twin-w", "-0.1")), "--twin-w")
for f in ("--twin-heads", "--twin-ade", "--twin-topk"):
    assert has(refusals(drop(ARM, f)), f), (f, refusals(drop(ARM, f)))
assert refusals(swap(ARM, "--twin-heads", "NC,DAC,TTC,EP")) and refusals(swap(ARM, "--twin-heads", "NC,DAC"))
assert refusals(swap(ARM, "--twin-ade", "2.0")) and refusals(swap(ARM, "--twin-ade", "0")) and refusals(swap(ARM, "--twin-topk", "16")) and refusals(swap(ARM, "--twin-topk", "0"))
SHA[L.SCORER1P6_CKPT] = "0" * 64
assert has(refusals(ARM), "registered P6 checkpoint")
SHA[L.SCORER1P6_CKPT] = L.SCORER1P6_CKPT_SHA256
assert refusals(swap(ARM, "--out", L.SCORER1P6_ROOT + "/c0")) and refusals(swap(ARM, "--out", "/workspace/data/refe_s1z/c0"))
assert refusals(ARM) == []
print("G2 ok")

# ---- G3 fingerprints


def fp(n):
    return hashlib.sha256(repr(sorted(L.PROFILES[n].items())).encode()).hexdigest()


REFE_SCORER1P6_FP = "e74d64617f514d35"          # the digest of refe_scorer1p6 at the tip 62df1e64 (its data pin set): S1Z must not move it
assert fp("refe_scorer1p6").startswith(REFE_SCORER1P6_FP), fp("refe_scorer1p6")
assert fp("refe_s1z_p6") == fp("refe_s1z_p6") and "function" not in repr(sorted(PROF.items())) and "0x" not in repr(sorted(PROF.items())), "the digest of the new profile must be stable"
base = L.PROFILES["refe_scorer1p6"]
assert PROF is not base and PROF["recipe"] is not base["recipe"] and PROF["required_flags"] is not base["required_flags"] and PROF["arm_seed"] is not base["arm_seed"]
assert PROF["data_expect"] == base["data_expect"] and PROF["data_pin"] is base["data_pin"] and PROF["data_pin"] == L.SCORER1P6_DATA_PIN and PROF["ckpt_sha256"] == base["ckpt_sha256"]
assert PROF["navtest_sha256"] == base["navtest_sha256"] and PROF["recipe"] == base["recipe"] and PROF["data_expect"]["nav_mode"] == "command"
assert base["arm_seed"] == L.SCORER1P6_ARM_SEED and "S1Z" not in L.SCORER1P6_ARM_SEED, "the SCORER1P6 arm table is untouched"
assert not any(n.startswith("test_s1z") for n in [x["script"] for x in base["suite"]]), "the SCORER1P6 suite is untouched"
print("G3 ok (refe_scorer1p6 = %s...; refe_s1z_p6 = %s...)" % (fp("refe_scorer1p6")[:16], fp("refe_s1z_p6")[:16]))

# ---- G4 navtest in the training data
assert has(refusals(swap(ARM, "--onpolicy", "/workspace/data/navtest/sets")), "NAVTEST IS NEVER TRAINING DATA")
assert has(refusals(swap(ARM, "--pdm-labels-train", "/workspace/data/navtest_pdm")), "NAVTEST IS NEVER TRAINING DATA")
assert refusals(drop(ARM, "--navtest-logs")) and refusals(drop(ARM, "--navtest-tokens"))
ns, err = C.parse_scorer_argv(S.build_parser(), ARM)
assert not err, err
data_ok = {"nav_mode": "command", "heldout_logs": 24, "train_rows_in_heldout_logs": 0, "navtest": {k.split(".", 1)[1]: v for k, v in L.SCORER1_NAVTEST_EXPECT.items()}}
rows = [r for r in C.scorer_expect_rows(PROF, ns, data_ok) if r["flag"].startswith("data.navtest")]
assert len(rows) == 6 and all(r["ok"] for r in rows), rows
for key in ("train_rows_in_navtest_logs", "train_rows_in_navtest_tokens", "bank_sets_in_navtest_logs", "bank_sets_in_navtest_tokens"):
    bad = json.loads(json.dumps(data_ok))
    bad["navtest"][key] = 3
    rr = [r for r in C.scorer_expect_rows(PROF, ns, bad) if r["flag"] == "data.navtest." + key]
    assert len(rr) == 1 and rr[0]["ok"] is False, (key, rr)
bad = json.loads(json.dumps(data_ok))
bad["navtest"]["logs"] = 0
assert any(not r["ok"] for r in C.scorer_expect_rows(PROF, ns, bad) if r["flag"] == "data.navtest.logs"), "an empty navtest list cannot pass"
print("G4 ok")

# ---- G5 judge_g18 from the raw fields of the REAL TwinLoss
HEADS = ["NC", "DAC", "TTC"]


def real_g18(mutate=None):
    for e in S.TWIN_MUTATE_ENVS:
        os.environ.pop(e, None)
    if mutate in ("noradius", "notopk"):
        os.environ["REFE_TWIN_MUTATE_" + mutate.upper()] = "1"
    try:
        tl = S.TwinLoss(HEADS, 1.0, 8)
        g = torch.Generator().manual_seed(5)
        share, bank = {}, {}
        for h in HEADS:
            share[h] = {"sets": 40, "sets_with_clean": 40, "sets_with_pair": 31, "pairs": 80, "hard": 120, "share_with_pair": 31 / 40}
            bank[h] = {"sets": 100, "sets_with_clean": 90, "sets_with_pair": 70, "share_with_pair": 0.7}
        drop_ = mutate == "droploss"
        return {"loss_grad_delta_w1_minus_w0": {m: (0.0 if drop_ else 1.5e-3) for m in C.SCORER_GROUPS}, "control_w0_term_alone_grad_abs": {m: 0.0 for m in C.SCORER_GROUPS},
                "controls_exactly_zero": True, "micro_batch_pairs": {h: {"sets": 2, "sets_with_clean": 2, "sets_with_pair": 1, "pairs": 2, "hard": 3} for h in HEADS}, "share_micro": share,
                "micro_batches": 16, "bank_share": bank, "bank_stride": 5, "heads_built": list(tl.heads), "ade_built": tl.ade_built, "topk_built": tl.topk_built,
                "radius_filter_built": tl.radius_filter, "mining_built": tl.mining, "mutate_noradius": mutate == "noradius", "mutate_notopk": mutate == "notopk", "mutate_drop_loss": drop_,
                "loss_grad_noise_w0_vs_w0": {m: (0.0 if m == "score_head" else 1e-6) for m in C.SCORER_GROUPS}, "noise_factor": S.TWIN_NOISE_FACTOR,
                "probe_batch": {"found": True, "index": 3, "searched": 4, "search_limit": S.TWIN_PROBE_SEARCH_MAX, "pairs_per_head": {h: {"pairs": 2} for h in HEADS}, "reason": None},
                "ok": not mutate}
    finally:
        for e in S.TWIN_MUTATE_ENVS:
            os.environ.pop(e, None)


def judge(g):
    return C.judge_g18(g, ns)


assert judge(real_g18()) == [], judge(real_g18())
nr, nk, dl = judge(real_g18("noradius")), judge(real_g18("notopk")), judge(real_g18("droploss"))
assert any("radius filter is missing" in r for r in nr) and any("regression arm" in r for r in nr), nr
assert any("top-K mining is missing" in r for r in nk), nk
assert any("not live IN the loss" in r for r in dl), dl
g = real_g18()
g["bank_share"]["DAC"]["share_with_pair"] = 0.0
assert any("head DAC has share 0" in r for r in judge(g)), "share 0 on any head is a blocker"
g = real_g18()
g["share_micro"]["TTC"]["share_with_pair"] = 0.0
assert any("head TTC has share 0" in r and "share_micro" in r for r in judge(g))
g = real_g18()
g["control_w0_term_alone_grad_abs"]["score_head"] = 1e-9
assert any("EXACTLY 0.0" in r for r in judge(g))
g = real_g18()
g["heads_built"] = ["NC", "DAC"]
assert any("heads_built" in r for r in judge(g))
g = real_g18()
g["topk_built"] = 16
assert any("topk_built" in r for r in judge(g))
g = real_g18()
g["micro_batches"] = 0
assert any("no micro-batch" in r for r in judge(g))
assert any("not a dict" in r for r in C.judge_g18(None, ns))
# 2026-11-10: the noise rule and the probe batch are re-judged from the raw fields (the pod preflight probed a pairless micro-batch: delta was noise and score_head read 0.0)
assert C.G18_NOISE_FACTOR == S.TWIN_NOISE_FACTOR == 10.0, "the gate and the trainer must hold the same noise factor"
g = real_g18()
g["loss_grad_noise_w0_vs_w0"]["score_dec"] = 1.5e-3                       # noise == delta: RED by the 10x rule alone
assert any("does not exceed" in r and "score_dec" in r for r in judge(g)), judge(g)
g = real_g18()
g["loss_grad_noise_w0_vs_w0"]["score_q_mlp"] = 2e-4                       # delta 7.5x noise: under the 10x rule, RED
assert any("does not exceed" in r and "score_q_mlp" in r for r in judge(g)), judge(g)
g = real_g18()
g["loss_grad_noise_w0_vs_w0"]["score_q_mlp"] = 1e-4                       # delta 15x noise: passes (the rule is not a switch)
assert judge(g) == [], judge(g)
g = real_g18()
del g["loss_grad_noise_w0_vs_w0"]
assert any("no finite w=0 vs w=0 noise floor" in r for r in judge(g))
g = real_g18()
g["noise_factor"] = 2.0
assert any("noise_factor" in r for r in judge(g))
g = real_g18()
g["probe_batch"] = {"found": False, "reason": "searched 400"}
assert any("probe ran on no micro-batch" in r for r in judge(g))
g = real_g18()
g["micro_batch_pairs"]["DAC"]["pairs"] = 0                               # the pod's 2026-11-10 record: 0 pairs in the probe micro-batch
assert any("head DAC has no kept pair in the probe micro-batch" in r for r in judge(g))
print("G5 ok")

# ---- G6 G-DVB and the obliged gates
for d in ("twin_w", "twin_ade", "twin_topk", "twin_heads"):
    assert C.DVB_KINDS_SCORER[d] == "built" and d in C.SCORER_TWIN_DESTS, d
assert C.SCORER_GATES["G18"] == "twin_live"
data = {"twin": S.TwinLoss(HEADS, 1.0, 8).built_facts(0.37)}
rows = [r for r in C.scorer_dvb_rows(ns, data, PROF) if r["dest"].startswith("twin")]
assert len(rows) == 7 and all(r["ok"] for r in rows), [r for r in rows if not r["ok"]]
for key, val in (("ade", 1e6), ("topk", 16), ("heads", ["NC", "DAC"]), ("w", 0.5), ("radius_filter", False), ("mining", False)):
    bad = json.loads(json.dumps(data))
    bad["twin"][key] = val
    assert any(not r["ok"] for r in C.scorer_dvb_rows(ns, bad, PROF) if r["dest"].startswith("twin")), key
assert any(not r["ok"] for r in C.scorer_dvb_rows(ns, {}, PROF) if r["dest"].startswith("twin")), "no twin block in the data event: RED"
req = C.scorer_required_gates(ns)
assert "G18" in req and "G16" not in req, req
ns_cal, err = C.parse_scorer_argv(S.build_parser(), drop(ARM, "--twin-w") + ["--pair-calibrate"])
assert not err and "G18" not in C.scorer_required_gates(ns_cal)
ns_off, err = C.parse_scorer_argv(S.build_parser(), drop(drop(drop(drop(ARM, "--twin-w"), "--twin-heads"), "--twin-ade"), "--twin-topk"))
assert not err and "G18" not in C.scorer_required_gates(ns_off) and [r for r in C.scorer_dvb_rows(ns_off, {}, PROF) if r["dest"].startswith("twin")] == [], "OFF: no twin row, no twin gate"
print("G6 ok")

# ---- G7 the scripts
bash = None
for cand in ("C:/Program Files/Git/bin/bash.exe", "C:/Program Files/Git/usr/bin/bash.exe"):
    if os.path.exists(cand):
        bash = cand
        break
bash = bash or shutil.which("bash")
pod, chain, chain_old = os.path.join(HERE, "s1z_pod.sh"), os.path.join(HERE, "s1z_chain.sh"), os.path.join(HERE, "scorer1p6_chain.sh")
assert os.path.exists(pod) and os.path.exists(chain) and os.path.exists(chain_old)
ctext = open(chain, "rb").read().decode("utf-8")
assert "\r" not in ctext and "\r" not in open(pod, "rb").read().decode("utf-8"), "LF only"
joined = "\n".join(ln for ln in ctext.split("\n") if not ln.lstrip().startswith("#"))
assert "STAGE0_READING" not in joined and "P6_SPEED_READING" not in joined and "lane" not in joined.lower(), "no human gate, no lane"
assert "refe-plan-p6" not in joined and "refe-plan-p7" not in joined and "refe-plan-scorer1p6" not in joined and "$DATA/refe_scorer1p6" not in joined, "own dirs only"
assert "--profile refe_s1z_p6" in joined and "refe-plan-s1z" in joined and L.S1Z_CARD_SHA256 in ctext and L.SCORER1P6_CKPT_SHA256 in ctext
assert "--pair-calibrate" not in joined, "--pair-calibrate is in no launch argv"
assert re.search(r"for ARMN in S1Z; do", joined) and "ZZS1Z_CHAIN_DONE" in joined and '--prefix "s1z"' in joined
assert not re.search(r"\bkill -|SIGSTOP|SIGCONT|pkill|taskkill|ZZP7|P7_PAUSE|pause_p7|resume_p7|lane_pause|lane_restart|refe_p7/", joined), "the chain neither pauses nor resumes P7 (it only READS the P7 chain lock)"
for noradius in ("mutate_$MUT", "noradius notopk droploss"):
    assert noradius in joined, noradius
assert "gate_is $OUT/mutate_$MUT.log G18 ok false" in joined


def fn_body(text, name):
    m1 = re.search(r"^%s\(\) \{.*$" % re.escape(name), text, re.M)                           # the opening line (. does not cross a newline here)
    if m1 and "; }" in m1.group(0):                                                          # a one-line function (its closing brace sits on the same line)
        return m1.group(0)
    m = re.search(r"^%s\(\) \{.*?^\}" % re.escape(name), text, re.S | re.M)
    assert m, name
    return m.group(0)


old_text = open(chain_old, "rb").read().decode("utf-8")
norm = lambda t: t.replace("SCORER1P6_", "S1Z_")          # noqa: E731
for name in ("gpu_busy", "lock_held", "wait_clear"):
    assert norm(fn_body(old_text, name)) == fn_body(ctext, name), name                       # the wait for no chain lock and an idle GPU is the existing function, byte for byte
assert norm(fn_body(old_text, "others_hold")).replace("refe_p7; do", "refe_p7 refe_scorer1p6; do") == fn_body(ctext, "others_hold"), "others_hold differs by exactly the one added lock"
if bash:
    for sh in (pod, chain):
        r = subprocess.run([bash, "-n", sh], capture_output=True, encoding="utf-8")
        assert r.returncode == 0, (sh, r.stderr[:300])

    def gp(toks):
        return sorted((f, tuple(v)) for f, v in GT.flag_pairs(toks))
    env = dict(os.environ, ARM="S1Z", MODE="argv", S1Z_WS="/workspace")
    out = subprocess.run([bash, pod], env=env, capture_output=True, encoding="utf-8").stdout.strip().split()
    cal_real = "/workspace/data/refe_s1z/calibrate_s1z/pair_weight.json"
    tw_real = json.load(open(cal_real))["twin_w"] if os.path.exists(cal_real) else None
    assert gp(out) == gp(L.s1z_argv("S1Z", twin_w=tw_real)), set(gp(out)) ^ set(gp(L.s1z_argv("S1Z", twin_w=tw_real)))
    assert "--pair-calibrate" not in out
    with tempfile.TemporaryDirectory(dir=os.environ.get("SCORER1_TMP") or None) as td:
        cal = os.path.join(td, "data", "refe_s1z", "calibrate_s1z")
        os.makedirs(cal)
        json.dump({"twin_w": 0.37}, open(os.path.join(cal, "pair_weight.json"), "w"))
        out = subprocess.run([bash, pod], env=dict(os.environ, ARM="S1Z", MODE="argv", S1Z_WS=td.replace(chr(92), "/")), capture_output=True, encoding="utf-8").stdout.strip().split()
        assert "--twin-w" in out and out[out.index("--twin-w") + 1] == "0.37", out[-12:]
        r = subprocess.run([bash, pod], env=dict(os.environ, ARM="S1E", MODE="argv"), capture_output=True, encoding="utf-8")
        assert r.returncode == 2, "an unknown ARM is refused"
    print("G7 ok")
else:
    print("G7 SKIPPED (no bash)")
print("ZZTEST_PASS")
