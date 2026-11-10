#!/usr/bin/env python3
"""SCORER1P6 launch-gate tests (CPU, no data).  Prints ZZTEST_PASS.

  T1 THE DATA PIN: while launch_gate.SCORER1P6_DATA_PIN is None the gate REFUSES every arm
  T2 the legal argv of C0, C0r, S1E and S1 (with a pin set on a copy of the profile): no refusal; no lane flag or lane path anywhere
  T3 refusals: wrong seed / checkpoint sha / a lane flag / a control with a --pair-* flag / S1E without --pair-w, with EP balanced, without the EP gap, with the pooled mean, --pair-calibrate, a navtest data path
  T4 the 17 existing profile fingerprints are unmoved by the new profile (pinned digests = v3/fp_v3.json, the base tip); refe_scorer1p6 is a NEW dict sharing no mutable object with refe_scorer1
  T5 a BUILT nav_mode=command fact: the G-DVB data_expect row is RED when the data event lacks it or says goal, GREEN with 'command'; refe_scorer1 / refe_sft4r register no such row
  T6 the three S1E regressions THROUGH THE GATE (judge_g16 from the raw fields of the REAL PairLoss): control PASS; NOBALANCE (no balancing on a safety head), EPBALANCE (balancing applied to EP), DROP_LOSS go RED
  T7 the pod scripts: bash -n; MODE=argv of scorer1p6_pod.sh equals scorer1p6_argv arm by arm (a calibrate file adds exactly --pair-w); the chain has no human gate, no lane, its own dirs, and
     --pair-calibrate is in no launch argv; the pinned card sha256 is the frozen card's (when the card is beside this machine's artifacts)
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

PINNED_FP = {
    "refe_aux_s1": "f95f48727b62ccc6725eb833330b2674921d535de7d161ba746a770481e618cf",
    "refe_aux_s1c": "dcdf2dbd2a674f1c778d92f6bfdd1bdd4e9630c2cb8699a2c83c61390d2247c2",
    "refe_field": "5ddc49053a99fa78fa3118e49d036261addd8796a38c6b3b570358934e62d8e3",
    "refe_field_ctrl": "0b6281b1bfa996ed6c00a869dda11279cdb09b70408ebd006149c1c76c90cd4c",
    "refe_hnscreen": "d30126bbb5a3295745f3e8dc0ccb93818b84f33860aae1ee85f29428908cdbfe",
    "refe_p2": "627341286aea3506d7518345c8d9c71aafdb26b69677810c71cb0e876ad46390",
    "refe_p2rep": "44ca107f53aceee853ed0b465bf8fe6d8a8d31d0bdcf12313610ee30605e69ea",
    "refe_p6": "80fa4d7ec757ad4658df04b1ecfd453ae98653e729e2ee24458769d851cdf9c3",
    "refe_p7": "94cca167ce121a94c67ad57b7bda72a2929b87263f55548e7eb26b422eb78fd1",
    "refe_rehearsal": "821cd1ef3bf55d4e07ea8f6e774047a677ff3f70a763d957368a61ab2aa1d424",
    "refe_rls1": "bce7e25ee8718c08f5de8bdac9867aebc7704f995a360fda0effb561afc6ec13",
    "refe_rls1b": "5d6bfbc10152ae964949d10e901beccd9abe460427bef41e1249228f2c056384",
    "refe_rls1c": "26214cf954099d51ebb3ef9576f8a51a1561d860f58c7e4d5b235d8c560cbac6",
    "refe_rls1d": "fcc2585f2a5d9b942b6be687a006499656861982d01fd6ad700fd3437840f1a2",
    "refe_rls1e": "54d96289b19c3cd60ae739a52f2bd87ad8c32288adc619d45b97754ac8a8ba4d",
    "refe_scorer1": "88d02fb7b34d4fd8c2c07187649e139e2592d6b83fd31431c7f8af72ce7f5e5e",
    "refe_sft4r": "8972e3c348206b94cdd4bd6aa3d6cc90ab1b7f3df014acee5dde3b88434ffae2",
}

PROF = L.PROFILES["refe_scorer1p6"]
PIN = {"train_sets": 60000, "heldout_sets": 3137}
PROF_PIN = {**PROF, "data_pin": PIN, "data_expect": L.scorer1p6_data_expect(PIN)}
SHA = {L.SCORER1P6_CKPT: L.SCORER1P6_CKPT_SHA256}
for _k, _f in (("--navtest-logs", L.SCORER1P6_ROOT + "/navtest_ids/navtest_logs.json"), ("--navtest-tokens", L.SCORER1P6_ROOT + "/navtest_ids/navtest_tokens.txt.gz")):
    SHA[_f] = L.SCORER1_NAVTEST_SHA256[_k]


def fake_sha(path):
    if path in SHA:
        return SHA[path]
    raise FileNotFoundError(path)


L.sha256_of_file = fake_sha


def refusals(argv, prof=PROF_PIN):
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


ARMS = {a: L.scorer1p6_argv(a, 0.37 if a in L.SCORER1P6_PAIR_SPEC else None) for a in L.SCORER1P6_ARM_SEED}

# ---- T1 the data pin
if L.SCORER1P6_DATA_PIN is None:
    for a, av in ARMS.items():
        assert has(refusals(av, PROF), "DATA PIN IS UNSET"), (a, refusals(av, PROF))
    assert "train_sets" not in PROF["data_expect"], "an unset pin must register no label count"
else:
    for a, av in ARMS.items():
        assert not has(refusals(av, PROF), "DATA PIN IS UNSET"), a
print("T1 ok (pin unset: %s -> every arm refused)" % (L.SCORER1P6_DATA_PIN is None))

# ---- T2 legal argv per arm
for a, av in ARMS.items():
    assert refusals(av) == [], (a, refusals(av))
s1e = ARMS["S1E"]
assert s1e[s1e.index("--pair-heads") + 1] == "NC,DAC,EP,TTC" and s1e[s1e.index("--pair-gap-heads") + 1] == "EP=0.05" and s1e[s1e.index("--pair-mean") + 1] == "head"
assert "--pair-w" not in L.scorer1p6_argv("S1E") and "--pair-w" in s1e
assert ARMS["C0r"][ARMS["C0r"].index("--seed") + 1] == "1"
assert not any("lane" in t for av in ARMS.values() for t in av), "no lane flag or lane path in any arm"
print("T2 ok")

# ---- T3 refusals
C0, C0R, S1E, S1 = ARMS["C0"], ARMS["C0r"], ARMS["S1E"], ARMS["S1"]
assert has(refusals(swap(C0, "--seed", "1")), "--seed")
assert has(refusals(C0 + ["--lane-head"]), "--lane-head") and has(refusals(C0 + ["--teacher-lane-train", "/x"]), "teacher-lane")
assert refusals(C0 + ["--pair-heads", "NC"]) and refusals(C0 + ["--pair-mean", "head"]) and refusals(C0 + ["--pair-nobalance-heads", "EP"])
assert has(refusals(drop(S1E, "--pair-w")), "--pair-w") and has(refusals(swap(S1E, "--pair-w", "0")), "> 0")
assert refusals(drop(S1E, "--pair-nobalance-heads")), "EP balanced (flag dropped) must be refused"
assert refusals(drop(S1E, "--pair-nospeed-heads")) and refusals(drop(S1E, "--pair-gap-heads")) and refusals(swap(S1E, "--pair-gap-heads", "EP=0.0"))
assert refusals(swap(S1E, "--pair-mean", "pooled")) and refusals(drop(S1E, "--pair-mean"))
assert refusals(swap(S1E, "--pair-heads", "NC,DAC,TTC")), "S1E without EP is S1, not S1E"
assert refusals(S1 + ["--pair-gap-heads", "EP=0.05"])
assert has(refusals(S1E + ["--pair-calibrate"]), "--pair-calibrate")
SHA[L.SCORER1P6_CKPT] = "0" * 64
assert has(refusals(C0), "registered P6 checkpoint")
SHA[L.SCORER1P6_CKPT] = L.SCORER1P6_CKPT_SHA256
assert has(refusals(swap(C0, "--onpolicy", "/workspace/data/navtest/sets")), "NAVTEST IS NEVER TRAINING DATA"), "a navtest data path must be refused"
assert refusals(swap(C0, "--out", "/workspace/data/refe_scorer1/c0"))
assert refusals(C0) == []
print("T3 ok")


# ---- T4 fingerprints
def fp(n):
    return hashlib.sha256(repr(sorted(L.PROFILES[n].items())).encode()).hexdigest()


moved = [n for n, want in PINNED_FP.items() if fp(n) != want]
assert not moved, f"existing profile fingerprints moved: {moved}"
assert set(L.PROFILES) - set(PINNED_FP) == {"refe_scorer1p6", "refe_s1z_p6", "refe_s1zsm_p6", "refe_rep_p6", "refe_rep_s1z_p6", "refe_rep_s1zsm_p6", "refe_dose_s1z_p6"}, set(L.PROFILES) - set(PINNED_FP)         # S1Z (PREREG_S1Z_P6) added a profile; the 17 older fingerprints above are unchanged
assert PROF is not L.PROFILES["refe_scorer1"] and PROF["recipe"] is not L.PROFILES["refe_scorer1"]["recipe"] and "lane_weight" not in PROF["recipe"] and "lane_weight" in L.PROFILES["refe_scorer1"]["recipe"]
assert "--lane-head" not in PROF["required_flags"] and "--lane-head" in L.PROFILES["refe_scorer1"]["required_flags"]
print(f"T4 ok ({len(PINNED_FP)} existing fingerprints unmoved; refe_scorer1p6 = {fp('refe_scorer1p6')[:16]}...)")

# ---- T5 the built nav_mode fact
ns, err = C.parse_scorer_argv(S.build_parser(), S1E)
assert not err, err
rows = C.scorer_expect_rows({"data_expect": {"nav_mode": "command"}}, ns, {"nav_mode": "command"})
assert [r["ok"] for r in rows] == [True]
for bad in ({}, {"nav_mode": "goal"}):
    rows = C.scorer_expect_rows({"data_expect": {"nav_mode": "command"}}, ns, bad)
    assert [r["ok"] for r in rows] == [False], bad
assert PROF["data_expect"]["nav_mode"] == "command" and "nav_mode" not in L.PROFILES["refe_scorer1"]["data_expect"] and "nav_mode" not in L.PROFILES["refe_sft4r"]["data_expect"]
print("T5 ok")

# ---- T6 the three regressions through the gate
HEADS = ["NC", "DAC", "EP", "TTC"]


def real_g16(mutate=None):
    for e in S.PAIR_MUTATE_ENVS:
        os.environ.pop(e, None)
    if mutate in ("nobalance", "epbalance"):
        os.environ["REFE_PAIR_MUTATE_" + mutate.upper()] = "1"
    try:
        pl = S.PairLoss(HEADS, 0.15, nobalance=["EP"], nospeed=["EP"], gaps={"EP": 0.05}, reduce="head")
        g = torch.Generator().manual_seed(5)
        mb, bank = {}, {}
        for h in HEADS:
            kf = ks = sets = withp = 0
            for n in range(40):
                speed = torch.rand(64, generator=g) * 14.0
                cxy = torch.zeros(1, 64, 20, 3)
                cxy[0, :, :, 0] = (speed[:, None] * 0.2) * torch.arange(1, 21)[None, :]
                y = (torch.rand(64, generator=g) > 0.5).float() if h != "EP" else torch.rand(64, generator=g)
                ii, jj, cF, cS, kF, kS = pl.select(y, S.chord_speed(cxy)[0], 10.0, head=h)
                sets += 1
                withp += int(kF + kS > 0)
                kf, ks = kf + kF, ks + kS
            mb[h] = {"kept_F": kf, "kept_S": ks}
            bank[h] = {"kept_F": kf, "kept_S": ks, "share_with_pair": float(withp / sets)}
        drop_ = mutate == "droploss"
        return {"loss_grad_delta_w1_minus_w0": {m: (0.0 if drop_ else 1.5e-3) for m in C.SCORER_GROUPS}, "control_w0_term_alone_grad_abs": {m: 0.0 for m in C.SCORER_GROUPS},
                "controls_exactly_zero": True, "micro_batch_pairs": mb, "bank_share": bank, "balanced_built": bool(pl.balance), "rules_built": {h: dict(r) for h, r in pl.rules.items()},
                "reduce_built": pl.reduce, "mutate_nobalance": mutate == "nobalance", "mutate_epbalance": mutate == "epbalance", "mutate_drop_loss": drop_, "ok": not mutate}
    finally:
        for e in S.PAIR_MUTATE_ENVS:
            os.environ.pop(e, None)


ns6, _ = C.parse_scorer_argv(S.build_parser(), S1E)


def judge(mutate=None):
    return C.judge_g16(real_g16(mutate), ns6)


assert judge(None) == [], judge(None)
nb, ep, dl = judge("nobalance"), judge("epbalance"), judge("droploss")
assert nb and any("F/S ratio != 1 on head NC" in r for r in nb), nb
assert ep and any("head EP was built with rule" in r for r in ep), ep
assert dl and any("not live IN the loss" in r for r in dl), dl
g_old = real_g16(None)
g_old.pop("rules_built")
g_old.pop("reduce_built")
assert C.judge_g16(g_old, ns6), "the P6 argv needs rules_built"
g = real_g16(None)
g["micro_batch_pairs"]["EP"] = {"kept_F": 10, "kept_S": 3}
assert C.judge_g16(g, ns6) == [], "an unbalanced EP does not need kept F == kept S"
g["micro_batch_pairs"]["NC"] = {"kept_F": 10, "kept_S": 3}
assert C.judge_g16(g, ns6), "a safety head does"
g = real_g16(None)
g["bank_share"]["EP"]["share_with_pair"] = 0.0
assert any("head EP has share 0" in r for r in C.judge_g16(g, ns6)), "share 0 on any head is a blocker"
print("T6 ok")

# ---- T7 scripts
bash = None
for cand in ("C:/Program Files/Git/bin/bash.exe", "C:/Program Files/Git/usr/bin/bash.exe"):
    if os.path.exists(cand):
        bash = cand
        break
bash = bash or shutil.which("bash")
pod, chain = os.path.join(HERE, "scorer1p6_pod.sh"), os.path.join(HERE, "scorer1p6_chain.sh")
assert os.path.exists(pod) and os.path.exists(chain)
ctext = open(chain, "rb").read().decode("utf-8")
assert "\r" not in ctext and "\r" not in open(pod, "rb").read().decode("utf-8"), "LF only"
joined = "\n".join(ln for ln in ctext.split("\n") if not ln.lstrip().startswith("#"))
assert "STAGE0_READING" not in joined and "P6_SPEED_READING" not in joined and "lane" not in joined.lower(), "no human gate, no lane"
assert "refe-plan-p6" not in joined and "refe-plan-p7" not in joined, "the chain must not touch the P6 / P7 package dirs"
assert "--profile refe_scorer1p6" in joined and "refe-plan-scorer1p6" in joined and L.SCORER1P6_CARD_SHA256 in ctext and L.SCORER1P6_CKPT_SHA256 in ctext
assert "--pair-calibrate" not in joined, "--pair-calibrate is in no launch argv"
assert re.search(r"for ARMN in C0 C0r S1E; do", joined)
assert L.PROFILES["refe_scorer1p6"].get("data_expect_min") == {}, "no inherited floor: refe_sft4r's teacher-lane floor cannot hold for arms that forbid every lane flag"
card = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p6base/PREREG_SCORER1P6.md"
if os.path.exists(card):
    assert hashlib.sha256(open(card, "rb").read()).hexdigest() == L.SCORER1P6_CARD_SHA256, "the pinned card sha256 is not the frozen card"
if bash:
    for sh in (pod, chain):
        r = subprocess.run([bash, "-n", sh], capture_output=True, encoding="utf-8")
        assert r.returncode == 0, (sh, r.stderr[:300])

    def gp(toks):
        return sorted((f, tuple(v)) for f, v in GT.flag_pairs(toks))
    for arm in L.SCORER1P6_ARM_SEED:
        env = dict(os.environ, ARM=arm, MODE="argv", SCORER1P6_WS="/workspace")
        out = subprocess.run([bash, pod], env=env, capture_output=True, encoding="utf-8").stdout.strip().split()
        # on the pod, once the chain has calibrated, the script rightly adds the measured --pair-w; the pin is then compared WITH that value
        cal_real = "/workspace/data/refe_scorer1p6/calibrate_" + arm.lower() + "/pair_weight.json"
        pw_real = json.load(open(cal_real))["pair_w"] if L.SCORER1P6_PAIR_SPEC.get(arm) and os.path.exists(cal_real) else None
        assert gp(out) == gp(L.scorer1p6_argv(arm, pair_w=pw_real)), (arm, set(gp(out)) ^ set(gp(L.scorer1p6_argv(arm, pair_w=pw_real))))
        assert "--pair-calibrate" not in out
    with tempfile.TemporaryDirectory(dir=os.environ.get("SCORER1_TMP") or None) as td:
        cal = os.path.join(td, "data", "refe_scorer1p6", "calibrate_s1e")
        os.makedirs(cal)
        json.dump({"pair_w": 0.37}, open(os.path.join(cal, "pair_weight.json"), "w"))
        out = subprocess.run([bash, pod], env=dict(os.environ, ARM="S1E", MODE="argv", SCORER1P6_WS=td.replace(chr(92), "/")), capture_output=True, encoding="utf-8").stdout.strip().split()
        assert "--pair-w" in out and out[out.index("--pair-w") + 1] == "0.37", out[-12:]
    print("T7 ok")
else:
    print("T7 SKIPPED (no bash)")
print("ZZTEST_PASS")
