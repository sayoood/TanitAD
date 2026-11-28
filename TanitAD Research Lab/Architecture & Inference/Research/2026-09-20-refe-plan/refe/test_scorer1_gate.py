#!/usr/bin/env python3
"""SCORER1 launch-gate profile tests (CPU only, no data): launch_gate.PROFILES['refe_scorer1'].  Prints ZZTEST_PASS.

Every refusal below is a deliberate regression: the profile must go RED on it, and the legal argv must come back with no refusal.
  T1 the three arms' pinned argv are accepted (checkpoint and navtest lists bound by sha256 through a patched hasher)
  T2 --pair-calibrate (a measuring mode), --dump-scores and a stray flag are refused in a launch
  T3 the checkpoint: another sha256 / an unreadable file is refused (Amendment 1 item 7)
  T4 the arms: C0 / C0r with any --pair-* flag, S1 without --pair-w / with --pair-w 0 / with other heads / dv, a wrong seed are refused
  T5 --max-updates: 400 allowed, 200 / 2733 / absent refused
  T6 NAVTEST: a data path naming navtest, a missing / wrong navtest list are refused
  T7 the trainer's data event: the navtest block of zeros is G-DVB green; one training set on a navtest log or token, an empty list, an absent block are RED
  T8 trainer helpers: navtest_audit counts, load_navtest_ids fails closed on an empty list
  T9 refe_sft4r is unchanged by the generalised recipe (its own pin still has no refusal)
"""
import gzip
import json
import os
import sys
import tempfile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gate_checks as C  # noqa: E402
import launch_gate as L  # noqa: E402
import scorer_finetune as S  # noqa: E402

PROF = L.PROFILES["refe_scorer1"]
SHA = {L.SCORER1_CKPT: L.SCORER1_CKPT_SHA256, L.SCORER1_NAVTEST_LOGS: L.SCORER1_NAVTEST_SHA256["--navtest-logs"], L.SCORER1_NAVTEST_TOKENS: L.SCORER1_NAVTEST_SHA256["--navtest-tokens"]}
_real = L.sha256_of_file


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
    """replace (or add) a flag's values"""
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


C0, C0R, S1 = L.scorer1_argv("C0"), L.scorer1_argv("C0r"), L.scorer1_argv("S1", 0.37)

# ---- T1 the legal argv of every arm
for arm, av in (("C0", C0), ("C0r", C0R), ("S1", S1)):
    rs = refusals(av)
    assert rs == [], (arm, rs)
assert "--seed" in C0 and C0R[C0R.index("--seed") + 1] == "1" and "--pair-w" in S1 and "--pair-w" not in C0
print("T1 ok")

# ---- T2 forbidden flags
assert has(refusals(S1 + ["--pair-calibrate"]), "--pair-calibrate"), "--pair-calibrate must be FORBIDDEN in a launch"
assert has(refusals(C0 + ["--dump-scores", "/x"]), "--dump-scores")
assert refusals(C0 + ["--w-field", "0.1"]), "a stray flag must be refused (the pin)"
print("T2 ok")

# ---- T3 the checkpoint is bound by bytes
SHA[L.SCORER1_CKPT] = "0" * 64
assert has(refusals(C0), "registered u400"), "another checkpoint sha256 must be refused"
del SHA[L.SCORER1_CKPT]
assert has(refusals(C0), "cannot bind the checkpoint"), "an unreadable checkpoint must be refused"
SHA[L.SCORER1_CKPT] = L.SCORER1_CKPT_SHA256
assert refusals(C0) == []
assert has(refusals(swap(C0, "--ckpt", "/workspace/data/refe_runs/vitl16_navtrain10_grow_tau0.3/model_final.pt")), "--ckpt"), "the pin refuses another --ckpt path"
print("T3 ok")

# ---- T4 the arm rules
for av, nm in ((C0, "C0"), (C0R, "C0r")):
    for extra in (["--pair-w", "0.3"], ["--pair-w", "0"], ["--pair-heads", "NC"], ["--pair-dv", "0.15"]):
        assert refusals(av + extra), (nm, extra)
assert has(refusals(drop(S1, "--pair-w")), "--pair-w"), "S1 without --pair-w"
assert has(refusals(swap(S1, "--pair-w", "0")), "> 0"), "S1 with --pair-w 0 is the control, not S1"
assert has(refusals(swap(S1, "--pair-w", "nan")), "> 0")
assert refusals(swap(S1, "--pair-heads", "NC,TTC")) and refusals(swap(S1, "--pair-dv", "0.2")) and refusals(drop(S1, "--pair-heads"))
assert refusals(swap(C0, "--seed", "1")), "C0 must be seed 0"
assert refusals(swap(C0R, "--seed", "0")), "C0r must be seed 1"
assert refusals(swap(S1, "--seed", "1")), "S1 must be seed 0"
assert refusals(swap(C0, "--out", "/workspace/data/refe_scorer1/elsewhere")), "--out must be one of the three arm dirs"
print("T4 ok")

# ---- T5 --max-updates 400 allowed (and required); the rest refused
assert refusals(C0) == []
assert has(refusals(swap(C0, "--max-updates", "200")), "--max-updates") and has(refusals(swap(C0, "--max-updates", "2733")), "--max-updates")
assert has(refusals(drop(C0, "--max-updates")), "--max-updates")
assert "--max-updates" in L.SFT4R_FORBIDDEN and "--max-updates" not in PROF["forbidden_flags"], "refe_sft4r still forbids it; refe_scorer1 allows it"
print("T5 ok")

# ---- T6 navtest is never training data
for flag in ("--onpolicy", "--targets", "--pdm-labels-train", "--teacher-lane-train", "--images"):
    bad = swap(C0, flag, "/workspace/data/refe_navtest/x")
    assert has(refusals(bad), "NAVTEST IS NEVER TRAINING DATA"), flag
assert has(refusals(swap(C0, "--onpolicy", "/workspace/data/NavTest_sets")), "NAVTEST IS NEVER TRAINING DATA"), "case-insensitive"
assert has(refusals(drop(drop(C0, "--navtest-logs"), "--navtest-tokens")), "--navtest-logs"), "the navtest lists must be named"
SHA[L.SCORER1_NAVTEST_LOGS] = "1" * 64
assert has(refusals(C0), "registered list"), "a navtest list of other bytes must be refused"
SHA[L.SCORER1_NAVTEST_LOGS] = L.SCORER1_NAVTEST_SHA256["--navtest-logs"]
assert refusals(C0) == []
# the held-out list may be named 'heldout'; navtrain paths pass
assert not has(refusals(C0), "navtest")
print("T6 ok")

# ---- T7 G-DVB on the trainer's data event (scorer_expect_rows is the check G-DVB runs)
ns, err = C.parse_scorer_argv(S.build_parser(), C0)
assert not err, err
base_data = {"train_sets": 174912, "train_sets_in_bank": 174912, "heldout_sets": 3137, "heldout_in_bank": 3137, "heldout_logs": 24, "train_rows_in_heldout_logs": 0,
             "lane_labels": {"pdm_train": 174912, "pdm_train_stale": 0, "pdm_train_superseded": 0, "pdm_heldout": 3137, "heldout_sets": 3137, "heldout_stale": 0,
                             "teacher_lane_heldout": 3137, "teacher_lane_train": 2109}}
good_nt = {"logs": 136, "tokens": 12146, "train_rows_in_navtest_logs": 0, "train_rows_in_navtest_tokens": 0, "bank_sets_in_navtest_logs": 0, "bank_sets_in_navtest_tokens": 0}


def red_rows(data):
    return [r for r in C.scorer_expect_rows(PROF, ns, data) if not r["ok"]]


assert red_rows({**base_data, "navtest": good_nt}) == [], red_rows({**base_data, "navtest": good_nt})
# the schedule row: ceil(174912 / 8 / 1 / 8) = 2733 capped by --max-updates 400 -> 400
assert [r for r in C.scorer_expect_rows(PROF, ns, {**base_data, "navtest": good_nt}) if r["flag"] == "schedule"][0]["built"] == 400
for k in ("train_rows_in_navtest_logs", "train_rows_in_navtest_tokens", "bank_sets_in_navtest_logs", "bank_sets_in_navtest_tokens"):
    assert any(r["flag"] == "data.navtest." + k for r in red_rows({**base_data, "navtest": {**good_nt, k: 1}})), k
assert red_rows({**base_data, "navtest": {**good_nt, "logs": 0}}), "an EMPTY navtest log list must not pass"
assert red_rows({**base_data, "navtest": {**good_nt, "tokens": 12}}), "a short token list must not pass"
assert len(red_rows(base_data)) >= 6, "no navtest block at all: every navtest row is RED"
print("T7 ok")

# ---- T8 the trainer's own helpers
with tempfile.TemporaryDirectory(dir=os.environ.get("SCORER1_TMP") or None) as td:
    lp, tp = os.path.join(td, "l.json"), os.path.join(td, "t.txt.gz")
    json.dump({"schema": "p7-navtest-logs-v1", "logs": ["LOGN1", "LOGN2"]}, open(lp, "w"))
    with gzip.open(tp, "wt") as fh:
        fh.write("tokA tokB\n")
    logs, toks = S.load_navtest_ids(lp, tp)
    assert logs == {"LOGN1", "LOGN2"} and toks == {"tokA", "tokB"}
    tr = [("LOGT", "t1", 0, 0), ("LOGN1", "t2", 0, 0), ("LOGT", "tokB", 1, 0)]
    bank = tr + [("LOGN2", "t9", 0, 0)]
    a = S.navtest_audit(iter(tr), bank, logs, toks)
    assert a == {"logs": 2, "tokens": 2, "train_rows_in_navtest_logs": 1, "train_rows_in_navtest_tokens": 1, "bank_sets_in_navtest_logs": 2, "bank_sets_in_navtest_tokens": 1}, a
    assert S.navtest_audit(iter([("LOGT", "t1", 0, 0)]), [("LOGT", "t1", 0, 0)], logs, toks)["train_rows_in_navtest_logs"] == 0
    json.dump({"logs": []}, open(lp, "w"))
    try:
        S.load_navtest_ids(lp, "")
    except RuntimeError:
        pass
    else:
        raise AssertionError("an empty log list must raise")
    try:
        S.load_navtest_ids(os.path.join(td, "missing.json"), "")
    except OSError:
        pass
    else:
        raise AssertionError("a missing list must raise")
# the real registered lists (shipped beside the tests when present): sizes and sha256
for fname, flag, n in (("navtest_logs.json", "--navtest-logs", 136), ("navtest_tokens.txt.gz", "--navtest-tokens", 12146)):
    p = os.path.join(HERE, fname)
    if os.path.exists(p):
        assert _real(p) == L.SCORER1_NAVTEST_SHA256[flag], fname
print("T8 ok")

# ---- T9 refe_sft4r untouched
P4 = L.PROFILES["refe_sft4r"]
assert "recipe" not in P4 and "scorer1_rules" not in P4 and P4["pinned_argv"] == tuple(L.SFT4R_ARGV.split())
assert L.profile_refusals(P4, L.SFT4R_ARGV.split()) == [], L.profile_refusals(P4, L.SFT4R_ARGV.split())
assert has(L.profile_refusals(P4, L.SFT4R_ARGV.split() + ["--max-updates", "400"]), "--max-updates"), "refe_sft4r must still forbid --max-updates"
print("T9 ok")
# ---- T10 the pod script's argv (what the chain feeds to the gate AND the trainer) equals the profile's pin, arm by arm; both scripts parse
import shutil
import subprocess
bash = None
for cand in ("C:/Program Files/Git/bin/bash.exe", "C:/Program Files/Git/usr/bin/bash.exe"):
    if os.path.exists(cand):
        bash = cand
        break
bash = bash or shutil.which("bash")
pod = os.path.join(HERE, "scorer1_pod.sh")
chain = os.path.join(HERE, "scorer1_chain.sh")
assert os.path.exists(pod) and os.path.exists(chain), "scorer1_pod.sh / scorer1_chain.sh must ship beside the tests"
if bash:
    for sh in (pod, chain):
        r = subprocess.run([bash, "-n", sh], capture_output=True, text=True)
        assert r.returncode == 0, (sh, r.stderr[:300])
    import gate_token as GT

    def gp(toks):
        return sorted((f, tuple(v)) for f, v in GT.flag_pairs(toks))
    for arm in ("C0", "C0r", "S1"):
        env = dict(os.environ, ARM=arm, MODE="argv", SCORER1_WS="/workspace")
        out = subprocess.run([bash, pod], env=env, capture_output=True, text=True).stdout.strip().split()
        want = L.scorer1_argv(arm)
        assert gp(out) == gp(want), (arm, set(gp(out)) ^ set(gp(want)))
    with tempfile.TemporaryDirectory(dir=os.environ.get("SCORER1_TMP") or None) as td:      # a pair_weight.json beside it adds exactly --pair-w to S1's argv
        cal = os.path.join(td, "data", "refe_scorer1", "calibrate")
        os.makedirs(cal)
        json.dump({"pair_w": 0.37}, open(os.path.join(cal, "pair_weight.json"), "w"))
        out = subprocess.run([bash, pod], env=dict(os.environ, ARM="S1", MODE="argv", SCORER1_WS=td.replace(chr(92), "/")), capture_output=True, text=True).stdout.strip().split()
        assert "--pair-w" in out and out[out.index("--pair-w") + 1] == "0.37", out[-12:]
    print("T10 ok")
else:
    print("T10 SKIPPED (no bash)")
print("ZZTEST_PASS")
