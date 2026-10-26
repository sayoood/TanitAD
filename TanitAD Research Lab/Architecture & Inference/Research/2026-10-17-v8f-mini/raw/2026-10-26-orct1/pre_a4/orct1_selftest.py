#!/usr/bin/env python3
"""orct1_selftest.py -- SMOKE-level self-test of orct1_pullcheck.py and orct1_manifest.py (plain python, no pytest).   Fixtures are tiny synthetic files under
<root>/run_<timestamp>/ ; NOTHING here is an O-RECOVER result.  Every mutant must be caught; the positive control (the complete fixture) must read COMPLETE first.

    python orct1_selftest.py [<root>]      default root C:/Users/Admin/orct1/smoke/selftest          rc 0 = every check as expected
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import orct1_manifest as MF  # noqa: E402
import orct1_pullcheck as PC  # noqa: E402

ROOT = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/Admin/orct1/smoke/selftest"
STAMP = time.strftime("run_%m%d_%H%M%S")
BASE = os.path.join(ROOT, STAMP)
RES: list[tuple[str, bool, str]] = []
N = [0]


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def wjson(p, o):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(o, f)


def make_run(arm="orc", seed=0, done=True, init="a" * 64):
    """a complete pulled run under a fresh ART; returns (art, run dir)"""
    N[0] += 1
    art = os.path.join(BASE, f"art{N[0]}")
    d = os.path.join(art, PC.run_name(arm, seed))
    os.makedirs(d)
    names = PC.expected_files(arm, seed)
    lev = {"init_ckpt": {"sha256": init}, "ego_perturb": {"ego_perturb_p": 0.3 if arm == "orc" else 0.0}}
    for n in names:
        p = os.path.join(d, n)
        if n == "ckpt_last.pt":
            open(p, "wb").write(os.urandom(300_000))
        elif n == "summary.json":
            wjson(p, {"done": done, "steps": 3000, "levers": lev})
        elif n == "selfcheck.json":
            wjson(p, {"declared_vs_built": "PASS", "levers": lev})
        elif n.endswith(".json"):
            wjson(p, {"x": n})
        else:
            open(p, "wb").write(os.urandom(1000))
    write_sums(d, names)
    return art, d


def write_sums(d, names, drop=None, extra=None, newline=True):
    lines = [f"{sha(open(os.path.join(d, n), 'rb').read())}  {n}" for n in names if n != drop]
    if extra:
        lines.append(f"{'0' * 64}  {extra}")
    with open(os.path.join(d, "SHA256SUMS.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + ("\n" if newline else ""))


def age(d):
    """make SHA256SUMS.txt look old (the box-2 chain finished long ago)"""
    p = os.path.join(d, "SHA256SUMS.txt")
    t = time.time() - 60
    os.utime(p, (t, t))


def expect(name, ok_wanted, art, arm="orc", seed=0, min_age=PC.MIN_AGE_S):
    ok, why, s = PC.check(art, arm, seed, min_age_s=min_age)
    good = (ok == ok_wanted) and (bool(s) == ok_wanted)
    RES.append((name, good, f"{'COMPLETE' if ok else 'INCOMPLETE'}: {why[:130]}"))


def main():
    os.makedirs(BASE)
    # ---------------------------------------------------------------- pull completeness: the positive control first
    art, d = make_run()
    age(d)
    expect("PC0 positive control: a complete, aged run is COMPLETE", True, art)
    _, why, s = PC.check(art, "orc", 0)
    RES.append(("PC0b the returned sha256 is the ckpt's", s == sha(open(os.path.join(d, 'ckpt_last.pt'), 'rb').read()), s[:16]))
    # a sums file written a moment ago may still be filling
    art, d = make_run()
    expect("PC1 SHA256SUMS.txt modified < 3 s ago -> INCOMPLETE", False, art)
    ok, why, _ = PC.check(art, "orc", 0, min_age_s=0.0)
    RES.append(("PC1b same run with min_age 0 -> COMPLETE (so PC1 failed ONLY on the age)", ok, why[:80]))
    # no directory / no sums
    expect("PC2 no run directory -> INCOMPLETE", False, os.path.join(BASE, "nowhere"))
    art, d = make_run(); os.rename(os.path.join(d, "SHA256SUMS.txt"), os.path.join(d, "SUMS.old"))
    expect("PC3 no SHA256SUMS.txt (ckpt etc. present) -> INCOMPLETE", False, art)
    # the sums file is short / long / wrong names
    art, d = make_run(); write_sums(d, PC.expected_files("orc", 0), drop="launch_gate_token.json"); age(d)
    expect("PC4 SHA256SUMS.txt lists 7 of 8 -> INCOMPLETE", False, art)
    art, d = make_run(); write_sums(d, PC.expected_files("orc", 0), extra="train.log"); age(d)
    expect("PC5 SHA256SUMS.txt lists a 9th name -> INCOMPLETE", False, art)
    art, d = make_run(); write_sums(d, PC.expected_files("orc", 0), newline=False); age(d)
    expect("PC6 SHA256SUMS.txt without final newline (partial last line) -> INCOMPLETE", False, art)
    art, d = make_run()
    t = open(os.path.join(d, "SHA256SUMS.txt")).read().rstrip("\n")
    open(os.path.join(d, "SHA256SUMS.txt"), "w", newline="\n").write(t[:-9] + "\n"); age(d)
    expect("PC7 last line cut inside its name -> INCOMPLETE", False, art)
    art, d = make_run()
    t = open(os.path.join(d, "SHA256SUMS.txt")).read().rstrip("\n")
    open(os.path.join(d, "SHA256SUMS.txt"), "w", newline="\n").write(t[:-60] + "\n"); age(d)
    expect("PC7b last line cut inside its hash -> INCOMPLETE", False, art)
    # the HALF-PULLED checkpoint: the sums (from a complete earlier pull) are intact, the file is being re-pulled
    art, d = make_run(); age(d)
    data = open(os.path.join(d, "ckpt_last.pt"), "rb").read()
    open(os.path.join(d, "ckpt_last.pt"), "wb").write(data[: len(data) // 2]); age(d)
    expect("PC8 half-pulled ckpt_last.pt under an intact SHA256SUMS.txt -> INCOMPLETE", False, art)
    art, d = make_run(); data = bytearray(open(os.path.join(d, "ckpt_last.pt"), "rb").read()); data[1234] ^= 1
    open(os.path.join(d, "ckpt_last.pt"), "wb").write(bytes(data)); age(d)
    expect("PC9 one flipped bit in ckpt_last.pt (same size) -> INCOMPLETE", False, art)
    art, d = make_run(); os.remove(os.path.join(d, "eval_M2b-orc_s0.npz")); age(d)
    expect("PC10 a listed file is absent -> INCOMPLETE", False, art)
    art, d = make_run(); open(os.path.join(d, "train_log.jsonl"), "wb").write(b""); write_sums(d, PC.expected_files("orc", 0)); age(d)
    expect("PC11 an empty file with a matching sha -> INCOMPLETE", False, art)
    art, d = make_run(done=False); age(d)
    expect("PC12 summary.json done: false (sums match) -> INCOMPLETE", False, art)
    art, d = make_run(); open(os.path.join(d, "argv.json"), "w").write("{truncated"); write_sums(d, PC.expected_files("orc", 0)); age(d)
    expect("PC13 argv.json not parseable (sums match) -> INCOMPLETE", False, art)
    # the wrong arm / seed directory must not satisfy another run's wait
    art, d = make_run("orc", 0); age(d)
    expect("PC14 the orc s0 pull does not satisfy orc0 s0", False, art, "orc0", 0)
    expect("PC15 ... nor orc s1", False, art, "orc", 1)
    art, d = make_run("orc0", 1); age(d)
    expect("PC16 orc0 s1 complete under its own name", True, art, "orc0", 1)
    # the CLI
    art, d = make_run(); age(d)
    rc = PC.main([art, "orc", "0"])
    RES.append(("PC17 CLI rc 0 on a complete run", rc == 0, f"rc {rc}"))
    rc = PC.main([art, "orc0", "0"])
    RES.append(("PC18 CLI rc 1 on an absent run", rc == 1, f"rc {rc}"))

    # ---------------------------------------------------------------- manifest
    N[0] += 1
    art = os.path.join(BASE, f"mart{N[0]}")
    out = os.path.join(BASE, "mout")
    runs = MF.parse_runs(MF.DEFAULT_RUNS)
    ck = {}
    for arm, seed in runs:
        d = os.path.join(art, PC.run_name(arm, seed))
        os.makedirs(d)
        lev = {"init_ckpt": {"sha256": ("b" if seed else "a") * 64}}
        for n in ("argv.json", "launch_gate_token.json", f"eval_{PC.eval_tag(arm, seed)}.json"):
            wjson(os.path.join(d, n), {"x": 1})
        wjson(os.path.join(d, "summary.json"), {"done": True, "levers": lev}); wjson(os.path.join(d, "selfcheck.json"), {"levers": lev})
        np_ = os.path.join(d, f"eval_{PC.eval_tag(arm, seed)}.npz"); open(np_, "wb").write(b"x")
        for lim in MF.LIMITS:
            t = MF.tag_of("", arm, seed, lim)
            os.makedirs(os.path.join(out, "ticks"), exist_ok=True)
            open(os.path.join(out, "ticks", f"ticks_{t}.npz"), "wb").write(b"x")
            os.makedirs(os.path.join(out, f"dump_{t}"), exist_ok=True)
            for k in range(3):
                open(os.path.join(out, f"dump_{t}", f"ep{k:04d}.npz"), "wb").write(b"x")
            wjson(os.path.join(out, f"t1v8f_{t}.json"), {"t1v8f": {"limit": lim, "ticks_heads": True, "tag": t, "ckpt": f"E:/x/{PC.run_name(arm, seed)}/ckpt_last.pt"}})
    m = MF.build(art, out, "", runs)
    bad = MF.verify(m, 3)
    RES.append(("MF0 positive control: a complete fixture verifies with no complaint", bad == [], "; ".join(bad)[:150]))
    RES.append(("MF1 schema: arms ORC / ORC0 x seeds '0' '1', no M2b arm, t1 + t1_logged + t0 + token + train_dir + ckpt_tail",
                sorted(m["arms"]) == ["ORC", "ORC0"] and all(sorted(m["arms"][a]) == ["0", "1"] for a in m["arms"])
                and all(set(e) == {"ckpt_tail", "train_dir", "token", "t1", "t1_logged", "t0"} for a in m["arms"].values() for e in a.values()), ""))
    RES.append(("MF2 tags follow the T1-HEADS naming", m["arms"]["ORC0"]["1"]["t1"]["tag"] == "ORC0_s1_m10_causal_heads" and m["arms"]["ORC"]["0"]["t1_logged"]["tag"] == "ORC_s0_m10_logged_heads", ""))
    e0 = m["arms"]["ORC"]["0"]

    def mutant(name, edit, undo):
        edit()
        bad = MF.verify(m, 3)
        undo()
        RES.append((name, bool(bad), "; ".join(bad)[:150]))

    p = e0["t1"]["ticks"]; os.rename(p, p + ".moved")
    mutant("MF3 a missing ticks file is caught", lambda: None, lambda: None)
    os.rename(p + ".moved", p)
    j = e0["t1"]["json"]; good_json = open(j, encoding="utf-8").read()
    mutant("MF4 a json whose limit is not causal is caught", lambda: wjson(j, {"t1v8f": {"limit": "logged", "ticks_heads": True, "tag": e0["t1"]["tag"], "ckpt": "x/M2b_s0_orc/ckpt_last.pt"}}), lambda: open(j, "w", encoding="utf-8").write(good_json))
    mutant("MF5 a json without ticks_heads is caught", lambda: wjson(j, {"t1v8f": {"limit": "causal", "ticks_heads": False, "tag": e0["t1"]["tag"], "ckpt": "x/M2b_s0_orc/ckpt_last.pt"}}), lambda: open(j, "w", encoding="utf-8").write(good_json))
    mutant("MF6 a rollout that scored ANOTHER run's checkpoint is caught", lambda: wjson(j, {"t1v8f": {"limit": "causal", "ticks_heads": True, "tag": e0["t1"]["tag"], "ckpt": "x/M2b_s0_orc0/ckpt_last.pt"}}), lambda: open(j, "w", encoding="utf-8").write(good_json))
    bad = MF.verify(m, 4)
    RES.append(("MF7 fewer episode dumps than required is caught", bool(bad), "; ".join(bad)[:100]))
    # the optional fresh-age T0 eval (ORCT_T0_FRESH=1): default leaves `t0` = box 2's standard eval; --t0-use fresh swaps it and keeps the standard one as t0_standard
    fdir = os.path.join(BASE, "t0fresh"); os.makedirs(fdir)
    for arm, seed in runs:
        for ext in ("json", "npz"):
            open(os.path.join(fdir, f"eval_{PC.eval_tag(arm, seed)}-fresh.{ext}"), "wb").write(b"x")
    mf = MF.build(art, out, "", runs, None, None, fdir, "standard")
    mu = MF.build(art, out, "", runs, None, None, fdir, "fresh")
    s0 = mf["arms"]["ORC"]["0"]; u0 = mu["arms"]["ORC"]["0"]
    RES.append(("MF8 default: t0 stays box 2's standard eval, t0_fresh is added beside it", s0["t0"]["json"].endswith("/eval_M2b-orc_s0.json") and s0["t0_fresh"]["json"].endswith("/eval_M2b-orc_s0-fresh.json") and "t0_standard" not in s0, ""))
    RES.append(("MF9 --t0-use fresh: t0 is the fresh eval, the standard one is kept as t0_standard", u0["t0"]["json"].endswith("-fresh.json") and u0["t0_standard"]["json"].endswith("/eval_M2b-orc_s0.json"), ""))
    mp = MF.build(art, out, "SMOKE_", runs, None, None, fdir, "fresh")
    RES.append(("MF9b the SMOKE_ prefix reaches the fresh-eval file names (the chain's t0tag)", mp["arms"]["ORC"]["0"]["t0"]["json"].endswith("/eval_SMOKE_M2b-orc_s0-fresh.json"), ""))
    RES.append(("MF10 both verify with no complaint when the files exist", MF.verify(mf, 3) == [] and MF.verify(mu, 3) == [], ""))
    os.rename(os.path.join(fdir, "eval_M2b-orc0_s1-fresh.npz"), os.path.join(fdir, "eval_M2b-orc0_s1-fresh.npz.moved"))
    RES.append(("MF11 a missing fresh eval file is caught", bool(MF.verify(mf, 3)), "; ".join(MF.verify(mf, 3))[:120]))
    os.rename(os.path.join(fdir, "eval_M2b-orc0_s1-fresh.npz.moved"), os.path.join(fdir, "eval_M2b-orc0_s1-fresh.npz"))
    # m2b sha
    r = MF.m2b_sha(art, runs, None)
    RES.append(("MS0 m2bsha: both seeds from the four runs' records", r == {0: "a" * 64, 1: "b" * 64}, str({k: v[:6] for k, v in r.items()})))
    wjson(os.path.join(art, "M2b_s1_orc0", "summary.json"), {"done": True, "levers": {"init_ckpt": {"sha256": "c" * 64}}})
    try:
        MF.m2b_sha(art, runs, None)
        RES.append(("MS1 disagreeing init records are refused", False, "no error"))
    except SystemExit as ex:
        RES.append(("MS1 disagreeing init records are refused", "disagree" in str(ex), str(ex)[:100]))
    wjson(os.path.join(art, "M2b_s1_orc0", "summary.json"), {"done": True, "levers": {"init_ckpt": {"sha256": "b" * 64}}})
    b2 = os.path.join(BASE, "b2.json"); wjson(b2, {"M2b_s0": "a" * 64, "M2b_s1": "d" * 64})
    try:
        MF.m2b_sha(art, runs, b2)
        RES.append(("MS2 the box-2 chain's measured sha that differs from the records is refused", False, "no error"))
    except SystemExit as ex:
        RES.append(("MS2 the box-2 chain's measured sha that differs from the records is refused", "measured" in str(ex), str(ex)[:100]))
    wjson(b2, {"M2b_s0": "a" * 64, "M2b_s1": "b" * 64})
    RES.append(("MS3 ... and agrees when it agrees", MF.m2b_sha(art, runs, b2) == {0: "a" * 64, 1: "b" * 64}, ""))

    w = max(len(n) for n, _, _ in RES)
    for n, ok, why in RES:
        print(f"{'PASS' if ok else 'FAIL'}  {n.ljust(w)}  | {why}")
    bad = [n for n, ok, _ in RES if not ok]
    print(f"\nSELFTEST (SMOKE fixtures under {BASE}): {len(RES) - len(bad)}/{len(RES)} as expected" + (f"; FAILED: {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
