#!/usr/bin/env python3
"""dagt1_selftest.py -- SMOKE-level self-test of the dev-box DAgger T1 chain (dag_t1_chain.sh, dagt1_pullcheck.py, dagt1_manifest.py).  Plain python, no pytest.
Fixtures are tiny synthetic files under <root>/run_<timestamp>/ ; NOTHING here is a DAgger result.  Every deliberate regression must be caught; every positive control must read as expected first.

    python dagt1_selftest.py [<root>] [--quick] [--no-mutants]       default root E:/Projects/TanitAD-artifacts/dagt1/bt      rc 0 = every check as expected

Sections
  A  dagt1_pullcheck.run   completeness / WRONG-run of a box-2 pull (the orct1 mutants, the exact 8-name set, the identity records, and decision O1: a SUMS that carries a box-2 fresh-age eval is REFUSED)
  B  dagt1_pullcheck.bank  completeness of a pulled bank directory as box-2's chain now publishes it (11 names: PROVENANCE.json, bank.json, shard provenance, SHARD NPZ, bank_stage.log), the cross-file
                           contradictions (WRONG), bank.json == the --dagger-bank-sha256 box 2's trainer gate declared (the pin), and the REAL generator's smoke bank read by the reader's own loader
  C  dagt1_manifest        ids (derive + verify against the banked train_sha256), m2bsha, orc0 reuse, tool-tree parity, manifest + verify
  D  the READER           the manifest this chain writes is loaded and READ by dag_read.py's own main() (imported read-only from dagread, test_dag_read's synthetic world and bank
                          generator as fixtures); the positive control must read as a clean outcome, every deliberate regression must be REFUSED by the reader as well
  E  the chain script     dag_t1_chain.sh --dry-run renders every command; the C: gate, the wrong tree, a bad switch each STOP
  F  mutants              textual mutants of dagt1_pullcheck.py / dagt1_manifest.py, each must turn at least one check RED
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
READ_DIR = "C:/Users/Admin/dagread"
sys.path.insert(0, READ_DIR)
import numpy as np  # noqa: E402

import dagt1_manifest as MF  # noqa: E402
import dagt1_pullcheck as PC  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
QUICK = "--quick" in sys.argv
NO_MUT = "--no-mutants" in sys.argv
ROOT = args[0] if args else "E:/Projects/TanitAD-artifacts/dagt1/bt"
STAMP = time.strftime("run_%m%d_%H%M%S")
BASE = os.path.join(ROOT, STAMP)
RES: list[tuple[str, bool, str]] = []
N = [0]
ARTS: list[str] = []          # the pulled-run fixtures of the current pc_tests() call (removed at its end: a 1 MiB-cluster exFAT charges every small file a megabyte)
BASH = r"C:\Program Files\Git\bin\bash.exe"
R = "E:/Projects/TanitAD-artifacts"


def rec(name, ok, why=""):
    RES.append((name, bool(ok), str(why)))


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def wjson(p, o):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(o, f)


def rjson(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ====================================================================================================================== A: the pull check
def argv_for(arm, seed, bank_sha="b" * 64):
    a = ["--arm", "M2b", "--seed", str(seed), "--out", f"D:\\tanitad\\runs\\v8f_mini_full\\{PC.run_name(arm, seed)}", "--init-ckpt", f"D:\\tanitad\\runs\\v8f_mini_full\\M2b_s{seed}\\ckpt_last.pt",
         "--steps", "3000", "--lr", "1e-4", "--warmup", "100", "--p-gt-start", "0.2", "--p-gt-end", "0.2", "--ego-perturb-p", "0"]
    if arm == "dag":
        a += ["--dagger-p", "0.1", "--dagger-bank", "D:\\x\\bank", "--dagger-bank-sha256", bank_sha, "--dagger-seed", str(seed)]
    else:
        a += ["--dagger-p", "0"]
    return a


def make_run(P, arm="dag", seed=0, done=True, init="a" * 64, fresh=False, summary_edit=None, argv_edit=None):
    """a complete pulled run under a fresh ART; returns (art, run dir)"""
    N[0] += 1
    art = os.path.join(BASE, f"art{N[0]}")
    ARTS.append(art)
    d = os.path.join(art, P.run_name(arm, seed))
    os.makedirs(d)
    names = P.base_files(arm, seed) + (P.box2_fresh_files(arm, seed) if fresh else [])
    argv = argv_for(arm, seed)
    if argv_edit:
        argv = argv_edit(argv)
    lev = {"init_ckpt": {"sha256": init}, "dagger_p": 0.1 if arm == "dag" else 0.0}
    for n in names:
        p = os.path.join(d, n)
        if n == "ckpt_last.pt":
            open(p, "wb").write(os.urandom(300_000))
        elif n == "summary.json":
            sm = {"done": done, "steps": 3000, "arm": "M2b", "seed": seed, "levers": lev}
            if summary_edit:
                summary_edit(sm)
            wjson(p, sm)
        elif n == "argv.json":
            wjson(p, {"argv": argv, "sha": "x"})
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
    p = os.path.join(d, "SHA256SUMS.txt")
    t = time.time() - 60
    os.utime(p, (t, t))


def pc_tests(P, tag=""):
    """-> list of (name, ok, why): every check of section A against the module P (the real one, or a mutant of it)"""
    out = []

    ARTS.clear()

    def expect(name, want, art, arm="dag", seed=0, min_age=P.MIN_AGE_S):
        st, why, s = P.check(art, arm, seed, min_age_s=min_age)
        good = (st == want) and (bool(s) == (want == "COMPLETE"))
        out.append((name, good, f"{st}: {why[:120]}"))

    names8 = lambda arm="dag", seed=0: P.base_files(arm, seed)  # noqa: E731
    art, d = make_run(P); age(d)
    expect("PC0 positive control: a complete, aged dag run is COMPLETE", "COMPLETE", art)
    st, why, s = P.check(art, "dag", 0)
    out.append(("PC0b the returned sha256 is the ckpt's", s == sha(open(os.path.join(d, "ckpt_last.pt"), "rb").read()), s[:16]))
    art0, d0 = make_run(P, "dag0", 1); age(d0)
    expect("PC0c positive control: a complete dag0 s1 run is COMPLETE under its own name", "COMPLETE", art0, "dag0", 1)
    art, d = make_run(P)
    expect("PC1 SHA256SUMS.txt modified < 3 s ago -> INCOMPLETE", "INCOMPLETE", art)
    st, why, _ = P.check(art, "dag", 0, min_age_s=-3600.0)
    out.append(("PC1b same run with min_age 0 -> COMPLETE (so PC1 failed ONLY on the age)", st == "COMPLETE", why[:80]))
    expect("PC2 no run directory -> INCOMPLETE", "INCOMPLETE", os.path.join(BASE, "nowhere"))
    art, d = make_run(P); os.rename(os.path.join(d, "SHA256SUMS.txt"), os.path.join(d, "SUMS.old"))
    expect("PC3 no SHA256SUMS.txt (ckpt etc. present) -> INCOMPLETE", "INCOMPLETE", art)
    art, d = make_run(P); write_sums(d, names8(), drop="launch_gate_token.json"); age(d)
    expect("PC4 SHA256SUMS.txt lists 7 of 8 -> INCOMPLETE", "INCOMPLETE", art)
    art, d = make_run(P); write_sums(d, names8(), extra="train.log"); age(d)
    expect("PC5 SHA256SUMS.txt lists a 9th, unexpected name -> INCOMPLETE", "INCOMPLETE", art)
    art, d = make_run(P); write_sums(d, names8(), newline=False); age(d)
    expect("PC6 SHA256SUMS.txt without final newline (partial last line) -> INCOMPLETE", "INCOMPLETE", art)
    art, d = make_run(P)
    t = open(os.path.join(d, "SHA256SUMS.txt")).read().rstrip("\n")
    open(os.path.join(d, "SHA256SUMS.txt"), "w", newline="\n").write(t[:-9] + "\n"); age(d)
    expect("PC7 last line cut inside its name -> INCOMPLETE", "INCOMPLETE", art)
    art, d = make_run(P)
    t = open(os.path.join(d, "SHA256SUMS.txt")).read().rstrip("\n")
    open(os.path.join(d, "SHA256SUMS.txt"), "w", newline="\n").write(t[:-60] + "\n"); age(d)
    expect("PC7b last line cut inside its hash -> INCOMPLETE", "INCOMPLETE", art)
    art, d = make_run(P); data = open(os.path.join(d, "ckpt_last.pt"), "rb").read()
    open(os.path.join(d, "ckpt_last.pt"), "wb").write(data[: len(data) // 2]); age(d)
    expect("PC8 half-pulled ckpt_last.pt under an intact SHA256SUMS.txt -> INCOMPLETE", "INCOMPLETE", art)
    art, d = make_run(P); data = bytearray(open(os.path.join(d, "ckpt_last.pt"), "rb").read()); data[1234] ^= 1
    open(os.path.join(d, "ckpt_last.pt"), "wb").write(bytes(data)); age(d)
    expect("PC9 one flipped bit in ckpt_last.pt (same size) -> INCOMPLETE", "INCOMPLETE", art)
    art, d = make_run(P); os.remove(os.path.join(d, "eval_M2b-dag_s0.npz")); age(d)
    expect("PC10 a listed file is absent -> INCOMPLETE", "INCOMPLETE", art)
    art, d = make_run(P); open(os.path.join(d, "train_log.jsonl"), "wb").write(b""); write_sums(d, names8()); age(d)
    expect("PC11 an empty file with a matching sha -> INCOMPLETE", "INCOMPLETE", art)
    art, d = make_run(P, done=False); age(d)
    expect("PC12 summary.json done: false (sums match) -> INCOMPLETE", "INCOMPLETE", art)
    art, d = make_run(P); open(os.path.join(d, "argv.json"), "w").write("{truncated"); write_sums(d, names8()); age(d)
    expect("PC13 argv.json not parseable (sums match) -> INCOMPLETE", "INCOMPLETE", art)
    # decision O1 (2026-10-26): box 2 pulls NO fresh-age eval (the CPU comparator runs on the dev box only). The ONLY accepted SUMS is the 8-name one; a SUMS that carries a fresh eval is REFUSED (WRONG: the chain STOPs, it never waits 40 h for it)
    art, d = make_run(P, fresh=True); age(d)
    expect("PC20 [O1] the 10-name SUMS (box 2 also pulled the fresh-age eval) -> WRONG (refused), although every listed file exists and re-hashes", "WRONG", art)
    st, why, _ = P.check(art, "dag", 0)
    out.append(("PC20b ... and the refusal NAMES the fresh eval file and decision O1", "eval_M2b-dag_s0-fresh.json" in why and "O1" in why, why[:110]))
    art, d = make_run(P, fresh=True); write_sums(d, names8() + P.box2_fresh_files("dag", 0)[:1]); age(d)
    expect("PC21 [O1] 9 names: the 8 plus ONLY the fresh JSON -> WRONG", "WRONG", art)
    art, d = make_run(P, "dag0", 1, fresh=True); age(d)
    expect("PC22 [O1] the 10-name SUMS of a dag0 s1 run -> WRONG", "WRONG", art, "dag0", 1)
    art, d = make_run(P, fresh=True); write_sums(d, [n for n in names8() if n != "train_log.jsonl"] + P.box2_fresh_files("dag", 0)); age(d)
    expect("PC23 [O1] [single fault] a fresh pair in place of train_log.jsonl (still 9 names) -> WRONG, not INCOMPLETE: the fresh entry decides, not the count", "WRONG", art)
    art, d = make_run(P, fresh=True); age(d)
    rc = P.main(["run", art, "dag", "0"])
    out.append(("PC24 CLI rc 3 on the 10-name SUMS", rc == 3, f"rc {rc}"))
    with contextlib.redirect_stderr(io.StringIO()) as es_:
        rc = P.main(["run", art, "dag", "0", "--b2-fresh", "1"])
    out.append(("PC25 the removed --b2-fresh switch is REFUSED by the CLI (rc 2, says it was removed), not silently ignored", rc == 2 and "removed" in es_.getvalue(), f"rc {rc}: {es_.getvalue()[:100]}"))
    art, d = make_run(P); age(d)
    out.append(("PC26 the 8 names are the whole accepted set: exactly 8, none of them a fresh-age eval", len(names8()) == 8 and not any("fresh" in n for n in names8()), str(len(names8()))))
    # WRONG: the files verify, the run is not the one asked for -> the chain must STOP, not wait
    art, d = make_run(P, "dag0", 0, argv_edit=lambda a: [x if x != "M2b_s0_dag0" else x for x in a] + []); age(d)
    shutil.copytree(d, os.path.join(art, "M2b_s0_dag")); age(os.path.join(art, "M2b_s0_dag"))
    expect("PC30 a dag directory that holds DAG-0's records (argv --out ..._dag0, --dagger-p 0) -> WRONG", "WRONG", art, "dag", 0)
    art, d = make_run(P, "dag", 0); age(d)
    shutil.copytree(d, os.path.join(art, "M2b_s0_dag0")); age(os.path.join(art, "M2b_s0_dag0"))
    expect("PC31 a dag0 directory that holds DAG's records (argv names the bank, --dagger-p 0.1) -> WRONG", "WRONG", art, "dag0", 0)
    art, d = make_run(P, "dag", 1); age(d)
    shutil.copytree(d, os.path.join(art, "M2b_s0_dag")); age(os.path.join(art, "M2b_s0_dag"))
    expect("PC32 seed 1's records in seed 0's directory -> WRONG", "WRONG", art, "dag", 0)
    art, d = make_run(P, summary_edit=lambda sm: sm.update(seed=1)); age(d)
    expect("PC33 summary.json says seed 1 in the seed-0 directory -> WRONG", "WRONG", art)
    art, d = make_run(P, "dag0", 0, argv_edit=lambda a: a + ["--dagger-bank", "D:\\x\\bank"]); age(d)
    expect("PC34 DAG-0 whose argv names --dagger-bank -> WRONG", "WRONG", art, "dag0", 0)
    art, d = make_run(P, argv_edit=lambda a: a[: a.index("--dagger-seed")]); age(d)
    expect("PC35 DAG whose argv lacks --dagger-seed -> WRONG", "WRONG", art)
    art, d = make_run(P, summary_edit=lambda sm: sm["levers"].update(dagger_p=0.0)); age(d)
    expect("PC36 summary.json built dagger_p 0.0 under an argv naming 0.1 -> WRONG", "WRONG", art)
    art, d = make_run(P, argv_edit=lambda a: [("D:\\tanitad\\runs\\v8f_mini_full\\M2b_s1\\ckpt_last.pt" if x.endswith("M2b_s0\\ckpt_last.pt") else x) for x in a]); age(d)
    expect("PC37 --init-ckpt of the other seed's M2b -> WRONG", "WRONG", art)
    art, d = make_run(P, argv_edit=lambda a: ["D:\\tanitad\\runs\\v8f_mini_full\\M2b_s0_dag0" if x.endswith("\\M2b_s0_dag") else x for x in a]); age(d)
    expect("PC37b [single fault] only argv --out names another run's directory (a copied directory): every other record is DAG s0's -> WRONG", "WRONG", art)
    art, d = make_run(P, "dag0", 0); data = open(os.path.join(d, "ckpt_last.pt"), "rb").read()
    open(os.path.join(d, "ckpt_last.pt"), "wb").write(data[: len(data) // 2]); os.remove(os.path.join(d, "argv.json")); age(d)
    expect("PC38 a half-pulled run is INCOMPLETE (not WRONG): identity is judged only after the files verify", "INCOMPLETE", art, "dag0", 0)
    expect("PC39 an unknown arm is INCOMPLETE", "INCOMPLETE", art, "orc", 0)
    art, d = make_run(P); age(d)
    rc = P.main(["run", art, "dag", "0"])
    out.append(("PC40 CLI rc 0 on a complete run", rc == 0, f"rc {rc}"))
    rc = P.main(["run", art, "dag0", "0"])
    out.append(("PC41 CLI rc 1 on an absent run", rc == 1, f"rc {rc}"))
    art, d = make_run(P, "dag0", 0); age(d)
    shutil.copytree(d, os.path.join(art, "M2b_s0_dag")); age(os.path.join(art, "M2b_s0_dag"))
    rc = P.main(["run", art, "dag", "0"])
    out.append(("PC42 CLI rc 3 on a WRONG run", rc == 3, f"rc {rc}"))
    for a_ in ARTS:
        shutil.rmtree(a_, ignore_errors=True)
    ARTS.clear()
    return [(tag + n, o, w) for n, o, w in out]


# ====================================================================================================================== B: the pulled bank directory (the box-2 chain's `publish_bank_one` output) + the trainer gate's declared bank sha256
BKS: list[str] = []           # the bank worlds of the current bk_tests() call (removed at its end)


def pin_line(bank_sha, seed=0):
    """what box 2's gatedag pins and the box-2 chain pulls to <ART>/M2b_s<S>_dag/gate/pin.txt: the TRAINING line the queue prints (python, trainer, then the argv)"""
    argv = argv_for("dag", seed, bank_sha=bank_sha)
    return '"D:\\tanitad\\venv\\Scripts\\python.exe" "D:\\tanitad\\repo\\stack\\scripts\\v8f_mini_train.py" ' + " ".join(f'"{x}"' if " " in x else x for x in argv) + "\n"


def write_pin(art, seed, bank_sha, text=None):
    p = os.path.join(art, PC.run_name("dag", seed), "gate", "pin.txt")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(text if text is not None else pin_line(bank_sha, seed))
    return p


def seal_bank(art, seed, n_shards, bank_sha, pin=True):
    """the box-2 chain's last steps for one seed: bank_stage.log, SHA256SUMS.txt over the 3 + 2N names (written LAST, aged), and the DAG run's pinned gate argv declaring bank.json's sha256"""
    d = os.path.join(art, f"bank_s{seed}")
    with open(os.path.join(d, "bank_stage.log"), "w", encoding="utf-8", newline="\n") as f:
        f.write("bank stage log (synthetic)\n")
    write_sums(d, PC.bank_flat(n_shards))
    age(d)
    if pin:
        write_pin(art, seed, bank_sha)


def bk_tests(P, TD, tag=""):
    """-> list of (name, ok, why): every check of section B against the module P (the real one, or a mutant of it).  Each test builds on a fresh SEALED bank world (the TD fixture's REAL-layout bank: PROVENANCE.json, bank.json,
    2 shard provenance, 2 shard npz, + bank_stage.log + the 7-name SHA256SUMS.txt + the gate's pin)."""
    out = []
    BKS.clear()

    def world(seed=0):
        N[0] += 1
        art = os.path.join(BASE, f"bk{N[0]}")
        BKS.append(art)
        B = TD.make_bank("a" * 64)
        TD.write_bank(B, os.path.join(art, f"bank_s{seed}"))
        seal_bank(art, seed, len(B["shards"]), B["man_sha256"])
        return art, B

    def bdir(art, seed=0):
        return os.path.join(art, f"bank_s{seed}")

    def reseal(art, B, seed=0):
        write_sums(bdir(art, seed), PC.bank_flat(len(B["shards"])))
        age(bdir(art, seed))

    def edit_json(p, fn):
        o = rjson(p)
        fn(o)
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(o, sort_keys=True, indent=1) + "\n")

    def edit_manifest(art, B, fn, fix_provenance=True, seal=True):
        """edit bank.json; optionally make PROVENANCE.bank_sha256 follow it and reseal, so ONLY the intended contradiction is left"""
        edit_json(os.path.join(bdir(art), "bank.json"), fn)
        msha = sha(open(os.path.join(bdir(art), "bank.json"), "rb").read())
        write_pin(art, 0, msha)                                  # the gate's declared sha follows the edited bank.json, so the declared-sha guard stays out of every test below it
        if fix_provenance:
            edit_json(os.path.join(bdir(art), "PROVENANCE.json"), lambda o: o.update(bank_sha256=msha))
        if seal:
            reseal(art, B)

    def expect(name, want, art, seed=0, why_has=None, exp=None, min_age=P.MIN_AGE_S):
        st, why, s_ = P.check_bank_status(art, seed, min_age, expect_shards=exp)
        good = st == want and (bool(s_) == (want == "COMPLETE")) and (why_has is None or why_has in why)
        out.append((name, good, f"{st}: {why[:150]}"))

    art, B = world()
    nm0 = B["man"]["shards"][0]["file"]
    st, why, s_ = P.check_bank_status(art, 0)
    out.append(("BK0 positive control: a complete SEALED bank (PROVENANCE.json, bank.json, 2 shard provenance, 2 shard npz, bank_stage.log; 7-name SHA256SUMS.txt; the gate's pin declares bank.json's sha256) is COMPLETE and returns bank.json's sha256",
                st == "COMPLETE" and s_ == B["man_sha256"], why[:120]))
    out.append(("BK0b ... the reason counts 7 files (3 + 2 x 2 shards)", "7 files" in why, why[:60]))
    ok, why, s2 = P.check_bank(art, 0)
    out.append(("BK0c check_bank (the bool wrapper the manifest verifier calls) agrees", ok and s2 == B["man_sha256"], why[:60]))
    out.append(("BK0e the queue's 4-shard bank is 11 names: 3 + 2 x 4 (PROVENANCE.json, bank.json, 4 shard provenance, 4 shard npz, bank_stage.log)", len(P.bank_flat(4)) == 11 and len(set(P.bank_flat(4))) == 11 and len(P.bank_flat(2)) == 7, str(len(P.bank_flat(4)))))
    art, B = world(1)
    expect("BK0d positive control, seed 1: its own bank directory and ITS OWN DAG run's pin", "COMPLETE", art, 1)

    # ---- THE HOLE: what the box-2 chain published before (small files only, no sums, no shard npz) is not a bank the reader can read
    art, B = world()
    for f in B["man"]["shards"]:
        os.remove(os.path.join(bdir(art), f["file"]))
    os.remove(os.path.join(bdir(art), "SHA256SUMS.txt"))
    expect("BK1 [the hole] the OLD box-2 publication (small files only: no shard npz, no SHA256SUMS.txt) -> INCOMPLETE", "INCOMPLETE", art, why_has="no SHA256SUMS.txt")
    small = [n for n in PC.bank_flat(2) if not n.endswith(".npz")]
    write_sums(bdir(art), small)
    age(bdir(art))
    expect("BK1b [the hole] a SHA256SUMS.txt over the small files ONLY (the npz left out) -> INCOMPLETE, names the missing shard npz and says the reader needs them (K-B3.1)", "INCOMPLETE", art, why_has="K-B3.1")
    st, why, _ = P.check_bank_status(art, 0)
    out.append(("BK1c ... and the missing list names shard_000of002.npz", nm0 in why, why[:140]))

    # ---- a missing / damaged shard under an intact sums file
    art, B = world(); os.remove(os.path.join(bdir(art), nm0))
    expect("BK2 a shard npz that is listed but absent (SHA256SUMS.txt intact) -> INCOMPLETE", "INCOMPLETE", art, why_has="listed but absent")
    art, B = world(); b_ = bytearray(open(os.path.join(bdir(art), nm0), "rb").read()); b_[len(b_) // 2] ^= 1; open(os.path.join(bdir(art), nm0), "wb").write(bytes(b_))
    expect("BK3 [shard sha mismatch] one flipped bit in a shard npz under an intact SHA256SUMS.txt -> INCOMPLETE (half-pulled, or replaced)", "INCOMPLETE", art, why_has="sha256")
    art, B = world(); open(os.path.join(bdir(art), nm0), "wb").write(b""); reseal(art, B)
    expect("BK4 an EMPTY shard npz whose (empty) sha256 is the one listed -> INCOMPLETE", "INCOMPLETE", art, why_has="is empty")
    art, B = world(); os.remove(os.path.join(bdir(art), B["prov"]["shard_provenance"][1]))
    expect("BK5 a shard provenance file listed but absent -> INCOMPLETE", "INCOMPLETE", art, why_has="listed but absent")
    art, B = world(); os.remove(os.path.join(bdir(art), "PROVENANCE.json"))
    expect("BK6 PROVENANCE.json (the bank report) absent -> INCOMPLETE", "INCOMPLETE", art, why_has="PROVENANCE.json is absent")
    art, B = world(); os.remove(os.path.join(bdir(art), "bank_stage.log"))
    expect("BK7 bank_stage.log listed but absent -> INCOMPLETE", "INCOMPLETE", art, why_has="listed but absent")
    art, B = world(); open(os.path.join(bdir(art), "bank.json"), "ab").write(b" ")
    expect("BK8 bank.json edited AFTER SHA256SUMS.txt was written (a stale seal) -> INCOMPLETE", "INCOMPLETE", art, why_has="sha256")

    # ---- single-fault contradictions between files that each verify against the sums: WRONG, no wait fixes them
    art, B = world(); edit_manifest(art, B, lambda o: o["shards"][0].update(file_sha256="0" * 64))
    expect("BK9 [single fault] [shard sha mismatch vs bank.json] bank.json records another file_sha256 for shard 0 (PROVENANCE.bank_sha256 and the sums follow): the npz hashes to the listed value, not bank.json's -> WRONG", "WRONG", art, why_has="not bank.json's file_sha256")
    art, B = world(); edit_manifest(art, B, lambda o: o.update(note="edited"), fix_provenance=False)
    expect("BK10 [single fault] [bank.json sha != the report's] bank.json differs from what PROVENANCE.bank_sha256 names, the sums describe the edited file -> WRONG", "WRONG", art, why_has="PROVENANCE.bank_sha256")
    art, B = world(); edit_json(os.path.join(bdir(art), B["prov"]["shard_provenance"][1]), lambda o: o.update(file_sha256="0" * 64)); reseal(art, B)
    expect("BK11 [single fault] a shard provenance records another file_sha256 than its npz -> WRONG", "WRONG", art, why_has="shard provenance")
    art, B = world(); edit_manifest(art, B, lambda o: o["shards"][1].update(file="other.npz"))
    expect("BK12 bank.json names a shard npz that is not the generator's (shard_KKKofNNN.npz) -> WRONG", "WRONG", art, why_has="bank.json shards name")
    art, B = world(); edit_json(os.path.join(bdir(art), "PROVENANCE.json"), lambda o: o.update(shard_provenance=o["shard_provenance"][:1])); reseal(art, B)
    expect("BK13 PROVENANCE.json names one shard provenance file of two -> WRONG", "WRONG", art, why_has="shard_provenance")
    art, B = world(); edit_json(os.path.join(bdir(art), "PROVENANCE.json"), lambda o: o.update(schema="v0")); reseal(art, B)
    expect("BK14 PROVENANCE.json of another schema (sums describe it) -> INCOMPLETE", "INCOMPLETE", art, why_has="schema")
    art, B = world()
    expect("BK15 [--expect-shards] the queue's four shards demanded of a two-shard bank -> WRONG", "WRONG", art, why_has="bankdag runs 4", exp=4)

    # ---- the SUMS file itself
    art, B = world(); write_sums(bdir(art), PC.bank_flat(2), extra="gen_shard_0.log"); age(bdir(art))
    expect("BK16 SHA256SUMS.txt lists an 8th, unexpected name -> INCOMPLETE (exact set)", "INCOMPLETE", art, why_has="unexpected ['gen_shard_0.log']")
    art, B = world(); write_sums(bdir(art), [n for n in PC.bank_flat(2) if n != "bank_stage.log"]); age(bdir(art))
    expect("BK17 SHA256SUMS.txt lists 6 of the 7 names (the stage log left out) -> INCOMPLETE", "INCOMPLETE", art, why_has="missing ['bank_stage.log']")
    art, B = world(); write_sums(bdir(art), PC.bank_flat(2), newline=False); age(bdir(art))
    expect("BK18 SHA256SUMS.txt without its final newline -> INCOMPLETE", "INCOMPLETE", art, why_has="newline")
    art, B = world(); t_ = os.path.join(bdir(art), "SHA256SUMS.txt"); first_ = open(t_, "rb").read().splitlines(True)[0]
    with open(t_, "ab") as f_:
        f_.write(first_)
    age(bdir(art))
    expect("BK19 SHA256SUMS.txt lists one name twice -> INCOMPLETE", "INCOMPLETE", art, why_has="twice")
    art, B = world(); os.utime(os.path.join(bdir(art), "SHA256SUMS.txt"), None)
    expect("BK20 SHA256SUMS.txt modified < 3 s ago (it may still be filling) -> INCOMPLETE", "INCOMPLETE", art, why_has="modified")
    st, why, _ = P.check_bank_status(art, 0, min_age_s=-3600.0)
    out.append(("BK20b same bank with min_age 0 -> COMPLETE (so BK20 failed ONLY on the age)", st == "COMPLETE", why[:80]))
    expect("BK21 a seed whose bank directory does not exist -> INCOMPLETE", "INCOMPLETE", os.path.join(BASE, "nowhere"))

    # ---- the bank is the bank the TRAINER GATE declared
    art, B = world(); write_pin(art, 0, "e" * 64)
    expect("BK22 [declared sha] box 2's pinned gate argv declares another --dagger-bank-sha256 than bank.json's -> WRONG, naming the pin", "WRONG", art, why_has="pin.txt")
    art, B = world(); os.remove(os.path.join(art, "M2b_s0_dag", "gate", "pin.txt"))
    expect("BK23 [declared sha] the pin is not pulled yet -> INCOMPLETE (the bank cannot be tied to the gate)", "INCOMPLETE", art, why_has="not pulled yet")
    art, B = world(); write_pin(art, 0, None, text=pin_line(B["man_sha256"]) .replace("--dagger-bank-sha256", "--zz-bank-sha256"))
    expect("BK24 [declared sha] the pinned argv names no --dagger-bank-sha256 -> WRONG", "WRONG", art, why_has="was not gated on a declared bank")
    art, B = world(); write_pin(art, 0, None, text=pin_line(B["man_sha256"]).rstrip("\n") + f" --dagger-bank-sha256 {'e' * 64}\n")
    expect("BK25 [declared sha] the pinned argv names --dagger-bank-sha256 twice -> WRONG", "WRONG", art, why_has="time(s) or with no single value")
    art, B = world(); write_pin(art, 0, None, text="not a training line\n")
    expect("BK26 [declared sha] the pin is not a training line -> WRONG", "WRONG", art, why_has="not a training line")
    art, B = world(); write_pin(art, 0, None, text=pin_line("xyz"))
    expect("BK27 [declared sha] the pinned value is not a sha256 -> WRONG", "WRONG", art, why_has="not a sha256")
    art, B = world(1); write_pin(art, 0, B["man_sha256"], text=pin_line(B["man_sha256"], 0)); os.remove(os.path.join(art, "M2b_s1_dag", "gate", "pin.txt"))
    expect("BK28 [declared sha] seed 1's bank is judged against seed 1's own DAG pin only: a seed-0 pin does not stand in for it -> INCOMPLETE", "INCOMPLETE", art, 1, why_has="not pulled yet")

    # ---- CLI
    art, B = world()
    with contextlib.redirect_stdout(io.StringIO()) as so_:
        rc_ok = P.main(["bank", art, "0"])
    out.append(("BK30 CLI rc 0 on a complete bank, printing `COMPLETE <bank.json sha256> ...`", rc_ok == 0 and so_.getvalue().startswith(f"COMPLETE {B['man_sha256']} "), so_.getvalue()[:90]))
    with contextlib.redirect_stdout(io.StringIO()):
        rc_inc = P.main(["bank", art, "1"])
        rc_wrong = P.main(["bank", art, "0", "--expect-shards", "4"])
    out.append(("BK31 CLI rc 1 on an absent bank directory, rc 3 on a WRONG one", rc_inc == 1 and rc_wrong == 3, f"rc {rc_inc} / {rc_wrong}"))
    for a_ in BKS:
        shutil.rmtree(a_, ignore_errors=True)
    BKS.clear()
    return [(tag + n, o, w) for n, o, w in out]


SMOKE_BANK = f"{R}/di/tree/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-26-dagger-build/smoke_kb45"


def smoke_bank_tests(P, TD, tag=""):
    """The REAL bank generator's 12-clip, 1-shard smoke bank (read-only source, copied) as a 'pulled' bank_s0: sealed the way box 2's chain seals a bank (stage log + the 5-name SHA256SUMS.txt + the gate's pin),
    judged by P, and then READ BY THE READER's own loader (dag_read.load_bank / g_bank_files / bank_clip_names, imported read-only).  MEASURED on real generator output, not on a synthetic bank."""
    out = []
    if not all(os.path.isfile(f"{SMOKE_BANK}/{n}") for n in ("PROVENANCE.json", "bank.json", "shard_000of001.npz", "shard_000of001.provenance.json")):
        return [(tag + "SB0 the real smoke bank files are present (read-only source)", False, SMOKE_BANK)]
    N[0] += 1
    art = os.path.join(BASE, f"smoke{N[0]}")
    d = os.path.join(art, "bank_s0")
    os.makedirs(d)
    for nm in ("PROVENANCE.json", "bank.json", "shard_000of001.npz", "shard_000of001.provenance.json"):
        shutil.copyfile(f"{SMOKE_BANK}/{nm}", os.path.join(d, nm))
    msha = sha(open(os.path.join(d, "bank.json"), "rb").read())
    seal_bank(art, 0, 1, msha)
    st, why, s_ = P.check_bank_status(art, 0, expect_shards=1)
    out.append(("SB0 [real generator output] the 12-clip smoke bank, copied and sealed as box 2's chain seals it (5 names: PROVENANCE.json, bank.json, 1 shard provenance, 1 shard npz, bank_stage.log), is COMPLETE", st == "COMPLETE" and s_ == msha, f"{st}: {why[:110]}"))
    st, why, _ = P.check_bank_status(art, 0, expect_shards=4)
    out.append(("SB1 ... and the same bank is WRONG against the queue's 4 shards", st == "WRONG" and "bankdag runs 4" in why, f"{st}: {why[:80]}"))
    rp, bf = os.path.join(d, "PROVENANCE.json"), os.path.join(d, "bank.json")
    try:
        B = TD.dag.load_bank(rp, bf)
        names = TD.dag.bank_clip_names(B)
        ok_f, det = TD.dag.g_bank_files(B)
        out.append(("SB2 [the reader] dag_read.load_bank reads the PULLED copies (report + bank_file) with the shard npz beside them: bank.json's sha256 is the identity the pull check returned, the shard files are the manifest's, and K-B3.1's clip names come from the npz",
                    B["man_sha256"] == s_ == msha and ok_f and len(names) > 0, f"g_bank_files {ok_f}: {str(det)[:100]}; {len(names)} clip names"))
        hdr = json.loads(bytes(np.load(os.path.join(d, "shard_000of001.npz"))["header_json"]).decode("utf-8"))
        out.append(("SB3 ... and those names are the ones the npz header_json carries (read independently)", names == sorted(set(hdr["clips"].values() if isinstance(hdr["clips"], dict) else hdr["clips"])), f"{len(names)} names"))
    except Exception as ex:                                                  # noqa: BLE001
        out.append(("SB2 [the reader] dag_read.load_bank reads the PULLED copies", False, f"{type(ex).__name__}: {str(ex)[:160]}"))
    os.rename(os.path.join(d, "shard_000of001.npz"), os.path.join(d, "shard_000of001.npz.x"))
    try:
        TD.dag.load_bank(rp, bf)
        out.append(("SB4 [the hole, reproduced on the reader] WITHOUT the shard npz (what the box-2 chain pulled before: the small files only) dag_read.load_bank REFUSES", False, "load_bank accepted a bank with no shard npz"))
    except TD.dag.Refusal as ex:
        out.append(("SB4 [the hole, reproduced on the reader] WITHOUT the shard npz (what the box-2 chain pulled before: the small files only) dag_read.load_bank REFUSES, naming the shard", "shard_000of001.npz" in str(ex), str(ex)[:110]))
    st, why, _ = P.check_bank_status(art, 0, expect_shards=1)
    out.append(("SB5 ... and the pull check calls that same directory INCOMPLETE (never COMPLETE), so the chain waits for the npz rather than handing the reader a bank it will refuse", st == "INCOMPLETE", f"{st}: {why[:90]}"))
    shutil.rmtree(art, ignore_errors=True)
    return [(tag + n, o, w) for n, o, w in out]


# ====================================================================================================================== the synthetic reader world, in THIS chain's layout
def load_reader_fixtures():
    import test_dag_read as TD
    return TD


def good_ckpt_bytes(i):
    return (b"synthetic DAgger-chain checkpoint %d " % i) + os.urandom(2000)


class World:
    """the whole layout the chain reads and writes, synthetic: ART (box-2 pull + banks), OUT (rollouts, fresh evals), ORC_ART / ORC_T1 (the banked O-RECOVER outputs ORC-0 is read from)"""

    def __init__(self, root, TD):
        self.TD, self.root = TD, root
        self.art, self.out, self.orc_art, self.orc_t1 = (os.path.join(root, x) for x in ("art", "out", "orc_art", "orc_t1"))
        self.world = TD.World(**TD.STRONG)
        self.fx = None

    def build(self):
        TD = self.TD

        class MyFx(TD.Fx):
            def build_run(self, arm, s, logged):
                return {}
        self.fx = MyFx(self.art, self.world, logged=True)        # writes the banks to <art>/bank_s<S>/ (the real generator's layout), the id lists and the synthetic M2b inits to <art>/
        W = self.world
        for s in (0, 1):
            for arm, reader_arm in (("dag", "DAG"), ("dag0", "DAG0"), ("orc0", "ORC0")):
                self.write_run(arm, reader_arm, s, W.s[s])
        self.sums()
        for s in (0, 1):                                                  # the box-2 chain's bank publication: bank_stage.log + the 7-name SHA256SUMS.txt (this fixture's banks have 2 shards) + the gate's pin declaring bank.json's sha256
            seal_bank(self.art, s, 2, self.fx.bank_sha[s])
        return self

    def write_run(self, arm, reader_arm, s, Ws):
        TD, fx = self.TD, self.fx
        base = self.orc_art if arm == "orc0" else self.art
        outd = self.orc_t1 if arm == "orc0" else self.out
        run = f"M2b_s{s}_{arm}"
        d = os.path.join(base, run)
        os.makedirs(d, exist_ok=True)
        argv = TD.dag_argv(reader_arm, s, self.art, fx.bank_sha[s])
        k = argv.index("--out")
        argv[k + 1] = f"D:\\tanitad\\runs\\v8f_mini_full\\{run}"
        sha_a = TD.dag.argv_sha16(argv)
        lev = {"init_ckpt": {"path": argv[argv.index("--init-ckpt") + 1], "sha256": fx.m2b_sha[s]}}
        tot = {}
        if reader_arm == "DAG":
            lev.update(dagger_p=0.1, dagger_bank_sha256=fx.bank_sha[s])
            tot = {"dagger_n": 1003, "dagger_eligible": 10000}
        elif reader_arm == "DAG0":
            lev.update(dagger_p=0.0)
            tot = {"dagger_n": 0, "dagger_eligible": 10000}
        wjson(os.path.join(d, "argv.json"), {"argv": argv, "sha": sha_a})
        wjson(os.path.join(d, "summary.json"), {"done": True, "steps": 3000, "arm": "M2b", "seed": s, "n_clips": 4512, "n_windows": TD.N_WIN, "train_sha256": TD.TRAIN_SHA, "excluded_sha256": "bb" * 32,
                                                "argv_sha": sha_a, "levers": lev, **tot, **({"train_window_list_sha256": TD.WIN_SHA} if reader_arm != "ORC0" else {})})
        wjson(os.path.join(d, "selfcheck.json"), {"declared_vs_built": "PASS", "argv_sha": sha_a, "levers": lev})
        if reader_arm != "ORC0":
            dsha = "cd" * 32 if reader_arm == "DAG0" else "ce" * 32
            wjson(os.path.join(d, "launch_gate_token.json"), {"kind": "launch-gate-token", "verdict": "PASS", "hmac_sha256": "ef" * 32,
                                                              "binding": {"profile": "v8f-mini", "commit": "ab" * 20, "argv_sha256": TD.dag.gate_argv_sha256(argv), "argv": argv, "data_sha256": dsha}})
        else:
            wjson(os.path.join(d, "launch_gate_token.json"), {"kind": "launch-gate-token", "verdict": "PASS"})
        ck = good_ckpt_bytes(hash((arm, s)) & 0xFFFF)
        open(os.path.join(d, "ckpt_last.pt"), "wb").write(ck)
        open(os.path.join(d, "train_log.jsonl"), "wb").write(b'{"step": 1}\n')
        ckpt_box2 = f"D:\\tanitad\\runs\\v8f_mini_full\\{run}\\ckpt_last.pt"
        tg = f"M2b-{arm}_s{s}"
        self.write_t0(os.path.join(d, f"eval_{tg}"), reader_arm, Ws, ckpt_box2, "cycle")
        fdir = os.path.join(self.orc_t1, "t0fresh") if arm == "orc0" else os.path.join(self.out, "t0fresh")
        self.write_t0(os.path.join(fdir, f"eval_{tg}-fresh"), reader_arm, Ws, ckpt_box2, "fresh")
        ckpt_here = f"{base}/{run}/ckpt_last.pt".replace("\\", "/")
        for lim in ("causal", "logged"):
            tag = f"{reader_arm}_s{s}_m10_{lim}_heads"
            self.write_t1(reader_arm, s, Ws, tag, lim, ckpt_here, outd)
            wjson(os.path.join(outd, f"done_{tag}.txt"), {"tag": tag, "limit": lim, "ckpt_sha256": sha(ck), "tool_digest": "c" * 64, "n_dumps": 2, "smoke": False})
        wjson(os.path.join(outd, f"done_t0f_{tg}-fresh.txt"), {"tag": f"{tg}-fresh", "ckpt_sha256": sha(ck), "smoke": False})

    def write_t0(self, stem, reader_arm, Ws, ckpt, ages):
        ade = Ws["t0_1"][Ws["order"]] if reader_arm == "DAG" else Ws["t0_0"]
        if ages == "fresh":
            ade = ade + (7.0 if reader_arm == "DAG" else 0.0)
        wjson(stem + ".json", {"ckpt": ckpt, "ages": ages, "raw": {}})
        np.savez(stem + ".npz", ade6=ade, eid=Ws["ci"], ckpt=np.zeros((len(ade), 3, 3)), lat_logits=np.zeros((len(ade), 5)), lon_logits=np.zeros((len(ade), 7)))

    def write_t1(self, reader_arm, s, Ws, tag, limit, ckpt_path, outd):
        TD = self.TD
        cl = {"DAG": Ws["cl1"], "ORC0": Ws["cl0"] + 0.02}.get(reader_arm, Ws["cl0"])
        o = Ws["order"] if reader_arm == "DAG" else np.arange(len(Ws["ci"]))
        cl, gg, flx = cl[o], Ws["g"][o], Ws["fl"][o]
        ws, ci = Ws["t0"][o], Ws["ci"][o]
        d = os.path.join(outd, f"dump_{tag}")
        os.makedirs(d, exist_ok=True)
        for c in range(2):
            m = (ci * 2 // (int(ci.max()) + 1)) == c
            np.savez(os.path.join(d, f"ep{c:04d}.npz"), g=gg[m], ws=ws[m], eid=ci[m], clip_index=ci[m], v0=np.zeros(int(m.sum())), cl=cl[m], ol=cl[m], fl=flx[m])
        K = TD.TREE_K()
        A, B = K.comps(cl, gg)["ade6_m"], K.comps(flx, gg)["ade6_m"]
        import taniteval.ci as ci_mod
        delta = ci_mod.paired_episode_cluster_bootstrap(A, B, ci, n_boot=TD.NB, seed=0)["delta"]
        hdr = {"tag": tag, "ckpt": ckpt_path, "limit": limit, "replan_steps": 10, "schedule": "periodic", "scene": "logged", "nav": "deadreckon", "kin_mode": "yaw", "ticks_heads": True, "n_windows": int(len(ci))}
        wjson(os.path.join(outd, f"t1v8f_{tag}.json"), {"t1v8f": hdr, "paired_decision_grade": {"paired_model_minus_floor_t1": {"ade_m": {"delta": delta}}}})
        os.makedirs(os.path.join(outd, "ticks"), exist_ok=True)
        np.savez(os.path.join(outd, "ticks", f"ticks_{tag}.npz"), x=np.zeros(1))

    def sums(self):
        for s in (0, 1):
            for arm in ("dag", "dag0", "orc0"):
                d = os.path.join(self.orc_art if arm == "orc0" else self.art, f"M2b_s{s}_{arm}")
                names = PC.base_files(arm, s) if arm != "orc0" else ["ckpt_last.pt", f"eval_M2b-orc0_s{s}.json", f"eval_M2b-orc0_s{s}.npz", "selfcheck.json", "summary.json", "argv.json", "train_log.jsonl", "launch_gate_token.json"]
                write_sums(d, names)
                age(d)

    def manifest(self):
        runs = MF.parse_runs(MF.DEFAULT_RUNS)
        return MF.build(self.art, self.out, "", runs, self.orc_art, self.orc_t1, self.art, os.path.join(self.out, "t0fresh"))

    def read(self, m, name="DAG_READ.json"):
        """-> (rc, stderr text, result dict|None): dag_read.main() on manifest ``m`` with the synthetic tick-file backend"""
        TD = self.TD
        mp = os.path.join(self.root, "runs.json")
        wjson(mp, m)
        outp = os.path.join(self.root, name)
        if os.path.exists(outp):
            os.remove(outp)
        argv = ["--manifest", mp, "--out", outp, "--tool-root", str(TD.TREE), "--t1hd-read", str(TD.T1HD), "--t1k-read", str(TD.T1K), "--eval-cache", "x", "--tracks", "x", "--eval-grid", "x",
                "--eval-join", "x", "--n-boot", str(TD.NB), "--n-windows", str(self.world.N), "--n-episodes", str(self.world.n_ep),
                "--m2b-ckpt", f"0={self.art}/m2b/M2b_s0/ckpt_last.pt", "--m2b-ckpt", f"1={self.art}/m2b/M2b_s1/ckpt_last.pt", "--banked-sha", f"0={self.fx.m2b_sha[0]}", "--banked-sha", f"1={self.fx.m2b_sha[1]}",
                "--prereg", str(TD.PREREG), "--work-dir", os.path.join(self.root, "work")]
        world = self.world

        class Adapt(TD.SynBackend):
            def _pack(self, entry, label, seed, light=False):
                pk = super()._pack(entry, label, seed, light)
                arm = entry["tag"].split("_s", 1)[0].lower()                  # DAG_s0_m10_causal_heads -> dag
                pk.meta["ckpt"] = f"/x/M2b_s{seed}_{arm}/ckpt_last.pt"
                return pk
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            rc = TD.dag.main(argv, backend_factory=lambda T, a: Adapt(T, world))
        res = rjson(outp) if os.path.exists(outp) else None
        return rc, err.getvalue(), res


# ====================================================================================================================== main
def main():
    os.makedirs(BASE)
    t_start = time.time()
    # ------------------------------------------------------------------------------------------------ A
    RES.extend(pc_tests(PC))

    # ------------------------------------------------------------------------------------------------ B: the pulled bank directory
    TD = load_reader_fixtures()
    RES.extend(bk_tests(PC, TD))
    RES.extend(smoke_bank_tests(PC, TD))

    # ------------------------------------------------------------------------------------------------ C: manifest module
    PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
    grid = f"{R}/di/tree/{PK}/raw/eval_grid_windows.json"
    rel = f"{R}/_s2build-copy-20260919/release/v72/s2_labels_v7.2_train.jsonl.gz"
    idsd = os.path.join(BASE, "ids")
    r = MF.make_ids(grid, rel, idsd)
    rec("ID0 positive control (REAL grid + REAL v7.2 release): 141 eval ids, 4,512 TRAIN ids, TRAIN list = the banked train_sha256 of M2b", r["n_eval"] == 141 and r["n_train"] == 4512 and r["train_sha256"] == MF.M2B_TRAIN_SHA256,
        f"train {r['train_sha256'][:12]}, excluded {r['excluded_sha256'][:12]}")
    tr, ev = rjson(os.path.join(idsd, "train_clip_ids.json")), rjson(os.path.join(idsd, "eval141_clip_ids.json"))
    rec("ID1 the written lists: TRAIN and eval disjoint, TRAIN hashes to the banked value from the FILE", not (set(tr) & set(ev)) and MF.train_list_sha256(tr) == MF.M2B_TRAIN_SHA256, f"{len(tr)} / {len(ev)}")
    try:
        MF.make_ids(grid, rel, idsd, want_train="0" * 64, dry=True)
        rec("ID2 a TRAIN list that does not hash to the banked train_sha256 is REFUSED", False, "accepted")
    except SystemExit as ex:
        rec("ID2 a TRAIN list that does not hash to the banked train_sha256 is REFUSED", "banked train_sha256" in str(ex), str(ex)[:120])
    import gzip
    short = os.path.join(BASE, "release_short.jsonl.gz")
    with gzip.open(rel, "rt", encoding="utf-8") as f, gzip.open(short, "wt", encoding="utf-8") as g:
        for i, line in enumerate(f):
            if i != 7:
                g.write(line)
    try:
        MF.make_ids(grid, short, idsd, dry=True)
        rec("ID3 a release missing one clip is REFUSED", False, "accepted")
    except SystemExit as ex:
        rec("ID3 a release missing one clip is REFUSED", "REFUSED" in str(ex), str(ex)[:120])
    rec("ID4 the banked train_sha256 equals what the two real M2b summaries recorded (MEASURED from C:/Users/Admin/box2_ship/logs/v8f_mini/runs)",
        all(rjson(f"C:/Users/Admin/box2_ship/logs/v8f_mini/runs/M2b_s{s}/summary.json").get("train_sha256") == MF.M2B_TRAIN_SHA256 for s in (0, 1)), "")

    # ORC-0 reuse, on the REAL outputs
    bad = MF.orc0_check(f"{R}/v8f-orecover", f"{R}/v8f-orecover-t1")
    rec("OR0 positive control (REAL): the banked ORC-0 outputs are present and bound to the banked checkpoints", bad == [], "; ".join(bad)[:150])
    # tool-tree parity: real + synthetic mutants
    if not QUICK:
        pr = MF.parity(f"{R}/di/tree", f"{R}/v8f-t1heads-run/tool_81aa3e36")
        rec("TP0 positive control (REAL): the DAgger integration tree's T1/T0 import closure is the ORC-0 tool tree's, plus one additive block", not pr["violations"] and pr["additive_only"] == [("stack/tanitad/data/v8f_mini_data.py", 181)] and not pr["new"],
            f"{pr['closure_files']} files, {len(pr['identical'])} identical, additive {pr['additive_only']}")
    mini_ref, mini_tree = os.path.join(BASE, "tp_ref"), os.path.join(BASE, "tp_tree")
    files = {"stack/scripts/v8f_t1_eval.py": "from tanitad.models import v8f_mini as M\nfrom tanitad.data import v8f_mini_data as D\n", "stack/scripts/v8f_mini_eval.py": "from tanitad.models import v8f_mini as M\n",
             "taniteval/tools/t1_eval.py": "x = 1\n", "stack/tanitad/__init__.py": "", "stack/tanitad/models/__init__.py": "", "stack/tanitad/data/__init__.py": "",
             "stack/tanitad/models/v8f_mini.py": "def f():\n    return 1\n", "stack/tanitad/data/v8f_mini_data.py": "a = 1\nb = 2\n", "stack/tanitad/data/unrelated.py": "z = 0\n"}
    for root_ in (mini_ref, mini_tree):
        for rel_, txt in files.items():
            p = os.path.join(root_, rel_)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "w", newline="\n").write(txt)
    open(os.path.join(mini_tree, "stack/tanitad/data/v8f_dagger.py"), "w").write("class DaggerSwap: pass\n")
    p = MF.parity(mini_tree, mini_ref)
    rec("TP1 identical closure; a new file outside the closure (v8f_dagger.py) is neither new-in-closure nor a violation", not p["violations"] and not p["new"] and p["closure_files"] >= 5, str(p)[:140])
    open(os.path.join(mini_tree, "stack/tanitad/data/v8f_mini_data.py"), "w", newline="\r\n").write("a = 1\nb = 2\n")
    p = MF.parity(mini_tree, mini_ref)
    rec("TP2 a CRLF-only difference is no difference (CR-normalised)", not p["violations"] and "stack/tanitad/data/v8f_mini_data.py" in p["identical"], str(p)[:120])
    open(os.path.join(mini_tree, "stack/tanitad/data/v8f_mini_data.py"), "w", newline="\n").write("a = 1\nb = 2\nc = 3\n")
    p = MF.parity(mini_tree, mini_ref)
    rec("TP3 an inserted line is ADDITIVE-ONLY, not a violation", not p["violations"] and p["additive_only"] == [("stack/tanitad/data/v8f_mini_data.py", 1)], str(p)[:120])
    open(os.path.join(mini_tree, "stack/tanitad/data/v8f_mini_data.py"), "w", newline="\n").write("a = 1\nb = 3\n")
    p = MF.parity(mini_tree, mini_ref)
    rec("TP4 a MODIFIED line in a module the tool imports is a VIOLATION", len(p["violations"]) == 1 and "v8f_mini_data.py" in p["violations"][0], str(p["violations"])[:120])
    open(os.path.join(mini_tree, "stack/tanitad/data/v8f_mini_data.py"), "w", newline="\n").write("a = 1\nb = 2\n")
    open(os.path.join(mini_tree, "stack/tanitad/data/unrelated.py"), "w").write("z = 99\n")
    p = MF.parity(mini_tree, mini_ref)
    rec("TP5 a modified module the tool does NOT import is no violation (the closure decides, not the directory)", not p["violations"], str(p["violations"])[:120])
    os.remove(os.path.join(mini_tree, "stack/tanitad/models/v8f_mini.py"))
    p = MF.parity(mini_tree, mini_ref)
    rec("TP6 a module of the reference's closure that the tree lost is a VIOLATION", any("v8f_mini.py" in v for v in p["violations"]), str(p["violations"])[:140])
    open(os.path.join(mini_tree, "stack/tanitad/models/v8f_mini.py"), "w", newline="\n").write("def f():\n    return 1\n")          # restore: the mutant harness below reuses this tree

    # ------------------------------------------------------------------------------------------------ D: the reader reads this chain's manifest
    WD = World(os.path.join(BASE, "world"), TD).build()
    m = WD.manifest()
    bad = MF.verify(m, 2)
    rec("MF0 positive control: the synthetic world in the chain's layout verifies with no complaint", bad == [], "; ".join(bad)[:200])
    rec("MF1 schema: seeds, arms DAG / DAG0 / ORC0 x '0' '1', bank '0' '1', eval141_clip_ids, train_clip_ids", sorted(m["arms"]) == ["DAG", "DAG0", "ORC0"] and all(sorted(m["arms"][a]) == ["0", "1"] for a in m["arms"])
        and sorted(m["bank"]) == ["0", "1"] and m["seeds"] == [0, 1] and "eval141_clip_ids" in m and "train_clip_ids" in m, "")
    rec("MF2 entry keys: DAG / DAG0 carry token + t0 + t0_fresh + t1 + t1_logged + train_dir + ckpt_tail; ORC0 has no token",
        all(set(e) == {"ckpt_tail", "train_dir", "token", "t1", "t1_logged", "t0", "t0_fresh"} for a in ("DAG", "DAG0") for e in m["arms"][a].values())
        and all(set(e) == {"ckpt_tail", "train_dir", "t1", "t1_logged", "t0", "t0_fresh"} for e in m["arms"]["ORC0"].values()), "")
    rec("MF3 tags follow the T1-HEADS naming; ORC-0 points at the banked ORC outputs, not at ART / OUT",
        m["arms"]["DAG0"]["1"]["t1"]["tag"] == "DAG0_s1_m10_causal_heads" and m["arms"]["DAG"]["0"]["t1_logged"]["tag"] == "DAG_s0_m10_logged_heads"
        and m["arms"]["ORC0"]["0"]["train_dir"].startswith(MF.fwd(WD.orc_art)) and m["arms"]["ORC0"]["0"]["t1"]["json"].startswith(MF.fwd(WD.orc_t1))
        and m["arms"]["DAG"]["0"]["t0_fresh"]["json"].startswith(MF.fwd(os.path.join(WD.out, "t0fresh"))), "")
    rec("MF4 box 2's standard eval is `t0`, the dev-box CPU fresh eval is `t0_fresh` (never the same file)",
        all(e["t0"]["json"] != e["t0_fresh"]["json"] and e["t0"]["json"].endswith(f"eval_M2b-{a.lower()}_s{s}.json") for a, per in m["arms"].items() for s, e in per.items() if a != "ORC0"), "")
    # mutate-and-undo helper for the manifest verifier
    ents = m["arms"]
    e_d, e_0 = ents["DAG"]["0"], ents["DAG0"]["0"]

    def mutant(name, edit, undo, needle=None):
        edit()
        try:
            bad_ = MF.verify(m, 2)
        finally:
            undo()
        rec(name, bool(bad_) and (needle is None or any(needle in b for b in bad_)), "; ".join(bad_)[:170])

    def jedit(path, fn):
        keep_ = open(path, "rb").read()
        o = rjson(path)
        fn(o)
        wjson(path, o)
        return lambda: open(path, "wb").write(keep_)
    st = {}

    def e_json(key, ent, val):
        def edit():
            st["u"] = jedit(ent[key]["json"], lambda o: o["t1v8f"].__setitem__("ckpt", val))
        return edit
    # --- WRONG CHECKPOINT FOR AN ARM
    mutant("MF10 [wrong checkpoint] DAG-0's causal rollout scored DAG's checkpoint -> refused", e_json("t1", e_0, f"{WD.art}/M2b_s0_dag/ckpt_last.pt"), lambda: st["u"](), "expects .../M2b_s0_dag0/ckpt_last.pt")
    mutant("MF11 [wrong checkpoint] DAG-0's logged rollout scored seed 1's DAG-0 checkpoint -> refused", e_json("t1_logged", e_0, f"{WD.art}/M2b_s1_dag0/ckpt_last.pt"), lambda: st["u"](), "expects .../M2b_s0_dag0/ckpt_last.pt")

    def ed12():
        st["k"] = e_0["ckpt_tail"]
        e_0["ckpt_tail"] = e_d["ckpt_tail"]
    mutant("MF12 [wrong checkpoint] DAG-0's manifest entry names DAG's ckpt_tail -> refused", ed12, lambda: e_0.__setitem__("ckpt_tail", st["k"]), "another arm")

    def ed13():
        p_ = os.path.join(os.path.dirname(e_0["t1"]["json"]), f"done_{e_0['t1']['tag']}.txt")
        st["u"] = jedit(p_, lambda o: o.__setitem__("ckpt_sha256", sha(b"another checkpoint")))
    mutant("MF13 [wrong checkpoint] a rollout's done record names a checkpoint that is not the pulled one -> refused", ed13, lambda: st["u"](), "done record names")

    def ed14():
        a_, b_ = os.path.join(e_d["train_dir"], "ckpt_last.pt"), os.path.join(e_0["train_dir"], "ckpt_last.pt")
        st["keep"] = open(b_, "rb").read()
        open(b_, "wb").write(open(a_, "rb").read())
        write_sums(e_0["train_dir"], PC.base_files("dag0", 0))
        for key in ("t1", "t1_logged"):
            pass
    mutant("MF14 [wrong checkpoint] DAG's checkpoint bytes sit in DAG-0's directory (same sha256 in two arms) -> refused", ed14,
           lambda: (open(os.path.join(e_0["train_dir"], "ckpt_last.pt"), "wb").write(st["keep"]), write_sums(e_0["train_dir"], PC.base_files("dag0", 0))), "same as")

    def ed15():
        st["u"] = jedit(e_0["t0"]["json"], lambda o: o.__setitem__("ckpt", "D:\\tanitad\\runs\\v8f_mini_full\\M2b_s0_dag\\ckpt_last.pt"))
    mutant("MF15 [wrong checkpoint] DAG-0's standard T0 eval scored DAG's checkpoint -> refused", ed15, lambda: st["u"](), "the eval scored")

    def ed16():
        st["u"] = jedit(e_d["t0_fresh"]["json"], lambda o: o.__setitem__("ckpt", "D:\\tanitad\\runs\\v8f_mini_full\\M2b_s1_dag\\ckpt_last.pt"))
    mutant("MF16 [wrong checkpoint] DAG's fresh T0 eval scored seed 1's checkpoint -> refused", ed16, lambda: st["u"](), "t0_fresh: the eval scored")

    def ed17():
        st["u"] = jedit(e_d["t0"]["json"], lambda o: o.__setitem__("ages", "fresh"))
    mutant("MF17 the standard T0 eval is a fresh-age eval (it would feed R-T0 and K4 the same file) -> refused", ed17, lambda: st["u"](), "ages")
    # --- DAG0 POINTING AT DAG's RUN
    def ed20():
        st["k"] = dict(e_0)
        for k_ in ("train_dir", "token", "ckpt_tail"):
            e_0[k_] = e_d[k_]
    mutant("MF20 [DAG0 -> DAG's run] DAG-0's train_dir / token / ckpt_tail are DAG's -> refused", ed20, lambda: e_0.update(st["k"]), "another arm")

    def ed21():
        st["k"] = dict(e_0["t1"])
        e_0["t1"].update(e_d["t1"])
    mutant("MF21 [DAG0 -> DAG's run] DAG-0's decisive rollout files are DAG's -> refused", ed21, lambda: e_0["t1"].update(st["k"]), "share one run")

    def ed22():
        st["k"] = dict(e_0["t0"])
        e_0["t0"].update(e_d["t0"])
    mutant("MF22 [DAG0 -> DAG's run] DAG-0's standard T0 eval is DAG's -> refused", ed22, lambda: e_0["t0"].update(st["k"]), "share one run")
    # --- MISSING BANK REPORT, MISSING FRESH EVAL
    bp = m["bank"]["0"]["report"]
    mutant("MF30 [missing bank report] bank_s0/PROVENANCE.json absent -> refused", lambda: os.rename(bp, bp + ".x"), lambda: os.rename(bp + ".x", bp), "bank s0")
    # --- THE BANK FILES the reader needs beside the report (the hole): bank_file, the shard npz, the sha256 the gate declared
    b0, b1 = m["bank"]["0"], m["bank"]["1"]
    kb0 = dict(b0)
    bd0 = os.path.dirname(bp)
    npz0 = os.path.join(bd0, "shard_000of002.npz")
    pin0 = os.path.join(WD.art, "M2b_s0_dag", "gate", "pin.txt")
    rec("MF31a the manifest's bank entries carry report AND bank_file, both in the pulled bank_s<S>/ directory (the build writes them; nothing else)", all(sorted(v) == ["bank_file", "report"] and v["bank_file"] == v["report"][:-len("PROVENANCE.json")] + "bank.json" for v in m["bank"].values()), str(b0)[:140])
    pin_keep = open(pin0, "rb").read()
    bank_cases = [      # (label, edit, undo, the needle the refusal must carry); reused by the manifest-mutant harness below
        ("MF31 [bank_file elsewhere] bank_file is a bank.json that is not the pulled one beside the report -> refused", lambda: b0.__setitem__("bank_file", b0["bank_file"] + ".elsewhere"), lambda: b0.update(kb0), "is not the pulled"),
        ("MF32 [bank_file missing] the manifest's bank entry carries no bank_file -> refused", lambda: b0.pop("bank_file"), lambda: b0.update(kb0), "no bank_file"),
        ("MF33 [seed-swapped bank] seed 0's entry points at seed 1's bank directory (report AND bank_file: only the seed test sees it) -> refused", lambda: b0.update(report=b1["report"], bank_file=b1["bank_file"]), lambda: b0.update(kb0), "another seed's bank"),
        ("MF34 [shard npz missing] a shard npz of the pulled bank absent (what the box-2 chain pulled before this change: the small files only) -> refused", lambda: os.rename(npz0, npz0 + ".x"), lambda: os.rename(npz0 + ".x", npz0), "listed but absent"),
        ("MF35 [declared sha] box 2's gate declared another --dagger-bank-sha256 than the pulled bank.json hashes to -> refused", lambda: write_pin(WD.art, 0, "e" * 64), lambda: open(pin0, "wb").write(pin_keep), "not the bank that run was gated on"),
        ("MF36 [declared sha] the DAG s0 pin is not pulled -> refused", lambda: os.rename(pin0, pin0 + ".x"), lambda: os.rename(pin0 + ".x", pin0), "not pulled yet"),
    ]
    for lab_, ed_, un_, nd_ in bank_cases:
        mutant(lab_, ed_, un_, nd_)
    fz = e_d["t0_fresh"]["npz"]
    mutant("MF40 [fresh eval missing] DAG s0's fresh T0 eval npz absent -> refused", lambda: os.rename(fz, fz + ".x"), lambda: os.rename(fz + ".x", fz), "t0_fresh npz")
    fj = e_0["t0_fresh"]["json"]
    mutant("MF41 [fresh eval missing] DAG-0 s0's fresh T0 eval json absent -> refused", lambda: os.rename(fj, fj + ".x"), lambda: os.rename(fj + ".x", fj), "t0_fresh json")
    st["f"] = e_d["t0_fresh"]
    mutant("MF42 [fresh eval missing] the manifest carries no t0_fresh for DAG s0 -> refused", lambda: e_d.pop("t0_fresh"), lambda: e_d.__setitem__("t0_fresh", st["f"]), "t0_fresh: absent")
    bad_ = MF.verify(m, 3)
    rec("MF43 fewer episode dumps than required (3 asked, 2 present) is refused", any("ep*.npz" in b for b in bad_), "; ".join(bad_)[:120])
    ip = m["train_clip_ids"]
    mutant("MF44 the TRAIN id list absent -> refused", lambda: os.rename(ip, ip + ".x"), lambda: os.rename(ip + ".x", ip), "train_clip_ids")
    # --- single-fault regressions: ONE guard each (the MF10..MF22 faults trip several guards at once, so removing any one of them is invisible there)
    def ed50():
        st["k"] = e_0["ckpt_tail"]
        e_0["ckpt_tail"] = "M2b_s0_dag0z/ckpt_last.pt"
        st["u50"] = [jedit(e_0[k_]["json"], lambda o: o["t1v8f"].__setitem__("ckpt", f"{WD.art}/M2b_s0_dag0z/ckpt_last.pt")) for k_ in ("t1", "t1_logged")]
        st["u50"] += [jedit(e_0[k_]["json"], lambda o: o.__setitem__("ckpt", "D:\\tanitad\\runs\\v8f_mini_full\\M2b_s0_dag0z\\ckpt_last.pt")) for k_ in ("t0", "t0_fresh")]

    def un50():
        e_0["ckpt_tail"] = st["k"]
        for u_ in st["u50"]:
            u_()
    mutant("MF50 [single fault] DAG-0's entry is internally consistent but for ANOTHER run name (a stale directory's ckpt_tail, rollouts and evals agree) -> refused by the identity test alone", ed50, un50, "is not M2b_s0_dag0/ckpt_last.pt")

    def ed51():
        st["k"] = e_0["train_dir"]
        st["nd"] = st["k"].rstrip("/\\") + "z"
        shutil.copytree(st["k"], st["nd"])
        e_0["train_dir"] = st["nd"]

    def un51():
        e_0["train_dir"] = st["k"]
        shutil.rmtree(st["nd"], ignore_errors=True)
    mutant("MF51 [single fault] DAG-0's train_dir is a complete copy under another run name (nothing else differs) -> refused by the train_dir identity test alone", ed51, un51, "train_dir")

    def ed52():
        st["k"] = e_0["t1"]["dump_dir"]
        e_0["t1"]["dump_dir"] = e_d["t1"]["dump_dir"]
    mutant("MF52 [single fault] DAG-0's decisive dump directory IS DAG's (the 141 episode files the reader scores) -> refused by the shared-path test alone", ed52, lambda: e_0["t1"].__setitem__("dump_dir", st["k"]), "also the path of")

    def ed53():
        b_ = os.path.join(e_0["train_dir"], "ckpt_last.pt")
        st["keep"] = open(b_, "rb").read()
        new_ = open(os.path.join(e_d["train_dir"], "ckpt_last.pt"), "rb").read()
        open(b_, "wb").write(new_)
        write_sums(e_0["train_dir"], PC.base_files("dag0", 0))
        st["u53"] = [jedit(os.path.join(os.path.dirname(e_0[k_]["json"]), f"done_{e_0[k_]['tag']}.txt"), lambda o: o.__setitem__("ckpt_sha256", sha(new_))) for k_ in ("t1", "t1_logged")]

    def un53():
        open(os.path.join(e_0["train_dir"], "ckpt_last.pt"), "wb").write(st["keep"])
        write_sums(e_0["train_dir"], PC.base_files("dag0", 0))
        for u_ in st["u53"]:
            u_()
    mutant("MF53 [single fault] DAG's checkpoint bytes in DAG-0's directory WITH the done records re-bound to them (the sha256 compare passes; only 'four distinct checkpoints' fails) -> refused", ed53, un53, "same as")
    # --- m2bsha
    runs = MF.parse_runs(MF.DEFAULT_RUNS)
    r2 = MF.m2b_sha(WD.art, runs, WD.orc_art, None, require_banked=False)
    rec("MS0 m2bsha: both seeds from DAG, DAG-0 and ORC-0 records, all agree", r2 == {0: WD.fx.m2b_sha[0], 1: WD.fx.m2b_sha[1]}, str({k: v[:6] for k, v in r2.items()}))
    try:
        MF.m2b_sha(WD.art, runs, WD.orc_art, None, require_banked=True)
        rec("MS1 init shas that are not the banked M2b checkpoints are refused", False, "no error")
    except SystemExit as ex:
        rec("MS1 init shas that are not the banked M2b checkpoints are refused", "banked" in str(ex), str(ex)[:100])
    undo_ = jedit(os.path.join(WD.orc_art, "M2b_s1_orc0", "summary.json"), lambda o: o["levers"]["init_ckpt"].__setitem__("sha256", "c" * 64))
    try:
        MF.m2b_sha(WD.art, runs, WD.orc_art, None, require_banked=False)
        rec("MS2 an ORC-0 record that names another init than DAG's is refused (the replicate floor needs one init)", False, "no error")
    except SystemExit as ex:
        rec("MS2 an ORC-0 record that names another init than DAG's is refused (the replicate floor needs one init)", "disagree" in str(ex), str(ex)[:100])
    undo_()
    b2 = os.path.join(BASE, "b2.json")
    wjson(b2, {"M2b_s0": WD.fx.m2b_sha[0], "M2b_s1": "d" * 64})
    try:
        MF.m2b_sha(WD.art, runs, WD.orc_art, b2, require_banked=False)
        rec("MS3 the box-2 chain's measured sha that differs from the records is refused", False, "no error")
    except SystemExit as ex:
        rec("MS3 the box-2 chain's measured sha that differs from the records is refused", "measured" in str(ex), str(ex)[:100])
    # ORC-0 reuse on the synthetic world: mutate
    bad = MF.orc0_check(WD.orc_art, WD.orc_t1, n_dump_min=2)
    rec("OR1 positive control on the synthetic ORC-0 world", bad == [], "; ".join(bad)[:120])
    dn = os.path.join(WD.orc_t1, "done_ORC0_s0_m10_logged_heads.txt")
    undo_ = jedit(dn, lambda o: o.__setitem__("ckpt_sha256", "e" * 64))
    bad = MF.orc0_check(WD.orc_art, WD.orc_t1, n_dump_min=2)
    undo_()
    rec("OR2 an ORC-0 done record bound to another checkpoint than the banked pull is refused", any("names checkpoint" in b for b in bad), "; ".join(bad)[:120])
    fp = os.path.join(WD.orc_t1, "t0fresh", "eval_M2b-orc0_s1-fresh.npz")
    os.rename(fp, fp + ".x")
    bad = MF.orc0_check(WD.orc_art, WD.orc_t1, n_dump_min=2)
    os.rename(fp + ".x", fp)
    rec("OR3 a missing ORC-0 fresh eval is refused", any("missing or empty" in b for b in bad), "; ".join(bad)[:120])
    undo_ = jedit(os.path.join(WD.orc_t1, "done_t0f_M2b-orc0_s0-fresh.txt"), lambda o: o.__setitem__("smoke", True))
    bad = MF.orc0_check(WD.orc_art, WD.orc_t1, n_dump_min=2)
    undo_()
    rec("OR4 an ORC-0 record that is a SMOKE record is refused", any("SMOKE" in b for b in bad), "; ".join(bad)[:120])

    # ------------------------------------------------------------------------------------------------ D2: the reader itself
    t_rd = time.time()
    rc, err, res = WD.read(m)
    ctl = (res or {}).get("controls_PASS") or {}
    nfail = [k for k, v in ctl.items() if not v]
    rec("RD0 the READER (dag_read.main, imported read-only) reads this chain's manifest: rc 0, an outcome row, every control PASS", rc == 0 and res is not None and not nfail,
        f"rc {rc}; OUTCOME {((res or {}).get('OUTCOME') or {}).get('OUTCOME')}; {len(ctl)} controls, failed {nfail[:6]}; {err[-300:]}")
    keys_needed = {"ARGV:DAG_s0", "ARGV:DAG0_s0", "GATE-TOKEN:DAG_s0", "BANK-DECLARED:DAG_s1", "K-B3.2:s0", "TRAIN-LIST:s1", "PAIR:s0", "ORC0-REPLICATE:s1", "T0-CKPT:DAG0_s1", "T0F-CKPT:ORC0_s0", "T0-AGES:DAG_s0",
                   "T1-CONFIG:DAG_s0", "T1-CONFIG-logged:DAG0_s1", "K4:ORC0_s1", "DUMP-WINDOWS:DAG_s0", "SAME-WINDOWS:ORC0-vs-DAG0:s1", "BANK-FILES:s0", "K-B3.1:s1", "INIT-CKPT:ORC0_s0"}
    rec("RD1 the controls that bind this chain's files were all evaluated and PASS (ARGV, GATE-TOKEN, BANK-DECLARED, K-B3.2, TRAIN-LIST, PAIR, ORC0-REPLICATE, T0-CKPT, T0F-CKPT, T0-AGES, T1-CONFIG x2, K4, DUMP-WINDOWS, SAME-WINDOWS, BANK-FILES, K-B3.1, INIT-CKPT)",
        keys_needed <= set(ctl) and all(ctl[k] for k in keys_needed), f"missing {sorted(keys_needed - set(ctl))}")

    def refused_by_reader(name, edit, undo, needle):
        edit()
        try:
            m2 = json.loads(json.dumps(m))
            rc_, err_, res_ = WD.read(m2)
        finally:
            undo()
        rec(name, rc_ == 2 and res_ is None and needle in err_, f"rc {rc_}, written {res_ is not None}; {needle!r} in stderr: {needle in err_}")
    refused_by_reader("RD10 [wrong checkpoint] the reader REFUSES DAG-0's rollout that scored DAG's checkpoint (T1-CONFIG)", e_json("t1", e_0, f"{WD.art}/M2b_s0_dag/ckpt_last.pt"), lambda: st["u"](), "T1-CONFIG:DAG0_s0")

    def read_with(name, mut, needle, undo=lambda: None):
        m2 = json.loads(json.dumps(m))
        mut(m2)
        rc_, err_, res_ = WD.read(m2)
        undo()
        rec(name, rc_ == 2 and res_ is None and needle in err_, f"rc {rc_}, written {res_ is not None}; {needle!r} in stderr: {needle in err_}")
    read_with("RD11 [DAG0 -> DAG's run] the reader REFUSES a manifest whose DAG-0 slot is DAG's training run (ARGV / PAIR)", lambda m2: m2["arms"]["DAG0"]["0"].update(train_dir=e_d["train_dir"], token=e_d["token"]), "ARGV:DAG0_s0")
    read_with("RD12 [missing bank report] the reader REFUSES a manifest whose seed-0 bank report is absent", lambda m2: m2["bank"]["0"].update(report=m2["bank"]["0"]["report"] + ".gone"), "bank seed 0 report")
    read_with("RD13 [fresh eval missing] the reader REFUSES a manifest whose DAG-0 s1 fresh T0 eval is absent", lambda m2: m2["arms"]["DAG0"]["1"]["t0_fresh"].update(json=m2["arms"]["DAG0"]["1"]["t0_fresh"]["json"] + ".gone"), "fresh T0 eval json")
    read_with("RD14 [fresh eval missing] the reader REFUSES a manifest without t0_fresh", lambda m2: m2["arms"]["DAG"]["0"].pop("t0_fresh"), "t0_fresh")
    read_with("RD15 the reader REFUSES a manifest without the ORC-0 replicate (RF has no floor)", lambda m2: m2["arms"].pop("ORC0"), "ORC0")
    read_with("RD16 the reader REFUSES a manifest without the TRAIN id list", lambda m2: m2.pop("train_clip_ids"), "train_clip_ids")
    read_with("RD17 [wrong checkpoint] the reader REFUSES a standard T0 eval that scored another arm's checkpoint (T0-CKPT)", lambda m2: m2["arms"]["DAG0"]["0"]["t0"].update(json=e_d["t0"]["json"], npz=e_d["t0"]["npz"]), "T0-CKPT:DAG0_s0")
    read_with("RD18 [bank_file absent] the reader REFUSES a manifest whose bank_file names a bank.json that is not there", lambda m2: m2["bank"]["0"].update(bank_file=m2["bank"]["0"]["bank_file"] + ".gone"), "the bank manifest (bank.json)")
    sh0 = os.path.join(os.path.dirname(m["bank"]["0"]["report"]), "shard_000of002.npz")
    read_with("RD19 [shard npz missing] the reader REFUSES a bank whose shard npz is not beside the report (K-B3.1 reads the clip names from it)", lambda m2: os.rename(sh0, sh0 + ".x"), "shard shard_000of002.npz", lambda: os.rename(sh0 + ".x", sh0))
    old_layout = [os.path.join(os.path.dirname(m["bank"]["1"]["report"]), f"shard_{i:03d}of002.npz") for i in (0, 1)]
    read_with("RD20 [the old layout] the reader REFUSES seed 1's bank pulled the way the box-2 chain used to pull it (every small file, NO shard npz)", lambda m2: [os.rename(p_, p_ + ".x") for p_ in old_layout], "bank seed 1 report: shard", lambda: [os.rename(p_ + ".x", p_) for p_ in old_layout])
    WD_t_rd = time.time() - t_rd

    # ------------------------------------------------------------------------------------------------ E: the chain script
    def chain(env_extra, tree=None, timeout=420):
        env = dict(os.environ)
        env.update(env_extra)
        cmd = f"bash {HERE.replace(chr(92), '/')}/dag_t1_chain.sh --dry-run < /dev/null"
        t = time.time()
        p = subprocess.run([BASH, "-lc", cmd], env=env, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")
        return p.returncode, p.stdout + p.stderr, time.time() - t
    rc, out, dt = chain({"DAGT1_TEST_CFREE_MB": "1500"})
    rec("CH1 [C: < 2 GB] the dry run STOPs when C: has 1500 MB free (PREREG section 8: gated on C: >= 2 GB)", rc == 1 and "ZZDAGT1-STOP C: has 1500 MB free (< 2048 MB)" in out and "ZZDAGT1-DRY-OK" not in out, out.strip().splitlines()[-1][:150] if out.strip() else "")
    rc, out, dt = chain({"DAGT1_TEST_CFREE_MB": "2047"})
    rec("CH2 [C: boundary] 2047 MB STOPs, ", rc == 1 and "ZZDAGT1-STOP" in out, f"rc {rc}")
    rc, out, dt = chain({"DAGT1_TEST_CFREE_MB": "2048", "DAGT1_TREE": f"{R}/rt4/tree"})
    rec("CH3 [wrong tree] rt4/tree (no DAgger trainer) is REFUSED, naming the missing flag", rc == 1 and "ZZDAGT1-STOP" in out and "is not the DAgger tip" in out, out.strip().splitlines()[-1][:170] if out.strip() else "")
    for v_ in ("1", "2"):
        rc, out, dt = chain({"DAGT1_TEST_CFREE_MB": "2048", "DAGT1_B2_FRESH": v_})
        rec(f"CH4 [O1] DAGT1_B2_FRESH={v_} (the removed 'expect a box-2 fresh eval' switch) STOPs the chain, saying it was REMOVED and why, and never reaches the plan", rc == 1 and "ZZDAGT1-STOP" in out and "REMOVED" in out and "O1" in out and "ZZDAGT1-DRY-OK" not in out, out.strip().splitlines()[-1][:150] if out.strip() else "")
    rc, out, dt = chain({"DAGT1_TEST_CFREE_MB": "3000", "DAGT1_SMOKE": "0", "DAGT1_TREE": f"{R}/dagt1/nonexistent_tree"})
    rec("CH5 a tree that does not exist is refused", rc == 1 and "ZZDAGT1-STOP" in out, "")
    chain_src = open(os.path.join(HERE, "dag_t1_chain.sh"), encoding="utf-8").read()
    m_in = re.search(r"^SF=\$\{DAGT1_INPUTS:-([^}]+)\}", chain_src, re.M)
    rec("CH6 [inputs on E:] the chain's DEFAULT for the T1 inputs (DAGT1_INPUTS, overridable) is E:/Projects/TanitAD-artifacts/sfresh_inputs, and no C: / session-scratchpad path is left in the script",
        bool(m_in) and m_in.group(1) == "/e/Projects/TanitAD-artifacts/sfresh_inputs" and "scratchpad" not in chain_src and "AppData" not in chain_src, m_in.group(1)[:90] if m_in else "no SF= line")
    sf_dir = f"{R}/sfresh_inputs"
    n_clips = len(os.listdir(f"{sf_dir}/cache/eval/clips")) if os.path.isdir(f"{sf_dir}/cache/eval/clips") else -1
    rec("CH6b the default inputs directory exists on E: with the tracks, the v9 / p10 inputs and the 141-clip eval cache (the Master Mind's copy; the chain md5-asserts them again at start)",
        os.path.isfile(f"{sf_dir}/inputs/native_tracks.pt") and os.path.isdir(f"{sf_dir}/inputs/v9") and os.path.isfile(f"{sf_dir}/inputs/p10/p10_prior.pt") and n_clips == 141 and os.path.splitdrive(sf_dir)[0].upper() == "E:", f"{n_clips} clips")
    if not QUICK:
        rc, out, dt = chain({"DAGT1_TEST_CFREE_MB": "3000", "DAGT1_PARITY": "off", "DAGT1_INPUTS": f"{R}/dagt1/nonexistent_inputs"})
        rec("CH7 DAGT1_INPUTS overrides the default: pointing it elsewhere makes the chain STOP naming that path (it did not silently fall back to the default)", rc == 1 and "ZZDAGT1-STOP" in out and "nonexistent_inputs" in out, out.strip().splitlines()[-1][:150] if out.strip() else "")
        rc, out, dt = chain({"DAGT1_TEST_CFREE_MB": "3000"})
        full = out
        open(os.path.join(HERE, "dry_run_selftest.txt"), "w", encoding="utf-8").write(out)
        lines = [ln for ln in out.splitlines() if "[lane" in ln or "v8f_mini_eval.py run" in ln]
        causal = [ln for ln in lines if "[lane causal]" in ln]
        logged = [ln for ln in lines if "[lane logged]" in ln]
        evals = [ln for ln in lines if "v8f_mini_eval.py run" in ln]

        def has(ln, *subs):
            return all(s_ in ln for s_ in subs)
        order = [("dag", 0), ("dag0", 0), ("dag", 1), ("dag0", 1)]
        rec("CH10 dry run rc 0, ZZDAGT1-DRY-OK, and it touched nothing (no logs dir, no OUT dir created)", rc == 0 and "ZZDAGT1-DRY-OK" in out and not os.path.exists(os.path.join(HERE, "logs")) and not os.path.exists(f"{R}/v8f-dagger-t1"),
            f"rc {rc}, {dt:.0f} s")
        rec("CH11 four causal and four logged rollout commands, in the PREREG's order DAG s0, DAG-0 s0, DAG s1, DAG-0 s1, each on ITS OWN checkpoint and tag",
            len(causal) == 4 and len(logged) == 4 and all(has(causal[i], f"--ckpt {R}/v8f-dagger/M2b_s{s}_{a}/ckpt_last.pt", f"--tag {'DAG' if a == 'dag' else 'DAG0'}_s{s}_m10_causal_heads", "--limit causal")
                                                          and has(logged[i], f"--ckpt {R}/v8f-dagger/M2b_s{s}_{a}/ckpt_last.pt", f"--tag {'DAG' if a == 'dag' else 'DAG0'}_s{s}_m10_logged_heads", "--limit logged") for i, (a, s) in enumerate(order)), "")
        rec("CH12 every rollout carries the registered flags (--replan-steps 10 --nav deadreckon --scene logged --schedule periodic --dump-ticks --dump-heads --device cpu), plain (no post-map flag)",
            all(has(ln, "--replan-steps 10", "--nav deadreckon", "--scene logged", "--schedule periodic", "--dump-heads", "--dump-ticks", "--device cpu") and "--post" not in ln for ln in causal + logged), "")
        rec("CH13 four fresh-age T0 evals, --ages fresh --project --families, each on its run's checkpoint and the DAG tag M2b-<arm>_s<S>-fresh",
            len(evals) == 4 and all(has(evals[i], f"--ckpt {R}/v8f-dagger/M2b_s{s}_{a}/ckpt_last.pt", f"--tag M2b-{a}_s{s}-fresh", "--ages fresh", "--project", "--families") for i, (a, s) in enumerate(order)), "")
        rec("CH14 the tool, the eval grid / join / kin and the reader's --tool-root all come from the ONE tree parameter", all(f"{R}/di/tree/stack/scripts/v8f_t1_eval.py" in ln for ln in causal + logged)
            and all(f"{R}/di/tree/stack/scripts/v8f_mini_eval.py" in ln for ln in evals) and f"--tool-root {R}/di/tree" in out, "")
        rec("CH15 ORC-0 is not re-run: no rollout or eval command names an orc0 checkpoint", not any("orc0" in ln.lower() for ln in lines), "")
        rec("CH16 the manifest the dry run would write has DAG / DAG0 / ORC0, banks 0 and 1, both id lists, t0_fresh in OUT/t0fresh, ORC-0 under the banked O-RECOVER dirs",
            '"ORC0"' in out and '"bank"' in out and "train_clip_ids" in out and f"{R}/v8f-dagger-t1/t0fresh/eval_M2b-dag_s0-fresh.json" in out and f"{R}/v8f-orecover-t1/t0fresh/eval_M2b-orc0_s0-fresh.json" in out, "")
        rec("CH17 the reader command: dag_read.py -I, --manifest, --tool-root, --t1hd-read, --t1k-read, --m2b-sha 0 and 1, --prereg, --out DAG_READ.json", all(s_ in out for s_ in ("dag_read.py", "--manifest", "--tool-root", "--t1hd-read", "--t1k-read", "--m2b-sha", "--prereg", "DAG_READ.json")), "")
        rec("CH18 the parity guard ran against the ORC-0 tool tree and found the closure identical but one additive block", "tool parity with the ORC-0 tool tree" in out and "0 violations" in out, "")
        rec("CH19 the pins record the tree digest, the reader, the PREREG sha (a4df4b8f...), the release and the parity line", all(s_ in out for s_ in ("digest=", "reader=dag_read.py", "reader=PREREG a4df4b8f", "release=", "parity=closure")), "")
        rec("CH20 the dry run reports the bank and pull states (INCOMPLETE now) and the ORC-0 reuse check on the real outputs", "bank_s0  INCOMPLETE" in out and "ORC-0 outputs present" in out, "")
        c_paths = sorted(set(re.findall(r"[Cc]:[/\\][^\s'\"]*", full)))
        stray = [p_ for p_ in c_paths if not p_.replace("\\", "/").startswith(("C:/Users/Admin/venvs/tanitad/", "C:/Users/Admin/dagread/"))]
        rec("CH21 [nothing new on C:] every C: path in the rendered plan is the interpreter or the (read-only) reader; the inputs, tree, outputs, logs and ids are all on E:", not stray and f"{R}/sfresh_inputs/inputs/native_tracks.pt" in full, f"stray {stray[:3]}")
        bank_lines = [ln for ln in full.splitlines() if re.search(r"bank_s[01]  (INCOMPLETE|COMPLETE|WRONG)", ln)]
        rec("CH22 [bank files + O1] the plan states the 4-shard bank gate (11 names, bank.json == the gate's declared --dagger-bank-sha256), probes each bank with --expect-shards 4, accepts no box-2 fresh eval, and passes no --b2-fresh; the source STOPs on a WRONG bank (rc 3) and on the removed switch",
            "--expect-shards 4" in chain_src and "EXACTLY the 11 names" in full and "--dagger-bank-sha256" in full and "box-2 fresh eval: none accepted" in full and "--b2-fresh" not in full and "--b2-fresh" not in chain_src
            and len(bank_lines) == 2 and re.search(r"\[ \$rc -eq 3 \] && stop \"the pulled bank s\$s is WRONG", chain_src) is not None, f"{len(bank_lines)} bank lines")
        del full
    else:
        rec("CH10..CH21 skipped (--quick)", True, "")

    # ------------------------------------------------------------------------------------------------ F: mutants of the checkers
    if not NO_MUT:
        mut_dir = os.path.join(BASE, "mut")
        os.makedirs(mut_dir)
        src = open(os.path.join(HERE, "dagt1_pullcheck.py"), encoding="utf-8").read()
        muts = [
            ("M01 delete the sha256 compare", "        if got != listed[name]:\n            return \"INCOMPLETE\"", "        if False:\n            return \"INCOMPLETE\""),
            ("M02 delete the SHA256SUMS age test", "    if age < min_age_s:", "    if False:"),
            ("M03 accept any name set", "    if sorted(listed) != sorted(want):\n        for oa in ARMS:", "    if False:\n        for oa in ARMS:"),
            ("M04 delete the summary done test", "        if name == \"summary.json\" and rec.get(\"done\") is not True:", "        if False:"),
            ("M05 delete the final-newline test", "    if not text.endswith(\"\\n\"):", "    if False:"),
            ("M06 delete the non-empty test", "        if os.path.getsize(p) == 0:\n            return \"INCOMPLETE\", f\"{name} is empty\"", "        if False:\n            return \"INCOMPLETE\", f\"{name} is empty\""),
            ("M07 delete the exists test", "        if not os.path.isfile(p):\n            return \"INCOMPLETE\", f\"{name} is listed but absent\"", "        if False:\n            return \"INCOMPLETE\", f\"{name} is listed but absent\""),
            ("M08 delete the identity test (WRONG never fires)", "    if bad:\n        return \"WRONG\"", "    if False:\n        return \"WRONG\""),
            ("M09 drop the --out identity test", "    if out is None or os.path.basename(norm(out)) != run_name(arm, seed):", "    if False:"),
            ("M10 drop the dag0 DAG-only-flag test", "    if arm == \"dag0\":\n        for n in DAG_ONLY:", "    if False:\n        for n in DAG_ONLY:"),
            ("M11 [O1] a SUMS that carries a box-2 fresh-age eval is no longer refused", "    if carried:", "    if False:"),
            ("M12 the bank check does not require the shard npz to exist", "        if not os.path.isfile(p):\n            return \"INCOMPLETE\", f\"{nm} is listed but absent\"", "        if False:\n            return \"INCOMPLETE\", f\"{nm} is listed but absent\""),
            ("M13 the bank check does not compare bank.json's sha256 with PROVENANCE.bank_sha256", "    if prov.get(\"bank_sha256\") != msha:", "    if False:"),
            ("M14 the bank check does not compare bank.json with the sha256 the trainer gate declared", "    if v != msha:", "    if False:"),
            ("M15 the bank check does not compare a shard npz with bank.json's file_sha256", "        if listed[npz] != by_k[k].get(\"file_sha256\"):", "        if False:"),
            ("M16 the bank check does not compare a shard npz with its shard provenance", "        if not isinstance(sp, dict) or sp.get(\"file_sha256\") != listed[npz]:", "        if False:"),
            ("M17 the bank check accepts any name set in SHA256SUMS.txt", "    if sorted(listed) != sorted(want):\n        miss, extra = sorted(set(want) - set(listed)), sorted(set(listed) - set(want))\n        hint", "    if False:\n        miss, extra = sorted(set(want) - set(listed)), sorted(set(listed) - set(want))\n        hint"),
            ("M18 the bank check ignores --expect-shards", "    if expect_shards is not None and n != expect_shards:", "    if False:"),
            ("M19 the declared-sha lookup does not need the pin", "    if not os.path.isfile(pin) or os.path.getsize(pin) == 0:", "    if False:"),
            ("M20 the declared-sha lookup accepts a pin that names the flag twice / not at all", "    if v is None:\n        return \"WRONG\"", "    if False:\n        return \"WRONG\""),
            ("M21 the declared-sha lookup reads seed 0's pin for every seed", "    pin = os.path.join(art, run_name(\"dag\", seed), \"gate\", \"pin.txt\")", "    pin = os.path.join(art, run_name(\"dag\", 0), \"gate\", \"pin.txt\")"),
            ("M22 SHA256SUMS.txt may list a name twice", "        if name in listed:", "        if False:"),
            ("M23 bank.json's shard names are not checked against the generator's naming", "    if [s.get(\"file\") for s in sorted(shards, key=lambda s: s.get(\"k\", -1))] != names[\"shard_npz\"]:", "    if False:"),
            ("M24 PROVENANCE.shard_provenance is not checked against the generator's naming", "    if prov.get(\"shard_provenance\") != names[\"shard_provenance\"] or prov.get(\"manifest\", \"bank.json\") != \"bank.json\":", "    if False:"),
            ("M25 PROVENANCE.json of another kind / schema is accepted", "    if not isinstance(prov, dict) or prov.get(\"kind\") != BANK_KIND or prov.get(\"schema\") != BANK_SCHEMA:", "    if False:"),
            ("M26 the bank CLI drops --expect-shards", "        st, why, sha = check_bank_status(a[1], int(a[2]), expect_shards=exp)", "        st, why, sha = check_bank_status(a[1], int(a[2]))"),
            ("M27 the run CLI no longer refuses the removed --b2-fresh switch", "        if len(a) != 4 or not a[3].isdigit():", "        if False:"),
        ]
        killed = 0
        for name, a, b in muts:
            if a not in src:
                rec(f"{name}: mutation site found in the source", False, "site missing")
                continue
            mp = os.path.join(mut_dir, "dagt1_pullcheck_mut.py")
            open(mp, "w", encoding="utf-8", newline="\n").write(src.replace(a, b, 1))
            spec = importlib.util.spec_from_file_location("pc_mut", mp)
            pm = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(pm)
            bad_ = []
            for suite, fn_ in (("A", lambda: pc_tests(pm, "[mut] ")), ("B", lambda: bk_tests(pm, TD, "[mut] ")), ("SB", lambda: smoke_bank_tests(pm, TD, "[mut] "))):
                try:
                    bad_ += [n for n, o, _ in fn_() if not o]
                except Exception as ex:                      # noqa: BLE001  -- a mutant that crashes the checker is detected (the chain would STOP on a crashed checker)
                    bad_.append(f"crash {suite} {type(ex).__name__}")
                    for a_ in ARTS + BKS:
                        shutil.rmtree(a_, ignore_errors=True)
                    ARTS.clear()
                    BKS.clear()
            rec(f"{name}: a check turns RED", bool(bad_), f"{len(bad_)} red: {', '.join(x[:60] for x in bad_[:3])}")
            killed += bool(bad_)
        src2 = open(os.path.join(HERE, "dagt1_manifest.py"), encoding="utf-8").read()
        muts2 = [
            ("N01 drop the ckpt_tail identity test", "                if e[\"ckpt_tail\"] != f\"{want_run}/ckpt_last.pt\":", "                if False:"),
            ("N02 drop the train_dir identity test", "                if os.path.basename(fwd(e[\"train_dir\"])) != want_run:", "                if False:"),
            ("N03 drop the shared-path test", "                    if p in seen and seen[p] != lab:", "                    if False:"),
            ("N04 drop the rollout ckpt-tail test", "                if tail2(str(h.get(\"ckpt\", \"\"))) != e[\"ckpt_tail\"]:", "                if False:"),
            ("N05 drop the T0 eval ckpt-tail test", "                    if tail2(j.get(\"ckpt\", \"\")) != e[\"ckpt_tail\"]:", "                    if False:"),
            ("N06 drop the T0 ages test", "                    if j.get(\"ages\") != want_ages:", "                    if False:"),
            ("N07 drop the distinct-checkpoint test", "                if sha in shas:", "                if False:"),
            ("N08 drop the done-record sha test", "                    if dr.get(\"ckpt_sha256\") != sha:", "                    if False:"),
            ("N09 drop t0_fresh requirement", "            if \"t0_fresh\" not in e:", "            if False:"),
            ("N10 drop the bank check", "            ok, why, _ = PC.check_bank(os.path.dirname(bdir), bseed)", "            ok, why, _ = True, \"\", \"\""),
            ("N11 drop the id-list existence test", "    for k in (\"eval141_clip_ids\", \"train_clip_ids\"):\n        need_file(m[k], k)", "    pass"),
            ("N12 drop the banked-init test", "        if require_banked and res[seed] != BANKED_M2B_SHA[seed]:", "        if False:"),
            ("N13 drop the TRAIN-list hash test", "    if got_t != want_train:", "    if False:"),
            ("N14 drop the done-record sha at ORC-0", "            if dr.get(\"ckpt_sha256\") != got:", "            if False:"),
            ("N15 parity: a modified line is not a violation", "        if all(op[0] in (\"equal\", \"insert\") for op in ops):", "        if True:"),
            ("N16 bank_file may be a bank.json elsewhere than the pulled one beside the report", "            elif fwd(bf) != fwd(os.path.join(bdir, \"bank.json\")):", "            elif False:"),
            ("N17 a bank entry may point at another seed's bank directory", "            if bseed != int(s) or os.path.basename(be[\"report\"]) != \"PROVENANCE.json\":", "            if False:"),
            ("N18 a bank entry may carry no bank_file", "            if not bf:", "            if False:"),
        ]
        for name, a, b in muts2:
            if a not in src2:
                rec(f"{name}: mutation site found in the source", False, "site missing")
                continue
            mp = os.path.join(mut_dir, "dagt1_manifest.py")
            shutil.copy(os.path.join(HERE, "dagt1_pullcheck.py"), os.path.join(mut_dir, "dagt1_pullcheck.py"))
            open(mp, "w", encoding="utf-8", newline="\n").write(src2.replace(a, b, 1))
            spec = importlib.util.spec_from_file_location("mf_mut", mp)
            mm = importlib.util.module_from_spec(spec)
            sys.path.insert(0, mut_dir)
            try:
                spec.loader.exec_module(mm)
            finally:
                sys.path.pop(0)
            red = []
            # re-run the manifest regressions against the mutant: verify() on the mutated worlds, make_ids / m2b_sha / orc0 / parity on theirs
            try:
                def tryv(label, edit, undo, mmod=mm, needle=None):
                    edit()
                    try:
                        b_ = mmod.verify(m, 2)
                    finally:
                        undo()
                    if not b_ or (needle is not None and not any(needle in x for x in b_)):          # with a needle: the refusal must still carry the reason the real verifier gives
                        red.append(label)
                tryv("MF10", e_json("t1", e_0, f"{WD.art}/M2b_s0_dag/ckpt_last.pt"), lambda: st["u"]())
                tryv("MF12", ed12, lambda: e_0.__setitem__("ckpt_tail", st["k"]))
                tryv("MF13", ed13, lambda: st["u"]())
                tryv("MF14", ed14, lambda: (open(os.path.join(e_0["train_dir"], "ckpt_last.pt"), "wb").write(st["keep"]), write_sums(e_0["train_dir"], PC.base_files("dag0", 0))))
                tryv("MF15", ed15, lambda: st["u"]())
                tryv("MF17", ed17, lambda: st["u"]())
                tryv("MF20", ed20, lambda: e_0.update(st["k"]))
                tryv("MF21", ed21, lambda: e_0["t1"].update(st["k"]))
                tryv("MF30", lambda: os.rename(bp, bp + ".x"), lambda: os.rename(bp + ".x", bp))
                for lab_, ed_, un_, nd_ in bank_cases:
                    tryv(lab_[:4], ed_, un_, needle=nd_)
                tryv("MF40", lambda: os.rename(fz, fz + ".x"), lambda: os.rename(fz + ".x", fz))
                tryv("MF42", lambda: e_d.pop("t0_fresh"), lambda: e_d.__setitem__("t0_fresh", st["f"]))
                tryv("MF44", lambda: os.rename(ip, ip + ".x"), lambda: os.rename(ip + ".x", ip))
                tryv("MF50", ed50, un50)
                tryv("MF51", ed51, un51)
                tryv("MF52", ed52, lambda: e_0["t1"].__setitem__("dump_dir", st["k"]))
                tryv("MF53", ed53, un53)
                try:
                    mm.make_ids(grid, rel, idsd, want_train="0" * 64, dry=True)
                    red.append("ID2")
                except SystemExit:
                    pass
                try:
                    mm.m2b_sha(WD.art, runs, WD.orc_art, None, require_banked=True)
                    red.append("MS1")
                except SystemExit:
                    pass
                undo2 = jedit(os.path.join(WD.orc_t1, "done_ORC0_s0_m10_logged_heads.txt"), lambda o: o.__setitem__("ckpt_sha256", "e" * 64))
                b2_ = mm.orc0_check(WD.orc_art, WD.orc_t1, n_dump_min=2)
                undo2()
                if not b2_:
                    red.append("OR2")
                open(os.path.join(mini_tree, "stack/tanitad/data/v8f_mini_data.py"), "w", newline="\n").write("a = 1\nb = 3\n")
                p_ = mm.parity(mini_tree, mini_ref)
                open(os.path.join(mini_tree, "stack/tanitad/data/v8f_mini_data.py"), "w", newline="\n").write("a = 1\nb = 2\n")
                if not any("v8f_mini_data.py" in v_ for v_ in p_["violations"]):
                    red.append("TP4")
            except Exception as ex:                          # noqa: BLE001
                red.append(f"crash {type(ex).__name__}")
            rec(f"{name}: a check turns RED", bool(red), f"{len(red)} mutated-away check(s): {', '.join(red[:4])}")
        # the nominal manifest check must also be green on the unmutated module (control for the harness)
        rec("N00 control: the harness reads GREEN on the unmutated module (no mutation-site fired)", not MF.verify(m, 2), "")

    # ------------------------------------------------------------------------------------------------ report
    w = max(len(n) for n, _, _ in RES)
    for n, ok, why in RES:
        print(f"{'PASS' if ok else 'FAIL'}  {n.ljust(w)}  | {why}")
    bad = [n for n, ok, _ in RES if not ok]
    print(f"\nSELFTEST (fixtures under {BASE}; {time.time() - t_start:.0f} s, of which the reader runs {WD_t_rd:.0f} s): {len(RES) - len(bad)}/{len(RES)} as expected" + (f"; FAILED: {bad}" if bad else ""))
    try:
        shutil.rmtree(BASE)
        print(f"scratch {BASE} removed")
    except OSError as ex:
        print(f"scratch {BASE} NOT removed: {ex}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
