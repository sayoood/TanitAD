#!/usr/bin/env python3
"""lit1_fix.py -- SYNTHETIC fixtures of lit1_selftest.py: pull directories, box-2 chain logs, a stub tool tree and a stub reader, all inside F:/TanitAD-scratch/launch_intent_t1.
Nothing here is real data; the values that must be real (init / bank / window shas, the lever, the launch commit) are the registered constants, because that is what the checks compare.
The stub tools write the files the real tools write (a few bytes each) and obey a control file, so the chain's control flow can be exercised without a single real rollout."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
import lit1_lib as L  # noqa: E402

SCR = L.SCRATCH
RSHA = "ab" * 32        # the fixture reader's / PREREG's sha256 are overwritten by mk_world with the real hashes of the world's files
PSHA = "cd" * 32
COUNTS = {0: (30790, 191311), 1: (30601, 191122), 10: (30790, 191311), 11: (30601, 191122)}


def det_bytes(label: str, n: int = 4096) -> bytes:
    out, h = b"", label.encode()
    while len(out) < n:
        h = hashlib.sha256(h).digest()
        out += h
    return out[:n]


def jdump(p: str, o) -> None:
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(o, f, indent=1)


def base_parts(arm: str, seed: int, commit: str = L.LAUNCH_COMMIT, rsha: str = RSHA, psha: str = PSHA) -> dict:
    run, tseed = L.run_name(arm, seed), seed + L.SEED_OFF[arm]
    v = "D:\\tanitad\\runs\\v8f_mini_full\\"
    argv = ["--arm", "M2b", "--seed", str(tseed), "--grid", "pool4x10", "--out", v + run, "--cache-dir", "D:\\tanitad\\cache_full", "--init-ckpt", v + f"M2b_s{seed}\\ckpt_last.pt", "--steps", "3000", "--lr", "1e-4",
            "--warmup", "100", "--ego-perturb-p", "0", "--dagger-p", "0.1", "--dagger-bank", v + f"dagger_bank\\s{seed}", "--dagger-bank-sha256", L.BANK1_SHA[seed], "--dagger-seed", str(seed), "--w-acc", "49.6969",
            "--launch-intent-w", "1.58926"]
    sha16 = hashlib.sha256(" ".join(argv).encode()).hexdigest()[:16]
    n, e = COUNTS[tseed]
    li = {"launch_intent_w": L.LI_W, **L.LI_MASK, "axis": "along (x of the ego frame at NOW)", "mask_from": "the LOGGED kin3[:, 0] and the GT future, taken before any ego stage"}
    lever = {"init_ckpt": {"path": v + f"M2b_s{seed}\\ckpt_last.pt", "sha256": L.M2B_SHA[seed], "applied": True, "n_mismatch_after_load": 0}, "dagger_p": 0.1, "dagger_bank_sha256": L.BANK1_SHA[seed], "w_acc": 49.6969,
             "ego_perturb": {"ego_perturb_p": 0.0}, "ego_perturb_with_dagger": False, "launch_intent_w": L.LI_W, "launch_intent": dict(li), "launch_n": 100, "launch_eligible": 600}
    summary = {"done": True, "steps": 3000, "arm": "M2b", "seed": tseed, "train_window_list_sha256": L.WINDOW_SHA, "excluded_sha256": L.EXCLUDED_SHA, "argv_sha": sha16, "levers": lever,
               "launch_n": n, "launch_eligible": e, "launch_rows": e, "launch_intent_w": L.LI_W, "launch_weight_share": L.share_of(n, e, L.LI_W)}
    selfcheck = {"declared_vs_built": "PASS", "argv_sha": sha16, "levers": {"launch_intent_w": L.LI_W, "launch_intent": dict(li)}}
    token = {"kind": "launch-gate-token", "verdict": "PASS", "prereg": "PREREG_V8F_LAUNCH_INTENT", "run": run, "binding": {"commit": commit, "argv_sha256": "0" * 64, "data_sha256": "1" * 64},
             "prereg_sha256": psha, "reader_sha256": rsha, "key_files_sha256": {L.TRAINER_PATH: L.LAUNCH_TRAINER_SHA, L.MODEL_PATH: L.LAUNCH_MODEL_SHA}, "excluded_sha256": L.EXCLUDED_SHA,
             "init_ckpt_sha256": L.M2B_SHA[seed], "bank_sha256": L.BANK1_SHA[seed], "dagger_p": 0.1, "w_acc": 49.6969, "arm": arm}
    return {"argv": {"argv": argv, "sha": sha16}, "summary": summary, "selfcheck": selfcheck, "token": token, "ckpt": det_bytes(f"ckpt-{run}", 8192), "sha16": sha16}


def mk_run(art: str, arm: str, seed: int, mut=None, age: float = 120.0, sums: bool = True, **kw) -> str:
    """Write one pull directory (the 8 names + SHA256SUMS.txt aged ``age`` s).  ``mut(parts)`` edits the parts before they are written."""
    run, tag = L.run_name(arm, seed), L.eval_tag(arm, seed)
    d = f"{art}/{run}"
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    P = base_parts(arm, seed, **kw)
    if mut:
        mut(P)
    jdump(f"{d}/argv.json", P["argv"])
    jdump(f"{d}/summary.json", P["summary"])
    jdump(f"{d}/selfcheck.json", P["selfcheck"])
    jdump(f"{d}/launch_gate_token.json", P["token"])
    open(f"{d}/ckpt_last.pt", "wb").write(P["ckpt"])
    jdump(f"{d}/eval_{tag}.json", {"ckpt": f"D:/tanitad/runs/v8f_mini_full/{run}/ckpt_last.pt", "ages": "cycle", "tag": tag})
    open(f"{d}/eval_{tag}.npz", "wb").write(det_bytes("npz-" + run, 256))
    open(f"{d}/train_log.jsonl", "w", newline="\n").write('{"step": 1}\n')
    if sums:
        write_sums(d, tag, age)
    return d


def write_sums(d: str, tag: str, age: float = 120.0, names=None, text_mut=None) -> None:
    names = names or L.sums_names(tag)
    txt = "".join(f"{L.sha256_file(f'{d}/{n}')}  {n}\n" for n in names if os.path.isfile(f"{d}/{n}"))
    if text_mut:
        txt = text_mut(txt)
    open(f"{d}/SHA256SUMS.txt", "w", newline="\n").write(txt)
    t = time.time() - age
    os.utime(f"{d}/SHA256SUMS.txt", (t, t))


def ck_sha(d: str) -> str:
    return L.sha256_file(f"{d}/ckpt_last.pt")


LAB = {k: v for k, v in L.LABEL.items()}


def log_run(art: str, arm: str, seed: int, sha: str, rec="PASS", ksh="PASS", path=None, label=None, control=None, rec_label=None, ksh_label=None) -> list:
    g, lb = L.run_name(arm, seed), label or L.LABEL[(arm, seed)]
    ctl = control or L.CONTROL[(arm, seed)]
    n, e = COUNTS[seed + L.SEED_OFF[arm]]
    out = [f"10-28 09:{seed:02d} {g} pulled to {path or art + '/' + g} (checkpoint sha256 {sha}) [{lb}, control {ctl}]"]
    if rec:
        out.append(f"LAUNCH-INTENT run record: {rec} | {rec_label or lb} | {arm} s{seed} | trainer seed {seed + L.SEED_OFF[arm]} | argv sha16 0123456789abcdef | steps 3000 | launch_intent_w 1.58926 | w_acc 49.6969")
    if ksh:
        out.append(f"LAUNCH-INTENT K-SHARE: {ksh} | {ksh_label or lb} | launch {n} / {e} = {n / e:.6f} (K-MASK: f 0.1619 +- 0.02) | weight share 0.331800 (must lie in 1/3 +- 0.02) | rows {e} (== eligible) | ok")
    return out


def done_line(commit: str = L.LAUNCH_COMMIT) -> str:
    return (f"ZZV8FMINI-DONE 10-28 11:30 tip {commit}: PREREG_V8F_LAUNCH_INTENT arm A (launch-intent-w 1.58926): 4 runs (launchintent 0 launchintent 1 launchintentrep 0 launchintentrep 1) each through its gate, "
            "trained 3,000 steps, T0-evaluated (standard) and pulled (boot unchanged x)")


def write_log(path: str, lines: list, crlf: bool = False) -> None:
    open(path, "wb").write(("\r\n" if crlf else "\n").join(lines).encode() + (b"\r\n" if crlf else b"\n"))


STUB_TOOL = r'''#!/usr/bin/env python3
# a STUB of v8f_t1_eval.py / v8f_mini_eval.py: writes the files the real tool writes (a few bytes), obeys <world>/stub_ctl.txt, logs START / END.  NOT a rollout.
import json, os, sys, time
a = sys.argv[1:]
name = os.path.basename(sys.argv[0])
def opt(k, d=None):
    return a[a.index(k) + 1] if k in a else d
ckpt, tag, out = opt("--ckpt"), opt("--tag"), opt("--out-dir")
world = os.environ["LIT1_WORLD"]
ctl = open(world + "/stub_ctl.txt").read().split() if os.path.exists(world + "/stub_ctl.txt") else []
has = lambda k: (k + ":" + tag) in ctl
log = open(world + "/stub_calls.log", "a")
log.write("START %s %.3f %s\n" % (tag, time.time(), name)); log.write("ARGV %s %s\n" % (tag, " ".join(a))); log.flush()
if has("sleep"): time.sleep(3)
if has("mutate_ckpt"): open(ckpt, "ab").write(b"x")
if has("mutate_tree"): open(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "..", "tanitad", "added_by_stub.py"), "w").write("x = 1\n")
if has("mutate_other"): open(os.environ["LIT1_STUB_OTHER"], "ab").write(b"x")
if has("fail"):
    log.write("FAIL %s\n" % tag); sys.exit(1)
if name.startswith("v8f_t1"):
    lim = opt("--limit")
    if has("badlimit"): lim = "logged" if lim == "causal" else "causal"      # a rollout json that does not say what was run (only the manifest check can see it)
    os.makedirs(out, exist_ok=True); os.makedirs(opt("--dump-ticks"), exist_ok=True)
    n = 5 if has("fewdumps") else 141
    os.makedirs(out + "/dump_" + tag, exist_ok=True)
    for i in range(n): open(out + "/dump_%s/ep%03d.npz" % (tag, i), "wb").write(b"d")
    open(opt("--dump-ticks") + "/ticks_%s.npz" % tag, "wb").write(b"t")
    json.dump({"t1v8f": {"ckpt": ckpt, "limit": lim, "max_windows": None, "n_windows": 2399, "replan_steps": 10}}, open(out + "/t1v8f_%s.json" % tag, "w"))
else:
    os.makedirs(out, exist_ok=True)
    json.dump({"ckpt": ckpt, "ages": "fresh", "tag": tag}, open(out + "/eval_%s.json" % tag, "w"))
    open(out + "/eval_%s.npz" % tag, "wb").write(b"n")
log.write("END %s %.3f\n" % (tag, time.time()))
if has("lowram_after"): open(world + "/hook_ram_mb.txt", "w").write("100")
if has("b2stop"): open(os.environ["LIT1_B2_OUT"], "a").write("ZZV8FMINI-STOP stub: the box-2 chain stopped\n")
if has("failafter"): sys.exit(1)
'''

STUB_READER = r'''#!/usr/bin/env python3
# a STUB reader: records its call (argv, how many done records exist), checks the manifest parses, writes --out, exits with <world>/reader_rc.txt.  NOT the pre-registered reader.
import glob, json, os, sys, time
a = sys.argv[1:]
world = os.environ.get("LIT1_WORLD", "")
def opt(k, d=None):
    return a[a.index(k) + 1] if k in a else d
if "--static-check" in a:
    print("STUB-STATIC-OK"); sys.exit(0)
man = json.load(open(opt("--manifest")))
n_done = len(glob.glob(os.path.dirname(opt("--manifest")) + "/done_*.txt"))
open(world + "/reader_calls.log", "a").write("CALL %.3f argv=%s arms=%s n_done=%d\n" % (time.time(), " ".join(a), ",".join(sorted(man["arms"])), n_done))
os.makedirs(os.path.dirname(opt("--out")), exist_ok=True)
json.dump({"OUTCOME": "STUB", "arms": sorted(man["arms"])}, open(opt("--out"), "w"))
rcf = world + "/reader_rc.txt"
sys.exit(int(open(rcf).read().strip()) if os.path.exists(rcf) else 0)
'''


class World:
    """A throwaway world under F:/TanitAD-scratch/launch_intent_t1/w_<name>: a copy of the package (so a mutant can edit its copy), pulls, a box-2 log, a stub tool tree and a stub reader."""

    def __init__(self, name: str):
        self.name = name
        self.root = f"{SCR}/w_{name}"
        shutil.rmtree(self.root, ignore_errors=True)
        os.makedirs(self.root)
        for sub in ("art", "out", "logs", "tree_stub/stack/scripts", "tree_stub/stack/tanitad", "tree_stub/taniteval"):
            os.makedirs(f"{self.root}/{sub}")
        self.H = f"{self.root}/H"
        os.makedirs(self.H + "/argv")
        for f in ("lit1_chain.sh", "lit1_lib.py", "lit1_tree.py", "wake_lit1.sh"):
            shutil.copyfile(f"{HERE}/{f}", f"{self.H}/{f}")
        for f in os.listdir(f"{HERE}/argv"):
            shutil.copyfile(f"{HERE}/argv/{f}", f"{self.H}/argv/{f}")
        for f in ("v8f_t1_eval.py", "v8f_mini_eval.py"):
            open(f"{self.root}/tree_stub/stack/scripts/{f}", "w", newline="\n").write(STUB_TOOL)
        open(f"{self.root}/tree_stub/stack/tanitad/__init__.py", "w").write("")
        open(f"{self.root}/tree_stub/taniteval/__init__.py", "w").write("")
        open(f"{self.root}/reader_stub.py", "w", newline="\n").write(STUB_READER)
        open(f"{self.root}/PREREG.md", "w", newline="\n").write("# a stub PREREG\n")
        self.rsha = L.sha256_file(f"{self.root}/reader_stub.py")
        self.psha = L.sha256_file(f"{self.root}/PREREG.md")
        self.art, self.out, self.logs = f"{self.root}/art", f"{self.root}/out", f"{self.root}/logs"
        self.b2 = f"{self.root}/box2.out"
        self.sha = {}
        open(f"{self.root}/stub_ctl.txt", "w").write("")

    def env(self, **kw) -> dict:
        e = {"LIT1_WORLD": self.root, "LIT1_ART": self.art, "LIT1_OUT": self.out, "LIT1_LOG": self.logs, "LIT1_B2_OUT": self.b2, "LIT1_TREE_LI": f"{self.root}/tree_stub", "LIT1_READER": f"{self.root}/reader_stub.py",
             "LIT1_PREREG": f"{self.root}/PREREG.md", "LIT1_POLL_S": "1", "LIT1_MAX_WAIT_S": "6", "LIT1_B2DONE_WAIT_S": "6", "LIT1_GUARD_WAIT_S": "3", "LIT1_MIRROR": "C:/Users/Admin/tanitad-push"}
        e.update(kw)
        return e

    def pull(self, arm: str, seed: int, mut=None, commit: str = L.LAUNCH_COMMIT, **kw) -> str:
        kw.setdefault("rsha", self.rsha)
        kw.setdefault("psha", self.psha)
        d = mk_run(self.art, arm, seed, mut=mut, commit=commit, **kw)
        self.sha[(arm, seed)] = ck_sha(d)
        return d

    def log(self, specs=None, done=True, extra_before=None, extra_after=None, crlf=False, **kw) -> None:
        """specs = [(arm, seed, {log_run options})] in PULL ORDER; default all four in the box-2 order."""
        specs = specs if specs is not None else [(a, s, {}) for (a, s) in L.RUNS]
        lines = list(extra_before or [])
        for arm, seed, o in specs:
            lines += log_run(self.art, arm, seed, self.sha[(arm, seed)], **o)
        lines += list(extra_after or [])
        if done:
            lines.append(done_line())
        write_log(self.b2, lines, crlf)

    def all_pulls(self, **kw) -> None:
        for a, s in L.RUNS:
            self.pull(a, s, **kw)

    def ctl(self, *items) -> None:
        open(f"{self.root}/stub_ctl.txt", "w").write(" ".join(items))

    def calls(self) -> list:
        p = f"{self.root}/stub_calls.log"
        return open(p).read().splitlines() if os.path.exists(p) else []

    def reader_calls(self) -> list:
        p = f"{self.root}/reader_calls.log"
        return open(p).read().splitlines() if os.path.exists(p) else []

    def chain_out(self) -> str:
        p = f"{self.logs}/chain.out"
        return open(p, encoding="utf-8", errors="replace").read() if os.path.exists(p) else ""

    def chain_cmd(self, args: str = "") -> str:
        return f'bash "{self.H}/lit1_chain.sh" {args} < /dev/null'


class Chk:
    """Collects failures instead of raising, so a scenario reports ALL of its broken expectations."""

    def __init__(self):
        self.fails: list = []

    def ok(self, cond, msg):
        if not cond:
            self.fails.append(msg)

    def eq(self, a, b, msg):
        if a != b:
            self.fails.append(f"{msg}: got {a!r}, want {b!r}")

    def has(self, text, sub, msg):
        if sub not in text:
            self.fails.append(f"{msg}: {sub!r} not in {text[:300]!r}")

    def no(self, text, sub, msg):
        if sub in text:
            self.fails.append(f"{msg}: {sub!r} unexpectedly in {text[:300]!r}")
