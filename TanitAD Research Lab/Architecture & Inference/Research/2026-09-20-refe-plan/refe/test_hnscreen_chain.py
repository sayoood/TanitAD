#!/usr/bin/env python3
"""Tests for refe/hnscreen_devbox.py (P6-HN-S checklist item 8; PREREG_P6HN_SCREEN.md sections 4-5, 11, 13; MM_DECISIONS_SCREEN.md "Integration still owed").
Run:  PYTHONIOENCODING=utf-8 python refe/test_hnscreen_chain.py   ->   ZZTEST_PASS <n>   (prints the mutation table).

EVERYTHING RUNS IN TEMP DIRS WITH STUBBED TOOLS.  The trainer, the launch gate, hn_bank.py, hn_screen_families.py and hn_screen_read.py are tiny stub scripts that journal their argv and the content of the GPU-lock
file at the moment they ran; the chain calls them through their CLIs exactly as it calls the real ones.  The GPU lock is the REAL landed helper (eval/aux_navtest_forward.py `gpu_lock`) pointed at a TEMPORARY lock
path (wait_idle off, memory probe stubbed).  The shared dev-box lock files (devbox_gpu.lock / devbox_gpu.chain_active) are never opened for writing: the last check stats them before and after the whole suite.

WHAT IS PROVEN:
  A  stage order: the chain runs S0, S1, S2, T0-T2, P0-P2, S4, VERDICT, NEXT and the tools are called in exactly the registered sequence (the full label sequence is asserted); every stage leaves a DONE marker.
  B  one HANDOFF per stage (state, artifacts, exact next step, resume command, < 6000 chars) and the gate key path appears in no file under the state dir.
  C  resume after an interruption (the pass-1 trainer dies): finished stages are NOT re-run, the failed pass is archived and retried, the lock is released on the failure.
  D  resume is by CONTENT: an untouched resume re-runs nothing; a finished stage whose output changed bytes re-runs.
  E  no trainer without a token: no PASS token, a refusing verify, a verify that exits 0 without the MATCH marker, and a refusal at the START of pass 1 all end rc 5 with the trainer not started; every trainer
     start is IMMEDIATELY preceded by a verify of the same pass's token.
  F  the GPU lock is the landed one, held (pid = this process) around the preflights, the I6/G15 gates, the verify+trainer+post-pass gates and the offsets scoring; NOT held by the chain around the cache driver
     (which takes it per unit); gone after the run.
  G  another job's lock is never removed: a live foreign holder is waited for (and the chain proceeds only after ITS owner removed it); a dead-holder lock 2 h old is never removed (the chain yields, rc 75);
     the chain_active marker is respected; a busy GPU is retried to the deadline.  CONTROL: the landed helper called directly DOES remove that old dead-holder lock, which is why the chain only calls it on a free file.
  H  the integration note: `hn_screen_families.py --per-set-out P` and `hn_screen_read.py --families-per-set P`, the gates file complete; rule E adds seeds 3 and 4 exactly once; a second PENDING is INVALID.
  I  stop points: SCREEN-SPARSE (rc 3, no cache), SCREEN-INVALID (rc 2) taking precedence over SPARSE, I6/G15 failures train nothing, a failed post-pass gate stops before the read.
  J  the argv levers (--seed/--out/--dump-scores/--hn-bank/--ctx-cache exactly once, with this pass's values; no --epo/--aux-drv) and every missing named parameter refuse BEFORE anything runs.
  K  a changed config on an existing state dir is refused; a second live chain on the same root is refused; a low disk stops cleanly.
  N  the shared dev-box lock files were not touched by the suite.
Every mutation below edits hnscreen_devbox.py SOURCE (each snippet must occur EXACTLY once) and must turn >= 1 expected check RED.
"""
from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
REAL_PKG = os.path.dirname(HERE)
SRC_PATH = os.path.join(HERE, "hnscreen_devbox.py")
PY = getattr(sys, "_base_executable", None) or sys.executable
SHARED_LOCK = "C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock"
SHARED_MARK = "C:/Users/Admin/qland/work/refcv7/devbox_gpu.chain_active"
KEYPATH = "SECRETKEYPATH_must_not_leak"

STUBLIB = r'''
import hashlib, json, os, re, sys, time


def val(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv else None


def vals(argv, flag):
    return [argv[i + 1] for i, a in enumerate(argv) if a == flag]


def spec():
    try:
        return json.load(open(os.environ["STUB_SPEC"], encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def bump(name):
    d = os.environ["STUB_STATE"]
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, re.sub(r"[^A-Za-z0-9_.-]", "_", name) + ".cnt")
    n = (int(open(p).read()) if os.path.exists(p) else 0) + 1
    open(p, "w").write(str(n))
    return n


def seed_of(argv):
    s = val(argv, "--seed")
    if s is not None:
        return int(s)
    for a in argv:
        m = re.search(r"passes/s(\d+)/", a.replace("\\", "/"))
        if m:
            return int(m.group(1))
    return None


def record(label, argv):
    lk = None
    try:
        lk = json.load(open(os.environ["STUB_LOCK"], encoding="utf-8"))
    except (OSError, ValueError):
        pass
    with open(os.environ["STUB_JOURNAL"], "a", encoding="utf-8") as f:
        f.write(json.dumps({"label": label, "argv": argv, "lock": lk, "seed": seed_of(argv), "t": time.time()}) + "\n")


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w", encoding="utf-8", newline="\n").write(text)


def main(tool, argv):
    sp = spec()
    if tool == "step":
        name, rest = argv[0], argv[1:]
        sd = seed_of(rest)
        label = f"{name}:{sd}" if val(rest, "--seed") is not None else name
        n = bump(label)
        record(label, rest)
        b = sp.get(label) or sp.get(name) or {}
        if b.get("rc"):
            return int(b["rc"])
        for w in vals(rest, "--write"):
            p, _, t = w.partition("=")
            if b.get("ok") is False:
                t = t.replace('"ok": true', '"ok": false')
            write(p, t)
        if val(rest, "--arms"):
            for a in val(rest, "--arms").split(","):
                write(os.path.join(val(rest, "--arms-dir"), f"scores9_{a}.npz"), a)
        pbc = b.get("print_by_call")
        for t in (pbc[min(n, len(pbc)) - 1] if pbc else []):
            print(t)
        return 0
    if tool == "trainer":
        pre = "--preflight" in argv
        seed = int(val(argv, "--seed"))
        label = f"trainer:{'pre' if pre else 'train'}:{seed}"
        n = bump(label)
        record(label, argv)
        out, scores = val(argv, "--out"), val(argv, "--dump-scores")
        if pre:
            write(os.path.join(out, "preflight.jsonl"), "{}")
            print("ZZSFT_PREFLIGHT_PASS")
            return 0
        f = (sp.get("trainer") or {}).get("fail", {}).get(str(seed))
        if f and n <= f.get("calls", 10 ** 6):
            if f.get("partial"):
                write(os.path.join(scores, "partial.tmp"), "half")
            return 1
        for nm in ("scores_update1000_A.npz", "scores_update1000_B.npz"):
            write(os.path.join(scores, nm), f"seed{seed}{nm}")
        return 0
    if tool == "gate":
        sub = argv[0]
        af = val(argv, "--args-file")
        label = f"gate:{sub}:{seed_of(argv)}"
        n = bump("gate:" + sub)
        record(label, argv)
        sha = hashlib.sha256(open(af, "rb").read()).hexdigest()
        if sub == "run":
            if (sp.get("gate_run") or {}).get("no_token"):
                return 1
            write(os.path.join(val(argv, "--out-dir"), "PASS_tok.json"), json.dumps({"verdict": "PASS", "binding": {"argv_sha256": sha}, "checks": {"G14": {"status": "PASS"}, "G15": {"status": "PASS"}}}))
            return 0
        sv = sp.get("verify") or {}
        tok = json.load(open(val(argv, "--token"), encoding="utf-8"))
        ok = tok["binding"]["argv_sha256"] == sha and not (sv.get("refuse_from_call") and n >= sv["refuse_from_call"])
        write(val(argv, "--json"), json.dumps({"verdict": "MATCH" if ok else "REFUSED"}))
        if ok and not sv.get("no_marker"):
            print("ZZGATE-MATCHZZ")
        else:
            print("ZZGATE-REFUSEDZZ" if not ok else "")
        return 0 if ok else 5
    if tool == "hn_bank":
        bump("hn_bank")
        record("hn_bank:yield", argv)
        write(val(argv, "--out"), json.dumps({"share_with_hn": 0.4, "sparse": bool(sp.get("sparse", False))}))
        print("ZZHNBANK_SY")
        return 0
    if tool == "families":
        bump("families")
        record("families", argv)
        for fl in ("--inputs", "--picks"):
            if not os.path.exists(val(argv, fl) or ""):
                print("missing", fl)
                return 3
        write(val(argv, "--out"), "{}")
        write(val(argv, "--per-set-out"), "per-set")
        print("ZZHNFAM DONE")
        return 0
    if tool == "read":
        n = bump("read")
        record("read", argv)
        problems = []
        i = argv.index("--scores")
        for s in argv[i + 1:]:
            if s.startswith("--"):
                break
            if not os.path.exists(s.partition("=")[2]):
                problems.append("scores " + s)
        i = argv.index("--scores9")
        for s in argv[i + 1:]:
            if s.startswith("--"):
                break
            if not os.path.exists(s.partition("=")[2]):
                problems.append("scores9 " + s)
        g = json.load(open(val(argv, "--gates"), encoding="utf-8"))
        if sorted(g) != sorted(["I1", "I2", "I3", "I4", "I5", "I6", "D1", "G14", "G15", "launch_gate"]) or not all(g.values()):
            problems.append("gates " + json.dumps(g))
        for fl in ("--families", "--families-per-set", "--bank9", "--sy"):
            if not os.path.exists(val(argv, fl) or ""):
                problems.append(fl)
        if problems:
            print("READ REFUSED", problems)
            return 3
        seq = sp.get("read_sequence") or [{"row": 3, "name": "SCREEN-GO", "full_arm": "GO", "next_lever": "amend the draft"}]
        write(val(argv, "--out"), json.dumps({"outcome": seq[min(n, len(seq)) - 1]}))
        return 0
    return 9
'''

STUB = "import os, sys\nsys.path.insert(0, os.environ['STUB_DIR'])\nimport stublib\nsys.exit(stublib.main(sys.argv[1], sys.argv[2:]))\n"
FAKE = "import os, sys\nsys.path.insert(0, os.environ['STUB_DIR'])\nimport stublib\nsys.exit(stublib.main({tool!r}, sys.argv[1:]))\n"


def check(bad, name, fn):
    try:
        ok = bool(fn())
    except Exception as e:                                            # noqa: BLE001 -- an exception IS a failed check
        ok = False
        if os.environ.get("HNT_DEBUG"):
            import traceback
            traceback.print_exc()
            print("  [check raised]", name, type(e).__name__, str(e)[:300])
    if not ok:
        bad.append(name)


def load_module(src, tag):
    mod = types.ModuleType(f"hnscreen_devbox_{tag}")
    mod.__file__ = SRC_PATH
    sys.modules[mod.__name__] = mod
    exec(compile(src, f"hnscreen_devbox_{tag}.py", "exec"), mod.__dict__)    # noqa: S102 -- the module under test, possibly mutated
    return mod


class Clk:
    def __init__(self, hook=None):
        self.t, self.sleeps, self.hook = 1000.0, 0, hook

    def time(self):
        return self.t

    def sleep(self, s):
        self.sleeps += 1
        self.t += s
        if self.hook:
            self.hook(self.sleeps)


class World:
    """one temp tree for the whole suite: the stub library, the fake package the chain's CLIs live in"""

    def __init__(self):
        self.base = tempfile.mkdtemp(prefix="hnchain_")
        self.stub_dir = os.path.join(self.base, "stubdir")
        os.makedirs(self.stub_dir)
        open(os.path.join(self.stub_dir, "stublib.py"), "w", encoding="utf-8").write(STUBLIB)
        self.stub = os.path.join(self.stub_dir, "stub.py")
        open(self.stub, "w", encoding="utf-8").write(STUB)
        self.fake = os.path.join(self.base, "fakepkg")
        for rel, tool in (("refe/hn_bank.py", "hn_bank"), ("eval/hn_screen_families.py", "families"), ("eval/hn_screen_read.py", "read")):
            p = os.path.join(self.fake, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "w", encoding="utf-8").write(FAKE.format(tool=tool))
        os.environ["STUB_DIR"] = self.stub_dir
        self.n = 0

    def close(self):
        shutil.rmtree(self.base, ignore_errors=True)


class Sc:
    """one scenario: its own state dir, journal, spec, lock and marker"""

    def __init__(self, world, name):
        world.n += 1
        self.w = world
        self.d = os.path.join(world.base, f"sc{world.n}_{name}").replace("\\", "/")
        os.makedirs(self.d)
        self.root, self.lock, self.marker = f"{self.d}/state", f"{self.d}/gpu/devbox_gpu.lock", f"{self.d}/gpu/devbox_gpu.chain_active"
        os.makedirs(f"{self.d}/gpu")
        self.journal, self.specp, self.stp = f"{self.d}/journal.jsonl", f"{self.d}/spec.json", f"{self.d}/stubstate"
        self.set_spec({})
        os.environ.update({"STUB_JOURNAL": self.journal, "STUB_SPEC": self.specp, "STUB_STATE": self.stp, "STUB_LOCK": self.lock})

    def set_spec(self, spec):
        json.dump(spec, open(self.specp, "w", encoding="utf-8"))

    def activate(self):
        os.environ.update({"STUB_JOURNAL": self.journal, "STUB_SPEC": self.specp, "STUB_STATE": self.stp, "STUB_LOCK": self.lock})

    def J(self):
        if not os.path.exists(self.journal):
            return []
        return [json.loads(x) for x in open(self.journal, encoding="utf-8") if x.strip()]

    def labels(self):
        return [r["label"] for r in self.J()]

    def count(self, label):
        return self.labels().count(label)


def template(extra=None, drop=None):
    t = ["--ckpt", "CKPT", "--backbone", "vitl16", "--out", "{out}", "--require-pdm", "--lam", "0", "--seed", "{seed}", "--dump-scores", "{scores}", "--lane-head",
         "--hn-bank", "{bank_train}", "--ctx-cache", "{cache_dir}"]
    if drop:
        i = t.index(drop)
        del t[i:i + 2]
    return t + (extra or [])


def make_cfg(sc, **over):
    w = lambda p, obj: ["--write", f"{p}={json.dumps(obj)}"]            # noqa: E731
    stub = sc.w.stub

    def st(name, *rest):
        return [PY, stub, "step", name, *rest]
    ok = {"ok": True}
    cfg = {
        "python": PY, "pkg": sc.w.fake,
        "gpu": {"lock": sc.lock, "chain_marker": sc.marker, "wait_idle": None, "idle_timeout_s": 1, "poll_s": 10, "max_wait_s": 100, "settle_s": 0},
        "trainer": {"cmd": [PY, stub, "trainer"], "argv_template": template()},
        "gate": {"cmd": [PY, stub, "gate"], "profile": "refe_hnscreen", "key_file": KEYPATH, "arm_name_fmt": "S{seed}",
                 "ctx_mode": "content"},      # HNSCREEN-FP-COST: the stub gate has no anchor / end-verify, so this suite runs the chain in CONTENT mode = exactly the chain it always was (the anchored chain: test_hnscreen_ctx_chain.py)
        "paths": {"bank_train": f"{sc.d}/bank/train.hnbank", "bank_val": f"{sc.d}/bank/val.hnbank", "cache_dir": f"{sc.d}/cache"},
        "commands": {
            "s0_stage": st("s0_stage", *w("{s0}/staged_manifest.json", {"staged": 1})),
            "s1_build": st("s1_build", "--write", "{bank_train}=BANK", "--write", "{bank_val}=BANKV"),
            "s1_i1": st("i1", *w("{gates}/I1.json", ok)), "s1_i2": st("i2", *w("{gates}/I2.json", ok)),
            "s1_i3": st("i3", *w("{gates}/I3.json", ok)),
            "s2_cache": st("s2_cache", "--write", "{cache_dir}/manifest.json=MAN"),
            "s2_i6": st("i6", *w("{gates}/I6.json", ok)), "s2_g15": st("g15", *w("{gates}/G15.json", ok)),
            "post_pass": [{"name": n, "gate": g, "argv": st(n, "--seed", "{seed}", *w("{gates}/" + g + ".s{seed}.json", ok))} for n, g in (("i4", "I4"), ("i5", "I5"), ("d1", "D1"))],
            "s4_offsets": st("s4_offsets", "--arms", "{arms}", "--arms-dir", "{s4}", "--write", "{s4}/bank9.npz=B9"),
            "s4_prep": st("s4_prep", "--write", "{s4}/families_inputs.npz=FI", "--write", "{s4}/picks.npz=PK"),
            "next_lever": {"3": [{"name": "next_amend", "argv": st("next_amend")}]},
        },
        "s4": {"pdm_dir": f"{sc.d}/pdm", "rl_val_scenes": f"{sc.d}/scenes.tsv", "base_scores": f"{sc.d}/base.npz"},
        "guards": {"min_free_gb": 0, "min_free_ram_mb": 0},
    }
    for k, v in over.items():
        cur = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        if v is None:
            cur.pop(parts[-1], None)
        else:
            cur[parts[-1]] = v
    open(f"{sc.d}/base.npz", "w").write("base")
    return cfg


def run_chain(M, sc, *, spec=None, stop_after=None, hook=None, used=0, cfg_over=None, cfg_doc=None, guard_gb=None, max_wait=None):
    sc.activate()
    if spec is not None:
        sc.set_spec(spec)
    doc = cfg_doc or make_cfg(sc, **(cfg_over or {}))
    cp = f"{sc.d}/config.json"
    json.dump(doc, open(cp, "w", encoding="utf-8"))
    cfg = M.Cfg.load(cp)
    clk = Clk(hook)
    kw = {"printer": lambda *_a, **_k: None, "guard_gb": guard_gb or (lambda p: 1e9), "guard_ram": lambda: 10 ** 9, "sleep": lambda s: None}
    ch = M.Chain(cfg, sc.root, **kw)
    ch._door = M.GpuDoor(lock=sc.lock, marker=sc.marker, landed=lambda **k: M.landed_gpu_lock(REAL_PKG, **k),
                         landed_kw={"lock": sc.lock, "wait_idle": None, "wait_timeout_s": 1.0, "settle_s": 0.0, "used_fn": (lambda: used)},
                         poll_s=10.0, max_wait_s=max_wait if max_wait is not None else float(doc["gpu"]["max_wait_s"]), clock=clk.time, sleep=clk.sleep, log=ch.log)
    rc = ch.run(stop_after=stop_after)
    return rc, ch, clk


def expected_labels(seeds=(0, 1, 2), rounds=3, next_lever=True):
    L = ["s0_stage", "s1_build", "hn_bank:yield", "i1", "i2", "i3"] + ["s2_cache"] * rounds + ["i6", "g15"]
    for s in seeds:
        L += [f"trainer:pre:{s}", f"gate:run:{s}", f"gate:verify:{s}"]
    for s in seeds:
        L += [f"gate:verify:{s}", f"trainer:train:{s}", f"i4:{s}", f"i5:{s}", f"d1:{s}"]
    return L + ["s4_offsets", "s4_prep", "families", "read"] + (["next_amend"] if next_lever else [])


PARTIAL2 = {"s2_cache": {"print_by_call": [["ZZCTX_PARTIAL"], ["ZZCTX_PARTIAL"], ["ZZCTX_DONE"]]}}


def full(M, w, name, spec=None, **kw):
    sc = Sc(w, name)
    rc, ch, clk = run_chain(M, sc, spec={**PARTIAL2, **(spec or {})}, **kw)
    return sc, rc, ch, clk


def all_files(root):
    for dp, _dn, fn in os.walk(root):
        for f in fn:
            yield os.path.join(dp, f)


def me(rec):
    return rec["lock"] is not None and rec["lock"].get("pid") == os.getpid() and str(rec["lock"].get("owner", "")).startswith("hnscreen_devbox")


STAGES = ["S0", "S1", "S2", "T0", "T1", "T2", "P0", "P1", "P2", "S4", "VERDICT", "NEXT"]


# ======================================================================================================== the checks
def A_stage_order(M, w):
    sc, rc, ch, _ = full(M, w, "A")
    return (rc == 0 and ch.stage_order_run == STAGES and sc.labels() == expected_labels()
            and all(os.path.exists(f"{sc.root}/stage/{s}.done.json") for s in STAGES)
            and json.load(open(f"{sc.root}/outcome.json"))["row"] == 3 and os.path.exists(f"{sc.root}/NEXT_LEVER.md"))


def B_handoffs_and_no_key_leak(M, w):
    sc, rc, ch, _ = full(M, w, "B")
    ok = rc == 0
    for i, s in enumerate(STAGES):
        p = f"{sc.root}/handoffs/HANDOFF_{s}.md"
        t = open(p, encoding="utf-8").read() if os.path.exists(p) else ""
        ok = ok and "NEXT:" in t and "RESUME" in t and len(t) < 6000 and f"HANDOFF {s}: DONE" in t
        if i + 1 < len(STAGES) - 1:
            ok = ok and f"NEXT: stage {STAGES[i + 1]}" in t
    leaked = [p for p in all_files(sc.root) if KEYPATH in open(p, "rb").read().decode("utf-8", "replace")]
    return ok and not leaked and os.path.exists(f"{sc.root}/HANDOFF.md")


def C_resume_after_interruption(M, w):
    sc = Sc(w, "C")
    spec = {**PARTIAL2, "trainer": {"fail": {"1": {"partial": True, "calls": 1}}}}
    rc1, ch1, _ = run_chain(M, sc, spec=spec)
    lock_gone = not os.path.exists(sc.lock)
    done1 = [s for s in STAGES if os.path.exists(f"{sc.root}/stage/{s}.done.json")]
    sc.set_spec(PARTIAL2)
    rc2, ch2, _ = run_chain(M, sc)
    out1 = f"{sc.root}/passes/s1/out"
    archived = [x for x in os.listdir(out1) if x.startswith("scores.attempt")] if os.path.isdir(out1) else []
    c = sc.labels().count
    return (rc1 == 1 and lock_gone and done1 == STAGES[:7] and rc2 == 0 and ch2.stage_order_run == ["P1", "P2", "S4", "VERDICT", "NEXT"]
            and c("s0_stage") == 1 and c("s1_build") == 1 and c("hn_bank:yield") == 1 and c("s2_cache") == 3 and c("trainer:pre:0") == 1 and c("trainer:pre:1") == 1
            and c("trainer:train:0") == 1 and c("trainer:train:1") == 2 and c("trainer:train:2") == 1 and len(archived) == 1
            and os.path.exists(f"{sc.root}/passes/s1/out/scores/scores_update1000_B.npz") and not os.path.exists(f"{sc.root}/passes/s1/out/scores/partial.tmp"))


def D_resume_by_content(M, w):
    sc = Sc(w, "D")
    rc1, ch1, _ = run_chain(M, sc, spec=PARTIAL2)
    n1 = len(sc.J())
    rc2, ch2, _ = run_chain(M, sc)                                    # nothing changed: nothing re-runs
    untouched = rc2 == 0 and ch2.stage_order_run == [] and len(sc.J()) == n1
    p = f"{sc.root}/s0/staged_manifest.json"
    data = open(p, "rb").read()
    open(p, "wb").write(data.replace(b"1", b"2"))                     # same size, other bytes
    rc3, ch3, _ = run_chain(M, sc)
    return rc1 == 0 and untouched and rc3 == 0 and "S0" in ch3.stage_order_run and sc.count("s0_stage") == 2


def E_no_trainer_without_a_token(M, w):
    res = {}
    for nm, spec in (("no_token", {"gate_run": {"no_token": True}}), ("refuse", {"verify": {"refuse_from_call": 1}}), ("no_marker", {"verify": {"no_marker": True}})):
        sc, rc, ch, _ = full(M, w, "E_" + nm, spec)
        res[nm] = rc == 5 and not any(x.startswith("trainer:train") for x in sc.labels()) and not os.path.exists(sc.lock)
    sc, rc, ch, _ = full(M, w, "E_start1", {"verify": {"refuse_from_call": 5}})       # T0-T2 mint verifies are calls 1-3, P0's start check is 4, P1's is 5
    L = sc.labels()
    res["start1"] = rc == 5 and L.count("trainer:train:0") == 1 and "trainer:train:1" not in L and "trainer:train:2" not in L
    sc, rc, ch, _ = full(M, w, "E_adj")
    J = sc.J()
    res["adjacent"] = rc == 0 and all(J[i - 1]["label"] == f"gate:verify:{r['seed']}" for i, r in enumerate(J) if r["label"].startswith("trainer:train"))
    res["same_argv"] = all(r["argv"] == json.load(open(f"{sc.root}/passes/s{r['seed']}/argv.json", encoding="utf-8")) for r in J if r["label"].startswith("trainer:train"))
    return all(res.values())


def F_lock_discipline(M, w):
    sc, rc, ch, _ = full(M, w, "F")
    J = sc.J()
    held = [r for r in J if r["label"].startswith("trainer:") or r["label"] in ("i6", "g15", "s4_offsets") or r["label"].split(":")[0] in ("i4", "i5", "d1")]
    starts = [i for i, r in enumerate(J) if r["label"].startswith("trainer:train")]
    cpu = [r for r in J if r["label"] in ("s0_stage", "s1_build", "i1", "i2", "gate:run:0", "s4_prep", "families", "read")]
    return (rc == 0 and bool(held) and all(me(r) for r in held) and all(me(J[i - 1]) for i in starts)
            and all(r["lock"] is None for r in J if r["label"] == "s2_cache") and all(r["lock"] is None for r in cpu) and not os.path.exists(sc.lock))


def _sleeper():
    return subprocess.Popen([PY, "-c", "import time; time.sleep(300)"])


def G_foreign_lock_is_waited_for_never_removed(M, w):
    res = {}
    sc = Sc(w, "G1")
    p = _sleeper()
    try:
        body = json.dumps({"job": "refcv8-battery", "pid": p.pid, "win_pid_of_locker": 1, "acquired": "2026-10-18T14:56:21+0200"})
        open(sc.lock, "w", encoding="utf-8").write(body)
        seen = {"bodies": [], "removed": None}

        def hook(n):
            seen["bodies"].append(open(sc.lock, encoding="utf-8").read() if os.path.exists(sc.lock) else None)
            if n == 3:
                seen["removed"] = os.path.exists(sc.lock)                      # the foreign JOB finishes (the test plays it): the file must still be there to be removed
                if seen["removed"]:
                    os.remove(sc.lock)
        rc, ch, clk = run_chain(M, sc, spec=PARTIAL2, hook=hook)
        J = sc.J()
        i6 = [r for r in J if r["label"] == "i6"]
        yields = open(ch.chain_log, encoding="utf-8").read().count("ZZHNCHAIN_YIELD")
        res["g1"] = (rc == 0 and seen["removed"] is True and seen["bodies"][:2] == [body, body] and yields >= 3 and len(i6) == 1 and me(i6[0]) and sc.labels() == expected_labels())
    finally:
        p.terminate()
    sc = Sc(w, "G2")                                                           # a dead holder (no such pid), 2 h old: the landed helper would remove it; the chain must not
    old = json.dumps({"job": "navsim", "pid": 4000001})
    open(sc.lock, "w", encoding="utf-8").write(old)
    os.utime(sc.lock, (time.time() - 7200, time.time() - 7200))
    mt = os.path.getmtime(sc.lock)
    rc, ch, clk = run_chain(M, sc, spec=PARTIAL2)
    res["g2"] = (rc == 75 and open(sc.lock, encoding="utf-8").read() == old and abs(os.path.getmtime(sc.lock) - mt) < 1 and "i6" not in sc.labels()
                 and not any(x.startswith("trainer") for x in sc.labels()) and "YIELDED" in open(f"{sc.root}/HANDOFF.md", encoding="utf-8").read())
    sc = Sc(w, "G4")                                                           # the chain_active marker
    open(sc.marker, "w", encoding="utf-8").write(json.dumps({"chain": "refcv8"}))
    st = {"removed": None}

    def hook4(n):
        if n == 2:
            st["removed"] = os.path.exists(sc.marker)
            os.remove(sc.marker)
    rc, ch, clk = run_chain(M, sc, spec=PARTIAL2, hook=hook4)
    res["g4"] = rc == 0 and st["removed"] is True and sc.labels() == expected_labels()
    sc = Sc(w, "G6")                                                           # a busy GPU (memory.used above the bar after taking the lock): the landed helper gives the lock back, the chain retries to its deadline
    rc, ch, clk = run_chain(M, sc, spec=PARTIAL2, used=9999)
    res["g6"] = rc == 75 and not os.path.exists(sc.lock) and "i6" not in sc.labels()
    return all(res.values())


def G_control_landed_helper_alone_removes_such_a_lock(M, w):
    """the fixture can SEE a removal, and the landed helper no longer steals (LOCK-STEAL fix, TRAPS "TWO LOCK SCHEMAS SHARE devbox_gpu.lock", 2026-10-18).
    (a) the landed acquire, called directly on the G2 lock (a FOREIGN schema, dead pid, 2 h old), raises GpuBusy and leaves the lock byte-for-byte: until 2026-10-18 it removed it,
    which is what this control used to pin; (b) on a lock carrying the helper's OWN owner marker, dead pid, 2 h old, it removes it and takes the lock, so the fixture still sees a
    removal.  The chain's pre-check stays as a second guard."""
    sys.path.insert(0, os.path.join(REAL_PKG, "eval"))
    import aux_navtest_forward as ANF
    sc = Sc(w, "Gc")
    old = json.dumps({"job": "navsim", "pid": 4000001})
    open(sc.lock, "w", encoding="utf-8").write(old)
    os.utime(sc.lock, (time.time() - 7200, time.time() - 7200))
    try:
        ANF.acquire_gpu_lock(lock=sc.lock, wait_idle=None, used_fn=lambda: 0, settle_s=0.0)
        refused = False
    except ANF.GpuBusy:
        refused = True
    kept = refused and open(sc.lock, encoding="utf-8").read() == old
    os.remove(sc.lock)                                                         # the test plays the foreign job finishing
    open(sc.lock, "w", encoding="utf-8").write(json.dumps({"pid": 4000001, "owner": ANF.OWNER, "t": time.time() - 7200}))
    os.utime(sc.lock, (time.time() - 7200, time.time() - 7200))
    body = ANF.acquire_gpu_lock(lock=sc.lock, wait_idle=None, used_fn=lambda: 0, settle_s=0.0)
    mine = json.loads(open(sc.lock, encoding="utf-8").read()).get("pid") == os.getpid()
    ANF.release_gpu_lock(sc.lock, body)
    return kept and mine and not os.path.exists(sc.lock)


def H_families_read_integration_and_rule_e(M, w):
    res = {}
    sc, rc, ch, _ = full(M, w, "H1")
    J = {r["label"]: r for r in sc.J()}
    fam, rd = J["families"]["argv"], J["read"]["argv"]
    P = fam[fam.index("--per-set-out") + 1]
    sc_names = [x.partition("=")[0] for x in rd[rd.index("--scores") + 1:rd.index("--pdm")]]
    res["argv"] = (rc == 0 and rd[rd.index("--families-per-set") + 1] == P and "--families" in rd and "--gates" in rd and rd[rd.index("--rule") + 1] == "navsim_v1_lane"
                   and sc_names == ["base", "A0", "B0", "A1", "B1", "A2", "B2"] and "next_amend" in sc.labels()
                   and "amend the draft" in open(f"{sc.root}/NEXT_LEVER.md", encoding="utf-8").read())
    pend = {"row": None, "name": "PENDING (rule E)", "reason": "Z1 holds at seeds 0-2 but Z2 or Z3 fails"}
    go = {"row": 3, "name": "SCREEN-GO", "full_arm": "GO", "next_lever": "amend the draft"}
    sc, rc, ch, _ = full(M, w, "H2", {"read_sequence": [pend, go]})
    L = sc.labels()
    rd2 = [r for r in sc.J() if r["label"] == "read"]
    nm2 = [x.partition("=")[0] for x in rd2[1]["argv"][rd2[1]["argv"].index("--scores") + 1:rd2[1]["argv"].index("--pdm")]] if len(rd2) == 2 else []
    res["rule_e"] = (rc == 0 and all(L.count(f"trainer:train:{s}") == 1 for s in range(5)) and len(rd2) == 2 and nm2 == ["base"] + [f"{a}{s}" for s in range(5) for a in "AB"]
                     and os.path.exists(f"{sc.root}/rule_e.json") and ch.stage_order_run[-7:] == ["T3", "T4", "P3", "P4", "S4", "VERDICT", "NEXT"])
    sc, rc, ch, _ = full(M, w, "H3", {"read_sequence": [pend, pend]})
    o = json.load(open(f"{sc.root}/outcome.json"))
    res["once"] = rc == 2 and o["row"] == 0 and "rule E already fired" in o["reason"] and sc.labels().count("trainer:train:3") == 1 and sc.labels().count("read") == 2
    return all(res.values())


def I_stop_points(M, w):
    res = {}
    sc, rc, ch, _ = full(M, w, "I1", {"sparse": True})
    o = json.load(open(f"{sc.root}/outcome.json"))
    res["sparse"] = rc == 3 and o["row"] == 1 and "s2_cache" not in sc.labels() and not any(x.startswith("trainer") for x in sc.labels())
    sc, rc, ch, _ = full(M, w, "I2", {"sparse": True, "i1": {"ok": False}})
    o = json.load(open(f"{sc.root}/outcome.json"))
    res["invalid_before_sparse"] = rc == 2 and o["row"] == 0 and "s2_cache" not in sc.labels()
    sc, rc, ch, _ = full(M, w, "I3", {"i6": {"ok": False}})
    res["i6"] = rc == 2 and not any(x.startswith("trainer") for x in sc.labels()) and json.load(open(f"{sc.root}/outcome.json"))["row"] == 0
    sc, rc, ch, _ = full(M, w, "I4", {"i5:1": {"ok": False}})
    res["post_pass"] = rc == 2 and "s4_offsets" not in sc.labels() and "read" not in sc.labels() and sc.labels().count("trainer:train:2") == 1
    sc, rc, ch, _ = full(M, w, "I5", {"s2_cache": {"print_by_call": [["ZZCTX_PARTIAL"]]}}, cfg_over={"cache.max_rounds": 2})
    res["cache_never_done"] = rc == 1 and "i6" not in sc.labels()
    return all(res.values())


def J_named_parameters_and_levers(M, w):
    res = {}
    for nm, t in (("no_hn_bank", template(drop="--hn-bank")), ("doubled_seed", template(["--seed", "9"])), ("no_ctx", template(drop="--ctx-cache")),
                  ("epo", template(["--epo"])), ("lam", template()[:template().index("--lam") + 1] + ["0.5"] + template()[template().index("--lam") + 2:]),
                  ("wrong_bank", [("/elsewhere/train.hnbank" if a == "{bank_train}" else a) for a in template()])):
        sc = Sc(w, "J_" + nm)
        rc, ch, _ = run_chain(M, sc, cfg_over={"trainer.argv_template": t})
        res[nm] = rc == 4 and sc.J() == []
    sc = Sc(w, "J_missing")
    rc, ch, _ = run_chain(M, sc, cfg_over={"commands.s2_cache": None})
    res["missing_param"] = rc == 4 and sc.J() == []
    sc = Sc(w, "J_unknown_ph")
    d = make_cfg(sc)
    d["commands"]["s0_stage"] = [PY, w.stub, "step", "s0_stage", "--write", "{nowhere}/x=1"]
    rc, ch, _ = run_chain(M, sc, cfg_doc=d)
    res["unknown_placeholder"] = rc == 4 and sc.J() == []
    return all(res.values())


def K_config_frozen_single_instance_and_disk(M, w):
    res = {}
    sc = Sc(w, "K1")
    rc1, ch, _ = run_chain(M, sc, stop_after="S0")
    d = make_cfg(sc)
    d["trainer"]["argv_template"] = template(["--extra-lever", "1"])
    rc2, ch, _ = run_chain(M, sc, cfg_doc=d)
    res["frozen"] = rc1 == 0 and rc2 == 4 and sc.count("s0_stage") == 1
    sc = Sc(w, "K2")
    p = _sleeper()
    try:
        os.makedirs(sc.root, exist_ok=True)
        open(f"{sc.root}/chain.pid", "w").write(str(p.pid))
        rc, ch, _ = run_chain(M, sc, stop_after="S0")
        res["live_instance"] = rc == 7 and sc.J() == [] and os.path.exists(f"{sc.root}/chain.pid")
    finally:
        p.terminate()
    sc = Sc(w, "K3")
    os.makedirs(sc.root, exist_ok=True)
    open(f"{sc.root}/chain.pid", "w").write("4000001")
    rc, ch, _ = run_chain(M, sc, stop_after="S0")
    res["dead_instance_taken_over"] = rc == 0 and sc.count("s0_stage") == 1 and not os.path.exists(f"{sc.root}/chain.pid")
    sc = Sc(w, "K4")
    rc, ch, _ = run_chain(M, sc, cfg_over={"guards.min_free_gb": 5.0}, guard_gb=lambda p: 1.0)
    res["low_disk"] = rc == 6 and sc.J() == [] and "low disk" in open(f"{sc.root}/HANDOFF.md", encoding="utf-8").read()
    sc = Sc(w, "K5")                                                           # the CLI: `skeleton` writes every named parameter, `plan` reads a finished state dir without running anything
    rc, ch, _ = run_chain(M, sc, spec=PARTIAL2)
    n_before = len(sc.J())
    buf, sk = io.StringIO(), f"{sc.d}/skeleton.json"
    with contextlib.redirect_stdout(buf):
        r1 = M.main(["skeleton", "--out", sk])
        r2 = M.main(["plan", "--config", f"{sc.d}/config.json", "--root", sc.root])
    res["cli"] = (rc == 0 and r1 == 0 and r2 == 0 and "argv_template" in json.load(open(sk, encoding="utf-8"))["trainer"] and "missing named parameters: none" in buf.getvalue()
                  and buf.getvalue().count("DONE") == len(STAGES) and len(sc.J()) == n_before)
    return all(res.values())


CHECKS = [("A_stage_order_and_markers", A_stage_order), ("B_handoff_per_stage_and_no_key_leak", B_handoffs_and_no_key_leak), ("C_resume_after_interruption", C_resume_after_interruption),
          ("D_resume_by_content_not_presence", D_resume_by_content), ("E_no_trainer_without_a_token", E_no_trainer_without_a_token), ("F_lock_discipline", F_lock_discipline),
          ("G_foreign_lock_waited_for_never_removed", G_foreign_lock_is_waited_for_never_removed), ("G_control_landed_helper_alone_removes_a_stale_lock", G_control_landed_helper_alone_removes_such_a_lock),
          ("H_families_read_integration_and_rule_e", H_families_read_integration_and_rule_e), ("I_stop_points", I_stop_points), ("J_named_parameters_and_levers", J_named_parameters_and_levers),
          ("K_config_frozen_single_instance_and_disk", K_config_frozen_single_instance_and_disk)]


def suite(M, tag, only=None):
    w = World()
    bad = []
    try:
        for name, fn in CHECKS:
            if only is None or name in only:
                check(bad, name, lambda fn=fn: fn(M, w))
    finally:
        w.close()
    return bad


def stat_of(p):
    try:
        s = os.stat(p)
        return (s.st_size, s.st_mtime_ns)
    except OSError:
        return None


# each mutation: (name, [(old, new)], [checks that must go RED])
MUTATIONS = [
    ("the token check is deleted (verify never refuses)", [("        ok = r[0] == 0 and MATCH in r[1]\n", "        ok = True\n")], ["E_no_trainer_without_a_token"]),
    ("the token check before the trainer start is deleted", [("            argv = self.verify_pass_token(seed, where=\"start\")[1]", "            argv = self.pass_argv(seed)")], ["E_no_trainer_without_a_token"]),
    ("the GPU lock acquisition is deleted", [("                    st.enter_context(self.landed(owner=f\"{OWNER}:{job}\", **self.landed_kw))\n", "                    pass\n")], ["F_lock_discipline", "G_foreign_lock_waited_for_never_removed"]),
    ("the chain REMOVES a foreign lock while it waits", [("            self.log(f\"ZZHNCHAIN_YIELD poll {n}: {f} -- NOT removed\")\n", "            self.log(f\"ZZHNCHAIN_YIELD poll {n}: {f} -- NOT removed\")\n            os.remove(self.lock)\n")], ["G_foreign_lock_waited_for_never_removed"]),
    ("the chain ignores the chain_active marker", [("        for p, what in ((self.lock, \"lock\"), (self.marker, \"chain_active\")):\n", "        for p, what in ((self.lock, \"lock\"),):\n")], ["G_foreign_lock_waited_for_never_removed"]),
    ("a DONE check is skipped (every stage re-runs)", [("        if self.marks.is_done(sid, fp):\n", "        if False:\n")], ["C_resume_after_interruption", "D_resume_by_content_not_presence"]),
    ("resume trusts presence, not content", [("if not os.path.isfile(p) or sha256_file(p) != sha:", "if not os.path.isfile(p):")], ["D_resume_by_content_not_presence"]),
    ("the read is not given --families-per-set", [("\"--families-per-set\", self.per_set, ", "")], ["H_families_read_integration_and_rule_e"]),
    ("the chain holds the lock around the cache driver", [("            text = self.must(self.cmd(\"s2_cache\"), \"S2_cache\", lock=None)", "            text = self.must(self.cmd(\"s2_cache\"), \"S2_cache\", lock=\"S2\")")], ["F_lock_discipline"]),
    ("SY < 0.10 does not stop the chain", [("            if sparse is True:\n", "            if False:\n")], ["I_stop_points"]),
    ("a failed I6 / G15 does not stop the chain", [("            bad = [g for g in (\"I6\", \"G15\") if not ev_ok(self.gate_path(g))]\n", "            bad = []\n")], ["I_stop_points"]),
    ("the argv lever check is skipped", [("        if lever_issues:\n", "        if False:\n")], ["J_named_parameters_and_levers"]),
    ("rule E may fire twice", [("            if os.path.exists(self.rule_e_path):\n                raise ChainStop(RC_INVALID, \"rule E already fired", "            if False:\n                raise ChainStop(RC_INVALID, \"rule E already fired")], ["H_families_read_integration_and_rule_e"]),
    ("a failed gate before the read does not stop it", [("        bad = sorted(k for k, v in gates.items() if not v)\n", "        bad = []\n")], ["I_stop_points"]),
    ("the trainer is not released-and-archived on a retry (stale scores stay)", [("            if os.path.exists(p):\n                os.replace(p, f\"{p}.attempt{int(self.clock())}\")\n", "            pass\n")], ["C_resume_after_interruption"]),
]


def apply_edits(src, edits, name):
    for old, new in edits:
        n = src.count(old)
        if n != 1:
            raise AssertionError(f"mutation {name!r}: snippet occurs {n} times (need exactly 1): {old[:70]!r}")
        src = src.replace(old, new)
    return src


def main():
    before = (stat_of(SHARED_LOCK), stat_of(SHARED_MARK))
    src = open(SRC_PATH, encoding="utf-8").read()
    t0 = time.time()
    real = suite(load_module(src, "real"), "real")
    print(f"unmutated suite: {len(CHECKS)} checks, {len(real)} failing {real}  ({time.time() - t0:.0f} s)", flush=True)
    assert not real, f"the unmutated hnscreen_devbox.py FAILS its own checks: {real}"
    n, problems = 1, []
    print(f"{'mutation':72s} caught-by (failing checks)")
    for name, edits, expect in MUTATIONS:
        m_src = apply_edits(src, edits, name)
        try:
            ast.parse(m_src)
            failing = suite(load_module(m_src, f"m{n}"), f"m{n}", only=set(expect))
        except (Exception, SystemExit) as e:                           # noqa: BLE001 -- a mutated source that breaks the suite's own setup is RED, and said so
            failing = [f"suite setup raised {type(e).__name__}: {str(e)[:80]}"]
        if not failing:
            problems.append(f"mutation {name!r} stayed GREEN: the suite cannot see it")
            continue
        for c in expect:
            if c not in failing and not any(f.startswith("suite setup raised") for f in failing):
                problems.append(f"mutation {name!r}: the check {c} must go RED and stayed green (RED: {failing[:8]})")
        print(f"{name[:72]:72s} {', '.join(failing[:3])}", flush=True)
        n += 1
    after = (stat_of(SHARED_LOCK), stat_of(SHARED_MARK))
    if before != after:
        problems.append(f"N: the shared dev-box lock files changed during the suite: {before} -> {after}")
    if problems:
        print("\n".join("PROBLEM " + p for p in problems))
        print(f"ZZTEST_FAIL {len(problems)} problem(s)")
        return 1
    print(f"ZZTEST_PASS {n}  (1 unmutated suite of {len(CHECKS)} checks + {len(MUTATIONS)} RED mutations; the shared dev-box lock files untouched: {after})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
