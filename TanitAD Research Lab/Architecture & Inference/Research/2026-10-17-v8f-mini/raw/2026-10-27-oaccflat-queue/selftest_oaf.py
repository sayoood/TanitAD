#!/usr/bin/env python3
"""selftest_oaf.py -- the proof for the box-2 DAG-OACC / FLAT-DAG queue stages, helpers and DAG-OACC chain (PREREG_V8F_DAG_OACC, PREREG_V8F_FLATDAG). CPU only, no box 2, nothing launched.
CHECK = must hold. DELIBERATE REGRESSION = a defect put in on purpose that the instrument must CATCH (a regression that is not caught is a FAILURE of the proof).

  A  the REAL queue through cmd.exe (V8F_DRYRUN): dagoacc == DAG's REAL banked argv.json argv + `--w-acc W` (both seeds); flatdag / flatdag0 == FLAT's REAL banked argv minus the fine-tune flags, `--o-interface cut` named, the pair rule;
     waccdag, bankflat, evaldag for the new arms; refusals; the text diff of the queue is insertions only and is exactly what build_queue.py produces
  S  BYTE-IDENTITY of every existing stage: the tip queue (1d4f4fbb) and the staged queue print the same rc and lines on a sweep of EVERY dispatched stage name x 4 argument shapes; every pre-existing label differs from the tip only by insertions
  B  the REAL queue, NOT dry, on a fake box (stub trainer / eval / eval_oacc / queue tool): waccdag into %RUNSM%\\dagoacc only, REFUSED paths, the dagoacc pin (own file, DAG's pin untouched), traindag runs the pin, byte-identity
  C  the decisions on files: C-O3 (wacc-check), the F1 lever grep (init-match), the disk floor, the predecessor rule (pred-verdict), argv helpers
  D  the DAG-OACC chain script: --dry-run, --check-tree, --co3-only, --wait-only (the predecessor branches on a fake box dir), the disk floor through the REAL pull(), early stops without a box-2 call
  E  gate_oaf_assert.py on the REAL DAG corpus gate (selfcheck.json written by the real trainer on box 2), doctored into dagoacc / flatdag / flatdag0 + doctored negatives

    CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 python selftest_oaf.py [--sections A,S,B,C,D,E] [--out selftest_output.txt]
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oaccflat_lib as L  # noqa: E402

SHIP = Path(os.environ["OAF_SHIP"]) if os.environ.get("OAF_SHIP") else HERE / "box2_ship"           # the mutation runner points sections C / D / E at a MUTATED copy (all helper scripts beside the chain)
sys.path.insert(0, str(SHIP))
import v8f_oaf_chain_checks as CK  # noqa: E402
import gate_oaf_assert as GA  # noqa: E402

W = L.BT.parent
FW = Path(os.environ.get("OAF_FW", r"C:\Users\Admin\oqf"))                                           # short path for the fake boxes
BASH = r"C:\Program Files\Git\bin\bash.exe"
DAGART = Path("E:/Projects/TanitAD-artifacts/v8f-dagger")
OFFLINE = Path("E:/Projects/TanitAD-artifacts/dagoacc_dry/wacc_offline_DAG_s0.json")
PYTREE = W / "pytree"
MIRROR, TIP = "C:/Users/Admin/tanitad-push", "1d4f4fbb"
PK = L.PK
LOG = []
N = {"check": 0, "check_fail": 0, "reg": 0, "reg_missed": 0}
NEW_LABELS = {"dagoaccarm", "dagoaccarm_dry", "dagoaccarm_have", "dagoaccwacc", "dagoaccwacc_dry", "dagoaccwacc_have", "waccdag", "flatdagarm", "flatdagarm_dry", "flatdagarm_have", "evalflatdag", "evalflatdag_1", "evalflatdag_2", "bankflat"}


def say(s):
    LOG.append(s)
    print(s, flush=True)


def check(name, cond, detail=""):
    N["check"] += 1
    if not cond:
        N["check_fail"] += 1
    say(("PASS  " if cond else "FAIL  ") + name + (f"   [{str(detail)[:1800]}]" if (detail and not cond) else ""))
    return bool(cond)


def regress(name, caught, detail=""):
    N["reg"] += 1
    if not caught:
        N["reg_missed"] += 1
    say(("CAUGHT " if caught else "MISSED ") + "(deliberate regression) " + name + (f"   [{str(detail)[:1800]}]" if (detail and not caught) else ""))
    return bool(caught)


def posix(p) -> str:
    return str(p).replace("\\", "/")


def sh(cmd, env=None, cwd=None, timeout=600, inp=None):
    e = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": posix(W / "tmp"), **(env or {})}
    (W / "tmp").mkdir(parents=True, exist_ok=True)
    p = subprocess.run(cmd, capture_output=True, text=True, env=e, cwd=cwd, timeout=timeout, stdin=subprocess.DEVNULL, errors="replace")
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def jl(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def sha_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git_blob(path) -> bytes:
    return subprocess.run(["git", "-C", MIRROR, "-c", "safe.directory=*", "show", f"{TIP}:{path}"], capture_output=True, check=True).stdout


def nout(a):
    return L.drop_flag(a, {"--out"})


FLAT_FT = {"--out", "--init-ckpt", "--steps", "--lr", "--warmup", "--p-gt-start", "--p-gt-end", "--ego-perturb-p", "--dagger-p", "--dagger-bank", "--dagger-bank-sha256", "--dagger-seed"}
BANK1 = [L.flags(jl(DAGART / f"M2b_s{s}_dag" / "argv.json")["argv"])["--dagger-bank-sha256"][0][0] for s in (0, 1)]
FLATSHA = ["8496934a2ce262c128a3e975da156de2f40e422925893c9979169a4c114493cc", "6fdfc9b02ce87248c5315dd7604d3642a3b39f7ae0f593ec6b6ef5ef8df64c95"]
WACC = "49.6969"


def lines3(arm, seed, sha=None, wacc=None):
    p = ["0.1"] if arm in ("dag", "dag2", "dagoacc", "flatdag") else []
    g = L.dry(f"gatedag {arm} {seed} " + " ".join(p), sha=sha, wacc=wacc)
    t = L.dry(f"traindag {arm} {seed} " + " ".join(p), sha=sha, wacc=wacc)
    return L.trainer_line(g), L.trainer_line(t), L.dry(f"evaldag {arm} {seed}")


# ================================================================================================ A. rendering
def section_a():
    say("== A. the REAL queue through cmd.exe (V8F_DRYRUN)")
    for s in (0, 1):
        dagref = jl(DAGART / f"M2b_s{s}_dag" / "argv.json")["argv"]
        g, t, e = lines3("dagoacc", s, BANK1[s], WACC)
        ta, ga = L.train_argv(t), L.train_argv(g)
        check(f"A1 dagoacc s{s}: DAG's REAL banked argv.json argv + `--w-acc {WACC}` at the end, only --out differing (C-O1)", nout(ta) == nout(dagref) + ["--w-acc", WACC] and ta[ta.index("--out") + 1].endswith(f"M2b_s{s}_dagoacc"))
        check(f"A2 dagoacc s{s}: the gate line is the train line with --out and --selfcheck-only changed", nout([x for x in ga if x != "--selfcheck-only"]) == nout(ta) and ga.count("--selfcheck-only") == 1)
        check(f"A3 dagoacc s{s}: --w-acc named exactly once, init M2b_s{s}, the ITERATION-1 bank dir and DAG's bank sha", L.flags(ta)["--w-acc"] == [[WACC]] and L.flags(ta)["--init-ckpt"][0][0].endswith(f"M2b_s{s}\\ckpt_last.pt")
              and L.flags(ta)["--dagger-bank"][0][0].endswith(f"dagger_bank\\s{s}") and L.flags(ta)["--dagger-bank-sha256"] == [[BANK1[s]]])
        check(f"A4 dagoacc s{s}: eval = the standard T0 eval of M2b_s{s}_dagoacc, tag M2b-dagoacc_s{s}", len(e) == 1 and f"M2b_s{s}_dagoacc\\ckpt_last.pt" in e[0] and f"--tag M2b-dagoacc_s{s} " in e[0] and "--project --families" in e[0])
    t0 = L.trainer_line(L.dry("traindag dagoacc 0 0.1", sha=BANK1[0]))
    check("A5 no w_acc supplied: the dry run prints the placeholder WACCDAG_FROM_RULE (never a number)", "--w-acc WACCDAG_FROM_RULE" in t0)
    for a, why in (("gatedag dagoacc 0", "no P"), ("gatedag dagoacc 0 0.3", "P 0.3"), ("gatedag dagoacc 0 0", "P 0"), ("gatedag dagoacc 2 0.1", "seed 2"), ("gatedag dagoacc", "no seed")):
        rc, out = L.cmd(a, dryrun=True)
        check(f"A6 `{a}` ({why}) is REFUSED rc 2", rc == 2 and any("REFUSED" in x for x in out), (rc, out))
    # ---- FLAT-DAG
    for s in (0, 1):
        fref = json.loads(git_blob(f"{PK}/raw/2026-10-26-flat/runs/M2b_s{s}_flat/argv.json"))["argv"]
        g, t, e = lines3("flatdag", s, "cd" * 32)
        g0, t0_, e0 = lines3("flatdag0", s)
        ta, ta0 = L.train_argv(t), L.train_argv(t0_)
        check(f"A7 flatdag s{s}: FLAT's REAL banked argv minus the fine-tune / bank flags, `--o-interface cut` NAMED exactly once (every other flag is FLAT's)",
              L.flags(L.drop_flag(ta, FLAT_FT)) == L.flags(L.drop_flag(fref, FLAT_FT)) and L.flags(ta)["--o-interface"] == [["cut"]] and L.flags(L.drop_flag(ta, FLAT_FT))["--o-interface"] == [["cut"]])
        fl = L.flags(ta)
        check(f"A8 flatdag s{s}: init FLAT_s{s}, the FLAT bank dir, p 0.1, seed, bank sha", fl["--init-ckpt"][0][0].endswith(f"M2b_s{s}_flat\\ckpt_last.pt") and fl["--dagger-bank"][0][0].endswith(f"dagger_bank_flat\\s{s}")
              and fl["--dagger-p"] == [["0.1"]] and fl["--dagger-seed"] == [[str(s)]] and fl["--dagger-bank-sha256"] == [["cd" * 32]] and "--w-acc" not in fl)
        check(f"A9 flatdag0 s{s}: C-F2 -- differs from flatdag only in --out, --dagger-p's value and the three bank flags", L.drop_flag(ta0, {"--out", "--dagger-p"}) == L.drop_flag(ta, {"--out", "--dagger-p", "--dagger-bank", "--dagger-bank-sha256", "--dagger-seed"})
              and L.flags(ta0)["--dagger-p"] == [["0"]] and L.flags(ta0)["--o-interface"] == [["cut"]] and not (set(L.flags(ta0)) & {"--dagger-bank", "--dagger-bank-sha256", "--dagger-seed"}))
        check(f"A10 flatdag s{s}: eval = standard eval + the GT-code oracle (tacorc) of M2b_s{s}_flatdag", len(e) == 2 and "--project --families" in e[0] and e[1].count("--tac oracle") == 1 and f"--tag M2b-flatdag_s{s}-tacorc" in e[1] and len(e0) == 2)
    for a, why in (("gatedag flatdag 0", "no P"), ("gatedag flatdag 0 0.3", "P 0.3"), ("gatedag flatdag0 0 0.1", "flatdag0 with a P"), ("gatedag flatdag 3 0.1", "seed 3")):
        rc, out = L.cmd(a, dryrun=True)
        check(f"A11 `{a}` ({why}) is REFUSED rc 2", rc == 2 and any("REFUSED" in x for x in out), (rc, out))
    # ---- waccdag / bankflat
    wd, wo = L.dry("waccdag")[0], L.dry("waccoacc")[0]
    check("A12 waccdag: the rule's tool, DAG_s0's checkpoint with its sha256 pinned, banked fit windows of DAG_s0, everything in %RUNSM%\\dagoacc and NOTHING in OACC's %RUNSM%\\oacc",
          "M2b_s0_dag\\ckpt_last.pt" in wd and "--expect-sha256 d186fd0bcfa7ee9fbfe5c9de543fe85edda7c3e6b9c56c898958f027ae2b8cb1" in wd and "fit_windows_DAG_s0.npz" in wd and "\\dagoacc\\w_acc.json" in wd and "\\dagoacc\\w_acc.txt" in wd and "\\oacc\\" not in wd and "M2b_s0\\ckpt_last.pt" not in wd)
    check("A13 waccdag is waccoacc's rule: the same tool, subcommand and data arguments (only the checkpoint, the pin and the directory differ)",
          L.tokens(wo)[:3] == L.tokens(wd)[:3] and all(L.flags(L.tokens(wd))[k] == L.flags(L.tokens(wo))[k] for k in ("--cache-dir", "--join-train", "--tracks")))
    bd2, bf = L.dry("bankdag2 0")[0], L.dry("bankflat 0")[0]
    check("A14 bankflat 0 == bankdag2 0 with the generator DAG -> FLAT (M2b_s0_dag -> M2b_s0_flat, dagger_bank_dag2 -> dagger_bank_flat) and nothing else", bd2.replace("M2b_s0_dag", "M2b_s0_flat").replace("dagger_bank_dag2", "dagger_bank_flat") == bf and "M2b_s0\\" not in bf)
    for s in (0, 1):
        bfs = L.dry(f"bankflat {s}")[0]
        check(f"A15 bankflat {s}: generator FLAT_s{s} (ckpt and argv.json), bank dir dagger_bank_flat\\s{s}, the expectations of bankdag", f"M2b_s{s}_flat\\ckpt_last.pt" in bfs and f"M2b_s{s}_flat\\argv.json" in bfs and f"dagger_bank_flat\\s{s}" in bfs and "--expect-windows 306566" in bfs and "--device cuda" in bfs)
    # ---- the text diff
    import build_queue as BQ
    staged, tip = L.QUEUE.read_bytes(), L.TIP_QUEUE.read_bytes()
    check("A16 the staged queue is exactly what build_queue.py derives from the tip queue", staged == BQ.build() or os.environ.get("OAF_QUEUE") is not None)
    import difflib
    a, b = tip.decode().split("\r\n"), staged.decode().split("\r\n")
    ops = difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
    check("A17 the text diff against the tip queue is INSERTIONS ONLY (0 removed / replaced lines) and CRLF throughout", all(o[0] in ("equal", "insert") for o in ops) and staged.count(b"\r\n") == staged.count(b"\n") and staged.endswith(b"\r\n"), [o for o in ops if o[0] not in ("equal", "insert")][:3])
    lab = lambda x: set(re.findall(r"(?m)^:([A-Za-z0-9_]+)\r?$", x.decode()))
    check("A18 the new labels are exactly the 14 declared (and every old label is still there)", lab(staged) - lab(tip) == NEW_LABELS and lab(tip) <= lab(staged), sorted(lab(staged) - lab(tip)))
    disp = L.dispatched(staged.decode())
    check("A19 the two new stages are dispatched, and every dispatch target exists", {"waccdag", "bankflat"} <= set(disp) and all(d in lab(staged) for d in disp))


# ================================================================================================ S. byte-identity
def section_s():
    say("== S. byte-identity of every existing stage (tip 1d4f4fbb vs staged)")
    tip_text = L.TIP_QUEUE.read_text(encoding="utf-8")
    stages = L.dispatched(tip_text)
    shapes = ("", "0", "dag 0 0.1", "dag2 1")
    bad = []
    t0 = time.time()
    for st in stages:
        for sh_ in shapes:
            args = f"{st} {sh_}".strip()
            r1 = L.cmd(args, dryrun=True, queue=L.TIP_QUEUE)
            r2 = L.cmd(args, dryrun=True, queue=L.QUEUE)
            if r1 != r2:
                bad.append((args, r1[0], r2[0]))
    check(f"S1 all {len(stages)} dispatched stages x {len(shapes)} argument shapes print byte-identical lines and rc ({time.time() - t0:.0f} s)", not bad, bad[:3])
    for a in ("gatedag dag 0 0.1", "traindag dag0 1", "evaldag dag 0", "evaldag dag0 1", "gatedag dag2 0 0.1", "traindag dag20 1", "evaldag dag2 0", "bankdag 0", "bankdag2 1", "waccoacc", "gateoacc", "trainoacc 0", "gateflat", "trainflat 1", "evalflat 0", "gatedag dag 0 0.3", "gatedag dag 0", "gatedag dag0 0 0.1"):
        sha = "ab" * 32 if "dag" in a else None
        r1, r2 = L.cmd(a, env={"DAGSHA_DRY": sha} if sha else {}, dryrun=True, queue=L.TIP_QUEUE), L.cmd(a, env={"DAGSHA_DRY": sha} if sha else {}, dryrun=True, queue=L.QUEUE)
        check(f"S2 `{a}` prints the same as on the tip", r1 == r2, (r1, r2))
    qa, qb = L.TIP_QUEUE.read_bytes().decode("utf-8").split("\r\n"), L.QUEUE.read_bytes().decode("utf-8").split("\r\n")      # bytes: read_text would translate CRLF and the split would find ONE line
    labels = lambda ls: {m.group(1): i for i, x in enumerate(ls) if (m := re.match(r"^:([A-Za-z0-9_]+)$", x))}
    la, lb = labels(qa), labels(qb)

    def block(ls, lab_d, name):
        order = sorted(lab_d.values()) + [len(ls)]
        i = lab_d[name]
        b = ls[i:order[order.index(i) + 1]]
        while b and b[-1] == "":                         # trailing blank lines (the file's final CRLF, or the separator before an appended label) are not a stage's text
            b.pop()
        return b
    changed = []
    for name in la:
        ba, bb = block(qa, la, name), block(qb, lb, name)
        if ba != bb:
            changed.append(name)
            import difflib
            ok = all(o[0] in ("equal", "insert") for o in difflib.SequenceMatcher(None, ba, bb, autojunk=False).get_opcodes())
            check(f"S3 pre-existing label :{name} differs from the tip only by inserted lines", ok)
    check("S4 only :dagname, :dagarm, :evaldag (hand-offs) changed among the pre-existing labels", sorted(changed) == ["dagarm", "dagname", "evaldag"], changed)


# ================================================================================================ B. fake box
FAKE_TRAINER = '''import json, os, sys
a = sys.argv[1:]
out = a[a.index("--out") + 1]
os.makedirs(out, exist_ok=True)
json.dump(a, open(os.path.join(out, "stub_argv.json"), "w"))
if "--selfcheck-only" in a:
    json.dump({"declared_vs_built": "PASS"}, open(os.path.join(out, "selfcheck.json"), "w"))
else:
    json.dump({"done": True}, open(os.path.join(out, "summary.json"), "w"))
    open(os.path.join(out, "ckpt_last.pt"), "wb").write(b"x")
'''
FAKE_EVALOACC = '''import json, os, sys
a = sys.argv[1:]
os.makedirs(os.path.dirname(a[a.index("--out") + 1]), exist_ok=True)
json.dump(a, open(a[a.index("--out") + 1] + ".stub_argv.json", "w"))
if os.environ.get("STUB_FAIL"):
    sys.exit(3)
json.dump({"w_acc": 49.6969}, open(a[a.index("--out") + 1], "w"))
open(a[a.index("--out-txt") + 1], "w", newline="\\n").write("49.6969\\n")
'''


FAKE_BANKTOOL = '''import json, os, runpy, sys
a = sys.argv[1:]
if a and a[0] == "bank":
    out = a[a.index("--out-dir") + 1]
    os.makedirs(out, exist_ok=True)
    json.dump(a, open(os.path.join(out, "tool_argv.json"), "w"))
    json.dump({"fake_bank": True}, open(os.path.join(out, "bank.json"), "w"))
    json.dump({"fake": True}, open(os.path.join(out, "PROVENANCE.json"), "w"))
    sys.exit(0)
sys.argv = [os.path.join(os.path.dirname(__file__), "v8f_dagger_queue_tool_real.py")] + a
runpy.run_path(sys.argv[0], run_name="__main__")
'''


def fake_world(root: Path) -> dict:
    if root.exists():
        shutil.rmtree(root)
    repo, cache, v9, runsm = root / "repo", root / "cache", root / "v9", root / "runs"
    sc = repo / "stack" / "scripts"
    sc.mkdir(parents=True)
    (sc / "v8f_mini_train.py").write_text(FAKE_TRAINER)
    (sc / "v8f_mini_eval_oacc.py").write_text(FAKE_EVALOACC)
    shutil.copy(PYTREE / "stack" / "scripts" / "v8f_dagger_queue_tool.py", sc / "v8f_dagger_queue_tool_real.py")
    (sc / "v8f_dagger_queue_tool.py").write_text(FAKE_BANKTOOL, newline="\n")                       # `bank` is recorded + faked; every other subcommand (the bank sha) is the REAL tool
    pk = repo / "TanitAD Research Lab" / "Architecture & Inference" / "Research"
    (pk / "2026-10-17-v8f-mini" / "raw" / "2026-10-25-hblend").mkdir(parents=True)
    (pk / "2026-10-17-v8f-mini" / "raw" / "2026-10-25-hblend" / "fit_clips.json").write_text("{}")
    (pk / "2026-10-17-v8f-mini" / "raw" / "eval_grid_windows.json").write_text("{}")             # :prepdag copies eval-141's grid
    hj = pk / "2026-10-17-refav1-hj-mini" / "raw" / "p10_yaw4512"
    hj.mkdir(parents=True)
    (hj / "p10_prior.pt").write_bytes(b"p")
    (hj / "p10_eval_pred6.npy").write_bytes(b"p")
    for d in ("train", "eval"):
        (cache / d).mkdir(parents=True)
        (cache / d / "manifest.jsonl").write_text("{}\n")
    (cache / "native_tracks.pt").write_bytes(b"t")
    v9.mkdir()
    for s in (0, 1):
        for nm in (f"M2b_s{s}", f"M2b_s{s}_dag"):
            (runsm / nm).mkdir(parents=True)
            (runsm / nm / "ckpt_last.pt").write_bytes(b"ckpt")
        (runsm / f"M2b_s{s}_dag" / "summary.json").write_text(json.dumps({"done": True}))
        (runsm / "dagger_bank" / f"s{s}").mkdir(parents=True)
        (runsm / "dagger_bank" / f"s{s}" / "bank.json").write_text(json.dumps({"fake": s}))
    (runsm / "join_train.npz").write_bytes(b"j")
    return {"REPO": str(repo), "PY": L.PY, "CACHEFULL": str(cache), "V9": str(v9), "RUNSM": str(runsm)}


def digest(root: Path) -> dict:
    return {str(p.relative_to(root)): sha_file(p) for p in sorted(root.rglob("*")) if p.is_file()} if root.exists() else {}


def section_b():
    say("== B. the real queue through cmd.exe, NOT dry, on a fake box")
    env = fake_world(FW / "box")
    runsm = Path(env["RUNSM"])
    # waccdag
    rc, out = L.cmd("waccdag", env=env, dryrun=False)
    check("B1 waccdag without the shipped banked fit windows is REFUSED rc 2 (nothing in OACC's directory)", rc == 2 and any("fit_windows_DAG_s0.npz is missing" in x for x in out) and not (runsm / "oacc").exists(), (rc, out))
    (runsm / "dagoacc").mkdir(exist_ok=True)
    (runsm / "dagoacc" / "fit_windows_DAG_s0.npz").write_bytes(b"npz")
    (runsm / "M2b_s0_dag" / "summary.json").rename(runsm / "M2b_s0_dag" / "summary.moved")
    rc, out = L.cmd("waccdag", env=env, dryrun=False)
    check("B2 waccdag with DAG_s0 unfinished (no summary.json) is REFUSED rc 2", rc == 2 and any("has not finished" in x for x in out), (rc, out))
    (runsm / "M2b_s0_dag" / "summary.moved").rename(runsm / "M2b_s0_dag" / "summary.json")
    rc, out = L.cmd("waccdag", env={**env, "STUB_FAIL": "1"}, dryrun=False)
    check("B3 a failing rule tool is reported [FAIL] rc 1 and leaves no w_acc.txt", rc == 1 and not (runsm / "dagoacc" / "w_acc.txt").exists(), (rc, out))
    rc, out = L.cmd("waccdag", env=env, dryrun=False)
    sa = jl(runsm / "dagoacc" / "w_acc.json.stub_argv.json")
    check("B4 waccdag runs the rule tool with DAG_s0's checkpoint, into %RUNSM%\\dagoacc, and the number lands in w_acc.txt", rc == 0 and (runsm / "dagoacc" / "w_acc.txt").read_text().strip() == WACC and sa[sa.index("--ckpt") + 1].endswith("M2b_s0_dag\\ckpt_last.pt")
          and sa[sa.index("--out") + 1].startswith(str(runsm / "dagoacc")) and (runsm / "dagoacc" / "fit_clips.json").exists() and not (runsm / "oacc").exists(), (rc, out))
    rc, out = L.cmd("waccdag", env=env, dryrun=False)
    check("B5 a second waccdag is skipped", rc == 0 and any("[skip]" in x for x in out))
    # gatedag / traindag dagoacc
    d0 = digest(runsm / "dagpin")
    rc, out = L.cmd("gatedag dagoacc 0 0.1", env=env, dryrun=False)
    pin = runsm / "dagpin" / "M2b_s0_dagoacc.train.argv.txt"
    gate_stub = jl(runsm / "gate_M2b-dagoacc_s0" / "stub_argv.json") if (runsm / "gate_M2b-dagoacc_s0" / "stub_argv.json").exists() else []
    check("B6 gatedag dagoacc pins the training argv in its OWN file and the gate ran with --w-acc <the file's number>, --selfcheck-only, the iteration-1 bank and its hash", rc == 0 and pin.exists() and "--w-acc 49.6969" in pin.read_text() and gate_stub[-1] == "--selfcheck-only"
          and gate_stub[gate_stub.index("--w-acc") + 1] == WACC and (runsm / "dagpin" / "bank_sha_dagoacc_s0.txt").read_text().strip() == hashlib.sha256(b'{"fake": 0}').hexdigest() and gate_stub[gate_stub.index("--dagger-bank") + 1].endswith("dagger_bank\\s0"), (rc, out))
    check("B7 DAG's own pin files are untouched (no dag pin written, no bank_sha_s0.txt)", not (runsm / "dagpin" / "M2b_s0_dag.train.argv.txt").exists() and not (runsm / "dagpin" / "bank_sha_s0.txt").exists())
    (runsm / "dagoacc" / "w_acc.txt").write_text("50.0\n")
    rc, out = L.cmd("gatedag dagoacc 0 0.1", env=env, dryrun=False)
    check("B8 a changed w_acc.txt makes a re-pin REFUSED (the pin is never silently replaced)", rc == 2 and any("pins another training argv" in x for x in out), (rc, out))
    rc, out = L.cmd("traindag dagoacc 0 0.1", env=env, dryrun=False)
    check("B9 traindag dagoacc with a changed w_acc since the gate is REFUSED", rc == 2 and any("differs from the one the gate tokenised" in x for x in out), (rc, out))
    (runsm / "dagoacc" / "w_acc.txt").write_text("")                      # `set /p X=<emptyfile` LEAVES X unchanged: only the clearing of an inherited WACCDAG stops the environment standing in for the rule
    rc, out = L.cmd("gatedag dagoacc 0 0.1", env={**env, "WACCDAG": "99.9"}, dryrun=False)
    regress("B9b an INHERITED WACCDAG cannot stand in for the rule: with an empty w_acc.txt the stage is REFUSED (no w_acc, rc 2) instead of using the environment's 99.9", rc == 2 and any("no w_acc" in x for x in out) and not any("99.9" in x for x in out), (rc, out))
    (runsm / "dagoacc" / "w_acc.txt").write_text("49.6969\n")
    rc, out = L.cmd("traindag dagoacc 0 0.1", env=env, dryrun=False)
    ran = jl(runsm / "M2b_s0_dagoacc" / "stub_argv.json") if (runsm / "M2b_s0_dagoacc" / "stub_argv.json").exists() else []
    pinned_tokens = L.train_argv(pin.read_text())
    check("B10 traindag dagoacc runs EXACTLY the pinned argv (byte-identical tokens) into M2b_s0_dagoacc", rc == 0 and ran == pinned_tokens and ran[ran.index("--out") + 1].endswith("M2b_s0_dagoacc"), (rc, out))
    rc, out = L.cmd("gatedag dagoacc 1 0.1", env={**env, "RUNSM": str(runsm)}, dryrun=False)
    (runsm / "dagoacc" / "w_acc.txt").unlink()
    rc2, out2 = L.cmd("gatedag dagoacc 1 0.1", env=env, dryrun=False)
    check("B11 with no w_acc.txt the dagoacc arm is REFUSED rc 2 (the placeholder is dry-run only; no env stand-in)", rc2 == 2 and any("run the waccdag stage first" in x for x in out2), (rc2, out2))
    rc3, out3 = L.cmd("gatedag dagoacc 1 0.1", env={**env, "WACCDAG": "49.6969", "WACCDAG_DRY": "49.6969"}, dryrun=False)
    check("B12 an inherited WACCDAG / WACCDAG_DRY cannot stand in for the rule in a real run", rc3 == 2, (rc3, out3))
    # flat arms need FLAT's directory and bank
    for s in (0, 1):
        (runsm / f"M2b_s{s}_flat").mkdir(exist_ok=True)
        (runsm / f"M2b_s{s}_flat" / "ckpt_last.pt").write_bytes(b"f")
    rc, out = L.cmd("bankflat 0", env=env, dryrun=False)
    check("B13 bankflat with FLAT unfinished (no summary.json) is REFUSED rc 2 and creates no bank directory", rc == 2 and any("has not finished training" in x for x in out) and not (runsm / "dagger_bank_flat" / "s0" / "bank.json").exists(), (rc, out))
    rc, out = L.cmd("gatedag flatdag 0 0.1", env=env, dryrun=False)
    check("B14 gatedag flatdag without a FLAT bank is REFUSED rc 2 (it does not fall back to the iteration-1 bank)", rc == 2 and any("run bankflat 0 first" in x for x in out), (rc, out))
    (runsm / "dagger_bank_flat" / "s0").mkdir(parents=True)
    (runsm / "dagger_bank_flat" / "s0" / "bank.json").write_text("{}")
    rc, out = L.cmd("gatedag flatdag 0 0.1", env=env, dryrun=False)
    fp = runsm / "dagpin" / "M2b_s0_flatdag.train.argv.txt"
    check("B15 gatedag flatdag pins a line that names --o-interface cut, the FLAT init and the FLAT bank", rc == 0 and fp.exists() and "--o-interface cut" in fp.read_text() and "M2b_s0_flat\\ckpt_last.pt" in fp.read_text() and "dagger_bank_flat\\s0" in fp.read_text(), (rc, out))
    rc, out = L.cmd("gatedag flatdag0 0", env=env, dryrun=False)
    f0 = runsm / "dagpin" / "M2b_s0_flatdag0.train.argv.txt"
    check("B16 gatedag flatdag0 pins a line with --o-interface cut and --dagger-p 0 and no bank flag (no bank needed)", rc == 0 and "--o-interface cut" in f0.read_text() and "--dagger-p 0" in f0.read_text() and "--dagger-bank" not in f0.read_text(), (rc, out))
    # bankflat, real, on seed 1 (FLAT_s1 finished, no bank yet)
    (runsm / "M2b_s1_flat" / "summary.json").write_text("{}")
    (runsm / "M2b_s1_flat" / "argv.json").write_text("{}")
    rc, out = L.cmd("bankflat 1", env=env, dryrun=False)
    bt = jl(runsm / "dagger_bank_flat" / "s1" / "tool_argv.json") if (runsm / "dagger_bank_flat" / "s1" / "tool_argv.json").exists() else []
    check("B17 bankflat 1 (FLAT_s1 finished) runs the bank tool with FLAT_s1's checkpoint AND argv.json (never M2b_s1's) into dagger_bank_flat\\s1, four shards, bankdag's expectations",
          rc == 0 and bt[bt.index("--ckpt") + 1].endswith("M2b_s1_flat\\ckpt_last.pt") and bt[bt.index("--m2b-argv") + 1].endswith("M2b_s1_flat\\argv.json") and bt[bt.index("--out-dir") + 1].endswith("dagger_bank_flat\\s1")
          and bt[bt.index("--n-shards") + 1] == "4" and bt[bt.index("--expect-clips") + 1] == "4512" and bt[bt.index("--expect-windows") + 1] == "306566" and not (runsm / "dagger_bank" / "s1" / "tool_argv.json").exists(), (rc, out, bt))
    rc, out = L.cmd("bankflat 1", env=env, dryrun=False)
    check("B18 a second bankflat 1 is skipped (a finished bank is never regenerated)", rc == 0 and any("[skip]" in x for x in out), (rc, out))


# ================================================================================================ C. decisions
def section_c():
    say("== C. the chain's decisions on files")
    d = W / "ckdata"
    d.mkdir(parents=True, exist_ok=True)
    off = jl(OFFLINE)
    box = {**off, "banked_crosscheck": {"pass": True, "same_windows": True, "fit_windows_sha256": off["source"]["fit_windows_sha256"]}, "source": {"ckpt_sha256": "d186fd0bcfa7ee9fbfe5c9de543fe85edda7c3e6b9c56c898958f027ae2b8cb1"}}
    txt = d / "w_acc.txt"
    txt.write_text("49.6969\n", encoding="ascii", newline="")

    def co3(mut=None, txt_text=None, **kw):
        b = copy.deepcopy(box)
        if mut:
            mut(b)
        p = d / "box.json"
        p.write_text(json.dumps(b))
        # w_acc.txt always names the box record's own number unless the case sets it: so a w_acc case trips the 1e-4 tolerance, not the txt consistency check
        (d / "t2.txt").write_text(txt_text if txt_text is not None else f"{b['w_acc']!r}" + chr(10), newline="")
        rc, rec = CK.wacc_check(str(p), str(OFFLINE), kw.get("sha", sha_file(OFFLINE)), str(d / "t2.txt"), "d186fd0bcfa7ee9fbfe5c9de543fe85edda7c3e6b9c56c898958f027ae2b8cb1")
        return rc
    check("C1 C-O3: the box record equal to the offline recomputation is accepted", co3() == 0)
    check("C2 C-O3: a rounding-level difference (rel 5e-5) is accepted", co3(lambda b: b.update(w_acc=49.6969 * (1 + 5e-5))) == 0)
    for nm, mut in (("w_acc +2e-4 relative", lambda b: b.update(w_acc=49.6969 * (1 + 2e-4))), ("w_acc = OACC's M2b value 42.9311", lambda b: b.update(w_acc=42.9311)), ("L_traj +1e-3", lambda b: b.update(L_traj=b["L_traj"] * 1.001)),
                    ("L_acc -1e-3", lambda b: b.update(L_acc=b["L_acc"] * 0.999)), ("n_windows 9581", lambda b: b.update(n_windows=9581)), ("the cross-check not passed", lambda b: b["banked_crosscheck"].update({"pass": False})),
                    ("the cross-check on other windows", lambda b: b["banked_crosscheck"].update({"same_windows": False})), ("no cross-check record", lambda b: b.pop("banked_crosscheck")), ("frac 0.4", lambda b: b.update(frac=0.4)),
                    ("the checkpoint is M2b_s0's", lambda b: b["source"].update({"ckpt_sha256": "601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8"})), ("cross-check on other fit windows file", lambda b: b["banked_crosscheck"].update({"fit_windows_sha256": "0" * 64}))):
        regress(f"C3 C-O3 REFUSES: {nm}", co3(nm and mut) == 2)
    regress("C4 C-O3 REFUSES: w_acc.txt is not the bare number", co3(txt_text="49.69\n") == 2)
    regress("C5 C-O3 REFUSES: w_acc.txt without the LF", co3(txt_text="49.6969") == 2)
    regress("C6 C-O3 REFUSES: the offline record is not the pinned file", co3(sha="0" * 64) == 2)
    # ---- the F1 lever grep
    base = "INIT_CKPT_MATCH = (\"d\", \"heads\",\n                   \"kin_mode\", \"max_age_t\", \"max_age_s\"{extra})\nx = 1\n"
    check("C7 init-match: a trainer whose INIT_CKPT_MATCH names o_interface (any formatting) passes", CK.init_match_has(base.format(extra=',\n                   "o_interface"'))[0] and CK.init_match_has("INIT_CKPT_MATCH = ('a', \"o_interface\")")[0])
    regress("C8 init-match REFUSES the tip's trainer (no o_interface in the tuple)", not CK.init_match_has(base.format(extra=""))[0])
    regress("C9 init-match REFUSES a trainer where o_interface sits only in a comment / another statement", not CK.init_match_has(base.format(extra="") + "# o_interface\nOTHER = ('o_interface',)\n")[0])
    real_tip = git_blob("stack/scripts/v8f_mini_train.py").decode("utf-8")
    regress("C10 init-match REFUSES the REAL trainer at the tip 1d4f4fbb (F1 is not landed there)", not CK.init_match_has(real_tip)[0])
    fixed = Path("E:/Projects/TanitAD-artifacts/oaccflat_check/stage/stack/scripts/v8f_mini_train.py")
    check("C11 init-match ACCEPTS the staged F1 fix (oaccflat_check/stage)", CK.init_match_has(fixed.read_text(encoding="utf-8"))[0])
    # ---- the disk floor
    g = 1 << 30
    check("C12 disk-check: free - size == floor exactly passes", CK.disk_check(str(W), 1000, g, g + 1000)[0] == 0)
    regress("C13 disk-check REFUSES one byte under the floor", CK.disk_check(str(W), 1000, g, g + 999)[0] == 2)
    regress("C14 disk-check REFUSES a pull larger than the free space", CK.disk_check(str(W), 10 * g, g, 5 * g)[0] == 2)
    # ---- the predecessor rule
    P = CK.pred_verdict
    done, stop_, guard = "x\nZZV8FMINI-DONE 10-27 12:00 tip abc: done\n", "x\nZZV8FMINI-STOP bank s0: BLOCKED\n", "ZZGUARD-STOP TEMP 10-27 USB 64 C\nZZV8FMINI-STOP t_M2b_s0_dag2 rc=1\n"
    check("C15 pred: DONE + BOX2_BUSY released -> GO", P(done, False)[0] == "GO")
    check("C16 pred: a STOP of the predecessor's own (BLOCKED) + BOX2_BUSY released -> GO (a predecessor STOP must not idle box 2)", P(stop_, False)[0] == "GO")
    check("C17 pred: a failed run's STOP + BOX2_BUSY released -> GO", P("ZZV8FMINI-STOP t_M2b_s0_dag2 rc=1\n", False)[0] == "GO")
    regress("C18 pred: a GUARD stop -> HALT", P(guard, False)[0] == "HALT")
    regress("C19 pred: a STOP with BOX2_BUSY still held -> not GO (BUSY: the caller re-checks, then halts)", P(stop_, True)[0] == "BUSY")
    regress("C20 pred: DONE with BOX2_BUSY still held -> HALT", P(done, True)[0] == "HALT")
    check("C21 pred: no end marker -> WAIT", P("log line\n", False)[0] == "WAIT" and P("", False)[0] == "WAIT")
    check("C22 pred: the 'waiting for the predecessor ... ^ZZV8FMINI-STOP' echo (it starts with a date) is not a marker", P("10-27 10:13 waiting for the predecessor chain (x: end marker ^ZZV8FMINI-DONE, stop marker ^ZZV8FMINI-STOP)\n", False)[0] == "WAIT")
    check("C23 pred: CRLF logs are read", P(done.replace("\n", "\r\n"), False)[0] == "GO")
    # ---- COVERAGE (PREREG outcome table): the judge on doctored copies of the REAL banks' small files (bank.json, PROVENANCE.json, the 4 shard provenance files)
    kd = W / "kbcov"
    if kd.exists():
        shutil.rmtree(kd)
    shas = []
    for sd in (0, 1):
        (kd / f"s{sd}").mkdir(parents=True)
        for fn in ["bank.json", "PROVENANCE.json"] + [f"shard_00{k}of004.provenance.json" for k in range(4)]:
            shutil.copy2(DAGART / f"bank_s{sd}" / fn, kd / f"s{sd}" / fn)
        shas.append(jl(kd / f"s{sd}" / "bank.json")["common"]["ckpt_sha256"])

    def judge_cov(n_e0):
        """seed 0's coverage set to n_e0 eligible windows (flag and PROVENANCE kept consistent), seed 1 untouched."""
        for fn in ("bank.json", "PROVENANCE.json"):
            j = jl(DAGART / "bank_s0" / fn)
            j["coverage"]["n_eligible"] = n_e0
            if fn == "bank.json":
                j["coverage"]["blocked"] = (n_e0 / j["coverage"]["n_windows"]) < 0.5
            (kd / "s0" / fn).write_text(json.dumps(j))
        return sh([L.PY, posix(SHIP / "v8f_oaf_chain_checks.py"), "bank-judge", "--banked-sha0", shas[0], "--banked-sha1", shas[1], "--seed0-dir", posix(kd / "s0"), "--seed1-dir", posix(kd / "s1")])
    nw = jl(DAGART / "bank_s0" / "bank.json")["coverage"]["n_windows"]
    rc, out = judge_cov(jl(DAGART / "bank_s0" / "bank.json")["coverage"]["n_eligible"])
    check("C-cov1 bank-judge on the real banks' small files: PASS rc 0 (CONTROL for the COVERAGE cases below)", rc == 0 and '"outcome": "PASS"' in out, out[-300:])
    rc, out = judge_cov(nw // 2)
    check(f"C-cov2 COVERAGE exactly 0.5 ({nw // 2} / {nw}) is NOT blocked: PASS rc 0 (the threshold is inclusive)", nw % 2 == 0 and rc == 0 and '"outcome": "PASS"' in out, out[-300:])
    rc, out = judge_cov(nw // 2 - 1)
    regress(f"C-cov3 COVERAGE just under 0.5 ({nw // 2 - 1} / {nw}) at seed 0: BLOCKED, rc 3, blocked_seeds [0] (the chain stops with nothing trained)", rc == 3 and '"outcome": "BLOCKED"' in out and '"blocked_seeds": [0]' in out, out[-300:])
    rc, out = judge_cov(1000)
    regress("C-cov4 COVERAGE 1000 windows at seed 0: BLOCKED, rc 3", rc == 3 and '"outcome": "BLOCKED"' in out, out[-300:])
    shutil.rmtree(kd, ignore_errors=True)



# ================================================================================================ D. the chain script
# box 2 emulated by a directory ($FAKEBOX): ssh answers only the size query (rsize), scp copies from it (SCP_SHORT: a transfer that lands short)
SSH_SHIM = r'''#!/usr/bin/env bash
cmd="$*"; q="'"
case "$cmd" in
  *"Get-Item -LiteralPath "*)
    f=${cmd#*LiteralPath $q}; f=${f%%$q*}; f=${f#D:/tanitad/runs/v8f_mini_full/}
    stat -c %s "$FAKEBOX/$f" 2>/dev/null ;;
esac
exit 0
'''
SCP_SHIM = r'''#!/usr/bin/env bash
src="${@: -2:1}"; dst="${@: -1}"; f=${src#tanitad-box2:D:/tanitad/runs/v8f_mini_full/}
if [ -n "$SCP_SHORT" ]; then head -c 10 "$FAKEBOX/$f" > "$dst"; else cp "$FAKEBOX/$f" "$dst"; fi
'''


def bash(args, env=None, timeout=300, chain="chain_v8f_dagoacc.sh", pre=""):
    """pre: shell text before the chain (git bash puts /usr/bin first in PATH on startup, so a shim dir on the Windows PATH is NOT found: the shim dir is exported inside the shell)."""
    return sh([BASH, "-c", f"{pre}cd '{posix(SHIP)}' && bash {chain} {args}"], env=env, timeout=timeout)


def section_d():
    say("== D. the DAG-OACC chain script")
    chain = (SHIP / "chain_v8f_dagoacc.sh").read_text(encoding="utf-8")
    rc, out = sh([BASH, "-n", posix(SHIP / "chain_v8f_dagoacc.sh")])
    check("D1 bash -n: chain, lib, guard, waker parse", rc == 0 and all(sh([BASH, "-n", posix(SHIP / f)])[0] == 0 for f in ("oaf_lib.sh", "v8f_guard_dagoacc.sh", "wake_v8fdagoacc.sh")), out)
    rc, out = bash("--dry-run", env={"V8F_QUEUE": posix(L.QUEUE)})
    check("D2 --dry-run: rc 0, both runs planned, ONE-LEVER on both planned lines, gate == train, C-O3 reference printed, no STOP marker line (a line that STARTS with it: the echo of the wait rule names it mid-line)",
          rc == 0 and out.count("ONE-LEVER") == 2 and "NOT-ONE-LEVER" not in out and out.count(": SAME") == 2 and "49.6969" in out and not re.search(r"(?m)^ZZV8FMINI-STOP", out), out[-600:])
    rc, out = bash("--dry-run", env={"V8F_QUEUE": posix(L.QUEUE), "DAGOACC_DRY_WACC": "50.0"})
    check("D3 --dry-run with another W still plans one lever (--w-acc 50.0 in both train lines)", rc == 0 and out.count("--w-acc 50.0") >= 4 and out.count("ONE-LEVER") == 2, out[-300:])
    # check-tree on a tree = the tip's python files + the staged queue + HBLEND's fit_clips placeholder
    ct = FW / "checktree"
    if ct.exists():
        subprocess.run(["cmd", "/c", "rmdir", "/s", "/q", str(ct)])
    (ct / PK / "code").mkdir(parents=True)
    (ct / PK / "raw" / "2026-10-25-hblend").mkdir(parents=True)
    shutil.copy(L.QUEUE, ct / PK / "code" / "v8f_mini_box2_queue.cmd")
    (ct / PK / "raw" / "2026-10-25-hblend" / "fit_clips.json").write_text("{}")
    for top in ("stack", "taniteval"):
        subprocess.run(["cmd", "/c", "mklink", "/J", str(ct / top), str(PYTREE / top)], capture_output=True)
    rc, out = bash(f"--check-tree '{posix(ct)}'")
    check("D4 --check-tree: every lever pattern is in the staged tree (trainer, model, DAgger, oacc tool, the queue insertions)", rc == 0 and "every one of" in out, out[-500:])
    q0 = ct / PK / "code" / "v8f_mini_box2_queue.cmd"
    q0.write_bytes(L.TIP_QUEUE.read_bytes())
    rc, out = bash(f"--check-tree '{posix(ct)}'")
    regress("D5 --check-tree REFUSES a tree whose queue is the tip's (no DAG-OACC stages): the chain would stop before touching box 2", rc == 1 and "MISSING" in out, out[-300:])
    # co3-only
    d = W / "ckdata"
    off = jl(OFFLINE)
    good = {**off, "banked_crosscheck": {"pass": True, "same_windows": True, "fit_windows_sha256": off["source"]["fit_windows_sha256"]}, "source": {"ckpt_sha256": "d186fd0bcfa7ee9fbfe5c9de543fe85edda7c3e6b9c56c898958f027ae2b8cb1"}}
    (d / "g.json").write_text(json.dumps(good))
    (d / "g.txt").write_text("49.6969\n", newline="")
    rc, out = bash(f"--co3-only '{posix(d / 'g.json')}' '{posix(d / 'g.txt')}'")
    check("D6 --co3-only on an equal record: ZZDAGOACC-CO3-PASS", rc == 0 and "ZZDAGOACC-CO3-PASS" in out, out[-300:])
    bad = dict(good, w_acc=42.9311)
    (d / "b.json").write_text(json.dumps(bad))
    rc, out = bash(f"--co3-only '{posix(d / 'b.json')}' '{posix(d / 'g.txt')}'")
    regress("D7 --co3-only on OACC's M2b value 42.9311: ZZDAGOACC-CO3-STOP, rc 1 (the chain stops before any gate)", rc == 1 and "ZZDAGOACC-CO3-STOP" in out and "PASS" not in out, out[-300:])
    check("D8 the real run stops at C-O3 BEFORE any gate: the co3 call sits before the first gate_one / queue gate in the source", chain.index('co3 "$ART/wacc/w_acc.json"') < chain.index('for run in "${RUNS[@]}"; do set -- $run; gate_one') and "ZZDAGOACC-CO3-STOP" in chain.split('co3 "$ART/wacc/w_acc.json"')[1].split("WACC=$(tr")[0])
    m_co3 = re.search(r'(?m)^co3 "\$ART/wacc/w_acc\.json" "\$ART/wacc/w_acc\.txt" \|\| \{ echo "ZZDAGOACC-CO3-STOP[^\n]*; stop "C-O3 mismatch"; \}\r?$', chain)
    check("D8b the real run's C-O3 line is in its exact STOP form (`co3 ... || { echo ZZDAGOACC-CO3-STOP ...; stop ...; }`): a mismatch cannot fall through (TEXT-PINNED: the decision itself is executed by D6 / D7 through --co3-only; the real-run wiring is not executed)", m_co3 is not None)
    # the pins in the chain text are the BANKED artifacts' hashes (re-derived here from the files themselves, not from the generator's output)
    import make_chains as MC
    pins = MC.dag_pins()
    arr = lambda name: re.search(rf"(?m)^{name}=\(([0-9a-f ]+)\)", chain).group(1).split()
    one = lambda name: re.search(rf"(?m)^{name}=([0-9a-f]+)", chain).group(1)
    bad_pin = [k for k in ("CKPT_SHA", "ARGV_SHA", "SUMM_SHA", "M2B_SHA", "BANK1_SHA") if arr(k) != pins[k]]
    if one("OFFLINE_SHA") != sha_file(OFFLINE):
        bad_pin.append("OFFLINE_SHA")
    if one("FITPLANS_SHA") != sha_file("E:/Projects/TanitAD-artifacts/daghacc/plans/fitplans_DAG_s0.npz"):
        bad_pin.append("FITPLANS_SHA")
    check("D21 every pinned hash in the chain (DAG ckpt / argv.json / summary.json, M2b init, iteration-1 bank, the offline record, DAG_s0's fit plans) equals the hash of the banked file", not bad_pin, bad_pin)
    check("D22 the chain's run list is s0 then s1 of the arm dagoacc, P = 0.1", 'RUNS=("dagoacc 0" "dagoacc 1")' in chain and re.search(r"(?m)^PVAL=0\.1\b", chain) is not None)
    # early stops without a box-2 call (ssh / scp shims log every call)
    shim = W / "shim"
    shim.mkdir(parents=True, exist_ok=True)
    (shim / "ssh").write_text("#!/usr/bin/env bash\necho ssh \"$@\" >> \"$SHIM_LOG\"\nexit 1\n", newline="\n")
    (shim / "scp").write_text("#!/usr/bin/env bash\necho scp \"$@\" >> \"$SHIM_LOG\"\nexit 1\n", newline="\n")
    slog = W / "shim.log"
    slog.write_text("")
    env = {"PATH": posix(shim) + os.pathsep + os.environ["PATH"], "SHIM_LOG": posix(slog)}
    sp = posix(shim)
    sp = "/" + sp[0].lower() + sp[2:] if sp[1:2] == ":" else sp                  # a POSIX PATH entry: `C:/x` would split at the colon
    pre = "export PATH='" + sp + "':\"$PATH\"; "
    sh([BASH, "-c", pre + "ssh control-call; scp control-call"], env=env)
    ctl = slog.read_text()
    check("D9a control: the shims ARE what `ssh` / `scp` resolve to under the chain's shell (the log must be non-empty)", "ssh control-call" in ctl and "scp control-call" in ctl, ctl)
    slog.write_text("")
    rc, out = bash("", env=env, pre=pre)
    check("D9 the real entry without AFTER_OUT: ZZV8FMINI-STOP, rc 1, no ssh / scp call (log empty, after the control above read non-empty)", rc == 1 and "ZZV8FMINI-STOP AFTER_OUT is not set" in out and slog.read_text() == "", out[-300:])
    # ---- the predecessor branches on a fake ship dir: --wait-only
    fb = FW / "ship"

    def wait(log_text, busy, extra=None):
        if fb.exists():
            shutil.rmtree(fb)
        (fb / "logs").mkdir(parents=True)
        (fb / "logs" / "pred.out").write_text(log_text, newline="")
        if busy:
            (fb / "BOX2_BUSY.txt").write_text("busy")
        e = {**env, "OAF_SHIP_DIR": posix(fb), "AFTER_OUT": posix(fb / "logs" / "pred.out"), "OAF_POLL_S": "1", "OAF_GRACE_S": "1", **(extra or {})}
        return bash("--wait-only", env=e, timeout=60)
    rc, out = wait("a\nZZV8FMINI-DONE 10-27 done\n", False)
    check("D10 wait: predecessor DONE, box released -> ZZOAF-WAIT-GO", rc == 0 and "ZZOAF-WAIT-GO" in out, out[-300:])
    rc, out = wait("a\nZZV8FMINI-STOP bank s0: BLOCKED (COVERAGE)\n", False)
    check("D11 wait: predecessor STOP of its own (BLOCKED), box released -> GO (box 2 is not idled)", rc == 0 and "ZZOAF-WAIT-GO" in out, out[-300:])
    rc, out = wait("a\nZZV8FMINI-STOP t_M2b_s0_dag2 rc=1\n", False)
    check("D12 wait: predecessor STOP after a failed run, box released -> GO", rc == 0 and "ZZOAF-WAIT-GO" in out, out[-300:])
    rc, out = wait("a\nZZGUARD-STOP TEMP 10-27 USB 64 C\nZZV8FMINI-STOP guard stop\n", False)
    regress("D13 wait: predecessor stopped by the GUARD -> ZZV8FMINI-STOP, no GO", rc == 1 and "ZZV8FMINI-STOP predecessor verdict: HALT" in out and "ZZOAF-WAIT-GO" not in out, out[-300:])
    rc, out = wait("a\nZZV8FMINI-STOP bank s0: BLOCKED\n", True)
    regress("D14 wait: predecessor STOP with BOX2_BUSY still held (after 3 re-checks) -> ZZV8FMINI-STOP, no GO", rc == 1 and "still held" in out and "ZZOAF-WAIT-GO" not in out, out[-300:])
    rc, out = wait("a\nZZV8FMINI-DONE done\n", True)
    regress("D15 wait: predecessor DONE with BOX2_BUSY held -> ZZV8FMINI-STOP", rc == 1 and "ZZOAF-WAIT-GO" not in out, out[-300:])
    if fb.exists():
        shutil.rmtree(fb)
    (fb / "logs").mkdir(parents=True)
    (fb / "logs" / "pred.out").write_text("10-27 10:13 waiting for the predecessor chain (x: end marker ^ZZV8FMINI-DONE, stop marker ^ZZV8FMINI-STOP)\nsome log line\n", newline="")
    p = subprocess.run([BASH, "-c", f"cd '{posix(SHIP)}' && OAF_SHIP_DIR='{posix(fb)}' AFTER_OUT='{posix(fb / 'logs' / 'pred.out')}' OAF_POLL_S=1 timeout 6 bash chain_v8f_dagoacc.sh --wait-only"], capture_output=True, text=True, env={**os.environ, **env}, stdin=subprocess.DEVNULL)
    check("D16 wait: no end marker (only the echo of the rule) -> keeps waiting (still polling after 6 s, killed by the test's timeout rc 124)", p.returncode == 124 and "ZZOAF-WAIT-GO" not in (p.stdout or ""), (p.returncode, p.stdout))
    # D16b: the wait must be on the REAL path too (D16 runs --wait-only): the real entry, a predecessor that has not ended -> it keeps waiting, takes no BOX2_BUSY, makes no ssh / scp call
    slog.write_text("")
    p = subprocess.run([BASH, "-c", pre + f"cd '{posix(SHIP)}' && OAF_SHIP_DIR='{posix(fb)}' AFTER_OUT='{posix(fb / 'logs' / 'pred.out')}' OAF_POLL_S=1 timeout 8 bash chain_v8f_dagoacc.sh"], capture_output=True, text=True, env={**os.environ, **env}, stdin=subprocess.DEVNULL)
    regress("D16b the REAL entry with a predecessor that has not ended: still waiting after 8 s (killed by the test's timeout, rc 124), BOX2_BUSY not taken, no ssh / scp call", p.returncode == 124 and not (fb / "BOX2_BUSY.txt").exists() and slog.read_text() == "" and "ZZOAF-WAIT-GO" not in (p.stdout or ""), (p.returncode, (p.stdout or "")[-300:], slog.read_text()[-200:]))
    # ---- the disk floor through the REAL pull(): shims emulate box 2 from FAKE dir
    fx = FW / "pullbox"
    if fx.exists():
        shutil.rmtree(fx)
    (fx / "dagoacc").mkdir(parents=True)
    (fx / "dagoacc" / "w_acc.json").write_bytes(b"x" * 5000)
    shim2 = FW / "shim2"
    shim2.mkdir(parents=True, exist_ok=True)
    (shim2 / "ssh").write_text(SSH_SHIM, newline="\n")
    (shim2 / "scp").write_text(SCP_SHIM, newline="\n")
    m_fl = re.search(r"(?m)^FLOOR_BYTES=(\d+)", chain)                                   # the driver uses the CHAIN's own floor, not a restated one
    chain_floor = m_fl.group(1) if m_fl else "0"
    check("D17a the chain's own disk floor is exactly 1 GiB (1073741824 bytes)", chain_floor == "1073741824", chain_floor)
    drv = ('set -u\nsource "' + posix(SHIP / "oaf_lib.sh") + '"\nR=D:/tanitad/runs/v8f_mini_full; B=/tmp; PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe; CHECKS=' + posix(SHIP / "v8f_oaf_chain_checks.py")
           + '\nSSH="' + posix(shim2 / "ssh") + '"; SCP="' + posix(shim2 / "scp") + '"; BUSY_TAKEN=0; FLOOR_BYTES=' + chain_floor + '; CUR=/tmp/oaf_cur; E=/tmp\npull "dagoacc/w_acc.json" "' + posix(FW / "pulled") + '"\necho PULLED-OK\n')
    (FW / "drv.sh").write_text(drv, newline="\n")

    def run_pull(free, short=False):
        if (FW / "pulled").exists():
            shutil.rmtree(FW / "pulled")
        e = {"PATH": posix(shim2) + os.pathsep + os.environ["PATH"], "FAKEBOX": posix(fx), "OAF_TEST_FREE_BYTES": str(free)}
        if short:
            e["SCP_SHORT"] = "1"
        return sh([BASH, posix(FW / "drv.sh")], env=e)
    g = 1 << 30
    rc, out = run_pull(g + 5000)
    check("D17 pull(): exactly the floor left -> the file lands (size verified, .part renamed)", rc == 0 and "PULLED-OK" in out and (FW / "pulled" / "w_acc.json").stat().st_size == 5000 and not list((FW / "pulled").glob("*.part")), out[-300:])
    rc, out = run_pull(g + 4999)
    regress("D18 pull(): one byte under the floor -> ZZDISK-FLOOR-STOP + ZZV8FMINI-STOP, NOTHING written (no file, no .part)", rc == 1 and "ZZDISK-FLOOR-STOP" in out and "ZZV8FMINI-STOP" in out and "PULLED-OK" not in out and not list((FW / "pulled").glob("*")), out[-300:])
    rc, out = run_pull(g + 9999, short=True)
    regress("D19 pull(): a transfer that lands short is refused and the .part removed (nothing half-written)", rc == 1 and "PULLED-OK" not in out and not list((FW / "pulled").glob("*")), out[-300:])

    record_tests("DR", chain, [("dagoacc", 0.1, "cd" * 32, {}, ["--dagger-p", "0.1"], {})])

# ================================================================================================ E. gate assertions
def gate_files(arm, seed, sha, wacc=None):
    g, t, _ = lines3(arm, seed, sha, wacc)
    return g, t


def sc_input(name, rel):
    """a REAL banked gate selfcheck from the tip (1d4f4fbb), materialised into the scratch dir on first use (the proof never depends on a hand-copied file)."""
    f = W / name
    if not f.exists():
        f.write_bytes(git_blob(f"{PK}/raw/{rel}"))
    return jl(f)


def doctored(arm, seed, W_=WACC):
    """The REAL DAG corpus gate selfcheck (written by the real trainer on box 2), doctored into the arm's gate: nothing in it is invented that the real gate does not write; the arm's own keys are copied from the real OACC / FLAT gates."""
    sc = jl(DAGART / f"M2b_s{seed}_dag" / "selfcheck.json")
    sc["levers"].setdefault("w_acc", 0.0)
    if arm == "dagoacc":
        oacc = sc_input("oacc_sc.json", "2026-10-25-oacc/run/gate_M2b-oacc/selfcheck.json")
        sc["levers"]["w_acc"] = float(W_)
        sc["live"]["terms"] = {**(sc["live"].get("terms") or {}), "acc": 0.0107}
        sc["live"].setdefault("missing_terms", [])
        sc["live"]["head_o_grad_step0"] = 738.9
        ap = dict(oacc["acc_probe"])
        ap.update(w_acc_declared=float(W_), w_probe=float(W_), contribution=float(W_) * ap["acc"], w_x_acc=float(W_) * ap["acc"])
        sc["acc_probe"] = ap
    else:
        flat = sc_input("flat_sc.json", "2026-10-26-flat/runs/M2b_s0_flat/selfcheck.json")
        sc["levers"]["o_interface"] = "cut"
        sc["o_interface_probe"] = flat["o_interface_probe"]
        sc["live"]["dead_params"] = ["e_code.weight"]
        sc["live"]["declared_dead"] = ["e_code.weight"]
        if arm == "flatdag0":
            for k in ("dagger_bank_sha256", "dagger_bank_sha256_declared"):
                sc["levers"][k] = None
            sc["levers"]["dagger_p"] = 0.0
            sc["levers"].pop("dagger", None)
            sc["info"].pop("dagger", None)
            sc["dagger_probe"] = {"dagger": "off", "pass": True}
        else:
            bank = "cd" * 32
            sc["levers"]["dagger_bank_sha256"] = sc["levers"]["dagger_bank_sha256_declared"] = bank
            sc["info"]["dagger"]["bank_sha256"] = bank
            sc["levers"]["dagger"]["dagger_bank"] = sc["levers"]["dagger"]["dagger_bank"].replace("\\dagger_bank\\", "\\dagger_bank_flat\\")
            sc["info"]["dagger"]["kb2"]["ckpt_sha256"] = sc["info"]["dagger"]["kb2"]["init_ckpt_sha256"] = FLATSHA[seed]
        ic = sc["levers"]["init_ckpt"]
        ic["sha256"] = FLATSHA[seed]
        ic["path"] = ic["path"].replace(f"M2b_s{seed}\\", f"M2b_s{seed}_flat\\")
        sc["info"]["init_ckpt"] = ic
    return sc


def run_assert(arm, seed, mutate_sc=None, mutate_lines=None, mode="check", extra=None, mutate_pin=None, ref_mut=None, dag_pin_mut=None):
    d = W / "assert_case"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    sha = BANK1[seed] if arm == "dagoacc" else ("cd" * 32 if arm == "flatdag" else None)
    g, t = gate_files(arm, seed, sha, WACC if arm == "dagoacc" else None)
    if mutate_lines:
        g, t = mutate_lines(g, t)
    pin = mutate_pin(t) if mutate_pin else t
    sc = doctored(arm, seed)
    sc["argv_sha"] = hashlib.sha256(" ".join(L.train_argv(g)).encode()).hexdigest()[:16]
    if mutate_sc:
        mutate_sc(sc)
    (d / "g.txt").write_text(g)
    (d / "t.txt").write_text(t)
    (d / "p.txt").write_text(pin)
    (d / "sc.json").write_text(json.dumps(sc))
    args = ["--selfcheck", d / "sc.json", "--gate-line", d / "g.txt", "--train-line", d / "t.txt", "--pin-line", d / "p.txt", "--arm", arm, "--seed", str(seed), "--b-summary"]
    if arm == "dagoacc":
        dagref = jl(DAGART / f"M2b_s{seed}_dag" / "argv.json")
        if ref_mut:
            ref_mut(dagref)
        (d / "ref.json").write_text(json.dumps(dagref))
        dp = L.dry(f"traindag dag {seed} 0.1", sha=BANK1[seed])
        dpl = L.trainer_line(dp)
        if dag_pin_mut:
            dpl = dag_pin_mut(dpl)
        (d / "dagpin.txt").write_text(dpl)
        args += [DAGART / f"M2b_s{seed}_dag" / "summary.json", "--p", "0.1", "--bank-sha", BANK1[seed], "--bank-json", DAGART / f"bank_s{seed}" / "bank.json", "--banked-init-sha", jl(DAGART / f"M2b_s{seed}_dag" / "summary.json")["levers"]["init_ckpt"]["sha256"],
                 "--w-acc", WACC, "--dag-pin-line", d / "dagpin.txt", "--ref-argv-json", d / "ref.json"]
    else:
        fb = git_blob(f"{PK}/raw/2026-10-26-flat/runs/M2b_s{seed}_flat/summary.json")
        (d / "fsum.json").write_bytes(fb)
        fa = json.loads(git_blob(f"{PK}/raw/2026-10-26-flat/runs/M2b_s{seed}_flat/argv.json"))
        if ref_mut:
            ref_mut(fa)
        (d / "ref.json").write_text(json.dumps(fa))
        args += [d / "fsum.json", "--banked-init-sha", FLATSHA[seed], "--ref-argv-json", d / "ref.json"]
        if arm == "flatdag":
            args += ["--p", "0.1", "--bank-sha", "cd" * 32, "--bank-json", DAGART / f"bank_s{seed}" / "bank.json"]
    if mode == "token":
        args += ["--commit", "a" * 40, "--queue-sha", "b" * 64, "--run", f"x_{arm}", "--out", d / "tok.json"]
    args += list(extra or [])
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = GA.main([mode] + [str(x) for x in args])
    return rc, buf.getvalue(), d


def section_e():
    say("== E. gate_oaf_assert.py on the REAL DAG corpus gate (doctored into the arms) + doctored negatives")
    for arm in ("dagoacc", "flatdag", "flatdag0"):
        for s in (0, 1):
            rc, out, d = run_assert(arm, s)
            check(f"E1 {arm} s{s}: the check passes on the doctored real gate", rc == 0, out[:600])
    rc, out, d = run_assert("dagoacc", 0, mode="token")
    tok = jl(d / "tok.json") if (d / "tok.json").exists() else {}
    check("E2 the token is written for dagoacc (prereg PREREG_V8F_DAG_OACC, w_acc, bank = DAG's, o_interface full)", rc == 0 and tok.get("prereg") == "PREREG_V8F_DAG_OACC" and tok.get("w_acc") == float(WACC) and tok.get("bank_sha256") == BANK1[0] and tok.get("o_interface") == "full", out[:300])
    rc, out, d = run_assert("flatdag", 1, mode="token")
    tok = jl(d / "tok.json") if (d / "tok.json").exists() else {}
    check("E3 the token is written for flatdag (prereg PREREG_V8F_FLATDAG, o_interface cut)", rc == 0 and tok.get("prereg") == "PREREG_V8F_FLATDAG" and tok.get("o_interface") == "cut", out[:300])
    neg = []

    def red(name, arm, **kw):
        rc, out, d = run_assert(arm, 0, **kw)
        regress(f"E4 {arm}: {name}  <- " + out.replace("OAF gate: ", "").strip()[:150], rc != 0, out[:300])
    # ---- dagoacc
    red("acc term absent from the loss dict", "dagoacc", mutate_sc=lambda sc: sc["live"]["terms"].pop("acc"))
    red("acc term exactly 0.0", "dagoacc", mutate_sc=lambda sc: sc["live"]["terms"].update(acc=0.0))
    red("acc term NaN", "dagoacc", mutate_sc=lambda sc: sc["live"]["terms"].update(acc=float("nan")))
    red("head_o gradient from the term is 0", "dagoacc", mutate_sc=lambda sc: sc["acc_probe"].update(head_o_grad_diff=0.0))
    red("acc_probe control not exactly 0", "dagoacc", mutate_sc=lambda sc: sc["acc_probe"].update(control_repeat_grad_diff=1e-9))
    red("acc_probe failed", "dagoacc", mutate_sc=lambda sc: sc["acc_probe"].update({"pass": False}))
    red("built w_acc != the argv's", "dagoacc", mutate_sc=lambda sc: sc["levers"].update(w_acc=42.9311))
    red("the total does not move by w x acc", "dagoacc", mutate_sc=lambda sc: sc["acc_probe"].update(contribution=0.0))
    red("the swap did not reach the loss (dagger_probe grad diff 0)", "dagoacc", mutate_sc=lambda sc: sc["dagger_probe"].update(grad_diff_all=0.0))
    red("dagger_probe p0 control not 0", "dagoacc", mutate_sc=lambda sc: sc["dagger_probe"].update(control_p0_grad_diff=3.0))
    red("the init is FLAT/other checkpoint (sha)", "dagoacc", mutate_sc=lambda sc: sc["levers"]["init_ckpt"].update(sha256="0" * 64))
    red("the bank is the iteration-2 directory", "dagoacc", mutate_sc=lambda sc: sc["levers"]["dagger"].update(dagger_bank=str(sc["levers"]["dagger"]["dagger_bank"]).replace("dagger_bank\\", "dagger_bank_dag2\\")))
    red("the bank hash differs from DAG's", "dagoacc", mutate_sc=lambda sc: sc["levers"].update(dagger_bank_sha256="0" * 64))
    red("ONE-LEVER broken: --lr changed in the train/gate/pin lines", "dagoacc", mutate_lines=lambda g, t: (g.replace("--lr 1e-4", "--lr 2e-4"), t.replace("--lr 1e-4", "--lr 2e-4")))
    red("ONE-LEVER broken: --w-acc not last (a flag after it)", "dagoacc", mutate_lines=lambda g, t: (g.replace(" --w-acc 49.6969", " --w-acc 49.6969 --hist 4", 1), t.replace(" --w-acc 49.6969", " --w-acc 49.6969 --hist 4", 1)))
    red("the pin differs from the planned training line (box 2 pinned another argv)", "dagoacc", mutate_pin=lambda t: t.replace("--steps 3000", "--steps 3001"))
    red("box 2's DAG pin is not DAG's banked argv.json (the comparator is not the banked run)", "dagoacc", dag_pin_mut=lambda s: s.replace("--steps 3000", "--steps 2999"))
    red("the pin differs from the planned training line ONLY in --out (box 2 pinned a run that writes elsewhere; no other check reads --out)", "dagoacc", mutate_pin=lambda t: re.sub(r'--out "([^"]*)"', r'--out "\1_x"', t))
    red("ONE-LEVER broken by a flag outside every lever list (--threads 3, identically on the train / gate / pin lines): only the comparison with DAG's pin sees it", "dagoacc",
        mutate_lines=lambda g, t: (g.replace(" --w-acc 49.6969", " --threads 3 --w-acc 49.6969", 1), t.replace(" --w-acc 49.6969", " --threads 3 --w-acc 49.6969", 1)))
    red("the reference argv.json (DAG's) carries another bank hash", "dagoacc", ref_mut=lambda ref: ref["argv"].__setitem__(ref["argv"].index("--dagger-bank-sha256") + 1, "0" * 64))
    red("a second lever on the line (--code-mix)", "dagoacc", mutate_lines=lambda g, t: (g.replace("--selfcheck-only", "--code-mix kind --selfcheck-only"), t + " --code-mix kind"))
    # ---- flatdag
    red("the cut is not built (levers.o_interface full)", "flatdag", mutate_sc=lambda sc: sc["levers"].update(o_interface="full"))
    red("o_interface_probe: O moved with the donor code", "flatdag", mutate_sc=lambda sc: sc["o_interface_probe"]["perturbations"]["donor_code"].update(max_abs_dtraj60=0.01))
    red("o_interface_probe: O read a non-constant code", "flatdag", mutate_sc=lambda sc: sc["o_interface_probe"]["perturbations"]["gt_code"].update(o_read_code_abs_max=0.5))
    red("o_interface_probe failed", "flatdag", mutate_sc=lambda sc: sc["o_interface_probe"].update({"pass": False}))
    red("e_code is not the declared dead weight (dead set empty)", "flatdag", mutate_sc=lambda sc: sc["live"].update(dead_params=[], declared_dead=[]))
    red("the checkpoint perturbation did not reach S / T (blind probe)", "flatdag", mutate_sc=lambda sc: sc["o_interface_probe"]["perturbations"]["shifted_checkpoints"].update(t_reads_s_max_abs_dgoal_logits=0.0))
    red("the swap did not reach (dagger_probe failed) while the cut passes", "flatdag", mutate_sc=lambda sc: sc["dagger_probe"].update({"pass": False}))
    red("the argv forgets --o-interface cut (the gate would still build FULL: F1)", "flatdag", mutate_lines=lambda g, t: (g.replace(" --o-interface cut", ""), t.replace(" --o-interface cut", "")))
    red("the init is M2b_s0, not FLAT_s0", "flatdag", mutate_sc=lambda sc: sc["levers"]["init_ckpt"].update(sha256="601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8"))
    red("the bank is the iteration-1 (M2b) bank directory", "flatdag", mutate_sc=lambda sc: sc["levers"]["dagger"].update(dagger_bank=str(sc["levers"]["dagger"]["dagger_bank"]).replace("dagger_bank_flat\\", "dagger_bank\\")))
    red("K-B2: the bank's generator is M2b, not FLAT", "flatdag", mutate_sc=lambda sc: sc["info"]["dagger"]["kb2"].update(ckpt_sha256="601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8"))
    red("a FLAT flag moved (--hist) vs FLAT's banked argv", "flatdag", ref_mut=lambda fa: fa["argv"].extend(["--hist", "4"]))
    red("FLAT's banked argv does not name the cut", "flatdag", ref_mut=lambda fa: [fa["argv"].pop(fa["argv"].index("--o-interface") + 1), fa["argv"].pop(fa["argv"].index("--o-interface"))])
    red("--w-acc on a FLAT arm (a second lever)", "flatdag", mutate_lines=lambda g, t: (g.replace("--selfcheck-only", "--w-acc 1.0 --selfcheck-only"), t + " --w-acc 1.0"))
    # ---- flatdag0
    red("flatdag0 carries a bank flag", "flatdag0", mutate_lines=lambda g, t: (g.replace("--selfcheck-only", "--dagger-seed 0 --selfcheck-only"), t + " --dagger-seed 0"))
    red("flatdag0's dagger probe is live (a swap at p = 0)", "flatdag0", mutate_sc=lambda sc: sc["dagger_probe"].update(dagger="on"))
    red("flatdag0: the cut is not built", "flatdag0", mutate_sc=lambda sc: sc["levers"].update(o_interface="full"))
    rc, out, _ = run_assert("flatdag0", 0, extra=["--p", "0.1"])
    regress("E5 flatdag0 with --p is REFUSED rc 2", rc == 2, out[:200])


# ================================================================================================ F. the FLAT-DAG chain
def _tree_for_flat(root: Path, trainer: Path, queue: Path):
    """a tree holding exactly the files the FLAT-DAG lever greps read (real copies, so a test can swap the trainer)."""
    if root.exists():
        shutil.rmtree(root)
    for rel in ("stack/scripts/v8f_dagger_bank.py", "stack/scripts/v8f_dagger_queue_tool.py", "stack/tanitad/data/v8f_dagger.py"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PYTREE / rel, root / rel)
    (root / "stack/scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy2(trainer, root / "stack/scripts/v8f_mini_train.py")
    (root / PK / "code").mkdir(parents=True, exist_ok=True)
    shutil.copy2(queue, root / PK / "code/v8f_mini_box2_queue.cmd")


# ================================================================================================ the chains' post-run assertions (embedded python heredocs of runone)
def record_tests(tag, chain_text, arms):
    """Run the chain's OWN two post-run heredocs (the run-record assertion and the REACHED control), extracted from the chain text, on synthetic run records: a good record passes, every doctored one is refused (rc 1).
    arms: [(arm, p, bank sha or 'none', extra built levers, extra argv tokens, extra selfcheck keys)]."""
    hds = re.findall(r"\$PY - [^\n]*<<'EOF'[^\n]*\n(.*?)\nEOF\n", chain_text, re.S)
    rec = [h for h in hds if "run record:" in h]
    rea = [h for h in hds if "swap-reached:" in h]
    check(f"{tag}0 the chain carries exactly one run-record heredoc and one REACHED-control heredoc (extracted and run below)", len(rec) == 1 and len(rea) == 1, (len(rec), len(rea)))
    if len(rec) != 1 or len(rea) != 1:
        return
    td = FW / f"rec_{tag}"
    if td.exists():
        shutil.rmtree(td)
    td.mkdir(parents=True)
    (td / "rec.py").write_text(rec[0], encoding="utf-8")
    (td / "rea.py").write_text(rea[0], encoding="utf-8")
    banked, wacc = "ab" * 32, 49.6969
    cnt = [0]
    for arm, p, bsha, xl, xa, xs in arms:
        base_argv = ["--out", "D:/o/M2b_s0_" + arm, "--init-ckpt", "D:/c.pt", "--steps", "3000"] + xa + (["--w-acc", "49.6969"] if arm == "dagoacc" else [])
        sha16 = hashlib.sha256(" ".join(base_argv).encode()).hexdigest()[:16]

        def make(**dv):
            cnt[0] += 1
            d = td / f"r{cnt[0]}"
            d.mkdir()
            planned = dv.get("planned", base_argv)
            (d / "train.txt").write_text("C:/py.exe D:/repo/stack/scripts/v8f_mini_train.py " + " ".join(planned), encoding="utf-8")
            (d / "pin.txt").write_text("C:/py.exe D:/repo/stack/scripts/v8f_mini_train.py " + " ".join(dv.get("pin", planned)), encoding="utf-8")
            lv = {"init_ckpt": {"sha256": banked}, "dagger_p": p, "dagger_bank_sha256": (None if bsha == "none" else bsha), **xl}
            if arm == "dagoacc":
                lv["w_acc"] = wacc
            lv.update(dv.get("lv", {}))
            sm = {"argv_sha": sha16, "done": True, "steps": 3000, "arm": "M2b", "seed": 0, "levers": lv, "dagger_n": 0 if p == 0 else 490, "dagger_eligible": 4900}
            sm.update(dv.get("sm", {}))
            sc = {"argv_sha": sha16, "declared_vs_built": "PASS", **xs}
            sc.update(dv.get("sc", {}))
            (d / "summary.json").write_text(json.dumps(sm))
            (d / "selfcheck.json").write_text(json.dumps(sc))
            (d / "argv.json").write_text(json.dumps({"argv": dv.get("aj_argv", planned), "sha": dv.get("aj_sha", sha16)}))
            return d

        def run_rec(d):
            a = [L.PY, posix(td / "rec.py"), posix(d), posix(d / "train.txt"), posix(d / "pin.txt"), arm, "0", str(p), "3000", banked, bsha]
            return sh(a + ([str(wacc)] if arm == "dagoacc" else []))
        rc, out = run_rec(make())
        check(f"{tag}1 {arm}: the chain's run-record assertion PASSES a good record (CONTROL for the doctored ones below)", rc == 0 and "PASS" in out, out[-400:])
        doctored = {
            "the pinned argv differs from the planned one": dict(pin=base_argv[:-1] + ["999"]),
            "argv.json is not the planned argv": dict(aj_argv=base_argv[:-1] + ["999"]),
            "argv.json sha differs": dict(aj_sha="0" * 16),
            "summary.json argv_sha differs": dict(sm={"argv_sha": "0" * 16}),
            "selfcheck.json argv_sha differs": dict(sc={"argv_sha": "0" * 16}),
            "summary.json done is false": dict(sm={"done": False}),
            "summary.json steps differ": dict(sm={"steps": 2999}),
            "summary.json seed differs": dict(sm={"seed": 1}),
            "selfcheck declared_vs_built is not PASS": dict(sc={"declared_vs_built": "FAIL"}),
            "the init checkpoint differs": dict(lv={"init_ckpt": {"sha256": "cd" * 32}}),
            "the built dagger_p differs": dict(lv={"dagger_p": p + 0.2}),
        }
        if bsha != "none":
            doctored["the built bank sha differs"] = dict(lv={"dagger_bank_sha256": "ef" * 32})
        else:
            doctored["the p = 0 arm built a bank"] = dict(lv={"dagger_bank_sha256": "ef" * 32})
        if arm == "dagoacc":
            doctored["the built w_acc differs from the argv's"] = dict(lv={"w_acc": 50.0})
        else:
            doctored["the cut is not built (o_interface full)"] = dict(lv={"o_interface": "full"})
            doctored["the planned argv does not name --o-interface cut"] = dict(planned=[x for x in base_argv if x not in ("--o-interface", "cut")])
            doctored["o_interface_probe did not pass"] = dict(sc={"o_interface_probe": {"pass": False}})
        for what, dv in doctored.items():
            rc, out = run_rec(make(**dv))
            regress(f"{tag}2 {arm}: the run-record assertion REFUSES a record where {what} (rc 1)", rc == 1 and "run record: PASS" not in out, (rc, out[-300:]))

        def reach(n, e):
            d = make(sm={"dagger_n": n, "dagger_eligible": e})
            a = [posix(d / "summary.json"), "0", str(p)] if arm == "dagoacc" else [posix(d / "summary.json"), arm, "0", str(p)]
            return sh([L.PY, posix(td / "rea.py")] + a)
        if p > 0:
            rc, out = reach(490, 4900)
            check(f"{tag}3 {arm}: the REACHED control PASSES dagger_n / dagger_eligible = 0.100 (CONTROL)", rc == 0 and "PASS" in out, out[-300:])
            rc, out = reach(0, 4900)
            regress(f"{tag}4 {arm}: the REACHED control REFUSES a swap that never happened (dagger_n 0)", rc == 1, out[-300:])
            rc, out = reach(980, 4900)
            regress(f"{tag}5 {arm}: the REACHED control REFUSES a swap rate of 0.20 (outside 0.1 +- 0.02)", rc == 1, out[-300:])
            rc, out = reach(490, 0)
            regress(f"{tag}6 {arm}: the REACHED control REFUSES zero eligible windows", rc == 1, out[-300:])
        else:
            rc, out = reach(0, 4900)
            check(f"{tag}3 {arm}: the REACHED control PASSES dagger_n exactly 0 (CONTROL)", rc == 0 and "PASS" in out, out[-300:])
            rc, out = reach(5, 4900)
            regress(f"{tag}4 {arm}: the REACHED control REFUSES a p = 0 arm that swapped 5 windows", rc == 1, out[-300:])
    shutil.rmtree(td, ignore_errors=True)


def section_f():
    say("== F. the FLAT-DAG chain script")
    CHN = "chain_v8f_flatdag.sh"
    chain = (SHIP / CHN).read_text(encoding="utf-8")
    refdir = W / "flatref"
    for s in (0, 1):
        (refdir / f"s{s}").mkdir(parents=True, exist_ok=True)
        (refdir / f"s{s}" / "argv.json").write_bytes(git_blob(f"{PK}/raw/2026-10-26-flat/runs/M2b_s{s}_flat/argv.json"))
    ok = all(sh([BASH, "-n", posix(SHIP / f)])[0] == 0 for f in (CHN, "oaf_lib.sh", "v8f_guard_flatdag.sh", "wake_v8fflatdag.sh"))
    check("F1 bash -n: the FLAT-DAG chain, the lib, its guard copy and its waker parse", ok)
    env = {"V8F_QUEUE": posix(L.QUEUE), "FLATDAG_REF_DIR": posix(refdir)}
    rc, out = bash("--dry-run", env=env, chain=CHN)
    check("F2 --dry-run: rc 0; the four runs in PREREG order; on every planned line gate == train, FLAT-CONSISTENT (FLAT's banked argv minus the fine-tune flags, --o-interface cut named, init FLAT_s<S>); both argv.json references hash to the pins",
          rc == 0 and out.count("FLAT-CONSISTENT") == 4 and "NOT-FLAT-CONSISTENT" not in out and out.count(": SAME") == 4 and out.count("sha256 == the pinned value") == 2 and not re.search(r"(?m)^ZZV8FMINI-STOP", out)
          and [m for m in re.findall(r"(?m)^== (M2b_s\d_flatdag0?):", out)] == ["M2b_s0_flatdag", "M2b_s0_flatdag0", "M2b_s1_flatdag", "M2b_s1_flatdag0"], out[-500:])
    # the pins are the banked files' hashes
    import make_chains as MC
    fp = MC.flat_pins()
    arr = lambda name: re.search(rf"(?m)^{name}=\(([0-9a-f ]+)\)", chain).group(1).split()
    bad_pin = [k for k in ("CKPT_SHA", "ARGV_SHA", "SUMM_SHA") if arr(k) != fp[k]]
    ck_files = [sha_file(f"E:/Projects/TanitAD-artifacts/v8f-mini-flat/M2b_s{s}_flat.pt") for s in (0, 1)]
    check("F3 the pinned hashes in the chain are the banked FLAT files' (checkpoints re-hashed from v8f-mini-flat/, argv.json / summary.json from the repo blobs); the prefixes are 8496934a / 6fdfc9b0",
          not bad_pin and arr("CKPT_SHA") == ck_files and ck_files[0].startswith("8496934a") and ck_files[1].startswith("6fdfc9b0"), (bad_pin, ck_files))
    check("F4 the run list is flatdag s0, flatdag0 s0, flatdag s1, flatdag0 s1; P = 0.1; the chain's floor is 1 GiB; the bank directory is dagger_bank_flat and the M2b bank directory is named nowhere",
          'RUNS=("flatdag 0" "flatdag0 0" "flatdag 1" "flatdag0 1")' in chain and re.search(r"(?m)^PVAL=0\.1\b", chain) is not None and re.search(r"(?m)^FLOOR_BYTES=1073741824\b", chain) is not None
          and "dagger_bank_flat" in chain and "dagger_bank/s" not in chain and "dagger_bank_dag2" not in chain)
    # find(), never index(): a REMOVED line must read as a FAILED check, not as a crash; the bank / judge / publish order is read in the REAL run's segment (the --publish-only block comes first in the file and has the same lines)
    i_f1, i_arch, i_gates, i_runs = (chain.find(x) for x in ('lever_f1 git "$T"', "archive --format=tar", 'for run in "${RUNS[@]}"; do set -- $run; gate_one', 'for run in "${RUNS[@]}"; do set -- $run; runone'))
    seg = chain[chain.rfind("for s in 0 1; do bank_one $s; done"):]
    f1_line = re.search(r'(?m)^lever_f1 git "\$T" \|\| \{ echo "ZZFLATDAG-F1-STOP[^\n]*"; stop "F1 not landed"; \}\r?$', chain)
    s_judge, s_pub, s_gate = seg.find('judge_banks "$ART/bank_s0"'), seg.find("publish_bank_one $s; done"), seg.find('for run in "${RUNS[@]}"; do set -- $run; gate_one')
    check("F5 order in the real run: the F1 check (in its exact STOP form) precedes the code archive; all four gates precede the first training; in the real run's segment the bank is judged once, published once (AFTER the judge) and the gates follow (judge before publish before gate)",
          f1_line is not None and 0 <= i_f1 < i_arch and 0 <= i_gates < i_runs and 0 <= s_judge < s_pub < s_gate and seg.count("publish_bank_one $s; done") == 1 and seg.count('judge_banks "$ART/bank_s0"') == 1
          and "ZZFLATDAG-F1-STOP" in chain[i_f1:i_arch], (i_f1, i_arch, i_gates, i_runs, s_judge, s_pub, s_gate))
    # ---- --check-tree: the lever greps AND F1
    tip_trainer = PYTREE / "stack/scripts/v8f_mini_train.py"
    fixed_trainer = Path("E:/Projects/TanitAD-artifacts/oaccflat_check/stage/stack/scripts/v8f_mini_train.py")
    ct = FW / "ft_tree"
    _tree_for_flat(ct, fixed_trainer, L.QUEUE)
    rc, out = bash(f"--check-tree '{posix(ct)}'", chain=CHN)
    check("F6 --check-tree on a tree with the F1-fixed trainer and the staged queue: every lever pattern is there and F1 holds", rc == 0 and "F1 holds" in out and "FAILED" not in out, out[-400:])
    _tree_for_flat(ct, tip_trainer, L.QUEUE)
    rc, out = bash(f"--check-tree '{posix(ct)}'", chain=CHN)
    regress("F7 --check-tree REFUSES the tip's trainer (INIT_CKPT_MATCH does not name o_interface): ZZFLATDAG-F1-STOP, rc 1, nothing would be touched on box 2", rc == 1 and "ZZFLATDAG-F1-STOP" in out and "MISSING" not in out, out[-400:])
    _tree_for_flat(ct, fixed_trainer, L.TIP_QUEUE)
    rc, out = bash(f"--check-tree '{posix(ct)}'", chain=CHN)
    regress("F8 --check-tree REFUSES a tree whose queue is the tip's (no FLAT-DAG stages)", rc == 1 and "MISSING" in out, out[-300:])
    shutil.rmtree(ct, ignore_errors=True)
    # ---- --judge-only: FLAT's shas against banks whose generator is M2b (K-B2) -> REFUSED; the same machinery with the generator the banks have -> PASS
    d1 = DAGART
    jd = FW / "ship_f"
    if jd.exists():
        shutil.rmtree(jd)
    shutil.copytree(SHIP, jd)
    rc, out = sh([BASH, "-c", f"cd '{posix(jd)}' && bash {CHN} --judge-only '{posix(d1 / 'bank_s0')}' '{posix(d1 / 'bank_s1')}'"])
    regress("F9 --judge-only: the M2b banks against FLAT's checkpoint shas -> ZZFLATDAG-REFUSED (K-B2: a FLAT bank must be generated by FLAT), rc 2, ZZV8FMINI-STOP REFUSED", rc == 2 and "ZZFLATDAG-REFUSED" in out and "ZZFLATDAG-JUDGE-PASS" not in out and "ZZV8FMINI-STOP REFUSED" in out, out[-400:])
    m2b = MC.dag_pins()["M2B_SHA"]
    ctext = (jd / CHN).read_text(encoding="utf-8")
    old_line = re.search(r"(?m)^CKPT_SHA=\([0-9a-f ]+\)", ctext).group(0)
    (jd / CHN).write_bytes(ctext.replace(old_line, f"CKPT_SHA=({m2b[0]} {m2b[1]})").encode("utf-8"))
    rc, out = sh([BASH, "-c", f"cd '{posix(jd)}' && bash {CHN} --judge-only '{posix(d1 / 'bank_s0')}' '{posix(d1 / 'bank_s1')}'"])
    check("F10 --judge-only (CONTROL: the same chain with the generator shas the banks actually have): ZZFLATDAG-JUDGE-PASS, rc 0 -- so F9's refusal is the generator check and nothing else", rc == 0 and "ZZFLATDAG-JUDGE-PASS" in out and "ZZFLATDAG-REFUSED" not in out, out[-400:])
    # ---- --publish-only on the real iteration-1 banks (644 MB each) through the FLAT paths, into a scratch directory on C: (the chain's patched copy)
    free = shutil.disk_usage(FW).free
    if os.environ.get("OAF_SKIP_F11") or free < 4_000_000_000:
        say("SKIP  F11 --publish-only: " + ("OAF_SKIP_F11 is set (the mutant runs: six concurrent copies of two 644 MB banks off the USB drive; the CONTROL run executes F11)" if os.environ.get("OAF_SKIP_F11") else f"the scratch volume has only {free / 1e9:.1f} GB free (the two 644 MB banks are copied)") + "; not run")
    else:
        fk, art = FW / "fakebox_f", FW / "art_f"
        for dd in (fk, art):
            if dd.exists():
                subprocess.run(["cmd", "/c", "rmdir", "/s", "/q", str(dd)], capture_output=True)
        (fk / "dagger_bank_flat").mkdir(parents=True)
        for s in (0, 1):
            subprocess.run(["cmd", "/c", "mklink", "/J", str(fk / "dagger_bank_flat" / f"s{s}"), str(d1 / f"bank_s{s}")], capture_output=True, check=True)
        rc, out = sh([BASH, "-c", f"cd '{posix(jd)}' && FLATDAG_ART='{posix(art)}' bash {CHN} --publish-only '{posix(fk)}'"], timeout=900)
        sums = all((art / f"bank_s{s}" / "SHA256SUMS.txt").exists() for s in (0, 1))
        check("F11 --publish-only (the real fetch -> judge -> publish on a stand-in for box 2, the 644 MB iteration-1 banks through the FLAT paths): rc 0, ZZFLATDAG-PUBLISH-ONLY-DONE, both seeds sealed (SHA256SUMS.txt, bank-verify --require-sums)", rc == 0 and "ZZFLATDAG-PUBLISH-ONLY-DONE" in out and sums, out[-500:])
        ok_sums = sums
        for s in (0, 1):
            sf = art / f"bank_s{s}" / "SHA256SUMS.txt"
            lines = sf.read_text().splitlines() if sf.exists() else []
            ok_sums = ok_sums and len(lines) >= 11 and all(sha_file(art / f"bank_s{s}" / ln.split(" *", 1)[-1].strip()) == ln.split()[0] for ln in lines[:3])
        check("F12 the published files hash to what SHA256SUMS.txt says (first three re-hashed here)", ok_sums)
        subprocess.run(["cmd", "/c", "rmdir", "/s", "/q", str(fk)], capture_output=True)       # junctions: removes the links, not the banks
        shutil.rmtree(art, ignore_errors=True)
    shutil.rmtree(jd, ignore_errors=True)
    # ---- the predecessor branches (the shared lib: one GO and one HALT per chain) and the early stop
    fb = FW / "ship_fw"

    def wait(log_text, busy):
        if fb.exists():
            shutil.rmtree(fb)
        (fb / "logs").mkdir(parents=True)
        (fb / "logs" / "pred.out").write_text(log_text, newline="")
        if busy:
            (fb / "BOX2_BUSY.txt").write_text("busy")
        return bash("--wait-only", env={"OAF_SHIP_DIR": posix(fb), "AFTER_OUT": posix(fb / "logs" / "pred.out"), "OAF_POLL_S": "1", "OAF_GRACE_S": "1"}, timeout=60, chain=CHN)
    rc, out = wait("a\nZZV8FMINI-STOP bank s0: BLOCKED (COVERAGE)\n", False)
    check("F13 wait: the DAG-OACC chain (the predecessor) STOPPED on its own, box released -> ZZOAF-WAIT-GO (box 2 is not idled)", rc == 0 and "ZZOAF-WAIT-GO" in out, out[-300:])
    rc, out = wait("a\nZZGUARD-STOP TEMP 10-27 USB 64 C\nZZV8FMINI-STOP guard stop\n", False)
    regress("F14 wait: the predecessor was stopped by the GUARD -> ZZV8FMINI-STOP, no GO", rc == 1 and "ZZOAF-WAIT-GO" not in out and "predecessor verdict: HALT" in out, out[-300:])
    rc, out = wait("a\nZZV8FMINI-STOP bank s0: BLOCKED\n", True)
    regress("F15 wait: the predecessor STOPPED but BOX2_BUSY is still held -> ZZV8FMINI-STOP, no GO", rc == 1 and "ZZOAF-WAIT-GO" not in out, out[-300:])
    rc, out = wait("a\nZZV8FMINI-DONE done\n", False)
    check("F16 wait: the predecessor DONE, box released -> GO", rc == 0 and "ZZOAF-WAIT-GO" in out, out[-300:])
    shutil.rmtree(fb, ignore_errors=True)
    shim = W / "shim_f"
    shim.mkdir(parents=True, exist_ok=True)
    slog = W / "shim_f.log"
    for nm in ("ssh", "scp"):
        (shim / nm).write_text("#!/usr/bin/env bash\necho " + nm + " \"$@\" >> \"$SHIM_LOG\"\nexit 1\n", newline="\n")
    sp = posix(shim)
    sp = "/" + sp[0].lower() + sp[2:] if sp[1:2] == ":" else sp
    pre = "export PATH='" + sp + "':\"$PATH\"; "
    e2 = {"SHIM_LOG": posix(slog)}
    slog.write_text("")
    sh([BASH, "-c", pre + "ssh control-call; scp control-call"], env=e2)
    ctl = slog.read_text()
    slog.write_text("")
    rc, out = bash("", env=e2, chain=CHN, pre=pre)
    check("F17 the real entry without AFTER_OUT: ZZV8FMINI-STOP, rc 1, no ssh / scp call (the shim control read non-empty first)", "ssh control-call" in ctl and rc == 1 and "ZZV8FMINI-STOP AFTER_OUT is not set" in out and slog.read_text() == "", out[-300:])
    # F20: the wait on the REAL path (F13-F16 run --wait-only): a predecessor that has not ended -> still waiting, BOX2_BUSY not taken, no ssh / scp call
    wb = FW / "ship_f20"
    if wb.exists():
        shutil.rmtree(wb)
    (wb / "logs").mkdir(parents=True)
    (wb / "logs" / "pred.out").write_text("10-27 10:13 waiting for the predecessor chain (x: end marker ^ZZV8FMINI-DONE, stop marker ^ZZV8FMINI-STOP)\nsome log line\n", newline="")
    slog.write_text("")
    p = subprocess.run([BASH, "-c", pre + f"cd '{posix(SHIP)}' && OAF_SHIP_DIR='{posix(wb)}' AFTER_OUT='{posix(wb / 'logs' / 'pred.out')}' OAF_POLL_S=1 timeout 8 bash {CHN}"], capture_output=True, text=True, env={**os.environ, **e2}, stdin=subprocess.DEVNULL)
    regress("F20 the REAL entry with a predecessor that has not ended: still waiting after 8 s (killed by the test's timeout, rc 124), BOX2_BUSY not taken, no ssh / scp call", p.returncode == 124 and not (wb / "BOX2_BUSY.txt").exists() and slog.read_text() == "" and "ZZOAF-WAIT-GO" not in (p.stdout or ""), (p.returncode, (p.stdout or "")[-300:], slog.read_text()[-200:]))
    shutil.rmtree(wb, ignore_errors=True)
    # ---- F18 / F19: the REAL entry's F1 stop, BEHAVIOURALLY (not only its text): a fake mirror = a git repo holding exactly the lever files (origin = itself), a box-2 shim that answers only the boot probe
    fm, fo, fs_, fa = FW / "fm_f", FW / "fm_f_origin", FW / "ship_f18", FW / "art_f18"

    def rmtree_rw(d):
        if d.exists():
            shutil.rmtree(d, onerror=lambda f, pth, ei: (os.chmod(pth, 0o700), f(pth)))

    def real_entry(trainer):
        rmtree_rw(fm)
        rmtree_rw(fo)
        _tree_for_flat(fm, trainer, L.QUEUE)
        gg = lambda *a: sh(["git", "-C", posix(fm), "-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.autocrlf=false", *a])
        gg("init", "-q")
        gg("symbolic-ref", "HEAD", "refs/heads/agent/arch-inf-20260803")
        gg("add", "-A")
        gg("commit", "-q", "-m", "fake tip")
        sh(["git", "init", "-q", "--bare", posix(fo)])
        gg("remote", "add", "origin", posix(fo))                                  # a SEPARATE origin: with origin = the mirror itself `ls-remote origin <branch>` matches refs/heads/ AND refs/remotes/origin/ (two lines)
        gg("push", "-q", "origin", "agent/arch-inf-20260803")
        rmtree_rw(fs_)
        (fs_ / "logs").mkdir(parents=True)
        (fs_ / "logs" / "pred.out").write_text("a\nZZV8FMINI-DONE done\n", newline="")
        sdir = W / "shim_f18"
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / "ssh").write_text('#!/usr/bin/env bash\necho ssh "$@" >> "$SHIM_LOG"\ncase "$*" in *LastBootUpTime*) echo "2026-10-27 08:00:00";; esac\nexit 0\n', newline="\n")
        (sdir / "scp").write_text('#!/usr/bin/env bash\necho scp "$@" >> "$SHIM_LOG"\nexit 1\n', newline="\n")
        lg = W / "shim_f18.log"
        lg.write_text("")
        spp = posix(sdir)
        spp = "/" + spp[0].lower() + spp[2:] if spp[1:2] == ":" else spp
        e3 = {"SHIM_LOG": posix(lg), "OAF_SHIP_DIR": posix(fs_), "AFTER_OUT": posix(fs_ / "logs" / "pred.out"), "OAF_POLL_S": "1", "OAF_GRACE_S": "1", "OAF_MIRROR": posix(fm), "OAF_BR": "agent/arch-inf-20260803", "FLATDAG_ART": posix(fa)}
        rc_, out_ = bash("", env=e3, chain=CHN, pre="export PATH='" + spp + "':\"$PATH\"; ", timeout=120)
        return rc_, out_, lg.read_text()
    rc, out, slg = real_entry(tip_trainer)
    regress("F18 the REAL entry against a fake mirror whose trainer is the tip's (INIT_CKPT_MATCH without o_interface): ZZFLATDAG-F1-STOP + ZZV8FMINI-STOP F1 not landed, rc 1, BOX2_BUSY released, the only box-2 call is the boot probe (no scp, no code archive)",
            rc == 1 and "ZZFLATDAG-F1-STOP" in out and "ZZV8FMINI-STOP F1 not landed" in out and "scp" not in slg and slg.count("ssh ") == 1 and "LastBootUpTime" in slg and not (fs_ / "BOX2_BUSY.txt").exists() and not list(fs_.glob("repo_v8fflatdag_tip.tar")), (rc, out[-1500:], slg[-300:]))
    rc, out, slg = real_entry(fixed_trainer)
    check("F19 the same REAL entry with the F1-fixed trainer (CONTROL): no ZZFLATDAG-F1-STOP; it goes past F1 and stops at the export-list check of the fake tip (so F18's stop is the F1 check and nothing else), still without an scp call",
          rc == 1 and "ZZFLATDAG-F1-STOP" not in out and "INIT_CKPT_MATCH has" in out and "names o_interface" in out and "ZZV8FMINI-STOP the tip has no" in out and "scp" not in slg, (rc, out[-1500:], slg[-300:]))
    rmtree_rw(fm)
    rmtree_rw(fo)
    rmtree_rw(fs_)

    record_tests("FR", chain, [("flatdag", 0.1, "cd" * 32, {"o_interface": "cut"}, ["--o-interface", "cut", "--dagger-p", "0.1"], {"o_interface_probe": {"pass": True}}), ("flatdag0", 0.0, "none", {"o_interface": "cut"}, ["--o-interface", "cut", "--dagger-p", "0"], {"o_interface_probe": {"pass": True}})])

# ================================================================================================ main
def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    secs = argv[argv.index("--sections") + 1].split(",") if "--sections" in argv else ["A", "S", "B", "C", "D", "E", "F"]
    t0 = time.time()
    for s in secs:
        {"A": section_a, "S": section_s, "B": section_b, "C": section_c, "D": section_d, "E": section_e, "F": section_f}[s]()
    say(f"\nRESULT: {N['check'] - N['check_fail']}/{N['check']} checks pass, {N['reg'] - N['reg_missed']}/{N['reg']} deliberate regressions caught, {N['reg_missed']} missed; {N['check_fail']} failed checks ({time.time() - t0:.0f} s)")
    if "--out" in argv:
        Path(argv[argv.index("--out") + 1]).write_text("\n".join(LOG) + "\n", encoding="utf-8")
    return 0 if N["check_fail"] == 0 and N["reg_missed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
