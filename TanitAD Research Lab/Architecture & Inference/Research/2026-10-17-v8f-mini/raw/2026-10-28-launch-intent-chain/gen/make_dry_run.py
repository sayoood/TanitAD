#!/usr/bin/env python3
"""Write DRY_RUN.out: the offline runs of the LAUNCH-INTENT chain (four runs: A_0, A_1, A_rep,0, A_rep,1) against an EXPORT OF THE LANDED COMMIT b09ac89f (the flag and the model are in it; the queue stage is NOT) with the STAGED queue laid over it.

  1  the landed export + the staged queue = the tree the Master Mind's queue landing is expected to produce (CPU, nothing touches box 2, box2_ship, git (read-only `git archive` on the mirror) or the network)
  2  --check-tree on the PRE-LEVER tip 94a47f4b export             -> REFUSES: every new pattern missing
  2b --check-tree on the LANDED export (flag yes, queue stage no)  -> REFUSES, for the queue stage alone (the trainer and model sha256 are the pinned ones)
  3  --check-tree on the overlay with the SHIPPED chain            -> PASS (trainer, model sha256 = the landed pins; the PREREG / reader pins are not needed by a tree check)
  4  --check-tree on the overlay with a SCRATCH chain pinned to a wrong trainer sha256 -> REFUSES (the pin is load-bearing)
  5  --dry-run (shipped chain, V8F_QUEUE = the staged queue): five pins (PREREG_SHA / READER_SHA UNSET by design), gate plan, SAME and ONE-LEVER for all four runs, the K-SHARE / K-MASK bands, no warning line
  6  --dry-run against the LANDED commit's own queue (no :launchintentarm)  -> STOP (the queue prints no gate line)
  7  the predecessor verdict on the REAL DAG-ORC r2 log (read-only), the REP licence on the REAL banked pulls (read-only)
  8  the shipped chain as a REAL run (scratch world, stub ssh): REFUSED at once because PREREG_SHA and READER_SHA are placeholders
EVERY bash step runs through sealed.run (gen/sealed.py): the guard asserts, in the same shell, the stubs, the unresolved box-2 alias, the empty HOME and the scratch-only write paths before the step's text runs.
    python make_dry_run.py          writes ../DRY_RUN.out
"""
from __future__ import annotations

import io
import re
import shutil
import sys
import tarfile
from pathlib import Path

import common as C
import sealed

SCR = C.SCRATCH / "dry"
OUTF = C.OUT / "DRY_RUN.out"
LOG = []
px = sealed.px


def say(s=""):
    LOG.append(s)
    print(s)


def sh(cmd, env=None, cwd=None, timeout=600, need=()):
    return sealed.run(cmd, env=env, cwd=cwd, timeout=timeout, need=need)


def export_tip(dst: Path, paths, ref: str | None = None):
    tip = C.tip_full(ref)
    tar = C.git("archive", "--format=tar", tip, "--", *paths, binary=True)
    with tarfile.open(fileobj=io.BytesIO(tar)) as t:
        t.extractall(dst)


def main() -> int:
    sealed.ensure_global()
    shutil.rmtree(SCR, ignore_errors=True)
    tipd, pred, ov, shipd = SCR / "tip_export", SCR / "pre_export", SCR / "overlay", SCR / "ship"
    for d in (tipd, pred, ov, shipd):
        d.mkdir(parents=True)
    chain = (C.SHIP / "chain_v8f_launchintent.sh").read_text(encoding="utf-8")
    exp = re.findall(r"^  '([^']+)'$", re.search(r"^EXPORT=\(\n(.*?)\n\)\n", chain, re.M | re.S).group(1), re.M)
    paths = sorted(set(exp + [C.QUEUE_PATH, f"{C.PK}/raw/eval_grid_windows.json"]))
    export_tip(tipd, paths)
    export_tip(pred, paths, C.PRE_TIP)
    shutil.copytree(tipd, ov, dirs_exist_ok=True)
    staged = []
    for rel in ("stack/scripts/v8f_mini_train.py", "stack/tanitad/models/v8f_mini.py", "stack/tests/test_v8f_launch_intent.py", "stack/tests/launch_intent_golden.json"):
        dst = ov / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(C.STAGE / rel, dst)
        staged.append(rel)
    qdst = ov / C.QUEUE_PATH
    shutil.copyfile(C.QSTAGE / "v8f_mini_box2_queue.cmd", qdst)
    tr_sha, md_sha = C.sha(ov / "stack/scripts/v8f_mini_train.py"), C.sha(ov / "stack/tanitad/models/v8f_mini.py")
    # a ship dir = the generated files + the INSTALLED helpers the chain pins (copies; nothing is installed)
    for f in C.SHIP.iterdir():
        if f.is_file():
            shutil.copyfile(f, shipd / f.name)
    for f in ("oaf_lib.sh", "v8f_oaf_chain_checks.py", "mk_v8f_job.py", "waitjob.sh"):
        shutil.copyfile(C.INSTALLED / f, shipd / f)
    wj = (shipd / "waitjob.sh").read_bytes()                       # the installed helpers hardcode C:/Users/Admin/box2_ship: the scratch copies are pointed at the scratch ship dir
    assert wj.count(b"L=/c/Users/Admin/box2_ship/logs;") == 1
    (shipd / "waitjob.sh").write_bytes(wj.replace(b"L=/c/Users/Admin/box2_ship/logs;", ("L=" + px(shipd) + "/logs;").encode()))
    mk = (shipd / "mk_v8f_job.py").read_bytes()
    assert mk.count(b"C:/Users/Admin/box2_ship/") == 1
    (shipd / "mk_v8f_job.py").write_bytes(mk.replace(b"C:/Users/Admin/box2_ship/", (px(shipd) + "/").encode()))
    (shipd / "logs").mkdir(exist_ok=True)
    say("DRY_RUN.out of the LAUNCH-INTENT chain, FOUR runs (gen/make_dry_run.py). CPU only; box 2 never contacted; nothing installed into box2_ship; no git write.")
    say(f"every bash step below ran behind the sealed guard (gen/sealed.py): stubs for ssh and scp first on PATH inside the shell, the box-2 alias unresolved (ssh -G), HOME = the empty scratch dir, every write path inside {C.SCRATCH.as_posix()}")
    say(f"the LANDED commit {C.LANDED_COMMIT} (coordinator message 2026-10-28; ls-remote verified) = {C.tip_full()} ({len(paths)} exported paths); overlay = that export + the STAGED files {staged} + the STAGED queue ({C.sha(qdst)[:16]})")
    say(f"trainer sha256 {tr_sha} (pinned LAUNCH_TRAINER_SHA {C.LANDED_TRAINER_SHA}; equal: {tr_sha == C.LANDED_TRAINER_SHA})")
    say(f"model   sha256 {md_sha} (pinned LAUNCH_MODEL_SHA   {C.LANDED_MODEL_SHA}; equal: {md_sha == C.LANDED_MODEL_SHA})")
    say(f"the pre-lever tip {C.tip_full(C.PRE_TIP)}: trainer blob sha256 {C.sha_b(C.git('show', f'{C.tip_full(C.PRE_TIP)}:stack/scripts/v8f_mini_train.py', binary=True))}")
    say()
    env = {"OAF_SHIP_DIR": px(shipd), "TMPDIR": px(SCR)}
    chain_p = px(shipd / "chain_v8f_launchintent.sh")
    need = ("OAF_SHIP_DIR", "TMPDIR")
    say("=" * 30 + " 2. --check-tree on the PRE-LEVER tip 94a47f4b export (must REFUSE: the lever and the queue stage are not there)")
    rc, o = sh(f'bash "{chain_p}" --check-tree "{px(pred)}"', env, need=need)
    say(o.rstrip()); say(f"rc={rc}"); say()
    say("=" * 30 + " 2b. --check-tree on the LANDED export (the flag and the model are in; the queue stage is NOT: must REFUSE for the queue stage alone)")
    rc, o = sh(f'bash "{chain_p}" --check-tree "{px(tipd)}"', env, need=need)
    say(o.rstrip()); say(f"rc={rc}"); say()
    say("=" * 30 + " 3. --check-tree on the OVERLAY with the SHIPPED chain (must PASS: trainer and model sha256 are the landed pins; a tree check needs neither the PREREG nor the reader pin)")
    rc, o = sh(f'bash "{chain_p}" --check-tree "{px(ov)}"', env, need=need)
    say(o.rstrip()); say(f"rc={rc}"); say()
    say("=" * 30 + " 4. --check-tree on the OVERLAY with a SCRATCH chain pinned to a WRONG trainer sha256 (0 x 64): must REFUSE")
    sc_chain = shipd / "chain_v8f_launchintent_wrongpin.sh"
    rc0, o0 = sh(f'"{C.PYV}" make_chain_li.py --trainer-sha {"0" * 64} --ship-dir "{px(shipd)}" --out "{px(sc_chain)}"', cwd=str(C.HERE))
    say(o0.strip().splitlines()[-1][:300] if o0.strip() else "")
    rc, o = sh(f'bash "{px(sc_chain)}" --check-tree "{px(ov)}"', env, need=need)
    say(o.rstrip()); say(f"rc={rc}"); say()
    say("=" * 30 + " 5. --dry-run (the SHIPPED chain, V8F_QUEUE = the staged queue; AFTER_OUT = the real DAG-ORC r2 log, read only)")
    env5 = {**env, "V8F_QUEUE": px(qdst), "AFTER_OUT": "/c/Users/Admin/box2_ship/logs/chain_v8f_dagoaccorc_r2.out", "LI_TREE": px(ov)}
    rc, o = sh(f'bash "{chain_p}" --dry-run', env5, need=need)
    say(o.rstrip()); say(f"rc={rc}")
    bad = [x for x in o.split("\n") if x.startswith("!" * 2)]
    nsame, nol = o.count(": SAME"), o.count(": ONE-LEVER")
    say(f"CHECKS: rc={rc} | SAME lines {nsame} (expected 4) | ONE-LEVER lines {nol} (expected 4) | warning lines {len(bad)}"); say()
    say("=" * 30 + " 6. --dry-run against the LANDED commit's own queue (no :launchintentarm: must STOP, not print a plan)")
    tq = tipd / C.QUEUE_PATH
    rc, o = sh(f'bash "{chain_p}" --dry-run', {**env5, "V8F_QUEUE": px(tq)}, need=need)
    say("\n".join(x for x in o.split("\n") if x.startswith(("ZZV8FMINI", "== runs", "REFUSED"))) or o[-600:]); say(f"rc={rc}"); say()
    say("=" * 30 + " 7a. the predecessor verdict on the REAL DAG-ORC r2 log (read-only; GO only after its DONE with BOX2_BUSY released)")
    rc, o = sh(f'"{C.PYV}" "{(C.SHIP / "v8f_launchintent_chain_checks.py").as_posix()}" pred-verdict-li --out C:/Users/Admin/box2_ship/logs/chain_v8f_dagoaccorc_r2.out --busy C:/Users/Admin/box2_ship/BOX2_BUSY.txt')
    say(o.rstrip()); say()
    say("=" * 30 + " 7b. --rep-licence on the REAL banked pulls of BOTH seeds (read-only)")
    rc, o = sh(f'AFTER_OUT=/c/Users/Admin/box2_ship/logs/chain_v8f_dagoaccorc_r2.out bash "{chain_p}" --rep-licence', env, need=need)
    say("\n".join(x[:420] for x in o.rstrip().split("\n"))); say(f"rc={rc}"); say()
    say("=" * 30 + " 8. the SHIPPED chain as a REAL run in the scratch world (stub ssh / scp): must be REFUSED at once, before the predecessor is waited for (PREREG_SHA and READER_SHA are placeholders)")
    (shipd / "logs" / "chain_v8f_dagoaccorc_r2.out").write_bytes(b"")
    env8 = {**env, "AFTER_OUT": px(shipd / "logs" / "chain_v8f_dagoaccorc_r2.out"), "LAUNCHINTENT_ART": px(SCR / "art"), "LAUNCHINTENT_NEED_BYTES": "1000", "LI_STUB_DIR": str(shipd / "stub"), "LI_STUB_LOG": px(shipd / "stub_calls.log")}
    sealed.make_stub_dir(shipd / "stub")
    (shipd / "stub_calls.log").write_bytes(b"")
    rc, o = sh(f'bash "{chain_p}"', env8, timeout=120, need=need + ("LAUNCHINTENT_ART",))
    say(o.rstrip()); say(f"rc={rc} | stub ssh/scp calls: {len([x for x in (shipd / 'stub_calls.log').read_text(errors='replace').split(chr(10)) if x.strip()])} | BOX2_BUSY.txt created: {(shipd / 'BOX2_BUSY.txt').exists()}")
    say(f"sealed runs in this file: {sealed.STATS['sealed_runs']}, refused by the guard: {sealed.STATS['refused']}")
    OUTF.write_bytes(("\n".join(LOG) + "\n").encode("utf-8"))
    print("wrote", OUTF)
    return 0


if __name__ == "__main__":
    sys.exit(main())
