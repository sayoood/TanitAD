#!/usr/bin/env python3
"""Render box2_ship/chain_v8f_launchintent.sh from gen/chain_launchintent.sh.in: the box-2 chain for PREREG_V8F_LAUNCH_INTENT, arm A: FOUR runs (A_0, A_1, A_rep,0, A_rep,1; MM_REDESIGN_2026-10-28 section 3).

Same method as the DAG-ORC r2 chain's generator (E:/Projects/TanitAD-artifacts/dagorc_r2/gen/make_chain_r2.py, the model of this file; NOT touched): every pinned value is COMPUTED from a banked artifact or a reviewed file, never retyped.
  DAG-OACC's AND REP's (DAG-ORC r2's M2b_s<S>_dagoaccorcrep) checkpoint / argv.json / summary.json sha256, M2b's checkpoint sha256, the iteration-1 bank sha256, w_acc and its file's sha256, the train window list sha256 and the K-BUILD reference commit  <-  v8f-dagoacc/, v8f-dag-orc-oacc/ (and the INSTALLED DAG-OACC r2 chain's arrays, cross-checked)
  the sha256 of every reviewed helper (asserter, checks, lib, guard) and of the INSTALLED oaf_lib.sh / v8f_oaf_chain_checks.py  <-  the files themselves
  the EXPORT list  <-  the INSTALLED chain_v8f_dagoacc_r2.sh (the import closure of the entry scripts)
  the lever greps  <-  LEVER below, VALIDATED by --check-patterns against the staged trainer/queue (each new pattern must also FAIL on the tip's blobs)
FIVE pins are set by hand, in two groups:
  PINNED TO THE LANDED COMMIT (coordinator message 2026-10-28; common.py LANDED_*; RE-MEASURED against the mirror's `git show` here):  LAUNCH_COMMIT b09ac89f.., LAUNCH_TRAINER_SHA 0d708b3c.., LAUNCH_MODEL_SHA b337ccec..
  NAMED PLACEHOLDERS the launch step fills:  PREREG_SHA (the landed PREREG v2) and READER_SHA (the reader that scores the runs). A real run REFUSES while either is unset; --check-tree and --dry-run do not need them.

    python make_chain_li.py [--trainer-sha H] [--model-sha H] [--commit C40] [--prereg-sha H] [--reader-sha H]     writes box2_ship/chain_v8f_launchintent.sh (LF); a pin not given keeps the value already on disk (first write: the landed values and the two placeholders)
    python make_chain_li.py --check                      regenerates in memory (the pin lines are READ from the file on disk) and compares
    python make_chain_li.py --check-patterns             every LEVER pattern against the staged trainer + queue (must match) and against the tip's blobs (the new ones must NOT)
    --ship-dir DIR --out FILE                            (self-test hooks) read the helper files from DIR, write FILE
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import common as C

TEMPLATE = C.HERE / "chain_launchintent.sh.in"
PH_PIN = '<<UNSET: sha256 of the landed stack/scripts/v8f_mini_train.py blob (LAUNCH.txt)>>'
PH_MODEL = '<<UNSET: sha256 of the landed stack/tanitad/models/v8f_mini.py blob (LAUNCH.txt)>>'
PH_PREREG = '<<UNSET: sha256 of the LANDED PREREG_V8F_LAUNCH_INTENT v2 (the launch step fills it: LAUNCH.txt)>>'
PH_READER = '<<UNSET: sha256 of the LAUNCH-INTENT reader that scores the runs (the launch step fills it: LAUNCH.txt)>>'
PIN_NAMES = ("LAUNCH_COMMIT", "LAUNCH_TRAINER_SHA", "LAUNCH_MODEL_SHA", "PREREG_SHA", "READER_SHA")
TR = "stack/scripts/v8f_mini_train.py"
MD = "stack/tanitad/models/v8f_mini.py"
DG = "stack/tanitad/data/v8f_dagger.py"
Q = C.QUEUE_PATH

# (file-or-"$PK queue", ERE, NEW?)  NEW = introduced by this build: must be ABSENT from the tip's blob and PRESENT in the staged one.
LEVER_BASE = [
    (TR, r'add_argument\("--w-acc"', False), (TR, r'add_argument\("--dagger-p"', False), (TR, r'add_argument\("--dagger-bank"', False), (TR, r'add_argument\("--dagger-bank-sha256"', False),
    (TR, r'add_argument\("--dagger-seed"', False), (TR, r'add_argument\("--init-ckpt"', False), (TR, r'"w_acc": "cfg:w_acc"', False), (TR, r'"dagger_p": "dagger:dagger_p"', False),
    (TR, r'"dagger_bank_sha256": "dagger:dagger_bank_sha256"', False), (TR, r'"init_ckpt": "data:init_ckpt"', False), (TR, r'^def acc_probe', False), (TR, r'^ +b = dag\.apply\(b, step', False),
    (TR, r'"dagger_n": dag_tot\["n"\]', False),
    (MD, r'T\["acc"\] = accel_term', False), (MD, r'total = total \+ cfg\.w_acc \* T\["acc"\] if cfg\.w_acc > 0\.0 else total', False),
    (DG, r'^class DaggerSwap', False),
    (Q, r'^:dagoaccarm', False), (Q, r'call :dagoaccarm %~1 %~2 0\.1', False),
]
LEVER_NEW = [
    (TR, r'add_argument\("--launch-intent-w"', True), (TR, r'"launch_intent_w": "cfg:launch_intent_w"', True), (TR, r'^def launch_intent_probe', True), (TR, r'^def launch_weight_share', True),
    (TR, r'^def launch_stage', True), (TR, r'sc\["launch_intent_probe"\] = lip', True), (TR, r'summary\["launch_weight_share"\] = launch_weight_share\(', True),
    (TR, r'summary\["launch_n"\], summary\["launch_eligible"\], summary\["launch_rows"\]', True), (TR, r'levers\["launch_intent"\] = \{"launch_intent_w": cfg\.launch_intent_w', True),
    (TR, r'lmask = launch_stage\(b\)', True), (TR, r'b\["launch"\] = lmask', True),
    (MD, r'^def launch_mask', True), (MD, r'^def launch_along_term', True), (MD, r'T\["launch_along"\] = launch_along_term\(', True),
    (MD, r'total = total \+ cfg\.w_traj \* cfg\.launch_intent_w \* T\["launch_along"\] / 3\.0 if cfg\.launch_intent_w > 0\.0 else total', True),
    (MD, r'^LAUNCH_V0_MAX, LAUNCH_DV, LAUNCH_DT = 6\.0, 1\.0, 0\.2', True), (MD, r'^LAUNCH_KNOT_LO, LAUNCH_KNOT_HI = 1, 12', True),
    (Q, r'^:launchintentarm', True), (Q, r'goto :launchintentarm', True), (Q, r'DAGS=launchintent"', True), (Q, r'DAGS=launchintentrep"', True), (Q, r'--launch-intent-w 1\.58926"', True),
    (Q, r'launchintentrep" set "DAGSEED=1%~2"', True),
]
LEVER_ALL = LEVER_BASE + LEVER_NEW


def lever_lines() -> str:
    out = []
    for f, pat, _new in LEVER_ALL:
        assert "'" not in pat or f != Q, pat
        if f == Q:
            out.append('  "$PK/code/v8f_mini_box2_queue.cmd|' + pat.replace('"', '\\"') + '"')
        else:
            assert "'" not in pat and "'" not in f
            out.append(f"  '{f}|{pat}'")
    return "\n".join(out)


_LANDED_OK = []


def landed_consistency() -> None:
    """The coordinator's landed pins are INHERITED until measured: re-hash the blobs at LANDED_COMMIT in the mirror (read-only `git show`) and abort on a mismatch. The staged files must be those blobs."""
    if _LANDED_OK:
        return
    assert C.git("rev-parse", f"{C.LANDED_COMMIT}^{{commit}}").strip() == C.LANDED_COMMIT, "the landed commit is not in the mirror"
    for path, want in ((TR, C.LANDED_TRAINER_SHA), (MD, C.LANDED_MODEL_SHA)):
        got = C.sha_b(C.git("show", f"{C.LANDED_COMMIT}:{path}", binary=True))
        assert got == want, f"{path} at the landed commit hashes to {got}, not the pinned {want}"
        assert C.sha(C.STAGE / path) == want, f"the staged {path} is not the landed blob"
    _LANDED_OK.append(1)


def pins(ship: Path) -> dict:
    landed_consistency()
    sub = {}
    inst = (C.INSTALLED / "chain_v8f_dagoacc_r2.sh").read_text(encoding="utf-8")

    def arr(var):
        m = re.search(rf"^{var}=\(([0-9a-f]{{64}}) ([0-9a-f]{{64}})\)", inst, re.M)
        assert m, f"{var} not in the INSTALLED DAG-OACC r2 chain"
        return m.group(1), m.group(2)
    r2_m2b, r2_bank = arr("M2B_SHA"), arr("BANK1_SHA")
    wacc_txt = (C.OACC / "wacc" / "w_acc.txt").read_bytes()
    wacc = wacc_txt.decode("ascii").strip()
    assert re.fullmatch(r"[0-9]+(\.[0-9]+)?", wacc) and wacc == C.WACC, wacc
    wins = set()
    for s in (0, 1):
        d = C.OACC / f"M2b_s{s}_dagoacc"
        sm = C.sums(d / "SHA256SUMS.txt")
        sub[f"OACC_CKPT_SHA{s}"], sub[f"OACC_ARGV_SHA{s}"], sub[f"OACC_SUMM_SHA{s}"] = sm["ckpt_last.pt"], sm["argv.json"], sm["summary.json"]
        assert C.sha(d / "argv.json") == sm["argv.json"] and C.sha(d / "summary.json") == sm["summary.json"], f"DAG-OACC_s{s}: banked files != SHA256SUMS.txt"
        tok, summ, argv = C.load(d / "launch_gate_token.json"), C.load(d / "summary.json"), C.load(d / "argv.json")["argv"]
        m2b = tok["init_ckpt_sha256"]
        assert m2b == summ["levers"]["init_ckpt"]["sha256"] == r2_m2b[s], f"M2b_s{s}: token / summary / installed chain disagree"
        bank = argv[argv.index("--dagger-bank-sha256") + 1]
        assert bank == r2_bank[s] == tok.get("bank_sha256", bank), f"bank s{s}: argv.json / installed chain disagree"
        assert argv[argv.index("--w-acc") + 1] == wacc, "DAG-OACC's argv.json --w-acc != w_acc.txt"
        assert "--launch-intent-w" not in argv
        sub[f"M2B_SHA{s}"], sub[f"BANK1_SHA{s}"] = m2b, bank
        commit = tok["binding"]["commit"]
        assert re.fullmatch(r"[0-9a-f]{40}", commit) and commit.startswith("aeb54468"), commit
        sub["BUILD_REF"] = commit
        wins.add(summ["train_window_list_sha256"])
    for s in (0, 1):
        d = C.REPDIR / f"M2b_s{s}_dagoaccorcrep"                     # REP_s: A_rep,s's pair control (DAG-OACC_s's argv at --seed 10 + s), banked from DAG-ORC r2
        sm = C.sums(d / "SHA256SUMS.txt")
        sub[f"REP_CKPT_SHA{s}"], sub[f"REP_ARGV_SHA{s}"], sub[f"REP_SUMM_SHA{s}"] = sm["ckpt_last.pt"], sm["argv.json"], sm["summary.json"]
        assert C.sha(d / "argv.json") == sm["argv.json"] and C.sha(d / "summary.json") == sm["summary.json"], f"REP_s{s}: banked files != SHA256SUMS.txt"
        rargv, oargv = C.load(d / "argv.json")["argv"], C.load(C.OACC / f"M2b_s{s}_dagoacc" / "argv.json")["argv"]
        diff = [i for i, (x, y) in enumerate(zip(rargv, oargv)) if x != y]
        assert len(rargv) == len(oargv) and [rargv[i - 1] for i in diff] == ["--seed", "--out"] and rargv[rargv.index("--seed") + 1] == str(10 + s), f"REP_s{s}'s argv is not DAG-OACC_s{s}'s apart from --seed {10 + s} and --out"
        assert "--launch-intent-w" not in rargv
    assert len(wins) == 1, wins                                      # both banked DAG-OACC runs share one train window list (PREREG section 3: e144adcd...8168)
    sub["WINDOW_SHA"] = wins.pop()
    assert C.load(C.LI / "f_measure.json")["window_list_sha256"] == sub["WINDOW_SHA"]
    sub["WACC"], sub["WACC_TXT_SHA"] = wacc, C.sha_b(wacc_txt)
    mir = C.git("rev-parse", "aeb54468^{commit}").strip()
    assert mir == sub["BUILD_REF"], "the mirror's aeb54468 != DAG-OACC's token commit"
    sub["ASSERT_SHA"] = C.sha(ship / "gate_launchintent_assert.py")
    sub["LCHECKS_SHA"] = C.sha(ship / "v8f_launchintent_chain_checks.py")
    sub["LILIB_SHA"] = C.sha(ship / "oaf_launchintent_lib.sh")
    sub["GUARD_SHA"] = C.sha(ship / "v8f_guard_launchintent.sh")
    sub["OAF_LIB_SHA"], sub["OAF_CHECKS_SHA"] = C.sha(C.INSTALLED / "oaf_lib.sh"), C.sha(C.INSTALLED / "v8f_oaf_chain_checks.py")
    m = re.search(r"^EXPORT=\(\n(.*?)\n\)\n", inst, re.M | re.S)
    assert m, "no EXPORT block in the INSTALLED DAG-OACC r2 chain"
    lines = [x for x in m.group(1).split("\n") if x.strip()]
    assert all(re.fullmatch(r"  '[^']+'", x) for x in lines), "EXPORT lines must be quoted paths"
    for need in (TR, "stack/scripts/v8f_mini_eval.py", DG, "stack/tanitad/data/v8f_mini_data.py", MD):
        assert any(f"'{need}'" in x for x in lines), need
    sub["EXPORT"] = "\n".join(lines)
    sub["LI_W"] = C.LI_W
    sub["LEVER"] = lever_lines()
    return sub


def default_pins() -> dict:
    return {"LAUNCH_COMMIT": C.LANDED_COMMIT, "LAUNCH_TRAINER_SHA": C.LANDED_TRAINER_SHA, "LAUNCH_MODEL_SHA": C.LANDED_MODEL_SHA, "PREREG_SHA": PH_PREREG, "READER_SHA": PH_READER}


def render(trainer_sha: str | None = None, ship: Path = C.SHIP, **pin_over) -> str:
    """``pin_over``: LAUNCH_COMMIT / LAUNCH_MODEL_SHA / PREREG_SHA / READER_SHA (the self-test renders worlds with its own scratch commit and with filled or placeholder pins)."""
    pv = {**default_pins(), **pin_over}
    if trainer_sha is not None:
        pv["LAUNCH_TRAINER_SHA"] = trainer_sha
    assert set(pv) == set(PIN_NAMES), set(pv) ^ set(PIN_NAMES)
    for k, v in pv.items():
        assert "'" not in v and "\n" not in v, f"the pin {k} is a single-quoted shell word"
    t = TEMPLATE.read_text(encoding="utf-8")
    assert "\r" not in t
    sub = pins(ship)
    sub.update(pv)
    for k, v in sub.items():
        assert t.count(f"@{k}@") >= 1, f"placeholder @{k}@ not in the template"
        t = t.replace(f"@{k}@", v)
    left = re.findall(r"@[A-Z0-9_]+@", t)
    assert not left, f"unfilled placeholders {left}"
    return t


def read_pins_from_disk(path: Path, strict: bool = True) -> dict:
    txt = path.read_text(encoding="utf-8")
    out = {}
    for k in PIN_NAMES:
        m = re.search(rf"^{k}='([^'\n]*)'$", txt, re.M)
        if not m and not strict:
            out[k] = default_pins()[k]
            continue
        assert m, f"no {k} line in {path}"
        out[k] = m.group(1)
    return out


def read_pin_from_disk(path: Path) -> str:
    return read_pins_from_disk(path)["LAUNCH_TRAINER_SHA"]


def check_patterns() -> int:
    """Every LEVER pattern must match the STAGED trainer / model / queue AND (trainer, model, dagger) the LANDED commit's blobs; every NEW one must NOT match the PRE-LEVER tip's blob (it discriminates)."""
    landed_consistency()
    tip, landed = C.tip_full(C.PRE_TIP), C.LANDED_COMMIT
    staged = {TR: (C.STAGE / TR).read_text(encoding="utf-8"), MD: (C.STAGE / MD).read_text(encoding="utf-8"), Q: (C.QSTAGE / "v8f_mini_box2_queue.cmd").read_text(encoding="utf-8").replace("\r", ""), DG: C.git("show", f"{tip}:{DG}")}
    bad = 0
    for f, pat, new in LEVER_ALL:
        if f != Q:
            in_landed = bool(re.search(pat, C.git("show", f"{landed}:{f}").replace("\r", ""), re.M))
            if not in_landed:
                bad += 1
                print(f"BAD landed commit lacks {pat} in {f}")
        tipt = C.git("show", f"{tip}:{f}").replace("\r", "")
        in_staged, in_tip = bool(re.search(pat, staged[f], re.M)), bool(re.search(pat, tipt, re.M))
        ok = in_staged and (not in_tip if new else in_tip)
        bad += (not ok)
        print(("ok  " if ok else "BAD ") + f"{'NEW' if new else 'base'} staged={in_staged} tip={in_tip}  {f.split('/')[-1]}: {pat}")
    print("patterns:", len(LEVER_ALL), "bad:", bad)
    return 1 if bad else 0


def main() -> int:
    a = sys.argv[1:]
    out = Path(a[a.index("--out") + 1]) if "--out" in a else C.SHIP / "chain_v8f_launchintent.sh"
    ship = Path(a[a.index("--ship-dir") + 1]) if "--ship-dir" in a else C.SHIP
    if "--check-patterns" in a:
        return check_patterns()
    if "--check" in a:
        text = render(ship=ship, **read_pins_from_disk(out))
        ok = out.exists() and out.read_bytes() == text.encode("utf-8")
        print("chain_v8f_launchintent.sh == a fresh regeneration:", ok)
        return 0 if ok else 1
    pv = read_pins_from_disk(out, strict=False) if out.exists() else default_pins()
    flags = {"--trainer-sha": ("LAUNCH_TRAINER_SHA", r"[0-9a-f]{64}"), "--model-sha": ("LAUNCH_MODEL_SHA", r"[0-9a-f]{64}"), "--commit": ("LAUNCH_COMMIT", r"[0-9a-f]{40}"),
             "--prereg-sha": ("PREREG_SHA", r"[0-9a-f]{64}"), "--reader-sha": ("READER_SHA", r"[0-9a-f]{64}")}
    for fl, (name, rx) in flags.items():
        if fl in a:
            v = a[a.index(fl) + 1]
            assert re.fullmatch(rx, v), f"{fl} must match {rx}"
            pv[name] = v
    text = render(ship=ship, **pv)
    C.write_lf(out, text)
    print(" | ".join(f"{k} = {(v if '<<' not in v else '(placeholder)')}" for k, v in pv.items()), "| sha256 of the chain", C.sha_b(text.encode()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
