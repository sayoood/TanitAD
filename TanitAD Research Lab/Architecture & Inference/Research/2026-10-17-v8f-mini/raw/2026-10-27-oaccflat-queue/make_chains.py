#!/usr/bin/env python3
"""Generate the box-2 chain sources of PREREG_V8F_DAG_OACC and PREREG_V8F_FLATDAG into box2_ship/:

    oaf_lib.sh                  the functions both chains share: the DAG2 chain's boot / launch / clean / stop / waitrc / queue / rsha / ckset / ckget / finish EXTRACTED VERBATIM from chain_v8f_dag2.sh.in (read-only reference,
                                a few exact-count renames), plus the new ones: rsize, disk_floor, pull / pull_opt (the 1 GiB floor on E:, .part then rename), dry_stage (+ WACCDAG_DRY), wait_predecessor
    chain_v8f_dagoacc.sh        <- chain_v8f_dagoacc.sh.in   (written by hand; @@PINS@@ and @@EXPORT@@ filled here)
    chain_v8f_flatdag.sh        <- the DAG2 chain's template with exact-count edits (flatdag_edits.py)
    v8f_guard_dagoacc.sh / v8f_guard_flatdag.sh, run_chain_*.cmd (CRLF), wake_*.sh

Pinned values are COPIED from the banked artifacts, never retyped: DAG's checkpoint / argv.json / summary.json sha256 (v8f-dagger/M2b_s<S>_dag/SHA256SUMS.txt), M2b_s<S>'s checkpoint sha256 (DAG's summary.json levers.init_ckpt),
the iteration-1 bank.json sha256 (DAG's argv.json --dagger-bank-sha256, cross-checked against the pulled bank), the offline w_acc record and DAG_s0's banked fit plans (sha256 of the files), FLAT's checkpoints (v8f-mini-flat/*.pt).
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DAG2 = Path("E:/Projects/TanitAD-artifacts/dag2q/box2_ship")
DAGART = Path("E:/Projects/TanitAD-artifacts/v8f-dagger")
FLATART = Path("E:/Projects/TanitAD-artifacts/v8f-mini-flat")
OFFLINE = Path("E:/Projects/TanitAD-artifacts/dagoacc_dry/wacc_offline_DAG_s0.json")
FITPLANS = Path("E:/Projects/TanitAD-artifacts/daghacc/plans/fitplans_DAG_s0.npz")
MIRROR = "C:/Users/Admin/tanitad-push"
TIP = "1d4f4fbb"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
NL = chr(10)


def sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rep(t, old, new, n=1):
    assert t.count(old) == n, (old[:100], t.count(old))
    return t.replace(old, new)


def fn(text: str, name: str) -> str:
    """The text of the bash function ``name() {`` ... in ``text``: one line if it closes on the same line, else up to the first line that is exactly ``}``."""
    lines = text.split(NL)
    k = [i for i, x in enumerate(lines) if x.startswith(f"{name}() {{")]
    assert len(k) == 1, (name, k)
    if lines[k[0]].rstrip().endswith("}") and lines[k[0]].count("{") <= lines[k[0]].count("}"):
        return lines[k[0]]
    j = next(i for i in range(k[0] + 1, len(lines)) if lines[i] == "}")
    return NL.join(lines[k[0]:j + 1])


def git_blob(path: str) -> bytes:
    return subprocess.run(["git", "-C", MIRROR, "-c", "safe.directory=*", "show", f"{TIP}:{path}"], capture_output=True, check=True).stdout


# ------------------------------------------------------------------------------------------------------------------------------- the pins
def dag_pins() -> dict:
    ck, av, sm, m2b, b1 = [], [], [], [], []
    for s in (0, 1):
        d = {}
        for line in (DAGART / f"M2b_s{s}_dag" / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
            h, n = line.split(" *", 1)
            d[n.strip()] = h
        ck.append(d["ckpt_last.pt"]); av.append(d["argv.json"]); sm.append(d["summary.json"])
        for nm in ("ckpt_last.pt", "argv.json", "summary.json"):
            assert sha_file(DAGART / f"M2b_s{s}_dag" / nm) == d[nm], (s, nm, "the pulled DAG file is not what SHA256SUMS.txt says")
        summ = json.loads((DAGART / f"M2b_s{s}_dag" / "summary.json").read_text(encoding="utf-8"))
        m2b.append(summ["levers"]["init_ckpt"]["sha256"])
        argv = json.loads((DAGART / f"M2b_s{s}_dag" / "argv.json").read_text(encoding="utf-8"))["argv"]
        b = argv[argv.index("--dagger-bank-sha256") + 1]
        assert sha_file(DAGART / f"bank_s{s}" / "bank.json") == b, (s, "the pulled iteration-1 bank.json is not the bank DAG's argv names")
        b1.append(b)
    return {"CKPT_SHA": ck, "ARGV_SHA": av, "SUMM_SHA": sm, "M2B_SHA": m2b, "BANK1_SHA": b1}


def flat_pins() -> dict:
    ck, av, sm = [], [], []
    for s in (0, 1):
        ck.append(sha_file(FLATART / f"M2b_s{s}_flat.pt"))
        av.append(hashlib.sha256(git_blob(f"{PK}/raw/2026-10-26-flat/runs/M2b_s{s}_flat/argv.json")).hexdigest())
        sm.append(hashlib.sha256(git_blob(f"{PK}/raw/2026-10-26-flat/runs/M2b_s{s}_flat/summary.json")).hexdigest())
    assert ck[0].startswith("8496934a") and ck[1].startswith("6fdfc9b0"), ck
    return {"CKPT_SHA": ck, "ARGV_SHA": av, "SUMM_SHA": sm}


def arr(name: str, vals: list, comment: str) -> str:
    return f"{name}=({' '.join(vals)})      # {comment}"


# ------------------------------------------------------------------------------------------------------------------------------- oaf_lib.sh
def build_lib() -> str:
    t = (DAG2 / "chain_v8f_dag2.sh.in").read_text(encoding="utf-8")
    out = ['#!/usr/bin/env bash',
           '# oaf_lib.sh -- the functions chain_v8f_dagoacc.sh and chain_v8f_flatdag.sh share. SOURCED by them after they set: B L E R PK PY CHECKS CH CUR GUARD_OUT TAG BOOT0 T BUSY_TAKEN SSH SCP FLOOR_BYTES MIRROR.',
           '# GENERATED by make_chains.py: boot / launch / clean / stop / waitrc / queue / rsha / ckset / ckget / finish are the DAG2 chain\'s (chain_v8f_dag2.sh.in) byte for byte except the renames listed in make_chains.py;',
           '# rsize / disk_floor / pull / pull_opt / dry_stage / argvsha / wait_predecessor are new. Nothing in this file touches box 2 until a chain calls it.']
    out.append(fn(t, "argvsha"))
    out.append(fn(t, "stop"))
    out.append(fn(t, "boot"))
    out.append(fn(t, "launch"))
    out.append(fn(t, "clean"))
    out.append(rep(fn(t, "waitrc"), "$L/v8fdag2_guard.out", "$GUARD_OUT", 1))
    out.append(rep(fn(t, "queue"), 'local j="v8fdag2${TAG}_$1"', 'local j="v8f${CH}${TAG}_$1"'))
    out.append(fn(t, "rsha"))
    out.append(fn(t, "ckset"))
    out.append(fn(t, "ckget"))
    out.append(rep(fn(t, "finish"), "$E/telemetry_v8fdag2.csv", "$E/telemetry_v8f${CH}.csv"))
    out.append(r'''rsize() {   # rsize <box-2 path>: the file's length in bytes on box 2 (digits only)
  $SSH "powershell -NoProfile -Command \"(Get-Item -LiteralPath '$1').Length\"" 2>/dev/null | tr -d '\r' | grep -E '^[0-9]+$' | head -1
}
disk_floor() {   # disk_floor <local dir> <bytes to be written> <what>: REFUSES (a named STOP marker, nothing written) if the write would leave < $FLOOR_BYTES free on the drive of <local dir>
  local o rc
  o=$($PY "$(cygpath -m $CHECKS)" disk-check --dir "$(cygpath -m "$1")" --size "$2" --floor $FLOOR_BYTES ${OAF_TEST_FREE_BYTES:+--free-bytes $OAF_TEST_FREE_BYTES}); rc=$?      # OAF_TEST_FREE_BYTES: the fake-box harness only; unset on a real run
  [ $rc = 0 ] || { echo "ZZDISK-FLOOR-STOP pulling $3 ($2 bytes) into $1 would leave less than $FLOOR_BYTES bytes free: $o"; stop "disk floor: $o"; }
}
pull() {   # pull <box-2 path under R> <local dir>: box 2's size first, the FLOOR next (nothing is written if it fails), then scp to <name>.part and a rename only when the size on disk is box 2's
  local rel=$1 d=$2 n=${1##*/} sz
  mkdir -p "$d"
  sz=$(rsize "$R/$rel"); [[ $sz =~ ^[0-9]+$ ]] || stop "pull $R/$rel: cannot read its size on box 2"
  disk_floor "$d" "$sz" "$rel"
  rm -f "$d/$n.part"
  $SCP "tanitad-box2:$R/$rel" "$d/$n.part" || { rm -f "$d/$n.part"; stop "pull $R/$rel"; }
  [ "$(stat -c %s "$d/$n.part")" = "$sz" ] || { rm -f "$d/$n.part"; stop "pull $R/$rel: $(stat -c %s "$d/$n.part" 2>/dev/null) bytes on disk != box 2's $sz"; }
  mv -f "$d/$n.part" "$d/$n"
}
pull_opt() {   # pull_opt <box-2 path under R> <local dir>: an OPTIONAL file (absent on box 2 = fine, rc 1); the floor still applies when it is there
  local sz; sz=$(rsize "$R/$1"); [[ $sz =~ ^[0-9]+$ ]] || return 1
  pull "$1" "$2"
}
argvsha_file() { $PY "$(cygpath -m $CHECKS)" argv-sha "$(cygpath -m "$1")"; }
dry_stage() {   # dry_stage <queue file> <bank sha or ""> <w_acc or ""> <stage args...>: the line(s) the queue prints for a stage under V8F_DRYRUN, with BOX 2's paths pinned (the dev box's environment must not decide what box 2 runs)
  local qf=$1 sha=$2 wacc=$3; shift 3
  local w=${TMPDIR:-/tmp}/v8f${CH}_dry_$$.cmd
  { printf '@echo off\r\nset V8F_DRYRUN=1\r\nset "REPO=%s"\r\nset "PY=%s"\r\nset "CACHEFULL=%s"\r\nset "V9=%s"\r\nset "RUNSM=%s"\r\n' 'D:\tanitad\repo' 'D:\tanitad\venv\Scripts\python.exe' 'D:\tanitad\cache_full' 'D:\tanitad\data\v9' 'D:\tanitad\runs\v8f_mini_full'
    [ -z "$sha" ] || printf 'set "DAGSHA_DRY=%s"\r\n' "$sha"
    [ -z "$wacc" ] || printf 'set "WACCDAG_DRY=%s"\r\n' "$wacc"
    printf 'call "%s" %s\r\n' "$(cygpath -w "$qf")" "$*"; } > "$w"
  MSYS_NO_PATHCONV=1 cmd /c "$(cygpath -w "$w")" </dev/null | tr -d '\r'
  rm -f "$w"
}
wait_predecessor() {   # wait_predecessor: needs AFTER_OUT. Polls the predecessor's log (v8f_oaf_chain_checks.py pred-verdict); returns 0 = GO. A predecessor STOP must not idle box 2: GO after DONE or after a STOP of its own once BOX2_BUSY is released;
                       # HALT (STOP, nothing taken) on a guard stop or a BOX2_BUSY that stays held (3 re-checks, OAF_GRACE_S apart). OAF_POLL_S = the poll (300 s)
  local v w k=0 poll=${OAF_POLL_S:-300} grace=${OAF_GRACE_S:-20}
  echo "$(date '+%m-%d %H:%M') waiting for the predecessor chain ($AFTER_OUT: end markers ^ZZV8FMINI-DONE / ^ZZV8FMINI-STOP; log present: $([ -f "$AFTER_OUT" ] && echo yes || echo NO))"
  while :; do
    v=$($PY "$(cygpath -m $CHECKS)" pred-verdict --out "$(cygpath -m "$AFTER_OUT")" --busy "$(cygpath -m "$B/BOX2_BUSY.txt")"); w=${v%% *}
    case $w in
      GO) echo "$(date '+%m-%d %H:%M') predecessor verdict: $v"; return 0 ;;
      HALT) stop "predecessor verdict: $v" ;;
      BUSY) k=$((k+1)); [ $k -gt 3 ] && stop "predecessor verdict: $v (still held after $k checks)"; sleep $grace ;;
      *) k=0; sleep $poll ;;
    esac
  done
}''')
    return NL.join(out) + NL


# ------------------------------------------------------------------------------------------------------------------------------- the export block
def export_block(name: str, clo: dict) -> str:
    files, data = clo["files"], clo["data"]
    items = sorted(set(files) | set(data))
    q = lambda s: "'" + s.replace("'", "'\\''") + "'"
    body = NL.join("  " + q(x) for x in items)
    return (f"# EXPORT: what box 2's queue READS from the repo (export_closure.py: the import closure of the entry scripts = {len(files)} python files, plus the {len(data)} repo data paths those programs open by default).{NL}"
            f"# Generated by make_chains.py from export_closure_{name}.json, never typed. The tar has a NEW name; no existing tar is overwritten or deleted.{NL}"
            f"EXPORT=({NL}{body}{NL}){NL}EXPORT_MAX_BYTES=314572800{NL}")


def main() -> int:
    which = sys.argv[1:] or ["lib", "dagoacc", "flatdag"]
    out = HERE / "box2_ship"
    out.mkdir(exist_ok=True)
    if "lib" in which:
        (out / "oaf_lib.sh").write_text(build_lib(), encoding="utf-8", newline=NL)
        print("wrote oaf_lib.sh")
    if "dagoacc" in which:
        import chain_dagoacc_gen as G
        G.write(out, dag_pins(), export_block, sha_file, OFFLINE, FITPLANS)
    if "flatdag" in which:
        import chain_flatdag_gen as G
        G.write(out, flat_pins(), export_block, sha_file, rep, fn)
    return 0


if __name__ == "__main__":
    sys.exit(main())
