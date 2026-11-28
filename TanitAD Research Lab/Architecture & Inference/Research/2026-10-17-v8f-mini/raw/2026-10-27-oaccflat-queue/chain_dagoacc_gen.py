"""make_chains.py's DAG-OACC part: fill chain_v8f_dagoacc.sh.in and write the guard / launcher / waker copies (own markers, own files, every name NEW)."""
import json
from pathlib import Path

NL = chr(10)
DAG2 = Path("E:/Projects/TanitAD-artifacts/dag2q/box2_ship")


def rep(t, old, new, n=1):
    assert t.count(old) == n, (old[:100], t.count(old))
    return t.replace(old, new)


def guard_copy(comment: str, extra_pattern: str) -> str:
    g = (DAG2 / "v8f_guard_dag2.sh").read_text(encoding="utf-8")
    lines = g.split(NL)
    assert lines[1].startswith("# COPY of v8f_guard_dag.sh")
    lines[1] = comment
    g = NL.join(lines)
    if extra_pattern:
        g = rep(g, "v8f_dagger_bank|v8f_dagger_queue_tool|", "v8f_dagger_bank|v8f_dagger_queue_tool|" + extra_pattern + "|")
    return g


def launcher(chain: str, after: str, note: str) -> bytes:
    bs = chr(92)
    lines = ("@echo off", note,
             '"C:' + bs + 'Program Files' + bs + 'Git' + bs + 'bin' + bs + 'bash.exe" -lc "AFTER_OUT=/c/Users/Admin/box2_ship/logs/' + after + '.out bash /c/Users/Admin/box2_ship/' + chain + '.sh < /dev/null > /c/Users/Admin/box2_ship/logs/' + chain + '.out 2>&1" < NUL')
    return ("\r\n".join(lines) + "\r\n").encode("utf-8")


def write(out: Path, pins: dict, export_block, sha_file, OFFLINE, FITPLANS):
    here = Path(__file__).resolve().parent
    clo = json.loads((here / "export_closure_dagoacc.json").read_text(encoding="utf-8"))
    t = (out / "chain_v8f_dagoacc.sh.in").read_text(encoding="utf-8")
    arrs = [f"{k}=({' '.join(pins[k])})      # {c}" for k, c in (("CKPT_SHA", "DAG_s0 / DAG_s1 ckpt_last.pt (v8f-dagger/M2b_s<S>_dag/SHA256SUMS.txt)"), ("ARGV_SHA", "DAG_s0 / DAG_s1 argv.json"), ("SUMM_SHA", "DAG_s0 / DAG_s1 summary.json"),
                                        ("M2B_SHA", "M2b_s0 / M2b_s1 ckpt_last.pt: DAG's summary.json levers.init_ckpt.sha256 (the init of DAG and DAG-OACC)"), ("BANK1_SHA", "the ITERATION-1 bank.json sha256 of seed 0 / 1: DAG's argv.json --dagger-bank-sha256"))]
    arrs.append(f"OFFLINE_SHA={sha_file(OFFLINE)}      # wacc_offline_DAG_s0.json (w_acc 49.6969: C-O3's reference)")
    arrs.append(f"FITPLANS_SHA={sha_file(FITPLANS)}      # fitplans_DAG_s0.npz: DAG_s0's banked fit windows (the stage's cross-check)")
    t = rep(t, "@@PINS@@", NL.join(arrs))
    t = rep(t, "@@EXPORT@@" + NL, export_block("dagoacc", clo))
    assert "@@" not in t
    (out / "chain_v8f_dagoacc.sh").write_text(t, encoding="utf-8", newline=NL)
    g = guard_copy("# COPY of v8f_guard_dag2.sh for PREREG_V8F_DAG_OACC's chain: byte-identical logic -- the anchored end test reads V8F_CHAIN_OUT; the ONLY change is that kill_ours also stops the w_acc stage (v8f_mini_eval_oacc)", "v8f_mini_eval_oacc")
    (out / "v8f_guard_dagoacc.sh").write_text(g, encoding="utf-8", newline=NL)
    (out / "run_chain_v8f_dagoacc.cmd").write_bytes(launcher("chain_v8f_dagoacc", "chain_v8f_dag2", "rem PREREG_V8F_DAG_OACC on box 2. NOT RUN: needs the PI's go. AFTER_OUT = the DAGGER2 chain's log; a predecessor STOP of its own with BOX2_BUSY released does not stop this chain, a ZZGUARD-STOP or a held BOX2_BUSY does."))
    w = (DAG2 / "wake_v8fdag2.sh").read_text(encoding="utf-8")
    w = rep(w, "chain_v8f_dag2.out", "chain_v8f_dagoacc.out", 2)
    w = rep(w, "v8fdag2_guard.out", "v8fdagoacc_guard.out", 2)
    w = rep(w, "^ZZDAG2-BLOCKED\\|^ZZDAG2-REFUSED", "^ZZDAGOACC-CO3-STOP\\|^ZZDISK-FLOOR-STOP\\|^ZZDAGOACC-CONTROL-FAIL")
    w = rep(w, "BANK JUDGE: $(grep -a '^ZZDAG2-' $O | tail -2)", "NAMED STOP: $(grep -a '^ZZDAGOACC-\\|^ZZDISK-' $O | tail -2)")
    w = rep(w, "-ge 4 ] && [ ! -f $L/.dag2_gates_seen ] && { touch $L/.dag2_gates_seen", "-ge 2 ] && [ ! -f $L/.dagoacc_gates_seen ] && { touch $L/.dagoacc_gates_seen")
    w = rep(w, "^DAG2 gate: PASS", "^OAF gate: PASS")
    w = rep(w, "grep -a '^DAG2 gate'", "grep -a '^OAF gate'")
    w = w.replace("PREREG_V8F_DAGGER2's box-2 chain", "PREREG_V8F_DAG_OACC's box-2 chain")
    (out / "wake_v8fdagoacc.sh").write_text(w, encoding="utf-8", newline=NL)
    print("wrote chain_v8f_dagoacc.sh", len(t), "bytes + guard, launcher, waker")
