"""make_chains.py's FLAT-DAG part: fill chain_v8f_flatdag.sh.in and write the guard / launcher / waker copies (own markers, own files, every name NEW)."""
import json
from pathlib import Path

from chain_dagoacc_gen import DAG2, guard_copy, launcher

NL = chr(10)


def write(out: Path, pins: dict, export_block, sha_file, rep, fn):
    here = Path(__file__).resolve().parent
    clo = json.loads((here / "export_closure_flatdag.json").read_text(encoding="utf-8"))
    t = (out / "chain_v8f_flatdag.sh.in").read_text(encoding="utf-8")
    arrs = [f"{k}=({' '.join(pins[k])})      # {c}" for k, c in (("CKPT_SHA", "FLAT_s0 / FLAT_s1 ckpt_last.pt (v8f-mini-flat/M2b_s<S>_flat.pt: the base, the bank generator and the --init-ckpt of all four arms)"),
                                                                   ("ARGV_SHA", "FLAT_s0 / FLAT_s1 argv.json (the repo blobs raw/2026-10-26-flat/runs/M2b_s<S>_flat/)"),
                                                                   ("SUMM_SHA", "FLAT_s0 / FLAT_s1 summary.json (the repo blobs)"))]
    t = rep(t, "@@PINS@@", NL.join(arrs))
    t = rep(t, "@@EXPORT@@" + NL, export_block("flatdag", clo))
    assert "@@" not in t
    (out / "chain_v8f_flatdag.sh").write_text(t, encoding="utf-8", newline=NL)
    g = guard_copy("# COPY of v8f_guard_dag2.sh for PREREG_V8F_FLATDAG's chain: byte-identical logic -- the anchored end test reads V8F_CHAIN_OUT; kill_ours already stops the bank generators (v8f_dagger_bank) and the trainer / eval", "")
    (out / "v8f_guard_flatdag.sh").write_text(g, encoding="utf-8", newline=NL)
    (out / "run_chain_v8f_flatdag.cmd").write_bytes(launcher("chain_v8f_flatdag", "chain_v8f_dagoacc", "rem PREREG_V8F_FLATDAG on box 2. NOT RUN: needs the PI's go. AFTER_OUT = the DAG-OACC chain's log (which itself waits on the DAGGER2 chain's); a predecessor STOP of its own with BOX2_BUSY released does not stop this chain, a ZZGUARD-STOP or a held BOX2_BUSY does."))
    w = (DAG2 / "wake_v8fdag2.sh").read_text(encoding="utf-8")
    w = rep(w, "chain_v8f_dag2.out", "chain_v8f_flatdag.out", 2)
    w = rep(w, "v8fdag2_guard.out", "v8fflatdag_guard.out", 2)
    w = rep(w, "^ZZDAG2-BLOCKED\\|^ZZDAG2-REFUSED", "^ZZFLATDAG-BLOCKED\\|^ZZFLATDAG-REFUSED\\|^ZZFLATDAG-F1-STOP\\|^ZZFLATDAG-CONTROL-FAIL\\|^ZZDISK-FLOOR-STOP")
    w = rep(w, "BANK JUDGE: $(grep -a '^ZZDAG2-' $O | tail -2)", "NAMED STOP: $(grep -a '^ZZFLATDAG-\\|^ZZDISK-' $O | tail -2)")
    w = rep(w, ".dag2_gates_seen", ".flatdag_gates_seen", 2)
    w = rep(w, "^DAG2 gate: PASS", "^OAF gate: PASS")
    w = rep(w, "grep -a '^DAG2 gate'", "grep -a '^OAF gate'")
    w = w.replace("PREREG_V8F_DAGGER2's box-2 chain", "PREREG_V8F_FLATDAG's box-2 chain")
    (out / "wake_v8fflatdag.sh").write_text(w, encoding="utf-8", newline=NL)
    print("wrote chain_v8f_flatdag.sh", len(t), "bytes + guard, launcher, waker")
