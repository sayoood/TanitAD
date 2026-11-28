"""SMOKE ONLY.  Stages ONE stand-in 'pulled run' so the chain's plumbing can be exercised on CPU.  NOTHING here is an O-RECOVER run, and no number from it is a result.

    python make_smoke_standin.py <ART dir>        -> <ART>/M2b_s0_orc/ with the eight files box 2's chain pulls + SHA256SUMS.txt (+ SMOKE_STANDIN.txt)

 ckpt_last.pt                 = a copy of the banked M2b_s0 HACC checkpoint (E:/.../v8f-mini-hacc/M2b_s0_hacc.pt; M2b seed 0 + the HACC post-map: the only M2b_s0 on this box)
 eval_M2b-orc_s0.{json,npz}   = the banked T0 eval of M2b_s0 (t1hdread/banked/eval_M2b_s0-sfresh-fresh.*) renamed
 selfcheck/summary/argv/train_log = the REAL trainer's output on SYNTHETIC data (orcchain_build/scratch/synth1/M2b_s0_orc): a 6-step fine-tune, not 3,000
 launch_gate_token.json       = a SMOKE placeholder that says so (it is not a launch-gate token)
"""
import hashlib
import json
import os
import shutil
import sys

art = sys.argv[1]
run = os.path.join(art, "M2b_s0_orc")
os.makedirs(run, exist_ok=True)
SYN = "E:/Projects/TanitAD-artifacts/orcchain_build/scratch/synth1/M2b_s0_orc"
BANK = "C:/Users/Admin/t1hdread/banked"
cp = [("E:/Projects/TanitAD-artifacts/v8f-mini-hacc/M2b_s0_hacc.pt", "ckpt_last.pt"),
      (f"{BANK}/eval_M2b_s0-sfresh-fresh.json", "eval_M2b-orc_s0.json"), (f"{BANK}/eval_M2b_s0-sfresh-fresh.npz", "eval_M2b-orc_s0.npz"),
      (f"{SYN}/selfcheck.json", "selfcheck.json"), (f"{SYN}/summary.json", "summary.json"), (f"{SYN}/argv.json", "argv.json"), (f"{SYN}/train_log.jsonl", "train_log.jsonl")]
for src, dst in cp:
    shutil.copyfile(src, os.path.join(run, dst))
with open(os.path.join(run, "launch_gate_token.json"), "w", encoding="utf-8") as f:
    json.dump({"kind": "SMOKE-STANDIN-NOT-A-LAUNCH-GATE-TOKEN", "verdict": "SMOKE", "binding": {}, "note": "placeholder written by orct1/smoke/make_smoke_standin.py; NOT an O-RECOVER run"}, f, indent=1)
with open(os.path.join(run, "SMOKE_STANDIN.txt"), "w", encoding="utf-8") as f:
    f.write(__doc__)
names = ["ckpt_last.pt", "eval_M2b-orc_s0.json", "eval_M2b-orc_s0.npz", "selfcheck.json", "summary.json", "argv.json", "train_log.jsonl", "launch_gate_token.json"]
lines = []
for n in names:
    h = hashlib.sha256()
    with open(os.path.join(run, n), "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    lines.append(f"{h.hexdigest()}  {n}")
with open(os.path.join(run, "SHA256SUMS.txt"), "w", encoding="utf-8", newline="\n") as f:      # the box-2 chain's order and format (sha256sum | tee)
    f.write("\n".join(lines) + "\n")
print("staged", run)
print("\n".join(lines))
