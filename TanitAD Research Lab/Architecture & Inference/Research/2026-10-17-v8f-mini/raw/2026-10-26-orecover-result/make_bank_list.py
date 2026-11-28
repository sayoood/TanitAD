"""Writes bank_list.txt: every small file worth banking with its sha256, size and a suggested repo path. Nothing > 20 MB, no checkpoint."""
import hashlib, os
H = "C:/Users/Admin/results_orc"
A = "E:/Projects/TanitAD-artifacts"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RAW = PK + "/raw/2026-10-26-orecover-result"
items = []
def add(src, dst):
    items.append((src, dst))
add(H + "/RESULT_V8F_ORECOVER.md", PK + "/RESULT_V8F_ORECOVER.md")
for f in ("ORC_FAMILIES.json", "PAIRFAM_T0_s0.json", "PAIRFAM_T0_s1.json", "numbers_manifest.json", "verify_numbers.out", "orc_families.py", "build_result.py",
          "verify_numbers.py", "consolidate_dumps.py", "make_bank_list.py", "run_all.sh", "register_cell.md"):
    add(f"{H}/{f}", f"{RAW}/{f}")
for f in sorted(os.listdir(H + "/bank")):
    add(f"{H}/bank/{f}", f"{RAW}/bank/{f}")
add(H + "/_prereg.md", RAW + "/inputs/PREREG_V8F_ORECOVER.07dcb513.md")
T1 = A + "/v8f-orecover-t1"
for f in ("read/ORC_READ.json", "runs.json", "PINS.txt"):
    add(f"{T1}/{f}", f"{RAW}/inputs/{os.path.basename(f)}")
for a in ("ORC", "ORC0"):
    for s in (0, 1):
        for lim in ("causal", "logged"):
            add(f"{T1}/t1v8f_{a}_s{s}_m10_{lim}_heads.json", f"{RAW}/inputs/t1v8f_{a}_s{s}_m10_{lim}_heads.json")
        add(f"{T1}/t0fresh/eval_M2b-{a.lower()}_s{s}-fresh.json", f"{RAW}/inputs/eval_M2b-{a.lower()}_s{s}-fresh.json")
        d = f"{A}/v8f-orecover/M2b_s{s}_{a.lower()}"
        for f in (f"eval_M2b-{a.lower()}_s{s}.json", f"eval_M2b-{a.lower()}_s{s}.npz", "argv.json", "summary.json", "selfcheck.json", "launch_gate_token.json", "SHA256SUMS.txt", "train_log.jsonl"):
            add(f"{d}/{f}", f"{RAW}/inputs/M2b_s{s}_{a.lower()}/{f}")
for src, nm in (("C:/Users/Admin/orct1/logs/chain.out", "t1_chain.out"), ("C:/Users/Admin/orct1/logs/orc_read.out", "orc_read.out"),
                ("C:/Users/Admin/box2_ship/logs/chain_v8f_orc.out", "box2_chain_v8f_orc.out")):
    add(src, f"{RAW}/inputs/{nm}")
lines = ["# bank_list.txt: small files to bank for RESULT_V8F_ORECOVER (sha256, bytes, source => suggested repo path under the v8f-mini package).",
         "# No checkpoint and nothing above 20 MB. NOT banked on purpose: M2b_s{0,1}_{orc,orc0}/ckpt_last.pt (87 MB each; their sha256 are in each SHA256SUMS.txt),",
         "# ticks_*.npz (18.4 MB each; sha256 in runs.json), dump_*_heads/ (141 files, 141 MB per run; the causal arrays are in bank/t1_causal_dump_*.npz), *-fresh.npz.",
         "# register_cell.md is text to paste into the H-V8F-ORECOVER-1 status cell, not a file to land.", ""]
tot = 0
for src, dst in items:
    if not os.path.isfile(src):
        raise SystemExit("missing " + src)
    b = os.path.getsize(src)
    assert b <= 20 * 1024 * 1024, src
    h = hashlib.sha256(open(src, "rb").read()).hexdigest()
    tot += b
    lines.append(f"{h}  {b:>9}  {src}  =>  {dst}")
lines.append("")
lines.append(f"# {len(items)} files, {tot} bytes")
open(H + "/bank_list.txt", "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
print(len(items), tot)
