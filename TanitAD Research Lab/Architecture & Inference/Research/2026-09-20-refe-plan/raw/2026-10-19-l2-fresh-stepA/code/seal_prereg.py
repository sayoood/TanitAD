#!/usr/bin/env python3
"""Seal PREREG_L2_FRESH.md: fills PREREG_L2_FRESH.src.md (the edited draft) with the pins as the files on disk say they are and with the MM ruling VERBATIM, writes PREREG_L2_FRESH.md (LF bytes) +
PREREG_L2_FRESH.sha256 + SEALED_AT.txt.  Refuses to overwrite an existing sealed file and refuses an unresolved placeholder or a draft marker.  Reads NO label and NO score.

    python seal_prereg.py --src PREREG_L2_FRESH.src.md --ruling MM_RULING_L2_FRESH.md --sample raw/fresh_sample_token_disjoint.json --out PREREG_L2_FRESH.md
"""
import argparse
import datetime
import json
import os
import re
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyse_fresh as AF  # noqa: E402

P2_CKPT_SHA256 = "ea31db2296210b6a35dbf32ad95f1813edfa259ab330002b3ca7564aac727e51"       # feats_manifest.json inputs.ckpt_model_final.pt (L2)
B_DIR = "D:/Projects/TanitAD-artifacts/l2_fresh_stepA/B_token_disjoint"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--ruling", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stepA-sample", default="D:/Projects/TanitAD-artifacts/l1_stepA_run/sample.json")
    ap.add_argument("--note", default="", help="a line recorded in SEALED_AT.txt (e.g. why this is a re-seal)")
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        raise SystemExit(f"refusing to overwrite the sealed file {a.out}")
    sample = json.load(open(a.sample))
    pins = AF.disk_pins(sample, AF.HEADS_DIR, AF.LOGOOF_DIR)
    code = AF.current_code_pins()
    import hashlib
    stepA_sha = hashlib.sha256("\n".join(sorted(str(t) for t in json.load(open(a.stepA_sample))["tokens"])).encode()).hexdigest()
    if stepA_sha != sample["stepA_sample_tokens_sha256"]:
        raise SystemExit("Step A's token-list sha256 (recomputed from its own sample.json) differs from the one the carve record names")
    ruling = open(a.ruling, "rb").read().replace(b"\r\n", b"\n").decode("utf-8").rstrip("\n")
    fill = {"MM_RULING": ruling, "CARVE_SHA": pins["carve token-list sha256"], "STEPA_SHA": stepA_sha, "S0": pins["final head s0 sha256"], "S1": pins["final head s1 sha256"],
            "P2_FILE": pins["P2 head file sha256 (L2 reference)"], "P2_CONTENT": pins["P2 head content sha256 (arrays)"], "P2_CKPT": P2_CKPT_SHA256,
            "LOGOOF_NPZ": pins["log-OOF fold-heads file sha256"], "LOGOOF_JSON": pins["log-OOF fold-heads manifest sha256"],
            "CODE_PIN_LINES": "\n".join(f"  * `{n}` {h}" for n, h in code.items())}
    txt = open(a.src, encoding="utf-8").read()
    for k, v in fill.items():
        txt = txt.replace("{{" + k + "}}", v)
    left = re.findall(r"\{\{[A-Z0-9_]+\}\}", txt)
    if left:
        raise SystemExit(f"unresolved placeholders: {left}")
    if AF.DRAFT_MARKER in txt or AF.SEALED_MARKER not in txt:
        raise SystemExit("the sealed text must carry the SEALED marker and no draft marker")
    if ruling not in txt:
        raise SystemExit("the ruling is not carried verbatim")
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(txt)
    sha = AF.sha_lf(a.out)
    open(os.path.splitext(a.out)[0] + ".sha256", "w", newline="\n").write(f"{sha}  {os.path.basename(a.out)}\n")
    chain = open(os.path.join(os.path.dirname(os.path.abspath(a.out)), "chain_B.log"), encoding="utf-8", errors="replace").read().splitlines()
    b_files = sorted(os.listdir(B_DIR)) if os.path.isdir(B_DIR) else []
    now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [f"SEALED_AT {now}", f"file {os.path.basename(a.out)}", f"sha256 (LF bytes) {sha}"] + ([f"note: {a.note}"] if a.note else []) + [""] + [
             "state of the pre-launched B chain at the seal (no fresh label or score existed):", f"  chain_B.log last line: {chain[-1] if chain else '(empty)'}", f"  B_token_disjoint/ holds: {b_files}", "",
             "pins named in the sealed text:"] + [f"  {k}: {v}" for k, v in pins.items()] + ["  P2 checkpoint sha256: " + P2_CKPT_SHA256, "", "code pins (LF sha256):"] + [f"  {n}: {h}" for n, h in code.items()]
    open(os.path.join(os.path.dirname(os.path.abspath(a.out)), "SEALED_AT.txt"), "w", newline="\n").write("\n".join(lines) + "\n")
    print("ZZSEALED", sha, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
