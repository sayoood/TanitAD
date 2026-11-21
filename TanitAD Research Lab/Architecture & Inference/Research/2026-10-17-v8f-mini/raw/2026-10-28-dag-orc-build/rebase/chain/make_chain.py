#!/usr/bin/env python3
"""Render box2_ship/chain_v8f_dagoaccorc.sh from chain_v8f_dagoaccorc.sh.in. EVERY pinned value is COMPUTED from a banked artifact, never retyped:

  DAG-OACC_s<S> checkpoint / argv.json / summary.json    v8f-dagoacc/M2b_s<S>_dagoacc/SHA256SUMS.txt  (each small file is re-hashed here and must equal it)
  M2B_SHA (K-INIT)                                      v8f-dagoacc/M2b_s<S>_dagoacc/launch_gate_token.json init_ckpt_sha256 == summary.json levers.init_ckpt.sha256 == the DAG-OACC r2 chain's own M2B_SHA
  BANK1_SHA (K-PAIR)                                    DAG-OACC's argv.json --dagger-bank-sha256 == the DAG-OACC r2 chain's own BANK1_SHA
  WACC / WACC_TXT_SHA                                   v8f-dagoacc/wacc/w_acc.txt (== the --w-acc token of DAG-OACC's argv.json)
  BUILD_REF (K-BUILD)                                   DAG-OACC's launch-gate token binding.commit (== aeb54468 in the mirror)
  ASSERT_SHA / DCHECKS_SHA                              sha256 of the reviewed gate_dagoaccorc_assert.py / v8f_dagoaccorc_chain_checks.py beside this file
  the sourced lib and checks helper                     sha256 of the INSTALLED C:/Users/Admin/box2_ship/oaf_lib.sh and v8f_oaf_chain_checks.py
  EXPORT                                                the EXPORT=( ... ) block of the INSTALLED chain_v8f_dagoacc_r2.sh (the same trainer closure plus the DAgger files)

    python make_chain.py            writes box2_ship/chain_v8f_dagoaccorc.sh (LF)
    python make_chain.py --check    regenerates in memory and compares with the file on disk (rc 0 = equal)
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHIP = HERE / "box2_ship"
OACC = Path("E:/Projects/TanitAD-artifacts/v8f-dagoacc")
INSTALLED = Path("C:/Users/Admin/box2_ship")
MIRROR = "C:/Users/Admin/tanitad-push"


def sums(path: Path) -> dict:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        h, n = line.split(" ", 1)
        out[n.strip().lstrip("*")] = h
    return out


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render() -> str:
    t = (SHIP / "chain_v8f_dagoaccorc.sh.in").read_bytes().decode("utf-8")
    assert "\r" not in t, "the template must be LF"
    sub = {}
    r2 = (INSTALLED / "chain_v8f_dagoacc_r2.sh").read_text(encoding="utf-8")

    def r2arr(var):
        m = re.search(rf"^{var}=\(([0-9a-f]{{64}}) ([0-9a-f]{{64}})\)", r2, re.M)
        assert m, f"{var} not in the DAG-OACC r2 chain"
        return m.group(1), m.group(2)
    r2_m2b, r2_bank = r2arr("M2B_SHA"), r2arr("BANK1_SHA")
    banks, m2bs = [], []
    wacc_txt = (OACC / "wacc" / "w_acc.txt").read_bytes()
    wacc = wacc_txt.decode("ascii").strip()
    assert re.fullmatch(r"[0-9]+(\.[0-9]+)?", wacc), wacc
    for s in (0, 1):
        d = OACC / f"M2b_s{s}_dagoacc"
        sm = sums(d / "SHA256SUMS.txt")
        sub[f"OACC_CKPT_SHA{s}"], sub[f"OACC_ARGV_SHA{s}"], sub[f"OACC_SUMM_SHA{s}"] = sm["ckpt_last.pt"], sm["argv.json"], sm["summary.json"]
        assert sha(d / "argv.json") == sm["argv.json"] and sha(d / "summary.json") == sm["summary.json"], f"DAG-OACC_s{s}: banked files != SHA256SUMS.txt"
        tok = json.loads((d / "launch_gate_token.json").read_text(encoding="utf-8"))
        summ = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        argv = json.loads((d / "argv.json").read_text(encoding="utf-8"))["argv"]
        m2b = tok["init_ckpt_sha256"]
        assert m2b == summ["levers"]["init_ckpt"]["sha256"] == r2_m2b[s], f"M2b_s{s}: token / summary / r2 chain disagree"
        bank = argv[argv.index("--dagger-bank-sha256") + 1]
        assert bank == r2_bank[s] == tok.get("bank_sha256", bank), f"bank s{s}: argv.json / r2 chain disagree"
        assert argv[argv.index("--w-acc") + 1] == wacc, "DAG-OACC's argv.json --w-acc != w_acc.txt"
        sub[f"M2B_SHA{s}"], sub[f"BANK1_SHA{s}"] = m2b, bank
        commit = tok["binding"]["commit"]
        assert re.fullmatch(r"[0-9a-f]{40}", commit) and commit.startswith("aeb54468"), commit
        sub["BUILD_REF"] = commit
    sub["WACC"], sub["WACC_TXT_SHA"] = wacc, hashlib.sha256(wacc_txt).hexdigest()
    mir = subprocess.run(["git", "-C", MIRROR, "-c", "safe.directory=*", "rev-parse", "aeb54468^{commit}"], capture_output=True, text=True, check=True).stdout.strip()
    assert mir == sub["BUILD_REF"], "the mirror's aeb54468 != DAG-OACC's token commit"
    sub["ASSERT_SHA"], sub["DCHECKS_SHA"] = sha(SHIP / "gate_dagoaccorc_assert.py"), sha(SHIP / "v8f_dagoaccorc_chain_checks.py")
    sub["OAF_LIB_SHA"], sub["OAF_CHECKS_SHA"] = sha(INSTALLED / "oaf_lib.sh"), sha(INSTALLED / "v8f_oaf_chain_checks.py")
    m = re.search(r"^EXPORT=\(\n(.*?)\n\)\n", r2, re.M | re.S)
    assert m, "no EXPORT block in the DAG-OACC r2 chain"
    lines = [x for x in m.group(1).split("\n") if x.strip()]
    assert all(re.fullmatch(r"  '[^']+'", x) for x in lines), "EXPORT lines must be quoted paths"
    for need in ("stack/scripts/v8f_mini_train.py", "stack/scripts/v8f_mini_eval.py", "stack/tanitad/data/v8f_dagger.py", "stack/tanitad/data/v8f_mini_data.py", "stack/tanitad/models/v8f_mini.py"):
        assert any(f"'{need}'" in x for x in lines), need
    sub["EXPORT"] = "\n".join(lines)
    for k, v in sub.items():
        assert t.count(f"@{k}@") >= 1, f"placeholder @{k}@ not in the template"
        t = t.replace(f"@{k}@", v)
    left = re.findall(r"@[A-Z0-9_]+@", t)
    assert not left, f"unfilled placeholders {left}"
    return t


def main() -> int:
    text = render()
    out = SHIP / "chain_v8f_dagoaccorc.sh"
    if "--check" in sys.argv:
        ok = out.exists() and out.read_bytes() == text.encode("utf-8")
        print("chain_v8f_dagoaccorc.sh == a fresh regeneration:", ok)
        return 0 if ok else 1
    out.write_bytes(text.encode("utf-8"))
    print(f"wrote {out} sha256 {hashlib.sha256(text.encode()).hexdigest()} ({len(text)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
