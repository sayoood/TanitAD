#!/usr/bin/env python3
"""Write CODE_PIN_AXN.json (sha256 of the YAXN code + IMPL_P2.md). Run after the unit tests, before arming the chain; the chain's pins stage REFUSES on a mismatch."""
import hashlib, json
from pathlib import Path
H = Path(__file__).resolve().parent
files = ["axd_ft.py", "axn_common.py", "axn_seam.py", "axn_analyze.py", "axn_chain.py", "axn_run_all.py", "IMPL_P2.md", "AMENDMENT_1.md"]
pin = {f: hashlib.sha256((H / f).read_bytes()).hexdigest() for f in files}
(H / "CODE_PIN_AXN.json").write_text(json.dumps(pin, indent=1, sort_keys=True), encoding="utf-8")
print(json.dumps(pin, indent=1))
