"""Build the PINNED Thor failure lists G-SUITE consumes (Master Mind 2026-09-27).

    python make_env_failure_pin.py <thor gate dir (res/results/CAND/*.xml + verdict.json)> <out.json>

ENV: the fixes-batch-2 Thor full-suite run (`thorgate_b2_0201`): `verdict.json`'s
`pre_existing_fail` (165 ids, failing on BOTH the tip and the candidate) -- the Master Mind
classified every one as environmental on Thor's stack/taniteval/tools-only gate tree. Each id is
mapped to the test FILE it lives in by reading the CAND junit file that holds it, so the dev-box
half of G-SUITE can run exactly those files on a full-commit archive.

FLAKY (a separate category): tests MEASURED to fail intermittently on BOTH trees. A flaky id may
fail without failing G-SUITE ONLY because its record is cited here (repo path + sha256).
"""
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

d, out = Path(sys.argv[1]), Path(sys.argv[2])
vpath = d / "verdict.json"
v = json.loads(vpath.read_text(encoding="utf-8"))
pinned = sorted(set(v["pre_existing_fail"]))
where: dict[str, dict] = {}
for x in sorted((d / "res" / "results" / "CAND").glob("*.xml")):
    sub, rest = x.name.split("__", 1)                 # "stack__tests__test_x.py.xml"
    rel = rest[: -len(".xml")].replace("__", "/")
    for tc in ET.parse(x).getroot().iter("testcase"):
        tid = f"{tc.get('classname')}::{tc.get('name')}"
        if tid in pinned:
            where[tid] = {"sub": sub, "file": rel}
missing = [t for t in pinned if t not in where]
assert not missing, f"{len(missing)} pinned ids not found in any CAND junit: {missing[:5]}"
FLAKY = [{
    "id": "tests.test_flagship_v4::test_graft_tau_to_zero_makes_the_class_posterior_one_hot",
    "sub": "stack", "file": "tests/test_flagship_v4.py",
    "record": "TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-declared-vs-built/"
              "raw/thor_gate_b3/verdict_addendum.json",
    "record_sha256": "91016fd69b9835605dff45c7acfcc708c6ae4cc2a27661c8fd7cc10fe4bacab7",
    "measured": "Thor gate env, 3+3 reruns (fixes batch 3): TIP failed 1/3 at 0.8314, CAND 1/3 at "
                "0.8716; the launch gate re-measured 1/6 as-is and 0/6 with torch.manual_seed(0) "
                "as the test's first line (2026-09-27, /home/nvidia/gate_lg_0353/flaky/)",
    "why": "unseeded head init and a stochastic decoder; the one-hot threshold 1 - 1e-6 sits on a "
           "knife-edge",
    "proposed_fix": "insert `torch.manual_seed(0)` as the first statement of the test (the pattern "
                    "test_flagship_v4.py already uses at its line 432)"}]
files = sorted({(w["sub"], w["file"]) for w in where.values()} | {(f["sub"], f["file"]) for f in FLAKY})
rec = {"schema": "tanitad.launch_gate.thor_env_failures/2",
       "what": "Thor full-suite failures G-SUITE admits. ENV: classified ENVIRONMENTAL (absent Research "
               "Lab files in a stack/taniteval/tools-only tree, HF offline without the weights, no "
               "git repo, no scipy in the training venv, aarch64 numerics / core count) -- Master "
               "Mind 2026-09-27. FLAKY: intermittent on BOTH trees, each with its cited record. "
               "G-SUITE (a) admits a Thor failure ONLY if its id is here; G-SUITE (b) runs every "
               "FILE below on the dev box, on a full-commit archive, where they must PASS (or skip "
               "for a registered reason; a FLAKY id may fail only with its record cited).",
       "source": {"verdict_json": "qland/work/thorgate_b2_0201/verdict.json (fixes batch 2 gate)",
                  "verdict_sha256": hashlib.sha256(vpath.read_bytes()).hexdigest(),
                  "tip": "06380de73d82673433db79bcdfd266e417a32063"},
       "n_ids": len(pinned), "ids": pinned,
       "n_flaky": len(FLAKY), "flaky": FLAKY,
       "n_files": len(files), "files": [{"sub": s, "file": f} for s, f in files],
       "id_to_file": dict(sorted(where.items()))}
out.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
print(json.dumps({"n_ids": rec["n_ids"], "n_flaky": rec["n_flaky"], "n_files": rec["n_files"],
                  "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}, indent=1))
