#!/usr/bin/env python3
"""r8nt_prep.py -- refcv8 EMA-48,852 navtest read (A-R8-NS1 + Amendments 1-3) on Thor: gate G1 + the token files.  (tanitad-train python)

    python r8nt_prep.py <inputs.json.gz> <bank_dir> <logs_root> <out_dir>

Writes into <out_dir>:
  * G1_CHECK.json -- verdict PASS only if: the inputs hold exactly 12,146 navtest tokens (all stage 1); EVERY token is in the 416 bank
    index with 4 rows (the 4 history frames; the builder admitted a token only when all 4 x {CAM_L0, CAM_F0, CAM_R0} jpgs existed, KB3);
    every referenced shard file opens as uint8 [N, 416, 1024, 3] with every row index < N; every shard DONE file reads 0 partial /
    0 refused tokens; every token's OpenScene log pickle exists under <logs_root>.  Assert on THIS file, never on an exit code.
  * shard_k{k}of4.json (k = 0..3) -- W3-format {"tokens", "token_log"} subsets: sorted navtest tokens, index % 4 (the Thor backend's
    sharding, finalize_navtest_arm.py "how" field).  Used by the SCORER only (the roll is one process over all 12,146 tokens).
  * smoke20_tokens.json -- 20 tokens for the G3 smoke (sorted tokens, every 607th, i.e. a deterministic spread); VALIDATION ONLY.
"""
import glob
import gzip
import json
import os
import sys

import numpy as np

inp, bank, logs, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
doc = json.load(gzip.open(inp, "rt", encoding="utf-8"))
toks = doc["tokens"]
order = sorted(toks)
idx = json.load(open(os.path.join(bank, "index.json"), encoding="utf-8"))
fr = idx["frame"]
prov = idx["tokens"]
rec = {"inputs": os.path.abspath(inp), "bank": os.path.abspath(bank), "logs_root": os.path.abspath(logs), "checks": {}, "facts": {}}
c = rec["checks"]
f = rec["facts"]
f["n_tokens_inputs"] = len(order)
f["n_stage1"] = sum(1 for t in order if toks[t].get("stage") == 1)
c["inputs_hold_12146_navtest_tokens"] = (len(order) == 12146 == int(doc.get("n_tokens", -1)) and f["n_stage1"] == 12146)
c["bank_frame_is_416x1024_cylindrical"] = (fr["h"], fr["w"], fr["projection"]) == (416, 1024, "cylindrical")
missing = [t for t in order if t not in prov]
bad_rows = [t for t in order if t in prov and len(prov[t]["rows"]) != 4]
f["n_bank_index_tokens"] = len(prov)
f["n_tokens_missing_from_bank"] = len(missing)
f["missing_examples_index_positions"] = [order.index(t) for t in missing[:20]]
f["n_tokens_without_4_rows"] = len(bad_rows)
c["every_token_in_bank_with_4_history_rows"] = (not missing and not bad_rows)
files = sorted({prov[t]["file"] for t in order if t in prov})
shapes, bad_files = {}, []
max_row = {}
for t in order:
    if t in prov:
        p = prov[t]
        max_row[p["file"]] = max(max_row.get(p["file"], -1), max(p["rows"]))
for fn in files:
    pth = os.path.join(bank, fn)
    try:
        a = np.load(pth, mmap_mode="r")
        shapes[fn] = [str(a.dtype), list(a.shape)]
        if a.dtype != np.uint8 or tuple(a.shape[1:]) != (416, 1024, 3) or max_row[fn] >= a.shape[0]:
            bad_files.append(fn)
    except Exception as e:  # noqa: BLE001
        shapes[fn] = repr(e)[:200]
        bad_files.append(fn)
f["n_shard_files_referenced"] = len(files)
f["shard_shapes"] = shapes
c["every_shard_file_valid_and_rows_in_range"] = (len(files) > 0 and not bad_files)
dones = sorted(glob.glob(os.path.join(bank, "shard_*.DONE.json")))
part = ref = 0
for d in dones:
    j = json.load(open(d, encoding="utf-8"))
    part += int(j.get("n_tokens_partial", 0))
    ref += int(j.get("n_tokens_refused", 0))
f["n_done_files"], f["sum_partial_tokens"], f["sum_refused_tokens"] = len(dones), part, ref
c["bank_build_reports_zero_partial_zero_refused"] = (len(dones) == 32 and part == 0 and ref == 0)
logn = sorted({toks[t]["log_name"] for t in order})
nolog = [ln for ln in logn if not os.path.isfile(os.path.join(logs, ln + ".pkl"))]
f["n_logs_needed"], f["n_logs_missing"] = len(logn), len(nolog)
c["every_token_log_pickle_present"] = not nolog
rec["verdict"] = "PASS" if all(c.values()) else "FAIL"
json.dump(rec, open(os.path.join(out, "G1_CHECK.json"), "w", encoding="utf-8"), indent=1)
for k in range(4):
    sub = [t for i, t in enumerate(order) if i % 4 == k]
    json.dump({"tokens": sub, "token_log": {t: toks[t]["log_name"] for t in sub},
               "how": f"sorted navtest tokens, index % 4 == {k}"},
              open(os.path.join(out, f"shard_k{k}of4.json"), "w", encoding="utf-8"))
sm = [order[i * 607] for i in range(20)]
json.dump({"tokens": sm, "token_log": {t: toks[t]["log_name"] for t in sm},
           "label": "VALIDATION ONLY -- G3 smoke tokens (sorted navtest tokens, every 607th)",
           "maps": {t: toks[t]["map_name"] for t in sm}},
          open(os.path.join(out, "smoke20_tokens.json"), "w", encoding="utf-8"))
print("QQG1 " + json.dumps({"verdict": rec["verdict"], **c, "missing": len(missing), "logs_missing": len(nolog)}))
