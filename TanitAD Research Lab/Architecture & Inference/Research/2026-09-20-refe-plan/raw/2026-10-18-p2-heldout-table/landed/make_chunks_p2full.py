#!/usr/bin/env python3
"""P2 FULL TABLE -- STEP 1: cut the 12,146 navtest tokens into whole-log chunks BEFORE anything is dumped or scored.  Written ONCE (refuses to overwrite a
different plan).  Output: chunks/c{ii}_tokens.json (W3 format, consumed unchanged by refe_navtest_seam.py and score_navtest_refe.py) and chunks/chunks.json.
Nothing here reads a model output."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import c3_run_p2full as C  # noqa: E402


def main():
    assert C.sha256_file(C.TOKENS_FULL) == C.TOKENS_FULL_SHA256, "navtest_full_tokens.json is not the banked original"
    doc = json.load(open(C.TOKENS_FULL, encoding="utf-8"))
    assert len(doc["tokens"]) == C.N_FULL and len(set(doc["token_log"].values())) == C.N_LOGS_FULL
    plan = C.write_chunks(doc, os.path.join(HERE, "chunks"), C.MAX_CHUNK_TOKENS,
                          provenance={"source": C.TOKENS_FULL, "source_sha256": C.TOKENS_FULL_SHA256, "n_logs": C.N_LOGS_FULL})
    longest = 0
    for c in plan["chunks"]:
        cd = json.load(open(os.path.join(HERE, "chunks", c["tokens_file"]), encoding="utf-8"))
        longest = max(longest, C.cmdline_chars(cd))
    assert longest < 28000, f"a chunk's Hydra overrides are {longest} chars: too close to the Windows 32,767 command-line limit"
    union = [t for c in plan["chunks"] for t in json.load(open(os.path.join(HERE, "chunks", c["tokens_file"]), encoding="utf-8"))["tokens"]]
    assert len(union) == C.N_FULL and set(union) == set(doc["tokens"])
    print(json.dumps({"n_chunks": plan["n_chunks"], "sizes": [c["n"] for c in plan["chunks"]], "longest_override_chars": longest,
                      "n_logs_per_chunk": [c["n_logs"] for c in plan["chunks"]]}))


if __name__ == "__main__":
    main()
