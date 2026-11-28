"""Merge signed-link job files (hf_signed_links.py output, one per relay prefix) into ONE job for signed_pull.py.

    python merge_jobs.py <out.json> <job1.json> [<job2.json> ...]      # missing inputs are skipped
The files hold LIVE links: they live in %TEMP% only and the caller deletes them after `colab upload`. Prints names
and counts, never a link.
"""
import json
import os
import sys

out, ins = sys.argv[1], [p for p in sys.argv[2:] if os.path.exists(p)]
job = json.load(open(ins[0]))
for p in ins[1:]:
    for k, v in json.load(open(p))["files"].items():
        assert k not in job["files"], f"duplicate relay file name {k}"
        job["files"][k] = v
job["dest"] = "/content/relay"
job["prefix"] = "+".join(json.load(open(p)).get("prefix", "?") for p in ins)
json.dump(job, open(out, "w"))
print("ZZMERGED", len(job["files"]), sorted(job["files"]))
