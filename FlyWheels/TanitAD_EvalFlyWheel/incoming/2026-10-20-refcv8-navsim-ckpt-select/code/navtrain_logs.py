"""Emit the navtrain filter's sorted log list + token list (v1.1 tree copy) as json; report overlap with given log-name files."""
import json, sys, hashlib
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_filter import parse  # noqa
yaml_path, out = sys.argv[1], sys.argv[2]
hdr, d, crlf, fsha = parse(yaml_path)
logs = sorted(set(d["log_names"])); toks = sorted(set(d["tokens"]))
json.dump({"source": yaml_path, "file_sha256": fsha, "header": hdr, "n_logs": len(logs), "n_tokens": len(toks),
           "sha256_sorted_logs": hashlib.sha256("\n".join(logs).encode()).hexdigest(),
           "sha256_sorted_tokens": hashlib.sha256("\n".join(toks).encode()).hexdigest(),
           "log_names_sorted": logs, "tokens_sorted": toks}, open(out, "w"), indent=0)
print("first 25 sorted navtrain logs:"); [print(i, l) for i, l in enumerate(logs[:25])]
