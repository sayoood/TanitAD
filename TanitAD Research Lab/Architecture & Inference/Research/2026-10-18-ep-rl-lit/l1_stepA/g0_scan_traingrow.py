"""G0 source B: stream the assembled FINAL-RUN training bank (refe_train_grow.tar.zst) and record every (token, log_name) of every target row.
Read-only. CPU. One process. Python 3.14 (stdlib compression.zstd)."""
import json, re, sys, tarfile, time, hashlib
from compression import zstd
SRC = sys.argv[1]; OUT = sys.argv[2]
HO = set(open(sys.argv[3]).read().split()) if len(sys.argv) > 3 else set()
RX_T = re.compile(rb'"token": "([0-9a-f]{16})"'); RX_L = re.compile(rb'"log_name": "([^"]+)"')
members = []; tl = {}; per_member = {}
t0 = time.time()
with zstd.open(SRC, "rb") as f, tarfile.open(fileobj=f, mode="r|") as tf:
    for m in tf:
        if not m.isfile():
            continue
        members.append((m.name, m.size))
        if not m.name.endswith(".jsonl"):
            continue
        fh = tf.extractfile(m); n = 0; miss = 0; ho_rows = 0; ho_tok = set()
        for line in fh:
            t = RX_T.search(line); l = RX_L.search(line)
            if t and l:
                tk = t.group(1).decode(); tl.setdefault(tk, set()).add(l.group(1).decode()); n += 1
                if tk in HO: ho_rows += 1; ho_tok.add(tk)
            else:
                miss += 1
        per_member[m.name] = {"rows": n, "rows_without_token_or_log": miss, "bytes": m.size, "heldout_rows": ho_rows, "heldout_tokens": len(ho_tok)}
res = {"src": SRC, "seconds": round(time.time() - t0, 1), "n_members": len(members), "n_jsonl": len(per_member),
       "n_tokens": len(tl), "n_logs": len({x for v in tl.values() for x in v}),
       "per_member": per_member, "members_non_jsonl_head": [m for m in members if not m[0].endswith('.jsonl')][:40],
       "token_to_logs": {k: sorted(v) for k, v in tl.items()}}
json.dump(res, open(OUT, "w"))
print("DONE", res["n_members"], res["n_jsonl"], res["n_tokens"], res["n_logs"], res["seconds"])
