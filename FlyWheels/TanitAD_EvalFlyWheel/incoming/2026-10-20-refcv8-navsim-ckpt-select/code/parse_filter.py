"""Parse a NavSim scene_filter yaml (log_names + tokens) without a yaml dep; print counts + sha256 of sorted lists."""
import hashlib, json, sys


def parse(path):
    cur = None; out = {"log_names": [], "tokens": []}; hdr = {}
    raw = open(path, "rb").read()
    crlf = raw.count(b"\r\n")
    for line in raw.decode("utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if not line.startswith(" ") and ":" in s:
            k, v = s.split(":", 1)
            cur = k if k in out else None
            if cur is None:
                hdr[k] = v.strip()
            continue
        if cur and s.startswith("- "):
            out[cur].append(s[2:].strip().strip("'\""))
    return hdr, out, crlf, hashlib.sha256(raw).hexdigest()


def h(lst):
    return hashlib.sha256("\n".join(sorted(lst)).encode()).hexdigest()


def main():
    res = {}
    for p in sys.argv[1:]:
        hdr, out, crlf, fsha = parse(p)
        res[p] = {"file_sha256": fsha, "crlf_lines": crlf, "header": hdr,
                  "n_logs": len(out["log_names"]), "n_logs_unique": len(set(out["log_names"])),
                  "n_tokens": len(out["tokens"]), "n_tokens_unique": len(set(out["tokens"])),
                  "sha256_sorted_logs": h(set(out["log_names"])), "sha256_sorted_tokens": h(set(out["tokens"]))}
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
