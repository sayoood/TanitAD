#!/usr/bin/env python3
"""navtest_city_census.py -- per-city token count of the navtest tokens refcv8 would be scored on (CPU, read-only).

Reads the W3 navtest export (`navtest_inputs.json.gz`: token -> log_name) and each navtest log pickle's
`map_location` (OpenScene `navsim_logs/test/<log>.pkl`, one frame dict list per log; PosixPath shim as in E2's
`build_frames.load`).  Writes `navtest_city_census.json`.  Why: Singapore is LEFT-hand traffic; PhysicalAI-AV's
handedness mix is a transfer risk, so the share of navtest it governs must be a number, not a guess.
"""
import gzip, json, pathlib, pickle, sys, collections

LOGS = pathlib.Path("D:/Archive/devbox-C/navsim/data/openscene/navsim_logs/test")
INPUTS = pathlib.Path("D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz")


class _U(pickle.Unpickler):
    def find_class(self, mod, name):
        if mod == "pathlib" and name == "PosixPath":
            return pathlib.PurePosixPath
        return super().find_class(mod, name)


def main(out):
    doc = json.load(gzip.open(INPUTS, "rt", encoding="utf-8"))
    toks = doc["tokens"] if "tokens" in doc else doc
    per_log = collections.Counter(r["log_name"] for r in toks.values())
    city, unread = {}, []
    for ln in sorted(per_log):
        p = LOGS / f"{ln}.pkl"
        try:
            with open(p, "rb") as f:
                fr = _U(f).load()
            city[ln] = str(fr[0].get("map_location"))
        except Exception as e:                       # counted, never silently dropped
            unread.append([ln, repr(e)[:160]])
    tok_city = collections.Counter()
    log_city = collections.Counter()
    for ln, n in per_log.items():
        c = city.get(ln, "UNREAD")
        tok_city[c] += n
        log_city[c] += 1
    res = {"n_tokens": sum(per_log.values()), "n_logs": len(per_log), "tokens_by_city": dict(tok_city),
           "logs_by_city": dict(log_city), "unread_logs": unread}
    pathlib.Path(out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res))
    return 0 if not unread and res["n_tokens"] == 12146 else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
