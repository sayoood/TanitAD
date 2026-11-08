"""AMENDMENT 2: selc_dump.consolidate on synthetic chunks. Run with the driverl venv python, cwd = this package."""
import json
import os
import sys
import tempfile
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import selc_dump as SD  # noqa: E402

LOGS = {"log_a": ["t5", "t1"], "log_b": ["t4", "t0"], "log_c": ["t3", "t2"]}


def build(root, logs):
    out = os.path.join(root, "dump")
    os.makedirs(os.path.join(out, "chunks"))
    for lg, toks in logs.items():
        n = len(toks)
        v = {"token": np.array(toks), "log": np.array([lg] * n), "fingerprint": np.array([f"fp_{t}" for t in toks])}
        for f in SD.FIELDS:
            shape = {"pick": (n,), "gt_valid": (n,), "frame_ctrl_m": (n,), "props8": (n, 64, 8, 3), "logits": (n, 64, 7)}.get(f, (n, 2))
            arr = np.zeros(shape, np.float32)
            for i, t in enumerate(toks):
                arr[i] = float(t[1:])                # every field of a row carries its token's number
            v[f] = arr
        np.savez(SD.chunk_path(out, lg), **v)
    return out


def doc_file(root, toks, logs):
    p = os.path.join(root, "tokens.json")
    tl = {t: lg for lg, ts in logs.items() for t in ts}
    json.dump({"tokens": toks, "token_log": {t: tl.get(t, "log_a") for t in toks}}, open(p, "w", encoding="utf-8"))
    return p


def run(root, tokens_path, out):
    a = types.SimpleNamespace(tokens=tokens_path, out=out)
    return SD.consolidate(a, lambda m: None)


def main():
    with tempfile.TemporaryDirectory(dir=os.environ.get("SCORER1_TMP") or None) as r:
        toks = sorted(t for ts in LOGS.values() for t in ts)
        out = build(r, LOGS)
        run(r, doc_file(r, toks, LOGS), out)
        with np.load(os.path.join(out, "dump_confirm.npz"), allow_pickle=False) as z:
            assert [str(t) for t in z["token"]] == toks, "not in token order"
            for f in SD.FIELDS:
                for i, t in enumerate(toks):
                    assert float(np.ravel(z[f][i])[0]) == float(t[1:]), f"{f} row {i} is not its token's"
            assert [str(x) for x in z["fingerprint"]] == [f"fp_{t}" for t in toks]
    n = 0
    for name, logs, want in (("missing", LOGS, sorted([t for ts in LOGS.values() for t in ts] + ["t9"])),
                             ("duplicate", {"log_a": ["t5", "t1"], "log_b": ["t1", "t0"]}, ["t0", "t1", "t5"])):
        with tempfile.TemporaryDirectory(dir=os.environ.get("SCORER1_TMP") or None) as r:
            out = build(r, logs)
            try:
                run(r, doc_file(r, want, logs), out)
            except AssertionError:
                n += 1
                continue
            raise SystemExit(f"RED arm {name} did not raise")
    assert n == 2
    print("ZZTEST_PASS consolidate order, field alignment, 2 RED arms")


if __name__ == "__main__":
    main()
