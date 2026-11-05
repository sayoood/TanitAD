#!/usr/bin/env python3
"""cl_precheck.py -- the pulled-file controls of the CL read chain, in seconds, BEFORE any rollout (raw/CL_PRECHECK.json; exit 0 = PASS, 8 = a control failed, 5 = a file is absent).

Per seed s in {0, 1}, run dir  <rdb>/M2b_s{s}_recft_CL/  and marker  <rdb>/cl_markers/RUN_CL_{s}.done :
  KP1  ckpt_last.pt, SHA256SUMS.txt, argv.json exist (REFUSED when the checkpoint is absent: no CL checkpoint, no read)
  KP2  sha256(ckpt_last.pt) == the SHA256SUMS.txt line == the box-2 marker's evidence.ckpt_sha256 (a marker without one is refused)
  KP3  argv audit: the argv equals C0's argv (<rdb>/M2b_s{s}_recft_C0/argv.json) plus EXACTLY `--level-head on --w-level W` (``--out`` ignored), W = the frozen W (cl_prereg/FROZEN_W.txt, first non-comment line,
       optionally `W=<number>`), the same W at both seeds, `--seed s`
  KP4  the checkpoint's own cfg: level_head == 'on', w_level == W, step == the argv's --steps; the state dict carries every level-head key (lvl_head.*, e_level.*)
  KP5  REPORTED (not a gate): sum |e_level| of the trained checkpoint.  Exactly 0.0 means the lever never reached O in training (the read is then a read of a vacuous head) and the reader's headline says so.
No GPU, torch is used to read the checkpoint dict only (CPU).
usage: python -I cl_precheck.py [--rdb E:/Projects/TanitAD-artifacts/v8f-recover] [--frozen-w <file>] [--raw <dir>] [--no-ckpt-load]"""
import argparse
import hashlib
import json
import os
import re
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
RDB = "E:/Projects/TanitAD-artifacts/v8f-recover"
FROZEN_W = "E:/Projects/TanitAD-artifacts/dagland/motion_c/cl_prereg/FROZEN_W.txt"
LEVEL_PREFIXES = ("lvl_head.", "e_level.")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_frozen_w(path):
    """The first non-comment line of FROZEN_W.txt: a positive number, optionally `W=<number>`.  Returns the string exactly as written."""
    for line in open(path, encoding="utf-8"):
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        m = re.fullmatch(r"(?:W\s*=\s*)?([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", t)
        if not m or not float(m.group(1)) > 0:
            raise ValueError(f"FROZEN_W.txt first line {t!r} is not a positive number")
        return m.group(1)
    raise ValueError("FROZEN_W.txt has no value")


def parse_argv(tokens):
    d, i = {}, 0
    while i < len(tokens):
        t = tokens[i]
        if not t.startswith("--"):
            raise ValueError(f"argv token {t!r} is not a flag")
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            d[t], i = tokens[i + 1], i + 2
        else:
            d[t], i = True, i + 1
    return d


def argv_audit(cl_tokens, c0_tokens, s, w):
    """CL's argv = C0's argv + `--level-head on --w-level W`, nothing else (``--out`` ignored)."""
    cl, c0 = parse_argv(cl_tokens), parse_argv(c0_tokens)
    x, r = {k: v for k, v in cl.items() if k != "--out"}, {k: v for k, v in c0.items() if k != "--out"}
    added, removed = {k: x[k] for k in x if k not in r}, sorted(k for k in r if k not in x)
    changed = {k: (r[k], x[k]) for k in x if k in r and r[k] != x[k]}
    c = {"added_exactly_the_two_level_tokens": bool(set(added) == {"--level-head", "--w-level"} and added.get("--level-head") == "on"), "nothing_removed": not removed, "nothing_changed": not changed,
         "w_level_is_the_frozen_W": bool(added.get("--w-level") is not None and _same_num(added.get("--w-level"), w)), "seed_is_s": cl.get("--seed") == str(s), "init_ckpt_is_c0s": cl.get("--init-ckpt") == c0.get("--init-ckpt")}
    c["diff"] = {"added": added, "removed": removed, "changed": {k: list(v) for k, v in changed.items()}}
    c["PASS"] = all(v for k, v in c.items() if k != "diff")
    return c


def _same_num(a, b):
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return False


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rdb", default=RDB)
    ap.add_argument("--frozen-w", default=FROZEN_W)
    ap.add_argument("--raw", default=f"{PK}/raw")
    ap.add_argument("--no-ckpt-load", action="store_true", help="skip KP4/KP5 (the checkpoint dict is not opened)")
    a = ap.parse_args(argv)
    os.makedirs(a.raw, exist_ok=True)
    res = {"card": "CL read chain pre-check", "rdb": a.rdb, "seeds": {}, "failures": [], "clock": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    try:
        w = read_frozen_w(a.frozen_w)
    except Exception as e:
        res["failures"].append(f"frozen W: {type(e).__name__}: {e}")
        w = None
    res["w"], res["frozen_w_file"] = w, a.frozen_w
    absent = False
    for s in (0, 1):
        k, d = f"s{s}", f"{a.rdb}/M2b_s{s}_recft_CL"
        r = {"run_dir": d}
        for f in ("ckpt_last.pt", "SHA256SUMS.txt", "argv.json"):
            if not os.path.isfile(f"{d}/{f}"):
                res["failures"].append(f"{k}: {d}/{f} is absent")
                absent = True
        mk = f"{a.rdb}/cl_markers/RUN_CL_{s}.done"
        if not os.path.isfile(mk):
            res["failures"].append(f"{k}: box-2 marker {mk} is absent")
            absent = True
        if os.path.isfile(f"{d}/ckpt_last.pt") and os.path.isfile(f"{d}/SHA256SUMS.txt"):
            got = sha256_file(f"{d}/ckpt_last.pt")
            sums = [l.split()[0] for l in open(f"{d}/SHA256SUMS.txt", encoding="utf-8") if l.strip().split()[-1].lstrip("*").endswith("ckpt_last.pt")]
            r["ckpt_sha256"], r["sha256sums_line"] = got, sums[0] if sums else None
            if not sums or got != sums[0]:
                res["failures"].append(f"{k}: ckpt sha256 {got} != SHA256SUMS.txt {sums[0] if sums else None} (a partial copy?)")
            if os.path.isfile(mk):
                ev = (json.load(open(mk, encoding="utf-8")).get("evidence") or {}).get("ckpt_sha256")
                r["marker_ckpt_sha256"] = ev
                if not ev:
                    res["failures"].append(f"{k}: RUN_CL_{s}.done carries no evidence.ckpt_sha256")
                elif ev != got:
                    res["failures"].append(f"{k}: ckpt sha256 {got} != box-2 marker {ev}")
        if os.path.isfile(f"{d}/argv.json") and os.path.isfile(f"{a.rdb}/M2b_s{s}_recft_C0/argv.json"):
            ct = json.load(open(f"{d}/argv.json", encoding="utf-8"))["argv"]
            c0 = json.load(open(f"{a.rdb}/M2b_s{s}_recft_C0/argv.json", encoding="utf-8"))["argv"]
            r["argv_audit"] = argv_audit(ct, c0, s, w)
            if not r["argv_audit"]["PASS"]:
                res["failures"].append(f"{k}: argv audit failed: {r['argv_audit']}")
            r["steps_argv"] = parse_argv(ct).get("--steps")
        if os.path.isfile(f"{d}/ckpt_last.pt") and not a.no_ckpt_load:
            import torch
            ck = torch.load(f"{d}/ckpt_last.pt", map_location="cpu", weights_only=False)
            cfg, sd = ck["cfg"], ck["model"]
            lv = sorted(k_ for k_ in sd if k_.startswith(LEVEL_PREFIXES))
            e_abs = float(sum(sd[k_].abs().sum() for k_ in lv if k_.startswith("e_level.")))
            r.update({"cfg_level_head": cfg.get("level_head"), "cfg_w_level": cfg.get("w_level"), "step": ck.get("step"), "level_keys": lv, "e_level_abs_sum": e_abs, "e_level_all_zero": bool(e_abs == 0.0)})
            if cfg.get("level_head") != "on":
                res["failures"].append(f"{k}: the checkpoint's cfg level_head is {cfg.get('level_head')!r}, not 'on'")
            if w is not None and not _same_num(cfg.get("w_level"), w):
                res["failures"].append(f"{k}: the checkpoint's cfg w_level {cfg.get('w_level')!r} is not the frozen W {w!r}")
            if not any(x.startswith("lvl_head.") for x in lv) or not any(x.startswith("e_level.") for x in lv):
                res["failures"].append(f"{k}: the state dict lacks level-head keys ({lv})")
            if r.get("steps_argv") is not None and str(ck.get("step")) != str(r["steps_argv"]):
                res["failures"].append(f"{k}: checkpoint step {ck.get('step')} != argv --steps {r['steps_argv']}")
        res["seeds"][k] = r
    if w is not None and len({json.dumps(res['seeds'][k].get("argv_audit", {}).get("diff", {}).get("added", {}).get("--w-level")) for k in res["seeds"]}) > 1:
        res["failures"].append("the two seeds carry different --w-level values")
    res["PASS"] = not res["failures"]
    json.dump(res, open(f"{a.raw}/CL_PRECHECK.json", "w", encoding="utf-8", newline="\n"), indent=1)
    for f in res["failures"]:
        print("PRECHECK FAIL:", f)
    print("PRECHECK PASS" if res["PASS"] else "PRECHECK FAIL")
    return 0 if res["PASS"] else (5 if absent else 8)


if __name__ == "__main__":
    sys.exit(main())
