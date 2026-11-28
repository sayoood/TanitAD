#!/usr/bin/env python3
"""v8f_speedsub_t1_pin.py -- the pin step and the argv renderer of the SPEEDSUB-T1 chain (PREREG_V8F_SPEEDSUB_T1 section 9, steps 2-3). Pure file work: no model, no torch, no rollout.

  pin          hash every file the read depends on and write pins.json (+ PINS.txt). The PREREG prints the four checkpoint prefixes; the pin REFUSES any file whose sha256 does not start with them, and
               `mlp_ego_c` must be the container the fit recorded (fit_c.json / READOUT_C_PIN.json). An existing pins.json is never overwritten: the new pins must EQUAL it, or the step stops.
  render-argv  print one token per line: the BANKED rollout argv of a run with ONLY --tag / --out-dir / --dump-ticks (and --limit for the logged lane, --max-windows for the smoke) changed, after
               checking every other token against the registered values. The tool path and python are the chain's.
  check-done   exit 0 iff a done record names this tag, this checkpoint sha and the pinned tool digest (resume rule: a finished rollout is skipped, an incomplete one is moved aside by the chain).
  write-done   after a rollout: write its done record (the banked shape) ONLY if the tool json, the tool ticks, the wrapper record and all 141 episode dumps (smoke: at least one) are there and the
               checkpoint still hashes to the sha it had before the rollout. Exit 1 and no file otherwise.
  get          print the value at a dotted path of pins.json (the chain reads the pinned shas with it).

    python v8f_speedsub_t1_pin.py pin --pins OUT/pins.json --code-dir <dir with the scripts> --reader <speedsub_t1_read.py> --fit-dir F:/TanitAD-scratch/speedsub_t1/fit
    python v8f_speedsub_t1_pin.py render-argv --run CTRL_s0 --mode egoc --out-root OUT [--smoke 200]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True
ART = "E:/Projects/TanitAD-artifacts"
TREE = f"{ART}/dagt1_tree_b43c7a9a"
BANKED_ARGV = {"CTRL_s0": f"{ART}/v8f-dagoacc-t1/argv/DAGOACC_s0_m10_causal_heads.argv", "CTRL_s1": f"{ART}/v8f-dagoacc-t1/argv/DAGOACC_s1_m10_causal_heads.argv",
               "REP_s0": "F:/TanitAD-scratch/dagorc_t1_out/argv/DAGORCREP_s0_m10_causal_heads.argv", "REP_s1": "F:/TanitAD-scratch/dagorc_t1_out/argv/DAGORCREP_s1_m10_causal_heads.argv"}
BANKED_DONE = {"CTRL_s0": f"{ART}/v8f-dagoacc-t1/done_DAGOACC_s0_m10_causal_heads.txt", "CTRL_s1": f"{ART}/v8f-dagoacc-t1/done_DAGOACC_s1_m10_causal_heads.txt",
               "REP_s0": "F:/TanitAD-scratch/dagorc_t1_out/done_DAGORCREP_s0_m10_causal_heads.txt", "REP_s1": "F:/TanitAD-scratch/dagorc_t1_out/done_DAGORCREP_s1_m10_causal_heads.txt"}
CKPT = {"CTRL_s0": f"{ART}/v8f-dagoacc/M2b_s0_dagoacc/ckpt_last.pt", "CTRL_s1": f"{ART}/v8f-dagoacc/M2b_s1_dagoacc/ckpt_last.pt",
        "REP_s0": f"{ART}/v8f-dag-orc-oacc/M2b_s0_dagoaccorcrep/ckpt_last.pt", "REP_s1": f"{ART}/v8f-dag-orc-oacc/M2b_s1_dagoaccorcrep/ckpt_last.pt"}
CKPT_PREFIX = {"CTRL_s0": "67265ec1", "CTRL_s1": "90b01474", "REP_s0": "b03fd6ef", "REP_s1": "77853e56"}
TOOL = f"{TREE}/stack/scripts/v8f_t1_eval.py"
TOOL_SHA256 = "61e140edb9da01acbbba4eef81e74bbb68b57e2ff31842e05958759446d97397"
TOOL_DIGEST = "6a4b47c8acfadf07ffc82ab174cccce78aa0404f6a4f89231f331013bbcda3d9"           # the pinned tool tree's digest, as the banked done records carry it
BANKED_READOUT = "F:/TanitAD-scratch/speedread/fit/readout.pt"
BANKED_READOUT_SHA256 = "a205609ab47325a0ad9c0141be8ae3a0fec653d59c6db17ec1cf0d460bf2316d"
REG_PERM_SHA = "64785f8ad0797b02bdf9845cedf6cf425f8759a2993da889cf33146a0e37ab64"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
EXPECT = {"--tracks": f"{ART}/sfresh_inputs/inputs/native_tracks.pt", "--eval-cache": f"{ART}/sfresh_inputs/cache", "--eval-grid": f"{TREE}/{PK}/eval_grid_windows.json",
          "--eval-join": f"{TREE}/{PK}/join_eval141.npz", "--replan-steps": "10", "--nav": "deadreckon", "--scene": "logged", "--schedule": "periodic", "--limit": "causal", "--device": "cpu"}
REPLACED = ("--tag", "--out-dir", "--dump-ticks")
MODES = ("identity", "egoc", "egoshufc", "identity_l", "egol")
LOGGED = ("identity_l", "egol")
N_EPISODES = 141


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def tag_of(run, mode, smoke=0):
    return f"SST1_{'SMOKE_' if smoke else ''}{run}_{mode}"


def banked_tokens(run):
    return [ln.strip() for ln in open(BANKED_ARGV[run], encoding="utf-8").read().splitlines() if ln.strip()]


def render_argv(run, mode, out_root, smoke=0):
    """-> the tool argv (starting at `run`) for (run, mode). Every token that is not replaced is checked against the registered one."""
    t = banked_tokens(run)
    if t[2] != "run":
        raise SystemExit(f"REFUSED: {BANKED_ARGV[run]} does not start with `run`")
    body = t[2:]
    out, i, seen = [], 0, {}
    while i < len(body):
        tok = body[i]
        if tok.startswith("--") and i + 1 < len(body) and not body[i + 1].startswith("--"):
            seen[tok] = body[i + 1]
            out += [tok, body[i + 1]]
            i += 2
        else:
            seen[tok] = None
            out.append(tok)
            i += 1
    if seen.get("--ckpt") != CKPT[run]:
        raise SystemExit(f"REFUSED: banked --ckpt {seen.get('--ckpt')!r} is not {CKPT[run]!r}")
    for k, v in EXPECT.items():
        if seen.get(k) != v:
            raise SystemExit(f"REFUSED: banked {k} {seen.get(k)!r} is not the registered {v!r}")
    for k in ("--dump-heads",):
        if k not in seen:
            raise SystemExit(f"REFUSED: banked argv lacks {k}")
    if mode not in MODES:
        raise SystemExit(f"REFUSED: mode {mode!r}")
    if mode in LOGGED and run not in ("CTRL_s0", "CTRL_s1"):
        raise SystemExit("REFUSED: the beside logged lane is CTRL only")
    d = out_dir_of(out_root, run, mode, smoke)
    new = {"--tag": tag_of(run, mode, smoke), "--out-dir": d, "--dump-ticks": f"{d}/ticks"}
    if mode in LOGGED:
        new["--limit"] = "logged"
    res, j = [], 0
    while j < len(out):
        tok = out[j]
        if tok in new:
            res += [tok, new[tok]]
            j += 2
        elif tok.startswith("--") and j + 1 < len(out) and not out[j + 1].startswith("--"):
            res += [tok, out[j + 1]]
            j += 2
        else:
            res.append(tok)
            j += 1
    if smoke:
        res += ["--max-windows", str(int(smoke))]
    return res


def done_ok(done_path, tag, ckpt_sha):
    try:
        d = json.load(open(done_path, encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool(d.get("tag") == tag and d.get("ckpt_sha256") == ckpt_sha and d.get("ckpt_sha256_before") == ckpt_sha and d.get("ckpt_sha256_after") == ckpt_sha and d.get("tool_digest") == TOOL_DIGEST)


def out_dir_of(out_root, run, mode, smoke=0):
    return os.path.join(out_root, "SMOKE" if smoke else "", run, mode).replace("\\", "/").replace("//", "/")


def cmd_write_done(a):
    tag, d = tag_of(a.run, a.mode, a.smoke), out_dir_of(a.out_root.rstrip("/"), a.run, a.mode, a.smoke)
    need = [f"{d}/t1v8f_{tag}.json", f"{d}/ticks/ticks_{tag}.npz", f"{d}/ticks/speedsub_{tag}.npz"]
    missing = [p for p in need if not os.path.isfile(p)]
    dumps = sorted(glob.glob(f"{d}/dump_{tag}/ep*.npz"))
    after = sha256_file(CKPT[a.run])
    problems = []
    if missing:
        problems.append(f"missing {missing}")
    if (a.smoke and not dumps) or (not a.smoke and len(dumps) != N_EPISODES):
        problems.append(f"{len(dumps)} episode dumps (need {'at least one' if a.smoke else N_EPISODES})")
    if after != a.sha_before:
        problems.append(f"the checkpoint hashes to {after[:12]}... now, {a.sha_before[:12]}... before the rollout")
    if problems:
        print(f"ZZSPEEDSUB-T1-DONE-STOP {tag}: {'; '.join(problems)}")
        return 1
    rec = {"tag": tag, "run": a.run, "mode": a.mode, "limit": "logged" if a.mode in LOGGED else "causal", "ckpt_sha256": a.sha_before, "ckpt_sha256_before": a.sha_before, "ckpt_sha256_after": after,
           "tool_digest": TOOL_DIGEST, "n_dumps": len(dumps), "wall_s": a.wall_s, "smoke": bool(a.smoke)}
    with open(f"{d}/done_{tag}.txt", "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f)
    print(f"ZZSPEEDSUB-T1-DONE-WRITTEN {tag}")
    return 0


def cmd_get(a):
    v = json.load(open(a.pins, encoding="utf-8"))
    for k in a.key.split("."):
        v = v[k]
    print(v)
    return 0


def build_pins(a):
    code = a.code_dir
    pins = {"schema": "v8f-speedsub-t1-pins-v1", "ckpts": {}, "banked_argv": {}, "banked_done_ckpt_sha": {}}
    for r in CKPT:
        sha = sha256_file(CKPT[r])
        if not sha.startswith(CKPT_PREFIX[r]):
            raise SystemExit(f"REFUSED: {CKPT[r]} has sha256 {sha[:12]}..., the PREREG prints {CKPT_PREFIX[r]}...")
        done = json.load(open(BANKED_DONE[r], encoding="utf-8"))
        if done.get("ckpt_sha256") != sha or done.get("tool_digest") != TOOL_DIGEST:
            raise SystemExit(f"REFUSED: the banked done record of {r} names another checkpoint or tool digest")
        pins["ckpts"][r] = {"path": CKPT[r], "sha256": sha}
        pins["banked_argv"][r] = {"path": BANKED_ARGV[r], "sha256": sha256_file(BANKED_ARGV[r])}
        pins["banked_done_ckpt_sha"][r] = done["ckpt_sha256"]
    ro = os.path.join(a.fit_dir, "readout_c.pt")
    pin_rec = json.load(open(os.path.join(a.fit_dir, "READOUT_C_PIN.json"), encoding="utf-8"))
    fit = json.load(open(os.path.join(a.fit_dir, "fit_c.json"), encoding="utf-8"))
    sha_c = sha256_file(ro)
    if sha_c != pin_rec["readout_c_sha256"] or sha_c != fit["container"]["sha256"] or fit.get("outcome") != "FITTED" or not fit.get("controls_pass"):
        raise SystemExit("REFUSED: readout_c.pt is not the container the fit recorded, or the fit's controls did not all pass")
    if sha256_file(os.path.join(a.fit_dir, "fit_c.json")) != pin_rec["fit_c_json_sha256"]:
        raise SystemExit("REFUSED: fit_c.json is not the one READOUT_C_PIN.json names")
    if sha256_file(BANKED_READOUT) != BANKED_READOUT_SHA256:
        raise SystemExit("REFUSED: the banked readout.pt is not a205609a...")
    pins["readout_c"] = {"path": ro, "sha256": sha_c, "key": "mlp_ego_c", "fit_c_json_sha256": pin_rec["fit_c_json_sha256"]}
    pins["readout_banked"] = {"path": BANKED_READOUT, "sha256": BANKED_READOUT_SHA256, "key": "mlp_ego"}
    if sha256_file(TOOL) != TOOL_SHA256:
        raise SystemExit("REFUSED: the T1 tool is not the pinned one")
    pins["tool_sha256"], pins["tool_path"], pins["tool_digest"] = TOOL_SHA256, TOOL, TOOL_DIGEST
    for k, f in (("wrapper", "v8f_t1_speedsub.py"), ("instrument", "v8f_mini_eval_speedread.py"), ("fit_script", "v8f_speedsub_t1_fit.py"), ("pin_script", "v8f_speedsub_t1_pin.py")):
        p = os.path.join(code, f)
        pins[f"{k}_sha256"], pins[f"{k}_path"] = sha256_file(p), p.replace("\\", "/")
    pins["reader_sha256"], pins["reader_path"] = sha256_file(a.reader), a.reader.replace("\\", "/")
    pins["perm_sha256"] = REG_PERM_SHA
    return pins


def cmd_pin(a):
    pins = build_pins(a)
    if os.path.exists(a.pins):
        old = json.load(open(a.pins, encoding="utf-8"))
        if old != pins:
            diff = sorted(k for k in set(old) | set(pins) if old.get(k) != pins.get(k))
            print(f"ZZSPEEDSUB-T1-PIN-STOP the existing pins differ from the files on disk: {diff}")
            return 2
        print("ZZSPEEDSUB-T1-PIN-OK unchanged", a.pins)
        return 0
    os.makedirs(os.path.dirname(a.pins), exist_ok=True)
    with open(a.pins, "w", encoding="utf-8", newline="\n") as f:
        json.dump(pins, f, indent=1)
    with open(os.path.splitext(a.pins)[0] + ".txt", "w", encoding="utf-8", newline="\n") as f:
        for r, v in pins["ckpts"].items():
            f.write(f"{v['sha256']}  {v['path']}  # {r}\n")
        for k in ("readout_c", "readout_banked"):
            f.write(f"{pins[k]['sha256']}  {pins[k]['path']}  # {k}\n")
        for k in ("tool", "wrapper", "instrument", "fit_script", "pin_script", "reader"):
            f.write(f"{pins[k + '_sha256']}  {pins.get(k + '_path')}  # {k}\n")
    print("ZZSPEEDSUB-T1-PIN-WRITTEN", a.pins)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pin")
    p.add_argument("--pins", required=True)
    p.add_argument("--code-dir", required=True)
    p.add_argument("--reader", required=True)
    p.add_argument("--fit-dir", default="F:/TanitAD-scratch/speedsub_t1/fit")
    p = sub.add_parser("render-argv")
    p.add_argument("--run", required=True, choices=sorted(CKPT))
    p.add_argument("--mode", required=True, choices=MODES)
    p.add_argument("--out-root", required=True)
    p.add_argument("--smoke", type=int, default=0)
    p = sub.add_parser("write-done")
    p.add_argument("--run", required=True, choices=sorted(CKPT))
    p.add_argument("--mode", required=True, choices=MODES)
    p.add_argument("--out-root", required=True)
    p.add_argument("--smoke", type=int, default=0)
    p.add_argument("--sha-before", required=True)
    p.add_argument("--wall-s", type=int, default=0)
    p = sub.add_parser("get")
    p.add_argument("--pins", required=True)
    p.add_argument("--key", required=True)
    p = sub.add_parser("check-done")
    p.add_argument("--done", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--ckpt-sha", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "pin":
        return cmd_pin(a)
    if a.cmd == "write-done":
        return cmd_write_done(a)
    if a.cmd == "get":
        return cmd_get(a)
    if a.cmd == "render-argv":
        print("\n".join(render_argv(a.run, a.mode, a.out_root.rstrip("/"), a.smoke)))
        return 0
    return 0 if done_ok(a.done, a.tag, a.ckpt_sha) else 1


if __name__ == "__main__":
    sys.exit(main())
