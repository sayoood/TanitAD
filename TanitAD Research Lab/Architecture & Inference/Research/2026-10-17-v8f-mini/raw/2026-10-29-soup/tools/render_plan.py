#!/usr/bin/env python3
"""Render (never execute) every dev-box step of PREREG_V8F_SOUP section 13 from the templates in ../argv, with the parameter values of the commands
the LAUNCH-INTENT chain RAN (launch_intent_t1/logs/chain.out, 2026-10-28).  Writes ../argv/PLAN.txt and ../argv/TEMPLATES.sha256; checks that rendering
the causal and fresh templates for A_0 reproduces the two commands that ran, token for token."""
import hashlib
import json
import os
import re
import shlex

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.dirname(HERE).replace("\\", "/")
A = "E:/Projects/TanitAD-artifacts"
TREE = "F:/TanitAD-scratch/launch_intent_t1/tree_li"
RAW = f"{A}/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
SF = f"{A}/sfresh_inputs"
OUT = "F:/TanitAD-scratch/soup_t1_out"
P = {"PY": "C:/Users/Admin/venvs/tanitad/Scripts/python.exe", "TREE": TREE, "TRACKS": f"{SF}/inputs/native_tracks.pt", "CACHE": f"{SF}/cache",
     "GRID": f"{RAW}/eval_grid_windows.json", "JOIN": f"{RAW}/join_eval141.npz", "KIN": f"{RAW}/eval141_kin.npz", "V9": f"{SF}/inputs/v9",
     "P10": f"{SF}/inputs/p10/p10_prior.pt"}


def tmpl(name):
    with open(f"{W}/argv/{name}", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f if ln.strip() and not ln.startswith("#")]


def render(name, **kw):
    v = {**P, **kw}
    toks = []
    for t in tmpl(name):
        toks.append(re.sub(r"@([A-Z0-9_]+)@", lambda m: v[m.group(1)], t))
    return toks


def ran(prefix):
    with open(f"{A}/launch_intent_t1/logs/chain.out", encoding="utf-8") as f:
        for ln in f:
            if prefix in ln:
                return shlex.split(ln.split("): " if "limit causal" in ln else "-fresh: ", 1)[1].strip())
    raise SystemExit(f"no ran line for {prefix}")


def main():
    li = f"{A}/v8f-launch-intent/M2b_s0_launchintent/ckpt_last.pt"
    c = render("rollout_causal.tmpl", CKPT=li, TAG="LI_s0_m10_causal_heads", OUT="F:/TanitAD-scratch/launch_intent_t1_out")
    f = render("t0fresh.tmpl", CKPT=li, TAG="M2b-launchintent_s0-fresh", T0DIR="F:/TanitAD-scratch/launch_intent_t1_out/t0fresh")
    proof = {"causal": c == ran("start LI_s0_m10_causal_heads"), "t0fresh": f == ran("start fresh-T0 eval M2b-launchintent_s0-fresh")}
    lines = [f"# PREREG_V8F_SOUP section 13: every dev-box step, RENDERED, NEVER EXECUTED. Template proof vs the LAUNCH-INTENT commands that ran: {json.dumps(proof)}",
             "# one torch process at a time; OMP/MKL 8 threads; RAM >= 4096 MB and F: >= 5 GB before every step (the LAUNCH-INTENT chain's guards)"]
    spec = json.load(open(f"{W}/soups.json", encoding="utf-8"))
    for name, s in spec["soups"].items():
        if s.get("expect"):
            continue
        ck = f"{W}/ckpt/{s['run_dir']}/ckpt_last.pt"
        tag = name
        for t, kw in (("rollout_causal.tmpl", {"TAG": f"{tag}_m10_causal_heads", "OUT": OUT}), ("rollout_logged.tmpl", {"TAG": f"{tag}_m10_logged_heads", "OUT": OUT}),
                      ("t0fresh.tmpl", {"TAG": f"M2b-{tag.lower()}-fresh", "T0DIR": f"{OUT}/t0fresh"}), ("t0std.tmpl", {"TAG": f"M2b-{tag.lower()}", "T0DIR": f"{OUT}/t0std"})):
            lines.append(shlex.join(render(t, CKPT=ck, **kw)))
    for role, ck in (("DAGOACC_0", f"{A}/v8f-dagoacc/M2b_s0_dagoacc/ckpt_last.pt"), ("DAGOACC_1", f"{A}/v8f-dagoacc/M2b_s1_dagoacc/ckpt_last.pt"),
                     ("A_0", f"{A}/v8f-launch-intent/M2b_s0_launchintent/ckpt_last.pt"), ("A_1", f"{A}/v8f-launch-intent/M2b_s1_launchintent/ckpt_last.pt")):
        lines.append(shlex.join(render("t0std.tmpl", CKPT=ck, TAG=f"M2b-{role.lower()}-dev", T0DIR=f"{OUT}/t0dev")))
    with open(f"{W}/argv/PLAN.txt", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(f"{W}/argv/TEMPLATES.sha256", "w", encoding="utf-8", newline="\n") as fh:
        for t in ("rollout_causal.tmpl", "rollout_logged.tmpl", "t0fresh.tmpl", "t0std.tmpl"):
            fh.write(f"{hashlib.sha256(open(f'{W}/argv/{t}', 'rb').read()).hexdigest()}  {t}\n")
    print("template proof", proof, "steps", len(lines) - 2)
    return 0 if all(proof.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
