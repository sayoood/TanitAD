"""run_mutations_g.py -- the mutation runner of card PREREG_V8F_PROOF_ADD41 (controls g4 and g5). Each mutation is applied to a TEMP COPY of tok_probe_g.py (the real file is never touched); the test suite
test_tok_probe_g.py is run against the copy (TOK_PROBE_G_FILE) and must go RED; the unmutated copy (control) must be GREEN. Sequential, one pytest process at a time, CPU only.

  python run_mutations_g.py [--only m1,m2] [--out MUTATIONS_G.json]
Exit 0 = every mutation red and the control green.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "tok_probe_g.py"
WORK = HERE / "tmp" / "mut"
PT = HERE / "tmp" / "pt"
TMPDIR = HERE / "tmp" / "t"

# name -> (what it breaks, exact text of ONE line fragment, its replacement)
MUTATIONS = {
    "m1_loss_line_removed": ("the loss line of the probe step is replaced by a loss with no graph to the parameters (g4: gradients)",
                             'loss = w_loss * T["lead_sup"]  # LOSS-LINE', 'loss = 0.0 * out.sum()  # LOSS-LINE'),
    "m2_4x10_grid_under_the_8x20_flag": ("the trainer's data builder is handed pool4x10 although the run declares pool8x20 (g5: declared against built; g4: 160 tokens)",
                                         'return res + ["--grid", grid, "--out"', 'return res + ["--grid", "pool4x10", "--out"'),
    "m3_token_count_refusal_removed": ("the shape check of every training batch is replaced by a read of the shape without a refusal (g5)",
                                      'n_tok_seen = check_batch_shape(b["feats"], n_tok, expect_batch, hist)  # TOKCOUNT-REFUSAL', 'n_tok_seen = int(b["feats"].shape[2])  # TOKCOUNT-REFUSAL'),
    "m4_checkpoint_records_40_tokens": ("the checkpoint records 40 tokens seen for an 8x20 run (g5: the checkpoint records the token count actually seen)",
                                        'blob["n_tok_seen"] = None if n_tok_seen is None else int(n_tok_seen)  # CKPT-NTOK', 'blob["n_tok_seen"] = 40  # CKPT-NTOK'),
    "m4b_checkpoint_declares_40_tokens": ("the checkpoint's declared n_tok is 40 for an 8x20 probe (g5)", '"seed": parse_arm(arm)["seed"], "grid": grid, "n_tok": p.n_tok,', '"seed": parse_arm(arm)["seed"], "grid": grid, "n_tok": 40,'),
    "m5_extract_reads_the_4x10_key": ("extract reads the 4 by 10 cache key under the 8 by 20 flag (g5; the key the unchanged tool reads)", 'x = d[grid]  ', 'x = d["pool4x10"]  '),
    "m6_arm_grid_refusal_removed": ("an F arm under pool4x10 or a TK arm under pool8x20 is accepted", 'if ag != grid:', 'if False:'),
    "m7_g2_bar_always_passes": ("control g2 reports pass whatever the correlation", 'out["pass"] = bool(out["corr"] >= args.bar)', 'out["pass"] = True'),
    "m8_control_parameter_in_the_graph": ("the control parameter is added to the forward output, so its gradient differs from exactly 0.0 (g4)",
                                          'return self.head(self.project(tokens))  ', 'return self.head(self.project(tokens)) + self.ctrl  '),
}


# the tests each mutation is meant to be caught by (a second, narrower run: the whole-suite run above stops at the first failure of ANY test)
INTENDED = {
    "m1_loss_line_removed": "g4",
    "m2_4x10_grid_under_the_8x20_flag": "g5_the_cli or extract_at_160 or host_memory",
    "m3_token_count_refusal_removed": "g5_a_stream",
    "m4_checkpoint_records_40_tokens": "g5_a_stream or g5_the_cli",
    "m4b_checkpoint_declares_40_tokens": "g5_a_stream or g5_the_cli or extract_at_160",
    "m5_extract_reads_the_4x10_key": "extract_at_160 or extract_refusals",
    "m6_arm_grid_refusal_removed": "g5_batch_shape or cli_refusals or extract_refusals or g5_a_stream",
    "m7_g2_bar_always_passes": "g2_pool2x2",
    "m8_control_parameter_in_the_graph": "g4_gradients",
}


def apply(name, text):
    _, old, new = MUTATIONS[name]
    n = text.count(old)
    if n != 1:
        raise SystemExit("mutation %s: the fragment occurs %d times in tok_probe_g.py, expected exactly 1" % (name, n))
    return text.replace(old, new)


def run_suite(tool_file, tag, timeout=3000, k=None):
    env = dict(os.environ)
    env.update({"TOK_PROBE_G_FILE": str(tool_file), "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "-1", "TMP": str(TMPDIR), "TEMP": str(TMPDIR)})
    TMPDIR.mkdir(parents=True, exist_ok=True)
    (PT).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", str(HERE / "test_tok_probe_g.py"), "--basetemp", str(PT / ("mut_" + tag))] + (["-k", k] if k else []), env=env, capture_output=True,
                       text=True, encoding="utf-8", timeout=timeout, cwd=str(HERE))
    out = r.stdout + r.stderr
    failed = re.findall(r"^FAILED (\S+)", out, re.M) + re.findall(r"^ERROR (\S+::\S+)", out, re.M)      # a failed test, or a test whose fixture raised; never a collection error
    summ = re.findall(r"(\d+) (passed|failed|error)", out)
    return {"rc": r.returncode, "failed_tests": failed, "summary": " ".join("%s %s" % x for x in summ), "seconds": round(time.time() - t0, 1), "tail": out.strip().splitlines()[-3:]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=str(HERE / "MUTATIONS_G.json"))
    ap.add_argument("--no-control", action="store_true")
    a = ap.parse_args(argv)
    names = [n for n in (a.only.split(",") if a.only else list(MUTATIONS)) if n]
    text = SRC.read_text(encoding="utf-8")
    res = {"tool": str(SRC), "mutations": {}, "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for n in names:
        d = WORK / n
        d.mkdir(parents=True, exist_ok=True)
        f = d / "tok_probe_g.py"
        f.write_text(apply(n, text), encoding="utf-8", newline="\n")
        r = run_suite(f, n)
        r["what"] = MUTATIONS[n][0]
        r["red"] = bool(r["rc"] != 0 and r["failed_tests"])                       # a failed TEST, not a collection error
        ri = run_suite(f, n + "_intended", k=INTENDED[n])
        ri["red"] = bool(ri["rc"] != 0 and ri["failed_tests"])
        r["intended_k"], r["intended"] = INTENDED[n], ri
        res["mutations"][n] = r
        print("%-40s %s  (%s; first failed: %s; %.0f s) | intended tests (-k %r): %s %s" % (n, "RED" if r["red"] else "NOT RED", r["summary"], r["failed_tests"][:1], r["seconds"], INTENDED[n],
                                                                                         "RED" if ri["red"] else "NOT RED", ri["failed_tests"][:1]), flush=True)
    if not a.no_control:
        d = WORK / "control_unmutated"
        d.mkdir(parents=True, exist_ok=True)
        f = d / "tok_probe_g.py"
        f.write_text(text, encoding="utf-8", newline="\n")
        c = run_suite(f, "control")
        c["green"] = bool(c["rc"] == 0)
        res["control"] = c
        print("%-40s %s  (%s; %.0f s)" % ("control_unmutated", "GREEN" if c["green"] else "NOT GREEN", c["summary"], c["seconds"]), flush=True)
    res["n_mutations"] = len(res["mutations"])
    res["n_red"] = sum(1 for r in res["mutations"].values() if r["red"])
    res["n_red_on_intended_tests"] = sum(1 for r in res["mutations"].values() if r["intended"]["red"])
    res["control_green"] = bool(res.get("control", {}).get("green"))
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8", newline="\n")
    print("mutations red: %d of %d (on the intended tests: %d); control green: %s" % (res["n_red"], res["n_mutations"], res["n_red_on_intended_tests"], res["control_green"]))
    return 0 if res["n_red"] == res["n_mutations"] == res["n_red_on_intended_tests"] and (a.no_control or res["control_green"]) else 2


if __name__ == "__main__":
    sys.exit(main())
