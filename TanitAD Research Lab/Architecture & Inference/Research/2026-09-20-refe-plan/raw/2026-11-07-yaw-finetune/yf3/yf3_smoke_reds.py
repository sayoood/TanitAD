#!/usr/bin/env python3
"""YF3 SMOKE: the EVAL-SIDE and IDENTITY-SIDE deliberate-regression arms, run on the smoke pipeline's artifacts. Every one must be RED (or, for the green controls, green).
    python yf3_smoke_reds.py --root <smoke_run> [--live]      (--live also runs the G7 live-refusal arm through the real seam: takes the GPU lock, ~3-6 min)
"""
import argparse
import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import yf3_common as C  # noqa: E402
import yf3_analyze as A  # noqa: E402

sys.path.insert(0, str(C.Y2))
import yaw_common as YC  # noqa: E402

RES = []


def check(name, ok, detail=""):
    RES.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--live", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    toks = sorted(C.jload(root / "tokens" / "eval_tokens.json")["tokens"])
    table, _, _ = YC.load_yaw_table(C.TABLE_TEST)
    A0, B0 = A.load_arm(root, "A0", toks), A.load_arm(root, "R1_B_s0", toks)
    g6 = A.gate_g6(A0, toks)
    check("G6 green control: A0 re-run == banked plans and per-token scores, bit for bit (through the YT launcher path)", g6["ok"], f"plans {g6['plans_bit_identical']} scores {g6['scores_identical']} n {g6['n']}")
    g6r = A.gate_g6(B0, toks)
    check("RED-AS-REQUIRED G6: a different checkpoint standing in for A0", not g6r["ok"], f"plans_bit_identical {g6r['plans_bit_identical']}")
    g7 = A.gate_g7(B0, toks, table, True)
    check("G7 green control: B_s0 fed float32(10 x navtest table) on every token, >= 90 % non-zero, nothing incoming", g7["ok"], f"exact {g7.get('equal_float32_10x_table')}/{g7.get('n_tokens')} nonzero {g7.get('nonzero_fraction')}")
    check("G7 green control: A0 -> YT's shim not installed, 0 calls", A.gate_g7(A0, toks, table, False)["ok"])
    check("RED-AS-REQUIRED G7: A0 expected to be fed a yaw (shim absent)", not A.gate_g7(A0, toks, table, True)["ok"])
    check("RED-AS-REQUIRED G7: a yaw-fed arm judged as a non-yaw arm (shim must stay out)", not A.gate_g7(B0, toks, table, False)["ok"])
    bad = json.loads(json.dumps(B0["g7"]))
    t0 = toks[0]
    bad["by_token"][t0] = bad["by_token"][t0] * 0.5
    check("RED-AS-REQUIRED G7: one token fed a wrong value (x0.5)", not A.gate_g7({**B0, "g7": bad}, toks, table, True)["ok"])
    zero = json.loads(json.dumps(B0["g7"]))
    zero["by_token"] = {t: 0.0 for t in zero["by_token"]}
    check("RED-AS-REQUIRED G7: every ego[4] equal to 0.0 (yaw loaded but not applied)", not A.gate_g7({**B0, "g7": zero}, toks, table, True)["ok"])
    # identity-side RED arms (CPU)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="12", MKL_NUM_THREADS="12")
    outd = root / "reds"
    outd.mkdir(exist_ok=True)

    def ident(kind, ck, fin, out, extra=()):
        r = subprocess.run([C.PY_D, str(HERE / "yf3_identity.py"), "--kind", kind, "--out", str(out), "--frames", "128", *extra] + (["--ckpt", str(ck)] if ck else []) + (["--final", str(fin)] if fin else []),
                           capture_output=True, text=True, env=env, encoding="utf-8", errors="replace")
        return r.returncode
    rc = ident("C", root / "ckpt" / "R1_B_s0.pt", root / "arms" / "R1_C_s0" / "final.pt", outd / "id_C_given_B.json")
    check("RED-AS-REQUIRED identity: a yaw-trained B checkpoint presented as arm C (C == A0 must fail)", rc == 3, f"rc {rc}")
    rc = ident("B0", None, None, outd / "id_B0_defaultcol.json", ["--regress", "default_col"])
    check("RED-AS-REQUIRED identity: B0 with u400's random column left in place", rc == 3, f"rc {rc}")
    rc = ident("B0", None, None, outd / "id_B0.json")
    check("identity green control: B0 (zero column) decodes bit-identically to A0 with omega = 0 and omega != 0", rc == 0, f"rc {rc}")
    if a.live:
        sys.path.insert(0, str(HERE))
        import yf3_eval_arm as E
        tab, meta, sha = YC.load_yaw_table(C.TABLE_TEST)
        drop = {toks[0]} | set(sorted(tab)[:200])                               # the ONE token of the live run is absent from the table
        one = outd / "one_token.json"
        tk = C.jload(root / "tokens" / "eval_tokens.json")
        C.jdump({"rule": tk.get("rule", ""), "tokens": [toks[0]], "token_log": {toks[0]: tk["token_log"][toks[0]]}}, one)
        bad_tab = {k: v for k, v in tab.items() if k not in drop}
        p = outd / "ego_yaw_navtest_DROPPED_RED.json.gz"
        with gzip.GzipFile(filename="", mode="wb", fileobj=open(p, "wb"), mtime=0) as gz:
            gz.write(json.dumps({"meta": {**meta, "n_tokens": len(bad_tab)}, "table": bad_tab}).encode())
        ck = root / "ckpt" / "R1_B_s0.pt"
        if not ck.exists():                                                      # the pipeline deletes a non-ADOPT rung's composed checkpoints; recompose deterministically
            subprocess.run([C.PY_D, str(HERE / "build_yaw_ckpt_yf3.py"), "--final", str(root / "arms" / "R1_B_s0" / "final.pt"), "--out", str(outd / "R1_B_s0_recomposed.pt")], check=True, env=env, capture_output=True)
            ck = outd / "R1_B_s0_recomposed.pt"
        d2 = outd / "live_g7_missing_token"
        d2.mkdir(exist_ok=True)
        try:
            E.run_seam("smoke_RED_missing_token", ck, one, d2, "yf3_yaw_eval", 3.0, table=p)
            ok = False
        except RuntimeError as e:
            ok = "NOT in the yaw table" in (d2 / "seam.log").read_text(encoding="utf-8", errors="replace")
        check("RED-AS-REQUIRED G7 LIVE: the real seam REFUSES a token absent from the yaw table (never feeds 0.0)", ok)
    n_fail = RES.count(False)
    print(f"\n{len(RES) - n_fail}/{len(RES)} checks as required")
    print("ZZYF3_SMOKE_REDS_OK" if n_fail == 0 else "ZZYF3_SMOKE_REDS_FAIL")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
