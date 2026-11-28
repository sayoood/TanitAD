#!/usr/bin/env python3
"""P5C-PROP: per-proposal NAVSIM v1.1 sub-scores for all 64 proposals of a P5-C arm, tracked AND transparent, ONE scorer call per token.

Why one call is exact (and controlled, not assumed): transp_score.score_arms scores [PDM-Closed, *tracked, *executed] in one
PDMScorer.score_proposals call and recomputes EP PAIRWISE (raw_j / max(raw_PDMClosed, raw_j)), so no proposal's EP is normalised by
another's; NC/DAC/DDC/TTC/comfort are per-proposal in the scorer. CTRL-T below is what proves it on the data.

Per token (navtest metric-cache route = the harness's own parts, as P5-C read B6 used):
  tracked[k]      the 8-pose proposal k handed to NAVSIM's LQR + bicycle (the harness's input form; proposal[pick] == seam poses)
  transp8[k]      proposal k executed as given (transp_score.transparent_states, dt 0.5; no pick20 exists for k != pick)
  transp20_pick   the pick's pick20 executed as given (dt 0.2) -- the exact P5-C pass1 transparent arm
CONTROLS (read by p5c_prop_read.py; the read refuses on failure):
  CTRL-T  tracked[pick] == the harness CSV row (all sub-scores + score) to 1e-9 on EVERY token
  CTRL-P  transp20_pick == P5-C pass1 transparent row for the arm to 1e-9 on EVERY token
  RED     --red rolls the proposal axis by one (k -> k+1): CTRL-T MUST then FAIL (exit of the read != 0)
Output: jsonl, one row per token: {token, log, t_s, pick, tr:{nc,dac,ddc,ttc,c (lists of 64), ep, pdms (lists of 64)}, tp:{...},
  tp20_pick:{...}}. Resumable (rows without "error" are skipped). Gate: refuses to start, and pauses between tokens, while the dev
  box's free commit (FreeVirtualMemory) is below 5.5 GB. CPU only; at most 2 processes (shards) on the shared box.
Usage (navsim-crun venv python; PYTHONPATH = <navsim devkit 3e8291b>;<PK/refe> with ';' on Windows):
  p5c_prop_score.py --arm C1 --proposals <stage2_p5c_c1.proposals.npz> --out <rows_s0of2.jsonl> --shard 0/2 [--tokens tok.json]
                    [--limit n] [--red]
"""
from __future__ import annotations

import argparse, hashlib, json, os, subprocess, sys, time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

TIP_PK = "F:/TanitAD_p5cread_tip/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
NAVSIM_ROOT = "F:/Archive/devbox-C/navsim"
CACHE = f"{NAVSIM_ROOT}/exp/w3_navtest_v1/metric_cache_navtest"
TOKLOG = f"{NAVSIM_ROOT}/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
COMMIT_FLOOR_GB = 5.5


def commit_free_gb() -> float:
    r = subprocess.run(["powershell", "-NoProfile", "-Command",
                        "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB,2)"], capture_output=True, text=True)
    try:
        return float(r.stdout.strip().replace(",", "."))
    except ValueError:
        return 0.0


def wait_commit(tag: str):
    while True:
        g = commit_free_gb()
        if g >= COMMIT_FLOOR_GB:
            return g
        print(f"[{tag}] commit free {g} GB < {COMMIT_FLOOR_GB}; waiting 60 s", flush=True)
        time.sleep(60)


def pack(res: list) -> dict:
    return {"nc": [r["nc"] for r in res], "dac": [r["dac"] for r in res], "ddc": [r["ddc"] for r in res], "ttc": [r["ttc"] for r in res],
            "c": [r["c"] for r in res], "ep": [r["ep"] for r in res], "pdms": [r["pdms"] for r in res]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--proposals", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--tokens", default=None, help="json list of tokens (default: all tokens of the dump)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--red", action="store_true", help="RED arm: roll the proposal axis by one; CTRL-T must FAIL")
    ap.add_argument("--check-every", type=int, default=200, help="re-check the commit floor every n tokens")
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(TIP_PK, "refe"))
    sys.path.insert(0, os.path.join(TIP_PK, "eval", "instruments"))
    import transp_score as TS
    from navsim_pdm.pdm_array_representation import ego_state_to_state_array
    g = wait_commit(f"{a.arm} start")
    print(f"[{a.arm}] commit free {g} GB; transp_score {TS.__file__}", flush=True)
    z = np.load(a.proposals, allow_pickle=False)
    toks = [str(t) for t in z["token"]]
    P, pick, p20 = z["proposals"], z["pick"], z["pick20"]
    assert P.shape[1:] == (64, 8, 3), P.shape
    if a.red:
        P = np.roll(P, 1, axis=1)
    want = set(json.load(open(a.tokens, encoding="utf-8"))) if a.tokens else set(toks)
    tlog = TS.load_token_log(TOKLOG)
    done = TS.done_tokens(a.out)
    si, sn = (int(x) for x in a.shard.split("/"))
    todo = [i for i, t in enumerate(toks) if t in want and t not in done and int(hashlib.sha1(t.encode()).hexdigest(), 16) % sn == si]
    print(f"[{a.arm}] shard {a.shard}: {len(todo)} tokens to score ({len(done)} done)", flush=True)
    n, t0 = 0, time.time()
    with open(a.out, "a", encoding="utf-8") as fo:
        for i in todo:
            if n and n % a.check_every == 0:
                wait_commit(f"{a.arm} {n}")
            t, lg = toks[i], tlog[toks[i]]
            ts = time.time()
            try:
                parts = TS.load_cache_parts(CACHE, lg, t)
                ego = parts[1]
                s0 = ego_state_to_state_array(ego)
                tracked = [np.asarray(P[i, k], np.float64) for k in range(64)]
                ex = [TS.transparent_states(tracked[k], ego, s0, 0.5) for k in range(64)]
                ex.append(TS.transparent_states(np.asarray(p20[i], np.float64), ego, s0, 0.2))
                res, _ = TS.score_arms(parts, tracked, ex)
                assert len(res) == 129, len(res)
                rec = {"token": t, "log": lg, "pick": int(pick[i]), "tr": pack(res[:64]), "tp": pack(res[64:128]), "tp20_pick": res[128],
                       "t_s": round(time.time() - ts, 3), "red": bool(a.red)}
            except Exception as e:  # noqa: BLE001
                rec = {"token": t, "log": lg, "error": f"{type(e).__name__}: {str(e)[:200]}"}
            fo.write(json.dumps(rec) + "\n")
            fo.flush()
            n += 1
            if a.limit and n >= a.limit:
                break
    print(f"ZZPROP_SHARD_DONE {a.arm} {a.shard} {n} {round(time.time() - t0, 1)} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
