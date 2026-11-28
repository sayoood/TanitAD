#!/usr/bin/env python3
"""Bank ONE finished Thor arm back to the dev box and, if the dev box has a PASS for the same arm,
write the cross-backend diff + admission verdict under RULING_BACKEND_POLICY.md (any python).

    python bank_and_compare.py --split navtest --label r7thor_s50400_R7_A1 --arm R7_A1 \
        --thor-dir /dev/shm/navsim/thor_scores/navtest/r7thor_s50400_R7_A1 \
        --dest <…/step50400/thor_scores/navtest> --devbox-counts <…/scores_navtest/r7s50400_R7_A1/r7s50400_R7_A1.counts.json>

* pulls the Thor dir with scp (never the standard dev-box names: the destination is ``thor_scores/``);
* refuses to bank as PASS unless the Thor counts.json says PASS (artifact, not exit code);
* writes ``<label>.thor`` (provenance: backend, host, seam sha256, agent, policy file) next to it;
* if the dev-box counts.json exists AND says PASS: ``full_split_compare`` on the per-token CSV (navtest)
  or final frame + devkit CSV (navhard), and ``<label>.cross_backend.json`` with the policy checks
  (0 discrete flips incl. valid state, per-token |d| <= 1e-9, split-mean |d| <= 1e-12).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RULING = os.path.join(os.path.dirname(HERE), "RULING_BACKEND_POLICY.md")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("navtest", "navhard"), required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--thor-dir", default="", help="pull this dir from Thor (omit when already local)")
    ap.add_argument("--dest", required=True)
    ap.add_argument("--devbox-counts", default="")
    ap.add_argument("--devbox-csv", default="")
    ap.add_argument("--devbox-frame", default="")
    ap.add_argument("--thor-csv", default="")
    ap.add_argument("--thor-frame", default="")
    ap.add_argument("--thor-counts", default="")
    a = ap.parse_args()
    os.makedirs(a.dest, exist_ok=True)
    local = os.path.join(a.dest, a.label)
    if a.thor_dir:
        # MSYS scp would rewrite the remote "/dev/shm/..." into "C:/Program Files/Git/dev/shm/..." (MEASURED) -> disable path conversion
        r = subprocess.run(["scp", "-q", "-r", f"tanitad-thor-wifi:{a.thor_dir}", a.dest], capture_output=True, text=True,
                           env=dict(os.environ, MSYS_NO_PATHCONV="1", MSYS2_ARG_CONV_EXCL="*"))
        if r.returncode != 0 or not os.path.isdir(local):
            print(json.dumps({"label": a.label, "banked": False, "why": f"scp rc={r.returncode} {r.stderr[-300:]}"}))
            return 2
    tc = a.thor_counts or os.path.join(local, f"{a.label}.counts.json")
    try:
        cnt = json.load(open(tc, encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(json.dumps({"label": a.label, "banked": False, "why": f"no Thor counts.json: {e!r}"}))
        return 2
    seam_sha = cnt.get("seam_sha256")
    tj = os.path.join(local, f"{a.label}.thor.json")
    if not seam_sha and os.path.exists(tj):
        seam_sha = json.load(open(tj, encoding="utf-8")).get("seam_sha256")
    prov = {"backend": "thor", "host": "tanitad-thor-wifi (thor6, linux-aarch64, glibc 2.39)",
            "label": a.label, "arm": a.arm, "split": a.split, "thor_status": cnt.get("status"),
            "thor_failures": cnt.get("failures"), "seam_sha256": seam_sha,
            "policy": "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-04-navsim-thor-backend/RULING_BACKEND_POLICY.md",
            "admission_record": "…/2026-10-04-navsim-thor-backend/raw/THOR_BACKEND_VALIDATED.json (ADMITTED_UNDER_POLICY)",
            "banked_local": time.strftime("%Y-%m-%d %H:%M:%S"),
            "note": "Thor scores are early reads + cross-backend checks; the dev-box milestone owns the standard names."}
    out = {"label": a.label, "banked": True, "thor_status": cnt.get("status")}
    if a.devbox_counts and os.path.exists(a.devbox_counts):
        dc = json.load(open(a.devbox_counts, encoding="utf-8"))
        prov["devbox_status"] = dc.get("status")
        if dc.get("status") == "PASS":
            pairs = []
            if a.split == "navtest":
                pairs.append(("csv", a.devbox_csv or a.devbox_counts.replace(".counts.json", ".csv"),
                              a.thor_csv or os.path.join(local, f"{a.label}.csv")))
            else:
                pairs.append(("csv", a.devbox_csv, a.thor_csv))
                pairs.append(("frame", a.devbox_frame, a.thor_frame))
            res = {}
            ok = True
            for kind, ref, got in pairs:
                o = os.path.join(local, f"{a.label}.cross_backend_{kind}.json")
                subprocess.run([sys.executable, os.path.join(HERE, "full_split_compare.py"), "--ref", ref, "--got", got,
                                "--out", o], check=True, capture_output=True)
                d = json.load(open(o, encoding="utf-8"))
                chk = {"discrete_flips_0": d["total_discrete_flips"] == 0 and not d["failures"],
                       "per_token_abs_le_1e-9": (d["max_abs_diff_any_column"] or 0) <= 1e-9,
                       "split_mean_abs_le_1e-12": (d["max_abs_mean_delta_any_column"] or 0) <= 1e-12}
                ok &= all(chk.values())
                res[kind] = {"ref": ref, "got": got, "n_tokens": d["n_tokens_compared"],
                             "n_tokens_differ": d["n_tokens_any_column_differs"],
                             "discrete_flips": d["total_discrete_flips"], "max_abs": d["max_abs_diff_any_column"],
                             "max_abs_split_mean_delta": d["max_abs_mean_delta_any_column"], "bit_exact": d["bit_exact"],
                             "policy_checks": chk, "failures": d["failures"]}
            xb = {"label": a.label, "arm": a.arm, "split": a.split, "verdict": "ADMISSIBLE" if ok else "NOT_ADMISSIBLE",
                  "policy": prov["policy"], "comparisons": res}
            json.dump(xb, open(os.path.join(local, f"{a.label}.cross_backend.json"), "w", encoding="utf-8"), indent=1)
            out["cross_backend"] = xb["verdict"]
            prov["cross_backend"] = xb["verdict"]
    json.dump(prov, open(os.path.join(local, f"{a.label}.thor"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
