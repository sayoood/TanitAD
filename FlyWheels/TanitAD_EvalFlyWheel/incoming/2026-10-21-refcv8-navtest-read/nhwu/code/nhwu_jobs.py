#!/usr/bin/env python3
"""nhwu_jobs.py -- build thor_pool jobs for refcv8 warmup (N3) / navhard (N2) scoring on the Thor CPU backend.
(navsim-cpu venv python: needs numpy.)  EvalFlyWheel, 2026-10-21.

REUSES the 2026-10-04 Thor backend recipe verbatim (RESULT.md "Job argv"): one ``score_arm_thor.py`` job per
warmup arm (``--split warmup_two_stage``), and 6 whole-group shard jobs per navhard arm
(``--split navhard_two_stage --tokens navhard_shard_{k}of6.json``, ``tanitad_seam_agent_ulp`` +
``+agent.ulp_reference=ulp_reference_navhard_devbox.json +agent.atol=1e-9``; arm name ``<ARM>_THOR`` so
``finalize_navhard_arm.py`` finds the shards).  Nothing here computes a score.
WARMUP uses the SAME accepted fallback (RULING_BACKEND_POLICY.md item 3: "applies only when a dev-box reference re-hashes
to the seam fingerprint and Thor's values are within 1e-9 of it") with its own dev-box reference
``ulp_reference_warmup_devbox.json`` (make_ulp_reference.py lite mode, navsim-crun venv, 204/204 synthetic tokens re-hash to
the seam fingerprint).  MEASURED 2026-10-21 (control run 1, E2's byte agent): 19/204 warmup synthetic tokens are
FINGERPRINT_MISMATCH on Thor -> no CSV; the byte agent cannot score warmup on Thor.

Gates (every one asserts on an ARTIFACT; any failure -> verdict FAIL and NO jobs file):
  * the seam's bridge manifest (``seam_<ARM>.manifest.json`` beside it): partial False, n_missing 0,
    n_rows == n_expected == the split's token count (unless --smoke), navhard n_cv_standin_rows == 0;
    warmup n_cv_standin_rows == the npz's cv_standin count (declared stage-1 CV stand-ins);
  * the npz: unique tokens, token count == manifest n_rows; the npz carries a fingerprint for EVERY token
    and it EQUALS refcv7's banked seam fingerprint for that token (same dev-box AgentInput export -> the
    ulp reference re-hashes to it); a token absent from the refcv7 seam is a failure;
  * navhard: the union of the 6 shard files' expected_tokens == the seam's token set (full run) and is
    disjoint across shards; warmup: the seam's token set == refcv7's warmup seam token set (full run);
  * the seam is copied to the tmpfs seam dir and its sha256 re-read there == the source sha256.
Writes ``--jobs-out`` (only on PASS) and ``--report`` (always) with ``verdict``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys

import numpy as np

PY = "/home/nvidia/venvs/navsim-cpu/bin/python"
DRV = "/home/nvidia/navsim-thor/code/thor/score_arm_thor.py"
SHM = "/dev/shm/navsim"
ULP_REFS = {"navhard": f"{SHM}/subsets/ulp_reference_navhard_devbox.json",
            "warmup": f"{SHM}/subsets/ulp_reference_warmup_devbox.json"}
ULP = ["--agent-target", "tanitad_seam_agent_ulp.TanitADSeamAgentULP"]
SHARD = f"{SHM}/subsets/navhard_shard_{{k}}of6.json"
SPLITS = {"warmup": ("warmup_two_stage", 220), "navhard": ("navhard_two_stage", 5912)}
R7SEAM = {"warmup": f"{SHM}/seams/step50400/seam_R7_A1__warmup.npz",
          "navhard": f"{SHM}/seams/step50400/seam_R7_A1__navhard.npz"}


def sha256(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=sorted(SPLITS))
    ap.add_argument("--arm", action="append", required=True, metavar="ARM=SEAM_NPZ")
    ap.add_argument("--seam-dir", required=True, help="tmpfs dir the seams are copied into")
    ap.add_argument("--out-root", required=True, help="score output root (warmup: flat; navhard: <root>/_shards/...)")
    ap.add_argument("--exp-tag", required=True)
    ap.add_argument("--jobs-out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--smoke", action="store_true", help="VALIDATION ONLY: allow a --limit seam (n_rows < split size)")
    ap.add_argument("--no-manifest", action="store_true", help="control seams (refcv7) have no refcv8 manifest")
    ap.add_argument("--subset", default="", help="control: ONE token-subset json instead of the 6 navhard shards")
    ap.add_argument("--keep-arm-name", action="store_true", help="control: navhard tag <ARM> instead of <ARM>_THOR")
    a = ap.parse_args()
    split, n_split = SPLITS[a.split]
    bad, arms_rep, jobs = [], {}, []
    r7 = np.load(R7SEAM[a.split], allow_pickle=False)
    r7fp = {str(t): str(f) for t, f in zip(r7["token"], r7["fingerprint"])}
    shard_tokens = []
    if a.split == "navhard" and not a.subset:
        for k in range(6):
            shard_tokens.append(json.load(open(SHARD.format(k=k), encoding="utf-8"))["expected_tokens"])
        flat = [t for s in shard_tokens for t in s]
        if len(flat) != len(set(flat)) or len(flat) != n_split:
            bad.append(f"shard files: {len(flat)} rows / {len(set(flat))} unique != {n_split}")
    os.makedirs(a.seam_dir, exist_ok=True)
    for spec in a.arm:
        arm, seam = spec.split("=", 1)
        rep = {"seam": seam}
        arms_rep[arm] = rep
        if not os.path.isfile(seam):
            bad.append(f"{arm}: seam missing {seam}")
            continue
        z = np.load(seam, allow_pickle=False)
        toks = [str(t) for t in z["token"]]
        src = [str(s) for s in z["source"]]
        rep.update(n_tokens=len(toks), n_unique=len(set(toks)), sources={s: src.count(s) for s in sorted(set(src))})
        if len(toks) != len(set(toks)):
            bad.append(f"{arm}: duplicate tokens in the seam")
        if not a.no_manifest:
            mp = seam[:-4] + ".manifest.json"
            if not os.path.isfile(mp):
                bad.append(f"{arm}: no manifest {mp}")
            else:
                m = json.load(open(mp, encoding="utf-8"))
                mf = {k: m.get(k) for k in ("n_expected", "n_rows", "n_missing", "partial", "n_cv_standin_rows", "n_model_rows")}
                mf["KPR"] = (m.get("KPR") or {}).get("verdict")
                mf["split"] = m.get("split")
                rep["manifest"] = mf
                if m.get("partial") is not False or m.get("n_missing") != 0 or m.get("n_rows") != len(toks):
                    bad.append(f"{arm}: manifest {mf} vs npz rows {len(toks)}")
                if not a.smoke and (m.get("n_rows") != n_split or m.get("n_expected") != n_split):
                    bad.append(f"{arm}: n_rows/n_expected {m.get('n_rows')}/{m.get('n_expected')} != {n_split}")
                if m.get("n_cv_standin_rows") != src.count("cv_standin"):
                    bad.append(f"{arm}: manifest cv_standin {m.get('n_cv_standin_rows')} != npz {src.count('cv_standin')}")
                if a.split == "navhard" and src.count("cv_standin"):
                    bad.append(f"{arm}: navhard seam carries {src.count('cv_standin')} cv_standin rows")
        fps = [str(f) for f in z["fingerprint"]]
        miss = [t for t in toks if t not in r7fp]
        diff = [t for t, f in zip(toks, fps) if t in r7fp and r7fp[t] != f]
        empty = sum(1 for f in fps if not f)
        rep.update(fp_not_in_refcv7_seam=len(miss), fp_differs_from_refcv7=len(diff), fp_empty=empty)
        if miss or diff or empty:
            bad.append(f"{arm}: fingerprints vs refcv7 seam: {len(miss)} absent, {len(diff)} differ, {empty} empty")
        if not a.smoke and not a.subset:
            want = set(t for s in shard_tokens for t in s) if a.split == "navhard" else set(r7fp)
            if set(toks) != want:
                bad.append(f"{arm}: seam token set != expected ({len(set(toks) ^ want)} differ)")
        dst = os.path.join(a.seam_dir, f"seam_{arm}__{a.split}.npz")
        shutil.copyfile(seam, dst)
        s_src, s_dst = sha256(seam), sha256(dst)
        rep.update(seam_sha256=s_src, staged=dst, staged_sha256=s_dst)
        if s_src != s_dst or len(s_src) != 64:
            bad.append(f"{arm}: staged sha256 {s_dst} != source {s_src}")
        if a.split == "warmup":
            jobs.append({"name": f"wu_{arm}", "cwd": SHM, "log": os.path.join(a.out_root, f"{arm}.driver.txt"),
                         "argv": [PY, DRV, "--arm", arm, "--seam", dst, "--split", split, "--out", a.out_root,
                                  "--exp-tag", a.exp_tag] + ULP +
                                 ["--extra-override", f"+agent.ulp_reference={ULP_REFS['warmup']}",
                                  "--extra-override", "+agent.atol=1e-9"]})
        else:
            ks = [None] if a.subset else list(range(6))
            for k in ks:
                lab = arm if a.keep_arm_name else f"{arm}_THOR"
                tok = a.subset if a.subset else SHARD.format(k=k)
                sub = lab if a.subset else f"{lab}_k{k}of6"
                od = os.path.join(a.out_root, sub) if a.subset else os.path.join(a.out_root, "_shards", sub)
                jobs.append({"name": f"nh_{sub}", "cwd": SHM, "log": od + ".driver.txt",
                             "argv": [PY, DRV, "--arm", lab, "--seam", dst, "--split", split, "--out", od,
                                      "--exp-tag", f"{a.exp_tag}k{k}" if k is not None else a.exp_tag, "--tokens", tok,
                                      ] + ULP + [
                                      "--extra-override", f"+agent.ulp_reference={ULP_REFS['navhard']}",
                                      "--extra-override", "+agent.atol=1e-9"]})
    for p in [ULP_REFS[a.split], DRV]:
        if not os.path.isfile(p):
            bad.append(f"missing {p}")
    verdict = "FAIL" if bad else "PASS"
    rep = {"verdict": verdict, "split": split, "bad": bad, "arms": arms_rep, "n_jobs": len(jobs) if not bad else 0,
           "smoke_VALIDATION_ONLY": bool(a.smoke), "subset": a.subset or None, "exp_tag": a.exp_tag,
           "out_root": a.out_root, "refcv7_fp_reference": R7SEAM[a.split]}
    if not bad:
        os.makedirs(os.path.dirname(os.path.abspath(a.jobs_out)), exist_ok=True)
        with open(a.jobs_out, "w", encoding="utf-8") as f:
            for j in jobs:
                f.write(json.dumps(j) + "\n")
        rep["jobs_out"] = a.jobs_out
    json.dump(rep, open(a.report, "w", encoding="utf-8"), indent=1)
    print(json.dumps({"verdict": verdict, "n_jobs": rep["n_jobs"], "bad": bad[:5]}))
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
