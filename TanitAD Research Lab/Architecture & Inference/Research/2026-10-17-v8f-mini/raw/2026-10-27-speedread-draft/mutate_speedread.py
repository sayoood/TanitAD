#!/usr/bin/env python3
"""Mutation check for v8f_mini_eval_speedread.py (PREREG_V8F_SPEEDREAD, DRAFT): each mutant deletes or breaks ONE line that emits a value, and the self-test must go RED.

A mutant is KILLED when pytest exits non-zero on its target tests; SURVIVED = exit 0 (a gap in the tests); BAD = the anchor text is not found exactly once (the mutant was not applied).
The baseline (no mutation) must pass first. CPU only. Usage: python mutate_speedread.py --module <instrument.py> --tests <test file> --tmp <dir> --out <json>
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

MUTANTS = [
    ("M01 substitute: the write of the new cells deleted", "    out[:, list(idx)] = torch.as_tensor(np.asarray(new_std), dtype=own_code.dtype, device=own_code.device)\n", "", "substitute or hook_hands"),
    ("M02 new_speed_std: the read-out's value replaced by T's own", "    return phys_to_std(readout.predict_batch(batch), mu, sd)\n", "    return _np(own_code[:, list(idx)])\n", "hook_hands"),
    ("M03 hook: O handed T's own code instead of the new one", "                return self.orig(model, batch, new)\n", "                return self.orig(model, batch, own)\n", "hook"),
    ("M04 hook: the tick-mirror check deleted", "                    self.mirror_ok = bool(np.array_equal(_np(pad), np.repeat(~ok, pad.shape[1] // ok.shape[1], axis=1)))\n", "", "hook"),
    ("M05 tick slice: a_t ignored", "    last = (Fn - 1) - a_t\n", "    last = (Fn - 1) - 0 * a_t\n", "tick_slice or no_frame"),
    ("M06 tick slice: p10f read at NOW, not at T's tick", '"p10f": _np(batch["p10f"])[b, last]', '"p10f": _np(batch["p10f"])[b, -1 + 0 * last]', "tick_slice"),
    ("M07 scene: the padded-frame mask dropped", "    x = tok.astype(np.float32) * ok[:, :, None, None].astype(np.float32)\n", "    x = tok.astype(np.float32)\n", "padded or tick_slice"),
    ("M08 ridge: the intercept dropped", "    return w, ym - float(xm @ w)\n", "    return w, 0.0\n", "planted"),
    ("M09 read-out: the linear bias dropped", "            return X.astype(np.float64) @ self.lin[0] + self.lin[1]\n", "            return X.astype(np.float64) @ self.lin[0]\n", "planted"),
    ("M10 selection inverted", '    return "mlp" if float(np.mean(ho_mlp_sd)) < float(np.mean(ho_lin_sd)) else "lin"\n', '    return "lin" if float(np.mean(ho_mlp_sd)) < float(np.mean(ho_lin_sd)) else "mlp"\n', "nonlinear or selection"),
    ("M11 permutation removed", "        Yp[i, c] = Y[rng.permutation(i), c]\n", "        Yp[i, c] = Y[i, c]\n", "permute or perm"),
    ("M12 skill: the train-mean base replaced by the eval mean", "    B = np.broadcast_to(np.asarray(base, np.float64), Y.shape)\n", "    B = np.broadcast_to(np.nanmean(np.where(M, Y, np.nan), 0), Y.shape)\n", "const"),
    ("M13 decide: the INFO IN INPUT row deleted", '        return "INFO IN INPUT"\n', "        pass\n", "decide"),
    ("M14 separated-better reads lo, not hi", '    return r["hi"] < 0.0\n', '    return r["lo"] < 0.0\n', "decide"),
    ("M15 C1 loosened", '    c["C1_identity_eq_own"] = float(np.max(np.abs(a["identity"]["ade6"] - a["own"]["ade6"]))) == 0.0', '    c["C1_identity_eq_own"] = float(np.max(np.abs(a["identity"]["ade6"] - a["own"]["ade6"]))) <= 1.0', "controls"),
    ("M16 C-PATH deleted", '            c["C7_path_read_eq_identity_pred"] = float(np.max(np.abs(a["read"]["cells"]["used_std"] - want))) <= 1e-5\n', "", "controls"),
    ("M17 the read/alt aliases not restored on load", "    for a, kind in st[\"alias\"].items():\n        ro[a] = ro[kind]\n", "    for a, kind in st[\"alias\"].items():\n        ro[a] = ro[\"const\"]\n", "roundtrip"),
    ("M18 folds not by clip", "    return np.array([int(hashlib.sha256(str(c).encode(\"utf-8\")).hexdigest()[:8], 16) % N_FOLDS for c in clips], np.int64)\n",
     "    return np.arange(len(clips), dtype=np.int64) % N_FOLDS\n", "folds"),
    ("M19 the perm control computed from the REAL read-out", '    rec["perm_holdout_skill_max"] = float(np.nanmax(skill(Pp, Y[ho], M[ho], const)))\n',
     '    rec["perm_holdout_skill_max"] = float(np.nanmax(skill(Xz[ho] @ outs["lin"]["lin"][0] + outs["lin"]["lin"][1], Y[ho], M[ho], const)))\n', "const_reads"),
]


def run_tests(module, tests, tmp, k, tag):
    env = dict(os.environ, SPEEDREAD_MODULE=module, CUDA_VISIBLE_DEVICES="-1", TMP=tmp, TEMP=tmp, OMP_NUM_THREADS="4")
    cmd = [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "--basetemp", os.path.join(tmp, "bt_" + tag), tests]
    if k:
        cmd += ["-k", k]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=200)
    return p.returncode, (p.stdout + p.stderr).strip().splitlines()[-1:] if (p.stdout + p.stderr).strip() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--tests", required=True)
    ap.add_argument("--tmp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    os.makedirs(a.tmp, exist_ok=True)
    src = open(a.module, encoding="utf-8").read()
    rc, tail = run_tests(a.module, a.tests, a.tmp, None, "baseline")
    res = {"baseline": {"rc": rc, "tail": tail}, "mutants": []}
    if rc != 0:
        print("BASELINE FAILS", tail)
        json.dump(res, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
        return 2
    for i, (name, old, new, k) in enumerate(MUTANTS):
        if a.only and not name.startswith(a.only):
            continue
        n = src.count(old)
        if n != 1:
            res["mutants"].append({"name": name, "status": "BAD", "anchor_count": n})
            print(f"BAD      {name} (anchor found {n} times)")
            continue
        mp = os.path.join(a.tmp, f"mutant_{i:02d}.py")
        open(mp, "w", encoding="utf-8", newline="\n").write(src.replace(old, new))
        rc, tail = run_tests(mp, a.tests, a.tmp, k, f"m{i:02d}")
        st = "KILLED" if rc != 0 else "SURVIVED"
        res["mutants"].append({"name": name, "status": st, "rc": rc, "tail": tail, "k": k})
        print(f"{st:8s} {name}  {tail}", flush=True)
    res["summary"] = {s: sum(m["status"] == s for m in res["mutants"]) for s in ("KILLED", "SURVIVED", "BAD")}
    rc, tail = run_tests(a.module, a.tests, a.tmp, None, "baseline_after")
    res["baseline_after"] = {"rc": rc, "tail": tail}
    json.dump(res, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    print("SUMMARY", res["summary"], "baseline after rc", rc)
    return 0 if res["summary"]["SURVIVED"] == 0 and res["summary"]["BAD"] == 0 and rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
