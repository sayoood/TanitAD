#!/usr/bin/env python3
"""mutation_speedsub_wrapper.py -- the mutation pass over the value-emitting lines of ``stack/scripts/v8f_t1_speedsub.py`` (PREREG_V8F_SPEEDSUB_T1 section 4 / the build brief).

Each mutant is a copy of the wrapper with ONE line changed; ``test_v8f_t1_speedsub.py`` is run against it (``SPEEDSUB_WRAPPER=<mutant>``) as a real pytest subprocess. A mutant is KILLED iff pytest
exits non-zero (a pytest FAILURE); the failing test ids are recorded. A SURVIVOR is a finding. The unmutated wrapper must pass first (the baseline), or the pass proves nothing.

    python mutation_speedsub_wrapper.py --tree <repo root with stack/, taniteval/> --work <scratch dir> --out mutation.json [--workers 2] [--only NAME ...]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time

#: (name, the exact text in the wrapper, its replacement). Each `old` must occur exactly once.
MUTANTS = [
    ("idx_last_cell_off", "SPEED_CODE_IDX = (46, 56, 57, 58, 59)", "SPEED_CODE_IDX = (46, 56, 57, 58, 58)"),
    ("idx_wider_than_declared", "ctx, sr, idx = self.ctx, self.sr, list(self.idx)", "ctx, sr, idx = self.ctx, self.sr, list(self.idx)[:4] + [45]"),
    ("identity_returns_changed_code", "return code                                                  # the ORIGINAL tensor", "return code * 1.0001                                         # the ORIGINAL tensor"),
    ("substitution_not_returned", "        ctx[\"used_std\"] = new[:, idx].detach().float().cpu().numpy()\n        return new", "        ctx[\"used_std\"] = new[:, idx].detach().float().cpu().numpy()\n        return code"),
    ("sd_wrong_in_standardisation", "std = sr.phys_to_std(self.readout.predict_ego_batch(ctx[\"batch\"]), mu, sd)", "std = sr.phys_to_std(self.readout.predict_ego_batch(ctx[\"batch\"]), mu, sd * 2.0)"),
    ("mu_dropped_in_standardisation", "std = sr.phys_to_std(self.readout.predict_ego_batch(ctx[\"batch\"]), mu, sd)", "std = sr.phys_to_std(self.readout.predict_ego_batch(ctx[\"batch\"]), 0.0, sd)"),
    ("readout_reads_zero_ego", "ctx[\"ego\"] = sr.ego_vectors(ctx[\"batch\"])", "ctx[\"ego\"] = sr.ego_vectors(ctx[\"batch\"]) * 0.0"),
    ("shuffle_seed", "SHUF_SEED, SHUF_N = 20261029, 2399", "SHUF_SEED, SHUF_N = 20261030, 2399"),
    ("shuffle_length", "SHUF_SEED, SHUF_N = 20261029, 2399", "SHUF_SEED, SHUF_N = 20261029, 2400"),
    ("shuffle_tick_off_by_one", "std = self.src[ctx[\"src_win\"], ctx[\"tick\"]].astype(np.float64)", "std = self.src[ctx[\"src_win\"], (ctx[\"tick\"] + 1) % self.n_ticks].astype(np.float64)"),
    ("shuffle_not_applied", "ctx[\"src_win\"] = self.perm[ctx[\"win\"]]", "ctx[\"src_win\"] = ctx[\"win\"]"),
    ("perm_sha_dtype", "return hashlib.sha256(np.asarray(perm).astype(np.int64).tobytes()).hexdigest()", "return hashlib.sha256(np.asarray(perm).astype(np.int32).tobytes()).hexdigest()"),
    ("window_base_not_advanced", "                self.base += self.chunk_b", "                self.base += 0"),
    ("tick_counter", "tick = self.k % self.n_ticks", "tick = (self.k + 1) % self.n_ticks"),
    ("o_cells_taken_from_t", "o_std = own[:, idx].float().cpu().numpy(), o[:, idx].float().cpu().numpy()", "o_std = own[:, idx].float().cpu().numpy(), own[:, idx].float().cpu().numpy()"),
    ("o_code_recorder_returns_t_code", "hook.ctx[\"o_code\"] = code.detach().clone()", "hook.ctx[\"o_code\"] = hook.ctx[\"own_code\"]"),
    ("nonspeed_check_blind", "non_dev = (o[:, self.non] - own[:, self.non]).abs().amax(1).float().cpu().numpy()", "non_dev = (own[:, self.non] - own[:, self.non]).abs().amax(1).float().cpu().numpy()"),
    ("cons_mu_from_sd", "self.mu, self.sd = model.cons_mu[ci].double().cpu().numpy(), model.cons_sd[ci].double().cpu().numpy()", "self.mu, self.sd = model.cons_sd[ci].double().cpu().numpy(), model.cons_mu[ci].double().cpu().numpy()"),
    ("readout_key_egoc", "READOUT_KEY = {\"egoc\": \"mlp_ego_c\"", "READOUT_KEY = {\"egoc\": \"mlp_ego\""),
    ("lane_identity", "MODE_LIMIT = {\"identity\": \"causal\"", "MODE_LIMIT = {\"identity\": \"logged\""),
    ("cut_not_refused", "== \"cut\":", "== \"never\":"),
    ("run_o_not_restored", "V.forward_infer, V._code_pred, V.run_o = self.orig[\"forward_infer\"], self.orig[\"_code_pred\"], self.orig[\"run_o\"]", "V.forward_infer, V._code_pred = self.orig[\"forward_infer\"], self.orig[\"_code_pred\"]"),
    ("alignment_always_ok", "ok = all(det[k] for k in (\"win_equal\", \"tick_equal\", \"age_t_equal\", \"v0_equal\"))", "ok = True"),
    ("v0_record_wrong", "r[\"v0_last\"].append(ctx[\"batch\"][\"p10f\"][:, -1, 0].double().cpu().numpy())", "r[\"v0_last\"].append(ctx[\"batch\"][\"p10f\"][:, -1, 0].double().cpu().numpy() + 1e-9)"),
    ("used_recorded_as_own", "used = ctx.get(\"used_std\", own_std)", "used = own_std"),
    ("ckpt_sha_unchecked", "if wa.expect_ckpt_sha and ckpt_sha != wa.expect_ckpt_sha:", "if False:"),
    ("readout_sha_unchecked", "if not wa.readout_sha or readout_sha != wa.readout_sha:", "if False:"),
    ("egoshufc_source_unchecked", "if src_meta.get(\"mode\") != \"egoc\" or src_meta.get(\"ckpt_sha256\") != ckpt_sha or src_meta.get(\"readout_sha256\") != readout_sha:", "if False:"),
    ("hook_without_code_silent", "if ctx is None or \"own_code\" not in ctx or \"o_code\" not in ctx:", "if ctx is None:"),
]


def apply(src: str, old: str, new: str) -> str:
    if src.count(old) != 1:
        raise SystemExit(f"mutant anchor occurs {src.count(old)} times (need exactly 1): {old[:90]!r}")
    return src.replace(old, new, 1)


def run_pytest(tree, wrapper, work, name, timeout):
    env = dict(os.environ, SPEEDSUB_WRAPPER=wrapper, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=os.pathsep.join([f"{tree}/stack", f"{tree}/taniteval", f"{tree}/stack/scripts"]))
    base = os.path.join(work, "pytmp", name)
    os.makedirs(os.path.dirname(base), exist_ok=True)
    t0 = time.time()
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "--basetemp", base, f"{tree}/stack/tests/test_v8f_t1_speedsub.py"], cwd=tree, env=env,
                       capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    failed = [ln.split(" - ")[0].replace("FAILED ", "").replace("ERROR ", "").strip() for ln in p.stdout.splitlines() if ln.startswith(("FAILED", "ERROR"))]
    return {"rc": p.returncode, "failed": failed, "seconds": round(time.time() - t0, 1), "tail": p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args(argv)
    wsrc = open(f"{a.tree}/stack/scripts/v8f_t1_speedsub.py", encoding="utf-8").read()
    os.makedirs(a.work, exist_ok=True)
    jobs = {}
    for name, old, new in MUTANTS:
        if a.only and name not in a.only:
            continue
        d = os.path.join(a.work, "mutants", name)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "v8f_t1_speedsub.py")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(apply(wsrc, old, new))
        jobs[name] = path
    res = {"baseline": run_pytest(a.tree, f"{a.tree}/stack/scripts/v8f_t1_speedsub.py", a.work, "baseline", a.timeout), "mutants": {}}
    print("baseline:", res["baseline"], flush=True)
    if res["baseline"]["rc"] != 0:
        json.dump(res, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
        print("ZZ-MUTATION-STOP baseline does not pass")
        return 2
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_pytest, a.tree, p, a.work, n, a.timeout): n for n, p in jobs.items()}
        for f in cf.as_completed(futs):
            n = futs[f]
            r = f.result()
            r["killed"] = r["rc"] != 0
            res["mutants"][n] = r
            print(f"{'KILLED  ' if r['killed'] else 'SURVIVED'} {n}: {r['failed'][:2]} ({r['seconds']}s)", flush=True)
    res["n_mutants"], res["n_killed"] = len(res["mutants"]), sum(r["killed"] for r in res["mutants"].values())
    res["survivors"] = sorted(n for n, r in res["mutants"].items() if not r["killed"])
    json.dump(res, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    print(f"ZZ-MUTATION-DONE {res['n_killed']}/{res['n_mutants']} killed; survivors={res['survivors']}", flush=True)
    return 0 if not res["survivors"] else 1


if __name__ == "__main__":
    sys.exit(main())
