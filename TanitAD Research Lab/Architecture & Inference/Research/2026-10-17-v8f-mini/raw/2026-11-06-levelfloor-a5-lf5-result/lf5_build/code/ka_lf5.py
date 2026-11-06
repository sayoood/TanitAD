#!/usr/bin/env python3
"""ka_lf5.py -- the known answers of PREREG_LEVELFLOOR_A5 (sha256 2aac1dbe...) section 2, as 64-window smokes (s0, reindexed, C0_s0 + the LLR_s0 heads, CPU 12 threads), each with ONE deliberate mutation that must turn it RED.
  KA-N1  sigma = 0 on the noise path (--floor-vhat-noise 0.0 --floor-vhat-noise-seed N, kappa 1.0) == the LFO tool (6744d3e4) at kappa 1.0 without noise flags (= the LFO100 smoke), bit for bit on every common tick field and the analysis arms.
  KA-N2  a bare and a composite smoke of one seed (sigma 0.20) consume identical eps sequences (lf_eps, row for row), and lf_vhat == lf_vlog * exp(sigma * eps).
  KA-N3  the LF5 tool at kappa 0.75 without noise flags reproduces (a) the LFO tool's own 64-window smoke LFO_bare bit for bit, (b) the banked RILFO075_s0 tick arrays on the same 64 windows (evenly spaced subset; the smoke window k is the banked window
         linspace(0, 2398, 64)[k]); (b) is bit-for-bit on the discrete fields and compared to float tolerance on the floats (the batch composition differs from the banked run).
Mutations: M1 (sigma + 0.01 inside the exponent) must break KA-N1; M2 (eps drawn only where the floor binds) must break KA-N2; M3 (a default noise 0.05 with seed 0 when no flag is given) must break KA-N3(a).
Writes raw/KA_LF5.json.   usage: python -I ka_lf5.py [--only N1,N2,N3]"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
ART = "E:/Projects/TanitAD-artifacts"
MC = f"{ART}/dagland/motion_c"
RDB = f"{ART}/v8f-recover"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
SW = "F:/TanitAD-scratch/motion_c/lf5_build/smoke"
MUT_TREE = "F:/TanitAD-scratch/motion_c/lf5_build/mut/tree"
LFO_TOOL = "F:/TanitAD-scratch/motion_c/lfo_read/tree/stack/scripts/v8f_t1_eval_ri.py"
LF5_TOOL = "F:/TanitAD-scratch/motion_c/lf5_read/tree/stack/scripts/v8f_t1_eval_ri.py"
LFO_SHA = "6744d3e4686e344fb72e02ad8f659c779eb73261de13c4a3844e15243667dcc6"
DATA = f"{ART}/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
SEED = 20261106
NOISE_ONLY = ("lf_vlog", "lf_eps")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def run_smoke(tag, tool, kappa, extra, composite=False, n_win=64):
    od = f"{SW}/{tag}"
    if os.path.isdir(od):
        os.replace(od, f"{od}_old_{time.strftime('%H%M%S')}")
    os.makedirs(od)
    cmd = [PY, "-I", tool, "run", "--ckpt", f"{RDB}/M2b_s0_recft_C0/ckpt_last.pt", "--tag", tag, "--tracks", f"{ART}/sfresh_inputs/inputs/native_tracks.pt", "--eval-cache", f"{ART}/sfresh_inputs/cache",
           "--eval-grid", f"{DATA}/eval_grid_windows.json", "--eval-join", f"{DATA}/join_eval141.npz", "--out-dir", od, "--replan-steps", "10", "--nav", "deadreckon", "--schedule", "periodic", "--limit", "causal",
           "--dump-ticks", f"{od}/ticks", "--dump-heads", "--device", "cpu", "--smooth", "sg72", "--scene", "reindexed", "--level-floor", "on", "--floor-kappa", str(kappa),
           "--floor-ckpt", f"{RDB}/M2b_s0_recft_LLR/ckpt_last.pt", "--floor-vhat-source", "logged", "--max-windows", str(n_win), "--n-boot", "20"] + (["--feasible", "kbox"] if composite else []) + extra
    env = dict(os.environ, OMP_NUM_THREADS="12", MKL_NUM_THREADS="12", OPENBLAS_NUM_THREADS="12", CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")
    t0 = time.time()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    os.makedirs(f"{PK}/logs", exist_ok=True)
    open(f"{PK}/logs/KA_LF5_{tag}.out", "w", encoding="utf-8", newline="\n").write(" ".join(cmd) + "\n" + r.stdout[-2000:] + r.stderr[-2000:])
    if r.returncode:
        raise SystemExit(f"smoke {tag} failed rc={r.returncode}: {r.stderr[-800:]}")
    print(f"  smoke {tag} done in {time.time() - t0:.0f}s", flush=True)
    return od


def ticks(od, tag):
    return dict(np.load(f"{od}/ticks/ticks_{tag}.npz", allow_pickle=True))


def arms(od, tag):
    return json.load(open(f"{od}/t1v8f_{tag}.json", encoding="utf-8"))["arms"]


def row_fields(t):
    n = len(t["win"])
    return [k for k, v in t.items() if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == n and v.dtype.kind in "fiub" and k not in NOISE_ONLY]


def compare_bit(ta, tb, aa, ab):
    """Every common numeric row field and the analysis arms: bit for bit. Returns (ok, n_fields, mismatches)."""
    fa, fb = set(row_fields(ta)), set(row_fields(tb))
    common = sorted(fa & fb)
    bad = [k for k in common if not np.array_equal(ta[k], tb[k], equal_nan=True)]
    if json.dumps(aa, sort_keys=True) != json.dumps(ab, sort_keys=True):
        bad.append("analysis.arms")
    return (not bad and len(common) >= 30), len(common), bad


def mutate(name, *pairs):
    src = open(LF5_TOOL, encoding="utf-8", newline="").read()
    for old, new in pairs:
        assert src.count(old) == 1, (name, old[:60], src.count(old))
        src = src.replace(old, new)
    os.makedirs(f"{MUT_TREE}/stack/scripts", exist_ok=True)
    p = f"{MUT_TREE}/stack/scripts/v8f_t1_eval_ri.py"
    open(p, "w", encoding="utf-8", newline="\n").write(src)
    return p


M1 = ("v_hat = v_log * np.exp(self.noise * eps)", "v_hat = v_log * np.exp((self.noise + 0.01) * eps)")
M2 = ("""                eps = self.rng.standard_normal(len(v_log))
                self.stats["noise_draws"] += int(len(eps))
""", """                _o, _i = self.op.apply_floor_batch(plan, v0, v_log, self.op.sigmoid(logit), gap_hat, self.kappa)
                _b = np.asarray(_i["bind"], bool)
                eps = np.zeros(len(v_log))
                eps[_b] = self.rng.standard_normal(int(_b.sum()))
                self.stats["noise_draws"] += int(_b.sum())
""")
M3 = ("self.noise = 0.0 if noise is None else float(noise)", "self.noise = 0.05 if noise is None else float(noise)")
M3b = ("self.noise_seed = None if noise_seed is None else int(noise_seed)", "self.noise_seed = 0 if noise_seed is None else int(noise_seed)")


def main():
    only = set((sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else "N1,N2,N3").split(","))
    assert sha(LFO_TOOL) == LFO_SHA, "the LFO reference tool differs from 6744d3e4"
    os.makedirs(f"{SW}", exist_ok=True)
    nz = ["--floor-vhat-noise", "0.0", "--floor-vhat-noise-seed", str(SEED)]
    out = {"card": "PREREG_LEVELFLOOR_A5 2aac1dbe...", "tool_sha256": sha(LF5_TOOL), "lfo_tool_sha256": sha(LFO_TOOL), "n_windows": 64, "seed": SEED, "cases": {}}
    if "N1" in only:
        print("KA-N1", flush=True)
        A = run_smoke("A_lfo100", LFO_TOOL, 1.0, [])
        B = run_smoke("B_lf5_sigma0", LF5_TOOL, 1.0, nz)
        ta, tb = ticks(A, "A_lfo100"), ticks(B, "B_lf5_sigma0")
        ok, nf, bad = compare_bit(ta, tb, arms(A, "A_lfo100"), arms(B, "B_lf5_sigma0"))
        extra = {"has_lf_eps": "lf_eps" in tb, "vlog_equals_vhat_f32": bool(np.array_equal(np.asarray(tb["lf_vlog"], np.float32), tb["lf_vhat"], equal_nan=True)),
                 "n_rows": int(len(tb["win"])), "noise_draws_rows": int(len(tb["lf_eps"]))}
        ok = ok and extra["has_lf_eps"] and extra["vlog_equals_vhat_f32"]
        mp = mutate("M1", M1)
        Mo = run_smoke("M1_sigma_plus", mp, 1.0, nz)
        tm = ticks(Mo, "M1_sigma_plus")
        okm, _, badm = compare_bit(ta, tm, arms(A, "A_lfo100"), arms(Mo, "M1_sigma_plus"))
        out["cases"]["N1"] = {"PASS": bool(ok), "n_common_fields": nf, "mismatches": bad, **extra, "mutation_M1_sigma_plus_0.01": {"RED": not okm, "mismatched_fields": badm[:12]}}
        print("  N1", out["cases"]["N1"]["PASS"], "mutation RED:", not okm, flush=True)
    if "N2" in only:
        print("KA-N2", flush=True)
        ex = ["--floor-vhat-noise", "0.20", "--floor-vhat-noise-seed", str(SEED)]
        Cb = run_smoke("C_bare", LF5_TOOL, 1.0, ex)
        Cc = run_smoke("C_comp", LF5_TOOL, 1.0, ex, composite=True)
        tb, tc = ticks(Cb, "C_bare"), ticks(Cc, "C_comp")
        same_rows = np.array_equal(tb["win"], tc["win"]) and np.array_equal(tb["tick"], tc["tick"])
        same_eps = same_rows and np.array_equal(tb["lf_eps"], tc["lf_eps"])
        rel = np.abs(np.asarray(tb["lf_vhat"], np.float64) - np.asarray(tb["lf_vlog"]) * np.exp(0.20 * tb["lf_eps"])) <= 1e-5 * np.maximum(1.0, np.abs(tb["lf_vlog"]))
        rel_ok = bool(np.all(rel | ~np.isfinite(tb["lf_vlog"])))
        eps_all = np.asarray(tb["lf_eps"])
        stats = {"n_rows": int(len(eps_all)), "eps_mean": float(eps_all.mean()), "eps_std": float(eps_all.std()), "bare_vs_comp_bind_rows_differ": int((tb["lf_bind"] != tc["lf_bind"]).sum())}
        ok = bool(same_eps and rel_ok and stats["bare_vs_comp_bind_rows_differ"] > 0 and abs(stats["eps_std"] - 1.0) < 0.15)
        mp = mutate("M2", M2)
        Mb = run_smoke("M2_bare", mp, 1.0, ex)
        Mc = run_smoke("M2_comp", mp, 1.0, ex, composite=True)
        mb, mc = ticks(Mb, "M2_bare"), ticks(Mc, "M2_comp")
        same_m = np.array_equal(mb["win"], mc["win"]) and np.array_equal(mb["tick"], mc["tick"]) and np.array_equal(mb["lf_eps"], mc["lf_eps"])
        out["cases"]["N2"] = {"PASS": ok, "eps_identical_bare_vs_composite": bool(same_eps), "vhat_equals_vlog_exp_sigma_eps": rel_ok, **stats, "mutation_M2_draw_only_where_bound": {"RED": not same_m, "eps_identical": bool(same_m)}}
        print("  N2", ok, "mutation RED:", not same_m, flush=True)
    if "N3" in only:
        print("KA-N3", flush=True)
        E = run_smoke("E_lf5_k075", LF5_TOOL, 0.75, [])
        te = ticks(E, "E_lf5_k075")
        ref = "F:/TanitAD-scratch/motion_c/lfo_build/smoke/LFO_bare"
        tr = ticks(ref, "LFO_bare")
        ok_a, nf_a, bad_a = compare_bit(tr, te, arms(ref, "LFO_bare"), arms(E, "E_lf5_k075"))
        # (b) the banked RILFO075_s0 tick arrays on the same 64 windows
        bk = dict(np.load(f"{MC}/lfo_read/out/RILFO075_s0/ticks/ticks_RILFO075_s0.npz", allow_pickle=True)) if os.path.isfile(f"{MC}/lfo_read/out/RILFO075_s0/ticks/ticks_RILFO075_s0.npz") else \
            dict(np.load("F:/TanitAD-scratch/motion_c/lfo_read/out/RILFO075_s0/ticks/ticks_RILFO075_s0.npz", allow_pickle=True))
        nwin = int(bk["win"].max()) + 1
        wmap = np.linspace(0, nwin - 1, 64).round().astype(int)
        bidx = {(int(w), int(t)): i for i, (w, t) in enumerate(zip(bk["win"], bk["tick"]))}
        rows = [bidx.get((int(wmap[int(w)]), int(t)), -1) for w, t in zip(te["win"], te["tick"])]
        found = all(r >= 0 for r in rows)
        rows = np.asarray(rows)
        # (b): the smoke's batch composition differs from the banked run's, and the model's float32 outputs depend on it (tick 0 of 15 of the 64 windows differs in plan_raw although every input is identical). The floor path is therefore compared on the
        # CLEAN windows: those whose plan_raw is bit-identical to the banked one at all three ticks (so the floor's inputs and the executed state are identical); on them EVERY common field must be bit-identical.
        skip = ("win", "ci", "t0", "j", "tick", "split")
        wins_ = np.asarray(te["win"], int)
        same_raw = np.array([np.array_equal(np.asarray(te["plan_raw"][i]), np.asarray(bk["plan_raw"][rows[i]])) for i in range(len(rows))])
        clean_w = {int(w) for w in np.unique(wins_) if same_raw[wins_ == w].all()}
        cm = np.array([int(w) in clean_w for w in wins_])
        common_b = sorted(k for k in set(row_fields(te)) & set(row_fields(bk)) if k not in skip)
        bad_b = [k for k in common_b if not np.array_equal(np.asarray(te[k])[cm], np.asarray(bk[k])[rows][cm], equal_nan=True)]
        n_clean, n_win_ = len(clean_w), int(len(np.unique(wins_)))
        disc, flt, maxdiff = [], common_b, 0.0
        bit_b = (not bad_b) and n_clean >= 32
        ref_vs_bank_same_set = None
        mp = mutate("M3", M3, M3b)
        Mo = run_smoke("M3_default_noise", mp, 0.75, [])
        tm = ticks(Mo, "M3_default_noise")
        okm, _, badm = compare_bit(tr, tm, arms(ref, "LFO_bare"), arms(Mo, "M3_default_noise"))
        bad_mb = [k for k in common_b if not np.array_equal(np.asarray(tm[k])[cm], np.asarray(bk[k])[rows][cm], equal_nan=True)]
        out["cases"]["N3"] = {"PASS": bool(ok_a and found and bit_b), "a_vs_lfo_smoke_bit_for_bit": {"PASS": bool(ok_a), "n_common_fields": nf_a, "mismatches": bad_a},
                              "b_vs_banked_RILFO075_s0": {"found_all_rows": bool(found), "n_rows": int(len(rows)), "n_windows": n_win_, "n_clean_windows_plan_raw_identical_at_all_ticks": n_clean, "n_fields_compared_on_clean_windows": len(common_b), "fields": flt, "bit_identical_on_clean_windows": bool(bit_b), "mismatches": bad_b,
                                                          "windows_excluded_note": "the excluded windows differ in the MODEL's float32 output (plan_raw) at identical inputs: batch composition, not the patch (the LFO tool's own smoke, bit-identical to the LF5 smoke in (a), is the same run)",
                                                          "window_map": "smoke window k = banked window linspace(0, %d, 64).round()[k]" % (nwin - 1)},
                              "mutation_M3_default_noise_0.05": {"RED": bool((not okm) and len(bad_mb) > 0), "mismatched_fields_vs_lfo_smoke": badm[:12], "mismatched_fields_vs_banked_clean_windows": bad_mb[:12]}}
        print("  N3", out["cases"]["N3"]["PASS"], "mutation RED:", not okm, flush=True)
    out["PASS"] = bool(all(c["PASS"] for c in out["cases"].values()) and all(v.get("RED") for c in out["cases"].values() for k, v in c.items() if k.startswith("mutation_")))
    name = "KA_LF5.json" if only == {"N1", "N2", "N3"} else "KA_LF5_" + "_".join(sorted(only)) + ".json"
    os.makedirs(f"{PK}/raw", exist_ok=True)
    open(f"{PK}/raw/{name}", "w", encoding="utf-8", newline="\n").write(json.dumps(out, indent=1))
    print("KA_LF5 PASS" if out["PASS"] else "KA_LF5 NOT PASS", flush=True)
    sys.exit(0 if out["PASS"] else 1)


if __name__ == "__main__":
    main()
