#!/usr/bin/env python3
"""YAXN CPU tests (run under CUDA_VISIBLE_DEVICES=-1): every gate with its deliberate-regression (RED) arm, the retry logic, the verdict combination, the pins.
Writes test_axn.out lines `PASS|FAIL <name> <note>`; last line ZZAXNTEST_DONE n_fail=<k>.   python test_axn.py [--no-integration]"""
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import torch  # noqa: E402
import axn_common as N  # noqa: E402
import axd_ft as W  # noqa: E402
import axn_analyze as AZ  # noqa: E402
import axn_chain as CH  # noqa: E402

fails, out = [], open(H / "test_axn.out", "w", encoding="utf-8")


def rec(ok, name, note=""):
    out.write(f"{'PASS' if ok else 'FAIL'} {name} {note}\n")
    out.flush()
    if not ok:
        fails.append(name)


class FakeCache:
    def scene(self, key):
        return torch.full((3,), float(key))


def fake_rows(n=600):
    g = torch.Generator().manual_seed(1)
    return [{"key": i, "ego": [float(x) for x in torch.randn(7, generator=g)], "goal": [0.0, 1.0], "traj": [[0.0, 0.0, 0.0]]} for i in range(n)]


def stats_of(x):
    s = W.Stats()
    s.add(torch.as_tensor(x, dtype=torch.float32))
    return s


# ---- pure gates and their RED inputs
g = torch.Generator().manual_seed(5)
z = torch.randn(8192, generator=g)
rec(W.std_gate(stats_of(z), 1.0)["pass"], "std_gate_green_sigma1")
rec(not W.std_gate(stats_of(0.5 * z), 1.0)["pass"], "RED_std_gate_half_sigma")
rec(not W.std_gate(stats_of(z * 0.0), 1.0)["pass"], "RED_std_gate_no_noise")
rec(not W.std_gate(stats_of(z + 0.5), 1.0)["pass"], "RED_std_gate_biased_mean")
rec(W.std_gate(stats_of(torch.zeros(100)), 0.0)["pass"], "std_gate_green_sigma0_exact_zero")
rec(not W.std_gate(stats_of(torch.full((100,), 0.01)), 0.0)["pass"], "RED_std_gate_sigma0_offset")
rec(not W.std_gate(W.Stats(), 1.0)["pass"], "RED_std_gate_no_draws")
rec(W.honest_gate(stats_of(torch.zeros(50)))["pass"], "honest_gate_green")
rec(not W.honest_gate(stats_of(torch.full((50,), 1e-7)))["pass"], "RED_honest_gate_1e-7")
rec(not W.honest_gate(W.Stats())["pass"], "RED_honest_gate_no_calls")
rec(W.std_gate(stats_of(torch.randn(160, generator=torch.Generator().manual_seed(2))), 1.0, widen_below_n=450)["pass"], "std_gate_mini_widened_green")

# ---- make_batch_noised on stub rows/cache: real code path, no model
rows, cache, ids = fake_rows(), FakeCache(), list(range(256))
W._orig_make_batch = lambda r, i, c, pol, reg="none": (torch.stack([c.scene(r[k]["key"]) for k in i]), torch.tensor([r[k]["ego"] for k in i], dtype=torch.float32).clone().index_fill_(1, torch.tensor([2]), 0.0) if pol == "zero" else torch.tensor([r[k]["ego"] for k in i], dtype=torch.float32), torch.zeros(len(i), 2), torch.zeros(len(i), 1, 3))
import ax_ft  # noqa: E402
real = ax_ft.real_ego(rows, ids)


def setup(sigma, regress="none", seed=0):
    W.S.sigma, W.S.regress, W.S.seed, W.S.off, W.S.mode = sigma, regress, seed, 7919, "train"
    W.S.gen, W.S.gen_leak = torch.Generator().manual_seed(seed + 7919), torch.Generator().manual_seed(seed + 7920)
    W.S.train, W.S.honest = W.Stats(), W.Stats()


setup(1.0)
rng0 = torch.get_rng_state()
_, eg, _, _ = W.make_batch_noised(rows, ids, cache, "real")
rec(torch.equal(rng0, torch.get_rng_state()), "noise_does_not_touch_global_rng")
exp = real[:, 2] + 1.0 * torch.randn(len(ids), generator=torch.Generator().manual_seed(7919))
rec(torch.equal(eg[:, 2], exp), "noise_is_exactly_the_dedicated_generator_stream(seed+7919)")
rec(torch.equal(eg[:, [0, 1, 3, 4, 5, 6]], real[:, [0, 1, 3, 4, 5, 6]]), "only_ax_column_changes")
rec(W.S.train.n == 256 and abs(W.S.train.std() - 1.0) < 0.15, "train_stats_recorded", f"std={W.S.train.std():.3f}")
with W.mode("honest"):
    _, eh, _, _ = W.make_batch_noised(rows, ids, cache, "real")
    _, ez, _, _ = W.make_batch_noised(rows, ids, cache, "zero")
rec(torch.equal(eh, real) and bool((ez[:, 2] == 0).all()) and W.honest_gate(W.S.honest)["pass"], "honest_mode_feeds_exact_real_ax_and_zero_policy_zero")
setup(1.0, "val_noised")
with W.mode("honest"):
    W.make_batch_noised(rows, ids, cache, "real")
rec(not W.honest_gate(W.S.honest)["pass"], "RED_val_noised_turns_honest_gate_red")
setup(1.0, "noise_not_applied")
for _ in range(4):
    W.make_batch_noised(rows, ids, cache, "real")
rec(not W.std_gate(W.S.train, 1.0)["pass"], "RED_noise_not_applied_turns_std_gate_red")
setup(0.0)
_, e0, _, _ = W.make_batch_noised(rows, ids, cache, "real")
rec(torch.equal(e0, real), "sigma0_feeds_bit_identical_ego")
setup(1.0)
try:
    W.make_batch_noised(rows, ids, cache, "zero")
    # policy zero in train mode must refuse
    rec(False, "RED_noise_refuses_policy_zero")
except RuntimeError:
    rec(True, "noise_refuses_policy_zero")

# ---- verdict combination
ov = AZ.overall
rec(ov("CANDIDATE", "CANDIDATE") == "CANDIDATE" and ov("CANDIDATE", "NOT PROVEN") == "NOT PROVEN" and ov("REFUTED", "REFUTED") == "REFUTED" and ov("REFUTED", "NOT PROVEN") == "NOT PROVEN"
    and ov("VOID", "CANDIDATE") == "VOID" and ov("CANDIDATE", "VOID") == "VOID", "overall_verdict_truth_table(both seeds must pass)")

# ---- AMENDMENT_1: the relative gain clause reads AXON_s's IN-RUN gain; seam gates for the reference; literal checkpoint deletion
import numpy as np  # noqa: E402
import axd_analyze as AN  # noqa: E402
pd_ = {"point": 0.0, "ci95": [-0.2, 0.2]}
gT = {"down": {"point": 0.45, "ci95": [0.35, 0.55]}}
g_axon = {"down": {"point": 0.80, "ci95": [0.70, 0.90]}}
g_u = {"down": {"point": 1.08, "ci95": [0.84, 1.31]}}
dA, dU = AN.decide("AXN", [], pd_, gT, g_axon, 0.0, True), AN.decide("AXN", [], pd_, gT, g_u, 0.0, True)
rec(dA["verdict"] == "NOT PROVEN" and dA.get("failing") == ["gain_half"] and dU["verdict"] == "CANDIDATE", "relative_clause_reads_the_reference_it_is_given(AXON in-run stricter than u400 here)", f"axon={dA['verdict']} u400={dU['verdict']}")
src = Path(AZ.__file__).read_text(encoding="utf-8")
rec('AN.decide("AXN", red, pdms, g_own, g_ref, d_lon, floor_ok)' in src and 'AN.decide("AXN", red, pdms, g_own, g_u400' in src and src.index("g_own, g_ref, d_lon") < src.index("g_own, g_u400"), "decisive_decide_uses_g_ref_and_u400_variant_is_descriptive_only")
rng_ = np.random.default_rng(0)
n_, nc_ = 12, 5
raw_ = rng_.normal(0, 1, (n_, 7)).astype(np.float32)
hist_ = rng_.normal(0, 1, (n_, 3))
a0_ = rng_.normal(0, 5, (n_, 8, 3)).astype(np.float32)
rr = {"raw": raw_, "hist": hist_, "a0_poses": a0_, "a0_seam_poses": a0_.copy(), "fed__IDT": raw_.copy(), "fed__IDTm": raw_.copy(), "poses__IDT": a0_.copy(), "poses__IDTm": a0_.copy(),
      "cruise_raw": raw_[:nc_], "cruise_hist": hist_[:nc_], "token": np.array([f"t{i}" for i in range(n_)])}
for nm_, t_, d_, m_ in N.A.gain_arms():
    if t_ == "IDT":
        f_ = raw_[:nc_].copy()
        f_[:, 2] = AN.expected_fed("IDT", raw_[:nc_], hist_[:nc_], d_, m_)
        rr[f"gfed__{nm_}"] = f_
tk_, mt_ = [f"t{i}" for i in range(n_)], {"rc": 0, "missing_records": []}
gg = AZ.seam_gates(rr, mt_, tk_, "REF:")
rec(all(v["PASS"] for v in gg.values()) and set(gg) == {"REF:G-SEAM", "REF:G-IDT", "REF:G-MEMO", "REF:G-REACH:IDTgain"}, "seam_gates_green_on_reference_records(prefix REF:)")
r2 = dict(rr)
r2["fed__IDT"] = raw_.copy()
r2["fed__IDT"][:, 2] += np.float32(1e-3)
r3 = dict(rr)
r3.update({k: raw_[:nc_].copy() for k in rr if k.startswith("gfed__")})
rec(not AZ.seam_gates(r2, mt_, tk_, "REF:")["REF:G-IDT"]["PASS"] and not AZ.seam_gates(r3, mt_, tk_, "REF:")["REF:G-REACH:IDTgain"]["PASS"]
    and not AZ.seam_gates(rr, {"rc": 1, "missing_records": []}, tk_, "REF:")["REF:G-SEAM"]["PASS"], "RED_seam_gates_catch_identity_perturbation_unreached_gain_arms_and_bad_rc")
base_ = "E:/Projects/TanitAD-artifacts/yawrate_wp/YAXN/run/"
lit_ok = len(N.CKPT_LITERAL) == 12 and all(v == f"{base_}{k[0]}/ckpt/{k[1]}_s{k[2]}.pt" for k, v in N.CKPT_LITERAL.items())
rec(lit_ok and N.remove_ckpt("unit", "AXON", 1) is False, "ckpt_deletion_only_through_literal_paths(12 entries; absent file -> False)")
try:
    N.remove_ckpt("full", "OTHER", 0)
    rec(False, "RED_remove_ckpt_refuses_unlisted_path")
except KeyError:
    rec(True, "RED_remove_ckpt_refuses_unlisted_path")
rec(all(N.AXON_FINAL[s_].exists() and N.AXON_FINAL[s_].parent.name == f"K1_AXON_s{s_}" for s_ in N.SEEDS), "axon_final_pt_are_the_K1_arms_X1_used(select.json chosen_X1=K1)")

# ---- retry logic with a fake seam command (smoke ctx: no lock)
CH.time.sleep = lambda *_: None
N.wait_mem = lambda *a, **k: {}
N.require_disk = lambda *a, **k: None
tmp = Path(tempfile.mkdtemp(prefix="axn_test_", dir="C:/Users/Admin/AppData/Local/Temp"))
ctx = CH.Ctx("unit")
ctx.smoke = True
ctx.root = tmp
for d in ("logs", "stages"):
    (tmp / d).mkdir()
ctx.log = lambda m: None
cnt = tmp / "cnt.txt"
script = tmp / "fake_seam.py"
script.write_text("import sys,pathlib\np=pathlib.Path(sys.argv[1]);mode=sys.argv[2];n=int(p.read_text()) if p.exists() else 0;p.write_text(str(n+1))\n"
                  "if mode=='memtwice' and n<2: print('numpy.core._exceptions._ArrayMemoryError: Unable to allocate 5.62 MiB'); sys.exit(1)\n"
                  "if mode=='alwaysmem': print('MemoryError'); sys.exit(1)\n"
                  "if mode=='other': print('ValueError: boom'); sys.exit(1)\n"
                  "print('ZZAXDSEAM OK')\n", encoding="utf-8")


def fake(mode):
    cnt.unlink(missing_ok=True)
    return [sys.executable, str(script), str(cnt), mode]


r = CH.run_seam(ctx, "t1", fake("memtwice"), tmp / "arms1")
rec(r["attempts"] == 3 and (tmp / "logs" / "retries.jsonl").exists(), "seam_retries_on_MemoryError_then_succeeds", f"attempts={r['attempts']}")
try:
    CH.run_seam(ctx, "t2", fake("alwaysmem"), tmp / "arms2", max_attempts=3)
    rec(False, "RED_seam_retry_is_bounded")
except RuntimeError as e:
    rec("attempt 3/3" in str(e) and cnt.read_text() == "3", "seam_retry_is_bounded_and_fails_loud", f"runs={cnt.read_text()}")
try:
    CH.run_seam(ctx, "t3", fake("other"), tmp / "arms3")
    rec(False, "RED_non_memory_failure_not_retried")
except RuntimeError:
    rec(cnt.read_text() == "1", "non_memory_failure_is_not_retried", f"runs={cnt.read_text()}")
rec(N.is_mem_failure("x" * 100 + "numpy.core._exceptions._ArrayMemoryError: Unable to allocate") and not N.is_mem_failure("ZeroDivisionError"), "is_mem_failure_signatures")

# ---- pins: a tampered code file must be REFUSED
td = Path(tempfile.mkdtemp(prefix="axn_pins_", dir="C:/Users/Admin/AppData/Local/Temp"))
for f in N.CODE + N.PINNED_DOCS + ["CODE_PIN_AXN.json"]:
    if (H / f).exists():
        shutil.copyfile(H / f, td / f)
pin = json.loads((td / "CODE_PIN_AXN.json").read_text())
pin["axn_analyze.py"] = "0" * 64
(td / "CODE_PIN_AXN.json").write_text(json.dumps(pin))
CH.HERE = td
try:
    CH.check_pins(ctx)
    rec(False, "RED_pins_refuse_tampered_pin")
except RuntimeError as e:
    rec("axn_analyze.py" in str(e), "RED_pins_refuse_tampered_pin", "REFUSING raised")
CH.HERE = H
shutil.rmtree(td, ignore_errors=True)
shutil.rmtree(tmp, ignore_errors=True)

# ---- integration on the real cache: the chain's own gates_axn stage (sigma0 == AXON bit for bit, sigma1 != AXON, determinism, 4 RED arms) + compose/identity of a sigma-1 mini checkpoint
if "--no-integration" not in sys.argv:
    importlib_ctx = CH.Ctx("unit")
    importlib_ctx.log = lambda m: out.write("  # " + m + "\n")
    N.wait_mem = lambda *a, **k: {}
    t0 = time.time()
    try:
        CH.gates_axn(importlib_ctx)
        ga = N.jload(importlib_ctx.root / "analysis" / "gates_axn.json")
        rec(ga["pass"], "gates_axn_stage(sigma0==AXON bit-for-bit, sigma1 differs, 4 RED arms read RED)", f"{json.dumps({k: v for k, v in ga.items() if k != 'RED_arms'})[:600]} RED={ {k: v['RED'] for k, v in ga['RED_arms'].items()} } {time.time() - t0:.0f}s")
    except Exception as e:
        rec(False, "gates_axn_stage", repr(e)[:800])
    else:
        d = importlib_ctx.root / "gates_axn" / "sigma1"
        ck = importlib_ctx.root / "ckpt_unit.pt"
        try:
            CH.sh_retry(importlib_ctx, "u_build", [N.PY, N.YAX / "ax_ckpt_tools.py", "build", "--final", d / "final.pt", "--out", ck], "ZZAXCKPT_OK", env=N.CPU_ENV)
            CH.sh_retry(importlib_ctx, "u_id", [N.PY, N.YAX / "ax_ckpt_tools.py", "identity", "--ckpt", ck, "--final", d / "final.pt", "--out", importlib_ctx.root / "id_unit.json", "--frames", "64"], "ZZAXID_OK", env=N.CPU_ENV)
            meta = torch.load(ck, map_location="cpu", weights_only=False)["meta"]
            rec(meta["ax_ft"]["arm"] == "AXNOISE" and meta["ego_ax_policy"] == "real", "compose_and_identity_of_AXNOISE_checkpoint(K1-K5, G-CKPT)", f"arm={meta['ax_ft']['arm']}")
        except Exception as e:
            rec(False, "compose_and_identity", repr(e)[:600])
        finally:
            if ck.exists():
                os.remove(ck)

    try:
        ckx = CH.compose_axon(importlib_ctx, 0)
        gk = N.jload(str(ckx) + ".gates.json")
        idj = N.jload(importlib_ctx.root / "identity" / "AXON_s0.json")
        rec(all(v for k_, v in gk.items() if k_.startswith("K") and v is not None) and idj["pass"], "compose_axon_from_banked_K1_final(K1-K5 + identity green)", f"G_CKPT={idj['G_CKPT_file_equals_final']} gain_visible={idj['GA_GAIN']}")
    except Exception as e:
        rec(False, "compose_axon", repr(e)[:600])
    finally:
        N.remove_ckpt("unit", "AXON", 0)
        rec(not (importlib_ctx.root / "ckpt" / "AXON_s0.pt").exists(), "axon_ckpt_deleted_via_literal_path")
out.write(f"ZZAXNTEST_DONE n_fail={len(fails)} {fails}\n")
out.close()
