"""Mutation proof for the DDv2 port: every mutant of a mechanism must turn at least one test RED.

The REAL modules are never edited. Each mutant is the module source with ONE snippet replaced,
written to a temp file and swapped into ``sys.modules`` by a tiny pytest plugin before collection,
so every importer (tests and dependent modules) sees the mutant. A mutant that leaves the suite
GREEN is a SURVIVOR: the check it names is not able to see that defect.

Run:  python mutation_sweep.py --out ../raw/MUTATION_SWEEP.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

WT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5))
STACK = os.path.join(WT, "stack")
PY = sys.executable
TESTS = ["tests/test_ddv2_rl.py", "tests/test_ddv2_refc_chain.py", "tests/test_pdm_proxy.py"]

MUTANTS = [
    # --- ddv2_rl.py -------------------------------------------------------------------------
    ("ddv2_rl", "M01 no clip_sample clamp", "    if clip_sample:\n", "    if False:\n"),
    ("ddv2_rl", "M02 abar(-1) = table[0] not 1.0",
     "        return torch.ones((), dtype=table.dtype, device=table.device)", "        return table[0]"),
    ("ddv2_rl", "M03 per-coordinate exploration",
     "        mul = torch.cat((horizon, vert), dim=-1).repeat(1, 1, s, 1)",
     "        mul = torch.cat((horizon, vert), dim=-1).repeat(1, 1, s, 1) * (1 + 0.04 * torch.randn(b, n, s, 2, dtype=dtype, device=x0_hat.device))"),
    ("ddv2_rl", "M04 no exploration floor",
     "        std_mul = torch.clip(std_dev_t, min=explore_std_floor)", "        std_mul = std_dev_t"),
    ("ddv2_rl", "M05 likelihood floor 0.04 not 0.1",
     "    lik_std = torch.clip(std_dev_t, min=likelihood_std_floor)", "    lik_std = torch.clip(std_dev_t, min=0.04)"),
    ("ddv2_rl", "M06 GT bar dropped",
     "        adv = adv.clamp(min=0) * mask_positive.to(adv.dtype)", "        adv = adv.clamp(min=0)"),
    ("ddv2_rl", "M07 veto dropped",
     "    adv = torch.where(constraint_fail, torch.full_like(adv, consts.veto_value), adv)", "    adv = adv"),
    ("ddv2_rl", "M08 std eps 1e-8 not 1e-4",
     "    adv = (reward - mean_g) / (std_g + consts.adv_std_eps)", "    adv = (reward - mean_g) / (std_g + 1e-8)"),
    ("ddv2_rl", "M09 RL loss averaged over all samples",
     "             / mask_nz.sum(dim=1).clamp_min(1))", "             / per_token.shape[1])"),
    ("ddv2_rl", "M10 IL weights swapped",
     "    return torch.where(has_positive,\n                       torch.full(has_positive.shape, consts.il_weight_with_positive,",
     "    return torch.where(has_positive,\n                       torch.full(has_positive.shape, consts.il_weight_no_positive,"),
    ("ddv2_rl", "M11 discount reversed",
     "    return torch.tensor([gamma ** (step_num - i - 1) for i in range(step_num)],",
     "    return torch.tensor([gamma ** i for i in range(step_num)],"),
    ("ddv2_rl", "M12 rollout transition t -> t-2",
     "        prev, lp, _ = ddim_logprob_step(\n            x0_hat, x, t, t - 1,",
     "        prev, lp, _ = ddim_logprob_step(\n            x0_hat, x, t, t - 2,"),
    ("ddv2_rl", "M13 per-step decomposition divides by M",
     "    c_bt = nz.sum(dim=1).clamp_min(1).to(adv.dtype)", "    c_bt = torch.full_like(nz.sum(dim=1), nz.shape[1]).to(adv.dtype)"),
    ("ddv2_rl", "M14 anchor-major tiling",
     "    return x.unsqueeze(1).expand(b, groups, *x.shape[1:]).reshape(b, groups * n, *x.shape[2:])",
     "    return x.repeat_interleave(groups, dim=1)"),
    ("ddv2_rl", "M15 grad pass evaluates at a fresh sample, not the stored one",
     "        prev_sample=chain[..., i + 1])", "        prev_sample=None)"),
    # --- ddv2_refc_chain.py -----------------------------------------------------------------
    ("ddv2_refc_chain", "C01 residual dropped", "        return x_in + du", "        return du"),
    ("ddv2_refc_chain", "C02 input clamp ignored",
     "        x_in = x_n.clamp(-1.0, 1.0) if input_clamp else x_n", "        x_in = x_n"),
    ("ddv2_refc_chain", "C03 sampler speed = reference speed",
     "        v = (bank.new_full((b,), decoder.anchor_ref_speed) if v_ms is None\n             else v_ms.reshape(-1).to(torch.float32))",
     "        v = bank.new_full((b,), decoder.anchor_ref_speed)"),
    ("ddv2_refc_chain", "C04 capture accepts agents",
     "        if agents is not None:\n            raise", "        if False:\n            raise"),
    # --- pdm_proxy.py -----------------------------------------------------------------------
    ("pdm_proxy", "P01 no rear-axle offset",
     "    return (states[..., 0] + cfg.rear_axle_to_center * torch.cos(states[..., 2]),",
     "    return (states[..., 0] + 0.0 * torch.cos(states[..., 2]),"),
    ("pdm_proxy", "P02 every overlap at fault",
     "    atfault = ego_moving[..., None] & (track_stopped[None] | (front & ~behind))",
     "    atfault = torch.ones_like(front)"),
    ("pdm_proxy", "P03 static object scores 0 not 0.5", " * 0.5,\n", " * 0.0,\n"),
    ("pdm_proxy", "P04 initial overlap not ignored", "    over = over & ~initial[:, None]", "    over = over"),
    ("pdm_proxy", "P05 TTC forward cone dropped",
     "        fail |= (over & ahead & moving[..., None]).any(dim=(1, 2))",
     "        fail |= (over & moving[..., None]).any(dim=(1, 2))"),
    ("pdm_proxy", "P06 no lower lon-accel bound",
     "    ok = ((a_lon <= cfg.max_lon_accel) & (a_lon >= cfg.min_lon_accel)",
     "    ok = ((a_lon <= cfg.max_lon_accel) & (a_lon >= -100.0)"),
    ("pdm_proxy", "P07 EP reference = best candidate", "    ref = raw[0]", "    ref = raw.max()"),
    ("pdm_proxy", "P08 comfort term weighted on EP",
     "    return nc * dac * (cfg.w_ep * ep + cfg.w_ttc * ttc + cfg.w_c * c)",
     "    return nc * dac * (cfg.w_ep * ep + cfg.w_ttc * ttc + cfg.w_c * ep)"),
    ("pdm_proxy", "P09 human frame rotation sign",
     "    c, s = torch.cos(-yaw0), torch.sin(-yaw0)", "    c, s = torch.cos(yaw0), torch.sin(yaw0)"),
    ("pdm_proxy", "P10 agent ego@k -> world sign",
     "                wy = p[k, 1] + a[\"cx\"] * sk + a[\"cy\"] * ck",
     "                wy = p[k, 1] - a[\"cx\"] * sk + a[\"cy\"] * ck"),
    ("pdm_proxy", "P11 DAC never off-road",
     "    off = inside & sn & (frac < threshold)", "    off = inside & sn & (frac < 0.0)"),
    ("pdm_proxy", "P12 candidates rolled at 10 m/s",
     "    state0[:, 3] = v[:, None].expand(b, m).reshape(-1)\n    path = rollout_unicycle(state0, ticks, dt=cfg.dt)",
     "    state0[:, 3] = 10.0\n    path = rollout_unicycle(state0, ticks, dt=cfg.dt)"),
]

PLUGIN = '''
import importlib.util, os, sys
def pytest_configure(config):
    name = os.environ.get("MUT_MODULE")
    if not name:
        return
    import tanitad.rl as pkg
    full = "tanitad.rl." + name
    spec = importlib.util.spec_from_file_location(full, os.environ["MUT_PATH"])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    setattr(pkg, name, mod)
'''


def run_suite(env):
    t = time.time()
    p = subprocess.run([PY, "-m", "pytest", *TESTS, "-q", "-p", "no:cacheprovider", "-p", "mutplug"],
                       cwd=STACK, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    tail = [l for l in p.stdout.splitlines() if " passed" in l or " failed" in l or " error" in l]
    failed = [l.split("::")[-1].split(" ")[0] for l in p.stdout.splitlines() if l.startswith("FAILED")]
    return {"returncode": p.returncode, "summary": tail[-1] if tail else p.stdout[-300:],
            "failed_tests": failed, "secs": round(time.time() - t, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tmp = tempfile.mkdtemp(prefix="ddv2mut_")
    with open(os.path.join(tmp, "mutplug.py"), "w", encoding="utf-8") as fh:
        fh.write(PLUGIN)
    base_env = dict(os.environ, PYTHONPATH=os.pathsep.join([tmp, STACK]), OMP_NUM_THREADS="2",
                    PYTHONIOENCODING="utf-8")
    base_env.pop("MUT_MODULE", None)
    rec = {"_what": "mutation proof, PREREG_DDV2_RL_VALIDATION.md §2 machinery", "tests": TESTS}
    rec["baseline"] = run_suite(base_env)
    print("baseline", rec["baseline"]["summary"], flush=True)
    results = []
    for module, label, old, new in MUTANTS:
        src_path = os.path.join(STACK, "tanitad", "rl", module + ".py")
        src = open(src_path, encoding="utf-8").read()
        n = src.count(old)
        if n != 1:
            results.append({"mutant": label, "module": module, "status": "NOT-APPLIED",
                            "why": f"snippet occurs {n} times"})
            print(label, "NOT-APPLIED", n, flush=True)
            continue
        mpath = os.path.join(tmp, f"{module}__{label.split()[0]}.py")
        with open(mpath, "w", encoding="utf-8") as fh:
            fh.write(src.replace(old, new))
        env = dict(base_env, MUT_MODULE=module, MUT_PATH=mpath)
        r = run_suite(env)
        status = "KILLED" if r["failed_tests"] or r["returncode"] != 0 else "SURVIVED"
        results.append({"mutant": label, "module": module, "status": status, **r})
        print(label, status, r["summary"], r["failed_tests"][:4], flush=True)
    rec["mutants"] = results
    rec["n_killed"] = sum(1 for r in results if r["status"] == "KILLED")
    rec["n_survived"] = sum(1 for r in results if r["status"] == "SURVIVED")
    rec["n_not_applied"] = sum(1 for r in results if r["status"] == "NOT-APPLIED")
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    print(json.dumps({k: rec[k] for k in ("n_killed", "n_survived", "n_not_applied")}), flush=True)


if __name__ == "__main__":
    main()
