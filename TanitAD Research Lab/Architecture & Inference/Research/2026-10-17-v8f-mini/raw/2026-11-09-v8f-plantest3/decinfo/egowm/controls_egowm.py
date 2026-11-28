"""PREREG_V8F_EGOWM controls 1 to 5, each with its deliberate-regression arm (a mutation that must make the check go RED).
CPU only (CUDA_VISIBLE_DEVICES=-1, 2 threads). Usage:
    python controls_egowm.py --tree <patched tree> --base <baseline tree> --out <dir>      (all controls)
    python controls_egowm.py --dump-c1 <tree> <npz>                                         (child of control 1)
Batch: the REAL val windows of ladder_read/cache/frames_s0.npz (zmean, act of 141 clips). That file carries no speed, so the
measured speed `v` of a window is PLANTED: 12 m/s at the window start, then the floored integration of the real accelerations.
"""
import argparse, copy, json, os, subprocess, sys, hashlib
import numpy as np

FRAMES = "F:/TanitAD-scratch/ladder_read/cache/frames_s0.npz"
ARGV0 = "C:/Users/Admin/t2v8f_pulls/ladder_R35H_s0/argv.txt"
F_WIN = 91


def setup(tree):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0, os.path.join(tree, "stack")), sys.path.insert(0, os.path.join(tree, "stack", "scripts"))
    import tanitad
    assert os.path.normcase(os.path.abspath(tanitad.__file__)).startswith(os.path.normcase(os.path.abspath(tree))), tanitad.__file__
    import v8f_mini_train as T
    from tanitad.models import v8f_mini as M
    return torch, T, M


def real_batch(torch, n=4, M=None, v0=12.0):
    f = np.load(FRAMES)
    offs = np.concatenate([[0], np.cumsum(f["T"])])
    zs, acts, oks = [], [], []
    for i in range(len(f["T"])):
        T_ = int(f["T"][i])
        if T_ < F_WIN:
            continue
        lo = offs[i] + T_ - F_WIN
        if not f["ok"][lo:lo + F_WIN].all():
            continue
        zs.append(f["zmean"][lo:lo + F_WIN]), acts.append(np.nan_to_num(f["act"][lo:lo + F_WIN]))
        if len(zs) == n:
            break
    z, a = torch.as_tensor(np.stack(zs), dtype=torch.float32), torch.as_tensor(np.stack(acts), dtype=torch.float32)
    # the planted measured speed: v0 at the window start, floored integration of the real accelerations (None if M is None -> caller integrates)
    v = M.reflect_integrate(torch.full((z.shape[0],), v0), a[:, :-1, 0]) if hasattr(M, "reflect_integrate") else None
    if v is None:
        return {"zmean": z, "act": a}
    return {"zmean": z, "act": a, "v": v, "t": torch.zeros(n, dtype=torch.long), "ci": torch.zeros(n, dtype=torch.long)}


def cfg_for(T, extra):
    argv = open(ARGV0).read().split()
    a = T.build_parser().parse_args(argv + extra)
    return a, T.make_config(a)


def full_model(torch, M, cfg, seed=0):
    torch.manual_seed(seed)
    return M.V8FMini(cfg, M.P10Prior.synthetic(0), None, None)


# ------------------------------------------------------------------------------------------ control 1 (child): dump one step
def dump_c1(tree, out):
    torch, T, M = setup(tree)
    b = real_batch(torch, M=M)
    _, cfg = cfg_for(T, [])
    m = full_model(torch, M, cfg)
    res = {}
    sd0 = {k: v.clone() for k, v in m.state_dict().items()}
    for k, v in sd0.items():
        res["init/" + k] = v.numpy()
    params = [p for n, p in m.named_parameters() if n.startswith("wm.")]
    opt = torch.optim.AdamW(params, lr=1e-3)
    m.train()
    torch.manual_seed(7)
    terms = m.wm_only_loss({k: b[k] for k in ("zmean", "act")})["terms"]
    for k, v in terms.items():
        res["term/" + k] = v.detach().numpy()
    terms["total"].backward()
    for n, p in m.named_parameters():
        if n.startswith("wm."):
            res["grad/" + n] = p.grad.detach().numpy().copy()
    opt.step()
    for n, p in m.named_parameters():
        if n.startswith("wm."):
            res["after/" + n] = p.detach().numpy().copy()
    np.savez(out, **res)


def control1(tree, base, outdir):
    me = os.path.abspath(__file__)
    py = sys.executable
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2")
    paths = {}
    for nm, t in (("base", base), ("patched", tree)):
        paths[nm] = os.path.join(outdir, f"_c1_{nm}.npz")
        env2 = dict(env, PYTHONPATH=";".join([os.path.join(t, "stack"), os.path.join(t, "stack", "scripts")]))
        r = subprocess.run([py, me, "--dump-c1", t, paths[nm]], env=env2, capture_output=True, text=True, encoding="utf-8")
        if r.returncode:
            raise RuntimeError(r.stderr[-2000:])
    A, B = np.load(paths["base"]), np.load(paths["patched"])
    keys = sorted(A.files)
    same = [k for k in keys if k in B.files and np.array_equal(A[k], B[k])]
    res = {"n_arrays": len(keys), "n_bit_equal": len(same), "keys_only_in_patched": sorted(set(B.files) - set(A.files)),
           "differing": sorted(set(keys) - set(same))[:6], "pass_off_identity": len(same) == len(keys) and set(A.files) == set(B.files)}
    A.close(); B.close()
    for p in paths.values():
        os.remove(p)
    return res


def control1b_and_mutation(torch, T, M, b):
    """Zero-initialised speed weights reproduce the control: E-KIN and E-NOW at step 0 (same seed) equal R35H's loss terms and every non-spd init tensor bit for bit;
    the deliberate-regression arm perturbs the speed weights and the comparison must go RED."""
    _, c_off = cfg_for(T, [])
    m_off = full_model(torch, M, c_off)
    base_terms = {k: v.detach() for k, v in m_off.wm_only_loss({k: b[k] for k in ("zmean", "act")})["terms"].items()}
    out = {}
    for mode in ("kin", "now"):
        _, c = cfg_for(T, ["--wm-speed", mode])
        m = full_model(torch, M, c)
        sd_o, sd_n = m_off.state_dict(), m.state_dict()
        extra = sorted(set(sd_n) - set(sd_o))
        eq_init = all(torch.equal(sd_o[k], sd_n[k]) for k in sd_o)
        t = {k: v.detach() for k, v in m.wm_only_loss(b)["terms"].items()}
        eq_terms = sorted(k for k in base_terms if k in t and torch.equal(base_terms[k], t[k]))
        # mutation: speed weights non-zero -> terms must differ
        m2 = full_model(torch, M, c)
        for p in m2.wm.stack.spd:
            torch.nn.init.normal_(p.weight, std=0.5)
        t2 = {k: v.detach() for k, v in m2.wm_only_loss(b)["terms"].items()}
        eq_mut = sorted(k for k in base_terms if k in t2 and torch.equal(base_terms[k], t2[k]))
        out[mode] = {"extra_keys": extra, "all_control_tensors_bit_equal_at_init": bool(eq_init), "n_terms": len(base_terms), "n_terms_bit_equal_zero_weights": len(eq_terms),
                     "mutation_nonzero_weights_n_terms_bit_equal": len(eq_mut),
                     "pass": bool(eq_init and len(eq_terms) == len(base_terms) and len(eq_mut) < len(base_terms) and extra == [f"wm.stack.spd.{i}.weight" for i in range(3)])}
    return out


# ------------------------------------------------------------------------------------------ control 2
def control2(torch, T, M, b):
    out = {}
    for mode in ("kin", "now"):
        _, c = cfg_for(T, ["--wm-speed", mode])
        m = full_model(torch, M, c).cpu()
        live = T.wm_only_live_check(m, b, c)                       # the trainer's own G-LIVE on the real batch
        # per-level gradient of the speed channel, from the same path (one backward)
        mm = copy.deepcopy(m)
        for p in mm.parameters():
            p.requires_grad_(True)
        torch.manual_seed(0)
        mm.wm_only_loss(b)["terms"]["total"].backward()
        lv = {i: float(mm.wm.stack.spd[i].weight.grad.abs().sum()) for i in range(3)}
        out[mode] = {"live_pass": live["pass"], "dead_wm_groups": live["dead_wm_groups"], "control_nonwm_grad_sum": live["control_nonwm_grad_sum"], "spd_group_grad_sum": live["wm_groups_grad"].get("wm.stack.spd"),
                     "per_level_grad_abs_sum": lv, "all_levels_nonzero": all(g > 0 for g in lv.values())}
    # deliberate regression: delete the speed input (the embedding is multiplied by 0 -> the line that feeds the speed to the channel emits nothing)
    _, c = cfg_for(T, ["--wm-speed", "kin"])
    orig = M.ego_stack_cls().speed_embed
    M.ego_stack_cls().speed_embed = lambda self, level, speed: self.spd[level - 1](speed.unsqueeze(-1) * 0.0)
    try:
        try:
            T.wm_only_live_check(full_model(torch, M, c).cpu(), b, c)
            red = False
            why = "live check PASSED under the mutation"
        except AssertionError as e:
            red, why = True, str(e)[:240]
    finally:
        M.ego_stack_cls().speed_embed = orig
    out["mutation_speed_input_deleted"] = {"went_red": red, "why": why}
    out["pass"] = bool(all(out[m_]["live_pass"] and out[m_]["all_levels_nonzero"] and out[m_]["control_nonwm_grad_sum"] == 0.0 for m_ in ("kin", "now")) and red)
    return out


# ------------------------------------------------------------------------------------------ control 3
def hand_speeds(mode, v, acc, lvl_cum, k, idx):
    """independent python reference: [A][k] speeds for one window row."""
    res = []
    for i in idx:
        f0 = i * lvl_cum
        row = []
        for h in range(k):
            f = f0 + h * lvl_cum
            if mode == "now":
                row.append(float(v[f0]))
            else:
                s = float(v[f0])
                for mm in range(f0, f):
                    s = max(0.0, s + 0.2 * (float(acc[mm]) if mm < len(acc) else 0.0))   # steps past the window are masked in the loss; the reference pads with 0 as the code does
                row.append(s)
        res.append(row)
    return res


def spy_run(torch, M, mode, b, cfg):
    """Run the BUILT stack's loss on the batch with `speed_embed` spied: returns the speeds the channel was HANDED, in call order, with the level."""
    cls = M.ego_stack_cls()
    calls = []
    orig = cls.speed_embed

    def spy(self, level, speed):
        calls.append((level, speed.detach().clone()))
        return orig(self, level, speed)
    cls.speed_embed = spy
    try:
        m = full_model(torch, M, cfg)
        m.wm_only_loss(b)
    finally:
        cls.speed_embed = orig
    return calls, m


def check_speeds(torch, M, T, mode, b, cfg):
    calls, m = spy_run(torch, M, mode, b, cfg)
    st = m.wm.stack
    cum = st.cum
    k_of = [c.rollout_k for c in st.cfg]
    stride_of = [c.rollout_stride for c in st.cfg]
    v, acc = b["v"], b["act"][:, :-1, 0]
    ok_tf, ok_ro, n_tf, n_ro, maxerr = True, True, 0, 0, 0.0
    by_level = {}
    for lvl, s in calls:
        by_level.setdefault(lvl, []).append(s)
    for L in (1, 2, 3):
        c = cum[L - 1]
        tf, ro = by_level[L][0], by_level[L][1]                     # call order per level: encode (teacher-forced) then the free run
        n = tf.shape[1]
        want = v[:, 0:n * c:c][:, :n]
        ok_tf &= bool(torch.equal(tf, want))
        n_tf += 1
        k, stride = k_of[L - 1], stride_of[L - 1]
        n_states = (F_WIN - 1) // c + 1 if L > 1 else F_WIN
        # states per level of the BUILT stack (measured)
        n_states = int(m.wm.stack.states_per_level(F_WIN)[L - 1]) if hasattr(m.wm.stack, "states_per_level") else n_states
        idx = list(range(0, n_states - 1, stride))
        for bi in range(v.shape[0]):
            ref = torch.tensor(hand_speeds(mode, v[bi], acc[bi], c, k, idx), dtype=torch.float32)
            got = ro.reshape(v.shape[0], len(idx), k)[bi]
            err = float((ref - got).abs().max())
            maxerr = max(maxerr, err)
            ok_ro &= err < 1e-5
        n_ro += 1
    floor_hit = bool(((v == 0).any()).item())
    return {"teacher_forced_equal_measured": ok_tf, "free_run_matches_hand_integration": ok_ro, "max_abs_err_vs_hand": maxerr, "levels_checked": 3, "window_rows": int(v.shape[0]), "floor_engaged_in_batch": floor_hit}


def control3(torch, T, M, b):
    out = {}
    # a planted deceleration that must hit the floor, plus the real-acceleration windows
    pl = copy.deepcopy(b)
    pl["act"][0, :, 0] = -1.0
    pl["act"][1, :, 0] = 0.0
    pl["act"][1, 40:60, 0] = -3.0
    pl["v"] = M.reflect_integrate(torch.tensor([12.0, 12.0, 0.0, 3.0]), pl["act"][:, :-1, 0])
    for mode in ("kin", "now"):
        _, c = cfg_for(T, ["--wm-speed", mode])
        out[mode] = check_speeds(torch, M, T, mode, pl, c)
    # a direct check of the primitive: the floor is applied at every step
    v0 = torch.tensor([1.0]); acc = torch.tensor([[-10.0, 10.0, 10.0]])
    out["floor_primitive"] = {"got": M.reflect_integrate(v0, acc).tolist()[0], "hand": [1.0, 0.0, 2.0, 4.0]}
    out["floor_primitive"]["pass"] = out["floor_primitive"]["got"] == out["floor_primitive"]["hand"] or np.allclose(out["floor_primitive"]["got"], out["floor_primitive"]["hand"])
    # mutations
    real_ra = M.reflect_integrate
    real_as = M.anchored_speeds
    muts = {}
    M.reflect_integrate = lambda v0, acc, dt=0.2: real_ra(v0, acc, -dt)           # the sign of the integration flipped
    try:
        muts["integration_sign_flipped"] = check_speeds(torch, M, T, "kin", pl, cfg_for(T, ["--wm-speed", "kin"])[1])
    finally:
        M.reflect_integrate = real_ra

    def future_now(mode, vm, acc, idx, k, cum, dt=0.2):                           # E-NOW takes the speed one rollout length later
        f0 = torch.clamp(idx * cum + k * cum, max=vm.shape[1] - 1)
        return vm[:, f0].unsqueeze(-1).expand(-1, -1, k).contiguous() if mode == "now" else real_as(mode, vm, acc, idx, k, cum, dt)
    M.anchored_speeds = future_now
    try:
        muts["now_takes_future_speed"] = check_speeds(torch, M, T, "now", pl, cfg_for(T, ["--wm-speed", "now"])[1])
    finally:
        M.anchored_speeds = real_as
    out["mutations"] = {k: {"went_red": not (v["free_run_matches_hand_integration"] and v["teacher_forced_equal_measured"]), **v} for k, v in muts.items()}
    out["pass"] = bool(all(out[m_]["teacher_forced_equal_measured"] and out[m_]["free_run_matches_hand_integration"] for m_ in ("kin", "now")) and out["floor_primitive"]["pass"]
                       and all(v["went_red"] for v in out["mutations"].values()) and out["kin"]["floor_engaged_in_batch"])
    return out


# ------------------------------------------------------------------------------------------ control 4
def control4(torch, T, M, b):
    _, c0 = cfg_for(T, [])
    m_off = full_model(torch, M, c0)
    out = {}
    for mode in ("kin", "now"):
        _, c = cfg_for(T, ["--wm-speed", mode])
        m = full_model(torch, M, c)
        m.load_state_dict(m_off.state_dict(), strict=False)           # identical base tensors; the speed weights are then made NON-ZERO so any leak would show
        for p in m.wm.stack.spd:
            torch.nn.init.normal_(p.weight, std=0.5)
        z1 = m_off.wm.z1_from_mean(b["zmean"])
        a = m_off.wm.scale_actions(b["act"])[:, :-1]
        enc_off = m_off.wm.stack.encode(z1, a)
        with torch.no_grad():
            m.wm.stack._ego = m.wm.ego_of(b["act"], b["v"])
            enc_on = m.wm.stack.encode(m.wm.z1_from_mean(b["zmean"]), a)
            v2 = b["v"].clone(); v2[:, 30:] += 7.0                     # a different FUTURE speed
            m.wm.stack._ego = m.wm.ego_of(b["act"], v2)
            enc_on2 = m.wm.stack.encode(m.wm.z1_from_mean(b["zmean"]), a)
            m.wm.stack._ego = None
        tgt_equal = [bool(torch.equal(enc_off.z[i], enc_on.z[i])) for i in range(3)]
        tgt_inv = [bool(torch.equal(enc_on.z[i], enc_on2.z[i])) for i in range(3)]
        z1on = m.wm.z1_from_mean(b["zmean"])
        goal_equal = [bool(torch.equal(m_off.wm.stack.lift(z1[:, :31], 1, L, strict=False), m.wm.stack.lift(z1on[:, :31], 1, L, strict=False))) for L in (2, 3)]   # goal latents: a window lifted to level 2 / 3
        lift_equal = bool(torch.equal(m_off.wm.stack.lift(z1[:, ::3], 1, 3, strict=False), m.wm.stack.lift(m.wm.z1_from_mean(b["zmean"])[:, ::3], 1, 3, strict=False)))
        act_differs = [not torch.equal(enc_off.act[i], enc_on.act[i]) for i in range(3)]
        out[mode] = {"target_latents_equal_off_vs_on": tgt_equal, "target_latents_invariant_to_future_speed": tgt_inv, "goal_latents_equal": goal_equal, "lifted_goal_equal": lift_equal,
                     "control_that_actions_DO_differ": act_differs, "pass": bool(all(tgt_equal) and all(tgt_inv) and all(goal_equal) and lift_equal and all(act_differs))}
    # mutation: the speed leaks into the target latent (encode adds the speed embedding to z)
    _, c = cfg_for(T, ["--wm-speed", "kin"])
    cls = M.ego_stack_cls()
    orig_enc = cls.encode

    def leaky(self, z1_seq, a_seq):
        enc = orig_enc(self, z1_seq, a_seq)
        enc.z[0] = enc.z[0] + self._ego["v"].unsqueeze(-1) / M.WM_SPEED_SCALE      # the measured speed leaks into the level-1 target latent
        return enc
    cls.encode = leaky
    try:
        m = full_model(torch, M, c)
        m.load_state_dict(m_off.state_dict(), strict=False)
        for p in m.wm.stack.spd:
            torch.nn.init.normal_(p.weight, std=0.5)
        m.wm.stack._ego = m.wm.ego_of(b["act"], b["v"])
        enc_leak = m.wm.stack.encode(m.wm.z1_from_mean(b["zmean"]), m_off.wm.scale_actions(b["act"])[:, :-1])
        m.wm.stack._ego = None
        enc_ref = m_off.wm.stack.encode(m_off.wm.z1_from_mean(b["zmean"]), m_off.wm.scale_actions(b["act"])[:, :-1])
        red = not torch.equal(enc_leak.z[0], enc_ref.z[0])
    finally:
        cls.encode = orig_enc
    out["mutation_speed_leaks_into_target"] = {"went_red": bool(red)}
    out["pass"] = bool(out["kin"]["pass"] and out["now"]["pass"] and red)
    return out


# ------------------------------------------------------------------------------------------ control 5
def control5(torch, T, M, argvs_path):
    import dataclasses
    lines = [l for l in open(argvs_path, encoding="utf-8").read().splitlines() if l.strip() and not l.startswith("#")]
    res = []
    dfl = T.build_parser()
    for l in lines:
        name, argv = l.split("\t", 1)
        toks = argv.split()
        a = dfl.parse_args(toks)
        cfg = T.make_config(a)
        want = a.wm_speed
        named = "--wm-speed" in toks
        torch_ = torch
        torch_.manual_seed(a.seed)
        model = M.V8FMini(cfg, M.P10Prior.synthetic(0), None, None)
        data_info = {"wm_windows": {"ego": want != "off"}}
        bad = T.bad_wm_speed(a, cfg, model, data_info)
        # deliberate regression: the line that applies --wm-speed is deleted (cfg built without it) -> the built stack is 'off' -> must be refused
        cfg_m = dataclasses.replace(cfg, wm_speed="off")
        model_m = M.V8FMini(cfg_m, M.P10Prior.synthetic(0), None, None)
        bad_m = T.bad_wm_speed(a, cfg_m, model_m, data_info) if want != "off" else ["n/a"]
        res.append({"arm": name, "seed": a.seed, "wm_speed_named": named, "declared": want, "built": model.wm.built_speed()["mode"], "bad": bad, "mutation_refused": bool(bad_m)})
    ok = all(r["wm_speed_named"] and r["declared"] == r["built"] and not r["bad"] and r["mutation_refused"] for r in res if r["declared"] != "off")
    a0 = dfl.parse_args(open(ARGV0).read().split())               # the control R35H argv: no --wm-speed -> off, built off, nothing refused
    c0 = T.make_config(a0)
    torch.manual_seed(a0.seed)
    m0 = M.V8FMini(c0, M.P10Prior.synthetic(0), None, None)
    ctrl = {"declared": a0.wm_speed, "built": m0.wm.built_speed()["mode"], "bad": T.bad_wm_speed(a0, c0, m0, {"wm_windows": {"ego": False}})}
    return {"arms": res, "control_R35H": ctrl, "pass": bool(ok and len(res) == 5 and ctrl["declared"] == "off" == ctrl["built"] and not ctrl["bad"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree"), ap.add_argument("--base"), ap.add_argument("--out"), ap.add_argument("--argvs")
    ap.add_argument("--dump-c1", nargs=2)
    a = ap.parse_args()
    if a.dump_c1:
        return dump_c1(*a.dump_c1)
    torch, T, M = setup(a.tree)
    b = real_batch(torch, M=M)
    r = {"batch": {"source": FRAMES, "windows": int(b["zmean"].shape[0]), "F": F_WIN, "note": "zmean and act are REAL val frames; v is PLANTED (12 m/s at the window start + the floored integration of the real accelerations): that file carries no speed"}}
    r["control1_off_identity"] = control1(a.tree, a.base, a.out)
    r["control1b_zero_weights_and_mutation"] = control1b_and_mutation(torch, T, M, b)
    r["control2_reachability"] = control2(torch, T, M, b)
    r["control3_speed_values"] = control3(torch, T, M, b)
    r["control4_echo_guard"] = control4(torch, T, M, b)
    r["control5_launch"] = control5(torch, T, M, a.argvs)
    r["all_pass"] = bool(r["control1_off_identity"]["pass_off_identity"] and all(v["pass"] for v in r["control1b_zero_weights_and_mutation"].values()) and r["control2_reachability"]["pass"]
                         and r["control3_speed_values"]["pass"] and r["control4_echo_guard"]["pass"] and r["control5_launch"]["pass"])
    json.dump(r, open(os.path.join(a.out, "CONTROLS_EGOWM.json"), "w"), indent=1, default=str)
    print(json.dumps({k: (v.get("pass") if isinstance(v, dict) and "pass" in v else (v if not isinstance(v, dict) else "...")) for k, v in r.items()}, default=str))


if __name__ == "__main__":
    main()
