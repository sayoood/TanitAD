"""PREREG_V8F_WMPOOL read side (READSIDE_WMPOOL.md items 1, 2 and 5): controls of the cell path in ladder_read/lr_lib.py, ladder_read/run_ladder_read.py and ladder_plantest/lp_lib.py,
each with a deliberate-regression arm that must go RED. CPU only, 2 threads. Built models with RANDOM weights (the identities below hold for any adapter and any query), on 2 REAL eval clips.
Usage:  python controls_readside.py --tree <export of abc01481 (stack, taniteval)> --orig <dir holding the unedited lr_lib.py and lp_lib.py> --out <json>
  R1  lr_lib.zcell_of == zcells.zcell_of (the box-2 cell builder)                                  max abs, want <= 1e-6
  R2  reader z1 (encode_clip on z_in cells) == the TRAINING path wm.z1(adapter tokens)             max abs, want <= 1e-5, with a non-zero query (non-uniform weights)
  R3  encode_clip states at every level == stack.encode(wm.z1(tokens), scaled actions)             max abs, want <= 1e-5
  R4  lp_lib.clip_z1 (P-ATT) == wm.z1(tokens)                                                       max abs, want <= 1e-5
  R5  a MEAN model reads exactly as before: new encode_clip / clip_z1 == the unedited files          torch.equal, also when the frames dict carries cells
  R6  refusals: P-ATT on frames without cells; P-ATT given a token mean; a cells file whose adapter hash differs; a matching cells file splits to [T, 10, d]
  R7  pool_stats: zero query -> entropy == ln 10 (1e-6, fp32 weights); the random query -> entropy < ln 10
  RED arms: is_pattn forced False (R2 must fail); z_in chosen by the FILE (R5 must fail); a cell builder with the 2 x 2 blocks transposed (R1 and R2 must fail)
"""
import argparse, importlib.util, json, math, os, sys, tempfile, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
THEORY = os.path.normpath(os.path.join(HERE, ".."))


def load_orig(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def mx(a, b):
    return float((a.float() - b.float()).abs().max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--orig", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["LADDER_READ_CODE_ROOT"] = a.tree
    sys.path.insert(0, HERE)
    import controls_wmpool as CW
    torch, T, M = CW.setup(a.tree)
    CW.need_ram(float(os.environ.get("RS_NEED_GB", "3.0")))                         # two small random-weight models and two clips: well under 1 GB measured (wall line in the json)
    sys.path.insert(0, os.path.join(THEORY, "ladder_read")), sys.path.insert(0, os.path.join(THEORY, "ladder_plantest"))
    import lr_lib as L
    import lp_lib as LP
    import zcells as Z
    LO = load_orig(os.path.join(a.orig, "lr_lib.py"), "lr_lib_orig")
    PO = load_orig(os.path.join(a.orig, "lp_lib.py"), "lp_lib_orig")
    for mod in (L, LP, LO, PO):
        assert os.path.normcase(mod.__file__) != "", mod
    _, c_att = CW.cfg_for(T, ["--wm-pool", "att25"])
    _, c_mean = CW.cfg_for(T, [])
    m_att = CW.full_model(torch, M, c_att, seed=0).eval()
    m_mean = CW.full_model(torch, M, c_mean, seed=0).eval()
    assert L.is_pattn(m_att) and not L.is_pattn(m_mean)
    g = torch.Generator().manual_seed(11)
    with torch.no_grad():
        m_att.wm.pool_q.copy_(3.0 * torch.randn(m_att.wm.pool_q.shape, generator=g))   # a non-zero query: the weights are not uniform, so R2 tests the attention, not the mean
    # two REAL eval clips (feats [T, 40, d_feat]) and their logged actions, as build_eval_frames reads them
    tracks = L.D.load_native_tracks(f"{L.SF}/inputs/native_tracks.pt")
    store = L.D.ClipStore(f"{L.SF}/cache", "eval", lru=2, grid="pool4x10", kin_mode="yaw", tracks=tracks)
    clips = L.list_eval_clips()[:2]
    feats, acts = [], []
    for c in clips:
        d = store.get(c)
        feats.append(torch.as_tensor(d["feats"]))
        acts.append(np.stack([np.nan_to_num(d["a"]), d["ch1"]], -1).astype(np.float32))
    out = {"tier": "T0 instrument (no model statistic)", "tree": a.tree, "clips": clips, "T": [int(f.shape[0]) for f in feats], "feats_dtype": str(feats[0].dtype)}

    def tokens(model, f):
        with torch.no_grad():
            return model.adapter(f.float()[None]) + model.pos[None, None]                # [1, T, 40, d], V8FMini._encode

    def frames_dict(model, with_cells):
        fr = {"clips": clips, "T": [int(f.shape[0]) for f in feats], "act": acts, "zmean": [L.zmean_of(model, f).numpy() for f in feats]}
        if with_cells:
            fr["zcell"] = [L.zcell_of(model, f).numpy() for f in feats]
        return fr

    def r1(cell_fn):
        return max(mx(cell_fn(m_att, f), Z.zcell_of(m_att, f)) for f in feats)

    def r2r3r4():
        fr = frames_dict(m_att, True)
        e2 = e3 = e4 = 0.0
        z1c = LP.clip_z1(m_att.wm, fr)
        for ci, f in enumerate(feats):
            with torch.no_grad():
                z1_train = m_att.wm.z1(tokens(m_att, f))
                enc_ref = m_att.wm.stack.encode(z1_train, m_att.wm.scale_actions(torch.as_tensor(acts[ci])[None])[:, :-1])
            z, _ = L.encode_clip(m_att, L.z_in(m_att, fr, ci), acts[ci])
            with torch.no_grad():
                z1_read = m_att.wm.z1_from_cells(torch.as_tensor(L.z_in(m_att, fr, ci))[None])
            e2 = max(e2, mx(z1_read, z1_train))
            e3 = max(e3, max(mx(zl, rl) for zl, rl in zip(z, enc_ref.z)))
            e4 = max(e4, mx(z1c[clips[ci]], z1_train[0]))
        return e2, e3, e4

    out["R1_zcell_lr_vs_box2_builder_max_abs"] = r1(L.zcell_of)
    e2, e3, e4 = r2r3r4()
    out["R2_reader_z1_vs_training_z1_max_abs"], out["R3_encode_states_vs_training_max_abs"], out["R4_lp_clip_z1_vs_training_max_abs"] = e2, e3, e4
    st = L.pool_stats(m_att, frames_dict(m_att, True))
    out["R2_weights_not_uniform"] = {"entropy_mean": st["entropy_mean"], "max_weight_mean": st["max_weight_mean"], "q_norm": st["q_norm"]}

    # R5: the mean model, new files vs the unedited ones (bit-identical), with and without cells in the frames dict
    eq = True
    for with_cells in (False, True):
        fr = frames_dict(m_mean, with_cells)
        n_new, n_old = LP.clip_z1(m_mean.wm, fr), PO.clip_z1(m_mean.wm, fr)
        eq &= all(torch.equal(n_new[c], n_old[c]) for c in clips)
        for ci in range(len(clips)):
            zn, pn = L.encode_clip(m_mean, L.z_in(m_mean, fr, ci), acts[ci])
            zo, po = LO.encode_clip(m_mean, fr["zmean"][ci], acts[ci])
            eq &= all(torch.equal(x, y) for x, y in zip(zn, zo)) and all(torch.equal(x, y) for x, y in zip(pn, po))
            for ctr in L.CONTRAST.values():
                zn, _ = L.encode_clip(m_mean, L.z_in(m_mean, fr, ci), acts[ci], contrast=ctr)
                zo, _ = LO.encode_clip(m_mean, fr["zmean"][ci], acts[ci], contrast=ctr)
                eq &= all(torch.equal(x, y) for x, y in zip(zn, zo))
    out["R5_mean_model_bit_identical_to_unedited"] = bool(eq)

    # R6 refusals and the cells-file check
    ref = {}
    def refuses(fn):
        try:
            fn()
            return False
        except L.LadderMismatch:
            return True
    fr_nc = frames_dict(m_att, False)
    ref["patt_without_cells"] = refuses(lambda: L.z_in(m_att, fr_nc, 0))
    ref["patt_given_token_mean"] = refuses(lambda: L.encode_clip(m_att, fr_nc["zmean"][0], acts[0]))
    tmp = tempfile.mkdtemp(prefix="rs_", dir=os.environ.get("RS_TMP", None))
    try:
        sha = L.adapter_sha(m_att)
        Tn = np.array([int(f.shape[0]) for f in feats])
        fp = os.path.join(tmp, "frames.npz")
        np.savez(fp, clips=np.array(clips), T=Tn, zmean=np.concatenate(fr_nc["zmean"]), act=np.concatenate(acts), ok=np.ones(int(Tn.sum()), bool), adapter_sha=sha)
        zc =np.concatenate([L.zcell_of(m_att, f).numpy() for f in feats])
        good, bad = os.path.join(tmp, "cells.npz"), os.path.join(tmp, "cells_bad.npz")
        np.savez(good, clips=np.array(clips), T=Tn, zcell=zc, adapter_sha=sha)
        np.savez(bad, clips=np.array(clips), T=Tn, zcell=zc, adapter_sha="0" * 64)
        ref["cells_file_wrong_adapter_hash"] = refuses(lambda: L.load_eval_frames(fp, cells=bad))
        frc = L.load_eval_frames(fp, cells=good)
        ref["matching_cells_file_shapes"] = [list(x.shape) for x in frc["zcell"]]
        ref["matching_cells_file_ok"] = bool([x.shape[0] for x in frc["zcell"]] == list(Tn) and all(x.shape[1:] == (10, 256) for x in frc["zcell"]))
        ref["no_cells_arg_has_no_zcell"] = "zcell" not in L.load_eval_frames(fp)
    finally:
        for f in os.listdir(tmp):
            os.remove(os.path.join(tmp, f))
        os.rmdir(tmp)
    out["R6"] = ref

    # R7 entropy
    m_zero = CW.full_model(torch, M, c_att, seed=0).eval()
    s0 = L.pool_stats(m_zero, frames_dict(m_zero, True))
    out["R7_zero_query_entropy"], out["R7_random_query_entropy"] = s0["entropy_mean"], st["entropy_mean"]
    # 1e-6: the weights are fp32 by design (pool_weights), so 0.1 is 0.100000001; the first run used 1e-9 and failed on exactly this (measured 1.94e-8, 2026-11-11 ~07:12)
    r7 = abs(s0["entropy_mean"] - math.log(10)) <= 1e-6 and st["entropy_mean"] < math.log(10) - 1e-3

    # RED arms (each must make its check fail)
    red = {}
    keep = L.is_pattn
    L.is_pattn = lambda m: False
    try:
        try:
            e2b, _, _ = r2r3r4()
            red["is_pattn_forced_false_R2_red"] = bool(e2b > 1e-5)
        except Exception as e:                                                  # z1_from_mean refuses on a P-ATT model, or z_in hands the mean
            red["is_pattn_forced_false_R2_red"] = True
            red["is_pattn_forced_false_raised"] = f"{type(e).__name__}: {str(e)[:160]}"
    finally:
        L.is_pattn = keep
    keep_z = L.z_in
    L.z_in = lambda model, fr, ci: fr["zcell"][ci] if "zcell" in fr else fr["zmean"][ci]
    try:
        fr = frames_dict(m_mean, True)
        try:
            zn, _ = L.encode_clip(m_mean, L.z_in(m_mean, fr, 0), acts[0])
            zo, _ = LO.encode_clip(m_mean, fr["zmean"][0], acts[0])
            red["z_in_by_file_R5_red"] = not all(torch.equal(x, y) for x, y in zip(zn, zo))
        except Exception as e:
            red["z_in_by_file_R5_red"] = True
            red["z_in_by_file_raised"] = f"{type(e).__name__}: {str(e)[:160]}"
    finally:
        L.z_in = keep_z

    def transposed(model, f):
        t = tokens(model, f)[0]
        Fn, N, d = t.shape
        return t.reshape(Fn, 2, 2, 5, 2, d).mean(dim=(1, 4)).reshape(Fn, 10, d)          # blocks over the wrong axes
    red["transposed_blocks_R1_red"] = bool(r1(transposed) > 1e-6)
    keep_c = L.zcell_of
    L.zcell_of = transposed
    try:
        e2t, _, _ = r2r3r4()
        red["transposed_blocks_R2_red"] = bool(e2t > 1e-5)
    finally:
        L.zcell_of = keep_c
    out["RED"] = red
    out["pass"] = bool(out["R1_zcell_lr_vs_box2_builder_max_abs"] <= 1e-6 and e2 <= 1e-5 and e3 <= 1e-5 and e4 <= 1e-5 and out["R5_mean_model_bit_identical_to_unedited"]
                       and ref["patt_without_cells"] and ref["patt_given_token_mean"] and ref["cells_file_wrong_adapter_hash"] and ref["matching_cells_file_ok"] and ref["no_cells_arg_has_no_zcell"]
                       and r7 and all(v for k, v in red.items() if k.endswith("_red")))
    out["wall_s"] = round(time.time() - t0, 1)
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: out[k] for k in out if k.startswith(("R", "pass", "wall"))}, indent=1))
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
