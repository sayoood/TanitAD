#!/usr/bin/env python3
"""bake_hacc.py -- PREREG_V8F_HACC committed next step 2 (part 1): bake the scored HACC weights into the deployable v8F-mini checkpoints.
For each seed: the SCORED checkpoint (sha256 pinned, the one HBLEND / HACC read) -> V8FMini -> load_hacc_weights(the banked JSON, its canonical
hash pinned by test_v8f_hacc_infer) -> torch.save({**ck, "model": state_dict incl. hacc_w, "hacc": provenance}). Then it VERIFIES, on the
cached real eval clip's windows (the fixture test_v8f_hacc_infer uses):
  1. every source tensor is carried bit for bit, plus exactly ONE new key, hacc_w, equal to the JSON table in float32;
  2. the baked file loads through the eval harness's own call (strict load_state_dict, v8f_mini_eval.py) and forward_infer equals the source
     checkpoint with the weights set at runtime, bit for bit, on every output tensor;
  3. control: the source checkpoint still loads with hacc_w None and its plan is the plain O plan (it differs from the baked one: power);
  4. control: the OTHER seed's weights give a different plan (the bake carries the right table);
  5. the baked plan equals the banked HACC plan on those windows within the float noise bound test_v8f_hacc_infer pins (< 5e-4 m).
CPU only.  python bake_hacc.py <repo export root> <out dir>"""
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True
ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
for p in (ROOT / "stack", ROOT / "taniteval", ROOT / "stack" / "scripts", ROOT / "stack" / "tests"):
    sys.path.insert(0, str(p))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import test_v8f_hacc_infer as T  # noqa: E402  (its fixtures, weights paths, canonical hashes and the eval-batch builder)
from tanitad.models import v8f_mini as M  # noqa: E402

SHA = {"M2b_s0": "601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8",     # the scored checkpoints (v8f_hblend.py SHA, HSPEC's)
       "M2b_s1": "6104fb15dc21d524b6ed53be632b8d53ec5413526ac587734c22684c7e50c2e6"}
POST_MAP_COMMIT = "ba27d62ddd5780c6965cdc17a589b1552221b9c4"
torch.set_num_threads(4)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def canon(w):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(w, np.float64)).tobytes()).hexdigest()


def build(ck):
    m = M.V8FMini(M.V8FMiniConfig(**ck["cfg"]), M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    m.load_state_dict(ck["model"])                       # strict, exactly as v8f_mini_eval.py:263
    return m.eval()


def infer(m, b):
    with torch.no_grad():
        return {k: v for k, v in m.forward_infer(b).items() if torch.is_tensor(v)}


def same(a, b):
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


assert T.HAVE_REAL and all(T.WEIGHTS.values()), "fixtures or banked weights missing"
OUT.mkdir(parents=True, exist_ok=True)
rep = {"prereg": "Project Steering/PREREG_V8F_HACC.md", "post_map_commit": POST_MAP_COMMIT, "seeds": {}}
for seed in T.SEEDS:
    other = [s for s in T.SEEDS if s != seed][0]
    src = T.CKPT[seed]
    assert sha256(src) == SHA[seed], f"{seed}: not the scored checkpoint"
    wj = Path(T.WEIGHTS[seed])
    w = M.read_hacc_weights(wj)
    assert canon(w) == T.WEIGHTS_SHA[seed], f"{seed}: the weights table is not the scored one"
    ck = torch.load(src, map_location="cpu", weights_only=False)
    m_src = build(ck)
    assert m_src.hacc_w is None
    m_bake = build(ck)
    m_bake.load_hacc_weights(wj)
    sd = m_bake.state_dict()
    prov = {"prereg": rep["prereg"], "result": "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/RESULT_V8F_HACC.md",
            "weights_json": wj.relative_to(ROOT).as_posix(), "weights_json_sha256": sha256(wj), "weights_canonical_sha256": canon(w),
            "source_ckpt_sha256": SHA[seed], "post_map_commit": POST_MAP_COMMIT, "tier_of_the_scored_result": "T1-class open loop",
            "what": "forward_infer returns P + w (O - P) with this [30, 2] table (column 0 along, column 1 cross); training (_forward) is untouched"}
    out = OUT / f"{seed}_hacc.pt"
    assert not out.exists(), f"{out} exists"
    torch.save({**ck, "model": sd, "hacc": prov}, out)
    # 1. carried bit for bit + exactly one new key
    ck2 = torch.load(out, map_location="cpu", weights_only=False)
    extra = set(ck2["model"]) - set(ck["model"])
    assert extra == {"hacc_w"} and set(ck["model"]) <= set(ck2["model"]), extra
    assert all(torch.equal(ck2["model"][k], ck["model"][k]) for k in ck["model"]), "a source tensor changed"
    assert torch.equal(ck2["model"]["hacc_w"], torch.as_tensor(w, dtype=torch.float32)), "hacc_w is not the JSON table"
    assert {k: v for k, v in ck2.items() if k not in ("model", "hacc")}.keys() == {k for k in ck if k != "model"}
    # 2. the baked file through the harness's strict load == runtime-set weights, bit for bit
    m_file = build(ck2)
    assert m_file.hacc_active
    b, idx = T._real_clip_batch(m_file.cfg)
    y_file, y_rt, y_src = infer(m_file, b), infer(m_bake, b), infer(m_src, b)
    assert same(y_file, y_rt), "the baked checkpoint's forward_infer differs from the runtime-set weights"
    # 3. the source still loads plain, and the map has power on these windows
    assert set(y_src) | {"traj60_o", "traj30_o"} == set(y_file) and torch.equal(y_file["traj30_o"], y_src["traj30"])
    d_src = float((y_file["traj30"] - y_src["traj30"]).abs().max())
    assert d_src > 1e-2, d_src
    # 4. the other seed's table gives a different plan
    m_oth = build(ck)
    m_oth.load_hacc_weights(T.WEIGHTS[other])
    d_oth = float((infer(m_oth, b)["traj30"] - y_file["traj30"]).abs().max())
    assert d_oth > 1e-3, d_oth
    # 5. agreement with the banked HACC plan (the bound test_v8f_hacc_infer pins)
    hb = np.load(T.BANKED_HACC[seed])["HACC"][idx]
    d_bank = float(np.abs(y_file["traj30"].double().numpy() - hb).max())
    assert d_bank < 5e-4, d_bank
    r = {"out": out.name, "out_sha256": sha256(out), "out_bytes": out.stat().st_size, "n_windows_checked": int(len(idx)),
         "baked_vs_runtime": "bit-identical on every output tensor", "max_abs_baked_vs_plain_m": d_src,
         "max_abs_baked_vs_other_seed_table_m": d_oth, "max_abs_baked_vs_banked_hacc_m": d_bank, "provenance": prov}
    rep["seeds"][seed] = r
    print(seed, json.dumps({k: v for k, v in r.items() if k != "provenance"}), flush=True)
json.dump(rep, open(OUT / "BAKE_REPORT.json", "w"), indent=1)
print("ZZBAKE-OK")
