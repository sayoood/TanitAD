"""CPU tests for the YF2 package (CUDA_VISIBLE_DEVICES=-1). Real u400 decoder weights, SYNTHETIC cache (random scene_ctx, yaw-dependent targets): these tests prove the MECHANISM
(gates, reach, identity, refusals, checkpoint composition), never an accuracy. Every gate has its deliberate-regression arm and the test REQUIRES it to be RED.
    python test_yaw_ft.py            # prints one line per check: PASS / RED-as-required / FAIL
"""
import gzip
import json
import math
import os
import sys
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import yaw_common as YC
import yaw_ft as FT

SCRATCH = Path(os.environ.get("YF2_TEST_DIR", str(HERE / "_test_scratch")))
N, LOGS = 96, 12
RES = []


def check(name, ok, detail=""):
    RES.append((name, bool(ok), detail))
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)
    return ok


def make_synth(d: Path, seed=7, drop_table=0.0):
    """cache + manifest + table. Targets bend with the yaw: y(t) += 0.5 * omega * t^2 (so the column HAS something to learn)."""
    g = np.random.default_rng(seed)
    d.mkdir(parents=True, exist_ok=True)
    scene = (g.standard_normal((N, 64, 256)) * 0.5).astype(np.float32)
    bits = torch.from_numpy(scene).to(torch.bfloat16).view(torch.int16).numpy()
    bits.tofile(d / "shard_0000.scene.bin")
    open(d / "shard_0000.visual.bin", "wb").close()
    frames, rows, mframes, table = [], [], [], {}
    t = np.arange(1, 21) * 0.2
    for i in range(N):
        log, tok = f"log{i % LOGS:02d}", f"{i:016x}"
        om = float(g.normal(0, 0.12))
        table[tok] = om
        v = float(g.uniform(4, 12))
        ego = [v, 0.0, float(g.normal(0, .5)), float(g.normal(0, .5)), 0.0, 0.0, v]
        for rk in (0, 1):
            traj = np.stack([v * t, 0.5 * om * v * t ** 2 * 0.5 + 0.1 * rk, om * t], 1).tolist()
            rows.append({"key": [log, tok, 0], "rank": rk, "ego": ego, "goal": [20.0, 0.0, 40.0, 0.0], "traj": traj})
        frames.append({"key": [log, tok, 0], "shard": 0, "index": i, "vindex": -1, "scene_sha256": "0"})
        mframes.append({"log_name": log, "token": tok, "step": 0, "ranks": {"0": {}, "1": {}}, "yf2_split": "VAL" if int(log[3:]) % 4 == 0 else "TRAIN"})
    if drop_table:
        for k in list(table)[:max(1, int(len(table) * drop_table))]:
            del table[k]
    json.dump({"dtype": "bf16"}, open(d / "manifest.json", "w"))
    open(d / "frames.jsonl", "w").write("".join(json.dumps(f) + "\n" for f in frames))
    open(d / "rows.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in rows))
    json.dump({"frames": mframes}, open(d / "manifest_syn.json", "w"))
    meta = {"schema": "ego-yaw-table-v1", "field": "ego_pose.angular_rate_z", "n_tokens": len(table)}
    with gzip.GzipFile(filename="", mode="wb", fileobj=open(d / "table.json.gz", "wb"), mtime=0) as gz:
        gz.write(json.dumps({"meta": meta, "table": table}).encode())
    return d / "manifest_syn.json", d / "table.json.gz"


def run(d, manifest, table, arm, out, extra=()):
    argv = ["--cache", str(d), "--manifest", str(manifest), "--yaw-table", str(table), "--arm", arm, "--out", str(out), "--gate-frames", "64", "--batch", "16", "--accum", "2",
            "--val-every", "10", *extra]
    try:
        return FT.main(argv)
    except Exception as e:
        return type(e).__name__


def main():
    import shutil
    shutil.rmtree(SCRATCH, ignore_errors=True)          # an arm is never overwritten: every run starts from an empty scratch
    d = SCRATCH / "synth"
    man, tab = make_synth(d)
    # --- G4/G5 green on B and C, RED on the two deliberate-regression arms
    r = run(d, man, tab, "B", SCRATCH / "g_B", ["--gates-only"])
    g = json.load(open(SCRATCH / "g_B" / "gates.json"))
    check("G4 identity (zero-init column: yaw0 AND yaw!=0 decode == original u400, torch.equal)", r == 0 and g["G4_pass"], f"rc {r}")
    check("G5 reach (grad col4 > 0 with table; EXACTLY 0.0 with yaw=0)", g["G5_pass"] and g["G5_gradabs_zero_control"] == 0.0, f"table {g['G5_gradabs_table']:.4g} control {g['G5_gradabs_zero_control']}")
    r = run(d, man, tab, "B", SCRATCH / "g_dc", ["--gates-only", "--regress", "default_col"])
    g = json.load(open(SCRATCH / "g_dc" / "gates.json"))
    check("RED-AS-REQUIRED G4: default random column left in place", r == 3 and not g["G4_pass"], f"rc {r} identity_yaw_nonzero {g['G4_identity_yaw_nonzero']}")
    r = run(d, man, tab, "B", SCRATCH / "g_tn", ["--gates-only", "--regress", "table_not_applied"])
    g = json.load(open(SCRATCH / "g_tn" / "gates.json"))
    check("RED-AS-REQUIRED G5: table loaded but not applied", r == 3 and not g["G5_pass"], f"rc {r} grad_table {g['G5_gradabs_table']}")
    d2 = SCRATCH / "synth_miss"
    man2, tab2 = make_synth(d2, drop_table=0.02)
    r = run(d2, man2, tab2, "B", SCRATCH / "g_miss", ["--gates-only"])
    check("RED-AS-REQUIRED G9: 2% of tokens missing from the table -> TableRefused", r == "TableRefused", str(r))
    # --- train B and C (R1), same seed/steps: B learns, C is exactly A0
    args = ["--steps", "24", "--lr-col", "3e-2", "--seed", "0"]
    rb, rc = run(d, man, tab, "B", SCRATCH / "armB", args), run(d, man, tab, "C", SCRATCH / "armC", args)
    sb, sc = json.load(open(SCRATCH / "armB" / "summary.json")), json.load(open(SCRATCH / "armC" / "summary.json"))
    fb, fc = torch.load(SCRATCH / "armB" / "final.pt", weights_only=False), torch.load(SCRATCH / "armC" / "final.pt", weights_only=False)
    check("B trains (rc 0), col4 moved off zero", rb == 0 and sb["col4_norm"] > 0, f"|col4| {sb['col4_norm']:.3f}")
    check("C (yaw=0, same code/argv/seed/steps): col4 stays EXACTLY 0 and no tensor differs from u400", rc == 0 and float(fc["col4"].abs().sum()) == 0.0 and fc["changed_keys"] == [], f"changed {fc['changed_keys']}")
    check("B changes only ego_enc.0.weight (R1)", fb["changed_keys"] == ["ego_enc.0.weight"], str(fb["changed_keys"]))
    base = torch.load(YC.U400, map_location="cpu", weights_only=False)["model"]["ego_enc.0.weight"]
    W = fb["state"]["ego_enc.0.weight"]
    oth = [i for i in range(W.shape[1]) if i != 4]
    check("B differs from u400 ONLY in column 4 of ego_enc.0.weight", torch.equal(W[:, oth], base[:, oth]) and not torch.equal(W[:, 4], base[:, 4]))
    check("B uses the yaw: VAL loss with yaw < VAL loss with yaw zeroed (mechanism reaches the loss)", sb["val_final"]["loss"] < sb["val_final_yaw0"]["loss"], f"{sb['val_final']['loss']:.4f} < {sb['val_final_yaw0']['loss']:.4f}")
    check("C is indifferent to yaw (VAL loss identical with/without)", sc["val_final"]["loss"] == sc["val_final_yaw0"]["loss"])
    rb2 = run(d, man, tab, "B", SCRATCH / "armB_rep", args)
    f2 = torch.load(SCRATCH / "armB_rep" / "final.pt", weights_only=False)
    check("G8 determinism: B rerun (same seed) col4 torch.equal on CPU", rb2 == 0 and torch.equal(f2["col4"], fb["col4"]))
    # --- checkpoint composition gates
    import build_yaw_ckpt as BC
    base_sd = torch.load(YC.U400, map_location="cpu", weights_only=False)
    st, meta, fed = BC.compose(base_sd, fb, str(YC.U400))
    check("B ckpt meta: ego_yaw/ego_yaw_scale/slot/table sha set", meta.get("ego_yaw") == YC.YAW_TAG and meta.get("ego_yaw_scale") == 10.0 and meta.get("ego_yaw_slot") == 4 and fed)
    st_c, meta_c, fed_c = BC.compose(base_sd, fc, str(YC.U400))
    check("C ckpt meta carries NO ego_yaw keys; state_dict torch.equal to u400 (C == A0 bit for bit)", "ego_yaw" not in meta_c and not fed_c and all(torch.equal(st_c[k], base_sd["model"][k]) for k in st_c))
    # train-time decode == composed-checkpoint decode (same F.linear on the same values)
    cache = FT.Cache(d)
    rows = FT.load_rows(cache, json.load(open(man)))["VAL"][:48]
    tbl, _, _ = YC.load_yaw_table(tab)
    sc_, eg, go, tg = FT.make_batch(rows, list(range(len(rows))), cache, tbl, "table", YC.YAW_SCALE)
    model, cfg, _ = FT.build_decoder(str(YC.U400))
    dec = FT.YawDecoder(model, cfg, fb["col4"])
    with torch.no_grad():
        t_train = dec(sc_, eg, go)
        model.load_state_dict({k: v for k, v in st.items() if k.startswith(YC.DEC)}, strict=False)
        t_ckpt = FT.original_decode(model, cfg, sc_, eg, go)
    check("G-CKPT: trainer decode == decode of the composed checkpoint's own ego_enc (torch.equal)", torch.equal(t_train, t_ckpt))
    # K gates RED arms
    bad = dict(fb)
    bad["state"] = dict(fb["state"])
    bad["state"]["queries"] = fb["state"]["queries"] + 1e-3
    st_b, _, _ = BC.compose(base_sd, bad, str(YC.U400))
    changed = [k for k in st_b if not torch.equal(st_b[k], base_sd["model"][k])]
    check("RED-AS-REQUIRED K2: a tampered extra tensor (queries) in an R1 arm is detected", set(changed) != {"ego_enc.0.weight"}, str(changed))
    # --- scaled shim (G7)
    sys.path.insert(0, str(YC.YT_REFE))
    import ego_yaw as EY
    sh = EY.EgoYawShim(tbl, "x" * 64, meta["ego_yaw_scale"])          # YT's shim, scale from the checkpoint meta
    ego = torch.zeros(3, 7)
    toks = [f"{i:016x}" for i in (3, 5, 7)]
    sh.set_token(toks)
    o = sh.apply(ego)
    want = torch.tensor([10.0 * tbl[t] for t in toks], dtype=torch.float32)
    check("G7 YT shim: ego[4] == float32(meta scale 10 * table) exactly, the SAME arithmetic as yaw_ft.make_batch, other slots untouched", torch.equal(o[:, 4], want) and torch.equal(o[:, [0, 1, 2, 3, 5, 6]], ego[:, [0, 1, 2, 3, 5, 6]]))
    for name, fn in (("token not in table", lambda: (sh.set_token(["ffffffffffffffff"]), sh.apply(ego[:1]))), ("non-zero incoming ego[4]", lambda: (sh.set_token(toks[:1]), sh.apply(torch.ones(1, 7))))):
        try:
            fn()
            ok = False
        except RuntimeError:
            ok = True
        check(f"RED-AS-REQUIRED G7 YT shim REFUSES: {name}", ok)
    n_fail = sum(1 for _, ok, _ in RES if not ok)
    print(f"\n{len(RES) - n_fail}/{len(RES)} checks as required")
    print("ZZYF2_TESTS_OK" if n_fail == 0 else "ZZYF2_TESTS_FAIL")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())

# also pinned: YAW_SCALE agreement between YF2 and YT (a disagreement raises inside import_yt_scale)
