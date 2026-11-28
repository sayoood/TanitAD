"""AMEND_HWMPLAN_6, section 3: STREAMED pull + level-1 latents of the 859 new TRAIN clips of clips6.json (dev-box CPU; box 2 is READ through sftp, nothing runs there).
The per-clip logic is `collect_latents_train_R35.py` (module C0, imported unchanged: ClipWins, window_starts, pull, control4); only the clip list, the scratch folder and the output (chunks of 50 clips, each with its sha256) differ.
  python pull_latents6.py run [--max-clips N]     resumable: clips already in CHUNKS.json are skipped; a crash loses at most the open chunk
  python pull_latents6.py selfcheck               imports, models, ClipStore on the filtered manifest, window starts for one clip; touches no box-2 file
Outputs (amend6/lat6/): chunk_NNN.pt (Z1 {R35_s0, R35_s1} (nw,16,128), A_raw, ok, t, eid, starts, pull_s, clips), CHUNKS.json (clips, sha256 per chunk, ctrl4, md5-verified clip count).
Env (as the original pull): CUDA_VISIBLE_DEVICES=-1, OMP_NUM_THREADS=2, WMACT_CODE_ROOT=F:/TanitAD-scratch/ladder_tree, PYTHONPATH = ladder_tree stack + taniteval + wmact + hwm_plan."""
import hashlib, json, os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CHUNK = 50
LAT6 = ROOT + "/lat6"
TS6 = ROOT + "/work/ts6"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def read_chunks():
    p = LAT6 + "/CHUNKS.json"
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else dict(chunks=[], ctrl4=None)


def write_chunks(J):
    tmp = LAT6 + "/CHUNKS.json.tmp"
    json.dump(J, open(tmp, "w", encoding="utf-8", newline="\n"), indent=1)
    os.replace(tmp, LAT6 + "/CHUNKS.json")


def prepare_ts(meta):
    """The scratch folder with a minimal manifest of the new clips (the original pull's layout: ts/cache/train/{manifest.jsonl, meta.json, clips/})."""
    d = TS6 + "/cache/train"
    os.makedirs(d + "/clips", exist_ok=True)
    keep = set(meta["clips"])
    with open(ROOT + "/work/manifest_box2.jsonl", encoding="utf-8") as fi, open(d + "/manifest.jsonl", "w", encoding="utf-8", newline="\n") as fo:
        for line in fi:
            if json.loads(line)["clip"] in keep:
                fo.write(line)
    json.dump(dict(purpose="AMEND_HWMPLAN_6 pull: minimal manifest of the new clips; each clip is streamed into clips/_current.pt", source="box2 D:/tanitad/cache_full/train", n_clips=len(keep)),
              open(d + "/meta.json", "w", encoding="utf-8", newline="\n"), indent=1)


def main():
    import numpy as np, torch
    sys.path.insert(0, "F:/TanitAD-scratch/hwm_plan")
    import collect_latents_train_R35 as C0
    import wmact_lib as W
    D = W.D
    C0.TS = TS6
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    maxc = int(sys.argv[sys.argv.index("--max-clips") + 1]) if "--max-clips" in sys.argv else 10 ** 9
    meta = json.load(open(ROOT + "/clips6.json", encoding="utf-8"))
    clips = meta["clips"]
    assert hashlib.sha256("\n".join(clips).encode()).hexdigest() == meta["clips_sha256"], "clips6.json does not hash to its record"
    os.makedirs(LAT6, exist_ok=True)
    prepare_ts(meta)
    models, cfgs = {}, {}
    for tag, path in {"R35_s0": "C:/Users/Admin/t2v8f_pulls/ladder_R35_s0/ckpt_last.pt", "R35_s1": "C:/Users/Admin/t2v8f_pulls/ladder_R35_s1/ckpt_last.pt"}.items():
        models[tag], cfgs[tag], _ = W.load_model(path)
    cfg = cfgs["R35_s0"]
    assert all(c.hmax == 91 and c.wm == "b" and c.kin_mode == "yaw" and c.hist == 4 and c.window_start_effective == 3 for c in cfgs.values())
    tracks_all = D.load_native_tracks(f"{W.SF}/inputs/native_tracks.pt")
    store = D.ClipStore(TS6 + "/cache", "train", lru=1, grid="pool4x10", kin_mode="yaw", tracks={"_current": None})
    dest_rel = "cache/train/clips/_current.pt"
    if mode == "selfcheck":
        c = clips[0]
        ts, valid = C0.window_starts(c, meta["n_frames"][c], cfg)
        print("selfcheck ok: clip", c[:8], "n_frames", meta["n_frames"][c], "valid", len(valid), "starts", len(ts), "in tracks", c in tracks_all, "md5 in manifest", meta["md5"][c][:8])
        return
    J = read_chunks()
    done = {c for ch in J["chunks"] for c in ch["clips"]}
    for ch in J["chunks"]:
        assert sha256_file(LAT6 + "/" + ch["file"]) == ch["sha256"], f"chunk {ch['file']} does not match CHUNKS.json"
    todo = [c for c in clips if c not in done][:maxc]
    t0 = time.time()
    buf = None
    c4buf = None
    for i0 in range(0, len(todo), CHUNK):
        group = todo[i0:i0 + CHUNK]
        S = {"Z": {k: [] for k in models}, "A": [], "OK": [], "T": [], "eid": [], "pull_s": [], "starts": {}}
        for c in group:
            pdt = C0.pull(c, meta["md5"][c], dest_rel)
            store.tracks.clear(); store.tracks["_current"] = tracks_all[c]
            store._c.clear()
            d = store.get("_current")
            assert len(d["pose"]) == meta["n_frames"][c], (c, len(d["pose"]), meta["n_frames"][c])
            ts, valid = C0.window_starts(c, meta["n_frames"][c], cfg)
            cw = C0.ClipWins(d, ts, 16)
            Z, A, OK, T = W.collect_latents(models, cw, len(ts))
            for k in models:
                S["Z"][k].append(Z[k])
            S["A"].append(A); S["OK"].append(OK); S["T"].append(T); S["eid"] += [c] * len(ts); S["pull_s"].append(pdt); S["starts"][c] = [int(x) for x in ts]
            if J["ctrl4"] is None:
                f = torch.from_numpy(np.stack([cw.item(0, int(t))["feats"] for t in ts]))
                c4buf = (f, A.clone()) if c4buf is None else (torch.cat([c4buf[0], f]), torch.cat([c4buf[1], A]))
                if c4buf[0].shape[0] >= 64:
                    J["ctrl4"] = C0.control4(models, c4buf[0][:64], c4buf[1][:64])
                    bad = {k: v["max_abs_diff"] for k, v in J["ctrl4"].items() if v["max_abs_diff"] != 0.0}
                    if bad:
                        raise RuntimeError(f"control 4 reads {bad}, expected 0.0 at both chassis")
                    print("control 4:", {k: v["max_abs_diff"] for k, v in J["ctrl4"].items()}, flush=True)
            print(f"clip {len(done) + 1:4d}/{len(clips)} {c[:8]} win={len(ts)} pull={pdt:.1f}s total={time.time() - t0:.0f}s", flush=True)
            done.add(c)
        n = len(J["chunks"])
        out = {"Z1": {k: torch.cat(v) for k, v in S["Z"].items()}, "A_raw": torch.cat(S["A"]), "ok": torch.cat(S["OK"]), "t": np.concatenate(S["T"]), "eid": S["eid"], "starts": S["starts"], "pull_s": S["pull_s"], "clips": group}
        fn = f"chunk_{n:03d}.pt"
        torch.save(out, LAT6 + "/" + fn + ".tmp"); os.replace(LAT6 + "/" + fn + ".tmp", LAT6 + "/" + fn)
        J["chunks"].append(dict(file=fn, clips=group, n_windows=len(S["eid"]), sha256=sha256_file(LAT6 + "/" + fn), md5_verified=len(group)))
        write_chunks(J)
        print(f"ZZ CHUNK {fn} {len(group)} clips ZZ  total {len(done)} clips, {time.time() - t0:.0f}s", flush=True)
    print("ZZ PULL6 DONE ZZ", len(done), flush=True)


if __name__ == "__main__":
    main()
