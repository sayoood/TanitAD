"""CAPLADDER R0 stage (IMPL_CHOICES section 4). DINOv3-L patch tokens 16 x 40 re-encoded from the bank's own source frames with the cache's own encoder path
(enc.encode_episode imported from the hash-checked module; local HF cache; bf16), the rows of set D stored as fp16 in one memmap, and control 4 (R0 pooled to 8 x 20 / 4 x 10 against the
cached fp8 pool8x20 / pool4x10) computed on the fly for ALL 9,596 bank frames. Each source episode is deleted once its rows are written. GPU only if no other compute job holds it.
Writes F:/TanitAD-scratch/capladder_feat/r0_fp16.bin, r0_progress.jsonl, R0_STATUS.json, control4_capladder.json, control4_arrays.npz.  Exit 0 also when R0 is NOT RUN (status says why)."""
import hashlib, importlib.util, json, os, re, shutil, subprocess, sys, time
sys.dont_write_bytecode = True
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import capladder_lib as CL

FRAMES = os.environ.get("CAPLADDER_FRAMES", "F:/TanitAD-scratch/capladder_frames")
FEAT = os.environ.get("CAPLADDER_FEAT", "F:/TanitAD-scratch/capladder_feat")
SMOKE = os.environ.get("CAPLADDER_SMOKE") == "1"          # smoke: only the episodes present in CAPLADDER_FRAMES, no all-bank aggregation
CACHE = "E:/Projects/TanitAD-artifacts/sfresh_inputs/cache/eval/clips"
ENC = "E:/Projects/TanitAD/stack/scripts/dinov3_fp8_encode_ship.py"
ENC_SHA = "361bc287c306a8f40092ec95f6d3b06562ef32de75c9f5ea6235091196c93b48"
MID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
D0 = 640 * 1024
JOB_PAT = r"(python|ollama|llama|colab|torch)"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def gpu_free_rule():
    """(ok, detail): no python / ollama / llama / colab / torch process in nvidia-smi's compute-apps list (WDDM desktop applications appear there with no memory and are not jobs), >= 3 GB free VRAM."""
    try:
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], capture_output=True, text=True, timeout=60).stdout.strip().splitlines()
        mem = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:
        return False, {"error": repr(e)}
    held = [a for a in apps if re.search(JOB_PAT, a.split(",")[1] if len(a.split(",")) > 1 else a, re.I)]
    free_mb = float(mem.split(",")[0])
    return (not held and free_mb >= 3000), {"compute_apps_listed": len(apps), "held_by_jobs": held, "free_vram_mb": free_mb, "apps": apps}


def status(d):
    d["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(d, open(f"{FEAT}/R0_STATUS.json", "w"), indent=1)
    print("[r0]", json.dumps({k: v for k, v in d.items() if k != "gpu"}), flush=True)


def control4_from_progress(done, beid, bg):
    """Control 4 over all bank frames from the per-episode progress records: (summary dict, per-frame array [n_bank, 4] = err8, err4, eq8, eq4)."""
    key = {}
    for clip, j in done.items():
        for k, g in enumerate(j["g"]):
            key[(clip, g)] = (j["err8"][k], j["err4"][k], j["eq8"][k], j["eq4"][k])
    E = np.array([key[(str(beid[b]), int(bg[b]))] for b in range(len(beid))])
    e8, e4 = E[:, 0], E[:, 1]
    q = lambda a: {f"p{p}": float(np.percentile(a, p)) for p in (50, 90, 99, 99.9, 100)}
    c4 = {"n_bank_frames": int(len(beid)), "tol": 0.08, "need_share": 0.99,
          "share_ok_8x20": float((e8 <= 0.08).mean()), "share_ok_4x10": float((e4 <= 0.08).mean()), "err_8x20": q(e8), "err_4x10": q(e4),
          "mean_elementwise_bit_equal_8x20": float(E[:, 2].mean()), "mean_elementwise_bit_equal_4x10": float(E[:, 3].mean()),
          "frames_fully_bit_equal_8x20": float((E[:, 2] == 1.0).mean()), "frames_fully_bit_equal_4x10": float((E[:, 3] == 1.0).mean()),
          "pass": CL.control4_pass(e8, e4)}
    return c4, E


def main():
    os.makedirs(FEAT, exist_ok=True)
    S = np.load(os.path.join(HERE, "sets_capladder.npz"), allow_pickle=False)
    beid, bg, rowsD, nD = S["beid"], S["bg"], S["rowsD"], int(S["nD"])
    if os.path.exists(f"{FEAT}/control4_capladder.json"):
        print("[r0] already done", flush=True)
        return 0
    if not os.path.exists(f"{FRAMES}/pull_done.json"):
        status({"status": "NOT RUN", "reason": "frames not reachable: pull_done.json absent"})
        return 0
    ok, det = gpu_free_rule()
    if not ok:
        status({"status": "NOT RUN", "reason": "GPU held or too little free VRAM", "gpu": det})
        return 0
    assert sha(ENC) == ENC_SHA, "encoder module differs from the cache's encoder_module_sha256"
    try:
        import truststore  # noqa: F401
    except Exception:
        import types
        m_ = types.ModuleType("truststore")
        m_.inject_into_ssl = lambda: None
        sys.modules["truststore"] = m_
    spec = importlib.util.spec_from_file_location("dinov3_fp8_encode_ship", ENC)
    enc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(enc)
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained(MID, local_files_only=True)           # pull_encode_pooled.py build_model, :225-236
    model = AutoModel.from_pretrained(MID, local_files_only=True, dtype=torch.bfloat16).eval().to("cuda")
    mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1).to("cuda", torch.bfloat16)
    std = torch.tensor(proc.image_std).view(1, 3, 1, 1).to("cuda", torch.bfloat16)
    n_special = 1 + getattr(model.config, "num_register_tokens", 0)
    assert n_special == 5 and tuple(proc.image_mean) == (0.485, 0.456, 0.406)
    path = f"{FEAT}/r0_fp16.bin"
    r0 = np.memmap(path, dtype=np.float16, mode="r+" if os.path.exists(path) else "w+", shape=(nD, D0))
    clips = sorted(set(beid.tolist()))
    if SMOKE:
        clips = [c for c in clips if os.path.exists(f"{FRAMES}/{c}.v2ep.pt")]
    rowD_of = {}                                                                        # (clip, g) -> row in D
    for i, b in enumerate(rowsD):
        rowD_of[(str(beid[b]), int(bg[b]))] = i
    bank_of = {}
    for b in range(len(beid)):
        bank_of.setdefault(str(beid[b]), []).append(b)
    prog_path = f"{FEAT}/r0_progress.jsonl"
    done = {}
    if os.path.exists(prog_path):
        for l in open(prog_path, encoding="utf-8"):
            if l.strip():
                j = json.loads(l)
                done[j["clip"]] = j
    status({"status": "RUNNING", "gpu": det, "episodes_done": len(done)})
    t0 = time.time()
    for clip in clips:
        if clip in done:
            continue
        f_src = f"{FRAMES}/{clip}.v2ep.pt"
        o = torch.load(f_src, map_location="cpu", weights_only=False)
        f = enc.encode_episode(o, model, mean, std, n_special, "cuda")                  # fp32 [T, 640, 1024]
        c = torch.load(f"{CACHE}/{clip}.pt", map_location="cpu", weights_only=True)
        c8, c4 = c["pool8x20"], c["pool4x10"]
        assert f.shape[0] == c8.shape[0] == c4.shape[0], (clip, f.shape, c8.shape, c4.shape)
        rec = {"clip": clip, "T": int(f.shape[0]), "g": [], "err8": [], "err4": [], "eq8": [], "eq4": []}
        for b in bank_of[clip]:
            g = int(bg[b])
            p8, p4 = CL.pool_field(f[g:g + 1], 8, 20), CL.pool_field(f[g:g + 1], 4, 10)
            rec["g"].append(g)
            rec["err8"].append(float(CL.rel_l2(p8, c8[g:g + 1].float())[0]))
            rec["err4"].append(float(CL.rel_l2(p4, c4[g:g + 1].float())[0]))
            rec["eq8"].append(float((p8.to(torch.float8_e4m3fn).view(torch.uint8) == c8[g:g + 1].view(torch.uint8)).float().mean()))
            rec["eq4"].append(float((p4.to(torch.float8_e4m3fn).view(torch.uint8) == c4[g:g + 1].view(torch.uint8)).float().mean()))
        for g in range(f.shape[0]):
            i = rowD_of.get((clip, g))
            if i is not None:
                r0[i] = f[g].reshape(-1).half().numpy()
        r0.flush()
        rec["secs"] = round(time.time() - t0, 1)
        with open(prog_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        done[clip] = rec
        del o, f, c
        os.remove(f_src)                                                                # the source episode is deleted once its rows are banked in the memmap
        print(f"[r0] {len(done)}/{len(clips)} {clip[:8]} T={rec['T']} err8 max {max(rec['err8']):.4f} err4 max {max(rec['err4']):.4f} {time.time() - t0:.0f}s", flush=True)
    if SMOKE:
        e8 = np.concatenate([done[c]["err8"] for c in clips]); e4 = np.concatenate([done[c]["err4"] for c in clips])
        print("[r0-smoke] episodes", len(clips), "err8 max/mean", e8.max(), e8.mean(), "err4 max/mean", e4.max(), e4.mean(), "bit-equal frac 8x20", float(np.mean([np.mean(done[c]["eq8"]) for c in clips])), flush=True)
        return 0
    c4, E = control4_from_progress(done, beid, bg)
    e8, e4 = E[:, 0], E[:, 1]
    np.savez(f"{FEAT}/control4_arrays.npz", err8=e8, err4=e4, eq8=E[:, 2], eq4=E[:, 3])
    json.dump(c4, open(f"{FEAT}/control4_capladder.json", "w"), indent=1)
    status({"status": "OK", "episodes": len(done), "control4_pass": c4["pass"], "seconds": round(time.time() - t0, 1)})
    print("[r0] control 4:", json.dumps(c4), flush=True)
    if len(done) == len(clips):
        shutil.rmtree(FRAMES, ignore_errors=True)                                       # all 141 encoded: delete the (now empty) frames folder, manifest and pull marker
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception as e:                                                              # a crash is a NOT RUN with its reason, never a silent half result
        import traceback
        traceback.print_exc()
        os.makedirs(FEAT, exist_ok=True)
        status({"status": "NOT RUN", "reason": "r0_capladder.py crashed: " + repr(e)[:300]})
        rc = 0
    sys.exit(rc)
