import msgpack, numpy as np, json

PATH = "/home/nvidia/nurec_work/x/volume.msgpack"
with open(PATH, "rb") as f:
    raw = f.read()
d = msgpack.unpackb(raw, raw=False, strict_map_key=False)
nre = d["nre_data"]
sd = nre["state_dict"]
cfg = nre["config"]

LAYERS = ["background", "road", "dynamic_rigids", "dynamic_deformables"]
# component counts derived from config, to be CHECKED against byte counts
EXPECT = {
    "positions": 3,
    "rotations": 4,
    "scales": 3,
    "densities": 1,
    "features_albedo": None,  # fourier_features_dim * 3
    "features_specular": 45,  # (sph_degree=3 -> 16 coeffs, minus DC) * 3
    "camera_extra_signal": 20,
}

print("### DTYPE / SHAPE DERIVATION (fp16 hypothesis) ###")
for L in LAYERS:
    pos = sd[f".gaussians_nodes.{L}.positions"]
    assert len(pos) % 6 == 0, (L, len(pos))
    N = len(pos) // 6
    ff = cfg["layers"][L]["fourier_features_dim"]
    print(f"\n--- layer {L}: N = {N}  (positions {len(pos)} B / 6)   fourier_features_dim={ff}")
    for name in [
        "positions",
        "rotations",
        "scales",
        "densities",
        "features_albedo",
        "features_specular",
        "camera_extra_signal",
    ]:
        k = f".gaussians_nodes.{L}.{name}"
        if k not in sd:
            print(f"    {name}: ABSENT")
            continue
        b = sd[k]
        if len(b) == 0:
            print(f"    {name}: len 0")
            continue
        per = len(b) / N
        comps = per / 2.0
        exp = EXPECT[name]
        if name == "features_albedo":
            exp = ff * 3
        tag = "OK" if exp is not None and abs(comps - exp) < 1e-9 else f"EXPECTED {exp}"
        a = np.frombuffer(b, dtype=np.float16).astype(np.float32)
        print(
            f"    {name:22s} bytes={len(b):>10d} B/gauss={per:6.2f} "
            f"fp16-comps={comps:6.2f} [{tag}]  "
            f"min={a.min():+11.4g} max={a.max():+11.4g} mean={a.mean():+10.4g} "
            f"nan={int(np.isnan(a).sum())} inf={int(np.isinf(a).sum())}"
        )

print("\n\n### ACTIVATION EVIDENCE ###")
for L in LAYERS:
    N = len(sd[f".gaussians_nodes.{L}.positions"]) // 6
    P = np.frombuffer(sd[f".gaussians_nodes.{L}.positions"], dtype=np.float16).astype(np.float32).reshape(N, 3)
    R = np.frombuffer(sd[f".gaussians_nodes.{L}.rotations"], dtype=np.float16).astype(np.float32).reshape(N, 4)
    S = np.frombuffer(sd[f".gaussians_nodes.{L}.scales"], dtype=np.float16).astype(np.float32).reshape(N, 3)
    D = np.frombuffer(sd[f".gaussians_nodes.{L}.densities"], dtype=np.float16).astype(np.float32).reshape(N)
    print(f"\n=== {L}  N={N}")
    for nm, arr in [("pos", P), ("scale_raw", S)]:
        q = np.percentile(arr, [0, 1, 50, 99, 100], axis=0)
        print(f"  {nm} pct0/1/50/99/100 per-axis:\n{np.array2string(q, precision=4, suppress_small=True)}")
    nrm = np.linalg.norm(R, axis=1)
    print(
        f"  |quat|  min={nrm.min():.4f} p1={np.percentile(nrm,1):.4f} "
        f"med={np.median(nrm):.4f} p99={np.percentile(nrm,99):.4f} max={nrm.max():.4f}  "
        f"frac in [0.9,1.1]={np.mean((nrm>0.9)&(nrm<1.1)):.4f}"
    )
    print(f"  scale exp() -> m: p1={np.exp(np.percentile(S,1)):.5f} med={np.exp(np.median(S)):.5f} p99={np.exp(np.percentile(S,99)):.5f}")
    hist, edges = np.histogram(D, bins=[-1e9, -100, -20, -5, -1, 0, 1, 5, 20, 100, 1e9])
    print(f"  density_raw hist  edges={[-1e9,-100,-20,-5,-1,0,1,5,20,100,1e9]}\n              counts={hist.tolist()}")
    op = 1.0 / (1.0 + np.exp(-D.astype(np.float64)))
    print(
        f"  sigmoid(density): min={op.min():.5f} med={np.median(op):.5f} max={op.max():.5f} "
        f"frac>0.005={np.mean(op>0.005):.4f} frac>0.5={np.mean(op>0.5):.4f}"
    )
    A = np.frombuffer(sd[f".gaussians_nodes.{L}.features_albedo"], dtype=np.float16).astype(np.float32)
    ff = cfg["layers"][L]["fourier_features_dim"]
    A = A.reshape(N, -1)
    print(f"  albedo shape {A.shape}  first-3-cols min/med/max: "
          f"{A[:,:3].min():+.4f}/{np.median(A[:,:3]):+.4f}/{A[:,:3].max():+.4f}")
    if A.shape[1] > 3:
        print(f"           rest-cols min/med/max: {A[:,3:].min():+.4f}/{np.median(A[:,3:]):+.4f}/{A[:,3:].max():+.4f}")
    # 3DGS DC convention test: color = 0.5 + C0*dc  with C0 = 0.28209479
    C0 = 0.28209479177387814
    col = 0.5 + C0 * A[:, :3]
    print(f"  0.5+C0*albedo[:, :3] -> min={col.min():+.4f} med={np.median(col):+.4f} max={col.max():+.4f} "
          f"frac in [0,1]={np.mean((col>=0)&(col<=1)):.4f}")
    sig = 1.0 / (1.0 + np.exp(-A[:, :3].astype(np.float64)))
    print(f"  sigmoid(albedo[:, :3])   -> med={np.median(sig):+.4f} frac in [0.02,0.98]={np.mean((sig>0.02)&(sig<0.98)):.4f}")
    print(f"  raw albedo[:, :3] frac in [0,1]={np.mean((A[:,:3]>=0)&(A[:,:3]<=1)):.4f}")
    SP = np.frombuffer(sd[f".gaussians_nodes.{L}.features_specular"], dtype=np.float16).astype(np.float32).reshape(N, -1)
    print(f"  specular shape {SP.shape} min={SP.min():+.4f} med={np.median(SP):+.4f} max={SP.max():+.4f} "
          f"absmean={np.abs(SP).mean():.5f}")

print("\n\n### post_processing config ###")
print(json.dumps(cfg["post_processing"], indent=1, default=str)[:2000])
print("\n### strategy ###")
print(json.dumps(cfg["strategy"], indent=1, default=str)[:1200])
print("\n### renderer/projection+culling ###")
print(json.dumps({k: cfg["renderer"][k] for k in ("projection", "culling", "render", "antialiasing", "outputs")}, indent=1, default=str)[:2500])
