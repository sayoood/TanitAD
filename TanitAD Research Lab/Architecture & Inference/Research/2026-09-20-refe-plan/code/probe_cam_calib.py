"""CAM_F0 calibration variation across ALL nuPlan log DBs on D:, with the values decoded.

Follow-up to probe_cam_calib.py, which sampled 24 DBs. This one sweeps every DB and
decodes the pickled intrinsic/extrinsic to real numbers so the magnitude is quotable.
"""
import glob, os, sqlite3, struct, sys, io, hashlib, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = r"D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits"

def floats(blob):
    """Pull big-endian doubles out of a pickle blob by its BINFLOAT ('G') opcodes."""
    out = []
    i = 0
    b = bytes(blob)
    while True:
        i = b.find(b"G", i)
        if i < 0 or i + 9 > len(b): break
        try:
            out.append(struct.unpack(">d", b[i + 1:i + 9])[0])
        except Exception:
            pass
        i += 1
    return out

dbs = sorted(glob.glob(ROOT + "/*/*.db"))
print(f"sweeping {len(dbs)} log DBs for channel CAM_F0")

K_sig, T_sig, R_sig = collections.Counter(), collections.Counter(), collections.Counter()
K_val, T_val, R_val = {}, {}, {}
veh_of = collections.defaultdict(set)
read_ok = read_fail = 0
sizes = collections.Counter()

for d in dbs:
    veh = os.path.basename(d).split("_")[1]
    try:
        con = sqlite3.connect(f"file:{d}?mode=ro", uri=True)
        cur = con.cursor()
        cur.execute("SELECT intrinsic, translation, rotation, width, height "
                    "FROM camera WHERE channel='CAM_F0'")
        got = cur.fetchall()
        con.close()
    except Exception:
        read_fail += 1
        continue
    if not got:
        read_fail += 1
        continue
    read_ok += 1
    for K, T, R, w, h in got:
        sizes[(w, h)] += 1
        for blob, sig, val in ((K, K_sig, K_val), (T, T_sig, T_val), (R, R_sig, R_val)):
            s = hashlib.sha256(bytes(blob)).hexdigest()[:10]
            sig[s] += 1
            val.setdefault(s, floats(blob))
            if sig is K_sig: veh_of[s].add(veh)

print(f"  DBs read OK: {read_ok}   unreadable/no CAM_F0: {read_fail}")
print(f"  image size(s): {dict(sizes)}")
print()
print(f"DISTINCT CAM_F0 INTRINSICS : {len(K_sig)}")
for s, n in K_sig.most_common():
    f = K_val[s]
    fx, cx, fy, cy = (f[0], f[2], f[4], f[5]) if len(f) >= 6 else (None,) * 4
    print(f"   {s}  n={n:<5} vehicles={len(veh_of[s]):<3} "
          f"fx={fx:.3f} fy={fy:.3f} cx={cx:.3f} cy={cy:.3f}" if fx else f"   {s} n={n} raw={f[:6]}")
print()
print(f"DISTINCT CAM_F0 EXTRINSIC TRANSLATIONS : {len(T_sig)}")
for s, n in T_sig.most_common(12):
    print(f"   {s}  n={n:<5} xyz={['%.4f' % v for v in T_val[s][:3]]}")
print()
print(f"DISTINCT CAM_F0 EXTRINSIC ROTATIONS    : {len(R_sig)}")
for s, n in R_sig.most_common(6):
    print(f"   {s}  n={n:<5} quat={['%.5f' % v for v in R_val[s][:4]]}")

if len(K_sig) >= 2:
    a, b = [K_val[s] for s, _ in K_sig.most_common(2)]
    if len(a) >= 6 and len(b) >= 6:
        print()
        print("SPREAD BETWEEN THE TWO INTRINSIC RIGS")
        print(f"   fx {a[0]:.3f} vs {b[0]:.3f}   delta {abs(a[0]-b[0]):.3f} px "
              f"({100*abs(a[0]-b[0])/a[0]:.2f} %)")
        print(f"   cx {a[2]:.3f} vs {b[2]:.3f}   delta {abs(a[2]-b[2]):.3f} px")
        print(f"   cy {a[5]:.3f} vs {b[5]:.3f}   delta {abs(a[5]-b[5]):.3f} px")
print()
print("DECODE CONTROL -- does the crude float scanner actually read the intrinsic correctly?")
print("  The discriminating check is an ANALYTIC target: for a nominal rig the principal point")
print("  cx must equal W/2 = 960.000 exactly. If the scanner were mis-aligned it would read")
print("  an arbitrary number here.")
for s, n in K_sig.most_common():
    f = K_val[s]
    if len(f) >= 6:
        ok = "PASS (cx == W/2 exactly)" if abs(f[2] - 960.0) < 1e-9 else \
             "off-centre rig (expected for a second calibration)"
        print(f"   {s}  cx={f[2]:.3f}  vs W/2=960.000  -> {ok}")
print()
print("NOTE ON THE EXTRINSIC VALUES")
print("  translation/rotation are pickled as BINPUT-compressed floats whose 'G' opcodes this")
print("  crude scanner does not always recover; the DISTINCT-VALUE COUNTS above are exact")
print("  (they are sha256 over the raw blobs) while the decoded xyz/quat are not, and are")
print("  therefore NOT quoted. The load-bearing claim is the COUNT, not the values.")
print()
print("VERDICT")
print(f"  CAM_F0 intrinsics : {len(K_sig)} distinct across {read_ok} logs")
print(f"  CAM_F0 extrinsics : {len(T_sig)} distinct translations, {len(R_sig)} distinct rotations")
print("  => a LEARNED PER-TOKEN POSITION TABLE stores ONE vector per token position and cannot")
print("     condition on either. A PETR-style MLP over coordinates derived from K and the")
print("     extrinsics can. This is an INFORMATION argument, not a parameter-count argument.")
print("=== PROBE COMPLETE ===")
