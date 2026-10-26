"""Extract HACC's float64 causal-bucket m10 dumps from the banked tar (E:/.../t1kin_causalbucket_dumps_f64_2026-10-26.tar), IN MEMORY, verify the
manifest's sha256_f64_concat (consolidate.py's formula), and write ONE compressed float64 npz per seed.  Read-only on the tar."""
import hashlib, io, json, tarfile, numpy as np
TAR = "E:/Projects/TanitAD-artifacts/v8f-mini-hacc/t1kin_causalbucket_dumps_f64_2026-10-26.tar"
MAN = json.load(open("C:/Users/Admin/hmc/inputs/banked_causal/dumps_manifest.json")) if False else None
man = json.load(open("C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/t1kin/consolidated_c/dumps_manifest.json"))
t = tarfile.open(TAR)
out = {}
for s in (0, 1):
    tag = f"HACC_s{s}_m10_causal"
    mem = sorted(n for n in t.getnames() if n.startswith(f"dump_{tag}/ep") and n.endswith(".npz"))
    zs = [np.load(io.BytesIO(t.extractfile(n).read())) for n in mem]
    keys = zs[0].files
    d = {k: np.concatenate([z[k] for z in zs]) for k in keys}
    h = hashlib.sha256(b"".join(np.ascontiguousarray(d[k]).tobytes() for k in sorted(keys))).hexdigest()
    ok = h == man[tag]["sha256_f64_concat"]
    print(tag, len(mem), "files", len(d["eid"]), "windows", "sha256_f64_concat", h[:16], "== manifest" if ok else "!= MANIFEST", {k: str(v.dtype) for k, v in d.items()})
    assert ok
    np.savez_compressed(f"C:/Users/Admin/hmc/inputs/banked_causal_f64/dumps_{tag}_f64.npz", **d)
