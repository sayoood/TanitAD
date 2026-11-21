"""Local md5 of the refcv8 eval-side inputs, keyed by the THOR-relative path (sam3_gt_v3_eval -> sam3_gt_v3)."""
import hashlib, os, sys
K6 = "E:/refcv6_eval_kit/data"      # the existing kit (matched files are reused from here)
K8 = "E:/refcv8_eval_kit/data"      # copies of the NEW refcv8 files
names = [ln.split()[1].lstrip("*") for ln in open("E:/refcv8_eval_kit/kit_checks/thor_md5_script.names", encoding="utf-8") if ln.strip()]
def local(rel):
    if rel.startswith("sam3_gt_v3/"):
        return K6 + "/sam3_gt_v3_eval/" + rel[len("sam3_gt_v3/"):]
    if os.path.exists(K8 + "/" + rel):
        return K8 + "/" + rel
    return K6 + "/" + rel
def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()
out = open("E:/refcv8_eval_kit/kit_checks/local_md5.txt", "w", encoding="utf-8")
err = open("E:/refcv8_eval_kit/kit_checks/local_md5.err", "w", encoding="utf-8")
for rel in names:
    p = local(rel)
    try:
        out.write(f"{md5(p)}  {rel}  <= {p}\n"); out.flush()
    except Exception as e:
        err.write(f"{rel}: {type(e).__name__}: {e}\n"); err.flush()
out.write("ZZLOCALDONEZZ\n"); out.close(); err.close()
