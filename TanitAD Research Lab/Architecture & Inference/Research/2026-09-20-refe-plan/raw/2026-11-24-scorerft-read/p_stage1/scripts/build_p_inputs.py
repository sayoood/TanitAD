"""MM 2026-11-25: assemble P's read inputs in read_p/pod/ from VERIFIED copies, then write PULLED.txt LAST.

usage: build_p_inputs.py <hf dir (holds scorerft/p/...)> <pod sha file> <weight rule local> <out pod dir>
<pod sha file> lines: "<sha256>  <pod path>" (sha256sum output) plus "MODEL <sha256>" (the pod's model_final_sha256_SFTP.txt)
and "LFS scorerft/p/model_final.pt <sha256>" (HF's LFS record).
Every HF copy must equal the pod's sha256, and HF's LFS sha256 of model_final.pt must equal the pod's model sha. The weight rule must equal
the census pin ef1c36bf.... Any mismatch exits non-zero and writes NO PULLED.txt, so the read keeps waiting rather than reading a wrong file.
argv_p.json and w0_p.json are derived from P's config.json in the same form as Stage 1's argv_c.json / w0_c.json.
"""
import hashlib, json, os, shutil, sys
from datetime import datetime, timezone

WRULE_P_SHA256 = "ef1c36bf4071f8a667432a2198e531c1611c9cd29f9eaa0706a6af166f8a4e1f"
RUN = "/workspace/data/refe_runs/scorerft_p"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(hf, shafile, wrule, out):
    pod, model, lfs = {}, None, None
    for ln in open(shafile, encoding="utf-8"):
        t = ln.split()
        if len(t) >= 2 and t[0] == "MODEL":
            model = t[1]
        elif len(t) >= 3 and t[0] == "LFS":
            lfs = t[2]
        elif len(t) == 2 and len(t[0]) == 64:
            pod[t[1].lstrip("*")] = t[0]
    fails, rec = [], {}
    for name in ("config.json", "summary.json", "metrics.jsonl"):
        local = os.path.join(hf, "scorerft", "p", name)
        want = pod.get(f"{RUN}/{name}")
        got = sha256(local) if os.path.exists(local) else None
        rec[name] = {"hf": got, "pod": want}
        if not want or got != want:
            fails.append(f"{name}: HF {str(got)[:12]} != pod {str(want)[:12]}")
    if not (model and lfs and len(model) == 64 and model == lfs):
        fails.append(f"model_final.pt: HF LFS {str(lfs)[:12]} != pod {str(model)[:12]}")
    ws = sha256(wrule) if os.path.exists(wrule) else None
    if ws != WRULE_P_SHA256:
        fails.append(f"weight_rule_rank_p.json {str(ws)[:12]} != the census pin {WRULE_P_SHA256[:12]}")
    if fails:
        raise SystemExit("REFUSING (no PULLED.txt): " + "; ".join(fails))
    cfg_path = os.path.join(hf, "scorerft", "p", "config.json")
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    os.makedirs(out, exist_ok=True)
    shutil.copyfile(cfg_path, os.path.join(out, "config_p.json"))
    shutil.copyfile(os.path.join(hf, "scorerft", "p", "summary.json"), os.path.join(out, "summary.json"))
    shutil.copyfile(os.path.join(hf, "scorerft", "p", "metrics.jsonl"), os.path.join(out, "metrics_p.jsonl"))
    shutil.copyfile(wrule, os.path.join(out, "weight_rule_rank_p.json"))
    src = f"P's config.json argv (HF scorerft/p/config.json sha256 {rec['config.json']['hf']})"
    open(os.path.join(out, "argv_p.json"), "w", encoding="utf-8").write(json.dumps({"argv": cfg["argv"], "source": src}, indent=1) + "\n")
    ws_ = cfg.get("warm_start") or {}
    w0 = {"step0_digest": ws_.get("sha256"), "source": "SFTP config.json warm_start.sha256 (the checkpoint train.py loaded at step 0)",
          "path": ws_.get("path"), "tensors": ws_.get("tensors")}
    open(os.path.join(out, "w0_p.json"), "w", encoding="utf-8").write(json.dumps(w0) + "\n")
    if not os.path.exists(os.path.join(out, "clip_probe.json")):
        raise SystemExit("REFUSING (no PULLED.txt): clip_probe.json is not in " + out)
    lines = [f"PULLED {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} by build_p_inputs.py (MM)"]
    for f in sorted(os.listdir(out)):
        if f != "PULLED.txt":
            lines.append(f"{sha256(os.path.join(out, f))}  {f}")
    lines.append(f"verified: HF == pod sha256 for config/summary/metrics; HF LFS model_final.pt == pod {model}; weight rule == census pin")
    open(os.path.join(out, "PULLED.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")   # LAST
    print("\n".join(lines))


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
