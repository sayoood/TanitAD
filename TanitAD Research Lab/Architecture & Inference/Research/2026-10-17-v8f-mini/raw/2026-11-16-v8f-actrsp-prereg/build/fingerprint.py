"""OFF-equals-tip fingerprint (ACTRSP build). Usage: python fingerprint.py <out.json> [--trainer]
Builds the default (flag absent) WMAux on the tree named by PYTHONPATH, records per-config: state_dict sha256, loss terms (float.hex), gradient sha256, built_levers, built_rollout.
--trainer adds a flag-absent ladder-path --selfcheck-only on the synthetic recipe (live.terms, key sets)."""
import sys, os, json, hashlib, tempfile
import torch
torch.set_num_threads(1)
import tanitad
from tanitad.models import v8f_mini as M

def cfg_of(**kw):
    return M.V8FMiniConfig(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6, wm=kw.pop("wm", "b"), hmax=kw.pop("hmax", 28), kin_mode="yaw", **kw)

def sha_t(t):
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()

def fp(cfg):
    torch.manual_seed(0)
    wm = M.WMAux(cfg)
    sd = wm.state_dict()
    r = {"sd_keys": list(sd), "sd_sha": hashlib.sha256("".join(k + sha_t(v) for k, v in sd.items()).encode()).hexdigest()}
    g = torch.Generator().manual_seed(7)
    z, a = torch.randn(2, cfg.hmax, cfg.wm_dim, generator=g), torch.randn(2, cfg.hmax - 1, 2, generator=g)
    out = wm.stack.loss(z, a, generator=torch.Generator().manual_seed(11))
    out["total"].backward()
    r["terms"] = {k: float(v.detach()).hex() for k, v in out.items()}
    r["grad_sha"] = hashlib.sha256("".join(n + (sha_t(p.grad) if p.grad is not None else "none") for n, p in wm.named_parameters()).encode()).hexdigest()
    r["built_levers"] = json.loads(json.dumps(wm.stack.built_levers(), default=str))
    r["built_rollout"] = json.loads(json.dumps(wm.stack.built_rollout(cfg.hmax), default=str))
    return r

if __name__ == "__main__":
    out = {"tanitad_file": tanitad.__file__}
    out["plain"] = fp(cfg_of())
    out["ladder"] = fp(cfg_of(hmax=25, wm_strides=(2, 3), wm_rollout_k=(4, 3, 2), wm_rollout_stride=(2, 2, 1), wm_rollout_w=1.0, wm_ctx=(3, 3, 3)))
    out["wm_a"] = fp(cfg_of(wm="a"))
    if "--trainer" in sys.argv:
        sys.path.insert(0, os.path.join(os.path.dirname(tanitad.__file__), "..", "scripts"))
        import v8f_mini_train as TR
        BASE = ["--synthetic", "--batch", "16", "--d", "32", "--n-q", "2", "--kin-mode", "yaw", "--seed", "5", "--arm", "M2b", "--threads", "1", "--warmup", "2"]
        LADDER = ["--hmax", "25", "--wm-only", "--wm-strides", "2", "3", "--wm-rollout-k", "4", "3", "2", "--wm-rollout-stride", "2", "2", "1", "--wm-rollout-w", "1.0", "--wm-ctx", "3", "3", "3", "--wm-windows", "wmonly"]
        root = tempfile.mkdtemp(prefix="fp_", dir=os.environ.get("TMP"))
        assert TR.main(["--out", root + "/m2b", *BASE, "--log-every", "1", "--ckpt-every", "100", "--steps", "2"]) == 0
        rc = TR.main(["--out", root + "/l", *BASE, "--init-ckpt", root + "/m2b/ckpt_last.pt", *LADDER, "--selfcheck-only"])
        sc = json.load(open(root + "/l/selfcheck.json", encoding="utf-8"))
        out["trainer"] = {"rc": rc, "live_terms": {k: float(v).hex() for k, v in sc["live"]["terms"].items()}, "levers_keys": sorted(sc["levers"]), "sc_keys": sorted(sc), "live_keys": sorted(sc["live"]), "declared_vs_built": sc["declared_vs_built"],
                          "wm_groups_grad": {k: float(v).hex() for k, v in sc["live"]["wm_groups_grad"].items()}}
        import shutil; shutil.rmtree(root, ignore_errors=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    print("OK", sys.argv[1])
