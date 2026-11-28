"""OPTB read-chain loading route. Thin wrapper; edits NO landed file.

Why: an FT checkpoint (trainer 4c396aff) carries cfg fields the ladder tree's V8FMiniConfig does not know, and with wm_rsp_readout=fixed_k the model BUILD loads the frozen probe from cfg.wm_rsp_probe,
a BOX-2 path. The route is:  WMACT_CODE_ROOT = OPTB_TREE (F:/TanitAD-scratch/actrsp/optb/tree, blob-equal to the stack/taniteval of 86da79cd (the CUDA-fixed built_rsp; 4c396aff manifest kept as tree_manifest.4c396aff.sha256); main() re-checks tree_manifest.sha256 on every start)
plus install(): wraps tanitad.models.v8f_mini.load_rsp_probe so a probe path that does not exist is remapped to the local copy of the SAME BASENAME, and the ORIGINAL function then still checks the file's
sha256 against cfg.wm_rsp_probe_sha256 (never skipped; a wrong-sha probe refuses). The remap itself also refuses a basename not in the allow-list and a local copy whose sha differs from the declared one.

Use as a launcher so the patch is in process for the planner and for ia3pk_read:  python optb_route.py <script.py> [args...]   (runpy; the planner must run --workers 1, spawned workers would not carry the patch)."""
import hashlib, os, sys
sys.dont_write_bytecode = True

OPTB_TREE = os.environ.get("OPTB_TREE", "F:/TanitAD-scratch/optb3_read/tree")
PROBE_DIRS = [d for d in os.environ.get("OPTB_PROBE_DIRS", "F:/TanitAD-scratch/actrsp/amend1/out;C:/Users/Admin/optb_pkg/frozen/inputs").split(";") if d]
ALLOWED = ("readout_K10_R35_s0.pt", "readout_K10_R35_s1.pt")
HERE = os.path.dirname(os.path.abspath(__file__))
STATE = {"remaps": []}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def env_setup():
    """Set WMACT_CODE_ROOT and the import path BEFORE anything imports tanitad / wmact_lib. Idempotent."""
    os.environ["WMACT_CODE_ROOT"] = OPTB_TREE
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    for p in (OPTB_TREE + "/taniteval", OPTB_TREE + "/stack"):
        if p not in sys.path:
            sys.path.insert(0, p)


def local_probe(path):
    base = os.path.basename(str(path).replace("\\", "/"))
    if base not in ALLOWED:
        raise ValueError(f"OPTB route: probe basename {base!r} is not one of {ALLOWED}: refused")
    for d in PROBE_DIRS:
        c = os.path.join(d, base)
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(f"OPTB route: no local copy of {base} in {PROBE_DIRS}")


def install():
    env_setup()
    import tanitad
    assert os.path.abspath(tanitad.__file__).replace("\\", "/").lower().startswith(os.path.abspath(OPTB_TREE).replace("\\", "/").lower()), tanitad.__file__
    from tanitad.models import v8f_mini as M
    if getattr(M, "_optb_route_installed", False):
        return M
    orig = M.load_rsp_probe

    def routed(path, sha256, dim):
        p = path
        if not os.path.isfile(path):
            p = local_probe(path)
            STATE["remaps"].append({"declared": str(path), "local": p, "declared_sha256": sha256})
        return orig(p, sha256, dim)          # the original re-hashes the file and raises on a mismatch: the sha is never skipped
    M.load_rsp_probe = routed
    M._optb_route_installed = True
    return M


def load_model(path):
    """As wmact_lib.load_model but STRICT: every key present, nothing unexpected except the legacy hacc_w buffer is tolerated NOWHERE here (an FT/R35 checkpoint must load exactly)."""
    import torch
    M = install()
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**ck["cfg"])
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    r = model.load_state_dict(ck["model"], strict=False)
    if r.missing_keys or r.unexpected_keys:
        raise RuntimeError(f"OPTB route strict load: missing {list(r.missing_keys)[:5]} unexpected {list(r.unexpected_keys)[:5]}")
    return model.eval(), cfg, ck


def check_tree_manifest():
    """tree_manifest.sha256 (written at build: sha256 + relpath of every .py under tree/stack and tree/taniteval). Refuses on any difference."""
    mp = os.path.join(HERE, "tree_manifest.sha256")
    want = {}
    for ln in open(mp, encoding="utf-8"):
        s, r = ln.rstrip("\n").split("  ", 1)
        want[r] = s
    got = {}
    for sub in ("stack", "taniteval"):
        for dp, dn, fn in os.walk(os.path.join(OPTB_TREE, sub)):
            dn[:] = [d for d in dn if d not in ("__pycache__", ".pytest_cache")]
            for f in fn:
                if f.endswith(".py"):
                    full = os.path.join(dp, f)
                    got[os.path.relpath(full, OPTB_TREE).replace("\\", "/")] = full
    if set(got) != set(want):
        raise RuntimeError(f"tree manifest: file set differs (extra {sorted(set(got) - set(want))[:3]}, missing {sorted(set(want) - set(got))[:3]})")
    bad = [r for r, p in got.items() if sha256_file(p) != want[r]]
    if bad:
        raise RuntimeError(f"tree manifest: {len(bad)} files differ, e.g. {bad[:3]}")
    return len(got)


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    if "--workers" in argv and int(argv[argv.index("--workers") + 1]) > 1:
        raise SystemExit("OPTB route: --workers > 1 spawns processes that do not carry the probe remap; use --workers 1")
    check_tree_manifest()   # MM 2026-11-18: the docstring promised a re-check on every start, but only test_route.py called it; the tree is 86da79cd (blob-verified), manifest regenerated
    install()
    import runpy
    script = argv[1]
    sys.argv = [script] + argv[2:]
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    runpy.run_path(script, run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
