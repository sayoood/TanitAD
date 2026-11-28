"""Prove ``integration/refc_wiring.patch`` before anyone applies it.

Run it from anywhere:

    python integration/verify_patch.py [--repo <root>] [--tmp <dir>]

``--repo`` defaults to the parent of this file's directory (so the script works
from a clone, a worktree or a checkout with no arguments); ``--tmp`` defaults to
``tempfile.mkdtemp()``. ⛔ **No path in this file is machine- or
session-specific.** An earlier version hard-coded the author's scratchpad, whose
directory name contains a UUID-shaped session id; the zero-id guard refused it,
correctly — nothing can tell a session UUID from a clip UUID by looking at it,
and neither belongs in a repo artifact. ``tests/test_refcv6_no_session_paths.py``
now scans this branch's files for that class and is mutation-proven.

## What it does

It never modifies the repo. It copies ``stack/tanitad/refs/refc.py`` into a
temporary tree, applies the patch there with ``git apply``, imports BOTH the
repo module and the patched copy in one process, and compares them:

===  =========================================================================
 A   with the coupling OFF, the parameter SET and every VALUE are identical
 B   forward on **64 fixed windows** is BIT-IDENTICAL
 C   with the coupling ON, the new parameters appear, the provenance is
     reported, ``n_points == n_steps``, and the SHARED parameters are still
     bit-identical (which is what proves the attach is the last statement of
     ``__init__``)
 D   the gates are zero-init, and a non-zero gate changes the output — GATED,
     not dead
 E   WP-B's own heads are bit-identical across BEV on/off, so the 2x2 ablation
     differs in the levers and in nothing else
===  =========================================================================

Exit code 0 means all five passed.
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REL_REFC = Path("stack") / "tanitad" / "refs" / "refc.py"
PATCH_NAME = "refc_wiring.patch"
N_WINDOWS = 64


def build_patched(repo: Path, patch: Path, tmp: Path) -> Path:
    """Apply the patch to a COPY of refc.py in ``tmp``; return the copy."""
    dst = tmp / REL_REFC
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(repo / REL_REFC, dst)
    r = subprocess.run(["git", "apply", "-p1", "--verbose", str(patch)],
                       cwd=str(tmp), capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"git apply FAILED in {tmp}:\n{r.stderr}{r.stdout}")
    return dst


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def build(mod, torch, seed=0, bev_cfg=None, wp=None, agents=False, d_bev=16):
    torch.manual_seed(seed)
    cfg = mod.refc_smoke_config()
    if agents:
        cfg.decoder.cross_agent = True
    if wp is not None:
        cfg.decoder.wp_index = wp
    if bev_cfg is not None:
        cfg.decoder.bev_coupling = bev_cfg
        cfg.decoder.bev_coupling_d_bev = int(d_bev)
    m = mod.RefCModel(cfg)
    m.eval()
    return m, cfg


def forward(m, cfg, torch, seed=0):
    torch.manual_seed(1234 + seed)
    b, w = 2, 1
    h, wid = cfg.encoder.image_hw()
    x = torch.randn(b, w, cfg.encoder.in_channels, h, wid)
    with torch.no_grad():
        return m(x, nav_cmd=torch.randint(0, 4, (b,)), v0=torch.rand(b) * 15.0)


def tensors(out) -> dict:
    return {k: v for k, v in out.items() if hasattr(v, "shape")} \
        if isinstance(out, dict) else {"out": out}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent,
                    help="repo/worktree root (default: the parent of integration/)")
    ap.add_argument("--patch", type=Path, default=None,
                    help=f"the patch (default: <repo>/integration/{PATCH_NAME})")
    ap.add_argument("--tmp", type=Path, default=None,
                    help="scratch directory (default: a fresh mkdtemp)")
    ap.add_argument("--keep", action="store_true", help="do not delete --tmp")
    a = ap.parse_args(argv)

    repo = a.repo.resolve()
    patch = (a.patch or repo / "integration" / PATCH_NAME).resolve()
    if not (repo / REL_REFC).is_file():
        raise SystemExit(f"no {REL_REFC} under {repo} — pass --repo")
    if not patch.is_file():
        raise SystemExit(f"no patch at {patch} — pass --patch")
    sys.path.insert(0, str(repo / "stack"))

    import torch                                                  # noqa: E402
    from tanitad.models.refc_bev_coupling import BEVCouplingConfig  # noqa: E402
    import tanitad.refs.refc as R0                                # noqa: E402
    import tanitad.refs.refc_wp_index as wpi                      # noqa: E402

    tmp = Path(a.tmp) if a.tmp else Path(tempfile.mkdtemp(prefix="refcpatch_"))
    tmp.mkdir(parents=True, exist_ok=True)
    print(f"repo  {repo}\npatch {patch}\ntmp   {tmp}")
    try:
        R1 = load_module(build_patched(repo, patch, tmp), "refc_patched")
        print(f"patch applies; patched module loaded ({R1.__name__})")

        d_bev = 16
        m0, c0 = build(R0, torch, 0)
        m1, c1 = build(R1, torch, 0, bev_cfg=None)
        p0, p1 = m0.state_dict(), m1.state_dict()
        assert set(p0) == set(p1), (
            f"parameter SET changed with the coupling off: "
            f"+{sorted(set(p1) - set(p0))[:5]} -{sorted(set(p0) - set(p1))[:5]}")
        bad = [k for k in p0 if not torch.equal(p0[k], p1[k])]
        print(f"A. coupling OFF: {len(p0)} tensors, {len(bad)} differ -> "
              f"{'BIT-IDENTICAL' if not bad else 'DIFFER: ' + str(bad[:5])}")
        assert not bad

        n_diff = 0
        for wnd in range(N_WINDOWS):
            x, y = tensors(forward(m0, c0, torch, wnd)), tensors(forward(m1, c1, torch, wnd))
            extra = set(y) - set(x)
            missing = set(x) - set(y)
            assert not missing, f"the patch DROPPED output keys: {sorted(missing)}"
            for k in x:
                if not torch.equal(x[k], y[k]):
                    n_diff += 1
                    print(f"   window {wnd} key {k}: max|d| "
                          f"{float((x[k] - y[k]).abs().max()):.3e}")
                    break
        print(f"B. {N_WINDOWS} fixed windows, coupling OFF: "
              f"{N_WINDOWS - n_diff}/{N_WINDOWS} BIT-IDENTICAL"
              + (f"  (+{len(extra)} new key(s): {sorted(extra)})" if extra else ""))
        assert n_diff == 0

        m2, _ = build(R1, torch, 0, bev_cfg=BEVCouplingConfig(d_bev=d_bev),
                      d_bev=d_bev)
        n_new = m2.decoder.bev_coupling_params()
        prov = m2.decoder.bev_coupling_provenance()
        print(f"C. coupling ON: +{n_new:,} params over "
              f"{len(m2.decoder.layers)} layers, provenance "
              f"{prov.get('provenance')!r}, n_points={prov.get('n_points')}")
        assert n_new > 0 and prov["enabled"]
        assert prov["n_points"] == m2.decoder.n_steps
        shared = [k for k in p0 if not torch.equal(p0[k], m2.state_dict()[k])]
        print(f"   shared params vs OFF build: {len(shared)} differ -> "
              f"{'BIT-IDENTICAL' if not shared else 'DIFFER'}")
        assert not shared, ("attaching the BEV coupling re-drew shared weights "
                            "— it is not the last statement of __init__")

        gates = [float(ly.bev_wp.gate.detach()) for ly in m2.decoder.layers]
        print(f"D. gates at init: {gates} (all zero: {all(g == 0.0 for g in gates)})")
        assert all(g == 0.0 for g in gates)
        cc = m2.decoder.layers[0].bev_wp.cfg
        q = torch.randn(2, 5, cc.d_model)
        wp_xy = torch.randn(2, 5, cc.n_points, 2) * 8.0
        bev = torch.randn(2, d_bev, *cc.grid.shape)
        assert torch.equal(m2.decoder.layers[0].bev_wp(q, wp_xy, bev), q)
        m2.decoder.layers[0].bev_wp.gate.data.fill_(1.0)
        assert not torch.equal(m2.decoder.layers[0].bev_wp(q, wp_xy, bev), q)
        print("   gate 0 -> identity; gate 1 -> changed. GATED, not dead.")

        wcfg = wpi.WaypointIndexConfig(enable=True)
        m3, _ = build(R1, torch, 0, wp=wcfg, agents=True)
        m4, _ = build(R1, torch, 0, wp=wcfg, agents=True,
                      bev_cfg=BEVCouplingConfig(d_bev=d_bev), d_bev=d_bev)
        s3, s4 = m3.state_dict(), m4.state_dict()
        wpk = [k for k in s3 if "wp_index" in k]
        drift = [k for k in wpk if not torch.equal(s3[k], s4[k])]
        print(f"E. WP-B heads ({len(wpk)} tensors) across BEV on/off: "
              f"{len(drift)} differ -> "
              f"{'BIT-IDENTICAL' if not drift else 'DIFFER: ' + str(drift[:3])}")
        assert wpk, "WP-B did not attach — the 2x2 check is vacuous"
        assert not drift

        print("\nPATCH VERIFIED")
        return 0
    finally:
        if not a.keep and a.tmp is None:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
