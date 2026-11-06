"""Derives proxy_cache_yf2.py from PK's proxy_cache.py by EXACT-string edits (each must match exactly once), so the diff is auditable. Run once; writes the tool + a diff."""
import difflib
import sys
from pathlib import Path

SRC = Path("C:/Users/Admin/pxyaw/refe/proxy_cache.py")           # byte-identical to PK/refe/proxy_cache.py (cmp, 2026-11-06)
DST = Path(__file__).resolve().parent / "proxy_cache_yf2.py"
s = SRC.read_text(encoding="utf-8")


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)


rep('"""The SHARED dev-box proxy cache:', '"""[YF2 COPY of PK/refe/proxy_cache.py with FOUR edits, see proxy_cache_yf2.diff; PK is not touched]\nThe SHARED dev-box proxy cache:')
# edit 2: paths (D: is absent; the data lives on E:)
rep('HERE = Path(__file__).resolve().parent\nsys.path.insert(0, str(HERE))\nBANK = Path("D:/Projects/TanitAD/data/refe_navtrain10")',
    'HERE = Path(__file__).resolve().parent\nsys.path.insert(0, str(HERE))\nimport yaw_common as YC  # noqa: E402  (YF2)\nsys.path.insert(0, str(YC.REFE_DIR))                       # EDIT 2: model.py / train.py / ckpt_io.py come from PK (read-only)\nBANK = Path(os.environ.get("YF2_BANK", str(YC.BANK)))      # EDIT 2: was D:/Projects/TanitAD/data/refe_navtrain10 (D: is absent)')
# edit 3: the decode as a top-level function with a nav_mode branch (P6 final = command mode)
rep('def build(a) -> int:\n    import torch\n    import ckpt_io',
    '''def decode_traj(model, cfg, scene_ctx, ego, goal):
    """EDIT 3: REFe.forward after scene_proj, trajectory side, for BOTH nav modes (goal: u400; command: P6 final -- `goal` carries the one-hot, model.py:763-771)."""
    import torch
    if cfg.nav_mode == "goal":
        g = goal.unsqueeze(-1) * model.goal_freqs
        nav = torch.cat([goal, g.sin().flatten(1), g.cos().flatten(1)], -1)
    else:
        nav = model.cmd_embed(goal.to(ego.dtype))
    q = model.queries.expand(scene_ctx.shape[0], -1, -1) + model.ego_enc(torch.cat([ego, nav], -1)).unsqueeze(1)
    for blk in model.dec:
        q = blk(q, scene_ctx)
    return model.traj_head(q).view(scene_ctx.shape[0], cfg.n_proposals, cfg.horizon_steps, cfg.traj_dim)


def build(a) -> int:
    import torch
    import ckpt_io''')
# edit 1: size the config from the checkpoint (u400 has a 7-row score head; a 6-row model refuses its strict load)
rep('    cfg = REFeConfig.for_backbone(a.backbone)\n    model = REFe(cfg).to(dev).eval()',
    '''    cfg = ckpt_io.config_for_checkpoint(REFeConfig.for_backbone(a.backbone), a.snapshot)      # EDIT 1: n_score_components / nav_mode from the checkpoint's meta
    if a.expect_snapshot_sha256 and sha256_file(a.snapshot)[:len(a.expect_snapshot_sha256)] != a.expect_snapshot_sha256:
        raise SystemExit(f"REFUSED: snapshot sha256 {sha256_file(a.snapshot)[:12]} != expected {a.expect_snapshot_sha256}")
    model = REFe(cfg).to(dev).eval()''')
rep('''    def decode(scene_ctx, ego, goal):
        g = goal.unsqueeze(-1) * model.goal_freqs
        gf = torch.cat([goal, g.sin().flatten(1), g.cos().flatten(1)], -1)
        q = model.queries.expand(scene_ctx.shape[0], -1, -1) + model.ego_enc(torch.cat([ego, gf], -1)).unsqueeze(1)
        for blk in model.dec:
            q = blk(q, scene_ctx)
        return model.traj_head(q).view(scene_ctx.shape[0], cfg.n_proposals, cfg.horizon_steps, cfg.traj_dim)
''', '''    def decode(scene_ctx, ego, goal):
        return decode_traj(model, cfg, scene_ctx, ego, goal)
''')
# edit 4: fail on the FIRST frame whose C1 is not bit-exact (the original only reported at the end, after hours)
rep('''        c1_worst = max(c1_worst, float((dt.float() - traj.float()).abs().max()))
''', '''        c1_worst = max(c1_worst, float((dt.float() - traj.float()).abs().max()))
        if c1_worst != 0.0:
            raise SystemExit(f"C1 FAILED on frame {k}: max|decode - forward| = {c1_worst:.3g} -- stopping at once (EDIT 4)")
''')
rep('"input_path": "train.TargetBank.__getitem__"', '"yf2_snapshot_meta": json.loads(json.dumps(meta, default=str)), "yf2_code_sha256": YC.code_hashes(), "yf2_nav_mode": cfg.nav_mode, "yf2_n_score_components": cfg.n_score_components,\n          "input_path": "train.TargetBank.__getitem__"')
rep('    ap.add_argument("--images")\n    ap.add_argument("--calib")', '    ap.add_argument("--images", default=str(YC.PIXELS))\n    ap.add_argument("--calib", default=str(YC.CALIB))\n    ap.add_argument("--expect-snapshot-sha256", default=None, help="YF2: refuse unless the snapshot file\'s sha256 starts with this")')
DST.write_text(s, encoding="utf-8", newline="\n")
d = "".join(difflib.unified_diff(SRC.read_text(encoding="utf-8").splitlines(True), s.splitlines(True), "PK/refe/proxy_cache.py", "YF2/proxy_cache_yf2.py"))
(DST.parent / "proxy_cache_yf2.diff").write_text(d, encoding="utf-8", newline="\n")
print("wrote", DST, "diff lines", d.count("\n"))
