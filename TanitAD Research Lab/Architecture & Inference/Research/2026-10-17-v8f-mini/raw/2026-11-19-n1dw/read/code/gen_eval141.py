"""gen_eval141.py -- one-shot generator: adapts the NAVRBD eval-141 wrappers to the four N1DW arms (only arm list, tags and paths change)."""
C = "F:/TanitAD-scratch/n1dw/read/code/"
E = "F:/TanitAD-scratch/n1dw/read/eval141/"


def rd(p):
    return open(p, encoding="utf-8").read()


def must(s, a, b):
    assert a in s, a[:80]
    return s.replace(a, b)


# ---------------- n1dw_dumps.py
s = rd(C + "navrbd_dumps.py")
s = must(s, '"""navrbd_dumps.py -- T1-kin dumps of the N1D arms on EVAL-141 (the guard cells, Z-EVAL, Z-USE beside), dev box CPU, 2 threads, zero training.',
         '"""n1dw_dumps.py -- STAGED, NOT RUN. T1-kin dumps of the four N1DW arms on EVAL-141 (card PREREG_V8F_N1DW 5.2 guard, Z-EVAL, Z-USE beside), dev box CPU, 2 threads, zero training.\n'
         'Adapted from navrbd_dumps.py: ONLY the arm list, the tags (N1DW-<arm>[Z]) and the paths change; the pinned evaluator (v8f_t1_eval.py sha256 180044eb = READ_PINS.json, wrapped by v8f_t1_eval_rbd.py of the 134f5da1 tree), threads, grid, join and every --nav-rb flag are NAVRBD\'s.')
s = must(s, "Usage: python navrbd_dumps.py TAG [TAG ...]   TAG in NAVRBD-N1D_s0 NAVRBD-N1D_s1 and their zero-channel twins NAVRBD-N1D_s0Z NAVRBD-N1D_s1Z",
         "Usage: python n1dw_dumps.py TAG [TAG ...]   TAG in N1DW-N1DW4_s0 N1DW-N1DW4_s1 N1DW-N1DW2_s0 N1DW-N1DW2_s1 and their zero twins (suffix Z)")
s = must(s, "import navrbd_read as NR  # noqa: E402\n", "")
s = must(s, 'OUT = os.environ.get("NAVRBD_D_141", NR.HERE + "/dumps141")',
         'ROOT = "F:/TanitAD-scratch/strat1/navrbd"      # NAVRBD tables: b2/tables/navtext_nav_rbd.npz (sha 298049df), the file N1D eval-141 dumps read\nOUT = os.environ.get("N1DW_D_141", HERE + "/dumps141")')
s = must(s, 'TREE = os.environ.get("NAVRBD_TREE", NR.ROOT + "/tree")', 'TREE = os.environ.get("N1DW_TREE", HERE + "/tree")          # git -c core.autocrlf=false archive 134f5da1 stack taniteval')
s = must(s, 'CK = os.environ.get("NAVRBD_CKPTS", NR.HERE + "/ckpts")', 'CK = os.environ.get("N1DW_CKPTS", os.path.dirname(HERE) + "/ckpts")')
s = must(s, 'RB_NPZ = NR.ROOT + "/b2/tables/navtext_nav_rbd.npz"', 'RB_NPZ = ROOT + "/b2/tables/navtext_nav_rbd.npz"')
s = s.replace("NAVRBD_SMOKE_WINDOWS", "N1DW_SMOKE_WINDOWS").replace("NAVRBD_SMOKE_GRID", "N1DW_SMOKE_GRID")
s = must(s, 'D_ARMS = ("N1D_s0", "N1D_s1")\nTAGS = tuple(f"NAVRBD-{a}{z}" for a in D_ARMS for z in ("", NL.Z))',
         'D_ARMS = ("N1DW4_s0", "N1DW4_s1", "N1DW2_s0", "N1DW2_s1")\nPFX = "N1DW-"\nTAGS = tuple(f"{PFX}{a}{z}" for a in D_ARMS for z in ("", NL.Z))')
s = must(s, 'ARMS = {t: (f"{CK}/{t[7:].rstrip(NL.Z)}/ckpt_last.pt", ("bank", f"{CK}/{t[7:].rstrip(NL.Z)}/BANK_MANIFEST.json")) for t in TAGS}',
         'def arm_of(t):\n    a = t[len(PFX):]\n    return a[:-1] if a.endswith(NL.Z) else a\n\n\nARMS = {t: (f"{CK}/{arm_of(t)}/ckpt_last.pt", ("bank", f"{CK}/{arm_of(t)}/BANK_MANIFEST.json")) for t in TAGS}')
s = must(s, "return refusal_ck(tag[7:])", "return refusal_ck(arm_of(tag))")
s = s.replace("NAVRBD extra:", "N1DW extra:").replace("DUMPS_NAVRBD_", "DUMPS_N1DW_")
open(E + "n1dw_dumps.py", "w", encoding="utf-8").write(s)

# ---------------- n1dw_t0.py
t = rd(C + "navrbd_t0.py")
t = must(t, "navrbd_t0.py -- the T0 STRATEGIC passes of the NAVRBD read for the two N1D arms", "n1dw_t0.py -- STAGED, NOT RUN. The T0 STRATEGIC passes of the N1DW read for the four N1DW arms")
t = must(t, "import navrbd_dumps as ND", "import n1dw_dumps as ND")
t = must(t, 'T0_OUT = os.environ.get("NAVRBD_T0", ND.NR.HERE + "/out/t0")', 'T0_OUT = os.environ.get("N1DW_T0", ND.HERE + "/out/t0")')
t = t.replace("NAVRBD-{arm}_{variant}", "N1DW-{arm}_{variant}").replace("eval_NAVRBD-{arm}_{v}", "eval_N1DW-{arm}_{v}").replace("argv_NAVRBD-{arm}_{v}", "argv_N1DW-{arm}_{v}").replace("log_NAVRBD-{arm}_{v}", "log_N1DW-{arm}_{v}")
open(E + "n1dw_t0.py", "w", encoding="utf-8").write(t)

# ---------------- n1dw_view.py
v = rd(C + "navrbd_view.py")
v = must(v, "navrbd_view.py -- builds the T0 / ticks VIEW the frozen STRAT1 STRATEGIC readers read for the N1D arms (card 5.2).",
         "n1dw_view.py -- STAGED, NOT RUN. Builds the T0 / ticks VIEW the frozen STRAT1 STRATEGIC readers read for ONE N1DW weight (card 5.2). Usage: python n1dw_view.py VIEW_DIR WEIGHT(2|4) [--no-ticks].")
v = must(v, 'T0_DEFAULT = os.environ.get("NAVRBD_T0", HERE + "/out/t0")', 'T0_DEFAULT = os.environ.get("N1DW_T0", HERE + "/out/t0")')
v = must(v, 'T1_DEFAULT = os.environ.get("NAVRBD_D_141", HERE + "/dumps141")', 'T1_DEFAULT = os.environ.get("N1DW_D_141", HERE + "/dumps141")')
v = must(v, "def plan(t0=None, t1=None, banked_t0=None, banked_t1=None, ticks=True):", "def plan(w=4, t0=None, t1=None, banked_t0=None, banked_t1=None, ticks=True):")
v = must(v, "eval_NAVRBD-N1D_s{s}_{v}.npz", "eval_N1DW-N1DW{w}_s{s}_{v}.npz")
v = must(v, "ticks_NAVRBD-N1D_s{s}_m10", "ticks_N1DW-N1DW{w}_s{s}_m10")
v = must(v, "def build(view, **kw):\n    p = plan(**kw)", "def build(view, w=4, **kw):\n    p = plan(w=w, **kw)")
v = must(v, 'print("view files:", build(sys.argv[1], ticks="--no-ticks" not in sys.argv))', 'print("view files:", build(sys.argv[1], w=int(sys.argv[2]), ticks="--no-ticks" not in sys.argv))')
open(E + "n1dw_view.py", "w", encoding="utf-8").write(v)

# ---------------- n1dw_strategic.py
g = rd(C + "navrbd_strategic.py")
g = must(g, "navrbd_strategic.py -- the STRATEGIC contrast N1D - N1R of the NAVRBD read", "n1dw_strategic.py -- STAGED, NOT RUN. The STRATEGIC contrast N1DW{2,4} - N1R of the N1DW read (adapted from navrbd_strategic.py; the ROUTE-COST rule is unchanged)")
g = must(g, 'T0_DEFAULT = os.environ.get("NAVRBD_T0", HERE + "/out/t0")', 'T0_DEFAULT = os.environ.get("N1DW_T0", HERE + "/out/t0")')
g = must(g, 'def pair_name():\n    return "N1D-N1R"', 'def pair_name(w):\n    return f"N1DW{w}-N1R"')
g = must(g, '    ap.add_argument("--hmax", type=int, default=HMAX)', '    ap.add_argument("--hmax", type=int, default=HMAX)\n    ap.add_argument("--weights", type=int, nargs="+", default=[2, 4])')
old = '''    per_seed = {}
    for s in NL.SEEDS:
        R = {v: load(a.t0, f"NAVRBD-N1D_s{s}", v) for v in VARIANTS}
        B = {v: load(a.n1r_t0, f"NAVRB2-N1R_s{s}", v) for v in VARIANTS}
        per_seed[s] = N2S.pair_contrasts(CI, G, eid, R, B, rb, floors, a.n_boot)
    rep["pairs"][pair_name()] = {str(s): per_seed[s] for s in per_seed}
    rep["route_cost"][pair_name()] = route_cost_of({s: per_seed[s]["route_setting_RC-A_R_minus_banked"] for s in per_seed})'''
new = '''    for w in a.weights:
        per_seed = {}
        for s in NL.SEEDS:
            R = {v: load(a.t0, f"N1DW-N1DW{w}_s{s}", v) for v in VARIANTS}
            B = {v: load(a.n1r_t0, f"NAVRB2-N1R_s{s}", v) for v in VARIANTS}
            per_seed[s] = N2S.pair_contrasts(CI, G, eid, R, B, rb, floors, a.n_boot)
        rep["pairs"][pair_name(w)] = {str(s): per_seed[s] for s in per_seed}
        rep["route_cost"][pair_name(w)] = route_cost_of({s: per_seed[s]["route_setting_RC-A_R_minus_banked"] for s in per_seed})'''
g = must(g, old, new)
g = must(g, '"tool": "navrbd_strategic.py"', '"tool": "n1dw_strategic.py"')
open(E + "n1dw_strategic.py", "w", encoding="utf-8").write(g)
print("generated")
