import importlib.util, sys, torch
REFE = r"D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe"
sys.path.insert(0, REFE)
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m
new = load(REFE + "/model.py", "model_new")
old = load(sys.argv[1], "model_old")
allok = True
for bb in ("vits16", "vitl16"):
    for und in (True, False):
        cn = new.REFeConfig.for_backbone(bb); co = old.REFeConfig.for_backbone(bb)
        cn.undistort = und; co.undistort = und
        torch.manual_seed(0); mn = new.REFe(cn)
        torch.manual_seed(0); mo = old.REFe(co)
        gh, gw = cn.img_h // cn.patch, cn.img_w // cn.patch
        for dt in (torch.float32, torch.bfloat16):
            fn = mn._frustum(gh, gw, "cpu", dt, cn.n_cameras); fo = mo._frustum(gh, gw, "cpu", dt, co.n_cameras)
            eq = torch.equal(fn, fo); allok &= eq
            print(f"  {bb} undistort={und!s:5s} {str(dt):15s} frustum identical: {eq}  {tuple(fn.shape)}")
        img = torch.zeros(1, cn.n_cameras, 3, cn.img_h, cn.img_w)
        with torch.no_grad():
            en = mn._pos3d(img, cn.n_cameras); eo = mo._pos3d(img, co.n_cameras)
        eq = torch.equal(en, eo); allok &= eq
        print(f"  {bb} undistort={und!s:5s} pos3d embedding identical: {eq}")
print("ZZOLD_NEW_IDENTICALZZ" if allok else "ZZOLD_NEW_DIFFERZZ")
