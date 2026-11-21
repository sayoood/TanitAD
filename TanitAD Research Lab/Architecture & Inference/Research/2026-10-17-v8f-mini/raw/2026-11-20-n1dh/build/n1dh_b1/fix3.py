def sub(p, old, new):
    t = open(p, encoding="utf-8", newline="").read()
    if "\r\n" in t:
        old, new = old.replace("\n", "\r\n"), new.replace("\n", "\r\n")
    assert t.count(old) == 1, (p, t.count(old))
    open(p, "w", encoding="utf-8", newline="").write(t.replace(old, new))


for p in ("patch_scripts/p2_trainer.py", "tree/stack/scripts/v8f_mini_train.py"):
    sub(p, '    gt = batch["gt"].float()\n    dyaw, dv, v0, v1, _prov = FF.maneuver_kinematics(gt, ROUTE_DT)',
        '    gt = batch["gt"].double()                                      # float64, as D1-HO\'s reader does (``labels``: torch.as_tensor(np.asarray(X, np.float64))): the label is then bit-for-bit the reader\'s on the same tracks\n    dyaw, dv, v0, v1, _prov = FF.maneuver_kinematics(gt, ROUTE_DT)')
print("ok")
