import sys, torch
for p in sys.argv[1:]:
    ck = torch.load(p, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck)
    for k in ("core.encoder.fuse16.proj.weight", "core.encoder.fuse32.proj.weight"):
        W = sd[k].float()[:, :, 0, 0]; co = W.shape[0]
        a, b, c = W[:, :co], W[:, co:2*co], W[:, 2*co:]
        I = torch.eye(co)
        cos = float((a*b).sum() / (a.norm()*b.norm()))
        s = (a + b).norm(); d = (a - b).norm()
        print(p.split('/')[-1], k.split('.')[2], "fro old/mid/new-I:", round(float(a.norm()),3), round(float(b.norm()),3), round(float((c-I).norm()),3),
              "cos(old,mid)=", round(cos,4), "|old+mid|=", round(float(s),3), "|old-mid|=", round(float(d),3),
              "ratio |old+mid|/|new|=", round(float(s/c.norm()),4))
