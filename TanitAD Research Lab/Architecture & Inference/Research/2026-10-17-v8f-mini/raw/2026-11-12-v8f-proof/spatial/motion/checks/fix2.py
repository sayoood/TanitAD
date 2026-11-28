P="C:/Users/Admin/v8proof_c/motion/tree_patch/stack/tanitad/models/hjepa.py"
s=open(P,encoding="utf-8",newline="").read()
a="""        z0 = torch.cat([z[:, s] for s in starts], dim=0)
        ea = torch.cat([a[:, s:s + k] for s in starts], dim=0)
        out = self._roll_embedded(pred, z0, ea, ctx)
        tgt = torch.stack([z[:, s + 1:s + k + 1] for s in starts], dim=0).detach()
        return out.reshape(len(starts), z.shape[0], k, -1), tgt, starts
"""
b="""        out = torch.stack([self._roll_embedded(pred, z[:, s], a[:, s:s + k], ctx) for s in starts], dim=0)     # one call per start, exactly the shapes rollout() sees (stacking the starts on the batch axis moves the result by ~2e-6)
        tgt = torch.stack([z[:, s + 1:s + k + 1] for s in starts], dim=0).detach()
        return out, tgt, starts
"""
assert s.count(a)==1; s=s.replace(a,b)
a="""The starts are stacked on the batch axis
        (rows are independent), the context is the predictor's own"""
assert s.count(a)==1; s=s.replace(a,"""Each start is its own call of the shared
        rollout core (the shapes :meth:`rollout` sees), the context is the predictor's own""")
open(P,"w",encoding="utf-8",newline="").write(s)
