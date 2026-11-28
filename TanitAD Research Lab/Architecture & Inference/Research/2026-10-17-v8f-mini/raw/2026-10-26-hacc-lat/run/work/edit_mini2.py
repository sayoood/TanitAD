"""Second edit: the gated post-map computes HACC's plan with HACC's own lines for EVERY sample and then overrides the fired samples' cross column with the prior (torch.where = a selection, no arithmetic),
so a gate that never fires is bit-identical to the ungated model (control C-B1 is 1e-9 m)."""
p = "C:/Users/Admin/hlat/code/stack/tanitad/models/v8f_mini.py"
s = open(p, encoding="utf-8", newline="").read().replace("\r\n", "\n")
old = '''        w60 = hacc_w_midpoints(w30)
        if self.hacc_gate_v is not None:
            if v0 is None:
                raise ValueError("hacc_gate_v is set: the post-map needs the measured ego speed v0 of every sample")
            fire = v0.to(device=w60.device, dtype=w60.dtype) < self.hacc_gate_v.to(device=w60.device, dtype=w60.dtype)       # [B]
            w60 = w60[None].expand(fire.shape[0], -1, -1).clone()
            w60[:, :, HACC_GATE_AXIS] = torch.where(fire[:, None], torch.zeros_like(w60[:, :, HACC_GATE_AXIS]), w60[:, :, HACC_GATE_AXIS])
            out["hacc_gate_fired"] = fire
        out["traj60"] = hacc_apply(out["traj60"], upsample_prior(out["p10"]), w60)
        out["traj30"] = subsample_02(out["traj60"])
        return out
'''
new = '''        prior60 = upsample_prior(out["p10"])
        plan60 = hacc_apply(out["traj60"], prior60, hacc_w_midpoints(w30))        # HACC's own lines, for EVERY sample: a gate that never fires is bit-identical to no gate
        if self.hacc_gate_v is not None:
            if v0 is None:
                raise ValueError("hacc_gate_v is set: the post-map needs the measured ego speed v0 of every sample")
            fire = v0.to(device=plan60.device, dtype=plan60.dtype) < self.hacc_gate_v.to(device=plan60.device, dtype=plan60.dtype)       # [B]
            axis = torch.zeros(1, 1, 2, dtype=torch.bool, device=plan60.device)
            axis[..., HACC_GATE_AXIS] = True
            plan60 = torch.where(fire[:, None, None] & axis, prior60, plan60)      # a fired sample's cross column IS the prior (weight 0: lerp's exact end); a selection, no arithmetic
            out["hacc_gate_fired"] = fire
        out["traj60"] = plan60
        out["traj30"] = subsample_02(out["traj60"])
        return out
'''
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new)
open(p, "w", encoding="utf-8", newline="").write(s.replace("\n", "\r\n"))
print("ok")
