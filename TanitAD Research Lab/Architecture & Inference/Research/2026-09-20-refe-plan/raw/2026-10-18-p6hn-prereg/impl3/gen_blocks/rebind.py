    # ---------------- P6-HN-S: the ON path.  REBINDS fwd / train_losses / log (the originals above stay byte-identical; with no flag nothing below runs and the loop's four statements are the stock ones).
    if cc_obj is not None or hn_obj is not None:
        fwd_live, train_losses_stock, log_stock = fwd, train_losses, log

        def fwd(batch):
            """the stock fwd's four returns.  batch = the stock nine + this run's ScreenItems tail: (cached ctx)? (hn_xy, hn_tg, hn_m, bank row)?  The cached context REPLACES the trunk forward (arm A scores through
            model.score_trajectories -- the function REFe.forward calls -- and B / the base through their own scorer calls, on the SAME tensor); the hn fields of THIS batch are stashed for train_losses."""
            body, tail = tuple(batch[:9]), tuple(batch[9:])
            hn_cur["tail"] = tail[1:] if cc_obj is not None else tail
            if cc_obj is None:
                out = fwd_live(body)
                hn_cur["cxy"], hn_cur["sctx"] = body[4].to(dev, non_blocking=True), cap["sctx"]
                return out
            cxy_, ctg_, ctx_ = body[4].to(dev, non_blocking=True), body[5].to(dev, non_blocking=True), tail[0].to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                sxA = model.score_trajectories(cxy_, ctx_)
                sxB = armB(cxy_, ctx_)
                with torch.no_grad():
                    sx0 = base(cxy_, ctx_)
            hn_cur["cxy"], hn_cur["sctx"], cap["sctx"] = cxy_, ctx_, ctx_                 # cap['sctx'] = what the scoring decoder reads: every later consumer (G5's base call, G11 / G13's probes) sees the cached tensor
            return sx0.float(), sxA.float(), sxB.float(), ctg_.float()

        def train_losses(sx0, sxA, sxB, tg, rows, upd, lam_pg=None):
            """the stock assembly (train_losses_stock, untouched) + the hard-negative term on arm B ONLY, read from the fields of the batch the latest fwd() served.  Arm A is the BCE-only control."""
            lA, lB, ln, ep = train_losses_stock(sx0, sxA, sxB, tg, rows, upd, lam_pg)
            if hn_obj is None or hn_cur["off"] or len(hn_cur["tail"]) != 4:
                return lA, lB, ln, ep
            hxy, htg, hm, hrow = (x.to(dev, non_blocking=True) for x in hn_cur["tail"])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hn_out = HN.hn_term(armB, hn_cur["cxy"], hxy, htg, hm, hn_cur["sctx"])
            acc = hn_cur["acc"]
            acc["rows"] += int(hn_out.n_hn); acc["expected"] += int(hn_mask_all[hrow.cpu().numpy()].sum())
            acc["loss"] += float(hn_out.loss); acc["n"] += 1; acc["nonempty"] += int(int(hn_out.n_hn) > 0)
            lB = lB + hn_out.loss
            return lA, lB, ln, ep

        def log(row):
            if hn_obj is not None and row.get("event") == "train":          # D1: the hard-negative rows that entered L_B in this window vs the rows the bank holds for the served scenes
                ac = hn_cur["acc"]
                row.update({"hn_rows": ac["rows"], "hn_rows_expected": ac["expected"], "hn_loss": ac["loss"] / max(ac["n"], 1), "hn_micro": ac["n"], "hn_micro_nonempty": ac["nonempty"]})
                hn_cur["acc"] = {"rows": 0, "expected": 0, "loss": 0.0, "n": 0, "nonempty": 0}
            log_stock(row)

        hn_env = HNEnv(a=a, dev=dev, model=model, armB=armB, base=base, cap=cap, cur=hn_cur, bank=hn_obj, cache=cc_obj, ds=ds, ds_ho=ds_ho, tr_idx=tr_idx, ho_idx=ho_idx, fwd=fwd, fwd_live=fwd_live,
                       train_losses=train_losses, train_losses_stock=train_losses_stock, trunk_fp=trunk_fp,
                       cache_manifest_sha256=(CC.sha256_file(os.path.join(a.ctx_cache, "manifest.json")) if cc_obj is not None else None))

