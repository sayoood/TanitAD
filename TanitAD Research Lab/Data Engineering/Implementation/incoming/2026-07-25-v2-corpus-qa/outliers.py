import json, os, sys
import pandas as pd
SP = os.path.dirname(os.path.abspath(__file__))
d = pd.concat([pd.read_csv(os.path.join(SP, f"qa_full_{t}.csv")).assign(shard=t)
               for t in ("pod1", "pod3")], ignore_index=True)
SEL = (r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\TanitAD Research Hub"
       r"\Data Engineering\Implementation\incoming\2026-07-24-v2-corpus-50h-balanced"
       r"\r0_selection_v2.parquet")
sel = pd.read_parquet(SEL); sel["clip_id"] = sel["clip_id"].astype(str)
m = d.merge(sel[["clip_id", "mean_v", "dist_m", "stop_frac", "nlab", "chunk",
                 "country"]], on="clip_id", suffixes=("", "_proj"))
cols = ["clip_id", "shard", "T_out", "dist_m", "dist_m_proj", "mean_v",
        "mean_v_proj", "stop_frac", "stop_frac_proj", "country"]
out = {
 "low_motion_built_lt20m": m.loc[m.dist_m < 20, cols].round(3).to_dict("records"),
 "short_T_lt190": d.loc[d.T_out < 190, ["clip_id", "shard", "T_out", "n_raw",
                                        "dist_m", "mean_v"]].round(3).to_dict("records"),
 "long_T_gt200": int((d.T_out > 200).sum()),
 "T_out_value_counts_top": d.T_out.value_counts().head(8).to_dict(),
 "bytes_min_mb": round(float(d.bytes.min())/1e6, 3),
 "bytes_max_mb": round(float(d.bytes.max())/1e6, 3),
 "bytes_p01_mb": round(float(d.bytes.quantile(0.01))/1e6, 3),
 "smallest_files": d.nsmallest(5, "bytes")[["clip_id", "shard", "bytes", "T_out",
                                            "mean_v"]].to_dict("records"),
 "stopped_frac_gt50pct_clips": int((d.stopped > 0.5).sum()),
 "clips_with_no_turn_label": int((d.has_turn == 0).sum()),
 "country_top": m.country.value_counts().head(6).to_dict(),
 "country_max_share": round(float(m.country.value_counts(normalize=True).max()), 4),
}
print(json.dumps(out, indent=2, default=str))
json.dump(out, open(os.path.join(SP, "outliers.json"), "w"), indent=2, default=str)
