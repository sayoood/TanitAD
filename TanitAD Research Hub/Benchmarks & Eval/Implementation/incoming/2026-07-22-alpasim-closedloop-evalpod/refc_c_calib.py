"""C — probe the NuRec scene USDZ for extractable f-theta camera intrinsics (cx,cy,poly,res)."""
import io, json, zipfile

usdz = ("/workspace/scene_dl/sample_set/26.04_release/"
        "01d503d4-449b-46fc-8d78-9085e70d3554/"
        "01d503d4-449b-46fc-8d78-9085e70d3554.usdz")
z = zipfile.ZipFile(usdz)


def show_text(name, maxlen=2500):
    try:
        b = z.read(name)
        print(f"\n===== {name} ({len(b)} bytes) =====")
        print(b.decode("utf-8", "replace")[:maxlen])
    except Exception as e:
        print(f"  {name}: {e!r}")


for n in ("metadata.yaml", "parsed_config.yaml", "data_info.json",
          "datasource_summary.json"):
    show_text(n)

print("\n===== clipgt/calibration_estimate.parquet =====")
try:
    import pandas as pd
    df = pd.read_parquet(io.BytesIO(z.read("clipgt/calibration_estimate.parquet")))
    print("columns:", list(df.columns))
    print("shape:", df.shape)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.head(6).to_string())
    # dump full first row as dict for the f-theta fields
    print("row0:", json.dumps({k: str(v)[:200] for k, v in df.iloc[0].to_dict().items()},
                              indent=2)[:3000])
except Exception as e:
    print("  parquet read failed:", repr(e)[:200])
print("C_CALIB_DONE")
