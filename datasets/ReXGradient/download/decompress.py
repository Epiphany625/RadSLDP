import zstandard as zstd

inp = "ReXGradient/deid_png.tar"
outp = "ReXGradient/deid_png.raw.tar"

with open(inp, "rb") as f_in:
    dctx = zstd.ZstdDecompressor()
    with open(outp, "wb") as f_out:
        dctx.copy_stream(f_in, f_out)

print("Decompression finished:", outp)