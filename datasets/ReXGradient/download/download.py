from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="rajpurkarlab/ReXGradient-160K",
    repo_type="dataset",   
    local_dir="ReXGradient",
    allow_patterns=[
        "README.md",
        "metadata/*",
        "deid_png.part*",
    ],
)

print("Done!")