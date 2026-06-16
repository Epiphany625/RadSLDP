# please make sure you have access right to ReXGradient dataset

# prepare and activate conda environment
conda create -n rexgradient python=3.11 -y
conda activate rexgradient
pip install huggingface_hub zstandard

# download dataset
python download.py

# combine the files
cat ReXGradient/deid_png.part* > ReXGradient/deid_png.tar

# decompress the generated file
python decompress.py


# convert to images
tar -xf ReXGradient/deid_png.raw.tar