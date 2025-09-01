conda create -n gencompositor python=3.10 -y
conda activate gencompositor

pip install -r requirements.txt

cd ./app
pip install -e .

cd ./diffusers
pip install -e .

conda install -c conda-forge ffmpeg -y
