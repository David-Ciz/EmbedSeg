Instalation:

ml Python/3.11.5-GCCcore-13.2.0
python embedseg_venv venv
source embedseg_venv/bin/activate
pip install torch torchvision torchaudio
git clone git@github.com:David-Ciz/EmbedSeg.git
cd EmbedSeg
pip install -e .

Start:
srun --account=OPEN-28-13 --partition=qgpu --nodes=1 --gpus=1 --time=04:00:00 --pty bash -i
conda deactivate
source embedseg_venv/bin/activate
python path/to/python_script.py
