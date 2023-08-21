git clone https://github.com/TencentARC/ViT-Lens.git
cd ViT-Lens

conda create -n vit-lens python=3.10

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install -c dglteam/label/cu113 dgl

pip install huggingface_hub wandb omegaconf torch_redstone einops tqdm
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
pip install -e open_clip/
pip install -r open_clip/requirements-training.txt