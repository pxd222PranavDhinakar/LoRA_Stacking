# Create new environment
conda create -n llama_lora python=3.10 -y

# Activate the environment
conda activate llama_lora

# Install PyTorch with CUDA 12.1 (adjust CUDA version if needed)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install other dependencies
pip install transformers==4.36.2
pip install datasets==2.16.1
pip install accelerate==0.25.0
pip install peft==0.7.1
pip install bitsandbytes==0.41.3
pip install scipy
pip install sentencepiece
pip install huggingface-hub
pip install wandb  # for experiment tracking (optional)

# Verify CUDA is available
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'NA')"

