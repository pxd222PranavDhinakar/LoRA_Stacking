# Meta-Llama 3 on HPC Setup Guide

This guide describes how to set up and run Meta-Llama 3 inference on the Case HPC cluster.

## Server Connection

1. Connect to the server:
```bash
ssh -X pxd222@pioneer.case.edu
```

2. Request a GPU node:
```bash
srun --x11 -p markov_gpu --gres=gpu:1 --mem=32gb --pty /bin/bash
```
Note: We're using 32GB memory since Llama 3 8B requires more RAM than smaller models.

## Environment Setup

Load the necessary modules:
```bash
module load GCCcore
module load Python/3.10.4
module load CUDA/12.1.1
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
```

Install required Python packages:
```bash
pip install transformers
pip install accelerate
pip install sentencepiece
pip install huggingface-hub
```

## Model Setup

1. Log in to Hugging Face (do this before cloning):
```bash
# Install the Hugging Face CLI if needed
pip install --user huggingface_hub

# Login to Hugging Face
huggingface-cli login
# Enter your token from https://huggingface.co/settings/tokens
```

2. Clone Meta-Llama 3 (lightweight clone):
```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/meta-llama/Meta-Llama-3-8B
```

## Running Inference

Create a new file named `test_llama3.py`:
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def run_inference():
    # Model and tokenizer paths
    model_path = "./Meta-Llama-3-8B"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    # Test prompt
    prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    Explain why the sky appears blue during the day.

    ### Response:"""
    
    print("\nGenerating response for prompt:", prompt)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    print("Generating...")
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and print response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\nFull Response:\n", response)

if __name__ == "__main__":
    run_inference()
```

## Running the Test

1. Submit as a batch job:
```bash
cat << EOF > run_llama3.sh
#!/bin/bash
#SBATCH --job-name=llama3_test
#SBATCH --output=llama3_%j.log
#SBATCH --error=llama3_error_%j.log
#SBATCH --partition=markov_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32gb
#SBATCH --time=1:00:00

module load GCCcore
module load Python/3.10.4
module load CUDA/12.1.1
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

python test_llama3.py
EOF

sbatch run_llama3.sh
```

2. Monitor the output:
```bash
tail -f llama3_*.log
```

## Troubleshooting

- If you run into memory issues, modify the model loading to use 8-bit quantization:
```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    load_in_8bit=True,
    device_map="auto",
    trust_remote_code=True
)
```

- Check GPU usage:
```bash
nvidia-smi
```

- Check running jobs:
```bash
squeue -u pxd222
```

- Cancel a job:
```bash
scancel <job_id>
```