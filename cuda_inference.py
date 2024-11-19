import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import warnings
from torch.utils.benchmark import Timer
warnings.filterwarnings("ignore")

def run_inference(prompt, model_path="./Llama-3.2-1B", max_length=100):
    print("\nLoading model and tokenizer...")
    start_time = time.time()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        torch.set_num_threads(6)
    print(f"Using device: {device}")
    
    # Load model with appropriate settings
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device)
    
    if device == "cpu":
        if hasattr(torch._C, '_jit_set_profiling_executor'):
            torch._C._jit_set_profiling_executor(True)
        if hasattr(torch._C, '_jit_set_profiling_mode'):
            torch._C._jit_set_profiling_mode(True)
    
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")
    print(f"Model size: {sum(p.numel() for p in model.parameters())/1e9:.2f}B parameters")

    formatted_prompt = f"Question: {prompt}\nAnswer:"
    print("\nGenerating response...")
    gen_start_time = time.time()
    
    with torch.inference_mode():
        with torch.cuda.amp.autocast() if device == "cuda" else torch.cpu.amp.autocast():
            inputs = tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                add_special_tokens=True
            ).to(device)
            
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                max_length=max_length,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                use_cache=True
            )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    gen_time = time.time() - gen_start_time
    print(f"Response generated in {gen_time:.2f} seconds")
    
    return response

if __name__ == "__main__":
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024
    print(f"\nInitial memory usage: {initial_memory:.2f} MB")
    print(f"CPU cores available: {psutil.cpu_count()}")
    print(f"CPU physical cores: {psutil.cpu_count(logical=False)}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f}GB")
    
    try:
        question = "What color is the sky?"
        print(f"\nPrompt: {question}")
        
        response = run_inference(question)
        print(f"\nResponse: {response}")
        
        final_memory = process.memory_info().rss / 1024 / 1024
        print(f"\nFinal memory usage: {final_memory:.2f} MB")
        print(f"Memory increase: {final_memory - initial_memory:.2f} MB")
        print(f"CPU usage: {process.cpu_percent()}%")
        
        if torch.cuda.is_available():
            print(f"GPU memory used: {torch.cuda.max_memory_allocated()/1024**3:.2f}GB")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("\nFull error traceback:")
        import traceback
        print(traceback.format_exc())

    print(f"\nTorch version: {torch.__version__}")
    print(f"Torch config:")
    print(f" - Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f" - Number of threads: {torch.get_num_threads()}")
    print(f" - MKL enabled: {torch.backends.mkl.is_available()}")
    if hasattr(torch.backends, 'openmp'):
        print(f" - OpenMP enabled: {torch.backends.openmp.is_available()}")