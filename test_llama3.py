import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def run_inference():
    # Model and tokenizer paths
    model_path = "./Meta-Llama-3-8B"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    print("Loading model...")
    # For macOS with Apple Silicon, we'll use the MPS device if available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True
    ).to(device)
    
    # Test prompt
    prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    Explain why the sky appears blue during the day.

    ### Response:"""
    
    print("\nGenerating response for prompt:", prompt)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
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
    try:
        run_inference()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
