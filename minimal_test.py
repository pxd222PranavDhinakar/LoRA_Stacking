import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

def minimal_inference(model_dir="./Llama-3.2-1B"):
    """
    Bare minimum inference test to verify model loading and basic functionality
    """
    print("\nSystem Check:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Transformers version: {transformers.__version__}")
    
    try:
        # Simply try to load the model
        print("\nAttempting to load model...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir)
        print("Model loaded successfully!")
        
        # Try a simple tokenization
        print("\nTesting tokenization...")
        test_text = "Hello, world!"
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"Tokenization successful! Input text tokenized to length: {len(tokens['input_ids'][0])}")
        
        return True
        
    except Exception as e:
        print(f"\nError encountered: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting minimal functionality test...")
    success = minimal_inference()
    print(f"\nTest completed! Success: {success}")