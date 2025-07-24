#!/usr/bin/env python3
"""
Debug script to test if quantization is working properly
"""
import torch
import os
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer

def get_model_memory_usage():
    """Get GPU memory usage"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3  # GB
    return 0

def test_quantization():
    model_name = "Qwen/Qwen3-Reranker-0.6B"
    print(f"Testing quantization with {model_name}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"Initial GPU memory: {get_model_memory_usage():.2f} GB")
    
    # Test 1: Load without quantization (FP16)
    print("\n=== TEST 1: FP16 (no quantization) ===")
    try:
        torch.cuda.empty_cache()
        model_fp16 = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cpu"  # Force CPU to avoid GPU OOM
        )
        param_count = sum(p.numel() for p in model_fp16.parameters())
        memory_fp16 = get_model_memory_usage()
        
        print(f"Parameters: {param_count:,}")
        print(f"Expected size (FP16): {param_count * 2 / 1024**3:.2f} GB")
        print(f"GPU memory after load: {memory_fp16:.2f} GB")
        print(f"First param dtype: {next(model_fp16.parameters()).dtype}")
        print(f"First param device: {next(model_fp16.parameters()).device}")
        
        del model_fp16
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"FP16 test failed: {e}")
    
    # Test 2: Load with 8-bit quantization
    print("\n=== TEST 2: 8-bit quantization ===")
    try:
        torch.cuda.empty_cache()
        
        # Configure quantization according to Hugging Face docs
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
            llm_int8_threshold=6.0
        )
        
        print(f"Quantization config: {quant_config}")
        
        model_8bit = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        param_count = sum(p.numel() for p in model_8bit.parameters())
        memory_8bit = get_model_memory_usage()
        
        print(f"Parameters: {param_count:,}")
        print(f"Expected size (8-bit): {param_count * 1 / 1024**3:.2f} GB")
        print(f"GPU memory after load: {memory_8bit:.2f} GB")
        
        # Check if layers are actually quantized
        quantized_layers = 0
        total_layers = 0
        for name, module in model_8bit.named_modules():
            total_layers += 1
            if hasattr(module, 'weight') and hasattr(module.weight, 'CB'):
                quantized_layers += 1
                print(f"Quantized layer found: {name}")
            elif hasattr(module, 'weight'):
                print(f"Regular layer: {name} - dtype: {module.weight.dtype}")
                
        print(f"Quantized layers: {quantized_layers}/{total_layers}")
        
        # Test inference to make sure model works
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        test_input = "This is a test input"
        inputs = tokenizer(test_input, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
        with torch.no_grad():
            outputs = model_8bit(**inputs)
            print(f"Inference test successful. Output shape: {outputs.logits.shape}")
        
        del model_8bit
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"8-bit quantization test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_quantization() 