#!/usr/bin/env python3
"""
Architecture validation tool to verify LLaMA compliance and architectural consistency.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

def validate_config_consistency(config_dict: Dict) -> List[str]:
    """Validate that configuration follows LLaMA architectural principles."""
    issues = []
    warnings = []
    
    # Required fields
    required_fields = ['model_name', 'n_layer', 'd_model', 'n_head', 'd_ff', 'vocab_size', 'sequence_length']
    for field in required_fields:
        if field not in config_dict:
            issues.append(f"❌ Missing required field: {field}")
    
    if issues:  # Don't continue validation if missing required fields
        return issues
    
    # Extract values
    n_layer = config_dict['n_layer']
    d_model = config_dict['d_model'] 
    n_head = config_dict['n_head']
    d_ff = config_dict['d_ff']
    vocab_size = config_dict['vocab_size']
    sequence_length = config_dict['sequence_length']
    
    # 1. Head dimension should be 64 for optimal performance
    head_dim = d_model // n_head
    if head_dim != 64:
        if head_dim < 32:
            issues.append(f"❌ head_dim={head_dim} too small (< 32), may hurt performance")
        elif head_dim > 128:
            issues.append(f"❌ head_dim={head_dim} too large (> 128), inefficient")
        else:
            warnings.append(f"⚠️ head_dim={head_dim} not optimal (should be 64)")
    else:
        print(f"✅ head_dim={head_dim} (optimal)")
    
    # 2. FFN ratio should be ~4.0 
    ffn_ratio = d_ff / d_model
    if abs(ffn_ratio - 4.0) > 0.5:
        warnings.append(f"⚠️ ffn_ratio={ffn_ratio:.1f} unusual (typically 4.0)")
    else:
        print(f"✅ ffn_ratio={ffn_ratio:.1f} (standard)")
    
    # 3. Vocabulary size should be power-of-2 friendly
    if vocab_size not in [32768, 65536, 50257]:  # Common vocab sizes
        if vocab_size % 1024 != 0:
            warnings.append(f"⚠️ vocab_size={vocab_size} not aligned to 1024 boundary")
    else:
        print(f"✅ vocab_size={vocab_size} (standard)")
    
    # 4. Layer count should be reasonable
    if n_layer < 6:
        warnings.append(f"⚠️ n_layer={n_layer} very small, may limit capacity")
    elif n_layer > 32:
        warnings.append(f"⚠️ n_layer={n_layer} very large, may be slow to train")
    else:
        print(f"✅ n_layer={n_layer} (reasonable)")
    
    # 5. Context length should be power of 2
    if sequence_length & (sequence_length - 1) != 0:  # Not power of 2
        warnings.append(f"⚠️ sequence_length={sequence_length} not power of 2")
    else:
        print(f"✅ sequence_length={sequence_length} (power of 2)")
    
    # 6. Check for GQA configuration
    if 'num_key_value_heads' in config_dict:
        num_kv_heads = config_dict['num_key_value_heads']
        if num_kv_heads > n_head:
            issues.append(f"❌ num_key_value_heads ({num_kv_heads}) > num_heads ({n_head})")
        elif num_kv_heads < n_head:
            ratio = n_head / num_kv_heads
            if ratio != int(ratio):
                warnings.append(f"⚠️ GQA ratio {n_head}:{num_kv_heads} = {ratio:.1f} not integer")
            else:
                print(f"✅ GQA enabled: {n_head}:{num_kv_heads} = {int(ratio)}:1 ratio")
    
    return issues + warnings

def estimate_memory_usage(config_dict: Dict) -> Dict[str, float]:
    """Estimate memory usage for different scenarios."""
    
    d_model = config_dict['d_model']
    n_layer = config_dict['n_layer'] 
    d_ff = config_dict['d_ff']
    vocab_size = config_dict['vocab_size']
    sequence_length = config_dict['sequence_length']
    
    # Parameter estimation (simplified)
    # Token embeddings
    token_emb = vocab_size * d_model
    
    # Per layer: attention + ffn + norms
    attention = d_model * d_model * 4  # Q,K,V,O projections
    ffn = d_model * d_ff * 3  # SwiGLU: up, gate, down
    norms = d_model * 2  # attention_norm, ffn_norm
    per_layer = attention + ffn + norms
    
    # Total parameters
    total_params = token_emb + n_layer * per_layer + d_model  # + final_norm
    
    # Memory estimates (GB)
    param_memory_fp16 = total_params * 2 / (1024**3)  # FP16
    param_memory_fp32 = total_params * 4 / (1024**3)  # FP32
    
    # Training memory (rough estimate)
    # Parameters + gradients + optimizer states (Adam: 2x params) + activations
    training_memory = param_memory_fp32 * 4  # Very rough estimate
    
    # Inference memory with activations
    batch_size = 1
    activation_memory = batch_size * sequence_length * d_model * n_layer * 2 / (1024**3)  # FP16
    inference_memory = param_memory_fp16 + activation_memory
    
    return {
        'parameters': int(total_params),
        'param_memory_fp16_gb': param_memory_fp16,
        'param_memory_fp32_gb': param_memory_fp32,
        'training_memory_gb': training_memory,
        'inference_memory_gb': inference_memory
    }

def compare_with_reference(config_dict: Dict) -> None:
    """Compare with reference LLaMA architectures."""
    
    params = estimate_memory_usage(config_dict)['parameters']
    
    # Reference architectures (approximate)
    references = {
        'LLaMA-7B': {'n_layer': 32, 'd_model': 4096, 'n_head': 32, 'd_ff': 11008, 'params': '7B'},
        'LLaMA-13B': {'n_layer': 40, 'd_model': 5120, 'n_head': 40, 'd_ff': 13824, 'params': '13B'},
        'GPT-2-small': {'n_layer': 12, 'd_model': 768, 'n_head': 12, 'd_ff': 3072, 'params': '117M'},
        'GPT-2-medium': {'n_layer': 24, 'd_model': 1024, 'n_head': 16, 'd_ff': 4096, 'params': '345M'},
    }
    
    print(f"\n📊 Model Comparison (your model: {params:,} parameters)")
    print("=" * 60)
    
    closest_ref = None
    min_ratio = float('inf')
    
    for name, ref in references.items():
        # Simple comparison based on layer count and hidden size
        ref_complexity = ref['n_layer'] * ref['d_model']  
        your_complexity = config_dict['n_layer'] * config_dict['d_model']
        ratio = max(ref_complexity / your_complexity, your_complexity / ref_complexity)
        
        if ratio < min_ratio:
            min_ratio = ratio
            closest_ref = name
        
        print(f"{name:15} | L={ref['n_layer']:2d} d={ref['d_model']:4d} h={ref['n_head']:2d} | {ref['params']:>4}")
    
    print("-" * 60)
    your_config = config_dict
    print(f"{'Your model':15} | L={your_config['n_layer']:2d} d={your_config['d_model']:4d} h={your_config['n_head']:2d} | {params/1e6:.0f}M")
    
    if closest_ref:
        print(f"\n🎯 Closest reference: {closest_ref}")

def validate_architecture_file(config_path: str) -> None:
    """Validate a configuration file."""
    
    print(f"🔍 Validating architecture: {config_path}")
    print("=" * 60)
    
    # Load config
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return
    
    # Basic info
    model_name = config.get('model_name', 'unknown')
    print(f"📋 Model: {model_name}")
    
    # Architecture validation
    print(f"\n🔧 Architecture Validation:")
    issues = validate_config_consistency(config)
    
    if any(issue.startswith('❌') for issue in issues):
        print(f"\n❌ CRITICAL ISSUES FOUND:")
        for issue in issues:
            if issue.startswith('❌'):
                print(f"   {issue}")
    
    warnings = [issue for issue in issues if issue.startswith('⚠️')]
    if warnings:
        print(f"\n⚠️ Warnings:")
        for warning in warnings:
            print(f"   {warning}")
    
    # Memory estimation
    print(f"\n💾 Memory Estimates:")
    memory = estimate_memory_usage(config)
    print(f"   Parameters: {memory['parameters']:,}")
    print(f"   Model (FP16): {memory['param_memory_fp16_gb']:.1f} GB")
    print(f"   Training: ~{memory['training_memory_gb']:.1f} GB")
    print(f"   Inference: ~{memory['inference_memory_gb']:.1f} GB")
    
    # RTX 4090 compatibility  
    rtx4090_vram = 16.0  # GB
    if memory['training_memory_gb'] > rtx4090_vram:
        print(f"   ⚠️ Training may exceed RTX 4090 VRAM ({rtx4090_vram} GB)")
    elif memory['training_memory_gb'] > rtx4090_vram * 0.8:
        print(f"   ⚠️ Training will use most RTX 4090 VRAM")
    else:
        print(f"   ✅ Compatible with RTX 4090 ({rtx4090_vram} GB VRAM)")
    
    # Comparison with references
    compare_with_reference(config)
    
    # Overall assessment
    critical_issues = [issue for issue in issues if issue.startswith('❌')]
    
    print(f"\n🎯 OVERALL ASSESSMENT:")
    if not critical_issues:
        if not warnings:
            print("✅ Architecture is well-configured and follows LLaMA best practices!")
        else:
            print("✅ Architecture is valid with minor suggestions for optimization.")
    else:
        print("❌ Architecture has issues that should be addressed before training.")
    
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Validate LLaMA architecture configuration")
    parser.add_argument("config_path", help="Path to configuration JSON file")
    
    args = parser.parse_args()
    
    config_path = Path(args.config_path)
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        return
    
    validate_architecture_file(str(config_path))

if __name__ == "__main__":
    main()