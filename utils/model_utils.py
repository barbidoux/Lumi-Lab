"""
Utilities for model creation, loading and management.
"""

import json
import os
import random
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedModel,
)


def create_model_config(config_dict, use_flash_attention: bool = True) -> AutoConfig:
    """
    Create a LLaMA configuration from a dictionary.
    
    Architecture followed: LLaMA (decoder-only transformer)
    - RMSNorm for normalization
    - SwiGLU for FFN activation
    - RoPE for positional encoding
    - No bias terms
    - GQA support (Grouped Query Attention)
    
    Args:
        config_dict: Model configuration with architectural parameters
        
    Returns:
        AutoConfig: HuggingFace compatible LLaMA configuration
    """
    
    # LLaMA configuration with all architectural parameters
    llama_config = {
        # === Base architecture ===
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "transformers_version": "4.40.0",
        
        # === Model dimensions ===
        "hidden_size": _get_config_value(config_dict, "d_model"),
        "intermediate_size": _get_config_value(config_dict, "d_ff"), 
        "num_hidden_layers": _get_config_value(config_dict, "n_layer"),
        "num_attention_heads": _get_config_value(config_dict, "n_head"),
        
        # === Grouped Query Attention (GQA) ===
        # Default: MHA (num_kv_heads = num_heads)
        # For GQA: reduce num_key_value_heads (ex: 4 for 8 heads = 2:1 ratio)
        "num_key_value_heads": _get_config_value(config_dict, "num_key_value_heads", _get_config_value(config_dict, "n_head")),
        
        # === Vocabulary and tokens ===
        "vocab_size": _get_config_value(config_dict, "vocab_size"),
        "max_position_embeddings": _get_config_value(config_dict, "sequence_length"),
        "bos_token_id": _get_config_value(config_dict, "bos_token_id", 1),
        "eos_token_id": _get_config_value(config_dict, "eos_token_id", 2),
        "pad_token_id": _get_config_value(config_dict, "pad_token_id", 0),
        "tie_word_embeddings": _get_config_value(config_dict, "tie_word_embeddings", False),
        
        # === Activation and normalization functions ===
        "hidden_act": "silu",  # SwiGLU uses SiLU (Swish)
        "rms_norm_eps": _get_config_value(config_dict, "layer_norm_epsilon", 1e-5),
        
        # === RoPE positional encoding ===
        "rope_theta": _get_config_value(config_dict, "rope_theta", 10000.0),
        "rope_scaling": _get_config_value(config_dict, "rope_scaling", None),
        
        # === Regularization and dropout ===
        "attention_dropout": _get_config_value(config_dict, "attention_dropout", _get_config_value(config_dict, "dropout", 0.0)),
        "hidden_dropout": _get_config_value(config_dict, "hidden_dropout", _get_config_value(config_dict, "dropout", 0.0)),
        "dropout": _get_config_value(config_dict, "dropout", 0.0),
        
        # === Initialization ===
        "initializer_range": _get_config_value(config_dict, "initializer_range", 0.02),
        
        # === Optimizations ===
        "use_cache": _get_config_value(config_dict, "use_cache", True),
        "use_flash_attention_2": _get_config_value(config_dict, "use_flash_attention_2", True),
        
        # === Technical configuration ===
        "torch_dtype": "float16" if use_flash_attention and detect_flash_attention()[0] else _get_config_value(config_dict, "torch_dtype", "float32"),
        "pretraining_tp": _get_config_value(config_dict, "pretraining_tp", 1),
    }
    
    config = LlamaConfig(**llama_config)
    
    return config


def detect_flash_attention() -> tuple[bool, str]:
    """Detect FlashAttention-2 availability."""
    try:
        import flash_attn
        return True, f"FlashAttention-2 {flash_attn.__version__} detected"
    except ImportError:
        return False, "FlashAttention-2 not available (see installation)"


def _get_config_value(config, key, default=None):
    """Helper function to get configuration values from either dict or object."""
    if hasattr(config, key):
        return getattr(config, key)
    elif isinstance(config, dict):
        return config.get(key, default)
    else:
        return default

def create_model(config_dict, use_flash_attention: bool = True) -> PreTrainedModel:
    """
    Create a model from scratch based on configuration.
    
    Args:
        config_dict: Model configuration (from JSON) or PretrainConfig object
        use_flash_attention: Use FlashAttention-2 if available
    """
    # Handle both dict and PretrainConfig object
    model_name = _get_config_value(config_dict, 'model_name')
    print(f"🏗️  Creating model {model_name}...")
    
    # Display architectural details
    n_params_est = estimate_parameters(config_dict)
    d_model = _get_config_value(config_dict, 'd_model')
    n_head = _get_config_value(config_dict, 'n_head')
    d_ff = _get_config_value(config_dict, 'd_ff')
    n_layer = _get_config_value(config_dict, 'n_layer')
    
    head_dim = d_model // n_head
    ffn_ratio = d_ff / d_model
    
    print(f"   📊 Architecture: {n_layer} layers, {d_model} hidden, {n_head} heads")
    print(f"   🧮 Estimated parameters: {n_params_est:,}")
    print(f"   🔢 head_dim={head_dim}, ffn_ratio={ffn_ratio:.1f}x")
    
    # GQA verification
    num_kv_heads = _get_config_value(config_dict, 'num_key_value_heads', n_head)
    if num_kv_heads < n_head:
        ratio = n_head / num_kv_heads
        print(f"   🎯 GQA enabled: {n_head}:{num_kv_heads} = {ratio:.1f}:1 ratio")
    
    # Configuration creation
    model_config = create_model_config(config_dict, use_flash_attention)
    
    # Attention management with automatic fallback
    attention_type = "sdpa"  # Default fallback (PyTorch SDPA)
    
    if use_flash_attention:
        fa_available, fa_msg = detect_flash_attention()
        print(f"FlashAttention-2: {fa_msg}")
        
        if fa_available:
            try:
                model_config._attn_implementation = "flash_attention_2"
                attention_type = "flash_attention_2"
                print("✅ FlashAttention-2 enabled")
            except Exception as e:
                print(f"⚠️  FlashAttention-2 failed, fallback to SDPA: {e}")
                model_config._attn_implementation = "sdpa"
        else:
            print("⚠️  FlashAttention-2 not available, fallback to SDPA")
            model_config._attn_implementation = "sdpa"
    else:
        print("FlashAttention-2 manually disabled, using SDPA")
        model_config._attn_implementation = "sdpa"
    
    # Model creation with error handling
    try:
        # Create model - let Accelerate handle dtype conversion
        model = LlamaForCausalLM(model_config)
        if attention_type == "flash_attention_2":
            print("✅ Model created for FlashAttention-2 (dtype managed by Accelerate)")
        else:
            print(f"✅ Model created with attention: {attention_type}")
            
    except Exception as e:
        # Ultimate fallback to standard attention
        print(f"⚠️  Error with {attention_type}, fallback to standard attention: {e}")
        model_config._attn_implementation = "eager"
        model = LlamaForCausalLM(model_config)
        print("Standard attention enabled")
    
    # Weight initialization
    model.apply(lambda module: init_weights(module, model_config))
    
    actual_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created: {actual_params:,} parameters")
    print(f"⚡ Attention: {model_config._attn_implementation}")
    
    # Comparison with estimation
    if abs(actual_params - n_params_est) / n_params_est > 0.1:
        print(f"⚠️  Estimation difference: {n_params_est:,} → {actual_params:,} ({((actual_params/n_params_est)-1)*100:+.1f}%)")
    
    return model


def init_weights(module: nn.Module, config: AutoConfig):
    """Model weight initialization."""
    if isinstance(module, nn.Linear):
        # Xavier/Glorot initialization for linear layers
        torch.nn.init.normal_(module.weight, mean=0.0, std=config.initializer_range)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        # Normal initialization for embeddings
        torch.nn.init.normal_(module.weight, mean=0.0, std=config.initializer_range)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        # Standard initialization for LayerNorm
        torch.nn.init.zeros_(module.bias)
        torch.nn.init.ones_(module.weight)


def save_checkpoint(model: PreTrainedModel, optimizer: torch.optim.Optimizer, 
                   scheduler: torch.optim.lr_scheduler._LRScheduler, 
                   global_step: int, output_dir: str, accelerator=None, scaler=None, 
                   extra_state: dict = None):
    """Save a complete checkpoint with deterministic state using Accelerate's standard format."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if accelerator:
        # Using accelerate's standardized saving method
        accelerator.wait_for_everyone()
        
        if accelerator.is_main_process:
            # Use accelerator.save_state() for standardized checkpoint format
            accelerator.save_state(str(output_path))
            
            # Accelerate peut créer des noms non-standard, créons des liens compatibles
            model_files = list(output_path.glob("model.*"))
            if model_files and not (output_path / "pytorch_model.bin").exists():
                # Créer un lien symbolique pour compatibilité
                try:
                    import os
                    os.link(str(model_files[0]), str(output_path / "pytorch_model.bin"))
                except:
                    # Si le lien échoue, copier le fichier
                    import shutil
                    shutil.copy(str(model_files[0]), str(output_path / "pytorch_model.bin"))
            
            # Save additional metadata and RNG states in separate files for better organization
            # RNG states for reproducibility
            rng_state = {
                "random_state": random.getstate(),
                "numpy_random_state": np.random.get_state(),
                "torch_random_state": torch.get_rng_state(),
                "torch_cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }
            torch.save(rng_state, output_path / "rng_state.pth")
            
            # Save metadata in JSON format
            meta_data = {
                "global_step": global_step,
                "accelerator_mixed_precision": accelerator.mixed_precision,
                "accelerator_gradient_accumulation_steps": accelerator.gradient_accumulation_steps,
            }
            
            # Add extra state to metadata
            if extra_state is not None:
                meta_data.update(extra_state)
            
            with open(output_path / "meta.json", 'w') as f:
                json.dump(meta_data, f, indent=2)

            # Save model config for HuggingFace compatibility
            if hasattr(model, 'config'):
                model.config.save_pretrained(output_path)

            print(f"✅ Complete checkpoint saved to {output_path}")
            print(f"   - Model, optimizer, scheduler: Accelerate format")
            print(f"   - RNG states: rng_state.pth")
            print(f"   - Metadata: meta.json (step {global_step})")
            if hasattr(model, 'config'):
                print(f"   - Model config: config.json")
    
    else:
        # Fallback: manual saving when not using accelerate
        # Model saving
        model.save_pretrained(output_path)
        
        # Save components separately for consistency with Accelerate format
        torch.save(optimizer.state_dict(), output_path / "optimizer.pt")
        torch.save(scheduler.state_dict(), output_path / "scheduler.pt")
        
        if scaler is not None:
            torch.save(scaler.state_dict(), output_path / "scaler.pt")
        
        # RNG states
        rng_state = {
            "random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "torch_cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        torch.save(rng_state, output_path / "rng_state.pth")
        
        # Metadata
        meta_data = {
            "global_step": global_step,
            "mixed_precision": "no",
            "gradient_accumulation_steps": 1,
        }
        if extra_state is not None:
            meta_data.update(extra_state)
        
        with open(output_path / "meta.json", 'w') as f:
            json.dump(meta_data, f, indent=2)
        
        print(f"✅ Checkpoint saved to {output_path} (manual format)")


def load_checkpoint(model: PreTrainedModel, optimizer: torch.optim.Optimizer,
                   scheduler: torch.optim.lr_scheduler._LRScheduler,
                   checkpoint_dir: str, accelerator=None, scaler=None) -> tuple[int, dict]:
    """Load a complete checkpoint with deterministic restoration using Accelerate's standard format."""
    
    checkpoint_path = Path(checkpoint_dir)
    
    # Check for new format first (meta.json + rng_state.pth)
    meta_path = checkpoint_path / "meta.json"
    rng_path = checkpoint_path / "rng_state.pth"
    
    # Try loading from Accelerate standard format
    if meta_path.exists() and accelerator:
        try:
            # Check if we have the standard Accelerate files
            model_files = list(checkpoint_path.glob("model.*")) + list(checkpoint_path.glob("pytorch_model.*"))
            opt_files = list(checkpoint_path.glob("optimizer.*"))
            sched_files = list(checkpoint_path.glob("scheduler.*"))

            if model_files and opt_files and sched_files:
                # Load metadata first
                with open(meta_path, 'r') as f:
                    meta_data = json.load(f)
                global_step = meta_data["global_step"]

                # Try to load using accelerator.load_state() for model, optimizer, scheduler, scaler
                try:
                    accelerator.load_state(str(checkpoint_path))
                    print(f"✅ Loaded using accelerator.load_state()")
                except Exception as accel_e:
                    print(f"⚠️  accelerator.load_state() failed: {accel_e}")
                    # Manually load individual components
                    print("📁 Loading components manually...")

                    # Load model state
                    if checkpoint_path.joinpath("model.safetensors").exists():
                        from safetensors.torch import load_file
                        model.load_state_dict(load_file(checkpoint_path / "model.safetensors"))
                    elif checkpoint_path.joinpath("pytorch_model.bin").exists():
                        model.load_state_dict(torch.load(checkpoint_path / "pytorch_model.bin", weights_only=False))

                    # Load optimizer state
                    if checkpoint_path.joinpath("optimizer.bin").exists():
                        optimizer.load_state_dict(torch.load(checkpoint_path / "optimizer.bin", weights_only=False))

                    # Load scheduler state
                    if checkpoint_path.joinpath("scheduler.bin").exists():
                        scheduler.load_state_dict(torch.load(checkpoint_path / "scheduler.bin", weights_only=False))

                    print("✅ Manual loading completed")

                # Restore RNG states for reproducibility
                if rng_path.exists():
                    try:
                        rng_state = torch.load(rng_path, map_location="cpu", weights_only=False)

                        if "random_state" in rng_state:
                            try:
                                random.setstate(rng_state["random_state"])
                            except (TypeError, ValueError) as e:
                                print(f"⚠️  Could not restore random state: {e}")

                        if "numpy_random_state" in rng_state:
                            try:
                                np.random.set_state(rng_state["numpy_random_state"])
                            except (TypeError, ValueError) as e:
                                print(f"⚠️  Could not restore numpy random state: {e}")

                        if "torch_random_state" in rng_state:
                            try:
                                torch.set_rng_state(rng_state["torch_random_state"])
                            except (TypeError, ValueError) as e:
                                print(f"⚠️  Could not restore torch random state: {e}")

                        if "torch_cuda_random_state" in rng_state and torch.cuda.is_available():
                            if rng_state["torch_cuda_random_state"] is not None:
                                try:
                                    torch.cuda.set_rng_state_all(rng_state["torch_cuda_random_state"])
                                except (TypeError, ValueError) as e:
                                    print(f"⚠️  Could not restore cuda random state: {e}")

                        print("✅ RNG states restored for reproducibility")
                    except Exception as e:
                        print(f"⚠️  Failed to load RNG states: {e}, continuing without RNG restoration")

                # Extract extra state (remove standard keys)
                extra_state = {k: v for k, v in meta_data.items()
                              if k not in ["global_step", "accelerator_mixed_precision", "accelerator_gradient_accumulation_steps"]}

                print(f"✅ Checkpoint loaded from {checkpoint_path} using Accelerate format")
                print(f"   Step: {global_step}")
                print(f"   Mixed precision: {meta_data.get('accelerator_mixed_precision', 'unknown')}")

                if extra_state:
                    print(f"   Extra state: {list(extra_state.keys())}")

                return global_step, extra_state
            else:
                print(f"⚠️  Incomplete Accelerate checkpoint (missing files)")
                print(f"   Model files: {[f.name for f in model_files]}")
                print(f"   Optimizer files: {[f.name for f in opt_files]}")
                print(f"   Scheduler files: {[f.name for f in sched_files]}")
                print("   Falling back to legacy format...")

        except Exception as e:
            print(f"⚠️  Failed to load with Accelerate format: {e}")
            print("   Falling back to legacy format...")
    
    # Fallback: try loading legacy format (training_state.pt)
    training_state_path = checkpoint_path / "training_state.pt"
    
    if training_state_path.exists():
        print(f"📁 Loading from legacy format: {training_state_path}")
        
        # Loading training state
        training_state = torch.load(training_state_path, map_location="cpu", weights_only=False)
        
        # Restoring training states
        optimizer.load_state_dict(training_state["optimizer_state_dict"])
        scheduler.load_state_dict(training_state["scheduler_state_dict"])
        global_step = training_state["global_step"]
        
        # Restoring random states for reproducibility
        if "random_state" in training_state:
            random.setstate(training_state["random_state"])
        if "numpy_random_state" in training_state:
            np.random.set_state(training_state["numpy_random_state"])
        if "torch_random_state" in training_state:
            torch.set_rng_state(training_state["torch_random_state"])
        if "torch_cuda_random_state" in training_state and torch.cuda.is_available():
            if training_state["torch_cuda_random_state"] is not None:
                torch.cuda.set_rng_state_all(training_state["torch_cuda_random_state"])
        
        # Restoring scaler if provided
        if scaler is not None and "scaler_state_dict" in training_state:
            scaler.load_state_dict(training_state["scaler_state_dict"])
            print("✅ Scaler state restored")
        
        # Loading the model
        if checkpoint_path.exists() and (checkpoint_path / "pytorch_model.bin").exists():
            if accelerator:
                # Try using accelerate if available
                try:
                    accelerator.load_state(str(checkpoint_path))
                except:
                    # Manual loading as fallback
                    state_dict = torch.load(checkpoint_path / "pytorch_model.bin", map_location="cpu")
                    model.load_state_dict(state_dict)
            else:
                # Standard loading
                state_dict = torch.load(checkpoint_path / "pytorch_model.bin", map_location="cpu")
                model.load_state_dict(state_dict)
        
        # Extract extra state if available
        extra_state = training_state.get("extra_state", {})
        
        print(f"✅ Legacy checkpoint loaded from {checkpoint_path}, step {global_step}")
        print("✅ Random states restored for reproducibility")
        
        if extra_state:
            print(f"✅ Extra state loaded: {list(extra_state.keys())}")
        
        return global_step, extra_state
    
    # If neither format found
    else:
        raise FileNotFoundError(
            f"No valid checkpoint found in {checkpoint_path}\n"
            f"Expected either 'meta.json' (new format) or 'training_state.pt' (legacy format)"
        )


def load_pretrained_model(model_path: str, device: str = "auto") -> PreTrainedModel:
    """Load a pre-trained model."""
    
    model_path = Path(model_path)
    
    # Device detection
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model from {model_path} on {device}...")
    
    try:
        # Attempt loading with AutoModel
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
            use_flash_attention_2=True
        )
        
        if device != "cuda":
            model = model.to(device)
            
    except Exception as e:
        print(f"Error loading with AutoModel: {e}")
        print("Attempting fallback: manual loading with create_model()...")

        # Fallback: manual loading
        config_path = model_path / "config.json"
        if not config_path.exists():
            raise ValueError(f"Unable to load model from {model_path}: config.json not found")

        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        print(f"Creating model from config: {config_dict.get('n_layer', '?')} layers, "
              f"{config_dict.get('d_model', '?')} dims")
        model = create_model(config_dict)

        # Loading weights if available
        weights_path = model_path / "pytorch_model.bin"
        if not weights_path.exists():
            print(f"Warning: No weights file found at {weights_path}")
            raise ValueError(f"Unable to load model weights from {model_path}")

        print(f"Loading weights from {weights_path}...")
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        print("✓ Weights loaded successfully")

        model = model.to(device)

    # Activate evaluation mode by default
    model.eval()

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters on {device}")

    return model


def estimate_parameters(config_dict) -> int:
    """
    Estimate the number of parameters based on configuration.
    
    Calculation based on LLaMA architecture:
    - Token embeddings: vocab_size × d_model
    - Transformer layers: n_layer × (attention + ffn + norms)
    - Output head: d_model × vocab_size (if no tie_word_embeddings)
    
    Args:
        config_dict: Model configuration
        
    Returns:
        int: Estimated number of parameters
    """
    d_model = _get_config_value(config_dict, 'd_model')
    n_layer = _get_config_value(config_dict, 'n_layer')
    d_ff = _get_config_value(config_dict, 'd_ff')
    vocab_size = _get_config_value(config_dict, 'vocab_size')
    n_head = _get_config_value(config_dict, 'n_head')
    
    # Token embeddings
    token_embeddings = vocab_size * d_model
    
    # Per transformer layer
    # Attention: Q, K, V projections + output projection
    attention_params = d_model * d_model * 4  # qkv_proj + o_proj
    
    # Feed-forward: SwiGLU has 3 projections (up, gate, down)
    ffn_params = d_model * d_ff + d_ff * d_model + d_model * d_ff  # up + gate + down
    
    # RMSNorm: only scale parameters (γ)
    norm_params = d_model * 2  # attention_norm + ffn_norm
    
    # Total per layer
    per_layer = attention_params + ffn_params + norm_params
    
    # All layers
    all_layers = n_layer * per_layer
    
    # Final norm
    final_norm = d_model
    
    # Output head (lm_head)
    tied_embeddings = _get_config_value(config_dict, 'tie_word_embeddings', False)
    output_head = 0 if tied_embeddings else d_model * vocab_size
    
    total = token_embeddings + all_layers + final_norm + output_head
    
    return int(total)


def get_model_size(model: PreTrainedModel) -> Dict[str, int]:
    """Calculate model size in parameters and memory."""
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Memory estimation (in MB)
    param_size = total_params * 4  # 4 bytes per float32 parameter
    memory_mb = param_size / (1024 * 1024)
    
    # GPU memory estimation needed (with activations)
    # Approximation: ~3-4x the parameter size
    gpu_memory_mb = memory_mb * 3.5
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "memory_mb": int(memory_mb),
        "estimated_gpu_memory_mb": int(gpu_memory_mb)
    }


def optimize_model_for_inference(model: PreTrainedModel, enable_torch_compile: bool = False) -> PreTrainedModel:
    """Optimize model for inference."""
    
    # Evaluation mode
    model.eval()
    
    # Disable gradient computation
    for param in model.parameters():
        param.requires_grad_(False)
    
    # Compile with torch.compile if requested (PyTorch 2.0+)
    if enable_torch_compile and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode="reduce-overhead")
    
    # Fuse operations if possible
    if hasattr(torch.jit, 'optimize_for_inference'):
        model = torch.jit.optimize_for_inference(model)
    
    print("Model optimized for inference")
    return model


def setup_device_and_precision(model: PreTrainedModel, use_fp16: bool = True, 
                              device: str = "auto") -> PreTrainedModel:
    """Configure device and precision of the model."""
    
    # Device detection
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            print(f"GPU detected: {torch.cuda.get_device_name()}")
            print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            device = "cpu"
            print("Using CPU")
    
    # Move to device
    model = model.to(device)
    
    # Configure precision
    if use_fp16 and device == "cuda":
        model = model.half()
        print("FP16 precision enabled")
    else:
        print("FP32 precision used")
    
    return model


def estimate_training_memory(config_dict: Dict, batch_size: int = 8, 
                           sequence_length: int = 1024) -> Dict[str, float]:
    """Estimate memory required for training."""
    
    # Approximate calculation of parameter count
    d_model = config_dict["d_model"]
    n_layer = config_dict["n_layer"]
    vocab_size = config_dict["vocab_size"]
    d_ff = config_dict["d_ff"]
    
    # Parameter estimation
    embedding_params = vocab_size * d_model
    attention_params = n_layer * d_model * d_model * 4  # Q, K, V, O projections
    ffn_params = n_layer * (d_model * d_ff * 2 + d_ff + d_model)  # up, down, gate
    norm_params = n_layer * d_model * 2  # layer norms
    
    total_params = embedding_params + attention_params + ffn_params + norm_params
    
    # Memory for parameters (in GB)
    param_memory = total_params * 4 / 1e9  # 4 bytes per float32
    
    # Memory for gradients (same size as parameters)
    gradient_memory = param_memory
    
    # Memory for Adam optimizer (2x parameters for momentum and variance)
    optimizer_memory = param_memory * 2
    
    # Memory for activations
    activation_memory = batch_size * sequence_length * d_model * n_layer * 4 / 1e9
    
    # Total with safety buffer
    total_memory = (param_memory + gradient_memory + optimizer_memory + activation_memory) * 1.2
    
    return {
        "parameters_gb": param_memory,
        "gradients_gb": gradient_memory,
        "optimizer_gb": optimizer_memory,
        "activations_gb": activation_memory,
        "total_estimated_gb": total_memory,
        "estimated_parameters": int(total_params)
    }


def print_model_summary(model: PreTrainedModel, config_dict: Optional[Dict] = None):
    """Display a model summary."""
    
    size_info = get_model_size(model)
    
    print("\n" + "="*50)
    print("MODEL SUMMARY")
    print("="*50)
    
    if config_dict:
        print(f"Architecture: {config_dict.get('model_name', 'Unknown')}")
        print(f"Layers: {config_dict.get('n_layer', 'N/A')}")
        print(f"Dimension: {config_dict.get('d_model', 'N/A')}")
        print(f"Attention heads: {config_dict.get('n_head', 'N/A')}")
        print(f"Vocabulary: {config_dict.get('vocab_size', 'N/A'):,}")
        print(f"Sequence length: {config_dict.get('sequence_length', 'N/A')}")
    
    print(f"Total parameters: {size_info['total_parameters']:,}")
    print(f"Trainable parameters: {size_info['trainable_parameters']:,}")
    print(f"Model memory: {size_info['memory_mb']} MB")
    print(f"Estimated GPU memory: {size_info['estimated_gpu_memory_mb']} MB")
    print(f"Device: {next(model.parameters()).device}")
    print(f"Dtype: {next(model.parameters()).dtype}")
    print("="*50 + "\n")