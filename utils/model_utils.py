"""
Utilitaires pour la création, le chargement et la gestion des modèles.
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


def create_model_config(config_dict: Dict) -> AutoConfig:
    """
    Crée une configuration LLaMA à partir d'un dictionnaire.
    
    Architecture suivie: LLaMA (decoder-only transformer)
    - RMSNorm pour la normalisation
    - SwiGLU pour l'activation FFN
    - RoPE pour l'encodage positionnel
    - Pas de termes de biais
    - Support GQA (Grouped Query Attention)
    
    Args:
        config_dict: Configuration du modèle avec les paramètres architecturaux
        
    Returns:
        AutoConfig: Configuration HuggingFace compatible LLaMA
    """
    
    # Configuration LLaMA avec tous les paramètres architecturaux
    llama_config = {
        # === Architecture de base ===
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "transformers_version": "4.40.0",
        
        # === Dimensions du modèle ===
        "hidden_size": config_dict["d_model"],
        "intermediate_size": config_dict["d_ff"], 
        "num_hidden_layers": config_dict["n_layer"],
        "num_attention_heads": config_dict["n_head"],
        
        # === Grouped Query Attention (GQA) ===
        # Par défaut: MHA (num_kv_heads = num_heads)
        # Pour GQA: réduire num_key_value_heads (ex: 4 pour 8 têtes = 2:1 ratio)
        "num_key_value_heads": config_dict.get("num_key_value_heads", config_dict["n_head"]),
        
        # === Vocabulaire et tokens ===
        "vocab_size": config_dict["vocab_size"],
        "max_position_embeddings": config_dict["sequence_length"],
        "bos_token_id": config_dict.get("bos_token_id", 1),
        "eos_token_id": config_dict.get("eos_token_id", 2),
        "pad_token_id": config_dict.get("pad_token_id", 0),
        "tie_word_embeddings": config_dict.get("tie_word_embeddings", False),
        
        # === Fonctions d'activation et normalisation ===
        "hidden_act": "silu",  # SwiGLU utilise SiLU (Swish)
        "rms_norm_eps": config_dict.get("layer_norm_epsilon", 1e-5),
        
        # === Encodage positionnel RoPE ===
        "rope_theta": config_dict.get("rope_theta", 10000.0),
        "rope_scaling": config_dict.get("rope_scaling", None),
        
        # === Régularisation et dropout ===
        "attention_dropout": config_dict.get("attention_dropout", config_dict.get("dropout", 0.0)),
        "hidden_dropout": config_dict.get("hidden_dropout", config_dict.get("dropout", 0.0)),
        "dropout": config_dict.get("dropout", 0.0),
        
        # === Initialisation ===
        "initializer_range": config_dict.get("initializer_range", 0.02),
        
        # === Optimisations ===
        "use_cache": config_dict.get("use_cache", True),
        "use_flash_attention_2": config_dict.get("use_flash_attention_2", True),
        
        # === Configuration technique ===
        "torch_dtype": config_dict.get("torch_dtype", "float32"),
        "pretraining_tp": config_dict.get("pretraining_tp", 1),
    }
    
    config = AutoConfig.from_dict(llama_config)
    
    return config


def detect_flash_attention() -> tuple[bool, str]:
    """Détecte la disponibilité de FlashAttention-2."""
    try:
        import flash_attn
        return True, f"FlashAttention-2 {flash_attn.__version__} détecté"
    except ImportError:
        return False, "FlashAttention-2 non disponible (voir installation)"


def create_model(config_dict: Dict, use_flash_attention: bool = True) -> PreTrainedModel:
    """
    Crée un modèle from scratch basé sur la configuration.
    
    Args:
        config_dict: Configuration du modèle (depuis le JSON)
        use_flash_attention: Utiliser FlashAttention-2 si disponible
    """
    print(f"🏗️  Création du modèle {config_dict['model_name']}...")
    
    # Affichage des détails architecturaux
    n_params_est = estimate_parameters(config_dict)
    head_dim = config_dict['d_model'] // config_dict['n_head']
    ffn_ratio = config_dict['d_ff'] / config_dict['d_model']
    
    print(f"   📊 Architecture: {config_dict['n_layer']} layers, {config_dict['d_model']} hidden, {config_dict['n_head']} heads")
    print(f"   🧮 Paramètres estimés: {n_params_est:,}")
    print(f"   🔢 head_dim={head_dim}, ffn_ratio={ffn_ratio:.1f}x")
    
    # Vérification GQA
    num_kv_heads = config_dict.get('num_key_value_heads', config_dict['n_head'])
    if num_kv_heads < config_dict['n_head']:
        ratio = config_dict['n_head'] / num_kv_heads
        print(f"   🎯 GQA activé: {config_dict['n_head']}:{num_kv_heads} = {ratio:.1f}:1 ratio")
    
    # Création de la configuration
    model_config = create_model_config(config_dict)
    
    # Gestion de l'attention avec fallback automatique
    attention_type = "sdpa"  # Fallback par défaut (PyTorch SDPA)
    
    if use_flash_attention:
        fa_available, fa_msg = detect_flash_attention()
        print(f"FlashAttention-2: {fa_msg}")
        
        if fa_available:
            try:
                model_config._attn_implementation = "flash_attention_2"
                attention_type = "flash_attention_2"
                print("✅ FlashAttention-2 activé")
            except Exception as e:
                print(f"⚠️  FlashAttention-2 échec, fallback vers SDPA: {e}")
                model_config._attn_implementation = "sdpa"
        else:
            print("⚠️  FlashAttention-2 non disponible, fallback vers SDPA")
            model_config._attn_implementation = "sdpa"
    else:
        print("FlashAttention-2 désactivé manuellement, utilisation SDPA")
        model_config._attn_implementation = "sdpa"
    
    # Création du modèle avec gestion d'erreur
    try:
        model = LlamaForCausalLM(model_config)
    except Exception as e:
        # Fallback ultime vers attention standard
        print(f"⚠️  Erreur avec {attention_type}, fallback vers attention standard: {e}")
        model_config._attn_implementation = "eager"
        model = LlamaForCausalLM(model_config)
        print("Attention standard activée")
    
    # Initialisation des poids
    model.apply(lambda module: init_weights(module, model_config))
    
    actual_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Modèle créé: {actual_params:,} paramètres")
    print(f"⚡ Attention: {model_config._attn_implementation}")
    
    # Comparaison avec l'estimation
    if abs(actual_params - n_params_est) / n_params_est > 0.1:
        print(f"⚠️  Différence estimation: {n_params_est:,} → {actual_params:,} ({((actual_params/n_params_est)-1)*100:+.1f}%)")
    
    return model


def init_weights(module: nn.Module, config: AutoConfig):
    """Initialisation des poids du modèle."""
    if isinstance(module, nn.Linear):
        # Initialisation Xavier/Glorot pour les couches linéaires
        torch.nn.init.normal_(module.weight, mean=0.0, std=config.initializer_range)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        # Initialisation normale pour les embeddings
        torch.nn.init.normal_(module.weight, mean=0.0, std=config.initializer_range)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        # Initialisation standard pour LayerNorm
        torch.nn.init.zeros_(module.bias)
        torch.nn.init.ones_(module.weight)


def save_checkpoint(model: PreTrainedModel, optimizer: torch.optim.Optimizer, 
                   scheduler: torch.optim.lr_scheduler._LRScheduler, 
                   global_step: int, output_dir: str, accelerator=None, scaler=None):
    """Sauvegarde un checkpoint complet avec état déterministe."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if accelerator:
        # Utilisation d'accelerate pour la sauvegarde
        accelerator.wait_for_everyone()
        
        # Sauvegarde du modèle
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            output_path,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model)
        )
        
        # Sauvegarde de l'état d'entraînement
        if accelerator.is_main_process:
            training_state = {
                "global_step": global_step,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                # État déterministe pour reproductibilité
                "random_state": random.getstate(),
                "numpy_random_state": np.random.get_state(),
                "torch_random_state": torch.get_rng_state(),
                "torch_cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }
            
            # Ajout du scaler si fourni
            if scaler is not None:
                training_state["scaler_state_dict"] = scaler.state_dict()
            
            torch.save(training_state, output_path / "training_state.pt")
            
            print(f"Checkpoint sauvegardé dans {output_path}")
    
    else:
        # Sauvegarde standard
        model.save_pretrained(output_path)
        
        training_state = {
            "global_step": global_step,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            # État déterministe pour reproductibilité
            "random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "torch_cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        
        # Ajout du scaler si fourni
        if scaler is not None:
            training_state["scaler_state_dict"] = scaler.state_dict()
        
        torch.save(training_state, output_path / "training_state.pt")
        print(f"Checkpoint sauvegardé dans {output_path}")


def load_checkpoint(model: PreTrainedModel, optimizer: torch.optim.Optimizer,
                   scheduler: torch.optim.lr_scheduler._LRScheduler,
                   checkpoint_dir: str, accelerator=None, scaler=None) -> int:
    """Charge un checkpoint complet avec restauration déterministe."""
    
    checkpoint_path = Path(checkpoint_dir)
    training_state_path = checkpoint_path / "training_state.pt"
    
    if not training_state_path.exists():
        raise FileNotFoundError(f"État d'entraînement non trouvé: {training_state_path}")
    
    # Chargement de l'état d'entraînement
    training_state = torch.load(training_state_path, map_location="cpu")
    
    # Restauration des états d'entraînement
    optimizer.load_state_dict(training_state["optimizer_state_dict"])
    scheduler.load_state_dict(training_state["scheduler_state_dict"])
    global_step = training_state["global_step"]
    
    # Restauration des états aléatoires pour reproductibilité
    if "random_state" in training_state:
        random.setstate(training_state["random_state"])
    if "numpy_random_state" in training_state:
        np.random.set_state(training_state["numpy_random_state"])
    if "torch_random_state" in training_state:
        torch.set_rng_state(training_state["torch_random_state"])
    if "torch_cuda_random_state" in training_state and torch.cuda.is_available():
        if training_state["torch_cuda_random_state"] is not None:
            torch.cuda.set_rng_state_all(training_state["torch_cuda_random_state"])
    
    # Restauration du scaler si fourni
    if scaler is not None and "scaler_state_dict" in training_state:
        scaler.load_state_dict(training_state["scaler_state_dict"])
        print("État du scaler restauré")
    
    # Chargement du modèle
    if checkpoint_path.exists() and (checkpoint_path / "pytorch_model.bin").exists():
        if accelerator:
            # Utilisation d'accelerate
            accelerator.load_state(str(checkpoint_path))
        else:
            # Chargement standard
            state_dict = torch.load(checkpoint_path / "pytorch_model.bin", map_location="cpu")
            model.load_state_dict(state_dict)
    
    print(f"Checkpoint chargé depuis {checkpoint_path}, step {global_step}")
    print("✅ États aléatoires restaurés pour reproductibilité")
    return global_step


def load_pretrained_model(model_path: str, device: str = "auto") -> PreTrainedModel:
    """Charge un modèle pré-entraîné."""
    
    model_path = Path(model_path)
    
    # Détection du device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Chargement du modèle depuis {model_path} sur {device}...")
    
    try:
        # Tentative de chargement avec AutoModel
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
        print(f"Erreur lors du chargement avec AutoModel: {e}")
        
        # Fallback: chargement manuel
        config_path = model_path / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            model = create_model(config_dict)
            
            # Chargement des poids si disponibles
            weights_path = model_path / "pytorch_model.bin"
            if weights_path.exists():
                state_dict = torch.load(weights_path, map_location=device)
                model.load_state_dict(state_dict)
            
            model = model.to(device)
        else:
            raise ValueError(f"Impossible de charger le modèle depuis {model_path}")
    
    # Activation du mode évaluation par défaut
    model.eval()
    
    print(f"Modèle chargé: {sum(p.numel() for p in model.parameters()):,} paramètres sur {device}")
    
    return model


def estimate_parameters(config_dict: Dict) -> int:
    """
    Estime le nombre de paramètres basé sur la configuration.
    
    Calcul basé sur l'architecture LLaMA:
    - Token embeddings: vocab_size × d_model
    - Transformer layers: n_layer × (attention + ffn + norms)
    - Output head: d_model × vocab_size (si pas de tie_word_embeddings)
    
    Args:
        config_dict: Configuration du modèle
        
    Returns:
        int: Nombre estimé de paramètres
    """
    d_model = config_dict['d_model']
    n_layer = config_dict['n_layer']
    d_ff = config_dict['d_ff']
    vocab_size = config_dict['vocab_size']
    n_head = config_dict['n_head']
    
    # Token embeddings
    token_embeddings = vocab_size * d_model
    
    # Par couche transformer
    # Attention: Q, K, V projections + output projection
    attention_params = d_model * d_model * 4  # qkv_proj + o_proj
    
    # Feed-forward: SwiGLU a 3 projections (up, gate, down)
    ffn_params = d_model * d_ff + d_ff * d_model + d_model * d_ff  # up + gate + down
    
    # RMSNorm: seulement les paramètres d'échelle (γ)
    norm_params = d_model * 2  # attention_norm + ffn_norm
    
    # Total par couche
    per_layer = attention_params + ffn_params + norm_params
    
    # Toutes les couches
    all_layers = n_layer * per_layer
    
    # Norm finale
    final_norm = d_model
    
    # Output head (lm_head)
    tied_embeddings = config_dict.get('tie_word_embeddings', False)
    output_head = 0 if tied_embeddings else d_model * vocab_size
    
    total = token_embeddings + all_layers + final_norm + output_head
    
    return int(total)


def get_model_size(model: PreTrainedModel) -> Dict[str, int]:
    """Calcule la taille du modèle en paramètres et en mémoire."""
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimation de la mémoire (en MB)
    param_size = total_params * 4  # 4 bytes par paramètre float32
    memory_mb = param_size / (1024 * 1024)
    
    # Estimation de la mémoire GPU nécessaire (avec activations)
    # Approximation: ~3-4x la taille des paramètres
    gpu_memory_mb = memory_mb * 3.5
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "memory_mb": int(memory_mb),
        "estimated_gpu_memory_mb": int(gpu_memory_mb)
    }


def optimize_model_for_inference(model: PreTrainedModel, enable_torch_compile: bool = False) -> PreTrainedModel:
    """Optimise le modèle pour l'inférence."""
    
    # Mode évaluation
    model.eval()
    
    # Désactivation du calcul des gradients
    for param in model.parameters():
        param.requires_grad_(False)
    
    # Compilation avec torch.compile si demandé (PyTorch 2.0+)
    if enable_torch_compile and hasattr(torch, 'compile'):
        print("Compilation du modèle avec torch.compile...")
        model = torch.compile(model, mode="reduce-overhead")
    
    # Fusion des opérations si possible
    if hasattr(torch.jit, 'optimize_for_inference'):
        model = torch.jit.optimize_for_inference(model)
    
    print("Modèle optimisé pour l'inférence")
    return model


def setup_device_and_precision(model: PreTrainedModel, use_fp16: bool = True, 
                              device: str = "auto") -> PreTrainedModel:
    """Configure le device et la précision du modèle."""
    
    # Détection du device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            print(f"GPU détecté: {torch.cuda.get_device_name()}")
            print(f"Mémoire GPU disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            device = "cpu"
            print("Utilisation du CPU")
    
    # Déplacement vers le device
    model = model.to(device)
    
    # Configuration de la précision
    if use_fp16 and device == "cuda":
        model = model.half()
        print("Précision FP16 activée")
    else:
        print("Précision FP32 utilisée")
    
    return model


def estimate_training_memory(config_dict: Dict, batch_size: int = 8, 
                           sequence_length: int = 1024) -> Dict[str, float]:
    """Estime la mémoire nécessaire pour l'entraînement."""
    
    # Calcul approximatif du nombre de paramètres
    d_model = config_dict["d_model"]
    n_layer = config_dict["n_layer"]
    vocab_size = config_dict["vocab_size"]
    d_ff = config_dict["d_ff"]
    
    # Estimation des paramètres
    embedding_params = vocab_size * d_model
    attention_params = n_layer * d_model * d_model * 4  # Q, K, V, O projections
    ffn_params = n_layer * (d_model * d_ff * 2 + d_ff + d_model)  # up, down, gate
    norm_params = n_layer * d_model * 2  # layer norms
    
    total_params = embedding_params + attention_params + ffn_params + norm_params
    
    # Mémoire pour les paramètres (en GB)
    param_memory = total_params * 4 / 1e9  # 4 bytes par float32
    
    # Mémoire pour les gradients (même taille que les paramètres)
    gradient_memory = param_memory
    
    # Mémoire pour l'optimiseur Adam (2x paramètres pour momentum et variance)
    optimizer_memory = param_memory * 2
    
    # Mémoire pour les activations
    activation_memory = batch_size * sequence_length * d_model * n_layer * 4 / 1e9
    
    # Total avec buffer de sécurité
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
    """Affiche un résumé du modèle."""
    
    size_info = get_model_size(model)
    
    print("\n" + "="*50)
    print("RÉSUMÉ DU MODÈLE")
    print("="*50)
    
    if config_dict:
        print(f"Architecture: {config_dict.get('model_name', 'Unknown')}")
        print(f"Couches: {config_dict.get('n_layer', 'N/A')}")
        print(f"Dimension: {config_dict.get('d_model', 'N/A')}")
        print(f"Têtes d'attention: {config_dict.get('n_head', 'N/A')}")
        print(f"Vocabulaire: {config_dict.get('vocab_size', 'N/A'):,}")
        print(f"Longueur de séquence: {config_dict.get('sequence_length', 'N/A')}")
    
    print(f"Paramètres totaux: {size_info['total_parameters']:,}")
    print(f"Paramètres entraînables: {size_info['trainable_parameters']:,}")
    print(f"Mémoire modèle: {size_info['memory_mb']} MB")
    print(f"Mémoire GPU estimée: {size_info['estimated_gpu_memory_mb']} MB")
    print(f"Device: {next(model.parameters()).device}")
    print(f"Dtype: {next(model.parameters()).dtype}")
    print("="*50 + "\n")