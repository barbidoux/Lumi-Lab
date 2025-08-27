#!/usr/bin/env python3
"""
Script d'entraînement from scratch d'un mini-LLM.
Utilise accelerate pour la gestion de l'entraînement et des checkpoints.
"""

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset_utils import TokenizedDataset
from utils.model_utils import create_model, save_checkpoint, load_checkpoint


def set_deterministic_training(seed: int = 42):
    """Configure l'entraînement déterministe avec seed fixe."""
    print(f"Configuration entraînement déterministe avec seed {seed}")
    
    # Seeds pour reproducibilité
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Configuration CUDNN pour reproducibilité (plus lent mais déterministe)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Générateur pour DataLoader
    g = torch.Generator()
    g.manual_seed(seed)
    
    print("✅ Entraînement déterministe configuré")
    return g


class PretrainConfig:
    """Configuration pour l'entraînement de pré-entraînement."""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        
        # Paramètres du modèle
        self.model_name = model_config['model_name']
        self.n_layer = model_config['n_layer']
        self.d_model = model_config['d_model']
        self.n_head = model_config['n_head']
        self.d_ff = model_config['d_ff']
        self.vocab_size = model_config['vocab_size']
        self.sequence_length = model_config['sequence_length']
        self.dropout = model_config.get('dropout', 0.1)
        
        # Hyperparamètres d'entraînement (par défaut)
        self.learning_rate = 3e-4
        self.batch_size = 8
        self.gradient_accumulation_steps = 8
        self.num_train_epochs = 1
        self.max_steps = -1
        self.warmup_steps = 2000
        self.weight_decay = 0.1
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.max_grad_norm = 1.0
        self.logging_steps = 100
        self.save_steps = 5000
        self.eval_steps = 1000


def calculate_perplexity(model: nn.Module, dataloader: DataLoader, device: torch.device, accelerator = None) -> float:
    """Calcule la perplexité sur un dataset de validation."""
    model.eval()
    
    # For FlashAttention models, temporarily switch to eager attention and float32
    original_dtype = None
    original_attn_implementation = None
    
    if hasattr(model, 'dtype') and model.dtype == torch.float16:
        original_dtype = model.dtype
        
        # Switch to eager attention to avoid FlashAttention dtype issues
        if hasattr(model, 'config') and hasattr(model.config, '_attn_implementation'):
            original_attn_implementation = model.config._attn_implementation
            model.config._attn_implementation = "eager"
            if accelerator:
                accelerator.print("Debug: Switching to eager attention for evaluation")
        
        model = model.to(torch.float32)
        if accelerator:
            accelerator.print("Debug: Converting model to float32 for stable evaluation")
            
            # Check for NaN/Inf in model weights
            nan_params = 0
            inf_params = 0
            total_params = 0
            for name, param in model.named_parameters():
                total_params += param.numel()
                if torch.isnan(param).any():
                    nan_params += torch.isnan(param).sum().item()
                if torch.isinf(param).any():
                    inf_params += torch.isinf(param).sum().item()
            
            if nan_params > 0 or inf_params > 0:
                accelerator.print(f"Debug: Found {nan_params} NaN and {inf_params} Inf values in model weights out of {total_params} total parameters!")
            else:
                accelerator.print("Debug: Model weights are clean (no NaN/Inf)")
    
    total_loss = 0.0
    total_tokens = 0
    batch_count = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calcul perplexité"):
            batch_count += 1
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # Convertir la loss en float32 pour éviter les problèmes numériques
            loss_value = float(loss.item())
            
            if accelerator and batch_count == 1:  # Debug only first batch
                accelerator.print(f"Debug: loss_value = {loss_value}")
            
            # Vérifier si la loss est valide
            if math.isnan(loss_value) or math.isinf(loss_value):
                if accelerator:
                    accelerator.print(f"Debug: Skipping batch due to invalid loss: {loss_value}")
                continue
            
            # Compter seulement les tokens non-masqués
            valid_tokens = (labels != -100).sum().item()
            total_loss += loss_value * valid_tokens
            total_tokens += valid_tokens
    
    if accelerator:
        accelerator.print(f"Debug: total_tokens = {total_tokens}, batch_count = {batch_count}")
    
    if total_tokens == 0:
        if accelerator:
            accelerator.print("Debug: Returning inf due to total_tokens == 0")
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    
    # Debug: afficher la loss moyenne
    if accelerator:
        accelerator.print(f"Debug: avg_loss = {avg_loss:.4f}, total_tokens = {total_tokens}")
    else:
        print(f"Debug: avg_loss = {avg_loss:.4f}, total_tokens = {total_tokens}")
    
    # Clamp avg_loss pour éviter overflow dans exp() 
    # exp(20) ≈ 485M, exp(30) ≈ 10^13, largement suffisant
    avg_loss = min(avg_loss, 20.0)
    
    perplexity = math.exp(avg_loss)
    
    # Restore original configuration and dtype
    if original_attn_implementation is not None:
        model.config._attn_implementation = original_attn_implementation
        if accelerator:
            accelerator.print("Debug: Restoring FlashAttention implementation")
    
    if original_dtype is not None:
        model = model.to(original_dtype)
        if accelerator:
            accelerator.print("Debug: Restoring model to float16")
    
    model.train()
    return perplexity


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    accelerator: Accelerator,
    config: PretrainConfig,
    epoch: int,
    global_step: int,
    use_flash_attention: bool = False,
    mixed_precision: str = "no"
) -> tuple[int, float]:
    """Entraîne le modèle pour une époque."""
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Époque {epoch}")
    
    for step, batch in enumerate(progress_bar):
        with accelerator.accumulate(model):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            
            # Let Accelerate handle mixed precision automatically
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                # Use accelerator's gradient clipping (handles mixed precision correctly)
                accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        if accelerator.sync_gradients:
            global_step += 1
            total_loss += loss.item()
            
            # Logging
            if global_step % config.logging_steps == 0:
                avg_loss = total_loss / config.logging_steps
                current_lr = scheduler.get_last_lr()[0]
                accelerator.print(f"Step {global_step}: loss={avg_loss:.4f}, lr={current_lr:.2e}")
                total_loss = 0.0
            
            # Sauvegarde des checkpoints
            if global_step % config.save_steps == 0:
                if accelerator.is_main_process:
                    checkpoint_dir = f"./checkpoints/pretrain/{config.model_name}/step_{global_step}"
                    save_checkpoint(model, optimizer, scheduler, global_step, checkpoint_dir, accelerator, scaler=None)
                    accelerator.print(f"Checkpoint sauvegardé à l'étape {global_step}")
        
        progress_bar.set_postfix({"loss": loss.item(), "lr": scheduler.get_last_lr()[0]})
        
        # Arrêt anticipé si max_steps atteint
        if config.max_steps > 0 and global_step >= config.max_steps:
            break
    
    return global_step, total_loss


def main():
    parser = argparse.ArgumentParser(description="Pré-entraînement d'un mini-LLM")
    parser.add_argument("--config", type=str, required=True,
                       help="Chemin vers le fichier de configuration du modèle")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Chemin vers les données tokenisées")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/pretrain",
                       help="Dossier de sortie pour les checkpoints")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Chemin vers le checkpoint à reprendre")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                       help="Taux d'apprentissage")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Taille de batch")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                       help="Étapes d'accumulation de gradient")
    parser.add_argument("--num_train_epochs", type=int, default=1,
                       help="Nombre d'époques d'entraînement")
    parser.add_argument("--max_steps", type=int, default=-1,
                       help="Nombre maximum d'étapes (-1 pour unlimited)")
    parser.add_argument("--warmup_steps", type=int, default=2000,
                       help="Nombre d'étapes de warmup")
    parser.add_argument("--save_steps", type=int, default=5000,
                       help="Intervalle de sauvegarde")
    parser.add_argument("--logging_steps", type=int, default=100,
                       help="Intervalle de logging")
    parser.add_argument("--use_flash_attn", action="store_true", default=True,
                       help="Utiliser FlashAttention-2 (défaut: True)")
    parser.add_argument("--no_flash_attn", action="store_true",
                       help="Désactiver FlashAttention-2 (force fallback SDPA)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Seed pour la reproducibilité")
    parser.add_argument("--no_deterministic", action="store_true",
                       help="Désactiver l'entraînement déterministe (plus rapide)")
    
    args = parser.parse_args()
    
    # Gestion FlashAttention-2 - utiliser mixed_precision d'Accelerate
    use_flash_attention = args.use_flash_attn and not args.no_flash_attn
    mixed_precision = "fp16" if use_flash_attention else "no"
    
    # Initialisation d'accelerate  
    from accelerate import DistributedDataParallelKwargs
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with="wandb" if os.getenv("WANDB_API_KEY") else None,
        kwargs_handlers=[ddp_kwargs]
    )
    
    # Configuration
    config = PretrainConfig(args.config)
    config.learning_rate = args.learning_rate
    config.batch_size = args.batch_size
    config.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.num_train_epochs = args.num_train_epochs
    config.max_steps = args.max_steps
    config.warmup_steps = args.warmup_steps
    config.save_steps = args.save_steps
    config.logging_steps = args.logging_steps
    
    # Configuration déterministe
    dataloader_generator = None
    if not args.no_deterministic:
        dataloader_generator = set_deterministic_training(args.seed)
    
    # FlashAttention-2 déjà configuré plus haut
    
    # Création du modèle
    accelerator.print("Création du modèle...")
    model = create_model(config, use_flash_attention=use_flash_attention)
    accelerator.print(f"Modèle créé: {sum(p.numel() for p in model.parameters()):,} paramètres")
    
    # Chargement des données
    accelerator.print("Chargement des données...")
    train_dataset = TokenizedDataset(args.data_path, config.sequence_length)
    
    # Division train/validation (90/10)
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        generator=dataloader_generator
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        generator=dataloader_generator
    )
    
    # Optimiseur et scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay
    )
    
    # Calcul du nombre total d'étapes
    if config.max_steps > 0:
        max_train_steps = config.max_steps
    else:
        max_train_steps = len(train_dataloader) * config.num_train_epochs
        max_train_steps = max_train_steps // config.gradient_accumulation_steps
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=max_train_steps
    )
    
    # Préparation avec accelerate
    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler
    )
    
    # Reprise depuis un checkpoint si spécifié
    global_step = 0
    start_epoch = 0
    
    if args.resume_from_checkpoint:
        accelerator.print(f"Reprise depuis {args.resume_from_checkpoint}")
        global_step = load_checkpoint(model, optimizer, scheduler, args.resume_from_checkpoint, accelerator, scaler=None)
        start_epoch = global_step // len(train_dataloader)
    
    # Boucle d'entraînement
    accelerator.print("Début de l'entraînement...")
    
    for epoch in range(start_epoch, config.num_train_epochs):
        accelerator.print(f"Époque {epoch + 1}/{config.num_train_epochs}")
        
        global_step, _ = train_epoch(
            model, train_dataloader, optimizer, scheduler, 
            accelerator, config, epoch + 1, global_step, use_flash_attention, mixed_precision
        )
        
        # Évaluation
        if accelerator.is_main_process and (epoch + 1) % 1 == 0:  # Évaluation chaque époque
            accelerator.print("Évaluation...")
            perplexity = calculate_perplexity(model, val_dataloader, accelerator.device, accelerator)
            accelerator.print(f"Perplexité de validation: {perplexity:.2f}")
        
        # Arrêt si max_steps atteint
        if config.max_steps > 0 and global_step >= config.max_steps:
            break
    
    # Sauvegarde finale
    if accelerator.is_main_process:
        final_dir = f"{args.output_dir}/{config.model_name}/final"
        save_checkpoint(model, optimizer, scheduler, global_step, final_dir, accelerator, scaler=None)
        accelerator.print(f"Modèle final sauvegardé dans {final_dir}")
    
    accelerator.print("Entraînement terminé!")


if __name__ == "__main__":
    main()