#!/usr/bin/env python3
"""
Script d'entraînement from scratch d'un mini-LLM.
Utilise accelerate pour la gestion de l'entraînement et des checkpoints.
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Optional

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

from utils.dataset_utils import TokenizedDataset
from utils.model_utils import create_model, save_checkpoint, load_checkpoint


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


def calculate_perplexity(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    """Calcule la perplexité sur un dataset de validation."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calcul perplexité"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # Compter seulement les tokens non-masqués
            valid_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
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
    global_step: int
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
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
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
                    save_checkpoint(model, optimizer, scheduler, global_step, checkpoint_dir, accelerator)
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
    
    args = parser.parse_args()
    
    # Initialisation d'accelerate
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="wandb" if os.getenv("WANDB_API_KEY") else None
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
    
    # Création du modèle
    accelerator.print("Création du modèle...")
    model = create_model(config)
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
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
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
        global_step = load_checkpoint(model, optimizer, scheduler, args.resume_from_checkpoint, accelerator)
        start_epoch = global_step // len(train_dataloader)
    
    # Boucle d'entraînement
    accelerator.print("Début de l'entraînement...")
    
    for epoch in range(start_epoch, config.num_train_epochs):
        accelerator.print(f"Époque {epoch + 1}/{config.num_train_epochs}")
        
        global_step, _ = train_epoch(
            model, train_dataloader, optimizer, scheduler, 
            accelerator, config, epoch + 1, global_step
        )
        
        # Évaluation
        if accelerator.is_main_process and (epoch + 1) % 1 == 0:  # Évaluation chaque époque
            accelerator.print("Évaluation...")
            perplexity = calculate_perplexity(model, val_dataloader, accelerator.device)
            accelerator.print(f"Perplexité de validation: {perplexity:.2f}")
        
        # Arrêt si max_steps atteint
        if config.max_steps > 0 and global_step >= config.max_steps:
            break
    
    # Sauvegarde finale
    if accelerator.is_main_process:
        final_dir = f"{args.output_dir}/{config.model_name}/final"
        save_checkpoint(model, optimizer, scheduler, global_step, final_dir, accelerator)
        accelerator.print(f"Modèle final sauvegardé dans {final_dir}")
    
    accelerator.print("Entraînement terminé!")


if __name__ == "__main__":
    main()