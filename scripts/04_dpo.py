#!/usr/bin/env python3
"""
Script d'alignement via Direct Preference Optimization (DPO).
Utilise trl DPOTrainer pour aligner le modèle selon les préférences humaines.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import Dataset, load_dataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import DPOTrainer

from utils.model_utils import load_pretrained_model


def format_dpo_example(example: Dict) -> Dict:
    """Formate un exemple DPO avec prompt, chosen et rejected."""
    return {
        "prompt": example["prompt"],
        "chosen": example["chosen"],
        "rejected": example["rejected"]
    }


def create_dpo_dataset(dataset_path: str) -> Dataset:
    """Crée un dataset formaté pour DPO."""
    
    # Chargement du dataset
    if dataset_path.endswith(('.json', '.jsonl')):
        dataset = load_dataset('json', data_files=dataset_path)['train']
    else:
        dataset = load_dataset(dataset_path)['train']
    
    # Vérification des colonnes requises
    required_columns = ["prompt", "chosen", "rejected"]
    for col in required_columns:
        if col not in dataset.column_names:
            raise ValueError(f"Colonne manquante: {col}. Dataset doit contenir: {required_columns}")
    
    # Formatage des exemples
    def format_examples(examples):
        formatted_examples = {"prompt": [], "chosen": [], "rejected": []}
        
        for i in range(len(examples['prompt'])):
            example = {k: v[i] for k, v in examples.items()}
            formatted = format_dpo_example(example)
            
            formatted_examples["prompt"].append(formatted["prompt"])
            formatted_examples["chosen"].append(formatted["chosen"])
            formatted_examples["rejected"].append(formatted["rejected"])
        
        return formatted_examples
    
    # Application du formatage
    formatted_dataset = dataset.map(
        format_examples,
        batched=True,
        desc="Formatage des exemples DPO"
    )
    
    return formatted_dataset


class EarlyStoppingCallback:
    """Callback pour l'arrêt anticipé basé sur la loss de validation."""
    
    def __init__(self, patience: int = 3, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.should_stop = False
    
    def __call__(self, logs: Dict):
        if "eval_loss" in logs:
            current_loss = logs["eval_loss"]
            
            if current_loss < self.best_loss - self.min_delta:
                self.best_loss = current_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.patience:
                self.should_stop = True
                print(f"Early stopping déclenché après {self.patience} évaluations sans amélioration")


def main():
    parser = argparse.ArgumentParser(description="Direct Preference Optimization (DPO)")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Chemin vers le modèle SFT")
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Chemin vers le dataset DPO (avec pairs chosen/rejected)")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/dpo",
                       help="Dossier de sortie")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                       help="Chemin vers le tokenizer")
    parser.add_argument("--beta", type=float, default=0.1,
                       help="Paramètre beta pour DPO")
    parser.add_argument("--learning_rate", type=float, default=5e-7,
                       help="Taux d'apprentissage pour DPO")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Taille de batch")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                       help="Étapes d'accumulation de gradient")
    parser.add_argument("--num_train_epochs", type=int, default=1,
                       help="Nombre d'époques")
    parser.add_argument("--max_steps", type=int, default=-1,
                       help="Nombre maximum d'étapes")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                       help="Ratio de warmup")
    parser.add_argument("--max_length", type=int, default=1024,
                       help="Longueur maximale des séquences")
    parser.add_argument("--max_prompt_length", type=int, default=512,
                       help="Longueur maximale des prompts")
    
    args = parser.parse_args()
    
    print("Chargement du modèle SFT...")
    
    # Chargement du modèle SFT (avec adapters LoRA si applicable)
    if os.path.exists(os.path.join(args.model_path, "adapter_config.json")):
        # Modèle avec adapters LoRA
        base_model = load_pretrained_model(args.model_path.replace("/lora_adapters", ""))
        model = PeftModel.from_pretrained(base_model, args.model_path)
        model = model.merge_and_unload()  # Fusion des adapters pour DPO
    else:
        # Modèle standard
        model = load_pretrained_model(args.model_path)
    
    # Chargement du tokenizer
    if args.tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Configuration du tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Chargement du dataset DPO
    print("Chargement du dataset DPO...")
    train_dataset = create_dpo_dataset(args.dataset_path)
    
    print(f"Dataset DPO: {len(train_dataset)} exemples")
    print("Exemple DPO:")
    print(f"Prompt: {train_dataset[0]['prompt'][:100]}...")
    print(f"Chosen: {train_dataset[0]['chosen'][:100]}...")
    print(f"Rejected: {train_dataset[0]['rejected'][:100]}...")
    
    # Division train/validation si dataset assez grand
    if len(train_dataset) > 100:
        train_size = int(0.9 * len(train_dataset))
        indices = list(range(len(train_dataset)))
        train_indices = indices[:train_size]
        eval_indices = indices[train_size:]
        
        eval_dataset = train_dataset.select(eval_indices)
        train_dataset = train_dataset.select(train_indices)
    else:
        eval_dataset = None
    
    # Configuration de l'entraînement
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=100,
        eval_steps=50 if eval_dataset else -1,
        evaluation_strategy="steps" if eval_dataset else "no",
        save_strategy="steps",
        max_grad_norm=1.0,
        dataloader_num_workers=4,
        fp16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="wandb" if os.getenv("WANDB_API_KEY") else "none",
        run_name="dpo_training",
        save_total_limit=2,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
    )
    
    # Callback pour early stopping
    early_stopping = EarlyStoppingCallback(patience=3, min_delta=0.001)
    
    # Initialisation du trainer DPO
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # DPOTrainer créera automatiquement une copie de référence
        args=training_args,
        beta=args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
    )
    
    print(f"Début de l'entraînement DPO avec beta={args.beta}...")
    print(f"Modèle de référence créé automatiquement")
    
    # Entraînement avec monitoring des métriques
    trainer.train()
    
    # Sauvegarde du modèle final
    print("Sauvegarde du modèle DPO...")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    print(f"Modèle DPO sauvegardé dans {training_args.output_dir}")
    
    # Test rapide de comparaison
    print("\nTest de génération comparative:")
    test_prompt = "Comment puis-je améliorer ma productivité au travail ?"
    
    # Tokenisation
    inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True, max_length=256)
    
    # Génération
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {test_prompt}")
    print(f"Réponse DPO: {response[len(test_prompt):]}")
    
    # Affichage des métriques finales
    if hasattr(trainer.state, 'log_history'):
        final_logs = trainer.state.log_history[-1]
        print(f"\nMétriques finales:")
        for key, value in final_logs.items():
            if key not in ['epoch', 'step']:
                print(f"  {key}: {value:.4f}")
    
    print("Entraînement DPO terminé!")


if __name__ == "__main__":
    main()