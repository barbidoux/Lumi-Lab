#!/usr/bin/env python3
"""
Script de Supervised Fine-Tuning (SFT) avec LoRA.
Utilise trl et PEFT pour le fine-tuning efficace du modèle pré-entraîné.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

from utils.model_utils import load_pretrained_model


def format_prompt_response(example: Dict, prompt_key: str = "prompt", response_key: str = "response") -> str:
    """Formate un exemple en prompt-response pour l'entraînement SFT."""
    prompt = example[prompt_key]
    response = example[response_key]
    
    # Format standard avec séparateurs clairs
    formatted_text = f"### Instruction:\n{prompt}\n\n### Response:\n{response}<|endoftext|>"
    return formatted_text


def create_sft_dataset(dataset_path: str, tokenizer, max_seq_length: int = 2048) -> Dataset:
    """Crée un dataset formaté pour le SFT."""
    
    # Chargement du dataset
    if dataset_path.endswith(('.json', '.jsonl')):
        dataset = load_dataset('json', data_files=dataset_path)['train']
    else:
        dataset = load_dataset(dataset_path)['train']
    
    # Formatage des exemples
    def format_examples(examples):
        formatted_texts = []
        for i in range(len(examples['prompt'])):
            example = {k: v[i] for k, v in examples.items()}
            formatted_text = format_prompt_response(example)
            formatted_texts.append(formatted_text)
        return {"text": formatted_texts}
    
    # Application du formatage
    formatted_dataset = dataset.map(
        format_examples,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Formatage des exemples"
    )
    
    return formatted_dataset


def setup_lora_model(model, lora_config: Dict):
    """Configure le modèle avec LoRA."""
    
    peft_config = LoraConfig(
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        target_modules=lora_config["target_modules"],
        lora_dropout=lora_config["lora_dropout"],
        bias=lora_config["bias"],
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning avec LoRA")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Chemin vers le modèle pré-entraîné")
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Chemin vers le dataset SFT")
    parser.add_argument("--config_path", type=str, default="./config/sft.json",
                       help="Chemin vers la configuration SFT")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/sft",
                       help="Dossier de sortie")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                       help="Chemin vers le tokenizer (optionnel)")
    
    args = parser.parse_args()
    
    # Chargement de la configuration
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    
    print("Chargement du modèle et tokenizer...")
    
    # Chargement du modèle pré-entraîné
    model = load_pretrained_model(args.model_path)
    
    # Chargement du tokenizer
    if args.tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    else:
        # Utilise un tokenizer par défaut si pas spécifié
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    
    # Configuration du tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configuration LoRA
    print("Configuration LoRA...")
    model = setup_lora_model(model, config["lora_config"])
    
    # Chargement du dataset
    print("Chargement et formatage du dataset...")
    train_dataset = create_sft_dataset(args.dataset_path, tokenizer)
    
    print(f"Dataset SFT: {len(train_dataset)} exemples")
    print("Exemple formaté:")
    print(train_dataset[0]["text"][:500] + "...")
    
    # Configuration de l'entraînement
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["num_train_epochs"],
        max_steps=config["max_steps"] if config["max_steps"] > 0 else -1,
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        eval_steps=config.get("eval_steps", 500),
        max_grad_norm=config["max_grad_norm"],
        dataloader_num_workers=config["dataloader_num_workers"],
        fp16=config.get("fp16", False),
        bf16=config.get("bf16", False),
        gradient_checkpointing=config["gradient_checkpointing"],
        remove_unused_columns=False,
        report_to="wandb" if os.getenv("WANDB_API_KEY") else "none",
        run_name=config.get("run_name", "sft_training"),
        save_total_limit=3,
        load_best_model_at_end=False,
    )
    
    # Data collator pour masquer les prompts
    response_template = "### Response:"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialisation du trainer SFT
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        max_seq_length=2048,
        packing=False,  # Désactivé pour un meilleur contrôle des séquences
    )
    
    print("Début de l'entraînement SFT...")
    
    # Entraînement
    trainer.train()
    
    # Sauvegarde du modèle final
    print("Sauvegarde du modèle final...")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    # Sauvegarde des adapters LoRA séparément
    lora_output_dir = Path(training_args.output_dir) / "lora_adapters"
    model.save_pretrained(lora_output_dir)
    
    print(f"Modèle SFT sauvegardé dans {training_args.output_dir}")
    print(f"Adapters LoRA sauvegardés dans {lora_output_dir}")
    
    # Test rapide de génération
    print("\nTest de génération:")
    test_prompt = "### Instruction:\nExpliquez ce qu'est l'intelligence artificielle.\n\n### Response:\n"
    
    # Tokenisation
    inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True, max_length=512)
    
    # Génération
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {test_prompt}")
    print(f"Réponse générée: {response[len(test_prompt):]}")
    
    print("Entraînement SFT terminé!")


if __name__ == "__main__":
    main()