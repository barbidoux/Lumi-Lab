#!/usr/bin/env python3
"""
Script d'évaluation du modèle final.
Calcule la perplexité, effectue des benchmarks zero-shot et des smoke-tests.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from evaluate import load
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.dataset_utils import TokenizedDataset, create_dataloader
from utils.model_utils import load_pretrained_model


def calculate_perplexity(model, tokenizer, dataset_name: str = "wikitext-2-raw-v1") -> float:
    """Calcule la perplexité sur un dataset de référence comme Wikitext-2."""
    print(f"Calcul de la perplexité sur {dataset_name}...")
    
    # Chargement du dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")["test"]
    
    # Préparation des données
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=1024,
            padding=False,
            return_overflowing_tokens=False
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenisation pour perplexité"
    )
    
    # Filtrage des séquences vides
    tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) > 10)
    
    # Création du dataloader
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: {
            'input_ids': torch.tensor(x[0]['input_ids']).unsqueeze(0),
            'attention_mask': torch.ones(1, len(x[0]['input_ids']))
        }
    )
    
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calcul perplexité"):
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            
            # Labels sont les mêmes que input_ids pour la perplexité
            labels = input_ids.clone()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # Compter les tokens valides (non-padding)
            valid_tokens = attention_mask.sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity


def evaluate_boolq(model, tokenizer, max_samples: int = 100) -> Dict[str, float]:
    """Évalue le modèle sur BoolQ (Yes/No questions)."""
    print(f"Évaluation BoolQ (max {max_samples} échantillons)...")
    
    # Chargement du dataset BoolQ
    dataset = load_dataset("boolq")["validation"]
    if len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))
    
    correct = 0
    total = 0
    
    model.eval()
    
    with torch.no_grad():
        for example in tqdm(dataset, desc="Évaluation BoolQ"):
            question = example["question"]
            passage = example["passage"]
            answer = example["answer"]  # True/False
            
            # Format du prompt
            prompt = f"Passage: {passage[:500]}...\nQuestion: {question}\nAnswer: "
            
            # Tokenisation
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=800)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Génération des probabilités pour "Yes" et "No"
            yes_token = tokenizer.encode("Yes", add_special_tokens=False)[0]
            no_token = tokenizer.encode("No", add_special_tokens=False)[0]
            
            # Forward pass
            outputs = model(**inputs)
            logits = outputs.logits[0, -1]  # Logits pour le dernier token
            
            # Probabilités pour Yes/No
            yes_prob = torch.softmax(logits, dim=0)[yes_token].item()
            no_prob = torch.softmax(logits, dim=0)[no_token].item()
            
            # Prédiction
            predicted_answer = yes_prob > no_prob
            
            if predicted_answer == answer:
                correct += 1
            total += 1
    
    accuracy = correct / total
    return {"boolq_accuracy": accuracy, "boolq_correct": correct, "boolq_total": total}


def run_smoke_tests(model, tokenizer, test_prompts: List[str]) -> List[Dict[str, str]]:
    """Effectue des smoke-tests avec une liste de prompts."""
    print("Exécution des smoke-tests...")
    
    results = []
    model.eval()
    
    generation_config = {
        "max_new_tokens": 100,
        "temperature": 0.7,
        "do_sample": True,
        "top_p": 0.9,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.1
    }
    
    with torch.no_grad():
        for i, prompt in enumerate(test_prompts):
            print(f"\nTest {i+1}/{len(test_prompts)}")
            print(f"Prompt: {prompt}")
            
            # Tokenisation
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Génération
            outputs = model.generate(inputs.input_ids, **generation_config)
            
            # Décodage
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = response[len(prompt):].strip()
            
            result = {
                "prompt": prompt,
                "response": generated_text,
                "full_text": response
            }
            results.append(result)
            
            print(f"Réponse: {generated_text}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Évaluation complète du modèle")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Chemin vers le modèle à évaluer")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                       help="Chemin vers le tokenizer")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                       help="Dossier pour sauvegarder les résultats")
    parser.add_argument("--skip_perplexity", action="store_true",
                       help="Ignorer le calcul de perplexité")
    parser.add_argument("--skip_boolq", action="store_true",
                       help="Ignorer l'évaluation BoolQ")
    parser.add_argument("--max_boolq_samples", type=int, default=100,
                       help="Nombre max d'échantillons BoolQ")
    parser.add_argument("--custom_prompts", type=str, default=None,
                       help="Fichier JSON avec des prompts personnalisés")
    
    args = parser.parse_args()
    
    # Création du dossier de sortie
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Chargement du modèle...")
    model = load_pretrained_model(args.model_path)
    
    # Chargement du tokenizer
    if args.tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Modèle: {sum(p.numel() for p in model.parameters()):,} paramètres")
    print(f"Device: {model.device}")
    
    # Structure pour stocker tous les résultats
    evaluation_results = {
        "model_path": args.model_path,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "device": str(model.device)
    }
    
    # 1. Calcul de la perplexité
    if not args.skip_perplexity:
        try:
            perplexity = calculate_perplexity(model, tokenizer)
            evaluation_results["perplexity"] = perplexity
            print(f"Perplexité (Wikitext-2): {perplexity:.2f}")
        except Exception as e:
            print(f"Erreur lors du calcul de perplexité: {e}")
            evaluation_results["perplexity_error"] = str(e)
    
    # 2. Benchmark BoolQ (adapté selon le mode)
    if not args.skip_boolq:
        boolq_samples = min(args.max_boolq_samples, 20 if args.fast_mode else args.max_boolq_samples)
        try:
            boolq_results = evaluate_boolq(model, tokenizer, boolq_samples)
            evaluation_results.update(boolq_results)
            print(f"BoolQ Accuracy: {boolq_results['boolq_accuracy']:.3f} ({boolq_results['boolq_correct']}/{boolq_results['boolq_total']})")
        except Exception as e:
            print(f"Erreur lors de l'évaluation BoolQ: {e}")
            evaluation_results["boolq_error"] = str(e)
    
    # 3. Smoke-tests
    # Prompts par défaut
    default_prompts = [
        "Qu'est-ce que l'intelligence artificielle ?",
        "Expliquez le concept de machine learning en termes simples.",
        "Comment fonctionne un réseau de neurones ?",
        "Quels sont les avantages et inconvénients de l'IA ?",
        "Décrivez l'impact de l'IA sur la société moderne.",
        "What is the capital of France?",
        "Write a short story about a robot.",
        "Explain quantum computing to a 10-year-old.",
        "List three benefits of renewable energy.",
        "How do you make a good cup of coffee?"
    ]
    
    # Chargement des prompts personnalisés si fournis
    if args.custom_prompts:
        with open(args.custom_prompts, 'r', encoding='utf-8') as f:
            custom_data = json.load(f)
            if isinstance(custom_data, list):
                test_prompts = custom_data
            else:
                test_prompts = custom_data.get("prompts", default_prompts)
    else:
        test_prompts = default_prompts
    
    # Exécution des smoke-tests
    try:
        smoke_results = run_smoke_tests(model, tokenizer, test_prompts)
        evaluation_results["smoke_tests"] = smoke_results
        print(f"\nSmoke-tests terminés: {len(smoke_results)} prompts testés")
    except Exception as e:
        print(f"Erreur lors des smoke-tests: {e}")
        evaluation_results["smoke_tests_error"] = str(e)
    
    # Sauvegarde des résultats
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    
    # Sauvegarde détaillée si demandée
    if args.detailed_output and "smoke_tests" in evaluation_results:
        detailed_file = output_dir / "detailed_smoke_tests.md"
        with open(detailed_file, 'w', encoding='utf-8') as f:
            f.write("# Smoke Tests - Résultats Détaillés\n\n")
            
            for i, result in enumerate(evaluation_results["smoke_tests"]["individual_results"], 1):
                f.write(f"## Test {i}\n\n")
                f.write(f"**Prompt:** {result['prompt']}\n\n")
                f.write(f"**Réponse:** {result['response']}\n\n")
                f.write(f"**Qualité:** {result['quality_scores']['overall_quality']:.3f}/1.0\n")
                f.write(f"**Longueur:** {result['word_count']} mots\n\n")
                f.write("---\n\n")
        
        print(f"Résultats détaillés dans: {detailed_file}")
    
    # Rapport de synthèse
    print(f"\n{'='*50}")
    print("RAPPORT D'ÉVALUATION")
    print(f"{'='*50}")
    print(f"Modèle: {args.model_path}")
    print(f"Paramètres: {evaluation_results['model_parameters']:,}")
    
    # Gestion de la compatibilité avec l'ancien format
    if "perplexity_metrics" in evaluation_results:
        ppl_metrics = evaluation_results["perplexity_metrics"]
        print(f"Perplexité (WikiText-2): {ppl_metrics['perplexity']:.2f}")
        print(f"  └─ {ppl_metrics['total_tokens']:,} tokens @ {ppl_metrics['tokens_per_second']:.0f} tok/sec")
    elif "perplexity" in evaluation_results:
        print(f"Perplexité (Wikitext-2): {evaluation_results['perplexity']:.2f}")
    
    if "boolq_accuracy" in evaluation_results:
        print(f"BoolQ Accuracy: {evaluation_results['boolq_accuracy']:.3f}")
    
    if "smoke_tests" in evaluation_results:
        if "summary_stats" in evaluation_results["smoke_tests"]:
            smoke_summary = evaluation_results["smoke_tests"]["summary_stats"]
            print(f"Smoke-tests: {smoke_summary['total_prompts']} prompts")
            print(f"  └─ Qualité moyenne: {smoke_summary['avg_quality_score']:.3f}/1.0")
        else:
            # Compatibilité avec l'ancien format
            print(f"Smoke-tests: {len(evaluation_results['smoke_tests'])} prompts testés")
    
    print(f"\nRésultats sauvegardés dans: {results_file}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()