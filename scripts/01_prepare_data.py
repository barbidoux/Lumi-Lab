#!/usr/bin/env python3
"""
Script de préparation des données pour l'entraînement d'un mini-LLM.
Implémente le pipeline de nettoyage, déduplication, filtrage et tokenization.
"""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Set

import sentencepiece as spm
from datasets import Dataset, load_dataset
from tqdm import tqdm


def clean_text(text: str) -> str:
    """Nettoie le texte en supprimant les URLs, code et caractères indésirables."""
    # Suppression des URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Suppression des blocs de code
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`[^`]*`', '', text)
    
    # Suppression des caractères de contrôle
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Normalisation des espaces
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def is_english_text(text: str, threshold: float = 0.7) -> bool:
    """Vérifie si le texte est probablement en anglais."""
    # Liste de mots anglais fréquents
    english_words = {
        'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 
        'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 
        'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my',
        'one', 'all', 'would', 'there', 'their', 'what', 'up', 'out', 'if', 'about',
        'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time',
        'no', 'just', 'him', 'know', 'take', 'people', 'into', 'year', 'your',
        'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then', 'now',
        'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back', 'after',
        'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new'
    }
    
    words = text.lower().split()
    if not words:
        return False
    
    english_count = sum(1 for word in words if word in english_words)
    return (english_count / len(words)) >= threshold


def filter_by_length(text: str, min_length: int = 50, max_length: int = 10000) -> bool:
    """Filtre par longueur de texte."""
    return min_length <= len(text) <= max_length


def compute_hash(text: str) -> str:
    """Calcule le hash MD5 d'un texte pour la déduplication."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def deduplicate_texts(texts: List[str]) -> List[str]:
    """Supprime les doublons basés sur le hash MD5."""
    seen_hashes: Set[str] = set()
    deduplicated = []
    
    for text in tqdm(texts, desc="Déduplication"):
        text_hash = compute_hash(text)
        if text_hash not in seen_hashes:
            seen_hashes.add(text_hash)
            deduplicated.append(text)
    
    return deduplicated


def train_tokenizer(texts: List[str], vocab_size: int, output_path: str) -> None:
    """Entraîne un tokenizer SentencePiece sur le corpus."""
    # Écriture temporaire du corpus
    temp_corpus = output_path + "_temp_corpus.txt"
    with open(temp_corpus, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + "\n")
    
    # Entraînement du tokenizer
    spm.SentencePieceTrainer.train(
        input=temp_corpus,
        model_prefix=output_path,
        vocab_size=vocab_size,
        model_type='unigram',
        character_coverage=0.995,
        normalization_rule_name='nmt_nfkc_cf'
    )
    
    # Suppression du fichier temporaire
    os.remove(temp_corpus)


def tokenize_corpus(texts: List[str], tokenizer_path: str) -> List[List[int]]:
    """Tokenise le corpus avec le tokenizer SentencePiece."""
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path + ".model")
    
    tokenized = []
    for text in tqdm(texts, desc="Tokenisation"):
        tokens = sp.encode_as_ids(text)
        tokenized.append(tokens)
    
    return tokenized


def save_tokenized_data(tokenized_data: List[List[int]], output_path: str) -> None:
    """Sauvegarde les données tokenisées au format JSON."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tokenized_data, f)


def main():
    parser = argparse.ArgumentParser(description="Préparation des données pour l'entraînement")
    parser.add_argument("--input_path", type=str, required=True,
                       help="Chemin vers le dataset d'entrée")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Dossier de sortie pour les données préparées")
    parser.add_argument("--vocab_size", type=int, default=32768,
                       help="Taille du vocabulaire pour le tokenizer")
    parser.add_argument("--min_length", type=int, default=50,
                       help="Longueur minimale des textes")
    parser.add_argument("--max_length", type=int, default=10000,
                       help="Longueur maximale des textes")
    
    args = parser.parse_args()
    
    # Création du dossier de sortie
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Chargement du dataset...")
    if args.input_path.endswith(('.json', '.jsonl')):
        dataset = load_dataset('json', data_files=args.input_path)['train']
    else:
        dataset = load_dataset(args.input_path)['train']
    
    # Extraction du texte (adapté selon la structure du dataset)
    if 'text' in dataset.column_names:
        texts = dataset['text']
    elif 'content' in dataset.column_names:
        texts = dataset['content']
    else:
        raise ValueError("Aucune colonne 'text' ou 'content' trouvée dans le dataset")
    
    print(f"Dataset original: {len(texts)} textes")
    
    # Pipeline de nettoyage
    print("Nettoyage des textes...")
    cleaned_texts = []
    for text in tqdm(texts, desc="Nettoyage"):
        cleaned = clean_text(str(text))
        if cleaned and filter_by_length(cleaned, args.min_length, args.max_length):
            if is_english_text(cleaned):
                cleaned_texts.append(cleaned)
    
    print(f"Après nettoyage: {len(cleaned_texts)} textes")
    
    # Déduplication
    print("Déduplication...")
    deduplicated_texts = deduplicate_texts(cleaned_texts)
    print(f"Après déduplication: {len(deduplicated_texts)} textes")
    
    # Entraînement du tokenizer
    tokenizer_path = str(output_dir / "tokenizer")
    print(f"Entraînement du tokenizer (vocab_size={args.vocab_size})...")
    train_tokenizer(deduplicated_texts, args.vocab_size, tokenizer_path)
    
    # Tokenisation
    print("Tokenisation du corpus...")
    tokenized_data = tokenize_corpus(deduplicated_texts, tokenizer_path)
    
    # Sauvegarde
    output_path = output_dir / "tokenized_data.json"
    print(f"Sauvegarde vers {output_path}...")
    save_tokenized_data(tokenized_data, str(output_path))
    
    # Statistiques finales
    total_tokens = sum(len(tokens) for tokens in tokenized_data)
    avg_tokens = total_tokens / len(tokenized_data) if tokenized_data else 0
    
    stats = {
        "num_texts": len(tokenized_data),
        "total_tokens": total_tokens,
        "avg_tokens_per_text": avg_tokens,
        "vocab_size": args.vocab_size
    }
    
    with open(output_dir / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Préparation terminée!")
    print(f"- Nombre de textes: {stats['num_texts']}")
    print(f"- Total de tokens: {stats['total_tokens']:,}")
    print(f"- Moyenne tokens/texte: {stats['avg_tokens_per_text']:.1f}")


if __name__ == "__main__":
    main()