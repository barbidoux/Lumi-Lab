"""
Utilitaires pour la gestion des datasets et du chargement des données.
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Optional, Union
from torch.utils.data import Dataset, DataLoader
import numpy as np


class TokenizedDataset(Dataset):
    """Dataset pour les données tokenisées sauvegardées au format JSON."""
    
    def __init__(self, data_path: str, sequence_length: int = 1024, stride: int = None):
        """
        Args:
            data_path: Chemin vers le fichier JSON contenant les données tokenisées
            sequence_length: Longueur de séquence pour l'entraînement
            stride: Décalage entre les séquences (par défaut = sequence_length)
        """
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.stride = stride or sequence_length
        
        # Chargement des données
        print(f"Chargement des données depuis {data_path}...")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.tokenized_texts = json.load(f)
        
        # Concaténation de tous les tokens
        self.all_tokens = []
        for tokens in self.tokenized_texts:
            self.all_tokens.extend(tokens)
        
        self.all_tokens = torch.tensor(self.all_tokens, dtype=torch.long)
        
        # Calcul du nombre de séquences
        self.num_sequences = max(0, (len(self.all_tokens) - sequence_length) // self.stride + 1)
        
        print(f"Dataset chargé: {len(self.tokenized_texts)} textes, "
              f"{len(self.all_tokens):,} tokens, "
              f"{self.num_sequences:,} séquences de longueur {sequence_length}")
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.sequence_length
        
        # Extraction de la séquence
        sequence = self.all_tokens[start_idx:end_idx]
        
        # Pour l'entraînement, input_ids et labels sont décalés d'un token
        input_ids = sequence[:-1]
        labels = sequence[1:]
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': torch.ones_like(input_ids)
        }


class ConversationalDataset(Dataset):
    """Dataset pour les conversations (SFT, DPO)."""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048, format_template: str = "chat"):
        """
        Args:
            data_path: Chemin vers le dataset
            tokenizer: Tokenizer à utiliser
            max_length: Longueur maximale des séquences
            format_template: Template de formatage ("chat", "instruct", etc.)
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.format_template = format_template
        
        # Chargement des données
        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        else:
            raise ValueError(f"Format de fichier non supporté: {data_path}")
        
        print(f"Dataset conversationnel chargé: {len(self.data)} exemples")
    
    def __len__(self):
        return len(self.data)
    
    def format_example(self, example: Dict) -> str:
        """Formate un exemple selon le template choisi."""
        if self.format_template == "chat":
            return f"Human: {example['prompt']}\n\nAssistant: {example['response']}<|endoftext|>"
        elif self.format_template == "instruct":
            return f"### Instruction:\n{example['prompt']}\n\n### Response:\n{example['response']}<|endoftext|>"
        else:
            # Format personnalisé
            return f"{example['prompt']}\n{example['response']}<|endoftext|>"
    
    def __getitem__(self, idx):
        example = self.data[idx]
        formatted_text = self.format_example(example)
        
        # Tokenisation
        encoded = self.tokenizer(
            formatted_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'labels': encoded['input_ids'].squeeze().clone()
        }


def create_dataloader(dataset: Dataset, batch_size: int = 8, shuffle: bool = True, 
                     num_workers: int = 4, pin_memory: bool = True) -> DataLoader:
    """Crée un DataLoader avec les paramètres optimisés."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Important pour l'entraînement stable
    )


def split_dataset(dataset: Dataset, train_ratio: float = 0.9, seed: int = 42):
    """Divise un dataset en train/validation."""
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size
    
    # Split déterministe
    torch.manual_seed(seed)
    return torch.utils.data.random_split(dataset, [train_size, val_size])


def collate_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Fonction de collation personnalisée pour les batchs."""
    keys = batch[0].keys()
    collated = {}
    
    for key in keys:
        if key in ['input_ids', 'labels', 'attention_mask']:
            # Padding pour les séquences de longueurs différentes
            sequences = [item[key] for item in batch]
            padded = torch.nn.utils.rnn.pad_sequence(
                sequences, 
                batch_first=True, 
                padding_value=0 if key != 'labels' else -100
            )
            collated[key] = padded
        else:
            collated[key] = torch.stack([item[key] for item in batch])
    
    return collated


def load_and_prepare_dataset(
    data_path: str, 
    dataset_type: str = "pretrain",
    sequence_length: int = 1024,
    batch_size: int = 8,
    tokenizer = None,
    train_ratio: float = 0.9
) -> Dict[str, DataLoader]:
    """
    Fonction de haut niveau pour charger et préparer un dataset.
    
    Args:
        data_path: Chemin vers les données
        dataset_type: Type de dataset ("pretrain", "sft", "dpo")
        sequence_length: Longueur de séquence
        batch_size: Taille de batch
        tokenizer: Tokenizer (requis pour sft/dpo)
        train_ratio: Ratio train/validation
    
    Returns:
        Dict contenant les DataLoaders train et validation
    """
    if dataset_type == "pretrain":
        dataset = TokenizedDataset(data_path, sequence_length)
        train_dataset, val_dataset = split_dataset(dataset, train_ratio)
        
        train_loader = create_dataloader(train_dataset, batch_size, shuffle=True)
        val_loader = create_dataloader(val_dataset, batch_size, shuffle=False)
        
    elif dataset_type in ["sft", "dpo"]:
        if tokenizer is None:
            raise ValueError("Tokenizer requis pour les datasets SFT/DPO")
        
        dataset = ConversationalDataset(data_path, tokenizer, sequence_length)
        train_dataset, val_dataset = split_dataset(dataset, train_ratio)
        
        train_loader = create_dataloader(train_dataset, batch_size, shuffle=True)
        val_loader = create_dataloader(val_dataset, batch_size, shuffle=False)
    
    else:
        raise ValueError(f"Type de dataset non supporté: {dataset_type}")
    
    return {
        "train": train_loader,
        "validation": val_loader,
        "train_size": len(train_dataset) if 'train_dataset' in locals() else 0,
        "val_size": len(val_dataset) if 'val_dataset' in locals() else 0
    }


def get_dataset_stats(data_path: str) -> Dict:
    """Calcule des statistiques sur un dataset tokenisé."""
    with open(data_path, 'r', encoding='utf-8') as f:
        tokenized_texts = json.load(f)
    
    # Calcul des statistiques
    lengths = [len(tokens) for tokens in tokenized_texts]
    total_tokens = sum(lengths)
    
    stats = {
        "num_texts": len(tokenized_texts),
        "total_tokens": total_tokens,
        "avg_tokens_per_text": np.mean(lengths),
        "min_tokens": min(lengths),
        "max_tokens": max(lengths),
        "median_tokens": np.median(lengths),
        "std_tokens": np.std(lengths)
    }
    
    return stats


def create_vocabulary_stats(tokenizer, data_path: str) -> Dict:
    """Analyse l'utilisation du vocabulaire dans un dataset."""
    with open(data_path, 'r', encoding='utf-8') as f:
        tokenized_texts = json.load(f)
    
    # Comptage des tokens
    token_counts = {}
    total_tokens = 0
    
    for tokens in tokenized_texts:
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
            total_tokens += 1
    
    # Statistiques de vocabulaire
    unique_tokens = len(token_counts)
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else len(tokenizer)
    coverage = unique_tokens / vocab_size
    
    # Tokens les plus fréquents
    most_common = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    
    stats = {
        "vocab_size": vocab_size,
        "unique_tokens_used": unique_tokens,
        "vocabulary_coverage": coverage,
        "total_token_occurrences": total_tokens,
        "most_common_tokens": [
            {"token_id": token_id, "count": count, "token": tokenizer.decode([token_id])}
            for token_id, count in most_common
        ]
    }
    
    return stats