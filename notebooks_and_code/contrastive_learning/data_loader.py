"""
Data Loading and Triplet Generation for Contrastive Learning

This module provides utilities for:
1. Loading and preprocessing enzyme-metabolite interaction data
2. Generating triplets for contrastive learning
3. Creating data loaders for training and evaluation
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
from typing import List, Tuple, Dict, Optional
from sklearn.model_selection import train_test_split
import pickle
import os
from os.path import join


class EnzymeMetaboliteDataset(Dataset):
    """
    Dataset class for enzyme-metabolite interaction data with triplet sampling.
    """
    
    def __init__(self, 
                 metabolite_features: np.ndarray,
                 enzyme_features: np.ndarray, 
                 labels: np.ndarray,
                 metabolite_ids: List[str],
                 enzyme_ids: List[str],
                 mode: str = 'classification',
                 negative_sampling_ratio: int = 5):
        """
        Initialize dataset.
        
        Args:
            metabolite_features: Metabolite features [n_samples, metabolite_dim]
            enzyme_features: Enzyme features [n_samples, enzyme_dim]
            labels: Binary labels [n_samples]
            metabolite_ids: List of metabolite IDs
            enzyme_ids: List of enzyme IDs
            mode: 'classification', 'contrastive', or 'combined'
            negative_sampling_ratio: Number of negatives per positive for contrastive learning
        """
        self.metabolite_features = metabolite_features
        self.enzyme_features = enzyme_features
        self.labels = labels
        self.metabolite_ids = metabolite_ids
        self.enzyme_ids = enzyme_ids
        self.mode = mode
        self.negative_sampling_ratio = negative_sampling_ratio
        
        # Create indices for positive and negative samples
        self.positive_indices = np.where(labels == 1)[0]
        self.negative_indices = np.where(labels == 0)[0]
        
        # Create mappings for efficient triplet sampling
        self._create_interaction_mappings()
        
        if mode == 'contrastive':
            # Generate triplets for contrastive learning
            self.triplets = self._generate_triplets()
        elif mode == 'combined':
            # Generate both classification data and triplets
            self.triplets = self._generate_triplets()
    
    def _create_interaction_mappings(self):
        """Create mappings for efficient negative sampling."""
        self.metabolite_to_enzymes = {}
        self.enzyme_to_metabolites = {}
        
        # Map metabolites to their interacting enzymes
        for idx in self.positive_indices:
            met_id = self.metabolite_ids[idx]
            enz_id = self.enzyme_ids[idx]
            
            if met_id not in self.metabolite_to_enzymes:
                self.metabolite_to_enzymes[met_id] = set()
            self.metabolite_to_enzymes[met_id].add(enz_id)
            
            if enz_id not in self.enzyme_to_metabolites:
                self.enzyme_to_metabolites[enz_id] = set()
            self.enzyme_to_metabolites[enz_id].add(met_id)
        
        # Get all unique metabolite and enzyme IDs
        self.all_metabolite_ids = set(self.metabolite_ids)
        self.all_enzyme_ids = set(self.enzyme_ids)
    
    def _generate_triplets(self) -> List[Tuple[int, int, int]]:
        """
        Generate triplets for contrastive learning.
        
        Returns:
            List of (anchor_idx, positive_idx, negative_idx) tuples
        """
        triplets = []
        
        for pos_idx in self.positive_indices:
            met_id = self.metabolite_ids[pos_idx]
            enz_id = self.enzyme_ids[pos_idx]
            
            # Generate negatives for this positive pair
            for _ in range(self.negative_sampling_ratio):
                # Randomly choose to use enzyme as anchor or metabolite as anchor
                if random.random() < 0.5:
                    # Enzyme as anchor, find negative metabolite
                    negative_metabolites = (self.all_metabolite_ids - 
                                          self.enzyme_to_metabolites.get(enz_id, set()))
                    if negative_metabolites:
                        neg_met_id = random.choice(list(negative_metabolites))
                        # Find index of negative metabolite paired with same enzyme
                        neg_idx = self._find_pair_index(neg_met_id, enz_id)
                        if neg_idx is not None:
                            triplets.append((pos_idx, pos_idx, neg_idx))
                else:
                    # Metabolite as anchor, find negative enzyme  
                    negative_enzymes = (self.all_enzyme_ids - 
                                      self.metabolite_to_enzymes.get(met_id, set()))
                    if negative_enzymes:
                        neg_enz_id = random.choice(list(negative_enzymes))
                        # Find index of same metabolite paired with negative enzyme
                        neg_idx = self._find_pair_index(met_id, neg_enz_id)
                        if neg_idx is not None:
                            triplets.append((pos_idx, pos_idx, neg_idx))
        
        return triplets
    
    def _find_pair_index(self, metabolite_id: str, enzyme_id: str) -> Optional[int]:
        """Find index of a specific metabolite-enzyme pair."""
        for i, (met_id, enz_id) in enumerate(zip(self.metabolite_ids, self.enzyme_ids)):
            if met_id == metabolite_id and enz_id == enzyme_id:
                return i
        return None
    
    def __len__(self) -> int:
        if self.mode == 'classification':
            return len(self.labels)
        elif self.mode == 'contrastive':
            return len(self.triplets)
        else:  # combined
            return max(len(self.labels), len(self.triplets))
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.mode == 'classification':
            return {
                'metabolite_features': torch.FloatTensor(self.metabolite_features[idx]),
                'enzyme_features': torch.FloatTensor(self.enzyme_features[idx]),
                'labels': torch.FloatTensor([self.labels[idx]])
            }
        
        elif self.mode == 'contrastive':
            anchor_idx, pos_idx, neg_idx = self.triplets[idx]
            return {
                'anchor_metabolite': torch.FloatTensor(self.metabolite_features[anchor_idx]),
                'anchor_enzyme': torch.FloatTensor(self.enzyme_features[anchor_idx]),
                'positive_metabolite': torch.FloatTensor(self.metabolite_features[pos_idx]),
                'positive_enzyme': torch.FloatTensor(self.enzyme_features[pos_idx]),
                'negative_metabolite': torch.FloatTensor(self.metabolite_features[neg_idx]),
                'negative_enzyme': torch.FloatTensor(self.enzyme_features[neg_idx])
            }
        
        else:  # combined mode
            # Alternate between classification and contrastive samples
            if idx < len(self.labels):
                # Classification sample
                return {
                    'mode': 'classification',
                    'metabolite_features': torch.FloatTensor(self.metabolite_features[idx]),
                    'enzyme_features': torch.FloatTensor(self.enzyme_features[idx]),
                    'labels': torch.FloatTensor([self.labels[idx]])
                }
            else:
                # Contrastive sample
                triplet_idx = idx - len(self.labels)
                if triplet_idx < len(self.triplets):
                    anchor_idx, pos_idx, neg_idx = self.triplets[triplet_idx]
                    return {
                        'mode': 'contrastive',
                        'anchor_metabolite': torch.FloatTensor(self.metabolite_features[anchor_idx]),
                        'anchor_enzyme': torch.FloatTensor(self.enzyme_features[anchor_idx]),
                        'positive_metabolite': torch.FloatTensor(self.metabolite_features[pos_idx]),
                        'positive_enzyme': torch.FloatTensor(self.enzyme_features[pos_idx]),
                        'negative_metabolite': torch.FloatTensor(self.metabolite_features[neg_idx]),
                        'negative_enzyme': torch.FloatTensor(self.enzyme_features[neg_idx])
                    }
                else:
                    # Fallback to classification
                    idx = idx % len(self.labels)
                    return {
                        'mode': 'classification',
                        'metabolite_features': torch.FloatTensor(self.metabolite_features[idx]),
                        'enzyme_features': torch.FloatTensor(self.enzyme_features[idx]),
                        'labels': torch.FloatTensor([self.labels[idx]])
                    }


def load_enzyfind_data(data_dir: str, split: str = 'train') -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Load EnzyFind dataset from pickle files.
    
    Args:
        data_dir: Data directory path
        split: 'train' or 'test'
        
    Returns:
        Tuple of (metabolite_features, enzyme_features, labels, metabolite_ids, enzyme_ids)
    """
    # Load the appropriate split
    if split == 'train':
        df = pd.read_pickle(join(data_dir, "splits", "df_train_with_ESM1b_ts_GNN.pkl"))
    else:
        df = pd.read_pickle(join(data_dir, "splits", "df_test_with_ESM1b_ts_GNN.pkl"))
    
    # Filter out missing data
    df = df.loc[df["ESM1b"] != ""]
    df = df.loc[df["type"] != "engqvist"]
    df = df.loc[df["GNN rep"] != ""]
    df.reset_index(inplace=True, drop=True)
    
    # Convert string representations to arrays
    def string_to_array(series):
        return np.array([np.array(eval(item)) for item in series])
    
    # Extract features
    metabolite_features = string_to_array(df["GNN rep"])  # Using GNN representation for metabolites
    enzyme_features = string_to_array(df["ESM1b"])  # Using ESM1b for enzymes
    
    # Stack arrays properly
    metabolite_features = np.vstack(metabolite_features)
    enzyme_features = np.vstack(enzyme_features)
    
    # Extract labels and IDs
    labels = df["Binding"].values.astype(np.float32)
    metabolite_ids = df["molecule ID"].tolist()
    enzyme_ids = df["Uniprot ID"].tolist()
    
    return metabolite_features, enzyme_features, labels, metabolite_ids, enzyme_ids


def load_unimol_features(data_dir: str, metabolite_ids: List[str]) -> Optional[np.ndarray]:
    """
    Load Unimol features for metabolites.
    
    Args:
        data_dir: Data directory path
        metabolite_ids: List of metabolite IDs
        
    Returns:
        Unimol features or None if not available
    """
    try:
        # Try to load Unimol features if available
        unimol_file = join(data_dir, "substrate_data", "unimol_features.pkl")
        if os.path.exists(unimol_file):
            with open(unimol_file, 'rb') as f:
                unimol_data = pickle.load(f)
            
            features = []
            for met_id in metabolite_ids:
                if met_id in unimol_data:
                    features.append(unimol_data[met_id])
                else:
                    # Use zero vector if metabolite not found
                    features.append(np.zeros(unimol_data[list(unimol_data.keys())[0]].shape))
            
            return np.vstack(features)
    except:
        pass
    
    return None


def create_data_loaders(metabolite_features: np.ndarray,
                       enzyme_features: np.ndarray,
                       labels: np.ndarray,
                       metabolite_ids: List[str],
                       enzyme_ids: List[str],
                       mode: str = 'classification',
                       batch_size: int = 32,
                       test_size: float = 0.2,
                       random_state: int = 42) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        metabolite_features: Metabolite features
        enzyme_features: Enzyme features
        labels: Binary labels
        metabolite_ids: Metabolite IDs
        enzyme_ids: Enzyme IDs
        mode: Dataset mode ('classification', 'contrastive', 'combined')
        batch_size: Batch size
        test_size: Validation split ratio
        random_state: Random seed
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Split data
    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(indices, test_size=test_size, 
                                         random_state=random_state, stratify=labels)
    
    # Create datasets
    train_dataset = EnzymeMetaboliteDataset(
        metabolite_features[train_idx],
        enzyme_features[train_idx],
        labels[train_idx],
        [metabolite_ids[i] for i in train_idx],
        [enzyme_ids[i] for i in train_idx],
        mode=mode
    )
    
    val_dataset = EnzymeMetaboliteDataset(
        metabolite_features[val_idx],
        enzyme_features[val_idx],
        labels[val_idx],
        [metabolite_ids[i] for i in val_idx],
        [enzyme_ids[i] for i in val_idx],
        mode='classification'  # Always use classification mode for validation
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def create_test_loader(metabolite_features: np.ndarray,
                      enzyme_features: np.ndarray,
                      labels: np.ndarray,
                      metabolite_ids: List[str],
                      enzyme_ids: List[str],
                      batch_size: int = 32) -> DataLoader:
    """
    Create test data loader.
    
    Args:
        metabolite_features: Metabolite features
        enzyme_features: Enzyme features
        labels: Binary labels
        metabolite_ids: Metabolite IDs
        enzyme_ids: Enzyme IDs
        batch_size: Batch size
        
    Returns:
        Test data loader
    """
    test_dataset = EnzymeMetaboliteDataset(
        metabolite_features,
        enzyme_features,
        labels,
        metabolite_ids,
        enzyme_ids,
        mode='classification'
    )
    
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class BalancedBatchSampler:
    """
    Batch sampler that ensures balanced positive/negative samples in each batch.
    """
    
    def __init__(self, labels: np.ndarray, batch_size: int, positive_ratio: float = 0.5):
        """
        Initialize balanced batch sampler.
        
        Args:
            labels: Binary labels
            batch_size: Batch size
            positive_ratio: Ratio of positive samples in each batch
        """
        self.labels = labels
        self.batch_size = batch_size
        self.positive_ratio = positive_ratio
        
        self.positive_indices = np.where(labels == 1)[0]
        self.negative_indices = np.where(labels == 0)[0]
        
        self.n_positive_per_batch = int(batch_size * positive_ratio)
        self.n_negative_per_batch = batch_size - self.n_positive_per_batch
    
    def __iter__(self):
        """Generate balanced batches."""
        # Shuffle indices
        np.random.shuffle(self.positive_indices)
        np.random.shuffle(self.negative_indices)
        
        pos_idx = 0
        neg_idx = 0
        
        while (pos_idx + self.n_positive_per_batch <= len(self.positive_indices) and
               neg_idx + self.n_negative_per_batch <= len(self.negative_indices)):
            
            # Get positive and negative samples for this batch
            batch_pos = self.positive_indices[pos_idx:pos_idx + self.n_positive_per_batch]
            batch_neg = self.negative_indices[neg_idx:neg_idx + self.n_negative_per_batch]
            
            # Combine and shuffle
            batch_indices = np.concatenate([batch_pos, batch_neg])
            np.random.shuffle(batch_indices)
            
            yield batch_indices.tolist()
            
            pos_idx += self.n_positive_per_batch
            neg_idx += self.n_negative_per_batch
    
    def __len__(self):
        """Number of batches."""
        max_pos_batches = len(self.positive_indices) // self.n_positive_per_batch
        max_neg_batches = len(self.negative_indices) // self.n_negative_per_batch
        return min(max_pos_batches, max_neg_batches)