"""
ConPLex-style Model Architecture for Enzyme-Metabolite Interaction Prediction

This module implements the neural network architecture that:
1. Projects metabolite and enzyme features to a shared latent space
2. Performs binary classification using the joint embeddings
3. Supports contrastive learning with triplet loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class MLPProjector(nn.Module):
    """Multi-layer perceptron for projecting features to latent space."""
    
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int, 
                 dropout_rate: float = 0.2, activation: str = 'relu'):
        """
        Initialize MLP projector.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output latent space dimension
            dropout_rate: Dropout probability
            activation: Activation function ('relu', 'gelu', 'tanh')
        """
        super(MLPProjector, self).__init__()
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(activation),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer (no activation/dropout on final layer)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _get_activation(self, activation: str):
        """Get activation function."""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU()
        }
        return activations.get(activation, nn.ReLU())
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Projected features [batch_size, output_dim]
        """
        return self.mlp(x)


class ConPLexModel(nn.Module):
    """
    ConPLex-style model for enzyme-metabolite interaction prediction.
    
    This model:
    1. Projects metabolite and enzyme features to a shared latent space
    2. Performs binary classification using concatenated latent features
    3. Supports contrastive learning through triplet loss
    """
    
    def __init__(self, 
                 metabolite_dim: int,
                 enzyme_dim: int,
                 latent_dim: int = 128,
                 hidden_dims: list = [256, 128],
                 dropout_rate: float = 0.2,
                 activation: str = 'relu'):
        """
        Initialize ConPLex model.
        
        Args:
            metabolite_dim: Dimension of metabolite features (e.g., Unimol output)
            enzyme_dim: Dimension of enzyme features (e.g., ESM output)
            latent_dim: Dimension of shared latent space
            hidden_dims: Hidden layer dimensions for MLPs
            dropout_rate: Dropout probability
            activation: Activation function
        """
        super(ConPLexModel, self).__init__()
        
        self.metabolite_dim = metabolite_dim
        self.enzyme_dim = enzyme_dim
        self.latent_dim = latent_dim
        
        # Projectors to shared latent space
        self.metabolite_projector = MLPProjector(
            input_dim=metabolite_dim,
            hidden_dims=hidden_dims,
            output_dim=latent_dim,
            dropout_rate=dropout_rate,
            activation=activation
        )
        
        self.enzyme_projector = MLPProjector(
            input_dim=enzyme_dim,
            hidden_dims=hidden_dims,
            output_dim=latent_dim,
            dropout_rate=dropout_rate,
            activation=activation
        )
        
        # Binary classifier using concatenated latent features
        classifier_input_dim = latent_dim * 2  # Concatenated metabolite + enzyme embeddings
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[1], 1),
            nn.Sigmoid()
        )
    
    def forward(self, metabolite_features: torch.Tensor, 
                enzyme_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            metabolite_features: Metabolite features [batch_size, metabolite_dim]
            enzyme_features: Enzyme features [batch_size, enzyme_dim]
            
        Returns:
            Tuple of (predictions, metabolite_embeddings, enzyme_embeddings)
        """
        # Project to latent space
        metabolite_emb = self.metabolite_projector(metabolite_features)
        enzyme_emb = self.enzyme_projector(enzyme_features)
        
        # L2 normalize embeddings for contrastive learning
        metabolite_emb_norm = F.normalize(metabolite_emb, p=2, dim=1)
        enzyme_emb_norm = F.normalize(enzyme_emb, p=2, dim=1)
        
        # Concatenate for classification
        combined_features = torch.cat([metabolite_emb_norm, enzyme_emb_norm], dim=1)
        
        # Binary classification
        predictions = self.classifier(combined_features)
        
        return predictions, metabolite_emb_norm, enzyme_emb_norm
    
    def predict(self, metabolite_features: torch.Tensor, 
                enzyme_features: torch.Tensor) -> torch.Tensor:
        """
        Prediction mode (no gradient computation).
        
        Args:
            metabolite_features: Metabolite features [batch_size, metabolite_dim]
            enzyme_features: Enzyme features [batch_size, enzyme_dim]
            
        Returns:
            Interaction probabilities [batch_size, 1]
        """
        with torch.no_grad():
            predictions, _, _ = self.forward(metabolite_features, enzyme_features)
            return predictions
    
    def get_embeddings(self, metabolite_features: torch.Tensor = None, 
                      enzyme_features: torch.Tensor = None) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get latent embeddings for metabolites and/or enzymes.
        
        Args:
            metabolite_features: Metabolite features [batch_size, metabolite_dim]
            enzyme_features: Enzyme features [batch_size, enzyme_dim]
            
        Returns:
            Tuple of (metabolite_embeddings, enzyme_embeddings)
        """
        with torch.no_grad():
            metabolite_emb = None
            enzyme_emb = None
            
            if metabolite_features is not None:
                metabolite_emb = F.normalize(
                    self.metabolite_projector(metabolite_features), p=2, dim=1
                )
            
            if enzyme_features is not None:
                enzyme_emb = F.normalize(
                    self.enzyme_projector(enzyme_features), p=2, dim=1
                )
            
            return metabolite_emb, enzyme_emb
    
    def compute_similarity(self, metabolite_features: torch.Tensor, 
                          enzyme_features: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between metabolite and enzyme embeddings.
        
        Args:
            metabolite_features: Metabolite features [batch_size, metabolite_dim]
            enzyme_features: Enzyme features [batch_size, enzyme_dim]
            
        Returns:
            Cosine similarities [batch_size]
        """
        metabolite_emb, enzyme_emb = self.get_embeddings(metabolite_features, enzyme_features)
        
        # Compute cosine similarity
        similarities = torch.sum(metabolite_emb * enzyme_emb, dim=1)
        return similarities


class TripletLoss(nn.Module):
    """
    Triplet loss implementation for contrastive learning.
    
    Uses margin-based triplet loss to pull positive pairs closer and
    push negative pairs apart in the latent space.
    """
    
    def __init__(self, margin: float = 1.0, distance_metric: str = 'euclidean'):
        """
        Initialize triplet loss.
        
        Args:
            margin: Margin for triplet loss
            distance_metric: Distance metric ('euclidean' or 'cosine')
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.distance_metric = distance_metric
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, 
                negative: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet loss.
        
        Args:
            anchor: Anchor embeddings [batch_size, embedding_dim]
            positive: Positive embeddings [batch_size, embedding_dim]
            negative: Negative embeddings [batch_size, embedding_dim]
            
        Returns:
            Triplet loss value
        """
        if self.distance_metric == 'euclidean':
            pos_dist = F.pairwise_distance(anchor, positive, p=2)
            neg_dist = F.pairwise_distance(anchor, negative, p=2)
        elif self.distance_metric == 'cosine':
            pos_dist = 1 - F.cosine_similarity(anchor, positive)
            neg_dist = 1 - F.cosine_similarity(anchor, negative)
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
        
        # Margin-based triplet loss
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss function that balances binary classification and contrastive learning.
    """
    
    def __init__(self, classification_weight: float = 1.0, 
                 contrastive_weight: float = 1.0, margin: float = 1.0):
        """
        Initialize combined loss.
        
        Args:
            classification_weight: Weight for binary classification loss
            contrastive_weight: Weight for contrastive loss
            margin: Margin for triplet loss
        """
        super(CombinedLoss, self).__init__()
        self.classification_weight = classification_weight
        self.contrastive_weight = contrastive_weight
        
        self.bce_loss = nn.BCELoss()
        self.triplet_loss = TripletLoss(margin=margin)
    
    def forward(self, predictions: torch.Tensor, labels: torch.Tensor,
                anchor_emb: torch.Tensor, pos_emb: torch.Tensor, 
                neg_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            predictions: Model predictions [batch_size, 1]
            labels: True labels [batch_size, 1]
            anchor_emb: Anchor embeddings [batch_size, embedding_dim]
            pos_emb: Positive embeddings [batch_size, embedding_dim]
            neg_emb: Negative embeddings [batch_size, embedding_dim]
            
        Returns:
            Tuple of (total_loss, classification_loss, contrastive_loss)
        """
        # Classification loss
        cls_loss = self.bce_loss(predictions, labels)
        
        # Contrastive loss
        cont_loss = self.triplet_loss(anchor_emb, pos_emb, neg_emb)
        
        # Combined loss
        total_loss = (self.classification_weight * cls_loss + 
                     self.contrastive_weight * cont_loss)
        
        return total_loss, cls_loss, cont_loss