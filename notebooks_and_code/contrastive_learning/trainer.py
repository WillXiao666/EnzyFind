"""
Training Pipeline for ConPLex-style Contrastive Learning

This module implements the two-stage training approach:
1. Classification stage: Basic binary classification training
2. Contrastive stage: Contrastive learning with triplet loss
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, matthews_corrcoef, accuracy_score, precision_recall_fscore_support
from typing import Dict, List, Tuple, Optional
import wandb
import logging
from tqdm import tqdm
import os
import pickle
from datetime import datetime

from .model import ConPLexModel, CombinedLoss
from .data_loader import EnzymeMetaboliteDataset


class ConPLexTrainer:
    """
    Trainer class for ConPLex-style contrastive learning.
    
    Implements two-stage training:
    1. Classification stage with binary cross-entropy loss
    2. Contrastive stage with combined classification + triplet loss
    """
    
    def __init__(self,
                 model: ConPLexModel,
                 device: torch.device,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 classification_weight: float = 1.0,
                 contrastive_weight: float = 1.0,
                 margin: float = 1.0,
                 use_wandb: bool = False,
                 wandb_project: str = "enzyfind-contrastive"):
        """
        Initialize trainer.
        
        Args:
            model: ConPLex model
            device: Compute device
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            classification_weight: Weight for classification loss
            contrastive_weight: Weight for contrastive loss
            margin: Margin for triplet loss
            use_wandb: Whether to use Weights & Biases logging
            wandb_project: W&B project name
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Initialize loss functions
        self.bce_loss = nn.BCELoss()
        self.combined_loss = CombinedLoss(
            classification_weight=classification_weight,
            contrastive_weight=contrastive_weight,
            margin=margin
        )
        
        # Training history
        self.train_history = []
        self.val_history = []
        
        # Logging
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project=wandb_project)
            wandb.config.update({
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "classification_weight": classification_weight,
                "contrastive_weight": contrastive_weight,
                "margin": margin
            })
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def train_classification_stage(self,
                                 train_loader: DataLoader,
                                 val_loader: DataLoader,
                                 epochs: int = 10,
                                 patience: int = 5) -> Dict[str, float]:
        """
        Stage 1: Classification training with binary cross-entropy loss.
        
        Args:
            train_loader: Training data loader (classification mode)
            val_loader: Validation data loader
            epochs: Number of epochs
            patience: Early stopping patience
            
        Returns:
            Best validation metrics
        """
        self.logger.info("Starting classification stage training...")
        
        best_val_auc = 0.0
        patience_counter = 0
        best_metrics = {}
        
        for epoch in range(epochs):
            # Training
            train_metrics = self._train_classification_epoch(train_loader)
            
            # Validation
            val_metrics = self._validate_classification(val_loader)
            
            # Log metrics
            self.logger.info(f"Epoch {epoch+1}/{epochs}")
            self.logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, AUC: {train_metrics['auc']:.4f}")
            self.logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, AUC: {val_metrics['auc']:.4f}")
            
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "stage": "classification",
                    "train_loss": train_metrics['loss'],
                    "train_auc": train_metrics['auc'],
                    "val_loss": val_metrics['loss'],
                    "val_auc": val_metrics['auc']
                })
            
            # Early stopping
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                best_metrics = val_metrics.copy()
                patience_counter = 0
                # Save best model
                self._save_checkpoint("classification_best.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        self.logger.info("Classification stage completed!")
        return best_metrics
    
    def train_contrastive_stage(self,
                              train_loader: DataLoader,
                              val_loader: DataLoader,
                              epochs: int = 20,
                              patience: int = 10,
                              margin_decay: float = 0.95) -> Dict[str, float]:
        """
        Stage 2: Contrastive training with combined loss.
        
        Args:
            train_loader: Training data loader (contrastive or combined mode)
            val_loader: Validation data loader
            epochs: Number of epochs
            patience: Early stopping patience
            margin_decay: Decay factor for margin (following ConPLex approach)
            
        Returns:
            Best validation metrics
        """
        self.logger.info("Starting contrastive stage training...")
        
        # Load best classification model
        self._load_checkpoint("classification_best.pth")
        
        best_val_auc = 0.0
        patience_counter = 0
        best_metrics = {}
        current_margin = self.combined_loss.triplet_loss.margin
        
        for epoch in range(epochs):
            # Update margin (gradually decrease as in ConPLex)
            if epoch > 0:
                current_margin *= margin_decay
                self.combined_loss.triplet_loss.margin = current_margin
            
            # Training
            train_metrics = self._train_contrastive_epoch(train_loader)
            
            # Validation
            val_metrics = self._validate_classification(val_loader)
            
            # Log metrics
            self.logger.info(f"Epoch {epoch+1}/{epochs}")
            self.logger.info(f"Train - Total Loss: {train_metrics['total_loss']:.4f}, "
                           f"Cls Loss: {train_metrics['cls_loss']:.4f}, "
                           f"Cont Loss: {train_metrics['cont_loss']:.4f}")
            self.logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, AUC: {val_metrics['auc']:.4f}")
            self.logger.info(f"Current margin: {current_margin:.4f}")
            
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "stage": "contrastive",
                    "train_total_loss": train_metrics['total_loss'],
                    "train_cls_loss": train_metrics['cls_loss'],
                    "train_cont_loss": train_metrics['cont_loss'],
                    "val_loss": val_metrics['loss'],
                    "val_auc": val_metrics['auc'],
                    "margin": current_margin
                })
            
            # Early stopping
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                best_metrics = val_metrics.copy()
                patience_counter = 0
                # Save best model
                self._save_checkpoint("contrastive_best.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        self.logger.info("Contrastive stage completed!")
        return best_metrics
    
    def _train_classification_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train one epoch with classification loss only."""
        self.model.train()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            # Move to device
            metabolite_features = batch['metabolite_features'].to(self.device)
            enzyme_features = batch['enzyme_features'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            predictions, _, _ = self.model(metabolite_features, enzyme_features)
            
            # Compute loss
            loss = self.bce_loss(predictions, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            all_predictions.extend(predictions.cpu().detach().numpy())
            all_labels.extend(labels.cpu().detach().numpy())
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        auc = roc_auc_score(all_labels, all_predictions)
        
        return {'loss': avg_loss, 'auc': auc}
    
    def _train_contrastive_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train one epoch with combined classification and contrastive loss."""
        self.model.train()
        
        total_loss = 0.0
        total_cls_loss = 0.0
        total_cont_loss = 0.0
        batch_count = 0
        
        progress_bar = tqdm(train_loader, desc="Training (Contrastive)")
        for batch in progress_bar:
            batch_count += 1
            
            if 'mode' in batch and batch['mode'][0] == 'classification':
                # Classification batch
                metabolite_features = batch['metabolite_features'].to(self.device)
                enzyme_features = batch['enzyme_features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                predictions, _, _ = self.model(metabolite_features, enzyme_features)
                loss = self.bce_loss(predictions, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                total_cls_loss += loss.item()
                
            else:
                # Contrastive batch (triplet)
                anchor_met = batch['anchor_metabolite'].to(self.device)
                anchor_enz = batch['anchor_enzyme'].to(self.device)
                pos_met = batch['positive_metabolite'].to(self.device)
                pos_enz = batch['positive_enzyme'].to(self.device)
                neg_met = batch['negative_metabolite'].to(self.device)
                neg_enz = batch['negative_enzyme'].to(self.device)
                
                # Forward pass for anchor (positive pair)
                anchor_pred, anchor_met_emb, anchor_enz_emb = self.model(anchor_met, anchor_enz)
                
                # Forward pass for positive
                pos_pred, pos_met_emb, pos_enz_emb = self.model(pos_met, pos_enz)
                
                # Forward pass for negative
                neg_pred, neg_met_emb, neg_enz_emb = self.model(neg_met, neg_enz)
                
                # Create labels (positive pairs have label 1)
                labels = torch.ones_like(anchor_pred)
                
                # Compute combined loss
                # Use enzyme embeddings as anchor, metabolite embeddings as positive/negative
                total_loss_batch, cls_loss_batch, cont_loss_batch = self.combined_loss(
                    anchor_pred, labels,
                    anchor_enz_emb, anchor_met_emb, neg_met_emb
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss_batch.backward()
                self.optimizer.step()
                
                total_loss += total_loss_batch.item()
                total_cls_loss += cls_loss_batch.item()
                total_cont_loss += cont_loss_batch.item()
            
            progress_bar.set_postfix({
                'total_loss': total_loss / batch_count,
                'cls_loss': total_cls_loss / batch_count,
                'cont_loss': total_cont_loss / batch_count
            })
        
        return {
            'total_loss': total_loss / len(train_loader),
            'cls_loss': total_cls_loss / len(train_loader),
            'cont_loss': total_cont_loss / len(train_loader)
        }
    
    def _validate_classification(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model on classification task."""
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                metabolite_features = batch['metabolite_features'].to(self.device)
                enzyme_features = batch['enzyme_features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                predictions, _, _ = self.model(metabolite_features, enzyme_features)
                loss = self.bce_loss(predictions, labels)
                
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        auc = roc_auc_score(all_labels, all_predictions)
        
        # Additional metrics
        binary_predictions = (np.array(all_predictions) > 0.5).astype(int)
        accuracy = accuracy_score(all_labels, binary_predictions)
        mcc = matthews_corrcoef(all_labels, binary_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, binary_predictions, average='binary'
        )
        
        return {
            'loss': avg_loss,
            'auc': auc,
            'accuracy': accuracy,
            'mcc': mcc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Test metrics
        """
        self.logger.info("Evaluating model...")
        
        # Load best model
        try:
            self._load_checkpoint("contrastive_best.pth")
        except:
            try:
                self._load_checkpoint("classification_best.pth")
            except:
                self.logger.warning("No saved model found, using current model")
        
        return self._validate_classification(test_loader)
    
    def predict(self, metabolite_features: torch.Tensor, 
                enzyme_features: torch.Tensor) -> torch.Tensor:
        """
        Make predictions for metabolite-enzyme pairs.
        
        Args:
            metabolite_features: Metabolite features [batch_size, metabolite_dim]
            enzyme_features: Enzyme features [batch_size, enzyme_dim]
            
        Returns:
            Interaction probabilities [batch_size, 1]
        """
        self.model.eval()
        metabolite_features = metabolite_features.to(self.device)
        enzyme_features = enzyme_features.to(self.device)
        
        return self.model.predict(metabolite_features, enzyme_features)
    
    def get_embeddings(self, metabolite_features: torch.Tensor = None,
                      enzyme_features: torch.Tensor = None) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get latent embeddings for metabolites and/or enzymes.
        
        Args:
            metabolite_features: Metabolite features
            enzyme_features: Enzyme features
            
        Returns:
            Tuple of (metabolite_embeddings, enzyme_embeddings)
        """
        self.model.eval()
        
        if metabolite_features is not None:
            metabolite_features = metabolite_features.to(self.device)
        if enzyme_features is not None:
            enzyme_features = enzyme_features.to(self.device)
        
        return self.model.get_embeddings(metabolite_features, enzyme_features)
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_history': self.train_history,
            'val_history': self.val_history
        }
        
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(checkpoint, os.path.join("checkpoints", filename))
    
    def _load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint_path = os.path.join("checkpoints", filename)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.train_history = checkpoint.get('train_history', [])
            self.val_history = checkpoint.get('val_history', [])
            self.logger.info(f"Loaded checkpoint: {filename}")
        else:
            raise FileNotFoundError(f"Checkpoint {filename} not found")


def train_conplex_model(config: Dict) -> ConPLexTrainer:
    """
    Main training function for ConPLex model.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Trained ConPLexTrainer
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data (implementation would depend on your data structure)
    # This is a placeholder - you would implement based on your data loading needs
    
    # Create model
    model = ConPLexModel(
        metabolite_dim=config['metabolite_dim'],
        enzyme_dim=config['enzyme_dim'],
        latent_dim=config['latent_dim'],
        hidden_dims=config['hidden_dims'],
        dropout_rate=config['dropout_rate']
    )
    
    # Create trainer
    trainer = ConPLexTrainer(
        model=model,
        device=device,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        classification_weight=config['classification_weight'],
        contrastive_weight=config['contrastive_weight'],
        margin=config['margin'],
        use_wandb=config.get('use_wandb', False)
    )
    
    return trainer