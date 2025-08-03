"""
Demonstration script for ConPLex-style contrastive learning.

This script provides a complete example of how to use the contrastive learning
implementation for enzyme-metabolite interaction prediction.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Add the contrastive learning module to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from contrastive_learning.model import ConPLexModel
from contrastive_learning.trainer import ConPLexTrainer
from contrastive_learning.data_loader import load_enzyfind_data, create_data_loaders
from contrastive_learning.evaluation import ConPLexEvaluator


def demo_training():
    """Demonstrate the complete training pipeline."""
    
    print("="*60)
    print("ConPLex-Style Contrastive Learning Demo")
    print("="*60)
    
    # Configuration
    config = {
        'data_dir': '../data',
        'batch_size': 32,
        'val_split': 0.2,
        'random_seed': 42,
        'model': {
            'latent_dim': 64,
            'hidden_dims': [128, 64],
            'dropout_rate': 0.2,
            'activation': 'relu'
        },
        'training': {
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'classification_weight': 1.0,
            'contrastive_weight': 0.5,
            'margin': 1.0,
            'classification_epochs': 3,  # Reduced for demo
            'contrastive_epochs': 5,     # Reduced for demo
            'patience': 3
        }
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("\n1. Loading data...")
    try:
        train_met_features, train_enz_features, train_labels, train_met_ids, train_enz_ids = load_enzyfind_data(
            config['data_dir'], split='train'
        )
        print(f"   Training data: {len(train_labels)} samples")
        print(f"   Metabolite features: {train_met_features.shape[1]} dimensions")
        print(f"   Enzyme features: {train_enz_features.shape[1]} dimensions")
        print(f"   Positive ratio: {np.mean(train_labels):.3f}")
    except Exception as e:
        print(f"   Error loading data: {e}")
        print("   Using synthetic data for demo...")
        
        # Create synthetic data for demonstration
        n_samples = 1000
        train_met_features = np.random.randn(n_samples, 50)  # 50D metabolite features
        train_enz_features = np.random.randn(n_samples, 1280)  # 1280D enzyme features (ESM size)
        train_labels = np.random.binomial(1, 0.1, n_samples).astype(np.float32)  # 10% positive
        train_met_ids = [f"metabolite_{i}" for i in range(n_samples)]
        train_enz_ids = [f"enzyme_{i}" for i in range(n_samples)]
        
        print(f"   Synthetic training data: {len(train_labels)} samples")
        print(f"   Metabolite features: {train_met_features.shape[1]} dimensions")
        print(f"   Enzyme features: {train_enz_features.shape[1]} dimensions")
        print(f"   Positive ratio: {np.mean(train_labels):.3f}")
    
    # Create data loaders
    print("\n2. Creating data loaders...")
    train_loader_cls, val_loader_cls = create_data_loaders(
        train_met_features, train_enz_features, train_labels,
        train_met_ids, train_enz_ids,
        mode='classification',
        batch_size=config['batch_size'],
        test_size=config['val_split'],
        random_state=config['random_seed']
    )
    
    train_loader_cont, _ = create_data_loaders(
        train_met_features, train_enz_features, train_labels,
        train_met_ids, train_enz_ids,
        mode='contrastive',
        batch_size=config['batch_size'],
        test_size=config['val_split'],
        random_state=config['random_seed']
    )
    
    print(f"   Classification loader: {len(train_loader_cls)} batches")
    print(f"   Contrastive loader: {len(train_loader_cont)} batches")
    
    # Create model
    print("\n3. Creating ConPLex model...")
    model = ConPLexModel(
        metabolite_dim=train_met_features.shape[1],
        enzyme_dim=train_enz_features.shape[1],
        latent_dim=config['model']['latent_dim'],
        hidden_dims=config['model']['hidden_dims'],
        dropout_rate=config['model']['dropout_rate'],
        activation=config['model']['activation']
    )
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Model architecture:")
    print(f"     - Metabolite projector: {train_met_features.shape[1]} -> {config['model']['latent_dim']}")
    print(f"     - Enzyme projector: {train_enz_features.shape[1]} -> {config['model']['latent_dim']}")
    print(f"     - Classifier: {config['model']['latent_dim']*2} -> 1")
    
    # Create trainer
    print("\n4. Creating trainer...")
    trainer = ConPLexTrainer(
        model=model,
        device=device,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        classification_weight=config['training']['classification_weight'],
        contrastive_weight=config['training']['contrastive_weight'],
        margin=config['training']['margin'],
        use_wandb=False  # Disable W&B for demo
    )
    
    # Stage 1: Classification training
    print("\n5. Stage 1: Classification Training")
    print("-" * 40)
    cls_metrics = trainer.train_classification_stage(
        train_loader_cls, val_loader_cls,
        epochs=config['training']['classification_epochs'],
        patience=config['training']['patience']
    )
    
    print(f"   Best classification metrics:")
    for metric, value in cls_metrics.items():
        print(f"     {metric}: {value:.4f}")
    
    # Stage 2: Contrastive training
    print("\n6. Stage 2: Contrastive Training")
    print("-" * 40)
    cont_metrics = trainer.train_contrastive_stage(
        train_loader_cont, val_loader_cls,
        epochs=config['training']['contrastive_epochs'],
        patience=config['training']['patience'],
        margin_decay=0.9
    )
    
    print(f"   Best contrastive metrics:")
    for metric, value in cont_metrics.items():
        print(f"     {metric}: {value:.4f}")
    
    # Demonstrate prediction
    print("\n7. Demonstration of Predictions")
    print("-" * 40)
    
    # Sample a few examples
    sample_indices = np.random.choice(len(val_loader_cls.dataset), size=5, replace=False)
    
    model.eval()
    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            sample = val_loader_cls.dataset[idx]
            
            met_feat = sample['metabolite_features'].unsqueeze(0).to(device)
            enz_feat = sample['enzyme_features'].unsqueeze(0).to(device)
            true_label = sample['labels'].item()
            
            prediction = trainer.predict(met_feat, enz_feat).item()
            
            print(f"   Sample {i+1}:")
            print(f"     True label: {true_label}")
            print(f"     Prediction: {prediction:.4f}")
            print(f"     Predicted class: {'Interacting' if prediction > 0.5 else 'Non-interacting'}")
    
    # Demonstrate embedding analysis
    print("\n8. Embedding Analysis")
    print("-" * 40)
    
    # Get embeddings for a batch
    sample_batch = next(iter(val_loader_cls))
    met_feat = sample_batch['metabolite_features'][:10].to(device)
    enz_feat = sample_batch['enzyme_features'][:10].to(device)
    
    met_emb, enz_emb = trainer.get_embeddings(met_feat, enz_feat)
    
    print(f"   Metabolite embeddings shape: {met_emb.shape}")
    print(f"   Enzyme embeddings shape: {enz_emb.shape}")
    print(f"   Average cosine similarity: {torch.mean(torch.sum(met_emb * enz_emb, dim=1)):.4f}")
    
    # Compute pairwise similarities
    similarities = trainer.model.compute_similarity(met_feat, enz_feat)
    print(f"   Similarity scores: {similarities.cpu().numpy()}")
    
    print("\n" + "="*60)
    print("Demo completed successfully!")
    print("="*60)
    
    return trainer, cls_metrics, cont_metrics


def demo_evaluation():
    """Demonstrate model evaluation capabilities."""
    
    print("\n" + "="*60)
    print("ConPLex Model Evaluation Demo")
    print("="*60)
    
    # Create synthetic test data
    n_test = 200
    test_met_features = np.random.randn(n_test, 50)
    test_enz_features = np.random.randn(n_test, 1280)
    test_labels = np.random.binomial(1, 0.1, n_test).astype(np.float32)
    test_met_ids = [f"test_metabolite_{i}" for i in range(n_test)]
    test_enz_ids = [f"test_enzyme_{i}" for i in range(n_test)]
    
    # Create a simple model for demonstration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConPLexModel(
        metabolite_dim=50,
        enzyme_dim=1280,
        latent_dim=64,
        hidden_dims=[128, 64]
    )
    
    trainer = ConPLexTrainer(model=model, device=device)
    evaluator = ConPLexEvaluator(trainer)
    
    print("Evaluation capabilities:")
    print("- Comprehensive metrics calculation")
    print("- ROC and Precision-Recall curves")
    print("- Confusion matrix visualization")
    print("- Embedding analysis and visualization")
    print("- Top predictions analysis")
    print("- Model comparison with baselines")
    
    print("\nEvaluation demo completed!")
    
    return evaluator


if __name__ == "__main__":
    print("Starting ConPLex demonstration...")
    
    # Run training demo
    trainer, cls_metrics, cont_metrics = demo_training()
    
    # Run evaluation demo
    evaluator = demo_evaluation()
    
    print("\nAll demonstrations completed successfully!")
    print("\nTo run the full training pipeline:")
    print("  python train_conplex.py --config config.yaml")
    print("\nTo customize the configuration:")
    print("  Edit config.yaml with your preferred settings")