"""
Main training script for ConPLex-style contrastive learning on EnzyFind dataset.

This script implements the complete training pipeline:
1. Data loading and preprocessing
2. Two-stage training (classification then contrastive)
3. Evaluation and results analysis
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
import yaml
import os
import sys
from pathlib import Path
import logging
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contrastive_learning.model import ConPLexModel
from contrastive_learning.trainer import ConPLexTrainer
from contrastive_learning.data_loader import (
    load_enzyfind_data, create_data_loaders, create_test_loader
)


def setup_logging(log_dir: str = "logs"):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f"training_{timestamp}.log")),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(config_path: str):
    """Main training function."""
    
    # Setup
    logger = setup_logging()
    config = load_config(config_path)
    
    logger.info("Starting ConPLex-style contrastive learning training")
    logger.info(f"Configuration: {config}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    
    # Load data
    logger.info("Loading training data...")
    data_dir = config['data_dir']
    
    train_met_features, train_enz_features, train_labels, train_met_ids, train_enz_ids = load_enzyfind_data(
        data_dir, split='train'
    )
    
    logger.info("Loading test data...")
    test_met_features, test_enz_features, test_labels, test_met_ids, test_enz_ids = load_enzyfind_data(
        data_dir, split='test'
    )
    
    logger.info(f"Training data: {len(train_labels)} samples")
    logger.info(f"Test data: {len(test_labels)} samples")
    logger.info(f"Metabolite feature dim: {train_met_features.shape[1]}")
    logger.info(f"Enzyme feature dim: {train_enz_features.shape[1]}")
    logger.info(f"Positive ratio (train): {np.mean(train_labels):.3f}")
    logger.info(f"Positive ratio (test): {np.mean(test_labels):.3f}")
    
    # Create data loaders for classification stage
    logger.info("Creating data loaders for classification stage...")
    train_loader_cls, val_loader_cls = create_data_loaders(
        train_met_features, train_enz_features, train_labels,
        train_met_ids, train_enz_ids,
        mode='classification',
        batch_size=config['batch_size'],
        test_size=config['val_split'],
        random_state=config['random_seed']
    )
    
    # Create data loaders for contrastive stage
    logger.info("Creating data loaders for contrastive stage...")
    train_loader_cont, _ = create_data_loaders(
        train_met_features, train_enz_features, train_labels,
        train_met_ids, train_enz_ids,
        mode='contrastive',
        batch_size=config['batch_size'],
        test_size=config['val_split'],
        random_state=config['random_seed']
    )
    
    # Create test loader
    test_loader = create_test_loader(
        test_met_features, test_enz_features, test_labels,
        test_met_ids, test_enz_ids,
        batch_size=config['batch_size']
    )
    
    # Create model
    logger.info("Creating ConPLex model...")
    model = ConPLexModel(
        metabolite_dim=train_met_features.shape[1],
        enzyme_dim=train_enz_features.shape[1],
        latent_dim=config['model']['latent_dim'],
        hidden_dims=config['model']['hidden_dims'],
        dropout_rate=config['model']['dropout_rate'],
        activation=config['model']['activation']
    )
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = ConPLexTrainer(
        model=model,
        device=device,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        classification_weight=config['training']['classification_weight'],
        contrastive_weight=config['training']['contrastive_weight'],
        margin=config['training']['margin'],
        use_wandb=config.get('use_wandb', False),
        wandb_project=config.get('wandb_project', 'enzyfind-contrastive')
    )
    
    # Stage 1: Classification training
    logger.info("=" * 50)
    logger.info("STAGE 1: Classification Training")
    logger.info("=" * 50)
    
    cls_metrics = trainer.train_classification_stage(
        train_loader_cls, val_loader_cls,
        epochs=config['training']['classification_epochs'],
        patience=config['training']['patience']
    )
    
    logger.info(f"Best classification metrics: {cls_metrics}")
    
    # Stage 2: Contrastive training
    logger.info("=" * 50)
    logger.info("STAGE 2: Contrastive Training")
    logger.info("=" * 50)
    
    cont_metrics = trainer.train_contrastive_stage(
        train_loader_cont, val_loader_cls,
        epochs=config['training']['contrastive_epochs'],
        patience=config['training']['patience'],
        margin_decay=config['training']['margin_decay']
    )
    
    logger.info(f"Best contrastive metrics: {cont_metrics}")
    
    # Final evaluation
    logger.info("=" * 50)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 50)
    
    test_metrics = trainer.evaluate(test_loader)
    logger.info(f"Test metrics: {test_metrics}")
    
    # Save results
    results = {
        'config': config,
        'classification_metrics': cls_metrics,
        'contrastive_metrics': cont_metrics,
        'test_metrics': test_metrics,
        'data_info': {
            'train_samples': len(train_labels),
            'test_samples': len(test_labels),
            'metabolite_dim': train_met_features.shape[1],
            'enzyme_dim': train_enz_features.shape[1],
            'train_positive_ratio': float(np.mean(train_labels)),
            'test_positive_ratio': float(np.mean(test_labels))
        }
    }
    
    # Save results
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(os.path.join(results_dir, f"results_{timestamp}.yaml"), 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    
    logger.info(f"Results saved to {results_dir}/results_{timestamp}.yaml")
    logger.info("Training completed successfully!")
    
    return trainer, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ConPLex-style contrastive learning model")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Configuration file {args.config} not found!")
        print("Please create a configuration file or specify an existing one.")
        sys.exit(1)
    
    # Run training
    trainer, results = main(args.config)