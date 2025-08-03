"""
Evaluation and Utility Functions for ConPLex Model

This module provides:
1. Model evaluation metrics
2. Visualization utilities
3. Embedding analysis tools
4. Comparison with baseline methods
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, 
    matthews_corrcoef, accuracy_score, confusion_matrix,
    average_precision_score
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional
import os
from pathlib import Path

from .trainer import ConPLexTrainer
from .model import ConPLexModel


class ConPLexEvaluator:
    """
    Evaluation utilities for ConPLex model.
    """
    
    def __init__(self, trainer: ConPLexTrainer):
        """
        Initialize evaluator.
        
        Args:
            trainer: Trained ConPLexTrainer instance
        """
        self.trainer = trainer
        self.model = trainer.model
        self.device = trainer.device
    
    def comprehensive_evaluation(self, test_loader, save_dir: str = "evaluation_results") -> Dict:
        """
        Perform comprehensive evaluation of the model.
        
        Args:
            test_loader: Test data loader
            save_dir: Directory to save evaluation results
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Get predictions and embeddings
        predictions, labels, met_embeddings, enz_embeddings, met_ids, enz_ids = self._get_predictions_and_embeddings(test_loader)
        
        # Calculate metrics
        metrics = self._calculate_metrics(predictions, labels)
        
        # Create visualizations
        self._plot_roc_curve(predictions, labels, save_dir)
        self._plot_precision_recall_curve(predictions, labels, save_dir)
        self._plot_confusion_matrix(predictions, labels, save_dir)
        self._plot_prediction_distribution(predictions, labels, save_dir)
        
        # Embedding analysis
        self._analyze_embeddings(met_embeddings, enz_embeddings, labels, save_dir)
        
        # Top predictions analysis
        top_predictions = self._analyze_top_predictions(predictions, labels, met_ids, enz_ids)
        
        results = {
            'metrics': metrics,
            'top_predictions': top_predictions,
            'embedding_stats': self._get_embedding_statistics(met_embeddings, enz_embeddings)
        }
        
        # Save results
        pd.DataFrame([metrics]).to_csv(os.path.join(save_dir, "metrics.csv"), index=False)
        
        return results
    
    def _get_predictions_and_embeddings(self, test_loader):
        """Get model predictions and embeddings for test data."""
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_met_embeddings = []
        all_enz_embeddings = []
        all_met_ids = []
        all_enz_ids = []
        
        with torch.no_grad():
            for batch in test_loader:
                metabolite_features = batch['metabolite_features'].to(self.device)
                enzyme_features = batch['enzyme_features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Get predictions and embeddings
                predictions, met_emb, enz_emb = self.model(metabolite_features, enzyme_features)
                
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
                all_met_embeddings.extend(met_emb.cpu().numpy())
                all_enz_embeddings.extend(enz_emb.cpu().numpy())
                
                # Note: This assumes test_loader provides IDs - you may need to modify based on your data loader
                if 'metabolite_ids' in batch:
                    all_met_ids.extend(batch['metabolite_ids'])
                if 'enzyme_ids' in batch:
                    all_enz_ids.extend(batch['enzyme_ids'])
        
        return (np.array(all_predictions), np.array(all_labels), 
                np.array(all_met_embeddings), np.array(all_enz_embeddings),
                all_met_ids, all_enz_ids)
    
    def _calculate_metrics(self, predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        # Threshold predictions
        binary_predictions = (predictions > 0.5).astype(int)
        
        metrics = {
            'auc_roc': roc_auc_score(labels, predictions),
            'auc_pr': average_precision_score(labels, predictions),
            'accuracy': accuracy_score(labels, binary_predictions),
            'mcc': matthews_corrcoef(labels, binary_predictions),
            'sensitivity': np.sum((labels == 1) & (binary_predictions == 1)) / np.sum(labels == 1),
            'specificity': np.sum((labels == 0) & (binary_predictions == 0)) / np.sum(labels == 0),
            'precision': np.sum((labels == 1) & (binary_predictions == 1)) / np.sum(binary_predictions == 1) if np.sum(binary_predictions == 1) > 0 else 0,
            'recall': np.sum((labels == 1) & (binary_predictions == 1)) / np.sum(labels == 1),
        }
        
        # F1 score
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1'] = 0
        
        return metrics
    
    def _plot_roc_curve(self, predictions: np.ndarray, labels: np.ndarray, save_dir: str):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(labels, predictions)
        auc = roc_auc_score(labels, predictions)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ConPLex (AUC = {auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "roc_curve.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_precision_recall_curve(self, predictions: np.ndarray, labels: np.ndarray, save_dir: str):
        """Plot Precision-Recall curve."""
        precision, recall, _ = precision_recall_curve(labels, predictions)
        auc_pr = average_precision_score(labels, predictions)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'ConPLex (AUC-PR = {auc_pr:.3f})', linewidth=2)
        plt.axhline(y=np.mean(labels), color='k', linestyle='--', alpha=0.5, 
                   label=f'Random (AUC-PR = {np.mean(labels):.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "precision_recall_curve.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrix(self, predictions: np.ndarray, labels: np.ndarray, save_dir: str):
        """Plot confusion matrix."""
        binary_predictions = (predictions > 0.5).astype(int)
        cm = confusion_matrix(labels, binary_predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Non-interacting', 'Interacting'],
                   yticklabels=['Non-interacting', 'Interacting'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_prediction_distribution(self, predictions: np.ndarray, labels: np.ndarray, save_dir: str):
        """Plot distribution of predictions by class."""
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(predictions[labels == 0], bins=50, alpha=0.7, label='Non-interacting', density=True)
        plt.hist(predictions[labels == 1], bins=50, alpha=0.7, label='Interacting', density=True)
        plt.xlabel('Prediction Score')
        plt.ylabel('Density')
        plt.title('Prediction Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.boxplot([predictions[labels == 0], predictions[labels == 1]], 
                   labels=['Non-interacting', 'Interacting'])
        plt.ylabel('Prediction Score')
        plt.title('Prediction Box Plot')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "prediction_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _analyze_embeddings(self, met_embeddings: np.ndarray, enz_embeddings: np.ndarray, 
                          labels: np.ndarray, save_dir: str):
        """Analyze and visualize embeddings."""
        # Combine embeddings for joint analysis
        all_embeddings = np.vstack([met_embeddings, enz_embeddings])
        
        # PCA analysis
        pca = PCA(n_components=2)
        embeddings_pca = pca.fit_transform(all_embeddings)
        
        plt.figure(figsize=(12, 5))
        
        # PCA plot
        plt.subplot(1, 2, 1)
        n_met = len(met_embeddings)
        plt.scatter(embeddings_pca[:n_met, 0], embeddings_pca[:n_met, 1], 
                   c=labels, alpha=0.6, label='Metabolites', s=20)
        plt.scatter(embeddings_pca[n_met:, 0], embeddings_pca[n_met:, 1], 
                   c=labels, alpha=0.6, label='Enzymes', marker='^', s=20)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('PCA of Embeddings')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # t-SNE plot (if data is not too large)
        if len(all_embeddings) <= 5000:
            tsne = TSNE(n_components=2, random_state=42)
            embeddings_tsne = tsne.fit_transform(all_embeddings)
            
            plt.subplot(1, 2, 2)
            plt.scatter(embeddings_tsne[:n_met, 0], embeddings_tsne[:n_met, 1], 
                       c=labels, alpha=0.6, label='Metabolites', s=20)
            plt.scatter(embeddings_tsne[n_met:, 0], embeddings_tsne[n_met:, 1], 
                       c=labels, alpha=0.6, label='Enzymes', marker='^', s=20)
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.title('t-SNE of Embeddings')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "embedding_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _analyze_top_predictions(self, predictions: np.ndarray, labels: np.ndarray, 
                               met_ids: List[str], enz_ids: List[str], top_k: int = 100) -> Dict:
        """Analyze top predictions."""
        # Sort by prediction score
        sorted_indices = np.argsort(predictions)[::-1]
        
        # Top predictions
        top_indices = sorted_indices[:top_k]
        top_predictions_analysis = {
            'top_k': top_k,
            'true_positives': np.sum(labels[top_indices] == 1),
            'precision_at_k': np.sum(labels[top_indices] == 1) / top_k,
            'avg_score': np.mean(predictions[top_indices])
        }
        
        # Create DataFrame of top predictions
        if met_ids and enz_ids:
            top_df = pd.DataFrame({
                'metabolite_id': [met_ids[i] for i in top_indices],
                'enzyme_id': [enz_ids[i] for i in top_indices],
                'prediction_score': predictions[top_indices],
                'true_label': labels[top_indices]
            })
            top_df.to_csv(os.path.join("evaluation_results", f"top_{top_k}_predictions.csv"), index=False)
        
        return top_predictions_analysis
    
    def _get_embedding_statistics(self, met_embeddings: np.ndarray, enz_embeddings: np.ndarray) -> Dict:
        """Get statistics about the learned embeddings."""
        return {
            'metabolite_embedding_mean': np.mean(met_embeddings, axis=0).tolist(),
            'metabolite_embedding_std': np.std(met_embeddings, axis=0).tolist(),
            'enzyme_embedding_mean': np.mean(enz_embeddings, axis=0).tolist(),
            'enzyme_embedding_std': np.std(enz_embeddings, axis=0).tolist(),
            'avg_cosine_similarity': np.mean([
                np.dot(met_embeddings[i], enz_embeddings[i]) / 
                (np.linalg.norm(met_embeddings[i]) * np.linalg.norm(enz_embeddings[i]))
                for i in range(len(met_embeddings))
            ])
        }
    
    def compare_with_baseline(self, baseline_predictions: np.ndarray, 
                            test_predictions: np.ndarray, test_labels: np.ndarray, 
                            save_dir: str = "comparison_results"):
        """
        Compare ConPLex model with baseline method.
        
        Args:
            baseline_predictions: Baseline model predictions
            test_predictions: ConPLex model predictions
            test_labels: True labels
            save_dir: Directory to save comparison results
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Calculate metrics for both models
        baseline_metrics = self._calculate_metrics(baseline_predictions, test_labels)
        conplex_metrics = self._calculate_metrics(test_predictions, test_labels)
        
        # Create comparison plot
        metrics_names = ['auc_roc', 'auc_pr', 'accuracy', 'mcc', 'f1']
        baseline_values = [baseline_metrics[m] for m in metrics_names]
        conplex_values = [conplex_metrics[m] for m in metrics_names]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, baseline_values, width, label='Baseline', alpha=0.8)
        plt.bar(x + width/2, conplex_values, width, label='ConPLex', alpha=0.8)
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Model Comparison')
        plt.xticks(x, [m.upper().replace('_', '-') for m in metrics_names])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (b_val, c_val) in enumerate(zip(baseline_values, conplex_values)):
            plt.text(i - width/2, b_val + 0.01, f'{b_val:.3f}', ha='center', va='bottom')
            plt.text(i + width/2, c_val + 0.01, f'{c_val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "model_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save comparison metrics
        comparison_df = pd.DataFrame({
            'Metric': metrics_names,
            'Baseline': baseline_values,
            'ConPLex': conplex_values,
            'Improvement': [c - b for c, b in zip(conplex_values, baseline_values)]
        })
        comparison_df.to_csv(os.path.join(save_dir, "comparison_metrics.csv"), index=False)
        
        return comparison_df


def load_and_evaluate_model(model_path: str, test_loader, config: Dict) -> ConPLexEvaluator:
    """
    Load a trained model and create evaluator.
    
    Args:
        model_path: Path to saved model checkpoint
        test_loader: Test data loader
        config: Model configuration
        
    Returns:
        ConPLexEvaluator instance
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Recreate model
    model = ConPLexModel(
        metabolite_dim=config['metabolite_dim'],
        enzyme_dim=config['enzyme_dim'],
        latent_dim=config['latent_dim'],
        hidden_dims=config['hidden_dims'],
        dropout_rate=config['dropout_rate']
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create trainer (for compatibility)
    trainer = ConPLexTrainer(model=model, device=device)
    
    return ConPLexEvaluator(trainer)