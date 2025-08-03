"""
Integration guide for ConPLex-style contrastive learning with existing EnzyFind codebase.
"""

import sys
import os
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Add the parent directory to import existing utilities
sys.path.append('..')
sys.path.append('../additional_code')

# Import existing EnzyFind utilities
try:
    from data_preprocessing import *
    from xgboost_training import *
except ImportError:
    print("Warning: Could not import existing EnzyFind modules. Make sure you're in the right directory.")


def integrate_with_existing_pipeline():
    """
    Integration example showing how to use ConPLex with existing EnzyFind data pipeline.
    """
    
    print("ConPLex Integration with Existing EnzyFind Pipeline")
    print("=" * 60)
    
    # 1. Load existing processed data
    print("1. Loading existing EnzyFind data...")
    try:
        # Use existing data loading from main pipeline
        df_train = pd.read_pickle("../../data/splits/df_train_with_ESM1b_ts_GNN.pkl")
        df_test = pd.read_pickle("../../data/splits/df_test_with_ESM1b_ts_GNN.pkl")
        
        # Filter data as in original pipeline
        df_train = df_train.loc[df_train["ESM1b"] != ""]
        df_train = df_train.loc[df_train["type"] != "engqvist"]
        df_train = df_train.loc[df_train["GNN rep"] != ""]
        df_train.reset_index(inplace=True, drop=True)
        
        df_test = df_test.loc[df_test["ESM1b"] != ""]
        df_test = df_test.loc[df_test["type"] != "engqvist"]
        df_test = df_test.loc[df_test["GNN rep"] != ""]
        df_test.reset_index(inplace=True, drop=True)
        
        print(f"   Loaded {len(df_train)} training and {len(df_test)} test samples")
        
    except FileNotFoundError:
        print("   Data files not found. Please ensure data is properly processed.")
        return
    
    # 2. Prepare data for ConPLex
    print("2. Preparing data for ConPLex training...")
    
    def string_to_array(series):
        """Convert string representations to numpy arrays."""
        return np.array([np.array(eval(item)) for item in series])
    
    # Extract features (using existing representations)
    train_met_features = np.vstack(string_to_array(df_train["GNN rep"]))
    train_enz_features = np.vstack(string_to_array(df_train["ESM1b"]))
    train_labels = df_train["Binding"].values.astype(np.float32)
    
    test_met_features = np.vstack(string_to_array(df_test["GNN rep"]))
    test_enz_features = np.vstack(string_to_array(df_test["ESM1b"]))
    test_labels = df_test["Binding"].values.astype(np.float32)
    
    print(f"   Training features: metabolites {train_met_features.shape}, enzymes {train_enz_features.shape}")
    print(f"   Test features: metabolites {test_met_features.shape}, enzymes {test_enz_features.shape}")
    
    # 3. Train baseline XGBoost for comparison
    print("3. Training baseline XGBoost model...")
    
    # Concatenate features as in original pipeline
    X_train = np.hstack([train_met_features, train_enz_features])
    X_test = np.hstack([test_met_features, test_enz_features])
    
    try:
        import xgboost as xgb
        from sklearn.metrics import roc_auc_score
        
        # Simple XGBoost baseline
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        xgb_model.fit(X_train, train_labels)
        baseline_predictions = xgb_model.predict_proba(X_test)[:, 1]
        baseline_auc = roc_auc_score(test_labels, baseline_predictions)
        
        print(f"   Baseline XGBoost AUC: {baseline_auc:.4f}")
        
    except ImportError:
        print("   XGBoost not available, skipping baseline comparison")
        baseline_predictions = None
    
    # 4. Train ConPLex model
    print("4. Training ConPLex model...")
    
    try:
        # Import ConPLex modules
        from contrastive_learning.model import ConPLexModel
        from contrastive_learning.trainer import ConPLexTrainer
        from contrastive_learning.data_loader import create_data_loaders, create_test_loader
        
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model
        model = ConPLexModel(
            metabolite_dim=train_met_features.shape[1],
            enzyme_dim=train_enz_features.shape[1],
            latent_dim=128,
            hidden_dims=[256, 128],
            dropout_rate=0.2
        )
        
        # Create trainer
        trainer = ConPLexTrainer(
            model=model,
            device=device,
            learning_rate=0.001,
            weight_decay=0.0001,
            classification_weight=1.0,
            contrastive_weight=1.0,
            margin=1.0
        )
        
        # Create data loaders
        train_loader_cls, val_loader = create_data_loaders(
            train_met_features, train_enz_features, train_labels,
            df_train["molecule ID"].tolist(), df_train["Uniprot ID"].tolist(),
            mode='classification',
            batch_size=32,
            test_size=0.2,
            random_state=42
        )
        
        train_loader_cont, _ = create_data_loaders(
            train_met_features, train_enz_features, train_labels,
            df_train["molecule ID"].tolist(), df_train["Uniprot ID"].tolist(),
            mode='contrastive',
            batch_size=32,
            test_size=0.2,
            random_state=42
        )
        
        test_loader = create_test_loader(
            test_met_features, test_enz_features, test_labels,
            df_test["molecule ID"].tolist(), df_test["Uniprot ID"].tolist(),
            batch_size=32
        )
        
        # Train model (reduced epochs for demo)
        print("   Stage 1: Classification training...")
        cls_metrics = trainer.train_classification_stage(
            train_loader_cls, val_loader, epochs=3, patience=2
        )
        
        print("   Stage 2: Contrastive training...")
        cont_metrics = trainer.train_contrastive_stage(
            train_loader_cont, val_loader, epochs=5, patience=3
        )
        
        # Evaluate
        test_metrics = trainer.evaluate(test_loader)
        
        print(f"   ConPLex AUC: {test_metrics['auc']:.4f}")
        print(f"   ConPLex MCC: {test_metrics['mcc']:.4f}")
        
        # Compare with baseline
        if baseline_predictions is not None:
            improvement = test_metrics['auc'] - baseline_auc
            print(f"   Improvement over baseline: {improvement:+.4f} AUC")
        
    except ImportError as e:
        print(f"   ConPLex modules not available: {e}")
        print("   Please ensure PyTorch and ConPLex modules are properly installed")
    
    # 5. Save integration results
    print("5. Saving integration results...")
    
    results = {
        'data_info': {
            'train_samples': len(df_train),
            'test_samples': len(df_test),
            'metabolite_dim': train_met_features.shape[1],
            'enzyme_dim': train_enz_features.shape[1],
            'positive_ratio': float(np.mean(train_labels))
        },
        'baseline_metrics': {
            'auc': baseline_auc if baseline_predictions is not None else None
        } if baseline_predictions is not None else None,
        'conplex_metrics': test_metrics if 'test_metrics' in locals() else None
    }
    
    # Save to file
    os.makedirs("integration_results", exist_ok=True)
    with open("integration_results/integration_results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    print("   Results saved to integration_results/")
    
    print("\nIntegration completed successfully!")
    print("\nNext steps:")
    print("1. Review the results and tune hyperparameters")
    print("2. Run full training with more epochs")
    print("3. Perform comprehensive evaluation")
    print("4. Integrate into production pipeline")


def create_integration_script():
    """Create a standalone integration script."""
    
    script_content = '''#!/usr/bin/env python3
"""
Standalone integration script for ConPLex with EnzyFind.
Run this script to integrate ConPLex training with existing EnzyFind pipeline.
"""

import sys
import os
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

# Import and run integration
from integration import integrate_with_existing_pipeline

if __name__ == "__main__":
    integrate_with_existing_pipeline()
'''
    
    with open("run_integration.py", "w") as f:
        f.write(script_content)
    
    os.chmod("run_integration.py", 0o755)
    print("Created run_integration.py script")


if __name__ == "__main__":
    # Run integration demo
    integrate_with_existing_pipeline()
    
    # Create standalone script
    create_integration_script()