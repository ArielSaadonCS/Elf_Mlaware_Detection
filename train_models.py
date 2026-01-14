"""
Train and Compare Models for all three pipelines
Trains RF and XGBoost for each pipeline and compares results.
"""
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from xgboost import XGBClassifier

# Create models directory
# ===========================
# Ensure that a directory exists for saving trained models.
# This allows the training script to persist models for later inference by the worker.
# ===========================
os.makedirs("models", exist_ok=True)

# ===========================
# Train two classifiers (Random Forest and XGBoost) on a given feature pipeline,
# evaluate them on a held-out test set, print metrics, and save the trained models.
# ===========================

def train_and_evaluate(X_train, X_test, y_train, y_test, pipeline_name):
    """Train RF and XGBoost, return metrics."""

    # Print basic information about the current training run
    print(f"\n{'='*50}")
    print(f"Training {pipeline_name}")
    print(f"{'='*50}")
    print(f"Features: {X_train.shape[1]}, Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Dictionary for collecting evaluation metrics of both models
    results = {}
    
    # ============ Random Forest ============
    # Random Forest training and evaluation:
    # Uses many decision trees trained on random feature 
    # subsets to reduce overfitting.
    # ===========================
    print(f"\n--- Random Forest ---")
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Predict class labels and probabilities on test set
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)
    
    # Compute standard classification metrics
    results['rf'] = {
        'accuracy': accuracy_score(y_test, rf_pred),
        'precision': precision_score(y_test, rf_pred, zero_division=0),
        'recall': recall_score(y_test, rf_pred, zero_division=0),
        'f1': f1_score(y_test, rf_pred, zero_division=0),
    }
    
    # Print evaluation results
    print(f"Accuracy:  {results['rf']['accuracy']:.4f}")
    print(f"Precision: {results['rf']['precision']:.4f}")
    print(f"Recall:    {results['rf']['recall']:.4f}")
    print(f"F1-Score:  {results['rf']['f1']:.4f}")
    
    # Save model
    joblib.dump(rf, f"models/{pipeline_name}_rf.pkl")
    
    # ===========================
    # Display top-10 most important features according to Random Forest.
    # Helps analyze which features contribute most to the decision.
    # ===========================
    if hasattr(X_train, 'columns'):
        importances = pd.Series(rf.feature_importances_, index=X_train.columns)
        print(f"\nTop 10 important features:")
        for feat, imp in importances.nlargest(10).items():
            print(f"  {feat}: {imp:.4f}")
    
    # ============ XGBoost ============
    # XGBoost training and evaluation:
    # Gradient boosting of decision trees, 
    # often stronger but more sensitive to tuning.
    # ==========================

    print(f"\n--- XGBoost ---")
    xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb.fit(X_train, y_train)
    
    # Predict on test set
    xgb_pred = xgb.predict(X_test)
    
    # Compute evaluation metrics
    results['xgb'] = {
        'accuracy': accuracy_score(y_test, xgb_pred),
        'precision': precision_score(y_test, xgb_pred, zero_division=0),
        'recall': recall_score(y_test, xgb_pred, zero_division=0),
        'f1': f1_score(y_test, xgb_pred, zero_division=0),
    }
    
    # Print evaluation results
    print(f"Accuracy:  {results['xgb']['accuracy']:.4f}")
    print(f"Precision: {results['xgb']['precision']:.4f}")
    print(f"Recall:    {results['xgb']['recall']:.4f}")
    print(f"F1-Score:  {results['xgb']['f1']:.4f}")
    
    # Save model
    joblib.dump(xgb, f"models/{pipeline_name}_xgb.pkl")
    
    return results

# ===========================
# Main experiment runner:
# Loads feature CSV files for each pipeline, splits them into train/test,
# trains models, and prints a final comparison summary.
# ===========================

def main():
    print("="*60)
    print("ELF Malware Detection - Model Training & Comparison")
    print("="*60)
    
    # Collect results from all pipelines for final comparison
    all_results = {}
    
    # ===========================
    # Pipeline A: Structural ELF features
    # ===========================
    if os.path.exists("features_pipeline_a.csv"):
        df_a = pd.read_csv("features_pipeline_a.csv")
        X_a = df_a.drop("label", axis=1)
        y_a = df_a["label"]

        # Stratified split keeps class balance in both train and test sets
        X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
            X_a, y_a, test_size=0.2, random_state=42, stratify=y_a
        )
        all_results['Pipeline A (Structural)'] = train_and_evaluate(
            X_train_a, X_test_a, y_train_a, y_test_a, "pipeline_a"
        )
    else:
        print("Warning: features_pipeline_a.csv not found")
    
    # ===========================
    # Pipeline B: Statistical features
    # ===========================
    if os.path.exists("features_pipeline_b.csv"):
        df_b = pd.read_csv("features_pipeline_b.csv")
        X_b = df_b.drop("label", axis=1)
        y_b = df_b["label"]
        X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
            X_b, y_b, test_size=0.2, random_state=42, stratify=y_b
        )
        all_results['Pipeline B (Statistical)'] = train_and_evaluate(
            X_train_b, X_test_b, y_train_b, y_test_b, "pipeline_b"
        )
    else:
        print("Warning: features_pipeline_b.csv not found")
    
    # ===========================
    # Pipeline C: Hybrid features (structural + statistical + domain/security)
    # ===========================
    if os.path.exists("features_pipeline_c.csv"):
        df_c = pd.read_csv("features_pipeline_c.csv")
        X_c = df_c.drop("label", axis=1)
        y_c = df_c["label"]
        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
            X_c, y_c, test_size=0.2, random_state=42, stratify=y_c
        )
        all_results['Pipeline C (Hybrid)'] = train_and_evaluate(
            X_train_c, X_test_c, y_train_c, y_test_c, "pipeline_c"
        )
    else:
        print("Warning: features_pipeline_c.csv not found")
    
    # ===========================
    # Print final comparison table across all pipelines and models.
    # ===========================
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    
    # Create comparison table
    print(f"\n{'Pipeline':<30} {'Model':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 80)
    
    for pipeline, results in all_results.items():
        for model, metrics in results.items():
            print(f"{pipeline:<30} {model.upper():<10} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1']:<10.4f}")
    
    # ===========================
    # Select the best configuration based on highest F1-score.
    # ==========================
    print("\n" + "-" * 80)
    best_f1 = 0
    best_config = ""
    for pipeline, results in all_results.items():
        for model, metrics in results.items():
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                best_config = f"{pipeline} + {model.upper()}"
    
    print(f"Best configuration: {best_config} (F1: {best_f1:.4f})")
    
    # Save comparison to CSV
    comparison_rows = []
    for pipeline, results in all_results.items():
        for model, metrics in results.items():
            comparison_rows.append({
                'pipeline': pipeline,
                'model': model,
                **metrics
            })
    
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv("model_comparison.csv", index=False)
    print("\nComparison saved to model_comparison.csv")
    
    # ===========================
    # List all saved model files.
    # ===========================
    print("\nModels saved to models/ directory:")
    for f in os.listdir("models"):
        print(f"  - {f}")


if __name__ == "__main__":
    main()
