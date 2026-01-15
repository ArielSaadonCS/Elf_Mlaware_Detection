"""
Train and Compare Models for all three pipelines

This module trains Random Forest and XGBoost classifiers for three feature
pipelines (A - structural features, B - statistical features, C - hybrid features).
It evaluates each trained model on a held-out test set, prints metrics,
saves models to disk, and writes a comparison CSV.

Notes:
    - Requires features CSV files to exist for the pipelines (features_pipeline_a.csv, etc.)
    - Saves trained model files under the ./models directory and writes model_comparison.csv.
"""
import pandas as pd                       # pandas: data loading, DataFrame manipulation
import numpy as np                        # numpy: numeric operations (not heavily used here but commonly available)
import joblib                             # joblib: save/load scikit-learn compatible models
import os                                 # os: filesystem operations (create directories, list files)

# scikit-learn imports for classifiers, model selection and metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# Note: GradientBoostingClassifier is imported but not used in this script; it could be used later.
from sklearn.model_selection import train_test_split, cross_val_score
# train_test_split: split data into train/test sets
# cross_val_score: (imported) for cross-validation if desired (not used below)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
# accuracy_score, precision_score, recall_score, f1_score: standard classification metrics
# classification_report: textual summary of precision/recall/f1 (imported but not used below)

# XGBoost classifier import (external dependency: xgboost)
from xgboost import XGBClassifier         # XGBClassifier: gradient-boosted trees implementation

# Create models directory
# ===========================
# Ensure that a directory exists for saving trained models.
# This allows the training script to persist models for later inference by the worker.
# We use exist_ok=True to avoid raising if the directory already exists.
os.makedirs("models", exist_ok=True)


def train_and_evaluate(X_train, X_test, y_train, y_test, pipeline_name):
    """
    Train Random Forest and XGBoost classifiers on the provided training set, evaluate them on the test set,
    print metrics, save trained models to disk, and return a dictionary of evaluation results.

    Parameters
    ----------
    X_train : pandas.DataFrame or numpy.ndarray
        Feature matrix used for training. If DataFrame is provided, column names will be used for feature importances.
    X_test : pandas.DataFrame or numpy.ndarray
        Feature matrix used for evaluation (held-out).
    y_train : pandas.Series or numpy.ndarray
        Target labels corresponding to X_train.
    y_test : pandas.Series or numpy.ndarray
        Target labels corresponding to X_test.
    pipeline_name : str
        Short name used for saving model files (e.g., 'pipeline_a'). Also used in printing logs.

    Returns
    -------
    results : dict
        Dictionary with keys 'rf' and 'xgb', each mapping to a dict with metrics:
        {'accuracy', 'precision', 'recall', 'f1'}.
    """
    # Print basic information about the current training run
    # Provide a visual separator and the pipeline name for clarity in console logs
    print(f"\n{'='*50}")
    print(f"Training {pipeline_name}")
    print(f"{'='*50}")
    # Print number of features and number of train/test samples to help sanity-check shapes
    print(f"Features: {X_train.shape[1]}, Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Dictionary for collecting evaluation metrics of both models
    results = {}

    # ============ Random Forest ============
    # Random Forest training and evaluation:
    # Uses many decision trees trained on random feature subsets to reduce overfitting.
    # ===========================
    print(f"\n--- Random Forest ---")
    # Instantiate RandomForestClassifier with the chosen hyperparameters:
    # n_estimators=300: number of trees in the forest (more trees generally increase stability but cost more compute)
    # random_state=42: seed for reproducibility
    # n_jobs=-1: use all available CPU cores for parallel training
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    # Fit Random Forest to the training data
    rf.fit(X_train, y_train)
    
    # Predict class labels on the test set using the fitted Random Forest
    rf_pred = rf.predict(X_test)
    # Predict class probabilities on the test set (useful if probability thresholds or ROC analysis required)
    rf_proba = rf.predict_proba(X_test)
    # Note: rf_proba is computed and available for downstream analysis; not used for metrics printed below.

    # Compute standard classification metrics using true labels and predicted labels
    # zero_division=0 ensures that if a precision/recall calculation would divide by zero, it returns 0 instead of raising
    results['rf'] = {
        'accuracy': accuracy_score(y_test, rf_pred),
        'precision': precision_score(y_test, rf_pred, zero_division=0),
        'recall': recall_score(y_test, rf_pred, zero_division=0),
        'f1': f1_score(y_test, rf_pred, zero_division=0),
    }
    
    # Print evaluation results for Random Forest with formatting to 4 decimal places
    print(f"Accuracy:  {results['rf']['accuracy']:.4f}")
    print(f"Precision: {results['rf']['precision']:.4f}")
    print(f"Recall:    {results['rf']['recall']:.4f}")
    print(f"F1-Score:  {results['rf']['f1']:.4f}")
    
    # Save the trained Random Forest model to the models directory using joblib for efficient serialization
    joblib.dump(rf, f"models/{pipeline_name}_rf.pkl")
    
    # ===========================
    # Display top-10 most important features according to Random Forest.
    # Helps analyze which features contribute most to the decision.
    # ===========================
    # Check whether X_train provides column names (i.e., is a pandas DataFrame)
    if hasattr(X_train, 'columns'):
        # Create a pandas Series mapping feature names to importance values
        importances = pd.Series(rf.feature_importances_, index=X_train.columns)
        # Print heading for feature importances
        print(f"\nTop 10 important features:")
        # Loop through the top 10 features sorted by importance (largest first)
        for feat, imp in importances.nlargest(10).items():
            # Print each feature name and its importance value to 4 decimal places
            print(f"  {feat}: {imp:.4f}")
    
    # ============ XGBoost ============
    # XGBoost training and evaluation:
    # Gradient boosting of decision trees, often stronger but more sensitive to tuning.
    # ==========================
    print(f"\n--- XGBoost ---")
    # Instantiate XGBClassifier with typical defaults tuned for moderate complexity:
    # n_estimators=300: number of boosting rounds
    # learning_rate=0.1: step size shrinkage used to prevent overfitting
    # max_depth=6: max tree depth per boosting round
    # random_state=42: seed for reproducibility
    # use_label_encoder=False: avoid label encoder warnings for new versions of xgboost
    # eval_metric='logloss': specify evaluation metric to silence warnings and define objective
    xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    # Fit XGBoost to the training data
    xgb.fit(X_train, y_train)
    
    # Predict class labels on the test set using the fitted XGBoost model
    xgb_pred = xgb.predict(X_test)
    
    # Compute evaluation metrics for XGBoost using predicted labels
    results['xgb'] = {
        'accuracy': accuracy_score(y_test, xgb_pred),
        'precision': precision_score(y_test, xgb_pred, zero_division=0),
        'recall': recall_score(y_test, xgb_pred, zero_division=0),
        'f1': f1_score(y_test, xgb_pred, zero_division=0),
    }
    
    # Print evaluation results for XGBoost
    print(f"Accuracy:  {results['xgb']['accuracy']:.4f}")
    print(f"Precision: {results['xgb']['precision']:.4f}")
    print(f"Recall:    {results['xgb']['recall']:.4f}")
    print(f"F1-Score:  {results['xgb']['f1']:.4f}")
    
    # Save the trained XGBoost model to disk using joblib
    joblib.dump(xgb, f"models/{pipeline_name}_xgb.pkl")
    
    # Return a dictionary containing the evaluation metrics for both models
    return results


def main():
    """
    Main experiment runner.

    Purpose:
        - Load feature CSV files for multiple pipelines (A, B, C).
        - Split each dataset into stratified training and test sets.
        - Call train_and_evaluate for each pipeline to train models and collect metrics.
        - Print a summary comparison table and identify the best configuration by F1-score.
        - Save the comparison table and list the saved model files.

    No parameters.
    """
    # Print header for the script run
    print("="*60)
    print("ELF Malware Detection - Model Training & Comparison")
    print("="*60)
    
    # Collect results from all pipelines for final comparison in a dictionary
    all_results = {}
    
    # ===========================
    # Pipeline A: Structural ELF features
    # ===========================
    # Check whether the features CSV file for pipeline A exists in the working directory
    if os.path.exists("features_pipeline_a.csv"):
        # Load CSV into a pandas DataFrame
        df_a = pd.read_csv("features_pipeline_a.csv")
        # Prepare feature matrix X_a by dropping the 'label' column
        X_a = df_a.drop("label", axis=1)
        # Prepare target vector y_a as the 'label' column
        y_a = df_a["label"]

        # Stratified split keeps class balance in both train and test sets
        X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
            X_a, y_a, test_size=0.2, random_state=42, stratify=y_a
        )
        # Train models and evaluate; store results under a descriptive name
        all_results['Pipeline A (Structural)'] = train_and_evaluate(
            X_train_a, X_test_a, y_train_a, y_test_a, "pipeline_a"
        )
    else:
        # Warn the user if the file is missing so they know which pipelines didn't run
        print("Warning: features_pipeline_a.csv not found")
    
    # ===========================
    # Pipeline B: Statistical features
    # ===========================
    # Repeat the same pattern as pipeline A for pipeline B
    if os.path.exists("features_pipeline_b.csv"):
        # Load features for pipeline B
        df_b = pd.read_csv("features_pipeline_b.csv")
        # Separate features and labels
        X_b = df_b.drop("label", axis=1)
        y_b = df_b["label"]
        # Perform stratified train/test split
        X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
            X_b, y_b, test_size=0.2, random_state=42, stratify=y_b
        )
        # Train and evaluate; add results to collection
        all_results['Pipeline B (Statistical)'] = train_and_evaluate(
            X_train_b, X_test_b, y_train_b, y_test_b, "pipeline_b"
        )
    else:
        # Warn when dataset not found
        print("Warning: features_pipeline_b.csv not found")
    
    # ===========================
    # Pipeline C: Hybrid features (structural + statistical + domain/security)
    # ===========================
    # Repeat the same pattern for pipeline C
    if os.path.exists("features_pipeline_c.csv"):
        # Load features for pipeline C
        df_c = pd.read_csv("features_pipeline_c.csv")
        # Separate features and labels
        X_c = df_c.drop("label", axis=1)
        y_c = df_c["label"]
        # Perform stratified train/test split
        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
            X_c, y_c, test_size=0.2, random_state=42, stratify=y_c
        )
        # Train and evaluate models for pipeline C
        all_results['Pipeline C (Hybrid)'] = train_and_evaluate(
            X_train_c, X_test_c, y_train_c, y_test_c, "pipeline_c"
        )
    else:
        # Warn when dataset not found
        print("Warning: features_pipeline_c.csv not found")
    
    # ===========================
    # Print final comparison table across all pipelines and models.
    # ===========================
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    
    # Print table header for human-readable summary comparing pipelines and models
    print(f"\n{'Pipeline':<30} {'Model':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 80)
    
    # Iterate over all collected results to print rows of the comparison table
    for pipeline, results in all_results.items():
        # Each 'results' is a dict mapping model keys ('rf', 'xgb') to metric dictionaries
        for model, metrics in results.items():
            # Print pipeline name, model name (uppercase), and formatted metric values
            print(f"{pipeline:<30} {model.upper():<10} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1']:<10.4f}")
    
    # ===========================
    # Select the best configuration based on highest F1-score.
    # ==========================
    print("\n" + "-" * 80)
    # Initialize variables used to track the best model
    best_f1 = 0
    best_config = ""
    # Loop through all results to find the entry with the highest F1-score
    for pipeline, results in all_results.items():
        for model, metrics in results.items():
            # If current model's F1 is higher than the stored best, update the best trackers
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                best_config = f"{pipeline} + {model.upper()}"
    
    # Print the configuration with the highest F1-score
    print(f"Best configuration: {best_config} (F1: {best_f1:.4f})")
    
    # Save comparison to CSV
    comparison_rows = []
    # Build a list of rows (dicts) representing the comparison table
    for pipeline, results in all_results.items():
        for model, metrics in results.items():
            # Each row contains pipeline name, model key, and metric values
            comparison_rows.append({
                'pipeline': pipeline,
                'model': model,
                **metrics
            })
    
    # Convert rows to a pandas DataFrame for easy CSV export
    comparison_df = pd.DataFrame(comparison_rows)
    # Write the DataFrame to a CSV file; index=False avoids writing row indices into the file
    comparison_df.to_csv("model_comparison.csv", index=False)
    # Inform the user that the comparison was saved
    print("\nComparison saved to model_comparison.csv")
    
    # ===========================
    # List all saved model files.
    # ===========================
    print("\nModels saved to models/ directory:")
    # List filenames in the models directory and print them; this shows the user what was saved
    for f in os.listdir("models"):
        print(f"  - {f}")


# If this script is executed as the main program, run the main() function.
# This check prevents main() from running if the module is imported elsewhere.
if __name__ == "__main__":
    main()
