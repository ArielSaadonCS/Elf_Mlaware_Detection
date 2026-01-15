"""
Overfitting Detection Script
Uses K-Fold Cross-Validation to verify model performance
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

def check_overfitting():
    print("=" * 60)
    print("OVERFITTING DETECTION ANALYSIS")
    print("=" * 60)
    
    results = {}
    
    for pipeline in ['a', 'b', 'c']:
        csv_file = f"features_pipeline_{pipeline}.csv"
        try:
            df = pd.read_csv(csv_file)
        except:
            print(f"File {csv_file} not found, skipping...")
            continue
            
        X = df.drop("label", axis=1)
        y = df["label"]
        
        print(f"\n{'='*60}")
        print(f"Pipeline {pipeline.upper()}: {len(X)} samples, {X.shape[1]} features")
        print(f"{'='*60}")
        
        # K-Fold Cross-Validation (K=10)
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        
        # Random Forest
        rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        rf_scores = cross_val_score(rf, X, y, cv=kfold, scoring='f1')
        
        print(f"\n--- Random Forest (10-Fold CV) ---")
        print(f"F1 Scores per fold: {[f'{s:.4f}' for s in rf_scores]}")
        print(f"Mean F1:  {rf_scores.mean():.4f}")
        print(f"Std F1:   {rf_scores.std():.4f}")
        print(f"Min F1:   {rf_scores.min():.4f}")
        print(f"Max F1:   {rf_scores.max():.4f}")
        
        # XGBoost
        xgb = XGBClassifier(n_estimators=300, random_state=42, eval_metric='logloss', verbosity=0)
        xgb_scores = cross_val_score(xgb, X, y, cv=kfold, scoring='f1')
        
        print(f"\n--- XGBoost (10-Fold CV) ---")
        print(f"F1 Scores per fold: {[f'{s:.4f}' for s in xgb_scores]}")
        print(f"Mean F1:  {xgb_scores.mean():.4f}")
        print(f"Std F1:   {xgb_scores.std():.4f}")
        print(f"Min F1:   {xgb_scores.min():.4f}")
        print(f"Max F1:   {xgb_scores.max():.4f}")
        
        results[pipeline] = {
            'rf_mean': rf_scores.mean(),
            'rf_std': rf_scores.std(),
            'xgb_mean': xgb_scores.mean(),
            'xgb_std': xgb_scores.std(),
        }
        
        # Train vs Test comparison (overfitting indicator)
        print(f"\n--- Train vs Test Score Comparison ---")
        rf.fit(X, y)
        train_score = rf.score(X, y)
        cv_mean = rf_scores.mean()
        gap = train_score - cv_mean
        
        print(f"Train Accuracy: {train_score:.4f}")
        print(f"CV Mean F1:     {cv_mean:.4f}")
        print(f"Gap:            {gap:.4f}")
        
        if gap > 0.05:
            print(f"⚠️  WARNING: Gap > 5% - Possible overfitting!")
        elif gap > 0.02:
            print(f"⚡ NOTICE: Small gap detected")
        else:
            print(f"✅ OK: No significant overfitting detected")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY - Cross-Validation Results")
    print(f"{'='*60}")
    print(f"\n{'Pipeline':<12} {'Model':<8} {'Mean F1':<10} {'Std':<10} {'Status'}")
    print("-" * 50)
    
    for pipeline, res in results.items():
        # RF
        status_rf = "✅" if res['rf_std'] < 0.03 else "⚠️"
        print(f"{pipeline.upper():<12} {'RF':<8} {res['rf_mean']:<10.4f} {res['rf_std']:<10.4f} {status_rf}")
        
        # XGB
        status_xgb = "✅" if res['xgb_std'] < 0.03 else "⚠️"
        print(f"{'':<12} {'XGB':<8} {res['xgb_mean']:<10.4f} {res['xgb_std']:<10.4f} {status_xgb}")
    
    print(f"\n{'='*60}")
    print("INTERPRETATION GUIDE")
    print(f"{'='*60}")
    print("""
    ✅ Good signs (NO overfitting):
       - Low Std (< 0.03): Consistent across folds
       - Mean CV ≈ Train Score: Generalizes well
       
    ⚠️  Warning signs (Possible overfitting):
       - High Std (> 0.05): Inconsistent performance
       - Train Score >> CV Score: Memorizing, not learning
       - Perfect 100% on train, much lower on CV
    
    📊 Your results interpretation:
       - If Mean F1 > 0.95 with Std < 0.03: REAL performance
       - If Mean F1 > 0.95 with Std > 0.05: Might be overfitting
       - Compare with original train/test split results
    """)


def plot_learning_curves():
    """Plot learning curves to visualize overfitting"""
    print("\nGenerating learning curves...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, pipeline in enumerate(['a', 'b', 'c']):
        csv_file = f"features_pipeline_{pipeline}.csv"
        try:
            df = pd.read_csv(csv_file)
        except:
            continue
            
        X = df.drop("label", axis=1)
        y = df["label"]
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        
        train_sizes, train_scores, test_scores = learning_curve(
            rf, X, y, 
            cv=5, 
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='f1'
        )
        
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        test_mean = test_scores.mean(axis=1)
        test_std = test_scores.std(axis=1)
        
        ax = axes[idx]
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='orange')
        ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
        ax.plot(train_sizes, test_mean, 'o-', color='orange', label='Cross-validation score')
        ax.set_xlabel('Training examples')
        ax.set_ylabel('F1 Score')
        ax.set_title(f'Pipeline {pipeline.upper()} - Learning Curve')
        ax.legend(loc='lower right')
        ax.grid(True)
        ax.set_ylim([0.8, 1.05])
    
    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=150)
    print("✅ Saved: learning_curves.png")
    print("\nInterpretation:")
    print("  - If train and CV curves converge: NO overfitting")
    print("  - If large gap between curves: OVERFITTING")
    print("  - If both curves are low: UNDERFITTING")


if __name__ == "__main__":
    check_overfitting()
    
    try:
        plot_learning_curves()
    except Exception as e:
        print(f"\nCouldn't generate plots: {e}")
        print("Install matplotlib: pip install matplotlib")
