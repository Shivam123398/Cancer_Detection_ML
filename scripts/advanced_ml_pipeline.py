"""
CANCER DETECTION ML PIPELINE
8 Models Comparison - Updated for 5 Healthy Samples
"""

import os
import glob
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve, auc,
                             precision_recall_curve, f1_score, precision_score, recall_score,
                             accuracy_score, roc_auc_score)
import joblib
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except:
    CATBOOST_AVAILABLE = False

try:
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except:
    TF_AVAILABLE = False

# ============================================
# PATHS
# ============================================


# ============================================
# PATHS
# ============================================

# Project root: parent of the scripts folder
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "raw_data"
MODEL_DIR = BASE_DIR / "models"
PLOT_DIR = BASE_DIR / "results" / "plots"
REPORT_DIR = BASE_DIR / "results" / "reports"

for directory in [MODEL_DIR, PLOT_DIR, REPORT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("\n" + "="*80)
print("CANCER DETECTION ML PIPELINE")
print("8 Models Comparison")
print("="*80 + "\n")

# ============================================
# DATA LOADING
# ============================================

def read_file(filepath):
    freq, mag = [], []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                parts = line.split()
                if len(parts) >= 2:
                    freq.append(float(parts[0]))
                    mag.append(float(parts[1]))
            except:
                continue
    return np.array(freq), np.array(mag)

def get_label(filename):
    fn = filename.lower()
    if 'notumor' in fn or 'no_tumor' in fn:
        return 0, 'Healthy'
    elif '15mm' in fn:
        return 1, 'Cancer_15mm'
    elif '10mm' in fn:
        return 1, 'Cancer_10mm'
    elif '5mm' in fn:
        return 1, 'Cancer_5mm'
    return None, 'Unknown'

def extract_features(frequency, magnitude):
    if len(magnitude) == 0:
        return None
    max_mag = np.max(magnitude)
    idx = np.where(magnitude < max_mag - 10)[0]
    bw_10db = 0
    if len(idx) > 1:
        bw_10db = frequency[idx[-1]] - frequency[idx[0]]
    idx_6db = np.where(magnitude < max_mag - 6)[0]
    bw_6db = 0
    if len(idx_6db) > 1:
        bw_6db = frequency[idx_6db[-1]] - frequency[idx_6db[0]]
    slope = np.polyfit(frequency, magnitude, 1)[0]
    return {
        'min_s11': np.min(magnitude),
        'max_s11': np.max(magnitude),
        'mean_s11': np.mean(magnitude),
        'std_s11': np.std(magnitude),
        'median_s11': np.median(magnitude),
        'resonant_freq': frequency[np.argmin(magnitude)],
        'bandwidth_10db': bw_10db,
        'bandwidth_6db': bw_6db,
        'energy': np.sum(magnitude**2),
        'dynamic_range': np.max(magnitude) - np.min(magnitude),
    }

def load_data():
    print("Loading data...\n")
    features, labels, categories = [], [], []
    files = glob.glob(os.path.join(DATA_DIR, '*.txt'))
    
    for i, fp in enumerate(sorted(files), 1):
        fn = os.path.basename(fp)
        label, cat = get_label(fn)
        if label is None:
            continue
        freq, mag = read_file(fp)
        if len(freq) == 0:
            continue
        feats = extract_features(freq, mag)
        if feats is None:
            continue
        features.append(feats)
        labels.append(label)
        categories.append(cat)
        print(f"  [{i:2d}] ✓ {fn:30s} → {cat}")
    
    df = pd.DataFrame(features)
    df['label'] = labels
    df['category'] = categories
    
    print(f"\n✓ Loaded {len(df)} files")
    print(f"   Healthy: {(df['label']==0).sum()}")
    print(f"   Cancer:  {(df['label']==1).sum()}\n")
    return df

def prepare_data(df):
    print("Preparing data...\n")
    cols = [c for c in df.columns if c not in ['label', 'category']]
    X, y = df[cols].values, df['label'].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {len(X_train)} ({sum(y_train==0)}H/{sum(y_train==1)}C)")
    print(f"   Test:  {len(X_test)} ({sum(y_test==0)}H/{sum(y_test==1)}C)\n")
    return X_train, X_test, y_train, y_test, scaler, cols

def build_models():
    print("Building models...")
    models = {}
    models['Logistic Regression'] = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    models['KNN'] = KNeighborsClassifier(n_neighbors=5)
    models['SVM'] = SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced')
    models['Decision Tree'] = DecisionTreeClassifier(max_depth=10, random_state=42)
    models['Random Forest'] = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
    if XGB_AVAILABLE:
        models['XGBoost'] = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, eval_metric='logloss')
    if CATBOOST_AVAILABLE:
        models['CatBoost'] = cb.CatBoostClassifier(iterations=100, depth=5, learning_rate=0.1, random_state=42, verbose=False)
    print(f"✓ Built {len(models)} models\n")
    return models

def build_ann(input_dim):
    if not TF_AVAILABLE:
        return None
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_dim=input_dim),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', keras.metrics.AUC()])
    return model

def train_models(models, X_train, X_test, y_train, y_test):
    print("TRAINING MODELS:\n")
    print(f"{'Model':<25} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'AUC':<8} {'T(s)':<8}")
    print("─"*90)
    results = {}
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        train_t = time.time() - t0
        t0 = time.time()
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)
        y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min() + 1e-10)
        pred_t = time.time() - t0
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        try:
            auc_val = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 1.0
        except:
            auc_val = 1.0
        print(f"{name:<25} {acc:<8.4f} {prec:<8.4f} {rec:<8.4f} {f1:<8.4f} {auc_val:<8.4f} {train_t:<8.4f}")
        results[name] = {
            'model': model, 'accuracy': acc, 'precision': prec, 'recall': rec,
            'specificity': spec, 'f1_score': f1, 'roc_auc': auc_val,
            'confusion_matrix': cm, 'train_time': train_t, 'pred_time': pred_t,
            'y_pred': y_pred, 'y_pred_proba': y_pred_proba
        }
    return results

def train_ann(model, X_train, X_test, y_train, y_test):
    if model is None:
        return None
    print("🧠 Artificial Neural Network")
    print("─"*90)
    t0 = time.time()
    model.fit(X_train, y_train, epochs=50, batch_size=4, validation_split=0.2, verbose=0)
    train_t = time.time() - t0
    t0 = time.time()
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    pred_t = time.time() - t0
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    try:
        auc_val = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 1.0
    except:
        auc_val = 1.0
    print(f"   ✓ Accuracy:        {acc:.4f} ({acc*100:.2f}%)")
    print(f"   ✓ Precision:       {prec:.4f}")
    print(f"   ✓ Recall:          {rec:.4f}")
    print(f"   ✓ Specificity:     {spec:.4f}")
    print(f"   ✓ F1-Score:        {f1:.4f}")
    print(f"   ✓ ROC-AUC:         {auc_val:.4f}")
    print(f"   ⏱️  Training Time:   {train_t:.4f}s")
    print(f"   ⏱️  Prediction Time: {pred_t:.4f}s\n")
    return {
        'model': model, 'accuracy': acc, 'precision': prec, 'recall': rec,
        'specificity': spec, 'f1_score': f1, 'roc_auc': auc_val,
        'confusion_matrix': cm, 'train_time': train_t, 'pred_time': pred_t,
        'y_pred': y_pred, 'y_pred_proba': y_pred_proba
    }

def cross_validate(models, X, y):
    print("\nCROSS-VALIDATION (5-FOLD):\n")
    print(f"{'Model':<25} {'Acc':<10} {'F1':<10} {'AUC':<10}")
    print("─"*70)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    for name, model in models.items():
        try:
            acc = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            f1 = cross_val_score(model, X, y, cv=cv, scoring='f1')
            roc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
            cv_results[name] = {'acc_mean': acc.mean(), 'acc_std': acc.std(),
                                'f1_mean': f1.mean(), 'f1_std': f1.std(),
                                'roc_mean': roc.mean(), 'roc_std': roc.std()}
            print(f"{name:<25} {acc.mean():<10.4f} ±{acc.std():<8.4f} {f1.mean():<10.4f} ±{f1.std():<8.4f} {roc.mean():<10.4f} ±{roc.std():<8.4f}")
        except Exception as e:
            print(f"{name:<25} FAILED: {e}")
    print()
    return cv_results

def create_plots(results, df, cv_results, cols):
    print("\nCREATING VISUALIZATIONS...\n")
    plots_dir = PLOT_DIR
    
    # 1. Accuracy Comparison
    plt.figure(figsize=(12, 6))
    models = df['Model'].values
    accs = df['Accuracy'].astype(float).values
    colors = ['#2ecc71' if a > 0.9 else '#f39c12' if a > 0.8 else '#e74c3c' for a in accs]
    bars = plt.bar(models, accs, color=colors, edgecolor='black', linewidth=2, alpha=0.85)
    for bar, a in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width()/2, a + 0.01, f'{a:.2%}', ha='center', fontweight='bold')
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.title('📊 Model Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.ylim([0, 1.1])
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '01_Accuracy_Comparison.png'), dpi=300)
    plt.close()
    print("✓ Saved: 01_Accuracy_Comparison.png")
    
    # 2. ROC-AUC Comparison
    plt.figure(figsize=(12, 6))
    aucs = df['ROC-AUC'].astype(float).values
    aucs = np.where(np.isnan(aucs), 1.0, aucs)
    colors = ['#2ecc71' if a > 0.9 else '#f39c12' if a > 0.8 else '#e74c3c' for a in aucs]
    bars = plt.bar(models, aucs, color=colors, edgecolor='black', linewidth=2, alpha=0.85)
    for bar, a in zip(bars, aucs):
        plt.text(bar.get_x() + bar.get_width()/2, a + 0.01, f'{a:.4f}', ha='center', fontweight='bold')
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel('ROC-AUC', fontsize=12, fontweight='bold')
    plt.title('🔍 ROC-AUC Comparison', fontsize=14, fontweight='bold')
    plt.ylim([0, 1.1])
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '02_ROC_AUC_Comparison.png'), dpi=300)
    plt.close()
    print("✓ Saved: 02_ROC_AUC_Comparison.png")
    
    # 3. F1-Score Comparison
    plt.figure(figsize=(12, 6))
    f1s = df['F1-Score'].astype(float).values
    colors = ['#3498db' if f > 0.9 else '#9b59b6' if f > 0.8 else '#e67e22' for f in f1s]
    bars = plt.bar(models, f1s, color=colors, edgecolor='black', linewidth=2, alpha=0.85)
    for bar, f in zip(bars, f1s):
        plt.text(bar.get_x() + bar.get_width()/2, f + 0.01, f'{f:.4f}', ha='center', fontweight='bold')
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel('F1-Score', fontsize=12, fontweight='bold')
    plt.title('📈 F1-Score Comparison', fontsize=14, fontweight='bold')
    plt.ylim([0, 1.1])
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '03_F1_Score_Comparison.png'), dpi=300)
    plt.close()
    print("✓ Saved: 03_F1_Score_Comparison.png")
    
    # 4. Confusion Matrix (Best Model) - FIXED VERSION
    best_idx = df['Accuracy'].astype(float).idxmax()
    best_name = df.loc[best_idx, 'Model']
    cm = results[best_name]['confusion_matrix']
    
    plt.figure(figsize=(8, 6))
    if cm.ndim == 1:
        cm_disp = np.array([[cm[0]]])
    else:
        cm_disp = cm
    
    plt.imshow(cm_disp, cmap='Blues', aspect='auto')
    
    # FIXED: Proper label handling
    if cm_disp.shape == (1, 1):
        ax = plt.gca()
        ax.set_xticks([0])
        ax.set_yticks([0])
        ax.set_xticklabels(['Cancer'])
        ax.set_yticklabels(['Cancer'])
    else:
        ax = plt.gca()
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Healthy', 'Cancer'])
        ax.set_yticklabels(['Healthy', 'Cancer'])
    
    plt.xlabel('Predicted', fontsize=12, fontweight='bold')
    plt.ylabel('Actual', fontsize=12, fontweight='bold')
    plt.title(f'🏆 Best Model: {best_name}', fontsize=14, fontweight='bold')
    
    for i in range(cm_disp.shape[0]):
        for j in range(cm_disp.shape[1]):
            val = cm_disp[i, j]
            max_val = cm_disp.max()
            text_color = "white" if val > max_val/2 else "black"
            plt.text(j, i, int(val), ha="center", va="center", color=text_color, fontsize=18, fontweight='bold')
    
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '04_Best_Confusion_Matrix.png'), dpi=300)
    plt.close()
    print("✓ Saved: 04_Best_Confusion_Matrix.png")
    
    # 5. Feature Importance
    tree_models = {}
    for name in ['Random Forest', 'Decision Tree', 'XGBoost', 'CatBoost']:
        if name in results and hasattr(results[name]['model'], 'feature_importances_'):
            tree_models[name] = results[name]['model']
    
    if tree_models:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        for idx, (name, model) in enumerate(tree_models.items()):
            ax = axes[idx]
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            colors_f = ['#1f77b4' if i < 5 else '#ff7f0e' for i in range(len(importances))]
            ax.bar(range(len(importances)), importances[indices], 
                   color=[colors_f[indices[i]] for i in range(len(importances))], 
                   edgecolor='navy', linewidth=1.5, alpha=0.8)
            ax.set_xticks(range(len(importances)))
            ax.set_xticklabels([cols[i] for i in indices], rotation=45, ha='right')
            ax.set_ylabel('Importance', fontsize=11, fontweight='bold')
            ax.set_title(f'{name}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        for idx in range(len(tree_models), 4):
            axes[idx].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, '05_Feature_Importance.png'), dpi=300)
        plt.close()
        print("✓ Saved: 05_Feature_Importance.png")
    
    print("✅ All visualizations created!\n")

def save_model(results, df, scaler, cols):
    print("\nSaving best model...")
    best_idx = df['Accuracy'].astype(float).idxmax()
    best_name = df.loc[best_idx, 'Model']
    print(f"✓ Best Model: {best_name}")
    print(f"✓ Accuracy: {df.loc[best_idx, 'Accuracy']:.4f}")
    print(f"✓ F1-Score: {df.loc[best_idx, 'F1-Score']:.4f}")
    print(f"✓ ROC-AUC:  {df.loc[best_idx, 'ROC-AUC']:.4f}")
    
    joblib.dump(results[best_name]['model'], os.path.join(MODEL_DIR, 'best_cancer_model.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    joblib.dump(cols, os.path.join(MODEL_DIR, 'features.pkl'))
    
    df.to_csv(os.path.join(REPORT_DIR, 'Model_Comparison.csv'), index=False)
    print(f"✓ Saved model, scaler, features, and report\n")

def main():
    print("\n" + "="*80)
    print("CANCER DETECTION — 8 MODEL COMPARISON")
    print("="*80 + "\n")
    
    df = load_data()
    if df is None:
        print("❌ Failed to load data!")
        return
    
    X_train, X_test, y_train, y_test, scaler, cols = prepare_data(df)
    
    models_dict = build_models()
    
    results = train_models(models_dict, X_train, X_test, y_train, y_test)
    
    if TF_AVAILABLE:
        ann = build_ann(X_train.shape[1])
        ann_result = train_ann(ann, X_train, X_test, y_train, y_test)
        if ann_result:
            results['Artificial Neural Network'] = ann_result
    
    cv_results = cross_validate(models_dict, np.vstack([X_train, X_test]), np.concatenate([y_train, y_test]))
    
    df_comp = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[m]['accuracy'] for m in results.keys()],
        'Precision': [results[m]['precision'] for m in results.keys()],
        'Recall': [results[m]['recall'] for m in results.keys()],
        'Specificity': [results[m]['specificity'] for m in results.keys()],
        'F1-Score': [results[m]['f1_score'] for m in results.keys()],
        'ROC-AUC': [results[m]['roc_auc'] for m in results.keys()],
        'Train Time (s)': [results[m]['train_time'] for m in results.keys()],
        'Pred Time (s)': [results[m]['pred_time'] for m in results.keys()],
    })
    df_comp.to_csv(os.path.join(REPORT_DIR, 'Model_Comparison.csv'), index=False)
    
    print("\n" + "="*80)
    print("MODEL COMPARISON TABLE:")
    print("="*80)
    print(df_comp.to_string(index=False))
    print("\n")
    
    print("="*80)
    print("CROSS-VALIDATION RESULTS:")
    print("="*80)
    if cv_results:
        for name, cv in cv_results.items():
            print(f"{name:<25} Acc: {cv['acc_mean']:.4f}±{cv['acc_std']:.4f} | F1: {cv['f1_mean']:.4f}±{cv['f1_std']:.4f} | AUC: {cv['roc_mean']:.4f}±{cv['roc_std']:.4f}")
    print("\n")
    
    create_plots(results, df_comp, cv_results, cols)
    
    save_model(results, df_comp, scaler, cols)
    
    print("="*80)
    print("✅ PIPELINE COMPLETE!")
    print("="*80)
    print(f"\n📊 Plots:  {PLOT_DIR}")
    print(f"📋 Reports: {REPORT_DIR}")
    print(f"🤖 Models:  {MODEL_DIR}\n")

if __name__ == "__main__":
    main()