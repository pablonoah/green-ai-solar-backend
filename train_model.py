"""
Green AI Solar - Script d'entraînement du modèle
================================================
Ce script entraîne le modèle GradientBoostingRegressor
et le sauvegarde pour utilisation par l'API
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# ============================================
# Configuration
# ============================================

# Colonnes numériques à utiliser
NUMERIC_FEATURES = [
    'temperature', 'irradiance', 'humidity', 'panel_age',
    'maintenance_count', 'soiling_ratio', 'voltage', 'current',
    'module_temperature', 'cloud_coverage', 'wind_speed', 'pressure'
]

TARGET = 'efficiency'
MODEL_DIR = 'model'
MODEL_PATH = os.path.join(MODEL_DIR, 'solar_efficiency_model.joblib')

# ============================================
# Fonctions de nettoyage
# ============================================

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Charge et nettoie les données
    """
    print(f"📂 Chargement des données depuis {filepath}...")
    df = pd.read_csv(filepath)
    print(f"   Taille initiale: {len(df)} lignes")
    print(f"   Colonnes: {list(df.columns)}")
    
    # Supprimer la colonne id si elle existe
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    
    # Supprimer les colonnes catégorielles
    cols_to_drop = ['string_id', 'error_code', 'installation_type']
    for col in cols_to_drop:
        if col in df.columns:
            print(f"   🗑️ Suppression de la colonne catégorielle: {col}")
            df = df.drop(col, axis=1)
    
    # Garder uniquement les colonnes nécessaires
    cols_to_keep = [col for col in NUMERIC_FEATURES + [TARGET] if col in df.columns]
    df = df[cols_to_keep].copy()
    print(f"   Colonnes conservées: {cols_to_keep}")
    
    # ========================================
    # CONVERSION EN NUMÉRIQUE (FIX IMPORTANT)
    # ========================================
    print("\n🔄 Conversion des colonnes en numérique...")
    for col in df.columns:
        # Convertir en numérique, les erreurs deviennent NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Compter les NaN après conversion
    nan_counts = df.isnull().sum()
    total_nans = nan_counts.sum()
    print(f"   Total de valeurs non-numériques converties en NaN: {total_nans}")
    
    if total_nans > 0:
        print("   Détail par colonne:")
        for col in df.columns:
            if nan_counts[col] > 0:
                print(f"      - {col}: {nan_counts[col]} NaN")
    
    # ========================================
    # Remplir les valeurs manquantes par la médiane
    # ========================================
    print("\n🔧 Remplissage des valeurs manquantes...")
    for col in df.columns:
        nan_count = df[col].isnull().sum()
        if nan_count > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"   ✓ {col}: {nan_count} valeurs remplies par la médiane ({median_val:.2f})")
    
    # ========================================
    # Filtrer les valeurs aberrantes
    # ========================================
    print("\n🧹 Nettoyage des valeurs aberrantes...")
    initial_len = len(df)
    
    # Temperature: entre -20 et 60°C (convertir Fahrenheit si nécessaire)
    mask_fahrenheit = df['temperature'] > 60
    if mask_fahrenheit.sum() > 0:
        print(f"   ✓ {mask_fahrenheit.sum()} températures > 60°C converties de Fahrenheit")
        df.loc[mask_fahrenheit, 'temperature'] = (df.loc[mask_fahrenheit, 'temperature'] - 32) * 5/9
    
    df = df[df['temperature'] >= -20]
    df = df[df['temperature'] <= 60]
    
    # Irradiance: doit être positive (entre 0 et 1500)
    df.loc[df['irradiance'] < 0, 'irradiance'] = 0
    df = df[df['irradiance'] <= 1500]
    
    # Efficacité: doit être > 0 et <= 1
    df = df[df['efficiency'] > 0]
    df = df[df['efficiency'] <= 1]
    
    # Cloud coverage: max 100%
    df.loc[df['cloud_coverage'] > 100, 'cloud_coverage'] = 100
    
    # Humidity: entre 0 et 100
    df.loc[df['humidity'] < 0, 'humidity'] = 0
    df.loc[df['humidity'] > 100, 'humidity'] = 100
    
    # Soiling ratio: entre 0 et 1
    df.loc[df['soiling_ratio'] < 0, 'soiling_ratio'] = 0
    df.loc[df['soiling_ratio'] > 1, 'soiling_ratio'] = 1
    
    print(f"   Lignes supprimées: {initial_len - len(df)}")
    print(f"   Taille après nettoyage: {len(df)} lignes")
    
    # Vérification finale
    print("\n✅ Vérification finale des données:")
    print(f"   Shape: {df.shape}")
    print(f"   Types:\n{df.dtypes}")
    print(f"   NaN restants: {df.isnull().sum().sum()}")
    
    return df

# ============================================
# Entraînement du modèle
# ============================================

def train_model(df: pd.DataFrame) -> tuple:
    """
    Entraîne le modèle GradientBoosting
    """
    print("\n🔧 Préparation des données...")
    
    # Vérifier que toutes les features sont présentes
    missing_features = [f for f in NUMERIC_FEATURES if f not in df.columns]
    if missing_features:
        print(f"   ⚠️ Features manquantes: {missing_features}")
        # Utiliser uniquement les features disponibles
        features_to_use = [f for f in NUMERIC_FEATURES if f in df.columns]
    else:
        features_to_use = NUMERIC_FEATURES
    
    print(f"   Features utilisées: {features_to_use}")
    
    X = df[features_to_use]
    y = df[TARGET]
    
    # Vérification des données
    print(f"\n   Vérification X:")
    print(f"   - Shape: {X.shape}")
    print(f"   - Types: {X.dtypes.unique()}")
    print(f"   - NaN: {X.isnull().sum().sum()}")
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n   Train: {len(X_train)} | Test: {len(X_test)}")
    
    # Pipeline avec StandardScaler + GradientBoosting
    print("\n🚀 Entraînement du modèle...")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), features_to_use)
        ]
    )
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        ))
    ])
    
    model.fit(X_train, y_train)
    print("   ✅ Modèle entraîné!")
    
    # Évaluation
    print("\n📊 Évaluation du modèle...")
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"   MSE: {mse:.4f}")
    print(f"   R²:  {r2:.4f}")
    
    # Importance des features
    print("\n📈 Importance des features:")
    feature_imp = model.named_steps['regressor'].feature_importances_
    for name, imp in sorted(zip(features_to_use, feature_imp), key=lambda x: -x[1]):
        bar = "█" * int(imp * 50)
        print(f"   {name:20s} {imp:.4f} {bar}")
    
    return model, {'mse': mse, 'r2': r2, 'features': features_to_use}

# ============================================
# Sauvegarde du modèle
# ============================================

def save_model(model, metrics: dict):
    """
    Sauvegarde le modèle entraîné
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print(f"\n💾 Sauvegarde du modèle dans {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)
    
    # Sauvegarder aussi les métriques
    metrics_path = os.path.join(MODEL_DIR, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"MSE: {metrics['mse']:.4f}\n")
        f.write(f"R²: {metrics['r2']:.4f}\n")
        f.write(f"Features: {metrics['features']}\n")
    
    print(f"   ✅ Modèle sauvegardé!")
    print(f"   📄 Métriques sauvegardées dans {metrics_path}")

# ============================================
# Main
# ============================================

def main():
    print("=" * 50)
    print("🌞 GREEN AI SOLAR - ENTRAÎNEMENT DU MODÈLE")
    print("=" * 50)
    
    # Chemin vers les données
    data_path = "data/train.csv"
    
    if not os.path.exists(data_path):
        print(f"\n❌ Fichier non trouvé: {data_path}")
        print("   Veuillez placer votre fichier train.csv dans le dossier 'data/'")
        print("\n   Création du dossier data/...")
        os.makedirs("data", exist_ok=True)
        return
    
    try:
        # Pipeline complet
        df = load_and_clean_data(data_path)
        model, metrics = train_model(df)
        save_model(model, metrics)
        
        print("\n" + "=" * 50)
        print("✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
        print("=" * 50)
        print(f"\n📊 Résultats:")
        print(f"   - R² Score: {metrics['r2']:.4f} ({metrics['r2']*100:.1f}%)")
        print(f"   - MSE: {metrics['mse']:.4f}")
        print(f"\n🚀 Pour démarrer l'API:")
        print("   uvicorn main:app --reload --port 8000")
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
