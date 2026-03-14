import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import ast
import os
from pathlib import Path

# Prevent font issues and set style
plt.rcParams['axes.unicode_minus'] = False
sns.set(style="whitegrid")
pd.set_option('display.max_columns', None)

# 1. Load Processed Data
try:
    NOTEBOOK_DIR = Path(__file__).resolve().parent
except NameError:
    NOTEBOOK_DIR = Path(os.getcwd())

PROJECT_DIR = NOTEBOOK_DIR.parent
    
PROCESSED_DATA_PATH = PROJECT_DIR / 'data' / 'processed' / 'clean_tmdb_movies.csv'
TEST_DATA_PATH = PROJECT_DIR / 'data' / 'processed' / 'test_tmdb_movies.csv'

try:
    df_train = pd.read_csv(PROCESSED_DATA_PATH)
    df_test = pd.read_csv(TEST_DATA_PATH)
    df = pd.concat([df_train, df_test], ignore_index=True)
except FileNotFoundError:
    print("Processed files not found, please run 01_Data_Cleaning.ipynb first.")
    exit(1)

# Parse list columns
if df['genres'].dtype == object and df['genres'].str.startswith('[').any():
    df['genres'] = df['genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

# Ensure date column is datetime
if 'release_date' in df.columns:
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

print(f"Data Loaded: {df.shape}")

# 2.1 Define High ROI Threshold (Top 30%)
threshold = df['roi'].quantile(0.70)
print(f"Top 30% ROI Threshold: {threshold:.2f}")

# Generate Target Variable
df['is_high_roi'] = (df['roi'] >= threshold).astype(int)

# 2.2 Seasonality Features
df['release_year'] = df['release_date'].dt.year
df['release_month'] = df['release_date'].dt.month
df['release_quarter'] = df['release_date'].dt.quarter

# Holiday Seasons
# Summer: 6, 7, 8
df['is_summer_season'] = df['release_month'].isin([6, 7, 8]).astype(int)
# Holiday/Christmas: 11, 12
df['is_holiday_season'] = df['release_month'].isin([11, 12]).astype(int)
# Spring: 1, 2
df['is_spring_season'] = df['release_month'].isin([1, 2]).astype(int)

# 2.3 Categorical Features: Genres One-Hot Encoding
if df['genres'].apply(type).eq(list).any():
    # explode and create dummies
    exploded = df['genres'].explode()
    # Create dummy variables
    genres_df = pd.get_dummies(exploded).groupby(level=0).sum()
else:
    # Assume comma separated
    genres_df = df['genres'].str.get_dummies(sep=', ')
    
# Merge Features
df_model = pd.concat([df, genres_df], axis=1)

print(f"Data Shape after Feature Engineering: {df_model.shape}")

# Split Dataset
train_data = df_model[df_model['release_year'] <= 2023]
test_data = df_model[df_model['release_year'] > 2023]

# Select Features
feature_cols = ['budget', 'runtime', 'release_month', 
                'is_summer_season', 'is_holiday_season', 'is_spring_season'] + list(genres_df.columns)

# Handle potential missing columns in test/train split due to get_dummies
for col in feature_cols:
    if col not in train_data.columns:
        train_data[col] = 0
    if col not in test_data.columns:
        test_data[col] = 0

X_train = train_data[feature_cols].fillna(0)
y_train = train_data['is_high_roi']

X_test = test_data[feature_cols].fillna(0)
y_test = test_data['is_high_roi']

print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# Train Model
clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluate
print("\nClassification Report (Test Set > 2023):")
print(classification_report(y_test, y_pred))

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

FIGURES_DIR = PROJECT_DIR / 'reports' / 'figures'
if not FIGURES_DIR.exists():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(12, 6))
plt.title("Top 15 Feature Importances for High ROI Prediction")
plt.bar(range(15), importances[indices[:15]], align="center")
plt.xticks(range(15), [feature_cols[i] for i in indices[:15]], rotation=45)
plt.xlim([-1, 15])
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'rq3_revised_feature_importance.png')
print(f"Saved feature importance plot to {FIGURES_DIR / 'rq3_revised_feature_importance.png'}")
