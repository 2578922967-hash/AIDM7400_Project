import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import ast
import os

# Set paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, '../data/processed/clean_tmdb_movies.csv')
FIGURES_DIR = os.path.join(BASE_DIR, '../reports/figures')
DRAFTS_DIR = os.path.join(BASE_DIR, '../reports/drafts')

if not os.path.exists(FIGURES_DIR):
    os.makedirs(FIGURES_DIR)
if not os.path.exists(DRAFTS_DIR):
    os.makedirs(DRAFTS_DIR)

sns.set(style="whitegrid")

# Load Data
print(f"Loading data from {PROCESSED_DATA_PATH}...")
try:
    df = pd.read_csv(PROCESSED_DATA_PATH)
except FileNotFoundError:
    print("Error: Processed data file not found.")
    exit(1)

# Parse list columns
list_cols = ['genres', 'production_companies']
for col in list_cols:
    df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

# Recalculate ROI
df['roi'] = (df['revenue'] - df['budget']) / df['budget']

# Filter reasonable ROI range (-100% to 5000% ROI) to avoid extreme outliers skewing the model
# ROI = -1 means lost all money. ROI = 50 means 50x return.
df_clean = df[(df['budget'] > 10000) & (df['revenue'] > 10000)].copy()
df_clean = df_clean[(df_clean['roi'] > -1) & (df_clean['roi'] < 50)]

print(f"Data for Modeling: {df_clean.shape} movies (filtered for ROI outliers)")

# --- Feature Engineering ---
print("Engineering Features...")
# 1. Genres (One-Hot)
# Explode genres to get all unique genres first
all_genres = set([g for sublist in df_clean['genres'] for g in sublist])
for genre in all_genres:
    df_clean[f'genre_{genre}'] = df_clean['genres'].apply(lambda x: 1 if genre in x else 0)

# 2. Production Company Count
df_clean['prod_company_count'] = df_clean['production_companies'].apply(len)

# 3. Seasonality (Quarter/Month)
df_clean['release_quarter'] = df_clean['release_month'].apply(lambda x: (x-1)//3 + 1)

# Features list
genre_features = [f'genre_{g}' for g in all_genres]
features = ['budget', 'runtime', 'release_month', 'prod_company_count'] + genre_features
target = 'roi'

# Drop NaN
df_model = df_clean.dropna(subset=features + [target])

X = df_model[features]
y = df_model[target]

print(f"Features: {len(features)} | Samples: {len(X)}")

# --- Model Training ---
print("Training Random Forest Regressor...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# --- Evaluation ---
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("-" * 30)
print(f"Model Evaluation (Random Forest):")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R2 Score: {r2:.4f}")
print("-" * 30)

# Save results to text file
with open(os.path.join(DRAFTS_DIR, 'rq3_model_results.txt'), 'w') as f:
    f.write(f"Model Evaluation (Random Forest):\n")
    f.write(f"MSE: {mse:.4f}\n")
    f.write(f"MAE: {mae:.4f}\n")
    f.write(f"R2: {r2:.4f}\n")

# --- Feature Importance ---
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
top_n = 10

plt.figure(figsize=(12, 6))
plt.title(f"Top {top_n} Feature Importances for ROI Prediction")
plt.bar(range(top_n), importances[indices[:top_n]], align="center")
plt.xticks(range(top_n), [features[i] for i in indices[:top_n]], rotation=45)
plt.xlim([-1, top_n])
plt.tight_layout()

output_path = os.path.join(FIGURES_DIR, "rq3_feature_importance.png")
plt.savefig(output_path)
print(f"Saved Feature Importance plot to {output_path}")

print("\nRQ3 Analysis Complete.")
