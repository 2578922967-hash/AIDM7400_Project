import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import os

# Set paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, '../data/processed/clean_tmdb_movies.csv')
FIGURES_DIR = os.path.join(BASE_DIR, '../reports/figures')

if not os.path.exists(FIGURES_DIR):
    os.makedirs(FIGURES_DIR)

sns.set(style="whitegrid")

# Load Data
print(f"Loading data from {PROCESSED_DATA_PATH}...")
try:
    df = pd.read_csv(PROCESSED_DATA_PATH)
except FileNotFoundError:
    print("Error: Processed data file not found. Please run Data Cleaning first.")
    exit(1)

# Parse list columns
list_cols = ['genres', 'production_countries']
for col in list_cols:
    df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

print(f"Data Loaded: {df.shape}")

# --- RQ1.1 Genre Evolution ---
print("Analyzing Genre Trends...")
# Explode genres
df_genres = df.explode('genres')

# Get top 8 genres
top_genres = df_genres['genres'].value_counts().head(8).index.tolist()
print(f"Top 8 Genres: {top_genres}")

# Filter data
genre_trends = df_genres[df_genres['genres'].isin(top_genres)]
genre_trends_yearly = genre_trends.groupby(['release_year', 'genres']).size().reset_index(name='count')

# Plot
plt.figure(figsize=(14, 7))
sns.lineplot(data=genre_trends_yearly, x='release_year', y='count', hue='genres', marker='o', linewidth=2.5)
plt.title('Evolution of Top Movie Genres (2004-2023)', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Movies', fontsize=12)
plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

output_path_genre = os.path.join(FIGURES_DIR, "rq1_genre_evolution.png")
plt.savefig(output_path_genre)
plt.close()
print(f"Saved Genre Evolution plot to {output_path_genre}")

# --- RQ1.2 Production Powerhouses ---
print("Analyzing Production Country Trends...")
# Explode countries
df_countries = df.explode('production_countries')

# Get top 5 countries
top_countries = df_countries['production_countries'].value_counts().head(5).index.tolist()
print(f"Top 5 Countries: {top_countries}")

# Filter data
country_trends = df_countries[df_countries['production_countries'].isin(top_countries)]
country_trends_yearly = country_trends.groupby(['release_year', 'production_countries']).size().reset_index(name='count')

# Plot
plt.figure(figsize=(14, 7))
sns.lineplot(data=country_trends_yearly, x='release_year', y='count', hue='production_countries', marker='s', linewidth=2.5)
plt.title('Top Production Countries Trends (2004-2023)', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Movies Produced', fontsize=12)
plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

output_path_country = os.path.join(FIGURES_DIR, "rq1_country_trends.png")
plt.savefig(output_path_country)
plt.close()
print(f"Saved Country Trends plot to {output_path_country}")

print("\nRQ1 Analysis Complete.")
