import pandas as pd
import numpy as np
import ast  # For safe evaluation of stringified lists
import warnings
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# Cross-platform robust paths handling
# Get current working directory and adjust base path accordingly
try:
    NOTEBOOK_DIR = Path(__file__).resolve().parent
except NameError:
    # Fallback for Jupyter interactive execution
    NOTEBOOK_DIR = Path(os.getcwd())

PROJECT_DIR = NOTEBOOK_DIR.parent

# Construct standard absolute paths recognized by Windows/Mac/Linux
RAW_DATA_PATH = PROJECT_DIR / 'data' / 'raw' / 'TMDB_movie_dataset_v11.csv'
PROCESSED_DATA_PATH = PROJECT_DIR / 'data' / 'processed' / 'clean_tmdb_movies.csv'
TEST_DATA_PATH = PROJECT_DIR / 'data' / 'processed' / 'test_tmdb_movies.csv'

# Convert to string to maintain compatibility with legacy pandas methods
RAW_DATA_PATH = str(RAW_DATA_PATH)
PROCESSED_DATA_PATH = str(PROCESSED_DATA_PATH)
TEST_DATA_PATH = str(TEST_DATA_PATH)

# Check if file exists
if not os.path.exists(RAW_DATA_PATH):
    print(f"Error: File not found at {RAW_DATA_PATH}")
    print("Current working directory:", os.getcwd())
# 1. Load Data
try:
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Original Shape: {df.shape}")
    display(df.head(2))
except Exception as e:
    print(f"Failed to load data: {e}")
# 2. Filter out movies with 0 revenue or budget
# Many older or small movies have 0 revenue recorded. For serious box office analysis, these are noise.
# We will filter keeping only movies with revenue > 10000 and budget > 10000 to be safe.

print("Rows before filtering:", len(df))
clean_df = df[ (df['revenue'] > 10000) & (df['budget'] > 10000) ].copy()
print(f"Rows after filtering (Revenue > 10k & Budget > 10k): {len(clean_df)}")
# 3. Parse Metadata Columns
# Based on inspection, these columns are simple comma-separated strings.
# Example: "Action, Science Fiction, Adventure"

def parse_list_col(x):
    if isinstance(x, str):
        # Split by comma and strip whitespace
        return [item.strip() for item in x.split(',') if item.strip()]
    return []

# Columns that need parsing
list_cols = ['genres', 'production_companies', 'production_countries', 'spoken_languages']

# Inspect before
print("Sample genre before parsing:", clean_df['genres'].iloc[0])

for col in list_cols:
    # Check if column exists to avoid errors
    if col in clean_df.columns:
        print(f"Parsing {col}...")
        clean_df[col] = clean_df[col].apply(parse_list_col)

# Check the results
print("Sample genre after parsing:", clean_df['genres'].iloc[0])
clean_df[list_cols].head()
# 4. Feature Engineering & Time Filtering
# Convert release_date to datetime
clean_df['release_date'] = pd.to_datetime(clean_df['release_date'], errors='coerce')

# Extract Year, Month, and Quarter
clean_df['release_year'] = clean_df['release_date'].dt.year
clean_df['release_month'] = clean_df['release_date'].dt.month
clean_df['release_quarter'] = clean_df['release_date'].dt.quarter

# Drop rows with missing dates
clean_df = clean_df.dropna(subset=['release_date'])

# Filter data to include recent years for future prediction (e.g. 2004-2025)
start_year, end_year = 2004, 2025
print(f"Filtering movies from {start_year} to {end_year}...")
original_count = len(clean_df)
clean_df = clean_df[(clean_df['release_year'] >= start_year) & (clean_df['release_year'] <= end_year)]
print(f"Retained {len(clean_df)} movies (dropped {original_count - len(clean_df)}).")

# Calculate ROI (Return on Investment)
# ROI = (Revenue - Budget) / Budget
clean_df['roi'] = (clean_df['revenue'] - clean_df['budget']) / clean_df['budget']

# Filter ROI Outliers (Extreme ROIs can skew modeling)
clean_df = clean_df[(clean_df['roi'] >= -1) & (clean_df['roi'] < 50)]

print("Date columns processed, time filter applied, and ROI outliers removed.")
# 5. Save to Processed
# Since list columns can be tricky in CSVs (they load as strings), we accept that they will be saved as string reps.

processed_dir = Path(PROCESSED_DATA_PATH).parent
if not processed_dir.exists():
    processed_dir.mkdir(parents=True, exist_ok=True)

# Split data: 2004-2023 goes to train/original table, >2023 goes to test table
train_df = clean_df[clean_df['release_year'] <= 2023]
test_df = clean_df[clean_df['release_year'] > 2023]

# Enforce utf-8-sig encoding to prevent bad characters across platforms
train_df.to_csv(PROCESSED_DATA_PATH, index=False, encoding='utf-8-sig')
test_df.to_csv(TEST_DATA_PATH, index=False, encoding='utf-8-sig')

print(f"Saved 2004-2023 processed data to {PROCESSED_DATA_PATH} (Shape: {train_df.shape})")
print(f"Saved >2023 test data to {TEST_DATA_PATH} (Shape: {test_df.shape})")
