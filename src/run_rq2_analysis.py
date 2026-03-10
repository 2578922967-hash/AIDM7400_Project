import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import os

# Set paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, '../data/processed/clean_tmdb_movies.csv')
FIGURES_DIR = os.path.join(BASE_DIR, '../reports/figures')

if not os.path.exists(FIGURES_DIR):
    os.makedirs(FIGURES_DIR)

# Ensure NLTK resources
print("Checking NLTK resources...")
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    print("Downloading vader_lexicon...")
    nltk.download('vader_lexicon')
    
try:
    nltk.data.find('corpora/stopwords.zip')
except LookupError:
    print("Downloading stopwords...")
    nltk.download('stopwords')

# Load Data
print(f"Loading data from {PROCESSED_DATA_PATH}...")
try:
    df = pd.read_csv(PROCESSED_DATA_PATH)
except FileNotFoundError:
    print("Error: Processed data file not found. Please run Data Cleaning first.")
    exit(1)

df = df.dropna(subset=['overview'])
print(f"Data Loaded: {df.shape}")

# --- Keyword Extraction ---
def save_wordcloud(text_data, title, filename):
    print(f"Generating {title}...")
    stop_words = set(stopwords.words('english'))
    stop_words.update(['movie', 'film', 'one', 'two', 'new', 'story', 'life', 'man', 'world'])
    
    wordcloud = WordCloud(width=800, height=400, 
                          background_color='white', 
                          stopwords=stop_words,
                          min_font_size=10).generate(text_data)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    output_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved to {output_path}")

# Top 10% Revenue
high_revenue_df = df[df['revenue'] > df['revenue'].quantile(0.90)]
text_revenue = " ".join(high_revenue_df['overview'].astype(str))
save_wordcloud(text_revenue, "High Revenue Movies Keywords", "rq2_wordcloud_revenue.png")

# Top 10% Rating
high_rating_df = df[df['vote_average'] > df['vote_average'].quantile(0.90)]
text_rating = " ".join(high_rating_df['overview'].astype(str))
save_wordcloud(text_rating, "High Rated Movies Keywords", "rq2_wordcloud_rating.png")

# --- Sentiment Analysis ---
print("Running Sentiment Analysis (VADER)...")
sia = SentimentIntensityAnalyzer()
df['sentiment_score'] = df['overview'].apply(lambda x: sia.polarity_scores(str(x))['compound'])

# --- Correlation Analysis ---
print("Generating Correlation Plots...")
plt.figure(figsize=(12, 5))

# Sentiment vs Revenue
plt.subplot(1, 2, 1)
sns.scatterplot(x='sentiment_score', y='revenue', data=df, alpha=0.5)
plt.title('Sentiment vs. Revenue')
plt.xlabel('Sentiment Score (-1 to 1)')
plt.ylabel('Revenue')
plt.yscale('log')

# Sentiment vs Rating
plt.subplot(1, 2, 2)
sns.scatterplot(x='sentiment_score', y='vote_average', data=df, alpha=0.5)
plt.title('Sentiment vs. Vote Average')
plt.xlabel('Sentiment Score (-1 to 1)')
plt.ylabel('Vote Average')

plt.tight_layout()
output_path = os.path.join(FIGURES_DIR, "rq2_sentiment_scatter.png")
plt.savefig(output_path)
plt.close()
print(f"Saved scatter plots to {output_path}")

# Correlation Matrix
corr = df[['sentiment_score', 'revenue', 'vote_average', 'budget']].corr()
print("\nCorrelation Matrix:")
print(corr)

# Save correlation matrix to text file for report
with open(os.path.join(BASE_DIR, '../reports/drafts/rq2_correlation.txt'), 'w') as f:
    f.write(str(corr))

print("\nAnalysis Complete.")
