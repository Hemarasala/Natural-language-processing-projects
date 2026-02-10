import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def preprocess_text(text):
    text_clean = re.sub(r'http\S+', '', text)
    text_clean = text_clean.lower()
    stop_words = set(stopwords.words('english'))
    text_clean = ' '.join([word for word in text_clean.split() if word not in stop_words])
    lemmatizer = WordNetLemmatizer()
    text_clean = ' '.join([lemmatizer.lemmatize(word) for word in text_clean.split()])
    return text_clean

# Sample data
data = {
    'review': [
        'This movie is absolutely fantastic! I loved every minute of it.',
        'Terrible film, waste of time. Very disappointed.',
        'Amazing performance by the actors. Highly recommended!!!',
        'Not worth watching. Poor plot and bad acting.',
        'Best movie I\'ve seen in years. Definitely a masterpiece.'
    ],
    'sentiment': ['positive', 'negative', 'positive', 'negative', 'positive']
}

df = pd.DataFrame(data)

print("Original Reviews:")
print("="*100)
print(df[['review']].to_string())

print("\n\nApplying Preprocessing...")
print("="*100)

df['clean_review'] = df['review'].apply(preprocess_text)

print("\nCleaned Reviews:")
print("="*100)
for idx, row in df.iterrows():
    print(f"\nReview {idx+1}:")
    print(f"Original:  {row['review']}")
    print(f"Cleaned:   {row['clean_review']}")
    print(f"Sentiment: {row['sentiment']}")

print("\n" + "="*100)
print("\nSample of Original vs Cleaned:")
print(df[['review', 'clean_review']].to_string())

df.to_csv("imdb_cleaned.csv", index=False)
print("\n\nCleaned dataset saved as 'imdb_cleaned.csv'")
