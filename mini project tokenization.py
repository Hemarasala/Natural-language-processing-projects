#problem statement:
#1.dataset only:IMDb Movie Reviews (Sentiment Analysis)
# 2. pretrained dataset:
# 3. hybrid (dataset+pretained model )
#Raw Text Dataset
#Lowercasing
#Remove URLs, emojis, punctuation
#Tokenization
#Stopword removal
#Stemming / Lemmatization
#Clean Text Dataset (Output)

#mini project - 1(IMDb Movie Reviews (Sentiment Analysis)
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import re
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize

#download necessary nltk data
nltk.download('stopwords')
nltk.download('wordnet')

#load spacy model
nlp = spacy.load('en_core_web_sm')

#load dataset
df = pd.read_csv("IMDB.csv")
print(df.head())

# If dataset is empty or corrupted, create sample data for demonstration
if df.empty or len(df.columns) <= 1:
    df = pd.DataFrame({
        'review': [
            "This movie is absolutely fantastic! I loved every minute of it.",
            "Terrible film, waste of time. Very disappointed.",
            "Amazing performance by the actors. Highly recommended!!!",
            "Not worth watching. Poor plot and bad acting.",
            "Best movie I've seen in years. Definitely a masterpiece."
        ],
        'sentiment': ['positive', 'negative', 'positive', 'negative', 'positive']
    })
    print("\nUsing sample data:")
    print(df.head())

def preprocess_text(text):
    #url removal
    text_clean = re.sub(r"http\S+", "", text)
    
    #lowercase 
    text_clean = text_clean.lower()
    
    #stopword removal 
    stop_words = set(stopwords.words("english"))
    text_clean = " ".join([word for word in text_clean.split() if word not in stop_words])
    
    #stemming/lemmatization
    lemmatizer = WordNetLemmatizer()
    text_clean = " ".join([lemmatizer.lemmatize(word) for word in text_clean.split()])
    return text_clean
    
sample_text = "This is a sample text for tokenization"
doc = nlp(sample_text)
tokens = [token.text for token in doc]
print("Spacy tokens:", tokens)

#word tokenization 
words = sample_text.split()
print("Word tokens:", words)

#sentence tokenization 
sentences = sent_tokenize("This is the first sentence. This is the second sentence.")
print("Sentence tokens:", sentences)

#lower and upper case 
text_lower = sample_text.lower()
print("Lowercase:", text_lower)
text_upper = sample_text.upper()
print("Uppercase:", text_upper)

#subword tokenization 
word = "understanding"
subwords = []
if word.startswith("un"):
    subwords.append("un")
    word = word[2:]

if word.endswith("ness"):
    core = word[:-4]
    subwords.append(core)
    subwords.append("ness")
else:
    subwords.append(word)

print("Subword tokens:", subwords)

#compare word vs character tokenization 
text = "tokenization"
word_tokens = text.split()
char_tokens = list(text)
print("Word Tokens:", word_tokens)
print("Word Token Count:", len(word_tokens))
print("Character Tokens:", char_tokens)
print("Character Token Count:", len(char_tokens))

#Apply preprocessing to dataset

df['clean_review'] = df['review'].apply(preprocess_text)

print(df[['review', 'clean_review']].head())
df.to_csv("imdb_cleaned.csv", index=False)
print("Cleaned dataset saved successfully!")
