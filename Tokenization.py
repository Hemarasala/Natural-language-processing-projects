import re
print("PROBLEM 1: Basic Word Tokenization")
text1 = "Hi, my name is hema Rasala. I would like to read books."
text2 = "I am a student of class 12th"
tokens1 = text1.split()
tokens2 = text2.split()
tokens3 = tokens2 + tokens1
print(tokens2)
print(tokens1)
print(tokens3)

#PROBLEM 2: Sentence Tokenization"
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
nltk.download('punkt_tab') 
text1 = "AI is powerful. It is used in healthcare. NLP is a branch of AI."
text2 = "I am a student of Woxsen University."
text = text1 + " " + text2
sentences = sent_tokenize(text)
print(sentences)

#tokenization with punctuation Removal
print("\nPROBLEM 3: Tokenization with Punctuation Removal")
text1= "Hello!!! Are you ready, for NLP?"
text2 = "i strucked in a rain!! i need some help?"
text1 = text1.replace("!!!", ",").replace("?", "").replace(",","")
text2 = text2.replace("!!", "").replace("?", ",")
print(text1)
print (text2)

#lowercase and upper case 
print("\nPROBLEM 4: Lowercasing and uppercasing + tokenization")
text1 = "Machine Learning Is AMAZING"
text2 = "i am a boy and she is a girl"
tokens2 = text2.upper().split()
tokens1 = text1.lower().split()
print("Tokens from text 2 (Upper):", tokens2)
print("Tokens from text 1 (Lower):", tokens1)


#stopword
print("\nPROBLEM 5: Stopword Removal during Tokenization")
text1 = "AI is a branch of computer science"
stopwords = ["is", "a", "of"]
text2 = "i love coding but i hate it sometime as well"
stopwords2 = ["i", "it", "as"]
tokens1= [word for word in text1.lower().split() if word not in stopwords]
tokens2= [word for word in text2.lower().split() if word not in stopwords2]
print(tokens1)
print(tokens2)


#spacy a smart NLP engine that understands text like humans
print("\nPROBLEM 6: spacy Tokenization" )
import spacy
nlp = spacy.load("en_core_web_sm")
text1 = "I bought 3 phones and 15 laptops in 2024."
text2 = "I am a student of Woxsen University."
text3 = text1 + " " + text2
doc = nlp(text3)
words = [token.text.lower() for token in doc if token.is_alpha]
numbers = [token.text for token in doc if token.like_num]
print("Words:", words)
print("Numbers:", numbers)


print("\nPROBLEM 7: Tokenization of Social Media Text")
text1 = "Learning #NLP is fun 🤖! Visit https://example.com"
text2 = "removing some url of instagram user! visit https://www.reddit.com/r/GeminiAI/comments/1ppik6d/fix_for_google_antigravitys_terminal_blindness_it/"
# Remove URL
text1_no_url = re.sub(r"http\S+", "", text1)
text2_no_url = re.sub(r"http\S+", "", text2)
# Keep hashtags, remove emojis
tokens1 = re.findall(r"#\w+|[A-Za-z]+", text1_no_url)
tokens2 = re.findall(r"#\w+|[A-Za-z]+", text1_no_url)
print(tokens2)
print(tokens1)


print("\nPROBLEM 8: Subword Tokenization (Rule-Based)")
word = "unhappiness"
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

print(subwords)


print("\nPROBLEM 9: Compare Word vs Character Tokenization")
text9 = "tokenization"
word_tokens = text9.split()
char_tokens = list(text9)
print("Word Tokens:", word_tokens)
print("Word Token Count:", len(word_tokens))
print("Character Tokens:", char_tokens)
print("Character Token Count:", len(char_tokens))
