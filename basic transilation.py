
from transformers import MarianMTModel, MarianTokenizer

# Load pretrained English → Hindi model
model_name = "Helsinki-NLP/opus-mt-en-hi"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_english_to_hindi(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated_tokens = model.generate(
        **inputs,
        max_length=128,
        num_beams=5,
        early_stopping=True
    )
    return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

#test
while True:
    text = input("\nEnter English sentence (or 'exit'): ")
    if text.lower() == "exit":
        break
    print("Hindi Translation:", translate_english_to_hindi(text))