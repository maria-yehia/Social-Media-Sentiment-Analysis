
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.data.path.append("/home/ubuntu/nltk_data")

def preprocess_text(text):
    if not isinstance(text, str): # Handle non-string data, e.g., NaN
        return ""
    text = text.lower() # Lowercasing
    text = re.sub(r'[^a-z\s]', '', text) # Remove punctuation and numbers
    tokens = word_tokenize(text) # Tokenization
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word not in stop_words] # Remove stop words
    return " ".join(filtered_tokens)

def main():
    # The Kaggle dataset has no header, so we define column names
    column_names = ["target", "ids", "date", "flag", "user", "text"]
    df = pd.read_csv("/home/ubuntu/kaggle_data/training.1600000.processed.noemoticon.csv", encoding="ISO-8859-1", names=column_names)
    
    # Select a smaller subset for faster processing during demonstration
    df = df.sample(n=10000, random_state=42) # Taking 10,000 random samples

    df["cleaned_text"] = df["text"].apply(preprocess_text)
    df.to_csv("preprocessed_social_media_data.csv", index=False)
    print("Preprocessed data saved to preprocessed_social_media_data.csv")

if __name__ == "__main__":
    main()


