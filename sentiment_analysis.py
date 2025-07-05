
import pandas as pd
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.data.path.append("/home/ubuntu/nltk_data")

def analyze_sentiment_textblob(text):
    if not isinstance(text, str):
        return None # Return None for non-string inputs
    analysis = TextBlob(text)
    return analysis.sentiment.polarity # Polarity ranges from -1 (negative) to 1 (positive)

def analyze_sentiment_vader(text):
    if not isinstance(text, str):
        return None # Return None for non-string inputs
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(text)
    return vs["compound"] # Compound score ranges from -1 (most negative) to 1 (most positive)

def main():
    df = pd.read_csv("preprocessed_social_media_data.csv")
    
    # TextBlob Sentiment
    df["sentiment_textblob"] = df["cleaned_text"].apply(analyze_sentiment_textblob)
    
    # VADER Sentiment
    df["sentiment_vader"] = df["cleaned_text"].apply(analyze_sentiment_vader)
    
    df.to_csv("sentiment_analyzed_data.csv", index=False)
    print("Sentiment analyzed data saved to sentiment_analyzed_data.csv")

if __name__ == "__main__":
    main()


