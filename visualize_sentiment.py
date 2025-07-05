
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_sentiment(df):
    # Convert 'date' column to datetime objects
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    # Overall sentiment trend over time (VADER)
    plt.figure(figsize=(12, 6))
    df["sentiment_vader"].resample("D").mean().plot()
    plt.title("Overall Sentiment Trend Over Time (VADER)")
    plt.xlabel("Date")
    plt.ylabel("Average VADER Sentiment Score")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("overall_sentiment_trend_kaggle.png")
    plt.close()

    # Convert VADER compound score to categorical sentiment
    def categorize_sentiment(score):
        if score >= 0.05:
            return "Positive"
        elif score <= -0.05:
            return "Negative"
        else:
            return "Neutral"

    df["vader_sentiment_category"] = df["sentiment_vader"].apply(categorize_sentiment)

    # Sentiment counts by category
    plt.figure(figsize=(8, 8))
    df["vader_sentiment_category"].value_counts().plot(kind="pie", autopct="%1.1f%%")
    plt.title("Overall Sentiment Distribution (VADER)")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig("overall_sentiment_distribution_kaggle.png")
    plt.close()

def main():
    df = pd.read_csv("sentiment_analyzed_data.csv")
    visualize_sentiment(df)
    print("Sentiment visualizations generated: overall_sentiment_trend_kaggle.png, overall_sentiment_distribution_kaggle.png")

if __name__ == "__main__":
    main()


