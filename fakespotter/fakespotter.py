import os
import praw
import pandas as pd
from transformers import pipeline
import time

# Reddit API setup: credentials come from environment variables (see README).
# Create an app at https://www.reddit.com/prefs/apps to get them.
reddit = praw.Reddit(
    client_id=os.environ["REDDIT_CLIENT_ID"],
    client_secret=os.environ["REDDIT_CLIENT_SECRET"],
    user_agent=os.environ.get("REDDIT_USER_AGENT", "FakeSpotter/1.0 (sentiment research script)"),
)

# Sentiment pipeline — DistilBERT SST-2
sentiment_analyzer = pipeline('sentiment-analysis')

def get_post_sentiment(submission):
    try:
        # Combine title and body
        combined_text = (submission.title or "") + " " + (submission.selftext or "")

        # Fetch all comments
        submission.comments.replace_more(limit=0)
        comments = [comment.body for comment in submission.comments.list()]
        combined_text += " " + " ".join(comments)

        # Truncate to 512 tokens worth of text (~2000 chars conservatively)
        combined_text = combined_text.strip()
        if len(combined_text) > 2048:
            combined_text = combined_text[:2048].rsplit(' ', 1)[0]

        # Run sentiment analysis
        result = sentiment_analyzer(combined_text)[0]
        return result['label'], result['score']

    except Exception as e:
        print(f"[!] Sentiment analysis failed: {e}")
        return "ERROR", 0.0

def scrape_subreddit(subreddit_name, limit=30):
    subreddit = reddit.subreddit(subreddit_name)
    data = []

    for idx, submission in enumerate(subreddit.hot(limit=limit), 1):
        print(f"[{idx}/{limit}] Processing: {submission.title[:60]}...")

        label, score = get_post_sentiment(submission)

        data.append({
            'title': submission.title,
            'score': submission.score,
            'num_comments': submission.num_comments,
            'url': submission.url,
            'created_utc': submission.created_utc,
            'sentiment_label': label,
            'sentiment_score': score
        })

        time.sleep(2)  # Respect Reddit API limits

    return pd.DataFrame(data)

if __name__ == "__main__":
    df = scrape_subreddit('Reps', limit=100)
    df['created_date'] = pd.to_datetime(df['created_utc'], unit='s')

    df.to_csv('fakewatches_deep_sentiment.csv', index=False)
    print("\n✅ Done! Saved to 'fakewatches_deep_sentiment.csv'")
