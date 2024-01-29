def get_youtube_sentiment(video_stats_df):
    video_stats_df["sentiment"] = video_stats_df["likes"] / (
        video_stats_df["likes"] + video_stats_df["dislikes"]
    )
    mean_sentiment_df = video_stats_df.groupby("category_id")["sentiment"].mean()
    mean_sentiment_df = mean_sentiment_df.sort_values(ascending=False)

    return mean_sentiment_df.rename({"sentiment": "mean_sentiment"})
