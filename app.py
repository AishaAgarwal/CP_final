import praw
import pandas as pd

reddit = praw.Reddit(client_id='rdEmiAcrLvYof7CH5zt_KA', client_secret='aHCKg6c1WuljfiOL2d9XsE73Y8gstA', user_agent='Competetive Programming')

# Create a list of subreddit names
subreddit_names = ['socialcauses', 'climatechange', 'environment', 'humanrights', 'gender', 'racialjustice']

# Iterate over the list of subreddit names and get the top 10 hot posts from each subreddit
posts = []

for subreddit in subreddit_names:
    for submission in reddit.subreddit(subreddit).hot(limit=100):
        submission_data = {
            "author": submission.author,
            "created_utc": submission.created_utc,
            "selftext": submission.selftext,
            "ups": submission.ups,
            "downs": submission.downs
        }
        posts.append(submission_data)

df = pd.DataFrame(posts)

df.to_csv('reddit_posts.csv')