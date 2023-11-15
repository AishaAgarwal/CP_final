import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('reddit_posts.csv')

# data cleaning and handling missing values 

df = df.dropna(subset=['selftext'])

# feature engineering 


usernames = df['author'].tolist()
timestamps = df['created_utc']

# Create topic features using Latent Dirichlet Allocation (LDA)
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['selftext'])

lda = LatentDirichletAllocation(n_components=5)
topic_model = lda.fit(tfidf_matrix)
topic_features = pd.DataFrame(topic_model.transform(tfidf_matrix), index=df.index)

# Perform sentiment analysis using VaderSentiment
sentiment_analyzer = SentimentIntensityAnalyzer()
sentiment_scores = df['selftext'].apply(lambda text: sentiment_analyzer.polarity_scores(text)['compound'])
df['sentiment_score'] = sentiment_scores

# data normalisation and scaling 

df['ups'] = (df['ups'] - df['ups'].mean()) / df['ups'].std()
df['downs'] = (df['downs'] - df['downs'].mean()) / df['downs'].std()

# Step 4: Data Splitting and Preparation for Modeling

# Divide the preprocessed data into training and testing sets



X_train, X_test, y_train, y_test = train_test_split(pd.concat([df[['selftext', 'sentiment_score']], topic_features], axis=1), df['subreddit'], test_size=0.2)





# Represent communities using a combination of structural features, content features, and sentiment scores




# Encode categorical features for machine learning algorithms

print("Preprocessing and preparation of Reddit data completed.")

df.to_csv('preprocessed_data.csv', index=False)