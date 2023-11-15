import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data():
    # Read the raw data
    df = pd.read_csv('reddit_posts.csv')

    # Drop rows with missing values in the 'selftext' column
    df = df.dropna(subset=['selftext'])

    # Feature engineering
    usernames = df['author'].tolist()
    timestamps = df['created_utc']

    # Create topic features using Latent Dirichlet Allocation (LDA)
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['selftext'])
    text_features = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out(), index=df.index)

    lda = LatentDirichletAllocation(n_components=5)
    topic_model = lda.fit(tfidf_matrix)
    topic_features = pd.DataFrame(topic_model.transform(tfidf_matrix), index=df.index)

    # Perform sentiment analysis using VaderSentiment
    sentiment_analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = df['selftext'].apply(lambda text: sentiment_analyzer.polarity_scores(text)['compound'])
    df['sentiment_score'] = sentiment_scores

    # Data normalization and scaling
    df['ups'] = (df['ups'] - df['ups'].mean()) / df['ups'].std()
    df['downs'] = (df['downs'] - df['downs'].mean()) / df['downs'].std()

    le_subreddit = LabelEncoder()
    df['subreddit_encoded'] = le_subreddit.fit_transform(df['subreddit'])

    le_author = LabelEncoder()
    df['author_encoded'] = le_author.fit_transform(df['author'])

    # Include the 'author' column in the feature set
    features = pd.concat([df[['author_encoded']], text_features, df[['sentiment_score']], topic_features], axis=1)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, df['subreddit_encoded'], test_size=0.2)
    
    # Convert column names to string to avoid potential issues
    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)

    
    preprocessed_data = pd.concat([X_train, X_test, pd.concat([y_train, y_test])], axis=1)
    preprocessed_data.to_csv('preprocessed_data.csv', index=False)

    # Print a message to indicate that preprocessing is completed
    return X_train, X_test, y_train, y_test

# Print a message to indicate that preprocessing is completed

if __name__ == "__main__":
    load_and_preprocess_data()