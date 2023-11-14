import pandas as pd 
posts_df = pd.read_csv('data.csv')
from wordcloud import WordCloud
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split


print(posts_df['title'].str.len().describe())
print(posts_df['score'].describe())
print(posts_df['num_comments'].describe())

posts_df['title'].str.len().hist()



X = posts_df[['sentiment', 'topic_probs']]
y = posts_df['subreddit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

plt.scatter(X_test['score'],X_test['num_comments'])


wordcloud = WordCloud().generate(' '.join(posts_df['title']))
plt.imshow(wordcloud, interpolation='bilinear')
plt.show()