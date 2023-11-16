import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import networkx as nx
from preprocess import load_and_preprocess_data
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def load_model():
    clf = joblib.load('community_detection_model.pkl')
    return clf

def train_and_save_model(X_train, X_test, y_train, y_test):
    print("Training and saving the model...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    print("Classification Report:\n", classification_report(y_test, y_pred))

    joblib.dump(clf, 'community_detection_model.pkl')

def community_detection():
    print("Performing community detection...")
    df = pd.read_csv('preprocessed_data.csv')  # Load preprocessed data

    interaction_graph = nx.Graph()

    for index, row in df.iterrows():
        if not pd.isnull(row['author_encoded']):
            interaction_graph.add_node(row['author_encoded'], type='author')
        if not pd.isnull(row['subreddit_encoded']):
            interaction_graph.add_node(row['subreddit_encoded'], type='subreddit')
        if not pd.isnull(row['author_encoded']) and not pd.isnull(row['subreddit_encoded']):
            interaction_graph.add_edge(row['author_encoded'], row['subreddit_encoded'])

    communities = nx.algorithms.community.modularity_max.greedy_modularity_communities(interaction_graph)

    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2)

    for i, community in enumerate(communities, start=1):
        subgraph = interaction_graph.subgraph(community)

        ax = fig.add_subplot(gs[i - 1])
        node_colors = ['blue' if subgraph.nodes[n]['type'] == 'author' else 'red' for n in subgraph.nodes]
        nx.draw(subgraph, with_labels=True, font_weight='bold', node_color=node_colors, ax=ax)
        plt.title(f"Community {i}")

    plt.tight_layout()
    plt.show()

    print(f"Community {i}: {list(community)}")



if __name__ == "__main__":
    print("Loading and preprocessing data...")
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # Train and save the machine learning model
    train_and_save_model(X_train, X_test, y_train, y_test)

    # Perform community detection
    community_detection()
