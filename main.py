from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO
import base64
from preprocess import load_and_preprocess_data, get_subreddit_name
from model import load_model

app = Flask(__name__)

clf = load_model()

# X_train, X_test, y_train, y_test = load_and_preprocess_data()
def train_and_save_model(X_train, X_test, y_train, y_test):
    print("Training and saving the model...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=1))


    joblib.dump(clf, 'community_detection_model.pkl')

def community_detection():
    print("Performing community detection...")
    df = pd.read_csv('preprocessed_data.csv')
    print(df)
    interaction_graph = nx.DiGraph()

    for index, row in df.iterrows():
        if not pd.isnull(row['author_encoded']):
            interaction_graph.add_node(row['author_encoded'], type='author')
        if not pd.isnull(row['subreddit_encoded']):
            subreddit_name = row['subreddit'] if not pd.isnull(row['subreddit']) else 'Unknown'  # Replace 'Unknown' with your default value
            interaction_graph.add_node(row['subreddit_encoded'], type='subreddit', name=subreddit_name)
        if not pd.isnull(row['author_encoded']) and not pd.isnull(row['subreddit_encoded']):
            interaction_graph.add_edge(row['author_encoded'], row['subreddit_encoded'])

    communities = nx.algorithms.community.modularity_max.greedy_modularity_communities(interaction_graph)

    return interaction_graph, communities


def draw_community_graph(graph, communities):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    gs = axes.flatten()

    for i, community in enumerate(communities, start=1):
        subgraph = graph.subgraph(community)
        node_colors = ['blue' if subgraph.nodes[n]['type'] == 'author' else 'red' for n in subgraph.nodes]

        # Add labels to the nodes (subreddits)
        node_labels = {n: subgraph.nodes[n]['name'] for n in subgraph.nodes if subgraph.nodes[n]['type'] == 'subreddit'}

        nx.draw(subgraph, with_labels=True, font_weight='bold', labels=node_labels,
                node_color=node_colors, ax=gs[i - 1])
        gs[i - 1].set_title(f"Community {i}")

    plt.tight_layout()
    img_data = BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)
    encoded_img = base64.b64encode(img_data.read()).decode('utf-8')
    plt.close()

    return encoded_img

def user_engagement_analysis(df):
    print("Performing user engagement analysis...")
    df = pd.read_csv('preprocessed_data.csv')

    # Create a directed graph for user engagement
    engagement_graph = nx.DiGraph()

    # Add nodes and edges based on user interactions
    for index, row in df.iterrows():
        if not pd.isnull(row['author_encoded']):
            engagement_graph.add_node(row['author_encoded'], type='author')
        if not pd.isnull(row['subreddit_encoded']):
            engagement_graph.add_node(row['subreddit_encoded'], type='subreddit')
        if not pd.isnull(row['author_encoded']) and not pd.isnull(row['subreddit_encoded']):
            engagement_graph.add_edge(row['author_encoded'], row['subreddit_encoded'])

    # Calculate degree centrality for each user
    user_degree_centrality = nx.degree_centrality(engagement_graph)

    # Identify highly engaging users
    engaging_users = [user for user, centrality in user_degree_centrality.items() if centrality > 0.5]

    # Create a subgraph containing only engaging users and their interactions
    subgraph = engagement_graph.subgraph(engaging_users)

    # Visualize the user engagement graph
    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw(subgraph, with_labels=True, font_weight='bold', ax=ax)
    plt.title("User Engagement Network")

    img_data = BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)
    encoded_img = base64.b64encode(img_data.read()).decode('utf-8')
    plt.close()

    return encoded_img


def get_subreddit_name(subreddit_encoded):
    # Load the original data
    original_data = pd.read_csv('reddit_posts.csv')
    # Extract the subreddit name corresponding to the encoded value
    subreddit_name = original_data.loc[original_data['subreddit_encoded'] == subreddit_encoded, 'subreddit'].iloc[0]
    return subreddit_name

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        subreddit_names = request.form.get('subreddit_names')
        print(subreddit_names)
        subreddit_names = [sub.strip() for sub in subreddit_names.split(',')]
        print(subreddit_names)
        
        # Include logic to update subreddit_names and rerun the analysis\

        
        graph, communities = community_detection()
        user_engagement_img = user_engagement_analysis(X_train)
        encoded_img = draw_community_graph(graph, communities)

        return render_template('index.html', graph=encoded_img, communities=list(communities))

    return render_template('index.html', graph=None, communities=None)

@app.route('/get_subreddit_name/<subreddit_encoded>', methods=['GET'])
def get_subreddit_name_route(subreddit_encoded):
    subreddit_name = get_subreddit_name(int(subreddit_encoded))
    return f"Subreddit Name: {subreddit_name}"

if __name__ == '__main__':
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # Train and save the machine learning model
    X_test = X_test.drop(['subreddit'],axis=1)
    X_train = X_train.drop(['subreddit'],axis=1)
    print(X_test)
    print(X_train)
    train_and_save_model(X_train, X_test, y_train, y_test)

    # Run the Flask app
    app.run(debug=True)
