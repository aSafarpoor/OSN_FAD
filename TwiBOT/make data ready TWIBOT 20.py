"""
Script for:
1. Extracting keywords from tweets using YAKE.
2. Generating and fitting FastText embeddings (with PCA) for keywords and user descriptions.
3. Building a heterogeneous graph from user, tweet, and keyword nodes.
4. Assigning embeddings to each node in the graph.
5. Saving graph data (nodes, edges, embeddings, labels) to CSV files.
"""

import csv
import json
import pickle
import random
import re
from collections import Counter
from tqdm import tqdm
import json
import networkx as nx
import numpy as np
import yake

from gensim.models.fasttext import load_facebook_vectors
from sklearn.decomposition import IncrementalPCA


# --------------------
# Parameters / Constants
# --------------------
SEED = 1
random.seed(SEED)

# File paths for the datasets
DATASET_PATH = "."  

# Number of keywords to extract per tweet
NUMBER_OF_KEYWORDS = 10

# FastText model path
FASTTEXT_BIN_PATH = 'cc.en.300.bin'

# PCA output path
PCA_MODEL_PATH = "pca_model.pkl"

# Pickle files to store intermediate data
ALL_KEYWORDS_PATH = 'all_keywords.pickle'
KEYWORDS_USERS_DIC_PATH = 'keywords_users_dic.pickle'
WORD_EMBEDDING_DICTIONARY_PATH = 'word_embedding_dictionary.pickle'
TWEETS_USERS_DIC_PATH = 'tweets_users_dic.pkl'
TWEET_ID_EMBEDDING_DIC_PATH = 'tweet_id_embedding_dic.pickle'
ID_DESCRIPTION_DIC_PATH = 'embeddings_id_description_dic.pickle'
ID_LABEL_DICT_PATH = 'id_label_dict.pickle'

# Final heterogeneous graph
HG_PATH_0 = 'HG0.pickle'
HG_PATH_2 = 'HG2.pickle'

# Final CSV outputs
NODE_EMBEDDINGS_CSV = 'node_embeddings.csv'
NODE_LABELS_CSV = 'node_labels.csv'
NODE_INFO_CSV = 'node_information.csv'
EDGES_CSV = 'edges.csv'

File_Names = ['dev.json','test.json','train.json','support.json']
yake_extractor = yake.KeywordExtractor(lan="en", n=1, dedupLim=0.9)


# --------------------
# Simple Text Processing Functions   
# --------------------
def extract_keywords(text,NUMBER_OF_KEYWORDS) -> list:
    keywords = [kw[0] for kw in yake_extractor.extract_keywords(text)[:NUMBER_OF_KEYWORDS]]
    return keywords

def preprocess_text(text: str) -> str:
    """
    Preprocess text by:
    1. Replacing mentions, hashtags, links.
    2. Removing non-alphabetic characters.
    3. Converting to lowercase.
    4. Removing extra spaces.
    """
    text = re.sub(r"@\w+", "mention", text)
    text = re.sub(r"#\w+", "hashtags", text)
    text = re.sub(r"https?://\S+|www\.\S+", "link", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

# --------------------
# Make PCA Ready based on Sampling   
# --------------------

class FastTextEmbeddingModel:
    def __init__(self, fasttext_model, target_size=32):
        """
        Initialize the embedding model with a FastText model and PCA configuration.
        """
        self.model = fasttext_model
        self.target_size = target_size
        self.pca = None

    def _compute_word_embedding(self, word: str) -> np.ndarray:
        """ Fetch embedding for a single word. """
        if word in self.model:
            return self.model[word]
        return np.zeros(self.model.vector_size)

    def _compute_sentence_embedding(self, sentence: str) -> np.ndarray:
        """
        Compute the embedding for a sentence by averaging the embeddings of its words.
        """
        words = sentence.split()  # Tokenize
        embeddings = [self._compute_word_embedding(w) for w in words if w in self.model]
        if embeddings:
            return np.mean(embeddings, axis=0)
        return np.zeros(self.model.vector_size)

    def _compute_embedding(self, text: str) -> np.ndarray:
        """
        Determine if the input is a sentence or a single word and compute its embedding.
        """
        if " " in text:  # Sentence
            return self._compute_sentence_embedding(text)
        else:  # Word
            return self._compute_word_embedding(text)

    def fit_and_save(self, inputs, pca_save_path=PCA_MODEL_PATH, batch_size=10000):
        """
        Fit PCA incrementally on a large dataset and save the trained PCA model.
        """
        if isinstance(inputs, set):
            inputs = list(inputs)

        self.pca = IncrementalPCA(n_components=self.target_size)

        # Process inputs in batches to reduce memory usage
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            embeddings = [self._compute_embedding(text) for text in batch]
            self.pca.partial_fit(embeddings)

        # Save the PCA model
        with open(pca_save_path, 'wb') as f:
            pickle.dump(self.pca, f)
        print(f"PCA model trained and saved to {pca_save_path}")

    def load_pca(self, pca_load_path=PCA_MODEL_PATH):
        """ Load a pre-trained PCA model from a file. """
        with open(pca_load_path, 'rb') as f:
            self.pca = pickle.load(f)
        print(f"PCA model loaded from {pca_load_path}")

    def transform(self, inputs):
        """
        Transform a list of words or sentences into reduced-dimensional embeddings.
        Returns a dictionary {text: embedding_array}.
        """
        if not self.pca:
            raise ValueError("PCA is not fitted or loaded. Fit PCA or load a model before transforming.")
        embeddings = [self._compute_embedding(text) for text in inputs]
        reduced_embeddings = self.pca.transform(embeddings)
        return reduced_embeddings
    
        # return {text: reduced_embeddings[i] for i, text in enumerate(inputs)}

    def transform_dictionary(self, inputs: dict):
        """
        Transform inputs (dictionary with keys -> text) into reduced embeddings.
        Returns {key: embedding_array}.
        """
        if not self.pca:
            raise ValueError("PCA is not fitted or loaded. Fit PCA or load a model before transforming.")
        embeddings = [self._compute_embedding(text) for text in inputs.values()]
        reduced_embeddings = self.pca.transform(embeddings)
        return {key: reduced_embeddings[i] for i, key in enumerate(inputs.keys())}



    
SAMPLES_PER_FILE = 100000  #  samples per file
def sample_tweets_from_file(file_name, samples_per_file,SEED=SEED):
    random.seed(SEED)
    sampled_tweets = []

    with open(file_name, "r", encoding="utf-8") as file:
        data = json.load(file)

        # Ensure we don't request more samples than there are users
        users_to_sample = min(samples_per_file, len(data))

        # Randomly select users
        sampled_users = random.sample(data, users_to_sample)

        for user in tqdm(sampled_users, desc=f"Processing {file_name}"):
            tweets = user.get("tweet", [])
            if tweets:  # Ensure the user has tweets
                sampled_tweet = random.choice(tweets)  # Select one random tweet
                sampled_tweets.append(preprocess_text(sampled_tweet))

    return sampled_tweets

sampled_tweets = sample_tweets_from_file('support.json',samples_per_file=SAMPLES_PER_FILE)


print("Loading FastText model...")
fasttext_model = load_facebook_vectors(FASTTEXT_BIN_PATH)
embedding_model = FastTextEmbeddingModel(fasttext_model, target_size=32)

# Fit PCA on all keywords
embedding_model.fit_and_save(sampled_tweets, pca_save_path=PCA_MODEL_PATH)

# 4. Transform all_keywords to get word embeddings
# keywords_embedding = embedding_model.transform(all_keywords)
# with open(WORD_EMBEDDING_DICTIONARY_PATH, 'wb') as f:
#     pickle.dump(keywords_embedding, f)
# print("Saved word embedding dictionary.")

# 5. Build description dictionary and embed
# embeddings_id_description_dic = build_description_dictionary(embedding_model)

# 6. Build tweet dictionary and embed
# tweets_users_dic, tweet_id_embedding_dic = build_tweet_dictionary(embedding_model)

 
# --------------------
# Read and Prepare Files    
# --------------------


dic_user_tweetid = {}
dic_user_description_embedding = {}
dic_tweetid_keywords = {}
dic_tweet_id_embedding = {}
all_keywords = set()
dic_user_label = {}

tweetid_counter = 0
for file_name in File_Names:
    with open(file_name, "r", encoding="utf-8") as file:
        data = json.load(file)
    
        for user in tqdm(data):
            
            ID = user.get("ID", -1)   
            dic_user_tweetid[ID] = []
            
            tweets = user.get("tweet", [])
      
            for tweet in tweets:
                cleaned_tweet = preprocess_text(tweet)
                
                keywords = extract_keywords(text=cleaned_tweet,NUMBER_OF_KEYWORDS=NUMBER_OF_KEYWORDS)
                for keyword in keywords:
                    all_keywords.add(keyword)
                
                
                tweet_embedding = embedding_model.transform([cleaned_tweet])[0]
                tweet_id = 't'+str(tweetid_counter)
                dic_user_tweetid[ID].append(tweet_id)
                tweetid_counter += 1
                
                dic_tweetid_keywords[tweet_id] = keywords[:]
                
                dic_tweet_id_embedding[tweet_id] = tweet_embedding[:]
    
            label = user.get("label", 'unknown')  
            dic_user_label[ID] = label    
            
            description = user.get("profile", {})     
            description = description['description']
            cleaned_description = preprocess_text(description)
            description_embedding = embedding_model.transform([cleaned_description])[0]
            
            dic_user_description_embedding[ID] = description_embedding[:]
            

OUTPUT_FILES = {
    "dic_user_tweetid": "dic_user_tweetid.txt",
    "dic_user_description_embedding": "dic_user_description_embedding.txt",
    "dic_tweetid_keywords": "dic_tweetid_keywords.txt",
    "dic_tweet_id_embedding": "dic_tweet_id_embedding.txt",
    "all_keywords": "all_keywords.txt",
    "dic_user_label": "dic_user_label.txt",
}

def save_dict_to_file(dictionary, file_name):
    """
    Save a dictionary to a file in a readable format (JSON lines).
    """
    with open(file_name, "w", encoding="utf-8") as file:
        for key, value in dictionary.items():
            file.write(f"{key}: {value}\n")
    print(f"Saved dictionary to {file_name}")

def save_set_to_file(data_set, file_name):
    """
    Save a set to a file, one element per line.
    """
    with open(file_name, "w", encoding="utf-8") as file:
        for item in data_set:
            file.write(f"{item}\n")
    print(f"Saved set to {file_name}")


save_dict_to_file(dic_user_tweetid, OUTPUT_FILES["dic_user_tweetid"])
save_dict_to_file(dic_user_description_embedding, OUTPUT_FILES["dic_user_description_embedding"])
save_dict_to_file(dic_tweetid_keywords, OUTPUT_FILES["dic_tweetid_keywords"])
save_dict_to_file(dic_tweet_id_embedding, OUTPUT_FILES["dic_tweet_id_embedding"])
save_dict_to_file(dic_user_label, OUTPUT_FILES["dic_user_label"])
save_set_to_file(all_keywords, OUTPUT_FILES["all_keywords"])         
            
     
print("finish")
zzzzz
       
            
            
            
        
# --------------------
# Helper Functions
# --------------------



# --------------------
# Part 1: Keyword Extraction with YAKE
# --------------------

                        

             

# --------------------
# Part 2: FastText Embedding + PCA
# --------------------



def build_tweet_dictionary(embedding_model):
    """
    Read tweet texts from CSVs, preprocess them, compute embeddings,
    and store in a dictionary: {user_id: [ {tweet_id: embedding}, ... ] }.
    """
    tweets_users_dic = {}
    ts_id = 0

    for folder_name in FOLDER_NAMES:
        path = f"{DATASET_PATH}/{folder_name}/tweets.csv"
        try:
            with open(path, mode='r', encoding='utf-8', errors='ignore') as file:
                reader = csv.DictReader(file)
                for row in tqdm(reader, desc=f"Processing {folder_name} for tweets"):
                    try:
                        user = row["user_id"]
                        text = preprocess_text(row["text"])
                        embedding = embedding_model.transform([text])

                        if user not in tweets_users_dic:
                            tweets_users_dic[user] = []
                        tweets_users_dic[user].append({'t' + str(ts_id): list(embedding.values())[0]})
                        ts_id += 1

                    except KeyError as e:
                        print(f"Missing key in row for file {folder_name}: {e}")
                        continue
                    except Exception as e:
                        print(f"Error processing row in {path}: {e}")
                        continue
        except FileNotFoundError:
            print(f"File {path} not found!")
        except Exception as e:
            print(f"Error processing file {path}: {e}")

    # Save
    with open(TWEETS_USERS_DIC_PATH, 'wb') as f:
        pickle.dump(tweets_users_dic, f)
    print(f"Dictionary saved to {TWEETS_USERS_DIC_PATH}")

    # Build a single dictionary for all tweet embeddings: {tweet_id: embedding}
    tweet_id_embedding_dic = {}
    for user_id in tqdm(tweets_users_dic.keys(), desc="Collecting tweet embeddings into a single dictionary"):
        for subrow in tweets_users_dic[user_id]:
            k = list(subrow.keys())[0]
            tweet_id_embedding_dic[k] = subrow[k]

    with open(TWEET_ID_EMBEDDING_DIC_PATH, 'wb') as handle:
        pickle.dump(tweet_id_embedding_dic, handle)

    return tweets_users_dic, tweet_id_embedding_dic


def build_id_label_dict():
    """
    Build dictionary of {user_id: label} from CSVs.
    Label is 'benign' for E13.csv/TFP.csv and 'sybil' for others.
    """
    id_label_dict = {}
    for folder_name in FOLDER_NAMES:
        path = f"{DATASET_PATH}/{folder_name}/users.csv"
        try:
            with open(path, mode='r', encoding='utf-8', errors='ignore') as file:
                reader = csv.DictReader(file)
                for row in tqdm(reader, desc=f"Processing {folder_name} for labels"):
                    user_id = row["id"]
                    if folder_name in ['E13.csv', 'TFP.csv']:
                        id_label_dict[user_id] = 'benign'
                    else:
                        id_label_dict[user_id] = 'sybil'
        except FileNotFoundError:
            print(f"File {path} not found!")
        except Exception as e:
            print(f"Error processing file {path}: {e}")

    with open(ID_LABEL_DICT_PATH, 'wb') as f:
        pickle.dump(id_label_dict, f)
    return id_label_dict


# --------------------
# Part 4: Build the Heterogeneous Graph
# --------------------
def build_graph():
    """
    Build a MultiDiGraph with:
      - user nodes
      - keyword nodes
      - tweet nodes

    Edges:
      - follower/following
      - friend
      - hastweet/tweetedby
      - haskeyword/keywordintweet
    """
    # Load required pickles
    with open(KEYWORDS_USERS_DIC_PATH, 'rb') as handle:
        keywords_users_dic = pickle.load(handle)

    with open(TWEETS_USERS_DIC_PATH, 'rb') as handle:
        tweets_users_dic = pickle.load(handle)

    with open(ID_DESCRIPTION_DIC_PATH, 'rb') as handle:
        embeddings_id_description_dic = pickle.load(handle)

    with open(WORD_EMBEDDING_DICTIONARY_PATH, 'rb') as handle:
        word_embedding_dictionary = pickle.load(handle)

    with open(TWEET_ID_EMBEDDING_DIC_PATH, 'rb') as handle:
        tweet_id_embedding_dic = pickle.load(handle)

    # Build user label dict
    with open(ID_LABEL_DICT_PATH, 'rb') as f:
        id_label_dict = pickle.load(f)

    # 1. Prepare node sets
    all_keywords = list(word_embedding_dictionary.keys())
    all_users = list(embeddings_id_description_dic.keys())

    # Determine last tweet number from tweet_id_embedding_dic
    # tweet keys are like 't0', 't1' ...
    temp_tweet_ids = [int(i[1:]) for i in list(tweet_id_embedding_dic.keys())]
    last_tweet_number = max(temp_tweet_ids)

    # Create a MultiDiGraph
    HG = nx.MultiDiGraph()

    # 2. Add user nodes
    for user in all_users:
        HG.add_node(user, node_type="user")

    # 3. Add keyword nodes
    for kw in tqdm(all_keywords, desc="Adding keyword nodes"):
        HG.add_node(kw, node_type="keyword")

    # 4. Add tweet nodes
    for i in tqdm(range(0, last_tweet_number + 1), desc="Adding tweet nodes"):
        HG.add_node('t' + str(i), node_type="tweet")

    # 5. Add edges: follower / following
    set_all_user = set(all_users)
    for folder_name in FOLDER_NAMES:
        path_followers = f"{DATASET_PATH}/{folder_name}/followers.csv"
        try:
            with open(path_followers, 'r') as csv_file:
                reader = csv.reader(csv_file)
                next(reader)  # Skip the header line
                for edge in tqdm(reader, desc=f"Adding follower edges from {folder_name}"):
                    if edge[0] not in set_all_user:
                        HG.add_node(edge[0], node_type="user")
                        set_all_user.add(edge[0])
                    if edge[1] not in set_all_user:
                        HG.add_node(edge[1], node_type="user")
                        set_all_user.add(edge[1])
                    HG.add_edge(edge[0], edge[1], edge_type="follower")
                    HG.add_edge(edge[1], edge[0], edge_type="following")
        except FileNotFoundError:
            pass  # Some folders may not have this file

    # 6. Add edges: friend
    for folder_name in FOLDER_NAMES:
        path_friends = f"{DATASET_PATH}/{folder_name}/friends.csv"
        try:
            with open(path_friends, 'r') as csv_file:
                reader = csv.reader(csv_file)
                next(reader)  # Skip the header line
                for edge in tqdm(reader, desc=f"Adding friend edges from {folder_name}"):
                    if edge[0] not in set_all_user:
                        HG.add_node(edge[0], node_type="user")
                        set_all_user.add(edge[0])
                    if edge[1] not in set_all_user:
                        HG.add_node(edge[1], node_type="user")
                        set_all_user.add(edge[1])
                    HG.add_edge(edge[0], edge[1], edge_type="friend")
                    HG.add_edge(edge[1], edge[0], edge_type="friend")
        except FileNotFoundError:
            pass

    # 7. Add edges: user -> tweet (hastweet / tweetedby)
    nodes = set(list(HG.nodes()))
    for user in tqdm(all_users, desc="Adding user->tweet edges"):
        if user not in tweets_users_dic:
            continue
        tweets = tweets_users_dic[user]
        for row in tweets:
            tweet = list(row.keys())[0]
            if user in nodes and tweet in nodes:
                HG.add_edge(user, tweet, edge_type="hastweet")
                HG.add_edge(tweet, user, edge_type="tweetedby")

    # 8. Add edges: keyword -> tweet (keywordintweet / haskeyword)
    for user in tqdm(keywords_users_dic.keys(), desc="Adding keyword->tweet edges"):
        for row in keywords_users_dic[user]:
            tweet = list(row.keys())[0]
            keywords = row[tweet]
            for kw in keywords:
                if kw in nodes and tweet in nodes:
                    HG.add_edge(kw, tweet, edge_type="keywordintweet")
                    HG.add_edge(tweet, kw, edge_type="haskeyword")

    # Save intermediate graph
    with open(HG_PATH_0, 'wb') as handle:
        pickle.dump(HG, handle)

    return HG


# --------------------
# Part 5: Assign Node Embeddings
# --------------------
def assign_node_embeddings(HG):
    """
    Assign embeddings to each node in the heterogeneous graph:
      - user nodes: from user description embedding
      - keyword nodes: from word embedding dictionary
      - tweet nodes: from tweet embedding dictionary

    If a user node has no embedding (missing description), 
    compute the average of neighbor embeddings.
    """
    with open(TWEET_ID_EMBEDDING_DIC_PATH, 'rb') as handle:
        tweet_id_embedding_dic = pickle.load(handle)

    with open(ID_DESCRIPTION_DIC_PATH, 'rb') as handle:
        embeddings_id_description_dic = pickle.load(handle)

    with open(WORD_EMBEDDING_DICTIONARY_PATH, 'rb') as handle:
        word_embedding_dictionary = pickle.load(handle)

    with open(ID_LABEL_DICT_PATH, 'rb') as handle:
        id_label_dict = pickle.load(handle)

    # 1. Assign embeddings to relevant nodes
    nodes_without_description = []
    embedding_dim = len(next(iter(word_embedding_dictionary.values())))

    for node, data in HG.nodes(data=True):
        ntype = data["node_type"]
        if ntype == "keyword":
            data["embedding"] = word_embedding_dictionary[node]
        elif ntype == "tweet":
            data["embedding"] = tweet_id_embedding_dic[node]
        elif ntype == "user":
            data["label"] = id_label_dict.get(node, 'unknown')
            if node in embeddings_id_description_dic:
                data["embedding"] = embeddings_id_description_dic[node]
            else:
                # Mark user node for neighbor-based embedding
                nodes_without_description.append((node, data))

    # 2. For users without embeddings, average neighbor embeddings
    for node, data in nodes_without_description:
        neighbors = list(HG.neighbors(node))
        neighbor_embeddings = []
        for neigh in neighbors:
            neigh_data = HG.nodes[neigh]
            if "embedding" in neigh_data:
                neighbor_embeddings.append(neigh_data["embedding"])
        if neighbor_embeddings:
            avg_embedding = np.mean(neighbor_embeddings, axis=0)
            noise = np.random.normal(loc=0, scale=0.01, size=avg_embedding.shape)
            data["embedding"] = avg_embedding + noise
        else:
            # If no neighbors have embeddings, assign random embedding
            data["embedding"] = np.random.normal(loc=0, scale=0.1, size=embedding_dim)

    # 3. Save final graph
    with open(HG_PATH_2, 'wb') as handle:
        pickle.dump(HG, handle)

    return HG


# --------------------
# Part 6: Convert Node IDs & Save to CSV
# --------------------
def save_graph_data(HG):
    """
    Convert heterogeneous graph node IDs to a range [0 ... n-1],
    downcast embeddings to float32, and save to CSV:
      - node_embeddings.csv
      - node_labels.csv
      - node_information.csv
      - edges.csv
    """
    # Extract nodes and sort by node_type (for consistency)
    node_list = [{"node": node, "attributes": data} for node, data in HG.nodes(data=True)]
    node_list = sorted(node_list, key=lambda x: x["attributes"].get("node_type", ""))

    # Assign new IDs and downcast embeddings
    node_mapping = {}
    updated_nodes = []
    for new_id, node_data in enumerate(tqdm(node_list, desc="Assigning new IDs and downcasting embeddings")):
        old_id = node_data["node"]
        node_mapping[old_id] = new_id

        # Downcast embeddings to float32
        if "embedding" in node_data["attributes"]:
            embedding = node_data["attributes"]["embedding"]
            if isinstance(embedding, list):
                embedding = np.array(embedding, dtype=np.float32).tolist()
            elif isinstance(embedding, np.ndarray):
                embedding = embedding.astype(np.float32).tolist()
            node_data["attributes"]["embedding"] = embedding

        updated_node = {
            "node_id": new_id,
            "old_id": old_id,
            "attributes": node_data["attributes"]
        }
        updated_nodes.append(updated_node)

    # Save node embeddings
    with open(NODE_EMBEDDINGS_CSV, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['node_id', 'embedding'])
        for node in updated_nodes:
            embedding = node["attributes"].get("embedding", [])
            writer.writerow([node["node_id"], ','.join(map(str, embedding))])

    # Save node labels
    with open(NODE_LABELS_CSV, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['node_id', 'label'])
        for node in updated_nodes:
            label = node["attributes"].get("label", None)
            writer.writerow([node["node_id"], label])

    # Save other node information
    with open(NODE_INFO_CSV, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['node_id', 'old_id', 'attributes'])
        for node in updated_nodes:
            # Exclude embedding from attributes
            attributes = {k: v for k, v in node["attributes"].items() if k != "embedding"}
            writer.writerow([node["node_id"], node["old_id"], attributes])

    # Save edges
    with open(EDGES_CSV, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['source', 'target', 'attributes'])
        for old_u, old_v, attributes in tqdm(HG.edges(data=True), desc="Saving edges"):
            new_u = node_mapping.get(old_u, old_u)
            new_v = node_mapping.get(old_v, old_v)
            writer.writerow([new_u, new_v, attributes])

    # Shuffle edges file (optional, if desired)
    with open(EDGES_CSV, 'r') as infile:
        header = infile.readline()
        rows = infile.readlines()
    random.shuffle(rows)
    with open(EDGES_CSV, 'w') as outfile:
        outfile.write(header)
        outfile.writelines(rows)
    print("Edges shuffled and saved.")


# --------------------
# Part 7: Statistics
# --------------------
def print_graph_statistics(HG):
    """
    Print some statistics (counts) of node types, labels, edge types.
    """
    node_types_counter = Counter()
    node_labels_counter = Counter()

    for node, data in HG.nodes(data=True):
        node_type = data.get("node_type", "Unknown")
        node_label = data.get("label", "NoLabel")
        node_types_counter[node_type] += 1
        node_labels_counter[node_label] += 1

    edge_types_counter = Counter()
    for _, _, attributes in HG.edges(data=True):
        edge_type = attributes.get("edge_type", "Unknown")
        edge_types_counter[edge_type] += 1

    # Print node type information
    print("\nNode Types and Counts:")
    for node_type, count in node_types_counter.items():
        print(f"Type: {node_type}, Count: {count}")

    # Print node label information
    print("\nNode Labels and Counts:")
    for label, count in node_labels_counter.items():
        print(f"Label: {label}, Count: {count}")

    # Print edge type information
    print("\nEdge Types and Counts:")
    for edge_type, count in edge_types_counter.items():
        print(f"Type: {edge_type}, Count: {count}")

    print("\nSummary:")
    print(f"Total Node Types: {len(node_types_counter)}")
    print(f"Total Edge Types: {len(edge_types_counter)}")
    print(f"Total Node Labels: {len(node_labels_counter)}")


# --------------------
# Main Execution Flow
# --------------------
if __name__ == "__main__":
    # 1. Extract Keywords
    all_keywords, keywords_users_dic = extract_keywords()

    # 2. Load FastText and build embedding model
    print("Loading FastText model...")
    fasttext_model = load_facebook_vectors(FASTTEXT_BIN_PATH)
    embedding_model = FastTextEmbeddingModel(fasttext_model, target_size=32)

    # 3. Fit PCA on all keywords
    embedding_model.fit_and_save(all_keywords, pca_save_path=PCA_MODEL_PATH)

    # 4. Transform all_keywords to get word embeddings
    keywords_embedding = embedding_model.transform(all_keywords)
    with open(WORD_EMBEDDING_DICTIONARY_PATH, 'wb') as f:
        pickle.dump(keywords_embedding, f)
    print("Saved word embedding dictionary.")

    # 5. Build description dictionary and embed
    embeddings_id_description_dic = build_description_dictionary(embedding_model)

    # 6. Build tweet dictionary and embed
    tweets_users_dic, tweet_id_embedding_dic = build_tweet_dictionary(embedding_model)

    # 7. Build user label dict
    id_label_dict = build_id_label_dict()

    # 8. Build the heterogeneous graph
    HG = build_graph()

    # 9. Assign node embeddings
    HG = assign_node_embeddings(HG)

    # 10. Convert node IDs and save data
    save_graph_data(HG)

    # 11. Print statistics
    print_graph_statistics(HG)

    print("All tasks completed.")
