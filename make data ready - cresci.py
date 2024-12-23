# -*- coding: utf-8 -*-
"""Untitled27.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ZMLsTkwJado-1yLshxY6VBSwEf-IXS13

#install phase
"""

# !pip install yake

# !pip install fasttext

"""# imports"""

import json
import csv
import re
import numpy as np
import random
import networkx as nx
from matplotlib import pyplot as plt
from tqdm import tqdm
import pickle

import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, Linear
from sklearn.model_selection import train_test_split
from collections import defaultdict

import yake

from gensim.models.fasttext import load_facebook_vectors
from sklearn.decomposition import PCA,IncrementalPCA

SEED = 1



"""# cresci Keywords"""
'''
yake_extractor = yake.KeywordExtractor(lan="en", n=1, dedupLim=0.9)


tweet_keywords = {}
all_keywords = set()


## keyword extraction for cresci dataset
keywords_users_dic = {}
ts_id = 0
number_of_keywords = 10

def preprocess_tweet(text):
    text = re.sub(r"@\w+", "mention", text)
    
    # Replace hashtags with "hashtags"
    text = re.sub(r"#\w+", "hashtags", text)
    
    # Replace links with "link"
    text = re.sub(r"https?://\S+|www\.\S+", "link", text)
    
    # Remove non-English characters and punctuation
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove any extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


for folder_name in ['E13.csv', 'FSF.csv', 'INT.csv', 'TFP.csv', 'TWT.csv']:
    path = f"Fake_project_dataset_csv/{folder_name}/tweets.csv"
    
    try:
        with open(path, mode='r', encoding='utf-8', errors='ignore') as file:
            reader = csv.DictReader(file)
            for row in tqdm(reader, desc=f"Processing {folder_name}"):
                try:
                    user = row["user_id"]
                    text = preprocess_tweet(row["text"])

                    # Extract keywords
                    keywords = [kw[0] for kw in yake_extractor.extract_keywords(text)[:number_of_keywords]]

                    # Update dictionary
                    if user in keywords_users_dic:
                        keywords_users_dic[user].append({'t'+str(ts_id): keywords[:]})
                    else:
                        keywords_users_dic[user] = [{'t'+str(ts_id): keywords[:]}]

                    ts_id += 1
                    all_keywords.update(keywords[:])
                    
                    # Optional: Break every 5 iterations for debugging or testing
                    # if ts_id % 5 == 4:
                        # break

                except Exception as e:
                    print(f"Error processing row in {path}: {e}")
                    continue  # Skip problematic rows
                
        print(f"{path} is done")
    
    except FileNotFoundError:
        print(f"File {path} not found!")
    except Exception as e:
        print(f"Error processing file {path}: {e}")

with open('all_keywords.pickle', 'wb') as handle:
    pickle.dump(all_keywords, handle)

with open('keywords_users_dic.pickle', 'wb') as handle:
    pickle.dump(keywords_users_dic, handle)




# cresci embedding
"""#embedding"""

with open('all_keywords.pickle', 'rb') as handle:
    all_keywords = pickle.load(handle)



class FastTextEmbeddingModel:
    def __init__(self, fasttext_model, target_size=64):
        """
        Initialize the embedding model with a FastText model and PCA configuration.
        """
        self.model = fasttext_model
        self.target_size = target_size
        self.pca = None

    def _compute_word_embedding(self, word):
        """
        Fetch embedding for a single word. If the word is not in the model, return a zero vector.
        """
        if word in self.model:
            return self.model[word]
        else:
            return np.zeros(self.model.vector_size)

    def _compute_sentence_embedding(self, sentence):
        """
        Compute the embedding for a sentence by averaging the embeddings of its words.
        """
        words = sentence.split()  # Tokenize the sentence
        embeddings = [self._compute_word_embedding(word) for word in words if word in self.model]
        if embeddings:
            return np.mean(embeddings, axis=0)
        else:
            return np.zeros(self.model.vector_size)

    def fit_and_save(self, inputs, pca_save_path, batch_size=10000):
        """
        Fit PCA incrementally to handle large datasets and save the trained PCA model.
        """
        # Convert the inputs to a list if it's not already one
        if isinstance(inputs, set):
            inputs = list(inputs)

        self.pca = IncrementalPCA(n_components=self.target_size)

        # Process inputs in batches to reduce memory usage
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i+batch_size]
            embeddings = [self._compute_embedding(text) for text in batch]
            self.pca.partial_fit(embeddings)

        # Save the PCA model
        with open(pca_save_path, 'wb') as f:
            pickle.dump(self.pca, f)
        print(f"PCA model trained and saved to {pca_save_path}")

    def load_pca(self, pca_load_path):
        """
        Load a pre-trained PCA model from a file.
        """
        with open(pca_load_path, 'rb') as f:
            self.pca = pickle.load(f)
        print(f"PCA model loaded from {pca_load_path}")

    def transform(self, inputs):
        """
        Transform inputs (list of words or sentences) into reduced-dimensional embeddings.
        """
        if not self.pca:
            raise ValueError("PCA is not fitted or loaded. Fit PCA or load a model before transforming.")
        
        embeddings = [self._compute_embedding(text) for text in inputs]
        reduced_embeddings = self.pca.transform(embeddings)
        return {text: reduced_embeddings[i] for i, text in enumerate(inputs)}

    def transform_dictionary(self, inputs):
        """
        Transform inputs (dictionary with keys and sentences/words as values) into reduced embeddings.
        """
        if not self.pca:
            raise ValueError("PCA is not fitted or loaded. Fit PCA or load a model before transforming.")
        
        embeddings = [self._compute_embedding(text) for text in inputs.values()]
        reduced_embeddings = self.pca.transform(embeddings)
        return {key: reduced_embeddings[i] for i, key in enumerate(inputs.keys())}
    
    def _compute_embedding(self, text):
        """
        Determine if the input is a sentence or a word and compute its embedding.
        """
        if " " in text:  # Sentence
            return self._compute_sentence_embedding(text)
        else:  # Word
            return self._compute_word_embedding(text)


fasttext_model = load_facebook_vectors('cc.en.300.bin')
embedding_model = FastTextEmbeddingModel(fasttext_model, target_size=32)
embedding_model.fit_and_save(all_keywords, pca_save_path="pca_model.pkl")
keywords_embedding = embedding_model.transform(all_keywords)

with open('word_embedding_dictionary.pickle', 'wb') as f:
    pickle.dump(keywords_embedding, f)
print("keywords_embedding done")




tweets_sentences_dic = {}


def preprocess_description(text):
    # Replace mentions with "mention"
    text = re.sub(r"@\w+", "mention", text)
    
    # Replace hashtags with "hashtags"
    text = re.sub(r"#\w+", "hashtags", text)
    
    # Replace links with "link"
    text = re.sub(r"https?://\S+|www\.\S+", "link", text)
    
    # Remove non-English characters and punctuation
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove any extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


# List of folders or files to process
folder_names = ['E13.csv', 'FSF.csv', 'INT.csv', 'TFP.csv', 'TWT.csv']

id_description_dict = {}

for folder_name in folder_names:
    path = f"Fake_project_dataset_csv/{folder_name}/users.csv"  # Assuming the file contains user data
    
    try:
        # Open the file for reading
        with open(path, mode='r', encoding='utf-8', errors='ignore') as file:
            reader = csv.DictReader(file)
            
            for row in tqdm(reader, desc=f"Processing {folder_name}"):
                try:
                    user_id = row["id"]
                    description = preprocess_description(row["description"]) if row["description"] else ""
                    
                    # Add to dictionary
                    id_description_dict[user_id] = description
                
                except KeyError as e:
                    print(f"Missing key in row for file {folder_name}: {e}")
                    continue
                except Exception as e:
                    print(f"Error processing row in {path}: {e}")
                    continue  # Skip problematic rows
            
        print(f"{path} is done")
    
    except FileNotFoundError:
        print(f"File {path} not found!")
    except Exception as e:
        print(f"Error processing file {path}: {e}")






embeddings_id_description_dic = embedding_model.transform_dictionary(id_description_dict)
with open('embeddings_id_description_dic.pickle', 'wb') as f:
    pickle.dump(embeddings_id_description_dic, f)
print("embeddings_id_description_dic done")



tweets_users_dic = {}
ts_id = 0  # Timestamp or unique identifier for tweets

for folder_name in ['E13.csv', 'FSF.csv', 'INT.csv', 'TFP.csv', 'TWT.csv']:
    path = f"Fake_project_dataset_csv/{folder_name}/tweets.csv"
    try:
        with open(path, mode='r', encoding='utf-8', errors='ignore') as file:
            reader = csv.DictReader(file)
            for row in tqdm(reader, desc=f"Processing {folder_name}"):
                try:
                    # Extract user ID and preprocess text
                    user = row["user_id"]
                    text = preprocess_description(row["text"])

                    # Compute embedding for the preprocessed text
                    embedding = embedding_model.transform([text])

                    # Update dictionary
                    if user in tweets_users_dic:
                        tweets_users_dic[user].append({'t' + str(ts_id):  list(embedding.values())[0]})
                    else:
                        tweets_users_dic[user] = [{'t' + str(ts_id):  list(embedding.values())[0]}]

                    
                    ts_id += 1  # Increment timestamp ID

                except KeyError as e:
                    print(f"Missing key in row for file {folder_name}: {e}")
                    continue
                except Exception as e:
                    print(f"Error processing row in {path}: {e}")
                    continue  # Skip problematic rows
            
        print(f"{path} is done")
    
    except FileNotFoundError:
        print(f"File {path} not found!")
    except Exception as e:
        print(f"Error processing file {path}: {e}")


# def print_sample(dictionary, sample_size=3):
#     if dictionary:
#         sample = {k: dictionary[k] for k in list(dictionary)[:sample_size]}
#         print(f"Sample from dictionary ({len(dictionary)} entries):\n{sample}\n")
# print_sample(tweets_users_dic, sample_size=5) 

# Optional: Save the dictionary for future use
output_path = "tweets_users_dic.pkl"
try:
    with open(output_path, 'wb') as f:
        pickle.dump(tweets_users_dic, f)
    print(f"Dictionary saved to {output_path}")
except Exception as e:
    print(f"Error saving dictionary: {e}")

'''

'''
with open('tweets_users_dic.pkl', 'rb') as handle:
    tweets_users_dic = pickle.load(handle)

tweet_id_embedding_dic = {}
for row in tqdm(tweets_users_dic.keys()):
    for subrow in tweets_users_dic[row]:
        k = list(subrow.keys())[0]
        tweet_id_embedding_dic[k] = subrow[k]

with open('tweet_id_embedding_dic.pickle', 'wb') as handle:
    pickle.dump(tweet_id_embedding_dic, handle)
'''

    

"""# creat graph"""
'''
with open('keywords_users_dic.pickle', 'rb') as handle:
    keywords_users_dic = pickle.load(handle)



with open('tweets_users_dic.pkl', 'rb') as handle:
    tweets_users_dic = pickle.load(handle)

with open('embeddings_id_description_dic.pickle', 'rb') as handle:
    embeddings_id_description_dic = pickle.load(handle)

with open('word_embedding_dictionary.pickle', 'rb') as handle:
    word_embedding_dictionary = pickle.load(handle)

with open('tweet_id_embedding_dic.pickle', 'rb') as handle:
    tweet_id_embedding_dic = pickle.load(handle)






# def print_sample(dictionary, sample_size=3):
#     if dictionary:
#         sample = {k: dictionary[k] for k in list(dictionary)[:sample_size]}
#         print(f"Sample from dictionary ({len(dictionary)} entries):\n{sample}\n")
#     else:
#         print("Dictionary is not available.\n")

# Print samples from each dictionary
# print("Keywords Users Dictionary:")
# print_sample(tweets_users_dic)

# print("Embeddings ID Description Dictionary:")
# print_sample(embeddings_id_description_dic)

# print("Embeddings keywords_embedding:")
# print_sample(word_embedding_dictionary)




all_keywords = list(word_embedding_dictionary.keys())
all_users = list(embeddings_id_description_dic.keys())

temp = [int(i[1:]) for i in  list(tweet_id_embedding_dic.keys())]
# print(max(temp),min(temp),len(temp))
last_tweet_number = max(temp)

HG = nx.MultiDiGraph()

# Add nodes of different types include 'user', 'keyword', 'tweet'
for node in list(all_users):
    HG.add_node(node, node_type="user")
for node in tqdm(all_keywords):
    HG.add_node(node, node_type="keyword")

print(' ----------------------          ',"251603425" in HG.nodes)

print("last_tweet_number")
for row in tqdm(range(0,last_tweet_number+1)):
    HG.add_node('t'+str(row), node_type="tweet")

print(' ----------------------          ',"251603425" in HG.nodes)

print("nodes added")

# Add directed edges with types include 'following', 'follower', 'friend', 'haskeyword', 'keywordintweet', 'hastweet', 'tweetedby'
for folder_name in ['E13.csv', 'FSF.csv', 'INT.csv', 'TFP.csv', 'TWT.csv']:

    path = f"Fake_project_dataset_csv/{folder_name}/followers.csv"
    with open(path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skip the header line
        set_all_user = set(all_users)
        for edge in tqdm(reader):
            if edge[0] not in  set_all_user:
                HG.add_node(edge[0], node_type="user")
                set_all_user.add(edge[0])
            if edge[1] not in  set_all_user:
                HG.add_node(edge[1], node_type="user")
                set_all_user.add(edge[1])

            HG.add_edge(edge[0], edge[1], edge_type="follower")
            HG.add_edge(edge[1], edge[0], edge_type="following")
print(' ----------------------          ',"251603425" in HG.nodes)

for folder_name in ['E13.csv', 'FSF.csv', 'INT.csv', 'TFP.csv', 'TWT.csv']:
    path = f"Fake_project_dataset_csv/{folder_name}/friends.csv"
    with open(path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skip the header line
        for edge in reader:
            if edge[0] not in  set_all_user:
                HG.add_node(edge[0], node_type="user")
                set_all_user.add(edge[0])
            if edge[1] not in  set_all_user:
                HG.add_node(edge[1], node_type="user")
                set_all_user.add(edge[1])
            HG.add_edge(edge[0], edge[1], edge_type="friend")
            HG.add_edge(edge[1], edge[0], edge_type="friend")

print(' ----------------------          ',"251603425" in HG.nodes)

nodes = set(list(HG.nodes()))



for user in tqdm(all_users):

    try:
        tweets = tweets_users_dic[user]
        for row in tweets:
            tweet = list(row.keys())[0]
            if user in nodes and tweet in nodes:
                HG.add_edge(user, tweet, edge_type="hastweet")
                HG.add_edge(tweet, user, edge_type="tweetedby")          
            else:
                print(user,tweet, user in nodes , tweet in nodes)
                break

    except:
        print("no tweet by ",user)
        
print(' ----------------------          ',"251603425" in HG.nodes)



for user in tqdm(keywords_users_dic.keys()):
    for row in keywords_users_dic[user]:
        tweet = list(row.keys())[0]
        keywords = row[tweet]
        for kw in keywords:
            if kw in nodes and tweet in nodes:  
                HG.add_edge(kw, tweet, edge_type="keywordintweet")
                HG.add_edge(tweet, kw, edge_type="haskeyword")
            else:
                print('sssssssss   ',kw,tweet, kw in nodes , tweet in nodes)
                break

print(' ----------------------          ',"251603425" in HG.nodes)

with open('HG0.pickle', 'wb') as handle:
    pickle.dump(HG, handle)





 # convertion in dictionary of tweets, remove user keys and now it is just tweetid and tweet

# with open('tweets_users_dic.pkl', 'rb') as handle:
#     tweets_users_dic = pickle.load(handle)
# print("read twt")



'''




'''

with open('tweet_id_embedding_dic.pickle', 'rb') as handle:
    tweet_id_embedding_dic = pickle.load(handle)

with open('embeddings_id_description_dic.pickle', 'rb') as handle:
    embeddings_id_description_dic = pickle.load(handle)
print("read desc")

with open('tweets_users_dic.pkl', 'rb') as handle:
    tweets_users_dic = pickle.load(handle)
print("read twt")

with open('word_embedding_dictionary.pickle', 'rb') as handle:
    word_embedding_dictionary = pickle.load(handle)
print("read wrd")

with open('HG0.pickle', 'rb') as handle:
    HG = pickle.load(handle)
print("read gh")


all_users = [int(i) for i in list(embeddings_id_description_dic.keys())]
print(len(all_users),min(all_users),max(all_users))






# def print_sample(dictionary, sample_size=1):
#     sample = {k: dictionary[k] for k in list(dictionary)[:sample_size]}
#     print(f"Sample from dictionary ({len(dictionary)} entries):\n{sample}\n")
   


# node_types = set()
# for n, data in HG.nodes(data=True):
#     # Assumes node data includes a 'type' attribute, e.g. data['type']
#     if 'type' in data:
#         node_types.add(data['type'])

# print("Node types found in the graph:", node_types)

# # 2. Print a small sample of nodes for each type
# for nt in node_types:
#     # collect nodes of this type
#     nodes_of_type = [n for n, d in HG.nodes(data=True) if d.get('type') == nt]
#     sample_nodes = nodes_of_type[:5]  # take up to 5 nodes as a sample
#     print(f"Sample nodes of type '{nt}':", sample_nodes)

# # 3. Extract the unique edge types
# edge_types = set()
# for u, v, key, data in HG.edges(keys=True, data=True):
#     # Assumes edge data includes a 'type' attribute, e.g. data['type']
#     if 'type' in data:
#         edge_types.add(data['type'])

# print("\nEdge types found in the graph:", edge_types)

# # 4. Print a small sample of edges for each type
# for et in edge_types:
#     # collect edges of this type
#     edges_of_type = [(u, v, k) for u, v, k, d in HG.edges(keys=True, data=True) if d.get('type') == et]
#     sample_edges = edges_of_type[:5]  # take up to 5 edges as a sample
#     print(f"Sample edges of type '{et}':", sample_edges)

# print_sample(embeddings_id_description_dic)
# print_sample(tweets_users_dic)
# print_sample(word_embedding_dictionary)


nodes = set(list(HG.nodes()))

print(' ----------------------          ',"12" in HG.nodes)



nodes_without_description = []

# Assign embeddings to keyword nodes
for node, data in tqdm(HG.nodes(data=True)):

    if data["node_type"] == "keyword":
        keyword = node
        data["embedding"] = word_embedding_dictionary[keyword]
    if data["node_type"] == "user":
        user = node
        try:
            data["embedding"] = embeddings_id_description_dic[user]
        except:
            nodes_without_description.append((node,data))
    if data["node_type"] == "tweet":
        tweet = node
        data["embedding"] = tweet_id_embedding_dic[tweet]


print("terreterrtete")


embedding_dim = len(next(iter(word_embedding_dictionary.values())))

for node, data in tqdm(nodes_without_description):
    neighbors = list(HG.neighbors(node))
    
    neighbor_embeddings = []
    for neigh in neighbors:
        neigh_data = HG.nodes[neigh]
        if "embedding" in neigh_data:
            neighbor_embeddings.append(neigh_data["embedding"])
    
    if len(neighbor_embeddings) > 0:
        # Compute average embedding of neighbors
        avg_embedding = np.mean(neighbor_embeddings, axis=0)
        
        # Add small random noise to the embedding to make it slightly distinct
        noise = np.random.normal(loc=0, scale=0.01, size=avg_embedding.shape)
        data["embedding"] = avg_embedding + noise
  
    else:
        # If no neighbors have embeddings, default to a random embedding
        # (You can also choose another strategy here)
        data["embedding"] = np.random.normal(loc=0, scale=0.1, size=embedding_dim)
 
# At this point, all nodes in nodes_without_description should have embeddings

with open('HG2.pickle', 'wb') as handle:
    pickle.dump(HG, handle)
'''



## nopde renaming to 0 to n-1

##### RAM intensive
# def relabel_heterogeneous_graph(HG):
#     # Extract nodes and create a mapping
#     old_to_new = {node: idx for idx, node in enumerate(HG.nodes())}
#     new_to_old = {idx: node for node, idx in old_to_new.items()}
    
#     # Relabel nodes in the graph
#     HG_relabelled = nx.relabel_nodes(HG, old_to_new)
    
#     return HG_relabelled, old_to_new, new_to_old

# HG, old_to_new, new_to_old = relabel_heterogeneous_graph(HG)


'''
# Load the heterogeneous graph
with open('HG2.pickle', 'rb') as handle:
    HG = pickle.load(handle)

# Extract nodes and sort them
print("Extracting and sorting nodes...")
node_list = [{"node": node, "attributes": attributes} for node, attributes in tqdm(HG.nodes(data=True))]
node_list = sorted(node_list, key=lambda x: x["attributes"].get("node_type", ""))

# Assign new IDs to nodes and downcast embeddings to float32
print("Assigning new IDs to nodes and downcasting embeddings...")
node_mapping = {}
updated_nodes = []
for new_id, node_data in tqdm(enumerate(node_list), total=len(node_list)):
    old_id = node_data["node"]
    node_mapping[old_id] = new_id
    
    # Downcast embeddings to float32
    if "embedding" in node_data["attributes"]:
        embedding = node_data["attributes"]["embedding"]
        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32).tolist()  # Downcast
        elif isinstance(embedding, np.ndarray):
            embedding = embedding.astype(np.float32).tolist()  # Downcast
        node_data["attributes"]["embedding"] = embedding

    updated_node = {
        "node_id": new_id,
        "old_id": old_id,
        "attributes": node_data["attributes"]
    }
    updated_nodes.append(updated_node)

# Save node embeddings to a CSV file
print("Saving node embeddings...")
with open('node_embeddings.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['node_id', 'embedding'])
    for node in updated_nodes:
        embedding = node["attributes"].get("embedding", [])
        writer.writerow([node["node_id"], ','.join(map(str, embedding))])

# Save other node information to a CSV file
print("Saving node information...")
with open('node_information.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['node_id', 'old_id', 'attributes'])
    for node in updated_nodes:
        attributes = {k: v for k, v in node["attributes"].items() if k != "embedding"}
        writer.writerow([node["node_id"], node["old_id"], attributes])

# Process and save edges in batches
print("Processing and saving edges...")
with open('edges.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['source', 'target', 'attributes'])
    for old_u, old_v, attributes in tqdm(HG.edges(data=True)):
        new_u = node_mapping.get(old_u, old_u)  # Default to old ID if not found
        new_v = node_mapping.get(old_v, old_v)
        writer.writerow([new_u, new_v, attributes])

print("Data saved successfully with efficient edge processing.")


'''






    
    

'''
import csv
import ast
import torch
from torch_geometric.data import HeteroData
from collections import defaultdict
print("step  1")

# -------------------------------------
# Step 1: Read node information
# -------------------------------------
# We'll store for each node:
# - node_type (string)
# - old_id (might be useful if you want to preserve original node ID)
# We'll need this to group nodes by type and to map them later.
node_types = {}     # node_id -> node_type
node_old_ids = {}   # node_id -> old_id

with open('node_information.csv', 'r', newline='') as f:
    reader = csv.DictReader(f)
    for row in tqdm(reader):
        node_id = int(row['node_id'])
        old_id = row['old_id']
        attributes = ast.literal_eval(row['attributes'])
        node_type = attributes.get('node_type', None)

        if node_type is None:
            # If node_type is missing, skip this node (or handle differently)
            continue

        node_types[node_id] = node_type
        node_old_ids[node_id] = old_id
print("step  2")

# -------------------------------------
# Step 2: Read node embeddings
# -------------------------------------
# We'll directly group embeddings by node_type to avoid storing large intermediate structures.
node_features = defaultdict(list)           # node_type -> list of embeddings
node_ids_by_type = defaultdict(list)        # node_type -> list of node_ids in insertion order

with open('node_embeddings.csv', 'r', newline='') as f:
    reader = csv.DictReader(f)
    for row in tqdm(reader):
        node_id = int(row['node_id'])
        embedding_str = row['embedding'].strip()

        # Some nodes may have been skipped if node_type was missing
        if node_id not in node_types:
            continue

        if embedding_str:
            embedding = list(map(float, embedding_str.split(',')))
        else:
            embedding = []

        ntype = node_types[node_id]
        node_features[ntype].append(embedding)
        node_ids_by_type[ntype].append(node_id)
print("step  3")

# -------------------------------------
# Step 3: Create the HeteroData and assign node embeddings
# -------------------------------------
hg = HeteroData()

# Convert grouped embeddings to tensors
for ntype, embeddings in tqdm(node_features.items()):
    hg[ntype].x = torch.tensor(embeddings, dtype=torch.float)

# Create a mapping from node_id to node index per node_type
type_node_index_map = {
    ntype: {nid: i for i, nid in enumerate(node_ids)}
    for ntype, node_ids in tqdm(node_ids_by_type.items())
}
print("step  4")
# -------------------------------------
# Step 4: Add edges to HeteroData
# -------------------------------------
# We'll read each edge and directly insert it into the appropriate edge type block without storing all edges.
with open('edges.csv', 'r', newline='') as f:
    reader = csv.DictReader(f)
    # To minimize memory usage, we process edges one by one.
    # We'll store edge_index incrementally.
    # For efficiency in a large graph, you might first store them in a buffer per edge_type and then create tensors.
    # But here we'll just append them as we go. This might be less memory-efficient if there are a huge number of edges,
    # but it avoids large intermediate structures.
    edge_buffers = defaultdict(lambda: [[], []]) 
    # edge_buffers[(src_type, edge_type, tgt_type)] = [[list_of_source_indices], [list_of_target_indices]]

    for row in tqdm(reader):
        source = int(row['source'])
        target = int(row['target'])
        attributes = ast.literal_eval(row['attributes'])
        edge_type = attributes.get('edge_type', None)

        # If edge_type or node_type missing, skip
        if edge_type is None or source not in node_types or target not in node_types:
            continue

        src_type = node_types[source]
        tgt_type = node_types[target]

        etype = (src_type, edge_type, tgt_type)

        # Convert source and target to local indices
        src_idx = type_node_index_map[src_type][source]
        tgt_idx = type_node_index_map[tgt_type][target]

        edge_buffers[etype][0].append(src_idx)
        edge_buffers[etype][1].append(tgt_idx)

# Convert collected edges into tensors
for etype, (src_list, tgt_list) in tqdm(edge_buffers.items()):
    hg[etype].edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long)

print("Constructed HeteroData with reduced memory usage:")
print(hg)

# HeteroData(
#   keyword={ x=[796590, 32] },
#   tweet={ x=[2827757, 32] },
#   user={ x=[1292763, 32] },
#   (user, following, user)={ edge_index=[2, 1849092] },
#   (user, friend, user)={ edge_index=[2, 4818570] },
#   (user, follower, user)={ edge_index=[2, 1849092] },
#   (user, hastweet, tweet)={ edge_index=[2, 2827757] },
#   (keyword, keywordintweet, tweet)={ edge_index=[2, 19323433] },
#   (tweet, tweetedby, user)={ edge_index=[2, 2827757] },
#   (tweet, haskeyword, keyword)={ edge_index=[2, 19323433] }
# )
'''
