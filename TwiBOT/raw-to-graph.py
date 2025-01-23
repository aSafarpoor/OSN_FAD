import json
import numpy as np
from tqdm import tqdm
import re
import pickle
import random
SEED = 1
random.seed(SEED)

import yake
import networkx as nx

from gensim.models.fasttext import load_facebook_vectors
from sklearn.decomposition import IncrementalPCA
yake_extractor = yake.KeywordExtractor(lan="en", n=1, dedupLim=0.9)

FASTTEXT_BIN_PATH = 'cc.en.300.bin'

# File names and output file
File_Names = ['test.json', 'dev.json', 'train.json', 'support.json']
Node_class = ['test','dev','train',"support"]
NUMBER_OF_KEYWORDS = 10

PCA_MODEL_PATH = 'PCA.pkl'

def read_pickle(name):
    with open(name,'rb') as f:
        temp = pickle.load(f)
    return temp

def write_pickle(variable,name):
    with open(name , 'wb') as f:
        pickle.dump(variable, f)

##############################################
## Part 1: Removing nodes without tweet or without neighbor: output: 11309 nodes
##############################################
'''
def preprocess_text(text: str) -> str:
    """
    Preprocess text by:
    1. Replacing mentions, hashtags, links.
    2. Removing non-alphabetic characters.
    3. Converting to lowercase.
    4. Removing extra spaces.
    """
    try:
        text = re.sub(r"@\w+", "mention", text)
        text = re.sub(r"#\w+", "hashtags", text)
        text = re.sub(r"https?://\S+|www\.\S+", "link", text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
    
        return text
    except:
        return ""

nodeid = []
nodeid_tweets = {}
edges_nodeid_nodeid = {}
nodeid_label = {}
nodeid_class = {} # dev,test,train,support
nodeid_description = {}

# Process and translate tweets, saving results to the output file
for node_class,file_name in zip(Node_class,File_Names):  # Process only the first two files as per the original code 
    no_neighbor_counter = 0
    no_tweet_counter = 0
    no_description_counter = 0
    with open(file_name, "r", encoding="utf-8") as file:
        data = json.load(file)

        for user in tqdm(data):
            user_id = user.get("ID", -1)  # Get user ID
            label = user.get('label',-1)
            
            description = user.get("profile", '')  
            try:   
                description = description['description']
                description = preprocess_text(description)
                if len(description)<5:
                    no_description_counter += 1
                    continue
            except:
                no_description_counter += 1
                continue
            tweets = user.get("tweet", [])  # Get user's tweets

            # Skip users with no tweets
            if not tweets or tweets is None:
                no_tweet_counter+=1
                continue
            
            edges = user.get("neighbor",None)
            if edges==None:
                no_neighbor_counter += 1
                continue
            else:
                follower = edges['follower']
                following = edges['following']
            
            if len(follower)+len(following) == 0 :
                no_neighbor_counter += 1
                continue
                
            else:
                cleaned_tweets = []
                
                for tweet in tweets:
                    temp = preprocess_text(tweet)
                    if len(temp)>5:
                        cleaned_tweets.append(temp[:])
                if len(cleaned_tweets)>0:    
                       
                    nodeid.append(user_id)
                    nodeid_description[user_id] = description[:]
                    nodeid_tweets[user_id] =  cleaned_tweets[:]
                    edges_nodeid_nodeid[user_id] = {'follower':follower[:],'following':following[:]}
                    nodeid_label[user_id] = label
                    nodeid_class[user_id] = node_class
                    
    print(len(nodeid),no_neighbor_counter,no_tweet_counter,no_description_counter) 
    




with open("nodeids.pkl", 'wb') as f:
    pickle.dump(nodeid, f)

with open("nodeid_tweets.pkl", 'wb') as f:
    pickle.dump(nodeid_tweets, f)

with open("edges_nodeid_nodeid.pkl", 'wb') as f:
    pickle.dump(edges_nodeid_nodeid, f)

with open("nodeid_label.pkl", 'wb') as f:
    pickle.dump(nodeid_label, f)

with open("nodeid_class.pkl", 'wb') as f:
    pickle.dump(nodeid_class, f)

with open("nodeid_description.pkl", 'wb') as f:
    pickle.dump(nodeid_description, f)


##############################################
## Part 2: Edge to Embedding and assigning new node name
##############################################

with open("nodeids.pkl", 'rb') as f:
    nodeids = pickle.load(f)

with open("nodeid_tweets.pkl", 'rb') as f:
    nodeid_tweets = pickle.load(f)

with open("nodeid_description.pkl", 'rb') as f:
    nodeid_description = pickle.load(f)


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

def extract_keywords(text,NUMBER_OF_KEYWORDS) -> list:
    keywords = [kw[0] for kw in yake_extractor.extract_keywords(text)[:NUMBER_OF_KEYWORDS]]
    return keywords

random_samples = []

for node in tqdm(nodeids):
    x = random.choice([1,1,1,1,2])
    if x == 2:
        desc = nodeid_description[node]
        random_samples.append(desc[:])
    else:
        tweet = random.choice(nodeid_tweets[node])
        random_samples.append(tweet[:])


fasttext_model = load_facebook_vectors(FASTTEXT_BIN_PATH)
embedding_model = FastTextEmbeddingModel(fasttext_model, target_size=32)

try:
    embedding_model.load_pca()
    print("PCA loaded")
except:
    sampled_tweets = random_samples
    embedding_model.fit_and_save(sampled_tweets, pca_save_path=PCA_MODEL_PATH)
    print("PCA learning process finished.")


nodeid_embedding = {}
tweetid_embedding = {}
nodeid_tweetid = {}

all_keywords = set()
tweetid_keywords = {}

tweet_counter_for_id = 0

for node in tqdm(nodeids):
    description = nodeid_description[node]
    description_embedding = embedding_model.transform([description])[0]
    nodeid_embedding[node] = description_embedding[:]
    
    tweets = nodeid_tweets[node]
    tweet_embeddings = embedding_model.transform(tweets)
    nodeid_tweetid[node] = []
    for emb,tweet in zip(tweet_embeddings,tweets):
        tweetid = 't' + str(tweet_counter_for_id)
        tweet_counter_for_id += 1
        
        nodeid_tweetid[node].append(tweetid)
        tweetid_embedding[tweetid] = emb[:]
        
        # Keywords
        keywords = extract_keywords(text=tweet,NUMBER_OF_KEYWORDS=NUMBER_OF_KEYWORDS)
        all_keywords.update(keywords[:])
        tweetid_keywords[tweetid] = keywords[:]

print(f"tweet_counter_for_id{tweet_counter_for_id}")

with open("tweetid_embedding.pkl", 'wb') as f:
    pickle.dump(tweetid_embedding, f)
tweetid_embedding = ''
print(f"   ----- 1 -----")  
with open("nodeid_embedding.pkl", 'wb') as f:
    pickle.dump(nodeid_embedding, f)
nodeid_embedding = ''
print(f"   ----- 2 -----")  

with open("nodeid_tweetid.pkl", 'wb') as f:
    pickle.dump(nodeid_tweetid, f)
print(f"   ----- 3 -----")  
    
with open("all_keywords.pkl", 'wb') as f:
    pickle.dump(all_keywords, f)
print(f"   ----- 4 ----- number of keywords:{len(all_keywords)}")  

with open("tweetid_keywords.pkl", 'wb') as f:
    pickle.dump(tweetid_keywords, f)
print(f"   ----- 5 -----")  

##############################################
## Part 3: Keywords Embedding
##############################################

nodes = read_pickle('nodeids.pkl')
all_keywords = read_pickle('all_keywords.pkl')

set_of_nodes = set(nodes)
keywords_embedding = {}


fasttext_model = load_facebook_vectors(FASTTEXT_BIN_PATH)
embedding_model = FastTextEmbeddingModel(fasttext_model, target_size=32)
embedding_model.load_pca()

keywords = list(all_keywords)
emb_keywords = embedding_model.transform(keywords)

for keyword,emb in tqdm(zip(keywords,emb_keywords)):
    keywords_embedding[keyword] = emb[:]

write_pickle(variable=keywords_embedding,name='keywords_embedding.pkl')


##############################################
## Part 4: creat H-Graph
##############################################

nodes = read_pickle('nodeids.pkl')
set_of_nodes = set(nodes)
all_keywords = read_pickle('all_keywords.pkl')

edges_nodeid_nodeid = read_pickle('edges_nodeid_nodeid.pkl')

nodeid_class = read_pickle('nodeid_class.pkl')
nodeid_labels = read_pickle('nodeid_label.pkl')
nodeid_tweetid = read_pickle('nodeid_tweetid.pkl')
tweetid_keywords = read_pickle('tweetid_keywords.pkl')

keywords_embedding = read_pickle('keywords_embedding.pkl')
tweetid_embedding = read_pickle('tweetid_embedding.pkl')
nodeid_embedding = read_pickle('nodeid_embedding.pkl')




# node_types : 'u','t','k'
# edge_types: 'fr','fg','ut','tu','tk','kt'
# node_classes: 'test','dev','train',"support"
#check follower and following and clearing useless edges
G = nx.DiGraph()  
for node in tqdm(nodes):
    G.add_node(node, label=nodeid_labels[node], type="u", node_class=nodeid_class[node], embedding=nodeid_embedding[node][:])

print('user nodes added')

for node in tqdm(all_keywords):
    G.add_node(node, label=-1, type="k", node_class="support", embedding=keywords_embedding[node][:])

print('keyword nodes added')

all_tweets = list(tweetid_embedding.keys())
for node in tqdm(all_tweets):
    G.add_node(node, label=-1, type="t", node_class="support", embedding=tweetid_embedding[node][:])

print('tweet nodes added')

print('adding edges started')

## we had: edges_nodeid_nodeid[user_id] = {'follower':follower[:],'following':following[:]}
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")


for node in tqdm(nodes):
    followeredges =  edges_nodeid_nodeid[node]['follower']
    followingedges =  edges_nodeid_nodeid[node]['following']
    
    for node2 in followeredges:
        if node2 in set_of_nodes:
            G.add_edge(node, node2, type="fr")
    
    for node2 in followingedges:
        if node2 in set_of_nodes:
            G.add_edge(node, node2, type="fg")


    tweets = nodeid_tweetid[node]
    for tweet in tweets:
        G.add_edge(node, tweet, type="ut")
        G.add_edge(tweet, node, type="tu")
        
        
        keywords = tweetid_keywords[tweet]
        for keyword in keywords:
            G.add_edge(tweet, keyword, type="tk")
            G.add_edge(keyword, tweet, type="kt")

print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")


write_pickle(variable=G,name='G1.pkl')
print('initial graph created')

'''

G = read_pickle(name='G1.pkl')
print("read")
mapping = {old_name: new_name for new_name, old_name in tqdm(enumerate(G.nodes()))}
G = nx.relabel_nodes(G, mapping)
print("relabelled")
write_pickle(variable=G,name='G2.pkl')

###################################
####### just some statistics#######
###################################

G = read_pickle(name='G2.pkl')

# Some Statistics
print("computing statistics")
node_stats = {}
for node, data in G.nodes(data=True):
    node_type = data.get('type', 'Unknown')
    if node_type not in node_stats:
        node_stats[node_type] = {'count': 0, 'classes': {}, 'labels': set()}
    
    node_stats[node_type]['count'] += 1
    node_class = data.get('node_class', 'Unknown')
    node_stats[node_type]['classes'][node_class] = node_stats[node_type]['classes'].get(node_class, 0) + 1
    node_stats[node_type]['labels'].add(data.get('label', 'Unknown'))  # Use a set for unique labels

# Compute average degree for different types of nodes
avg_degrees = {}
for node_type in node_stats:
    nodes_of_type = [n for n, d in G.nodes(data=True) if d.get('type') == node_type]
    total_degree = sum(dict(G.degree(nodes_of_type)).values())
    avg_degrees[node_type] = total_degree / len(nodes_of_type)

# Collect edge statistics
edge_stats = {}
for _, _, data in G.edges(data=True):
    edge_type = data.get('type', 'Unknown')
    edge_stats[edge_type] = edge_stats.get(edge_type, 0) + 1

# Step 3: Print graph statistics
print("Graph Statistics:")
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}\n")

print("Node Statistics:")
for node_type, stats in node_stats.items():
    print(f"Type: {node_type}")
    print(f"  Count: {stats['count']}")
    print(f"  Classes: {stats['classes']}")
    print(f"  Labels: {stats['labels']}")  # Convert set to sorted list for display
    print(f"  Avg Degree: {avg_degrees[node_type]:.2f}")

print("\nEdge Statistics:")
for edge_type, count in edge_stats.items():
    print(f"Type: {edge_type}, Count: {count}")



'''
Input: G2.pkl
outputs:
### for Homogenous G
"user_labels.csv"
"user_embeddings.csv"   
"user_edges.csv"

### for heterogenous G
"node_information.csv"
"node_labels.csv"
"node_embeddings.csv"
"edges.csv"

train_set.txt
test_set.txt

'''

import csv
import numpy as np
from tqdm import tqdm
import pickle
import random
SEED = 1
random.seed(SEED)
import networkx as nx

def read_pickle(name):
    with open(name,'rb') as f:
        temp = pickle.load(f)
    return temp

def write_pickle(variable,name):
    with open(name , 'wb') as f:
        pickle.dump(variable, f)

# node_types : 'u','t','k'
# edge_types: 'fr','fg','ut','tu','tk','kt'
# node_classes: 'test','dev','train',"support"

# node_id,label
# node_id,old_id,attributes
# 0,fcolomboil,{'node_type': 'keyword'}


# source,target,attributes
# 3624719,3728235,"{""edge_type"": ""user_to_user""}"


###################################################################################
###############change files for input of learning codes############################
###################################################################################


G = read_pickle(name='G2.pkl')

usernodes = []
allnodes = []
nodeinformation = {}
nodelabel = {}
trainset = []
testset = []
nodeembedding = {}
userembedding = {}

alledges = []
useredges = []




for node, data in tqdm(G.nodes(data=True)):
    node_type = data.get('type', 'Unknown')
    node_embedding = data.get('embedding', '')
    if len(node_embedding) == 0:
        raise ValueError("An error for no embedding") 

    if node_type == 'u':
        usernodes.append(node)
        
        node_type = data.get('type', -1)
        if node_type == '1':
            nodelabel[node] = "sybil"
        elif node_type == '0':
            nodelabel[node] = "benign"
        else:
            nodelabel[node] = "unknown"
        
        node_class = data.get('node_class', 'Unknown')
        if node_class == 'train':
            trainset.append(node)
        elif node_class == 'test':
            testset.append(node)    
        userembedding[node] = node_embedding
           
    allnodes.append(node)
    nodeinformation[node] = node_type
    nodeembedding[node] = node_embedding

userset = set(usernodes)



for x, y, data in tqdm(G.edges(data=True)): 
    edge_type = data.get('type', 'Unknown') #'fr','fg','ut','tu','tk','kt'
    if edge_type == 'Unknown':
        raise ValueError("An error: no label for edge") 
    
    if x in userset and y in userset:
        useredges.append([x,y,data])

    alledges.append([x,y,data])
    

        
        
    
with open("user_labels.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["node_id", "label"])
    for node, label in tqdm(nodelabel.items()):
        writer.writerow([node, label])

with open("user_embeddings.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["node_id", "embedding"])
    for node, embedding in tqdm(userembedding.items()):
        writer.writerow([node, embedding])

with open("user_edges.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["source", "target", "attributes"])
    for edge in tqdm(useredges):
        writer.writerow(edge)

with open("node_information.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["node_id", "attributes"])
    for node, attributes in tqdm(nodeinformation.items()):
        writer.writerow([node, attributes])

with open("node_labels.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["node_id", "label"])
    for node, label in tqdm(nodelabel.items()):
        writer.writerow([node, label])

with open("node_embeddings.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["node_id", "embedding"])
    for node, embedding in tqdm(nodeembedding.items()):
        writer.writerow([node, embedding])

with open("edges.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["source", "target", "attributes"])
    for edge in tqdm(alledges):
        writer.writerow(edge)

with open("train_set.txt", "w") as f:
    for node in tqdm(trainset):
        f.write(f"{node}\n")

with open("test_set.txt", "w") as f:
    for node in tqdm(testset):
        f.write(f"{node}\n")

print("Generated files:")
print("- user_labels.csv")
print("- user_embeddings.csv")
print("- user_edges.csv")
print("- node_information.csv")
print("- node_labels.csv")
print("- node_embeddings.csv")
print("- edges.csv")
print("- train_set.txt")
print("- test_set.txt")

print("\nStatistics:")
print(f"Number of user nodes: {len(usernodes)}")
print(f"Number of all nodes: {len(allnodes)}")
print(f"Number of user edges: {len(useredges)}")
print(f"Number of all edges: {len(alledges)}")
print(f"Train set size: {len(trainset)}")
print(f"Test set size: {len(testset)}")
