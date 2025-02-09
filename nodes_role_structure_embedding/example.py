import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from karateclub import Role2Vec

class HybridGraphEmbedding:
    def __init__(self, G, role2vec_dimensions=64, wl_iterations=2, pca_components=16, random_state=42):
        """
        Initialize the hybrid embedding model.
        
        Parameters:
            G (networkx.Graph): The input graph.
            role2vec_dimensions (int): Dimension for Role2Vec embeddings.
            wl_iterations (int): Number of iterations for computing WL labels.
            pca_components (int): Number of PCA components for reducing WL one-hot features.
            random_state (int): Random state for reproducibility.
        """
        self.G = G
        self.role2vec_dimensions = role2vec_dimensions
        self.wl_iterations = wl_iterations
        self.pca_components = pca_components
        self.random_state = random_state
        
        # Placeholders for embeddings and mappings.
        self.role_embeddings = None
        self.wl_features_reduced = None
        self.hybrid_embeddings = None
        self.emb_dict = {}
        self.nodes_sorted = None

    def _compute_wl_labels(self):
        """
        Compute Weisfeiler-Lehman (WL) labels for each node.
        Initialization is done with the node degree (as a string) and then updated iteratively.
        """
        # Initialize labels using node degree.
        labels = {node: str(self.G.degree[node]) for node in self.G.nodes()}
        for _ in range(self.wl_iterations):
            new_labels = {}
            for node in self.G.nodes():
                neighbor_labels = sorted([labels[neigh] for neigh in self.G.neighbors(node)])
                new_labels[node] = labels[node] + "_" + "_".join(neighbor_labels)
            labels = new_labels
        return labels

    def _one_hot_encode_labels(self, labels):
        """
        One-hot encode the WL labels.
        Returns:
            one_hot: numpy.ndarray of shape (num_nodes, num_unique_labels)
            nodes_sorted: list of nodes sorted in ascending order.
            label_to_index: mapping from label to column index.
        """
        nodes_sorted = sorted(self.G.nodes())
        unique_labels = sorted(set(labels.values()))
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        num_features = len(unique_labels)
        one_hot = np.zeros((len(nodes_sorted), num_features))
        for i, node in enumerate(nodes_sorted):
            idx = label_to_index[labels[node]]
            one_hot[i, idx] = 1
        return one_hot, nodes_sorted, label_to_index

    def fit(self):
        """
        Fit the hybrid model to the graph. This method:
            1. Computes Role2Vec embeddings.
            2. Computes WL labels and one-hot encodes them.
            3. Reduces the one-hot WL features with PCA.
            4. Concatenates Role2Vec and WL features to form the hybrid embedding.
        """
        # 1. Role2Vec Embeddings
        role_model = Role2Vec(dimensions=self.role2vec_dimensions)
        role_model.fit(self.G)
        self.role_embeddings = role_model.get_embedding()  # shape: (num_nodes, role2vec_dimensions)
        
        # 2. WL Labels and One-hot Encoding
        wl_labels = self._compute_wl_labels()
        one_hot, nodes_sorted, _ = self._one_hot_encode_labels(wl_labels)
        self.nodes_sorted = nodes_sorted
        
        # 3. PCA on WL Features
        pca = PCA(n_components=self.pca_components, random_state=self.random_state)
        self.wl_features_reduced = pca.fit_transform(one_hot)  # shape: (num_nodes, pca_components)
        
        # 4. Concatenate the features
        self.hybrid_embeddings = np.concatenate([self.role_embeddings, self.wl_features_reduced], axis=1)
        
        # Create a mapping from node to its hybrid embedding.
        self.emb_dict = {node: self.hybrid_embeddings[i] for i, node in enumerate(self.nodes_sorted)}

    def get_embedding(self):
        """
        Returns the final hybrid embedding as a numpy array of shape 
        (num_nodes, role2vec_dimensions + pca_components).
        """
        if self.hybrid_embeddings is None:
            raise ValueError("Model not fitted yet. Call fit() before getting embeddings.")
        return self.hybrid_embeddings

    def cosine_similarity(self, node1, node2):
        """
        Compute the cosine similarity between the embeddings of two nodes.
        
        Parameters:
            node1, node2: Node identifiers present in the graph.
        
        Returns:
            A float value representing the cosine similarity.
        """
        if node1 not in self.emb_dict or node2 not in self.emb_dict:
            raise ValueError("One or both nodes are not in the graph.")
        vec1 = torch.tensor(self.emb_dict[node1], dtype=torch.float32).unsqueeze(0)
        vec2 = torch.tensor(self.emb_dict[node2], dtype=torch.float32).unsqueeze(0)
        return F.cosine_similarity(vec1, vec2).item()

# Example usage:
if __name__ == "__main__":
    # Build your graph (the same as in your earlier example)
    G = nx.Graph()
    
    # Add triangle motifs
    triangle_nodes = []
    for i in range(50):
        n1, n2, n3 = len(G), len(G)+1, len(G)+2
        G.add_edges_from([(n1, n2), (n2, n3), (n3, n1)])
        triangle_nodes.extend([n1, n2, n3])
    
    # Add star motifs
    star_nodes = []
    for i in range(20):
        center = len(G)
        leaf_nodes = [len(G)+j for j in range(1, 6)]
        G.add_edges_from([(center, leaf) for leaf in leaf_nodes])
        star_nodes.append(center)
    
    # Add chain structure
    chain_nodes = []
    for i in range(100):
        G.add_edge(len(G), len(G)+1)
        chain_nodes.append(len(G))
    
    # Add background random edges
    num_random_nodes = 500
    random_G = nx.erdos_renyi_graph(num_random_nodes, p=0.02, seed=42)
    offset = len(G)
    for u, v in random_G.edges():
        G.add_edge(u + offset, v + offset)
    
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Create the hybrid embedding model with the graph as input.
    model = HybridGraphEmbedding(G)
    model.fit()
    
    # Get hybrid embeddings and test cosine similarity
    emb = model.get_embedding()
    print(f"Example hybrid embedding for node 0: {model.emb_dict[0]}")
    
    # Compute some cosine similarities
    sim_tri = model.cosine_similarity(triangle_nodes[0], triangle_nodes[10])
    sim_star = model.cosine_similarity(star_nodes[0], star_nodes[5])
    sim_chain = model.cosine_similarity(chain_nodes[0], chain_nodes[10])
    sim_mixed = model.cosine_similarity(triangle_nodes[0], star_nodes[0])
    
    print(f"Similarity between two triangle nodes: {sim_tri:.4f}")
    print(f"Similarity between two star nodes: {sim_star:.4f}")
    print(f"Similarity between two chain nodes: {sim_chain:.4f}")
    print(f"Similarity between triangle and star node: {sim_mixed:.4f}")
