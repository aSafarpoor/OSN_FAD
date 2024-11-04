from sklearn import metrics

from social_networks import *
from algorithms import *
from gnn import *

import utils
import random

import seaborn as sns
import matplotlib.pyplot as plt

import pickle



class Evaluator:
    def __init__(self, social_network: SocialNetwork,
                 train_honest_nodes: [int] = None,
                 train_sybil_nodes: [int] = None,
                 test_honest_nodes: [int] = None,
                 test_sybil_nodes: [int] = None,
                 evaluate_on_full_data: bool = False,
                 label_noise_fraction: float = 0.0,
                 verbose: bool = True) -> None:
        self.social_network = social_network
        if self.social_network.network is None:
            raise Exception("The social network has no network graph")

        
        if train_honest_nodes is None or train_sybil_nodes is None or test_honest_nodes is None or test_sybil_nodes is None:
            train_honest_nodes, train_sybil_nodes, test_honest_nodes, test_sybil_nodes = social_network.get_train_test_split()
        
        '''
        with open('train_honest_nodes.pickle', 'wb') as handle:
            pickle.dump(train_honest_nodes, handle)
        with open('train_sybil_nodes.pickle', 'wb') as handle:
            pickle.dump(train_sybil_nodes, handle)
        with open('test_honest_nodes.pickle', 'wb') as handle:
            pickle.dump(test_honest_nodes, handle)
        with open('test_sybil_nodes.pickle', 'wb') as handle:
            pickle.dump(test_sybil_nodes, handle)
        '''

        if train_honest_nodes is None or train_sybil_nodes is None or test_honest_nodes is None or test_sybil_nodes is None:
            with open('train_honest_nodes.pickle', 'rb') as handle:
                train_honest_nodes = pickle.load(handle)
            with open('train_sybil_nodes.pickle', 'rb') as handle:
                train_sybil_nodes = pickle.load(handle)
            with open('test_honest_nodes.pickle', 'rb') as handle:
                test_honest_nodes = pickle.load(handle)
            with open('test_sybil_nodes.pickle', 'rb') as handle:
                test_sybil_nodes = pickle.load(handle)
        
        
        self.train_honest_nodes = train_honest_nodes
        self.train_sybil_nodes = train_sybil_nodes
        self.test_honest_nodes = test_honest_nodes
        self.test_sybil_nodes = test_sybil_nodes
        self.evaluate_on_full_data = evaluate_on_full_data
        self.label_noise_fraction = label_noise_fraction
        self.verbose = verbose

        self.algorithms = None

    def evaluate(self, algo_name, algorithm: SybilFinder, reinitialize_GNNs: bool = True,
                 override_train_honest_nodes: [int] = None, override_train_sybil_nodes: [int] = None):
        
        
    
        algorithm.verbose = self.verbose
        
        if algorithm.uses_directed_graph:
            algorithm.set_graph(self.social_network.directed_network)
        else:
            algorithm.set_graph(self.social_network.undirected_network)
        # algorithm.set_graph(self.social_network.network)

        if override_train_honest_nodes is not None and override_train_honest_nodes is not None:
            train_honest_nodes = override_train_honest_nodes
            train_sybil_nodes = override_train_sybil_nodes
            
        elif not np.isclose(self.label_noise_fraction, 0.0):
            # Apply label noise to training sets
            num_honest_victims = round(self.label_noise_fraction * len(self.train_honest_nodes))
            honest_victims = random.sample(self.train_honest_nodes, k=num_honest_victims)
            honest_remaining = list(set(self.train_honest_nodes) - set(honest_victims))

            num_sybil_victims = round(self.label_noise_fraction * len(self.train_sybil_nodes))
            sybil_victims = random.sample(self.train_sybil_nodes, k=num_sybil_victims)
            sybil_remaining = list(set(self.train_sybil_nodes) - set(sybil_victims))

            train_honest_nodes = honest_remaining + sybil_victims
            train_sybil_nodes = sybil_remaining + honest_victims
        else:
            train_honest_nodes = self.train_honest_nodes
            train_sybil_nodes = self.train_sybil_nodes

        test_honest_nodes = self.test_honest_nodes
        test_sybil_nodes = self.test_sybil_nodes

        if isinstance(algorithm, SybilGNN):
            print("Yes is instance of SybilGNN")
            
            if reinitialize_GNNs:
                algorithm.reinitialize()
            if algorithm.train_model:
                if algorithm.fine_tune:
                    algorithm.set_train_nodes(honest_nodes=train_honest_nodes,
                                              sybil_nodes=train_sybil_nodes)
                else:
                    algorithm.set_train_nodes(honest_nodes=self.social_network.honest_nodes,
                                              sybil_nodes=self.social_network.sybil_nodes)
            else:
                algorithm.set_train_nodes(honest_nodes=train_honest_nodes,
                                          sybil_nodes=train_sybil_nodes)
        else:
            algorithm.set_train_nodes(honest_nodes=train_honest_nodes,
                                      sybil_nodes=train_sybil_nodes)
        if isinstance(algorithm, LegacyAlgorithm):
            print("Yes is instance of LegacyAlgorithm")
            
            self.social_network.write_graph_train_test_split(directory="data/legacy_temp",
                                                             both_edges_for_undirected=True,
                                                             train_honest_nodes=train_honest_nodes,
                                                             train_sybil_nodes=train_sybil_nodes,
                                                             test_honest_nodes=train_honest_nodes,
                                                             test_sybil_nodes=train_sybil_nodes)


        algorithm.predicted_sybils = algorithm.find_sybils(algo_name)

    def evaluate_all(self, algorithms: [SybilFinder], reinitialize_GNNs: bool = True):
        
        
        self.algorithms = algorithms
        
        '''
        if self.verbose:
            print(f"Evaluating {len(algorithms)} algorithms:")

        if not np.isclose(self.label_noise_fraction, 0.0):
            # Apply label noise to training sets

            num_honest_victims = round(self.label_noise_fraction * len(self.train_honest_nodes))
            honest_victims = random.sample(self.train_honest_nodes, k=num_honest_victims)
            honest_remaining = list(set(self.train_honest_nodes) - set(honest_victims))

            num_sybil_victims = round(self.label_noise_fraction * len(self.train_sybil_nodes))
            sybil_victims = random.sample(self.train_sybil_nodes, k=num_sybil_victims)
            sybil_remaining = list(set(self.train_sybil_nodes) - set(sybil_victims))

            train_honest_nodes = honest_remaining + sybil_victims
            train_sybil_nodes = sybil_remaining + honest_victims
        else:
        '''
        train_honest_nodes = self.train_honest_nodes
        train_sybil_nodes = self.train_sybil_nodes

        for algo in algorithms:
            print("Evaluate all --> algorithm is:",algo)
            self.evaluate(algo_name = str(algo), algorithm=algo, reinitialize_GNNs=reinitialize_GNNs,
                          override_train_honest_nodes=train_honest_nodes, override_train_sybil_nodes=train_sybil_nodes)

    def get_stats(self, algorithm: SybilFinder, compute_AUC: bool = True, evaluate_on_full_data: bool = None,
                  stop_after_auc: bool = False):
        if algorithm.predicted_sybils is None:
            raise Exception("Algorithm has not been evaluated yet")
        stats = {}
        nodes = self.social_network.network.nodes_list()
        num_nodes = self.social_network.network.num_nodes()

        labels = utils.mask_from_lists([self.social_network.honest_nodes, self.social_network.sybil_nodes],
                                       labels=[-1, 1], no_label=0) # old version
       
        
        
        if algorithm.has_trust_values and compute_AUC:

            y_true = labels
            trust_values = -algorithm.trust_values

            FPR, TPR, thresholds = metrics.roc_curve(y_true, trust_values)
            optimal_threshold = thresholds[np.argmax(TPR - FPR)]
            
            if self.verbose:
                print(f"test optimal threshold: {optimal_threshold}")
            
            AUC = metrics.auc(FPR, TPR)
            stats["AUC"] = AUC
            if stop_after_auc:
                return stats

        # predicted_sybils = algorithm.predicted_sybils
        # confusion_matrix = self.get_confusion_matrix(predicted_sybils=predicted_sybils,
        #                                              labels=labels,
        #                                              evaluate_on_full_data=evaluate_on_full_data)

        # stats["confusion_matrix"] = confusion_matrix
        # TP = confusion_matrix["TP"]
        # FP = confusion_matrix["FP"]
        # FN = confusion_matrix["FN"]
        # TN = confusion_matrix["TN"]

        # F1 = 2 * TP / (2 * TP + FP + FN)
        # stats["F1"] = F1

        # N = TN + FP
        # P = FN + TP
        # N_hat = TN + FN
        # P_hat = FP + TP

        # accuracy = (TP + TN) / (P + N)
        # stats["accuracy"] = accuracy

        # if P_hat != 0:
        #     precision = TP / P_hat
        # else:
        #     precision = np.inf
        # stats["precision"] = precision

        # TPR = TP / P  # = recall / sensitivity
        # stats["recall"] = TPR

        # FPR = FP / N
        # stats["FPR"] = FPR

        # TNR = TN / N  # = specificity
        # stats["TNR"] = TNR

        # stats["pretrain_runtime"] = algorithm.pretrain_runtime
        # stats["runtime"] = algorithm.runtime

        # stats['TP'] = TP
        # stats['TN'] = TN
        # stats['FP'] = FP
        # stats['FN'] = FN
        
        return stats

    def get_stats_for_test_targets(self, algorithm: SybilFinder, compute_AUC: bool = True, evaluate_on_full_data: bool = None,
                  stop_after_auc: bool = False):
        if algorithm.predicted_sybils is None:
            raise Exception("Algorithm has not been evaluated yet")
        stats = {}
        nodes = self.social_network.network.nodes_list()
        num_nodes = self.social_network.network.num_nodes()

        labels = utils.mask_from_lists([self.social_network.honest_nodes, self.social_network.sybil_nodes],
                                       labels=[-1, 1], no_label=0) # old version
        

        if algorithm.has_trust_values and compute_AUC:

            y_true = labels[:]
            trust_values = -algorithm.trust_values
            
            
            
            target_set = set(self.social_network.targets)
            # Combine extraction in a single loop
            y_true, trust_values = zip(
                *[
                    (label, trust)
                    for node, label, trust in zip(self.social_network.honest_nodes + self.social_network.sybil_nodes, y_true, trust_values)
                    if node in target_set
                ]
            )

            # Convert to lists if needed
            y_true = list(y_true)
            trust_values = list(trust_values)
            
        
            # creat label and value for target
            FPR, TPR, thresholds = metrics.roc_curve(y_true, trust_values)
            
            optimal_threshold = thresholds[np.argmax(TPR - FPR)]
            
            if self.verbose:
                print(f"test optimal threshold: {optimal_threshold}")
            
            AUC = metrics.auc(FPR, TPR)
            stats["AUC"] = AUC
            if stop_after_auc:
                return stats

    
     
        
        
        return stats 
    
    def get_confusion_matrix(self, predicted_sybils: [int], labels: np.ndarray, evaluate_on_full_data: bool = None):
        if evaluate_on_full_data is None:
            evaluate_on_full_data = self.evaluate_on_full_data

        if evaluate_on_full_data:
            nodes = self.social_network.network.nodes_list()
        else:
            nodes = self.test_honest_nodes + self.test_sybil_nodes
        num_nodes = len(nodes)

        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for i in nodes:

            if i in predicted_sybils:
                # Predicted as sybil (POSITIVE)
                if labels[i] == 1:  # actual sybil
                    TP += 1
                elif labels[i] == -1:  # actual honest
                    FP += 1
                else:
                    raise Exception(f"Node is neither in honest nor sybil nodes, the label is {labels[i]}")
            else:
                # Not predicted as sybil, assumed honest (NEGATIVE)
                if labels[i] == -1:  # actual honest
                    TN += 1
                elif labels[i] == 1:  # actual sybil
                    FN += 1
                else:
                    raise Exception("Node is neither in honest nor sybil nodes")

        return {
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN
        }

    def get_all_stats(self, compute_AUC: bool = True, evaluate_on_full_data: bool = None, stop_after_auc: bool = False):
        if self.algorithms is None:
            raise Exception("Algorithms have not been evaluated yet")
        if self.verbose:
            print(f"\nGetting stats of {len(self.algorithms)} algorithms:")
        all_stats = []
        c = 0
        for algo in self.algorithms:
            c+=1
            print(c ," azz ", len(self.algorithms))
            stats = self.get_stats(algorithm=algo,
                                   compute_AUC=compute_AUC,
                                   evaluate_on_full_data=evaluate_on_full_data,
                                   stop_after_auc=stop_after_auc)
            all_stats.append(stats)
            if self.verbose:
                print(f"Stats of {algo} ...")
                print(stats)
                if "confusion_matrix" in stats.keys():
                    Evaluator.print_confusion_matrix(stats["confusion_matrix"])

            try:
                Evaluator.print_confusion_matrix(stats["confusion_matrix"])
            except:
                print("noconfusion matrix")
        return all_stats

    def get_all_stats_for_targets(self, compute_AUC: bool = True, evaluate_on_full_data: bool = None, stop_after_auc: bool = False):
        if self.algorithms is None:
            raise Exception("Algorithms have not been evaluated yet")
        if self.verbose:
            print(f"\nGetting stats of {len(self.algorithms)} algorithms:")
        all_stats = []
        c = 0
        for algo in self.algorithms:
            c+=1
            print(c ," azz ", len(self.algorithms))
            stats = self.get_stats_for_test_targets(algorithm=algo,
                                   compute_AUC=compute_AUC,
                                   evaluate_on_full_data=evaluate_on_full_data,
                                   stop_after_auc=stop_after_auc)
            all_stats.append(stats)
            if self.verbose:
                print(f"Stats of {algo} ...")
                print(stats)
                if "confusion_matrix" in stats.keys():
                    Evaluator.print_confusion_matrix(stats["confusion_matrix"])

            try:
                Evaluator.print_confusion_matrix(stats["confusion_matrix"])
            except:
                print(
                    "no confusion matrix"
                )
            
        return all_stats




    @staticmethod
    def print_confusion_matrix(confusion_matrix):
        print("Confusion Matrix:")
        print(f"TP: {confusion_matrix['TP']} \t| FP: {confusion_matrix['FP']}")
        print(f"FN: {confusion_matrix['FN']} \t| TN: {confusion_matrix['TN']}")
