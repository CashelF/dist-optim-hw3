import numpy as np
import matplotlib.pyplot as plt
from hw3_functions import generate_W_matrix, synchronous_consensus, randomized_gossip

def calculate_dissagreement(consensus_matrix):
    return np.sum((consensus_matrix-np.mean(consensus_matrix[0,])) ** 2, axis=1)

def plot_disagreement(disagreement_synchronous, disagreement_gossip):
    plt.plot(range(disagreement_synchronous.shape[0]), disagreement_synchronous, label="synchronous")
    plt.plot(range(disagreement_gossip.shape[0]), disagreement_gossip, label="random gossip")
    plt.xlabel("Steps")
    plt.ylabel("Disagreement")
    plt.title("Disagreement Comparison")
    plt.legend()
    plt.savefig("disagreement_comparison.jpg")
    plt.clf()

def plot_trajectories(synchronous_consensus_matrix, gossip_consensus_matrix, nodes):
    plt.plot(range(synchronous_consensus_matrix.shape[0]), synchronous_consensus_matrix[:, nodes[0]], label=f"Node {nodes[0]} Synchronous")
    plt.plot(range(synchronous_consensus_matrix.shape[0]), synchronous_consensus_matrix[:, nodes[1]], label=f"Node {nodes[1]} Synchronous")

    plt.plot(range(gossip_consensus_matrix.shape[0]), gossip_consensus_matrix[:, nodes[0]], label=f"Node {nodes[0]} Random Gossip")
    plt.plot(range(gossip_consensus_matrix.shape[0]), gossip_consensus_matrix[:, nodes[1]], label=f"Node {nodes[1]} Random Gossip")

    plt.xlabel("Steps")
    plt.ylabel("Node Value")
    plt.title("Node Trajectories")
    plt.legend(loc='upper left')
    plt.savefig("node_trajectories")
    plt.clf()


seed = 42
np.random.seed(seed)
iid_data = np.random.random(20)
steps = 20

ring_graph_W = generate_W_matrix(iid_data.shape[0], 1/2)
synchronous_consensus_matrix = synchronous_consensus(iid_data, steps, ring_graph_W)
disagreement_synchronous = calculate_dissagreement(synchronous_consensus_matrix)

gossip_consensus_matrix = randomized_gossip(iid_data, steps, seed)
disagreement_gossip = calculate_dissagreement(gossip_consensus_matrix)

plot_disagreement(disagreement_synchronous, disagreement_gossip)
plot_trajectories(synchronous_consensus_matrix, gossip_consensus_matrix, (5, 15))
