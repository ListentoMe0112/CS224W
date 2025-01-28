import torch
import networkx as nx
import matplotlib.pyplot as plt
import math
from sklearn.decomposition._pca import PCA 
G = nx.karate_club_graph()
nx.draw(G, with_labels = True)

def custom_round(number):
    fractional_part = number - math.floor(number)
    if fractional_part >= 0.5:
        return math.ceil(number)
    else:
        return math.floor(number)

def average_degree(num_edges, num_nodes):
  # TODO: Implement this function that takes number of edges
  # and number of nodes, and returns the average node degree of
  # the graph. Round the result to nearest integer (for example
  # 3.3 will be rounded to 3 and 3.7 will be rounded to 4)

  avg_degree = custom_round(num_edges / num_nodes)

  ############# Your code here ############

  #########################################

  return avg_degree

num_edges = G.number_of_edges()
num_nodes = G.number_of_nodes()
avg_degree = average_degree(num_edges, num_nodes)
print("Average degree of karate club network is {}".format(avg_degree))

def average_clustering_coefficient(G):
  # TODO: Implement this function that takes a nx.Graph
  # and returns the average clustering coefficient. Round
  # the result to 2 decimal places (for example 3.333 will
  # be rounded to 3.33 and 3.7571 will be rounded to 3.76)

  avg_cluster_coef = round(nx.average_clustering(G), 2)

  ############# Your code here ############
  ## Note:
  ## 1: Please use the appropriate NetworkX clustering function

  #########################################

  return avg_cluster_coef

avg_cluster_coef = average_clustering_coefficient(G)
print("Average clustering coefficient of karate club network is {}".format(avg_cluster_coef))

def one_iter_pagerank(G, beta, r0, node_id):
  # TODO: Implement this function that takes a nx.Graph, beta, r0 and node id.
  # The return value r1 is one interation PageRank value for the input node.
  # Please round r1 to 2 decimal places.

  r1 = 0

  ############# Your code here ############
  r1 = nx.pagerank(G, alpha=beta, max_iter=1, nstart = {node: r0 for node in G.nodes()}, tol=1)
  r1 = round(r1[node_id], 2)
  ## Note:
  ## 1: You should not use nx.pagerank

  #########################################

  return r1

beta = 0.8
r0 = 1 / G.number_of_nodes()
node = 0
r1 = one_iter_pagerank(G, beta, r0, node)
print("The PageRank value for node 0 after one iteration is {}".format(r1))

def closeness_centrality(G, node=5):
  # TODO: Implement the function that calculates closeness centrality
  # for a node in karate club network. G is the input karate club
  # network and node is the node id in the graph. Please round the
  # closeness centrality result to 2 decimal places.
    total_distance = sum(nx.shortest_path_length(G, source=node).values())
    # Closeness centrality is the reciprocal of the total distance
    closeness = round(1.0 / total_distance, 2)

  ## Note:
  ## 1: You can use networkx closeness centrality function.
  ## 2: Notice that networkx closeness centrality returns the normalized
  ## closeness directly, which is different from the raw (unnormalized)
  ## one that we learned in the lecture.

  #########################################

    return closeness

node = 5
closeness = closeness_centrality(G, node=node)
print("The node 5 has closeness centrality {}".format(closeness))

def graph_to_edge_list(G):
  # TODO: Implement the function that returns the edge list of
  # an nx.Graph. The returned edge_list should be a list of tuples
  # where each tuple is a tuple representing an edge connected
  # by two nodes.


  ############# Your code here ############

  edge_list = list(nx.edges(G))

  #########################################

  return edge_list

def edge_list_to_tensor(edge_list):
  # TODO: Implement the function that transforms the edge_list to
  # tensor. The input edge_list is a list of tuples and the resulting
  # tensor should have the shape [2, len(edge_list)].

  # edge_index = torch.tensor([edge_list], dtype=torch.long).transpose(1, 2).view(2, -1)
  edge_index = torch.einsum("ijk->kj", torch.tensor([edge_list], dtype=torch.long))
  ############# Your code here ############

  #########################################

  return edge_index

print(G)
pos_edge_list = graph_to_edge_list(G)
pos_edge_index = edge_list_to_tensor(pos_edge_list)
print("The pos_edge_index tensor has shape {}".format(pos_edge_index.shape))
print("The pos_edge_index tensor has sum value {}".format(torch.sum(pos_edge_index)))


import random

def sample_negative_edges(G, num_neg_samples):
  # TODO: Implement the function that returns a list of negative edges.
  # The number of sampled negative edges is num_neg_samples. You do not
  # need to consider the corner case when the number of possible negative edges
  # is less than num_neg_samples. It should be ok as long as your implementation
  # works on the karate club network. In this implementation, self loops should
  # not be considered as either a positive or negative edge. Also, notice that
  # the karate club network is an undirected graph, if (0, 1) is a positive
  # edge, do you think (1, 0) can be a negative one?


  neg_edge_list = []

  ############# Your code here ############
  valid_num = 0
  while valid_num < num_neg_samples:
    nnodes = nx.number_of_nodes(G)
    src = random.randint(0, nnodes)
    dst = random.randint(0, nnodes)
    if src != dst and G.has_edge(src, dst):
        valid_num += 1
        neg_edge_list.append((src, dst))
  #########################################

  return neg_edge_list

# Sample 78 negative edges
neg_edge_list = sample_negative_edges(G, len(pos_edge_list))

# Transform the negative edge list to tensor
neg_edge_index = edge_list_to_tensor(neg_edge_list)
print("The neg_edge_index tensor has shape {}".format(neg_edge_index.shape))

# Which of following edges can be negative ones?
edge_1 = (7, 1)
edge_2 = (1, 33)
edge_3 = (33, 22)
edge_4 = (0, 4)
edge_5 = (4, 2)
############# Your code here ############
## Note:
## 1: For each of the 5 edges, print whether it can be negative edge

#########################################
print(G.has_edge(edge_1[0], edge_1[1]))
print(G.has_edge(edge_2[0], edge_2[1]))
print(G.has_edge(edge_3[0], edge_3[1]))
print(G.has_edge(edge_4[0], edge_4[1]))
print(G.has_edge(edge_5[0], edge_5[1]))

# Please do not change / reset the random seed
torch.manual_seed(1)

def create_node_emb(num_node=34, embedding_dim=16):
  # TODO: Implement this function that will create the node embedding matrix.
  # A torch.nn.Embedding layer will be returned. You do not need to change
  # the values of num_node and embedding_dim. The weight matrix of returned
  # layer should be initialized under uniform distribution.

  emb = torch.nn.Embedding(num_embeddings=num_node, embedding_dim = embedding_dim)

  ############# Your code here ############

  #########################################

  return emb

emb = create_node_emb()
ids = torch.LongTensor([0, 3])

# Print the embedding layer
print("Embedding: {}".format(emb))

# An example that gets the embeddings for node 0 and 3
print(emb(ids))

fig, axes = plt.subplots(2, figsize=(10, 8))

def visualize_emb(emb, idx):
  X = emb.weight.data.numpy()
  pca = PCA(n_components=2)
  components = pca.fit_transform(X)
  club1_x = []
  club1_y = []
  club2_x = []
  club2_y = []
  for node in G.nodes(data=True):
    if node[1]['club'] == 'Mr. Hi':
      club1_x.append(components[node[0]][0])
      club1_y.append(components[node[0]][1])
    else:
      club2_x.append(components[node[0]][0])
      club2_y.append(components[node[0]][1])
  axes[idx].scatter(club1_x, club1_y, color="red", label="Mr. Hi")
  axes[idx].scatter(club2_x, club2_y, color="blue", label="Officer")
  axes[idx].legend()
  axes[idx].set_title(f"{idx}")
  
visualize_emb(emb, 0)

from torch.optim import SGD
import torch.nn as nn

def accuracy(pred, label):
  # TODO: Implement the accuracy function. This function takes the
  # pred tensor (the resulting tensor after sigmoid) and the label
  # tensor (torch.LongTensor). Predicted value greater than 0.5 will
  # be classified as label 1. Else it will be classified as label 0.
  # The returned accuracy should be rounded to 4 decimal places.
  # For example, accuracy 0.82956 will be rounded to 0.8296.
    # Convert predictions to binary labels (0 or 1)
    pred_labels = (pred > 0.5).long()

    # Calculate the number of correct predictions
    correct = (pred_labels == label).sum().item()

    # Calculate accuracy
    accu = correct / len(label)

    # Round the accuracy to 4 decimal places
    accu = round(accu, 4)

    return accu
  ############# Your code here ############

  #########################################


def train(emb, loss_fn, sigmoid, train_label, train_edge):
  # TODO: Train the embedding layer here. You can also change epochs and
  # learning rate. In general, you need to implement:
  # (1) Get the embeddings of the nodes in train_edge
  # (2) Dot product the embeddings between each node pair
  # (3) Feed the dot product result into sigmoid
  # (4) Feed the sigmoid output into the loss_fn
  # (5) Print both loss and accuracy of each epoch
  # (6) Update the embeddings using the loss and optimizer
  # (as a sanity check, the loss should decrease during training)

  epochs = 500
  learning_rate = 0.1

  optimizer = SGD(emb.parameters(), lr=learning_rate, momentum=0.9)

  for i in range(epochs):

    ############# Your code here ############
    edge_embeddings = emb(train_edge)
    src = edge_embeddings[0, :, :]
    dst = edge_embeddings[1, :, :]
    logits = torch.einsum("ij,ij->i", src, dst)
    pred = sigmoid(logits)
    loss = loss_fn(pred, train_label)
    if i % 50 == 0:  # Print every 10 epochs
            acc = accuracy(pred, train_label)
            print(f"Epoch {i}: Loss = {loss.item():.4f}, Accuracy = {acc:.4f}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #########################################

loss_fn = nn.BCELoss()
sigmoid = nn.Sigmoid()

print(pos_edge_index.shape)

# Generate the positive and negative labels
pos_label = torch.ones(pos_edge_index.shape[1], )
neg_label = torch.zeros(neg_edge_index.shape[1], )

# Concat positive and negative labels into one tensor
train_label = torch.cat([pos_label, neg_label], dim=0)

# Concat positive and negative edges into one tensor
# Since the network is very small, we do not split the edges into val/test sets
train_edge = torch.cat([pos_edge_index, neg_edge_index], dim=1)
print(train_edge.shape)

train(emb, loss_fn, sigmoid, train_label, train_edge)

# Visualize the initial random embeddding
visualize_emb(emb, 1)


plt.show()
