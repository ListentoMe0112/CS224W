from typing import Dict, List, Tuple
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
import torch
import matplotlib.pyplot as plt
import copy
from pylab import show
from deepsnap.hetero_graph import HeteroGraph, Graph
from deepsnap.hetero_gnn import HeteroConv
from deepsnap.dataset import GraphDataset
import deepsnap
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

from sklearn.metrics import f1_score
from deepsnap.hetero_gnn import forward_op
from deepsnap.hetero_graph import HeteroGraph
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import scatter
import torch_sparse
import pandas as pd


def assign_node_types(G: nx.Graph, community_map):
    # TODO: Implement a function that takes in a NetworkX graph
    # G and community map assignment (mapping node id --> 0/1 label)
    # and adds 'node_type' as a node_attribute in G.

    ############# Your code here ############
    ## (~2 line of code) It's alright if you take up more lines!
    ## Note
    ## 1. Look up NetworkX `nx.classes.function.set_node_attributes`
    ## 2. Look above for the two node type values!

    #########################################

    community_label = {k: "n0" if v == 0 else "n1" for k, v in community_map.items()}
    nx.classes.function.set_node_attributes(G, community_label, "node_type")


def assign_node_labels(G, community_map):
    # TODO: Implement a function that takes in a NetworkX graph
    # G and community map assignment (mapping node id --> 0/1 label)
    # and adds 'node_label' as a node_attribute in G.

    ############# Your code here ############
    ## (~2 line of code) It's alright if you take up more lines!
    ## Note
    ## 1. Look up NetworkX `nx.classes.function.set_node_attributes`

    #########################################

    nx.classes.function.set_node_attributes(G, community_map, "node_label")


def assign_node_features(G):
    # TODO: Implement a function that takes in a NetworkX graph
    # G and adds 'node_feature' as a node_attribute in G. Each node
    # in the graph has the same feature vector - a torchtensor with
    # data [1., 1., 1., 1., 1.]

    ############# Your code here ############
    ## (~2 line of code) It's alright if you take up more lines!
    ## Note
    ## 1. Look up NetworkX `nx.classes.function.set_node_attributes`

    #########################################

    community_label = {k: [1, 1, 1, 1, 1] for k in community_map}
    nx.classes.function.set_node_attributes(G, community_label, "node_feature")


def assign_edge_types(G, community_map):
    # TODO: Implement a function that takes in a NetworkX graph
    # G and community map assignment (mapping node id --> 0/1 label)
    # and adds 'edge_type' as a edge_attribute in G.

    ############# Your code here ############
    ## (~5 line of code) It's alright if you take up more lines!
    ## Note
    ## 1. Create an edge assignment dict following rules above
    ## 2. Look up NetworkX `nx.classes.function.set_edge_attributes`

    nodes = G.nodes()
    community_edges = {}
    for edge in G.edges(data=True):
        if nodes[edge[0]]["node_type"] == "n0" and nodes[edge[1]]["node_type"] == "n0":
            community_edges[(edge[0], edge[1])] = {"edge_type": "e0"}
        elif (
            nodes[edge[0]]["node_type"] == "n1" and nodes[edge[1]]["node_type"] == "n1"
        ):
            community_edges[(edge[0], edge[1])] = {"edge_type": "e1"}
        else:
            community_edges[(edge[0], edge[1])] = {"edge_type": "e2"}
    nx.set_edge_attributes(G, community_edges)

    #########################################


def Question1_1() -> None:
    assign_node_types(G, community_map)
    assign_node_labels(G, community_map)
    assign_node_features(G)

    # Explore node properties for the node with id: 20
    node_id = 20
    print(f"Node {node_id} has properties:", G.nodes(data=True)[node_id])

    node_id = 19
    print(f"Node {node_id} has properties:", G.nodes(data=True)[node_id])


def Question1_2() -> None:
    assign_edge_types(G, community_map)

    # Explore edge properties for a sampled edge and check the corresponding
    # node types
    edge_idx = 15
    n1 = 0
    n2 = 31
    edge = list(G.edges(data=True))[edge_idx]
    print(f"Edge ({edge[0]}, {edge[1]}) has properties:", edge[2])
    print(f"Node {n1} has properties:", G.nodes(data=True)[n1])
    print(f"Node {n2} has properties:", G.nodes(data=True)[n2])


def draw_question_1_2() -> None:
    edge_color = {}
    for edge in G.edges():
        n1, n2 = edge
        edge_color[edge] = (
            community_map[n1] if community_map[n1] == community_map[n2] else 2
        )
        if community_map[n1] == community_map[n2] and community_map[n1] == 0:
            edge_color[edge] = "blue"
        elif community_map[n1] == community_map[n2] and community_map[n1] == 1:
            edge_color[edge] = "red"
        else:
            edge_color[edge] = "green"

    pos = nx.spring_layout(G)
    nx.classes.function.set_edge_attributes(G, edge_color, name="color")
    node_color = []
    color_map = {0: 0, 1: 1}
    node_color = [color_map[community_map[node]] for node in G.nodes()]
    colors = nx.get_edge_attributes(G, "color").values()
    labels = nx.get_node_attributes(G, "node_type")
    plt.figure(figsize=(8, 8))
    nx.draw(
        G,
        pos=pos,
        cmap=plt.get_cmap("coolwarm"),
        node_color=node_color,
        edge_color=colors,
        labels=labels,
        font_color="white",
    )
    show()


def get_nodes_per_type(hete: HeteroGraph):
    # TODO: Implement a function that takes a DeepSNAP dataset object
    # and return the number of nodes per `node_type`.

    num_nodes_n0 = hete.num_nodes("n0")
    num_nodes_n1 = hete.num_nodes("n1")

    ############# Your code here ############
    ## (~2 line of code)
    ## Note
    ## 1. Colab autocomplete functionality might be useful. Explore the attributes of HeteroGraph class.

    #########################################

    return num_nodes_n0, num_nodes_n1


def Question1_3():
    num_nodes_n0, num_nodes_n1 = get_nodes_per_type(hete)
    print("Node type n0 has {} nodes".format(num_nodes_n0))
    print("Node type n1 has {} nodes".format(num_nodes_n1))


def get_num_message_edges(hete: HeteroGraph):
    # TODO: Implement this function that takes a DeepSNAP dataset object
    # and return the number of edges for each message type.
    # You should return a list of tuples as
    # (message_type, num_edge)
    message_types = hete.message_types
    message_type_edges = []

    for message_type in message_types:
        num = hete.num_edges(message_type)
        message_type_edges.append((message_type, num))

    ############# Your code here ############
    ## (~2 line of code)
    ## Note
    ## 1. Colab autocomplete functionality might be useful. Explore the attributes of HeteroGraph class.

    #########################################

    return message_type_edges


def Question1_4():
    message_type_edges = get_num_message_edges(hete)
    for message_type, num_edges in message_type_edges:
        print("Message type {} has {} edges".format(message_type, num_edges))


def compute_dataset_split_counts(datasets: Dict[str, Tuple[Graph]]):
    # TODO: Implement a function that takes a dict of datasets in the form
    # {'train': dataset_train, 'val': dataset_val, 'test': dataset_test}
    # and returns a dict mapping dataset names to the number of labeled
    # nodes used for supervision in that respective dataset.

    data_set_splits = {}
    for k in datasets:
        graph = datasets[k][0]
        labeled_nodes_count = 0
        node_label_index = graph.node_label_index
        for node_type in node_label_index:
            labeled_nodes_count += len(node_label_index[node_type])
        data_set_splits[k] = labeled_nodes_count

    ############# Your code here ############
    ## (~3 line of code)
    ## Note
    ## 1. The DeepSNAP `node_label_index` dictionary will be helpful.
    ## 2. Remember to count both node_types
    ## 3. Remember each dataset only has one graph that we need to access
    ##    (i.e. dataset[0])

    #########################################

    return data_set_splits


def Question1_5():
    dataset = GraphDataset([hete], task="node")
    # Splitting the dataset
    dataset_train, dataset_val, dataset_test = dataset.split(
        transductive=True, split_ratio=[0.4, 0.3, 0.3]
    )
    datasets = {"train": dataset_train, "val": dataset_val, "test": dataset_test}

    data_set_splits = compute_dataset_split_counts(datasets)
    for dataset_name, num_nodes in data_set_splits.items():
        print("{} dataset has {} nodes".format(dataset_name, num_nodes))


def draw_question_1_5() -> None:
    dataset = GraphDataset([hete], task="node")
    edge_color = {}
    for edge in G.edges():
        n1, n2 = edge
        edge_color[edge] = (
            community_map[n1] if community_map[n1] == community_map[n2] else 2
        )
        if community_map[n1] == community_map[n2] and community_map[n1] == 0:
            edge_color[edge] = "blue"
        elif community_map[n1] == community_map[n2] and community_map[n1] == 1:
            edge_color[edge] = "red"
        else:
            edge_color[edge] = "green"

    pos = nx.spring_layout(G)
    nx.classes.function.set_edge_attributes(G, edge_color, name="color")
    node_color = []
    color_map = {0: 0, 1: 1}
    node_color = [color_map[community_map[node]] for node in G.nodes()]
    colors = nx.get_edge_attributes(G, "color").values()
    labels = nx.get_node_attributes(G, "node_type")

    # Splitting the dataset
    dataset_train, dataset_val, dataset_test = dataset.split(
        transductive=True, split_ratio=[0.4, 0.3, 0.3]
    )
    titles = ["Train", "Validation", "Test"]
    pos = nx.spring_layout(G)
    nx.classes.function.set_edge_attributes(G, edge_color, name="color")
    colors = nx.get_edge_attributes(G, "color").values()
    labels = nx.get_node_attributes(G, "node_type")
    plt.figure(figsize=(8, 8))
 

    for i, dataset in enumerate([dataset_train, dataset_val, dataset_test]):
        n0 = hete._convert_to_graph_index(
            dataset[0].node_label_index["n0"], "n0"
        ).tolist()
        n1 = hete._convert_to_graph_index(
            dataset[0].node_label_index["n1"], "n1"
        ).tolist()

        plt.figure(figsize=(7, 7))
        plt.title(titles[i])
        nx.draw(
            G_orig,
            pos=pos,
            node_color="grey",
            edge_color=colors,
            labels=labels,
            font_color="white",
        )
        nx.draw_networkx_nodes(G_orig.subgraph(n0), pos=pos, node_color="blue")
        nx.draw_networkx_nodes(G_orig.subgraph(n1), pos=pos, node_color="red")
        show()

class HeteroGNNConv(pyg_nn.MessagePassing):
    def __init__(self, in_channels_src, in_channels_dst, out_channels):
        super(HeteroGNNConv, self).__init__(aggr="mean")

        self.in_channels_src = in_channels_src
        self.in_channels_dst = in_channels_dst
        self.out_channels = out_channels

        # To simplify implementation, please initialize both self.lin_dst
        # and self.lin_src out_features to out_channels
        ############# Your code here #############
        ## (~3 lines of code)
        ## Note:
        ## 1. Initialize the 3 linear layers.
        ## 2. Think through the connection between the mathematical
        ##    definition of the update rule and torch linear layers!

        ##########################################

        self.lin_dst = nn.Linear(in_channels_dst, out_channels) 
        self.lin_src =nn.Linear(in_channels_src, out_channels)
        self.lin_update = nn.Linear(2*out_channels, out_channels)

    def forward(
        self,
        node_feature_src,
        node_feature_dst,
        edge_index,
        size=None
    ):
        ############# Your code here #############
        ## (~1 line of code)
        ## Note:
        ## 1. Unlike Colabs 3 and 4, we just need to call self.propagate with
        ## proper/custom arguments.

        ##########################################
        return self.propagate(edge_index = edge_index, size = size, node_feature_src = node_feature_src, node_feature_dst = node_feature_dst)

    def message_and_aggregate(self, edge_index, node_feature_src):

        ############# Your code here #############
        ## (~1 line of code)
        ## Note:
        ## 1. Different from what we implemented in Colabs 3 and 4, we use message_and_aggregate
        ##    to combine the previously seperate message and aggregate functions.
        ##    The benefit is that we can avoid materializing x_i and x_j
        ##    to make the implementation more efficient.
        ## 2. To implement efficiently, refer to PyG documentation for message_and_aggregate
        ##    and sparse-matrix multiplication:
        ##    https://pytorch-geometric.readthedocs.io/en/latest/notes/sparse_tensor.html
        ## 3. Here edge_index is torch_sparse SparseTensor. Although interesting, you
        ##    do not need to deeply understand SparseTensor represenations!
        ## 4. Conceptually, think through how the message passing and aggregation
        ##    expressed mathematically can be expressed through matrix multiplication.

        ##########################################

        out = torch_sparse.matmul(edge_index, node_feature_src, reduce=self.aggr)

        return out

    def update(self, aggr_out, node_feature_dst):

        ############# Your code here #############
        ## (~4 lines of code)
        ## Note:
        ## 1. The update function is called after message_and_aggregate
        ## 2. Think through the one-one connection between the mathematical update
        ##    rule and the 3 linear layers defined in the constructor.
        
        ##########################################
        aggr_out = torch.concatenate([self.lin_dst(node_feature_dst), self.lin_src(aggr_out)], dim = -1)
        aggr_out = self.lin_update(aggr_out)

        return aggr_out

class HeteroGNNWrapperConv(HeteroConv):
    def __init__(self, convs, args, aggr="mean"):
        super(HeteroGNNWrapperConv, self).__init__(convs, None)
        self.aggr = aggr

        # Map the index and message type
        self.mapping = {}

        # A numpy array that stores the final attention probability
        self.alpha = None

        self.attn_proj = None

        if self.aggr == "attn":
            ############# Your code here #############
            ## (~1 line of code)
            ## Note:
            ## 1. Initialize self.attn_proj, where self.attn_proj should include
            ##    two linear layers. Note, make sure you understand
            ##    which part of the equation self.attn_proj captures.
            ## 2. You should use nn.Sequential for self.attn_proj
            ## 3. nn.Linear and nn.Tanh are useful.
            ## 4. You can model a weight vector (rather than matrix) by using:
            ##    nn.Linear(some_size, 1, bias=False).
            ## 5. The first linear layer should have out_features as args['attn_size']
            ## 6. You can assume we only have one "head" for the attention.
            ## 7. We recommend you to implement the mean aggregation first. After
            ##    the mean aggregation works well in the training, then you can
            ##    implement this part.

            ##########################################
            self.attn_proj = nn.Sequential(
                nn.Linear(args['hidden_size'], args['attn_size'], bias=True),
                nn.Tanh(),
                nn.Linear(args['attn_size'], 1, bias=False)
            )

    def reset_parameters(self):
        super(HeteroConvWrapper, self).reset_parameters()
        if self.aggr == "attn":
            for layer in self.attn_proj.children():
                layer.reset_parameters()

    def forward(self, node_features, edge_indices):
        message_type_emb = {}
        for message_key, message_type in edge_indices.items():
            src_type, edge_type, dst_type = message_key
            node_feature_src = node_features[src_type]
            node_feature_dst = node_features[dst_type]
            edge_index = edge_indices[message_key]
            message_type_emb[message_key] = (
                self.convs[message_key](
                    node_feature_src,
                    node_feature_dst,
                    edge_index,
                )
            )
        node_emb = {dst: [] for _, _, dst in message_type_emb.keys()}
        mapping = {}
        for (src, edge_type, dst), item in message_type_emb.items():
            mapping[len(node_emb[dst])] = (src, edge_type, dst)
            node_emb[dst].append(item)
        self.mapping = mapping
        for node_type, embs in node_emb.items():
            if len(embs) == 1:
                node_emb[node_type] = embs[0]
            else:
                node_emb[node_type] = self.aggregate(embs)
        return node_emb

    def aggregate(self, xs):
        # TODO: Implement this function that aggregates all message type results.
        # Here, xs is a list of tensors (embeddings) with respect to message
        # type aggregation results.

        if self.aggr == "mean":

            ############# Your code here #############
            ## (~2 lines of code)
            ## Note:
            ## 1. Explore the function parameter `xs`!
            xs = torch.stack(xs, 0)
            out = torch.mean(xs, dim=0)
            return out 

            ##########################################

        elif self.aggr == "attn":
            N = xs[0].shape[0] # Number of nodes for that node type
            M = len(xs) # Number of message types for that node type

            x = torch.cat(xs, dim=0).view(M, N, -1) # M * N * D
            z = self.attn_proj(x).view(M, N) # M * N * 1
            z = z.mean(1) # M * 1
            alpha = torch.softmax(z, dim=0) # M * 1

            # Store the attention result to self.alpha as np array
            self.alpha = alpha.view(-1).data.cpu().numpy()

            alpha = alpha.view(M, 1, 1)
            x = x * alpha
            return x.sum(dim=0)

def generate_convs(hetero_graph : HeteroGraph, conv, hidden_size, first_layer=False):
    # TODO: Implement this function that returns a dictionary of `HeteroGNNConv`
    # layers where the keys are message types. `hetero_graph` is deepsnap `HeteroGraph`
    # object and the `conv` is the `HeteroGNNConv`.

    convs = {}

    ############# Your code here #############
    ## (~9 lines of code)
    ## Note:
    ## 1. See the hints above!
    ## 2. conv is of type `HeteroGNNConv`

    ##########################################
    all_message_types = hetero_graph.message_types
    if first_layer :
        for message_type in all_message_types:
            src_type, _, dst_type = message_type
            convs[message_type] =  conv(hetero_graph.num_node_features(src_type), hetero_graph.num_node_features(dst_type), hidden_size)
    else:
        for message_type in all_message_types:
            convs[message_type] =  conv(hidden_size, hidden_size, hidden_size)

    return convs

class HeteroGNN(torch.nn.Module):
    def __init__(self, hetero_graph : HeteroGraph, args, aggr="mean"):
        super(HeteroGNN, self).__init__()

        self.aggr = aggr
        self.hidden_size = args['hidden_size']

        self.convs1 = None
        self.convs2 = None

        self.bns1 = nn.ModuleDict()
        self.bns2 = nn.ModuleDict()
        self.relus1 = nn.ModuleDict()
        self.relus2 = nn.ModuleDict()
        self.post_mps = nn.ModuleDict()

        ############# Your code here #############
        ## (~10 lines of code)
        ## Note:
        ## 1. For self.convs1 and self.convs2, call generate_convs at first and then
        ##    pass the returned dictionary of `HeteroGNNConv` to `HeteroGNNWrapperConv`.
        ## 2. For self.bns, self.relus and self.post_mps, the keys are node_types.
        ##    `deepsnap.hetero_graph.HeteroGraph.node_types` will be helpful.
        ## 3. Initialize all batchnorms to torch.nn.BatchNorm1d(hidden_size, eps=1).
        ## 4. Initialize all relus to nn.LeakyReLU().
        ## 5. For self.post_mps, each value in the ModuleDict is a linear layer
        ##    where the `out_features` is the number of classes for that node type.
        ##    `deepsnap.hetero_graph.HeteroGraph.num_node_labels(node_type)` will be
        ##    useful.

        ##########################################

        self.convs1 = HeteroGNNWrapperConv(generate_convs(hetero_graph, HeteroGNNConv, self.hidden_size, True), args, aggr)
        self.convs2 = HeteroGNNWrapperConv(generate_convs(hetero_graph, HeteroGNNConv, self.hidden_size, False), args, aggr)

        all_node_types = hetero_graph.node_types

        for node_type in all_node_types:
            self.bns1[node_type] = torch.nn.BatchNorm1d(self.hidden_size, eps=1)
            self.bns2[node_type] = torch.nn.BatchNorm1d(self.hidden_size, eps=1)
            self.relus1[node_type] = nn.LeakyReLU()
            self.relus2[node_type] = nn.LeakyReLU()
            self.post_mps[node_type] = nn.Linear(self.hidden_size, hetero_graph.num_node_labels(node_type))


    def forward(self, node_feature, edge_index):
        # TODO: Implement the forward function. Notice that `node_feature` is
        # a dictionary of tensors where keys are node types and values are
        # corresponding feature tensors. The `edge_index` is a dictionary of
        # tensors where keys are message types and values are corresponding
        # edge index tensors (with respect to each message type).

        x = node_feature

        ############# Your code here #############
        ## (~7 lines of code)
        ## Note:
        ## 1. `deepsnap.hetero_gnn.forward_op` can be helpful.

        ##########################################
        x = self.convs1(x, edge_index)
        x = forward_op(x, self.bns1)
        x = forward_op(x, self.relus1)
        x = self.convs2(x,  edge_index)
        x = forward_op(x, self.bns2)
        x = forward_op(x, self.relus2)
        x = forward_op(x, self.post_mps)
        return x

    def loss(self, preds, y, indices):

        loss = 0
        loss_func = F.cross_entropy

        ############# Your code here #############
        ## (~3 lines of code)
        ## Note:
        ## 1. For each node type in preds, accumulate computed loss to `loss`
        ## 2. Loss need to be computed with respect to the given index
        ## 3. preds is a dictionary of model predictions keyed by node_type.
        ## 4. indeces is a dictionary of labeled supervision nodes keyed
        ##    by node_type

        ##########################################
        
        for node_type, pred in preds.items():
            node_indices = indices[node_type]
            loss += loss_func(pred[node_indices], y[node_type][node_indices])

        return loss


def train(model, optimizer, hetero_graph:HeteroGraph, train_idx):
    model.train()
    optimizer.zero_grad()
    preds = model(hetero_graph.node_feature, hetero_graph.edge_index)

    loss = None

    ############# Your code here #############
    ## Note:
    ## 1. Compute the loss here
    ## 2. `deepsnap.hetero_graph.HeteroGraph.node_label` is useful

    ##########################################
    
    loss = model.loss(preds, hetero_graph.node_label, train_idx)

    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, graph, indices, best_model=None, best_val=0, save_preds=False, agg_type=None):
    model.eval()
    accs = []
    for i, index in enumerate(indices):
        preds = model(graph.node_feature, graph.edge_index)
        num_node_types = 0
        micro = 0
        macro = 0
        for node_type in preds:
            idx = index[node_type]
            pred = preds[node_type][idx]
            pred = pred.max(1)[1]
            label_np = graph.node_label[node_type][idx].cpu().numpy()
            pred_np = pred.cpu().numpy()
            micro = f1_score(label_np, pred_np, average='micro')
            macro = f1_score(label_np, pred_np, average='macro')
            num_node_types += 1

        # Averaging f1 score might not make sense, but in our example we only
        # have one node type
        micro /= num_node_types
        macro /= num_node_types
        accs.append((micro, macro))

        # Only save the test set predictions and labels!
        if save_preds and i == 2:
          print ("Saving Heterogeneous Node Prediction Model Predictions with Agg:", agg_type)
          print()

          data = {}
          data['pred'] = pred_np
          data['label'] = label_np

          df = pd.DataFrame(data=data)
          # Save locally as csv
          df.to_csv('ACM-Node-' + agg_type + 'Agg.csv', sep=',', index=False)

    if accs[1][0] > best_val:
        best_val = accs[1][0]
        best_model = copy.deepcopy(model)
    return accs, best_model, best_val

if __name__ == "__main__":
    # G = nx.karate_club_graph()
    # community_map = {}
    # for node in G.nodes(data=True):
    #     if node[1]["club"] == "Mr. Hi":
    #         community_map[node[0]] = 0
    #     else:
    #         community_map[node[0]] = 1

    # Question1_1()
    # Question1_2()

    # G_orig = copy.deepcopy(G)
    # hete = HeteroGraph(G_orig)
    # # draw_question_1_2()
    # Question1_3()
    # Question1_4()
    # Question1_5()
    # draw_question_1_5()
    args = {
        'device': torch.device('cpu'),
        'hidden_size': 64,
        'epochs': 100,
        'weight_decay': 1e-5,
        'lr': 0.003,
        'attn_size': 32,
    }

    print("Device: {}".format(args['device']))

    # Load the data
    data = torch.load("acm.pkl")

    # Message types
    message_type_1 = ("paper", "author", "paper")
    message_type_2 = ("paper", "subject", "paper")

    # Dictionary of edge indices
    edge_index = {}
    edge_index[message_type_1] = data['pap']
    edge_index[message_type_2] = data['psp']

    # Dictionary of node features
    node_feature = {}
    node_feature["paper"] = data['feature']

    # Dictionary of node labels
    node_label = {}
    node_label["paper"] = data['label']

    # Load the train, validation and test indices
    train_idx = {"paper": data['train_idx'].to(args['device'])}
    val_idx = {"paper": data['val_idx'].to(args['device'])}
    test_idx = {"paper": data['test_idx'].to(args['device'])}

    # Construct a deepsnap tensor backend HeteroGraph
    hetero_graph = HeteroGraph(
        node_feature=node_feature,
        node_label=node_label,
        edge_index=edge_index,
        directed=True
    )

    print(f"ACM heterogeneous graph: {hetero_graph.num_nodes()} nodes, {hetero_graph.num_edges()} edges")

    # Node feature and node label to device
    for key in hetero_graph.node_feature:
        hetero_graph.node_feature[key] = hetero_graph.node_feature[key].to(args['device'])
    for key in hetero_graph.node_label:
        hetero_graph.node_label[key] = hetero_graph.node_label[key].to(args['device'])


    # Edge_index to sparse tensor and to device
    for key in hetero_graph.edge_index:
        edge_index = hetero_graph.edge_index[key]
        adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(hetero_graph.num_nodes('paper'), hetero_graph.num_nodes('paper')))
        hetero_graph.edge_index[key] = adj.t().to(args['device'])

    # best_model = None
    # best_val = 0

    # model = HeteroGNN(hetero_graph, args, aggr="mean").to(args['device'])
    # optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    # for epoch in range(args['epochs']):
    #     loss = train(model, optimizer, hetero_graph, train_idx)
    #     accs, best_model, best_val = test(model, hetero_graph, [train_idx, val_idx, test_idx], best_model, best_val)
    #     print(
    #         f"Epoch {epoch + 1}: loss {round(loss, 5)}, "
    #         f"train micro {round(accs[0][0] * 100, 2)}%, train macro {round(accs[0][1] * 100, 2)}%, "
    #         f"valid micro {round(accs[1][0] * 100, 2)}%, valid macro {round(accs[1][1] * 100, 2)}%, "
    #         f"test micro {round(accs[2][0] * 100, 2)}%, test macro {round(accs[2][1] * 100, 2)}%"
    #     )
    # best_accs, _, _ = test(best_model, hetero_graph, [train_idx, val_idx, test_idx], save_preds=True, agg_type="Mean")
    # print(
    #     f"Best model: "
    #     f"train micro {round(best_accs[0][0] * 100, 2)}%, train macro {round(best_accs[0][1] * 100, 2)}%, "
    #     f"valid micro {round(best_accs[1][0] * 100, 2)}%, valid macro {round(best_accs[1][1] * 100, 2)}%, "
    #     f"test micro {round(best_accs[2][0] * 100, 2)}%, test macro {round(best_accs[2][1] * 100, 2)}%"
    # )

    best_model = None
    best_val = 0

    output_size = hetero_graph.num_node_labels('paper')
    model = HeteroGNN(hetero_graph, args, aggr="attn").to(args['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    for epoch in range(args['epochs']):
        loss = train(model, optimizer, hetero_graph, train_idx)
        accs, best_model, best_val = test(model, hetero_graph, [train_idx, val_idx, test_idx], best_model, best_val)
        print(
            f"Epoch {epoch + 1}: loss {round(loss, 5)}, "
            f"train micro {round(accs[0][0] * 100, 2)}%, train macro {round(accs[0][1] * 100, 2)}%, "
            f"valid micro {round(accs[1][0] * 100, 2)}%, valid macro {round(accs[1][1] * 100, 2)}%, "
            f"test micro {round(accs[2][0] * 100, 2)}%, test macro {round(accs[2][1] * 100, 2)}%"
        )
    best_accs, _, _ = test(best_model, hetero_graph, [train_idx, val_idx, test_idx])
    print(
        f"Best model: "
        f"train micro {round(best_accs[0][0] * 100, 2)}%, train macro {round(best_accs[0][1] * 100, 2)}%, "
        f"valid micro {round(best_accs[1][0] * 100, 2)}%, valid macro {round(best_accs[1][1] * 100, 2)}%, "
        f"test micro {round(best_accs[2][0] * 100, 2)}%, test macro {round(best_accs[2][1] * 100, 2)}%"
    )


