from collections import Counter
import torch
import torch_geometric
import networkx as nx
from torch_geometric.utils import k_hop_subgraph
from torch_sparse import SparseTensor
import torch.nn.functional as F
import time
import tqdm
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.utils import to_networkx
import os
import torchvision.models as models
import torch.nn as nn
from torch_geometric.loader import NeighborLoader

from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.utils import add_self_loops, remove_self_loops, to_undirected, degree


import matplotlib.pyplot as plt

import random
import numpy as np
import torch
import torch.nn.functional as F


import torch


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def random_planetoid_splits(data, num_classes, percls_trn=20, val_lb=500, Flag=0):
    # Set new random planetoid splits:
    # * round(train_rate*len(data)/num_classes) * num_classes labels for training
    # * val_rate*len(data) labels for validation
    # * rest labels for testing

    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)

    if Flag is 0:
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(rest_index[:val_lb], size=data.num_nodes)
        data.test_mask = index_to_mask(
            rest_index[val_lb:], size=data.num_nodes)
    else:
        val_index = torch.cat([i[percls_trn:percls_trn+val_lb]
                               for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn+val_lb:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(val_index, size=data.num_nodes)
        data.test_mask = index_to_mask(rest_index, size=data.num_nodes)
    return data

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def straight_through(logits):
    index = logits.argmax(dim = -1)
    hard = F.one_hot(logits.argmax(dim=-1), num_classes=logits.size(-1))
    soft = F.softmax(logits, dim=-1)
    return index, hard.detach() + (soft - soft.detach())


def edge_index_to_dense(edge_index, num_nodes):
    # Create a sparse tensor from the edge index
    sparse_tensor = torch_geometric.utils.to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)

    # Convert the sparse tensor to a dense tensor
    dense_tensor = torch.tensor(sparse_tensor.todense())

    return dense_tensor

def extract_k_hop(edge_index, target_node, k=1, num_nodes=None):
    node_idx = torch.tensor([target_node])
    node_subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx, k, edge_index, relabel_nodes=True, num_nodes=num_nodes
    )
    return node_subset, sub_edge_index, mapping, edge_mask

def get_degree_set(edge_index, num_nodes=None):
    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1
    degree = torch_geometric.utils.degree(edge_index[0], num_nodes=num_nodes)
    return degree


#### Graphon related utils

def convert_2_graphon(dense_adj, size):
    x = dense_adj.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    if dense_adj.shape[0] < size:
        # x_up = F.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)
        x_up = F.interpolate(x, size=(size, size), mode='nearest')
        x_up = x_up.squeeze(0).squeeze(0)  # Remove batch and channel dimensions
        return x_up
    # Implement the conversion logic here
    else:
        # Adaptive mean pooling to exactly 512x512
        x_down = F.adaptive_avg_pool2d(x, (size, size))
        x_down = x_down.squeeze(0).squeeze(0)  # Remove batch and channel dimensions
        return x_down

def make_adj_to_graphon_scale(edge_index, num_nodes, hops, img_size, batch_size=512, device='cpu', save_file='./graphon/cora.pt'):
    graphon_list = []

    # convert once to SparseTensor for faster slicing
    adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(num_nodes, num_nodes)).to(device)

    all_nodes = torch.arange(num_nodes, device=device)
    for j in range(1, hops+1):
        all_adj = 0
        for batch_start in tqdm.tqdm(range(0, num_nodes, batch_size), desc="Processing nodes in batches"):
            batch_nodes = all_nodes[batch_start:batch_start+batch_size]
            batch_graphons = []

        
            # extract k-hop neighborhood for a batch
            # unfortunately PyG k_hop_subgraph is per-node, so here we approximate:
            # take adjacency power
            print(j)
            adj_power = adj[batch_nodes]
            adj_power_dense = adj_power.to_dense()
            for _ in range(j-1):
                adj_power = adj_power @ adj  # multiply sparse

            # slice adjacency rows for this batch
            sub_adj = adj_power
            
            # degree & sorting
            degrees = sub_adj.sum(dim=1)
            sorted_idx = torch.argsort(degrees, descending=True)

            # turn into fixed-size image (downsample if needed)
            dense = sub_adj.to_dense()
            if all_adj == 0:
                add_adj = dense
                all_adj = 1
            else:
                add_adj = torch.cat((add_adj, dense), dim=0)
            # dense = dense[sorted_idx][:, sorted_idx]
            # graphon = convert_2_graphon(dense, size=img_size)
            # batch_graphons.append(graphon)
        for i in tqdm.tqdm(range(0, add_adj.shape[0]), desc="Converting to graphon"):
            batch_dense = add_adj[i]
            nonzero_idx = batch_dense.nonzero().reshape(-1)
            nonzero_idx = torch.cat((nonzero_idx, torch.tensor([i], device=device)), dim=0)
            # print(batch_dense.shape, nonzero_idx, nonzero_idx.shape)
            temp_adj = add_adj[nonzero_idx][:, nonzero_idx]
            graphon = convert_2_graphon(temp_adj, size=img_size)
            graphon_list.append(graphon)
            # batch_dense = batch_dense[:, sorted_idx][:, :, sorted_idx]
            # batch_graphon = F.adaptive_avg_pool2d(batch_dense.unsqueeze(1), (img_size, img_size)).squeeze(1)
            # batch_graphons.append(batch_graphon)
        # batch_graphons = torch.stack(batch_graphons, dim=1)  # [batch, hops, img, img]
        # graphon_list.append(batch_graphons)

    graphon_list = torch.cat(graphon_list, dim=0)
    print(graphon_list.shape)  # [hops, num_nodes, img, img]
    graphon_list = torch.transpose(graphon_list, 0, 1)  # [num_nodes, hops, img, img]
    torch.save(graphon_list, save_file)
    return graphon_list

def make_adj_to_graphon_neighborloader(data):
    # NeighborLoader for k-hop subgraphs
    dict_list = [{},{},{},{}]
    for k in tqdm.tqdm(range(1, 4), desc="Processing k-hop subgraphs"):
        k = 2  # number of hops
        loader = NeighborLoader(
            data,
            input_nodes=None,                 # all nodes
            num_neighbors=[-1] * k,           # -1 => take *all* neighbors per hop
            batch_size=1,                     # 1 node per batch → local subgraph per node
            shuffle=False
        )

        # Iterate over all nodes
        subgraphs = []
        for subgraph in loader:
            center_node = subgraph.input_id.item()   # the node for which subgraph is extracted
            node_subset, sub_edge_index, mapping, edge_mask = extract_k_hop(subgraph.edge_index, center_node, k, num_nodes=data.num_nodes)
            dict_list[k-1][center_node] = edge_index_to_dense(sub_edge_index, num_nodes=len(node_subset))
    print('finish')
    exit(0)
def make_adj_to_graphon(edge_index, num_nodes, hops, img_size, save_file='./graphon/cora.pt'):
    graphon_list = []
    for i in tqdm.tqdm(range(0, num_nodes, 1), desc="Processing nodes"):
        for j in range(hops):
            node_subset, sub_edge_index, mapping, edge_mask = extract_k_hop(edge_index, i, j+1, num_nodes=num_nodes)
            degree_set = get_degree_set(sub_edge_index, num_nodes=len(node_subset))
            degree_tensor = torch.tensor(degree_set, device=sub_edge_index.device)
            _, sorted_indices = torch.sort(degree_tensor, descending=True)
            sub_dense_adj = edge_index_to_dense(sub_edge_index, num_nodes=len(node_subset))

            sub_dense_adj = sub_dense_adj[sorted_indices, :][:, sorted_indices]

            graphon = convert_2_graphon(sub_dense_adj, size=img_size)
            graphon_list.append(graphon)
    graphon_list = torch.stack(graphon_list)
    graphon_list = graphon_list.reshape(num_nodes, hops, img_size, img_size)
    print(graphon_list.shape)
    torch.save(graphon_list, save_file)
    return graphon_list



def pretrained_embed(graphon_list, resnet):
    batch_size = 128  # Adjust based on your GPU memory
    dataset = TensorDataset(graphon_list)
    loader = DataLoader(dataset, batch_size=batch_size)

    embeddings = []
    with torch.no_grad():
        for batch in loader:
            batch_data = batch[0].cuda()
            batch_embedding = resnet(batch_data)
            embeddings.append(batch_embedding)
    embedding = torch.cat(embeddings, dim=0)

    embedding = embedding / embedding.norm(dim=1, keepdim=True)
    return embedding

def get_max_score(embedding):
    # embedding = embedding / embedding.norm(dim=1, keepdim=True)
    cos_score = torch.matmul(embedding, embedding.T)
    cos_score[torch.arange(embedding.size(0)), torch.arange(embedding.size(0))] = 0

    max_score_value, max_score_indices = torch.max(cos_score, dim=1)
    return max_score_indices

def draw_pretrained_emb(max_score_indices, data, graphon_list, dataset_name):
    for i, target in enumerate(max_score_indices):
        print(f"Image {i} is most similar to image {target}")
        # print(f"Similarity score: {max_score_value[i]}")
        G = to_networkx(data, to_undirected=True)
        # Nodes between which you want the shortest path
        src, dst = i, target.item()
        # Get shortest path length
        try:
            length = nx.shortest_path_length(G, source=src, target=dst)
            print(f"The shortest path length between {src} and {dst} is {length}")
        except nx.NetworkXNoPath:
            print(f"No path between {src} and {dst}")
        for j in range(3):
            img1 = graphon_list[i,j].cpu().squeeze().numpy()
            img2 = graphon_list[target,j].cpu().squeeze().numpy()

            fig, axs = plt.subplots(1, 2, figsize=(8, 4))
            axs[0].imshow(img1, cmap='gray')
            axs[0].set_title(f'Image {i}')
            axs[0].axis('off')
            axs[1].imshow(img2, cmap='gray')
            axs[1].set_title(f'Image {target}')
            axs[1].axis('off')
            save_dir = f'./images/{dataset_name}/images_pretrained_encoder'
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f'./images/{dataset_name}/images_pretrained_encoder/similarity_{i}_{target}_{j}.png')

def draw_initial_emb(max_score_indices, data, graphon_list, dataset_name):
    correct = 0
    total = 0
    for i, target in enumerate(max_score_indices):
        print(f"Image {i} is most similar to image {target}. Labels: {data.y[i]}, {data.y[target]}")
        if data.y[i] == data.y[target]:
            correct += 1
        total += 1
        # print(f"Similarity score: {max_score_value[i]}")
        G = to_networkx(data, to_undirected=True)
        # Nodes between which you want the shortest path
        src, dst = i, target.item()
        # Get shortest path length
        try:
            length = nx.shortest_path_length(G, source=src, target=dst)
            print(f"The shortest path length between {src} and {dst} is {length}")
        except nx.NetworkXNoPath:
            print(f"No path between {src} and {dst}")
        for j in range(3):
            img1 = graphon_list[i,j].cpu().squeeze().numpy()
            img2 = graphon_list[target,j].cpu().squeeze().numpy()

            fig, axs = plt.subplots(1, 2, figsize=(8, 4))
            axs[0].imshow(img1, cmap='gray')
            axs[0].set_title(f'Image {i}')
            axs[0].axis('off')
            axs[1].imshow(img2, cmap='gray')
            axs[1].set_title(f'Image {target}')
            axs[1].axis('off')
            save_dir = f'./images/{dataset_name}/images_initial_encoder_norm'
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f'./images/{dataset_name}/images_initial_encoder_norm/similarity_{i}_{target}_{j}.png')
    print("ratio of correct: ", correct / total if total > 0 else 0)
def print_pretrained_emb_label_dist(max_score_indices, data, graphon_list, dataset_name):
    for i, target in enumerate(max_score_indices):
        print(f"Image {i} is most similar to image {target}")
        # print(f"Similarity score: {max_score_value[i]}")
        G = to_networkx(data, to_undirected=True)
        # Nodes between which you want the shortest path
        src, dst = i, target.item()
        # Get shortest path length
        try:
            length = nx.shortest_path_length(G, source=src, target=dst)
            
        except nx.NetworkXNoPath:
            # print(f"No path between {src} and {dst}")
            length = 0
        if length < 4:
            continue
        print(f"Image {i} is most similar to image {target}")
        print(f"The shortest path length between {src} and {dst} is {length}")
        for j in range(3):
            src_node_subset, src_sub_edge_index, src_mapping, src_edge_mask = k_hop_subgraph(
                src, j+1, data.edge_index, relabel_nodes=False, num_nodes=data.num_nodes
            )
            dst_node_subset, dst_sub_edge_index, dst_mapping, dst_edge_mask = k_hop_subgraph(
                dst, j+1, data.edge_index, relabel_nodes=False, num_nodes=data.num_nodes
            )
            src_node_subset_wo_src = src_node_subset[src_node_subset != src]
            dst_node_subset_wo_dst = dst_node_subset[dst_node_subset != dst]
            print(torch.bincount(data.y[src_node_subset_wo_src], minlength=int(data.y.max()) + 1), data.y[src], j)
            print(torch.bincount(data.y[dst_node_subset_wo_dst], minlength=int(data.y.max()) + 1), data.y[dst], j)
            print("*" * 50)
        print("8" * 50)
        time.sleep(1)

def print_initial_emb_label_dist(max_score_indices, data, graphon_list, dataset_name):
    for i, target in enumerate(max_score_indices):
        
        # print(f"Similarity score: {max_score_value[i]}")
        G = to_networkx(data, to_undirected=True)
        # Nodes between which you want the shortest path
        src, dst = i, target.item()
        # Get shortest path length
        try:
            length = nx.shortest_path_length(G, source=src, target=dst)
            # print(f"The shortest path length between {src} and {dst} is {length}")
        except nx.NetworkXNoPath:
            # print(f"No path between {src} and {dst}")
            length = 0
        if length < 4:
            continue
        print(f"Image {i} is most similar to image {target}")
        print(f"The shortest path length between {src} and {dst} is {length}")
        for j in range(3):
            src_node_subset, src_sub_edge_index, src_mapping, src_edge_mask = k_hop_subgraph(
                src, j+1, data.edge_index, relabel_nodes=False, num_nodes=data.num_nodes
            )
            dst_node_subset, dst_sub_edge_index, dst_mapping, dst_edge_mask = k_hop_subgraph(
                dst, j+1, data.edge_index, relabel_nodes=False, num_nodes=data.num_nodes
            )
            src_node_subset_wo_src = src_node_subset[src_node_subset != src]
            dst_node_subset_wo_dst = dst_node_subset[dst_node_subset != dst]
            print(torch.bincount(data.y[src_node_subset_wo_src], minlength=int(data.y.max()) + 1), data.y[src], j)
            print(torch.bincount(data.y[dst_node_subset_wo_dst], minlength=int(data.y.max()) + 1), data.y[dst], j)
            print("*" * 50)
        print("8" * 50)
        time.sleep(1)


def get_max_score_raw(graphon_list):
    # Norm need?
    
    graphon_list = graphon_list / (graphon_list.norm(dim=-1, keepdim=True)+1)
    
    graphon_squeeze = graphon_list.reshape(graphon_list.shape[0], graphon_list.shape[1], -1)
    graphon_squeeze = graphon_squeeze.reshape(-1, graphon_squeeze.shape[-1])

    # Norm need? (X)
    # graphon_squeeze = graphon_squeeze / graphon_squeeze.norm(dim=1, keepdim=True)

    cos_score = torch.matmul(graphon_squeeze, graphon_squeeze.T)
    cos_score[torch.arange(cos_score.size(0)), torch.arange(cos_score.size(0))] = 0
    max_score_value, max_score_indices = torch.max(cos_score, dim=1)
    return max_score_indices

def draw_raw_emb(max_score_indices, data, graphon_list, dataset_name):
    for i, target in enumerate(max_score_indices):
        print(f"Image {i} ({i//graphon_list.shape[1]}, {i - (i//graphon_list.shape[1])*graphon_list.shape[1]}) is most similar to image {target} ({target//graphon_list.shape[1]}, {target - (target//graphon_list.shape[1])*graphon_list.shape[1]})")
        # print(f"Similarity score: {max_score_value[i]}")
        G = to_networkx(data, to_undirected=True)
        # Nodes between which you want the shortest path
        src, dst = i//graphon_list.shape[1], target.item()//graphon_list.shape[1]
        # Get shortest path length
        try:
            length = nx.shortest_path_length(G, source=src, target=dst)
            print(f"The shortest path length between {src} and {dst} is {length}")
        except nx.NetworkXNoPath:
            print(f"No path between {src} and {dst}")
        img1 = graphon_list[i//graphon_list.shape[1], i - (i//graphon_list.shape[1])*graphon_list.shape[1]].cpu().squeeze().numpy()
        img2 = graphon_list[target//graphon_list.shape[1], target - (target//graphon_list.shape[1])*graphon_list.shape[1]].cpu().squeeze().numpy()

        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(img1, cmap='gray')
        axs[0].set_title(f'Image {i}')
        axs[0].axis('off')
        axs[1].imshow(img2, cmap='gray')
        axs[1].set_title(f'Image {target}')
        axs[1].axis('off')
        save_dir = f'./images/{dataset_name}/images_raw'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f'./images/{dataset_name}/images_raw/similarity_{i//graphon_list.shape[1]}_{target//graphon_list.shape[1]}_{i - (i//graphon_list.shape[1])*graphon_list.shape[1]}_{target - (target//graphon_list.shape[1])*graphon_list.shape[1]}.png')


def print_raw_emb_label_dist(max_score_indices, data, graphon_list, dataset_name):
    for i, target in enumerate(max_score_indices):
        
        # print(f"Similarity score: {max_score_value[i]}")
        G = to_networkx(data, to_undirected=True)
        # Nodes between which you want the shortest path
        src, dst = i//graphon_list.shape[1], target.item()//graphon_list.shape[1]
        if src == dst:
            continue
        print(f"Image {i} ({i//graphon_list.shape[1]}, {i - (i//graphon_list.shape[1])*graphon_list.shape[1]}) is most similar to image {target} ({target//graphon_list.shape[1]}, {target - (target//graphon_list.shape[1])*graphon_list.shape[1]})")
        src_hop = i - (i//graphon_list.shape[1])*graphon_list.shape[1] + 1
        dst_hop = target - (target//graphon_list.shape[1])*graphon_list.shape[1] + 1
        # Get shortest path length
        try:
            length = nx.shortest_path_length(G, source=src, target=dst)
            print(f"The shortest path length between {src} and {dst} is {length}")
        except nx.NetworkXNoPath:
            print(f"No path between {src} and {dst}")
        src_node_subset, src_sub_edge_index, src_mapping, src_edge_mask = k_hop_subgraph(
        src, src_hop, data.edge_index, relabel_nodes=False, num_nodes=data.num_nodes
        )
        dst_node_subset, dst_sub_edge_index, dst_mapping, dst_edge_mask = k_hop_subgraph(
            dst, dst_hop, data.edge_index, relabel_nodes=False, num_nodes=data.num_nodes
        )
        src_node_subset_wo_src = src_node_subset[src_node_subset != src]
        dst_node_subset_wo_dst = dst_node_subset[dst_node_subset != dst]
        print(torch.bincount(data.y[src_node_subset_wo_src], minlength=int(data.y.max()) + 1), data.y[src])
        print(torch.bincount(data.y[dst_node_subset_wo_dst], minlength=int(data.y.max()) + 1), data.y[dst])
        time.sleep(1)



def load_graph_classification_dataset(dataset_name, deg4feat=False):
    dataset_name = dataset_name.upper()
    dataset = TUDataset(root="./data", name=dataset_name)
    dataset = list(dataset)
    graph = dataset[0]


    if graph.x == None:
        if graph.y and not deg4feat:
            print("Use node label as node features")
            feature_dim = 0
            for g in dataset:
                feature_dim = max(feature_dim, int(g.y.max().item()))
            
            feature_dim += 1
            for i, g in enumerate(dataset):
                node_label = g.y.view(-1)
                feat = F.one_hot(node_label, num_classes=int(feature_dim)).float()
                dataset[i].x = feat
        else:
            print("Using degree as node features")
            feature_dim = 0
            degrees = []
            for g in dataset:
                feature_dim = max(feature_dim, degree(g.edge_index[0]).max().item())
                degrees.extend(degree(g.edge_index[0]).tolist())
            MAX_DEGREES = 400

            oversize = 0
            for d, n in Counter(degrees).items():
                if d > MAX_DEGREES:
                    oversize += n
            # print(f"N > {MAX_DEGREES}, #NUM: {oversize}, ratio: {oversize/sum(degrees):.8f}")
            feature_dim = min(feature_dim, MAX_DEGREES)

            feature_dim += 1
            for i, g in enumerate(dataset):
                degrees = degree(g.edge_index[0])
                degrees[degrees > MAX_DEGREES] = MAX_DEGREES
                degrees = torch.Tensor([int(x) for x in degrees.numpy().tolist()])
                feat = F.one_hot(degrees.to(torch.long), num_classes=int(feature_dim)).float()
                g.x = feat
                dataset[i] = g

    else:
        print("******** Use `attr` as node features ********")
    feature_dim = int(graph.num_features)

    labels = torch.tensor([x.y for x in dataset])
    
    num_classes = torch.max(labels).item() + 1
    for i, g in enumerate(dataset):
        dataset[i].edge_index = remove_self_loops(dataset[i].edge_index)[0]
        dataset[i].edge_index = add_self_loops(dataset[i].edge_index)[0]
    #dataset = [(g, g.y) for g in dataset]

    print(f"******** # Num Graphs: {len(dataset)}, # Num Feat: {feature_dim}, # Num Classes: {num_classes} ********")
    return dataset, (feature_dim, num_classes)


def make_adj_to_graphon_graph(graphs, hops, img_size, save_file='./graphon/cora.pt'):
    graph_list = []
    for graph in graphs:
        graphon_list = []
        edge_index = graph.edge_index
        num_nodes = graph.num_nodes
        for i in tqdm.tqdm(range(0, num_nodes, 1), desc="Processing nodes"):
            for j in range(hops):
                node_subset, sub_edge_index, mapping, edge_mask = extract_k_hop(edge_index, i, j+1, num_nodes=num_nodes)
                degree_set = get_degree_set(sub_edge_index, num_nodes=len(node_subset))
                degree_tensor = torch.tensor(degree_set, device=sub_edge_index.device)
                _, sorted_indices = torch.sort(degree_tensor, descending=True)
                sub_dense_adj = edge_index_to_dense(sub_edge_index, num_nodes=len(node_subset))

                sub_dense_adj = sub_dense_adj[sorted_indices, :][:, sorted_indices]

                graphon = convert_2_graphon(sub_dense_adj, size=img_size)
                graphon_list.append(graphon)
        graphon_list = torch.stack(graphon_list)
        graphon_list = graphon_list.reshape(num_nodes, hops, img_size, img_size)
        graph_list.append(graphon_list)
    graph_list = torch.cat(graph_list, dim=0)
    print(graph_list.shape)
    torch.save(graph_list, save_file)
    return graph_list

def input_struct_graph(graphs, graphon_list):
    struct_graph = []
    pointer = 0
    for graph in graphs:
        num_nodes = graph.num_nodes
        struct_graph.append({
            'edge_index': graph.edge_index,
            'num_nodes': num_nodes,
            'y': graph.y,
            'graphon': graphon_list[pointer:pointer+num_nodes]
        })
        pointer += num_nodes
    return struct_graph