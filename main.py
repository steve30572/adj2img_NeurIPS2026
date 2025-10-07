import torch
import networkx as nx
from torch_geometric.datasets import Planetoid, WikipediaNetwork, HeterophilousGraphDataset, WebKB, Actor, Amazon
from torch_geometric.utils import to_networkx
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from torch_geometric.data import Data
from ogb.nodeproppred import PygNodePropPredDataset
from utils import *
from model import *

import torch_geometric.transforms as T

import torchvision.models as models
import torch.nn as nn

import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--hops', type=int, default=3)
parser.add_argument('--img_size', type=int, default=512)
parser.add_argument('--dataset_name', type=str, default='chameleon')
parser.add_argument('--embed_dim', type=int, default=64) # 128 256 512
parser.add_argument('--num_layers', type=int, default=12) # 2 3 4
parser.add_argument('--num_heads', type=int, default=4) # 16 32
parser.add_argument('--mlp_dim', type=int, default=64) # 128 256 512
parser.add_argument('--dropout', type=float, default=0.7) # 0.7 0.8 0.9
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--img_embed_num', type=int, default=100) # 20 30 40
parser.add_argument('--seed', type=int, default=2026)
parser.add_argument('--missing_rate', type=int, default=0)

args = parser.parse_args()

args.save_file = f'./graphon/{args.dataset_name}.pt'

hops = args.hops
img_size = args.img_size
dataset_name = args.dataset_name
save_file = args.save_file

set_seed(args.seed)

device = "cpu"

transform = T.Compose([
    T.ToUndirected(),
    T.ToDevice(device),
])

# Load the Citeseer dataset
if args.dataset_name == "cora":
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]
elif args.dataset_name == "citeseer":
    dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')
    data = dataset[0]
elif args.dataset_name =='texas' or args.dataset_name == 'cornell' or args.dataset_name == 'wisconsin':
    preProcDs = WebKB(root='../data/',
                        name=args.dataset_name, transform=T.NormalizeFeatures())
    dataset = WebKB(root='../data/',
                            name=args.dataset_name, transform=T.NormalizeFeatures())
    data = dataset[0]
    data.edge_index = preProcDs[0].edge_index
    percls_trn = int(round(0.6*len(data.y)/dataset.num_classes))
    val_lb = int(round(0.2*len(data.y)))

    data = random_planetoid_splits(data, dataset.num_classes, percls_trn, val_lb)
elif args.dataset_name =='actor':
    preProcDs = Actor(
            root='../data/film', transform=T.NormalizeFeatures())
    dataset = Actor(
            root='../data/film', transform=T.NormalizeFeatures())
    data = dataset[0]
    data.edge_index = preProcDs[0].edge_index
    percls_trn = int(round(0.6*len(data.y)/dataset.num_classes))
    val_lb = int(round(0.2*len(data.y)))

    data = random_planetoid_splits(data, dataset.num_classes, percls_trn, val_lb)
elif args.dataset_name == 'roman-empire':
    
    preProcDs = HeterophilousGraphDataset(root='../data/', name='roman-empire')
    dataset = HeterophilousGraphDataset(root='../data/', name='roman-empire')
    data = dataset[0]
    data.edge_index = preProcDs[0].edge_index
    percls_trn = int(round(0.6*len(data.y)/dataset.num_classes))
    val_lb = int(round(0.2*len(data.y)))

    data = random_planetoid_splits(data, dataset.num_classes, percls_trn, val_lb)
elif args.dataset_name in {'Photo', 'Computers'}:
    dataset = Amazon('../data', args.dataset_name)
    data = transform(dataset[0])
    data = T.RandomNodeSplit(num_val=0.05, num_test=0.9)(data)
elif args.dataset_name in {'ogbn-arxiv', 'ogbn-products'}:
    transform = T.Compose([
    T.ToUndirected(),
    T.ToDevice(device),
])
    dataset = PygNodePropPredDataset(name=args.dataset_name, root='../data/')
    data = dataset[0]
    data = transform(data)
    split_idx = dataset.get_idx_split()
    data.train_mask = index_to_mask(split_idx['train'], data.num_nodes)
    data.val_mask = index_to_mask(split_idx['valid'], data.num_nodes)
    data.test_mask = index_to_mask(split_idx['test'], data.num_nodes)
    data.y = data.y.squeeze()
    data.edge_index = to_undirected(data.edge_index)
    data.edge_index, _ = remove_self_loops(data.edge_index)
    data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
    
else:
    preProcDs = WikipediaNetwork(
            root='../data/', name=args.dataset_name, geom_gcn_preprocess=False)#, transform=T.NormalizeFeatures())
    dataset = WikipediaNetwork(
        root='../data/', name=args.dataset_name, geom_gcn_preprocess=True)#, transform=T.NormalizeFeatures())
    data = dataset[0]
    data.edge_index = preProcDs[0].edge_index
    percls_trn = int(round(0.6*len(data.y)/dataset.num_classes))
    val_lb = int(round(0.2*len(data.y)))

    data = random_planetoid_splits(data, dataset.num_classes, percls_trn, val_lb)



try:
    graphon_list = torch.load(save_file)
except FileNotFoundError:
    graphon_list = make_adj_to_graphon(data.edge_index, data.num_nodes, hops, img_size, save_file=save_file)
    


try:
    embedding = torch.load(f'./graphon/{args.dataset_name}_embedding_dino.pt').cuda()
    fan_in = embedding.shape[-1]
    fan_out = embedding.shape[-1]  # you can adjust depending on how you interpret the dims
    limit = (6 / (fan_in + fan_out)) ** 0.5  # Xavier uniform bound
    embedding.uniform_(-limit, limit)
    torch.nn.init.orthogonal_(embedding, gain=0.5)
except FileNotFoundError:
    resnet = models.resnet50(pretrained=True)
    resnet.fc = nn.Identity()  # remove final classifier
    resnet = resnet.cuda()
    graphon_list = graphon_list.reshape(-1, 1, graphon_list.shape[-1], graphon_list.shape[-1])
    graphon_list = graphon_list.expand(-1, 3, -1, -1)
    embedding = pretrained_embed(graphon_list, resnet)
    embedding = embedding.reshape(-1, 3, embedding.shape[-1])

    torch.save(embedding.cpu(), f'./graphon/{args.dataset_name}_embedding.pt')
    # resnet = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    # # resnet.fc = nn.Identity()  # remove final classifier
    # resnet = resnet.cuda()
    # graphon_list = graphon_list.reshape(-1, 1, graphon_list.shape[-1], graphon_list.shape[-1])
    # graphon_list = graphon_list.expand(-1, 3, -1, -1)

    # mask = torch.rand_like(graphon_list) > 0.1
    # # graphon_list = graphon_list * mask

    # # change image into 224 * 224 (in case of dino)
    # graphon_list = torch.nn.functional.interpolate(graphon_list, size=(224, 224), mode='bilinear', align_corners=False)


    # embedding = pretrained_embed(graphon_list, resnet)
    # embedding = embedding.reshape(-1, 3, embedding.shape[-1])
    # torch.save(embedding.cpu(), f'./graphon/{args.dataset_name}_embedding_dino.pt')


if args.missing_rate == 1:
    ratio = 0.3

    # Mask out a ratio of features and replace them with the mean of the remaining features
    num_nodes, num_features = data.x.shape
    num_mask = int(ratio * num_features)

    # For reproducibility
    # torch.manual_seed(args.seed)

    # Randomly select features to mask for each node
    mask_indices = torch.rand(num_features).argsort()[:num_mask]
    # Compute mean of unmasked features for each node
    unmasked_indices = torch.tensor([i for i in range(num_features) if i not in mask_indices], device=data.x.device)
    # mean_values = data.x[:, unmasked_indices].mean(dim=1, keepdim=True)

    # Replace masked features with mean value
    x_masked = data.x.clone()
    x_masked[:, mask_indices] = 0

    data.x = x_masked
elif args.missing_rate == 2:
    ratio = 0.6

    # Mask out a ratio of features and replace them with the mean of the remaining features
    num_nodes, num_features = data.x.shape
    num_mask = int(ratio * num_features)

    # For reproducibility
    # torch.manual_seed(args.seed)

    # Randomly select features to mask for each node
    mask_indices = torch.rand(num_features).argsort()[:num_mask]
    # Compute mean of unmasked features for each node
    unmasked_indices = torch.tensor([i for i in range(num_features) if i not in mask_indices], device=data.x.device)
    # mean_values = data.x[:, unmasked_indices].mean(dim=1, keepdim=True)

    # Replace masked features with mean value
    x_masked = data.x.clone()
    x_masked[:, mask_indices] = 0

    data.x = x_masked



##### Train model
data = data.cuda()

my_model = MyModel_without_VQ_Gate_Mix(data.num_features, int(max(data.y))+1, args.embed_dim, args.num_layers, args.num_heads, args.mlp_dim, args.dropout, image_dim=embedding.shape[-1], img_embed_num=args.img_embed_num) # later change it into parse (Citeseer) # Current Best
if dataset_name in {'ogbn-arxiv', 'ogbn-products'}:
    my_model = MyModel_without_VQ_Gate_Mix_Large(data.num_features, int(max(data.y))+1, args.embed_dim, args.num_layers, args.num_heads, args.mlp_dim, args.dropout, image_dim=embedding.shape[-1], img_embed_num=args.img_embed_num) # later change it into parse (Citeseer) # Current Best
elif dataset_name in {'squirrel', 'chameleon' }:
    my_model = MyModel_with_VQ_Hetero_squirrel(data.num_features, int(max(data.y))+1, args.embed_dim, args.num_layers, args.num_heads, args.mlp_dim, args.dropout, image_dim=embedding.shape[-1], img_embed_num=args.img_embed_num) # later change it into parse (Citeseer) # Current Best
my_model = my_model.cuda()


optimizer = torch.optim.Adam(my_model.parameters(), lr=args.lr, weight_decay=5e-4) #(Cora, CIteseer)
criterion = nn.CrossEntropyLoss()


best_val_acc = 0
best_acc = 0

my_model.train()
for epoch in range(3000):
    optimizer.zero_grad()
    out = my_model(data.x, data.edge_index, embedding)

    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 5 == 0:
        my_model.eval()
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
        pred = out.argmax(dim=1)
        correct = pred[data.train_mask] == data.y[data.train_mask]
        train_acc = int(correct.sum()) / int(data.train_mask.sum())
        print(f'Train Accuracy: {train_acc:.4f}')
        
        with torch.no_grad():
            out = my_model(data.x, data.edge_index, embedding)
            pred = out.argmax(dim=1)
            correct = pred[data.valid_mask] == data.y[data.valid_mask]
            test_acc = int(correct.sum()) / int(data.valid_mask.sum())
            # img_label = my_model.get_img_label(embedding).reshape(-1)
            # print(torch.bincount(img_label), img_label[:3])
            print(f'Test Accuracy: {test_acc:.4f}')
            if test_acc >= best_val_acc:
                best_val_acc = test_acc
                out = my_model(data.x, data.edge_index, embedding)
                pred = out.argmax(dim=1)
                correct = pred[data.test_mask] == data.y[data.test_mask]
                test_acc = int(correct.sum()) / int(data.test_mask.sum())
                best_acc = test_acc
        my_model.train()

# with open(f'./results/{args.dataset_name}__{args.missing_rate}.txt', 'a') as f:
#     f.write(f'embed_dim: {args.embed_dim}, num_layers: {args.num_layers}, num_heads: {args.num_heads}, mlp_dim: {args.mlp_dim}, dropout: {args.dropout}, lr: {args.lr}, img_embed_num: {args.img_embed_num}, best_acc: {best_acc}\n')

