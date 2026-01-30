import torch
import networkx as nx
from torch_geometric.datasets import Planetoid, WikipediaNetwork, HeterophilousGraphDataset, WebKB, Actor, Amazon
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from ogb.nodeproppred import PygNodePropPredDataset
from utils import *
from model import *
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

import torch_geometric.transforms as T

import torchvision.models as models
import torch.nn as nn

import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--hops', type=int, default=3)
parser.add_argument('--img_size', type=int, default=512)
parser.add_argument('--dataset_name', type=str, default='cora')
# parser.add_argument('--save_file', type=str, default=f'./graphon/{dataset_name}.pt')
parser.add_argument('--embed_dim', type=int, default=64) # 128 256 512
#embed_dim, num_layers, num_heads, mlp_dim, dropout
parser.add_argument('--num_layers', type=int, default=12) # 2 3 4
parser.add_argument('--num_heads', type=int, default=4) # 16 32
parser.add_argument('--mlp_dim', type=int, default=64) # 128 256 512
parser.add_argument('--dropout', type=float, default=0.7) # 0.7 0.8 0.9
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--img_embed_num', type=int, default=100) # 20 30 40
parser.add_argument('--seed', type=int, default=2026)
parser.add_argument('--missing_rate', type=int, default=0)
parser.add_argument('--model', type=str, default='gcn') # MyModel_without_VQ_Gate_Mix, SimpleGCN
parser.add_argument('--mode', type=str, default='homo') # MyModel_without_VQ_Gate_Mix, SimpleGCN

args = parser.parse_args()

args.save_file = f'./graphon/{args.dataset_name}.pt'

hops = args.hops
img_size = args.img_size
dataset_name = args.dataset_name
save_file = args.save_file

set_seed(args.seed)

device = "cpu"#f"cuda:0" if torch.cuda.is_available() else "cpu"

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
    # dataset = HeterophilousGraphDataset(root='../data/', name='roman-empire')
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
    # dataset = PygNodePropPredDataset(name=args.dataset_name, root='../data/')
    # data = dataset[0]
    split_idx = dataset.get_idx_split()
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[split_idx['train']] = True
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask[split_idx['valid']] = True
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask[split_idx['test']] = True
    data.y = data.y.squeeze()
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


image_dataloader_three = ScalableSubgraphDataset(
        edge_index=data.edge_index, 
        num_nodes=data.num_nodes, 
        k_hops=3,          # Look 3 steps out
        matrix_size=512    # Output is always 512x512
)

image_dataloader_two = ScalableSubgraphDataset(
        edge_index=data.edge_index, 
        num_nodes=data.num_nodes, 
        k_hops=2,          # Look 2 steps out
        matrix_size=512    # Output is always 512x512
)

image_dataloader_one = ScalableSubgraphDataset(
        edge_index=data.edge_index, 
        num_nodes=data.num_nodes, 
        k_hops=1,          # Look 1 step out
        matrix_size=512    # Output is always 512x512
)

resnet = models.resnet50(pretrained=True)
resnet.fc = nn.Identity()  # remove final classifier
resnet = resnet.cuda()

try:
    embedding = torch.load(f'./graphon/{args.dataset_name}_resnet50.pt')
except:
    embedding = None

    loader_one = DataLoader(image_dataloader_one, batch_size=128, shuffle=False, num_workers=32) # set num_workers > 0 on Linux
    loader_two = DataLoader(image_dataloader_two, batch_size=128, shuffle=False, num_workers=32) # set num_workers > 0 on Linux
    loader_three = DataLoader(image_dataloader_three, batch_size=128, shuffle=False, num_workers=32) # set num_workers > 0 on Linux

    # for batch_one, batch_two, batch_three in zip(loader_one, loader_two, loader_three):
    #     batch_one = batch_one.cuda()
    #     batch_two = batch_two.cuda()
    #     batch_three = batch_three.cuda()
    #     with torch.no_grad():
    #         img_embed_one = resnet(batch_one.expand(-1, 3, -1, -1))
    #         img_embed_two = resnet(batch_two.expand(-1, 3, -1, -1))
    #         img_embed_three = resnet(batch_three.expand(-1, 3, -1, -1))
            
    #         img_embed = torch.stack([img_embed_one, img_embed_two, img_embed_three], dim=1)
    #         print(img_embed.shape)
    #         exit(0)
    #         if embedding is None:
    #             embedding = img_embed
    #         else:
    #             embedding = torch.cat([embedding, img_embed], dim=0)
    #     del batch_one, batch_two, batch_three, img_embed_one, img_embed_two, img_embed_three, img_embed
    #     torch.cuda.empty_cache()

    # We assume all loaders have the same length.
    total_batches = len(loader_one) 

    # Wrap the zip with tqdm
    for batch_one, batch_two, batch_three in tqdm(zip(loader_one, loader_two, loader_three), total=total_batches, desc="Extracting Features"):
        
        batch_one = batch_one.cuda()
        batch_two = batch_two.cuda()
        batch_three = batch_three.cuda()
        
        with torch.no_grad():
            # Expand 1-channel input to 3-channel for ResNet (1, H, W) -> (3, H, W)
            img_embed_one = resnet(batch_one.expand(-1, 3, -1, -1))
            img_embed_two = resnet(batch_two.expand(-1, 3, -1, -1))
            img_embed_three = resnet(batch_three.expand(-1, 3, -1, -1))
            
            # Concatenate features
            img_embed = torch.stack([img_embed_one, img_embed_two, img_embed_three], dim=1)
            
            # Accumulate results
            # NOTE: Keeping this on GPU might cause Out Of Memory (OOM) for large datasets.
            # Consider .cpu() if you crash.
            if embedding is None:
                embedding = img_embed.cpu() # Recommended: Move to CPU immediately
            else:
                embedding = torch.cat([embedding, img_embed.cpu()], dim=0)
                
        # Cleanup to help VRAM
        # del batch_one, batch_two, batch_three, img_embed_one, img_embed_two, img_embed_three, img_embed
        # torch.cuda.empty_cache() is usually slow inside a loop; 
        # relying on Python's garbage collection (del) is often sufficient and faster.

    fan_in = embedding.shape[-1]
    fan_out = embedding.shape[-1]  # you can adjust depending on how you interpret the dims
    limit = (6 / (fan_in + fan_out)) ** 0.5  # Xavier uniform bound

    # apply Xavier uniform initialization
    embedding.uniform_(-limit, limit)
    torch.nn.init.orthogonal_(embedding, gain=0.5)
    embedding = embedding.cuda()
    torch.save(embedding, f'./graphon/{args.dataset_name}_resnet50.pt')
# print(embedding.shape)
# exit(0)




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
                    ### input_dim, class_dim, embed_dim, num_layers, num_heads, mlp_dim, dropout=0.1

# my_model = MyModel(data.num_features, int(max(data.y))+1, 128, 4, 16, 256, 0.5) # later change it into parse (Cora)
my_model = MyModel_large_KDD_diff(data.num_features, int(max(data.y))+1, args.embed_dim, args.num_layers, args.num_heads, args.mlp_dim, args.model, args.dropout, image_dim=embedding.shape[-1], img_embed_num=args.img_embed_num) # later change it into parse (Citeseer) # Current Best
# my_model = MyModel_large_KDD_squirrel(data.num_features, int(max(data.y))+1, args.embed_dim, args.num_layers, args.num_heads, args.mlp_dim, args.dropout, image_dim=embedding.shape[-1], img_embed_num=args.img_embed_num) # later change it into parse (Citeseer) # Current Best
my_model = my_model.cuda()

# optimizer = torch.optim.Adam(my_model.parameters(), lr=args.lr, weight_decay=5e-4) #(Cora, CIteseer)
optimizer = torch.optim.Adam(my_model.parameters(), lr=args.lr, weight_decay=2e-3) #(Cora, CIteseer)
# optimizer = torch.optim.Adam(my_model.parameters(), lr=args.lr, weight_decay=5e-4) #(Cora, CIteseer)
# optimizer = torch.optim.Adam(my_model.parameters(), lr=0.0002, weight_decay=2e-3) # Pubmed
criterion = nn.CrossEntropyLoss()

best_acc = 0

my_model.train()
for epoch in range(400):
    optimizer.zero_grad()
    out = my_model(data.x, data.edge_index, embedding, mode=args.mode)
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
            out = my_model(data.x, data.edge_index, embedding, mode=args.mode)
            pred = out.argmax(dim=1)
            correct = pred[data.val_mask] == data.y[data.val_mask]
            val_acc = int(correct.sum()) / int(data.val_mask.sum())
            loss_val = criterion(out[data.val_mask], data.y[data.val_mask])
            # img_label = my_model.get_img_label(embedding).reshape(-1)
            # print(torch.bincount(img_label), img_label[:3])
            print(f'Validation Accuracy: {val_acc:.4f}')
            if val_acc >= best_acc:
                best_acc = val_acc
                correct = pred[data.test_mask] == data.y[data.test_mask]
                test_acc = int(correct.sum()) / int(data.test_mask.sum())
                best_test_acc = test_acc
        my_model.train()

with open(f'./results/{args.dataset_name}_gate_{args.missing_rate}_{args.model}_ICML.txt', 'a') as f:
    f.write(f'embed_dim: {args.embed_dim}, num_layers: {args.num_layers}, num_heads: {args.num_heads}, mlp_dim: {args.mlp_dim}, dropout: {args.dropout}, lr: {args.lr}, img_embed_num: {args.img_embed_num}, best_acc: {best_test_acc}\n')

