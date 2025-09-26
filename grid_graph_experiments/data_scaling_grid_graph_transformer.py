#!/usr/bin/env python
# coding: utf-8

# # Transformers Can Learn Connectivity in Some Graphs but Not Others

# # Import Libraries

# In[ ]:


import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, ConfusionMatrixDisplay, f1_score
from tqdm import tqdm
from copy import deepcopy


import sys
import math
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


# # Grid Graph generation

# In[ ]:


from math import ceil

class Node(object):
    def __init__(self, id):
        self.id = id
        self.children = []
        self.parents = []
        
def add_edge(dag, u, v):
    u.children.append(v)
    v.parents.append(u)

def remove_edge(dag, u, v):
    u.children.remove(v)
    v.parents.remove(u)

def has_edge(dag, u, v):
    return v in u.children

def generate_grid_dag(n,p):
    
    b = ceil(n ** (1/p))
    
    dag = []
    node_map = {}

    for node in range(n):
        new_node = Node(node)
        dag.append(new_node)
        node_map[node] = new_node
        
        
    cell2node = {}
    node2cell = {}
    
    for node in range(n):
        rep = []
        base = b
        num = node
        
        for j in range(p):
            rep.append(num%b)
            num = num // b
        
        node2cell[Node(node)] = tuple(rep)
        cell2node[tuple(rep)] = Node(node)
        
    
    for u in node2cell:
        rep = list(node2cell[u])
        
        for j in range(p):
            cur = deepcopy(rep)
            cur[j] = cur[j] + 1
            cur = tuple(cur)
            
            if cur in cell2node.keys():
                v = cell2node[cur]
                add_edge(dag,node_map[u.id],node_map[v.id])
    
    
    return dag, node_map


# # DAG Utils

# In[ ]:




def compute_distances(start):
    queue = [(start,0)]
    distances = {}
    while len(queue) != 0:
        current,distance = queue.pop()
        if current in distances and distances[current] <= distance:
            continue
        distances[current] = distance
        for child in current.children:
            queue.append((child,distance+1))
    return distances

def compute_all_distances(graph):
    distances = {}
    for u in graph:
        distances[u] = compute_distances(u)
    return distances


def generate_train_test_pairs(dag,node_map,M_test):
    
    distances = compute_all_distances(dag)
    
    train_positive_edges = []
    train_negative_edges = []
    
    test_positive_edges = []
    test_negative_edges = []
    
    for u in dag:
        for v in dag:
            if u == v:
                continue
            
            
            if v in distances[u]:
                train_positive_edges.append((u.id,v.id))
            else:
                train_negative_edges.append((u.id,v.id))
    
    
    i = 0
    
    while i < (M_test//2):
        
        idx = random.randrange(len(train_positive_edges))
        u,v = train_positive_edges[idx]
        
        node1 = node_map[u]
        node2 = node_map[v]
        
        if distances[node1][node2] == 1 :
            continue
            
        i = i + 1
        
        test_positive_edges.append((u,v))
        del train_positive_edges[idx]
        
        test_negative_edges.append((v,u))
        negative_idx = train_negative_edges.index((v,u))
        del train_negative_edges[negative_idx]
    
    
    
    non_reverse_edges = []
    node_list = sorted([u.id for u in dag])
    for i in range(len(node_list)):
        for j in range(i+1,len(node_list)):
            
            u = node_map[node_list[i]]
            v = node_map[node_list[j]]
            
            if v not in distances[u] and u not in distances[v]:
                non_reverse_edges.append((u.id,v.id))
    
    random.shuffle(non_reverse_edges)
    sampled_edges = non_reverse_edges[:(M_test//2)]
    
    for edge in sampled_edges:
        u,v = edge[0],edge[1]
        
        test_negative_edges.append((u,v))
        test_negative_edges.append((v,u))
                
        negative_idx = train_negative_edges.index((u,v))
        del train_negative_edges[negative_idx]
        negative_idx = train_negative_edges.index((v,u))
        del train_negative_edges[negative_idx]
    

    return train_positive_edges, train_negative_edges, test_positive_edges, test_negative_edges



def merge_pairs(positive_pairs,positive_labels,negative_pairs,negative_labels):

    pairs = positive_pairs + negative_pairs
    labels = positive_labels + negative_labels
    
    temp = list(zip(pairs, labels))
    random.shuffle(temp)
    pairs, labels = zip(*temp)
    pairs, labels = list(pairs), list(labels)
    
    return pairs, labels


# # Dataset Processing

# In[ ]:



def build_graph_vocab(node_list):
    vocab = {}
    for node in sorted(node_list):
        vocab[str(node)] = node
    
    vocab["NOT"] = len(vocab)
    vocab["N"] = 0
    vocab["Y"] = 1
    
    idx_to_word = {idx: word for word, idx in vocab.items()}
    return vocab, idx_to_word

class DAGDataset(Dataset):
    def __init__(self, vocab, graph_prompts, label_prompts, max_length):
        self.vocab = vocab
        self.max_length = max_length
        self.graph_prompts = graph_prompts
        self.label_prompts = label_prompts
        self.data = self.generate_data()
        
    
    def generate_data(self):
        
        data = []
        total_samples = len(self.graph_prompts)
        for idx in range(total_samples):
            graph_tokens = self.graph_prompts[idx]
            label_token = self.label_prompts[idx]
            input_ids = graph_tokens
            target = [self.vocab[label_token]] 
            
            data.append((input_ids, target))
            
        return data
            
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids, target = self.data[idx][0], self.data[idx][1]  
        return torch.tensor(input_ids), torch.tensor(target)
    

def create_dataloader(vocab, graph_prompts, label_prompts, max_length, batch_size):
    num_negatives = sum([x == 'N' for x in label_prompts])
    num_positives = len(label_prompts) - num_negatives
    weights = [1/num_negatives if x == 'N' else 1/num_positives for x in label_prompts]

    dataset = DAGDataset(vocab, graph_prompts, label_prompts, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=WeightedRandomSampler(weights, batch_size, replacement=True), pin_memory=True)
    return dataloader


# # Position Encoding

# In[ ]:


def absolute_position_embeddings(seq_length):
    position = torch.arange(seq_length).unsqueeze(0)
    position_embeddings = F.one_hot(position, num_classes=seq_length).float().to(device)
    return position_embeddings


# # Transformer Block

# In[ ]:


class SelfAttention(nn.Module):
    def __init__(self,d_model,dropout):
        super(SelfAttention,self).__init__()
        
        self.d_model = d_model
        self.query_embed = nn.Linear(d_model,d_model)
        self.key_embed = nn.Linear(d_model,d_model)
        self.value_embed = nn.Linear(d_model,d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,x):
        x_residual = x
        query = self.query_embed(x)
        key = self.key_embed(x)
        key_transposed = torch.transpose(key, 1, 2)
        value = self.value_embed(x)
        attention_weights = torch.matmul(query, key_transposed)  # (n_query,n_key)
        attention_weights = attention_weights / math.sqrt(self.d_model)
        attention_weights = F.softmax(attention_weights, dim=2)
        attention_weighted_value = torch.matmul(
            attention_weights, value
        )
        attention_weighted_value = self.dropout(attention_weighted_value)
        return attention_weighted_value
    

class Transformer_Layer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, dropout,device):
        super(Transformer_Layer, self).__init__()
        
        self.attention_blocks = nn.ModuleList(
            [SelfAttention(d_model,dropout) for _ in range(num_heads)]
        )
        
        self.out_proj = nn.Linear(d_model*num_heads,d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_ff, d_model)        
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        
        self.vocab_size = vocab_size
        self.device = device
        self.num_heads = num_heads
        

    def forward(self, x):
        
                
        residual_x = x
        attention_out = torch.tensor([], requires_grad=True).to(self.device)
        for attention in self.attention_blocks:
            attention_out = torch.cat((attention_out, attention(x)), dim=2)
        attention_out = self.out_proj(attention_out)
        
        x_after_attention = self.norm1(attention_out+residual_x)
        
        x = self.fc1(x_after_attention)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x_out = self.norm2(x+x_after_attention)
        
        return x_out

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, dropout,device):
        super(Transformer, self).__init__()
        
        
        
        self.transformer_blocks = nn.ModuleList(
            [Transformer_Layer(vocab_size, d_model, num_heads, d_ff, dropout,device) for _ in range(num_layers)]
        )
        
        
        self.encoder_embedding = nn.Embedding(vocab_size, d_emb)
        self.pe = absolute_position_embeddings(max_seq_length)
            
        self.vocab_size = vocab_size
    
    def forward(self, x):
        
        x = self.encoder_embedding(x)
        
        pe = self.pe[:,:x.size(1),:]
        pe = pe.repeat(x.size(0),1,1)
        hidden_encoding = torch.zeros(x.size(0), x.size(1), args.d_hid).to(device)
        
        concatenated_input = (x,pe,hidden_encoding)
        x = torch.cat(concatenated_input, dim = 2)

        
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        
        x_out = x[:,-1,:total_nodes]
        return x_out


# # Training and Testing

# In[ ]:


# Training loop
def train(model, dataloader, optimizer, criterion, device, total_samples):
    model.train()
    total_loss = 0
    cnt = 0
    total_correct = 0
    
    
    y = []
    y_pred = []
    
    for inputs, targets in dataloader:
        
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        loss = criterion(logits, targets.reshape(-1))
        
        no_token = vocab["N"]
        prediction_logits = logits[:,no_token:no_token+2]
        predicted_label = no_token + torch.argmax(prediction_logits, dim=-1).unsqueeze(-1)
        
        targets_list = targets.cpu().numpy().tolist()
        predicted_list = predicted_label.cpu().numpy().tolist()
        
        y = y + targets_list
        y_pred = y_pred + predicted_list

        total_correct += torch.sum(predicted_label == targets)
        total_loss += loss.item()
        cnt += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    avg_loss = total_loss / cnt
    acc = accuracy_score(y,y_pred)
    return avg_loss, acc

def test(model, dataloader, optimizer, criterion, device, total_samples,print_flag = False):
    model.eval()
    total_loss = 0
    cnt = 0
    total_correct = 0
    
    
    y = []
    y_pred = []
    
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        loss = criterion(logits, targets.reshape(-1))
        
        no_token = vocab["N"]
        prediction_logits = logits[:,no_token:no_token+2]
        predicted_label = no_token + torch.argmax(prediction_logits, dim=-1).unsqueeze(-1)
         
        
        targets_list = targets.cpu().numpy().tolist()
        predicted_list = predicted_label.cpu().numpy().tolist()
        
        y = y + targets_list
        y_pred = y_pred + predicted_list
        
        if print_flag:
            input_list = inputs.cpu().numpy().tolist()
            for i in range(len(input_list)):
                u = input_list[i][0]
                v = input_list[i][1]
                target = targets_list[i]
                predicted = predicted_list[i]
                
                if target != predicted:
                    print(u,v,target,predicted)
            
        
        
        total_correct += torch.sum(predicted_label == targets)
        total_loss += loss.item()
        cnt += 1

    
    avg_loss = total_loss / cnt
    acc = accuracy_score(y,y_pred)
    c_matrix = confusion_matrix(y,y_pred,labels=[0,1])
    c_matrix = c_matrix/total_samples
    
    return avg_loss, acc


# # Configure Hyperparameter and Execute Training

# In[ ]:


parser = argparse.ArgumentParser(description='parameters')
parser.add_argument('-f')

parser.add_argument('--d_emb', type=int, default=256)
parser.add_argument('--d_hid', type=int, default=32) 
parser.add_argument('--d_pos', type=int, default=2)  
parser.add_argument('--heads', type=int, default=2)
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--max_seq_length', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--num_epochs', type=int, default=10000)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--show_loss', type=bool, default=True)

parser.add_argument('--total_nodes', type=int, default=100)
parser.add_argument('--grid_dim', type=int, default=2)
parser.add_argument('--M_test', type=int, default=40)
args = parser.parse_args()
print(args)


# ## Execute Data Scaling Grid Graph Experiments

# In[ ]:


num_nodes= [50,100,200,400,800]
num_epochs = [15000,10000,10000,5000,5000]


for i in range(len(num_nodes)):
               
    args.total_nodes = total_nodes = num_nodes[i]
    args.num_epochs = num_epochs[i]

    
    seeds = [10,18,42,7,96,43,44,45,46,47]
    for seed in seeds:

        print(f"Running with d_emb = {args.d_emb} and seed = {seed}")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


        # generate dag
        dag, node_map = generate_grid_dag(args.total_nodes,args.grid_dim)


        train_positive_pairs, train_negative_pairs, test_positive_pairs, test_negative_pairs = generate_train_test_pairs (dag, node_map, args.M_test)
        train_positive_labels = ["Y"] * len(train_positive_pairs)
        train_negative_labels = ["N"] * len(train_negative_pairs)
        test_positive_labels = ["Y"] * len(test_positive_pairs)
        test_negative_labels = ["N"] * len(test_negative_pairs)

        node_list = sorted([u.id for u in dag])
        total_nodes = len(node_list)
        print("Total Nodes: ",total_nodes)


        train_x, train_y = merge_pairs(train_positive_pairs,train_positive_labels,train_negative_pairs,train_negative_labels)
        test_x, test_y = merge_pairs(test_positive_pairs,test_positive_labels,test_negative_pairs,test_negative_labels)

        print("Total Train Edges: ", len(train_x))
        print("Total Test  Edge: ", len(test_x))


        # build vocab
        vocab, idx_to_word = build_graph_vocab(node_list)
        vocab_size = total_nodes


        heads = args.heads
        num_layers = args.num_layers
        dropout = args.dropout
        max_seq_length = args.max_seq_length  # Maximum length of input sequence
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        batch_size = args.batch_size

        d_emb = args.d_emb
        d_pos = args.d_pos
        d_hid = args.d_hid

        d_model = d_emb + d_pos + d_hid

        d_ff = d_model 
        args.d_ff = d_model

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_dataloader = create_dataloader(vocab, train_x, train_y , max_seq_length, batch_size)
        test_dataloader = create_dataloader(vocab, test_x, test_y , max_seq_length, batch_size)

        def lr_lambda(epoch):
            base_lr = learning_rate
            factor = 0.01
            return base_lr/(1+factor*epoch)


        model = Transformer(vocab_size, d_model, heads, num_layers, d_ff, dropout, device).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay = weight_decay)
        scheduler = lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=1000)
        criterion = nn.CrossEntropyLoss()

        best_train_acc, best_train_precision, best_train_recall, best_train_f1score, best_train_support, best_train_actual_acc = 0,0,0,0,0,0
        best_val_acc, best_val_precision, best_val_recall, best_val_f1score, best_val_support = 0,0,0,0,0
        best_test_acc, best_test_precision, best_test_recall, best_test_f1score, best_test_support  = 0,0,0,0,0

        train_loss_list, test_loss_list, train_actual_loss_list = [], [] , []
        train_acc_list, test_acc_list = [],[]


        best_train_epoch, best_train_actual_epoch, best_test_epoch = 0,0,0
        best_model = None
        best_all_model = None

        best_pos_acc = 0
        best_neg_acc = 0

        best_pos_epoch = 0
        best_neg_epoch = 0

        best_all_acc, besh_all_epoch = 0, 0

        flops_file = "flops_" + str(total_nodes) + "_grid_dim_" + str(args.grid_dim) + "_d_model_" + str(d_model) + ".npy"

        train_loss_file = "seed_" +  str(seed) + "_train_loss_" + str(total_nodes) + "_grid_dim_" + str(args.grid_dim) + "_d_model_" + str(d_model) + ".npy"
        test_loss_file = "seed_" +  str(seed) + "_test_loss_" + str(total_nodes) + "_grid_dim_" + str(args.grid_dim) + "_d_model_" + str(d_model) + ".npy"

        train_acc_file = "seed_" +  str(seed) + "_train_acc_" + str(total_nodes) + "_grid_dim_" + str(args.grid_dim) + "_d_model_" + str(d_model) + ".npy"
        test_acc_file = "seed_" +  str(seed) + "_test_acc_" + str(total_nodes) + "_grid_dim_" + str(args.grid_dim) + "_d_model_" + str(d_model) + ".npy"

        flops = []

        N= 2* d_model*args.num_layers*(2*d_model+args.d_ff)


        epoch = 0

        while True:
            epoch += 1

            if epoch > args.num_epochs:
                break

            avg_train_loss, train_acc = train(model, train_dataloader, optimizer, criterion, device, len(train_dataloader))

            if train_acc > best_train_acc : 
                best_train_epoch = epoch

            flop_per_epoch = 6*N*2*len(train_x)
            flops.append(flop_per_epoch*epoch)

            best_train_acc = max(best_train_acc, train_acc)
            train_loss_list.append(avg_train_loss)
            train_acc_list.append(train_acc)

            avg_test_loss, test_acc = test(model, test_dataloader, optimizer, criterion, device, len(test_dataloader))

            if test_acc > best_test_acc:
                best_test_epoch = epoch

            best_test_acc = max(best_test_acc, test_acc)
            test_loss_list.append(avg_test_loss)
            test_acc_list.append(test_acc)

            print("Epoch {}: Best Acc : Tr : {:0.6f} ({}),  Te: {:0.6f} ({}), Loss : Tr : {:0.6f} Te: {:0.6f}".format(
                epoch+1,best_train_acc,best_train_epoch , best_test_acc,best_test_epoch , avg_train_loss, avg_test_loss))


            np.save(flops_file,flops)

            np.save(train_loss_file,train_loss_list)
            np.save(test_loss_file,test_loss_list)

            np.save(train_acc_file,train_acc_list)
            np.save(test_acc_file,test_acc_list)

            plt.show()

            if args.show_loss and epoch % 100 == 0:
                plt.plot(flops,train_loss_list,label="Train Loss",color='g')
                plt.plot(flops,test_loss_list,label="Test Loss",color='r')
                ax = plt.gca()
                ax.set_xscale('log',base=10)
                ax.set_yscale('log',base=10)
                plt.legend(loc="upper right")
                plt.show()
                
                plt.plot(flops,train_acc_list,label="Train Acc",color='g')
                plt.plot(flops,test_acc_list,label="Test Acc",color='r')
                ax = plt.gca()
                ax.set_xscale('log',base=10)
                plt.legend(loc="lower right")
                plt.show()


# In[ ]:




