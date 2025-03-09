import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FGM_GIN():
    def __init__(self, model, emb_name="embedding_layer.weight"):
        self.model = model
        self.emb_name = emb_name
        self.backup = {}
    
    def attack(self, epsilon=1.0):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    param.data.add_(epsilon * param.grad / norm)
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


class GINLayer(nn.Module):
    def __init__(self, input_dim, output_dim, eps=0, train_eps=False):
        super(GINLayer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        self.eps = eps
        self.train_eps = train_eps

    def forward(self, x, adj_matrix):
        if self.train_eps:
            self.eps = nn.Parameter(torch.Tensor([self.eps]))
        x = x + (1 + self.eps) * torch.matmul(adj_matrix, x)
        x = self.mlp(x)
        return x

class Decoder(nn.Module):
    def __init__(self, train_W):
        super().__init__()
        self.train_W = train_W  
        
    def forward(self, H, drug_num, target_num):
        HR = H[0:drug_num]
        HD = H[drug_num:(drug_num+target_num)]
        supp1 = torch.mm(HR, self.train_W)
        decoder = torch.mm(supp1, HD.transpose(0, 1))    
        return decoder 

class GIN_autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, train_W):
        super(GIN_autoencoder, self).__init__()
        self.embedding_layer = nn.Linear(input_dim, hidden_dim)
        self.gin_layers = nn.ModuleList([
            GINLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.decoder = Decoder(train_W) 

    def forward(self, x, adj_matrix, drug_num, target_num):
        x = self.embedding_layer(x)
        for layer in self.gin_layers:
            x = layer(x, adj_matrix)
        x = F.relu(x)
        decoder = self.decoder(x, drug_num, target_num) 
        return decoder 