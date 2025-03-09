import torch
import torch.nn as nn
import torch.nn.functional as F


class FGM_GCN():
    def __init__(self, model, layer_names=["gc1", "gc2", "gc3"], epsilon=1.0):
        """
        Fast Gradient Method (FGM) for adversarial training on GCN layers.
        """
        self.model = model
        self.epsilon = epsilon
        self.layer_names = layer_names  
        self.backup = {}

    def attack(self):

        for name, param in self.model.named_parameters():
            if param.requires_grad and any(layer in name for layer in self.layer_names):
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


class GraphConvolution(nn.Module):
    def __init__(self, in_feature, out_feature, bias=True):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.weight = nn.Parameter(torch.FloatTensor(in_feature, out_feature))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_feature))
        nn.init.xavier_normal_(self.weight.data)
        if bias:
            self.bias.data.fill_(0.0)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.sparse.mm(adj, support)
        if hasattr(self, "bias") and self.bias is not None:
            return output + self.bias
        else:
            return output


class Decoder(nn.Module):
    def __init__(self, train_W):
        super().__init__()
        self.train_W = train_W

    def forward(self, H, drug_num, target_num):
        HR = H[0:drug_num]
        HD = H[drug_num:(drug_num + target_num)]
        supp1 = torch.mm(HR, self.train_W)
        decoder = torch.mm(supp1, HD.transpose(0, 1))
        return decoder


class GCN_decoder(nn.Module):
    def __init__(self, in_dim, hgcn_dim, train_W, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(in_dim, hgcn_dim)
        self.gc2 = GraphConvolution(hgcn_dim, hgcn_dim)
        self.gc3 = GraphConvolution(hgcn_dim, hgcn_dim)
        self.decoder = Decoder(train_W)
        self.dropout = dropout

    def forward(self, H, G, drug_num, target_num, fgm=None):
        H = self.gc1(H, G)
        H = F.leaky_relu(H, 0.25)
        H = self.gc2(H, G)
        H = F.leaky_relu(H, 0.25)
        H = self.gc3(H, G)
        H = F.leaky_relu(H, 0.25)

        if fgm is not None:
            fgm.attack()  # Apply adversarial perturbation

        decoder_output = self.decoder(H, drug_num, target_num)

        if fgm is not None:
            fgm.restore()  # Restore original parameters after adversarial training

        return decoder_output






     
    
        
    
    