import torch
import torch.nn as nn
import torch.nn.functional as F

class FGM_GraphSAGE():
    def __init__(self, model, layer_names=["conv1", "conv2"], epsilon=1.0):
        """
        Fast Gradient Method (FGM) for adversarial training on GraphSAGE layers.
        :param model: The GraphSAGE autoencoder model
        :param layer_names: List of layer names to apply adversarial perturbation
        :param epsilon: The perturbation strength
        """
        self.model = model
        self.epsilon = epsilon
        self.layer_names = layer_names
        self.backup = {}
    
    def attack(self):
        """ Apply adversarial perturbation to the model's target layers. """
        for name, param in self.model.named_parameters():
            if param.requires_grad and any(layer in name for layer in self.layer_names):
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)
    
    def restore(self):
        """ Restore the original parameters after adversarial training. """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

class GraphSAGEConv(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphSAGEConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.FloatTensor(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data)

    def forward(self, node_features, adjacency_matrix):
        neighbors_aggregated = torch.matmul(adjacency_matrix, node_features)
        output = torch.matmul(neighbors_aggregated, self.W)
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

class GraphSAGE_autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, train_W, dropout):
        super(GraphSAGE_autoencoder, self).__init__()
        self.conv1 = GraphSAGEConv(input_size, hidden_size)
        self.conv2 = GraphSAGEConv(hidden_size, output_size)
        self.decoder = Decoder(train_W)     
        self.dropout = dropout 
        
    def forward(self, node_features, adjacency_matrix, drug_num, target_num, fgm=None):
        x = F.relu(self.conv1(node_features, adjacency_matrix))
        x = F.dropout(x, p=self.dropout, training=self.training)  # Dropout
        x = F.relu(self.conv2(x, adjacency_matrix))
        x = F.dropout(x, p=self.dropout, training=self.training)  # Dropout
        
        if fgm is not None:
            fgm.attack()  # Apply adversarial perturbation
        
        decoder_output = self.decoder(x, drug_num, target_num)
        
        if fgm is not None:
            fgm.restore()  # Restore original parameters
        
        return decoder_output





