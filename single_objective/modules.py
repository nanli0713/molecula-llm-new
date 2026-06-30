import torch
from torch import nn
import math

class GraphGather(nn.Module):

    def __init__(self, node_features,        
                 out_features,               
                 att_depth=2, att_hidden_dim=100, att_dropout_p=0.0,
                 emb_depth=2, emb_hidden_dim=100, emb_dropout_p=0.0):
        super(GraphGather, self).__init__()

        self.att_nn = FeedForwardNetwork(
            in_features      = node_features * 2,
            hidden_layer_sizes = [att_hidden_dim] * att_depth,
            out_features     = out_features,
            dropout_p        = att_dropout_p,
            bias             = False)

        self.emb_nn = FeedForwardNetwork(
            in_features      = node_features,
            hidden_layer_sizes = [emb_hidden_dim] * emb_depth,
            out_features     = out_features,
            dropout_p        = emb_dropout_p,
            bias             = False)

    # ------------------------------------------------------------------ #
    def forward(self, hidden_nodes,          # (B, N, d_node)  
                input_nodes,                 # (B, N, d_node)  
                node_mask):                  # (B, N)          1
        cat = torch.cat([hidden_nodes, input_nodes], dim=2)   # (B,N,2*d_node)
        energy_mask = (node_mask == 0).float() * 1e6           
        energies = self.att_nn(cat) - energy_mask.unsqueeze(-1)  # (B,N,d_out)

        attention = torch.sigmoid(energies)                    
        embedding = self.emb_nn(hidden_nodes)                  # (B,N,d_out)

        return torch.sum(attention * embedding, dim=1)         # (B,d_out)

class FeedForwardNetwork(nn.Module):

    def __init__(self, in_features,                
                 hidden_layer_sizes,               
                 out_features,                     
                 activation='SELU',                
                 bias=False,                       
                 dropout_p=0.0):                   
        super(FeedForwardNetwork, self).__init__()

        if activation == 'SELU':
            Activation   = nn.SELU
            Dropout      = nn.AlphaDropout         
            init_constant = 1.0                    # std = √(1/fan_in)
        elif activation == 'ReLU':
            Activation   = nn.ReLU
            Dropout      = nn.Dropout
            init_constant = 2.0                    
        else:
            raise ValueError('Unsupported activation')

        layer_sizes = [in_features] + hidden_layer_sizes + [out_features]
        layers = []
        for i in range(len(layer_sizes) - 2):      
            layers.append(Dropout(dropout_p))
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias))
            layers.append(Activation())

        layers.append(Dropout(dropout_p))
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1], bias))

        self.seq = nn.Sequential(*layers)          

        for idx in range(1, len(layers), 3):       
            lin = layers[idx]
            nn.init.normal_(lin.weight, std=math.sqrt(init_constant / lin.weight.size(1)))

    # ------------------------------------------------------------------ #
    def forward(self, input):
        return self.seq(input)

    # ------------------------------------------------------------------ #
    def __repr__(self):
        ffnn = type(self).__name__
        in_features  = self.seq[1].in_features
        hidden_sizes = [linear.out_features for linear in self.seq[1:-1:3]]
        out_features = self.seq[-1].out_features
        activation   = str(self.seq[2]) if len(self.seq) > 2 else 'None'
        bias         = self.seq[1].bias is not None
        dropout_p    = self.seq[0].p
        return (f'{ffnn}(in_features={in_features}, hidden_layer_sizes={hidden_sizes}, '
                f'out_features={out_features}, activation={activation}, '
                f'bias={bias}, dropout_p={dropout_p})')