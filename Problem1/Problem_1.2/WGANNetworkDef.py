import torch.nn as nn

class MLP_Network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP_Network, self).__init__()
        #Input
        module = [ nn.Linear(input_size, hidden_size) ,  nn.ReLU() ]
        #Hidden layers 1
        module.append(nn.Linear(hidden_size, hidden_size) )
        module.append(nn.ReLU())
        #Ouput layer
        module.append(nn.Linear(hidden_size, 1) )
        self.Network = nn.Sequential(*module)
        self.Network.apply(self.init_weights)
        
    def forward(self, x):
        return (self.Network( x ))
    
    def init_weights(self, layer):
        if type(layer) == nn.Linear:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                layer.bias.data.fill_(0.0)