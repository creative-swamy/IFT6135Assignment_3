import torch
import WGANNetworkDef as classdef
from torch.autograd import Variable
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class WGAN():
    def __init__(self, hidden_size, mini_batch, learning_rate, num_epochs, print_interval):
        super(WGAN, self).__init__()
        self.input_size = 2    
        self.hidden_size = hidden_size
        self.output_size = 1
        self.minibatch_size = mini_batch
        self.learning_rate = learning_rate        
        self.num_epochs = num_epochs
        self.print_interval = print_interval
        self.Network = classdef.MLP_Network(self.input_size, self.hidden_size, self.output_size)
        self.Network.to(device)
        self.optimizer = optim.Adam(self.Network.parameters(), lr=self.learning_rate)
        self.lambda_penalty = 10
    
    def gradient_penalty(self, x, y):
        aplha = (torch.empty(self.minibatch_size,1).uniform_(0,1)).to(device)
        interpol = aplha * x + (1-aplha)* y
        interpol.requires_grad = True
        interpol = interpol.to(device)
        D_interpol_data = self.Network(interpol)
        gradients = autograd.grad(outputs=D_interpol_data, inputs=interpol, grad_outputs=torch.ones(D_interpol_data.size()).to(device),create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(self.minibatch_size, -1)
        gradient_penalty = self.lambda_penalty * torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
        return gradient_penalty  

    def run_main_loop(self, p_distribution, q_distribution):
        self.Network.train()
        for epoch in range(self.num_epochs):
            self.optimizer.zero_grad()            
            p_iterator = iter(p_distribution)
            q_iterator = iter(q_distribution)
            px = next(p_iterator)
            qx = next(q_iterator)
            p_tensor = Variable( torch.from_numpy(np.float32(px.reshape(self.minibatch_size, self.input_size))) ).to(device)
            q_tensor = Variable( torch.from_numpy(np.float32(qx.reshape(self.minibatch_size, self.input_size))) ).to(device)
            D_x = self.Network(p_tensor)
            D_y = self.Network(q_tensor)
            gradient_penalty = self.gradient_penalty(p_tensor, q_tensor)
            loss_variable = -(torch.mean(D_x) - torch.mean(D_y)- gradient_penalty)
            loss_variable.backward()
            self.optimizer.step()
            if( epoch % self.print_interval) == (self.print_interval-1):
                print( "Epoch %6d. WD %5.7f" % ( epoch+1, loss_variable.item()))
                WD = -loss_variable.item()
        return WD