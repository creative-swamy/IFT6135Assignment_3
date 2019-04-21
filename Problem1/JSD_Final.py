import torch
import simpleclassdef as classdef
from torch.autograd import Variable
import math
import torch.optim as optim
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class JSD():
    def __init__(self, hidden_size, mini_batch, learning_rate, num_epochs, print_interval):
        super(JSD, self).__init__()
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
        
    def loss_function(self, x, y):
        loss_fun = -(math.log(2) + 0.5*(torch.mean(torch.log(x)) + torch.mean(torch.log(1.0 - y))))
        return loss_fun
    
    def run_main_loop(self, p_distribution, q_distribution):
        self.Network.train()
        for epoch in range(self.num_epochs):            
            self.optimizer.zero_grad()            
            p_iterator = iter(p_distribution)
            q_iterator = iter(q_distribution)
            px = next(p_iterator)
            qx = next(q_iterator)
            p_tensor = Variable( torch.from_numpy(np.float32(px.reshape(self.minibatch_size, self.input_size)))).to(device)
            q_tensor = Variable( torch.from_numpy(np.float32(qx.reshape(self.minibatch_size, self.input_size)))).to(device)
            D_x = self.Network(p_tensor)
            D_y = self.Network(q_tensor)
            loss_variable = self.loss_function(D_x, D_y)
            loss_variable.backward()
            self.optimizer.step()
            if( epoch % self.print_interval) == (self.print_interval-1):
                print( "Epoch %6d. WD %5.7f" % ( epoch+1, loss_variable.item()))
                JSD = -loss_variable.item()
        return JSD