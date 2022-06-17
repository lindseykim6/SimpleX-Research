import torch
from torch import nn, autograd

class Projection(nn.Module):
    def __init__(self, num_particles, dimension, func, num_iter = 3, stiffness = 1, boundary_nodes = None):
        super(Projection, self).__init__()
        self.num_iter = num_iter
        self.num_particles = num_particles
        self.dimension = dimension
        self.func = func
    
    def cal_delta_x(self, input_x):
        input_ = input_x.requires_grad_(True)
        output_ = self.func(input_) #?
        grad = autograd.grad(
            outputs=output_,
            inputs=input_,
            grad_outputs=torch.ones_like(output_),
            create_graph=True,
            retain_graph=True
        )[0]
        cons = output_
        # cons : B * 1; grad: B * num_particles * dimension
        eps = 1e-7  # avoid dividing by zero
        s = (cons.squeeze() / ((grad*grad).sum([1,2])+eps) ).expand(input_x.size()[1],input_x.size()[2],-1).permute(2,0,1)
        # delta_x = - ( c / sum(grad_x^2) ) * grad_x 
        return - s * grad 

    # x: x_{data}
    def forward(self, x):
        # x : B * num_particles * dimension 
        upd_x = x
        for i in range(self.num_iter):
            delta_x = self.cal_delta_x(upd_x)
            upd_x = upd_x + delta_x 
        return upd_x

