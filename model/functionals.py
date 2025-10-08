import torch
import torch.nn.functional as F
from model import nets

dev = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")


class Encoder(object):
    def __init__(self, dims= [], net= 'Base'):
        self.input_dim, output_dim = dims
        self.output_dim = int(2*output_dim)

        dims = (self.input_dim, self.output_dim)
        self.net = getattr(nets, net)(dims)
        self.net.to(dev)
    
    def encode(self, X):
        X = self.net(X)
        
        mu = X[:,:self.output_dim//2]
        std = F.softplus(X[:,self.output_dim//2:]-5,beta=1)

        return torch.distributions.Normal(mu, std)


class EncoderDet(object):
    def __init__(self, dims=[], net='Base'):
        self.net = getattr(nets, net)(dims)
        self.net.to(dev)

    def encode(self, X):
        X = self.net(X)

        return X


class Decoder(object):
    def __init__(self, dims= [], net= 'Base'):
        self.net = getattr(nets, net)(dims)
        self.net.to(dev)
    
    def decode(self, X):
        return self.net(X)
    

class Selector(object):
    def __init__(self, dims= [], net= 'Base'):
        em_dim, num_range, self.num_of_sel= dims
        self.net = getattr(nets, net)(
            [em_dim, int(num_range+self.num_of_sel)]
        )
        self.net.to(dev)
    
    def select(self, X):
        X = self.net(X)
        return F.softmax(X[:self.num_of_sel],dim= -1), \
            F.softmax(X[self.num_of_sel:],dim= -1)
