import torch
import torch.nn as nn
import torchvision.models as models


class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResBlock, self).__init__()
        
        self.left = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(True),
            nn.Linear(out_dim, out_dim)
        )
        self.shortcut = nn.Sequential()
        if in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(in_dim, out_dim)
            )
    
    def forward(self, x):
        identity = self.shortcut(x) 
        out = self.left(x)
        out += identity
        return nn.ReLU(True)(out)


class Base(nn.Module):
    def __init__(self, dims=[]):
        super(Base, self).__init__()
        self.input_dim, self.output_dim = dims
        self.init_layers()

    def init_layers(self):
        self.layers = nn.Sequential(
            ResBlock(self.input_dim, 512),
            nn.LayerNorm(normalized_shape=512),
            ResBlock(512, 256),
            nn.LayerNorm(normalized_shape=256),
            nn.Linear(256, self.output_dim)
        )
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                nn.init.zeros_(m.bias.data)

    def forward(self, X):
        return self.layers(X)
    

class EnhancedBase(nn.Module):
    def __init__(self, dims=[]):
        super(EnhancedBase, self).__init__()
        self.input_dim, self.output_dim = dims
        self.init_layers()

    def init_layers(self):
        self.layers = nn.Sequential(
            nn.LayerNorm(normalized_shape=self.input_dim),
            ResBlock(self.input_dim, 1024),
            nn.LayerNorm(normalized_shape=1024),
            ResBlock(1024, 512),
            nn.LayerNorm(normalized_shape=512),
            ResBlock(512, 256),
            nn.LayerNorm(normalized_shape=256),
            nn.Linear(256, self.output_dim)
        )
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                nn.init.zeros_(m.bias.data)

    def forward(self, X):
        return self.layers(X)


class MobileNet(nn.Module):
    def __init__(self, dims= []):
        super(MobileNet, self).__init__()
        _, output_dim = dims
        self.output_dim= output_dim
        self.init_layers()

        self.model = models.mobilenet_v3_large(pretrained=True)
        self.model.classifier[-1] = nn.Linear(
            self.model.classifier[-1].in_features, output_dim
        )
        self.model.features[0][0] = torch.nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )

    def init_layers(self):
        self.layers = nn.Sequential(
            ResBlock(int(self.output_dim), 128),
            nn.LayerNorm(normalized_shape=128),
            nn.Linear(128, self.output_dim),
        )
        self.pos_cd = nn.Linear(2*self.output_dim, self.output_dim)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                nn.init.zeros_(m.bias.data)

    def forward(self, x):
        x_img, x_em = x
        x_img = self.model(x_img)
        x_em = self.layers(x_em)

        return self.pos_cd(torch.cat((x_img, x_em), dim= 1))
    
#[TODO] +++ any other architectures. +++
