import torch
import torch.nn as nn

from torchvision.models import vgg19
from typing import Union, List, cast

class ResidualBlock(nn.Module):
    def __init__(self, 
                in_channels, 
                embedding_channels, 
                out_channels,
                kernel_size = (3,3),
                stride = 1,
                padding = 1,
                ) -> None:
        
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels = in_channels,
            out_channels = embedding_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
        )
        self.bn1 = nn.BatchNorm2d(embedding_channels)
        self.relu1 = nn.ReLU()
        self.conv2 =  nn.Conv2d(
            in_channels =embedding_channels,
            out_channels =out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        x_res = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = x + x_res
        return x

class SRGAN_g(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        """_summary_
         'p' stands for previous Image
         'n' stands for next Image
        """
        self.p_conv1 = nn.Conv2d(1, 64, (3,3), 1, 1) # in, out, kernel_size, stride, padding
        self.p_relu1 = nn.ReLU()
        
        self.n_conv1 = nn.Conv2d(1, 64, (3,3), 1, 1)
        self.n_relu1 = nn.ReLU()
        self.p_encoder = nn.ModuleList([
            ResidualBlock(
                in_channels=64,
                embedding_channels=64,
                out_channels=64
            ) for i in range(8)
        ])
        self.n_encoder = nn.ModuleList([
            ResidualBlock(
                in_channels=64,
                embedding_channels=64,
                out_channels=64
            ) for _ in range(8)
        ])
        
        self.p_conv2 = nn.Conv2d(64, 64, (3,3), 1, 1)
        self.p_bn2 = nn.BatchNorm2d(64)
        self.p_relu2 = nn.ReLU()
        
        self.n_conv2 = nn.Conv2d(64, 64, (3,3), 1, 1)
        self.n_bn2 = nn.BatchNorm2d(64)
        self.n_relu2 = nn.ReLU()
        
        self.combined_decoder = nn.ModuleList([
            ResidualBlock(
                in_channels=64,
                embedding_channels=64,
                out_channels=64
            ) for _ in range(4)
        ])
        self.combined_conv = nn.Conv2d(64, 64, (3,3), 1, 1)
        self.combined_bn = nn.BatchNorm2d(64)
        self.combined_relu = nn.ReLU()
        
        self.predict_conv = nn.Conv2d(64, 1, (1,1), 1, 0) # in, out, kernel_size, stride, padding
        self.predict_act = nn.Tanh()
        
    def forward(self, prev, next):
        # Embedding at first
        prev = self.p_conv1(prev)
        prev = self.p_relu1(prev)
        temp_prev = prev
        
        next = self.n_conv1(next)
        next = self.n_relu1(next)
        temp_next = next

        # Encoding them
        for blk in self.p_encoder:
            prev = blk(prev)
        for blk in self.n_encoder:
            next = blk(next)

        # post embedding and residual connectionis
        prev = self.p_conv2(prev)
        prev = self.p_bn2(prev)
        prev = self.p_relu2(prev)
        prev = prev + temp_prev
        
        next = self.n_conv2(next)
        next = self.n_bn2(next)
        next = self.n_relu2(next)
        next = next + temp_next
        
        # fusion with adding
        combined = prev + next
        temp_combined = combined
        
        for blk in self.combined_decoder:
            combined = blk(combined)
        
        combined = self.combined_conv(combined)
        combined = self.combined_bn(combined)
        combined = self.combined_relu(combined)
        
        # residual connections
        combined = combined + temp_combined
        
        pred = self.predict_conv(combined)
        pred = self.predict_act(pred)
        return pred

class ConvWithNorm(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding
                 ) -> None:
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size =kernel_size,
            stride = stride,
            padding = padding
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.activate = nn.LeakyReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activate(x)
        return x

class SRGAN_d(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv_list1 = nn.Sequential(
            # ConvWithNorm(1, 64, (5, 3), 1, 1),   #1
            # ConvWithNorm(64, 128, (3, 3), 1, 1),  #1
             # in_channels, out_channels, stride,kernel_size, padding
            ConvWithNorm(1, 64, (4, 4), 2, 1),      # n, 64, 112, 112  
            ConvWithNorm(64, 128, (4, 4), 2, 1),
            ConvWithNorm(128, 256, (4, 4), 2, 1),
            ConvWithNorm(256, 512, (4, 4), 2, 1),   # n, 512, 14, 14
            ConvWithNorm(512, 1024, (4, 4), 2, 1),  # n, 1024, 7, 7
            ConvWithNorm(1024, 1024, (1, 1), 1, 0),  # n, 1024, 7, 7
            ConvWithNorm(1024, 512, (1, 1), 1, 0),  # n, 1024, 7, 7
        )
        self.conv_list2 = nn.Sequential(
            ConvWithNorm(512, 256, (1, 1), 1, 0),  # n, 256, 7, 7
            ConvWithNorm(256, 128, (3, 3), 1, 1),  # n, 128, 7, 7
            ConvWithNorm(128, 512, (3, 3), 1, 1),  # n, 512, 7, 7
     
            # ConvWithNorm(128,  (1, 1), 1, 0)  # n, 1024, 7, 7
        )
        self.lrelu1 = nn.LeakyReLU()
        self.linear = nn.Linear(512 * 7 * 7, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv_list1(x)
        x_res = x
        x = self.conv_list2(x)
        x = x + x_res
        x = self.lrelu1(x)                      # n , 512, 7, 7
        x = torch.flatten(x, start_dim=1)
        logits = self.linear(x)
        # output = self.sigmoid(logits)
        return  logits
    
### Perceptual loss

class vgg19_perceptual_loss(nn.Module):
    
    def make_layers(self, cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
        layers: List[nn.Module] = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                v = cast(int, v)
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
    
    def __init__(self) -> None:
        super().__init__()
        vgg_19_structure = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M']
        self.features = self.make_layers(vgg_19_structure)
        vgg = vgg19(pretrained= True)
        self.load_state_dict(vgg.state_dict(), strict=False) # Only load first 12 layers
        
                # Freeze model weights
        for param in self.features.parameters():
            param.requires_grad_(False)
            
    def forward(self, x):
        return self.features(x)


"""
Code below for testing 
"""

if __name__ == '__main__':
      
    c = SRGAN_g()
    p = torch.randn((3,1,224,224)) 
    print(torch.max(p) , torch.min(p))
    n = torch.randn((3,3,224,224)) 
    
    # print(p, n)
    # d = c(p, n)
    # print(d)
    
    d = SRGAN_d()
    
    p = vgg19_perceptual_loss()
    c = p( (n+1)/2 )    
    print(c.shape)
    print(c)
    # print(a.shape)