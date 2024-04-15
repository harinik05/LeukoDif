import torch
from torch import nn
import math

# Positional Embeddings Class
class IncorporatePositionalEmbeddings(nn.Module):
    # Constructor
    def __init__(self, inputDimensionality):
        super().__init__()
        self.inputDimensionality = inputDimensionality
    

    # Forward Pass
    def forward_pass(self, currentTimeStep):
        # Calculate the half dimension
        halfDimensionality = self.inputDimensionality//2
        embeddings = math.log(10000) / (halfDimensionality - 1)
        embeddings = torch.exp(torch.arange(halfDimensionality, device=currentTimeStep.device) * -embeddings)
        embeddings = currentTimeStep[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class BackwardDiffusion(nn.Module):
    def __init__( self,image_channels=3, out_dim=3, time_emb_dim=32):
        super().__init__()

        # Increase in the number of channels for down, flatten, and decrease (less depth) for upsample
        down_channels = (32, 64, 128, 256, 512)
        up_channels = (512, 256, 128, 64, 32)
      

        # Call the positional embeddings stuff
        self.time_mlp = nn.Sequential(
            IncorporatePositionalEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.LeakyReLU(0.2)
        )

        # initially, use conv layer
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([self._create_block(down_channels[i], down_channels[i+1], 
                                                       time_emb_dim) 
                                    for i in range(len(down_channels)-1)])
        
        # Upsample
        self.ups = nn.ModuleList([self._create_block(up_channels[i], up_channels[i+1], 
                                                     time_emb_dim, up=True) 
                                  for i in range(len(up_channels)-1)])

        # Output convolution
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    # magic block for creation of layers (sequential per sample)
    def _create_block(self, in_ch, out_ch, time_emb_dim, up=False):
        layers = []
        layers.append(nn.Linear(time_emb_dim, out_ch))
        if up:
            layers.append(nn.Conv2d(2 * in_ch, out_ch, 3, padding=1))
            layers.append(nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1))
        else:
            layers.append(nn.Conv2d(in_ch, out_ch, 3, padding=1))
            layers.append(nn.Conv2d(out_ch, out_ch, 4, 2, 1))
        layers.append(nn.Conv2d(out_ch, out_ch, 3, padding=1))
        layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def forward(self, x, timestep):
        # Embed time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # residual connection
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        return self.output(x)
    