import torch
import torch.nn as nn
from ldm.modules.diffusionmodules.util import zero_module
from ldm.modules.diffusionmodules.openaimodel import QKVAttention, QKVAttentionLegacy
import timm.models.layers as tml


class ResMLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers, zero_output=False):
        super(ResMLP, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        layers = []
        layers.append(nn.Linear(in_channels, hidden_channels))
        for i in range(n_layers):
            layers.append(tml.Mlp(hidden_channels, hidden_channels))
        if zero_output:
            layers.append(zero_module(nn.Linear(hidden_channels, out_channels)))
        else:
            layers.append(nn.Linear(hidden_channels, out_channels))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        z = self.model(x)
        return z

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers, zero_output=False):
        super(MLP, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        layers = []
        layers.append(nn.Linear(in_channels, hidden_channels))
        layers.append(nn.ReLU())
        for i in range(n_layers):
            layers.append(nn.BatchNorm1d(hidden_channels))
            layers.append(nn.Linear(hidden_channels, hidden_channels))
            layers.append(nn.ReLU())
        if zero_output:
            layers.append(zero_module(nn.Linear(hidden_channels, out_channels)))
        else:
            layers.append(nn.Linear(hidden_channels, out_channels))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        z = self.model(x)
        return z


class NormLayer(nn.Module):
    def __init__(self, in_channels, scale=2):
        super(NormLayer, self).__init__()
        self.norm = nn.BatchNorm1d(in_channels, affine=False)
        self.scale = scale
    
    def normalize(self, x):
        return self.norm(x) / self.scale
    
    def denormalize(self, x):
        return x * self.scale


class SpectralAttention(nn.Module):
    def __init__(self, in_channels, n_head, n_features):
        super(SpectralAttention, self).__init__()
        self.in_channels = in_channels
        self.n_head = n_head
        self.n_features = n_features
        assert n_features % n_head == 0, f"Number of features should be multiple of number of heads, but got {n_features} and {n_head}"
        self.norm = nn.BatchNorm1d(in_channels)
        self.in_conv = nn.Conv1d(1, n_features * 3, 3, 1, 1)
        self.attention = QKVAttentionLegacy(n_head)
        self.proj_h = zero_module(nn.Conv1d(n_features, 1, 1, 1, 0))
    
    def forward(self, x):
        b, c = x.shape
        x = self.norm(x)
        x = x.view(b, 1, c)
        qkv = self.in_conv(x)
        h = self.attention(qkv)
        h = self.proj_h(h)
        return (h + x).view(b, c)
        

class RGBOnlyModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, z_channels, n_layers, zero_encoder, zero_decoder, norm_encoder, scale):
        super(RGBOnlyModel, self).__init__()
        self.z_channels = z_channels
        self.in_channels = in_channels
        # self.skip = nn.Linear(3, in_channels)
        self.decoder = MLP(z_channels, hidden_channels, in_channels, n_layers, zero_output=zero_decoder)
    
    def decode(self, rgb):
        # cat z and rgb
        input = rgb
        input = input.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = input.shape
        assert C % 3 == 0, f"Input channels should be multiple of 3, but got {C}"
        input = input.view(B * H * W, C)
        # skip = self.skip(input)
        output = self.decoder(input) 
        output = output.view(B, H, W, self.in_channels)
        output = output.permute(0, 3, 1, 2).contiguous()
        return output

    def forward(self, rgb):
        return self.decode(rgb)
    
    
class SpectralModelNew(nn.Module):
    def __init__(self, in_channels, hidden_channels, z_channels, n_layers, zero_encoder, zero_decoder, norm_encoder, scale, attention=False):
        super(SpectralModelNew, self).__init__()
        assert z_channels % 3 == 0, f"Z channels should be multiple of 3, but got {z_channels}"
        self.z_channels = z_channels
        self.in_channels = in_channels
        self.norm_encoder = norm_encoder
        self.skip = nn.Linear(3, in_channels)
        self.encoder = MLP(in_channels, hidden_channels, z_channels, n_layers, zero_output=zero_encoder)
        self.decoder = MLP(z_channels + 3, hidden_channels, in_channels, n_layers, zero_output=zero_decoder)
        self.use_attention = attention
        if attention:
            self.att = SpectralAttention(in_channels, 4, 64)
        
    def encode(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = x.shape
        x = x.view(B * H * W, C)
        z = self.encoder(x)
        z = z.view(B, H, W, self.z_channels)
        z = z.permute(0, 3, 1, 2).contiguous()
        return z
    
    def decode(self, z, rgb):
        # cat z and rgb
        input = torch.cat([z, rgb], dim=1)
        input = input.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = input.shape
        assert C % 3 == 0, f"Input channels should be multiple of 3, but got {C}"
        input = input.view(B * H * W, C)
        # skip connection of the rgb
        skip = self.skip(input[:, -3:])
        output = self.decoder(input) + skip
        if self.use_attention:
            output = self.att(output)
        output = output.view(B, H, W, self.in_channels)
        output = output.permute(0, 3, 1, 2).contiguous()
        return output
        


class SpectralModelResidual(nn.Module):
    def __init__(self, in_channels, hidden_channels, z_channels, n_layers, zero_encoder, zero_decoder, norm_encoder, scale):
        super(SpectralModelResidual, self).__init__()
        assert z_channels % 3 == 0, f"Z channels should be multiple of 3, but got {z_channels}"
        self.z_channels = z_channels
        self.in_channels = in_channels
        self.norm_encoder = norm_encoder
        if norm_encoder:
            self.norm = NormLayer(z_channels, scale)
        self.skip = nn.Linear(3, in_channels)
        self.encoder = ResMLP(in_channels, hidden_channels, z_channels, n_layers, zero_output=zero_encoder)
        self.decoder = ResMLP(z_channels + 3, hidden_channels, in_channels, n_layers, zero_output=zero_decoder)
        
    def encode(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = x.shape
        x = x.view(B * H * W, C)
        z = self.encoder(x)
        if self.norm_encoder:
            z = self.norm.normalize(z)
        z = z.view(B, H, W, self.z_channels)
        z = z.permute(0, 3, 1, 2).contiguous()
        return z
    
    def decode(self, z, rgb):
        # cat z and rgb
        input = torch.cat([z, rgb], dim=1)
        input = input.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = input.shape
        assert C % 3 == 0, f"Input channels should be multiple of 3, but got {C}"
        input = input.view(B * H * W, C)
        if self.norm_encoder:
            # denormalize the z
            input[:, :-3] = self.norm.denormalize(input[:, :-3])
        # skip connection of the rgb
        skip = self.skip(input[:, -3:])
        output = self.decoder(input) + skip
        output = output.view(B, H, W, self.in_channels)
        output = output.permute(0, 3, 1, 2).contiguous()
        return output


class SpectralModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, z_channels, n_layers, zero_encoder, zero_decoder, norm_encoder, scale):
        super(SpectralModel, self).__init__()
        assert z_channels % 3 == 0, f"Z channels should be multiple of 3, but got {z_channels}"
        self.z_channels = z_channels
        self.in_channels = in_channels
        self.norm_encoder = norm_encoder
        if norm_encoder:
            self.norm = NormLayer(z_channels, scale)
        self.skip = nn.Linear(3, in_channels)
        self.encoder = MLP(in_channels, hidden_channels, z_channels, n_layers, zero_output=zero_encoder)
        self.decoder = MLP(z_channels + 3, hidden_channels, in_channels, n_layers, zero_output=zero_decoder)
        
    def encode(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = x.shape
        x = x.view(B * H * W, C)
        z = self.encoder(x)
        if self.norm_encoder:
            z = self.norm.normalize(z)
        z = z.view(B, H, W, self.z_channels)
        z = z.permute(0, 3, 1, 2).contiguous()
        return z
    
    def decode(self, z, rgb):
        # cat z and rgb
        input = torch.cat([z, rgb], dim=1)
        input = input.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = input.shape
        assert C % 3 == 0, f"Input channels should be multiple of 3, but got {C}"
        input = input.view(B * H * W, C)
        if self.norm_encoder:
            # denormalize the z
            input[:, :-3] = self.norm.denormalize(input[:, :-3])
        # skip connection of the rgb
        skip = self.skip(input[:, -3:])
        output = self.decoder(input) + skip
        output = output.view(B, H, W, self.in_channels)
        output = output.permute(0, 3, 1, 2).contiguous()
        return output
        

if __name__ == "__main__":
    model = SpectralModel(31, 128, 3, 3, False, False, False)
    x = torch.randn(4, 31, 256, 256)
    rgb = torch.randn(4, 3, 256, 256)
    z = model.encode(x)
    output = model.decode(z, rgb)
    print(output.shape)
    print('Parameters number is ', sum(param.numel() for param in model.parameters()))
    print("Test passed")
