import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import MultiheadAttention
from typing import Tuple, List, Optional
from network.norm import *
from network.linear_group import *


class SpatialNetLayer(nn.Module):
    def __init__(self, dim_hidden: int, dim_ffn: int, num_heads: int, dropout: Tuple[float, float] = (0, 0)):
        super().__init__()
        # Narrow-band block (Time domain modeling)
        self.norm_mhsa = nn.LayerNorm(dim_hidden)
        self.mhsa = MultiheadAttention(embed_dim=dim_hidden, num_heads=num_heads, batch_first=True)
        self.dropout_mhsa = nn.Dropout(dropout[0])
        
        # Temporal Convolutional Feed-Forward Network (T-ConvFFN)
        self.tconvffn = nn.Sequential(
            nn.Conv1d(in_channels=dim_hidden, out_channels=dim_ffn, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(in_channels=dim_ffn, out_channels=dim_ffn, kernel_size=3, padding='same'),
            nn.SiLU(),
            nn.Conv1d(in_channels=dim_ffn, out_channels=dim_hidden, kernel_size=1),
        )
        self.dropout_tconvffn = nn.Dropout(dropout[1])

    def forward(self, x: torch.Tensor, att_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, F, T, H = x.shape
        x = x.reshape(B * F, T, H)  # [B*F, T, H]
        
        # Multi-Head Self-Attention
        x, _ = self.mhsa(self.norm_mhsa(x), self.norm_mhsa(x), self.norm_mhsa(x), attn_mask=att_mask)
        x = self.dropout_mhsa(x)

        # Temporal Convolutional Feed-Forward Network
        x = x.transpose(1, 2)  # [B*F, H, T]
        x = self.tconvffn(x)
        x = x.transpose(1, 2)  # [B*F, T, H]

        x = x.reshape(B, F, T, H)
        return x
    
# class SpatialNet(nn.Module):
#     def __init__(
#             self,
#             dim_input: int,  # the input dim for each time-frequency point
#             dim_output: int,  # the output dim for each time-frequency point
#             dim_squeeze: int,
#             num_layers: int,
#             num_freqs: int,
#             encoder_kernel_size: int = 5,
#             dim_hidden: int = 192,
#             dim_ffn: int = 384,
#             num_heads: int = 2,
#             dropout: Tuple[float, float, float] = (0, 0, 0),
#             kernel_size: Tuple[int, int] = (5, 3),
#             conv_groups: Tuple[int, int] = (8, 8),
#             norms: List[str] = ("LN", "LN", "GN", "LN", "LN", "LN"),
#             padding: str = 'zeros',
#             full_share: int = 0,  # share from layer 0
#     ):
#         super().__init__()

#         # encoder
#         self.encoder = nn.Conv1d(in_channels=dim_input, out_channels=dim_hidden, kernel_size=encoder_kernel_size, stride=1, padding="same")

#         # spatialnet layers
#         full = None
#         layers = []
#         for l in range(num_layers):
#             layer = SpatialNetLayer(
#                 dim_hidden=dim_hidden,
#                 dim_ffn=dim_ffn,
#                 dim_squeeze=dim_squeeze,
#                 num_freqs=num_freqs,
#                 num_heads=num_heads,
#                 dropout=dropout,
#                 kernel_size=kernel_size,
#                 conv_groups=conv_groups,
#                 norms=norms,
#                 padding=padding,
#                 full=full if l > full_share else None,
#             )
#             if hasattr(layer, 'full'):
#                 full = layer.full
#             layers.append(layer)
#         self.layers = nn.ModuleList(layers)

#         # decoder
#         self.decoder = nn.Linear(in_features=dim_hidden, out_features=dim_output)

#     def forward(self, x: Tensor, return_attn_score: bool = False) -> Tensor:
#         # x: [Batch, Freq, Time, Feature]
#         B, F, T, H0 = x.shape
#         x = self.encoder(x.reshape(B * F, T, H0).permute(0, 2, 1)).permute(0, 2, 1)
#         H = x.shape[2]

#         attns = [] if return_attn_score else None
#         x = x.reshape(B, F, T, H)
#         for m in self.layers:
#             setattr(m, "need_weights", return_attn_score)
#             x, attn = m(x)
#             if return_attn_score:
#                 attns.append(attn)

#         y = self.decoder(x)
#         if return_attn_score:
#             return y.contiguous(), attns
#         else:
#             return y.contiguous()
class SpatialNet(nn.Module):
    def __init__(self, dim_input: int, dim_hidden: int, dim_ffn: int, num_layers: int, num_heads: int, num_output_bands: int, num_freqs: int, encoder_kernel_size: int = 5, dropout: Tuple[float, float] = (0, 0)):
        super().__init__()
        # Encoder
        self.encoder = nn.Conv1d(in_channels=dim_input, out_channels=dim_hidden, kernel_size=encoder_kernel_size, stride=1, padding="same")

        # SpatialNet layers
        self.layers = nn.ModuleList([
            SpatialNetLayer(dim_hidden, dim_ffn, num_heads, dropout) for _ in range(num_layers)
        ])

        # Frequency reduction to 10 subbands
        # self.freq_reduction = nn.Conv2d(in_channels=dim_hidden, out_channels=num_output_bands, kernel_size=(1, 1))
        self.freq_reduction = nn.Conv2d(in_channels=num_freqs, out_channels=num_output_bands, kernel_size=(1, 1))

        # Decoder
        self.decoder = nn.Linear(dim_hidden, 1)

    def forward(self, x: torch.Tensor, return_attn_score: bool = False) -> torch.Tensor:
        # x: [Batch, Frequency, Time, Feature]
        B, F, T, H0 = x.shape
        # Encode input
        x = self.encoder(x.reshape(B * F, T, H0).permute(0, 2, 1)).permute(0, 2, 1)  # [B, dim_hidden, F, T]
        
        H = x.shape[2]

        attns = [] if return_attn_score else None
        x = x.reshape(B, F, T, H)
        # Apply SpatialNet layers
        for layer in self.layers:
            x = layer(x)

        # Reduce frequency dimension to 10 subbands
        x = self.freq_reduction(x).squeeze(2)  # [B, 10, T, H]

        # Average across time and feature dimensions
        # x = x.mean(dim=[2, 3])  # [B, 10]

        return x

class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv2DBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.pool = nn.MaxPool2d((2, 2))
        # self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x = F.relu(self.bn(self.conv1(x)))
        # x = F.relu(self.bn(self.conv2(x)))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout(self.pool(x))
        return x

class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv3DBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding)
        self.pool = nn.MaxPool3d((1, 2, 2))
        # self.bn = nn.BatchNorm3d(out_channels)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = F.elu(self.conv1(x))
        # x = F.elu(self.conv2(x))
        # x = self.pool(x)
        x = self.dropout(self.pool(x))
        return x

class SpectrogramEncoder(nn.Module):
    def __init__(self):
        super(SpectrogramEncoder, self).__init__()
        # self.conv1 = Conv3DBlock(16, 32, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        # spectral only --> 1
        self.conv1 = Conv3DBlock(15, 32, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        # self.conv2 = Conv3DBlock(32, 64, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv2 = Conv3DBlock(32, 64, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        
        self.conv3 = Conv3DBlock(64, 128, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv4 = Conv3DBlock(128, 256, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        # self.conv3 = Conv3DBlock(64, 128, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        # self.conv4 = Conv3DBlock(128, 256, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        # self.dropout = nn.Dropout(0.2)
        # self.pool = nn.MaxPool3d((1, 2, 2))

    def forward(self, x):
        # if torch.isnan(self.conv1.conv.weight).any():
        #     print("NaN found in conv1 weights")
        # if self.conv1.conv.bias is not None and torch.isnan(self.conv1.conv.bias).any():
        #     print("NaN found in conv1 bias")
            
        x = x.unsqueeze(2)  # Reshape to [batch_size, channels, depth=1, height, width] --> [128,16,1,54,84]
        x = self.conv1(x)  # [128,32,1,27,42]
        x = self.conv2(x)  # [128,64,1,13,21]
        x = self.conv3(x)  # [128,128,1,6,10]
        x = self.conv4(x)  # [128,256,1,3,5]
        x = x.reshape(x.size(0), -1)  # Flatten the tensor --> [128,3840]
        return x
        
    def _calculate_output_size(self, input_shape):
        # Create a dummy tensor with the input shape
        x = torch.randn(input_shape).unsqueeze(2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x.view(x.size(0), -1).size(1)

class SALSA_SpectrogramEncoder(nn.Module):
    def __init__(self):
        super(SALSA_SpectrogramEncoder, self).__init__()
        self.conv1 = Conv3DBlock(7, 32, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        # self.conv2 = Conv3DBlock(32, 64, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv2 = Conv3DBlock(32, 64, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        
        self.conv3 = Conv3DBlock(64, 128, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv4 = Conv3DBlock(128, 256, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        # self.conv3 = Conv3DBlock(64, 128, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        # self.conv4 = Conv3DBlock(128, 256, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        # self.dropout = nn.Dropout(0.2)
        # self.pool = nn.MaxPool3d((1, 2, 2))

    def forward(self, x):
        # if torch.isnan(self.conv1.conv.weight).any():
        #     print("NaN found in conv1 weights")
        # if self.conv1.conv.bias is not None and torch.isnan(self.conv1.conv.bias).any():
        #     print("NaN found in conv1 bias")
            
        x = x.unsqueeze(2)  # Reshape to [batch_size, channels, depth=1, height, width] --> [128,16,1,54,84]
        x = self.conv1(x)  # [128,32,1,27,42]
        x = self.conv2(x)  # [128,64,1,13,21]
        x = self.conv3(x)  # [128,128,1,6,10]
        x = self.conv4(x)  # [128,256,1,3,5]
        x = x.reshape(x.size(0), -1)  # Flatten the tensor --> [128,3840]
        return x
        
    def _calculate_output_size(self, input_shape):
        # Create a dummy tensor with the input shape
        x = torch.randn(input_shape).unsqueeze(2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x.view(x.size(0), -1).size(1)

class PV_Spec_Encoder(nn.Module):
    def __init__(self):
        super(PV_Spec_Encoder, self).__init__()
        self.conv1 = Conv2DBlock(1, 32, kernel_size=(3, 3), stride=1, padding=(1, 1))
        self.conv2 = Conv2DBlock(32, 64, kernel_size=(3, 3), stride=1, padding=(1, 1))
        self.conv3 = Conv2DBlock(64, 128, kernel_size=(3, 3), stride=1, padding=(1, 1))
        self.conv4 = Conv2DBlock(128, 256, kernel_size=(3, 3), stride=1, padding=(1, 1))

    def forward(self, x):
        x = self.conv1(x)  # Apply conv1 block
        x = self.conv2(x)  # Apply conv2 block
        x = self.conv3(x)  # Apply conv3 block
        x = self.conv4(x)  # Apply conv4 block
        x = x.view(x.size(0), -1)  # Flatten the tensor
        return x
    
class PV_Spatial_Encoder(nn.Module):
    def __init__(self):
        super(PV_Spatial_Encoder, self).__init__()
        self.conv1 = Conv3DBlock(15, 32, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv2 = Conv3DBlock(32, 64, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv3 = Conv3DBlock(64, 128, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv4 = Conv3DBlock(128, 256, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))

        self.pool = nn.MaxPool3d((1, 2, 2))

    def forward(self, x):
        x = x.unsqueeze(2)  # Reshape to [batch_size, channels, depth=1, height, width] --> [128,16,1,54,84]
        x = self.conv1(x)  # [128,32,1,27,42]
        x = self.conv2(x)  # [128,64,1,13,21]
        x = self.conv3(x)  # [128,128,1,6,10]
        x = self.conv4(x)  # [128,256,1,3,5]
        x = x.reshape(x.size(0), -1)  # Flatten the tensor --> [128,3840]
        return x

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob=0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

class Spatial_Net_MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob=0.5):
        super(Spatial_Net_MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()  # Use nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()  # Use nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        # Input shape: [batch, 10, 84, 96]
        B, N, F, H = x.shape

        # Flatten the last two dimensions
        x = x.reshape(B * N, -1)  # [batch * 10, 84 * 96]

        # Pass through the MLP
        x = self.relu1(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.relu2(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)  # [batch * 10, 1]

        # Reshape back to [batch, 10, 1]
        x = x.view(B, N, -1)
        return x
    
def apply_masks(Y, mask1, mask2):
    X1 = Y * mask1
    X2 = Y * mask2
    return X1**2, X2**2

# banding with mel-filterbank
def tri_filterbank_banding(fs, Nbin, fband_reduced):
    # Define the range and create a frequency bin array
    fbin = torch.linspace(0, Nbin, Nbin) * fs / (Nbin * 2)

    # Calculate the corresponding mel frequencies for the custom bands
    fband_reduced_tensor = torch.tensor(fband_reduced, dtype=torch.float32)
    
    banding = torch.zeros((len(fbin), len(fband_reduced)))

    for b in range(len(fbin)):
        for band in range(len(fband_reduced)):
            if band == 0:
                left_band_edge = 0
            else:
                left_band_edge = fband_reduced[band - 1]

            if band == len(fband_reduced) - 1:
                right_band_edge = fs / 2
            else:
                right_band_edge = fband_reduced[band + 1]

            if fbin[b] >= left_band_edge and fbin[b] <= fband_reduced[band]:
                banding[b, band] = (fbin[b] - left_band_edge) / (fband_reduced[band] - left_band_edge)
            elif fbin[b] > fband_reduced[band] and fbin[b] <= right_band_edge:
                banding[b, band] = (right_band_edge - fbin[b]) / (right_band_edge - fband_reduced[band])
            else:
                banding[b, band] = 0
    
    return banding
