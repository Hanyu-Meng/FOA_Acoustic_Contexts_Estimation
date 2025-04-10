# all of models for acoustic context estimation
# writen by Hanyu Meng
import torch
import torch.nn as nn
import torch.nn.functional as F
from feature import *
from network.utility import *
from ufb_banding.banding import BandingParams, BandingShape, LowerBandMode
from ufb_banding.ufb import TransformParams

class BLSTM_Mask(nn.Module):
    def __init__(self,args):
        super(BLSTM_Mask, self).__init__()
        # 80ms -> 641, 96ms --> 769
        # 32ms --> 257
        # banding --> 10
        self.blstm1 = nn.LSTM(input_size=257, hidden_size=600, num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.blstm2 = nn.LSTM(input_size=1200, hidden_size=600, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(1200, 2 * 257)  # For two masks
        self.fband_reduced = [1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000]
        self.fband_reduced = torch.from_numpy(np.asarray(self.fband_reduced))
        self.Nbin = 257 # FFT bins
        self.banding = tri_filterbank_banding(args.dataset.sample_rate, self.Nbin, self.fband_reduced)

    def forward(self, x):
        # feature extraction
        device = x.device
        Y =  cal_w_spec_feature(x, device)
        Y = Y.permute(0,2,1)
        # batch first --> [N,L,Hin] 600x2x2x321
        x, _ = self.blstm1(Y)
        x = self.dropout(x) # [B,201,1200]
        x, _ = self.blstm2(x)
        x = self.fc(x) #[B,201,642]
        x = torch.sigmoid(x)
        # change if change the STFT window size
        mask1, mask2 = torch.split(x, 257, dim=-1)
        X1, X2 = apply_masks(Y, mask1, mask2)
        banding_sqrt = torch.sqrt(self.banding.to(device))
        banding_sqrt = banding_sqrt.unsqueeze(0)
        X1 = torch.matmul(X1, banding_sqrt)
        X2 = torch.matmul(X2, banding_sqrt)
        X1 = X1.sum(axis=1)
        X2 = X2.sum(axis=1)
        output = 10*torch.log10(X1/X2) # [b,321,201] --> need to be [b,201,321]
        return output

class CRNN(nn.Module):
    def __init__(self, args):
        super(CRNN, self).__init__()
        self.num_classes = args.CRNN.num_classes
        self.conv1 = nn.Conv2d(args.CRNN.input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.dropout2 = nn.Dropout(0.2)

        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout(0.2)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.dropout4 = nn.Dropout(0.2)

        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2))

        self.gru1 = nn.GRU(128, 32, batch_first=True,bidirectional=True)  # Adjust according to the flattened feature size
        self.gru2 = nn.GRU(64, 32, batch_first=True)

        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, self.num_classes*10)

    def forward(self, x):
        # feature extraction
        x = compute_mfcc_batch(x, device=x.device)
        x = x.unsqueeze(1)
        # Apply convolutional layers
        x = self.max_pool(F.elu(self.bn1(self.conv1(x)))) #[64,64,10,199]
        x = self.dropout1(x)

        x = self.max_pool(F.elu(self.bn2(self.conv2(x)))) #[64,128,5,99]
        x = self.dropout2(x)

        x = self.max_pool(F.elu(self.bn3(self.conv3(x)))) #[64,128,2,49]
        x = self.dropout3(x)

        x = self.max_pool(F.elu(self.bn4(self.conv4(x))))
        x = self.dropout4(x)  # [batch, 128, h, w]
        # []
        # Reshape for GRU input: (batch_size, time_steps, features)
        x = x.permute(0, 3, 2, 1).contiguous()  # Change to (batch_size, time_steps, features, channels)c# [128,5,1,128]
        batch_size, time_steps, features, channels = x.shape
        x = x.view(batch_size, time_steps, -1)  # Flatten last two dimensions

        # Apply GRU layers
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)  # [batch, time_steps, 64]

        x = F.elu(self.fc1(x[:, -1, :]))  # [64,24,32]
        x = F.elu(self.fc2(x))  # [batch, 64]
        x = self.fc3(x)  # Output size is (batch_size, num_classes)
        # Reshape the output to (batch_size, 4, 33)
        x = x.view(batch_size, self.num_classes, 10) # [batch,4,10]

        return x.squeeze()

class ParamNet(torch.nn.Module):
    def __init__(self,args):
        super(ParamNet,self).__init__()
        # Pipeline 1
        # window size same as the STFT window size
        self.conv1d_down_1_depth = nn.Conv1d(769, 769, kernel_size=11, stride=1, groups=769, padding=5)
        self.conv1d_down_1_point = nn.Conv1d(769, 384, kernel_size=1, stride=1, padding=0)
        self.ln_1 = nn.LayerNorm([384])
        self.relu=nn.ReLU()
        self.single_channel = args.single_channel
        #Pipeline 2 --> 16*48
        # original: 2307,1152
        # if self.single_channel == False:
        self.pip_conv_1d = nn.Conv1d(16*54, 16*54, kernel_size=11, stride=1, groups=16*54, padding=5) # ks=2,st=2
        self.pip_conv_1p = nn.Conv1d(16*54, 8*54, kernel_size=1, stride=1, padding=0)
        self.ln_pip_1 = nn.LayerNorm([8*54])
        self.band_params = BandingParams.Log(args.PV.dt_ms, args.dataset.sample_rate, BandingShape.SOFT, TransformParams.RaisedSine(), lower_band_mode=LowerBandMode.VSV_HPF)
        self.power_vector = SimplePowerVector(
            params=self.band_params, 
            fs=args.dataset.sample_rate, 
            nch=args.PV.channels, 
            cov_to_pv_trainable=args.PV.cov_to_pv_trainable,
            band_matrix_trainable=args.PV.band_matrix_trainable,
            smoothing_trainable=args.PV.smooth_pv,
            normalise=args.PV.normalise,
            hz_s_per_band=5 if args.PV.smooth_pv else None, 
            stack_bands=args.PV.pv_stack_bands
        )

        #Pipeline 1
        self.conv1d_down_2_depth = nn.Conv1d(384, 384, kernel_size=11, stride=1, groups=384, dilation=2, padding=10)
        self.conv1d_down_2_point = nn.Conv1d(384, 192, kernel_size=1, stride=1)
        self.ln_2 = nn.LayerNorm([192])

        #Pipeline 1
        self.conv1d_down_3_depth = nn.Conv1d(192, 192, kernel_size=11, stride=1, groups=192, dilation=4, padding=20)
        self.conv1d_down_3_point = nn.Conv1d(192, 96, kernel_size=1, stride=1)
        self.ln_3 = nn.LayerNorm([96])
    
        self.drp_1=nn.Dropout(p=0.2)
        self.drp = nn.Dropout(p=0.5)

        # change to 1248 if using both spectral and spatial features
        # if self.single_channel == False:
        self.fc_1 = nn.Linear(528, 96)
        self.fc_2 = nn.Linear(96, 48)
        self.fc_3 = nn.Linear(48, 10)
        # self.softplus = nn.Softplus()
        self.avgpool_1 = nn.AvgPool1d(84, stride=1)

    def forward(self, audio):
        x = cal_features(audio)
        # x~ ch1, x2~ ch2 
        x = self.relu(self.conv1d_down_1_depth(x))
        x = self.relu(self.conv1d_down_1_point(x))
        x = self.ln_1(x.reshape(-1,84,384))
        x = self.drp_1(x.reshape(-1,384,84)) # [1,384,84]

        x = self.relu(self.conv1d_down_2_depth(x))
        x = self.relu(self.conv1d_down_2_point(x))
        x = self.ln_2(x.reshape(-1,84,192))
        x = self.drp_1(x.reshape(-1,192,84))

        x = self.relu(self.conv1d_down_3_depth(x))
        x = self.relu(self.conv1d_down_3_point(x))
        x = self.ln_3(x.reshape(-1,84,96))
        x= x.reshape(-1,96,84) # [1, 96, 84]
        pv = self.power_vector(audio).rename(None)
        pv = pv.permute(0,2,3,1)
        pv = pv.reshape(pv.shape[0],-1,pv.shape[3])       
        x2=self.relu(self.pip_conv_1d(pv))
        x2=self.relu(self.pip_conv_1p(x2))
        x2=self.ln_pip_1(x2.reshape(-1,84,432))
        x2=self.drp_1(x2.reshape(-1,432,84)) # [1,1152,84]
        x=torch.cat((x,x2),axis=1) # [1,528,84]
        x=self.avgpool_1(x)
        x=x.reshape(-1, 528)
        x=self.fc_1(x)
        x=self.drp(x)
        x=self.fc_3(self.fc_2(x))
        # joinly estimate room volume, area, acoustic parameters (T60,DRR)
        # 1-10 --> DRR, 11-20 --> RT60, 21 --> Volume, 22 --> Area
        return x

class ParamNet_Single(torch.nn.Module):
    def __init__(self,args):
        super(ParamNet_Single,self).__init__()
        # Pipeline 1
        # window size same as the STFT window size
        self.conv1d_down_1_depth = nn.Conv1d(769, 769, kernel_size=11, stride=1, groups=769, padding=5)
        self.conv1d_down_1_point = nn.Conv1d(769, 384, kernel_size=1, stride=1, padding=0)
        self.ln_1 = nn.LayerNorm([384])
        self.relu=nn.ReLU()
        self.single_channel = args.single_channel
        #Pipeline 2 --> 16*48
        # original: 2307,1152
        self.pip_conv_1d = nn.Conv1d(16*54, 16*54, kernel_size=11, stride=1, groups=16*54, padding=5) # ks=2,st=2
        self.pip_conv_1p = nn.Conv1d(16*54, 8*54, kernel_size=1, stride=1, padding=0)
        #Pipeline 1
        self.conv1d_down_2_depth = nn.Conv1d(384, 384, kernel_size=11, stride=1, groups=384, dilation=2, padding=10)
        self.conv1d_down_2_point = nn.Conv1d(384, 192, kernel_size=1, stride=1)
        self.ln_2 = nn.LayerNorm([192])

        #Pipeline 1
        self.conv1d_down_3_depth = nn.Conv1d(192, 192, kernel_size=11, stride=1, groups=192, dilation=4, padding=20)
        self.conv1d_down_3_point = nn.Conv1d(192, 96, kernel_size=1, stride=1)
        self.ln_3 = nn.LayerNorm([96])
        self.drp_1=nn.Dropout(p=0.2)
        self.drp = nn.Dropout(p=0.5)
        
        self.fc_1_single = nn.Linear(96, 96)
        self.fc_2 = nn.Linear(96, 48)
        self.fc_3 = nn.Linear(48, 10)
        self.avgpool_1 = nn.AvgPool1d(84, stride=1)

    def forward(self, audio):
        x = cal_features(audio)
        # x~ ch1, x2~ ch2 
        x = self.relu(self.conv1d_down_1_depth(x))
        x = self.relu(self.conv1d_down_1_point(x))
        x = self.ln_1(x.reshape(-1,84,384))
        x = self.drp_1(x.reshape(-1,384,84)) # [1,384,84]

        x = self.relu(self.conv1d_down_2_depth(x))
        x = self.relu(self.conv1d_down_2_point(x))
        x = self.ln_2(x.reshape(-1,84,192))
        x = self.drp_1(x.reshape(-1,192,84))

        x = self.relu(self.conv1d_down_3_depth(x))
        x = self.relu(self.conv1d_down_3_point(x))
        x = self.ln_3(x.reshape(-1,84,96))
        x= x.reshape(-1,96,84) # [1, 96, 84]
        x=self.avgpool_1(x)
        x=x.reshape(-1, 96)
        x=self.fc_1_single(x)
        x=self.drp(x)
        x=self.fc_3(self.fc_2(x))
        # joinly estimate room volume, area, acoustic parameters (T60,DRR)
        # 1-10 --> DRR, 11-20 --> RT60, 21 --> Volume, 22 --> Area
        return x

class Conv3D_Net(nn.Module):
    def __init__(self,args):
        super(Conv3D_Net, self).__init__()
        self.band_params = BandingParams.Log(args.PV.dt_ms, args.dataset.sample_rate, BandingShape.SOFT, TransformParams.RaisedSine(), lower_band_mode=LowerBandMode.VSV_HPF)
        self.power_vector = SimplePowerVector(
                params=self.band_params, 
                fs=args.dataset.sample_rate, 
                nch=args.PV.channels, 
                cov_to_pv_trainable=args.PV.cov_to_pv_trainable,
                band_matrix_trainable=args.PV.band_matrix_trainable,
                smoothing_trainable=args.PV.smooth_pv,
                normalise=args.PV.normalise,
                hz_s_per_band=5 if args.PV.smooth_pv else None, 
                stack_bands=args.PV.pv_stack_bands
            )
        self.encoder = SpectrogramEncoder()
        # Calculate the output dimension after the convolutions and pooling
        input_shape = (1, 13, 54, 84)
        output_size = self.encoder._calculate_output_size(input_shape)
        # input_dim=256 * (54 // 16) * (84 // 16)
        self.mlp = MLP(input_dim=256 * (54 // 16) * (84 // 16), output_dim=10)
        # self.mlp = MLP(input_dim=1920, output_dim=10)

    def forward(self, audio):
        pv = self.power_vector(audio)
        pv = pv.rename(None)
        pv = pv.permute(0,2,3,1) #[128,16,54,84]
        # pv = pv[:,0,:,:].unsqueeze(1)
        pv = torch.cat((pv[:,0,:,:].unsqueeze(1),pv[:,4:,:,:]), dim=1)
        # pv = pv[:,1:,:,:]
        # pv = pv[:,1:4,:,:]
        # Convert pv to NumPy and save as .npy
        # pv_1 = pv[0, :, :, :].detach().cpu().numpy()  # Ensure it's detached and on the CPU
        # np.save('/media/sbsprl/data/Hanyu/Acoustic_context_estimation/network/pv_1.npy', pv_1)  # Save pv_1 as a .npy file
        x = self.encoder(pv)
        x = self.mlp(x)
        return x  # [128,10]

class SALSA_Conv3D_Net(nn.Module):
    def __init__(self,args):
        super(SALSA_Conv3D_Net, self).__init__()
        self.encoder = SALSA_SpectrogramEncoder()
        input_shape = (1, 7, 200, 85)
        output_size = self.encoder._calculate_output_size(input_shape)
        self.mlp = MLP(input_dim=256 * 12 * 5, output_dim=10)

    def forward(self, feature):
        salsa = feature.rename(None)
        salsa = salsa.permute(0,3,1,2) #[128,16,54,84]
        x = self.encoder(salsa)
        x = self.mlp(x)
        return x  # [128,10]

class Hybrid_Conv_Net(nn.Module):
    def __init__(self,args):
        super(Hybrid_Conv_Net, self).__init__()
        self.band_params = BandingParams.Log(args.PV.dt_ms, args.dataset.sample_rate, BandingShape.SOFT, TransformParams.RaisedSine(), lower_band_mode=LowerBandMode.VSV_HPF)
        self.power_vector = SimplePowerVector(
                params=self.band_params, 
                fs=args.dataset.sample_rate, 
                nch=args.PV.channels, 
                cov_to_pv_trainable=args.PV.cov_to_pv_trainable,
                band_matrix_trainable=args.PV.band_matrix_trainable,
                smoothing_trainable=args.PV.smooth_pv,
                normalise=args.PV.normalise,
                hz_s_per_band=5 if args.PV.smooth_pv else None, 
                stack_bands=args.PV.pv_stack_bands
            )
        self.spec_encoder = PV_Spec_Encoder()
        self.spatial_encoder = PV_Spatial_Encoder()
        self.mlp = MLP(input_dim=2*3840, output_dim=10)

    def forward(self, audio):
        pv = self.power_vector(audio)
        pv = pv.rename(None)
        pv = pv.permute(0,2,3,1) 

        spec_feature = self.spec_encoder(pv[:, 0, :, :].unsqueeze(1)) # [128,3840]
        spatial_feature = self.spatial_encoder(pv[:, 1:, :, :]) # [128,3840]
        x_combined = torch.cat((spec_feature, spatial_feature), dim=1)

        # Pass through the MLP
        out = self.mlp(x_combined) 
        return out

class CRNN_PV(nn.Module):
    def __init__(self, args):
        super(CRNN_PV, self).__init__()

        self.band_params = BandingParams.Log(args.PV.dt_ms, args.dataset.sample_rate, BandingShape.SOFT, TransformParams.RaisedSine(), lower_band_mode=LowerBandMode.VSV_HPF)
        self.power_vector = SimplePowerVector(
            params=self.band_params, 
            fs=args.dataset.sample_rate, 
            nch=args.PV.channels, 
            cov_to_pv_trainable=args.PV.cov_to_pv_trainable,
            band_matrix_trainable=args.PV.band_matrix_trainable,
            smoothing_trainable=args.PV.smooth_pv,
            normalise=args.PV.normalise,
            hz_s_per_band=5 if args.PV.smooth_pv else None, 
            stack_bands=args.PV.pv_stack_bands
        )
        self.num_classes = args.CRNN.num_classes
        self.conv1 = nn.Conv2d(args.CRNN.input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.dropout2 = nn.Dropout(0.2)

        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout(0.2)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.dropout4 = nn.Dropout(0.2)

        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2))

        self.gru1 = nn.GRU(128*54, 32, batch_first=True,bidirectional=True)  # Adjust according to the flattened feature size
        self.gru2 = nn.GRU(64, 32, batch_first=True)

        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, self.num_classes*10)

    def forward(self, audio):
        # feature extraction
        # x = compute_mfcc_batch(audio, device=audio.device)
        pv = self.power_vector(audio).rename(None) #[128,84,16,54] --> [b, time, ch, freq]
        pv = pv.permute(0,2,3,1)
        pv = pv.reshape(pv.shape[0],-1,pv.shape[3]) # [b, time*ch, freq]
        x = pv.unsqueeze(1) #[128,1,864,84]
        # x = torch.cat((x,pv),dim=1).unsqueeze(1)
        # Apply convolutional layers
        x = self.max_pool(F.elu(self.bn1(self.conv1(x)))) #[128,64,432,42]
        x = self.dropout1(x)

        x = self.max_pool(F.elu(self.bn2(self.conv2(x)))) #[128,128,216,21]
        x = self.dropout2(x)

        x = self.max_pool(F.elu(self.bn3(self.conv3(x)))) #[128,128,108,10]
        x = self.dropout3(x)

        x = self.max_pool(F.elu(self.bn4(self.conv4(x))))
        x = self.dropout4(x)  # [batch, 128, 54, 5]

        # Reshape for GRU input: (batch_size, time_steps, features)
        x = x.permute(0, 3, 2, 1).contiguous()  # Change to (batch_size, time_steps, features, channels)
        batch_size, time_steps, features, channels = x.shape # [128,5,54,128]
        x = x.view(batch_size, time_steps, -1)  # Flatten last two dimensions --> [128,5,54*128]

        # Apply GRU layers
        x, _ = self.gru1(x) # [128, 5, 64]
        x, _ = self.gru2(x)  # [batch, 5, 32]

        x = F.elu(self.fc1(x[:, -1, :]))  # [64,24,32]
        x = F.elu(self.fc2(x))  # [batch, 64]
        x = self.fc3(x)  # Output size is (batch_size, num_classes)
        # Reshape the output to (batch_size, 4, 33)
        x = x.view(batch_size, self.num_classes, 10) # [batch,4,10]

        return x.squeeze()
    
class SpatialNet_PV(nn.Module):
    def __init__(self, args):
        super(SpatialNet_PV, self).__init__()

        self.band_params = BandingParams.Log(args.PV.dt_ms, args.dataset.sample_rate, BandingShape.SOFT, TransformParams.RaisedSine(), lower_band_mode=LowerBandMode.VSV_HPF)
        self.power_vector = SimplePowerVector(
            params=self.band_params, 
            fs=args.dataset.sample_rate, 
            nch=args.PV.channels, 
            cov_to_pv_trainable=args.PV.cov_to_pv_trainable,
            band_matrix_trainable=args.PV.band_matrix_trainable,
            smoothing_trainable=args.PV.smooth_pv,
            normalise=args.PV.normalise,
            hz_s_per_band=5 if args.PV.smooth_pv else None, 
            stack_bands=args.PV.pv_stack_bands
        )
        
        self.spatialnet_sscv = SpatialNet(
                dim_input=16,          # SSCV has 16 features per time-frequency point
                # dim_output=10,          # Change based on desired output dimensions
                num_layers=4,          # Adjust number of layers for experimentation
                dim_hidden=96,         # Hidden dimension
                dim_ffn=192,           # Feed-forward network dimension
                num_heads=4,
                num_output_bands=10,
                # kernel_size=(5, 3),    # Kernel sizes
                # conv_groups=(8, 8),    # Convolution groups
                # norms=("LN", "LN", "GN", "LN", "LN", "LN"),
                # dim_squeeze=8,         # Dimension squeeze
                num_freqs=54,         # Number of frequency bands in SSCV
                # full_share=0,
            )
        self.mlp = Spatial_Net_MLP(input_dim=84 * 96, output_dim=1)

    def forward(self, audio):
        # feature extraction
        # x = compute_mfcc_batch(audio, device=audio.device)
        pv = self.power_vector(audio).rename(None) #[128,84,16,54] --> [b, time, ch, freq]
        pv = pv.permute(0,3,1,2) #[batch, 54, 84, 16]
        y = self.spatialnet_sscv(pv)
        y = self.mlp(y)
        return y.squeeze()