import torch
import torch.nn as nn
import torch.nn.functional as F
from feature import *
from network.utility import *
from ufb_banding.banding import BandingParams, BandingShape, LowerBandMode
from ufb_banding.ufb import TransformParams

class CRNN_Orientation(nn.Module):
    def __init__(self, args):
        super(CRNN_Orientation, self).__init__()
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
        self.fc3 = nn.Linear(64, self.num_classes)

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

        # Reshape for GRU input: (batch_size, time_steps, features)
        x = x.permute(0, 3, 2, 1).contiguous()  # Change to (batch_size, time_steps, features, channels)
        batch_size, time_steps, features, channels = x.shape
        x = x.view(batch_size, time_steps, -1)  # Flatten last two dimensions

        # Apply GRU layers
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)  # [batch, time_steps, 64]

        x = F.elu(self.fc1(x[:, -1, :]))  # [64,24,32]
        x = F.elu(self.fc2(x))  # [batch, 64]
        x = self.fc3(x)  # Output size is (batch_size, num_classes)
        # Reshape the output to (batch_size, 4, 33)
        x = x.view(batch_size, self.num_classes) # [batch,4,10]

        return x.squeeze()

class Conv3D_Orientation(nn.Module):
    def __init__(self,args):
        super(Conv3D_Orientation, self).__init__()
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
        input_shape = (1, 15, 54, 84)
        output_size = self.encoder._calculate_output_size(input_shape)

        self.mlp = MLP(input_dim=256 * (54 // 16) * (84 // 16), output_dim=1)

    def forward(self, audio):
        pv = self.power_vector(audio)
        pv = pv.rename(None)
        pv = pv.permute(0,2,3,1) #[128,16,54,84]
        pv = pv[:,1:,:,:]
        # pv = pv[:,0,:,:].unsqueeze(1)
        # pv = pv[:,:4,:,:]
        # pv = pv[:,1:4,:,:]
        # pv = torch.cat((pv[:,0,:,:].unsqueeze(1),pv[:,4:,:,:]), dim=1)
        x = self.encoder(pv)
        x = self.mlp(x)
        return x.squeeze()  # [128,1]

class Hybrid_Conv_Orientation(nn.Module):
    def __init__(self,args):
        super(Hybrid_Conv_Orientation, self).__init__()
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
        self.mlp = MLP(input_dim=2*3840, output_dim=1)

    def forward(self, audio):
        pv = self.power_vector(audio)
        pv = pv.rename(None)
        pv = pv.permute(0,2,3,1) 

        spec_feature = self.spec_encoder(pv[:, 0, :, :].unsqueeze(1)) # [128,3840]
        spatial_feature = self.spatial_encoder(pv[:, 1:, :, :]) # [128,3840]
        x_combined = torch.cat((spec_feature, spatial_feature), dim=1)

        # Pass through the MLP
        out = self.mlp(x_combined) 
        return out.squeeze()

class CRNN_PV_Orientation(nn.Module):
    def __init__(self, args):
        super(CRNN_PV_Orientation, self).__init__()

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
        self.fc3 = nn.Linear(64, self.num_classes)

    def forward(self, audio):
        # feature extraction
        # x = compute_mfcc_batch(audio, device=audio.device)
        pv = self.power_vector(audio).rename(None)
        pv = pv.permute(0,2,3,1)
        pv = pv.reshape(pv.shape[0],-1,pv.shape[3])
        x = pv.unsqueeze(1)
        # x = torch.cat((x,pv),dim=1).unsqueeze(1)
        # Apply convolutional layers
        x = self.max_pool(F.elu(self.bn1(self.conv1(x)))) #[64,64,10,199]
        x = self.dropout1(x)

        x = self.max_pool(F.elu(self.bn2(self.conv2(x)))) #[64,128,5,99]
        x = self.dropout2(x)

        x = self.max_pool(F.elu(self.bn3(self.conv3(x)))) #[64,128,2,49]
        x = self.dropout3(x)

        x = self.max_pool(F.elu(self.bn4(self.conv4(x))))
        x = self.dropout4(x)  # [batch, 128, h, w]

        # Reshape for GRU input: (batch_size, time_steps, features)
        x = x.permute(0, 3, 2, 1).contiguous()  # Change to (batch_size, time_steps, features, channels)
        batch_size, time_steps, features, channels = x.shape # [128,5,54,128]
        x = x.view(batch_size, time_steps, -1)  # Flatten last two dimensions

        # Apply GRU layers
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)  # [batch, time_steps, 64]

        x = F.elu(self.fc1(x[:, -1, :]))  # [64,24,32]
        x = F.elu(self.fc2(x))  # [batch, 64]
        x = self.fc3(x)  # Output size is (batch_size, num_classes)
        # Reshape the output to (batch_size, 4, 33)
        x = x.view(batch_size, self.num_classes,1) # [batch,4,10]

        return x.squeeze()

class ParamNet_Orientation(torch.nn.Module):
    def __init__(self,args):
        super(ParamNet_Orientation,self).__init__()
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
        # else:
        self.fc_1_single = nn.Linear(96, 96)
        self.fc_2 = nn.Linear(96, 48)
        self.fc_3 = nn.Linear(48, 1)
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

        if self.single_channel == False:
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
        else:
            x=self.avgpool_1(x)
            x=x.reshape(-1, 96)
            x=self.fc_1_single(x)
            x=self.drp(x)
            x=self.fc_3(self.fc_2(x))
        # joinly estimate room volume, area, acoustic parameters (T60,DRR)
        # 1-10 --> DRR, 11-20 --> RT60, 21 --> Volume, 22 --> Area
        return x.squeeze()

class ParamNet_Single_Orientation(torch.nn.Module):
    def __init__(self,args):
        super(ParamNet_Single_Orientation,self).__init__()
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
        self.fc_3 = nn.Linear(48, 1)
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
