
from torchinfo import summary
import torch
from optimization import *
from feature import *
from network import *
from network.acoustic import *
from network.room import *
from network.oritentation import *
import hydra, os

def gpu_config(gpu_id):
    # GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.set_device(gpu_id)
        print('Device: {}_GPU_{}'.format(device, str(torch.cuda.current_device())))
    return device

@hydra.main(config_path="confs", config_name="config.yaml")
def main(args):
    if args.context_type == 'acoustic':
        if args.model_type == 'BLSTM':
            net = BLSTM_Mask(args).to(torch.device("cuda"))
        elif args.model_type == 'CRNN':
            net = CRNN(args).to(torch.device("cuda"))
        elif args.model_type == 'ParamNet':
            net = ParamNet(args).to(torch.device("cuda"))
        elif args.model_type == 'Conv3D':
            net = Conv3D_Net(args).to(torch.device("cuda"))
        elif args.model_type == 'Hybrid_Conv':
            net = Hybrid_Conv_Net(args).to(torch.device("cuda"))
        elif args.model_type == 'CRNN_PV':
            net = CRNN_PV(args).to(torch.device("cuda"))
        elif args.model_type == 'SpatialNet':
            net = SpatialNet_PV(args).to(torch.device("cuda"))
    elif args.context_type == 'room':
        if args.model_type == 'Room_CNN':
            net = Room_CNN().to(torch.device("cuda"))
        elif args.model_type == 'ParamNet':
            net = Room_ParamNet(args).to(torch.device("cuda"))
        elif args.model_type == 'Conv3D':
            net = Room_Conv3D_Net(args).to(torch.device("cuda"))
        elif args.model_type == 'CRNN_PV':
            net = Room_CRNN_PV(args).to(torch.device("cuda"))
        elif args.model_type == 'CNN':
            net = Room_Model(args).to(torch.device("cuda"))
        elif args.model_type == 'CRNN':
            net = Room_CRNN(args).to(torch.device("cuda"))
        elif args.model_type == 'ParamNet_Single':
            net = Room_ParamNet_Single(args).to(torch.device("cuda"))
    elif args.context_type == 'orientation':
        if args.model_type == 'Conv3D':
            net = Conv3D_Orientation(args).to(torch.device("cuda"))
        elif args.model_type == 'Hybrid_Conv':
            net = Hybrid_Conv_Orientation(args).to(torch.device("cuda"))
        elif args.model_type == 'CRNN':
            net = CRNN_Orientation(args).to(torch.device("cuda"))
        elif args.model_type == 'CRNN_PV':
            net = CRNN_PV_Orientation(args).to(torch.device("cuda"))
        elif args.model_type == 'ParamNet':
            net = ParamNet_Orientation(args).to(torch.device("cuda"))
        elif args.model_type == 'ParamNet_Single':
            net = ParamNet_Single_Orientation(args).to(torch.device("cuda"))
        summary(net, input_size=(128, 4, 64000))

if __name__ == '__main__':
    main()

