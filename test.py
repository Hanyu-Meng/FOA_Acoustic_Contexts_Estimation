import hydra, os
from tqdm import tqdm
from datetime import datetime
import torch.optim as optim
import torch
from data import test_loader
from save_model import Load_Model
from optimization import *
from feature import *
from network import *
import numpy as np
import warnings
from train import get_target_test
from network.acoustic import *
from network.room import *
from network.oritentation import *
from metrics import *
import pickle
warnings.filterwarnings("ignore", category=UserWarning)

def gpu_config(gpu_id):
    # GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.set_device(gpu_id)
        print('Device: {}_GPU_{}'.format(device, str(torch.cuda.current_device())))
    return device

def test(args, device, model):
    tt_loader = test_loader(args)
    if args.dataset.type == 'full':
        model_path = os.path.join(args.model_path,args.context_type, args.task,f"{args.model_type}_full")
        log_path =  os.path.join(args.log_path, args.context_type, args.task,f"{args.model_type}_full")
    else:
        if args.single_channel == True:
            model_path = os.path.join(args.model_path,args.context_type, args.task,f"{args.model_type}_Single")
            log_path =  os.path.join(args.log_path, args.context_type, args.task,f"{args.model_type}_Single")
        else:
            model_path = os.path.join(args.model_path,args.context_type, args.task, args.model_type)
            log_path =  os.path.join(args.log_path, args.context_type, args.task, args.model_type)

    optimizer = optim.Adam(model.parameters(), lr=args.optimizer.lr)
    model,_ ,_ = Load_Model(args, model, model_path, optimizer, device)

    print('Test ...')
    model.eval()
    all_targets = [] 
    all_outputs = []
    with torch.no_grad():
        for audio, ref_param in tqdm(tt_loader):
            if audio.dim() == 2:
                audio = audio.to(device).unsqueeze(0)
            else: 
                audio = audio.to(device)

            target = get_target_test(ref_param, args, device)

            audio = audio.transpose(1,2)            
            if args.loss == 'MSE':
                output = model(audio) # [batch,4,64000]
            elif args.loss == 'NLL':
                mean,variance = model(audio)

            if len(target.size()) == 2:
                target = target.unsqueeze(1)
            
            output = output.squeeze()

            all_targets.append(target.cpu().numpy())
            all_outputs.append(output.cpu().numpy())
    
    with open(os.path.join(log_path, 'test_output.pkl'), 'wb') as f:
        pickle.dump({'targets': all_targets, 'outputs': all_outputs}, f)   

@hydra.main(config_path="confs", config_name="test_config.yaml")
def main(args):
    device = gpu_config(args.gpu_id)
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
    test(args, device, net)

if __name__ == '__main__':
    main()

