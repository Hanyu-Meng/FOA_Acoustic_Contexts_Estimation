import hydra, os
from tqdm import tqdm
from datetime import datetime
import torch.optim as optim
import torch
# from room_data import tr_val_loader
from data import tr_val_loader
from save_model import Save_Max_Keep, Load_Model
from optimization import *
from feature import *
from network import *
import numpy as np
import warnings
from network.acoustic import *
from network.room import *
from network.oritentation import *
from metrics import *
import time
warnings.filterwarnings("ignore", category=UserWarning)

def gpu_config(gpu_id):
    # GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.set_device(gpu_id)
        print('Device: {}_GPU_{}'.format(device, str(torch.cuda.current_device())))
    return device

def NLLloss(y, mean, var):
    """ Negative log-likelihood loss function. """
    return (torch.log(var) + ((y - mean).pow(2)) / var).mean()

def safe_log(tensor, epsilon=1e-6):
    # Ensure all values are positive and clip to avoid log(0) or negative values
    tensor = torch.clamp(tensor, min=epsilon)
    return torch.log(tensor)

def get_target_test(ref_param, args, device):
    fband = [12.5, 25, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400,
        500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000]
    # only consider the band >= 1000 Hz
    if isinstance(ref_param, list):
        ref_param = [{k: v.to(device) for k, v in ref_dict.items()} for ref_dict in ref_param]
    else:
        ref_param = {k: v.to(device) for k, v in ref_param.items()}

    if isinstance(args.acoustic_param, str):
        task = args.acoustic_param.split(',')
    
    targets = []
    for ref_dict in ref_param:
        task_values = []
        for i, key in enumerate(task):
            value = ref_dict[key].to(torch.float32)
            if args.context_type == 'acoustic':
                value = value[19:len(fband)+3]
                if key == 'acoustics/t30_ms':
                    value = 2 * value / 1000
            elif args.context_type == 'room':
                # value = safe_log(value.unsqueeze(0)/100)  # for the room volume, and surface area, make sure the scales are similar with others
                value = value.unsqueeze(0)/100
            elif args.context_type == 'orientation':
                if key == 'speech/speaking_azimuth':
                    azimuth = torch.pi - value
                    continue
                elif key == 'speech/speaking_elevation':
                    elevation = value
                    value = torch.cos(azimuth)*torch.cos(elevation)
            task_values.append(value)
        if len(task_values) > 1:
            concatenated_task_values = torch.cat(task_values)
        else:
            concatenated_task_values = torch.unsqueeze(task_values[0], 0)
        targets.append(concatenated_task_values)
    
    target = torch.stack(targets).squeeze()
    return target

def get_target(ref_param, args, device):
    fband = [12.5, 25, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400,
        500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000]
    # only consider the band >= 1000 Hz
    if isinstance(ref_param, list):
        ref_param = [{k: v.to(device) for k, v in ref_dict.items()} for ref_dict in ref_param]
    else:
        ref_param = {k: v.to(device) for k, v in ref_param.items()}

    if isinstance(args.acoustic_param, str):
        task = args.acoustic_param.split(',')
    
    targets = []
    for ref_dict in ref_param:
        task_values = []
        for i, key in enumerate(task):
            value = ref_dict[key].to(torch.float32)
            if args.context_type == 'acoustic':
                value = value[19:len(fband)+3]
                if key == 'acoustics/t30_ms':
                    value = 2 * value / 1000
                    if args.model_type=='Conv3D' or args.model_type == 'Hybrid_Conv':
                        value = safe_log(value)
            elif args.context_type == 'room':
                value = value.unsqueeze(0)/100
                # value = safe_log(value.unsqueeze(0)/100)  # for the room volume, and surface area, make sure the scales are similar with others
            elif args.context_type == 'orientation':
                if key == 'speech/speaking_azimuth':
                    azimuth = torch.pi - value
                    continue
                elif key == 'speech/speaking_elevation':
                    elevation = value
                    value = torch.cos(azimuth)*torch.cos(elevation)
            task_values.append(value)
        if len(task_values) > 1:
            concatenated_task_values = torch.cat(task_values)
        else:
            concatenated_task_values = torch.unsqueeze(task_values[0], 0)
        targets.append(concatenated_task_values)
    
    target = torch.stack(targets).squeeze()
    return target

def train_epoch(args, tr_loader, model, optimizer, device):
    total_loss = 0.0
    model.train()
    step_num = 0.0
    for audio, ref_param in tqdm(tr_loader):
        batchloss = 0.0
        audio = audio.to(device)
        if len(audio.size()) == 2:
            audio = audio.unsqueeze(0)

        # target = torch.tensor(ref_param, dtype=torch.float32).unsqueeze(1).to(device)
        target = get_target(ref_param, args, device) 
        audio = audio.transpose(1,2)
        # [batch_idx, audio, target].pickle 
        if args.loss == 'MSE':
            est_out = model(audio) # [batch,4,64000]
            # est_out = safe_log(est_out) 
            if 'acoustics/t30_ms' in args.acoustic_param:
                if args.model_type=='Conv3D' or args.model_type=='Hybrid_Conv':
                    # or args.model_type == 'CRNN_PV'
                    est_out = safe_log(est_out) 
            # if 'room/volume' in args.acoustic_param:
            #     if args.model_type=='Conv3D' or args.model_type=='Hybrid_Conv':
            #         est_out = safe_log(est_out)  
            batchloss = F.mse_loss(est_out,target) #[32,10]
        elif args.loss == 'NLL':
            est_mean, est_variance = model(audio) # [batch,4,64000]
            # batchloss = NLLloss(target/train_std_drr,est_mean,est_variance) #[32,10]
        
        if not torch.isnan(batchloss).any():
            total_loss += batchloss.item()
            step_num += 1

        optimizer.zero_grad()
        batchloss.backward()
        gradient_clip_norm(model.parameters(), args.clip_value)
        optimizer.step()

    return total_loss/step_num

def val_epoch(args, val_loader, model, device):
    val_loss = 0.0
    batch_num = 0.0
    model.eval()
    with torch.no_grad():
        for audio, ref_param in tqdm(val_loader):
            audio = audio.to(device)
            if len(audio.size()) == 2:
                audio = audio.unsqueeze(0)

            target = get_target(ref_param, args, device)
            # target = torch.tensor(ref_param, dtype=torch.float32).unsqueeze(1).to(device)
            audio = audio.transpose(1,2)
            
            if torch.isnan(target).any():
                print("Warning: NaN detected in target tensor.")
                continue  # Skip this batch or handle it as appropriate

            if args.loss == 'MSE':
                est_out = model(audio) # [batch,4,64000]
                # est_out = safe_log(est_out) 
                if 'acoustics/t30_ms' in args.acoustic_param:
                    if args.model_type=='Conv3D' or args.model_type=='Hybrid_Conv':
                        est_out = safe_log(est_out)
                         
                # if 'room/volume' in args.acoustic_param:
                #     if args.model_type=='Conv3D' or args.model_type=='Hybrid_Conv':
                #         est_out = safe_log(est_out) 
                batchloss = F.mse_loss(est_out,target) #[32,10]
            elif args.loss == 'NLL':
                est_mean, est_variance = model(audio) # [batch,4,64000]
                # batchloss = NLLloss(target/val_std_drr,est_mean,est_variance) #[32,10]
            if not torch.isnan(batchloss):
                val_loss += batchloss.item()
                batch_num += 1

        return val_loss/batch_num
    
def run_one_epoch(args, tr_loader, val_loader, model, optimizer, device, Lrscheduler):
    tr_loss = train_epoch(args, tr_loader, model, optimizer, device)
    val_loss = val_epoch(args,val_loader, model, device)
    if args.optimizer.half_lr:
        Lrscheduler.step(val_loss)

    return tr_loss, val_loss

def train(args, device, model):
    tr_loader, val_loader = tr_val_loader(args)
    if args.dataset.type == 'full':
        if args.single_channel == True:
            model_path = os.path.join(args.model_path,args.context_type, args.task,f"{args.model_type}_Single_full")
            log_path =  os.path.join(args.log_path, args.context_type, args.task,f"{args.model_type}_Single_full")
        else:
            model_path = os.path.join(args.model_path,args.context_type, args.task,f"{args.model_type}_full")
            log_path =  os.path.join(args.log_path, args.context_type, args.task,f"{args.model_type}_full")
    elif args.single_channel == True:
        model_path = os.path.join(args.model_path,args.context_type, args.task,f"{args.model_type}_Single")
        log_path =  os.path.join(args.log_path, args.context_type, args.task,f"{args.model_type}_Single")
    else:
        model_path = os.path.join(args.model_path,args.context_type, args.task,args.model_type)
        log_path =  os.path.join(args.log_path, args.context_type, args.task,args.model_type)

    optimizer = optim.Adam(model.parameters(), lr=args.optimizer.lr)
    Lrscheduler = LRScheduler(args, optimizer)
    if args.cont: model, optimizer, epoch_start = Load_Model(args, model, model_path, optimizer, device)
    else: epoch_start = 0
    save_checkpoint = Save_Max_Keep(args.max_to_keep)
    best_val_loss = float('inf') 
    print('Training ...')
    if not os.path.exists(log_path): os.makedirs(log_path)
    with open(log_path + '/' + args.model_type + ".csv", "a") as results:
        results.write("'"'Validation error'"', '"'Training error'"', '"'Epoch'"', '"'D/T'"'\n")
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    for epoch in range(epoch_start, args.num_epoch):
        if not os.path.exists(model_path): os.makedirs(model_path)
        if not os.path.exists(log_path): os.makedirs(log_path)
        print('Training E%d (ver=%s, gpu=%d, params=%g)' %(epoch+1, model_path, args.gpu_id, model_size))
        tr_loss, val_loss = run_one_epoch(args, tr_loader, val_loader, model, optimizer, device, Lrscheduler)
        current_lr = get_learning_rate(optimizer)
        print(f'Train loss: {tr_loss:.2f} | Val loss: {val_loss:.2f} | Learning Rate: {current_lr:.6f}')
        with open(log_path + '/' + args.model_type + ".csv", "a") as results:
            results.write("%g, %g, %d, %s\n" % (val_loss, tr_loss, epoch+1, datetime.now().strftime('%Y-%m-%d/%H:%M:%S')))
        # --------- Save last max_to_keep checkpoints ------------ #

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            current_time = time.strftime("%Y%m%d-%H%M%S")
            save_checkpoint.save(model, optimizer, model_path + '/best_model_epoch_{}_{}.pth'.format(str(epoch+1),current_time))
            print(f'Saved new best model with validation loss: {val_loss:.2f}')
        # save_checkpoint.save(model, optimizer, model_path+'/epoch_{}.pth'.format(str(epoch+1)))

@hydra.main(config_path="confs", config_name="config.yaml")
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
        elif args.model_type == 'ParamNet_Single':
            net = ParamNet_Single(args).to(torch.device("cuda"))
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
    if args.train: train(args, device, net)


if __name__ == '__main__':
    main()

