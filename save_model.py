import os
import torch


class Save_Max_Keep():
    def __init__(self, max_to_keep):
        self.max_to_keep = max_to_keep
        self.last_checkpoint = []

    def save(self, model, optimizer, model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.last_checkpoint.append(model_path)
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()                 
                    }, model_path)
        # torch.save(model.state_dict(), model_path)
        if len(self.last_checkpoint) > self.max_to_keep:
            checkpoint_delete = self.last_checkpoint.pop(0)
            os.remove(checkpoint_delete)

def Load_Model(args, model, model_path, optimizer=None, device='cpu'):
    checkpoint_path = f"{model_path}/best_model_epoch_{args.epoch}.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load the optimizer state dict if an optimizer is provided
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch_start = args.epoch

    return model, optimizer, epoch_start
