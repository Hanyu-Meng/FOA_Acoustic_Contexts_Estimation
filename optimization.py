import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

def gradient_clip_value(parameters, clip_value):
    torch.nn.utils.clip_grad_value_(parameters, clip_value)

def gradient_clip_norm(parameters, clip_norm):
    torch.nn.utils.clip_grad_norm_(parameters, clip_norm)

def LRScheduler(args, optimizer):
    if args.optimizer.half_lr:
        return ReduceLROnPlateau(optimizer, factor=0.5, patience=args.optimizer.patience)
    else:
        return None
    
def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class EarlyStopping:
    def __init__(self, patience=15, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

    def __call__(self, val_loss, model, optimizer, save_checkpoint, model_path, epoch):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, save_checkpoint, model_path, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, save_checkpoint, model_path, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, save_checkpoint, model_path, epoch):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        save_checkpoint.save(model, optimizer, model_path+'/epoch_{}.pth'.format(str(epoch+1)))
        self.val_loss_min = val_loss