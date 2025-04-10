import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def convert_filename_to_int(filename):
    name = os.path.splitext(filename)[0]
    number = int(name.lstrip('0'))
    return number

class SS_Dataset(Dataset):
    def __init__(self, base_dir, split='train', task='acoustics/c50_db', sample_rate=16000, type='lite', scp_dir='/media/sbsprl/data/Hanyu/Acoustic_context_estimation/scp_files', noise=False):
        self.sr = sample_rate
        if type == 'lite':
            scp_path = os.path.join(scp_dir, 'lite_' + split + '.scp')
        else:
            scp_path = os.path.join(scp_dir, split + '.scp')
            
            
        if split == 'test':
            base_dir = os.path.join(base_dir, split)
        else:
            base_dir = os.path.join(base_dir, 'train')
        

        with open(scp_path, 'r') as file:
            file_names = file.readlines()
        
        self.file_paths = []
        self.refs = {}
        parquet_path = os.path.join(scp_dir, 'metadata.parquet')
        
        # Attempt to read the Parquet file using the default engine.
        try:
            self.metadata = pd.read_parquet(parquet_path)
        except AttributeError as e:
            # If the error is related to pyarrow's missing 'Device', switch to 'fastparquet'
            if "Device" in str(e):
                warnings.warn("AttributeError: pyarrow.lib has no attribute 'Device'. Falling back to fastparquet engine.")
                self.metadata = pd.read_parquet(parquet_path, engine='fastparquet')
            else:
                raise
        # self.metadata = pd.read_parquet(os.path.join(scp_dir, 'metadata.parquet'))
        
        if isinstance(task, str):
            task = task.split(',')
        self.refs = {key: [] for key in task}
        # file_names = file_names[:16612]
        for file_name in file_names:
            file_name = file_name.strip()
            idx = convert_filename_to_int(file_name)
            # if idx == 16611:
            #     continue  # Skip the index 16611
            label_data = self.metadata.loc[idx, task]
            for key in task:
                self.refs[key].append(label_data[key])
            self.file_paths.append(os.path.join(base_dir, file_name))
 
    def __getitem__(self, index):
        file_path = self.file_paths[index]
        ref = {key: torch.tensor(self.refs[key][index], dtype=torch.float32) for key in self.refs}

        try:
            waveform, sr = sf.read(file_path)
            waveform = waveform.astype(np.float32).T
            target_length = 4 * self.sr
            waveform = self._pad_or_trim(waveform, target_length)
            assert sr == self.sr, f"Sample rate mismatch: {sr} != {self.sr}"
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            raise
        
        audio_tensor = torch.from_numpy(waveform)
        return audio_tensor, ref
    
    def _pad_or_trim(self, audio, target_length):
        current_length = audio.shape[1]
        if current_length > target_length:
            audio = audio[:, :target_length]
        elif current_length < target_length:
            padding = target_length - current_length
            audio = np.pad(audio, ((0, 0), (0, padding)), 'constant', constant_values=0)
        return audio
    
    def __len__(self):
        return len(self.file_paths)

def collate_fn(batch):
    mix_batch = []
    ref_batch = []
    for mix, ref in batch:
        mix_batch.append(mix)
        ref_batch.append(ref)
    return torch.stack(mix_batch), ref_batch

def tr_val_loader(args):
    trainset = SS_Dataset(**args.dataset, split='train', task=args.acoustic_param)
    tr_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False, collate_fn=collate_fn)
    val_set = SS_Dataset(**args.dataset, split='val',task=args.acoustic_param)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False, collate_fn=collate_fn)
    return tr_loader, val_loader

def test_loader(args):
    test_set = SS_Dataset(**args.dataset, split='test', task=args.acoustic_param)
    tt_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False, collate_fn=collate_fn)
    return tt_loader