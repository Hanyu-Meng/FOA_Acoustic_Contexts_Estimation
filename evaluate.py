# calculate evaluation metrics from the inference outputs
import pickle
import numpy as np
from metrics import *
import hydra
import os
import csv
@hydra.main(config_path="confs", config_name="test_config.yaml")
def main(args):
    if args.single_channel == True:
        log_path = os.path.join(args.log_path, args.context_type, args.task, f"{args.model_type}_Single")
        csv_file_path = os.path.join(args.model_path,args.context_type, args.task, f"{args.model_type}_Single", f'evaluation_results.csv')
    else:
        log_path = os.path.join(args.log_path, args.context_type, args.task, f"{args.model_type}")
        csv_file_path = os.path.join(args.model_path,args.context_type, args.task, args.model_type, f'evaluation_results.csv')
    pickle_file_path = os.path.join(log_path,'test_output.pkl')
    with open(pickle_file_path, 'rb') as f:
        metrics = pickle.load(f)

    targets = metrics['targets']
    outputs = metrics['outputs']
    fband_reduced = [1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000]
    fband_reduced = np.asarray(fband_reduced)
    targets = np.concatenate(targets[0:395], axis=0)
    outputs = np.concatenate(outputs[0:395], axis=0)
    if args.context_type == 'acoustic':
        pov = calculate_pov(targets, outputs)[0,:]
        mae = calculate_mae(targets, outputs)[0,:]
        pcc = calculate_pcc(targets, outputs)
    else:
        pov = calculate_pov(targets, outputs)[0]
        mae = calculate_mae(targets, outputs)[0]
        # mae_median = calculate_mae_median(targets, outputs)[0]
        pcc = calculate_pcc(targets, outputs)
        mm = mean_mult(outputs, targets)

    if args.context_type == 'acoustic':
        with open(csv_file_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Frequency Band', 'MAE', 'POV', 'PCC'])
            for i in range(len(fband_reduced)):
                csvwriter.writerow([fband_reduced[i], mae[i], pov[i], pcc[i]])
    else:
        print(f"POV value: {pov}")
        print(f"MAE value: {mae}")
        print(f"PCC value: {pcc}")
        # print(f"MAE median value: {mae_median}")
        print(f"MM value: {mm}")


if __name__ == '__main__':
    main()