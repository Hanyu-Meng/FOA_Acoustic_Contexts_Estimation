import numpy as np
from scipy.stats import pearsonr

def calculate_pov(targets, predictions):
    """
    Calculate the Percentage of Variance (PoV) for each frequency band.

    Parameters:
    targets (array-like): True values.
    predictions (array-like): Predicted values.
    fband_reduced (array-like): Frequency bands.

    Returns:
    numpy.ndarray: PoV for each frequency band.
    """
    # Convert the lists of targets and predictions into NumPy arrays
    if len(targets.shape) != 1:
        targets = np.array(targets.squeeze(), dtype=np.float32)[:, np.newaxis, :]
        predictions = np.array(predictions.squeeze(), dtype=np.float32)[:, np.newaxis, :]
    else:
        targets = np.array(targets.squeeze(), dtype=np.float32)[:, np.newaxis]
        predictions = np.array(predictions.squeeze(), dtype=np.float32)[:, np.newaxis]

    # Calculate mean ground truth for each frequency band
    mean_gt = np.mean(targets, axis=0, keepdims=True)
    
    # Calculate SS_total and SS_residual using vectorized operations
    SS_total_list = np.mean((targets - mean_gt) ** 2, axis=0)
    SS_residual_list = np.mean((predictions - targets) ** 2, axis=0)
    
    # Calculate PoV for each frequency band
    PoV = (1 - SS_residual_list / SS_total_list) * 100

    return PoV


def mean_mult(estimated_volumes, true_volumes):
    """
    Calculate the MeanMult metric.
    
    Parameters:
    estimated_volumes (array-like): Estimated volumes.
    true_volumes (array-like): True volumes.
    
    Returns:
    float: The MeanMult metric.
    """
    estimated_volumes = np.array(estimated_volumes)
    true_volumes = np.array(true_volumes)
    
    # Ensure the lengths of the input arrays match
    assert len(estimated_volumes) == len(true_volumes), "Arrays must be of the same length"
    
    # Calculate MeanMult using vectorized operations
    log_ratios = np.abs(np.log(estimated_volumes / true_volumes))
    mean_log_ratio = np.mean(log_ratios)
    mean_mult = np.exp(mean_log_ratio)
    
    return mean_mult

def calculate_mae(targets, predictions):
    """
    Calculate the Mean Absolute Error (MAE) for each frequency band.

    Parameters:
    targets (array-like): True values.
    predictions (array-like): Predicted values.
    fband_reduced (array-like): Frequency bands.

    Returns:
    numpy.ndarray: MAE for each frequency band.
    """
    if len(targets.shape) != 1:
        targets = np.array(targets.squeeze(), dtype=np.float32)[:, np.newaxis, :]
        predictions = np.array(predictions.squeeze(), dtype=np.float32)[:, np.newaxis, :]
    else:
        targets = np.array(targets.squeeze(), dtype=np.float32)[:, np.newaxis]
        predictions = np.array(predictions.squeeze(), dtype=np.float32)[:, np.newaxis]

    # Calculate MAE using vectorized operations
    mae_list = np.mean(np.abs(predictions - targets), axis=0)
    
    return mae_list

def calculate_mae_median(targets, predictions):
    """
    Calculate the Mean Absolute Error (MAE) for each frequency band.

    Parameters:
    targets (array-like): True values.
    predictions (array-like): Predicted values.
    fband_reduced (array-like): Frequency bands.

    Returns:
    numpy.ndarray: MAE for each frequency band.
    """
    if len(targets.shape) != 1:
        targets = np.array(targets.squeeze(), dtype=np.float32)[:, np.newaxis, :]
        predictions = np.array(predictions.squeeze(), dtype=np.float32)[:, np.newaxis, :]
    else:
        targets = np.array(targets.squeeze(), dtype=np.float32)[:, np.newaxis]
        predictions = np.array(predictions.squeeze(), dtype=np.float32)[:, np.newaxis]

    # Calculate MAE using vectorized operations
    mae_list = np.median(np.abs(predictions - targets), axis=0)
    
    return mae_list

def calculate_pcc(targets, predictions):
    """
    Calculate the Pearson Cross-Correlation (PCC) for each frequency band.

    Parameters:
    targets (array-like): True values.
    predictions (array-like): Predicted values.

    Returns:
    numpy.ndarray: PCC for each frequency band.
    """
    pcc_list = []
    if len(targets.shape) == 3:
        targets = targets.squeeze()
        predictions = predictions.squeeze()
    if len(targets.shape) != 1:
        # targets = np.array(targets.squeeze(), dtype=np.float32)[:, np.newaxis, :]
        # predictions = np.array(predictions.squeeze(), dtype=np.float32)[:, np.newaxis, :]
        # Calculate PCC for each frequency band
        for i in range(targets.shape[1]):
            pcc, _ = pearsonr(targets[:, i], predictions[:, i])
            pcc_list.append(pcc)
    else:
        targets = np.array(targets.squeeze(), dtype=np.float32)[:, np.newaxis]
        predictions = np.array(predictions.squeeze(), dtype=np.float32)[:, np.newaxis]
        pcc, _ = pearsonr(targets[:, 0], predictions[:, 0])
        pcc_list.append(pcc)

    # # Calculate PCC for each frequency band
    # pcc_list = []
    # for i in range(targets.shape[1]):
    #     pcc, _ = pearsonr(targets[:, i], predictions[:, i])
    #     pcc_list.append(pcc)
    
    return np.array(pcc_list)

def calculate_mae_log(targets, predictions):
    """
    Calculate the Mean Absolute Error (MAE) for each frequency band.

    Parameters:
    targets (array-like): True values.
    predictions (array-like): Predicted values.
    fband_reduced (array-like): Frequency bands.

    Returns:
    numpy.ndarray: MAE for each frequency band.
    """
    if len(targets.shape) != 1:
        targets = np.log10(np.array(targets.squeeze(), dtype=np.float32)[:, np.newaxis, :])
        predictions = np.log10(np.array(predictions.squeeze(), dtype=np.float32)[:, np.newaxis, :])
    else:
        targets = np.log10(np.array(targets.squeeze(), dtype=np.float32)[:, np.newaxis])
        predictions = np.log10(np.array(predictions.squeeze(), dtype=np.float32)[:, np.newaxis])

    # Calculate MAE using vectorized operations
    mae_list = np.mean(np.abs(predictions - targets), axis=0)
    
    return mae_list
