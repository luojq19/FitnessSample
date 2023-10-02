import torch
import numpy as np
from pymoo.indicators.hv import HV
from polyleven import levenshtein

# Function to calculate Root Mean Square Error (RMSE)
def calculate_rmse(prediction, ground_truth):
    """
    Calculate the Root Mean Square Error between two 1D tensors.
    
    Parameters:
        prediction (torch.Tensor): 1D tensor containing the predicted values.
        ground_truth (torch.Tensor): 1D tensor containing the ground truth values.
        
    Returns:
        float: Root Mean Square Error (RMSE)
    """
    squared_diffs = (prediction - ground_truth) ** 2
    rmse = torch.sqrt(torch.mean(squared_diffs))
    return rmse

# Function to calculate Mean Absolute Error (MAE)
def calculate_mae(prediction, ground_truth):
    """
    Calculate the Mean Absolute Error between two 1D tensors.
    
    Parameters:
        prediction (torch.Tensor): 1D tensor containing the predicted values.
        ground_truth (torch.Tensor): 1D tensor containing the ground truth values.
        
    Returns:
        float: Mean Absolute Error (MAE)
    """
    mae = torch.mean(torch.abs(prediction - ground_truth))
    return mae

def calc_hypervolume(scores1, scores2, ref_score1, ref_score2):
    '''calculate hypervolume'''
    indicator = HV(ref_point=np.array([ref_score1, ref_score2]))
    hv = indicator(np.array([scores1, scores2]).T)
    
    return hv

def diversity(seqs):
    num_seqs = len(seqs)
    total_dist = 0
    for i in range(num_seqs):
        for j in range(num_seqs):
            x = seqs[i]
            y = seqs[j]
            if x == y:
                continue
            total_dist += levenshtein(x, y)
    return total_dist / (num_seqs*(num_seqs-1))

def novelty(sampled_seqs, base_pool_seqs):
    # sampled_seqs: top k
    # existing_seqs: range dataset
    all_novelty = []
    for src in sampled_seqs:  
        min_dist = 1e9
        for known in base_pool_seqs:
            dist = levenshtein(src, known)
            if dist < min_dist:
                min_dist = dist
        all_novelty.append(min_dist)
        
    return all_novelty

if __name__ == '__main__':
    a = torch.arange(10, dtype=torch.float32).reshape(2,5)
    b = torch.arange(10, dtype=torch.float32).reshape(2,5)
    a = torch.randn((512,2))
    b = torch.randn((512,2))
    acc = calculate_rmse(a, b)
    print(acc)
