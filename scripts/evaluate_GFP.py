import sys
sys.path.append('.')

from typing import List, Optional, Tuple
from biotite.sequence.io import fasta
from polyleven import levenshtein
import numpy as np
import torch
import pyrootutils
import logging
import os
from omegaconf import DictConfig
import pandas as pd
from utils import common
from models import BaseCNN
from omegaconf import OmegaConf
from models.GWG_module import Encoder
import glob
from tqdm import tqdm
from datasets.fitness_dataset import AA_LIST, seq2indices
import argparse

to_np = lambda x: x.cpu().detach().numpy()
to_list = lambda x: to_np(x).tolist()
alphabet = AA_LIST
logger = common.get_logger('evaluate_GFP')

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

def evaluate_GFP(sampled_seqs, config, args):
    for root, dirs, files in os.walk(config.GFP_oracle_dir):
        for file in files:
            if file.endswith('.yml'):
                oracle_config = common.load_config(os.path.join(root, file))
                break
    oracle = globals()[oracle_config.model.model_type](oracle_config.model)
    ckpt = torch.load(os.path.join(config.GFP_oracle_dir, 'checkpoints/best_checkpoints.pt'))
    oracle.load_state_dict(ckpt)
    oracle.eval()
    oracle.to(args.device)
    logger.info(f'Loaded oracle from {config.GFP_oracle_dir}')
    outputs = []
    for seq in tqdm(sampled_seqs, desc='running oracle'):
        seq = seq2indices(seq).unsqueeze(0).to(args.device)
        with torch.no_grad():
            output = oracle(seq).item()
        outputs.append(output)
        
    gt = pd.read_csv(config.ground_truth_GFP)
    max_seen = np.max(gt.target).item()
    min_seen = np.min(gt.target).item()
    normalize = lambda x: (x - min_seen) / (max_seen - min_seen)
    outputs_normalized = [normalize(x) for x in outputs]
    logger.info(f'task\tmean\tmedian\tstd\tmax\tmin')
    logger.info(f'GFP\t{np.mean(outputs):.3f}\t{np.median(outputs):.3f}\t{np.std(outputs):.3f}\t{np.max(outputs):.3f}\t{np.min(outputs):.3f}')
    logger.info(f'GFP_normalized\t{np.mean(outputs_normalized):.3f}\t{np.median(outputs_normalized):.3f}\t{np.std(outputs_normalized):.3f}\t{np.max(outputs_normalized):.3f}\t{np.min(outputs_normalized):.3f}')
    results = {'scores_origin': outputs, 
               'scores_normalized': outputs_normalized,}

def select_all(config):
    sampled_data = pd.read_csv(config.sample_path)
    sampled_seqs = sampled_data['mutant_sequences'].tolist()
    
    return sampled_seqs

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('config', type=str, default='configs/gfp_ddg/evaluate.yml')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--sample_path', type=str, default=None)
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    config = common.load_config(args.config)
    if args.sample_path is not None:
        config.sample_path = args.sample_path
    logger.info(f'sample path: {config.sample_path}')
    sampled_seqs = select_all(config)
    evaluate_GFP(sampled_seqs, config, args)