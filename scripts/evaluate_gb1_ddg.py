import sys
sys.path.append('.')
import os, argparse, time, datetime, json
from utils import common
import pandas as pd
from utils.eval import calc_hypervolume, diversity, novelty, greedy_selection
import numpy as np

now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
logger = common.get_logger(__name__)


def load_all_seqs(csv_path):
    df = pd.read_csv(csv_path)
    seq_column = 'mutant_sequences' if 'mutant_sequences' in df.columns else 'sequence'
    
    return df[seq_column].tolist()

def load_ground_truth(gt_csv_path):
    df = pd.read_csv(gt_csv_path)
    seqs = df['sequence'].tolist()
    gb1 = df['gb1'].tolist()
    ddg = df['ddg'].tolist()
    assert len(seqs) == len(gb1) == len(ddg), f'Lengths of sequences, gb1, and ddg are not equal: {len(seqs)}, {len(gb1)}, {len(ddg)}'
    n = len(seqs)
    seq2label = {seq: {'gb1': gb1[i], 'ddg': ddg[i]} for i, seq in enumerate(seqs)}
    logger.info(f'Loaded {len(seq2label)} ground truth sequences with labels')
    
    return seq2label

def get_metrics(data_list):
    mean_, median_, std_, min_, max_ = np.mean(data_list), np.median(data_list), np.std(data_list), np.min(data_list), np.max(data_list)
    
    return mean_, median_, std_, min_, max_

def evaluate(sampled_seqs, gt_seq2label, save_dir=None, tag=None):
    seqs, gb1, ddg = [], [], []
    for seq in sampled_seqs:
        if seq in gt_seq2label:
            seqs.append(seq)
            gb1.append(gt_seq2label[seq]['gb1'])
            ddg.append(gt_seq2label[seq]['ddg'])
    assert len(seqs) == len(gb1) == len(ddg), f'Lengths of sequences, gb1, and ddg are not equal: {len(seqs)}, {len(gb1)}, {len(ddg)}'
    results_df = pd.DataFrame({'sequence': seqs, 'gb1': gb1, 'ddg': ddg})
    tag = '' if tag is None else f'_{tag}'
    results_df.to_csv(os.path.join(save_dir, f'evaluation_results{tag}.csv'), index=False)
    gt_gb1 = [gt_seq2label[seq]['gb1'] for seq in seqs]
    gt_ddg = [gt_seq2label[seq]['ddg'] for seq in seqs]
    min_gb1, max_gb1 = np.min(gt_gb1), np.max(gt_gb1)
    normalize_gb1 = lambda x: (x - min_gb1) / (max_gb1 - min_gb1)
    min_ddg, max_ddg = np.min(gt_ddg), np.max(gt_ddg)
    normalize_ddg = lambda x: (x - min_ddg) / (max_ddg - min_ddg)
    gb1 = np.array(gb1)
    ddg = np.array(ddg)
    gb1_normalized = [normalize_gb1(gb1[i]) for i in range(len(gb1))]
    ddg_normalized = [normalize_ddg(ddg[i]) for i in range(len(ddg))]
    gb1_mean, gb1_median, gb1_std, gb1_min, gb1_max = get_metrics(gb1)
    ddg_mean, ddg_median, ddg_std, ddg_min, ddg_max = get_metrics(ddg)
    gb1_normalized_mean, gb1_normalized_median, gb1_normalized_std, gb1_normalized_min, gb1_normalized_max = get_metrics(gb1_normalized)
    ddg_normalized_mean, ddg_normalized_median, ddg_normalized_std, ddg_normalized_min, ddg_normalized_max = get_metrics(ddg_normalized)
    logger.info(f'GB1: mean={gb1_mean:.4f}, median={gb1_median:.4f}, std={gb1_std:.4f}, min={gb1_min:.4f}, max={gb1_max:.4f}')
    logger.info(f'DDG: mean={ddg_mean:.4f}, median={ddg_median:.4f}, std={ddg_std:.4f}, min={ddg_min:.4f}, max={ddg_max:.4f}')
    metrics_df = pd.DataFrame({'metric': ['gb1', 'ddg', 'gb1_normalized', 'ddg_normalized'],
                               'mean': [gb1_mean, ddg_mean, gb1_normalized_mean, ddg_normalized_mean],
                               'median': [gb1_median, ddg_median, gb1_normalized_median, ddg_normalized_median],
                               'std': [gb1_std, ddg_std, gb1_normalized_std, ddg_normalized_std],
                               'min': [gb1_min, ddg_min, gb1_normalized_min, ddg_normalized_min],
                               'max': [gb1_max, ddg_max, gb1_normalized_max, ddg_normalized_max]})
    metrics_df.to_csv(os.path.join(save_dir, f'evaluation_metrics{tag}.csv'), index=False)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='configs/gb1_ddg/evaluate.yml')
    # parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--sample_path', type=str, default=None)
    # parser.add_argument('--num_threads', type=int, default=64)

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    config = common.load_config(args.config)
    if args.sample_path is not None:
        config.sample_path = args.sample_path
    logger.info(f'Evaluating {config.sample_path}...')
    # sampled_seqs = load_all_seqs(config.sample_path)
    sampled_seqs = greedy_selection(config.sample_path, config.num_select, ref_score_1=config.ref_score_1, ref_score_2=config.ref_score_2, inverse_sign_1=config.inverse_sign_1, inverse_sign_2=config.inverse_sign_2)
    logger.info(f'Loaded {len(sampled_seqs)} sequences')
    gt_seq2label = load_ground_truth(config.gt_csv_path)
    evaluate(sampled_seqs, gt_seq2label, save_dir=os.path.dirname(config.sample_path), tag=args.tag)
    
    
    