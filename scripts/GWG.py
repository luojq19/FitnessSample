import sys
sys.path.append('.')
from typing import List, Optional, Tuple

import hydra
# import pytorch_lightning as L
import pyrootutils
import copy
import time
import os
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import pandas as pd
import torch
from models.GWG_module import GwgPairSampler
from datasets.sequence_dataset import PreScoredSequenceDataset
import argparse, shutil, yaml
from utils import common
import datetime

# pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

import utils

# log = utils.get_pylogger(__name__)
to_list = lambda x: x.cpu().detach().numpy().tolist()

# def save_pairs(pairs_dict, save_path):
#     """Save generated pairs to file"""
#     log.info(f"Saving generated pairs to {save_path}")
#     with open(save_path, "w") as f:
#         for src_seq in pairs_dict:
#             for tgt_seq in pairs_dict[src_seq]:
#                 f.write(f"{src_seq},{tgt_seq}\n")

def _worker_fn(args, logger):
    """Worker function for multiprocessing.

    Args:
        args (tuple): (worker_i, exp_cfg, inputs)
            worker_i: worker id. Used for setting CUDA device.
            exp_cfg: model config.
            inputs: list of inputs to process.

    Returns:
        all_outputs: results of GWG.
    """
    worker_i, exp_cfg, inputs = args
    model = GwgPairSampler(
        exp_cfg.predictor_dir,
        exp_cfg.temperature,
        exp_cfg.ckpt_name,
        exp_cfg.verbose,
        exp_cfg.gibbs_samples,
        device=f"cuda:0",
        inverse_sign=exp_cfg.inverse_sign,
        mutation_sites=exp_cfg.mutation_sites,
    )
    all_outputs = []
    for batch in inputs:
        all_outputs.append(model(batch))
    logger.info(f'Done with worker: {worker_i}')
    return all_outputs

def _setup_dataset(cfg):
    # if cfg.data.csv_path is not None:
    #     raise ValueError(f'cfg.data.csv_file must be None.')
    # data_dir = os.path.dirname(cfg.experiment.predictor_dir).replace('ckpt', 'data')
    # if '_custom' in data_dir:
    #     data_dir = data_dir.replace('_custom', '')
    # for i, subdir in enumerate(data_dir.split('/')):
    #     if 'percentile' in subdir:
    #         break
    # data_dir = '/'.join(data_dir.split('/')[:i+1])
    # cfg.data.csv_path = os.path.join(data_dir, 'base_seqs.csv')
    # if not os.path.exists(cfg.data.csv_path):
    #     raise ValueError(f'Could not find dataset at {cfg.data.csv_path}.')

    return PreScoredSequenceDataset(cfg.csv_path, cfg.cluster_cutoff, cfg.max_visits, cfg.task)

def generate_pairs(cfg: DictConfig, sample_write_path: str, logger) -> Tuple[dict, dict]:
    """Generate pairs using GWG."""
    cfg = copy.deepcopy(cfg)
    # run_cfg = cfg.run

    # set seed for random number generators in pytorch, numpy and python.random
    common.seed_all(cfg.seed)

    dataset = _setup_dataset(cfg)
    logger.info('Set up dataset.')

    # Special settings for debugging.
    exp_cfg = cfg
    epoch = 0
    start_time = time.time()
    while epoch < cfg.max_epochs and len(dataset):
        epoch += 1
        epoch_start_time = time.time()
        if cfg.debug:
            batch_size = 2
        else:
            batch_size = len(dataset)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        # Run sampling with workers
        batches_per_worker = [[]]
        for i, batch in enumerate(dataloader):
            batches_per_worker[i].append(batch)
            if cfg.debug:
                break
        logger.info(f"using GPU: {torch.device('cuda')}" )
        all_worker_outputs = [
            _worker_fn((0, exp_cfg, batches_per_worker[0]), logger)
        ]

        # Process results.
        epoch_pair_count = 0
        candidate_seqs = []
        for worker_results in all_worker_outputs:
            for new_pairs in worker_results:
                if new_pairs is None:
                    continue
                candidate_seqs.append(
                    new_pairs[['mutant_sequences', 'mutant_scores']].rename(
                        columns={'mutant_sequences': 'sequences', 'mutant_scores': 'scores'}
                    )
                )
                epoch_pair_count += dataset.add_pairs(new_pairs, epoch)
        if len(candidate_seqs) > 0:
            candidate_seqs = pd.concat(candidate_seqs)
            candidate_seqs.drop_duplicates(subset='sequences', inplace=True)
        epoch_elapsed_time = time.time() - epoch_start_time

        logger.info(f"Epoch {epoch} finished in {epoch_elapsed_time:.2f} seconds")
        logger.info("------------------------------------")
        logger.info(f"Generated {epoch_pair_count} pairs in this epoch")
        dataset.reset()
        if epoch < cfg.max_epochs and len(candidate_seqs) > 0:
            dataset.add(candidate_seqs)
            dataset.cluster()
        logger.info(f"Next dataset = {len(dataset)} sequences")
    dataset.pairs.to_csv(sample_write_path, index=False)
    elapsed_time = time.time() - start_time
    logger.info(f'Finished generation in {elapsed_time:.2f} seconds.')
    logger.info(f'Samples written to {sample_write_path}.')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='./configs/train.yml')
    parser.add_argument('--logdir', type=str, default='./logs_new')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--gen', action='store_true')
    args = parser.parse_args()
    
    return args

def main():
    start_overall = time.time()
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    args = get_args()
    # Load configs
    config = common.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    common.seed_all(config.seed if args.seed is None else args.seed)
    
    # Logging
    log_dir = common.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
    logger = common.get_logger('GWG', log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    
    logger.info(f'predictor_dir: {config.predictor_dir}')
    
    # Set-up output path
    output_dir = os.path.join(config.predictor_dir, 'samples_' + now)
    cfg_write_path = os.path.join(output_dir, 'config.yaml')
    os.makedirs(os.path.dirname(cfg_write_path), exist_ok=True)
    with open(cfg_write_path, 'w') as f:
        yaml.dump(config, f)
    logger.info(f'Config saved to {cfg_write_path}')
    
    # Generate samples for multiple seeds.
    seed = config.seed if args.seed is None else args.seed
    sample_write_path = os.path.join(output_dir, f'seed_{seed}.csv')
    logger.info(f'On seed {seed}. Saving results to {sample_write_path}')
    if config.inverse_sign:
        logger.info(f'Inverse sign is on.')
    if config.mutation_sites is not None:
        logger.info(f'Mutation sites: {config.mutation_sites}')
    generate_pairs(config, sample_write_path, logger)

if __name__ == "__main__":
    main()
