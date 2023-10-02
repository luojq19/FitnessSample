import sys
sys.path.append('.')
from typing import List, Optional, Tuple
import hydra
import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from models import BaseCNN
from sklearn.model_selection import train_test_split
from random import sample
from petsc4py import PETSc
from slepc4py import SLEPc
from sporco.admm import bpdn
from scipy.sparse.csgraph import laplacian
from scipy.sparse import csr_matrix
from omegaconf import DictConfig
from omegaconf import OmegaConf
import pyrootutils
import torch
from copy import deepcopy
import logging
import time
import os
from datetime import datetime
import argparse
from utils import common
import shutil
from datasets.fitness_dataset import AA_LIST
import yaml
import torch

torch.set_num_threads(4)
# logging.basicConfig()
# logging.root.setLevel(logging.NOTSET)
# logger = logging.getLogger('Graph-based Smoothing')
# pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# ALPHABET = list("ARNDCQEGHILKMFPSTWYV")
ALPHABET = AA_LIST

def run_predictor(seqs, batch_size, predictor):
    batches = torch.split(seqs, batch_size, 0)
    scores = []
    for b in batches:
        if b is None:
            continue
        results = predictor(b).detach()
        scores.append(results)
    return torch.concat(scores, dim=0)


def get_neighbours_via_mutations(seq, num, single_level_only=False):
    seq_list = list(seq)
    seq_len = len(seq)
    positions = sample(list(range(seq_len)), num)
    substitutions = np.random.choice(ALPHABET, num)
    neighbours = []
    for pos, new_val in zip(positions, substitutions):
        seq_new = seq_list.copy()
        seq_new[pos] = new_val
        neighbours.append(''.join(seq_new))
    if single_level_only:
        return neighbours
    neighbours_of_neighbours = sum([get_neighbours_via_mutations(seq_neighb, num, single_level_only=True)
                                    for seq_neighb in neighbours], [])
    return neighbours_of_neighbours


def solve_eigensystem(A, number_of_requested_eigenvectors, problem_type=SLEPc.EPS.ProblemType.HEP):
    xr, xi = A.createVecs()

    E = SLEPc.EPS().create()
    E.setOperators(A, None)
    E.setDimensions(number_of_requested_eigenvectors, PETSc.DECIDE)
    E.setProblemType(problem_type)
    E.setFromOptions()
    E.setWhichEigenpairs(E.Which.SMALLEST_REAL)

    E.solve()
    nconv = E.getConverged()

    eigenvalues, eigenvectors = [], []
    if nconv > 0:
        for i in range(min(nconv, number_of_requested_eigenvectors)):
            k = E.getEigenpair(i, xr, xi)
            if k.imag == 0.0:
                eigenvalues.append(k.real)
                eigenvectors.append(xr.array.copy())
    return eigenvalues, eigenvectors


def soft_thr_matrices(x, y, gamma=0.25):
    z_1 = np.maximum(x - gamma, y)
    z_2 = np.maximum(0, np.minimum(x + gamma, y))
    f_1 = 0.5 * np.power(z_1 - x, 2) + gamma * np.absolute(z_1 - y)
    f_2 = 0.5 * np.power(z_2 - x, 2) + gamma * np.absolute(z_2 - y)
    return np.where(f_1 <= f_2, z_1, z_2)


def get_smoothed(eigenvalues, eigenvectors, weak_labels_global, iter_max = 1000):
    # Init denoising
    l1_weights = np.array([eig ** 0.5 for eig in eigenvalues])
    l1_weights = np.expand_dims(l1_weights, axis=-1)

    Y_init = weak_labels_global

    # Construct random dictionary and random sparse coefficients
    V_m = np.array(eigenvectors).T
    Y_opt = Y_init.copy()
    opt = bpdn.BPDN.Options({'Verbose': False, 'MaxMainIter': 5000,
                             'RelStopTol': 1e-5, 'AutoRho': {'RsdlTarget': 1.0}, 'L1Weight': l1_weights})

    def solve_for_label(j=0,lmbda=0.001,opt=opt):
        Y_j = Y_opt[:, [j]]
        b = bpdn.BPDN(V_m, Y_j, lmbda, opt)
        A_j = b.solve()
        return A_j

    def get_current_A(Y_opt):
        A_list = []
        for j in range(Y_opt.shape[-1]):
            A_list.append(solve_for_label(j))
        return np.hstack(A_list)

    # Optimization
    Y_opt_prev = None
    iter_i = 0
    while np.any(Y_opt != Y_opt_prev) and iter_i < iter_max:
        Y_opt_prev = deepcopy(Y_opt)
        A = get_current_A(Y_opt)
        F = V_m.dot(A)
        Y_opt = soft_thr_matrices(F, Y_init)
        iter_i += 1
    return Y_opt


def to_seq_tensor(seq):
    seq_ints = [
        ALPHABET.index(x) for x in seq
    ]
    return torch.tensor(seq_ints)


def to_batch_tensor(seq_list, subset=None, device='cpu'):
    if subset is not None:
        seq_list = seq_list[:subset]
    return torch.stack([to_seq_tensor(x) for x in seq_list]).to(device)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='./configs/train.yml')
    parser.add_argument('--logdir', type=str, default='./logs_new')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    
    return args  

def main():
    start_overall = time.time()
    args = get_args()
    # Load configs
    config = common.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    common.seed_all(config.seed if args.seed is None else args.seed)
    
    # Logging
    log_dir = common.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
    logger = common.get_logger('smooth', log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    
    # Extract data path from predictor_dir
    predictor_dir = config.predictor_dir
    base_pool_path = config.base_pool_path
    df_base = pd.read_csv(base_pool_path)
    logger.info(f'Loaded base sequences {base_pool_path}')
    
    # Load predictor
    ckpt_path = os.path.join(predictor_dir, 'checkpoints/best_checkpoints.pt')
    for root, dirs, files in os.walk(predictor_dir):
        for file in files:
            if file.endswith('.yml'):
                predictor_cfg_path = os.path.join(predictor_dir, file)
    predictor_config = common.load_config(predictor_cfg_path)
    predictor = globals()[config.predictor_type](predictor_config.model)
    ckpt = torch.load(ckpt_path)
    predictor.load_state_dict(ckpt)
    predictor.to(args.device).eval()
    logger.info(f'Loading base predictor {ckpt_path}')
    
    # Random walk
    logger.info('Generating sequences by random walk from the base sequence pool..')
    start_time = time.time()
    init_seqs = df_base['sequence'].values
    all_seqs_generated = list(init_seqs)
    max_n_seqs = config.max_n_seqs
    i_pointer = 0
    while len(all_seqs_generated) < max_n_seqs:
        next_seq = all_seqs_generated[i_pointer]
        neighbs = get_neighbours_via_mutations(next_seq, num=config.random_traversal_neighborhood)
        all_seqs_generated.extend(neighbs)
        i_pointer += 1

    all_seqs = list(sorted(set(all_seqs_generated)))
    all_seqs_pt = to_batch_tensor(all_seqs, subset=None, device=args.device)
    node_scores_init = run_predictor(all_seqs_pt, batch_size=256, predictor=predictor).cpu().numpy()
    
    print('node_scores_init:', node_scores_init.shape)
    # preserving the explored upper tail of predictor's outputs
    _, indices_all = train_test_split(
        np.arange(len(all_seqs)),
        test_size=config.subsample,
        stratify=np.digitize(
            node_scores_init,
            bins=np.quantile(
                node_scores_init,
                q=np.arange(0, 1, 0.01))
            )
    )
    elapsed_time = time.time() - start_time
    logger.info(f'Finished generation in {elapsed_time:.2f} seconds')
    
    all_seqs_list = [all_seqs[i] for i in indices_all]
    import json
    print(all_seqs_list[0])
    # with open('data/all_seqs_list.json', 'w') as f:
    #     json.dump(all_seqs_list, f)
    # input()
    # to access later the original list of strings, some of the following methods perform inplace operations
    all_seqs_list_orig = deepcopy(all_seqs_list)
    node_scores_init = node_scores_init[indices_all]

    logger.info('Creating KNN graph..')
    start_time = time.time()
    ohe = OneHotEncoder()
    all_seqs_list = ohe.fit_transform([list(seq) for seq in all_seqs_list])
    print(all_seqs_list.shape)
    knn_graph = kneighbors_graph(
        all_seqs_list, n_neighbors=500, metric='l1', mode='distance',
        include_self=True, n_jobs=20)
    
    knn_graph = (knn_graph + knn_graph.T) / 2
    knn_graph = csr_matrix((1 / knn_graph.data, knn_graph.indices, knn_graph.indptr))
    elapsed_time = time.time() - start_time
    logger.info(f'Finished kNN construction in {elapsed_time:.2f} seconds')

    logger.info('Computing Laplacian..')
    start_time = time.time()
    laplacian_normed = laplacian(knn_graph, normed=True)
    laplacian_normed_csr = laplacian_normed.tocsr()
    p1 = laplacian_normed_csr.indptr
    p2 = laplacian_normed_csr.indices
    p3 = laplacian_normed_csr.data
    petsc_laplacian_normed_mat = PETSc.Mat().createAIJ(size=laplacian_normed_csr.shape, csr=(p1, p2, p3))
    elapsed_time = time.time() - start_time
    logger.info(f'Finished Laplacian calculation in {elapsed_time:.2f} seconds')

    logger.info('Computing eigenvectors..')
    start_time = time.time()
    eigenvalues, eigenvectors = solve_eigensystem(
        petsc_laplacian_normed_mat,
        number_of_requested_eigenvectors=config.num_eigenvalues)
    elapsed_time = time.time() - start_time
    logger.info(f'Finished eigenvalue calculation in {elapsed_time:.2f} seconds')

    logger.info('De-noising scores of the base model..')
    weak_labels_global_orig = np.array(node_scores_init).reshape(-1, 1)
    weak_labels_global_min, weak_labels_global_max = weak_labels_global_orig.min(), weak_labels_global_orig.max()
    scaled_ub = 1
    weak_labels_global = (weak_labels_global_orig - weak_labels_global_min) / (
                weak_labels_global_max - weak_labels_global_min)
    Y_opt = get_smoothed(eigenvalues, eigenvectors, weak_labels_global)
    
    logger.info('Returning de-noised values to the original scale and storing results..')
    bool_idx = Y_opt < scaled_ub
    if config.rescaling == 'ratio':
        new_99_perc = np.quantile(Y_opt, 0.99)
        orig_99_perc = np.quantile(weak_labels_global_orig, 0.99)
        ratio = orig_99_perc/new_99_perc
        Y_opt_scaled = Y_opt.reshape((len(Y_opt),))*ratio
    elif config.rescaling == 'minmax':
        Y_opt_scaled = Y_opt.reshape((len(Y_opt),))*(weak_labels_global_max - weak_labels_global_min) + weak_labels_global_min
    else:
        raise NotImplementedError
    df_smoothed = pd.DataFrame({'sequence': all_seqs_list_orig, 'target': Y_opt_scaled})
    df_smoothed = df_smoothed[bool_idx]

    now = datetime.now()
    now = now.strftime("%Y-%m-%d_%H-%M-%S")
    if config.results_file is None:
        results_file = f'smoothed'
    results_file = f'{config.results_file}-{now}'
    results_path = os.path.join(
        log_dir, results_file+'.csv')
    logger.info(f'Writing results to {results_path}')
    df_smoothed.to_csv(results_path, index=None)
    cfg_write_path = os.path.join(
        log_dir, results_file+'.yaml')
    with open(cfg_write_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

if __name__ == '__main__':
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    main()