import sys
sys.path.append('.')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dists
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import time, os, argparse, shutil, datetime, yaml, random
from tqdm import tqdm
from utils import common
from datasets.fitness_dataset import AA_LIST, seq2indices, indices2seq
from models.predictors import BaseCNN
from models.GWG_module_2 import Encoder
from utils.pareto_pref_vec import get_d_paretomtl, circle_points, get_d_paretomtl_batch

to_np = lambda x: x.cpu().detach().numpy()
to_list = lambda x: to_np(x).tolist()

class SampleSequenceDataset(Dataset):
    def __init__(self, csv_path, seq_col='sequence', obj_1='GFP', obj_2='stability', subsamples=None, logger=None) -> None:
        super().__init__()
        self.logger = logger if logger is not None else common.get_logger('SampleSequenceDataset')
        self.raw_data = pd.read_csv(csv_path)
        if subsamples is not None:
            self.raw_data = self.raw_data.sample(n=subsamples)
        self.sequence = self.raw_data[seq_col].tolist()
        # self.sequence = [seq2indices(seq) for seq in self.sequence]
        self.obj_1 = self.raw_data[obj_1].tolist()
        self.obj_2 = self.raw_data[obj_2].tolist()
        self.logger.info(f'Loaded {len(self.sequence)} sequences from {csv_path}')
        
    def __len__(self):
        return len(self.sequence)
    
    def __getitem__(self, index):
        return self.sequence[index]
    
    def clear(self):
        self.raw_data = None
        self.sequence = []
        self.logger.info('Cleared dataset')

    def add(self, seq):
        if isinstance(seq, str):
            self.sequence.append(seq)
        elif isinstance(seq, list) and isinstance(seq[0], str):
            self.sequence.extend(seq)
        else:
            raise ValueError(f'Invalid sequence type: {type(seq)}')
        
    def random_sample(self, n):
        self.sequence = random.sample(self.sequence, min(n, len(self.sequence)))

# Wrapper class to reverse the sign of the output
class InversePredictor(nn.Module):
    def __init__(self, model):
        super(InversePredictor, self).__init__()
        print(f'InversePredictor: {model.__class__.__name__}')
        self.model = model

    def forward(self, x):
        return -self.model(x)
    
class ParetoSampler(nn.Module):
    def __init__(self, config, args, logger=None) -> None:
        super().__init__()
        self.logger = logger if logger is not None else common.get_logger('ParetoSampler')
        self.config = config
        self.args = args
        self.ckpt_name = config.ckpt_name
        self.device = args.device
        self.logger.info(f'Using GPU: {torch.device(self.device)}')
        self.inverse_sign_1 = config.inverse_sign_1
        self.inverse_sign_2 = config.inverse_sign_2
        self.predictor_tokenizer = Encoder(alphabet=AA_LIST)
        self.predictor_1 = self._setup_predictor(config.predictor_1_dir, self.inverse_sign_1)
        self.predictor_2 = self._setup_predictor(config.predictor_2_dir, self.inverse_sign_2)
        self.logger.info(f'predictor_1 dir: {config.predictor_1_dir}')
        self.logger.info(f'predictor_2 dir: {config.predictor_2_dir}')
        self.num_tokens = len(self.predictor_tokenizer.alphabet)
        self.temp = config.temperature
        self.gibbs_samples = config.gibbs_samples
        self._verbose = config.verbose
        self.mutation_sites = config.mutation_sites
        self.num_pref_vec = config.num_pref_vec
        self.pref_vec = torch.tensor(circle_points([1], [self.num_pref_vec])[0]).float().to(self.device)
        self.logger.info(f'pref_vec: {self.pref_vec}')
        self.pref_index = config.pref_index
        self.logger.info(f'pref_index: {self.pref_index}; pref_vec: {self.pref_vec[self.pref_index]}')
        
    def _setup_predictor(self, predictor_dir, inverse_sign=False):
        # Load model weights.
        ckpt_path = os.path.join(predictor_dir, 'checkpoints', self.ckpt_name)
        ckpt = torch.load(ckpt_path)
        for root, dirs, files in os.walk(predictor_dir):
            for file in files:
                if file.endswith('.yml'):
                    cfg_path = os.path.join(root, file)
        ckpt_cfg = common.load_config(cfg_path)
        ckpt_cfg.model.make_one_hot = False
        predictor = BaseCNN(ckpt_cfg.model)
        predictor.load_state_dict(ckpt)
        predictor.eval()
        predictor.to(self.device)
        self.logger.info(predictor)
        if inverse_sign:
            predictor = InversePredictor(predictor)
        return predictor
    
    def _compose_two_gradients(self, grad1, grad2, inputs, score_1, score_2, initial=False):
        # debug: downgrade to single-obj by setting one gradient to 0
        # grad1 = torch.zeros_like(grad1)
        # grad2 = torch.zeros_like(grad2)
        # return grad1 # tag: GFP_single
        grads = [grad1, grad2]
        scores = [score_1, score_2]
        # print(f'initial: {initial}')
        weights = get_d_paretomtl_batch(grads, scores, self.pref_vec, self.pref_index, initial=initial)
        # print(f'grads: {grads[0].shape}, {grads[1].shape}, scores: {scores[0].shape}, {scores[1].shape}')
        # input()
        # weights = torch.tensor([0.5, 0.5])
        weights = torch.max(torch.min(weights, torch.tensor(1.0)), torch.tensor(0.0))
        weights = weights.to(self.device)
        # if len(grad1) > 1:
        #     weights_ = weights.squeeze(-1).squeeze(-1)
        #     print(weights_[:, 0].detach().cpu())
        #     print(weights_[:, 1].detach().cpu())
            # print(f'score_1: {score_1.detach().cpu()}')
            # print(f'score_2: {score_2.detach().cpu()}')
            # input()
        # print(f'grad1.shape: {grad1.shape}, weights: {weights.shape}, weights[:, 0]: {weights[:, 0].shape}, weights[:, 1]: {weights[:, 1].shape}')
        # print((weights[:, 0].reshape(-1, 1, 1) * grad1 + weights[:, 1].reshape(-1, 1, 1) * grad2).shape)
        # input()
        return weights[:, 0].reshape(-1, 1, 1) * grad1 + weights[:, 1].reshape(-1, 1, 1) * grad2
    
    def _calc_local_diff_2(self, seq_one_hot, initial=False):
        # construct local difference 
        score_1 = self.predictor_1(seq_one_hot)
        score_2 = self.predictor_2(seq_one_hot)
        gx1 = torch.autograd.grad(score_1.sum(), seq_one_hot)[0]
        gx2 = torch.autograd.grad(score_2.sum(), seq_one_hot)[0]
        gx = self._compose_two_gradients(gx1, gx2, seq_one_hot, score_1, score_2, initial=initial)
        gx_cur = (gx * seq_one_hot).sum(-1)[:, :, None]
        delta_ij = gx - gx_cur
        return delta_ij
    
    def set_rows_to_neg_inf(self, tensor, indices):
        n = tensor.shape[0]
        for i in range(n):
            if i not in indices:
                tensor[i] = torch.full_like(tensor[i], float(-1e9), device=tensor.device)
        return tensor
    
    def _gibbs_sampler(self, seq_one_hot, initial=False):
        delta_ij = self._calc_local_diff_2(seq_one_hot, initial=initial)
        delta_ij = delta_ij[0]
        if self.mutation_sites is not None:
            delta_ij = self.set_rows_to_neg_inf(delta_ij, self.mutation_sites)
        # One step of GWG sampling.
        def _gwg_sample():
            seq_len, num_tokens = delta_ij.shape
            # Construct proposal distributions
            gwg_proposal = dists.OneHotCategorical(logits = delta_ij.flatten() / self.temp)
            r_ij = gwg_proposal.sample((self.gibbs_samples,)).reshape(
                self.gibbs_samples, seq_len, num_tokens)

            # [num_samples, L, 20]
            seq_token = torch.argmax(seq_one_hot, dim=-1)
            mutated_seqs = seq_token.repeat(self.gibbs_samples, 1)
            seq_idx, res_idx, aa_idx = torch.where(r_ij)
            mutated_seqs[(seq_idx, res_idx)] = aa_idx
            return mutated_seqs
        
        return _gwg_sample
    
    def _make_one_hot(self, seq, differentiable=False):
        seq_one_hot = F.one_hot(seq, num_classes=self.num_tokens)
        if differentiable:
            seq_one_hot = seq_one_hot.float().requires_grad_()
        return seq_one_hot
    
    def _evaluate_one_hot_2(self, seq):
        input_one_hot = self._make_one_hot(seq)
        model_1_out = self.predictor_1(input_one_hot)
        model_2_out = self.predictor_2(input_one_hot)
        
        return model_1_out, model_2_out
    
    def _decode(self, one_hot_seq):
        return self.predictor_tokenizer.decode(one_hot_seq)
    
    def _metropolis_hasting(self, mutants, source_one_hot, delta_score, initial=False):
        # print(f'mutants: {mutants.shape}')
        source = torch.argmax(source_one_hot, dim=-1)
    
        # [num_seq, L]
        mutated_indices = mutants != source[None]
        # [num_seq, L, 20]
        mutant_one_hot = self._make_one_hot(mutants, differentiable=True)
        mutated_one_hot = mutant_one_hot * mutated_indices[..., None]
        # print(f'MH: {source_one_hot[None].shape}')
        # print(f'source_delta_ij:')
        source_delta_ij = self._calc_local_diff_2(source_one_hot[None], initial=initial)
        # print(f'MH: {mutant_one_hot.shape}')
        # print(f'mutant_delta_ij:')
        mutant_delta_ij = self._calc_local_diff_2(mutant_one_hot, initial=initial)

        orig_source_shape = source_delta_ij.shape
        orig_mutant_shape = mutant_delta_ij.shape

        # Flatten starting from the second to last dimension and apply softmax
        q_source = source_delta_ij.flatten(start_dim=-2)
        q_source = F.softmax(q_source / self.temp, dim=-1)

        q_mutant = mutant_delta_ij.flatten(start_dim=-2)
        q_mutant = F.softmax(q_mutant / self.temp, dim=-1)

        # Reshape back to the original shape
        q_source = q_source.view(orig_source_shape).squeeze(0)
        q_mutant = q_mutant.view(orig_mutant_shape)
        
        mutation_tuple = torch.nonzero(mutated_one_hot, as_tuple=True)
        q_ij_source = q_source[mutation_tuple[1], mutation_tuple[2]]
        q_ij_mutant = q_mutant[torch.arange(q_mutant.shape[0]).to(self.device), mutation_tuple[1], mutation_tuple[2]] 
        q_ij_ratio = q_ij_mutant / q_ij_source
        accept_prob = torch.exp(delta_score)*q_ij_ratio.to(self.device)
        
        mh_step = accept_prob < torch.rand(accept_prob.shape).to(self.device)
        return mh_step
    
    def _evaluate_mutants(
            self,
            *,
            mutants,
            score_1,
            score_2,
            source_one_hot,
            initial=False,
        ):
        all_mutated_scores_1, all_mutated_scores_2 = self._evaluate_one_hot_2(mutants)
        delta_score = (all_mutated_scores_1 - score_1 + all_mutated_scores_2 - score_2) / 2

        accept_mask = self._metropolis_hasting(
            mutants, source_one_hot, delta_score, initial=initial)
        accepted_x = to_list(mutants[accept_mask])
        accepted_seq = [self._decode(x) for x in accepted_x]
        accepted_score_1 = to_list(all_mutated_scores_1[accept_mask])
        accepted_score_2 = to_list(all_mutated_scores_2[accept_mask])
        return pd.DataFrame({
            'mutant_sequences': accepted_seq,
            'mutant_scores_1': accepted_score_1,
            'mutant_scores_2': accepted_score_2
        }), mutants[accept_mask]
    
    def compute_mutant_stats(self, source_seq, mutant_seqs):
        num_mutated_res = torch.sum(
            ~(mutant_seqs == source_seq[None]), dim=-1)
        return num_mutated_res
    
    def forward(self, seqs, initial=False):
        # Tokenize
        tokenized_seqs = self.predictor_tokenizer.encode(seqs).to(self.device)
        total_num_seqs = len(tokenized_seqs)
        
        all_mutant_pairs = []
        pbar = tqdm(enumerate(zip(seqs, tokenized_seqs)), total=total_num_seqs, desc='Sampling', dynamic_ncols=True)
        for i, (real_seq, token_seq) in pbar:
            start_time = time.time()
            seq_one_hot = self._make_one_hot(token_seq, differentiable=True)
            pred_score_1, pred_score_2 = self._evaluate_one_hot_2(token_seq[None])
            pred_score_1, pred_score_2 = pred_score_1.item(), pred_score_2.item()
            
            # construct Gibbs sampler
            sampler = self._gibbs_sampler(seq_one_hot[None], initial=initial)
            seq_pairs = []
            total_num_proposals = 0
            all_proposed_mutants = []
            all_accepted_mutants = []
            
            # sample mutants
            proposed_mutants = sampler()
            num_proposals = proposed_mutants.shape[0]
            total_num_proposals += num_proposals
            proposed_num_edits = self.compute_mutant_stats(token_seq, proposed_mutants)
            proposed_mutants = proposed_mutants[proposed_num_edits > 0]
            all_proposed_mutants.append(to_np(proposed_mutants))
            
            # run gibbs generation of pairs
            sample_outputs, acepted_mutants = self._evaluate_mutants(
                mutants=proposed_mutants,
                score_1=pred_score_1,
                score_2=pred_score_2,
                source_one_hot=seq_one_hot,
                initial=initial,
            )
            
            all_accepted_mutants.append(to_np(acepted_mutants))
            sample_outputs['source_sequences'] = real_seq
            sample_outputs['source_scores_1'] = pred_score_1
            sample_outputs['source_scores_2'] = pred_score_2

            seq_pairs.append(sample_outputs)
            num_samples = len(sample_outputs)
            pbar.set_postfix_str(f'Accepted: {num_samples}/{num_proposals} ({num_samples / num_proposals:.2f})')
            
            if len(seq_pairs) > 0:
                seq_pairs = pd.concat(seq_pairs).drop_duplicates(
                    subset=['source_sequences', 'mutant_sequences'],
                    ignore_index=True,
                )
                all_mutant_pairs.append(seq_pairs)
            
            # showing some info
            # elapsed_time = time.time() - start_time
            # num_new_pairs = len(seq_pairs)
            # all_proposed_mutants = np.concatenate(all_proposed_mutants, axis=0)
            # proposed_res_freq = np.mean(
            #     all_proposed_mutants != to_np(token_seq)[None], axis=0
            # ).round(decimals=2)
            # n_proposed = all_proposed_mutants.shape[0]

            # all_accepted_mutants = np.concatenate(all_accepted_mutants, axis=0)
            # accepted_res_freq = np.mean(
            #     all_accepted_mutants != to_np(token_seq)[None], axis=0).round(decimals=2)
            # n_accepted = all_accepted_mutants.shape[0]

            # self.logger.info(
            #     f'Done with sequence {i+1}/{total_num_seqs} in {elapsed_time:.1f}s. '
            #     f'Accepted {num_new_pairs}/{total_num_proposals} ({num_new_pairs/total_num_proposals:.2f}) sequences. \n'
                # f'Proposed sites (n={n_proposed}): {proposed_res_freq}. \n'
                # f'Accepted sites (n={n_accepted}): {accepted_res_freq}.'
            # )
            
        if len(all_mutant_pairs) == 0:
            return None
        return pd.concat(all_mutant_pairs).drop_duplicates(
            subset=['source_sequences', 'mutant_sequences'],
            ignore_index=True
        )


def pareto_sampling(args, config, logger, output_dir):
    # setup dataset
    dataset = SampleSequenceDataset(csv_path=config.csv_path, seq_col='sequence', obj_1=config.obj_1, obj_2=config.obj_2, subsamples=config.subsamples)
    logger.info('Set up dataset')
    
    all_sample_results = []
    sampler = ParetoSampler(config, args)
    for epoch in range(config.max_epochs):
        if len(dataset) == 0:
            logger.info('Dataset is empty')
            break
        epoch_start_time = time.time()
        batch_size = len(dataset)
        dataloader = DataLoader(dataset, batch_size=batch_size)
                
        outputs = []
        for batch in dataloader:
            outputs.append(sampler(batch, initial=epoch < config.initial_epochs))
        
        candidate_seqs = []
        for new_pairs in outputs:
            if new_pairs is None:
                continue
            candidate_seqs.extend(new_pairs['mutant_sequences'].tolist())
            new_pairs['epoch'] = epoch + 1
            all_sample_results.append(new_pairs)
        candidate_seqs = list(set(candidate_seqs))
        epoch_elapsed_time = time.time() - epoch_start_time

        logger.info(f"Epoch {epoch} finished in {epoch_elapsed_time:.2f} seconds")
        logger.info("------------------------------------")
        
        dataset.clear()
        if epoch < config.max_epochs and len(candidate_seqs) > 0:
            dataset.add(candidate_seqs)
            dataset.random_sample(n=config.cluster_cutoff)
        logger.info(f"Next dataset = {len(dataset)} sequences")
    all_sample_results = pd.concat(all_sample_results)
    os.makedirs(output_dir, exist_ok=True)
    sample_save_path = os.path.join(output_dir, f'seed_{config.seed}.csv')
    all_sample_results.to_csv(sample_save_path, index=False)
    logger.info(f'{len(all_sample_results)} samples written to {sample_save_path}')

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('config', type=str)
    parser.add_argument('--logdir', type=str, default='logs_pref_vec_debug')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--pref_index', type=int, default=None)

    args = parser.parse_args()
    
    return args

def main():
    start_overall = time.time()
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    args = get_args()
    # Load configs
    config = common.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    config.seed = args.seed if args.seed is not None else config.seed
    config.pref_index = args.pref_index if args.pref_index is not None else config.pref_index
    config.device = args.device
    common.seed_all(config.seed if args.seed is None else args.seed)
    
    # Logging
    log_dir = common.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
    logger = common.get_logger('GWG', log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    
    logger.info(f'predictor_1_dir: {config.predictor_1_dir}')
    logger.info(f'predictor_2_dir: {config.predictor_2_dir}')
    
    # Set-up output path
    output_dir = os.path.join(log_dir, 'samples_' + now)
    
    # pareto sampling
    pareto_sampling(args, config, logger, output_dir)
    
    elapsed_time = time.time() - start_overall
    logger.info(f'Finished generation in {elapsed_time:.2f} seconds.')
    
if __name__ == '__main__':
    main()

