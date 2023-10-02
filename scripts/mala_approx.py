import sys
sys.path.append('.')
import torch
import numpy as np
import random, argparse, datetime, logging, os, shutil
import pandas as pd
from tqdm import tqdm
# from ppde.base_sampler import BaseSampler
# from ppde.third_party.hsu import data_utils
# from ppde.metrics import n_hops
from datasets.fitness_dataset import aa2idx, AA_LIST, seq2indices, indices2seq
from models import BaseCNN
from utils import common
from utils.sample import load_oracle, run_predictor, load_seqs_from_csv, n_hops

class MALAApprox:
    """
    Treats proteins as samples from a one-hot Categorical distribution,
    hence uses a temperature-controlled relaxation of a Categorical
    to then apply MALA in.
    """
    def __init__(self, config):
        self.diffusion_relaxation_tau = config.diffusion_relaxation_tau
        self.diffusion_step_size = config.diffusion_step_size        
        self.alphabet_size = len(AA_LIST)
        self.task = config.task

    def straight_through_sample(self, x_relaxed_discrete_distribution):
        x_soft = x_relaxed_discrete_distribution.rsample()
        x_hard = torch.argmax(x_soft, -1)
        x_hard = torch.nn.functional.one_hot(x_hard, self.alphabet_size).to(x_hard.device).float()
        # [N,L,V]
        return (x_soft + x_hard) - x_soft

    def run(self, initial_population, num_steps, oracle, args, logger, save_path, min_pos=None, max_pos=None, log_every=50):
        n_chains = initial_population.shape[0]
        seq_len = initial_population.shape[1]
        x = initial_population
        min_pos = min_pos if min_pos is not None else 0
        max_pos = max_pos if max_pos is not None else seq_len
        preserve_left = x[:,:min_pos].clone()
        preserve_right = x[:,max_pos:].clone()
        x_center = x[:,min_pos:max_pos+1]

        random_idx = np.random.randint(0,n_chains)
        x_soft = (1 - self.diffusion_relaxation_tau) * ((1./self.alphabet_size) * torch.ones_like(x_center)) + self.diffusion_relaxation_tau * x_center
        x_relaxed_discrete = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(
                             torch.Tensor([self.diffusion_relaxation_tau]).to(x.device), probs=x_soft)  
        x_logits = x_relaxed_discrete.logits
        x_hard_differentiable = self.straight_through_sample(x_relaxed_discrete)
        
        x_hard_full = torch.cat([preserve_left, x_hard_differentiable, preserve_right],1)
        
        # convert population to List of tensors
        state_energy = run_predictor(oracle, x, args)
        state_seqs = torch.chunk(x.clone(), n_chains)
        energy_history = [state_energy.clone()]
        # print(f'state_energy shape: {state_energy.shape}')
        all_x = []
        
        for i in tqdm(range(num_steps), desc='MALA Approx', dynamic_ncols=True):

            x_logits = x_logits.requires_grad_()
            x_relaxed_discrete = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(
                                  torch.Tensor([self.diffusion_relaxation_tau]).to(x_hard_differentiable.device), logits=x_logits) 
   
            x_hard_differentiable = self.straight_through_sample(x_relaxed_discrete)

            x_hard_full = torch.cat([preserve_left, x_hard_differentiable, preserve_right],1)

            state_energy = run_predictor(oracle, x_hard_full, args)
            # print(state_energy, x_logits, state_energy.requires_grad, x_logits.requires_grad)
            # input()
            x_grad = torch.autograd.grad([state_energy.sum()], [x_logits])[0]

            # preserve_left = x_logits[:,:min_pos].clone()
            # preserve_right = x_logits[:,max_pos+1:].clone()

            x_logits = torch.distributions.normal.Normal(x_logits + (self.diffusion_step_size/2.) * x_grad,
                                                        self.diffusion_step_size ** 2).sample()
            
            # x_logits[:,:min_pos] = preserve_left
            # x_logits[:,max_pos+1:] = preserve_right
            
            all_x += [torch.argmax(x_hard_full.detach().cpu(), -1)]
        all_x = torch.vstack(all_x)
        # print(all_x.shape)
        oracle.enable_one_hot()
        fitness = run_predictor(oracle, all_x, args, no_grad=True)
        # print(f'Fitness shape: {fitness.shape}')
        if self.task == 'GFP':
            best_fitness, best_idxs = torch.sort(fitness, descending=True)
        elif self.task == 'stability':
            best_fitness, best_idxs = torch.sort(fitness, descending=False)
        else:
            logger.error(f'Unknown task {self.task}')
            raise NotImplementedError
        # print(best_idxs.shape)
        best_x = [all_x[best_idxs[i].item()] for i in range(n_chains * 10)]
        best_scores = [fitness[best_idxs[i]].item() for i in range(n_chains * 10)]
        best_seqs = [indices2seq(best_x[i]) for i in range(len(best_x))]
        df = pd.DataFrame({'mutant_sequences': best_seqs, 'mutant_scores': best_scores})
        logger.info(f'Saving {len(df)} sampled sequences to {save_path}')
        df.to_csv(save_path, index=False)
    
    def run_old(self, initial_population, num_steps, energy_function, min_pos, max_pos, oracle, log_every=50):
    
        seq_len = initial_population.size(1)
        n_chains = initial_population.size(0)
        x = initial_population
        preserve_left = x[:,:min_pos].clone()
        preserve_right = x[:,max_pos+1:].clone()
        x_center = x[:,min_pos:max_pos+1]

        random_idx = np.random.randint(0,n_chains)
        
        x_soft = (1 - self.diffusion_relaxation_tau) * ((1./self.alphabet_size) * torch.ones_like(x_center)) + self.diffusion_relaxation_tau * x_center
        x_relaxed_discrete = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(
                             torch.Tensor([self.diffusion_relaxation_tau]).to(x.device), probs=x_soft)    
        x_logits = x_relaxed_discrete.logits
        x_hard_differentiable = self.straight_through_sample(x_relaxed_discrete)

        x_hard_full = torch.cat([preserve_left, x_hard_differentiable, preserve_right],1)

        state_energy, fitness = energy_function.get_energy(x_hard_full)
        fitness_history = [fitness.detach()]
        energy_history = [state_energy.detach()]
        random_traj = [x_hard_full[random_idx].detach().cpu().numpy()]
        gt_fitness = oracle(x_hard_full).detach()
        all_x = [x_hard_full.detach().cpu().numpy()]

        fitness_quantiles = np.quantile(fitness_history[-1].cpu().numpy(), [0.5,0.9])
        gt_score_quantiles = np.quantile(gt_fitness.cpu().numpy(), [0.5, 0.9])
        energy_quantiles = np.quantile(energy_history[-1].cpu().numpy(), [0.5,0.9])
        mean_hops, std_hops = n_hops(x_hard_full, torch.from_numpy(data_utils.seqs_to_onehot(oracle.potts.wtseqs)[0]).to(x.device).float())

        print(f'[Iteration 0] energy: 50% {energy_quantiles[0]:.3f}, 90% {energy_quantiles[1]:.3f}')
        print(f'[Iteration 0] pred fitness 50% {fitness_quantiles[0]:.3f}, 90% {fitness_quantiles[1]:.3f}')
        print(f'[Iteration 0] oracle fitness 50% {gt_score_quantiles[0]:.3f}, 90% {gt_score_quantiles[1]:.3f}')
        print(f'[Iteration 0] mean hops = {mean_hops:.2f}, std hops = {std_hops:.2f}')

        print('')

        for i in range(num_steps):

            x_logits = x_logits.requires_grad_()
            x_relaxed_discrete = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(
                                  torch.Tensor([self.diffusion_relaxation_tau]).to(x_hard_differentiable.device), logits=x_logits) 
   
            x_hard_differentiable = self.straight_through_sample(x_relaxed_discrete)

            x_hard_full = torch.cat([preserve_left, x_hard_differentiable, preserve_right],1)

            state_energy, fitness = energy_function.get_energy(x_hard_full)
            
            x_grad = torch.autograd.grad([state_energy.sum()], [x_logits])[0]

            # preserve_left = x_logits[:,:min_pos].clone()
            # preserve_right = x_logits[:,max_pos+1:].clone()

            x_logits = torch.distributions.normal.Normal(x_logits + (self.diffusion_step_size/2.) * x_grad,
                                                        self.diffusion_step_size ** 2).sample()
            
            # x_logits[:,:min_pos] = preserve_left
            # x_logits[:,max_pos+1:] = preserve_right
            fitness = fitness.detach()
            
            random_traj += [x_hard_full[random_idx].detach().cpu().numpy()]
            energy_history += [state_energy.detach()]
            fitness_history.append(fitness)
            all_x += [x_hard_full.detach().cpu().numpy()]

            if i > 0 and (i+1) % log_every == 0:
                gt_fitness = oracle(x_hard_full).detach()

                energy_quantiles = np.quantile(energy_history[-1].cpu().numpy(), [0.5,0.9])
                fitness_quantiles = np.quantile(fitness.cpu().numpy(), [0.5,0.9])
                gt_score_quantiles = np.quantile(gt_fitness.cpu().numpy(), [0.5, 0.9])

                mean_hops, std_hops = n_hops(x_hard_full, torch.from_numpy(data_utils.seqs_to_onehot(oracle.potts.wtseqs)[0]).to(x.device).float())

                print(f'[Iteration {i}] energy: 50% {energy_quantiles[0]:.3f}, 90% {energy_quantiles[1]:.3f}')
                print(f'[Iteration {i}] pred fitness 50% {fitness_quantiles[0]:.3f}, 90% {fitness_quantiles[1]:.3f}')
                print(f'[Iteration {i}] oracle fitness 50% {gt_score_quantiles[0]:.3f}, 90% {gt_score_quantiles[1]:.3f}')
                print(f'[Iteration {i}] mean hops = {mean_hops:.2f}, std hops = {std_hops:.2f}')
                print('')
        
        energy_history = torch.stack(energy_history)  # [num_steps,num_chains]
        best_energy, best_idxs = torch.max(energy_history,0)

        all_x = np.stack(all_x,0)

        #all_x = torch.stack(all_x,0)  # [num_steps,...]
        best_x = np.stack([all_x[best_idxs[i],i] for i in range(n_chains)],0)
        fitness_history = torch.stack(fitness_history, 0)
        best_fitness = torch.stack([fitness_history[best_idxs[i],i] for i in range(n_chains)],0)
        # best predicted samples - torch.Tensor
        # best predicted energy - numpy
        # best predicted fitness - numpy
        # all predicted energy
        # all predicted fitness
        # random_traj
        return torch.from_numpy(best_x).to(initial_population.device), best_energy.cpu().numpy(), best_fitness.cpu().numpy(), \
            energy_history.cpu().numpy(), fitness_history.cpu().numpy(), random_traj

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--logdir', type=str, default='./logs_baseline_new')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda:0')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    config = common.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    common.seed_all(config.seed)
    log_dir = common.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
    logger = common.get_logger('mala_approx', log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    
    oracle = load_oracle(config, args, logger)
    oracle.disable_one_hot()
    sampler = MALAApprox(config)

    initial_seqs = load_seqs_from_csv(csv_path=config.base_pool_path)
    initial_seqs = torch.nn.functional.one_hot(initial_seqs, len(AA_LIST))
    logger.info(initial_seqs.shape)
    logger.info(f'Loaded {len(initial_seqs)} initial sequences from {config.base_pool_path}')

    sampler.run(initial_population=initial_seqs,
                num_steps=config.num_steps,
                oracle=oracle,
                args=args,
                logger=logger,
                save_path=os.path.join(log_dir, f'samples_seed_{config.seed}.csv'))
