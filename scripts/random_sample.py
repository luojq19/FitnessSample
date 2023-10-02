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

now = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
# logger = logging.getLogger(__name__)

class RandomSampler:
    def __init__(self, config):
        # self.T_max = args.simulated_annealing_temp
        # self.T = self.T_max
        self.muts_per_seq_param = config.muts_per_seq_param # 1.5
        # self.decay_rate = args.decay_rate
        self.AA_idxs = [i for i in range(len(AA_LIST))]
        self.task = config.task
    
    def make_n_random_edits(self, seq, nedits, min_pos=None, max_pos=None):
        seq = seq.squeeze()
        if min_pos is None:
            min_pos = 0
        
        if max_pos is None:
            max_pos = seq.size(0)
        # print(min_pos, max_pos)
        # Create non-redundant list of positions to mutate.
        l = list(range(min_pos,max_pos))
        nedits = min(len(l), nedits)
        random.shuffle(l)
        # pick n random positions
        pos_to_mutate = l[:nedits]    
        
        for i in range(nedits):
            # random mutation
            pos = pos_to_mutate[i]
            # print(seq.shape)
            # print(seq[pos].shape)
            # input()
            cur_AA = seq[pos].item()
            # cur_AA = torch.argmax(seq[pos]).item()
            candidates = list(set(self.AA_idxs) - set([cur_AA]))
            seq[pos] = torch.tensor(np.random.choice(candidates)).to(seq.dtype).to(seq.device)
            # seq[pos][cur_AA] = 0
            # seq[pos][np.random.choice(candidates)] = 1
            
        return seq.unsqueeze(0)

    def propose_seqs(self,seqs, mu_muts_per_seq, min_pos=None, max_pos=None):
        mseqs = []
        for i,s in enumerate(seqs):
            n_edits = torch.poisson(mu_muts_per_seq[i]-1) + 1
            mseqs.append(self.make_n_random_edits(s, n_edits.int().item(), min_pos, max_pos)) 
        return mseqs

    def run(self, initial_population, num_steps, oracle, args, logger, save_path, min_pos=None, max_pos=None, log_every=50):
        with torch.no_grad():
            n_chains = initial_population.shape[0]
            seq_len = initial_population.shape[1]
            x = initial_population
            random_idx = np.random.randint(0, n_chains)
            mu_muts_per_seq = torch.from_numpy(self.muts_per_seq_param * np.random.rand(n_chains) + 1)
            all_x = []
            
            # convert population to List of tensors
            state_seqs = torch.chunk(x.clone(), n_chains)
            
            for i in tqdm(range(num_steps), desc='Random sampling'):
                proposal_seqs = self.propose_seqs(state_seqs, mu_muts_per_seq, min_pos, max_pos)
                x_proposal = torch.cat(proposal_seqs)
                
                x_new = x_proposal
                
                # resets
                state_seqs = torch.chunk(x.clone(), n_chains)
                # print(x_new.shape)
                all_x.append(x_new)
            all_x = torch.vstack(all_x)
            # print(all_x.shape)
            fitness = run_predictor(oracle, all_x, args)
            # print(f'Fitness shape: {fitness.shape}')
            if self.task == 'GFP':
                best_fitness, best_idxs = torch.sort(fitness, descending=True)
            elif self.task == 'stability':
                best_fitness, best_idxs = torch.sort(fitness, descending=False)
            else:
                logger.error(f'Unknown task {self.task}')
                raise NotImplementedError
            # print(best_idxs.shape)
            best_x = [all_x[best_idxs[i].item()] for i in range(n_chains)]
            best_scores = [fitness[best_idxs[i]].item() for i in range(n_chains)]
            best_seqs = [indices2seq(best_x[i]) for i in range(len(best_x))]
            df = pd.DataFrame({'mutant_sequences': best_seqs, 'mutant_scores': best_scores})
            logger.info(f'Saving {len(df)} sampled sequences to {save_path}')
            df.to_csv(save_path, index=False)

    def run_old(self, initial_population, num_steps, energy_function, min_pos, max_pos, oracle, log_every=50):

        with torch.no_grad():
            n_chains = initial_population.size(0)
            seq_len = initial_population.size(1)
            x = initial_population
            random_idx = np.random.randint(0,n_chains)
            mu_muts_per_seq = torch.from_numpy(self.muts_per_seq_param * np.random.rand(n_chains) + 1)

            # convert population to List of tensors
            state_energy, fitness = energy_function.get_energy(x)
            state_seqs = torch.chunk(x.clone(),n_chains)
            #seq_history = [state_seqs]
            fitness_history = [fitness]
            energy_history = [state_energy.clone()]
            random_traj = [state_seqs[random_idx].cpu().numpy()]
            gt_fitness = oracle(x)
            all_x = [x.detach().cpu().numpy()]

            fitness_quantiles = np.quantile(fitness_history[-1].cpu().numpy(), [0.5,0.9])
            gt_score_quantiles = np.quantile(gt_fitness.cpu().numpy(), [0.5, 0.9])
            energy_quantiles = np.quantile(state_energy.cpu().numpy(), [0.5,0.9])

            print(f'[Iteration 0] energy: 50% {energy_quantiles[0]:.3f}, 90% {energy_quantiles[1]:.3f}')
            print(f'[Iteration 0] pred fitness 50% {fitness_quantiles[0]:.3f}, 90% {fitness_quantiles[1]:.3f}')
            print(f'[Iteration 0] oracle fitness 50% {gt_score_quantiles[0]:.3f}, 90% {gt_score_quantiles[1]:.3f}')
            print('')

            for i in range(num_steps):
                
                #current_energy = energy_history[-1]
               
                # random proposals
                proposal_seqs = self.propose_seqs(state_seqs, mu_muts_per_seq, min_pos, max_pos)
                x_proposal = torch.cat(proposal_seqs)           
                proposal_energy, proposal_fitness = energy_function.get_energy(x_proposal)

                x_new = x_proposal 
                
                # resets
                state_seqs = torch.chunk(x.clone(), n_chains)
                
                #aprob = aprob.view(n_chains)
               
                # replace -inf with 0
                proposal_energy[torch.isneginf(proposal_energy)] = 0
                proposal_fitness[torch.isneginf(proposal_fitness)] = 0

                state_energy = proposal_energy# * aprob + current_energy * (1. - aprob)
                energy_history += [state_energy.clone()]
                fitness = proposal_fitness #* aprob + fitness * (1. - aprob)
                fitness_history += [fitness.detach()]
                all_x += [x_new.detach().cpu().numpy()]

                random_traj += [state_seqs[random_idx].cpu().numpy()]
                # self.T = self.T_max * self.decay_rate**i
                
                if i > 0 and (i+1) % log_every == 0:
                    gt_fitness = oracle(x_new)
                    #fitness_history += [fitness]
                    energy_quantiles = np.quantile(state_energy.cpu().numpy(), [0.5,0.9])
                    fitness_quantiles = np.quantile(fitness.cpu().numpy(), [0.5,0.9])
                    gt_score_quantiles = np.quantile(gt_fitness.cpu().numpy(), [0.5, 0.9])
                    mean_hops, std_hops = n_hops(x_new, torch.from_numpy(data_utils.seqs_to_onehot(oracle.potts.wtseqs)[0]).to(x.device).float())

                    print(f'[Iteration {i}] energy: 50% {energy_quantiles[0]:.3f}, 90% {energy_quantiles[1]:.3f}')
                    print(f'[Iteration {i}] pred fitness 50% {fitness_quantiles[0]:.3f}, 90% {fitness_quantiles[1]:.3f}')
                    print(f'[Iteration {i}] oracle fitness 50% {gt_score_quantiles[0]:.3f}, 90% {gt_score_quantiles[1]:.3f}')
                    print(f'[Iteration {i}] mean hops = {mean_hops:.2f}, std hops = {std_hops:.2f}')
                    
                    print('',flush=True)

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

def load_oracle(config, args, logger):
    for root, dirs, files in os.walk(config.oracle_dir):
        for file in files:
            if file.endswith('.yml'):
                oracle_config = common.load_config(os.path.join(root, file))
                break
    oracle = globals()[oracle_config.model.model_type](oracle_config.model)
    ckpt = torch.load(os.path.join(config.oracle_dir, 'checkpoints/best_checkpoints.pt'))
    oracle.load_state_dict(ckpt)
    oracle.eval()
    oracle.to(args.device)
    logger.info(f'Loaded oracle from {config.oracle_dir}')
    
    return oracle

def run_predictor(predictor, x, args):
    batch_size = 128
    x_batches = torch.split(x, batch_size)
    outputs = []
    with torch.no_grad():
        for x_batch in x_batches:
            x_batch = x_batch.to(args.device)
            outputs.append(predictor(x_batch))
    outputs = torch.cat(outputs)
    
    return outputs

def load_seqs_from_csv(csv_path, seq_column='sequence'):
    df = pd.read_csv(csv_path)
    seqs = df[seq_column].tolist()
    seqs = [seq2indices(seq) for seq in seqs]
    seqs = torch.stack(seqs)
    
    return seqs

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
    logger = common.get_logger('random_sample', log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    
    oracle = load_oracle(config, args, logger)
    sampler = RandomSampler(config)

    initial_seqs = load_seqs_from_csv(config.base_pool_path)
    logger.info(initial_seqs.shape)
    logger.info(f'Loaded {len(initial_seqs)} initial sequences from {config.base_pool_path}')

    sampler.run(initial_population=initial_seqs,
                num_steps=config.num_steps,
                oracle=oracle,
                args=args,
                logger=logger,
                save_path=os.path.join(log_dir, f'samples_seed_{config.seed}.csv'))