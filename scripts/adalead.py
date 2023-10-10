import sys
sys.path.append('.')
import random, argparse, os, warnings, datetime, time, shutil
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
from datasets.fitness_dataset import AA_LIST, seq2indices, indices2seq
from utils import common
from utils.sample import load_oracle, run_predictor, load_seqs_from_csv
import torch
from tqdm import tqdm
from models import BaseCNN

torch.set_num_threads(1)

def generate_random_mutant(sequence: str, mu: float, alphabet: str, mutation_sites=None) -> str:
    """
    Generate a mutant of `sequence` where each residue mutates with probability `mu`.

    So the expected value of the total number of mutations is `len(sequence) * mu`.

    Args:
        sequence: Sequence that will be mutated from.
        mu: Probability of mutation per residue.
        alphabet: Alphabet string.

    Returns:
        Mutant sequence string.

    """
    mutant = []
    if mutation_sites is None:
        for s in sequence:
            if random.random() < mu:
                mutant.append(random.choice(alphabet))
            else:
                mutant.append(s)
        return "".join(mutant)
    else:
        for i, s in enumerate(sequence):
            if i in mutation_sites:
                if random.random() < mu:
                    mutant.append(random.choice(alphabet))
                else:
                    mutant.append(s)
            else:
                mutant.append(s)
        return "".join(mutant)

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, seqs, labels):
        super().__init__()
        self.data = [[seq2indices(seq), torch.tensor(label)] for seq, label in zip(seqs, labels)]
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

class LandscapeWrapper:
    def __init__(self, model, args) -> None:
        self.model = model
        self.cost = 0
        self.device = args.device
        self.model.to(args.device)
    
    def get_fitness(self, seqs):
        self.cost += len(seqs)
        seqs_tensor = torch.vstack([seq2indices(seq) for seq in seqs]).to(self.device)
        batch_size = 128

        x_batches = torch.split(seqs_tensor, batch_size, 0)
        outputs = []
        with torch.no_grad():
            for x_batch in x_batches:
                x_batch = x_batch.to(args.device)
                outputs.append(self.model(x_batch))

        outputs = torch.cat(outputs).tolist()
        
        return outputs
    
    def train(self, seqs, labels):
        self.model.train()
        self.model.to(self.device)
        dataset = SimpleDataset(seqs, labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
        num_epochs = 1000
        n_bad = 0
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        best_loss = 1e9
        early_stop = 15
        for epoch in tqdm(range(num_epochs)):
            for data, label in dataloader:
                data, label = data.to(self.device), label.to(self.device)
                output = self.model(data)
                loss = criterion(output, label).squeeze(-1)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                n_bad = 0
            else:
                n_bad += 1
                if n_bad >= early_stop: 
                    break
        self.model.eval()

class Adalead:
    def __init__(self,
                 model,
                 rounds: int,
                 sequences_batch_size: int,
                 model_queries_per_batch: int,
                 base_pool_path: str,
                 alphabet: str,
                 mu: int = 1,
                 recomb_rate: float = 0,
                 threshold: float = 0.05,
                 rho: int = 0,
                 eval_batch_size: int = 20,
                 logger=None,
                 config=None):
        self.model = model
        self.rounds = rounds
        self.sequences_batch_size = sequences_batch_size
        self.model_queries_per_batch = model_queries_per_batch
        self.starting_sequences = pd.read_csv(base_pool_path)['sequence'].tolist()
        self.alphabet = alphabet
        self.mu = mu
        self.recomb_rate = recomb_rate
        self.threshold = threshold
        self.rho = rho
        self.eval_batch_size = eval_batch_size
        self.logger = logger
        if hasattr(config, 'mutation_sites'):
            self.mutation_sites = config.mutation_sites
            print(f'Restrict mutation sites {self.mutation_sites}')
        else:
            self.mutation_sites = None
    
    def _recombine_population(self, gen):
        # If only one member of population, can't do any recombining
        if len(gen) == 1:
            return gen

        random.shuffle(gen)
        ret = []
        for i in range(0, len(gen) - 1, 2):
            strA = []
            strB = []
            switch = False
            for ind in range(len(gen[i])):
                if random.random() < self.recomb_rate:
                    switch = not switch

                # putting together recombinants
                if switch:
                    strA.append(gen[i][ind])
                    strB.append(gen[i + 1][ind])
                else:
                    strB.append(gen[i][ind])
                    strA.append(gen[i + 1][ind])

            ret.append("".join(strA))
            ret.append("".join(strB))
        return ret

    def propose_sequences(
        self, measured_sequences: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Propose top `sequences_batch_size` sequences for evaluation."""
        measured_sequence_set = set(measured_sequences["sequence"])

        # Get all sequences within `self.threshold` percentile of the top_fitness
        top_fitness = measured_sequences["true_score"].max()
        top_inds = measured_sequences["true_score"] >= top_fitness * (
            1 - np.sign(top_fitness) * self.threshold
        )

        parents = np.resize(
            measured_sequences["sequence"][top_inds].to_numpy(),
            self.sequences_batch_size,
        )

        sequences = {}
        previous_model_cost = self.model.cost
        while self.model.cost - previous_model_cost < self.model_queries_per_batch:
            # generate recombinant mutants
            for i in range(self.rho):
                parents = self._recombine_population(parents)

            for i in range(0, len(parents), self.eval_batch_size):
                # Here we do rollouts from each parent (root of rollout tree)
                roots = parents[i : i + self.eval_batch_size]
                root_fitnesses = self.model.get_fitness(roots)

                nodes = list(enumerate(roots))

                while (
                    len(nodes) > 0
                    and self.model.cost - previous_model_cost + self.eval_batch_size
                    < self.model_queries_per_batch
                ):
                    child_idxs = []
                    children = []
                    while len(children) < len(nodes):
                        idx, node = nodes[len(children) - 1]

                        child = generate_random_mutant(
                            node,
                            self.mu * 1 / len(node),
                            self.alphabet,
                            self.mutation_sites,
                        )

                        # Stop when we generate new child that has never been seen
                        # before
                        if (
                            child not in measured_sequence_set
                            and child not in sequences
                        ):
                            child_idxs.append(idx)
                            children.append(child)

                    # Stop the rollout once the child has worse predicted
                    # fitness than the root of the rollout tree.
                    # Otherwise, set node = child and add child to the list
                    # of sequences to propose.
                    fitnesses = self.model.get_fitness(children)
                    sequences.update(zip(children, fitnesses))

                    nodes = []
                    for idx, child, fitness in zip(child_idxs, children, fitnesses):
                        if fitness >= root_fitnesses[idx]:
                            nodes.append((idx, child))

        if len(sequences) == 0:
            raise ValueError(
                "No sequences generated. If `model_queries_per_batch` is small, try "
                "making `eval_batch_size` smaller"
            )

        # We propose the top `self.sequences_batch_size` new sequences we have generated
        new_seqs = np.array(list(sequences.keys()))
        preds = np.array(list(sequences.values()))
        sorted_order = np.argsort(preds)[: -self.sequences_batch_size : -1]

        return new_seqs[sorted_order], preds[sorted_order]
    
    def run(
        self, oracle, verbose: bool, config, logger, save_path,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Run the exporer.

        Args:
            landscape: Ground truth fitness landscape.
            verbose: Whether to print output or not.

        """
        # self.model.cost = 0

        # Metadata about run that will be used for logging purposes
        # metadata = {
        #     "run_id": datetime.now().strftime("%H:%M:%S-%m/%d/%Y"),
        #     "exp_name": self.name,
        #     "model_name": self.model.name,
        #     "landscape_name": oracle.name,
        #     "rounds": self.rounds,
        #     "sequences_batch_size": self.sequences_batch_size,
        #     "model_queries_per_batch": self.model_queries_per_batch,
        # }

        # Initial sequences and their scores
        sequences_data = pd.DataFrame(
            {
                "sequence": self.starting_sequences,
                "model_score": [np.nan for i in range(len(self.starting_sequences))],
                "true_score": oracle.get_fitness(self.starting_sequences),
                "round": [0 for i in range(len(self.starting_sequences))],
                "model_cost": [self.model.cost for i in range(len(self.starting_sequences))],
                "measurement_cost": [1 for i in range(len(self.starting_sequences))],
            }
        )
        # self._log(sequences_data, metadata, 0, verbose, time.time())

        # For each round, train model on available data, propose sequences,
        # measure them on the true landscape, add to available data, and repeat.
        range_iterator = range if verbose else tqdm.trange
        for r in tqdm(range(1, self.rounds + 1), desc='adalead', dynamic_ncols=True):
            round_start_time = time.time()
            self.model.train(
                sequences_data["sequence"].tolist(),
                sequences_data["true_score"].tolist(),
            )

            seqs, preds = self.propose_sequences(sequences_data)
            true_score = oracle.get_fitness(seqs)

            if len(seqs) > self.sequences_batch_size:
                warnings.warn(
                    "Must propose <= `self.sequences_batch_size` sequences per round"
                )

            sequences_data = pd.concat([sequences_data, pd.DataFrame(
                {
                    "sequence": seqs,
                    "model_score": preds,
                    "true_score": true_score,
                    "round": r,
                    "model_cost": self.model.cost,
                    "measurement_cost": len(sequences_data) + len(seqs),
                }
            )])
            # logger.info(sequences_data, r, verbose, round_start_time)
            
        df = pd.DataFrame({'mutant_sequences': sequences_data['sequence'],
                           'mutant_scores': sequences_data['true_score']})
        df = df[len(self.starting_sequences):]
        logger.info(f'Saving {len(df)} sampled sequences to {save_path}')
        logger.info(f'Intersection with starting sequences: {len(set(df["mutant_sequences"].tolist()).intersection(set(self.starting_sequences)))}')
        df.to_csv(save_path, index=False)

        return sequences_data
    

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
    logger = common.get_logger('adalead', log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    
    oracle = LandscapeWrapper(load_oracle(config, args, logger), args)
    model = LandscapeWrapper(globals()[config.model.model_type](config.model), args)
    sampler = Adalead(model, config.rounds, config.sequences_batch_size, config.model_queries_per_batch, config.base_pool_path, AA_LIST, config.mu, config.recomb_rate, config.threshold, config.rho, config.eval_batch_size, logger, config)

    # Run the explorer
    logger.info("Running explorer...")
    sampler.run(oracle, verbose=True, config=config, logger=logger, 
                                           save_path=os.path.join(log_dir, f'samples_seed_{config.seed}.csv'))