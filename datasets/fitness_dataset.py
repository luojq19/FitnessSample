import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
import random

AA_LIST = list("ARNDCQEGHILKMFPSTWYV")
aa2idx = {aa: idx for idx, aa in enumerate(AA_LIST)}
worst_fitness = {'stability': 68.6938,
                 'GFP': 0.0,
                 'gb1': 0.0,
                 'ddg': 116.583,
                 'fit_E3': 0.0,
                 'fit_E2': 0.0,}

def seq2indices(seq):
    return torch.tensor([aa2idx[aa] for aa in seq])

def indices2seq(indices):
    return ''.join([AA_LIST[int(idx)] for idx in indices])

class FitnessDataset(Dataset):
    def __init__(self, data_file, label_column):
        self.label_column = label_column
        self.df = pd.read_csv(data_file)[['sequence', label_column]]
        self.data = self.df.values.tolist()
        for i in tqdm(range(len(self.data)), desc='processing data'):
            seq, label = self.data[i]
            seq = torch.tensor([aa2idx[aa] for aa in seq])
            label = torch.tensor(label)
            self.data[i] = (seq, label)
        
    def __getitem__(self, index):
        seq, label = self.data[index]
        return seq, label
        
    def __len__(self):
        return len(self.data)
    
    def split_train_test_by_site(self, ref_seq_fasta, train_ratio=0.7, test_ratio=0.3):
        records = list(SeqIO.parse(ref_seq_fasta, 'fasta'))
        ref_seq = str(records[0].seq)
        all_indices = list(range(len(self.data)))
        all_seqs = self.df['sequence'].tolist()
        all_mut_site_indices = []
        mut_sites_set = []
        for seq in all_seqs:
            mut_site_indices = [i for i, aa in enumerate(seq) if aa!= ref_seq[i]]
            all_mut_site_indices.append(mut_site_indices)
            mut_sites_set.extend(mut_site_indices)
        mut_sites_set = list(set(mut_sites_set))
        print(f'Total {len(mut_sites_set)} mutated sites')
        num_mutation = [len(sites) for sites in all_mut_site_indices]
        print(f'num of mutated site range: {min(num_mutation)}, {max(num_mutation)}')
        random.shuffle(mut_sites_set)
        train_mut_site_indices = set(mut_sites_set[:int(train_ratio*len(mut_sites_set))])
        test_mut_site_indices = set(mut_sites_set[int(train_ratio*len(mut_sites_set)):])
        print(f'Train on {len(train_mut_site_indices)} mutated sites')
        print(f'Test on {len(test_mut_site_indices)} mutated sites')
        train_indices, test_indices = [], []
        for i in range(len(all_mut_site_indices)):
            mut_sites = set(all_mut_site_indices[i])
            if mut_sites.issubset(test_mut_site_indices):
                test_indices.append(i)
            else:
                train_indices.append(i)
        print(f'Train indices: {len(train_indices)}')
        print(f'Test indices: {len(test_indices)}')
        
        return train_indices, test_indices

    def augment_with_negative_sample(self, task='stability', num_agument=None):
        augment_values = worst_fitness[task]
        num_agument = len(self.data) if num_agument is None else num_agument
        seq_len = len(self.data[0][0])
        print(f'num of augment: {num_agument}')
        print(f'seq_len: {seq_len}, augment_values: {augment_values}')
        for i in tqdm(range(num_agument), desc='augmenting data'):
            seq = ''.join(random.choices(AA_LIST, k=seq_len))
            seq = torch.tensor([aa2idx[aa] for aa in seq])
            label = torch.tensor(augment_values)
            self.data.append((seq, label))
        
    
if __name__ == "__main__":
    # print(AA_LIST)
    # print(aa2idx)
    ds = FitnessDataset('../data/stability/stability_foldx_deduplicate_af2.csv')
    print(len(ds))
    # ds.augment_with_negative_sample()
    print(len(ds))
    # for seq, label in ds:
    #     print(seq, label, seq.shape, label.shape)
    #     break
    dl = DataLoader(ds, batch_size=10, shuffle=False)
    # for i, (seq, label) in enumerate(dl):
    #     print(seq, label, seq.shape, label.shape)
    #     break
    
    ds.split_train_test_by_site('../data/foldx/GFP_reference_seq_aa.fasta', train_ratio=0.4, test_ratio=0.6)
    