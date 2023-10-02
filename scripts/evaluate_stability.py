import sys
sys.path.append('.')
import os
from tqdm import tqdm
from joblib import Parallel, delayed
from Bio import SeqIO
import argparse, datetime, json, time
import pandas as pd
import numpy as np
from utils.common import sec2min_sec

now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

def prepare_foldx_data(sample_path, tmp_dir, ref_seq_fasta, topk=128):
    df = pd.read_csv(sample_path)
    df = df.drop_duplicates(subset='mutant_sequences', ignore_index=True)
    df = df.sort_values('mutant_scores', ascending=False)
    sampled_seqs = df.mutant_sequences.tolist()[:topk]
    print((f'Sampled {len(set(sampled_seqs))} unique sequences.'))
    records = list(SeqIO.parse(ref_seq_fasta, 'fasta'))
    ref_seq = str(records[0].seq)
    for s in sampled_seqs:
        assert len(s) == len(ref_seq), f'{len(s)}!= {len(ref_seq)}'
    individual_list_dir = os.path.join(tmp_dir, 'individual_list')
    os.makedirs(individual_list_dir, exist_ok=True)
    mut_str_list = []
    for i in range(len(sampled_seqs)):
        assert len(sampled_seqs[i]) == len(ref_seq)
        mut_str = []
        for k in range(len(sampled_seqs[i])):
            if sampled_seqs[i][k] != ref_seq[k]:
                mut_str.append(f'{ref_seq[k]}A{k+1}{sampled_seqs[i][k]}')
        mut_str = ','.join(mut_str)
        mut_str_list.append(mut_str)
    print(f'mut_str: {mut_str_list[:10]}')
    assert len(mut_str_list) == len(sampled_seqs) == len(set(mut_str_list))
    with open(os.path.join(tmp_dir, 'mut_seqs.json'), 'w') as f:
        json.dump(mut_str_list, f)
    batch_size = 2
    num_batches = int(np.ceil(len(mut_str_list) / batch_size).item())
    print(f'num_batches: {num_batches}')
    for i in range(num_batches):
        start = i * batch_size
        end = min((i+1) * batch_size, len(mut_str_list))
        with open(os.path.join(individual_list_dir, f'individual_list_{i}.txt'), 'w') as f:
            for k in range(start, end):
                f.write(f'{mut_str_list[k]};\n')
    out_dir = os.path.join(tmp_dir, 'output')
    # input()
    return individual_list_dir, out_dir, num_batches, sampled_seqs

def foldx_runner(batch_idx, pdb_dir, pdb_file, mut_file, out_dir, num_runs=5):
    cmd = f'FoldX --command=BuildModel --pdb-dir={pdb_dir} --pdb={pdb_file} --output-dir={out_dir} --mutant-file={mut_file} --numberOfRuns={num_runs} --out-pdb=false --output-file=batch_{batch_idx} > logs_foldx/foldx_batch_{batch_idx}.log'
    os.system(cmd)

def run_foldx(pdb_dir, repaired_ref_pdb, individual_list_dir, out_dir, num_batches):
    start = time.time()
    os.makedirs(out_dir, exist_ok=True)
    mut_file_list = [os.path.join(individual_list_dir, f'individual_list_{i}.txt') for i in range(num_batches)]
    batches = [i for i in range(num_batches)]
    Parallel(n_jobs=64, verbose=10)(delayed(foldx_runner)(i, pdb_dir, repaired_ref_pdb, mut_file_list[i], out_dir) for i in tqdm(batches))
    end = time.time()
    print(f'FoldX running time: {sec2min_sec(end - start)}')
    
def collect_foldx_results(out_dir, num_batches, sampled_seqs, save_dir, args):
    ddg_list = []
    for i in range(num_batches):
        with open(os.path.join(out_dir, f'Average_batch_{i}_ref_seq_af2_Repair.fxout')) as f:
            lines = f.readlines()[9:]
            ddg_list.extend([float(l.split('\t')[2]) for l in lines])
    print('ddg:', len(ddg_list))
    assert len(ddg_list) == len(sampled_seqs)
    df = pd.DataFrame({
            'sequence': sampled_seqs,
            'ddg': ddg_list
        })
    df.to_csv(os.path.join(save_dir, 'foldx_results.csv'), index=False)
    
    gt = pd.read_csv(args.ground_truth)
    max_seen_ddg = np.max(gt.target).item()
    min_seen_ddg = np.min(gt.target).item()
    normalize = lambda x: (x - min_seen_ddg) / (max_seen_ddg - min_seen_ddg)
    normalized_ddg_list = [normalize(x) for x in ddg_list]
    
    with open(os.path.join(save_dir, 'stability_metrics.txt'), 'w') as f:
        f.write('num_unique,mean_fitness,median_fitness,std_fitness,max_fitness,min_fitness,source_path\n')
        f.write(f'{len(set(sampled_seqs))},{np.mean(ddg_list)},{np.median(ddg_list)},{np.std(ddg_list)},{np.max(ddg_list)},{np.min(ddg_list)},{args.sample}\n')
        f.write(f'{len(set(sampled_seqs))},{np.mean(normalized_ddg_list)},{np.median(normalized_ddg_list)},{np.std(normalized_ddg_list)},{np.max(normalized_ddg_list)},{np.min(normalized_ddg_list)},{args.sample}\n')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=str, required=True)
    parser.add_argument('--tmp_dir', type=str, default='tmp_' + now)
    parser.add_argument('--ref_seq_fasta', type=str, default='data/foldx/GFP_reference_seq_aa.fasta')
    parser.add_argument('--topk', type=int, default=128)
    parser.add_argument('--repaired_ref_pdb', type=str, default='data/foldx/outputs_af2/ref_seq_af2_Repair.pdb')
    parser.add_argument('--ground_truth', type=str, default='data/stability/stability_foldx_deduplicate_af2.csv')

    return parser.parse_args()

def main():
    args = get_args()
    os.makedirs(args.tmp_dir, exist_ok=True)
    individual_list_dir, out_dir, num_batches, sampled_seqs = prepare_foldx_data(args.sample, args.tmp_dir, args.ref_seq_fasta, args.topk)
    run_foldx(os.path.dirname(args.repaired_ref_pdb), os.path.basename(args.repaired_ref_pdb), individual_list_dir, out_dir, num_batches)
    collect_foldx_results(out_dir, num_batches, sampled_seqs, os.path.dirname(args.sample), args)

if __name__ == '__main__':
    main()
    
    

