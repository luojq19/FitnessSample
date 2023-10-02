import sys
sys.path.append('.')
import torch
import os
from utils import common
from models import BaseCNN
from utils.sample import load_seqs_from_csv

device = 'cuda:0'

def load_model(model_dir, device):
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            if file.endswith('.yml'):
                oracle_config = common.load_config(os.path.join(root, file))
                break
    model = globals()[oracle_config.model.model_type](oracle_config.model)
    ckpt = torch.load(os.path.join(model_dir, 'checkpoints/best_checkpoints.pt'))
    model.load_state_dict(ckpt)
    model.eval()
    model.to(device)
    print(f'Loaded model from {model_dir}')
    
    return model

def run_predictor(predictor, x, device):
    batch_size = 128
    x_batches = torch.split(x, batch_size)
    outputs = []
    with torch.no_grad():
        for x_batch in x_batches:
            x_batch = x_batch.to(device)
            outputs.append(predictor(x_batch))
    outputs = torch.cat(outputs)
    
    return outputs

def choose_scale(model_dir_1, model_dir_2, csv_path):
    model1 = load_model(model_dir_1, device)
    model2 = load_model(model_dir_2, device)
    seqs = load_seqs_from_csv(csv_path)
    outputs_1 = run_predictor(model1, seqs, device)
    outputs_2 = run_predictor(model2, seqs, device)
    print(outputs_1.shape, outputs_2.shape)
    stats_1 = {'mean': outputs_1.mean().item(), 'median': outputs_1.median().item(), 'max': outputs_1.max().item(), 'min': outputs_1.min().item(), '75percentile': outputs_1.quantile(0.75).item(), '25percentile': outputs_1.quantile(0.25).item()}
    stats_2 = {'mean': outputs_2.mean().item(), 'median': outputs_2.median().item(), 'max': outputs_2.max().item(), 'min': outputs_2.min().item(), '75percentile': outputs_2.quantile(0.75).item(), '25percentile': outputs_2.quantile(0.25).item()}
    print('stats_1:', stats_1)
    print('stats_2:', stats_2)
    ratio = {k: stats_2[k] / stats_1[k] for k in stats_1}
    print('ratio:', ratio)

if __name__ == '__main__':
    choose_scale('logs_new/train_predictor_GFP_0.2_0.4_2023_09_30__10_42_45',
                 'logs_new/train_predictor_stability_0.2_0.4_2023_09_30__10_43_49',
                 'data/GFP_stability_percentile_0.2_0.4.csv')
    
