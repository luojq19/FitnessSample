import sys
sys.path.append('.')

from typing import List, Optional, Tuple
from biotite.sequence.io import fasta
from polyleven import levenshtein
import numpy as np
import torch
import pyrootutils
import logging
import os
from omegaconf import DictConfig
import pandas as pd
from utils import common
from models import BaseCNN
from omegaconf import OmegaConf
from models.GWG_module import Encoder
import glob
from tqdm import tqdm
from datasets.fitness_dataset import AA_LIST

to_np = lambda x: x.cpu().detach().numpy()
to_list = lambda x: to_np(x).tolist()
alphabet = AA_LIST

def diversity(seqs):
    num_seqs = len(seqs)
    total_dist = 0
    for i in range(num_seqs):
        for j in range(num_seqs):
            x = seqs[i]
            y = seqs[j]
            if x == y:
                continue
            total_dist += levenshtein(x, y)
    return total_dist / (num_seqs*(num_seqs-1))

