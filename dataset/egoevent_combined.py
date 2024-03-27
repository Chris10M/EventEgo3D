import torch
import random
import cv2
import tempfile
import copy
import numpy as np
from rich import print
from pathlib import Path
from torch.utils.data import Dataset
from collections import OrderedDict

from dataset.Joints3DDataset import Joints3DDataset
from dataset.representation import EROS, LNES, EventFrame
from dataset.egoevent_synthetic import SyntheticEventStream
from dataset.egoevent_real import RealEventStream
from dataset.dataset_utils import generate_path_split, generate_indices
from dataset.egoevent import SingleSequenceDataset


class CombinedDataset(Dataset):
    def __init__(self, cfg, dataset_root, split):
        super().__init__()
        
        dataset_root = Path(dataset_root)
        
        generate_path_split(dataset_root, cfg)
        assert split in ['train', 'val', 'test']

        if split == 'train':
            split_path = dataset_root / 'train.txt'
        elif split == 'val':
            split_path = dataset_root / 'val.txt'
        elif split == 'test':
            split_path = dataset_root / 'test.txt'
                        
        with open(split_path, 'r') as f:
            self.items = f.read().splitlines()
    
        if split == 'train':
            is_train = True
        else:
            is_train = False

        datasets = list()
        for item in self.items:
            data_path = dataset_root / item
            dataset = SingleSequenceDataset(cfg, data_path, is_train)
            if dataset.isvalid():
                self.visualize = dataset.visualize
                datasets.append(dataset)

        self.datasets = datasets
        self.lengths = [len(dataset) for dataset in datasets]
        self.total_length = sum(self.lengths)

        self.data_path = dataset_root
        self.dataset_root = dataset_root

        print(f'Dataset root: {dataset_root}, split: {split}')
        print('Datasets: ')
        for dataset in datasets:
            print(dataset.data_path)
        print('Total number of events: ', self.total_length)

        self.indices = generate_indices(self.dataset_root, self.datasets)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        idx, kwargs = idx
        dataset_idx, sample_idx = self.indices[idx]   

        return self.datasets[dataset_idx][sample_idx, kwargs]
    
    
class CombinedEgoEvent(Dataset): 
    def __init__(self, cfg, split):
        super().__init__()

        if split == 'train':
            is_train = True
        else:
            is_train = False

        cfg = copy.deepcopy(cfg)

        datasets = list()
        for root, type, split_ratio in cfg.DATASET.ENSEMBLE_DATASETS:
            cfg.DATASET.TRAIN_TEST_SPLIT = split_ratio                  
            cfg.DATASET.TYPE = type
            dataset = CombinedDataset(cfg, root, split)
            self.visualize = dataset.visualize
        
            datasets.append(dataset)
                        
        self.is_train = is_train
        self.datasets = datasets

        self.dataset_root = Path(tempfile.mkdtemp('egoevent_combined'))
        self.indices = generate_indices(self.dataset_root, self.datasets, make_equal_len=True)
        
        self.total_len = len(self.indices)
        print(f"Combined dataset size: {self.total_len}")
        
    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):            
        if isinstance(idx, tuple):
            idx, kwargs = idx
        else:
            kwargs = {}
        
        dataset_idx, sample_idx = self.indices[idx]   
                
        data, meta = self.datasets[dataset_idx][sample_idx, kwargs]
        
        # meta.pop('coord_x')
        # meta.pop('coord_y')
        # meta.pop('segmentation_indices')

        return data, meta
        
    @classmethod
    def evaluate_joints(self, *args, **kwargs):
        return Joints3DDataset.evaluate_joints(*args, **kwargs)

    @classmethod    
    def evaluate_dataset(cls, *args, **kwargs):
        return Joints3DDataset.evaluate_dataset(*args, **kwargs)