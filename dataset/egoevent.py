import platform
import copy
import numpy as np
from rich import print
from pathlib import Path
from torch.utils.data import Dataset

from dataset.Joints3DDataset import Joints3DDataset
from dataset.representation import EROS, LNES, EventFrame
from dataset.egoevent_synthetic import SyntheticEventStream
from dataset.egoevent_real import RealEventStream
from dataset.dataset_utils import generate_path_split, generate_indices


def get_representation(cfg, width, height):
    representation = cfg.DATASET.REPRESENTATION
    
    if representation == 'EROS':
        eros_config = cfg.DATASET.EROS    
        repr = EROS(eros_config.kernel_size, height, width, eros_config.DECAY_BASE)
    elif representation == 'LNES':
        repr = LNES(cfg, height, width)
    elif representation == 'EventFrame':
        repr = EventFrame(cfg, height, width)
    else:
        raise NotImplementedError

    return repr


class SingleSequenceDataset(Joints3DDataset):    
    def prepare_anno(self, item):
        ego_to_global_space = item['ego_to_global_space']
        j3d = item['ego_j3d']
        j2d = item['ego_j2d']
        segmentation_mask = item['segmentation_mask']
        valid_seg = item['valid_seg']
        rgb_frame_index = item['rgb_frame_index']

        if j3d is None or j2d is None:
            vis_j2d = np.zeros((self.num_joints, 2))
            vis_j3d = np.zeros((self.num_joints, 3))
            j3d = np.ones((self.num_joints, 3)) * -1
            j2d = np.ones((self.num_joints, 2)) * -1
        else:
            vis_j2d = np.ones_like(j2d)
            vis_j3d = np.ones_like(j3d)

        if ego_to_global_space is None:
            ego_to_global_space = np.eye(4)

        return {
                'valid_seg': valid_seg,
                'j2d': j2d.astype(np.float32),
                'j3d': j3d.astype(np.float32),	
                'vis_j2d': vis_j2d.astype(np.float32),
                'vis_j3d': vis_j3d.astype(np.float32),
                'segmentation_mask': segmentation_mask,
                'ego_to_global_space': ego_to_global_space,
                'rgb_frame_index': int(rgb_frame_index)
            }
    
    def __init__(self, cfg, data_path, is_train):
        super().__init__(cfg, data_path, is_train)

        if cfg.DATASET.TYPE == 'Synthetic':
            dataset = SyntheticEventStream(data_path, cfg, is_train)
        else:
            dataset = RealEventStream(data_path, cfg, is_train)
        
        self.data_path = data_path

        self.dataset = dataset
        width, height = dataset.width, dataset.height
        self.num_joints = cfg.NUM_JOINTS

        self.width = width
        self.height = height

        self.repr = get_representation(cfg, width, height)

        self.visualize = self.repr.visualize
        print(f'{data_path} => load {len(self.dataset)} samples')

        self.is_train = is_train

    def isvalid(self):
        return len(self.dataset) > 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            idx, kwargs = idx
        else:
            kwargs = {}
                
        data_batch = self.dataset[idx]
                
        if len(data_batch) == 0:
            raise StopIteration

        if self.is_train:
            dist = np.random.uniform(0.2, 1.0)
            n_events = data_batch.shape[0]
            choice_len = np.random.randint(int(n_events * dist), n_events)
            choices = np.random.choice(np.arange(n_events), choice_len, replace=False)
            data_batch = data_batch[choices, :]

        data = self.repr(data_batch)
        frame_index = data['frame_index']

        anno = self.dataset.get_annoation(frame_index)
        anno = self.prepare_anno(anno)

        return self.transform(data, anno, kwargs)


class EgoEvent(Dataset): 
    def __init__(self, cfg, split, finetune=False):
        super().__init__()

        cfg = copy.deepcopy(cfg)
    
        if finetune:
            cfg.DATASET.TYPE = 'Real'
            
        if cfg.DATASET.TYPE == 'Synthetic':
            dataset_root = Path(cfg.DATASET.SYN_ROOT)
        else:
            dataset_root = Path(cfg.DATASET.REAL_ROOT)
        
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

        self.is_train = is_train
    
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

        self.dataset_root = dataset_root
        
        print(f'Dataset root: {dataset_root}, split: {split}, finetune: {finetune}')
        print('Datasets: ')
        for dataset in datasets:
            print(dataset.data_path)
        print('Total number of events: ', self.total_length)

        self.indices = generate_indices(self.dataset_root, self.datasets)
    
    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):    
        if isinstance(idx, tuple):
            idx, kwargs = idx
        else:
            kwargs = {}
        
        dataset_idx, sample_idx = self.indices[idx]   
                
        data, meta = self.datasets[dataset_idx][sample_idx, kwargs]
        
        return data, meta

    def evaluate_joints(self, *args, **kwargs):
        return Joints3DDataset.evaluate_joints(*args, **kwargs)

    @classmethod    
    def evaluate_dataset(cls, *args, **kwargs):
        return Joints3DDataset.evaluate_dataset(*args, **kwargs)