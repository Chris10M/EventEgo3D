import sys; sys.path.append('../')
import json
import numpy as np
import json
import random
import logging
import cv2
import os

from dataset.dataset_utils import h5py_File

from pathlib import Path
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


class SyntheticEventStream(Dataset):
    def __init__(self, data_path, cfg, is_train):
        super().__init__()

        self.data_path = Path(data_path)
        self.cfg = cfg

        with open(self.data_path / 'event_meta.json', 'r') as f:
            meta = json.load(f)
            
        self.height = meta['height']
        self.width = meta['width']
        
        self.stream_path = self.data_path / 'events.h5'
        self.fin = None 

        self.batch_size = cfg.DATASET.EVENT_BATCH_SIZE
        self.max_frame_time = cfg.DATASET.SYNTHETIC.MAX_FRAME_TIME_IN_MS
        self.is_train = is_train

    def init_stream(self):
        self.fin = h5py_File(self.stream_path, 'r')['event']
        
    def __len__(self):    
        with h5py_File(self.stream_path, 'r') as f:
            return f['event'].shape[0] // self.batch_size
    
    def get_event_batch(self, idx, num_events):
        if self.is_train:
            max_frame_time = random.randint(2, self.max_frame_time)
        else:
            max_frame_time = self.max_frame_time

        frame_time = 0
        data_batches = []           
        while frame_time < max_frame_time: 
            data_batch = self.fin[idx: idx + num_events]        
            ts = data_batch[:, 2] 
            
            if not len(ts): 
                break

            ts = (ts[-1] - ts[0]) * 1e-3 # microseconds to milliseconds 

            data_batches.append(data_batch)
            
            frame_time += ts
            idx += num_events

        if len(data_batches) == 0:
            raise StopIteration
        
        data_batches_np = np.concatenate(data_batches, axis=0)
    
        del data_batches

        return data_batches_np
    
    def __getitem__(self, idx):
        if self.fin is None: self.init_stream() # Done to ensure multiprocessing works

        data_batch = self.get_event_batch(idx * self.batch_size, self.batch_size)
        return data_batch

    def get_annoation(self, index):
        metadata_path = self.data_path / str(int(index)) / 'metadata.json'
        segmentation_path = self.data_path / str(int(index)) / 'Segmentation' / 'Image0003.jpg'
        valid_joint_path = self.data_path / str(int(index)) / 'valid_joints.json'

        segmentation_mask = cv2.cvtColor(cv2.imread(str(segmentation_path)), cv2.COLOR_BGR2GRAY)   
        human_indices = (segmentation_mask < 127)

        segmentation_mask[human_indices] = 1
        segmentation_mask[~human_indices] = 0
    
        if not os.path.exists(metadata_path):
            return {
                'rgb_frame_index': -1,
                'ego_to_global_space': None,
                'valid_seg': False,
                'ego_j3d': None, 
                'ego_j2d': None, 
                'segmentation_mask': segmentation_mask,
                'valid_joints': None
            }

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        ego_j3d = np.array(metadata['human_body']['camera']['joints_3d'])
        ego_j2d = np.array(metadata['human_body']['camera']['joints_2d'])

        smpl_to_ego_joint_indices = list(self.cfg.SMPL_TO_JOINTS16.values())
                
        ego_j3d = ego_j3d[smpl_to_ego_joint_indices]
        ego_j2d = ego_j2d[smpl_to_ego_joint_indices]

        if valid_joint_path.exists():
            with open(valid_joint_path, 'r') as f:
                valid_joints = json.load(f)
                valid_joints = list(valid_joints.values()) 
                valid_joints = np.array(valid_joints, dtype=np.float32)
        else:
            valid_joints = np.ones(16, dtype=np.float32)
    
        return {
            'rgb_frame_index': index,
            'valid_seg': True,
            'ego_j3d': ego_j3d, 
            'ego_j2d': ego_j2d, 
            'segmentation_mask': segmentation_mask,
            'ego_to_global_space': None,
            'valid_joints': valid_joints,
        }
