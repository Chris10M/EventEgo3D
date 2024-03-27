import sys; sys.path.append('../')
import cv2
import random
import json
import logging
import pickle
import numpy as np

from dataset.dataset_utils import h5py_File
from dataset.transforms import rotate_points

from torch.utils.data import Dataset
from pathlib import Path
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


class RealEventStream(Dataset):
    def __init__(self, data_path, cfg, is_train):
        super().__init__()

        self.data_path = Path(data_path)

        with open(self.data_path / 'event_meta.json', 'r') as f:
            meta = json.load(f)
            
        self.height = meta['height']
        self.width = meta['width']
        
        self.stream_path = self.data_path / 'events.h5'
        self.fin = None 

        self.local_pose_gt_path = self.data_path / 'synced_local_pose_gt.pickle'
        
        marker_to_fisheye_matrix = pickle.load(open(self.data_path / 'marker_to_fisheye_matrix.pickle', 'rb'))['marker_to_fisheye_matrix']
        self.ego_to_board_space = np.linalg.inv(marker_to_fisheye_matrix)
        
        self.global_pose_gt_path = self.data_path / 'synced_pose_gt.pickle'

        if not self.local_pose_gt_path.exists():
            self.pose_list = None
            
        else:  
            with open(self.local_pose_gt_path, 'rb') as f:
                self.pose_list = pickle.load(f)

        if not self.global_pose_gt_path.exists():
            self.frame_start_index = 0
        else:
            with open(self.global_pose_gt_path, 'rb') as f:
                self.frame_start_index = pickle.load(f)['frame_start_index']    


        self.batch_size = cfg.DATASET.EVENT_BATCH_SIZE
        self.max_frame_time = cfg.DATASET.REAL.MAX_FRAME_TIME_IN_MS
        self.is_train = is_train

    def init_stream(self):
        self.fin = h5py_File(self.stream_path, 'r')['event']
        
    def __len__(self):
        if self.pose_list is None:
            return 0
        
        with h5py_File(self.stream_path, 'r') as f:
            return f['event'].shape[0] // self.batch_size
        
    def get_event_batch(self, idx, num_events):
        if self.is_train:
            max_frame_time = random.randint(15, self.max_frame_time)
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
        index = int(index)
        try:
            anno = self.pose_list[index] # TODO: Fix this -1 one frame offset
        except IndexError:
            return {
            'rgb_frame_index': -1,
            'ego_to_global_space': None,
            'valid_seg': False,
            'ego_j3d': None, 
            'ego_j2d': None, 
            'valid_joints': None,
            'segmentation_mask': np.zeros((self.height, self.width), dtype=np.uint8)
        }
        
        ego_j3d = anno['ego_j3d']
        ego_j2d = anno['ego_j2d']
        global_to_board_space = anno.get('global_to_board_space', None)

        segmentation_path = self.data_path / 'Blender_Segmentation' /str(int(index)) / 'Segmentation' / 'Image0003.jpg'
        
        valid_joint_path = self.data_path / 'valid_joints' /f'{int(index)}.json'
        
        if valid_joint_path.exists():
            with open(valid_joint_path, 'r') as f:
                valid_joints = json.load(f)
                valid_joints = list(valid_joints.values()) 
                valid_joints = np.array(valid_joints, dtype=np.float32)
        else:
            valid_joints = np.zeros(16, dtype=np.float32)
        
        if segmentation_path.exists():
            segmentation_mask = cv2.imread(str(segmentation_path))
            valid_seg = True        
        else:
            segmentation_mask = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            valid_seg = False

        segmentation_mask = cv2.cvtColor(segmentation_mask, cv2.COLOR_BGR2GRAY)
        
        segmentation_mask[segmentation_mask > 127] = 1
        
        if not np.any(segmentation_mask):
            valid_seg = False

        if ego_j3d is not None:
            ego_j3d = rotate_points(ego_j3d, 'x', 180.0)
        
        if global_to_board_space is not None:
            board_to_global_space = np.linalg.inv(global_to_board_space)
            ego_to_global_space = board_to_global_space @ self.ego_to_board_space
        else:
            ego_to_global_space = None
        
        return {
            'rgb_frame_index': self.frame_start_index + index,
            'ego_to_global_space': ego_to_global_space,
            'valid_seg': valid_seg,
            'ego_j3d': ego_j3d, 
            'ego_j2d': ego_j2d, 
            'valid_joints': valid_joints,
            'segmentation_mask': segmentation_mask
        }
