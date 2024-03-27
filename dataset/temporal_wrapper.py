import numpy as np
import random

from torch.utils.data import Dataset
from dataset import transforms
from settings import config


class TemoralWrapper(Dataset):
    def __init__(self, dataset, timesteps, augment) -> None:
        super().__init__()

        self.dataset = dataset
        self.timesteps = timesteps
        self.augment = augment  

    def get_random_transform(self):
        target_width, target_height = config.MODEL.IMAGE_SIZE
        
        center_shift_x = target_width / 2 + target_width * 0.1 * random.uniform(-1, 1)
        center_shift_y = target_height / 2 + target_height * 0.1 * random.uniform(-1, 1)
        center = np.array([center_shift_x, center_shift_y])
        scale = 1 + random.uniform(-1, 1) * config.DATASET.SCALE_FACTOR
        rot = random.uniform(-1, 1) * config.DATASET.ROT_FACTOR
        
        output_size = np.array((target_width, target_height))

        A = transforms.get_affine_transform(center, scale, rot, output_size)

        return A
    
    def get_random_flip_lr(self):
        if random.random() < 0.5:
            flip = True
        else:
            flip = False
        return flip
    
    def get_random_flip_axis(self):
        if random.random() < 0.5:
            flip = True
        else:
            flip = False
        return flip
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        start_index = max(0, idx - self.timesteps)
        end_index = idx
        
        end_index += self.timesteps - (end_index - start_index)

        kwargs = {'start_index': idx}
        if self.augment:
            kwargs['augment'] = True	
        
            # kwargs['A'] = self.get_random_transform()
            # kwargs['flip_lr'] = self.get_random_flip_lr()

            kwargs['flip_axis'] = self.get_random_flip_axis()    
            
        data = []
        for i in range(start_index, end_index): 
            kwargs['offset_index'] = i - start_index
            data.append(self.dataset[i, kwargs])
            
        return data

    def visualize(self, *args, **kwargs):
        return self.dataset.visualize(*args, **kwargs)

    def evaluate_joints(self, *args, **kwargs):
        return self.dataset.evaluate_joints(*args, **kwargs)
    
    def evaluate_dataset(self, *args, **kwargs):
        return self.dataset.evaluate_dataset(*args, **kwargs)
